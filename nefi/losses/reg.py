"""Regularizers on fields. Gradients use physical spacing so weights transfer across resolutions."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from ..registry import register
from .base import Context, Loss


def forward_differences(
    x: torch.Tensor, spacing: Sequence[float] | None = None, periodic_axes: Sequence[int] = ()
) -> list[torch.Tensor]:
    """Per-axis forward differences of a ``(*shape)`` field, same shape as ``x``.

    Non-periodic axes get a zero difference on the last slice (NeFTY Eq. 22 convention); periodic
    axes wrap around. If ``spacing`` is given the differences are divided by it (gradient estimate).
    """
    out = []
    d = x.ndim
    for ax in range(d):
        if ax in periodic_axes or (ax - d) in periodic_axes:
            diff = torch.roll(x, shifts=-1, dims=ax) - x
        else:
            diff = torch.narrow(x, ax, 1, x.shape[ax] - 1) - torch.narrow(x, ax, 0, x.shape[ax] - 1)
            pad = [0, 0] * d
            pad[2 * (d - 1 - ax) + 1] = 1  # pad end of axis `ax`
            diff = F.pad(diff, pad)
        if spacing is not None:
            diff = diff / float(spacing[ax])
        out.append(diff)
    return out


class FieldLoss(Loss):
    def __init__(self, field: str | None = None, name: str | None = None) -> None:
        super().__init__(name)
        self.field_name = field

    def value(self, ctx: Context) -> torch.Tensor:
        return ctx.field(self.field_name)


@register("loss", "l1")
class L1(FieldLoss):
    """``mean |x|`` — sparsity prior for point-like sources (NeTMY Eq. 3)."""

    def forward(self, ctx: Context) -> torch.Tensor:
        return self.value(ctx).abs().mean()


@register("loss", "tikhonov")
class Tikhonov(FieldLoss):
    """``mean x²`` (ridge)."""

    def forward(self, ctx: Context) -> torch.Tensor:
        return (self.value(ctx) ** 2).mean()


@register("loss", "tv")
class TV(FieldLoss):
    """Total variation.

    ``isotropic=True``: ``mean sqrt(Σ_i (∂_i x)² + eps²)`` (NeFTY Eq. 22, piecewise-constant prior).
    ``isotropic=False``: ``mean Σ_i |∂_i x|`` (NeTMY Eq. 3, anisotropic; cheap but axis-aligned).
    ``use_spacing`` divides differences by the physical cell size (resolution-consistent weights).
    """

    def __init__(
        self,
        field: str | None = None,
        isotropic: bool = True,
        eps: float = 1e-6,
        periodic_axes: Sequence[int] = (),
        use_spacing: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(field, name)
        self.isotropic, self.eps = isotropic, float(eps)
        self.periodic_axes = tuple(periodic_axes)
        self.use_spacing = use_spacing

    def forward(self, ctx: Context) -> torch.Tensor:
        x = self.value(ctx)
        sp = ctx.domain.spacing(x.shape) if self.use_spacing else None
        diffs = forward_differences(x, sp, self.periodic_axes)
        if self.isotropic:
            g2 = sum(d**2 for d in diffs)
            return torch.sqrt(g2 + self.eps**2).mean()
        return sum(d.abs().mean() for d in diffs)


@register("loss", "laplacian")
class Laplacian(FieldLoss):
    """``mean (Δx)²`` — second-order smoothness; suppresses axis-aligned cross artifacts
    (NeTMY App. E.10) at the price of slightly blurrier peaks."""

    def __init__(
        self,
        field: str | None = None,
        periodic_axes: Sequence[int] = (),
        use_spacing: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(field, name)
        self.periodic_axes = tuple(periodic_axes)
        self.use_spacing = use_spacing

    def forward(self, ctx: Context) -> torch.Tensor:
        x = self.value(ctx)
        sp = ctx.domain.spacing(x.shape) if self.use_spacing else None
        lap = torch.zeros_like(x)
        for ax in range(x.ndim):
            h = float(sp[ax]) if sp is not None else 1.0
            if ax in self.periodic_axes:
                lap = lap + (torch.roll(x, 1, ax) - 2 * x + torch.roll(x, -1, ax)) / h**2
            else:
                n = x.shape[ax]
                first = torch.narrow(x, ax, 0, 1)
                last = torch.narrow(x, ax, n - 1, 1)
                xp = torch.cat([first, x, last], dim=ax)  # replicate padding
                lap = lap + (torch.narrow(xp, ax, 0, n) - 2 * x + torch.narrow(xp, ax, 2, n)) / h**2
        return (lap**2).mean()


@register("loss", "range")
class RangePenalty(FieldLoss):
    """Soft box constraint ``mean relu(lo - x)² + relu(x - hi)²`` (for grids without bounded
    heads).
    """

    def __init__(
        self,
        lo: float | None = 0.0,
        hi: float | None = None,
        field: str | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(field, name)
        self.lo, self.hi = lo, hi

    def forward(self, ctx: Context) -> torch.Tensor:
        x = self.value(ctx)
        p = torch.zeros((), device=x.device, dtype=x.dtype)
        if self.lo is not None:
            p = p + (F.relu(self.lo - x) ** 2).mean()
        if self.hi is not None:
            p = p + (F.relu(x - self.hi) ** 2).mean()
        return p


@register("loss", "prior_mse")
class PriorMSE(FieldLoss):
    """``mean (x - x_prior)²`` towards a reference field (warm start / external prior)."""

    def __init__(self, prior: torch.Tensor, field: str | None = None, name: str | None = None):
        super().__init__(field, name)
        self.register_buffer("prior", torch.as_tensor(prior).clone())

    def forward(self, ctx: Context) -> torch.Tensor:
        x = self.value(ctx)
        p = self.prior
        if tuple(p.shape) != tuple(x.shape):
            from ..utils.tensor import resample

            p = resample(p, x.shape)
        return ((x - p.to(x)) ** 2).mean()
