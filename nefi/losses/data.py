"""Data-fidelity terms. All reduce by *mean* over observed entries so weights are
resolution-free.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from ..registry import register
from .base import Context, Loss


def _reduce(x: torch.Tensor, axes: Sequence[int] | None) -> torch.Tensor:
    return x if not axes else x.sum(dim=tuple(axes))


def _normalize(x: torch.Tensor, how: str, eps: float) -> torch.Tensor:
    if how == "max":
        return x / x.max().clamp_min(eps)
    if how == "mean":
        return x / x.mean().clamp_min(eps)
    if how == "sum":
        return x / x.sum().clamp_min(eps)
    if how in ("none", None):
        return x
    raise ValueError(f"unknown normalization {how!r}")


class DataLoss(Loss):
    is_data = True

    def __init__(self, name: str | None = None, noise_aware: bool = False) -> None:
        super().__init__(name)
        self.noise_aware = noise_aware

    def residual_weight(self, ctx: Context) -> torch.Tensor | float:
        if self.noise_aware and ctx.obs.noise_std is not None:
            ns = torch.as_tensor(ctx.obs.noise_std, device=ctx.pred.device, dtype=ctx.pred.dtype)
            return 1.0 / (ns**2).clamp_min(1e-30)
        return 1.0


@register("loss", "mse")
class MSE(DataLoss):
    """``mean((pred - obs)²)`` over observed entries (optionally divided by noise variance)."""

    def forward(self, ctx: Context) -> torch.Tensor:
        r = (ctx.pred - ctx.obs.data) ** 2 * self.residual_weight(ctx)
        return ctx.obs.masked_mean(r)


L2 = MSE
register("loss", "l2")(MSE)


@register("loss", "rmse")
class RMSE(DataLoss):
    def forward(self, ctx: Context) -> torch.Tensor:
        return torch.sqrt(ctx.obs.masked_mean((ctx.pred - ctx.obs.data) ** 2) + 1e-30)


@register("loss", "relative_mse")
class RelativeMSE(DataLoss):
    """``‖pred - obs‖² / ‖obs‖²`` — scale-free fidelity."""

    def forward(self, ctx: Context) -> torch.Tensor:
        num = ctx.obs.masked_mean((ctx.pred - ctx.obs.data) ** 2)
        den = ctx.obs.masked_mean(ctx.obs.data**2).clamp_min(1e-30)
        return num / den


@register("loss", "huber")
class Huber(DataLoss):
    """Robust fidelity (quadratic below ``delta``, linear above) for outlier-contaminated data."""

    def __init__(self, delta: float = 1.0, name: str | None = None, noise_aware: bool = False):
        super().__init__(name, noise_aware)
        self.delta = float(delta)

    def forward(self, ctx: Context) -> torch.Tensor:
        r = F.huber_loss(ctx.pred, ctx.obs.data, reduction="none", delta=self.delta)
        return ctx.obs.masked_mean(r * self.residual_weight(ctx))


@register("loss", "poisson_nll")
class PoissonNLL(DataLoss):
    """Poisson negative log-likelihood for count data (``pred`` = expected counts > 0)."""

    def __init__(self, eps: float = 1e-8, name: str | None = None) -> None:
        super().__init__(name)
        self.eps = eps

    def forward(self, ctx: Context) -> torch.Tensor:
        lam = ctx.pred.clamp_min(self.eps)
        r = lam - ctx.obs.data * torch.log(lam)
        return ctx.obs.masked_mean(r)


@register("loss", "log_mse")
class LogMSE(DataLoss):
    """Pixelwise log-MSE on (optionally reduced and) normalized maps — NeTMY Eq. (19).

    ``D = mean( [log10(N̂(pred) + eps) − log10(N̂(obs) + eps)]² )`` where ``N`` sums
    ``pred``/``obs`` over
    ``reduce_axes`` (e.g. the frequency axis) and ``N̂`` normalizes by ``normalize`` (``"max"`` is
    the NV
    convention; note that max-normalization induces the (P3) peak-coupling gradient).
    """

    def __init__(
        self,
        normalize: str = "max",
        reduce_axes: Sequence[int] | None = None,
        eps: float = 1e-10,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.normalize, self.reduce_axes, self.eps = normalize, reduce_axes, eps

    def maps(self, ctx: Context) -> tuple[torch.Tensor, torch.Tensor]:
        p = _normalize(_reduce(ctx.pred, self.reduce_axes), self.normalize, self.eps)
        o = _normalize(_reduce(ctx.obs.data, self.reduce_axes), self.normalize, self.eps)
        return p, o

    def forward(self, ctx: Context) -> torch.Tensor:
        p, o = self.maps(ctx)
        r = (torch.log10(p.clamp_min(0) + self.eps) - torch.log10(o.clamp_min(0) + self.eps)) ** 2
        return r.mean()


@register("loss", "normalized_mse")
class NormalizedMSE(DataLoss):
    """MSE between reduced maps normalized by ``mean`` (NeTMY R_nm) or ``max``/``sum``.

    Mean normalization avoids the nonlocal peak coupling of max-normalization (NeTMY App. C.3) and
    is the fine-stage fidelity in NeTMY (App. D.4).
    """

    def __init__(
        self,
        normalize: str = "mean",
        reduce_axes: Sequence[int] | None = None,
        eps: float = 1e-12,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.normalize, self.reduce_axes, self.eps = normalize, reduce_axes, eps

    def forward(self, ctx: Context) -> torch.Tensor:
        p = _normalize(_reduce(ctx.pred, self.reduce_axes), self.normalize, self.eps)
        o = _normalize(_reduce(ctx.obs.data, self.reduce_axes), self.normalize, self.eps)
        return ((p - o) ** 2).mean()
