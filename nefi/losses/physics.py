"""Physics-knowledge losses: soft PDE residuals, conservation, symmetry, support, monotonicity.

These encode *what you know about the unknown* as penalties. Prefer hard encodings when available
(heads, :class:`~nefi.fields.symmetric.SymmetricField`, a differentiable solver as the forward
operator) — penalties need weights and are only approximately satisfied — but penalties work with
every representation, including raster grids, and can be combined freely.

Example::

    from nefi.losses import LossSet, MSE
    from nefi.losses.physics import Conservation, Monotone, SymmetryLoss

    losses = LossSet({"data": MSE(), "mass": Conservation("x", total=1.0),
                      "sym": SymmetryLoss("x", "mirror_x"), "mono": Monotone("x", axis=1)},
                     weights={"mass": 10.0, "sym": 1.0, "mono": 1.0})
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from ..errors import ConfigError
from ..registry import register
from ..utils.tensor import resample
from .base import Context, Loss
from .reg import FieldLoss, RangePenalty, forward_differences

log = logging.getLogger("nefi")

#: Soft box constraint. ``RangeStat`` is an alias of the existing
#: :class:`~nefi.losses.reg.RangePenalty` (``mean relu(lo − x)² + relu(x − hi)²``).
RangeStat = RangePenalty
register("loss", "range_stat")(RangePenalty)


def gradient(
    x: torch.Tensor, spacing: Sequence[float], axis: int, scheme: str = "central"
) -> torch.Tensor:
    """Finite-difference ``∂x/∂x_axis`` on a grid (same shape; one-sided at the boundaries).

    Example::

        u = torch.linspace(0, 1, 11) ** 2
        du_dx = gradient(u, spacing=(0.1,), axis=0)        # ≈ 2x
    """
    h = float(spacing[axis])
    if scheme == "forward":
        return forward_differences(x, [1.0] * x.ndim)[axis] / h
    return torch.gradient(x, spacing=h, dim=axis)[0]


def laplacian(
    x: torch.Tensor, spacing: Sequence[float], periodic_axes: Sequence[int] = ()
) -> torch.Tensor:
    """Five-point (2-D) / seven-point (3-D) Laplacian with replicate (Neumann) or periodic edges.

    Example::

        u, f = torch.rand(16, 16), torch.ones(16, 16)
        r = laplacian(u, (1 / 16, 1 / 16)) - f          # Poisson residual Δu = f
    """
    out = torch.zeros_like(x)
    for ax in range(x.ndim):
        h = float(spacing[ax])
        if ax in periodic_axes or (ax - x.ndim) in periodic_axes:
            out = out + (torch.roll(x, 1, ax) - 2 * x + torch.roll(x, -1, ax)) / h**2
        else:
            n = x.shape[ax]
            xp = torch.cat([x.narrow(ax, 0, 1), x, x.narrow(ax, n - 1, 1)], dim=ax)
            out = out + (xp.narrow(ax, 0, n) - 2 * x + xp.narrow(ax, 2, n)) / h**2
    return out


@register("loss", "pde_residual")
class PDEResidual(Loss):
    """Soft PDE residual penalty ``mean(R(fields)²)`` — for problems **without** a hard solver.

    .. warning::
       A soft residual is a weak coupling. When a coefficient field (e.g. a diffusivity) is
       constrained *only* through a residual while the data term reaches a *different* network
       (a state surrogate), the data gradient never reaches the coefficient —
       ``∇_θ L_data = 0`` — the soft-constraint **decoupling pathology** analysed in NeFTY §3.3 /
       App. C: the residual shrinks while the coefficient stays near a trivial constant. Whenever
       you can, put the physics in the forward operator instead (``FunctionOperator``,
       :class:`~nefi.operators.timestepping.TimeStepper`, ``nefi.operators.pde``) so data
       gradients flow through the discretized PDE. This loss logs a warning when it detects that
       the fields it constrains are not consumed by the forward operator.

    Args:
        residual_fn: ``residual_fn(fields, ctx) -> Tensor`` returning the pointwise residual on
            the current grid (use ``ctx.domain.spacing(shape)`` and :func:`gradient` /
            :func:`laplacian` for derivatives).
        fields: names of the fields the residual constrains (default: all fields; used by the
            decoupling check).
        name: loss name.
        normalize: divide by ``mean(scale²)`` where ``scale = normalize(fields, ctx)`` (optional
            callable) to make the weight dimensionless.

    Example::

        # steady heat source recovery with u and f both unknown (soft coupling Δu + f = 0)
        res = PDEResidual(lambda f, ctx: laplacian(f["u"], ctx.domain.spacing(f["u"].shape))
                          + f["f"], fields=("u", "f"))
    """

    def __init__(
        self,
        residual_fn: Callable[[dict[str, torch.Tensor], Context], torch.Tensor],
        fields: Sequence[str] | None = None,
        name: str | None = None,
        normalize: Callable[[dict[str, torch.Tensor], Context], torch.Tensor] | None = None,
    ) -> None:
        super().__init__(name or "pde_residual")
        self.residual_fn = residual_fn
        self.fields = tuple(fields) if fields is not None else None
        self.normalize = normalize
        self._checked = False

    def _check_coupling(self, ctx: Context) -> None:
        self._checked = True
        names = self.fields or tuple(ctx.fields)
        consumed = set(ctx.operator.required_fields()) if ctx.operator is not None else set()
        decoupled = [n for n in names if n not in consumed]
        if decoupled:
            log.warning(
                "PDEResidual: fields %s are constrained only by the soft residual (the forward "
                "operator does not consume them). This is the soft-constraint decoupling "
                "pathology of NeFTY §3.3: data gradients cannot reach these fields directly. "
                "Prefer a hard differentiable solver as the operator (FunctionOperator / "
                "TimeStepper / nefi.operators.pde).",
                decoupled,
            )

    def forward(self, ctx: Context) -> torch.Tensor:
        if not self._checked:
            self._check_coupling(ctx)
        r = self.residual_fn(ctx.fields, ctx)
        val = (r**2).mean()
        if self.normalize is not None:
            s = self.normalize(ctx.fields, ctx)
            val = val / (torch.as_tensor(s) ** 2).mean().clamp_min(1e-30)
        return val


@register("loss", "conservation")
class Conservation(FieldLoss):
    """Penalize deviation of the field's total from a known value (mass / charge / energy).

    ``kind="sum"``: total = ``∫ x dV = Σ x · cell_volume`` (physical, resolution independent);
    ``kind="mean"``: total = ``mean(x)``. Loss = ``((total − target) / scale)²`` with
    ``scale = |target|`` when ``relative`` (default) and ``target ≠ 0``, else 1.

    Example::

        loss = Conservation("rho", total=1.0)      # ∫ρ = 1
    """

    def __init__(
        self,
        field: str | None = None,
        total: float = 1.0,
        kind: str = "sum",
        relative: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(field, name)
        if kind not in ("sum", "mean"):
            raise ConfigError(f"Conservation kind must be 'sum' or 'mean', got {kind!r}")
        self.total, self.kind, self.relative = float(total), kind, relative

    def current(self, ctx: Context) -> torch.Tensor:
        x = self.value(ctx)
        if self.kind == "mean":
            return x.mean()
        vol = 1.0
        for h in ctx.domain.spacing(x.shape):
            vol *= float(h)
        return x.sum() * vol

    def forward(self, ctx: Context) -> torch.Tensor:
        scale = abs(self.total) if (self.relative and self.total != 0.0) else 1.0
        return ((self.current(ctx) - self.total) / scale) ** 2


def _radial_mean(x: torch.Tensor, n_bins: int | None = None) -> torch.Tensor:
    """Per-pixel mean of ``x`` over its radial shell (about the grid center, normalized coords).

    With ``n_bins=None`` pixels are grouped by their *exact* radius (equal ``r²`` on the
    cell-centered grid), so any exactly radial field is a fixed point; an integer ``n_bins`` uses
    coarser annuli (approximate symmetry).
    """
    axes = [
        -1.0 + (2.0 * torch.arange(s, device=x.device, dtype=torch.float64) + 1.0) / s
        for s in x.shape
    ]
    r2 = sum(g**2 for g in torch.meshgrid(*axes, indexing="ij")).reshape(-1)
    if n_bins is None:
        _, idx = torch.unique(torch.round(r2 * 1e9), return_inverse=True)
        nb = int(idx.max()) + 1
    else:
        r = torch.sqrt(r2)
        nb = int(n_bins)
        idx = torch.clamp((r / r.max() * (nb - 1)).round().long(), 0, nb - 1)
    flat = x.reshape(-1)
    sums = torch.zeros(nb, device=x.device, dtype=x.dtype).index_add(0, idx, flat)
    cnt = torch.zeros(nb, device=x.device, dtype=x.dtype).index_add(0, idx, torch.ones_like(flat))
    return (sums / cnt.clamp_min(1.0))[idx].reshape(x.shape)


@register("loss", "symmetry")
class SymmetryLoss(FieldLoss):
    """Soft symmetry: ``mean (x − S x)²`` for a mirror (flip about the grid center) or radial
    symmetry (``S x`` = radial shell average). Zero iff the field has the symmetry.

    Works for every representation (including raster grids); for an exact constraint on coordinate
    fields use :class:`~nefi.fields.symmetric.SymmetricField`.

    Args:
        field: field name.
        kind: ``"mirror_x" | "mirror_y" | "mirror_z" | "mirror_xy" | "mirror" | "radial"``.
        axes: axes for ``"mirror"`` / radial plane (default all axes for radial).
        n_bins: radial annuli for ``"radial"`` (default: group pixels of exactly equal radius).

    Example::

        loss = SymmetryLoss("x", "mirror_x")
    """

    def __init__(
        self,
        field: str | None = None,
        kind: str = "mirror_x",
        axes: Sequence[int] | None = None,
        n_bins: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(field, name)
        self.kind, self.axes, self.n_bins = kind, axes, n_bins

    def forward(self, ctx: Context) -> torch.Tensor:
        from ..fields.symmetric import symmetry_axes

        x = self.value(ctx)
        ax = symmetry_axes(self.kind, x.ndim, self.axes)
        if self.kind == "radial":
            if len(ax) != x.ndim:
                raise ConfigError("SymmetryLoss(kind='radial') currently needs all axes")
            return ((x - _radial_mean(x, self.n_bins)) ** 2).mean()
        return ((x - x.flip(ax)) ** 2).mean()


@register("loss", "known_support")
class KnownSupportLoss(FieldLoss):
    """Soft known support: ``mean over the exterior of (x − fill)²`` (mask 1 = inside).

    The mask is given at the native resolution and resampled to the current grid.

    Example::

        mask = torch.zeros(32, 32); mask[8:24, 8:24] = 1
        loss = KnownSupportLoss("x", mask)        # push x → 0 outside the mask
    """

    def __init__(
        self,
        field: str | None = None,
        mask: torch.Tensor | None = None,
        fill: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(field, name)
        if mask is None:
            raise ConfigError("KnownSupportLoss needs mask=... (1 inside the support)")
        self.register_buffer("mask", torch.as_tensor(mask).float().clone(), persistent=False)
        self.fill = float(fill)

    def forward(self, ctx: Context) -> torch.Tensor:
        x = self.value(ctx)
        m = self.mask.to(x)
        if tuple(m.shape) != tuple(x.shape):
            m = (resample(m, x.shape) > 0.5).to(x.dtype)
        out = 1.0 - m
        return ((x - self.fill) ** 2 * out).sum() / out.sum().clamp_min(1.0)


@register("loss", "monotone")
class Monotone(FieldLoss):
    """Penalize violations of monotonicity along an axis: ``mean relu(∓∂x/∂x_axis)²``.

    Uses physical spacing, so the weight is resolution independent. Zero iff the field is
    non-decreasing (``direction="increasing"``) / non-increasing along ``axis``.

    Example::

        loss = Monotone("temperature", axis=-1, direction="decreasing")
    """

    def __init__(
        self,
        field: str | None = None,
        axis: int = 0,
        direction: str = "increasing",
        name: str | None = None,
    ) -> None:
        super().__init__(field, name)
        if direction not in ("increasing", "decreasing"):
            raise ConfigError("Monotone direction must be 'increasing' or 'decreasing'")
        self.axis, self.direction = int(axis), direction

    def forward(self, ctx: Context) -> torch.Tensor:
        x = self.value(ctx)
        ax = self.axis % x.ndim
        h = float(ctx.domain.spacing(x.shape)[ax])
        n = x.shape[ax]
        if n < 2:
            return x.new_zeros(())
        dx = (x.narrow(ax, 1, n - 1) - x.narrow(ax, 0, n - 1)) / h
        sign = -1.0 if self.direction == "increasing" else 1.0
        return (F.relu(sign * dx) ** 2).mean()


@register("loss", "gradient_l2")
class GradientL2(FieldLoss):
    """First-order Tikhonov (``H¹`` seminorm) ``mean |∇x|²`` with physical spacing.

    Example::

        loss = GradientL2("x")
    """

    def __init__(
        self, field: str | None = None, periodic_axes: Sequence[int] = (), name: str | None = None
    ) -> None:
        super().__init__(field, name)
        self.periodic_axes = tuple(periodic_axes)

    def forward(self, ctx: Context) -> torch.Tensor:
        x = self.value(ctx)
        diffs = forward_differences(x, ctx.domain.spacing(x.shape), self.periodic_axes)
        return sum((d**2).mean() for d in diffs)


__all__ = [
    "Conservation",
    "GradientL2",
    "KnownSupportLoss",
    "Monotone",
    "PDEResidual",
    "RangeStat",
    "SymmetryLoss",
    "gradient",
    "laplacian",
]
