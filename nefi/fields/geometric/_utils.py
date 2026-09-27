"""Internal helpers shared by the geometric representations.

* grid detection / cell-centered grid interpolation (``grid_axes``, ``axis_spacing``,
  ``interp_grid``) — every field in this package works on arbitrary point sets and takes a fast
  path on tensor-product grids (what the solver always passes);
* smooth and *cell-averaged* Heaviside functions (``smooth_step``) with sharpness schedules
  (``anneal_value``) — the continuation device shared by layered, shape and level-set fields;
* progress maps (``progress_fn`` / :class:`ProgressWindow`) — per-component annealing curricula;
* the stable softplus inverse (``inv_softplus``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from ...errors import ConfigError, ShapeError
from ..heads import Head

ProgressFn = Callable[[float], float]

SCHEDULES = ("geometric", "linear", "cosine", "constant")


def inv_softplus(y: float | Sequence[float] | torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Inverse of ``softplus(β x) / β`` for positive ``y`` (numerically stable, float32)."""
    t = torch.as_tensor(y, dtype=torch.float32).clamp_min(1e-12) * float(beta)
    return (t + torch.log(-torch.expm1(-t))) / float(beta)


def anneal_value(progress: float, start: float, end: float, schedule: str = "geometric") -> float:
    """Interpolate a continuation parameter from ``start`` (progress 0) to ``end`` (progress 1).

    Used for interface sharpness ``ε(progress)``: a soft indicator early (wide basins, gradients
    reach far from the interface) and a sharp one late (piecewise-constant fields), the same
    coarse-to-fine logic as annealed Fourier features (NeTMY Eq. 27, NeFTY Eq. 21).

    Args:
        progress: training progress in ``[0, 1]`` (clipped).
        start, end: values at progress 0 and 1 (``> 0`` for ``"geometric"``).
        schedule: ``"geometric"`` (log-linear), ``"linear"``, ``"cosine"`` or ``"constant"``
            (always ``end``).
    """
    p = min(max(float(progress), 0.0), 1.0)
    a, b = float(start), float(end)
    if schedule == "geometric":
        if a <= 0 or b <= 0:
            raise ConfigError("a geometric schedule needs positive start/end values")
        return a * (b / a) ** p
    if schedule == "linear":
        return a + (b - a) * p
    if schedule == "cosine":
        return b + (a - b) * 0.5 * (1.0 + math.cos(math.pi * p))
    if schedule == "constant":
        return b
    raise ConfigError(f"unknown schedule {schedule!r}; choose from {SCHEDULES}")


def softplus_diff(b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """``softplus(b) − softplus(a)`` without cancellation when both arguments are large."""
    pos = (a > 0) & (b > 0)
    stable = (b - a) + F.softplus(-b) - F.softplus(-a)
    direct = F.softplus(b) - F.softplus(a)
    return torch.where(pos, stable, direct)


def smooth_step(s: torch.Tensor, eps: float, h: float | None = None) -> torch.Tensor:
    """Smooth Heaviside ``σ(s/ε)``, or its average over a cell of width ``h`` centered at ``s``.

    The cell average ``(1/h)∫_{s−h/2}^{s+h/2} σ(t/ε) dt = (ε/h)[softplus((s+h/2)/ε) −
    softplus((s−h/2)/ε)]`` tends to the exact *partial-volume fraction* ``clip(s/h + 1/2, 0, 1)``
    as ``ε → 0``: the rendered field stays differentiable in the interface position with
    sub-cell resolution even for a perfectly sharp interface.

    Args:
        s: signed distance past the interface (positive on the "high" side).
        eps: softness (same units as ``s``).
        h: cell width for cell-averaged rendering; ``None``/``<= 0`` means point sampling.
    """
    eps = max(float(eps), 1e-12)
    if h is None or h <= 0:
        return torch.sigmoid(s / eps)
    h = float(h)
    return (eps / h) * softplus_diff((s + 0.5 * h) / eps, (s - 0.5 * h) / eps)


# ------------------------------------------------------------------------------------------
# grids
# ------------------------------------------------------------------------------------------
def grid_axes(coords: torch.Tensor) -> list[torch.Tensor] | None:
    """1-D axis vectors if ``coords`` ``(*shape, d)`` is a tensor-product (meshgrid) grid."""
    d = coords.shape[-1]
    if coords.ndim != d + 1 or d == 0:
        return None
    axes = []
    for i in range(d):
        idx: list[Any] = [0] * d
        idx[i] = slice(None)
        axes.append(coords[tuple(idx) + (i,)])
    mesh = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    if mesh.shape != coords.shape or not torch.equal(mesh, coords):
        return None
    return axes


def _uniform_step(v: torch.Tensor) -> float | None:
    if v.numel() == 1:
        return 2.0  # a single cell spans the whole normalized axis [-1, 1]
    dv = v[1:] - v[:-1]
    h = float(dv.mean())
    if h <= 0 or float((dv - h).abs().max()) > 1e-4 * abs(h):
        return None
    return h


def axis_spacing(coords: torch.Tensor, axis: int) -> float | None:
    """Uniform cell size along ``axis`` for grid-shaped ``coords`` (``None`` if not uniform).

    Cheap heuristic: only the coordinate line through the first cell is inspected, which is
    exact for the tensor-product grids produced by :meth:`nefi.domain.Domain.coords`.
    """
    d = coords.shape[-1]
    if coords.ndim != d + 1:
        return None
    axis = axis % d
    idx: list[Any] = [0] * d
    idx[axis] = slice(None)
    v = coords[tuple(idx) + (axis,)].detach()
    return _uniform_step(v)


def interp_grid(
    values: torch.Tensor,
    coords: torch.Tensor,
    mode: str = "linear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """Sample a cell-centered grid ``(C, *g)`` over ``[-1, 1]^d`` at ``coords (..., d)``.

    Multilinear interpolation with the cell-centered convention of
    :meth:`nefi.domain.Domain.coords` (same as ``grid_sample(align_corners=False)`` with border
    padding), written with gathers so that forward-mode autodiff (``torch.func.jvp``, used by the
    filtering-view diagnostics) works — ``grid_sample`` has no forward-AD rule. Any ``d >= 1``;
    differentiable in ``values`` and ``coords``.

    Args:
        values: table ``(C, g_0, …, g_{d−1})``.
        coords: query points ``(..., d)`` in ``[-1, 1]``.
        mode: ``"linear"`` (multilinear) or ``"nearest"``.
        padding_mode: ``"border"`` (clamp to the outermost cell centers; the only mode).

    Returns:
        ``(..., C)`` tensor.
    """
    if padding_mode != "border":
        raise ConfigError("interp_grid only supports padding_mode='border'")
    if mode not in ("linear", "bilinear", "trilinear", "nearest"):
        raise ConfigError(f"unknown interpolation mode {mode!r}")
    c = values.shape[0]
    g = tuple(values.shape[1:])
    d = len(g)
    if d < 1:
        raise ShapeError("interp_grid needs a table with at least one spatial dimension")
    if coords.shape[-1] != d:
        raise ShapeError(f"coords have {coords.shape[-1]} components but the table is {d}-D")
    batch = coords.shape[:-1]
    pts = coords.reshape(-1, d).to(values.dtype)
    flat = values.reshape(c, -1)
    strides = [1] * d
    for a in range(d - 2, -1, -1):
        strides[a] = strides[a + 1] * g[a + 1]
    lo_idx, hi_idx, w_hi = [], [], []
    for a in range(d):
        t = ((pts[:, a] + 1.0) * g[a] - 1.0) * 0.5  # continuous cell index
        t = t.clamp(0.0, float(g[a] - 1))
        if mode == "nearest":
            i = torch.round(t.detach()).long()
            lo_idx.append(i)
            hi_idx.append(i)
            w_hi.append(torch.zeros_like(t))
            continue
        i0 = torch.floor(t.detach()).long().clamp(0, max(g[a] - 2, 0))
        i1 = (i0 + 1).clamp(max=g[a] - 1)
        lo_idx.append(i0)
        hi_idx.append(i1)
        w_hi.append(t - i0.to(t.dtype))
    out = None
    for corner in range(2**d):
        idx = torch.zeros_like(lo_idx[0])
        w = None
        for a in range(d):
            up = (corner >> a) & 1
            idx = idx + (hi_idx[a] if up else lo_idx[a]) * strides[a]
            wa = w_hi[a] if up else 1.0 - w_hi[a]
            w = wa if w is None else w * wa
        term = flat.index_select(1, idx) * w
        out = term if out is None else out + term
    return out.transpose(0, 1).reshape(*batch, c)


# ------------------------------------------------------------------------------------------
# progress maps
# ------------------------------------------------------------------------------------------
class ProgressWindow:
    """Map the global progress ``p`` to ``lo + (hi − lo) · clip((p − start)/(end − start))^power``.

    Special cases: ``ProgressWindow()`` is the identity, ``lo = hi = 1`` is "fully annealed from
    the start" (e.g. a smooth background with all bands open), ``start = 0.5`` delays a
    component's annealing to the second half of each stage ("background first, anomaly second").

    Args:
        start, end: global-progress window over which the component anneals (``start < end``).
        lo, hi: component progress before / after the window.
        power: shape of the ramp inside the window.
    """

    def __init__(
        self,
        start: float = 0.0,
        end: float = 1.0,
        lo: float = 0.0,
        hi: float = 1.0,
        power: float = 1.0,
    ) -> None:
        if not end > start:
            raise ConfigError(f"ProgressWindow needs end > start, got {(start, end)}")
        self.start, self.end = float(start), float(end)
        self.lo, self.hi, self.power = float(lo), float(hi), float(power)

    def __call__(self, p: float) -> float:
        t = min(max((float(p) - self.start) / (self.end - self.start), 0.0), 1.0)
        return self.lo + (self.hi - self.lo) * t**self.power

    def __repr__(self) -> str:
        return (
            f"ProgressWindow(start={self.start}, end={self.end}, lo={self.lo}, hi={self.hi}, "
            f"power={self.power})"
        )


class _ClippedFn:
    def __init__(self, fn: ProgressFn) -> None:
        self.fn = fn

    def __call__(self, p: float) -> float:
        return min(max(float(self.fn(float(p))), 0.0), 1.0)

    def __repr__(self) -> str:
        return f"clipped({self.fn!r})"


def progress_fn(spec: Any) -> ProgressFn:
    """Build a progress map from a compact spec.

    Accepted specs: ``None`` / ``"linear"`` / ``"same"`` (identity); ``"full"`` (always 1, i.e.
    fully annealed); ``"frozen"`` / ``"zero"`` (always 0); a float ``c`` (constant); a tuple
    ``("delay", start)``, ``("window", start, end)``, ``("fast", end)`` or ``("power", γ)``; a
    dict of :class:`ProgressWindow` kwargs; a :class:`ProgressWindow`; or any callable
    ``p -> p'`` (clipped to ``[0, 1]``).
    """
    if spec is None or (isinstance(spec, str) and spec.lower() in ("linear", "same", "identity")):
        return ProgressWindow()
    if isinstance(spec, ProgressWindow):
        return spec
    if isinstance(spec, str):
        s = spec.lower()
        if s in ("full", "annealed", "open", "one"):
            return ProgressWindow(lo=1.0, hi=1.0)
        if s in ("frozen", "zero", "closed", "none"):
            return ProgressWindow(lo=0.0, hi=0.0)
        raise ConfigError(f"unknown progress spec {spec!r}")
    if isinstance(spec, bool):
        raise ConfigError("a boolean is not a progress spec")
    if isinstance(spec, int | float):
        c = min(max(float(spec), 0.0), 1.0)
        return ProgressWindow(lo=c, hi=c)
    if isinstance(spec, Mapping):
        return ProgressWindow(**spec)
    if isinstance(spec, tuple | list) and spec and isinstance(spec[0], str):
        kind, args = spec[0].lower(), [float(a) for a in spec[1:]]
        if kind == "delay" and len(args) == 1:
            return ProgressWindow(start=args[0])
        if kind == "window" and len(args) == 2:
            return ProgressWindow(start=args[0], end=args[1])
        if kind == "fast" and len(args) == 1:
            return ProgressWindow(end=args[0])
        if kind == "power" and len(args) == 1:
            return ProgressWindow(power=args[0])
        raise ConfigError(f"cannot parse progress spec {spec!r}")
    if callable(spec):
        return _ClippedFn(spec)
    raise ConfigError(f"cannot interpret {spec!r} as a progress map")


def level_set_head(
    lo: float = 0.0,
    hi: float = 1.0,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    schedule: str = "geometric",
    init_value: float | None = None,
):
    """A sharpening two-phase head ``lo + (hi − lo) σ(φ / ε(progress))``.

    Uses :class:`nefi.fields.levelset.LevelSetHead` when available and falls back to an
    equivalent local implementation otherwise (the level-set module is developed in parallel).
    """
    try:  # lazy: optional sibling module
        from ..levelset import LevelSetHead

        return LevelSetHead(lo, hi, eps_start, eps_end, schedule, init_value)
    except Exception:  # pragma: no cover - fallback when the sibling module is unavailable
        return SharpeningStepHead(lo, hi, eps_start, eps_end, schedule, init_value)


class SharpeningStepHead(Head):
    """Fallback two-phase head ``lo + (hi − lo) σ(φ / ε(progress))`` (same formula as
    ``nefi.fields.levelset.LevelSetHead``; used only when that module is unavailable)."""

    n_in = 1
    uses_progress = True

    def __init__(
        self,
        lo: float = 0.0,
        hi: float = 1.0,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        schedule: str = "geometric",
        init_value: float | None = None,
    ) -> None:
        super().__init__()
        if not hi > lo:
            raise ConfigError(f"step head needs hi > lo, got {(lo, hi)}")
        self.lo, self.hi = float(lo), float(hi)
        self.eps_start, self.eps_end, self.schedule = float(eps_start), float(eps_end), schedule
        self.init_value = init_value

    def eps(self, progress: float = 1.0) -> float:
        return anneal_value(progress, self.eps_start, self.eps_end, self.schedule)

    def transform(self, raw, others, progress: float = 1.0):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(raw[..., 0] / self.eps(progress))

    def init_bias(self):
        if self.init_value is None:
            return None
        u = (float(self.init_value) - self.lo) / (self.hi - self.lo)
        u = min(max(u, 1e-6), 1 - 1e-6)
        return [math.log(u / (1 - u)) * self.eps_start]


__all__ = [
    "ProgressWindow",
    "SharpeningStepHead",
    "anneal_value",
    "axis_spacing",
    "grid_axes",
    "interp_grid",
    "inv_softplus",
    "level_set_head",
    "progress_fn",
    "smooth_step",
    "softplus_diff",
]
