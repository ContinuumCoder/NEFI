"""Coordinate warps: reshape *where* a field spends its capacity.

A :class:`WarpedField` evaluates an inner coordinate field in transformed coordinates,
``x(u) = f_θ(φ(u))``. Its parameterization Jacobian is the inner one composed with the warp, so the
update kernel (NeTMY Lemma 2) becomes ``G_θ(u, u') = G_f(φ(u), φ(u'))``: a kernel of inner
bandwidth ``B`` acts with *local* bandwidth ``B · |φ'(u)|`` in the original coordinates.
Stretching a region (``|φ'| > 1``) grants it more resolution; compressing it makes the
representation smoother there. This is the representation-side counterpart of adaptive meshing
(de Boor's equidistribution principle).

Warps shipped here:

* :func:`polar` / :func:`cylindrical` — radial systems (``(r, θ)``; optionally the periodic
  embedding ``(r, cos θ, sin θ)`` so that Fourier features respect periodicity);
* :func:`depth_stretch` (power law) and :func:`log_depth` — concentrate capacity near the observed
  face. The thermal signal of a defect at depth ``z`` decays like ``exp(−z²/4αt)`` in time and
  the operator singular values decay algebraically (NeFTY App. B.4, Prop. 2): deep structure
  is barely constrained, so a representation that is smoother at depth is better matched — and it
  suppresses the spurious back-face artifacts of NeFTY's failure mode (App. G.7);
* :func:`sensitivity_warp` — per-axis monotone reparameterization by the *cumulative sensitivity*
  ``u' = 2 C(u) − 1``, ``C(u) = ∫_{-1}^{u} w / ∫ w``, ``w ∝ s^κ`` from a sensitivity map (e.g.
  ``nefi.diagnostics.sensitivity_map``): in the warped coordinates the operator sensitivity is
  (for ``κ = 1``) uniform, so a stationary inner prior becomes sensitivity-adaptive;
* :class:`DisplacementWarp` / :class:`DeformableField` — a template plus a learned, small,
  identity-initialized displacement ``u + D(u)`` (image registration-style shape inversion), with
  :class:`WarpRegularizer` for smooth, fold-free deformations.

The inner field must be *coordinate-based* (neural, Fourier-basis, shape, layered fields, ...);
a :class:`~nefi.fields.GridField` ignores coordinate values and is therefore not warpable.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from ...errors import ConfigError, ShapeError
from ...losses.base import Context, Loss
from ...losses.reg import forward_differences
from ...registry import register
from ..base import Field
from ..grid import GridField
from ..heads import Heads
from ..neural import NeuralField
from ._utils import progress_fn

log = logging.getLogger("nefi")


class CoordinateWarp(nn.Module):
    """Base class: ``φ: (..., in_dim) → (..., out_dim)`` (optionally progress-dependent)."""

    in_dim: int | None = None
    out_dim: int | None = None

    def forward(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, coords: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} has no inverse")

    def on_stage_start(self, stage: Any, domain: Any) -> None:
        """Hook forwarded by :class:`WarpedField`."""

    def reset_parameters(self) -> None:
        """Re-initialize learnable warps (no-op for fixed ones)."""


class IdentityWarp(CoordinateWarp):
    def forward(self, coords, progress=1.0):
        return coords

    def inverse(self, coords):
        return coords


class ComposeWarp(CoordinateWarp):
    """Apply warps in sequence (``φ_n ∘ … ∘ φ_1``)."""

    def __init__(self, warps: Sequence[CoordinateWarp]) -> None:
        super().__init__()
        if not warps:
            raise ConfigError("ComposeWarp needs at least one warp")
        self.warps = nn.ModuleList(warps)
        self.in_dim, self.out_dim = warps[0].in_dim, warps[-1].out_dim

    def forward(self, coords, progress=1.0):
        for w in self.warps:
            coords = w(coords, progress)
        return coords

    def inverse(self, coords):
        for w in reversed(self.warps):
            coords = w.inverse(coords)
        return coords

    def on_stage_start(self, stage, domain):
        for w in self.warps:
            w.on_stage_start(stage, domain)

    def reset_parameters(self):
        for w in self.warps:
            w.reset_parameters()


# ------------------------------------------------------------------------------------------
# radial systems
# ------------------------------------------------------------------------------------------
class PolarWarp(CoordinateWarp):
    """Polar coordinates in a plane: ``(x, y, …) → (r̃, θ/π, …)`` or ``(r̃, cos θ, sin θ, …)``.

    ``r̃ = 2 r / r_max − 1 ∈ [-1, 1]`` (``r_max`` defaults to the largest distance from the center
    to a corner of ``[-1, 1]²``), ``θ = atan2(y − c_y, x − c_x)``. With ``embed="angle"`` the output
    has the input dimension (invertible); ``embed="circle"`` replaces ``θ`` by the periodic pair
    ``(cos θ, sin θ)`` (one extra dimension; the inner field must take ``ndim + 1`` coordinates),
    which is what a neural field needs to be continuous across ``θ = ±π``. Other axes (e.g. the
    cylinder axis) pass through unchanged after the two polar components.

    Args:
        ndim: input dimension (2 for polar, 3 for cylindrical).
        center: pole in normalized coordinates.
        axes: the two plane axes.
        r_max: radius mapped to ``r̃ = 1``.
        embed: ``"angle"`` or ``"circle"``.
    """

    def __init__(
        self,
        ndim: int = 2,
        center: Sequence[float] = (0.0, 0.0),
        axes: Sequence[int] = (0, 1),
        r_max: float | None = None,
        embed: str = "angle",
    ) -> None:
        super().__init__()
        self.in_dim = int(ndim)
        self.axes = tuple(int(a) % self.in_dim for a in axes)
        if len(self.axes) != 2 or len(set(self.axes)) != 2:
            raise ConfigError(f"axes must be two distinct axes, got {axes}")
        if embed not in ("angle", "circle"):
            raise ConfigError(f"embed must be 'angle' or 'circle', got {embed!r}")
        self.embed = embed
        self.center = tuple(float(c) for c in center)
        if r_max is None:
            cx, cy = self.center
            r_max = max(math.hypot(sx - cx, sy - cy) for sx in (-1, 1) for sy in (-1, 1))
        self.r_max = float(r_max)
        self.rest = tuple(a for a in range(self.in_dim) if a not in self.axes)
        self.out_dim = self.in_dim + (1 if embed == "circle" else 0)

    def forward(self, coords, progress=1.0):
        if coords.shape[-1] != self.in_dim:
            raise ShapeError(f"PolarWarp expects {self.in_dim}-D coordinates")
        c = torch.tensor(self.center, device=coords.device, dtype=coords.dtype)
        d = coords[..., list(self.axes)] - c
        r = torch.sqrt((d**2).sum(-1) + 1e-12)
        rt = 2.0 * r / self.r_max - 1.0
        parts = [rt.unsqueeze(-1)]
        if self.embed == "angle":
            parts.append((torch.atan2(d[..., 1], d[..., 0]) / math.pi).unsqueeze(-1))
        else:
            parts += [(d[..., 0] / r).unsqueeze(-1), (d[..., 1] / r).unsqueeze(-1)]
        if self.rest:
            parts.append(coords[..., list(self.rest)])
        return torch.cat(parts, dim=-1)

    def inverse(self, coords):
        r = (coords[..., 0] + 1.0) * 0.5 * self.r_max
        if self.embed == "angle":
            th = coords[..., 1] * math.pi
            rest = coords[..., 2:]
        else:
            th = torch.atan2(coords[..., 2], coords[..., 1])
            rest = coords[..., 3:]
        out = torch.empty(*coords.shape[:-1], self.in_dim, device=coords.device, dtype=coords.dtype)
        out[..., self.axes[0]] = self.center[0] + r * torch.cos(th)
        out[..., self.axes[1]] = self.center[1] + r * torch.sin(th)
        for j, a in enumerate(self.rest):
            out[..., a] = rest[..., j]
        return out

    def extra_repr(self) -> str:
        return (
            f"center={self.center}, axes={self.axes}, r_max={self.r_max:.3g}, embed={self.embed!r}"
        )


def polar(
    center: Sequence[float] = (0.0, 0.0), r_max: float | None = None, embed: str = "angle"
) -> PolarWarp:
    """2-D polar warp ``(x, y) → (r̃, θ/π)`` (or ``(r̃, cos θ, sin θ)`` with ``embed="circle"``)."""
    return PolarWarp(2, center, (0, 1), r_max, embed)


def cylindrical(
    center: Sequence[float] = (0.0, 0.0),
    axis: int = 2,
    r_max: float | None = None,
    embed: str = "angle",
) -> PolarWarp:
    """3-D cylindrical warp around ``axis``: ``(x, y, z) → (r̃, θ/π, z)`` (plane = other axes)."""
    plane = tuple(a for a in range(3) if a != axis % 3)
    return PolarWarp(3, center, plane, r_max, embed)


# ------------------------------------------------------------------------------------------
# depth warps (observed-face concentration)
# ------------------------------------------------------------------------------------------
class _AxisWarp(CoordinateWarp):
    """Monotone 1-D map ``t ↦ t'`` on ``[0, 1]`` applied to one axis (``t = 0`` at the face)."""

    def __init__(self, ndim: int | None = None, axis: int = -1, face: str = "low") -> None:
        super().__init__()
        if face not in ("low", "high"):
            raise ConfigError(f"face must be 'low' (u = -1) or 'high' (u = +1), got {face!r}")
        self.axis, self.face = int(axis), face
        self.in_dim = self.out_dim = ndim

    def _map(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _inv(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _warp_axis(self, coords: torch.Tensor, fn) -> torch.Tensor:
        a = self.axis % coords.shape[-1]
        u = coords[..., a]
        t = (u + 1.0) * 0.5 if self.face == "low" else (1.0 - u) * 0.5
        t2 = fn(t)
        u2 = 2.0 * t2 - 1.0 if self.face == "low" else 1.0 - 2.0 * t2
        return torch.cat([coords[..., :a], u2.unsqueeze(-1), coords[..., a + 1 :]], dim=-1)

    def forward(self, coords, progress=1.0):
        return self._warp_axis(coords, self._map)

    def inverse(self, coords):
        return self._warp_axis(coords, self._inv)


class DepthStretchWarp(_AxisWarp):
    """Power-law depth stretch ``t' = ((t + δ)^γ − δ^γ) / ((1 + δ)^γ − δ^γ)``.

    ``γ < 1`` stretches the region near the observed face (``t = 0``) and compresses depth:
    the inner field resolves shallow structure finely and deep structure smoothly (NeFTY App. B.4:
    sensitivity decays with depth). ``γ = 1`` is the identity. ``δ`` keeps the slope at the face
    finite (``φ'(0) ∝ γ δ^{γ−1}``).
    """

    def __init__(
        self,
        gamma: float = 0.5,
        axis: int = -1,
        face: str = "low",
        offset: float = 0.02,
        ndim: int | None = None,
    ) -> None:
        super().__init__(ndim, axis, face)
        if gamma <= 0 or offset <= 0:
            raise ConfigError("gamma and offset must be positive")
        self.gamma, self.offset = float(gamma), float(offset)
        g, dl = self.gamma, self.offset
        self._lo, self._span = dl**g, (1.0 + dl) ** g - dl**g

    def _map(self, t):
        return ((t + self.offset).clamp_min(1e-12) ** self.gamma - self._lo) / self._span

    def _inv(self, t):
        return (t * self._span + self._lo).clamp_min(1e-12) ** (1.0 / self.gamma) - self.offset

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, axis={self.axis}, face={self.face!r}, offset={self.offset}"


class LogDepthWarp(_AxisWarp):
    """Logarithmic depth warp ``t' = log(1 + t/s) / log(1 + 1/s)``.

    Resolution decays like ``1/(s + t)`` with depth — matched to exponentially decaying
    sensitivity (NeTMY Lemma 1 ``e^{−k z0}``; pointwise heat damping, NeFTY App. B.4). Smaller
    ``scale`` concentrates capacity more strongly at the face.
    """

    def __init__(
        self, scale: float = 0.1, axis: int = -1, face: str = "low", ndim: int | None = None
    ) -> None:
        super().__init__(ndim, axis, face)
        if scale <= 0:
            raise ConfigError("scale must be positive")
        self.scale = float(scale)
        self._norm = math.log1p(1.0 / self.scale)

    def _map(self, t):
        return torch.log1p((t / self.scale).clamp_min(-1 + 1e-9)) / self._norm

    def _inv(self, t):
        return self.scale * torch.expm1(t * self._norm)

    def extra_repr(self) -> str:
        return f"scale={self.scale}, axis={self.axis}, face={self.face!r}"


def depth_stretch(
    gamma: float = 0.5, axis: int = -1, face: str = "low", offset: float = 0.02
) -> DepthStretchWarp:
    """Power-law depth stretch (``γ < 1`` concentrates capacity near the observed face)."""
    return DepthStretchWarp(gamma, axis, face, offset)


def log_depth(scale: float = 0.1, axis: int = -1, face: str = "low") -> LogDepthWarp:
    """Logarithmic depth warp (capacity ``∝ 1/(scale + depth)``)."""
    return LogDepthWarp(scale, axis, face)


# ------------------------------------------------------------------------------------------
# sensitivity warp
# ------------------------------------------------------------------------------------------
def _interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation (linear extrapolation) for increasing ``xp``."""
    xp, fp = xp.to(x), fp.to(x)
    idx = torch.searchsorted(xp, x.detach().contiguous().reshape(-1)).clamp(1, xp.numel() - 1)
    idx = idx.reshape(x.shape)
    x0, x1, f0, f1 = xp[idx - 1], xp[idx], fp[idx - 1], fp[idx]
    t = (x - x0) / (x1 - x0).clamp_min(1e-12)
    return f0 + t * (f1 - f0)


class SensitivityWarp(CoordinateWarp):
    """Per-axis equidistribution warp ``u'_a = 2 C_a(u_a) − 1`` from sensitivity profiles.

    For each warped axis ``a`` with a 1-D profile ``p_a`` sampled at cell centers of ``[-1, 1]``:
    ``w = max((1 − λ) + λ (p / mean p)^κ, floor)``, ``C_a`` = normalized cumulative integral of the
    piecewise-constant ``w`` (exact at cell edges), evaluated by piecewise-linear interpolation
    between edges (exact inverse by swapping the tables). High-sensitivity regions are stretched
    (more capacity), low-sensitivity regions compressed (smoother) — for ``λ = κ = 1`` the
    operator sensitivity per unit warped coordinate is uniform.

    Args:
        profiles: mapping ``axis -> 1-D profile`` (non-negative).
        ndim: coordinate dimension (default: inferred as ``max(axis) + 1``).
        strength: ``λ ∈ [0, 1]`` blend with the identity (``0`` = no warp).
        power: ``κ`` exponent on the normalized profile.
        floor: minimum density (keeps the warp strictly monotone / invertible).
    """

    def __init__(
        self,
        profiles: Mapping[int, torch.Tensor | Sequence[float]],
        ndim: int | None = None,
        strength: float = 1.0,
        power: float = 1.0,
        floor: float = 0.05,
    ) -> None:
        super().__init__()
        if not profiles:
            raise ConfigError("SensitivityWarp needs at least one axis profile")
        if not 0.0 <= strength <= 1.0:
            raise ConfigError("strength must be in [0, 1]")
        self.strength, self.power, self.floor = float(strength), float(power), float(floor)
        axes = sorted(int(a) for a in profiles)
        self.in_dim = self.out_dim = ndim if ndim is not None else max(axes) + 1
        self.warp_axes = tuple(a % self.in_dim for a in axes)
        for a in axes:
            p = torch.as_tensor(profiles[a], dtype=torch.float64).flatten().clamp_min(0.0)
            if p.numel() < 2:
                raise ConfigError("each profile needs at least 2 samples")
            if float(p.sum()) <= 0:
                raise ConfigError(f"profile for axis {a} is identically zero")
            n = p.numel()
            w = (1.0 - self.strength) + self.strength * (p / p.mean()) ** self.power
            w = w.clamp_min(self.floor)
            edges = torch.linspace(-1.0, 1.0, n + 1, dtype=torch.float64)
            cdf = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(w, 0)])
            cdf = 2.0 * cdf / cdf[-1] - 1.0
            self.register_buffer(f"_x{a % self.in_dim}", edges.float(), persistent=False)
            self.register_buffer(f"_y{a % self.in_dim}", cdf.float(), persistent=False)

    @classmethod
    def from_map(
        cls,
        sensitivity: torch.Tensor,
        axes: Sequence[int] | None = None,
        strength: float = 1.0,
        power: float = 1.0,
        floor: float = 0.05,
    ) -> SensitivityWarp:
        """Build from a field-shaped sensitivity map (marginal mean profile per warped axis)."""
        s = torch.as_tensor(sensitivity).detach().double().abs()
        axes = tuple(range(s.ndim)) if axes is None else tuple(int(a) % s.ndim for a in axes)
        profiles = {}
        for a in axes:
            other = tuple(i for i in range(s.ndim) if i != a)
            profiles[a] = s.mean(dim=other) if other else s
        return cls(profiles, ndim=s.ndim, strength=strength, power=power, floor=floor)

    def _table(self, a: int) -> tuple[torch.Tensor, torch.Tensor]:
        return getattr(self, f"_x{a}"), getattr(self, f"_y{a}")

    def _warp_columns(self, coords: torch.Tensor, inverse: bool) -> torch.Tensor:
        cols = list(coords.unbind(-1))
        for a in self.warp_axes:
            xp, yp = self._table(a)
            cols[a] = _interp1d(cols[a], yp, xp) if inverse else _interp1d(cols[a], xp, yp)
        return torch.stack(cols, dim=-1)

    def forward(self, coords, progress=1.0):
        if coords.shape[-1] != self.in_dim:
            raise ShapeError(f"SensitivityWarp expects {self.in_dim}-D coordinates")
        return self._warp_columns(coords, inverse=False)

    def inverse(self, coords):
        return self._warp_columns(coords, inverse=True)

    def density(self, axis: int, u: torch.Tensor) -> torch.Tensor:
        """Local stretch ``du'/du`` along ``axis`` at coordinates ``u`` (piecewise constant)."""
        xp, yp = self._table(axis % self.in_dim)
        slope = (yp[1:] - yp[:-1]) / (xp[1:] - xp[:-1])
        idx = torch.searchsorted(xp, u.contiguous()).clamp(1, xp.numel() - 1) - 1
        return slope[idx]

    def extra_repr(self) -> str:
        return (
            f"axes={self.warp_axes}, strength={self.strength}, power={self.power}, "
            f"floor={self.floor}"
        )


def sensitivity_warp(
    sensitivity: torch.Tensor | Mapping[int, torch.Tensor],
    axes: Sequence[int] | None = None,
    strength: float = 1.0,
    power: float = 1.0,
    floor: float = 0.05,
) -> SensitivityWarp:
    """Equidistribution warp from a sensitivity map (field-shaped tensor) or per-axis profiles.

    Example::

        from nefi.diagnostics import sensitivity_map
        s = sensitivity_map(problem)                       # (*shape) Jacobian column norms
        warp = sensitivity_warp(s, axes=(-1,))             # warp only the depth axis
        field = WarpedField(NeuralField(3, heads), warp)
    """
    if isinstance(sensitivity, Mapping):
        return SensitivityWarp(sensitivity, strength=strength, power=power, floor=floor)
    return SensitivityWarp.from_map(sensitivity, axes, strength, power, floor)


# ------------------------------------------------------------------------------------------
# learned displacement
# ------------------------------------------------------------------------------------------
class DisplacementWarp(CoordinateWarp):
    """Learned displacement ``φ(u) = u + D(u)``, ``D = m · tanh(net(u))``, identity at init.

    The displacement network is a small annealed :class:`~nefi.fields.NeuralField` whose output
    layer is initialized to ``init_scale ×`` Xavier (``0`` → exactly the identity warp at init,
    gradients first reach the output layer, as in zero-initialized residual branches).

    Args:
        ndim: coordinate dimension.
        max_displacement: bound ``m`` on each displacement component (normalized units).
        axes: displaced axes (default all).
        hidden, depth, n_octaves: network size (smooth, low-octave warps by default).
        init_scale: output-layer init scale (``0`` = identity).
        net: a custom field producing ``len(axes)`` raw channels (overrides the default net).
    """

    def __init__(
        self,
        ndim: int,
        max_displacement: float = 0.2,
        axes: Sequence[int] | None = None,
        hidden: int = 32,
        depth: int = 2,
        n_octaves: int = 3,
        init_scale: float = 0.0,
        net: Field | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = self.out_dim = int(ndim)
        self.axes = tuple(range(self.in_dim)) if axes is None else tuple(a % ndim for a in axes)
        self.max_displacement = float(max_displacement)
        if net is None:
            net = NeuralField(
                self.in_dim,
                Heads({f"d{a}": "identity" for a in self.axes}),
                hidden=hidden,
                depth=depth,
                skip_at=None,
                n_octaves=n_octaves,
                out_init_scale=init_scale,
            )
        elif net.heads.n_in != len(self.axes):
            raise ConfigError(f"displacement net must output {len(self.axes)} raw channels")
        self.net = net

    def displacement(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        """``D(u)`` of shape ``(*shape, len(axes))``."""
        m = self.max_displacement
        return m * torch.tanh(self.net.raw(coords, progress) / m)

    def forward(self, coords, progress=1.0):
        d = self.displacement(coords, progress)
        cols = list(coords.unbind(-1))
        for j, a in enumerate(self.axes):
            cols[a] = cols[a] + d[..., j]
        return torch.stack(cols, dim=-1)

    def reset_parameters(self) -> None:
        self.net.reset_parameters()

    def extra_repr(self) -> str:
        return f"axes={self.axes}, max_displacement={self.max_displacement}"


_NAMED = {
    "identity": lambda nd: IdentityWarp(),
    "polar": lambda nd: PolarWarp(nd),
    "cylindrical": lambda nd: cylindrical(),
    "depth_stretch": lambda nd: DepthStretchWarp(ndim=nd),
    "log_depth": lambda nd: LogDepthWarp(ndim=nd),
}


@register("field", "warped")
class WarpedField(Field):
    """``x(u) = f_θ(φ(u))``: an inner coordinate field evaluated in warped coordinates.

    The inner field's heads are used (``raw`` = inner raw at the warped coordinates), so the
    warped field has the same outputs as the inner one. Progress is passed to both the inner field
    and the warp (``warp_progress`` / ``inner_progress`` maps optional).

    Args:
        inner: a coordinate-based field whose ``ndim`` equals the warp's output dimension.
        warp: a :class:`CoordinateWarp`, a sequence of warps (composed left to right), or one of
            ``"identity"``, ``"polar"``, ``"cylindrical"``, ``"depth_stretch"``, ``"log_depth"``.
        inner_progress, warp_progress: optional progress maps (see ``progress_fn``).
    """

    def __init__(
        self,
        inner: Field,
        warp: CoordinateWarp | str | Sequence[CoordinateWarp],
        inner_progress: Any = None,
        warp_progress: Any = None,
    ) -> None:
        super().__init__(inner.heads)
        if isinstance(inner, GridField):
            log.warning(
                "WarpedField: a GridField ignores coordinate values; the warp will have no effect"
            )
        self.inner = inner
        nd = getattr(inner, "ndim", None)
        if isinstance(warp, str):
            key = warp.lower()
            if key not in _NAMED:
                raise ConfigError(f"unknown warp {warp!r}; named warps: {sorted(_NAMED)}")
            warp = _NAMED[key](nd or 2)
        elif isinstance(warp, Sequence):
            warp = ComposeWarp(list(warp))
        if not isinstance(warp, CoordinateWarp):
            raise ConfigError(f"warp must be a CoordinateWarp, got {type(warp).__name__}")
        if nd is not None and warp.out_dim is not None and warp.out_dim != nd:
            raise ConfigError(
                f"the warp outputs {warp.out_dim}-D coordinates but the inner field is {nd}-D "
                "(e.g. polar(embed='circle') needs an inner field with ndim + 1 inputs)"
            )
        self.warp = warp
        self._inner_progress = progress_fn(inner_progress)
        self._warp_progress = progress_fn(warp_progress)

    @property
    def ndim(self) -> int | None:
        return self.warp.in_dim

    def warped_coords(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        return self.warp(coords, self._warp_progress(progress))

    def raw(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        return self.inner.raw(self.warped_coords(coords, progress), self._inner_progress(progress))

    def on_stage_start(self, stage, domain) -> None:
        self.inner.on_stage_start(stage, domain)
        self.warp.on_stage_start(stage, domain)

    def reset_parameters(self) -> None:
        self.inner.reset_parameters()
        self.warp.reset_parameters()


@register("field", "deformable")
class DeformableField(WarpedField):
    """Template field + learned smooth displacement: ``x(u) = T(u + D(u))``.

    ``D`` is a :class:`DisplacementWarp` initialized to zero (the field *equals the template* at
    init). Freeze the template (``Stage(freeze=("inner.",))``) to register a known shape to the
    data, or train both. Pair with :class:`WarpRegularizer` for smooth, fold-free deformations.

    Args:
        template: the template field (any coordinate field; e.g. a shape field or a fitted
            neural field).
        warp_field: optional custom :class:`DisplacementWarp`.
        max_displacement, hidden, depth, n_octaves, init_scale, axes: displacement-network
            settings (see :class:`DisplacementWarp`).
        template_progress, warp_progress: progress maps (default: template fully annealed,
            ``"full"``; warp annealed with the global progress).
    """

    def __init__(
        self,
        template: Field,
        warp_field: DisplacementWarp | None = None,
        max_displacement: float = 0.2,
        hidden: int = 32,
        depth: int = 2,
        n_octaves: int = 3,
        init_scale: float = 0.0,
        axes: Sequence[int] | None = None,
        template_progress: Any = "full",
        warp_progress: Any = None,
    ) -> None:
        nd = getattr(template, "ndim", None)
        if warp_field is None:
            if nd is None:
                raise ConfigError("template has no `ndim`; pass warp_field explicitly")
            warp_field = DisplacementWarp(
                nd, max_displacement, axes, hidden, depth, n_octaves, init_scale
            )
        super().__init__(template, warp_field, template_progress, warp_progress)

    @property
    def template(self) -> Field:
        return self.inner

    def displacement(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        return self.warp.displacement(coords, self._warp_progress(progress))  # type: ignore[operator]


@register("loss", "warp_regularizer")
class WarpRegularizer(Loss):
    """Smoothness / magnitude / folding penalty on every :class:`DisplacementWarp` in the field.

    ``smoothness · mean ‖∇D‖²`` (Dirichlet energy, forward differences in normalized units)
    ``+ magnitude · mean ‖D‖²`` ``+ folding · mean relu(margin − det(I + ∇D))²`` (penalizes
    non-invertible / folding deformations; ``len(axes) == ndim`` only).
    """

    def __init__(
        self,
        smoothness: float = 1.0,
        magnitude: float = 0.0,
        folding: float = 0.0,
        margin: float = 0.1,
        module: DisplacementWarp | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name or "warp")
        self.smoothness, self.magnitude = float(smoothness), float(magnitude)
        self.folding, self.margin = float(folding), float(margin)
        self._module = [module] if module is not None else None

    def forward(self, ctx: Context) -> torch.Tensor:
        warps = self._module
        if warps is None:
            if ctx.field_module is None:
                raise ConfigError("WarpRegularizer needs ctx.field_module or an explicit module")
            warps = [m for m in ctx.field_module.modules() if isinstance(m, DisplacementWarp)]
        if not warps:
            raise ConfigError("WarpRegularizer: no DisplacementWarp in the field")
        dom = ctx.domain
        ref = ctx.pred
        coords = dom.coords(device=ref.device, dtype=ref.dtype)
        h = tuple(2.0 / n for n in dom.shape)
        total = ref.new_zeros(())
        for w in warps:
            d = w.displacement(coords, ctx.progress)  # (*shape, k)
            grads = [forward_differences(d[..., j], h) for j in range(d.shape[-1])]
            if self.smoothness:
                total = total + self.smoothness * sum((g**2).mean() for gj in grads for g in gj)
            if self.magnitude:
                total = total + self.magnitude * (d**2).sum(-1).mean()
            if self.folding and len(w.axes) == dom.ndim:
                jac = torch.eye(dom.ndim, device=ref.device, dtype=ref.dtype).expand(
                    *d.shape[:-1], dom.ndim, dom.ndim
                )
                jac = jac + torch.stack(
                    [torch.stack(grads[i], dim=-1) for i in range(dom.ndim)], dim=-2
                )
                det = torch.linalg.det(jac) if dom.ndim > 1 else jac[..., 0, 0]
                total = total + self.folding * (torch.relu(self.margin - det) ** 2).mean()
        return total


def warp_regularizer(
    smoothness: float = 1.0, magnitude: float = 0.0, folding: float = 0.0, margin: float = 0.1
) -> WarpRegularizer:
    """Factory for :class:`WarpRegularizer` (smooth, small, fold-free displacements)."""
    return WarpRegularizer(smoothness, magnitude, folding, margin)


__all__ = [
    "ComposeWarp",
    "CoordinateWarp",
    "DeformableField",
    "DepthStretchWarp",
    "DisplacementWarp",
    "IdentityWarp",
    "LogDepthWarp",
    "PolarWarp",
    "SensitivityWarp",
    "WarpRegularizer",
    "WarpedField",
    "cylindrical",
    "depth_stretch",
    "log_depth",
    "polar",
    "sensitivity_warp",
    "warp_regularizer",
]
