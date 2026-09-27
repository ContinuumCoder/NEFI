"""Explicit shape representations rendered with smooth, sharpening indicators.

When the unknown is "a few inclusions with unknown boundaries" (NDE defects, NeFTY §5 / App. E.1:
ellipsoidal, cylindrical or box-shaped voids; tumors; mineral bodies), a boundary
parameterization is the most compact geometric prior there is: a few dozen numbers per object
instead of a voxel field. :class:`StarShapeField` represents each inclusion by a center and a
Fourier-descriptor radius function

    r(θ) = r_0 · (1 + Σ_{m=1}^{M} a_m cos(mθ) + b_m sin(mθ)),

and renders the field ``x = v_out + (v_in − v_out) · χ(x)`` with a soft indicator
``χ = σ((r(θ) − ρ)/ε)`` whose softness ``ε(progress)`` shrinks during training (continuation,
like the level-set head and the annealed Fourier features, NeTMY Eq. 27). In 3-D the planar
star-shape is extruded over a learnable depth interval (a cylinder / prism with a smooth
top and bottom). :class:`PolygonField` does the same with polygons (box-like defects).

Filtering view (NeTMY Lemma 2): ``J_θ`` of a shape field is concentrated in an ``O(ε)`` band
around the boundaries (and the columns of the values are the indicators), so ``G_θ`` moves
*interfaces* rather than pixels; its rank is ``≤ #shapes × (2 + 1 + 2M + #values)`` — it cannot fit
noise away from the boundaries.

Out of scope: topology changes — birth/death of shapes (use a capacity pool with
:meth:`StarShapeField.grow` and :class:`~nefi.fields.adaptive.GrowCapacity`, which places new
shapes at the extremum of the field-space gradient, a topological-derivative heuristic), merging,
and holes (star-shapedness).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ...errors import ConfigError, ShapeError
from ...losses.base import Context, Loss
from ...losses.reg import forward_differences
from ...registry import register
from ..base import Field
from ..heads import Heads
from ._utils import anneal_value, inv_softplus

UNIONS = ("prob", "max", "sum")


def _aspect_from(domain: Any, axes: Sequence[int]) -> tuple[float, ...]:
    sizes = [float(domain.size[a]) for a in axes]
    m = max(sizes)
    return tuple(s / m for s in sizes)


def _union(chi: torch.Tensor, how: str) -> torch.Tensor:
    """Combine per-shape indicators ``(..., S)`` into one ``(...)``."""
    if chi.shape[-1] == 1:
        return chi[..., 0]
    if how == "prob":
        return 1.0 - torch.prod(1.0 - chi, dim=-1)
    if how == "max":
        return chi.amax(dim=-1)
    return chi.sum(dim=-1).clamp(max=1.0)


def _default_centers(n: int, ndim: int = 2, radius: float = 0.45) -> torch.Tensor:
    if n == 1:
        return torch.zeros(1, ndim)
    ang = torch.arange(n, dtype=torch.float32) * (2 * math.pi / n)
    c = torch.zeros(n, ndim)
    c[:, 0], c[:, 1] = radius * torch.cos(ang), radius * torch.sin(ang)
    return c


def hint_location(
    hint: Mapping | None, contrast: float, extent: float = 0.9
) -> torch.Tensor | None:
    """Best location for a new inclusion from a growth hint (topological-derivative heuristic).

    Adding an inclusion of contrast ``Δv`` at ``x`` changes the loss by ``≈ Δv · ∂L/∂x(x) · area``;
    the most favourable location minimizes ``Δv · ∂L/∂x``. ``hint`` carries ``"gradient"`` (the
    field-space data gradient) and ``"coords"`` (matching normalized coordinates) — see
    :class:`~nefi.fields.adaptive.GrowCapacity`. Locations are clipped to ``[-extent, extent]``.
    """
    if not hint or "gradient" not in hint or "coords" not in hint:
        return None
    g = torch.as_tensor(hint["gradient"]).detach()
    coords = torch.as_tensor(hint["coords"]).detach()
    score = (math.copysign(1.0, contrast) if contrast != 0 else 1.0) * g
    idx = int(torch.argmin(score.flatten()))
    loc = coords.reshape(-1, coords.shape[-1])[idx]
    return loc.clamp(-extent, extent)


class _ShapeBase(Field):
    """Shared machinery: values, sharpness schedule, active pool, aspect, rendering."""

    def __init__(
        self,
        n_shapes: int,
        heads: Heads | Mapping | None,
        ndim: int,
        inside: float | Sequence[float],
        outside: float | Sequence[float],
        learn_values: bool,
        per_shape_values: bool,
        eps_start: float,
        eps_end: float,
        schedule: str,
        union: str,
        aspect: Sequence[float] | None,
        domain: Any,
        plane_axes: Sequence[int],
        n_active: int | None,
    ) -> None:
        super().__init__(heads)
        self.ndim = int(ndim)
        self.S = int(n_shapes)
        if self.S < 1:
            raise ConfigError("need at least one shape")
        if union not in UNIONS:
            raise ConfigError(f"union must be one of {UNIONS}, got {union!r}")
        self.union = union
        self.plane_axes = tuple(int(a) % self.ndim for a in plane_axes)
        if len(self.plane_axes) != 2 or len(set(self.plane_axes)) != 2:
            raise ConfigError(f"plane_axes must name two distinct axes, got {plane_axes}")
        if domain is not None and aspect is None:
            aspect = _aspect_from(domain, self.plane_axes)
        self.aspect = tuple(float(a) for a in (aspect or (1.0, 1.0)))
        if len(self.aspect) != 2 or min(self.aspect) <= 0:
            raise ConfigError(f"aspect needs two positive factors, got {aspect}")
        self.eps_start, self.eps_end, self.schedule = float(eps_start), float(eps_end), schedule
        anneal_value(0.5, self.eps_start, self.eps_end, schedule)
        self.per_shape_values = bool(per_shape_values)
        c = self.heads.n_in
        self._in_init = _values(inside, self.S if per_shape_values else 1, c, "inside")
        self._out_init = _values(outside, 1, c, "outside")[0]
        self.inside = nn.Parameter(self._in_init.clone(), requires_grad=learn_values)
        self.outside = nn.Parameter(self._out_init.clone(), requires_grad=learn_values)
        self._n_active_init = self.S if n_active is None else int(n_active)
        if not 0 <= self._n_active_init <= self.S:
            raise ConfigError(f"n_active must be in [0, {self.S}], got {n_active}")
        self.register_buffer("active", torch.arange(self.S) < self._n_active_init, persistent=True)

    # --- common API ----------------------------------------------------------------------
    def eps(self, progress: float = 1.0) -> float:
        """Boundary softness ``ε(progress)`` (normalized, aspect-scaled distance units)."""
        return anneal_value(progress, self.eps_start, self.eps_end, self.schedule)

    def _plane(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.shape[-1] != self.ndim:
            raise ShapeError(f"{type(self).__name__} expects {self.ndim}-D coordinates")
        p = coords[..., list(self.plane_axes)]
        return p * torch.tensor(self.aspect, device=coords.device, dtype=coords.dtype)

    def shape_indicators(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        """Per-shape soft indicators ``(*shape, S)`` (inactive shapes are zero)."""
        raise NotImplementedError

    def indicator(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        """Union indicator ``χ(x) ∈ [0, 1]`` of all active shapes, shape ``(*shape)``."""
        return _union(self.shape_indicators(coords, progress), self.union)

    def raw(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        chi = self.shape_indicators(coords, progress)
        out = self.outside.to(chi)
        if self.per_shape_values:
            return out + (chi.unsqueeze(-1) * (self.inside.to(chi) - out)).sum(-2)
        u = _union(chi, self.union).unsqueeze(-1)
        return out + u * (self.inside[0].to(chi) - out)

    def _reset_values(self) -> None:
        with torch.no_grad():
            self.inside.copy_(self._in_init.to(self.inside))
            self.outside.copy_(self._out_init.to(self.outside))
            self.active.copy_(torch.arange(self.S, device=self.active.device) < self._n_active_init)

    def contrast_of(self, i: int) -> float:
        """Scalar contrast ``v_in − v_out`` of shape ``i`` (first channel)."""
        vin = self.inside[i if self.per_shape_values else 0, 0]
        return float(vin - self.outside[0])

    def can_grow(self) -> bool:
        return bool((~self.active).any())

    @property
    def n_active(self) -> int:
        return int(self.active.sum())


@register("field", "star_shape")
class StarShapeField(_ShapeBase):
    """``n_shapes`` star-shaped inclusions with Fourier-descriptor boundaries.

    Each shape ``i`` has a center ``c_i``, a base radius ``r_{0,i} = softplus(·)``, and
    coefficients ``(a_m, b_m)_{m ≤ M}``; the radius is ``r(θ) = r_0 · max_s(1 + Σ_m a_m cos mθ +
    b_m sin mθ, floor)`` (``max_s`` a smooth max that is the identity for ordinary shapes and keeps
    ``r > 0`` for extreme coefficients). Angles are computed from the regularized unit vector
    ``(x − c)/sqrt(|x − c|² + δ²)``, so the indicator is smooth (no ``atan2`` singularity).

    Args:
        ndim: 2, or 3 with ``extrude_axis`` (prism/cylinder extrusion over a depth interval).
        n_shapes: pool size ``S`` (all active unless ``n_active``).
        n_harmonics: ``M`` (0 = disks/ellipses via ``aspect``).
        heads: heads applied to the rendered raw field (default identity ``"x"``).
        centers: ``(S, 2)`` initial centers in normalized plane coordinates (default: a ring).
        radii: initial base radius (float or ``S`` values, aspect-scaled normalized units).
        inside, outside: initial raw values (floats or per-channel lists; ``inside`` per shape
            with ``per_shape_values``).
        learn_values: optimize the inside/outside values.
        per_shape_values: one inside value per shape (additive over overlaps) instead of a union.
        eps_start, eps_end, schedule: indicator softness ``ε(progress)``.
        union: ``"prob"`` (``1 − Π(1 − χ_i)``, default), ``"max"`` or ``"sum"`` (clipped).
        aspect / domain: scale the plane axes so shapes are round in *physical* units (``domain``
            sets ``aspect`` from its extents).
        plane_axes: the two axes of the shape plane.
        extrude_axis: 3-D only — axis of extrusion; ``depth_centers`` / ``half_heights`` give the
            initial interval ``[z_c − h, z_c + h]`` (normalized).
        n_active: number of initially active shapes (capacity pool for :meth:`grow`).
        radius_floor: floor of the normalized radius factor (keeps ``r > 0``).

    Per-group step sizes: ``OptimConfig(lr_mult={"field.center": ..., "field.radius_raw": ...,
    "field.coef": ...})`` (prefix ``"field.anomaly."`` when used as an anomaly component).

    Example::

        f = StarShapeField(n_shapes=2, n_harmonics=3, inside=0.1, outside=1.0)
        x = f(nefi.Domain.unit((64, 64)).coords(), progress=1.0)["x"]
        per = interface_length(f)          # (2,) perimeters (differentiable)
    """

    def __init__(
        self,
        ndim: int = 2,
        n_shapes: int = 1,
        n_harmonics: int = 4,
        heads: Heads | Mapping | None = None,
        centers: Sequence[Sequence[float]] | torch.Tensor | None = None,
        radii: float | Sequence[float] = 0.3,
        inside: float | Sequence[float] = 1.0,
        outside: float | Sequence[float] = 0.0,
        learn_values: bool = True,
        per_shape_values: bool = False,
        eps_start: float = 0.15,
        eps_end: float = 0.01,
        schedule: str = "geometric",
        union: str = "prob",
        aspect: Sequence[float] | None = None,
        domain: Any = None,
        plane_axes: Sequence[int] = (0, 1),
        extrude_axis: int | None = None,
        depth_centers: float | Sequence[float] = 0.0,
        half_heights: float | Sequence[float] = 0.3,
        n_active: int | None = None,
        radius_floor: float = 0.1,
    ) -> None:
        super().__init__(
            n_shapes,
            heads,
            ndim,
            inside,
            outside,
            learn_values,
            per_shape_values,
            eps_start,
            eps_end,
            schedule,
            union,
            aspect,
            domain,
            plane_axes,
            n_active,
        )
        if self.ndim == 3 and extrude_axis is None:
            others = [a for a in range(3) if a not in self.plane_axes]
            extrude_axis = others[0]
        if self.ndim not in (2, 3):
            raise ConfigError("StarShapeField supports ndim 2 (planar) or 3 (extruded)")
        self.extrude_axis = None if self.ndim == 2 else int(extrude_axis) % 3  # type: ignore[arg-type]
        self.M = int(n_harmonics)
        self.radius_floor = float(radius_floor)
        c0 = (
            _default_centers(self.S)
            if centers is None
            else torch.as_tensor(centers, dtype=torch.float32).reshape(self.S, 2)
        )
        self._center_init = c0.clone()
        r = torch.as_tensor(radii, dtype=torch.float32).flatten()
        r = r.expand(self.S).clone() if r.numel() == 1 else r
        if r.numel() != self.S or bool((r <= 0).any()):
            raise ConfigError(f"radii needs 1 or {self.S} positive values")
        self._r_init = inv_softplus(r)
        self.center = nn.Parameter(c0.clone())
        self.radius_raw = nn.Parameter(self._r_init.clone())
        self.coef = nn.Parameter(torch.zeros(self.S, self.M, 2))
        if self.extrude_axis is not None:
            zc = torch.as_tensor(depth_centers, dtype=torch.float32).flatten()
            hh = torch.as_tensor(half_heights, dtype=torch.float32).flatten()
            self._zc_init = zc.expand(self.S).clone() if zc.numel() == 1 else zc.clone()
            self._hh_init = inv_softplus(hh.expand(self.S).clone() if hh.numel() == 1 else hh)
            self.depth_center = nn.Parameter(self._zc_init.clone())
            self.half_height_raw = nn.Parameter(self._hh_init.clone())

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.center.copy_(self._center_init.to(self.center))
            self.radius_raw.copy_(self._r_init.to(self.radius_raw))
            self.coef.zero_()
            if self.extrude_axis is not None:
                self.depth_center.copy_(self._zc_init.to(self.depth_center))
                self.half_height_raw.copy_(self._hh_init.to(self.half_height_raw))
        self._reset_values()

    # --- geometry ------------------------------------------------------------------------
    @property
    def r0(self) -> torch.Tensor:
        """Base radii ``(S,)``."""
        return F.softplus(self.radius_raw)

    def radius_factor(self, cos1: torch.Tensor, sin1: torch.Tensor) -> torch.Tensor:
        """``max_s(1 + Σ_m a_m cos mθ + b_m sin mθ, floor)`` for ``cos θ, sin θ`` ``(..., S)``."""
        s = torch.ones_like(cos1)
        re, im = cos1, sin1
        coef = self.coef.to(cos1)
        for m in range(self.M):
            s = s + coef[:, m, 0] * re + coef[:, m, 1] * im
            re, im = re * cos1 - im * sin1, re * sin1 + im * cos1
        k = 20.0
        f = self.radius_floor
        return f + F.softplus(k * (s - f)) / k

    def radius(self, theta: torch.Tensor) -> torch.Tensor:
        """Boundary radius ``r_i(θ)`` for angles ``theta (..., S)`` or ``(n,)`` (broadcast)."""
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1).expand(-1, self.S)
        return self.r0.to(theta) * self.radius_factor(torch.cos(theta), torch.sin(theta))

    def shape_indicators(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        eps = self.eps(progress)
        p = self._plane(coords)  # (*shape, 2), aspect-scaled
        asp = torch.tensor(self.aspect, device=coords.device, dtype=coords.dtype)
        c = self.center.to(coords) * asp
        d = p.unsqueeze(-2) - c  # (*shape, S, 2)
        rho = torch.sqrt((d**2).sum(-1) + 1e-12)
        cos1, sin1 = d[..., 0] / rho, d[..., 1] / rho
        r = self.r0.to(coords) * self.radius_factor(cos1, sin1)
        chi = torch.sigmoid((r - rho) / eps)
        if self.extrude_axis is not None:
            z = coords[..., self.extrude_axis].unsqueeze(-1)
            hz = F.softplus(self.half_height_raw).to(coords)
            dz = torch.sqrt((z - self.depth_center.to(coords)) ** 2 + 1e-12)
            chi = chi * torch.sigmoid((hz - dz) / eps)
        return chi * self.active.to(chi)

    # --- analytic geometry ---------------------------------------------------------------
    def _theta_grid(self, n_quad: int, like: torch.Tensor) -> torch.Tensor:
        return torch.arange(n_quad, device=like.device, dtype=like.dtype) * (2 * math.pi / n_quad)

    def perimeters(self, n_quad: int = 256) -> torch.Tensor:
        """Boundary lengths ``(S,)`` (aspect-scaled normalized units): ``∫ sqrt(r² + r'²) dθ``."""
        th = self._theta_grid(n_quad, self.coef)
        r = self.radius(th)  # (n, S)
        dr = (torch.roll(r, -1, 0) - torch.roll(r, 1, 0)) / (2 * (2 * math.pi / n_quad))
        return torch.sqrt(r**2 + dr**2).mean(0) * (2 * math.pi)

    def areas(self, n_quad: int = 256) -> torch.Tensor:
        """Enclosed areas ``(S,)``: ``½ ∫ r(θ)² dθ`` (aspect-scaled normalized units)."""
        th = self._theta_grid(n_quad, self.coef)
        r = self.radius(th)
        return 0.5 * (r**2).mean(0) * (2 * math.pi)

    def descriptor_energy(self, order: float = 1.0) -> torch.Tensor:
        """``Σ_i Σ_m m^{2·order} (a_m² + b_m²)`` — a Sobolev penalty on boundary roughness."""
        if self.M == 0:
            return self.coef.sum() * 0.0
        m = torch.arange(1, self.M + 1, device=self.coef.device, dtype=self.coef.dtype)
        w = m ** (2.0 * order)
        return ((self.coef**2).sum(-1) * w).sum()

    # --- capacity growth -----------------------------------------------------------------
    @torch.no_grad()
    def grow(self, hint: Mapping | None = None) -> bool:
        """Activate the next pooled shape, at the hinted location if a gradient hint is given."""
        idle = (~self.active).nonzero().flatten()
        if idle.numel() == 0:
            return False
        i = int(idle[0])
        loc = hint_location(hint, self.contrast_of(i))
        if loc is not None:
            c = loc.to(self.center)
            self.center[i].copy_(c[list(self.plane_axes)])
            if self.extrude_axis is not None and c.numel() > self.extrude_axis:
                self.depth_center[i] = c[self.extrude_axis]
        self.coef[i].zero_()
        self.radius_raw[i].copy_(self._r_init[i].to(self.radius_raw))
        self.active[i] = True
        return True

    def capacity(self) -> dict[str, int]:
        return {"shapes": self.n_active, "max_shapes": self.S}

    def extra_repr(self) -> str:
        return (
            f"ndim={self.ndim}, n_shapes={self.S}, n_harmonics={self.M}, union={self.union!r}, "
            f"eps=({self.eps_start}->{self.eps_end}), extrude_axis={self.extrude_axis}, "
            f"heads={self.heads.names}"
        )


@register("field", "polygon")
class PolygonField(_ShapeBase):
    """``n_polygons`` simple polygons with learnable vertices, soft-rendered via a signed distance.

    The signed distance to each polygon (exact Euclidean distance to the edges, sign from the
    even-odd crossing rule) gives ``χ = σ(−d_signed/ε)``. Vertex gradients flow through the
    distance term; the sign is piecewise constant. Useful for box-like defects (NeFTY App. E.1).

    Args:
        n_polygons: number of polygons.
        n_vertices: vertices per polygon (initialized as regular polygons).
        radius: circumradius of the initial polygons.
        rotation: initial rotation (radians).
        Other args: see :class:`StarShapeField` (2-D only).
    """

    def __init__(
        self,
        n_polygons: int = 1,
        n_vertices: int = 4,
        heads: Heads | Mapping | None = None,
        centers: Sequence[Sequence[float]] | torch.Tensor | None = None,
        radius: float = 0.3,
        rotation: float = math.pi / 4,
        inside: float | Sequence[float] = 1.0,
        outside: float | Sequence[float] = 0.0,
        learn_values: bool = True,
        per_shape_values: bool = False,
        eps_start: float = 0.15,
        eps_end: float = 0.01,
        schedule: str = "geometric",
        union: str = "prob",
        aspect: Sequence[float] | None = None,
        domain: Any = None,
        plane_axes: Sequence[int] = (0, 1),
        n_active: int | None = None,
    ) -> None:
        super().__init__(
            n_polygons,
            heads,
            2,
            inside,
            outside,
            learn_values,
            per_shape_values,
            eps_start,
            eps_end,
            schedule,
            union,
            aspect,
            domain,
            plane_axes,
            n_active,
        )
        if n_vertices < 3:
            raise ConfigError("a polygon needs at least 3 vertices")
        self.V = int(n_vertices)
        c0 = (
            _default_centers(self.S)
            if centers is None
            else torch.as_tensor(centers, dtype=torch.float32).reshape(self.S, 2)
        )
        ang = rotation + torch.arange(self.V, dtype=torch.float32) * (2 * math.pi / self.V)
        ring = float(radius) * torch.stack([torch.cos(ang), torch.sin(ang)], -1)  # (V, 2)
        self._vert_init = c0[:, None, :] + ring[None]
        self.vertices = nn.Parameter(self._vert_init.clone())

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.vertices.copy_(self._vert_init.to(self.vertices))
        self._reset_values()

    def signed_distance(self, coords: torch.Tensor) -> torch.Tensor:
        """Signed distance ``(*shape, S)`` to each polygon (negative inside; aspect-scaled)."""
        p = self._plane(coords)
        batch = p.shape[:-1]
        pts = p.reshape(-1, 1, 1, 2)  # (N, 1, 1, 2)
        asp = torch.tensor(self.aspect, device=coords.device, dtype=coords.dtype)
        v = self.vertices.to(coords) * asp  # (S, V, 2)
        vj = torch.roll(v, 1, dims=1)  # previous vertex (IQ's j = i − 1)
        e = (vj - v)[None]  # (1, S, V, 2)
        w = pts - v[None]  # (N, S, V, 2)
        t = ((w * e).sum(-1) / (e * e).sum(-1).clamp_min(1e-12)).clamp(0.0, 1.0)
        b = w - e * t.unsqueeze(-1)
        d2 = (b * b).sum(-1).amin(-1)  # (N, S)
        py = pts[..., 1]
        c1 = py >= v[None, ..., 1]
        c2 = py < vj[None, ..., 1]
        c3 = e[..., 0] * w[..., 1] > e[..., 1] * w[..., 0]
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        inside = (flip.sum(-1) % 2) == 1
        sign = torch.where(inside, -torch.ones_like(d2), torch.ones_like(d2))
        return (sign * torch.sqrt(d2 + 1e-12)).reshape(*batch, self.S)

    def shape_indicators(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        chi = torch.sigmoid(-self.signed_distance(coords) / self.eps(progress))
        return chi * self.active.to(chi)

    def perimeters(self) -> torch.Tensor:
        asp = torch.tensor(self.aspect, device=self.vertices.device, dtype=self.vertices.dtype)
        v = self.vertices * asp
        return (torch.roll(v, -1, dims=1) - v).norm(dim=-1).sum(-1)

    def areas(self) -> torch.Tensor:
        """Shoelace areas ``(S,)`` (absolute value; aspect-scaled)."""
        asp = torch.tensor(self.aspect, device=self.vertices.device, dtype=self.vertices.dtype)
        v = self.vertices * asp
        x, y = v[..., 0], v[..., 1]
        return 0.5 * (x * torch.roll(y, -1, 1) - torch.roll(x, -1, 1) * y).sum(-1).abs()

    @torch.no_grad()
    def grow(self, hint: Mapping | None = None) -> bool:
        idle = (~self.active).nonzero().flatten()
        if idle.numel() == 0:
            return False
        i = int(idle[0])
        loc = hint_location(hint, self.contrast_of(i))
        base = self._vert_init[i].to(self.vertices)
        if loc is not None:
            base = base - base.mean(0) + loc.to(self.vertices)[list(self.plane_axes)]
        self.vertices[i].copy_(base)
        self.active[i] = True
        return True

    def extra_repr(self) -> str:
        return f"n_polygons={self.S}, n_vertices={self.V}, heads={self.heads.names}"


# ------------------------------------------------------------------------------------------
# interface measures and regularizers
# ------------------------------------------------------------------------------------------
def interface_length(
    field: Any,
    spacing: Sequence[float] | None = None,
    lo: float = 0.0,
    hi: float = 1.0,
    n_quad: int = 256,
    periodic_axes: Sequence[int] = (),
) -> torch.Tensor:
    """Interface measure of a shape field or of a two-phase field tensor (differentiable).

    * :class:`StarShapeField` / :class:`PolygonField` → per-shape boundary lengths ``(S,)``
      (analytic, in aspect-scaled normalized units; inactive shapes included);
    * a field tensor → coarea formula ``Per ≈ Σ |∇u| · cell`` with ``u = (x − lo)/(hi − lo)``
      (perimeter in 2-D, area in 3-D; ``spacing`` = physical cell size, default unit cells).

    Use as a perimeter regularizer (the shape analogue of TV, NeFTY Eq. 22).
    """
    if isinstance(field, StarShapeField):
        return field.perimeters(n_quad)
    if isinstance(field, PolygonField):
        return field.perimeters()
    x = torch.as_tensor(field)
    spacing = tuple(spacing) if spacing is not None else (1.0,) * x.ndim
    if len(spacing) != x.ndim:
        raise ConfigError(f"spacing has {len(spacing)} entries for a {x.ndim}-D field")
    u = (x - lo) / (hi - lo)
    diffs = forward_differences(u, spacing, tuple(periodic_axes))
    g = torch.sqrt(sum(d**2 for d in diffs) + 1e-16)
    return g.sum() * math.prod(float(h) for h in spacing)


def contour_regularizer(
    field: Field, perimeter: float = 1.0, smoothness: float = 0.0, order: float = 1.0
) -> torch.Tensor:
    """``perimeter · Σ|∂Ω_i| + smoothness · Σ m^{2·order}(a_m² + b_m²)`` over every shape field in
    ``field`` (searched recursively, so composites work)."""
    total = None
    for m in field.modules():
        if isinstance(m, StarShapeField | PolygonField):
            per = m.perimeters() * m.active.to(m.inside)
            term = perimeter * per.sum()
            if smoothness and isinstance(m, StarShapeField):
                term = term + smoothness * m.descriptor_energy(order)
            total = term if total is None else total + term
    if total is None:
        raise ConfigError("contour_regularizer: no StarShapeField / PolygonField found")
    return total


@register("loss", "contour")
class ContourRegularizer(Loss):
    """Loss wrapper of :func:`contour_regularizer` on ``ctx.field_module`` (perimeter + boundary
    smoothness of every shape component)."""

    def __init__(
        self,
        perimeter: float = 1.0,
        smoothness: float = 0.0,
        order: float = 1.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name or "contour")
        self.perimeter, self.smoothness, self.order = float(perimeter), float(smoothness), order

    def forward(self, ctx: Context) -> torch.Tensor:
        if ctx.field_module is None:
            raise ConfigError("ContourRegularizer needs ctx.field_module")
        return contour_regularizer(ctx.field_module, self.perimeter, self.smoothness, self.order)


def _values(v: float | Sequence[float], rows: int, c: int, label: str) -> torch.Tensor:
    t = torch.as_tensor(v, dtype=torch.float32)
    if t.ndim == 0:
        return t.expand(rows, c).clone()
    if t.ndim == 1:
        if t.numel() == c and rows == 1:
            return t.view(1, c).clone()
        if t.numel() == rows:
            return t.view(rows, 1).expand(rows, c).clone()
        if t.numel() == 1:
            return t.view(1, 1).expand(rows, c).clone()
    if tuple(t.shape) == (rows, c):
        return t.clone()
    raise ConfigError(
        f"{label} must be a float, {c} channel values or ({rows}, {c}); got {t.shape}"
    )


__all__ = [
    "ContourRegularizer",
    "PolygonField",
    "StarShapeField",
    "contour_regularizer",
    "hint_location",
    "interface_length",
]
