"""Shared plumbing of the volumetric (3-D) exemplars: ``ct3d``, ``deconvolution3d``, ``dot3d`` and
``photoacoustic3d``.

* :func:`render_volume` — cell-averaged rendering of an analytic 3-D scene. Scene parameters are
  drawn first (independently of the grid); the continuous scene is then evaluated on a sub-grid
  whose resolution is fixed relative to the *native* grid (``render_factor ×`` native per axis) and
  area-averaged. The native ground truth and the supersampled data-generation grid are therefore
  exact area-averages of one and the same point-sampled volume (no aliasing mismatch between the
  ground truth and the data). Evaluation is chunked along the first axis, so paper-scale volumes
  fit in memory. :func:`gaussian_cell_average` is the exact (erf) cell average of an axis-aligned
  Gaussian, used for sub-voxel point sources.
* 3-D primitives (point functions of coordinate grids ``x, y, z``): ellipsoids, boxes, cylinders,
  spheres, Gaussian blobs, tubes around polylines, ellipsoidal shells; ZXZ Euler rotations;
  persistent random walks for filaments and vessels; :class:`Painter3D` for composition.
* :class:`PresetInstance` — an :class:`~nefi.instances.base.Instance` with named ``PRESETS``
  (``preset="smoke"``), the pattern of ``thermal_tomography``.
* Volumetric metrics: :func:`iou_above`, excess-weighted depth centroids and 2.5-D depth maps (the
  NeFTY App. H projections generalized to "excess over a background").
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from ..domain import Domain
from ..errors import ConfigError, ShapeError
from ..utils.tensor import shape_tuple
from .base import Instance

__all__ = [
    "Painter3D",
    "PresetInstance",
    "SceneFn3",
    "box",
    "cylinder",
    "depth_centroid",
    "depth_map",
    "ellipsoid",
    "ellipsoid_level",
    "excess",
    "gaussian_blob",
    "gaussian_cell_average",
    "iou_above",
    "local_coords",
    "polyline_distance",
    "random_curve",
    "random_unit_vector",
    "render_volume",
    "rotation_zxz",
    "smooth_indicator",
    "sphere",
    "tube",
]

#: a 3-D scene maps coordinate grids ``(X, Y, Z)`` (float64) to values of the same shape.
SceneFn3 = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------
def render_volume(
    fn: SceneFn3,
    domain: Domain,
    shape: Sequence[int] | None = None,
    render_factor: int = 2,
    normalized: bool = False,
    max_points: int = 1 << 21,
) -> torch.Tensor:
    """Cell-averaged rendering of a continuous 3-D scene (float64).

    Args:
        fn: point function of the coordinate grids ``(X, Y, Z)`` (physical coordinates, or
            normalized ``[-1, 1]`` ones with ``normalized=True``).
        domain: the *native* domain (fixes the sub-grid resolution).
        shape: target grid (default native); every target cell averages
            ``ceil(render_factor · native / shape)`` sub-samples per axis, so a 2× finer grid and
            the native grid share the same sub-samples when ``render_factor`` is even.
        render_factor: sub-grid resolution relative to the native grid.
        normalized: pass normalized coordinates instead of physical ones.
        max_points: sub-samples evaluated per chunk (memory bound; chunks are whole cell rows).
    """
    if domain.ndim != 3:
        raise ShapeError(f"render_volume needs a 3-D domain, got {domain.ndim}-D")
    shape = tuple(domain.shape) if shape is None else shape_tuple(shape)
    if len(shape) != 3:
        raise ShapeError(f"render_volume needs a 3-D target shape, got {shape}")
    ss = [max(1, math.ceil(render_factor * n0 / n)) for n0, n in zip(domain.shape, shape)]
    fine = tuple(n * s for n, s in zip(shape, ss))
    axes = domain.at(fine).axis_coords(normalized=normalized, dtype=torch.float64)
    per_cell_row = fine[1] * fine[2] * ss[0]
    rows = max(1, int(max_points) // max(1, per_cell_row)) * ss[0]
    out = []
    for i0 in range(0, fine[0], rows):
        X, Y, Z = torch.meshgrid(axes[0][i0 : i0 + rows], axes[1], axes[2], indexing="ij")
        v = torch.as_tensor(fn(X, Y, Z), dtype=torch.float64)
        if tuple(v.shape) != tuple(X.shape):
            v = v.expand(X.shape)
        if ss != [1, 1, 1]:
            v = F.avg_pool3d(v[None, None], kernel_size=tuple(ss))[0, 0]
        out.append(v)
    return torch.cat(out, dim=0)


def gaussian_cell_average(
    domain: Domain,
    shape: Sequence[int] | None,
    centers: torch.Tensor | Sequence[Sequence[float]],
    sigmas: torch.Tensor | Sequence[float],
    amplitudes: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Exact cell averages of a sum of axis-aligned Gaussians ``A exp(−|x − c|²/2σ²)`` (float64).

    The average over the cell ``[lo, hi]`` factorizes into
    ``∏_d σ√(π/2) [erf((hi_d − c_d)/(σ√2)) − erf((lo_d − c_d)/(σ√2))] / h_d``, so sub-voxel point
    sources are rendered without aliasing and consistently at every resolution (physical units).

    Args:
        domain: the domain (physical extents).
        shape: target grid (default native).
        centers: ``(n, 3)`` physical centers.
        sigmas: ``(n,)`` isotropic widths, or ``(n, 3)`` per-axis widths.
        amplitudes: ``(n,)`` peak values.
    """
    dom = domain if shape is None else domain.at(shape)
    c = torch.as_tensor(centers, dtype=torch.float64).reshape(-1, 3)
    s = torch.as_tensor(sigmas, dtype=torch.float64)
    s = s.reshape(-1, 1).expand(-1, 3) if s.ndim <= 1 else s.reshape(-1, 3)
    a = torch.as_tensor(amplitudes, dtype=torch.float64).reshape(-1)
    out = torch.zeros(dom.shape, dtype=torch.float64)
    edges = []
    for (lo, hi), n in zip(dom.extent, dom.shape):
        edges.append(torch.linspace(lo, hi, n + 1, dtype=torch.float64))
    for k in range(c.shape[0]):
        factors = []
        for d in range(3):
            e = edges[d]
            h = float(e[1] - e[0])
            z = (e - c[k, d]) / (s[k, d] * math.sqrt(2.0))
            cdf = torch.erf(z)
            factors.append(s[k, d] * math.sqrt(math.pi / 2.0) * (cdf[1:] - cdf[:-1]) / h)
        out += a[k] * factors[0][:, None, None] * factors[1][None, :, None] * factors[2][None, None]
    return out


# --------------------------------------------------------------------------------------------
# primitives (point functions of coordinate grids)
# --------------------------------------------------------------------------------------------
def rotation_zxz(phi: float = 0.0, theta: float = 0.0, psi: float = 0.0) -> torch.Tensor:
    """Rotation matrix ``R = R_z(φ) R_x(θ) R_z(ψ)`` (degrees, float64), body → world.

    The Euler convention of the 3-D Shepp–Logan phantom (Kak & Slaney 1988; Schabel's
    ``phantom3d``). A point of a body is ``p = c + R q`` with ``q`` in the body frame.
    """
    a, b, g = (math.radians(float(v)) for v in (phi, theta, psi))

    def rz(t: float) -> torch.Tensor:
        return torch.tensor(
            [[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )

    rx = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, math.cos(b), -math.sin(b)], [0.0, math.sin(b), math.cos(b)]],
        dtype=torch.float64,
    )
    return rz(a) @ rx @ rz(g)


def local_coords(x, y, z, center: Sequence[float], rot: torch.Tensor | None = None):
    """Body-frame coordinates ``q = Rᵀ (p − c)`` of the grids ``(x, y, z)``."""
    dx, dy, dz = x - float(center[0]), y - float(center[1]), z - float(center[2])
    if rot is None:
        return dx, dy, dz
    r = [[float(rot[i, j]) for j in range(3)] for i in range(3)]
    u = r[0][0] * dx + r[1][0] * dy + r[2][0] * dz
    v = r[0][1] * dx + r[1][1] * dy + r[2][1] * dz
    w = r[0][2] * dx + r[1][2] * dy + r[2][2] * dz
    return u, v, w


def ellipsoid_level(x, y, z, center, semi_axes, rot=None) -> torch.Tensor:
    """``(u/a)² + (v/b)² + (w/c)²`` in the body frame (≤ 1 inside the ellipsoid)."""
    u, v, w = local_coords(x, y, z, center, rot)
    a, b, c = (float(s) for s in semi_axes)
    return (u / a) ** 2 + (v / b) ** 2 + (w / c) ** 2


def ellipsoid(x, y, z, center, semi_axes, rot=None) -> torch.Tensor:
    """Indicator of an ellipsoid with semi-axes ``(a, b, c)`` rotated by ``rot``."""
    return (ellipsoid_level(x, y, z, center, semi_axes, rot) <= 1.0).to(x.dtype)


def sphere(x, y, z, center, radius: float) -> torch.Tensor:
    """Indicator of a ball."""
    r = float(radius)
    return ellipsoid(x, y, z, center, (r, r, r))


def box(x, y, z, center, size, rot=None) -> torch.Tensor:
    """Indicator of a box with side lengths ``size`` rotated by ``rot``."""
    u, v, w = local_coords(x, y, z, center, rot)
    sx, sy, sz = (0.5 * float(s) for s in size)
    return ((u.abs() <= sx) & (v.abs() <= sy) & (w.abs() <= sz)).to(x.dtype)


def cylinder(x, y, z, center, radius: float, half_length: float, rot=None) -> torch.Tensor:
    """Indicator of a finite cylinder (axis = body ``w`` axis)."""
    u, v, w = local_coords(x, y, z, center, rot)
    return ((u**2 + v**2 <= float(radius) ** 2) & (w.abs() <= float(half_length))).to(x.dtype)


def gaussian_blob(x, y, z, center, sigma, rot=None) -> torch.Tensor:
    """Unit-peak (anisotropic) Gaussian ``exp(−½ Σ (q_i/σ_i)²)``."""
    s = (float(sigma),) * 3 if isinstance(sigma, int | float) else tuple(float(v) for v in sigma)
    u, v, w = local_coords(x, y, z, center, rot)
    return torch.exp(-0.5 * ((u / s[0]) ** 2 + (v / s[1]) ** 2 + (w / s[2]) ** 2))


def smooth_indicator(signed_distance: torch.Tensor, width: float) -> torch.Tensor:
    """``½(1 − tanh(d / w))`` — a smooth-edged indicator of ``{d < 0}`` (``w = 0``: hard)."""
    if width <= 0:
        return (signed_distance <= 0).to(signed_distance.dtype)
    return 0.5 * (1.0 - torch.tanh(signed_distance / float(width)))


def polyline_distance(x, y, z, vertices: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """Euclidean distance from every grid point to an open polyline ``vertices (V, 3)``.

    Segments are processed in chunks of ``chunk`` to bound memory (``points × chunk``).
    """
    v = torch.as_tensor(vertices, dtype=torch.float64).reshape(-1, 3)
    p = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(torch.float64)
    if v.shape[0] == 1:
        return (p - v[0]).norm(dim=-1).reshape(x.shape)
    best = torch.full((p.shape[0],), float("inf"), dtype=torch.float64)
    a_all, b_all = v[:-1], v[1:]
    for s0 in range(0, a_all.shape[0], chunk):
        a, b = a_all[s0 : s0 + chunk], b_all[s0 : s0 + chunk]
        ab = b - a
        t = ((p[:, None, :] - a[None]) * ab[None]).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-30)
        proj = a[None] + t.clamp(0.0, 1.0)[..., None] * ab[None]
        d = (p[:, None, :] - proj).norm(dim=-1).min(dim=1).values
        best = torch.minimum(best, d)
    return best.reshape(x.shape)


def tube(
    x, y, z, vertices: torch.Tensor, radius: float, profile: str = "hard", width: float = 0.0
) -> torch.Tensor:
    """Tube around a polyline: ``"hard"`` / ``"smooth"`` (tanh edge of ``width``) indicator of
    ``{d ≤ radius}``, or ``"gaussian"`` cross-section ``exp(−d²/2r²)`` (unit peak)."""
    d = polyline_distance(x, y, z, vertices)
    if profile == "gaussian":
        return torch.exp(-0.5 * (d / float(radius)) ** 2)
    if profile == "smooth":
        return smooth_indicator(d - float(radius), width)
    if profile == "hard":
        return (d <= float(radius)).to(x.dtype)
    raise ConfigError(f"unknown tube profile {profile!r}; use 'hard', 'smooth' or 'gaussian'")


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """A direction uniformly distributed on the sphere."""
    v = rng.standard_normal(3)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def random_curve(
    rng: np.random.Generator,
    start: Sequence[float],
    direction: Sequence[float],
    n_steps: int,
    step: float,
    persistence: float,
    lo: Sequence[float],
    hi: Sequence[float],
) -> torch.Tensor:
    """Persistent random walk (a smooth random curve) confined to the box ``[lo, hi]``.

    ``d_{k+1} = normalize(persistence · d_k + (1 − persistence) · ξ_k)``, ``ξ_k`` uniform on the
    sphere; directions reflect at the box faces. Returns ``(n_steps + 1, 3)`` vertices (float64).
    """
    p = np.asarray(start, dtype=np.float64).copy()
    d = np.asarray(direction, dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    lo_a, hi_a = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    pts = [p.copy()]
    for _ in range(int(n_steps)):
        xi = random_unit_vector(rng)
        d = persistence * d + (1.0 - persistence) * xi
        d = d / max(float(np.linalg.norm(d)), 1e-12)
        q = p + step * d
        for ax in range(3):
            if q[ax] < lo_a[ax] or q[ax] > hi_a[ax]:
                d[ax] = -d[ax]
                q[ax] = np.clip(p[ax] + step * d[ax], lo_a[ax], hi_a[ax])
        p = q
        pts.append(p.copy())
    return torch.as_tensor(np.stack(pts), dtype=torch.float64)


class Painter3D:
    """Composes 3-D primitives: additive terms and painted (overwriting) regions.

    Args:
        background: constant background value.
        clip: optional ``(lo, hi)`` range applied to the *point-sampled* volume (before cell
            averaging), so clipped scenes stay consistent across rendering resolutions.
    """

    def __init__(
        self, background: float = 0.0, clip: tuple[float | None, float | None] | None = (0.0, 1.0)
    ) -> None:
        self.background = float(background)
        self.clip = clip
        self.ops: list[tuple[str, float, SceneFn3]] = []

    def add(self, value: float, fn: SceneFn3) -> Painter3D:
        self.ops.append(("add", float(value), fn))
        return self

    def paint(self, value: float, fn: SceneFn3) -> Painter3D:
        self.ops.append(("paint", float(value), fn))
        return self

    def __call__(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        vol = torch.full_like(x, self.background)
        for mode, value, fn in self.ops:
            m = fn(x, y, z)
            vol = vol + value * m if mode == "add" else vol * (1.0 - m) + value * m
        if self.clip is not None:
            vol = vol.clamp(min=self.clip[0], max=self.clip[1])
        return vol


# --------------------------------------------------------------------------------------------
# instances with named presets
# --------------------------------------------------------------------------------------------
class PresetInstance(Instance):
    """:class:`~nefi.instances.base.Instance` with named partial configs (``PRESETS``).

    ``Cls(cfg=None, preset=None, **overrides)``: the preset is applied first, then ``cfg`` (a
    dict or dataclass), then ``overrides``; a ``"preset"`` key inside a ``cfg`` mapping (YAML
    configs, ``nefi run --set preset=smoke``) is honoured too. Subclasses may implement
    ``_validate()``.
    """

    PRESETS: ClassVar[dict[str, dict[str, Any]]] = {}

    def __init__(self, cfg: Any = None, preset: str | None = None, **overrides: Any) -> None:
        if isinstance(cfg, Mapping) and "preset" in cfg:
            cfg = dict(cfg)
            preset = cfg.pop("preset") or preset
        if preset is not None:
            if preset not in self.PRESETS:
                raise ConfigError(f"unknown preset {preset!r}; known: {sorted(self.PRESETS)}")
            if dataclasses.is_dataclass(cfg):
                cfg = dataclasses.asdict(cfg)
            cfg = {**self.PRESETS[preset], **dict(cfg or {})}
        super().__init__(cfg, **overrides)
        self.preset = preset
        self._validate()

    def _validate(self) -> None:
        """Hook: raise :class:`~nefi.errors.ConfigError` on inconsistent settings."""


# --------------------------------------------------------------------------------------------
# volumetric metrics
# --------------------------------------------------------------------------------------------
def iou_above(pred, gt, tau: float) -> float:
    """IoU of the super-level sets ``{pred > τ}`` and ``{gt > τ}`` (1.0 if both are empty)."""
    p = torch.as_tensor(pred).detach()
    g = torch.as_tensor(gt).detach().to(p.device)
    a, b = p > tau, g > tau
    union = int((a | b).sum())
    if union == 0:
        return 1.0
    return float((a & b).sum()) / union


def excess(field: torch.Tensor, background: float | torch.Tensor) -> torch.Tensor:
    """Non-negative excess ``[f − background]_+`` (float64)."""
    f = torch.as_tensor(field).detach().double()
    return (f - torch.as_tensor(background, dtype=torch.float64)).clamp_min(0.0)


def depth_centroid(weights: torch.Tensor, depths: torch.Tensor, axis: int = -1) -> float:
    """Weighted mean depth ``Σ w z / Σ w`` of a non-negative 3-D weight volume (NaN if empty).

    ``depths`` holds the cell-center depth along ``axis`` (1-D).
    """
    w = torch.as_tensor(weights, dtype=torch.float64)
    z = torch.as_tensor(depths, dtype=torch.float64)
    view = [1] * w.ndim
    view[axis] = -1
    tot = float(w.sum())
    if tot <= 0:
        return float("nan")
    return float((w * z.view(view)).sum()) / tot


def depth_map(weights: torch.Tensor, depths: torch.Tensor, mask2d: torch.Tensor | None = None):
    """2.5-D depth map: per lateral column the weighted mean depth along the last axis.

    Columns with zero weight (or outside ``mask2d``) are NaN (NeFTY App. H-style projection with
    the absorption / attenuation excess as the weight).
    """
    w = torch.as_tensor(weights, dtype=torch.float64)
    z = torch.as_tensor(depths, dtype=torch.float64)
    tot = w.sum(-1)
    d = (w * z).sum(-1) / tot.clamp_min(1e-300)
    ok = tot > 0
    if mask2d is not None:
        ok = ok & torch.as_tensor(mask2d).bool()
    return torch.where(ok, d, torch.full_like(d, float("nan")))
