"""Synthetic subsurface-defect scenes for pulsed-thermography tomography (NeFTY App. E.1).

Each sample is a diffusivity field on the slab ``[0, L]² × [0, H]``: a bulk (homogeneous
``α_base ~ U(0.1, 0.2)``, or 3–4 strata along ``z`` with independent bulk values) containing one to
four ellipsoidal, cylindrical or box-shaped defects with ``α_defect ~ U(0.005, 0.015)`` buried at
random depths (≈ 1:20 defect-to-bulk contrast, App. E.1 "Defect contrast scaling").

Scenes are drawn as *parameters* first (:meth:`ThermalScenes.draw`, a fixed number of random draws)
and rasterized analytically at any resolution (:meth:`ThermalScenes.rasterize`), so the same
scene can be evaluated on a finer grid by a supersampling data generator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from ...bench.base import SceneGenerator
from ...domain import Domain
from ...errors import ConfigError

DEFECT_SHAPES = ("ellipsoid", "cylinder", "box")

__all__ = ["DEFECT_SHAPES", "DefectSpec", "SceneSpec", "ThermalScenes"]


@dataclass
class DefectSpec:
    """One buried defect (physical units).

    Attributes:
        kind: ``"ellipsoid"`` | ``"cylinder"`` (circular, axis along z — flat-bottom-hole like) |
            ``"box"``.
        center: ``(x, [y,] z)`` centre; ``z`` is the depth below the front face.
        radii: lateral semi-axes (``(r_x, r_y)`` in 3-D; the cylinder uses ``r_x``).
        half_thickness: half extent along ``z``.
        angle: in-plane rotation (radians, 3-D only).
        alpha: defect diffusivity.
    """

    kind: str
    center: tuple[float, ...]
    radii: tuple[float, ...]
    half_thickness: float
    angle: float
    alpha: float


@dataclass
class SceneSpec:
    """Parameters of one sample: scene class, bulk strata and defects."""

    cls: str
    layer_bounds: tuple[float, ...]  # interface depths (len = n_layers - 1)
    layer_alpha: tuple[float, ...]  # bulk diffusivity per stratum (front to back)
    defects: list[DefectSpec] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ThermalScenes(SceneGenerator):
    """Scene generator with classes ``"homogeneous"`` and ``"layered"`` (NeFTY App. E.1).

    Args:
        domain: the slab (last axis = depth ``z``; the front face is at ``extent[-1][0]``).
        alpha_base_range: bulk diffusivity range ``U(0.1, 0.2)``.
        alpha_defect_range: defect diffusivity range ``U(0.005, 0.015)``.
        n_defects: inclusive range of the number of defects (1–4).
        n_layers: inclusive range of the number of strata for ``"layered"`` (3–4).
        shapes: defect shapes to draw from.
        radius_range: lateral semi-axis range (physical units).
        half_thickness_range: half-thickness range along ``z`` (physical units).
        depth_range: range of the defect-centre depth as a fraction of the thickness ``H``
            (clipped so the defect stays inside the slab).
        lateral_margin: minimum distance of defect centres from the lateral edges (irrelevant for
            periodic lateral boundaries but keeps defects inside the camera footprint).
        min_layer_fraction: minimum stratum thickness as a fraction of ``H``.
        periodic: wrap defects around the periodic lateral boundaries (minimum image).
    """

    classes = ("homogeneous", "layered")

    def __init__(
        self,
        domain: Domain,
        alpha_base_range: Sequence[float] = (0.1, 0.2),
        alpha_defect_range: Sequence[float] = (0.005, 0.015),
        n_defects: Sequence[int] = (1, 4),
        n_layers: Sequence[int] = (3, 4),
        shapes: Sequence[str] = DEFECT_SHAPES,
        radius_range: Sequence[float] = (0.6, 1.6),
        half_thickness_range: Sequence[float] = (0.08, 0.2),
        depth_range: Sequence[float] = (0.25, 0.75),
        lateral_margin: float = 1.0,
        min_layer_fraction: float = 0.15,
        periodic: bool = True,
    ) -> None:
        super().__init__(domain)
        if domain.ndim < 2:
            raise ConfigError("thermal scenes need a >= 2-D domain (lateral axes + depth)")
        bad = [s for s in shapes if s not in DEFECT_SHAPES]
        if bad or not shapes:
            raise ConfigError(f"unknown defect shapes {bad}; use a subset of {DEFECT_SHAPES}")
        self.alpha_base_range = tuple(float(v) for v in alpha_base_range)
        self.alpha_defect_range = tuple(float(v) for v in alpha_defect_range)
        self.n_defects = (int(n_defects[0]), int(n_defects[1]))
        self.n_layers = (int(n_layers[0]), int(n_layers[1]))
        self.shapes = tuple(shapes)
        self.radius_range = tuple(float(v) for v in radius_range)
        self.half_thickness_range = tuple(float(v) for v in half_thickness_range)
        self.depth_range = tuple(float(v) for v in depth_range)
        self.lateral_margin = float(lateral_margin)
        self.min_layer_fraction = float(min_layer_fraction)
        self.periodic = bool(periodic)

    # ---- parameters --------------------------------------------------------------------
    @property
    def thickness(self) -> float:
        lo, hi = self.domain.extent[-1]
        return hi - lo

    def draw(self, rng: np.random.Generator, cls: str | None = None) -> SceneSpec:
        """Draw the scene parameters (the number of random draws does not depend on ``shape``)."""
        cls = self.check_class(cls)
        H = self.thickness
        lat = self.domain.extent[:-1]
        if cls == "layered":
            n = int(rng.integers(self.n_layers[0], self.n_layers[1] + 1))
            free = max(0.0, 1.0 - n * self.min_layer_fraction)
            frac = self.min_layer_fraction + free * rng.dirichlet(np.ones(n))
            bounds = tuple(float(b) for b in np.cumsum(frac)[:-1] * H)
            layer_alpha = tuple(float(v) for v in rng.uniform(*self.alpha_base_range, size=n))
        else:
            bounds = ()
            layer_alpha = (float(rng.uniform(*self.alpha_base_range)),)
        k = int(rng.integers(self.n_defects[0], self.n_defects[1] + 1))
        defects = []
        for _ in range(k):
            kind = self.shapes[int(rng.integers(len(self.shapes)))]
            radii = tuple(float(r) for r in rng.uniform(*self.radius_range, size=len(lat)))
            hz = float(rng.uniform(*self.half_thickness_range))
            centre_lat = [
                float(rng.uniform(lo + self.lateral_margin, hi - self.lateral_margin))
                if hi - lo > 2 * self.lateral_margin
                else 0.5 * (lo + hi)
                for lo, hi in lat
            ]
            zlo, zhi = max(self.depth_range[0] * H, hz), min(self.depth_range[1] * H, H - hz)
            u = float(rng.uniform())
            zc = zlo + u * (zhi - zlo) if zhi > zlo else 0.5 * H
            angle = float(rng.uniform(0.0, math.pi))
            a_def = float(rng.uniform(*self.alpha_defect_range))
            defects.append(DefectSpec(kind, (*centre_lat, zc), radii, hz, angle, a_def))
        return SceneSpec(cls, bounds, layer_alpha, defects)

    # ---- rasterization -----------------------------------------------------------------
    def rasterize(
        self, spec: SceneSpec, shape: Sequence[int] | None = None
    ) -> tuple[dict[str, torch.Tensor], dict]:
        """Evaluate a scene on the cell centres of ``shape`` (default: the domain's grid).

        Returns:
            ``({"alpha": α}, meta)`` with ``meta = {"alpha_base": bulk field, "defect_mask":
            bool mask, "spec": spec}``.
        """
        c = self.domain.physical_coords(shape, dtype=torch.float64)
        z = c[..., -1] - self.domain.extent[-1][0]
        base = torch.full(z.shape, spec.layer_alpha[0], dtype=torch.float64)
        for b, a in zip(spec.layer_bounds, spec.layer_alpha[1:]):
            base = torch.where(z >= b, torch.full_like(base, a), base)
        alpha = base.clone()
        mask = torch.zeros(z.shape, dtype=torch.bool)
        for d in spec.defects:
            m = self._defect_mask(d, c, z)
            alpha = torch.where(m, torch.full_like(alpha, d.alpha), alpha)
            mask |= m
        meta = {"alpha_base": base.float(), "defect_mask": mask, "spec": spec}
        return {"alpha": alpha.float()}, meta

    def _lateral_offsets(self, d: DefectSpec, c: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for ax, (lo, hi) in enumerate(self.domain.extent[:-1]):
            off = c[..., ax] - d.center[ax]
            if self.periodic:
                L = hi - lo
                off = torch.remainder(off + 0.5 * L, L) - 0.5 * L
            out.append(off)
        return out

    def _defect_mask(self, d: DefectSpec, c: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        offs = self._lateral_offsets(d, c)
        dz = z - d.center[-1]
        if len(offs) == 2:
            ca, sa = math.cos(d.angle), math.sin(d.angle)
            u = ca * offs[0] + sa * offs[1]
            v = -sa * offs[0] + ca * offs[1]
            lat = [u, v]
        else:
            lat = offs
        if d.kind == "ellipsoid":
            q = sum((x / r) ** 2 for x, r in zip(lat, d.radii)) + (dz / d.half_thickness) ** 2
            m = q <= 1.0
        elif d.kind == "cylinder":
            r2 = sum(x**2 for x in lat)
            m = (r2 <= d.radii[0] ** 2) & (dz.abs() <= d.half_thickness)
        else:  # box
            m = dz.abs() <= d.half_thickness
            for x, r in zip(lat, d.radii):
                m = m & (x.abs() <= r)
        if not bool(m.any()):  # sub-cell defect at this resolution: keep the nearest cell
            dist = sum(x**2 for x in lat) + dz**2
            m = dist == dist.min()
        return m

    def sample(
        self,
        rng: np.random.Generator,
        cls: str | None = None,
        shape: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Ground-truth fields ``{"alpha": (*shape)}`` for one random scene."""
        return self.rasterize(self.draw(rng, cls), shape)[0]

    def sample_with_meta(
        self,
        rng: np.random.Generator,
        cls: str | None = None,
        shape: Sequence[int] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        """Like :meth:`sample` but also returns the bulk field, defect mask and parameters."""
        return self.rasterize(self.draw(rng, cls), shape)
