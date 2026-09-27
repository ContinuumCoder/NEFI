"""Absorbing inclusions in a scattering slab (physical mm coordinates, ``μ_a`` in mm⁻¹).

Inclusions are spheres of raised absorption in a homogeneous background, drawn first (centers,
radii, contrasts) and rendered as cell averages at any grid (:func:`~nefi.instances._volumetric.
render_volume`), so the native ground truth and the 2× data-generation grid agree exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...bench.base import SceneGenerator
from ...domain import Domain
from .._volumetric import Painter3D, render_volume, sphere


@dataclass
class Inclusion:
    """One spherical absorber: physical center ``(x, y, depth)``, radius and ``μ_a``."""

    center: tuple[float, float, float]
    radius: float
    mua: float

    @property
    def depth(self) -> float:
        return float(self.center[2])


def _u(rng: np.random.Generator, r: tuple[float, float]) -> float:
    return float(rng.uniform(float(r[0]), float(r[1])))


class DOT3DScenes(SceneGenerator):
    """Absorption maps ``μ_a(x, y, z)`` of a slab (depth = last axis, top face at ``z = 0``).

    * ``single`` — one inclusion at a moderate depth (``depth_single``);
    * ``multi`` — 2–3 laterally separated inclusions at different depths (``depth_multi``);
    * ``deep`` — one inclusion deep in the slab (``depth_deep``), where the surface sensitivity
      has decayed (the hard class, as the depth-decaying sensitivity map shows).

    Args:
        domain: native slab domain (mm).
        mua_background: homogeneous background absorption (mm⁻¹).
        inclusion_mua: ``(lo, hi)`` inclusion absorption (mm⁻¹).
        radius: ``(lo, hi)`` inclusion radius (mm).
        depth_single / depth_multi / depth_deep: ``(lo, hi)`` center depths (mm) per class.
        lateral_span: centers lie in the central ``lateral_span`` fraction of the top face.
        min_gap: minimum lateral gap between inclusions of the ``multi`` class (mm).
        render_factor: sub-samples per voxel and axis of the cell-averaged rendering.
    """

    classes = ("single", "multi", "deep")

    def __init__(
        self,
        domain: Domain,
        mua_background: float = 0.01,
        inclusion_mua: tuple[float, float] = (0.03, 0.05),
        radius: tuple[float, float] = (3.0, 5.0),
        depth_single: tuple[float, float] = (5.0, 8.0),
        depth_multi: tuple[float, float] = (4.0, 11.0),
        depth_deep: tuple[float, float] = (10.0, 13.0),
        lateral_span: float = 0.6,
        min_gap: float = 2.0,
        render_factor: int = 2,
    ) -> None:
        super().__init__(domain)
        self.mua_background = float(mua_background)
        self.inclusion_mua = tuple(inclusion_mua)
        self.radius = tuple(radius)
        self.depths = {"single": depth_single, "multi": depth_multi, "deep": depth_deep}
        self.lateral_span = float(lateral_span)
        self.min_gap = float(min_gap)
        self.render_factor = int(render_factor)

    def _lateral(self, rng: np.random.Generator) -> tuple[float, float]:
        (x0, x1), (y0, y1) = self.domain.extent[0], self.domain.extent[1]
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        hx, hy = 0.5 * self.lateral_span * (x1 - x0), 0.5 * self.lateral_span * (y1 - y0)
        return float(rng.uniform(cx - hx, cx + hx)), float(rng.uniform(cy - hy, cy + hy))

    def _depth(self, rng: np.random.Generator, cls: str, r: float) -> float:
        z0, z1 = self.domain.extent[2]
        lo, hi = self.depths[cls]
        # keep the sphere inside the slab (one voxel of margin at the faces)
        h = self.domain.spacing()[2]
        lo = max(float(lo), z0 + r + h)
        hi = max(lo, min(float(hi), z1 - r - h))
        return float(rng.uniform(lo, hi))

    def draw(self, rng: np.random.Generator, cls: str | None = None) -> list[Inclusion]:
        """Draw the inclusions of one scene (grid independent)."""
        cls = self.check_class(cls)
        n = int(rng.integers(2, 4)) if cls == "multi" else 1
        out: list[Inclusion] = []
        for _ in range(n):
            r = _u(rng, self.radius)
            for _attempt in range(50):  # rejection sampling of laterally separated inclusions
                x, y = self._lateral(rng)
                if all(
                    np.hypot(x - o.center[0], y - o.center[1]) >= r + o.radius + self.min_gap
                    for o in out
                ):
                    break
            out.append(Inclusion((x, y, self._depth(rng, cls, r)), r, _u(rng, self.inclusion_mua)))
        return out

    def render(self, inclusions: list[Inclusion], shape=None) -> torch.Tensor:
        p = Painter3D(self.mua_background, clip=None)
        for inc in inclusions:
            p.paint(inc.mua, lambda x, y, z, q=(inc.center, inc.radius): sphere(x, y, z, *q))
        return render_volume(p, self.domain, shape, self.render_factor)

    def sample(self, rng, cls=None, shape=None):
        return {"mu_a": self.render(self.draw(rng, cls), shape).float()}


__all__ = ["DOT3DScenes", "Inclusion"]
