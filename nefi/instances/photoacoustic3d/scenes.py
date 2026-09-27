"""Initial-pressure (absorbed-energy) phantoms for 3-D photoacoustic tomography (physical mm).

Scenes are analytic: parameters are drawn first, then the volume is rendered as cell averages
(:func:`~nefi.instances._volumetric.render_volume`) with smooth (``tanh``) edges of physical width
``edge_width``, so the native ground truth and the 2× data-generation grid agree exactly and the
initial pressure is band-limited enough for the leapfrog wave solver.
"""

from __future__ import annotations

import numpy as np

from ...bench.base import SceneGenerator
from ...domain import Domain
from .._volumetric import (
    Painter3D,
    random_curve,
    random_unit_vector,
    render_volume,
    smooth_indicator,
    tube,
)


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


class PAT3DScenes(SceneGenerator):
    """Initial pressure ``p0(x, y, z) ≥ 0`` (arbitrary units, ≲ 1; depth = last axis).

    * ``vessels`` — 2–4 smooth random vessel trees (persistent random walks, mostly running
      parallel to the surface, one branch each) with radii ``vessel_radius``: the canonical
      photoacoustic angiography target, and a hard limited-view case (vertical segments are
      invisible to a planar array);
    * ``spheres`` — 3–6 balls of radii ``sphere_radius`` with different amplitudes (absorbers /
      tumour-like blobs).

    Args:
        domain: native domain (mm).
        vessel_radius / sphere_radius: ``(lo, hi)`` radii (mm).
        edge_width: tanh edge width (mm) of every structure (band-limits ``p0``).
        depth_range: ``(lo, hi)`` fraction of the slab depth where structures are placed.
        render_factor: sub-samples per voxel and axis of the cell-averaged rendering.
    """

    classes = ("vessels", "spheres")

    def __init__(
        self,
        domain: Domain,
        vessel_radius: tuple[float, float] = (0.3, 0.6),
        sphere_radius: tuple[float, float] = (0.6, 1.5),
        edge_width: float = 0.15,
        depth_range: tuple[float, float] = (0.15, 0.8),
        render_factor: int = 2,
    ) -> None:
        super().__init__(domain)
        self.vessel_radius = tuple(float(v) for v in vessel_radius)
        self.sphere_radius = tuple(float(v) for v in sphere_radius)
        self.edge_width = float(edge_width)
        self.depth_range = tuple(float(v) for v in depth_range)
        self.render_factor = int(render_factor)

    def _bounds(self, margin: float) -> tuple[np.ndarray, np.ndarray]:
        (x0, x1), (y0, y1), (z0, z1) = self.domain.extent
        lo = np.array([x0 + margin * (x1 - x0), y0 + margin * (y1 - y0), z0], dtype=np.float64)
        hi = np.array([x1 - margin * (x1 - x0), y1 - margin * (y1 - y0), z1], dtype=np.float64)
        lo[2] = z0 + self.depth_range[0] * (z1 - z0)
        hi[2] = z0 + self.depth_range[1] * (z1 - z0)
        return lo, hi

    def vessels(self, rng: np.random.Generator) -> Painter3D:
        lo, hi = self._bounds(0.1)
        width = float(min(self.domain.size[:2]))
        p = Painter3D(0.0, clip=(0.0, 1.0))
        w = self.edge_width
        for _ in range(int(rng.integers(2, 5))):
            amp = _u(rng, 0.6, 1.0)
            d = random_unit_vector(rng)
            d[2] *= 0.3  # vessels mostly run parallel to the skin
            start = rng.uniform(lo, hi)
            trunk = random_curve(
                rng, start, d, int(rng.integers(8, 14)), 0.08 * width, 0.85, lo, hi
            )
            r = _u(rng, *self.vessel_radius)
            p.add(amp, lambda x, y, z, q=(trunk, r): tube(x, y, z, *q, "smooth", w))
            k = int(rng.integers(2, trunk.shape[0] - 2))
            db = random_unit_vector(rng)
            db[2] *= 0.5
            branch = random_curve(
                rng, trunk[k].numpy(), db, int(rng.integers(4, 8)), 0.07 * width, 0.8, lo, hi
            )
            rb = max(self.vessel_radius[0], 0.7 * r)
            p.add(amp, lambda x, y, z, q=(branch, rb): tube(x, y, z, *q, "smooth", w))
        return p

    def spheres(self, rng: np.random.Generator) -> Painter3D:
        lo, hi = self._bounds(0.15)
        p = Painter3D(0.0, clip=(0.0, 1.0))
        w = self.edge_width
        for _ in range(int(rng.integers(3, 7))):
            c = rng.uniform(lo, hi)
            r = _u(rng, *self.sphere_radius)

            def ball(x, y, z, c=c, r=r):
                d = ((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2).sqrt()
                return smooth_indicator(d - r, w)

            p.add(_u(rng, 0.4, 1.0), ball)
        return p

    def sample(self, rng, cls=None, shape=None):
        cls = self.check_class(cls)
        scene = getattr(self, cls)(rng)
        return {"p0": render_volume(scene, self.domain, shape, self.render_factor).float()}


__all__ = ["PAT3DScenes"]
