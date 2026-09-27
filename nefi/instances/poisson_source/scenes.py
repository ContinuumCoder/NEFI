"""Non-negative source fields for the Poisson source problem (cell-averaged, grid-consistent)."""

from __future__ import annotations

import numpy as np

from ...bench.base import SceneGenerator
from ...domain import Domain
from ..deconvolution.phantoms import Painter, gaussian, point_max, render, uniform


class SourceScenes(SceneGenerator):
    """Source classes (field ``"f"``, maximum ≈ 1):

    * ``blobs`` — 2-4 compact Gaussian blobs;
    * ``points`` — 3-6 near-point sources (width ``point_cells`` native cells);
    * ``smooth`` — 2-3 broad overlapping blobs.

    Sources sit in the interior ``[0.2, 0.8]²`` of the domain (relative coordinates).
    """

    classes = ("blobs", "points", "smooth")

    def __init__(self, domain: Domain, point_cells: float = 0.8, render_factor: int = 4) -> None:
        super().__init__(domain)
        self.point_cells = float(point_cells)
        self.render_factor = int(render_factor)

    def _pos(self, rng: np.random.Generator, lo: float = 0.2, hi: float = 0.8):
        (x0, x1), (y0, y1) = self.domain.extent
        return (
            x0 + uniform(rng, lo, hi) * (x1 - x0),
            y0 + uniform(rng, lo, hi) * (y1 - y0),
        )

    def blobs(self, rng: np.random.Generator) -> Painter:
        s = min(self.domain.size)
        p = Painter(0.0, clip=None)
        for _ in range(int(rng.integers(2, 5))):
            cx, cy = self._pos(rng)
            sg = uniform(rng, 0.03, 0.08) * s
            p.add(uniform(rng, 0.5, 1.0), lambda x, y, q=(cx, cy, sg): gaussian(x, y, *q))
        return p

    def points(self, rng: np.random.Generator) -> Painter:
        p = Painter(0.0, clip=None)
        sg = self.point_cells * min(self.domain.spacing())
        for _ in range(int(rng.integers(3, 7))):
            cx, cy = self._pos(rng)
            p.add(uniform(rng, 0.5, 1.0), lambda x, y, q=(cx, cy, sg): gaussian(x, y, *q))
        return p

    def smooth(self, rng: np.random.Generator) -> Painter:
        s = min(self.domain.size)
        p = Painter(0.0, clip=None)
        for _ in range(int(rng.integers(2, 4))):
            cx, cy = self._pos(rng, 0.3, 0.7)
            sx, sy = uniform(rng, 0.1, 0.2) * s, uniform(rng, 0.1, 0.2) * s
            th = uniform(rng, 0.0, np.pi)
            p.add(uniform(rng, 0.3, 1.0), lambda x, y, q=(cx, cy, sx, sy, th): gaussian(x, y, *q))
        return p

    def sample(self, rng, cls=None, shape=None):
        cls = self.check_class(cls)
        scene = getattr(self, cls)(rng)
        img = render(scene, self.domain, shape, self.render_factor)
        img = img / max(1.0, point_max(scene, self.domain, self.render_factor))
        return {"f": img.float()}


__all__ = ["SourceScenes"]
