"""Ground-truth scene classes for image deblurring (cell-averaged, resolution-consistent)."""

from __future__ import annotations

import math

import numpy as np
import torch

from ...bench.base import SceneGenerator
from ...domain import Domain
from .phantoms import Painter, disk, gaussian, point_max, rectangle, render, uniform


class DeconvolutionScenes(SceneGenerator):
    """Scene classes (values in ``[0, 1]``, zero-ish background):

    * ``phantom`` — procedural piecewise-constant disks and (rotated) rectangles with varying
      contrast, soft Gaussian blobs and a few sparse dots;
    * ``smooth`` — broad blobs plus a low-frequency cosine texture;
    * ``sparse_dots`` — isolated near-point sources (width ``dot_cells`` native cells) on a zero
      background: the regime where localized priors (Gaussian splats, ℓ1) shine.

    Args:
        domain: native domain.
        dot_cells: dot standard deviation in native cells.
        render_factor: sub-grid factor of the cell-averaged rendering.
    """

    classes = ("phantom", "smooth", "sparse_dots")

    def __init__(self, domain: Domain, dot_cells: float = 0.7, render_factor: int = 4) -> None:
        super().__init__(domain)
        self.dot_cells = float(dot_cells)
        self.render_factor = int(render_factor)

    def _lo_hi(self) -> tuple[float, float, float, float]:
        (x0, x1), (y0, y1) = self.domain.extent
        return x0, x1, y0, y1

    def _pos(self, rng: np.random.Generator, margin: float = 0.12) -> tuple[float, float]:
        x0, x1, y0, y1 = self._lo_hi()
        wx, wy = x1 - x0, y1 - y0
        return (
            uniform(rng, x0 + margin * wx, x1 - margin * wx),
            uniform(rng, y0 + margin * wy, y1 - margin * wy),
        )

    def _dot_sigma(self) -> float:
        return self.dot_cells * min(self.domain.spacing())

    def phantom(self, rng: np.random.Generator) -> Painter:
        s = min(self.domain.size)
        p = Painter(background=uniform(rng, 0.0, 0.08))
        for _ in range(int(rng.integers(2, 5))):
            cx, cy = self._pos(rng, 0.15)
            r = uniform(rng, 0.05, 0.15) * s
            p.paint(uniform(rng, 0.3, 0.9), lambda x, y, a=(cx, cy, r): disk(x, y, *a))
        for _ in range(int(rng.integers(1, 4))):
            cx, cy = self._pos(rng, 0.15)
            w, h = uniform(rng, 0.08, 0.3) * s, uniform(rng, 0.06, 0.25) * s
            th = uniform(rng, 0.0, math.pi)
            p.paint(uniform(rng, 0.2, 1.0), lambda x, y, a=(cx, cy, w, h, th): rectangle(x, y, *a))
        for _ in range(int(rng.integers(1, 3))):
            cx, cy = self._pos(rng)
            sg = uniform(rng, 0.03, 0.08) * s
            p.add(uniform(rng, 0.15, 0.4), lambda x, y, a=(cx, cy, sg): gaussian(x, y, *a))
        ds = self._dot_sigma()
        for _ in range(int(rng.integers(2, 6))):
            cx, cy = self._pos(rng, 0.08)
            p.add(uniform(rng, 0.5, 1.0), lambda x, y, a=(cx, cy, ds): gaussian(x, y, *a))
        return p

    def smooth(self, rng: np.random.Generator) -> Painter:
        s = min(self.domain.size)
        p = Painter(background=uniform(rng, 0.0, 0.1), clip=None)  # rescaled in sample()
        for _ in range(int(rng.integers(3, 7))):
            cx, cy = self._pos(rng, 0.1)
            sx, sy = uniform(rng, 0.08, 0.2) * s, uniform(rng, 0.08, 0.2) * s
            th = uniform(rng, 0.0, math.pi)
            p.add(uniform(rng, 0.2, 0.6), lambda x, y, a=(cx, cy, sx, sy, th): gaussian(x, y, *a))
        k = rng.uniform(-2.0, 2.0, size=2) * 2.0 * math.pi / s
        ph = uniform(rng, 0.0, 2 * math.pi)
        amp = uniform(rng, 0.05, 0.2)
        p.add(amp, lambda x, y: 0.5 * (1.0 + torch.cos(k[0] * x + k[1] * y + ph)))
        return p

    def sparse_dots(self, rng: np.random.Generator) -> Painter:
        p = Painter(background=0.0)
        ds = self._dot_sigma()
        for _ in range(int(rng.integers(6, 16))):
            cx, cy = self._pos(rng, 0.1)
            p.add(uniform(rng, 0.4, 1.0), lambda x, y, a=(cx, cy, ds): gaussian(x, y, *a))
        return p

    def sample(self, rng, cls=None, shape=None):
        cls = self.check_class(cls)
        scene = getattr(self, cls)(rng)
        img = render(scene, self.domain, shape, self.render_factor)
        if cls == "smooth":  # grid-independent rescaling to a 0.9 maximum
            img = 0.9 * img / max(point_max(scene, self.domain, self.render_factor), 1e-12)
        return {"x": img.clamp(0.0, 1.0).float()}


__all__ = ["DeconvolutionScenes"]
