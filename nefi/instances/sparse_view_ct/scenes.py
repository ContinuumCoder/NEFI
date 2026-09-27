"""Procedural CT phantoms (normalized coordinates, inside the field of view, values in [0, 1])."""

from __future__ import annotations

import math

import numpy as np

from ...bench.base import SceneGenerator
from ...domain import Domain
from ..deconvolution.phantoms import (
    Painter,
    ellipse,
    gaussian,
    point_max,
    rectangle,
    render,
    uniform,
)

#: modified Shepp-Logan phantom (Toft 1996): (intensity, a, b, x0, y0, phi_deg), additive.
SHEPP_LOGAN = (
    (1.0, 0.69, 0.92, 0.0, 0.0, 0.0),
    (-0.8, 0.6624, 0.8740, 0.0, -0.0184, 0.0),
    (-0.2, 0.1100, 0.3100, 0.22, 0.0, -18.0),
    (-0.2, 0.1600, 0.4100, -0.22, 0.0, 18.0),
    (0.1, 0.2100, 0.2500, 0.0, 0.35, 0.0),
    (0.1, 0.0460, 0.0460, 0.0, 0.1, 0.0),
    (0.1, 0.0460, 0.0460, 0.0, -0.1, 0.0),
    (0.1, 0.0460, 0.0230, -0.08, -0.605, 0.0),
    (0.1, 0.0230, 0.0230, 0.0, -0.606, 0.0),
    (0.1, 0.0230, 0.0460, 0.06, -0.605, 0.0),
)


class CTScenes(SceneGenerator):
    """Attenuation phantoms for sparse-view CT.

    * ``shepp`` — Shepp-Logan-like: the modified Shepp-Logan ellipse table with random global
      scale / rotation / shift and random per-ellipse axes, centers, angles and intensities;
    * ``blobs`` — smooth sums of Gaussian blobs;
    * ``piecewise`` — piecewise-constant random ellipses / rectangles painted inside a body
      ellipse (sharp edges, where TV's piecewise-constant prior pays off).

    Coordinates are normalized (``x`` = axis 0, ``y`` = axis 1, ``[-1, 1]``); every scene lies
    inside the inscribed circle (the scanner's field of view).
    """

    classes = ("shepp", "blobs", "piecewise")

    def __init__(self, domain: Domain, render_factor: int = 4) -> None:
        super().__init__(domain)
        self.render_factor = int(render_factor)

    def shepp(self, rng: np.random.Generator) -> Painter:
        s = uniform(rng, 0.85, 1.0)
        rot = math.radians(uniform(rng, -20.0, 20.0))
        sx, sy = uniform(rng, -0.05, 0.05), uniform(rng, -0.05, 0.05)
        outer = (uniform(rng, 0.95, 1.05), uniform(rng, 0.95, 1.05))
        cr, sr = math.cos(rot), math.sin(rot)
        p = Painter(0.0)
        for i, (val, a, b, x0, y0, phi) in enumerate(SHEPP_LOGAN):
            if i < 2:  # skull ring: shared perturbation keeps it closed
                a, b = a * outer[0], b * outer[1]
            else:
                a, b = a * uniform(rng, 0.85, 1.15), b * uniform(rng, 0.85, 1.15)
                x0, y0 = x0 + rng.normal(0.0, 0.02), y0 + rng.normal(0.0, 0.02)
                phi = phi + uniform(rng, -10.0, 10.0)
                val = val * uniform(rng, 0.7, 1.3)
            cx = s * (cr * x0 - sr * y0) + sx
            cy = s * (sr * x0 + cr * y0) + sy
            th = math.radians(phi) + rot
            p.add(val, lambda x, y, q=(cx, cy, s * a, s * b, th): ellipse(x, y, *q))
        return p

    def blobs(self, rng: np.random.Generator) -> Painter:
        p = Painter(0.0, clip=None)  # rescaled (not clipped) to a unit maximum in sample()
        for _ in range(int(rng.integers(3, 9))):
            r, ang = 0.55 * math.sqrt(rng.uniform()), rng.uniform(0.0, 2 * math.pi)
            cx, cy = r * math.cos(ang), r * math.sin(ang)
            sgx, sgy = uniform(rng, 0.05, 0.15), uniform(rng, 0.05, 0.15)
            th = uniform(rng, 0.0, math.pi)
            p.add(uniform(rng, 0.2, 0.8), lambda x, y, q=(cx, cy, sgx, sgy, th): gaussian(x, y, *q))
        return p

    def piecewise(self, rng: np.random.Generator) -> Painter:
        a, b = uniform(rng, 0.6, 0.85), uniform(rng, 0.6, 0.85)
        th = uniform(rng, 0.0, math.pi)
        p = Painter(0.0)
        p.paint(uniform(rng, 0.15, 0.3), lambda x, y, q=(0.0, 0.0, a, b, th): ellipse(x, y, *q))
        rmax = 0.5 * min(a, b)
        for _ in range(int(rng.integers(3, 8))):
            r, ang = rmax * math.sqrt(rng.uniform()), rng.uniform(0.0, 2 * math.pi)
            cx, cy = r * math.cos(ang), r * math.sin(ang)
            w, h = uniform(rng, 0.06, 0.3), uniform(rng, 0.06, 0.3)
            phi = uniform(rng, 0.0, math.pi)
            val = uniform(rng, 0.0, 1.0)
            if rng.uniform() < 0.5:
                p.paint(val, lambda x, y, q=(cx, cy, 0.5 * w, 0.5 * h, phi): ellipse(x, y, *q))
            else:
                p.paint(val, lambda x, y, q=(cx, cy, w, h, phi): rectangle(x, y, *q))
        return p

    def sample(self, rng, cls=None, shape=None):
        cls = self.check_class(cls)
        scene = getattr(self, cls)(rng)
        img = render(scene, self.domain, shape, self.render_factor, normalized=True)
        if cls == "blobs":  # grid-independent rescaling to a maximum of at most 1
            img = img / max(1.0, point_max(scene, self.domain, self.render_factor, True))
        return {"mu": img.clamp(0.0, 1.0).float()}


__all__ = ["SHEPP_LOGAN", "CTScenes"]
