"""Procedural 3-D CT phantoms (normalized coordinates, inside the scanner cylinder, in [0, 1]).

All scenes are analytic: random parameters are drawn first (independently of the grid), then the
continuous volume is rendered as cell averages on any grid (:func:`~nefi.instances._volumetric.
render_volume`), so the native ground truth and the 2× data-generation grid agree exactly.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ...bench.base import SceneGenerator
from ...domain import Domain
from .._volumetric import Painter3D, box, cylinder, ellipsoid, gaussian_blob, render_volume
from .._volumetric import rotation_zxz as rot

#: 3-D modified Shepp–Logan phantom (Kak & Slaney 1988 3-D extension with the Toft 1996 contrast,
#: as in Schabel's ``phantom3d``): ``(A, a, b, c, x0, y0, z0, φ, θ, ψ)``, additive intensities,
#: semi-axes ``(a, b, c)`` along ``(x, y, z)``, ZXZ Euler angles in degrees.
SHEPP_LOGAN_3D = (
    (1.0, 0.6900, 0.920, 0.810, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (-0.8, 0.6624, 0.874, 0.780, 0.0, -0.0184, 0.0, 0.0, 0.0, 0.0),
    (-0.2, 0.1100, 0.310, 0.220, 0.22, 0.0, 0.0, -18.0, 0.0, 10.0),
    (-0.2, 0.1600, 0.410, 0.280, -0.22, 0.0, 0.0, 18.0, 0.0, 10.0),
    (0.1, 0.2100, 0.250, 0.410, 0.0, 0.35, -0.15, 0.0, 0.0, 0.0),
    (0.1, 0.0460, 0.046, 0.050, 0.0, 0.1, 0.25, 0.0, 0.0, 0.0),
    (0.1, 0.0460, 0.046, 0.050, 0.0, -0.1, 0.25, 0.0, 0.0, 0.0),
    (0.1, 0.0460, 0.023, 0.050, -0.08, -0.605, 0.0, 0.0, 0.0, 0.0),
    (0.1, 0.0230, 0.023, 0.020, 0.0, -0.606, 0.0, 0.0, 0.0, 0.0),
    (0.1, 0.0230, 0.046, 0.020, 0.06, -0.605, 0.0, 0.0, 0.0, 0.0),
)


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


class CT3DScenes(SceneGenerator):
    """Attenuation volumes for 3-D parallel-beam CT (rotation axis ``z``).

    * ``ellipsoids`` — 3-D Shepp–Logan-like head: the :data:`SHEPP_LOGAN_3D` table with a random
      global scale / rotation about ``z`` / shift and random per-ellipsoid axes, centers, Euler
      angles and intensities (the skull shell shares one perturbation so it stays closed);
    * ``blobs`` — smooth sums of anisotropic Gaussian blobs (clipped to ``[0, 1]`` pointwise);
    * ``piecewise`` — random constant ellipsoids / boxes / cylinders (bright inserts and dark
      cavities) painted inside a low-attenuation body ellipsoid: sharp 3-D edges, where TV pays off.

    Coordinates are normalized (``[-1, 1]³``); every scene lies inside the inscribed cylinder
    ``x² + y² ≤ 1`` (the field of view of every slice). Feature values are chosen so that the
    ``iou_tau = 0.25`` super-level set is the "feature" mask (skull + bright lesions, blob cores,
    bright inserts) and the background tissue stays below it.
    """

    classes = ("ellipsoids", "blobs", "piecewise")

    def __init__(self, domain: Domain, render_factor: int = 2) -> None:
        super().__init__(domain)
        self.render_factor = int(render_factor)

    def ellipsoids(self, rng: np.random.Generator) -> Painter3D:
        s = _u(rng, 0.85, 0.98)
        spin = _u(rng, -20.0, 20.0)
        shift = (_u(rng, -0.04, 0.04), _u(rng, -0.04, 0.04), _u(rng, -0.04, 0.04))
        outer = (_u(rng, 0.95, 1.02), _u(rng, 0.95, 1.02), _u(rng, 0.95, 1.02))
        g = rot(spin, 0.0, 0.0)
        p = Painter3D(0.0)
        for i, (val, a, b, c, x0, y0, z0, phi, theta, psi) in enumerate(SHEPP_LOGAN_3D):
            if i < 2:  # skull shell: shared perturbation keeps it closed
                a, b, c = a * outer[0], b * outer[1], c * outer[2]
            else:
                a, b, c = (v * _u(rng, 0.85, 1.15) for v in (a, b, c))
                x0, y0, z0 = (v + float(rng.normal(0.0, 0.02)) for v in (x0, y0, z0))
                phi, theta, psi = (v + _u(rng, -10.0, 10.0) for v in (phi, theta, psi))
                val = val * _u(rng, 0.7, 1.3)
            ctr = g @ torch.tensor([x0, y0, z0], dtype=torch.float64) * s
            center = tuple(float(ctr[k]) + shift[k] for k in range(3))
            r = g @ rot(phi, theta, psi)
            p.add(val, lambda x, y, z, q=(center, (s * a, s * b, s * c), r): ellipsoid(x, y, z, *q))
        return p

    def blobs(self, rng: np.random.Generator) -> Painter3D:
        p = Painter3D(0.0)
        for _ in range(int(rng.integers(4, 10))):
            r, ang = 0.5 * math.sqrt(rng.uniform()), rng.uniform(0.0, 2 * math.pi)
            center = (r * math.cos(ang), r * math.sin(ang), _u(rng, -0.5, 0.5))
            sig = tuple(_u(rng, 0.07, 0.16) for _ in range(3))
            r3 = rot(_u(rng, 0, 180), _u(rng, 0, 180), _u(rng, 0, 180))
            p.add(
                _u(rng, 0.3, 0.8),
                lambda x, y, z, q=(center, sig, r3): gaussian_blob(x, y, z, *q),
            )
        # hard field-of-view window: nothing outside the scanner cylinder
        p.paint(0.0, lambda x, y, z: (x**2 + y**2 > 0.95**2).to(x.dtype))
        return p

    def piecewise(self, rng: np.random.Generator) -> Painter3D:
        a, b, c = _u(rng, 0.6, 0.85), _u(rng, 0.6, 0.85), _u(rng, 0.6, 0.85)
        body = rot(_u(rng, 0.0, 180.0), 0.0, 0.0)
        p = Painter3D(0.0)
        p.paint(
            _u(rng, 0.1, 0.2),
            lambda x, y, z, q=((0.0, 0.0, 0.0), (a, b, c), body): ellipsoid(x, y, z, *q),
        )
        rmax = 0.5 * min(a, b)
        for _ in range(int(rng.integers(4, 9))):
            r, ang = rmax * math.sqrt(rng.uniform()), rng.uniform(0.0, 2 * math.pi)
            center = (r * math.cos(ang), r * math.sin(ang), _u(rng, -0.5, 0.5) * c)
            r3 = rot(_u(rng, 0, 180), _u(rng, 0, 180), _u(rng, 0, 180))
            val = _u(rng, 0.35, 1.0) if rng.uniform() < 0.75 else _u(rng, 0.0, 0.05)
            kind = int(rng.integers(0, 3))
            if kind == 0:
                axes = tuple(_u(rng, 0.05, 0.18) for _ in range(3))
                p.paint(val, lambda x, y, z, q=(center, axes, r3): ellipsoid(x, y, z, *q))
            elif kind == 1:
                size = tuple(_u(rng, 0.1, 0.3) for _ in range(3))
                p.paint(val, lambda x, y, z, q=(center, size, r3): box(x, y, z, *q))
            else:
                rad, half = _u(rng, 0.04, 0.1), _u(rng, 0.08, 0.2)
                p.paint(val, lambda x, y, z, q=(center, rad, half, r3): cylinder(x, y, z, *q))
        return p

    def sample(self, rng, cls=None, shape=None):
        cls = self.check_class(cls)
        scene = getattr(self, cls)(rng)
        vol = render_volume(scene, self.domain, shape, self.render_factor, normalized=True)
        return {"mu": vol.clamp(0.0, 1.0).float()}


__all__ = ["SHEPP_LOGAN_3D", "CT3DScenes"]
