"""Fluorescence-microscopy ground-truth volumes (physical µm coordinates, intensities ~[0, 1]).

Scenes are analytic: random parameters are drawn first, then the volume is rendered as cell
averages (:func:`~nefi.instances._volumetric.render_volume` for tubes / shells / fills, exact erf
cell averages for sub-voxel point sources, :func:`~nefi.instances._volumetric.
gaussian_cell_average`), so the native ground truth and the 2× data-generation grid agree exactly.
"""

from __future__ import annotations

import numpy as np
import torch

from ...bench.base import SceneGenerator
from ...domain import Domain
from .._volumetric import (
    Painter3D,
    ellipsoid_level,
    gaussian_cell_average,
    random_curve,
    random_unit_vector,
    render_volume,
    rotation_zxz,
    smooth_indicator,
    tube,
)


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


class Deconvolution3DScenes(SceneGenerator):
    """Z-stack scene classes (fluorophore density, zero-ish background):

    * ``filaments`` — 3–6 smooth random polylines (persistent random walks, mostly in-plane like
      cytoskeletal fibres) with a Gaussian cross-section of ``filament_sigma`` µm;
    * ``puncta`` — sparse sub-voxel bright spots (vesicles, single molecules): exact cell averages
      of Gaussians of ``puncta_sigma`` µm — the regime where ℓ1 / localized priors shine;
    * ``cells`` — 1–2 ellipsoidal cells with a bright membrane shell, dim cytoplasm, a nucleus
      and a few vesicles.

    Args:
        domain: native domain (physical µm extents).
        render_factor: sub-samples per voxel and axis of the generic renderer (even values keep
            the native and 2× grids on the same sub-samples).
        filament_sigma: ``(lo, hi)`` Gaussian cross-section of the filaments (µm).
        puncta_sigma: ``(lo, hi)`` width of the point sources (µm).
        membrane_width: membrane shell width (µm).
    """

    classes = ("filaments", "puncta", "cells")

    def __init__(
        self,
        domain: Domain,
        render_factor: int = 2,
        filament_sigma: tuple[float, float] = (0.06, 0.09),
        puncta_sigma: tuple[float, float] = (0.04, 0.07),
        membrane_width: float = 0.06,
    ) -> None:
        super().__init__(domain)
        self.render_factor = int(render_factor)
        self.filament_sigma = tuple(float(v) for v in filament_sigma)
        self.puncta_sigma = tuple(float(v) for v in puncta_sigma)
        self.membrane_width = float(membrane_width)

    # ---- geometry helpers ----------------------------------------------------------------
    def _box(self, margin: float) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([e[0] for e in self.domain.extent], dtype=np.float64)
        hi = np.array([e[1] for e in self.domain.extent], dtype=np.float64)
        pad = margin * (hi - lo)
        return lo + pad, hi - pad

    def _point(self, rng: np.random.Generator, margin: float = 0.1) -> np.ndarray:
        lo, hi = self._box(margin)
        return rng.uniform(lo, hi)

    # ---- classes ---------------------------------------------------------------------------
    def filaments(self, rng: np.random.Generator):
        lo, hi = self._box(0.05)
        width = float(min(self.domain.size[:2]))
        p = Painter3D(background=_u(rng, 0.0, 0.02), clip=(0.0, None))
        for _ in range(int(rng.integers(3, 7))):
            d = random_unit_vector(rng)
            d[2] *= 0.4  # fibres mostly run in-plane (thin adherent cells)
            verts = random_curve(
                rng,
                self._point(rng, 0.15),
                d,
                n_steps=int(rng.integers(10, 18)),
                step=0.09 * width,
                persistence=0.85,
                lo=lo,
                hi=hi,
            )
            sig = _u(rng, *self.filament_sigma)
            p.add(_u(rng, 0.5, 1.0), lambda x, y, z, q=(verts, sig): tube(x, y, z, *q, "gaussian"))
        return p, None

    def puncta(self, rng: np.random.Generator):
        k = int(rng.integers(12, 30))
        centers = np.stack([self._point(rng, 0.06) for _ in range(k)])
        sig = rng.uniform(*self.puncta_sigma, size=k)
        amp = rng.uniform(0.8, 1.6, size=k)
        bg = _u(rng, 0.0, 0.02)
        return Painter3D(background=bg, clip=None), (centers, sig, amp)

    def cells(self, rng: np.random.Generator):
        lx, ly, lz = self.domain.size
        p = Painter3D(background=_u(rng, 0.0, 0.02), clip=(0.0, None))
        spots: list[tuple[np.ndarray, float, float]] = []
        w = self.membrane_width
        for _ in range(int(rng.integers(1, 3))):
            axes = (_u(rng, 0.22, 0.34) * lx, _u(rng, 0.22, 0.34) * ly, _u(rng, 0.25, 0.4) * lz)
            center = self._point(rng, 0.3)
            rot = rotation_zxz(_u(rng, 0.0, 180.0), _u(rng, -10.0, 10.0), 0.0)
            r_mean = float(np.mean(axes))

            def signed(x, y, z, c=center, a=axes, r=rot, rm=r_mean):
                q = ellipsoid_level(x, y, z, c, a, r)
                return (torch.sqrt(q) - 1.0) * rm  # ≈ signed distance to the surface

            p.add(_u(rng, 0.08, 0.15), lambda x, y, z, s=signed: smooth_indicator(s(x, y, z), 0.03))
            p.add(
                _u(rng, 0.6, 1.0),
                lambda x, y, z, s=signed: torch.exp(-0.5 * (s(x, y, z) / w) ** 2),
            )
            nuc_axes = tuple(v * _u(rng, 0.35, 0.5) for v in axes)
            nuc_c = center + rng.normal(0.0, 0.08, size=3) * np.array(axes)

            def nucleus(x, y, z, c=nuc_c, a=nuc_axes, r=rot, rm=float(np.mean(nuc_axes))):
                q = ellipsoid_level(x, y, z, c, a, r)
                return smooth_indicator((torch.sqrt(q) - 1.0) * rm, 0.04)

            p.add(_u(rng, 0.15, 0.25), nucleus)
            for _ in range(int(rng.integers(2, 6))):  # vesicles inside the cytoplasm
                dirn = random_unit_vector(rng)
                rr = _u(rng, 0.55, 0.8)
                pos = center + (rot.numpy() @ (dirn * np.array(axes) * rr))
                spots.append((pos, _u(rng, *self.puncta_sigma), _u(rng, 0.6, 1.2)))
        if spots:
            centers = np.stack([s[0] for s in spots])
            return p, (centers, np.array([s[1] for s in spots]), np.array([s[2] for s in spots]))
        return p, None

    # ---- sampling --------------------------------------------------------------------------
    def sample(self, rng, cls=None, shape=None):
        cls = self.check_class(cls)
        painter, points = getattr(self, cls)(rng)
        vol = render_volume(painter, self.domain, shape, self.render_factor)
        if points is not None:
            centers, sig, amp = points
            vol = vol + gaussian_cell_average(self.domain, shape, centers, sig, amp)
        return {"x": vol.clamp_min(0.0).float()}


__all__ = ["Deconvolution3DScenes"]
