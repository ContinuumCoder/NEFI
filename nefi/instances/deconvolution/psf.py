"""Point-spread functions as *kernel functions of the grid spacing* (rebuilt at every resolution).

Each factory returns ``kernel_fn(spacing, shape, device, dtype) -> Tensor`` (center at
``shape // 2``, unit sum) for :class:`~nefi.operators.FFTConvolution`, so the same physical PSF is
sampled consistently on coarse curriculum grids, the native grid and the 2× data-generation grid.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ...errors import ConfigError
from ...operators.conv import KernelFn, gaussian_kernel_fn, kernel_offsets


def _delta(shape, device, dtype) -> torch.Tensor:
    k = torch.zeros(shape, device=device, dtype=dtype)
    k[tuple(s // 2 for s in shape)] = 1.0
    return k


def gaussian_psf(sigma: float | Sequence[float]) -> KernelFn:
    """Gaussian blur with physical standard deviation ``sigma`` (isotropic or per axis)."""
    return gaussian_kernel_fn(sigma, normalize=True)


def motion_psf(length: float, angle_deg: float = 0.0, samples_per_cell: int = 8) -> KernelFn:
    """Linear motion blur: a uniform line segment of physical ``length`` at ``angle_deg``.

    The angle is measured from axis 0 towards axis 1. The segment is sampled densely
    (``samples_per_cell`` points per cell) and splatted bilinearly onto the kernel grid, which
    anti-aliases it at every resolution.
    """
    a = math.radians(float(angle_deg))

    def fn(spacing, shape, device, dtype):
        if length <= 0:
            return _delta(shape, device, dtype)
        h = min(spacing)
        m = max(2, int(math.ceil(length / h * samples_per_cell)) + 1)
        t = torch.linspace(-0.5 * length, 0.5 * length, m, dtype=torch.float64)
        p0 = t * math.cos(a) / spacing[0] + shape[0] // 2
        p1 = t * math.sin(a) / spacing[1] + shape[1] // 2
        k = torch.zeros(shape, dtype=torch.float64)
        i0, i1 = torch.floor(p0), torch.floor(p1)
        f0, f1 = p0 - i0, p1 - i1
        for di, w0 in ((0, 1.0 - f0), (1, f0)):
            for dj, w1 in ((0, 1.0 - f1), (1, f1)):
                ii, jj = (i0 + di).long(), (i1 + dj).long()
                ok = (ii >= 0) & (ii < shape[0]) & (jj >= 0) & (jj < shape[1])
                k.index_put_((ii[ok], jj[ok]), (w0 * w1)[ok], accumulate=True)
        k = k / k.sum()
        return k.to(device=device, dtype=dtype)

    return fn


def disk_psf(radius: float, supersample: int = 8) -> KernelFn:
    """Uniform out-of-focus (pillbox) blur of physical ``radius``, anti-aliased per pixel."""

    def fn(spacing, shape, device, dtype):
        r = kernel_offsets(spacing, shape, dtype=torch.float64)
        ss = int(supersample)
        sub = (torch.arange(ss, dtype=torch.float64) + 0.5) / ss - 0.5
        inside = torch.zeros(shape, dtype=torch.float64)
        for a in sub:
            for b in sub:
                x = r[..., 0] + a * spacing[0]
                y = r[..., 1] + b * spacing[1]
                inside += (x**2 + y**2 <= radius**2).to(torch.float64)
        if float(inside.sum()) == 0.0:
            return _delta(shape, device, dtype)
        k = inside / inside.sum()
        return k.to(device=device, dtype=dtype)

    return fn


def make_psf(
    kind: str,
    sigma: float = 0.02,
    motion_length: float = 0.1,
    motion_angle: float = 30.0,
    disk_radius: float = 0.04,
) -> KernelFn:
    """PSF kernel function by name: ``"gaussian"`` | ``"motion"`` | ``"disk"``."""
    if kind == "gaussian":
        return gaussian_psf(sigma)
    if kind == "motion":
        return motion_psf(motion_length, motion_angle)
    if kind == "disk":
        return disk_psf(disk_radius)
    raise ConfigError(f"unknown psf {kind!r}; use 'gaussian', 'motion' or 'disk'")


__all__ = ["disk_psf", "gaussian_psf", "make_psf", "motion_psf"]
