"""Tensor helpers: resampling across resolutions, cropping, FFT sizes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fft import next_fast_len as _next_fast_len

from ..errors import ShapeError


def shape_tuple(shape: Sequence[int] | int) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (int(shape),)
    return tuple(int(s) for s in shape)


def next_fast_len(n: int) -> int:
    """Smallest FFT-friendly length >= n (real FFT)."""
    return int(_next_fast_len(int(n), real=True))


def as_tensor(x, device=None, dtype=None) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x)
    return t.to(device=device, dtype=dtype) if (device is not None or dtype is not None) else t


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def resample(x: torch.Tensor, shape: Sequence[int], mode: str = "auto") -> torch.Tensor:
    """Resample the trailing ``len(shape)`` dims of ``x`` to ``shape`` (differentiable).

    Leading dims are treated as batch. Downsampling uses area averaging, upsampling uses
    (bi/tri)linear interpolation with ``align_corners=False`` (cell-centered convention, matching
    :meth:`nefi.domain.Domain.coords`). Supports 1, 2 and 3 spatial dims.

    Args:
        x: tensor of shape ``(*batch, *spatial)``.
        shape: target spatial shape.
        mode: ``"auto"`` | ``"area"`` | ``"linear"`` | ``"nearest"``.
    """
    shape = shape_tuple(shape)
    d = len(shape)
    if d not in (1, 2, 3):
        raise ShapeError(f"resample supports 1-3 spatial dims, got target shape {shape}")
    if x.ndim < d:
        raise ShapeError(f"tensor with shape {tuple(x.shape)} has fewer dims than target {shape}")
    old = tuple(x.shape[-d:])
    if old == shape:
        return x
    batch = tuple(x.shape[:-d])
    xb = x.reshape(-1, 1, *old)
    if mode == "auto":
        mode = "area" if all(o >= n for o, n in zip(old, shape)) else "linear"
    if mode == "linear":
        interp = {1: "linear", 2: "bilinear", 3: "trilinear"}[d]
        y = F.interpolate(xb, size=shape, mode=interp, align_corners=False)
    elif mode == "area":
        y = F.interpolate(xb, size=shape, mode="area")
    elif mode == "nearest":
        y = F.interpolate(xb, size=shape, mode="nearest")
    else:
        raise ValueError(f"unknown resample mode {mode!r}")
    return y.reshape(*batch, *shape)


def center_crop(x: torch.Tensor, shape: Sequence[int], offsets: Sequence[int] | None = None):
    """Crop the trailing dims of ``x`` to ``shape`` starting at ``offsets`` (default: centered)."""
    shape = shape_tuple(shape)
    d = len(shape)
    old = tuple(x.shape[-d:])
    if offsets is None:
        offsets = tuple((o - n) // 2 for o, n in zip(old, shape))
    slices = [slice(None)] * (x.ndim - d) + [slice(o, o + n) for o, n in zip(offsets, shape)]
    return x[tuple(slices)]


def pad_trailing(x: torch.Tensor, shape: Sequence[int], value: float = 0.0) -> torch.Tensor:
    """Zero-pad the trailing dims of ``x`` (at the end of each axis) up to ``shape``."""
    shape = shape_tuple(shape)
    d = len(shape)
    pads: list[int] = []
    for i in range(d - 1, -1, -1):
        pads += [0, shape[i] - x.shape[-d + i]]
    if any(p < 0 for p in pads):
        raise ShapeError(f"cannot pad {tuple(x.shape)} to smaller shape {shape}")
    return F.pad(x, pads, value=value)
