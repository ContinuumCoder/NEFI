"""Generic N-D linear convolution via FFT with per-resolution kernel caches.

This is the spatial core of the NeTMY operators (``|G_az|² ∗ ρ``), of image deblurring, and of any
translation-invariant linear forward map. Kernels are given as a *function of the grid spacing*
so the operator can be rebuilt at every curriculum resolution (NeTMY App. D.3: "precomputed FFT
kernel cache" per stage).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from ..domain import Domain
from ..errors import ShapeError
from ..registry import register
from ..utils.tensor import center_crop, next_fast_len, resample, shape_tuple
from .base import Fields, Operator

KernelFn = Callable[[tuple[float, ...], tuple[int, ...], torch.device, torch.dtype], torch.Tensor]


def kernel_offsets(spacing: Sequence[float], shape: Sequence[int], device=None, dtype=None):
    """Physical offset grid ``(*shape, d)`` centered at the kernel center ``shape // 2``."""
    axes = []
    for s, n in zip(spacing, shape):
        i = torch.arange(n, device=device, dtype=dtype or torch.get_default_dtype())
        axes.append((i - n // 2) * s)
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


def fft_convolve(
    x: torch.Tensor,
    kernel: torch.Tensor,
    periodic: bool = False,
    kernel_fft: torch.Tensor | None = None,
    fft_shape: Sequence[int] | None = None,
):
    """Convolve the trailing ``kernel.ndim`` dims of ``x`` with ``kernel`` (center at ``shape//2``).

    Non-periodic mode performs a zero-padded *linear* convolution and returns the ``same``-size
    output; periodic mode wraps around. Leading dims of ``x`` are batch.
    Pass ``kernel_fft``/``fft_shape`` (from :func:`prepare_kernel`) to reuse a cached transform.
    """
    d = kernel.ndim
    n = tuple(x.shape[-d:])
    dims = tuple(range(-d, 0))
    if periodic:
        if kernel_fft is None:
            kernel_fft, fft_shape = prepare_kernel(kernel, n, periodic=True)
        X = torch.fft.rfftn(x, dim=dims)
        y = torch.fft.irfftn(X * kernel_fft, s=n, dim=dims)
        return y
    if kernel_fft is None:
        kernel_fft, fft_shape = prepare_kernel(kernel, n, periodic=False)
    X = torch.fft.rfftn(x, s=fft_shape, dim=dims)
    y_full = torch.fft.irfftn(X * kernel_fft, s=fft_shape, dim=dims)
    m = tuple(kernel.shape)
    offsets = tuple(mi // 2 for mi in m)
    return center_crop(y_full, n, offsets)


def prepare_kernel(kernel: torch.Tensor, n: Sequence[int], periodic: bool = False):
    """Precompute the kernel transform for inputs with trailing shape ``n``."""
    d = kernel.ndim
    n = shape_tuple(n)
    dims = tuple(range(-d, 0))
    if periodic:
        k = kernel
        if tuple(k.shape) != n:
            # crop or zero-pad symmetrically around the center, then roll center to index 0
            k = _fit_centered(k, n)
        shifts = tuple(-(ni // 2) for ni in n)
        k = torch.roll(k, shifts=shifts, dims=dims)
        return torch.fft.rfftn(k, dim=dims), n
    fft_shape = tuple(next_fast_len(ni + mi - 1) for ni, mi in zip(n, kernel.shape))
    return torch.fft.rfftn(kernel, s=fft_shape, dim=dims), fft_shape


def _fit_centered(k: torch.Tensor, n: tuple[int, ...]) -> torch.Tensor:
    out = torch.zeros(n, device=k.device, dtype=k.dtype)
    src, dst = [], []
    for ki, ni in zip(k.shape, n):
        kc, nc = ki // 2, ni // 2
        lo = min(kc, nc)
        hi = min(ki - kc, ni - nc)
        src.append(slice(kc - lo, kc + hi))
        dst.append(slice(nc - lo, nc + hi))
    out[tuple(dst)] = k[tuple(src)]
    return out


@register("operator", "fft_convolution")
class FFTConvolution(Operator):
    """Linear convolution ``y = k ∗ x`` on the domain grid, rebuilt at any resolution.

    Args:
        kernel: either a tensor sampled on the native grid (center at ``shape//2``) or a callable
            ``kernel_fn(spacing, shape, device, dtype) -> Tensor`` producing the kernel for a grid
            of
            physical cell size ``spacing`` (see :func:`kernel_offsets`).
        domain: the field domain (needed for spacing / resampling).
        field: name of the input field.
        periodic: wrap-around (circular) convolution instead of zero-padded linear convolution.
        kernel_extent: kernel size relative to the field grid when ``kernel`` is callable;
            ``"full"`` -> ``2n-1`` (exact linear support), ``"same"`` -> ``n``.
        post: optional pointwise map applied to the convolution output (e.g. ``torch.square`` for
        the
            NeTMY coherent operator F1). Set ``homogeneity`` accordingly.
        homogeneity: degree of homogeneity of the whole operator (1 without ``post``).

    Batchable: leading axes of the field are batch axes (:func:`fft_convolve`); ``post`` must be
    pointwise.
    """

    batchable = True

    def __init__(
        self,
        kernel: torch.Tensor | KernelFn,
        domain: Domain,
        field: str = "x",
        periodic: bool = False,
        kernel_extent: str = "full",
        post: Callable[[torch.Tensor], torch.Tensor] | None = None,
        homogeneity: float | None = 1.0,
    ) -> None:
        super().__init__()
        self.domain = domain
        self.primary = field
        self.periodic = periodic
        self.kernel_extent = kernel_extent
        self.post = post
        self.homogeneity = homogeneity
        self._kernel_fn: KernelFn | None = kernel if callable(kernel) else None
        if not callable(kernel):
            self.register_buffer(
                "_kernel_native", torch.as_tensor(kernel).clone(), persistent=False
            )
        self._cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]] = {}

    # --- kernels ------------------------------------------------------------------------
    def kernel(self, shape: Sequence[int] | None = None, device=None, dtype=None) -> torch.Tensor:
        """Kernel sampled for a field grid of ``shape`` (default the operator's domain shape)."""
        shape = self.domain.shape if shape is None else shape_tuple(shape)
        spacing = self.domain.spacing(shape)
        if self._kernel_fn is not None:
            if self.kernel_extent == "full":
                kshape = tuple(2 * n - 1 for n in shape)
            elif self.kernel_extent == "same":
                kshape = shape
            else:
                raise ShapeError(f"unknown kernel_extent {self.kernel_extent!r}")
            return self._kernel_fn(spacing, kshape, device, dtype or torch.get_default_dtype())
        k = self._kernel_native
        if tuple(k.shape) != shape:
            # resample the native kernel to the new spacing, conserving its integral
            k = resample(k, shape) * (k.numel() / float(torch.tensor(shape).prod()))
        return k.to(device=device, dtype=dtype)

    def _prepared(self, n: tuple[int, ...], device, dtype):
        key = (n, str(device), str(dtype))
        if key not in self._cache:
            k = self.kernel(n, device=device, dtype=dtype)
            kf, fs = prepare_kernel(k, n, periodic=self.periodic)
            self._cache[key] = (k, kf, fs)
        return self._cache[key]

    def clear_cache(self) -> None:
        self._cache.clear()

    # --- Operator API -------------------------------------------------------------------
    def forward(self, fields: Fields) -> torch.Tensor:
        x = self.get_field(fields)
        d = self.domain.ndim
        n = tuple(x.shape[-d:])
        k, kf, fs = self._prepared(n, x.device, x.dtype)
        y = fft_convolve(x, k, periodic=self.periodic, kernel_fft=kf, fft_shape=fs)
        return self.post(y) if self.post is not None else y

    def at_resolution(self, shape):
        shape = shape_tuple(shape)
        if shape == self.domain.shape:
            return self
        kernel = self._kernel_fn if self._kernel_fn is not None else self._kernel_native
        return FFTConvolution(
            kernel,
            self.domain.at(shape),
            self.primary,
            self.periodic,
            self.kernel_extent,
            self.post,
            self.homogeneity,
        )

    def output_shape(self, shape):
        return shape_tuple(shape)


def gaussian_kernel_fn(sigma: float | Sequence[float], normalize: bool = True) -> KernelFn:
    """Kernel function for an isotropic/anisotropic Gaussian blur with physical std ``sigma``."""

    def fn(spacing, shape, device, dtype):
        r = kernel_offsets(spacing, shape, device=device, dtype=dtype)
        s = torch.as_tensor(sigma, device=device, dtype=dtype)
        s = s.expand(len(shape)) if s.ndim == 0 else s
        k = torch.exp(-0.5 * ((r / s) ** 2).sum(-1))
        if normalize:
            k = k / k.sum()
        return k

    return fn
