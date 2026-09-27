"""Classical 3-D deconvolution references: Wiener filtering and Richardson–Lucy.

* :func:`wiener3d` — the closed-form Wiener filter ``X = conj(H) Y / (|H|² + 1/snr)`` (Wiener 1949)
  of the 2-D :func:`~nefi.instances.deconvolution.wiener.wiener`, generalized to N-D volumes:
  boundary normalization by the blurred domain indicator (undoes the zero-padding edge darkening),
  replicate padding by a quarter of each axis, FFT, crop, clip. :func:`wiener3d_discrepancy` picks
  the SNR by Morozov's discrepancy principle.
* :func:`richardson_lucy` — the maximum-likelihood EM iteration for Poisson data (Richardson 1972;
  Lucy 1974), the workhorse of fluorescence deconvolution:
  ``x ← x · Aᵀ(y / (A x + b)) / Aᵀ1`` with the exact adjoint of the zero-padded blur (autograd), a
  known background ``b`` and a fixed iteration count (early stopping is the regularizer).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...errors import ConfigError, ShapeError
from ...operators.conv import FFTConvolution, fft_convolve, prepare_kernel

__all__ = ["richardson_lucy", "wiener3d", "wiener3d_discrepancy"]


def wiener3d(
    image: torch.Tensor,
    psf: FFTConvolution | torch.Tensor,
    snr: float = 100.0,
    pad: tuple[int, ...] | None = None,
    clip: tuple[float | None, float | None] | None = (0.0, None),
    boundary: str = "normalize",
    pad_mode: str = "replicate",
) -> torch.Tensor:
    """Wiener deconvolution of a 1-/2-/3-D image (``image`` in image units).

    Args:
        image: blurred volume ``(*grid)``.
        psf: the inversion :class:`~nefi.operators.FFTConvolution` (its kernel is sampled on the
            image grid) or a kernel tensor centered at ``shape // 2``.
        snr: signal-to-noise power ratio (``1/snr`` regularizes).
        pad: per-axis padding in voxels (default: a quarter of each axis).
        clip: optional ``(lo, hi)`` clamp (default non-negativity).
        boundary: ``"normalize"`` (divide by the blurred domain indicator first) or ``"none"``.
        pad_mode: :func:`torch.nn.functional.pad` mode (``replicate`` | ``reflect`` | ``constant``).
    """
    y = torch.as_tensor(image)
    d = y.ndim
    if d not in (1, 2, 3):
        raise ShapeError(f"wiener3d expects a 1-3-D image, got shape {tuple(y.shape)}")
    if boundary not in ("normalize", "none"):
        raise ConfigError(f"boundary must be 'normalize' or 'none', got {boundary!r}")
    n = tuple(y.shape)
    if isinstance(psf, FFTConvolution):
        k = psf.kernel(n, device=y.device, dtype=y.dtype)
    else:
        k = torch.as_tensor(psf).to(device=y.device, dtype=y.dtype)
    if boundary == "normalize":
        w = fft_convolve(torch.ones_like(y), k, periodic=False)
        y = y / w.clamp_min(1e-3)
    pads_ax = tuple(max(0, s // 4) for s in n) if pad is None else tuple(int(p) for p in pad)
    pads_ax = tuple(min(p, s - 1) for p, s in zip(pads_ax, n))
    flat: list[int] = []
    for p in reversed(pads_ax):
        flat += [p, p]
    yp = F.pad(y[None, None], flat, mode=pad_mode)[0, 0] if any(pads_ax) else y
    big = tuple(yp.shape)
    dims = tuple(range(-d, 0))
    kf, _ = prepare_kernel(k, big, periodic=True)
    Y = torch.fft.rfftn(yp, dim=dims)
    X = torch.conj(kf) * Y / (kf.abs() ** 2 + 1.0 / float(snr))
    x = torch.fft.irfftn(X, s=big, dim=dims)
    x = x[tuple(slice(p, p + s) for p, s in zip(pads_ax, n))]
    if clip is not None:
        x = x.clamp(min=clip[0], max=clip[1])
    return x


def wiener3d_discrepancy(
    image: torch.Tensor,
    psf: FFTConvolution,
    noise_std: float,
    tau: float = 1.0,
    snrs: torch.Tensor | None = None,
    **kw,
) -> tuple[torch.Tensor, float]:
    """Wiener filter with the SNR chosen by Morozov's discrepancy principle.

    Scans ``snrs`` (default ``logspace(0, 6, 25)``) from strong to weak regularization and returns
    the first reconstruction whose data residual ``RMS(A x − y)`` (the non-periodic forward model)
    drops to ``tau · noise_std``; if none does, the one with the smallest residual.

    Returns:
        ``(reconstruction, snr)``.
    """
    y = torch.as_tensor(image)
    snrs = torch.logspace(0, 6, 25) if snrs is None else torch.as_tensor(snrs)
    best, best_r, best_s = None, float("inf"), float(snrs[0])
    for s in snrs.tolist():
        x = wiener3d(y, psf, s, **kw)
        r = float(torch.sqrt(((psf({psf.primary: x}) - y) ** 2).mean()))
        if r <= tau * noise_std:
            return x, float(s)
        if r < best_r:
            best, best_r, best_s = x, r, float(s)
    return best, best_s  # type: ignore[return-value]


def richardson_lucy(
    counts: torch.Tensor,
    blur: FFTConvolution,
    n_iter: int = 40,
    gain: float = 1.0,
    background: float = 0.0,
    x0: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Richardson–Lucy deconvolution of (photon-count or intensity) data.

    Model ``E[y] = gain · A x + background`` with ``A`` the zero-padded blur ``blur``; the EM update
    ``x ← x · Aᵀ(y / (gain A x + b)) / (Aᵀ1)`` keeps ``x ≥ 0`` and preserves the flux. Negative data
    (Gaussian noise) are clamped to 0.

    Args:
        counts: data ``(*grid)``.
        blur: inversion blur operator (image units).
        n_iter: number of EM iterations (the regularization; RL amplifies noise when run long).
        gain / background: count scaling and additive background (Poisson data).
        x0: initial estimate (default: the flat image with the data's mean flux).
        eps: floor of the predicted counts.

    Returns:
        The estimate in image units.
    """
    y = torch.as_tensor(counts).clamp_min(0.0)
    x = (
        torch.full_like(y, float((y.mean() - background).clamp_min(eps) / gain))
        if x0 is None
        else torch.as_tensor(x0).to(y).clamp_min(eps)
    )

    def adjoint(r: torch.Tensor) -> torch.Tensor:
        z = torch.zeros_like(r, requires_grad=True)
        with torch.enable_grad():
            (g,) = torch.autograd.grad(blur({blur.primary: z}), z, grad_outputs=r)
        return g

    with torch.no_grad():
        norm = adjoint(torch.ones_like(y)).clamp_min(eps)
        for _ in range(int(n_iter)):
            lam = gain * blur({blur.primary: x}) + background
            x = x * adjoint(y / lam.clamp_min(eps)) / norm
    return x
