"""Classical Wiener deconvolution — the closed-form reference reconstruction."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...errors import ConfigError, ShapeError
from ...measurement import Measurement
from ...operators.conv import FFTConvolution, fft_convolve, prepare_kernel


def _kernel(psf: torch.Tensor | FFTConvolution, n, device, dtype) -> torch.Tensor:
    if isinstance(psf, FFTConvolution):
        return psf.kernel(n, device=device, dtype=dtype)
    return torch.as_tensor(psf).to(device=device, dtype=dtype)


def wiener(
    measurement: Measurement | torch.Tensor,
    psf: torch.Tensor | FFTConvolution,
    snr: float = 100.0,
    pad: int | None = None,
    clip: tuple[float | None, float | None] | None = (0.0, None),
    boundary: str = "normalize",
    pad_mode: str = "replicate",
) -> torch.Tensor:
    """Wiener filter ``X = conj(H) Y / (|H|² + 1/snr)`` (Wiener 1949; e.g. Gonzalez & Woods §5.8).

    The circular FFT is adapted to the zero-padded (non-periodic) blur of the forward model by
    (i) dividing the image by the blurred domain indicator ``k ∗ 1_Ω`` (``boundary="normalize"``,
    undoes the edge darkening of zero padding for locally smooth images; ``"none"`` skips it),
    then (ii) padding by ``pad`` pixels (``pad_mode`` replicate / reflect / constant) before the
    FFT and cropping afterwards. Without (i) the dark rim is deconvolved as a real feature and the
    error explodes near the boundary.

    Args:
        measurement: blurred image ``(n0, n1)`` (or a :class:`~nefi.Measurement`), in image units.
        psf: kernel tensor with its center at ``shape // 2`` (any size), or the
            :class:`~nefi.operators.FFTConvolution` used for inversion (its kernel is sampled on the
            measurement grid).
        snr: signal-to-noise power ratio; ``1/snr`` is the constant noise-to-signal regularizer
            (see :func:`wiener_discrepancy` for an automatic choice).
        pad: padding in pixels (default ``min(n) // 4``).
        clip: optional ``(lo, hi)`` clamp of the result (default: non-negativity).
        boundary: ``"normalize"`` or ``"none"``.
        pad_mode: ``torch.nn.functional.pad`` mode.

    Returns:
        Deconvolved image with the measurement's shape.
    """
    y = measurement.data if isinstance(measurement, Measurement) else torch.as_tensor(measurement)
    if y.ndim != 2:
        raise ShapeError(f"wiener expects a 2-D image, got shape {tuple(y.shape)}")
    if boundary not in ("normalize", "none"):
        raise ConfigError(f"boundary must be 'normalize' or 'none', got {boundary!r}")
    n = tuple(y.shape)
    k = _kernel(psf, n, y.device, y.dtype)
    if boundary == "normalize":
        w = fft_convolve(torch.ones_like(y), k, periodic=False)
        y = y / w.clamp_min(1e-3)
    p = min(n) // 4 if pad is None else int(pad)
    p = min(p, min(n) - 1)
    yp = F.pad(y[None, None], (p, p, p, p), mode=pad_mode)[0, 0] if p > 0 else y
    big = tuple(yp.shape)
    kf, _ = prepare_kernel(k, big, periodic=True)
    Y = torch.fft.rfftn(yp, dim=(-2, -1))
    X = torch.conj(kf) * Y / (kf.abs() ** 2 + 1.0 / float(snr))
    x = torch.fft.irfftn(X, s=big, dim=(-2, -1))
    x = x[p : p + n[0], p : p + n[1]] if p > 0 else x
    if clip is not None:
        x = x.clamp(min=clip[0], max=clip[1])
    return x


def wiener_discrepancy(
    measurement: Measurement | torch.Tensor,
    psf: FFTConvolution,
    noise_std: float,
    tau: float = 1.0,
    snrs: torch.Tensor | None = None,
    **kw,
) -> tuple[torch.Tensor, float]:
    """Wiener filter with the SNR chosen by Morozov's discrepancy principle.

    Scans ``snrs`` (default ``logspace(0, 6, 25)``) from strong to weak regularization and returns
    the first reconstruction whose data residual ``RMS(k ∗ x - y)`` (non-periodic forward model)
    drops to ``tau · noise_std``; if none does, the one with the smallest residual.

    Returns:
        ``(reconstruction, snr)``.
    """
    y = measurement.data if isinstance(measurement, Measurement) else torch.as_tensor(measurement)
    snrs = torch.logspace(0, 6, 25) if snrs is None else torch.as_tensor(snrs)
    best, best_r, best_s = None, float("inf"), float(snrs[0])
    for s in snrs.tolist():
        x = wiener(y, psf, s, **kw)
        r = float(torch.sqrt(((psf({psf.primary: x}) - y) ** 2).mean()))
        if r <= tau * noise_std:
            return x, float(s)
        if r < best_r:
            best, best_r, best_s = x, r, float(s)
    return best, best_s


__all__ = ["wiener", "wiener_discrepancy"]
