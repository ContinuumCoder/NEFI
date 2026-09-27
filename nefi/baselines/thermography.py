"""Classical pulsed-thermography heuristics: PPT and TSR (NeFTY App. F.4).

Both methods are **1-D, pixel-wise** inversions of the surface temperature decay after a flash:

* **Pulsed Phase Thermography** (Maldague & Marinetti 1996; Ibarra-Castanedo & Maldague 2004):
  the per-pixel DFT of the decay, phase contrast against a sound reference, and the depth
  ``d = C1 · sqrt(α / (π f))`` from the frequency ``f`` of the dominant phase contrast
  (``C1 ≈ 1.82``).
* **Thermographic Signal Reconstruction** (Shepard et al. 2002, 2015): a low-order polynomial fit of
  ``log T`` against ``log t`` per pixel; the time ``t_peak`` at which the second logarithmic
  derivative peaks marks the arrival of the defect echo and gives ``d = C2 · sqrt(α t_peak)``
  (``C2 ≈ 1.8``).

They ignore lateral diffusion and assume a semi-infinite geometry (NeFTY §2, App. F.4), which is
why they are the natural classical baselines for the thermal-tomography instance: they produce a
2-D defect mask and a 2.5-D depth map, which :func:`depth_to_alpha_volume` lifts to a volumetric
diffusivity field so the same 3-D / 2-D / 2.5-D metrics apply.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ..registry import register

__all__ = [
    "PPT_C1",
    "TSR_C2",
    "contrast_mask",
    "depth_to_alpha_volume",
    "ppt",
    "tsr",
]

PPT_C1 = 1.82
TSR_C2 = 1.8


def _mad_sigma(x: torch.Tensor) -> torch.Tensor:
    med = x.median()
    return 1.4826 * (x - med).abs().median()


def contrast_mask(contrast: torch.Tensor, k: float = 2.0) -> torch.Tensor:
    """Binary defect mask ``|c − median(c)| > k · 1.4826 · MAD(c)`` (robust noise floor, App. H)."""
    med = contrast.median()
    sig = _mad_sigma(contrast).clamp_min(1e-12)
    return (contrast - med).abs() > k * sig


def ppt(
    frames: torch.Tensor,
    dt: float,
    alpha: float,
    *,
    c1: float = PPT_C1,
    mode: str = "peak",
    freq_range: tuple[float, float] | None = None,
    reference: torch.Tensor | None = None,
    mask_k: float = 2.0,
    tail: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Pulsed Phase Thermography depth map from surface frames ``(n_t, H, W)``.

    Args:
        frames: ambient-shifted surface temperatures at uniformly spaced times (post flash).
        dt: frame spacing (physical or unitless time).
        alpha: bulk thermal diffusivity in matching units.
        c1: depth constant of the PPT formula.
        mode: ``"peak"`` uses the frequency of the largest phase contrast (NeFTY App. F.4);
            ``"blind"`` uses the highest frequency at which the contrast still exceeds
            ``tail × max`` (the classic blind frequency).
        freq_range: optional ``(f_lo, f_hi)`` restricting the analysed frequencies (the PVC
            benchmark uses 0.01–0.5 Hz); default: all non-zero DFT bins.
        reference: sound-region phase reference ``(n_freq,)``; default: the per-frequency median
            phase over all pixels (sound majority assumption).
        mask_k: MAD multiplier for the defect mask.
        tail: threshold fraction for ``mode="blind"``.

    Returns:
        ``{"depth": (H, W), "contrast": (H, W) max |Δφ|, "f_peak": (H, W), "mask": bool (H, W),
        "phase_contrast": (n_freq, H, W), "freqs": (n_freq,)}``.
    """
    x = torch.as_tensor(frames).detach().double()
    n_t = x.shape[0]
    spec = torch.fft.rfft(x - x.mean(0, keepdim=True), dim=0)  # remove DC before the DFT
    freqs = torch.fft.rfftfreq(n_t, d=float(dt)).to(x.device)
    phase = torch.angle(spec)
    ref = reference.to(phase) if reference is not None else phase.flatten(1).median(dim=1).values
    dphi = phase - ref.view(-1, *([1] * (x.ndim - 1)))
    dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi)).abs()  # wrap to [0, π]
    valid = freqs > 0
    if freq_range is not None:
        valid &= (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if int(valid.sum()) == 0:
        raise ValueError("ppt: no DFT frequency inside freq_range")
    dphi_v = dphi[valid]
    f_v = freqs[valid]
    contrast, idx = dphi_v.max(dim=0)
    if mode == "peak":
        f_sel = f_v[idx]
    elif mode == "blind":
        above = dphi_v > tail * contrast.clamp_min(1e-12)
        # highest frequency index still above the tail threshold
        pos = torch.arange(len(f_v), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        f_sel = f_v[(above * pos).amax(dim=0)]
    else:
        raise ValueError("mode must be 'peak' or 'blind'")
    depth = c1 * torch.sqrt(alpha / (math.pi * f_sel.clamp_min(1e-12)))
    return {
        "depth": depth,
        "contrast": contrast,
        "f_peak": f_sel,
        "mask": contrast_mask(contrast, mask_k),
        "phase_contrast": dphi_v,
        "freqs": f_v,
    }


def tsr(
    frames: torch.Tensor,
    dt: float,
    alpha: float,
    *,
    order: int = 5,
    c2: float = TSR_C2,
    t0: float | None = None,
    mask_k: float = 2.0,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Thermographic Signal Reconstruction depth map from surface frames ``(n_t, H, W)``.

    Fits ``log T = Σ_k a_k (log t)^k`` per pixel (``order`` 5 is the standard choice), evaluates the
    second logarithmic derivative on the sampled times and takes its peak time ``t_peak``; the depth
    is ``c2 · sqrt(α t_peak)``. The contrast used for the mask is the second-derivative peak
    height relative to the median pixel.

    Args:
        frames: ambient-shifted, positive surface temperatures at times ``t0 + i·dt``.
        dt: frame spacing; ``t0`` defaults to ``dt`` (first frame one step after the flash).
        alpha: bulk diffusivity.
    """
    x = torch.as_tensor(frames).detach().double().clamp_min(eps)
    n_t = x.shape[0]
    t = (float(t0) if t0 is not None else float(dt)) + float(dt) * torch.arange(
        n_t, dtype=torch.float64, device=x.device
    )
    lt = torch.log(t)
    powers = torch.stack([lt**k for k in range(order + 1)], dim=1)  # (n_t, order+1)
    y = torch.log(x).reshape(n_t, -1)  # (n_t, P)
    coef = torch.linalg.lstsq(powers, y).solution  # (order+1, P)
    # second derivative w.r.t. log t: Σ_k k (k-1) a_k (log t)^(k-2)
    d2 = torch.zeros_like(y)
    for k in range(2, order + 1):
        d2 += k * (k - 1) * coef[k][None, :] * lt[:, None] ** (k - 2)
    peak, idx = d2.max(dim=0)
    t_peak = t[idx]
    depth = c2 * torch.sqrt(alpha * t_peak)
    spatial = x.shape[1:]
    contrast = (peak - peak.median()).reshape(spatial)
    return {
        "depth": depth.reshape(spatial),
        "t_peak": t_peak.reshape(spatial),
        "contrast": contrast,
        "mask": contrast_mask(peak.reshape(spatial), mask_k),
        "second_derivative": d2.reshape(n_t, *spatial),
        "coef": coef.reshape(order + 1, *spatial),
    }


def depth_to_alpha_volume(
    depth: torch.Tensor,
    mask: torch.Tensor,
    nz: int,
    thickness: float,
    alpha_bulk: float,
    alpha_defect: float,
    defect_thickness: float | None = None,
    z_coords: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Lift a 2-D depth map + mask to a volumetric diffusivity field ``(H, W, nz)``.

    Voxels under masked pixels whose depth coordinate (from the observed face at ``z = 0``) lies in
    ``[d, d + defect_thickness]`` receive ``alpha_defect``; everything else ``alpha_bulk``.
    ``defect_thickness`` defaults to 15 % of the slab thickness.
    """
    d = torch.as_tensor(depth).double()
    m = torch.as_tensor(mask).bool()
    if z_coords is None:
        z = (torch.arange(nz, dtype=torch.float64) + 0.5) * (float(thickness) / nz)
    else:
        z = torch.as_tensor(z_coords, dtype=torch.float64)
    dz = float(defect_thickness) if defect_thickness is not None else 0.15 * float(thickness)
    d = d.clamp(0.0, float(thickness) - 1e-9)
    inside = (z.view(*([1] * d.ndim), -1) >= d[..., None]) & (
        z.view(*([1] * d.ndim), -1) <= (d + dz)[..., None]
    )
    inside &= m[..., None]
    vol = torch.full((*d.shape, nz), float(alpha_bulk), dtype=torch.float64)
    vol[inside] = float(alpha_defect)
    return vol


@register("baseline", "ppt")
def ppt_baseline(*args, **kwargs):  # pragma: no cover - registry convenience
    return ppt(*args, **kwargs)


@register("baseline", "tsr")
def tsr_baseline(*args, **kwargs):  # pragma: no cover - registry convenience
    return tsr(*args, **kwargs)
