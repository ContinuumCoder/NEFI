"""Segmentation, depth and frequency-domain metrics (NeFTY App. E.2 / G.5).

* :func:`iou_below` — volumetric IoU of the defect masks ``{x < τ}`` (NeFTY App. E.2, τ = 0.03).
* :func:`edge_f1` — defect-boundary F1 of thresholded gradient magnitudes with a dilation
  tolerance (NeFTY App. G.5).
* :func:`abs_rel`, :func:`depth_rmse`, :func:`delta_threshold` — the depth metrics of Eigen et al.
  used for the 2.5-D PVC depth maps (NeFTY App. E.2).
* :func:`radial_power_spectrum` — radially averaged power spectrum of a 1-3-D field (App. G.5).

All functions accept tensors or arrays, work on any device and return Python floats (except the
spectrum, which returns a ``(frequencies, power)`` pair).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from ..registry import register

__all__ = [
    "abs_rel",
    "binary_dilation",
    "delta_threshold",
    "depth_rmse",
    "edge_f1",
    "field_gradient_magnitude",
    "iou_below",
    "radial_power_spectrum",
]


def _t(x) -> torch.Tensor:
    return torch.as_tensor(x).detach()


def _valid(pred: torch.Tensor, gt: torch.Tensor, mask) -> torch.Tensor:
    """Pixels with a positive, finite ground-truth depth (and inside ``mask`` if given)."""
    m = torch.isfinite(gt) & (gt > 0)
    if mask is not None:
        m = m & _t(mask).to(device=gt.device).bool()
    return m


@register("metric", "iou_below")
def iou_below(pred, gt, tau: float = 0.03) -> float:
    """IoU of the sub-threshold masks ``{pred < τ}`` and ``{gt < τ}`` (NeFTY App. E.2: defect
    voxels have ``α < τ = 0.03``, twice the upper end of the defect diffusivity range). Returns 1.0
    when both masks are empty."""
    a, b = _t(pred) < tau, _t(gt).to(_t(pred).device) < tau
    union = (a | b).sum()
    if int(union) == 0:
        return 1.0
    return float((a & b).sum() / union)


def field_gradient_magnitude(x, spacing: Sequence[float] | None = None) -> torch.Tensor:
    """``‖∇x‖`` of a 1-3-D field by central differences (one-sided at the borders)."""
    x = _t(x).double()
    grads = torch.gradient(x, spacing=list(spacing) if spacing is not None else 1.0)
    return torch.sqrt(sum(g**2 for g in grads))


def binary_dilation(mask, radius: int = 1) -> torch.Tensor:
    """Binary dilation of a 1-3-D mask with a ``(2r+1)^d`` box structuring element."""
    m = _t(mask).bool()
    if radius <= 0:
        return m
    d = m.ndim
    pool = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}[d]
    x = m.float().reshape(1, 1, *m.shape)
    y = pool(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return y.reshape(m.shape) > 0.5


@register("metric", "edge_f1")
def edge_f1(
    pred,
    gt,
    threshold: float = 0.5,
    dilate: int = 1,
    spacing: Sequence[float] | None = None,
) -> float:
    """Defect Edge F1 (NeFTY App. G.5).

    Edges are voxels whose gradient magnitude ``‖∇x‖`` exceeds ``threshold ×`` the maximum
    ground-truth gradient magnitude. Matching uses a dilated defect-boundary mask as tolerance:
    precision is the fraction of predicted edge voxels inside the ``dilate``-voxel dilation of the
    ground-truth edges, recall the fraction of ground-truth edge voxels inside the dilation of the
    predicted edges; the score is their harmonic mean (1.0 if both edge sets are empty).

    Args:
        pred / gt: fields of identical shape (1-3-D).
        threshold: fraction of the ground-truth maximum gradient magnitude.
        dilate: tolerance radius in voxels.
        spacing: optional physical spacing for the gradients (default: index units).
    """
    gp = field_gradient_magnitude(pred, spacing)
    gg = field_gradient_magnitude(gt, spacing).to(gp.device)
    thr = threshold * float(gg.max())
    ep, eg = gp > thr, gg > thr
    n_p, n_g = int(ep.sum()), int(eg.sum())
    if n_p == 0 and n_g == 0:
        return 1.0
    if n_p == 0 or n_g == 0:
        return 0.0
    precision = float((ep & binary_dilation(eg, dilate)).sum()) / n_p
    recall = float((eg & binary_dilation(ep, dilate)).sum()) / n_g
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


@register("metric", "abs_rel")
def abs_rel(pred, gt, mask=None) -> float:
    """Absolute relative depth error ``mean |d̂ − d*| / d*`` over valid pixels (Eigen et al.)."""
    p, g = _t(pred).double(), _t(gt).double().to(_t(pred).device)
    m = _valid(p, g, mask)
    if int(m.sum()) == 0:
        return float("nan")
    return float(((p[m] - g[m]).abs() / g[m]).mean())


@register("metric", "depth_rmse")
def depth_rmse(pred, gt, mask=None) -> float:
    """Depth RMSE ``sqrt(mean (d̂ − d*)²)`` over valid pixels (Eigen et al.)."""
    p, g = _t(pred).double(), _t(gt).double().to(_t(pred).device)
    m = _valid(p, g, mask)
    if int(m.sum()) == 0:
        return float("nan")
    return float(((p[m] - g[m]) ** 2).mean().sqrt())


@register("metric", "delta_threshold")
def delta_threshold(pred, gt, k: int = 1, mask=None, base: float = 1.25) -> float:
    """Threshold accuracy ``|{p : max(d̂/d*, d*/d̂) < base^k}| / |P|`` (Eigen et al., NeFTY App. E.2).

    Non-positive or non-finite predictions count as failures.
    """
    p, g = _t(pred).double(), _t(gt).double().to(_t(pred).device)
    m = _valid(p, g, mask)
    n = int(m.sum())
    if n == 0:
        return float("nan")
    pm, gm = p[m], g[m]
    ok = torch.isfinite(pm) & (pm > 0)
    ratio = torch.where(ok, torch.maximum(pm / gm, gm / pm.clamp_min(1e-300)), torch.inf)
    return float((ratio < base**k).sum()) / n


@register("metric", "radial_power_spectrum")
def radial_power_spectrum(
    x,
    n_bins: int | None = None,
    spacing: Sequence[float] | None = None,
    subtract_mean: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Radially averaged power spectrum of a 1-3-D field (NeFTY App. G.5 / Fig. 9).

    Computes ``|FFT(x)|²`` (optionally of the mean-free field), then averages it over shells of
    constant ``|k|``. With ``spacing`` the frequencies are in cycles per unit length (the
    through-thickness Nyquist limit of the paper's grid is ``1/(2Δz) = 8``), otherwise in cycles
    per sample.

    Args:
        x: field (1-3-D), e.g. a ``16 × 16 × 8`` crop around the defect cluster.
        n_bins: number of radial bins (default ``max(shape) // 2``).
        spacing: physical sample spacing per axis.
        subtract_mean: remove the mean (DC) before transforming.

    Returns:
        ``(frequencies, power)``: bin centres and the mean power per bin (empty bins are NaN).
    """
    x = _t(x).double().cpu()
    if subtract_mean:
        x = x - x.mean()
    sp = [1.0] * x.ndim if spacing is None else [float(s) for s in spacing]
    power = torch.fft.fftn(x).abs() ** 2 / x.numel()
    freqs = [torch.fft.fftfreq(n, d=s, dtype=torch.float64) for n, s in zip(x.shape, sp)]
    kk = torch.stack(torch.meshgrid(*freqs, indexing="ij"), dim=-1).norm(dim=-1)
    n_bins = int(n_bins or max(max(x.shape) // 2, 1))
    k_max = float(kk.max())
    edges = torch.linspace(0.0, k_max * (1 + 1e-9), n_bins + 1, dtype=torch.float64)
    idx = torch.bucketize(kk.reshape(-1), edges[1:-1], right=True)
    sums = torch.zeros(n_bins, dtype=torch.float64).index_add_(0, idx, power.reshape(-1))
    counts = torch.zeros(n_bins, dtype=torch.float64).index_add_(
        0, idx, torch.ones_like(power.reshape(-1))
    )
    mean = torch.where(counts > 0, sums / counts.clamp_min(1), torch.nan)
    return 0.5 * (edges[1:] + edges[:-1]), mean
