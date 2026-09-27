"""Sparse-localization and distributional metrics — NeTMY App. E.3.

* :func:`gmsd` — gradient-magnitude similarity deviation (Xue et al. 2014) with Prewitt gradients
  on maps rescaled to a shared maximum; lower is better.
* :func:`hungarian_f1` — one-to-one peak matching (3×3 local maxima above 5 % of the per-image
  maximum, Hungarian assignment within a fixed radius); higher is better.
* :func:`sliced_wasserstein` — sliced Wasserstein-2 distance between the two normalized mass
  distributions over pixel coordinates (mean over random projections of the 1-D W2); lower is
  better.

All functions accept 2-D tensors / arrays ``(H, W)`` and return python floats. They are
insensitive to the absolute scale of the maps (NeTMY App. D.5), except where stated.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..registry import register

_PREWITT_X = torch.tensor([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]) / 3.0


def _as_2d(x, name: str = "input") -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x).detach()
    t = t.to("cpu", torch.float64)
    t = t.squeeze()
    if t.ndim != 2:
        raise ValueError(f"{name} must be a 2-D map, got shape {tuple(t.shape)}")
    return t


# ---------------------------------------------------------------------------------------------
# GMSD
# ---------------------------------------------------------------------------------------------
def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Prewitt gradient magnitude ``sqrt((Dx ∗ x)² + (Dy ∗ x)²)`` with replicate padding."""
    kx = _PREWITT_X.to(x)
    ky = kx.T.contiguous()
    xp = F.pad(x[None, None], (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(xp, kx[None, None])[0, 0]
    gy = F.conv2d(xp, ky[None, None])[0, 0]
    return torch.sqrt(gx * gx + gy * gy)


@register("metric", "gmsd")
def gmsd(pred, gt, c: float = 0.0026, rescale: bool = True) -> float:
    """Gradient Magnitude Similarity Deviation (NeTMY App. E.3; Xue et al. 2014).

    Both maps are rescaled to a shared maximum (1) before Prewitt gradients are taken;
    ``GMS = (2 G_p G_g + c) / (G_p² + G_g² + c)`` and ``GMSD = std_r GMS(r)``. The stability
    constant ``c = 0.0026`` is the standard value for unit-range images. 0 for identical maps.
    """
    p, g = _as_2d(pred, "pred"), _as_2d(gt, "gt")
    if p.shape != g.shape:
        raise ValueError(f"gmsd: shape mismatch {tuple(p.shape)} vs {tuple(g.shape)}")
    if rescale:
        p = p / p.abs().max().clamp_min(1e-30)
        g = g / g.abs().max().clamp_min(1e-30)
    gp, gg = gradient_magnitude(p), gradient_magnitude(g)
    gms = (2.0 * gp * gg + c) / (gp * gp + gg * gg + c)
    return float(gms.std(unbiased=False))


# ---------------------------------------------------------------------------------------------
# Peaks and Hungarian F1
# ---------------------------------------------------------------------------------------------
def peak_positions(x, threshold: float = 0.05, relative: bool = True) -> torch.Tensor:
    """Local maxima of a 2-D map above a threshold (NeTMY App. E.3).

    A pixel is a peak if it equals the maximum of its 3×3 neighbourhood and exceeds
    ``threshold · max(x)`` (``relative=True``) or ``threshold`` (absolute). Plateaus are
    de-duplicated by greedy non-maximum suppression (strongest first, Chebyshev radius 1).

    Returns:
        Long tensor ``(n_peaks, 2)`` of ``(row, col)`` indices, strongest first.
    """
    t = _as_2d(x)
    empty = torch.zeros(0, 2, dtype=torch.long)
    mx = float(t.max()) if t.numel() else 0.0
    if relative and mx <= 0.0:
        return empty
    thr = threshold * mx if relative else float(threshold)
    pooled = F.max_pool2d(t[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    idx = torch.nonzero((t >= pooled) & (t > thr))
    if idx.shape[0] <= 1:
        return idx
    vals = t[idx[:, 0], idx[:, 1]]
    order = torch.argsort(vals, descending=True, stable=True)
    kept: list[torch.Tensor] = []
    for k in order.tolist():
        p = idx[k]
        if all(int((p - q).abs().max()) > 1 for q in kept):
            kept.append(p)
    return torch.stack(kept)


def peak_match(pred, gt, radius: float = 2.0, threshold: float = 0.05) -> dict[str, float]:
    """Hungarian peak matching counts: ``{"tp", "fp", "fn", "precision", "recall", "f1"}``.

    Peaks of both maps (:func:`peak_positions`) are paired one-to-one by
    :func:`scipy.optimize.linear_sum_assignment` on Euclidean pixel distances; only pairs within
    ``radius`` pixels count as true positives. ``f1 = 2TP / (2TP + FP + FN)`` and is 0 when
    either map has no supra-threshold peak (NeTMY App. E.3).
    """
    pp = peak_positions(pred, threshold).double()
    gp = peak_positions(gt, threshold).double()
    n_p, n_g = pp.shape[0], gp.shape[0]
    if n_p == 0 or n_g == 0:
        return {
            "tp": 0.0,
            "fp": float(n_p),
            "fn": float(n_g),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    d = torch.cdist(pp, gp).numpy()
    big = 1e6 + d.max()
    cost = np.where(d <= radius, d, big)
    rows, cols = linear_sum_assignment(cost)
    tp = int(np.sum(d[rows, cols] <= radius))
    fp, fn = n_p - tp, n_g - tp
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": tp / n_p,
        "recall": tp / n_g,
        "f1": 2.0 * tp / (2.0 * tp + fp + fn),
    }


@register("metric", "hungarian_f1")
def hungarian_f1(pred, gt, radius: float = 2.0, threshold: float = 0.05) -> float:
    """Hungarian-matched localization F1 at a fixed radius (NeTMY App. E.3; 0 if degenerate)."""
    return float(peak_match(pred, gt, radius, threshold)["f1"])


# ---------------------------------------------------------------------------------------------
# Sliced Wasserstein
# ---------------------------------------------------------------------------------------------
def wasserstein2_1d(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, squared: bool = False
) -> torch.Tensor:
    """Exact W2 between two discrete 1-D distributions on shared support points.

    Args:
        x: support positions ``(..., n)`` (any order).
        a, b: non-negative weights ``(..., n)``, each summing to 1 along the last axis.

    The quantile functions of both measures are piecewise constant between the merged CDF
    breakpoints, so ``W2² = Σ_k Δt_k (Q_a(t_k) − Q_b(t_k))²`` evaluated at interval midpoints
    (sorting + CDF quantile interpolation; batched over leading dims).
    """
    order = torch.argsort(x, dim=-1)
    xs = torch.gather(x, -1, order)
    ca = torch.cumsum(torch.gather(a, -1, order), -1).contiguous()
    cb = torch.cumsum(torch.gather(b, -1, order), -1).contiguous()
    t = torch.sort(torch.cat([ca, cb], dim=-1), dim=-1).values
    t = t.clamp(0.0, 1.0)
    t0 = torch.cat([torch.zeros_like(t[..., :1]), t[..., :-1]], dim=-1)
    dt = (t - t0).clamp_min(0.0)
    mid = (0.5 * (t + t0)).contiguous()
    n = x.shape[-1]
    ia = torch.searchsorted(ca, mid).clamp_max(n - 1)
    ib = torch.searchsorted(cb, mid).clamp_max(n - 1)
    qa = torch.gather(xs, -1, ia)
    qb = torch.gather(xs, -1, ib)
    w2sq = (dt * (qa - qb) ** 2).sum(-1)
    return w2sq if squared else torch.sqrt(w2sq.clamp_min(0.0))


@register("metric", "sliced_wasserstein")
def sliced_wasserstein(
    pred, gt, n_proj: int = 128, seed: int = 0, pixel_units: bool = False
) -> float:
    """Sliced Wasserstein-2 distance between normalized mass maps (NeTMY App. E.3).

    Both maps are clipped at 0 and normalized to unit mass; pixel centres are projected on
    ``n_proj`` random unit directions (uniform angles, seeded) and the 1-D W2 distances are
    averaged. Coordinates are ``((i + ½)/H, (j + ½)/W) ∈ [0, 1]²`` (``pixel_units=False``) or
    pixel indices. Returns ``nan`` if either map has no positive mass.
    """
    p, g = _as_2d(pred, "pred"), _as_2d(gt, "gt")
    if p.shape != g.shape:
        raise ValueError(f"sliced_wasserstein: shape mismatch {tuple(p.shape)} vs {tuple(g.shape)}")
    p, g = p.clamp_min(0.0).reshape(-1), g.clamp_min(0.0).reshape(-1)
    sp, sg = float(p.sum()), float(g.sum())
    if sp <= 0.0 or sg <= 0.0:
        return math.nan
    h, w = _as_2d(pred).shape
    ii, jj = torch.meshgrid(
        torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij"
    )
    if pixel_units:
        coords = torch.stack([ii, jj], -1).reshape(-1, 2)
    else:
        coords = torch.stack([(ii + 0.5) / h, (jj + 0.5) / w], -1).reshape(-1, 2)
    gen = torch.Generator().manual_seed(int(seed))
    ang = 2.0 * math.pi * torch.rand(int(n_proj), generator=gen, dtype=torch.float64)
    dirs = torch.stack([torch.cos(ang), torch.sin(ang)], -1)  # (P, 2)
    proj = dirs @ coords.T  # (P, N)
    a = (p / sp).expand_as(proj)
    b = (g / sg).expand_as(proj)
    return float(wasserstein2_1d(proj, a, b).mean())


register("metric", "swd")(sliced_wasserstein)


__all__ = [
    "gmsd",
    "gradient_magnitude",
    "hungarian_f1",
    "peak_match",
    "peak_positions",
    "sliced_wasserstein",
    "wasserstein2_1d",
]
