"""Basic reconstruction metrics (torch, device-agnostic). All take ``(pred, gt)`` tensors."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
import torch.nn.functional as F

from ..registry import register


def _prep(pred, gt):
    pred = torch.as_tensor(pred).detach()
    gt = torch.as_tensor(gt).detach().to(pred)
    return pred, gt


@register("metric", "mse")
def mse(pred, gt, mask=None) -> float:
    pred, gt = _prep(pred, gt)
    r = (pred - gt) ** 2
    if mask is not None:
        m = torch.as_tensor(mask).to(pred)
        return float((r * m).sum() / m.sum().clamp_min(1))
    return float(r.mean())


@register("metric", "rmse")
def rmse(pred, gt, mask=None) -> float:
    return mse(pred, gt, mask) ** 0.5


@register("metric", "mae")
def mae(pred, gt) -> float:
    pred, gt = _prep(pred, gt)
    return float((pred - gt).abs().mean())


@register("metric", "relative_error")
def relative_error(pred, gt) -> float:
    """``‖pred − gt‖₂ / ‖gt‖₂``."""
    pred, gt = _prep(pred, gt)
    return float(
        torch.linalg.vector_norm(pred - gt) / torch.linalg.vector_norm(gt).clamp_min(1e-30)
    )


@register("metric", "psnr")
def psnr(pred, gt, data_range: float | None = None) -> float:
    """Peak SNR in dB; ``data_range`` defaults to ``gt.max() - gt.min()`` (NeFTY uses α_max −
    α_min).
    """
    pred, gt = _prep(pred, gt)
    if data_range is None:
        data_range = float(gt.max() - gt.min())
    m = float(((pred - gt) ** 2).mean())
    if m <= 0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(data_range**2 / m)))


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    return g / g.sum()


def _ssim_nd(x: torch.Tensor, y: torch.Tensor, data_range: float, win: int, sigma: float, k1, k2):
    """SSIM for 1-D/2-D tensors (no batch), Gaussian window, 'valid' convolution."""
    d = x.ndim
    g = _gaussian_window(win, sigma, x.device, x.dtype)
    if d == 1:
        w = g.view(1, 1, -1)
        conv = F.conv1d
    else:
        w = (g[:, None] * g[None, :]).view(1, 1, win, win)
        conv = F.conv2d
    xb, yb = x.view(1, 1, *x.shape), y.view(1, 1, *y.shape)
    mu_x, mu_y = conv(xb, w), conv(yb, w)
    sxx = conv(xb * xb, w) - mu_x**2
    syy = conv(yb * yb, w) - mu_y**2
    sxy = conv(xb * yb, w) - mu_x * mu_y
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return float(s.mean())


@register("metric", "ssim")
def ssim(
    pred,
    gt,
    data_range: float | None = None,
    window: int = 7,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural similarity. 2-D: standard; 3-D: averaged over slices along the last axis
    (NeFTY App. E.2); 1-D: 1-D Gaussian window."""
    pred, gt = _prep(pred, gt)
    pred, gt = pred.float(), gt.float()
    if data_range is None:
        data_range = float(gt.max() - gt.min()) or 1.0
    win = min(window, *[s for s in pred.shape[: min(pred.ndim, 2)]])
    if win % 2 == 0:
        win -= 1
    if pred.ndim <= 2:
        return _ssim_nd(pred, gt, data_range, win, sigma, k1, k2)
    if pred.ndim == 3:
        vals = [
            _ssim_nd(pred[..., z], gt[..., z], data_range, win, sigma, k1, k2)
            for z in range(pred.shape[-1])
        ]
        return float(sum(vals) / len(vals))
    raise ValueError("ssim supports 1-3 dims")


def threshold_mask(x, threshold: float, mode: str = "below") -> torch.Tensor:
    x = torch.as_tensor(x)
    return (x < threshold) if mode == "below" else (x > threshold)


@register("metric", "iou")
def iou(pred_mask, gt_mask) -> float:
    a, b = torch.as_tensor(pred_mask).bool(), torch.as_tensor(gt_mask).bool()
    union = (a | b).sum()
    if union == 0:
        return 1.0
    return float((a & b).sum() / union)


@register("metric", "dice")
def dice(pred_mask, gt_mask) -> float:
    a, b = torch.as_tensor(pred_mask).bool(), torch.as_tensor(gt_mask).bool()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2 * (a & b).sum() / denom)


def masked_ssim(pred, gt, mask, **kw) -> float:
    """SSIM restricted to a support: values outside ``mask`` are zeroed in both inputs (NeTMY App.
    E.3).
    """
    pred, gt = _prep(pred, gt)
    m = torch.as_tensor(mask).to(pred)
    return ssim(pred * m, gt * m, **kw)


def evaluate(
    pred: Mapping[str, torch.Tensor] | torch.Tensor,
    gt: Mapping[str, torch.Tensor] | torch.Tensor,
    metrics: Mapping[str, Callable],
    field: str | None = None,
    **kw,
) -> dict[str, float]:
    """Apply a dict of metrics to (pred, gt); accepts field dicts (uses ``field`` or the first
    key).
    """
    if isinstance(pred, Mapping):
        key = field or next(iter(pred))
        pred = pred[key]
    if isinstance(gt, Mapping):
        key = field or next(iter(gt))
        gt = gt[key]
    return {name: float(fn(pred, gt, **kw)) for name, fn in metrics.items()}


BASIC_METRICS: dict[str, Callable] = {"mse": mse, "psnr": psnr, "relative_error": relative_error}
