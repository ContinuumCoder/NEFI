"""EIT ground-truth scenes: homogeneous background with elliptical inclusions.

Classes (benchmark difficulty):

* ``single``   — one conductive or resistive ellipse, moderate contrast (``contrast`` range);
* ``multi``    — 2–3 non-overlapping ellipses of either polarity;
* ``contrast`` — one ellipse with high contrast (``high_contrast`` range, e.g. 1:10–1:20), the
  regime where the harmonic-mean face coefficient matters most (NeFTY Prop. 1).

Inclusions are analytic, rendered with sub-cell area coverage, so the fine-grid data generator and
the native inversion grid see the same object.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from ...bench.base import SceneGenerator
from ...domain import Domain
from .._elliptic_common import cell_coverage

__all__ = ["EITScenes", "inclusion_iou"]


class EITScenes(SceneGenerator):
    """Background ``σ_bg`` plus elliptical inclusions (see module docstring).

    Args:
        domain: 2-D domain (centred square recommended).
        sigma_bg: background conductivity.
        contrast: ``(lo, hi)`` conductivity ratio of ``single``/``multi`` inclusions.
        high_contrast: ratio range of the ``contrast`` class.
        radius: semi-axis range as a fraction of the domain half-width.
        center_radius: inclusion centres lie within this fraction of the half-width.
        gap: minimum clearance between inclusions (fraction of the half-width).
        subsample: sub-cell samples per axis for area coverage.
    """

    classes = ("single", "multi", "contrast")

    def __init__(
        self,
        domain: Domain,
        sigma_bg: float = 1.0,
        contrast: Sequence[float] = (3.0, 6.0),
        high_contrast: Sequence[float] = (10.0, 20.0),
        radius: Sequence[float] = (0.15, 0.35),
        center_radius: float = 0.55,
        gap: float = 0.1,
        subsample: int = 4,
    ) -> None:
        super().__init__(domain)
        self.sigma_bg = float(sigma_bg)
        self.contrast = tuple(map(float, contrast))
        self.high_contrast = tuple(map(float, high_contrast))
        self.radius = tuple(map(float, radius))
        self.center_radius = float(center_radius)
        self.gap = float(gap)
        self.subsample = int(subsample)
        (x0, x1), (y0, y1) = domain.extent
        self.center = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
        self.half_width = 0.5 * min(x1 - x0, y1 - y0)

    def inclusions(self, rng: np.random.Generator, cls: str) -> list[dict]:
        """Sample inclusion parameters (independent of the rendering resolution)."""
        n = int(rng.integers(2, 4)) if cls == "multi" else 1
        crange = self.high_contrast if cls == "contrast" else self.contrast
        hw = self.half_width
        out: list[dict] = []
        for _ in range(n):
            for _attempt in range(200):
                a = rng.uniform(*self.radius) * hw
                b = rng.uniform(*self.radius) * hw
                ang = rng.uniform(0.0, math.pi)
                rr = self.center_radius * hw * math.sqrt(rng.uniform())
                th = rng.uniform(0.0, 2 * math.pi)
                cx = self.center[0] + rr * math.cos(th)
                cy = self.center[1] + rr * math.sin(th)
                ok = all(
                    math.hypot(cx - o["cx"], cy - o["cy"])
                    > max(a, b) + max(o["a"], o["b"]) + self.gap * hw
                    for o in out
                )
                if ok:
                    break
            c = rng.uniform(*crange)
            conductive = bool(rng.uniform() < 0.5)
            sigma = self.sigma_bg * (c if conductive else 1.0 / c)
            out.append({"cx": cx, "cy": cy, "a": a, "b": b, "angle": ang, "sigma": sigma})
        return out

    def render(self, inclusions: list[dict], shape: Sequence[int] | None = None) -> torch.Tensor:
        """σ on the grid ``shape`` (float64), area-weighted at inclusion boundaries."""
        dom = self.domain if shape is None else self.domain.at(shape)
        sigma = torch.full(dom.shape, self.sigma_bg, dtype=torch.float64)
        for inc in inclusions:
            c, s = math.cos(inc["angle"]), math.sin(inc["angle"])

            def inside(x, y, inc=inc, c=c, s=s):
                dx, dy = x - inc["cx"], y - inc["cy"]
                u, v = c * dx + s * dy, -s * dx + c * dy
                return (u / inc["a"]) ** 2 + (v / inc["b"]) ** 2 <= 1.0

            frac = cell_coverage(dom, None, inside, self.subsample)
            sigma = sigma + (inc["sigma"] - self.sigma_bg) * frac
        return sigma

    def sample(self, rng, cls=None, shape=None):
        cls = self.check_class(cls)
        return {"sigma": self.render(self.inclusions(rng, cls), shape).float()}


def inclusion_iou(pred, gt, sigma_bg: float = 1.0, fraction: float = 0.5) -> float:
    """IoU of the recovered inclusion support (half-maximum, sign-aware, log-contrast).

    Each map is thresholded at ``fraction ×`` its *own* peak log-contrast ``max |log(σ/σ_bg)|``
    (the half-maximum criterion; EIT reconstructions systematically underestimate the contrast,
    which PSNR/MSE measure separately); conductive and resistive supports are matched separately.
    """
    lp = torch.log(torch.as_tensor(pred).double().clamp_min(1e-12) / sigma_bg)
    lg = torch.log(torch.as_tensor(gt).double().clamp_min(1e-12) / sigma_bg)
    tg = fraction * float(lg.abs().max())
    tp = fraction * float(lp.abs().max())
    if tg <= 0:
        return 1.0 if tp <= 1e-6 else 0.0
    gp, gn = lg > tg, lg < -tg
    pp, pn = lp > tp, lp < -tp
    inter = (gp & pp).sum() + (gn & pn).sum()
    union = (gp | pp).sum() + (gn | pn).sum()
    return float(inter) / float(union)
