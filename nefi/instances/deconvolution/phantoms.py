"""Procedural 2-D phantom primitives shared by the imaging instances.

Scenes are *analytic*: random parameters are drawn first (independently of the requested grid) and
the resulting continuous image is rendered as **cell averages** on any grid. Rendering evaluates the
scene on a sub-grid whose resolution is fixed relative to the *native* grid (``render_factor ×``
native per axis) and area-averages it, so the native ground truth and a supersampled data-generation
grid are exact area-averages of one and the same point-sampled image (no aliasing mismatch between
ground truth and data). Used by ``deconvolution``, ``sparse_view_ct`` and ``poisson_source``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ...domain import Domain
from ...errors import ShapeError
from ...utils.tensor import shape_tuple

#: a scene maps physical coordinate grids ``(X, Y)`` (float64) to an image of the same shape.
SceneFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def render(
    fn: SceneFn,
    domain: Domain,
    shape: Sequence[int] | None = None,
    render_factor: int = 4,
    normalized: bool = False,
) -> torch.Tensor:
    """Cell-averaged rendering of a continuous 2-D scene (float64).

    Args:
        fn: scene function of physical (or normalized, see ``normalized``) coordinates.
        domain: the native domain (fixes the sub-grid resolution).
        shape: target grid (default native); must divide ``render_factor × native`` or be finer.
        render_factor: sub-grid resolution relative to the native grid.
        normalized: pass normalized ``[-1, 1]`` coordinates instead of physical ones.
    """
    if domain.ndim != 2:
        raise ShapeError(f"phantoms are 2-D, got a {domain.ndim}-D domain")
    shape = tuple(domain.shape) if shape is None else shape_tuple(shape)
    ss = [max(1, math.ceil(render_factor * n0 / n)) for n0, n in zip(domain.shape, shape)]
    fine = tuple(n * s for n, s in zip(shape, ss))
    grid = domain.at(fine)
    c = (
        grid.coords(dtype=torch.float64)
        if normalized
        else grid.physical_coords(dtype=torch.float64)
    )
    img = fn(c[..., 0], c[..., 1])
    if ss != [1, 1]:
        img = F.avg_pool2d(img[None, None], kernel_size=tuple(ss))[0, 0]
    return img


# ---- primitives (indicator / smooth functions of coordinate grids) ----------------------------
def _rot(x: torch.Tensor, y: torch.Tensor, cx: float, cy: float, theta: float):
    c, s = math.cos(theta), math.sin(theta)
    dx, dy = x - cx, y - cy
    return c * dx + s * dy, -s * dx + c * dy


def ellipse(x, y, cx: float, cy: float, a: float, b: float, theta: float = 0.0) -> torch.Tensor:
    """Indicator of an ellipse with semi-axes ``a``, ``b`` rotated by ``theta`` (radians)."""
    u, v = _rot(x, y, cx, cy, theta)
    return ((u / a) ** 2 + (v / b) ** 2 <= 1.0).to(x.dtype)


def disk(x, y, cx: float, cy: float, r: float) -> torch.Tensor:
    return ellipse(x, y, cx, cy, r, r)


def rectangle(x, y, cx: float, cy: float, w: float, h: float, theta: float = 0.0) -> torch.Tensor:
    """Indicator of a ``w × h`` rectangle centered at ``(cx, cy)`` rotated by ``theta``."""
    u, v = _rot(x, y, cx, cy, theta)
    return ((u.abs() <= 0.5 * w) & (v.abs() <= 0.5 * h)).to(x.dtype)


def gaussian(
    x, y, cx: float, cy: float, sx: float, sy: float | None = None, theta: float = 0.0
) -> torch.Tensor:
    """Unit-peak anisotropic Gaussian."""
    sy = sx if sy is None else sy
    u, v = _rot(x, y, cx, cy, theta)
    return torch.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))


class Painter:
    """Accumulates primitives into a scene: additive terms and painted (overwriting) regions.

    Args:
        background: constant background value.
        clip: optional ``(lo, hi)`` range applied to the *point-sampled* image (before cell
            averaging), so clipped scenes stay consistent across rendering resolutions.
    """

    def __init__(
        self, background: float = 0.0, clip: tuple[float | None, float | None] | None = (0.0, 1.0)
    ) -> None:
        self.background = float(background)
        self.clip = clip
        self.ops: list[tuple[str, float, Callable]] = []

    def add(self, value: float, fn: Callable) -> Painter:
        self.ops.append(("add", float(value), fn))
        return self

    def paint(self, value: float, fn: Callable) -> Painter:
        self.ops.append(("paint", float(value), fn))
        return self

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        img = torch.full_like(x, self.background)
        for mode, value, fn in self.ops:
            m = fn(x, y)
            img = img + value * m if mode == "add" else img * (1.0 - m) + value * m
        if self.clip is not None:
            img = img.clamp(min=self.clip[0], max=self.clip[1])
        return img


def point_max(fn: SceneFn, domain: Domain, render_factor: int = 4, normalized: bool = False):
    """Maximum of the point-sampled rendering sub-grid (grid-independent normalization)."""
    fine = tuple(n * render_factor for n in domain.shape)
    return float(render(fn, domain, fine, render_factor, normalized).max())


def uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


__all__ = [
    "Painter",
    "SceneFn",
    "disk",
    "ellipse",
    "gaussian",
    "point_max",
    "rectangle",
    "render",
    "uniform",
]
