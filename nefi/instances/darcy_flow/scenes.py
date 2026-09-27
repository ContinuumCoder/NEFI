"""Darcy-flow ground-truth permeability fields (log-permeability ``Y = log k``).

Classes:

* ``smooth``   — log-normal permeability: ``Y = μ + s·G`` with ``G`` a unit-variance Gaussian random
  field of correlation length ``ℓ`` (squared-exponential covariance), the classical geostatistical
  prior (Dagan 1989; Gelhar 1993);
* ``channels`` — a smoother, weaker background plus 1–3 sinuous high-permeability channels
  (fluvial facies), a standard hard case for smooth priors (Zhou, Gómez-Hernández & Li 2011).

Both are closed-form functions of position (random Fourier features, analytic channel
centrelines), so the fine data-generation grid and the native inversion grid see the same field.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from ...bench.base import SceneGenerator
from ...domain import Domain
from .._elliptic_common import random_fourier_field

__all__ = ["DarcyScenes", "high_k_iou"]


class DarcyScenes(SceneGenerator):
    """Log-permeability scenes (see module docstring).

    Args:
        domain: 2-D reservoir domain.
        mean: mean log-permeability ``μ``.
        std: log-permeability standard deviation ``s`` of the ``smooth`` class.
        corr_length: correlation length ``ℓ`` (fraction of the domain side).
        n_modes: random Fourier modes of the Gaussian random field.
        channel_contrast: log-permeability increase inside channels.
        channel_bg_std: background std of the ``channels`` class.
        n_channels: ``(lo, hi)`` number of channels (inclusive).
        channel_width: ``(lo, hi)`` channel width (fraction of the domain side).
        channel_amplitude: ``(lo, hi)`` meander amplitude (fraction of the side).
        channel_wavelength: ``(lo, hi)`` meander wavelength (fraction of the side).
        clip: optional ``(lo, hi)`` clamp of ``Y`` (keeps scenes inside the field's bounds).
    """

    classes = ("smooth", "channels")

    def __init__(
        self,
        domain: Domain,
        mean: float = 0.0,
        std: float = 0.8,
        corr_length: float = 0.15,
        n_modes: int = 128,
        channel_contrast: float = 2.5,
        channel_bg_std: float = 0.4,
        n_channels: Sequence[int] = (1, 3),
        channel_width: Sequence[float] = (0.05, 0.08),
        channel_amplitude: Sequence[float] = (0.05, 0.12),
        channel_wavelength: Sequence[float] = (0.4, 0.8),
        clip: Sequence[float] | None = (-2.8, 3.8),
    ) -> None:
        super().__init__(domain)
        self.mean, self.std = float(mean), float(std)
        self.corr_length, self.n_modes = float(corr_length), int(n_modes)
        self.channel_contrast = float(channel_contrast)
        self.channel_bg_std = float(channel_bg_std)
        self.n_channels = tuple(int(v) for v in n_channels)
        self.channel_width = tuple(map(float, channel_width))
        self.channel_amplitude = tuple(map(float, channel_amplitude))
        self.channel_wavelength = tuple(map(float, channel_wavelength))
        self.clip = None if clip is None else tuple(map(float, clip))
        (x0, x1), (y0, y1) = domain.extent
        self.side = min(x1 - x0, y1 - y0)
        self.center = (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    def _channels(self, rng: np.random.Generator) -> list[dict]:
        n = int(rng.integers(self.n_channels[0], self.n_channels[1] + 1))
        L = self.side
        return [
            {
                "psi": rng.uniform(0.0, math.pi),
                "offset": rng.uniform(-0.3, 0.3) * L,
                "amp": rng.uniform(*self.channel_amplitude) * L,
                "wavelength": rng.uniform(*self.channel_wavelength) * L,
                "phase": rng.uniform(0.0, 2 * math.pi),
                "width": rng.uniform(*self.channel_width) * L,
            }
            for _ in range(n)
        ]

    @staticmethod
    def _channel_indicator(xy: torch.Tensor, ch: dict, center) -> torch.Tensor:
        c, s = math.cos(ch["psi"]), math.sin(ch["psi"])
        dx, dy = xy[..., 0] - center[0], xy[..., 1] - center[1]
        u, v = c * dx + s * dy, -s * dx + c * dy
        k = 2 * math.pi / ch["wavelength"]
        vc = ch["offset"] + ch["amp"] * torch.sin(k * u + ch["phase"])
        slope = ch["amp"] * k * torch.cos(k * u + ch["phase"])
        d = (v - vc).abs() / torch.sqrt(1.0 + slope**2)  # first-order distance to the centreline
        w = ch["width"]
        return 0.5 * (1.0 - torch.tanh((d - 0.5 * w) / (0.15 * w)))

    def sample(self, rng, cls=None, shape=None):
        cls = self.check_class(cls)
        grf = random_fourier_field(rng, self.n_modes, self.corr_length * self.side)
        chans = self._channels(rng) if cls == "channels" else []
        dom = self.domain if shape is None else self.domain.at(shape)
        xy = dom.physical_coords(dtype=torch.float64)
        if cls == "smooth":
            y = self.mean + self.std * grf(xy)
        else:
            y = self.mean + self.channel_bg_std * grf(xy)
            ind = torch.zeros(dom.shape, dtype=torch.float64)
            for ch in chans:
                ind = torch.maximum(ind, self._channel_indicator(xy, ch, self.center))
            y = y + self.channel_contrast * ind
        if self.clip is not None:
            y = y.clamp(*self.clip)
        return {"log_k": y.float()}


def high_k_iou(pred, gt, n_std: float = 1.0) -> float:
    """IoU of the high-permeability region ``Y > mean(Y_gt) + n_std·std(Y_gt)`` (flow paths)."""
    p = torch.as_tensor(pred).double()
    g = torch.as_tensor(gt).double()
    thr = float(g.mean() + n_std * g.std())
    a, b = p > thr, g > thr
    union = (a | b).sum()
    return 1.0 if int(union) == 0 else float((a & b).sum()) / float(union)
