"""Band-limited trace misfit for multiscale (frequency-continuation) full-waveform inversion.

FWI with the L2 trace misfit suffers from **cycle skipping**: if the initial model predicts an
arrival more than half a period late/early, the misfit is locally minimized by matching the *wrong*
cycle, so gradient descent converges to a spurious model (Virieux & Operto 2009, §"Multiscale").
The classic remedy (Bunks et al. 1995, *Geophysics* 60:1457) is **frequency continuation**: invert
low-pass filtered traces first (long periods → wide basin of attraction), then progressively add
higher frequencies. :class:`BandLimitedMSE` implements that misfit; the instance switches between
band-limited terms with ``Stage.loss_weights``. The neural field's annealed Fourier features add the
complementary *model-space* continuation (coarse, smooth updates first; NeTMY Eq. 27), and coarse
curriculum grids are only asked to explain the low-frequency data they can resolve.
"""

from __future__ import annotations

import math

import torch

from ...losses.base import Context
from ...losses.data import DataLoss
from ...registry import register
from ...utils.tensor import next_fast_len

__all__ = ["BandLimitedMSE", "lowpass"]


def _window(n_fft: int, dt: float, f_max: float, taper: float, device, dtype) -> torch.Tensor:
    f = torch.fft.rfftfreq(n_fft, d=dt, dtype=torch.float64)
    f1 = (1.0 - taper) * f_max
    w = torch.ones_like(f)
    roll = (f > f1) & (f < f_max)
    w[roll] = 0.5 * (1.0 + torch.cos(math.pi * (f[roll] - f1) / max(f_max - f1, 1e-30)))
    w[f >= f_max] = 0.0
    return w.to(device=device, dtype=dtype)


def lowpass(x: torch.Tensor, dt: float, f_max: float, taper: float = 0.3) -> torch.Tensor:
    """Zero-phase low-pass filter along the last (time) axis with a cosine roll-off.

    The pass band is flat up to ``(1 − taper) f_max`` and rolls off to zero at ``f_max`` (Hz); the
    signal is zero-padded to twice its length so the filter does not wrap around.
    """
    n = x.shape[-1]
    n_fft = next_fast_len(2 * n)
    w = _window(n_fft, dt, f_max, taper, x.device, x.dtype)
    return torch.fft.irfft(torch.fft.rfft(x, n=n_fft, dim=-1) * w, n=n_fft, dim=-1)[..., :n]


@register("loss", "band_limited_mse")
class BandLimitedMSE(DataLoss):
    """(Relative) MSE between low-pass filtered predicted and observed traces.

    ``L = mean(|lowpass(pred − obs)|²) / mean(|lowpass(obs)|²)`` (``relative=True``, scale-free).

    Args:
        f_max: cutoff frequency (Hz); ``None`` → full band (plain relative MSE).
        dt: trace sampling interval (s).
        taper: relative width of the cosine roll-off below ``f_max``.
        relative: divide by the energy of the filtered observation.
    """

    def __init__(
        self,
        f_max: float | None,
        dt: float,
        taper: float = 0.3,
        relative: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.f_max = None if f_max is None else float(f_max)
        self.dt = float(dt)
        self.taper = float(taper)
        self.relative = relative

    def forward(self, ctx: Context) -> torch.Tensor:
        r = ctx.pred - ctx.obs.data
        o = ctx.obs.data
        if self.f_max is not None:
            r = lowpass(r, self.dt, self.f_max, self.taper)
            o = lowpass(o, self.dt, self.f_max, self.taper) if self.relative else o
        num = ctx.obs.masked_mean(r**2)
        if not self.relative:
            return num
        return num / ctx.obs.masked_mean(o**2).clamp_min(1e-30)

    def extra_repr(self) -> str:
        return f"f_max={self.f_max}, dt={self.dt}, relative={self.relative}"
