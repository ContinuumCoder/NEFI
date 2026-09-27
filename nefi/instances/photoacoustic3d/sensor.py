"""Finite sensor bandwidth: a zero-phase Gaussian low-pass of the pressure traces.

Ultrasound transducers integrate the pressure with a band-limited impulse response; modeling it
is standard in PAT (e.g. the k-Wave ``sensor.frequency_response``, Treeby & Cox 2010). Here the
response is the zero-phase Gaussian ``H(f) = exp(−½ (f / f_c)²)`` applied along the time axis
(zero-padded FFT, no wrap-around). It is part of the forward model of both the data generator and
the inversion, and it band-limits the traces to what the discrete wave solver propagates
accurately (at ``f_c`` = 1 MHz and 0.25 mm voxels the 2nd-order-native vs 4th-order-2×-grid model
mismatch drops from ≈ 4.5 % to ≈ 2 % of the peak trace).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ...operators.base import Fields, Operator
from ...utils.tensor import shape_tuple

__all__ = ["SensorBandwidth", "gaussian_lowpass"]


def gaussian_lowpass(traces: torch.Tensor, dt: float, f_cutoff: float | None) -> torch.Tensor:
    """Zero-phase Gaussian low-pass ``exp(−½ (f/f_c)²)`` along the last axis (sample step ``dt``).

    ``f_cutoff=None`` (or ≤ 0) returns the traces unchanged.
    """
    if f_cutoff is None or f_cutoff <= 0:
        return traces
    n = traces.shape[-1]
    size = 2 * n
    f = torch.fft.rfftfreq(size, d=float(dt), dtype=torch.float64)
    h = torch.exp(-0.5 * (f / float(f_cutoff)) ** 2).to(device=traces.device, dtype=traces.dtype)
    spec = torch.fft.rfft(traces, n=size, dim=-1)
    return torch.fft.irfft(spec * h, n=size, dim=-1)[..., :n]


class SensorBandwidth(Operator):
    """Wraps a trace operator ``(…, n_t)`` with the sensors' low-pass response.

    Args:
        inner: operator producing traces with time as the last axis (e.g.
            :class:`~nefi.physics.wave.WaveInitialConditionOperator`).
        dt_obs: trace sampling interval.
        f_cutoff: Gaussian cutoff frequency (same time unit as ``dt_obs``; ``None`` = ideal
            sensors).

    Linear if ``inner`` is; ``at_resolution`` re-wraps the inner operator at the new grid.
    """

    traceable = False

    def __init__(self, inner: Operator, dt_obs: float, f_cutoff: float | None) -> None:
        super().__init__()
        self.inner = inner
        self.primary = inner.primary
        self.homogeneity = inner.homogeneity
        self.dt_obs = float(dt_obs)
        self.f_cutoff = None if f_cutoff is None or f_cutoff <= 0 else float(f_cutoff)
        tag = getattr(inner, "fidelity_tag", type(inner).__name__)
        self.fidelity_tag = f"{tag}-lp{self.f_cutoff:g}" if self.f_cutoff else str(tag)
        self._res: dict[tuple[int, ...], SensorBandwidth] = {}

    def forward(self, fields: Fields) -> torch.Tensor:
        return gaussian_lowpass(self.inner(fields), self.dt_obs, self.f_cutoff)

    def at_resolution(self, shape: Sequence[int]) -> SensorBandwidth:
        shape = shape_tuple(shape)
        inner = self.inner.at_resolution(shape)
        if inner is self.inner:
            return self
        if shape not in self._res:
            op = SensorBandwidth(inner, self.dt_obs, self.f_cutoff)
            op.fidelity_tag = self.fidelity_tag
            self._res[shape] = op
        return self._res[shape]

    def output_shape(self, shape: Sequence[int]) -> tuple[int, ...] | None:
        return self.inner.output_shape(shape)

    def required_fields(self) -> tuple[str, ...]:
        return self.inner.required_fields()

    def extra_repr(self) -> str:
        return f"f_cutoff={self.f_cutoff}, dt_obs={self.dt_obs:g}"
