"""Measurement generation for NV relaxometry (NeTMY App. E.1).

:class:`NVDataGenerator` wraps an *independent* forward model — by default the float64 source-side
direct simulator F3 (:class:`~.operator.NVDirectSimulator`) — and adds white Gaussian sensor noise
``ε ~ N(0, σ²)`` with ``σ`` relative to the per-sample dynamic range of the clean spectrum
("sensor noise σ_ε² matched to per-sample dynamic range", App. E.1).
"""

from __future__ import annotations

import numpy as np
import torch

from ...bench.base import DataGenerator
from ...measurement import Measurement
from ...utils.tensor import resample, shape_tuple


class NVDataGenerator(DataGenerator):
    """Data generator with noise relative to ``max(S) − min(S)`` of each clean spectrum.

    Args:
        operator: data-generation operator (F3 for the cross-fidelity benchmark; F2/F1 for the
            inverse-crime "matched-operator" benchmark of App. E.4).
        noise_std: noise level; relative to the dynamic range if ``relative`` else absolute.
        relative: interpret ``noise_std`` relative to the clean spectrum's dynamic range.
        fidelity_tag: physics/discretization tag compared by the benchmark protocol.
        meta: extra metadata stored in every generated :class:`~nefi.measurement.Measurement`.
    """

    def __init__(
        self,
        operator,
        noise_std: float = 0.01,
        relative: bool = True,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        fidelity_tag: str | None = None,
        meta: dict | None = None,
    ) -> None:
        super().__init__(
            operator,
            noise_std=noise_std,
            relative=relative,
            dtype=dtype,
            device=device,
            supersample=1,
            fidelity_tag=fidelity_tag or getattr(operator, "fidelity_tag", None),
        )
        self.meta = dict(meta or {})

    def add_noise(
        self, clean: torch.Tensor, rng: np.random.Generator, noise_std: float | None = None
    ) -> tuple[torch.Tensor, float]:
        ns = self.noise_std if noise_std is None else float(noise_std)
        if self.relative:
            ns = ns * float(clean.max() - clean.min())
        if ns <= 0:
            return clean, 0.0
        noise = torch.as_tensor(
            rng.standard_normal(tuple(clean.shape)), device=clean.device, dtype=clean.dtype
        )
        return clean + ns * noise, ns

    def generate(
        self,
        gt_fields: dict[str, torch.Tensor],
        rng: np.random.Generator,
        noise_std: float | None = None,
        target_shape=None,
    ) -> Measurement:
        clean = self.clean(gt_fields)
        if target_shape is not None and tuple(clean.shape) != shape_tuple(target_shape):
            clean = resample(clean, shape_tuple(target_shape))
        data, ns = self.add_noise(clean, rng, noise_std)
        meta = {"fidelity": self.fidelity_tag, **self.meta}
        return Measurement(data.float(), noise_std=ns if ns > 0 else None, meta=meta)


def downsample_spectrum(measurement: Measurement, shape) -> Measurement:
    """Area-average the spatial dims of a ``(n_freq, H, W)`` spectrum to ``shape = (h, w)``.

    Used as ``InverseProblem.downsample_obs`` so coarse curriculum stages compare against a
    spatially area-averaged observation (NeTMY App. D.3). The known noise level is rescaled by the
    number of averaged pixels (exact for integer factors).
    """
    shape = shape_tuple(shape)
    if len(shape) == 3:  # accept the full output shape (n_freq, h, w)
        shape = shape[1:]
    h0, w0 = measurement.data.shape[-2:]
    if (h0, w0) == shape:
        return measurement
    m = measurement.resampled(shape)
    ns = measurement.noise_std
    if ns is not None:  # averaging (H·W)/(h·w) pixels divides the noise std by its square root
        ns = ns * ((shape[0] * shape[1]) / float(h0 * w0)) ** 0.5
    return Measurement(m.data, m.mask, ns, dict(m.meta))


__all__ = ["NVDataGenerator", "downsample_spectrum"]
