"""Inverse-crime-safe blur data generation with Gaussian or Poisson (photon-count) noise."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ...bench.base import DataGenerator
from ...errors import ConfigError
from ...measurement import Measurement
from ...operators.base import Operator
from ...utils.tensor import resample, shape_tuple


class BlurDataGenerator(DataGenerator):
    """Blur on a finer grid in float64, area-average to the detector grid, then add noise.

    The generator's operator is the *same physics* (PSF kernel function) evaluated on a
    ``supersample ×`` finer grid, so it differs from the native-resolution inversion operator in
    discretization and precision (inverse-crime guard, Kaipio & Somersalo 2007).

    Args:
        operator: fine-grid blur operator (image units).
        noise: ``"gaussian"`` (additive, std ``noise_std`` × max of the clean image if
            ``relative``) or ``"poisson"`` (counts ``~ Poisson(peak_counts · clean + background)``).
        noise_std: Gaussian noise level.
        peak_counts: photon counts at unit image intensity (Poisson).
        background_counts: dark/background counts per pixel (Poisson).
        relative / dtype / device / supersample / fidelity_tag: see
            :class:`~nefi.bench.DataGenerator`.
    """

    def __init__(
        self,
        operator: Operator,
        noise: str = "gaussian",
        noise_std: float = 0.01,
        peak_counts: float = 200.0,
        background_counts: float = 0.0,
        relative: bool = True,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        supersample: int = 2,
        fidelity_tag: str = "blur-2x-float64",
    ) -> None:
        super().__init__(
            operator,
            noise_std=noise_std,
            relative=relative,
            dtype=dtype,
            device=device,
            supersample=supersample,
            fidelity_tag=fidelity_tag,
        )
        if noise not in ("gaussian", "poisson"):
            raise ConfigError(f"noise must be 'gaussian' or 'poisson', got {noise!r}")
        if noise == "poisson" and peak_counts <= 0:
            raise ConfigError("peak_counts must be > 0 for Poisson noise")
        self.noise = noise
        self.peak_counts = float(peak_counts)
        self.background_counts = float(background_counts)

    def generate(
        self,
        gt_fields: dict[str, torch.Tensor],
        rng: np.random.Generator,
        noise_std: float | None = None,
        target_shape: Sequence[int] | None = None,
    ) -> Measurement:
        clean = self.clean(gt_fields)
        if target_shape is not None and tuple(clean.shape) != shape_tuple(target_shape):
            clean = resample(clean, shape_tuple(target_shape), mode="area")
        meta = {
            "fidelity": self.fidelity_tag,
            "supersample": self.supersample,
            "noise": self.noise,
        }
        if self.noise == "gaussian":
            data, ns = self.add_noise(clean, rng, noise_std)
            return Measurement(data.float(), noise_std=ns if ns > 0 else None, meta=meta)
        lam = self.peak_counts * clean.clamp_min(0.0) + self.background_counts
        counts = rng.poisson(lam.detach().cpu().numpy())
        data = torch.as_tensor(counts, dtype=torch.float32)
        meta.update(peak_counts=self.peak_counts, background_counts=self.background_counts)
        return Measurement(data, noise_std=float(torch.sqrt(lam.mean())), meta=meta)


__all__ = ["BlurDataGenerator"]
