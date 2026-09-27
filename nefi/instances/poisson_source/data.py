"""Sparse / boundary observation data for the Poisson source problem, and masked downsampling."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ...bench.base import DataGenerator
from ...errors import ConfigError
from ...measurement import Measurement
from ...operators.base import Operator
from ...utils.tensor import resample, shape_tuple

OBS_MODES = ("random", "boundary", "full")


def observation_mask(
    shape: Sequence[int],
    rng: np.random.Generator,
    mode: str = "random",
    fraction: float = 0.1,
    boundary_width: int = 3,
) -> torch.Tensor:
    """Binary observation mask: a random pixel subset, a boundary strip, or everything.

    Args:
        shape: grid shape.
        rng: numpy generator (random mode).
        mode: ``"random"`` (``round(fraction · N)`` pixels without replacement), ``"boundary"``
            (pixels within ``boundary_width`` cells of the domain boundary) or ``"full"``.
        fraction: observed fraction for ``"random"``.
        boundary_width: strip width in cells for ``"boundary"``.
    """
    shape = shape_tuple(shape)
    if mode == "full":
        return torch.ones(shape)
    if mode == "random":
        n = int(np.prod(shape))
        k = max(1, int(round(fraction * n)))
        m = np.zeros(n)
        m[rng.choice(n, size=k, replace=False)] = 1.0
        return torch.as_tensor(m.reshape(shape), dtype=torch.float32)
    if mode == "boundary":
        grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
        dist = np.min([np.minimum(g, s - 1 - g) for g, s in zip(grids, shape)], axis=0)
        return torch.as_tensor(dist < int(boundary_width), dtype=torch.float32)
    raise ConfigError(f"unknown observation mode {mode!r}; known: {OBS_MODES}")


def masked_downsample(measurement: Measurement, shape: Sequence[int]) -> Measurement:
    """Mask-aware area downsampling for coarse curriculum stages.

    The coarse datum of a cell is the average of its *observed* fine pixels and the coarse mask is
    the observed fraction (a weight in ``[0, 1]`` used by masked means), instead of thresholding
    an area-averaged binary mask (which would drop most cells of a sparse random mask).
    """
    shape = shape_tuple(shape)
    if measurement.mask is None:
        return measurement.resampled(shape)
    m = measurement.mask.expand_as(measurement.data).to(measurement.data.dtype)
    num = resample(measurement.data * m, shape, mode="area")
    den = resample(m, shape, mode="area")
    data = torch.where(den > 0, num / den.clamp_min(1e-12), torch.zeros_like(num))
    return Measurement(data, den, measurement.noise_std, dict(measurement.meta))


class PoissonDataGenerator(DataGenerator):
    """Finite-difference solve on a finer grid (float64), area-average, sample a mask, add noise.

    Args:
        operator: fine-grid forward operator (normally :class:`FDPoissonOperator`).
        noise_std: Gaussian noise std (relative to ``max |u|`` if ``relative``).
        obs_mode / obs_fraction / boundary_width: see :func:`observation_mask`.
        relative / dtype / device / supersample / fidelity_tag: see
            :class:`~nefi.bench.DataGenerator`.
    """

    def __init__(
        self,
        operator: Operator,
        noise_std: float = 0.01,
        obs_mode: str = "random",
        obs_fraction: float = 0.1,
        boundary_width: int = 3,
        relative: bool = True,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        supersample: int = 2,
        fidelity_tag: str = "poisson-fd-2x-float64",
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
        if obs_mode not in OBS_MODES:
            raise ConfigError(f"unknown observation mode {obs_mode!r}; known: {OBS_MODES}")
        self.obs_mode = obs_mode
        self.obs_fraction = float(obs_fraction)
        self.boundary_width = int(boundary_width)

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
        mask = observation_mask(
            clean.shape, rng, self.obs_mode, self.obs_fraction, self.boundary_width
        ).to(clean)
        noisy, ns = self.add_noise(clean, rng, noise_std)
        return Measurement(
            (noisy * mask).float(),
            mask=mask.float(),
            noise_std=ns if ns > 0 else None,
            meta={
                "fidelity": self.fidelity_tag,
                "supersample": self.supersample,
                "obs_mode": self.obs_mode,
                "obs_fraction": float(mask.mean()),
            },
        )


__all__ = ["OBS_MODES", "PoissonDataGenerator", "masked_downsample", "observation_mask"]
