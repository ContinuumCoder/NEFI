"""DOT data generation (fine-grid physics, multiplicative noise, source–detector mask) and the
relative-residual data loss."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ...bench.base import DataGenerator
from ...losses.base import Context
from ...losses.data import DataLoss
from ...measurement import Measurement
from ...operators.base import Operator

__all__ = ["DOTDataGenerator", "RelativeResidualMSE", "separation_mask"]


def separation_mask(
    sources: torch.Tensor, detectors: torch.Tensor, min_separation: float
) -> torch.Tensor:
    """``(n_sources, n_dx, n_dy)`` mask of source–detector pairs with a lateral separation of at
    least ``min_separation`` (the diffusion approximation fails close to the source, and the huge
    near-source dynamic range would dominate any fit)."""
    s = torch.as_tensor(sources, dtype=torch.float64)[:, None, None, :2]
    d = torch.as_tensor(detectors, dtype=torch.float64)[None]
    return ((s - d).norm(dim=-1) >= float(min_separation)).to(torch.float64)


class DOTDataGenerator(DataGenerator):
    """Fine-grid CW DOT data with multiplicative (shot-noise-like) Gaussian noise.

    ``y = y_clean (1 + σ_rel ε)``, ``ε ~ N(0, 1)`` on the observed source–detector pairs; the
    observable is resolution independent (detector footprints), so the fine-grid simulation
    needs no resampling. The measurement stores ``data · mask``, the mask, the per-entry absolute
    noise std ``σ_rel |y_clean|`` and ``meta["relative_noise"]``.

    Args:
        operator: the fine-grid :class:`~nefi.instances.dot3d.operator.DiffuseOpticalOperator`.
        noise_std: relative noise level ``σ_rel``.
        min_separation: minimum lateral source–detector separation of the observed pairs (mm).
        supersample / fidelity_tag / dtype / device: see :class:`~nefi.bench.DataGenerator`.
    """

    def __init__(
        self,
        operator: Operator,
        noise_std: float = 0.01,
        min_separation: float = 6.0,
        supersample: int = 2,
        fidelity_tag: str = "dot-fv-robin-2x-float64",
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            operator,
            noise_std=noise_std,
            relative=True,
            dtype=dtype,
            device=device,
            supersample=supersample,
            fidelity_tag=fidelity_tag,
        )
        self.min_separation = float(min_separation)

    def generate(
        self,
        gt_fields: dict[str, torch.Tensor],
        rng: np.random.Generator,
        noise_std: float | None = None,
        target_shape: Sequence[int] | None = None,
    ) -> Measurement:
        clean = self.clean(gt_fields)
        op = self.operator
        mask = separation_mask(op.sources, op.detectors, self.min_separation).to(clean)
        rel = self.noise_std if noise_std is None else float(noise_std)
        eps = torch.as_tensor(rng.standard_normal(tuple(clean.shape)), dtype=clean.dtype)
        data = clean * (1.0 + rel * eps) * mask
        sigma = rel * clean.abs() * mask
        return Measurement(
            data.float(),
            mask.float(),
            noise_std=sigma.float() if rel > 0 else None,
            meta={
                "fidelity": self.fidelity_tag,
                "supersample": self.supersample,
                "relative_noise": rel,
                "min_separation": self.min_separation,
            },
        )


class RelativeResidualMSE(DataLoss):
    """``mean_obs [(pred − obs) / |obs|]²`` — the relative (per-entry normalized) residual.

    The weighted least-squares form of multiplicative noise and the standard CW-DOT fidelity for
    readings spanning decades (near vs far source–detector pairs); ``floor`` (relative to the
    largest reading) guards the division.
    """

    def __init__(self, floor: float = 1e-6, name: str | None = None) -> None:
        super().__init__(name)
        self.floor = float(floor)

    def forward(self, ctx: Context) -> torch.Tensor:
        obs = ctx.obs.data
        a = obs.abs()
        scale = torch.maximum(a, self.floor * a.amax())
        return ctx.obs.masked_mean(((ctx.pred - obs) / scale) ** 2)
