"""Benchmark primitives: ground-truth scene generators and inverse-crime-safe data generators."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ..domain import Domain
from ..errors import ConfigError
from ..measurement import Measurement
from ..operators.base import Operator
from ..utils.tensor import shape_tuple


class SceneGenerator:
    """Samples ground-truth fields for a domain, optionally by difficulty class.

    Subclasses implement :meth:`sample` returning ``{name: Tensor(*shape)}``. Analytic scenes should
    honour the ``shape`` argument so data generators can evaluate them on a finer grid.
    """

    classes: tuple[str, ...] = ("default",)

    def __init__(self, domain: Domain) -> None:
        self.domain = domain

    def sample(
        self, rng: np.random.Generator, cls: str | None = None, shape: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError

    def sample_many(
        self, n: int, seed: int = 0, cls: str | None = None, shape: Sequence[int] | None = None
    ) -> list[dict[str, torch.Tensor]]:
        rng = np.random.default_rng(seed)
        return [self.sample(rng, cls, shape) for _ in range(n)]

    def check_class(self, cls: str | None) -> str:
        cls = cls or self.classes[0]
        if cls not in self.classes:
            raise ConfigError(f"unknown scene class {cls!r}; known: {self.classes}")
        return cls


class DataGenerator:
    """Turns ground-truth fields into a :class:`Measurement` with an *independent* forward model.

    To avoid the inverse crime (Kaipio & Somersalo 2007; NeTMY §3.1, NeFTY §5.1), the operator used
    here must differ from the inversion operator: a higher-fidelity physics (NeTMY F3 vs F2), a
    different discretization/integrator (NeFTY explicit PhiFlow vs implicit Euler), or at least a
    finer grid (``supersample``). :func:`nefi.bench.protocol` checks ``fidelity_tag`` differs.

    Args:
        operator: forward model used for data generation.
        noise_std: additive Gaussian noise std; if ``relative`` it multiplies the clean signal's
        max.
        relative: interpret ``noise_std`` relative to ``max|clean|``.
        dtype / device: simulation precision (float64 on CPU by default — MPS lacks float64).
        supersample: evaluate the scene at ``supersample ×`` the field resolution, then average
            down the measurement's trailing spatial dims to the native measurement shape.
        fidelity_tag: string identifying the physics/discretization (compared against the inversion
            operator's tag in the benchmark protocol).
    """

    fidelity_tag: str = "independent"

    def __init__(
        self,
        operator: Operator,
        noise_std: float = 0.0,
        relative: bool = True,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        supersample: int = 1,
        fidelity_tag: str | None = None,
    ) -> None:
        self.operator = operator.to(device=device, dtype=dtype)
        self.noise_std, self.relative = float(noise_std), relative
        self.dtype, self.device = dtype, torch.device(device)
        self.supersample = int(supersample)
        if fidelity_tag:
            self.fidelity_tag = fidelity_tag

    def clean(self, gt_fields: dict[str, torch.Tensor]) -> torch.Tensor:
        fields = {k: torch.as_tensor(v).to(self.device, self.dtype) for k, v in gt_fields.items()}
        shape = tuple(next(iter(fields.values())).shape)
        with torch.no_grad():
            return self.operator.at_resolution(shape)(fields)

    def add_noise(
        self, clean: torch.Tensor, rng: np.random.Generator, noise_std: float | None = None
    ) -> tuple[torch.Tensor, float]:
        ns = self.noise_std if noise_std is None else float(noise_std)
        if self.relative:
            ns = ns * float(clean.abs().max())
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
        target_shape: Sequence[int] | None = None,
    ) -> Measurement:
        """Measurement for ``gt_fields`` (sampled at the generator's resolution).

        ``target_shape`` (the inversion-time measurement shape) triggers area-averaging of the
        supersampled simulation.
        """
        clean = self.clean(gt_fields)
        if target_shape is not None and tuple(clean.shape) != shape_tuple(target_shape):
            from ..utils.tensor import resample

            clean = resample(clean, shape_tuple(target_shape))
        data, ns = self.add_noise(clean, rng, noise_std)
        return Measurement(
            data.float(),
            noise_std=ns if ns > 0 else None,
            meta={"fidelity": self.fidelity_tag, "supersample": self.supersample},
        )
