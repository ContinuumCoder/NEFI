"""toy1d — recover a 1-D non-negative signal from its blurred, noisy observation.

The five-second tutorial and CI smoke test. It exercises every core mechanism of nefi:
neural field with annealed Fourier features, softplus head, FFT convolution operator rebuilt per
resolution, two-stage multiscale curriculum, TV/L1 regularization, and an inverse-crime-safe data
generator (direct simulation on a 4× finer grid, area-averaged).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ...bench.base import DataGenerator, SceneGenerator
from ...domain import Domain
from ...fields import GridField, Heads, NeuralField, Softplus
from ...losses import L1, MSE, TV, LossSet
from ...measurement import Measurement
from ...metrics.basic import mse, psnr, relative_error
from ...operators.conv import FFTConvolution, gaussian_kernel_fn
from ...problem import InverseProblem
from ...registry import register
from ...solve.curriculum import Curriculum
from ..base import Instance


@dataclass
class Toy1DConfig:
    n: int = 128  # grid points on [0, 1]
    sigma: float = 0.02  # blur std in physical units
    noise_std: float = 0.01  # relative to the clean signal's max
    scene: str = "bumps"  # bumps | spikes | mixed
    supersample: int = 4  # data generated on a 4x finer grid (inverse-crime guard)
    hidden: int = 64
    depth: int = 4
    n_octaves: int = 6
    steps: tuple[int, int] = (300, 600)
    lr: float = 1e-2
    tv: float = 1e-4
    l1: float = 0.0


class Toy1DScenes(SceneGenerator):
    classes = ("bumps", "spikes", "mixed")

    def _bumps(self, rng, x):
        y = torch.zeros_like(x)
        for _ in range(rng.integers(2, 4)):
            c, w, a = rng.uniform(0.15, 0.85), rng.uniform(0.03, 0.08), rng.uniform(0.5, 1.5)
            y = y + a * torch.exp(-0.5 * ((x - c) / w) ** 2)
        return y

    def _spikes(self, rng, x):
        y = torch.zeros_like(x)
        w = 0.5 / self.domain.shape[0]  # ~half a native cell
        for _ in range(rng.integers(3, 6)):
            c, a = rng.uniform(0.1, 0.9), rng.uniform(0.5, 1.5)
            y = y + a * torch.exp(-0.5 * ((x - c) / w) ** 2)
        return y

    def sample(self, rng, cls=None, shape=None):
        cls = self.check_class(cls)
        x = self.domain.physical_coords(shape)[..., 0].double()
        if cls == "bumps":
            y = self._bumps(rng, x)
        elif cls == "spikes":
            y = self._spikes(rng, x)
        else:
            y = self._bumps(rng, x) + self._spikes(rng, x)
        return {"x": y.float()}


@register("instance", "toy1d")
class Toy1D(Instance):
    name = "toy1d"
    Config = Toy1DConfig
    description = "1-D blurred-signal recovery (tutorial / smoke test)"

    def domain(self) -> Domain:
        return Domain.unit((self.cfg.n,), axes=("x",))

    def scene_generator(self) -> SceneGenerator:
        return Toy1DScenes(self.domain())

    def operator(self, domain: Domain | None = None) -> FFTConvolution:
        return FFTConvolution(
            gaussian_kernel_fn(self.cfg.sigma), domain or self.domain(), field="x"
        )

    def data_generator(self) -> DataGenerator:
        fine = self.domain().refine(self.cfg.supersample)
        return DataGenerator(
            self.operator(fine),
            noise_std=self.cfg.noise_std,
            relative=True,
            supersample=self.cfg.supersample,
            fidelity_tag="gaussian-blur-4x-float64",
        )

    def build_problem_measurement_shape(self, field_shape):
        return tuple(field_shape)

    def field(self) -> NeuralField:
        c = self.cfg
        return NeuralField(
            1,
            Heads({"x": Softplus(init_value=0.1)}),
            hidden=c.hidden,
            depth=c.depth,
            skip_at=2,
            n_octaves=c.n_octaves,
        )

    def build_problem(self, measurement: Measurement) -> InverseProblem:
        c = self.cfg
        losses = LossSet(
            {"data": MSE(), "tv": TV("x", isotropic=False), "l1": L1("x")},
            weights={"data": 1.0, "tv": c.tv, "l1": c.l1},
        )
        return InverseProblem(
            self.domain(),
            self.field(),
            self.operator(),
            losses,
            measurement,
            curriculum=self.default_curriculum(),
            name="toy1d",
        )

    def default_curriculum(self) -> Curriculum:
        c = self.cfg
        return Curriculum.multiscale((c.n,), n_stages=2, steps=c.steps, lr=c.lr, lr_decay=0.5)

    def metrics(self) -> dict[str, Callable]:
        return {"mse": mse, "psnr": psnr, "relative_error": relative_error}

    def baselines(self):
        def grid(measurement):
            c = self.cfg
            losses = LossSet(
                {"data": MSE(), "tv": TV("x", isotropic=False), "l1": L1("x")},
                weights={"data": 1.0, "tv": c.tv, "l1": c.l1},
            )
            field = GridField((c.n,), Heads({"x": Softplus(init_value=0.1)}))
            cur = Curriculum.multiscale(
                (c.n,), n_stages=2, steps=c.steps, lr=c.lr * 10, lr_decay=0.5
            )
            prob = InverseProblem(
                self.domain(),
                field,
                self.operator(),
                losses,
                measurement,
                curriculum=cur,
                name="toy1d-grid",
            )
            return prob, cur

        return {"grid": grid}


def make_problem(
    n: int = 128, seed: int = 0, **cfg
) -> tuple[InverseProblem, dict[str, torch.Tensor], Measurement]:
    """Convenience: (problem, gt, measurement) for docs and tests."""
    inst = Toy1D(n=n, **cfg)
    gt, meas = inst.make_measurement(seed)
    return inst.build_problem(meas), gt, meas


__all__ = ["Toy1D", "Toy1DConfig", "Toy1DScenes", "make_problem"]
_ = (np, Sequence)  # keep imports referenced for type checkers
