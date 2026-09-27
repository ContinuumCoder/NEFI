"""reaction_diffusion — recover a spatially varying Gray–Scott feed rate ``F(x)`` from snapshots.

Unknown: the feed-rate field ``F(x)`` on ``[0, L]²`` (Pearson units), parameterized by a neural
field with a ``Bounded(F_min, F_max)`` head started at the background ``F_background``. Known: the
kill rate ``k``, the diffusivities, and a smooth initial condition that keeps the whole domain away
from the trivial state (where ``F`` would be unobservable). Data: snapshots of ``u`` and ``v`` at a
few times — ``(n_times · n_species, n, n)`` — from the explicit-Euler Gray–Scott solver of
:mod:`nefi.physics.reaction_diffusion` (nonlinear, diffusive coupling: each pixel's ``F`` shapes a
neighbourhood of radius ≈ sqrt(2 D t)).

Inverse-crime guard: the data generator runs 4× smaller time steps in float64
(``fidelity_tag="gray-scott-substep4-float64"`` vs ``"gray-scott-euler"``), an O(dt) discretization
gap of ≈ 1 % of the data. Scene classes: ``blobs`` and ``stripes``. Baseline: ``"grid"``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from ...bench.base import DataGenerator, SceneGenerator
from ...domain import Domain
from ...fields import Bounded, GridField, Heads, NeuralField
from ...losses import TV, LossSet, RelativeMSE
from ...measurement import Measurement
from ...metrics.basic import psnr, relative_error, ssim
from ...physics.reaction_diffusion import GrayScottIC, ReactionDiffusionOperator
from ...problem import InverseProblem
from ...registry import register
from ...solve.curriculum import Curriculum
from .._wave_family import spatial_downsample_obs
from ..base import Instance


@dataclass
class ReactionDiffusionConfig:
    """Configuration (defaults = CPU smoke preset: 32², 4 snapshots of u and v, 100 steps)."""

    n: int = 32  # grid (n × n)
    extent: float = 0.64  # domain side (Pearson length units)
    scene: str = "blobs"  # blobs | stripes
    F_background: float = 0.045  # background feed rate and initial model
    F_amplitude: float = 0.015  # peak |F − F_background| of the phantom
    F_min: float = 0.02  # Bounded head range (and stability bound F_max)
    F_max: float = 0.07
    k: float = 0.06  # kill rate (known)
    Du: float = 2e-5  # diffusivities (Pearson 1993)
    Dv: float = 1e-5
    dt: float = 1.0  # base time step (time unit of the reaction rates)
    obs_times: tuple[float, ...] = (25.0, 50.0, 75.0, 100.0)
    observe: tuple[str, ...] = ("u", "v")
    boundary: str = "neumann"  # neumann | periodic
    ic_modes: tuple[int, int] = (2, 3)  # initial-condition pattern (cos modes per axis)
    noise_std: float = 0.01  # Gaussian, relative to max |snapshot|
    gen_substeps: int = 4  # generator time-step refinement (inverse-crime guard)
    supersample: int = 1  # optional generator grid refinement
    grad_mode: str = "autograd"  # autograd | checkpoint
    checkpoint_every: int | None = None
    hidden: int = 64
    depth: int = 4
    n_octaves: int = 6
    activation: str = "tanh"
    tv: float = 1e-4
    steps: tuple[int, int] = (100, 150)
    lr: float = 1e-2
    lr_decay: float = 0.5
    grid_lr_mult: float = 10.0


class ReactionDiffusionScenes(SceneGenerator):
    """Feed-rate fields ``F(x)`` (analytic, any resolution)."""

    classes = ("blobs", "stripes")

    def __init__(self, domain: Domain, cfg: ReactionDiffusionConfig) -> None:
        super().__init__(domain)
        self.cfg = cfg

    def sample(self, rng: np.random.Generator, cls: str | None = None, shape=None):
        cls = self.check_class(cls)
        c = self.cfg
        L = max(self.domain.size)
        xy = self.domain.physical_coords(shape, dtype=torch.float64) / L
        x, y = xy[..., 0], xy[..., 1]
        dF = torch.zeros_like(x)
        if cls == "blobs":
            for _ in range(int(rng.integers(2, 5))):
                cx, cy = rng.uniform(0.2, 0.8, size=2)
                s = rng.uniform(0.08, 0.15)
                a = rng.uniform(0.5, 1.0) * rng.choice([-1.0, 1.0])
                dF = dF + a * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s**2))
            dF = dF / dF.abs().max().clamp_min(1e-12)
        else:  # stripes
            th = rng.uniform(0, math.pi)
            lam = rng.uniform(0.35, 0.55)
            ph = rng.uniform(0, 2 * math.pi)
            s = torch.sin(2 * math.pi * (x * math.cos(th) + y * math.sin(th)) / lam + ph)
            dF = torch.tanh(2.0 * s) / math.tanh(2.0)
        F = c.F_background + c.F_amplitude * dF
        F = F.clamp(c.F_min + 0.002, c.F_max - 0.002)
        return {"F": F.float()}


@register("instance", "reaction_diffusion")
class ReactionDiffusion(Instance):
    """Gray–Scott feed-rate inversion instance (see module docstring)."""

    name = "reaction_diffusion"
    Config = ReactionDiffusionConfig
    description = "Gray-Scott reaction-diffusion: recover the feed-rate field F(x) from snapshots"

    def domain(self) -> Domain:
        L = self.cfg.extent
        return Domain((self.cfg.n, self.cfg.n), ((0.0, L), (0.0, L)), axes=("x", "y"))

    def operator(
        self, domain: Domain | None = None, substeps: int = 1
    ) -> ReactionDiffusionOperator:
        c = self.cfg
        return ReactionDiffusionOperator(
            domain or self.domain(),
            c.obs_times,
            field="F",
            unknown="F",
            k=c.k,
            Du=c.Du,
            Dv=c.Dv,
            dt=c.dt,
            observe=c.observe,
            boundary=c.boundary,
            ic=GrayScottIC(modes=tuple(c.ic_modes)),
            param_max=c.F_max,
            substeps=substeps,
            grad_mode=c.grad_mode,
            checkpoint_every=c.checkpoint_every,
        )

    def scene_generator(self) -> SceneGenerator:
        return ReactionDiffusionScenes(self.domain(), self.cfg)

    def data_generator(self) -> DataGenerator:
        c = self.cfg
        fine = self.domain().refine(c.supersample) if c.supersample > 1 else self.domain()
        tag = f"gray-scott-substep{c.gen_substeps}-float64"
        if c.supersample > 1:
            tag += f"-{c.supersample}x"
        return DataGenerator(
            self.operator(fine, substeps=c.gen_substeps),
            noise_std=c.noise_std,
            relative=True,
            supersample=c.supersample,
            fidelity_tag=tag,
        )

    def build_problem_measurement_shape(self, field_shape):
        return (len(self.cfg.obs_times) * len(self.cfg.observe), *tuple(field_shape))

    def heads(self) -> Heads:
        c = self.cfg
        return Heads({"F": Bounded(c.F_min, c.F_max, init_value=c.F_background)})

    def field(self) -> NeuralField:
        c = self.cfg
        return NeuralField(
            2,
            self.heads(),
            hidden=c.hidden,
            depth=c.depth,
            skip_at=c.depth // 2,
            activation=c.activation,
            n_octaves=c.n_octaves,
        )

    def losses(self) -> LossSet:
        return LossSet(
            {"fit": RelativeMSE(), "tv": TV("F", isotropic=True)},
            weights={"fit": 1.0, "tv": self.cfg.tv},
        )

    def _problem(self, field, measurement, curriculum, name, meta=None) -> InverseProblem:
        return InverseProblem(
            self.domain(),
            field,
            self.operator(),
            self.losses(),
            measurement,
            curriculum=curriculum,
            downsample_obs=spatial_downsample_obs(2),
            name=name,
            meta=meta or {},
        )

    def build_problem(self, measurement: Measurement) -> InverseProblem:
        return self._problem(
            self.field(), measurement, self.default_curriculum(), "reaction_diffusion"
        )

    def default_curriculum(self, lr_mult: float = 1.0) -> Curriculum:
        c = self.cfg
        return Curriculum.multiscale(
            (c.n, c.n), n_stages=2, steps=c.steps, lr=c.lr * lr_mult, lr_decay=c.lr_decay
        )

    def metrics(self) -> dict[str, Callable]:
        return {"psnr": psnr, "ssim": ssim, "relative_error": relative_error}

    def initial_model(self) -> torch.Tensor:
        return torch.full((self.cfg.n, self.cfg.n), float(self.cfg.F_background))

    def baselines(self):
        def grid(measurement: Measurement):
            cur = self.default_curriculum(lr_mult=self.cfg.grid_lr_mult)
            field = GridField((self.cfg.n, self.cfg.n), self.heads())
            prob = self._problem(
                field, measurement, cur, "reaction_diffusion-grid", {"baseline": "grid"}
            )
            return prob, cur

        return {"grid": grid}


PRESETS: dict[str, dict] = {
    "smoke": {},
    "full": {
        "n": 128,
        "extent": 2.56,
        "obs_times": (25.0, 50.0, 75.0, 100.0, 150.0, 200.0),
        "grad_mode": "checkpoint",
        "hidden": 256,
        "depth": 6,
        "n_octaves": 7,
        "steps": (1500, 3500),
        "lr": 2e-3,
    },
}


def make_problem(seed: int = 0, **cfg):
    """Convenience: ``(problem, gt, measurement)``."""
    inst = ReactionDiffusion(**cfg)
    gt, meas = inst.make_measurement(seed)
    return inst.build_problem(meas), gt, meas


def run(cfg=None, seed: int = 0, **kw):
    """End-to-end demo (DESIGN §3.10): ``(result, metrics)``."""
    out = ReactionDiffusion(cfg).run(seed=seed, **kw)
    return out.result, out.metrics


__all__ = [
    "PRESETS",
    "ReactionDiffusion",
    "ReactionDiffusionConfig",
    "ReactionDiffusionScenes",
    "make_problem",
    "run",
]
