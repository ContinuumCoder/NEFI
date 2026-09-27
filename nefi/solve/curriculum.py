"""Optimization curriculum: multiscale stages, frequency annealing, learning-rate schedules.

NeTMY (Tab. 6): two stages 32² (3000 steps, η) → 64² (7000 steps, 0.5η), cosine LR to 0.01η per
stage, annealing β reset to 0 at each stage start. NeFTY (Tab. 5): single stage, 10k steps, step
decay ×0.1 / 1000, annealing over the first 2500 steps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..errors import ConfigError
from ..utils.tensor import shape_tuple


@dataclass
class Stage:
    """One curriculum stage (fixed resolution)."""

    name: str = "stage"
    shape: tuple[int, ...] | None = None
    steps: int = 1000
    lr: float = 1e-3
    lr_schedule: str = "cosine"  # cosine | step | constant | warmup_cosine
    lr_min_ratio: float = 0.01  # floor of cosine decay relative to lr
    lr_step_size: int = 1000  # for "step"
    lr_gamma: float = 0.1  # for "step"
    warmup_steps: int = 0  # for "warmup_cosine"
    anneal: bool = True  # reset β to 0 at stage start, ramp to K
    anneal_fraction: float = 1.0  # fraction of the stage over which β ramps
    loss_weights: dict[str, float] | None = None
    freeze: tuple[str, ...] = ()  # parameter-name prefixes frozen in this stage

    def __post_init__(self) -> None:
        if self.shape is not None:
            self.shape = shape_tuple(self.shape)
        if self.steps <= 0:
            raise ConfigError("Stage.steps must be positive")
        if not 0.0 < self.anneal_fraction <= 1.0:
            raise ConfigError("anneal_fraction must be in (0, 1]")

    def lr_at(self, step: int) -> float:
        t, T = step, max(1, self.steps)
        if self.lr_schedule == "constant":
            return self.lr
        if self.lr_schedule == "step":
            return self.lr * (self.lr_gamma ** (t // max(1, self.lr_step_size)))
        if self.lr_schedule == "warmup_cosine":
            if t < self.warmup_steps:
                return self.lr * (t + 1) / self.warmup_steps
            t, T = t - self.warmup_steps, max(1, T - self.warmup_steps)
        if self.lr_schedule in ("cosine", "warmup_cosine"):
            lo = self.lr * self.lr_min_ratio
            return lo + (self.lr - lo) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t / T)))
        raise ConfigError(f"unknown lr_schedule {self.lr_schedule!r}")

    def progress_at(self, step: int) -> float:
        """Annealing progress in [0, 1] (β/K)."""
        if not self.anneal:
            return 1.0
        horizon = max(1.0, self.anneal_fraction * self.steps)
        return min(1.0, step / horizon)


@dataclass
class OptimConfig:
    optimizer: str = "adamw"  # adam | adamw | sgd | lbfgs
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0
    ema: float | None = None  # e.g. 0.999; applied at each stage end
    betas: tuple[float, float] = (0.9, 0.999)
    lbfgs_history: int = 20
    lbfgs_max_iter: int = 20  # inner iterations per outer step
    #: learning-rate multipliers by parameter-name prefix, e.g.
    #: ``{"field.values": 10.0, "operator.": 0.1}`` (names: ``field.<name>``, ``operator.<name>``).
    lr_mult: dict[str, float] = field(default_factory=dict)


@dataclass
class Curriculum:
    stages: list[Stage] = field(default_factory=lambda: [Stage()])
    optim: OptimConfig = field(default_factory=OptimConfig)
    early_stop_patience: int | None = None  # steps without data-loss improvement (per stage)
    early_stop_min_delta: float = 1e-6
    discrepancy_tau: float | None = None  # Morozov: stop stage when RMSE <= tau * noise_std
    restarts: int = 1  # multi-restart; best final data loss wins
    time_budget_s: float | None = None

    def __post_init__(self) -> None:
        if not self.stages:
            raise ConfigError("Curriculum needs at least one stage")
        if self.restarts < 1:
            raise ConfigError("restarts must be >= 1")

    @property
    def total_steps(self) -> int:
        return sum(s.steps for s in self.stages)

    @property
    def final_shape(self) -> tuple[int, ...] | None:
        return self.stages[-1].shape

    # ---- factories --------------------------------------------------------------------
    @staticmethod
    def single(
        shape: Sequence[int] | None = None, steps: int = 2000, lr: float = 1e-3, **kw
    ) -> Curriculum:
        return Curriculum([Stage("main", shape, steps, lr, **kw)])

    @staticmethod
    def multiscale(
        shape: Sequence[int],
        n_stages: int = 2,
        steps: Sequence[int] | int = (3000, 7000),
        lr: float = 1e-3,
        lr_decay: float = 0.5,
        min_size: int = 8,
        **stage_kw,
    ) -> Curriculum:
        """Coarse-to-fine stages ending at ``shape`` (NeTMY: 32² → 64², lr halved in stage 2)."""
        shape = shape_tuple(shape)
        if isinstance(steps, int):
            steps = [steps] * n_stages
        if len(steps) != n_stages:
            raise ConfigError("steps must have one entry per stage")
        stages = []
        for i in range(n_stages):
            f = 2 ** (n_stages - 1 - i)
            s = tuple(max(min_size, n // f) for n in shape)
            stages.append(Stage(f"stage{i + 1}", s, int(steps[i]), lr * (lr_decay**i), **stage_kw))
        return Curriculum(stages)

    def scaled(self, factor: float) -> Curriculum:
        """Copy with every stage's step count multiplied by ``factor`` (smoke tests)."""
        import copy

        c = copy.deepcopy(self)
        for s in c.stages:
            s.steps = max(1, int(round(s.steps * factor)))
        return c
