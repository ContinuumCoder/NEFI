"""Solver callbacks: logging, progress bars, checkpoints, plotting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from ..registry import register

if TYPE_CHECKING:  # pragma: no cover
    from .curriculum import Stage
    from .result import Result
    from .solver import Solver

log = logging.getLogger("nefi")


@dataclass
class StepState:
    stage_idx: int
    stage: Stage
    step: int
    global_step: int
    lr: float
    progress: float
    total: float
    components: dict[str, float]
    data_loss: float
    elapsed: float
    fields: dict[str, torch.Tensor] | None = None
    pred: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Callback:
    def on_run_start(self, solver: Solver) -> None: ...
    def on_stage_start(self, solver: Solver, stage_idx: int, stage: Stage) -> None: ...
    def on_step(self, solver: Solver, state: StepState) -> None: ...
    def on_stage_end(self, solver: Solver, stage_idx: int, stage: Stage, info: dict) -> None: ...
    def on_run_end(self, solver: Solver, result: Result) -> None: ...


@register("callback", "logging")
class LoggingCallback(Callback):
    """Log loss components every ``every`` steps through the ``nefi`` logger."""

    def __init__(self, every: int = 100) -> None:
        self.every = every

    def on_stage_start(self, solver, stage_idx, stage):
        log.info(
            "stage %d '%s' shape=%s steps=%d lr=%.2e",
            stage_idx,
            stage.name,
            stage.shape,
            stage.steps,
            stage.lr,
        )

    def on_step(self, solver, state):
        if state.step % self.every == 0 or state.step == state.stage.steps - 1:
            comps = " ".join(f"{k}={v:.3e}" for k, v in state.components.items())
            log.info(
                "[%s %5d] total=%.3e lr=%.2e β=%.2f %s",
                state.stage.name,
                state.step,
                state.total,
                state.lr,
                state.progress,
                comps,
            )

    def on_stage_end(self, solver, stage_idx, stage, info):
        log.info(
            "stage %d done in %.1fs (%d steps)",
            stage_idx,
            info.get("seconds", 0),
            info.get("steps", 0),
        )


@register("callback", "progress")
class ProgressBar(Callback):
    """tqdm progress bar over all curriculum steps."""

    def __init__(self, leave: bool = False) -> None:
        self.leave = leave
        self.bar = None

    def on_run_start(self, solver):
        from tqdm.auto import tqdm

        self.bar = tqdm(total=solver.curriculum.total_steps, leave=self.leave, dynamic_ncols=True)

    def on_step(self, solver, state):
        if self.bar is not None:
            self.bar.update(1)
            if state.step % 20 == 0:
                self.bar.set_postfix(
                    {"loss": f"{state.total:.3e}", "stage": state.stage.name}, refresh=False
                )

    def on_run_end(self, solver, result):
        if self.bar is not None:
            self.bar.close()


@register("callback", "checkpoint")
class CheckpointCallback(Callback):
    """Save field/operator state every ``every`` steps and at the end of each stage."""

    def __init__(self, path: str | Path, every: int = 500) -> None:
        self.path = Path(path)
        self.every = every

    def _save(self, solver, tag: str) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "field": solver.problem.field.state_dict(),
                "operator": solver.problem.operator.state_dict(),
                "history": dict(solver.history),
            },
            self.path / f"ckpt_{tag}.pt",
        )

    def on_step(self, solver, state):
        if self.every and state.global_step > 0 and state.global_step % self.every == 0:
            self._save(solver, f"step{state.global_step}")

    def on_stage_end(self, solver, stage_idx, stage, info):
        self._save(solver, f"stage{stage_idx}")


@register("callback", "history_fields")
class FieldSnapshots(Callback):
    """Keep detached CPU snapshots of the primary field every ``every`` steps (for animations)."""

    def __init__(self, every: int = 100, name: str | None = None) -> None:
        self.every, self.name = every, name
        self.snapshots: list[tuple[int, torch.Tensor]] = []

    def on_step(self, solver, state):
        if state.fields is not None and state.global_step % self.every == 0:
            key = self.name or solver.problem.field.primary
            self.snapshots.append((state.global_step, state.fields[key].detach().cpu().clone()))
