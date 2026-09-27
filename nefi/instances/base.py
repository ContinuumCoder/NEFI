"""Instance protocol: a self-contained, registered exemplar inverse problem.

An :class:`Instance` bundles a config dataclass, a domain, scene + data generators, the problem
builder, a default curriculum and instance-appropriate metrics. The CLI, benchmark protocol and
tutorials only talk to this interface.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..bench.base import DataGenerator, SceneGenerator
from ..config import from_dict, to_dict
from ..domain import Domain
from ..errors import ConfigError
from ..measurement import Measurement
from ..metrics.basic import evaluate
from ..problem import InverseProblem
from ..solve.curriculum import Curriculum
from ..solve.result import Result
from ..solve.solver import Solver
from ..utils.seed import seed_everything


@dataclass
class RunOutput:
    result: Result
    metrics: dict[str, float]
    gt: dict[str, torch.Tensor] | None
    measurement: Measurement
    extra: dict[str, Any] = field(default_factory=dict)


class Instance:
    """Base class for registered instances. Subclass, set ``name`` and ``Config``, implement
    hooks.
    """

    name: str = "instance"
    Config: type | None = None
    description: str = ""

    def __init__(self, cfg: Any | Mapping | None = None, **overrides: Any) -> None:
        if self.Config is None:
            self.cfg = dict(cfg or {}, **overrides)
        elif cfg is None or isinstance(cfg, Mapping):
            self.cfg = from_dict(self.Config, {**dict(cfg or {}), **overrides})
        elif dataclasses.is_dataclass(cfg):
            self.cfg = dataclasses.replace(cfg, **overrides) if overrides else cfg
        else:
            raise ConfigError(f"cannot build {type(self).__name__} config from {type(cfg)}")

    # --- hooks (override) ---------------------------------------------------------------
    def domain(self) -> Domain:  # pragma: no cover
        raise NotImplementedError

    def scene_generator(self) -> SceneGenerator:  # pragma: no cover
        raise NotImplementedError

    def data_generator(self) -> DataGenerator:  # pragma: no cover
        raise NotImplementedError

    def build_problem(self, measurement: Measurement) -> InverseProblem:  # pragma: no cover
        raise NotImplementedError

    def default_curriculum(self) -> Curriculum:
        return Curriculum.multiscale(self.domain().shape)

    def metrics(self) -> dict[str, Callable]:
        from ..metrics.basic import BASIC_METRICS

        return dict(BASIC_METRICS)

    def evaluate(self, result: Result, gt: Mapping[str, torch.Tensor]) -> dict[str, float]:
        name = next(iter(result.fields))
        return evaluate(result.fields[name], gt[name], self.metrics())

    def baselines(self) -> dict[str, Callable[[Measurement], tuple[InverseProblem, Curriculum]]]:
        """Optional: name -> builder returning (problem, curriculum) for baseline methods."""
        return {}

    def refine(self, result: Result, measurement: Measurement, **kw: Any):
        """Edge refinement of a reconstruction of ``measurement``: a short second optimization
        under the same operator and data with a level-set (piecewise-constant) prior, phase
        values from the instance configuration (``nefi.solve.refine.REFINE_DEFAULTS`` or a
        ``refine_defaults`` attribute). Returns ``(result, report)``; a refinement that degrades
        the data fit is refused (the smooth result is returned). See ``docs/refinement.md``.

        Keyword arguments (``gt=``, ``problem=``, ``mode=``, ...) go to
        :func:`nefi.solve.refine.refine_instance`.
        """
        from ..solve.refine import refine_instance

        return refine_instance(self, result, measurement, **kw)

    # --- end-to-end ---------------------------------------------------------------------
    def make_measurement(
        self,
        seed: int = 0,
        scene_class: str | None = None,
        gt: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], Measurement]:
        rng = np.random.default_rng(seed)
        scene_class = scene_class or self.default_scene_class()
        scenes = self.scene_generator()
        gen = self.data_generator()
        dom = self.domain()
        if gt is None:
            fine = tuple(s * gen.supersample for s in dom.shape) if gen.supersample > 1 else None
            gt_fine = scenes.sample(rng, scene_class, fine)
            gt = scenes.sample(np.random.default_rng(seed), scene_class, None) if fine else gt_fine
        else:
            gt_fine = gt
        target = self.build_problem_measurement_shape(dom.shape)
        meas = gen.generate(gt_fine, rng, target_shape=target)
        return {k: torch.as_tensor(v) for k, v in gt.items()}, meas

    def default_scene_class(self) -> str | None:
        """Scene class used when ``run``/``make_measurement`` get none (``cfg.scene`` if set)."""
        return getattr(self.cfg, "scene", None) if self.Config is not None else None

    def build_problem_measurement_shape(self, field_shape: Sequence[int]) -> tuple[int, ...] | None:
        """Measurement shape at native resolution (default: ask a throwaway problem's operator)."""
        return None

    def run(
        self,
        seed: int = 0,
        device: str = "auto",
        scene_class: str | None = None,
        curriculum: Curriculum | None = None,
        callbacks: Sequence = (),
        **solver_kw,
    ) -> RunOutput:
        gt, meas = self.make_measurement(seed, scene_class)
        seed_everything(seed)  # the field initialization must be controlled by ``seed``
        problem = self.build_problem(meas)
        cur = curriculum or self.default_curriculum()
        result = Solver(
            problem, cur, device=device, seed=seed, callbacks=callbacks, **solver_kw
        ).run()
        metrics = self.evaluate(result, gt)
        return RunOutput(result, metrics, gt, meas, {"scene_class": scene_class, "seed": seed})

    def config_dict(self) -> dict:
        return to_dict(self.cfg)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.config_dict()})"
