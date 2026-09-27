"""Multi-seed ensembles: cheap uncertainty maps for per-measurement inversion.

Both papers note that reconstructions in low-sensitivity regions should be reported with
uncertainty; a seed ensemble (different network initializations) is the simplest estimator.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from .curriculum import Curriculum
from .result import Result
from .solver import Solver


@dataclass
class EnsembleResult:
    results: list[Result]
    mean: dict[str, torch.Tensor]
    std: dict[str, torch.Tensor]
    seeds: list[int] = field(default_factory=list)

    def coefficient_of_variation(self, name: str | None = None) -> torch.Tensor:
        name = name or next(iter(self.mean))
        return self.std[name] / self.mean[name].abs().clamp_min(1e-12)


def ensemble(
    problem_factory: Callable[[], object],
    curriculum: Curriculum | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    **solver_kw,
) -> EnsembleResult:
    """Solve the same problem from several seeds and aggregate the fields.

    Args:
        problem_factory: zero-arg callable returning a *fresh* :class:`InverseProblem` each call
            (a fresh field module per seed).
        curriculum: shared curriculum.
        seeds: seeds to run.
    """
    results = []
    for s in seeds:
        prob = problem_factory()
        results.append(Solver(prob, curriculum, seed=int(s), **solver_kw).run())
    names = results[0].fields.keys()
    stacked = {k: torch.stack([r.fields[k] for r in results]) for k in names}
    mean = {k: v.mean(0) for k, v in stacked.items()}
    std = {k: v.std(0, unbiased=len(results) > 1) for k, v in stacked.items()}
    return EnsembleResult(results, mean, std, list(seeds))
