"""Shared helpers for baseline solvers (result assembly, resolution resolution, head analysis)."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from ..config import config_hash, to_dict
from ..fields.heads import Bounded, Exp, GatedSoftplus, Head, Identity, Softplus
from ..solve.curriculum import Curriculum
from ..solve.result import Result
from ..solve.solver import RESERVED_HISTORY_KEYS as SOLVER_RESERVED_KEYS
from ..utils.tensor import shape_tuple

if TYPE_CHECKING:  # pragma: no cover
    from ..problem import InverseProblem

#: history keys written by the baseline solvers themselves (the core solver's reserved keys plus
#: ADMM's per-cycle diagnostics); a loss component with one of these names is logged as
#: ``loss/<name>``, exactly as :class:`~nefi.solve.Solver` does.
RESERVED_HISTORY_KEYS = frozenset(SOLVER_RESERVED_KEYS) | frozenset(
    {
        "penalty",
        "outer",
        "outer_step",
        "objective",
        "data_z",
        "primal_residual",
        "dual_residual",
        "mu",
    }
)


def component_key(name: str) -> str:
    """History key of a loss component (``loss/<name>`` when it collides with a reserved key)."""
    return f"loss/{name}" if name in RESERVED_HISTORY_KEYS else name


def resolve_shape(
    problem: InverseProblem,
    shape: Sequence[int] | None = None,
    curriculum: Curriculum | None = None,
) -> tuple[int, ...]:
    """Working resolution of a single-resolution baseline.

    Priority: explicit ``shape`` > ``curriculum.final_shape`` > ``problem.curriculum.final_shape``
    > ``problem.domain.shape`` (i.e. baselines honour the curriculum's final resolution).
    """
    if shape is not None:
        return shape_tuple(shape)
    for cur in (curriculum, getattr(problem, "curriculum", None)):
        if cur is not None and cur.final_shape is not None:
            return shape_tuple(cur.final_shape)
    return tuple(problem.domain.shape)


def head_range(head: Head) -> tuple[float | None, float | None]:
    """Value range ``(lo, hi)`` enforced by a head (``None`` = unbounded side)."""
    inner = getattr(head, "inner", None)
    if inner is not None:  # SupportMasked and friends: report the inner head's range
        return head_range(inner)
    if isinstance(head, Bounded):
        return head.lo, head.hi
    if isinstance(head, Softplus | Exp | GatedSoftplus):
        return 0.0, None
    return None, None


def head_init_value(head: Head) -> float:
    """Field value produced by the head's suggested initial raw bias (0 raw if none)."""
    bias = head.init_bias()
    raw = torch.zeros(1, head.n_in) if bias is None else torch.tensor([bias], dtype=torch.float32)
    with torch.no_grad():
        return float(head(raw).reshape(-1)[0])


def is_positivity_head(head: Head) -> bool:
    """True for heads whose only role is non-negativity (softplus / exp / gated softplus)."""
    return isinstance(head, Softplus | Exp | GatedSoftplus)


def is_identity_head(head: Head) -> bool:
    return isinstance(head, Identity)


def fields_to_cpu(d: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in d.items()}


def assemble_result(
    problem: InverseProblem,
    fields: dict[str, torch.Tensor],
    shape: tuple[int, ...],
    history: Mapping[str, list[float]],
    stage_results: list[dict[str, Any]],
    t0: float,
    solver_config: Any,
    extra: dict[str, Any],
) -> Result:
    """Apply the problem's post-processing and package a :class:`~nefi.solve.Result`.

    Mirrors :meth:`nefi.solve.Solver._finalize` so baseline results are interchangeable with
    neural-field results in benchmarks (same keys, same post-processing, same config hashing).
    """
    op = problem.operator.at_resolution(shape)
    with torch.no_grad():
        pred = op(fields)
    raw_fields = dict(fields)
    post_info: dict[str, Any] = {}
    out = dict(fields)
    for pp in problem.postprocess:
        out, info = pp(out, pred, problem, shape)
        post_info.update(info)
    if problem.postprocess:
        with torch.no_grad():
            pred = op(out)
    total_s = time.perf_counter() - t0
    cfg = {"solver": to_dict(solver_config), "problem": problem.describe()}
    return Result(
        fields=fields_to_cpu(out),
        raw_fields=fields_to_cpu(raw_fields),
        pred=pred.detach().cpu(),
        history=dict(history),
        stage_results=list(stage_results),
        timing={"total_s": total_s, "per_stage_s": [s.get("seconds", 0.0) for s in stage_results]},
        post_info=post_info,
        config_hash=config_hash(cfg),
        extra=dict(extra),
    )
