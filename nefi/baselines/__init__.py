"""Baseline parameterizations and solvers (NeTMY App. E.2, NeFTY App. F.2).

Every instance gets the full classical baseline family for free through
:func:`baseline_problem`, which swaps the *parameterization* of an existing
:class:`~nefi.problem.InverseProblem` while keeping its domain, operator, losses, measurement and
post-processing — so the comparison isolates the prior (NeTMY §4.4: the parameterization is the
filtering kernel ``G_θ = J_θ J_θᵀ`` applied to the raw field-space gradient):

====================  ==========================================  ===========================
kind                  parameterization / solver                    ``G_θ``
====================  ==========================================  ===========================
``"grid"``            :class:`~nefi.fields.GridField` + Adam       ``I`` (raw gradient)
``"lbfgs"``           grid + L-BFGS (:func:`lbfgs_curriculum`)     ``I`` (+ quasi-Newton)
``"admm"``            grid + :class:`ADMMSolver` (ℓ1 + box prox)   ``I``
``"gaussian_splat"``  :class:`GaussianSplatField` + SplatControl   low-rank, localized
``"deep_decoder"``    :class:`DeepDecoderField`                    global low-pass
====================  ==========================================  ===========================

Iterative baselines that are not plain gradient curricula (ADMM, closed-form references) mark
themselves in ``problem.meta["solver"]`` (a name registered under the ``"baseline"`` kind) and
may request callbacks in ``problem.meta["callbacks"]``; :func:`solve` dispatches accordingly and
falls back to :class:`~nefi.solve.Solver` otherwise. Running such a problem with the plain solver
still works (it degrades to a grid descent / returns the closed-form initialization).
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from ..errors import ConfigError
from ..fields.base import Field
from ..fields.grid import GridField
from ..fields.heads import Bounded, Heads, Identity
from ..problem import InverseProblem
from ..registry import get as _registry_get
from ..solve.curriculum import Curriculum, Stage
from ..solve.result import Result
from ..solve.solver import Solver
from ._common import head_init_value, head_range, is_positivity_head, resolve_shape
from .admm import ADMMConfig, ADMMSolver, prox_l1_box
from .deep_decoder import ChannelNorm, DeepDecoderField, stage_sizes
from .direct import DirectSolver
from .gaussian_splat import GaussianSplatField, SplatControl, axis_factor, grid_axes
from .lbfgs import lbfgs_curriculum
from .thermography import contrast_mask, depth_to_alpha_volume, ppt, tsr

log = logging.getLogger("nefi")

#: kinds understood by :func:`baseline_problem`.
BASELINE_KINDS = ("grid", "lbfgs", "admm", "gaussian_splat", "deep_decoder")

#: default learning-rate multipliers relative to the (neural-field) curriculum of the problem.
#: Free grids and explicit primitives take larger Adam steps than an MLP (NeTMY App. E.2 uses
#: 5e-3 for the free density vs 1e-3 for the MLP; toy1d uses ×10).
LR_MULT: dict[str, float] = {"grid": 10.0, "gaussian_splat": 10.0, "deep_decoder": 1.0}


def _copy_heads(heads: Heads) -> Heads:
    return copy.deepcopy(heads)


def _auto_heads(heads: Heads, kind: str, split_field: str) -> Heads:
    """Kind-dependent head mapping used by :func:`baseline_problem` with ``heads="auto"``."""
    items = []
    for name, head in heads.items():
        if kind == "admm" and name == split_field:
            items.append((name, Identity()))  # ADMM enforces the range by projection instead
        elif kind == "gaussian_splat" and (
            is_positivity_head(head) or (isinstance(head, Bounded) and head.lo == 0.0)
        ):
            items.append((name, Identity()))  # splat density is already >= 0
        else:
            items.append((name, copy.deepcopy(head)))
    return Heads(items, primary=heads.primary)


def _resolve_heads(heads: Any, problem: InverseProblem, kind: str, split_field: str) -> Heads:
    src = problem.field.heads
    if isinstance(heads, Heads):
        return copy.deepcopy(heads)
    if isinstance(heads, Mapping):
        return Heads(heads)
    if heads == "keep":
        return _copy_heads(src)
    if heads == "identity":
        return Heads([(n, Identity()) for n in src.names], primary=src.primary)
    if heads == "auto":
        return _auto_heads(src, kind, split_field)
    raise ConfigError(f"heads must be 'auto', 'keep', 'identity' or a Heads mapping, got {heads!r}")


def _base_curriculum(problem: InverseProblem, curriculum: Curriculum | None) -> Curriculum:
    cur = curriculum or problem.curriculum
    return copy.deepcopy(cur) if cur is not None else Curriculum.multiscale(problem.domain.shape)


def _rescale(cur: Curriculum, lr: float | None, lr_mult: float, steps) -> Curriculum:
    base_lr = cur.stages[0].lr
    factor = (float(lr) / base_lr if base_lr > 0 else 1.0) if lr is not None else lr_mult
    for s in cur.stages:
        s.lr = s.lr * factor
    if steps is not None:
        if isinstance(steps, int):
            tot = max(1, cur.total_steps)
            for s in cur.stages:
                s.steps = max(1, round(steps * s.steps / tot))
        else:
            if len(steps) != len(cur.stages):
                raise ConfigError(f"steps needs {len(cur.stages)} entries, got {len(steps)}")
            for s, n in zip(cur.stages, steps):
                s.steps = int(n)
    return cur


def baseline_problem(
    problem: InverseProblem,
    kind: str,
    *,
    curriculum: Curriculum | None = None,
    lr: float | None = None,
    steps: int | Sequence[int] | None = None,
    heads: str | Heads | Mapping = "auto",
    density_control: bool = True,
    control: Mapping[str, Any] | None = None,
    admm: Mapping[str, Any] | None = None,
    lbfgs_history: int = 20,
    lbfgs_max_iter: int = 20,
    name: str | None = None,
    **field_kw: Any,
) -> tuple[InverseProblem, Curriculum | None]:
    """Swap the parameterization of ``problem`` for a baseline one.

    Domain, operator, losses, measurement, post-processing and ``downsample_obs`` are kept (an
    operator with trainable nuisance parameters is deep-copied so runs do not interfere).

    Args:
        problem: the reference (usually neural-field) problem.
        kind: one of :data:`BASELINE_KINDS`.
        curriculum: base curriculum (default ``problem.curriculum``, else the library default).
        lr: first-stage learning rate (stage ratios kept); default: base LR × :data:`LR_MULT`.
        steps: total steps (int, distributed proportionally) or per-stage steps; for ``"lbfgs"``
            the number of outer L-BFGS iterations (default 100, NeTMY).
        heads: ``"auto"`` (grid / L-BFGS / DeepDecoder keep the heads; ADMM uses ``Identity`` on the
            split field and moves its range into the box projection; GaussianSplat maps positivity
            heads and ``Bounded(0, hi)`` to ``Identity`` since splat densities are non-negative),
            ``"keep"``, ``"identity"`` or an explicit :class:`~nefi.fields.Heads` mapping.
        density_control: attach :class:`SplatControl` (``problem.meta["callbacks"]``) for splats.
        control: keyword overrides for :class:`SplatControl`.
        admm: keyword overrides for :class:`ADMMConfig` (the box defaults to the head's range).
        lbfgs_history / lbfgs_max_iter: L-BFGS memory and inner iterations.
        name: problem name (default ``"<name>-<kind>"``).
        **field_kw: forwarded to the field constructor (e.g. ``width=64`` for DeepDecoder,
            ``n_primitives=32`` for splats, ``init=0.1`` for grids).

    Returns:
        ``(baseline_problem, curriculum)``; run it with :func:`solve` (which also handles ADMM and
        requested callbacks) or with :class:`~nefi.solve.Solver` for gradient baselines.
    """
    if kind not in BASELINE_KINDS:
        raise ConfigError(f"unknown baseline kind {kind!r}; known: {BASELINE_KINDS}")
    base = _base_curriculum(problem, curriculum)
    shape = resolve_shape(problem, None, base)
    split = (admm or {}).get("field") or problem.field.primary
    new_heads = _resolve_heads(heads, problem, kind, split)
    meta: dict[str, Any] = {k: v for k, v in problem.meta.items() if k not in ("solver",)}
    meta.update({"baseline": kind, "reference_field": type(problem.field).__name__})
    field: Field
    cur: Curriculum | None

    if kind in ("grid", "lbfgs"):
        field = GridField(shape, new_heads, **field_kw)
        if kind == "grid":
            cur = _rescale(base, lr, LR_MULT["grid"], steps)
        else:
            n = 100 if steps is None else steps
            cur = lbfgs_curriculum(
                shape, n, 1.0 if lr is None else lr, lbfgs_history, lbfgs_max_iter
            )
    elif kind == "admm":
        src_head = problem.field.heads[split]
        lo, hi = head_range(src_head)
        cfg = {"lo": lo, "hi": hi, "field": split, "shape": shape, **dict(admm or {})}
        admm_cfg = ADMMConfig(**cfg)
        init = field_kw.pop("init", None)
        if init is None:
            raw0 = new_heads.init_bias().tolist()
            x0 = head_init_value(src_head)
            if admm_cfg.lo is not None:
                x0 = max(x0, admm_cfg.lo)
            if admm_cfg.hi is not None:
                x0 = min(x0, admm_cfg.hi)
            raw0[new_heads.slices[split].start] = x0
            init = raw0
        field = GridField(shape, new_heads, init=init, **field_kw)
        meta["solver"] = "admm"
        meta["solver_config"] = admm_cfg
        steps_total = admm_cfg.n_inner * admm_cfg.max_outer
        cur = Curriculum(
            [Stage("admm", shape, steps_total, admm_cfg.lr, lr_schedule="constant", anneal=False)]
        )
    elif kind == "gaussian_splat":
        field_kw.setdefault("min_sigma", 0.5 / max(shape))
        field = GaussianSplatField(problem.domain.ndim, new_heads, **field_kw)
        cur = _rescale(base, lr, LR_MULT["gaussian_splat"], steps)
        if density_control:
            every = max(10, cur.total_steps // 10)
            ctrl = {"every": every, "start": every, **dict(control or {})}
            meta["callbacks"] = [*meta.get("callbacks", []), SplatControl(**ctrl)]
    else:  # deep_decoder
        field = DeepDecoderField(shape, new_heads, **field_kw)
        cur = _rescale(base, lr, LR_MULT["deep_decoder"], steps)

    operator = problem.operator
    if any(q.requires_grad for q in operator.parameters()):
        operator = copy.deepcopy(operator)
    new = InverseProblem(
        problem.domain,
        field,
        operator,
        problem.losses,
        problem.measurement,
        postprocess=list(problem.postprocess),
        curriculum=cur,
        downsample_obs=problem.downsample_obs,
        name=name or f"{problem.name}-{kind}",
        meta=meta,
    )
    return new, cur


def solve(
    problem: InverseProblem, curriculum: Curriculum | None = None, **solver_kw: Any
) -> Result:
    """Run a (baseline) problem with the solver it asks for.

    Dispatches on ``problem.meta["solver"]`` (a name registered under the ``"baseline"`` registry
    kind, e.g. ``"admm"`` or ``"direct"``); uses :class:`~nefi.solve.Solver` when absent. Callbacks
    requested in ``problem.meta["callbacks"]`` (e.g. :class:`SplatControl`) are appended to
    ``solver_kw["callbacks"]``.
    """
    meta = getattr(problem, "meta", None) or {}
    callbacks = [*solver_kw.pop("callbacks", ()), *meta.get("callbacks", ())]
    kind = meta.get("solver")
    if kind in (None, "gradient", "solver"):
        return Solver(problem, curriculum, callbacks=callbacks, **solver_kw).run()
    cls = _registry_get("baseline", kind)
    return cls(
        problem, meta.get("solver_config"), curriculum=curriculum, callbacks=callbacks, **solver_kw
    ).run()


__all__ = [
    "contrast_mask",
    "depth_to_alpha_volume",
    "ppt",
    "tsr",
    "ADMMConfig",
    "ADMMSolver",
    "BASELINE_KINDS",
    "ChannelNorm",
    "DeepDecoderField",
    "DirectSolver",
    "GaussianSplatField",
    "LR_MULT",
    "SplatControl",
    "axis_factor",
    "baseline_problem",
    "grid_axes",
    "lbfgs_curriculum",
    "prox_l1_box",
    "solve",
    "stage_sizes",
]
