"""L-BFGS curricula — the quasi-Newton free-density reference (NeTMY App. E.2, §5.3).

The core :class:`~nefi.solve.Solver` already drives ``torch.optim.LBFGS`` (strong-Wolfe line
search) when ``OptimConfig(optimizer="lbfgs")``; this module only packages the right curriculum:
constant step length (the line search chooses the actual step), no gradient clipping or weight
decay, no frequency annealing. NeTMY settings: history depth 20, Wolfe line search, 100 outer
iterations on the free density (a :class:`~nefi.fields.GridField`).

Filtering view (NeTMY App. D.6): L-BFGS replaces the raw gradient by ``-H_t^{-1} ∇ρ L`` with a
low-rank inverse-Hessian estimate — a *parameter-space preconditioner*, orthogonal to the
parameterization-Jacobian filter ``G_θ`` of neural fields (for a grid, ``G_θ = I``).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..errors import ConfigError
from ..solve.curriculum import Curriculum, OptimConfig, Stage
from ..utils.tensor import shape_tuple


def lbfgs_curriculum(
    shape: Sequence[int] | None,
    steps: int | Sequence[int] = 100,
    lr: float = 1.0,
    history: int = 20,
    max_iter: int = 20,
    *,
    n_stages: int = 1,
    anneal: bool = False,
    min_size: int = 8,
    **stage_kw,
) -> Curriculum:
    """Curriculum running ``torch.optim.LBFGS`` with a strong-Wolfe line search.

    Args:
        shape: final grid shape (``None``: the problem's native domain shape).
        steps: outer L-BFGS iterations (one per stage entry when ``n_stages > 1``); each outer
            iteration performs up to ``max_iter`` function evaluations.
        lr: L-BFGS step-length scale (1.0 is standard with a line search).
        history: number of stored curvature pairs (NeTMY: 20).
        max_iter: inner iterations per outer step.
        n_stages: > 1 builds a coarse-to-fine sequence (each stage halves the resolution of the
            next, as :meth:`~nefi.solve.Curriculum.multiscale`); requires ``shape``.
        anneal: keep frequency annealing on (only meaningful for neural fields).
        min_size: smallest per-axis size for coarse stages.
        **stage_kw: extra :class:`~nefi.solve.Stage` fields (e.g. ``loss_weights``).

    Returns:
        A :class:`~nefi.solve.Curriculum` with ``OptimConfig(optimizer="lbfgs")``.
    """
    optim = OptimConfig(
        optimizer="lbfgs",
        weight_decay=0.0,
        grad_clip=None,
        lbfgs_history=int(history),
        lbfgs_max_iter=int(max_iter),
    )
    stage_kw = {"lr_schedule": "constant", "anneal": anneal, **stage_kw}
    if n_stages == 1:
        n = steps if isinstance(steps, int) else int(sum(steps))
        return Curriculum([Stage("lbfgs", shape, int(n), float(lr), **stage_kw)], optim=optim)
    if shape is None:
        raise ConfigError("a multi-stage L-BFGS curriculum needs an explicit final shape")
    base = Curriculum.multiscale(
        shape_tuple(shape), n_stages, steps, lr=float(lr), lr_decay=1.0, min_size=min_size
    )
    stages = [
        Stage(f"lbfgs{i + 1}", s.shape, s.steps, float(lr), **stage_kw)
        for i, s in enumerate(base.stages)
    ]
    return Curriculum(stages, optim=optim)


__all__ = ["lbfgs_curriculum"]
