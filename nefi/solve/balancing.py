"""Adaptive loss balancing (GradNorm-style) as a solver callback.

Hand-tuned loss weights (NeTMY Tab. 7, NeFTY Tab. 5) are the right choice for reported results,
but when porting the recipe to a new problem the relative gradient magnitudes of the data term and
the regularizers are unknown. :class:`GradNormBalancing` measures, every few steps, the gradient
norm each term induces on the field parameters and rescales the weights so that the *weighted*
gradient norms follow prescribed shares (Chen et al. 2018, simplified: no learnable weights, a
damped multiplicative update with an anchor term whose weight never changes).

It operates on the active stage's :class:`~nefi.losses.LossSet` (``solver.stage_losses``), so
per-stage overrides are respected, and it is a no-op under ``Solver(compile="step")`` (weights are
baked into the compiled graph) — a warning is logged in that case.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch

from ..registry import register
from .callbacks import Callback, StepState

log = logging.getLogger("nefi")


@register("callback", "gradnorm")
class GradNormBalancing(Callback):
    """Rebalance loss weights so weighted gradient norms follow target shares.

    Args:
        shares: target share per loss name (relative). Missing data terms default to 1.0 and
            missing regularizers to ``reg_share``. A term with share 0 is left untouched.
        every: rebalance every ``every`` steps (each rebalance costs one backward per active term).
        alpha: damping exponent of the multiplicative update (0 = no change, 1 = full correction).
        ema: smoothing of the measured gradient norms across rebalances (0 = none).
        anchor: loss name whose weight is held fixed (default: the first data term), so the absolute
            scale of the objective does not drift.
        bounds: ``(min, max)`` multiplicative bounds on every weight relative to its initial value.
        warmup: steps before the first rebalance.
        reg_share: default share of non-data terms.
    """

    def __init__(
        self,
        shares: Mapping[str, float] | None = None,
        every: int = 50,
        alpha: float = 0.5,
        ema: float = 0.5,
        anchor: str | None = None,
        bounds: tuple[float, float] = (1e-3, 1e3),
        warmup: int = 20,
        reg_share: float = 0.1,
    ) -> None:
        self.shares = dict(shares or {})
        self.every = int(every)
        self.alpha = float(alpha)
        self.ema = float(ema)
        self.anchor = anchor
        self.bounds = bounds
        self.warmup = int(warmup)
        self.reg_share = float(reg_share)
        self.log: list[dict[str, float]] = []
        self._norms: dict[str, float] = {}
        self._initial: dict[str, float] = {}
        self._disabled = False

    # ------------------------------------------------------------------------------------
    def on_stage_start(self, solver, stage_idx, stage) -> None:
        self._norms = {}
        self._initial = {}
        self._disabled = getattr(solver, "compile_mode", None) == "step"
        if self._disabled:
            log.warning(
                "GradNormBalancing is disabled under compile='step' (weights are compiled in)"
            )

    def _target_shares(self, losses) -> dict[str, float]:
        data = set(losses.data_terms())
        out = {}
        for k in losses.names:
            if losses.weights[k] == 0.0:
                continue
            out[k] = float(self.shares.get(k, 1.0 if k in data else self.reg_share))
        return out

    @torch.enable_grad()
    def _gradient_norms(self, solver, losses, state: StepState) -> dict[str, float]:
        problem = solver.problem
        stage = state.stage
        ctx = problem.context(stage.shape, progress=state.progress, stage=stage, step=state.step)
        params = [p for p in solver.stage_params if p.requires_grad]
        norms: dict[str, float] = {}
        for k, term in losses.terms.items():
            if losses.weights[k] == 0.0:
                continue
            v = term(ctx)
            if not v.requires_grad:
                norms[k] = 0.0
                continue
            grads = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
            sq = sum(float((g.detach() ** 2).sum()) for g in grads if g is not None)
            norms[k] = sq**0.5
        return norms

    def on_step(self, solver, state: StepState) -> None:
        if self._disabled or state.step < self.warmup or state.step % self.every != 0:
            return
        losses = getattr(solver, "stage_losses", None)
        if losses is None:
            return
        shares = self._target_shares(losses)
        if len(shares) < 2:
            return
        raw = self._gradient_norms(solver, losses, state)
        for k, g in raw.items():
            prev = self._norms.get(k)
            self._norms[k] = g if prev is None else self.ema * prev + (1.0 - self.ema) * g
        weighted = {k: losses.weights[k] * self._norms.get(k, 0.0) for k in shares}
        total = sum(weighted.values())
        if total <= 0:
            return
        anchor = self.anchor or (losses.data_terms()[0] if losses.data_terms() else None)
        share_sum = sum(shares.values())
        new = dict(losses.weights)
        for k, sh in shares.items():
            if k == anchor or sh <= 0 or self._norms.get(k, 0.0) <= 0:
                continue
            target = sh / share_sum * total
            ratio = target / max(weighted[k], 1e-30)
            w = losses.weights[k] * ratio**self.alpha
            init = self._initial.setdefault(k, losses.weights[k] if losses.weights[k] > 0 else w)
            lo, hi = self.bounds
            new[k] = float(min(max(w, lo * init), hi * init))
        losses.weights.update(new)
        entry = {"step": float(state.global_step), **{f"w/{k}": v for k, v in new.items()}}
        entry.update({f"g/{k}": v for k, v in self._norms.items()})
        self.log.append(entry)
        log.debug(
            "gradnorm step %d: %s", state.global_step, {k: round(v, 4) for k, v in new.items()}
        )
