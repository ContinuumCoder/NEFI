"""Capacity growth: start small, add degrees of freedom when the residual plateaus.

Representations with a *capacity pool* implement the growth protocol

    can_grow() -> bool
    grow(hint: Mapping | None = None) -> bool      # True if capacity was added

(:class:`~nefi.fields.geometric.FourierBasisField` opens the next shell of modes,
:class:`~nefi.fields.geometric.LayeredField` activates the next layer,
:class:`~nefi.fields.geometric.StarShapeField` / :class:`~nefi.fields.geometric.PolygonField`
activate the next pooled shape, placing it at the extremum of the field-space data gradient — a
topological-derivative heuristic). All pooled parameters exist from the start, so the optimizer
the :class:`~nefi.solve.Solver` builds at stage start already owns them; growth only unmasks.

:class:`GrowCapacity` watches the data loss and, on a plateau, performs the next action of its
``order``: grow a module (round robin over growable modules) or open the next annealing band
(``solver.progress_override``, like :class:`~.annealing.ResidualDrivenAnnealing`). Starting with
few degrees of freedom is the representation-level analogue of early stopping: the fit stays in
the low-dimensional, well-conditioned subspace until the data demand more (NeFTY Cor. 1).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn

from ...errors import ConfigError
from ...losses.base import Context
from ...registry import register
from ...solve.callbacks import Callback, StepState
from .annealing import PlateauDetector, n_levels, stage_noise_std

log = logging.getLogger("nefi")


@runtime_checkable
class Growable(Protocol):
    """Protocol for representations with a capacity pool."""

    def can_grow(self) -> bool: ...

    def grow(self, hint: Mapping | None = None) -> bool: ...


def growable_modules(field: nn.Module) -> list[nn.Module]:
    """Sub-modules implementing :class:`Growable` (depth-first order)."""
    out = []
    for m in field.modules():
        if callable(getattr(m, "grow", None)) and callable(getattr(m, "can_grow", None)):
            out.append(m)
    return out


def field_space_gradient(solver: Any, state: StepState, data_only: bool = True) -> dict[str, Any]:
    """Gradient of the (data) loss with respect to the field values at the current state.

    Returns ``{"gradient": ∂L/∂x (primary field, stage grid), "coords": normalized coordinates}``
    — the hint consumed by shape fields' :meth:`grow` (NeTMY §4.4: this is the raw gradient a
    free-pixel solver would execute).
    """
    p = solver.problem
    stage = state.stage
    dom = p.domain if stage.shape is None else p.domain.at(stage.shape)
    if state.fields is None:
        raise ConfigError("field_space_gradient needs StepState.fields")
    device = next(iter(state.fields.values())).device
    dtype = next(iter(state.fields.values())).dtype
    op = p.operator.at_resolution(dom.shape)
    obs = p.measurement_at(dom.shape).to(device, dtype)
    losses = p.losses.with_weights(stage.loss_weights)
    if data_only:
        keep = set(losses.data_terms())
        if keep:
            losses = losses.with_weights({k: 0.0 for k in losses.names if k not in keep})
    name = p.field.primary
    with torch.enable_grad():
        fields = {k: v.detach().clone().requires_grad_(k == name) for k, v in state.fields.items()}
        pred = op(fields)
        ctx = Context(fields, pred, obs, dom, op, p.field, stage, state.step, state.progress)
        total, _ = losses(ctx)
        (g,) = torch.autograd.grad(total, fields[name], allow_unused=True)
    grad = torch.zeros_like(fields[name]) if g is None else g.detach()
    return {"gradient": grad, "coords": dom.coords(device=device, dtype=dtype), "name": name}


@register("callback", "grow_capacity")
class GrowCapacity(Callback):
    """On data-loss plateaus, add capacity: grow modules and/or open annealing bands.

    Args:
        patience: plateau length (steps).
        delta: relative improvement that counts as progress.
        order: actions tried on each plateau, in order: ``"modules"`` (grow one growable module,
            round robin) and/or ``"progress"`` (open the next annealing band).
        use_progress: manage ``solver.progress_override`` at all (False leaves the stage's own
            annealing clock untouched and only grows modules).
        n_levels: annealing levels (default: from the field's encoding).
        start_level: open bands at stage start when managing progress.
        ramp_steps: ramp length of a newly opened band.
        placement: ``"gradient"`` (pass the field-space data gradient as a growth hint) or
            ``"default"`` (no hint).
        max_grows: cap on the number of module growths (``None`` = unlimited).
        stop_when_exhausted: end the stage when a plateau finds nothing left to grow.
        reset_each_stage: restart the band ladder each stage (modules keep their capacity).
        min_steps_between: minimum steps between growth actions (default ``patience``).
        rate_fraction: diminishing-returns criterion (see
            :class:`~nefi.fields.adaptive.annealing.PlateauDetector`).
        discrepancy_tau: never grow once ``RMSE ≤ τ · σ`` (Morozov: the residual is noise, extra
            capacity would fit it); ``None`` disables, inactive without ``noise_std``.
        modules: explicit growable modules (default: all found in the field).

    Attributes:
        events: list of ``{"stage", "step", "global_step", "action", "target", "capacity"}``.
    """

    ORDER = ("modules", "progress")

    def __init__(
        self,
        patience: int = 25,
        delta: float = 1e-3,
        order: Sequence[str] = ("modules", "progress"),
        use_progress: bool = True,
        n_levels: int | None = None,
        start_level: int = 0,
        ramp_steps: int = 50,
        placement: str = "gradient",
        max_grows: int | None = None,
        stop_when_exhausted: bool = False,
        reset_each_stage: bool = False,
        min_steps_between: int | None = None,
        rate_fraction: float = 0.25,
        discrepancy_tau: float | None = 1.1,
        modules: Sequence[nn.Module] | None = None,
    ) -> None:
        bad = [o for o in order if o not in self.ORDER]
        if bad or not order:
            raise ConfigError(f"order entries must be in {self.ORDER}, got {tuple(order)}")
        if placement not in ("gradient", "default"):
            raise ConfigError("placement must be 'gradient' or 'default'")
        self.detector = PlateauDetector(
            patience,
            delta,
            True,
            min_steps=patience if min_steps_between is None else min_steps_between,
            rate_fraction=rate_fraction,
        )
        self.order = tuple(order)
        self.use_progress = bool(use_progress)
        self.n_levels_arg, self.start_level = n_levels, int(start_level)
        self.ramp_steps = max(0, int(ramp_steps))
        self.placement = placement
        self.max_grows = max_grows
        self.stop_when_exhausted = bool(stop_when_exhausted)
        self.reset_each_stage = bool(reset_each_stage)
        self.discrepancy_tau = discrepancy_tau
        self._sigma: float | None = None
        self._obs: Any = None
        self.at_noise_floor = False
        self._explicit = list(modules) if modules is not None else None
        self._modules: list[nn.Module] = []
        self._rr = 0
        self.K = 1
        self.level = self.start_level
        self.progress = 0.0
        self._ramp: tuple[float, float, int] | None = None
        self._started = False
        self.n_grows = 0
        self.events: list[dict[str, Any]] = []

    def on_run_start(self, solver) -> None:
        field = solver.problem.field
        self._modules = self._explicit if self._explicit is not None else growable_modules(field)
        self.K = max(1, int(self.n_levels_arg) if self.n_levels_arg else n_levels(field))
        self.events, self.n_grows, self._rr, self._started = [], 0, 0, False

    def on_stage_start(self, solver, stage_idx, stage) -> None:
        p = solver.problem
        shape = p.domain.shape if stage.shape is None else tuple(stage.shape)
        self._sigma = None if self.discrepancy_tau is None else stage_noise_std(p, shape)
        self._obs = p.measurement_at(shape) if self._sigma is not None else None
        self.at_noise_floor = False
        if self.use_progress:
            if self.reset_each_stage or not self._started:
                self.level = min(max(self.start_level, 0), self.K)
                self.progress = self.level / self.K
            self._ramp = None
            solver.progress_override = self.progress
        self._started = True
        self.detector.reset()

    def _noise_floor(self, state: StepState) -> bool:
        if self._sigma is None or self._obs is None or state.pred is None:
            return False
        pred = state.pred.detach()
        if tuple(self._obs.data.shape) != tuple(pred.shape):
            return False
        obs = self._obs.to(pred.device, pred.dtype)
        rmse = float(torch.sqrt(obs.masked_mean((pred - obs.data) ** 2)))
        return rmse <= float(self.discrepancy_tau) * self._sigma  # type: ignore[arg-type]

    # --- actions ------------------------------------------------------------------------
    def _grow_module(self, solver, state: StepState) -> tuple[bool, str]:
        if self.max_grows is not None and self.n_grows >= self.max_grows:
            return False, ""
        cands = [m for m in self._modules if m.can_grow()]  # type: ignore[operator]
        if not cands:
            return False, ""
        m = cands[self._rr % len(cands)]
        self._rr += 1
        hint = None
        if self.placement == "gradient" and state.fields is not None:
            try:
                hint = field_space_gradient(solver, state)
            except Exception as e:  # pragma: no cover - exotic operators
                log.debug("GrowCapacity: no gradient hint (%s)", e)
        ok = bool(m.grow(hint))  # type: ignore[operator]
        if ok:
            self.n_grows += 1
        return ok, type(m).__name__

    def _open_band(self) -> bool:
        if not self.use_progress or self.level >= self.K:
            return False
        self.level += 1
        target = self.level / self.K
        if self.ramp_steps > 0:
            self._ramp = (self.progress, target, self.ramp_steps)
        else:
            self.progress = target
        return True

    def on_step(self, solver, state: StepState) -> None:
        if self.use_progress and self._ramp is not None:
            a, b, left = self._ramp
            left -= 1
            self.progress = b - (b - a) * left / max(1, self.ramp_steps)
            self._ramp = None if left <= 0 else (a, b, left)
        if self.use_progress and self._ramp is not None:  # judge after the band ramped in
            self.detector.reset()
            solver.progress_override = self.progress
            return
        if self.detector.update(state.data_loss):
            self.at_noise_floor = self._noise_floor(state)
            done = self.at_noise_floor  # nothing to add: the residual is noise
            for action in () if done else self.order:
                if action == "modules":
                    ok, target = self._grow_module(solver, state)
                else:
                    ok, target = self._open_band(), "progress"
                if ok:
                    cap = {
                        type(m).__name__: getattr(m, "capacity", lambda: {})()
                        for m in self._modules
                    }
                    self.events.append(
                        {
                            "stage": state.stage_idx,
                            "step": state.step,
                            "global_step": state.global_step,
                            "action": action,
                            "target": target,
                            "level": self.level,
                            "capacity": cap,
                            "loss": float(state.data_loss),
                        }
                    )
                    log.debug("GrowCapacity: %s (%s) at step %d", action, target, state.step)
                    done = True
                    break
            if not done and self.stop_when_exhausted:
                solver.stop_stage = "capacity_exhausted"
            self.detector.reset()
        if self.use_progress:
            solver.progress_override = self.progress


__all__ = ["GrowCapacity", "Growable", "field_space_gradient", "growable_modules"]
