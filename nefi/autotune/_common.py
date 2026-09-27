"""Shared machinery of the auto-tuners (internal): noise level, curricula, probe fits, reports.

Every tuner works on *copies*: a probe deep-copies ``problem.field`` (so each probe starts from the
same initialization and the caller's field is never trained), shares the operator (pure by
contract; its trainable nuisance parameters are snapshotted and restored after each probe) and
shares the loss modules (weight overrides go through :meth:`LossSet.with_weights`).
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from ..errors import ConfigError
from ..measurement import Measurement
from ..solve.curriculum import Curriculum
from ..solve.result import Result
from ..solve.solver import Solver

log = logging.getLogger("nefi")


# ------------------------------------------------------------------------------------------
# noise level
# ------------------------------------------------------------------------------------------
def noise_float(ns: Any) -> float | None:
    """A measurement's ``noise_std`` as one float (mean of a tensor), ``None`` if unset."""
    if ns is None:
        return None
    if torch.is_tensor(ns):
        return float(ns.detach().double().mean())
    return float(ns)


def resolve_sigma(
    problem: Any, sigma: float | None = None, *, estimate: bool = True
) -> tuple[float | None, str]:
    """Noise level ``σ`` and where it came from: given > ``measurement.noise_std`` > estimate.

    The estimate prefers the operator's own ``estimate_noise`` hook and falls back to
    :func:`nefi.auto.estimate_noise` (Immerkær Laplacian / high-order differences / MAD). Never
    writes into the problem's measurement.

    Returns:
        ``(σ, source)``; ``σ`` is ``None`` (source ``"unknown"``) when it is not known and
        ``estimate`` is False or the estimator failed.
    """
    if sigma is not None:
        if not float(sigma) > 0:
            raise ConfigError(f"sigma must be positive, got {sigma}")
        return float(sigma), "given"
    meas = problem.measurement
    ns = noise_float(meas.noise_std)
    if ns is not None and ns > 0:
        est = bool(meas.meta.get("noise_std_estimated"))
        return ns, "measurement (estimated earlier)" if est else "measurement"
    if not estimate:
        return None, "unknown"
    from ..auto import _estimate

    probe = Measurement(meas.data, meas.mask, None, {})
    try:
        s = float(_estimate(probe, problem.operator, store=True))
    except Exception as e:  # noqa: BLE001 - reported, the caller falls back
        log.warning("autotune: noise estimation failed (%s)", e)
        return None, f"unknown (estimation failed: {type(e).__name__})"
    if not (math.isfinite(s) and s > 0):
        return None, "unknown (estimate is zero or not finite)"
    return s, f"estimated ({probe.meta.get('noise_estimator', 'estimate_noise')})"


def with_sigma(problem: Any, sigma: float | None, source: str = "given") -> Any:
    """Copy of ``problem`` whose measurement carries ``noise_std = σ`` (discrepancy stopping).

    The measurement tensors are shared; only the container is new. Returns ``problem`` itself
    when the value is already set.
    """
    meas = problem.measurement
    if sigma is None or noise_float(meas.noise_std) == float(sigma):
        return problem
    new = Measurement(meas.data, meas.mask, float(sigma), dict(meas.meta))
    if source.startswith("estimated"):
        new.meta["noise_std_estimated"] = True
    return dataclasses.replace(problem, measurement=new, meta=dict(problem.meta))


# ------------------------------------------------------------------------------------------
# curricula
# ------------------------------------------------------------------------------------------
def base_curriculum(problem: Any, curriculum: Curriculum | None = None) -> Curriculum:
    """Deep copy of the curriculum a solve would use (explicit > ``problem.curriculum`` >
    ``Curriculum.multiscale(domain.shape)``, the :class:`~nefi.solve.Solver` default)."""
    cur = curriculum or getattr(problem, "curriculum", None)
    if cur is None:
        cur = Curriculum.multiscale(problem.domain.shape)
    return copy.deepcopy(cur)


def rescaled(cur: Curriculum, total: int, min_stage: int = 2) -> Curriculum:
    """Copy of ``cur`` with about ``total`` steps, stage proportions kept (each >= min_stage)."""
    total = max(int(total), min_stage * len(cur.stages))
    c = copy.deepcopy(cur)
    ref = max(1, cur.total_steps)
    for s in c.stages:
        s.steps = max(int(min_stage), int(round(s.steps * total / ref)))
    return c


def probe_version(cur: Curriculum) -> Curriculum:
    """Copy for probes: one restart, no early / discrepancy / wall-clock stopping."""
    c = copy.deepcopy(cur)
    c.restarts = 1
    c.early_stop_patience = None
    c.discrepancy_tau = None
    c.time_budget_s = None
    return c


def peak_lr(cur: Curriculum) -> float:
    """Largest stage learning rate of a curriculum."""
    return max(float(s.lr) for s in cur.stages)


def scale_lr(cur: Curriculum, factor: float) -> Curriculum:
    """Copy with every stage's learning rate multiplied by ``factor`` (ratios kept)."""
    c = copy.deepcopy(cur)
    for s in c.stages:
        s.lr = float(s.lr) * float(factor)
    return c


def scale_weights(
    base: Mapping[str, float], cur: Curriculum, factors: Mapping[str, float]
) -> tuple[dict[str, float], Curriculum]:
    """Multiply loss terms by ``factors`` everywhere they are set.

    The base weights *and* every stage override (``Stage.loss_weights``) of a term are scaled by
    the same factor, so the per-stage structure is kept (a stage that switches a term off keeps it
    off; a stage that sets its own value — e.g. the instances that repeat ``tv`` in every stage —
    is tuned too).

    Returns:
        ``(new_base_weights, new_curriculum)``.
    """
    w = {k: float(v) for k, v in base.items()}
    c = copy.deepcopy(cur)
    for k, f in factors.items():
        if k in w:
            w[k] *= float(f)
        for s in c.stages:
            if s.loss_weights and k in s.loss_weights:
                s.loss_weights = {**s.loss_weights, k: float(s.loss_weights[k]) * float(f)}
    return w, c


def effective_weight(base: Mapping[str, float], cur: Curriculum, name: str) -> float:
    """Weight of ``name`` in the final stage (its override, else the base weight)."""
    last = cur.stages[-1].loss_weights or {}
    return float(last.get(name, base.get(name, 0.0)))


# ------------------------------------------------------------------------------------------
# probe fits
# ------------------------------------------------------------------------------------------
def residual(pred: torch.Tensor, meas: Measurement) -> tuple[float, float]:
    """``(RMSE, relative RMSE)`` of ``pred`` against the observed entries of ``meas``."""
    p = pred.detach()
    y = meas.data.detach().to(p.device)
    m = meas.mask
    if p.is_complex() or y.is_complex():
        p = torch.view_as_real(p.to(torch.complex128))
        y = torch.view_as_real(y.to(torch.complex128))
        if m is not None:
            m = m.real if m.is_complex() else m
            m = torch.broadcast_to(m.to(p.device), meas.data.shape).unsqueeze(-1)
    p, y = p.double(), y.double()
    if tuple(p.shape) != tuple(y.shape):
        raise ConfigError(f"prediction {tuple(p.shape)} vs measurement {tuple(y.shape)}")
    r2 = (p - y) ** 2
    if m is None:
        mse, ms = float(r2.mean()), float((y**2).mean())
    else:
        w = m.detach().to(p.device, torch.float64).expand_as(r2)
        den = float(w.sum())
        if den <= 0:
            raise ConfigError("the measurement mask has no observed entry")
        mse, ms = float((r2 * w).sum()) / den, float((y**2 * w).sum()) / den
    rmse = math.sqrt(max(mse, 0.0))
    return rmse, rmse / math.sqrt(ms) if ms > 0 else math.inf


@dataclass
class FitOutcome:
    """One probe fit: the solver result, its data misfit and where the fitted field lives."""

    result: Result
    field: Any
    rmse: float
    chi: float | None
    data_loss: float
    steps: int
    seconds: float
    progress: float
    stops: list[str]

    @property
    def ok(self) -> bool:
        return math.isfinite(self.rmse) and math.isfinite(self.data_loss)


def fit(
    problem: Any,
    curriculum: Curriculum,
    *,
    field: Any = None,
    weights: Mapping[str, float] | None = None,
    measurement: Measurement | None = None,
    sigma: float | None = None,
    seed: int = 0,
    device: str | torch.device = "cpu",
    callbacks: Sequence[Any] = (),
    solver_kw: Mapping[str, Any] | None = None,
) -> FitOutcome:
    """Solve a private copy of ``problem`` with ``curriculum`` and measure the data misfit.

    Args:
        problem: the problem (never modified: its field is deep-copied, the operator's trainable
            parameters are restored afterwards).
        curriculum: the curriculum to run.
        field: start from this field instead of ``problem.field`` (deep-copied; warm starts).
        weights: loss-weight overrides (:meth:`LossSet.with_weights`).
        measurement: fit this measurement instead (e.g. a hold-out training mask).
        sigma: noise level for ``chi = RMSE / σ`` (default: the measurement's).
        seed / device / callbacks / solver_kw: forwarded to :class:`~nefi.solve.Solver`.
    """
    fld = copy.deepcopy(problem.field if field is None else field)
    # always a private LossSet (shared modules): callbacks such as GradNormBalancing mutate
    # the weights of the stage copy, which must never be the caller's LossSet
    losses = problem.losses.with_weights(dict(weights or problem.losses.weights))
    meas = problem.measurement if measurement is None else measurement
    prob = dataclasses.replace(
        problem,
        field=fld,
        losses=losses,
        measurement=meas,
        curriculum=curriculum,
        meta=dict(problem.meta),
    )
    op_params = {k: p.detach().clone() for k, p in problem.operator.named_parameters()}
    t0 = time.perf_counter()
    try:
        res = Solver(
            prob,
            curriculum,
            device=device,
            seed=seed,
            callbacks=list(callbacks),
            **(solver_kw or {}),
        ).run()
    finally:
        with torch.no_grad():
            for k, p in problem.operator.named_parameters():
                if k in op_params:
                    p.copy_(op_params[k].to(p))
    secs = time.perf_counter() - t0
    target = meas
    if tuple(res.pred.shape) != tuple(meas.data.shape):
        target = prob.measurement_at(tuple(res.fields[prob.field.primary].shape))
    try:
        rmse, _ = residual(res.pred, target)
    except ConfigError:
        rmse = math.inf
    s = sigma if sigma is not None else noise_float(meas.noise_std)
    hist = res.history.get("data_loss") or [math.nan]
    last = res.stage_results[-1] if res.stage_results else {}
    return FitOutcome(
        result=res,
        field=fld,
        rmse=rmse,
        chi=(rmse / s) if s else None,
        data_loss=float(hist[-1]),
        steps=len(res.history.get("total", [])),
        seconds=secs,
        progress=float(last.get("final_progress", 1.0)),
        stops=[str(r.get("stop", "?")) for r in res.stage_results],
    )


# ------------------------------------------------------------------------------------------
# decisions and formatting
# ------------------------------------------------------------------------------------------
@dataclass
class Decision:
    """One automatic decision: what changed, from what to what, why, and what it cost."""

    step: str
    what: str
    before: Any = None
    after: Any = None
    why: str = ""
    cost_steps: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "what": self.what,
            "before": jsonable(self.before),
            "after": jsonable(self.after),
            "why": self.why,
            "cost_steps": int(self.cost_steps),
            "seconds": float(self.seconds),
        }


def fmt(v: Any, digits: int = 3) -> str:
    """Compact number formatting for reports (``—`` for ``None``)."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "∞" if v > 0 else "-∞"
        if v != 0 and (abs(v) >= 1e4 or abs(v) < 1e-3):
            return f"{v:.{digits - 1}e}"
        return f"{v:.{digits}g}"
    if isinstance(v, Mapping):
        return ", ".join(f"{k}={fmt(x, digits)}" for k, x in v.items())
    if isinstance(v, list | tuple):
        return "[" + ", ".join(fmt(x, digits) for x in v) + "]"
    return str(v)


def md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """A GitHub-flavoured markdown table (cells formatted with :func:`fmt`)."""
    head = "| " + " | ".join(header) + " |"
    sep = "|" + "|".join("---" for _ in header) + "|"
    body = ["| " + " | ".join(c if isinstance(c, str) else fmt(c) for c in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def jsonable(obj: Any) -> Any:
    """JSON-friendly copy (dataclasses / tensors / curricula → plain Python)."""
    from ..bench.report import to_jsonable
    from ..config import to_dict

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = to_dict(obj)
    return to_jsonable(obj)


@dataclass
class Timer:
    """Wall-clock and optimization-step accounting of a tuner."""

    t0: float = field(default_factory=time.perf_counter)
    steps: int = 0

    @property
    def seconds(self) -> float:
        return time.perf_counter() - self.t0

    def add(self, outcome: FitOutcome | None = None, steps: int = 0) -> None:
        self.steps += int(steps) + (outcome.steps if outcome is not None else 0)


__all__ = [
    "Decision",
    "FitOutcome",
    "Timer",
    "base_curriculum",
    "effective_weight",
    "fit",
    "fmt",
    "jsonable",
    "md_table",
    "noise_float",
    "peak_lr",
    "probe_version",
    "rescaled",
    "residual",
    "resolve_sigma",
    "scale_lr",
    "scale_weights",
    "with_sigma",
]
