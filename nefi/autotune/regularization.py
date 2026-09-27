"""Regularization strength by the discrepancy principle (Morozov), with a GradNorm fallback.

Morozov's principle chooses the *largest* regularization under which the data are still explained
to the noise level: the final data misfit ``RMSE(w)`` grows with the weight ``w`` of a penalty
(TV, ℓ1, Laplacian, …), so the weight that puts ``RMSE(w) = τ·σ`` (τ ≈ 1) is a root of a
monotone function of ``log w`` — found by bracketing (×10 steps) and bisection / regula falsi in
``log w``. Each trial is a short fit (``budget_scale`` × the curriculum's steps); trials are
*warm-started* from the fitted field of the nearest previous trial (a single native-resolution
stage without re-annealing), so the optimization accumulates along the search path (a
continuation method) and the last trials are close to converged even though each is short.

Several regularizers are tuned either *jointly* (one common factor, their ratios kept — the
default, a well-posed 1-D search) or *each* in turn. Stage-specific weights
(``Stage.loss_weights``) are scaled together with the base weights, so instances that repeat a
weight in every stage are tuned as well.

Tolerance: the weights are left alone while ``RMSE/τσ`` lies in ``band = (0.85, 1.1)`` — σ and
the effective number of fitted degrees of freedom are uncertain at that level (``m`` data fitted
with ``d`` effective parameters leave ``χ² ≈ 1 − d/m``; forcing ``χ = 1`` there over-smooths:
poisson_source at 102 observations would need a 300× stronger ℓ1 and lose 2.5 dB). Outside the
band the root is searched to ``rtol = 5 %``.

Budget-limited problems: when even ``w / 10⁴`` leaves the misfit above ``τσ`` the target is out
of reach at this budget — the regularizer is not what limits the fit — and the weight is only
lowered to where it stops mattering (the largest weight whose misfit is within 5 % of the
weakest trial), else kept. Without a noise level the tuner falls back to
:class:`~nefi.solve.GradNormBalancing` shares (each regularizer's weighted gradient norm a fixed
share of the data term's).

Known limitation: Morozov chooses the *largest* admissible regularization; with a prior that does
not match the unknown (TV on spikes) that over-smooths — the held-out search
(:func:`nefi.autotune.autotune`) is the better judge there.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..errors import ConfigError
from ..solve.curriculum import Curriculum, Stage
from ._common import (
    Timer,
    base_curriculum,
    effective_weight,
    fit,
    fmt,
    md_table,
    noise_float,
    probe_version,
    rescaled,
    scale_weights,
    with_sigma,
)

log = logging.getLogger("nefi")

MODES = ("joint", "each")


class RegularizationResult(dict):
    """``{term: tuned base weight}`` plus the evidence (a ``dict`` subclass).

    Attributes:
        factors: multiplier applied to each term (base weight and stage overrides).
        before / after: effective final-stage weight of each term.
        status: ``"discrepancy"`` (root found), ``"budget_limited"``, ``"bracket"`` (search
            range exhausted), ``"unchanged"`` (misfit inside the tolerance band), ``"gradnorm"``
            (σ unknown) or ``"no_regularizers"``.
        sigma / tau: noise level and Morozov factor (target ``RMSE = τσ``).
        trials: one dict per fit (``names``, ``factor``, ``chi``, ``rmse``, ``steps``, ``warm``).
        curriculum: the input curriculum with the tuned stage overrides.
        probe_steps / seconds: cost.
    """

    def __init__(self, weights: Mapping[str, float] | None = None, **attrs: Any) -> None:
        super().__init__(weights or {})
        self.factors: dict[str, float] = {}
        self.before: dict[str, float] = {}
        self.after: dict[str, float] = {}
        self.status = "unchanged"
        self.sigma: float | None = None
        self.tau = 1.0
        self.trials: list[dict[str, Any]] = []
        self.curriculum: Curriculum | None = None
        self.base_weights: dict[str, float] = {}
        self.notes: list[str] = []
        self.probe_steps = 0
        self.seconds = 0.0
        for k, v in attrs.items():
            setattr(self, k, v)

    def apply(self, problem: Any, curriculum: Curriculum | None = None) -> tuple[Any, Curriculum]:
        """``(problem copy with the tuned base weights, curriculum with tuned overrides)``.

        The loss modules are shared; the caller's problem and curriculum are not modified.
        """
        import dataclasses

        cur = base_curriculum(problem, curriculum)
        weights, cur = scale_weights(problem.losses.weights, cur, self.factors)
        losses = problem.losses.with_weights({k: weights[k] for k in self.factors})
        new = dataclasses.replace(problem, losses=losses, curriculum=cur, meta=dict(problem.meta))
        return new, cur

    def to_markdown(self) -> str:
        rows = [[k, self.before.get(k), self.after.get(k), self.factors.get(k)] for k in self]
        lines = [
            f"**{self.status}** — target RMSE = {fmt(self.tau)}·σ, σ = {fmt(self.sigma)}",
            "",
            md_table(["term", "weight before", "weight after", "factor"], rows),
        ]
        if self.trials:
            trows = [
                [t["names"], t["factor"], t["chi"], t["rmse"], t["steps"], t["warm"]]
                for t in self.trials
            ]
            lines += [
                "",
                md_table(["terms", "factor", "χ = RMSE/σ", "RMSE", "steps", "warm start"], trows),
            ]
        lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(self),
            "factors": dict(self.factors),
            "before": dict(self.before),
            "after": dict(self.after),
            "status": self.status,
            "sigma": self.sigma,
            "tau": self.tau,
            "trials": list(self.trials),
            "notes": list(self.notes),
            "probe_steps": self.probe_steps,
            "seconds": self.seconds,
        }


def regularizer_names(problem: Any, curriculum: Curriculum | None = None) -> list[str]:
    """Non-data loss terms active in some stage (gauge penalties excluded)."""
    losses = problem.losses
    data = set(losses.data_terms())
    cur = curriculum or getattr(problem, "curriculum", None)
    out = []
    for k in losses.names:
        if k in data or k.startswith("gauge_"):
            continue
        ws = [losses.weights[k]]
        if cur is not None:
            ws += [float((s.loss_weights or {}).get(k, losses.weights[k])) for s in cur.stages]
        if any(w != 0.0 for w in ws):
            out.append(k)
    return out


def _warm_curriculum(cur: Curriculum, steps: int, lr_scale: float) -> Curriculum:
    """One native-resolution stage continuing a fit: no re-annealing, cosine decay."""
    last = cur.stages[-1]
    st = Stage(
        "warm",
        last.shape,
        max(2, int(steps)),
        float(last.lr) * lr_scale,
        lr_schedule="cosine",
        lr_min_ratio=last.lr_min_ratio,
        anneal=False,
        loss_weights=dict(last.loss_weights) if last.loss_weights else None,
        freeze=tuple(last.freeze),
    )
    return Curriculum([st], optim=copy.deepcopy(cur.optim), restarts=1)


class _Search:
    """Log-space root finding of ``log(χ(f)/τ)`` over a common factor ``f`` on some terms."""

    def __init__(self, problem, names, base_w, cur, budget, sigma, tau, kw) -> None:
        self.problem, self.names, self.base_w, self.cur = problem, list(names), base_w, cur
        self.budget, self.sigma, self.tau, self.kw = int(budget), sigma, tau, kw
        self.trials: list[dict[str, Any]] = []
        self.fields: list[tuple[float, Any]] = []  # (log f, fitted field)
        self.steps = 0

    def value(self, f: float, start: Any = None) -> float:
        weights, cur = scale_weights(self.base_w, self.cur, dict.fromkeys(self.names, f))
        warm = self.kw["warm_start"]
        field = start
        if field is None and warm and self.fields:
            ref = self.fields[0] if warm == "anchor" else None
            if ref is None:  # the most recent of the closest trials
                ref = min(reversed(self.fields), key=lambda t: abs(t[0] - math.log(f)))
            field = ref[1]
        if field is None:
            curriculum = probe_version(rescaled(cur, self.budget))
        else:
            curriculum = _warm_curriculum(cur, self.budget, self.kw["warm_lr"])
        out = fit(
            self.problem,
            curriculum,
            field=field,
            weights=weights,
            sigma=self.sigma,
            seed=self.kw["seed"],
            device=self.kw["device"],
        )
        self.steps += out.steps
        chi = out.chi if out.chi is not None and math.isfinite(out.chi) else math.inf
        self.trials.append(
            {
                "names": "+".join(self.names),
                "factor": float(f),
                "chi": float(chi),
                "rmse": float(out.rmse),
                "steps": int(out.steps),
                "warm": field is not None,
            }
        )
        if out.ok:
            self.fields.append((math.log(f), out.field))
        g = math.log(max(chi, 1e-300) / self.tau) if math.isfinite(chi) else math.inf
        log.debug("tune_regularization %s ×%.3g: χ = %.4g", self.names, f, chi)
        return g


def _solve_factor(
    search: _Search, bracket, max_trials, rtol, band, init, init_chi, settle: int
) -> tuple[float, str, str]:
    lo_f, hi_f = bracket
    tol = math.log1p(rtol)
    evals: list[tuple[float, float]] = []  # (log f, g)

    def ev(f: float, start: Any = None) -> float:
        g = search.value(f, start)
        evals.append((math.log(f), g))
        return g

    if init is not None and init_chi is not None and math.isfinite(init_chi):
        g1 = math.log(max(float(init_chi), 1e-300) / search.tau)  # a converged fit: no trial
        search.fields.append((0.0, init))
    else:
        g1 = ev(1.0, init)
        # settle: continue at the current weight until the misfit stops moving, so that the
        # search direction reflects the weight and not the extra optimization of warm starts
        for _ in range(settle if (search.kw["warm_start"] and init is None) else 0):
            g_next = ev(1.0)
            moved = abs(g_next - g1) > tol
            g1 = g_next
            if not moved:
                break
    evals[:] = [(0.0, g1)]  # the (settled) value represents f = 1
    chi1 = math.exp(g1) * search.tau
    if band[0] * search.tau <= chi1 <= band[1] * search.tau:
        return (
            1.0,
            "unchanged",
            (
                f"χ = {fmt(chi1)} is inside the tolerance band [{fmt(band[0])}, {fmt(band[1])}]·τ "
                "(σ and the effective degrees of freedom are uncertain at that level): kept"
            ),
        )
    if abs(g1) <= tol:
        return 1.0, "discrepancy", f"χ = {fmt(chi1)} already at the target"
    step = math.log(10.0)
    direction = 1.0 if g1 < 0 else -1.0  # below the target: more regularization
    limit = math.log(hi_f if direction > 0 else lo_f)
    x, g = 0.0, g1
    while len(evals) < max_trials:
        nx = x + direction * step
        if (direction > 0 and nx > limit + 1e-9) or (direction < 0 and nx < limit - 1e-9):
            break
        ng = ev(math.exp(nx))
        if (ng > 0) != (g > 0) or abs(ng) <= tol:
            x, g = nx, ng
            break
        x, g = nx, ng
    if abs(g) <= tol:
        return math.exp(x), "discrepancy", "root found while bracketing"
    pos = [e for e in evals if e[1] > 0 and math.isfinite(e[1])]
    neg = [e for e in evals if e[1] <= 0]
    if not pos or not neg:
        chis = [(lx, math.exp(gg) * search.tau) for lx, gg in evals if math.isfinite(gg)]
        if direction < 0 and chis:  # misfit above target even for the weakest regularization
            weakest = min(chis, key=lambda t: t[0])
            base_chi = math.exp(g1) * search.tau
            if weakest[1] < 0.9 * base_chi:
                # lower the weight only as far as it still matters (knee)
                ok = [lx for lx, c in chis if c <= 1.05 * weakest[1]]
                f = math.exp(max(ok))
                return (
                    f,
                    "budget_limited",
                    f"χ stays above {fmt(search.tau)} down to ×{fmt(math.exp(weakest[0]))} "
                    f"(χ = {fmt(weakest[1])}): the fit is budget-limited; weight lowered to where "
                    "it stops raising the misfit",
                )
            return (
                1.0,
                "budget_limited",
                f"χ = {fmt(base_chi)} > {fmt(search.tau)} does not respond to the weight (down to "
                f"×{fmt(math.exp(weakest[0]))}): the misfit is budget- or representation-limited, "
                "weight unchanged",
            )
        chi_x = math.exp(g) * search.tau if math.isfinite(g) else math.inf
        chi_1 = math.exp(g1) * search.tau
        if not chi_x > 1.05 * chi_1:  # the term barely affects the misfit: leave it alone
            return (
                1.0,
                "unchanged",
                f"χ stays at {fmt(chi_x)} up to ×{fmt(math.exp(x))} (from {fmt(chi_1)} at ×1): "
                "the term does not control the misfit, weight unchanged",
            )
        return (
            math.exp(x),
            "bracket",
            f"χ = {fmt(chi_x)} is still below the target at ×{fmt(math.exp(x))} (from "
            f"{fmt(chi_1)} at ×1): the search range is exhausted; the weight is set to its edge",
        )
    # regula falsi (Illinois) in log f between the closest bracketing pair
    a = max(neg, key=lambda e: e[1])  # g <= 0, closest to 0
    b = min(pos, key=lambda e: e[1])  # g > 0, closest to 0
    side = 0
    while len(evals) < max_trials:
        xa, ga = a
        xb, gb = b
        x = xa - ga * (xb - xa) / (gb - ga) if gb != ga else 0.5 * (xa + xb)
        if not (min(xa, xb) < x < max(xa, xb)):
            x = 0.5 * (xa + xb)
        g = ev(math.exp(x))
        if abs(g) <= tol:
            return math.exp(x), "discrepancy", "root found"
        if g > 0:
            b = (x, g)
            if side == 1:
                a = (a[0], a[1] / 2)
            side = 1
        else:
            a = (x, g)
            if side == -1:
                b = (b[0], b[1] / 2)
            side = -1
    xa, ga = a
    xb, gb = b
    x = xa - ga * (xb - xa) / (gb - ga) if gb != ga else 0.5 * (xa + xb)
    return math.exp(x), "discrepancy", "root interpolated after the trial budget"


def tune_regularization(
    problem: Any,
    names: Sequence[str] | None = None,
    *,
    sigma: float | None = None,
    budget_scale: float = 0.2,
    tau: float = 1.0,
    mode: str = "joint",
    curriculum: Curriculum | None = None,
    bracket: tuple[float, float] = (1e-4, 1e4),
    max_trials: int = 10,
    rtol: float = 0.05,
    band: tuple[float, float] = (0.85, 1.1),
    warm_start: bool | str = "nearest",
    warm_lr: float = 0.5,
    init: Any = None,
    init_chi: float | None = None,
    settle: int = 2,
    shares: Mapping[str, float] | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> RegularizationResult:
    """Regularization weights by the discrepancy principle: final ``RMSE ≈ τ·σ``.

    Args:
        problem: the problem (not modified; use ``result.apply(problem)``).
        names: loss terms to tune (default: every active non-data term except gauge penalties).
        sigma: noise level (default: ``measurement.noise_std``; if unknown, GradNorm fallback).
        budget_scale: steps per trial as a fraction of the curriculum's total.
        tau: Morozov factor (target ``RMSE = τσ``).
        mode: ``"joint"`` (one factor for all ``names``, ratios kept) or ``"each"`` (one term
            after the other).
        curriculum: base curriculum (default: the problem's).
        bracket: ``(min, max)`` multiplier searched.
        max_trials: fits of the bracketing and root search (the settle rounds come on top).
        rtol: relative tolerance on ``RMSE / τσ`` of the root finding.
        band: the weights are left alone while ``RMSE / τσ`` lies in this band — the noise level
            and the effective number of fitted degrees of freedom are uncertain at the 10 %
            level (a good fit of ``m`` data with ``d`` effective parameters has
            ``χ² ≈ 1 − d/m``: 0.91 for 17 parameters and 102 data).
        warm_start: ``"nearest"`` (continue from the fitted field of the trial with the closest
            weight — a continuation along the search path), ``"anchor"`` (always from the first
            trial) or ``False`` (every trial from the initial field).
        warm_lr: learning-rate factor of warm-started trials (relative to the final stage).
        init: optional fitted field to warm-start the first trial from (e.g. a converged probe).
        init_chi: the misfit of ``init``; with both, no trial is run at the current weights.
        settle: extra warm-started rounds at the current weight before the search starts (until
            the misfit changes by less than ``rtol``), so that short, still-converging trials
            are not mistaken for a weight effect.
        shares: GradNorm target shares for the σ-unknown fallback (default 0.1 per term).
        seed / device: forwarded to the solver.

    Returns:
        :class:`RegularizationResult` — ``{term: tuned base weight}`` with the evidence.

    Example::

        weights = tune_regularization(problem, ["tv"])
        problem, curriculum = weights.apply(problem)
    """
    if mode not in MODES:
        raise ConfigError(f"mode must be one of {MODES}, got {mode!r}")
    if warm_start not in (True, False, "nearest", "anchor"):
        raise ConfigError("warm_start must be 'nearest', 'anchor', True or False")
    timer = Timer()
    base_cur = base_curriculum(problem, curriculum)
    terms = list(names) if names is not None else regularizer_names(problem, base_cur)
    unknown = [k for k in terms if k not in problem.losses.names]
    if unknown:
        raise ConfigError(f"unknown loss terms {unknown}; known: {problem.losses.names}")
    data = set(problem.losses.data_terms())
    if any(k in data for k in terms):
        raise ConfigError(f"{[k for k in terms if k in data]} are data terms, not regularizers")
    base_w = dict(problem.losses.weights)
    res = RegularizationResult(base_weights=dict(base_w), tau=float(tau))
    res.before = {k: effective_weight(base_w, base_cur, k) for k in terms}
    if not terms:
        res.status = "no_regularizers"
        res.notes.append("no active regularizer to tune")
        res.curriculum = base_cur
        return res
    s = sigma if sigma is not None else noise_float(problem.measurement.noise_std)
    res.sigma = s
    budget = max(10 * len(base_cur.stages), int(round(budget_scale * base_cur.total_steps)))
    if s is None or not s > 0:
        return _gradnorm(problem, terms, base_cur, budget, shares, res, timer, seed, device)
    prob = with_sigma(problem, float(s))
    kw = {
        "warm_start": "nearest" if warm_start is True else warm_start,
        "warm_lr": float(warm_lr),
        "seed": seed,
        "device": device,
    }
    groups = [terms] if mode == "joint" else [[k] for k in terms]
    factors: dict[str, float] = dict.fromkeys(terms, 1.0)
    statuses = []
    cur_w, cur_c = base_w, base_cur
    start = init
    for group in groups:
        search = _Search(prob, group, cur_w, cur_c, budget, float(s), float(tau), kw)
        f, status, why = _solve_factor(
            search, bracket, max_trials, rtol, band, start, init_chi if start is init else None,
            settle,
        )  # fmt: skip
        res.trials += search.trials
        timer.add(steps=search.steps)
        statuses.append(status)
        for k in group:
            factors[k] *= f
        cur_w, cur_c = scale_weights(cur_w, cur_c, dict.fromkeys(group, f))
        res.notes.append(f"{'+'.join(group)}: ×{fmt(f)} — {why}")
        if kw["warm_start"] and search.fields:  # the next group continues from the best fit
            best = min(search.fields, key=lambda t: abs(t[0] - math.log(f)))
            start = best[1]
    res.factors = factors
    res.update({k: cur_w[k] for k in terms})
    res.after = {k: effective_weight(cur_w, cur_c, k) for k in terms}
    res.curriculum = cur_c
    order = ("discrepancy", "bracket", "budget_limited", "unchanged")
    res.status = next((st for st in order if st in statuses), "unchanged")
    if all(abs(math.log(f)) < 1e-12 for f in factors.values()) and res.status != "discrepancy":
        res.status = "unchanged"
    res.probe_steps, res.seconds = timer.steps, timer.seconds
    log.info(
        "tune_regularization (%s): %s",
        res.status,
        ", ".join(f"{k} {fmt(res.before[k])} → {fmt(res.after[k])}" for k in terms),
    )
    return res


def _gradnorm(problem, terms, cur, budget, shares, res, timer, seed, device):
    """σ unknown: one probe with :class:`~nefi.solve.GradNormBalancing`, keep its weights."""
    from ..solve.balancing import GradNormBalancing

    sh = {k: float((shares or {}).get(k, 0.1)) for k in terms}
    for k in problem.losses.names:
        if k not in terms and k not in problem.losses.data_terms():
            sh[k] = 0.0  # leave the other terms alone
    every = max(5, budget // 10)
    cb = GradNormBalancing(shares=sh, every=every, warmup=every)
    probe = probe_version(rescaled(cur, budget))
    out = fit(problem, probe, callbacks=[cb], seed=seed, device=device)
    timer.add(out)
    res.status = "gradnorm"
    res.notes.append(
        "noise level unknown: weights from GradNormBalancing (each regularizer's weighted "
        f"gradient norm at share {', '.join(f'{k}={v:g}' for k, v in sh.items() if k in terms)} "
        "of the data term's)"
    )
    final = cb.log[-1] if cb.log else {}
    factors = {}
    for k in terms:
        w_new = final.get(f"w/{k}")
        before = res.before[k]
        factors[k] = float(w_new) / before if (w_new is not None and before > 0) else 1.0
    w, c = scale_weights(problem.losses.weights, cur, factors)
    res.factors = factors
    res.update({k: w[k] for k in terms})
    res.after = {k: effective_weight(w, c, k) for k in terms}
    res.curriculum = c
    res.trials.append(
        {
            "names": "+".join(terms),
            "factor": math.nan,
            "chi": math.nan,
            "rmse": float(out.rmse),
            "steps": int(out.steps),
            "warm": False,
        }
    )
    res.probe_steps, res.seconds = timer.steps, timer.seconds
    return res


__all__ = ["MODES", "RegularizationResult", "regularizer_names", "tune_regularization"]
