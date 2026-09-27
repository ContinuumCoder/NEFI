"""Fit-quality driven budget and learning rate: does the fit reach the noise floor, and when?

A per-measurement inversion is *budget-limited* when its data misfit is still falling at the end
of the schedule (EIT smoke preset: ``RMSE = 1.38 σ`` after 200 steps, 16.5 dB; 1600 steps reach
``1.03 σ`` and 23.8 dB), *at the noise floor* when ``RMSE ≈ σ`` (more steps only fit noise unless
the discrepancy principle stops them), and *stalled* when the misfit plateaus far above ``σ`` —
then more steps do not help: the learning rate, the regularization strength, the representation
or the forward model must change.

:func:`probe_convergence` measures which case applies with short, curriculum-shaped probes:

1. **doubling budgets** — probes of ``steps = (50, 100, 200)`` at the base learning rate, each a
   complete schedule (all stages, cosine decay, annealing), so the final misfit
   ``χ(T) = RMSE / σ`` is what a budget of ``T`` steps delivers. While the misfit still falls
   above the floor — or sits at the floor while the *reconstruction itself* still changes by more
   than 5 % between doublings (``Δfield = ‖x_T − x_{T/2}‖ / ‖x_T − mean‖``; a data-free
   convergence check on the unknown: EIT keeps gaining 2.6 dB between 800 and 1600 steps while χ
   only moves from 1.04 to 1.03) — the budget keeps doubling (``extend`` more times, up to
   ``max_probe_steps``). Measuring beats extrapolating: the misfit of an annealed neural field
   does not follow a power law (it accelerates once the fine bands open);
2. **classification** — *at the noise floor* if ``χ ≤ 1.05`` (1.2 when σ was estimated),
   *converging* when the excess ``e = χ² − 1`` fell by ≥ 15 % over the last doubling, else
   *stalled*;
3. **learning rate** — searched only when the base rate does not reach the floor (a rate that
   wins a 50-step race usually plateaus higher over a long schedule: EIT at 3× the base rate
   stalls at χ = 1.16 where the base rate reaches 1.04): the base rate × {⅓, 3} and the
   :func:`nefi.auto.lr_range_test` pick (20 Adam steps per candidate) are compared at the last
   probe budget and adopted if they lower the misfit by ≥ 5 %; a base rate that diverges is
   replaced by the range-test pick before anything else;
4. **recommendation** — at the floor: the first probed schedule that reached it and settled
   (``χ ≤ 1`` or ``Δfield ≤ 5 %``), a validated recipe, with a discrepancy stop at its own misfit
   (``τ = min(1, χ_ref)``, so the stop cannot cut the validated schedule short); a default budget
   that already reaches the floor is *never shortened* (a shorter schedule can reach the misfit
   floor while the reconstruction is still improving). Converging at the probe cap: twice the
   last probe (at least the base budget) with the last probe's *absolute* annealing length (the
   solver checks discrepancy / early stops only after annealing, so a ramp stretched over a longer
   budget would overshoot the floor before the stop can fire) and ``τ = 1``. Stalled: the base
   budget, with the diagnosis in words.

:func:`tune_budget` turns the report into a :class:`~nefi.solve.Curriculum`: the problem's stage
structure (resolutions, per-stage weights, freezing) with the recommended steps and learning rate,
cosine decay, the rescaled annealing fraction and the recommended ``discrepancy_tau``.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors import ConfigError
from ..solve.curriculum import Curriculum
from ._common import (
    Timer,
    base_curriculum,
    fit,
    fmt,
    md_table,
    peak_lr,
    probe_version,
    rescaled,
    resolve_sigma,
    scale_lr,
    with_sigma,
)

log = logging.getLogger("nefi")

STATUSES = ("at_noise_floor", "converging", "stalled", "unknown")
#: minimum relative drop of the excess misfit per doubling that counts as "converging"
CONVERGING_DROP = 0.15
#: relative change of the reconstruction per doubling below which it counts as settled
FIELD_TOL = 0.05


@dataclass
class ProbeRecord:
    """One probe fit: budget, learning rate and the resulting misfit."""

    steps: int
    lr: float
    chi: float | None
    rmse: float
    data_loss: float
    seconds: float
    kind: str = "budget"
    #: ``‖x_T − x_{T/2}‖ / ‖x_T − mean x_T‖`` of the primary field against the previous budget
    field_change: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ConvergenceReport:
    """Outcome of :func:`probe_convergence`.

    Attributes:
        status: ``"at_noise_floor"``, ``"converging"``, ``"stalled"`` or ``"unknown"`` (no σ and
            no clear trend).
        sigma / sigma_source: noise level used for ``χ = RMSE / σ``.
        probes: the doubling-budget probes (chosen learning rate).
        lr_probes: the curriculum-shaped learning-rate probes.
        lr / base_lr: recommended and original peak learning rate.
        lr_range_pick: the :func:`~nefi.auto.lr_range_test` choice (``None`` if skipped).
        recommended_steps / base_steps: recommended and original total steps.
        anneal_scale: factor for every stage's ``anneal_fraction`` (keeps the reference probe's
            absolute annealing length).
        tau: recommended ``discrepancy_tau`` (``None``: keep the base curriculum's).
        reached_floor_at: the reference probe at the noise floor (``None`` if none): the first
            one whose misfit is at most 1 or whose reconstruction stopped changing, else the last
            one at the floor.
        rate: exponent ``α`` of ``e(T) ∝ T^{−α}`` over the last doubling (``e = χ² − 1``).
        floor: ``χ`` at or below which the fit counts as at the noise floor.
        overfit: the largest probe went below ``χ = 0.9`` (it fitted noise).
        reasons: the decisions in words.
        probe_steps / seconds: cost.
    """

    status: str
    sigma: float | None
    sigma_source: str
    probes: list[ProbeRecord]
    lr_probes: list[ProbeRecord]
    lr: float
    base_lr: float
    lr_range_pick: float | None
    recommended_steps: int
    base_steps: int
    anneal_scale: float = 1.0
    tau: float | None = None
    reached_floor_at: int | None = None
    rate: float | None = None
    floor: float = 1.05
    overfit: bool = False
    reasons: list[str] = field(default_factory=list)
    probe_steps: int = 0
    seconds: float = 0.0
    #: fitted fields of the probes, keyed by ``(steps, lr)`` (not serialized)
    fitted: dict[tuple[int, float], Any] = field(default_factory=dict, repr=False)

    def fitted_field(self, steps: int | None = None, lr: float | None = None) -> Any:
        """The fitted field of the probe with ``steps`` (default: the recommendation) and
        ``lr`` (default: the recommended rate), or ``None``."""
        key = (int(self.recommended_steps if steps is None else steps), float(lr or self.lr))
        return self.fitted.get(key)

    @property
    def chi(self) -> float | None:
        """Misfit of the largest probe."""
        return self.probes[-1].chi if self.probes else None

    def summary(self) -> str:
        chis = ", ".join(f"{p.steps}: {fmt(p.chi)}" for p in self.probes)
        return (
            f"{self.status} (χ at {chis}); recommend {self.recommended_steps} steps "
            f"(was {self.base_steps}) at lr {fmt(self.lr)} (was {fmt(self.base_lr)})"
        )

    def to_markdown(self) -> str:
        rows = [
            [p.kind, p.steps, p.lr, p.chi, p.rmse, p.data_loss, p.field_change, p.seconds]
            for p in self.lr_probes + self.probes
        ]
        head = ["probe", "steps", "lr", "χ = RMSE/σ", "RMSE", "data loss", "Δfield", "s"]
        lines = [
            f"**{self.status}** — noise σ = {fmt(self.sigma)} ({self.sigma_source}); floor "
            f"χ ≤ {fmt(self.floor)}; rate α = {fmt(self.rate)}",
            "",
            md_table(head, rows),
            "",
            f"Recommendation: {self.recommended_steps} steps (was {self.base_steps}), peak lr "
            f"{fmt(self.lr)} (was {fmt(self.base_lr)}"
            + (f", range test {fmt(self.lr_range_pick)}" if self.lr_range_pick else "")
            + f"), annealing fraction ×{fmt(self.anneal_scale)}, discrepancy τ = "
            + f"{fmt(self.tau)}.",
        ]
        lines += [f"- {r}" for r in self.reasons]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_markdown()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "sigma": self.sigma,
            "sigma_source": self.sigma_source,
            "probes": [p.to_dict() for p in self.probes],
            "lr_probes": [p.to_dict() for p in self.lr_probes],
            "lr": self.lr,
            "base_lr": self.base_lr,
            "lr_range_pick": self.lr_range_pick,
            "recommended_steps": self.recommended_steps,
            "base_steps": self.base_steps,
            "anneal_scale": self.anneal_scale,
            "tau": self.tau,
            "reached_floor_at": self.reached_floor_at,
            "rate": self.rate,
            "floor": self.floor,
            "overfit": self.overfit,
            "reasons": list(self.reasons),
            "probe_steps": self.probe_steps,
            "seconds": self.seconds,
            "summary": self.summary(),
        }


def _score(rec: ProbeRecord) -> float:
    if rec.chi is not None and math.isfinite(rec.chi):
        return rec.chi
    return rec.data_loss if math.isfinite(rec.data_loss) else math.inf


def _record(outcome, lr: float, kind: str) -> ProbeRecord:
    return ProbeRecord(
        steps=outcome.steps,
        lr=float(lr),
        chi=None if outcome.chi is None else float(outcome.chi),
        rmse=float(outcome.rmse),
        data_loss=float(outcome.data_loss),
        seconds=float(outcome.seconds),
        kind=kind,
    )


def _cosine(cur: Curriculum) -> Curriculum:
    """Copy whose ``constant`` / ``step`` stages use cosine decay (budget-scale free)."""
    c = copy.deepcopy(cur)
    for st in c.stages:
        if st.lr_schedule in ("constant", "step"):
            st.lr_schedule = "cosine"
    return c


def _finite(rec: ProbeRecord, chi: bool = False) -> bool:
    if chi:
        return rec.chi is not None and math.isfinite(rec.chi)
    return math.isfinite(rec.rmse) and math.isfinite(rec.data_loss)


def _range_pick(prob: Any, base: Curriculum, base_lr: float, timer: Timer) -> float | None:
    """:func:`nefi.auto.lr_range_test` over ``base_lr × {0.1, 0.3, 1, 3, 10}`` on a field copy."""
    from ..auto import lr_range_test

    grid = [base_lr * f for f in (0.1, 0.3, 1.0, 3.0, 10.0)]
    tmp = dataclasses.replace(prob, field=copy.deepcopy(prob.field), meta=dict(prob.meta))
    try:
        pick = float(lr_range_test(tmp, grid, steps=20, shape=base.stages[0].shape))
    except Exception as e:  # noqa: BLE001 - the curriculum probes still decide
        log.warning("probe_convergence: lr_range_test failed (%s)", e)
        return None
    timer.add(steps=20 * len(grid))
    return pick


def _trend(a: ProbeRecord, b: ProbeRecord, use_chi: bool) -> tuple[float, float | None]:
    """Relative drop of the excess misfit from probe ``a`` to ``b`` and the power-law rate."""
    if use_chi:
        ea = max(float(a.chi) ** 2 - 1.0, 1e-12)  # type: ignore[arg-type]
        eb = max(float(b.chi) ** 2 - 1.0, 1e-12)  # type: ignore[arg-type]
    else:
        ea, eb = a.data_loss, b.data_loss
    if not (math.isfinite(ea) and math.isfinite(eb)) or ea <= 0:
        return -math.inf, None
    drop = (ea - eb) / ea
    rate = None
    if eb > 0 and b.steps > a.steps:
        rate = math.log(ea / eb) / math.log(b.steps / a.steps)
    return drop, rate


def probe_convergence(
    problem: Any,
    *,
    steps: Sequence[int] = (50, 100, 200),
    extend: int = 3,
    max_probe_steps: int | None = None,
    sigma: float | None = None,
    curriculum: Curriculum | None = None,
    lr: float | None = None,
    lr_test: bool = True,
    lr_factors: Sequence[float] = (1.0 / 3.0, 3.0),
    max_steps: int | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> ConvergenceReport:
    """Short probes with doubling budgets: at the noise floor, converging, or stalled?

    Args:
        problem: the problem (not modified; probes run on copies).
        steps: initial probe budgets (total steps per probe, increasing).
        extend: extra doublings beyond ``steps[-1]`` while the misfit still falls above the
            floor, or sits at the floor while the reconstruction still changes by more than
            :data:`FIELD_TOL` per doubling (0 = only ``steps``).
        max_probe_steps: largest probe budget (default ``max(8 × steps[-1], 2 × base steps)``).
        sigma: noise level (default: the measurement's, else estimated with
            :func:`nefi.auto.estimate_noise`).
        curriculum: base curriculum (default: the problem's).
        lr: fix the peak learning rate (skips the learning-rate search).
        lr_test: add the :func:`nefi.auto.lr_range_test` pick to the learning-rate candidates.
        lr_factors: multipliers of the base rate tried at the last probe budget when the base
            rate does not reach the noise floor (``()`` disables the search).
        max_steps: cap on the recommended budget (default ``8 × max(steps[-1], base steps)``).
        seed / device: forwarded to the solver.

    Returns:
        :class:`ConvergenceReport` (``report.status``, ``report.recommended_steps``,
        ``report.lr``).

    Example::

        report = probe_convergence(problem)
        print(report.summary())
        curriculum = tune_budget(problem, report)
    """
    budgets = sorted({int(s) for s in steps})
    if not budgets or budgets[0] < 2:
        raise ConfigError("steps must be probe budgets >= 2")
    timer = Timer()
    s, src = resolve_sigma(problem, sigma)
    prob = with_sigma(problem, s, src) if s is not None else problem
    # probes run the schedule tune_budget returns: cosine decay (a step decay with an absolute
    # step size does not rescale with the budget)
    base = _cosine(probe_version(base_curriculum(problem, curriculum)))
    base_lr, base_steps = peak_lr(base), base.total_steps
    known = s is not None and "estimated" not in src
    floor = 1.05 if known else 1.2
    reasons: list[str] = []
    probe_cap = int(max_probe_steps or max(8 * budgets[-1], 2 * base_steps))
    cap = int(max_steps or 8 * max(budgets[-1], base_steps))

    fitted: dict[tuple[int, float], Any] = {}
    values: dict[tuple[int, float], Any] = {}
    primary = problem.field.primary

    def run(total: int, lr_value: float, kind: str) -> ProbeRecord:
        cur = scale_lr(rescaled(base, total), lr_value / base_lr)
        out = fit(prob, cur, sigma=s, seed=seed, device=device)
        timer.add(out)
        key = (int(out.steps), float(lr_value))
        fitted[key] = out.field
        values[key] = out.result.fields.get(primary)
        return _record(out, lr_value, kind)

    def change(a: ProbeRecord, b: ProbeRecord) -> float | None:
        xa, xb = values.get((a.steps, a.lr)), values.get((b.steps, b.lr))
        if xa is None or xb is None or tuple(xa.shape) != tuple(xb.shape):
            return None
        xa, xb = xa.double(), xb.double()
        den = float((xb - xb.mean()).norm())
        return float((xb - xa).norm()) / den if den > 0 else None

    # ---- doubling budgets at the base (or given) learning rate ------------------------------
    chosen = float(lr) if lr is not None else base_lr
    if lr is not None:
        reasons.append(f"learning rate fixed by the caller: {fmt(chosen)}")
    pick = None
    probes = [run(budgets[0], chosen, "budget")]
    if lr is None and not _finite(probes[0]):
        # the base rate diverges: bracket a stable one first
        pick = _range_pick(prob, base, base_lr, timer) if lr_test else None
        chosen = pick if pick is not None else base_lr / 10.0
        reasons.append(
            f"the base learning rate {fmt(base_lr)} diverged; restarting at {fmt(chosen)}"
        )
        probes = [run(budgets[0], chosen, "budget")]
    for b in budgets[1:]:
        probes.append(run(b, chosen, "budget"))
    for a, b in zip(probes, probes[1:]):
        b.field_change = change(a, b)
    use_chi = s is not None and all(_finite(p, chi=True) for p in probes)

    def at_floor(p: ProbeRecord) -> bool:
        return use_chi and p.chi is not None and p.chi <= floor

    def settled(p: ProbeRecord) -> bool:
        """At the floor and done: below the Morozov target, or the reconstruction no longer
        changes between doublings (a data-free convergence check on the unknown itself)."""
        if not at_floor(p):
            return False
        return p.chi <= 1.0 or (p.field_change is not None and p.field_change <= FIELD_TOL)

    extra = 0
    while extra < extend and len(probes) >= 2 and not settled(probes[-1]):
        if not at_floor(probes[-1]):
            drop, _ = _trend(probes[-2], probes[-1], use_chi)
            if drop < CONVERGING_DROP:
                break
        nxt = 2 * probes[-1].steps
        if nxt > probe_cap:
            break
        probes.append(run(nxt, chosen, "budget"))
        probes[-1].field_change = change(probes[-2], probes[-1])
        use_chi = use_chi and _finite(probes[-1], chi=True)
        extra += 1
    # ---- learning rate: only searched when the base rate does not reach the floor -----------
    lr_probes: list[ProbeRecord] = []
    if lr is None and not any(at_floor(p) for p in probes) and lr_factors:
        last = probes[-1]
        cands = [chosen * float(f) for f in lr_factors if abs(float(f) - 1.0) > 1e-9]
        if lr_test and pick is None:
            pick = _range_pick(prob, base, base_lr, timer)
            if pick is not None and all(
                abs(math.log(pick / c)) > math.log(1.5) for c in [*cands, chosen]
            ):
                cands.append(pick)
        lr_probes = [run(last.steps, c, "lr") for c in sorted(set(cands))]
        best = min(lr_probes, key=_score, default=None)
        if best is not None and _score(best) < 0.95 * _score(last):
            reasons.append(
                f"learning rate {fmt(chosen)} → {fmt(best.lr)}: misfit {fmt(_score(last))} → "
                f"{fmt(_score(best))} at {last.steps} steps (candidates "
                f"{', '.join(fmt(r.lr) for r in lr_probes)})"
            )
            chosen = best.lr
            probes[-1] = dataclasses.replace(best, kind="budget")
            if len(probes) >= 2:
                probes[-1].field_change = change(probes[-2], probes[-1])
            use_chi = use_chi and _finite(best, chi=True)
        else:
            tried = ", ".join(fmt(r.lr / chosen) for r in lr_probes)
            reasons.append(
                f"learning rate kept at {fmt(chosen)}: ×{{{tried}}} did not lower the misfit at "
                f"{last.steps} steps by ≥ 5 %"
            )
    # ---- classification ------------------------------------------------------------------------
    last = probes[-1]
    drop, rate = _trend(probes[-2], last, use_chi) if len(probes) >= 2 else (-math.inf, None)
    overfit = bool(use_chi and last.chi is not None and last.chi < 0.9)
    ref = next((p for p in probes if settled(p)), None)
    if ref is None and at_floor(last):
        ref = last  # at the floor, but the reconstruction still moved at the probe cap
    anneal_scale = 1.0
    tau: float | None = base.discrepancy_tau
    if ref is not None:
        status = "at_noise_floor"
        how = (
            f"χ = {fmt(ref.chi)} ≤ 1"
            if ref.chi is not None and ref.chi <= 1.0
            else f"the reconstruction changed by {fmt(ref.field_change)} over the last doubling"
        )
        if ref.steps >= base_steps:
            rec = int(ref.steps)
            tau = min(1.0, float(ref.chi)) if ref.chi is not None else 1.0
            reasons.append(
                f"the misfit reaches the noise floor (χ ≤ {fmt(floor)}) and settles after "
                f"{ref.steps} steps ({how}): that validated schedule is the recommendation, with "
                f"a discrepancy stop at its misfit (τ = {fmt(tau)})"
                + ("; the largest probe went below χ = 0.9 (fitting noise)" if overfit else "")
            )
        else:
            rec = int(base_steps)
            reasons.append(
                f"the default budget suffices: the misfit reaches the noise floor and settles "
                f"after {ref.steps} of its {base_steps} steps ({how}); budget and stopping kept "
                "(a shorter schedule reaches the floor but the reconstruction keeps improving)"
            )
        if ref is last and not settled(ref):
            limit = "probe cap" if 2 * last.steps > probe_cap else f"extend = {extend} doublings"
            reasons.append(
                f"the reconstruction still changed by {fmt(ref.field_change)} over the last "
                f"doubling when the probes stopped ({limit}): a larger budget may still improve "
                "it (level='standard' / 'thorough' probe further)"
            )
    elif drop >= CONVERGING_DROP:
        status = "converging" if use_chi else "unknown"
        rec = int(min(max(2 * last.steps, base_steps), cap))
        anneal_scale = min(1.0, last.steps / rec)
        tau = 1.0 if use_chi else base.discrepancy_tau
        what = f"χ = {fmt(last.chi)}" if use_chi else "the data loss"
        reasons.append(
            f"still converging after {last.steps} steps ({what}, excess misfit −"
            f"{100 * drop:.0f} % over the last doubling; α = {fmt(rate)}): budget {rec} with the "
            "last probe's annealing length and discrepancy stopping"
            + (f" (probe cap {probe_cap} reached)" if 2 * last.steps > probe_cap else "")
        )
    else:
        status = "stalled" if use_chi else "unknown"
        rec = int(max(probes[-2].steps if len(probes) >= 2 else last.steps, base_steps))
        reasons.append(
            f"stalled at χ = {fmt(last.chi)} (excess misfit −{100 * max(drop, 0.0):.0f} % over "
            "the last doubling): more steps will not reach the noise floor — lower the "
            "regularization, raise the representation's capacity / bandwidth, or check the "
            "forward model"
            if use_chi
            else "noise level unknown and the data loss no longer falls: budget unchanged"
        )
    reach = ref.steps if ref is not None else None
    report = ConvergenceReport(
        status=status,
        sigma=s,
        sigma_source=src,
        probes=probes,
        lr_probes=lr_probes,
        lr=float(chosen),
        base_lr=float(base_lr),
        lr_range_pick=pick,
        recommended_steps=int(rec),
        base_steps=int(base_steps),
        anneal_scale=float(anneal_scale),
        tau=tau,
        reached_floor_at=reach,
        rate=rate,
        floor=floor,
        overfit=overfit,
        reasons=reasons,
        probe_steps=timer.steps,
        seconds=timer.seconds,
        fitted=fitted,
    )
    log.info("probe_convergence: %s", report.summary())
    return report


def tune_budget(
    problem: Any,
    report: ConvergenceReport | None = None,
    *,
    curriculum: Curriculum | None = None,
    tau: float | None = None,
    anneal_fraction: float | None = None,
    **probe_kw: Any,
) -> Curriculum:
    """A curriculum with the budget, learning rate, annealing and stopping the probes call for.

    Args:
        problem: the problem.
        report: a :class:`ConvergenceReport` (computed with ``probe_kw`` when ``None``).
        curriculum: base curriculum (default: the problem's).
        tau: Morozov factor (default ``report.tau``: the misfit of the validated probe, capped
            at 1, when the floor was reached; 1 when still converging; the base curriculum's
            otherwise; ``0`` disables).
        anneal_fraction: set every stage's annealing fraction (default: the base fraction ×
            ``report.anneal_scale``, i.e. the reference probe's absolute annealing length).
        **probe_kw: forwarded to :func:`probe_convergence`.

    Returns:
        The tuned :class:`~nefi.solve.Curriculum` (stage structure of the base curriculum).
        Discrepancy stopping needs ``measurement.noise_std``: with an estimated σ solve the
        problem returned by :func:`nefi.autotune.autotune_problem` (which stores it) or set it.

    Example::

        curriculum = tune_budget(problem)
        result = nefi.invert(problem, curriculum)
    """
    if report is None:
        report = probe_convergence(problem, curriculum=curriculum, **probe_kw)
    base = base_curriculum(problem, curriculum)
    unchanged = (
        report.recommended_steps == base.total_steps
        and abs(report.lr - report.base_lr) <= 1e-12 * max(1.0, report.base_lr)
        and report.anneal_scale == 1.0
        and anneal_fraction is None
    )
    if unchanged:  # the default budget suffices (or is the validated probe): keep it as it is
        t = report.tau if tau is None else tau
        base.discrepancy_tau = float(t) if t else None
        return base
    cur = scale_lr(_cosine(rescaled(base, report.recommended_steps)), report.lr / report.base_lr)
    for st in cur.stages:
        if anneal_fraction is not None:
            st.anneal_fraction = float(anneal_fraction)
        elif st.anneal:
            st.anneal_fraction = float(
                min(1.0, max(0.05, st.anneal_fraction * report.anneal_scale))
            )
    if tau is None:
        tau = report.tau
    cur.discrepancy_tau = float(tau) if tau else None
    return cur


__all__ = [
    "CONVERGING_DROP",
    "FIELD_TOL",
    "STATUSES",
    "ConvergenceReport",
    "ProbeRecord",
    "probe_convergence",
    "tune_budget",
]
