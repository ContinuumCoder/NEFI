"""One-call auto-tuning: noise → gauges → acquisition → budget / learning rate → regularization →
(held-out search), every decision logged and reported.

:func:`autotune_problem` never modifies the problem it is given: it returns a tuned copy (private
field, measurement carrying σ, gauge fixes, tuned loss weights), the tuned
:class:`~nefi.solve.Curriculum` and an :class:`AutotuneReport` (``report.to_markdown()``).

=========  =====================================================================================
level      what runs (cost in optimization steps, 32² smoke problems)
=========  =====================================================================================
quick      σ, gauges, light acquisition report (k = 4), doubling budget probes up to 4× the last
           initial probe, no learning-rate search, no regularization tuning (≈ 1–2 k steps)
standard   + full acquisition report (k = 8, match report), probes up to 8×, learning-rate search
           when the base rate misses the floor, Morozov regularization (≈ 3–5 k steps)
thorough   + held-out random / Sobol search (8 trials at 30 % of the tuned budget) around the
           tuned configuration, adopted only if it beats it by > 5 % (≈ 6–10 k steps)
=========  =====================================================================================
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ConfigError
from ..solve.curriculum import Curriculum
from ._common import (
    Decision,
    Timer,
    base_curriculum,
    effective_weight,
    fmt,
    jsonable,
    md_table,
    noise_float,
    peak_lr,
    resolve_sigma,
    with_sigma,
)
from .acquisition import AcquisitionReport, acquisition_report
from .convergence import ConvergenceReport, probe_convergence, tune_budget
from .gauge import GaugeReport, detect_gauges, repair_gauges
from .regularization import RegularizationResult, regularizer_names, tune_regularization
from .search import TuneReport, autotune

log = logging.getLogger("nefi")

LEVELS: dict[str, dict[str, Any]] = {
    "quick": {
        "steps": (50, 100, 200),
        "extend": 2,
        "lr_factors": (),
        "lr_test": False,
        "acquisition": {"k": 4, "n_probes": 8, "match": False},
        "regularization": None,
        "search": None,
    },
    "standard": {
        "steps": (50, 100, 200),
        "extend": 3,
        "lr_factors": (1.0 / 3.0, 3.0),
        "lr_test": True,
        "acquisition": {"k": 8, "n_probes": 16, "match": True},
        "regularization": {"budget_scale": 0.2},
        "search": None,
    },
    "thorough": {
        "steps": (50, 100, 200),
        "extend": 4,
        "lr_factors": (1.0 / 3.0, 3.0),
        "lr_test": True,
        "acquisition": {"k": 8, "n_probes": 32, "match": True},
        "regularization": {"budget_scale": 0.3},
        "search": {"trials": 8, "budget_scale": 0.3, "holdout": 0.1},
    },
}


@dataclass
class AutotuneReport:
    """Everything :func:`autotune_problem` measured and decided.

    Attributes:
        problem / level / seed: what was tuned, how thoroughly.
        sigma / sigma_source: the noise level every decision is measured against.
        decisions: the log of decisions (:class:`~nefi.autotune._common.Decision`).
        gauges / acquisition / convergence / regularization / search: the sub-reports (``None``
            when skipped).
        curriculum: the tuned curriculum.
        base: the configuration before tuning (steps, peak lr, regularizer weights).
        tuned: the configuration after tuning.
        comparison: default vs tuned metrics (:func:`autotune_instance` with ``compare=True``).
        notes: warnings and limitations that apply to this problem.
        probe_steps / seconds: total tuning cost.
        meta: free-form (instance name, scene, …).
    """

    problem: str
    level: str
    seed: int
    sigma: float | None
    sigma_source: str
    decisions: list[Decision]
    gauges: GaugeReport | None
    acquisition: AcquisitionReport | None
    convergence: ConvergenceReport | None
    regularization: RegularizationResult | None
    search: TuneReport | None
    curriculum: Curriculum
    base: dict[str, Any]
    tuned: dict[str, Any]
    comparison: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    probe_steps: int = 0
    seconds: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """One paragraph: what was changed and why."""
        changed = [d for d in self.decisions if d.before != d.after and d.step != "acquisition"]
        parts = [f"{d.what}: {fmt(d.before)} → {fmt(d.after)}" for d in changed]
        acq = f"; acquisition: {self.acquisition.verdict}" if self.acquisition else ""
        return (
            f"autotune[{self.level}] {self.problem}: "
            + ("; ".join(parts) if parts else "no change needed")
            + acq
            + f" ({self.probe_steps} probe steps, {self.seconds:.1f} s)"
        )

    def to_markdown(self) -> str:
        rows = [
            [d.step, d.what, fmt(d.before), fmt(d.after), d.why, d.cost_steps]
            for d in self.decisions
        ]
        lines = [
            f"# nefi autotune — {self.problem} ({self.level})",
            "",
            f"Noise σ = {fmt(self.sigma)} ({self.sigma_source}); tuning cost {self.probe_steps} "
            f"optimization steps, {self.seconds:.1f} s.",
            "",
            md_table(["step", "decision", "before", "after", "why", "probe steps"], rows),
        ]
        if self.comparison:
            lines += ["", "## Default vs tuned", "", _comparison_table(self.comparison)]
        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        for title, sub in (
            ("Gauges", self.gauges),
            ("Acquisition / identifiability", self.acquisition),
            ("Budget and learning rate", self.convergence),
            ("Regularization (discrepancy principle)", self.regularization),
            ("Held-out search", self.search),
        ):
            if sub is not None:
                lines += ["", f"## {title}", "", sub.to_markdown()]
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.to_markdown()

    def to_dict(self) -> dict[str, Any]:
        from ..config import to_dict

        return jsonable(
            {
                "problem": self.problem,
                "level": self.level,
                "seed": self.seed,
                "sigma": self.sigma,
                "sigma_source": self.sigma_source,
                "summary": self.summary(),
                "decisions": [d.to_dict() for d in self.decisions],
                "base": self.base,
                "tuned": self.tuned,
                "curriculum": to_dict(self.curriculum),
                "gauges": None if self.gauges is None else self.gauges.to_dict(),
                "acquisition": None if self.acquisition is None else self.acquisition.to_dict(),
                "convergence": None if self.convergence is None else self.convergence.to_dict(),
                "regularization": None
                if self.regularization is None
                else self.regularization.to_dict(),
                "search": None if self.search is None else self.search.to_dict(),
                "comparison": self.comparison,
                "notes": list(self.notes),
                "probe_steps": self.probe_steps,
                "seconds": self.seconds,
                "meta": {k: v for k, v in self.meta.items() if isinstance(v, str | int | float)},
            }
        )

    def save(self, directory: str | Path) -> dict[str, Path]:
        """Write ``autotune.json`` and ``autotune.md`` to ``directory``."""
        import json

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        paths = {"json": d / "autotune.json", "markdown": d / "autotune.md"}
        paths["json"].write_text(json.dumps(self.to_dict(), indent=2, default=str))
        paths["markdown"].write_text(self.to_markdown())
        return paths


def _comparison_table(comp: Mapping[str, Any]) -> str:
    runs = [k for k in ("default", "tuned") if k in comp]
    keys: list[str] = []
    for r in runs:
        for k in comp[r].get("metrics", {}):
            if k not in keys:
                keys.append(k)
    head = ["run", *keys, "steps", "solve s", "χ = RMSE/σ"]
    rows = []
    for r in runs:
        c = comp[r]
        rows.append(
            [r, *[fmt(c["metrics"].get(k)) for k in keys], c.get("steps"), c.get("seconds"),
             c.get("chi")]
        )  # fmt: skip
    extra = []
    if "tuning_seconds" in comp:
        extra.append(
            f"Tuning cost: {comp.get('probe_steps', '?')} probe steps, "
            f"{fmt(comp['tuning_seconds'])} s."
        )
    return "\n".join([md_table(head, rows), "", *extra])


def _config(problem, cur) -> dict[str, Any]:
    names = regularizer_names(problem, cur)
    return {
        "steps": int(cur.total_steps),
        "lr": float(peak_lr(cur)),
        "stages": [[s.name, list(s.shape) if s.shape else None, int(s.steps)] for s in cur.stages],
        "anneal_fraction": [float(s.anneal_fraction) for s in cur.stages],
        "discrepancy_tau": cur.discrepancy_tau,
        "weights": {k: effective_weight(problem.losses.weights, cur, k) for k in names},
        "postprocess": [type(p).__name__ for p in problem.postprocess],
    }


def _clone(problem) -> Any:
    return dataclasses.replace(problem, field=copy.deepcopy(problem.field), meta=dict(problem.meta))


def autotune_problem(
    problem: Any,
    *,
    level: str = "standard",
    sigma: float | None = None,
    curriculum: Curriculum | None = None,
    problem_factory: Callable[..., Any] | None = None,
    seed: int = 0,
    device: str = "cpu",
    gauges: bool = True,
    acquisition: bool = True,
    candidates: Mapping[str, Any] | None = None,
    mean_prior: Any = None,
    anchor: Any = "mean",
    mass: Mapping[str, float] | None = None,
    max_steps: int | None = None,
    options: Mapping[str, Any] | None = None,
) -> tuple[Any, Curriculum, AutotuneReport]:
    """Detect and fix the common failure modes of a per-measurement inversion, in one call.

    1. **noise** — σ from the measurement, else estimated (stored in the returned problem so the
       discrepancy principle can stop the solver);
    2. **gauges** — :func:`detect_gauges` / :func:`repair_gauges` (mean anchor, scale
       correction, sign convention, penalties);
    3. **acquisition** — :func:`acquisition_report` (reported: the physics cannot be tuned);
    4. **budget and learning rate** — :func:`probe_convergence` / :func:`tune_budget`;
    5. **regularization** (``standard``, ``thorough``) — :func:`tune_regularization` warm-started
       from the probe that reached the noise floor; the budget is re-probed when a weight moved by
       more than 3×;
    6. **held-out search** (``thorough``) — :func:`nefi.autotune.autotune` around the tuned
       configuration; adopted only when it lowers the held-out error by more than 5 %.

    Args:
        problem: the problem (not modified).
        level: ``"quick"``, ``"standard"`` or ``"thorough"`` (see the module table).
        sigma: noise level (default: measurement / estimate).
        curriculum: base curriculum (default: the problem's).
        problem_factory: ``(**hparams) -> InverseProblem`` for the held-out search (lets it vary
            factory-level settings such as ``n_octaves``); default: copies of the tuned problem
            (curriculum and loss-weight dimensions only).
        seed / device: forwarded to every probe.
        gauges / acquisition: run these steps.
        candidates: extra gauge directions (:func:`detect_gauges`).
        mean_prior / anchor / mass: gauge-fix options (:func:`repair_gauges`).
        max_steps: cap on the recommended budget.
        options: override entries of :data:`LEVELS` (e.g. ``{"extend": 1}``).

    Returns:
        ``(tuned_problem, tuned_curriculum, report)``.

    Example::

        problem, curriculum, report = nefi.autotune.autotune_problem(problem, level="quick")
        print(report.to_markdown())
        result = nefi.invert(problem, curriculum)
    """
    if level not in LEVELS:
        raise ConfigError(f"level must be one of {tuple(LEVELS)}, got {level!r}")
    cfg = {**LEVELS[level], **dict(options or {})}
    timer = Timer()
    decisions: list[Decision] = []
    notes: list[str] = []
    base_cur = base_curriculum(problem, curriculum)
    base = _config(problem, base_cur)
    work = _clone(problem)
    # ---- 1. noise --------------------------------------------------------------------------------
    s, src = resolve_sigma(work, sigma)
    if s is not None:
        work = with_sigma(work, s, src)
    if s is not None:
        why = f"{src}; stored in the measurement (discrepancy stopping)"
    else:
        why = "unknown: no discrepancy stopping, regularization by GradNorm shares"
    decisions.append(
        Decision("noise", "noise level σ", noise_float(problem.measurement.noise_std), s, why)
    )
    if s is not None and "estimated" in src:
        notes.append(
            "σ was estimated from the data (typically within ±20 %): the noise-floor band is "
            "widened to χ ≤ 1.2"
        )
    # ---- 2. gauges -------------------------------------------------------------------------------
    g_rep = None
    if gauges:
        t0 = time.perf_counter()
        g_rep = detect_gauges(work, curriculum=base_cur, candidates=candidates, seed=seed)
        if g_rep.unresolved:
            work = repair_gauges(work, g_rep, mean_prior=mean_prior, anchor=anchor, mass=mass)
            for r in g_rep.repairs:
                decisions.append(
                    Decision("gauges", "gauge fix", None, r, "direction invisible to the data",
                             seconds=time.perf_counter() - t0)
                )  # fmt: skip
        else:
            decisions.append(
                Decision("gauges", "gauge check", None, "none needed", g_rep.summary(),
                         seconds=time.perf_counter() - t0)
            )  # fmt: skip
    # ---- 3. acquisition --------------------------------------------------------------------------
    acq = None
    if acquisition:
        t0 = time.perf_counter()
        try:
            acq = acquisition_report(work, seed=seed, **cfg["acquisition"])
            decisions.append(
                Decision(
                    "acquisition",
                    "identifiability",
                    None,
                    acq.verdict,
                    "reported, not tunable: " + (acq.findings[0] if acq.findings else "—"),
                    seconds=time.perf_counter() - t0,
                )
            )
            if acq.flagged:
                notes.append(
                    f"acquisition: {acq.verdict} — auto-tuning adapts the prior and the budget to "
                    "what the data support; it cannot add information (see the recommendations)"
                )
        except Exception as e:  # noqa: BLE001 - a report must not abort the tuning
            log.warning("autotune: acquisition report failed (%s)", e)
            notes.append(f"acquisition report skipped: {type(e).__name__}: {e}")
    # ---- 4. budget and learning rate -------------------------------------------------------------
    conv = probe_convergence(
        work,
        curriculum=base_cur,
        steps=cfg["steps"],
        extend=cfg["extend"],
        lr_factors=cfg["lr_factors"],
        lr_test=cfg["lr_test"],
        max_steps=max_steps,
        seed=seed,
        device=device,
    )
    timer.add(steps=conv.probe_steps)
    cur = tune_budget(work, conv, curriculum=base_cur)
    decisions += _budget_decisions(base_cur, cur, conv)
    if conv.status == "stalled":
        notes.append("convergence: " + conv.reasons[-1])
    # ---- 5. regularization -----------------------------------------------------------------------
    reg = None
    if cfg["regularization"] and regularizer_names(work, cur):
        init = init_chi = None
        if conv.status == "at_noise_floor":
            init = conv.fitted_field()
            init_chi = next((p.chi for p in conv.probes if p.steps == conv.recommended_steps), None)
        reg = tune_regularization(
            work, curriculum=cur, init=init, init_chi=init_chi, seed=seed, device=device,
            sigma=s, **cfg["regularization"],
        )  # fmt: skip
        timer.add(steps=reg.probe_steps)
        moved = {k: f for k, f in reg.factors.items() if abs(math.log(f)) > 1e-9}
        for k in reg:
            decisions.append(
                Decision("regularization", f"weight of {k!r}", reg.before[k], reg.after[k],
                         f"{reg.status}: " + (reg.notes[-1] if reg.notes else ""),
                         cost_steps=reg.probe_steps, seconds=reg.seconds)
            )  # fmt: skip
        if moved:
            work, cur = reg.apply(work, cur)
            if max(abs(math.log(f)) for f in moved.values()) >= math.log(3.0):
                t = cur.total_steps
                conv2 = probe_convergence(
                    work,
                    curriculum=cur,
                    steps=(max(20, t // 4), max(40, t // 2), t),
                    extend=1,
                    lr=peak_lr(cur),
                    lr_factors=(),
                    lr_test=False,
                    max_steps=max_steps,
                    seed=seed,
                    device=device,
                )
                timer.add(steps=conv2.probe_steps)
                new_cur = tune_budget(work, conv2, curriculum=cur)
                decisions.append(
                    Decision("budget", "total steps (re-probed after the weight change)",
                             cur.total_steps, new_cur.total_steps, conv2.summary(),
                             cost_steps=conv2.probe_steps, seconds=conv2.seconds)
                )  # fmt: skip
                cur = new_cur
    elif cfg["regularization"]:
        notes.append("regularization: no active regularizer to tune")
    # ---- 6. held-out search ----------------------------------------------------------------------
    srch = None
    if cfg["search"]:
        factory = _search_factory(work, problem_factory, s, src, g_rep, reg)
        srch = autotune(factory, curriculum=cur, seed=seed, device=device, **cfg["search"])
        timer.add(steps=srch.probe_steps)
        imp = srch.improvement
        adopt = bool(srch.best) and math.isfinite(imp) and imp > 0.05
        decisions.append(
            Decision(
                "search",
                "held-out configuration",
                "tuned",
                ", ".join(f"{k}={fmt(v)}" for k, v in srch.best.items()) if adopt else "tuned",
                f"held-out MSE {fmt(srch.default_score)} → {fmt(srch.best_score)}"
                + ("" if adopt else " (kept: < 5 % better)"),
                cost_steps=srch.probe_steps,
                seconds=srch.seconds,
            )
        )
        if adopt:
            work, cur = srch.build(factory, cur)
    meta = dict(work.meta)
    meta["autotune"] = {"level": level, "summary": None}
    work = dataclasses.replace(work, curriculum=cur, meta=meta)
    report = AutotuneReport(
        problem=str(getattr(problem, "name", "problem")),
        level=level,
        seed=int(seed),
        sigma=s,
        sigma_source=src,
        decisions=decisions,
        gauges=g_rep,
        acquisition=acq,
        convergence=conv,
        regularization=reg,
        search=srch,
        curriculum=cur,
        base=base,
        tuned=_config(work, cur),
        notes=notes,
        probe_steps=timer.steps,
        seconds=timer.seconds,
    )
    meta["autotune"]["summary"] = report.summary()
    log.info("%s", report.summary())
    return work, cur, report


def _budget_decisions(base: Curriculum, cur: Curriculum, conv: ConvergenceReport) -> list[Decision]:
    out = [
        Decision("budget", "total steps", base.total_steps, cur.total_steps,
                 f"{conv.status}: " + (conv.reasons[-1] if conv.reasons else ""),
                 cost_steps=conv.probe_steps, seconds=conv.seconds),
        Decision("learning rate", "peak lr", peak_lr(base), peak_lr(cur),
                 next((r for r in conv.reasons if r.startswith("learning rate")),
                      "the base rate reached the target" if conv.status == "at_noise_floor"
                      else "unchanged")),
    ]  # fmt: skip
    af0 = [s.anneal_fraction for s in base.stages]
    af1 = [s.anneal_fraction for s in cur.stages]
    if any(abs(a - b) > 1e-9 for a, b in zip(af0, af1)):
        out.append(
            Decision("budget", "annealing fraction", af0, af1,
                     "keeps the reference probe's absolute annealing length (stops are checked "
                     "only after annealing)")
        )  # fmt: skip
    if cur.discrepancy_tau != base.discrepancy_tau:
        out.append(
            Decision("budget", "discrepancy_tau", base.discrepancy_tau, cur.discrepancy_tau,
                     "Morozov stop at RMSE ≤ τσ")
        )  # fmt: skip
    return out


def _search_factory(work, problem_factory, s, src, g_rep, reg) -> Callable[..., Any]:
    """Problems for the held-out search, carrying σ, the gauge fixes and the tuned weights."""
    if problem_factory is None:

        def clone() -> Any:  # no keyword arguments: only curriculum / weight dimensions
            return _clone(work)

        return clone

    def prepare(**hp: Any) -> Any:
        p = problem_factory(**hp)
        if s is not None:
            p = with_sigma(p, s, src)
        if g_rep is not None and g_rep.unresolved:
            p = repair_gauges(p, dataclasses.replace(g_rep, repairs=[]))
        if reg is not None and reg.factors:
            w = {k: p.losses.weights[k] * f for k, f in reg.factors.items() if k in p.losses.names}
            p = dataclasses.replace(p, losses=p.losses.with_weights(w), meta=dict(p.meta))
        return p

    try:  # expose the factory's keyword names (the default space checks for n_octaves)
        prepare.__signature__ = inspect.signature(problem_factory)  # type: ignore[attr-defined]
    except (TypeError, ValueError):
        pass
    return prepare


# ------------------------------------------------------------------------------------------
# instances
# ------------------------------------------------------------------------------------------
def instance_factory(instance: Any, measurement: Any) -> Callable[..., Any]:
    """``(**config_overrides) -> problem`` on a fixed measurement (fresh instance per call)."""
    from ..config import to_dict

    cls = type(instance)
    base = to_dict(instance.cfg) if getattr(instance, "Config", None) else dict(instance.cfg)
    names = list(base) if isinstance(base, dict) else []

    def factory(**hp: Any) -> Any:
        unknown = [k for k in hp if k not in names]
        if unknown:
            raise ConfigError(f"{cls.__name__} has no config fields {unknown}")
        return cls({**base, **hp}).build_problem(measurement)

    factory.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=base[k]) for k in names]
    )
    return factory


def autotune_instance(
    instance: Any,
    seed: int = 0,
    level: str = "standard",
    *,
    scene_class: str | None = None,
    curriculum: Curriculum | None = None,
    device: str = "cpu",
    compare: bool = False,
    **kw: Any,
) -> tuple[Any, Curriculum, AutotuneReport]:
    """Auto-tune a registered instance on one of its own measurements.

    Generates the measurement (``seed``, ``scene_class``), builds the instance's problem, runs
    :func:`autotune_problem` (the held-out search of ``level="thorough"`` may vary instance config
    fields such as ``n_octaves``) and, with ``compare=True``, solves the default and the tuned
    configuration and stores both metrics (and the raw, non-mean-subtracted PSNR) in
    ``report.comparison``.

    Args:
        instance: an :class:`~nefi.instances.Instance`.
        seed: data and optimization seed.
        level: tuning level.
        scene_class: scene class (default: the instance's).
        curriculum: base curriculum (default: the instance's).
        device: solver device.
        compare: run default and tuned configurations and report both.
        **kw: forwarded to :func:`autotune_problem`.

    Returns:
        ``(tuned_problem, tuned_curriculum, report)``.

    Example::

        from nefi.instances.toy1d import Toy1D
        problem, cur, report = autotune_instance(Toy1D(n=64), level="quick", compare=True)
        print(report.comparison["tuned"]["metrics"])
    """
    from ..utils.seed import seed_everything

    gt, meas = instance.make_measurement(seed=seed, scene_class=scene_class)
    seed_everything(seed)
    problem = instance.build_problem(meas)
    factory = instance_factory(instance, meas)
    tuned, cur, report = autotune_problem(
        problem,
        level=level,
        curriculum=curriculum,
        problem_factory=factory,
        seed=seed,
        device=device,
        **kw,
    )
    report.meta.update(
        instance=str(getattr(instance, "name", type(instance).__name__)),
        scene=str(scene_class or instance.default_scene_class() or "-"),
    )
    if compare:
        base_cur = base_curriculum(problem, curriculum)
        report.comparison = compare_runs(instance, gt, problem, base_cur, tuned, cur, seed, device)
        report.comparison.update(probe_steps=report.probe_steps, tuning_seconds=report.seconds)
    return tuned, cur, report


def compare_runs(
    instance: Any,
    gt: Mapping[str, Any],
    problem: Any,
    curriculum: Curriculum,
    tuned: Any,
    tuned_curriculum: Curriculum,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Solve the default and the tuned configuration; metrics, raw PSNR, steps, time, χ."""
    from ..metrics.basic import psnr
    from ..solve.solver import Solver
    from ._common import residual

    out: dict[str, Any] = {}
    for key, prob, cur in (("default", problem, curriculum), ("tuned", tuned, tuned_curriculum)):
        p = _clone(prob)
        t0 = time.perf_counter()
        res = Solver(p, cur, device=device, seed=seed).run()
        secs = time.perf_counter() - t0
        metrics = {k: float(v) for k, v in instance.evaluate(res, gt).items()}
        name = p.field.primary
        if name in gt and tuple(res.fields[name].shape) == tuple(gt[name].shape):
            metrics["raw_psnr"] = float(psnr(res.fields[name], gt[name]))
        chi = None
        s = noise_float(tuned.measurement.noise_std)
        if s:
            try:
                chi = residual(res.pred, prob.measurement)[0] / s
            except Exception:  # noqa: BLE001 - informative only
                chi = None
        out[key] = {
            "metrics": metrics,
            "steps": len(res.history.get("total", [])),
            "seconds": secs,
            "chi": chi,
            "stops": [str(r.get("stop")) for r in res.stage_results],
        }
    return out


# ------------------------------------------------------------------------------------------
# benchmark method
# ------------------------------------------------------------------------------------------
def autotuned_method(level: str = "quick", name: str | None = None, **autotune_kw: Any) -> Any:
    """A :class:`~nefi.bench.protocol.Method` that auto-tunes every measurement before solving.

    The tuning runs inside the method's runner, so the benchmark's ``time_s`` column includes it
    (``result.extra["autotune_probe_steps"]`` holds the probe cost). The tuner decides the budget:
    the benchmark's ``budget_scale`` only scales the *base* curriculum the tuner starts from.

    Example::

        from nefi.bench import run_benchmark
        from nefi.bench.protocol import default_method
        res = run_benchmark(inst, [default_method(), autotuned_method("quick")], n_samples=2)
    """
    from ..bench.protocol import Method, PreparedRun
    from ..solve.solver import Solver

    def build(instance: Any, measurement: Any) -> Any:
        problem = instance.build_problem(measurement)

        def runner(prob: Any, curriculum: Curriculum, **solver_kw: Any):
            seed = solver_kw.get("seed") or 0
            dev = solver_kw.get("device") or "auto"
            tuned, cur, rep = autotune_problem(
                prob, level=level, curriculum=curriculum, seed=seed, device=dev, **autotune_kw
            )
            res = Solver(tuned, cur, **solver_kw).run()
            res.extra["autotune_probe_steps"] = rep.probe_steps
            res.extra["autotune_seconds"] = rep.seconds
            res.extra["autotune"] = rep.summary()
            return res

        return PreparedRun(problem, None, {}, runner)

    return Method(name or f"autotuned-{level}", build, description=f"auto-tuned ({level})")


__all__ = [
    "LEVELS",
    "AutotuneReport",
    "autotune_instance",
    "autotune_problem",
    "autotuned_method",
    "compare_runs",
    "instance_factory",
]
