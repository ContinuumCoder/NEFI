"""Benchmark protocol: paired, seeded, inverse-crime-safe comparisons with confidence intervals.

The protocol of both papers, generalized to any registered :class:`~nefi.instances.Instance`:

* **samples × seeds** — every method sees the *same* measurements (data seed ``i`` for sample
  ``i``, optionally per scene class) and is run with every optimization seed; statistics are over
  all ``samples × seeds`` runs (NeTMY Tab. 1: "mean and 95% CI across 3 seeds"; NeFTY Tab. 1:
  "mean ± 95% CI" over the benchmark);
* **cross-fidelity by construction** — the measurement is simulated by the instance's
  :class:`~nefi.bench.base.DataGenerator` and the inversion must use a *different* operator
  (NeTMY: F3 data, F1/F2 inversion; NeFTY: explicit PhiFlow data, implicit-Euler inversion). The
  guard compares ``fidelity_tag``\\ s and refuses matched operators unless
  ``allow_inverse_crime=True`` — the "matched-operator" regime of NeTMY Tab. 2, which must be
  requested explicitly and is labelled as such in every report;
* **runtime columns** — wall-clock, steps, peak GPU memory per run (NeTMY App. E.9, NeFTY Tab. 9);
* **throughput for clusters** — ``batched=True`` solves all (sample, seed) runs of a batchable
  method in one optimization loop (:func:`nefi.solve.batch_invert`; per-run times are the batch
  wall-clock divided by the batch size); ``shard=(i, n)`` runs a deterministic subset of the
  (class, sample, seed) units and :meth:`BenchmarkResult.merge` concatenates shard files back into
  the table of the unsharded run (SLURM array jobs, ``nefi bench --shard i/n`` +
  ``nefi bench-merge``).

Everything public from :mod:`~nefi.bench.report`, :mod:`~nefi.bench.ablation` and
:mod:`~nefi.bench.sweep` is re-exported here (and from :mod:`nefi.bench`).
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..config import to_dict
from ..errors import ConfigError, NefiError
from ..solve.batched import batch_invert, batchable_reason
from ..solve.curriculum import Curriculum
from ..solve.result import Result
from ..solve.solver import Solver
from ..utils.device import peak_memory_mb, reset_peak_memory, resolve_device, synchronize
from ..utils.seed import seed_everything
from .report import (
    format_mean_ci,
    format_value,
    markdown_table,
    mean_ci,
    metric_direction,
    rows_to_csv,
    to_jsonable,
    write_json,
    write_text,
)

log = logging.getLogger("nefi")

DEFAULT_METHOD_ALIASES = ("neural", "nefi", "default", "ours", "netmy", "nefty")


# ------------------------------------------------------------------------------------------
# inverse-crime guard
# ------------------------------------------------------------------------------------------
class InverseCrimeError(NefiError, AssertionError):
    """Raised when data generation and inversion use the same forward model (NeTMY §3.1)."""


def operator_fidelity_tag(operator: Any) -> str:
    """Fidelity tag of an inversion operator.

    ``operator.fidelity_tag`` if set; wrappers exposing ``.inner`` (e.g.
    :class:`~nefi.operators.Nuisance`) are unwrapped; otherwise the class name.
    """
    tag = getattr(operator, "fidelity_tag", None)
    if tag:
        return str(tag)
    inner = getattr(operator, "inner", None)
    if isinstance(inner, torch.nn.Module) and inner is not operator:
        return operator_fidelity_tag(inner)
    return type(operator).__name__


def check_inverse_crime(
    data_generator: Any, operator: Any, allow: bool = False, label: str = ""
) -> str:
    """Assert that the data were simulated with a different model than the inversion operator.

    Args:
        data_generator: a :class:`~nefi.bench.base.DataGenerator` (or its ``fidelity_tag``).
        operator: the inversion operator of the problem.
        allow: permit the matched-operator regime (logs a loud warning instead of raising).
        label: method name used in messages.

    Returns:
        ``"cross-fidelity"`` or ``"matched"``.

    Raises:
        InverseCrimeError: if the tags (or the operator objects) coincide and ``allow`` is False.
    """
    if isinstance(data_generator, str):
        gen_tag, same = data_generator, False
    else:
        gen_tag = str(getattr(data_generator, "fidelity_tag", type(data_generator).__name__))
        same = getattr(data_generator, "operator", None) is operator
    inv_tag = operator_fidelity_tag(operator)
    if gen_tag != inv_tag and not same:
        return "cross-fidelity"
    msg = (
        f"inverse crime{f' ({label})' if label else ''}: the data generator (fidelity_tag="
        f"{gen_tag!r}) and the inversion operator ({inv_tag!r}) are the same forward model. "
        "Benchmarks must simulate data with an independent operator/discretization (NeTMY F3 vs "
        "F2, NeFTY explicit PhiFlow vs implicit Euler). Pass allow_inverse_crime=True "
        "(CLI: --allow-inverse-crime) to run the matched-operator regime deliberately."
    )
    if not allow:
        raise InverseCrimeError(msg)
    bar = "!" * 88
    log.warning(
        "\n%s\n MATCHED-OPERATOR REGIME (inverse crime explicitly allowed)%s:\n data %r == "
        "inversion %r. Results are optimistic and must be reported as matched-operator\n (NeTMY "
        "Tab. 2), never as the primary cross-fidelity evidence.\n%s",
        bar,
        f" for {label}" if label else "",
        gen_tag,
        inv_tag,
        bar,
    )
    return "matched"


# ------------------------------------------------------------------------------------------
# methods
# ------------------------------------------------------------------------------------------
Runner = Callable[..., Result]


def solve_problem(problem: Any, curriculum: Curriculum | None = None, **solver_kw: Any) -> Result:
    """Solve a problem with the solver it asks for.

    Problems built by :func:`nefi.baselines.baseline_problem` may request a non-gradient solver
    (``problem.meta["solver"]``, e.g. ADMM or a closed-form reconstruction) or extra callbacks
    (``problem.meta["callbacks"]``, e.g. Gaussian-splat densification); those are dispatched
    through :func:`nefi.baselines.solve`. Everything else runs the standard
    :class:`~nefi.solve.Solver`.
    """
    meta = getattr(problem, "meta", None) or {}
    if meta.get("solver") not in (None, "gradient", "solver") or meta.get("callbacks"):
        try:
            from ..baselines import solve as dispatch
        except ImportError:  # pragma: no cover - baselines package unavailable
            dispatch = None
        if dispatch is not None:
            return dispatch(problem, curriculum, **solver_kw)
    return Solver(problem, curriculum, **solver_kw).run()


@dataclass
class PreparedRun:
    """A method instantiated on one measurement: problem + curriculum + solver settings."""

    problem: Any
    curriculum: Curriculum | None = None
    solver_kwargs: dict[str, Any] = field(default_factory=dict)
    runner: Runner | None = None

    def resolved_curriculum(self) -> Curriculum:
        cur = self.curriculum or getattr(self.problem, "curriculum", None)
        return cur if cur is not None else Curriculum.multiscale(self.problem.domain.shape)

    def run(
        self,
        curriculum: Curriculum | None = None,
        *,
        device: str | torch.device = "auto",
        seed: int | None = 0,
        callbacks: Sequence[Any] = (),
    ) -> Result:
        """Solve (custom ``runner`` if set, else :func:`solve_problem`)."""
        cur = curriculum if curriculum is not None else self.resolved_curriculum()
        kw = {"device": device, "seed": seed, "callbacks": list(callbacks), **self.solver_kwargs}
        if self.runner is not None:
            return self.runner(self.problem, cur, **kw)
        return solve_problem(self.problem, cur, **kw)


def _normalize_prepared(out: Any) -> PreparedRun:
    if isinstance(out, PreparedRun):
        return out
    if isinstance(out, Mapping):
        return PreparedRun(
            out["problem"],
            out.get("curriculum"),
            dict(out.get("solver_kwargs") or {}),
            out.get("runner"),
        )
    if isinstance(out, tuple | list):
        if len(out) == 2:
            return PreparedRun(out[0], out[1])
        if len(out) == 3:
            kw = dict(out[2] or {})
            runner = kw.pop("runner", None)
            return PreparedRun(out[0], out[1], kw, runner)
        raise ConfigError(f"a method must return (problem, curriculum[, solver_kwargs]), got {out}")
    if hasattr(out, "operator") and hasattr(out, "field"):
        return PreparedRun(out)
    raise ConfigError(f"cannot interpret method output of type {type(out).__name__}")


@dataclass
class Method:
    """A named way of solving an instance's measurement.

    Args:
        name: label in tables.
        build: ``None`` → the instance's own method (``build_problem`` + its curriculum); a string
            → a key of ``instance.baselines()``; or a callable ``(instance, measurement) ->
            (problem, curriculum | None[, solver_kwargs])`` (a bare problem, a dict with those
            keys or a :class:`PreparedRun` are accepted too). ``solver_kwargs`` may contain a
            ``"runner"`` callable ``(problem, curriculum, **kw) -> Result`` for non-gradient
            solvers.
        solver_kwargs: extra keyword arguments for :class:`~nefi.solve.Solver` (e.g. ``dtype``,
            ``compile``).
        description: free text for reports.
    """

    name: str
    build: Callable[[Any, Any], Any] | str | None = None
    solver_kwargs: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def prepare(self, instance: Any, measurement: Any) -> PreparedRun:
        """Instantiate the method on ``measurement``."""
        if self.build is None:
            problem = instance.build_problem(measurement)
            cur = getattr(problem, "curriculum", None) or instance.default_curriculum()
            prep = PreparedRun(problem, cur)
        elif isinstance(self.build, str):
            baselines = instance.baselines()
            if self.build not in baselines:
                raise ConfigError(
                    f"instance {getattr(instance, 'name', instance)!r} has no baseline "
                    f"{self.build!r}; available: {sorted(baselines)}"
                )
            prep = _normalize_prepared(baselines[self.build](measurement))
        else:
            prep = _normalize_prepared(self.build(instance, measurement))
        if self.solver_kwargs:
            kw = {**prep.solver_kwargs, **self.solver_kwargs}
            runner = kw.pop("runner", prep.runner)
            prep = PreparedRun(prep.problem, prep.curriculum, kw, runner)
        return prep


def default_method(name: str = "neural") -> Method:
    """The instance's own (neural-field) method."""
    return Method(name, None, description="instance default (neural field)")


def baseline_method(key: str, name: str | None = None) -> Method:
    """A baseline from ``instance.baselines()``."""
    return Method(name or key, key, description=f"baseline {key!r}")


def resolve_methods(instance: Any, methods: Any = None) -> list[Method]:
    """Methods from names / :class:`Method` objects / a comma-separated string.

    ``None`` → the instance default plus every baseline it declares. Names ``neural``, ``nefi``,
    ``default`` and ``ours`` mean the instance default; other names must be baseline keys.
    """
    baselines = sorted(instance.baselines())
    if methods is None:
        return [default_method()] + [baseline_method(b) for b in baselines]
    if isinstance(methods, str):
        methods = [m.strip() for m in methods.split(",") if m.strip()]
    if isinstance(methods, Method):
        methods = [methods]
    out: list[Method] = []
    for m in methods:
        if isinstance(m, Method):
            out.append(m)
        elif isinstance(m, str) and m.lower() in DEFAULT_METHOD_ALIASES:
            out.append(default_method(m))
        elif isinstance(m, str) and m in baselines:
            out.append(baseline_method(m))
        elif isinstance(m, str):
            raise ConfigError(
                f"unknown method {m!r} for instance {getattr(instance, 'name', '?')!r}; "
                f"available: {['neural', *baselines]}"
            )
        else:
            raise ConfigError(f"cannot interpret method {m!r}")
    names = [m.name for m in out]
    if len(set(names)) != len(names):
        raise ConfigError(f"duplicate method names: {names}")
    return out


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
def scale_budget(
    curriculum: Curriculum,
    scale: float = 1.0,
    max_total_steps: int | None = None,
    min_stage_steps: int = 1,
) -> Curriculum:
    """Copy of a curriculum with every stage's steps scaled (and the total optionally capped).

    Used for ``budget_scale`` in benchmarks and for CLI ``--smoke`` runs. Stage count,
    resolutions, learning rates and annealing fractions are unchanged.
    """
    cur = curriculum.scaled(scale) if scale != 1.0 else copy.deepcopy(curriculum)
    if max_total_steps and cur.total_steps > max_total_steps:
        cur = cur.scaled(max_total_steps / cur.total_steps)
    for s in cur.stages:
        s.steps = max(int(min_stage_steps), int(s.steps))
    return cur


def resolve_classes(instance: Any, classes: Any = None) -> list[str | None]:
    """Scene classes to benchmark: ``None`` → the instance default, ``"all"`` → every class."""
    if classes is None:
        return [instance.default_scene_class()]
    scenes = instance.scene_generator()
    if classes == "all":
        return list(scenes.classes)
    if isinstance(classes, str):
        classes = [c.strip() for c in classes.split(",") if c.strip()]
    return [scenes.check_class(c) for c in classes]


def resolve_metric_fns(metrics: Any) -> dict[str, Callable]:
    """``{name: fn}`` from a mapping or a list of registered metric names."""
    from ..registry import get

    if isinstance(metrics, Mapping):
        return dict(metrics)
    if isinstance(metrics, str):
        metrics = [m.strip() for m in metrics.split(",") if m.strip()]
    return {m: get("metric", m) for m in metrics}


def evaluate_metrics(
    instance: Any, result: Result, gt: Mapping[str, torch.Tensor], metrics: Any = None
) -> dict[str, float]:
    """Instance metrics (``instance.evaluate``) or an explicit metric set on the primary field."""
    if metrics is None:
        return {k: float(v) for k, v in instance.evaluate(result, gt).items()}
    from ..metrics.basic import evaluate

    name = next(iter(result.fields))
    return evaluate(result.fields[name], gt[name], resolve_metric_fns(metrics))


def _final_data_loss(res: Result) -> float | None:
    """Last aggregate data-fidelity value (``history["data_loss"]``; older results: ``"data"``)."""
    v = res.final("data_loss")
    return res.final("data") if v is None else v


def _warmup(prep: PreparedRun, device: torch.device) -> None:
    """One untimed optimization step (lazy imports, CUDA context, FFT plans, kernel caches)."""
    try:
        cur = copy.deepcopy(prep.resolved_curriculum())
        stage = cur.stages[0]
        stage.steps = 1
        cur.stages, cur.restarts = [stage], 1
        Solver(prep.problem, cur, device=device, seed=0, **prep.solver_kwargs).run()
    except Exception as e:  # warm-up is best effort
        log.debug("benchmark warm-up failed: %s", e)


def scene_columns(measurement: Any) -> dict[str, Any]:
    """Scalar scene metadata of a measurement as benchmark columns.

    Scalars in ``measurement.meta`` become ``meta/<key>`` and scalars in ``meta["scene"]`` become
    ``scene/<key>``; list/tuple values contribute their length as ``<prefix>/n_<key>`` (e.g. a
    ``defects`` list gives ``scene/n_defects``, NeFTY Tab. 6 strata). Use them with
    ``BenchmarkResult.table(by="scene/n_defects")``.
    """
    meta = getattr(measurement, "meta", None) or {}
    out: dict[str, Any] = {}

    def scalars(d: Mapping[str, Any], prefix: str) -> None:
        for k, v in d.items():
            if k.startswith("_"):
                continue
            if isinstance(v, bool | int | float | str):
                out[f"{prefix}/{k}"] = v
            elif isinstance(v, list | tuple):  # e.g. a list of defects -> scene/n_defects
                out[f"{prefix}/n_{k}"] = len(v)

    scalars(meta, "meta")
    scene = meta.get("scene")
    if isinstance(scene, Mapping):
        scalars(scene, "scene")
    return out


def _class_label(instance: Any, cls: str | None) -> str:
    if cls is not None:
        return cls
    try:
        return instance.scene_generator().check_class(None)
    except Exception:  # pragma: no cover - instances without scene classes
        return "default"


# ------------------------------------------------------------------------------------------
# results
# ------------------------------------------------------------------------------------------
_RUNTIME = ("time_s", "peak_mem_mb", "steps")
_RUNTIME_LABEL = {"time_s": "time (s)", "peak_mem_mb": "peak mem (MB)", "steps": "steps"}


@dataclass
class BenchmarkResult:
    """Per-run rows of a benchmark plus aggregation and report writers.

    Attributes:
        instance: instance name.
        rows: one dict per (class, sample, method, seed) run with metrics, ``time_s``,
            ``peak_mem_mb``, ``steps``, ``ms_per_step``, ``stop``, ``final_data_loss``,
            ``n_params`` and ``error`` (``None`` if the run succeeded).
        methods: method names in table order.
        metric_names: metric columns in table order.
        config: instance configuration used to generate the data.
        regime: ``method -> "cross-fidelity" | "matched"``.
        fidelity: ``{"data": tag, <method>: inversion tag}``.
        meta: protocol settings (samples, seeds, classes, budget, device, sweep/ablation info).
    """

    instance: str
    rows: list[dict[str, Any]]
    methods: list[str]
    metric_names: list[str]
    config: dict[str, Any] = field(default_factory=dict)
    regime: dict[str, str] = field(default_factory=dict)
    fidelity: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- aggregation -------------------------------------------------------------------
    def ok_rows(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if not r.get("error")]

    def direction(self, metric: str) -> bool | None:
        """``True`` if higher is better for ``metric`` (``None`` if unknown)."""
        d = self.meta.get("directions", {})
        return d[metric] if metric in d else metric_direction(metric)

    def table(
        self,
        ci: float = 0.95,
        by_class: bool = False,
        metrics: Sequence[str] | None = None,
        by: str | None = None,
    ) -> list[dict[str, Any]]:
        """Aggregated rows: mean and t-distribution CI half-width over samples × seeds.

        Each output row has ``method`` (and the grouping column when ``by_class`` / ``by``),
        ``n`` (successful runs), ``failed`` and, per metric and runtime column ``m``, ``m`` (mean)
        and ``m_ci`` (half-width, ``nan`` for n < 2). ``by`` names any row column, e.g.
        ``"scene/n_defects"`` for per-defect-count strata (NeFTY Tab. 6); ``by_class`` is
        ``by="class"``.
        """
        metrics = list(metrics or self.metric_names)
        key = "class" if by_class else by
        groups = sorted({r.get(key) for r in self.rows}, key=str) if key else [None]
        out = []
        for m in self.methods:
            for c in groups:
                sel = [
                    r for r in self.rows if r["method"] == m and (key is None or r.get(key) == c)
                ]
                if not sel:
                    continue
                ok = [r for r in sel if not r.get("error")]
                row: dict[str, Any] = {"method": m}
                if key:
                    row[key] = c
                row["n"] = len(ok)
                row["failed"] = len(sel) - len(ok)
                for k in [*metrics, *_RUNTIME]:
                    mean, hw, _ = mean_ci((r.get(k) for r in ok), ci)
                    row[k], row[f"{k}_ci"] = mean, hw
                out.append(row)
        return out

    def best(self, metric: str) -> str | None:
        """Method with the best mean ``metric`` (``None`` if the direction is unknown)."""
        d = self.direction(metric)
        rows = [r for r in self.table(metrics=[metric]) if math.isfinite(r[metric])]
        if d is None or not rows:
            return None
        pick = max if d else min
        return pick(rows, key=lambda r: r[metric])["method"]

    # ---- reports -----------------------------------------------------------------------
    def header(self, ci: float = 0.95) -> str:
        m = self.meta
        regimes = (set(self.regime.values()) - {"unknown"}) or {"cross-fidelity"}
        if regimes == {"cross-fidelity"}:
            reg = "cross-fidelity"
        else:
            matched = sorted(k for k, v in self.regime.items() if v == "matched")
            reg = f"⚠ MATCHED-OPERATOR (inverse crime allowed) for {', '.join(matched)}"
        inv = sorted({v for k, v in self.fidelity.items() if k != "data" and v != "?"})

        def count(n: Any, word: str) -> str:
            return f"{n} {word}{'' if n == 1 else ('es' if word.endswith('s') else 's')}"

        return (
            f"**{self.instance}** — {reg} (data: `{self.fidelity.get('data', '?')}`; inversion: "
            f"{', '.join(f'`{t}`' for t in inv) or '?'}) · "
            f"{count(m.get('n_samples', '?'), 'sample')}"
            f" × {count(len(m.get('seeds', [])), 'seed')} × "
            f"{count(len(m.get('classes', [])) or 1, 'class')} · mean ± {int(round(ci * 100))}% CI "
            "(Student t over samples × seeds)"
        )

    def to_markdown(
        self,
        ci: float = 0.95,
        by_class: bool = False,
        precision: int = 4,
        bold_best: bool = True,
        runtime: bool = True,
        header: bool = True,
        by: str | None = None,
    ) -> str:
        """Markdown table ``method | metric ↑/↓ … | time | peak mem | steps`` with best in bold.

        ``by`` groups rows by any column (e.g. ``"scene/n_defects"``); ``by_class`` means
        ``by="class"``.
        """
        key = "class" if by_class else by
        tab = self.table(ci, by=key)
        cols = list(self.metric_names) + (list(_RUNTIME) if runtime else [])
        heads = ["Method"] + ([key.split("/")[-1].replace("_", " ").title()] if key else []) + ["n"]
        for c in cols:
            if c in _RUNTIME:
                heads.append(_RUNTIME_LABEL[c])
            else:
                d = self.direction(c)
                heads.append(f"{c} {'↑' if d else '↓' if d is False else ''}".strip())
        best: dict[tuple[Any, str], float] = {}
        if bold_best:
            for c in self.metric_names:
                d = self.direction(c)
                if d is None:
                    continue
                groups: dict[Any, list[float]] = {}
                for r in tab:
                    if math.isfinite(r[c]):
                        groups.setdefault(r.get(key) if key else None, []).append(r[c])
                for g, vals in groups.items():
                    if len(vals) > 1:
                        best[(g, c)] = max(vals) if d else min(vals)
        body = []
        for r in tab:
            line = [r["method"]] + ([str(r[key])] if key else [])
            line.append(f"{r['n']}" + (f" ({r['failed']} failed)" if r["failed"] else ""))
            for c in cols:
                if c in ("steps", "peak_mem_mb"):  # budget / memory: the mean is enough
                    cell = format_value(r[c], precision)
                else:
                    cell = format_mean_ci(r[c], r[f"{c}_ci"], precision)
                if best.get((r.get(key) if key else None, c)) == r[c] and cell != "—":
                    cell = f"**{cell}**"
                line.append(cell)
            body.append(line)
        align = ["l"] * (2 if key else 1) + ["r"] * (len(heads) - (2 if key else 1))
        text = markdown_table(heads, body, align)
        return (self.header(ci) + "\n\n" + text if header else text) + "\n"

    def efficiency_table(self, precision: int = 3) -> str:
        """Training-level efficiency (NeFTY Tab. 9): params, steps, ms/step, wall-clock, memory."""
        heads = ["Method", "params", "steps", "ms / step", "wall-clock (s)", "peak mem (MB)"]
        body = []
        for m in self.methods:
            ok = [r for r in self.ok_rows() if r["method"] == m]
            if not ok:
                continue
            params = [r.get("n_params") for r in ok if r.get("n_params") is not None]
            body.append(
                [
                    m,
                    format_value(params[0], precision) if params else "—",
                    format_value(mean_ci(r.get("steps") for r in ok)[0], precision),
                    format_value(mean_ci(r.get("ms_per_step") for r in ok)[0], precision),
                    format_value(mean_ci(r.get("time_s") for r in ok)[0], precision),
                    format_value(mean_ci(r.get("peak_mem_mb") for r in ok)[0], precision),
                ]
            )
        return markdown_table(heads, body) + "\n"

    def to_csv(self, path: str | Path | None = None) -> str:
        """Raw per-run rows as CSV (written to ``path`` when given)."""
        text = rows_to_csv(self.rows)
        if path is not None:
            write_text(path, text)
        return text

    def summary_csv(self, ci: float = 0.95, by_class: bool = False, by: str | None = None) -> str:
        """Aggregated table as CSV (``by`` groups by any row column, e.g. ``"scene/n_defects"``)."""
        return rows_to_csv(self.table(ci, by_class, by=by))

    def strata(self, prefix: str = "scene/") -> list[str]:
        """Row columns available for stratification (``scene/*`` by default)."""
        cols: set[str] = set()
        for r in self.rows:
            cols.update(k for k in r if k.startswith(prefix))
        return sorted(cols)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "instance": self.instance,
                "methods": self.methods,
                "metric_names": self.metric_names,
                "config": self.config,
                "regime": self.regime,
                "fidelity": self.fidelity,
                "meta": self.meta,
                "rows": self.rows,
            }
        )

    def to_json(self, path: str | Path | None = None) -> str:
        """Everything (settings + raw rows) as JSON (written to ``path`` when given)."""
        text = json.dumps(self.to_dict(), indent=2) + "\n"
        if path is not None:
            write_text(path, text)
        return text

    def save(self, directory: str | Path, ci: float = 0.95) -> dict[str, Path]:
        """Write ``summary.md``, ``summary.csv``, ``rows.csv`` and ``benchmark.json``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        md = self.to_markdown(ci)
        if len({r["class"] for r in self.rows}) > 1:
            md += "\nPer class:\n\n" + self.to_markdown(ci, by_class=True, header=False)
        md += "\nEfficiency:\n\n" + self.efficiency_table()
        paths = {
            "markdown": write_text(d / "summary.md", md),
            "summary_csv": write_text(d / "summary.csv", self.summary_csv(ci)),
            "rows_csv": write_text(d / "rows.csv", self.to_csv()),
            "json": write_json(d / "benchmark.json", self.to_dict()),
        }
        return paths

    @staticmethod
    def load(path: str | Path) -> BenchmarkResult:
        """Load from a ``benchmark.json`` file or a directory written by :meth:`save`."""
        p = Path(path)
        if p.is_dir():
            p = p / "benchmark.json"
        d = json.loads(p.read_text())
        rows = [
            {k: (float("nan") if v is None and k not in ("error",) else v) for k, v in r.items()}
            for r in d["rows"]
        ]
        return BenchmarkResult(
            d["instance"],
            rows,
            d["methods"],
            d["metric_names"],
            d.get("config", {}),
            d.get("regime", {}),
            d.get("fidelity", {}),
            d.get("meta", {}),
        )

    # ---- shards ---------------------------------------------------------------------------
    @property
    def shard(self) -> tuple[int, int] | None:
        """``(i, n)`` if this result is shard ``i`` of ``n`` (see ``run_benchmark(shard=...)``)."""
        s = self.meta.get("shard")
        return None if not s else (int(s[0]), int(s[1]))

    def shard_filename(self) -> str:
        i, n = self.shard or (0, 1)
        return f"shard-{i:03d}-of-{n:03d}.json"

    def save_shard(self, directory: str | Path) -> Path:
        """Write this shard as ``<directory>/shard-III-of-NNN.json`` (merge with :meth:`merge`)."""
        return write_json(Path(directory) / self.shard_filename(), self.to_dict())

    @staticmethod
    def shard_files(directory: str | Path) -> list[Path]:
        """Shard files (``shard-*-of-*.json``) in ``directory``, sorted."""
        return sorted(Path(directory).glob("shard-*-of-*.json"))

    @staticmethod
    def merge(paths: str | Path | Sequence[str | Path], strict: bool = True) -> BenchmarkResult:
        """Concatenate benchmark shards into one result (the table of the unsharded run).

        Args:
            paths: shard files, directories containing ``shard-*-of-*.json`` files, or a mix.
            strict: raise if a shard is missing or duplicated (otherwise log a warning).

        The shards must come from the same protocol (instance, methods, configuration, samples,
        seeds, classes, budget); rows are put back into the canonical order of
        :func:`run_benchmark` (class → sample → method → seed).
        """
        items = [paths] if isinstance(paths, str | Path) else list(paths)
        files: list[Path] = []
        for it in items:
            q = Path(it)
            files.extend(BenchmarkResult.shard_files(q) if q.is_dir() else [q])
        if not files:
            raise ConfigError(f"no shard files (shard-*-of-*.json) found in {items}")
        parts = [BenchmarkResult.load(f) for f in files]
        ref = parts[0]
        keys = ("n_samples", "seeds", "classes", "budget_scale", "max_total_steps")
        keys += ("data_seed_offset", "allow_inverse_crime")
        n_shards = {(p.shard or (0, 1))[1] for p in parts}
        if len(n_shards) != 1:
            raise ConfigError(
                f"shards of different splits cannot be merged: n = {sorted(n_shards)}"
            )
        (n,) = n_shards
        for f, part in zip(files, parts):
            if (part.instance, part.methods) != (ref.instance, ref.methods):
                raise ConfigError(f"{f}: instance/methods differ from {files[0]}")
            if part.config != ref.config or any(part.meta.get(k) != ref.meta.get(k) for k in keys):
                raise ConfigError(f"{f}: protocol settings differ from {files[0]}")
        ids = [(p.shard or (0, 1))[0] for p in parts]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        missing = sorted(set(range(n)) - set(ids))
        if dup or missing:
            msg = f"shards: duplicated {dup}, missing {missing} (of {n})"
            if strict:
                raise ConfigError(msg + "; pass strict=False to merge anyway")
            log.warning(msg)
        rows = [r for p in parts for r in p.rows]
        classes = list(ref.meta.get("classes") or [])
        seeds = list(ref.meta.get("seeds") or [])

        def order(r: dict) -> tuple:
            c = classes.index(r["class"]) if r["class"] in classes else len(classes)
            m = ref.methods.index(r["method"]) if r["method"] in ref.methods else len(ref.methods)
            s = seeds.index(r["seed"]) if r["seed"] in seeds else len(seeds)
            return (c, int(r["sample"]), m, s)

        rows.sort(key=order)
        metric_names = list(ref.metric_names)
        for p in parts[1:]:
            metric_names += [k for k in p.metric_names if k not in metric_names]
        meta = {k: v for k, v in ref.meta.items() if k != "shard"}
        meta["merged_shards"] = sorted(ids)
        meta["n_shards"] = n
        if missing:
            meta["missing_shards"] = missing
        regime = {k: v for p in parts for k, v in p.regime.items()}
        fidelity = {k: v for p in parts for k, v in p.fidelity.items()}
        return BenchmarkResult(
            ref.instance, rows, list(ref.methods), metric_names, ref.config, regime, fidelity, meta
        )

    def __str__(self) -> str:
        return self.to_markdown()


def parse_shard(shard: Any) -> tuple[int, int]:
    """``None`` → ``(0, 1)``; ``(i, n)`` or ``"i/n"`` → ``(i, n)`` with ``0 ≤ i < n``."""
    if shard is None:
        return 0, 1
    if isinstance(shard, str):
        head, sep, tail = shard.partition("/")
        if not sep:
            raise ConfigError(f"shard must look like 'i/n' (e.g. 0/4), got {shard!r}")
        shard = (head, tail)
    try:
        i, n = (int(v) for v in shard)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"shard must be (i, n) or 'i/n', got {shard!r}") from e
    if n < 1 or not 0 <= i < n:
        raise ConfigError(f"shard index must satisfy 0 <= i < n, got {i}/{n}")
    return i, n


def benchmark_units(
    class_list: Sequence[Any], n_samples: int, seeds: Sequence[int], shard: Any = None
) -> list[tuple[Any, int, int]]:
    """The (class, sample, seed) run units of a benchmark, restricted to shard ``i`` of ``n``.

    Units are enumerated class → sample → seed and dealt round-robin to the shards (unit ``k``
    goes to shard ``k mod n``), so shards are balanced and every method of a unit runs in the same
    shard (paired comparisons stay within one job).
    """
    i, n = parse_shard(shard)
    units = [(c, s, z) for c in class_list for s in range(n_samples) for z in seeds]
    return units[i::n]


_BATCH_SOLVER_KWARGS = frozenset({"dtype", "nan_guard", "checkpoint_every", "max_bad_steps"})


def _batch_reason(prep: PreparedRun, cur: Curriculum, callbacks: Sequence[Any]) -> str | None:
    """Why a prepared method run cannot go through :func:`batch_invert` (``None`` if it can)."""
    if prep.runner is not None:
        return "it uses a custom runner"
    if callbacks:
        return "solver callbacks are not supported by batch_invert"
    extra = set(prep.solver_kwargs) - _BATCH_SOLVER_KWARGS
    if extra:
        return f"solver options {sorted(extra)} are not supported by batch_invert"
    return batchable_reason(prep.problem, cur)


# ------------------------------------------------------------------------------------------
# the protocol
# ------------------------------------------------------------------------------------------
def run_benchmark(
    instance: Any,
    methods: Any = None,
    n_samples: int = 4,
    seeds: Sequence[int] = (0, 1, 2),
    classes: Any = None,
    metrics: Any = None,
    device: str | torch.device = "auto",
    budget_scale: float = 1.0,
    out_dir: str | Path | None = None,
    progress: bool = True,
    *,
    allow_inverse_crime: bool = False,
    max_total_steps: int | None = None,
    data_seed_offset: int = 0,
    callbacks: Sequence[Any] = (),
    fail_fast: bool = False,
    warmup: bool = True,
    batched: bool = False,
    batch_size: int | None = None,
    shard: Any = None,
) -> BenchmarkResult:
    """Run every method on the same measurements with several seeds and aggregate the metrics.

    Args:
        instance: an :class:`~nefi.instances.Instance` (anything with ``make_measurement``,
            ``build_problem``, ``data_generator``, ``baselines``, ``evaluate``).
        methods: names (``"neural"`` or baseline keys), :class:`Method` objects, a
            comma-separated string, or ``None`` (default method + every baseline).
        n_samples: measurements per scene class (data seeds ``data_seed_offset + i``).
        seeds: optimization seeds (network init + solver) per measurement.
        classes: scene classes (``None`` = instance default, ``"all"``, or a list).
        metrics: ``None`` (``instance.evaluate``), metric names, or ``{name: fn}``.
        device: solver device.
        budget_scale: multiply every stage's step count (e.g. 0.1 for quick studies).
        out_dir: write the reports there (see :meth:`BenchmarkResult.save`).
        progress: show a progress bar over runs.
        allow_inverse_crime: permit matched operators (labelled "matched" everywhere).
        max_total_steps: cap on the total steps of every run (after ``budget_scale``).
        data_seed_offset: first data seed.
        callbacks: solver callbacks for every run.
        fail_fast: re-raise run errors instead of recording them in the ``error`` column.
        warmup: run one untimed optimization step per method first, so one-time costs (lazy
            imports of the optimizer stack, CUDA context / FFT plans) do not pollute the first
            timed run.
        batched: solve all (sample, seed) runs of a method and class in one batched loop
            (:func:`nefi.solve.batch_invert`) when the method allows it (batchable operator,
            Adam(W), no callbacks / custom runner); other methods run sequentially (logged).
            Results equal the sequential ones up to float rounding; ``time_s`` is the batch
            wall-clock divided by the batch size and rows carry the ``batch`` size.
        batch_size: maximum number of problems per batch (``None``: all runs of a method and
            class at once).
        shard: ``(i, n)`` or ``"i/n"``: run only shard ``i`` of ``n`` of the (class, sample, seed)
            units (see :func:`benchmark_units`); the result records ``meta["shard"]`` and can be
            written with :meth:`BenchmarkResult.save_shard` and merged with
            :meth:`BenchmarkResult.merge`.

    Returns:
        A :class:`BenchmarkResult`.

    Raises:
        InverseCrimeError: when a method inverts with the data-generation model and
            ``allow_inverse_crime`` is False (checked before any run).
    """
    from tqdm.auto import tqdm

    methods = resolve_methods(instance, methods)
    class_list = resolve_classes(instance, classes)
    seeds = [int(s) for s in seeds]
    if n_samples < 1 or not seeds:
        raise ConfigError("need n_samples >= 1 and at least one seed")
    dev = resolve_device(device)
    gen = instance.data_generator()
    fidelity = {"data": str(getattr(gen, "fidelity_tag", type(gen).__name__))}
    regime: dict[str, str] = {}

    # ---- pre-flight: inverse-crime guard on the first measurement --------------------
    cache: dict[tuple[Any, int], Any] = {}
    first = (class_list[0], data_seed_offset)
    cache[first] = instance.make_measurement(seed=data_seed_offset, scene_class=class_list[0])
    for m in methods:
        try:
            prep = m.prepare(instance, cache[first][1])
        except Exception as e:  # recorded per run below; the guard only aborts on crimes
            if fail_fast:
                raise
            log.warning("method %s cannot be built: %s: %s", m.name, type(e).__name__, e)
            fidelity[m.name], regime[m.name] = "?", "unknown"
            continue
        fidelity[m.name] = operator_fidelity_tag(prep.problem.operator)
        regime[m.name] = check_inverse_crime(
            gen, prep.problem.operator, allow_inverse_crime, label=m.name
        )
        if warmup:
            _warmup(prep, dev)

    shard_i, shard_n = parse_shard(shard)
    units = benchmark_units(class_list, n_samples, seeds, (shard_i, shard_n))
    by_row: dict[tuple, dict[str, Any]] = {}
    total = len(units) * len(methods)
    bar = tqdm(total=total, disable=not progress, desc=f"bench {getattr(instance, 'name', '')}")

    def measurement(cls: Any, i: int) -> tuple[Any, Any]:
        key = (cls, data_seed_offset + i)
        hit = cache.get(key)
        if hit is None:
            hit = cache[key] = instance.make_measurement(seed=key[1], scene_class=cls)
        return hit

    def base_row(m: Method, cls: Any, i: int, s: int) -> dict[str, Any]:
        row = {
            "method": m.name,
            "class": _class_label(instance, cls),
            "sample": i,
            "data_seed": data_seed_offset + i,
            "seed": s,
        }
        row.update(scene_columns(measurement(cls, i)[1]))
        return row

    def finish_row(row, res: Result, gt, dt: float, peak: float | None, batch: int | None) -> None:
        row.update(evaluate_metrics(instance, res, gt, metrics))
        n_steps = len(res.history.get("total", []))
        row.update(
            {
                "time_s": dt,
                "peak_mem_mb": peak,
                "steps": n_steps,
                "ms_per_step": 1000.0 * dt / n_steps if n_steps else None,
                "stop": ",".join(str(st.get("stop", "?")) for st in res.stage_results),
                "final_data_loss": _final_data_loss(res),
                "n_params": res.extra.get("n_parameters"),
                "error": res.extra.get("error"),
            }
        )
        if batch is not None:
            row["batch"] = batch

    def run_one(m: Method, cls: Any, i: int, s: int) -> None:
        gt, meas = measurement(cls, i)
        row = base_row(m, cls, i, s)
        try:
            seed_everything(s)  # network initialization depends on the seed
            prep = m.prepare(instance, meas)
            cur = scale_budget(prep.resolved_curriculum(), budget_scale, max_total_steps)
            reset_peak_memory(dev)
            t0 = time.perf_counter()
            res = prep.run(cur, device=dev, seed=s, callbacks=callbacks)
            synchronize(dev)
            finish_row(row, res, gt, time.perf_counter() - t0, peak_memory_mb(dev), None)
        except InverseCrimeError:
            raise
        except Exception as e:
            if fail_fast:
                raise
            log.warning("run %s failed: %s: %s", row, type(e).__name__, e)
            row["error"] = f"{type(e).__name__}: {e}"
        by_row[(cls, i, m.name, s)] = row
        bar.update(1)
        bar.set_postfix(method=m.name, sample=i, refresh=False)

    def run_batched(m: Method, cls: Any, items: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Batch the (sample, seed) items of method ``m``; returns the items left to run
        sequentially (non-batchable method, or a failed batch)."""
        preps = []
        try:
            for i, s in items:
                seed_everything(s)
                prep = m.prepare(instance, measurement(cls, i)[1])
                cur = scale_budget(prep.resolved_curriculum(), budget_scale, max_total_steps)
                why = _batch_reason(prep, cur, callbacks)
                if why is not None:
                    log.warning("method %s runs sequentially: %s", m.name, why)
                    return items
                preps.append((prep, cur))
        except InverseCrimeError:
            raise
        except Exception as e:  # preparation errors are recorded by the sequential path
            log.debug("batched preparation of %s failed: %s", m.name, e)
            return items
        leftover: list[tuple[int, int]] = []
        groups: dict[str, list[int]] = {}
        for k, (_, cur) in enumerate(preps):  # one batch per distinct curriculum
            groups.setdefault(json.dumps(to_jsonable(to_dict(cur)), sort_keys=True), []).append(k)
        size = max(1, int(batch_size)) if batch_size else None
        for ks in groups.values():
            chunks = [ks[j : j + size] for j in range(0, len(ks), size)] if size else [ks]
            for chunk in chunks:
                problems = [preps[k][0].problem for k in chunk]
                cur = preps[chunk[0]][1]
                kw = {k: v for k, v in preps[chunk[0]][0].solver_kwargs.items()}
                try:
                    reset_peak_memory(dev)
                    t0 = time.perf_counter()
                    results = batch_invert(
                        problems,
                        cur,
                        device=dev,
                        seeds=[items[k][1] for k in chunk],
                        raise_on_error=False,
                        **kw,
                    )
                    synchronize(dev)
                    dt = (time.perf_counter() - t0) / len(chunk)
                    peak = peak_memory_mb(dev)
                except Exception as e:
                    if fail_fast:
                        raise
                    log.warning(
                        "batched solve of %s failed (%s: %s); running it sequentially",
                        m.name,
                        type(e).__name__,
                        e,
                    )
                    leftover.extend(items[k] for k in chunk)
                    continue
                for k, res in zip(chunk, results):
                    i, s = items[k]
                    gt = measurement(cls, i)[0]
                    row = base_row(m, cls, i, s)
                    try:
                        finish_row(row, res, gt, dt, peak, len(chunk))
                    except Exception as e:
                        if fail_fast:
                            raise
                        row["error"] = f"{type(e).__name__}: {e}"
                    by_row[(cls, i, m.name, s)] = row
                    bar.update(1)
        return leftover

    if not batched:  # class -> sample -> method -> seed (measurements freed when done)
        for (cls, i), grp in itertools.groupby(units, key=lambda u: (u[0], u[1])):
            seeds_i = [u[2] for u in grp]
            for m in methods:
                for s in seeds_i:
                    run_one(m, cls, i, s)
            cache.pop((cls, data_seed_offset + i), None)
    else:
        for cls in class_list:
            items = [(i, s) for c, i, s in units if c == cls]
            if not items:
                continue
            for m in methods:
                for i, s in run_batched(m, cls, items):
                    run_one(m, cls, i, s)
            for i in {i for i, _ in items}:
                cache.pop((cls, data_seed_offset + i), None)
    bar.close()
    rows = [
        by_row[(cls, i, m.name, s)]
        for cls in class_list
        for i in range(n_samples)
        for m in methods
        for s in seeds
        if (cls, i, m.name, s) in by_row
    ]

    metric_names: list[str] = []
    reserved = {"method", "class", "sample", "data_seed", "seed", "error", "stop"}
    reserved |= {"time_s", "peak_mem_mb", "steps", "ms_per_step", "final_data_loss", "n_params"}
    reserved |= {"batch"}
    for r in rows:
        for k in r:
            if k.startswith(("meta/", "scene/")):
                continue  # scene strata columns (scene_columns) are not metrics
            if k not in reserved and k not in metric_names:
                metric_names.append(k)
    cfg = instance.config_dict() if hasattr(instance, "config_dict") else {}
    result = BenchmarkResult(
        instance=str(getattr(instance, "name", type(instance).__name__)),
        rows=rows,
        methods=[m.name for m in methods],
        metric_names=metric_names,
        config=to_dict(cfg),
        regime=regime,
        fidelity=fidelity,
        meta={
            "n_samples": n_samples,
            "seeds": seeds,
            "classes": [_class_label(instance, c) for c in class_list],
            "budget_scale": budget_scale,
            "max_total_steps": max_total_steps,
            "device": str(dev),
            "data_seed_offset": data_seed_offset,
            "allow_inverse_crime": allow_inverse_crime,
            "descriptions": {m.name: m.description for m in methods},
            **({"batched": True, "batch_size": batch_size} if batched else {}),
            **({"shard": [shard_i, shard_n]} if shard_n > 1 else {}),
        },
    )
    if out_dir is not None:
        if shard_n > 1:
            result.save_shard(out_dir)
        else:
            result.save(out_dir)
    return result


__all__ = [
    "BenchmarkResult",
    "InverseCrimeError",
    "Method",
    "PreparedRun",
    "baseline_method",
    "benchmark_units",
    "check_inverse_crime",
    "default_method",
    "evaluate_metrics",
    "operator_fidelity_tag",
    "parse_shard",
    "resolve_classes",
    "resolve_methods",
    "resolve_metric_fns",
    "run_benchmark",
    "scale_budget",
    "solve_problem",
]

# Re-export the other bench modules (they import from this module, hence the bottom imports).
from . import report as _report  # noqa: E402
from .ablation import *  # noqa: E402,F403
from .ablation import __all__ as _ablation_all  # noqa: E402
from .sweep import *  # noqa: E402,F403
from .sweep import __all__ as _sweep_all  # noqa: E402

__all__ += list(_report.__all__) + list(_ablation_all) + list(_sweep_all)
from .report import *  # noqa: E402,F403
