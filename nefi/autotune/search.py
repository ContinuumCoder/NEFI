"""Held-out hyper-parameter search: let the measurement itself choose the configuration.

There is no validation set for a per-measurement inverse problem, but the measurement has many
entries and the physics-faithful forward model predicts *every* entry from the unknown: hold out
a random ``holdout`` fraction of the observed entries (:func:`nefi.fields.adaptive.holdout_masks`),
fit each candidate configuration on the rest (short budgets suffice to rank, with a leakage-free
coarse-stage down-sampler), and score the prediction on the held-out entries — an estimate of the
prediction risk ``E‖F(x̂) − F(x*)‖² + σ²`` that penalizes both under-regularized configurations
(they fit noise) and over-regularized ones (they cannot express the unknown). With
``objective="discrepancy"`` the score is instead ``|log(RMSE / τσ)|`` on all entries (Morozov as a
selection rule).

:func:`autotune` draws configurations from a small space with a scrambled Sobol sequence
(``torch.quasirandom``, no extra dependency); trial 0 is always the unmodified default, so the
search never reports something worse than the default on its own objective. Dimensions:

* ``lr`` (log-uniform, ×⅓…×3 of the default peak rate), ``anneal_fraction`` (0.25…1),
  ``steps`` (total steps) — applied to the curriculum;
* ``weight.<term>`` (log-uniform, ×0.1…×10 of the default) — applied to the base weight *and*
  every stage override of the term;
* anything else (``n_octaves``, ``hidden``, ``representation``, instance config fields …) —
  passed to ``problem_factory(**params)``; ``n_octaves`` is in the default space when the
  factory accepts it and the field has annealed Fourier features.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..errors import ConfigError
from ..measurement import Measurement
from ..solve.curriculum import Curriculum
from ._common import (
    Timer,
    base_curriculum,
    effective_weight,
    fit,
    fmt,
    md_table,
    noise_float,
    peak_lr,
    probe_version,
    rescaled,
    residual,
    scale_lr,
    scale_weights,
)
from .regularization import regularizer_names

log = logging.getLogger("nefi")

OBJECTIVES = ("heldout", "discrepancy")
#: space keys applied to the curriculum / loss weights by :func:`autotune` itself
CURRICULUM_KEYS = ("lr", "anneal_fraction", "steps")


@dataclass
class Dimension:
    """One search dimension: ``kind`` ∈ ``"log"``, ``"linear"``, ``"int"``, ``"choice"``."""

    name: str
    kind: str
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] | None = None
    default: Any = None

    def __post_init__(self) -> None:
        if self.kind not in ("log", "linear", "int", "choice"):
            raise ConfigError(f"dimension {self.name!r}: unknown kind {self.kind!r}")
        if self.kind == "choice":
            if not self.choices:
                raise ConfigError(f"dimension {self.name!r}: a choice needs choices")
        elif self.low is None or self.high is None or not self.high >= self.low:
            raise ConfigError(f"dimension {self.name!r}: needs low <= high")
        if self.kind == "log" and not (self.low and self.low > 0):
            raise ConfigError(f"dimension {self.name!r}: a log range needs low > 0")

    def sample(self, u: float) -> Any:
        """Map ``u ∈ [0, 1)`` to a value of the dimension."""
        u = min(max(float(u), 0.0), 1.0 - 1e-12)
        if self.kind == "choice":
            return self.choices[int(u * len(self.choices))]  # type: ignore[index]
        lo, hi = float(self.low), float(self.high)  # type: ignore[arg-type]
        if self.kind == "log":
            return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        if self.kind == "int":
            return int(math.floor(lo + u * (hi - lo + 1)))
        return lo + u * (hi - lo)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "low": self.low,
            "high": self.high,
            "choices": list(self.choices) if self.choices else None,
            "default": self.default,
        }


def as_dimension(name: str, spec: Any) -> Dimension:
    """``Dimension`` from a spec: a :class:`Dimension`, ``("log"|"linear"|"int", lo, hi)``, a
    list of choices, or a ``(lo, hi)`` pair (log-uniform when both are positive)."""
    if isinstance(spec, Dimension):
        return spec
    if isinstance(spec, list):
        return Dimension(name, "choice", choices=tuple(spec))
    if isinstance(spec, tuple) and len(spec) == 3 and isinstance(spec[0], str):
        return Dimension(name, spec[0], float(spec[1]), float(spec[2]))
    if isinstance(spec, tuple) and len(spec) == 2:
        lo, hi = float(spec[0]), float(spec[1])
        return Dimension(name, "log" if lo > 0 else "linear", lo, hi)
    raise ConfigError(f"cannot interpret the search dimension {name!r}: {spec!r}")


def _accepts(fn: Callable, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()) or name in sig.parameters


def _octaves(problem: Any) -> int | None:
    enc = getattr(problem.field, "encoding", None)
    k = getattr(enc, "n_octaves", None)
    if isinstance(k, int) and getattr(enc, "annealed", True):
        return k
    return None


def default_space(
    problem_factory: Callable[..., Any], problem: Any, curriculum: Curriculum | None = None
) -> dict[str, Dimension]:
    """The default search space for a problem built by ``problem_factory()``.

    ``lr`` (×⅓…×3), ``weight.<term>`` for every active regularizer (×0.1…×10), ``anneal_fraction``
    (0.25…1, when a stage anneals) and ``n_octaves`` (±2, when the factory accepts it and the field
    has annealed Fourier features).
    """
    cur = base_curriculum(problem, curriculum)
    lr0 = peak_lr(cur)
    space: dict[str, Dimension] = {"lr": Dimension("lr", "log", lr0 / 3.0, lr0 * 3.0, default=lr0)}
    for k in regularizer_names(problem, cur):
        w0 = effective_weight(problem.losses.weights, cur, k)
        if w0 > 0:
            space[f"weight.{k}"] = Dimension(f"weight.{k}", "log", w0 / 10, w0 * 10, default=w0)
    if any(s.anneal for s in cur.stages):
        af = max(s.anneal_fraction for s in cur.stages)
        space["anneal_fraction"] = Dimension("anneal_fraction", "linear", 0.25, 1.0, default=af)
    k0 = _octaves(problem)
    if k0 is not None and _accepts(problem_factory, "n_octaves"):
        space["n_octaves"] = Dimension("n_octaves", "int", max(2, k0 - 2), k0 + 2, default=k0)
    return space


def apply_params(
    problem_factory: Callable[..., Any],
    params: Mapping[str, Any],
    curriculum: Curriculum | None = None,
) -> tuple[Any, Curriculum]:
    """Build ``(problem, curriculum)`` for one configuration.

    Keys in :data:`CURRICULUM_KEYS` and ``weight.<term>`` are applied to the curriculum / loss
    weights of the built problem; every other key is passed to ``problem_factory``.
    """
    fkw = {
        k: v for k, v in params.items() if k not in CURRICULUM_KEYS and not k.startswith("weight.")
    }
    problem = problem_factory(**fkw)
    cur = base_curriculum(problem, curriculum)
    if "steps" in params:
        cur = rescaled(cur, int(params["steps"]))
    if "lr" in params:
        cur = scale_lr(cur, float(params["lr"]) / peak_lr(cur))
    if "anneal_fraction" in params:
        for s in cur.stages:
            if s.anneal:
                s.anneal_fraction = float(min(1.0, max(0.05, params["anneal_fraction"])))
    factors: dict[str, float] = {}
    absolute: dict[str, float] = {}
    for k, v in params.items():
        if k.startswith("weight."):
            term = k.split(".", 1)[1]
            if term not in problem.losses.names:
                raise ConfigError(f"{k}: no loss term {term!r}; terms: {problem.losses.names}")
            w0 = effective_weight(problem.losses.weights, cur, term)
            if w0 > 0:
                factors[term] = float(v) / w0
            else:  # a disabled term is switched on with the given weight
                absolute[term] = float(v)
                for st in cur.stages:
                    if st.loss_weights and term in st.loss_weights:
                        st.loss_weights = {**st.loss_weights, term: float(v)}
    if factors or absolute:
        weights, cur = scale_weights(problem.losses.weights, cur, factors)
        weights.update(absolute)
        problem = dataclasses.replace(
            problem,
            losses=problem.losses.with_weights({k: weights[k] for k in [*factors, *absolute]}),
            meta=dict(problem.meta),
        )
    problem = dataclasses.replace(problem, curriculum=cur, meta=dict(problem.meta))
    return problem, cur


@dataclass
class Trial:
    """One evaluated configuration."""

    index: int
    params: dict[str, Any]
    score: float
    val_mse: float | None
    train_rmse: float | None
    chi: float | None
    steps: int
    seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class TuneReport:
    """Outcome of :func:`autotune`: best configuration, all trials, and its curriculum."""

    best: dict[str, Any]
    best_score: float
    default_score: float
    trials: list[Trial]
    space: dict[str, Dimension]
    objective: str
    curriculum: Curriculum | None
    holdout: float
    budget_scale: float
    seed: int
    probe_steps: int = 0
    seconds: float = 0.0

    @property
    def improvement(self) -> float:
        """Relative objective improvement of the best trial over the default (trial 0)."""
        if not (math.isfinite(self.default_score) and self.default_score > 0):
            return math.nan
        return 1.0 - self.best_score / self.default_score

    def build(
        self, problem_factory: Callable[..., Any], curriculum: Curriculum | None = None
    ) -> tuple[Any, Curriculum]:
        """``(problem, curriculum)`` of the best configuration (full budget)."""
        return apply_params(problem_factory, self.best, curriculum)

    def table(self) -> str:
        names = list(self.space)
        head = ["#", *names, "score", "val MSE", "train χ", "steps", "s"]
        rows = []
        for t in sorted(self.trials, key=lambda t: (not math.isfinite(t.score), t.score)):
            cells = [str(t.index) + (" (default)" if t.index == 0 else "")]
            cells += [fmt(t.params.get(n, self.space[n].default)) for n in names]
            score = fmt(t.score) + (f" ({t.error})" if t.error else "")
            cells += [score, fmt(t.val_mse), fmt(t.chi), t.steps, t.seconds]
            rows.append(cells)
        return md_table(head, rows)

    def to_markdown(self) -> str:
        imp = self.improvement
        lines = [
            f"**{self.objective}** search, {len(self.trials)} trials (budget ×"
            f"{self.budget_scale:g}"
            + (f", hold-out {self.holdout:.0%}" if self.objective == "heldout" else "")
            + f"): best {fmt(self.best_score)} vs default {fmt(self.default_score)}"
            + (f" ({100 * imp:+.0f} %)" if math.isfinite(imp) else ""),
            "",
            self.table(),
            "",
            "Best: " + (", ".join(f"{k} = {fmt(v)}" for k, v in self.best.items()) or "defaults"),
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_markdown()

    def to_dict(self) -> dict[str, Any]:
        return {
            "best": dict(self.best),
            "best_score": self.best_score,
            "default_score": self.default_score,
            "improvement": self.improvement,
            "trials": [t.to_dict() for t in self.trials],
            "space": {k: d.to_dict() for k, d in self.space.items()},
            "objective": self.objective,
            "holdout": self.holdout,
            "budget_scale": self.budget_scale,
            "seed": self.seed,
            "probe_steps": self.probe_steps,
            "seconds": self.seconds,
        }


def _heldout_mse(pred: torch.Tensor, meas: Measurement, val: torch.Tensor) -> float:
    return residual(pred, Measurement(meas.data, val, None, {}))[0] ** 2


def autotune(
    problem_factory: Callable[..., Any],
    *,
    space: Mapping[str, Any] | None = None,
    holdout: float = 0.1,
    budget_scale: float = 0.2,
    trials: int = 8,
    seed: int = 0,
    objective: str = "heldout",
    curriculum: Curriculum | None = None,
    group_axes: Sequence[int] | None = None,
    tau: float = 1.0,
    device: str = "cpu",
) -> TuneReport:
    """Held-out (or discrepancy) search over a small hyper-parameter space.

    Args:
        problem_factory: ``problem_factory(**params) -> InverseProblem``; called with no
            arguments for the default and with the non-curriculum keys of each configuration.
            All problems must share the measurement.
        space: ``{name: Dimension | (kind, lo, hi) | [choices] | (lo, hi)}`` (default:
            :func:`default_space`).
        holdout: held-out fraction of the observed entries (``objective="heldout"``).
        budget_scale: steps per trial as a fraction of the configuration's curriculum.
        trials: number of configurations including the default (trial 0).
        seed: split, Sobol and solver seed.
        objective: ``"heldout"`` (held-out MSE) or ``"discrepancy"`` (``|log(RMSE/τσ)|``).
        curriculum: base curriculum (default: each problem's own).
        group_axes: share the hold-out split along these measurement axes (e.g. whole sensors).
        tau: Morozov factor of the ``"discrepancy"`` objective.
        device: solver device.

    Returns:
        :class:`TuneReport` (``report.best``, ``report.table()``, ``report.curriculum``).

    Example::

        from nefi.instances.toy1d import Toy1D
        inst = Toy1D(n=64); gt, meas = inst.make_measurement(0)
        report = autotune(lambda **hp: Toy1D(n=64, **hp).build_problem(meas), trials=6)
        problem, curriculum = report.build(lambda **hp: Toy1D(n=64, **hp).build_problem(meas))
    """
    from ..fields.adaptive.selection import holdout_masks, masked_downsampler

    if objective not in OBJECTIVES:
        raise ConfigError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    if trials < 1:
        raise ConfigError("trials must be >= 1")
    timer = Timer()
    ref = problem_factory()
    dims = (
        {k: as_dimension(k, v) for k, v in space.items()}
        if space is not None
        else default_space(problem_factory, ref, curriculum)
    )
    meas = ref.measurement
    sigma = noise_float(meas.noise_std)
    if objective == "discrepancy" and not sigma:
        raise ConfigError("objective='discrepancy' needs measurement.noise_std (or estimate it)")
    train = val = None
    if objective == "heldout":
        train, val = holdout_masks(meas, holdout, seed, group_axes)[0]
    names = list(dims)
    configs: list[dict[str, Any]] = [{}]
    if trials > 1 and names:
        engine = torch.quasirandom.SobolEngine(len(names), scramble=True, seed=int(seed))
        for row in engine.draw(trials - 1).tolist():
            configs.append({n: dims[n].sample(u) for n, u in zip(names, row)})
    results: list[Trial] = []
    for i, params in enumerate(configs):
        t0 = time.perf_counter()
        try:
            problem, cur = apply_params(problem_factory, params, curriculum)
            if tuple(problem.measurement.data.shape) != tuple(meas.data.shape):
                raise ConfigError("problem_factory returned a different measurement")
            probe = probe_version(cur.scaled(budget_scale) if budget_scale != 1.0 else cur)
            if objective == "heldout":
                assert train is not None and val is not None
                tm = Measurement(meas.data, train, meas.noise_std, dict(meas.meta))
                problem = dataclasses.replace(
                    problem,
                    measurement=tm,
                    downsample_obs=masked_downsampler(problem.operator, problem.downsample_obs),
                )
                out = fit(problem, probe, seed=seed, device=device, sigma=sigma)
                vm = _heldout_mse(out.result.pred, meas, val)
                score, val_mse = vm, vm
            else:
                out = fit(problem, probe, seed=seed, device=device, sigma=sigma)
                chi = out.chi if out.chi is not None else math.inf
                score = abs(math.log(max(chi, 1e-300) / tau)) if math.isfinite(chi) else math.inf
                val_mse = None
            timer.add(out)
            results.append(
                Trial(
                    i,
                    dict(params),
                    float(score) if math.isfinite(score) else math.inf,
                    val_mse,
                    out.rmse,
                    out.chi,
                    out.steps,
                    time.perf_counter() - t0,
                )
            )
        except Exception as e:  # noqa: BLE001 - a failing configuration must not abort the search
            log.warning("autotune: trial %d %s failed: %s", i, params, e)
            results.append(
                Trial(i, dict(params), math.inf, None, None, None, 0, time.perf_counter() - t0,
                      f"{type(e).__name__}: {e}")
            )  # fmt: skip
        log.info("autotune trial %d %s: score %.4g", i, params, results[-1].score)
    best = min(results, key=lambda t: (not math.isfinite(t.score), t.score))
    try:
        _, best_cur = apply_params(problem_factory, best.params, curriculum)
    except Exception:  # noqa: BLE001 - reported through the trial table
        best_cur = None
    return TuneReport(
        best=dict(best.params),
        best_score=best.score,
        default_score=results[0].score,
        trials=results,
        space=dims,
        objective=objective,
        curriculum=best_cur,
        holdout=float(holdout),
        budget_scale=float(budget_scale),
        seed=int(seed),
        probe_steps=timer.steps,
        seconds=timer.seconds,
    )


__all__ = [
    "CURRICULUM_KEYS",
    "OBJECTIVES",
    "Dimension",
    "Trial",
    "TuneReport",
    "apply_params",
    "as_dimension",
    "autotune",
    "default_space",
]
