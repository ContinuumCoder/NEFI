"""Measurement-driven model selection: cross-validation for per-measurement inverse problems.

There is no training set to validate a representation on — but the measurement itself has many
entries, and a physics-faithful forward model predicts *every* entry from the unknown. Holding out
a random subset of entries (``Measurement.mask``), fitting each candidate representation on the
rest, and scoring the prediction on the held-out entries is therefore principled cross-validation
for inverse problems: a representation that fits noise (under-regularized, e.g. a free grid with
all frequencies open, NeFTY Cor. 1) predicts held-out entries poorly, and so does one that cannot
express the unknown (over-regularized). The held-out error estimates the *prediction* risk
``E‖F(x̂) − F(x*)‖² + σ²``, which is minimized by the representation whose prior best matches the
system — the data choose the geometry.

Leakage guard: coarse curriculum stages compare against down-sampled data; the down-sampler used
for the training fit averages only *observed training* entries (masked normalized averaging), so
held-out values never enter the fit at any resolution.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from ...errors import ConfigError
from ...losses.base import Context
from ...measurement import Measurement
from ...registry import register
from ...solve.curriculum import Curriculum
from ...solve.result import Result
from ...solve.solver import Solver
from ...utils.seed import seed_everything
from ...utils.tensor import resample, shape_tuple
from ..base import Field
from ..heads import Heads

log = logging.getLogger("nefi")


# ------------------------------------------------------------------------------------------
# hold-out masks and leakage-free down-sampling
# ------------------------------------------------------------------------------------------
def holdout_masks(
    measurement: Measurement,
    holdout: float = 0.1,
    seed: int = 0,
    group_axes: Sequence[int] | None = None,
    n_folds: int = 1,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Random ``(train_mask, val_mask)`` pairs over the observed entries of a measurement.

    Args:
        measurement: the observation (its own mask, if any, is respected).
        holdout: held-out fraction for a single split (``n_folds == 1``).
        seed: split seed.
        group_axes: axes along which the split is shared (e.g. ``(0,)`` to hold out whole
            sensor pixels of a ``(frames, H, W)`` measurement across all frames).
        n_folds: ``1`` = one random split; ``k >= 2`` = k-fold (every entry held out once).

    Returns:
        List of ``(train, val)`` float masks shaped like ``measurement.data``.
    """
    data = measurement.data
    shape = tuple(data.shape)
    base = (
        torch.ones(shape, dtype=torch.bool)
        if measurement.mask is None
        else (measurement.mask.expand_as(data) > 0).cpu()
    )
    group = {int(a) % len(shape) for a in (group_axes or ())}
    draw_shape = tuple(1 if i in group else s for i, s in enumerate(shape))
    gen = torch.Generator().manual_seed(int(seed))
    u = torch.rand(draw_shape, generator=gen).expand(shape)
    if n_folds <= 1:
        if not 0.0 < holdout < 1.0:
            raise ConfigError("holdout must be in (0, 1)")
        val = (u < holdout) & base
        folds = [(base & ~val, val)]
    else:
        fold_id = torch.clamp((u * n_folds).long(), max=n_folds - 1)
        folds = [(base & (fold_id != k), base & (fold_id == k)) for k in range(n_folds)]
    out = []
    for tr, va in folds:
        if int(va.sum()) == 0 or int(tr.sum()) == 0:
            raise ConfigError("hold-out split left an empty train or validation set")
        out.append((tr.to(data.dtype), va.to(data.dtype)))
    return out


def masked_downsampler(
    operator: Any,
    original: Callable[[Measurement, tuple[int, ...]], Measurement] | None = None,
    min_fraction: float = 0.0,
) -> Callable[[Measurement, tuple[int, ...]], Measurement]:
    """Down-sampler that averages only observed entries: data ``D(y·m) / D(m)``, weights ``D(m)``.

    The coarse mask keeps the *fractional* observed weight of each coarse cell (the convention
    of :meth:`Measurement.resampled`); cells whose weight is below ``min_fraction`` are dropped.
    With ``original`` (the problem's own ``downsample_obs``) it is applied to ``y·m`` and ``m``
    separately (exact for linear down-samplers such as area averaging); otherwise the data are
    area-resampled to ``operator.output_shape(field_shape)``.
    """

    def down(m: Measurement, field_shape: tuple[int, ...]) -> Measurement:
        mask = (
            torch.ones_like(m.data) if m.mask is None else m.mask.expand_as(m.data).to(m.data.dtype)
        )
        if original is not None:
            num = original(Measurement(m.data * mask, None, m.noise_std, dict(m.meta)), field_shape)
            den = original(Measurement(mask, None, None, {}), field_shape)
            num_t, den_t = num.data, den.data
        else:
            out = operator.output_shape(shape_tuple(field_shape))
            if out is None:
                raise ConfigError(
                    "cannot infer the coarse measurement shape: implement Operator.output_shape or "
                    "give the problem a downsample_obs"
                )
            out = shape_tuple(out)
            if tuple(m.data.shape) == out:
                return m
            num_t, den_t = resample(m.data * mask, out), resample(mask, out)
        den_t = den_t.clamp(0.0, 1.0)
        ok = den_t > 1e-12
        data = torch.where(ok, num_t / den_t.clamp_min(1e-12), torch.zeros_like(num_t))
        weight = torch.where(den_t >= min_fraction, den_t, torch.zeros_like(den_t))
        return Measurement(data, weight.to(data.dtype), m.noise_std, dict(m.meta))

    return down


# ------------------------------------------------------------------------------------------
# reports
# ------------------------------------------------------------------------------------------
@dataclass
class CandidateScore:
    """Cross-validation outcome of one candidate representation.

    Attributes:
        name: candidate name.
        score: the ranked quantity (lower is better; ``inf`` on failure).
        val_mse: mean squared prediction error on held-out entries.
        train_mse: same on the training entries (``val_mse / train_mse`` > 1 indicates overfit).
        val_relative: ``val_mse / mean(y²)`` on held-out entries.
        val_nmse: ``val_mse / σ²`` (≈ 1 at the noise floor) when the noise level is known.
        n_parameters: trainable parameters of the fitted field.
        seconds: fit time (all folds).
        steps: optimization steps (last fold).
        final_progress: annealing progress at which the fit was evaluated.
        error: exception text if the candidate failed.
        field: the fitted field (last fold) when ``keep_fields``.
        result: the solver :class:`~nefi.solve.Result` (last fold) when ``keep_results``.
    """

    name: str
    score: float
    val_mse: float
    train_mse: float
    val_relative: float
    val_nmse: float | None
    n_parameters: int
    seconds: float
    steps: int
    final_progress: float
    error: str | None = None
    field: Field | None = None
    result: Result | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            k: getattr(self, k)
            for k in (
                "name",
                "score",
                "val_mse",
                "train_mse",
                "val_relative",
                "val_nmse",
                "n_parameters",
                "seconds",
                "steps",
                "final_progress",
                "error",
            )
        }
        return d


@dataclass
class SelectionReport:
    """Ranked cross-validation report (best first); ``report.winner`` is the chosen name."""

    ranking: list[CandidateScore]
    holdout: float
    n_folds: int
    n_train: int
    n_val: int
    noise_std: float | None
    budget_scale: float
    seed: int
    score_name: str
    masks: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list, repr=False)
    refit: Result | None = None
    refit_field: Field | None = None

    @property
    def winner(self) -> str:
        return self.ranking[0].name

    @property
    def winner_field(self) -> Field | None:
        return self.refit_field if self.refit_field is not None else self.ranking[0].field

    def scores(self) -> dict[str, float]:
        return {c.name: c.score for c in self.ranking}

    def __getitem__(self, name: str) -> CandidateScore:
        for c in self.ranking:
            if c.name == name:
                return c
        raise KeyError(name)

    def table(self) -> str:
        """Markdown table of the ranking."""
        head = (
            "| rank | candidate | score | val MSE | train MSE | val/train | val NMSE | params | "
            "time [s] |\n|---|---|---|---|---|---|---|---|---|"
        )
        rows = []
        for i, c in enumerate(self.ranking, 1):
            ratio = c.val_mse / c.train_mse if c.train_mse > 0 else math.nan
            nmse = "—" if c.val_nmse is None else f"{c.val_nmse:.3g}"
            err = f" ({c.error})" if c.error else ""
            rows.append(
                f"| {i} | {c.name}{err} | {c.score:.4g} | {c.val_mse:.4g} | {c.train_mse:.4g} | "
                f"{ratio:.3g} | {nmse} | {c.n_parameters} | {c.seconds:.2f} |"
            )
        info = (
            f"hold-out {self.holdout:.0%} of {self.n_train + self.n_val} entries "
            f"({self.n_folds} fold(s)), budget ×{self.budget_scale:g}, score = {self.score_name}; "
            f"winner: **{self.winner}**"
        )
        return "\n".join([head, *rows, "", info])

    def __str__(self) -> str:
        return self.table()

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "ranking": [c.to_dict() for c in self.ranking],
            "holdout": self.holdout,
            "n_folds": self.n_folds,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "noise_std": self.noise_std,
            "budget_scale": self.budget_scale,
            "seed": self.seed,
            "score": self.score_name,
        }


# ------------------------------------------------------------------------------------------
# selection
# ------------------------------------------------------------------------------------------
def _n_required_args(fn: Callable) -> int:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return 1
    n = 0
    for prm in sig.parameters.values():
        if prm.kind in (prm.VAR_POSITIONAL,):
            return 1
        if (
            prm.kind in (prm.POSITIONAL_ONLY, prm.POSITIONAL_OR_KEYWORD)
            and prm.default is prm.empty
        ):
            n += 1
    return n


def build_problem(problem_factory: Callable[..., Any], field: Field) -> Any:
    """``problem_factory(field)``, or ``replace(problem_factory(), field=field)`` for 0-arg
    factories (the replacement re-validates the operator's required fields)."""
    if _n_required_args(problem_factory) == 0:
        p = problem_factory()
        return dataclasses.replace(p, field=field)
    return problem_factory(field)


def _make_field(spec: Callable[[], Field] | Field) -> Field:
    if isinstance(spec, Field):
        return copy.deepcopy(spec)
    f = spec()
    if not isinstance(f, Field):
        raise ConfigError(f"candidate factory returned {type(f).__name__}, not a Field")
    return f


def _noise_float(ns: Any) -> float | None:
    if ns is None:
        return None
    return float(ns.mean()) if torch.is_tensor(ns) else float(ns)


def _score_fit(problem, full: Measurement, train, val, progress: float, score):
    with torch.no_grad():
        fields, pred = problem.evaluate(None, progress=progress)
    y = full.data.to(pred)
    if tuple(pred.shape) != tuple(y.shape):
        raise ConfigError(f"prediction {tuple(pred.shape)} vs measurement {tuple(y.shape)}")
    tr, va = train.to(pred), val.to(pred)
    r2 = (pred - y) ** 2
    val_mse = float((r2 * va).sum() / va.sum())
    train_mse = float((r2 * tr).sum() / tr.sum())
    val_rel = val_mse / max(float((y**2 * va).sum() / va.sum()), 1e-300)
    if callable(score):
        s = float(score(pred, y, va))
    elif score == "mse":
        s = val_mse
    elif score == "relative":
        s = val_rel
    elif score == "data_loss":
        obs = Measurement(y, va, full.noise_std, dict(full.meta))
        dom = problem.domain
        op = problem.operator.at_resolution(dom.shape)
        ctx = Context(dict(fields), pred, obs, dom, op, problem.field, None, 0, progress)
        losses = problem.losses
        keep = set(losses.data_terms())
        with torch.no_grad():
            if keep:
                sel = losses.with_weights({k: 0.0 for k in losses.names if k not in keep})
                total, _ = sel(ctx)
            else:
                total, _ = losses(ctx)
        s = float(total)
    else:
        raise ConfigError(
            f"unknown score {score!r}; use 'mse', 'relative', 'data_loss' or a callable"
        )
    return s, val_mse, train_mse, val_rel


def select_representation(
    problem_factory: Callable[..., Any],
    candidates: Mapping[str, Callable[[], Field] | Field],
    curriculum: Curriculum | None = None,
    holdout: float = 0.1,
    seed: int = 0,
    budget_scale: float = 0.2,
    *,
    curricula: Mapping[str, Curriculum] | None = None,
    score: str | Callable[[torch.Tensor, torch.Tensor, torch.Tensor], float] = "mse",
    group_axes: Sequence[int] | None = None,
    n_folds: int = 1,
    device: str = "auto",
    callbacks: Callable[[str], Sequence[Any]] | None = None,
    keep_fields: bool = True,
    keep_results: bool = False,
    refit: bool = False,
    min_fraction: float = 0.0,
    solver_kw: Mapping[str, Any] | None = None,
) -> SelectionReport:
    """Choose a representation by held-out prediction error on the measurement itself.

    For every candidate (and fold): a fresh field is built (after seeding, so all candidates see
    the same randomness), the problem is rebuilt around it with the measurement mask restricted to
    the training entries (and a leakage-free coarse-stage down-sampler), the solver runs the
    curriculum scaled by ``budget_scale``, and the fitted forward prediction is scored on the
    held-out entries. Candidates that raise are reported with ``score = inf``.

    Args:
        problem_factory: ``field -> InverseProblem`` (or a 0-arg factory whose problem's field is
            replaced).
        candidates: ``name -> (() -> Field)`` factories (or Field instances, deep-copied).
        curriculum: shared curriculum (default: the problem's, else two-stage multiscale).
        holdout: held-out fraction (single split).
        seed: seed for the split, the fields and the solver.
        budget_scale: multiply every stage's step count (short fits suffice to rank).
        curricula: per-candidate curricula (e.g. a higher learning rate for grids).
        score: ``"mse"`` (default), ``"relative"``, ``"data_loss"`` (the problem's data terms on
            the held-out mask; only meaningful for entry-separable losses) or a callable
            ``(pred, data, val_mask) -> float``.
        group_axes: share the split along these measurement axes (see :func:`holdout_masks`).
        n_folds: ``k >= 2`` → k-fold cross-validation (scores averaged over folds).
        device: solver device.
        callbacks: optional ``name -> list of callbacks`` factory (fresh callbacks per fit).
        keep_fields: keep the fitted fields in the report (last fold).
        keep_results: keep the solver results (last fold).
        refit: refit the winner on all entries with the unscaled curriculum
            (``report.refit`` / ``report.refit_field``).
        min_fraction: coarse cells observed below this fraction are dropped (masked
            down-sampler).
        solver_kw: extra :class:`~nefi.solve.Solver` kwargs.

    Returns:
        :class:`SelectionReport` (``report.winner``, ``report.ranking``, ``report.table()``).

    Example::

        report = select_representation(
            lambda f: InverseProblem(dom, f, op, losses, meas),
            {"neural": lambda: NeuralField(2, heads), "layered": lambda: LayerCakeField(2)},
            holdout=0.1, budget_scale=0.2)
        print(report.table()); best = report.winner
    """
    if not candidates:
        raise ConfigError("no candidates given")
    names = list(candidates)
    seed_everything(seed)
    ref_problem = build_problem(problem_factory, _make_field(candidates[names[0]]))
    full = ref_problem.measurement
    base_cur = curriculum or getattr(ref_problem, "curriculum", None)
    if base_cur is None:
        base_cur = Curriculum.multiscale(ref_problem.domain.shape)
    masks = holdout_masks(full, holdout, seed, group_axes, n_folds)
    noise = _noise_float(full.noise_std)
    score_name = score if isinstance(score, str) else getattr(score, "__name__", "custom")
    solver_kw = dict(solver_kw or {})

    results: list[CandidateScore] = []
    for name in names:
        cur = (curricula or {}).get(name, base_cur)
        cur = cur.scaled(budget_scale) if budget_scale != 1.0 else cur
        t0 = time.perf_counter()
        fold_scores: list[tuple[float, float, float, float]] = []
        fld = res = None
        steps, prog, n_par = 0, 1.0, 0
        err = None
        try:
            for train, val in masks:
                seed_everything(seed)
                fld = _make_field(candidates[name])
                problem = build_problem(problem_factory, fld)
                op_state = copy.deepcopy(problem.operator.state_dict())
                train_meas = Measurement(full.data, train, full.noise_std, dict(full.meta))
                problem = dataclasses.replace(
                    problem,
                    measurement=train_meas,
                    downsample_obs=masked_downsampler(
                        problem.operator, problem.downsample_obs, min_fraction
                    ),
                )
                cbs = list(callbacks(name)) if callbacks is not None else []
                res = Solver(
                    problem, cur, device=device, seed=seed, callbacks=cbs, **solver_kw
                ).run()
                last = res.stage_results[-1] if res.stage_results else {}
                prog = float(last.get("final_progress", 1.0))
                steps = len(res.history.get("total", []))
                n_par = fld.n_parameters()
                fold_scores.append(_score_fit(problem, full, train, val, prog, score))
                problem.operator.load_state_dict(op_state)
        except Exception as e:  # a failing candidate must not abort the selection
            log.warning("select_representation: candidate %r failed: %s", name, e)
            err = f"{type(e).__name__}: {e}"
        secs = time.perf_counter() - t0
        if err is not None or not fold_scores:
            results.append(
                CandidateScore(
                    name,
                    math.inf,
                    math.inf,
                    math.inf,
                    math.inf,
                    None,
                    n_par,
                    secs,
                    steps,
                    prog,
                    err,
                )
            )
            continue
        agg = [sum(v[i] for v in fold_scores) / len(fold_scores) for i in range(4)]
        s, val_mse, train_mse, val_rel = agg
        results.append(
            CandidateScore(
                name=name,
                score=s,
                val_mse=val_mse,
                train_mse=train_mse,
                val_relative=val_rel,
                val_nmse=None if not noise else val_mse / noise**2,
                n_parameters=n_par,
                seconds=secs,
                steps=steps,
                final_progress=prog,
                field=fld if keep_fields else None,
                result=res if keep_results else None,
            )
        )
    results.sort(key=lambda c: (not math.isfinite(c.score), c.score))
    n_val = int(masks[0][1].sum())
    n_train = int(masks[0][0].sum())
    report = SelectionReport(
        ranking=results,
        holdout=holdout if n_folds <= 1 else 1.0 / n_folds,
        n_folds=max(1, n_folds),
        n_train=n_train,
        n_val=n_val,
        noise_std=noise,
        budget_scale=budget_scale,
        seed=seed,
        score_name=score_name,
        masks=masks,
    )
    if refit and math.isfinite(results[0].score):
        name = results[0].name
        seed_everything(seed)
        fld = _make_field(candidates[name])
        problem = build_problem(problem_factory, fld)
        cur = (curricula or {}).get(name, base_cur)
        cbs = list(callbacks(name)) if callbacks is not None else []
        report.refit = Solver(
            problem, cur, device=device, seed=seed, callbacks=cbs, **solver_kw
        ).run()
        report.refit_field = fld
    return report


# ------------------------------------------------------------------------------------------
# ensembles
# ------------------------------------------------------------------------------------------
def softmax_weights(
    scores: Mapping[str, float] | Sequence[float], temperature: float | None = None
) -> torch.Tensor:
    """``w_i ∝ exp(−(s_i − s_min) / T)`` (non-finite scores get weight 0).

    ``temperature=None`` uses the *relative* temperature ``T = s_min`` (a candidate with twice the
    best held-out loss gets ``e^{−1}`` of its weight), which makes the weights independent of the
    loss scale.
    """
    vals = list(scores.values()) if isinstance(scores, Mapping) else list(scores)
    s = torch.tensor([float(v) for v in vals], dtype=torch.float64)
    finite = torch.isfinite(s)
    if not bool(finite.any()):
        raise ConfigError("no finite scores to weight")
    smin = float(s[finite].min())
    t = float(temperature) if temperature is not None else max(abs(smin), 1e-300)
    if t <= 0:
        raise ConfigError("temperature must be positive")
    logits = torch.where(finite, -(s - smin) / t, torch.full_like(s, -math.inf))
    return torch.softmax(logits, dim=0)


def _key(name: str) -> str:
    k = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(name))
    return k if k and not k[0].isdigit() else f"m_{k}"


@register("field", "representation_ensemble")
class RepresentationEnsemble(Field):
    """Weighted average of fitted representations, weights from held-out fit.

    ``x = Σ_i w_i f_i(x)`` with ``w = softmax(−(s − s_min)/T)`` over held-out losses ``s`` (see
    :func:`softmax_weights`); :meth:`spread` gives the weighted standard deviation — a cheap
    model-uncertainty map (members that disagree where the data are uninformative, NeFTY App.
    B.4 / G.7).

    Args:
        fields: ``name -> fitted Field`` (same output names).
        scores: ``name -> held-out loss`` (used when ``weights`` is None).
        weights: explicit weights (normalized to sum 1).
        temperature: softmax temperature (``None`` = relative, ``T = s_min``).
        member_progress: progress at which each member is evaluated (default 1; members fitted
            with adaptive annealing may have stopped earlier).
    """

    def __init__(
        self,
        fields: Mapping[str, Field],
        scores: Mapping[str, float] | None = None,
        weights: Mapping[str, float] | Sequence[float] | None = None,
        temperature: float | None = None,
        member_progress: Mapping[str, float] | None = None,
    ) -> None:
        if not fields:
            raise ConfigError("RepresentationEnsemble needs at least one field")
        names = list(fields)
        outs = fields[names[0]].names
        for n in names[1:]:
            if tuple(fields[n].names) != tuple(outs):
                raise ConfigError(f"member {n!r} outputs {fields[n].names}, expected {outs}")
        super().__init__(Heads({o: "identity" for o in outs}))
        self.member_names = tuple(names)
        self._keys = {n: _key(n) for n in names}
        self.members = nn.ModuleDict({self._keys[n]: fields[n] for n in names})
        if weights is not None:
            w = (
                [float(weights[n]) for n in names]
                if isinstance(weights, Mapping)
                else [float(v) for v in weights]
            )
            wt = torch.tensor(w, dtype=torch.float64).clamp_min(0)
            if float(wt.sum()) <= 0:
                raise ConfigError("weights must have a positive sum")
            wt = wt / wt.sum()
        else:
            if scores is None:
                wt = torch.full((len(names),), 1.0 / len(names), dtype=torch.float64)
            else:
                wt = softmax_weights({n: scores[n] for n in names}, temperature)
        self.register_buffer("weights", wt.float())
        self.member_progress = {n: float((member_progress or {}).get(n, 1.0)) for n in names}

    @classmethod
    def from_report(
        cls,
        report: SelectionReport,
        top_k: int | None = None,
        temperature: float | None = None,
    ) -> RepresentationEnsemble:
        """Ensemble of the (top-k) fitted candidates of a :class:`SelectionReport`."""
        cands = [c for c in report.ranking if c.field is not None and math.isfinite(c.score)]
        if top_k is not None:
            cands = cands[: int(top_k)]
        if not cands:
            raise ConfigError("the report kept no fitted fields (use keep_fields=True)")
        return cls(
            {c.name: c.field for c in cands},  # type: ignore[misc]
            scores={c.name: c.score for c in cands},
            temperature=temperature,
            member_progress={c.name: c.final_progress for c in cands},
        )

    def weight_dict(self) -> dict[str, float]:
        return {n: float(w) for n, w in zip(self.member_names, self.weights)}

    def _member_stack(self, coords: torch.Tensor, progress: float) -> torch.Tensor:
        outs = []
        for n in self.member_names:
            f = self.members[self._keys[n]]
            o = f(coords, min(float(progress), self.member_progress[n]))
            outs.append(torch.stack([o[k] for k in self.names], dim=-1))
        return torch.stack(outs, dim=0)  # (M, *shape, C)

    def raw(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        st = self._member_stack(coords, progress)
        w = self.weights.to(st).view(-1, *([1] * (st.ndim - 1)))
        return (w * st).sum(0)

    @torch.no_grad()
    def spread(self, coords: torch.Tensor, progress: float = 1.0) -> dict[str, torch.Tensor]:
        """Weighted standard deviation across members per output field."""
        st = self._member_stack(coords, progress)
        w = self.weights.to(st).view(-1, *([1] * (st.ndim - 1)))
        mean = (w * st).sum(0)
        var = (w * (st - mean) ** 2).sum(0)
        return {k: var[..., i].sqrt() for i, k in enumerate(self.names)}

    def on_stage_start(self, stage, domain) -> None:
        for m in self.members.values():
            m.on_stage_start(stage, domain)

    def reset_parameters(self) -> None:
        for m in self.members.values():
            m.reset_parameters()

    def extra_repr(self) -> str:
        w = ", ".join(f"{n}={v:.3f}" for n, v in self.weight_dict().items())
        return f"weights=({w})"


__all__ = [
    "CandidateScore",
    "RepresentationEnsemble",
    "SelectionReport",
    "build_problem",
    "holdout_masks",
    "masked_downsampler",
    "select_representation",
    "softmax_weights",
]
