"""Diagnostic reports: the data-fit paradox check and a one-call :func:`diagnose` bundle.

The *data-fit paradox* (NeFTY §5.2, App. G.2): a method can fit the measurement almost perfectly
and still recover the wrong field — the soft-constrained PINN reaches a surface PSNR of ~63 dB at a
volumetric IoU of 0.01, while NeFTY reaches ~82 dB at IoU 0.45. Always report measurement-space fit
*next to* field-space metrics; :func:`data_fit_paradox` does exactly that.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..errors import ConfigError
from ._common import resolve_fields, tensor_summary
from .filtering import (
    RealizedUpdate,
    effective_bandwidth,
    filter_kernel_row,
    kernel_spread,
    realized_update,
)
from .landscape import (
    EnergyBarrier,
    Iter0Gradient,
    center_mass_ratio,
    energy_barrier,
    uniform_center_mass,
)
from .landscape import iter0_gradient as _iter0_gradient
from .sensitivity import center_to_outer_ratio, sensitivity_map
from .spectrum import singular_values as _singular_values

log = logging.getLogger("nefi")

DIAGNOSTICS = (
    "sensitivity",
    "iter0",
    "realized",
    "filter",
    "bandwidth",
    "spectrum",
    "barrier",
    "data_fit",
)


def _resolve_metrics(field_metrics: Any) -> dict[str, Callable]:
    from ..metrics.basic import BASIC_METRICS
    from ..registry import get

    if field_metrics is None:
        return dict(BASIC_METRICS)
    if isinstance(field_metrics, Mapping):
        return dict(field_metrics)
    return {name: get("metric", name) for name in field_metrics}


def data_fit_paradox(
    result: Any,
    problem: Any,
    gt: Mapping[str, torch.Tensor] | torch.Tensor | None = None,
    field_metrics: Mapping[str, Callable] | Sequence[str] | None = None,
    *,
    field: str | None = None,
) -> dict[str, float]:
    """Measurement-space fit next to field-space accuracy (NeFTY §5.2, App. G.2, Tab. 7).

    Args:
        result: a :class:`~nefi.solve.result.Result` (uses ``result.pred`` and ``result.fields``).
        problem: the inverse problem (uses ``problem.measurement``).
        gt: ground-truth fields (mapping or primary-field tensor); field metrics are skipped if
            ``None``.
        field_metrics: ``{name: fn(pred, gt)}`` or registered metric names (default: MSE, PSNR,
            relative error).
        field: field to score (default: the problem's primary field).

    Returns:
        ``data_mse``, ``data_rmse``, ``data_psnr`` (dB, range of the observation),
        ``data_rel_residual`` and, when the noise level is known, ``noise_std`` and
        ``discrepancy_ratio = data_rmse / noise_std`` (≈1 is the Morozov target; ≪1 means the
        noise is being fitted); plus ``field_<metric>`` for every field metric and
        ``psnr_gap = data_psnr − field_psnr`` when both exist.
    """
    pred = result.pred.detach().double().cpu()
    meas = problem.measurement
    if tuple(meas.data.shape) != tuple(pred.shape):
        meas = meas.resampled(pred.shape)
    meas = meas.to("cpu", torch.float64)
    r2 = (pred - meas.data) ** 2
    mse = float(meas.masked_mean(r2))
    obs = meas.data if meas.mask is None else meas.data[meas.mask.expand_as(meas.data) > 0]
    rng = float(obs.max() - obs.min()) if obs.numel() else 0.0
    den = float(meas.masked_mean(meas.data**2))
    out: dict[str, float] = {
        "data_mse": mse,
        "data_rmse": math.sqrt(mse),
        "data_psnr": (10.0 * math.log10(rng**2 / mse)) if mse > 0 and rng > 0 else float("inf"),
        "data_rel_residual": math.sqrt(mse / den) if den > 0 else float("nan"),
    }
    ns = meas.noise_std
    if ns is not None:
        ns = float(torch.as_tensor(ns).double().mean())
        if ns > 0:
            out["noise_std"] = ns
            out["discrepancy_ratio"] = out["data_rmse"] / ns
    if gt is not None:
        name = field or problem.field.primary
        g = gt if torch.is_tensor(gt) else gt[name]
        p = result.fields[name]
        g = torch.as_tensor(g).detach().to(p.dtype).cpu()
        if tuple(g.shape) != tuple(p.shape):
            from ..utils.tensor import resample

            g = resample(g, p.shape)
        for mname, fn in _resolve_metrics(field_metrics).items():
            try:
                out[f"field_{mname}"] = float(fn(p, g))
            except Exception as e:  # metric not applicable to this field
                log.debug("field metric %s failed: %s", mname, e)
        if "field_psnr" in out and math.isfinite(out["data_psnr"]):
            out["psnr_gap"] = out["data_psnr"] - out["field_psnr"]
    return out


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        if x != 0 and (abs(x) >= 1e4 or abs(x) < 1e-3):
            return f"{x:.{digits}e}"
        return f"{x:.{digits}g}" if abs(x) >= 1 else f"{x:.{digits}f}"
    return str(x)


@dataclass
class DiagnosticsReport:
    """Bundle of ill-posedness / optimization-geometry diagnostics for one problem.

    Every entry is optional: a diagnostic that was skipped or failed is ``None`` (failures are
    recorded in ``errors``). Use :meth:`to_markdown` for a human-readable summary with the paper
    reference numbers, :meth:`to_dict` for JSON and :meth:`save` to write both.
    """

    name: str
    field: str
    shape: tuple[int, ...]
    sensitivity: torch.Tensor | None = None
    sensitivity_ratio: float | None = None
    iter0: Iter0Gradient | None = None
    realized: RealizedUpdate | None = None
    filter_rows: dict[str, torch.Tensor] = field(default_factory=dict)
    filter_spread: dict[str, float] = field(default_factory=dict)
    bandwidth: dict[str, float | None] = field(default_factory=dict)
    singular_values: torch.Tensor | None = None
    energy_barrier: EnergyBarrier | None = None
    data_fit: dict[str, float] | None = None
    center_mass: float | None = None
    timings: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Flat dict of the scalar findings."""
        s: dict[str, Any] = {"name": self.name, "field": self.field, "shape": list(self.shape)}
        if self.sensitivity is not None:
            sm = self.sensitivity
            s["sensitivity_ratio"] = self.sensitivity_ratio
            s["sensitivity_min"] = float(sm.min())
            s["sensitivity_max"] = float(sm.max())
        if self.iter0 is not None:
            s.update({f"iter0_{k}": v for k, v in self.iter0.to_dict().items()})
        if self.realized is not None:
            s.update({f"realized_{k}": v for k, v in self.realized.to_dict().items()})
        for k, v in self.filter_spread.items():
            s[f"filter_spread[{k}]"] = v
        for k, v in self.bandwidth.items():
            s[f"bandwidth[{k}]"] = v
        if self.singular_values is not None and len(self.singular_values):
            sv = self.singular_values
            s["singular_values"] = [float(x) for x in sv]
            s["sv_decay"] = float(sv[-1] / sv[0]) if float(sv[0]) > 0 else float("nan")
        if self.energy_barrier is not None:
            eb = self.energy_barrier
            s.update(
                {
                    "barrier_height": eb.height,
                    "barrier_t": eb.t_max,
                    "barrier_monotone": eb.monotone,
                }
            )
        if self.data_fit is not None:
            s.update(self.data_fit)
        if self.center_mass is not None:
            s["center_mass"] = self.center_mass
        return s

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict (scalars, small lists, timings, errors)."""
        d = {"summary": self.summary(), "timings": self.timings, "errors": self.errors}
        if self.energy_barrier is not None:
            d["energy_barrier"] = self.energy_barrier.to_dict()
        if self.sensitivity is not None:
            d["sensitivity_stats"] = tensor_summary(self.sensitivity)
        d["meta"] = {k: v for k, v in self.meta.items() if isinstance(v, str | int | float | bool)}
        return d

    # ----------------------------------------------------------------------------------
    def _rows(self) -> list[tuple[str, str, str, str]]:
        rows: list[tuple[str, str, str, str]] = []
        if self.sensitivity is not None:
            r = self.sensitivity_ratio or float("nan")
            sm = self.sensitivity
            dyn = float(sm.max() / sm.min()) if float(sm.min()) > 0 else float("inf")
            if r > 2:
                reading = "strong window-center bias: free-pixel solvers get a centered descent"
            elif r > 1.2:
                reading = "mild center bias"
            else:
                reading = "no significant center bias"
            rows.append(
                (
                    "Sensitivity center/outer ratio ‖∂F/∂x_i‖",
                    f"{_fmt(r)}× (max/min {_fmt(dyn)})",
                    "NeTMY (P2), Eq. 23",
                    reading + "; low-sensitivity pixels are weakly constrained (low trust)",
                )
            )
        if self.iter0 is not None:
            it = self.iter0
            centered = it.ratio > 3 and it.peak_radius < 0.25
            rows.append(
                (
                    f"Iter-0 {it.init} data-gradient center/outer ratio",
                    f"{_fmt(it.ratio)}× (peak at r={_fmt(it.peak_radius)}, "
                    f"center mass {_fmt(it.center_mass)})",
                    "NeTMY §5.3: 18.29× under F2, peak at the grid center",
                    "centered iter-0 gradient: expect centered collapse for grid/ADMM/Tikhonov"
                    if centered
                    else "no centered-collapse signature",
                )
            )
        if self.realized is not None:
            ru = self.realized
            # paper-style: iter-0 data-gradient ratio / realized-update ratio (18.29 / 1.6)
            damping_vs_iter0 = float("nan")
            paper_damping = ""
            if self.iter0 is not None and ru.delta_ratio and math.isfinite(ru.delta_ratio):
                damping_vs_iter0 = self.iter0.ratio / ru.delta_ratio
                paper_damping = f"; vs iter-0 data gradient: {_fmt(damping_vs_iter0)}× damping"
            rows.append(
                (
                    f"Realized first update |Δx| vs |∇ₓL| "
                    f"({ru.optimizer}, β/K={_fmt(ru.progress)})",
                    f"{_fmt(ru.delta_ratio)}× vs {_fmt(ru.grad_ratio)}× "
                    f"(damping {_fmt(ru.damping)}, cos {_fmt(ru.alignment)})" + paper_damping,
                    "NeTMY App. E.5: 1.6× vs 18.29× (≈11× damping)",
                    "parameterization redistributes the raw gradient (Lemma 2)"
                    if max(ru.damping, damping_vs_iter0) > 1.5
                    else "update follows the raw gradient closely",
                )
            )
        if self.filter_spread:
            val = ", ".join(f"{k}: {_fmt(v)} px" for k, v in self.filter_spread.items())
            smooth = max(self.filter_spread.values()) > 1.0
            rows.append(
                (
                    "Filter-kernel main-lobe width G_θ e_i (half max)",
                    val,
                    "NeTMY Lemma 2 / Fig. 10 (grid: 1 px, G_θ = I)",
                    "smoothing, spatially coupled kernel" if smooth else "no smoothing (delta)",
                )
            )
        if self.bandwidth and any(v is not None for v in self.bandwidth.values()):
            val = ", ".join(f"{k}: {_fmt(v)}" for k, v in self.bandwidth.items())
            rows.append(
                (
                    "Effective encoding bandwidth B_β",
                    val,
                    "NeTMY Eq. 37 (B_β ~ 2^k_β π)",
                    "annealing opens high frequencies progressively",
                )
            )
        if self.singular_values is not None and len(self.singular_values):
            sv = self.singular_values
            decay = float(sv[-1] / sv[0]) if float(sv[0]) > 0 else float("nan")
            head = ", ".join(_fmt(float(x)) for x in sv[:6])
            if decay < 1e-3:
                reading = "severe decay: high-order modes are invisible to the data"
            elif decay < 0.1:
                reading = "clear decay: regularization / prior needed for fine detail"
            else:
                reading = "flat over the top-k modes (increase k to see the decay)"
            rows.append(
                (
                    f"Top-{len(sv)} singular values of dF/dx",
                    f"{head}{', …' if len(sv) > 6 else ''} (σ_k/σ_1 = {_fmt(decay)})",
                    "NeFTY Prop. 2: σ_n ≲ n^(-1/3); NeTMY Lemma 1: e^(-k z0)",
                    reading,
                )
            )
        if self.energy_barrier is not None:
            eb = self.energy_barrier
            rows.append(
                (
                    "Energy barrier along x(t) (current → GT)",
                    f"h = {_fmt(eb.height)} at t = {_fmt(eb.t_max)}"
                    + (" (monotone)" if eb.monotone else ""),
                    "NeTMY Fig. 4b: h ≈ 1.12 at t = 0.20 (F2); monotone (F1)",
                    "no barrier along the straight path"
                    if eb.monotone or eb.height == 0
                    else "descent must climb a barrier to reach the ground truth",
                )
            )
        if self.center_mass is not None:
            base = uniform_center_mass(self.shape)
            rel = self.center_mass / base if base > 0 else float("nan")
            if rel > 3:
                reading = "centered collapse"
            elif rel > 1.5:
                reading = "center-heavy solution"
            else:
                reading = "no centered collapse"
            rows.append(
                (
                    "Center-mass ratio of the solution",
                    f"{_fmt(self.center_mass)} (uniform field: {_fmt(base)})",
                    "NeTMY Fig. 4c (2-D, uniform 0.071): 0.00 NeTMY … 0.223 Tikhonov",
                    reading,
                )
            )
        if self.data_fit is not None:
            df = self.data_fit
            val = f"data PSNR {_fmt(df.get('data_psnr'))} dB"
            if "field_psnr" in df:
                val += f" vs field PSNR {_fmt(df['field_psnr'])} dB"
            if "discrepancy_ratio" in df:
                val += f"; RMSE/σ = {_fmt(df['discrepancy_ratio'])}"
            reading = "report field metrics next to data fit"
            if df.get("psnr_gap", 0) > 20:
                reading = "large data/field gap: good data fit does not imply a correct field"
            if df.get("discrepancy_ratio", 1.0) < 0.5:
                reading += "; fitting below the noise level (try Curriculum.discrepancy_tau)"
            rows.append(
                ("Data-fit paradox", val, "NeFTY App. G.2: PINN 63 dB at IoU 0.01", reading)
            )
        return rows

    def to_markdown(self) -> str:
        """Markdown report: findings table with the paper reference numbers and readings."""
        lines = [
            f"# Diagnostics — {self.name}",
            "",
            f"Field `{self.field}` on grid {tuple(self.shape)}"
            + (f"; {self.meta['point']}" if "point" in self.meta else "")
            + ".",
            "",
            "| Diagnostic | Value | Paper reference | Reading |",
            "|---|---|---|---|",
        ]
        for a, b, c, d in self._rows():
            lines.append(f"| {a} | {b} | {c} | {d} |")
        if self.timings:
            t = ", ".join(f"{k} {v:.2f}s" for k, v in self.timings.items())
            lines += ["", f"Timings: {t}."]
        if self.errors:
            lines += ["", "Skipped / failed:"]
            lines += [f"- `{k}`: {v}" for k, v in self.errors.items()]
        return "\n".join(lines) + "\n"

    def save(self, directory: str | Path) -> dict[str, Path]:
        """Write ``diagnostics.md``, ``diagnostics.json`` and ``diagnostics_tensors.pt``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        paths = {"markdown": d / "diagnostics.md", "json": d / "diagnostics.json"}
        paths["markdown"].write_text(self.to_markdown())
        paths["json"].write_text(json.dumps(self.to_dict(), indent=2, default=_json_default))
        tensors = {"filter_rows": self.filter_rows}
        if self.sensitivity is not None:
            tensors["sensitivity"] = self.sensitivity.cpu()
        if self.singular_values is not None:
            tensors["singular_values"] = self.singular_values.cpu()
        if self.iter0 is not None:
            tensors["iter0_grad"] = self.iter0.grad.cpu()
        if self.realized is not None:
            tensors["realized_delta"] = self.realized.delta.cpu()
        paths["tensors"] = d / "diagnostics_tensors.pt"
        torch.save(tensors, paths["tensors"])
        return paths


def _json_default(o: Any) -> Any:
    if torch.is_tensor(o):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def diagnose(
    problem: Any,
    gt: Mapping[str, torch.Tensor] | torch.Tensor | None = None,
    result: Any = None,
    *,
    shape: Sequence[int] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
    field: str | None = None,
    n_probes: int = 32,
    k: int = 8,
    n_iter: int = 24,
    seed: int = 0,
    progress: float = 1.0,
    exact_max: int = 1024,
) -> DiagnosticsReport:
    """Run the cheap diagnostics on a problem and bundle them into a :class:`DiagnosticsReport`.

    The operator is linearized at the solution (``result.fields``) when a result is given and at
    the current field values otherwise. Each diagnostic is independent and failures are recorded
    in ``report.errors`` instead of raised, so ``diagnose`` works on any operator.

    Args:
        problem: the inverse problem.
        gt: ground-truth fields (enables the energy barrier current → GT and field metrics).
        result: a solver :class:`~nefi.solve.result.Result` (enables the data-fit paradox check and
            the solution's center-mass ratio).
        shape: grid shape (default: native / the result's shape).
        include: subset of ``DIAGNOSTICS`` (default: all applicable).
        exclude: diagnostics to skip.
        field: field to analyze (default primary).
        n_probes: probes of the sensitivity estimator.
        k: number of singular values.
        n_iter: Lanczos iterations.
        seed: probe / start-vector seed.
        progress: annealing progress of the linearization point when no result is given.
        exact_max: compute the sensitivity exactly (one JVP per pixel) up to this many pixels.
    """
    todo = [d for d in (include or DIAGNOSTICS) if d not in set(exclude)]
    unknown = set(todo) - set(DIAGNOSTICS)
    if unknown:
        raise ConfigError(f"unknown diagnostics {sorted(unknown)}; known: {DIAGNOSTICS}")
    name = field or problem.field.primary
    point = result.fields if result is not None else None
    x_fields, dom = resolve_fields(problem, point, shape, progress)
    report = DiagnosticsReport(
        name=getattr(problem, "name", "problem"),
        field=name,
        shape=tuple(dom.shape),
        meta={"point": "linearized at the solution" if result is not None else "at initialization"},
    )

    def run(key: str, fn: Callable[[], None]) -> None:
        if key not in todo:
            return
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as e:  # keep going: every diagnostic is optional
            report.errors[key] = f"{type(e).__name__}: {e}"
            log.warning("diagnostic %s failed: %s", key, e)
        report.timings[key] = time.perf_counter() - t0

    def _sens() -> None:
        exact = x_fields[name].numel() <= exact_max
        s = sensitivity_map(problem, name, dom.shape, n_probes, exact, fields=x_fields, seed=seed)
        report.sensitivity = s
        report.sensitivity_ratio = center_to_outer_ratio(s)

    def _iter0() -> None:
        report.iter0 = _iter0_gradient(problem, "uniform", shape=dom.shape, field=name)

    def _realized() -> None:
        report.realized = realized_update(problem, field=name)

    def _filter() -> None:
        for p in (0.0, 1.0):
            row = filter_kernel_row(problem, "center", dom.shape, field=name, progress=p)
            key = f"β/K={p:g}"
            report.filter_rows[key] = row
            report.filter_spread[key] = kernel_spread(row, pixel="center")

    def _bandwidth() -> None:
        for p in (0.0, 0.5, 1.0):
            report.bandwidth[f"β/K={p:g}"] = effective_bandwidth(problem.field, p)

    def _spectrum() -> None:
        report.singular_values = _singular_values(
            problem, k, n_iter, dom.shape, field=name, fields=x_fields, seed=seed
        )

    def _barrier() -> None:
        if gt is None:
            raise ConfigError("needs ground truth (gt=...)")
        report.energy_barrier = energy_barrier(problem, x_fields, gt, shape=dom.shape)

    def _data_fit() -> None:
        if result is None:
            raise ConfigError("needs a solver result (result=...)")
        report.data_fit = data_fit_paradox(result, problem, gt, field=name)
        report.center_mass = center_mass_ratio(result.fields[name])

    for key, fn in (
        ("sensitivity", _sens),
        ("iter0", _iter0),
        ("realized", _realized),
        ("filter", _filter),
        ("bandwidth", _bandwidth),
        ("spectrum", _spectrum),
        ("barrier", _barrier),
        ("data_fit", _data_fit),
    ):
        if key == "barrier" and gt is None and include is None:
            continue
        if key == "data_fit" and result is None and include is None:
            continue
        run(key, fn)
    return report


__all__ = ["DIAGNOSTICS", "DiagnosticsReport", "data_fit_paradox", "diagnose"]
