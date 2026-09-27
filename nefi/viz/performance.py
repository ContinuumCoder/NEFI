"""Performance dashboards: per-step time, scaling with grid size, memory by gradient mode,
wall-clock breakdowns and benchmark tables with confidence intervals.

Every plot takes **rows** — a list of plain dicts — so the same functions read
``tools/profile_instance.py``-style measurements (:func:`collect_performance`), the raw rows of a
:class:`~nefi.bench.BenchmarkResult` (``benchmark.json`` / ``rows.csv``), a
:class:`~nefi.bench.report.RuntimeTable`, or a gallery ``manifest.json`` (:func:`load_rows`).

Row keys (all optional except the ones a plot needs):

=================  =============================================================================
key                meaning
=================  =============================================================================
``instance``       instance / problem name (``name`` is accepted as an alias)
``method``         method (benchmarks) — ``variant`` / ``mode``: e.g. ``adjoint`` / ``autograd``
``device``         ``cpu`` / ``cuda`` / ``mps``
``class``          scene class (benchmarks)
``stage``          curriculum stage profiled
``shape``          field grid (list or ``"64x64"``); ``n`` = number of grid points (derived)
``params``         trainable parameters
``steps``          optimization steps timed (or run)
``ms_per_step``    wall-clock per optimization step (ms)
``fwd_ms``         forward (field + operator + losses) per step (ms)
``bwd_ms``         backward per step (ms); ``opt_ms`` optimizer update per step (ms)
``time_s``         total wall-clock of a run (s); ``per_stage_s`` list of stage seconds
``peak_mem_mb``    peak allocated device memory (CUDA only)
``saved_mb``       autograd saved-tensor memory of one step (device-agnostic proxy, CPU too)
<metric>           any metric column (``psnr``, ``ssim``, ...) — :func:`plot_bench_table`
``error``          failure message (rows with an error are skipped by the plots)
=================  =============================================================================

Profiles written by ``tools/profile_instance.py --json runs/_perf/<label>.jsonl`` (one JSON line
per run: ``instance``, ``device``, ``threads``, ``loop``, ``compile``, ``autocast``, ``overrides``,
``ms_median``, ...) are read by :func:`load_profiles`; :func:`profile_speedups` compares every
accelerated variant with the latest plain ("eager") run of the same instance, device and thread
count and :func:`plot_speedup` draws the per-instance "speedup vs eager" bars.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .style import (
    grid_on,
    message_axes,
    new_figure,
    panel_size,
    styled,
    theme,
)

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

log = logging.getLogger("nefi")


# ---------------------------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------------------------
def _num(v: Any) -> float:
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _parse_cell(v: Any) -> Any:
    """CSV cell → ``None`` (empty), number, JSON list or the original string."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def _shape_of(v: Any) -> list[int] | None:
    if v is None or v == "":
        return None
    if isinstance(v, str):
        txt = v.strip().strip("[]()")
        for sep in ("x", "×", ",", " "):
            if sep in txt:
                try:
                    return [int(float(p)) for p in txt.replace(" ", sep).split(sep) if p.strip()]
                except ValueError:
                    return None
        try:
            return [int(float(txt))]
        except ValueError:
            return None
    try:
        return [int(s) for s in v]
    except TypeError:
        return None


def shape_label(shape: Any) -> str:
    """``"64×64×16"`` from a shape list / string."""
    s = _shape_of(shape)
    return "×".join(str(v) for v in s) if s else "?"


def normalize_row(r: Mapping[str, Any]) -> dict[str, Any]:
    """Harmonize aliases and derive ``n`` / ``ms_per_step`` / ``fwd_ms`` where possible."""
    row = dict(r)
    if "instance" not in row and "name" in row:
        row["instance"] = row["name"]
    if "mode" not in row and "variant" in row:
        row["mode"] = row["variant"]
    for src, dst in (("fwd_s_mean", "fwd_ms"), ("bwd_s_mean", "bwd_ms")):
        if dst not in row and src in row:
            row[dst] = 1e3 * _num(row[src])
    shp = _shape_of(row.get("shape"))
    if shp is not None:
        row["shape"] = shp
        if "n" not in row or not math.isfinite(_num(row.get("n"))):
            row["n"] = int(np.prod(shp))
    if "ms_per_step" not in row or not math.isfinite(_num(row.get("ms_per_step"))):
        ts, st = _num(row.get("time_s")), _num(row.get("steps"))
        if math.isfinite(ts) and math.isfinite(st) and st > 0:
            row["ms_per_step"] = 1e3 * ts / st
    if isinstance(row.get("per_stage_s"), str):
        try:
            row["per_stage_s"] = json.loads(row["per_stage_s"])
        except json.JSONDecodeError:
            pass
    return row


def _manifest_rows(d: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for e in d.get("entries", []):
        t = dict(e.get("timing") or {})
        row = {
            "instance": e.get("name"),
            "device": e.get("device") or d.get("device"),
            "steps": e.get("steps"),
            "ms_per_step": e.get("ms_per_step"),
            "time_s": t.get("solve_s"),
            "per_stage_s": t.get("per_stage_s"),
            "generate_s": t.get("generate_s"),
            "evaluate_s": t.get("evaluate_s"),
            "plots_s": t.get("plots_s"),
            "total_s": t.get("total_s"),
            "params": e.get("n_parameters"),
            "shape": next(iter((e.get("fields") or {}).values()), None),
            "error": e.get("error"),
            **{k: v for k, v in (e.get("metrics") or {}).items()},
        }
        out.append(row)
    return out


def load_rows(source: Any) -> list[dict[str, Any]]:
    """Rows from a list of dicts, a ``BenchmarkResult`` / ``RuntimeTable`` (``.rows``), a dict
    with ``rows`` (``benchmark.json``) or ``entries`` (gallery ``manifest.json``), or a path to a
    ``.json`` / ``.csv`` file or a directory containing one of them."""
    if source is None:
        return []
    if isinstance(source, str | Path):
        p = Path(source)
        if p.is_dir():
            for cand in ("benchmark.json", "manifest.json", "performance.json", "rows.csv"):
                if (p / cand).exists():
                    return load_rows(p / cand)
            raise FileNotFoundError(f"no benchmark.json / manifest.json / rows.csv in {p}")
        if p.suffix.lower() == ".csv":
            with p.open(newline="") as f:
                raw = list(csv.DictReader(f))
            return [normalize_row({k: _parse_cell(v) for k, v in r.items()}) for r in raw]
        return load_rows(json.loads(p.read_text()))
    if hasattr(source, "rows") and not isinstance(source, Mapping):
        inst = getattr(source, "instance", None)
        rows = [dict(r) for r in source.rows]
        if isinstance(inst, str):
            for r in rows:
                r.setdefault("instance", inst)
        return [normalize_row(r) for r in rows]
    if isinstance(source, Mapping):
        if "entries" in source:
            return [normalize_row(r) for r in _manifest_rows(source)]
        if "rows" in source:
            inst = source.get("instance")
            rows = [dict(r) for r in source["rows"]]
            for r in rows:
                if inst is not None:
                    r.setdefault("instance", inst)
            return [normalize_row(r) for r in rows]
        return [normalize_row(source)]
    return [normalize_row(r) for r in source]


def _ok(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows if not r.get("error")]


def _cats(rows: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    out: list[str] = []
    for r in rows:
        v = str(r.get(key, "—"))
        if v not in out:
            out.append(v)
    return out


def _fold(cats: list[str], limit: int = 8) -> list[str]:
    if len(cats) > limit:
        log.warning(
            "%d series exceed the 8 categorical slots; showing the first %d", len(cats), limit
        )
    return cats[:limit]


# ---------------------------------------------------------------------------------------------
# step time
# ---------------------------------------------------------------------------------------------
def plot_step_time(
    rows: Any,
    *,
    value: str = "ms_per_step",
    by: str = "instance",
    hue: str = "device",
    split: bool = True,
    log: bool | str = "auto",
    shapes: bool = True,
    title: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """Per-step time per instance (horizontal bars, value at every bar tip).

    With a single ``hue`` value and ``fwd_ms`` / ``bwd_ms`` columns (``split=True``) a second
    panel shows the forward | backward | optimizer **shares** (100 % stacked, linear), so the
    total can use a log axis without distorting the segments; otherwise bars are grouped by
    ``hue`` (e.g. device). ``shapes`` appends the profiled grid to every label (``ct3d ·
    16×16×16``), so volumetric instances stand out.
    """
    data = [r for r in _ok(load_rows(rows)) if math.isfinite(_num(r.get(value)))]
    if shapes and by == "instance":
        for r in data:
            shp = _shape_of(r.get("shape"))
            if shp:
                r["_label"] = f"{r.get(by)} · {shape_label(shp)}"
        if data and all("_label" in r for r in data):
            by = "_label"
    with styled(dark):
        t = theme()
        w, h = panel_size()
        cats = _cats(data, by)
        hues = _fold(_cats(data, hue))
        n = max(1, len(cats))
        do_split = (
            bool(data)
            and split
            and len(hues) == 1
            and all(math.isfinite(_num(r.get("fwd_ms"))) for r in data)
            and all(math.isfinite(_num(r.get("bwd_ms"))) for r in data)
        )
        ncols = 2 if do_split else 1
        fig, axes = new_figure(
            1,
            ncols,
            figsize=(
                w * 2.4 * ncols + 1.2,
                0.9 + 0.28 * n * (1 if do_split else max(1, len(hues))),
            ),
            sharey=True,
            width_ratios=[1.6, 1.0] if do_split else None,
        )
        ax = axes[0, 0]
        if not data:
            message_axes(ax, f"no rows with '{value}'", title or "step time")
            return fig
        ys = np.arange(len(cats))[::-1].astype(float)
        vals_all: list[float] = []
        k = 1 if do_split else len(hues)
        hgt = 0.7 / k
        for j, hv in enumerate([None] if do_split else hues):
            for yi, c in zip(ys, cats):
                sel = [
                    r for r in data if str(r.get(by)) == c and (hv is None or str(r.get(hue)) == hv)
                ]
                if not sel:
                    continue
                v = float(np.mean([_num(r.get(value)) for r in sel]))
                y = yi + (k - 1 - 2 * j) * hgt / 2
                ax.barh(
                    y,
                    v,
                    height=hgt * (0.8 if do_split else 0.9),
                    color=t.palette[j],
                    label=hv if (hv is not None and c == cats[0]) else None,
                )
                ax.text(v, y, f" {v:.3g}", va="center", fontsize="x-small", color=t.ink2)
                vals_all.append(v)
        if k > 1:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.set_yticks(ys, cats)
        pos = [v for v in vals_all if v > 0]
        use_log = (len(pos) > 1 and max(pos) / min(pos) > 30) if log == "auto" else bool(log)
        if use_log:
            ax.set_xscale("log")
        ax.margins(x=0.2)
        grid_on(ax, "x")
        ax.set_xlabel("ms per optimization step" if value == "ms_per_step" else value)
        dev = ", ".join(_cats(data, "device")) if any("device" in r for r in data) else ""
        ax.set_title(title or f"step time{f' ({dev})' if dev else ''}")
        if do_split:
            axs = axes[0, 1]
            names = ("forward", "backward", "optimizer / other")
            colors = (t.palette[0], t.palette[1], t.muted)
            for yi, c in zip(ys, cats):
                r = next(r for r in data if str(r.get(by)) == c)
                tot = _num(r.get(value))
                fwd, bwd = _num(r.get("fwd_ms")), _num(r.get("bwd_ms"))
                parts = np.array([fwd, bwd, max(0.0, tot - fwd - bwd)])
                parts = 100.0 * parts / max(float(parts.sum()), 1e-12)
                left = 0.0
                for nm, v, col in zip(names, parts, colors):
                    axs.barh(
                        yi,
                        v,
                        left=left,
                        height=0.56,
                        color=col,
                        edgecolor=t.surface,
                        linewidth=1.0,
                        label=nm if c == cats[0] else None,
                    )
                    left += v
            axs.set_xlim(0, 100)
            axs.set_xlabel("share of the step (%)")
            grid_on(axs, "x")
            axs.set_title("forward / backward / optimizer")
            axs.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=3)
        return fig


# ---------------------------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------------------------
def _memory_key(rows: Sequence[Mapping[str, Any]]) -> str:
    if any(math.isfinite(_num(r.get("peak_mem_mb"))) for r in rows):
        return "peak_mem_mb"
    return "saved_mb"


_Y_LABEL = {
    "ms_per_step": "ms per step",
    "peak_mem_mb": "peak memory (MB)",
    "saved_mb": "autograd saved tensors (MB)",
    "time_s": "wall-clock (s)",
    "fwd_ms": "forward (ms)",
    "bwd_ms": "backward (ms)",
}


def loglog_slope(x: Sequence[float], y: Sequence[float]) -> float:
    """Least-squares slope of ``log y`` vs ``log x`` (nan with < 2 positive points)."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(xa) & np.isfinite(ya) & (xa > 0) & (ya > 0)
    if ok.sum() < 2 or np.unique(xa[ok]).size < 2:
        return float("nan")
    return float(np.polyfit(np.log(xa[ok]), np.log(ya[ok]), 1)[0])


def plot_scaling(
    rows: Any,
    *,
    x: str = "n",
    y: Sequence[str] = ("ms_per_step", "memory"),
    by: str = "instance",
    fit: bool = True,
    reference: bool = True,
    title: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """Step time and memory vs grid size, log-log, with fitted slopes (``O(N^slope)``).

    ``"memory"`` in ``y`` resolves to ``peak_mem_mb`` (CUDA) or ``saved_mb`` (any device).
    ``reference`` draws a muted slope-1 guide (linear scaling).
    """
    data = _ok(load_rows(rows))
    keys = [(_memory_key(data) if k == "memory" else k) for k in y]
    keys = [k for k in keys if any(math.isfinite(_num(r.get(k))) for r in data)]
    with styled(dark):
        t = theme()
        w, h = panel_size()
        ncols = max(1, len(keys))
        fig, axes = new_figure(1, ncols, figsize=(ncols * w * 1.7 + 0.9, h * 1.25 + 0.3))
        if not keys:
            message_axes(axes[0, 0], "no scaling rows", title or "scaling")
            return fig
        groups = _fold(_cats(data, by))
        for p, k in enumerate(keys):
            ax = axes[0, p]
            for j, gname in enumerate(groups):
                sel = sorted(
                    ((_num(r.get(x)), _num(r.get(k))) for r in data if str(r.get(by)) == gname),
                    key=lambda v: v[0],
                )
                sel = [
                    (a, b)
                    for a, b in sel
                    if math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0
                ]
                if not sel:
                    continue
                xs, ys = zip(*sel)
                slope = loglog_slope(xs, ys) if fit else float("nan")
                lab = f"{gname} (slope {slope:.2f})" if math.isfinite(slope) else gname
                ax.plot(xs, ys, color=t.palette[j], lw=1.4, marker="o", ms=4, label=lab)
            if reference:
                pts = [
                    (_num(r.get(x)), _num(r.get(k)))
                    for r in data
                    if _num(r.get(x)) > 0 and _num(r.get(k)) > 0
                ]
                if len(pts) >= 2:
                    x0 = min(v[0] for v in pts)
                    x1 = max(v[0] for v in pts)
                    y0 = min(v[1] for v in pts if v[0] == x0)
                    ax.plot([x0, x1], [y0, y0 * x1 / x0], color=t.muted, lw=0.8, label="O(N) guide")
            ax.set_xscale("log")
            ax.set_yscale("log")
            grid_on(ax, "both")
            ax.set_xlabel("grid points N" if x == "n" else x)
            ax.set_ylabel(_Y_LABEL.get(k, k))
            ax.legend(loc="upper left")
        dev = ", ".join(_cats(data, "device")) if any("device" in r for r in data) else ""
        fig.suptitle(title or f"scaling with grid size{f' ({dev})' if dev else ''}")
        return fig


def plot_memory(
    rows: Any,
    *,
    by: str = "shape",
    hue: str = "mode",
    value: str = "auto",
    title: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """Memory per gradient mode (e.g. adjoint vs autograd vs checkpoint) and grid size.

    ``value="auto"`` uses ``peak_mem_mb`` (CUDA) when available, else the device-agnostic
    ``saved_mb`` (autograd saved-tensor bytes of one step), else ``ms_per_step``.
    """
    data = _ok(load_rows(rows))
    if value == "auto":
        value = _memory_key(data)
        if not any(math.isfinite(_num(r.get(value))) for r in data):
            value = "ms_per_step"
    data = [r for r in data if math.isfinite(_num(r.get(value)))]
    with styled(dark):
        t = theme()
        w, h = panel_size()
        for r in data:
            r["_x"] = shape_label(r.get("shape")) if by == "shape" else str(r.get(by, "—"))
            if len({str(q.get("instance")) for q in data}) > 1:
                r["_x"] = f"{r.get('instance')}\n{r['_x']}"
        xs = _cats(data, "_x")
        hues = _fold(_cats(data, hue))
        fig, axes = new_figure(
            1,
            1,
            figsize=(max(w * 2.4, 0.55 * len(xs) * max(1, len(hues)) + 2.2), h * 1.3 + 0.3),
        )
        ax = axes[0, 0]
        if not data:
            message_axes(ax, "no memory rows", title or "memory")
            return fig
        k = max(1, len(hues))
        bw = min(0.8 / k, 0.3)
        vals = []
        for j, hv in enumerate(hues):
            for i, xc in enumerate(xs):
                sel = [r for r in data if r["_x"] == xc and str(r.get(hue)) == hv]
                if not sel:
                    continue
                v = float(np.mean([_num(r.get(value)) for r in sel]))
                xpos = i + (j - (k - 1) / 2) * bw
                ax.bar(xpos, v, width=bw * 0.92, color=t.palette[j], label=hv if i == 0 else None)
                ax.text(
                    xpos, v, f"{v:.3g}", ha="center", va="bottom", fontsize="x-small", color=t.ink2
                )
                vals.append(v)
        ax.set_xticks(range(len(xs)), xs)
        pos = [v for v in vals if v > 0]
        if len(pos) > 1 and max(pos) / min(pos) > 30:
            ax.set_yscale("log")
        ax.margins(y=0.2)
        grid_on(ax)
        ax.set_ylabel(_Y_LABEL.get(value, value))
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) > 1:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.set_title(title or f"memory by {'gradient mode' if hue == 'mode' else hue}")
        return fig


# ---------------------------------------------------------------------------------------------
# wall-clock breakdown
# ---------------------------------------------------------------------------------------------
def _segments(obj: Any) -> list[tuple[str, float]]:
    """``[(segment, seconds)]`` of one run (Result, gallery entry / manifest row, or dict)."""
    if hasattr(obj, "timing") and hasattr(obj, "stage_results"):
        timing = dict(obj.timing or {})
        per = [float(s.get("seconds") or 0.0) for s in obj.stage_results] or list(
            timing.get("per_stage_s") or []
        )
        names = [str(s.get("name") or f"stage{i + 1}") for i, s in enumerate(obj.stage_results)]
        names = names or [f"stage{i + 1}" for i in range(len(per))]
        segs = list(zip(names, per))
        tot = float(timing.get("total_s") or sum(per))
        if tot - sum(per) > 1e-3:
            segs.append(("other", tot - sum(per)))
        return segs
    d = obj.to_dict() if hasattr(obj, "to_dict") else dict(obj)
    timing = dict(d.get("timing") or d)
    segs: list[tuple[str, float]] = []
    if _num(timing.get("generate_s")) > 0:
        segs.append(("generate data", _num(timing["generate_s"])))
    per = timing.get("per_stage_s") or []
    if isinstance(per, str):
        per = json.loads(per)
    for i, v in enumerate(per):
        segs.append((f"stage {i + 1}", float(v)))
    solve = _num(timing.get("solve_s", timing.get("time_s")))
    if not per and math.isfinite(solve):
        segs.append(("solve", solve))
    elif per and math.isfinite(solve) and solve - sum(per) > 1e-3:
        segs.append(("solver overhead", solve - sum(float(v) for v in per)))
    for key, lab in (("evaluate_s", "evaluate"), ("plots_s", "figures")):
        if _num(timing.get(key)) > 0:
            segs.append((lab, _num(timing[key])))
    return segs


def _run_label(r: Any, i: int) -> str:
    if isinstance(r, Mapping):
        for k in ("instance", "name", "method"):
            if r.get(k):
                return str(r[k])
    return str(getattr(r, "name", None) or f"run {i + 1}")


def plot_wallclock_breakdown(
    results: Any,
    *,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """Stacked horizontal bars of where the wall-clock went, one bar per run.

    Accepts a :class:`~nefi.solve.Result`, ``{label: Result}``, gallery entries / a gallery
    manifest, or rows with ``per_stage_s`` / ``generate_s`` / ``solve_s``. Curriculum stages take
    the categorical colors (stage 1 = slot 1, ...); data generation, overheads, evaluation and
    figures are neutral grays.
    """
    if hasattr(results, "timing") and hasattr(results, "stage_results"):
        items = [(labels[0] if labels else "run", results)]
    elif isinstance(results, Mapping) and "entries" in results:
        items = [(str(e.get("name")), e) for e in results["entries"] if e.get("status") == "ok"]
    elif isinstance(results, Mapping):
        items = list(results.items())
    else:
        items = []
        for i, r in enumerate(list(results)):
            if isinstance(r, Mapping) and r.get("status") not in (None, "ok"):
                continue
            items.append((labels[i] if labels and i < len(labels) else _run_label(r, i), r))
    with styled(dark):
        t = theme()
        w, h = panel_size()
        n = max(1, len(items))
        fig, axes = new_figure(1, 1, figsize=(w * 3.2 + 1.2, 0.95 + 0.3 * n))
        ax = axes[0, 0]
        if not items:
            message_axes(ax, "no timings", title or "wall-clock")
            return fig
        neutral = {  # four clearly separated gray steps; stages carry the categorical colors
            "generate data": t.ink2,
            "solver overhead": t.muted,
            "other": t.muted,
            "evaluate": t.axis,
            "figures": t.grid,
            "solve": t.palette[0],
        }
        seen: dict[str, str] = {}
        ys = np.arange(len(items))[::-1].astype(float)
        for yi, (lab, obj) in zip(ys, items):
            left = 0.0
            try:
                segs = _segments(obj)
            except Exception as e:  # pragma: no cover - malformed timing dicts
                log.debug("timing of %s unavailable: %s", lab, e)
                segs = []
            stage_i = 0
            for name, sec in segs:
                if not (math.isfinite(sec) and sec > 0):
                    continue
                if name in neutral:
                    col = neutral[name]
                else:
                    col = t.palette[min(stage_i, 7)]
                    stage_i += 1
                    name = f"stage {stage_i}"
                ax.barh(
                    yi,
                    sec,
                    left=left,
                    height=0.6,
                    color=col,
                    edgecolor=t.surface,
                    linewidth=1.0,
                    label=name if name not in seen else None,
                )
                seen.setdefault(name, col)
                left += sec
            ax.text(left, yi, f" {left:.3g} s", va="center", fontsize="x-small", color=t.ink2)
        ax.set_yticks(ys, [lab for lab, _ in items])
        ax.margins(x=0.12)
        grid_on(ax, "x")
        ax.set_xlabel("seconds")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.set_title(title or "wall-clock breakdown")
        return fig


# ---------------------------------------------------------------------------------------------
# profiles (tools/profile_instance.py --json) and speedups
# ---------------------------------------------------------------------------------------------
_FALSY = ("", "none", "false", "0", "off", "eager", "no")


def load_profiles(source: Any = "runs/_perf") -> list[dict[str, Any]]:
    """Rows of ``tools/profile_instance.py --json`` files.

    Args:
        source: a directory (every ``*.jsonl`` in it), a ``.jsonl`` file, or a list of them.
            Missing paths give no rows (never an error).

    Returns:
        One normalized row per JSON line, with ``source`` (file stem) and ``_order`` (file
        modification time, line number) so later measurements can win.
    """
    if source is None:
        return []
    items = source if isinstance(source, list | tuple) else [source]
    paths: list[Path] = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            paths += sorted(p.glob("*.jsonl"))
        elif p.is_file():
            paths.append(p)
    rows: list[dict[str, Any]] = []
    for p in sorted(set(paths), key=lambda q: (q.stat().st_mtime, q.name)):
        mtime = p.stat().st_mtime
        for i, line in enumerate(p.read_text(errors="replace").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, Mapping) or "instance" not in r:
                continue
            row = normalize_row(r)
            row["source"] = p.stem
            row["_order"] = (mtime, i)
            rows.append(row)
    return rows


def profile_variant(row: Mapping[str, Any]) -> str:
    """``"eager"`` for a plain solver-loop profile, else a short label of its switches
    (``"compile=step"``, ``"autocast=bf16"``, ``"solver=chebyshev · jacobi_iters=13"``, ...)."""
    parts = []
    for key in ("compile", "autocast"):
        v = row.get(key)
        if v is not None and str(v).strip().lower() not in _FALSY:
            parts.append(f"{key}={v}")
    if row.get("cuda_graphs") and str(row.get("cuda_graphs")).lower() not in _FALSY:
        parts.append("cuda graphs")
    ov = row.get("overrides") or []
    if isinstance(ov, str):
        ov = [ov]
    if isinstance(ov, Mapping):
        ov = [f"{k}={v}" for k, v in ov.items()]
    parts += [str(o) for o in ov]
    loop = str(row.get("loop") or "solver")
    if loop != "solver":
        parts.append(f"{loop} loop")
    return " · ".join(parts) or "eager"


def _profile_ms(row: Mapping[str, Any]) -> float:
    for key in ("ms_median", "ms_per_step", "ms_mean"):
        v = _num(row.get(key))
        if math.isfinite(v) and v > 0:
            return v
    return float("nan")


def profile_speedups(rows: Any) -> list[dict[str, Any]]:
    """Speedup of every accelerated profile over plain eager execution.

    The reference of a row is the **latest** eager row (:func:`profile_variant` == ``"eager"``)
    of the same instance, device and thread count (any thread count as a fallback); speedup =
    ``eager ms / variant ms`` (> 1: faster). Harness variants (``loop`` other than ``solver``)
    are skipped; repeated variants keep their latest measurement.

    Returns:
        Rows ``{instance, variant, device, threads, eager_ms, ms, speedup, source}``.
    """
    if isinstance(rows, str | Path):
        rows = load_profiles(rows)
    data = list(rows) if isinstance(rows, list | tuple) else load_rows(rows)
    data = [r for r in data if math.isfinite(_profile_ms(r)) and not r.get("error")]
    data.sort(key=lambda r: tuple(r.get("_order") or (0.0, 0)))
    eager: dict[tuple, Mapping[str, Any]] = {}
    for r in data:
        if profile_variant(r) == "eager":
            inst, dev = str(r.get("instance")), str(r.get("device", "?"))
            eager[(inst, dev, str(r.get("threads")))] = r
            eager[(inst, dev, "*")] = r
    out: dict[tuple, dict[str, Any]] = {}
    for r in data:
        lab = profile_variant(r)
        if lab == "eager" or str(r.get("loop") or "solver") != "solver":
            continue
        inst, dev, thr = str(r.get("instance")), str(r.get("device", "?")), str(r.get("threads"))
        ref = eager.get((inst, dev, thr)) or eager.get((inst, dev, "*"))
        if ref is None:
            continue
        e_ms, v_ms = _profile_ms(ref), _profile_ms(r)
        out[(inst, lab, dev, thr)] = {
            "instance": inst,
            "variant": lab,
            "device": dev,
            "threads": r.get("threads"),
            "eager_ms": e_ms,
            "ms": v_ms,
            "speedup": e_ms / v_ms,
            "source": r.get("source"),
        }
    return list(out.values())


def plot_speedup(
    rows: Any,
    *,
    title: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """Per-instance "speedup vs eager" bars (log axis, ``1×`` = eager, value at every tip).

    Args:
        rows: :func:`profile_speedups` output, or profile rows / a path for
            :func:`load_profiles` (converted automatically).
        title: axes title.
        dark: dark theme.

    Variants (compile modes, autocast, solver switches...) take the categorical colours; when the
    profiles cover several thread counts the count is part of the variant label.
    """
    if isinstance(rows, str | Path):
        rows = load_profiles(rows)
    data = [dict(r) for r in (rows or [])]
    if data and "speedup" not in data[0]:
        data = profile_speedups(data)
    data = [r for r in data if math.isfinite(_num(r.get("speedup"))) and _num(r["speedup"]) > 0]
    threads = {str(r.get("threads")) for r in data}
    for r in data:
        r["_hue"] = r["variant"] + (f" ({r.get('threads')} thr)" if len(threads) > 1 else "")
    with styled(dark):
        t = theme()
        w, h = panel_size()
        insts = _cats(data, "instance")
        hues = _fold(_cats(data, "_hue"))
        # one packed block per instance: only the variants it was profiled with
        blocks = [
            (
                inst,
                [hv for hv in hues if any(r["instance"] == inst and r["_hue"] == hv for r in data)],
            )
            for inst in insts
        ]
        n_bars = sum(len(b) for _, b in blocks)
        fig, axes = new_figure(
            1, 1, figsize=(w * 3.0 + 1.6, 0.8 + 0.2 * max(1, n_bars) + 0.2 * len(insts))
        )
        ax = axes[0, 0]
        if not data:
            message_axes(ax, "no profiles with an eager reference", title or "speedup vs eager")
            return fig
        bar, gap = 1.0, 0.8
        y, centers, vals = 0.0, [], []
        seen: set[str] = set()
        for inst, present in reversed(blocks):
            y0 = y
            for hv in reversed(present):
                j = hues.index(hv)
                sel = [r for r in data if r["instance"] == inst and r["_hue"] == hv]
                v = float(np.mean([_num(r["speedup"]) for r in sel]))
                ax.barh(
                    y,
                    abs(v - 1.0) if abs(v - 1.0) > 1e-3 else 0.01,
                    left=min(1.0, v),
                    height=bar * 0.82,
                    color=t.palette[j],
                    label=hv if hv not in seen else None,
                )
                seen.add(hv)
                ax.text(max(v, 1.0), y, f" ×{v:.2f}", va="center", fontsize="x-small", color=t.ink2)
                vals.append(v)
                y += bar
            centers.append((y0 + y - bar) / 2)
            y += gap
        ys = centers[::-1]
        handles, labels = ax.get_legend_handles_labels()
        order = sorted(range(len(labels)), key=lambda i: hues.index(labels[i]))
        ax.axvline(1.0, color=t.ink, lw=0.9, zorder=3)
        ax.set_xscale("log")
        lo = min(0.5, min(vals) / 1.25)
        hi = max(2.0, max(vals) * 1.6)
        ax.set_xlim(lo, hi)
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

        ticks = [v for v in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0) if lo <= v <= hi]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}×"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks(ys, insts)
        ax.set_ylim(-0.7 * bar, y - gap + 0.2 * bar)
        grid_on(ax, "x")
        ax.set_xlabel("speedup vs eager (eager ms / variant ms; > 1× is faster)")
        ax.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
        )
        dev = ", ".join(sorted({str(r.get("device")) for r in data}))
        ax.set_title(title or f"speedup vs eager ({dev}; tools/profile_instance.py)")
        return fig


# ---------------------------------------------------------------------------------------------
# benchmark table
# ---------------------------------------------------------------------------------------------
def _mean_ci(values: Sequence[float], ci: float) -> tuple[float, float, int]:
    try:
        from ..bench.report import mean_ci

        return mean_ci(values, ci)
    except ImportError:  # pragma: no cover - bench package unavailable
        vals = [v for v in values if math.isfinite(v)]
        if not vals:
            return float("nan"), float("nan"), 0
        m = statistics.fmean(vals)
        if len(vals) < 2:
            return m, float("nan"), len(vals)
        from scipy.stats import t as student_t

        hw = float(student_t.ppf(0.5 + ci / 2, len(vals) - 1)) * statistics.stdev(vals)
        return m, hw / math.sqrt(len(vals)), len(vals)


def aggregate(
    rows: Any, metric: str, by: str = "method", group: str | None = None, ci: float = 0.95
) -> list[dict[str, Any]]:
    """Mean ± CI of ``metric`` per ``by`` (and ``group``) over rows without errors.

    Already-aggregated rows (``BenchmarkResult.table()`` output with ``<metric>_ci``) pass
    through unchanged.
    """
    data = _ok(load_rows(rows))
    if data and f"{metric}_ci" in data[0] and "n" in data[0]:
        return [
            {
                by: r.get(by),
                "group": r.get(group) if group else None,
                "mean": _num(r.get(metric)),
                "ci": _num(r.get(f"{metric}_ci")),
                "n": int(_num(r.get("n")) or 0),
            }
            for r in data
        ]
    out = []
    groups = _cats(data, group) if group and any(group in r for r in data) else [None]
    for gv in groups:
        for m in _cats(data, by):
            vals = [
                _num(r.get(metric))
                for r in data
                if str(r.get(by)) == m and (gv is None or str(r.get(group)) == gv)
            ]
            vals = [v for v in vals if math.isfinite(v)]
            if not vals:
                continue
            mean, hw, n = _mean_ci(vals, ci)
            out.append({by: m, "group": gv, "mean": mean, "ci": hw, "n": n})
    return out


def plot_bench_table(
    rows: Any,
    metric: str,
    *,
    by: str = "method",
    group: str | None = "class",
    ci: float = 0.95,
    higher_is_better: bool | None = None,
    title: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """Mean ± confidence interval of ``metric`` per method (dots with CI whiskers), one small
    multiple per scene class.

    Args:
        rows: benchmark rows (see :func:`load_rows`), aggregated rows, or a ``BenchmarkResult``.
        metric: metric column (``"psnr"``, ``"ssim"``, ``"time_s"`` ...).
        by: row category on the y axis.
        group: facet column (``None`` or absent → one panel).
        ci: confidence level of the Student-t interval.
        higher_is_better: direction for the arrow in the title (default: guessed from the name).
        title: figure title.
        dark: dark theme.
    """
    agg = aggregate(rows, metric, by, group, ci)
    groups = [g for g in dict.fromkeys(a["group"] for a in agg)]
    if higher_is_better is None:
        try:
            from ..bench.report import metric_direction

            higher_is_better = metric_direction(metric)
        except ImportError:  # pragma: no cover
            higher_is_better = None
    arrow = {True: " ↑", False: " ↓"}.get(higher_is_better, "")
    with styled(dark):
        t = theme()
        w, h = panel_size()
        ng = max(1, len(groups))
        cats = list(dict.fromkeys(str(a[by]) for a in agg))
        fig, axes = new_figure(
            1,
            ng,
            figsize=(ng * w * 1.5 + 1.0, 0.9 + 0.3 * max(1, len(cats))),
            sharey=True,
            squeeze=False,
        )
        if not agg:
            message_axes(axes[0, 0], f"no '{metric}' values", title or metric)
            return fig
        ys = {c: float(i) for i, c in enumerate(reversed(cats))}
        for p, gv in enumerate(groups):
            ax = axes[0, p]
            sel = [a for a in agg if a["group"] == gv]
            best = None
            if higher_is_better is not None and sel:
                pick = max if higher_is_better else min
                best = pick(
                    sel,
                    key=lambda a: (
                        a["mean"]
                        if math.isfinite(a["mean"])
                        else -math.inf
                        if higher_is_better
                        else math.inf
                    ),
                )
            for a in sel:
                y = ys[str(a[by])]
                hw = a["ci"] if math.isfinite(a["ci"]) else 0.0
                col = t.palette[0]
                ax.errorbar(
                    a["mean"],
                    y,
                    xerr=hw,
                    fmt="o",
                    ms=5,
                    color=col,
                    ecolor=col,
                    elinewidth=1.4,
                    capsize=2.5,
                    mec=t.surface,
                    mew=1.0,
                )
                unit = " dB" if "psnr" in metric.lower() else ""
                txt = f"{a['mean']:#.3g}".rstrip(".")
                if math.isfinite(a["ci"]):
                    txt += f" ± {a['ci']:.2g}"
                txt += unit
                weight = "bold" if (best is not None and a is best and len(sel) > 1) else "normal"
                ax.annotate(
                    txt,
                    (a["mean"] + hw, y),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize="x-small",
                    color=t.ink2,
                    fontweight=weight,
                )
            ax.set_yticks(list(ys.values()), list(ys.keys()))
            ax.margins(x=0.3, y=0.15)
            grid_on(ax, "x")
            ax.set_title(str(gv) if gv is not None else "")
            ax.set_xlabel(f"{metric}{arrow}")
        n_txt = ", ".join(str(v) for v in sorted({a["n"] for a in agg}))
        pct = int(round(100 * ci))
        fig.suptitle(title or f"{metric}{arrow}: mean ± {pct}% CI (n = {n_txt})")
        return fig


# ---------------------------------------------------------------------------------------------
# collection (profiler logic of tools/profile_instance.py)
# ---------------------------------------------------------------------------------------------
def _saved_tensor_bytes(fn: Any) -> tuple[Any, int]:
    """Run ``fn()`` (a forward pass) while counting unique autograd saved-tensor bytes."""
    import torch

    seen: set[tuple[int, int]] = set()
    total = [0]

    def pack(t: torch.Tensor) -> torch.Tensor:
        try:
            st = t.untyped_storage()
            key = (st.data_ptr(), st.nbytes())
            if key not in seen:
                seen.add(key)
                total[0] += int(st.nbytes())
        except Exception:  # pragma: no cover - exotic tensor subclasses
            total[0] += int(t.numel() * t.element_size())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, total[0]


def profile_problem(
    problem: Any,
    stage: Any,
    *,
    device: Any = "auto",
    steps: int = 20,
    warmup: int = 3,
    saved_tensors: bool = True,
) -> dict[str, Any]:
    """Time optimization steps of one curriculum stage (forward / backward / optimizer split).

    Mirrors ``tools/profile_instance.py``: the stage's resolution, operator, measurement and
    loss weights; AdamW on the trainable field (and operator) parameters.
    """
    import torch

    from ..losses.base import Context
    from ..utils.device import peak_memory_mb, reset_peak_memory, resolve_device, synchronize

    dev = resolve_device(device)
    problem.to(dev)
    dom = problem.domain if stage.shape is None else problem.domain.at(stage.shape)
    coords = dom.coords(device=dev)
    op = problem.operator.at_resolution(dom.shape).to(dev)
    obs = problem.measurement_at(dom.shape).to(dev)
    problem.field.on_stage_start(stage, dom)
    problem.field.to(dev)
    losses = problem.losses.with_weights(stage.loss_weights)
    params = [p for p in problem.field.parameters() if p.requires_grad]
    params += [p for p in problem.operator.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=max(stage.lr, 1e-8))
    acc = {"fwd": 0.0, "bwd": 0.0, "opt": 0.0}

    def forward(progress: float):
        fields = problem.field(coords, progress)
        pred = op(fields)
        ctx = Context(fields, pred, obs, dom, op, problem.field, stage, 0, progress)
        total, _ = losses(ctx)
        return total

    def step(progress: float, timed: bool) -> None:
        opt.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        total = forward(progress)
        if timed:
            synchronize(dev)
        t1 = time.perf_counter()
        if total.requires_grad:
            total.backward()
        if timed:
            synchronize(dev)
        t2 = time.perf_counter()
        opt.step()
        if timed:
            synchronize(dev)
        t3 = time.perf_counter()
        if timed:
            acc["fwd"] += t1 - t0
            acc["bwd"] += t2 - t1
            acc["opt"] += t3 - t2

    for i in range(max(0, warmup)):
        step(i / max(1, steps), timed=False)
    synchronize(dev)
    reset_peak_memory(dev)
    t0 = time.perf_counter()
    for i in range(max(1, steps)):
        step(min(1.0, i / max(1, steps)), timed=True)
    synchronize(dev)
    dt = (time.perf_counter() - t0) / max(1, steps)
    saved = None
    if saved_tensors:
        opt.zero_grad(set_to_none=True)
        total, nbytes = _saved_tensor_bytes(lambda: forward(1.0))
        if total.requires_grad:
            total.backward()
        saved = nbytes / 2**20
    n = max(1, steps)
    return {
        "stage": getattr(stage, "name", "stage"),
        "shape": [int(s) for s in dom.shape],
        "n": int(np.prod(dom.shape)),
        "params": int(sum(p.numel() for p in params)),
        "steps": int(n),
        "ms_per_step": 1e3 * dt,
        "fwd_ms": 1e3 * acc["fwd"] / n,
        "bwd_ms": 1e3 * acc["bwd"] / n,
        "opt_ms": 1e3 * acc["opt"] / n,
        "peak_mem_mb": peak_memory_mb(dev),
        "saved_mb": saved,
        "device": str(dev),
    }


def collect_performance(
    instances: Any = None,
    device: Any = "auto",
    steps: int = 20,
    *,
    warmup: int = 3,
    stage: int = -1,
    smoke: bool = True,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    variants: Mapping[str, Mapping[str, Any]] | None = None,
    saved_tensors: bool = True,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Profile the optimization step of several instances (rows for the plots above).

    Args:
        instances: names / classes / objects (default: every registered instance).
        device: device to profile on.
        steps: timed steps per instance (after ``warmup`` untimed ones).
        warmup: untimed steps (kernel caches, lazy init).
        stage: curriculum stage index to profile (default: last = finest).
        smoke: apply the instances' smoke presets (small grids).
        overrides: ``{instance: {config_key: value}}`` (e.g. grid sizes for scaling studies).
        variants: ``{label: {config_key: value}}`` applied to every instance, one row each
            (e.g. ``{"adjoint": {"grad_mode": "adjoint"}, "autograd": {...}}``); rows carry
            ``mode = label``. Variants an instance rejects produce an error row.
        saved_tensors: also measure autograd saved-tensor MB (device-agnostic memory proxy).
        seed: data seed.

    Returns:
        One row per (instance, variant); failures carry ``error`` instead of timings.
    """
    from ..utils.seed import seed_everything
    from ._instances import as_list, registered_instances, resolve_instance

    specs = as_list(instances) or registered_instances()
    variants = dict(variants or {"default": {}})
    rows: list[dict[str, Any]] = []
    for spec in specs:
        name = spec if isinstance(spec, str) else getattr(spec, "name", type(spec).__name__)
        for vname, vcfg in variants.items():
            row: dict[str, Any] = {"instance": str(name), "mode": vname, "error": None}
            try:
                cfg = {**dict((overrides or {}).get(str(name), {})), **dict(vcfg)}
                inst, info = resolve_instance(spec, smoke=smoke, overrides=cfg)
                row["smoke"] = info.get("smoke")
                _, meas = inst.make_measurement(seed=seed)
                seed_everything(seed)
                problem = inst.build_problem(meas)
                cur = getattr(problem, "curriculum", None) or inst.default_curriculum()
                st = cur.stages[stage]
                row.update(
                    profile_problem(
                        problem,
                        st,
                        device=device,
                        steps=steps,
                        warmup=warmup,
                        saved_tensors=saved_tensors,
                    )
                )
            except Exception as e:  # one broken instance must not stop the dashboard
                log.warning("profiling %s (%s) failed: %s: %s", name, vname, type(e).__name__, e)
                row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
    return rows


def collect_scaling(
    instance: Any,
    sizes: Sequence[Any],
    *,
    key: str | None = None,
    device: Any = "auto",
    steps: int = 10,
    warmup: int = 2,
    stage: int = -1,
    smoke: bool = True,
    variants: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Profile one instance at several sizes (rows for :func:`plot_scaling`).

    Args:
        instance: registered name (or class / object).
        sizes: config-override dicts (``[{"n": 32}, {"n": 64}]``) or plain values with ``key``
            (``key="n", sizes=[32, 64, 128]``; tuples are passed as-is, e.g. ``grid``).
        key: config field the plain ``sizes`` values set.
        device, steps, warmup, stage, smoke, variants: as in :func:`collect_performance`.
    """
    rows: list[dict[str, Any]] = []
    name = instance if isinstance(instance, str) else getattr(instance, "name", "instance")
    for sz in sizes:
        cfg = dict(sz) if isinstance(sz, Mapping) else {str(key): sz}
        if not isinstance(sz, Mapping) and key is None:
            raise ValueError("collect_scaling: pass dict sizes or key=<config field>")
        out = collect_performance(
            [instance],
            device,
            steps,
            warmup=warmup,
            stage=stage,
            smoke=smoke,
            overrides={str(name): cfg},
            variants=variants,
        )
        for r in out:
            r["size"] = cfg
        rows.extend(out)
    return rows


__all__ = [
    "aggregate",
    "collect_performance",
    "collect_scaling",
    "load_profiles",
    "load_rows",
    "loglog_slope",
    "normalize_row",
    "plot_bench_table",
    "plot_memory",
    "plot_scaling",
    "plot_speedup",
    "plot_step_time",
    "plot_wallclock_breakdown",
    "profile_problem",
    "profile_speedups",
    "profile_variant",
    "shape_label",
]
