"""Training-dynamics figures: loss history with stage boundaries and the annealing schedule,
per-stage summaries, the coarse→fine multiscale view and snapshot animations.

* :func:`plot_history` — loss components (log scale) with curriculum stage boundaries and shaded
  annealing windows (β < K), plus the annealing progress β/K and the learning-rate schedule in
  stacked panels that share the step axis (never a second y-axis), and an optional MSE
  noise-floor line σ².
* :func:`plot_stage_summary` — steps, seconds (ms/step) and final data loss per stage.
* :func:`plot_multiscale` — end-of-stage fields at each stage's own resolution (NeTMY Tab. 6:
  32² → 64²) next to the final and ground-truth fields, above the loss curve.
* :func:`animate_snapshots` — GIF / MP4 of :class:`~nefi.solve.callbacks.FieldSnapshots`
  (1-D lines, 2-D images; 3-D fields as an evolving depth mosaic above the GT mosaic, or the
  slice through the anomaly), frames rendered off-screen with Agg and kept under a size budget.
* :class:`StageSnapshots` — callback that keeps every field at the end of each stage.

History columns are aligned with :func:`history_arrays`: loss components switched off in some
stages (weight 0 → not logged) are padded with NaN using ``Result.stage_results``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..solve.callbacks import Callback
from .fields import (
    representative_slice,
    resolve_field,
    show_image,
    squeeze_field,
    take_slice,
)
from .style import (
    cmap_for,
    color_limits,
    get_cmap,
    grid_on,
    message_axes,
    new_figure,
    panel_size,
    styled,
    theme,
)

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

log = logging.getLogger("nefi")

#: history columns that are bookkeeping, not loss values.
BOOKKEEPING = frozenset(
    {"step", "global_step", "stage", "lr", "progress", "elapsed", "time", "restart"}
)


# ---------------------------------------------------------------------------------------------
# history alignment
# ---------------------------------------------------------------------------------------------
def _history_of(obj: Any) -> tuple[dict[str, list], list[dict]]:
    if hasattr(obj, "history"):
        return dict(obj.history or {}), list(getattr(obj, "stage_results", None) or [])
    return dict(obj or {}), []


def history_arrays(
    history: Any, stage_results: Sequence[Mapping] | None = None
) -> dict[str, np.ndarray]:
    """Per-step numpy columns of ``Result.history`` (or a history dict), all of equal length.

    Components that were not logged in some stages (zero weight → skipped by the solver) are
    placed in the stages that list them in ``stage_results[i]["final"]`` and padded with NaN.

    Args:
        history: a :class:`~nefi.solve.Result` or its ``history`` dict.
        stage_results: per-stage summaries (default: ``history.stage_results``).
    """
    h, sr = _history_of(history)
    sr = list(stage_results) if stage_results is not None else sr
    ref = next((h[k] for k in ("total", "global_step", "step") if h.get(k)), None)
    n = len(ref) if ref is not None else max((len(v) for v in h.values()), default=0)
    stages = np.asarray(h.get("stage", [0] * n), dtype=float)
    if len(stages) != n:
        stages = np.zeros(n)
    out: dict[str, np.ndarray] = {}
    for k, v in h.items():
        try:
            arr = np.asarray(v, dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        if arr.shape[0] == n:
            out[k] = arr
            continue
        filled = np.full(n, np.nan)
        base = k[5:] if k.startswith("loss/") else k
        active = [i for i, s in enumerate(sr) if base in (s.get("final") or {})]
        sel = np.isin(stages, active) if active else np.zeros(n, dtype=bool)
        if active and int(sel.sum()) == arr.shape[0]:
            filled[sel] = arr
        else:
            m = min(n, arr.shape[0])
            filled[:m] = arr[:m]
            log.debug("history column %r has %d/%d entries; left-aligned", k, arr.shape[0], n)
        out[k] = filled
    return out


def loss_keys(arrays: Mapping[str, np.ndarray]) -> list[str]:
    """Loss-valued columns (everything but bookkeeping), in history order."""
    return [k for k, v in arrays.items() if k not in BOOKKEEPING and np.isfinite(v).any()]


def _last_finite(a: np.ndarray) -> float:
    """``|last finite value|`` of a column (0 if there is none)."""
    v = a[np.isfinite(a)]
    return float(abs(v[-1])) if v.size else 0.0


def _stage_starts(arrays: Mapping[str, np.ndarray]) -> list[tuple[int, int]]:
    """``[(stage_index, first_row), ...]`` from the ``stage`` column."""
    st = arrays.get("stage")
    if st is None or st.size == 0:
        return [(0, 0)]
    out = [(int(st[0]), 0)]
    for i in range(1, st.shape[0]):
        if st[i] != st[i - 1]:
            out.append((int(st[i]), i))
    return out


def _stage_label(sr: Sequence[Mapping], idx: int) -> str:
    if idx < len(sr):
        s = sr[idx]
        shape = s.get("shape")
        shape_txt = "×".join(str(int(v)) for v in shape) if shape else ""
        name = str(s.get("name") or f"stage {idx + 1}")
        return f"{name} {shape_txt}".strip()
    return f"stage {idx + 1}"


def _x_axis(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    gs = arrays.get("global_step")
    n = len(next(iter(arrays.values()))) if arrays else 0
    if gs is not None and gs.shape[0] == n and np.isfinite(gs).all():
        return gs
    return np.arange(n, dtype=float)


def _annealing_spans(
    x: np.ndarray, progress: np.ndarray | None, stages: np.ndarray | None
) -> list[tuple[float, float]]:
    """``[(x_start, x_end)]`` where β < K (progress < 1), split at stage boundaries."""
    if progress is None or progress.size == 0:
        return []
    active = progress < 1.0 - 1e-9
    if stages is not None and stages.size == progress.size:
        brk = np.concatenate([[True], stages[1:] != stages[:-1]])
    else:
        brk = np.zeros_like(active)
        brk[0] = True
    spans, start = [], None
    for i in range(active.shape[0]):
        if start is not None and (not active[i] or brk[i]):
            spans.append((x[start], x[i]))
            start = None
        if active[i] and start is None:
            start = i
    if start is not None:
        spans.append((x[start], x[-1]))
    return spans


def _draw_stage_marks(
    ax: Axes, x: np.ndarray, arrays: Mapping[str, np.ndarray], sr: Sequence[Mapping], labels: bool
) -> None:
    t = theme()
    starts = _stage_starts(arrays)
    for j, (sidx, row) in enumerate(starts):
        if j > 0:
            ax.axvline(x[row], color=t.axis, lw=0.8, zorder=1)
        if labels and len(starts) > 1:
            ax.text(
                x[row],
                1.0,
                " " + _stage_label(sr, sidx),
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="bottom",
                fontsize="x-small",
                color=t.ink2,
                clip_on=False,
            )


# ---------------------------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------------------------
def plot_history(
    result: Any,
    *,
    keys: Sequence[str] | None = None,
    noise_floor: float | None = None,
    noise_std: float | None = None,
    logy: bool | str = "auto",
    show_progress: bool = True,
    show_lr: bool = True,
    max_series: int = 7,
    title: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """Loss history with curriculum stage boundaries, annealing windows, β/K and LR panels.

    Args:
        result: :class:`~nefi.solve.Result` (or a ``history`` dict).
        keys: loss columns to draw (default: ``total``, ``data_loss`` and every component).
        noise_floor: horizontal reference on the loss panel (e.g. the expected data loss at the
            truth).
        noise_std: convenience: ``noise_floor = σ²`` — the floor of an **MSE** data loss (do not
            use it with log- or normalized data losses).
        logy: log-scaled loss axis (``"auto"``: when all values are positive).
        show_progress: add the annealing-progress panel (β/K).
        show_lr: add the learning-rate panel.
        max_series: at most this many loss curves (≤ 8 categorical slots; extras are listed).
        title: figure title.
        dark: dark theme.
    """
    arrays = history_arrays(result)
    _, sr = _history_of(result)
    n = len(arrays.get("total", arrays.get("global_step", [])))
    with styled(dark):
        t = theme()
        w, h = panel_size()
        if n == 0:
            fig, axes = new_figure(1, 1, figsize=(w * 2.4, h))
            message_axes(axes[0, 0], "empty history", title or "training history")
            return fig
        x = _x_axis(arrays)
        prog = arrays.get("progress")
        lr = arrays.get("lr")
        panels = ["loss"]
        if show_progress and prog is not None and np.isfinite(prog).any():
            panels.append("progress")
        if show_lr and lr is not None and np.isfinite(lr).any() and np.nanmax(lr) > 0:
            panels.append("lr")
        ratios = [3.0] + [1.0] * (len(panels) - 1)
        fig, axes = new_figure(
            len(panels),
            1,
            figsize=(w * 3.2, h * (1.15 + 0.42 * (len(panels) - 1)) + 0.3),
            sharex=True,
            height_ratios=ratios,
        )
        ax = axes[0, 0]
        cand = list(keys) if keys is not None else loss_keys(arrays)
        cand = [k for k in cand if k in arrays]
        series: list[tuple[str, str, float]] = []  # (key, color, lw); drawn in reverse order
        slots = list(t.palette)
        if "total" in cand:
            series.append(("total", t.ink, 1.1))
        tot = arrays.get("total")
        data = arrays.get("data_loss")
        comps = [k for k in cand if k not in ("total", "data_loss")]
        merged: dict[str, list[str]] = {}
        if data is not None:  # components identical to the aggregate data loss are merged
            same = [k for k in comps if np.allclose(arrays[k], data, equal_nan=True)]
            comps = [k for k in comps if k not in same]
            if same:
                merged["data_loss"] = same
        if (
            "data_loss" in cand
            and data is not None
            and not (tot is not None and np.allclose(data, tot, equal_nan=True))
        ):
            series.append(("data_loss", slots.pop(0), 2.2))
        final = {k: _last_finite(arrays[k]) for k in comps}
        room = min(len(slots), max(0, max_series - len(series)))
        kept = sorted(comps, key=lambda k: -final.get(k, 0.0))[:room]
        dropped = [k for k in comps if k not in kept]
        for k in comps:
            if k not in kept:
                continue
            twin = next(
                (
                    s_[0]
                    for s_ in series
                    if s_[0] in comps and np.allclose(arrays[s_[0]], arrays[k], equal_nan=True)
                ),
                None,
            )
            if twin is not None:  # identical curves (e.g. two L1 terms on the same field)
                merged.setdefault(twin, []).append(k)
                continue
            series.append((k, slots.pop(0), 1.1))
        vals = np.concatenate([arrays[k][np.isfinite(arrays[k])] for k, _, _ in series] or [[]])
        use_log = (vals.size > 0 and bool(np.all(vals > 0))) if logy == "auto" else bool(logy)
        spans = _annealing_spans(x, prog, arrays.get("stage"))
        for j, (a0, a1) in enumerate(spans):
            for axx in axes[:, 0]:
                axx.axvspan(
                    a0,
                    a1,
                    color=t.muted,
                    alpha=0.09,
                    lw=0,
                    label="annealing (β < K)" if (j == 0 and axx is ax) else None,
                )
        for z, (k, c, lw) in enumerate(series):
            names = merged.get(k, [])
            if k == "data_loss":
                label = "data loss" + (f" (= {', '.join(names)})" if names else "")
            else:
                label = " = ".join([k, *names])
            # total on top (thin ink), data loss beneath (thick) so coinciding curves stay visible
            ax.plot(x, arrays[k], color=c, lw=lw, label=label, zorder=3 + len(series) - z)
        floor = noise_floor if noise_floor is not None else (noise_std**2 if noise_std else None)
        if floor:
            ax.axhline(floor, color=t.ink2, lw=0.8, zorder=2)
            ax.text(
                x[-1],
                floor,
                " σ² floor" if noise_std and noise_floor is None else " floor",
                va="bottom",
                ha="right",
                fontsize="x-small",
                color=t.ink2,
            )
        if use_log:
            ax.set_yscale("log")
        grid_on(ax)
        _draw_stage_marks(ax, x, arrays, sr, labels=True)
        ax.set_ylabel("loss")
        leg_title = f"+{len(dropped)} more: {', '.join(dropped[:3])}" if dropped else None
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title=leg_title)
        row = 1
        if "progress" in panels:
            axp = axes[row, 0]
            axp.plot(x, prog, color=t.palette[0], lw=1.2)
            axp.set_ylim(-0.03, 1.08)
            axp.set_yticks([0, 0.5, 1])
            axp.set_ylabel("β / K")
            grid_on(axp)
            _draw_stage_marks(axp, x, arrays, sr, labels=False)
            row += 1
        if "lr" in panels:
            axl = axes[row, 0]
            axl.plot(x, lr, color=t.palette[0], lw=1.2)
            if np.nanmin(lr) > 0 and np.nanmax(lr) / max(np.nanmin(lr), 1e-300) > 5:
                axl.set_yscale("log")
            axl.set_ylabel("learning rate")
            grid_on(axl)
            _draw_stage_marks(axl, x, arrays, sr, labels=False)
        axes[-1, 0].set_xlabel("step")
        secs = getattr(result, "timing", {}) or {}
        tot_s = secs.get("total_s")
        n_st = len(_stage_starts(arrays))
        info = f"{n} steps · {n_st} stage{'s' if n_st != 1 else ''}"
        if tot_s:
            info += f" · {tot_s:.3g} s"
        fig.suptitle(title or f"training history ({info})")
        return fig


def plot_stage_summary(
    result: Any, *, title: str | None = None, dark: bool | None = None
) -> Figure:
    """Per-stage bars: steps, wall-clock seconds (ms/step at the bar tip) and final data loss."""
    _, sr = _history_of(result)
    arrays = history_arrays(result)
    with styled(dark):
        t = theme()
        w, h = panel_size()
        if not sr:
            fig, axes = new_figure(1, 1, figsize=(w * 2.2, h))
            message_axes(axes[0, 0], "no stage results", title or "stages")
            return fig
        labels = []
        for i, s in enumerate(sr):
            lab = _stage_label(sr, i).replace(" ", "\n", 1)
            stop = s.get("stop")
            if stop and stop not in ("completed",):
                lab += f"\n({stop})"
            labels.append(lab)
        steps = [float(s.get("steps") or 0) for s in sr]
        secs = [float(s.get("seconds") or 0.0) for s in sr]
        final_data = []
        st = arrays.get("stage")
        dl = arrays.get("data_loss", arrays.get("total"))
        for i in range(len(sr)):
            v = np.nan
            if st is not None and dl is not None:
                sel = np.where(st == i)[0]
                if sel.size:
                    v = float(dl[sel[-1]])
            final_data.append(v)
        fig, axes = new_figure(1, 3, figsize=(w * 3.4, h * 1.15 + 0.3))
        xs = np.arange(len(sr))
        bw = min(0.6, 0.18 * len(sr) + 0.3)
        specs = [
            (axes[0, 0], steps, "steps", [f"{int(v)}" for v in steps]),
            (
                axes[0, 1],
                secs,
                "seconds",
                [
                    f"{s:.2g} s\n{1e3 * s / max(n, 1):.3g} ms/step" if n else f"{s:.2g} s"
                    for s, n in zip(secs, steps)
                ],
            ),
            (axes[0, 2], final_data, "final data loss", [f"{v:.3g}" for v in final_data]),
        ]
        for ax, vals, name, texts in specs:
            v = np.nan_to_num(np.asarray(vals, dtype=float))
            ax.bar(xs, v, width=bw, color=t.palette[0])
            for xi, vi, tx in zip(xs, v, texts):
                ax.text(xi, vi, tx, ha="center", va="bottom", fontsize="x-small", color=t.ink2)
            ax.set_xticks(xs, labels, fontsize="x-small")
            ax.set_title(name)
            grid_on(ax)
            if name == "final data loss" and np.all(v > 0) and v.max() / max(v.min(), 1e-300) > 20:
                ax.set_yscale("log")
            ax.margins(y=0.25)
        tot = (getattr(result, "timing", {}) or {}).get("total_s")
        fig.suptitle(title or "curriculum stages" + (f" (total {tot:.3g} s)" if tot else ""))
        return fig


# ---------------------------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------------------------
class StageSnapshots(Callback):
    """Keep detached CPU copies of every field at the end of each curriculum stage.

    The fields are stored at the stage's own resolution, which is what :func:`plot_multiscale`
    shows. With several restarts only the last restart's stages are kept.

    Args:
        names: fields to keep (default: all).
    """

    def __init__(self, names: Sequence[str] | None = None) -> None:
        self.names = tuple(names) if names else None
        self.snapshots: list[dict[str, Any]] = []
        self._last: dict[str, Any] | None = None
        self._step = 0

    def on_step(self, solver: Any, state: Any) -> None:
        if state.fields is not None:
            self._last = {k: v.detach() for k, v in state.fields.items()}
            self._step = int(state.global_step)

    def on_stage_end(self, solver: Any, stage_idx: int, stage: Any, info: dict) -> None:
        if stage_idx == 0:
            self.snapshots = []  # a new restart begins
        if self._last is None:
            return
        keep = {
            k: v.cpu().clone()
            for k, v in self._last.items()
            if self.names is None or k in self.names
        }
        self.snapshots.append(
            {
                "stage": int(stage_idx),
                "name": str(getattr(stage, "name", f"stage{stage_idx + 1}")),
                "shape": tuple(info.get("shape") or ()),
                "step": self._step,
                "fields": keep,
            }
        )
        self._last = None


def snapshot_list(snapshots: Any, field: str | None = None) -> list[tuple[int, np.ndarray]]:
    """``[(global_step, array)]`` from a :class:`~nefi.solve.callbacks.FieldSnapshots` /
    :class:`StageSnapshots` callback, their ``.snapshots`` list, ``(step, tensor)`` pairs or bare
    tensors."""
    snaps = getattr(snapshots, "snapshots", snapshots)
    out: list[tuple[int, np.ndarray]] = []
    for i, s in enumerate(snaps or []):
        if isinstance(s, Mapping) and "fields" in s:
            a, _ = resolve_field(s["fields"], field)
            out.append((int(s.get("step", i)), a))
        elif isinstance(s, tuple | list) and len(s) == 2:
            a, _ = resolve_field(s[1], field)
            out.append((int(s[0]), a))
        else:
            a, _ = resolve_field(s, field)
            out.append((i, a))
    return out


def _per_stage(snapshots: Any, result: Any, field: str | None) -> list[tuple[str, np.ndarray]]:
    """End-of-stage fields ``[(label, array)]`` from Stage- or FieldSnapshots."""
    snaps = getattr(snapshots, "snapshots", snapshots) or []
    _, sr = _history_of(result) if result is not None else ({}, [])
    if snaps and isinstance(snaps[0], Mapping) and "fields" in snaps[0]:
        out = []
        for s in snaps:
            a, _ = resolve_field(s["fields"], field)
            out.append((_stage_label(sr, s["stage"]) if sr else s.get("name", "stage"), a))
        return out
    lst = snapshot_list(snaps, field)
    if not lst:
        return []
    if result is None:
        return [(f"step {lst[-1][0]}", lst[-1][1])]
    arrays = history_arrays(result)
    x = _x_axis(arrays)
    st = arrays.get("stage")
    out = []
    for sidx, _row in _stage_starts(arrays):
        rows = np.where(st == sidx)[0] if st is not None else np.arange(len(x))
        end = x[rows[-1]] if rows.size else x[-1]
        cand = [(s, a) for s, a in lst if s <= end and (not rows.size or s >= x[rows[0]])]
        if cand:
            out.append((_stage_label(sr, sidx), cand[-1][1]))
    return out


def _draw_field_panel(
    ax: Axes, a: np.ndarray, spec, lo: float, hi: float, title: str, gt1d: np.ndarray | None = None
) -> None:
    t = theme()
    a = squeeze_field(a)
    if np.iscomplexobj(a):
        a = np.abs(a)
    if a.ndim == 1:
        xs = (np.arange(a.shape[0]) + 0.5) / a.shape[0]
        if gt1d is not None:
            ax.plot(
                (np.arange(gt1d.shape[0]) + 0.5) / gt1d.shape[0],
                gt1d,
                color=t.ink,
                lw=1.0,
                ls=(0, (4, 2)),
            )
        ax.plot(xs, a, color=t.palette[0], lw=1.3)
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        grid_on(ax)
        ax.tick_params(labelsize="x-small")
        ax.set_title(title)
        return
    if a.ndim == 3:
        a = take_slice(a, representative_slice(a))
    show_image(ax, a, spec=spec, vmin=lo, vmax=hi, extent=(0, 1, 0, 1), title=title)


def plot_multiscale(
    result: Any,
    snapshots: Any = None,
    *,
    gt: Any = None,
    field: str | None = None,
    history: bool = True,
    title: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """Coarse→fine view of a multiscale curriculum.

    Top row: the field at the end of every stage (each at its own resolution — pixels grow
    coarser for early stages), the final post-processed field and the ground truth. Bottom row
    (``history=True``): total / data loss with stage boundaries.

    Args:
        result: :class:`~nefi.solve.Result`.
        snapshots: :class:`StageSnapshots` (exact) or :class:`~nefi.solve.callbacks.FieldSnapshots`
            (last snapshot of each stage); ``None`` shows only the final field.
        gt: ground truth (tensor or fields dict).
        field: field to show (default: primary).
        history: add the loss panel.
        title: figure title.
        dark: dark theme.
    """
    final, name = resolve_field(result, field)
    stages = _per_stage(snapshots, result, name) if snapshots is not None else []
    g = resolve_field(gt, name)[0] if gt is not None else None
    panels = [(lab, a) for lab, a in stages] + [("final", final)]
    if g is not None:
        panels.append(("ground truth", g))
    ref = squeeze_field(g if g is not None else final)
    spec = cmap_for(name, ref)
    lo, hi = color_limits([np.abs(p) if np.iscomplexobj(p) else p for _, p in panels], spec)
    with styled(dark):
        t = theme()
        w, h = panel_size()
        ncols = len(panels)
        rows = 2 if history else 1
        from matplotlib.figure import Figure as _Figure

        fig = _Figure(
            figsize=(
                max(ncols * w * 1.05, w * 3.0) + 0.4,
                h * (1.0 + (0.9 if history else 0)) + 0.4,
            ),
            layout="constrained",
        )
        gs = fig.add_gridspec(rows, ncols, height_ratios=[1.0, 0.85] if history else [1.0])
        im_axes = []
        g1d = squeeze_field(g) if (g is not None and squeeze_field(g).ndim == 1) else None
        for j, (lab, a) in enumerate(panels):
            ax = fig.add_subplot(gs[0, j])
            shape_txt = "×".join(str(s) for s in squeeze_field(a).shape)
            label = lab if lab.endswith(shape_txt) else f"{lab}\n{shape_txt}"
            _draw_field_panel(ax, a, spec, lo, hi, label, g1d if lab != "ground truth" else None)
            im_axes.append(ax)
        if ref.ndim >= 2 and im_axes and im_axes[-1].images:
            from .fields import add_colorbar

            add_colorbar(fig, im_axes[-1].images[0], im_axes, label=name)
        if history:
            ax = fig.add_subplot(gs[1, :])
            arrays = history_arrays(result)
            _, sr = _history_of(result)
            if arrays.get("total") is not None and arrays["total"].size:
                x = _x_axis(arrays)
                ax.plot(x, arrays["total"], color=t.ink, lw=1.3, label="total")
                dl = arrays.get("data_loss")
                if dl is not None and not np.allclose(dl, arrays["total"], equal_nan=True):
                    ax.plot(x, dl, color=t.palette[0], lw=1.2, label="data loss")
                vals = arrays["total"][np.isfinite(arrays["total"])]
                if vals.size and np.all(vals > 0):
                    ax.set_yscale("log")
                _draw_stage_marks(ax, x, arrays, sr, labels=True)
                grid_on(ax)
                ax.set_xlabel("step")
                ax.set_ylabel("loss")
                ax.legend(loc="upper right")
            else:
                message_axes(ax, "no history")
        if not stages:
            fig.text(
                0.01,
                0.01,
                "no stage snapshots (pass StageSnapshots / FieldSnapshots)",
                fontsize="x-small",
                color=t.muted,
            )
        fig.suptitle(title or f"{name or 'field'}: coarse → fine curriculum")
        return fig


# ---------------------------------------------------------------------------------------------
# animation
# ---------------------------------------------------------------------------------------------
def _subsample(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    idx = np.unique(np.linspace(0, n - 1, k).round().astype(int))
    return [int(i) for i in idx]


def _thin(frames: list) -> list:
    """Every other frame, always keeping the last one (the final state)."""
    out = frames[::2]
    if frames and out[-1] is not frames[-1]:
        out.append(frames[-1])
    return out


def _save_gif(frames: list, path: Path, fps: float, max_kb: float | None) -> Path:
    """Write a GIF within ``max_kb``: frames are thinned (the last one is kept), then the
    palette shrinks (128 → 32 colours), then the frames are downscaled (×0.8 per attempt)."""
    from PIL import Image

    duration = int(round(1000.0 / max(fps, 0.1)))
    colors, scale = 128, 1.0
    cur = frames
    for _attempt in range(10):
        pal = []
        for f in cur:
            rgb = f.convert("RGB")
            if scale < 0.999:
                size = (max(8, int(rgb.width * scale)), max(8, int(rgb.height * scale)))
                rgb = rgb.resize(size, Image.Resampling.LANCZOS)
            pal.append(rgb.quantize(colors=colors, method=Image.Quantize.MEDIANCUT))
        pal[0].save(
            path,
            save_all=True,
            append_images=pal[1:],
            duration=duration,
            loop=0,
            optimize=True,
            disposal=1,
        )
        if max_kb is None or path.stat().st_size <= max_kb * 1024:
            break
        if len(cur) > 8:
            cur = _thin(cur)
        elif colors > 32:
            colors //= 2
        elif scale > 0.35:
            scale *= 0.8
        else:
            break
    if max_kb is not None and path.stat().st_size > max_kb * 1024:
        log.warning("%s is %.0f kB (> %s kB budget)", path, path.stat().st_size / 1024, max_kb)
    return path


def _volume_frames(
    frames: list[tuple[int, np.ndarray]],
    g: np.ndarray | None,
    name: str | None,
    spec: Any,
    lo: float,
    hi: float,
    arrays: Mapping[str, np.ndarray] | None,
    axis: int,
    n_slices: int,
    title: str | None,
    dpi: float,
    domain: Any = None,
) -> list:
    """Frames of a 3-D animation: the evolving depth mosaic (top row) above the GT mosaic, with
    the loss curve and a moving marker on the right. Snapshots at coarser curriculum stages are
    sliced at the same relative depths."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image

    from .volume import depth_label, label_inside, mosaic_slices

    t = theme()
    ref = squeeze_field(g) if g is not None else squeeze_field(frames[-1][1])
    ref = np.abs(ref) if np.iscomplexobj(ref) else ref
    ax_v = axis % 3
    n = ref.shape[ax_v]
    idx = mosaic_slices(n, n_slices)
    fracs = [(k + 0.5) / n for k in idx]
    has_loss = arrays is not None and "total" in arrays
    nrows = 1 + int(g is not None)
    k = len(idx)

    def sl(a: np.ndarray, frac: float) -> np.ndarray:
        a = squeeze_field(a)
        a = np.abs(a) if np.iscomplexobj(a) else a
        if a.ndim != 3:
            return a
        m = a.shape[ax_v]
        return take_slice(a, min(m - 1, int(frac * m)), ax_v)

    fig = Figure(
        figsize=(k * 1.2 + (2.3 if has_loss else 0.0) + 0.4, nrows * 1.25 + 0.55),
        layout="constrained",
    )
    gs = fig.add_gridspec(
        nrows, k + int(has_loss), width_ratios=[1.0] * k + ([2.0] if has_loss else [])
    )
    ims = []
    for j, frac in enumerate(fracs):
        ax = fig.add_subplot(gs[0, j])
        ims.append(
            show_image(ax, sl(frames[0][1], frac), spec=spec, vmin=lo, vmax=hi, extent=(0, 1, 0, 1))
        )
        label_inside(ax, depth_label(idx[j], n, ax_v, domain, compact=True))
        if j == 0:
            ax.set_ylabel("reconstruction", fontsize="x-small")
            ax.yaxis.set_visible(True)
            ax.set_yticks([])
        if g is not None:
            bx = fig.add_subplot(gs[1, j])
            show_image(bx, sl(g, frac), spec=spec, vmin=lo, vmax=hi, extent=(0, 1, 0, 1))
            if j == 0:
                bx.set_ylabel("ground truth", fontsize="x-small")
                bx.yaxis.set_visible(True)
                bx.set_yticks([])
    marker = None
    x = st = None
    if has_loss:
        axh = fig.add_subplot(gs[:, -1])
        x = _x_axis(arrays)
        st = arrays.get("stage")
        axh.plot(x, arrays["total"], color=t.ink2, lw=1.0)
        vals = arrays["total"][np.isfinite(arrays["total"])]
        if vals.size and np.all(vals > 0):
            axh.set_yscale("log")
        grid_on(axh)
        axh.set_title("total loss", fontsize="small")
        axh.tick_params(labelsize="x-small")
        marker = axh.plot([x[0]], [arrays["total"][0]], "o", color=t.palette[1], ms=4)[0]
    head = title or f"{name or 'field'} during optimization"
    ttl = fig.suptitle(head, fontsize="small")
    fig.set_dpi(dpi)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    fig.set_layout_engine("none")  # freeze the layout: later frames only swap pixel data
    images = []
    for step, a in frames:
        for im, frac in zip(ims, fracs):
            im.set_data(sl(a, frac).T)
        shape_txt = "×".join(str(s) for s in squeeze_field(a).shape)
        stage_txt = ""
        if marker is not None and x is not None and x.size:
            r = int(np.clip(np.searchsorted(x, step, side="right") - 1, 0, x.size - 1))
            marker.set_data([x[r]], [arrays["total"][r]])
            if st is not None:
                stage_txt = f" · stage {int(st[r]) + 1}"
        ttl.set_text(f"{head} — step {step}{stage_txt} · {shape_txt}")
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        images.append(Image.fromarray(buf[..., :3].copy()))
    return images


def animate_snapshots(
    snapshots: Any,
    path: str | Path,
    *,
    gt: Any = None,
    result: Any = None,
    field: str | None = None,
    fps: float = 8.0,
    max_frames: int = 40,
    dpi: float = 72,
    axis: int = -1,
    volume: str = "auto",
    n_slices: int = 4,
    domain: Any = None,
    title: str | None = None,
    max_kb: float | None = 300,
    dark: bool | None = None,
) -> Path:
    """Animate field snapshots (GIF, or MP4 when ffmpeg is available).

    Args:
        snapshots: :class:`~nefi.solve.callbacks.FieldSnapshots` (or its list of
            ``(step, tensor)``), :class:`StageSnapshots`, or a list of tensors.
        path: ``*.gif`` or ``*.mp4`` (MP4 falls back to GIF without ffmpeg).
        gt: ground truth shown next to the evolving field (and fixing the color scale).
        result: optional :class:`~nefi.solve.Result` — adds a loss curve with a moving marker.
        field: field name for dict snapshots / colormap choice.
        fps: frames per second.
        max_frames: frames kept (evenly subsampled, last frame always included).
        dpi: raster resolution of the frames.
        axis: slicing axis of 3-D fields (depth; the same relative depth is tracked across
            curriculum stages of different resolution).
        volume: 3-D fields: ``"mosaic"`` (default via ``"auto"``: ``n_slices`` evenly spaced
            depth slices of the evolving field above the same slices of the GT) or ``"slice"``
            (one slice through the GT anomaly centroid — the representative slice without GT).
            Mosaic animations are written as GIF.
        n_slices: slices of the 3-D mosaic.
        domain: :class:`~nefi.domain.Domain` (physical depth labels of the mosaic).
        title: figure title.
        max_kb: GIF size budget (frames are thinned — keeping the last —, colours reduced and
            frames downscaled until it is met).
        dark: dark theme.

    Returns:
        The written path.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = snapshot_list(snapshots, field)
    if not frames:
        raise ValueError("no snapshots to animate")
    frames = [frames[i] for i in _subsample(len(frames), max_frames)]
    name = field
    g = resolve_field(gt, field)[0] if gt is not None else None
    last = squeeze_field(frames[-1][1])
    ref = squeeze_field(g) if g is not None else last
    spec = cmap_for(name, ref)
    arrays_all = [
        np.abs(squeeze_field(a)) if np.iscomplexobj(a) else squeeze_field(a) for _, a in frames
    ]
    lo, hi = color_limits(([ref] if g is not None else []) + arrays_all, spec)
    frac = None
    arrays = history_arrays(result) if result is not None else None
    if ref.ndim == 3 and volume in ("auto", "mosaic"):
        with styled(dark):
            images = _volume_frames(
                frames, g, name, spec, lo, hi, arrays, axis, n_slices, title, dpi, domain
            )
        return _save_gif(images, path.with_suffix(".gif"), fps, max_kb)
    if ref.ndim == 3:
        if g is not None:
            from .fields import anomaly_centroid

            k = anomaly_centroid(ref)[axis % 3]
        else:
            k = representative_slice(ref, axis)
        frac = (k + 0.5) / ref.shape[axis]
    with styled(dark):
        t = theme()
        w, h = panel_size()
        ncols = 1 + int(g is not None) + int(arrays is not None and "total" in arrays)
        fig, axes = new_figure(1, ncols, figsize=(w * ncols * 1.1 + 0.2, h + 0.45))
        ax = axes[0, 0]

        def view(a: np.ndarray) -> np.ndarray:
            a = squeeze_field(a)
            if np.iscomplexobj(a):
                a = np.abs(a)
            if a.ndim == 3 and frac is not None:
                n = a.shape[axis]
                a = take_slice(a, min(n - 1, int(frac * n)), axis)
            return a

        a0 = view(frames[0][1])
        if a0.ndim == 1:
            line = ax.plot(
                (np.arange(a0.shape[0]) + 0.5) / a0.shape[0], a0, color=t.palette[0], lw=1.3
            )[0]
            if g is not None:
                gg = squeeze_field(g)
                ax.plot(
                    (np.arange(gg.shape[0]) + 0.5) / gg.shape[0],
                    gg,
                    color=t.ink,
                    lw=1.0,
                    ls=(0, (4, 2)),
                )
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            ax.set_xlim(0, 1)
            grid_on(ax)
            im = None
        else:
            im = ax.imshow(
                a0.T,
                cmap=get_cmap(spec),
                vmin=lo,
                vmax=hi,
                origin="lower",
                extent=(0, 1, 0, 1),
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            line = None
        col = 1
        if g is not None:
            gv = view(g)
            if gv.ndim == 1:
                axes[0, col].plot(
                    (np.arange(gv.shape[0]) + 0.5) / gv.shape[0], gv, color=t.ink, lw=1.2
                )
                axes[0, col].set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
                grid_on(axes[0, col])
                axes[0, col].set_title("ground truth")
            else:
                show_image(
                    axes[0, col],
                    gv,
                    spec=spec,
                    vmin=lo,
                    vmax=hi,
                    extent=(0, 1, 0, 1),
                    title="ground truth",
                )
            col += 1
        marker = None
        if arrays is not None and "total" in arrays:
            axh = axes[0, col]
            x = _x_axis(arrays)
            axh.plot(x, arrays["total"], color=t.ink2, lw=1.0)
            vals = arrays["total"][np.isfinite(arrays["total"])]
            if vals.size and np.all(vals > 0):
                axh.set_yscale("log")
            grid_on(axh)
            axh.set_title("total loss")
            axh.tick_params(labelsize="x-small")
            marker = axh.plot([x[0]], [arrays["total"][0]], "o", color=t.palette[1], ms=4)[0]
        st = arrays.get("stage") if arrays is not None else None
        gsteps = _x_axis(arrays) if arrays is not None else None
        ttl = fig.suptitle(title or f"{name or 'field'} during optimization")

        def update(step: int, a: np.ndarray) -> None:
            v = view(a)
            if line is not None:
                line.set_data((np.arange(v.shape[0]) + 0.5) / v.shape[0], v)
            elif im is not None:
                im.set_data(v.T)
            shape_txt = "×".join(str(s) for s in squeeze_field(a).shape)
            stage_txt = ""
            if st is not None and gsteps is not None and gsteps.size:
                r = int(
                    np.clip(np.searchsorted(gsteps, step, side="right") - 1, 0, gsteps.size - 1)
                )
                stage_txt = f" · stage {int(st[r]) + 1}"
                if marker is not None:
                    marker.set_data([gsteps[r]], [arrays["total"][r]])
            ax.set_title(f"step {step}{stage_txt}\n{shape_txt}", fontsize="small")
            ttl.set_text(title or f"{name or 'field'} during optimization")

        if path.suffix.lower() == ".mp4":
            try:
                import matplotlib as mpl
                import matplotlib.animation as manim

                if not manim.writers.is_available("ffmpeg"):
                    raise RuntimeError("ffmpeg not available")
                writer = manim.FFMpegWriter(fps=fps)
                with mpl.rc_context({"savefig.bbox": None}), writer.saving(fig, str(path), dpi):
                    for step, a in frames:
                        update(step, a)
                        writer.grab_frame()
                return path
            except Exception as e:
                log.warning("mp4 export unavailable (%s); writing a GIF instead", e)
                path = path.with_suffix(".gif")
        fig.set_dpi(dpi)
        canvas = FigureCanvasAgg(fig)
        images = []
        for i, (step, a) in enumerate(frames):
            update(step, a)
            canvas.draw()
            if i == 0:  # freeze the layout: later frames only swap data (much faster)
                fig.set_layout_engine("none")
            buf = np.asarray(canvas.buffer_rgba())
            images.append(Image.fromarray(buf[..., :3].copy()))
        return _save_gif(images, path, fps, max_kb)


__all__ = [
    "BOOKKEEPING",
    "StageSnapshots",
    "animate_snapshots",
    "history_arrays",
    "loss_keys",
    "plot_history",
    "plot_multiscale",
    "plot_stage_summary",
    "snapshot_list",
]
