"""The multi-physics gallery: one smoke run per registered instance, one compact tile per physical
system (GT | measurement | reconstruction, headline metric and wall-clock), a single overview
figure, per-instance figures and a JSON manifest.

Robustness: every instance runs inside its own ``try`` block — an instance that fails to import,
build, generate data, solve or plot becomes an error tile (with the exception text) and a
``"failed"`` manifest entry; the gallery itself always completes.

Budgets: problem sizes come from each instance's smoke preset (``nefi run --smoke`` lookup);
step counts are ``budget_scale ×`` the instance's *default* curriculum, floored by the preset's
own budget and capped by ``max_steps`` (:func:`nefi.viz._instances.budget_curriculum`).

3-D systems: ``gallery_3d.png`` shows depth mosaics, multi-level contours on the central slices
and shaded isosurfaces — the GT at its threshold rule (the instance's IoU rule), the
reconstruction at the volume-matched level (:mod:`nefi.viz.isosurface`); with ``interactive``
each 3-D system also gets a drag-to-rotate browser viewer (:mod:`nefi.viz.interactive`,
``<name>/viewer.html`` + ``viewer.json``) embedded in the reports right under its static row
(``<name>/block3d.png``).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import textwrap
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .fields import (
    _axis_names,
    add_colorbar,
    anomaly_centroid,
    compare_fields,
    depth_mosaic,
    plot_line,
    resolve_field,
    show_image,
    squeeze_field,
    to_numpy,
    voxel_view,
)
from .hints import compare_extras, volume_axis
from .measurement import draw_measurement, plot_fit, plot_measurement, resolve_layout, unpack
from .style import (
    color_limits,
    format_metric,
    grid_on,
    message_axes,
    primary_metric,
    savefig,
    styled,
    theme,
    use_style,
)
from .training import (
    StageSnapshots,
    animate_snapshots,
    plot_history,
    plot_multiscale,
    plot_stage_summary,
)
from .volume import (
    Outline,
    draw_projection,
    draw_sections,
    draw_slices,
    mosaic_shape,
    mosaic_slices,
    projection_mode,
)

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

log = logging.getLogger("nefi")

#: tile geometry (inches): three panels + a thin colorbar, two-line title.
TILE_PANEL = (1.3, 1.45)
#: 3-D systems are under-converged at gallery budgets: from this ``budget_scale`` on, their step
#: count is at least ``steps_3d_multiplier ×`` the smoke preset's (``run_instance_smoke``).
STEPS_3D_MIN_BUDGET = 0.2
STEPS_3D_MULTIPLIER = 2.0


@dataclass
class GalleryRun:
    """In-memory artifacts of one gallery run (not serialized)."""

    instance: Any
    info: dict[str, Any]
    gt: dict[str, Any] | None
    measurement: Any
    problem: Any
    result: Any
    stage_snapshots: Any = None
    field_snapshots: Any = None
    noise_floor: float | None = None
    refined: Any = None  # RunOutput of nefi.solve.refine_run_output (3-D, ``refine=True``)
    refine_view: Any = None  # the Result drawn as "refined" (the refused candidate if refused)


@dataclass
class GalleryEntry:
    """Manifest entry of one physical system (JSON-serializable except ``run``)."""

    name: str
    status: str = "pending"  # ok | failed
    description: str = ""
    scene_class: str | None = None
    smoke: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    steps: int = 0
    fields: dict[str, list[int]] = field(default_factory=dict)
    measurement_shape: list[int] = field(default_factory=list)
    layout: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    ms_per_step: float | None = None
    n_parameters: int | None = None
    config_hash: str = ""
    device: str = ""
    stops: list[str] = field(default_factory=list)
    error: str | None = None
    traceback: str | None = None
    figures: dict[str, str] = field(default_factory=dict)
    figure_errors: dict[str, str] = field(default_factory=dict)
    iso: dict[str, Any] = field(default_factory=dict)  # 3-D: GT-rule vs matched levels, IoU
    viewer: dict[str, Any] = field(default_factory=dict)  # 3-D: interactive viewer files, sizes
    refine: dict[str, Any] = field(default_factory=dict)  # 3-D: edge refinement verdict, numbers
    run: GalleryRun | None = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def headline(self) -> tuple[str, float] | None:
        """``(metric, value)`` shown on the tile."""
        return primary_metric(self.metrics)

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "run"}
        try:
            from ..bench.report import to_jsonable

            return to_jsonable(d)
        except ImportError:  # pragma: no cover
            return d


class GalleryManifest(dict):
    """The JSON manifest (a ``dict``) plus the in-memory :class:`GalleryEntry` list
    (``.entries``, with ``.run`` artifacts when ``keep_runs=True``)."""

    entries: list[GalleryEntry]


# ---------------------------------------------------------------------------------------------
# running one instance
# ---------------------------------------------------------------------------------------------
def run_instance_smoke(
    spec: Any,
    *,
    budget_scale: float = 0.15,
    device: str = "auto",
    seed: int = 0,
    smoke: bool = True,
    max_steps: int | None = 1500,
    min_steps: int = 20,
    time_budget_s: float | None = None,
    overrides: Mapping[str, Any] | None = None,
    scene_class: str | None = None,
    snapshot_frames: int = 30,
    steps_3d_multiplier: float | None = STEPS_3D_MULTIPLIER,
) -> GalleryEntry:
    """Generate → invert → evaluate one instance at a demo budget; never raises.

    Args:
        spec: registered name, ``Instance`` subclass or instance object.
        budget_scale: fraction of the instance's default step budget.
        device: solver device.
        seed: data and optimization seed.
        smoke: use the smoke preset (small grids / models).
        max_steps: cap on the total optimization steps.
        min_steps: floor when the instance has no smoke preset.
        time_budget_s: wall-clock cap of the solve (the solver stops cleanly).
        overrides: instance config overrides.
        scene_class: scene class (default: the instance's).
        snapshot_frames: approximate number of field snapshots kept for animations.
        steps_3d_multiplier: for 3-D fields at ``budget_scale ≥`` :data:`STEPS_3D_MIN_BUDGET`,
            at least this many times the smoke preset's steps (they are under-converged at the
            plain budget rule); it takes precedence over ``max_steps``. ``None`` / ``0``: off.

    Returns:
        A :class:`GalleryEntry` (``status`` ``"ok"`` or ``"failed"``) with ``run`` attached.
    """
    from ..bench.protocol import default_method, evaluate_metrics
    from ..solve.callbacks import FieldSnapshots
    from ..utils.seed import seed_everything
    from ._instances import budget_curriculum, mse_noise_floor, resolve_instance, shapes

    name = spec if isinstance(spec, str) else str(getattr(spec, "name", type(spec).__name__))
    entry = GalleryEntry(name=name)
    t_all = time.perf_counter()
    stage = "resolve"
    try:
        inst, info = resolve_instance(spec, smoke=smoke, overrides=overrides)
        entry.description = str(getattr(inst, "description", "") or "")
        entry.smoke = info.get("smoke")
        entry.config = dict(info.get("config") or {})
        stage = "generate"
        t0 = time.perf_counter()
        gt, meas = inst.make_measurement(seed=seed, scene_class=scene_class)
        entry.timing["generate_s"] = time.perf_counter() - t0
        try:
            entry.scene_class = scene_class or inst.default_scene_class()
        except Exception:  # pragma: no cover - instances without scene classes
            entry.scene_class = scene_class
        entry.fields = shapes(gt)
        entry.measurement_shape = [int(s) for s in meas.data.shape]
        stage = "build"
        seed_everything(seed)
        prep = default_method().prepare(inst, meas)
        cur, budget = budget_curriculum(
            prep.resolved_curriculum(),
            info,
            budget_scale,
            max_steps=max_steps,
            min_steps=min_steps,
            time_budget_s=time_budget_s,
        )
        dims = [int(d) for d in (next(iter(gt.values())).shape if gt else ()) if int(d) > 1]
        if steps_3d_multiplier and len(dims) == 3 and budget_scale >= STEPS_3D_MIN_BUDGET:
            floor = int(round(float(steps_3d_multiplier) * budget["preset_steps"]))
            budget["steps_3d_multiplier"] = float(steps_3d_multiplier)
            budget["steps_3d_floor"] = floor
            if cur.total_steps < floor:  # time_budget_s and restarts carry over
                cur = cur.scaled(floor / max(1, cur.total_steps))
                budget["steps"] = int(cur.total_steps)
                budget["stages"] = [
                    {
                        "name": st.name,
                        "shape": list(st.shape) if st.shape else None,
                        "steps": st.steps,
                    }
                    for st in cur.stages
                ]
        entry.budget = budget
        stage_snaps = StageSnapshots()
        field_snaps = FieldSnapshots(every=max(1, cur.total_steps // max(1, snapshot_frames)))
        stage = "solve"
        t0 = time.perf_counter()
        res = prep.run(cur, device=device, seed=seed, callbacks=[stage_snaps, field_snaps])
        entry.timing["solve_s"] = time.perf_counter() - t0
        entry.timing["per_stage_s"] = list(res.timing.get("per_stage_s", []))
        stage = "evaluate"
        t0 = time.perf_counter()
        entry.metrics = {k: float(v) for k, v in evaluate_metrics(inst, res, gt).items()}
        entry.timing["evaluate_s"] = time.perf_counter() - t0
        entry.steps = len(res.history.get("total", []))
        entry.ms_per_step = 1e3 * entry.timing["solve_s"] / entry.steps if entry.steps else None
        entry.n_parameters = res.extra.get("n_parameters")
        entry.config_hash = res.config_hash
        entry.device = str(res.extra.get("device", device))
        entry.stops = [str(s.get("stop", "?")) for s in res.stage_results]
        field_shape = next(iter(gt.values())).shape if gt else None
        try:
            _, mask, meta, _ = unpack(meas)
            from .measurement import instance_hints

            entry.layout = resolve_layout(meas.data, meta, field_shape, mask, instance_hints(inst))
        except Exception:  # pragma: no cover - layout is informative only
            entry.layout = None
        entry.run = GalleryRun(
            inst,
            info,
            {k: v for k, v in gt.items()} if gt else None,
            meas,
            prep.problem,
            res,
            stage_snaps,
            field_snaps,
            mse_noise_floor(prep.problem, meas),
        )
        entry.status = "ok"
    except Exception as e:
        entry.status = "failed"
        entry.error = f"{type(e).__name__} during {stage}: {e}"
        entry.traceback = traceback.format_exc(limit=12)
        log.warning("gallery: %s failed during %s: %s", name, stage, e)
    entry.timing["total_s"] = time.perf_counter() - t_all
    return entry


def _instance_metrics(run: GalleryRun, result: Any) -> dict[str, float]:
    from ..bench.protocol import evaluate_metrics

    return {k: float(v) for k, v in evaluate_metrics(run.instance, result, run.gt).items()}


def refine_entry(entry: GalleryEntry) -> dict[str, Any]:
    """Edge refinement of a 3-D gallery entry (``nefi.solve.refine_run_output``, see
    ``docs/refinement.md``); never raises.

    Stores the returned run on ``entry.run.refined`` and the field drawn in the "refined"
    column on ``entry.run.refine_view`` — the refined result, or the *refused candidate*
    (clearly labelled; the smooth result stays the answer). ``entry.metrics`` (the headline)
    stay the smooth result's. Returns (and sets) ``entry.refine``: verdict, the report's
    ``summary()`` (the caption), reason, mode, steps, seconds, the data fit before → after
    (``χ`` when the noise level is known, else the RMSE), IoU and Edge-F1 before → after (the
    instance's own metrics when it has them, else the refinement's at the GT threshold) and the
    instance metrics of the drawn field.
    """
    from .. import solve as _solve

    run = entry.run
    t0 = time.perf_counter()
    info: dict[str, Any] = {}
    try:
        refined = _solve.refine_run_output(entry)
        report = refined.extra["refine_report"]
        accepted = bool(getattr(report, "accepted", True))
        view = refined.result if accepted else (getattr(report, "candidate", None) or None)
        run.refined, run.refine_view = refined, view
        chi = getattr(report, "fit_measure", "chi") == "chi"
        fb, fa = getattr(report, "fit_before", {}) or {}, getattr(report, "fit_after", {}) or {}
        key = "chi" if chi else "rmse"
        info = {
            "accepted": accepted,
            "summary": str(report.summary()),
            "reason": str(getattr(report, "reason", "")),
            "mode": str(getattr(report, "mode", "")),
            "steps": int(getattr(report, "steps", 0) or 0),
            "seconds": float(getattr(report, "seconds", 0.0) or 0.0),
            "fit_measure": key,
            "fit_before": fb.get(key),
            "fit_after": fa.get(key),
        }
        drawn = dict(refined.metrics) if accepted else {}
        if not accepted and view is not None:
            try:
                drawn = _instance_metrics(run, view)
            except Exception as e:  # the report's own numbers remain
                log.debug("gallery: metrics of the refused candidate failed: %s", e)
        info["metrics"] = drawn
        mb = getattr(report, "metrics_before", None) or {}
        ma = getattr(report, "metrics_after", None) or {}
        for q in ("iou", "edge_f1"):
            if q in entry.metrics and q in drawn:
                info[f"{q}_before"], info[f"{q}_after"] = entry.metrics[q], drawn[q]
                info[f"{q}_source"] = "instance"
            elif q in mb and q in ma:
                info[f"{q}_before"], info[f"{q}_after"] = mb[q], ma[q]
                info[f"{q}_source"] = "refinement"
    except Exception as e:
        info = {"error": f"{type(e).__name__}: {e}"}
        entry.figure_errors["refine"] = info["error"]
        log.warning("gallery: %s refinement failed: %s", entry.name, e)
    entry.timing["refine_s"] = time.perf_counter() - t0
    entry.refine = info
    return info


def refine_line(refine: Mapping[str, Any], width: int | None = None) -> str:
    """``"refined ✓ IoU 0.59 → 0.82 · Edge-F1 0.42 → 1 · χ 1.15 → 1.10"`` (tile titles)."""
    if not refine or refine.get("error"):
        return f"refinement failed: {refine.get('error')}" if refine else ""

    def pair(b: Any, a: Any, digits: int = 2) -> str:
        try:
            return f"{float(b):.{digits}f} → {float(a):.{digits}f}"
        except (TypeError, ValueError):
            return "—"

    parts = []
    if "iou_before" in refine:
        parts.append(f"IoU {pair(refine['iou_before'], refine['iou_after'])}")
    if "edge_f1_before" in refine:
        parts.append(f"Edge-F1 {pair(refine['edge_f1_before'], refine['edge_f1_after'])}")
    lab = "χ" if refine.get("fit_measure") == "chi" else "RMSE"
    digits = 2 if lab == "χ" else 3
    parts.append(f"{lab} {pair(refine.get('fit_before'), refine.get('fit_after'), digits)}")
    head = "refined ✓" if refine.get("accepted") else "refinement refused (smooth kept):"
    text = head + " " + " · ".join(parts)
    return textwrap.fill(text, width) if width else text


# ---------------------------------------------------------------------------------------------
# tiles
# ---------------------------------------------------------------------------------------------
#: Mask keys looked up in ``Measurement.meta`` for GT outlines / anomaly centroids (first match
#: with the field's shape wins).
MASK_KEYS = ("defect_mask", "anomaly_mask", "support_mask", "support", "inclusion_mask")


def meta_mask(measurement: Any, shape: Sequence[int]) -> np.ndarray | None:
    """A field-shaped mask from ``measurement.meta`` (:data:`MASK_KEYS`), else ``None``."""
    meta = getattr(measurement, "meta", None) or {}
    for key in MASK_KEYS:
        v = meta.get(key) if isinstance(meta, Mapping) else None
        if v is None:
            continue
        try:
            m = squeeze_field(to_numpy(v)).astype(float)
        except (TypeError, ValueError):
            continue
        if tuple(m.shape) == tuple(shape):
            return m
    return None


@dataclass
class TileData:
    """The arrays a tile shows (after the instance's display hints)."""

    name: str | None
    label: str
    note: str
    g: np.ndarray | None
    rec: np.ndarray
    mask: np.ndarray | None
    spec: Any
    hints: dict[str, Any]
    domain: Any

    @property
    def volume(self) -> bool:
        return (self.g if self.g is not None else self.rec).ndim == 3

    def titles(self) -> tuple[str, str]:
        tag = f"\n({self.note})" if self.note else ""
        return f"ground truth{tag}", f"reconstruction{tag}"


def _domain_of(instance: Any) -> Any:
    try:
        return instance.domain()
    except Exception:  # pragma: no cover - domain is optional for plots
        return None


def tile_data(run: GalleryRun) -> TileData:
    """Primary field of a gallery run (GT and reconstruction on the GT grid), with the
    instance's ``field_transform`` / ``field_label`` / ``field_cmap`` hints applied; complex
    fields are shown by magnitude."""
    from ._instances import primary_field
    from .fields import _field_spec, match_shape
    from .hints import apply_transform, field_hint, instance_hints, transform_name, transform_note

    name = primary_field(run.result, run.gt)
    rec = squeeze_field(resolve_field(run.result, name)[0])
    g = squeeze_field(resolve_field(run.gt, name)[0]) if run.gt and name in run.gt else None
    if g is not None and rec.shape != g.shape:
        rec = match_shape(rec, g.shape)
    hn = instance_hints(run.instance)
    domain = _domain_of(run.instance)
    notes = []
    if np.iscomplexobj(g if g is not None else rec):
        g = np.abs(g) if g is not None else None
        rec = np.abs(rec)
        notes.append("|·|")
    tf = field_hint(hn, "field_transform", name)
    label = field_hint(hn, "field_label", name)
    if transform_name(tf) is not None:
        g = apply_transform(g, tf, domain=domain, instance=run.instance) if g is not None else None
        rec = apply_transform(rec, tf, domain=domain, instance=run.instance)
        notes.append(str(label) if label else transform_note(tf))
    elif label:
        notes.append(str(label))
    ref = g if g is not None else rec
    spec = _field_spec(name, ref, None, tf, field_hint(hn, "field_cmap", name))
    mask = meta_mask(run.measurement, ref.shape) if ref.ndim == 3 else None
    return TileData(
        name, str(label or name or "field"), " · ".join(notes), g, rec, mask, spec, hn, domain
    )


def display_field(run: GalleryRun, result: Any, td: TileData) -> np.ndarray:
    """``result``'s primary field as the tile shows it: on the GT grid, complex → magnitude,
    the instance's ``field_transform`` applied (like :func:`tile_data`'s reconstruction)."""
    from .fields import match_shape
    from .hints import apply_transform, field_hint, transform_name

    a = squeeze_field(resolve_field(result, td.name)[0])
    ref = td.g if td.g is not None else td.rec
    if a.shape != ref.shape:
        a = match_shape(a, ref.shape)
    if np.iscomplexobj(a):
        a = np.abs(a)
    tf = field_hint(td.hints, "field_transform", td.name)
    if transform_name(tf) is not None:
        a = apply_transform(a, tf, domain=td.domain, instance=run.instance)
    return a


def refined_view(run: GalleryRun, td: TileData) -> tuple[np.ndarray, str] | None:
    """``(array, label)`` of the "refined" column of a 3-D run (``None`` without refinement):
    ``"refined"``, or ``"refined — refused"`` for the refused candidate (the smooth result is
    kept)."""
    view = getattr(run, "refine_view", None)
    if view is None:
        return None
    rep = getattr(getattr(run, "refined", None), "extra", {}).get("refine_report")
    accepted = bool(getattr(rep, "accepted", True))
    return display_field(run, view, td), ("refined" if accepted else "refined — refused")


def is_volume_entry(entry: GalleryEntry) -> bool:
    """True for a successful entry whose primary field has three (non-singleton) dimensions."""
    if not entry.ok or entry.run is None:
        return False
    from ._instances import primary_field

    try:
        name = primary_field(entry.run.result, entry.run.gt)
        src = entry.run.gt if (entry.run.gt and name in entry.run.gt) else entry.run.result
        return squeeze_field(resolve_field(src, name)[0]).ndim == 3
    except Exception:  # pragma: no cover - a broken result is drawn as a flat tile
        return False


def tile_title(entry: GalleryEntry) -> str:
    """``name`` + a second line with the headline metric (the smooth result's), wall-clock and
    steps + (3-D systems with ``refine``) a third line: the refinement's verdict, IoU and
    Edge-F1 before → after and the data fit χ before → after (:func:`refine_line`)."""
    parts = []
    hl = entry.headline()
    if hl:
        parts.append(format_metric(*hl))
    solve = entry.timing.get("solve_s")
    if solve is not None:
        parts.append(f"{solve:.1f} s")
    if entry.steps:
        parts.append(f"{entry.steps} steps")
    title = entry.name + ("\n" + " · ".join(parts) if parts else "")
    extra = refine_line(entry.refine) if entry.refine else ""
    return title + (f"\n{extra}" if extra else "")


def _tile_measurement(ax: Any, entry: GalleryEntry, field_shape: Sequence[int]) -> None:
    run = entry.run
    try:
        draw_measurement(
            ax, run.measurement, field_shape=field_shape, instance=run.instance, title=""
        )
        ax.title.set_fontsize("small")
    except Exception as e:  # never let a measurement view break the gallery
        log.debug("tile measurement view failed for %s: %s", entry.name, e)
        message_axes(ax, f"measurement\n{tuple(entry.measurement_shape)}")
    head = ax.get_title()
    if head:
        ax.set_title(textwrap.fill(" ".join(head.split()), 26), fontsize="small")


def draw_tile(container: Any, entry: GalleryEntry) -> None:
    """Draw one tile (GT | measurement | reconstruction) into a figure or sub-figure.

    3-D systems get the volume tile (:func:`draw_volume_tile`); failed entries become a single
    panel with the error text.
    """
    t = theme()
    if not entry.ok or entry.run is None:
        ax = container.subplots(1, 1)
        msg = textwrap.fill(entry.error or "not run", 46)
        message_axes(ax, f"✗ failed\n\n{msg}", error=True)
        container.suptitle(entry.name, fontsize="medium", color=t.ink)
        return
    if is_volume_entry(entry):
        draw_volume_tile(container, entry)
        return
    run = entry.run
    axes = container.subplots(1, 3)
    td = tile_data(run)
    g, rec, spec = td.g, td.rec, td.spec
    lo, hi = color_limits([a for a in (g, rec) if a is not None], spec)
    im = None
    if rec.ndim == 1:
        if g is not None:
            plot_line(axes[0], g, color=t.ink)
            plot_line(axes[2], g, style="gt")
        else:
            message_axes(axes[0], "no ground truth")
        plot_line(axes[2], rec, color=t.palette[0])
        for ax in (axes[0], axes[2]):
            pad = 0.06 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
            grid_on(ax)
            ax.tick_params(labelsize="xx-small", length=1.5)
    else:
        if g is not None:
            show_image(axes[0], g, spec=spec, vmin=lo, vmax=hi)
        else:
            message_axes(axes[0], "no ground truth")
        im = show_image(axes[2], rec, spec=spec, vmin=lo, vmax=hi)
    gt_title, rec_title = td.titles()
    axes[0].set_title(gt_title, fontsize="small")
    axes[2].set_title(rec_title, fontsize="small")
    field_shape = next(iter(run.gt.values())).shape if run.gt else tuple(run.result.primary.shape)
    _tile_measurement(axes[1], entry, field_shape)
    if im is not None:
        cb = add_colorbar(container, im, [axes[0], axes[2]], nticks=3, shrink=0.8)
        cb.ax.tick_params(labelsize="xx-small")
    container.suptitle(tile_title(entry), fontsize="medium", color=t.ink)


def _mosaic_axes(sf: Any, k: int, max_cols: int) -> list:
    rows, cols = mosaic_shape(k, max_cols=max_cols)
    grid = sf.subplots(rows, cols, squeeze=False)
    flat = list(grid.ravel())
    for extra in flat[k:]:
        extra.remove()
    return flat[:k]


def draw_volume_tile(container: Any, entry: GalleryEntry, *, n_slices: int = 6) -> None:
    """The gallery row of a 3-D system (full overview width).

    ground-truth depth mosaic (up to ``n_slices`` evenly spaced slices, depth inside each panel)
    | measurement (compact view) | reconstruction mosaic with the GT outlined (with ``refine``:
    the smooth and the refined reconstruction, two mosaics) | cross-sections through the anomaly
    centroid (one row per volume: GT, smooth[, refined]) | anomaly-side projection (same rows) —
    one shared colour scale; the title carries the refinement's verdict and numbers.
    """
    t = theme()
    run = entry.run
    td = tile_data(run)
    g, rec = td.g, td.rec
    ref = g if g is not None else rec
    ax_v = volume_axis(td.hints) % 3
    letter = _axis_letter(td.domain, ax_v)
    n = ref.shape[ax_v]
    idx = mosaic_slices(n, n_slices)
    k = len(idx)
    _, cols = mosaic_shape(k, max_cols=3)
    center = anomaly_centroid(ref, mask=td.mask)
    outline = Outline.of(g, td.mask)
    mode = projection_mode(ref)
    rv = refined_view(run, td)
    recs = [("smooth" if rv else "reconstruction", rec)] + ([(rv[1], rv[0])] if rv else [])
    arrays = [a for a in (g, rec, rv[0] if rv else None) if a is not None]
    lo, hi = color_limits(arrays, td.spec)
    note = f" ({td.note})" if td.note else ""
    widths = [cols, 2.0] + [cols] * len(recs) + [2.0, 1.4]
    subs = container.subfigures(1, len(widths), width_ratios=widths, wspace=0.012, squeeze=False)[0]
    kw = {"axis": ax_v, "spec": td.spec, "vmin": lo, "vmax": hi, "domain": td.domain}
    gt_axes = _mosaic_axes(subs[0], k, 3)
    if g is not None:
        draw_slices(gt_axes, g, idx, labels="inside", **kw)
    else:
        for ax in gt_axes:
            message_axes(ax, "no GT")
    subs[0].suptitle(f"ground truth{note}\n{k} of {n} {letter}-slices", fontsize="small")
    ax_m = subs[1].subplots(1, 1)
    _tile_measurement(ax_m, entry, ref.shape)
    for j, (lab, vol) in enumerate(recs):
        axes = _mosaic_axes(subs[2 + j], k, 3)
        draw_slices(axes, vol, idx, labels="inside", outline=outline, **kw)
        sub = "GT outlined" if outline is not None else "depth slices"
        if lab == "refined — refused":
            lab, sub = "refused refinement", "not used · " + sub
        subs[2 + j].suptitle(f"{lab}{note}\n{sub}", fontsize="small")
    s_sec, s_pr = subs[2 + len(recs)], subs[3 + len(recs)]
    rows_ = [("GT", g, None)] + [(lab.split(" ")[0], vol, outline) for lab, vol in recs]
    sec = s_sec.subplots(len(rows_), 2, squeeze=False)
    for r, (lab, vol, ol) in enumerate(rows_):
        if vol is None:
            for ax in sec[r]:
                message_axes(ax, "no GT")
            continue
        draw_sections(sec[r], vol, center, titles=False, outline=ol, **kw)
        sec[r, 0].set_ylabel(lab, fontsize="x-small")
        sec[r, 0].yaxis.set_visible(True)
        sec[r, 0].set_yticks([])
    names = _axis_names(td.domain, 3)
    other = [d for d in range(3) if d != ax_v]
    sec[0, 0].set_title(f"{names[other[0]]}–{names[ax_v]} ↓", fontsize="x-small")
    sec[0, 1].set_title(f"{names[other[1]]}–{names[ax_v]} ↓", fontsize="x-small")
    s_sec.suptitle("cross-sections through\nthe anomaly centroid (+)", fontsize="small")
    pr = s_pr.subplots(len(rows_), 1, squeeze=False)[:, 0]
    im = None
    for r, (_, vol, ol) in enumerate(rows_):
        if vol is None:
            message_axes(pr[r], "no GT")
            continue
        im = draw_projection(
            pr[r],
            vol,
            axis=ax_v,
            mode=mode,
            spec=td.spec,
            vmin=lo,
            vmax=hi,
            domain=td.domain,
            outline=ol,
            title=False,
        )
    who = " / ".join(r[0] for r in rows_)
    s_pr.suptitle(f"{mode} projection\nover {letter} ({who})", fontsize="small")
    if im is not None:
        cb = add_colorbar(s_pr, im, list(pr), nticks=3, shrink=0.9)
        cb.ax.tick_params(labelsize="xx-small")
    container.suptitle(tile_title(entry), fontsize="medium", color=t.ink)


def _axis_letter(domain: Any, axis: int) -> str:
    return _axis_names(domain, 3)[axis % 3]


def entry_rule(run: GalleryRun, td: TileData | None = None) -> dict[str, Any]:
    """The GT threshold rule of a 3-D gallery run: the instance's IoU rule
    (:func:`~nefi.viz.isosurface.voxel_rule`) — or, after a display transform of the field,
    half-way from the GT background to its extreme."""
    from .isosurface import auto_rule, voxel_rule

    td = td or tile_data(run)
    ref = td.g if td.g is not None else td.rec
    if td.note:  # transformed / complex fields: the instance's rule applies to raw values
        return auto_rule(ref)
    return voxel_rule(run.instance, ref, field=td.name)


def draw_volume_block(container: Any, entry: GalleryEntry, *, n_slices: int = 8) -> None:
    """One system of ``gallery_3d.png``: larger GT and reconstruction depth mosaics (GT outlined
    on the reconstruction), multi-level contours of GT (solid) and reconstruction (dashed) on
    the central slice and the depth cross-section, and shaded isosurfaces — the GT at its
    threshold rule (the instance's IoU rule), the reconstruction at the **volume-matched**
    level (as many voxels as the GT; same camera; levels, IoU and Dice in the titles, with the
    IoU at the GT's own threshold for comparison). A ``volume_transform`` display hint is
    applied to the reconstruction's contours and isosurface and named in the titles.

    With an edge refinement (``refine``) every reconstruction panel comes twice — smooth and
    refined (or the refused candidate, labelled): GT | smooth | refined mosaics (6 slices), the
    contours of each on the central slice, and GT | smooth | refined isosurfaces, each
    reconstruction at its own volume-matched level, one camera."""
    import mpl_toolkits.mplot3d  # noqa: F401  (registers the 3d projection)

    from .isosurface import (
        draw_isosurfaces,
        draw_level_contours,
        hinted_transform,
        iso_levels,
        iso_titles,
        rule_level,
        transform_label,
        volume_transform,
    )

    t = theme()
    run = entry.run
    td = tile_data(run)
    g, rec = td.g, td.rec
    ref = g if g is not None else rec
    rv = refined_view(run, td)
    recs = [("smooth" if rv else "reconstruction", rec)] + ([(rv[1], rv[0])] if rv else [])
    ax_v = volume_axis(td.hints) % 3
    letter = _axis_letter(td.domain, ax_v)
    n = ref.shape[ax_v]
    idx = mosaic_slices(n, min(n_slices, 6) if rv else n_slices)
    k = len(idx)
    _, cols = mosaic_shape(k, max_cols=3 if rv else 4)
    outline = Outline.of(g, td.mask)
    lo, hi = color_limits([a for a in (g, *[a for _, a in recs]) if a is not None], td.spec)
    note = f" ({td.note})" if td.note else ""
    vt = hinted_transform(td.hints, td.name)
    disp = [volume_transform(a, vt) if vt else a for _, a in recs]
    dnote = f" ({transform_label(vt)})" if vt else ""
    widths = [cols] * (1 + len(recs)) + [1.5, 2.45 * (1 + len(recs))]
    subs = container.subfigures(1, len(widths), width_ratios=widths, wspace=0.012, squeeze=False)[0]
    kw = {"axis": ax_v, "spec": td.spec, "vmin": lo, "vmax": hi, "domain": td.domain}
    mcols = 3 if rv else 4
    if g is not None:
        draw_slices(_mosaic_axes(subs[0], k, mcols), g, idx, labels="inside", **kw)
    else:
        message_axes(subs[0].subplots(1, 1), "no ground truth")
    subs[0].suptitle(f"ground truth{note} — {k} of {n} {letter}-slices", fontsize="small")
    ims, rec_axes = [], []
    for j, (lab, vol) in enumerate(recs):
        rec_axes = _mosaic_axes(subs[1 + j], k, mcols)
        ims = draw_slices(rec_axes, vol, idx, labels="inside", outline=outline, **kw)
        head = "refused refinement (not used)" if lab == "refined — refused" else lab
        subs[1 + j].suptitle(
            f"{head}{note}" + (" — GT outlined" if outline is not None else ""), fontsize="small"
        )
    if ims:
        cb = add_colorbar(subs[len(recs)], ims[0], rec_axes, nticks=4, shrink=0.9, label=td.label)
        cb.ax.tick_params(labelsize="x-small")
    rule = entry_rule(run, td)
    s_con, s_iso = subs[1 + len(recs)], subs[2 + len(recs)]
    if g is not None:
        center = anomaly_centroid(g, mask=td.mask)
        if rv:  # the central slice, GT vs each reconstruction
            cax = list(s_con.subplots(len(recs), 1, squeeze=False)[:, 0])
            for j, ((lab, _), d) in enumerate(zip(recs, disp)):
                draw_level_contours(
                    [cax[j]],
                    g,
                    d,
                    center=center,
                    axis=ax_v,
                    domain=td.domain,
                    legend=j == len(recs) - 1,
                )
                cax[j].set_title(f"{lab.split(' ')[0]}: {cax[j].get_title()}", fontsize="x-small")
        else:
            cax = list(s_con.subplots(2, 1, squeeze=False)[:, 0])
            draw_level_contours(cax, g, disp[0], center=center, axis=ax_v, domain=td.domain)
        s_con.suptitle("contrast contours\nGT solid · recon dashed", fontsize="small")
        entries = []
        for (lab, _), d in zip(recs, disp):
            lv = iso_levels(g, d, rule)
            gt_t, rec_t = iso_titles(lab + dnote, lv, "matched", td.label)
            if not entries:
                entries.append((gt_t, g, lv["gt_level"], lv["side"]))
            entries.append((rec_t, d, lv["levels"]["matched"], lv["side"]))
    else:
        message_axes(s_con.subplots(1, 1), "no ground truth")
        entries = [
            (lab + dnote, d, rule_level(rule, d), rule["side"]) for (lab, _), d in zip(recs, disp)
        ]
    vax = [s_iso.add_subplot(1, len(entries), i + 1, projection="3d") for i in range(len(entries))]
    draw_isosurfaces(vax, entries, domain=td.domain, max_faces=4000, zoom=1.0)
    s_iso.suptitle(
        "isosurfaces: GT at its threshold rule, reconstruction"
        + ("s" if len(recs) > 1 else "")
        + " at the volume-matched level",
        fontsize="small",
    )
    container.suptitle(tile_title(entry), fontsize="medium", color=t.ink)


#: Height (inches) of a 3-D row in the overview and of a ``gallery_3d`` block; minimum width of
#: figures holding 3-D rows.
VOLUME_ROW_HEIGHT = 3.25
VOLUME_BLOCK_HEIGHT = 3.35
VOLUME_ROW_MIN_WIDTH = 12.5
#: Extra height / width of 3-D rows and blocks with a refined column (three title lines, a
#: third row of cross-sections, three isosurfaces).
REFINED_EXTRA_HEIGHT = 0.6
REFINED_EXTRA_WIDTH = 3.0


def has_refined(entry: GalleryEntry) -> bool:
    """True when the entry carries a refined (or refused-candidate) field to draw."""
    return bool(entry.run is not None and getattr(entry.run, "refine_view", None) is not None)


def tile_figure(entry: GalleryEntry, dark: bool | None = None) -> Figure:
    """A stand-alone figure with one tile (a full-width row for 3-D systems)."""
    from matplotlib.figure import Figure

    with styled(dark):
        w, h = TILE_PANEL
        if is_volume_entry(entry):
            extra = has_refined(entry)
            fig = Figure(
                figsize=(
                    VOLUME_ROW_MIN_WIDTH + 0.5 + REFINED_EXTRA_WIDTH * extra,
                    VOLUME_ROW_HEIGHT + 0.1 + REFINED_EXTRA_HEIGHT * extra,
                ),
                layout="constrained",
            )
        else:
            fig = Figure(figsize=(3 * w + 0.6, h + 0.55), layout="constrained")
        draw_tile(fig, entry)
        return fig


def volume_block_figure(
    entry: GalleryEntry, *, n_slices: int = 8, dark: bool | None = None
) -> Figure:
    """One 3-D system's block of ``gallery_3d.png`` as its own figure (``<name>/block3d.png``:
    the static row shown above the system's interactive viewer in reports)."""
    from matplotlib.figure import Figure

    extra = has_refined(entry)
    with styled(dark):
        fig = Figure(
            figsize=(
                15.0 + REFINED_EXTRA_WIDTH * extra,
                VOLUME_BLOCK_HEIGHT + 0.3 + REFINED_EXTRA_HEIGHT * extra,
            ),
            layout="constrained",
        )
        draw_volume_block(fig, entry, n_slices=n_slices)
        return fig


def iso_summary(entry: GalleryEntry) -> dict[str, Any]:
    """Level-set comparison of a 3-D entry (raw fields, no display transform): the GT threshold
    rule and, for the reconstruction, the level, voxel count, IoU and Dice at the GT threshold
    (``fixed``), at the volume-matched level and at Otsu's level
    (:func:`~nefi.viz.isosurface.iso_levels`); ``refined``: the same for the refined field.
    ``{}`` without a ground truth."""
    from .isosurface import iso_levels

    td = tile_data(entry.run)
    if td.g is None:
        return {}
    rule = entry_rule(entry.run, td)
    lv = iso_levels(td.g, td.rec, rule)
    out = {
        "rule": rule.get("source"),
        "mode": rule.get("mode"),
        "side": lv["side"],
        "gt_level": lv["gt_level"],
        "gt_count": lv["gt_count"],
        "levels": lv["levels"],
        "counts": lv["counts"],
        "iou": lv["iou"],
        "dice": lv["dice"],
    }
    rv = refined_view(entry.run, td)
    if rv is not None:  # the refined (or refused-candidate) field under the same rules
        lr = iso_levels(td.g, rv[0], rule)
        out["refined"] = {k: lr[k] for k in ("levels", "counts", "iou", "dice")}
        out["refined"]["label"] = rv[1]
    return out


def physics_overview(
    entries: Sequence[GalleryEntry],
    *,
    ncols: int = 3,
    title: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """One figure with a tile per physical system (failed systems show their error).

    1-D / 2-D systems (and failures) fill a grid of ``ncols`` tiles; every 3-D system gets a
    full-width row below it: GT depth mosaic | measurement | reconstruction mosaic (GT outlined)
    | cross-sections through the anomaly | projection (:func:`draw_volume_tile`; with an edge
    refinement, smooth and refined columns).
    """
    from matplotlib.figure import Figure

    entries = list(entries)
    vols = [e for e in entries if is_volume_entry(e)]
    flat = [e for e in entries if e not in vols]
    ncols = max(1, min(ncols, max(1, len(flat))))
    with styled(dark):
        w, h = TILE_PANEL
        tw, th = 3 * w + 0.6, h + 0.6
        W = ncols * tw
        if vols:
            wide = any(has_refined(e) for e in vols)
            W = max(W, VOLUME_ROW_MIN_WIDTH + 0.5 + REFINED_EXTRA_WIDTH * wide)
            ncols = max(ncols, int(round(W / tw)))  # flat tiles keep their proportions
        n_flat_rows = math.ceil(len(flat) / ncols) if flat else 0
        heights = ([n_flat_rows * th] if flat else []) + [
            VOLUME_ROW_HEIGHT + REFINED_EXTRA_HEIGHT * has_refined(e) for e in vols
        ]
        heights = heights or [th]
        fig = Figure(figsize=(W, sum(heights) + 0.45), layout="constrained")
        blocks = fig.subfigures(len(heights), 1, height_ratios=heights, squeeze=False)[:, 0]

        def safe(container: Any, e: GalleryEntry) -> None:
            try:
                draw_tile(container, e)
            except Exception as err:  # pragma: no cover - a tile must never kill the overview
                log.warning("tile %s failed: %s", e.name, err)
                ax = container.subplots(1, 1)
                message_axes(ax, f"{e.name}: tile failed\n{err}", error=True)

        b = 0
        if flat:
            subs = blocks[0].subfigures(n_flat_rows, ncols, squeeze=False, wspace=0.02, hspace=0.02)
            for i, e in enumerate(flat):
                safe(subs[i // ncols, i % ncols], e)
            b = 1
        for e in vols:
            safe(blocks[b], e)
            b += 1
        n_ok = sum(e.ok for e in entries)
        extra = f" · {len(vols)} volumetric" if vols else ""
        fig.suptitle(
            title or f"nefi physics gallery — {n_ok}/{len(entries)} systems reconstructed{extra}",
            fontsize="x-large",
        )
        return fig


def physics_volumes(
    entries: Sequence[GalleryEntry],
    *,
    n_slices: int = 8,
    title: str | None = None,
    dark: bool | None = None,
) -> Figure | None:
    """``gallery_3d.png``: one block per 3-D system (larger depth mosaics of GT and
    reconstruction, contrast contours and side-by-side isosurfaces at matched levels,
    :func:`draw_volume_block`); ``None`` when no entry is volumetric."""
    from matplotlib.figure import Figure

    vols = [e for e in entries if is_volume_entry(e)]
    if not vols:
        return None
    heights = [VOLUME_BLOCK_HEIGHT + REFINED_EXTRA_HEIGHT * has_refined(e) for e in vols]
    width = 15.0 + REFINED_EXTRA_WIDTH * any(has_refined(e) for e in vols)
    with styled(dark):
        fig = Figure(figsize=(width, sum(heights) + 0.45), layout="constrained")
        blocks = fig.subfigures(len(vols), 1, squeeze=False, hspace=0.03, height_ratios=heights)[
            :, 0
        ]
        for blk, e in zip(blocks, vols):
            try:
                draw_volume_block(blk, e, n_slices=n_slices)
            except Exception as err:  # pragma: no cover - a block must never kill the figure
                log.warning("3-D block %s failed: %s", e.name, err)
                message_axes(blk.subplots(1, 1), f"{e.name}: 3-D view failed\n{err}", error=True)
        fig.suptitle(
            title
            or "nefi 3-D systems — depth slices, contrast contours and isosurfaces "
            "(GT at its threshold rule, reconstruction volume-matched)"
            + ("; smooth vs edge-refined" if any(has_refined(e) for e in vols) else ""),
            fontsize="x-large",
        )
        return fig


# ---------------------------------------------------------------------------------------------
# per-instance figures
# ---------------------------------------------------------------------------------------------
def _headline_metrics(metrics: Mapping[str, float], n: int = 2) -> dict[str, float]:
    from .qualitative import _headline

    return _headline(metrics, n)


def instance_figures(
    entry: GalleryEntry,
    out_dir: str | Path,
    *,
    formats: Sequence[str] = ("png",),
    dpi: float = 130,
    max_kb: float | None = 300,
    animate: bool = False,
    dark: bool | None = None,
) -> dict[str, str]:
    """Write the standard figures of one gallery entry into ``out_dir/<name>/``.

    Files: ``tile``, ``compare`` (+ ``compare_<field>`` for other GT fields), ``measurement``,
    ``fit``, ``history``, ``stages``, ``multiscale`` (multi-stage curricula), ``mosaic`` and
    ``voxels`` (3-D fields: the reconstruction's depth mosaic with the GT outlined; GT vs
    reconstruction isosurfaces — GT at its threshold rule, reconstruction at the
    volume-matched level — with contrast contours, :func:`~nefi.viz.isosurface.iso_compare`)
    and ``evolution.gif`` (``animate=True``; 3-D: the depth mosaic).
    ``compare`` shows the signed error by default and the GT overlay for 3-D fields (the
    ``compare_panel`` display hint overrides both). Figure failures are recorded in
    ``entry.figure_errors`` and never raised.

    Returns:
        ``{figure: path relative to out_dir}`` (also stored in ``entry.figures``).
    """
    out = Path(out_dir)
    d = out / entry.name
    run = entry.run
    figs: dict[str, str] = {}

    def save(key: str, make: Callable[[], Any]) -> None:
        try:
            with styled(dark):
                fig = make()
                paths = savefig(fig, d / key, formats, dpi, max_kb=max_kb)
            figs[key] = os.path.relpath(paths[0], out)
        except Exception as e:  # figures are best effort
            entry.figure_errors[key] = f"{type(e).__name__}: {e}"
            log.warning("gallery: %s/%s failed: %s", entry.name, key, e)

    save("tile", lambda: tile_figure(entry))
    if run is None or not entry.ok:
        entry.figures.update(figs)
        return figs
    from ._instances import primary_field

    name = primary_field(run.result, run.gt)
    domain = _domain_of(run.instance)
    field_shape = next(iter(run.gt.values())).shape if run.gt else None
    label = "nefi"
    td = tile_data(run)
    extras = compare_extras(td.hints, td.rec.ndim)
    methods, mets = {label: run.result}, {label: _headline_metrics(entry.metrics)}
    view = getattr(run, "refine_view", None)
    if view is not None:  # 3-D with an edge refinement: smooth and refined rows
        rl = "refined" if entry.refine.get("accepted", True) else "refused refinement"
        methods = {"smooth": run.result, rl: view}
        mets = {
            "smooth": _headline_metrics(entry.metrics),
            rl: _headline_metrics(entry.refine.get("metrics") or {}),
        }
    save(
        "compare",
        lambda: compare_fields(
            run.gt,
            methods,
            run.measurement,
            field=name,
            metrics=mets,
            panels=("measurement", "gt", "recon", *extras),
            domain=domain,
            mask=td.mask,
            instance=run.instance,
            title=f"{entry.name}: {td.label} — reconstruction vs ground truth"
            + (f" ({td.note})" if td.note and td.note != td.label else ""),
        ),
    )
    extra = [k for k in (run.gt or {}) if k != name and k in run.result.fields][:2]
    for k in extra:
        save(
            f"compare_{k}",
            lambda k=k: compare_fields(
                run.gt,
                {label: run.result},
                None,
                field=k,
                domain=domain,
                instance=run.instance,
                title=f"{entry.name}: {k} (secondary field)",
            ),
        )
    save(
        "measurement",
        lambda: plot_measurement(
            run.measurement,
            field_shape=field_shape,
            instance=run.instance,
            title=f"{entry.name}: measurement {tuple(entry.measurement_shape)}",
        ),
    )
    save(
        "fit",
        lambda: plot_fit(
            run.result.pred,
            run.measurement,
            field_shape=field_shape,
            instance=run.instance,
        ),
    )
    save("history", lambda: plot_history(run.result, noise_floor=run.noise_floor))
    save("stages", lambda: plot_stage_summary(run.result))
    if len(run.result.stage_results) > 1:
        save(
            "multiscale",
            lambda: plot_multiscale(run.result, run.stage_snapshots, gt=run.gt, field=name),
        )
    if td.volume:
        ax_v = volume_axis(td.hints)
        save(
            "mosaic",
            lambda: depth_mosaic(
                td.rec,
                field=name,
                axis=ax_v,
                domain=domain,
                spec=td.spec,
                mask=td.mask,
                title=f"{entry.name}: reconstructed {td.label}"
                + (" (GT defects outlined)" if td.mask is not None else ""),
            ),
        )
        if td.g is not None:
            from .isosurface import hinted_transform, iso_compare

            rv = refined_view(run, td)
            recs = {"smooth": td.rec, rv[1]: rv[0]} if rv else {"reconstruction": td.rec}
            save(
                "voxels",
                lambda: iso_compare(
                    td.g,
                    recs,
                    field=name,
                    rule=entry_rule(run, td),
                    domain=domain,
                    mask=td.mask,
                    axis=volume_axis(td.hints),
                    transform=hinted_transform(td.hints, name),
                    hints={"field_transform": None},  # tile_data already applied it
                    title=f"{entry.name}: {td.label} — isosurfaces, GT at its threshold rule "
                    "vs reconstruction at the volume-matched level",
                ),
            )
        else:
            save("voxels", lambda: voxel_view(td.rec, field=name, domain=domain))
    if animate and run.field_snapshots is not None and run.field_snapshots.snapshots:
        try:
            with styled(dark):
                gt_arr = run.gt.get(name) if (run.gt and name in run.gt) else None
                p = animate_snapshots(
                    run.field_snapshots,
                    d / "evolution.gif",
                    gt=gt_arr,
                    result=run.result,
                    field=name,
                    axis=volume_axis(td.hints),
                    domain=domain,
                    max_kb=max_kb,
                )
            figs["evolution"] = os.path.relpath(p, out)
        except Exception as e:
            entry.figure_errors["evolution"] = f"{type(e).__name__}: {e}"
            log.warning("gallery: %s animation failed: %s", entry.name, e)
    entry.figures.update(figs)
    return figs


# ---------------------------------------------------------------------------------------------
# the gallery
# ---------------------------------------------------------------------------------------------
def summary_rows(entries: Sequence[GalleryEntry]) -> list[dict[str, Any]]:
    """One table row per system: status, scene, grid, steps, timings, headline metrics."""
    rows = []
    for e in entries:
        hl = e.headline()
        grid = next(iter(e.fields.values()), None)
        rows.append(
            {
                "system": e.name,
                "status": e.status,
                "scene": e.scene_class,
                "grid": grid,
                "measurement": e.measurement_shape or None,
                "layout": e.layout,
                "steps": e.steps or None,
                "solve s": e.timing.get("solve_s"),
                "ms/step": e.ms_per_step,
                "metric": hl[0] if hl else None,
                "value": hl[1] if hl else None,
                "error": (e.error or "")[:90] or None,
            }
        )
    return rows


def physics_gallery(
    instances: Any = None,
    budget_scale: float = 0.15,
    device: str = "auto",
    out_dir: str | Path = "runs/gallery",
    *,
    seed: int = 0,
    smoke: bool = True,
    max_steps: int | None = 1500,
    min_steps: int = 20,
    time_budget_s: float | None = None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    scene_classes: Mapping[str, str] | None = None,
    per_instance: bool = True,
    animate: bool = False,
    ncols: int = 3,
    formats: Sequence[str] = ("png",),
    dpi: float = 130,
    max_kb: float | None = 300,
    dark: bool = False,
    html: bool = False,
    title: str | None = None,
    keep_runs: bool = True,
    on_entry: Callable[[GalleryEntry], None] | None = None,
    interactive: bool | str | None = True,
    refine: bool = False,
    steps_3d_multiplier: float | None = STEPS_3D_MULTIPLIER,
) -> GalleryManifest:
    """Run every available instance at a small budget and draw the multi-physics gallery.

    Args:
        instances: names / classes / objects, a comma-separated string, or ``None`` for every
            registered instance (in :data:`nefi.instances.available` order).
        budget_scale: fraction of each instance's default step budget (see module docs).
        device: solver device (``"auto"``, ``"cpu"``, ``"cuda"``).
        out_dir: output directory (``gallery.png``, ``gallery_3d.png`` when a system is
            volumetric, ``manifest.json``, ``<name>/*.png``).
        seed: data and optimization seed.
        smoke: use the instances' smoke presets (small grids / models).
        max_steps: cap on each instance's total optimization steps.
        min_steps: floor for instances without a smoke preset.
        time_budget_s: per-instance wall-clock cap of the solve.
        overrides: ``{name: {config_key: value}}``.
        scene_classes: ``{name: scene_class}``.
        per_instance: write the per-instance figures (:func:`instance_figures`).
        animate: also write ``<name>/evolution.gif``.
        ncols: tiles per row of the overview.
        formats: figure formats (``("png", "svg")`` ...).
        dpi: raster resolution (≤ 150).
        max_kb: PNG size budget.
        dark: dark theme.
        html: also write ``gallery.html`` (self-contained report of the gallery).
        title: overview title.
        keep_runs: keep the in-memory results on ``manifest.entries[i].run``.
        on_entry: callback after each system (e.g. progress printing).
        interactive: interactive viewer of every 3-D system: ``True`` / ``"canvas"`` (the
            dependency-free viewer of :mod:`nefi.viz.interactive`), ``"plotly"`` (CDN-backed
            plotly isosurfaces, not self-contained) or ``False`` / ``None`` / ``"none"``. Writes
            ``<name>/viewer.json`` (embedded by :func:`gallery_sections`), ``<name>/viewer.html``
            (standalone) and ``<name>/block3d.png`` (the system's static row).
        refine: after each 3-D system's smooth solve, run the edge refinement
            (:func:`refine_entry`, ``nefi.solve.refine_run_output``) and draw the refined field
            as a third column (GT | smooth | refined) in the overview rows, ``gallery_3d.png``,
            ``compare`` / ``voxels`` and the viewer; the smooth result stays the headline.
        steps_3d_multiplier: 3-D systems get at least this many times their smoke preset's
            steps when ``budget_scale ≥`` :data:`STEPS_3D_MIN_BUDGET` (``None``: off).

    Returns:
        :class:`GalleryManifest` — the JSON manifest (also written to ``manifest.json``) with
        ``.entries`` attached.
    """
    from ._instances import as_list, registered_instances
    from .interactive import normalize_kind
    from .report import environment_info, write_json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = as_list(instances) or registered_instances()
    kind = normalize_kind(interactive)
    entries: list[GalleryEntry] = []
    t0 = time.perf_counter()
    with use_style(dark=dark):
        for spec in specs:
            name = spec if isinstance(spec, str) else getattr(spec, "name", str(spec))
            log.info("gallery: running %s", name)
            entry = run_instance_smoke(
                spec,
                budget_scale=budget_scale,
                device=device,
                seed=seed,
                smoke=smoke,
                max_steps=max_steps,
                min_steps=min_steps,
                time_budget_s=time_budget_s,
                overrides=(overrides or {}).get(str(name)),
                scene_class=(scene_classes or {}).get(str(name)),
                steps_3d_multiplier=steps_3d_multiplier,
            )
            if refine and is_volume_entry(entry):
                refine_entry(entry)
            if per_instance:
                tp = time.perf_counter()
                instance_figures(
                    entry, out, formats=formats, dpi=dpi, max_kb=max_kb, animate=animate
                )
                entry.timing["plots_s"] = time.perf_counter() - tp
            if is_volume_entry(entry):
                volume_extras(entry, out, kind, formats=formats, dpi=dpi, max_kb=max_kb)
            entries.append(entry)
            if on_entry is not None:
                on_entry(entry)
        overview_path = volumes_path = None
        try:
            fig = physics_overview(entries, ncols=ncols, title=title)
            overview_path = savefig(fig, out / "gallery", formats, dpi, max_kb=max_kb)[0]
        except Exception as e:  # pragma: no cover - keep the manifest even without overview
            log.warning("gallery overview failed: %s", e)
        try:
            fig3 = physics_volumes(entries)
            if fig3 is not None:
                volumes_path = savefig(fig3, out / "gallery_3d", formats, dpi, max_kb=max_kb)[0]
        except Exception as e:  # pragma: no cover - keep the manifest even without it
            log.warning("gallery 3-D overview failed: %s", e)
    manifest = GalleryManifest(
        {
            "title": title or "nefi physics gallery",
            "budget_scale": budget_scale,
            "device": device,
            "seed": seed,
            "smoke": smoke,
            "max_steps": max_steps,
            "time_budget_s": time_budget_s,
            "overview": None if overview_path is None else os.path.relpath(overview_path, out),
            "overview_3d": None if volumes_path is None else os.path.relpath(volumes_path, out),
            "volumetric": [e.name for e in entries if is_volume_entry(e)],
            "interactive": kind,
            "refine": bool(refine),
            "steps_3d_multiplier": steps_3d_multiplier,
            "n_ok": sum(e.ok for e in entries),
            "n_failed": sum(not e.ok for e in entries),
            "total_s": time.perf_counter() - t0,
            "summary": summary_rows(entries),
            "entries": [e.to_dict() for e in entries],
            "environment": environment_info(device),
        }
    )
    write_json(out / "manifest.json", dict(manifest))
    manifest.entries = entries
    if html:
        from .report import html_report

        html_report(
            out,
            manifest["title"],
            gallery_sections(manifest, out, interactive=kind is not None),
            filename="gallery.html",
            subtitle=f"{manifest['n_ok']}/{len(entries)} systems · budget ×{budget_scale:g} · "
            f"device {device}",
        )
    if not keep_runs:
        for e in entries:
            e.run = None
    return manifest


def volume_extras(
    entry: GalleryEntry,
    out_dir: str | Path,
    kind: str | None = "canvas",
    *,
    formats: Sequence[str] = ("png",),
    dpi: float = 130,
    max_kb: float | None = 300,
) -> None:
    """3-D extras of a gallery entry: the level-set comparison (``entry.iso``) and, with an
    interactive ``kind``, the static row ``<name>/block3d.png`` plus the viewer files
    (``entry.viewer``, :func:`~nefi.viz.interactive.write_entry_viewer`). Never raises."""
    out = Path(out_dir)
    t0 = time.perf_counter()
    try:
        entry.iso = iso_summary(entry)
    except Exception as e:
        entry.figure_errors["iso"] = f"{type(e).__name__}: {e}"
        log.warning("gallery: %s level comparison failed: %s", entry.name, e)
    if kind is not None:
        try:
            paths = savefig(
                volume_block_figure(entry),
                out / entry.name / "block3d",
                formats,
                dpi,
                max_kb=max_kb,
            )
            entry.figures["block3d"] = os.path.relpath(paths[0], out)
        except Exception as e:
            entry.figure_errors["block3d"] = f"{type(e).__name__}: {e}"
            log.warning("gallery: %s/block3d failed: %s", entry.name, e)
        try:
            from .interactive import write_entry_viewer

            entry.viewer = write_entry_viewer(entry, out, kind=kind)
        except Exception as e:
            entry.figure_errors["viewer"] = f"{type(e).__name__}: {e}"
            log.warning("gallery: %s viewer failed: %s", entry.name, e)
    entry.timing["volume_extras_s"] = time.perf_counter() - t0


#: Per-instance figures embedded in reports by default (``evolution`` GIFs and ``stages`` stay
#: on disk; the training-dynamics section of ``examples/gallery.py`` embeds them for two systems).
REPORT_FIGURES = ("compare", "measurement", "fit", "history", "multiscale", "mosaic", "voxels")
#: Captions of 3-D figures whose file names predate their content.
FIGURE_CAPTIONS = {"voxels": "voxels: isosurfaces, GT threshold rule vs volume-matched level"}


def iso_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per 3-D system: the reconstruction's level, IoU and Dice at the GT threshold
    and at the volume-matched level (and Otsu's IoU) — from the manifest's ``iso`` entries."""
    rows = []
    for e in manifest.get("entries", []):
        iso = e.get("iso") or {}
        if not iso:
            continue
        lv, iou, dice = iso.get("levels") or {}, iso.get("iou") or {}, iso.get("dice") or {}
        src = str(iso.get("rule") or "")
        kind = "IoU rule" if src.startswith("IoU rule") else "hint" if "hint" in src else "half-way"
        sym = "<" if iso.get("side") == "below" else ">"
        gt = (
            "per volume"
            if iso.get("mode") == "relative"
            else f"{sym} {fmt_level(iso.get('gt_level'))}"
        )
        rows.append(
            {
                "system": e.get("name"),
                "GT rule": f"{kind} ({'half max' if iso.get('mode') == 'relative' else gt})",
                "GT voxels": iso.get("gt_count"),
                "IoU at GT threshold": iou.get("fixed"),
                "Dice at GT threshold": dice.get("fixed"),
                "matched level": lv.get("matched"),
                "IoU matched": iou.get("matched"),
                "Dice matched": dice.get("matched"),
                "IoU Otsu": iou.get("otsu"),
            }
        )
    return rows


def _before_after(r: Mapping[str, Any], q: str) -> str | None:
    if f"{q}_before" not in r:
        return None
    return f"{fmt_level(r[f'{q}_before'])} → {fmt_level(r[f'{q}_after'])}"


def refine_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per refined 3-D system: verdict, mode, data fit, IoU and Edge-F1 before → after,
    the refined field's volume-matched IoU and the cost — from the manifest's ``refine``."""
    rows = []
    for e in manifest.get("entries", []):
        r = e.get("refine") or {}
        if not r:
            continue
        if r.get("error"):
            rows.append({"system": e.get("name"), "verdict": "failed", "error": r["error"]})
            continue

        lab = "χ" if r.get("fit_measure") == "chi" else "RMSE"
        ref = ((e.get("iso") or {}).get("refined") or {}).get("iou") or {}
        rows.append(
            {
                "system": e.get("name"),
                "verdict": "accepted" if r.get("accepted") else "refused (smooth kept)",
                "mode": r.get("mode"),
                "data fit": f"{lab} {fmt_level(r.get('fit_before'))} → "
                f"{fmt_level(r.get('fit_after'))}",
                "IoU": _before_after(r, "iou"),
                "Edge-F1": _before_after(r, "edge_f1"),
                "IoU matched (refined)": ref.get("matched"),
                "steps": r.get("steps"),
                "seconds": r.get("seconds"),
            }
        )
    return rows


def fmt_level(v: Any) -> str:
    """``3 significant digits`` or ``—``."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{f:.3g}" if math.isfinite(f) else "—"


_VOLUME_TEXT = (
    "Every volumetric system at a larger scale: depth slices of the ground truth and of the "
    "reconstruction (GT outlined), contrast contours of both on the central slice and the depth "
    "cross-section (solid: GT, dashed: reconstruction, at 25 / 50 / 75 % of the GT contrast — "
    "soft reconstructed edges show as spread-out dashed contours) and shaded isosurfaces: the "
    "GT cut by its threshold rule (the instance's IoU rule, else half-way from its background "
    "to its extreme), the reconstruction at the **volume-matched** level, which encloses as "
    "many voxels as the GT does. Reconstructions recover the structure with softer edges, so "
    "the GT's own threshold under- or over-segments them; the table compares IoU / Dice at the "
    "GT threshold (the instance's IoU metric) and at the matched level."
)
_REFINE_TEXT = (
    "After each 3-D system's smooth solve, `nefi.solve.refine_run_output` sharpens its "
    "interfaces under the same forward operator and data (`docs/refinement.md`) and keeps the "
    "result only when the data fit does not degrade and the interfaces stay inside the smooth "
    "solution's transition band. A refused refinement keeps the smooth result; its candidate is "
    'drawn labelled "refused" for inspection. The systems\' headline metrics stay the smooth '
    "results'; IoU / Edge-F1 are the instance's own metrics where it has them. Refinement is a "
    "prior: an accepted field is consistent with the data, not proven by them."
)
_VIEWER_TEXT = (
    "Under each static row, an interactive viewer (self-contained, no network): drag to "
    "rotate, wheel or pinch to zoom, right- or shift-drag to pan, double-click to reset. Views: "
    "isosurface, voxels, slice (movable x / y / z) and combinations; the GT threshold, the "
    "reconstruction's level rule (volume-matched, GT threshold, Otsu, manual) and a display "
    "transform (smooth, sharpen, edge-preserving — labelled display) can be changed live; IoU / "
    "Dice are computed in the browser. The same static rows are in `gallery_3d.png`."
)


def volume_section(
    manifest: Mapping[str, Any],
    out_dir: str | Path,
    *,
    interactive: bool = True,
    height: int = 420,
) -> Any:
    """The "3-D systems" report section: with ``interactive`` and viewer files, one subsection
    per 3-D system — its static row (``block3d.png``) and right under it the interactive viewer
    (script and stylesheet included once) — else ``gallery_3d.png``; the level table
    (:func:`iso_rows`) either way. ``None`` without 3-D systems."""
    from .report import Section, slug

    out = Path(out_dir)
    names = list(manifest.get("volumetric") or [])
    vols = [e for e in manifest.get("entries", []) if e.get("name") in names]
    rows = iso_rows(manifest) or None
    viewers = [
        e
        for e in vols
        if interactive
        and (e.get("viewer") or {}).get("data")
        and (out / e["viewer"]["data"]).exists()
    ]
    if viewers:
        from .interactive import viewer_fragment_from_file

        subs, first = [], {"canvas": True, "plotly": True}
        for e in vols:
            v, figs = e.get("viewer") or {}, e.get("figures") or {}
            frag, kind = "", v.get("kind", "canvas")
            if e in viewers:
                frag = viewer_fragment_from_file(
                    out / v["data"],
                    height=height,
                    include_assets=first["canvas"] and kind == "canvas",
                    include_plotlyjs=first["plotly"] and kind == "plotly",
                )
                first[kind] = False
            size = v.get("fragment_bytes")
            note = (
                "Plotly isosurfaces (loads plotly.js from its CDN: not self-contained)."
                if kind == "plotly"
                else f"Interactive viewer ({size / 1024:.0f} kB embedded)."
                if size
                else ""
            )
            imgs = [{"path": out / figs["block3d"], "wide": True}] if figs.get("block3d") else []
            text = (
                f"Static row (`{figs.get('block3d', 'block3d.png')}`), then the viewer; "
                f"standalone page `{v.get('html', '')}`. {note}".strip()
            )
            cap = (e.get("refine") or {}).get("summary")
            subs.append(
                Section(
                    e["name"],
                    text + (f"\n\nEdge refinement: `{cap}`" if cap else ""),
                    images=imgs,
                    html=frag,
                    id=f"volumes-3d-{slug(e['name'])}",
                )
            )
        return Section(
            "3-D systems",
            _VOLUME_TEXT + "\n\n" + _VIEWER_TEXT,
            table=rows,
            subsections=_refine_subsection(manifest) + subs,
            id="volumes-3d",
        )
    if manifest.get("overview_3d"):
        return Section(
            "3-D systems",
            _VOLUME_TEXT,
            images=[{"path": out / manifest["overview_3d"], "wide": True}],
            table=rows,
            subsections=_refine_subsection(manifest),
            id="volumes-3d",
        )
    return None


def _refine_subsection(manifest: Mapping[str, Any]) -> list[Any]:
    """The "Edge refinement" subsection (table + captions), or nothing without refinements."""
    from .report import Section

    rows = refine_rows(manifest)
    if not rows:
        return []
    caps = [
        f"- `{(e.get('refine') or {}).get('summary')}`"
        for e in manifest.get("entries", [])
        if (e.get("refine") or {}).get("summary")
    ]
    return [
        Section(
            "Edge refinement",
            _REFINE_TEXT + ("\n\n" + "\n".join(caps) if caps else ""),
            table=rows,
            id="volumes-3d-refinement",
        )
    ]


def gallery_sections(
    manifest: Mapping[str, Any],
    out_dir: str | Path,
    include: Sequence[str] | None = REPORT_FIGURES,
    *,
    interactive: bool = True,
    viewer_height: int = 420,
) -> list[Any]:
    """Report sections of a gallery: overview + summary table, the 3-D systems
    (:func:`volume_section`), then one section per system.

    Args:
        manifest: :func:`physics_gallery` manifest (or its ``manifest.json`` loaded).
        out_dir: the gallery directory (figure paths in the manifest are relative to it).
        include: per-instance figure keys to embed (``None`` = all, incl. GIF animations).
        interactive: embed the 3-D viewers written by :func:`physics_gallery` (HTML reports;
            Markdown ignores them — use ``False`` there to get ``gallery_3d.png``).
        viewer_height: canvas height of the viewers (CSS pixels).
    """
    from .report import Section

    out = Path(out_dir)
    secs = [
        Section(
            "Multi-physics overview",
            "One tile per physical system: ground truth | measurement (compact view) | "
            "reconstruction, with the headline metric, solve time and steps. Volumetric "
            "systems get a full-width row: GT depth mosaic | measurement | reconstruction "
            "mosaic (GT outlined) | cross-sections through the anomaly centroid | projection. "
            "Failed systems show their error.",
            images=[{"path": out / manifest["overview"], "wide": True}]
            if manifest.get("overview")
            else [],
            table=manifest.get("summary"),
            id="overview",
        )
    ]
    vol = volume_section(manifest, out, interactive=interactive, height=viewer_height)
    if vol is not None:
        secs.append(vol)
    for e in manifest.get("entries", []):
        figs = e.get("figures") or {}
        order = [
            "compare",
            "measurement",
            "fit",
            "history",
            "multiscale",
            "stages",
            "mosaic",
            "voxels",
            "evolution",
        ]
        skip = ("tile", "block3d")  # block3d: shown in the 3-D systems section
        keys = [k for k in order if k in figs] + [
            k for k in figs if k not in order and k not in skip
        ]
        if include is not None:
            keys = [k for k in keys if k in include or k.split("_")[0] in include]
        imgs = [
            {
                "path": out / figs[k],
                "caption": FIGURE_CAPTIONS.get(k, k) if e.get("iso") else k,
                "wide": k in ("compare", "measurement", "fit"),
            }
            for k in keys
        ]
        mt = [{"metric": k, "value": v} for k, v in (e.get("metrics") or {}).items()]
        t = e.get("timing") or {}
        if e.get("status") == "ok":
            solve = float(t.get("solve_s", float("nan")))
            gen = float(t.get("generate_s", float("nan")))
            detail = (
                f"Scene `{e.get('scene_class')}` · {e.get('steps')} steps · solve {solve:.2f} s"
                f" · data {gen:.2f} s · preset `{e.get('smoke')}` · config hash "
                f"`{e.get('config_hash')}`"
            )
        else:
            detail = f"**Failed:** `{e.get('error')}`"
        text = f"{e.get('description') or ''}\n\n{detail}".strip()
        cap = (e.get("refine") or {}).get("summary")
        if cap:
            text += f"\n\nEdge refinement: `{cap}` (headline metrics: the smooth result)."
        secs.append(
            Section(
                e["name"],
                text,
                images=imgs,
                table=mt or None,
                code=(e.get("traceback") or "") if e.get("status") != "ok" else "",
            )
        )
    return secs


__all__ = [
    "FIGURE_CAPTIONS",
    "MASK_KEYS",
    "REPORT_FIGURES",
    "VOLUME_BLOCK_HEIGHT",
    "VOLUME_ROW_HEIGHT",
    "GalleryEntry",
    "GalleryManifest",
    "GalleryRun",
    "TileData",
    "draw_tile",
    "draw_volume_block",
    "draw_volume_tile",
    "display_field",
    "entry_rule",
    "gallery_sections",
    "has_refined",
    "instance_figures",
    "iso_rows",
    "iso_summary",
    "is_volume_entry",
    "meta_mask",
    "physics_gallery",
    "physics_overview",
    "physics_volumes",
    "refine_entry",
    "refine_line",
    "refine_rows",
    "refined_view",
    "run_instance_smoke",
    "summary_rows",
    "tile_data",
    "tile_figure",
    "tile_title",
    "volume_block_figure",
    "volume_extras",
    "volume_section",
]
