"""Qualitative comparison figures: the scenes × methods grid of a paper, failure-mode panels and
a one-call baselines panel.

* :func:`method_grid` — rows = scenes, columns = methods (+ GT / measurement), one shared color
  scale per row, metrics under every method, optional signed-error rows (NeTMY Fig. 3 / NeFTY
  Fig. 4 style).
* :func:`failure_modes` — residual maps that expose typical failure modes: **leakage** (mass
  outside the true support), **cross / streak artifacts** (error energy concentrated on the
  Fourier axes, typical of separable or limited-angle operators) and ringing, with the numbers
  in the titles (:func:`failure_stats`).
* :func:`baselines_panel` — run the instance's own method and its ``baselines()`` at a small
  budget on one measurement and draw the grid.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .fields import (
    add_colorbar,
    field_metrics,
    image_extent,
    match_shape,
    metrics_label,
    plot_line,
    representative_slice,
    resolve_field,
    show_image,
    squeeze_field,
    take_slice,
    to_numpy,
)
from .style import (
    cmap_for,
    color_limits,
    figsize,
    format_metric,
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

GT_KEYS = ("GT", "gt", "ground truth", "ground_truth", "truth")
MEAS_KEYS = ("measurement", "data", "meas", "y")


# ---------------------------------------------------------------------------------------------
# method grid
# ---------------------------------------------------------------------------------------------
def _is_measurement(v: Any) -> bool:
    return hasattr(v, "data") and hasattr(v, "noise_std")


def _cell_array(v: Any, field: str | None) -> np.ndarray | None:
    if v is None or isinstance(v, str) or _is_measurement(v):
        return None
    a, _ = resolve_field(v, field)
    a = squeeze_field(a)
    return np.abs(a) if np.iscomplexobj(a) else a


def _find_key(row: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for k in keys:
        if k in row:
            return k
    return None


def method_grid(
    rows: Sequence[Mapping[str, Any]],
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    *,
    field: str | None = None,
    gt_key: str | None = "auto",
    metrics: Any = "auto",
    error: bool = False,
    shared: str = "row",
    quantity: str | None = None,
    instance: Any = None,
    axis: int = -1,
    title: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """The classic qualitative figure: scenes (rows) × methods (columns).

    Args:
        rows: one dict per scene, ``{column: value}``; values may be tensors, arrays,
            :class:`~nefi.solve.Result`\\ s, fields dicts, a :class:`~nefi.measurement.Measurement`
            (drawn with its compact view), an error string (drawn as a message) or ``None``.
        row_labels: scene labels (left of every row).
        col_labels: column order (default: keys in first-seen order; GT and measurement first).
        field: field to show from Results / dicts.
        gt_key: column holding the ground truth (``"auto"``: ``GT``/``gt``/``ground truth``;
            ``None``: no GT → no metrics / errors).
        metrics: ``"auto"`` (PSNR + SSIM, or PSNR + rel. error in 1-D), ``{name: fn}``, a single
            callable, ``None``, or precomputed ``[{column: {metric: value}}]`` (one per row).
        error: add a signed-error row (``RdBu_r``) under every scene.
        shared: color scale shared per ``"row"`` (default), over ``"all"`` rows, or ``"none"``.
        quantity: colormap quantity override.
        instance: optional instance for measurement cells (display hints).
        axis: slicing axis of 3-D fields (representative slice of the row's GT).
        title: figure title.
        dark: dark theme.
    """
    rows = [dict(r) for r in rows]
    if not rows:
        raise ValueError("method_grid needs at least one row")
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    if col_labels is not None:
        cols = [c for c in col_labels if any(c in r for r in rows)]
    gk = None
    if gt_key == "auto":
        gk = next((k for k in GT_KEYS if k in cols), None)
    elif gt_key is not None:
        gk = gt_key if gt_key in cols else None
    mk = next((k for k in MEAS_KEYS if k in cols), None)
    front = [c for c in (gk, mk) if c is not None]
    cols = front + [c for c in cols if c not in front]
    methods = [c for c in cols if c not in front]
    precomputed = isinstance(metrics, Sequence) and not isinstance(metrics, str)
    if callable(metrics) and not isinstance(metrics, Mapping):
        metrics = {"metric": metrics}
    nsc = len(rows)
    nrows = nsc * (2 if (error and gk) else 1)
    ncols = len(cols)
    with styled(dark):
        t = theme()
        fig, axes = new_figure(
            nrows, ncols, figsize=figsize(ncols, nrows, cbar=1, title=bool(title))
        )
        # 1-D fields are drawn as lines: widen the panels
        all_lims = None
        if shared == "all":
            vals = [_cell_array(r.get(c), field) for r in rows for c in cols]
            ref = next((v for v in vals if v is not None), None)
            if ref is not None:
                all_lims = color_limits(
                    [v for v in vals if v is not None], cmap_for(field, ref, quantity)
                )
        for i, row in enumerate(rows):
            r0 = i * (2 if (error and gk) else 1)
            gt = _cell_array(row.get(gk), field) if gk else None
            arrays = {c: _cell_array(row.get(c), field) for c in cols}
            ref = (
                gt if gt is not None else next((a for a in arrays.values() if a is not None), None)
            )
            sl = None
            if ref is not None and ref.ndim == 3:
                sl = representative_slice(ref, axis)
            if ref is not None and gt is not None:
                arrays = {
                    c: (match_shape(a, gt.shape) if (a is not None and a.shape != gt.shape) else a)
                    for c, a in arrays.items()
                }
            spec = cmap_for(field, ref, quantity) if ref is not None else None
            present = [a for a in arrays.values() if a is not None]
            lims = all_lims or (
                color_limits(present, spec) if (shared != "none" and present) else (None, None)
            )
            if precomputed:
                row_m = dict(metrics[i]) if i < len(metrics) else {}
            elif gt is not None and metrics is not None:
                row_m = {
                    c: field_metrics(arrays[c], gt, metrics)
                    for c in methods
                    if arrays.get(c) is not None
                }
            else:
                row_m = {}
            im_axes, ims, err_axes, err_ims = [], [], [], []
            errs = {
                c: arrays[c] - gt for c in methods if gt is not None and arrays.get(c) is not None
            }
            elim = color_limits(list(errs.values()), cmap_for(quantity="error")) if errs else None
            for j, c in enumerate(cols):
                ax = axes[r0, j]
                v = row.get(c)
                head = c if i == 0 else None
                label = head
                if c in methods and row_m.get(c):
                    txt = metrics_label(row_m[c], 2)
                    label = f"{head}\n{txt}" if head else txt
                if v is None:
                    message_axes(ax, "—", label)
                elif isinstance(v, str):
                    message_axes(ax, v[:120], label, error=True)
                elif _is_measurement(v):
                    from .measurement import draw_measurement

                    try:
                        draw_measurement(
                            ax,
                            v,
                            field_shape=None if gt is None else gt.shape,
                            instance=instance,
                            title=label or "",
                        )
                    except Exception as e:  # never break the grid on a measurement
                        message_axes(ax, f"measurement\n{type(e).__name__}", label)
                else:
                    a = arrays[c]
                    if a.ndim == 1:
                        if gt is not None and c != gk:
                            plot_line(ax, gt, style="gt")
                        plot_line(ax, a, color=t.ink if c == gk else t.palette[0])
                        if lims[0] is not None:
                            pad = 0.06 * (lims[1] - lims[0])
                            ax.set_ylim(lims[0] - pad, lims[1] + pad)
                        grid_on(ax)
                        ax.tick_params(labelsize="x-small")
                        if label:
                            ax.set_title(label)
                    else:
                        a2 = take_slice(a, min(sl, a.shape[axis] - 1), axis) if a.ndim == 3 else a
                        ims.append(
                            show_image(ax, a2, spec=spec, vmin=lims[0], vmax=lims[1], title=label)
                        )
                        im_axes.append(ax)
                if j == 0 and row_labels is not None and i < len(row_labels):
                    ax.set_ylabel(str(row_labels[i]), fontsize="small", color=t.ink)
                    ax.yaxis.set_visible(True)
                if error and gk:
                    eax = axes[r0 + 1, j]
                    e = errs.get(c)
                    if e is None:
                        eax.remove()
                        continue
                    if e.ndim == 1:
                        eax.axhline(0.0, color=t.axis, lw=0.6)
                        plot_line(eax, e, color=t.palette[1])
                        grid_on(eax)
                        eax.tick_params(labelsize="x-small")
                    else:
                        e2 = take_slice(e, min(sl, e.shape[axis] - 1), axis) if e.ndim == 3 else e
                        err_ims.append(
                            show_image(
                                eax,
                                e2,
                                spec=cmap_for(quantity="error"),
                                vmin=elim[0],
                                vmax=elim[1],
                                title="error" if i == 0 else None,
                            )
                        )
                        err_axes.append(eax)
            if ims and shared != "none":
                add_colorbar(fig, ims[0], im_axes)
            if err_ims:
                add_colorbar(fig, err_ims[0], err_axes)
        if title:
            fig.suptitle(title)
        return fig


# ---------------------------------------------------------------------------------------------
# failure modes
# ---------------------------------------------------------------------------------------------
def _support(gt: np.ndarray, threshold: float, dilate: int) -> np.ndarray:
    g = np.abs(gt - np.median(gt)) if np.min(gt) > 0.05 * np.max(np.abs(gt)) else np.abs(gt)
    mx = float(g.max()) if g.size else 0.0
    sup = g > threshold * mx if mx > 0 else np.zeros(g.shape, dtype=bool)
    if dilate > 0 and sup.any():
        from scipy.ndimage import binary_dilation

        sup = binary_dilation(sup, iterations=int(dilate))
    return sup


def axis_excess(err: np.ndarray) -> float:
    """Cross / streak-artifact indicator of an error map.

    Ratio of the error's Fourier power on the axes ``k_x = 0`` / ``k_y = 0`` to the power an
    isotropic error would have there — the radial profile of the off-axis power, interpolated in
    log space at the on-axis radii. A 2-D Hann window suppresses the FFT boundary cross and the
    lowest frequencies (``|k| ≤ 1.5``) are ignored. ≈ 1 for isotropic errors (white noise, blobs),
    ≫ 1 for axis-aligned streaks (separable / limited-angle operators), < 1 for diagonal ones.
    """
    if err.ndim != 2 or min(err.shape) < 6:
        return float("nan")
    win = np.outer(np.hanning(err.shape[0]), np.hanning(err.shape[1]))
    P = np.abs(np.fft.fft2((err - err.mean()) * win)) ** 2
    kx = np.fft.fftfreq(err.shape[0]) * err.shape[0]
    ky = np.fft.fftfreq(err.shape[1]) * err.shape[1]
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.hypot(KX, KY)
    axes_ = (KX == 0) | (KY == 0)
    on, off = axes_ & (K > 1.5), ~axes_ & (K > 1.5)
    if not on.any() or not off.any():
        return float("nan")
    tiny = 1e-12 * max(float(P.max()), 1e-300)
    edges = np.arange(0.0, float(K.max()) + 0.75, 0.5)
    b = np.digitize(K[off], edges)
    ks = [float(K[off][b == i].mean()) for i in np.unique(b)]
    vs = [float(np.log(P[off][b == i].mean() + tiny)) for i in np.unique(b)]
    expect = np.exp(np.interp(K[on], ks, vs))
    return float(P[on].sum() / max(float(expect.sum()), tiny))


def failure_stats(
    pred: Any, gt: Any, *, support_threshold: float = 0.05, dilate: int = 1
) -> dict[str, float]:
    """Failure-mode numbers of a reconstruction.

    Returns:
        ``relative_error``; ``leakage`` — fraction of the reconstruction's absolute mass outside
        the (dilated) ground-truth support ``|gt − background| > threshold · max``; ``cross`` —
        :func:`axis_excess` of the error (≈ 1 isotropic, ≫ 1 cross / streak artifacts).
    """
    p = squeeze_field(to_numpy(pred))
    g = squeeze_field(to_numpy(gt))
    p = np.abs(p) if np.iscomplexobj(p) else p
    g = np.abs(g) if np.iscomplexobj(g) else g
    if p.shape != g.shape:
        p = match_shape(p, g.shape)
    if g.ndim == 3:
        k = representative_slice(g)
        p, g = take_slice(p, k), take_slice(g, k)
    e = p - g
    rel = float(np.linalg.norm(e) / (np.linalg.norm(g) or 1e-30))
    sup = _support(g, support_threshold, dilate)
    bg = float(np.median(g)) if np.min(g) > 0.05 * np.max(np.abs(g)) else 0.0
    mass = np.abs(p - bg)
    leak = float(mass[~sup].sum() / (mass.sum() or 1e-30)) if sup.any() else float("nan")
    cross = axis_excess(e) if e.ndim == 2 else float("nan")
    return {"relative_error": rel, "leakage": leak, "cross": cross}


def failure_modes(
    gt: Any,
    recons: Any,
    *,
    field: str | None = None,
    support_threshold: float = 0.05,
    dilate: int = 1,
    spectrum: bool = True,
    title: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """Residual maps that expose failure modes, one row per method.

    Columns (2-D; 3-D uses the representative slice): reconstruction with the GT support
    outlined | signed error | leakage map (reconstruction outside the dilated support) | error
    power spectrum (log, centred) — cross / streak artifacts appear as bright axes and raise the
    :func:`axis_excess` in the title. 1-D fields:
    overlay | signed error | error spectrum. Titles carry :func:`failure_stats`.

    Args:
        gt: ground truth (tensor / fields dict).
        recons: ``{method: reconstruction}`` (tensors, Results or fields dicts) or one of them.
        field: field name.
        support_threshold: support = ``|gt − background| > threshold · max``.
        dilate: support dilation (pixels) before measuring leakage.
        spectrum: include the error-spectrum column.
        title: figure title.
        dark: dark theme.
    """
    from .fields import _recon_items, draw_contour

    g, name = resolve_field(gt, field)
    g = squeeze_field(g)
    g = np.abs(g) if np.iscomplexobj(g) else g
    items = [
        (lab, squeeze_field(np.abs(a) if np.iscomplexobj(a) else a))
        for lab, a in _recon_items(recons, field or name)
    ]
    items = [(lab, match_shape(a, g.shape) if a.shape != g.shape else a) for lab, a in items]
    if not items:
        raise ValueError("failure_modes needs at least one reconstruction")
    if g.ndim == 3:
        k = representative_slice(g)
        g = take_slice(g, k)
        items = [(lab, take_slice(a, k)) for lab, a in items]
    stats = {
        lab: failure_stats(a, g, support_threshold=support_threshold, dilate=dilate)
        for lab, a in items
    }
    with styled(dark):
        t = theme()
        if g.ndim == 1:
            ncols = 2 + int(spectrum)
            w, h = panel_size()
            fig, axes = new_figure(
                len(items), ncols, figsize=(ncols * w * 1.4, len(items) * h + 0.4)
            )
            for i, (lab, a) in enumerate(items):
                st = stats[lab]
                ax = axes[i, 0]
                plot_line(ax, g, style="gt", label="GT")
                plot_line(ax, a, label=lab)
                grid_on(ax)
                ax.set_title(f"{lab} · leakage {100 * st['leakage']:.1f} %")
                ax.set_ylabel(lab, fontsize="small")
                ax2 = axes[i, 1]
                ax2.axhline(0.0, color=t.axis, lw=0.6)
                plot_line(ax2, a - g, color=t.palette[1])
                grid_on(ax2)
                ax2.set_title(
                    f"signed error · {format_metric('relative_error', st['relative_error'])}"
                )
                if spectrum:
                    ax3 = axes[i, 2]
                    E = np.abs(np.fft.rfft(a - g)) ** 2
                    ax3.semilogy(np.arange(E.shape[0]), E + 1e-30, color=t.palette[0])
                    grid_on(ax3)
                    ax3.set_title("error power spectrum")
                    ax3.set_xlabel("frequency index")
            fig.suptitle(title or f"{name or 'field'}: failure modes")
            return fig
        ncols = 3 + int(spectrum)
        fig, axes = new_figure(
            len(items), ncols, figsize=figsize(ncols, len(items), cbar=2, title=True)
        )
        spec = cmap_for(name, g)
        lo, hi = color_limits([g] + [a for _, a in items], spec)
        espec = cmap_for(quantity="error")
        elo, ehi = color_limits([a - g for _, a in items], espec)
        sup = _support(g, support_threshold, dilate)
        ext = image_extent(g.shape)
        bg = float(np.median(g)) if np.min(g) > 0.05 * np.max(np.abs(g)) else 0.0
        leak_maps = [np.where(sup, np.nan, np.abs(a - bg)) for _, a in items]
        llo, lhi = color_limits(
            [np.nan_to_num(m) for m in leak_maps], cmap_for(quantity="magnitude")
        )
        rec_axes, err_axes, leak_axes = [], [], []
        im_r = im_e = im_l = None
        for i, (lab, a) in enumerate(items):
            st = stats[lab]
            ax = axes[i, 0]
            im_r = show_image(ax, a, spec=spec, vmin=lo, vmax=hi, extent=ext, title=lab)
            draw_contour(ax, sup.astype(float), ext, color=t.palette[2])
            rec_axes.append(ax)
            im_e = show_image(
                axes[i, 1],
                a - g,
                spec=espec,
                vmin=elo,
                vmax=ehi,
                extent=ext,
                title=f"signed error\n{format_metric('relative_error', st['relative_error'])}",
            )
            err_axes.append(axes[i, 1])
            im_l = show_image(
                axes[i, 2],
                leak_maps[i],
                spec=cmap_for(quantity="magnitude"),
                vmin=llo,
                vmax=lhi,
                extent=ext,
                title=f"leakage\n{100 * st['leakage']:.1f} % of mass outside support",
            )
            leak_axes.append(axes[i, 2])
            if spectrum:
                e = a - g
                P = np.fft.fftshift(np.abs(np.fft.fft2(e - e.mean())) ** 2)
                with np.errstate(divide="ignore"):
                    logP = np.log10(P + 1e-12 * max(float(P.max()), 1e-30))
                cross = st["cross"]
                txt = f"axis excess ×{cross:.2g} (≈1 isotropic)" if math.isfinite(cross) else ""
                show_image(
                    axes[i, 3],
                    logP,
                    spec=cmap_for(quantity="power"),
                    title=f"log error spectrum\n{txt}",
                )
        if im_r is not None:
            add_colorbar(fig, im_r, rec_axes)
        if im_e is not None:
            add_colorbar(fig, im_e, err_axes)
        if im_l is not None:
            add_colorbar(fig, im_l, leak_axes)
        fig.suptitle(title or f"{name or 'field'}: failure modes (support outlined)")
        return fig


# ---------------------------------------------------------------------------------------------
# baselines panel
# ---------------------------------------------------------------------------------------------
def run_methods(
    instance: Any,
    measurement: Any = None,
    gt: Mapping[str, Any] | None = None,
    methods: Any = None,
    budget_scale: float = 0.1,
    *,
    device: str = "cpu",
    seed: int = 0,
    scene_class: str | None = None,
    max_steps: int | None = None,
    min_steps: int = 5,
    smoke: bool = True,
) -> tuple[Any, dict[str, Any], Any, dict[str, dict[str, Any]]]:
    """Run the instance's own method and baselines on one measurement at a small budget.

    Args:
        instance: registered name, ``Instance`` subclass or object.
        measurement: :class:`~nefi.measurement.Measurement` (default: generated with ``seed``).
        gt: ground truth fields (required for metrics; generated with the measurement).
        methods: method names (``"neural"`` + baseline keys) or ``None`` for all.
        budget_scale: every stage's step count is multiplied by this factor.
        device: solver device.
        seed: data / optimization seed.
        scene_class: scene class for generated data.
        max_steps: cap on the total steps of every method.
        min_steps: at least this many steps per stage.
        smoke: apply the smoke preset when ``instance`` is a name.

    Returns:
        ``(instance, gt, measurement, runs)`` with ``runs[name] = {"result", "metrics",
        "time_s", "steps", "error"}``.
    """
    from ..bench.protocol import evaluate_metrics, resolve_methods, scale_budget
    from ..utils.seed import seed_everything
    from ._instances import resolve_instance

    inst, _ = resolve_instance(instance, smoke=smoke)
    if measurement is None:
        gt, measurement = inst.make_measurement(seed=seed, scene_class=scene_class)
    runs: dict[str, dict[str, Any]] = {}
    for m in resolve_methods(inst, methods):
        entry: dict[str, Any] = {"result": None, "metrics": {}, "time_s": None, "steps": 0}
        try:
            seed_everything(seed)
            prep = m.prepare(inst, measurement)
            cur = scale_budget(
                prep.resolved_curriculum(), budget_scale, max_steps, min_stage_steps=min_steps
            )
            cur.restarts = 1
            t0 = time.perf_counter()
            res = prep.run(cur, device=device, seed=seed)
            entry["time_s"] = time.perf_counter() - t0
            entry["result"] = res
            entry["steps"] = len(res.history.get("total", []))
            if gt is not None:
                entry["metrics"] = evaluate_metrics(inst, res, gt)
            entry["error"] = None
        except Exception as e:  # one failing baseline must not kill the panel
            log.warning("method %s failed: %s: %s", m.name, type(e).__name__, e)
            entry["error"] = f"{type(e).__name__}: {e}"
        runs[m.name] = entry
    return inst, gt, measurement, runs


def _headline(metrics: Mapping[str, float], n: int = 2) -> dict[str, float]:
    from .style import PRIMARY_METRICS

    out: dict[str, float] = {}
    for k in PRIMARY_METRICS:
        if k in metrics and len(out) < n:
            out[k] = metrics[k]
    for k, v in metrics.items():
        if len(out) >= n:
            break
        out.setdefault(k, v)
    return out


def baselines_panel(
    instance: Any,
    measurement: Any = None,
    gt: Mapping[str, Any] | None = None,
    methods: Any = None,
    budget_scale: float = 0.1,
    *,
    device: str = "cpu",
    seed: int = 0,
    scene_class: str | None = None,
    max_steps: int | None = None,
    error: bool = True,
    title: str | None = None,
    dark: bool | None = None,
) -> tuple[Figure, dict[str, dict[str, Any]]]:
    """Run the instance's method and its ``baselines()`` at a small budget and draw the grid.

    Columns: ground truth | measurement | every method (headline metrics and wall-clock in the
    titles; failed methods show their error). Arguments as in :func:`run_methods`.

    Returns:
        ``(figure, runs)`` — see :func:`run_methods` for ``runs``.
    """
    inst, gt, meas, runs = run_methods(
        instance,
        measurement,
        gt,
        methods,
        budget_scale,
        device=device,
        seed=seed,
        scene_class=scene_class,
        max_steps=max_steps,
    )
    field = None
    for r in runs.values():
        if r["result"] is not None:
            field = next((k for k in r["result"].fields if gt is None or k in gt), None)
            break
    row: dict[str, Any] = {}
    if gt is not None:
        row["GT"] = gt[field] if (field and field in gt) else next(iter(gt.values()))
    row["measurement"] = meas
    mets: dict[str, dict[str, float]] = {}
    for name, r in runs.items():
        if r["error"]:
            row[name] = f"failed: {r['error']}"
            continue
        row[name] = r["result"]
        m = _headline(r["metrics"])
        if r["time_s"] is not None:
            m["time_s"] = r["time_s"]
        mets[name] = m
    iname = getattr(inst, "name", type(inst).__name__)
    fig = method_grid(
        [row],
        field=field,
        metrics=[mets],
        error=error and gt is not None,
        instance=inst,
        title=title or f"{iname}: methods at budget ×{budget_scale:g}",
        dark=dark,
    )
    return fig, runs


__all__ = [
    "GT_KEYS",
    "MEAS_KEYS",
    "axis_excess",
    "baselines_panel",
    "failure_modes",
    "failure_stats",
    "method_grid",
    "run_methods",
]
