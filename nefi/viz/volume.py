"""3-D field views: depth mosaics, cross-sections through the anomaly, projections and voxel
comparisons — the building blocks of the 3-D layout of :func:`~nefi.viz.compare_fields`, the
gallery's 3-D rows and ``gallery_3d.png``.

Conventions (as in :mod:`nefi.viz.fields`): fields are indexed ``(x, y, z)`` and sliced along the
*volume axis* (default: the last one, depth ``z``; ``volume_axis`` display hint). In
cross-sections the volume axis points **down** (the first slice — the observed face of surface
measurements — is on top), the other axis runs horizontally. Everything is drawn into
caller-provided axes so the same panels serve figures, tiles and sub-figures.

* :func:`mosaic_slices` / :func:`mosaic_shape` — 4–6 evenly spaced slices (all of them for thin
  volumes) and their grid.
* :func:`anomaly_centroid` (from :mod:`nefi.viz.fields`) — where the cross-sections are cut.
* :func:`section` / :func:`project` — a depth cross-section / a min / max / mean projection.
* :class:`Outline` — what to outline on reconstructions: a GT mask or the GT anomaly level set.
* :func:`draw_slices`, :func:`draw_sections`, :func:`draw_projection` — panel drawers.
* :func:`compare_volume` — the 3-D ``compare_fields`` figure; :func:`voxel_compare` — GT and
  reconstructions as thresholded voxels with the same threshold, colour scale and camera.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .fields import (
    Comparison,
    _axis_names,
    _cell_centers,
    _domain_extents,
    _meas_panel,
    _outline_legend,
    _profile_panel,
    _recon_items,
    add_colorbar,
    anomaly_background,
    anomaly_centroid,
    anomaly_levels,
    axis_coords,
    draw_levels,
    image_extent,
    match_shape,
    metrics_label,
    resolve_field,
    show_image,
    squeeze_field,
    take_slice,
    voxel_threshold,
    voxel_view,
)
from .style import CmapSpec, cmap_for, color_limits, styled

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

log = logging.getLogger("nefi")

#: Slices of a compact depth mosaic (tiles, compare figures) and of the larger ``gallery_3d``.
MOSAIC_SLICES = 6
LARGE_MOSAIC_SLICES = 8


# ---------------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------------
def mosaic_slices(n: int, max_slices: int = MOSAIC_SLICES) -> list[int]:
    """Up to ``max_slices`` evenly spaced slice indices along an axis of length ``n``.

    Every slice of a thin volume (``n ≤ max_slices``, e.g. ``nz < 4``); otherwise the centres of
    ``max_slices`` equal depth bins (``n = 16, k = 4`` → 2, 6, 10, 14), which avoids spending
    panels on the outermost slices — usually empty for objects centred in the volume.
    """
    n, k = int(n), max(1, int(max_slices))
    if n <= k:
        return list(range(n))
    return sorted({min(n - 1, int(math.floor((i + 0.5) * n / k))) for i in range(k)})


def mosaic_shape(k: int, max_cols: int | None = None) -> tuple[int, int]:
    """``(rows, cols)`` of a mosaic of ``k`` slices: one row up to 3, 2×2 for 4, two rows up to
    8, then square-ish."""
    k = max(1, int(k))
    if k <= 3:
        rows, cols = 1, k
    elif k == 4:
        rows, cols = 2, 2
    elif k <= 8:
        rows, cols = 2, math.ceil(k / 2)
    else:
        cols = math.ceil(math.sqrt(k))
        rows = math.ceil(k / cols)
    if max_cols is not None and cols > max_cols:
        cols = max(1, int(max_cols))
        rows = math.ceil(k / cols)
    return rows, cols


def depth_label(k: int, n: int, axis: int, domain: Any = None, compact: bool = False) -> str:
    """``"z = 0.417"`` (physical, with a domain) or ``"z 3/6"`` (1-based slice number)."""
    ax = axis % 3
    name = _axis_names(domain, 3)[ax]
    ext = _domain_extents(domain)
    if ext is not None and ax < len(ext):
        v = _cell_centers(n, *ext[ax])[k]
        return f"{name}={v:.2g}" if compact else f"{name} = {v:.3g}"
    return f"{name} {k + 1}/{n}"


def section(
    vol: np.ndarray, center: Sequence[int], which: int, axis: int = -1
) -> tuple[np.ndarray, int, int]:
    """Cross-section containing the volume axis through ``center``.

    ``which=0``: the plane of the first lateral axis and the volume axis (``x–z`` for
    ``axis=-1``) at the second lateral axis = ``center``; ``which=1``: the ``y–z`` plane.

    Returns:
        ``(array[i_lateral, j_depth], lateral_axis, fixed_axis)``.
    """
    ax_v = axis % 3
    other = [d for d in range(3) if d != ax_v]
    lateral, fixed = other[which], other[1 - which]
    sl = np.take(vol, int(center[fixed]), axis=fixed)
    dims = [d for d in range(3) if d != fixed]
    if dims != [lateral, ax_v]:
        sl = sl.T
    return sl, lateral, fixed


def section_extent(
    shape: Sequence[int], domain: Any, lateral: int, depth: int
) -> tuple[float, float, float, float]:
    """``imshow`` extent of a cross-section drawn with ``origin="upper"`` (depth grows down)."""
    ext = _domain_extents(domain)
    if ext is not None and max(lateral, depth) < len(ext):
        (l0, l1), (z0, z1) = ext[lateral], ext[depth]
    else:
        l0, l1, z0, z1 = -0.5, shape[0] - 0.5, -0.5, shape[1] - 0.5
    return (l0, l1, z1, z0)


def projection_mode(ref: Any, background: float | None = None) -> str:
    """``"min"`` when the anomalies of ``ref`` lie below the background (low-diffusivity
    defects), ``"max"`` otherwise (sources, absorbers, densities)."""
    a = np.asarray(ref, dtype=float)
    fin = a[np.isfinite(a)]
    if fin.size == 0:
        return "max"
    med = float(np.median(fin)) if background is None else float(background)
    return "min" if (med - float(fin.min())) > (float(fin.max()) - med) else "max"


def project(
    vol: np.ndarray, axis: int = -1, mode: str = "auto", reference: Any = None
) -> tuple[np.ndarray, str]:
    """Projection along ``axis``: ``"max"``, ``"min"``, ``"mean"`` or ``"auto"`` (the anomaly
    side of ``reference`` — default ``vol`` — :func:`projection_mode`). Returns
    ``(array, mode)``."""
    if mode == "auto":
        mode = projection_mode(vol if reference is None else reference)
    fns = {"max": np.nanmax, "min": np.nanmin, "mean": np.nanmean}
    if mode not in fns:
        raise ValueError("projection mode must be 'max', 'min', 'mean' or 'auto'")
    return fns[mode](vol, axis=axis % 3), mode


@dataclass
class Outline:
    """What is outlined on reconstruction panels: a GT mask (level 0.5) or the GT anomaly level
    set (:func:`~nefi.viz.fields.anomaly_levels`)."""

    vol: np.ndarray
    levels: list[float]
    mask: bool = False

    @staticmethod
    def of(gt: Any = None, mask: Any = None) -> Outline | None:
        if mask is not None:
            m = np.asarray(mask, dtype=float)
            if m.ndim == 3 and bool(np.any(m > 0.5)):
                return Outline(m, [0.5], True)
        if gt is not None:
            g = np.asarray(gt)
            g = np.abs(g) if np.iscomplexobj(g) else g.astype(float)
            lv = anomaly_levels(g)
            return Outline(g, lv) if lv else None
        return None

    def slice(self, k: int, axis: int) -> np.ndarray:
        return take_slice(self.vol, k, axis)

    def section(self, center: Sequence[int], which: int, axis: int) -> np.ndarray:
        return section(self.vol, center, which, axis)[0]

    def projection(self, axis: int, mode: str, max_fraction: float = 0.2) -> np.ndarray | None:
        """The outlined quantity projected along ``axis`` — ``None`` for mean projections and
        for extended anomalies (covering more than ``max_fraction`` of the projection, e.g. a
        skull or a filament network), whose projected outline would only add clutter."""
        if self.mask:
            return np.max(self.vol, axis=axis % 3)
        if mode not in ("min", "max") or not self.levels:
            return None
        pr = project(self.vol, axis, mode)[0]
        region = pr <= min(self.levels) if mode == "min" else pr >= max(self.levels)
        if float(np.mean(region)) > max_fraction:
            return None
        return pr


# ---------------------------------------------------------------------------------------------
# panel drawers
# ---------------------------------------------------------------------------------------------
def _halo(width: float = 1.6, alpha: float = 0.6) -> list:
    import matplotlib.patheffects as pe

    return [pe.withStroke(linewidth=width, foreground="#000000", alpha=alpha)]


def label_inside(ax: Axes, text: str, loc: str = "upper left") -> None:
    """A small white label with a dark halo inside an image panel (depth of a mosaic slice)."""
    x, ha = (0.04, "left") if "left" in loc else (0.96, "right")
    y, va = (0.95, "top") if "upper" in loc else (0.05, "bottom")
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize="xx-small",
        color="#ffffff",
        path_effects=_halo(),
    )


def draw_slices(
    axes: Sequence[Axes],
    vol: np.ndarray,
    idx: Sequence[int],
    *,
    axis: int = -1,
    spec: CmapSpec,
    vmin: float,
    vmax: float,
    domain: Any = None,
    outline: Outline | None = None,
    outline_recon: bool = False,
    labels: str | None = "inside",
) -> list:
    """Depth slices ``idx`` of ``vol`` into ``axes`` (shared colour limits).

    ``outline`` draws the GT contour (solid white) on every slice, ``outline_recon`` adds the
    slice's own contour at the same level (dashed). ``labels``: ``"inside"`` (small depth label
    in the panel), ``"title"`` or ``None``.
    """
    ax_v = axis % 3
    n = vol.shape[ax_v]
    other = [d for d in range(3) if d != ax_v]
    ims = []
    for ax, k in zip(axes, idx):
        sl = take_slice(vol, int(k), ax_v)
        ext = image_extent(sl.shape, domain, (other[0], other[1]))
        ims.append(show_image(ax, sl, spec=spec, vmin=vmin, vmax=vmax, extent=ext))
        if outline is not None:
            draw_levels(ax, outline.slice(int(k), ax_v), outline.levels, ext, linewidth=0.8)
            if outline_recon and not outline.mask:
                draw_levels(ax, sl, outline.levels, ext, style="recon", linewidth=0.7)
        if labels == "inside":
            label_inside(ax, depth_label(int(k), n, ax_v, domain, compact=True))
        elif labels == "title":
            ax.set_title(depth_label(int(k), n, ax_v, domain), fontsize="small")
    return ims


def draw_sections(
    axes: Sequence[Axes],
    vol: np.ndarray,
    center: Sequence[int],
    *,
    axis: int = -1,
    spec: CmapSpec,
    vmin: float,
    vmax: float,
    domain: Any = None,
    outline: Outline | None = None,
    outline_recon: bool = False,
    titles: bool = True,
    marker: bool = True,
    axis_labels: bool = False,
    square: bool = True,
) -> list:
    """The two depth cross-sections through ``center`` (``x–z`` and ``y–z`` for ``axis=-1``),
    depth pointing down, a ``+`` at the centroid; outlines as in :func:`draw_slices`.
    ``square`` draws them in square boxes (the depth axis is stretched to the slice size, so thin
    slabs stay readable next to the mosaic)."""
    ax_v = axis % 3
    names = _axis_names(domain, 3)
    ims = []
    for which, ax in enumerate(list(axes)[:2]):
        sl, lateral, fixed = section(vol, center, which, ax_v)
        ext = section_extent(sl.shape, domain, lateral, ax_v)
        ims.append(
            show_image(
                ax,
                sl,
                spec=spec,
                vmin=vmin,
                vmax=vmax,
                extent=ext,
                origin="upper",
                aspect="auto",
                axes=axis_labels,
            )
        )
        if square:
            ax.set_box_aspect(1.0)
        if outline is not None:
            draw_levels(
                ax,
                outline.section(center, which, ax_v),
                outline.levels,
                ext,
                origin="upper",
                linewidth=0.8,
            )
            if outline_recon and not outline.mask:
                draw_levels(
                    ax, sl, outline.levels, ext, style="recon", origin="upper", linewidth=0.7
                )
        if marker:
            cx = axis_coords(vol.shape[lateral], domain, lateral)[int(center[lateral])]
            cz = axis_coords(vol.shape[ax_v], domain, ax_v)[int(center[ax_v])]
            ax.plot([cx], [cz], "+", ms=6, mew=1.0, color="#ffffff", path_effects=_halo(2.0, 0.5))
        if titles:
            fv = axis_coords(vol.shape[fixed], domain, fixed)[int(center[fixed])]
            where = f"{fv:.3g}" if _domain_extents(domain) is not None else f"{int(center[fixed])}"
            ax.set_title(
                f"{names[lateral]}–{names[ax_v]} at {names[fixed]} = {where}", fontsize="small"
            )
        if axis_labels:
            ax.set_xlabel(names[lateral])
            ax.set_ylabel(f"{names[ax_v]} ↓")
            ax.tick_params(labelsize="x-small")
    return ims


def draw_projection(
    ax: Axes,
    vol: np.ndarray,
    *,
    axis: int = -1,
    mode: str = "auto",
    reference: Any = None,
    spec: CmapSpec,
    vmin: float | None = None,
    vmax: float | None = None,
    domain: Any = None,
    outline: Outline | None = None,
    title: bool = True,
) -> Any:
    """A min / max / mean projection along the volume axis (``auto``: the anomaly side of
    ``reference``), outlined with the GT footprint when ``outline`` is given."""
    ax_v = axis % 3
    other = [d for d in range(3) if d != ax_v]
    pr, mode = project(vol, ax_v, mode, reference)
    ext = image_extent(pr.shape, domain, (other[0], other[1]))
    im = show_image(ax, pr, spec=spec, vmin=vmin, vmax=vmax, extent=ext)
    if outline is not None:
        op = outline.projection(ax_v, mode)
        if op is not None:
            draw_levels(ax, op, outline.levels, ext, linewidth=0.8)
    if title:
        ax.set_title(f"{mode} over {_axis_names(domain, 3)[ax_v]}", fontsize="small")
    return im


# ---------------------------------------------------------------------------------------------
# the 3-D comparison figure
# ---------------------------------------------------------------------------------------------
def _slice_indices(n: int, slices: Any, default: int = MOSAIC_SLICES) -> list[int]:
    if slices is None:
        return mosaic_slices(n, default)
    if isinstance(slices, int | np.integer):
        return mosaic_slices(n, int(slices))
    idx = [int(k) % n for k in slices]
    return idx or [n // 2]


def _row_label(ax: Axes, text: str) -> None:
    ax.set_ylabel(text, fontsize="small")
    ax.yaxis.set_visible(True)
    ax.set_yticks([])


def compare_volume(ctx: Comparison) -> Figure:
    """The 3-D layout of :func:`~nefi.viz.compare_fields`.

    One row per entry — ground truth, each reconstruction, (``"error"``) each signed error —
    with the same columns: a depth mosaic (up to 6 evenly spaced slices, depth in the column
    titles), the two depth cross-sections through the anomaly centroid and the anomaly-side
    projection; the measurement spans the first two rows on the left. ``"overlay"`` outlines
    the GT (mask or anomaly level set) on every reconstruction panel; ``"profile"`` adds line
    cuts along x, y and depth through the centroid.
    """
    from matplotlib.figure import Figure

    g, items = ctx.g, ctx.items
    ref = g if g is not None else items[0][1]
    ax_v = ctx.axis % 3
    n = ref.shape[ax_v]
    idx = _slice_indices(n, ctx.slices)
    k = len(idx)
    center = anomaly_centroid(ref, mask=ctx.mask)
    outline = Outline.of(g, ctx.mask) if "overlay" in ctx.extras else None
    mode = projection_mode(ref)
    rows: list[tuple[str, str, np.ndarray]] = []
    if ctx.show_gt and g is not None:
        rows.append(("gt", ctx.gt_title(), g))
    for lab, a in items:
        rows.append(("recon", lab, a))
    if "error" in ctx.extras and g is not None:
        for lab, a in items:
            rows.append(("error", lab, a - g))
    has_prof = "profile" in ctx.extras
    n_img = len(rows)
    cols = (["meas", "gap"] if ctx.show_meas else []) + [f"s{i}" for i in range(k)]
    cols += ["gap", "xz", "yz", "gap", "proj"]
    ratios = [1.75 if c == "meas" else 0.16 if c == "gap" else 1.0 for c in cols]
    unit, row_h = 1.02, 1.08
    n_cb = 1 + int(any(r[0] == "error" for r in rows))
    W = sum(ratios) * unit + 0.7 * n_cb + 0.35
    H = n_img * row_h + (1.75 if has_prof else 0.0) + 0.8 + (0.25 if outline else 0.0)
    fig = Figure(figsize=(round(W, 2), round(H, 2)), layout="constrained")
    gs = fig.add_gridspec(
        n_img + int(has_prof),
        len(cols),
        width_ratios=ratios,
        height_ratios=[1.0] * n_img + ([1.5] if has_prof else []),
    )
    s0 = cols.index("s0")
    lo, hi = ctx.lims()
    espec = cmap_for(quantity="error")
    errs = [r[2] for r in rows if r[0] == "error"]
    elo, ehi = color_limits(errs, espec) if errs else (0.0, 1.0)
    field_axes, err_axes, im_f, im_e = [], [], None, None
    first_err = True
    for r, (kind, lab, vol) in enumerate(rows):
        is_err = kind == "error"
        sp, vmin, vmax = (espec, elo, ehi) if is_err else (ctx.spec, lo, hi)
        ol = outline if kind == "recon" else None
        sax = [fig.add_subplot(gs[r, s0 + i]) for i in range(k)]
        ims = draw_slices(
            sax,
            vol,
            idx,
            axis=ax_v,
            spec=sp,
            vmin=vmin,
            vmax=vmax,
            domain=ctx.domain,
            outline=ol,
            outline_recon=True,
            labels="title" if r == 0 else None,
        )
        tag = f" ({ctx.tag})" if ctx.tag else ""  # e.g. "mean removed": row labels say so
        if kind == "gt":
            _row_label(sax[0], f"ground truth{tag}".replace(" (", "\n(", 1))
        elif kind == "recon":
            txt = metrics_label(ctx.mets.get(lab), 1)
            _row_label(sax[0], "\n".join(s for s in (f"{lab}{tag}", txt) if s))
        else:
            _row_label(sax[0], f"error\n{lab}")
        xz, yz = fig.add_subplot(gs[r, cols.index("xz")]), fig.add_subplot(gs[r, cols.index("yz")])
        draw_sections(
            [xz, yz],
            vol,
            center,
            axis=ax_v,
            spec=sp,
            vmin=vmin,
            vmax=vmax,
            domain=ctx.domain,
            outline=ol,
            outline_recon=True,
            titles=r == 0,
        )
        pax = fig.add_subplot(gs[r, cols.index("proj")])
        pim = draw_projection(
            pax,
            vol,
            axis=ax_v,
            mode="mean" if is_err else mode,
            spec=sp,
            vmin=vmin,
            vmax=vmax,
            domain=ctx.domain,
            outline=ol,
            title=r == 0 or (is_err and first_err),
        )
        panels = [*sax, xz, yz, pax]
        if is_err:
            err_axes += panels
            im_e = pim if im_e is None else im_e
            first_err = False
        else:
            field_axes += panels
            im_f = ims[0] if im_f is None else im_f
    if ctx.show_meas:
        max_ = fig.add_subplot(gs[0 : min(2, n_img), 0])
        _meas_panel(max_, ctx.measurement, ref.shape, ctx.instance, ctx.hints)
    if im_f is not None:
        add_colorbar(fig, im_f, field_axes, label=ctx.label if len(ctx.label) < 24 else None)
    if im_e is not None:
        add_colorbar(fig, im_e, err_axes, label="error")
    if has_prof:
        sub = gs[n_img, s0:].subgridspec(1, 3, wspace=0.08)
        other = [d for d in range(3) if d != ax_v]
        names = _axis_names(ctx.domain, 3)
        for j, d in enumerate([*other, ax_v]):
            pax = fig.add_subplot(sub[0, j])
            ttl = "depth profile" if d == ax_v else f"profile along {names[d]}"
            _profile_panel(pax, ctx, center, d, title=f"{ttl} through the anomaly")
            if j == 0:
                pax.legend(loc="best", fontsize="x-small")
    if outline is not None:
        _outline_legend(fig, with_recon=not outline.mask)
    return fig


# ---------------------------------------------------------------------------------------------
# voxels
# ---------------------------------------------------------------------------------------------
def voxel_sets(g: np.ndarray, a: np.ndarray, mode: str, threshold: float) -> tuple[int, float]:
    """``(voxels of a past the threshold, IoU with the GT voxel set)``."""
    sg = g < threshold if mode == "below" else g > threshold
    sa = a < threshold if mode == "below" else a > threshold
    union = int(np.count_nonzero(sg | sa))
    iou = float(np.count_nonzero(sg & sa)) / union if union else float("nan")
    return int(np.count_nonzero(sa)), iou


def draw_voxels(
    axes: Sequence[Axes],
    entries: Sequence[tuple[str, np.ndarray]],
    *,
    name: str | None = None,
    reference: np.ndarray | None = None,
    threshold: float | None = None,
    mode: str = "auto",
    spec: CmapSpec | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    domain: Any = None,
    elev: float = 24.0,
    azim: float = -58.0,
    max_points: int = 3000,
    zoom: float = 1.15,
) -> Any:
    """Thresholded voxels of every entry into ``projection="3d"`` axes with one threshold (from
    ``reference``, default the first entry), one colour scale and one camera; titles give the
    voxel count and the IoU with the reference voxel set. Returns the first scatter (for a
    shared colorbar) or ``None``."""
    ref = entries[0][1] if reference is None else reference
    if threshold is None or mode == "auto":
        mode, thr = voxel_threshold(ref, mode)
        threshold = thr if threshold is None else threshold
    arrays = [a for _, a in entries]
    sp = spec or cmap_for(name, ref)
    lo, hi = color_limits(arrays, sp)
    vmin = lo if vmin is None else vmin
    vmax = hi if vmax is None else vmax
    mappable = None
    sign = "<" if mode == "below" else ">"
    coords = [axis_coords(ref.shape[d], domain, d) for d in range(3)]
    spans = [float(c[-1] - c[0]) if len(c) > 1 else 1.0 for c in coords]
    spans = [s if s > 0 else 1.0 for s in spans]
    zx = max(1.0, 0.6 * max(spans[:2]) / spans[2])  # thin slabs: a readable depth axis
    for i, (ax, (lab, a)) in enumerate(zip(axes, entries)):
        count, iou = voxel_sets(ref, a, mode, threshold)
        if i == 0 and reference is None:
            head = f"{lab}\n{count} voxels {sign} {threshold:.3g}"
        else:
            iou_txt = f"IoU {iou:.2f}" if math.isfinite(iou) else "IoU —"
            head = f"{lab}\n{count} voxels · {iou_txt}"
        voxel_view(
            a,
            field=name,
            threshold=threshold,
            mode=mode,
            max_points=max_points,
            domain=domain,
            ax=ax,
            title=head,
            elev=elev,
            azim=azim,
            z_exaggeration=zx,
            spec=sp,
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
            zoom=zoom,
        )
        ax.title.set_fontsize("small")
        if mappable is None and ax.collections:
            mappable = ax.collections[0]
    return mappable


def voxel_compare(
    gt: Any,
    recons: Any = None,
    *,
    field: str | None = None,
    threshold: float | None = None,
    mode: str = "auto",
    domain: Any = None,
    elev: float = 24.0,
    azim: float = -58.0,
    max_points: int = 3000,
    transform: Any = None,
    hints: Mapping[str, Any] | None = None,
    instance: Any = None,
    title: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """Ground truth and reconstruction(s) as thresholded voxels, side by side.

    Every panel uses the **same threshold** (half-way from the GT background to its anomaly
    extreme, :func:`~nefi.viz.fields.voxel_threshold`, unless given), the same colour scale, the
    same camera and the same axis limits, so the panels are directly comparable; titles give the
    voxel count and the IoU of each reconstruction's voxel set with the GT's.

    Args:
        gt: 3-D ground truth (tensor / array / fields dict).
        recons: a reconstruction or ``{method: reconstruction}`` (``None``: GT only).
        field: field name (dict selection, colormap).
        threshold, mode: voxel threshold and side (``"above"`` / ``"below"`` / ``"auto"``).
        domain: physical coordinates.
        elev, azim: camera angles.
        max_points: random subsample cap per panel (fixed seed).
        transform: display transform (default: the ``field_transform`` hint).
        hints, instance: display hints (:mod:`nefi.viz.hints`).
        title: figure title.
        dark: dark theme.
    """
    import mpl_toolkits.mplot3d  # noqa: F401  (registers the 3d projection)
    from matplotlib.figure import Figure

    from .hints import apply_transform, field_hint, instance_hints, transform_name

    g, name = resolve_field(gt, field)
    g = squeeze_field(g)
    g = np.abs(g) if np.iscomplexobj(g) else g
    if g.ndim != 3:
        raise ValueError(f"voxel_compare expects a 3-D field, got shape {g.shape}")
    items = []
    for lab, a in _recon_items(recons, field or name):
        a = squeeze_field(a)
        a = np.abs(a) if np.iscomplexobj(a) else a
        items.append((lab, match_shape(a, g.shape) if a.shape != g.shape else a))
    hn = instance_hints(instance, hints)
    tf = transform if transform is not None else field_hint(hn, "field_transform", name)
    if transform_name(tf) is not None:
        g = apply_transform(g, tf, domain=domain, instance=instance)
        items = [
            (lab, apply_transform(a, tf, domain=domain, instance=instance)) for lab, a in items
        ]
    entries = [("ground truth", g), *items]
    with styled(dark):
        n = len(entries)
        fig = Figure(figsize=(3.1 * n + 0.8, 3.2), layout="constrained")
        axes = [fig.add_subplot(1, n, i + 1, projection="3d") for i in range(n)]
        m = draw_voxels(
            axes,
            entries,
            name=name,
            reference=None,
            threshold=threshold,
            mode=mode,
            domain=domain,
            elev=elev,
            azim=azim,
            max_points=max_points,
        )
        if m is not None:
            add_colorbar(fig, m, axes, label=name, shrink=0.35, location="bottom", pad=0.12)
        fig.suptitle(title or f"{name or 'field'}: thresholded voxels (same threshold and camera)")
        return fig


def anomaly_summary(vol: Any, mask: Any = None) -> dict[str, Any]:
    """Centroid, background and projection side of a volume (for titles and reports)."""
    a = squeeze_field(np.asarray(vol))
    a = np.abs(a) if np.iscomplexobj(a) else a
    return {
        "center": anomaly_centroid(a, mask=mask),
        "background": anomaly_background(a),
        "projection": projection_mode(a),
    }


__all__ = [
    "LARGE_MOSAIC_SLICES",
    "MOSAIC_SLICES",
    "Outline",
    "anomaly_summary",
    "compare_volume",
    "depth_label",
    "draw_projection",
    "draw_sections",
    "draw_slices",
    "draw_voxels",
    "label_inside",
    "mosaic_shape",
    "mosaic_slices",
    "project",
    "projection_mode",
    "section",
    "section_extent",
    "voxel_compare",
    "voxel_sets",
]
