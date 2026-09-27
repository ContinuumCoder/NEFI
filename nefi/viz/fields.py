"""Field viewers: 1-D lines, 2-D images, 3-D slices / mosaics / projections / voxels, and the
standard reconstruction figure :func:`compare_fields` (GT | measurement | reconstruction(s) |
signed error, metrics in the panel titles).

Conventions (matching :meth:`nefi.domain.Domain.coords`, ``indexing="ij"``): a 2-D field
``f[i, j]`` has ``i`` along the first axis (x, drawn horizontally) and ``j`` along the second (y,
drawn vertically, origin at the bottom). 3-D fields are sliced along the last axis (depth ``z``)
unless ``axis`` says otherwise, as in NeFTY Fig. 4. Complex fields (or ``complex_stack``-ed
real/imag pairs) are shown as magnitude and phase.

Every function returns a :class:`matplotlib.figure.Figure` built without ``pyplot`` and accepts
torch tensors, numpy arrays, :class:`~nefi.solve.Result` objects or ``{name: tensor}`` dicts.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .style import (
    CmapSpec,
    blank_axes,
    cmap_for,
    color_limits,
    figsize,
    format_metric,
    get_cmap,
    grid_on,
    message_axes,
    new_figure,
    panel_size,
    quantity_of,
    styled,
    theme,
)

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

log = logging.getLogger("nefi")

KINDS = ("1d", "2d", "3d", "complex")


# ---------------------------------------------------------------------------------------------
# conversion helpers
# ---------------------------------------------------------------------------------------------
def to_numpy(x: Any) -> np.ndarray:
    """Tensor / array / list → numpy (float64 for real data, complex128 for complex data)."""
    try:
        import torch

        if torch.is_tensor(x):
            t = x.detach().cpu()
            if t.is_complex():
                return t.to(torch.complex128).numpy()
            return t.to(torch.float64).numpy()
    except ImportError:  # pragma: no cover
        pass
    a = np.asarray(x)
    if np.iscomplexobj(a):
        return a.astype(np.complex128)
    return a.astype(np.float64)


def resolve_field(obj: Any, field: str | None = None) -> tuple[np.ndarray, str | None]:
    """Array of one field from a tensor, a :class:`~nefi.solve.Result`, a fields dict or a
    :class:`~nefi.measurement.Measurement`.

    Returns:
        ``(array, field_name)`` (the name is ``None`` for bare tensors).
    """
    if obj is None:
        raise ValueError("resolve_field got None")
    if hasattr(obj, "fields") and isinstance(obj.fields, Mapping):  # Result
        return resolve_field(obj.fields, field)
    if hasattr(obj, "data") and hasattr(obj, "noise_std"):  # Measurement
        return to_numpy(obj.data), field
    if isinstance(obj, Mapping):
        if not obj:
            raise ValueError("empty fields dict")
        key = field if (field is not None and field in obj) else next(iter(obj))
        return to_numpy(obj[key]), str(key)
    return to_numpy(obj), field


def as_complex(a: np.ndarray, stacked: str | None = None) -> np.ndarray:
    """Combine a stacked real/imag pair (``stacked="first"``: shape ``(2, ...)``; ``"last"``:
    ``(..., 2)``) into a complex array; other arrays are returned unchanged."""
    if stacked == "first" and a.shape[0] == 2 and not np.iscomplexobj(a):
        return a[0] + 1j * a[1]
    if stacked == "last" and a.shape[-1] == 2 and not np.iscomplexobj(a):
        return a[..., 0] + 1j * a[..., 1]
    return a


def squeeze_field(a: np.ndarray) -> np.ndarray:
    """Drop singleton dims (``(1, 64, 64)`` → ``(64, 64)``) but keep at least 1-D."""
    s = np.squeeze(a)
    return s.reshape(1) if s.ndim == 0 else s


def detect_kind(x: Any, kind: str = "auto") -> str:
    """``"1d"`` | ``"2d"`` | ``"3d"`` | ``"complex"`` from the dimensionality / dtype of ``x``."""
    if kind != "auto":
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}; use one of {KINDS} or 'auto'")
        return kind
    a = x if isinstance(x, np.ndarray) else to_numpy(x)
    if np.iscomplexobj(a):
        return "complex"
    a = squeeze_field(a)
    if a.ndim == 1:
        return "1d"
    if a.ndim == 2:
        return "2d"
    if a.ndim == 3:
        return "3d"
    raise ValueError(f"cannot display a {a.ndim}-D field of shape {a.shape} (1-3 dims supported)")


def match_shape(a: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Resample ``a`` to ``shape`` (area / linear, cell-centered) when the shapes differ."""
    shape = tuple(int(s) for s in shape)
    if tuple(a.shape) == shape or a.ndim != len(shape) or a.ndim > 3:
        return a
    import torch

    from ..utils.tensor import resample

    if np.iscomplexobj(a):
        re = resample(torch.as_tensor(a.real), shape).numpy()
        im = resample(torch.as_tensor(a.imag), shape).numpy()
        return re + 1j * im
    return resample(torch.as_tensor(a), shape).numpy()


# ---------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------
def default_metrics(kind: str) -> dict[str, Callable]:
    """PSNR + relative error (1-D), PSNR + SSIM (2-D / 3-D) from :mod:`nefi.metrics.basic`."""
    from ..metrics.basic import psnr, relative_error, ssim

    if kind == "1d":
        return {"psnr": psnr, "relative_error": relative_error}
    return {"psnr": psnr, "ssim": ssim}


def field_metrics(
    pred: Any, gt: Any, metrics: Mapping[str, Callable] | str | None = "auto"
) -> dict[str, float]:
    """Metrics of ``pred`` against ``gt`` (failures become ``nan``, never exceptions).

    ``metrics="auto"`` picks :func:`default_metrics` by dimensionality; complex fields are
    scored on their magnitude plus the complex relative error.
    """
    if metrics is None:
        return {}
    import torch

    p, g = to_numpy(pred), to_numpy(gt)
    if p.shape != g.shape:
        p = match_shape(p, g.shape)
    out: dict[str, float] = {}
    if np.iscomplexobj(p) or np.iscomplexobj(g):
        num = float(np.linalg.norm((p - g).ravel()))
        den = float(np.linalg.norm(g.ravel())) or 1e-30
        out["relative_error"] = num / den
        p, g = np.abs(p), np.abs(g)
    kind = detect_kind(squeeze_field(g))
    fns = default_metrics(kind) if metrics == "auto" else dict(metrics)  # type: ignore[arg-type]
    tp, tg = torch.as_tensor(squeeze_field(p)), torch.as_tensor(squeeze_field(g))
    for name, fn in fns.items():
        if name in out:
            continue
        try:
            out[name] = float(fn(tp, tg))
        except Exception as e:  # metrics must never break a figure
            log.debug("metric %s failed: %s", name, e)
            out[name] = float("nan")
    return out


def metrics_label(metrics: Mapping[str, Any] | None, max_items: int = 2) -> str:
    """``"PSNR 24.1 dB · SSIM 0.912"`` from a metrics dict (first ``max_items`` entries)."""
    if not metrics:
        return ""
    items = list(metrics.items())[:max_items]
    return " · ".join(format_metric(k, v) for k, v in items)


# ---------------------------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------------------------
def _domain_extents(domain: Any) -> list[tuple[float, float]] | None:
    if domain is None:
        return None
    ext = getattr(domain, "extent", domain)
    try:
        return [(float(lo), float(hi)) for lo, hi in ext]
    except (TypeError, ValueError):
        return None


def _axis_names(domain: Any, ndim: int) -> list[str]:
    names = getattr(domain, "axes", None)
    if names and len(names) == ndim:
        return [str(n) for n in names]
    return ["x", "y", "z"][:ndim] if ndim <= 3 else [f"x{i}" for i in range(ndim)]


def image_extent(
    shape: Sequence[int], domain: Any = None, dims: tuple[int, int] = (0, 1)
) -> tuple[float, float, float, float]:
    """``imshow`` extent ``(x0, x1, y0, y1)`` of a 2-D slice: physical when ``domain`` (a
    :class:`~nefi.domain.Domain` or an extent list) is given, pixel indices otherwise."""
    ext = _domain_extents(domain)
    if ext is not None and max(dims) < len(ext):
        (x0, x1), (y0, y1) = ext[dims[0]], ext[dims[1]]
        return (x0, x1, y0, y1)
    return (-0.5, shape[0] - 0.5, -0.5, shape[1] - 0.5)


def _cell_centers(n: int, lo: float, hi: float) -> np.ndarray:
    return lo + (np.arange(n) + 0.5) * (hi - lo) / n


def axis_coords(n: int, domain: Any = None, dim: int = 0) -> np.ndarray:
    """Cell-center coordinates of axis ``dim`` (physical if ``domain`` is given)."""
    ext = _domain_extents(domain)
    if ext is not None and dim < len(ext):
        return _cell_centers(n, *ext[dim])
    return np.arange(n, dtype=float)


def representative_slice(vol: np.ndarray, axis: int = -1) -> int:
    """Index of the slice with the largest in-slice variance (ties → the middle slice)."""
    a = np.moveaxis(np.abs(vol) if np.iscomplexobj(vol) else vol, axis, 0)
    var = np.array([np.nanvar(s) for s in a])
    if not np.isfinite(var).any() or np.allclose(var, var[0]):
        return a.shape[0] // 2
    return int(np.nanargmax(var))


def select_slices(n: int, n_slices: int) -> list[int]:
    """``n_slices`` evenly spaced indices in ``[0, n)`` (all of them if ``n <= n_slices``)."""
    if n <= n_slices:
        return list(range(n))
    return sorted({int(round(v)) for v in np.linspace(0, n - 1, n_slices)})


def take_slice(vol: np.ndarray, index: int, axis: int = -1) -> np.ndarray:
    """2-D slice ``vol[..., index]`` along ``axis`` (remaining dims keep their order)."""
    return np.take(vol, index, axis=axis)


def _slice_label(index: int, n: int, axis: int, domain: Any, ndim: int = 3) -> str:
    ax = axis % ndim
    name = _axis_names(domain, ndim)[ax]
    ext = _domain_extents(domain)
    if ext is not None and ax < len(ext):
        v = _cell_centers(n, *ext[ax])[index]
        return f"{name} = {v:.3g}"
    return f"{name} = {index}/{n - 1}"


# ---------------------------------------------------------------------------------------------
# drawing primitives
# ---------------------------------------------------------------------------------------------
def add_colorbar(
    fig: Any,
    mappable: Any,
    axes: Any,
    label: str | None = None,
    nticks: int = 4,
    shrink: float = 0.85,
    location: str | None = None,
    pad: float = 0.015,
) -> Any:
    """Thin colorbar attached to one axes or a list of axes (constrained-layout friendly);
    ``location="bottom"`` puts it below (e.g. next to 3-D axes, whose tick labels constrained
    layout does not see)."""
    from matplotlib.ticker import MaxNLocator

    t = theme()
    kw = {"location": location} if location else {}
    cb = fig.colorbar(mappable, ax=axes, shrink=shrink, aspect=22, pad=pad, fraction=0.05, **kw)
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor(t.axis)
    cb.ax.tick_params(length=2, width=0.5, pad=1.5, labelsize="small", colors=t.axis)
    for lab in cb.ax.get_yticklabels() + cb.ax.get_xticklabels():
        lab.set_color(t.ink2)
    with np.errstate(all="ignore"):
        try:
            cb.locator = MaxNLocator(nticks)
            cb.update_ticks()
        except Exception:  # pragma: no cover - log-norm colorbars keep their locator
            pass
    if label:
        cb.set_label(label, color=t.ink2)
    return cb


def show_image(
    ax: Axes,
    img: Any,
    *,
    spec: CmapSpec | str | None = None,
    name: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    extent: tuple[float, float, float, float] | None = None,
    domain: Any = None,
    title: str | None = None,
    colorbar: bool = False,
    cbar_label: str | None = None,
    transpose: bool = True,
    origin: str = "lower",
    aspect: str | float = "equal",
    axes: bool = False,
    mask: Any = None,
):
    """Draw a 2-D array as an image panel.

    Args:
        ax: target axes.
        img: 2-D array indexed ``[i, j]`` = ``[x, y]`` (``transpose=True``, the field
            convention) or ``[row, col]`` (``transpose=False``, e.g. sinograms, traces).
        spec: :class:`CmapSpec` or colormap name (default: :func:`cmap_for` of ``name`` / data).
        name: field name used to choose the colormap.
        vmin, vmax: color limits (default: data range; symmetric for diverging specs).
        extent: ``imshow`` extent; default from ``domain`` or pixel indices.
        domain: :class:`~nefi.domain.Domain` for physical axes.
        title: panel title.
        colorbar: attach a colorbar to this panel.
        cbar_label: colorbar label.
        transpose: display ``img.T`` (field convention).
        origin: ``"lower"`` (fields) or ``"upper"`` (time-down seismic gathers).
        aspect: ``"equal"`` for fields, ``"auto"`` for data matrices.
        axes: keep ticks / spines (physical axes) instead of a clean panel.
        mask: 1 = observed / inside; 0 pixels are drawn in the theme's "bad" color.

    Returns:
        The ``AxesImage``.
    """
    a = squeeze_field(to_numpy(img))
    if np.iscomplexobj(a):
        a = np.abs(a)
    if a.ndim != 2:
        raise ValueError(f"show_image expects a 2-D array, got shape {a.shape}")
    if mask is not None:
        m = squeeze_field(to_numpy(mask)).astype(bool)
        if m.shape == a.shape:
            a = np.where(m, a, np.nan)
    if spec is None or isinstance(spec, str):
        spec = CmapSpec(spec, "sequential") if isinstance(spec, str) else cmap_for(name, a)
    if vmin is None or vmax is None:
        lo, hi = color_limits(a, spec)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    disp = a.T if transpose else a
    if extent is None:
        shape = a.shape if transpose else (a.shape[1], a.shape[0])
        extent = image_extent(shape, domain) if transpose else image_extent(shape, None)
    im = ax.imshow(
        disp,
        cmap=get_cmap(spec),
        vmin=vmin,
        vmax=vmax,
        origin=origin,
        extent=extent,
        aspect=aspect,
        interpolation="nearest",
    )
    if not axes:
        blank_axes(ax)
    else:
        ax.grid(False)
    if title:
        ax.set_title(title)
    if colorbar:
        add_colorbar(ax.figure, im, ax, cbar_label)
    return im


def peak_indices(img: Any, threshold: float = 0.05) -> np.ndarray:
    """``(k, 2)`` integer ``(i, j)`` indices of 3×3 local maxima above ``threshold · max``."""
    a = squeeze_field(to_numpy(img))
    if np.iscomplexobj(a):
        a = np.abs(a)
    try:
        from ..metrics.localization import peak_positions

        return peak_positions(a, threshold).numpy().astype(int)
    except Exception:  # pragma: no cover - fallback without the metrics module
        from scipy.ndimage import maximum_filter

        mx = float(np.nanmax(a)) if a.size else 0.0
        if mx <= 0:
            return np.zeros((0, 2), dtype=int)
        peaks = (a >= maximum_filter(a, size=3, mode="nearest")) & (a > threshold * mx)
        return np.argwhere(peaks)


def draw_peaks(
    ax: Axes,
    idx: np.ndarray,
    shape: Sequence[int],
    extent: tuple[float, float, float, float],
    style: str = "gt",
) -> None:
    """Overlay peak markers: ``"gt"`` hollow white circles, ``"recon"`` aqua crosses."""
    if idx is None or len(idx) == 0:
        return
    import matplotlib.patheffects as pe

    x0, x1, y0, y1 = extent
    xs = x0 + (idx[:, 0] + 0.5) * (x1 - x0) / shape[0]
    ys = y0 + (idx[:, 1] + 0.5) * (y1 - y0) / shape[1]
    halo = [pe.withStroke(linewidth=1.8, foreground="#000000", alpha=0.6)]
    if style == "gt":
        ax.plot(
            xs,
            ys,
            "o",
            ms=5.5,
            mfc="none",
            mec="#ffffff",
            mew=0.9,
            path_effects=halo,
            label="GT peaks",
        )
    else:
        ax.plot(
            xs,
            ys,
            "x",
            ms=4.5,
            color=theme().palette[2],
            mew=1.1,
            path_effects=halo,
            label="recovered peaks",
        )


def draw_contour(
    ax: Axes,
    mask: Any,
    extent: tuple[float, float, float, float],
    color: str | None = None,
    level: float = 0.5,
) -> None:
    """Outline of a 2-D mask (e.g. a defect / support) on an image panel."""
    m = squeeze_field(to_numpy(mask)).astype(float)
    if m.ndim != 2 or not np.isfinite(m).all() or m.min() == m.max():
        return
    x0, x1, y0, y1 = extent
    xs = _cell_centers(m.shape[0], x0, x1)
    ys = _cell_centers(m.shape[1], y0, y1)
    ax.contour(
        xs,
        ys,
        m.T,
        levels=[level],
        colors=[color or "#ffffff"],
        linewidths=0.8,
    )


def _is_sparse(a: np.ndarray, threshold: float = 0.05, max_fraction: float = 0.06) -> bool:
    """True for sparse non-negative maps (few pixels above ``threshold · max``)."""
    if np.iscomplexobj(a) or a.size == 0:
        return False
    mx = float(np.nanmax(a))
    if mx <= 0 or float(np.nanmin(a)) < -0.05 * mx:
        return False
    return float(np.mean(a > threshold * mx)) <= max_fraction


def plot_line(
    ax: Axes,
    y: Any,
    *,
    x: Any = None,
    domain: Any = None,
    label: str | None = None,
    color: str | None = None,
    style: str = "recon",
    **kw: Any,
) -> None:
    """One 1-D field on an axes. ``style``: ``"gt"`` (ink, dashed), ``"recon"`` (series color,
    solid), ``"data"`` (muted dots)."""
    t = theme()
    a = squeeze_field(to_numpy(y))
    if np.iscomplexobj(a):
        a = np.abs(a)
    xs = to_numpy(x) if x is not None else axis_coords(a.shape[0], domain, 0)
    if style == "gt":
        ax.plot(xs, a, color=color or t.ink, lw=1.1, ls=(0, (4, 2)), label=label, **kw)
    elif style == "data":
        ax.plot(xs, a, ".", color=color or t.muted, ms=2.5, alpha=0.9, label=label, zorder=1, **kw)
    else:
        ax.plot(xs, a, color=color or t.palette[0], lw=1.3, label=label, zorder=3, **kw)


# ---------------------------------------------------------------------------------------------
# single-field viewers
# ---------------------------------------------------------------------------------------------
def plot_field(
    x: Any,
    *,
    field: str | None = None,
    kind: str = "auto",
    ax: Axes | None = None,
    domain: Any = None,
    title: str | None = None,
    colorbar: bool = True,
    peaks: bool = False,
    mask: Any = None,
    axis: int = -1,
    n_slices: int = 8,
    complex_stack: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """Draw one field: line (1-D), image (2-D), depth mosaic (3-D; the representative slice when
    ``ax`` is given) or magnitude + phase (complex).

    Args:
        x: tensor / array / Result / fields dict.
        field: field name (selects from dicts; chooses the colormap).
        kind: ``"auto"`` or one of ``"1d"``, ``"2d"``, ``"3d"``, ``"complex"``.
        ax: draw into an existing axes (single panel) instead of a new figure.
        domain: :class:`~nefi.domain.Domain` for physical axes / slice labels.
        title: title (default: the field name).
        colorbar: add a colorbar.
        peaks: mark 3×3 local maxima (sparse density maps).
        mask: contour overlay (same shape as the displayed slice / image).
        axis: slicing axis of 3-D fields.
        n_slices: slices of the 3-D mosaic.
        complex_stack: ``"first"`` / ``"last"`` if real and imaginary parts are stacked.
        dark: dark theme (``None`` = active style).
    """
    a, name = resolve_field(x, field)
    a = squeeze_field(as_complex(a, complex_stack))
    k = detect_kind(a, kind)
    with styled(dark):
        if k == "3d" and ax is None:
            return depth_mosaic(
                a, field=name, axis=axis, n_slices=n_slices, domain=domain, title=title
            )
        if k == "complex" and ax is None:
            return plot_complex(a, field=name, domain=domain, title=title)
        if ax is None:
            fig, axes = new_figure(1, 1, figsize=figsize(1, 1, cbar=colorbar))
            ax = axes[0, 0]
        fig = ax.figure
        if k == "1d":
            plot_line(ax, a, domain=domain, label=name)
            grid_on(ax)
            ax.set_xlabel(_axis_names(domain, 1)[0])
            ax.set_title(title or name or "")
            return fig
        if k == "3d":
            i = representative_slice(a, axis)
            sl = take_slice(a, i, axis)
            title = title or f"{name or 'field'} ({_slice_label(i, a.shape[axis], axis, domain)})"
            a = sl
        if np.iscomplexobj(a):
            a = np.abs(a)
        ext = image_extent(a.shape, domain)
        show_image(ax, a, name=name, domain=domain, title=title or name, colorbar=colorbar)
        if peaks:
            draw_peaks(ax, peak_indices(a), a.shape, ext, "gt")
        if mask is not None:
            draw_contour(ax, mask, ext)
        return fig


def image_panels(
    images: Mapping[str, Any],
    *,
    field: str | None = None,
    spec: CmapSpec | str | None = None,
    shared: bool = True,
    ncols: int | None = None,
    peaks: bool = False,
    mask: Any = None,
    domain: Any = None,
    title: str | None = None,
    colorbar: bool = True,
    dark: bool | None = None,
) -> Figure:
    """Grid of 2-D panels ``{label: image}`` with one shared color scale (default).

    3-D entries are shown at the representative slice of the first entry; complex ones by
    magnitude.
    """
    items = []
    name = field
    for lab, v in images.items():
        a, nm = resolve_field(v, field)
        name = name or nm
        items.append((str(lab), squeeze_field(a)))
    if not items:
        raise ValueError("image_panels needs at least one image")
    ref = items[0][1]
    if ref.ndim == 3:
        k = representative_slice(ref)
        items = [(lab, take_slice(a, k) if a.ndim == 3 else a) for lab, a in items]
    items = [(lab, np.abs(a) if np.iscomplexobj(a) else a) for lab, a in items]
    n = len(items)
    ncols = ncols or min(n, 4)
    nrows = math.ceil(n / ncols)
    with styled(dark):
        sp = spec if isinstance(spec, CmapSpec) else cmap_for(name, items[0][1])
        if isinstance(spec, str):
            sp = CmapSpec(spec)
        lims = color_limits([a for _, a in items], sp) if shared else (None, None)
        fig, axes = new_figure(
            nrows, ncols, figsize=figsize(ncols, nrows, cbar=colorbar, title=bool(title))
        )
        ims, used = [], []
        for i, (lab, a) in enumerate(items):
            ax = axes[i // ncols, i % ncols]
            ext = image_extent(a.shape, domain)
            im = show_image(
                ax, a, spec=sp, vmin=lims[0], vmax=lims[1], extent=ext, title=lab, domain=domain
            )
            if peaks:
                draw_peaks(ax, peak_indices(a), a.shape, ext, "recon")
            if mask is not None:
                draw_contour(ax, mask, ext)
            ims.append(im)
            used.append(ax)
            if colorbar and not shared:
                add_colorbar(fig, im, ax)
        for j in range(n, nrows * ncols):
            axes[j // ncols, j % ncols].remove()
        if colorbar and shared:
            add_colorbar(fig, ims[0], used)
        if title:
            fig.suptitle(title)
        return fig


# ---------------------------------------------------------------------------------------------
# 3-D viewers
# ---------------------------------------------------------------------------------------------
def depth_mosaic(
    vol: Any,
    *,
    field: str | None = None,
    axis: int = -1,
    slices: Sequence[int] | None = None,
    n_slices: int = 8,
    ncols: int | None = None,
    domain: Any = None,
    spec: CmapSpec | str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    mask: Any = None,
    title: str | None = None,
    colorbar: bool = True,
    dark: bool | None = None,
) -> Figure:
    """Depth-slice mosaic of a 3-D field with one shared color scale (NeFTY Fig. 4 style).

    Thin volumes are handled gracefully: with fewer than ``n_slices`` slices every slice is
    shown (one row for ≤ 4), and a 2-D array (a single slice, e.g. ``nz = 1`` squeezed away) is
    drawn as a one-panel mosaic.

    Args:
        vol: 3-D tensor / array / Result / fields dict.
        field: field name (colormap choice, dict selection).
        axis: slicing axis (default: last = depth).
        slices: explicit slice indices (default: ``n_slices`` evenly spaced).
        n_slices: number of slices when ``slices`` is None.
        ncols: mosaic columns (default ≤ 4).
        domain: :class:`~nefi.domain.Domain` for physical slice positions.
        spec, vmin, vmax: colormap and limits (default: by quantity, shared data range).
        mask: optional 3-D mask outlined on every slice (e.g. GT defects).
        title: figure title.
        colorbar: shared colorbar.
        dark: dark theme.
    """
    a, name = resolve_field(vol, field)
    a = squeeze_field(a)
    if np.iscomplexobj(a):
        a = np.abs(a)
    if a.ndim == 2:  # a single slice (nz = 1): a one-panel mosaic
        a = np.expand_dims(a, axis=axis if axis in (-1, 2) else axis % 3)
    if a.ndim != 3:
        raise ValueError(f"depth_mosaic expects a 3-D field, got shape {a.shape}")
    n = a.shape[axis]
    idx = [int(k) % n for k in slices] if slices is not None else select_slices(n, n_slices)
    idx = idx or [n // 2]
    ncols = max(1, ncols or min(4, len(idx)))
    nrows = math.ceil(len(idx) / ncols)
    m3 = squeeze_field(to_numpy(mask)) if mask is not None else None
    if m3 is not None and m3.ndim == 2 and a.shape[axis] == 1:
        m3 = np.expand_dims(m3, axis=axis if axis in (-1, 2) else axis % 3)
    other = [d for d in range(3) if d != axis % 3]
    with styled(dark):
        sp = spec if isinstance(spec, CmapSpec) else cmap_for(field or name, a)
        if isinstance(spec, str):
            sp = CmapSpec(spec)
        lo, hi = color_limits(a, sp)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
        fig, axes = new_figure(
            nrows, ncols, figsize=figsize(ncols, nrows, cbar=colorbar, title=bool(title))
        )
        used, im = [], None
        for p, k in enumerate(idx):
            ax = axes[p // ncols, p % ncols]
            sl = take_slice(a, k, axis)
            ext = image_extent(sl.shape, domain, (other[0], other[1]))
            im = show_image(
                ax,
                sl,
                spec=sp,
                vmin=vmin,
                vmax=vmax,
                extent=ext,
                title=_slice_label(k, n, axis, domain),
            )
            if m3 is not None and m3.shape == a.shape:
                draw_contour(ax, take_slice(m3, k, axis), ext)
            used.append(ax)
        for j in range(len(idx), nrows * ncols):
            axes[j // ncols, j % ncols].remove()
        if colorbar and im is not None:
            add_colorbar(fig, im, used, label=name)
        fig.suptitle(title or f"{name or 'field'}: depth slices")
        return fig


def _anomaly_index(a: np.ndarray) -> tuple[int, ...]:
    dev = np.abs(a - np.nanmedian(a))
    return tuple(int(i) for i in np.unravel_index(int(np.nanargmax(dev)), a.shape))


def orthoslices(
    vol: Any,
    *,
    field: str | None = None,
    index: Sequence[int] | None = None,
    domain: Any = None,
    spec: CmapSpec | str | None = None,
    title: str | None = None,
    crosshair: bool = True,
    colorbar: bool = True,
    dark: bool | None = None,
) -> Figure:
    """Three orthogonal slices through ``index`` (default: the most anomalous voxel, i.e. the
    largest difference from the median — typically inside a defect)."""
    a, name = resolve_field(vol, field)
    a = squeeze_field(a)
    if np.iscomplexobj(a):
        a = np.abs(a)
    if a.ndim != 3:
        raise ValueError(f"orthoslices expects a 3-D field, got shape {a.shape}")
    idx = tuple(int(i) for i in index) if index is not None else _anomaly_index(a)
    names = _axis_names(domain, 3)
    planes = [(2, (0, 1)), (1, (0, 2)), (0, (1, 2))]  # (fixed axis, displayed dims)
    t = None
    with styled(dark):
        t = theme()
        sp = spec if isinstance(spec, CmapSpec) else cmap_for(field or name, a)
        if isinstance(spec, str):
            sp = CmapSpec(spec)
        lo, hi = color_limits(a, sp)
        fig, axes = new_figure(1, 3, figsize=figsize(3, 1, cbar=colorbar, title=True))
        im = None
        for p, (fixed, dims) in enumerate(planes):
            ax = axes[0, p]
            sl = take_slice(a, idx[fixed], fixed)
            ext = image_extent(sl.shape, domain, dims)
            im = show_image(
                ax,
                sl,
                spec=sp,
                vmin=lo,
                vmax=hi,
                extent=ext,
                aspect="auto",
                axes=True,
                title=_slice_label(idx[fixed], a.shape[fixed], fixed, domain),
            )
            ax.set_xlabel(names[dims[0]])
            ax.set_ylabel(names[dims[1]])
            ax.tick_params(labelsize="small")
            if crosshair:
                cx = axis_coords(a.shape[dims[0]], domain, dims[0])[idx[dims[0]]]
                cy = axis_coords(a.shape[dims[1]], domain, dims[1])[idx[dims[1]]]
                if _domain_extents(domain) is None:
                    cx, cy = float(idx[dims[0]]), float(idx[dims[1]])
                ax.axvline(cx, color=t.ink2, lw=0.5, alpha=0.7)
                ax.axhline(cy, color=t.ink2, lw=0.5, alpha=0.7)
        if colorbar and im is not None:
            add_colorbar(fig, im, list(axes[0]), label=name)
        fig.suptitle(title or f"{name or 'field'}: orthogonal slices through {idx}")
        return fig


def projections(
    vol: Any,
    *,
    field: str | None = None,
    mode: str = "auto",
    domain: Any = None,
    spec: CmapSpec | str | None = None,
    title: str | None = None,
    colorbar: bool = True,
    dark: bool | None = None,
) -> Figure:
    """Max / min / mean projections of a 3-D field along each axis.

    ``mode="auto"`` projects the *anomalies*: ``min`` when the field's outliers lie below the
    background (e.g. low-diffusivity defects, NeFTY), ``max`` otherwise (sources, densities).
    """
    a, name = resolve_field(vol, field)
    a = squeeze_field(a)
    if np.iscomplexobj(a):
        a = np.abs(a)
    if a.ndim != 3:
        raise ValueError(f"projections expects a 3-D field, got shape {a.shape}")
    if mode == "auto":
        med = float(np.nanmedian(a))
        mode = "min" if (med - np.nanmin(a)) > (np.nanmax(a) - med) else "max"
    fn = {"max": np.nanmax, "min": np.nanmin, "mean": np.nanmean}.get(mode)
    if fn is None:
        raise ValueError("mode must be 'max', 'min', 'mean' or 'auto'")
    names = _axis_names(domain, 3)
    views = [(2, (0, 1)), (1, (0, 2)), (0, (1, 2))]
    with styled(dark):
        sp = spec if isinstance(spec, CmapSpec) else cmap_for(field or name, a)
        if isinstance(spec, str):
            sp = CmapSpec(spec)
        projs = [fn(a, axis=ax) for ax, _ in views]
        lo, hi = color_limits(projs, sp)
        fig, axes = new_figure(1, 3, figsize=figsize(3, 1, cbar=colorbar, title=True))
        im = None
        for p, ((ax_i, dims), pr) in enumerate(zip(views, projs)):
            ax = axes[0, p]
            ext = image_extent(pr.shape, domain, dims)
            im = show_image(
                ax,
                pr,
                spec=sp,
                vmin=lo,
                vmax=hi,
                extent=ext,
                aspect="auto",
                axes=True,
                title=f"{mode} over {names[ax_i]}",
            )
            ax.set_xlabel(names[dims[0]])
            ax.set_ylabel(names[dims[1]])
        if colorbar and im is not None:
            add_colorbar(fig, im, list(axes[0]), label=name)
        fig.suptitle(title or f"{name or 'field'}: {mode}-intensity projections")
        return fig


def voxel_threshold(
    a: np.ndarray, mode: str = "auto", background: float | None = None, fraction: float = 0.5
) -> tuple[str, float]:
    """``(mode, threshold)`` of a voxel view: the anomaly side (``"below"`` / ``"above"`` the
    background, the side of the larger deviation for ``mode="auto"``) and the level
    ``fraction`` of the way from the background (default: the median) to that extreme."""
    fin = a[np.isfinite(a)]
    if fin.size == 0:
        return ("above" if mode == "auto" else mode), 0.0
    med = float(np.median(fin)) if background is None else float(background)
    lo_, hi_ = float(fin.min()), float(fin.max())
    if mode == "auto":
        mode = "below" if (med - lo_) > (hi_ - med) else "above"
    if mode == "below":
        return mode, med - fraction * (med - lo_)
    return mode, med + fraction * (hi_ - med)


def voxel_view(
    vol: Any,
    *,
    field: str | None = None,
    threshold: float | None = None,
    mode: str = "auto",
    max_points: int = 4000,
    domain: Any = None,
    ax: Axes | None = None,
    title: str | None = None,
    elev: float = 24.0,
    azim: float = -58.0,
    z_exaggeration: float | None = None,
    spec: CmapSpec | str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    zoom: float = 1.0,
    dark: bool | None = None,
) -> Figure:
    """Isosurface-free 3-D view: scatter of thresholded voxels, colored by value.

    Args:
        vol: 3-D field.
        field: field name (colormap).
        threshold: voxel threshold (default: half-way between the median and the extreme on the
            anomaly side, :func:`voxel_threshold`).
        mode: ``"above"``, ``"below"`` or ``"auto"`` (the side of the larger deviation).
        max_points: random subsample cap (fixed seed).
        domain: physical coordinates.
        ax: an existing ``projection="3d"`` axes.
        title: panel title.
        elev, azim: camera angles.
        z_exaggeration: stretch of the depth axis (default: so thin slabs stay visible).
        spec, vmin, vmax: colormap and color limits (default: by quantity, data range) — pass
            the same values to compare volumes side by side (:func:`nefi.viz.voxel_compare`).
        colorbar: attach a colorbar.
        zoom: zoom of the 3-D box inside its axes (> 1 fills more of the panel).
        dark: dark theme.
    """
    import mpl_toolkits.mplot3d  # noqa: F401  (registers the 3d projection)

    a, name = resolve_field(vol, field)
    a = squeeze_field(a)
    if np.iscomplexobj(a):
        a = np.abs(a)
    if a.ndim != 3:
        raise ValueError(f"voxel_view expects a 3-D field, got shape {a.shape}")
    if threshold is None or mode == "auto":
        mode, thr = voxel_threshold(a, mode)
        threshold = thr if threshold is None else threshold
    sel = a < threshold if mode == "below" else a > threshold
    idx = np.argwhere(sel)
    if len(idx) > max_points:
        rng = np.random.default_rng(0)
        idx = idx[rng.choice(len(idx), max_points, replace=False)]
    coords = [axis_coords(a.shape[d], domain, d) for d in range(3)]
    with styled(dark):
        t = theme()
        if ax is None:
            fig = new_figure(0, 0, figsize=figsize(1, 1, panel=(3.4, 3.0)))[0]
            ax = fig.add_subplot(111, projection="3d")
        fig = ax.figure
        sp = spec if isinstance(spec, CmapSpec) else cmap_for(field or name, a)
        if isinstance(spec, str):
            sp = CmapSpec(spec)
        lo_c, hi_c = color_limits(a, sp)
        vmin = lo_c if vmin is None else vmin
        vmax = hi_c if vmax is None else vmax
        names = _axis_names(domain, 3)
        if len(idx):
            xs, ys, zs = (coords[d][idx[:, d]] for d in range(3))
            vals = a[idx[:, 0], idx[:, 1], idx[:, 2]]
            size = float(np.clip(1500.0 / max(len(idx), 1) ** 0.5, 2.0, 18.0))
            sc = ax.scatter(
                xs,
                ys,
                zs,
                c=vals,
                cmap=get_cmap(sp),
                vmin=vmin,
                vmax=vmax,
                s=size,
                depthshade=True,
                edgecolors="none",
            )
            if colorbar:
                add_colorbar(fig, sc, ax, label=name, shrink=0.6)
        else:
            ax.text2D(0.5, 0.5, "no voxels past threshold", transform=ax.transAxes, ha="center")
        spans = [float(c[-1] - c[0]) if len(c) > 1 else 1.0 for c in coords]
        spans = [s if s > 0 else 1.0 for s in spans]
        zx = z_exaggeration or max(1.0, 0.35 * max(spans[:2]) / spans[2])
        ax.set_box_aspect((spans[0], spans[1], spans[2] * zx), zoom=zoom)
        for d, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            c = coords[d]
            half = (c[1] - c[0]) / 2 if len(c) > 1 else 0.5
            setter(c[0] - half, c[-1] + half)
        if _domain_extents(domain) is None or names[2] == "z":
            ax.invert_zaxis()  # depth increases downwards
        ax.set_xlabel(names[0], labelpad=-8)
        ax.set_ylabel(names[1], labelpad=-8)
        ax.set_zlabel(f"{names[2]} (×{zx:.2g})" if zx > 1.01 else names[2], labelpad=-8)
        ax.tick_params(labelsize="x-small", pad=-3)
        from matplotlib.ticker import MaxNLocator

        for axis_ in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis_.set_major_locator(MaxNLocator(4))
        ax.view_init(elev=elev, azim=azim)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor(t.surface)
            pane.set_edgecolor(t.grid)
        ax.grid(False)
        ax.set_title(
            title or f"{name or 'field'} voxels {'<' if mode == 'below' else '>'} {threshold:.3g}"
        )
        return fig


def plot_complex(
    x: Any,
    *,
    field: str | None = None,
    domain: Any = None,
    title: str | None = None,
    complex_stack: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """Magnitude (``magma``) and phase (``twilight`` on ``[-π, π]``) of a complex 1-D / 2-D
    field (3-D: representative slice)."""
    a, name = resolve_field(x, field)
    a = squeeze_field(as_complex(a, complex_stack))
    if not np.iscomplexobj(a):
        a = a.astype(np.complex128)
    if a.ndim == 3:
        a = take_slice(a, representative_slice(a))
    with styled(dark):
        fig, axes = new_figure(1, 2, figsize=figsize(2, 1, cbar=2, title=True))
        if a.ndim == 1:
            plot_line(axes[0, 0], np.abs(a), domain=domain)
            axes[0, 0].set_title("magnitude")
            grid_on(axes[0, 0])
            plot_line(axes[0, 1], np.angle(a), domain=domain, color=theme().palette[1])
            axes[0, 1].set_ylim(-math.pi, math.pi)
            axes[0, 1].set_title("phase [rad]")
            grid_on(axes[0, 1])
        else:
            show_image(
                axes[0, 0],
                np.abs(a),
                spec=cmap_for(quantity="magnitude"),
                domain=domain,
                title="magnitude",
                colorbar=True,
            )
            show_image(
                axes[0, 1],
                np.angle(a),
                spec=cmap_for(quantity="phase"),
                domain=domain,
                title="phase [rad]",
                colorbar=True,
            )
        fig.suptitle(title or f"{name or 'field'} (complex)")
        return fig


# ---------------------------------------------------------------------------------------------
# anomaly helpers (overlays, line profiles, 3-D cross-sections)
# ---------------------------------------------------------------------------------------------
def anomaly_background(a: Any) -> float:
    """Background level of a field: its median (the bulk value for localized anomalies)."""
    arr = np.asarray(a, dtype=float) if not np.iscomplexobj(a) else np.abs(a)
    fin = arr[np.isfinite(arr)]
    return float(np.median(fin)) if fin.size else 0.0


def anomaly_centroid(
    vol: Any,
    *,
    mask: Any = None,
    background: float | None = None,
    level: float = 0.5,
) -> tuple[int, ...]:
    """Grid index of the centroid of the dominant anomaly of a 2-D / 3-D field.

    The anomaly region is ``mask`` when given (e.g. GT defects), else ``|a − background| ≥
    level · max |a − background|`` (background: the median). Among its connected components the
    one with the largest total deviation wins (several defects: the strongest one), and its
    deviation-weighted centroid is returned. A featureless field returns the grid centre.
    """
    a = squeeze_field(to_numpy(vol))
    if np.iscomplexobj(a):
        a = np.abs(a)
    center = tuple(int(s) // 2 for s in a.shape)
    region, w = None, None
    if mask is not None:
        m = squeeze_field(to_numpy(mask))
        if np.iscomplexobj(m):
            m = np.abs(m)
        if m.shape == a.shape and bool(np.any(m > 0.5)):
            region = m > 0.5
            w = region.astype(float)
    if region is None:
        bg = anomaly_background(a) if background is None else float(background)
        w = np.abs(np.nan_to_num(a - bg))
        mx = float(w.max()) if w.size else 0.0
        if mx <= 0:
            return center
        region = w >= level * mx
    try:
        from scipy import ndimage

        lab, n = ndimage.label(region, structure=np.ones((3,) * a.ndim))
        if n > 1:
            sums = ndimage.sum(w, lab, index=np.arange(1, n + 1))
            region = lab == int(np.argmax(sums)) + 1
    except ImportError:  # pragma: no cover - scipy is a core dependency
        pass
    ww = np.where(region, w, 0.0)
    tot = float(ww.sum())
    if tot <= 0:
        return center
    grids = np.indices(a.shape)
    # round half up: an even-width defect maps to its upper middle cell
    cents = [math.floor(float((gi * ww).sum() / tot) + 0.5) for gi in grids]
    return tuple(int(np.clip(c, 0, s - 1)) for c, s in zip(cents, a.shape))


def anomaly_levels(
    gt: Any,
    *,
    background: float | None = None,
    fraction: float = 0.5,
    min_share: float = 0.2,
) -> list[float]:
    """Contour levels outlining the anomalies of a ground truth: ``background ± fraction ×`` the
    largest deviation on each side (a side whose deviation is below ``min_share`` of the other is
    skipped, so a positive-only source map gets one level and a signed map two)."""
    a = squeeze_field(to_numpy(gt))
    if np.iscomplexobj(a):
        a = np.abs(a)
    fin = a[np.isfinite(a)]
    if fin.size == 0:
        return []
    bg = float(np.median(fin)) if background is None else float(background)
    up, down = float(fin.max()) - bg, bg - float(fin.min())
    big = max(up, down)
    if big <= 0:
        return []
    levels = []
    if down >= min_share * big and down > 0:
        levels.append(bg - fraction * down)
    if up >= min_share * big and up > 0:
        levels.append(bg + fraction * up)
    return levels


def draw_levels(
    ax: Axes,
    arr: Any,
    levels: Sequence[float],
    extent: tuple[float, float, float, float],
    *,
    style: str = "gt",
    origin: str = "lower",
    color: str | None = None,
    linewidth: float | None = None,
) -> bool:
    """Iso-lines of a 2-D array (field convention ``a[x, y]``) on an image panel.

    ``style="gt"``: solid white lines with a dark halo (visible on every colormap); ``"recon"``:
    dashed aqua lines. ``origin="upper"`` matches panels whose second axis points down (depth
    cross-sections). The axes limits are preserved. Returns True when something was drawn.
    """
    import matplotlib.patheffects as pe

    a = squeeze_field(to_numpy(arr))
    if np.iscomplexobj(a):
        a = np.abs(a)
    a = a.astype(float)
    if a.ndim != 2 or min(a.shape) < 2 or not levels:
        return False
    fin = a[np.isfinite(a)]
    if fin.size == 0:
        return False
    lv = sorted({float(v) for v in levels if float(fin.min()) < float(v) < float(fin.max())})
    if not lv:
        return False
    x0, x1, y0, y1 = extent
    xs = _cell_centers(a.shape[0], x0, x1)
    ys = (
        _cell_centers(a.shape[1], y1, y0)
        if origin == "upper"
        else _cell_centers(a.shape[1], y0, y1)
    )
    if style == "gt":
        col, ls, lw = "#ffffff", "solid", 1.0
    else:
        col, ls, lw = theme().palette[2], (0, (3.0, 1.6)), 0.9
    lw = linewidth or lw
    xl, yl = ax.get_xlim(), ax.get_ylim()
    cs = ax.contour(xs, ys, a.T, levels=lv, colors=[color or col], linewidths=lw, linestyles=[ls])
    cs.set_path_effects([pe.withStroke(linewidth=lw + 1.3, foreground="#000000", alpha=0.45)])
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    return True


def line_profile(a: np.ndarray, center: Sequence[int], axis: int) -> np.ndarray:
    """Values of ``a`` along ``axis`` through the grid point ``center``."""
    idx: list[Any] = [int(c) for c in center]
    idx[axis % a.ndim] = slice(None)
    return np.asarray(a[tuple(idx)])


def _cut_line(ax: Axes, value: float, horizontal: bool) -> None:
    """A thin dashed marker line (white with a dark halo) showing where a profile is cut."""
    import matplotlib.patheffects as pe

    kw = {
        "color": "#ffffff",
        "lw": 0.8,
        "ls": (0, (3.0, 2.0)),
        "alpha": 0.95,
        "path_effects": [pe.withStroke(linewidth=1.8, foreground="#000000", alpha=0.4)],
    }
    (ax.axhline if horizontal else ax.axvline)(value, **kw)


# ---------------------------------------------------------------------------------------------
# the standard reconstruction figure
# ---------------------------------------------------------------------------------------------
#: Panel names of :func:`compare_fields` (``panels=``).
PANELS = ("measurement", "gt", "recon", "error", "overlay", "profile")
EXTRA_PANELS = ("error", "overlay", "profile")


def _recon_items(recons: Any, field: str | None) -> list[tuple[str, np.ndarray]]:
    if recons is None:
        return []
    if isinstance(recons, Mapping) and not (
        hasattr(recons, "fields") or hasattr(recons, "noise_std")
    ):
        out = []
        for lab, v in recons.items():
            a, _ = resolve_field(v, field)
            out.append((str(lab), a))
        return out
    a, _ = resolve_field(recons, field)
    return [("reconstruction", a)]


def _gt_array(gt: Any, field: str | None) -> tuple[np.ndarray | None, str | None]:
    if gt is None:
        return None, field
    return resolve_field(gt, field)


def _first_field_name(obj: Any) -> str | None:
    """First field name of a Result / fields dict (``None`` for bare tensors)."""
    fields = getattr(obj, "fields", None)
    if isinstance(fields, Mapping) and fields:
        return str(next(iter(fields)))
    if isinstance(obj, Mapping) and obj and not hasattr(obj, "noise_std"):
        return str(next(iter(obj)))
    return None


def _field_name(field: str | None, gt: Any, recons: Any) -> str | None:
    """The compared field: explicit ``field`` > first GT field > first reconstruction field."""
    if field:
        return field
    name = _first_field_name(gt) if gt is not None else None
    if name:
        return name
    if hasattr(recons, "fields"):
        return _first_field_name(recons)
    if isinstance(recons, Mapping):  # {method: reconstruction}
        for v in recons.values():
            name = _first_field_name(v)
            if name:
                return name
    return None


def _resolve_metrics(
    metrics: Any, items: list[tuple[str, np.ndarray]], gt: np.ndarray | None
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {lab: {} for lab, _ in items}
    if metrics is None:
        return out
    if (
        isinstance(metrics, Mapping)
        and metrics
        and all(isinstance(v, Mapping) for v in metrics.values())
    ):
        for lab in out:
            out[lab] = {k: float(v) for k, v in dict(metrics.get(lab, {})).items()}
        return out
    if gt is None:
        return out
    for lab, a in items:
        out[lab] = field_metrics(a, gt, metrics)
    return out


def _title_with_metrics(label: str, m: Mapping[str, float] | None, max_items: int = 2) -> str:
    txt = metrics_label(m, max_items)
    return f"{label}\n{txt}" if txt else label


def panel_set(panels: Any = None, error: bool = True) -> tuple[str, ...]:
    """Normalize ``compare_fields(panels=...)``: ``None`` → measurement, GT, reconstruction and
    the signed error; a string names the extra panel (``"overlay"`` → measurement, GT,
    reconstruction, overlay); a sequence lists the panels (any of :data:`PANELS`)."""
    if panels is None:
        chosen = ["measurement", "gt", "recon"] + (["error"] if error else [])
    elif isinstance(panels, str):
        if panels not in PANELS:
            raise ValueError(f"unknown panel {panels!r}; use one of {PANELS}")
        chosen = ["measurement", "gt", "recon"] + ([panels] if panels not in ("recon",) else [])
    else:
        chosen = [str(p) for p in panels]
        bad = [p for p in chosen if p not in PANELS]
        if bad:
            raise ValueError(f"unknown panel(s) {bad}; use {PANELS}")
    if not error:
        chosen = [p for p in chosen if p != "error"]
    return tuple(dict.fromkeys(chosen))


@dataclass
class Comparison:
    """Everything a comparison layout needs (resolved by :func:`compare_fields`).

    ``g`` and ``items`` hold the *displayed* arrays (after the field transform); ``tag`` is the
    text appended to panel titles (transform note or field label) and ``label`` the display name.
    """

    name: str | None
    label: str
    tag: str
    g: np.ndarray | None
    items: list[tuple[str, np.ndarray]]
    mets: dict[str, dict[str, float]]
    spec: CmapSpec
    domain: Any
    mask: np.ndarray | None
    measurement: Any
    instance: Any
    hints: dict[str, Any]
    show_meas: bool
    show_gt: bool
    extras: tuple[str, ...]
    peaks: bool | None
    axis: int
    slices: Any
    profile_axis: int | None

    def lims(self) -> tuple[float, float]:
        vals = [a for a in [self.g] + [a for _, a in self.items] if a is not None]
        return color_limits(vals, self.spec)

    def gt_title(self) -> str:
        return "ground truth" + (f" · {self.tag}" if self.tag else "")

    def recon_title(self, lab: str, n_metrics: int = 2) -> str:
        head = f"{lab} · {self.tag}" if self.tag else lab
        return _title_with_metrics(head, self.mets.get(lab), n_metrics)


def _field_spec(
    name: str | None, ref: np.ndarray, quantity: str | None, tf: Any, cmap: Any
) -> CmapSpec:
    from .hints import transform_is_magnitude, transform_name
    from .style import spec_from_hint

    hinted = spec_from_hint(cmap, ref, quantity or quantity_of(name))
    if hinted is not None:
        return hinted
    if quantity is None and transform_is_magnitude(tf):
        return cmap_for(quantity="magnitude")
    if quantity is None and transform_name(tf) == "log":
        return CmapSpec("viridis", "sequential", quantity_of(name) or "generic")
    return cmap_for(name, ref, quantity)


def compare_fields(
    gt: Any,
    recons: Any,
    measurement: Any = None,
    *,
    kind: str = "auto",
    field: str | None = None,
    quantity: str | None = None,
    metrics: Any = "auto",
    error: bool = True,
    panels: Any = None,
    title: str | None = None,
    domain: Any = None,
    slices: Sequence[int] | int | None = None,
    axis: int | None = None,
    peaks: bool | None = None,
    mask: Any = None,
    complex_stack: str | None = None,
    instance: Any = None,
    hints: Mapping[str, Any] | None = None,
    transform: Any = None,
    cmap: Any = None,
    profile_axis: int | None = None,
    dark: bool | None = None,
) -> Figure:
    """The standard reconstruction figure: measurement | GT | reconstruction(s) | extra panels.

    Args:
        gt: ground truth (tensor, fields dict, or ``None`` for real data without GT).
        recons: one reconstruction (tensor / :class:`~nefi.solve.Result` / fields dict) or a
            mapping ``{method: reconstruction}``.
        measurement: optional :class:`~nefi.measurement.Measurement` (or data tensor) drawn with
            :func:`nefi.viz.measurement.draw_measurement` (complex data: magnitude and phase).
        kind: ``"auto"`` (from dimensionality / dtype), ``"1d"``, ``"2d"``, ``"3d"``,
            ``"complex"``.
        field: which field to compare (default: the first GT / result field).
        quantity: override the colormap quantity (``"density"``, ``"diffusivity"``, ...).
        metrics: ``"auto"`` (PSNR / SSIM or relative error, computed on the *displayed* maps,
            i.e. after ``transform``), a dict ``{name: fn(pred, gt)}``, precomputed
            ``{method: {metric: value}}``, or ``None``.
        error: include the signed-error panels (shorthand for dropping ``"error"`` from
            ``panels``).
        panels: which panels to draw — any of :data:`PANELS` (``"measurement"``, ``"gt"``,
            ``"recon"``, and the extras ``"error"`` (signed error, ``RdBu_r`` centred at 0),
            ``"overlay"`` (GT contours — solid white — over the reconstruction, with the
            reconstruction's own contour at the same level dashed) and ``"profile"`` (a line cut
            through the anomaly centroid, GT dashed vs reconstructions; the cut is marked on the
            image panels; 3-D: cuts along x, y and depth)). A string names one extra panel, e.g.
            ``panels="overlay"``. Default: measurement, GT, reconstruction(s), error.
        title: figure title.
        domain: :class:`~nefi.domain.Domain` for physical axes / slice positions.
        slices: 3-D: slice indices or their number (default: up to 6 evenly spaced).
        axis: 3-D slicing / projection axis (default: the ``volume_axis`` hint, else last = z).
        peaks: mark local maxima (GT circles, reconstruction crosses); ``None`` = automatic for
            sparse non-negative maps (NeTMY-style sources).
        mask: contour overlay / anomaly region (e.g. a support or defect mask, field-shaped).
        complex_stack: ``"first"`` / ``"last"`` if real and imaginary parts are stacked.
        instance: optional instance: ``measurement_image`` hook and display hints
            (:mod:`nefi.viz.hints`).
        hints: explicit display hints (win over the instance's): ``field_transform``,
            ``field_label``, ``field_cmap``, ``volume_axis``, measurement hints...
        transform: display transform of GT and reconstructions (overrides the
            ``field_transform`` hint): ``"zero_mean"`` (fields defined up to a constant — each
            map loses its own mean and the titles say so), ``"abs"``, ``"log"``,
            ``"grad_magnitude"``, ``"curl_magnitude"``, a callable...
        cmap: colormap (overrides the ``field_cmap`` hint): a matplotlib name, ``"signed"``, a
            quantity, or a :class:`~nefi.viz.style.CmapSpec`.
        profile_axis: axis of the 2-D line cut (default 0 = x).
        dark: dark theme.

    Returns:
        The figure. Panel axes are followed by colorbar axes in ``fig.axes``.
    """
    from .hints import (
        apply_transform,
        field_hint,
        instance_hints,
        transform_name,
        transform_note,
        volume_axis,
    )

    name = _field_name(field, gt, recons)
    g, gname = _gt_array(gt, name)
    name = name or gname
    items = _recon_items(recons, name)
    if not items and g is None:
        raise ValueError("compare_fields needs a ground truth or at least one reconstruction")
    hn = instance_hints(instance, hints)
    if g is not None:
        g = squeeze_field(as_complex(g, complex_stack))
    items = [(lab, squeeze_field(as_complex(a, complex_stack))) for lab, a in items]
    if g is not None:
        items = [(lab, match_shape(a, g.shape) if a.shape != g.shape else a) for lab, a in items]
    tf = transform if transform is not None else field_hint(hn, "field_transform", name)
    label_hint = field_hint(hn, "field_label", name)
    note = transform_note(tf)
    if transform_name(tf) is not None:
        g = apply_transform(g, tf, domain=domain, instance=instance) if g is not None else None
        items = [
            (lab, apply_transform(a, tf, domain=domain, instance=instance)) for lab, a in items
        ]
    ref = g if g is not None else items[0][1]
    k = detect_kind(ref, kind)
    mets = _resolve_metrics(metrics, items, g)
    chosen = panel_set(panels, error)
    extras = tuple(p for p in EXTRA_PANELS if p in chosen)
    if g is None:
        extras = tuple(p for p in extras if p == "profile")
    m_arr = None
    if mask is not None:
        m_arr = squeeze_field(to_numpy(mask))
        if m_arr.shape != ref.shape:
            m_arr = m_arr if m_arr.ndim == 2 and k != "3d" else None
    display = str(label_hint) if label_hint else (f"{name} ({note})" if note else (name or ""))
    # panel titles say how the maps were modified; a label already names derived quantities
    keep = transform_name(tf) in ("zero_mean", "log") or not label_hint
    tag = note if keep else ""
    spec = (
        _field_spec(name, np.abs(ref) if np.iscomplexobj(ref) else ref, quantity, tf, cmap)
        if k != "complex"
        else cmap_for(quantity="magnitude")
    )
    ctx = Comparison(
        name=name,
        label=display or "field",
        tag=tag,
        g=g,
        items=items,
        mets=mets,
        spec=spec,
        domain=domain,
        mask=m_arr,
        measurement=measurement,
        instance=instance,
        hints=hn,
        show_meas=measurement is not None and "measurement" in chosen,
        show_gt=g is not None and "gt" in chosen,
        extras=extras,
        peaks=peaks,
        axis=volume_axis(hn) if axis is None else int(axis),
        slices=slices,
        profile_axis=profile_axis,
    )
    with styled(dark):
        if k == "1d":
            fig = _compare_1d(ctx)
        elif k == "complex":
            fig = _compare_complex(ctx)
        elif k == "3d":
            from .volume import compare_volume

            fig = compare_volume(ctx)
        else:
            fig = _compare_2d(ctx)
        fig.suptitle(title or f"{ctx.label}: reconstruction vs ground truth")
        return fig


def _meas_panel(
    ax: Axes,
    measurement: Any,
    field_shape: Sequence[int],
    instance: Any,
    hints: Mapping[str, Any] | None = None,
) -> None:
    from .measurement import draw_measurement

    try:
        draw_measurement(ax, measurement, field_shape=field_shape, instance=instance, hints=hints)
    except Exception as e:  # never let an exotic measurement break the figure
        log.debug("measurement panel failed: %s", e)
        message_axes(ax, f"measurement\n{tuple(getattr(measurement, 'shape', ()))}", "measurement")


def _compare_1d(ctx: Comparison) -> Figure:
    t = theme()
    from .measurement import measurement_view

    g, items, domain = ctx.g, ctx.items, ctx.domain
    measurement = ctx.measurement if ctx.show_meas else None
    has_err = "error" in ctx.extras
    view = None
    if measurement is not None:
        try:
            view = measurement_view(
                measurement,
                field_shape=(g if g is not None else items[0][1]).shape,
                instance=ctx.instance,
                hints=ctx.hints,
            )
        except Exception as e:  # pragma: no cover - exotic layouts
            log.debug("measurement view failed: %s", e)
    overlay = view is not None and view.layout == "signal" and view.phase is None
    extra_meas = measurement is not None and not overlay
    ncols = 1 + int(has_err) + int(extra_meas)
    w, h = panel_size()
    fig, axes = new_figure(1, ncols, figsize=(ncols * w * 1.5 + 0.2, h * 1.2 + 0.45))
    ax = axes[0, 0]
    ref_n = (g if g is not None else items[0][1]).shape[0]
    if overlay:
        m = view.data.shape[0]
        xm = None
        if domain is None and m != ref_n:  # map sample i to the field's index axis
            xm = (np.arange(m) + 0.5) * ref_n / m - 0.5
        plot_line(ax, view.data, x=xm, domain=domain, style="data", label="measurement")
    if g is not None and ctx.show_gt:
        plot_line(ax, g, domain=domain, style="gt", label="ground truth")
    for i, (lab, a) in enumerate(items[:7]):
        txt = metrics_label(ctx.mets.get(lab), 1)
        plot_line(ax, a, domain=domain, color=t.palette[i], label=f"{lab} ({txt})" if txt else lab)
    grid_on(ax)
    ax.set_xlabel(_axis_names(domain, 1)[0])
    ax.set_title(ctx.label + (f" · {ctx.tag}" if ctx.tag and ctx.tag not in ctx.label else ""))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=min(4, len(labels)))
    col = 1
    if has_err:
        ax2 = axes[0, col]
        ax2.axhline(0.0, color=t.axis, lw=0.6)
        for i, (lab, a) in enumerate(items[:7]):
            plot_line(ax2, np.real(a) - np.real(g), domain=domain, color=t.palette[i], label=lab)
        grid_on(ax2)
        ax2.set_xlabel(_axis_names(domain, 1)[0])
        ax2.set_title("signed error (recon − GT)")
        col += 1
    if extra_meas:
        _meas_panel(
            axes[0, col],
            measurement,
            (g if g is not None else items[0][1]).shape,
            ctx.instance,
            ctx.hints,
        )
    return fig


def _overlay_panel(
    ax: Axes,
    a: np.ndarray,
    ctx: Comparison,
    lo: float,
    hi: float,
    ext: tuple[float, float, float, float],
    title: str,
) -> Any:
    """Reconstruction image with the GT anomaly contours (solid) and its own (dashed)."""
    im = show_image(ax, a, spec=ctx.spec, vmin=lo, vmax=hi, extent=ext, title=title)
    if ctx.mask is not None and ctx.mask.shape == a.shape:
        draw_levels(ax, ctx.mask.astype(float), [0.5], ext, style="gt")
        return im
    levels = anomaly_levels(ctx.g) if ctx.g is not None else []
    draw_levels(ax, ctx.g, levels, ext, style="gt")
    draw_levels(ax, a, levels, ext, style="recon")
    return im


def _outline_legend(fig: Any, with_recon: bool = True) -> None:
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D

    t = theme()
    halo = [pe.withStroke(linewidth=2.4, foreground="#000000", alpha=0.6)]
    handles = [Line2D([], [], color="#ffffff", lw=1.0, path_effects=halo, label="GT contour")]
    if with_recon:
        handles.append(
            Line2D([], [], color=t.palette[2], lw=0.9, ls=(0, (3.0, 1.6)), label="recon contour")
        )
    fig.legend(handles=handles, loc="outside lower right", ncols=len(handles), fontsize="small")


def _profile_panel(
    ax: Axes,
    ctx: Comparison,
    center: Sequence[int],
    axis: int,
    *,
    title: str | None = None,
    xlabel: str | None = None,
) -> None:
    """Line cut through ``center`` along ``axis``: GT dashed ink, reconstructions solid."""
    t = theme()
    ref = ctx.g if ctx.g is not None else ctx.items[0][1]
    xs = axis_coords(ref.shape[axis], ctx.domain, axis)
    if ctx.g is not None:
        ax.plot(
            xs,
            line_profile(np.real(ctx.g), center, axis),
            color=t.ink,
            lw=1.1,
            ls=(0, (4, 2)),
            label="ground truth",
        )
    for i, (lab, a) in enumerate(ctx.items[:7]):
        ax.plot(xs, line_profile(np.real(a), center, axis), color=t.palette[i], lw=1.3, label=lab)
    grid_on(ax)
    ax.tick_params(labelsize="x-small")
    names = _axis_names(ctx.domain, ref.ndim)
    ax.set_xlabel(xlabel or names[axis])
    ax.set_title(title or f"profile along {names[axis]}")


def _compare_2d(ctx: Comparison) -> Figure:
    g, items = ctx.g, ctx.items
    ref = g if g is not None else items[0][1]
    has_meas, has_gt = ctx.show_meas, ctx.show_gt
    per_item = [p for p in ("overlay", "error") if p in ctx.extras]
    prof = "profile" in ctx.extras
    n = len(items)
    lead = int(has_meas) + int(has_gt)
    if n <= 1:
        nrows, ncols = 1, lead + n + len(per_item) + int(prof)
    else:
        nrows, ncols = 1 + len(per_item), lead + n + int(prof)
    ratios = [1.0] * ncols
    if prof:
        ratios[-1] = 1.45
    n_cb = 1 + int("error" in per_item)
    fig, axes = new_figure(
        nrows,
        ncols,
        figsize=figsize(ncols + 0.45 * int(prof), nrows, cbar=n_cb, title=True),
        width_ratios=ratios,
    )
    spec = ctx.spec
    lo, hi = ctx.lims()
    ext = image_extent(ref.shape, ctx.domain)
    peaks = ctx.peaks
    if peaks is None:
        peaks = g is not None and spec.quantity in ("density", "source") and _is_sparse(g)
    gt_peaks = peak_indices(g) if (peaks and g is not None) else None
    center = anomaly_centroid(g if g is not None else ref, mask=ctx.mask) if prof else None
    p_axis = (ctx.profile_axis if ctx.profile_axis is not None else 0) % 2
    col = 0
    field_axes, ims = [], []
    if has_meas:
        _meas_panel(axes[0, col], ctx.measurement, ref.shape, ctx.instance, ctx.hints)
        col += 1

    def mark_cut(ax: Axes) -> None:
        if center is None:
            return
        other = 1 - p_axis
        v = axis_coords(ref.shape[other], ctx.domain, other)[center[other]]
        _cut_line(ax, float(v), horizontal=p_axis == 0)

    if has_gt:
        ax = axes[0, col]
        ims.append(show_image(ax, g, spec=spec, vmin=lo, vmax=hi, extent=ext, title=ctx.gt_title()))
        if gt_peaks is not None:
            draw_peaks(ax, gt_peaks, g.shape, ext, "gt")
        if ctx.mask is not None:
            draw_contour(ax, ctx.mask, ext)
        mark_cut(ax)
        field_axes.append(ax)
        col += 1
    rec_col0 = col
    for i, (lab, a) in enumerate(items):
        ax = axes[0, col + i]
        ims.append(
            show_image(ax, a, spec=spec, vmin=lo, vmax=hi, extent=ext, title=ctx.recon_title(lab))
        )
        if peaks:
            if gt_peaks is not None:
                draw_peaks(ax, gt_peaks, a.shape, ext, "gt")
            draw_peaks(ax, peak_indices(a), a.shape, ext, "recon")
        if ctx.mask is not None:
            draw_contour(ax, ctx.mask, ext)
        mark_cut(ax)
        field_axes.append(ax)
    col += n
    err_axes, err_ims = [], []
    espec = cmap_for(quantity="error")
    errs = [(lab, a - g) for lab, a in items] if "error" in per_item else []
    elim = color_limits([e for _, e in errs], espec) if errs else (0, 1)
    overlay_drawn = False
    for r, kind in enumerate(per_item):
        for i, (lab, a) in enumerate(items):
            if n <= 1:
                ax = axes[0, col]
            else:
                ax = axes[1 + r, rec_col0 + i]
            if kind == "error":
                e = a - g
                err_ims.append(
                    show_image(
                        ax,
                        e,
                        spec=espec,
                        vmin=elim[0],
                        vmax=elim[1],
                        extent=ext,
                        title=f"error ({lab})" if n > 1 else "signed error",
                    )
                )
                err_axes.append(ax)
            else:
                head = f"GT contours on {lab}" if n > 1 else "overlay: GT contours"
                ims.append(_overlay_panel(ax, a, ctx, lo, hi, ext, head))
                field_axes.append(ax)
                overlay_drawn = True
        if n <= 1:
            col += 1
        else:
            for j in range(rec_col0):
                axes[1 + r, j].remove()
            if prof:
                axes[1 + r, ncols - 1].remove()
    if prof and center is not None:
        names = _axis_names(ctx.domain, 2)
        other = 1 - p_axis
        cv = axis_coords(ref.shape[other], ctx.domain, other)[center[other]]
        _profile_panel(
            axes[0, ncols - 1],
            ctx,
            center,
            p_axis,
            title=f"profile along {names[p_axis]} at {names[other]} = {cv:.3g}",
        )
        axes[0, ncols - 1].legend(loc="best", fontsize="x-small")
    if ims:
        add_colorbar(fig, ims[0], field_axes, label=ctx.label if len(ctx.label) < 24 else None)
    if err_ims:
        add_colorbar(fig, err_ims[0], err_axes)
    handles, labels = [], []
    if peaks and gt_peaks is not None and field_axes:
        for ax in field_axes:
            for hnd, lab in zip(*ax.get_legend_handles_labels()):
                if lab not in labels:
                    handles.append(hnd)
                    labels.append(lab)
    if handles:
        fig.legend(handles, labels, loc="outside lower right", ncols=len(labels))
    elif overlay_drawn:
        _outline_legend(fig, with_recon=ctx.mask is None)
    return fig


def _compare_complex(ctx: Comparison) -> Figure:
    g, items, domain = ctx.g, ctx.items, ctx.domain
    ref = g if g is not None else items[0][1]
    if ref.ndim == 3:
        k = representative_slice(ref)
        g = take_slice(g, k) if g is not None else None
        items = [(lab, take_slice(a, k)) for lab, a in items]
        ref = g if g is not None else items[0][1]
    if ref.ndim == 1:
        t = theme()
        fig, axes = new_figure(1, 2, figsize=figsize(2, 1, panel=(3.0, 2.2), title=True))
        for row, (fn, lab) in enumerate(((np.abs, "magnitude"), (np.angle, "phase [rad]"))):
            ax = axes[0, row]
            if g is not None:
                plot_line(ax, fn(g), domain=domain, style="gt", label="ground truth")
            for i, (m, a) in enumerate(items[:7]):
                plot_line(ax, fn(a), domain=domain, color=t.palette[i], label=m)
            grid_on(ax)
            ax.set_title(lab)
        axes[0, 0].legend(loc="best")
        return fig
    has_meas = ctx.show_meas
    has_err = "error" in ctx.extras and g is not None
    overlay = "overlay" in ctx.extras and g is not None
    prof = "profile" in ctx.extras
    n = len(items)
    entries = ([("ground truth", g)] if (g is not None and ctx.show_gt) else []) + list(items)
    ncols = int(has_meas) + len(entries) + (n if has_err else 0) + int(prof)
    fig, axes = new_figure(2, ncols, figsize=figsize(ncols, 2, cbar=2, title=True))
    mag_spec = cmap_for(quantity="magnitude")
    ph_spec = cmap_for(quantity="phase")
    mags = [np.abs(a) for a in [g] + [a for _, a in items] if a is not None]
    mlo, mhi = color_limits(mags, mag_spec)
    ext = image_extent(ref.shape, domain)
    col = 0
    mag_axes, ph_axes, err_axes = [], [], []
    im_m = im_p = im_e = None
    if has_meas:
        _meas_panel(axes[0, 0], ctx.measurement, ref.shape, ctx.instance, ctx.hints)
        axes[1, 0].remove()
        col = 1
    levels = anomaly_levels(np.abs(g)) if (overlay and g is not None) else []
    for i, (lab, a) in enumerate(entries):
        title = lab if lab == "ground truth" else _title_with_metrics(lab, ctx.mets.get(lab), 1)
        im_m = show_image(
            axes[0, col + i], np.abs(a), spec=mag_spec, vmin=mlo, vmax=mhi, extent=ext, title=title
        )
        if overlay and lab != "ground truth":
            draw_levels(axes[0, col + i], np.abs(g), levels, ext, style="gt")
        im_p = show_image(
            axes[1, col + i],
            np.angle(a),
            spec=ph_spec,
            extent=ext,
            title="phase" if i == 0 else None,
        )
        mag_axes.append(axes[0, col + i])
        ph_axes.append(axes[1, col + i])
    col += len(entries)
    if has_err:
        aspec = cmap_for(quantity="absolute_error")
        errs = [np.abs(a - g) for _, a in items]
        elo, ehi = color_limits(errs, aspec)
        for i, ((lab, a), e) in enumerate(zip(items, errs)):
            im_e = show_image(
                axes[0, col + i],
                e,
                spec=aspec,
                vmin=elo,
                vmax=ehi,
                extent=ext,
                title=f"|error| ({lab})",
            )
            dphi = np.angle(np.exp(1j * (np.angle(a) - np.angle(g))))
            show_image(axes[1, col + i], dphi, spec=ph_spec, extent=ext, title="phase error")
            err_axes.append(axes[0, col + i])
            ph_axes.append(axes[1, col + i])
        col += n
    if prof:
        center = anomaly_centroid(np.abs(g if g is not None else ref), mask=ctx.mask)
        mag_ctx = dataclasses.replace(
            ctx,
            g=None if g is None else np.abs(g),
            items=[(lab, np.abs(a)) for lab, a in items],
        )
        ph_ctx = dataclasses.replace(
            ctx,
            g=None if g is None else np.angle(g),
            items=[(lab, np.angle(a)) for lab, a in items],
        )
        _profile_panel(axes[0, col], mag_ctx, center, 0, title="|·| profile")
        _profile_panel(axes[1, col], ph_ctx, center, 0, title="phase profile")
        axes[0, col].legend(loc="best", fontsize="x-small")
    if im_m is not None:
        add_colorbar(fig, im_m, mag_axes, label="|·|")
    if im_p is not None:
        add_colorbar(fig, im_p, ph_axes, label="rad")
    if im_e is not None:
        add_colorbar(fig, im_e, err_axes)
    return fig


__all__ = [
    "EXTRA_PANELS",
    "KINDS",
    "PANELS",
    "Comparison",
    "add_colorbar",
    "anomaly_background",
    "anomaly_centroid",
    "anomaly_levels",
    "as_complex",
    "axis_coords",
    "compare_fields",
    "default_metrics",
    "depth_mosaic",
    "detect_kind",
    "draw_contour",
    "draw_levels",
    "draw_peaks",
    "field_metrics",
    "image_extent",
    "image_panels",
    "line_profile",
    "match_shape",
    "metrics_label",
    "orthoslices",
    "panel_set",
    "peak_indices",
    "plot_complex",
    "plot_field",
    "plot_line",
    "projections",
    "representative_slice",
    "resolve_field",
    "select_slices",
    "show_image",
    "squeeze_field",
    "take_slice",
    "to_numpy",
    "voxel_threshold",
    "voxel_view",
]
