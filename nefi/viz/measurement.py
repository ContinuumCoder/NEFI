"""Measurement viewers: spectra, time-series frames, sinograms, receiver traces, k-space masks,
signals, sparse point observations — and :func:`plot_fit` (data vs prediction vs residual with
the noise floor).

The layout of a measurement is detected from its shape relative to the unknown field, from
``Measurement.meta`` hints, or given explicitly (:func:`detect_layout`):

==============  ==============================================  ==============================
layout          typical shape                                   compact view (gallery tile)
==============  ==============================================  ==============================
``signal``      ``(n,)`` for a 1-D field                        line / dots
``vector``      ``(n_sensors,)`` for a 2-/3-D field             dots vs sensor index
``image``       field-shaped ``(H, W)``                         image (field orientation)
``points``      field-shaped + sparse ``mask``                  scatter of observed pixels
``matrix``      ``(n_views, n_detectors)``, other 2-D           image, rows vertical
``sinogram``    ``matrix`` with ``meta["angles"]``              image (angle × detector)
``spectra``     ``(n_freq, H, W)`` with frequency metadata      Σ over frequencies (noise map)
``frames``      ``(n_t, H, W)`` with ``meta["frame_times"]``    mean over time
``stack``       ``(n, H, W)`` without metadata                  mean over the leading axis
``traces``      ``(n_src, n_rec, n_t)``                         gather of the middle source
``kspace``      complex / masked Fourier samples                log-magnitude (masked)
``complex``     any complex tensor                              magnitude + phase
``volume``      field-shaped 3-D ``(nx, ny, nz)``               max (signed: mean) projection
==============  ==============================================  ==============================

3-D ``matrix`` / ``sinogram`` data (sinogram stacks ``(n_views, n_det, n_z)``) are shown one slice
at a time: the slice axis is the ``stack_axis`` hint, else the axis whose length equals the
field's depth, else the last one.

Colormaps: signed data (:func:`nefi.viz.style.is_signed`: min < 0 < max beyond noise) get a
diverging map centred at 0, other data ``viridis``; complex data are shown as magnitude and
phase. Instances can steer all of this without code changes through ``meta["layout"]`` (one of
the names above), ``meta["frame_times"]`` / ``meta["freqs"]`` / ``meta["angles"]``, display hints
(:mod:`nefi.viz.hints`: ``measurement_cmap``, ``measurement_transform``, ``measurement_label``,
``stack_axis``, ...) or by implementing ``measurement_image(measurement) -> Tensor`` (used for the
compact view when present).
"""

from __future__ import annotations

import logging
import math
import textwrap
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .fields import (
    add_colorbar,
    axis_coords,
    image_extent,
    show_image,
    squeeze_field,
    to_numpy,
)
from .hints import INSTANCE_HINTS, apply_transform, instance_hints, transform_note
from .style import (
    CmapSpec,
    blank_axes,
    cmap_for,
    color_limits,
    figsize,
    format_metric,
    get_cmap,
    grid_on,
    new_figure,
    panel_size,
    plain_log_ticks,
    spec_from_hint,
    styled,
    theme,
)

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

log = logging.getLogger("nefi")

LAYOUTS = (
    "signal",
    "vector",
    "image",
    "points",
    "matrix",
    "sinogram",
    "spectra",
    "frames",
    "stack",
    "traces",
    "kspace",
    "complex",
    "volume",
)
_LAYOUT_KEYS = ("layout", "measurement_layout", "kind")
_TIME_KEYS = ("frame_times", "times", "time", "t_obs")
_FREQ_KEYS = ("freqs", "frequencies", "frequency", "omega", "omegas")
_ANGLE_KEYS = ("angles", "thetas", "view_angles")
_STACK_LAYOUTS = ("spectra", "frames", "stack")
#: Layouts that fit a 2-D ``measurement_image`` hook result as it is (others fall back to image).
_HOOK_LAYOUTS = ("image", "points", "matrix", "sinogram", "kspace", "complex", "signal", "vector")


# ---------------------------------------------------------------------------------------------
# unpacking / detection
# ---------------------------------------------------------------------------------------------
def unpack(measurement: Any) -> tuple[np.ndarray, np.ndarray | None, dict, float | None]:
    """``(data, mask, meta, noise_std)`` from a :class:`~nefi.measurement.Measurement`, a tensor
    or an array."""
    if hasattr(measurement, "data") and hasattr(measurement, "noise_std"):
        data = to_numpy(measurement.data)
        mask = None if measurement.mask is None else to_numpy(measurement.mask)
        meta = dict(measurement.meta or {})
        ns = measurement.noise_std
    else:
        data, mask, meta, ns = to_numpy(measurement), None, {}, None
    if ns is not None:
        try:
            ns = float(np.mean(to_numpy(ns)))
        except (TypeError, ValueError):
            ns = None
    if mask is not None and mask.shape != data.shape:
        try:
            mask = np.broadcast_to(mask, data.shape)
        except ValueError:
            mask = None
    return data, mask, meta, ns


def _meta_value(meta: Mapping, keys: Sequence[str]) -> Any:
    for k in keys:
        if k in meta and meta[k] is not None:
            return meta[k]
    return None


def _lateral(field_shape: Sequence[int] | None) -> tuple[int, ...] | None:
    if field_shape is None:
        return None
    fs = tuple(int(s) for s in field_shape)
    return fs[:2] if len(fs) == 3 else fs


def detect_layout(
    data: Any,
    meta: Mapping | None = None,
    field_shape: Sequence[int] | None = None,
    mask: Any = None,
) -> str:
    """Guess the measurement layout (see the module table) from shape, dtype and metadata.

    Args:
        data: measurement tensor / array.
        meta: ``Measurement.meta`` (``layout``, ``frame_times``, ``freqs``, ``angles`` hints).
        field_shape: shape of the unknown field (makes the guess much more reliable).
        mask: observation mask (sparse masks on field-shaped data → ``"points"``).
    """
    meta = dict(meta or {})
    for k in _LAYOUT_KEYS:
        hint = meta.get(k)
        if isinstance(hint, str) and hint.lower() in LAYOUTS:
            return hint.lower()
    a = data if isinstance(data, np.ndarray) else to_numpy(data)
    shape = tuple(a.shape)
    fs = tuple(int(s) for s in field_shape) if field_shape is not None else None
    lat = _lateral(fs)
    m = None if mask is None else np.asarray(mask)
    if np.iscomplexobj(a):
        if m is not None and lat is not None and shape[-2:] == tuple(lat[-2:]):
            return "kspace"
        return "complex"
    nd = len(shape)
    if nd == 1:
        return "signal" if fs is None or len(fs) == 1 else "vector"
    if nd == 2:
        if lat is not None and shape == tuple(lat):
            if m is not None and m.shape == shape and float(np.mean(m > 0)) < 0.3:
                return "points"
            return "image"
        if _meta_value(meta, _ANGLE_KEYS) is not None:
            return "sinogram"
        if fs is None:
            return "image"
        if fs is not None and len(fs) == 1 and shape[0] == 2 and shape[1] == fs[0]:
            return "complex"
        return "matrix"
    if nd == 3:
        timed = (
            _meta_value(meta, _TIME_KEYS) is not None or _meta_value(meta, _FREQ_KEYS) is not None
        )
        if fs is not None and len(fs) == 3 and shape == fs and not timed:
            return "volume"  # field-shaped 3-D data (e.g. a blurred volume)
        if fs is None or (lat is not None and shape[1:] == tuple(lat)):
            if _meta_value(meta, _TIME_KEYS) is not None:
                return "frames"
            if _meta_value(meta, _FREQ_KEYS) is not None:
                return "spectra"
            return "stack"
        if _meta_value(meta, _ANGLE_KEYS) is not None:
            return "sinogram"  # a stack of sinograms, e.g. (n_views, n_det, n_z)
        return "traces"
    return "stack"


# ---------------------------------------------------------------------------------------------
# compact view
# ---------------------------------------------------------------------------------------------
# Display hints (INSTANCE_HINTS, instance_hints) live in :mod:`nefi.viz.hints` and are
# re-exported here for backward compatibility.


def _axis_values(instance: Any, hints: Mapping[str, Any], n: int) -> np.ndarray | None:
    name = hints.get("axis_values")
    fn = getattr(instance, name, None) if (instance is not None and name) else None
    if not callable(fn):
        return None
    try:
        v = to_numpy(fn()).ravel()
        return v[:n] if v.shape[0] >= n else None
    except Exception:  # pragma: no cover - hook failures just drop the axis values
        return None


def resolve_layout(
    data: Any,
    meta: Mapping | None,
    field_shape: Sequence[int] | None,
    mask: Any,
    hints: Mapping[str, Any] | None = None,
) -> str:
    """Layout from ``meta["layout"]``, then instance hints, then :func:`detect_layout`."""
    meta = dict(meta or {})
    for k in _LAYOUT_KEYS:
        v = meta.get(k)
        if isinstance(v, str) and v.lower() in LAYOUTS:
            return v.lower()
    if hints and hints.get("layout") in LAYOUTS:
        return str(hints["layout"])
    return detect_layout(data, meta, field_shape, mask)


@dataclass
class MeasurementView:
    """A single-panel representation of a measurement.

    Attributes:
        layout: detected / requested layout.
        kind: ``"line"``, ``"image"`` or ``"scatter"``.
        data: 1-D (line) or 2-D (image) array; images follow ``transpose`` / ``origin``. Complex
            data are stored as their magnitude (see ``phase``).
        label: short description (``"Σ over 50 frequencies"``, ``"gather, source 3"``).
        spec: colormap spec (diverging and centred at 0 for signed data).
        mask: observed-pixel mask of an image view (``None`` = everything observed).
        points: ``(i, j, value)`` arrays of observed pixels for scatter views.
        transpose: images use the field convention (``data[i, j]`` = ``(x, y)``).
        origin: ``"lower"`` or ``"upper"`` (time-down gathers).
        aspect: ``"equal"`` or ``"auto"``.
        xlabel, ylabel: axis labels for full views.
        phase: phase of complex data (same orientation as ``data``), else ``None``.
        head: display name of the measurement (``measurement_label`` hint, default
            ``"measurement"``).
    """

    layout: str
    kind: str
    data: np.ndarray
    label: str
    spec: CmapSpec
    mask: np.ndarray | None = None
    points: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    transpose: bool = True
    origin: str = "lower"
    aspect: str = "equal"
    xlabel: str = ""
    ylabel: str = ""
    phase: np.ndarray | None = None
    head: str = "measurement"


def _reduce_stack(
    a: np.ndarray, how: str | int | None, layout: str, word: str | None = None
) -> tuple[np.ndarray, str]:
    n = a.shape[0]
    word = word or {"spectra": "frequencies", "frames": "frames"}.get(layout, "channels")
    if isinstance(how, int | np.integer):
        i = int(how) % n
        return a[i], f"{word.rstrip('s')} {i} of {n}"
    how = how or ("sum" if layout == "spectra" else "mean")
    fns = {
        "sum": (np.nansum, "Σ over"),
        "mean": (np.nanmean, "mean of"),
        "max": (np.nanmax, "max over"),
        "std": (np.nanstd, "std over"),
    }
    if how not in fns:
        raise ValueError("reduce must be 'sum', 'mean', 'max', 'std' or an index")
    fn, txt = fns[how]
    with warnings.catch_warnings():  # all-NaN pixels (unobserved) are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        out = fn(a, axis=0)
    if how == "sum" and np.isnan(a).all(axis=0).any():
        out = np.where(np.isnan(a).all(axis=0), np.nan, out)
    return out, f"{txt} {n} {word}"


def _measurement_spec(data: np.ndarray) -> CmapSpec:
    """``viridis``, or ``RdBu_r`` centred at 0 for signed data (:func:`.style.is_signed`)."""
    return cmap_for(quantity="measurement", data=data)


def _scatter_view(layout: str, img: np.ndarray, m2: np.ndarray, prefix: str) -> MeasurementView:
    ij = np.argwhere(m2 > 0)
    vals = img[ij[:, 0], ij[:, 1]]
    pct = 100 * float(np.mean(m2 > 0))
    return MeasurementView(
        layout,
        "scatter",
        img,
        f"{prefix}{len(vals)} obs. px ({pct:.0f} %)",
        _measurement_spec(vals),
        m2,
        points=(ij[:, 0].astype(float), ij[:, 1].astype(float), vals),
    )


def _depth_axis(field_shape: Sequence[int] | None, hints: Mapping[str, Any]) -> int | None:
    """Length of the field's depth (volume) axis, ``None`` for 1-D / 2-D fields."""
    if field_shape is None or len(field_shape) != 3:
        return None
    from .hints import volume_axis

    return int(field_shape[volume_axis(hints) % 3])


def stack_axis(
    shape: Sequence[int], field_shape: Sequence[int] | None, hints: Mapping[str, Any] | None
) -> int:
    """Slice axis of a 3-D matrix / sinogram measurement (e.g. ``(n_views, n_det, n_z)``).

    The ``stack_axis`` hint wins; otherwise the (last) axis whose length equals the field's depth
    (``volume_axis``); otherwise the last axis.
    """
    hints = hints or {}
    if "stack_axis" in hints:
        return int(hints["stack_axis"]) % len(shape)
    depth = _depth_axis(field_shape, hints)
    if depth is not None:
        cands = [i for i, s in enumerate(shape) if int(s) == depth]
        if cands:
            return cands[-1]
    return len(shape) - 1


def _axis_letter(field_shape: Sequence[int] | None, hints: Mapping[str, Any]) -> str:
    from .hints import volume_axis

    return "xyz"[volume_axis(hints) % 3] if field_shape is not None else "axis"


def _view_core(
    data: np.ndarray,
    mask: np.ndarray | None,
    layout: str,
    hn: Mapping[str, Any],
    field_shape: Sequence[int] | None,
    reduce: str | int | None,
    source: int | None,
) -> MeasurementView:
    """The compact view of one layout (before transforms / colormap hints)."""
    word = hn.get("word")
    lat = _lateral(field_shape)
    a = data
    while a.ndim > 3:  # e.g. (n_src, n_freq, H, W): take the middle entry of leading dims
        a = a[a.shape[0] // 2]
        mask = mask[mask.shape[0] // 2] if mask is not None and mask.ndim > a.ndim else mask
    if layout in ("complex", "kspace") or np.iscomplexobj(a):
        if a.ndim == 2 and field_shape is not None and len(field_shape) == 1 and a.shape[0] == 2:
            a = a[0] + 1j * a[1]
        if a.ndim == 3:
            a = a[a.shape[0] // 2]
        mag = np.abs(a)
        spec = cmap_for(quantity="magnitude")
        phase = None if layout == "kspace" else np.angle(a)
        if layout == "kspace":
            with np.errstate(divide="ignore"):
                mag = np.log10(mag + 1e-12 * max(float(np.nanmax(mag)), 1e-30))
            spec = CmapSpec("viridis", "sequential", "measurement")
        m2 = mask if (mask is not None and mask.shape == mag.shape) else None
        label = "log₁₀|y|" if layout == "kspace" else "|y|, arg y (complex)"
        if mag.ndim == 1:
            return MeasurementView(layout, "line", mag, label, spec, m2, phase=phase)
        field_like = lat is not None and tuple(mag.shape) == tuple(lat)
        if field_like or layout == "kspace":
            return MeasurementView(layout, "image", mag, label, spec, m2, phase=phase)
        return MeasurementView(
            layout,
            "image",
            mag,
            f"{label}, {mag.shape[0]} {hn.get('row', 'row')}s × {mag.shape[1]}",
            spec,
            m2,
            transpose=False,
            aspect="auto",
            xlabel=str(hn.get("col", "column")),
            ylabel=str(hn.get("row", "row")),
            phase=phase,
        )
    if layout in ("signal", "vector"):
        v = a.ravel() if a.ndim != 1 else a
        return MeasurementView(
            layout,
            "line",
            v,
            f"{v.shape[0]} samples" if layout == "signal" else f"{v.shape[0]} sensors",
            _measurement_spec(v),
            None if mask is None else np.asarray(mask).ravel()[: v.shape[0]],
            xlabel="sample" if layout == "signal" else "sensor",
        )
    if layout in ("image", "points"):
        img = a if a.ndim == 2 else a.reshape(a.shape[0], -1)
        m2 = mask if (mask is not None and mask.shape == img.shape) else None
        if layout == "points" and m2 is not None:
            return _scatter_view(layout, img, m2, "")
        label = "image"
        if m2 is not None:
            label += f" ({100 * float(np.mean(m2 > 0)):.0f} % observed)"
        return MeasurementView(layout, "image", img, label, _measurement_spec(img), m2)
    if layout == "volume":
        if a.ndim != 3:
            return _view_core(a, mask, "image", hn, field_shape, reduce, source)
        from .hints import volume_axis

        ax = volume_axis(hn) % 3
        n = a.shape[ax]
        letter = "xyz"[ax]
        if isinstance(reduce, int | np.integer):
            i = int(reduce) % n
            img, label = np.take(a, i, axis=ax), f"{letter}-slice {i + 1}/{n}"
        else:
            from .style import is_signed

            how = reduce or ("mean" if is_signed(a) else "max")
            fns = {"max": np.nanmax, "mean": np.nanmean, "sum": np.nansum, "std": np.nanstd}
            if how not in fns:
                raise ValueError("volume reduce must be 'max', 'mean', 'sum', 'std' or an index")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                img = fns[how](a, axis=ax)
            label = f"{how} over {letter} ({n} slices)"
        m2 = None
        if mask is not None and mask.shape == a.shape:
            m2 = (np.asarray(mask) > 0).any(axis=ax).astype(float)
        return MeasurementView(layout, "image", img, label, _measurement_spec(img), m2)
    if layout in ("matrix", "sinogram"):
        suffix = ""
        m3 = mask
        if a.ndim == 3:  # a stack of matrices / sinograms: one slice
            sax = stack_axis(a.shape, field_shape, hn)
            n = a.shape[sax]
            k = int(reduce) % n if isinstance(reduce, int | np.integer) else n // 2
            img = np.take(a, k, axis=sax)
            m3 = np.take(mask, k, axis=sax) if mask is not None and mask.shape == a.shape else None
            suffix = f" · {_axis_letter(field_shape, hn)}-slice {k + 1}/{n}"
        else:
            img = a if a.ndim == 2 else a.reshape(a.shape[0], -1)
        m2 = m3 if (m3 is not None and m3.shape == img.shape) else None
        row = str(hn.get("row", "view" if layout == "sinogram" else "row"))
        return MeasurementView(
            layout,
            "image",
            img,
            f"{img.shape[0]} {row}s × {img.shape[1]}{suffix}",
            _measurement_spec(img),
            m2,
            transpose=False,
            aspect="auto",
            xlabel=str(hn.get("col", "detector" if layout == "sinogram" else "column")),
            ylabel=row,
        )
    if layout in _STACK_LAYOUTS:
        if a.ndim == 2:
            a = a[None]
        m3 = mask if (mask is not None and mask.shape == a.shape) else None
        if m3 is not None:  # reduce over observed entries only
            a = np.where(m3 > 0, a, np.nan)
        img, label = _reduce_stack(a, reduce, layout, word)
        m2 = None if m3 is None else (m3 > 0).any(axis=0).astype(float)
        if m2 is not None and float(np.mean(m2 > 0)) < 0.3:
            return _scatter_view(layout, np.nan_to_num(img), m2, f"{label} · ")
        return MeasurementView(layout, "image", img, label, _measurement_spec(img), m2)
    if layout == "traces":
        if a.ndim == 2:
            a = a[None]
        s = a.shape[0] // 2 if source is None else int(source) % a.shape[0]
        gather = a[s]  # (n_rec, n_t)
        return MeasurementView(
            layout,
            "image",
            gather.T,  # rows = time (downwards), columns = receivers
            f"gather, source {s + 1} of {a.shape[0]}",
            cmap_for(quantity="wavefield"),
            transpose=False,
            origin="upper",
            aspect="auto",
            xlabel="receiver",
            ylabel="time sample",
        )
    raise ValueError(f"layout {layout!r} cannot be drawn for shape {a.shape}")


def _apply_view_hints(view: MeasurementView, hn: Mapping[str, Any]) -> MeasurementView:
    """``measurement_transform``, ``measurement_cmap`` and ``measurement_label`` hints."""
    from .hints import transform_name

    tf = hn.get("measurement_transform")
    name = transform_name(tf)
    if name is not None and view.kind in ("image", "line", "scatter"):
        view.data = apply_transform(np.asarray(view.data, dtype=float), tf)
        if view.points is not None:
            i, j, vals = view.points
            view.points = (i, j, apply_transform(np.asarray(vals, dtype=float), tf))
        view.label = f"{view.label} · {transform_note(tf)}"
        if name == "log":
            view.spec = CmapSpec("viridis", "sequential", "measurement")
        elif view.spec.quantity not in ("wavefield", "magnitude"):
            ref = view.points[2] if view.points is not None else view.data
            view.spec = _measurement_spec(ref)
    ref = view.points[2] if view.points is not None else view.data
    spec = spec_from_hint(hn.get("measurement_cmap"), ref, "measurement")
    if spec is not None:
        view.spec = spec
    if hn.get("measurement_label"):
        view.head = str(hn["measurement_label"])
    return view


def measurement_view(
    measurement: Any,
    *,
    field_shape: Sequence[int] | None = None,
    instance: Any = None,
    layout: str = "auto",
    reduce: str | int | None = None,
    source: int | None = None,
    hints: Mapping[str, Any] | None = None,
) -> MeasurementView:
    """Compact single-panel view of a measurement (used by tiles and :func:`compare_fields`).

    Args:
        measurement: :class:`~nefi.measurement.Measurement`, tensor or array.
        field_shape: shape of the unknown (for layout detection).
        instance: optional instance: its ``measurement_image(measurement)`` hook wins (a tensor, or
            ``(tensor, label)``; a 1-D / 2-D result is drawn as it is — an image with its own
            label even when the layout hint describes the raw data as a stack), and its display
            hints (:func:`~nefi.viz.hints.instance_hints`) are used when the data carry no
            metadata.
        layout: ``"auto"`` or one of :data:`LAYOUTS`.
        reduce: stacks: ``"sum"`` (spectra default), ``"mean"``, ``"max"``, ``"std"`` or an index;
            volumes: ``"max"`` (default; ``"mean"`` for signed data), ``"mean"``, ``"sum"``,
            ``"std"`` or a slice index; 3-D sinogram stacks: a slice index (default: middle).
        source: traces: source index (default: the middle one).
        hints: extra display hints (see :data:`~nefi.viz.hints.HINT_KEYS`): ``measurement_cmap``,
            ``measurement_transform``, ``measurement_label``, ``stack_axis``, ``complex_stack``...
    """
    data, mask, meta, _ = unpack(measurement)
    hn = instance_hints(instance, hints)
    hook = getattr(instance, "measurement_image", None) if instance is not None else None
    hooked, hook_label = False, None
    if callable(hook) and hasattr(measurement, "data"):
        try:
            out = hook(measurement)
            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], str):
                out, hook_label = out
            img = to_numpy(out)
            if img.ndim in (1, 2, 3):
                data = img
                mask = mask if (mask is not None and mask.shape == data.shape) else None
                # a 1-D / 2-D hook result is already the compact view: draw it as it is
                hooked = img.ndim in (1, 2) and layout == "auto"
        except Exception as e:  # pragma: no cover - hook failures fall back to raw data
            log.debug("measurement_image hook failed: %s", e)
    stacked = hn.get("complex_stack")
    if stacked and not np.iscomplexobj(data):
        from .fields import as_complex

        data = as_complex(data, stacked)
        if mask is not None and mask.shape != data.shape:
            mask = None
    if layout == "auto":
        meta_hint = next(
            (str(meta[k]).lower() for k in _LAYOUT_KEYS if isinstance(meta.get(k), str)), None
        )
        if meta_hint in LAYOUTS:
            layout = meta_hint
        elif hn.get("layout") in LAYOUTS:
            layout = str(hn["layout"])
        else:
            layout = detect_layout(data, meta, field_shape, mask)
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; use one of {LAYOUTS}")
    if hooked and layout not in _HOOK_LAYOUTS:
        # stack / volume / traces layouts describe the raw data, not the hook's image
        layout = (
            "image"
            if data.ndim == 2
            else ("signal" if field_shape is None or len(field_shape) == 1 else "vector")
        )
    reduce = hn.get("reduce") if reduce is None else reduce
    view = _view_core(data, mask, layout, hn, field_shape, reduce, source)
    if hooked:
        own = hook_label or hn.get("measurement_image_label")
        if own:
            view.label = str(own)
    return _apply_view_hints(view, hn)


def _complex_label(ax: Axes, text: str) -> None:
    t = theme()
    ax.text(
        0.03,
        0.95,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize="x-small",
        color=t.ink,
        bbox={"boxstyle": "round,pad=0.15", "fc": t.surface, "ec": "none", "alpha": 0.75},
    )


def _draw_complex_split(
    ax: Axes, view: MeasurementView, title: str, domain: Any, colorbar: bool
) -> None:
    """Magnitude and phase of a complex view in two insets of ``ax`` (side by side for
    field-shaped images, stacked otherwise)."""
    t = theme()
    blank_axes(ax, frame=False)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)
    ax.set_title(title)
    side = view.kind == "image" and view.aspect == "equal"
    boxes = (
        ([0.0, 0.18, 0.485, 0.64], [0.515, 0.18, 0.485, 0.64])
        if side
        else (
            [0.0, 0.53, 1.0, 0.47],
            [0.0, 0.0, 1.0, 0.47],
        )
    )
    top, bot = ax.inset_axes(boxes[0]), ax.inset_axes(boxes[1])
    ph = np.asarray(view.phase, dtype=float)
    if view.kind == "line":
        xs = np.arange(view.data.shape[0], dtype=float)
        top.plot(xs, view.data, color=t.palette[0], lw=0.9)
        bot.plot(xs, ph, color=t.palette[1], lw=0.9)
        bot.set_ylim(-math.pi, math.pi)
        for a in (top, bot):
            grid_on(a)
            a.tick_params(labelsize="xx-small", length=1.5)
    else:
        ph_spec = cmap_for("phase", ph)
        for a, arr, spec in ((top, view.data, view.spec), (bot, ph, ph_spec)):
            im = show_image(
                a,
                arr,
                spec=spec,
                transpose=view.transpose,
                origin=view.origin,
                aspect="auto" if not side else "equal",
                mask=view.mask,
                domain=domain if view.transpose else None,
            )
            if colorbar:
                add_colorbar(a.figure, im, a, nticks=3)
    _complex_label(top, "|y|")
    _complex_label(bot, "arg y")


def draw_measurement(
    ax: Axes,
    measurement: Any,
    *,
    field_shape: Sequence[int] | None = None,
    instance: Any = None,
    layout: str = "auto",
    title: str | None = None,
    colorbar: bool = False,
    domain: Any = None,
    axes: bool = False,
    hints: Mapping[str, Any] | None = None,
    split_complex: bool = True,
) -> MeasurementView:
    """Draw the compact view of a measurement into ``ax`` and return the view used (arguments
    as in :func:`measurement_view`; ``axes=True`` keeps ticks and axis labels).

    ``title=None`` uses the measurement's display name (``measurement_label`` hint, default
    ``"measurement"``) above the view label; ``title=""`` keeps only a ``measurement_label``.
    Complex data are drawn as magnitude and phase in two insets (``split_complex=True``).
    """
    t = theme()
    view = measurement_view(
        measurement, field_shape=field_shape, instance=instance, layout=layout, hints=hints
    )
    if title is None:
        head = view.head
    elif title == "":
        head = view.head if view.head != "measurement" else ""
    else:
        head = title
    label = textwrap.fill(view.label, 34) if len(view.label) > 34 else view.label
    full_title = f"{head}\n{label}" if head else label
    if view.phase is not None and split_complex:
        _draw_complex_split(ax, view, full_title, domain, colorbar)
        return view
    if view.kind == "line":
        y = np.real(view.data)
        xs = np.arange(y.shape[0], dtype=float)
        if view.layout != "signal":
            ax.plot(xs, y, color=t.muted, lw=0.7)
        ax.plot(xs, y, ".", color=t.ink2 if view.layout != "signal" else t.muted, ms=2.5)
        if view.spec.kind == "diverging":
            ax.axhline(0.0, color=t.axis, lw=0.6, zorder=0)
        grid_on(ax)
        ax.set_title(full_title)
        if not axes:
            ax.tick_params(labelsize="x-small")
        return view
    if view.kind == "scatter" and view.points is not None:
        xs, ys, vals = view.points
        shape = view.data.shape
        ext = image_extent(shape, domain)
        x0, x1, y0, y1 = ext
        px = x0 + (xs + 0.5) * (x1 - x0) / shape[0]
        py = y0 + (ys + 0.5) * (y1 - y0) / shape[1]
        lo, hi = color_limits(vals, view.spec)
        size = float(np.clip(2400.0 / max(len(vals), 1), 1.5, 30.0))
        sc = ax.scatter(
            px, py, c=vals, cmap=get_cmap(view.spec), vmin=lo, vmax=hi, s=size, edgecolors="none"
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_facecolor(t.bad)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(full_title)
        if colorbar:
            add_colorbar(ax.figure, sc, ax)
        return view
    show_image(
        ax,
        view.data,
        spec=view.spec,
        transpose=view.transpose,
        origin=view.origin,
        aspect=view.aspect,
        mask=view.mask,
        domain=domain if view.transpose else None,
        title=full_title,
        colorbar=colorbar,
        axes=axes,
    )
    if axes and view.xlabel:
        ax.set_xlabel(view.xlabel)
        ax.set_ylabel(view.ylabel)
    return view


# ---------------------------------------------------------------------------------------------
# full viewers
# ---------------------------------------------------------------------------------------------
def plot_signal(
    data: Any,
    *,
    x: Any = None,
    mask: Any = None,
    noise_std: float | None = None,
    title: str | None = None,
    xlabel: str = "sample",
    dark: bool | None = None,
) -> Figure:
    """A 1-D measurement (signal or sensor vector) with an optional ±σ band."""
    y = squeeze_field(to_numpy(data))
    if np.iscomplexobj(y):
        y = np.abs(y)
    xs = to_numpy(x) if x is not None else np.arange(y.shape[0], dtype=float)
    with styled(dark):
        t = theme()
        w, h = panel_size()
        fig, axes = new_figure(1, 1, figsize=(w * 2.2, h * 1.1))
        ax = axes[0, 0]
        if mask is not None:
            m = np.asarray(squeeze_field(to_numpy(mask))).astype(bool)
            if m.shape == y.shape:
                y = np.where(m, y, np.nan)
        if noise_std:
            ax.fill_between(
                xs, y - noise_std, y + noise_std, color=t.muted, alpha=0.18, lw=0, label="±σ"
            )
        ax.plot(xs, y, ".", color=t.ink2, ms=2.5, label="data")
        ax.plot(xs, y, color=t.palette[0], lw=0.8, alpha=0.6)
        grid_on(ax)
        ax.set_xlabel(xlabel)
        ax.set_title(title or f"measurement ({y.shape[0]} samples)")
        if noise_std:
            ax.legend(loc="best")
        return fig


def plot_spectra(
    data: Any,
    *,
    freqs: Sequence[float] | None = None,
    n_slices: int = 3,
    n_pixels: int = 3,
    domain: Any = None,
    title: str | None = None,
    xlabel: str = "frequency",
    reduce: str = "sum",
    transform: Any = None,
    dark: bool | None = None,
) -> Figure:
    """Spectral stacks ``(n_freq, H, W)`` (NeTMY): the noise map ``Σ_ω S(ω, r)``, a few
    frequency slices on a shared scale, and pixel spectra (mean ± std, brightest pixels).

    Args:
        data: ``(n_freq, H, W)`` tensor / array (field orientation in the last two dims).
        freqs: frequency of each slice (default: indices).
        n_slices: frequency slices shown.
        n_pixels: individual pixel spectra (brightest noise-map pixels).
        domain: :class:`~nefi.domain.Domain` of the lateral grid.
        title: figure title.
        xlabel: label of the spectral axis.
        reduce: reduction for the summary map (``"sum"`` = noise map).
        transform: display transform of the summary map (e.g. ``"log"``: log-scaled noise map;
            see :data:`nefi.viz.hints.TRANSFORMS`).
        dark: dark theme.
    """
    a = to_numpy(data)
    if np.iscomplexobj(a):
        a = np.abs(a)
    if a.ndim != 3:
        raise ValueError(f"plot_spectra expects (n_freq, H, W), got {a.shape}")
    nf = a.shape[0]
    f = np.asarray(freqs, dtype=float) if freqs is not None else np.arange(nf, dtype=float)
    idx = sorted({int(round(v)) for v in np.linspace(0, nf - 1, min(n_slices, nf) + 2)[1:-1]})
    idx = idx or [nf // 2]
    with styled(dark):
        t = theme()
        w, h = panel_size()
        ncols = 1 + len(idx) + 1
        fig, axes = new_figure(
            1,
            ncols,
            figsize=(min(11.0, w * (ncols + 0.9) + 2 * 0.6), h + 0.45),
            width_ratios=[1.0] * (ncols - 1) + [1.9],
        )
        summary, lab = _reduce_stack(a, reduce, "spectra")
        shown, note = summary, ""
        if transform not in (None, "", "none"):
            shown = apply_transform(np.asarray(summary, dtype=float), transform)
            note = f" · {transform_note(transform)}"
        from .hints import transform_name

        sspec = (
            CmapSpec("viridis", "sequential", "measurement")
            if transform_name(transform) == "log"
            else cmap_for(quantity="measurement", data=shown)
        )
        show_image(
            axes[0, 0],
            shown,
            spec=sspec,
            domain=domain,
            title=f"noise map\n{lab}{note}",
            colorbar=True,
        )
        sl_spec = cmap_for(quantity="measurement", data=a)
        lo, hi = color_limits([a[i] for i in idx], sl_spec)
        ims, sl_axes = [], []
        for p, i in enumerate(idx):
            ax = axes[0, 1 + p]
            ims.append(
                show_image(
                    ax,
                    a[i],
                    spec=sl_spec,
                    vmin=lo,
                    vmax=hi,
                    domain=domain,
                    title=f"{xlabel} = {f[i]:.3g}",
                )
            )
            sl_axes.append(ax)
        add_colorbar(fig, ims[0], sl_axes)
        ax = axes[0, -1]
        flat = a.reshape(nf, -1)
        mean, std = flat.mean(axis=1), flat.std(axis=1)
        ax.fill_between(
            f, mean - std, mean + std, color=t.muted, alpha=0.2, lw=0, label="mean ± std"
        )
        ax.plot(f, mean, color=t.ink, lw=1.2, label="pixel mean")
        order = np.argsort(summary.ravel())[::-1][: max(0, min(n_pixels, 7))]
        for c, k in enumerate(order):
            i, j = np.unravel_index(int(k), summary.shape)
            ax.plot(f, a[:, i, j], color=t.palette[c], lw=1.1, label=f"pixel ({i}, {j})")
        for i in idx:
            ax.axvline(f[i], color=t.axis, lw=0.6)
        grid_on(ax)
        ax.set_xlabel(xlabel)
        ax.set_title("spectra")
        ax.legend(loc="best")
        fig.suptitle(title or f"spectral measurement {tuple(a.shape)}")
        return fig


def _frame_indices(n: int, k: int) -> list[int]:
    """``k`` frame indices in ``[0, n)``, geometrically spaced (decays change fast early)."""
    if n <= k:
        return list(range(n))
    geo = {int(i) for i in np.unique(np.round(np.geomspace(1, n, k)).astype(int) - 1)}
    lin = [int(i) for i in np.linspace(0, n - 1, k).round().astype(int) if int(i) not in geo]
    idx = sorted(geo | set(lin[: max(0, k - len(geo))]))
    if len(idx) <= k:
        return idx
    return [idx[int(round(j))] for j in np.linspace(0, len(idx) - 1, k)]


def plot_frames(
    data: Any,
    *,
    times: Sequence[float] | None = None,
    n_frames: int = 5,
    pixels: Sequence[tuple[int, int]] | None = None,
    shared: bool = False,
    loglog: bool | str = "auto",
    domain: Any = None,
    title: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """Time-series frames ``(n_t, H, W)`` (NeFTY thermograms): a strip of frames (geometric time
    spacing) and per-pixel decay curves.

    Args:
        data: ``(n_t, H, W)``.
        times: time of each frame (default: indices; ``meta["frame_times"]`` for instances).
        n_frames: frames in the strip.
        pixels: ``(i, j)`` pixels for decay curves (default: hottest, coldest and center of the
            time-mean).
        shared: one color scale for all frames (default: per-frame scale, range in the title —
            decays span orders of magnitude).
        loglog: log-log decay axes (``"auto"``: when all values are positive).
        domain: lateral :class:`~nefi.domain.Domain`.
        title: figure title.
        dark: dark theme.
    """
    a = to_numpy(data)
    if np.iscomplexobj(a):
        a = np.abs(a)
    if a.ndim != 3:
        raise ValueError(f"plot_frames expects (n_t, H, W), got {a.shape}")
    nt = a.shape[0]
    tt = np.asarray(times, dtype=float) if times is not None else np.arange(nt, dtype=float)
    idx = _frame_indices(nt, n_frames)
    mean_map = a.mean(axis=0)
    if pixels is None:
        hi_ = np.unravel_index(int(np.argmax(mean_map)), mean_map.shape)
        lo_ = np.unravel_index(int(np.argmin(mean_map)), mean_map.shape)
        ctr = (mean_map.shape[0] // 2, mean_map.shape[1] // 2)
        pixels = list(dict.fromkeys([tuple(map(int, hi_)), tuple(map(int, lo_)), ctr]))
    with styled(dark):
        t = theme()
        w, h = panel_size()
        ncols = len(idx) + 1
        fig, axes = new_figure(
            1,
            ncols,
            figsize=(min(11.0, w * (ncols + 0.9) + 0.3), h + 0.5),
            width_ratios=[1.0] * len(idx) + [1.9],
        )
        spec = cmap_for(quantity="measurement", data=a)
        lo, hi = color_limits([a[i] for i in idx], spec)
        ims, used = [], []
        for p, i in enumerate(idx):
            ax = axes[0, p]
            if shared:
                ims.append(show_image(ax, a[i], spec=spec, vmin=lo, vmax=hi, domain=domain))
                ax.set_title(f"t = {tt[i]:.3g}")
            else:
                fmin, fmax = float(np.nanmin(a[i])), float(np.nanmax(a[i]))
                show_image(ax, a[i], spec=spec, domain=domain)
                ax.set_title(f"t = {tt[i]:.3g}\n[{fmin:.3g}, {fmax:.3g}]")
            used.append(ax)
        if shared and ims:
            add_colorbar(fig, ims[0], used)
        ax = axes[0, -1]
        use_log = bool(np.all(a > 0)) if loglog == "auto" else bool(loglog)
        tpos = tt.copy()
        if use_log and tpos.min() <= 0:  # shift so the first frame sits at one time step
            step = float(np.min(np.diff(tpos))) if nt > 1 else 1.0
            tpos = tpos - tpos.min() + (step if step > 0 else 1.0)
        for c, (i, j) in enumerate(list(pixels)[:7]):
            ax.plot(tpos, a[:, i, j], color=t.palette[c], lw=1.2, label=f"pixel ({i}, {j})")
        ax.plot(
            tpos, a.reshape(nt, -1).mean(axis=1), color=t.ink, lw=1.0, ls=(0, (4, 2)), label="mean"
        )
        if use_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            plain_log_ticks(ax)
        grid_on(ax, "both")
        ax.set_xlabel("time")
        ax.set_title("decay curves")
        ax.legend(loc="best")
        fig.suptitle(
            title or f"frames {tuple(a.shape)}" + ("" if shared else " (per-frame color scale)")
        )
        return fig


def plot_sinogram(
    data: Any,
    *,
    angles: Sequence[float] | None = None,
    detector: Sequence[float] | None = None,
    n_profiles: int = 3,
    mask: Any = None,
    title: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """Sinogram ``(n_views, n_detectors)``: image (angle vertical, detector horizontal) and a few
    projection profiles."""
    a = to_numpy(data)
    if np.iscomplexobj(a):
        a = np.abs(a)
    if a.ndim != 2:
        raise ValueError(f"plot_sinogram expects (n_views, n_detectors), got {a.shape}")
    nv, nd = a.shape
    ang = np.asarray(angles, dtype=float) if angles is not None else np.arange(nv, dtype=float)
    if angles is not None and float(np.max(np.abs(ang))) <= 2 * math.pi + 1e-6:
        ang = np.degrees(ang)
    det = np.asarray(detector, dtype=float) if detector is not None else np.arange(nd, dtype=float)
    with styled(dark):
        t = theme()
        w, h = panel_size()
        fig, axes = new_figure(1, 2, figsize=(w * 3.6, h * 1.2 + 0.3), width_ratios=[1.3, 1.0])
        ax = axes[0, 0]
        half_a = (ang[1] - ang[0]) / 2 if nv > 1 else 0.5
        half_d = (det[1] - det[0]) / 2 if nd > 1 else 0.5
        ext = (det[0] - half_d, det[-1] + half_d, ang[0] - half_a, ang[-1] + half_a)
        show_image(
            ax,
            a,
            spec=cmap_for(quantity="measurement", data=a),
            transpose=False,
            extent=ext,
            aspect="auto",
            axes=True,
            mask=mask,
            colorbar=True,
            title=f"sinogram ({nv} views)",
        )
        ax.set_xlabel("detector")
        ax.set_ylabel("angle [deg]" if angles is not None else "view")
        ax2 = axes[0, 1]
        for c, i in enumerate(
            sorted({int(round(v)) for v in np.linspace(0, nv - 1, min(n_profiles, nv))})[:7]
        ):
            lab = f"{ang[i]:.0f}°" if angles is not None else f"view {i}"
            ax2.plot(det, a[i], color=t.palette[c], lw=1.2, label=lab)
        grid_on(ax2)
        ax2.set_xlabel("detector")
        ax2.set_title("projections")
        ax2.legend(loc="best")
        fig.suptitle(title or f"sinogram {tuple(a.shape)}")
        return fig


def plot_traces(
    data: Any,
    *,
    source: int | None = None,
    dt: float | None = None,
    t0: float = 0.0,
    mode: str = "both",
    max_traces: int = 48,
    clip: float = 99.0,
    title: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """Receiver traces ``(n_src, n_rec, n_t)`` of one source: image gather (time downwards,
    ``RdBu_r`` centred) and/or a wiggle plot with filled positive lobes.

    Args:
        data: ``(n_src, n_rec, n_t)`` or ``(n_rec, n_t)``.
        source: source index (default: middle).
        dt, t0: time sampling (default: sample indices).
        mode: ``"image"``, ``"wiggle"`` or ``"both"``.
        max_traces: wiggle traces drawn (evenly subsampled).
        clip: percentile of ``|data|`` used as the color / amplitude limit.
        title: figure title.
        dark: dark theme.
    """
    a = to_numpy(data)
    if np.iscomplexobj(a):
        a = np.real(a)
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3:
        raise ValueError(f"plot_traces expects (n_src, n_rec, n_t), got {a.shape}")
    s = a.shape[0] // 2 if source is None else int(source) % a.shape[0]
    g = a[s]  # (n_rec, n_t)
    nr, nt = g.shape
    tt = t0 + (np.arange(nt) * (dt if dt else 1.0))
    amp = float(np.percentile(np.abs(g), clip)) or float(np.abs(g).max()) or 1.0
    modes = ["image", "wiggle"] if mode == "both" else [mode]
    with styled(dark):
        t = theme()
        w, h = panel_size()
        fig, axes = new_figure(1, len(modes), figsize=(w * 1.6 * len(modes) + 0.3, h * 1.35 + 0.3))
        for p, m in enumerate(modes):
            ax = axes[0, p]
            if m == "image":
                half = (tt[1] - tt[0]) / 2 if nt > 1 else 0.5
                show_image(
                    ax,
                    g.T,
                    spec=cmap_for(quantity="wavefield"),
                    vmin=-amp,
                    vmax=amp,
                    transpose=False,
                    origin="upper",
                    extent=(-0.5, nr - 0.5, tt[-1] + half, tt[0] - half),
                    aspect="auto",
                    axes=True,
                    colorbar=True,
                    title=f"gather, source {s}",
                )
                ax.set_xlabel("receiver")
                ax.set_ylabel("time" if dt else "time sample")
            elif m == "wiggle":
                recs = np.unique(np.linspace(0, nr - 1, min(nr, max_traces)).round().astype(int))
                spacing = max(1.0, (recs[1] - recs[0]) if len(recs) > 1 else 1.0)
                scale = 0.9 * spacing / amp
                for r in recs:
                    tr = np.clip(g[r] * scale, -1.5 * spacing, 1.5 * spacing)
                    ax.plot(r + tr, tt, color=t.ink, lw=0.45)
                    ax.fill_betweenx(tt, r, r + tr, where=tr > 0, color=t.ink, alpha=0.55, lw=0)
                ax.set_ylim(tt[-1], tt[0])
                ax.set_xlim(-spacing, nr - 1 + spacing)
                ax.set_xlabel("receiver")
                ax.set_ylabel("time" if dt else "time sample")
                ax.set_title("wiggle")
            else:
                raise ValueError("mode must be 'image', 'wiggle' or 'both'")
        fig.suptitle(title or f"receiver traces {tuple(a.shape)}")
        return fig


def plot_kspace(
    mask: Any,
    data: Any = None,
    *,
    shift: bool = True,
    title: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """k-space sampling mask (and the log-magnitude of the sampled data, if given).

    ``shift=True`` applies ``fftshift`` so the DC component sits in the center.
    """
    m = squeeze_field(to_numpy(mask)).astype(float)
    if m.ndim != 2:
        raise ValueError(f"plot_kspace expects a 2-D mask, got {m.shape}")
    sh = (lambda x: np.fft.fftshift(x)) if shift else (lambda x: x)
    frac = float(np.mean(m > 0))
    with styled(dark):
        n = 1 + int(data is not None)
        fig, axes = new_figure(1, n, figsize=figsize(n, 1, cbar=True, title=True))
        show_image(
            axes[0, 0],
            sh(m),
            spec=CmapSpec("Greys", "sequential", "mask"),
            vmin=0,
            vmax=1,
            title=f"sampling mask ({100 * frac:.1f} %)",
        )
        if data is not None:
            d = squeeze_field(to_numpy(data))
            mag = np.abs(d)
            with np.errstate(divide="ignore"):
                logm = np.log10(mag + 1e-12 * max(float(mag.max()), 1e-30))
            logm = np.where(m > 0, logm, np.nan) if m.shape == logm.shape else logm
            show_image(
                axes[0, 1],
                sh(logm),
                spec=CmapSpec("viridis", "sequential", "measurement"),
                title="log₁₀|y| (sampled)",
                colorbar=True,
            )
        fig.suptitle(title or "k-space")
        return fig


def plot_measurement(
    measurement: Any,
    *,
    layout: str = "auto",
    field_shape: Sequence[int] | None = None,
    instance: Any = None,
    domain: Any = None,
    hints: Mapping[str, Any] | None = None,
    title: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """Full view of a measurement, dispatched on its layout (see the module table).

    Args:
        measurement: :class:`~nefi.measurement.Measurement` (or data tensor).
        layout: ``"auto"`` (metadata → instance hints → shape) or one of :data:`LAYOUTS`.
        field_shape: shape of the unknown (layout detection).
        instance: optional instance (display hints, frequency / angle axes).
        domain: lateral :class:`~nefi.domain.Domain` of field-shaped data.
        hints: extra display hints (:data:`INSTANCE_HINTS` keys).
        title: figure title.
        dark: dark theme.
    """
    data, mask, meta, ns = unpack(measurement)
    hn = instance_hints(instance, hints)
    if hn.get("complex_stack") and not np.iscomplexobj(data):
        from .fields import as_complex

        data = as_complex(data, hn["complex_stack"])
        mask = mask if (mask is not None and mask.shape == data.shape) else None
    if layout == "auto":
        meta_hint = next(
            (str(meta[k]).lower() for k in _LAYOUT_KEYS if isinstance(meta.get(k), str)), None
        )
        layout = (
            meta_hint
            if meta_hint in LAYOUTS
            else (hn["layout"] if hn.get("layout") in LAYOUTS else None)
        ) or detect_layout(data, meta, field_shape, mask)
    lay = layout
    head = str(hn.get("measurement_label") or "measurement")
    title = title or f"{head} {tuple(data.shape)} · {lay}"
    n0 = data.shape[0] if data.ndim else 0
    axis_vals = _axis_values(instance, hn, n0)
    if lay == "volume" and data.ndim == 3 and not np.iscomplexobj(data):
        from .fields import depth_mosaic
        from .hints import volume_axis

        vol = np.where(mask > 0, data, np.nan) if mask is not None else data
        return depth_mosaic(
            vol,
            axis=volume_axis(hn),
            n_slices=8,
            domain=domain,
            spec=spec_from_hint(hn.get("measurement_cmap"), vol, "measurement")
            or _measurement_spec(vol),
            title=title,
            dark=dark,
        )
    if lay in ("signal", "vector"):
        return plot_signal(
            data,
            mask=mask,
            noise_std=ns,
            title=title,
            xlabel="sample" if lay == "signal" else "sensor",
            dark=dark,
        )
    if lay == "spectra" or (lay == "stack" and data.ndim == 3):
        freqs = _meta_value(meta, _FREQ_KEYS)
        freqs = to_numpy(freqs).ravel()[:n0] if freqs is not None else axis_vals
        word = str(hn.get("word", "frequency" if lay == "spectra" else "channel"))
        if mask is not None and mask.shape == data.shape:
            data = np.where(mask > 0, data, np.nan)
        return plot_spectra(
            data,
            freqs=freqs,
            domain=domain,
            title=title,
            xlabel="frequency" if lay == "spectra" else word.rstrip("s"),
            reduce="sum" if lay == "spectra" else "mean",
            transform=hn.get("measurement_transform"),
            dark=dark,
        )
    if lay == "frames":
        times = _meta_value(meta, _TIME_KEYS)
        return plot_frames(
            data,
            times=None if times is None else to_numpy(times).ravel()[:n0],
            domain=domain,
            title=title,
            dark=dark,
        )
    if lay in ("matrix", "sinogram"):
        sino, smask = data, mask if (mask is not None and mask.shape == data.shape) else None
        if data.ndim == 3 and not np.iscomplexobj(data):  # sinogram stack: the middle slice
            sax = stack_axis(data.shape, field_shape, hn)
            k = data.shape[sax] // 2
            sino = np.take(data, k, axis=sax)
            smask = None if smask is None else np.take(smask, k, axis=sax)
            title = f"{title} · {_axis_letter(field_shape, hn)}-slice {k + 1}/{data.shape[sax]}"
        elif data.ndim != 2:
            sino = data.reshape(n0, -1)
        n_rows = sino.shape[0]
        angles = _meta_value(meta, _ANGLE_KEYS)
        angles = to_numpy(angles).ravel() if angles is not None else None
        if angles is None or angles.shape[0] < n_rows:
            angles = _axis_values(instance, hn, n_rows)
        angles = None if angles is None else angles[:n_rows]
        return plot_sinogram(
            sino,
            angles=angles,
            mask=smask if (smask is not None and smask.shape == sino.shape) else None,
            title=title,
            dark=dark,
        )
    if lay == "traces":
        return plot_traces(data, dt=meta.get("dt"), title=title, dark=dark)
    if lay == "kspace":
        m = mask if mask is not None else np.ones(data.shape[-2:])
        while m.ndim > 2:
            m = m[0]
        d = data
        while d.ndim > 2:
            d = d[d.shape[0] // 2]
        return plot_kspace(m, d, title=title, dark=dark)
    if lay == "complex":
        return _plot_complex_measurement(data, hn, field_shape, title, dark)
    # image / points
    with styled(dark):
        fig, axes = new_figure(1, 1, figsize=figsize(1, 1, cbar=True, title=True))
        draw_measurement(
            axes[0, 0],
            measurement,
            field_shape=field_shape,
            instance=instance,
            layout=lay,
            title="",
            colorbar=True,
            domain=domain,
            axes=domain is not None,
        )
        fig.suptitle(title)
        return fig


def _plot_complex_measurement(
    data: np.ndarray, hn: Mapping[str, Any], field_shape: Any, title: str, dark: bool | None
) -> Figure:
    """Magnitude and phase of a complex measurement (matrix or field-shaped)."""
    a = data
    while a.ndim > 2:
        a = a[a.shape[0] // 2]
    lat = _lateral(field_shape)
    field_like = a.ndim == 2 and lat is not None and tuple(a.shape) == tuple(lat)
    if a.ndim == 1 or field_like:
        from .fields import plot_complex

        return plot_complex(a, title=title, dark=dark)
    with styled(dark):
        fig, axes = new_figure(1, 2, figsize=figsize(2, 1, panel=(2.8, 2.4), cbar=2, title=True))
        for ax, arr, spec, lab in (
            (axes[0, 0], np.abs(a), cmap_for(quantity="magnitude"), "magnitude"),
            (axes[0, 1], np.angle(a), cmap_for(quantity="phase"), "phase [rad]"),
        ):
            show_image(
                ax,
                arr,
                spec=spec,
                transpose=False,
                aspect="auto",
                axes=True,
                colorbar=True,
                title=lab,
            )
            ax.set_xlabel(str(hn.get("col", "column")))
            ax.set_ylabel(str(hn.get("row", "row")))
        fig.suptitle(title)
        return fig


# ---------------------------------------------------------------------------------------------
# data fit
# ---------------------------------------------------------------------------------------------
def _align_pred(pred: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Bring a prediction to the data's shape (reshape, or resample trailing dims)."""
    if pred.shape == data.shape:
        return pred
    if pred.size == data.size:
        return pred.reshape(data.shape)
    if pred.ndim == data.ndim:
        import torch

        from ..utils.tensor import resample

        for k in (3, 2, 1):  # resample the trailing k dims when the leading ones match
            if k > data.ndim or pred.shape[: data.ndim - k] != data.shape[: data.ndim - k]:
                continue
            target = data.shape[data.ndim - k :]
            if np.iscomplexobj(pred):
                re = resample(torch.as_tensor(pred.real), target).numpy()
                return re + 1j * resample(torch.as_tensor(pred.imag), target).numpy()
            return resample(torch.as_tensor(pred), target).numpy()
    raise ValueError(f"prediction shape {pred.shape} does not match the data shape {data.shape}")


def residual_stats(
    pred: Any, measurement: Any, noise_std: float | None = None
) -> dict[str, float | None]:
    """Data-fit summary: ``rmse`` (masked), ``noise_std``, ``chi`` = RMSE/σ (≈ 1 when the fit
    reaches the noise floor, Morozov) and the relative residual ``‖r‖ / ‖y‖``."""
    data, mask, _, ns = unpack(measurement)
    p = _align_pred(to_numpy(getattr(pred, "pred", pred)), data)
    r = p - data
    w = np.ones(data.shape) if mask is None else (np.asarray(mask) > 0).astype(float)
    n = max(float(w.sum()), 1.0)
    rmse = float(np.sqrt(np.sum(np.abs(r) ** 2 * w) / n))
    den = float(np.sqrt(np.sum(np.abs(data) ** 2 * w))) or 1e-30
    sigma = noise_std if noise_std is not None else ns
    return {
        "rmse": rmse,
        "noise_std": None if sigma is None else float(sigma),
        "chi": None if not sigma else rmse / float(sigma),
        "rel_residual": float(np.sqrt(np.sum(np.abs(r) ** 2 * w))) / den,
    }


def plot_fit(
    pred: Any,
    measurement: Any,
    *,
    noise_std: float | None = None,
    layout: str = "auto",
    field_shape: Sequence[int] | None = None,
    instance: Any = None,
    domain: Any = None,
    title: str | None = None,
    dark: bool | None = None,
) -> Figure:
    """Data vs prediction vs residual, with the noise floor.

    Args:
        pred: predicted measurement (``Result.pred``, or a :class:`~nefi.solve.Result`).
        measurement: :class:`~nefi.measurement.Measurement` (or data tensor).
        noise_std: noise level σ (default: ``measurement.noise_std``).
        layout: measurement layout (default: detected).
        field_shape: unknown's shape (layout detection).
        instance: optional instance (``measurement_image`` hook).
        domain: lateral :class:`~nefi.domain.Domain`.
        title: figure title (default: the RMSE / σ summary).
        dark: dark theme.

    The suptitle reports RMSE, σ and RMSE/σ: ≈ 1 means the prediction fits the data to the noise
    level (Morozov discrepancy principle); ≪ 1 over-fits the noise; ≫ 1 under-fits.
    """
    data, mask, meta, ns = unpack(measurement)
    sigma = noise_std if noise_std is not None else ns
    p = _align_pred(to_numpy(getattr(pred, "pred", pred)), data)
    r = p - data
    if mask is not None:
        r = np.where(np.asarray(mask) > 0, r, np.nan)
    stats = residual_stats(p, measurement, sigma)
    hn = instance_hints(instance)
    lay = resolve_layout(data, meta, field_shape, mask, hn) if layout == "auto" else layout
    one_d = squeeze_field(data).ndim == 1
    with styled(dark):
        t = theme()
        w, h = panel_size()
        if one_d:
            d1, p1 = squeeze_field(data), squeeze_field(p)
            r = squeeze_field(r)
            if np.iscomplexobj(d1) or np.iscomplexobj(p1):
                d1, p1, r = np.abs(d1), np.abs(p1), np.abs(p1) - np.abs(d1)
            fig, axes = new_figure(
                2, 1, figsize=(w * 2.6, h * 1.65), sharex=True, height_ratios=[1.6, 1.0]
            )
            xs = axis_coords(d1.shape[0], None, 0)
            ax = axes[0, 0]
            ax.plot(xs, d1, ".", color=t.muted, ms=2.5, label="data")
            ax.plot(xs, p1, color=t.palette[0], lw=1.3, label="prediction")
            grid_on(ax)
            ax.legend(loc="best")
            ax.set_title("data vs prediction")
            ax2 = axes[1, 0]
            if sigma:
                ax2.axhspan(-sigma, sigma, color=t.muted, alpha=0.18, lw=0, label="±σ")
                for s_ in (-2 * sigma, 2 * sigma):
                    ax2.axhline(s_, color=t.muted, lw=0.6)
            ax2.axhline(0.0, color=t.axis, lw=0.6)
            ax2.plot(xs, r, ".", color=t.palette[1], ms=2.2, label="residual")
            grid_on(ax2)
            ax2.set_xlabel("sample")
            ax2.set_title("residual (prediction − data)")
            ax2.legend(loc="best", ncols=2)
        else:
            stack = lay in _STACK_LAYOUTS + ("traces",) and data.ndim >= 3
            w4, h4 = figsize(4, 1, cbar=2, title=True)
            fig, axes = new_figure(1, 4, figsize=(w4 + 0.5, h4), width_ratios=[1.0, 1.0, 1.0, 1.3])
            vd = measurement_view(measurement, field_shape=field_shape, layout=lay, hints=hn)
            vp = measurement_view(
                _replace_data(measurement, p), field_shape=field_shape, layout=lay, hints=hn
            )
            lo, hi = color_limits([vd.data, vp.data], vd.spec)
            for ax, v, lab in ((axes[0, 0], vd, "data"), (axes[0, 1], vp, "prediction")):
                show_image(
                    ax,
                    v.data,
                    spec=vd.spec,
                    vmin=lo,
                    vmax=hi,
                    transpose=v.transpose,
                    origin=v.origin,
                    aspect=v.aspect,
                    mask=v.mask,
                    domain=domain if v.transpose else None,
                    title=f"{lab}\n{textwrap.fill(v.label, 30)}",
                )
            add_colorbar(fig, axes[0, 1].images[0], [axes[0, 0], axes[0, 1]])
            ax = axes[0, 2]
            if stack:
                rr = np.abs(r) ** 2
                flat = rr.reshape(rr.shape[0], -1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    per = np.sqrt(np.nanmean(flat, axis=1))
                xs = _axis_values(instance, hn, per.shape[0])
                if xs is None:
                    xs = np.arange(per.shape[0])
                ax.plot(
                    xs, per, color=t.palette[0], lw=1.3, marker="o", ms=2.5, label="RMS residual"
                )
                if sigma:
                    ax.axhline(sigma, color=t.ink, lw=0.9, label="noise floor σ")
                grid_on(ax)
                word = str(hn.get("word", "")).rstrip("s")
                ax.set_xlabel(word or {"spectra": "frequency", "frames": "frame"}.get(lay, "slice"))
                ax.set_title("residual per slice")
                ax.legend(loc="best")
            else:
                hn_res = {
                    k: v
                    for k, v in hn.items()
                    if k not in ("measurement_transform", "measurement_cmap")
                }
                vr = measurement_view(
                    _replace_data(measurement, np.nan_to_num(r)),
                    field_shape=field_shape,
                    layout=lay,
                    hints=hn_res,
                )
                rdata = vr.data
                magnitude = vr.spec.quantity == "magnitude" or lay in ("complex", "kspace")
                espec = cmap_for(quantity="absolute_error" if magnitude else "residual")
                elo, ehi = color_limits(rdata, espec)
                show_image(
                    ax,
                    rdata,
                    spec=espec,
                    vmin=elo,
                    vmax=ehi,
                    transpose=vr.transpose,
                    origin=vr.origin,
                    aspect=vr.aspect,
                    mask=vd.mask,
                    domain=domain if vr.transpose else None,
                    title="|residual|" if magnitude else "residual\n(prediction − data)",
                    colorbar=True,
                )
            ax = axes[0, 3]
            vals = (
                r[np.isfinite(r)]
                if not np.iscomplexobj(r)
                else np.concatenate([np.real(r[np.isfinite(r)]), np.imag(r[np.isfinite(r)])])
            )
            if vals.size:
                lo_, hi_ = (float(v) for v in np.percentile(vals, [0.5, 99.5]))
                if sigma:
                    lo_, hi_ = min(lo_, -4 * sigma), max(hi_, 4 * sigma)
                if hi_ <= lo_:
                    lo_, hi_ = lo_ - 1.0, hi_ + 1.0
                ax.hist(
                    vals,
                    bins=60,
                    range=(lo_, hi_),
                    density=True,
                    color=t.palette[0],
                    alpha=0.85,
                    label="residuals",
                )
                clipped = float(np.mean((vals < lo_) | (vals > hi_)))
                if clipped > 0:
                    ax.text(
                        0.02,
                        0.97,
                        f"{100 * clipped:.1f} % outside",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize="x-small",
                        color=t.ink2,
                    )
                if sigma:
                    xs = np.linspace(lo_, hi_, 200)
                    ax.plot(
                        xs,
                        np.exp(-0.5 * (xs / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)),
                        color=t.ink,
                        lw=1.0,
                        label="N(0, σ²)",
                    )
                ax.legend(loc="best")
            grid_on(ax)
            ax.set_title("residual distribution")
            ax.set_yticks([])
        parts = [format_metric("RMSE", stats["rmse"])]
        if stats["noise_std"]:
            parts.append(format_metric("σ", stats["noise_std"]))
            parts.append(f"RMSE/σ = {stats['chi']:.3g}")
        parts.append(f"rel. residual {100 * stats['rel_residual']:.2g} %")
        fig.suptitle(title or "data fit: " + " · ".join(parts))
        return fig


def _replace_data(measurement: Any, data: np.ndarray) -> Any:
    """A copy of ``measurement`` with new data (tensors are kept as numpy)."""
    if hasattr(measurement, "data") and hasattr(measurement, "noise_std"):
        import torch

        from ..measurement import Measurement

        return Measurement(
            torch.as_tensor(np.asarray(data)),
            measurement.mask,
            measurement.noise_std,
            dict(measurement.meta or {}),
        )
    return data


__all__ = [
    "INSTANCE_HINTS",
    "LAYOUTS",
    "MeasurementView",
    "detect_layout",
    "draw_measurement",
    "instance_hints",
    "measurement_view",
    "plot_fit",
    "plot_frames",
    "plot_kspace",
    "plot_measurement",
    "plot_signal",
    "plot_sinogram",
    "plot_spectra",
    "plot_traces",
    "resolve_layout",
    "residual_stats",
    "stack_axis",
    "unpack",
]
