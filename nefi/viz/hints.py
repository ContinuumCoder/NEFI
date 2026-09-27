"""Display hints: how an instance wants its measurement and its fields drawn.

Hints are plain dicts. They are resolved in this order (later wins):

1. :data:`INSTANCE_HINTS` — built-in defaults per registered instance name;
2. ``instance.viz_hints`` — a class / instance attribute of the instance (same keys);
3. the ``hints=`` argument of a plotting call.

``Measurement.meta["layout"]`` always wins for the measurement layout. Every key is optional;
unknown keys are ignored. :data:`HINT_KEYS` documents them (also in ``docs/visualization.md``).

Field-level keys (``field_transform``, ``field_label``, ``field_cmap``) accept either one value
for every field or a ``{field_name: value}`` mapping (:func:`field_hint`).

Transforms (:func:`apply_transform`) are display-only: they are applied to the ground truth and to
every reconstruction separately (``zero_mean`` removes each map's own mean, which is what a field
defined up to a constant needs) and the panel titles say so.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

#: Documented hint keys → meaning (see the module docstring for the resolution order).
HINT_KEYS: dict[str, str] = {
    "layout": "measurement layout, one of nefi.viz.LAYOUTS",
    "voxel_threshold": "3-D fields: the IoU / voxel rule ('<0.03', '>0.5', 'half_excess', or a "
    "callable) used for isosurfaces and thresholded voxel views",
    "volume_transform": "3-D fields: display-only volume transform before iso-extraction "
    "('smooth', 'sharpen', 'edge_preserve', or a callable); titles say 'display: ...'",
    "word": "name of the stacked axis ('frequencies', 'patterns', ...)",
    "reduce": "compact view of a stack / volume: 'sum' | 'mean' | 'max' | 'std' or a slice index",
    "complex_stack": "'first' / 'last': real and imaginary parts stacked along that axis",
    "axis_values": "name of an instance method returning the stacked axis' values",
    "row": "row-axis label of matrix / sinogram layouts",
    "col": "column-axis label of matrix / sinogram layouts",
    "stack_axis": "axis of a 3-D matrix / sinogram measurement that indexes slices "
    "(default: the axis whose length equals the field's depth, else the last)",
    "measurement_cmap": "colormap of measurement views: matplotlib name, 'signed' (RdBu_r centred "
    "at 0), 'sequential', a quantity ('density', ...) or 'auto'",
    "measurement_transform": "transform of the compact measurement view: 'log', 'abs', "
    "'zero_mean', ... (see TRANSFORMS)",
    "measurement_label": "display name of the measurement (panel titles)",
    "field_cmap": "colormap of the fields (same values as measurement_cmap); str or {field: str}",
    "field_transform": "display transform of the fields (see TRANSFORMS), a callable or the name "
    "of an instance method; str or {field: str}",
    "field_label": "display name of the fields; str or {field: str}",
    "volume_axis": "slicing / projection axis of 3-D fields (default -1: the last axis, depth z)",
    "compare_panel": "extra panel(s) of the gallery's compare figure: 'error' | 'overlay' | "
    "'profile' (or a list of them)",
}

#: Built-in display transforms: name → (description used in panel titles, kind of result).
TRANSFORMS: dict[str, str] = {
    "abs": "|·|",
    "log": "log₁₀",
    "zero_mean": "mean removed",
    "grad_magnitude": "|∇·|",
    "curl_magnitude": "|∇×(· ẑ)|",
    "real": "real part",
    "imag": "imaginary part",
    "phase": "phase",
    "magnitude": "|·|",
}
_ALIASES = {
    "angle": "phase",
    "mean_removed": "zero_mean",
    "demean": "zero_mean",
    "gradient_magnitude": "grad_magnitude",
    "current_magnitude": "curl_magnitude",
    "log10": "log",
    "modulus": "magnitude",
}
_IDENTITY = (None, "", "none", "identity", "raw")

#: Display hints for built-in instances whose data carry no display metadata. Instances override
#: them with a ``viz_hints`` attribute (same keys); an explicit ``meta["layout"]`` always wins.
INSTANCE_HINTS: dict[str, dict[str, Any]] = {
    "nv_relaxometry": {
        "layout": "spectra",
        "word": "frequencies",
        "axis_values": "frequencies",
        # the noise map is dominated by a few bright sources: log scale shows the background
        "measurement_transform": "log",
    },
    "thermal_tomography": {"layout": "frames", "word": "frames", "volume_axis": -1},
    "sparse_view_ct": {"layout": "sinogram", "axis_values": "angles", "row": "view"},
    "eit": {"layout": "stack", "word": "patterns"},
    "darcy_flow": {"layout": "stack", "word": "configs"},
    # the phase gauge is fixed in the model (ZeroMean head); no display transform needed
    "holography": {"layout": "stack", "word": "distances", "reduce": 0},
    "reaction_diffusion": {"layout": "stack", "word": "snapshots"},
    "diffraction_tomography": {
        "layout": "complex",
        "complex_stack": "first",
        "row": "angle",
        "col": "receiver",
    },
    "wave_fwi": {"layout": "traces"},
    # B_z is signed; the unknown is the stream function g, the physically meaningful map (and the
    # instance metric) is the sheet current |K| = |∇×(g ẑ)|
    "current_density": {
        "layout": "image",
        "measurement_cmap": "signed",
        "measurement_label": "B_z",
        "field_transform": {"g": "curl_magnitude"},
        "field_label": {"g": "|J| = |∇×g|"},
    },
}


def instance_hints(instance: Any = None, hints: Mapping[str, Any] | None = None) -> dict:
    """Display hints for ``instance`` (:data:`INSTANCE_HINTS` < ``instance.viz_hints`` <
    explicit ``hints``)."""
    out: dict[str, Any] = {}
    if instance is not None:
        out.update(INSTANCE_HINTS.get(str(getattr(instance, "name", "")), {}))
        own = getattr(instance, "viz_hints", None)
        if callable(own) and not isinstance(own, Mapping):
            try:
                own = own()
            except TypeError:
                own = None
        if isinstance(own, Mapping):
            out.update(own)
    out.update(dict(hints or {}))
    return out


def field_hint(hints: Mapping[str, Any] | None, key: str, field: str | None) -> Any:
    """Value of a field-level hint for ``field``: a plain value applies to every field, a mapping
    ``{field: value}`` only to the fields it names (``"*"`` = default)."""
    if not hints or key not in hints:
        return None
    v = hints[key]
    if isinstance(v, Mapping):
        if field is not None and field in v:
            return v[field]
        return v.get("*")
    return v


def volume_axis(hints: Mapping[str, Any] | None, default: int = -1) -> int:
    """The ``volume_axis`` hint (slicing axis of 3-D fields), ``default`` when absent."""
    try:
        return int((hints or {}).get("volume_axis", default))
    except (TypeError, ValueError):
        return default


def compare_extras(hints: Mapping[str, Any] | None, ndim: int) -> tuple[str, ...]:
    """Extra panels of the gallery's compare figure: the ``compare_panel`` hint, else
    ``("overlay",)`` for 3-D fields and ``("error",)`` otherwise."""
    v = (hints or {}).get("compare_panel")
    if isinstance(v, str) and v:
        return (v,)
    if isinstance(v, list | tuple) and v:
        return tuple(str(x) for x in v)
    return ("overlay",) if ndim == 3 else ("error",)


# ---------------------------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------------------------
def transform_name(transform: Any) -> str | None:
    """Canonical name of a transform spec (``None`` for the identity)."""
    if transform in _IDENTITY:
        return None
    if callable(transform):
        return getattr(transform, "__name__", "transform")
    key = str(transform).strip().lower()
    return _ALIASES.get(key, key)


def transform_note(transform: Any) -> str:
    """Short text for panel titles (``"mean removed"``, ``"log₁₀"``, ...); ``""`` = identity."""
    name = transform_name(transform)
    if name is None:
        return ""
    return TRANSFORMS.get(name, name.replace("_", " "))


def _spacing(domain: Any, ndim: int) -> list[float]:
    if domain is None:
        return [1.0] * ndim
    sp = getattr(domain, "spacing", None)
    try:
        vals = [float(s) for s in (sp() if callable(sp) else sp)]
        if len(vals) == ndim:
            return vals
    except (TypeError, ValueError):
        pass
    ext = getattr(domain, "extent", domain)
    shape = getattr(domain, "shape", None)
    try:
        if shape is not None and len(ext) == ndim:
            return [(float(hi) - float(lo)) / int(n) for (lo, hi), n in zip(ext, shape)]
    except (TypeError, ValueError):
        pass
    return [1.0] * ndim


def _log10(a: np.ndarray) -> np.ndarray:
    """``log10`` with a floor: 3 σ of the noise when the data dip below zero (negative values of
    a non-negative quantity are noise; their RMS estimates σ), else the 1st percentile of the
    positive values — never more than 4 decades below the maximum."""
    fin = a[np.isfinite(a)]
    if fin.size == 0:
        return a
    top = float(np.max(fin))
    if top <= 0:
        return a
    neg = fin[fin < 0]
    pos = fin[fin > 0]
    if neg.size >= 3:
        floor = 3.0 * float(np.sqrt(np.mean(neg**2)))
    else:
        floor = float(np.percentile(pos, 1.0)) if pos.size else 0.0
    floor = min(max(floor, top * 1e-4), top * 0.5)
    return np.log10(np.clip(a, floor, None))


def apply_transform(
    a: np.ndarray,
    transform: Any,
    *,
    domain: Any = None,
    instance: Any = None,
) -> np.ndarray:
    """Apply a display transform to one array (returns a new array; the input is unchanged).

    Args:
        a: real or complex array (1-D to 3-D).
        transform: ``None`` / ``"none"`` (identity), a name of :data:`TRANSFORMS` (aliases:
            ``"angle"``, ``"mean_removed"``, ``"log10"``, ...), a callable ``f(array) -> array``,
            or the name of a method of ``instance`` taking and returning a tensor / array.
        domain: :class:`~nefi.domain.Domain` (grid spacing of the gradient transforms).
        instance: the instance (method-name transforms).

    ``grad_magnitude`` uses one-sided differences at the edges; ``curl_magnitude`` treats the
    field as a stream function that vanishes outside the field of view (zero padding, central
    differences), so ``|∇×(g ẑ)|`` includes the edge current — the convention of
    :func:`nefi.physics.magnetostatics.stream_to_current`.
    """
    name = transform_name(transform)
    if name is None:
        return a
    if callable(transform):
        return np.asarray(transform(a))
    if name not in TRANSFORMS:
        fn = getattr(instance, str(transform), None) if instance is not None else None
        if callable(fn):
            import torch

            out = fn(torch.as_tensor(a))
            return out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
        raise ValueError(
            f"unknown transform {transform!r}; use one of {sorted(TRANSFORMS)}, a callable or "
            "an instance method name"
        )
    if name in ("abs", "magnitude"):
        return np.abs(a)
    if name == "real":
        return np.real(a).copy()
    if name == "imag":
        return np.imag(a).copy()
    if name == "phase":
        return np.angle(a)
    if np.iscomplexobj(a):
        a = np.abs(a)
    if name == "log":
        return _log10(np.asarray(a, dtype=float))
    if name == "zero_mean":
        fin = a[np.isfinite(a)]
        return a - (float(fin.mean()) if fin.size else 0.0)
    sp = _spacing(domain, a.ndim)
    if name == "grad_magnitude":
        grads = np.gradient(a, *sp) if a.ndim > 1 else [np.gradient(a, sp[0])]
        return np.sqrt(sum(g**2 for g in grads))
    # curl_magnitude: zero-padded central differences of a stream function
    p = np.pad(a, 1)
    sq = np.zeros(a.shape)
    for d in range(a.ndim):
        hi = [slice(1, -1)] * a.ndim
        lo = [slice(1, -1)] * a.ndim
        hi[d], lo[d] = slice(2, None), slice(None, -2)
        sq = sq + ((p[tuple(hi)] - p[tuple(lo)]) / (2.0 * sp[d])) ** 2
    return np.sqrt(sq)


def transform_is_magnitude(transform: Any) -> bool:
    """True for transforms whose result is a non-negative magnitude (``abs``, gradients)."""
    return transform_name(transform) in ("abs", "magnitude", "grad_magnitude", "curl_magnitude")


TransformSpec = str | Callable[[np.ndarray], np.ndarray] | None

__all__ = [
    "HINT_KEYS",
    "INSTANCE_HINTS",
    "TRANSFORMS",
    "TransformSpec",
    "apply_transform",
    "compare_extras",
    "field_hint",
    "instance_hints",
    "transform_is_magnitude",
    "transform_name",
    "transform_note",
    "volume_axis",
]
