"""Isosurfaces of 3-D fields: threshold rules, volume-matched iso-levels, display transforms,
marching-cubes meshes and the shaded static renderings of ``gallery_3d.png`` / ``voxels.png``.

Reconstructions of volumetric inverse problems usually recover the structure of the ground truth
with **softer edges**. A voxel view at the GT's fixed threshold then under- (or over-) segments
the reconstruction and makes it look worse than the field is. This module compares level sets
fairly:

* **rules and levels** — the ground truth is cut by a *threshold rule* (:func:`voxel_rule`: the
  instance's IoU rule, e.g. ``α < 0.03`` for thermal tomography, the half-maximum inclusion masks
  of dot3d; else half-way from the GT background to its extreme). A reconstruction is cut at

  - ``"fixed"`` — the same rule (what the instance's IoU metric does),
  - ``"matched"`` — the **volume-matched** level: it encloses as many voxels as the GT does at its
    threshold (:func:`matched_level`; the default of isosurface views),
  - ``"otsu"`` — Otsu's threshold of the reconstruction's histogram (:func:`otsu_level`),
  - ``"manual"`` — a given value;

  :func:`iso_levels` returns every level with the IoU and Dice of the resulting voxel sets.
* **display transforms** — :func:`volume_transform`: ``smooth`` (Gaussian, σ in voxels),
  ``sharpen`` (unsharp mask), ``edge_preserve`` (a few Perona–Malik diffusion iterations). They
  are display-only, applied to reconstructions before iso-extraction and labelled
  ``"display: …"`` in every title (the ``volume_transform`` display hint or keyword).
* **meshes** — :func:`iso_mesh`: ``skimage.measure.marching_cubes`` on the volume padded with an
  outside value (so surfaces are closed at the box), faces oriented outward, decimated with
  ``step_size`` to a face budget; ``None`` without scikit-image (callers fall back to voxels).
* **figures** — :func:`draw_isosurfaces` (one shaded ``Poly3DCollection`` per volume, one
  camera), :func:`draw_level_contours` (GT and reconstruction contours at several levels on the
  central slice and the depth cross-section) and :func:`iso_compare` (``voxels.png``).

The browser viewer (:mod:`nefi.viz.interactive`) runs the same rules, level algorithms and
transforms in JavaScript (``select``, ``matchedLevel``, ``otsuLevel``, ``transform`` in
``assets/volume_viewer.js`` mirror :func:`select_mask`, :func:`matched_level`,
:func:`otsu_level` and :func:`volume_transform` operation by operation).
"""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .fields import (
    _axis_names,
    _domain_extents,
    _recon_items,
    anomaly_centroid,
    axis_coords,
    draw_levels,
    image_extent,
    match_shape,
    resolve_field,
    show_image,
    squeeze_field,
    take_slice,
    voxel_threshold,
    voxel_view,
)
from .style import CmapSpec, color_limits, styled, theme

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

log = logging.getLogger("nefi")

SIDES = ("above", "below")
#: Level rules of a reconstruction (see the module docstring).
LEVEL_RULES = ("matched", "fixed", "otsu", "manual")
RULE_LABELS = {
    "matched": "volume-matched",
    "fixed": "GT threshold",
    "otsu": "Otsu",
    "manual": "manual",
}
#: Contour levels of :func:`draw_level_contours`: fractions of the GT contrast
#: (background → extreme).
CONTOUR_FRACTIONS = (0.25, 0.5, 0.75)


# ---------------------------------------------------------------------------------------------
# threshold rules
# ---------------------------------------------------------------------------------------------
def _finite(a: Any) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    return arr[np.isfinite(arr)]


def _auto_side(reference: Any) -> str:
    if reference is None:
        return "above"
    return voxel_threshold(np.asarray(reference, dtype=float), "auto")[0]


def auto_rule(reference: Any) -> dict[str, Any]:
    """Half-way from the reference's background (median) to its extreme on the anomaly side —
    the rule of the static voxel views (:func:`~nefi.viz.fields.voxel_threshold`)."""
    side, t = voxel_threshold(np.asarray(reference, dtype=float), "auto")
    return {
        "mode": "absolute",
        "side": side,
        "value": float(t),
        "source": "half-way from the GT background (median) to its extreme",
    }


def normalize_rule(threshold: Any, reference: Any = None) -> dict[str, Any]:
    """A threshold rule from ``None`` (automatic, :func:`auto_rule`), a number (side of the
    reference's larger deviation), ``(side, value)`` or a mapping.

    Rules are dicts: ``{"mode": "absolute", "side": "above" | "below", "value": t}`` (``v > t`` /
    ``v < t``) or ``{"mode": "relative", "side": ..., "fraction": f, "background": b}`` (each
    volume relative to its own extreme: ``v − b > f (peak − b)`` above, ``b − v > f (b − trough)``
    below — the half-maximum inclusion masks of dot3d); ``"source"`` describes the rule.
    """
    if threshold is None:
        if reference is None:
            raise ValueError("an automatic threshold needs a reference volume")
        return auto_rule(reference)
    if isinstance(threshold, Mapping):
        r = dict(threshold)
        side = str(r.get("side") or "")
        if r.get("mode") == "relative" or r.get("fraction") is not None:
            side = side if side in SIDES else "above"
            bg = r.get("background")
            if bg is None:
                fin = _finite(reference) if reference is not None else np.zeros(1)
                bg = float(np.median(fin)) if fin.size else 0.0
            f = float(r["fraction"])
            what = "excess" if side == "above" else "deficit"
            return {
                "mode": "relative",
                "side": side,
                "fraction": f,
                "background": float(bg),
                "source": str(
                    r.get("source") or f"{what} over {float(bg):.3g} beyond {f:.0%} of its peak"
                ),
            }
        if r.get("value") is None:
            raise ValueError(f"threshold rule needs a 'value' or a 'fraction': {threshold!r}")
        value = float(r["value"])
        return {
            "mode": "absolute",
            "side": side if side in SIDES else _auto_side(reference),
            "value": value,
            "source": str(r.get("source") or "given"),
        }
    if (
        isinstance(threshold, tuple | list)
        and len(threshold) == 2
        and isinstance(threshold[0], str)
    ):
        return normalize_rule({"side": threshold[0], "value": threshold[1]}, reference)
    return {
        "mode": "absolute",
        "side": _auto_side(reference),
        "value": float(threshold),
        "source": "given",
    }


def level_rule(value: float, side: str, source: str = "") -> dict[str, Any]:
    """An absolute rule ``v > value`` (``side="above"``) or ``v < value``."""
    return {"mode": "absolute", "side": side, "value": float(value), "source": source}


def rule_level(
    rule: Mapping[str, Any],
    values: Any = None,
    *,
    peak: float | None = None,
    trough: float | None = None,
) -> float:
    """The threshold of ``rule`` in data units for one volume (relative rules use the volume's
    own peak / trough: its max / min unless given)."""
    if rule.get("mode") != "relative":
        return float(rule["value"])
    bg, f = float(rule.get("background") or 0.0), float(rule["fraction"])
    if rule.get("side") == "below":
        tr = float(np.min(_finite(values))) if trough is None else float(trough)
        return bg - f * (bg - tr)
    pk = float(np.max(_finite(values))) if peak is None else float(peak)
    return bg + f * (pk - bg)


def select_mask(
    values: Any,
    rule: Mapping[str, Any],
    *,
    peak: float | None = None,
    trough: float | None = None,
) -> np.ndarray:
    """Voxels selected by a threshold rule (NaN never selected).

    The exact mirror of the viewer's ``select``: absolute ``v > t`` / ``v < t``; relative above
    ``(v − b) > f (peak − b)`` (nothing when ``peak ≤ b``), below ``(b − v) > f (b − trough)``.
    """
    v = np.asarray(values, dtype=float)
    side = rule.get("side", "above")
    with np.errstate(invalid="ignore"):
        if rule.get("mode") == "relative":
            bg, f = float(rule.get("background") or 0.0), float(rule["fraction"])
            fin = _finite(v)
            if side == "below":
                tr = (float(fin.min()) if fin.size else bg) if trough is None else float(trough)
                d = bg - tr
                return (bg - v) > f * d if d > 0 else np.zeros(v.shape, dtype=bool)
            pk = (float(fin.max()) if fin.size else bg) if peak is None else float(peak)
            d = pk - bg
            return (v - bg) > f * d if d > 0 else np.zeros(v.shape, dtype=bool)
        t = float(rule["value"])
        return v < t if side == "below" else v > t


def _metric_rule(instance: Any, name: str, field: str | None) -> dict[str, Any] | None:
    """An absolute rule from an IoU metric ``partial(iou_below | iou_above, tau=…)``."""
    fn = getattr(instance, "metrics", None)
    if not callable(fn):
        return None
    try:
        metrics = fn()
    except Exception:  # pragma: no cover - exotic instances
        return None
    if not isinstance(metrics, Mapping):
        return None
    for key, metric in metrics.items():
        if "iou" not in str(key).lower():
            continue
        base = getattr(metric, "func", metric)
        kw = dict(getattr(metric, "keywords", None) or {})
        fname = str(getattr(base, "__name__", ""))
        side = "below" if fname.endswith("below") else "above" if fname.endswith("above") else None
        if side is None:
            continue
        tau = kw.get("tau")
        if tau is None:
            try:
                tau = inspect.signature(base).parameters["tau"].default
            except (KeyError, TypeError, ValueError):
                tau = None
        if tau is None or tau is inspect.Parameter.empty:
            continue
        sym = "<" if side == "below" else ">"
        return {
            "mode": "absolute",
            "side": side,
            "value": float(tau),
            "source": f"IoU rule of {name}: {field or 'value'} {sym} {float(tau):g} ({fname})",
        }
    return None


def _fraction_rule(instance: Any, name: str) -> dict[str, Any] | None:
    """A relative rule from ``cfg.iou_fraction`` and one ``*background`` config value (the
    half-maximum inclusion masks of dot3d: ``[μ − μ_bg]_+ > f · max``)."""
    cfg = getattr(instance, "cfg", None)
    f = getattr(cfg, "iou_fraction", None)
    if cfg is None or f is None:
        return None
    bgs = []
    for k in dir(cfg):
        if k.startswith("_") or not k.endswith("background"):
            continue
        v = getattr(cfg, k, None)
        if isinstance(v, int | float) and not isinstance(v, bool):
            bgs.append((k, float(v)))
    if len(bgs) != 1:
        return None
    key, bg = bgs[0]
    return {
        "mode": "relative",
        "side": "above",
        "fraction": float(f),
        "background": bg,
        "source": f"IoU rule of {name}: excess over {key} = {bg:g} beyond {float(f):.0%} "
        "of each volume's peak excess",
    }


def voxel_rule(
    instance: Any = None,
    reference: Any = None,
    *,
    hints: Mapping[str, Any] | None = None,
    field: str | None = None,
) -> dict[str, Any]:
    """The GT threshold rule of voxel / isosurface views: the instance's IoU rule.

    Resolution order: the ``voxel_threshold`` display hint (a number, ``(side, value)`` or a rule
    mapping, :func:`normalize_rule`) → an IoU metric of ``instance.metrics()`` that is a
    ``functools.partial`` of a function named ``*_below`` / ``*_above`` with a ``tau``
    (thermal tomography: ``iou_below`` → ``α < 0.03``; ct3d: ``iou_above`` → ``μ > 0.25``) → a
    half-maximum rule from ``cfg.iou_fraction`` and one ``*background`` config value (dot3d:
    excess over ``mua_background`` beyond 50 % of each volume's own peak excess) → half-way from
    the reference's background to its extreme (:func:`auto_rule`).
    """
    from .hints import instance_hints

    hn = instance_hints(instance, hints) if (instance is not None or hints) else {}
    v = hn.get("voxel_threshold")
    if v is not None:
        r = normalize_rule(v, reference)
        if r.get("source") in (None, "", "given"):
            r["source"] = "voxel_threshold display hint"
        return r
    name = ""
    if instance is not None:
        name = str(getattr(instance, "name", "") or type(instance).__name__)
        r = _metric_rule(instance, name, field) or _fraction_rule(instance, name)
        if r is not None:
            return r
    if reference is None:
        raise ValueError("voxel_rule needs a reference volume when the instance has no IoU rule")
    return auto_rule(reference)


# ---------------------------------------------------------------------------------------------
# levels
# ---------------------------------------------------------------------------------------------
def iou_dice(a: Any, b: Any) -> tuple[float, float]:
    """``(IoU, Dice)`` of two boolean masks (NaN when both are empty)."""
    x, y = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    inter = int(np.count_nonzero(x & y))
    union = int(np.count_nonzero(x | y))
    tot = int(np.count_nonzero(x)) + int(np.count_nonzero(y))
    return (inter / union if union else float("nan"), 2.0 * inter / tot if tot else float("nan"))


def matched_level(values: Any, side: str, count: float) -> float:
    """The volume-matched level: ``count`` voxels of ``values`` lie past it on ``side``.

    Half-way between the ``count``-th and the next value in sorted order, so ``v > t``
    (``side="above"``) or ``v < t`` selects exactly ``count`` voxels when there are no ties.
    """
    a = np.sort(_finite(values))
    m = int(a.size)
    if m == 0:
        return float("nan")
    k = int(min(max(round(float(count)), 0), m))
    if side == "below":
        if k <= 0:
            return float(a[0])
        if k >= m:
            return float(a[m - 1] + max(abs(float(a[m - 1])), 1.0) * 1e-9)
        return float(a[k - 1] + (a[k] - a[k - 1]) / 2)
    if k <= 0:
        return float(a[m - 1])
    if k >= m:
        return float(a[0] - max(abs(float(a[0])), 1.0) * 1e-9)
    return float(a[m - k - 1] + (a[m - k] - a[m - k - 1]) / 2)


def otsu_level(values: Any, bins: int = 256) -> float:
    """Otsu's threshold: the histogram split (``bins`` bins over [min, max]) maximizing the
    between-class variance — the middle of the maximizing splits when an empty gap between the
    classes makes several of them tie."""
    a = _finite(values)
    if a.size == 0:
        return float("nan")
    lo, hi = float(a.min()), float(a.max())
    if not hi > lo:
        return lo
    idx = np.minimum(bins - 1, np.floor((a - lo) / (hi - lo) * bins).astype(np.int64))
    hist = [float(c) for c in np.bincount(idx, minlength=bins)]
    total = float(a.size)
    sum_all = 0.0
    for i in range(bins):
        sum_all += i * hist[i]
    w0 = s0 = 0.0
    best, kb, kl = -1.0, 0, 0
    for k in range(bins - 1):  # plain loops: the viewer's otsuLevel does the same operations
        w0 += hist[k]
        s0 += k * hist[k]
        w1 = total - w0
        if w0 == 0.0 or w1 == 0.0:
            continue
        d = s0 / w0 - (sum_all - s0) / w1
        var = w0 * w1 * d * d
        if var > best:
            best, kb, kl = var, k, k
        elif var == best:  # empty bins between the classes tie exactly: keep the plateau
            kl = k
    return lo + ((kb + kl) / 2 + 1) * (hi - lo) / bins


def recon_level(
    values: Any, rule: Mapping[str, Any], how: str, *, gt_count: int, manual: float | None = None
) -> float:
    """The level of a reconstruction under a level rule (:data:`LEVEL_RULES`)."""
    if how == "fixed":
        return rule_level(rule, values)
    if how == "matched":
        return matched_level(values, rule["side"], gt_count)
    if how == "otsu":
        return otsu_level(values)
    if how == "manual":
        if manual is None:
            raise ValueError("the manual level rule needs a value")
        return float(manual)
    raise ValueError(f"unknown level rule {how!r}; use one of {LEVEL_RULES}")


def iso_levels(
    gt: Any, recon: Any, rule: Any = None, *, manual: float | None = None
) -> dict[str, Any]:
    """The GT level and the reconstruction's fixed / volume-matched / Otsu (/ manual) levels,
    with the voxel count, IoU and Dice of each reconstruction set against the GT set.

    Returns ``{"rule", "side", "gt_level", "gt_count", "levels": {rule: t}, "counts",
    "iou", "dice"}`` (``fixed`` applies the GT rule itself, i.e. relative rules use the
    reconstruction's own peak).
    """
    g = np.asarray(gt, dtype=float)
    r = np.asarray(recon, dtype=float)
    rule = normalize_rule(rule, g)
    side = rule["side"]
    sg = select_mask(g, rule)
    n_g = int(np.count_nonzero(sg))
    out: dict[str, Any] = {
        "rule": rule,
        "side": side,
        "gt_level": rule_level(rule, g),
        "gt_count": n_g,
        "levels": {},
        "counts": {},
        "iou": {},
        "dice": {},
    }
    hows = ["fixed", "matched", "otsu"] + (["manual"] if manual is not None else [])
    for how in hows:
        t = recon_level(r, rule, how, gt_count=n_g, manual=manual)
        s = select_mask(r, rule) if how == "fixed" else select_mask(r, level_rule(t, side))
        out["levels"][how] = float(t)
        out["counts"][how] = int(np.count_nonzero(s))
        out["iou"][how], out["dice"][how] = iou_dice(sg, s)
    return out


# ---------------------------------------------------------------------------------------------
# display transforms
# ---------------------------------------------------------------------------------------------
#: Display transforms of volumes and their default parameters (σ in voxels).
VOLUME_TRANSFORMS: dict[str, dict[str, float]] = {
    "smooth": {"sigma": 1.0},
    "sharpen": {"amount": 1.0, "sigma": 1.0},
    "edge_preserve": {"iterations": 5, "kappa": 0.1, "step": 0.15},
}
_TRANSFORM_ALIASES = {
    "gaussian": "smooth",
    "blur": "smooth",
    "unsharp": "sharpen",
    "unsharp_mask": "sharpen",
    "tv": "edge_preserve",
    "bilateral": "edge_preserve",
    "anisotropic": "edge_preserve",
    "perona_malik": "edge_preserve",
    "edge-preserve": "edge_preserve",
}


def transform_spec(spec: Any) -> dict[str, Any] | None:
    """Normalize a volume display transform: ``None`` / ``"none"`` → ``None``; a name of
    :data:`VOLUME_TRANSFORMS` (aliases: ``gaussian``, ``unsharp``, ``tv``, ``bilateral``, …); a
    mapping ``{"name": ..., <parameters>}``; or ``(name, {parameters})``."""
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        d = dict(spec)
        name = str(d.pop("name", "") or "")
    elif isinstance(spec, str):
        name, d = spec, {}
    elif isinstance(spec, tuple | list) and len(spec) == 2:
        name, d = str(spec[0]), dict(spec[1] or {})
    else:
        raise ValueError(f"cannot interpret volume transform {spec!r}")
    key = name.strip().lower()
    key = _TRANSFORM_ALIASES.get(key, key)
    if key in ("", "none", "identity", "raw"):
        return None
    if key not in VOLUME_TRANSFORMS:
        raise ValueError(
            f"unknown volume transform {name!r}; use one of {sorted(VOLUME_TRANSFORMS)} or 'none'"
        )
    out: dict[str, Any] = {"name": key, **VOLUME_TRANSFORMS[key]}
    for k, v in d.items():
        if k not in out:
            raise ValueError(f"unknown parameter {k!r} of volume transform {key!r}")
        out[k] = int(v) if k == "iterations" else float(v)
    return out


def transform_label(spec: Any) -> str:
    """``"display: smooth σ=1"`` (empty for no transform) — used in every title."""
    s = transform_spec(spec)
    if s is None:
        return ""
    if s["name"] == "smooth":
        return f"display: smooth σ={s['sigma']:g}"
    if s["name"] == "sharpen":
        return f"display: sharpen ×{s['amount']:g} σ={s['sigma']:g}"
    return f"display: edge-preserving ×{int(s['iterations'])}"


def hinted_transform(hints: Mapping[str, Any] | None, field: str | None = None) -> Any:
    """The ``volume_transform`` display hint for ``field`` (a spec, or ``{field: spec}`` with
    ``"*"`` as default), normalized by :func:`transform_spec`."""
    v = (hints or {}).get("volume_transform")
    if isinstance(v, Mapping) and "name" not in v:
        v = v.get(field) if (field is not None and field in v) else v.get("*")
    return transform_spec(v)


def _gauss_axis(a: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    r = max(1, math.ceil(3.0 * sigma))
    w = np.exp(-(np.arange(-r, r + 1, dtype=float) ** 2) / (2.0 * sigma * sigma))
    w = w / w.sum()
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode="edge")
    out = np.zeros_like(a)
    for i in range(2 * r + 1):
        out = out + w[i] * np.take(p, np.arange(i, i + n), axis=axis)
    return out


def gaussian_smooth(a: Any, sigma: float) -> np.ndarray:
    """Separable Gaussian smoothing (radius ⌈3σ⌉, replicated edges), σ in voxels."""
    out = np.asarray(a, dtype=float)
    if sigma <= 0:
        return out.copy()
    for ax in range(out.ndim):
        out = _gauss_axis(out, float(sigma), ax)
    return out


def _perona_malik(x: np.ndarray, iterations: int, kappa: float, step: float) -> np.ndarray:
    """Perona–Malik diffusion ``x ← x + step Σ_nbr Δ / (1 + (Δ/κ)²)``, κ = ``kappa`` × range,
    zero-flux (replicated) borders: smooths inside regions, keeps edges steeper than κ."""
    y = x.copy()
    k = float(kappa) * float(y.max() - y.min())
    if not k > 0:
        return y
    for _ in range(max(0, int(iterations))):
        p = np.pad(y, 1, mode="edge")
        upd = np.zeros_like(y)
        for d in range(y.ndim):
            for s in (0, 2):
                sl = [slice(1, -1)] * y.ndim
                sl[d] = slice(s, s + y.shape[d])
                delta = p[tuple(sl)] - y
                upd = upd + delta / (1.0 + (delta / k) ** 2)
        y = y + float(step) * upd
    return y


def volume_transform(a: Any, spec: Any) -> np.ndarray:
    """Apply a display transform (:func:`transform_spec`) to a volume; returns a new array.

    ``smooth`` — Gaussian (σ voxels); ``sharpen`` — unsharp mask ``x + amount (x − G_σ x)``,
    clipped to the input range; ``edge_preserve`` — :func:`_perona_malik` diffusion. Display
    only: used before iso-extraction, never for metrics reported as the reconstruction's.
    """
    s = transform_spec(spec)
    x = np.asarray(a, dtype=float)
    if s is None:
        return x.copy()
    fin = np.isfinite(x)
    if not fin.all():
        x = np.where(fin, x, float(np.median(x[fin])) if fin.any() else 0.0)
    if s["name"] == "smooth":
        y = gaussian_smooth(x, s["sigma"])
    elif s["name"] == "sharpen":
        y = x + s["amount"] * (x - gaussian_smooth(x, s["sigma"]))
        y = np.clip(y, float(x.min()), float(x.max()))
    else:
        y = _perona_malik(x, int(s["iterations"]), s["kappa"], s["step"])
    return np.where(fin, y, np.nan)


# ---------------------------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------------------------
def has_marching_cubes() -> bool:
    """True when ``skimage.measure.marching_cubes`` is importable."""
    try:
        from skimage.measure import marching_cubes  # noqa: F401
    except ImportError:
        return False
    return True


def iso_mesh(
    values: Any,
    level: float,
    side: str = "above",
    *,
    max_faces: int = 20000,
    step_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Marching-cubes isosurface enclosing the voxels past ``level`` on ``side``.

    The volume is padded with an outside value so every surface is closed at the box (the caps
    lie on the box faces), vertices are in grid-index coordinates (voxel centres at integers,
    clipped to ``[-0.5, n - 0.5]``), faces are oriented outward (right-hand rule, positive
    enclosed volume) and ``step_size`` grows until the mesh has at most ``max_faces`` faces.

    Returns:
        ``(vertices (V, 3) float32, faces (F, 3) int64)`` — empty arrays when nothing is past
        the level — or ``None`` without scikit-image.
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        return None
    a = np.asarray(values, dtype=float)
    if a.ndim != 3:
        raise ValueError(f"iso_mesh expects a 3-D volume, got shape {a.shape}")
    empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64))
    fin = _finite(a)
    if fin.size == 0 or not math.isfinite(float(level)):
        return empty
    level = float(level)
    inside = fin > level if side == "above" else fin < level
    if not inside.any():
        return empty
    span = max(float(fin.max() - fin.min()), abs(level), 1e-12)
    pad = level - span if side == "above" else level + span
    p = np.pad(np.where(np.isfinite(a), a, pad), 1, mode="constant", constant_values=pad)
    step = max(1, int(step_size or 1))
    while True:
        try:
            verts, faces, _, _ = marching_cubes(
                p, level=level, step_size=step, allow_degenerate=False
            )
        except (ValueError, RuntimeError) as e:  # level outside the data range, degenerate
            log.debug("marching cubes failed at level %g: %s", level, e)
            return empty
        if len(faces) <= max_faces or step_size is not None or step >= 8:
            break
        step += 1
    verts = np.clip(verts - 1.0, -0.5, np.asarray(a.shape, dtype=float) - 0.5)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces):
        t = verts[faces]
        signed = float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum())
        if signed < 0:
            faces = faces[:, [0, 2, 1]]
    return verts.astype(np.float32), faces


# ---------------------------------------------------------------------------------------------
# static figures
# ---------------------------------------------------------------------------------------------
def surface_colors(n: int, dark: bool | None = None) -> list[str]:
    """Isosurface colours: the ground truth neutral, reconstructions the categorical slots."""
    t = theme(dark)
    gt = "#bdbcb2" if t.name == "dark" else "#a9a79d"
    return [gt] + [t.palette[i % len(t.palette)] for i in range(max(0, n - 1))]


def depth_exaggeration(shape: Sequence[int], domain: Any = None, factor: float = 0.6) -> float:
    """Stretch of the depth axis of 3-D panels: ``max(1, factor · lateral / depth)`` (thin slabs
    stay readable; the axis label says ``×k``)."""
    coords = [axis_coords(int(shape[d]), domain, d) for d in range(3)]
    spans = [float(c[-1] - c[0]) if len(c) > 1 else 1.0 for c in coords]
    spans = [s if s > 0 else 1.0 for s in spans]
    return max(1.0, factor * max(spans[:2]) / spans[2])


def _box_axes(
    ax: Any,
    shape: Sequence[int],
    domain: Any,
    zx: float,
    elev: float,
    azim: float,
    zoom: float,
) -> tuple[list[tuple[float, float]], list[float]]:
    """Limits (cell faces), box aspect, inverted depth, labels and camera of a 3-D panel."""
    from matplotlib.ticker import MaxNLocator

    t = theme()
    names = _axis_names(domain, 3)
    ext = _domain_extents(domain)
    lims = [(ext[d][0], ext[d][1]) if ext is not None else (-0.5, shape[d] - 0.5) for d in range(3)]
    spans = [hi - lo for lo, hi in lims]
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.set_zlim(*lims[2])
    ax.set_box_aspect((spans[0], spans[1], spans[2] * zx), zoom=zoom)
    if ext is None or names[2] == "z":
        ax.invert_zaxis()  # depth increases downwards
    ax.set_xlabel(names[0], labelpad=-8)
    ax.set_ylabel(names[1], labelpad=-8)
    ax.set_zlabel(f"{names[2]} (×{zx:.2g})" if zx > 1.01 else names[2], labelpad=-8)
    ax.tick_params(labelsize="x-small", pad=-3)
    for axis_ in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis_.set_major_locator(MaxNLocator(4))
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(t.surface)
        pane.set_edgecolor(t.grid)
    ax.grid(False)
    ax.view_init(elev=elev, azim=azim)
    return lims, spans


def draw_isosurfaces(
    axes: Sequence[Axes],
    entries: Sequence[tuple[str, np.ndarray, float, str]],
    *,
    domain: Any = None,
    colors: Sequence[str] | None = None,
    elev: float = 24.0,
    azim: float = -58.0,
    max_faces: int = 6000,
    zoom: float = 1.0,
    z_exaggeration: float | None = None,
) -> list[int]:
    """Shaded isosurfaces, one per ``projection="3d"`` axes, with one camera and one box.

    Args:
        axes: 3-D axes.
        entries: ``(title, volume, level, side)`` per panel.
        domain: physical coordinates.
        colors: surface colours (default :func:`surface_colors`: GT neutral, then the
            categorical slots).
        elev, azim: camera.
        max_faces: face budget per mesh (``step_size`` decimation).
        zoom: box zoom.
        z_exaggeration: depth stretch (default :func:`depth_exaggeration`).

    Returns:
        Face count per panel (``-1`` where the voxel fallback was drawn — no scikit-image).
    """
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if not entries:
        return []
    shape = entries[0][1].shape
    zx = z_exaggeration or depth_exaggeration(shape, domain)
    cols = list(colors) if colors is not None else surface_colors(len(entries))
    ext = _domain_extents(domain)
    lo = np.array([ext[d][0] if ext else 0.0 for d in range(3)])
    sp = np.array([(ext[d][1] - ext[d][0]) / shape[d] if ext else 1.0 for d in range(3)])
    off = 0.5 if ext else 0.0  # physical: cell centres at lo + (i + 1/2) h; indices otherwise
    e, a = math.radians(elev + 20.0), math.radians(azim - 25.0)
    light = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    counts: list[int] = []
    for i, (ax, (title, vol, level, side)) in enumerate(zip(axes, entries)):
        mesh = iso_mesh(vol, level, side, max_faces=max_faces)
        if mesh is None:  # no scikit-image: thresholded voxels instead
            voxel_view(
                vol,
                threshold=level,
                mode=side,
                domain=domain,
                ax=ax,
                title=title,
                elev=elev,
                azim=azim,
                z_exaggeration=zx,
                colorbar=False,
                zoom=zoom,
            )
            ax.title.set_fontsize("small")
            counts.append(-1)
            continue
        verts, faces = mesh
        _box_axes(ax, shape, domain, zx, elev, azim, zoom)
        if len(faces):
            pts = lo + (verts.astype(float) + off) * sp
            tri = pts[faces]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nb = n * np.array([1.0, 1.0, -1.0 / zx])  # data → box frame (stretched, inverted z)
            nb /= np.maximum(np.linalg.norm(nb, axis=1, keepdims=True), 1e-30)
            # two-sided Lambert, headlight from the upper left (no `@`: spurious BLAS warnings)
            c = np.abs(nb[:, 0] * light[0] + nb[:, 1] * light[1] + nb[:, 2] * light[2])
            shade = 0.3 + 0.7 * c
            rgb = np.asarray(mcolors.to_rgb(cols[i % len(cols)]))
            fc = np.clip(rgb[None, :] * shade[:, None] + 0.18 * (c[:, None] ** 12), 0.0, 1.0)
            coll = Poly3DCollection(tri, facecolors=fc, edgecolors=fc, linewidths=0.12)
            ax.add_collection3d(coll)
        else:
            ax.text2D(
                0.5, 0.5, "empty level set", transform=ax.transAxes, ha="center", fontsize="small"
            )
        ax.set_title(title, fontsize="small")
        counts.append(len(faces))
    return counts


def contour_levels(gt: Any, fractions: Sequence[float] = CONTOUR_FRACTIONS) -> list[float]:
    """Levels at ``fractions`` of the GT contrast, from its background (median) toward its
    extreme on the anomaly side."""
    g = _finite(gt)
    if g.size == 0:
        return []
    bg = float(np.median(g))
    side = _auto_side(g)
    ext = float(g.min()) if side == "below" else float(g.max())
    return [bg + float(f) * (ext - bg) for f in fractions]


def draw_level_contours(
    axes: Sequence[Axes],
    gt: np.ndarray,
    recon: np.ndarray,
    *,
    center: Sequence[int] | None = None,
    axis: int = -1,
    levels: Sequence[float] | None = None,
    fractions: Sequence[float] = CONTOUR_FRACTIONS,
    domain: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
    titles: bool = True,
    legend: bool = True,
) -> list[float]:
    """Multi-level contours of GT (solid) and reconstruction (dashed) on the central slices.

    Panel 0: the slice along the volume axis through ``center`` (x–y at the anomaly's depth);
    panel 1: the depth cross-section (x–z through ``center``, depth down). The background is the
    reconstruction's slice in grey; one colour per level (``fractions`` of the GT contrast,
    :func:`contour_levels`): soft reconstructed edges show as spread-out dashed contours where
    the GT's solid ones bunch together. Returns the levels.
    """
    from matplotlib.lines import Line2D

    from .volume import section, section_extent

    t = theme()
    ax_v = axis % 3
    g = np.asarray(gt, dtype=float)
    r = np.asarray(recon, dtype=float)
    center = tuple(center) if center is not None else anomaly_centroid(g)
    lv = list(levels) if levels is not None else contour_levels(g, fractions)
    labels = [f"{f:.0%}" for f in fractions] if levels is None else [f"{v:.3g}" for v in lv]
    cols = [t.palette[(3 + i) % len(t.palette)] for i in range(len(lv))]
    lo, hi = color_limits([g, r], None)
    vmin = lo if vmin is None else vmin
    vmax = hi if vmax is None else vmax
    grey = CmapSpec("gray", "sequential", "generic")
    names = _axis_names(domain, 3)
    other = [d for d in range(3) if d != ax_v]
    # panel 0: slice along the volume axis
    k = int(center[ax_v])
    ext0 = image_extent(take_slice(r, k, ax_v).shape, domain, (other[0], other[1]))
    show_image(axes[0], take_slice(r, k, ax_v), spec=grey, vmin=vmin, vmax=vmax, extent=ext0)
    for c, v in zip(cols, lv):
        draw_levels(axes[0], take_slice(g, k, ax_v), [v], ext0, color=c, linewidth=1.0)
        draw_levels(
            axes[0], take_slice(r, k, ax_v), [v], ext0, style="recon", color=c, linewidth=0.9
        )
    # panel 1: depth cross-section through the centre
    if len(axes) > 1:
        sl, lateral, fixed = section(r, center, 0, ax_v)
        ext1 = section_extent(sl.shape, domain, lateral, ax_v)
        show_image(
            axes[1],
            sl,
            spec=grey,
            vmin=vmin,
            vmax=vmax,
            extent=ext1,
            origin="upper",
            aspect="auto",
        )
        axes[1].set_box_aspect(1.0)
        gs = section(g, center, 0, ax_v)[0]
        for c, v in zip(cols, lv):
            draw_levels(axes[1], gs, [v], ext1, origin="upper", color=c, linewidth=1.0)
            draw_levels(
                axes[1], sl, [v], ext1, style="recon", origin="upper", color=c, linewidth=0.9
            )
        if titles:
            fv = axis_coords(g.shape[fixed], domain, fixed)[int(center[fixed])]
            axes[1].set_title(
                f"{names[lateral]}–{names[ax_v]} ↓ at {names[fixed]} = {fv:.3g}", fontsize="small"
            )
    if titles:
        zv = axis_coords(g.shape[ax_v], domain, ax_v)[k]
        axes[0].set_title(
            f"{names[other[0]]}–{names[other[1]]} at {names[ax_v]} = {zv:.3g}", fontsize="small"
        )
    if legend:
        handles = [Line2D([], [], color=t.ink, lw=1.0, label="GT")]
        handles.append(Line2D([], [], color=t.ink, lw=0.9, ls=(0, (3.0, 1.6)), label="recon"))
        handles += [Line2D([], [], color=c, lw=1.4, label=lab) for c, lab in zip(cols, labels)]
        axes[-1].legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.04),
            ncols=min(len(handles), 3),
            fontsize="xx-small",
            frameon=False,
            handlelength=1.6,
            columnspacing=0.8,
            title="contrast levels" if levels is None else "levels",
            title_fontsize="xx-small",
        )
    return lv


def _sym(side: str) -> str:
    return "<" if side == "below" else ">"


def iso_titles(
    label: str, levels: Mapping[str, Any], how: str = "matched", name: str | None = None
) -> tuple[str, str]:
    """Two-line panel titles of a GT / reconstruction isosurface pair: level and rule, voxels or
    IoU / Dice (with the IoU at the GT threshold for comparison)."""
    q = name or "value"
    s = _sym(levels["side"])
    gt = f"ground truth: {q} {s} {levels['gt_level']:.3g}\n{levels['gt_count']} voxels"
    iou, dice = levels["iou"].get(how, float("nan")), levels["dice"].get(how, float("nan"))
    txt = f"{label}: {q} {s} {levels['levels'][how]:.3g} ({RULE_LABELS.get(how, how)})"
    txt += f"\nIoU {iou:.2f} · Dice {dice:.2f}"
    if how != "fixed" and "fixed" in levels["iou"]:
        txt += f" (at GT threshold: IoU {levels['iou']['fixed']:.2f})"
    return gt, txt


def iso_compare(
    gt: Any,
    recons: Any = None,
    *,
    field: str | None = None,
    rule: Any = None,
    how: str = "matched",
    instance: Any = None,
    domain: Any = None,
    transform: Any = None,
    mask: Any = None,
    axis: int = -1,
    hints: Mapping[str, Any] | None = None,
    title: str | None = None,
    elev: float = 24.0,
    azim: float = -58.0,
    max_faces: int = 6000,
    dark: bool | None = None,
) -> Figure:
    """GT and reconstruction(s) as shaded isosurfaces with matched levels, plus contours.

    The GT is cut by its threshold rule (:func:`voxel_rule`: the instance's IoU rule unless
    ``rule`` is given), every reconstruction at its ``how`` level (default ``"matched"``: the
    level enclosing as many voxels as the GT; ``"fixed"``, ``"otsu"``). Left: multi-level
    contours of GT and (first) reconstruction on the central slice and the depth
    cross-section (:func:`draw_level_contours`); right: the isosurfaces, one camera. Titles give
    the levels, the rule and IoU / Dice (and the IoU at the GT threshold for comparison).

    Args:
        gt: 3-D ground truth (tensor / array / fields dict).
        recons: a reconstruction or ``{method: reconstruction}``.
        field: field name.
        rule: GT threshold rule (:func:`normalize_rule`); default :func:`voxel_rule`.
        how: level rule of the reconstructions (:data:`LEVEL_RULES` minus ``"manual"``).
        instance, hints: display hints (``field_transform``, ``volume_transform``,
            ``voxel_threshold``) and the instance's IoU rule.
        domain: physical coordinates.
        transform: display transform of the reconstructions (default: the
            ``volume_transform`` hint); titles say ``"display: …"``.
        mask: GT mask (anomaly centroid of the contour slices).
        axis: volume axis.
        title: figure title.
        elev, azim: camera.
        max_faces: face budget per mesh.
        dark: dark theme.
    """
    import mpl_toolkits.mplot3d  # noqa: F401  (registers the 3d projection)
    from matplotlib.figure import Figure

    from .hints import apply_transform, field_hint, instance_hints, transform_name

    g, name = resolve_field(gt, field)
    g = squeeze_field(g)
    g = np.abs(g) if np.iscomplexobj(g) else g
    if g.ndim != 3:
        raise ValueError(f"iso_compare expects a 3-D field, got shape {g.shape}")
    items = []
    for lab, a in _recon_items(recons, field or name):
        a = squeeze_field(a)
        a = np.abs(a) if np.iscomplexobj(a) else a
        items.append((lab, match_shape(a, g.shape) if a.shape != g.shape else a))
    hn = instance_hints(instance, hints)
    tf = field_hint(hn, "field_transform", name)
    if transform_name(tf) is not None:
        g = apply_transform(g, tf, domain=domain, instance=instance)
        items = [
            (lab, apply_transform(a, tf, domain=domain, instance=instance)) for lab, a in items
        ]
    rule = voxel_rule(instance, g, hints=hints, field=name) if rule is None else rule
    rule = normalize_rule(rule, g)
    vt = transform_spec(transform) if transform is not None else hinted_transform(hn, name)
    note = transform_label(vt)
    disp = [(lab, volume_transform(a, vt) if vt else a) for lab, a in items]
    with styled(dark):
        n3 = 1 + len(disp)
        fig = Figure(figsize=(round(1.9 + 3.05 * n3, 2), 3.6), layout="constrained")
        gs = fig.add_gridspec(2, 1 + n3, width_ratios=[0.62] + [1.0] * n3)
        cax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
        ax3 = [fig.add_subplot(gs[:, 1 + i], projection="3d") for i in range(n3)]
        center = anomaly_centroid(g, mask=mask)
        if disp:
            draw_level_contours(cax, g, disp[0][1], center=center, axis=axis, domain=domain)
        else:
            for ax in cax:
                ax.set_visible(False)
        entries = [("ground truth", g, rule_level(rule, g), rule["side"])]
        heads = []
        for lab, a in disp:
            lv = iso_levels(g, a, rule)
            gt_t, rec_t = iso_titles(lab + (f" ({note})" if note else ""), lv, how, name)
            heads.append(gt_t)
            entries.append((rec_t, a, lv["levels"][how], rule["side"]))
        if heads:
            entries[0] = (heads[0], *entries[0][1:])
        draw_isosurfaces(
            ax3, entries, domain=domain, elev=elev, azim=azim, max_faces=max_faces, zoom=1.0
        )
        head = title or (
            f"{name or 'field'}: isosurfaces — GT at its threshold rule, reconstruction at the "
            f"{RULE_LABELS.get(how, how)} level"
        )
        fig.suptitle(head + (f" ({note})" if note and note not in head else ""))
        fig.text(
            0.005,
            0.005,
            f"GT rule: {rule.get('source', '')}",
            fontsize="xx-small",
            color=theme().muted,
            ha="left",
            va="bottom",
        )
        return fig


__all__ = [
    "CONTOUR_FRACTIONS",
    "LEVEL_RULES",
    "RULE_LABELS",
    "SIDES",
    "VOLUME_TRANSFORMS",
    "auto_rule",
    "contour_levels",
    "depth_exaggeration",
    "draw_isosurfaces",
    "draw_level_contours",
    "gaussian_smooth",
    "has_marching_cubes",
    "hinted_transform",
    "iou_dice",
    "iso_compare",
    "iso_levels",
    "iso_mesh",
    "iso_titles",
    "level_rule",
    "matched_level",
    "normalize_rule",
    "otsu_level",
    "recon_level",
    "rule_level",
    "select_mask",
    "surface_colors",
    "transform_label",
    "transform_spec",
    "volume_transform",
    "voxel_rule",
]
