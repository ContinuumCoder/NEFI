"""Interactive 3-D viewer: drag-to-rotate isosurfaces, voxels and slices in the browser.

A dependency-free viewer in vanilla JavaScript (``nefi/viz/assets/volume_viewer.js`` + ``.css``:
Canvas 2D, no libraries, no network access) shows one or two volumes side by side — ground truth
and reconstruction — with one camera:

* **controls** — drag (one finger) orbits, wheel / pinch zooms, right- or shift-drag pans,
  double-click resets (arrows, ``+`` / ``-``, ``0`` on a focused view); an axes triad and a depth
  cue (farther voxels smaller and faded toward the background) keep the orientation readable;
* **views** — ``"iso"`` (shaded isosurfaces: lit triangles, smooth or flat shading, back faces
  culled, optional semi-transparent GT overlay), ``"voxels"`` (thresholded voxel centres as
  squares in the colormap of the static figures), ``"slice"`` (a movable x / y / z slice drawn as
  a textured quad in the rotated frame), ``"iso+slice"`` and ``"voxels+slice"``;
* **levels** — the GT is cut by its threshold rule (the instance's IoU rule,
  :func:`~nefi.viz.isosurface.voxel_rule`), each reconstruction by a level rule: volume-matched
  (default: as many voxels as the GT), GT threshold, Otsu or manual; the readout gives both
  levels, the voxel counts and IoU / Dice (computed in the browser) and names the active rule;
* **display transforms** — raw, smooth, sharpen, edge-preserving (applied to reconstructions,
  re-extracted live, labelled "display" everywhere; :func:`~nefi.viz.isosurface.volume_transform`).

Data travel as compact JSON: volumes area-averaged to at most ``max_side`` voxels per axis
(default 64, so ≤ 64³) and quantized to 16-bit (up to 48³ voxels) or 8-bit integers with an
affine scale, plus the
default isosurfaces from ``skimage.measure.marching_cubes`` (vertices uint16, faces uint16 /
uint32, decimated to ``max_faces``), all base64; shape, physical extent and axis names make the
aspect ratio physical. Codes are nudged so the GT rule selects exactly the voxels it selects in
the full-precision arrays (the in-browser IoU at the GT threshold equals the instance's IoU when
nothing was downsampled). Changing a level or transform re-extracts isosurfaces in the browser
(marching tetrahedra).

* :func:`volume_viewer_html` — an HTML fragment (``<div>`` + JSON data + inline script, unique
  element ids so several viewers coexist on one page; ``include_assets=False`` for all but the
  first) or, with ``standalone=True``, a whole document;
* :func:`save_volume_viewer` — a standalone ``.html`` file; :func:`viewer_size_bytes` — the
  embedded size;
* :func:`viewer_payload` / :func:`viewer_fragment` — the two steps separately;
* :func:`plotly_volume_html` — optional plotly isosurfaces (``kind="plotly"``). It loads
  plotly.js from the CDN (``include_plotlyjs="cdn"``) and is therefore **not** self-contained;
  never the default.
"""

from __future__ import annotations

import base64
import functools
import html
import json
import logging
import math
import os
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .fields import (
    _axis_names,
    _domain_extents,
    anomaly_centroid,
    match_shape,
    resolve_field,
    squeeze_field,
)
from .isosurface import (
    LEVEL_RULES,
    RULE_LABELS,
    auto_rule,
    has_marching_cubes,
    hinted_transform,
    iou_dice,
    iso_mesh,
    level_rule,
    matched_level,
    normalize_rule,
    otsu_level,
    rule_level,
    select_mask,
    transform_label,
    transform_spec,
    volume_transform,
    voxel_rule,
)
from .style import CmapSpec, cmap_for, color_limits, spec_from_hint

log = logging.getLogger("nefi")

#: Largest side of an embedded volume (area-averaged beyond it): ≤ 64³ voxels per volume.
MAX_SIDE = 64
#: Face budget of an embedded isosurface (marching-cubes ``step_size`` decimation).
MAX_FACES = 20000
#: ``bits="auto"``: 16-bit codes up to this many voxels per volume (48³), 8-bit beyond.
AUTO_16BIT_VOXELS = 48**3
#: View modes of the viewer.
MODES = ("iso", "voxels", "slice", "iso+slice", "voxels+slice")
_MODE_ALIASES = {
    "isosurface": "iso",
    "isosurfaces": "iso",
    "surface": "iso",
    "mesh": "iso",
    "voxel": "voxels",
    "points": "voxels",
    "slices": "slice",
    "both": "voxels+slice",
    "points+slice": "voxels+slice",
    "voxels + slice": "voxels+slice",
    "slice+voxels": "voxels+slice",
    "isosurface+slice": "iso+slice",
    "slice+iso": "iso+slice",
}
#: Colormaps implemented in the viewer (others fall back by kind: RdBu_r, twilight, viridis).
VIEWER_CMAPS = ("viridis", "magma", "inferno", "plasma", "cividis", "RdBu_r", "twilight", "Greys")
#: Viewer implementations: ``"canvas"`` (self-contained default) and ``"plotly"`` (CDN).
KINDS = ("canvas", "plotly")


def normalize_kind(kind: Any) -> str | None:
    """``"canvas"`` / ``"plotly"`` / ``None`` (off) from ``True``, ``False``, ``None`` or a name
    (``"none"``, ``"off"``, ``"js"``, ``"vanilla"``, ...)."""
    if kind is None or kind is False:
        return None
    if kind is True:
        return "canvas"
    k = str(kind).strip().lower()
    if k in ("", "canvas", "js", "vanilla", "default", "on", "true", "1"):
        return "canvas"
    if k in ("none", "off", "false", "0", "no"):
        return None
    if k == "plotly":
        return "plotly"
    raise ValueError(f"unknown interactive viewer {kind!r}; use 'canvas', 'plotly' or 'none'")


def normalize_mode(mode: str) -> str:
    """One of :data:`MODES` (aliases: ``"isosurface"``, ``"points+slice"``, ``"both"``, ...)."""
    m = str(mode or "voxels").strip().lower()
    m = _MODE_ALIASES.get(m, m)
    if m not in MODES:
        raise ValueError(f"unknown viewer mode {mode!r}; use one of {MODES}")
    return m


# ---------------------------------------------------------------------------------------------
# assets
# ---------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def viewer_assets() -> tuple[str, str]:
    """``(javascript, css)`` of the viewer (package data ``nefi/viz/assets``)."""
    from importlib import resources

    base = resources.files("nefi.viz").joinpath("assets")
    js = base.joinpath("volume_viewer.js").read_text(encoding="utf-8")
    css = base.joinpath("volume_viewer.css").read_text(encoding="utf-8")
    return js, css


def assets_html() -> str:
    """The viewer's ``<style>`` and ``<script>`` — include once per page."""
    js, css = viewer_assets()
    js = js.replace("</script", "<\\/script")  # never close the inline script early
    return f"<style>{css}</style>\n<script>{js}</script>"


def asset_bytes() -> int:
    """Size of :func:`assets_html` in bytes (shared by every viewer of a page)."""
    return len(assets_html().encode("utf-8"))


# ---------------------------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------------------------
def _num(x: Any) -> float | None:
    v = float(x)
    return v if math.isfinite(v) else None


def _volume_items(volumes: Any, field: str | None) -> list[tuple[str, np.ndarray]]:
    if isinstance(volumes, Mapping) and not (
        hasattr(volumes, "fields") or hasattr(volumes, "noise_std")
    ):
        pairs = list(volumes.items())
    else:
        pairs = [("volume", volumes)]
    out = []
    for name, v in pairs:
        a = squeeze_field(resolve_field(v, field)[0])
        a = np.abs(a) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
        if a.ndim != 3:
            raise ValueError(f"volume {name!r} must be 3-D, got shape {a.shape}")
        out.append((str(name), a))
    if not out:
        raise ValueError("no volumes given")
    if len(out) > 4:
        raise ValueError("the viewer shows at most four volumes side by side")
    return out


def _extent_axes(
    extent: Any, shape: Sequence[int], axes: Sequence[str] | None
) -> tuple[list[list[float]], list[str]]:
    """Physical ``[[lo, hi]] * 3`` and axis names from a Domain, pairs, lengths or ``None``
    (voxel units)."""
    ext = _domain_extents(extent) if extent is not None else None
    if ext is None and extent is not None:
        try:
            vals = [float(v) for v in extent]
            if len(vals) == 3:
                ext = [(0.0, v) for v in vals]
        except (TypeError, ValueError):
            ext = None
    if ext is None or len(ext) != 3:
        ext = [(0.0, float(n)) for n in shape]
    names = list(axes) if axes else _axis_names(extent, 3)
    if len(names) != 3:
        names = ["x", "y", "z"]
    return [[float(lo), float(hi)] for lo, hi in ext], [str(n) for n in names]


def downsample(a: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    """Area-average ``a`` so no side exceeds ``max_side`` (unchanged when it already fits)."""
    target = tuple(min(int(n), int(max_side)) for n in a.shape)
    if target == tuple(a.shape):
        return a
    return np.asarray(match_shape(a, target), dtype=float)


def auto_stretch(extent: Sequence[Sequence[float]], axis: int = -1) -> float:
    """Depth stretch of thin slabs (as the static voxel views): physical (1) unless the volume
    axis is shorter than 35 % of the largest lateral side, then stretched to 35 %."""
    spans = [abs(float(hi) - float(lo)) or 1.0 for lo, hi in extent]
    ax = axis % 3
    lateral = max(s for i, s in enumerate(spans) if i != ax)
    k = 0.35 * lateral / spans[ax]
    return 1.0 if k <= 1.0 else float(f"{k:.2g}")


def _spec(cmap: Any, field: str | None, ref: np.ndarray) -> CmapSpec:
    if isinstance(cmap, CmapSpec):
        return cmap
    if cmap is not None:
        from .style import quantity_of

        hinted = spec_from_hint(cmap, ref, quantity_of(field))
        if hinted is not None:
            return hinted
    return cmap_for(field, ref)


def viewer_cmap(spec: CmapSpec) -> str:
    """The viewer's colormap for a spec: the same name when implemented (``_r`` variants
    included), else by kind (diverging → ``RdBu_r``, cyclic → ``twilight``, ``viridis``)."""
    name = spec.cmap
    base = name[:-2] if name.endswith("_r") else name
    if name in VIEWER_CMAPS or base in VIEWER_CMAPS or f"{name}_r" in VIEWER_CMAPS:
        return name
    if spec.kind == "diverging":
        return "RdBu_r"
    return "twilight" if spec.kind == "cyclic" else "viridis"


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def quantize(
    a: np.ndarray, bits: int = 8, rules: Sequence[Mapping[str, Any]] = ()
) -> tuple[dict[str, Any], np.ndarray]:
    """Quantize a volume to ``bits``-bit codes ``lo + q · step`` (the top code is NaN when the
    volume has NaNs); codes are nudged so each rule in ``rules`` selects exactly the voxels it
    selects in ``a`` (relative rules use ``a``'s own max / min, embedded as ``peak`` /
    ``trough``). Returns ``(entry, decoded)`` — the JSON entry and the values the viewer sees."""
    if bits not in (8, 16):
        raise ValueError("bits must be 8 or 16")
    a = np.asarray(a, dtype=float)
    fin = np.isfinite(a)
    has_nan = not bool(fin.all())
    top = (1 << bits) - 1
    levels = top - 1 if has_nan else top
    vals = a[fin]
    lo = float(vals.min()) if vals.size else 0.0
    hi = float(vals.max()) if vals.size else 0.0
    step = (hi - lo) / levels if hi > lo else 1.0
    q = np.zeros(a.shape, dtype=np.int64)
    q[fin] = np.clip(np.rint((a[fin] - lo) / step), 0, levels).astype(np.int64)
    for rule in rules:
        want = select_mask(a, rule, peak=hi, trough=lo)
        up = 1 if rule.get("side", "above") == "above" else -1
        for _ in range(8):
            got = select_mask(lo + q.astype(np.float64) * step, rule, peak=hi, trough=lo)
            bad = fin & (got != want)
            if not bad.any():
                break
            q[bad & want] += up
            q[bad & ~want] -= up
            np.clip(q, 0, levels, out=q)
    if has_nan:
        q[~fin] = top
    codes = q.astype("<u2" if bits == 16 else np.uint8)
    decoded = lo + q.astype(np.float64) * step
    decoded[~fin] = np.nan
    entry = {
        "bits": bits,
        "lo": lo,
        "step": step,
        "nan": has_nan,
        "peak": hi,
        "trough": lo,
        "data": _b64(codes),
    }
    return entry, decoded


def decode_volume(entry: Mapping[str, Any], shape: Sequence[int]) -> np.ndarray:
    """The values the viewer decodes from a payload volume (mirror of the JS ``decode``)."""
    raw = base64.b64decode(entry["data"])
    bits = int(entry.get("bits", 8))
    q = np.frombuffer(raw, dtype="<u2" if bits == 16 else np.uint8).astype(np.float64)
    out = entry["lo"] + q * entry["step"]
    if entry.get("nan"):
        out[q == (1 << bits) - 1] = np.nan
    return out.reshape(tuple(int(s) for s in shape))


def encode_mesh(
    verts: np.ndarray, faces: np.ndarray, shape: Sequence[int], **meta: Any
) -> dict[str, Any]:
    """A mesh entry: vertices (grid index coordinates in ``[-0.5, n - 0.5]``) as uint16 over each
    axis, faces as uint16 (≤ 65535 vertices) or uint32; ``meta`` (level, side, rule) is kept."""
    n = np.asarray(shape, dtype=float)
    q = np.clip(np.rint((np.asarray(verts, float) + 0.5) / n * 65535.0), 0, 65535).astype("<u2")
    nv = int(len(verts))
    fbits = 16 if nv <= 65535 else 32
    f = np.asarray(faces).astype("<u2" if fbits == 16 else "<u4")
    return {**meta, "nv": nv, "nf": int(len(faces)), "fbits": fbits, "v": _b64(q), "f": _b64(f)}


def decode_mesh(entry: Mapping[str, Any], shape: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices (V, 3), faces (F, 3))`` of a mesh entry (mirror of the JS ``decodeMesh``)."""
    n = np.asarray(shape, dtype=float)
    q = np.frombuffer(base64.b64decode(entry["v"]), dtype="<u2").reshape(-1, 3)
    dt = "<u2" if entry.get("fbits", 16) == 16 else "<u4"
    f = np.frombuffer(base64.b64decode(entry["f"]), dtype=dt).reshape(-1, 3).astype(np.int64)
    return q / 65535.0 * n - 0.5, f


def _recon_levels(
    d: np.ndarray,
    rule: Mapping[str, Any],
    sg: np.ndarray,
    n_g: int,
    peak: float,
    trough: float,
) -> dict[str, dict[str, Any]]:
    """Fixed / matched / Otsu levels of one displayed reconstruction with count, IoU, Dice."""
    side = rule["side"]
    out = {}
    for how in ("fixed", "matched", "otsu"):
        if how == "fixed":
            t = rule_level(rule, peak=peak, trough=trough)
            s = select_mask(d, rule, peak=peak, trough=trough)
        else:
            t = matched_level(d, side, n_g) if how == "matched" else otsu_level(d)
            s = select_mask(d, level_rule(t, side))
        iou, dice = iou_dice(sg, s)
        out[how] = {"level": _num(t), "count": int(s.sum()), "iou": _num(iou), "dice": _num(dice)}
    return out


def viewer_payload(
    volumes: Any,
    *,
    extent: Any = None,
    threshold: Any = None,
    mode: str = "voxels",
    cmap: Any = None,
    title: str | None = None,
    field: str | None = None,
    label: str | None = None,
    axes: Sequence[str] | None = None,
    axis: int = -1,
    level_rule: str = "matched",
    transform: Any = None,
    meshes: bool | str = "auto",
    max_faces: int = MAX_FACES,
    max_side: int = MAX_SIDE,
    bits: int | str = "auto",
    stretch: float | str = "auto",
    vmin: float | None = None,
    vmax: float | None = None,
    center: Sequence[int] | None = None,
    opacity: float = 1.0,
    overlay: bool = False,
) -> dict[str, Any]:
    """The JSON payload of a viewer (see :func:`volume_viewer_html` for the arguments).

    Volumes (a ``{name: volume}`` mapping — the first is the reference, e.g. the ground truth —
    or one volume) are resampled to the first one's grid, area-averaged to ``≤ max_side`` per
    axis and quantized (:func:`quantize`). The default levels, IoU / Dice and isosurface meshes
    are computed from the decoded values exactly as the browser computes them.
    """
    items = _volume_items(volumes, field)
    src_shape = items[0][1].shape
    items = [(n, match_shape(a, src_shape) if a.shape != src_shape else a) for n, a in items]
    ds = [(n, downsample(a, max_side)) for n, a in items]
    shape = tuple(int(s) for s in ds[0][1].shape)
    ext, names = _extent_axes(extent, src_shape, axes)
    ax = int(axis) % 3
    ref = ds[0][1]
    rule = normalize_rule(threshold, ref)
    if level_rule not in LEVEL_RULES or level_rule == "manual":
        raise ValueError(f"level_rule must be one of {LEVEL_RULES[:3]}")
    tf = transform_spec(transform)
    spec = _spec(cmap, field, ref)
    lo, hi = color_limits([a for _, a in ds], spec)
    lo = float(lo if vmin is None else vmin)
    hi = float(hi if vmax is None else vmax)
    mode = normalize_mode(mode)
    if bits == "auto":  # small volumes: 16-bit (levels match the full-precision ones); 64³: 8
        bits = 16 if int(np.prod(shape)) <= AUTO_16BIT_VOXELS else 8
    vols: list[dict[str, Any]] = []
    dec: list[np.ndarray] = []
    for name, a in ds:
        entry, d = quantize(a, int(bits), [rule])
        vols.append({"name": name, **entry})
        dec.append(d)
    # displayed values: reconstructions after the display transform (the GT never)
    disp = [dec[0]] + [volume_transform(d, tf) if tf else d for d in dec[1:]]
    g0 = vols[0]
    sg = select_mask(disp[0], rule, peak=g0["peak"], trough=g0["trough"])
    n_g = int(sg.sum())
    g0["level"] = _num(rule_level(rule, peak=g0["peak"], trough=g0["trough"]))
    g0["count"] = n_g
    for k in range(1, len(vols)):
        d = disp[k]
        fin = d[np.isfinite(d)]
        pk, tr = (
            (float(fin.max()), float(fin.min()))
            if (tf and fin.size)
            else (vols[k]["peak"], vols[k]["trough"])
        )
        lv = _recon_levels(d, rule, sg, n_g, pk, tr)
        vols[k]["levels"] = lv
        vols[k]["level"] = lv[level_rule]["level"]
        vols[k]["count"] = lv[level_rule]["count"]
    want_mesh = meshes is True or (meshes == "auto" and mode.startswith("iso"))
    mesh_list = None
    if want_mesh and has_marching_cubes():
        mesh_list = []
        for k, v in enumerate(vols):
            t = v.get("level")
            got = iso_mesh(disp[k], t, rule["side"], max_faces=max_faces) if t is not None else None
            if got is None:
                mesh_list.append(None)
                continue
            how = "rule" if k == 0 else level_rule
            mesh_list.append(encode_mesh(*got, shape, level=t, side=rule["side"], rule=how))
    if center is not None:
        c = [
            int(round((int(ci) + 0.5) * s / n - 0.5)) for ci, s, n in zip(center, shape, src_shape)
        ]
    else:
        c = list(anomaly_centroid(ref))
    c = [int(min(max(ci, 0), s - 1)) for ci, s in zip(c, shape)]
    st = auto_stretch(ext, ax) if stretch == "auto" else float(stretch)
    rule_out = {k: (_num(v) if isinstance(v, float) else v) for k, v in rule.items()}
    return {
        "version": 2,
        "kind": "canvas",
        "title": title,
        "shape": list(shape),
        "source_shape": [int(s) for s in src_shape],
        "downsampled": shape != tuple(src_shape),
        "extent": ext,
        "axes": names,
        "axis": ax,
        "stretch": st,
        "cmap": viewer_cmap(spec),
        "vmin": lo,
        "vmax": hi,
        "label": str(label or field or "value"),
        "mode": mode,
        "opacity": float(opacity),
        "overlay": bool(overlay),
        "threshold": rule_out,
        "level_rule": level_rule,
        "transform": tf,
        "center": c,
        "slice": {"axis": ax, "index": c[ax]},
        "volumes": vols,
        "meshes": mesh_list,
    }


def payload_levels(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Levels, counts and IoU / Dice at the payload's defaults, recomputed from the decoded
    values like the browser does (tests and tools)."""
    shape = payload["shape"]
    rule = payload["threshold"]
    tf = payload.get("transform")
    vols = payload["volumes"]
    dec = [decode_volume(v, shape) for v in vols]
    disp = [dec[0]] + [volume_transform(d, tf) if tf else d for d in dec[1:]]
    g0 = vols[0]
    sg = select_mask(disp[0], rule, peak=g0["peak"], trough=g0["trough"])
    out = [
        {"level": rule_level(rule, peak=g0["peak"], trough=g0["trough"]), "count": int(sg.sum())}
    ]
    for k in range(1, len(vols)):
        t = vols[k]["level"]
        how = payload.get("level_rule", "matched")
        if how == "fixed":
            s = select_mask(disp[k], rule, peak=vols[k]["peak"], trough=vols[k]["trough"])
        else:
            s = select_mask(disp[k], level_rule(t, rule["side"]))
        iou, dice = iou_dice(sg, s)
        out.append({"level": t, "count": int(s.sum()), "iou": iou, "dice": dice})
    return out


# ---------------------------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------------------------
_ID = re.compile(r"[A-Za-z][\w-]*")


def viewer_fragment(
    payload: Mapping[str, Any],
    *,
    height: int = 420,
    include_assets: bool = True,
    element_id: str | None = None,
) -> str:
    """HTML fragment of one viewer: the (optional) assets, a ``<div>`` with a unique id, the JSON
    payload in a ``<script type="application/json">`` and a one-line mount script."""
    uid = element_id or f"nefi-vv-{uuid.uuid4().hex[:12]}"
    if not _ID.fullmatch(uid):
        raise ValueError(f"invalid element id {uid!r}")
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    note = (
        "<noscript><p class='nefi-vv-note'>The interactive 3-D viewer needs JavaScript; the "
        "static figures show the same volumes.</p></noscript>"
    )
    parts = [assets_html()] if include_assets else []
    parts += [
        f"<div class='nefi-vv' id='{uid}' style='--vv-h:{int(height)}px'>{note}</div>",
        f"<script type='application/json' id='{uid}-data'>{data}</script>",
        "<script>(window.NefiVolumeViewerQueue=window.NefiVolumeViewerQueue||[])"
        f".push('{uid}');window.NefiVolumeViewer&&window.NefiVolumeViewer.flush();</script>",
    ]
    return "\n".join(parts)


def standalone_page(body: str, title: str, subtitle: str | None = None) -> str:
    """A whole HTML document (report stylesheet, light / dark) around ``body``."""
    from .report import _CSS

    sub = f"<p class='subtitle'>{html.escape(subtitle)}</p>" if subtitle else ""
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang='en'>",
            "<head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            f"<title>{html.escape(title)}</title>",
            f"<style>{_CSS}</style></head>",
            "<body><main>",
            f"<header><h1>{html.escape(title)}</h1>{sub}</header>",
            body,
            "<footer><p>Generated by <code>nefi.viz.interactive</code>.</p></footer>",
            "</main></body></html>",
        ]
    )


def volume_viewer_html(
    volumes: Any,
    *,
    extent: Any = None,
    threshold: Any = None,
    mode: str = "voxels",
    cmap: Any = None,
    title: str | None = None,
    height: int = 420,
    standalone: bool = False,
    include_assets: bool = True,
    element_id: str | None = None,
    **kw: Any,
) -> str:
    """HTML of an interactive 3-D viewer of one or two volumes (same camera).

    Args:
        volumes: ``{name: volume}`` (tensors, arrays, Results with ``field``) — the first is the
            reference (ground truth: its threshold rule, IoU / Dice against it) — or one volume.
        extent: physical extent: a :class:`~nefi.domain.Domain`, ``[(lo, hi)] * 3`` or three
            lengths (default: voxel units); axis names come from the Domain.
        threshold: GT threshold rule: ``None`` (half-way from the GT background to its extreme),
            a number, ``(side, value)`` or a rule mapping (:func:`~nefi.viz.isosurface.voxel_rule`
            gives an instance's IoU rule).
        mode: ``"voxels"``, ``"iso"`` (isosurfaces), ``"slice"``, ``"voxels+slice"`` (alias
            ``"points+slice"``) or ``"iso+slice"``.
        cmap: colormap (name / :class:`~nefi.viz.style.CmapSpec`; default by field name / data).
        title: title above the panels (and of the page with ``standalone``).
        height: canvas height in CSS pixels.
        standalone: return a whole ``<!doctype html>`` document.
        include_assets: include the script and stylesheet (once per page is enough).
        element_id: explicit element id (default: unique).
        **kw: :func:`viewer_payload` options — ``field``, ``label``, ``axes``, ``axis`` (volume
            axis), ``level_rule`` (``"matched"`` | ``"fixed"`` | ``"otsu"``), ``transform``
            (display transform of the reconstructions), ``meshes`` (embed marching-cubes
            isosurfaces: ``"auto"`` = in isosurface modes), ``max_faces``, ``max_side``
            (default 64), ``bits`` (8 / 16; ``"auto"``: 16 up to 48³ voxels, else 8),
            ``stretch`` (depth stretch, ``"auto"``), ``vmin``, ``vmax``, ``center``,
            ``opacity``, ``overlay``.
    """
    payload = viewer_payload(
        volumes,
        extent=extent,
        threshold=threshold,
        mode=mode,
        cmap=cmap,
        title=None if standalone else title,  # the page header carries it
        **kw,
    )
    frag = viewer_fragment(
        payload,
        height=height,
        include_assets=include_assets or standalone,
        element_id=element_id,
    )
    if standalone:
        return standalone_page(frag, title or "nefi volume viewer")
    return frag


def save_volume_viewer(path: str | Path, volumes: Any, *, kind: str = "canvas", **kw: Any) -> Path:
    """Write a standalone viewer page (``kind="plotly"``: the CDN-backed plotly page)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if normalize_kind(kind) == "plotly":
        text = plotly_volume_html(volumes, standalone=True, **kw)
    else:
        text = volume_viewer_html(volumes, standalone=True, **kw)
    p.write_text(text, encoding="utf-8")
    return p


def viewer_size_bytes(volumes: Any, *, include_assets: bool = False, **kw: Any) -> int:
    """Bytes a viewer adds to a page: the fragment of ``volumes`` (or of a ready payload),
    without the shared script / stylesheet unless ``include_assets``."""
    if isinstance(volumes, Mapping) and "volumes" in volumes and "version" in volumes:
        frag = viewer_fragment(volumes, include_assets=include_assets, height=kw.get("height", 420))
    else:
        frag = volume_viewer_html(volumes, include_assets=include_assets, **kw)
    return len(frag.encode("utf-8"))


# ---------------------------------------------------------------------------------------------
# plotly (optional, CDN)
# ---------------------------------------------------------------------------------------------
_PLOTLY_SYNC = """
var gd = document.getElementById('{plot_id}'), busy = false;
if (gd && gd.on) gd.on('plotly_relayout', function (ev) {
  if (busy) return;
  var src = null, cam = null, k, upd = {};
  for (k in ev) { var m = k.match(/^(scene\\d*)\\.camera$/); if (m) { src = m[1]; cam = ev[k]; } }
  if (!cam) return;
  Object.keys(gd._fullLayout || {}).forEach(function (s) {
    if (/^scene\\d*$/.test(s) && s !== src) upd[s + '.camera'] = cam;
  });
  busy = true;
  Plotly.relayout(gd, upd).then(function () { busy = false; });
});
"""


def plotly_cdn_tag() -> str:
    """The ``<script src=…>`` tag of the plotly.js CDN build matching the installed plotly."""
    from plotly.offline import get_plotlyjs_version

    return (
        f"<script src='https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js' "
        "charset='utf-8'></script>"
    )


def plotly_volume_html(
    volumes: Any,
    *,
    extent: Any = None,
    threshold: Any = None,
    cmap: Any = None,
    title: str | None = None,
    height: int = 420,
    standalone: bool = False,
    field: str | None = None,
    label: str | None = None,
    axes: Sequence[str] | None = None,
    axis: int = -1,
    level_rule: str = "matched",
    transform: Any = None,
    max_side: int = 32,
    stretch: float | str = "auto",
    include_plotlyjs: bool | str = "cdn",
    opacity: float = 0.9,
    **_: Any,
) -> str:
    """Isosurfaces with ``plotly.graph_objects.Isosurface`` (one scene per volume, cameras
    synchronized): the GT at its threshold rule, reconstructions at their ``level_rule`` level.

    **Not self-contained**: ``include_plotlyjs="cdn"`` loads plotly.js from ``cdn.plot.ly``
    (``False`` when the page already loads it; ``True`` inlines ~3.5 MB). Volumes are
    area-averaged to ``max_side`` (default 32: plotly serializes every grid point as JSON text).

    Raises:
        ImportError: plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:  # pragma: no cover - exercised only without plotly
        raise ImportError(
            "the plotly viewer needs `pip install plotly`; the default canvas viewer has no "
            "dependencies"
        ) from e
    from .isosurface import surface_colors

    items = _volume_items(volumes, field)
    src_shape = items[0][1].shape
    items = [(n, match_shape(a, src_shape) if a.shape != src_shape else a) for n, a in items]
    ds = [(n, downsample(a, max_side)) for n, a in items]
    shape = ds[0][1].shape
    ext, names = _extent_axes(extent, src_shape, axes)
    ax = int(axis) % 3
    rule = normalize_rule(threshold, ds[0][1])
    tf = transform_spec(transform)
    side = rule["side"]
    disp = [ds[0][1]] + [volume_transform(a, tf) if tf else a for _, a in ds[1:]]
    sg = select_mask(disp[0], rule)
    n_g = int(sg.sum())
    q = label or field or "value"
    sym = "<" if side == "below" else ">"
    levels, heads = (
        [rule_level(rule, disp[0])],
        [f"{ds[0][0]}: {q} {sym} {rule_level(rule, disp[0]):.3g}"],
    )
    for k in range(1, len(ds)):
        fin = disp[k][np.isfinite(disp[k])]
        lv = _recon_levels(disp[k], rule, sg, n_g, float(fin.max()), float(fin.min()))
        t = lv[level_rule]["level"]
        levels.append(t)
        iou = lv[level_rule]["iou"]
        heads.append(
            f"{ds[k][0]}: {q} {sym} {t:.3g} ({RULE_LABELS[level_rule]}), IoU "
            + (f"{iou:.2f}" if iou is not None else "—")
        )
    coords = [
        ext[d][0] + (np.arange(shape[d]) + 0.5) * (ext[d][1] - ext[d][0]) / shape[d]
        for d in range(3)
    ]
    X, Y, Z = (g.ravel() for g in np.meshgrid(*coords, indexing="ij"))
    n = len(ds)
    fig = make_subplots(
        rows=1,
        cols=n,
        specs=[[{"type": "scene"}] * n],
        subplot_titles=heads,
        horizontal_spacing=0.02,
    )
    cols = surface_colors(n)
    for k, ((_, _a), d, t) in enumerate(zip(ds, disp, levels)):
        if t is None:
            continue
        v = np.nan_to_num(d, nan=float(np.nanmedian(d))).ravel()
        fig.add_trace(
            go.Isosurface(
                x=X,
                y=Y,
                z=Z,
                value=np.round(v, 6),
                isomin=float(t),
                isomax=float(t),
                surface_count=1,
                colorscale=[[0.0, cols[k]], [1.0, cols[k]]],
                showscale=False,
                opacity=opacity,
                name=ds[k][0],
            ),
            row=1,
            col=k + 1,
        )
    spans = [abs(e[1] - e[0]) or 1.0 for e in ext]
    st = auto_stretch(ext, ax) if stretch == "auto" else float(stretch)
    spans[ax] *= st
    top = max(spans)
    scene = {
        "aspectmode": "manual",
        "aspectratio": {"x": spans[0] / top, "y": spans[1] / top, "z": spans[2] / top},
        "xaxis": {"title": names[0]},
        "yaxis": {"title": names[1]},
        "zaxis": {"title": names[2] + (f" (×{st:g})" if st != 1 else ""), "autorange": "reversed"},
    }
    fig.update_scenes(**scene)
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        title=(title or "") + (f" ({transform_label(tf)})" if tf else ""),
        showlegend=False,
    )
    return fig.to_html(
        full_html=standalone,
        include_plotlyjs=include_plotlyjs,
        default_height=f"{int(height)}px",
        post_script=_PLOTLY_SYNC,
    )


# ---------------------------------------------------------------------------------------------
# gallery helpers
# ---------------------------------------------------------------------------------------------
def entry_viewer_inputs(entry: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """``(volumes, options)`` of a 3-D gallery entry: GT and reconstruction on the GT grid (the
    instance's display hints applied, :func:`~nefi.viz.multiphysics.tile_data`) — with an edge
    refinement GT | smooth | refined (or the refused candidate, labelled) —, the instance's IoU
    rule as the GT threshold, the anomaly centroid, the ``volume_transform`` hint."""
    from .hints import volume_axis
    from .multiphysics import refined_view, tile_data

    run = entry.run
    td = tile_data(run)
    ref = td.g if td.g is not None else td.rec
    rv = refined_view(run, td)
    vols = {"ground truth": td.g} if td.g is not None else {}
    vols["smooth" if rv is not None else "reconstruction"] = td.rec
    if rv is not None:
        vols[rv[1]] = rv[0]
    rule = voxel_rule(run.instance, ref, field=td.name) if not td.note else auto_rule(ref)
    opts = {
        "extent": td.domain,
        "threshold": rule,
        "cmap": td.spec,
        "field": td.name,
        "label": td.label,
        "axis": volume_axis(td.hints),
        "center": anomaly_centroid(ref, mask=td.mask),
        "transform": hinted_transform(td.hints, td.name),
        "title": f"{entry.name}: {td.label} — ground truth vs "
        + ("smooth vs refined" if rv is not None else "reconstruction"),
    }
    return vols, opts


def write_entry_viewer(entry: Any, out_dir: str | Path, *, kind: str = "canvas") -> dict[str, Any]:
    """Write the viewer of a 3-D gallery entry: ``<name>/viewer.json`` (what reports embed) and
    ``<name>/viewer.html`` (a standalone page). Returns the manifest info (paths relative to
    ``out_dir``, sizes in bytes, grid, rule, default levels and IoU / Dice)."""
    out = Path(out_dir)
    d = out / entry.name
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    vols, opts = entry_viewer_inputs(entry)
    kind = normalize_kind(kind) or "canvas"
    info: dict[str, Any] = {}
    if kind == "plotly":
        try:
            frag = plotly_volume_html(vols, include_plotlyjs=False, **opts)
            page = plotly_volume_html(vols, standalone=True, **opts)
            obj = {"kind": "plotly", "fragment": frag, "cdn": plotly_cdn_tag()}
            (d / "viewer.json").write_text(json.dumps(obj), encoding="utf-8")
            (d / "viewer.html").write_text(page, encoding="utf-8")
            info = {"kind": "plotly", "fragment_bytes": len(frag.encode("utf-8"))}
        except ImportError as e:
            log.warning("plotly viewer unavailable (%s); using the canvas viewer", e)
            kind = "canvas"
    if kind == "canvas":
        payload = viewer_payload(vols, mode="iso", meshes=True, **opts)
        text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        (d / "viewer.json").write_text(text, encoding="utf-8")
        page = standalone_page(viewer_fragment({**payload, "title": None}), opts["title"])
        (d / "viewer.html").write_text(page, encoding="utf-8")
        rec = payload["volumes"][min(1, len(payload["volumes"]) - 1)]  # the (smooth) recon
        info = {
            "kind": "canvas",
            "data_bytes": len(text.encode("utf-8")),
            "fragment_bytes": viewer_size_bytes(payload),
            "mesh_faces": [m["nf"] if m else None for m in (payload["meshes"] or [])],
            "shape": payload["shape"],
            "source_shape": payload["source_shape"],
            "threshold": payload["threshold"].get("source"),
            "level_rule": payload["level_rule"],
            "gt_level": payload["volumes"][0].get("level"),
            "levels": rec.get("levels"),
        }
    info["data"] = os.path.relpath(d / "viewer.json", out)
    info["html"] = os.path.relpath(d / "viewer.html", out)
    info["html_bytes"] = (d / "viewer.html").stat().st_size
    info["seconds"] = time.perf_counter() - t0
    return info


def viewer_fragment_from_file(
    path: str | Path,
    *,
    height: int = 420,
    include_assets: bool = True,
    include_plotlyjs: bool = True,
) -> str:
    """The report fragment of a ``viewer.json`` written by :func:`write_entry_viewer` (fresh
    element ids; the canvas assets / the plotly CDN tag only when asked — once per page)."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("kind") == "plotly":
        return (obj.get("cdn", "") if include_plotlyjs else "") + obj["fragment"]
    return viewer_fragment(obj, height=height, include_assets=include_assets)


__all__ = [
    "KINDS",
    "AUTO_16BIT_VOXELS",
    "MAX_FACES",
    "MAX_SIDE",
    "MODES",
    "VIEWER_CMAPS",
    "asset_bytes",
    "assets_html",
    "auto_stretch",
    "decode_mesh",
    "decode_volume",
    "downsample",
    "encode_mesh",
    "entry_viewer_inputs",
    "normalize_kind",
    "normalize_mode",
    "payload_levels",
    "plotly_cdn_tag",
    "plotly_volume_html",
    "quantize",
    "save_volume_viewer",
    "standalone_page",
    "viewer_assets",
    "viewer_cmap",
    "viewer_fragment",
    "viewer_fragment_from_file",
    "viewer_payload",
    "viewer_size_bytes",
    "volume_viewer_html",
    "write_entry_viewer",
]
