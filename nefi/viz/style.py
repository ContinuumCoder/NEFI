"""House style for nefi figures: palette, per-quantity colormaps, rc settings, sizes, saving.

Every plotting function of :mod:`nefi.viz` draws through this module so all figures read as one
system:

* **categorical palette** — eight colorblind-validated hues in a *fixed* order (never cycled past
  eight; extra series are folded), with a light and a dark variant (:data:`LIGHT`, :data:`DARK`);
* **colormaps chosen by the physical quantity** (:func:`cmap_for`): densities / sources →
  ``magma``; diffusivity / conductivity / permeability / velocity → ``viridis``; signed fields
  and errors → ``RdBu_r`` centred at 0; wrapped phases → ``twilight`` on ``[-π, π]`` (phases
  spanning less than π use the centred diverging map instead);
* **rc settings** (:func:`use_style`): small sans-serif type, hairline solid grids, no top/right
  spines, ``nearest`` interpolation, constrained layout;
* **size helpers** (:func:`figsize`, :func:`paper_figsize`) and :func:`savefig`, which writes PNG
  (optionally SVG/PDF), keeps PNGs under a size budget (palette quantization, then lower dpi) and
  never needs a GUI backend.

matplotlib is imported lazily inside functions: importing this module never requires it.
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

log = logging.getLogger("nefi")

# ---------------------------------------------------------------------------------------------
# palette and themes
# ---------------------------------------------------------------------------------------------
#: Categorical slots (identity), light surface. Order is part of the colorblind-safety design
#: (adjacent CVD ΔE ≥ 8, normal-vision ΔE ≥ 15); assign in order, never cycle past eight.
PALETTE_LIGHT: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
#: The same eight hues stepped for the dark surface.
PALETTE_DARK: tuple[str, ...] = (
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
)
PALETTE = PALETTE_LIGHT


@dataclass(frozen=True)
class Theme:
    """Surface / ink tokens and the categorical palette of one color mode."""

    name: str
    surface: str  # figure and axes background
    page: str  # page plane (HTML reports)
    ink: str  # primary text, ground-truth lines
    ink2: str  # secondary text (labels, legends)
    muted: str  # ticks, annotations, measurement dots
    grid: str  # hairline gridlines
    axis: str  # spines / baselines
    bad: str  # masked / unobserved pixels
    palette: tuple[str, ...]

    def color(self, i: int) -> str:
        """Categorical slot ``i`` (0-based). Slots are never cycled: ``i >= 8`` is an error."""
        if not 0 <= i < len(self.palette):
            raise IndexError(
                f"categorical slot {i} out of range (8 slots); fold extra series into 'other'"
            )
        return self.palette[i]


LIGHT = Theme(
    "light",
    surface="#fcfcfb",
    page="#f9f9f7",
    ink="#0b0b0b",
    ink2="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    bad="#dddcd6",
    palette=PALETTE_LIGHT,
)
DARK = Theme(
    "dark",
    surface="#1a1a19",
    page="#0d0d0d",
    ink="#ffffff",
    ink2="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    bad="#3a3a37",
    palette=PALETTE_DARK,
)

#: Paper-figure widths in inches (single / double column of a two-column journal layout).
COLUMN_WIDTH = 3.4
TEXT_WIDTH = 7.0

#: Default image-panel size (width, height) in inches per context.
PANEL_SIZES: dict[str, tuple[float, float]] = {
    "paper": (1.45, 1.75),
    "notebook": (1.95, 2.3),
    "talk": (2.6, 3.0),
}
#: Width reserved for one colorbar (inches).
CBAR_WIDTH = 0.6
_FONT_SIZES = {"paper": 7.5, "notebook": 8.5, "talk": 11.0}

# style stack: (theme, context); the bottom entry is the process-wide default
_STACK: list[tuple[Theme, str]] = [(LIGHT, "notebook")]
_DEPTH = [0]  # nesting depth of active use_style() blocks


def set_default_style(dark: bool = False, context: str = "notebook") -> None:
    """Process-wide default theme / context used when no :func:`use_style` block is active."""
    _check_context(context)
    _STACK[0] = (DARK if dark else LIGHT, context)


def theme(dark: bool | None = None) -> Theme:
    """The active theme (``dark=None``) or the light / dark theme explicitly."""
    if dark is None:
        return _STACK[-1][0]
    return DARK if dark else LIGHT


def current_context() -> str:
    """Active size context (``"paper"``, ``"notebook"`` or ``"talk"``)."""
    return _STACK[-1][1]


def palette(dark: bool | None = None) -> tuple[str, ...]:
    """The eight categorical colors of the active (or requested) theme."""
    return theme(dark).palette


def series_colors(n: int, dark: bool | None = None) -> list[str]:
    """Colors for ``n`` series in slot order. More than eight series must be folded first."""
    pal = palette(dark)
    if n > len(pal):
        raise ValueError(f"{n} series exceed the {len(pal)} categorical slots; fold or facet")
    return list(pal[:n])


def _check_context(context: str) -> None:
    if context not in PANEL_SIZES:
        raise ValueError(f"unknown context {context!r}; use one of {sorted(PANEL_SIZES)}")


def style_rc(dark: bool | None = None, context: str | None = None) -> dict[str, Any]:
    """matplotlib rcParams of the house style for a theme and size context."""
    t = theme(dark)
    context = context or current_context()
    _check_context(context)
    fs = _FONT_SIZES[context]
    from cycler import cycler

    return {
        "figure.facecolor": t.surface,
        "figure.edgecolor": t.surface,
        "axes.facecolor": t.surface,
        "savefig.facecolor": t.surface,
        "savefig.edgecolor": t.surface,
        "axes.edgecolor": t.axis,
        "axes.labelcolor": t.ink2,
        "axes.titlecolor": t.ink,
        "text.color": t.ink,
        "xtick.color": t.axis,
        "ytick.color": t.axis,
        "xtick.labelcolor": t.ink2,
        "ytick.labelcolor": t.ink2,
        "grid.color": t.grid,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.prop_cycle": cycler(color=list(t.palette)),
        "axes.titlepad": 3.0,
        "axes.labelpad": 2.0,
        "axes.formatter.limits": (-3, 4),
        "axes.formatter.use_mathtext": True,
        "axes.formatter.useoffset": False,
        "lines.linewidth": 1.3,
        "lines.markersize": 4.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "patch.linewidth": 0.0,
        "font.size": fs,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "axes.titlesize": fs,
        "axes.labelsize": fs,
        "xtick.labelsize": fs - 1,
        "ytick.labelsize": fs - 1,
        "legend.fontsize": fs - 1,
        "legend.title_fontsize": fs - 1,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3,
        "legend.labelcolor": t.ink2,
        "figure.titlesize": fs + 1.5,
        "figure.titleweight": "normal",
        "figure.labelsize": fs,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "image.interpolation": "nearest",
        "image.cmap": "viridis",
        "mathtext.default": "regular",
        "figure.dpi": 100,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "hatch.color": t.muted,
        "hatch.linewidth": 0.6,
    }


@contextlib.contextmanager
def use_style(dark: bool = False, context: str = "notebook", **rc: Any) -> Iterator[Theme]:
    """Context manager applying the house style (light or dark) and size context.

    Example::

        with nefi.viz.use_style(dark=True, context="paper"):
            fig = nefi.viz.compare_fields(gt, {"nefi": result})
            nefi.viz.savefig(fig, "runs/fig/compare_dark")

    Args:
        dark: dark surface / ink tokens and the dark categorical steps.
        context: ``"paper"`` (7.5 pt, compact panels), ``"notebook"`` (default) or ``"talk"``.
        **rc: extra rcParams overrides.

    Yields:
        The active :class:`Theme`.
    """
    import matplotlib as mpl

    _check_context(context)
    t = DARK if dark else LIGHT
    _STACK.append((t, context))
    _DEPTH[0] += 1
    try:
        with mpl.rc_context({**style_rc(dark, context), **rc}):
            yield t
    finally:
        _STACK.pop()
        _DEPTH[0] -= 1


@contextlib.contextmanager
def styled(dark: bool | None = None) -> Iterator[Theme]:
    """Apply the house style for one plotting call.

    Inside an active :func:`use_style` block with ``dark=None`` nothing changes (the caller's
    style wins); otherwise the default (or the requested) theme is applied.
    """
    if dark is None and _DEPTH[0] > 0:
        yield theme()
        return
    t, ctx = _STACK[-1]
    want_dark = (t is DARK) if dark is None else bool(dark)
    with use_style(dark=want_dark, context=ctx) as th:
        yield th


# ---------------------------------------------------------------------------------------------
# colormaps per quantity
# ---------------------------------------------------------------------------------------------
#: quantity -> (colormap, kind). kind: "sequential" | "diverging" (centred at 0) | "cyclic".
QUANTITY_CMAPS: dict[str, tuple[str, str]] = {
    # non-negative "amount" quantities: magma (dark = none)
    "density": ("magma", "sequential"),
    "source": ("magma", "sequential"),
    "intensity": ("magma", "sequential"),
    "attenuation": ("magma", "sequential"),
    "absorption": ("magma", "sequential"),
    "concentration": ("magma", "sequential"),
    "magnitude": ("magma", "sequential"),
    "energy": ("magma", "sequential"),
    "power": ("magma", "sequential"),
    "sensitivity": ("magma", "sequential"),
    # material coefficients: viridis
    "diffusivity": ("viridis", "sequential"),
    "conductivity": ("viridis", "sequential"),
    "permeability": ("viridis", "sequential"),
    "velocity": ("viridis", "sequential"),
    "refractive_index": ("viridis", "sequential"),
    "permittivity": ("viridis", "sequential"),
    "frequency": ("viridis", "sequential"),
    "temperature": ("inferno", "sequential"),
    "measurement": ("viridis", "sequential"),
    "generic": ("viridis", "sequential"),
    "absolute_error": ("magma", "sequential"),
    "mask": ("Greys", "sequential"),
    # signed quantities: diverging, centred at zero
    "signed": ("RdBu_r", "diverging"),
    "error": ("RdBu_r", "diverging"),
    "residual": ("RdBu_r", "diverging"),
    "current": ("RdBu_r", "diverging"),
    "potential": ("RdBu_r", "diverging"),
    "wavefield": ("RdBu_r", "diverging"),
    "field": ("RdBu_r", "diverging"),
    "contrast": ("RdBu_r", "diverging"),
    # cyclic
    "phase": ("twilight", "cyclic"),
    "angle": ("twilight", "cyclic"),
}

#: Common field names -> quantity (lower-case keys; see :func:`quantity_of`).
FIELD_QUANTITIES: dict[str, str] = {
    "rho": "density",
    "density": "density",
    "spin_density": "density",
    "n_spin": "density",
    "f": "source",
    "q": "source",
    "s": "source",
    "source": "source",
    "x": "intensity",
    "img": "intensity",
    "image": "intensity",
    "intensity": "intensity",
    "mu": "attenuation",
    "mu_a": "absorption",
    "attenuation": "attenuation",
    "c": "concentration",
    "conc": "concentration",
    "concentration": "concentration",
    "alpha": "diffusivity",
    "d": "diffusivity",
    "diff": "diffusivity",
    "diffusivity": "diffusivity",
    "kappa": "conductivity",
    "sigma": "conductivity",
    "gamma": "conductivity",
    "cond": "conductivity",
    "conductivity": "conductivity",
    "k": "permeability",
    "perm": "permeability",
    "permeability": "permeability",
    "log_k": "permeability",
    "logk": "permeability",
    "v": "velocity",
    "vp": "velocity",
    "vs": "velocity",
    "vel": "velocity",
    "velocity": "velocity",
    "n": "refractive_index",
    "ri": "refractive_index",
    "eps": "permittivity",
    "epsilon": "permittivity",
    "dn": "contrast",
    "delta_n": "contrast",
    "chi": "contrast",
    "contrast": "contrast",
    "scatterer": "contrast",
    "omega_l": "frequency",
    "omega": "frequency",
    "larmor": "frequency",
    "temperature": "temperature",
    "t": "temperature",
    "u": "signed",
    "p": "signed",
    "pressure": "signed",
    "j": "current",
    "jx": "current",
    "jy": "current",
    "jz": "current",
    "b": "field",
    "bz": "field",
    "phi": "phase",
    "phase": "phase",
    "theta": "phase",
    "error": "error",
    "residual": "residual",
    "mask": "mask",
    "sensitivity": "sensitivity",
}


@dataclass(frozen=True)
class CmapSpec:
    """A colormap choice: matplotlib name, kind and the quantity it was chosen for."""

    cmap: str
    kind: str = "sequential"  # sequential | diverging | cyclic
    quantity: str = "generic"

    @property
    def centered(self) -> bool:
        return self.kind == "diverging"


def quantity_of(name: str | None) -> str | None:
    """Physical quantity guessed from a field name (``"rho"`` → ``"density"``) or ``None``."""
    if not name:
        return None
    key = str(name).strip().lower()
    if key in QUANTITY_CMAPS:
        return key
    if key in FIELD_QUANTITIES:
        return FIELD_QUANTITIES[key]
    for suffix in ("_gt", "_true", "_pred", "_recon", "_est"):
        if key.endswith(suffix) and key[: -len(suffix)] in FIELD_QUANTITIES:
            return FIELD_QUANTITIES[key[: -len(suffix)]]
    for part in key.replace("-", "_").split("_"):
        if part in QUANTITY_CMAPS:
            return part
    return None


#: Signed-data rule (:func:`is_signed`): the minority sign must reach this share of the majority
#: sign's robust extreme — noise around a zero background (a few % of the peak) stays sequential.
SIGNED_FRACTION = 0.1


def is_signed(data: Any, min_fraction: float = SIGNED_FRACTION) -> bool:
    """True when ``data`` has a genuine negative *and* positive part (min < 0 < max beyond noise).

    The robust extremes (0.5 / 99.5 percentiles for ≥ 200 values, min / max otherwise) must have
    opposite signs and the smaller one must reach ``min_fraction`` of the larger one in magnitude.
    Noise dipping below a zero background (sinograms, counts, spectra: a few % of the peak) does
    not count; a magnetic field map with a 15 % negative lobe or a zero-mean phase does. Complex
    data are never "signed" (they are shown as magnitude and phase).
    """
    if data is None or np.iscomplexobj(data):
        return False
    a = np.asarray(data, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return False
    if a.size >= 200:
        lo, hi = (float(v) for v in np.percentile(a, [0.5, 99.5]))
    else:
        lo, hi = float(a.min()), float(a.max())
    if lo >= 0.0 or hi <= 0.0:
        return False
    return min(-lo, hi) >= float(min_fraction) * max(-lo, hi)


_is_signed = is_signed  # backward-compatible private alias

#: matplotlib colormaps treated as diverging (centred at 0) / cyclic when named in a hint.
DIVERGING_CMAPS = frozenset(
    {
        "RdBu",
        "RdBu_r",
        "coolwarm",
        "bwr",
        "seismic",
        "PuOr",
        "BrBG",
        "PiYG",
        "PRGn",
        "RdGy",
        "RdYlBu",
        "RdYlGn",
        "Spectral",
        "berlin",
        "managua",
        "vanimo",
    }
)
CYCLIC_CMAPS = frozenset({"twilight", "twilight_shifted", "hsv"})


def spec_from_hint(value: Any, data: Any = None, quantity: str | None = None) -> CmapSpec | None:
    """A colormap from a display hint (``measurement_cmap`` / ``field_cmap``).

    Accepted values: ``None`` / ``"auto"`` (→ ``None``: use the default choice); ``"signed"`` /
    ``"diverging"`` (``RdBu_r`` centred at 0); ``"sequential"`` (``viridis``); ``"cyclic"``
    (``twilight`` on [-π, π]); a quantity of :data:`QUANTITY_CMAPS` (``"density"`` → ``magma``,
    data-aware); a :class:`CmapSpec`; or any matplotlib colormap name (diverging / cyclic kind
    recognized from :data:`DIVERGING_CMAPS` / :data:`CYCLIC_CMAPS`, ``_r`` suffixes included).
    """
    if value is None:
        return None
    if isinstance(value, CmapSpec):
        return value
    v = str(value).strip()
    key = v.lower()
    if key in ("", "auto", "default"):
        return None
    q = quantity or "generic"
    if key in ("signed", "diverging", "centered", "centred"):
        return CmapSpec("RdBu_r", "diverging", q if q != "generic" else "signed")
    if key == "sequential":
        return CmapSpec("viridis", "sequential", q)
    if key == "cyclic":
        return CmapSpec("twilight", "cyclic", q)
    if key in QUANTITY_CMAPS:
        cm, kind = QUANTITY_CMAPS[key]
        return CmapSpec(cm, kind, key)
    base = v[:-2] if v.endswith("_r") else v
    kind = "diverging" if base in DIVERGING_CMAPS else "cyclic" if base in CYCLIC_CMAPS else None
    return CmapSpec(v, kind or "sequential", q)


def cmap_for(name: str | None = None, data: Any = None, quantity: str | None = None) -> CmapSpec:
    """Choose the colormap for a field.

    Priority: explicit ``quantity`` → quantity guessed from the field ``name`` → data-driven
    fallback (``RdBu_r`` centred at 0 if the data are signed — :func:`is_signed`: min < 0 < max
    beyond noise — else ``viridis``). A sequential quantity whose data turn out signed also
    switches to the centred diverging map.

    Args:
        name: field name (``"rho"``, ``"alpha"``, ``"phase"``, ...).
        data: array used for the data-driven fallback.
        quantity: a key of :data:`QUANTITY_CMAPS` (``"density"``, ``"error"``, ...).

    Returns:
        A :class:`CmapSpec`.
    """
    q = quantity or quantity_of(name)
    if q is not None and q in QUANTITY_CMAPS:
        cm, kind = QUANTITY_CMAPS[q]
        if kind == "cyclic" and data is not None:
            a = np.asarray(data, dtype=float)
            a = a[np.isfinite(a)]
            if a.size and float(a.max() - a.min()) < math.pi:
                # an unwrapped phase with a small range: a full-circle map would waste contrast
                return CmapSpec("RdBu_r", "diverging", q)
        if kind == "sequential" and data is not None and is_signed(data):
            # a "positive" quantity that turned out signed (e.g. a contrast) → diverging
            return CmapSpec("RdBu_r", "diverging", q)
        return CmapSpec(cm, kind, q)
    if data is not None and is_signed(data):
        return CmapSpec("RdBu_r", "diverging", "signed")
    return CmapSpec("viridis", "sequential", q or "generic")


def get_cmap(spec: CmapSpec | str, dark: bool | None = None):
    """matplotlib colormap for a spec, with the theme's color for masked (NaN) pixels."""
    import matplotlib

    name = spec.cmap if isinstance(spec, CmapSpec) else str(spec)
    return matplotlib.colormaps[name].with_extremes(bad=theme(dark).bad)


def color_limits(
    arrays: Sequence[Any] | Any,
    spec: CmapSpec | None = None,
    robust: float | None = None,
) -> tuple[float, float]:
    """Shared ``(vmin, vmax)`` of one or several arrays for a colormap spec.

    Diverging specs are symmetric around zero; cyclic specs span ``[-π, π]``. ``robust`` (e.g.
    99.5) uses percentiles instead of the extremes.
    """
    if spec is not None and spec.kind == "cyclic":
        return -math.pi, math.pi
    arrs = arrays if isinstance(arrays, list | tuple) else [arrays]
    vals = [np.asarray(a, dtype=float).ravel() for a in arrs if a is not None]
    vals = [v[np.isfinite(v)] for v in vals]
    vals = [v for v in vals if v.size]
    if not vals:
        return 0.0, 1.0
    allv = np.concatenate(vals)
    if spec is not None and spec.kind == "diverging":
        m = float(np.percentile(np.abs(allv), robust)) if robust else float(np.abs(allv).max())
        m = m if m > 0 else 1.0
        return -m, m
    if robust:
        lo, hi = float(np.percentile(allv, 100 - robust)), float(np.percentile(allv, robust))
    else:
        lo, hi = float(allv.min()), float(allv.max())
    if hi <= lo:
        pad = abs(hi) * 0.05 or 1.0
        lo, hi = lo - pad, hi + pad
    return lo, hi


# ---------------------------------------------------------------------------------------------
# figure sizes and creation
# ---------------------------------------------------------------------------------------------
def panel_size(context: str | None = None) -> tuple[float, float]:
    """Default (width, height) of one image panel in inches."""
    return PANEL_SIZES[context or current_context()]


def figsize(
    ncols: int = 1,
    nrows: int = 1,
    *,
    panel: tuple[float, float] | None = None,
    cbar: bool | int = False,
    title: bool = False,
    context: str | None = None,
    max_width: float | None = None,
) -> tuple[float, float]:
    """Figure size (inches) for a grid of ``nrows × ncols`` panels.

    Panels are slightly taller than wide so square images keep room for a two-line title.

    Args:
        ncols, nrows: panel grid.
        panel: (width, height) of one panel (default: :func:`panel_size`).
        cbar: number of colorbars placed side by side (``True`` = 1).
        title: reserve room for a figure title.
        context: size context (default: active).
        max_width: clamp the width (the height is scaled by the same factor).
    """
    w, h = panel or panel_size(context)
    n_cb = int(cbar) if not isinstance(cbar, bool) else (1 if cbar else 0)
    W = ncols * w + n_cb * CBAR_WIDTH + 0.15
    H = nrows * h + (0.35 if title else 0.0) + 0.1
    if max_width is not None and W > max_width:
        s = max_width / W
        W, H = max_width, H * s
    return (round(W, 3), round(H, 3))


_figsize = figsize  # module-level alias (``new_figure`` has a ``figsize`` parameter)


def paper_figsize(columns: int = 1, aspect: float = 0.75) -> tuple[float, float]:
    """Journal figure size: one or two columns wide, height = ``aspect × width``."""
    w = COLUMN_WIDTH if columns == 1 else TEXT_WIDTH
    return (w, round(w * aspect, 3))


def new_figure(
    nrows: int = 1,
    ncols: int = 1,
    *,
    figsize: tuple[float, float] | None = None,
    squeeze: bool = False,
    layout: str | None = "constrained",
    **subplots_kw: Any,
) -> tuple[Figure, Any]:
    """A pyplot-free figure (safe on headless servers) and its axes grid.

    Returns ``(fig, axes)`` with ``axes`` a 2-D array unless ``squeeze=True``.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize or _figsize(ncols, nrows), layout=layout)
    if nrows == 0 or ncols == 0:
        return fig, np.empty((0, 0), dtype=object)
    axes = fig.subplots(nrows, ncols, squeeze=squeeze, **subplots_kw)
    return fig, axes


def blank_axes(ax: Axes, frame: bool = True) -> None:
    """Hide ticks and tick labels of an image panel; keep a hairline frame (``frame=True``) so
    panels whose values sit at the surface color (diverging midpoint) keep visible edges."""
    t = theme()
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(frame)
        if frame:
            s.set_color(t.axis)
            s.set_linewidth(0.5)


def grid_on(ax: Axes, axis: str = "y") -> None:
    """Hairline solid gridlines on ``axis`` ("x", "y" or "both"), below the data."""
    t = theme()
    ax.grid(True, axis=axis, color=t.grid, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


def plain_log_ticks(ax: Axes, axis: str = "both") -> None:
    """Readable ticks on log axes spanning fewer than two decades: 1-2-5 steps printed as plain
    numbers (``20``, ``50``) instead of ``2×10¹`` or no label at all."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    fmt = FuncFormatter(lambda v, _pos: f"{v:g}")
    for which in ("x", "y"):
        if axis not in (which, "both"):
            continue
        a = ax.xaxis if which == "x" else ax.yaxis
        lo, hi = ax.get_xlim() if which == "x" else ax.get_ylim()
        lo, hi = min(lo, hi), max(lo, hi)
        if lo > 0 and hi / lo < 100:
            a.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            a.set_major_formatter(fmt)
            a.set_minor_formatter(NullFormatter())


def note(ax: Axes, text: str, loc: str = "upper left", fontsize: float | None = None) -> None:
    """Small secondary-ink annotation inside an axes."""
    t = theme()
    pos = {
        "upper left": (0.02, 0.97, "left", "top"),
        "upper right": (0.98, 0.97, "right", "top"),
        "lower left": (0.02, 0.03, "left", "bottom"),
        "lower right": (0.98, 0.03, "right", "bottom"),
        "center": (0.5, 0.5, "center", "center"),
    }[loc]
    ax.text(
        pos[0],
        pos[1],
        text,
        transform=ax.transAxes,
        ha=pos[2],
        va=pos[3],
        color=t.ink2,
        fontsize=fontsize,
    )


def message_axes(ax: Axes, text: str, title: str | None = None, error: bool = False) -> None:
    """Replace an axes by a centered text message (missing data, failed run)."""
    t = theme()
    blank_axes(ax)
    ax.set_facecolor(t.surface)
    if error:
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(t.axis)
            s.set_linewidth(0.6)
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=t.ink2,
        wrap=True,
        fontsize=max(5.5, _FONT_SIZES[current_context()] - 1.5),
    )
    if title:
        ax.set_title(title)


# ---------------------------------------------------------------------------------------------
# saving
# ---------------------------------------------------------------------------------------------
_FORMATS = ("png", "svg", "pdf", "jpg", "jpeg", "webp")


def _metadata(fmt: str) -> dict[str, Any] | None:
    # deterministic files: no timestamps
    if fmt == "png":
        return {"Software": "nefi.viz"}
    if fmt == "svg":
        return {"Date": None, "Creator": "nefi.viz"}
    if fmt == "pdf":
        return {"CreationDate": None, "ModDate": None, "Creator": "nefi.viz"}
    return None


def _quantize_png(path: Path) -> None:
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB")
    q = rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q.save(path, format="PNG", optimize=True)


def _render_png(fig: Figure, path: Path, dpi: float) -> None:
    fig.savefig(path, dpi=dpi, format="png", metadata=_metadata("png"))


def savefig(
    fig: Figure,
    path: str | Path,
    formats: Sequence[str] = ("png",),
    dpi: float = 130,
    *,
    close: bool = True,
    max_kb: float | None = 300,
    min_dpi: float = 55,
) -> list[Path]:
    """Save a figure as PNG and optionally SVG / PDF (parent directories are created).

    Args:
        fig: a matplotlib figure.
        path: output path; its suffix (if any) is added to ``formats``.
        formats: file formats to write, e.g. ``("png", "svg")``.
        dpi: raster resolution (≤ 150 keeps repository-friendly sizes).
        close: release the figure afterwards (``pyplot`` figures are closed; OO figures cleared
            of their canvas callbacks — the object stays usable).
        max_kb: size budget for PNGs; larger files are palette-quantized (256 colors), then
            re-rendered at decreasing dpi down to ``min_dpi``. ``None`` disables the check.
        min_dpi: lowest dpi tried when shrinking.

    Returns:
        The written paths (PNG first).
    """
    path = Path(path)
    suffix = path.suffix.lower().lstrip(".")
    stem = path.with_suffix("") if suffix in _FORMATS else path
    fmts = ([suffix] if suffix in _FORMATS else []) + [f.lower().lstrip(".") for f in formats]
    fmts = list(dict.fromkeys(fmts)) or ["png"]
    stem.parent.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for fmt in fmts:
        if fmt not in _FORMATS:
            raise ValueError(f"unsupported figure format {fmt!r}; use one of {_FORMATS}")
        p = stem.with_name(stem.name + "." + fmt)
        if fmt == "png":
            cur = float(dpi)
            _render_png(fig, p, cur)
            if max_kb is not None:
                attempts = 0
                while p.stat().st_size > max_kb * 1024 and attempts < 8:
                    _quantize_png(p)
                    if p.stat().st_size <= max_kb * 1024 or cur <= min_dpi:
                        break
                    cur = max(min_dpi, cur * 0.8)
                    _render_png(fig, p, cur)
                    attempts += 1
                if p.stat().st_size > max_kb * 1024:
                    log.warning(
                        "%s is %.0f kB (> %s kB budget) even at %d dpi",
                        p,
                        p.stat().st_size / 1024,
                        max_kb,
                        cur,
                    )
        else:
            fig.savefig(p, dpi=dpi, format=fmt, metadata=_metadata(fmt))
        out.append(p)
    out.sort(key=lambda q: q.suffix != ".png")
    if close:
        close_figure(fig)
    return out


def close_figure(fig: Figure) -> None:
    """Close a figure if it is managed by pyplot (never imports pyplot itself)."""
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        with contextlib.suppress(Exception):
            plt.close(fig)


def figure_to_png_bytes(fig: Figure, dpi: float = 110) -> bytes:
    """Render a figure to PNG bytes (for embedding)."""
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, format="png", metadata=_metadata("png"))
    return buf.getvalue()


def format_metric(name: str, value: Any, precision: int = 3) -> str:
    """Compact ``name value`` label for panel titles (``PSNR 24.1 dB``, ``SSIM 0.912``)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{name} {value}"
    key = name.lower()
    label = METRIC_LABELS.get(key, name)
    if not math.isfinite(v):
        return f"{label} {'∞' if v > 0 else '—'}"
    unit = " dB" if "psnr" in key else (" s" if key == "time_s" else "")
    if "psnr" in key and abs(v) < 1000:
        txt = f"{v:.1f}"
    elif key == "time_s" and abs(v) < 1e5:
        txt = f"{v:.1f}" if abs(v) >= 1 else f"{v:.2g}"
    elif abs(v) >= 1000 or (abs(v) < 1e-3 and v != 0):
        txt = f"{v:.2e}"
    else:  # keep trailing zeros: "1.00", "0.850", "12.3"
        txt = f"{v:#.{precision}g}".rstrip(".")
    return f"{label} {txt}{unit}"


#: Display names of common metrics.
METRIC_LABELS: dict[str, str] = {
    "psnr": "PSNR",
    "ssim": "SSIM",
    "masked_ssim": "mSSIM",
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "relative_error": "rel.err",
    "rel_err": "rel.err",
    "gmsd": "GMSD",
    "hungarian_f1": "HF1",
    "swd": "SWD",
    "iou": "IoU",
    "dice": "Dice",
    "edge_f1": "edge-F1",
    "larmor_mae": "ω_L MAE",
    "time_s": "time",
    "ms_per_step": "ms/step",
}

#: Preference order for the one headline metric of a tile.
PRIMARY_METRICS: tuple[str, ...] = (
    "psnr",
    "ssim",
    "hungarian_f1",
    "relative_error",
    "gmsd",
    "iou",
    "mse",
)


def primary_metric(metrics: Mapping[str, Any] | None) -> tuple[str, float] | None:
    """The headline ``(name, value)`` of a metrics dict (PSNR > SSIM > HF1 > rel.err > ...)."""
    if not metrics:
        return None
    for k in PRIMARY_METRICS:
        if k in metrics and metrics[k] is not None:
            try:
                return k, float(metrics[k])
            except (TypeError, ValueError):
                continue
    for k, v in metrics.items():
        try:
            return k, float(v)
        except (TypeError, ValueError):
            continue
    return None


__all__ = [
    "CBAR_WIDTH",
    "COLUMN_WIDTH",
    "CYCLIC_CMAPS",
    "DARK",
    "DIVERGING_CMAPS",
    "FIELD_QUANTITIES",
    "LIGHT",
    "METRIC_LABELS",
    "PALETTE",
    "PALETTE_DARK",
    "PALETTE_LIGHT",
    "PANEL_SIZES",
    "PRIMARY_METRICS",
    "QUANTITY_CMAPS",
    "SIGNED_FRACTION",
    "TEXT_WIDTH",
    "CmapSpec",
    "Theme",
    "blank_axes",
    "close_figure",
    "cmap_for",
    "color_limits",
    "current_context",
    "figsize",
    "figure_to_png_bytes",
    "format_metric",
    "get_cmap",
    "grid_on",
    "is_signed",
    "message_axes",
    "new_figure",
    "note",
    "palette",
    "panel_size",
    "plain_log_ticks",
    "paper_figsize",
    "primary_metric",
    "quantity_of",
    "savefig",
    "series_colors",
    "set_default_style",
    "spec_from_hint",
    "style_rc",
    "styled",
    "theme",
    "use_style",
]
