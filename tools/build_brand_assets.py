"""Draw the nefi mark, wordmarks, favicon, social card and the recipe diagram.

Usage::

    python tools/build_site_assets.py          # first: docs/assets/hero.png (used by the card)
    python tools/build_brand_assets.py         # -> docs/assets/brand/, docs/assets/recipe*.svg,
                                               #    docs/overrides/.icons/nefi/mark.svg

The mark is two level sets of a field and its peak (eccentric contours, closer together on the
steep side), coloured by level with the magma colormap of the reconstruction figures; the
wordmark is a monoline "nefi" drawn with strokes, so the SVGs need no font. One geometry feeds
every output: the colour and monochrome SVGs, the favicon and the PNG renders (matplotlib, for
the touch icon and the social card). The recipe diagram is written in a light and a dark variant
(``#only-light`` / ``#only-dark`` on the site, ``<picture>`` in the README).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / "docs" / "assets" / "brand"
ICONS = ROOT / "docs" / "overrides" / ".icons" / "nefi"

INK = "#14132b"
PAPER = "#f6f5fb"
# magma, sampled for light backgrounds (darker) and dark backgrounds (brighter)
LEVELS_LIGHT = ["#51127c", "#b73779", "#f7803c"]
LEVELS_DARK = ["#a26bf0", "#e8578f", "#fdae61"]

# ---------------------------------------------------------------------------------------------
# geometry (64 x 64 box for the mark; the wordmark glyphs share its baseline grid)
# ---------------------------------------------------------------------------------------------
MARK_STROKE = 4.0
_U = (1 / math.sqrt(2), -1 / math.sqrt(2))  # towards the peak (upper right)
_C1 = (30.5, 33.5)


def _shift(c: tuple[float, float], d: float) -> tuple[float, float]:
    return (c[0] + d * _U[0], c[1] + d * _U[1])


# (cx, cy, r): two level sets, closer together on the steep (upper right) side, and the peak
CONTOURS = [(*_C1, 24.0), (*_shift(_C1, 5.0), 13.0)]
PEAK = (*_shift(_C1, 7.0), 5.0)  # filled disc

TEXT_STROKE = 5.6
BASE, XH = 50.0, 25.0  # baseline and x-height (y grows downwards)
XTOP = BASE - XH


def glyph_paths(x0: float) -> tuple[list[str], tuple[float, float, float], float]:
    """SVG path data of the monoline "nefi" starting at ``x0``, the i-dot and the right edge."""
    paths = []
    r = 10.0  # n: stem + arch (the arch top touches the x-height)
    n0 = x0
    paths.append(f"M{n0:.2f} {BASE:.2f}V{XTOP:.2f}")
    paths.append(
        f"M{n0:.2f} {XTOP + r:.2f}A{r} {r} 0 0 1 {n0 + 2 * r:.2f} {XTOP + r:.2f}V{BASE:.2f}"
    )
    re_ = XH / 2  # e: bar + open circle
    ecx, ecy = n0 + 2 * r + 9.5 + re_, BASE - re_
    a = math.radians(42)
    ex, ey = ecx + re_ * math.cos(a), ecy + re_ * math.sin(a)
    paths.append(
        f"M{ecx - re_:.2f} {ecy:.2f}H{ecx + re_:.2f}A{re_:.2f} {re_:.2f} 0 1 0 {ex:.2f} {ey:.2f}"
    )
    fx = ecx + re_ + 7.5 + 7.0  # f: stem with a hook, crossbar at the x-height
    fr, ftop = 8.0, 10.5
    paths.append(
        f"M{fx:.2f} {BASE:.2f}V{ftop + fr:.2f}A{fr} {fr} 0 0 1 {fx + fr:.2f} {ftop:.2f}"
        f"H{fx + fr + 3.0:.2f}"
    )
    paths.append(f"M{fx - 7.0:.2f} {XTOP + 1.0:.2f}H{fx + 8.5:.2f}")
    ix = fx + fr + 3.0 + 8.0  # i: stem + dot
    paths.append(f"M{ix:.2f} {BASE:.2f}V{XTOP:.2f}")
    dot = (ix, 12.6, TEXT_STROKE * 0.62)
    return paths, dot, ix + TEXT_STROKE / 2


# ---------------------------------------------------------------------------------------------
# SVG writers
# ---------------------------------------------------------------------------------------------
def mark_svg_body(colors: list[str] | None, dx: float = 0.0, dy: float = 0.0) -> str:
    """Contours + peak; ``colors=None`` draws everything in ``currentColor``."""
    out = []
    for k, (cx, cy, r) in enumerate(CONTOURS):
        col = "currentColor" if colors is None else colors[k]
        out.append(
            f'<circle cx="{cx + dx:.2f}" cy="{cy + dy:.2f}" r="{r:.2f}" fill="none" '
            f'stroke="{col}" stroke-width="{MARK_STROKE}"/>'
        )
    cx, cy, r = PEAK
    col = "currentColor" if colors is None else colors[2]
    out.append(f'<circle cx="{cx + dx:.2f}" cy="{cy + dy:.2f}" r="{r:.2f}" fill="{col}"/>')
    return "\n  ".join(out)


def svg(view: str, body: str, title: str, w: float | None = None, h: float | None = None) -> str:
    size = f' width="{w:g}" height="{h:g}"' if w and h else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view}"{size} role="img" '
        f'aria-label="{title}">\n  <title>{title}</title>\n  {body}\n</svg>\n'
    )


def wordmark_svg(levels: list[str] | None, ink: str) -> str:
    x0 = 64 + 16.0
    paths, (ix, iy, ir), right = glyph_paths(x0)
    text = "\n  ".join(f'<path d="{d}"/>' for d in paths)
    body = (
        mark_svg_body(levels) + f'\n  <g fill="none" stroke="{ink}" stroke-width="{TEXT_STROKE}" '
        'stroke-linecap="round" stroke-linejoin="round">\n  '
        + text
        + f'\n  </g>\n  <circle cx="{ix:.2f}" cy="{iy:.2f}" r="{ir:.2f}" fill="{ink}"/>'
    )
    width = right + 4
    return svg(f"0 0 {width:.1f} 64", body, "nefi", w=round(width * 2), h=128)


def favicon_svg() -> str:
    pad = 6
    body = (
        f'<rect x="0" y="0" width="{64 + 2 * pad}" height="{64 + 2 * pad}" rx="16" fill="{INK}"/>'
        "\n  " + mark_svg_body(LEVELS_DARK, dx=pad, dy=pad)
    )
    return svg(f"0 0 {64 + 2 * pad} {64 + 2 * pad}", body, "nefi")


# ---------------------------------------------------------------------------------------------
# PNG renders (matplotlib)
# ---------------------------------------------------------------------------------------------
def _draw_mark(ax, levels: list[str], dx: float = 0.0, dy: float = 0.0, lw_scale: float = 1.0):
    from matplotlib.patches import Circle

    for k, (cx, cy, r) in enumerate(CONTOURS):
        ax.add_patch(
            Circle((cx + dx, cy + dy), r, fill=False, ec=levels[k], lw=MARK_STROKE * lw_scale)
        )
    cx, cy, r = PEAK
    ax.add_patch(Circle((cx + dx, cy + dy), r, fc=levels[2], ec="none"))


def touch_icon(path: Path, px: int = 180) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(1, 1), dpi=px)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 76)
    ax.set_ylim(76, 0)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch((0, 0), 76, 76, boxstyle="round,pad=0,rounding_size=16", fc=INK, ec="none")
    )
    _draw_mark(ax, LEVELS_DARK, dx=6, dy=6, lw_scale=px / 76 * 72 / px)
    fig.savefig(path, dpi=px, transparent=True)
    plt.close(fig)


def social_card(path: Path, hero: Path) -> None:
    """1280 x 640 preview image (link unfurls, GitHub social preview)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.image import imread

    W, H, dpi = 1280, 640, 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    bg = fig.add_axes((0, 0, 1, 1))
    bg.axis("off")
    yy, xx = np.mgrid[0 : 1 : H * 1j, 0 : 1 : W * 1j]
    top, bottom = np.array([20, 19, 43]) / 255, np.array([43, 22, 70]) / 255
    grad = top[None, None] * (1 - yy[..., None]) + bottom[None, None] * yy[..., None]
    glow = np.exp(-(((xx - 0.85) / 0.35) ** 2 + ((yy - 0.1) / 0.45) ** 2))[..., None]
    grad = np.clip(grad + 0.10 * glow * np.array([0.98, 0.45, 0.30]), 0, 1)
    bg.imshow(grad, extent=(0, W, H, 0), aspect="auto")
    bg.set_xlim(0, W)
    bg.set_ylim(H, 0)

    # mark + wordmark in px units: 1 geometry unit = s px
    s, ox, oy = 1.55, 72, 70
    ax = bg
    from matplotlib.patches import Circle

    for k, (cx, cy, r) in enumerate(CONTOURS):
        ax.add_patch(
            Circle((ox + s * cx, oy + s * cy), s * r, fill=False, ec=LEVELS_DARK[k], lw=4.4)
        )
    cx, cy, r = PEAK
    ax.add_patch(Circle((ox + s * cx, oy + s * cy), s * r, fc=LEVELS_DARK[2], ec="none"))
    for xs, ys in _glyph_polylines(64 + 16.0):
        ax.plot(
            [ox + s * x for x in xs],
            [oy + s * y for y in ys],
            color=PAPER,
            lw=6.2,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    _, (ix, iy, ir), _ = glyph_paths(64 + 16.0)
    ax.add_patch(Circle((ox + s * ix, oy + s * iy), s * ir, fc=PAPER, ec="none"))

    txt = {"family": "DejaVu Sans", "color": PAPER}
    ax.text(72, 250, "Neural-Field Inversion", fontsize=28, fontweight="bold", **txt)
    ax.text(
        72,
        305,
        "Recover hidden physical fields from a single\n"
        "measurement. A coordinate neural field is fitted\n"
        "through a differentiable model of the instrument,\n"
        "so no training data is needed.",
        fontsize=16,
        va="top",
        linespacing=1.45,
        family="DejaVu Sans",
        color="#cfcbe6",
    )
    ax.text(
        72,
        580,
        "CAB Lab · Princeton University   ·   github.com/ContinuumCoder/NEFI",
        fontsize=13,
        family="DejaVu Sans",
        color="#a39fc4",
    )
    img = imread(hero)
    h, w = img.shape[:2]
    box_w = 540
    box_h = box_w * h / w
    x1, y1 = W - 60 - box_w, (H - box_h) / 2
    ax.add_patch(
        matplotlib.patches.FancyBboxPatch(
            (x1 - 10, y1 - 10),
            box_w + 20,
            box_h + 20,
            boxstyle="round,pad=0,rounding_size=14",
            fc="#fcfcfb",
            ec="none",
        )
    )
    ax.imshow(img, extent=(x1, x1 + box_w, y1 + box_h, y1), interpolation="lanczos", zorder=3)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _glyph_polylines(x0: float) -> list[tuple[list[float], list[float]]]:
    """The wordmark strokes as polylines (arcs sampled) for matplotlib."""
    lines = []

    def arc(cx, cy, r, a0, a1, n=40):
        ts = [a0 + (a1 - a0) * k / (n - 1) for k in range(n)]
        return [cx + r * math.cos(t) for t in ts], [cy + r * math.sin(t) for t in ts]

    r = 10.0
    n0 = x0
    lines.append(([n0, n0], [BASE, XTOP]))
    xs, ys = arc(n0 + r, XTOP + r, r, math.pi, 2 * math.pi)
    lines.append((xs + [n0 + 2 * r], ys + [BASE]))
    re_ = XH / 2
    ecx, ecy = n0 + 2 * r + 9.5 + re_, BASE - re_
    xs, ys = arc(ecx, ecy, re_, 0.0, -(2 * math.pi - math.radians(42)), n=90)
    lines.append(([ecx - re_] + xs, [ecy] + ys))
    fx = ecx + re_ + 7.5 + 7.0
    fr, ftop = 8.0, 10.5
    xs, ys = arc(fx + fr, ftop + fr, fr, math.pi, 1.5 * math.pi)
    lines.append(([fx] + xs + [fx + fr + 3.0], [BASE] + ys + [ftop]))
    lines.append(([fx - 7.0, fx + 8.5], [XTOP + 1.0, XTOP + 1.0]))
    ix = fx + fr + 3.0 + 8.0
    lines.append(([ix, ix], [BASE, XTOP]))
    return lines


# ---------------------------------------------------------------------------------------------
# recipe diagram
# ---------------------------------------------------------------------------------------------
RECIPE_STEPS = [
    # (group, title, line 1, line 2, code)
    ("rep", "Coordinates", "r \u2208 [\u22121, 1]\u1d48", "any grid, any size", "Domain"),
    ("rep", "Encoding", "Fourier features", "annealed coarse \u2192 fine", "FourierFeatures"),
    (
        "rep",
        "Field + heads",
        "MLP \u00b7 grid \u00b7 hash \u00b7 \u2026",
        "heads = value priors",
        "nefi.fields",
    ),
    (
        "phys",
        "Operator",
        "differentiable F(x)",
        "PDE \u00b7 FFT \u00b7 wave \u00b7 \u2026",
        "nefi.operators",
    ),
    ("fit", "Losses", "fidelity to data y", "+ priors: TV, \u2113\u2081, \u2026", "nefi.losses"),
    ("fit", "Solver", "multiscale curriculum", "AdamW \u00b7 Morozov stop", "nefi.solve"),
    (
        "trust",
        "Diagnostics",
        "\u03c7 = RMSE / \u03c3",
        "sensitivity \u00b7 spectra",
        "nefi.diagnostics",
    ),
]
RECIPE_GROUPS = {
    # group: (label, first step, last step)
    "rep": ("REPRESENTATION \u00b7 the prior", 0, 2),
    "phys": ("PHYSICS \u00b7 hard constraint", 3, 3),
    "fit": ("FIT \u00b7 one measurement", 4, 5),
    "trust": ("TRUST", 6, 6),
}
RECIPE_THEMES = {
    "light": {
        "ink": "#14132b",
        "muted": "#5d5b72",
        "arrow": "#8a879c",
        "code": "#6b6880",
        "rep": ("#6d28d9", "#f4efff"),
        "phys": ("#c2410c", "#fff2e8"),
        "fit": ("#be185d", "#fdeff5"),
        "trust": ("#0f766e", "#e9f7f5"),
        "pill": ("#14132b", "#ffffff"),
    },
    "dark": {
        "ink": "#f1effa",
        "muted": "#b3afc9",
        "arrow": "#8f8ba8",
        "code": "#a9a4c4",
        "rep": ("#a78bfa", "#221a3d"),
        "phys": ("#fb923c", "#35201a"),
        "fit": ("#f472b6", "#341a2c"),
        "trust": ("#2dd4bf", "#122b2b"),
        "pill": ("#f1effa", "#1b1a33"),
    },
}


def recipe_svg(theme: str) -> str:
    T = RECIPE_THEMES[theme]
    W, H = 1280, 312
    bw, gap, x0, y0, bh = 166, 16, 11, 64, 124
    xs = [x0 + k * (bw + gap) for k in range(len(RECIPE_STEPS))]
    font = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
    mono = "'JetBrains Mono', 'SFMono-Regular', Menlo, Consolas, monospace"
    out = [
        '<defs><marker id="a" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
        f'fill="{T["arrow"]}"/></marker></defs>'
    ]
    # group labels with brackets
    for key, (label, a, b) in RECIPE_GROUPS.items():
        col = T[key][0]
        xa, xb = xs[a], xs[b] + bw
        out.append(
            f'<path d="M{xa + 2} 46 V40 H{xb - 2} V46" fill="none" stroke="{col}" '
            'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        out.append(
            f'<text x="{(xa + xb) / 2}" y="30" text-anchor="middle" fill="{col}" '
            f'font-size="12" font-weight="700" letter-spacing="0.08em">{label}</text>'
        )
    # boxes
    for k, (key, title, l1, l2, code) in enumerate(RECIPE_STEPS):
        stroke, fill = T[key]
        x = xs[k]
        out.append(
            f'<rect x="{x}" y="{y0}" width="{bw}" height="{bh}" rx="12" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.6"/>'
        )
        out.append(f'<circle cx="{x + 23}" cy="{y0 + 25}" r="11" fill="{stroke}"/>')
        out.append(
            f'<text x="{x + 23}" y="{y0 + 29.5}" text-anchor="middle" fill="{fill}" '
            f'font-size="12.5" font-weight="700">{k + 1}</text>'
        )
        out.append(
            f'<text x="{x + 42}" y="{y0 + 30}" fill="{T["ink"]}" font-size="15.5" '
            f'font-weight="700">{title}</text>'
        )
        out.append(
            f'<text x="{x + 13}" y="{y0 + 62}" fill="{T["ink"]}" font-size="12.5">{l1}</text>'
        )
        out.append(
            f'<text x="{x + 13}" y="{y0 + 81}" fill="{T["muted"]}" font-size="12.5">{l2}</text>'
        )
        out.append(
            f'<text x="{x + 13}" y="{y0 + 108}" fill="{T["code"]}" font-size="11.5" '
            f'font-family="{mono}">{code}</text>'
        )
        if k:
            xa = xs[k - 1] + bw + 2
            out.append(
                f'<path d="M{xa} {y0 + bh / 2} H{x - 2}" stroke="{T["arrow"]}" '
                'stroke-width="1.6" marker-end="url(#a)"/>'
            )
    bottom = y0 + bh
    # measurement y -> losses (between the boxes and the gradient loop)
    lx = xs[4] + bw / 2
    py = bottom + 38
    stroke, fill = T["pill"]
    out.append(
        f'<rect x="{lx - 74}" y="{py - 15}" width="148" height="30" rx="15" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.4"/>'
    )
    out.append(
        f'<text x="{lx}" y="{py + 4.5}" text-anchor="middle" fill="{T["ink"]}" font-size="13" '
        'font-weight="600">measurement y</text>'
    )
    out.append(
        f'<path d="M{lx} {py - 16} V{bottom + 4}" stroke="{T["arrow"]}" stroke-width="1.6" '
        'marker-end="url(#a)"/>'
    )
    # gradient: solver -> field parameters, below the measurement
    gx1, gx2 = xs[5] + bw / 2, xs[2] + bw / 2
    gy = bottom + 84
    col = T["fit"][0]
    out.append(
        f'<path d="M{gx1} {bottom + 3} V{gy - 12} Q{gx1} {gy} {gx1 - 12} {gy} H{gx2 + 12} '
        f'Q{gx2} {gy} {gx2} {gy - 12} V{bottom + 5}" fill="none" stroke="{col}" '
        'stroke-width="1.8" stroke-dasharray="6 4" marker-end="url(#a)"/>'
    )
    out.append(
        f'<text x="{(gx1 + gx2) / 2}" y="{gy + 22}" text-anchor="middle" fill="{col}" '
        'font-size="13" font-weight="600">\u2207\u03b8 \u2014 fitted to this one measurement, '
        "no labels, no training set</text>"
    )
    # what the diagnostics feed
    ox = xs[6] + bw / 2
    for j, line in enumerate(("benchmarks \u00b7 refinement", "figures \u00b7 3-D viewer")):
        out.append(
            f'<text x="{ox}" y="{bottom + 30 + 18 * j}" text-anchor="middle" '
            f'fill="{T["muted"]}" font-size="12.5">{line}</text>'
        )
    body = "\n  ".join(out)
    title = "nefi recipe: coordinates, encoding, field, operator, losses, solver, diagnostics"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" font-family="{font}" role="img" aria-label="{title}">\n'
        f"  <title>{title}</title>\n  {body}\n</svg>\n"
    )


def main() -> int:
    BRAND.mkdir(parents=True, exist_ok=True)
    ICONS.mkdir(parents=True, exist_ok=True)
    files = {
        BRAND / "nefi-mark.svg": svg("0 0 64 64", mark_svg_body(LEVELS_LIGHT), "nefi"),
        BRAND / "nefi-mark-dark.svg": svg("0 0 64 64", mark_svg_body(LEVELS_DARK), "nefi"),
        BRAND / "nefi-mark-mono.svg": svg("0 0 64 64", mark_svg_body(None), "nefi"),
        BRAND / "nefi-wordmark.svg": wordmark_svg(LEVELS_LIGHT, INK),
        BRAND / "nefi-wordmark-dark.svg": wordmark_svg(LEVELS_DARK, PAPER),
        BRAND / "nefi-wordmark-mono.svg": wordmark_svg(None, "currentColor"),
        BRAND / "favicon.svg": favicon_svg(),
        ICONS / "mark.svg": svg("0 0 64 64", mark_svg_body(None), "nefi"),
        ROOT / "docs" / "assets" / "recipe.svg": recipe_svg("light"),
        ROOT / "docs" / "assets" / "recipe-dark.svg": recipe_svg("dark"),
    }
    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
        print(f"  {path.relative_to(ROOT)}  {len(text.encode()) / 1000:.1f} kB")
    touch_icon(BRAND / "apple-touch-icon.png")
    social_card(BRAND / "social-card.png", ROOT / "docs" / "assets" / "hero.png")
    for p in (BRAND / "apple-touch-icon.png", BRAND / "social-card.png"):
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size / 1000:.1f} kB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
