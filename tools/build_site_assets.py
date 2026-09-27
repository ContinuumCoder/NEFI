"""Build the images and interactive viewers of the documentation site from gallery outputs.

Usage::

    python examples/gallery.py --budget 0.2 --out runs/gallery --html     # gallery, CPU ~3 min
    python examples/refine_edges.py --budget 0.1 --include-2d --seeds 0,1  # refinement figures
    python tools/build_site_assets.py                                      # -> docs/assets/

Reads the figures that ``examples/gallery.py`` writes under ``runs/gallery/`` (overview figures,
per-system comparisons and animations, baselines, performance dashboards, the 3-D viewer data),
the edge-refinement figures under ``runs/refine_edges/`` and a few single figures, and writes
web-sized copies under ``docs/assets/``: RGBA figures are flattened onto the figure background,
downscaled to a maximum width and palette-quantized so that every file stays below ``--max-kb``
(300 kB by default). Two composites are drawn from the gallery tiles (the landing-page hero and
the reconstruction thumbnails), and the standalone interactive 3-D viewers are re-exported with
:func:`nefi.viz.save_volume_viewer` from the gallery's ``viewer.json`` payloads, with a small
script that follows the light / dark scheme of the page that embeds them.

Nothing is computed here: every number shown in the figures comes from the gallery run itself.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BG = (252, 252, 251)  # background of every nefi figure (#fcfcfb)
INK = (20, 19, 43)
MUTED = (110, 108, 125)

TWO_D = [
    "toy1d",
    "nv_relaxometry",
    "deconvolution",
    "sparse_view_ct",
    "poisson_source",
    "eit",
    "darcy_flow",
    "current_density",
    "wave_fwi",
    "diffraction_tomography",
    "holography",
    "reaction_diffusion",
]
THREE_D = ["thermal_tomography", "ct3d", "deconvolution3d", "dot3d", "photoacoustic3d"]
EVOLUTION = ["ct3d", "wave_fwi", "current_density", "nv_relaxometry"]
REFINE = [
    "dot3d_seed0",
    "photoacoustic3d_seed0",
    "ct3d_seed0",
    "deconvolution3d_seed0",
    "thermal_tomography_seed0",
    "eit_seed0",
    "deconvolution_seed0",
]
PERF = ["step_time", "wallclock", "speedup", "scaling", "memory", "bench_psnr"]
# hero: measurement -> reconstruction pairs (name, caption)
HERO = [
    ("nv_relaxometry", "NV noise spectra → spin sources"),
    ("sparse_view_ct", "16-view sinogram → attenuation"),
    ("current_density", "NV stray field → sheet current"),
    ("wave_fwi", "acoustic traces → sound speed"),
]
THUMBS = [
    "nv_relaxometry",
    "deconvolution",
    "sparse_view_ct",
    "poisson_source",
    "eit",
    "darcy_flow",
    "current_density",
    "wave_fwi",
    "diffraction_tomography",
    "holography",
    "reaction_diffusion",
]
VIEWERS = ["ct3d", "photoacoustic3d", "dot3d"]


# ---------------------------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------------------------
def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    """DejaVu (bundled with matplotlib) so the output does not depend on system fonts."""
    import matplotlib

    name = "DejaVuSansMono" if mono else "DejaVuSans"
    name += "-Bold" if bold else ""
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / f"{name}.ttf"
    return ImageFont.truetype(str(path), size)


def flatten(im: Image.Image, bg: tuple[int, int, int] = BG) -> Image.Image:
    """RGB image with any transparency composited onto ``bg``."""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        out = Image.new("RGB", im.size, bg)
        out.paste(im, mask=im.getchannel("A"))
        return out
    return im.convert("RGB")


def fit_width(im: Image.Image, max_w: int | None) -> Image.Image:
    if max_w is None or im.width <= max_w:
        return im
    h = round(im.height * max_w / im.width)
    return im.resize((max_w, h), Image.Resampling.LANCZOS)


def save_png(im: Image.Image, path: Path, *, max_bytes: int, max_w: int | None = None) -> int:
    """Palette PNG of at most ``max_bytes``.

    Palette sources (the gallery's own quantized overviews) that fit are saved losslessly; other
    figures are flattened, quantized to 256 colours and downscaled only as far as needed (a
    matplotlib figure has few distinct colours, so 256 are visually lossless; fewer colours are
    a last resort because they posterize shaded surfaces).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if im.mode == "P" and (max_w is None or im.width <= max_w):
        im.save(path, optimize=True)
        if path.stat().st_size <= max_bytes:
            return path.stat().st_size
    rgb = fit_width(flatten(im), max_w)
    width = rgb.width
    while True:
        cur = fit_width(rgb, width)
        for colors in (256, 192, 128):
            if colors < 192 and width > 0.6 * rgb.width:
                break  # prefer a smaller image over a coarser palette
            q = cur.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
            q.save(path, optimize=True)
            if path.stat().st_size <= max_bytes:
                return path.stat().st_size
        width = int(width * 0.92)


def round_corners(path: Path, radius: int) -> int:
    """Make the corners of a saved palette PNG transparent (rounded cards in the README, where
    no CSS can round them); returns the new size."""
    from PIL import ImageDraw as _Draw

    im = Image.open(path)
    if im.mode != "P":
        im = im.convert("RGB").quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    pal = im.getpalette()[: 3 * 255]
    idx = np.asarray(im).copy()
    if idx.max() > 254:  # free index 255 for transparency
        im = im.convert("RGB").quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        pal = im.getpalette()[: 3 * 255]
        idx = np.asarray(im).copy()
    mask = Image.new("L", im.size, 0)
    _Draw.Draw(mask).rounded_rectangle((0, 0, im.width - 1, im.height - 1), radius, fill=255)
    idx[np.asarray(mask) == 0] = 255
    pal = pal + [0, 0, 0] * (255 - len(pal) // 3)  # pad: index 255 is the transparent entry
    out = Image.fromarray(idx.astype(np.uint8), mode="P")
    out.putpalette(pal + [255, 255, 255])
    out.save(path, transparency=255)  # no optimize: it would compact the palette and drop tRNS
    return path.stat().st_size


def copy_gif(src: Path, dst: Path, *, max_bytes: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    size = dst.stat().st_size
    if size > max_bytes:
        raise SystemExit(f"{src} has {size} bytes > {max_bytes}; re-render it smaller")
    return size


def blank_rows(im: Image.Image) -> list[tuple[int, int]]:
    """(start, stop) of the horizontal bands that contain anything but background."""
    a = np.asarray(flatten(im)).astype(int)
    busy = (np.abs(a - np.array(BG)).sum(-1) > 30).any(axis=1)
    blocks, inside, start = [], False, 0
    for y, b in enumerate(busy):
        if b and not inside:
            inside, start = True, y
        elif not b and inside:
            inside = False
            blocks.append((start, y))
    if inside:
        blocks.append((start, len(busy)))
    return blocks


def drop_title(im: Image.Image, pad: int = 6) -> Image.Image:
    """Crop the figure title (the first band of content); the page caption replaces it."""
    blocks = blank_rows(im)
    return im.crop((0, max(blocks[1][0] - pad, 0), im.width, im.height))


def panels(tile: Image.Image, min_side: int = 150) -> list[tuple[int, int, int, int]]:
    """Bounding boxes (x0, y0, x1, y1) of the image panels of a gallery tile, left to right."""
    from scipy import ndimage

    a = np.asarray(flatten(tile)).astype(int)
    mask = np.abs(a - np.array(BG)).sum(-1) > 30
    lab, _ = ndimage.label(mask)
    boxes = []
    for sl in ndimage.find_objects(lab):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h >= min_side and w >= min_side:
            boxes.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
    return sorted(boxes)


def square(im: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Crop ``box`` to its central square (the image area of a panel)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    s = min(w, h)
    x0 += (w - s) // 2
    y0 += (h - s) // 2
    return im.crop((x0, y0, x0 + s, y0 + s))


def text_w(draw: ImageDraw.ImageDraw, s: str, f: ImageFont.FreeTypeFont) -> int:
    x0, _, x1, _ = draw.textbbox((0, 0), s, font=f)
    return x1 - x0


# ---------------------------------------------------------------------------------------------
# composites
# ---------------------------------------------------------------------------------------------
def hero(gallery: Path, out: Path, *, max_bytes: int) -> int:
    """2 x 2 grid of measurement -> reconstruction pairs, pixel-doubled for high-DPI screens."""
    z = 2  # nearest-neighbour upscaling: the fields are 24^2-32^2 grids drawn as pixels
    side, gap, arrow, pad, lab = 173 * z, 16 * z, 30 * z, 14 * z, 50 * z
    cell_w = 2 * side + arrow
    W = 2 * cell_w + 3 * pad + gap
    H = 2 * (side + lab) + 3 * pad
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    f_name, f_cap = font(13 * z, bold=True, mono=True), font(12 * z)
    for k, (name, caption) in enumerate(HERO):
        tile = flatten(Image.open(gallery / "physics" / name / "tile.png"))
        boxes = panels(tile)
        meas, rec = square(tile, boxes[1]), square(tile, boxes[-1])
        r, c = divmod(k, 2)
        x = pad + c * (cell_w + pad + gap)
        y = pad + r * (side + lab + pad)
        for j, p in enumerate((meas, rec)):
            p = p.resize((side, side), Image.Resampling.NEAREST)
            canvas.paste(p, (x + j * (side + arrow), y))
        ya, xa = y + side // 2, x + side + 7 * z
        d.line([(xa, ya), (xa + arrow - 16 * z, ya)], fill=MUTED, width=2 * z)
        tip = xa + arrow - 14 * z
        d.polygon([(tip, ya - 5 * z), (tip + 8 * z, ya), (tip, ya + 5 * z)], fill=MUTED)
        d.text((x, y + side + 9 * z), name, font=f_name, fill=INK)
        d.text((x, y + side + 27 * z), caption, font=f_cap, fill=MUTED)
    return save_png(canvas, out, max_bytes=max_bytes)


def thumbnails(gallery: Path, out_dir: Path, *, max_bytes: int) -> dict[str, int]:
    """Reconstruction panel of every 2-D gallery tile, pixel-doubled."""
    sizes = {}
    for name in THUMBS:
        tile = flatten(Image.open(gallery / "physics" / name / "tile.png"))
        rec = square(tile, panels(tile)[-1]).resize((346, 346), Image.Resampling.NEAREST)
        sizes[f"gallery/thumbs/{name}.png"] = save_png(
            rec, out_dir / f"{name}.png", max_bytes=max_bytes
        )
    return sizes


def crop_2d_overview(src: Path, out: Path, *, max_bytes: int) -> int:
    """The 2-D rows of the gallery overview (everything above the first volumetric system)."""
    im = Image.open(src)
    # blocks[0] is the figure title; merge the rest when separated by small gaps (the titles and
    # panels of one row of tiles) and keep the first three rows
    rows: list[list[int]] = []
    for s, e in blank_rows(im)[1:]:
        if rows and s - rows[-1][1] < 18:
            rows[-1][1] = e
        else:
            rows.append([s, e])
    y0, y1 = rows[0][0] - 10, rows[2][1] + 12
    return save_png(im.crop((0, y0, im.width, y1)), out, max_bytes=max_bytes)


def crop_viewer(src: Path, out: Path, *, max_bytes: int) -> int:
    """The three panels and the controls of the ct3d viewer screenshot."""
    im = flatten(Image.open(src), bg=(249, 249, 247))
    return save_png(im.crop((118, 68, 1282, 690)), out, max_bytes=max_bytes)


def viewer_thumb(src: Path, out: Path, *, max_bytes: int) -> int:
    """Square thumbnail of the refined ct3d isosurface (right panel of the viewer screenshot)."""
    im = flatten(Image.open(src), bg=(249, 249, 247))
    return save_png(
        im.crop((938, 176, 1248, 486)).resize((346, 346), Image.Resampling.LANCZOS),
        out,
        max_bytes=max_bytes,
    )


# ---------------------------------------------------------------------------------------------
# interactive viewers
# ---------------------------------------------------------------------------------------------
THEME_SYNC = """<script>
/* nefi site: follow the light / dark scheme of the page that embeds this viewer */
(function () {
  var root = document.documentElement;
  function scheme() {
    try {
      if (window.parent && window.parent !== window) {
        var s = window.parent.document.body.getAttribute("data-md-color-scheme");
        if (s) return s === "slate" ? "dark" : "light";
      }
    } catch (e) {}
    var m = /[?&]theme=(dark|light)/.exec(window.location.search);
    return m ? m[1] : null;
  }
  var t = scheme();
  if (t) root.setAttribute("data-theme", t);
  if (window.parent && window.parent !== window) root.classList.add("nf-embedded");
  try {
    new window.parent.MutationObserver(function () {
      var n = scheme();
      if (n && n !== root.getAttribute("data-theme")) window.location.reload();
    }).observe(window.parent.document.body, {attributes: true,
                                             attributeFilter: ["data-md-color-scheme"]});
  } catch (e) {}
})();
</script>
<style>
.nf-embedded main { max-width: none; padding: 0 2px 4px; }
.nf-embedded header, .nf-embedded footer { display: none; }
.nf-embedded body { background: transparent; }
.nf-embedded .nefi-vv { margin: 0; border: 0; border-radius: 0; background: transparent; }
:root.nf-embedded { --surface: #ffffff; }
:root.nf-embedded[data-theme="dark"] { --surface: #262432; --hairline: #37354c; --axis: #4f4c68; }
</style>
"""


def export_viewer(gallery: Path, name: str, out: Path) -> int:
    """Re-export a gallery viewer with ``nefi.viz.save_volume_viewer`` (+ the theme script)."""
    from nefi.viz import save_volume_viewer
    from nefi.viz.interactive import decode_volume

    P = json.loads((gallery / "physics" / name / "viewer.json").read_text(encoding="utf-8"))
    shape = P["shape"]
    vols = {v["name"]: decode_volume(v, shape) for v in P["volumes"]}
    rule = {k: v for k, v in P["threshold"].items()}
    save_volume_viewer(
        out,
        vols,
        extent=P["extent"],
        axes=P["axes"],
        axis=P["axis"],
        threshold=rule,
        mode="iso",
        cmap=P["cmap"],
        label=P["label"],
        level_rule=P["level_rule"],
        transform=P["transform"],
        center=P["center"],
        stretch=P["stretch"],
        vmin=P["vmin"],
        vmax=P["vmax"],
        meshes=True,
        title=P["title"],
    )
    text = out.read_text(encoding="utf-8")
    text = text.replace("<meta charset='utf-8'>", "<meta charset='utf-8'>\n" + THEME_SYNC, 1)
    out.write_text(text, encoding="utf-8")
    return out.stat().st_size


# ---------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gallery", default="runs/gallery", help="examples/gallery.py output")
    ap.add_argument("--refine", default="runs/refine_edges", help="examples/refine_edges.py output")
    ap.add_argument("--out", default="docs/assets", help="destination (docs/assets)")
    ap.add_argument("--max-kb", type=int, default=300, help="size limit per image (1 kB = 1000 B)")
    ap.add_argument("--viewer-max-kb", type=int, default=1000, help="size limit per viewer page")
    args = ap.parse_args(argv)

    gal, ref, out = ROOT / args.gallery, ROOT / args.refine, ROOT / args.out
    lim = args.max_kb * 1000
    sizes: dict[str, int] = {}

    def png(src: Path | Image.Image, rel: str, max_w: int | None = None) -> None:
        im = Image.open(src) if isinstance(src, Path) else src
        sizes[rel] = save_png(im, out / rel, max_bytes=lim, max_w=max_w)

    # overview figures (the gallery's own palette PNGs, kept lossless)
    png(gal / "physics" / "gallery.png", "gallery/overview.png")
    sizes["gallery/overview-2d.png"] = crop_2d_overview(
        gal / "physics" / "gallery.png", out / "gallery/overview-2d.png", max_bytes=lim
    )
    png(drop_title(Image.open(gal / "physics" / "gallery_3d.png")), "gallery/isosurfaces-3d.png")
    sizes["gallery/viewer-ct3d.png"] = crop_viewer(
        gal / "viewer_screenshot_ct3d_refined.png", out / "gallery/viewer-ct3d.png", max_bytes=lim
    )
    sizes["hero.png"] = hero(gal, out / "hero.png", max_bytes=lim)
    for rel, radius in (("hero.png", 28), ("gallery/viewer-ct3d.png", 16)):
        sizes[rel] = round_corners(out / rel, radius)
    sizes.update(thumbnails(gal, out / "gallery" / "thumbs", max_bytes=lim))
    sizes["gallery/thumbs/ct3d.png"] = viewer_thumb(
        gal / "viewer_screenshot_ct3d_refined.png", out / "gallery/thumbs/ct3d.png", max_bytes=lim
    )
    for name in EVOLUTION:
        rel = f"gallery/evolution-{name}.gif"
        sizes[rel] = copy_gif(gal / "physics" / name / "evolution.gif", out / rel, max_bytes=lim)

    # per-system comparisons (instance pages)
    for name in TWO_D + THREE_D:
        png(gal / "physics" / name / "compare.png", f"instances/{name}.png", 1640)
    png(gal / "physics" / "nv_relaxometry" / "compare_omega_L.png", "instances/nv_larmor.png")
    for name in THREE_D:
        png(gal / "physics" / name / "block3d.png", f"instances/{name}-3d.png", 2000)

    # baselines, performance, refinement
    png(gal / "baselines" / "panel.png", "gallery/baselines-panel.png")
    png(gal / "baselines" / "failure_modes.png", "gallery/baselines-failure-modes.png")
    for name in PERF:
        png(gal / "performance" / f"{name}.png", f"gallery/perf-{name}.png")
    for name in REFINE:
        png(ref / f"{name}.png", f"refine/{name.replace('_seed0', '')}.png")

    # guides and research pages
    png(gal / "physics" / "deconvolution" / "multiscale.png", "concepts/multiscale.png")
    png(gal / "physics" / "eit" / "history.png", "concepts/history.png")
    png(ROOT / "runs" / "geometric_representations.png", "research/representations.png")

    # interactive viewers
    for name in VIEWERS:
        rel = f"viewers/{name}.html"
        (out / "viewers").mkdir(parents=True, exist_ok=True)
        sizes[rel] = export_viewer(gal, name, out / rel)
        if sizes[rel] > args.viewer_max_kb * 1000:
            raise SystemExit(f"{rel} has {sizes[rel]} bytes > {args.viewer_max_kb} kB")

    total = sum(sizes.values())
    width = max(len(k) for k in sizes)
    for k in sorted(sizes):
        print(f"  {k:<{width}}  {sizes[k] / 1000:7.1f} kB")
    print(f"  {'total':<{width}}  {total / 1000:7.1f} kB ({len(sizes)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
