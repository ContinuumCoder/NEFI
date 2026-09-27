# Visualization — `nefi.viz`

`nefi.viz` turns inversions into figures, animations, dashboards and shareable reports. It is the
presentation layer of the library: every figure in the demo gallery, the per-run figures on the
GPU servers and the paper-style panels come from the functions documented here.

* **Optional dependency.** matplotlib is imported lazily inside the plotting functions:
  `import nefi` and `import nefi.viz` never need it (`pip install "nefi[viz]"` to plot).
* **Headless-safe.** Every function builds a `matplotlib.figure.Figure` without `pyplot`, so
  nothing ever opens a window; scripts and tests call `matplotlib.use("Agg")` first.
* **Inputs are whatever the library produces**: torch tensors, numpy arrays, `Result` objects,
  `{name: tensor}` field dicts, `Measurement`s, benchmark rows.
* **Outputs go to `runs/`** (git-ignored). `viz.savefig` keeps PNGs at ≤ 150 dpi and under a size
  budget (300 kB by default).

```python
import matplotlib
matplotlib.use("Agg")
from nefi import viz
from nefi.instances.toy1d import Toy1D
from nefi.solve.callbacks import FieldSnapshots

snaps, stages = FieldSnapshots(every=20), viz.StageSnapshots()
out = Toy1D().run(seed=0, device="cpu", callbacks=[snaps, stages])

viz.savefig(viz.compare_fields(out.gt, {"nefi": out.result}, out.measurement), "runs/toy/compare")
viz.savefig(viz.plot_fit(out.result.pred, out.measurement), "runs/toy/fit")
viz.savefig(viz.plot_history(out.result, noise_std=out.measurement.noise_std), "runs/toy/history")
viz.savefig(viz.plot_multiscale(out.result, stages, gt=out.gt), "runs/toy/multiscale")
viz.animate_snapshots(snaps, "runs/toy/evolution.gif", gt=out.gt["x"], result=out.result)
```

3-D fields use the same calls (`compare_fields` switches to the volume layout of §2.1); the
instance's display hints (§3) are picked up when `instance=` is passed:

```python
from nefi.viz._instances import resolve_instance

inst, _ = resolve_instance("thermal_tomography")          # smoke preset
out = inst.run(seed=0, device="cpu")
mask = out.measurement.meta["defect_mask"]
fig = viz.compare_fields(out.gt, {"nefi": out.result}, out.measurement, instance=inst,
                         mask=mask, panels="overlay")                   # mosaic + GT outline
viz.savefig(fig, "runs/tt/compare")
viz.savefig(viz.voxel_compare(out.gt, {"nefi": out.result}, domain=inst.domain()),
            "runs/tt/voxels")
```

---

## 1. Style guide

### Color

| Job | Rule | Implementation |
|---|---|---|
| identity (methods, loss terms, stages) | eight categorical hues in a **fixed order**, never cycled; a ninth series is folded ("+k more") | `viz.PALETTE` / `viz.PALETTE_DARK`, `theme().palette` |
| magnitude of a density / source / intensity | `magma` | `cmap_for("rho")` |
| material coefficient (diffusivity, conductivity, permeability, velocity, refractive index) | `viridis` | `cmap_for("alpha")`, `cmap_for("sigma")` |
| signed fields, errors, residuals, wavefields | `RdBu_r`, **symmetric limits centred at 0** | `cmap_for(quantity="error")` |
| wrapped phase | `twilight` on `[-π, π]` | `cmap_for("phase")` |
| unwrapped phase with a range < π | centred `RdBu_r` (a full circle would waste the contrast) | automatic in `cmap_for(name, data)` |
| measurements | `viridis`, or centred `RdBu_r` when **signed** — distinct from the unknown's map | `cmap_for(quantity="measurement", data=...)` |
| complex data (fields and measurements) | magnitude (`magma`) **and** phase (`twilight`, or centred `RdBu_r` for a range < π) | `plot_complex`, `draw_measurement` |
| a hint says otherwise | the `measurement_cmap` / `field_cmap` display hint (§3) | `viz.spec_from_hint` |

**Signed data** (`viz.is_signed`): data count as signed when min < 0 < max *beyond noise* —
the robust extremes (0.5 / 99.5 percentiles) have opposite signs and the smaller one reaches
10 % (`SIGNED_FRACTION`) of the larger. Noise dipping below a zero background (sinograms, counts,
spectra: a few % of the peak) stays sequential; a stray-field map with a 15 % negative lobe, a
zero-mean phase or a boundary voltage pattern is drawn with the diverging map centred at 0.
This applies to measurements and fields alike (a sequential quantity whose data turn out signed
switches to the centred map).

The categorical palette is validated for color-vision deficiencies (adjacent ΔE ≥ 8 under
protan/deutan simulation, ≥ 15 in normal vision) in both themes; three light-mode slots are below
3:1 contrast on the light surface, which is why **every multi-series plot carries a legend** and
every report carries the numbers as a table. The quantity is guessed from the field name
(`FIELD_QUANTITIES`: `rho`, `f`, `x`, `mu` → magma; `alpha`, `sigma`, `k`, `c` → viridis;
`chi`, `dn`, `u`, `bz` → centred; `phi`, `phase` → twilight); anything unknown falls back to the
data (centred diverging when positive and negative parts are comparable, `viridis` otherwise).

### Marks and layout

* Thin lines (1.1–1.4 pt), solid hairline gridlines, no top/right spines, sans-serif 7.5–11 pt.
* Ground truth is drawn in ink with a dashed line (1-D) or as its own panel (2-D / 3-D); the
  reconstruction takes the first categorical slot; measurements are muted dots.
* **No dual y-axes.** Quantities with different units get stacked panels that share the x axis
  (e.g. loss / β/K / learning rate in `plot_history`).
* Images share one color scale per comparison (GT and every reconstruction), errors share their
  own symmetric scale; every image scale has a colorbar. Image panels keep a hairline frame so
  near-zero error maps stay visible on the light surface.
* Light and dark themes: `viz.use_style(dark=True, context="paper"|"notebook"|"talk")` as a
  context manager, or `viz.set_default_style(dark=True)` for a whole script.
* Sizes: `viz.figsize(ncols, nrows, cbar=..., title=...)`, `viz.paper_figsize(columns=1|2)` for
  3.4 in / 7.0 in journal widths.

### Saving

`viz.savefig(fig, "runs/x/compare", formats=("png", "svg"), dpi=130, max_kb=300)` writes every
requested format (the path suffix counts), creates directories, uses deterministic metadata, and
keeps PNGs under `max_kb` by palette quantization, then by lowering the dpi (≥ 55).

---

## 2. What every plot shows

### Fields (`nefi.viz.fields`)

| Function | Shows |
|---|---|
| `compare_fields(gt, {method: recon}, measurement, panels=...)` | **The standard reconstruction figure**: measurement (compact view) · ground truth · each reconstruction (metrics in the title) · extra panels. `kind` is detected: 1-D → overlaid lines + error panel; 2-D → images (optional `peaks=True` markers: GT = hollow circles, recovered = crosses; `mask=` contour); 3-D → the volume layout of §2.1; complex → magnitude and phase rows. Reconstructions at another resolution are resampled to the GT grid. `metrics="auto"` (PSNR/SSIM or PSNR/rel. error, computed on the displayed maps), `{name: fn}`, or precomputed `{method: {metric: value}}` (e.g. the instance's own metrics). `instance=` / `hints=` apply the display hints (§3: `field_transform`, `field_label`, `field_cmap`, `volume_axis`); `transform=` / `cmap=` override them. |
| `plot_field(x)` | One field: line, image, 3-D depth mosaic, or magnitude/phase. |
| `image_panels({label: image})` | A grid of 2-D panels with one shared scale. |
| `depth_mosaic(vol, n_slices=8, mask=...)` | Depth slices of a 3-D field, one shared scale, physical depths from a `Domain`, optional mask outline. Thin volumes are fine: with fewer slices than `n_slices` every slice is shown, a squeezed single slice (`nz = 1`) becomes a one-panel mosaic. |
| `orthoslices(vol, index=...)` | XY / XZ / YZ slices through `index` (default: the most anomalous voxel; `index=viz.anomaly_centroid(gt)` cuts through the GT anomaly). |
| `projections(vol, mode="auto")` | Max / min / mean projections along each axis (`auto` projects the anomaly side: `min` for low-diffusivity defects). |
| `voxel_view(vol)` | Isosurface-free 3-D scatter of thresholded voxels (depth axis exaggerated for thin slabs); `spec` / `vmin` / `vmax` / `threshold` to share scales. |
| `voxel_compare(gt, {method: recon})` | GT and reconstructions as thresholded voxels **side by side**: one threshold (half-way from the GT background to its anomaly extreme), one colour scale, one camera and one set of limits; titles give the voxel count and the IoU with the GT voxel set. |
| `plot_complex(x)` | Magnitude (`magma`) and phase (`twilight`). |
| `anomaly_centroid(vol, mask=None)` | Grid index of the dominant anomaly: the given mask, else the voxels whose absolute difference to the median reaches half the largest difference; the connected component with the largest deviation wins; deviation-weighted centroid (2-D and 3-D). |

**Extra panels of `compare_fields`** (`panels=("measurement", "gt", "recon", <extra>...)`, or
`panels="overlay"` as a shorthand for the default plus one extra; `error=False` drops the error):

| panel | 2-D | 3-D | 1-D |
|---|---|---|---|
| `"error"` (default) | signed error `recon − GT`, `RdBu_r` centred at 0, own colorbar | one error row per method (mosaic, cross-sections, **mean** projection) | error line panel |
| `"overlay"` | the reconstruction with the **GT contours** (solid white, dark halo — visible on every colormap; a `mask=` is outlined instead) and the reconstruction's own contour at the same level (dashed aqua). Levels: GT background ± ½ of the largest deviation on each side | GT outlined on every reconstruction panel (slices, cross-sections, projection) — **the gallery's default for 3-D systems** | — (the main panel already overlays) |
| `"profile"` | a line cut through the GT anomaly centroid (`profile_axis=0`: along x), GT dashed vs every reconstruction; the cut is marked on the image panels. Great for blurred / 1-D-like cases | cuts along x, y and depth through the centroid (three line panels) | — |

With several methods, `error` and `overlay` become extra rows under the reconstructions and
`profile` one panel with every method. Field transforms (e.g. `zero_mean` for a phase defined up
to a constant) are applied to GT and reconstructions *separately* and every panel title says so
("ground truth · mean removed").

### 2.1 3-D fields (`nefi.viz.volume`)

A field with three non-singleton dimensions `(nx, ny, nz)` is never reduced to one slice. Every
3-D view combines three readings of the volume, all on **one shared colour scale**:

* a **depth mosaic** — up to 6 evenly spaced slices along the *volume axis* (default: the last
  one, depth `z`; the `volume_axis` hint changes it) at the centres of equal depth bins
  (`mosaic_slices(16, 6)` → 1, 4, 6, 9, 12, 14; thin volumes, e.g. `nz < 4`, show every
  slice), with the depth of each slice as a label (physical with a `Domain`);
* the two **cross-sections through the anomaly centroid** (`anomaly_centroid`: from the mask —
  e.g. the thermal `defect_mask` — or from `|gt − median|`), `x–z` and `y–z`, depth pointing
  **down** (the first slice — the observed face of surface-measurement problems — on top), a
  `+` at the centroid;
* the **anomaly-side projection** along the volume axis: `min` when the anomalies lie below the
  background (low-diffusivity defects), `max` otherwise (sources, absorbers); error rows use the
  `mean`.

Where these appear:

| Figure | 3-D content |
|---|---|
| `gallery.png` (overview) | 1-D / 2-D systems keep their three-panel tiles; **every 3-D system gets a full-width row**: GT mosaic (2 × 3) · measurement (compact view) · reconstruction mosaic with the GT outlined · cross-sections GT / reconstruction · projection GT / reconstruction (`draw_volume_tile`). |
| `gallery_3d.png` | One block per 3-D system: larger GT and reconstruction mosaics (up to 8 slices, GT outlined in white, the reconstruction's own contour dashed), **contrast contours** of GT (solid) and reconstruction (dashed) at 25 / 50 / 75 % of the GT contrast on the central slice and the depth cross-section, and **shaded isosurfaces** — the GT at its threshold rule, the reconstruction at the **volume-matched** level, same camera, levels / IoU / Dice in the titles (§2.2; `physics_volumes`, `draw_volume_block`); with the edge refinement (`refine=True`) every reconstruction panel comes twice — smooth and refined — and the block title carries the refinement's verdict, IoU / Edge-F1 and χ before → after. The static fallback of the "3-D systems" section of `index.html`, where every system's row (`<name>/block3d.png`) is followed by its interactive viewer (§2.3). |
| `compare_fields` / `<name>/compare.png` | Rows = ground truth, each reconstruction (and each signed error with `"error"`), columns = mosaic · cross-sections · projection; the measurement spans the first two rows. The gallery uses `panels=(..., "overlay")` for 3-D systems (GT outlined on the reconstruction panels); `"profile"` adds x / y / depth line cuts through the centroid. |
| `<name>/mosaic.png`, `<name>/voxels.png` | The reconstruction's 8-slice depth mosaic (GT mask outlined) and `iso_compare(GT, reconstruction)`: contrast contours + shaded isosurfaces at the GT rule / volume-matched level (the file keeps its old name). |
| `<name>/block3d.png`, `<name>/viewer.html`, `<name>/viewer.json` | The system's `gallery_3d` row as its own figure, the standalone interactive viewer and its data (what `index.html` embeds) — written when the gallery runs with `interactive` (default). |
| `<name>/evolution.gif` | `animate_snapshots(..., volume="mosaic")`: 4 depth slices of the evolving field above the same slices of the GT, loss curve with a moving marker (coarse curriculum stages are sliced at the same relative depths); `volume="slice"` animates the slice through the GT anomaly centroid. |
| `examples/visualize_result.py` | compare (overlay), mosaic, orthoslices through the GT anomaly centroid, projections, isosurfaces (GT vs reconstruction, `field_isosurfaces.png`); with `--html` the report embeds the interactive viewer (also `viewer.html`). |

Building blocks for custom layouts (all draw into caller-provided axes): `draw_slices`,
`draw_sections`, `draw_projection`, `draw_voxels`, `Outline` (GT mask or anomaly level set),
`section`, `project`, `projection_mode`, `mosaic_slices`, `mosaic_shape`.

### 2.2 Isosurfaces and matched levels (`nefi.viz.isosurface`)

Reconstructions of volumetric problems recover the structure of the ground truth with **softer
edges**, so cutting them at the GT's own threshold under- or over-segments them and the voxel view
looks worse than the field is. Level sets are therefore compared with explicit rules:

* **GT threshold rule** (`voxel_rule(instance, gt)`): the instance's IoU rule — an IoU metric
  that is a `functools.partial` of `iou_below` / `iou_above` with a `tau` (thermal tomography:
  `α < 0.03`; ct3d: `μ > 0.25`), or a half-maximum rule from `cfg.iou_fraction` and one
  `*background` config value (dot3d: excess over `μ_a,bg` beyond 50 % of each volume's own peak
  excess, a *relative* rule) — else half-way from the GT background (median) to its extreme. The
  `voxel_threshold` display hint (a number, `(side, value)` or a rule mapping) overrides it.
  `select_mask(values, rule)` applies a rule (it reproduces the instances' IoU masks exactly).
* **reconstruction level rules** (`iso_levels(gt, recon, rule)` returns all of them with the
  voxel count, IoU and Dice against the GT set): `"fixed"` — the GT rule itself (what the
  instance's IoU metric does); `"matched"` — the **volume-matched** level, enclosing as many
  voxels as the GT does at its threshold (`matched_level`: half-way between the two sorted
  values that bracket the GT count); `"otsu"` — Otsu's threshold of the reconstruction's
  histogram (256 bins; the middle of the tied splits when the classes are separated by a gap);
  `"manual"` — a given value.
* **display transforms** (`volume_transform(a, spec)`, display only): `smooth` (Gaussian, σ in
  voxels), `sharpen` (unsharp mask `x + amount (x − G_σ x)`, clipped to the input range),
  `edge_preserve` (5 Perona–Malik diffusion iterations: smooths inside regions, keeps steep
  edges). They are applied to reconstructions (never to the GT) before iso-extraction, and every
  title says `display: …`. Choose them per instance with the `volume_transform` display hint
  (`"sharpen"`, `{"name": "smooth", "sigma": 2}` or `{field: spec}`) or the `transform=`
  keyword; metrics reported as the reconstruction's (manifest, tables) never use them.
* **meshes** (`iso_mesh(values, level, side)`): `skimage.measure.marching_cubes` on the volume
  padded with an outside value (surfaces close at the box), faces oriented outward, decimated by
  `step_size` to a face budget; `None` without scikit-image — the figures then fall back to the
  voxel scatter.

| Function | Shows |
|---|---|
| `iso_compare(gt, {method: recon}, instance=..., how="matched")` | The `voxels.png` figure: contrast contours on the central slice and the depth cross-section (GT solid, reconstruction dashed, 25 / 50 / 75 % of the GT contrast — spread-out dashed contours are soft edges), then the shaded isosurfaces of the GT (its rule) and of each reconstruction (`how` level), one camera, titles with the levels, IoU, Dice and the IoU at the GT threshold. |
| `draw_isosurfaces(axes, [(title, volume, level, side)])` | Shaded `Poly3DCollection`s (two-sided Lambert headlight, depth stretched for thin slabs as in `voxel_view`); GT neutral, reconstructions in the categorical slots. |
| `draw_level_contours(axes, gt, recon, center=...)` | The two contour panels. |

`examples/gallery.py` prints the reconstruction's IoU / Dice at the GT threshold and at the
volume-matched level and tabulates them in the "3-D systems" section (`manifest.json`:
`entries[i].iso`). At the gallery budget (`--budget 0.1`, CPU, seed 0):

| system | GT rule | IoU at GT threshold → matched | Dice at GT threshold → matched | IoU Otsu |
|---|---|---|---|---|
| thermal_tomography | `α < 0.03` (IoU rule) | 0.984 → 1.000 | 0.992 → 1.000 | 0.218 |
| ct3d | `μ > 0.25` (IoU rule) | 0.638 → 0.660 | 0.779 → 0.795 | 0.297 |
| deconvolution3d | half-way, `x > 0.464` | 0.149 → 0.423 | 0.260 → 0.594 | 0.241 |
| dot3d | half max of the excess (IoU rule) | 0.587 → 0.796 | 0.740 → 0.886 | 0.741 |
| photoacoustic3d | half-way, `p0 > 0.500` | 0.663 → 0.678 | 0.798 → 0.808 | 0.676 |

The IoU at the GT threshold is the instance's own IoU metric where it has one; the gain of the
matched level measures how much of the apparent error is edge softness rather than misplaced
structure (deconvolution3d: the blurred filaments are found but under-segmented at the GT's
half-way level).

### 2.3 Interactive 3-D viewer (`nefi.viz.interactive`)

Every 3-D system of the gallery's `index.html` gets a **drag-to-rotate viewer** right under its
static row: ground truth and reconstruction side by side with one camera (with the gallery's edge
refinement: GT | smooth | refined, each reconstruction with its own level and IoU / Dice),
rendered by a
dependency-free script (`nefi/viz/assets/volume_viewer.js` + `.css`, vanilla JavaScript on a 2-D
canvas — a small software rasterizer with a depth buffer; no libraries, no CDN, no network).

```python
from nefi import viz

html = viz.volume_viewer_html({"ground truth": gt, "reconstruction": rec},   # tensors / arrays
                              extent=inst.domain(), threshold=viz.voxel_rule(inst, gt),
                              mode="iso", field="alpha")         # an HTML fragment
viz.save_volume_viewer("runs/x/viewer.html", {"ground truth": gt, "reconstruction": rec},
                       extent=inst.domain(), mode="iso")          # a standalone page
viz.viewer_size_bytes({"ground truth": gt, "reconstruction": rec})  # bytes it adds to a page
```

**Controls.** Drag (one finger) orbits (turntable about the vertical axis; the depth axis points
down so the observed face `z = 0` is on top), wheel / pinch zooms, right-drag or shift-drag pans,
double-click resets; with a focused canvas: arrows, `+` / `−`, `0`. An axes triad, the physical
extents at the box corners and a depth cue (farther voxels smaller and faded toward the
background) keep the orientation readable. "link cameras" (on by default) keeps both panels on
one camera; "reset view" resets it.

**Views** (`mode=`): `iso` — shaded isosurfaces (lit triangles: headlight Lambert + Blinn–Phong,
smooth (Gouraud) or flat shading, back faces culled; "GT overlay" draws the GT surface
semi-transparent over each reconstruction); `voxels` — thresholded voxel centres as squares in
the colormap of the static figures (`magma` / `viridis` / `RdBu_r`, … as 33-stop tables) with an
opacity slider; `slice` — a movable x / y / z slice drawn as a textured quad in the rotated frame
(default: through the GT anomaly centroid); `iso+slice`, `voxels+slice` (alias `points+slice`).

**Levels and readout.** The GT threshold slider (and `<` / `>` side toggle) starts at the GT rule
(§2.2; relative rules become a "fraction of the peak" slider), the reconstruction's level follows
the selected rule — volume-matched (default), GT threshold, Otsu, or manual (slider); `↺` returns
to the defaults. Each panel shows its level and rule, the GT its voxel count, each reconstruction
the IoU and Dice with the GT set, computed in the browser; the colour bar marks every level. The
display transform select (raw, smooth σ = 1 / 2, sharpen, edge-preserving) re-computes the
reconstruction and its isosurface live and labels it "display: …".

**Data and sizes.** Volumes are area-averaged to at most `max_side = 64` voxels per axis
(≤ 64³; the info line says so) and quantized to integer codes `lo + q · step` — 16-bit up to
48³ voxels, 8-bit beyond (`bits=`) — base64 in a `<script type="application/json">` block; shape,
physical extent and axis names make the aspect ratio physical (thin slabs get the static
figures' depth stretch, labelled `z ×k`, adjustable). Codes are nudged so the GT rule selects
exactly the voxels it selects in the full-precision arrays: without downsampling, the in-browser
IoU at the GT threshold equals the instance's IoU metric. The default isosurfaces are marching-cubes meshes from Python (vertices
uint16, faces uint16 / uint32, ≤ `max_faces = 20000` faces each); other levels and transforms are
re-extracted in the browser (marching tetrahedra, ≈ 20 ms at 64³). A viewer adds its fragment
plus, once per page, the script and stylesheet (≈ 62 kB): the gallery's 3-D systems (smoke grids,
16-bit, with meshes) take thermal_tomography 17 kB, dot3d 28 kB, photoacoustic3d 77 kB,
deconvolution3d 134 kB and ct3d 250 kB; two 96³ volumes (→ 64³, 8-bit) with meshes ≈ 0.95 MB. Several viewers coexist on one page (unique element
ids; `include_assets=False` for all but the first).

**Plotly mode (optional, not self-contained).** `plotly_volume_html(...)` /
`save_volume_viewer(..., kind="plotly")` / `examples/gallery.py --interactive plotly` render
`plotly.graph_objects.Isosurface` scenes (same rules and levels, cameras synchronized) with
`include_plotlyjs="cdn"`: the page loads plotly.js from `cdn.plot.ly` and needs network access
(volumes ≤ 32³: plotly writes every grid point as JSON text). Never the default; the canvas viewer
is.

### Measurements (`nefi.viz.measurement`)

The layout is taken from `Measurement.meta["layout"]`, then from instance display hints, then
detected from the shape relative to the field (`detect_layout`):

| Layout | Typical shape | Compact view (tiles, `compare_fields`) | Full view (`plot_measurement`) |
|---|---|---|---|
| `signal` | `(n,)` for a 1-D field | dots | line + ±σ band |
| `vector` | `(n_sensors,)` | dots vs sensor | same |
| `image` | field-shaped | image | image + colorbar |
| `points` | field-shaped + sparse mask | scatter of observed pixels | same |
| `matrix` / `sinogram` | `(n_views, n_det)` | image, rows vertical | `plot_sinogram`: sinogram + projection profiles |
| `spectra` | `(n_freq, H, W)` | noise map Σ_ω S | `plot_spectra`: noise map, frequency slices, pixel spectra (NeTMY) |
| `frames` | `(n_t, H, W)` + `frame_times` | time mean | `plot_frames`: geometric-time frame strip (per-frame scale) + decay curves (NeFTY) |
| `stack` | `(n, H, W)` | mean over the stack (masked entries ignored) | channel slices + channel profiles |
| `traces` | `(n_src, n_rec, n_t)` | gather of the middle source | `plot_traces`: image gather (time down) + wiggle plot |
| `kspace` | masked Fourier samples | log-magnitude | `plot_kspace`: sampling mask + log-magnitude |
| `complex` | complex / stacked re-im (`complex_stack` hint) | **magnitude and phase** (two insets) | magnitude + phase |
| `volume` | field-shaped 3-D `(nx, ny, nz)` (e.g. a blurred stack) | max projection along the volume axis (signed data: mean; `reduce` = index → that slice) | depth mosaic |
| `sinogram` stack | `(n_views, n_det, n_z)` (+ `angles`) | the middle slice; the slice axis is the `stack_axis` hint, else the axis whose length equals the field's depth, else the last | `plot_sinogram` of the middle slice |

Colours follow §1: signed data (e.g. a stray-field map, boundary voltages, pressure traces) get
the centred diverging map, everything else `viridis`; complex data show magnitude and phase.
Instance hooks: when an instance implements `measurement_image(measurement)`, its result is the
compact view — a 1-D / 2-D result is drawn **as it is** (an image with its own label), even when
the layout hint describes the raw data as a stack (e.g. a boundary-difference view of EIT
patterns). The hook may return `(tensor, "label")`; otherwise the `measurement_image_label` hint
names it. `plot_fit` never calls the hook (data, prediction and residual stay in data space).

`plot_fit(pred, measurement)` puts data, prediction and residual side by side with the **noise
floor**: 1-D → overlay + residual with ±σ / ±2σ bands; images → data | prediction | residual |
residual histogram against N(0, σ²); stacks → data | prediction | RMS residual per slice against
the σ line | histogram. The title reports RMSE, σ and **RMSE/σ**: ≈ 1 means the fit reached the
noise level (Morozov); ≪ 1 over-fits the noise; ≫ 1 under-fits. `residual_stats` returns the
same numbers.

### Training dynamics (`nefi.viz.training`)

| Function | Shows |
|---|---|
| `plot_history(result, noise_std=σ)` | Loss components on a log scale (total in ink, data loss in slot 1, identical terms merged), curriculum stage boundaries (labelled with the stage grid), shaded annealing windows (β < K), the annealing progress β/K and the learning rate in stacked panels, and a σ² floor line (only meaningful for a plain MSE data loss — the gallery draws it only then). Components switched off in some stages are aligned with NaN (`history_arrays`). |
| `plot_stage_summary(result)` | Steps, seconds (ms/step) and final data loss per stage; early stops are named. |
| `plot_multiscale(result, StageSnapshots)` | End-of-stage fields at their own resolution (32² → 64²), the final and GT fields, above the loss curve. |
| `animate_snapshots(FieldSnapshots, "x.gif")` | GIF (or MP4 with ffmpeg) of the field during optimization with the GT and a moving marker on the loss curve; 3-D fields: an evolving depth mosaic above the GT mosaic (`volume="mosaic"`, default) or the slice through the GT anomaly (`volume="slice"`). The layout is frozen after the first frame (fast), and the GIF is kept under 300 kB by thinning frames (the last one is kept), shrinking the palette, then downscaling. |

`StageSnapshots` is a solver callback that keeps every field at the end of each stage; the
library's `FieldSnapshots(every=k)` keeps the primary field every `k` steps.

### Qualitative comparisons (`nefi.viz.qualitative`)

| Function | Shows |
|---|---|
| `method_grid(rows, row_labels)` | The classic scenes × methods figure: GT and measurement first, one shared color scale per row, headline metrics under each method, optional signed-error rows. Cells may be tensors, Results, Measurements, error strings or `None`. |
| `failure_modes(gt, {method: recon})` | Per method: reconstruction with the GT support outlined · signed error · **leakage** map (mass outside the dilated support, % in the title) · log error power spectrum with the **axis excess** — error power on the Fourier axes relative to an isotropic error with the same radial spectrum (Hann-windowed, lowest frequencies ignored): ≈ 1 isotropic, ≫ 1 **cross / streak artifacts**, < 1 diagonal structure. `failure_stats` / `axis_excess` return the numbers. |
| `baselines_panel(instance, budget_scale=0.1)` | Runs the instance's own method and every `baselines()` entry on one measurement (via the benchmark protocol, so closed-form and ADMM baselines work) and draws the grid; returns `(figure, runs)`. `run_methods` returns the raw runs. |

### Performance (`nefi.viz.performance`)

All plots take **rows** (lists of dicts): `collect_performance` output, `BenchmarkResult` /
`benchmark.json` / `rows.csv`, a `RuntimeTable`, or a gallery `manifest.json` (`load_rows`
normalizes aliases such as `name` → `instance`, derives `n` from `shape` and `ms_per_step` from
`time_s / steps`). Row keys: `instance`, `method`, `mode`, `device`, `class`, `stage`, `shape`,
`n`, `params`, `steps`, `ms_per_step`, `fwd_ms`, `bwd_ms`, `opt_ms`, `time_s`, `per_stage_s`,
`peak_mem_mb` (CUDA), `saved_mb`, metric columns, `error`.

| Function | Shows |
|---|---|
| `collect_performance(instances, device, steps)` | Re-implements `tools/profile_instance.py` as rows: smoke-sized problem, finest stage, forward / backward / optimizer split, peak CUDA memory and the **autograd saved-tensor volume** of one step (a device-agnostic memory proxy that separates adjoint from unrolled gradients). `variants={"adjoint": {"grad_mode": "adjoint"}, ...}` adds one row per variant; `collect_scaling(name, [32, 64, 128], key="n")` sweeps sizes. |
| `plot_step_time(rows)` | ms/step per instance; stacked forward / backward / optimizer when available, grouped by device otherwise. |
| `plot_scaling(rows)` | Step time and memory vs grid points, log-log, fitted slopes in the legend and an O(N) guide. |
| `plot_memory(rows)` | Memory per gradient mode (adjoint vs checkpoint vs autograd) and grid size. |
| `plot_wallclock_breakdown(results)` | Stacked bars of where the time went (data generation, each stage, overheads, evaluation, figures). |
| `plot_bench_table(rows, "psnr")` | Mean ± Student-t CI per method, one small multiple per scene class; best mean in bold. |
| `load_profiles("runs/_perf")` → `profile_speedups(rows)` → `plot_speedup(rows)` | **Speedup vs eager** per instance from `tools/profile_instance.py --json` profiles (`*.jsonl`): every accelerated variant (`compile=field/step`, `autocast=bf16`, `cuda graphs`, `--set` overrides such as `solver=chebyshev`) against the **latest** plain eager solver-loop run of the same instance, device and thread count; bars start at 1× (log axis), > 1× is faster; harness variants (`--minimal`) are skipped. The gallery adds the panel when profiles exist and skips it silently otherwise. |

`plot_step_time` appends the profiled grid to every label (`ct3d · 32×32×16`), so volumetric
instances stand out; the gallery's scaling plot includes the 3-D heat solver (thermal tomography,
adjoint gradients) next to toy1d and deconvolution.

### Multi-physics gallery (`nefi.viz.multiphysics`)

`physics_gallery(instances=None, budget_scale=0.15, device="auto", out_dir="runs/gallery")` runs
every registered instance (or the given names / classes / objects) and writes

* `gallery.png` — one tile per physical system: ground truth | measurement (compact view) |
  reconstruction, with the headline metric (PSNR > SSIM > HF1 > rel. error > …), solve time and
  steps; failed systems become a tile with the error text; **3-D systems get a full-width row**
  below the tiles (§2.1). The instance's display hints are applied (e.g. holography's phase is
  anchored to zero mean by the model's `ZeroMean` head, current_density shows `|J|`);
* `gallery_3d.png` — larger mosaics and GT vs reconstruction voxels of every 3-D system (§2.1);
* `<name>/{tile,compare,measurement,fit,history,stages,multiscale,mosaic,voxels}.png` and
  `evolution.gif` (`animate=True`); `compare` uses the signed error, 3-D systems the GT overlay
  (the `compare_panel` hint overrides both); `voxels.png` holds the isosurfaces of §2.2;
* 3-D systems with `interactive=True` (default; `"plotly"`, or `False` to skip):
  `<name>/block3d.png` (the system's `gallery_3d` row), `<name>/viewer.html` (standalone) and
  `<name>/viewer.json` (embedded by `gallery_sections`, §2.3);
* 3-D systems with `refine=True` (`examples/gallery.py`: on by default, `--no-refine`): after the
  smooth solve, `nefi.solve.refine_run_output(entry)` runs the edge refinement of
  `docs/refinement.md` (`refine_entry`); the refined field becomes a **third column — GT | smooth
  | refined** — in the overview's 3-D row, in `gallery_3d.png`, as a second method in
  `compare.png` / `voxels.png` and as a third panel of the viewer (same camera and level rule).
  The tile title adds `refined ✓ IoU a → b · Edge-F1 c → d · χ e → f` (IoU / Edge-F1: the
  instance's own metrics where it has them, else the refinement's at the GT threshold; χ =
  RMSE/σ, or the RMSE without a noise level); a refused refinement keeps the smooth result and
  its candidate is drawn labelled "refused refinement (not used)". The headline metric stays the
  smooth result's; the report's `summary()` is the caption (manifest: `entries[i].refine`; the
  "3-D systems" section gets an "Edge refinement" table);
* 3-D budgets: at `budget_scale ≥ 0.2` a 3-D system runs at least `steps_3d_multiplier` (default
  2) × its smoke preset's steps, above `max_steps` (they are under-converged at the plain rule;
  most of ct3d's refinement gain at the plain budget was just more steps) — `budget` in the
  manifest records `steps_3d_floor`;
* `manifest.json` — per system: status, error + traceback, scene class, smoke preset, budget,
  grid / measurement shapes, detected layout, metrics, timings (`generate_s`, `solve_s`,
  `per_stage_s`, `evaluate_s`, `plots_s`, `volume_extras_s`, `total_s`), ms/step, parameters,
  config hash, figures, and for 3-D systems `iso` (GT rule, levels, IoU / Dice at the GT
  threshold, volume-matched and Otsu levels) and `viewer` (files, bytes, grid); top level:
  `overview`, `overview_3d`, `volumetric` (names of the 3-D systems), `interactive`.

**Budget rule.** Problem sizes come from each instance's smoke preset (exactly the
`nefi run --smoke` lookup: `smoke_overrides`, `configs/<name>_smoke.yaml`, `PRESETS["smoke"]`).
Steps = `budget_scale ×` the instance's *default* curriculum, floored by the preset's own
(validated) budget and capped by `max_steps`; `time_budget_s` bounds the solve. Every instance
runs in its own `try` block — an import, build, data, solve or plotting failure never stops the
gallery.

### Reports (`nefi.viz.report`)

`html_report(out_dir, title, sections, config_hash=..., metrics=rows)` writes **one
self-contained HTML file**: figures embedded as base64 (PNG / SVG / GIF, animations included),
metric and timing tables (tabular figures, status glyphs), a table of contents, the config hash,
the environment (versions, device, CUDA, git revision read from `.git` files) and light / dark page
chrome. `Section(title, text, images, table, code, subsections)`; images may be paths,
`(path, caption)`, `{"path", "caption", "wide"}` or figures. `markdown_report` writes the same
content as Markdown with relative image links.

---

## 3. Instance hooks (optional)

Instances need nothing special, but three hooks improve their figures:

* `measurement_image(measurement) -> Tensor | (Tensor, label)` — the compact view of the
  measurement in display units (e.g. deconvolution converts Poisson counts to intensity, EIT
  shows a boundary-difference image, ct3d the most structured sinogram slice). A 1-D / 2-D
  result is drawn as it is, with its own label.
* `viz_hints = {...}` (class attribute, or a method returning the dict) — display hints, see
  below.
* `Measurement.meta` keys `layout`, `frame_times`, `freqs`, `angles`, `dt` (always win), and a
  field-shaped mask under `defect_mask` / `anomaly_mask` / `support_mask` / `support` /
  `inclusion_mask` (outlined on 3-D views and used for the anomaly centroid).

### Display hints (`viz_hints`, `nefi.viz.hints`)

Hints are resolved as `INSTANCE_HINTS[name]` (built-in defaults) < `instance.viz_hints` < the
`hints=` argument of a plotting call; `meta["layout"]` always wins for the layout. Field-level
keys accept one value for every field or a `{field_name: value}` mapping (`"*"` = default).
`viz.HINT_KEYS` holds this table.

| Key | Meaning | Example |
|---|---|---|
| `layout` | measurement layout (`viz.LAYOUTS`) | `"sinogram"` |
| `word` | name of the stacked axis | `"patterns"` |
| `reduce` | compact view of a stack / volume: `"sum"`, `"mean"`, `"max"`, `"std"` or an index | `0` |
| `complex_stack` | real and imaginary parts stacked along the `"first"` / `"last"` axis | `"first"` |
| `axis_values` | instance method returning the stacked axis values | `"angles"` |
| `row`, `col` | axis labels of matrix / sinogram layouts | `"view"`, `"detector"` |
| `stack_axis` | slice axis of a 3-D matrix / sinogram measurement | `-1` for `(n_views, n_det, n_z)` |
| `measurement_cmap` | colormap of measurement views: a matplotlib name, `"signed"` (centred `RdBu_r`), `"sequential"`, `"cyclic"`, a quantity (`"density"`), or `"auto"` | `"signed"` |
| `measurement_transform` | transform of the compact view (and of the noise map of spectra): `"log"` (noise-aware floor: 3σ of the negative noise values), `"abs"`, `"zero_mean"`, … | `"log"` |
| `measurement_label` | display name of the measurement in panel titles | `"B_z"` |
| `measurement_image_label` | label of the `measurement_image` hook's view | `"boundary difference"` |
| `field_cmap` | colormap of the fields (values as `measurement_cmap`) | `{"chi": "signed"}` |
| `field_transform` | display transform of GT and reconstructions (applied to each map separately; panel titles say so): `"zero_mean"`, `"abs"`, `"log"`, `"grad_magnitude"`, `"curl_magnitude"` (‖∇×(g ẑ)‖ of a stream function, zero outside the field of view), `"real"`, `"imag"`, `"phase"`, a callable, or the name of an instance method | `"zero_mean"` |
| `field_label` | display name of the (transformed) field | `{"g": "sheet current"}` |
| `volume_axis` | slicing / projection axis of 3-D fields | `-1` (depth z) |
| `voxel_threshold` | GT threshold rule of voxel / isosurface views and of the viewer (§2.2; default: the instance's IoU rule) | `("below", 0.03)` |
| `volume_transform` | display transform of 3-D reconstructions before iso-extraction (§2.2), str / mapping or `{field: spec}` | `"sharpen"` |
| `compare_panel` | extra panel(s) of the gallery's compare figure: `"error"`, `"overlay"`, `"profile"` | `"profile"` |

Built-in defaults (`viz.INSTANCE_HINTS`) cover the current instances, among them:
`holography` → no display transform (its phase gauge is fixed in the model by a `ZeroMean` head; use `field_transform="zero_mean"` for phase-like fields of your own instances);
`current_density` → signed `B_z` (`measurement_cmap="signed"`, `measurement_label="B_z"`) and the
sheet current `|J| = |∇×g|` instead of the stream function (`field_transform={"g":
"curl_magnitude"}`); `nv_relaxometry` → log-scaled noise map (`measurement_transform="log"`).
Instances override them with their own `viz_hints`.

---

## 4. Paper-style figures per instance

| Instance | Figure | Calls |
|---|---|---|
| `nv_relaxometry` (NeTMY) | density maps with peak markers, noise map / spectra, Larmor map | `compare_fields(gt, {"NeTMY": res, "Tikhonov": grid}, meas, field="rho", peaks=True)`; `plot_measurement(meas, instance=inst)` (noise map, frequency slices, pixel spectra); `compare_fields(..., field="omega_L")`; `failure_modes(gt["rho"], {...})` for cross / leakage artifacts |
| `thermal_tomography` (NeFTY) | depth-slice comparison (Fig. 4), surface frames, 3-D view | `compare_fields(gt, {"NeFTY": res, "Grid": grid}, meas, slices=6, mask=meas.meta["defect_mask"])`; `depth_mosaic(res, mask=...)`; `plot_frames` via `plot_measurement`; `voxel_view`, `projections(mode="min")` |
| `deconvolution`, `sparse_view_ct`, `poisson_source`, `holography`, `current_density`, `diffraction_tomography`, `reaction_diffusion`, `eit`, `darcy_flow`, `wave_fwi` | method comparison | `baselines_panel(name, budget_scale=1.0, device="cuda")` or `method_grid` over several scenes; `plot_fit` for the data fit |
| any | training dynamics | `plot_history`, `plot_multiscale(res, StageSnapshots)`, `animate_snapshots` |
| any | benchmarks | `plot_bench_table(BenchmarkResult.load(dir), "psnr")`; efficiency: `plot_step_time`, `plot_scaling`, `plot_memory` from `collect_performance` / `collect_scaling` |

For publication, wrap the calls in `with viz.use_style(context="paper"):` and save with
`formats=("pdf", "png")` (fonts stay text in SVG / PDF).

---

## 5. Scripts

### The demo gallery

```bash
python examples/gallery.py --budget 0.15 --device cpu --out runs/gallery --html
python examples/gallery.py --instances toy1d,deconvolution,nv_relaxometry --no-perf
python examples/gallery.py --device cuda --budget 1.0 --max-steps 0 --time-budget 0 --html  # server
python examples/gallery.py --instances thermal_tomography,dot3d --budget 0.1 --html --no-interactive
python examples/gallery.py --budget 0.2 --no-refine --steps-3d-multiplier 0   # plain 3-D budgets
```

Writes `physics/` (gallery, per-instance figures, manifest), `baselines/` (panel, failure modes,
`runs.json`), `performance/` (step time, scaling, memory, wall-clock, benchmark CI plot,
`performance.json`), `report.md` and, with `--html`, the self-contained `index.html` (its "3-D
systems" section interleaves every static row with its interactive viewer; `report.md` keeps
`gallery_3d.png`). `--no-interactive` skips the viewers; `--interactive plotly` uses the plotly
mode of §2.3 (then `index.html` loads plotly.js from its CDN). `--refine` (default) adds the
edge-refined column of every 3-D system (`--no-refine` skips it); `--steps-3d-multiplier`
(default 2) is the 3-D step floor at `--budget ≥ 0.2`; the step counts and the refinement
captions are printed per system.

### Figures for a finished run (on the server)

```bash
nefi run configs/nv_relaxometry_paper.yaml --device cuda --out runs/nv_paper   # result.pt, config.yaml
python examples/visualize_result.py runs/nv_paper --html                        # → runs/nv_paper/figures
python examples/visualize_result.py runs/x/result.pt --instance thermal_tomography \
    --config configs/thermal_tomography_paper.yaml --seed 0 --out runs/x/figs
python examples/visualize_result.py runs/x/result.pt --gt runs/x/gt.pt --measurement runs/x/meas.pt
```

For a `nefi run` directory the ground truth and measurement are **regenerated deterministically**
from the saved instance config, seed and scene; `--gt` / `--measurement` load `torch.save`d tensors,
`{name: tensor}` dicts or `Measurement` objects instead. Output: `compare*.png`, `field*.png`
(3-D: mosaic, orthoslices, projections, isosurfaces), `history.png`, `stages.png`, `wallclock.png`,
`measurement.png`, `fit.png`, `metrics.json`, and `report.html` with `--html` (3-D results: with the
interactive viewer, also written as `viewer.html`; `--interactive plotly|none`).

---

## 6. Limitations

* 3-D fields are shown as slices, cross-sections, projections, isosurfaces (marching cubes;
  without scikit-image the static figures fall back to voxel scatters) and the interactive
  viewer; there is no ray-cast volume rendering. The viewer is a software rasterizer: smooth up to
  ≈ 50k triangles per panel, slower for very large or noisy level sets at 64³ (interaction frames
  render at 1× device pixels, refined when idle). The volume-matched level assumes the GT set is
  the right size; it fixes the soft-edge bias, not a wrong shape.
* The anomaly centroid follows the *dominant* anomaly (largest total deviation); a second defect
  elsewhere shows up in the mosaic and the projection but not in the cross-sections. Extended
  "anomalies" (a skull, a filament network) are not outlined on projections (too cluttered).
* Signed-data detection is a threshold (the minority sign must reach 10 % of the majority);
  genuinely signed data with a weaker second polarity need `measurement_cmap="signed"` /
  `field_cmap="signed"`.
* Reports are static HTML except for the 3-D viewers; 2-D figures are not interactive (no hover
  read-outs, no zoom). The viewer's wheel zoom captures the wheel over its canvases.
* CPU memory is reported as the autograd saved-tensor volume, not the process peak; peak memory is
  exact on CUDA only.
* Measurement layouts without metadata rely on shape heuristics and the built-in hint table; new
  instances with ambiguous shapes should set `viz_hints` or `meta["layout"]`.
* The gallery runs instances sequentially in one process; a hard crash (segfault) or a hang in
  data generation cannot be isolated (the solve itself is bounded by `time_budget_s`).
