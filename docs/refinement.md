# Edge refinement: crisp interfaces under the same physics

A neural-field reconstruction of a piecewise-constant unknown — defects in a slab, inclusions in
tissue, an object in air — usually finds the structures but draws them with **soft edges and
reduced contrast**, so thresholded comparisons with the ground truth look worse than the
reconstruction deserves. `nefi.solve.refine` adds a *second, short* optimization stage that
starts from the smooth reconstruction and sharpens its interfaces **under the same forward
operator and the same data**, then reports — with the data fit before and after — whether the
sharper answer is still consistent with the measurement.

```python
import nefi
from nefi.solve.refine import refine_edges

result = nefi.invert(problem)                                  # the smooth reconstruction
refined, report = refine_edges(problem, result, levels=(0.01, None), gt=gt)
print(report.to_markdown())                                    # fit, phases, IoU / Edge-F1 before → after

# instances pick the phase values from their configuration:
refined, report = inst.refine(result, measurement, gt=gt)
```

The refined field is a **separate** `Result` (`refined.extra["refined_from"]` names its source);
nothing is ever applied silently, and a refinement that fails the acceptance test is *refused*:
the smooth result is returned and the report says why.

---

## 1. Why edges soften

Three mechanisms, each visible in the diagnostics of `docs/geometry.md`:

1. **Smoothing forward maps.** Blur, diffusion (heat, diffuse light), band-limited sensors and
   surface-only data all have decaying singular values (NeFTY Prop. 2: σ_n ≲ n^{-1/3} for the heat
   equation). An edge is broadband; the frequencies that make it sharp are exactly the ones the
   data stop constraining first (the Picard condition of `docs/geometry.md` §2). Any
   reconstruction that fits the data equally well with or without them is free to leave them out.
2. **The prior prefers gentle transitions.** A coordinate MLP with annealed Fourier features
   filters every gradient step through `G_θ = J_θ J_θᵀ` (NeTMY Lemma 2) — a low-pass kernel whose
   band opens only gradually; isotropic TV with a small weight and a smoothing `ε` (NeFTY Eq. 22)
   tolerates ramps; quadratic smoothness penalties (Laplacian, gradient-L2) actively prefer them.
3. **Finite budgets.** Low frequencies converge first (NeTMY (P1)); a gallery-budget run stops
   while the fine bands are still converging — at `--budget 0.1` the 3-D smoke runs of `ct3d`,
   `deconvolution3d` and `photoacoustic3d` stop at χ = RMSE/σ ≈ 1.7–3.2, i.e. *before* they reach
   the noise floor.

A fourth effect makes evaluation subtle: **model error**. Benchmarks simulate data with an
independent, finer model (the inverse-crime guard), so even the ground truth does not fit the data
perfectly under the inversion operator — and a converged smooth reconstruction can fit it
*better* than the truth by absorbing the model error into soft structure (the data-fit paradox,
NeFTY App. G.2). `examples/refine_edges.py` prints this "fit of the ground truth" next to every
row (§6).

## 2. The refinement stage

`refine_edges(problem, result, mode=...)` builds a private copy of the problem in which **only the
representation / prior of one field changes**: the operator (its nuisance parameters frozen), the
measurement and the data terms are shared; quadratic smoothness terms (`Laplacian`, `GradientL2`)
are dropped, every other term (TV, ℓ1, …) is kept. One curriculum stage at the result's resolution
runs a few hundred Adam steps; the sharpening schedule (ε, penalty ramp) progresses over the first
`sharpen_fraction` (60 %) of them and the rest runs at the final sharpness.

### `mode="levelset"` (default)

A multi-phase level set over **one** function φ (the multilayer level set of Chung & Vese, 2005):

$$
x \;=\; v_0 + \sum_{j=1}^{k-1} (v_j - v_{j-1})\, H_\varepsilon(\varphi - c_j),
\qquad \varepsilon:\ \varepsilon_{\text{start}} \to \varepsilon_{\text{end}},
$$

the nested super-level sets of φ are the phases; two phases give the `LevelSetHead` formula
`lo + (hi − lo) σ(φ/ε)`.

* **Initialization from the smooth result.** Value-space thresholds `τ_1 < … < τ_{k−1}` (Otsu /
  multi-Otsu, mass- or volume-matched, midpoints of given values, or explicit — §3) and phase values
  `v_j` give φ₀ = (x_s − τ₁)/(v_{k−1} − v₀) and level offsets c_j = (τ_j − τ₁)/(v_{k−1} − v₀): φ₀
  changes by ≈ 1 across an interface, so `ε_start = 0.25` reproduces the smooth transition (unit
  slope at the threshold) and `ε_end = 0.02` makes it sharp.
* **Partial volumes** (`render="area"`, default). Ground truths are cell averages, so a crisp
  interface still has fractional boundary voxels. The head evaluates the *cell average* of the
  Heaviside over each voxel, using the local φ-range `h = Σ_d |∂_d φ|` (stop-gradient):

  $$\frac1h\int_{\varphi-h/2}^{\varphi+h/2}\sigma(t/\varepsilon)\,dt
  = \frac{\varepsilon}{h}\Big[\operatorname{softplus}\tfrac{\varphi+h/2}{\varepsilon}
  - \operatorname{softplus}\tfrac{\varphi-h/2}{\varepsilon}\Big]
  \;\xrightarrow{\varepsilon\to0}\; \operatorname{clip}\!\big(\tfrac{\varphi}{h}+\tfrac12,0,1\big),$$

  a one-voxel ramp whose position is differentiable at sub-voxel resolution. On `dot3d` this is
  worth 1.7 dB over point sampling (`render="point"`: 31.4 vs 29.7 dB, same data fit).
* **Phase values** start at robust class statistics (the majority class's median, the extreme
  classes' far quantile — a softened blob undershoots its true value) and are learnable scalars
  (`learn_levels`, in units of the initial contrast, LR multiplier `level_lr_mult`); values known
  from physics can be fixed (`learn_levels=(1,)` learns only phase 1).
* **Free phases** (`free_phases=(j, …)`): the value of phase j stays free per voxel (an extra raw
  channel initialized at the smooth field), `x = Σ_j V_j χ_j` with the phase indicators
  `χ_j = H_ε(φ − c_j) − H_ε(φ − c_{j+1})`. Crisp interfaces around *smooth interiors*: absorbers
  of different amplitudes, filaments of varying brightness, several tissues inside one outline.
* **Representation.** φ is a `GridField` (default, fast: G_θ = I lets every voxel's interface
  move) or a coordinate MLP (`representation="neural"`, warm-started by `warm_steps` regression
  steps on φ₀): smoother, more compact interfaces — on dot3d it gains up to +4 dB on some scenes
  but fails the acceptance test on others where the grid φ passes, so the grid is the default.
* **Perimeter.** The problem's own TV, kept, already penalizes the perimeter of a phase field
  (`TV(x) = (hi − lo) · Per / Vol` on a two-phase field). `perimeter="auto"` adds an explicit
  interface-length term `mean|∇u|`, `u = (x − v₀)/(v_{k−1} − v₀)`, scaled like the penalties below.

### `mode="phasefield"`

The problem continued on a voxel grid (warm-started through the head's inverse, e.g. the logit of
a `Bounded` head) with a Cahn–Hilliard-style multi-well penalty

$$
W(x) = \prod_j \Big(\frac{x - v_j}{v_{k-1} - v_0}\Big)^2 \qquad(\tfrac1{16}\ \text{mid-gap for two phases}),
$$

whose weight ramps geometrically from 1 % to 100 % (`well_ramp`). The field stays continuous but
is pushed to the phases wherever the data do not hold it in between.

### `mode="tv_sharpen"`

The cheapest option: the problem continued with a stronger isotropic TV and — when the head is
bracketed (`Bounded`, `LogBounded`) and there are two phases — a *binarizing head swap*: the head
becomes the `LevelSetHead(v₀, v₁)` of the `Binary` prior (`nefi.priors.Binary`), warm-started at
the smooth field (logit of the normalized value), with ε shrinking from 1 to 0.1 (`swap_eps`). The
swap binarizes the *current* state: it suits a converged smooth solution; from an under-converged
one the values would need to move continuously, which a saturating sigmoid resists.

### `mode="continue"` — the control

The same grid continuation and budget with the problem's own head and losses, **no** sharpening
prior. A smooth result stopped early improves under *any* continuation (a free grid converges the
bands the MLP had not reached yet), so this row is what every prior must beat before it gets the
credit. `examples/refine_edges.py` always prints it.

### How strong is the added prior?

Penalties (double well, extra TV, `perimeter="auto"`) are scaled so that their energy at the smooth
solution is `γ ×` a reference data loss, with `γ = fit_tol² − 1` (0.21 for `fit_tol = 1.1`). A
descent started at a converged smooth solution can then raise the data loss by at most that much —
`L_data ≤ (1 + γ) L_data(x_s)`, i.e. the RMSE by at most `fit_tol`, the tolerance of the acceptance
test. The reference is the smooth solution's data loss brought down to the noise floor
(`× 1/χ²`) when the smooth run stopped early (otherwise the penalty would become χ² times too
strong once the fit converges). Changes of *representation* (level set, head swap) carry no such
bound — the acceptance test catches them.

## 3. Thresholds and phase values

| helper | what it does |
|---|---|
| `multi_otsu(x, k)` / `otsu_threshold(x)` | globally optimal k-class thresholds of the value histogram (1-D k-means by dynamic programming; thresholds centred in empty gaps) |
| `mass_matched_fraction(x, lo, hi)` | the `hi`-phase volume fraction that conserves the mean of `x` — smoothing operators preserve low moments, so the integral of a soft reconstruction is usually well determined even when its edges are not |
| `volume_matched_threshold(x, f)` | the level whose super-level set holds a fraction `f` of the voxels (`above=False`: sub-level set) |
| `estimate_levels(x, levels, threshold)` | phase values + thresholds: `levels="auto"` (two phases), `k`, or values with `None` entries estimated; `threshold="auto" \| "otsu" \| "midpoint" \| "mass" \| float \| list` |

`levels=(0.01, None)` starts the lower phase at a known value and estimates the upper one (both
stay learnable unless `learn_levels` says otherwise); values that do not bracket the field (a
background given *above* inclusions that lie below it) raise an error instead of producing a
degenerate phase model.

## 4. Reading the report

```text
# Edge refinement — dot3d (levelset)

Verdict: accepted. The refined field explains the data within tolerance (χ 1.149 → 1.102,
limit 1.221; sharpening cost 1.00×) and its interfaces lie within the smooth solution's
transition band (deviation 0 %).

| quantity                                   | smooth           | refined          |
|---|---|---|
| data RMSE                                  | 0.01019          | 0.00987          |
| χ = RMSE/σ                                 | 1.149            | 1.102            |
| χ of the control (continue, same budget)   | —                | 1.11             |
| phase volumes                              | [0.9912, 0.0088] | [0.9882, 0.0118] |
| interface 1: band [core, outer]            | [0.004, 0.0175]  | 0.01175          |
| phase values                               | [0.01, 0.04]     | [0.01001, 0.03925] |
| psnr / iou / iou_vm / edge_f1              | 28.22 / 0.50 / 0.80 / 0.42 | 31.44 / 0.72 / 0.80 / 1.00 |
```

* **Data fit** — RMSE, relative RMSE and, when the noise level is known (scalar or per-entry
  `noise_std`, or `sigma=`), χ = RMSE/σ (whitened: `sqrt(mean((pred − obs)/σ)²)`). χ ≈ 1 is a fit at
  the noise floor.
* **Phase volumes and the transition band** — for each interface, the smooth field's volume above
  its `½ + band` (core) and `½ − band` (outer) contrast levels (`band = 0.25`) and the refined
  field's volume above the midpoint. The smooth reconstruction's soft transition is exactly where the
  data leave the interface position uncertain; the refinement may place the interface anywhere
  *inside* it (volumes are compared on the minority side; a vanished phase has deviation 1).
* **Phase values** before and after (learned ones move; free phases are marked).
* **Metrics** (with `gt=`): PSNR / SSIM, IoU and Dice at the GT threshold (`iou_tau`: a float,
  `"otsu"` of the GT, or `"half_max"`), the **volume-matched IoU** `iou_vm` (the reconstruction
  thresholded at the level that gives the GT mask's volume — shape agreement independent of
  contrast calibration), Edge-F1 (NeFTY App. G.5) and, through `refine_instance`, the instance's own
  metrics (`inst_*`).

### Acceptance rules

A refinement is **accepted** only if

1. **the data fit does not degrade**: `χ_refined ≤ fit_tol · max(χ_ref, 1)` when σ is known — a
   reference that over-fits (χ < 1) may be matched at the noise level — and
   `RMSE_refined ≤ fit_tol · RMSE_ref` otherwise (`fit_tol = 1.1`). The reference is the smooth
   result; when that did **not** reach the noise floor (`χ_smooth > fit_tol`, or σ unknown) it is
   the better of the smooth result and the `continue` control run *internally* with the same budget
   (`reference="auto"`; `"smooth"` / `"control"` force one). Without it an early-stopped start
   makes the test vacuous: on `ct3d` at gallery budget (χ_smooth = 3.2) a two-phase level set
   that cannot represent the skull stays at χ = 3.2, "no worse than the smooth result", while the
   control reaches χ = 1.6 — it is now refused;
2. **sharpening does not cost more than the tolerance**: the final data loss against the best one
   the same stage reached before it sharpened (floored at the noise level),
   `sqrt(L_final / max(L_best, L_noise)) ≤ fit_tol`;
3. **the interfaces stay in the smooth solution's transition band**: every refined phase volume
   lies within `max_volume_change = 30 %` (relative) of the band `[core, outer]`;
4. the learned phase values stay ordered.

Otherwise it is **refused**: `refine_edges` returns the *unmodified* smooth result, the report
says which rule failed, and `report.candidate` still holds the refined field for inspection
(`refuse=False` returns the flagged candidate instead). The defaults are deliberately conservative:
because of model error the ground truth itself misfits several instances by χ ≈ 1.3–1.7 (§6), so
the 1.1 rule can refuse a field that is *closer* to the truth than the smooth one (eit, §6). If you
know your noise or model-error level, pass it (`sigma=`) or relax `fit_tol` — and say so.

Refinement is a prior — a piecewise-constant (or piecewise-smooth) world. A refined field that
explains the data as well as the smooth one is *consistent* with the measurement, not proven by
it; the data-fit rows exist so that nobody mistakes the prior's crispness for information the data
carry.

## 5. API

| call | purpose |
|---|---|
| `refine_edges(problem, result, *, mode, levels, …) -> (Result, RefineReport)` | the refinement stage (`nefi.refine_edges`) |
| `inst.refine(result, measurement, gt=…, **kw)` | instance convenience: `REFINE_DEFAULTS[inst.name]` (phase values from the config) + the instance metrics |
| `refine_instance(inst, result, measurement, …)` | the same as a function (`problem=` reuses a built problem) |
| `refine_run_output(run, instance=…)` | refine a `RunOutput` or a gallery entry / run → a new `RunOutput` (`extra["refine_report"]`, `extra["smooth_result"]`) |
| `refine_method(mode=None, name="neural+refine", **kw)` | a benchmark `Method`: the instance's own solve followed by the refinement (merged history: steps and time include it) |
| `REFINE_DEFAULTS` | `name -> fn(cfg) -> kwargs` for thermal_tomography, ct3d, dot3d, deconvolution3d, photoacoustic3d, eit, deconvolution; an instance may carry its own `refine_defaults` attribute |
| `edge_metrics`, `data_fit`, `transition_band_check`, `estimate_levels`, `multi_otsu`, `volume_matched_threshold`, `mass_matched_fraction`, `MultiPhaseHead`, `DoubleWell` | the building blocks |

The per-instance defaults (chosen on the smoke presets, §6):

| instance | phases | notes |
|---|---|---|
| `thermal_tomography` | `(mean(alpha_defect_range), None)` | defect diffusivity from the config, bulk estimated and learned |
| `ct3d` | 2, phase 1 free | a crisp air / object boundary around a free multi-tissue interior; no binarizing head swap |
| `dot3d` | `(mua_background, mean(inclusion_mua))`, both learned | grid φ: accepted on 4 of 5 smoke scenes (+2.8 to +4.5 dB), refused on the fifth, where it would have hurt; an MLP φ gains up to +4 dB on some scenes but is refused more often |
| `deconvolution3d` | 2 (estimated), phase 1 free | structures of varying brightness on a dark background |
| `photoacoustic3d` | `(0, None)`, phase 1 free, `perimeter="auto"` | absorbers of varying amplitude on a zero background |
| `eit` | 2 (estimated) | inclusion sign unknown (conductive or resistive) |
| `deconvolution` | 2 (estimated), phase 1 free | 2-D objects of varying brightness |

For the gallery (`examples/gallery.py`) the hook is one line per
entry: `refined = nefi.solve.refine_run_output(entry)` — then draw `entry.run.result` and
`refined.result` side by side; `refined.extra["refine_report"].summary()` is the caption.

## 6. Results on the volumetric instances

`python examples/refine_edges.py --budget 0.1 --include-2d --seeds 0,1` (smoke presets, the gallery
budget: 200–600 smooth steps; CPU, one process). Every cell is *seed 0 / seed 1* (two
scenes); the smooth row gives the smooth reconstruction's metrics and, in the data-fit column, its
χ = RMSE/σ (RMSE for the noise-free thermal data) next to **the ground truth's own misfit under the
inversion operator** (model + noise; `fit of GT`). Refinement rows give the value *after*
refinement and whether the default acceptance test passes (✓) or refuses (✗: `refine_edges` would
return the smooth result). `levelset` is the instance default (`REFINE_DEFAULTS`), `levelset-plain`
a piecewise-constant grid level set with the same phase values. Figures (GT | smooth | every mode,
central slices, line profiles through the anomaly) are written next to the table
(`runs/refine_edges/<instance>_seed<k>.png`).

| instance | mode | accepted | PSNR [dB] | IoU (GT τ) | IoU (vol.-matched) | Edge-F1 | data fit |
|---|---|---|---|---|---|---|---|
| **thermal_tomography** | smooth | — | 10.84 / 11.03 | 0.984 / 0.274 | 1 / 0.235 | 0.93 / 0.783 | RMSE 0.101 / 0.243 (GT: 1.38 / 1.17) |
| | `continue` (control) | ✓ / ✓ | 10.65 / 10.12 | 0.984 / 0.234 | 0.969 / 0.196 | 0.908 / 0.816 | RMSE 0.0842 / 0.162 |
| | `levelset` (default) | ✗ / ✗ | 9.922 / 11.09 | 0.969 / 0.253 | 1 / 0.24 | 0.61 / 0.661 | RMSE 0.305 / 0.584 |
| | `levelset-plain` | ✗ / ✗ | 9.922 / 11.09 | 0.969 / 0.253 | 1 / 0.24 | 0.61 / 0.661 | RMSE 0.305 / 0.584 |
| | `phasefield` | ✓ / ✓ | 10.5 / 10.45 | 0.984 / 0.214 | 0.969 / 0.179 | 0.926 / 0.827 | RMSE 0.0844 / 0.165 |
| | `tv_sharpen` | ✗ / ✗ | 10.22 / 10.95 | 0.928 / 0.318 | 1 / 0 | 0.933 / 0.736 | RMSE 0.381 / 1.37 |
| **ct3d** | smooth | — | 27.33 / 28 | 0.638 / 0.607 | 0.66 / 0.631 | 0.871 / 0.828 | χ 3.22 / 3.2 (GT: 1.73 / 1.58) |
| | `continue` (control) | ✓ / ✓ | 30.95 / 31.75 | 0.809 / 0.792 | 0.826 / 0.791 | 0.92 / 0.915 | χ 1.61 / 1.6 |
| | `levelset` (default) | ✓ / ✓ | 31.07 / 31.9 | 0.792 / 0.775 | 0.82 / 0.778 | 0.921 / 0.906 | χ 1.38 / 1.3 |
| | `levelset-plain` | ✗ / ✗ | 25.78 / 26.29 | 0.693 / 0.653 | 0.717 / 0.68 | 0 / 0 | χ 3.17 / 3.18 |
| | `phasefield` | ✓ / ✓ | 30.85 / 31.61 | 0.807 / 0.787 | 0.824 / 0.781 | 0.918 / 0.911 | χ 1.64 / 1.63 |
| | `tv_sharpen` | ✓ / ✓ | 30.94 / 31.73 | 0.806 / 0.791 | 0.824 / 0.79 | 0.92 / 0.911 | χ 1.62 / 1.61 |
| **deconvolution3d** | smooth | — | 25.24 / 28.07 | 0.559 / 0.412 | 0.58 / 0.447 | 0.557 / 0.452 | χ 1.71 / 1.84 (GT: 0.997 / 1.01) |
| | `continue` (control) | ✓ / ✓ | 26.64 / 28.64 | 0.664 / 0.457 | 0.664 / 0.468 | 0.509 / 0.168 | χ 1.24 / 1.35 |
| | `levelset` (default) | ✓ / ✓ | 26.9 / 28.97 | 0.681 / 0.479 | 0.666 / 0.488 | 0.59 / 0.361 | χ 1.06 / 1.13 |
| | `levelset-plain` | ✗ / ✗ | 25.42 / 27.83 | 0.605 / 0.511 | 0.676 / 0.538 | 0.46 / 0.0468 | χ 3.05 / 3.56 |
| | `phasefield` | ✓ / ✓ | 26.68 / 28.55 | 0.663 / 0.455 | 0.658 / 0.468 | 0.513 / 0.0989 | χ 1.24 / 1.35 |
| | `tv_sharpen` | ✓ / ✓ | 26.55 / 28.59 | 0.658 / 0.456 | 0.658 / 0.468 | 0.505 / 0.0989 | χ 1.25 / 1.36 |
| **dot3d** | smooth | — | 28.22 / 24.38 | 0.5 / 0 | 0.796 / 0.689 | 0.421 / 0 | χ 1.15 / 1.12 (GT: 1.45 / 1.21) |
| | `continue` (control) | ✓ / ✓ | 27.98 / 24.05 | 0.543 / 0 | 0.725 / 0.52 | 0.4 / 0 | χ 1.11 / 1.08 |
| | `levelset` (default) | ✓ / ✓ | 31.44 / 27.18 | 0.723 / 0.6 | 0.796 / 0.617 | 1 / 1 | χ 1.1 / 1.08 |
| | `levelset-plain` | ✓ / ✓ | 31.44 / 27.18 | 0.723 / 0.6 | 0.796 / 0.617 | 1 / 1 | χ 1.1 / 1.08 |
| | `phasefield` | ✓ / ✓ | 28.52 / 24.06 | 0.565 / 0 | 0.725 / 0.49 | 0.421 / 0 | χ 1.1 / 1.09 |
| | `tv_sharpen` | ✓ / ✓ | 30.87 / 26.94 | 0.74 / 0.6 | 0.76 / 0.652 | 1 / 1 | χ 1.06 / 1.04 |
| **photoacoustic3d** | smooth | — | 17.93 / 15.44 | 0.667 / 0.558 | 0.672 / 0.559 | 0.951 / 0.901 | χ 2.03 / 2.28 (GT: 1.35 / 1.51) |
| | `continue` (control) | ✓ / ✓ | 20.2 / 17.39 | 0.766 / 0.696 | 0.777 / 0.706 | 0.986 / 0.99 | χ 0.967 / 1.02 |
| | `levelset` (default) | ✓ / ✓ | 21.68 / 19.47 | 0.792 / 0.754 | 0.792 / 0.751 | 0.982 / 0.977 | χ 0.866 / 0.878 |
| | `levelset-plain` | ✓ / ✓ | 19.54 / 18.09 | 0.766 / 0.722 | 0.765 / 0.73 | 0.994 / 0.991 | χ 0.921 / 0.943 |
| | `phasefield` | ✓ / ✓ | 20.29 / 17.71 | 0.769 / 0.689 | 0.775 / 0.71 | 0.986 / 0.992 | χ 0.971 / 1.03 |
| | `tv_sharpen` | ✓ / ✓ | 20.59 / 17.78 | 0.777 / 0.708 | 0.784 / 0.719 | 0.986 / 0.99 | χ 0.982 / 1.03 |

2-D instances (`--include-2d`):

| instance | mode | accepted | PSNR [dB] | IoU (GT τ) | IoU (vol.-matched) | Edge-F1 | data fit |
|---|---|---|---|---|---|---|---|
| **eit** | smooth | — | 23.52 / 25.88 | 0.844 / 0.929 | 0.949 / 0.898 | 1 / 1 | χ 1.01 / 1.06 (GT: 1.33 / 1.62) |
| | `continue` (control) | ✓ / ✓ | 23.65 / 25.4 | 0.864 / 0.893 | 0.949 / 0.931 | 1 / 1 | χ 1.01 / 1.05 |
| | `levelset` (default) | ✗ / ✗ | 24.44 / 29.43 | 0.792 / 0.895 | 1 / 0.931 | 1 / 1 | χ 1.24 / 1.33 |
| | `phasefield` | ✓ / ✓ | 23.9 / 25.93 | 0.864 / 0.893 | 0.949 / 0.931 | 1 / 1 | χ 1.01 / 1.06 |
| | `tv_sharpen` | ✗ / ✗ | 28.2 / 19.62 | 0.949 / 0.786 | 0.949 / 0.931 | 1 / 1 | χ 1.88 / 2.13 |
| **deconvolution** | smooth | — | 28.05 / 24.18 | 0.93 / 0.859 | 0.905 / 0.878 | 0.971 / 0.971 | χ 1.05 / 1.06 (GT: 0.984 / 1.01) |
| | `continue` (control) | ✓ / ✓ | 30.16 / 26.33 | 0.931 / 0.911 | 0.945 / 0.913 | 0.99 / 0.965 | χ 0.934 / 0.972 |
| | `levelset` (default) | ✓ / ✓ | 30.59 / 26.19 | 0.931 / 0.923 | 0.932 / 0.925 | 0.99 / 0.965 | χ 0.947 / 0.981 |
| | `phasefield` | ✓ / ✓ | 30.52 / 26.48 | 0.931 / 0.905 | 0.932 / 0.913 | 1 / 0.965 | χ 0.93 / 0.97 |
| | `tv_sharpen` | ✓ / ✓ | 30.06 / 26.29 | 0.924 / 0.911 | 0.945 / 0.913 | 0.99 / 0.965 | χ 0.938 / 0.977 |

**At convergence.** The same comparison from smooth solutions at the instances' full default budget
(seed 0; `continue` and the default `levelset`, 200 steps each):

| instance (smooth steps) | mode | accepted | PSNR [dB] | IoU (GT τ) | IoU (vol.-matched) | Edge-F1 | data fit |
|---|---|---|---|---|---|---|---|
| **thermal_tomography** (2000) | `continue` | ✓ | 10.33 → 10.46 | 0.984 → 0.984 | 0.969 → 0.969 | 0.671 → 0.716 | RMSE 0.0881 → 0.0835 |
| | `levelset` | ✗ | 10.33 → 9.91 | 0.984 → 0.969 | 0.969 → 0.969 | 0.671 → 0.614 | RMSE 0.0881 → 0.304 |
| **ct3d** (1800) | `continue` | ✓ | 31.14 → 30.98 | 0.809 → 0.799 | 0.817 → 0.829 | 0.925 → 0.940 | χ 1.66 → 1.35 |
| | `levelset` | ✓ | 31.14 → 31.35 | 0.809 → 0.811 | 0.817 → 0.830 | 0.925 → 0.928 | χ 1.66 → 1.27 |
| **deconvolution3d** (1800) | `continue` | ✓ | 26.73 → 26.25 | 0.673 → 0.642 | 0.658 → 0.652 | 0.418 → 0.387 | χ 1.11 → 1.09 |
| | `levelset` | ✓ | 26.73 → 27.19 | 0.673 → 0.698 | 0.658 → 0.678 | 0.418 → 0.611 | χ 1.11 → 1.05 |
| **dot3d** (900) | `continue` | ✓ | 28.43 → 28.02 | 0.565 → 0.609 | 0.760 → 0.725 | 0.421 → 0.400 | χ 1.11 → 1.11 |
| | `levelset` | ✓ | 28.43 → 31.24 | 0.565 → 0.717 | 0.760 → 0.796 | 0.421 → 0.966 | χ 1.11 → 1.08 |
| **photoacoustic3d** (800) | `continue` | ✓ | 21.00 → 21.96 | 0.789 → 0.797 | 0.806 → 0.790 | 0.987 → 0.995 | χ 1.16 → 0.905 |
| | `levelset` | ✓ | 21.00 → 21.97 | 0.789 → 0.804 | 0.806 → 0.804 | 0.987 → 0.989 | χ 1.16 → 0.858 |

**Reading the numbers.**

* **dot3d** — the textbook case for a two-phase prior (a homogeneous inclusion in a known
  background, a strongly smoothing forward map): the level set gains **+3.2 / +2.8 dB** at the
  gallery budget and **+2.8 dB** at convergence, Edge-F1 0.42 / 0 → **1.0**, IoU at the GT
  half-maximum 0.50 / 0 → 0.72 / 0.60, at an *unchanged* data fit (χ 1.15 → 1.10). The control
  gains nothing (−0.2 / −0.3 dB): the improvement is the prior's, not extra optimization. Over five smoke
  scenes the default is accepted on four (+2.8 to +4.5 dB) and refused on the fifth, where every
  variant would have lowered the PSNR. The volume-matched IoU does not always improve (0.80 → 0.80
  / 0.69 → 0.62): DOT cannot separate the inclusion's contrast from its size, and the sharp sphere
  inherits that ambiguity.
* **photoacoustic3d** — free-valued absorbers on a zero background: **+1.5 / +2.1 dB and
  +0.03 / +0.06 IoU over the control** at the gallery budget (the smooth runs stop at χ ≈ 2); at
  convergence the level set ties the control (21.97 vs 21.96 dB). A constant-phase level set is
  worse than the free-phase default (vessels of different amplitudes).
* **deconvolution3d** — filaments are not piecewise constant; with a free foreground the level set
  still adds **+0.3 dB and +0.1–0.2 Edge-F1 over the control** (+0.5 dB, Edge-F1 0.42 → 0.61 at
  convergence). The constant-phase level set is **refused** (χ 1.7 → 3.1 / 1.8 → 3.6).
* **ct3d** — the gallery-budget gain (+3.7 / +3.9 dB) is almost entirely **continuation**: the
  smooth runs stop at χ ≈ 3.2 and the control alone recovers +3.6 / +3.8 dB. The default (a crisp
  air / object boundary around a free multi-tissue interior) adds +0.1 dB over the control and
  loses ≈ 0.02 IoU; at convergence +0.2 dB. A two-phase level set cannot hold the multi-tissue
  phantom and is **refused** (χ stays at 3.2 while the control reaches 1.6).
* **thermal_tomography** — **refused** on both scenes. The smoke preset's coarse implicit time step
  makes the *ground truth itself* misfit the data by 14× / 5× the smooth reconstruction's RMSE: the
  smooth field absorbs the model error into an inhomogeneous bulk, and any two-phase field fits
  2–3× worse. The refusal is correct — the data cannot certify a crisper field here. Note also that
  every *accepted* mode (the control and the phase field) improves the fit but *lowers* PSNR / IoU:
  the data-fit paradox in action; acceptance certifies consistency with the data, not accuracy.
* **eit** — the level set is refused on both scenes (χ 1.01 → 1.24 / 1.06 → 1.33) although it moves
  the PSNR by +0.9 / +3.5 dB: the ground truth's own χ is 1.33 / 1.62, so the default 1.1 rule is
  conservative here (`fit_tol=1.3` or `sigma=` a model-error level accepts it — a decision the
  user has to make and report). TV-sharpen shows why the rule exists: +4.7 dB on one scene,
  −6.3 dB on the other, refused on both.
* **deconvolution** (2-D) — the refinement ties the control (+0.4 / −0.1 dB).

**Cost.** 200 refinement steps take 0.2–8 s per smoke problem on one CPU core (photoacoustic3d's
wave solves are the most expensive); when the smooth start is not at the noise floor the
acceptance test runs the control internally, which costs about as much again.

## 7. Limitations

* **A prior, not information.** The refinement cannot recover what the data do not constrain;
  where the smooth reconstruction is soft because the physics is (deep DOT inclusions, the
  back of a thermography slab, limited-view photoacoustics), the crisp answer is the prior's
  choice within the data's tolerance — the band test only keeps it from leaving the region the
  smooth reconstruction marked as uncertain.
* **Piecewise-constant scenes only (or piecewise-smooth with free phases).** Filaments with
  Gaussian profiles, textured tissue and smooth backgrounds violate the constant-phase model; the
  fit test refuses it (deconvolution3d) — free phases or the `continue` control are the appropriate
  alternatives.
* **Accepted is not better.** The acceptance test certifies consistency with the data, not
  accuracy — it cannot see the ground truth. On toy1d's *smooth* bumps (smoke budget) the two-phase
  level set fits the data better than the control (χ 1.29 vs 3.65) and is accepted, yet it ends
  0.8 dB further from the truth; on thermal_tomography the accepted control and phase field improve
  the fit and lower PSNR / IoU. Choose the prior for physical reasons, and compare with the
  `continue` control when a ground truth is available.
* **Model error.** Under the inverse-crime guard the truth itself misfits the data; a smooth
  reconstruction can absorb model error into soft structure and then out-fit any piecewise-constant
  field (thermal_tomography smoke: the ground truth's RMSE is 14× the smooth one's), and the fit
  test refuses the level-set and TV-sharpen modes. That refusal is correct — the data cannot
  certify a crisper field — but it also means refinement cannot help there without a better
  forward model; conversely the 1.1 rule can refuse a field that is closer to the truth (eit), and
  relaxing it (`fit_tol`, `sigma=`) is the user's documented decision.
* **Nested phases.** One level function orders the phases (v₀ < v₁ < …); a jump from phase 0 to
  phase 2 crosses a thin sheath of phase 1. Unordered multi-material scenes need several level
  functions (not implemented).
* **Level-set motion is local.** On a grid, interfaces move only where the Heaviside has gradient
  (a band of ~4ε plus one cell); a refinement relocates interfaces by a few voxels, not across the
  domain — by design, since the smooth result is the anchor. There is no re-initialization of φ
  to a signed distance.
* **Budget.** 200 grid steps (0.2–8 s per smoke problem on one CPU core), plus about as much
  again for the internal control when the smooth start is not at the noise floor; the neural φ
  costs a 300-step warm start more. The refinement runs at the result's resolution only (no
  multiscale), on one field (others are held fixed).
