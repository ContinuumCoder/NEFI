# Geometry is the prior: adaptive representations for physics-faithful inversion

`nefi` solves inverse problems by optimizing a *field* — a parameterization $x = f_\theta$ of the
unknown — through a hard, differentiable forward operator $F$. This page is about the first half of
that sentence. Its thesis, distilled from the two papers `nefi` generalizes, is simple:

> **The representation is the geometric prior.** Choosing how the unknown is parameterized decides
> which image-space updates gradient descent can make, how fast each spatial frequency converges,
> and what the reconstruction can and cannot contain. Each physical system therefore calls for a
> representation whose *geometry* matches the unknown and whose *spectral filter* matches the
> operator's decay — and when we do not know which, the measurement itself can tell us.

The code lives in two packages:

* `nefi.fields.geometric` — a family of representations: composite (background ⊕ anomaly),
  layered (stratified media), explicit shapes, coordinate warps, Fourier bases and Fourier-domain
  preconditioning;
* `nefi.fields.adaptive` — the adaptivity: residual- and operator-driven annealing, capacity
  growth, measurement-driven model selection, and a spectral *match report* that explains, in
  words, whether a representation fits a system.

---

## 1. What the two papers teach

| | **NeTMY** (NV relaxometry, arXiv 2605.13988) | **NeFTY** (thermal tomography, arXiv 2603.11045) |
|---|---|---|
| unknown | sparse spin sources $\rho \ge 0$ (2-D) | piecewise-constant defects in a (layered) bulk, $\alpha \in [\alpha_{\min}, \alpha_{\max}]$ (3-D) |
| operator decay | $\lvert\widehat{G_{az}}\rvert^2 \le C (1+k)^m e^{-k z_0}$ (Lemma 1) | $\sigma_n \lesssim n^{-1/3}$ (Prop. 2), noise amplified by $1/\sigma_n$ (Cor. 1) |
| representation | coordinate MLP, **gated softplus** head (Eq. 5), annealed Fourier features, multiscale | coordinate MLP, **bounded sigmoid** head (Eq. 6), annealed Fourier features |
| explicit prior | $\ell_1$ + anisotropic TV (Eq. 3) | isotropic TV (Eq. 22) |
| failure analysed | free-density solvers *center-collapse* (P2)+(P3), §5.3 | grids ring at interfaces, PINNs collapse; spurious back-face artifacts for shallow defects (App. G.7) |

Both papers use one representation each and tune it to one geometry: the gate lets NeTMY drive
the background to zero so that *points* can emerge; the sigmoid bracket plus TV lets NeFTY carve
*piecewise-constant regions*. Neither paper represents what is actually known about its scenes —
that NeFTY's bulk is *stratified* (App. E.1: three to four strata along $z$), that defects are
*compact bodies* with smooth boundaries, that sensitivity decays with *depth* (App. B.4) — except
through a generic MLP plus penalties. That gap is what this package fills.

## 2. The filtering view

**Lemma 2 (NeTMY, App. D.6).** A first-order step $\theta \leftarrow \theta - \eta \nabla_\theta
L(f_\theta)$ realizes, to leading order, the image-space update

$$
\Delta x \;=\; -\eta\, J_\theta J_\theta^{\top} \nabla_x L \;+\; O(\eta^2) \;=\; -\eta\, G_\theta \nabla_x L,
\qquad J_\theta = \partial f_\theta / \partial\theta .
$$

$G_\theta$ is positive semidefinite and projects every update onto the span of the Jacobian
columns $\psi_p = \partial f_\theta / \partial \theta_p$:
$(G_\theta g)(r) = \sum_p \psi_p(r)\langle \psi_p, g\rangle$ (Eq. 34), with
$\operatorname{rank} G_\theta \le \min(|\Omega|, P)$ (Eq. 35). The representation *filters* the raw
gradient:

| representation | $G_\theta$ | consequence |
|---|---|---|
| free grid (`GridField`) | $I$ | executes $\nabla_x L$ verbatim — center bias (P2), noise at all frequencies |
| MLP + annealed Fourier features | smooth kernel, band $B_\beta \sim 2^{k_\beta}\pi$ (Eq. 37) | low-pass early, richer later |
| Fourier basis (`FourierBasisField`) | $\Phi\Phi^\top$, an ideal projector | hard bandwidth, no noise above the cutoff |
| Gaussian splats | rank $\le K(2d + C)$, localized | moves primitives, cannot fill dense regions |
| composite $x = c(f_1,\dots,f_n)$ | $\sum_i D_i G_i D_i^\top$, $D_i = \partial c/\partial f_i$ | each component contributes *its own* filter |
| warp $x(u) = f(\varphi(u))$ | $G_f(\varphi(u), \varphi(u'))$ | local bandwidth $B\,\lvert\varphi'(u)\rvert$: stretch where you need detail |
| spectral preconditioner $x = P f_\theta$ | $P\, G_\theta P^\top$ | reshapes the per-frequency step (below) |
| shapes / layers / level sets | columns concentrated on interfaces | moves *boundaries*, not pixels |

**Per-frequency dynamics.** Linearize the data term around the solution of a
translation-invariant problem with operator transfer $a(\nu)$ (amplitude sensitivity
$\sigma_F(\nu) = |a(\nu)|$) and representation transfer $g(\nu)$. The error at frequency $\nu$
evolves as

$$
\hat e_{t+1}(\nu) \;\approx\; \bigl(1 - \eta\, g(\nu)\, \sigma_F(\nu)^2\bigr)\, \hat e_t(\nu).
$$

Two things decide whether a representation fits a system:

1. **Band.** The data constrain frequency $\nu$ only if the signal it carries exceeds the noise,
   $\sigma_F(\nu)\,|\hat x(\nu)| \gtrsim \sigma_\varepsilon$ (a Picard condition; NeFTY Cor. 1 is
   its negative: above that band the pseudo-inverse amplifies noise by $1/\sigma_n$). A
   representation whose pass band is *narrower* than the resolvable band is **over-bandlimited**
   (it cannot express what the data resolve); one whose pass band is *wider* is
   **under-bandlimited** (it moves unconstrained frequencies and fits noise unless something —
   annealing, TV, early stopping — stops it).
2. **Conditioning.** Within the resolvable band, the spread of $g\,\sigma_F^2$ sets how uniformly
   detail converges. A grid inherits the operator's decay ($g \equiv 1$); an MLP adds its spectral
   bias (implicit regularization, slower fine detail); a spectral preconditioner with
   $g \approx 1/\max(\sigma_F, \text{floor})$ flattens it (a floored Gauss–Newton step).

`nefi.fields.adaptive.match_report(field, problem)` measures both — $g$ from rows $G_\theta e_i$
(a vjp then a jvp through the field, Fourier-transformed and radially averaged over random
pixels), $\sigma_F$ from rows $J^\top J e_i$ — and states the verdict in words (§5).

## 3. Decision guide

Frequencies are in cycles per unit *normalized* coordinate (each axis spans $[-1,1]$; a grid of
$n$ cells has Nyquist $n/4$). Instance names refer to `nefi.instances`.

| geometry of the unknown | representation | annealing / curriculum | regularizers | example instances |
|---|---|---|---|---|
| **sparse point-like sources** | `NeuralField` + `GatedSoftplus` (NeTMY) — or `AnomalyField(FourierBasisField, gated NeuralField, mode="add")` when a smooth background is present | annealed Fourier features, multiscale (NeTMY Tab. 6); `ResidualDrivenAnnealing` for unknown difficulty | $\ell_1$ + anisotropic TV | `nv_relaxometry`, `poisson_source`, `deconvolution` (sparse scenes) |
| **piecewise-constant defects in a homogeneous bulk** | `NeuralField` + `Bounded` (NeFTY), or `AnomalyField(constant background, level set / StarShapeField, mode="blend", inclusion_bounds=...)` | frequency annealing over the first 25 % (NeFTY Tab. 5); level-set sharpening via progress | isotropic TV; `ContourRegularizer` (perimeter) | `thermal_tomography` (homogeneous), `eit`, `sparse_view_ct` |
| **stratified media ⊕ defects** (laminates, geology) | `LayerCakeField` = `LayeredField` ⊕ gated anomaly (`mode="add", contrast=-1`); `mode="blend"` with a `StarShapeField` anomaly for compact voids | `progress_map={"background": ("fast", 0.5), "anomaly": ("delay", 0.1)}` — layers first, defects second | small TV; `render="area"` gives sub-voxel interfaces | `thermal_tomography` (layered), `wave_fwi`, `darcy_flow` |
| **few compact inclusions with smooth boundaries** | `StarShapeField` (2-D, or 3-D extruded) / `PolygonField`; capacity pool + `GrowCapacity` when the count is unknown | sharpness $\varepsilon$(progress) geometric decay | `ContourRegularizer(perimeter, smoothness)` | `eit`, `diffraction_tomography`, `holography` |
| **smooth background + localized anomaly** | `AnomalyField(FourierBasisField, gated NeuralField, support=...)` with a sensitivity-based support prior | background `"full"`, anomaly delayed; `GrowCapacity` on the Fourier shells | TV / $\ell_1$ on the anomaly | `current_density`, `reaction_diffusion`, `poisson_source` |
| **sensitivity decaying with depth** (surface data of volumes) | `WarpedField(inner, log_depth(...))`, `depth_stretch(γ<1)` or `sensitivity_warp(sensitivity_map(problem))` | `OperatorAwareAnnealing` with `operator_spectrum` | depth-aware TV weight | `thermal_tomography` (App. G.7 failure mode), `nv_relaxometry` (standoff $z_0$) |
| **radial / axisymmetric systems** | `WarpedField(inner, polar(embed="circle"))` / `cylindrical(...)` (inner field takes `ndim + 1` inputs) | as for the inner field | — | `diffraction_tomography`, `holography` |
| **deformations of a known template** | `DeformableField(template)` (identity-initialized displacement) | freeze the template: `Stage(freeze=("inner.",))` | `WarpRegularizer(smoothness, folding)` | registration-type problems |
| **broadband / textured fields** | `NeuralField` at full progress, `GridField` + Tikhonov/TV, `SpectralPreconditionedField` to equalize convergence | `OperatorAwareAnnealing` to cap the band at the noise level | Tikhonov / TV | `deconvolution`, `wave_fwi` (smooth velocity models) |
| **known operator spectrum, slow convergence** | `SpectralPreconditionedField.from_operator(inner, op, domain, floor=...)` | pick the floor from `match_report` ($\nu_{\text{data}}$) | as for the inner field | `deconvolution`, `sparse_view_ct` |

When in doubt: build two or three candidates and let `select_representation` decide (§4.3).

**Binary and level-set representations after the fact.** The two-phase rows of this table can
also be applied *to a finished reconstruction*: `nefi.solve.refine.refine_edges` (or
`inst.refine(result, measurement)`) swaps the representation of one field for a multi-phase level
set (`LevelSetHead` generalized to k nested phases over one φ, with cell-averaged Heavisides for
sub-voxel interfaces and optional *free* phases whose interiors stay smooth), a Cahn–Hilliard
double-well or a binarizing `Binary`-prior head swap, initializes it from the smooth field and runs
a short second stage under the same operator and data. Its report puts the data fit before and
after next to IoU / Edge-F1, refuses refinements the data do not support, and comes with a
`continue` control (the same budget without a prior) to separate the prior's gain from mere extra
optimization. See [Edge refinement](refinement.md) for the numbers on the 3-D instances.

**Learning rates of geometric parameters.** Geometric representations mix a few physical scalars
(layer values and thicknesses, shape centers and radii, inclusion values) with network weights;
the two often want different step sizes. `OptimConfig(lr_mult={prefix: factor})` builds optimizer
parameter groups by qualified name (`field.<name>` / `operator.<name>`, longest prefix wins), and
every stage's schedule is scaled per group. Useful prefixes:

| field | parameter prefixes (inside a problem) |
|---|---|
| `LayeredField` | `field.thickness_logit`, `field.layer_values`, `field.interface_model.`, `field.value_model.` |
| `LayerCakeField` / `AnomalyField` | `field.background.` (the layered medium), `field.anomaly.`, `field.inclusion`, `field.contrast` |
| `StarShapeField` | `field.center`, `field.radius_raw`, `field.coef` (Fourier descriptors), `field.inside`, `field.outside`, `field.depth_center`, `field.half_height_raw` |
| `PolygonField` | `field.vertices`, `field.inside`, `field.outside` |
| `FourierBasisField` | `field.coef` |

Treat the factors as hyperparameters (and compare them with `select_representation`): on the
example of §5, a uniform learning rate of 4·10⁻² for the whole `LayerCakeField` (30.9 dB) beat a
10⁻² network rate with 4–8× on `field.background.` (26–29 dB), because the gated anomaly network
also benefits from the larger step.

## 4. The adaptive tools

### 4.1 Annealing that listens to the data

The papers anneal on a clock: $\beta = K\,t/T$ per stage (NeTMY Eq. 27) or over the first
$T_{FA}$ steps (NeFTY Eq. 21). `nefi.fields.adaptive` offers three alternatives; all drive
`solver.progress_override` (composite fields map the global progress per component).

* **`ResidualDrivenAnnealing(patience, delta)`** — *earn your frequencies.* Opens the next band
  only when the data loss plateaus: the windowed improvement over `patience` steps falls below
  $\max(\delta, \text{rate\_fraction}\cdot\text{peak})$, where *peak* is the best windowed
  improvement at the current level (diminishing returns — scale-free, because a tanh MLP with few
  open bands keeps improving slowly forever). It never opens a band once
  $\text{RMSE} \le \tau\sigma$ (Morozov's discrepancy principle at the stage resolution;
  `stop_at_noise=True` ends the stage there), and ramps each new band in over `ramp_steps` with
  plateau detection paused — switching a band on abruptly feeds the untrained first-layer
  weights of its features and injects a random high-frequency perturbation, the shock the cosine
  gate of NeTMY App. D.2 exists to avoid (an integer staircase of levels loses ~6 dB against the
  smooth clock on `toy1d`). `max_steps_per_level` adds a clock for fixed budgets.
* **`OperatorAwareAnnealing(sensitivity_spectrum, noise_level)`** — opens frequency $\nu$ only
  when $\sigma_F(\nu)\,|\hat x(\nu)| \ge \text{margin}\cdot\sigma_\varepsilon$: the current field's
  radial spectrum (Hann-windowed) is extrapolated above the open band by a power law, multiplied
  by the operator sensitivity (a `Spectrum` from `operator_spectrum` or any 1-D radial profile),
  and compared with the per-mode noise (scaled to coarse stages). `signal="data"` instead fixes
  the cutoff from the measurement spectrum (a data-driven Picard rule).
* **`BandwidthSchedule([(t, B), ...])`** — prescribe the open bandwidth in physical terms;
  `bandwidth_to_progress` / `progress_to_bandwidth` convert for any annealed `FourierFeatures`
  encoding or `FourierBasisField`.

Illustrative single-seed numbers on `toy1d` (64 points, blur σ = 0.02, 1 % noise):

| schedule | budget | PSNR [dB] (bumps / mixed / spikes) | note |
|---|---|---|---|
| fixed clock (β over the whole stage) | 2000 steps | 32.4 / 27.2 / 21.5 | |
| `ResidualDrivenAnnealing(patience=25, ramp_steps=50, rate_fraction=0.25)` | 2000 steps | **37.1** / **28.0** / 20.5 | ends at progress 0.5 on bumps |
| … with `stop_at_noise=True` | stops at 711 / 676 / 2000 | **37.7** / 27.9 / 20.5 | 2.8× fewer steps |
| fixed clock | 400 steps | 21.7 (bumps) | |
| `OperatorAwareAnnealing(signal="field" / "data")` | 400 steps | 25.2 / 26.2 (bumps) | target band ≈ 5–6 cycles/unit |

Holding each level fixed for 300 steps on the same data shows why: levels 3–4 (≈ 4–8 cycles/unit,
where the data spectrum meets the noise floor) reach the noise level with the best PSNR (38.5 dB);
opening all six bands overfits (34.5 dB). The operator-aware target (≈ 5 cycles/unit) lands in that
window.

### 4.2 Capacity growth

Representations with a *capacity pool* implement `can_grow()` / `grow(hint)`:
`FourierBasisField` opens the next shell of modes, `LayeredField` activates the next layer (the new
layer inherits its neighbour's value, so the field is unchanged at the moment of growth),
`StarShapeField` / `PolygonField` activate the next pooled shape — placed at the extremum of the
field-space data gradient $\Delta v\cdot\partial L/\partial x$ (a topological-derivative
heuristic). All pooled parameters exist from the start, so the optimizer the solver builds at
stage start already owns them. `GrowCapacity` performs, on each plateau, the next action of its
`order` (grow a module, open an annealing band) — the representation-level analogue of early
stopping: stay in a small, well-conditioned subspace until the data demand more — and, like the
annealing ladder, adds nothing once the residual is at the noise floor ($\text{RMSE} \le
\tau\sigma$). On a blurred two-disk scene (48², 1 % noise) a pool of three star shapes starting
*empty* gives births at steps 15 and 49, each placed on one of the disks by the gradient hint, and
reaches IoU 0.99.

### 4.3 Let the measurement choose: `select_representation`

There is no training set, but the measurement has many entries and a physics-faithful forward
model predicts *every* entry. Hold out a random subset (via `Measurement.mask`), fit each candidate
on the rest (short budgets suffice to rank), and score the forward prediction on the held-out
entries:

```python
from nefi.fields.adaptive import select_representation, RepresentationEnsemble

report = select_representation(
    lambda field: InverseProblem(domain, field, operator, losses, measurement),
    {"neural": make_neural, "layer_cake": make_layer_cake, "grid": make_grid},
    holdout=0.1, seed=0, budget_scale=0.2,
    curricula={"grid": grid_curriculum},        # per-candidate learning rates
)
print(report.table())                           # ranked: val MSE, train MSE, val/σ², time
ensemble = RepresentationEnsemble.from_report(report)   # softmax(−(s − s_min)/s_min) weights
```

The held-out error estimates the prediction risk $\mathbb E\lVert F(\hat x) - F(x^\star)\rVert^2 +
\sigma^2$: under-regularized representations (free grids, all bands open) fit noise and predict
held-out entries poorly; over-regularized ones cannot express the unknown. With a known noise
level the report also shows $\text{val MSE}/\sigma^2$ — **≈ 1 means the representation predicts
unseen data to the noise floor**. Coarse curriculum stages average only observed *training*
entries (fractional mask weights), so held-out values never leak into the fit; `group_axes`
holds out whole sensor pixels across frames; `n_folds=k` gives k-fold CV. Candidates that fail are
reported with score ∞ instead of aborting the selection.

`RepresentationEnsemble` averages fitted candidates with weights from their held-out losses
(relative temperature by default: a candidate with twice the best loss gets $e^{-1}$ of its
weight) and its `spread()` is a cheap model-uncertainty map.

## 5. The match report in practice

`examples/geometric_representations.py` builds a 2-D *defect in a two-layer medium* (top 1.0 over
an undulating interface, bottom 0.55, elliptical defect 0.1; Gaussian blur of ≈ 1.7 cells, 2 %
noise, data simulated on a 2× finer grid). All four candidates get 600 steps (two-stage
multiscale) and the same losses (MSE + isotropic TV $2\cdot10^{-4}$):

| representation | params | PSNR [dB] | defect IoU | match verdict | $\nu_{\text{rep}}$ | $\nu_{\text{data}}$ | held-out val/σ² |
|---|---|---|---|---|---|---|---|
| `NeuralField` (6 octaves) | 11 777 | 29.2 | 0.93 | over-bandlimited | 1.05 | 4.77 | 2.37 |
| `GridField` + TV | 2 304 | **31.0** | **0.97** | under-bandlimited | 11.3 | 4.77 | 1.15 |
| `LayerCakeField` (layers ⊕ gated anomaly) | 2 926 | 30.9 | 0.96 | **matched** | 4.41 | 4.77 | **1.01** |
| `CompositeField` (Fourier background ⊕ level-set blend) | 2 946 | 27.0 | 0.72 | over-bandlimited | 1.95 | 4.77 | 1.88 |

The layered representation is the only one whose update-kernel pass band matches what the data
resolve; it recovers the physical parameters (layer values 1.04 / 0.552 vs 1.0 / 0.55, mean
interface depth 0.449 vs 0.450) with 2 926 parameters, and held-out cross-validation picks it —
predicting unseen entries at the noise floor. Total-variation-regularized pixels are a strong
baseline for piecewise-constant images under mild blur (the match report flags the grid as
under-bandlimited, and TV is what keeps it in check); the Fourier background cannot hold the sharp
layer boundary (over-bandlimited) and pushes the error into the defect indicator. A typical
report reads:

```text
Representation vs. operator at the noise level (LayerCakeField, progress=1.00, grid 48×48, 6 probes):
- the data resolve frequencies up to ν_data ≈ 4.77 cycles/unit [data spectrum vs noise floor ...]
- the representation passes frequencies up to ν_rep ≈ 4.41 cycles/unit (90% of a white gradient's
  realized update energy through G_θ = J_θ J_θᵀ lies below it; NeTMY Lemma 2).
- verdict: MATCHED (ν_rep/ν_data = 0.92): the pass band of G_θ coincides with the data-resolvable band.
- conditioning on the resolvable band: κ(g·σ_F²) = 1.11e+03 vs κ(σ_F²) = 50.6 for a free grid ...
```

Definitions used by the report: $\nu_{\text{rep}}$ is the radius holding 90 % of the energy of
$G_\theta$ applied to a white gradient (the DC ring is excluded — a bias parameter adds an $N$-fold
constant column); $\nu_{\text{data}}$ is where the Hann-windowed data power spectrum falls to
$(1+\text{margin})\sigma^2$ after half-octave smoothing (when the measurement lives on the field
grid), or the band where $\sigma_F^2 |\hat x|^2 \ge \text{margin}\cdot\sigma^2$ under a power-law
prior otherwise. Sanity checks in the test-suite: a grid's transfer is exactly 1; a 6-mode
Fourier basis concentrates its energy below its $\sqrt2\cdot 5/4$ corner frequency; the measured
$\sigma_F$ of a Gaussian blur matches $e^{-2\pi^2\sigma^2\nu^2}$ to ~1 %.

## 6. Writing a new geometric representation

A representation is a `Field`: it maps normalized coordinates to raw channels, and `Heads` turn
those into named physical fields. About fifteen lines are enough:

```python
import torch
from torch import nn
from nefi.fields import Field, Heads
from nefi.registry import register

@register("field", "slab")
class SlabField(Field):
    """A slab of value v_in between two learnable depths, v_out elsewhere (depth axis -1)."""

    def __init__(self, heads=None, top=-0.3, bottom=0.3, eps_start=0.2, eps_end=0.01):
        super().__init__(heads or Heads({"x": "identity"}))
        self.z = nn.Parameter(torch.tensor([top, bottom]))
        self.v = nn.Parameter(torch.tensor([0.0, 1.0]))          # outside, inside (raw)
        self.eps = lambda p: eps_start * (eps_end / eps_start) ** p

    def raw(self, coords, progress=1.0):                          # (*shape, ndim) -> (*shape, n_in)
        u, e = coords[..., -1], self.eps(progress)
        chi = torch.sigmoid((u - self.z[0]) / e) * torch.sigmoid((self.z[1] - u) / e)
        return (self.v[0] + (self.v[1] - self.v[0]) * chi).unsqueeze(-1)
```

The contract and the conventions that make a representation a good citizen:

* **`raw(coords, progress)`** returns `(*coords.shape[:-1], heads.n_in)` for *any* point set
  (take a fast path on tensor-product grids — `nefi.fields.geometric.interp_grid`, `grid_axes`);
  coordinates are cell-centered in $[-1,1]$.
* **`progress ∈ [0, 1]`** is the continuation parameter: open bands, sharpen interfaces
  (`anneal_value`, `smooth_step`), and give progress 1 its final meaning — the solver evaluates the
  result at the last progress used.
* **`on_stage_start(stage, domain)`** — resample grids, propagate to children.
  **`reset_parameters()`** — restore the initial state (multi-restart); composites must delegate
  to their children explicitly (the default skips sub-`Field`s).
* **Parameters exist from the start.** The solver builds its optimizer at stage start; to grow,
  keep a pool and unmask it (`can_grow()` / `grow(hint)`), ideally without changing the field.
* **Stay forward-differentiable.** Diagnostics and `match_report` compute $G_\theta e_i$ with
  `torch.func.vjp` + `jvp`; avoid ops without forward-mode rules (e.g. `grid_sample` — use
  `interp_grid`), and never name a method `_apply` (it is `nn.Module`'s device-move hook).
* **Check the filter.** `representation_spectrum(field, domain)` and `match_report(field, problem)`
  tell you, before any benchmark, what your parameterization will do to the gradient.

## 7. Limitations and open questions

* Shapes are star-shaped (2-D Fourier descriptors, 3-D by extrusion); topology changes (merging,
  holes), spherical-harmonic 3-D shapes and a principled *death* step are not implemented — the
  capacity pool only grows.
* `LayeredField` interfaces are single-valued height functions over the lateral coordinates
  (no overturned folds or pinch-outs below zero thickness; thicknesses are strictly positive).
* A free-form neural indicator in `mode="blend"` can swap roles with the layers; prefer compact
  shape anomalies there, or an additive gated anomaly.
* The match report assumes approximate translation invariance and isotropy (radial spectra of
  kernel rows away from the boundary). For strongly space-variant operators (surface-only heat
  data) read $\sigma_F$ as an average and use lateral `axes=` spectra; radial averaging hides
  *anisotropic* null spaces (sparse-view CT's missing angles report as fully resolvable). Operators
  without forward-mode AD (e.g. `grid_sample`-based Radon transforms) fall back to
  double-backward or finite-difference Jacobian-vector products (exact for linear operators).
* Spectra of non-periodic data use a Hann window; interference ripples of multi-object spectra
  are smoothed over half-octaves. Both are heuristics with documented knobs (`margin`, `energy`,
  `tol`).
* Representations mixing network weights with geometric scalars are sensitive to learning
  rates; per-group multipliers (`OptimConfig(lr_mult=...)`, §3) help, but good factors are
  problem-dependent — no automatic rule is provided yet.
