# Baselines: parameterizations and solvers

`nefi.baselines` provides the classical comparison family of NeTMY (App. E.2) and NeFTY (App. F.2)
for **every** instance. The design rule is that a baseline changes *only* the parameterization or
the optimizer: domain, forward operator, losses, measurement and post-processing stay identical,
so a benchmark row isolates the prior.

```python
from nefi.baselines import baseline_problem, solve

problem = instance.build_problem(measurement)            # the neural-field problem
bp, cur = baseline_problem(problem, "gaussian_splat")    # grid | lbfgs | admm | gaussian_splat | deep_decoder
result = solve(bp, cur, device="cuda")                   # dispatches ADMM / closed-form solvers
```

## 1. The filtering view (why the parameterization is the prior)

A method that optimizes parameters `θ` of a field `x = f_θ` takes, to first order, the image-space
step (NeTMY Lemma 2, Eq. 7 and App. D.6)

```
Δx ≈ J_θ Δθ = -η J_θ J_θᵀ ∇_x L = -η G_θ ∇_x L,      J_θ = ∂f_θ/∂θ,  G_θ = J_θ J_θᵀ ⪰ 0.
```

The raw field-space gradient `∇_x L` is what the physics produces (for NeTMY's F2 operator it has
an iteration-0 center spike, the (P2)/(P3) pathologies); the *realized* update is that gradient
filtered by the kernel `G_θ` of the parameterization, `rank G_θ ≤ min(|Ω|, P)`.

| method | parameterization | `G_θ` | consequence |
|---|---|---|---|
| grid (Tikhonov / Grid Opt.) | free pixels/voxels | `I` | executes the raw gradient verbatim, including any operator-induced bias; regularization only from explicit penalties |
| ADMM | free pixels + ℓ1/box prox | `I` (x-step) | same geometry as the grid; the prox adds sparsity and exact box feasibility |
| L-BFGS | free pixels | `I`, but step `-H_t⁻¹∇L` | a *parameter-space preconditioner* (curvature pairs), orthogonal to `G_θ`; an independent route out of bad basins |
| Gaussian splats | K anisotropic Gaussians | rank ≤ K(2d+C), columns are localized Gaussians and derivatives | low-rank, strongly *localized* smoother: damps isolated pixel updates, cannot represent dense fields with few primitives |
| Deep Decoder | untrained conv decoder | smooth upsampling patterns | *global low-pass*: strong smoothness bias, weak on point sources |
| neural field (default) | coordinate MLP + annealed Fourier features | smooth, fully coupled, bandwidth grows with β | coarse-to-fine filter (NeTMY Eq. 36-39); the annealing controls its bandwidth |

NeTMY's fixed-budget center-mass ranking under F2 — NeTMY (0.00) < GaussianSplat (0.064) < L-BFGS
(0.081) < ADMM (0.153) < Tikhonov (0.223) — follows this ordering of `G_θ` smoothness. The
library's `nefi.diagnostics` (filter-kernel rows, realized vs. raw update) measures `G_θ` directly.

## 2. The baselines

### Grid (`kind="grid"`, NeTMY "Tikhonov", NeFTY "Grid Opt.")
`GridField` at the final curriculum resolution with the problem's heads (softplus / bounded range
kept) and losses (e.g. TV, optionally an extra ℓ2 term in the instance). Runs the problem's own
multiscale curriculum (the grid is resampled at every stage) with the learning rate multiplied by
`LR_MULT["grid"] = 10` unless `lr=` is given. NeTMY: Adam 5e-3, weight decay 1e-5, clip 1, ℓ2 = TV
= 1e-3, 5000 epochs. NeFTY: same schedule/iterations/TV as the neural field (10 000 iterations).

### L-BFGS (`kind="lbfgs"`, `lbfgs_curriculum`)
```python
lbfgs_curriculum(shape, steps=100, lr=1.0, history=20, max_iter=20, *, n_stages=1, anneal=False)
```
A curriculum with `OptimConfig(optimizer="lbfgs")` (strong-Wolfe line search, constant step
length, no clipping / weight decay). NeTMY: history 20, Wolfe line search, 100 outer iterations,
used only in the mechanism analysis. Each outer step may evaluate the objective up to `max_iter`
times — compare budgets in function evaluations, not steps.

### ADMM (`kind="admm"`, `ADMMSolver`, `ADMMConfig`)
Variable splitting `x = z` for `min D(x) + λ1‖x‖₁ + ι_[lo,hi](x)` (Boyd et al. 2011):
x-step = `n_inner` Adam steps on `D(x) + (μ/2) mean(x − z + u)²` (warm-started), z-step =
soft-threshold at `λ1/μ` then clip to `[lo, hi]`, scaled dual `u += x − z`; stop when
`‖x − z‖_∞ < tol` (optionally also the dual residual `< dual_tol`) or after `max_outer` cycles.
`D` is the problem's `LossSet` restricted to its data terms (`keep_terms` adds others). The box
defaults to the range of the original head (softplus → `[0, ∞)`, `Bounded(lo, hi)` → `[lo, hi]`),
the grid uses an `Identity` head and starts at the head's initial value; the result is `z`.

NeTMY defaults (the `ADMMConfig` defaults): `μ = 1e-3`, `λ1 = 1e-2`, Adam lr 5e-3, 30 inner × 200
outer, `tol = 1e-3`. Because nefi's losses are *means* over pixels, the prox threshold is `λ1/μ`
per pixel; with the paper's `μ` this is 10, so `z` stays at zero until the dual has accumulated —
it converges, slowly. The instances therefore use `μ = 0.1`, `λ1 = 1e-3`, lr 2e-2 and
`adaptive_mu=True` (residual balancing, Boyd §3.4.1). With large `μ`, set `dual_tol` too: the
primal residual alone stops as soon as `x` is pinned to `z`.

When a curriculum is passed explicitly (`ADMMSolver(problem, cfg, curriculum=cur)` or
`solve(bp, cur)`), its final resolution is used and `max_outer = ceil(cur.total_steps / n_inner)`
(`budget_from_curriculum=True`), so `Curriculum.scaled()` shrinks ADMM like every other method.
History: per inner step `total`, `data_loss`, `penalty`, data terms; per cycle `objective`
(`D(z) + λ1 mean|z|`), `data_z`, `primal_residual`, `dual_residual`, `mu`.

### Gaussian splats (`kind="gaussian_splat"`, `GaussianSplatField`, `SplatControl`)
```
raw(x) = b + Σ_k a_k Π_i exp(−(x_i − μ_ki)² / (2σ_ki²)),   σ = σ_min + exp(s),  a = softplus(α)
```
Axis-aligned anisotropic primitives in normalized coordinates (1-D/2-D/3-D), heads applied after
the sum — resolution-free, so it runs multiscale curricula unchanged. Rendering is separable on
tensor-product grids and **cell-averaged** (erf closed form; mass-conserving across resolutions,
gradients alive for sub-pixel primitives); arbitrary point sets use chunked point sampling.
`heads="auto"` maps positivity heads (softplus/exp/gated softplus) and `Bounded(0, hi)` to
`Identity` because the splat density is already non-negative (the upper bound is then not
enforced); `amplitude="signed"` + `heads="keep"` models bounded fields around a background.

`SplatControl` (attached through `problem.meta["callbacks"]`) runs every `every` steps until
`stop_fraction` of the budget: **prune** tiny (`< prune_rel · max a`) or escaped primitives,
**split** primitives with high average positional gradient and width `> split_sigma_cells` cells
into two mass-preserving children (`σ/1.6`, positions sampled from the parent), **clone** small
ones (half the mass moved one σ along the descent direction), optional **merge**. The field has a
fixed capacity (`max_primitives`) with an `active` mask, so all edits are in place and the
solver's optimizer keeps working (freed slots are reused longest-idle first). NeTMY: K = 64
initial, cap 128, Adam 1e-3 for 400 iterations; `LR_MULT["gaussian_splat"] = 10`.

### Deep Decoder (`kind="deep_decoder"`, `DeepDecoderField`)
Fixed random latent `(c, *latent)` → `n_stages` × [1×1 conv → (bi/tri)linear upsample → ReLU →
channel norm] → 1×1 conv to the heads' channels (Heckel & Hand 2018). Stage sizes grow
geometrically from the latent to the native `shape` (exactly ×2 when `shape = latent · 2^n`);
other query grids resample the native output, so `raw(coords)` always matches the coordinate grid.
Channel norm is computed per call (no running statistics). NeTMY: 5 stages, widths
`[128]*4 + [1]` (here: `width=128` hidden channels, output = `heads.n_in`), Adam 1e-3, 5000 steps.

### Closed-form references (`DirectSolver`, `problem.meta["solver"] = "direct"`)
FBP (`sparse_view_ct`, Kak & Slaney ramp / Shepp-Logan / cosine / Hann filters) and Wiener
deconvolution (`deconvolution`, SNR by the discrepancy principle) are packaged as `Result`s with
the same metrics, history keys (`total`, `data_loss`) and post-processing. Their problems carry a
`GridField` initialized to the reconstruction and a 1-step, lr-0 curriculum, so even the plain
`Solver` returns the closed-form answer.

## 3. Running baselines: the dispatch protocol

`baseline_problem` returns `(problem, curriculum)` like every `Instance.baselines()` builder.
Problems that need more than a gradient curriculum say so in `problem.meta`:

| key | meaning |
|---|---|
| `meta["solver"]` | a name registered under the `"baseline"` registry kind (`"admm"`, `"direct"`) |
| `meta["solver_config"]` | its configuration (`ADMMConfig`, ...) |
| `meta["callbacks"]` | callbacks the method requires (e.g. `SplatControl`) |
| `meta["baseline"]` | kind name for reports |

`nefi.baselines.solve(problem, curriculum, **solver_kw)` (used by `nefi bench` / `nefi run
--baseline`) dispatches on `meta["solver"]`, falls back to `nefi.solve.Solver`, and appends the
requested callbacks. Every baseline solver has the `Solver` call shape
`Cls(problem, config, *, curriculum=None, device, dtype, callbacks, seed).run() -> Result`.

## 4. When is a comparison fair?

* **Same objective.** Baselines share the problem's losses; compare parameterizations at equal
  regularization weights (tune per method only when you report that you did, e.g. ADMM's `μ`).
* **Same data, independent simulator.** All instances generate data with an independent
  discretization (`DataGenerator.fidelity_tag` ≠ `operator.fidelity_tag`); the bench asserts it.
* **Budget parity.** Report steps *and* wall-clock: L-BFGS evaluates the objective several times
  per step; ADMM's budget is `n_inner × max_outer`; splat/decoder steps are more expensive than
  grid steps. `Curriculum.scaled()` scales all methods consistently.
* **Learning rates.** Adam steps are in parameter units, so each parameterization needs its own LR
  (`LR_MULT`, instance `*_lr` fields). A baseline at the neural-field LR is not a fair baseline.
* **Regime.** The priors differ in kind: on isolated point sources with a well-conditioned blur a
  pixel-sparse prior is hard to beat, on piecewise-constant CT phantoms TV + a smooth `G_θ` wins,
  and on operators with a biased raw gradient (NeTMY F2) the filtering kernel is decisive. Report
  per-class results rather than one average.

Example (`python examples/baselines_comparison.py`, `deconvolution/sparse_dots`, 32², σ = 1.3 px,
1 % noise, shared MSE + ℓ1 (1e-2) objective, 450 steps, CPU, seed 0; blurred input 23.3 dB):

| method | PSNR [dB] | SSIM | time [s] | parameters |
|---|---|---|---|---|
| neural field | 30.25 | 0.961 | 4.3 | 11 265 |
| grid | 33.37 | 0.974 | 0.4 | 1 024 |
| deep decoder | 28.80 | 0.942 | 5.3 | 5 633 |
| gaussian splat | 32.06 | 0.969 | 2.5 | 320 (≤ 64 primitives) |
| ADMM (ℓ1 + box) | 30.70 | 0.968 | 0.6 | 1 024 |
| Wiener (closed form) | 27.56 | 0.838 | 0.0 | 0 |

Isolated dots under a benign Gaussian blur are the natural regime of pixel/primitive-sparse priors
(and the neural field's smooth `G_θ` spreads mass), so this table is a sanity check of the
baselines, not a claim about the neural field.

**Explicit vs. implicit priors.** With a well-tuned explicit prior (TV) the free grid is the MAP
estimate of the objective and a strong baseline on the three linear imaging instances
(`deconvolution`, `sparse_view_ct`, `poisson_source`: grid + TV ≥ neural field at equal budget,
see the instance pages). Remove the explicit prior and the parameterization becomes the only
regularizer: at 32², 450 steps, seed 0, *no TV*:

| instance (no TV) | neural field | grid |
|---|---|---|
| deconvolution, 3 % noise | 29.39 dB / SSIM 0.939 | 22.99 dB / 0.675 (fits the noise) |
| sparse_view_ct, 12 views | 24.37 / 0.798 | 24.49 / 0.830 |
| poisson_source, no ℓ1 | 22.20 / rel. err 0.533 | 23.09 / 0.481 |

For CT and the Poisson source the grid is itself implicitly regularized by *early stopping*: the
data gradient `Aᵀr` of a smoothing operator is smooth, so gradient descent on a grid from a
constant start is a Landweber iteration whose semi-convergence acts as a low-pass filter. Budgets
therefore matter for grids too — report them.

## 5. API summary

```python
GaussianSplatField(ndim, heads=None, n_primitives=64, max_primitives=128, init="grid",
                   init_sigma=None, init_amplitude=0.1, min_sigma=1e-3, amplitude="softplus",
                   background=None, learn_background=False, render="area",
                   pos_lr_mult=1.0, scale_lr_mult=1.0, amp_lr_mult=1.0, chunk_size=65536)
SplatControl(every=100, start=100, stop_fraction=0.8, grad_quantile=0.8, grad_threshold=0.0,
             split_sigma_cells=2.0, split_factor=1.6, prune_rel=0.01, prune_abs=0.0,
             prune_outside=True, merge_cells=None, max_new_fraction=0.5, min_primitives=1)
DeepDecoderField(shape, heads=None, n_stages=5, width=128, channels=None, latent_shape=None,
                 min_latent=4, upsample="linear", out_init_scale=0.1, latent_scale=0.1,
                 norm_eps=1e-5, seed=0)
ADMMConfig(mu=1e-3, l1=1e-2, lo=0.0, hi=None, lr=5e-3, n_inner=30, max_outer=200, tol=1e-3,
           dual_tol=None, min_outer=1, shape=None, field=None, keep_terms=(), weight_decay=0.0,
           grad_clip=1.0, reset_optimizer=False, adaptive_mu=False, balance=10.0, tau=2.0,
           return_variable="z", time_budget_s=None, budget_from_curriculum=True)
ADMMSolver(problem, config=None, *, curriculum=None, device="auto", dtype=torch.float32,
           callbacks=(), seed=0).run() -> Result
DirectSolver(problem, config=None, *, curriculum=None, ...).run() -> Result
lbfgs_curriculum(shape, steps=100, lr=1.0, history=20, max_iter=20, *, n_stages=1,
                 anneal=False, min_size=8, **stage_kw) -> Curriculum
baseline_problem(problem, kind, *, curriculum=None, lr=None, steps=None, heads="auto",
                 density_control=True, control=None, admm=None, lbfgs_history=20,
                 lbfgs_max_iter=20, name=None, **field_kw) -> (InverseProblem, Curriculum)
solve(problem, curriculum=None, **solver_kw) -> Result
```
Registry: fields `gaussian_splat`, `deep_decoder`; baselines `admm`, `direct`; callback
`splat_control`.
