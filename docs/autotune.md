# Auto-tuning: detect and fix the common failure modes

The demo gallery (`examples/gallery.py`, smoke presets, 0.1 × the paper budgets) showed four
typical ways a per-measurement inversion goes wrong. `nefi.autotune` detects each of them for
*any* `InverseProblem`, fixes what can be fixed from the data alone, and says clearly what cannot:

| failure | detected by | what changes |
|---|---|---|
| too-conservative budget / learning rate — the fit never reaches the noise floor | `probe_convergence`: doubling budgets against `χ = RMSE/σ` | `tune_budget`: measured budget, learning rate, annealing length, Morozov stop |
| wrong regularization strength | misfit outside `[0.85, 1.1]·τσ` | `tune_regularization`: discrepancy principle (GradNorm shares without σ) |
| an unconstrained gauge (offset, scale, sign, user directions) | `detect_gauges`: directions invisible to the data | `repair_gauges`: mean anchor, scale correction, sign convention, penalty |
| insufficient acquisition geometry | `acquisition_report`: coverage, singular values, data count, match report | **nothing** — it is physics; the report says what would help and which prior fits |

Everything runs on copies (the problem you pass is never modified), every decision is logged
(`nefi.enable_logging()`) and collected in a report.

## One call

```python
import nefi

problem, curriculum, report = nefi.autotune.autotune_problem(problem, level="standard")
print(report.to_markdown())          # decisions table + one section per tuner
result = nefi.invert(problem, curriculum)
```

For a registered instance (generates the measurement, builds the problem, optionally solves the
default and the tuned configuration side by side):

```python
from nefi.instances.eit import EIT

problem, curriculum, report = nefi.autotune.autotune_instance(EIT(n=16, n_patterns=4),
                                                              level="quick", compare=True)
report.comparison["default"]["metrics"], report.comparison["tuned"]["metrics"]
```

On the command line:

```bash
nefi autotune eit --smoke --level standard --compare --out runs/eit-autotune
nefi run runs/eit-autotune/tuned_config.yaml          # reproduces the tuned solve
```

`nefi autotune` prints the report and writes `autotune.md`, `autotune.json` and
`tuned_config.yaml` (instance config + tuned curriculum; the tuned loss weights are stored as
per-stage overrides so `nefi run` picks them up; gauge fixes are applied in-process only and are
listed under `autotune.applied_in_process`). With `--smoke` the smoke preset sets the problem
size; the budget is *tuned*, not capped as in `nefi run --smoke`.

| level | runs | cost (32² smoke problems) |
|---|---|---|
| `quick` | σ, gauges, light acquisition report (k = 4), budget probes 50 → 800 steps, no learning-rate search, no regularization tuning | ≈ 1–2 k probe steps |
| `standard` | + full acquisition report (k = 8, match report), probes up to 1600 steps, learning-rate search when the base rate misses the floor, Morozov regularization | ≈ 1.5–3.5 k probe steps (EIT 23 s, poisson 5 s on a CPU) |
| `thorough` | + held-out Sobol search (8 trials at 30 % of the tuned budget) around the tuned configuration, adopted only if > 5 % better | ≈ 5–10 k probe steps |

`options={...}` overrides entries of `nefi.autotune.LEVELS` (e.g. `{"extend": 1}`).

## The four failure types, measured

Four deliberately under-configured problems (the configurations are in `tests/test_autotune.py`;
the slow test reproduces them), the gallery budget, seed 0, CPU, `level="standard"`:

| instance | failure | what the tuner found | what it changed | before → after |
|---|---|---|---|---|
| `eit` (4 current patterns, 200 steps) | too-conservative budget | χ = 8.8 / 1.86 / 1.38 / 1.08 / 1.04 / 1.01 at 50 … 1600 steps — at the floor from 800 steps, but the reconstruction still changes by 16–18 % per doubling; acquisition *moderately under-determined* (240 data for 256 unknowns) | budget 200 → 1600 (the last validated probe schedule), Morozov stop | PSNR **16.51 → 23.08 dB**, inclusion IoU 0.46 → 0.92, χ 1.38 → 1.03 |
| `wave_fwi` (2 sources × 8 receivers, 100 steps) | insufficient acquisition + non-convergence | acquisition *moderately under-determined*: `σ₈/σ₁ = 0.35` (only 4 of the top 8 modes within ×2 of the best), the low-z / high-z sides 51 % below the best side, representation over-bandlimited; the misfit reaches the floor only at 1600 steps (χ 3.92 → 1.03 at 50 → 1600) | budget 100 → 1600 | χ **2.92 → 1.03**, but PSNR **14.09 → 13.33 dB**: the data are now explained to the noise floor and the field is not better — more optimization cannot supply the missing illumination. The report says so and recommends sources / receivers on the z sides (a surround array: `σ₆/σ₁ = 0.82`, *well-determined*) or a smoother / layered prior |
| `holography` (intensities at 3 distances, 400 steps) | gauge: the mean phase | `constant[phase]` invisibility 1.7·10⁻⁶ (a random direction of the same size changes the data 6 × 10⁵ times more) | `ZeroMean` head (anchored at the head's prior level 0); budget 400 → 800 | raw PSNR **6.77 → 23.30 dB** (the offset of −0.69 rad is gone); mean-subtracted PSNR 34.49 → 37.62 dB. The repair alone, at 400 steps: raw 6.77 → 21.0 dB |
| `poisson_source` (10 % observed, 450 steps) | budget + observation coverage | *converging*: χ = 5.4 / 5.6 / 3.7 / 2.5 / 2.0 / 0.91 at 50 … 1600 steps; acquisition *severely under-determined* (102 data for 1024 unknowns) | budget 450 → 1600 with Morozov stop; ℓ1 / TV kept (χ = 0.91 is inside the tolerance band) | PSNR **22.23 → 27.34 dB**, all **3 of 3** sources (the default finds 2) |

Tuning cost: EIT 3150 probe steps (23 s), poisson 3150 (5 s), holography 1550 (6 s), wave_fwi
3150 (170 s: 26 ms per step). At `level="quick"` (used by the fast tests): EIT with one doubling
less 16.51 → 21.46 dB; poisson 22.23 → 24.98 dB — its probes stop at 800 steps (still
converging), so the budget is doubled with the Morozov stop, which fires at χ = 1.00 after 1109
steps. With only 102 observations the ideal misfit is below 1 (`χ² ≈ 1 − d/m`); `standard`
replicates the probe that reached χ = 0.91 and gets 27.34 dB. On toy1d with a 100× too strong TV
weight, `thorough` (7465 probe steps, 8 s) takes the problem from 16.3 to 32.1 dB: the budget
probe reports *stalled* at χ = 14.7, Morozov fixes the weight, the budget is re-probed and the
held-out search confirms.

On the current smoke presets (EIT 24² with 12 patterns, poisson 30 % observed, zero-mean
holography; `nefi autotune <name> --smoke --level quick --compare`) the tuner still improves the
result rather than regressing: EIT 23.54 → 25.42 dB (600 → 800 steps), poisson
30.65 → 31.31 dB (600 → 800), holography 35.01 → 37.69 dB (400 → 800); the holography gauge is
reported as *handled by* the new `ZeroMean` head.

## Noise level

Every decision is measured against the noise standard deviation σ: the measurement's
`noise_std` when known, else `nefi.auto.estimate_noise` (Immerkær Laplacian, high-order
differences, operator hooks such as the k-space annulus). An estimated σ is stored in the returned
problem (so the solver can stop by the discrepancy principle) and widens the "at the noise floor"
band from χ ≤ 1.05 to χ ≤ 1.2.

## Gauges — `detect_gauges`, `repair_gauges`

**What is probed.** At a *generic* point near the current field (a smooth random perturbation of
10 % of each field's natural scale — a symmetric initialization such as φ ≡ 0 would make every sign
flip look invisible), the masked forward model is differentiated by central differences along

* the additive constant of every field;
* the global scale of every field: the homogeneity degree `p` is measured from `F(c·x)` for
  `c = 0.9, 1.1` (consistent within 5 %, direction residual < 2 %), then the *final stage's* data
  terms are evaluated at `pred` and `c·pred` — a scale gauge needs both a homogeneous operator and
  a scale-free fidelity (max / mean-normalized); `p ≈ 0` (an operator that normalizes internally)
  is a gauge regardless of the loss;
* the sign of every signed field (`F(−x)` against a random move of the same size);
* any `candidates={"name": direction}` you pass: a tensor, `{field: tensor}` (scalars broadcast —
  e.g. `{"a": 1.0, "b": -1.0}` for "a ↔ b exchange") or a callable.

**How it decides.** The data change along the direction divided by the *largest* data change
along random smooth / white directions of the same size (the "invisibility"): below `tol = 1e-3`
it is a gauge. Exact gauges measure ≈ 10⁻⁶ (float32 round-off); ordinary directions ≈ 0.1–10.

**What it changes.** A gauge already fixed by the problem is reported as *handled* (a `ZeroMean` /
`MassNormalized` head, an `EnergyScaleCorrection`, a `Conservation` loss, a regularizer that pins
it softly — ℓ1 or Tikhonov change under a constant shift, TV does not). Otherwise
`repair_gauges` returns a copy with

| gauge | fix |
|---|---|
| constant | mean anchor on the head: `nefi.fields.ZeroMean` when the anchor is 0, else `MeanAnchor(inner, value, region)` — anchored at the head's `init_value` (the prior level) or `mean_prior=`; `anchor="border"` anchors the mean of a 10 % frame (an object in a known uniform background) |
| scale, `p ≠ 0` | `EnergyScaleCorrection(field, homogeneity=p)` (NeTMY Eq. 30) for data with a positive total, `LeastSquaresScale` (`α^p = ⟨y, ŷ⟩/⟨ŷ, ŷ⟩`) for signed data |
| scale, `p = 0` | `MassNormalized` head when you pass `mass={field: total}`; unresolved otherwise |
| sign | `SignConvention(field, rule)` post-processing (a convention, not information) |
| custom | `DirectionPenalty` loss pinning the direction's coefficient at its initial value |

NeTMY's max-normalized fidelity is detected as a scale gauge of degree 1 *handled by*
`EnergyScaleCorrection` (`nv_relaxometry`); EIT is homogeneous of degree −1 (`F(cσ) = F(σ)/c`)
but its MSE fidelity pins the scale — not a gauge.

**Cost and limits.** A few dozen forward evaluations, no gradients (< 0.1 s on the smoke
problems). The probe is local; discrete gauges (φ + 2π), grid translations and gauges hidden
inside nuisance parameters are not probed; a scale gauge is only detected for exactly scale-free
data terms.

## Budget and learning rate — `probe_convergence`, `tune_budget`

1. **Doubling budgets** at the base learning rate: probes of 50, 100, 200 steps, each a complete
   schedule (every stage, cosine decay, annealing) — so `χ(T)` is what a budget of `T` delivers —
   and up to `extend` more doublings (3 at `standard`: 400, 800, 1600) while the misfit still
   falls above the floor, *or* sits at the floor while the reconstruction itself still changes
   by more than 5 % per doubling (`Δfield = ‖x_T − x_{T/2}‖ / ‖x_T − mean‖`, a data-free
   convergence check on the unknown). Measuring beats extrapolating: the misfit of an annealed
   neural field does not follow a power law (poisson: χ = 2.0 at 800 steps, 0.91 at 1600 — a
   power-law fit through 100 → 200 predicted 22 600 steps); and a misfit at the floor does not
   mean a converged field (EIT: χ 1.04 → 1.01 from 800 to 1600 steps, PSNR 21.0 → 23.1 dB).
2. **Classification.** *At the noise floor* if χ ≤ 1.05 (1.2 with an estimated σ); *converging*
   if the excess misfit `e = χ² − 1` fell by ≥ 15 % over the last doubling; else *stalled* (then
   more steps do not help: the learning rate, the regularization, the representation's
   capacity or the forward model must change — the report says which to try).
3. **Learning rate** — searched only when the base rate misses the floor: ×⅓, ×3 and the
   `nefi.auto.lr_range_test` pick are compared at the last probe budget and adopted when they
   lower the misfit by ≥ 5 %. A rate that wins a 50-step race usually plateaus higher on a long
   schedule (EIT: 3× the base rate stalls at χ = 1.16 where the base rate reaches 1.04), which is
   why the budget is measured first. A diverging base rate is replaced by the range-test pick.
4. **Recommendation.** At the floor: the first probed schedule that reached it and *settled*
   (χ ≤ 1, or Δfield ≤ 5 %) — else the last one — a validated recipe, with a discrepancy stop
   at its own misfit (`τ = min(1, χ_ref)`, so the stop cannot cut the validated schedule short).
   A default budget that already reaches the floor is **never shortened**: an earlier version
   recommended "the first budget at the floor" and, on presets that were already good, cut 600
   steps to 400 — the misfit was at the floor, the reconstruction was not (EIT 23.4 → 19.5 dB).
   Converging at the probe cap: twice the last probe (at least the base budget) with the last
   probe's *absolute* annealing length — the solver checks discrepancy and early stops only after
   annealing has finished, so a ramp stretched over a longer budget would overshoot the floor
   before the stop can fire (an early version that stretched the ramp stopped poisson at
   χ = 0.82 and lost 2.4–3.3 dB) — and `τ = 1`. Stalled: the base budget. `tune_budget` keeps the
   stage structure (resolutions, per-stage weights, freezing) and switches to cosine decay.

Cost: the sum of the probe budgets (≈ 2 × the recommended budget).

## Regularization — `tune_regularization`

Morozov's discrepancy principle chooses the *largest* regularization under which the data are
still explained to the noise level: the final misfit grows with the weight of a penalty, so the
weight with `RMSE(w) = τσ` is a root of a monotone function of `log w`. The tuner brackets it in
×10 steps (range ×10⁻⁴ … ×10⁴) and refines by regula falsi. Each trial is a short fit
(`budget_scale` × the curriculum); trials are **warm-started** from the fitted field of the
nearest previous trial (a single native stage, no re-annealing), so the optimization accumulates
along the search path and the last trials are close to converged. A *settle* phase first
continues at the current weight until the misfit stops moving, so that a still-converging short
fit is not mistaken for a weight effect. Several regularizers are scaled jointly (one factor,
their ratios kept; `mode="each"` tunes them in turn); stage overrides are scaled with the base
weights. Without σ it falls back to `GradNormBalancing` shares (each regularizer's weighted
gradient norm 10 % of the data term's).

*Tolerance band.* Weights are left alone while `RMSE/τσ ∈ [0.85, 1.1]`: σ and the effective number
of fitted degrees of freedom `d` are uncertain at that level — a good fit of `m` data leaves
`χ² ≈ 1 − d/m` (0.91 for 17 parameters and 102 data). Forcing χ = 1 on poisson_source's 102
observations would need a 300× stronger ℓ1 and lose 2.5 dB.

Measured: toy1d (64 points, 900 steps) from a 100× too strong TV weight (0.1, 16.3 dB) back to
1.0·10⁻³ — exactly the PSNR-best weight of a sweep (34.6 dB) — in 4 trials; from 10⁻⁶ it stays
(χ = 0.85, inside the band; 33.9 dB). poisson_source with a 10⁴× too strong ℓ1 (10, 16.1 dB):
lowered to 10⁻³ (the misfit stays budget-limited above τσ, so the weight is lowered to where it
stops raising the misfit), 22.2 dB.

Limits: with a prior that does not match the unknown (TV on spikes) Morozov over-smooths (toy1d
"mixed" scene at 3000 steps: 28.3 dB unregularized vs 25.5 dB at the discrepancy weight) — the
held-out search is the better judge there. At short budgets the misfit is budget-limited and the
tuner can only lower weights.

## Acquisition — `acquisition_report`

Linearized at the current field, measured in data units against σ:

| criterion | moderate below | severe below |
|---|---|---|
| observed data / unknowns | 1 | 0.25 |
| blocks (8 per axis) whose perturbation of amplitude `a` exceeds 1 σ | 85 % | 50 % |
| pixels with sensitivity ≥ 10 % of the 95th percentile | 80 % | 50 % |
| `σ_k/σ_1` of `dF/dx`, top k = 8 (Lanczos) | 0.5 | 0.05 |
| top-k modes above the noise for amplitude `a` | k | k/2 |

Per-pixel and per-block sensitivities come from the same Rademacher vector-Jacobian probes (the
`sensitivity_map` estimator; the gradient is summed over a block before squaring, giving
`‖J 1_B‖`). The amplitude `a` is data-implied: the block amplitude that would explain the residual
at the current field. A flat spectral head means at least k combinations of the unknown are
constrained within a factor 2 of the best one; a steep head means few — sparse acquisition or
strong smoothing. The report also lists the boundary bands with the lowest mean sensitivity (the
sides without sensors), contiguous regions below the noise floor, and — when available —
`nefi.fields.adaptive.match_report` (does the representation pass what the data resolve?).

| geometry | verdict | evidence |
|---|---|---|
| wave_fwi 2 sources × 8 receivers, cross-well | moderately under-determined | `σ₈/σ₁ = 0.35`; low-z / high-z sides 51 % below the best side → "add sources / receivers near the low-z and high-z sides — e.g. a surround geometry" |
| wave_fwi 8 × 32, surround | well-determined | `σ₈/σ₁ = 0.81`, 12 200 observed data |
| wave_fwi reflection (surface array) | moderately under-determined | "the high-z 29 % of the domain (z > 0.708) is below the noise floor — prefer stronger smoothness or a layered representation there" |
| poisson_source 10 % observed | severely under-determined | 102 data for 1024 unknowns → "rely on a sparsity prior … strength by the discrepancy principle" |
| eit 4 patterns | moderately under-determined | 240 data for 256 unknowns, `σ₈/σ₁ = 0.35` |
| sparse_view_ct | moderately under-determined | 512 data for 1024 unknowns |
| toy1d, deconvolution, holography | well-determined | flat heads (`σ₈/σ₁ ≥ 0.7`), full coverage |

The report changes nothing. It states that the acquisition cannot be tuned and recommends where
more measurements would help and which prior / representation fits what the data support.
Limits: only the leading singular values are computed (k Lanczos values — a problem with a flat head
and a catastrophic tail is not flagged by this criterion); the linearization is at the initial
field; radial / side averages hide anisotropic null spaces; thresholds are heuristics with
documented values (`nefi.autotune.THRESHOLDS`).

## Held-out search — `autotune`

```python
report = nefi.autotune.autotune(lambda **hp: Toy1D(n=64, **hp).build_problem(meas), trials=8)
problem, curriculum = report.build(lambda **hp: Toy1D(n=64, **hp).build_problem(meas))
print(report.table())
```

A scrambled Sobol sequence (`torch.quasirandom`, no extra dependency) samples the default space —
`lr` (×⅓…×3), `weight.<term>` for every active regularizer (×0.1…×10), `anneal_fraction`
(0.25…1) and `n_octaves` (±2, when the factory accepts it) — or your `space={name: (kind, lo,
hi) | [choices] | Dimension}`. Trial 0 is always the unmodified default. Each trial fits the
problem on 90 % of the observed entries (`nefi.fields.adaptive.holdout_masks`, leakage-free
coarse stages) with `budget_scale` × its curriculum and is scored by the held-out MSE — the
prediction risk, which penalizes both noise fitting and over-smoothing; `objective="discrepancy"`
scores `|log(RMSE/τσ)|` instead. Curriculum keys (`lr`, `anneal_fraction`, `steps`) and
`weight.<term>` are applied by the search itself; every other key is passed to
`problem_factory(**params)`.

Measured on toy1d from a wrong TV weight (0.1): held-out MSE 0.25 → 0.032 in 8 trials (1440
steps), PSNR of the full solve 16.3 → 25.7 dB (the ×10 range cannot reach 10⁻³ in one round;
Morozov does).

## Benchmarking a tuned configuration

```python
from nefi.bench import run_benchmark
from nefi.bench.protocol import default_method

res = run_benchmark(inst, [default_method(), nefi.autotune.autotuned_method("quick")],
                    n_samples=4, seeds=(0, 1, 2))
```

`autotuned_method` tunes every measurement inside the method's runner, so the benchmark's
`time_s` column includes the tuning (`result.extra["autotune_probe_steps"]` holds its cost); the
benchmark's `budget_scale` only scales the base curriculum the tuner starts from.

## What auto-tuning cannot do

* **Add information.** An under-determined acquisition stays under-determined: fitting its data
  better does not make the field better (wave_fwi 2 × 8 above). The report says so and points to
  the physics (more / better placed sensors) and to the prior.
* **Validate the field.** Every criterion lives in data space (misfit, held-out error,
  sensitivities). The data-fit paradox (NeFTY §5.2) applies to the tuner too: null-space errors
  are invisible to all of them.
* **Replace a mismatched prior.** Morozov sets the *strength* of the prior you chose; a wrong
  *kind* of prior (TV on spikes) is only caught by the held-out search, and only among the
  configurations it tries.
* **Be free.** `standard` costs about two tuned solves; wave_fwi-sized operators make that
  minutes on a CPU. Use `level="quick"` or tune once on a representative measurement and reuse
  the curriculum (`tuned_config.yaml`) for a campaign.

## API

```python
detect_gauges(problem, *, tol=1e-3, candidates=None, fields=None, kinds=("constant", "scale", "sign"),
              amplitude=0.1, step=0.02, curriculum=None, shape=None, progress=1.0, seed=0) -> GaugeReport
repair_gauges(problem, report, *, fixes=None, mean_prior=None, anchor="mean", mass=None,
              sign_rule="max_abs_positive", penalty_weight=None) -> InverseProblem
probe_convergence(problem, *, steps=(50, 100, 200), extend=3, max_probe_steps=None, sigma=None,
                  curriculum=None, lr=None, lr_test=True, lr_factors=(1/3, 3), max_steps=None,
                  seed=0, device="cpu") -> ConvergenceReport
tune_budget(problem, report=None, *, curriculum=None, tau=None, anneal_fraction=None, **probe_kw) -> Curriculum
tune_regularization(problem, names=None, *, sigma=None, budget_scale=0.2, tau=1.0, mode="joint",
                    curriculum=None, bracket=(1e-4, 1e4), max_trials=10, rtol=0.05, band=(0.85, 1.1),
                    warm_start="nearest", warm_lr=0.5, init=None, init_chi=None, settle=2,
                    shares=None, seed=0, device="cpu") -> RegularizationResult   # a dict subclass
acquisition_report(problem, k=8, *, field=None, n_probes=16, n_iter=None, blocks=8, rel_tol=0.1,
                   snr=1.0, amplitude=None, sigma=None, match=True, jvp_mode=None, seed=0) -> AcquisitionReport
autotune(problem_factory, *, space=None, holdout=0.1, budget_scale=0.2, trials=8, seed=0,
         objective="heldout", curriculum=None, group_axes=None, tau=1.0, device="cpu") -> TuneReport
autotune_problem(problem, *, level="standard", sigma=None, curriculum=None, problem_factory=None,
                 seed=0, device="cpu", gauges=True, acquisition=True, candidates=None,
                 mean_prior=None, anchor="mean", mass=None, max_steps=None, options=None)
    -> tuple[InverseProblem, Curriculum, AutotuneReport]
autotune_instance(instance, seed=0, level="standard", *, scene_class=None, curriculum=None,
                  device="cpu", compare=False, **kw) -> tuple[InverseProblem, Curriculum, AutotuneReport]
autotuned_method(level="quick", name=None, **autotune_kw) -> nefi.bench.protocol.Method
```
