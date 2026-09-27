# 8 · Robustness and adaptivity

Per-measurement optimization runs unattended on hundreds of measurements (NeTMY: 512 samples ×
3 seeds × several methods), on real data with calibration errors and outliers, and on stiff
physics. The solver therefore carries a set of safety and adaptivity features. All are optional,
configured through `OptimConfig`, `Curriculum`, `Stage`, `Solver` arguments, the operator and the
losses — and all are available from config files and `--set` on the CLI.

## Safety features of the solver

| feature | how | default |
|---|---|---|
| gradient clipping | `OptimConfig(grad_clip=1.0)` (global norm) | 1.0 (NeTMY Tab. 5) |
| NaN guard | `Solver(nan_guard=True, max_bad_steps=10, checkpoint_every=25)`: on a non-finite loss, roll back to the last good parameters, clear the optimizer state and halve the LR; give up with an actionable `SolverError` after `max_bad_steps` consecutive failures | on |
| EMA of parameters | `OptimConfig(ema=0.999)`: evaluate the exponential moving average at each stage end | off |
| early stopping | `Curriculum(early_stop_patience=200, early_stop_min_delta=1e-6)` on the data loss, per stage | off |
| discrepancy principle | `Curriculum(discrepancy_tau=1.0)` + `Measurement(noise_std=σ)`: end a stage once RMSE ≤ τσ (Morozov) | off |
| restarts | `Curriculum(restarts=3)`: re-initialize and re-run, keep the best final data loss | 1 |
| time budget | `Curriculum(time_budget_s=600)`: stop everything after a wall-clock budget | off |
| optimizer | `OptimConfig(optimizer="adamw" \| "adam" \| "sgd" \| "lbfgs", weight_decay, betas)` | AdamW |
| checkpoints | `CheckpointCallback(path, every=500)` ([tutorial 7](07_running_on_gpu_servers.md)) | off |

The stop reason of every stage is recorded (`result.stage_results[i]["stop"]`: `completed`,
`early_stop`, `discrepancy`, `time_budget`) and appears in benchmark rows.

```python
cur = inst.default_curriculum()
cur.discrepancy_tau = 1.0          # needs meas.noise_std (DataGenerator sets it)
cur.restarts = 2
result = nefi.invert(problem, cur, device="cpu", seed=0)
```

On toy1d, two restarts selected a run at 33.7 dB instead of the single run's 28.2 dB — restarts
are the cheapest remedy for initialization-dependent basins (NeTMY App. E.10 reports ≈ 5 % of
samples trapped in the centered-collapse basin at stage 1).

## Knowing the noise level

The discrepancy principle and noise-aware weighting need `σ`. Synthetic data carry it
(`Measurement.noise_std`); for real data estimate it robustly:

```python
sigma = nefi.auto.estimate_noise(meas)            # MAD-based; toy1d: 0.0196 (true 0.0223)
meas = nefi.Measurement(meas.data, noise_std=sigma)
```

`nefi.diagnostics.data_fit_paradox(result, problem)` reports `RMSE / σ` after a run: ≈ 1 is the
target, ≪ 1 means the solver fitted noise, ≫ 1 means an under-fit (or model mismatch).

## Robust and count-based fidelities

| data | loss |
|---|---|
| Gaussian noise | `MSE()` (`noise_aware=True` divides by σ²) |
| outliers, dead pixels, model mismatch | `Huber(delta=…)` |
| photon / event counts | `PoissonNLL()` (prediction = expected counts > 0) |
| unknown global scale (normalized maps) | `LogMSE(normalize="max")` (NeTMY D), `NormalizedMSE("mean")` (R_nm), `RelativeMSE()` |
| missing entries | `Measurement(mask=…)`: every data loss averages over observed entries |

With scale-free fidelities, recover the absolute scale afterwards with
`EnergyScaleCorrection()` (NeTMY Eq. 30, exact for homogeneous operators in the noiseless limit).

## Nuisance parameters

```python
problem.operator = nefi.Nuisance(problem.operator, gain=True, offset=True)
result = nefi.invert(problem)
problem.operator.gain, float(problem.operator.offset)
```

`Nuisance(inner, gain, offset)` wraps any operator as `exp(log_gain)·F(x) + offset` with learnable
parameters shared across curriculum resolutions — a cheap way to absorb an unknown detector gain or
background level on real data. Identifiability matters: for a **homogeneous** operator a global
gain is exactly degenerate with the field's amplitude (`F(c·x) = c F(x)`); on toy1d with data
scaled by 1.7 the solver settles at gain 1.24 and puts the rest into the field. Use a gain only
when the field's scale is pinned — by a bounded head with a physical range (NeFTY's diffusivity),
a known total mass (`nefi.priors` `Conserved`), or a calibration — and prefer an offset term for
background levels.

## Balancing loss weights

```python
ctx = problem.context()                     # loss context at the current parameters
problem.losses.auto_balance(ctx, target=1.0)  # every active term contributes 1.0 at this iterate
```

A pragmatic start when porting to a new problem (`nefi.from_forward` balances automatically:
data = 1, each prior at its relative strength). Report paper results with the papers' fixed weights
(NeTMY Tab. 7, NeFTY Tab. 5), which are config fields of the instances. Per-stage re-weighting
(NeTMY App. D.4 swaps the log-MSE for R_nm in stage 2) is `Stage(loss_weights={...})`.

## Uncertainty: seed ensembles and sensitivity

```python
ens = nefi.ensemble(lambda: inst.build_problem(meas), curriculum, seeds=(0, 1, 2), device="cpu")
ens.mean["x"], ens.std["x"], ens.coefficient_of_variation()
```

Different initializations of the same per-measurement problem disagree where the data do not
constrain the field; the pixelwise standard deviation is a cheap uncertainty map. Combine it with
the forward sensitivity (`nefi.diagnostics.sensitivity_map`, [tutorial 5](05_diagnostics.md)) as a
trust mask — NeFTY recommends reporting uncertainty in low-sensitivity regions.

## Adaptivity: curriculum and annealing

* **Multiscale stages** — `Curriculum.multiscale(shape, n_stages, steps, lr, lr_decay)`; each stage
  re-anneals the Fourier bands (`Stage(anneal=True, anneal_fraction=...)`), fitting low
  frequencies first (NeTMY (P1), NeFTY Cor. 1).
* **Freezing** — `Stage(freeze=("operator.",))` optimizes only the field in a stage (e.g. estimate
  a nuisance gain first, then freeze it).
* **Adaptive annealing** — callbacks may set `solver.progress_override` to drive β from the data
  instead of the step counter (residual- or operator-aware schedules in
  `nefi.fields.adaptive.annealing`), and `solver.stop_stage` to end a stage early.
* **Budgets from the problem** — `nefi.auto.auto_curriculum` and `budget_steps` size a schedule
  from the grid size; `nefi.auto.lr_range_test` probes a learning rate.

## Letting nefi tune itself

Most of the knobs above can be set from the measurement itself. `nefi.autotune` probes the
problem and fixes what the data can decide: it finds gauge freedoms the data cannot see (a
global phase from intensities, a global scale under a max-normalized fidelity) and pins them
(`ZeroMean` / `MeanAnchor` heads, `EnergyScaleCorrection` with the measured homogeneity
degree); it runs short curriculum-shaped probes with doubling budgets to decide whether the fit
is at the noise floor, still converging or stalled, and sets the budget, learning rate,
annealing length and discrepancy stop accordingly; it sets regularization weights by the
discrepancy principle; and, at `level="thorough"`, it runs a held-out hyper-parameter search.
An acquisition report states what the measurement cannot determine (too few data, a steep
singular-value head, sides without sensors) — auto-tuning cannot change the physics, and the
report says so:

```python
problem, curriculum, report = nefi.autotune.autotune_problem(problem, level="standard")
print(report.to_markdown())
result = nefi.invert(problem, curriculum)
```

or `nefi autotune eit --smoke --compare`. On the demo gallery's failures this moves EIT from
16.5 to 23.1 dB and poisson_source from 22.2 to 27.3 dB; see [Auto-tuning](../autotune.md)
for what each tuner detects, how it decides, its cost and its limits.

## Sharpening edges after the solve

Reconstructions of piecewise-constant unknowns through smoothing physics come out soft: the data
stop constraining edges first, the MLP prior filters them, and a finite budget stops before the
fine bands converge. `nefi.solve.refine.refine_edges` runs a short second stage that sharpens the
interfaces under the **same** operator and data — a multi-phase level set initialized from the
smooth field (default), a double-well phase field, or stronger TV with a binarizing head swap —
and reports the data fit before and after:

```python
refined, report = inst.refine(result, measurement, gt=gt)      # phase values from the config
print(report.summary())    # refine[levelset] dot3d: accepted — χ 1.15 → 1.1, iou 0.5 → 0.723, …
```

It never replaces the smooth result silently (a separate `Result` with
`extra["refined_from"]`), refuses refinements that raise χ above 1.1 × max(χ_ref, 1) — the
reference being the smooth result or, for an early-stopped run, the `mode="continue"` control
with the same budget — or move interfaces out of the smooth solution's transition band. Run the
control yourself too: an early-stopped smooth result improves under *any* continuation, so compare
against it before crediting the prior. Numbers, equations and limits:
[Edge refinement](../refinement.md).

## Failure modes and what to try

Symptoms, diagnostics and remedies for centered collapse, cross artifacts, high-frequency leakage,
boundary artifacts and stalled solvers are collected in the [FAQ](../faq.md).

## Adaptive loss balancing (GradNorm-style)

When porting the recipe to a new problem the relative gradient magnitudes of the data term and the
regularizers are unknown. `nefi.solve.GradNormBalancing` is a callback that, every few steps,
measures the gradient norm each loss term induces on the field parameters and rescales the
weights of the active stage so that the *weighted* gradient norms follow prescribed shares
(the first data term is the anchor and never changes):

```python
from nefi.solve import GradNormBalancing
cb = GradNormBalancing(shares={"data": 1.0, "tv": 0.1, "l1": 0.05}, every=50, alpha=0.5)
result = nefi.invert(problem, callbacks=[cb])
cb.log[-1]          # the weights and gradient norms after the last rebalance
```

Use it to *find* weights on a new problem, then freeze them (`LossSet(weights=...)`) for
reported results — the papers use fixed weights. It is a no-op under `Solver(compile="step")`
(weights are compiled into the graph) and costs one backward pass per term at every rebalance.
`nefi.from_forward(weights="auto")` is the one-shot alternative (balanced at a probe iterate).
