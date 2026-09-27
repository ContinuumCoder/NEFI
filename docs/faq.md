# FAQ — failure modes and what to try

Each entry: the symptom, how to confirm it with nefi's diagnostics, why it happens (with the paper
reference), and the knobs to turn. Most checks start with

```bash
nefi diagnose <instance|config> [--solve] --plot
```

## Centered collapse

**Symptom.** The reconstruction concentrates mass in the middle of the window, away from the true
sources; free-pixel baselines (grid / Tikhonov / ADMM) are hit hardest.

**Confirm.** `iter0_gradient(problem)`: a center/outer ratio ≫ 1 with the peak near the center
(NeTMY §5.3: 18.29× under F2; nefi measures 14.3× on the 64² NV problem vs 0.24× under F1).
`center_mass_ratio(result.fields[...])` well above the uniform baseline
(`uniform_center_mass(shape)` = 0.071 in 2-D; NeTMY: Tikhonov 0.223, ADMM 0.78 under F2).
`energy_barrier(problem, collapsed, gt)`: a positive height means descent cannot leave the basin.

**Why.** The finite window makes edge pixels less visible (P2, `sensitivity_map`), and
max-normalized fidelities couple every pixel to the current peak (P3, NeTMY Eq. 24); a free-pixel
solver executes that centered gradient verbatim (`G_θ = I`).

**Try.** A neural field (its filter kernel `G_θ` redistributes the centered gradient —
`realized_update`); frequency annealing and a coarse first stage; `Curriculum(restarts=3)` (≈ 5 %
of NeTMY samples start in the basin, App. E.10); a mean-normalized companion fidelity
(`NormalizedMSE("mean")`, NeTMY R_nm); pad the domain so sources are not near the window edge; a
gated head (`GatedSoftplus`) so the background can switch off.

## Cross-shaped artifacts

**Symptom.** Axis-aligned streaks or crosses through bright sources, typically in dense scenes with
sources closer than the point-spread width (NeTMY App. E.10, Fig. 8).

**Confirm.** The artifact follows the grid axes, not the physics; `singular_values(...,
return_vectors=True)` shows near-degenerate modes; the effect grows with the anisotropic TV
weight.

**Why.** Merging below the resolution limit (P4): a family of densities explains the data equally
well, and anisotropic TV (NeTMY Eq. 3) prefers axis-aligned explanations.

**Try.** Isotropic TV (`TV(isotropic=True)`, NeFTY Eq. 22) or a `Laplacian` term (NeTMY's
anti-artifact variant: fewer crosses, slightly worse MSE); fewer Fourier octaves or a slower
annealing (`Stage(anneal_fraction=...)`); report localization metrics (Hungarian F1 with a
matching radius ≈ the PSF width) rather than MSE.

## High-frequency leakage

**Symptom.** Small clusters bleed into neighbouring pixels; speckle around compact sources (NeTMY
App. E.10, Fig. 9).

**Confirm.** `filter_kernel_row(problem, progress=1.0)` has a narrow main lobe and far side-lobes;
`effective_bandwidth(field, 1.0)` exceeds what `singular_values` says the data support.

**Why.** At full annealing (β = K) the network can fit pixels the data do not constrain; only the
sparsity prior and the gate stop it.

**Try.** Fewer octaves (`n_octaves`, NeTMY sweep: K = 12 is the smallest that resolves detail
without divergent peaks), `anneal_fraction < 1` with a longer tail at full bandwidth only if needed,
ℓ1 + gate, early stopping at the noise level (`discrepancy_tau=1.0`), or EMA of parameters.

## Back-face artifact

**Symptom.** In thermal tomography a spurious low-diffusivity sheet appears along the back face or a
lateral boundary, typically with shallow defects close to the heat source (NeFTY Fig. 14).

**Confirm.** `sensitivity_map` on the thermal problem: deep voxels are orders of magnitude less
visible than surface voxels (max/min ≈ 280 on the smoke problem); the artifact sits in the
low-sensitivity region and varies across seeds (`nefi.ensemble` standard deviation).

**Why.** Surface data barely constrain the far side of the slab (NeFTY Prop. 2 / App. B.4), so the
prior and the boundary condition decide what happens there; an adiabatic back face lets the
optimizer trade interior contrast against boundary heat accumulation.

**Try.** Check the boundary condition matches the experiment (adiabatic vs Robin back face,
`bc_back`, `robin_h`); stronger isotropic TV; a bounded head with a realistic `alpha_min`; report
the sensitivity map or ensemble spread as a trust mask; exclude the last layers from depth
metrics.

## Stalled solver at high contrast

**Symptom.** With extreme property contrast (e.g. air voids, 1:1000) the forward solve or its
gradient stops making progress; losses plateau, NaN guards fire, or the implicit solver needs many
iterations.

**Confirm.** The field hits its bounds (`Bounded` saturates, zero gradients); the heat operator's
Jacobi residual does not reach the noise floor; `singular_values` spans many orders of magnitude.

**Why.** The implicit-Euler system `A(α) = I − Δt L(α)` becomes severely ill-conditioned; NeFTY
(App. E.1) therefore uses a 1:20 defect contrast, which already saturates the surface signature.

**Try.** Keep the contrast physical but moderate via `Bounded(alpha_min, alpha_max)`; more inner
iterations or conjugate gradients (`solver="cg"` in the thermal config) instead of a fixed Jacobi
count; float64 for the forward model; smaller learning rates with `grad_clip`; the NaN guard's LR
back-off is automatic.

## Good data fit, wrong field (data-fit paradox)

**Symptom.** The measurement is reproduced almost perfectly but the reconstruction is wrong or
nearly constant (NeFTY §5.2: PINNs fit surfaces at ≈ 63 dB with a volumetric IoU of 0.01).

**Confirm.** `data_fit_paradox(result, problem, gt)`: high `data_psnr`, low field metrics;
`discrepancy_ratio` ≪ 1 means the noise itself was fitted.

**Try.** Keep the physics a hard constraint (an `Operator`, never a soft residual); stop at the
noise level (`discrepancy_tau`); stronger priors in the null space the data cannot see
(`singular_values(..., return_vectors=True)` shows it).

## My benchmark refuses to run: "inverse crime"

The data generator's `fidelity_tag` equals the inversion operator's tag — the data were simulated
with the inversion model. Simulate with an independent model (finer grid in float64, different
integrator, higher-fidelity physics; [tutorial 3](tutorials/03_new_forward_operator.md)). If you
deliberately want the matched-operator regime (NeTMY Tab. 2), pass `allow_inverse_crime=True`
(`--allow-inverse-crime`); the results are labelled "matched" everywhere.

## Results change between runs

Seeds control data (`data_seed`) and initialization (`seed`); the CLI and the benchmark seed
before building the problem. Remaining variation comes from non-deterministic GPU kernels
(`seed_everything(seed, deterministic=True)`) or from genuinely seed-dependent basins — measure
it with several seeds and report the confidence interval ([tutorial 6](tutorials/06_benchmarking.md)).

## The run is too slow or runs out of memory

Use `--smoke` (instance smoke configuration + a 40-step cap), `--budget-scale 0.1`, or smaller
grids (`--set n=32`). Paper-scale runs belong on a CUDA server
([tutorial 7](tutorials/07_running_on_gpu_servers.md)). For time-stepping physics make sure the
operator uses an adjoint or checkpointing, otherwise memory grows with the number of steps.
