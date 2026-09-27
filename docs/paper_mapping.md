# Paper → code mapping

Every equation, proposition, table and key figure of the two papers, mapped to the code that
implements, measures or reproduces it. Paths are relative to the repository root; `module.py::name`
points at a class or function. The papers: NeTMY, [arXiv 2605.13988](https://arxiv.org/abs/2605.13988),
and NeFTY, [arXiv 2603.11045](https://arxiv.org/abs/2603.11045).

Legend: **(planned)** = not implemented yet (the path is where it is expected to land);
**(n/a)** = intentionally out of scope for a label-free, dataset-free library (e.g. supervised
baselines, proprietary real data). Tests that pin a statement are listed after "test:".

- [NeTMY — Neural Fields for NV-Center Inverse Sensing (arXiv 2605.13988)](#netmy)
- [NeFTY — Neural Field Thermal Tomography (arXiv 2603.11045)](#nefty)
- [Shared protocol elements](#shared-protocol-elements)

---

## NeTMY

Instance: `nefi/instances/nv_relaxometry/` (registered as `nv_relaxometry`; docs:
[instances/nv_relaxometry.md](instances/nv_relaxometry.md); configs:
`configs/nv_relaxometry_paper.yaml`, `configs/nv_relaxometry_smoke.yaml`).

### Measurement model and forward operators (§3.1, App. A)

| Paper | Content | Code |
|---|---|---|
| Eq. (1), Eq. (8) | dipolar Green tensor `G_ia(R)` | `nefi/instances/nv_relaxometry/physics.py::dipolar_green_tensor`, `DipolarKernels` |
| Eq. (2) — F1 | scalar/coherent operator `(G_nv ∗ ρ)² L(ω; ω_L)` | `nefi/instances/nv_relaxometry/operator.py::NVOperator(mode="F1")`; kernel `physics.py::gzz_kernel_fn` |
| Eq. (2) — F2 | tensor/incoherent operator `Σ_a (|G_az|² ∗ ρ) L(ω; ω_L)` | `operator.py::NVOperator(mode="F2")`; kernel `physics.py::power_kernel_fn`; FFT convolution `nefi/operators/conv.py::fft_convolve` |
| Eq. (9) | pointwise derivation of F2 (source-side line = F3, factorized line = F2) | `NVOperator(mode="F2")`, `NVDirectSimulator`; test: `tests/test_nv_relaxometry.py::test_f2_matches_f3_for_constant_larmor` |
| Eq. (10) | F1 with diagonal `ρ²` and cross term `C(r)`; `F1(cρ) = c² F1(ρ)` | `NVOperator(mode="F1")` (`homogeneity = 2`); test: `test_operator_homogeneity` |
| Eq. (11) | direct simulator F3 (float64, Lorentzian at the source pixel) | `operator.py::NVDirectSimulator`, used only by `NVRelaxometry.data_generator` (`fidelity_tag` ≠ inversion) |
| Eq. (12) | operator difference F1 − F2 | evaluate both `NVOperator` modes on the same fields; test: `test_f2_departs_from_f3_when_larmor_varies_within_kernel` (F2 vs F3) |
| Eq. (13) | Lorentzian `L(ω; ω_L)` | `physics.py::lorentzian`, frequency grid `physics.py::frequency_grid` |
| App. A.8 | F2 robustness to in-kernel Larmor variation `χ` | test: `test_f2_departs_from_f3_when_larmor_varies_within_kernel` |
| Eq. (14)–(18), App. A.9 | `1/T1 ∝ σ_b/D⁴` consistency check of the forward model | (n/a) verification figure, not reproduced |
| App. D.3 | FFT kernel cache per stage resolution | `nefi/operators/conv.py::FFTConvolution` (per-shape cache), `NVOperator.at_resolution` |

### Inverse problem, data fidelity and scale (§3.2, App. B, App. D.4–D.5)

| Paper | Content | Code |
|---|---|---|
| §3.2, Eq. (19) | noise map `N(r;S) = Σ_ω S`, max-normalized log-MSE `D` | `nefi/losses/data.py::LogMSE(normalize="max", reduce_axes=(0,))`; `nefi/instances/nv_relaxometry/losses.py::noise_map`, `normalized_noise_map`; loss key `log_mse` |
| Eq. (3) | canonical objective `D + λ1‖ρ‖1 + λTV TV(ρ)` (anisotropic TV) | `nefi/losses/base.py::LossSet` of `LogMSE`, `nefi/losses/reg.py::L1`, `TV(isotropic=False)`; `NVRelaxometry.losses` |
| Eq. (6) | NeTMY objective `+ λN R_nm + λρ R_ds` | `NVRelaxometry.losses` (keys `log_mse`, `noise_map`, `direct_density`, `sparsity`, `l1`, `tv`) |
| App. D.4 — R_nm | mean-normalized noise-map MSE | `nefi/losses/data.py::NormalizedMSE(normalize="mean")` (key `noise_map`) |
| App. D.4 — R_ds | direct-density proxy `MSE(ρ²/mean, N_obs/mean)` | `nefi/instances/nv_relaxometry/losses.py::DirectDensityLoss` (key `direct_density`) |
| App. D.4 | stage-2 rebalancing (R_nm replaces D) | `nefi/solve/curriculum.py::Stage.loss_weights`; `NVRelaxometryConfig.stage_loss_weights` |
| Eq. (20), App. B.2 | scale-flat ray of max-normalization | test: `tests/test_core_operators_losses.py::test_log_mse_and_normalized_mse_invariances` |
| Prop. 1, App. B.3 | energy-ratio correction exact under linear F2 | `nefi/solve/postprocess.py::EnergyScaleCorrection` (`p = operator.homogeneity`); test: `test_energy_scale_correction_linear_and_quadratic`, `tests/test_nv_relaxometry.py::test_energy_scale_correction_is_exact` |
| Eq. (21), App. B.4 | square-root correction under quadratic F1 | `EnergyScaleCorrection` with `homogeneity = 2` |
| Eq. (28)–(30), App. D.5 | one-shot post-optimization scale correction | `EnergyScaleCorrection` in `InverseProblem.postprocess` (`NVRelaxometry.postprocess`) |

### Ill-posedness (§3.3, App. C)

| Paper | Content | Code |
|---|---|---|
| (P1), Lemma 1, Eq. (22) | exponential frequency suppression `e^(-k z0)` | measured by `nefi/diagnostics/spectrum.py::singular_values`; test: `tests/test_nv_relaxometry.py::test_power_kernel_fourier_decay_lemma1`; mitigated by `nefi/fields/encoding.py::FourierFeatures` annealing + multiscale |
| (P2), Eq. (23), App. C.2 | finite-window center bias; Jacobian column norm `‖∂F/∂ρ(r0)‖²` | `nefi/diagnostics/sensitivity.py::sensitivity_map`, `center_to_outer_ratio`; `nefi/diagnostics/landscape.py::iter0_gradient`; test: `tests/test_diagnostics.py::test_iter0_gradient_center_bias_signature`, `tests/test_nv_relaxometry.py::test_iter0_center_bias_signature` |
| (P3), Eq. (24), App. C.3 | max-normalization peak coupling | automatic through `LogMSE(normalize="max")`; diagnosed by `iter0_gradient` + `energy_barrier`; `NormalizedMSE(normalize="mean")` avoids it |
| (P4), Eq. (25), App. C.4 | resolution-limited merging, `w_psf`; ω_L identifiable only on supp ρ | `NVRelaxometryConfig.match_radius` (Hungarian F1 radius); `nefi/fields/heads.py::SupportMasked` |
| Tab. 4, App. C.5 | pathology → signature → experiment → mitigation | [tutorials/05_diagnostics.md](tutorials/05_diagnostics.md), [faq.md](faq.md) |

### Method (§4, App. D)

| Paper | Content | Code |
|---|---|---|
| Eq. (4), App. D.1, Tab. 5 | coordinate MLP `f_θ` (6 layers × 320, tanh, skip at 3) | `nefi/fields/neural.py::NeuralField`; `NVRelaxometry.field` (`hidden`, `depth`, `skip_at`, `activation`) |
| Eq. (5) | gated softplus density head | `nefi/fields/heads.py::GatedSoftplus` |
| Eq. (26) | support-masked Larmor head (τ = 0.3, stop-gradient mask) | `nefi/fields/heads.py::SupportMasked(Bounded(ω_min, ω_max), depends_on="rho", tau=0.3)` |
| Eq. (27), App. D.2 | annealed Fourier features (K = 12, cosine gate, β reset per stage) | `nefi/fields/encoding.py::FourierFeatures`, `nefi/solve/curriculum.py::Stage.progress_at` |
| §4.2, Tab. 6, App. D.3 | two-stage multiscale 32² → 64², cosine LR to 0.01 η, lr × 0.5 | `nefi/solve/curriculum.py::Curriculum.multiscale`; `NVRelaxometry.default_curriculum` |
| Tab. 5 (optimizer) | AdamW, wd 1e-4, grad clip 1 | `nefi/solve/curriculum.py::OptimConfig`, `nefi/solve/solver.py::Solver` |
| Tab. 7 | loss weights (2.0, 0.5, 0.1, 1e-2 + 1e-3, 1e-3) | `NVRelaxometryConfig.w_log_mse`, `w_noise_map`, `w_direct_density`, `w_sparsity`, `w_l1`, `w_tv` |
| Algorithm 1 | per-measurement optimization + scale correction | `nefi.invert` / `nefi/solve/solver.py::Solver.run` + `EnergyScaleCorrection` |
| Eq. (7), §4.4 | `Δρ ≈ −η J_θ J_θᵀ ∇ρL = −η G_θ ∇ρL` | `nefi/diagnostics/filtering.py::filter_kernel_row`, `realized_update` |
| Lemma 2, Eq. (31)–(32) | first-order filtering of the update; `G_θ ⪰ 0` | `filter_kernel_row` (vjp then jvp through the field); test: `tests/test_diagnostics.py::test_filter_kernel_row_grid_delta_and_neural_bump`, `test_realized_update_grid_verbatim_vs_neural` |
| Eq. (33)–(34) | Jacobian column functions `ψ_p`; `G_θ g ∈ span{ψ_p}` | rows of `G_θ` via `filter_kernel_row` |
| Eq. (35) | `rank(G_θ) ≤ min(|Ω|, P)` | `nefi/fields/neural.py::mlp_jacobian_rank_hint` |
| Eq. (36)–(37) | effective bandwidth `B_β ~ 2^(k_β) π` | `FourierFeatures.effective_bandwidth`, `nefi/diagnostics/filtering.py::effective_bandwidth` |
| Eq. (38)–(39) | `G_θ` is low-pass for small β | `filter_kernel_row(progress=β/K)` + `nefi/diagnostics/filtering.py::kernel_spread` ([tutorial 05](tutorials/05_diagnostics.md)) |
| App. D.6 | comparison with free-density / quasi-Newton / splat solvers | `nefi/fields/grid.py::GridField` (`G_θ = I`), `nefi/baselines/lbfgs.py::lbfgs_curriculum`, `nefi/baselines/gaussian_splat.py::GaussianSplatField` |

### Experiments (§5, App. E)

| Paper | Content | Code |
|---|---|---|
| §5.1, App. E.1 | 8 scene classes (few/medium/many × close/medium/far), 50 frequencies, ω_L ∈ [1.5, 2.5] GHz | `nefi/instances/nv_relaxometry/scenes.py::NVScenes`; `NVRelaxometryConfig` (`counts`, `separations`, `larmor_band`, `n_freq`) |
| App. E.1 | noise matched to per-sample dynamic range | `nefi/instances/nv_relaxometry/data.py::NVDataGenerator` |
| Tab. 1 | cross-fidelity benchmark (F3 data, F1/F2 inversion; GMSD, F1, SWD, MSE; mean ± 95 % CI, 3 seeds) | `nefi/bench/protocol.py::run_benchmark` with methods `netmy` (= default, F2), `f1`, `grid` (Tikhonov F2), `grid_f1`; `nefi bench configs/nv_relaxometry_paper.yaml --classes all --seeds 0,1,2` |
| Tab. 2, App. E.4 | matched-operator F1/F1, F2/F2 (inverse crime, labelled) | `run_benchmark(..., allow_inverse_crime=True)` with `NVRelaxometryConfig.data_operator` set to the inversion mode; CLI `--allow-inverse-crime` |
| Tab. 3, §5.4 | cumulative ablation (−TV, −annealed PE, −PE, −multiscale, −ℓ1, −gate, −R_ds) | `nefi/bench/ablation.py::cumulative_ablation` with `drop_loss("tv")`, `no_annealing()`, `no_positional_encoding()`, `single_stage()`, `drop_loss("l1")`, `no_gate()`, `drop_loss("direct_density")` |
| App. E.6, Tab. 10 | one-axis sweeps (LR, hidden width, PE octaves) | `nefi/bench/sweep.py::sweep(instance, "lr" \| "hidden" \| "n_octaves", values)` |
| Tab. 8 | F2/F2 leaderboard: density MSE, noise MSE, train time | `run_benchmark(allow_inverse_crime=True)` + `BenchmarkResult.efficiency_table`; noise-map fit via `nefi/diagnostics/report.py::data_fit_paradox` |
| App. E.2 — Tikhonov | free density, Adam 5e-3, ℓ2 + TV | `NVRelaxometry.build_grid_problem` (baseline `grid`); generic `nefi/baselines/__init__.py::baseline_problem(kind="grid")` |
| App. E.2 — ADMM | ℓ1 prox + box, μ = 1e-3 | `nefi/baselines/admm.py::ADMMSolver`, `prox_l1_box` |
| App. E.2 — GaussianSplat | K = 64 primitives, prune/split/clone/merge | `nefi/baselines/gaussian_splat.py::GaussianSplatField`, `SplatControl` |
| App. E.2 — DeepDecoder | untrained CNN prior | `nefi/baselines/deep_decoder.py::DeepDecoderField` |
| App. E.2 — L-BFGS | free density, history 20, Wolfe | `nefi/baselines/lbfgs.py::lbfgs_curriculum` |
| App. E.2/E.7 — U-Net, GAN, HybridNeTMY | supervised baselines | (n/a) supervised, not label-free |
| App. E.3 — GMSD | gradient-magnitude similarity deviation | `nefi/metrics/localization.py::gmsd` |
| App. E.3 — Hungarian F1 | peak matching at radius 2 px, 5 % threshold | `nefi/metrics/localization.py::hungarian_f1`, `peak_match`, `peak_positions` |
| App. E.3 — SWD | sliced Wasserstein, 128 projections | `nefi/metrics/localization.py::sliced_wasserstein` |
| App. E.3 — density MSE, masked SSIM | secondary metrics | `nefi/metrics/basic.py::mse`, `masked_ssim` |
| §5.3 — iter-0 gradient | center/outer ratio 18.29×, peak at grid center | `nefi/diagnostics/landscape.py::iter0_gradient`, `center_to_outer_ratio` |
| §5.3, Fig. 4b | energy barrier `h ≈ 1.12` at `t = 0.20` along `(1−t)ρ_collapse + tρ⋆` (F2); monotone (F1) | `nefi/diagnostics/landscape.py::energy_barrier` |
| §5.3, Fig. 4c | center-mass ratio at 200 iterations | `nefi/diagnostics/landscape.py::center_mass_ratio`, `uniform_center_mass` |
| App. E.5, Tab. 9 | ADMM center-mass aggregate 0.553 → 0.776 (F1 → F2) | `center_mass_ratio` over `run_benchmark` results of `ADMMSolver` runs |
| App. E.5, Fig. 10 | realized iter-0 update 1.6× vs raw 18.29× | `nefi/diagnostics/filtering.py::realized_update`; `DiagnosticsReport` "paper-style damping" |
| §5.5, App. E.8 steps 1–3 | α-RuCl3 depth-amplitude, power-law slope, spectral calibration | (n/a) requires the Kumar et al. dataset (no downloads in this repository) |
| §5.5, Fig. 5, App. E.8 step 4 | Hessian condition number on a Gaussian ansatz (`κ_F2 = 931` vs `κ_F1 = 301,139`) | `nefi/diagnostics/spectrum.py::hessian_condition_number`, `hessian_spectrum`; ansatz loss `nefi/diagnostics/landscape.py::ansatz_objective`; test: `test_hessian_condition_number_quadratic` |
| App. E.9 | runtime (median 273 s, mean 780 s per instance, A6000) | `BenchmarkResult` columns `time_s`, `ms_per_step`, `peak_mem_mb`; `efficiency_table` |
| App. E.10, Fig. 8 | cross-shaped artifacts on dense scenes | [faq.md](faq.md#cross-shaped-artifacts); `nefi/losses/reg.py::Laplacian`, `TV(isotropic=True)` |
| App. E.10, Fig. 9 | high-frequency leakage on small clusters | [faq.md](faq.md#high-frequency-leakage); gate + ℓ1, fewer octaves |
| App. E.10 | centered-minimum trapping under random init (~5 %) | `Curriculum.restarts`, [faq.md](faq.md#centered-collapse) |

---

## NeFTY

Instance: `nefi/instances/thermal_tomography/` (registered as `thermal_tomography`; presets
`paper`, `layered`, `smoke` in `nefi/instances/thermal_tomography/__init__.py::PRESETS`). Physics:
`nefi/operators/pde/` (`heat.py`, `stencil.py`, `linear_solvers.py`, `adjoint.py`).

### Forward model and ill-posedness (§3, App. A–C)

| Paper | Content | Code |
|---|---|---|
| Eq. (1)–(2), App. A.1–A.2 | heat equation, effective diffusivity `α = k/(ρ C_p)` | `nefi/operators/pde/heat.py::HeatOperator` (single effective α field) |
| Prop. 1, Eq. (12)–(14), App. A.3 | harmonic mean = discrete flux continuity | `nefi/operators/pde/stencil.py::face_coefficients(mode="harmonic")`; test: `tests/test_heat_solver.py::test_face_coefficient_values`, `test_harmonic_mean_throttles_flux_through_insulating_layer` |
| App. A.4 | multi-dimensional consequence, high contrast | `stencil.py::face_conductances`, `DiffusionStencil` |
| App. A.5 | initial conditions (Gaussian flash / near-uniform flash), BCs (periodic lateral, adiabatic / Robin back face) | `nefi/operators/pde/heat.py::GaussianFlash`, `UniformFlash`; `nefi/operators/pde/stencil.py::BoundarySpec` |
| Eq. (3) | reconstruction objective `Σ ‖K(α) − T̂‖² + λ R(α)` | `ThermalTomography.losses` (keys `data`, `tv`) |
| Prop. 2, Eq. (15)–(18), App. B.1–B.2 | compact linearized map, `σ_n ≲ n^(-1/3)` | measured on the discrete map by `nefi/diagnostics/spectrum.py::singular_values`; depth decay by `sensitivity_map` |
| Cor. 1, App. B.3 | Hadamard ill-posedness: noise amplified by `1/σ_n` | `singular_values` (+ [faq.md](faq.md)); mitigated by annealing + TV |
| App. B.4 | pointwise damping vs operator decay | [tutorials/05_diagnostics.md](tutorials/05_diagnostics.md) |
| Fig. 2 | distinct interiors → indistinguishable surfaces | `sensitivity_map` (surface-to-depth dynamic range), `data_fit_paradox` |
| Eq. (4), §3.3 | soft-constrained PINN loss | `nefi/instances/thermal_tomography/pinn.py::SoftPINNSurface` (temperature surrogate as operator), `PDEResidualLoss`, `InitialConditionLoss` — baseline `pinn_soft` of `thermal_tomography`; generic soft residual `nefi/losses/physics.py::PDEResidual` |
| Eq. (19), App. C.1–C.3 | PINN gradient decoupling `∇θ L_data = 0` | test: `tests/test_thermal_tomography.py::test_pinn_soft_baseline_decouples_data_from_diffusivity`; consequence measured by `data_fit_paradox`; [faq.md](faq.md) |

### Method (§4, App. D)

| Paper | Content | Code |
|---|---|---|
| Eq. (5), Eq. (20) | Fourier encoding of bandwidth N (no raw input) | `nefi/fields/encoding.py::FourierFeatures(include_input=False)`; `ThermalTomographyConfig.include_input`, `n_octaves` |
| Eq. (21), App. D.1 | cosine frequency annealing over `T_FA = 2500` steps | `FourierFeatures.band_weights`, `Stage.anneal_fraction` (`ThermalTomographyConfig.anneal_steps`) |
| Eq. (6), App. D.1 | bounded sigmoid output `α_min + (α_max − α_min) σ(f)` | `nefi/fields/heads.py::Bounded` |
| Tab. 5, App. D.4 | 10 × 512 ReLU MLP, skip at 4, Adam 5e-5, step decay ×0.1 / 1000, 10k steps, TV 1e-3 | `ThermalTomographyConfig` (`PRESETS["paper"]`), `NeuralField(activation="relu")`, `Stage(lr_schedule="step")` |
| Eq. (22) | isotropic TV, periodic lateral wrap, ε_TV | `nefi/losses/reg.py::TV(isotropic=True, periodic_axes=(0, 1))` |
| Eq. (7), App. D.2 | finite-volume diffusion operator `L(α)` | `nefi/operators/pde/stencil.py::DiffusionStencil`, `apply_diffusion` |
| Eq. (8), Eq. (23) | implicit Euler `A(α) T^{n+1} = T^n` | `nefi/operators/pde/heat.py::HeatOperator`, `stencil.py::ImplicitSystem` |
| Eq. (24) | unrolled Jacobi (K = 50) | `nefi/operators/pde/linear_solvers.py::jacobi` (`conjugate_gradient` alternative) |
| Eq. (9) | discrete objective on surface frames `Π_Γ T^{n_i}` | `nefi/operators/pde/adjoint.py::surface`, `HeatOperator` observation frames |
| Eq. (10)–(11), Eq. (25)–(27), App. D.3 | discrete adjoint, constant memory in `N_t` | `nefi/operators/pde/adjoint.py::ImplicitEulerAdjoint`, `implicit_euler_frames`; test: `tests/test_heat_solver.py::test_adjoint_matches_autograd`, `test_adjoint_gradcheck_alpha_and_T0`, `test_adjoint_memory_is_trajectory_sized` |

### Experiments (§5, App. E–I)

| Paper | Content | Code |
|---|---|---|
| §5.1, App. E.1 | PhiFlow explicit data (inverse-crime guard), adaptive substeps `N_sub = max(10, ⌈Δt/Δt_stable · 2⌉)` | `nefi/operators/pde/heat.py::ExplicitHeatSimulator`, `explicit_substeps` (`fidelity_tag` `explicit-substepped-float64` vs `implicit-euler-jacobi`) |
| App. E.1 | slab [0,10]²×[0,1], 64×64×16, 100 frames, 1–4 defects, homogeneous / layered, contrast ≈ 1:20 | `nefi/instances/thermal_tomography/scenes.py::ThermalScenes`; `ThermalTomographyConfig` |
| App. E.2 | MSE, PSNR (range α_max − α_min), slice-wise SSIM, IoU (α < 0.03) | `nefi/metrics/basic.py::mse`, `psnr`, `ssim`; `nefi/metrics/segmentation.py::iou_below` |
| Tab. 1 | label-free synthetic benchmark (mean ± 95 % CI) | `run_benchmark` on `thermal_tomography`, classes `homogeneous`, `layered` |
| Tab. 3, Tab. 8 | additive ablation BASE → +PE → +FA → +σ → +HM → +TV | `cumulative_ablation` with removals in reverse order: `drop_loss("tv")`, `set_config(face_mode="arithmetic")`, a head-swap `FnModifier` for σ, `no_annealing()`, `no_positional_encoding()` (read bottom-up) |
| Tab. 4, §5.5 | solver-level efficiency: autograd vs adjoint (fwd/bwd time, peak memory, simulation error) | `nefi/bench/report.py::runtime_table({"AD": ..., "AM": ...}, reference=...)` with `ThermalTomographyConfig.grad_mode` |
| Tab. 6, App. G.1 | robustness by defect count / layer count | `BenchmarkResult.table(by_class=True)` per scene class; per-defect / per-layer strata via `BenchmarkResult.table(by="scene/n_defects")` / `by="scene/n_layers"` (scene metadata flattened into benchmark rows by `nefi.bench.protocol.scene_columns`) |
| Tab. 7, App. G.2 | surface fidelity vs volumetric IoU (data-fit paradox) | `nefi/diagnostics/report.py::data_fit_paradox`; `ThermalTomography.surface_metrics` |
| Tab. 9, App. G.4 | training-level wall-clock / memory | `BenchmarkResult.efficiency_table` (params, steps, ms/step, wall-clock, peak memory) |
| App. G.5 | Edge F1 and radial power spectrum | `nefi/metrics/segmentation.py::edge_f1`, `radial_power_spectrum` |
| App. F.2 | Grid Opt. (voxel grid, same solver and schedule) | `ThermalTomography.baselines()["grid"]`, `nefi/baselines/__init__.py::baseline_problem(kind="grid")` |
| App. F.1 — PINN | vanilla soft-constrained PINN (fixed λ weights) | baseline `pinn_soft` (`nefi/instances/thermal_tomography/pinn.py`) |
| App. F.1 — variants | GradNorm balancing, SPINN, Causal-PINN, DCGD | **(planned)** extensions of `nefi/instances/thermal_tomography/pinn.py` |
| App. F.3, F.5 | supervised U-Net, Swin-UNETR | (n/a) supervised |
| App. F.4 | PPT / TSR thermography heuristics | `nefi/baselines/thermography.py::ppt`, `tsr`, `depth_to_alpha_volume`; thermal instance baselines `"ppt"`, `"tsr"` |
| §5.3, Tab. 2, App. H | real PVC: 2-D masks (MAD noise floor) and 2.5-D depth | `nefi/instances/thermal_tomography/projection.py::defect_mask_2d`, `depth_map_25d`, `mad_sigma`; metrics `nefi/metrics/segmentation.py::abs_rel`, `depth_rmse`, `delta_threshold`, `nefi/metrics/basic.py::iou`, `dice`; real-data path `nefi/instances/thermal_tomography/realdata.py::load_frames`, `measurement_from_frames`, `PhysicalScales` (Fourier-number rescaling, App. A.5/H.2), `pvc_overrides` (uniform flash, Robin back face, lateral Neumann), `config_for_frames`, `physical_depth_map` (no dataset is bundled; point it at a local copy of Wei et al. 2023) |
| App. H.2 | Robin back face, near-uniform flash, Fourier-number rescaling | `BoundarySpec` (Robin), `UniformFlash`, `ThermalTomographyConfig` (`bc_back`, `robin_h`, `initial`) |
| Tab. 10, App. H.4 | PVC depth metrics | as Tab. 2 |
| App. I, Eq. (28)–(32) | Gaussian variance growth `σ²(t) = σ0² + 2αt` | test: `tests/test_heat_solver.py::test_gaussian_variance_growth`, `test_gaussian_variance_growth_2d_uniform_depth_is_exact` |
| Fig. 14 | back-face artifact for shallow defects | [faq.md](faq.md#back-face-artifact) |
| App. E.1 | solver stall at 1:1000 contrast | [faq.md](faq.md#stalled-solver-at-high-contrast); `Bounded(α_min, α_max)`, `linear_solvers.conjugate_gradient` |

---

## Shared protocol elements

| Paper | Content | Code |
|---|---|---|
| NeTMY §3.1, NeFTY §5.1 | inverse-crime avoidance (independent data model) | `nefi/bench/base.py::DataGenerator.fidelity_tag`; guard `nefi/bench/protocol.py::check_inverse_crime`, `InverseCrimeError`; `operator_fidelity_tag` |
| NeTMY Tab. 1, NeFTY Tab. 1 | mean ± 95 % CI over samples × seeds | `nefi/bench/report.py::mean_ci` (Student t), `BenchmarkResult.table`, `to_markdown` |
| NeTMY Tab. 3, NeFTY Tab. 3 | cumulative ablations | `nefi/bench/ablation.py::cumulative_ablation`, `Modifier` |
| NeTMY Tab. 10 | hyperparameter sweeps | `nefi/bench/sweep.py::sweep` |
| NeTMY App. E.9, NeFTY Tab. 4 / Tab. 9 | runtime and memory | `run_benchmark` columns, `efficiency_table`, `runtime_table` |
| Both, §4 | multiscale + annealing curriculum | `nefi/solve/curriculum.py::Curriculum`, `Stage` |
| Both | per-measurement, label-free solve | `nefi/solve/solver.py::Solver`, `nefi.invert` |
| NeFTY Limitations | uncertainty in low-sensitivity regions | `nefi/solve/ensemble.py::ensemble` (seed ensembles), `sensitivity_map` as a trust mask |
| DESIGN §1.6 | diagnostics as library features | `nefi/diagnostics/` (`diagnose` → `DiagnosticsReport`) |

`docs/DESIGN.md` §6 is the condensed version of this table; when a planned item lands, replace
**(planned)** by its path and add the pinning test.
