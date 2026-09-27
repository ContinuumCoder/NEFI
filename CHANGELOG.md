# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow semantic versioning.

## [Unreleased]

## [0.1.1] - 2026-09-27

### Fixed
- The `nefi` command-line front-end works with typer 0.27 and later, which ship their own copy
  of click instead of depending on the `click` package.

### Changed
- Package summary, README, site and citation copy describe the library in plain language.
- The `release` workflow smoke-tests the typer front-end as well as the argparse fallback and
  creates a GitHub Release with the built distributions and the matching changelog section.
- The documentation site enables GitHub Pages on its first deployment and links the CAB Lab.

## [0.1.0] - 2026-09-27

First public release: the core library, the NeTMY (NV relaxometry) and NeFTY (thermal
tomography) instances with paper-scale configs, diagnostics, benchmark protocol and CLI.

### Added
- **Core**: `Domain`, `Measurement`, `Field` (`NeuralField` with annealed Fourier features,
  `GridField`), heads (`Softplus`, `Bounded`, `GatedSoftplus`, `SupportMasked`, ...), operators
  (`FFTConvolution`, `Nuisance`, `LambdaOperator`), losses (MSE, log-MSE, normalized MSE, Huber,
  Poisson NLL, L1, TV, Laplacian, ...), `Curriculum` / `Solver` (multiscale stages, annealing reset,
  grad clipping, NaN guard, EMA, early stopping, discrepancy principle, restarts, time budget),
  post-processing (energy-anchored scale correction), registry and YAML configs.
- **Diagnostics** (`nefi.diagnostics`): sensitivity maps (Hutchinson / exact Jacobian column
  norms), iter-0 field gradient and center/outer ratio, center-mass ratio, energy barrier along
  interpolation paths, filter-kernel rows `G_θ e_i`, realized first update vs raw gradient,
  effective encoding bandwidth, top-k singular values of the linearized operator (Lanczos on
  jvp/vjp), Hessian condition number on low-dimensional ansätze, data-fit paradox report,
  `diagnose()` → `DiagnosticsReport` (markdown / JSON), matplotlib plot helpers.
- **Benchmarking** (`nefi.bench`): `run_benchmark` (paired samples × seeds, Student-t 95 % CIs,
  wall-clock / steps / peak-memory columns, inverse-crime guard with an explicit matched-operator
  regime), `cumulative_ablation` with generic modifiers (`no_annealing`,
  `no_positional_encoding`, `single_stage`, `drop_loss`, `set_weight`, `grid_field`, `no_gate`,
  `set_config`, `set_stage`, ...), one-axis `sweep`, markdown / CSV / JSON reports and an
  operator `runtime_table` (forward / backward time, peak memory, simulation error).
- **CLI** (`nefi`): `list`, `run`, `bench`, `diagnose`, `ablate`, `sweep`; typer front-end with
  an argparse fallback; config YAMLs, `--set key=value` overrides and `--smoke` runs.
- **Instances**: `toy1d` plus the paper instances and additional exemplars (see `nefi list`).
- **Baselines**: grid (Tikhonov / Grid Opt.), L-BFGS, ADMM, Gaussian splats, DeepDecoder, direct
  reconstructions.
- **Deployment**: `scripts/run_server.sh`, `scripts/slurm_template.sbatch`,
  `scripts/sync_to_server.sh`, `environment.yml`, `requirements.txt`, GitHub Actions CI,
  pre-commit, `Makefile` (`test`, `lint`, `docs`, `smoke`).
- **Documentation**: tutorials 01–08, FAQ, paper → code mapping, generated API reference.
- **Documentation site** (MkDocs Material, <https://continuumcoder.github.io/NEFI/>): a landing
  page, navigation by task (get started, guides, gallery, physics, instances, research,
  reference), a gallery of all 17 systems with a 3-D showcase of interactive viewers, an "At a
  glance" card on every instance page, a physics overview and a heat-solver page, research pages
  (the two papers, the representation-is-the-prior thesis, limitations and caveats), CLI and
  configuration references; `tools/build_site_assets.py` and `tools/build_brand_assets.py`
  regenerate the images, viewers, logo and diagrams from a gallery run; the site is built in
  strict mode and published on GitHub Pages.
- **Repository**: `CITATION.cff` (the software and both papers), issue forms (bug, feature, new
  instance) and a pull-request template.
- **Bring-your-own-problem layer** (`nefi.from_forward`, `nefi.priors`, `nefi.auto`): wrap any
  differentiable function, declare priors in a small DSL, and get data-matched initialization,
  noise estimation, balanced weights, learning-rate probing, an automatic curriculum with
  discrepancy stopping and a `quick_report`. Operator zoo (`FunctionOperator`, `Downsample`,
  `Sampling`, `FourierSampling`, `PhaseRetrieval`, `BeerLambert`, `Saturation`, `TimeStepper` with
  checkpointed adjoints), representation zoo (hash grid, low rank, level set, parametric,
  symmetric, head modifiers) and physics losses (`PDEResidual`, `Conservation`, `SymmetryLoss`,
  `KnownSupportLoss`, `Monotone`, `GradientL2`).
- **Physics zoo** (`nefi.physics`): elliptic family (variable-coefficient Poisson with
  implicit-function-theorem adjoint; magnetostatic FFT kernels) behind the `eit`, `darcy_flow` and
  `current_density` instances; wave / optics / reaction family (acoustic wave with PML and
  checkpointing, Born / Lippmann–Schwinger scattering with the Vico–Greengard truncated kernel,
  angular-spectrum propagation and holography, Gray–Scott reaction–diffusion) behind `wave_fwi`,
  `diffraction_tomography`, `holography` and `reaction_diffusion`.
- **Heat PDE** (`nefi.operators.pde`): finite-volume stencil with harmonic-mean faces and
  periodic / adiabatic / Robin boundaries, Jacobi and CG inner solvers, implicit-Euler discrete
  adjoint with fused gradient assembly (O(N_g N_t) memory), explicit substepped simulator for
  inverse-crime-safe data; `thermal_tomography` instance with projections, soft-PINN baseline and
  segmentation / depth metrics.
- **NV relaxometry** (`nefi.instances.nv_relaxometry`): dipolar kernels, F1 / F2 FFT operators,
  F3 direct simulator, direct-density loss, eight scene classes, localization metrics (GMSD,
  Hungarian F1, sliced Wasserstein).
- **Adaptive geometric representations** (`nefi.fields.geometric`, `nefi.fields.adaptive`):
  composite / anomaly, layered / layer-cake, star-shape / polygon, warps (polar, depth,
  sensitivity, deformable), spectral preconditioning, Fourier basis; residual- and
  operator-aware annealing, capacity growth, held-out representation selection and ensembles,
  representation-vs-operator spectrum match reports.
- **Visualization** (`nefi.viz`): house style, field / measurement / training / qualitative /
  performance viewers, multi-physics gallery and self-contained HTML reports;
  `examples/gallery.py`, `examples/visualize_result.py`.
- **Thermography heuristics** (`nefi.baselines.thermography`): PPT and TSR pixel-wise depth maps
  (NeFTY App. F.4) lifted to volumetric fields; `"ppt"` / `"tsr"` baselines of `thermal_tomography`.
- **Auto-tuning** (`nefi.autotune`, `nefi autotune TARGET [--level] [--compare]`): gauge
  detection / repair (`detect_gauges`, `repair_gauges`: invisible additive constants, scales under
  scale-free fidelities, sign flips and user directions → `ZeroMean` / `MeanAnchor` heads,
  `EnergyScaleCorrection` with the measured homogeneity degree, `LeastSquaresScale`,
  `SignConvention`, `DirectionPenalty`), fit-quality driven budgets and learning rates
  (`probe_convergence`, `tune_budget`: doubling probes against RMSE/σ, at-floor / converging /
  stalled), Morozov regularization with warm-started trials and a GradNorm fallback
  (`tune_regularization`), an acquisition / identifiability report (`acquisition_report`:
  coverage, singular-value head, data count, weakest sides, match report), a held-out Sobol
  search (`autotune`), one-call orchestration with a decisions report (`autotune_problem`,
  `autotune_instance`, levels quick / standard / thorough), a benchmark method
  (`autotuned_method`) and `tuned_config.yaml` for `nefi run`. On four deliberately
  under-configured problems (level standard): eit 16.5 → 23.1 dB, poisson_source 22.2 → 27.3 dB (3 of 3
  sources), holography raw PSNR 6.8 → 23.3 dB; wave_fwi 2 × 8 is fitted to the noise floor
  (RMSE/σ 2.9 → 1.03) without a better field, and flagged as under-determined
  (`docs/autotune.md`).
- **Adaptive loss balancing** (`nefi.solve.GradNormBalancing`): GradNorm-style callback that
  rescales the active stage's loss weights so weighted gradient norms follow target shares.
- **Edge refinement** (`nefi.solve.refine`, `nefi.refine_edges`, `Instance.refine`): a short,
  physics-constrained second stage from a smooth result — `mode="levelset"` (k nested phases over
  one φ, `MultiPhaseHead` with cell-averaged Heavisides for sub-voxel interfaces, learnable or free
  phase values, grid or warm-started MLP φ), `"phasefield"` (growing Cahn–Hilliard multi-well
  penalty), `"tv_sharpen"` (stronger TV + binarizing `LevelSetHead` swap) and the `"continue"`
  control; penalties scaled to γ = fit_tol² − 1 of the noise-floor data loss; a `RefineReport`
  (data fit / χ before → after, phases, transition-band check, IoU / volume-matched IoU /
  Edge-F1 / instance metrics) and refusal when χ rises above 1.1 × max(χ_smooth, 1) or an interface
  leaves the smooth transition band; threshold helpers (`multi_otsu`, `volume_matched_threshold`,
  `mass_matched_fraction`, `estimate_levels`); `REFINE_DEFAULTS` for the five 3-D instances, eit and
  deconvolution; `refine_run_output` (gallery hook) and `refine_method` (benchmark
  `neural+refine`); `examples/refine_edges.py` benchmark; `docs/refinement.md`.
- **Solver**: `OptimConfig.lr_mult` per-prefix learning-rate multipliers; `progress_override` /
  `stop_stage` hooks for adaptive annealing; early / discrepancy stops wait for annealing and
  results are evaluated at the trained progress; fractional mask weights when coarsening
  measurements; complex tensors survive dtype casts.
- **Smoke presets sized for demonstration** (what the multi-physics gallery runs at `--budget 0.1`):
  `eit` 24² with 12 current patterns and an 8-12× inclusion; `wave_fwi` surround acquisition
  (6 sources × 32 receivers) with a near-offset mute (`min_offset`, `TraceDataGenerator`,
  `WaveFWI.offset_mask`); `poisson_source` 30 % observations and a ReLU field; `nv_relaxometry`
  at a 3-pixel standoff with 3-5 merging `few/close` sources; `current_density` at a 4-pixel
  standoff. `holography` fixes the invisible global phase in the model (new
  `nefi.fields.ZeroMean` head, zero-mean phantoms); `EIT.difference_data` / `measurement_image`
  (difference-EIT boundary view); `viz_hints` on `current_density`, `nv_relaxometry`,
  `holography`.

### Performance (see `docs/performance.md`)
- **Heat stencil**: flat padded layout (`nefi.operators.pde.stencil.FlatLayout`) — one ghost-refill
  gather + six `addcmul` on contiguous views, preallocated buffers: 7 kernels per Jacobi sweep
  instead of 12 (`torch.roll`), same numbers to float rounding; `stencil_backend="auto"|"flat"|"roll"`.
  Thermal smoke step 21.2 → 9.4 ms on CPU (15.4k → 9.3k ops); 4.5× less memory in the unrolled
  `grad_mode="autograd"` reference. The adjoint reuses the forward system and accumulates the face
  products in range layout.
- **`solver="chebyshev"`** (opt-in): Chebyshev-accelerated Jacobi with a Gershgorin bound and
  on-device weights — the accuracy of 50 Jacobi sweeps with ≈ 20 iterations at the Tab. 5 step.
- **`Solver(compile="field" | "step", cuda_graphs=..., autocast=...)`**: whole-step compile
  (field → operator → losses, one graph per stage, fresh code object per stage, annealing progress
  as a tensor — the previous `compile=True` recompiled at every annealing step and gave up after 8),
  eager first step per stage, eager fallback with a warning; `Operator.traceable` (time loops run
  eagerly between the compiled graphs); `Loss.step_dependent`; mixed precision for the field MLP
  only (fp32 output projection, heads and physics; GradScaler for fp16).
- **`nefi.batch_invert`**: N problems in one loop (stacked flat parameter groups, `vmap` +
  `functional_call`, one batched operator call, per-problem losses / Adam(W) / clipping / NaN guard
  / EMA / stopping); `Operator.batchable` (FFT convolution, NV F1/F2, Poisson, Born, holography,
  Radon, magnetostatics, pointwise, `Nuisance`/`Sequential`/`Sum` wrappers); batched == sequential
  to rounding; up to 8× throughput on CPU.
- **Benchmarks**: `run_benchmark(batched=True, batch_size=..., shard=(i, n))`, `nefi bench --batched
  --batch-size --shard i/n`, `BenchmarkResult.merge` / `nefi bench-merge DIR`; SLURM array example.
- **Micro**: `FourierFeatures` caches (sin/cos per coordinate tensor, gates per progress; bitwise
  identical), `foreach` Adam(W) for small CPU models (bitwise identical, ≈ 30 % faster step).
- `configs/thermal_tomography_paper.yaml` enables `compile_solver: true`; heat compiled sweeps clone
  their output under CUDA-graph modes; `tools/profile_instance.py` runs the real solver loop and
  reports ops/step.
