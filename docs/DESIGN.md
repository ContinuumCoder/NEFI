# nefi — Neural-Field Inversion: architecture and contract

`nefi` is a general-purpose library for **physics-faithful, label-free inverse problems** solved by
per-measurement (test-time) optimization of a **coordinate neural field** through a **differentiable
forward operator**. It generalizes two Princeton CAB-lab papers into one reusable infrastructure:

| Paper | Problem | Unknown field(s) | Forward operator | Key tricks |
|---|---|---|---|---|
| **NeTMY** (arXiv 2605.13988) | NV-center noise relaxometry | spin density ρ ≥ 0 (2D), Larmor field ω_L | FFT convolution with tensor power-summed dipolar kernels ×Lorentzian (F2); scalar-coherent F1; direct source-side F3 | gated softplus, support-masked Larmor head, log-MSE on max-normalized maps, mean-normalized companion loss, direct-density proxy, energy-anchored scale correction, 2-stage multiscale (32→64), annealed Fourier PE (K=12) |
| **NeFTY** (arXiv 2603.11045) | 3D inverse heat conduction (pulsed thermography NDE) | diffusivity α(x) ∈ [α_min, α_max] (3D) | implicit-Euler finite-volume heat solver, harmonic-mean face coefficients, Jacobi inner solve, **discrete adjoint** (O(N_g) memory) | bounded sigmoid, isotropic TV, frequency annealing, surface-only L2 fidelity, inverse-crime guard (independent explicit simulator), 2D-mask / 2.5D-depth projections |

Both are instances of ONE recipe, which is what `nefi` implements:

```
coords ──► Encoding(annealed) ──► MLP ──► Heads(bounded/gated/masked) ──► fields{ρ, ω_L | α}
                                                                          │
                                        Operator (differentiable physics) ▼
                                                                    prediction ──► Losses(data + reg + physics) ──► backprop to θ
Curriculum: stages (resolution, steps, lr, annealing reset) ── Solver ── Postprocess(scale correction, projection) ── Metrics/Diagnostics
```

The full paper texts are in `docs/_paper_netmy_2605.13988.txt` and `docs/_paper_nefty_2603.11045.txt`.
`docs/paper_mapping.md` (to be written) maps every paper equation/table to the code that implements it.

---

## 1. Design principles

1. **Physics is a hard constraint.** The forward operator is code, not a loss term. Soft PDE residual
   penalties (PINN-style) exist only as *baselines* to demonstrate the pathology (NeFTY §3.3, App. C).
2. **Parameterization is the prior.** Regularization comes first from the field representation
   (neural field ⇒ smoothing filter kernel G_θ = J_θ J_θᵀ, NeTMY Lemma 2), second from explicit penalties.
   Everything is a `Field`; baselines are just other `Field`s or other `Solver`s.
3. **Per-measurement, label-free by default.** No paired training data. Amortization/warm-starting is
   an optional accelerator, never a requirement.
4. **Multiscale + annealing everywhere.** Coarse-to-fine resolution and frequency annealing are first-class
   (`Curriculum`), and every `Operator` must be evaluable at any resolution (`at_resolution`).
5. **Inverse-crime guard.** Benchmarks must generate data with a *different* operator/discretization than
   the one used for inversion (NeTMY F3 vs F2; NeFTY PhiFlow-explicit vs implicit-Euler). The bench
   protocol enforces this by construction (`DataGenerator.operator is not problem.operator`).
6. **Diagnose, don't just score.** Ill-posedness diagnostics (sensitivity maps, iter-0 gradient, center-mass
   ratio, energy barrier, filter-kernel rows, singular-value decay, radial spectrum, data-fit paradox) are
   library features, because papers of this kind live or die by them.
7. **Small, dependency-light, CPU-friendly.** Hard deps: `torch`, `numpy`, `scipy`, `pyyaml`, `tqdm`.
   Optional: `rich`, `typer`, `matplotlib`, `scikit-image`. Everything must run on CPU at small sizes
   in tests (< 2 min total). GPU (CUDA/MPS) is a device choice, not a requirement.
8. **Config = dataclasses + YAML.** No Hydra/OmegaConf. Every public class is constructible from a plain dict;
   `nefi run config.yaml` builds the whole problem via the registry.
9. **Reproducible.** Global seeding, deterministic ops where possible, config hash in every `Result`.

---

## 2. Package layout

```
nefi/
  __init__.py            # public API re-exports + __version__
  domain.py              # Domain: physical extents, shape, normalized coordinate grids, spacing
  measurement.py         # Measurement: data tensor + mask + noise_std + meta
  registry.py            # @register("field"|"operator"|"loss"|"instance"|...) + build(kind, cfg)
  config.py              # dataclass <-> dict/yaml helpers, config hashing
  fields/
    base.py              # Field ABC, Heads container
    encoding.py          # FourierFeatures (annealed, Nerfies cosine gate), Identity, (HashGrid optional later)
    heads.py             # Head transforms: Identity, Softplus, GatedSoftplus, Bounded(sigmoid), Exp, SupportMasked
    neural.py            # NeuralField: coordinate MLP + skip + heads (NeTMY Tab.5 / NeFTY Tab.5 defaults)
    grid.py              # GridField: free pixel/voxel tensor + heads (Tikhonov / Grid-Opt baseline), resample across stages
  operators/
    base.py              # Operator ABC (homogeneity, at_resolution, output_shape), Compose, Scale/Offset nuisance
    conv.py              # FFTConvolution: generic N-D linear convolution with kernel cache per resolution
    pde/                 # heat solver, linear solvers, discrete adjoint
  losses/
    base.py              # Loss ABC, Context, LossSet (weighted sum, per-stage overrides, component logging)
    data.py              # L2, MSE, LogMSE(normalize=max|mean|none), NormalizedMSE, Huber, PoissonNLL, Masked wrappers
    reg.py               # L1, TV(isotropic/anisotropic), Laplacian, Tikhonov(L2), NonNegativity (for grids)
  solve/
    curriculum.py        # Stage, Curriculum (multiscale, annealing, lr schedules, weight overrides), OptimConfig
    solver.py            # Solver: runs stages; EMA, grad-clip, NaN guard, early stop, discrepancy stop, restarts, callbacks;
                         #   opt-in compile="field"|"step", cuda_graphs, autocast (docs/performance.md)
    batched.py           # batch_invert: N independent problems in one loop (stacked params, vmap, batchable operators)
    result.py            # Result (fields, history, timing, config hash, post info), save/load
    callbacks.py         # Callback ABC, Logger, ProgressBar, Checkpoint, PlotEvery (matplotlib optional)
    postprocess.py       # Postprocess ABC: EnergyScaleCorrection, ThresholdMask, Clip
    ensemble.py          # multi-seed ensemble -> mean/std maps (uncertainty)
    refine.py            # edge refinement: a short second stage from a smooth Result (level set /
                         #   phase field / TV sharpen / control), same operator + data, RefineReport
  metrics/
    basic.py             # mse, psnr, ssim, iou, dice, masked variants; evaluate()
    localization.py      # gmsd, hungarian_f1, sliced_wasserstein
    segmentation.py      # edge_f1, depth metrics (abs_rel, rmse, delta thresholds), radial_power_spectrum
  diagnostics/           # sensitivity_map, iter0_gradient, center_mass_ratio, energy_barrier,
                         #            filter_kernel_row, realized_update, singular_values, hessian_condition, data_fit_paradox
  bench/                 # SceneGenerator/DataGenerator ABCs, run_benchmark, ablation, sweep, report (md/csv/json), CI95
  auto.py, priors.py     # bring-your-own-problem layer: from_forward, prior DSL, adaptive defaults
  physics/               # physics zoo: elliptic (IFT adjoint), magnetostatics, wave (PML), scattering, optics, reaction_diffusion, timestep
  fields/geometric/      # composite, layered, shape, warp, spectral representations (representation = geometric prior)
  fields/adaptive/       # residual/operator-aware annealing, capacity growth, held-out selection, spectrum match reports
  viz/                   # figures, galleries, HTML reports
  baselines/             # admm.py, lbfgs.py, gaussian_splat.py, deep_decoder.py, pinn_soft.py
  instances/             # strong exemplars, each a subpackage with: operator(s), scene generator, losses, default config, run()
    toy1d/               # (core) 1-D blurred-signal recovery; the 5-second tutorial + CI smoke test
    nv_relaxometry/      # NeTMY
    thermal_tomography/  # NeFTY
    deconvolution/       # 2-D image deblurring (Gaussian / Poisson noise)
    sparse_view_ct/      # differentiable Radon, limited angles
    poisson_source/      # elliptic PDE source recovery with adjoint
    eit/, darcy_flow/, current_density/            # elliptic family (nefi.physics.elliptic / magnetostatics)
    wave_fwi/, diffraction_tomography/, holography/, reaction_diffusion/   # wave / optics / reaction family
  data/
    synthetic.py         # shared shape/scene primitives (blobs, ellipsoids, boxes, point sources)
  utils/
    seed.py, device.py, tensor.py, timing.py
  cli.py                 # `nefi run/bench/list/diagnose`
configs/                 # YAML for each instance
examples/                # runnable scripts, one per instance + diagnostics demo
tests/                   # pytest, fast
docs/                    # DESIGN.md (this), paper_mapping.md, tutorials/, api.md
```

---

## 3. Core contracts (exact signatures — every module builds against these)

### 3.1 `Domain`
```python
@dataclass(frozen=True)
class Domain:
    shape: tuple[int, ...]                     # native grid shape, e.g. (64, 64) or (64, 64, 16)
    extent: tuple[tuple[float, float], ...]    # physical [lo, hi] per axis (same length as shape)
    axes: tuple[str, ...] | None = None        # optional names ("x","y","z")

    @property
    def ndim(self) -> int
    def spacing(self, shape=None) -> tuple[float, ...]        # physical cell size per axis
    def coords(self, shape=None, device=None, dtype=None) -> Tensor   # (*shape, ndim) in [-1, 1]^ndim, cell-centered
    def physical_coords(self, shape=None, ...) -> Tensor      # (*shape, ndim) in physical units
    def at(self, shape) -> "Domain"                            # same extent, different resolution
    def coarsen(self, factor=2) -> "Domain"
```
Coordinates are normalized to `[-1, 1]` (NeTMY App. D.1, NeFTY App. D.1) and are **cell-centered**.

### 3.2 `Measurement`
```python
@dataclass
class Measurement:
    data: Tensor                      # observed measurement (any shape the operator produces)
    mask: Tensor | None = None        # 1 = observed, 0 = missing (same shape as data or broadcastable)
    noise_std: float | Tensor | None = None   # if known: enables discrepancy-principle stopping & noise-aware weighting
    meta: dict = field(default_factory=dict)
    def to(self, device, dtype=None) -> "Measurement"
```

### 3.3 `Field`
```python
class Field(nn.Module):
    """Parameterization of the unknown(s). Maps normalized coords -> dict[name, Tensor]."""
    heads: Heads                                     # ordered mapping name -> Head
    def forward(self, coords: Tensor, progress: float = 1.0) -> dict[str, Tensor]
        # coords: (*shape, ndim). Returns {name: Tensor(*shape)} for every head.
        # progress in [0,1] drives annealing (β = progress * K). progress=1.0 -> fully unmasked.
    def raw(self, coords, progress=1.0) -> Tensor      # pre-head outputs (*shape, n_out)
    def on_stage_start(self, stage: "Stage", domain: Domain) -> None   # hook (e.g. GridField resamples)
    def primary(self) -> str                            # name of the primary field (for scale correction / metrics)
```
`Heads` applies each `Head` to its slice of the raw output; heads may depend on other heads' *outputs*
(e.g. `SupportMasked(depends_on="rho")`), resolved in declaration order.

Head transforms (`fields/heads.py`), each `Head(n_in: int)` with `forward(raw_slice, others: dict) -> Tensor`:
- `Identity`, `Softplus(beta=1)`, `Exp`, `Bounded(lo, hi)` (= lo + (hi-lo)·σ(x), NeFTY Eq. 6, NeTMY Larmor band),
- `GatedSoftplus()` (= softplus(h)·σ(g), n_in=2, NeTMY Eq. 5),
- `SupportMasked(inner: Head, depends_on: str, tau: float = 0.3, fill: float = 0.0)`
  (hard mask 1{ρ > τ·max ρ} with **stop-gradient**, NeTMY Eq. 26).

`NeuralField(domain_ndim, heads, hidden=256, depth=6, skip_at=3, activation="tanh"|"relu"|"sine", encoding=FourierFeatures(n_octaves=12, base=2.0, include_input=True, annealed=True), init="xavier")`.

`GridField(shape, heads, init=0.0|Tensor)`: raw parameters are a tensor of shape `(*shape, n_raw)`; `on_stage_start` resamples (trilinear) to the stage resolution.

### 3.4 `Encoding`
```python
class Encoding(nn.Module):
    out_dim: int
    def forward(self, coords: Tensor, progress: float = 1.0) -> Tensor
class FourierFeatures(Encoding):
    # γ_β(x) = [x, w_k(β) sin(2^k π x), w_k(β) cos(2^k π x)]_{k<K},
    # w_k(β) = (1 - cos(π clip(β - k, 0, 1)))/2,  β = progress * K   (NeTMY Eq. 27, NeFTY Eq. 21)
    def __init__(self, in_dim, n_octaves=12, base=2.0, include_input=True, annealed=True, scale=math.pi)
    def band_weights(self, progress) -> Tensor   # (K,) for logging/tests
```

### 3.5 `Operator`
```python
class Operator(nn.Module):
    """Differentiable forward model. fields -> prediction tensor."""
    primary: str                              # name of the field the homogeneity refers to
    homogeneity: float | None = None          # F(c·x_primary) = c^p F(x). 1 for F2/heat-linear, 2 for F1, None if unknown
    batchable: bool = False                   # forward accepts fields (B, *shape) -> (B, *out), row b = unbatched output of field b
                                              # (plain differentiable tensor ops; required by nefi.batch_invert)
    traceable: bool = True                    # False for Python time loops / iterative solvers / custom autograd Functions:
                                              # Solver(compile="step") runs such operators eagerly between compiled graphs
    def forward(self, fields: dict[str, Tensor]) -> Tensor
    def at_resolution(self, shape: tuple[int, ...]) -> "Operator"   # return an operator evaluating fields sampled at `shape`
                                                                    # (rebuild kernel caches, grids...). Default: self.
    def output_shape(self, shape) -> tuple[int, ...] | None          # optional
    def required_fields(self) -> tuple[str, ...]                     # names it consumes
```
Operators must be pure functions of their inputs (no hidden state that changes across calls) so that
diagnostics can differentiate through them. Caches keyed on (shape, device, dtype) — FFT kernels, initial
states, stencil layouts — are allowed. Nuisance parameters (global gain, offset, background) are
modeled by `operators.base.Nuisance` wrappers whose parameters are optimized alongside the field
(robustness to model mismatch on real data).

### 3.6 Losses
```python
@dataclass
class Context:
    fields: dict[str, Tensor]        # field outputs at current resolution
    pred: Tensor                      # operator(fields)
    obs: Measurement                  # observation (possibly downsampled to stage resolution by the instance)
    domain: Domain                    # current-stage domain
    stage: "Stage"; step: int; progress: float
    operator: Operator; field_module: Field

class Loss(nn.Module):
    name: str
    def forward(self, ctx: Context) -> Tensor   # scalar

class LossSet(nn.Module):
    def __init__(self, losses: dict[str, Loss] | list[Loss], weights: dict[str, float])
    def forward(self, ctx) -> tuple[Tensor, dict[str, float]]   # total, detached components
    def with_weights(self, overrides: dict[str, float]) -> "LossSet"   # per-stage override (NeTMY App. D.4 rebalancing)
    def auto_balance(self, ctx, target=1.0) -> dict[str,float]  # optional: rescale so every term is O(target) at step 0
```
Data losses take `field=None` and read `ctx.pred`/`ctx.obs`; regularizers take `field: str`.
Reductions are means over elements (not sums) so weights are resolution-independent.
`TV` supports `isotropic` (NeFTY Eq. 22, with eps) and anisotropic (NeTMY Eq. 3), periodic/nonperiodic axes,
and physical spacing.

### 3.7 Curriculum & Solver
```python
@dataclass
class Stage:
    name: str = "stage"
    shape: tuple[int, ...] | None = None    # None -> domain.shape
    steps: int = 1000
    lr: float = 1e-3
    lr_schedule: str = "cosine"              # "cosine" | "step" | "constant" | "warmup_cosine"
    lr_min_ratio: float = 0.01               # cosine floor (NeTMY: 0.01·η_base)
    anneal: bool = True                      # reset β to 0 at stage start and ramp to K over anneal_fraction of steps
    anneal_fraction: float = 1.0             # NeFTY: 2500/10000 = 0.25
    loss_weights: dict[str, float] | None = None
    freeze: tuple[str, ...] = ()             # parameter name prefixes to freeze in this stage

@dataclass
class OptimConfig:
    optimizer: str = "adamw"                  # "adam" | "adamw" | "lbfgs" | "sgd"
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0
    ema: float | None = None                  # EMA of parameters for evaluation
    betas: tuple[float, float] = (0.9, 0.999)

@dataclass
class Curriculum:
    stages: list[Stage]
    optim: OptimConfig = OptimConfig()
    early_stop_patience: int | None = None    # on data loss plateau
    discrepancy_tau: float | None = None      # Morozov: stop when data RMSE <= tau * noise_std (needs Measurement.noise_std)
    restarts: int = 1                         # multi-restart, keep best data loss (mitigates centered-collapse trapping)
    time_budget_s: float | None = None
    @staticmethod
    def multiscale(shape, n_stages=2, steps=(3000, 7000), lr=1e-3, lr_decay=0.5) -> "Curriculum"  # NeTMY default

class Solver:
    def __init__(self, problem: "InverseProblem", curriculum: Curriculum, *, device="auto", dtype=torch.float32,
                 callbacks: list[Callback] = (), seed: int | None = 0, keep_fields_in_state=True,
                 compile: bool | str = False,          # False | True/"field" | "step" (field -> operator -> losses, one graph
                                                       #   per stage, dynamic=False, progress passed as a 0-dim tensor)
                 nan_guard=True, checkpoint_every=25, max_bad_steps=10,
                 cuda_graphs: bool = False,            # with compile: mode="reduce-overhead" (CUDA)
                 autocast: str | None = None)          # None | "bf16" | "fp16": field MLP only; physics + losses in dtype
    def run(self) -> Result

def batch_invert(problems: Sequence[InverseProblem], curriculum=None, *, device="auto", dtype=torch.float32,
                 seeds=0, nan_guard=True, checkpoint_every=25, max_bad_steps=10, loss_mode="auto",
                 check=True, raise_on_error=True) -> list[Result]
    # N problems sharing domain, field architecture and a batchable operator, solved in one loop: stacked
    # parameters, vmap(functional_call) fields, one batched operator call, per-problem mean-reduced losses
    # (objective = Σ_b L_b), per-problem Adam(W)/clip/NaN guard/EMA/early stop. == sequential up to rounding.
```
Loss terms whose value depends on ``ctx.step`` set ``Loss.step_dependent = True`` (``compile="step"``
holds the step index constant inside the graph and compiles only the field for such stages).

### 3.8 `InverseProblem` and `Result`
```python
@dataclass
class InverseProblem:
    domain: Domain
    field: Field
    operator: Operator
    losses: LossSet
    measurement: Measurement
    postprocess: list[Postprocess] = ()
    downsample_obs: Callable[[Measurement, tuple[int,...]], Measurement] | None = None  # for stages at lower resolution
    name: str = "problem"

    def evaluate(self, shape=None, progress=1.0) -> tuple[dict[str,Tensor], Tensor]   # fields, pred
    def loss(self, shape=None, progress=1.0, stage=None, step=0) -> tuple[Tensor, dict]

@dataclass
class Result:
    fields: dict[str, Tensor]        # final (post-processed) fields at final resolution (detached, cpu)
    raw_fields: dict[str, Tensor]    # before postprocess
    pred: Tensor
    history: dict[str, list[float]] # per-step components + lr + step + stage index
    stage_results: list[dict]
    timing: dict[str, float]
    post_info: dict                  # e.g. {"scale_factor": α}
    config_hash: str
    def save(path); @staticmethod load(path)
```
`nefi.invert(problem, curriculum=None, **kw) -> Result` is the one-call entry point (default curriculum
= `Curriculum.multiscale(domain.shape)`).

### 3.9 Postprocess
```python
class Postprocess:  def __call__(self, fields, pred, problem) -> tuple[dict[str,Tensor], dict]   # returns new fields + info
EnergyScaleCorrection(field="rho")  # α = (E_obs/E_pred)^(1/p), p = operator.homogeneity  (NeTMY Eq. 30, Prop. 1, Eq. 21)
```

### 3.9b Edge refinement (`nefi.solve.refine`, see `docs/refinement.md`)
```python
refine_edges(problem, result, *, mode="levelset" | "phasefield" | "tv_sharpen" | "continue",
             levels="auto" | k | (lo, hi, ...), ...) -> tuple[Result, RefineReport]
    # a separate Result (extra["refined_from"]); refused -> the source result is returned unchanged
Instance.refine(result, measurement, **kw)          # REFINE_DEFAULTS[instance.name] + instance metrics
refine_run_output(run, instance=None, **kw) -> RunOutput;  refine_method(**kw) -> bench.Method
```
The refinement never modifies the source result or the problem (operator nuisance parameters are
frozen, loss weights live in a private `LossSet`); acceptance requires the data misfit to stay
within `fit_tol` of the reference (the smooth result or, for early-stopped runs, a continuation
control with the same budget) and the interfaces to stay within the smooth field's transition band.

### 3.10 Instances
Each instance subpackage exposes:
```python
def build_problem(cfg: InstanceConfig, measurement: Measurement | None = None) -> InverseProblem
def default_curriculum(cfg) -> Curriculum
class SceneGenerator(bench.SceneGenerator)      # samples ground-truth fields (with difficulty classes)
class DataGenerator(bench.DataGenerator)        # gt fields -> Measurement using an INDEPENDENT operator + noise
METRICS: dict[str, Callable]                     # instance-appropriate metrics
def run(cfg=None, seed=0) -> tuple[Result, dict]  # end-to-end demo: generate -> invert -> evaluate
```
and registers itself: `@register("instance", "nv_relaxometry")`.

### 3.11 Registry & config
```python
register(kind: str, name: str) -> decorator      # kinds: field, encoding, head, operator, loss, postprocess, instance, baseline, metric
build(kind: str, cfg: dict | str) -> object      # cfg = {"type": name, **kwargs}
load_config(path) -> dict; save_config(cfg, path); config_hash(cfg) -> str
```

---

## 4. Method catalogue (what the library must offer out of the box)

**Representations (Fields):** NeuralField (tanh/relu/SIREN), GridField, GaussianSplatField (C), DeepDecoderField (C).
**Encodings:** annealed Fourier features; identity. (Hash-grid: nice-to-have, not v0.1.)
**Heads:** Identity, Softplus, Exp, Bounded, GatedSoftplus, SupportMasked.
**Operators:** FFTConvolution (N-D), NV dipolar F1/F2/F3 (A), Heat implicit-Euler+adjoint (B), Radon (C), Blur (C), Poisson (C), Nuisance gain/offset.
**Data losses:** L2/MSE, LogMSE (max/mean-normalized), NormalizedMSE, Huber, PoissonNLL, masked variants.
**Regularizers:** L1, TV iso/aniso, Laplacian, Tikhonov, NeTMY R_nm / R_ds (instance-specific).
**Curriculum:** multiscale stages, β annealing (reset per stage), cosine/step/warmup LR, per-stage weights, freezing.
**Solver robustness:** grad clip, NaN guard (skip + lr backoff), EMA, early stop, discrepancy principle, restarts, time budget, checkpoint/resume, deterministic seeding.
**Post:** energy-anchored scale correction (homogeneity-matched), threshold masks, projections (B).
**Baselines:** Grid+Adam (Tikhonov), ADMM (L1 prox + box), L-BFGS, Gaussian splats, DeepDecoder, soft-PINN (heat only).
**Metrics:** MSE, PSNR, SSIM, IoU, Dice, GMSD, Hungarian-F1, SWD, Edge-F1, AbsRel/RMSE/δ-thresholds, radial spectrum.
**Diagnostics:** sensitivity (Jacobian column norms, Hutchinson), iter-0 gradient + center/outer ratio, center-mass ratio, energy barrier along interpolation, filter kernel row G_θ e_i, realized vs raw update, top-k singular values (Lanczos via JVP/VJP), Hessian condition number on low-dim ansatz, data-fit paradox report.
**Bench:** cross-fidelity vs matched-operator protocols, seeds + 95% CI, cumulative ablation, 1-D hyperparameter sweeps, markdown/CSV/JSON reports, runtime/memory columns.
**Uncertainty:** multi-seed ensemble mean/std; sensitivity map as trust mask.

---

## 5. Conventions

- Tensors: fields are `(*shape)` per name (no channel dim). Many independent problems are solved
  together by `nefi.batch_invert` (stacked parameters; operators with `batchable=True` accept a
  leading batch axis `(B, *shape)`); a single `Field`/`Operator` call is otherwise unbatched.
- Dtype: default float32; direct simulators used for data generation may use float64 (`DataGenerator(dtype=torch.float64)`).
- Device: `utils.device.resolve("auto")` → cuda > mps > cpu. Everything must work on cpu.
- Randomness: `utils.seed.seed_everything(seed)`; scene generators take an explicit `numpy.random.Generator`.
- Logging: python `logging` under logger name `nefi`; `rich` only if installed.
- Style: ruff (line length 100), type hints everywhere, Google-style docstrings, no `print` in library code.
- Tests: `pytest -q` must pass on CPU in < 2 minutes; mark slow ones `@pytest.mark.slow`.
- Every public class: docstring with the paper reference (e.g. "NeTMY Eq. (5)") when applicable.
- Errors: raise `nefi.errors.NefiError` subclasses with actionable messages (shape mismatch → say which shapes).

---

## 6. Paper → code mapping (to be kept current in `docs/paper_mapping.md`)

| Paper item | Code |
|---|---|
| NeTMY Eq. (1)-(2), App. A: G_ia, F1, F2 | `instances/nv_relaxometry/operator.py` (`DipolarKernels`, `NVOperator(mode="F1"|"F2")`) |
| NeTMY Eq. (11) F3 direct simulator | `instances/nv_relaxometry/operator.py::NVDirectSimulator` (float64, used only by DataGenerator) |
| NeTMY Eq. (19) log-MSE D | `losses/data.py::LogMSE(normalize="max")` composed with `instances/nv_relaxometry/losses.py::NoiseMap` |
| NeTMY R_nm, R_ds (App. D.4) | `instances/nv_relaxometry/losses.py` |
| NeTMY Eq. (5) gated softplus | `fields/heads.py::GatedSoftplus` |
| NeTMY Eq. (26) masked Larmor | `fields/heads.py::SupportMasked` |
| NeTMY Eq. (27) annealed PE | `fields/encoding.py::FourierFeatures` |
| NeTMY Tab. 6 two-stage schedule | `solve/curriculum.py::Curriculum.multiscale` |
| NeTMY Eq. (30) scale correction | `solve/postprocess.py::EnergyScaleCorrection` |
| NeTMY Lemma 2 / Eq. (7) filtering view | `diagnostics/filtering.py::filter_kernel_row, realized_update` |
| NeTMY (P2) iter-0 center bias | `diagnostics/landscape.py::iter0_gradient, center_mass_ratio` |
| NeTMY Fig. 4b energy barrier | `diagnostics/landscape.py::energy_barrier` |
| NeTMY metrics GMSD/HF1/SWD | `metrics/localization.py` |
| NeFTY Prop. 1 harmonic mean | `operators/pde/heat.py::face_coefficients(mode="harmonic")` |
| NeFTY Eq. (8) implicit Euler + Jacobi | `operators/pde/heat.py::HeatOperator`, `operators/pde/linear_solvers.py` |
| NeFTY Eq. (10)-(11) discrete adjoint | `operators/pde/adjoint.py::ImplicitEulerAdjoint` (custom `torch.autograd.Function`) |
| NeFTY Eq. (6) bounded output | `fields/heads.py::Bounded` |
| NeFTY Eq. (22) isotropic TV | `losses/reg.py::TV(isotropic=True)` |
| NeFTY Prop. 2 singular-value decay | `diagnostics/spectrum.py::singular_values` |
| NeFTY App. H projections | `instances/thermal_tomography/projection.py` |
| NeFTY App. I Gaussian variance test | `tests/test_heat_solver.py::test_gaussian_variance_growth` |
| Both: inverse-crime guard | `bench/protocol.py` (asserts independent generator) |

---

## 7. Deployment model (important)

- **Tests and examples are CPU-sized.** They default to tiny grids (≤ 32² / 16³, ≤ a few hundred
  steps), never download datasets, and never write large artifacts into the repository. Outputs go
  to `./runs/` (git-ignored) or a path given by `--out`.
- **Production runs target CUDA servers.** The library is CUDA-first: `device="auto"` picks
  cuda > cpu (Apple MPS only when requested explicitly); kernel caches, adjoint solvers and FFT
  paths are device-agnostic; float64 is used only where the physics needs it. `torch.compile` is an
  opt-in flag (`Solver(compile=...)`), AMP is off by default.
- `scripts/` ships `run_server.sh` (environment setup + `nefi run configs/<instance>.yaml --device
  cuda --out runs/...`), `slurm_template.sbatch`, `sync_to_server.sh`, `environment.yml` and
  `requirements.txt`, so a fresh clone on a server is a two-command setup. Paper-scale configs
  (NeTMY 64², 10k steps; NeFTY 64×64×16, 100 frames, 10k steps) live in `configs/*_paper.yaml`;
  `configs/*_smoke.yaml` are the small CPU variants.
