# 2 · Anatomy of a problem

Every nefi problem is the same seven pieces. This tutorial builds a small 2-D deblurring problem
from scratch — no instance, no registry — so each piece is visible. The contracts are specified in
[DESIGN.md §3](../DESIGN.md#3-core-contracts-exact-signatures--every-module-builds-against-these).

```text
Domain ── Field (prior) ──► fields ──► Operator (physics) ──► prediction ──► LossSet ──► loss
             ▲                                                                              │
             └───────────── Solver runs a Curriculum of Stages (∇θ, lr, annealing) ◄────────┘
                                   └─► Postprocess ─► Result ─► metrics / diagnostics
```

## 1. Domain

A `Domain` is a physical box discretized on a uniform, **cell-centered** grid. Coordinates handed to
fields are normalized to `[-1, 1]` per axis (NeTMY App. D.1, NeFTY App. D.1), so a field can be
queried at any resolution of the same physical domain — the basis of multiscale curricula.

```python
import torch
import nefi

torch.manual_seed(0)           # the field below is initialized from torch's global RNG
domain = nefi.Domain((64, 64), extent=((0.0, 1.0), (0.0, 1.0)), axes=("x", "y"))
domain.spacing()               # (0.015625, 0.015625)   physical cell size
domain.coords().shape          # torch.Size([64, 64, 2]) normalized cell centers
domain.at((32, 32)).spacing()  # (0.03125, 0.03125)     same box, coarser grid
```

`Domain.from_spacing((64, 64), 20.0)` builds a domain from a pixel size (NeTMY: 64 px × 20 nm).

## 2. Field — the prior

A `Field` maps coordinates to named physical fields. The default representation is a coordinate
MLP with annealed Fourier features (`NeuralField`); `GridField` is the free-pixel baseline, and
many others exist (`nefi list fields`). *Heads* turn raw network outputs into physically valid
values and are where hard knowledge lives:

```python
field = nefi.NeuralField(
    2, nefi.Heads({"x": nefi.Softplus(init_value=0.1)}),   # x >= 0, starts near 0.1 everywhere
    hidden=64, depth=3, skip_at=2, n_octaves=6,
)
field.n_parameters()           # 11777
```

| head | value | used for |
|---|---|---|
| `Softplus(init_value=…)` | `≥ 0` | densities, intensities |
| `GatedSoftplus()` | `softplus(h)·σ(g)`, 2 raw channels | sparse sources (NeTMY Eq. 5) |
| `Bounded(lo, hi)` | `lo + (hi − lo)σ(h)` | material parameters (NeFTY Eq. 6) |
| `SupportMasked(inner, depends_on=…, tau=0.3)` | inner head × stop-gradient support mask | fields identifiable only where another is non-zero (NeTMY Eq. 26) |
| `Exp()`, `Identity()` | `exp(h)`, `h` | large dynamic range, signed fields |

`FourierFeatures` implements the annealed encoding of both papers (NeTMY Eq. 27, NeFTY Eq. 21):
band `k` is multiplied by `w_k(β) = (1 − cos(π·clip(β − k, 0, 1)))/2` with `β = progress·K`, so at
`progress = 0` only the raw coordinate passes and higher frequencies are unlocked progressively.

## 3. Operator — the physics

An `Operator` is a differentiable map `fields → prediction`. It is a *hard* constraint: the
prediction is exactly the physics applied to the current field, never a soft residual.

```python
from nefi.operators import FFTConvolution, gaussian_kernel_fn

operator = FFTConvolution(gaussian_kernel_fn(0.02), domain, field="x")   # σ in physical units
operator.homogeneity                                  # 1.0: F(c·x) = c·F(x)
operator.at_resolution((32, 32)).domain.shape         # (32, 32): kernel rebuilt for the grid
```

Two methods matter beyond `forward`: `at_resolution(shape)` returns the operator for fields
sampled on another grid (coarse curriculum stages), and `output_shape(shape)` tells the problem how
to resample the measurement for that stage. `homogeneity` enables energy-anchored scale correction.

## 4. Measurement

```python
yy, xx = torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 64), indexing="ij")
truth = ((xx - 0.35) ** 2 + (yy - 0.5) ** 2 < 0.02).float() \
      + 0.5 * ((xx - 0.7) ** 2 + (yy - 0.4) ** 2 < 0.01).float()
torch.manual_seed(0)
clean = operator({"x": truth})
meas = nefi.Measurement(clean + 0.01 * torch.randn_like(clean), noise_std=0.01)
```

`Measurement(data, mask=None, noise_std=None, meta={})`: the mask marks observed entries (losses
average over them); a known `noise_std` enables discrepancy-principle stopping and noise-aware
weighting. (Simulating the data with the inversion operator itself is an *inverse crime*; it is
fine for this illustration, and [tutorial 3](03_new_forward_operator.md) shows how to avoid it.)

## 5. Losses

A `LossSet` is a weighted sum of named terms. Data terms compare `ctx.pred` with `ctx.obs`;
regularizers act on a named field. All reduce by *means*, so weights do not depend on the
resolution.

```python
losses = nefi.LossSet(
    {"data": nefi.MSE(), "tv": nefi.TV("x", isotropic=True), "l1": nefi.L1("x")},
    weights={"data": 1.0, "tv": 1e-4, "l1": 1e-4},
)
```

Data fidelities: `MSE`, `LogMSE(normalize="max")` (NeTMY Eq. 19), `NormalizedMSE("mean")`,
`Huber`, `PoissonNLL`, `RelativeMSE`. Regularizers: `L1`, `TV` (isotropic NeFTY Eq. 22 or
anisotropic NeTMY Eq. 3, periodic axes, physical spacing), `Laplacian`, `Tikhonov`,
`RangePenalty`, `PriorMSE`; physics-knowledge terms live in `nefi.losses.physics`.

## 6. The problem

```python
problem = nefi.InverseProblem(domain, field, operator, losses, meas, name="blobs")
fields, pred = problem.evaluate()            # native resolution, no grad
total, comps = problem.loss()                # with autograd; comps = {"data": …, "tv": …, "l1": …}
```

Optional arguments: `postprocess=[...]` (one-shot post-processing), `curriculum=` (the problem's
default schedule), `downsample_obs=` (custom measurement resampling for coarse stages).

## 7. Curriculum and solver

A `Curriculum` is a list of `Stage`s run on the same parameters. Each stage fixes a resolution, a
step budget, a learning-rate schedule and whether β is re-annealed from 0 (NeTMY Tab. 6 resets β at
every stage; NeFTY anneals over the first 25 % of its single stage):

```python
cur = nefi.Curriculum(
    [
        nefi.Stage("coarse", shape=(32, 32), steps=150, lr=3e-3),
        nefi.Stage("fine", shape=(64, 64), steps=250, lr=1.5e-3, anneal_fraction=0.5),
    ],
    optim=nefi.OptimConfig(optimizer="adamw", weight_decay=1e-4, grad_clip=1.0),
    discrepancy_tau=1.0,   # Morozov: end a stage once RMSE <= 1.0 × noise_std
)
# the same shape of schedule in one line:
nefi.Curriculum.multiscale((64, 64), n_stages=2, steps=(150, 250), lr=3e-3)
```

`Stage` also takes `lr_schedule` (`"cosine"`, `"step"`, `"constant"`, `"warmup_cosine"`),
`lr_min_ratio`, `loss_weights` (per-stage overrides, e.g. NeTMY's stage-2 rebalancing) and
`freeze` (parameter-name prefixes). `Curriculum` adds `early_stop_patience`, `restarts` and
`time_budget_s` ([tutorial 8](08_robustness_and_adaptivity.md)).

```python
from nefi.solve import LoggingCallback

result = nefi.Solver(problem, cur, device="cpu", seed=0,
                     callbacks=[LoggingCallback(every=100)]).run()
print(result.summary())
print(nefi.metrics.psnr(result.fields["x"], truth))
```

```text
Result: fields={'x': (64, 64)}
  total time 15.2s, steps 400
  stage 0 (coarse, (32, 32)): 150 steps, 3.0s | data=0.00265, tv=1.62, l1=0.0897
  stage 1 (fine, (64, 64)): 250 steps, 11.6s | data=0.000682, tv=1.67, l1=0.0848
22.568004608154297
```

(CPU; times vary with the hardware, the numbers are reproducible with the seed.)

At each stage start the solver rebuilds the coordinates, calls `operator.at_resolution`, resamples
the measurement (`problem.measurement_at`), lets the field adapt (`field.on_stage_start`, e.g. a
grid resamples itself), builds a fresh optimizer and resets β. Every step it evaluates
`field → operator → losses`, clips gradients, steps, and logs. `Solver` options:
`device="auto" | "cpu" | "cuda"`, `dtype`, `seed`, `callbacks` (`LoggingCallback`, `ProgressBar`,
`CheckpointCallback`, `FieldSnapshots`, your own `Callback`), `compile=True`
(`torch.compile`), `nan_guard`.

## 8. Post-processing and the result

`Postprocess` objects run once after optimization. `EnergyScaleCorrection()` resolves the scale
ambiguity of normalized fidelities by `x ← α x`, `α = (E_obs/E_pred)^(1/p)` with `p` the operator's
homogeneity (NeTMY Eq. 30, Prop. 1); `Clip` and `ThresholdMask` are also available.

`Result` fields: `fields` (post-processed, CPU), `raw_fields`, `pred`, `history`,
`stage_results`, `timing`, `post_info` (e.g. the scale factor), `config_hash`, `extra`
(device, seed, parameter count). `Result.final("data_loss")` returns the last value of a history
key.

## Swapping pieces

Because the pieces are independent, experiments are one-line changes:

```python
grid = nefi.GridField((64, 64), nefi.Heads({"x": nefi.Softplus(init_value=0.1)}))
problem_grid = nefi.InverseProblem(domain, grid, operator, losses, meas)       # free pixels
problem_gain = nefi.InverseProblem(domain, field, nefi.Nuisance(operator, gain=True),
                                   losses, meas)                                # unknown gain
```

Instances ([tutorial 4](04_new_instance.md)) package exactly these pieces behind a config
dataclass so that the CLI, the benchmark protocol and the diagnostics can drive them.
