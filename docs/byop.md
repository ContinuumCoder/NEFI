# Bring your own problem

nefi's promise: if you can write your measurement process as differentiable PyTorch code, you can
invert it — in about twenty lines, with defaults that adapt to your data — and then open the hood
when you need to.

## The 20-line story

```python
import torch
import nefi

# 1. Your physics, in plain PyTorch: here a blurred, saturating camera.
k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
k = (k[:, None] * k[None]) / 256

def forward(x):                                   # x: the unknown image, shape (64, 64)
    blurred = torch.nn.functional.conv2d(x[None, None], k[None, None], padding=2)[0, 0]
    return torch.tanh(1.5 * blurred)

# 2. Your data (a tensor, a numpy array, or a nefi.Measurement with a mask / known noise).
x_true = torch.zeros(64, 64); x_true[20:44, 16:40] = 1.0
y = forward(x_true) + 0.01 * torch.randn(64, 64)

# 3. What you know about the unknown.
problem = nefi.from_forward(forward, y, shape=(64, 64), prior="nonnegative + piecewise_constant")

# 4. Solve, then check that the fit reached the noise floor.
result = nefi.invert(problem)
print(nefi.quick_report(result, problem, gt=x_true))
x_hat = result.fields["x"]
```

That is the whole contract: **forward model + data + grid shape + prior**. Everything else —
representation size, initialization, loss weights, learning rate, curriculum, stopping — is chosen
automatically, logged, stored in `problem.meta["auto"]`, and overridable by a keyword argument.
Three complete mini-cases (saturating camera, 30 % k-space MRI with a hash grid, 1-D
photoacoustics through a differentiable wave solver) are in
[`examples/bring_your_own_problem.py`](https://github.com/ContinuumCoder/NEFI/blob/main/examples/bring_your_own_problem.py).

### What the forward may be

| you pass | nefi does |
|---|---|
| `fn(x) -> y` | wraps it in `FunctionOperator` (autograd is the adjoint) |
| `fn(fields) -> y` with `takes_dict=True` or `prior={"a": ..., "b": ...}` | several unknowns, one network with one head per unknown |
| an `nn.Module` | its buffers follow the device; its parameters are frozen unless `FunctionOperator(fn, trainable=True)` |
| a nefi `Operator` (`FourierSampling`, `TimeStepper`, `FFTConvolution`, …) | used as is (it already knows its multiscale behaviour) |

The forward is dry-run on the initial field before anything else: a wrong output shape, a
non-differentiable step (`.detach()`, `.numpy()`, `argmax`), NaNs, or tensors left on the wrong
device raise an error that says what to change. Complex outputs and complex data are converted to
stacked real/imaginary parts automatically.

## The prior DSL

`prior=` takes a `+`-separated string, `Prior` objects, or a list of both
(`nefi.priors.NonNegative() + "tv"` works too). Exactly one prior defines the *output head*
(the most specific one wins when it implies the others; contradictions raise a `ConfigError` that
explains the fix); the others add penalties, head modifiers, field wrappers or hints.

| DSL name (aliases) | implements | key arguments (default) | notes |
|---|---|---|---|
| `nonnegative` (`nonneg`) | head `s·softplus(h)`, `s` fitted to the data | `init="auto"`, `peak_ratio=3` | the default prior |
| `positive` (`log`) | head `s·exp(h)` (log-parameterized) | `init="auto"` | large dynamic ranges |
| `bounded(lo, hi)` (`box`, `range`) | head `lo + (hi−lo)·σ(h)` (NeFTY Eq. 6) | `init=None` (midpoint) | physical brackets |
| `sparse` | gated softplus head (NeTMY Eq. 5) + L1 | `l1=1e-2`, `gated=True`, `peak_ratio=30` | point sources; `gated=False` = L1 only |
| `binary(lo, hi)` (`two_phase`, `level_set`) | sharpening level-set head `lo+(hi−lo)·σ(φ/ε(t))` | `eps_start=1`, `eps_end=0.05`, `perimeter=0` | inclusions / defects; hint `anneal_fraction=0.8` |
| `unconstrained` (`real`, `signed`) | affine head `offset + scale·h` | `init="auto"`, `scale`, `offset` | signed fields; default when no head prior |
| `piecewise_constant` (`tv`) | isotropic TV (NeFTY Eq. 22) | `tv=1e-2`, `eps=1e-6`, `isotropic=True` | edges |
| `smooth` | Laplacian / gradient / ridge penalty, 2 fewer octaves | `laplacian=1e-2` (or `gradient=`, `tikhonov=`) | smooth media |
| `monotone(axis, direction)` | penalty on monotonicity violations | `weight=1` | profiles, depth trends |
| `known_support` (`support`) | multiplicative mask head (stop-gradient) | `mask` (tensor, not in strings), `hard=True` | `hard=False` → penalty |
| `conserved(total)` (`mass`) | mass-normalized head (exact) or penalty | `kind="sum"` (integral) / `"mean"`, `hard="auto"` | hard needs a non-negative head |
| `symmetric(kind)` (`symmetry`) | exact coordinate folding (`SymmetricField`) or penalty | `mirror_x/y/z/xy`, `mirror`+`axes`, `radial`; `hard=True` | radial ⇒ 1-D inner network |
| `periodic` | TV/Laplacian wrap around; periodic encoding if all axes | `axes=None` (all) | |
| `scale` | energy-anchored scale correction after fitting (NeTMY Eq. 30) | `homogeneity=None` | for scale-free fidelities |

Numbers in penalties are **relative strengths** (see "weights" below), not raw weights. Arguments
are Python literals; bare identifiers are strings (`symmetric(radial)`), `inf` is allowed, names
are case-insensitive and `-`/`_` interchangeable. Custom priors: subclass `nefi.priors.Prior`
(a dataclass) and `nefi.priors.register_prior("name", Cls)`.

Resolution rules, for instance:

* `nonnegative + sparse` → gated softplus (sparse implies non-negative);
* `nonnegative + bounded(0, 1)` → sigmoid; `nonnegative + bounded(-1, 1)` → error;
* `positive + sparse` → error ("use `nonnegative + sparse` or `positive + sparse(gated=False)`");
* `bounded(0, 1) + conserved(2)` → bounded head + conservation *penalty* (a bounded field cannot be
  rescaled).

## What "auto" decides — and how to override it

| decision | default | why | override |
|---|---|---|---|
| domain | isotropic spacing, longest axis = 1 | resolution-free regularizer weights | `extent=[(lo, hi), ...]` |
| representation | `"neural"`: tanh MLP with annealed Fourier features, width/depth by grid size (≤1k px: 64×3, ≤16k: 128×4, ≤262k: 256×5, else 256×6), `n_octaves = clamp(ceil(log2 max(shape)), 4, 12)` | the papers' representation; spectral bias regularizes | `representation="hash" \| "grid" \| "lowrank" \| "parametric" \| Field`, `hidden=`, `depth=`, `n_octaves=`, `activation=` |
| head scale / init | constant field that best explains the data (closed form when `homogeneity` is known, else a log-grid search); head scale = `peak_ratio ×` that level | raw outputs stay O(1) whatever the units — LR, clipping and softplus curvature behave the same for 1e-3 or 1e4 | `prior="nonnegative(init=0.5)"`, `peak_ratio=`, `init=None` |
| data term | `noise="auto"` → MSE | Gaussian noise | `noise="gaussian" \| "poisson" \| "robust"` (Huber, δ = 1.345σ̂) or a float σ |
| noise level | robust MAD estimate when unknown: Immerkær Laplacian (≥ 2-D), 4th-order differences (1-D, time traces), operator hooks (k-space: outer annulus) | enables the discrepancy principle and the χ = RMSE/σ verdict | `noise=σ` or `Measurement(noise_std=σ)` |
| loss weights | data = 1 at the initial iterate; each penalty worth *strength ×* that at a reference iterate (max of the initial value and a 25-step data-only probe, floored at 1 % of its value on a synthetic structured field) | the initial field is nearly constant (TV ≈ 0 there), so balancing at the init alone is meaningless | `weights={"tv": 1e-3, ...}` (absolute), `weights="raw"`, `probe_steps=` |
| curriculum | `budget="default"`: 1500 steps for a 64² field, × (numel/4096)^¼ within ×[0.5, 2] (`quick` 300, `thorough` 5000 + early stopping); 2 stages (½-res 30 %, native 70 % at ½ LR) when the operator supports coarse grids; cosine LR to 1 %; annealing over the first half of each stage; AdamW, grad-clip 1 | NeTMY Tab. 6 structure at BYOP budgets | `budget=`, `multiscale=False \| True \| "upsample"`, `anneal_fraction=`, or edit `problem.curriculum` |
| learning rate | neural 1e-2 at width 64 (× √(64/width)), hash 1e-2, grid 1e-1, low-rank 1e-2 | calibrated on the BYOP suite below | `lr=3e-3` or `lr="auto"` (`nefi.auto.lr_range_test`) |
| multiscale | only when the operator can evaluate coarse fields: nefi operators that declare `output_shape`, or a function with `at_resolution=` / `output_shape=`; otherwise a single stage (logged) | a plain function is only valid on its native grid | `at_resolution=lambda shape: fwd_for(shape)`, `output_shape=`, `multiscale="upsample"` (coarse fields upsampled before your function) |
| stopping | Morozov discrepancy `RMSE ≤ τσ`, τ = 1, checked once annealing has finished, whenever σ is known or estimated (Gaussian noise) | stops at the noise floor instead of fitting noise | `discrepancy=False \| True \| τ` |

Opening the hood: `problem` is an ordinary `InverseProblem`. Swap `problem.field`, change
`problem.losses.weights`, replace `problem.curriculum` (or pass one to `nefi.invert`), add
`postprocess` steps, run `nefi.solve.ensemble` for uncertainty — nothing in the auto layer is
special.

## Reading `quick_report`

`quick_report(result, problem, gt=None)` prints a markdown table: parameter count, data RMSE and
**χ = RMSE / σ**, per-stage steps and stop reasons, the scale factor, timing, ground-truth metrics
if given, and every automatic decision.

| verdict | meaning | what to try |
|---|---|---|
| at the noise floor ✓ (χ ≈ 1; ±20 % band widened when σ is estimated) | the data are explained down to the noise | done — judge the priors on the reconstruction |
| slightly above (1.2 < χ ≤ 2) | not fully converged or over-regularized | larger `budget`, weaker penalty strengths, `lr="auto"` |
| underfit (χ > 2) | the model cannot explain the data | check units / forward model / `output_shape`; relax head bounds; `representation="hash"` for sharp detail |
| below the noise level (χ < 0.9) | fitting noise | stronger priors, smaller budget, keep discrepancy stopping on |

## Calibration evidence

Defaults were set on a small BYOP suite (single seed, 300 steps, CPU): 1-D deblurring of smooth
bumps (`smooth`) and boxes (`tv`), 2-D deblurring of smooth blobs, shapes and sparse spikes, 30 %
k-space MRI (`FourierSampling`, 32²) and 1-D photoacoustic wave inversion (`TimeStepper`).
PSNR in dB:

| problem | neural (first defaults: lr 3e-3, no headroom) | neural (default) | hash | grid | low-rank |
|---|---|---|---|---|---|
| 1-D bumps + smooth | 35.6 | 39.0 | 43.6 | 38.6 | 36.6 |
| 1-D boxes + tv | 24.9 | 26.9 | 27.9 | 28.5 | 27.4 |
| 2-D smooth + smooth | 42.8 | 43.4 | 43.5 | 42.3 | 34.8 |
| 2-D shapes + tv | 26.6 | 27.9 | 26.6 | 24.5 | 21.9 |
| 2-D spikes + sparse | 32.5 | 33.0 | 33.4 | 31.5 | 30.6 |
| MRI 30 % + tv | 16.3 | 21.8 | 23.8 | 21.4 | 16.9 |
| 1-D wave + smooth | 23.3 | 43.2 | 44.5 | 43.4 | 42.5 |

Other measured facts behind the defaults: on 1-D signals (blurred bumps / spikes, wave traces,
n = 32…256) the 4th-order-difference noise estimator has a mean error of 11 % (90th percentile
22 %) versus 18 % / 48 % for second differences, while first differences overestimate σ by
×1.4–3.6 (toy1d at n = 128: +1 %, +8–12 % and ×2.6–3.1); in 2-D the Immerkær estimator is
within 2–3 % (also with 30 % of the pixels masked). Discrepancy stopping with the *estimated* σ
matches the PSNR of a known σ while saving 17–33 % of the steps. `smooth(laplacian=1e-2)` beats
1e-3 by 3–4 dB on smooth scenes, and the default TV strength 1e-2 is the best of
{1e-3, 1e-2, 1e-1} on piecewise-constant 1-D scenes (at 30 % k-space sampling a 3× stronger TV
helps the hash grid, see the MRI case of the example).

## Recipes

* **Known noise level** — `from_forward(fwd, y, shape, noise=0.02)` (enables τ = 1 stopping with
  the true σ).
* **Missing data / sensors** — `from_forward(fwd, y, shape, mask=observed)`, or
  `Sampling(mask=...)` + `Measurement(mask=...)`.
* **A PDE you can time-step** — write `step(state, params, dt, n)` and use `TimeStepper`
  (`grad_mode="checkpoint"` for long horizons); do not fall back to a soft `PDEResidual` unless
  there is no solver (NeFTY §3.3 decoupling pathology). An implicit heat solver with an exact
  adjoint lives in `nefi.operators.pde`.
* **Several unknowns** — `prior={"rho": "nonnegative + sparse", "c": "bounded(1400, 1600)"}` and a
  forward taking the dict.
* **Few-parameter ansatz first** — `representation="parametric", fn=nefi.fields.gaussian_blobs(2)`
  (well conditioned, great for sanity checks and Hessian diagnostics, NeTMY §5.5).
* **Sharp or sparse detail, or a tight budget** — `representation="hash"`.
* **Huge 3-D, near-separable media** — `representation="lowrank", rank=16`.
* **Reproducibility** — `seed=` controls initialization and probes; `problem.meta["auto"]` records
  every decision (put it in your lab notebook).
