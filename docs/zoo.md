# The zoo: operators, representations, heads and losses

Everything here composes with everything else: any `Field` × any `Operator` × any `Loss`, driven
by the same `Curriculum` / `Solver`. Items marked *(core)* predate the BYOP layer.

## Forward operators (`nefi.operators`)

| operator | measures | homogeneity | multiscale | use when |
|---|---|---|---|---|
| `FunctionOperator(fn)` | anything you can write in torch | user-declared | `at_resolution=` / `output_shape=` / `coarse="upsample"` | bring your own physics (wrapped automatically by `from_forward`) |
| `FFTConvolution(kernel, domain)` *(core)* | translation-invariant linear blur, `kernel(spacing)` rebuilt per grid | 1 | native | deconvolution, NeTMY dipolar kernels |
| `Downsample(factor, mode)` | detector binning / decimation / low-res imaging | 1 | yes | super-resolution |
| `Sampling(mask=…)` / `Sampling(indices=…)` | point samples (full-size masked output, or compact vector) | 1 | coarse fields upsampled | inpainting, sparse sensors, scattered probes |
| `FourierSampling(mask)` | undersampled centered DFT, real/imag stacked, `compact=` | 1 | coarse fields upsampled | MRI, interferometry; `adjoint()` = zero-filled baseline; `estimate_noise()` from outer k-space |
| `PhaseRetrieval(oversample)` | `|F pad(x)|²` | 2 | coarse fields upsampled | coherent diffraction imaging (phase/translation/flip ambiguities!) |
| `BeerLambert(axis=… \| path_op=…)` | `I0·exp(−∫μ)` along an axis (total or depth-resolved) or through a path operator | – | yes | X-ray / optical transmission, attenuation |
| `Saturation("tanh" \| "sigmoid")` | pointwise detector response | – | yes | nonlinear cameras (compose with `Sequential`) |
| `Pointwise(fn)`, `Identity()` | elementwise map / identity | declared / 1 | yes | building blocks, denoising |
| `Sum(*ops)`, `Stack(*ops, mode)` | superposition / multi-modal data | common / – | yes | several contributions or several measurements of one unknown |
| `Sequential(*ops)`, `Nuisance(op)` *(core)* | composition / learnable gain + offset | product / inner | inner | calibration mismatch on real data |
| `TimeStepper(step, init, n_steps, dt, observe)` | any explicit/implicit time integration you write; `grad_mode="checkpoint"` for O(N/k) memory | declared | coarse fields upsampled | wave, advection–diffusion, reaction… (discrete adjoint by autograd) |
| steppers `wave_step`, `advection_diffusion_step`, helpers `stable_dt`, `wave_initial_state` | leapfrog wave (absorbing Mur / Neumann / Dirichlet / periodic); upwind advection + explicit (conservative) diffusion | – | – | photoacoustics, seismics-lite, transport |
| `nefi.operators.pde` | implicit-Euler heat solver with hand-written discrete adjoint | 1 in the initial data | yes | NeFTY thermography; stiff diffusion |

Rules of thumb: put the physics **in the operator** (hard constraint) rather than in a penalty; if
your function is only valid on its native grid, say nothing — `from_forward` will run a single
stage — or pass `multiscale="upsample"`; use `TimeStepper(grad_mode="checkpoint")` as soon as the
unrolled graph does not fit in memory (identical gradients, ~2× compute).

## Representations (`nefi.fields`)

| field | parameters | strengths | weaknesses | pick when |
|---|---|---|---|---|
| `NeuralField` *(core)* | MLP + annealed Fourier features | smooth-to-moderate detail, strong implicit regularization (NeTMY Lemma 2), paper method | slower per step, spectral bias on spikes | default; smooth or piecewise-smooth media |
| `HashGridField` / `HashGridEncoding` | multiresolution hash tables + tiny MLP (Instant-NGP-lite), level annealing | fast convergence, sharp edges, sparse peaks; best on 5 of the 7 BYOP-suite problems (see `byop.md`) | hash collisions at very large grids; fewer smoothness guarantees | sharp/sparse content, tight budgets, 3-D |
| `GridField` *(core)* | one value per voxel | exact, cheapest per step, the classic Tikhonov/TV baseline | no implicit prior; needs penalties | baselines, well-posed problems, very large but easy problems |
| `LowRankField` | CP decomposition, per-axis factors (resampled per stage) | memory `R·Σ N_a` (a 256³ volume at rank 32: 25k parameters) | only near-separable structure | layered / axis-aligned 3-D media, backgrounds |
| `ParametricField(fn)` + `gaussian_blobs`, `ellipses`, custom `ParamFn` | a handful of numbers | well conditioned, interpretable, Hessian diagnostics (NeTMY §5.5) | model mismatch | sanity checks, sources with known shape, initialization |
| `LevelSetField` / `LevelSetHead` | neural level set with sharpening | two-phase objects, crisp interfaces; `interface_length` perimeter | needs `lo`, `hi` | inclusions, defects, segmentation-like unknowns |
| `SymmetricField(inner, kind)` | wraps any coordinate field | exact mirror / radial symmetry, half (or 1-D) the unknowns | needs coordinate-based inner field (grids: `mode="average"`, mirror only) | known symmetry of sample or setup |
| `GaussianSplatField`, `DeepDecoderField` *(baselines package)* | anisotropic Gaussian splats / untrained conv decoder | alternative priors (NeTMY App. E.2 baselines) | splats need an identity head (used by default) | `representation="splat" \| "deep_decoder"` |

## Heads (value constraints)

*(core)* `Identity`, `Softplus`, `Exp`, `Bounded(lo, hi)`, `GatedSoftplus` (NeTMY Eq. 5),
`SupportMasked` (NeTMY Eq. 26). New:

| head | output | used by prior |
|---|---|---|
| `ScaledHead(inner, scale)` | `scale · inner(h)` — natural units | `nonnegative`, `positive`, `sparse` (auto scale) |
| `Affine(scale, offset)` | `offset + scale·h` | `unconstrained` |
| `ExpHead(init_value)` | `exp(h)` with an init value | `positive` |
| `LevelSetHead(lo, hi, eps_start, eps_end)` | `lo + (hi−lo)·σ(φ/ε(progress))`, ε shrinking with progress | `binary` |
| `MaskedHead(inner, mask, fill)` | `inner·m + fill·(1−m)`, mask resampled per grid, no gradient through `m` | `known_support` |
| `MassNormalized(inner, total, volume)` | rescaled so that `∫x = total` exactly | `conserved` |
| `ZeroMean(inner, n_dims)` | `inner(h) − mean(inner(h))`: spatial mean exactly 0 (gauge fixing for a phase / potential the data see only up to a constant; `holography`) | — |

## Losses (`nefi.losses`)

Data terms *(core)*: `MSE`/`L2`, `RMSE`, `RelativeMSE`, `Huber`, `PoissonNLL`, `LogMSE`,
`NormalizedMSE` (all masked means). Regularizers *(core)*: `L1`, `TV` (iso/aniso, periodic,
physical spacing), `Laplacian`, `Tikhonov`, `RangePenalty`, `PriorMSE`.

Physics knowledge (`nefi.losses.physics`):

| loss | penalizes | zero when |
|---|---|---|
| `PDEResidual(residual_fn, fields)` | `mean R(fields)²` — soft PDE constraint; warns when the constrained fields bypass the operator (NeFTY §3.3 decoupling) | the fields satisfy the discrete PDE |
| `Conservation(field, total, kind)` | `((∫x − total)/total)²` (integral with physical cell volume, or mean) | total matches |
| `SymmetryLoss(field, kind)` | `mean (x − Sx)²`, `S` = flip or exact-radius shell average | field symmetric |
| `KnownSupportLoss(field, mask, fill)` | exterior difference to `fill` | support respected |
| `Monotone(field, axis, direction)` | `mean relu(∓∂x)²` with physical spacing | monotone along the axis |
| `GradientL2(field)` | `mean |∇x|²` (first-order Tikhonov) | constant field |
| `RangeStat` | alias of `RangePenalty` (soft box) | inside the box |
| helpers `gradient`, `laplacian` | finite differences for writing residuals | – |

## Choosing — a short decision guide

1. Can you simulate the measurement? Write it as the operator (FunctionOperator / TimeStepper /
   built-in). Only if not, consider `PDEResidual` — and read NeFTY §3.3 first.
2. What values are physical? → head prior (`nonnegative`, `bounded`, `binary`, `positive`).
3. What structure? → penalties (`piecewise_constant`, `smooth`, `sparse`, `monotone`) and exact
   structure (`symmetric`, `known_support`, `conserved`, `periodic`).
4. Representation: start with `"neural"`; switch to `"hash"` for sharp/sparse detail or speed,
   `"parametric"` for a first sanity check, `"lowrank"` for big separable 3-D, `"grid"` as the
   baseline.
5. Check `quick_report`: χ ≈ 1 means the data are explained; the reconstruction quality is then
   up to the priors.
