# Physics zoo — wave, scattering, optics and reaction–diffusion

This page documents the hyperbolic / scattering / coherent-optics / nonlinear-parabolic family of
`nefi.physics`. Every module provides differentiable forward `Operator`s (hard physics
constraints) that work at any curriculum resolution (`at_resolution`), on CPU and CUDA, in float32
(inversion) and float64 (data generation), plus classical baselines. The four instances built on
them are documented in `docs/instances/{wave_fwi,diffraction_tomography,holography,reaction_diffusion}.md`.

| module | model | operator(s) (registry name) | unknown → data | homogeneity | classical baseline |
|---|---|---|---|---|---|
| `physics.timestep` | generic explicit time stepping | `run_timestepping` | — | — | — |
| `physics.wave` | `p_tt = c² Δp + s` (1-3 D) | `WaveOperator` (`"wave"`) | `c(x)` → traces `(n_src, n_rec, n_t)` | `None` | — (FWI grid baseline) |
| | photoacoustics, `p(0) = p0`, `p_t(0) = 0` | `WaveInitialConditionOperator` (`"wave_initial_condition"`) | `p0(x)` → `(n_rec, n_t)` | 1 | `time_reversal` |
| `physics.scattering` | Helmholtz, Born / Rytov (2-D) | `BornOperator` (`"born"`) | `χ = n² − 1` → `(2, n_angles, n_rec)` | 1 | `filtered_backpropagation`, `adjoint_reconstruction` |
| | Lippmann–Schwinger (multiple scattering) | `LippmannSchwingerOperator` (`"lippmann_schwinger"`) | same | `None` | (data generator) |
| `physics.optics` | thin object (projection approx.) | `PhaseObject` (`"phase_object"`) | `φ` (+ `a`) → complex `t` | `None` | — |
| | angular-spectrum propagation, inline holography | `HolographyOperator` (`"holography"`) | `φ` (+ `a`) → `(n_z, H, W)` | `None` | `gerchberg_saxton` |
| `physics.reaction_diffusion` | Gray–Scott | `ReactionDiffusionOperator` (`"gray_scott"`) | `F(x)` (or `k`, `D_u`) → `(n_t·n_species, H, W)` | `None` | — (grid baseline) |

```python
from nefi.domain import Domain
from nefi.physics.wave import WaveOperator

dom = Domain((64, 64), ((0.0, 1.0), (0.0, 1.0)))          # km
op = WaveOperator(dom, sources=[[0.1, 0.5]], receivers=[[0.9, z] for z in (0.2, 0.5, 0.8)],
                  n_t=200, dt_obs=0.005, f0=8.0, c_max=2.5)  # PML, 4th order, checkpointed adjoint
traces = op({"c": c})                                       # (1, 3, 200), differentiable in c
op.at_resolution((32, 32))                                  # same physics on a coarser grid
```

## 1. Time stepping and the memory trade-off (`physics.timestep`)

`run_timestepping(step_fn, state0, params, n_steps, dt, observe, *, checkpoint_every=None,
record=None)` integrates `state_{n+1} = step_fn(state_n, params, n, dt)` and collects
`observe(state_n, n)` at the recorded step indices. Gradients are the **discrete adjoint** of the
scheme, obtained by reverse-mode autodiff through the unrolled loop, so the physics remains a hard
constraint (NeFTY §3.3).

*Memory.* Plain backpropagation through time stores the intermediate tensors of every step:
`O(n_steps · S)` for a state of size `S`. NeFTY (§4.3, App. D.3) avoids this for its implicit heat
solver with a hand-written discrete adjoint — `O(S)` memory, exact, but it has to be re-derived for
every scheme and boundary condition. The wave/optics family instead uses the general-purpose
alternative: **per-block checkpointing** (Griewank & Walther 2000; Chen et al. 2016). The loop is
cut into blocks of `k = checkpoint_every` steps (`torch.utils.checkpoint`, non-reentrant); only the
block-boundary states are stored and each block is recomputed once in the backward pass:

| strategy | activation memory | extra compute | exactness | effort per new PDE |
|---|---|---|---|---|
| plain autograd (`grad_mode="autograd"`) | `n_steps · S` | — | exact | none |
| block checkpointing (`grad_mode="checkpoint"`) | `(n_steps/k + k) · S`, minimal at `k = √n_steps` (`auto_checkpoint_every`) | +1 forward (measured 1.2–2× wall-clock on CPU at small sizes) | **bit-identical** gradients (tested) | none |
| hand-written adjoint (NeFTY) | `O(S)` | ≈ +1 backward solve | exact | derive + test adjoint |

For the production FWI setting of `configs/wave_fwi_full.yaml` (96² + PML ≈ 120² cells, 8 shots,
600 steps, ≈ 3 saved tensors per step, 4 state tensors `p⁻, p, ψx, ψy` per checkpoint) plain
autograd needs ≈ 8·120²·600·3·4 B ≈ 0.8 GB of activations; with `k = 25` it is
≈ (24·4 + 25·3)/(600·3) ≈ 10 % of that, ≈ 0.08 GB. At 512² × 32 shots × 4000 steps (a realistic
seismic job) it is the difference between ≈ 400 GB and ≈ 15 GB (`k = 63`). The smoke configs use `grad_mode="autograd"` (tiny
problems, fastest).

Stability helpers: `stable_dt_wave(c_max, spacing, order, courant)` (leapfrog:
`dt² c² λ_max ≤ 4`, i.e. `c dt ≤ h/√d` for 2nd order and `c dt ≤ h √(3/(4d))` for 4th order) and
`stable_dt_diffusion(D_max, spacing, rate_max, safety)` (explicit Euler:
`dt (D Σ 4/h² + r) ≤ 2`). `laplacian_eig_max` gives the Laplacian's spectral radius.

## 2. Acoustic wave equation (`physics.wave`)

**Model.** `p_tt = c(x)² Δp + s(x, t)` with Ricker point sources `s = w(t) δ(x − x_s)`
(`ricker(t, f0, t0=1.2/f0)`). The unknown `c` covers only the physical domain; the simulation grid
(`WaveGrid`) adds an absorbing layer of physical width `absorb_width` around it, where `c` is
continued by edge replication.

**Discretization.** Leapfrog in time; centered 2nd- or 4th-order Laplacian in space (one
`convNd` per step, zero values beyond the padded grid); sources injected with the transposed
multilinear interpolation weights divided by the cell volume (a grid-independent discrete δ);
receivers sample `p` by multilinear interpolation. **Observation times are physical**:
`t_j = j·dt_obs`, and the simulation step is `dt = dt_obs/m` with the smallest integer `m` meeting
the CFL bound for the declared `c_max` (the `Bounded` head's upper limit; exceeding it raises an
`OperatorError`). Hence the output shape `(n_sources, n_receivers, n_t)` does not depend on the grid
and `at_resolution` only re-derives the padded grid, `dt`, and the source/receiver stencils at the
same physical positions. Sources are batched along a leading dimension (`source_batch` chunks them
for memory).

**Absorbing boundaries** (`absorbing=`):

* `"pml"` (default, 1-D/2-D) — the second-order PML of Grote & Sim (2010):
  `p_tt + (ζx+ζy) p_t + ζxζy p = c²(Δp + ∂xψx + ∂yψy)`,
  `ψx,t = −ζx ψx + (ζy − ζx) ∂x p` (and symmetrically), from the complex stretching
  `∂x → ∂x/(1 + iζx/ω)`. ψ lives on cell faces at half time levels, damping terms are
  time-centered; profile `ζ = ζ_max ξ²`, `ζ_max = 3 c_max ln(1/R) / (2W)` (Collino & Tsogka 2001).
  In the interior ζ = 0 and the scheme is exactly the plain leapfrog.
* `"sponge"` — damping layer `p_tt + σ p_t = …` with `σ_max = 3 c_max ln(1/R)/W` (the amplitude
  decays as `exp(−∫σ/(2c))`). Cheap per step but needs `W ≳ 1–1.5 λ`: the damping itself reflects
  (stronger damping → *more* reflection at low frequency).
* `"cerjan"` — multiplicative taper `exp(−(γξ)²)` on `p, p⁻` every step (Cerjan et al. 1985); its
  strength depends on `dt` (not resolution invariant).
* `"none"` — rigid pressure-release box (energy conserving; for tests).

Measured reflection (2-D, λ = 0.4, 20 cells/λ, reflected amplitude relative to the direct wave,
reference = 9× larger domain):

| layer | width | side | corner |
|---|---|---|---|
| PML | 0.25 λ (5 cells) | ≈ 1 % | ≈ 2 % |
| PML | 0.5 λ (10 cells) | **0.11 %** | **0.29 %** |
| PML | 1 λ (20 cells) | 0.01 % | 0.08 % |
| sponge | 0.5 λ | 14 % | 42 % |
| sponge | 1.5 λ (30 cells) | 1.3 % | 4.8 % |
| Cerjan | 1.5 λ (30 cells) | 1.0 % | 3.2 % |

The PML is long-time stable (20 000 steps, homogeneous and rough media, tested offline). 3-D uses
`"sponge"`/`"cerjan"` (a 3-D second-order PML needs an extra auxiliary field and is not implemented).

**Validation** (`tests/test_physics_wave.py`): a 1-D pulse travels at `c` (0.06 % / 0.02 % error for
order 2 / 4); the discrete energy
`E = ½ΔV Σ[(p − p⁻)²/(c² dt²) − p Δ_h p⁻]` (`wave_energy`) is conserved to 1e-12 in a
heterogeneous closed box and decays below 5 % with PML/sponge; the leapfrog stays bounded at
0.99 × the CFL step and blows up at 1.05 ×; checkpoint gradients equal autograd gradients to machine
precision and match central finite differences.

**Photoacoustics.** `WaveInitialConditionOperator` starts from `p⁻¹ = p¹ = p0 + ½dt²c²Δ_h p0`
(zero initial velocity, second order) and is exactly linear in `p0` (tested to 1e-15).
`time_reversal(traces, domain, receivers, dt_obs=…, c=…, mode=…)`:
`"dirichlet"` (k-Wave style: propagate the reversed traces into the domain as Dirichlet data at the
sensor cells; correlation 0.97 with a point absorber for a 64-sensor ring) or `"adjoint"` (exact
discrete adjoint `Aᵀd` by autodiff; correlation 0.99).

## 3. Diffraction tomography (`physics.scattering`)

**Model** (time dependence `e^{−iωt}`): `Δu + k0²(1 + χ)u = 0`, `χ = n²/n_b² − 1`,
`f = k0² χ`. The total field solves the Lippmann–Schwinger equation `u = u_inc + G ∗ (f u)` with
`G = (i/4)H₀⁽¹⁾(k0 r)`, `(Δ + k0²)G = −δ`. Born: `u_s ≈ G ∗ (f u_inc)`; first-order Rytov:
`φ = log(u/u_inc) ≈ u_s^{Born}/u_inc`. Plane waves `u_inc = exp(ik0 d̂·x)` from `n_angles`
directions; receivers on a ring (default radius `0.75 × extent`) or any explicit positions outside
the object domain. Output: stacked `(Re, Im)` → `(2, n_angles, n_receivers)`.

**Receivers by quadrature.** `BornOperator` evaluates `u_s(x_r) = Σ_j G(x_r − y_j) q_j ΔA` with a
precomputed `(n_rec, n_pix)` Hankel matrix (scipy, double precision): exact up to the midpoint rule,
no singularity (receivers are outside), one complex matmul per forward, linear in χ. The same
convolution evaluated *inside* the domain is `BornOperator.scattered_field(χ)` (FFT, below) — the
receivers could equally be sampled from an FFT field on a grid enlarged to contain them, but the
quadrature is exact for receivers anywhere outside the object and cheaper when `n_rec ≪ N²`.

**In-domain fields by FFT — the Green's kernel.** The textbook k-space form
`Ĝ(k) = 1/(|k|² − k0² − iε)` sampled on a zero-padded FFT grid is inaccurate: a small ε leaves the
periodic images undamped and under-resolves the singular circle `|k| = k0`, a large ε damps the field
(measured error: **47 %** beyond 2λ). nefi therefore uses the truncated Green's function of
Vico, Greengard & Ferrando (2016): `G_L = G·1{|x|<L}` (L ≥ domain diagonal) has the smooth transform
`Ĝ_L(s) = [1 + (iπ/2)L(s J₁(Ls)H₀(k0L) − k0 J₀(Ls)H₁(k0L))]/(s² − k0²)` (removable singularity at
`s = k0`); it is sampled on a 4× oversampled grid, transformed back, restricted to the offsets of a
2× padded box and re-transformed (`helmholtz_kernel_fft(..., method="truncated")`). Agreement with
`(i/4)H₀⁽¹⁾(k0r)h²` for a point source: **0.16 %** (λ/8 grid), 0.11 % (λ/12), 0.08 % (λ/16), max
over `r > 2λ`; 1.6 % at the far corner of the box. `method="regularized"` keeps the naive kernel for comparison.

**Multiple scattering (data generator).** `LippmannSchwingerOperator` solves the LS equation by
relaxed fixed-point iteration (Born series, `lippmann_schwinger`) on the FFT grid — 6–8 iterations
to 1e-10 at χ ≤ 0.05 — and evaluates the receivers from the total field. At χ_max = 0.01 / 0.02 /
0.05 over a 4λ domain the Born model differs from LS by 1.5 / 3.1 / 7.7 % of the data (the
inverse-crime gap, comparable to 1 % noise).

**Filtered backpropagation** (Devaney 1982), full-view ring geometry: circular-harmonic
near-to-far-field transform (`α_n = c_n / H_n⁽¹⁾(k0R)`), Fourier diffraction theorem
`F̂(k0(r̂ − d̂)) = −4i u_∞(φ)`, and back-propagation with the Jacobian filter
`|sin(φ − θ)|` of `(θ, φ) → K`: `f(x) ≈ k0²/(8π²) Σ F̂ e^{iK·x}|sin(φ − θ)| ΔθΔφ`. Output is χ
low-passed to `|K| ≤ 2k0`. On smooth full-angle LS data (16 angles × 64 receivers): PSNR ≈ 40 dB
(32² grid) to 47 dB (48²) (tested > 15 dB).
`adjoint_reconstruction(op, data, shape)` gives `Aᵀd` for any linear operator.

## 4. Coherent optics (`physics.optics`)

`propagate(u, dz, wavelength, spacing, *, band_limit=True, pad_factor=2.0, pad_mode="constant",
evanescent="drop")`: angular spectrum with the exact transfer function
`H_z = exp(i2πz√(1/λ² − f²))`, Matsushima–Shimobaba band limit
`f_lim = 1/(λ√((2z/S)² + 1))`, zero (`"constant"`), replicate (`"edge"`) or no (`"none"`, periodic)
padding; a list of distances returns all planes at once. Dropping evanescent waves makes `±z`
propagation exact inverses on the propagating band (round trip 6e-16 periodic, < 1e-9 padded +
band-limited); a Gaussian beam's second-moment width follows `w0√(1 + (z/z_R)²)`,
`z_R = πw0²/λ` to **≤ 3e-4** relative (tested ≤ 2 %).

`HolographyOperator(domain, distances, wavelength, field="phase", absorption=None|float|"name")`:
thin object `t = exp(iφ − a)` (`PhaseObject` operator / `phase_object` function), intensities `|P_{z_i} t|²` at all distances,
edge padding by default (object embedded in an infinite uniform background), transfer functions
cached per resolution; the pixel pitch always equals the field grid spacing. Invariant to a global
phase offset (1e-15), hence mean-subtracted phase metrics.

`gerchberg_saxton(intensities, op, n_iter, pure_phase=True)`: sequential multi-plane projections
with the operator's own propagation model (padding, band limit), object-plane modulus constraint,
global phase fixed so the mean transmission is real; returns the wrapped phase and the intensity
mismatch history (≈ 6e-2 → 5e-3 in 100 iterations on the test object).

## 5. Reaction–diffusion (`physics.reaction_diffusion`)

Gray–Scott `u_t = D_u Δu − uv² + F(1 − u)`, `v_t = D_v Δv + uv² − (F + k)v` with a spatially
varying unknown `F(x)` (`unknown="F"`), `k(x)` (`"k"`) or diffusivity `D_u(x)` (`"Du"`, conservative
`∇·(D_u∇u)` with harmonic-mean faces, NeFTY Prop. 1). Known analytic initial condition
`GrayScottIC` (perturbs the whole domain away from the trivial state `(1, 0)` where `F` is
unobservable); snapshots of `u`/`v` at physical `obs_times` → `(n_times · n_species, *shape)`.

Discretization: conservative flux-form Laplacian (`diffusion_term`; zero boundary flux → exact mass
conservation, or periodic), explicit Euler. Stability `dt (D_max Σ 4/h² + F_max + k_max + r) ≤ 2`
(`stable_dt`); the step is `dt/m` with the smallest integer `m` satisfying it (times `substeps`),
so observation times are hit exactly at every resolution. Validation: `u, v` stay in `[0, 1]`
(2 000 time units), `Σ(u + v)` is conserved to 1e-15 for `F = k = 0`, the 4× substepped float64
generator differs from the inversion stepper by 0.8 % (a genuine O(dt) gap), checkpoint = autograd
gradients, finite differences agree.

## 6. Inverse-crime guards (fidelity tags)

| instance | inversion operator tag | data-generator tag | what differs |
|---|---|---|---|
| `wave_fwi` | `wave-leapfrog-o4-pml` | `wave-leapfrog-o4-2x-float64` | 2× finer grid (dispersion, PML, stencils), float64 |
| `diffraction_tomography` | `born-1x` / `rytov-1x` | `lippmann-schwinger-iterated-2x-float64` | multiple scattering, FFT vs quadrature, 2× grid |
| `holography` | `angular-spectrum-1x` | `angular-spectrum-2x-float64` | 2× sampling, intensities area-averaged, float64 |
| `reaction_diffusion` | `gray-scott-euler` | `gray-scott-substep4-float64` | 4× smaller Euler step, float64 |

## 7. Known limitations

* 3-D PML (needs the extra auxiliary variable of Grote & Sim §2.3); 3-D uses sponge/Cerjan.
* Variable density / elastic waves, frequency-domain (Helmholtz) FWI, and source-wavelet estimation
  are not implemented; the wave operator assumes known Ricker sources.
* Born quadrature matrices are dense `(n_rec, N²)`: for N ≳ 512 with many receivers switch to the
  FFT field + interpolation or chunk the receivers.
* Filtered backpropagation requires a full-view circular array; limited-view geometries should use
  `adjoint_reconstruction` or the grid baseline.
* The Gray–Scott dynamics becomes chaotic for long windows (≫ 200 time units); keep observation
  windows in the transient regime for well-posed inversion.
