# Elliptic physics family: variable-coefficient Poisson + planar magnetostatics

`nefi.physics.elliptic` provides **one** reusable, differentiable steady-state solver for
`−∇·(σ∇u) + κu = f` with an exact implicit-function-theorem (IFT) adjoint. It powers

| application | σ | κ | u | f | instance |
|---|---|---|---|---|---|
| electrical impedance tomography | conductivity | — | potential | boundary current | [`eit`](instances/eit.md) |
| Darcy flow | permeability / viscosity | — | pressure | well rates | [`darcy_flow`](instances/darcy_flow.md) |
| steady-state heat conduction | thermal conductivity | convective sink | temperature | heat source | recipe below |
| diffuse optical tomography | diffusion D = 1/(3(μa+μs′)) | absorption μa | fluence | optode | recipe below |

`nefi.physics.magnetostatics` provides the FFT operators of planar current / magnetization
sources imaged at a standoff (NV, SQUID, Hall microscopy), used by the
[`current_density`](instances/current_density.md) instance.

The transient heat solver of NeFTY lives separately in `nefi/operators/pde/`; the steady solver here
is independent of it.

---

## 1. Discretization

Cell-centered finite volumes on a uniform Cartesian grid (1-, 2- or 3-D). The unknown occupies the
trailing `ndim = len(spacing)` axes; leading axes are independent batch entries (several drives /
right-hand sides share one σ). For every axis `d` with spacing `h_d` and face `i+½`:

```
K_{i+½} = σ̄_{i+½} / h_d²,        σ̄ = 2 σ_i σ_{i+1} / (σ_i + σ_{i+1})   (harmonic, default)
                                  σ̄ = (σ_i + σ_{i+1}) / 2               (arithmetic, ablation)
[A u]_i = Σ_d [ K_{i+½}(u_i − u_{i+1}) + K_{i−½}(u_i − u_{i−1}) ] + κ_i u_i   ≈  −∇·(σ∇u) + κu
```

**Why the harmonic mean** (NeFTY Prop. 1, App. A.3–A.4): with piecewise-constant σ in each cell,
flux continuity at the face between two half-cells in series gives exactly `σ̄ = 2σ_iσ_{i+1}/(σ_i +
σ_{i+1})`. It is exact for layered media, dominated by the smaller value (an insulating cell
throttles the flux, like a physical resistance), and second-order for smooth σ; the arithmetic
mean over-estimates the flux across high-contrast interfaces. The tests verify: identical results
for uniform σ; exact series resistance of alternating 1 : 0.05 layers with the harmonic mean
(arithmetic under-estimates it by > 20 %); a thin 1 : 100 insulating inclusion passes less than half
the arithmetic-mean flux.

**Boundary conditions** are given per axis *and per side*: `"dirichlet"`, `"neumann"`, `"periodic"`,
e.g. `bc=[("dirichlet", "neumann"), "periodic"]`.

* `dirichlet` — homogeneous `u = 0` on the boundary face through the ghost value `u_g = −u_0`
  (and `σ_g = σ_0`): the boundary face conductance is `2σ_0/h²` (half-cell distance). The global
  error is second order (measured orders 1.97–2.00, below).
* `neumann` — zero flux: the boundary face conductance is 0. **Non-homogeneous Neumann data**
  (injected current / heat flux `j` through a face) enter the right-hand side as the volume source
  `j/h_n` of the boundary cell — the finite-volume flux balance; see `eit_boundary`.
* `periodic` — circular wrap (`torch.roll`), must be set on both sides.
* Point sources/sinks (wells, optodes): `point_source_rhs(domain, positions, rates)` spreads each
  source over the 2^d nearest cells with cloud-in-cell weights, divided by the cell volume, so the
  discrete source integrates to the rate at every resolution and moves smoothly.

`A` is **symmetric positive semi-definite** under all these conditions (each face conductance appears
symmetrically in the two rows it couples; NeFTY App. D.2 argument) and positive definite as soon as
one Dirichlet face exists or `κ > 0`.

**Pure Neumann / periodic problems without κ** have the constant vector as null space. The solver
(`nullspace="auto"`) then enforces the compatibility condition by removing the mean of the
right-hand side (all cells have the same volume, so the Euclidean projection is the physical one)
and pins the solution to zero mean (the preconditioned residuals are projected every iteration).
Observables should not depend on this gauge: EIT references the potentials to their mean over the
observed boundary, Darcy to the mean over the pressure gauges.

Everything is plain tensor algebra (`narrow`, `roll`, `F.pad`, no Python loops over cells), hence
autodiff-compatible and device-agnostic. `apply_divgrad(u, σ, spacing, bc, face_mode, kappa)` is the
public matrix-free operator; `divgrad_diagonal` its diagonal; `face_conductances` /
`apply_conductances` split it for reuse inside the CG loop.

## 2. Linear solver

`conjugate_gradient` is a batched, matrix-free Jacobi-preconditioned CG (Hestenes & Stiefel 1952;
Saad 2003 §9.2): every leading batch index is an independent system with its own step sizes,
inner products reduce over the grid axes only. `solve_elliptic(σ, rhs, spacing, bc, *, kappa, x0,
tol, atol, max_iter, precond, face_mode, nullspace, grad_mode, check_every, adjoint_cache,
return_info)`:

* `tol` is relative to `‖b‖₂`; it is clamped below by `8·eps(dtype)` (≈ 1e-6 in float32), so a
  float64-style tolerance does not waste iterations in float32.
* `x0` warm start: per batch element it is only used if its residual beats the zero guess.
* `max_iter=None` → `50·max(grid) + 200` (a safety cap). Measured cold-start counts for an
  `n × n` grid, white-noise right-hand side, `tol = 1e-6`: 2.5·n (uniform σ, Dirichlet), 3.4·n
  (uniform σ, pure Neumann), 3.5·n / 5.3·n (log-uniform random σ with 1:100 contrast,
  Dirichlet / Neumann) — e.g. 320–665 iterations at 128². Smooth right-hand sides converge faster
  (1.5–1.7·n at `tol = 1e-12` for the manufactured solution); warm starts inside an optimization
  loop cut the forward counts by ~2× (16² EIT smoke: ~30–60 per solve).
* `check_every` sets the convergence-check period (one host sync); raise it on CUDA.
* A non-converged solve logs one warning per grid/dtype through the `nefi` logger.

## 3. Implicit-function-theorem adjoint

Let `A(σ, κ) u = b` and a scalar loss `L(u)`. Differentiating the constraint,

```
A du = db − (∂A/∂σ · dσ) u − (dκ ⊙ u).
```

Introduce the adjoint state `λ` with `Aᵀλ = ∂L/∂u` (`A` is symmetric: same PCG, same
preconditioner, same null-space projection). Then `dL = ⟨∂L/∂u, du⟩ = ⟨λ, A du⟩` gives

```
∂L/∂b = λ,        ∂L/∂σ = −λᵀ (∂A/∂σ) u,        ∂L/∂κ = −λ ⊙ u.
```

The σ-term is evaluated without forming `∂A/∂σ`. Because

```
λᵀ A(σ) u = Σ_faces K_f(σ) (λ_{i+1} − λ_i)(u_{i+1} − u_i) + Σ_Dirichlet K_b(σ) λ_i u_i + Σ κ λ u
```

is bilinear in `(λ, u)`, `∂L/∂σ` is the vector–Jacobian product of the tiny map `σ ↦ {K_f}` (face
means) with the per-face weights `−(Δλ)(Δu)` (summed over drives), computed by autograd on that map
only. For the pure-Neumann gauge the same formulas hold with `λ` the zero-mean solution of the
projected adjoint system (`A1 = 0` for every σ, so `dA·u ⊥ 1` and no gauge correction appears).

**Forward mode (JVP).** The same linearization gives the tangent directly, with one extra solve:

```
u̇ = A⁻¹ ( ḃ − (∂A/∂σ·σ̇) u − κ̇ ⊙ u ),     (∂A/∂σ·σ̇) u = A_{K̇} u,   K̇ = (∂K/∂σ)·σ̇
```

`A u` is linear in the face conductances, so `(∂A/∂σ·σ̇)u` is the stencil applied with the
conductance derivatives `K̇` (explicit face-mean partials, e.g. `∂σ̄/∂σ_i = 2σ_j²/(σ_i + σ_j)²`).
The tangent solve starts cold and runs to its own tolerance, so it is exact whatever warm start
the primal solve used. The σ-adjoint in reverse mode is the transpose of the same map (explicit
formulas, no nested `autograd.grad`).

`ImplicitSolveFunction` (a new-style `torch.autograd.Function` with `setup_context`) implements
this: the forward is one PCG solve without an autograd graph; `backward` is one adjoint PCG solve
plus the explicit conductance adjoint; `jvp` is one tangent PCG solve; `vmap` moves batched inputs
to a leading batch axis of a single batched solve (the CG loop has data-dependent control flow and
cannot be vmapped op by op). **Memory is O(N)** (no unrolled CG graph; the unrolled alternative
stores every iterate). The adjoint and tangent solves re-enter the Function, so every rule composes
with the others: double backward (`create_graph=True`, `gradgradcheck`), `torch.func.jvp`,
`torch.autograd.forward_ad`, `gradcheck(check_forward_ad=True)`, `torch.func.{grad, vjp, vmap,
jacrev, jacfwd, hessian}`.

For verification, `grad_mode="autograd"` differentiates the unrolled CG iterations (reverse mode);
`grad_mode="none"` disables gradients. **Forward-mode dual or `torch.func`-wrapped inputs are only
accepted with `grad_mode="ift"`**; the other modes raise `NotImplementedError` (pointing to
`jvp_mode="double_backward"`), because forward mode through the early-stopped CG iterations
silently truncates tangents — with a warm start at the solution CG stops at iteration 0 and the
tangent is exactly zero. `nefi.diagnostics` (`jvp_mode="auto"`) therefore uses the exact forward
rule for the default `"ift"` operators and falls back to double backward otherwise.
`EllipticOperator` never stores `torch.func`-wrapped tensors in its warm-start caches.

*History:* before this rule existed, `torch.func.jvp` bypassed the implicit Function and
differentiated the warm-started CG loop; on the `eit` smoke instance `singular_values` with
`jvp_mode="auto"` returned 0.316 / 0.291 / 0.277 instead of the exact 0.699 / 0.599 / 0.579
(found while building the 3-D instances; now a regression test).

`EllipticOperator(warm_start=True)` warm-starts both the forward and the adjoint solve from the
previous optimization step (`adjoint_cache`); inside a curriculum this roughly halves the forward
iteration count (less for the adjoint, whose right-hand side changes every step);
`op.last_info` / `op.last_adjoint_info` expose the CG diagnostics (the latter reports the most
recent auxiliary — adjoint or tangent — solve).

## 4. `EllipticOperator`

```python
EllipticOperator(domain, rhs, bc="dirichlet", *, field="sigma", sigma_transform="identity",
                 kappa=None, kappa_field=None, face_mode="harmonic", grad_mode="ift", tol=1e-8,
                 atol=0.0, max_iter=None, precond="jacobi", nullspace="auto", check_every=1,
                 warm_start=False, reference=None)
```

* fields `{field: (*grid)}` (plus `kappa_field` for an unknown absorption, DOT) → `u` of shape
  `(*rhs_batch, *grid)` (one solution per drive);
* `sigma_transform`: `"identity" | "exp" | "softplus" | callable` (e.g. a log-conductivity field);
* `rhs`/`kappa`/`reference` accept tensors at `domain.shape` **or callables of a `Domain`** — the
  callable is re-evaluated by `at_resolution(shape)` (exact re-derivation of sources and boundary
  drives at every curriculum resolution); tensors are resampled;
* `reference`: non-negative weights; the output is referenced to their weighted mean per drive;
* `output_shape`, `required_fields`, `homogeneity = None` (for κ = 0 and the identity transform
  `u(cσ) = u(σ)/c`, i.e. degree −1, which energy-anchored scale correction does not use);
* `fidelity_tag = "elliptic-fv"`; subclasses override `post(u, σ, fields)` to map the state to the
  observable (EIT: boundary traces on the native strip; Darcy: gauge pressures on the native grid).

`LogBounded(lo, hi)` (registered head `"log_bounded"`) is the log-uniform coefficient head
`σ = exp(log lo + (log hi − log lo)·sigmoid(h))` for coefficients spanning decades.

## 5. Validation (tests/test_physics_elliptic.py)

| check | result |
|---|---|
| constant σ ↔ 5-point (2-D Dirichlet ghost) and 7-point (3-D periodic) Laplacian | ≤ 1e-10 |
| `−Δu = 2π² sin πx sin πy`, Dirichlet, n = 8 → 128 | max error ratios 3.91, 3.98, 3.99, 4.00 (order 1.97–2.00) |
| variable σ = 1 + x/2 + y/4, manufactured solution | second order (ratios > 3.5) |
| symmetry ⟨Au, v⟩ = ⟨u, Av⟩, all BCs incl. per-side mixed, κ, 3-D | ≤ 1e-10 relative |
| CG residual | recursive and true residual ≤ tol (1e-12 in float64) |
| pure Neumann with incompatible RHS | `A u = b − mean b` to 1e-9, `mean u = 0` to 1e-12 |
| IFT vs unrolled-CG gradients (σ, b, κ; 5 BC sets × κ on/off, float64) | ≤ 1e-6 required, observed ≤ 5e-12 |
| `gradcheck` (operator w.r.t. σ; solve w.r.t. σ, b, κ) and `gradgradcheck` | pass |
| forward mode (`torch.func.jvp`, `forward_ad` duals), cold **and warm-started at the solution**, 4 BC sets × κ on/off | ≤ 1e-6 vs central differences required, observed ≤ 7e-10 |
| `gradcheck(check_forward_ad=True)` (σ, b, κ; pure Neumann and mixed) | pass |
| `torch.func.grad`/`vjp` vs autograd; `vmap` (batched σ or RHS) vs loop; `jacrev` vs `jacfwd` vs FD; `hessian` vs `autograd.functional.hessian` | ≤ 1e-10 / 2e-15 / 2e-17, 1e-11 / 7e-18 |
| `grad_mode="autograd"`/`"none"` with forward-mode or `torch.func` inputs | `NotImplementedError` (no silent truncation) |
| `eit` smoke `singular_values`, `jvp_mode="auto"` vs `"double_backward"` (auto uses the forward rule) | 4.8e-8 relative (float32; ≤ 1e-4 required) |
| harmonic vs arithmetic | see §1 |

## 6. Planar magnetostatics (`nefi.physics.magnetostatics`)

Source sheet at `z = 0`, sensor plane `z = z0`, `f̂(k) = Σ f e^{−ik·ρ}` (torch FFT convention), axes
`(x, y)` = tensor axes `(−2, −1)`. From the Biot–Savart law and the 2-D transform of
`1/√(ρ² + z0²)` (`2π e^{−kz0}/k`):

```
current sheet K:      B̂_z = (μ0/2) (e^{−kz0}/k) · i (k_x K̂_y − k_y K̂_x)
stream function g:    K = ∇×(g ẑ) = (∂_y g, −∂_x g)   ⇒   B̂_z = (μ0/2) k e^{−kz0} ĝ
magnetized film:      B̂_z = (μ0/2) e^{−kz0} (1 − e^{−kt}) [d_z − i(k_x d_x + k_y d_y)/k] M̂
                      (thin film, d = ẑ:  (μ0 t/2) k e^{−kz0} M̂_z)
in-plane components:  B̂_x = −i (k_x/k) B̂_z,   B̂_y = −i (k_y/k) B̂_z
NV-axis projection:   B̂_u = [u_z − i(u_x k_x + u_y k_y)/k] B̂_z
sensor layer [z0, z0+t]:  × (1 − e^{−kt})/(kt)
```

Roth, Sepulveda & Wikswo (1989) write the current formula as `i(k_y ĵ_x − k_x ĵ_y)/k` with the
opposite transform sign; the `1/k` is required dimensionally. Sign check: a wire along `+y` at
`x = 0` gives `B_z = −μ0 I x / (2π(x² + z0²))` (right-hand rule). `g = I·1_Ω` is a counter-clockwise
loop current (moment `+ẑ`) and the equivalent magnetization `M_z t = g` (Ampère equivalence).

* `CurrentDensityOperator(domain, z0, source="stream"|"current", mu0, pad_factor=2,
  components="z"|"x"|"y"|"xyz", nv_axis=None, nv_layer_thickness=0)` — linear, `homogeneity = 1`,
  zero-padded FFT (`pad_factor ≤ 1` = periodic), k-grids cached per device/dtype, `at_resolution`
  re-derives the spacing, `fidelity_tag = "magnetostatics-fft"`.
* `MagnetizationOperator(domain, z0, thickness, direction=(0,0,1), ...)`.
* `stream_to_current` / `current_divergence` (central differences with `g = 0` outside the view,
  or spectral): matching stencils commute, so `∇·K` vanishes to round-off (tested ≤ 1e-12).
* `upward_continuation(B, dz, spacing)` — `e^{−k dz}`; exact and composable on the periodic grid
  (tested ≤ 1e-12), ≈ 1 % with zero padding; downward continuation must be requested explicitly.
* `fourier_inversion(Bz, z0, spacing, reg, window="hanning"|"butterworth"|"none",
  cutoff_wavelength, source="stream"|"magnetization")` — the classical Tikhonov + low-pass k-space
  inverse (default cutoff `k_c z0 = 3`); PSNR > 20 dB on a smooth pattern at 0.2 % noise (tests: ≈
  30 dB).
* `biot_savart_sheet` (direct midpoint summation, chunked, float64), `biot_savart_segments` (exact
  thin straight segments), `BiotSavartOperator` (fine-grid `g` → direct `B` at the native pixels:
  the inverse-crime-safe data generator, `fidelity_tag = "biot-savart-direct"`).
* `magnetic_constant(length, current, field)` — e.g. `μ0 = 400π μT·μm/mA`.

Validation (tests/test_physics_magnetostatics.py; loop of radius R = 1 on a 64² grid of pixel 0.1,
z0 = 0.4):

| check | result |
|---|---|
| FFT vs direct Biot–Savart summation of the same Gaussian-profile sheet on a 2× finer grid | 0.74 % (z0 = 0.2 and 0.4; all three components < 3 % in the test) |
| FFT vs exact thin-wire loop (1024-gon segments), profile width w = z0/8 | 1.3 % (the difference grows as w/z0 grows: 4 % at w = z0/4 — a physical width effect) |
| on-axis field vs `μ0 I R² / (2(R² + z0²)^{3/2})` | 504.0 vs 502.9 μT (0.2 %) |
| hairpin wire vs exact segments; sign at the wire vs the infinite-wire law | < 3 %; < 5 % |
| stream ↔ current source forms; thin-film magnetization ≡ stream function | 4e-4; 1e-4 |
| in-plane magnetization vs point-dipole summation | 0.15 % |
| `∇·(∇×g ẑ)` (central / spectral) | ≤ 1e-12 relative |
| upward continuation: periodic exact / composition / zero-padded | ≤ 1e-12 / ≤ 1e-12 / ≈ 1 % |
| `fourier_inversion`, smooth pattern, 0.2 % noise | J-PSNR ≈ 30 dB (> 20 dB required) |
| linearity (`homogeneity = 1`) | round-off |

## 7. A new elliptic instance in ~30 lines

Steady-state heat conduction with a fixed-temperature (Dirichlet) plate edge, an adiabatic top
edge, heater sources and interior thermocouples — recover the conductivity:

```python
import torch, nefi
from nefi.domain import Domain
from nefi.fields import Heads, NeuralField
from nefi.losses import MSE, TV, LossSet
from nefi.measurement import Measurement
from nefi.physics.elliptic import EllipticOperator, LogBounded, point_source_rhs

dom = Domain.unit((32, 32))
heaters = [[(0.3, 0.3)], [(0.7, 0.3)], [(0.5, 0.7)]]                  # one heater per experiment
rhs = lambda d: torch.stack([point_source_rhs(d, h, [1.0]) for h in heaters])   # re-derived per stage
bc = [("dirichlet", "dirichlet"), ("dirichlet", "neumann")]           # cold edges, adiabatic top
op = EllipticOperator(dom, rhs, bc, field="k", warm_start=True, tol=1e-6)

k_true = torch.ones(32, 32); k_true[10:20, 12:24] = 0.2               # a poorly conducting defect
with torch.no_grad():
    T = op.at_resolution((64, 64))({"k": k_true.repeat_interleave(2, 0).repeat_interleave(2, 1)})
T = torch.nn.functional.avg_pool2d(T, 2)                              # 2× finer data, cell averages
mask = torch.zeros_like(T); mask[:, ::4, ::4] = 1.0                   # sparse thermocouples
meas = Measurement(T * mask, mask)

field = NeuralField(2, Heads({"k": LogBounded(0.05, 5.0, init_value=1.0)}), hidden=64, depth=3,
                    activation="relu", n_octaves=4)
losses = LossSet({"data": MSE(), "tv": TV("k")}, weights={"tv": 1e-5})
problem = nefi.InverseProblem(dom, field, op, losses, meas)
result = nefi.invert(problem, nefi.Curriculum.single((32, 32), steps=300, lr=5e-3))
```

(The snippet runs as is: PSNR ≈ 17.5 dB, data loss ↓ 160× in 300 steps on a CPU.)

For **DOT** pass `kappa=μa` (known absorption) or `kappa_field="mua"` with a second head (unknown
absorption); a Robin boundary `D ∂u/∂n + αu = 0` on a Neumann side is the extra sink `κ = α/h_n` on
that side's boundary cells (finite-volume flux balance). For **multiscale** stages
make the observable resolution-independent (see the EIT/Darcy `post` overrides: the prediction is
always sampled on the native observation grid) so the measurement never has to be downsampled.

## 8. Performance notes (CUDA servers)

* Per optimization step: one batched forward solve + one batched adjoint solve. With warm starts
  and `tol = 1e-6` the smoke problems need tens of CG iterations per step.
* Jacobi-PCG iteration counts grow ∝ n and mildly with contrast (§2: 2.5–5.3·n at tol 1e-6), so a
  256² solve needs ~650–1400 cold-start iterations. Set `check_every=8–16` on CUDA to avoid a host
  sync per iteration (the full configs use 8).
* For ≥ 256² grids a geometric-multigrid (or Chebyshev) preconditioner would make the iteration
  count resolution-independent (not implemented; the adjoint structure would be unaffected).
* float64 is only used by the data generators (CPU by default); the differentiable path runs in
  float32 (the tolerance clamp handles the precision floor).

## References

Calderón (1980); Cheney, Isaacson & Newell, *SIAM Rev.* 41 (1999); Somersalo, Cheney & Isaacson,
*SIAM J. Appl. Math.* 52 (1992); Bear (1972); Oliver, Reynolds & Liu (2008); Arridge, *Inverse
Problems* 15 (1999); Patankar (1980); LeVeque (2007); Hestenes & Stiefel (1952); Saad (2003); Giles &
Pierce (2000); Plessix, *Geophys. J. Int.* 167 (2006); Roth, Sepulveda & Wikswo, *J. Appl. Phys.* 65
(1989); Meltzer et al., *Phys. Rev. Applied* (2017); Broadway et al., *Phys. Rev. Applied* 14 (2020);
Midha et al., *Phys. Rev. Applied* 22 (2024); Blakely (1995); NeFTY App. A.3–A.4, D.2.
