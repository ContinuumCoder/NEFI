---
description: The transient heat solver of NeFTY — implicit-Euler finite volumes with harmonic-mean faces, a discrete adjoint at trajectory-sized memory, fast inner sweeps and an independent explicit data simulator.
---

# Heat equation (NeFTY)

The heat solver in `nefi/operators/pde/` is the physics of the
[`thermal_tomography`](../instances/thermal_tomography.md) instance (NeFTY, arXiv 2603.11045): it
maps a volumetric thermal diffusivity $\alpha(x, y, z)$ to the front-surface temperature after a
flash, and back-propagates the surface misfit to $\alpha$ through an exact discrete adjoint. This
page is a map; the full derivation, validation numbers and options are on the instance page.

## Model and discretization

$$
\partial_t T = \nabla\cdot(\alpha \nabla T), \qquad
\bar\alpha_{i+\frac12} = \frac{2\,\alpha_i\,\alpha_{i+1}}{\alpha_i + \alpha_{i+1}}, \qquad
\big(I - \Delta t\, L(\alpha)\big)\, T^{n+1} = T^{n}
$$

* **Finite volumes with harmonic-mean faces** (NeFTY Prop. 1): continuity of the flux across a
  face between constant cells gives the harmonic mean, dominated by the insulating side;
  `face_mode="arithmetic"` is the leaky ablation of the paper's Tab. 3.
* **Implicit Euler** (Eq. 8): the system matrix is symmetric, strictly diagonally dominant and
  SPD for periodic, adiabatic and Robin boundaries — all checked on dense matrices in the tests.
* **Cell-averaged initial conditions**: the flash is stored as exact cell averages, so the
  deposited heat is identical at every resolution and coarse curriculum stages stay consistent.

[Physics](../instances/thermal_tomography.md#physics-nefty-31-app-a) ·
[Discretization](../instances/thermal_tomography.md#discretization-nefty-42-app-d2--nefioperatorspdestencilpy)

## Inner solvers

| `solver` | what it does | when to use it |
|---|---|---|
| `"jacobi"` (default) | `K` warm-started sweeps, `K = 50` in the paper configuration (NeFTY Eq. 24) | reproducing the paper |
| `"chebyshev"` (opt-in) | Chebyshev acceleration of the same sweeps, Gershgorin bound, no host sync | the paper's accuracy with 20 instead of 50 sweeps |
| `"cg"` | Jacobi-preconditioned conjugate gradients with a tolerance | verification and tight tolerances |

Every sweep runs on a flat padded layout: one gather refills the periodic ghost cells, then six
in-place multiply-adds on contiguous views — 7 kernels per sweep instead of 12, with the same
numbers to float rounding. `compile_solver=True` (on in `configs/thermal_tomography_paper.yaml`)
fuses each sweep further with `torch.compile`. Measurements and CUDA estimates:
[Performance § Heat solver](../performance.md#heat-solver-nefty).

## Discrete adjoint

`ImplicitEulerAdjoint` (NeFTY §4.3, App. D.3) is a custom autograd function:

* the **forward** pass runs the recurrence without autograd and saves only the trajectory
  $T^1, \dots, T^{N_t}$;
* the **backward** pass solves $A\,\mu^n = \mu^{n+1} + \partial\ell^n/\partial T^n$ backwards in
  time with the same inner solver ($A^\top = A$) and assembles
  $\mathrm dJ/\mathrm d\alpha = \Delta t \sum_n (\mu^n)^\top\, \partial\big(L(\alpha) T^n\big)/\partial\alpha$
  in one fused pass over the faces.

| `grad_mode` | saved tensors | 64×64×16, K = 50, 100 frames |
|---|---|---|
| `"adjoint"` (default) | the trajectory | **26.5 MB** |
| `"checkpoint"` | trajectory + one step's graph | 0.25 GB + 0.16 GB |
| `"autograd"` (unrolled reference) | every sweep | 15.7 GB |

[Discrete adjoint on the instance page](../instances/thermal_tomography.md#discrete-adjoint-nefty-43-app-d3--nefioperatorspdeadjointpy)

## Independent data and the model-error floor

Benchmark data come from `ExplicitHeatSimulator` — forward Euler with enough substeps for
stability, its own ghost cells and flux differences, float64 — which shares no time-stepping or
boundary code with the inversion operator. This independence exposes a **model-error floor**:
implicit Euler at one step per camera frame does not resolve the fast decay of the thin
post-flash layer, so the ground truth re-simulated by the inversion operator fits the data to
about 46–51 dB surface PSNR at the paper grid, and surface-fit numbers are bounded by that floor.
Check the floor before reading IoU: [Reproducing the papers](../reproduce_papers.md#nefty--thermal-tomography)
and [Limitations and caveats](../guides/limitations.md#the-model-error-floor-thermal-tomography).

## Related

* The steady-state elliptic solver (EIT, Darcy, diffuse optics) is independent of this one:
  [Elliptic & magnetostatics](../physics_elliptic.md).
* Generic time stepping with block checkpointing (waves, reaction–diffusion) trades memory for
  recomputation without a hand-written adjoint: [Wave, scattering, optics & reaction](../physics_wave_optics.md#1-time-stepping-and-the-memory-trade-off-physicstimestep).
