---
description: Known limitations and caveats — the model-error floor of the heat operator, neural fields versus grids on linear problems, refused refinements, acquisition limits, seed variation and estimated CUDA numbers.
---

# Limitations and caveats

Each entry states what happens, why, and what to do about it, with the measurements behind it
and a link to the page where they are documented.

## The model-error floor (thermal tomography)

**What happens.** Benchmark data for `thermal_tomography` come from an independent explicit
simulator, so even the ground-truth diffusivity, re-simulated with the inversion operator, does
not fit the surface data exactly. At the paper grid this floor is about 46–51 dB surface PSNR
(flash-layer width `w_z` = 0.1 / 0.2); smoke runs reach 37–45 dB. The inversion absorbs the
mismatch by biasing α near the front face and in the bulk: the gallery reconstruction finds the
defect (IoU 0.98) at a PSNR of 10.8 dB.

**Why.** Implicit Euler at one step per camera frame under-resolves the fast through-thickness
decay of the thin post-flash layer: at the paper grid with `w_z = 0.1` the surface mismatch is
about 10 % (relative rms) in the first frame, and for some scenes the defect signal is barely
larger than that.

**What to do.** Compare `surface_psnr` with the floor before reading IoU. The knobs are
`implicit_substeps` (error `O(Δt/m)`), `first_frame` (drop the earliest frames) and
`flash_width_z`. Details: [thermal tomography · inverse-crime guard](../instances/thermal_tomography.md#inverse-crime-guard-nefty-51-app-e1),
[reproducing the papers](../reproduce_papers.md#nefty--thermal-tomography).

## Neural field versus grid on linear problems

**What happens.** On well-conditioned *linear* problems with a well-tuned explicit prior, a free
grid with TV matches or beats the neural field at equal budget:

| problem (seed 0, CPU) | neural field | grid + TV |
|---|---|---|
| deconvolution, 32², 450 steps | 28.05 dB / SSIM 0.950 | 29.07 dB / 0.971 |
| deconvolution, 64², 900 steps | 33.60 dB / 0.981 | 35.13 dB / 0.987 |
| 3-D CT, ellipsoids, 12 views | 27.3 dB / SSIM 0.853 | 29.2 dB / 0.804 |
| toy1d benchmark, gallery budget, 2 samples × 2 seeds | 22.0 ± 5.5 dB | 32.9 ± 4.2 dB |

**Why.** With TV in the objective the free grid is the MAP estimate of that objective; when the
operator is well conditioned and its gradient unbiased, the representation's filter
$G_\theta$ has little left to add.

**What to do.** Benchmark both (`nefi bench <name> --methods neural,grid`). The representation
matters when the explicit prior is removed (deconvolution with 3 % noise and no TV: neural field
29.39 dB, grid 22.99 dB), when data are missing (3-D CT with a 90° wedge: 24.9 dB / SSIM 0.777
against 23.7 dB / 0.616; the neural field has the higher SSIM in every 3-D CT setting), when the
raw gradient is biased (the center collapse of free densities under NeTMY's operator), and when
the physics hides the interior (surface-only heat data: grid IoU 0 against 0.97 for the neural
field on the same measurement). In NeTMY's cross-fidelity setting at a reduced budget the
neural field leads on SWD, GMSD and MSE while the Tikhonov grid stays level on Hungarian F1
(0.92 against 0.90), because a one-pixel point-spread function makes sparse scenes comparatively
easy. Details: [baselines § 4](../baselines.md#4-when-is-a-comparison-fair),
[deconvolution](../instances/deconvolution.md#validation), [ct3d](../instances/ct3d.md),
[thermal tomography](../instances/thermal_tomography.md#validation-tests),
[NV relaxometry](../instances/nv_relaxometry.md#expected-outputs).

## Refused and accepted refinements

**What happens.** [Edge refinement](../refinement.md) returns the smooth result, with the reason,
when the refined field raises the data misfit χ by more than the tolerance (`fit_tol = 1.1`) or
moves an interface outside the smooth solution's transition band. `thermal_tomography` is refused
on both smoke scenes; `eit` is refused although its PSNR would rise by 0.9 / 3.5 dB (χ 1.01 → 1.24
/ 1.06 → 1.33). Conversely, acceptance certifies consistency with the data, not accuracy: on
`toy1d`'s smooth bumps a two-phase level set fits better than the control (χ 1.29 against 3.65),
is accepted, and ends 0.8 dB further from the truth.

**Why.** The acceptance test works in data space and cannot see the ground truth; with model
error even the ground truth misfits the data (χ 1.33 / 1.62 on the `eit` scenes).

**What to do.** Compare every refinement with the `continue` control that the report includes (on
`ct3d` the +3.7 / +3.9 dB gain is almost all continuation; on `dot3d` the +3.2 / +2.8 dB gain is
the prior's). When the noise or model-error level is known, pass it (`sigma=`) or relax
`fit_tol`, and report that choice. Choose a piecewise-constant prior for physical reasons, not
because it looks sharper. Details: [edge refinement § 6–7](../refinement.md#6-results-on-the-volumetric-instances).

## Acquisition geometry

**What happens.** Auto-tuning fixes budgets, learning rates, gauges and regularization strength,
but not missing information. On `wave_fwi` with 2 sources × 8 receivers it brings the misfit to
the noise floor (χ 2.92 → 1.03) while the PSNR moves from 14.09 to 13.33 dB.

**Why.** An under-determined acquisition leaves directions of the unknown that no data term sees;
every tuning criterion lives in data space.

**What to do.** Read the acquisition report (`nefi autotune <name>`): it names the weakest sides
and the singular-value decay. Add illumination — a surround array is well determined
(σ₆/σ₁ = 0.82) — or choose a smoother or layered prior. Details:
[auto-tuning](../autotune.md#the-four-failure-types-measured).

## Variation across scenes and seeds

**What happens.** Per-measurement optimization depends on the scene and the seed:
`thermal_tomography` smoke IoU ranges over 0.26–0.97 across seeds 0–4 (mean 0.50);
`nv_relaxometry` at 100 steps gives Hungarian F1 0.75–1.0 across ten seeds (mean 0.86), and 1.0
on nine of ten seeds at 600 steps. NeTMY reports centred-minimum trapping on about 5 % of
samples.

**Why.** The objectives are non-convex and identifiability differs from scene to scene.

**What to do.** Use several seeds and `restarts > 1`, and report `nefi bench` confidence
intervals rather than single runs; the gallery numbers are single smoke runs. Details:
[thermal tomography](../instances/thermal_tomography.md#validation-tests),
[NV relaxometry](../instances/nv_relaxometry.md#expected-outputs),
[benchmarking](../tutorials/06_benchmarking.md).

## Measured and estimated performance

**What happens.** Performance numbers in this documentation were measured on a CPU unless they
are labelled otherwise. Expected CUDA speed-ups — the heat solver at the paper grid, batched
throughput approaching the batch size — are **estimates from kernel-launch counts**. Switches
meant for GPUs can be slower on a CPU: bf16 autocast by 1.4–4.6×, and whole-step compilation of
`deconvolution`, where the compiler falls back on complex FFT operations.

**What to do.** Profile on your own hardware with `tools/profile_instance.py`, and use
`autocast` on CUDA tensor cores. Details: [performance](../performance.md#expected-cuda-numbers-estimates-not-measured).

## Data and compute not included

The real PVC thermography recordings used by NeFTY (§5.3) are not distributed with nefi; the
[runbook](../reproduce_papers.md#real-pvc-data-nefty-53) describes how to load them. Paper-scale
tables need GPU time: the NeTMY Table 1 protocol is about ten GPU-hours.
