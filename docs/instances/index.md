---
hide:
  - toc
description: The 17 registered inverse problems, from the paper instances and classic imaging to elliptic PDEs, wave / optics / reaction systems and volumetric tomography, each runnable end to end with one command.
---

# Instances

An **instance** is a complete, reproducible inverse problem: a config dataclass in which every
paper number is a field, scene generators with difficulty classes, an independent
high-fidelity data generator, the problem builder, a default curriculum, metrics and baselines.
The CLI, the benchmark harness, the diagnostics, the auto-tuner and the gallery all work on any
instance, including your own ([A new instance](../tutorials/04_new_instance.md)).

```bash
nefi list instances -v               # the registry: scene classes, baselines, metrics, configs
nefi run <name> --smoke              # seconds on a CPU
nefi run configs/<name>_full.yaml --device cuda
nefi bench <name> --methods neural,grid --classes all --n 8 --seeds 0,1,2
```

<div class="nf-strip" markdown>

[![NV relaxometry](../assets/gallery/thumbs/nv_relaxometry.png){ .off-glb loading=lazy }<span>NV relaxometry<small>paper · NeTMY</small></span>](nv_relaxometry.md)
[![Sparse-view CT](../assets/gallery/thumbs/sparse_view_ct.png){ .off-glb loading=lazy }<span>Sparse-view CT<small>classic imaging</small></span>](sparse_view_ct.md)
[![EIT](../assets/gallery/thumbs/eit.png){ .off-glb loading=lazy }<span>EIT<small>elliptic</small></span>](eit.md)
[![Holography](../assets/gallery/thumbs/holography.png){ .off-glb loading=lazy }<span>Holography<small>optics</small></span>](holography.md)
[![Full-waveform inversion](../assets/gallery/thumbs/wave_fwi.png){ .off-glb loading=lazy }<span>Full-waveform inversion<small>waves</small></span>](wave_fwi.md)
[![3-D CT](../assets/gallery/thumbs/ct3d.png){ .off-glb loading=lazy }<span>3-D sparse-view CT<small>volumetric</small></span>](ct3d.md)

</div>

## The papers

| instance | problem | unknown | physics | measurement |
|---|---|---|---|---|
| [`nv_relaxometry`](nv_relaxometry.md) | **NeTMY** — NV-center noise sensing | spin density ρ ≥ 0 and Larmor field ω<sub>L</sub> (2-D) | dipolar tensor power kernel × Lorentzian (F2) | noise spectra, 50 frequencies × 64² |
| [`thermal_tomography`](thermal_tomography.md) | **NeFTY** — pulsed-thermography tomography | diffusivity α ∈ [α<sub>min</sub>, α<sub>max</sub>] (3-D) | implicit-Euler heat equation, discrete adjoint | 100 surface frames × 64² |

## Classic imaging

| instance | problem | unknown | physics | measurement |
|---|---|---|---|---|
| [`toy1d`](toy1d.md) | 1-D deblurring (tutorial, CI) | signal x ≥ 0 | FFT Gaussian blur | 128 samples |
| [`deconvolution`](deconvolution.md) | image deblurring, Gaussian or Poisson noise | image x ≥ 0 | FFT convolution (Gaussian / motion / disk PSF) | blurred image |
| [`sparse_view_ct`](sparse_view_ct.md) | sparse-view and limited-angle CT | attenuation μ ∈ [0, 1] | differentiable Radon transform | sinogram, 20 views |

## Elliptic PDEs

| instance | problem | unknown | physics | measurement |
|---|---|---|---|---|
| [`poisson_source`](poisson_source.md) | source recovery | source f ≥ 0 | `−κΔu = f`, spectral solver | potential on 10 % of pixels, a boundary strip or all |
| [`eit`](eit.md) | electrical impedance tomography | conductivity σ | `−∇·(σ∇u) = 0`, implicit-function adjoint | boundary voltages of current patterns |
| [`darcy_flow`](darcy_flow.md) | hydraulic tomography | log-permeability | `−∇·(k∇p) = q`, implicit-function adjoint | well-gauge pressures, 6 configurations |
| [`current_density`](current_density.md) | NV current imaging | stream function of a sheet current | planar magnetostatics (FFT) | stray field B<sub>z</sub> |

## Wave, optics and reaction

| instance | problem | unknown | physics | measurement |
|---|---|---|---|---|
| [`wave_fwi`](wave_fwi.md) | full-waveform inversion | sound speed c | acoustic wave equation, leapfrog + PML | shot gathers |
| [`diffraction_tomography`](diffraction_tomography.md) | optical diffraction tomography | scattering contrast χ | Born / Rytov scattering | complex fields, 16 angles × 64 receivers |
| [`holography`](holography.md) | inline phase retrieval | phase φ | angular-spectrum propagation | intensities at 3 distances |
| [`reaction_diffusion`](reaction_diffusion.md) | Gray–Scott parameter inversion | feed-rate field F | reaction–diffusion time stepping | 4 snapshots of both species |

## Volumetric (3-D)

| instance | problem | unknown | physics | measurement |
|---|---|---|---|---|
| [`ct3d`](ct3d.md) | 3-D sparse-view CT | attenuation μ(x, y, z) | slice-stacked Radon transform | sinogram stack, 30 views |
| [`deconvolution3d`](deconvolution3d.md) | fluorescence z-stack deconvolution | fluorophore density | 3-D FFT convolution, elongated PSF | blurred z-stack |
| [`dot3d`](dot3d.md) | diffuse optical tomography | absorption μ<sub>a</sub> | diffusion equation, Robin boundaries, implicit-function adjoint | top-face reflectance, 3 × 3 sources |
| [`photoacoustic3d`](photoacoustic3d.md) | photoacoustic tomography | initial pressure p<sub>0</sub> | 3-D wave equation, sponge layer | traces of a 16 × 16 planar array |

`thermal_tomography` is volumetric as well; with the four systems above it forms the
[3-D showcase](../gallery/3d.md).

## Every instance page has

* **At a glance** — the problem, the unknown, the physics, the measurement, the difficulty
  classes, the baselines, the metrics and the run command, with the gallery figure;
* the forward model and its discretization, and the **inverse-crime guard**: how the data are
  simulated independently of the inversion operator;
* the prior, the objective, the curriculum and the full configuration;
* measured results — validation against analytic cases, smoke and paper-scale runs, baselines —
  and the known limitations, where they have been measured.
