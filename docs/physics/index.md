---
hide:
  - toc
description: The differentiable forward models behind the 17 problems — convolution, NV dipolar spectra, magnetostatics, X-ray transforms, elliptic and parabolic PDEs, waves, scattering, optics and reaction–diffusion.
---

# Physics

In nefi the physics is a **hard constraint**: every candidate field is pushed through a
differentiable `Operator`, and the prediction is exactly the discretized physics applied to it.
Every operator evaluates at any curriculum resolution (`at_resolution`), runs on CPU and CUDA,
declares its homogeneity when it has one, and carries a *fidelity tag*. Every benchmark
measurement is simulated by a **different, higher-fidelity** model — a finer grid, float64, an
independent scheme — so that no result relies on an inverse crime.

## The families

| family | forward model | gradients | instances | benchmark data from |
|---|---|---|---|---|
| **Convolution**<br><small>`fft_convolution`</small> | `y = k ∗ x`, N-D, zero-padded FFT, kernel rebuilt per resolution | autograd · batchable | [`toy1d`](../instances/toy1d.md), [`deconvolution`](../instances/deconvolution.md), [`deconvolution3d`](../instances/deconvolution3d.md) | PSF on a 2× grid, float64 |
| **NV dipolar spectra**<br><small>`nv_relaxometry`, `nv_direct`</small> | tensor power kernel × Lorentzian (F2), coherent F1, source-side direct sum F3 | autograd · batchable (F1/F2) | [`nv_relaxometry`](../instances/nv_relaxometry.md) | F3 direct sum, float64 |
| **Planar magnetostatics**<br><small>`current_density`, `magnetization`, `biot_savart`</small> | stray field of sheet currents / magnetization at a standoff | autograd · batchable | [`current_density`](../instances/current_density.md) | real-space Biot–Savart sum, float64 |
| **X-ray transform**<br><small>`radon`, `radon3d`</small> | parallel-beam line integrals, all views in one `grid_sample` | exact autograd adjoint · batchable | [`sparse_view_ct`](../instances/sparse_view_ct.md), [`ct3d`](../instances/ct3d.md) | 2× finer image grid, float64 |
| **Poisson**<br><small>`poisson`</small> | `−κΔu = f`, spectral DST solver | autograd · batchable | [`poisson_source`](../instances/poisson_source.md) | finite differences on a 2× grid, float64 |
| **Variable-coefficient elliptic**<br><small>`elliptic`, `eit`, `darcy`, `diffuse_optical`</small> | `−∇·(σ∇u) + κu = f`, harmonic-mean finite volumes, Jacobi-PCG | implicit-function-theorem adjoint, warm starts | [`eit`](../instances/eit.md), [`darcy_flow`](../instances/darcy_flow.md), [`dot3d`](../instances/dot3d.md) | finite volumes on a 2× grid, float64 |
| **Transient heat**<br><small>`heat` (`heat_explicit` for data)</small> | implicit Euler, harmonic-mean faces, Jacobi / Chebyshev / CG sweeps | discrete adjoint at trajectory memory | [`thermal_tomography`](../instances/thermal_tomography.md) | explicit substepped simulator, float64 |
| **Acoustic waves**<br><small>`wave`, `wave_initial_condition`</small> | leapfrog, 4th-order Laplacian, PML (2-D) or sponge (3-D) | autograd or block checkpointing | [`wave_fwi`](../instances/wave_fwi.md), [`photoacoustic3d`](../instances/photoacoustic3d.md) | 2× finer grid, float64 |
| **Scattering**<br><small>`born`, `lippmann_schwinger`</small> | Born / Rytov; Lippmann–Schwinger for the data | autograd · batchable (Born) | [`diffraction_tomography`](../instances/diffraction_tomography.md) | iterated multiple scattering on a 2× grid, float64 |
| **Coherent optics**<br><small>`holography`, `phase_object`</small> | angular-spectrum propagation of a thin phase object | autograd · batchable | [`holography`](../instances/holography.md) | 2× sampling, float64 |
| **Reaction–diffusion**<br><small>`gray_scott`</small> | Gray–Scott, explicit Euler hitting the snapshot times exactly | autograd or block checkpointing | [`reaction_diffusion`](../instances/reaction_diffusion.md) | 4× smaller time step, float64 |
| **Bring your own**<br><small>`fourier_sampling`, `phase_retrieval`, `beer_lambert`, `sampling`, `saturation`, `time_stepper`, `function`</small> | Fourier sampling (MRI), phase retrieval, Beer–Lambert, sampling, saturation, generic time stepping, any function | autograd; checkpointed time stepping | your problem ([BYOP](../byop.md)) | your generator |

`nefi list operators` prints the full registry (36 entries, including the wrappers `nuisance`,
`stack`, `sum`, `pointwise` and `identity`). *Batchable* operators can solve many measurements in
one loop with [`nefi.batch_invert`](../performance.md#throughput-batched-multi-measurement-solving).

## Pages

<div class="grid cards" markdown>

-   :material-sine-wave: __[Elliptic & magnetostatics](../physics_elliptic.md)__

    ---

    One differentiable steady-state solver with an exact implicit-function-theorem adjoint
    behind EIT, Darcy flow and diffuse optics; planar magnetostatics for NV current imaging;
    a new elliptic instance in about 30 lines.

-   :material-waves: __[Wave, scattering, optics & reaction](../physics_wave_optics.md)__

    ---

    Time stepping and its memory trade-off, acoustic waves with PML, Born and
    Lippmann–Schwinger scattering, angular-spectrum optics, Gray–Scott dynamics.

-   :material-fire: __[Heat equation (NeFTY)](heat.md)__

    ---

    The implicit-Euler finite-volume solver, its discrete adjoint at trajectory-sized memory,
    the inner solvers and the explicit data simulator.

-   :material-function-variant: __[Write your own](../tutorials/03_new_forward_operator.md)__

    ---

    An operator is a differentiable function with `at_resolution` and a fidelity tag; pair it
    with an independent data generator and every tool in nefi works with it.

</div>
