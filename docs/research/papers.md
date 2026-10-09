---
description: NeTMY and NeFTY, the first two papers in the series nefi generalizes. What each contributes, what the library implements, and how to cite them.
---

# The papers

The series has two published papers so far. Both solve an inverse problem of scientific sensing
*without training data*: a coordinate
neural field is optimized for one measurement through a differentiable model of the instrument.
They differ in the instrument, in the geometry of the unknown, and therefore in the
representation that works.

| | **NeTMY** | **NeFTY** |
|---|---|---|
| paper | *Neural Fields for NV-Center Inverse Sensing* | *Neural Field Thermal Tomography: A Differentiable Physics Framework for Non-Destructive Evaluation* |
| arXiv | [2605.13988](https://arxiv.org/abs/2605.13988) | [2603.11045](https://arxiv.org/abs/2603.11045) |
| instrument | widefield NV-center array reading magnetic-noise spectra | infrared camera after a laser flash (pulsed thermography) |
| unknown | spin density ρ ≥ 0 and Larmor field ω<sub>L</sub> (2-D) | thermal diffusivity α ∈ [α<sub>min</sub>, α<sub>max</sub>] (3-D) |
| geometry of the unknown | sparse point-like sources | piecewise-constant defects in a (layered) bulk |
| physics | FFT convolution with tensor power-summed dipolar kernels × Lorentzian (F2) | implicit-Euler finite volumes, harmonic-mean faces, discrete adjoint |
| representation | gated softplus head, support-masked Larmor head, annealed Fourier features, two-stage multiscale | bounded sigmoid head, annealed Fourier features, isotropic TV |
| failure it explains | center collapse of free-density solvers | PINNs fit the surface data but not the interior |
| nefi instance | [`nv_relaxometry`](../instances/nv_relaxometry.md) | [`thermal_tomography`](../instances/thermal_tomography.md) |

## NeTMY — Neural Fields for NV-Center Inverse Sensing

*Zhixuan Zhao, Tao Zhong, Yixun Hu, Nathalie P. de Leon, Christine Allen-Blanchette.*
[arXiv:2605.13988](https://arxiv.org/abs/2605.13988), 2026.

A nitrogen-vacancy (NV) center in diamond senses the magnetic noise of nearby fluctuating spins;
a widefield array turns that into a noise spectrum at every pixel. Recovering the sparse spin
sources and their local Larmor frequencies from those spectra is nonlinear, spectrally coupled
and strongly blurred. NeTMY shows that replacing the common scalar (coherent) forward
approximation with a **tensor power-summed dipolar operator** changes the inverse landscape and
exposes a **center-collapse** failure of free-density optimization, and it solves the problem
with an amortization-free coordinate neural field: annealed positional encoding, multiscale
optimization, a sparsity gate and spectrum-fidelity losses. Its mechanism analysis shows that
the parameterization does not execute the raw density-space gradient — it smooths and
redistributes updates, which is exactly what defeats the collapse.

**In nefi:** the F1 / F2 operators and the F3 direct simulator (`nefi.instances.nv_relaxometry`),
the gated softplus and support-masked heads (`nefi.fields.heads`), log-MSE on normalized maps
and the direct-density term (`nefi.losses`), the energy-anchored scale correction (post-processing),
the eight scene classes and the localization metrics (Hungarian F1, sliced Wasserstein, GMSD); the
center-collapse diagnostics (iter-0 gradient, energy barrier, filter-kernel rows) are general
functions in `nefi.diagnostics`. `configs/nv_relaxometry_paper.yaml` holds Tables 5–7.

```bibtex
@article{zhao2026netmy,
  title   = {Neural Fields for {NV}-Center Inverse Sensing},
  author  = {Zhao, Zhixuan and Zhong, Tao and Hu, Yixun and de Leon, Nathalie P. and
             Allen-Blanchette, Christine},
  journal = {arXiv preprint arXiv:2605.13988},
  year    = {2026}
}
```

## NeFTY — Neural Field Thermal Tomography

*Tao Zhong, Yixun Hu, Dongzhe Zheng, Aditya Sood, Christine Allen-Blanchette.*
[arXiv:2603.11045](https://arxiv.org/abs/2603.11045), 2026.

Inverse heat conduction is severely ill-posed: the forward map damps interior structure before
it reaches the surface. Physics-informed networks that put the PDE into a soft residual penalty
fit the boundary measurements while leaving the interior essentially untouched. NeFTY keeps the
PDE **exact on the discretization**: at every step the candidate diffusivity field passes through
a differentiable implicit-Euler heat solver with harmonic-mean interface fluxes, and **adjoint
gradients** carry the surface error back to the network at solver-level memory cost. On
synthetic 3-D benchmarks it outperforms soft-constrained PINNs and a voxel grid, and it transfers
to real thermography data, where it beats classical signal-processing baselines in defect
segmentation and depth estimation.

**In nefi:** the finite-volume stencil, the inner solvers and the discrete adjoint
(`nefi/operators/pde/`, see [Heat equation](../physics/heat.md)), the explicit substepped data
simulator, the bounded head and isotropic TV, the soft-PINN, PPT and TSR baselines, the
segmentation / depth metrics and the 2-D / 2.5-D projections of the real-data protocol.
`configs/thermal_tomography_paper.yaml` holds Table 5.

```bibtex
@article{zhong2026nefty,
  title   = {Neural Field Thermal Tomography: A Differentiable Physics Framework for
             Non-Destructive Evaluation},
  author  = {Zhong, Tao and Hu, Yixun and Zheng, Dongzhe and Sood, Aditya and
             Allen-Blanchette, Christine},
  journal = {arXiv preprint arXiv:2603.11045},
  year    = {2026}
}
```

## One recipe

Put side by side, the two methods are the same five pieces with different settings: **field →
hard physics → losses → curriculum → solver**. nefi makes each piece a swappable component and
adds the parts both papers needed but did not package — diagnostics that explain a result, a
benchmark protocol with confidence intervals and an inverse-crime guard, and a representation
layer that treats geometry as a design axis ([The representation is the prior](thesis.md)).

* **Reproduce:** [the server runbook](../reproduce_papers.md) lists the exact commands for the
  headline tables, their cost and the caveats to check first.
* **Trace:** [Paper → code mapping](../paper_mapping.md) maps every equation, proposition and
  table to the code.
* **Caveats:** [Limitations and caveats](../guides/limitations.md) lists the model-error floor,
  seed variation and the other effects to account for when comparing numbers.
