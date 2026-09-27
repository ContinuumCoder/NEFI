---
hide:
  - toc
description: Task-oriented guides — build a problem, solve it well, evaluate it rigorously, and scale it up.
---

# Guides

Task-oriented pages, grouped by what you are trying to do. Each one is self-contained and links
to the API it uses; the [tutorials](../getting-started/index.md#the-tutorials) are the guided
path through the same material.

## Build a problem

<div class="grid cards" markdown>

-   :material-puzzle-outline: __[Bring your own problem](../byop.md)__

    ---

    `nefi.from_forward(forward, y, shape, prior=...)`: any differentiable PyTorch function in
    about twenty lines; the prior DSL; what "auto" decides and how to override it.

-   :material-view-grid-plus-outline: __[Operator, field & loss zoo](../zoo.md)__

    ---

    Every forward operator, representation, head and loss that composes with everything else —
    with a short guide to choosing among them.

-   :material-function-variant: __[A new forward operator](../tutorials/03_new_forward_operator.md)__

    ---

    Write the physics of your instrument, make it work at every curriculum resolution, check its
    gradient and pair it with an inverse-crime-safe data generator.

-   :material-package-variant-closed: __[A new instance](../tutorials/04_new_instance.md)__

    ---

    Package a complete, reproducible problem behind the `Instance` protocol and get the CLI,
    benchmarks, diagnostics and gallery for free.

</div>

## Solve it well

<div class="grid cards" markdown>

-   :material-shape-outline: __[Geometric representations](../geometry.md)__

    ---

    Layered media, anomalies, star shapes, warps and spectral preconditioning; residual-aware
    annealing, capacity growth, held-out representation selection and spectral match reports.

-   :material-tune-variant: __[Auto-tuning](../autotune.md)__

    ---

    Detect and fix gauges, budgets that stop short of the noise floor and mis-weighted priors;
    an acquisition report for what no tuning can fix.

-   :material-vector-difference: __[Edge refinement](../refinement.md)__

    ---

    Crisp interfaces from a smooth reconstruction, under the same physics and data — with an
    acceptance test that refuses what the data do not support.

-   :material-shield-check-outline: __[Robustness & adaptivity](../tutorials/08_robustness_and_adaptivity.md)__

    ---

    NaN guard with rollback, restarts, discrepancy stopping, nuisance parameters, ensembles and
    robust fidelities for unattended campaigns and real data.

</div>

## Evaluate

<div class="grid cards" markdown>

-   :material-stethoscope: __[Diagnostics](../tutorials/05_diagnostics.md)__

    ---

    Sensitivity, iter-0 gradients, filter-kernel rows, singular values, energy barriers and the
    data-fit paradox — why a method works, or fails.

-   :material-scale-balance: __[Benchmarking](../tutorials/06_benchmarking.md)__

    ---

    The papers' protocol for any instance: paired seeds, 95 % confidence intervals, ablations,
    sweeps, runtime and memory, and the inverse-crime guard.

-   :material-compare: __[Baselines](../baselines.md)__

    ---

    Grid (Tikhonov), L-BFGS, ADMM, Gaussian splats, Deep Decoder and closed-form references —
    and when a comparison is fair.

-   :material-chart-box-outline: __[Visualization](../visualization.md)__

    ---

    Reconstruction figures, measurement viewers, training dynamics, 3-D views, the interactive
    viewer and the multi-physics gallery.

-   :material-alert-circle-outline: __[Limitations and caveats](limitations.md)__

    ---

    The model-error floor, neural fields versus grids on linear problems, refused refinements,
    acquisition limits, seed variation and estimated CUDA numbers — each with what to do.

</div>

## Scale up

<div class="grid cards" markdown>

-   :material-speedometer: __[Performance](../performance.md)__

    ---

    What is fast and what is slow, batched solving, whole-step compilation, the heat stencil,
    sharded benchmarks and how to profile.

-   :material-server-network: __[GPU servers](../tutorials/07_running_on_gpu_servers.md)__

    ---

    Sync, environment, run scripts, SLURM arrays, memory and time expectations, checkpoints.

-   :material-file-document-check-outline: __[Reproducing the papers](../reproduce_papers.md)__

    ---

    The exact commands for the headline tables of NeTMY and NeFTY, their cost, and the caveats
    to check before comparing numbers.

-   :material-help-circle-outline: __[FAQ — failure modes](../faq.md)__

    ---

    Center collapse, cross artifacts, high-frequency leakage, back-face artifacts, stalls at high
    contrast, the data-fit paradox — symptoms, checks and fixes.

</div>
