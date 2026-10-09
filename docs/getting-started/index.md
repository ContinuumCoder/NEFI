---
description: Install nefi, run a first inversion from Python and from the command line, and find the next step.
---

# Get started

nefi needs Python ≥ 3.10 and PyTorch ≥ 2.1. A CPU is enough for every tutorial, every smoke run
and the whole test suite; paper-scale runs go to a CUDA machine.

## Install

=== "pip"

    ```bash
    pip install nefi
    ```

    Installs the library and the `nefi` command with its core dependencies (torch, numpy,
    scipy, pyyaml, tqdm). Extras add figures, the richer command line, or the development tools:

    ```bash
    pip install "nefi[viz]"        # + matplotlib: figures, --plot, the gallery
    pip install "nefi[cli]"        # + typer, rich: the richer command line
    pip install "nefi[dev]"        # + pytest, ruff, mypy, scikit-image, matplotlib, typer, rich
    ```

=== "Latest development version"

    ```bash
    pip install git+https://github.com/ContinuumCoder/NEFI.git
    ```

    The current state of the `main` branch.

=== "From source"

    ```bash
    git clone https://github.com/ContinuumCoder/NEFI.git && cd NEFI
    pip install -e ".[dev]"      # + pytest, ruff, mypy, scikit-image, matplotlib, typer, rich
    make test                    # fast CPU suite: python -m pytest -q -m "not slow"
    ```

    This is the setup for contributing and for the examples under `examples/`.

=== "Conda (GPU servers)"

    ```bash
    git clone https://github.com/ContinuumCoder/NEFI.git && cd NEFI
    conda env create -f environment.yml && conda activate nefi    # python 3.11, torch, nefi[dev]
    python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
    ```

    The pip `torch` wheel on Linux bundles its CUDA runtime; to pin a CUDA version install
    torch from the matching index before the package. See
    [Running on GPU servers](../tutorials/07_running_on_gpu_servers.md).

Check the installation:

```bash
nefi --version
nefi list instances        # the 17 registered problems
nefi run toy1d --smoke     # generate → invert → evaluate, a few seconds on a CPU
```

!!! tip "Devices"
    `device="auto"` (the default everywhere) uses `$NEFI_DEVICE` when set, else CUDA when
    available, else the CPU. Apple MPS is used only when requested explicitly (`--device mps`):
    it lacks float64 and some FFT and `grid_sample` kernels that several operators need.

## A first inversion

`toy1d` recovers a non-negative 1-D signal from its blurred, noisy observation. It exercises
every core mechanism — a neural field with annealed Fourier features, a softplus head, an FFT
convolution rebuilt at every resolution, a two-stage multiscale curriculum, TV regularization,
and an inverse-crime-safe data generator — and finishes in seconds.

=== "Python"

    ```python
    import nefi
    from nefi.instances.toy1d import make_problem

    problem, gt, meas = make_problem(n=128, seed=0)   # InverseProblem, ground truth, Measurement
    result = nefi.invert(problem, device="cpu", seed=0)
    print(result.summary())
    print("PSNR", nefi.metrics.psnr(result.fields["x"], gt["x"]))
    ```

=== "Command line"

    ```bash
    nefi run toy1d --plot        # → runs/toy1d-<hash>/{result.pt, config.yaml, metrics.json,
                                 #    history.csv, fields.png, history.png}
    nefi run runs/toy1d-<hash>/config.yaml     # re-runs exactly that configuration
    ```

=== "Your own forward model"

    ```python
    import torch
    import nefi

    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
    k = (k[:, None] * k[None]) / 256

    def forward(x):                                   # a blurred, saturating camera
        blurred = torch.nn.functional.conv2d(x[None, None], k[None, None], padding=2)[0, 0]
        return torch.tanh(1.5 * blurred)

    x_true = torch.zeros(64, 64); x_true[20:44, 16:40] = 1.0
    y = forward(x_true) + 0.01 * torch.randn(64, 64)

    problem = nefi.from_forward(forward, y, shape=(64, 64),
                                prior="nonnegative + piecewise_constant")
    result = nefi.invert(problem)
    print(nefi.quick_report(result, problem, gt=x_true))
    ```

The [quickstart tutorial](../tutorials/01_quickstart.md) walks through the output line by line,
and [Bring your own problem](../byop.md) explains every automatic decision of `from_forward`.

## Where next

<div class="grid cards" markdown>

-   :material-school-outline: __Learn the recipe__

    ---

    [Quickstart](../tutorials/01_quickstart.md) → [Anatomy of a problem](../tutorials/02_anatomy_of_a_problem.md)
    → [Core concepts](concepts.md). Seven pieces, one contract.

-   :material-puzzle-outline: __Invert your own physics__

    ---

    [Bring your own problem](../byop.md) with `from_forward` and the prior DSL, then the
    [zoo](../zoo.md) of operators, representations and losses.

-   :material-image-multiple-outline: __See what it does__

    ---

    The [gallery](../gallery/index.md) of 17 systems, the [3-D showcase](../gallery/3d.md) with
    interactive viewers, and the [instance catalogue](../instances/index.md).

-   :material-flask-outline: __Reproduce the papers__

    ---

    [The papers](../research/papers.md), the [paper → code map](../paper_mapping.md) and the
    [server runbook](../reproduce_papers.md) with exact commands and costs.

</div>

## The tutorials

| # | tutorial | what you learn |
|---|---|---|
| 1 | [Quickstart](../tutorials/01_quickstart.md) | `toy1d` end to end, from Python and from the CLI; what a run writes |
| 2 | [Anatomy of a problem](../tutorials/02_anatomy_of_a_problem.md) | Domain, Field, Operator, Losses, Curriculum, Solver, post-processing — built by hand |
| 3 | [A new forward operator](../tutorials/03_new_forward_operator.md) | an operator, an inverse-crime-safe data generator, `at_resolution` for multiscale |
| 4 | [A new instance](../tutorials/04_new_instance.md) | the `Instance` protocol, the registry, config YAML and the CLI |
| 5 | [Diagnostics](../tutorials/05_diagnostics.md) | what each diagnostic tells you, with the papers' numbers as reference points |
| 6 | [Benchmarking](../tutorials/06_benchmarking.md) | protocol, confidence intervals, ablations, sweeps; cross-fidelity vs matched operator |
| 7 | [Running on GPU servers](../tutorials/07_running_on_gpu_servers.md) | scripts, SLURM, memory and time expectations, `torch.compile`, checkpoints |
| 8 | [Robustness and adaptivity](../tutorials/08_robustness_and_adaptivity.md) | safety features, noise-aware stopping, nuisance parameters, ensembles, robust fidelities |
