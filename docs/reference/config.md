---
description: The YAML configuration files of nefi — layouts, the instance / curriculum / solver / run sections, overrides, smoke presets, plugins and what a run writes.
---

# Configuration files

Every run is described by a small YAML (or JSON) file, and every run writes back the *effective*
configuration it used, so `nefi run <that file>` reproduces it. The bundled files live in
`configs/`: a `<name>_smoke.yaml` (seconds on a CPU) and a `<name>_full.yaml` sized for a GPU
for every instance, plus `nv_relaxometry_paper.yaml` and `thermal_tomography_paper.yaml`, which
spell out the papers' settings table by table.

## Layout

Two equivalent layouts are accepted:

=== "instance + config"

    ```yaml
    instance: toy1d                  # a registered instance name
    config:                          # fields of the instance's config dataclass
      n: 128
      sigma: 0.02
      scene: bumps
      steps: [300, 600]
    curriculum: null                 # optional: a full nefi.solve.Curriculum as a dict
    solver:                          # optional: nefi.Solver keyword arguments
      nan_guard: true
    run:                             # optional: defaults for the CLI
      seed: 0
      device: auto
      methods: [neural, grid]
    ```

=== "instance mapping"

    ```yaml
    instance:
      type: nv_relaxometry           # the registered name
      n: 64                          # every other key is a config field
      spacing: 20.0
      z0: 20.0
      steps: [3000, 7000]
    run:
      seed: 0
    ```

| section | content |
|---|---|
| `instance` | a registered name, or a mapping with `type:` plus config fields |
| `config` | fields of the instance's config dataclass (`nefi list instances -v` prints them with their defaults); unknown keys are an error that lists the valid ones |
| `curriculum` | optional; replaces the instance's default schedule (see below) |
| `solver` | optional keyword arguments of `nefi.Solver`: `compile`, `cuda_graphs`, `autocast`, `nan_guard`, `dtype`, `checkpoint_every`, … |
| `run` | optional CLI defaults: `seed`, `device`, `scene`, `methods`, `out` |
| `imports` | optional module names or `.py` paths to import first (instances defined outside the package) |

## The curriculum

A curriculum is a list of stages plus the optimizer and the stopping rules; each stage fixes a
resolution, a budget, a learning-rate schedule and the annealing of the Fourier bands.

```yaml
curriculum:
  stages:
    - {name: coarse, shape: [32, 32], steps: 150, lr: 3.0e-3}
    - {name: fine, shape: [64, 64], steps: 250, lr: 1.5e-3, anneal_fraction: 0.5}
  optim: {optimizer: adamw, weight_decay: 1.0e-4, grad_clip: 1.0}
  discrepancy_tau: 1.0               # Morozov: end a stage once RMSE <= tau * noise_std
  restarts: 1
```

| `Stage` field | default | meaning |
|---|---|---|
| `name`, `shape` | `stage`, native grid | label and resolution of the stage |
| `steps`, `lr` | 1000, 1e-3 | budget and peak learning rate |
| `lr_schedule` | `cosine` | `cosine`, `step`, `constant`, `warmup_cosine` |
| `lr_min_ratio`, `lr_step_size`, `lr_gamma`, `warmup_steps` | 0.01, 1000, 0.1, 0 | schedule details |
| `anneal`, `anneal_fraction` | true, 1.0 | re-anneal the Fourier bands from 0 over this fraction of the stage |
| `loss_weights` | none | per-stage weight overrides, e.g. `{log_mse: 0.0, noise_map: 2.0}` |
| `freeze` | none | parameter-name prefixes to freeze in this stage |

| `Curriculum` / `OptimConfig` field | default | meaning |
|---|---|---|
| `optim.optimizer` | `adamw` | `adamw`, `adam`, `sgd`, `lbfgs` |
| `optim.weight_decay`, `optim.grad_clip`, `optim.betas` | 1e-4, 1.0, (0.9, 0.999) | optimizer settings |
| `optim.ema` | none | exponential moving average of the parameters |
| `optim.lr_mult` | {} | per-parameter-prefix learning-rate multipliers (geometric parameters) |
| `early_stop_patience`, `early_stop_min_delta` | none, 1e-6 | stop a stage when the loss stalls |
| `discrepancy_tau` | none | Morozov discrepancy stopping when the noise level is known |
| `restarts` | 1 | independent restarts; the best final data loss wins |
| `time_budget_s` | none | wall-clock limit per solve |

## Overrides

`--set KEY=VALUE` (repeatable) changes a configuration without editing the file. The prefix
decides what is changed; values are parsed as numbers, booleans, `none`, lists or mappings
(`[300, 600]`, `{tv: 0.001}`), and strings otherwise.

| key | changes | example |
|---|---|---|
| `NAME` or `config.NAME` | a config field | `--set n=64 --set scene=piecewise` |
| `stage.ATTR` | that attribute of every stage | `--set stage.anneal_fraction=0.5` |
| `curriculum.ATTR` | the curriculum | `--set curriculum.restarts=2` |
| `optim.ATTR` | the optimizer | `--set optim.weight_decay=0` |
| `solver.ATTR` | a solver option | `--set solver.compile=step` |
| `run.ATTR` | a CLI default | `--set run.seed=3` |

## Smoke presets

`--smoke` shrinks a problem for a quick local check. It applies, in this order of preference,
the instance's `smoke_overrides`, `configs/<name>_smoke.yaml`, or a `"smoke"` entry of the
instance module's presets — and `nefi run` additionally caps the total number of steps
(`--smoke-steps`, default 40). `nefi autotune --smoke` uses the preset without the cap.

## What a run writes

| file | content |
|---|---|
| `result.pt` | the `nefi.Result`: fields, prediction, history, timings |
| `config.yaml` | the effective configuration (instance config + curriculum + seed); re-runnable |
| `metrics.json` | instance metrics, data-fit summary, timings, stage results |
| `history.csv` | per-step loss components, learning rate, annealing progress |
| `fields.png`, `history.png` | with `--plot` |

Benchmarks write `summary.md`, `summary.csv`, `rows.csv` and `benchmark.json`; the auto-tuner
writes `autotune.md`, `autotune.json` and `tuned_config.yaml`. Everything goes under `runs/`,
which is ignored by git.
