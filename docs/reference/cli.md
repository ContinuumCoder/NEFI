---
description: The nefi command line — list, run, bench, bench-merge, diagnose, autotune, ablate and sweep — with every option.
---

# Command line

The `nefi` command drives any registered instance or YAML configuration through the same
pipeline as the Python API. It uses [typer](https://typer.tiangolo.com/) when installed
(`pip install "nefi[cli]"`) and falls back to `argparse` with identical commands and options
(`NEFI_CLI=argparse` or `NEFI_CLI=typer` forces one).

```text
nefi list [KIND] [-v] [--json]                      registered instances, fields, operators, losses, …
nefi run TARGET [--smoke] [--plot] [--baseline NAME] [--set KEY=VALUE …] [--device cuda]
nefi bench TARGET --methods neural,grid --n 4 --seeds 0,1,2 [--classes all] [--by-class]
nefi bench-merge DIR                                merge the shard files of a cluster campaign
nefi diagnose TARGET [--solve] [--plot]             ill-posedness diagnostics report
nefi autotune TARGET [--level quick|standard|thorough] [--compare] [--smoke]
nefi ablate TARGET --variants no_annealing,no_pe,single_stage,drop_loss:tv
nefi sweep TARGET --axis lr --values 1e-4,1e-3,1e-2
```

`TARGET` is a registered instance name (`nefi list instances`) or a configuration file
(`.yaml` / `.json`, see [Configuration files](config.md)). Every command accepts `--set KEY=VALUE`
(repeatable) to override configuration fields, curriculum settings and solver options, and
`--device auto | cpu | cuda | cuda:1 | mps`.

## Options shared by the solving commands

| option | meaning |
|---|---|
| `--device`, `-d` | `auto` (default: `$NEFI_DEVICE`, else CUDA if available, else CPU), `cpu`, `cuda`, `cuda:1`, `mps` |
| `--seed` | data and optimization seed (default 0) |
| `--scene` | scene class (`nefi list instances -v` lists them) |
| `--out`, `-o` | output directory (default `runs/<name>-<hash>`) |
| `--smoke` | the instance's small preset plus a step cap, for a quick check |
| `--smoke-steps` | total step cap with `--smoke` (default 40) |
| `--set KEY=VALUE` | override a config field (`n=64`), a curriculum setting (`stage.lr=1e-3`, `curriculum.restarts=2`, `optim.weight_decay=0`) or a solver option (`solver.compile=true`) |
| `--quiet`, `-q` / `--verbose`, `-v` | no progress bars / log solver progress |

## `nefi list`

```bash
nefi list                      # every registry: instances, fields, operators, losses, baselines, …
nefi list instances -v         # scene classes, baselines, metrics and full default configs
nefi list configs              # the bundled configs/*.yaml
nefi list operators --json     # machine-readable
```

## `nefi run`

Generate a measurement with the instance's independent data generator, invert it, evaluate it
and save `result.pt`, the effective `config.yaml`, `metrics.json` and `history.csv`.

| option | meaning |
|---|---|
| `--plot` | also save `fields.png` and `history.png` |
| `--baseline NAME`, `-b` | run a baseline (`grid`, `admm`, `fbp`, …) instead of the neural field |

```bash
nefi run deconvolution --smoke --plot
nefi run configs/nv_relaxometry_paper.yaml --scene many/close --seed 0 --plot --device cuda
nefi run eit --smoke --baseline grid
nefi run runs/toy1d-<hash>/config.yaml           # re-runs exactly the saved configuration
```

## `nefi bench`

Paired benchmark: every method sees the same measurements (samples × seeds); tables report the
mean ± a Student-t 95 % confidence interval, runtime and peak memory, as markdown, CSV and JSON.

| option | meaning |
|---|---|
| `--methods`, `-m` | comma list, e.g. `neural,grid` (default: all of the instance's methods) |
| `--n`, `-n` | measurements per scene class (default 4) |
| `--seeds` | optimization seeds (default `0,1,2`) |
| `--classes` | scene classes, a comma list or `all` |
| `--metrics` | registered metric names (default: the instance's) |
| `--budget-scale` | multiply every stage's steps (default 1.0) |
| `--by-class` | add a per-class table |
| `--allow-inverse-crime` | permit the matched-operator regime (labelled in every report) |
| `--batched`, `--batch-size` | solve batchable methods in one loop ([`nefi.batch_invert`](../performance.md#throughput-batched-multi-measurement-solving)) |
| `--shard I/N` | run shard `i` of `n` of the (class, sample, seed) units; merge with `bench-merge` |

```bash
nefi bench toy1d --methods neural,grid --n 4 --seeds 0,1
nefi bench configs/deconvolution_full.yaml --n 32 --seeds 0,1,2 --batched --shard 0/8 --out runs/dc
```

## `nefi bench-merge`

```bash
nefi bench-merge runs/dc [--by-class] [--allow-missing] [--out DIR]
```

Checks that the shards come from the same protocol, restores the canonical row order and writes
`summary.md`, `summary.csv`, `rows.csv` and `benchmark.json` — identical to the unsharded run.

## `nefi diagnose`

Ill-posedness diagnostics with the papers' numbers as reference points: sensitivity maps,
iter-0 gradient and center bias, filter-kernel rows, singular values, energy barriers.

| option | meaning |
|---|---|
| `--solve` | solve first, then diagnose at the solution |
| `--plot` | also save `diagnostics.png` |
| `--probes` | sensitivity probes (default 32) |
| `--k`, `--n-iter` | singular values to estimate (default 8) and Lanczos iterations (default 24) |

```bash
nefi diagnose toy1d
nefi diagnose configs/nv_relaxometry_paper.yaml --solve --plot --device cuda
```

## `nefi autotune`

Estimate the noise, detect and repair gauges, report the acquisition geometry, tune the budget
and learning rate against the noise floor and the regularization by the discrepancy principle;
writes `autotune.md`, `autotune.json` and a `tuned_config.yaml` that `nefi run` reproduces.

| option | meaning |
|---|---|
| `--level`, `-l` | `quick`, `standard` (default) or `thorough` (adds a held-out search) |
| `--compare` | also solve the default and the tuned configuration and report both |
| `--smoke` | the instance's smoke preset (the budget is tuned, not capped) |

```bash
nefi autotune eit --smoke --level standard --compare --out runs/eit-autotune
nefi run runs/eit-autotune/tuned_config.yaml
```

## `nefi ablate` and `nefi sweep`

Cumulative ablations (NeTMY Tab. 3) and one-axis sweeps (NeTMY Tab. 10) on the benchmark
protocol; they share `--n`, `--seeds`, `--classes`, `--budget-scale` and `--base` (the method to
modify, default `neural`) with `nefi bench`.

| `--variants` modifier | effect |
|---|---|
| `no_annealing`, `no_pe` | switch off frequency annealing / the positional encoding |
| `single_stage`, `grid_field`, `no_gate` | one stage / a free grid / no sparsity gate |
| `drop_loss:NAME`, `set_weight:NAME=W` | remove a loss term / change its weight |
| `set:KEY=V`, `stage:ATTR=V`, `optim:ATTR=V` | change a config field, a stage or the optimizer |

```bash
nefi ablate nv_relaxometry --set preset=paper \
     --variants no_annealing,no_pe,single_stage,drop_loss:l1,no_gate --n 16 --seeds 0,1,2
nefi ablate thermal_tomography --variants no_pe,no_annealing --independent   # one modifier per row
nefi sweep nv_relaxometry --axis lr --values 1e-4,5e-4,1e-3,5e-3 --n 8
nefi sweep toy1d --axis stage.anneal_fraction --values 0.25,0.5,1.0
```

`--axis` accepts a config field or `stage.*`, `optim.*` and `weight.*` settings.

## Your own instances

Instances defined outside the package become visible to every command by importing the module
that registers them: list it in `NEFI_PLUGINS` (comma-separated module names or `.py` paths), in
the `imports:` key of a configuration file, or expose it as a `nefi.plugins` entry point of an
installed package ([A new instance](../tutorials/04_new_instance.md)).
