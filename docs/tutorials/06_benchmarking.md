# 6 · Benchmarking: protocol, confidence intervals, ablations, sweeps

`nefi.bench` implements the evaluation protocol of both papers for any instance: the same
measurements for every method, several optimization seeds, mean ± 95 % confidence intervals,
runtime and memory columns, cumulative ablations, one-axis sweeps, and an inverse-crime guard that
makes the cross-fidelity regime the default and the matched-operator regime an explicit, labelled
choice.

```bash
nefi bench toy1d --methods neural,grid --n 4 --seeds 0,1,2
nefi ablate toy1d --variants no_annealing,no_pe,single_stage --n 4 --seeds 0,1
nefi sweep toy1d --axis lr --values 1e-3,3e-3,1e-2 --n 4
python examples/benchmark_demo.py            # all of the above from Python (~1 min on a CPU)
```

## The protocol

```python
from nefi.bench import run_benchmark
from nefi.instances.toy1d import Toy1D

inst = Toy1D(n=128, scene="mixed")
res = run_benchmark(
    inst,
    methods="neural,grid",      # instance default + baseline keys, or Method objects
    n_samples=3,                # measurements per scene class (data seeds 0, 1, 2)
    seeds=(0, 1),               # optimization seeds per measurement
    classes=None,               # None: instance default class; "all"; or a list
    budget_scale=0.3,           # scale every stage's step count
    device="cpu",
    out_dir="runs/bench-toy1d", # summary.md, summary.csv, rows.csv, benchmark.json
)
print(res.to_markdown())
print(res.efficiency_table())
```

```text
**toy1d** — cross-fidelity (data: `gaussian-blur-4x-float64`; inversion: `FFTConvolution`) · 3 samples × 2 seeds × 1 class · mean ± 95% CI (Student t over samples × seeds)

| Method | n | mse ↓ | psnr ↑ | relative_error ↓ | time (s) | peak mem (MB) | steps |
|:---|---:|---:|---:|---:|---:|---:|---:|
| neural | 6 | **0.03134 ± 0.0131** | **21.65 ± 3.27** | **0.2729 ± 0.0809** | 1.057 ± 0.196 | — | 270 |
| grid | 6 | 0.07099 ± 0.0725 | 19.63 ± 2.3 | 0.3426 ± 0.0938 | 0.2927 ± 0.0833 | — | 270 |

| Method | params | steps | ms / step | wall-clock (s) | peak mem (MB) |
|:---|---:|---:|---:|---:|---:|
| neural | 14273 | 270 | 3.92 | 1.06 | — |
| grid | 128 | 270 | 1.08 | 0.293 | — |
```

What the harness does, in order:

1. **Resolve methods.** `"neural"` (aliases `nefi`, `default`, `ours`, `netmy`, `nefty`) is the
   instance's own method (`build_problem` + its curriculum); other names are keys of
   `instance.baselines()`. A `Method(name, build)` wraps any callable
   `(instance, measurement) -> (problem, curriculum | None[, solver_kwargs])`.
2. **Pre-flight.** The first measurement is generated and every method's problem is built once:
   the **inverse-crime guard** compares the data generator's `fidelity_tag` with each inversion
   operator's tag, and an untimed one-step warm-up absorbs one-time costs (lazy imports, CUDA
   context, FFT plans) so the first timed run is not penalized.
3. **Runs.** For every class × sample (data seed `i`) the measurement is generated once and shared;
   for every method × seed the global RNG is seeded *before* building the problem (so the seed
   controls the network initialization), the curriculum is scaled by `budget_scale` (and capped by
   `max_total_steps`), and the problem is solved — non-gradient baselines (ADMM, closed-form
   references) are dispatched through `problem.meta["solver"]`. Wall-clock, peak GPU memory
   (`torch.cuda.max_memory_allocated`; `—` on CPU), steps, stop reasons, the final data loss and
   the parameter count are recorded next to the instance metrics. A failing run is recorded in the
   `error` column instead of aborting the benchmark (`fail_fast=True` re-raises).
4. **Aggregation.** `res.table(ci=0.95, by_class=False)` returns, per method (and class), the mean
   and the Student-t confidence half-width over all samples × seeds:
   `hw = t_{(1+ci)/2, n−1} · sd / sqrt(n)` (undefined for a single run). Metric directions (↑/↓)
   are inferred from the names (`psnr`, `ssim`, `iou`, `f1`, … higher; `mse`, `error`, `gmsd`,
   `swd`, … lower) and the best mean is bold.

`BenchmarkResult` API: `rows` (raw), `table()`, `best(metric)`, `to_markdown(by_class=...)`,
`efficiency_table()` (NeFTY Tab. 9 style), `to_csv()`, `summary_csv()`, `to_json()`,
`save(dir)`, `BenchmarkResult.load(dir)`. `metrics=` overrides the instance metrics with
registered metric names or `{name: fn}`.

## Cross-fidelity vs matched-operator

Both papers treat the **cross-fidelity** regime as primary evidence: data are simulated with a
higher-fidelity or differently discretized model (NeTMY: direct source-side simulator F3; NeFTY:
explicit finite volumes) and inverted with the method's operator (F1/F2; implicit Euler). The
**matched-operator** regime (NeTMY Tab. 2: F1/F1, F2/F2) reuses the inversion operator for the
data — an inverse crime that flatters every method, reported only as a complementary view.

```python
run_benchmark(inst_with_matched_data, "neural")                           # InverseCrimeError
run_benchmark(inst_with_matched_data, "neural", allow_inverse_crime=True) # loud warning, "matched"
```

With `allow_inverse_crime=True` (CLI `--allow-inverse-crime`) the run proceeds, a warning banner is
logged, `res.regime[method] == "matched"` and every report header says
`⚠ MATCHED-OPERATOR (inverse crime allowed)`. The guard compares tags, so give every operator a
descriptive `fidelity_tag` ([tutorial 3](03_new_forward_operator.md)).

## Cumulative ablations

```python
from nefi.bench import (cumulative_ablation, drop_loss, no_annealing, no_gate,
                        no_positional_encoding, single_stage)

abl = cumulative_ablation(
    inst,
    [no_annealing(), no_positional_encoding(), single_stage()],   # row k applies modifiers 1..k
    n_samples=3, seeds=(0,), budget_scale=0.3, device="cpu",
)
print(abl.to_markdown())
```

```text
| Method | n | mse ↓ | psnr ↑ | relative_error ↓ | time (s) | peak mem (MB) | steps |
|:---|---:|---:|---:|---:|---:|---:|---:|
| full | 3 | 0.03312 ± 0.0425 | 21.51 ± 9.94 | 0.281 ± 0.257 | 1.036 ± 0.885 | — | 270 |
| −annealing | 3 | **0.01512 ± 0.0147** | **24.75 ± 9.2** | **0.1919 ± 0.155** | 1.007 ± 0.347 | — | 270 |
| −PE | 3 | 0.0581 ± 0.128 | 19.76 ± 13.7 | 0.3626 ± 0.495 | 1.673 ± 1.79 | — | 270 |
| −multiscale | 3 | 0.05603 ± 0.11 | 19.7 ± 13.8 | 0.3641 ± 0.465 | 1.302 ± 1.18 | — | 270 |
```

(At this tiny budget annealing costs accuracy on the easy toy — three samples with a single seed
give wide intervals; run paper-scale ablations on the paper instances.)

Rows share measurements and seeds, so differences are paired. A *modifier* changes one component
through up to three hooks — `config(cfg) -> overrides`, `problem(problem)`, `curriculum(cur)` —
and ready-made ones cover the generic components:

| modifier | effect | paper row |
|---|---|---|
| `no_annealing()` | β = K from step 0 in every stage | NeTMY −annealed PE, NeFTY FA |
| `no_positional_encoding()` | raw coordinates into the MLP | NeTMY −PE, NeFTY PE |
| `single_stage()` | one stage at the final resolution (total steps, first stage's lr) | NeTMY −multiscale |
| `drop_loss(name)` / `set_weight(name, w)` | weight 0 / w everywhere (incl. stage overrides) | NeTMY −ℓ1, −R_ds, −TV; NeFTY TV |
| `no_gate()` | `GatedSoftplus` → `Softplus` | NeTMY −gate |
| `grid_field(lr_scale=1)` | free-pixel field with the same heads | Grid Opt. / Tikhonov |
| `set_config(**kw)` | instance-config overrides (inversion side only) | NeFTY −HM: `set_config(face_mode="arithmetic")` |
| `set_stage(**kw)`, `set_curriculum(**kw)`, `set_optim(**kw)` | curriculum settings | — |
| `FnModifier(fn)` / any callable `fn(problem, curriculum)` | anything else | NeFTY −σ (swap the bounded head) |

The NeTMY Tab. 3 sequence on the NV instance is

```python
cumulative_ablation(nv, [drop_loss("tv"), no_annealing(), no_positional_encoding(), single_stage(),
                         drop_loss("l1"), no_gate(), drop_loss("direct_density")], ...)
```

NeFTY's table is additive (each row *adds* a component); list the removals in reverse order and
read the table bottom-up. `cumulative=False` gives one-at-a-time ablations; `base="grid"` ablates
a baseline instead. On the CLI, modifiers are strings: `no_annealing`, `no_pe`, `single_stage`,
`grid_field[:lr_scale]`, `no_gate`, `drop_loss:NAME`, `set_weight:NAME=W`, `set:KEY=VALUE`,
`stage:ATTR=VALUE`, `curriculum:ATTR=VALUE`, `optim:ATTR=VALUE`.

## One-axis sweeps

```python
from nefi.bench import sweep

sw = sweep(inst, "lr", [1e-3, 3e-3, 1e-2], n_samples=3, seeds=(0,), budget_scale=0.3, device="cpu")
```

```text
| Method | n | mse ↓ | psnr ↑ | relative_error ↓ | time (s) | peak mem (MB) | steps |
|:---|---:|---:|---:|---:|---:|---:|---:|
| lr=0.001 | 3 | 0.04591 ± 0.0364 | 19.86 ± 9.11 | 0.337 ± 0.265 | 0.8341 ± 0.326 | — | 270 |
| lr=0.003 | 3 | **0.03312 ± 0.0425** | 21.51 ± 9.94 | 0.281 ± 0.257 | 1.005 ± 0.648 | — | 270 |
| lr=0.01 | 3 | 0.03334 ± 0.0611 | **21.83 ± 12.7** | **0.2808 ± 0.336** | 0.8659 ± 0.108 | — | 270 |
```

Axes: any instance-config field (validated up front: `lr`, `hidden`, `n_octaves`, …, as in NeTMY
Tab. 10), `stage.<attr>` (every stage, e.g. `stage.anneal_fraction`), `curriculum.<attr>`
(`curriculum.restarts`), `optim.<attr>` (`optim.weight_decay`) and `weight.<loss>` (a loss weight).
Note that the `lr=0.003` row equals the `full` row of the ablation above: same data, same seeds,
same configuration — the harness is deterministic.

## Operator runtime (solver level)

```python
from nefi.bench import runtime_table

rt = runtime_table({"FFTConvolution (n=128)": problem}, n_repeat=20)
print(rt.to_markdown())
```

```text
| Operator | grid | fwd time (s) ↓ | bwd time (s) ↓ | peak mem (MB) ↓ |
|:---|---:|---:|---:|---:|
| FFTConvolution (n=128) | 128 | 5.1e-05 ± 1.8e-05 | 9.0e-05 ± 3.5e-05 | — |
```

Pass several entries (`InverseProblem`s or `(operator, fields)` pairs) to compare implementations
and `reference=` (an entry name or a tensor) for a relative simulation-error column — e.g. the
heat operator with `grad_mode="autograd"` vs `"adjoint"`, NeFTY Tab. 4 (autograd 18.63 GB vs
adjoint 21.9 MB peak memory at identical outputs).

## Throughput: batched solving and shards

Benchmarks are many independent solves of the same problem. Two protocol switches make them
cheaper without changing a single number beyond float rounding:

```python
res = run_benchmark(inst, "neural,grid", n_samples=16, seeds=(0, 1, 2),
                    batched=True, batch_size=16)          # nefi.batch_invert per method & class
part = run_benchmark(inst, "neural,grid", n_samples=16, seeds=(0, 1, 2),
                     shard=(0, 4), out_dir="runs/b")      # writes runs/b/shard-000-of-004.json
merged = BenchmarkResult.merge("runs/b")                  # after all 4 shards: the full table
```

* `batched=True` solves all (sample, seed) runs of a *batchable* method (FFT convolution, NV,
  Poisson, Born, holography, Radon, …; Adam(W), no callbacks) in one optimization loop
  (`nefi.batch_invert`); other methods run sequentially (logged). `time_s` is the batch wall-clock
  divided by the batch size; rows carry the `batch` size.
* `shard=(i, n)` runs the (class, sample, seed) units `k ≡ i (mod n)` — every method of a unit in
  the same shard, so comparisons stay paired — and `BenchmarkResult.merge` restores the unsharded
  table (canonical row order, protocol checks, missing shards reported).
* CLI: `nefi bench … --batched --batch-size 16 --shard 0/4` and `nefi bench-merge DIR`; SLURM
  arrays in [tutorial 7](07_running_on_gpu_servers.md); measurements in
  [performance](../performance.md).

## Reports and files

`res.save(dir)` writes `summary.md` (main table, per-class table when there are several classes,
efficiency table), `summary.csv` (aggregated), `rows.csv` (one row per run) and `benchmark.json`
(settings + rows; `BenchmarkResult.load(dir)` restores it). The writers (`markdown_table`,
`format_mean_ci`, `mean_ci`, `write_csv`, `write_json`, `to_jsonable`) are public for custom
reports.

## Paper-scale recipes

```bash
# NeTMY Tab. 1 (cross-fidelity; F3 data, 8 classes, 3 seeds) — on a GPU server
nefi bench configs/nv_relaxometry_paper.yaml --classes all --n 64 --seeds 0,1,2 --device cuda
# NeTMY Tab. 2 (matched F2/F2): generate data with the inversion operator, explicitly
nefi bench configs/nv_relaxometry_paper.yaml --set data_operator=F2 --allow-inverse-crime ...
# NeFTY Tab. 1
nefi bench thermal_tomography --set preset=paper --classes all --n 32 --seeds 0,1,2 --device cuda
```

See [tutorial 7](07_running_on_gpu_servers.md) for running these on the lab servers.


## Strata beyond scene classes

Every benchmark row also carries the scalar scene metadata of its measurement (`meta/<key>` and
`scene/<key>` columns, see `nefi.bench.protocol.scene_columns`), so tables can be stratified by any
of them — e.g. NeFTY Tab. 6 (robustness by defect count / layer count):

```python
res = run_benchmark(ThermalTomography(preset="paper"), methods=["nefty", "grid"], n_samples=32)
res.strata()                                    # ["scene/n_defects", "scene/n_layers", ...]
print(res.to_markdown(by="scene/n_defects"))    # mean ± CI per defect count and method
res.summary_csv(by="scene/n_layers")
```
