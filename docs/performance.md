# Performance

nefi solves one inverse problem by a few thousand optimization steps. Each step is a field
evaluation, a forward operator, a loss and a backward pass. The problems are small (10³–10⁵ grid
cells, 10⁴–10⁶ parameters), so a step is almost never limited by arithmetic. It is limited by
**per-operator overhead**: roughly 1–3 µs of Python and dispatcher time per aten op on a CPU, and
about 5 µs per kernel launch on a GPU. Performance work in nefi therefore means **fewer ops per
step**, **fewer steps or sweeps for the same accuracy**, and **more problems per launch**. This page
lists what is fast, what is slow, the switches (all of them opt-in with safe defaults), and how to
measure them.

## Measuring: `tools/profile_instance.py`

```bash
python tools/profile_instance.py thermal_tomography --config configs/thermal_tomography_smoke.yaml
python tools/profile_instance.py nv_relaxometry --config configs/nv_relaxometry_smoke.yaml \
    --device cuda --steps 100 --table --trace runs/prof     # + torch.profiler table / chrome trace
python tools/profile_instance.py thermal_tomography --config configs/thermal_tomography_smoke.yaml \
    --set solver=chebyshev --set jacobi_iters=13 --threads 1 --json runs/prof/tt.jsonl
python tools/profile_instance.py toy1d --compile step --inductor-threads 1   # compiled step
```

The profiler runs the real `Solver` loop on one curriculum stage for a fixed number of steps,
with no early stopping. It reports:

* **ms/step**: the median over the timed steps, after `--warmup` steps;
* **ops/step**: aten operators dispatched per step, covering forward, backward and optimizer, counted
  with a `TorchDispatchMode`. View and metadata ops are counted separately. On a GPU the non-view
  count is a good proxy for kernel launches. The count is skipped with `--compile`, because
  compiled kernels bypass the dispatcher.
* peak CUDA memory, and the `torch.profiler` table with `--table`.

`--set key=value` overrides instance config fields, and `--threads` sets `torch.set_num_threads`.
`--minimal` times the old stand-alone loop, without the solver's bookkeeping.

## Effect of the optimizations (CPU)

Measured with `tools/profile_instance.py` on a CPU with torch 2.11: the last curriculum stage, 20
timed steps, 4 threads and 1 thread. *Reference* is the implementation without the optimizations
described on this page (the `torch.roll` stencil, per-tensor Adam, uncached encodings); *current*
is the default. For `eit`, `wave_fwi` and `nv_relaxometry` the problems in the table are smaller
than the current smoke presets; run the profiler for the configuration you use.

| instance | grid | ms/step, reference (4 thr / 1 thr) | ms/step, current (4 thr / 1 thr) | ops/step, reference → current |
|---|---|---|---|---|
| toy1d | 64 | 0.41 / 0.41 | 0.41 / 0.41 | 180 → 120 |
| deconvolution | 32×32 | 1.05 / 1.22 | 1.03 / 1.27 | 201 → 141 |
| nv_relaxometry | 16×16 | 0.91 / 0.93 | 0.87 / 0.89 | 303 → 234 |
| thermal_tomography (Jacobi K=20) | 16×16×6, 30 frames | 21.2 / 20.7 | **9.4 / 9.3** | 15 416 → **9 302** |
| thermal_tomography, `solver=chebyshev`, K=13 (same accuracy) | 16×16×6, 30 frames | — | **7.7 / 7.4** | 15 416 → **7 170** |
| wave_fwi | 24×24 | 25.8 / 23.3 | 25.9 / 23.3 | 9 498 → 9 422 |
| eit | 16×16 | 5.2 / 5.4 | 5.2–5.7 | 3 500–3 900 (varies with PCG iterations) |

* **thermal_tomography** is 2.3× faster at unchanged numbers (the flat heat stencil, below), and
  2.75× faster with the opt-in Chebyshev solver at equal accuracy.
* **The three FFT instances** (toy1d, deconvolution, nv_relaxometry) run 25–35 % fewer ops, from
  the `foreach` optimizer and the encoding caches. The wall-clock barely moves, because on the CPU
  the Python and autograd overhead per op dominates these tiny steps. The op reduction pays off on a
  GPU, and **batching** is the real lever for these instances (see below).
* **wave_fwi** and **eit** are unchanged (see "What is still slow").

## Heat solver (NeFTY)

### Stencil: flat padded layout (default, same numbers)

Almost all of a heat step is the `2 × N_t × K` Jacobi sweeps of the adjoint forward and backward
solves. With the reference `torch.roll` stencil (`stencil_backend="roll"`) one sweep is 6
`torch.roll` calls plus 6 `addcmul` calls, which is **12 kernels**. The default keeps the iterate in
a *flat padded layout* (`nefi.operators.pde.stencil.FlatLayout`).
In that layout every neighbour is a contiguous 1-D slice at a constant offset. One gather refills
the periodic ghost cells, and then six in-place `addcmul_` run on preallocated buffers with
precreated views. That makes one sweep **7 kernels, with no views and no allocations**. The same
multiply-add chain runs in the same order, so results match the `torch.roll` reference to float
rounding.

| per application (3-D) | `roll` | `flat` (default) |
|---|---|---|
| Jacobi sweep | 12 kernels | **7** |
| `L(α) T`, `A T` | 13 | 9 |
| `R T` (`apply_offdiag`) | 12 | 8 |
| adjoint face products per step | 15 | 11 |
| physics step (smoke: forward + adjoint, 30 frames, K=20) | 15 160 | **9 177** (1.65× fewer) |
| … with `solver="chebyshev"`, K=13 (equal accuracy) | | **7 045** (2.15× fewer) |

Stencil throughput at the paper grid (64×64×16, 20 frames, K=50, forward + adjoint, CPU):

| threads | `roll` | `flat` | `flat` + Chebyshev K=20 |
|---|---|---|---|
| 1 | 268 ms | 197 ms | **90 ms** |
| 4 (default) | 454 ms | 246 ms | 122 ms |
| 10 | 858 ms | 458 ms | — |

`stencil_backend="auto" | "flat" | "roll"` selects the implementation. It is available on
`HeatOperator`, on `HeatSolveConfig`, and as the `ThermalTomographyConfig.stencil_backend` field.
`"auto"` uses the flat layout everywhere, except when autograd records through a sweep on an
accelerator with `torch.use_deterministic_algorithms(True)`. The gather's backward is an atomic
scatter-add on CUDA, so that case falls back to the deterministic `torch.roll` reference. The flat
layout also cuts the memory of the unrolled `grad_mode="autograd"` reference about 4.5×: the six
neighbour views share one padded buffer instead of six rolled copies.

### Inner solver: `solver="jacobi" | "chebyshev" | "cg"`, `inner_iters`

The inner solver is chosen with `HeatOperator(solver=..., inner_iters=...)`, or with the
`solver` and `jacobi_iters` instance fields.

* `"jacobi"` (default, the paper): `K` warm-started sweeps (NeFTY Eq. 24). `jacobi_iters: 50` in
  `configs/thermal_tomography_paper.yaml` reproduces Tab. 5.
* `"chebyshev"` (opt-in): Chebyshev semi-iterative acceleration of the same sweeps. It uses a
  Gershgorin bound `ρ = max_i Σ_j W_j[i]` and weights computed on the device without a host sync,
  and costs one extra `lerp` per iteration. The error after `K` iterations is at most
  `1/T_K(1/ρ)` of the initial error, against `ρ^K` for plain Jacobi. Measured iteration counts at
  equal accuracy:

  | setting | ρ | Jacobi | Chebyshev at the same error |
  |---|---|---|---|
  | smoke (Δt = 0.1, 16×16×6) | 0.57 | 20 | 12–13 |
  | paper time step / depth spacing (Δt = 0.05, Δz = 1/16) | 0.82 | 50 | **20** |

  At the paper setting this means **2.5× fewer sweeps**. It is not the default: it changes the
  iterates at the 1e-7 level, and the instance defaults are the paper's Tab. 5 by contract. Use
  `--set solver=chebyshev --set jacobi_iters=20` for paper-scale runs where time matters.
* `"cg"`: Jacobi-preconditioned CG with a relative tolerance. It has one host sync per iteration
  unless `cg_tol=0`, and about 17 kernels per iteration. Use it for verification and tight
  tolerances, not for speed.

### `compile_solver` / `compile_mode`

`compile_solver=True` (`HeatOperator(compile=True)`) runs the gradient-free sweeps through
`torch.compile`. Inductor fuses each sweep into about one kernel. The paper's 57.5 ms/step was
measured this way. `configs/thermal_tomography_paper.yaml` **now enables it by default**. It costs
a one-time compilation of roughly 10–40 s on CUDA at the first step. If compilation is
unavailable, it falls back to eager with a warning. The code default stays `False`, so CPU, MPS
and tests remain eager. `solver="chebyshev"` compiles too.

`compile_mode="max-autotune-no-cudagraphs"` is safe. `"reduce-overhead"` and `"max-autotune"`
capture CUDA graphs. In those modes the solve output is cloned out of the graph-owned buffer,
because the rollout keeps every state. This path is **untested on CUDA**.

### Gradient modes and memory

Saved tensors for a 32×32×16 grid, 20 steps and K=50, counted per unique storage:

| `grad_mode` | memory | scales as |
|---|---|---|
| `"adjoint"` (default; NeFTY Eq. 10–11) | 1.3 MB (21 grids) | `O(N_g N_t)` trajectory only |
| `"checkpoint"` (per time step) | 2.7 MB (43 grids) | `O(N_g N_t)` plus one step's graph |
| `"autograd"` (unrolled reference), `roll` | 378 MB (6 049 grids) | `O(K N_g N_t)` |
| `"autograd"` (unrolled reference), `flat` | 83 MB (1 332 grids) | `O(K N_g N_t)` |

The paper reports 21.9 MB (adjoint) against 18.63 GB (autograd) at the solver level.

## Solver switches

```python
nefi.Solver(problem, cur, compile="step", cuda_graphs=True, autocast="bf16")
# config files: solver: {compile: step, cuda_graphs: true, autocast: bf16}
```

### `compile=False | True | "field" | "step"` and `cuda_graphs`

* `False` (default): eager.
* `True` / `"field"`: `torch.compile` of the field evaluation.
* `"step"`: the field, the operator and the losses compile as one graph with `dynamic=False`.
  AOTAutograd compiles both forward and backward.

Operators with Python time loops, iterative solvers or custom `autograd.Function`s (heat, wave,
elliptic IFT, reaction–diffusion, `TimeStepper`) declare `Operator.traceable = False`. In `"step"`
mode they run eagerly between the compiled field graph and the compiled loss graph, which is a
deliberate graph break.

Mechanics and guarantees:

* **One compile per curriculum stage.** Shapes are fixed within a stage. The annealing `progress`
  is passed as a 0-dim tensor. A Python float would be specialized and recompiled at every
  annealing step; the previous `compile=True` did exactly that and fell back to eager after 8
  recompiles. `FourierFeatures` evaluates tensor progress without a host sync and bitwise equal to
  the float path.
* **A fresh code object per stage.** Dynamo caches graphs per code object and stops after 8, so
  restarts and benchmark runs no longer exhaust that cache.
* The first step of every stage runs eagerly. It fills the operators' caches (FFT kernels, initial
  states), so the graph never traces a cache miss.
* The step index is a Python int that would be specialized, so it is held constant inside the
  graph. A loss term that reads `ctx.step` must set `step_dependent = True`; its stage then
  compiles only the field.
* If compilation fails, the solver logs a warning and continues eagerly.
* `cuda_graphs=True` compiles with `mode="reduce-overhead"` (CUDA only). The step becomes about one
  graph launch instead of hundreds of kernel launches. A step's outputs are valid only until the
  next step, so callbacks must copy the tensors they keep.

On a CPU, compiled toy1d and NV steps are about 20 % faster, and deconvolution is slower because
inductor falls back on complex FFT ops. Compiled and eager runs agree to about 1e-7 relative
(`tests/test_performance.py`). Two limitations apply: runs whose fields or heads call
`float(progress)` (hash grid, level sets) graph-break there, and inductor's CPU backend needs a C++
compiler; if a compiled CPU run stops with an OpenMP runtime error, set
`torch._inductor.config.cpp.threads = 1` (`tools/profile_instance.py --inductor-threads 1`).

### `autocast=None | "bf16" | "fp16"`

`autocast` runs the field MLP trunk under `torch.autocast` on the problem's device. The output
projection of `NeuralField` stays in full precision, so the heads see an unquantized
pre-activation. The heads, the operator (the physics) and the losses run in the solver `dtype`.
`"fp16"` adds dynamic loss scaling with `torch.amp.GradScaler`; it is not available with L-BFGS.

The feature is meant for CUDA tensor cores. On the CPU used for these measurements, bf16 is
1.4–4.6× *slower*, because it has no fast bf16 GEMM path. Results stay close to fp32: toy1d fields agree to about 1e-3
relative. Chaotic problems (NV support masks) follow different trajectories, just as they do with
any rounding change.

### Optimizer and encoding micro-optimizations (default, bitwise identical)

* **Adam(W) `foreach` on small CPU models.** Models with at most 2²⁰ parameter elements on a CPU use
  the multi-tensor kernels. This makes the optimizer step about 30 % faster and gives bitwise
  identical updates. Larger models are memory-bound and keep the per-tensor path, and CUDA keeps
  PyTorch's default.
* **`FourierFeatures` caches.** The encoding memoizes `[sin, cos](2^k π x)` for its coordinate
  tensor and the band gates per `progress`. During annealing, one forward costs one multiply and one
  concatenation instead of all the transcendentals. Once annealing is done, the features come from
  the cache with 0 kernels. The caches are keyed on storage and version counter, and are bypassed
  when the coordinates require grad or are batched by `vmap`, and under `torch.compile`. Copies and
  `.to()` drop them.

## Throughput: batched multi-measurement solving

`nefi.batch_invert(problems, curriculum)` solves N independent problems in **one** optimization
loop. It works for problems that share a domain, a field architecture and a batchable operator,
for example one instance with several measurements and seeds.

* The field parameters are stacked into flat `(N, P)` groups.
* The fields are evaluated with `torch.func.vmap` over `functional_call`.
* The operator is called **once** on `(N, *shape)` fields.
* The losses reduce per problem. They use `vmap` over one shared `LossSet` when all loss sets are
  identical, which is verified structurally and numerically at every stage start. Otherwise each
  problem's own loss set is evaluated in a loop.
* The objective is `Σ_b L_b`. A per-problem Adam(W) reproduces `torch.optim`'s arithmetic, so the
  first update is bitwise identical.
* Gradient clipping, the NaN guard, EMA, early stopping and the discrepancy stop all act per
  problem. A stopped or failed problem is frozen, and the others continue.

Batched runs equal sequential runs to about 1e-7 per step. The difference comes only from batched
vs single matmul rounding, which Adam amplifies slightly. The tests check agreement to 1e-5 on
toy1d and deconvolution.

**Valid operators** (`Operator.batchable = True`, each verified by a test): `FFTConvolution` (toy1d,
deconvolution, plus `Nuisance` without trainable parameters), NV F1/F2, Poisson (DST), Born,
holography, the phase object, Radon, planar magnetostatics, and pointwise maps (`Identity`,
`Pointwise`, `Saturation`). `Sequential` and `Sum` are batchable when every inner operator is.

**Not batchable**, refused with a clear message:

* the heat adjoint, wave, elliptic IFT, reaction–diffusion and `TimeStepper`, because they use
  custom autograd functions or Python time loops;
* operators with trainable (nuisance) parameters;
* non-gradient baselines;
* losses that evaluate `ctx.field_module` (PINN-type collocation losses);
* optimizers other than Adam and AdamW.

Measured per-problem step times on CPU (4 threads):

| instance (smoke, last stage) | sequential | batch 4 | batch 16 | batch 64 |
|---|---|---|---|---|
| toy1d | 0.41 ms | 0.20 ms (2.1×) | 0.09 ms (4.4×) | **0.048 ms (8.4×)** |
| nv_relaxometry (per-problem loss loop) | 0.90 ms | 0.61 ms (1.5×) | 0.36 ms (2.5×) | 0.37 ms (2.5×) |
| deconvolution (compute-bound on CPU) | 1.31 ms | 0.80 ms (1.6×) | 0.74 ms (1.8×) | 0.66 ms (2.0×) |

A GPU is launch-bound for these sizes, so the batched speedup there should approach the batch size
until the device saturates. That is an estimate that needs a CUDA machine to confirm. Memory grows
linearly with the batch, so use `batch_size` to cap it.

In benchmarks: `run_benchmark(..., batched=True, batch_size=16)` or
`nefi bench <target> --batched --batch-size 16`. Methods that cannot be batched run sequentially,
and the fallback is logged. `time_s` of a batched row is the batch wall-clock divided by the batch
size, and each row records its `batch` size.

## Clusters: sharded benchmarks

```bash
nefi bench configs/deconvolution_full.yaml --n 32 --seeds 0,1,2 --batched --shard 0/8 --out runs/dc
...                                                                        # shards 1/8 … 7/8
nefi bench-merge runs/dc              # → runs/dc/{summary.md, summary.csv, rows.csv, benchmark.json}
```

`--shard i/n` (or `run_benchmark(shard=(i, n))`) runs a deterministic, balanced subset of the
(class, sample, seed) units. Unit `k` goes to shard `k mod n`, and all methods of a unit run in the
same shard, so comparisons stay paired. Each shard writes `shard-III-of-NNN.json`.

`nefi bench-merge DIR` (or `BenchmarkResult.merge(paths)`) checks that the shards come from the
same protocol, restores the canonical row order, and reproduces the unsharded table exactly
(tested). Missing shards are an error unless `--allow-missing` is given. A SLURM array example is
in `scripts/slurm_template.sbatch` and [tutorial 7](tutorials/07_running_on_gpu_servers.md).

## Threads

Small elementwise ops lose time to intra-op threading on a CPU. The heat solve at the paper grid is
**1.7× slower with 4 threads and 3.2× slower with 10 threads** than single-threaded with `torch.roll`.
The flat stencil reduces this to 1.25× and 2.3×, but single-threaded is still the fastest.

Guidance for CPU runs of the time-stepping instances (heat, wave, reaction–diffusion): use
`OMP_NUM_THREADS=1`, or `torch.set_num_threads(1)`, and run parallel **shards** (`--shard i/n`)
instead of intra-op threads. The FFT instances are neutral to the thread count at smoke size. GPU
runs are unaffected; the SLURM template sets `OMP_NUM_THREADS` for data loading and generation.

## What is still slow

* **wave_fwi** runs about 26 ops per leapfrog step (convolution Laplacian, PML auxiliary fields,
  source injection), over 120 steps per stage with backpropagation through time. The eager step has
  no single hot spot. Folding constants into `add(alpha=…)` and `addcmul(value=…)` would save
  roughly 20 % of the ops but change its numerics at the rounding level, so it is not applied. `compile="step"` keeps it eager (`traceable=False`), so it needs the operator's own
  compilation, which does not exist yet.
* **eit / darcy_flow** (elliptic PCG with an implicit-function adjoint) run 3.5–4 k ops and about
  110 host syncs per step, from the convergence tests. On CUDA, pass `check_every=8` to the
  elliptic operator or solve config to test convergence every 8 iterations. That gives 8× fewer
  syncs for at most 7 extra cheap iterations.
* **CPU autocast** is slower than fp32.

## Expected CUDA numbers (estimates, not measured)

These are estimated from launch counts, assuming about 5 µs of host time per eager kernel launch
and 1 fused kernel per compiled sweep. They must be validated on a CUDA machine.

| NeFTY paper setting (64×64×16, 100 frames, K=50) | kernels / step (heat) | est. time / step |
|---|---|---|
| eager, `roll` (before) | ≈ 120 k | 0.6–1 s |
| eager, `flat` (now default) | ≈ 70 k | 0.35–0.55 s |
| eager, `flat` + Chebyshev K=20 | ≈ 33 k | 0.17–0.25 s |
| `compile_solver=true` (paper config; the paper measured 57.5 ms incl. MLP) | ≈ 10 k fused | ≈ 50–60 ms |
| `compile_solver=true` + Chebyshev K=20 | ≈ 4 k fused | ≈ 25–30 ms |
| + CUDA graphs per solve (`compile_mode=reduce-overhead`, untested) | 200 graph launches | ≈ 15–25 ms |

The small FFT problems (NeTMY 64², deconvolution) are launch-bound with a few hundred kernels per
step. `batch_invert` with N problems costs about the same number of launches per step, so the
per-problem throughput gain should approach N. `compile="step"` fuses the elementwise chains, and
`cuda_graphs=True` removes the per-launch host cost.

## What needs a CUDA machine to validate

* the per-sweep speedups of the flat stencil and of Chebyshev on GPU, and the default
  `compile_solver=true` in the paper config, including compile time and the eager fallback;
* `compile="step"` and `cuda_graphs=True`, including the cudagraph-tree step marker, output
  lifetimes and memory;
* `autocast="bf16"/"fp16"` speed and accuracy on tensor cores, and the fp16 GradScaler path;
* the scaling of `batch_invert` throughput with the batch size, and its memory;
* heat `compile_mode="reduce-overhead"`, where outputs are cloned out of the graph pool;
* the `flat` backend with `torch.use_deterministic_algorithms(True)`: `"auto"` switches to `roll`
  when autograd records on CUDA.
