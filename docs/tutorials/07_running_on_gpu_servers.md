# 7 · Running on GPU servers

Smoke runs and tests fit on any CPU; paper-scale runs belong on a CUDA server. nefi is CUDA-first
but device-agnostic: `device="auto"` picks `$NEFI_DEVICE` if set, else CUDA if available, else
CPU (Apple MPS only when requested explicitly, since it lacks float64 and some FFT kernels). The workflow is: develop and smoke-test locally → sync code → run on
the server (directly or via SLURM) → pull back the small reports.

## 1. Sync the code

```bash
HOST=gpu-server scripts/sync_to_server.sh              # rsync to gpu-server:~/nefi (REMOTE_DIR=... to change)
HOST=gpu-server scripts/sync_to_server.sh --dry-run    # see what would be sent
HOST=gpu-server DELETE=1 scripts/sync_to_server.sh     # also delete remote files removed locally
```

Never sent: `runs/`, `outputs/`, `.git/`, caches, virtualenvs, build artefacts and tensors
(`*.pt`, `*.npz`, …). `HOST` is any ssh alias from `~/.ssh/config`. (Alternatively `git clone`
on the server.)

## 2. Environment (once)

```bash
ssh gpu-server
cd ~/nefi
conda env create -f environment.yml && conda activate nefi     # python 3.11, torch, nefi[dev]
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nefi list
```

The pip `torch` wheel on Linux bundles its CUDA runtime; to pin a CUDA version install torch from
the matching index (`pip install torch --index-url https://download.pytorch.org/whl/cu121`) before
`pip install -e ".[dev]"`. `requirements.txt` lists the same dependencies for plain virtualenvs.

## 3. Run

```bash
scripts/run_server.sh configs/nv_relaxometry_paper.yaml                  # nefi run … --device cuda
scripts/run_server.sh thermal_tomography --set preset=paper --scene layered --seed 1
NEFI_CMD=bench scripts/run_server.sh configs/toy1d.yaml --n 16 --seeds 0,1,2
CUDA_VISIBLE_DEVICES=1 SKIP_INSTALL=1 scripts/run_server.sh deconvolution --plot
```

`run_server.sh` activates the conda env `nefi` (creating it from `environment.yml` if missing; set
`VENV=/path` for a virtualenv), installs the package (skip with `SKIP_INSTALL=1`), prints the
torch/CUDA versions and runs `nefi $NEFI_CMD <target> --device cuda --out
runs/<target>-<cmd>-<timestamp>`, teeing the log to `<out>.log`. Everything after the target is
passed to nefi. For long jobs without SLURM use `tmux`/`nohup`; nefi runs are independent
per measurement, so several GPUs are used by launching several jobs with different
`CUDA_VISIBLE_DEVICES`.

## 4. SLURM

```bash
mkdir -p runs/slurm                                      # SLURM does not create the log directory
sbatch --export=ALL,TARGET=configs/nv_relaxometry_paper.yaml scripts/slurm_template.sbatch
# three seeds as an array job (seed = array index)
sbatch --array=0-2 --export=ALL,TARGET=thermal_tomography,NEFI_ARGS="--set preset=paper" \
       scripts/slurm_template.sbatch
# a whole benchmark in one job (values with commas cannot go inside --export=...: export them)
export NEFI_CMD=bench TARGET=configs/toy1d.yaml NEFI_ARGS="--n 16 --seeds 0,1,2"
sbatch --export=ALL scripts/slurm_template.sbatch
```

The template requests 1 GPU, 8 CPUs, 32 GB and 12 h (edit partition/account for your cluster),
activates the conda env, sets `NEFI_DEVICE=cuda`, `OMP_NUM_THREADS` and unbuffered output, prints
the GPU, and writes to `runs/slurm/<target>-<cmd>-<jobid>-s<seed>`.

### Large benchmarks: shards as an array job, then merge

A benchmark is `classes × samples × seeds` independent (class, sample, seed) units, each solved
by every method. `nefi bench --shard i/n` runs the units `k ≡ i (mod n)`, so the shards are
balanced and every method of a unit runs in the same job. Each shard writes
`shard-III-of-NNN.json`, and `nefi bench-merge DIR` concatenates the shards into the table the
unsharded run would have produced (`BenchmarkResult.merge` in Python):

```bash
export NEFI_CMD=bench NEFI_SHARDS=8 TARGET=configs/deconvolution_full.yaml
export NEFI_ARGS="--n 32 --seeds 0,1,2 --batched"
JOB=$(sbatch --parsable --array=0-7 --export=ALL scripts/slurm_template.sbatch)
DIR=runs/slurm/deconvolution_full-bench-${JOB}      # all shards land here (array job id)
sbatch --dependency=afterok:${JOB} --export=ALL,NEFI_CMD=bench-merge,TARGET=${DIR} \
       scripts/slurm_template.sbatch
# or by hand once the array is done:  nefi bench-merge runs/slurm/deconvolution_full-bench-<id>
```

With `NEFI_SHARDS` set, the template passes `--shard ${SLURM_ARRAY_TASK_ID}/${NEFI_SHARDS}` and
writes every task into `runs/slurm/<target>-bench-<array job id>`. `--batched` solves all runs of
a batchable method (FFT convolution, NV, deconvolution, Poisson, Born, holography, Radon, …) in one
optimization loop (`nefi.batch_invert`), which fills a GPU that single small problems leave
launch-bound. The heat, wave and elliptic operators are not batchable and run one problem at a
time; for them, shard across array tasks or GPUs instead.

## 5. Bring results back

```bash
HOST=gpu-server PULL=1 scripts/sync_to_server.sh        # → runs/remote-gpu-server/
```

Pull mode copies only reports — `*.md`, `*.json`, `*.csv`, `*.yaml`, `*.png`, `*.log`,
`*.out` — never `result.pt` or checkpoints, so the local copy stays small. Every run directory is
self-describing: `config.yaml` is the effective configuration (`nefi run runs/<dir>/config.yaml`
reproduces the run), `metrics.json` has metrics, timings and the result's `config_hash`, and
benchmark directories have `summary.md` / `rows.csv` / `benchmark.json`.

## Time and memory expectations

Measured by the papers (single GPU, full budgets):

| workload | settings | time | memory |
|---|---|---|---|
| NeTMY, per measurement | 64², 2 stages, 10k steps, 4.4 × 10⁵ params (A6000) | median 273 s, trimmed mean 426 s, mean 780 s (dense many/close scenes take longest) | small (FFT operators) |
| NeTMY free-density baselines | Tikhonov / ADMM | 1.5 s / 1.1 s | — |
| NeTMY GaussianSplat / DeepDecoder | | 402 s / 259 s | — |
| NeFTY, per specimen | 64×64×16, 100 frames, 10k steps, 2.4 M params (RTX PRO 6000) | 9.6 min (57.5 ms/step, incl. ≈ 11 s `torch.compile`) | 4.3 GB peak |
| NeFTY Grid Opt. | same solver | 8.0 min | 1.2 GB |
| NeFTY solver level (50 steps) | discrete adjoint vs autograd | fwd 0.46 s / bwd 0.50 s vs 1.43 s / 1.30 s | 21.9 MB vs 18.63 GB |

Before launching a campaign, measure your own problem:

```bash
nefi bench <target> --n 1 --seeds 0 --budget-scale 0.02 --device cuda    # ms/step, peak memory
python tools/profile_instance.py <instance> --device cuda --steps 100     # per-step profile
```

and extrapolate linearly in steps (`efficiency_table()` reports ms/step and peak memory);
`nefi.bench.runtime_table` times the operator alone (forward/backward, peak memory). Memory of
time-stepping operators grows with the number of steps unless they use an adjoint
(`HeatOperator(grad_mode="adjoint")`) or checkpointing (`TimeStepper(grad_mode="checkpoint")`).

## Performance knobs

See [docs/performance.md](../performance.md) for measurements, the memory table and CUDA estimates.

* **`torch.compile`**: set it with `Solver(compile="field" | "step")`,
  `solver: {compile: step}` in a config file, or `--set solver.compile=step`. `"field"` compiles
  the field; `"step"` compiles field → operator → losses (forward and backward) as one graph, once
  per curriculum stage (annealing does not recompile). The first step of each stage runs eagerly
  to fill caches. It pays a one-time compile of seconds to minutes and helps on long CUDA runs.
  `compile=True` still means `"field"`. Add `cuda_graphs=True` (`--set solver.cuda_graphs=true`)
  for `mode="reduce-overhead"`, which replays each step as one CUDA graph. Leave all of this off
  for tests, CPU and MPS. If compilation fails, nefi logs a warning and runs eagerly.
* **Heat solver (NeFTY)**: `configs/thermal_tomography_paper.yaml` now sets
  `compile_solver: true`, so the Jacobi sweeps are fused (paper setting; one-time compile of
  roughly 10–40 s). `--set solver=chebyshev --set jacobi_iters=20` reaches the accuracy of the
  paper's 50 Jacobi sweeps with about 2.5× fewer sweeps (opt-in).
* **Mixed precision**: `Solver(autocast="bf16")` or `--set solver.autocast=bf16` runs the field
  MLP on tensor cores while the physics stays fp32. `"fp16"` adds loss scaling. This is a CUDA
  feature; on a CPU bf16 is slower.
* **Throughput**: `nefi bench --batched [--batch-size N]` and `nefi.batch_invert(problems)`
  solve many measurements and seeds of a batchable problem in one loop. `--shard i/n` together
  with `nefi bench-merge` splits a benchmark over SLURM array tasks (above).
* **dtype**: float32 by default; `--set solver.dtype=float64` for stiff physics (not on MPS).
  Data generators simulate in float64 on the CPU by default.
* **Budgets** — the curriculum dominates run time; `--budget-scale` and
  `Curriculum.scaled(f)` scale every stage, `Curriculum(time_budget_s=...)` stops at a wall-clock
  budget, `early_stop_patience` / `discrepancy_tau` stop when the data stop improving or the noise
  level is reached ([tutorial 8](08_robustness_and_adaptivity.md)).
* **Threads**: on shared CPU nodes set `OMP_NUM_THREADS`; the SLURM template does. CPU runs of
  the time-stepping instances (heat, wave) are fastest single-threaded. For those, use
  `OMP_NUM_THREADS=1` and parallel shards rather than intra-op threads.
* **Profile before optimizing**: run
  `python tools/profile_instance.py <instance> --config configs/<instance>_smoke.yaml --device cuda`.
  It reports ms/step, ops/step (≈ kernel launches) and the top ops; add `--table` / `--trace`.

## Checkpoints and resuming

```python
from nefi.solve import CheckpointCallback

ckpt = CheckpointCallback("runs/nv-long/ckpt", every=500)   # ckpt_step500.pt, …, ckpt_stage0.pt
result = nefi.Solver(problem, cur, device="cuda", seed=0, callbacks=[ckpt]).run()
```

Each checkpoint stores the field and operator `state_dict`s and the history. Stage boundaries are
exact resume points, because the solver builds a fresh optimizer and resets the annealing at every
stage start:

```python
state = torch.load("runs/nv-long/ckpt/ckpt_stage0.pt", map_location="cpu")
problem = inst.build_problem(meas)                # same config, same measurement (same data seed)
problem.field.load_state_dict(state["field"])
problem.operator.load_state_dict(state["operator"])
remaining = nefi.Curriculum(cur.stages[1:], optim=cur.optim)
result = nefi.Solver(problem, remaining, device="cuda", seed=0).run()
```

(Mid-stage checkpoints restore the parameters but not the optimizer moments or the position in
the learning-rate schedule; for a `GridField`, call `field.on_stage_start(stage, domain)` with the
checkpoint's stage first so the parameter shapes match.)

## Reproducibility

Runs are seeded (`--seed`; benchmarks seed before building every problem), configurations are saved
next to results and hashed (`Result.config_hash`), and data generation is deterministic per data
seed. GPU kernels are not bitwise deterministic by default; for exact reproduction call
`nefi.utils.seed_everything(seed, deterministic=True)` (slower).
