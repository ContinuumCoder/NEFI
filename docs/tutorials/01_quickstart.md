# 1 · Quickstart: toy1d end to end

`toy1d` is the five-second instance: recover a non-negative 1-D signal `x(t)` on `[0, 1]` from its
Gaussian-blurred, noisy observation. It exercises every core mechanism — neural field with
annealed Fourier features, softplus head, FFT convolution rebuilt per resolution, a two-stage
multiscale curriculum, TV regularization, and an inverse-crime-safe data generator (the data are
simulated on a 4× finer grid in float64, then area-averaged).

## Install

```bash
pip install "nefi[viz]"        # nefi + matplotlib; a CPU is enough for everything in this tutorial
nefi --version
```

For development (tests, examples, the configs under `configs/`), install from source:
`git clone https://github.com/ContinuumCoder/NEFI.git && cd NEFI && pip install -e ".[dev]"`.

## From Python

```python
import nefi
from nefi.instances.toy1d import make_problem

problem, gt, meas = make_problem(n=128, seed=0)   # (InverseProblem, ground truth, Measurement)
result = nefi.invert(problem, device="cpu", seed=0)
print(result.summary())
print("PSNR", nefi.metrics.psnr(result.fields["x"], gt["x"]))
```

Output (CPU):

```text
Result: fields={'x': (128,)}
  total time 4.1s, steps 900
  stage 0 (stage1, (64,)): 300 steps, 0.9s | data=0.00636, tv=15.8
  stage 1 (stage2, (128,)): 600 steps, 2.1s | data=0.00108, tv=7.76
PSNR 28.15
```

What happened:

* `make_problem` sampled a "bumps" scene, simulated the measurement with the *independent* data
  generator, and built the `InverseProblem` (neural field + blur operator + MSE/TV losses).
* `nefi.invert` ran the instance's default curriculum: 300 steps on a 64-point grid, then 600 on
  the native 128-point grid at half the learning rate, re-annealing the Fourier bands (β = 0 → K)
  in each stage.
* `result.fields` holds the reconstruction, `result.pred` the operator applied to it,
  `result.history` one entry per step (`total`, `data_loss`, every loss term, `lr`, `progress`,
  `stage`) and `result.stage_results` a per-stage summary.

`problem.describe()` summarizes the pieces, and `result.save("r.pt")` / `nefi.Result.load("r.pt")`
round-trip the result.

## From the command line

```bash
nefi run toy1d
```

```text
nefi run toy1d · neural · seed 0 · scene bumps · device cpu
  stage1 (64,): 300 steps, completed, 0.46s
  stage2 (128,): 600 steps, completed, 1.40s
  total 2.51s · 14273 parameters
  metrics  mse 0.008576 · psnr 27.99 · relative_error 0.1116
  data fit PSNR 38.32 dB · RMSE/σ 1.244
  saved    runs/toy1d-<hash>  (result.pt config.yaml metrics.json history.csv)
```

The output directory contains everything needed to reproduce and analyse the run:

| file | content |
|---|---|
| `result.pt` | `nefi.Result` (fields, prediction, history, timings) |
| `config.yaml` | the *effective* configuration (instance config + curriculum + seed); `nefi run runs/<dir>/config.yaml` re-runs it |
| `metrics.json` | instance metrics, data-fit summary, timings, stage results |
| `history.csv` | per-step loss components, learning rate, annealing progress |
| `fields.png`, `history.png` | with `--plot` |

Useful options: `--scene spikes` (scene class), `--seed 3`, `--device cuda`, `--set n=256 --set
lr=1e-3` (config overrides), `--set stage.anneal_fraction=0.5` (curriculum overrides),
`--baseline grid` (run a baseline instead), `--smoke` (tiny budget), `--out DIR`. The data-fit
line reports `RMSE/σ`: ≈ 1 means the measurement is explained down to its noise level (the Morozov
target, see [tutorial 8](08_robustness_and_adaptivity.md)).

## Compare with a baseline

```bash
nefi bench toy1d --methods neural,grid --n 4 --seeds 0,1
```

```text
**toy1d** — cross-fidelity (data: `gaussian-blur-4x-float64`; inversion: `FFTConvolution`) · 4 samples × 2 seeds × 1 class · mean ± 95% CI (Student t over samples × seeds)

| Method | n | mse ↓ | psnr ↑ | relative_error ↓ | time (s) | peak mem (MB) | steps |
|:---|---:|---:|---:|---:|---:|---:|---:|
| neural | 8 | 0.007331 ± 6.1e-03 | 28.62 ± 4.67 | 0.119 ± 0.0687 | 4.426 ± 0.332 | — | 900 |
| grid | 8 | **0.003632 ± 9.9e-04** | **29.29 ± 1.33** | **0.09171 ± 0.0136** | 0.8858 ± 0.0934 | — | 900 |
```

Both methods saw the same four measurements with two optimization seeds each; the table reports
mean ± 95 % confidence half-width (Student t over samples × seeds) and the header states the regime
(cross-fidelity: the data model differs from the inversion model). On this smooth, mildly blurred
1-D toy a free grid (the Tikhonov / Grid-Opt. baseline) is as good as the neural field — the
neural prior pays off when the problem is severely ill-posed and the free-pixel gradient is
misleading (centered collapse under the NV operator, ringing in 3-D heat conduction). The
[diagnostics](05_diagnostics.md) show *why*, and the [paper instances](../instances/index.md)
show the gap.

## Diagnose

```bash
nefi diagnose toy1d        # or: nefi diagnose toy1d --solve  (linearize at the solution)
```

prints a table of ill-posedness diagnostics with the papers' reference numbers (sensitivity
center/outer ratio, iter-0 gradient signature, realized first update vs raw gradient, filter-kernel
width, encoding bandwidth, singular values, energy barrier) — the subject of
[tutorial 5](05_diagnostics.md).

## Your own forward model in three lines

If you already have a differentiable forward model, `nefi.from_forward` builds a sensible problem
(domain, neural field, heads from a prior DSL, noise-aware data loss, balanced weights, curriculum,
discrepancy stopping) and records every automatic decision in `problem.meta["auto"]`:

```python
problem = nefi.from_forward(forward, y, shape=(128, 128), prior="nonnegative + sparse")
result = nefi.invert(problem)
print(nefi.quick_report(result, problem))
```

For full control — multiscale operators, independent data generators, benchmarks — continue with
[the anatomy of a problem](02_anatomy_of_a_problem.md) and
[writing a forward operator](03_new_forward_operator.md).
