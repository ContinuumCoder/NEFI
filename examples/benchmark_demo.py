"""Benchmark protocol tour on toy1d: paired benchmark, cumulative ablation, sweep, runtime.

Mirrors the evaluation protocol of the papers at toy scale (≈1 min on a CPU)::

    python examples/benchmark_demo.py [--out runs/benchmark_demo] [--budget 0.3]

1. ``run_benchmark`` — neural field vs free grid on the same measurements, 3 samples × 2 seeds,
   mean ± 95% CI (Student t), runtime columns (NeTMY Tab. 1 / NeFTY Tab. 1 protocol);
2. ``cumulative_ablation`` — remove annealing, then positional encoding, then multiscale
   (NeTMY Tab. 3 style: each row removes one more component);
3. ``sweep`` — one-axis learning-rate sweep (NeTMY Tab. 10);
4. ``runtime_table`` — forward/backward time of the operator (NeFTY Tab. 4 style).

Every table is written as markdown / CSV / JSON under ``--out``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nefi.bench import (
    cumulative_ablation,
    no_annealing,
    no_positional_encoding,
    run_benchmark,
    runtime_table,
    single_stage,
    sweep,
)
from nefi.instances.toy1d import Toy1D


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="runs/benchmark_demo")
    ap.add_argument("--budget", type=float, default=0.3, help="scale of the step budget")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = Path(args.out)
    inst = Toy1D(n=128, scene="mixed")
    common = {"device": args.device, "budget_scale": args.budget, "progress": True}

    print("\n## 1. Neural field vs grid (cross-fidelity: 4x-supersampled float64 data)\n")
    bench = run_benchmark(
        inst, "neural,grid", n_samples=3, seeds=(0, 1), out_dir=out / "bench", **common
    )
    print(bench.to_markdown())
    print(bench.efficiency_table())

    print("\n## 2. Cumulative ablation (each row removes one more component)\n")
    abl = cumulative_ablation(
        inst,
        [no_annealing(), no_positional_encoding(), single_stage()],
        n_samples=3,
        seeds=(0,),
        out_dir=out / "ablation",
        **common,
    )
    print(abl.to_markdown())

    print("\n## 3. Learning-rate sweep\n")
    sw = sweep(
        inst, "lr", [1e-3, 3e-3, 1e-2], n_samples=3, seeds=(0,), out_dir=out / "sweep", **common
    )
    print(sw.to_markdown())

    print("\n## 4. Operator runtime (forward / backward, current field)\n")
    _, meas = inst.make_measurement(0)
    problem = inst.build_problem(meas)
    rt = runtime_table({"FFTConvolution (n=128)": problem}, n_repeat=20)
    print(rt.to_markdown())
    (out / "runtime.md").write_text(rt.to_markdown())
    print(f"reports written under {out}/")


if __name__ == "__main__":
    main()
