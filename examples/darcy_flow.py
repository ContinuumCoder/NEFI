"""Darcy flow: generate → invert → evaluate → PNG.

Recovers a log-permeability field from sparse steady pressure gauges recorded for several
injector/producer well configurations (no-flow reservoir, harmonic-mean finite volumes,
pure-Neumann PCG, implicit-function adjoint), and compares the neural field with the ``grid``
baseline and the uniform initial guess.

Usage::

    python examples/darcy_flow.py --config configs/darcy_flow_smoke.yaml
    python examples/darcy_flow.py --config configs/darcy_flow_full.yaml --device cuda

Outputs ``runs/<name>/darcy_flow.png`` and ``runs/<name>/metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import nefi
from nefi.config import load_config
from nefi.registry import build
from nefi.solve.result import Result
from nefi.utils.seed import seed_everything


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/darcy_flow_smoke.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--scene", default=None, help="smooth | channels")
    ap.add_argument("--methods", default=None, help="comma-separated: neural,grid")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    run = cfg.get("run", {})
    inst = build("instance", cfg["instance"])
    seed = run.get("seed", 0) if args.seed is None else args.seed
    device = args.device or run.get("device", "auto")
    methods = (args.methods or ",".join(run.get("methods", ["neural", "grid"]))).split(",")
    out_dir = Path(args.out or run.get("out", "runs/darcy_flow"))
    out_dir.mkdir(parents=True, exist_ok=True)

    gt, meas = inst.make_measurement(seed, args.scene)
    c = inst.cfg
    print(f"Darcy {c.n}² grid, {c.n_configs} well configurations, scene {c.scene!r}")
    uni = torch.full_like(gt["log_k"], c.log_k_mean)
    results = {"uniform": (Result({"log_k": uni}, {"log_k": uni}, torch.zeros(1), {}), 0.0)}
    for m in methods:
        seed_everything(seed)  # the field is initialized when the problem is built
        if m in ("neural", "nefi"):
            problem = inst.build_problem(meas)
            cur = problem.curriculum
        else:
            problem, cur = inst.baselines()[m](meas)
        t0 = time.perf_counter()
        res = nefi.invert(problem, cur, device=device, seed=seed)
        results[m] = (res, time.perf_counter() - t0)

    report = {}
    for m, (res, secs) in results.items():
        metrics = inst.evaluate(res, gt)
        report[m] = {**metrics, "time_s": secs}
        vals = "  ".join(f"{k}={v:.4g}" for k, v in metrics.items())
        print(f"  {m:>8s}: {vals}  ({secs:.1f}s)")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("matplotlib not installed: skipping the figure")
        return report
    shown = [m for m in results if m != "uniform"]
    ncol = 2 + len(shown)
    fig, ax = plt.subplots(1, ncol, figsize=(3.6 * ncol, 3.4))
    ext = [e for pair in inst.domain().extent for e in pair]
    vmin, vmax = float(gt["log_k"].min()), float(gt["log_k"].max())
    panels = [("ground truth log k", gt["log_k"])]
    panels += [(m, results[m][0].fields["log_k"]) for m in shown]
    xy = inst.domain().physical_coords()
    sensors = meas.mask[0] > 0
    for a, (title, img) in zip(ax, panels):
        im = a.imshow(img.T, origin="lower", extent=ext, vmin=vmin, vmax=vmax, cmap="viridis")
        psnr = report.get(title, {}).get("psnr")
        a.set_title(title if psnr is None else f"{title} (PSNR {psnr:.1f} dB)")
        a.plot(xy[..., 0][sensors], xy[..., 1][sensors], "w.", ms=3)
        for config in inst.wells():
            for x, y, q in config:
                a.plot(x, y, "r^" if q > 0 else "kv", ms=4)
    fig.colorbar(im, ax=ax[: len(panels)], shrink=0.8)
    a = ax[-1]
    for m in shown:
        a.semilogy(results[m][0].history["data"], label=m)
    a.set_title("data loss")
    a.set_xlabel("step")
    a.legend(fontsize=7)
    fig.savefig(out_dir / "darcy_flow.png", dpi=110, bbox_inches="tight")
    print(f"wrote {out_dir / 'darcy_flow.png'}")
    return report


if __name__ == "__main__":
    main()
