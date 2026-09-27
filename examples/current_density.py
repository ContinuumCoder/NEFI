"""NV-magnetometry current imaging: generate → invert → evaluate → PNG.

Recovers a divergence-free sheet current (stream-function parameterization) from its stray field
``B_z`` at standoff ``z0``. The data are simulated by direct Biot–Savart summation on a finer grid
(float64); the inversion uses the zero-padded FFT operator. The neural field is compared with the
``grid`` baseline and the classical regularized Fourier inversion (``fourier``).

Usage::

    python examples/current_density.py --config configs/current_density_smoke.yaml
    python examples/current_density.py --config configs/current_density_full.yaml --device cuda

Outputs ``runs/<name>/current_density.png`` and ``runs/<name>/metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import nefi
from nefi.config import load_config
from nefi.physics.magnetostatics import stream_to_current
from nefi.registry import build
from nefi.solve.result import Result
from nefi.utils.seed import seed_everything


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/current_density_smoke.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--scene", default=None, help="wires | loops | branching")
    ap.add_argument("--methods", default=None, help="comma-separated: neural,grid,fourier")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    run = cfg.get("run", {})
    inst = build("instance", cfg["instance"])
    seed = run.get("seed", 0) if args.seed is None else args.seed
    device = args.device or run.get("device", "auto")
    default_methods = ["neural", "grid", "fourier"]
    methods = (args.methods or ",".join(run.get("methods", default_methods))).split(",")
    out_dir = Path(args.out or run.get("out", "runs/current_density"))
    out_dir.mkdir(parents=True, exist_ok=True)

    gt, meas = inst.make_measurement(seed, args.scene)
    c = inst.cfg
    print(
        f"NV current imaging {c.n}² px ({c.pixel} {c.length_unit}), z0 = {c.z0} {c.length_unit}, "
        f"scene {c.scene!r}"
    )
    zero = torch.zeros_like(gt["g"])
    results = {"zero": (Result({"g": zero}, {"g": zero}, torch.zeros_like(meas.data), {}), 0.0)}
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
    sp = inst.domain().spacing()

    def jmag(g):
        return torch.stack(stream_to_current(g.double(), sp)).norm(dim=0)

    shown = [m for m in results if m != "zero"]
    ncol = 2 + len(shown)
    fig, ax = plt.subplots(1, ncol, figsize=(3.5 * ncol, 3.3))
    ext = [e for pair in inst.domain().extent for e in pair]
    vmax = float(jmag(gt["g"]).max())
    panels = [("|K| ground truth", gt["g"])] + [(m, results[m][0].fields["g"]) for m in shown]
    for a, (title, g) in zip(ax, panels):
        a.imshow(jmag(g).T, origin="lower", extent=ext, vmin=0, vmax=vmax, cmap="magma")
        p = report.get(title, {}).get("j_psnr")
        a.set_title(title if p is None else f"|K| {title} ({p:.1f} dB)")
    im = ax[-1].imshow(meas.data.T, origin="lower", extent=ext, cmap="RdBu_r")
    ax[-1].set_title(f"measured B_z [{c.field_unit}]")
    fig.colorbar(im, ax=ax[-1], shrink=0.8)
    fig.savefig(out_dir / "current_density.png", dpi=110, bbox_inches="tight")
    print(f"wrote {out_dir / 'current_density.png'}")
    return report


if __name__ == "__main__":
    main()
