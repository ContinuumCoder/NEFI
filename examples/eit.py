"""Electrical impedance tomography: generate → invert → evaluate → PNG.

Recovers a conductivity map from the boundary potentials of trigonometric current patterns
(continuum EIT model, harmonic-mean finite volumes, implicit-function adjoint) with a neural field,
and compares it with the free-pixel ``grid`` baseline and the uniform initial guess.

Usage::

    python examples/eit.py --config configs/eit_smoke.yaml          # seconds on CPU
    python examples/eit.py --config configs/eit_full.yaml --device cuda

Outputs ``runs/<name>/eit.png`` and ``runs/<name>/metrics.json``.
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


def _uniform_result(inst, gt) -> Result:
    sigma = torch.full_like(gt["sigma"], inst.cfg.sigma_bg)
    return Result({"sigma": sigma}, {"sigma": sigma}, torch.zeros(1), {})


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/eit_smoke.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--scene", default=None, help="single | multi | contrast")
    ap.add_argument("--methods", default=None, help="comma-separated: neural,grid")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    run = cfg.get("run", {})
    inst = build("instance", cfg["instance"])
    seed = run.get("seed", 0) if args.seed is None else args.seed
    device = args.device or run.get("device", "auto")
    methods = (args.methods or ",".join(run.get("methods", ["neural", "grid"]))).split(",")
    out_dir = Path(args.out or run.get("out", "runs/eit"))
    out_dir.mkdir(parents=True, exist_ok=True)

    gt, meas = inst.make_measurement(seed, args.scene)
    print(f"EIT {inst.cfg.n}² grid, {inst.cfg.n_patterns} patterns, scene {inst.cfg.scene!r}")
    results = {"uniform": (_uniform_result(inst, gt), 0.0)}
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
    lo, hi = float(gt["sigma"].min()), float(gt["sigma"].max())
    norm = matplotlib.colors.LogNorm(vmin=min(lo, 0.9), vmax=max(hi, 1.1))
    panels = [("ground truth σ", gt["sigma"])] + [(m, results[m][0].fields["sigma"]) for m in shown]
    for a, (title, img) in zip(ax, panels):
        im = a.imshow(img.T, origin="lower", extent=ext, norm=norm, cmap="RdBu_r")
        psnr = report.get(title, {}).get("psnr")
        a.set_title(title if psnr is None else f"{title} (PSNR {psnr:.1f} dB)")
    fig.colorbar(im, ax=ax[: len(panels)], shrink=0.8)
    # boundary data of pattern 0 along the perimeter (observed vs prediction), arc-length order
    faces = inst.operator().obs_boundary.faces
    fi, fj = torch.as_tensor(faces["i"]), torch.as_tensor(faces["j"])
    s_arc = faces["s0"] + 0.5 * faces["length"]
    a = ax[-1]
    a.plot(s_arc, meas.data[0][fi, fj].numpy(), ".", ms=3, label="observed")
    for m in shown:
        a.plot(s_arc, results[m][0].pred[0][fi, fj].numpy(), "-", lw=1, label=m)
    a.set_title("pattern 0: boundary potential")
    a.set_xlabel("arc length s")
    a.legend(fontsize=7)
    fig.savefig(out_dir / "eit.png", dpi=110, bbox_inches="tight")
    print(f"wrote {out_dir / 'eit.png'}")
    return report


if __name__ == "__main__":
    main()
