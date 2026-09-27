"""Neural field vs. the baseline family on ``deconvolution/sparse_dots`` at a tiny budget.

Every method optimizes the *same* objective (MSE + the shared regularizers) through the *same*
blur operator on the *same* measurement — only the parameterization / solver differs, which is
exactly the NeTMY §4.4 filtering view: a grid executes the raw pixel gradient (``G_θ = I``), the
deep decoder filters it with a global low-pass kernel, Gaussian splats with a low-rank localized
kernel, the coordinate MLP with a smooth, annealed kernel. ADMM (ℓ1 + box prox) and the Wiener
filter are shown as classical references.

Usage::

    python examples/baselines_comparison.py                           # ~20 s on a CPU
    python examples/baselines_comparison.py --set n=64 --budget 3 --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from nefi.baselines import solve  # noqa: E402
from nefi.config import load_config  # noqa: E402
from nefi.instances.deconvolution import Deconvolution  # noqa: E402
from nefi.metrics import evaluate  # noqa: E402
from nefi.utils.seed import seed_everything  # noqa: E402

METHODS = ("neural", "grid", "deep_decoder", "gaussian_splat", "admm", "wiener")


def load(path: str, overrides: list[str]) -> tuple[dict, dict]:
    """(instance config, run options) from a YAML file in either nefi layout, plus overrides."""
    d = load_config(path)
    inst = d.get("instance")
    cfg = {k: v for k, v in inst.items() if k != "type"} if isinstance(inst, dict) else {}
    cfg.update(d.get("config") or {})
    for item in overrides:
        k, v = item.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg, dict(d.get("run") or {})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/deconvolution_smoke.yaml")
    ap.add_argument("--set", action="append", default=[], help="config override key=value")
    ap.add_argument("--budget", type=float, default=1.0, help="step-count multiplier")
    ap.add_argument("--out", default="runs/baselines_comparison")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # sparse point sources: an ℓ1 prior instead of TV, shared by every method
    base = ["scene=sparse_dots", "tv=0.0", "l1=1.0e-2"]
    cfg, _ = load(args.config, base + args.set)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    inst = Deconvolution(cfg)
    gt, meas = inst.make_measurement(args.seed)
    builders = inst.baselines()
    rows, images = [], {}
    for name in METHODS:
        seed_everything(args.seed)  # every parameterization is initialized from the same seed
        if name == "neural":
            prob = inst.build_problem(meas)
            cur = inst.default_curriculum()
        else:
            prob, cur = builders[name](meas)
        cur = cur.scaled(args.budget) if name != "wiener" else cur
        t0 = time.perf_counter()
        res = solve(prob, cur, device=args.device, seed=args.seed)
        dt = time.perf_counter() - t0
        m = evaluate(res.fields["x"], gt, inst.metrics())
        n_par = res.extra.get("n_parameters", prob.field.n_parameters())
        steps = len(res.history.get("total", []))
        rows.append((name, m["psnr"], m["ssim"], dt, n_par, steps))
        images[name] = (res.fields["x"], m)

    blurred = evaluate(inst.measurement_image(meas), gt, inst.metrics())
    print(f"deconvolution/sparse_dots, n={inst.cfg.n}, psf sigma={inst.cfg.psf_sigma}")
    print(f"blurred measurement: PSNR {blurred['psnr']:.2f} dB, SSIM {blurred['ssim']:.3f}")
    print(
        f"| {'method':15s} | {'PSNR':>6s} | {'SSIM':>6s} | {'time s':>7s} | {'params':>7s} |"
        f" {'steps':>5s} |"
    )
    print(f"|{'-' * 17}|{'-' * 8}|{'-' * 8}|{'-' * 9}|{'-' * 9}|{'-' * 7}|")
    for name, p, s, dt, n_par, steps in rows:
        print(f"| {name:15s} | {p:6.2f} | {s:6.3f} | {dt:7.2f} | {n_par:7d} | {steps:5d} |")

    fig, axes = plt.subplots(1, len(METHODS) + 2, figsize=(2.3 * (len(METHODS) + 2), 2.9))
    panels = [
        ("ground truth", gt["x"], None),
        ("measurement", inst.measurement_image(meas), blurred),
    ]
    panels += [(k, img, m) for k, (img, m) in images.items()]
    for ax, (title, img, m) in zip(axes, panels):
        ax.imshow(img.T, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
        if m is not None:
            title += f"\n{m['psnr']:.1f} dB / {m['ssim']:.2f}"
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    path = out / "baselines_comparison.png"
    fig.savefig(path, dpi=110)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
