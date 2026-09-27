"""Sparse-view CT: generate → invert (neural field) → evaluate → save a PNG in ``runs/``.

Compares the neural-field reconstruction from ``n_views`` projections with filtered
back-projection (FBP) of the same sinogram.

Usage::

    python examples/sparse_view_ct.py                                  # smoke config (~4 s CPU)
    python examples/sparse_view_ct.py --config configs/sparse_view_ct_full.yaml --device cuda
    python examples/sparse_view_ct.py --set scene=piecewise --set n_views=12
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from nefi.config import load_config  # noqa: E402
from nefi.instances.sparse_view_ct import SparseViewCT  # noqa: E402
from nefi.metrics import evaluate  # noqa: E402


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
    ap.add_argument("--config", default="configs/sparse_view_ct_smoke.yaml")
    ap.add_argument("--set", action="append", default=[], help="config override key=value")
    ap.add_argument("--out", default=None, help="output directory (default: run.out)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg, run = load(args.config, args.set)
    seed = run.get("seed", 0) if args.seed is None else args.seed
    device = args.device or run.get("device", "auto")
    out = Path(args.out or run.get("out", "runs/sparse_view_ct"))
    out.mkdir(parents=True, exist_ok=True)

    inst = SparseViewCT(cfg)
    t0 = time.perf_counter()
    run_out = inst.run(seed=seed, device=device)  # Radon on a 2x grid, 2x bins, float64
    t_nf = time.perf_counter() - t0
    gt, meas, result = run_out.gt, run_out.measurement, run_out.result
    t0 = time.perf_counter()
    rec_fbp = inst.fbp(meas)
    t_fbp = time.perf_counter() - t0

    rows = {
        f"FBP ({inst.cfg.fbp_filter})": (evaluate(rec_fbp, gt, inst.metrics()), t_fbp),
        "neural field": (inst.evaluate(result, gt), t_nf),
    }
    c = inst.cfg
    print(f"sparse-view CT ({c.scene}, {c.n}x{c.n}, {c.n_views} views over {c.angle_range:g} deg)")
    print(f"{'method':20s} {'PSNR':>7s} {'SSIM':>7s} {'time [s]':>9s}")
    for name, (m, t) in rows.items():
        print(f"{name:20s} {m['psnr']:7.2f} {m['ssim']:7.3f} {t:9.2f}")

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    axes[0].imshow(gt["mu"].T, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("ground truth", fontsize=9)
    axes[1].imshow(meas.data, aspect="auto", cmap="magma")
    axes[1].set_title(f"sinogram ({c.n_views} views)", fontsize=9)
    for ax, (name, img) in zip(axes[2:], [("FBP", rec_fbp), ("neural field", result.fields["mu"])]):
        m = rows["neural field" if name == "neural field" else f"FBP ({c.fbp_filter})"][0]
        ax.imshow(img.T, origin="lower", cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{name}\n{m['psnr']:.2f} dB / SSIM {m['ssim']:.3f}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    path = out / "sparse_view_ct.png"
    fig.savefig(path, dpi=110)
    result.save(out / "result.pt")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
