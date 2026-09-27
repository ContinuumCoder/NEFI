"""2-D image deblurring: generate → invert (neural field) → evaluate → save a PNG in ``runs/``.

Compares the neural-field reconstruction with the blurred measurement and the closed-form Wiener
filter (SNR chosen by the discrepancy principle).

Usage::

    python examples/deconvolution.py                                   # smoke config (~5 s CPU)
    python examples/deconvolution.py --config configs/deconvolution_full.yaml --device cuda
    python examples/deconvolution.py --set psf=motion --set noise=poisson
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
from nefi.instances.deconvolution import Deconvolution  # noqa: E402
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
    ap.add_argument("--config", default="configs/deconvolution_smoke.yaml")
    ap.add_argument("--set", action="append", default=[], help="config override key=value")
    ap.add_argument("--out", default=None, help="output directory (default: run.out)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg, run = load(args.config, args.set)
    seed = run.get("seed", 0) if args.seed is None else args.seed
    device = args.device or run.get("device", "auto")
    out = Path(args.out or run.get("out", "runs/deconvolution"))
    out.mkdir(parents=True, exist_ok=True)

    inst = Deconvolution(cfg)
    t0 = time.perf_counter()
    run_out = inst.run(seed=seed, device=device)  # data from the 2x-grid float64 simulator
    t_nf = time.perf_counter() - t0
    gt, meas, result = run_out.gt, run_out.measurement, run_out.result
    t0 = time.perf_counter()
    wien, snr = inst.wiener_reconstruction(meas)
    t_w = time.perf_counter() - t0

    metrics = inst.metrics()
    rows = {
        "blurred measurement": (evaluate(inst.measurement_image(meas), gt, metrics), 0.0),
        f"wiener (snr={snr:.3g})": (evaluate(wien, gt, metrics), t_w),
        "neural field": (inst.evaluate(result, gt), t_nf),
    }
    print(f"deconvolution ({inst.cfg.scene}, psf={inst.cfg.psf}, noise={inst.cfg.noise})")
    print(f"{'method':28s} {'PSNR':>7s} {'SSIM':>7s} {'time [s]':>9s}")
    for name, (m, t) in rows.items():
        print(f"{name:28s} {m['psnr']:7.2f} {m['ssim']:7.3f} {t:9.2f}")

    panels = [
        ("ground truth", gt["x"], None),
        ("measurement", inst.measurement_image(meas), rows["blurred measurement"][0]),
        ("Wiener", wien, rows[f"wiener (snr={snr:.3g})"][0]),
        ("neural field", result.fields["x"], rows["neural field"][0]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    for ax, (title, img, m) in zip(axes, panels):
        ax.imshow(img.T, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
        if m is not None:
            title += f"\n{m['psnr']:.2f} dB / SSIM {m['ssim']:.3f}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    path = out / "deconvolution.png"
    fig.savefig(path, dpi=110)
    result.save(out / "result.pt")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
