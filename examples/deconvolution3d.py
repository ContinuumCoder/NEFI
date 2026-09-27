"""3-D fluorescence deconvolution: generate → invert (neural field) → evaluate → PNG mosaic.

Deblurs a widefield z-stack (anisotropic Gaussian PSF, σ_z = 3 σ_xy) with a neural field and
compares it with the classical references on the same stack: Richardson–Lucy (ML-EM) and the
3-D Wiener filter (SNR by the discrepancy principle).

Usage::

    python examples/deconvolution3d.py                                # smoke config (≈ 8 s CPU)
    python examples/deconvolution3d.py --set scene=puncta --set l1=1e-3
    python examples/deconvolution3d.py --set noise=poisson --set peak_counts=100
    python examples/deconvolution3d.py --config configs/deconvolution3d_full.yaml --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from nefi.config import load_config  # noqa: E402
from nefi.instances.deconvolution3d import Deconvolution3D  # noqa: E402
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


def mosaic(path: Path, rows: dict, title: str, vmax: float, n: int = 5) -> None:
    """Rows = volumes; columns = z-slices, the max-intensity projection (MIP) and an x-z MIP."""
    nz = next(iter(rows.values())).shape[-1]
    zs = sorted({int(round(v)) for v in torch.linspace(1, nz - 2, min(n, nz)).tolist()})
    cols = len(zs) + 2
    fig, ax = plt.subplots(len(rows), cols, figsize=(1.9 * cols, 1.9 * len(rows)), squeeze=False)
    for i, (name, vol) in enumerate(rows.items()):
        panels = [(f"z-slice {k}", vol[..., k]) for k in zs]
        panels += [("MIP over z", vol.amax(-1)), ("x-z MIP", vol.amax(1))]
        for j, (lab, img) in enumerate(panels):
            ax[i, j].imshow(
                img.T,
                origin="lower",
                cmap="inferno",
                vmin=0.0,
                vmax=vmax,
                aspect="auto" if lab == "x-z MIP" else "equal",
            )
            if i == 0:
                ax[i, j].set_title(lab, fontsize=8)
        ax[i, 0].set_ylabel(name, fontsize=8)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/deconvolution3d_smoke.yaml")
    ap.add_argument("--set", action="append", default=[], help="config override key=value")
    ap.add_argument("--out", default=None, help="output directory (default: run.out)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg, run = load(args.config, args.set)
    seed = run.get("seed", 0) if args.seed is None else args.seed
    device = args.device or run.get("device", "auto")
    out = Path(args.out or run.get("out", "runs/deconvolution3d"))
    out.mkdir(parents=True, exist_ok=True)

    inst = Deconvolution3D(cfg)
    t0 = time.perf_counter()
    run_out = inst.run(seed=seed, device=device)  # PSF blur on a 2x grid in float64
    t_nf = time.perf_counter() - t0
    gt, meas, result = run_out.gt["x"], run_out.measurement, run_out.result
    stack = inst.measurement_stack(meas)
    t0 = time.perf_counter()
    rl = inst.richardson_lucy(meas)
    t_rl = time.perf_counter() - t0
    t0 = time.perf_counter()
    wi, snr = inst.wiener_reconstruction(meas)
    t_wi = time.perf_counter() - t0
    metrics = inst.metrics()
    rows = {
        "blurred stack": (evaluate(stack, gt, metrics), 0.0),
        f"Wiener (snr {snr:.3g})": (evaluate(wi, gt, metrics), t_wi),
        f"Richardson-Lucy ({inst.cfg.rl_iters} it)": (evaluate(rl, gt, metrics), t_rl),
        "neural field": (run_out.metrics, t_nf),
    }
    c = inst.cfg
    sx, _, sz = inst.psf_sigma()
    print(
        f"3-D deconvolution ({c.scene}, {c.n}x{c.n}x{c.n_z}, PSF σ_xy {sx:.2f} µm / σ_z "
        f"{sz:.2f} µm, {c.noise} noise)"
    )
    print(f"{'method':26s} {'PSNR':>7s} {'SSIM':>7s} {'time [s]':>9s}")
    for name, (m, t) in rows.items():
        print(f"{name:26s} {m['psnr']:7.2f} {m['ssim']:7.3f} {t:9.2f}")

    nf = result.fields["x"].float()
    vmax = float(gt.max())
    mosaic(
        out / "deconvolution3d.png",
        {
            "ground truth": gt,
            f"blurred\n{rows['blurred stack'][0]['psnr']:.1f} dB": stack.clamp_min(0.0),
            f"Richardson-Lucy\n{evaluate(rl, gt, metrics)['psnr']:.1f} dB": rl,
            f"neural field\n{run_out.metrics['psnr']:.1f} dB": nf,
        },
        f"widefield z-stack deconvolution ({c.scene}): SSIM NF {run_out.metrics['ssim']:.3f}"
        f" · RL {evaluate(rl, gt, metrics)['ssim']:.3f}"
        f" · Wiener {evaluate(wi, gt, metrics)['ssim']:.3f}",
        vmax,
    )
    result.save(out / "result.pt")
    summary = {name: {"metrics": m, "time_s": t} for name, (m, t) in rows.items()}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {out / 'deconvolution3d.png'}")


if __name__ == "__main__":
    main()
