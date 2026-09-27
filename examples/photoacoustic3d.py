"""3-D photoacoustic tomography: generate → invert (neural field) → evaluate → PNG mosaic.

Recovers the initial pressure of a tissue slab from a planar sensor array on its top face (the
limited-view geometry of planar PAT) through the 3-D acoustic wave equation, and compares the
neural field with the classical time-reversal reconstruction of the same traces.

Usage::

    python examples/photoacoustic3d.py                              # smoke config (≈ 6 s CPU)
    python examples/photoacoustic3d.py --set scene=spheres
    python examples/photoacoustic3d.py --config configs/photoacoustic3d_full.yaml --device cuda
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
from nefi.instances.photoacoustic3d import Photoacoustic3D  # noqa: E402
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
    ap.add_argument("--config", default="configs/photoacoustic3d_smoke.yaml")
    ap.add_argument("--set", action="append", default=[], help="config override key=value")
    ap.add_argument("--out", default=None, help="output directory (default: run.out)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg, run = load(args.config, args.set)
    seed = run.get("seed", 0) if args.seed is None else args.seed
    device = args.device or run.get("device", "auto")
    out = Path(args.out or run.get("out", "runs/photoacoustic3d"))
    out.mkdir(parents=True, exist_ok=True)

    inst = Photoacoustic3D(cfg)
    t0 = time.perf_counter()
    run_out = inst.run(seed=seed, device=device)  # wave solve on a 2x grid in float64
    t_nf = time.perf_counter() - t0
    gt, meas, result = run_out.gt["p0"], run_out.measurement, run_out.result
    t0 = time.perf_counter()
    tr = inst.time_reversal(meas)
    t_tr = time.perf_counter() - t0
    rows = {
        "time reversal": (evaluate(tr, gt, inst.metrics()), t_tr),
        "neural field": (run_out.metrics, t_nf),
    }
    c = inst.cfg
    print(
        f"3-D PAT ({c.scene}, grid {tuple(c.grid)} over {tuple(c.extent)} mm, "
        f"{tuple(c.sensor_grid)} sensors, {inst.n_t} samples, fc {c.sensor_fc} MHz)"
    )
    print(f"{'method':14s} {'PSNR':>7s} {'SSIM':>7s} {'time [s]':>9s}")
    for name, (m, t) in rows.items():
        print(f"{name:14s} {m['psnr']:7.2f} {m['ssim']:7.3f} {t:9.2f}")

    nf = result.fields["p0"].float()
    vols = {"ground truth": gt, "neural field": nf, "time reversal": tr}
    nz = gt.shape[-1]
    zs = sorted({int(round(v)) for v in torch.linspace(1, nz - 2, min(4, nz)).tolist()})
    cols = len(zs) + 2
    vmax = float(gt.max())
    fig, ax = plt.subplots(len(vols) + 1, cols, figsize=(2.0 * cols, 2.0 * (len(vols) + 1)))
    for i, (name, vol) in enumerate(vols.items()):
        panels = [(f"depth slice {k}", vol[..., k]) for k in zs]
        panels += [("MIP over depth", vol.amax(-1)), ("x-z MIP (depth ↓)", vol.amax(1))]
        for j, (lab, img) in enumerate(panels):
            xz = lab.startswith("x-z")
            ax[i, j].imshow(
                img.T,
                origin="upper" if xz else "lower",
                cmap="hot",
                vmin=0.0,
                vmax=vmax,
                aspect="auto" if xz else "equal",
            )
            if i == 0:
                ax[i, j].set_title(lab, fontsize=8)
        m = rows.get(name, (None,))[0]
        ax[i, 0].set_ylabel(name if m is None else f"{name}\n{m['psnr']:.1f} dB", fontsize=8)
    lim = float(meas.data.abs().max())
    for j, (lab, d) in enumerate([("measured traces", meas.data), ("NF prediction", result.pred)]):
        ax[-1, j].imshow(d.float().T, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax[-1, j].set_title(lab, fontsize=8)
        ax[-1, j].set_xlabel("sensor", fontsize=7)
        ax[-1, j].set_ylabel("time sample", fontsize=7)
    for j in range(2, cols):
        ax[-1, j].axis("off")
    for a in ax[:-1].ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(
        f"planar-array photoacoustic tomography ({c.scene}): SSIM NF "
        f"{run_out.metrics['ssim']:.3f} vs time reversal {rows['time reversal'][0]['ssim']:.3f}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out / "photoacoustic3d.png", dpi=110)
    plt.close(fig)
    result.save(out / "result.pt")
    summary = {name: {"metrics": m, "time_s": t} for name, (m, t) in rows.items()}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {out / 'photoacoustic3d.png'}")


if __name__ == "__main__":
    main()
