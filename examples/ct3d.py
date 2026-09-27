"""3-D sparse-view CT: generate → invert (neural field) → evaluate → PNG mosaic in ``runs/``.

Recovers a volumetric attenuation map from ``n_views`` parallel-beam projections of every axial
slice (rotation about z) and compares the neural field with slice-wise filtered back-projection
(FBP3D) of the same sinogram stack.

Usage::

    python examples/ct3d.py                                            # smoke config (≈ 6 s CPU)
    python examples/ct3d.py --set scene=piecewise --set angle_range=120  # limited angle
    python examples/ct3d.py --config configs/ct3d_full.yaml --device cuda
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
from nefi.instances.ct3d import CT3D  # noqa: E402
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


def slice_mosaic(path: Path, rows: dict, title: str, vmin: float, vmax: float, n: int = 6):
    """Rows = volumes, columns = evenly spaced z-slices plus one central x-z cut."""
    first = next(iter(rows.values()))
    nz = first.shape[-1]
    zs = sorted({int(round(v)) for v in torch.linspace(0, nz - 1, min(n, nz)).tolist()})
    cols = len(zs) + 1
    fig, ax = plt.subplots(len(rows), cols, figsize=(1.9 * cols, 1.95 * len(rows)), squeeze=False)
    for i, (name, vol) in enumerate(rows.items()):
        err = name.startswith("|")
        cmap, lo, hi = ("magma", 0.0, 0.5 * (vmax - vmin)) if err else ("gray", vmin, vmax)
        for j, k in enumerate(zs):
            ax[i, j].imshow(vol[..., k].T, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
            if i == 0:
                ax[i, j].set_title(f"z-slice {k}", fontsize=8)
        xz = vol[:, vol.shape[1] // 2, :]
        ax[i, -1].imshow(xz.T, origin="lower", cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        if i == 0:
            ax[i, -1].set_title("x-z cut", fontsize=8)
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
    ap.add_argument("--config", default="configs/ct3d_smoke.yaml")
    ap.add_argument("--set", action="append", default=[], help="config override key=value")
    ap.add_argument("--out", default=None, help="output directory (default: run.out)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg, run = load(args.config, args.set)
    seed = run.get("seed", 0) if args.seed is None else args.seed
    device = args.device or run.get("device", "auto")
    out = Path(args.out or run.get("out", "runs/ct3d"))
    out.mkdir(parents=True, exist_ok=True)

    inst = CT3D(cfg)
    t0 = time.perf_counter()
    run_out = inst.run(seed=seed, device=device)  # Radon on a 2x grid (4 sub-rays), float64 data
    t_nf = time.perf_counter() - t0
    gt, meas, result = run_out.gt["mu"], run_out.measurement, run_out.result
    t0 = time.perf_counter()
    rec_fbp = inst.fbp(meas)
    t_fbp = time.perf_counter() - t0
    rows = {
        f"FBP3D ({inst.cfg.fbp_filter})": (evaluate(rec_fbp, gt, inst.metrics()), t_fbp),
        "neural field": (run_out.metrics, t_nf),
    }
    c = inst.cfg
    print(f"3-D CT ({c.scene}, {c.n}x{c.n}x{c.n_z}, {c.n_views} views over {c.angle_range:g} deg)")
    print(f"{'method':18s} {'PSNR':>7s} {'SSIM':>7s} {'IoU':>6s} {'time [s]':>9s}")
    for name, (m, t) in rows.items():
        print(f"{name:18s} {m['psnr']:7.2f} {m['ssim']:7.3f} {m['iou']:6.3f} {t:9.2f}")

    nf = result.fields["mu"].float()
    m_nf, m_fbp = rows["neural field"][0], rows[f"FBP3D ({c.fbp_filter})"][0]
    slice_mosaic(
        out / "ct3d.png",
        {
            "ground truth": gt,
            f"neural field\n{m_nf['psnr']:.1f} dB": nf,
            f"FBP3D\n{m_fbp['psnr']:.1f} dB": rec_fbp,
            "|NF − GT|": (nf - gt).abs(),
        },
        f"3-D sparse-view CT: {c.n_views} views over {c.angle_range:g}° · SSIM NF "
        f"{m_nf['ssim']:.3f} vs FBP3D {m_fbp['ssim']:.3f}",
        0.0,
        1.0,
    )
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    k = int(meas.data.flatten(0, 1).var(0).argmax())
    for a, (name, s) in zip(ax, [("measured", meas.data), ("re-projected NF", result.pred)]):
        a.imshow(s[..., k].float(), aspect="auto", cmap="magma")
        a.set_title(f"{name} sinogram, slice {k}", fontsize=9)
        a.set_xlabel("detector")
        a.set_ylabel("view")
    fig.tight_layout()
    fig.savefig(out / "sinogram.png", dpi=110)
    plt.close(fig)
    result.save(out / "result.pt")
    summary = {name: {"metrics": m, "time_s": t} for name, (m, t) in rows.items()}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {out / 'ct3d.png'}, {out / 'sinogram.png'}")


if __name__ == "__main__":
    main()
