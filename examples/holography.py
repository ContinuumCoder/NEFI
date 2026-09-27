"""Multi-distance inline holography: generate → invert → evaluate → save a PNG in ``runs/``.

Retrieves the phase of a thin transparent sample from intensity images recorded at several
propagation distances (band-limited angular-spectrum propagation) with a neural field. Data are
simulated on a 2× finer grid in float64 and area-averaged onto the detector pixels. Phases are
compared after mean subtraction (the global offset is invisible in intensities). Compares with the
multi-plane Gerchberg–Saxton algorithm and the free-pixel ``grid`` baseline.

Usage::

    python examples/holography.py                                 # smoke config (CPU, seconds)
    python examples/holography.py --config configs/holography_full.yaml --device cuda
    python examples/holography.py --set scene=cells --set "distances=[20.0, 60.0]"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

import nefi  # noqa: E402
from nefi.config import load_config  # noqa: E402
from nefi.instances.holography import Holography  # noqa: E402
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
    if isinstance(cfg.get("distances"), list):
        cfg["distances"] = tuple(cfg["distances"])
    return cfg, dict(d.get("run") or {})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/holography_smoke.yaml")
    ap.add_argument("--set", action="append", default=[], help="config override key=value")
    ap.add_argument("--out", default=None, help="output directory (default: run.out)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg, run = load(args.config, args.set)
    seed = run.get("seed", 0) if args.seed is None else args.seed
    device = args.device or run.get("device", "auto")
    methods = run.get("methods", ["neural", "gerchberg_saxton", "grid"])
    out = Path(args.out or run.get("out", "runs/holography"))
    out.mkdir(parents=True, exist_ok=True)

    inst = Holography(cfg)
    metrics = inst.metrics()
    t0 = time.perf_counter()
    nf = inst.run(seed=seed, device=device)
    gt, meas = nf.gt["phase"], nf.measurement
    rows = {"neural field": (inst.evaluate(nf.result, nf.gt), time.perf_counter() - t0)}
    recon = {"neural field": nf.result.fields["phase"]}
    if "gerchberg_saxton" in methods:
        t0 = time.perf_counter()
        gs = inst.gerchberg_saxton(meas)
        recon["Gerchberg-Saxton"] = gs
        rows["Gerchberg-Saxton"] = (evaluate(gs, gt, metrics), time.perf_counter() - t0)
    if "grid" in methods:
        prob, cur = inst.baselines()["grid"](meas)
        t0 = time.perf_counter()
        res = nefi.invert(prob, cur, device=device, seed=seed)
        recon["grid"] = res.fields["phase"]
        rows["grid"] = (evaluate(res.fields["phase"], gt, metrics), time.perf_counter() - t0)

    c = inst.cfg
    print(f"holography ({c.scene}, {c.n}², λ={c.wavelength} μm, z={list(c.distances)} μm)")
    print(f"{'method':20s} {'PSNR':>7s} {'SSIM':>7s} {'RMSE[rad]':>10s} {'time [s]':>9s}")
    for name, (m, t) in rows.items():
        print(f"{name:20s} {m['psnr']:7.2f} {m['ssim']:7.3f} {m['rmse']:10.4f} {t:9.2f}")

    lo, hi = float((gt - gt.mean()).min()), float((gt - gt.mean()).max())
    panels = [("ground truth φ", gt, None)] + [(k, v, rows[k][0]) for k, v in recon.items()]
    n_int = meas.data.shape[0]
    fig, axes = plt.subplots(1, len(panels) + n_int, figsize=(2.9 * (len(panels) + n_int), 3.5))
    for ax, (title, img, m) in zip(axes, panels):
        ax.imshow((img - img.mean()).T, origin="lower", cmap="twilight_shifted", vmin=lo, vmax=hi)
        if m is not None:
            title += f"\n{m['psnr']:.1f} dB / SSIM {m['ssim']:.2f}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for i in range(n_int):
        ax = axes[len(panels) + i]
        ax.imshow(meas.data[i].T, origin="lower", cmap="gray")
        ax.set_title(f"intensity z = {c.distances[i]:g} μm", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    path = out / "holography.png"
    fig.savefig(path, dpi=110)
    nf.result.save(out / "result.pt")
    summary = {k: {**m, "time_s": tt} for k, (m, tt) in rows.items()}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
