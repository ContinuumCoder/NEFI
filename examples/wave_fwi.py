"""Acoustic full-waveform inversion: generate → invert → evaluate → save a PNG in ``runs/``.

Recovers a 2-D sound-speed map (km/s) from pressure traces simulated by the leapfrog + PML wave
solver (gradients = discrete adjoint by autodiff) with a neural field and a two-stage curriculum:
a coarse grid fitting low-pass filtered traces, then the full grid and the full band (frequency
continuation against cycle skipping). Compares with the uniform starting model and the free-pixel
``grid`` baseline.

Usage::

    python examples/wave_fwi.py                                   # smoke config (CPU)
    python examples/wave_fwi.py --config configs/wave_fwi_full.yaml --device cuda
    python examples/wave_fwi.py --set scene=layered --set geometry=reflection
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
from nefi.instances.wave_fwi import WaveFWI  # noqa: E402
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
    ap.add_argument("--config", default="configs/wave_fwi_smoke.yaml")
    ap.add_argument("--set", action="append", default=[], help="config override key=value")
    ap.add_argument("--out", default=None, help="output directory (default: run.out)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg, run = load(args.config, args.set)
    seed = run.get("seed", 0) if args.seed is None else args.seed
    device = args.device or run.get("device", "auto")
    methods = run.get("methods", ["neural", "grid"])
    out = Path(args.out or run.get("out", "runs/wave_fwi"))
    out.mkdir(parents=True, exist_ok=True)

    inst = WaveFWI(cfg)
    metrics = inst.metrics()
    t0 = time.perf_counter()
    nf = inst.run(seed=seed, device=device)  # data: 2x-finer float64 simulation + noise
    rows = {"neural field": (inst.evaluate(nf.result, nf.gt), time.perf_counter() - t0)}
    gt, meas = nf.gt["c"], nf.measurement
    recon = {"neural field": nf.result.fields["c"]}
    init = inst.initial_model()
    rows["initial (uniform)"] = (evaluate(init, gt, metrics), 0.0)
    if "grid" in methods:
        prob, cur = inst.baselines()["grid"](meas)
        t0 = time.perf_counter()
        res = nefi.invert(prob, cur, device=device, seed=seed)
        recon["grid"] = res.fields["c"]
        rows["grid"] = (evaluate(res.fields["c"], gt, metrics), time.perf_counter() - t0)

    c = inst.cfg
    print(f"wave_fwi ({c.scene}, {c.geometry}, {c.n}², {c.n_sources} src × {c.n_receivers} rec)")
    print(f"{'method':20s} {'PSNR':>7s} {'SSIM':>7s} {'anomaly':>8s} {'time [s]':>9s}")
    for name, (m, t) in rows.items():
        print(f"{name:20s} {m['psnr']:7.2f} {m['ssim']:7.3f} {m['anomaly_error']:8.3f} {t:9.2f}")

    src, rec = inst.acquisition()
    L = c.extent
    panels = [("ground truth", gt, None), ("initial", init, rows["initial (uniform)"][0])]
    panels += [(k, v, rows[k][0]) for k, v in recon.items()]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(3.3 * (len(panels) + 1), 3.4))
    for ax, (title, img, m) in zip(axes, panels):
        im = ax.imshow(img.T, cmap="viridis", vmin=c.c_min, vmax=c.c_max, extent=(0, L, L, 0))
        ax.plot(src[:, 0], src[:, 1], "r*", ms=8)
        ax.plot(rec[:, 0], rec[:, 1], "wv", ms=4)
        if m is not None:
            title += f"\n{m['psnr']:.1f} dB, anomaly err {m['anomaly_error']:.2f}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x [km]")
    axes[0].set_ylabel("z [km]")
    fig.colorbar(im, ax=axes[len(panels) - 1], fraction=0.046, label="c [km/s]")
    ax = axes[-1]
    times = [j * c.dt_obs for j in range(meas.data.shape[-1])]
    scale = float(meas.data.abs().max())
    for r in range(0, meas.data.shape[1], max(1, meas.data.shape[1] // 4)):
        ax.plot(times, meas.data[0, r] / scale + r, "k", lw=0.8)
        ax.plot(times, nf.result.pred[0, r] / scale + r, "r--", lw=0.8)
    ax.set_title("traces, source 0 (black: data, red: fit)", fontsize=9)
    ax.set_xlabel("t [s]")
    fig.tight_layout()
    path = out / "wave_fwi.png"
    fig.savefig(path, dpi=110)
    nf.result.save(out / "result.pt")
    summary = {k: {**m, "time_s": tt} for k, (m, tt) in rows.items()}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
