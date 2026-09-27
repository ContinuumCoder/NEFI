"""2-D diffraction tomography: generate → invert → evaluate → save a PNG in ``runs/``.

Recovers a weak scattering contrast ``χ = n² − 1`` from complex scattered fields on a ring of
receivers (plane-wave illumination from ``n_angles`` directions) with the linear Born operator and a
neural field. The data come from an independent multiple-scattering (Lippmann–Schwinger) simulation
on a 2× finer grid in float64. Compares with Devaney's filtered backpropagation (closed form) and
the free-pixel ``grid`` baseline.

Usage::

    python examples/diffraction_tomography.py                     # smoke config (CPU, seconds)
    python examples/diffraction_tomography.py --config configs/diffraction_tomography_full.yaml
    python examples/diffraction_tomography.py --set scene=cells --set n_angles=8
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
from nefi.instances.diffraction_tomography import DiffractionTomography  # noqa: E402
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
    ap.add_argument("--config", default="configs/diffraction_tomography_smoke.yaml")
    ap.add_argument("--set", action="append", default=[], help="config override key=value")
    ap.add_argument("--out", default=None, help="output directory (default: run.out)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg, run = load(args.config, args.set)
    seed = run.get("seed", 0) if args.seed is None else args.seed
    device = args.device or run.get("device", "auto")
    methods = run.get("methods", ["neural", "backpropagation", "grid"])
    out = Path(args.out or run.get("out", "runs/diffraction_tomography"))
    out.mkdir(parents=True, exist_ok=True)

    inst = DiffractionTomography(cfg)
    metrics = inst.metrics()
    t0 = time.perf_counter()
    nf = inst.run(seed=seed, device=device)
    gt, meas = nf.gt["chi"], nf.measurement
    rows = {"neural field": (inst.evaluate(nf.result, nf.gt), time.perf_counter() - t0)}
    recon = {"neural field": nf.result.fields["chi"]}
    if "backpropagation" in methods:
        t0 = time.perf_counter()
        fbp = inst.backpropagation(meas)
        recon["filtered backprop."] = fbp
        rows["filtered backprop."] = (evaluate(fbp, gt, metrics), time.perf_counter() - t0)
    if "grid" in methods:
        prob, cur = inst.baselines()["grid"](meas)
        t0 = time.perf_counter()
        res = nefi.invert(prob, cur, device=device, seed=seed)
        recon["grid"] = res.fields["chi"]
        rows["grid"] = (evaluate(res.fields["chi"], gt, metrics), time.perf_counter() - t0)

    c = inst.cfg
    print(
        f"diffraction_tomography ({c.scene}, {c.n}², {c.n_angles} angles × {c.n_receivers} "
        f"receivers, contrast {c.contrast}, rytov={c.rytov})"
    )
    print(f"{'method':22s} {'PSNR':>7s} {'SSIM':>7s} {'rel.err':>8s} {'time [s]':>9s}")
    for name, (m, t) in rows.items():
        print(f"{name:22s} {m['psnr']:7.2f} {m['ssim']:7.3f} {m['relative_error']:8.3f} {t:9.2f}")

    ext = (-c.extent / 2, c.extent / 2, -c.extent / 2, c.extent / 2)
    vmax = float(gt.abs().max())
    panels = [("ground truth χ", gt, None)] + [(k, v, rows[k][0]) for k, v in recon.items()]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(3.2 * (len(panels) + 1), 3.3))
    for ax, (title, img, m) in zip(axes, panels):
        ax.imshow(img.T, origin="lower", cmap="magma", vmin=-0.3 * vmax, vmax=vmax, extent=ext)
        if m is not None:
            title += f"\n{m['psnr']:.1f} dB / SSIM {m['ssim']:.2f}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x [μm]")
    ax = axes[-1]
    ax.imshow(meas.data[0].T, aspect="auto", cmap="RdBu_r", origin="lower")
    ax.set_title("data: Re u_s (receiver × angle)", fontsize=9)
    ax.set_xlabel("incidence angle index")
    fig.tight_layout()
    path = out / "diffraction_tomography.png"
    fig.savefig(path, dpi=110)
    nf.result.save(out / "result.pt")
    summary = {k: {**m, "time_s": tt} for k, (m, tt) in rows.items()}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
