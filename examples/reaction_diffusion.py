"""Gray–Scott feed-rate inversion: generate → invert → evaluate → save a PNG in ``runs/``.

Recovers the spatially varying feed rate ``F(x)`` of a Gray–Scott reaction–diffusion system from
snapshots of ``u`` and ``v`` at a few times (explicit-Euler solver, gradients by autodiff through
the unrolled loop) with a neural field. The data come from a 4× substepped float64 simulation.
Compares with the uniform starting model and the free-pixel ``grid`` baseline.

Usage::

    python examples/reaction_diffusion.py                         # smoke config (CPU)
    python examples/reaction_diffusion.py --config configs/reaction_diffusion_full.yaml
    python examples/reaction_diffusion.py --set scene=stripes
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
from nefi.instances.reaction_diffusion import ReactionDiffusion  # noqa: E402
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
    for key in ("obs_times", "observe", "ic_modes", "steps"):
        if isinstance(cfg.get(key), list):
            cfg[key] = tuple(cfg[key])
    return cfg, dict(d.get("run") or {})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/reaction_diffusion_smoke.yaml")
    ap.add_argument("--set", action="append", default=[], help="config override key=value")
    ap.add_argument("--out", default=None, help="output directory (default: run.out)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg, run = load(args.config, args.set)
    seed = run.get("seed", 0) if args.seed is None else args.seed
    device = args.device or run.get("device", "auto")
    methods = run.get("methods", ["neural", "grid"])
    out = Path(args.out or run.get("out", "runs/reaction_diffusion"))
    out.mkdir(parents=True, exist_ok=True)

    inst = ReactionDiffusion(cfg)
    metrics = inst.metrics()
    t0 = time.perf_counter()
    nf = inst.run(seed=seed, device=device)
    gt, meas = nf.gt["F"], nf.measurement
    rows = {"neural field": (inst.evaluate(nf.result, nf.gt), time.perf_counter() - t0)}
    recon = {"neural field": nf.result.fields["F"]}
    init = inst.initial_model()
    rows["initial (uniform)"] = (evaluate(init, gt, metrics), 0.0)
    if "grid" in methods:
        prob, cur = inst.baselines()["grid"](meas)
        t0 = time.perf_counter()
        res = nefi.invert(prob, cur, device=device, seed=seed)
        recon["grid"] = res.fields["F"]
        rows["grid"] = (evaluate(res.fields["F"], gt, metrics), time.perf_counter() - t0)

    c = inst.cfg
    print(
        f"reaction_diffusion ({c.scene}, {c.n}², k={c.k}, observe={list(c.observe)} "
        f"at t={list(c.obs_times)})"
    )
    print(f"{'method':20s} {'PSNR':>7s} {'SSIM':>7s} {'rel.err':>8s} {'time [s]':>9s}")
    for name, (m, t) in rows.items():
        print(f"{name:20s} {m['psnr']:7.2f} {m['ssim']:7.3f} {m['relative_error']:8.4f} {t:9.2f}")

    panels = [("ground truth F", gt, None)] + [(k, v, rows[k][0]) for k, v in recon.items()]
    n_species = len(c.observe)
    snaps = [meas.data[i * n_species] for i in range(len(c.obs_times))]  # u at each time
    fig, axes = plt.subplots(
        1, len(panels) + len(snaps), figsize=(2.9 * (len(panels) + len(snaps)), 3.5)
    )
    for ax, (title, img, m) in zip(axes, panels):
        ax.imshow(img.T, origin="lower", cmap="viridis", vmin=c.F_min, vmax=c.F_max)
        if m is not None:
            title += f"\n{m['psnr']:.1f} dB / rel.err {m['relative_error']:.3f}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for i, s in enumerate(snaps):
        ax = axes[len(panels) + i]
        ax.imshow(s.T, origin="lower", cmap="magma")
        ax.set_title(f"{c.observe[0]}(t = {c.obs_times[i]:g})", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    path = out / "reaction_diffusion.png"
    fig.savefig(path, dpi=110)
    nf.result.save(out / "result.pt")
    summary = {k: {**m, "time_s": tt} for k, (m, tt) in rows.items()}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
