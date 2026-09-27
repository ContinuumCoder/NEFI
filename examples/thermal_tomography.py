"""NeFTY thermal tomography: generate → invert → evaluate → save PNG slices.

Recovers the 3-D diffusivity of a slab from the front-surface temperature after a laser flash
(NeFTY, arXiv 2603.11045) with a neural field and the differentiable implicit-Euler heat solver
(discrete adjoint). Synthetic data come from the independent explicit simulator (inverse-crime
guard).

Usage::

    # CPU smoke run (≈ 4 s inversion on one idle core): writes runs/thermal_tomography_smoke/*.png
    python examples/thermal_tomography.py --config configs/thermal_tomography_smoke.yaml

    # paper settings on a CUDA server (64×64×16, 100 frames, 10k steps)
    python examples/thermal_tomography.py --config configs/thermal_tomography_paper.yaml \\
        --device cuda --set compile_solver=true

    # Grid Opt. baseline (App. F.2) on the same measurement
    python examples/thermal_tomography.py --config configs/thermal_tomography_smoke.yaml \\
        --baseline grid
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import yaml

import nefi
from nefi.config import load_config, save_config, to_dict
from nefi.instances.thermal_tomography import (
    ThermalTomography,
    defect_mask_2d,
    depth_map_25d,
    gt_depth_map,
)
from nefi.solve import LoggingCallback
from nefi.utils.seed import seed_everything

log = logging.getLogger("nefi")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/thermal_tomography_smoke.yaml")
    ap.add_argument("--device", default=None, help="auto | cpu | cuda (default: config run.device)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--scene", default=None, help="homogeneous | layered")
    ap.add_argument("--baseline", default=None, choices=["grid"], help="run a baseline instead")
    ap.add_argument("--out", default=None, help="output directory (default: runs/<config stem>)")
    ap.add_argument(
        "--threads", type=int, default=1, help="CPU threads (1 is fastest for tiny grids)"
    )
    ap.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override config fields"
    )
    ap.add_argument("--log-every", type=int, default=25)
    return ap.parse_args()


def load_instance(args: argparse.Namespace) -> tuple[ThermalTomography, dict]:
    cfg = load_config(args.config)
    inst_cfg = dict(cfg.get("instance") or {})
    inst_cfg.pop("type", None)
    for item in args.set:
        key, _, value = item.partition("=")
        inst_cfg[key.strip()] = yaml.safe_load(value)
    return ThermalTomography(inst_cfg), dict(cfg.get("run") or {})


def save_figures(out: Path, inst: ThermalTomography, result, gt, meas, metrics) -> list[str]:
    """PNG slices of GT / recovered / error, surface frames, App. H projections, loss curves."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        log.warning("matplotlib not installed: skipping figures")
        return []
    c = inst.cfg
    a, g = result.fields["alpha"].float(), gt["alpha"].float()
    nz = a.shape[-1]
    zs = sorted({int(round(i)) for i in torch.linspace(0, nz - 1, min(nz, 8)).tolist()})
    H = float(c.extent[-1])
    files = []

    fig, ax = plt.subplots(3, len(zs), figsize=(2.1 * len(zs), 6.4), squeeze=False)
    for j, k in enumerate(zs):
        for i, (img, cmap, lim) in enumerate(
            [
                (g[..., k], "viridis", (c.alpha_min, c.alpha_max)),
                (a[..., k], "viridis", (c.alpha_min, c.alpha_max)),
                ((a - g)[..., k].abs(), "magma", (0.0, 0.5 * (c.alpha_max - c.alpha_min))),
            ]
        ):
            im = ax[i, j].imshow(img.T, origin="lower", cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
        ax[0, j].set_title(f"z = {(k + 0.5) * H / nz:.2f}", fontsize=9)
    for i, name in enumerate(["ground truth α", "recovered α", "|error|"]):
        ax[i, 0].set_ylabel(name, fontsize=9)
    fig.suptitle(
        f"IoU {metrics['iou']:.2f} · PSNR {metrics['psnr']:.1f} dB · SSIM {metrics['ssim']:.2f} · "
        f"Edge F1 {metrics['edge_f1']:.2f}",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax[2].tolist(), shrink=0.6)
    fig.savefig(out / "alpha_slices.png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    files.append("alpha_slices.png")

    obs, pred = meas.data.float(), result.pred.float()
    idx = sorted({int(round(i)) for i in torch.linspace(0, obs.shape[0] - 1, 5).tolist()})
    times = meas.meta.get("frame_times", list(range(obs.shape[0])))
    vmax = float(obs.max())
    fig, ax = plt.subplots(3, len(idx), figsize=(2.3 * len(idx), 6.6), squeeze=False)
    for j, n in enumerate(idx):
        ax[0, j].imshow(obs[n].T, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
        ax[1, j].imshow(pred[n].T, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
        r = (pred[n] - obs[n]).T
        lim = float(r.abs().max()) or 1.0
        ax[2, j].imshow(r, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax[0, j].set_title(f"t = {times[n]:.2f}", fontsize=9)
        for i in range(3):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    for i, name in enumerate(["observed T", "re-simulated T", "residual"]):
        ax[i, 0].set_ylabel(name, fontsize=9)
    fig.suptitle(
        f"surface PSNR {metrics.get('surface_psnr', float('nan')):.1f} dB "
        f"(uniform init {metrics.get('surface_psnr_init', float('nan')):.1f} dB)",
        fontsize=10,
    )
    fig.savefig(out / "surface_frames.png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    files.append("surface_frames.png")

    m_pred = defect_mask_2d(a, k=c.projection_k)
    gt_def = torch.as_tensor(meas.meta.get("defect_mask", g < c.iou_tau))
    d_pred = depth_map_25d(a, None, m_pred, thickness=H, k=c.projection_k)
    d_gt = gt_depth_map(gt_def, thickness=H)
    fig, ax = plt.subplots(1, 4, figsize=(11, 3))
    ax[0].imshow(gt_def.any(-1).T, origin="lower", cmap="gray")
    ax[1].imshow(m_pred.T, origin="lower", cmap="gray")
    for axis, d in ((ax[2], d_gt), (ax[3], d_pred)):
        im = axis.imshow(d.T, origin="lower", cmap="cividis_r", vmin=0, vmax=H)
    fig.colorbar(im, ax=ax[3], shrink=0.8, label="depth")
    for axis, t in zip(ax, ["GT 2-D mask", "2-D mask (App. H)", "GT depth", "2.5-D depth"]):
        axis.set_title(t, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.savefig(out / "projections.png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    files.append("projections.png")

    h = result.history
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.semilogy(h["data"], label="surface MSE")
    if "tv" in h:
        ax.semilogy(h["tv"], label="TV")
    ax.set_xlabel("step")
    ax.legend()
    fig.savefig(out / "loss.png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    files.append("loss.png")
    return files


def main() -> None:
    args = parse_args()
    nefi.enable_logging(logging.INFO)
    inst, run_cfg = load_instance(args)
    device = args.device or run_cfg.get("device", "auto")
    seed = args.seed if args.seed is not None else int(run_cfg.get("seed", 0))
    scene = args.scene or run_cfg.get("scene")
    if nefi.utils.resolve_device(device).type == "cpu" and args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out or Path("runs") / Path(args.config).stem)
    if args.baseline:
        out = out / args.baseline
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    gt, meas = inst.make_measurement(seed=seed, scene_class=scene)
    t_data = time.perf_counter() - t0
    seed_everything(seed)
    if args.baseline:
        problem, curriculum = inst.baselines()[args.baseline](meas)
    else:
        problem, curriculum = inst.build_problem(meas), inst.default_curriculum()
    solver = nefi.Solver(
        problem, curriculum, device=device, seed=seed, callbacks=[LoggingCallback(args.log_every)]
    )
    result = solver.run()
    metrics = inst.evaluate(result, gt, meas)

    result.save(out / "result.pt")
    save_config(
        {
            "instance": {"type": inst.name, **to_dict(inst.cfg)},
            "run": {"seed": seed, "device": device},
        },
        out / "config.yaml",
    )
    summary = {
        "metrics": metrics,
        "timing": {"data_s": t_data, **result.timing},
        "device": result.extra.get("device"),
        "n_parameters": result.extra.get("n_parameters"),
        "scene": meas.meta.get("scene"),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    files = save_figures(out, inst, result, gt, meas, metrics)

    print(f"[{inst.name}{' / ' + args.baseline if args.baseline else ''}] seed {seed} -> {out}")
    dev = result.extra["device"]
    print(f"  data {t_data:.1f}s · inversion {result.timing['total_s']:.1f}s on {dev}")
    print("  " + " · ".join(f"{k} {v:.4g}" for k, v in metrics.items()))
    print("  wrote " + ", ".join(["result.pt", "config.yaml", "metrics.json", *files]))


if __name__ == "__main__":
    main()
