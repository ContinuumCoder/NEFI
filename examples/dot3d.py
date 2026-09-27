"""3-D diffuse optical tomography: generate → invert → evaluate → PNG mosaic + sensitivity.

Recovers the absorption ``μ_a(x, y, z)`` of a scattering slab from calibrated surface reflectance
(3×3 sources, a detector-pixel grid on the top face) with a neural field through the diffusion
equation (Robin boundaries, IFT adjoint), compares it with a free voxel grid under the same
objective, and plots the depth-decaying sensitivity ``‖∂y/∂μ_a(x)‖`` that makes deep inclusions
hard (``nefi.diagnostics.sensitivity_map``).

Usage::

    python examples/dot3d.py                                   # smoke config (≈ 7 s CPU)
    python examples/dot3d.py --set scene=deep                  # the hard class
    python examples/dot3d.py --config configs/dot3d_full.yaml --device cuda
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

from nefi.baselines import solve  # noqa: E402
from nefi.config import load_config  # noqa: E402
from nefi.instances.dot3d import DOT3D  # noqa: E402


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
    ap.add_argument("--config", default="configs/dot3d_smoke.yaml")
    ap.add_argument("--set", action="append", default=[], help="config override key=value")
    ap.add_argument("--out", default=None, help="output directory (default: run.out)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-grid", action="store_true", help="skip the free-voxel baseline")
    args = ap.parse_args()
    cfg, run = load(args.config, args.set)
    seed = run.get("seed", 0) if args.seed is None else args.seed
    device = args.device or run.get("device", "auto")
    out = Path(args.out or run.get("out", "runs/dot3d"))
    out.mkdir(parents=True, exist_ok=True)

    inst = DOT3D(cfg)
    t0 = time.perf_counter()
    run_out = inst.run(seed=seed, device=device)  # diffusion FV on a 2x grid, float64 data
    t_nf = time.perf_counter() - t0
    gt, meas, result = run_out.gt["mu_a"], run_out.measurement, run_out.result
    rows = {"neural field": (run_out.metrics, t_nf)}
    grid_mu = None
    if not args.no_grid:
        prob, cur = inst.baselines()["grid"](meas)
        t0 = time.perf_counter()
        res = solve(prob, cur, device=device, seed=seed)
        rows["voxel grid"] = (inst.evaluate(res, run_out.gt), time.perf_counter() - t0)
        grid_mu = res.fields["mu_a"].float()
    c = inst.cfg
    print(
        f"3-D DOT ({c.scene}, grid {tuple(c.grid)}, {tuple(c.source_grid)} sources, "
        f"{tuple(c.detector_grid)} detectors, {c.noise_std:.0%} noise)"
    )
    keys = ("psnr", "ssim", "iou", "depth_error", "contrast")
    print(f"{'method':14s} " + " ".join(f"{k:>11s}" for k in keys) + f" {'time [s]':>9s}")
    for name, (m, t) in rows.items():
        print(f"{name:14s} " + " ".join(f"{m[k]:11.3f}" for k in keys) + f" {t:9.2f}")

    # sensitivity at the background: stochastic column norms of the Jacobian (64 output probes =
    # 64 adjoint solves; exact=True costs one solve per voxel)
    t0 = time.perf_counter()
    sens = inst.sensitivity(n_probes=64)
    t_sens = time.perf_counter() - t0
    depths = inst.depths()
    profile = sens.mean(dim=(0, 1))
    print(
        f"sensitivity (mean over x, y) top/bottom slice ratio {float(profile[0] / profile[-1]):.1f}"
        f" ({t_sens:.1f} s)"
    )

    nf = result.fields["mu_a"].float()
    vols = {"ground truth": gt, "neural field": nf}
    if grid_mu is not None:
        vols["voxel grid"] = grid_mu
    nz = gt.shape[-1]
    zs = sorted({int(round(v)) for v in torch.linspace(0, nz - 1, min(5, nz)).tolist()})
    ex = (gt - c.mua_background).clamp_min(0)
    iy = int(ex.sum(dim=(0, 2)).argmax())  # y-plane through the inclusions
    cols = len(zs) + 1
    fig, ax = plt.subplots(len(vols) + 1, cols, figsize=(2.0 * cols, 2.0 * (len(vols) + 1)))
    ext_xz = [0.0, float(c.extent[0]), float(c.extent[2]), 0.0]
    lims = {"cmap": "viridis", "vmin": c.mua_background * 0.8, "vmax": float(gt.max())}
    for i, (name, vol) in enumerate(vols.items()):
        for j, k in enumerate(zs):
            ax[i, j].imshow(vol[..., k].T, origin="lower", **lims)
            if i == 0:
                ax[i, j].set_title(f"depth {float(depths[k]):.1f} mm", fontsize=8)
        ax[i, -1].imshow(vol[:, iy, :].T, extent=ext_xz, aspect="auto", **lims)
        if i == 0:
            ax[i, -1].set_title("x-z cut (depth ↓)", fontsize=8)
        m = rows.get(name, (None,))[0]
        label = name if m is None else f"{name}\nIoU {m['iou']:.2f} · Δz {m['depth_error']:.1f}"
        ax[i, 0].set_ylabel(label, fontsize=8)
    img = inst.measurement_image(meas)
    lim = float(img.abs().nan_to_num().max()) or 1.0
    b = ax[-1, 0].imshow(img.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[-1, 0].set_title("data: mean log(y/y₀) [%]", fontsize=8)
    fig.colorbar(b, ax=ax[-1, 0], shrink=0.8)
    ax[-1, 1].imshow(sens[..., 0].T, origin="lower", cmap="magma")
    ax[-1, 1].set_title("sensitivity, top slice", fontsize=8)
    ax[-1, 2].semilogy(depths.numpy(), profile.numpy(), "o-", ms=3)
    ax[-1, 2].set_xlabel("depth [mm]", fontsize=8)
    ax[-1, 2].set_title("mean sensitivity vs depth", fontsize=8)
    ax[-1, 3].imshow(sens[:, iy, :].T, cmap="magma", extent=ext_xz, aspect="auto")
    ax[-1, 3].set_title("sensitivity, x-z cut", fontsize=8)
    for j in range(4, cols):
        ax[-1, j].axis("off")
    for a in ax.ravel():
        if a is not ax[-1, 2]:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle(
        f"diffuse optical tomography ({c.scene}): μ_a slices [1/mm], calibrated reflectance and "
        "the depth-decaying sensitivity",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out / "dot3d.png", dpi=110)
    plt.close(fig)
    result.save(out / "result.pt")
    summary = {name: {"metrics": m, "time_s": t} for name, (m, t) in rows.items()}
    summary["sensitivity_profile"] = {"depth_mm": depths.tolist(), "mean": profile.tolist()}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {out / 'dot3d.png'}")


if __name__ == "__main__":
    main()
