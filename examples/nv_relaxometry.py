#!/usr/bin/env python
"""NV relaxometry (NeTMY) end-to-end: generate (F3) → invert (F2) → evaluate → figure.

Runs the NeTMY instance of nefi on one synthetic sample: a sparse spin scene is simulated with the
float64 source-side direct simulator F3 plus 1 % sensor noise, inverted with the coordinate neural
field through the FFT-factorized tensor operator F2 (two-stage 32² → 64² curriculum),
scale-corrected (Eq. 30) and scored with GMSD / Hungarian F1 / SWD / density MSE / masked SSIM
(App. E.3).

Examples::

    # tiny local smoke run (16×16, 100 steps, a few seconds on a CPU)
    python examples/nv_relaxometry.py --config configs/nv_relaxometry_smoke.yaml

    # paper settings on a CUDA server, NeTMY vs. Tikhonov under F2 and F1 (Tab. 1 protocol)
    python examples/nv_relaxometry.py --config configs/nv_relaxometry_paper.yaml --device cuda \
        --scene many/close --seed 3

    # override any config field (YAML syntax) and scale every curriculum for a quick look
    python examples/nv_relaxometry.py --config configs/nv_relaxometry_paper.yaml \
        --set n=32 hidden=128 --steps-scale 0.1 --methods netmy,grid

Outputs go to ``<run.out>/<scene>_seed<seed>/``: ``metrics.json``, ``config.yaml``,
``result_<method>.pt`` (``nefi.Result``) and ``nv_relaxometry.png``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

import nefi
from nefi.config import save_config, to_dict
from nefi.instances.nv_relaxometry import NVRelaxometry, noise_map
from nefi.metrics import peak_positions
from nefi.registry import build
from nefi.solve import LoggingCallback

log = logging.getLogger("nefi")

# Reference palette (dataviz skill): light chart surface, ink tokens, categorical slots 1-3
# (validated: CVD ΔE 9.2, normal-vision ΔE 27.6), and the one-hue blue sequential ramp.
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQUENTIAL = [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
METHOD_LABELS = {
    "netmy": "NeTMY (F2)",
    "f2": "NeTMY (F2)",
    "f1": "NeTMY (F1)",
    "grid": "Tikhonov (F2)",
    "grid_f1": "Tikhonov (F1)",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/nv_relaxometry_smoke.yaml", help="YAML config")
    ap.add_argument("--device", default=None, help="auto | cpu | cuda | mps (default: run.device)")
    ap.add_argument("--seed", type=int, default=None, help="sample + optimizer seed")
    ap.add_argument("--scene", default=None, help="scene class, e.g. few/far or many/close")
    ap.add_argument("--methods", default=None, help="comma list: netmy,f1,grid,grid_f1")
    ap.add_argument("--out", default=None, help="output directory (default: run.out)")
    ap.add_argument("--steps-scale", type=float, default=1.0, help="scale every curriculum")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--log-every", type=int, default=None, help="log losses every N steps")
    ap.add_argument("--no-plot", action="store_true", help="skip the PNG figure")
    return ap.parse_args(argv)


def load_config(path: str, overrides: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a config file into (instance config, run options); accepts nested or flat YAML."""
    cfg = nefi.load_config(path)
    inst = dict(cfg.get("instance", {k: v for k, v in cfg.items() if k != "run"}))
    run = dict(cfg.get("run", {}))
    inst.setdefault("type", "nv_relaxometry")
    for item in overrides:
        key, _, value = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        inst[key.strip()] = yaml.safe_load(value)
    return inst, run


def _fraction_of_max(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().cpu()
    return x / x.max().clamp_min(1e-30)


def save_figure(outs: dict, path: Path, title: str) -> Path | None:
    """Observation, ground truth and every reconstruction (value / max) + loss histories."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.lines import Line2D
    except ImportError:  # pragma: no cover - matplotlib is optional
        log.warning("matplotlib not installed; skipping the figure")
        return None

    first = next(iter(outs.values()))
    gt_rho = first.gt["rho"]
    gt_pk = peak_positions(gt_rho)
    obs = noise_map(first.measurement.data)
    panels = [("Observed noise map  Σω S", obs, None), ("Ground truth ρ⋆", gt_rho, None)]
    for name, o in outs.items():
        m = o.metrics
        sub = f"HF1 {m['hungarian_f1']:.2f} · SWD {m['swd']:.3f} · GMSD {m['gmsd']:.3f}"
        panels.append((METHOD_LABELS.get(name, name), o.result.fields["rho"], sub))

    cmap = LinearSegmentedColormap.from_list("nefi_blue", SEQUENTIAL)
    ncol = 3
    nrow_img = (len(panels) + ncol - 1) // ncol
    fig = plt.figure(figsize=(3.4 * ncol + 0.9, 3.35 * nrow_img + 2.9), facecolor=SURFACE)
    gs = fig.add_gridspec(
        nrow_img + 1,
        ncol + 1,
        width_ratios=[1] * ncol + [0.06],
        height_ratios=[1] * nrow_img + [0.85],
        hspace=0.42,
        wspace=0.12,
    )
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    im = None
    for k, (label, img, sub) in enumerate(panels):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        ax.set_facecolor(SURFACE)
        im = ax.imshow(_fraction_of_max(img), cmap=cmap, vmin=0.0, vmax=1.0, origin="upper")
        if len(gt_pk):
            ax.plot(gt_pk[:, 1], gt_pk[:, 0], "o", ms=10, mfc="none", mec=SERIES[1], mew=1.6)
        if k >= 2:
            pk = peak_positions(img)
            if len(pk):
                ax.plot(pk[:, 1], pk[:, 0], "+", ms=7, color=INK, mew=1.4)
        ax.set_title(label, color=INK, fontsize=10, loc="left", pad=16 if sub else 6)
        if sub:
            ax.text(0.0, 1.02, sub, transform=ax.transAxes, color=INK_2, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
    cax = fig.add_subplot(gs[0, ncol])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("value / max", color=INK_2)
    cb.outline.set_edgecolor(GRID)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    handles = [
        Line2D([], [], ls="none", marker="o", ms=9, mfc="none", mec=SERIES[1], mew=1.6),
        Line2D([], [], ls="none", marker="+", ms=8, color=INK, mew=1.4),
    ]
    fig.legend(
        handles,
        ["true source", "detected peak (> 5 % of max)"],
        loc="upper right",
        frameon=False,
        ncol=2,
        fontsize=8,
        labelcolor=INK_2,
    )

    # loss histories of the primary method: raw values on one log axis; each stage re-evaluates the
    # losses on its own grid, so values jump at the stage boundary (marked)
    main = outs.get("netmy") or first
    hist = main.result.history
    ax = fig.add_subplot(gs[nrow_img, :ncol])
    ax.set_facecolor(SURFACE)
    series = [
        ("log_mse", "D  log-MSE, max-normalized (stage 1)"),
        ("noise_map", "R_nm  mean-normalized MSE"),
        ("total", "total objective"),
    ]
    steps = hist.get("global_step", list(range(len(hist.get("total", [])))))
    last_step = steps[-1] if steps else 0
    for color, (key, label) in zip(SERIES, series):
        vals = hist.get(key)
        if not vals:
            continue
        v = torch.tensor(vals, dtype=torch.float64).abs().clamp_min(1e-12).numpy()
        x = steps[: len(v)]
        ax.plot(x, v, color=color, lw=2, solid_capstyle="round", label=label)
        if x and x[-1] == last_step:  # direct-label only the series that reach the end
            ax.annotate(
                label.split("  ")[0],
                (x[-1], float(v[-1])),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                color=INK_2,
                fontsize=8,
            )
    stage = hist.get("stage", [])
    for i in range(1, len(stage)):
        if stage[i] != stage[i - 1]:
            ax.axvline(steps[i], color=MUTED, lw=1)
            ax.text(
                steps[i],
                0.03,
                " stage 2 (finer grid)",
                transform=ax.get_xaxis_transform(),
                color=MUTED,
                fontsize=8,
                va="bottom",
            )
    ax.set_yscale("log")
    ax.set_xlabel("optimization step", color=INK_2)
    ax.set_ylabel("loss value", color=INK_2)
    ax.set_title(
        f"{METHOD_LABELS.get('netmy')} loss history", color=INK, loc="left", fontsize=10, pad=18
    )
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.legend(
        frameon=False,
        fontsize=8,
        labelcolor=INK_2,
        loc="lower right",
        ncol=3,
        bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0.2,
    )
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    nefi.enable_logging()
    inst_cfg, run = load_config(args.config, args.set)
    inst: NVRelaxometry = build("instance", inst_cfg)
    seed = int(args.seed if args.seed is not None else run.get("seed", 0))
    device = args.device or run.get("device", "auto")
    scene = args.scene or run.get("scene") or inst.cfg.scene
    methods = (args.methods.split(",") if args.methods else None) or run.get("methods", ["netmy"])
    out_root = Path(args.out or run.get("out", "runs/nv_relaxometry"))
    out_dir = out_root / f"{scene.replace('/', '_')}_seed{seed}"
    total_steps = inst.default_curriculum().scaled(args.steps_scale).total_steps
    every = args.log_every or max(1, total_steps // 10)

    log.info(
        "NeTMY NV relaxometry: scene %s, seed %d, methods %s, device %s",
        scene,
        seed,
        methods,
        device,
    )
    t0 = time.perf_counter()
    gt, meas = inst.make_measurement(seed, scene)
    log.info(
        "measurement %s from %s, noise σ = %.3g, %d sources",
        tuple(meas.shape),
        meas.meta.get("fidelity"),
        meas.noise_std or 0.0,
        int((gt["rho"] > 0).sum()),
    )
    outs = inst.compare(
        seed=seed,
        scene_class=scene,
        methods=methods,
        device=device,
        step_scale=args.steps_scale,
        callbacks=[LoggingCallback(every=every)],
        measurement=(gt, meas),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"scene": scene, "seed": seed, "device": device, "methods": {}}
    for name, o in outs.items():
        o.result.save(out_dir / f"result_{name}.pt")
        summary["methods"][name] = {
            **{k: round(v, 6) for k, v in o.metrics.items()},
            "scale_factor": o.result.post_info.get("scale_factor"),
            "seconds": round(o.result.timing.get("total_s", float("nan")), 2),
            "steps": len(o.result.history.get("total", [])),
        }
    summary["wall_seconds"] = round(time.perf_counter() - t0, 2)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    save_config(
        {"instance": to_dict(inst.cfg), "run": {**run, "seed": seed, "scene": scene}},
        out_dir / "config.yaml",
    )

    header = f"{'method':<16}{'HF1':>7}{'SWD':>9}{'GMSD':>9}{'MSE':>11}{'mSSIM':>8}{'time':>8}"
    lines = [header, "-" * len(header)]
    for name, m in summary["methods"].items():
        lines.append(
            f"{METHOD_LABELS.get(name, name):<16}{m['hungarian_f1']:>7.3f}{m['swd']:>9.4f}"
            f"{m['gmsd']:>9.4f}{m['mse']:>11.2e}{m['masked_ssim']:>8.3f}{m['seconds']:>7.1f}s"
        )
    log.info("results (%s, seed %d):\n%s", scene, seed, "\n".join(lines))
    if not args.no_plot:
        c = inst.cfg
        title = f"NeTMY · {scene} · seed {seed} · {c.n}×{c.n} · data {c.data_operator}"
        png = save_figure(outs, out_dir / "nv_relaxometry.png", title)
        if png is not None:
            log.info("figure written to %s", png)
    log.info("outputs in %s (%.1fs)", out_dir, summary["wall_seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
