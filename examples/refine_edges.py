"""Benchmark the edge-refinement stage (``nefi.solve.refine``) on the volumetric instances.

For every instance (smoke preset, gallery budget) the script solves the instance's own neural-field
problem, then refines the smooth reconstruction with each mode under the *same* operator and data:

* ``continue``   — the control: the same grid continuation and budget without a sharpening prior
  (a smooth result stopped early improves under any continuation; credit a prior only for what it
  adds beyond this row);
* ``levelset``   — the instance defaults (``REFINE_DEFAULTS``: phase values from the config, free
  phases for structures of varying intensity, a neural φ for DOT);
* ``levelset-plain`` — a piecewise-constant level set on the grid (the instance's phase values);
* ``phasefield`` — the problem continued with a growing double-well penalty;
* ``tv_sharpen`` — the problem continued with stronger TV (+ a binarizing head swap if bracketed).

It prints one table (PSNR / SSIM / IoU at the GT threshold / volume-matched IoU / Edge-F1 / data
fit, smooth → refined, with the verdict of the acceptance test and timings), the data fit of the
ground truth itself under the inversion operator (χ_GT: the misfit a perfect reconstruction would
have — model error included), and saves central-slice side-by-sides (GT | smooth | refined, plus
line profiles through the anomaly) as PNGs.

Usage::

    python examples/refine_edges.py --budget 0.1 --device cpu --out runs/refine_edges
    python examples/refine_edges.py --instances dot3d --modes continue,levelset --seeds 0,1
    python examples/refine_edges.py --include-2d          # + eit, deconvolution

Every refinement runs with ``refuse=False`` so the numbers of refused candidates are shown too;
the ``verdict`` column is what ``refine_edges`` does by default (``refused``: the smooth result is
kept).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

import nefi
from nefi.bench.protocol import default_method
from nefi.solve.refine import data_fit, refine_defaults, refine_instance
from nefi.utils.seed import seed_everything
from nefi.viz._instances import budget_curriculum, resolve_instance

VOLUMETRIC = ("thermal_tomography", "ct3d", "deconvolution3d", "dot3d", "photoacoustic3d")
PLANAR = ("eit", "deconvolution")
MODES = ("continue", "levelset", "levelset-plain", "phasefield", "tv_sharpen")
#: keys of the instance defaults shared by every mode (phase values, GT threshold, PSNR range)
COMMON = ("levels", "iou_tau", "iou_above", "data_range")


def mode_kwargs(instance, label: str) -> dict:
    """Keyword arguments of :func:`refine_instance` for a benchmark row."""
    d = refine_defaults(instance)
    common = {k: d[k] for k in COMMON if k in d}
    if label == "levelset":
        return {**d, "mode": "levelset"}
    if label == "levelset-plain":  # constant phases on the grid, whatever the defaults say
        plain = {"free_phases": (), "representation": "grid", "perimeter": 0.0}
        return {**common, **plain, "learn_levels": True, "mode": "levelset"}
    return {**common, "mode": label}


def smooth_run(name: str, args, seed: int):
    """Generate → solve at the gallery budget.

    Returns ``(instance, gt, measurement, problem, result, info)``.
    """
    inst, info = resolve_instance(name, smoke=True)
    gt, meas = inst.make_measurement(seed=seed)
    seed_everything(seed)
    prep = default_method().prepare(inst, meas)
    cur, budget = budget_curriculum(
        prep.resolved_curriculum(), info, args.budget, max_steps=args.max_steps or None
    )
    t0 = time.perf_counter()
    res = prep.run(cur, device=args.device, seed=seed)
    return (
        inst,
        gt,
        meas,
        prep.problem,
        res,
        {"steps": budget["steps"], "solve_s": time.perf_counter() - t0},
    )


def gt_fit(problem, gt, meas) -> dict:
    """Data fit of the ground truth under the inversion operator (noise + model error)."""
    key = problem.field.primary
    g = torch.as_tensor(gt[key]).float()
    with torch.no_grad():
        pred = problem.operator.to("cpu", torch.float32).at_resolution(tuple(g.shape))({key: g})
    return data_fit(pred, problem.measurement_at(tuple(g.shape)))


def misfit(fit: dict) -> tuple[float, str]:
    return (fit["chi"], "χ") if fit.get("chi") is not None else (fit["rmse"], "RMSE")


def arrow(a: float | None, b: float | None, digits: int = 3) -> str:
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)):
        return "—"
    return f"{a:.{digits}g} → {b:.{digits}g}"


def central_slice(g: torch.Tensor) -> int:
    """Depth slice through the strongest GT anomaly (largest difference from the median)."""
    dev = (g - g.median()).abs()
    return int(dev.amax(dim=tuple(range(g.ndim - 1))).argmax())


def save_figure(name, inst, gt, smooth, candidates, path: Path) -> str:
    """GT | smooth | refined side-by-side (``nefi.viz.compare_fields``, matplotlib fallback)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = next(iter(gt))
    g = torch.as_tensor(gt[key]).float()
    recons = {"smooth": smooth, **candidates}
    try:
        from nefi import viz

        kw = {"panels": ("gt", "recon", "profile"), "instance": inst, "title": name}
        if g.ndim == 3:
            kw.update(kind="3d", slices=[central_slice(g)], domain=inst.domain())
        fig = viz.compare_fields(gt, recons, **kw)
        how = "nefi.viz.compare_fields"
    except Exception as e:  # the viz layer is optional here: plain matplotlib
        print(f"  compare_fields failed ({type(e).__name__}: {e}); using matplotlib")
        k = central_slice(g) if g.ndim == 3 else None
        cut = (lambda t: t[..., k]) if k is not None else (lambda t: t)
        panels = {"ground truth": g, **{m: r.fields[key].float() for m, r in recons.items()}}
        fig, axes = plt.subplots(1, len(panels), figsize=(2.4 * len(panels), 2.6))
        lo, hi = float(g.min()), float(g.max())
        for ax, (lab, x) in zip(axes, panels.items()):
            ax.imshow(cut(x).T.numpy(), origin="lower", vmin=lo, vmax=hi, cmap="magma")
            ax.set_title(lab, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{name}" + (f" (slice z={k})" if k is not None else ""), fontsize=9)
        how = "matplotlib"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return how


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", type=float, default=0.1, help="fraction of the default budget")
    ap.add_argument("--max-steps", type=int, default=600, help="per-run step cap (0 = none)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--instances", default=",".join(VOLUMETRIC))
    ap.add_argument("--include-2d", action="store_true", help="also eit and deconvolution")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--seeds", default="0", help="comma-separated data / optimization seeds")
    ap.add_argument("--out", default="runs/refine_edges")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress lines also when redirected to a file
    names = [s.strip() for s in args.instances.split(",") if s.strip()]
    if args.include_2d:
        names += [n for n in PLANAR if n not in names]
    modes = [s.strip() for s in args.modes.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"nefi {nefi.__version__} · edge refinement · budget ×{args.budget:g} · {args.device}")
    rows, reports = [], []
    for name in names:
        for seed in seeds:
            inst, gt, meas, problem, res, info = smooth_run(name, args, seed)
            ref_fit = gt_fit(problem, gt, meas)
            m_gt, lab = misfit(ref_fit)
            print(
                f"\n## {name} (seed {seed}): smooth solve {info['steps']} steps, "
                f"{info['solve_s']:.1f} s · {lab} of the ground truth itself: {m_gt:.3g}"
            )
            candidates = {}
            for mode in modes:
                t0 = time.perf_counter()
                try:
                    _, rep = refine_instance(
                        inst, res, meas, gt=gt, problem=problem, refuse=False,
                        device=args.device, seed=seed, **mode_kwargs(inst, mode),
                    )  # fmt: skip
                except Exception as e:  # report and continue with the next mode
                    print(f"  ✗ {mode:<15} {type(e).__name__}: {e}")
                    rows.append({"instance": name, "seed": seed, "mode": mode, "error": str(e)})
                    continue
                # the acceptance test runs the continuation control internally when the smooth
                # start is not at the noise floor: its time is reported separately
                ctrl_s = float(rep.settings.get("control_seconds", 0.0))
                dt = time.perf_counter() - t0 - ctrl_s
                mb, ma = rep.metrics_before, rep.metrics_after
                fb, fa = misfit(rep.fit_before)[0], misfit(rep.fit_after)[0]
                row = {
                    "instance": name,
                    "seed": seed,
                    "mode": mode,
                    "verdict": "accepted" if rep.accepted else "refused",
                    "fit_measure": lab,
                    "fit_before": fb,
                    "fit_after": fa,
                    "fit_gt": m_gt,
                    "smooth_steps": info["steps"],
                    "smooth_s": info["solve_s"],
                    "refine_s": dt,
                    "control_s": ctrl_s,
                    "refine_steps": rep.steps,
                    "fit_control": None if rep.fit_control is None else misfit(rep.fit_control)[0],
                    "sharpening_cost": rep.sharpening_cost,
                    "band_deviation": rep.band_deviation,
                    "levels": rep.levels_final,
                    "free_phases": rep.free_phases,
                    "reason": rep.reason,
                }
                for k in ("psnr", "ssim", "iou", "iou_vm", "edge_f1"):
                    row[f"{k}_before"], row[f"{k}_after"] = mb[k], ma[k]
                rows.append(row)
                reports.append(f"## {name} · seed {seed} · {mode}\n\n" + rep.to_markdown())
                candidates[f"{mode} ({row['verdict']})"] = rep.candidate
                mark = "✓" if rep.accepted else "✗"
                print(
                    f"  {mark} {mode:<15} PSNR {arrow(mb['psnr'], ma['psnr'], 4)}"
                    f"  IoU {arrow(mb['iou'], ma['iou'])}"
                    f"  IoU_vm {arrow(mb['iou_vm'], ma['iou_vm'])}"
                    f"  Edge-F1 {arrow(mb['edge_f1'], ma['edge_f1'])}  {lab} {arrow(fb, fa)}"
                    f"  {dt:.1f} s"
                )
            if not args.no_plots and candidates:
                p = out / f"{name}_seed{seed}.png"
                how = save_figure(name, inst, gt, res, candidates, p)
                print(f"  figure: {p} ({how})")
    # ---- table -----------------------------------------------------------------------------------
    head = [
        "instance", "seed", "mode", "verdict", "PSNR [dB]", "SSIM", "IoU (GT τ)",
        "IoU (vol.-matched)", "Edge-F1", "data fit", "fit of GT", "smooth s", "refine s",
    ]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['instance']} | {r['seed']} | {r['mode']} | error: {r['error']} |")
            continue
        cells = [
            r["instance"], str(r["seed"]), r["mode"], r["verdict"],
            arrow(r["psnr_before"], r["psnr_after"], 4), arrow(r["ssim_before"], r["ssim_after"]),
            arrow(r["iou_before"], r["iou_after"]), arrow(r["iou_vm_before"], r["iou_vm_after"]),
            arrow(r["edge_f1_before"], r["edge_f1_after"]),
            f"{r['fit_measure']} {arrow(r['fit_before'], r['fit_after'])}",
            f"{r['fit_gt']:.3g}", f"{r['smooth_s']:.1f}", f"{r['refine_s']:.1f}",
        ]  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print("\n" + table)
    (out / "refine_edges.md").write_text(
        "# Edge refinement benchmark\n\n"
        f"Budget ×{args.budget:g} (smoke presets), device {args.device}; `data fit` is χ = RMSE/σ "
        "(RMSE when σ is unknown), `fit of GT` the same measure for the ground truth under the "
        "inversion operator; `refine s` excludes the acceptance test's internal control run.\n\n"
        + table
        + "\n\n"
        + "\n".join(reports)
    )
    (out / "refine_edges.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {out / 'refine_edges.md'} and {out / 'refine_edges.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
