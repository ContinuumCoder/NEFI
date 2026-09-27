"""Build the nefi demo gallery and bundle it into one shareable report.

Usage::

    python examples/gallery.py --budget 0.15 --device cpu --out runs/gallery [--instances a,b,c] \\
        [--html] [--max-steps 600] [--time-budget 120] [--dark] [--no-perf] [--no-baselines] \\
        [--no-interactive | --interactive plotly] [--no-refine] [--steps-3d-multiplier 2]

What it produces under ``--out``:

1. ``physics/`` — :func:`nefi.viz.physics_gallery`: one smoke run per registered instance,
   ``gallery.png`` (one tile per 1-D / 2-D system: GT | measurement | reconstruction; one
   full-width row per 3-D system: GT depth mosaic | measurement | reconstruction mosaic with the
   GT outlined | cross-sections through the anomaly | projection), ``gallery_3d.png`` (larger
   mosaics, contrast contours and shaded isosurfaces — GT at its threshold rule,
   reconstruction at the volume-matched level — for every 3-D system), and per instance
   ``compare`` (signed error; 3-D: GT overlay on the mosaic), ``measurement``, ``fit``,
   ``history``, ``stages``, ``multiscale``, 3-D ``mosaic`` / ``voxels`` (isosurfaces) and
   ``evolution.gif`` (3-D: the depth mosaic); ``manifest.json`` with metrics and timings. Each
   3-D system also gets an interactive viewer (``<name>/viewer.html`` standalone,
   ``viewer.json`` embedded in ``index.html`` under the system's static row ``block3d.png``;
   ``--no-interactive`` skips it, ``--interactive plotly`` uses CDN-backed plotly isosurfaces
   instead — not self-contained).
2. training dynamics — history (stages, annealing windows, β/K, LR, noise floor), multiscale
   view and snapshot animation of toy1d, one 2-D and one 3-D instance (reused from
   ``physics/``).
3. ``baselines/`` — the instance's own method vs its ``baselines()`` at a small budget
   (deconvolution, else toy1d) and the failure-mode panel.
4. ``performance/`` — step time per instance (grid shapes in the labels), scaling with grid size
   (log-log slopes; thermal tomography as the 3-D series), memory by gradient mode (thermal
   tomography: adjoint vs autograd vs checkpoint), speedup vs eager per instance when
   ``tools/profile_instance.py --json`` profiles exist in ``--perf-profiles`` (default
   ``runs/_perf``), wall-clock breakdown and a benchmark CI plot (toy1d neural vs grid); rows in
   ``performance.json``.
5. ``report.md`` and, with ``--html``, a self-contained ``index.html`` (figures embedded; the
   3-D systems section interleaves each static row with its drag-to-rotate viewer, plain
   JavaScript, no CDN).

Budgets: ``--budget`` scales each instance's *default* step budget; problem sizes come from the
smoke presets (``nefi run --smoke``); ``--max-steps`` caps every run. 3-D systems are
under-converged at that rule, so from ``--budget 0.2`` on they get at least
``--steps-3d-multiplier`` (default 2) × their preset's steps (above the cap); the step counts are
printed.

Edge refinement (``--refine``, default on; ``--no-refine``): after each 3-D system's smooth solve,
``nefi.solve.refine_run_output`` sharpens its interfaces under the same operator and data
(``docs/refinement.md``); the refined field is a third column (GT | smooth | refined) in the 3-D
rows of ``gallery.png``, ``gallery_3d.png``, ``compare`` / ``voxels`` and the viewer, with IoU /
Edge-F1 and the data fit before → after in the titles. A refused refinement keeps the smooth
result (its candidate is drawn labelled "refused"); headline metrics stay the smooth ones.

Paper-scale figures are produced on the CUDA servers with ``--device cuda --budget 1.0
--max-steps 0`` (no cap).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import nefi  # noqa: E402
from nefi import viz  # noqa: E402
from nefi.viz._instances import registered_instances, resolve_instance  # noqa: E402
from nefi.viz.report import Section, write_json  # noqa: E402
from nefi.viz.style import format_metric  # noqa: E402


def _print_entry(e: viz.GalleryEntry) -> None:
    t = e.timing
    if e.ok:
        hl = e.headline()
        metric = format_metric(*hl) if hl else "—"
        print(
            f"  ✓ {e.name:<24} {metric:<18} {e.steps:>5} steps  solve {t.get('solve_s', 0):6.2f}s"
            f"  data {t.get('generate_s', 0):5.2f}s  figures {t.get('plots_s', 0):5.1f}s"
        )
        b = e.budget
        if b.get("steps_3d_floor"):
            print(
                f"      3-D budget: {b['steps']} steps (preset {b['preset_steps']} × "
                f"{b['steps_3d_multiplier']:g} = floor {b['steps_3d_floor']}; plain rule "
                f"{b['budget_scale']:g} × {b['reference_steps']})"
            )
        if e.refine:
            cap = e.refine.get("summary") or e.refine.get("error")
            print(f"      {cap}  [{t.get('refine_s', 0):.1f}s]")
    else:
        print(f"  ✗ {e.name:<24} {e.error}")
    sys.stdout.flush()


def _save(fig, path: Path, args: argparse.Namespace) -> Path:
    return viz.savefig(fig, path, formats=args.formats, dpi=args.dpi)[0]


def print_volumes(manifest: viz.GalleryManifest) -> None:
    """IoU / Dice at the GT threshold vs the volume-matched level, and the viewer sizes."""
    from nefi.viz.multiphysics import iso_rows

    rows = iso_rows(manifest)
    if not rows:
        return
    print("\n## 3-D systems: reconstruction at the GT threshold → at the volume-matched level")
    for r in rows:
        f = {k: (v if isinstance(v, float) else float("nan")) for k, v in r.items()}
        print(
            f"  {r['system']:<18} {r['GT rule']:<26} IoU {f['IoU at GT threshold']:.3f} → "
            f"{f['IoU matched']:.3f}   Dice {f['Dice at GT threshold']:.3f} → "
            f"{f['Dice matched']:.3f}   (Otsu IoU {f['IoU Otsu']:.3f})"
        )
    for e in manifest.entries:
        v = e.viewer or {}
        if v.get("kind") == "canvas":
            print(
                f"  viewer {e.name:<18} {v['fragment_bytes'] / 1024:6.1f} kB embedded "
                f"(grid {'×'.join(map(str, v['shape']))}, meshes {v.get('mesh_faces')} faces) "
                f"· standalone {v['html']}"
            )
        elif v:
            print(f"  viewer {e.name:<18} plotly (CDN) · standalone {v['html']}")


def relative_budget(name: str, budget: float, max_steps: int | None) -> float:
    """The gallery's budget rule expressed as a factor on the instance's smoke curriculum."""
    from nefi.viz._instances import budget_curriculum

    inst, info = resolve_instance(name)
    _, b = budget_curriculum(inst.default_curriculum(), info, budget, max_steps=max_steps)
    return b["steps"] / max(1, b["preset_steps"])


def training_dynamics(manifest: viz.GalleryManifest, out: Path) -> Section:
    """Section pointing at the per-instance training figures of two representative systems."""
    ok = [e for e in manifest.entries if e.ok]
    picks = [e for e in ok if e.name == "toy1d"]
    two_d = [
        e
        for e in ok
        if e.name != "toy1d"
        and len(next(iter(e.fields.values()), [])) == 2
        and len(e.budget.get("stages", [])) > 1
    ]
    pref = [e for e in two_d if e.name in ("deconvolution", "nv_relaxometry", "sparse_view_ct")]
    picks += (pref or two_d)[:1]
    picks += [e for e in ok if e.name in manifest.get("volumetric", [])][:1]  # a 3-D system
    subs = []
    for e in picks:
        figs = e.figures
        imgs = [
            {"path": out / figs[k], "caption": cap, "wide": k == "history"}
            for k, cap in (
                ("history", "loss components, stage boundaries, annealing windows, β/K and LR"),
                ("multiscale", "end-of-stage fields at each stage's resolution"),
                ("stages", "steps, seconds and final data loss per stage"),
                ("evolution", "field snapshots during optimization (3-D: depth mosaic vs GT)"),
            )
            if k in figs
        ]
        subs.append(Section(e.name, images=imgs))
    return Section(
        "Training dynamics",
        "Loss curves are drawn on a log scale with one categorical color per component; vertical "
        "rules mark curriculum stages, shaded spans the frequency-annealing ramp (β < K). The "
        "horizontal σ² line is the expected MSE data loss at the truth (shown when the data term "
        "is a plain MSE and the noise level is known).",
        subsections=subs,
    )


def baselines_section(args: argparse.Namespace, out: Path, available: list[str]) -> Section:
    pick = "deconvolution" if "deconvolution" in available else "toy1d"
    scale = relative_budget(pick, args.budget, args.max_steps or None)
    print(f"\n## baselines panel ({pick}, ×{scale:.2g} of the smoke budget)")
    t0 = time.perf_counter()
    fig, runs = viz.baselines_panel(
        pick,
        budget_scale=scale,
        device=args.device,
        seed=args.seed,
        max_steps=args.max_steps or None,
    )
    d = out / "baselines"
    p_panel = _save(fig, d / "panel", args)
    rows = []
    for name, r in runs.items():
        row = {"method": name, "status": "ok" if not r["error"] else "failed"}
        row.update({k: v for k, v in (r["metrics"] or {}).items()})
        row.update({"time s": r["time_s"], "steps": r["steps"], "error": r["error"]})
        rows.append(row)
        status = "✓" if not r["error"] else "✗"
        t_txt = f"{r['time_s']:.2f}s" if r["time_s"] is not None else "-"
        print(f"  {status} {name:<16} {t_txt:>8}  {r['error'] or ''}")
    imgs = [{"path": p_panel, "caption": "methods at a small budget", "wide": True}]
    ok = {n: r["result"] for n, r in runs.items() if r["result"] is not None}
    if ok:
        try:
            gt, _ = resolve_instance(pick)[0].make_measurement(seed=args.seed)
            p_fail = _save(viz.failure_modes(gt, ok), d / "failure_modes", args)
            imgs.append(
                {
                    "path": p_fail,
                    "caption": "leakage outside the support and error spectra (cross artifacts)",
                    "wide": True,
                }
            )
        except Exception as e:  # the panel above already carries the main message
            print(f"  failure-mode panel skipped: {type(e).__name__}: {e}")
    write_json(d / "runs.json", rows)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    return Section(
        f"Baselines ({pick})",
        "The instance's own neural-field method next to every baseline it declares "
        "(`Instance.baselines()`), on the same measurement and at the same budget scale. The "
        "failure-mode panel reports the mass outside the ground-truth support (leakage) and the "
        "**axis excess** of the error spectrum — how much error power sits on the Fourier axes "
        "relative to an isotropic error (≈ 1); cross / streak artifacts raise it.",
        images=imgs,
        table=rows,
    )


def speedup_image(args: argparse.Namespace, d: Path) -> dict | None:
    """ "Speedup vs eager" bars from ``tools/profile_instance.py --json`` profiles (``None`` —
    silently — when there are none)."""
    rows = viz.load_profiles(args.perf_profiles) if args.perf_profiles else []
    speed = viz.profile_speedups(rows)
    if not speed:
        return None
    for r in speed:
        print(
            f"  » {r['instance']:<24} {r['variant']:<34} ×{r['speedup']:.2f}"
            f"  ({r['eager_ms']:.2f} → {r['ms']:.2f} ms/step, {r['threads']} thr)"
        )
    return {
        "path": _save(viz.plot_speedup(speed), d / "speedup", args),
        "caption": f"speedup vs eager per instance from the {len(rows)} profiles in "
        f"`{args.perf_profiles}` (tools/profile_instance.py --json): eager ms / variant ms, "
        "same instance, device and thread count",
        "wide": True,
    }


def performance_section(
    args: argparse.Namespace, out: Path, manifest: viz.GalleryManifest
) -> Section:
    d = out / "performance"
    ok = [e.name for e in manifest.entries if e.ok]
    print(f"\n## performance dashboard ({args.device}, {args.perf_steps} steps per instance)")
    t0 = time.perf_counter()
    rows = viz.collect_performance(ok, args.device, steps=args.perf_steps, warmup=2)
    for r in rows:
        if r.get("error"):
            print(f"  ✗ {r['instance']:<24} {r['error']}")
        else:
            print(
                f"  ✓ {r['instance']:<24} {r['ms_per_step']:8.2f} ms/step  (fwd {r['fwd_ms']:.2f}"
                f" / bwd {r['bwd_ms']:.2f})  saved {r.get('saved_mb') or 0:.2f} MB"
            )
    imgs = [
        {
            "path": _save(viz.plot_step_time(rows), d / "step_time", args),
            "caption": "ms per optimization step at the finest smoke stage (grid in the labels; "
            "3-D systems have three extents)",
            "wide": True,
        }
    ]
    speed = speedup_image(args, d)
    if speed is not None:
        imgs.append(speed)
    memory: list[dict] = []
    if "thermal_tomography" in ok:
        memory = viz.collect_scaling(
            "thermal_tomography",
            [{"grid": [12, 12, 4]}, {"grid": [16, 16, 6]}],
            device=args.device,
            steps=max(2, args.perf_steps // 2),
            variants={m: {"grad_mode": m} for m in ("adjoint", "checkpoint", "autograd")},
        )
    scaling: list[dict] = []
    if "toy1d" in ok:
        scaling += viz.collect_scaling(
            "toy1d", [64, 128, 256, 512, 1024], key="n", device=args.device, steps=args.perf_steps
        )
    if "deconvolution" in ok:
        scaling += viz.collect_scaling(
            "deconvolution", [16, 32, 64], key="n", device=args.device, steps=args.perf_steps
        )
    # the 3-D series reuses the adjoint rows of the memory study (no extra profiling)
    scaling += [r for r in memory if r.get("mode") == "adjoint" and not r.get("error")]
    if scaling:
        imgs.append(
            {
                "path": _save(viz.plot_scaling(scaling), d / "scaling", args),
                "caption": "per-step time and memory vs grid points (log-log, fitted slopes; "
                "thermal_tomography = 3-D heat solver, adjoint gradients)",
                "wide": True,
            }
        )
    if memory:
        imgs.append(
            {
                "path": _save(viz.plot_memory(memory), d / "memory", args),
                "caption": "heat-solver gradient memory: discrete adjoint vs checkpointing vs "
                "unrolled autograd (NeFTY §4)",
            }
        )
    imgs.append(
        {
            "path": _save(viz.plot_wallclock_breakdown(manifest), d / "wallclock", args),
            "caption": "where the gallery's wall-clock went",
            "wide": True,
        }
    )
    bench_rows: list[dict] = []
    if "toy1d" in ok:
        from nefi.bench import run_benchmark

        inst = resolve_instance("toy1d")[0]
        bench = run_benchmark(
            inst,
            "neural,grid",
            n_samples=2,
            seeds=(0, 1),
            device=args.device,
            budget_scale=relative_budget("toy1d", args.budget, args.max_steps or None),
            progress=False,
            out_dir=d / "bench_toy1d",
        )
        bench_rows = bench.rows
        imgs.append(
            {
                "path": _save(viz.plot_bench_table(bench, "psnr"), d / "bench_psnr", args),
                "caption": "toy1d: neural field vs free grid, 2 samples × 2 seeds",
            }
        )
    write_json(
        d / "performance.json",
        {"step_time": rows, "scaling": scaling, "memory": memory, "bench": bench_rows},
    )
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    table = [
        {
            k: r.get(k)
            for k in (
                "instance",
                "shape",
                "params",
                "ms_per_step",
                "fwd_ms",
                "bwd_ms",
                "saved_mb",
                "peak_mem_mb",
                "error",
            )
        }
        for r in rows
    ]
    return Section(
        "Performance",
        f"Per-step cost of each instance's optimization loop on `{args.device}` (smoke-sized "
        "problems, finest curriculum stage; forward = field + operator + losses). Memory is the "
        "peak allocated CUDA memory on GPUs and the autograd saved-tensor volume of one step "
        "elsewhere (a device-agnostic proxy that separates adjoint from unrolled gradients).",
        images=imgs,
        table=table,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", type=float, default=0.15, help="fraction of the default budget")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/gallery")
    ap.add_argument("--instances", default=None, help="comma-separated (default: all)")
    ap.add_argument("--html", action="store_true", help="write a self-contained index.html")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=600, help="per-run step cap (0 = none)")
    ap.add_argument("--time-budget", type=float, default=120.0, help="per-instance solve cap [s]")
    ap.add_argument("--ncols", type=int, default=3)
    ap.add_argument("--dpi", type=float, default=130)
    ap.add_argument("--formats", default="png", help="e.g. png,svg")
    ap.add_argument("--dark", action="store_true", help="dark theme")
    ap.add_argument("--no-anim", action="store_true", help="skip the snapshot GIFs")
    ap.add_argument("--no-perf", action="store_true", help="skip the performance dashboard")
    ap.add_argument("--no-baselines", action="store_true", help="skip the baselines panel")
    ap.add_argument("--perf-steps", type=int, default=8)
    ap.add_argument(
        "--interactive",
        default="canvas",
        choices=["canvas", "plotly", "none"],
        help="3-D viewers: self-contained canvas viewer (default), plotly (loads plotly.js from "
        "its CDN: not self-contained) or none",
    )
    ap.add_argument(
        "--no-interactive", action="store_true", help="no interactive 3-D viewers (= none)"
    )
    ap.add_argument(
        "--refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="edge refinement of every 3-D system after its smooth solve (refined column; "
        "--no-refine to skip)",
    )
    ap.add_argument(
        "--steps-3d-multiplier",
        type=float,
        default=2.0,
        help="3-D systems get at least this × their smoke preset's steps when --budget >= 0.2 "
        "(0 = the plain budget rule)",
    )
    ap.add_argument(
        "--perf-profiles",
        default="runs/_perf",
        help="directory / files of tools/profile_instance.py --json profiles (speedup vs eager)",
    )
    args = ap.parse_args()
    args.formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    interactive = None if args.no_interactive or args.interactive == "none" else args.interactive
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    available = registered_instances()
    wanted = [s.strip() for s in args.instances.split(",")] if args.instances else available
    print(f"nefi {nefi.__version__} · gallery of {len(wanted)} instances · budget ×{args.budget:g}")
    rule3d = (
        f"3-D systems ≥ {args.steps_3d_multiplier:g} × preset steps"
        if args.steps_3d_multiplier and args.budget >= viz.multiphysics.STEPS_3D_MIN_BUDGET
        else "3-D systems: plain budget rule"
    )
    print(f"{rule3d} · edge refinement {'on' if args.refine else 'off'} (3-D)")
    print(f"device {args.device} · out {out}\n\n## physics gallery")
    viz.set_default_style(dark=args.dark)
    manifest = viz.physics_gallery(
        wanted,
        args.budget,
        args.device,
        out / "physics",
        seed=args.seed,
        max_steps=args.max_steps or None,
        time_budget_s=args.time_budget or None,
        animate=not args.no_anim,
        ncols=args.ncols,
        formats=args.formats,
        dpi=args.dpi,
        dark=args.dark,
        on_entry=_print_entry,
        interactive=interactive,
        refine=args.refine,
        steps_3d_multiplier=args.steps_3d_multiplier or None,
    )
    print_volumes(manifest)
    phys = out / "physics"
    sections = viz.gallery_sections(manifest, phys, interactive=False)  # static (Markdown)
    lead = [s for s in sections if s.id in ("overview", "volumes-3d")]  # gallery.png, gallery_3d
    per_instance = [s for s in sections if s not in lead]
    lead_html = lead
    if interactive:  # the 3-D systems section with a viewer under every static row
        live = viz.gallery_sections(manifest, phys, interactive=True)
        lead_html = [s for s in live if s.id in ("overview", "volumes-3d")]
    sections = [training_dynamics(manifest, phys)]
    if not args.no_baselines:
        try:
            sections.append(baselines_section(args, out, available))
        except Exception as e:
            print(f"  baselines panel failed: {type(e).__name__}: {e}")
    if not args.no_perf:
        try:
            sections.append(performance_section(args, out, manifest))
        except Exception as e:
            print(f"  performance dashboard failed: {type(e).__name__}: {e}")
    sections.append(
        Section(
            "Systems",
            "Per-instance figures: reconstruction vs ground truth (signed error and metrics; 3-D "
            "systems: depth mosaic with the GT outlined, cross-sections through the anomaly and "
            "a projection), the measurement in its natural layout, the data fit against the "
            "noise floor, the training curves and, for 3-D systems, the depth mosaic and the "
            "GT vs reconstruction isosurfaces (GT at its threshold rule, reconstruction at the "
            "volume-matched level).",
            subsections=per_instance,
        )
    )
    total = time.perf_counter() - t_start
    subtitle = (
        f"{manifest['n_ok']}/{len(manifest.entries)} physical systems · budget ×{args.budget:g}"
        f" · device {args.device} · {total:.0f} s end-to-end"
    )
    md = viz.markdown_report(out, "nefi demo gallery", [*lead, *sections], subtitle=subtitle)
    print(f"\nwrote {md}")
    if args.html:
        html = viz.html_report(
            out,
            "nefi demo gallery",
            [*lead_html, *sections],
            subtitle=subtitle,
            meta={"budget": f"×{args.budget:g}", "device": args.device, "seed": args.seed},
        )
        how = "loads plotly.js from its CDN" if interactive == "plotly" else "self-contained"
        print(f"wrote {html} ({html.stat().st_size / 1e6:.1f} MB, {how})")
    ok = [e.name for e in manifest.entries if e.ok]
    bad = [e.name for e in manifest.entries if not e.ok]
    print(f"\nrendered: {', '.join(ok) or '—'}")
    print(f"failed:   {', '.join(bad) or '—'}")
    print(f"total {total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
