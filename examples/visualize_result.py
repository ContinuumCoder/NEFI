"""Standard figures for a saved :class:`nefi.solve.Result` (e.g. after a big run on a GPU server).

Usage::

    # a directory written by `nefi run` (result.pt + config.yaml): GT and measurement are
    # regenerated deterministically from the saved instance config, seed and scene
    python examples/visualize_result.py runs/nv_relaxometry_paper --html

    # a bare result file, with optional ground truth / measurement files (torch.save'd tensors
    # or {name: tensor} dicts / Measurement objects)
    python examples/visualize_result.py runs/x/result.pt --gt runs/x/gt.pt --out runs/x/figs

    # a bare result file, regenerating GT from an instance (+ optional YAML config)
    python examples/visualize_result.py runs/x/result.pt --instance thermal_tomography \\
        --config configs/thermal_tomography_paper.yaml --seed 0

Figures (under ``--out``, default ``<run dir>/figures``): ``compare`` (measurement | GT |
reconstruction | signed error, one per shared field; 3-D fields: depth mosaic with the GT
outlined, cross-sections through the anomaly and a projection — ``--panel`` chooses the extra
panel: error, overlay or profile), ``field`` (1-D line / 2-D image / 3-D depth mosaic,
orthogonal slices through the GT anomaly centroid, projections and GT vs reconstruction
isosurfaces — GT at its threshold rule, reconstruction at the volume-matched level),
``history``, ``stages``, ``wallclock``, ``measurement`` and ``fit`` (when the measurement is
known), ``metrics.json`` and, with ``--html``, a self-contained ``report.html``: for 3-D results
it embeds a drag-to-rotate viewer (isosurfaces / voxels / slices, GT vs reconstruction, one
camera; also written as ``viewer.html``). ``--interactive plotly`` uses CDN-backed plotly
isosurfaces instead (not self-contained), ``--interactive none`` skips the viewer. The
instance's display hints (``viz_hints``: field transforms, labels, colormaps) are applied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import torch  # noqa: E402

import nefi  # noqa: E402
from nefi import viz  # noqa: E402
from nefi.config import load_config  # noqa: E402
from nefi.solve.result import Result  # noqa: E402
from nefi.viz.report import Section, write_json  # noqa: E402


def _load_tensor_dict(path: str | None, key: str = "x") -> Any:
    if not path:
        return None
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(obj):
        return {key: obj}
    return obj


def _spec_from_run_dir(d: Path) -> tuple[str | None, dict, dict]:
    """(instance, config, run section) from a `nefi run` directory's config.yaml."""
    p = d / "config.yaml"
    if not p.exists():
        return None, {}, {}
    raw = load_config(p)
    inst = raw.get("instance")
    cfg: dict = {}
    if isinstance(inst, dict):
        cfg = {k: v for k, v in inst.items() if k != "type"}
        inst = inst.get("type")
    cfg.update(raw.get("config") or {})
    return (str(inst) if inst else None), cfg, dict(raw.get("run") or {})


def _regenerate(instance: str, cfg: dict, seed: int, scene: str | None) -> tuple[Any, Any, Any]:
    from nefi.registry import get

    inst = get("instance", instance)(dict(cfg)) if cfg else get("instance", instance)()
    gt, meas = inst.make_measurement(seed=seed, scene_class=scene)
    return inst, gt, meas


def volume_viewer(args, out: Path, inst, field, g3, rec, domain, axis, center) -> str:
    """The interactive 3-D viewer of a result: ``viewer.html`` (standalone) and the fragment
    embedded in ``report.html`` (GT vs reconstruction; GT threshold = the instance's IoU rule)."""
    from nefi.viz.fields import to_numpy
    from nefi.viz.isosurface import hinted_transform

    vols = {"reconstruction": rec} if g3 is None else {"ground truth": g3, "reconstruction": rec}
    ref = to_numpy(g3 if g3 is not None else rec)
    opts = {
        "extent": domain,
        "threshold": viz.voxel_rule(inst, ref, field=field),
        "field": field,
        "axis": axis,
        "center": center,
        "transform": hinted_transform(viz.instance_hints(inst), field),
        "title": f"{field}: ground truth vs reconstruction" if g3 is not None else field,
    }
    try:
        if args.interactive == "plotly":
            frag = viz.plotly_volume_html(vols, **opts)  # loads plotly.js from its CDN
            viz.save_volume_viewer(out / "viewer.html", vols, kind="plotly", **opts)
        else:
            frag = viz.volume_viewer_html(vols, mode="iso", **opts)
            viz.save_volume_viewer(out / "viewer.html", vols, mode="iso", **opts)
        print(f"  ✓ viewer ({len(frag.encode()) / 1024:.0f} kB)")
        return frag
    except Exception as e:  # the static figures carry the same content
        print(f"  ✗ viewer: {type(e).__name__}: {e}")
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("result", help="result.pt or a `nefi run` output directory")
    ap.add_argument("--gt", default=None, help="torch file with GT tensor / {name: tensor}")
    ap.add_argument("--measurement", default=None, help="torch file with a Measurement / tensor")
    ap.add_argument("--instance", default=None, help="regenerate GT / measurement from this")
    ap.add_argument("--config", default=None, help="YAML config of the instance")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--field", default=None, help="field to compare (default: primary)")
    ap.add_argument(
        "--panel",
        default=None,
        choices=["error", "overlay", "profile"],
        help="extra compare panel (default: error; 3-D: overlay; or the instance's hint)",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--formats", default="png")
    ap.add_argument("--dpi", type=float, default=130)
    ap.add_argument(
        "--interactive",
        default="canvas",
        choices=["canvas", "plotly", "none"],
        help="3-D viewer in report.html / viewer.html: canvas (self-contained, default), plotly "
        "(loads plotly.js from its CDN) or none",
    )
    args = ap.parse_args()
    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())

    src = Path(args.result)
    run_dir = src if src.is_dir() else src.parent
    result_path = src / "result.pt" if src.is_dir() else src
    result = Result.load(result_path)
    out = Path(args.out) if args.out else run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    print(f"loaded {result_path}\n{result.summary()}")

    instance_name, cfg, run = _spec_from_run_dir(run_dir)
    if args.instance:
        instance_name = args.instance
    if args.config:
        raw = load_config(args.config)
        sec = raw.get("instance")
        cfg = {k: v for k, v in sec.items() if k != "type"} if isinstance(sec, dict) else {}
        cfg.update(raw.get("config") or {})
    seed = args.seed if args.seed is not None else int(run.get("seed") or 0)
    scene = args.scene or run.get("scene")

    inst, gt, meas = None, _load_tensor_dict(args.gt), None
    if args.measurement:
        m = torch.load(args.measurement, map_location="cpu", weights_only=False)
        meas = m if hasattr(m, "data") else nefi.Measurement(torch.as_tensor(m))
    if instance_name and (gt is None or meas is None):
        print(f"regenerating ground truth / measurement: {instance_name} seed={seed} scene={scene}")
        try:
            inst, gt_new, meas_new = _regenerate(instance_name, cfg, seed, scene)
            gt = gt if gt is not None else gt_new
            meas = meas if meas is not None else meas_new
        except Exception as e:
            print(f"  could not regenerate ({type(e).__name__}: {e}); continuing without GT")

    field = args.field or next(
        (k for k in result.fields if gt is None or k in gt), next(iter(result.fields))
    )
    domain = inst.domain() if inst is not None else None
    field_shape = tuple(result.fields[field].shape)
    hints = viz.instance_hints(inst)
    ndim = len([s for s in field_shape if s > 1])
    extras = (args.panel,) if args.panel else viz.hints.compare_extras(hints, ndim)
    mask = (
        viz.multiphysics.meta_mask(meas, [s for s in field_shape if s > 1])
        if meas is not None
        else None
    )
    saved: dict[str, Path] = {}
    viewer_html = ""

    def save(key: str, make) -> None:
        try:
            saved[key] = viz.savefig(make(), out / key, formats, args.dpi)[0]
            print(f"  ✓ {key}")
        except Exception as e:
            print(f"  ✗ {key}: {type(e).__name__}: {e}")

    viz.set_default_style(dark=args.dark)
    metrics: dict[str, float] = {}
    if gt is not None:
        if inst is not None:
            try:
                metrics = {k: float(v) for k, v in inst.evaluate(result, gt).items()}
            except Exception as e:
                print(f"  instance metrics failed ({e}); using PSNR / SSIM")
        if not metrics and field in gt:
            metrics = viz.field_metrics(result.fields[field], gt[field])
        for k in [k for k in result.fields if k in gt]:
            save(
                "compare" if k == field else f"compare_{k}",
                lambda k=k: viz.compare_fields(
                    gt,
                    {"reconstruction": result},
                    meas if k == field else None,
                    field=k,
                    domain=domain,
                    instance=inst,
                    panels=("measurement", "gt", "recon", *extras),
                    mask=mask if k == field else None,
                    metrics={"reconstruction": dict(list(metrics.items())[:2])}
                    if k == field and metrics
                    else "auto",
                ),
            )
    rec = result.fields[field].squeeze()
    if rec.ndim == 3:
        ax_v = viz.hints.volume_axis(hints)
        g3 = gt[field].squeeze() if (gt is not None and field in gt) else None
        g3 = g3 if (g3 is not None and tuple(g3.shape) == tuple(rec.shape)) else None
        center = viz.anomaly_centroid(g3 if g3 is not None else rec, mask=mask)
        save(
            "field_mosaic",
            lambda: viz.depth_mosaic(rec, field=field, axis=ax_v, domain=domain, mask=mask),
        )
        save(
            "field_orthoslices",
            lambda: viz.orthoslices(rec, field=field, index=center, domain=domain),
        )
        save("field_projections", lambda: viz.projections(rec, field=field, domain=domain))
        if g3 is not None:
            save(
                "field_isosurfaces",
                lambda: viz.iso_compare(
                    {field: g3},
                    {"reconstruction": rec},
                    domain=domain,
                    instance=inst,
                    mask=mask,
                    axis=ax_v,
                ),
            )
        else:
            save("field_voxels", lambda: viz.voxel_view(rec, field=field, domain=domain))
        if args.interactive != "none":
            viewer_html = volume_viewer(args, out, inst, field, g3, rec, domain, ax_v, center)
    else:
        save("field", lambda: viz.plot_field(result, field=field, domain=domain))
    save("history", lambda: viz.plot_history(result))
    save("stages", lambda: viz.plot_stage_summary(result))
    save("wallclock", lambda: viz.plot_wallclock_breakdown({"run": result}))
    if meas is not None:
        save(
            "measurement",
            lambda: viz.plot_measurement(meas, field_shape=field_shape, instance=inst),
        )
        save(
            "fit",
            lambda: viz.plot_fit(result.pred, meas, field_shape=field_shape, instance=inst),
        )
        try:
            fit = viz.residual_stats(result.pred, meas)
        except Exception:
            fit = {}
    else:
        fit = {}
    write_json(
        out / "metrics.json",
        {
            "result": str(result_path),
            "field": field,
            "metrics": metrics,
            "data_fit": fit,
            "timing": result.timing,
            "config_hash": result.config_hash,
            "extra": result.extra,
        },
    )
    if args.html:
        sections = [
            Section(
                "Reconstruction",
                images=[
                    {"path": p, "caption": k, "wide": True}
                    for k, p in saved.items()
                    if k.startswith("compare") or k.startswith("field")
                ],
                table=[{"metric": k, "value": v} for k, v in metrics.items()] or None,
                html=viewer_html if args.html else "",
            ),
            Section(
                "Data fit",
                images=[saved[k] for k in ("measurement", "fit") if k in saved],
                table=[fit] if fit else None,
            ),
            Section(
                "Optimization",
                images=[saved[k] for k in ("history", "stages", "wallclock") if k in saved],
                code=result.summary(),
            ),
        ]
        html = viz.html_report(
            out,
            f"nefi result · {instance_name or result_path.stem}",
            sections,
            filename="report.html",
            config_hash=result.config_hash,
            subtitle=f"{result_path} · field `{field}`",
        )
        print(f"wrote {html}")
    print(f"figures in {out}")
    print(json.dumps({k: round(v, 5) for k, v in metrics.items()}, indent=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
