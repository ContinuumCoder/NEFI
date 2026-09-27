"""Tests for :mod:`nefi.viz` (Agg backend, tiny sizes; the whole file runs in well under a minute).

Checks that figures are built with the expected panels, files are written within the size budget,
the multi-physics gallery survives failing instances and the HTML report is self-contained.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import nefi  # noqa: E402
from nefi import viz  # noqa: E402
from nefi.instances.toy1d import Toy1D  # noqa: E402
from nefi.measurement import Measurement  # noqa: E402
from nefi.solve.callbacks import FieldSnapshots  # noqa: E402
from nefi.solve.result import Result  # noqa: E402

KB = 1024


def image_axes(fig) -> list:
    """Axes that display an image (excludes colorbars and line plots)."""
    return [ax for ax in fig.axes if ax.images]


def sparse_2d(n: int = 24) -> torch.Tensor:
    x = torch.zeros(n, n)
    x[4, 5], x[15, 18], x[10, 9] = 1.0, 0.7, 0.5
    return x


@pytest.fixture(scope="module")
def toy_run():
    """A tiny toy1d inversion with stage and field snapshots (shared by several tests)."""
    inst = Toy1D(n=32, steps=(12, 12), hidden=16, depth=2, n_octaves=3)
    stage_snaps, field_snaps = viz.StageSnapshots(), FieldSnapshots(every=3)
    out = inst.run(seed=0, device="cpu", callbacks=[stage_snaps, field_snaps])
    return inst, out, stage_snaps, field_snaps


# ---------------------------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------------------------
def test_import_does_not_need_matplotlib():
    code = "import sys, nefi, nefi.viz; sys.exit(int('matplotlib' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_style_palette_cmaps_and_context():
    assert len(viz.PALETTE) == 8 and len(viz.PALETTE_DARK) == 8
    assert viz.cmap_for("rho").cmap == "magma"
    assert viz.cmap_for("alpha").cmap == "viridis"
    assert viz.cmap_for("sigma").cmap == "viridis"
    assert viz.cmap_for(quantity="error").kind == "diverging"
    assert viz.cmap_for("phase").cmap == "twilight"
    assert viz.cmap_for("unknown_field", np.array([-1.0, 1.0])).cmap == "RdBu_r"
    assert viz.quantity_of("rho_gt") == "density"
    before = matplotlib.rcParams["axes.facecolor"]
    with viz.use_style(dark=True) as th:
        assert th.name == "dark"
        assert matplotlib.rcParams["axes.facecolor"] == viz.DARK.surface
    assert matplotlib.rcParams["axes.facecolor"] == before
    w, h = viz.figsize(3, 2, cbar=1, title=True)
    assert w > h > 0


def test_savefig_formats_and_size_budget(tmp_path):
    fig = viz.plot_field(torch.rand(48, 48), field="rho")
    paths = viz.savefig(fig, tmp_path / "noise.png", formats=("png", "svg"), max_kb=40)
    assert [p.suffix for p in paths] == [".png", ".svg"]
    assert all(p.exists() for p in paths)
    assert paths[0].stat().st_size <= 40 * KB


# ---------------------------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------------------------
def test_compare_fields_2d_single_and_multi(tmp_path):
    gt = sparse_2d()
    rec = gt + 0.02 * torch.randn_like(gt)
    meas = Measurement(torch.rand(10, 24, 24), noise_std=0.01, meta={"freqs": list(range(10))})
    fig = viz.compare_fields({"rho": gt}, {"nefi": rec}, meas)
    assert len(image_axes(fig)) == 4  # measurement | GT | recon | error
    assert "PSNR" in fig.axes[2].get_title()
    fig2 = viz.compare_fields({"rho": gt}, {"a": rec, "b": 0.9 * rec}, meas, peaks=True)
    assert len(image_axes(fig2)) == 6  # meas, GT, 2 recons, 2 errors
    assert viz.savefig(fig2, tmp_path / "cmp")[0].stat().st_size <= 300 * KB


def test_compare_fields_1d_3d_and_complex():
    x = torch.linspace(0, 1, 64)
    gt = torch.exp(-((x - 0.5) ** 2) / 0.01)
    fig = viz.compare_fields(gt, {"nefi": gt + 0.05}, Measurement(gt + 0.01))
    assert len(fig.axes) == 2 and not image_axes(fig)  # values + signed error
    vol = torch.full((12, 12, 5), 0.15)
    vol[3:7, 4:8, 1:3] = 0.01
    frames = Measurement(torch.rand(6, 12, 12) + 1, meta={"frame_times": [1, 2, 3, 4, 5, 6]})
    fig3 = viz.compare_fields({"alpha": vol}, {"nefi": vol * 1.1}, frames, slices=3)
    # measurement + (GT, recon, error) rows × (3 slices + 2 cross-sections + 1 projection)
    assert len(image_axes(fig3)) == 1 + 3 * (3 + 2 + 1)
    c = torch.polar(torch.rand(16, 16) + 0.5, torch.rand(16, 16) * 6 - 3)
    figc = viz.compare_fields(c, {"nefi": c * 1.05})
    assert len(image_axes(figc)) == 6  # (GT, recon, |error|) × (magnitude, phase)
    assert viz.detect_kind(c) == "complex" and viz.detect_kind(vol) == "3d"


def test_3d_viewers():
    vol = torch.full((10, 10, 6), 0.15)
    vol[2:5, 3:6, 2:4] = 0.01
    assert len(image_axes(viz.depth_mosaic(vol, field="alpha", n_slices=4))) == 4
    assert len(image_axes(viz.orthoslices(vol, field="alpha"))) == 3
    proj = viz.projections(vol, field="alpha")
    assert len(image_axes(proj)) == 3 and "min" in image_axes(proj)[0].get_title()
    fig = viz.voxel_view(vol, field="alpha")
    assert any(ax.name == "3d" for ax in fig.axes)
    assert viz.representative_slice(vol.numpy()) in (2, 3)


# ---------------------------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("shape", "meta", "field_shape", "expected"),
    [
        ((64,), {}, (64,), "signal"),
        ((40,), {}, (16, 16), "vector"),
        ((16, 16), {}, (16, 16), "image"),
        ((20, 32), {}, (16, 16), "matrix"),
        ((20, 32), {"angles": [0.1] * 20}, (16, 16), "sinogram"),
        ((10, 16, 16), {"freqs": [1.0] * 10}, (16, 16), "spectra"),
        ((10, 16, 16), {"frame_times": [1.0] * 10}, (16, 16, 4), "frames"),
        ((10, 16, 16), {}, (16, 16), "stack"),
        ((3, 8, 50), {}, (16, 16), "traces"),
        ((16, 16), {"layout": "kspace"}, (16, 16), "kspace"),
        ((12, 10, 8), {}, (12, 10, 8), "volume"),
        ((9, 20, 8), {"angles": [0.1] * 9}, (16, 16, 8), "sinogram"),
    ],
)
def test_detect_layout(shape, meta, field_shape, expected):
    assert viz.detect_layout(torch.zeros(shape), meta, field_shape) == expected


def test_measurement_viewers(tmp_path):
    spec = Measurement(torch.rand(12, 16, 16), meta={"freqs": list(np.linspace(1, 3, 12))})
    assert len(image_axes(viz.plot_measurement(spec, field_shape=(16, 16)))) == 4
    frames = Measurement(torch.rand(15, 16, 16) + 1, meta={"frame_times": list(range(1, 16))})
    assert len(image_axes(viz.plot_measurement(frames, field_shape=(16, 16, 4)))) == 5
    assert len(image_axes(viz.plot_sinogram(torch.rand(10, 24)))) == 1
    assert len(viz.plot_traces(torch.randn(3, 12, 80)).axes) >= 2
    mask = (torch.rand(16, 16) < 0.1).float()
    pts = Measurement(torch.rand(16, 16), mask=mask, noise_std=0.01)
    view = viz.measurement_view(pts, field_shape=(16, 16))
    assert view.layout == "points" and view.kind == "scatter"
    assert len(image_axes(viz.plot_kspace(torch.rand(16, 16) < 0.3))) == 1
    # a stacked real/imag measurement resolved through the instance hints
    dt = Measurement(torch.randn(2, 8, 20))
    hinted = viz.measurement_view(dt, field_shape=(16, 16), hints={"complex_stack": "first"})
    assert hinted.layout == "complex" and hinted.data.shape == (8, 20)
    fig = viz.plot_measurement(dt, field_shape=(16, 16), hints=viz.INSTANCE_HINTS["wave_fwi"])
    assert "traces" in fig._suptitle.get_text()


def test_plot_fit_and_residual_stats():
    y = torch.sin(torch.linspace(0, 6, 128))
    meas = Measurement(y + 0.01 * torch.randn(128), noise_std=0.01)
    stats = viz.residual_stats(y, meas)
    assert 0.5 < stats["chi"] < 2.0
    fig = viz.plot_fit(y, meas)
    assert len(fig.axes) == 2 and "RMSE/σ" in fig._suptitle.get_text()
    img = torch.rand(16, 16)
    fig2 = viz.plot_fit(img + 0.01, Measurement(img, noise_std=0.01), field_shape=(16, 16))
    assert len(image_axes(fig2)) == 3
    stack = torch.rand(6, 16, 16)
    fig3 = viz.plot_fit(stack, Measurement(stack + 0.01, noise_std=0.01), field_shape=(16, 16))
    assert len(image_axes(fig3)) == 2  # data, prediction (+ per-slice RMS and histogram lines)


# ---------------------------------------------------------------------------------------------
# training dynamics
# ---------------------------------------------------------------------------------------------
def test_history_alignment_with_ragged_components():
    hist = {
        "global_step": list(range(6)),
        "stage": [0, 0, 0, 1, 1, 1],
        "total": [3.0, 2.0, 1.0, 0.9, 0.8, 0.7],
        "lr": [1e-3] * 6,
        "progress": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
        "data_loss": [3.0, 2.0, 1.0, 0.9, 0.8, 0.7],
        "log_mse": [2.9, 1.9, 0.9],  # only in stage 1 (weight 0 afterwards)
    }
    stages = [{"name": "s1", "final": {"log_mse": 0.9}}, {"name": "s2", "final": {"x": 1.0}}]
    res = Result({"x": torch.zeros(4)}, {"x": torch.zeros(4)}, torch.zeros(4), hist, stages)
    arr = viz.history_arrays(res)
    assert np.isnan(arr["log_mse"][3:]).all() and np.allclose(arr["log_mse"][:3], [2.9, 1.9, 0.9])
    fig = viz.plot_history(res, noise_std=0.1)
    assert len(fig.axes) == 3  # loss, β/K, learning rate


def test_training_figures_from_a_real_run(tmp_path, toy_run):
    _, out, stage_snaps, field_snaps = toy_run
    res = out.result
    assert len(stage_snaps.snapshots) == 2 and field_snaps.snapshots
    assert len(viz.plot_history(res).axes) == 3
    assert len(viz.plot_stage_summary(res).axes) == 3
    fig = viz.plot_multiscale(res, stage_snaps, gt=out.gt)
    assert len(fig.axes) == 2 + 1 + 1 + 1  # two stages, final, GT, loss panel
    gif = viz.animate_snapshots(field_snaps, tmp_path / "evo.gif", gt=out.gt["x"], result=res)
    assert gif.exists() and gif.suffix == ".gif" and gif.stat().st_size <= 300 * KB
    wall = viz.plot_wallclock_breakdown({"toy1d": res})
    assert wall.axes[0].patches  # stacked stage segments


# ---------------------------------------------------------------------------------------------
# qualitative
# ---------------------------------------------------------------------------------------------
def test_method_grid_and_failure_modes():
    gts = [sparse_2d(), sparse_2d().flip(0)]
    rows = [{"GT": g, "nefi": g * 0.9, "grid": g + 0.05 * torch.rand_like(g)} for g in gts]
    fig = viz.method_grid(rows, row_labels=["scene 1", "scene 2"], error=True)
    assert len(image_axes(fig)) == 2 * 3 + 2 * 2  # fields + error rows (no GT error)
    assert "PSNR" in image_axes(fig)[1].get_title()
    stats = viz.failure_stats(gts[0] * 0.9 + 0.02, gts[0])
    assert 0.0 < stats["leakage"] < 1.0 and stats["cross"] > 0.0
    streaks = torch.zeros(24, 24)
    streaks[:, 7], streaks[15, :] = 1.0, 1.0
    iso = viz.failure_stats(gts[0] + 0.1 * torch.randn(24, 24), gts[0])["cross"]
    assert viz.failure_stats(gts[0] + streaks, gts[0])["cross"] > 3 * iso
    ff = viz.failure_modes(gts[0], {"nefi": gts[0] * 0.9, "grid": gts[0] + 0.05})
    assert len(image_axes(ff)) == 2 * 4


def test_baselines_panel_toy1d():
    inst = Toy1D(n=32, steps=(8, 8), hidden=16, depth=2, n_octaves=3)
    fig, runs = viz.baselines_panel(inst, budget_scale=1.0, device="cpu", max_steps=16)
    assert set(runs) == {"neural", "grid"}
    assert all(r["error"] is None and r["steps"] > 0 for r in runs.values())
    assert "psnr" in runs["neural"]["metrics"]
    assert len(fig.axes) >= 4  # GT, measurement, two methods (+ error rows)


# ---------------------------------------------------------------------------------------------
# performance
# ---------------------------------------------------------------------------------------------
def test_performance_plots_and_row_loading(tmp_path):
    rows = [
        {"instance": "a", "device": "cpu", "shape": [n], "ms_per_step": 0.01 * n, "saved_mb": n}
        for n in (32, 64, 128, 256)
    ]
    rows += [{"instance": "b", "device": "cpu", "shape": "16x16", "ms_per_step": 5.0}]
    assert viz.loglog_slope([32, 64, 128, 256], [0.32, 0.64, 1.28, 2.56]) == pytest.approx(1.0)
    fig = viz.plot_scaling(rows[:4])
    assert "slope 1.00" in fig.axes[0].get_legend().get_texts()[0].get_text()
    assert viz.plot_step_time(rows).axes[0].patches
    mem = [
        {"instance": "heat", "shape": [8, 8, 4], "mode": m, "saved_mb": v}
        for m, v in (("adjoint", 1.0), ("autograd", 20.0))
    ]
    assert len(viz.plot_memory(mem).axes[0].patches) == 2
    bench = [
        {"method": m, "class": c, "psnr": p + d, "error": None}
        for m, p in (("neural", 30.0), ("grid", 25.0))
        for c in ("x", "y")
        for d in (0.0, 1.0)
    ]
    fig_b = viz.plot_bench_table(bench, "psnr")
    assert len(fig_b.axes) == 2  # one panel per class
    (tmp_path / "rows.json").write_text(json.dumps({"instance": "toy", "rows": bench}))
    loaded = viz.load_rows(tmp_path / "rows.json")
    assert len(loaded) == len(bench) and loaded[0]["instance"] == "toy"
    csv = 'method,class,psnr,shape\nneural,x,30,"[8, 8]"\ngrid,x,25,8x8\n'
    (tmp_path / "rows.csv").write_text(csv)
    loaded_csv = viz.load_rows(tmp_path / "rows.csv")
    assert loaded_csv[0]["psnr"] == 30.0 and loaded_csv[0]["n"] == 64
    assert loaded_csv[1]["shape"] == [8, 8]


def test_collect_performance_toy1d():
    rows = viz.collect_performance(["toy1d"], "cpu", steps=2, warmup=1)
    (r,) = rows
    assert r["error"] is None and r["instance"] == "toy1d"
    assert r["ms_per_step"] > 0 and r["fwd_ms"] > 0 and r["saved_mb"] > 0
    assert r["params"] > 0 and r["shape"]


# ---------------------------------------------------------------------------------------------
# gallery
# ---------------------------------------------------------------------------------------------
class _Broken(Toy1D):
    name = "broken_instance"
    description = "raises during data generation"

    def make_measurement(self, seed=0, scene_class=None, gt=None):
        raise RuntimeError("synthetic failure for the gallery test")


def test_physics_gallery_toy1d(tmp_path):
    man = viz.physics_gallery(
        ["toy1d"], budget_scale=0.02, device="cpu", out_dir=tmp_path, max_steps=24
    )
    assert man["n_ok"] == 1 and man["n_failed"] == 0
    (e,) = man.entries
    assert e.ok and 0 < e.steps <= 24 and "psnr" in e.metrics
    assert e.timing["solve_s"] > 0 and e.layout == "signal"
    for key in ("tile", "compare", "measurement", "fit", "history", "stages"):
        p = tmp_path / e.figures[key]
        assert p.exists() and p.stat().st_size <= 300 * KB, key
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["entries"][0]["name"] == "toy1d" and saved["overview"] == "gallery.png"
    assert (tmp_path / "gallery.png").stat().st_size <= 300 * KB


def test_physics_gallery_survives_failures(tmp_path):
    man = viz.physics_gallery(
        ["toy1d", _Broken(n=32)],
        budget_scale=0.02,
        device="cpu",
        out_dir=tmp_path,
        max_steps=12,
        per_instance=False,
    )
    assert [e.status for e in man.entries] == ["ok", "failed"]
    bad = man.entries[1]
    assert "synthetic failure" in bad.error and "generate" in bad.error and bad.traceback
    assert (tmp_path / "gallery.png").exists()
    fig = viz.physics_overview(man.entries)
    texts = [" ".join(t.get_text().split()) for ax in fig.axes for t in ax.texts]
    assert any("synthetic failure" in t for t in texts)
    entry = viz.run_instance_smoke("definitely_not_registered")
    assert entry.status == "failed" and "definitely_not_registered" in entry.error


# ---------------------------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------------------------
class _Checker(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.imgs: list[str] = []
        self.tags: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == "img":
            self.imgs.append(dict(attrs)["src"])


def test_html_and_markdown_reports(tmp_path):
    fig = viz.plot_field(torch.rand(8, 8), field="rho")
    png = viz.savefig(fig, tmp_path / "field")[0]
    sections = [
        viz.Section(
            "Reconstruction",
            "Some **bold** text with `code`.",
            images=[(png, "a field"), fig],
            table=[{"method": "nefi", "psnr": 31.234567, "status": "ok"}],
        ),
        {"title": "Config", "code": "n: 64\nsteps: [300, 600]"},
    ]
    path = viz.html_report(
        tmp_path, "viz test", sections, config_hash="abc123", metrics=[{"psnr": 30.0}]
    )
    text = path.read_text()
    assert "<title>viz test</title>" in text and "abc123" in text
    assert "Environment" in text and nefi.__version__ in text
    parser = _Checker()
    parser.feed(text)
    assert len(parser.imgs) == 2 and all(
        s.startswith("data:image/png;base64,") for s in parser.imgs
    )
    assert "table" in parser.tags and "nav" in parser.tags
    md = viz.markdown_report(tmp_path, "viz test", sections, config_hash="abc123")
    mtext = md.read_text()
    assert "| method |" in mtext and "![a field](field.png)" in mtext
    assert (tmp_path / "_figures").is_dir()
    assert math.isfinite(len(mtext))
    env = viz.environment_info("cpu")
    assert env["device"] == "cpu" and "torch" in env
    _ = Path  # keep the import for type readers


# ---------------------------------------------------------------------------------------------
# signed / complex data, display hints
# ---------------------------------------------------------------------------------------------
def _bz_like(n: int = 32) -> np.ndarray:
    """A dipole-like map: a strong positive lobe and a ~20 % negative ring (signed data)."""
    x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    r2 = x**2 + y**2
    return np.exp(-r2 / 0.05) - 0.2 * np.exp(-((np.sqrt(r2) - 0.5) ** 2) / 0.02)


def test_signed_detection_and_colormap_hints():
    rng = np.random.default_rng(0)
    blob = np.exp(-(np.linspace(-3, 3, 64) ** 2))[:, None] * np.ones((1, 16))
    noisy_sino = blob + 0.01 * rng.standard_normal(blob.shape)  # dips below 0 by noise only
    assert noisy_sino.min() < 0 and not viz.is_signed(noisy_sino)
    bz = _bz_like()
    assert viz.is_signed(bz) and not viz.is_signed(np.abs(bz))
    spec = viz.cmap_for(quantity="measurement", data=bz)
    assert spec.cmap == "RdBu_r" and spec.kind == "diverging"
    view = viz.measurement_view(Measurement(torch.tensor(bz)), field_shape=(32, 32))
    assert view.spec.kind == "diverging"
    hinted = viz.measurement_view(
        Measurement(torch.tensor(bz)), field_shape=(32, 32), hints={"measurement_cmap": "magma"}
    )
    assert hinted.spec.cmap == "magma"
    assert viz.spec_from_hint("PuOr_r").kind == "diverging"
    assert viz.spec_from_hint("twilight").kind == "cyclic"
    assert viz.spec_from_hint("density").cmap == "magma"
    assert viz.spec_from_hint("signed").cmap == "RdBu_r"
    assert viz.spec_from_hint("auto") is None
    # signed fields get the centred map too; the image limits are symmetric around 0
    fig = viz.compare_fields({"q": torch.tensor(bz)}, {"nefi": torch.tensor(0.9 * bz)})
    lo, hi = image_axes(fig)[0].images[0].get_clim()
    assert lo == pytest.approx(-hi)


def test_hints_resolution_and_transforms():
    class Holo:
        name = "holography"
        viz_hints = {"field_label": "phase φ", "volume_axis": 0}

    hn = viz.instance_hints(Holo(), {"measurement_label": "hologram"})
    assert hn.get("field_transform") is None  # holography fixes its phase gauge in the model
    assert hn["field_label"] == "phase φ" and hn["measurement_label"] == "hologram"
    hn2 = viz.instance_hints(Holo(), {"field_transform": "zero_mean"})
    assert hn2["field_transform"] == "zero_mean"  # call-site hints win
    assert viz.field_hint({"field_transform": {"g": "abs"}}, "field_transform", "g") == "abs"
    assert viz.field_hint({"field_transform": {"g": "abs"}}, "field_transform", "h") is None
    a = np.arange(12.0).reshape(3, 4)
    assert abs(viz.apply_transform(a, "zero_mean").mean()) < 1e-12
    ramp = np.tile(np.arange(8.0)[:, None], (1, 8))  # g = x → |∇g| = 1 inside
    assert np.allclose(viz.apply_transform(ramp, "grad_magnitude"), 1.0)
    curl = viz.apply_transform(ramp, "curl_magnitude")  # zero outside → edge current
    assert np.allclose(curl[2:-2, 2:-2], 1.0) and curl[-1, 4] > 1.0
    logged = viz.apply_transform(np.array([-0.1, 0.05, 1.0, 100.0]), "log")
    assert np.isfinite(logged).all() and logged[-1] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        viz.apply_transform(a, "no_such_transform")
    # zero_mean: fields defined up to a constant compare equal, and the titles say so
    ph = np.zeros((24, 24))
    ph[6:14, 8:16], ph[16:20, 3:7] = 1.0, -0.5
    fig = viz.compare_fields(
        {"phase": torch.tensor(ph)},
        {"nefi": torch.tensor(ph - 0.7)},
        instance=Holo(),
        hints={"field_transform": "zero_mean"},  # explicit: no longer a holography default
    )
    titles = [ax.get_title() for ax in image_axes(fig)]
    assert "mean removed" in titles[0] and "mean removed" in titles[1]
    assert np.abs(image_axes(fig)[2].images[0].get_array()).max() < 1e-9  # error ≈ 0
    # a field label names a derived quantity (|J| of a stream function)
    figj = viz.compare_fields(
        {"g": torch.tensor(ramp)},
        {"nefi": torch.tensor(ramp)},
        hints={"field_transform": "curl_magnitude", "field_label": "|J|"},
    )
    assert "|J|" in figj._suptitle.get_text()
    assert image_axes(figj)[0].images[0].get_cmap().name.startswith("magma")


def test_complex_measurement_shows_magnitude_and_phase():
    c = torch.polar(torch.rand(8, 20) + 0.5, torch.rand(8, 20) * 6 - 3)
    meas = Measurement(c)
    view = viz.measurement_view(meas, field_shape=(16, 16))
    assert view.layout == "complex" and view.phase is not None
    stacked = Measurement(torch.stack([c.real, c.imag]))  # stacked real / imag parts
    v2 = viz.measurement_view(stacked, field_shape=(16, 16), hints={"complex_stack": "first"})
    assert v2.phase is not None and np.allclose(v2.phase, np.angle(c.numpy()), atol=1e-6)
    fig, axes = viz.new_figure(1, 1)
    viz.draw_measurement(axes[0, 0], meas, field_shape=(16, 16))
    insets = [ax for ax in axes[0, 0].child_axes if ax.images]
    assert len(insets) == 2  # |y| and arg y (inset axes of the measurement panel)
    assert insets[1].images[0].get_cmap().name.startswith(("twilight", "RdBu"))
    gt = sparse_2d()
    fig2 = viz.compare_fields({"chi": gt}, {"nefi": gt * 0.9}, meas)
    assert len(image_axes(fig2)) == 3  # GT, recon, error
    assert sum(len(ax.child_axes) for ax in fig2.axes) == 2  # measurement: |y| and arg y


def test_measurement_image_hook_is_drawn_as_image():
    class StackLike:
        name = "synthetic_stack_instance"
        viz_hints = {"layout": "stack", "word": "patterns"}

        def measurement_image(self, m):
            return m.data[1] - m.data[0]  # e.g. a boundary-difference view

    class Labelled(StackLike):
        def measurement_image(self, m):
            return m.data[1] - m.data[0], "boundary difference"

    meas = Measurement(torch.randn(4, 16, 16), noise_std=0.1)
    view = viz.measurement_view(meas, field_shape=(16, 16), instance=StackLike())
    assert view.layout == "image" and view.data.shape == (16, 16)
    assert "mean of" not in view.label
    own = viz.measurement_view(meas, field_shape=(16, 16), instance=Labelled())
    assert own.label == "boundary difference"
    raw = viz.measurement_view(meas, field_shape=(16, 16), hints=StackLike.viz_hints)
    assert raw.layout == "stack" and "mean of 4 patterns" in raw.label
    # plot_fit never calls the hook: the raw stack keeps its per-slice residual panel
    fig = viz.plot_fit(meas.data + 0.1, meas, field_shape=(16, 16), instance=StackLike())
    assert len(image_axes(fig)) == 2


def test_volume_and_sinogram_stack_measurements():
    vol = torch.zeros(12, 10, 8)
    vol[3:6, 4:7, 2:5] = 1.0
    view = viz.measurement_view(Measurement(vol), field_shape=(12, 10, 8))
    assert view.layout == "volume" and view.data.shape == (12, 10) and "max over z" in view.label
    fig = viz.plot_measurement(Measurement(vol), field_shape=(12, 10, 8))
    assert len(image_axes(fig)) == 8  # depth mosaic of the measurement volume
    sino = Measurement(torch.rand(9, 20, 8), meta={"angles": list(np.linspace(0, 3, 9))})
    sv = viz.measurement_view(sino, field_shape=(16, 16, 8))
    assert sv.layout == "sinogram" and sv.data.shape == (9, 20) and "z-slice" in sv.label
    assert viz.stack_axis((9, 20, 8), (16, 16, 8), {}) == 2
    assert viz.stack_axis((8, 9, 20), (16, 16, 8), {}) == 0
    assert len(image_axes(viz.plot_measurement(sino, field_shape=(16, 16, 8)))) == 1


# ---------------------------------------------------------------------------------------------
# 3-D views
# ---------------------------------------------------------------------------------------------
def _slab(nx: int = 14, ny: int = 12, nz: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    vol = torch.full((nx, ny, nz), 0.15)
    vol[3:7, 5:9, 2:4] = 0.01  # a low-diffusivity defect
    rec = vol + 0.005 * torch.randn(vol.shape, generator=torch.Generator().manual_seed(0))
    rec[3:7, 5:9, 1:5] = 0.04  # blurred in depth
    return vol, rec


def test_volume_helpers():
    vol, _ = _slab()
    assert viz.anomaly_centroid(vol) == (5, 7, 3)  # centre of the defect (3:7, 5:9, 2:4)
    mask = torch.zeros_like(vol)
    mask[10:13, 1:3, 4:6] = 1.0
    assert viz.anomaly_centroid(vol, mask=mask) == (11, 2, 5)
    assert viz.projection_mode(vol.numpy()) == "min"
    assert viz.mosaic_slices(16, 6) == [1, 4, 6, 9, 12, 14]
    assert viz.mosaic_slices(3, 6) == [0, 1, 2]  # thin volumes: every slice
    sec, lateral, fixed = viz.section(vol.numpy(), (5, 7, 3), 0)
    assert sec.shape == (14, 6) and (lateral, fixed) == (0, 1)
    pr, mode = viz.project(vol.numpy(), -1, "auto")
    assert mode == "min" and pr.shape == (14, 12) and pr.min() == pytest.approx(0.01)
    # depth mosaics of thin volumes: nz < 4 and a squeezed single slice
    assert len(image_axes(viz.depth_mosaic(vol[..., :2], field="alpha"))) == 2
    assert len(image_axes(viz.depth_mosaic(vol[..., 0], field="alpha"))) == 1
    thin = viz.compare_fields({"alpha": vol[..., :2]}, {"nefi": vol[..., :2] * 1.1})
    assert len(image_axes(thin)) == 3 * (2 + 2 + 1)  # (GT, recon, error) × (2 slices + 3)


def test_compare_fields_3d_panels():
    vol, rec = _slab()
    frames = Measurement(torch.rand(5, 14, 12) + 1, meta={"frame_times": [1, 2, 3, 4, 5]})
    over = viz.compare_fields({"alpha": vol}, {"nefi": rec}, frames, panels="overlay")
    axs = image_axes(over)
    assert len(axs) == 1 + 2 * (6 + 2 + 1)  # measurement + (GT, recon) × (mosaic, x–z, y–z, proj)
    recon_row = [ax for ax in axs if ax.collections][: 6 + 2 + 1]
    assert len(recon_row) >= 6  # GT contours on the reconstruction panels
    assert any("GT contour" in t.get_text() for leg in over.legends for t in leg.get_texts())
    prof = viz.compare_fields(
        {"alpha": vol}, {"nefi": rec}, None, panels=("gt", "recon", "profile")
    )
    lines = [ax for ax in prof.axes if not ax.images and ax.lines]
    assert len(lines) == 3 and all(len(ax.lines) == 2 for ax in lines)  # x, y, depth cuts
    err = viz.compare_fields({"alpha": vol}, {"a": rec, "b": vol}, None, slices=3)
    assert len(image_axes(err)) == (1 + 2 + 2) * (3 + 2 + 1)  # GT, 2 recons, 2 error rows
    zm = viz.compare_fields(
        {"phase": vol}, {"nefi": vol + 0.5}, hints={"field_transform": "zero_mean"}
    )
    assert sum("mean removed" in ax.get_ylabel() for ax in image_axes(zm)) == 2  # GT, recon rows


def test_compare_fields_2d_overlay_and_profile():
    gt = torch.zeros(24, 24)
    gt[5:10, 6:14], gt[15:19, 3:7] = 1.0, 0.6
    rec = gt + 0.03 * torch.randn(24, 24, generator=torch.Generator().manual_seed(1))
    over = viz.compare_fields({"rho": gt}, {"nefi": rec}, Measurement(gt + 0.1), panels="overlay")
    axs = image_axes(over)
    assert len(axs) == 4 and axs[3].collections and "GT contours" in axs[3].get_title()
    prof = viz.compare_fields({"rho": gt}, {"nefi": rec}, None, panels=("gt", "recon", "profile"))
    line_ax = [ax for ax in prof.axes if ax.lines and not ax.images]
    assert len(line_ax) == 1 and len(line_ax[0].lines) == 2 and "profile" in line_ax[0].get_title()
    both = viz.compare_fields(
        {"rho": gt},
        {"a": rec, "b": 0.8 * rec},
        None,
        panels=("gt", "recon", "error", "overlay", "profile"),
    )
    assert len(image_axes(both)) == 1 + 2 + 2 + 2  # GT, recons, overlays, errors
    with pytest.raises(ValueError):
        viz.compare_fields({"rho": gt}, {"nefi": rec}, panels="nonsense")


def test_voxel_compare_same_threshold_and_camera():
    vol, rec = _slab()
    fig = viz.voxel_compare({"alpha": vol}, {"nefi": rec})
    ax3 = [ax for ax in fig.axes if ax.name == "3d"]
    assert len(ax3) == 2
    assert {(ax.elev, ax.azim) for ax in ax3} == {(ax3[0].elev, ax3[0].azim)}
    assert "IoU" in ax3[1].get_title() and "voxels" in ax3[0].get_title()


def test_animate_3d_mosaic_and_slice(tmp_path):
    vol, rec = _slab()
    coarse = torch.nn.functional.interpolate(rec[None, None], size=(7, 6, 3))[0, 0]
    snaps = [(i * 5, coarse * (1 + 0.1 * i)) for i in range(4)]
    snaps += [(20 + i * 5, rec * (1 + 0.05 * i)) for i in range(40)]  # finer stage
    gif = viz.animate_snapshots(snaps, tmp_path / "vol.gif", gt=vol, max_frames=30, max_kb=120)
    assert gif.exists() and gif.stat().st_size <= 120 * KB
    from PIL import Image

    with Image.open(gif) as im:
        assert 4 <= im.n_frames <= 30
    one = viz.animate_snapshots(snaps, tmp_path / "slice.gif", gt=vol, volume="slice", max_frames=6)
    assert one.exists() and one.stat().st_size <= 300 * KB


def test_physics_gallery_thermal_3d(tmp_path):
    man = viz.physics_gallery(
        ["thermal_tomography"], budget_scale=0.01, device="cpu", out_dir=tmp_path, max_steps=6
    )
    (e,) = man.entries
    assert e.ok, e.error
    assert man["volumetric"] == ["thermal_tomography"] and man["overview_3d"] == "gallery_3d.png"
    for f in ("gallery.png", "gallery_3d.png"):
        assert (tmp_path / f).stat().st_size <= 300 * KB, f
    for key in ("tile", "compare", "mosaic", "voxels"):
        assert (tmp_path / e.figures[key]).stat().st_size <= 300 * KB, key
    tile = viz.physics_overview(man.entries)
    assert len(image_axes(tile)) >= 2 * 6 + 1 + 4 + 2  # two mosaics, measurement, sections, proj
    fig3 = viz.physics_volumes(man.entries)
    assert sum(ax.name == "3d" for ax in fig3.axes) == 2  # GT and reconstruction voxels
    secs = viz.gallery_sections(man, tmp_path)
    assert [s.id for s in secs[:2]] == ["overview", "volumes-3d"]


# ---------------------------------------------------------------------------------------------
# performance profiles
# ---------------------------------------------------------------------------------------------
def test_profile_speedups(tmp_path):
    def row(**kw):
        base = {"instance": "heat3d", "device": "cpu", "threads": 4, "loop": "solver"}
        return json.dumps({**base, "compile": "none", "autocast": "none", **kw})

    (tmp_path / "a_before.jsonl").write_text(row(ms_median=30.0) + "\n")
    after = [row(ms_median=10.0), row(instance="toy", ms_median=1.0)]
    (tmp_path / "b_after.jsonl").write_text("\n".join(after) + "\nnot json\n")
    variants = [
        row(compile="step", ms_median=5.0),
        row(overrides=["solver=chebyshev"], ms_median=8.0),
        row(instance="toy", autocast="bf16", ms_median=2.0),
        row(loop="minimal", ms_median=9.0),  # harness variant: skipped
    ]
    (tmp_path / "c_variants.jsonl").write_text("\n".join(variants) + "\n")
    import os

    for i, name in enumerate(("a_before", "b_after", "c_variants")):  # mtime order
        os.utime(tmp_path / f"{name}.jsonl", (1_000_000 + i, 1_000_000 + i))
    rows = viz.load_profiles(tmp_path)
    assert len(rows) == 7 and rows[0]["source"] == "a_before"
    sp = {(r["instance"], r["variant"]): r["speedup"] for r in viz.profile_speedups(rows)}
    assert sp[("heat3d", "compile=step")] == pytest.approx(2.0)  # vs the latest eager (10 ms)
    assert sp[("heat3d", "solver=chebyshev")] == pytest.approx(1.25)
    assert sp[("toy", "autocast=bf16")] == pytest.approx(0.5)
    assert len(sp) == 3
    fig = viz.plot_speedup(rows)
    assert len(fig.axes[0].patches) == 3
    assert viz.load_profiles(tmp_path / "missing") == []
    assert "no profiles" in " ".join(t.get_text() for t in viz.plot_speedup([]).axes[0].texts)
