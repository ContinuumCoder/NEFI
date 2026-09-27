"""Tests for the interactive 3-D viewer and the isosurface tools (``nefi.viz.interactive``,
``nefi.viz.isosurface``): HTML fragments and ids, quantized payloads and size limits, standalone
pages, instance threshold rules, marching-cubes embedding, volume-matched / Otsu levels, display
transforms, the JavaScript mirror of the level rules (run in node when available), the gallery
integration and the optional plotly mode. Tiny sizes; the file runs in a few seconds.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from nefi import viz  # noqa: E402
from nefi.viz import interactive as vi  # noqa: E402
from nefi.viz import isosurface as iso  # noqa: E402

ASSET = Path(vi.__file__).parent / "assets" / "volume_viewer.js"
NODE = shutil.which("node")


def soft_sphere(
    n: int = 20, radius: float = 5.0, width: float = 1.5, contrast: float = 0.7, noise: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """A sharp GT sphere (0.1 / 1.0) and a reconstruction with softer edges, lower contrast."""
    grid = np.meshgrid(*[np.arange(n, dtype=float)] * 3, indexing="ij")
    r = np.sqrt(sum((g - (n - 1) / 2) ** 2 for g in grid))
    gt = np.where(r < radius, 1.0, 0.1)
    rec = 0.1 + 0.9 * contrast / (1.0 + np.exp((r - radius) / width))
    if noise:
        rec = rec + noise * np.random.default_rng(0).standard_normal(rec.shape)
    return gt, rec


_DATA = re.compile(r"<script type='application/json' id='([\w-]+)-data'>(.*?)</script>", re.S)


def payload_of(html: str) -> dict:
    return json.loads(_DATA.search(html).group(2))


class _Ids(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.srcs: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.append(a["id"])
        for k in ("src", "href"):
            if a.get(k):
                self.srcs.append(a[k])


# ---------------------------------------------------------------------------------------------
# fragments, payloads, sizes
# ---------------------------------------------------------------------------------------------
def test_fragment_has_script_unique_ids_and_quantized_data():
    gt, rec = soft_sphere(12)
    html = viz.volume_viewer_html(
        {"ground truth": torch.as_tensor(gt), "reconstruction": rec},
        extent=[(0, 2), (0, 2), (0, 1)],
        field="rho",
        title="sphere",
    )
    assert "<script>" in html and "NefiVolumeViewer" in html and "<style>" in html
    div = re.findall(r"<div class='nefi-vv' id='([\w-]+)'", html)
    assert len(div) == 1 and f"id='{div[0]}-data'" in html and f"push('{div[0]}')" in html
    P = payload_of(html)
    assert P["shape"] == [12, 12, 12] and P["axes"] == ["x", "y", "z"] and not P["downsampled"]
    assert P["extent"] == [[0.0, 2.0], [0.0, 2.0], [0.0, 1.0]] and P["cmap"] == "magma"
    assert P["mode"] == "voxels" and P["meshes"] is None  # meshes only for isosurface views
    for vol, ref in zip(P["volumes"], (gt, rec)):  # small volumes: 16-bit codes
        assert vol["bits"] == 16 and len(base64.b64decode(vol["data"])) == 2 * 12**3
        dec = vi.decode_volume(vol, P["shape"])
        assert np.abs(dec - ref).max() <= 1.51 * vol["step"] < 1e-4  # rounding + one nudge
    narrow = vi.viewer_payload({"a": rec}, bits=8)
    assert len(base64.b64decode(narrow["volumes"][0]["data"])) == 12**3
    assert np.abs(vi.decode_volume(narrow["volumes"][0], [12] * 3) - rec).max() < 0.01
    with pytest.raises(ValueError):
        vi.viewer_payload({"flat": np.zeros((4, 4))})
    with pytest.raises(ValueError):
        vi.normalize_mode("hologram")
    assert vi.normalize_mode("points+slice") == "voxels+slice"
    assert vi.normalize_mode("isosurface") == "iso"


def test_two_viewers_on_one_page_do_not_clash():
    gt, rec = soft_sphere(10)
    first = viz.volume_viewer_html({"gt": gt, "rec": rec})
    second = viz.volume_viewer_html({"gt": gt}, include_assets=False, mode="slice")
    page = first + "\n" + second
    parser = _Ids()
    parser.feed(page)
    assert len(parser.ids) == len(set(parser.ids)) == 4  # two divs + two data scripts
    assert page.count("root.NefiVolumeViewer = API") == 1  # the library is included once
    assert page.count(".push('nefi-vv-") == 2
    assert [payload_of(h)["mode"] for h in (first, second)] == ["voxels", "slice"]
    # the library is idempotent when a page includes it twice (it only flushes the queue)
    assert "if (root.NefiVolumeViewer) return root.NefiVolumeViewer.flush();" in first


def test_size_limits_and_downsampling():
    rng = np.random.default_rng(0)
    big = rng.random((96, 96, 96))
    P = vi.viewer_payload({"a": big, "b": 0.5 * big})
    assert P["shape"] == [64, 64, 64] and P["source_shape"] == [96, 96, 96] and P["downsampled"]
    assert P["volumes"][0]["bits"] == 8  # 64³: 8-bit codes keep the page small
    assert viz.viewer_size_bytes(P) < 2 * 1024**2  # fragment without the shared assets
    blob = iso.gaussian_smooth((rng.random((96, 96, 96)) > 0.995).astype(float), 3.0)
    Q = vi.viewer_payload({"gt": blob, "rec": blob}, mode="iso", max_faces=6000)
    assert max(Q["shape"]) <= 64 and all(m["nf"] <= 6000 for m in Q["meshes"])
    assert viz.viewer_size_bytes(Q) < 2 * 1024**2
    assert vi.asset_bytes() < 80 * 1024
    thin = vi.viewer_payload({"a": big[:, :, :6]}, extent=[(0, 10), (0, 10), (0, 1)])
    assert thin["stretch"] == pytest.approx(3.5) and vi.auto_stretch([(0, 1)] * 3) == 1.0


def test_save_volume_viewer_is_a_standalone_self_contained_page(tmp_path):
    gt, rec = soft_sphere(10)
    p = viz.save_volume_viewer(
        tmp_path / "viewer.html", {"gt": gt, "rec": rec}, mode="iso", title="t"
    )
    text = p.read_text()
    assert text.startswith("<!doctype html>") and "<title>t</title>" in text
    parser = _Ids()
    parser.feed(text)
    assert not [s for s in parser.srcs if re.match(r"(https?:)?//", s)]  # no external resources
    assert "<script src" not in text and payload_of(text)["meshes"]


# ---------------------------------------------------------------------------------------------
# threshold rules and levels
# ---------------------------------------------------------------------------------------------
def test_threshold_rules_from_instances_reproduce_their_iou():
    from nefi.instances._volumetric import iou_above
    from nefi.metrics.segmentation import iou_below
    from nefi.viz._instances import resolve_instance

    thermal = resolve_instance("thermal_tomography")[0]
    r = viz.voxel_rule(thermal, np.zeros((4, 4, 4)), field="alpha")
    assert (r["mode"], r["side"], r["value"]) == ("absolute", "below", pytest.approx(0.03))
    ct = resolve_instance("ct3d")[0]
    r = viz.voxel_rule(ct, np.zeros((4, 4, 4)))
    assert (r["mode"], r["side"], r["value"]) == ("absolute", "above", pytest.approx(0.25))
    dot = resolve_instance("dot3d")[0]
    rd = viz.voxel_rule(dot, np.zeros((4, 4, 4)))
    assert (rd["mode"], rd["side"], rd["fraction"], rd["background"]) == (
        "relative",
        "above",
        0.5,
        pytest.approx(0.01),
    )
    # the rules select exactly the instances' masks
    rng = np.random.default_rng(1)
    shape = dot.domain().shape  # the metric uses the instance's depth axis
    g = 0.01 + 0.03 * (rng.random(shape) > 0.8)
    p = 0.01 + 0.02 * rng.random(shape)
    got = iso.iou_dice(iso.select_mask(g, rd), iso.select_mask(p, rd))[0]
    ref = dot.inclusion_metrics(torch.as_tensor(p), torch.as_tensor(g))["iou"]
    assert got == pytest.approx(ref, abs=1e-12)
    a, b = rng.random((8, 8, 8)) * 0.06, rng.random((8, 8, 8)) * 0.06
    rt = viz.voxel_rule(thermal, a)
    got = iso.iou_dice(iso.select_mask(a, rt), iso.select_mask(b, rt))[0]
    assert got == pytest.approx(iou_below(torch.as_tensor(b), torch.as_tensor(a), 0.03))
    rc = viz.voxel_rule(ct, a * 10)
    got = iso.iou_dice(iso.select_mask(a * 10, rc), iso.select_mask(b * 10, rc))[0]
    assert got == pytest.approx(iou_above(b * 10, a * 10, 0.25))
    # fallbacks: half-way from the background, the voxel_threshold display hint
    gt, _ = soft_sphere(16)
    auto = viz.voxel_rule(None, gt)
    assert auto["side"] == "above" and auto["value"] == pytest.approx(0.55)
    hinted = viz.voxel_rule(None, gt, hints={"voxel_threshold": ("below", 0.2)})
    assert hinted["side"] == "below" and hinted["value"] == 0.2 and "hint" in hinted["source"]


def test_volume_matched_level_recovers_the_gt_fraction():
    gt, rec = soft_sphere(24, radius=6.0, width=2.0, contrast=0.6)
    lv = iso.iso_levels(gt, rec)  # GT rule: half-way, 0.55
    n_gt = lv["gt_count"]
    assert 0 < n_gt < gt.size and lv["counts"]["matched"] == n_gt
    assert lv["iou"]["matched"] > 0.95 > 0.5 > lv["iou"]["fixed"]
    assert lv["dice"]["matched"] > lv["dice"]["fixed"]
    below = iso.iso_levels(1.1 - gt, 1.1 - rec)  # the same problem with a low-valued anomaly
    assert below["side"] == "below" and below["counts"]["matched"] == n_gt
    assert below["iou"]["matched"] == pytest.approx(lv["iou"]["matched"])
    v = np.arange(10.0)
    assert np.sum(v > iso.matched_level(v, "above", 3)) == 3
    assert np.sum(v < iso.matched_level(v, "below", 4)) == 4
    assert np.sum(v > iso.matched_level(v, "above", 0)) == 0
    assert np.sum(v > iso.matched_level(v, "above", 10)) == 10
    mix = np.concatenate([np.full(500, 0.2), np.full(300, 0.8)])
    mix = mix + 0.02 * np.random.default_rng(0).standard_normal(mix.size)
    assert 0.3 < iso.otsu_level(mix) < 0.7


def test_display_transforms():
    rng = np.random.default_rng(0)
    x = np.zeros((16, 16, 16))
    x[4:12, 4:12, 4:12] = 1.0
    noisy = x + 0.1 * rng.standard_normal(x.shape)
    sm = iso.volume_transform(noisy, "smooth")
    assert sm.shape == x.shape and sm.std() < noisy.std()
    assert abs(sm[6:10, 6:10, 6:10].mean() - 1.0) < 0.05  # the interior level is kept
    blurred = iso.gaussian_smooth(x, 1.5)
    sh = iso.volume_transform(blurred, {"name": "sharpen", "amount": 1.5})
    step = lambda a: np.abs(np.diff(a[8, 8, :])).max()  # noqa: E731
    assert step(sh) > step(blurred) and blurred.min() <= sh.min() and sh.max() <= blurred.max()
    ep = iso.volume_transform(noisy, "edge_preserve")
    flat = (slice(5, 11),) * 3
    assert ep[flat].std() < 0.7 * noisy[flat].std()  # noise inside the cube is reduced
    assert ep[8, 8, 8] - ep[8, 8, 1] > 0.8  # the step survives
    with_nan = noisy.copy()
    with_nan[0, 0, 0] = np.nan
    assert np.isnan(iso.volume_transform(with_nan, "smooth")[0, 0, 0])
    assert iso.transform_spec("gaussian")["name"] == "smooth"
    assert iso.transform_spec("none") is None and iso.transform_spec(None) is None
    assert iso.transform_label("smooth") == "display: smooth σ=1"
    hint = {"volume_transform": {"rho": ("sharpen", {"amount": 2})}}
    assert iso.hinted_transform(hint, "rho")["amount"] == 2.0
    assert iso.hinted_transform(hint, "other") is None
    for bad in ("median", {"name": "smooth", "radius": 2}):
        with pytest.raises(ValueError):
            iso.transform_spec(bad)
    P = vi.viewer_payload({"gt": x, "rec": blurred}, transform="sharpen")
    assert P["transform"]["name"] == "sharpen" and "display" in iso.transform_label(P["transform"])


# ---------------------------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------------------------
def test_marching_cubes_mesh_is_embedded_closed_and_on_the_level(monkeypatch):
    from scipy.ndimage import map_coordinates

    gt, rec = soft_sphere(20)
    P = vi.viewer_payload({"ground truth": gt, "reconstruction": rec}, mode="iso")
    shape = np.asarray(P["shape"])
    assert len(P["meshes"]) == 2
    for mesh, vol in zip(P["meshes"], P["volumes"]):
        assert mesh["nf"] > 0 and mesh["level"] == vol["level"] and mesh["side"] == "above"
        v, f = vi.decode_mesh(mesh, shape)
        assert f.max() < len(v) and (v >= -0.5 - 1e-6).all() and (v <= shape - 0.5 + 1e-6).all()
        t = v[f]
        enclosed = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6
        assert enclosed == pytest.approx(vol["count"], rel=0.25)  # outward, ≈ the voxel set
        dec = vi.decode_volume(vol, shape)
        on = map_coordinates(dec, v.T, order=1)  # vertices lie on the level set
        assert np.median(np.abs(on - vol["level"])) < 0.05 * (dec.max() - dec.min())
    raw_v, raw_f = iso.iso_mesh(gt, 0.55, "above")
    assert raw_f.shape[1] == 3 and len(raw_v) > 0
    assert iso.iso_mesh(gt, 5.0, "above")[1].shape == (0, 3)  # nothing past the level
    monkeypatch.setattr(vi, "has_marching_cubes", lambda: False)  # no scikit-image:
    assert vi.viewer_payload({"gt": gt}, mode="iso")["meshes"] is None  # the browser extracts


def test_iso_compare_and_static_isosurfaces(monkeypatch):
    gt, rec = soft_sphere(14, radius=4.0)
    fig = viz.iso_compare({"rho": gt}, {"nefi": rec})
    ax3 = [ax for ax in fig.axes if ax.name == "3d"]
    assert len(ax3) == 2 and {(ax.elev, ax.azim) for ax in ax3} == {(ax3[0].elev, ax3[0].azim)}
    assert "volume-matched" in ax3[1].get_title() and "Dice" in ax3[1].get_title()
    assert any(ax.images for ax in fig.axes if ax.name != "3d")  # the contour panels
    assert len(ax3[1].collections) >= 1  # a Poly3DCollection
    monkeypatch.setattr(iso, "iso_mesh", lambda *a, **k: None)  # no scikit-image: voxels
    fig2 = viz.iso_compare({"rho": gt}, {"nefi": rec})
    assert sum(ax.name == "3d" for ax in fig2.axes) == 2


# ---------------------------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_javascript_parses():
    subprocess.run([NODE, "--check", str(ASSET)], check=True, capture_output=True)


_JS_CHECK = r"""
const fs = require("fs");
require(process.argv[2]);
const V = globalThis.NefiVolumeViewer, cases = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const out = {};
for (const [name, P] of Object.entries(cases)) {
  const s = P.shape, N = s[0] * s[1] * s[2], r = P.threshold;
  const raw = P.volumes.map(v => V.decode(v, N));
  const disp = raw.map((d, k) => (k > 0 && P.transform ? V.transform(d, s, P.transform) : d));
  const m0 = new Uint8Array(N), m1 = new Uint8Array(N), g = P.volumes[0], rec = P.volumes[1];
  const n0 = V.select(disp[0], r, g.peak, g.trough, m0);
  const e1 = P.transform ? [Math.min(...disp[1]), Math.max(...disp[1])] : [rec.trough, rec.peak];
  const rule1 = P.level_rule === "fixed" ? r : { mode: "absolute", side: r.side, value: rec.level };
  const n1 = V.select(disp[1], rule1, e1[1], e1[0], m1);
  const mesh = V.isoMesh(disp[1], s, rec.level, r.side), f = mesh.f, v = mesh.v, edges = new Map();
  let vol = 0;
  for (let i = 0; i < f.length; i += 3) {
    const p = [f[i], f[i + 1], f[i + 2]].map(k => [v[3 * k], v[3 * k + 1], v[3 * k + 2]]);
    vol += (p[0][0] * (p[1][1] * p[2][2] - p[1][2] * p[2][1]) - p[0][1] * (p[1][0] * p[2][2] -
      p[1][2] * p[2][0]) + p[0][2] * (p[1][0] * p[2][1] - p[1][1] * p[2][0])) / 6;
    for (const [a, b] of [[f[i], f[i + 1]], [f[i + 1], f[i + 2]], [f[i + 2], f[i]]])
      edges.set(a + "," + b, (edges.get(a + "," + b) || 0) + 1);
  }
  let closed = true;
  for (const [k, c] of edges) {
    const [a, b] = k.split(",");
    if (c !== 1 || edges.get(b + "," + a) !== 1) closed = false;
  }
  out[name] = {
    counts: [n0, n1], overlap: V.overlap(m0, m1),
    matched: V.matchedLevel(V.sortedFinite(disp[1]), r.side, n0), otsu: V.otsuLevel(disp[1]),
    sum: disp[1].reduce((a, b) => a + b, 0), mt_closed: closed, mt_volume: vol,
  };
}
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_javascript_mirrors_the_python_levels_and_meshes(tmp_path):
    gt, rec = soft_sphere(16, radius=4.5, noise=0.01)
    cases = {
        "absolute": vi.viewer_payload({"gt": gt, "rec": rec}),
        "below": vi.viewer_payload({"gt": 1.1 - gt, "rec": 1.1 - rec}),
        "relative": vi.viewer_payload(
            {"gt": gt, "rec": rec}, threshold={"fraction": 0.5, "background": 0.1}
        ),
        "sharpen": vi.viewer_payload({"gt": gt, "rec": rec}, transform="sharpen"),
        "edge_otsu": vi.viewer_payload(
            {"gt": gt, "rec": rec}, transform="edge_preserve", level_rule="otsu", bits=16
        ),
        "fixed": vi.viewer_payload({"gt": gt, "rec": rec}, level_rule="fixed"),
    }
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    (tmp_path / "check.js").write_text(_JS_CHECK)
    res = subprocess.run(
        [NODE, str(tmp_path / "check.js"), str(ASSET), str(tmp_path / "cases.json")],
        check=True,
        capture_output=True,
        text=True,
    )
    js = json.loads(res.stdout)
    for name, P in cases.items():
        py = vi.payload_levels(P)
        dec = [vi.decode_volume(v, P["shape"]) for v in P["volumes"]]
        disp = iso.volume_transform(dec[1], P["transform"]) if P["transform"] else dec[1]
        out = js[name]
        assert out["counts"] == [py[0]["count"], py[1]["count"]], name
        assert out["overlap"][0] == pytest.approx(py[1]["iou"], abs=1e-12), name
        assert out["overlap"][1] == pytest.approx(py[1]["dice"], abs=1e-12), name
        side = P["threshold"]["side"]
        assert out["matched"] == pytest.approx(iso.matched_level(disp, side, py[0]["count"]))
        assert out["otsu"] == pytest.approx(iso.otsu_level(disp))
        assert out["sum"] == pytest.approx(float(disp.sum()), rel=1e-9), name  # transforms
        assert out["mt_closed"] and out["mt_volume"] > 0, name  # closed, outward surfaces
    # the GT threshold classifies the decoded data exactly like the raw data
    raw = iso.iou_dice(
        iso.select_mask(gt, cases["fixed"]["threshold"]),
        iso.select_mask(rec, cases["fixed"]["threshold"]),
    )[0]
    assert js["fixed"]["overlap"][0] == pytest.approx(raw, abs=1e-12)


# ---------------------------------------------------------------------------------------------
# gallery, plotly
# ---------------------------------------------------------------------------------------------
def test_gallery_embeds_a_viewer_under_the_static_row(tmp_path):
    man = viz.physics_gallery(
        ["thermal_tomography"],
        budget_scale=0.01,
        device="cpu",
        out_dir=tmp_path,
        max_steps=6,
        per_instance=False,
    )
    (e,) = man.entries
    assert e.ok, e.error
    assert man["interactive"] == "canvas" and e.viewer["kind"] == "canvas"
    for rel in (e.viewer["html"], e.viewer["data"], e.figures["block3d"]):
        assert (tmp_path / rel).exists(), rel
    assert e.iso["iou"]["fixed"] == pytest.approx(e.metrics["iou"], abs=1e-6)  # the IoU rule
    assert e.iso["counts"]["matched"] <= e.iso["gt_count"] + 1
    secs = viz.gallery_sections(man, tmp_path)
    vol = secs[1]
    assert vol.id == "volumes-3d" and vol.table and "IoU matched" in vol.table[0]
    (sub,) = vol.subsections
    assert sub.images and "nefi-vv" in sub.html and "root.NefiVolumeViewer = API" in sub.html
    page = viz.html_report(tmp_path, "g", secs).read_text()
    parser = _Ids()
    parser.feed(page)
    assert not [s for s in parser.srcs if re.match(r"(https?:)?//", s)]  # self-contained
    assert page.count("root.NefiVolumeViewer = API") == 1
    P = payload_of(page)
    fixed = P["volumes"][1]["levels"]["fixed"]
    assert fixed["iou"] == pytest.approx(e.metrics["iou"], abs=1e-6)  # exact on the 8-bit data
    static = viz.gallery_sections(man, tmp_path, interactive=False)[1]
    assert not static.subsections and "gallery_3d" in str(static.images[0]["path"])
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["entries"][0]["viewer"]["data_bytes"] > 0 and saved["interactive"] == "canvas"
    off = viz.physics_gallery(
        ["thermal_tomography"],
        budget_scale=0.01,
        device="cpu",
        out_dir=tmp_path / "off",
        max_steps=6,
        per_instance=False,
        interactive=False,
    )
    assert not off.entries[0].viewer and "block3d" not in off.entries[0].figures
    assert off.entries[0].iso  # the level comparison is always computed


def test_plotly_mode_references_the_cdn(tmp_path):
    pytest.importorskip("plotly")
    gt, rec = soft_sphere(10)
    html = vi.plotly_volume_html({"gt": gt, "rec": rec}, include_plotlyjs="cdn")
    assert "cdn.plot.ly" in html and "isosurface" in html.lower() and "<html" not in html[:20]
    p = viz.save_volume_viewer(tmp_path / "p.html", {"gt": gt, "rec": rec}, kind="plotly")
    assert "cdn.plot.ly" in p.read_text()
    assert vi.normalize_kind("none") is None and vi.normalize_kind(True) == "canvas"


def test_gallery_refine_column_with_a_stub(tmp_path, monkeypatch):
    """GT | smooth | refined in the overview row, gallery_3d, compare / voxels and the viewer,
    with the verdict and IoU / Edge-F1 / χ before → after in the titles; the headline stays the
    smooth result's. The refinement stage is stubbed (``nefi.solve.refine_run_output``)."""
    import dataclasses

    import nefi.solve
    from nefi.instances.base import RunOutput

    calls = []

    class Report:  # the RefineReport attributes the gallery reads
        accepted, mode, steps, seconds, reason = True, "levelset", 5, 0.01, "stub"
        fit_measure, fit_before, fit_after = "chi", {"chi": 1.5}, {"chi": 1.2}
        metrics_before = {"iou": 0.5, "edge_f1": 0.4}
        metrics_after = {"iou": 0.6, "edge_f1": 0.9}
        candidate = None

        def summary(self):
            return "refine[levelset] stub: accepted — χ 1.5 → 1.2"

    def stub(entry, **kw):
        calls.append(entry.name)
        res = entry.run.result
        crisp = {k: torch.where(v < 0.08, 0.01, 0.16).to(v) for k, v in res.fields.items()}
        refined = dataclasses.replace(res, fields=crisp)
        extra = {"refine_report": Report()}
        return RunOutput(refined, {"iou": 0.9}, entry.run.gt, entry.run.measurement, extra)

    monkeypatch.setattr(nefi.solve, "refine_run_output", stub)
    man = viz.physics_gallery(
        ["thermal_tomography"],
        budget_scale=0.01,
        device="cpu",
        out_dir=tmp_path,
        max_steps=6,
        refine=True,
    )
    (e,) = man.entries
    assert calls == ["thermal_tomography"] and e.ok and e.refine["accepted"]
    assert e.metrics["iou"] != 0.9 and e.headline()[0] == "psnr"  # headline: the smooth result
    assert e.refine["iou_after"] == 0.9 and e.refine["iou_source"] == "instance"
    title = viz.multiphysics.tile_title(e)
    # IoU: the instance's metric (smooth → refined); Edge-F1: the report's (the stub's refined
    # run carries no instance edge_f1); χ: the report's data fit
    assert "refined ✓" in title and "Edge-F1 0.40 → 0.90" in title and "χ 1.50 → 1.20" in title
    assert sum(ax.name == "3d" for ax in viz.physics_volumes(man.entries).axes) == 3
    for key in ("compare", "voxels", "block3d"):
        assert (tmp_path / e.figures[key]).exists(), key
    payload = json.loads((tmp_path / e.viewer["data"]).read_text())
    assert [v["name"] for v in payload["volumes"]] == ["ground truth", "smooth", "refined"]
    assert "refined" in e.iso and set(e.iso["refined"]["iou"]) >= {"fixed", "matched"}
    vol = viz.gallery_sections(man, tmp_path)[1]
    ref = [s for s in vol.subsections if s.id == "volumes-3d-refinement"]
    assert ref and ref[0].table[0]["verdict"] == "accepted" and "χ 1.5 → 1.2" in ref[0].text
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["refine"] and saved["entries"][0]["refine"]["summary"].startswith("refine[")
    refused = viz.multiphysics.refine_line(
        {"accepted": False, "fit_measure": "rmse", "fit_before": 0.1, "fit_after": 0.3}
    )
    assert (
        refused.startswith("refinement refused (smooth kept)") and "RMSE 0.100 → 0.300" in refused
    )


def test_3d_budget_floor_at_gallery_budgets():
    # thermal smoke preset: 150 steps; at budget ≥ 0.2 a 3-D system gets ≥ 2 × that, above the cap
    e = viz.run_instance_smoke(
        "thermal_tomography", budget_scale=0.2, device="cpu", max_steps=6, time_budget_s=0.3
    )
    b = e.budget
    assert e.ok and b["steps_3d_floor"] == 2 * b["preset_steps"] == b["steps"]
    plain = viz.run_instance_smoke(
        "thermal_tomography", budget_scale=0.19, device="cpu", max_steps=6, time_budget_s=0.3
    )
    assert "steps_3d_floor" not in plain.budget and plain.budget["steps"] == 6
