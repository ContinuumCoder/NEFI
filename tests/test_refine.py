"""Reconstruction-side edge refinement (``nefi.solve.refine``)."""

from __future__ import annotations

import json
import math

import pytest
import torch

import nefi
from nefi import (
    MSE,
    TV,
    Bounded,
    Curriculum,
    Domain,
    Heads,
    InverseProblem,
    LossSet,
    Measurement,
    NeuralField,
)
from nefi.errors import ConfigError
from nefi.operators import FFTConvolution, gaussian_kernel_fn
from nefi.solve.refine import (
    REFINE_DEFAULTS,
    DoubleWell,
    MultiPhaseHead,
    RefineReport,
    cell_heaviside,
    data_fit,
    edge_metrics,
    estimate_levels,
    mass_matched_fraction,
    multi_otsu,
    otsu_threshold,
    refine_defaults,
    refine_edges,
    refine_method,
    refine_run_output,
    transition_band_check,
    volume_matched_threshold,
)

pytestmark = pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad")

SIGMA = 0.01


def _phantom(n: int = 32) -> tuple[Domain, torch.Tensor]:
    """A body plus three small disks (values 1.0) on a 0.1 background."""
    dom = Domain.unit((n, n))
    xx = dom.coords()

    def disk(cx: float, cy: float, r: float) -> torch.Tensor:
        return ((xx[..., 0] - cx) ** 2 + (xx[..., 1] - cy) ** 2) < r**2

    body = (xx[..., 0].abs() < 0.5) & ((xx[..., 1] + 0.35).abs() < 0.3)
    small = disk(-0.5, 0.5, 0.14) | disk(0.05, 0.55, 0.12) | disk(0.55, 0.45, 0.16)
    return dom, torch.where(body | small, 1.0, 0.1)


def _blurred(steps: int):
    """Blurred phantom (σ_blur = 1.9 px, 1 % noise) and a soft, band-limited MLP reconstruction
    after ``steps`` steps."""
    torch.manual_seed(0)
    dom, gt = _phantom()
    op = FFTConvolution(gaussian_kernel_fn(0.06), dom, field="x", periodic=False)
    with torch.no_grad():
        clean = op({"x": gt})
    meas = Measurement(clean + SIGMA * torch.randn_like(clean), noise_std=SIGMA)
    field = NeuralField(
        2, Heads({"x": Bounded(0.0, 1.2, init_value=0.3)}), hidden=32, depth=2, n_octaves=3
    )
    losses = LossSet({"data": MSE(), "tv": TV("x")}, weights={"data": 1.0, "tv": 1e-4})
    problem = InverseProblem(dom, field, op, losses, meas, name="blurred-box")
    cur = Curriculum.single(dom.shape, steps=steps, lr=1e-2, anneal_fraction=0.5)
    result = nefi.invert(problem, cur, device="cpu", seed=0)
    return problem, result, {"x": gt}


@pytest.fixture(scope="module")
def blurred():
    return _blurred(300)


# ------------------------------------------------------------------------------------------
# the refinement stage
# ------------------------------------------------------------------------------------------
def test_levelset_sharpens_blurred_phantom_within_the_noise_level(blurred):
    problem, result, gt = blurred
    refined, rep = refine_edges(problem, result, gt=gt, device="cpu")
    mb, ma = rep.metrics_before, rep.metrics_after
    # the smooth reconstruction is soft: small objects lose contrast under the GT threshold
    assert mb["iou"] < ma["iou"] and ma["iou"] > 0.98
    assert mb["edge_f1"] < ma["edge_f1"]
    assert ma["psnr"] > mb["psnr"] + 3.0
    # physics stays the hard constraint: the refined field explains the data at the noise level
    assert rep.accepted and rep.fit_measure == "chi"
    assert rep.fit_after["chi"] <= 1.1 and rep.fit_after["chi"] <= 1.1 * max(
        rep.fit_before["chi"], 1.0
    )
    # a separate result, same units / shape, pointing back at its source
    assert refined is not result and refined is rep.candidate
    assert refined.fields["x"].shape == result.fields["x"].shape
    src = refined.extra["refined_from"]
    assert src["config_hash"] == result.config_hash and src["mode"] == "levelset"
    assert refined.extra["refine_accepted"] is True
    # phases: estimated around the true values (0.1 / 1.0)
    assert rep.levels_final[0] == pytest.approx(0.1, abs=0.05)
    assert rep.levels_final[1] == pytest.approx(1.0, abs=0.1)


def test_levelset_adds_beyond_the_continuation_control(blurred):
    problem, result, gt = blurred
    _, ctrl = refine_edges(problem, result, gt=gt, mode="continue", device="cpu")
    _, ls = refine_edges(problem, result, gt=gt, device="cpu")
    assert ctrl.accepted and ctrl.mode == "continue"
    # the control improves an early-stopped fit too; the piecewise-constant prior adds on top
    assert ls.metrics_after["psnr"] > ctrl.metrics_after["psnr"] + 0.5


def test_levelset_neural_and_grid_representations_run(blurred):
    problem, result, gt = blurred
    for rep_kind in ("grid", "neural"):
        out, rep = refine_edges(
            problem,
            result,
            gt=gt,
            representation=rep_kind,
            steps=60,
            warm_steps=60,
            device="cpu",
            refuse=False,
        )
        assert rep.representation == rep_kind and rep.steps == 60
        assert out.fields["x"].shape == result.fields["x"].shape
        assert torch.isfinite(out.fields["x"]).all()
        assert math.isfinite(rep.fit_after["chi"])
    assert "warm_rmse" in rep.settings["neural"]


@pytest.mark.parametrize("mode", ["phasefield", "tv_sharpen", "continue"])
def test_other_modes_run_and_report(blurred, mode):
    problem, result, gt = blurred
    out, rep = refine_edges(problem, result, gt=gt, mode=mode, steps=80, device="cpu", refuse=False)
    assert rep.mode == mode and rep.steps == 80
    assert set(out.fields) == {"x"} and out.fields["x"].shape == result.fields["x"].shape
    assert rep.metrics_after["psnr"] > rep.metrics_before["psnr"]
    if mode == "phasefield":
        assert rep.settings["double_well_weight"] > 0
    if mode == "tv_sharpen":  # Bounded head: binarizing swap between the two phases
        assert "head_swap" in rep.settings
        assert rep.settings["tv_weights"]["tv"] > problem.losses.weights["tv"]


def test_free_phase_keeps_interior_values(blurred):
    problem, result, gt = blurred
    out, rep = refine_edges(problem, result, gt=gt, free_phases=(1,), device="cpu")
    assert rep.free_phases == [1] and rep.accepted
    x = out.fields["x"]
    inside = gt["x"] > 0.5
    # the foreground is not forced to one value: it varies (it fits the blurred data)
    assert float(x[inside].std()) > 1e-4
    assert "phase 1 free" in rep.to_markdown()


def test_non_invertible_head_continues_in_the_same_value_range(blurred):
    from nefi.fields import GatedSoftplus

    problem, result, _ = blurred
    gated = NeuralField(2, Heads({"x": GatedSoftplus()}), hidden=8, depth=1, n_octaves=2)
    prob = InverseProblem(
        problem.domain, gated, problem.operator, problem.losses, problem.measurement
    )
    out, rep = refine_edges(prob, result, mode="continue", steps=30, device="cpu", refuse=False)
    assert any("continued with Softplus" in n for n in rep.notes)
    assert float(out.fields["x"].min()) >= 0.0  # the non-negative range is kept


def test_refusal_when_the_data_fit_degrades(blurred):
    problem, result, gt = blurred
    # a wrong, frozen contrast: the phases cannot explain the data
    out, rep = refine_edges(
        problem, result, gt=gt, levels=(0.0, 0.4), learn_levels=False, device="cpu"
    )
    assert not rep.accepted
    assert out is result  # the smooth result is kept, untouched
    assert "data fit degrades" in rep.reason
    assert rep.candidate is not None and rep.candidate.extra["refine_accepted"] is False
    assert "refined_from" in rep.candidate.extra
    assert "refused" in rep.to_markdown()
    # refuse=False hands the (flagged) candidate over
    out2, rep2 = refine_edges(
        problem, result, levels=(0.0, 0.4), learn_levels=False, refuse=False, device="cpu"
    )
    assert out2 is rep2.candidate and out2.extra["refine_accepted"] is False


def test_control_reference_closes_the_under_converged_loophole():
    # an early-stopped smooth start (χ ≫ 1) makes "no worse than the smooth fit" a loose test:
    # a frozen, too-low contrast passes it — the continuation control with the same budget does not
    problem, result, _ = _blurred(60)
    kw = {"levels": (0.1, 0.9), "learn_levels": False, "device": "cpu"}
    _, loose = refine_edges(problem, result, reference="smooth", **kw)
    _, strict = refine_edges(problem, result, **kw)  # reference="auto"
    assert result.history["data_loss"] and loose.fit_before["chi"] > 3.0
    assert loose.accepted and loose.fit_control is None
    assert not strict.accepted and "control" in strict.reason
    chi_c = strict.fit_control["chi"]
    assert chi_c < strict.fit_after["chi"] < strict.fit_before["chi"]
    assert strict.fit_limit == pytest.approx(1.1 * max(min(chi_c, strict.fit_before["chi"]), 1.0))
    assert "control (continue, same budget)" in strict.to_markdown()
    assert math.isfinite(strict.sharpening_cost) and strict.sharpening_cost > 0


def test_report_markdown_dict_and_history(blurred):
    problem, result, gt = blurred
    out, rep = refine_edges(
        problem, result, gt=gt, steps=40, merge_history=True, device="cpu", refuse=False
    )
    assert isinstance(rep, RefineReport) and out is rep.candidate
    md = rep.to_markdown()
    for key in ("data RMSE", "χ = RMSE/σ", "phase values", "iou", "edge_f1", "Verdict"):
        assert key in md
    d = rep.to_dict()
    json.dumps(d)  # serializable
    assert d["mode"] == "levelset" and "candidate" not in d
    assert "→" in rep.summary() and str(rep) == rep.summary()
    n_src = len(result.history["total"])
    assert len(out.history["total"]) == n_src + 40
    assert len(out.stage_results) == len(result.stage_results) + 1
    assert out.stage_results[-1]["name"] == "refine[levelset]"
    assert out.history["stage"][-1] == len(result.stage_results)


def test_sigma_override_and_argument_validation(blurred):
    problem, result, _ = blurred
    _, rep = refine_edges(problem, result, steps=20, sigma=2 * SIGMA, device="cpu", refuse=False)
    assert rep.sigma == pytest.approx(2 * SIGMA)
    assert rep.fit_before["chi"] == pytest.approx(rep.fit_before["rmse"] / (2 * SIGMA))
    with pytest.raises(ConfigError, match="unknown refinement mode"):
        refine_edges(problem, result, mode="magic")
    with pytest.raises(ConfigError, match="neural"):
        refine_edges(problem, result, mode="phasefield", representation="neural")
    with pytest.raises(ConfigError, match="no field"):
        refine_edges(problem, result, field="rho")
    with pytest.raises(ConfigError, match="bracket"):
        refine_edges(problem, result, levels=(1.1, None))  # nothing lies above 1.1
    cplx = nefi.Result({"x": torch.ones(4, 4, dtype=torch.complex64)}, {}, torch.zeros(4), {})
    with pytest.raises(ConfigError, match="complex"):
        refine_edges(problem, cplx)


# ------------------------------------------------------------------------------------------
# thresholds, levels, heads, fit
# ------------------------------------------------------------------------------------------
def test_volume_matched_threshold_recovers_a_softened_sphere():
    n = 24
    dom = Domain.unit((n, n, n))
    r = dom.coords().norm(dim=-1)
    sphere = (r < 0.55).float()
    frac = float(sphere.mean())
    blur = FFTConvolution(gaussian_kernel_fn(0.05), dom, field="x", periodic=False)
    with torch.no_grad():
        soft = blur({"x": sphere})
    # the blur conserves the integral, so the mass-implied volume is the true one
    f = mass_matched_fraction(soft, 0.0, 1.0)
    assert f == pytest.approx(frac, rel=0.02)
    tau = volume_matched_threshold(soft, f)
    mask = soft > tau
    assert int(mask.sum()) == round(f * soft.numel())
    iou = float((mask & (sphere > 0.5)).sum() / (mask | (sphere > 0.5)).sum())
    assert iou > 0.9
    # "below" counts from the other side
    tb = volume_matched_threshold(soft, 1.0 - f, above=False)
    assert int((soft < tb).sum()) == soft.numel() - round(f * soft.numel())


def test_multi_otsu_and_estimate_levels():
    torch.manual_seed(0)
    x = torch.cat(
        [torch.randn(2000) * 0.03, 0.5 + torch.randn(600) * 0.03, 1.0 + torch.randn(300) * 0.03]
    )
    t2 = otsu_threshold(torch.cat([torch.zeros(90), torch.ones(10)]))
    assert t2 == pytest.approx(0.5, abs=0.01)  # centered in the empty gap
    t3 = multi_otsu(x, 3)
    assert t3[0] == pytest.approx(0.25, abs=0.06) and t3[1] == pytest.approx(0.75, abs=0.06)
    vals, taus, how = estimate_levels(x, 3)
    assert how == "otsu" and len(vals) == 3 and len(taus) == 2
    assert vals[0] == pytest.approx(0.0, abs=0.03)  # majority: median
    assert vals[2] > 1.0  # extreme minority: its far quantile
    vals, taus, how = estimate_levels(x, (0.0, 0.5, 1.0))
    assert how == "midpoint" and taus == pytest.approx([0.25, 0.75])
    vals, taus, how = estimate_levels(x, (0.0, None), threshold="mass")
    assert how == "mass" and vals[0] == 0.0
    vals, _, _ = estimate_levels(x, "auto", value_range=(0.1, 0.8))
    assert 0.1 <= vals[0] < vals[1] <= 0.8
    with pytest.raises(ConfigError):
        estimate_levels(x, (0.5, 0.2))
    with pytest.raises(ConfigError):
        estimate_levels(x, 3, threshold=0.5)  # one threshold for three phases


def test_multiphase_head_area_rendering_gives_partial_volumes():
    n = 16
    area = MultiPhaseHead([0.0, 1.0], [0.0], eps_start=0.25, eps_end=1e-4, ndim=2)
    point = MultiPhaseHead([0.0, 1.0], [0.0], eps_start=0.25, eps_end=1e-4, render="point")
    # φ changes by 0.1 per cell (a planar interface): the cell average of the Heaviside is the
    # partial volume of the cell — ½ with the interface through the cell centre, ¼ off by ¼ cell
    for shift, frac in ((7.0, 0.5), (7.25, 0.25)):
        phi = (torch.arange(n, dtype=torch.float32) - shift)[:, None].expand(n, n) * 0.1
        xa = area(phi.unsqueeze(-1), {}, 1.0)
        assert torch.allclose(xa[7], torch.full((n,), frac), atol=1e-3)
        assert float(xa[:7].max()) < 1e-3 and float(xa[8:].min()) > 1 - 1e-3
    raw = phi.unsqueeze(-1)
    xp = point(raw, {}, 1.0)
    assert set(torch.unique(xp.round()).tolist()) <= {0.0, 1.0}
    # soft at progress 0, sharp at progress 1
    assert float((area(raw, {}, 0.0) - 0.5).abs().max()) < float((xa - 0.5).abs().max())
    # three phases, one level function; free phase values come from a raw channel
    h3 = MultiPhaseHead([0.0, 0.5, 1.0], [-0.2, 0.2], eps_end=1e-4, render="point")
    x3 = h3(torch.tensor([[-1.0], [0.0], [1.0]]), {}, 1.0)
    assert torch.allclose(x3, torch.tensor([0.0, 0.5, 1.0]), atol=1e-4)
    assert h3.phase(torch.tensor([[-1.0], [0.0], [1.0]])).tolist() == [0, 1, 2]
    hf = MultiPhaseHead([0.0, 1.0], [0.0], eps_end=1e-4, render="point", free=(1,))
    assert hf.n_in == 2
    xf = hf(torch.tensor([[-1.0, 0.3], [1.0, 0.3], [1.0, 0.7]]), {}, 1.0)
    assert torch.allclose(xf, torch.tensor([0.0, 0.3, 0.7]), atol=1e-4)
    # learnable values move only where allowed, and are clamped
    hl = MultiPhaseHead([0.0, 1.0], learn=(1,), value_range=(0.0, 1.5))
    with torch.no_grad():
        hl.delta.fill_(1.0)
    assert hl.values().tolist() == pytest.approx([0.0, 1.5])
    with pytest.raises(ConfigError):
        MultiPhaseHead([1.0, 0.0])


def test_cell_heaviside_and_double_well():
    s = torch.linspace(-1, 1, 41)
    h = torch.full_like(s, 0.5)
    avg = cell_heaviside(s, 1e-4, h)
    assert torch.allclose(avg, (s / 0.5 + 0.5).clamp(0, 1), atol=1e-3)
    assert torch.allclose(cell_heaviside(s, 0.1), torch.sigmoid(s / 0.1))
    well = DoubleWell("x", [0.0, 1.0])
    assert float(well.energy(torch.tensor([0.0, 1.0]))) == 0.0
    assert float(well.energy(torch.tensor([0.5]))) == pytest.approx(1 / 16)


def test_data_fit_chi_mask_and_per_entry_sigma():
    y = torch.ones(4, 5)
    p = y + 0.1
    fit = data_fit(p, Measurement(y, noise_std=0.1))
    assert fit["rmse"] == pytest.approx(0.1) and fit["chi"] == pytest.approx(1.0)
    assert fit["rel_rmse"] == pytest.approx(0.1)
    assert data_fit(p, Measurement(y))["chi"] is None
    sig = torch.full((4, 5), 0.05)
    sig[0] = 0.0  # unobserved rows carry σ = 0 (excluded)
    mask = torch.ones(4, 5)
    mask[0] = 0.0
    p2 = p.clone()
    p2[0] = 100.0  # masked out
    fit2 = data_fit(p2, Measurement(y, mask=mask, noise_std=sig))
    assert fit2["rmse"] == pytest.approx(0.1) and fit2["chi"] == pytest.approx(2.0)
    with pytest.raises(ConfigError):
        data_fit(torch.ones(3), Measurement(torch.ones(4)))


def test_edge_metrics_and_band_check():
    dom, gt = _phantom(24)
    soft = FFTConvolution(gaussian_kernel_fn(0.05), dom, field="x", periodic=False)(
        {"x": gt}
    ).detach()
    m = edge_metrics(soft, gt)
    assert 0 <= m["iou"] <= 1 and 0 <= m["iou_vm"] <= 1 and m["above"] == 1.0
    assert edge_metrics(gt, gt)["iou"] == 1.0 and edge_metrics(gt, gt)["edge_f1"] == 1.0
    # sign-aware: features below the background are measured on their side
    inv = 1.1 - gt
    assert edge_metrics(inv, inv)["above"] == 0.0
    # the band: the truth lies within the soft transition, an empty field does not
    dev, rows = transition_band_check(soft, gt, [0.1, 1.0])
    assert dev == 0.0 and rows[0]["core"] <= rows[0]["refined"] <= rows[0]["outer"]
    dev0, _ = transition_band_check(soft, torch.full_like(gt, 0.1), [0.1, 1.0])
    assert dev0 == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------
# instances, runs, benchmarks
# ------------------------------------------------------------------------------------------
def _tiny_deconvolution():
    from nefi.instances.deconvolution import Deconvolution

    return Deconvolution(n=16, hidden=16, depth=2, n_octaves=3, steps=(30, 60))


def test_instance_hook_and_run_output():
    inst = _tiny_deconvolution()
    assert refine_defaults(inst) == REFINE_DEFAULTS["deconvolution"](inst.cfg)
    run = inst.run(seed=0, device="cpu")
    out, rep = inst.refine(run.result, run.measurement, gt=run.gt, steps=40, device="cpu")
    assert rep.free_phases == [1] and rep.steps == 40
    assert "inst_psnr" in rep.metrics_before  # the instance's own metrics, before and after
    ro = refine_run_output(run, instance=inst, steps=40, device="cpu")
    assert ro.extra["smooth_result"] is run.result
    assert isinstance(ro.extra["refine_report"], RefineReport)
    assert set(ro.metrics) == set(run.metrics)
    assert ro.gt is run.gt and ro.measurement is run.measurement
    if ro.extra["refine_accepted"]:
        assert "refined_from" in ro.result.extra
    else:
        assert ro.result is run.result
    with pytest.raises(ConfigError, match="instance"):
        refine_run_output(run)  # a RunOutput does not know its instance
    # the gallery hook: an entry whose ``run`` carries instance, problem, result, data and GT
    from types import SimpleNamespace

    problem = inst.build_problem(run.measurement)
    entry = SimpleNamespace(
        run=SimpleNamespace(
            instance=inst,
            problem=problem,
            result=run.result,
            measurement=run.measurement,
            gt=run.gt,
        )
    )
    ge = refine_run_output(entry, steps=20, device="cpu")
    assert ge.extra["refine_report"].steps == 20 and ge.extra["smooth_result"] is run.result
    with pytest.raises(ConfigError, match="gallery entry"):
        refine_run_output(SimpleNamespace(run=None, status="failed"))
    # a mode other than levelset drops the level-set-only defaults
    _, rep2 = inst.refine(run.result, run.measurement, mode="phasefield", steps=10, device="cpu")
    assert rep2.mode == "phasefield" and rep2.free_phases == []
    assert set(REFINE_DEFAULTS) >= {
        "thermal_tomography",
        "ct3d",
        "deconvolution3d",
        "dot3d",
        "photoacoustic3d",
    }


def test_refine_method_in_a_benchmark():
    from nefi.bench import run_benchmark

    inst = _tiny_deconvolution()
    bench = run_benchmark(
        inst,
        ["neural", refine_method(steps=30)],
        n_samples=1,
        seeds=(0,),
        device="cpu",
        progress=False,
        warmup=False,
    )
    rows = {r["method"]: r for r in bench.rows}
    assert set(rows) == {"neural", "neural+refine"}
    assert not rows["neural+refine"].get("error")
    # an accepted refinement carries the merged history (its steps are part of the method's
    # cost); a refused one returns the smooth result
    n = rows["neural"]["steps"]
    assert rows["neural+refine"]["steps"] in (n, n + 30)
    assert rows["neural+refine"]["stop"].count(",") in (
        rows["neural"]["stop"].count(","),
        rows["neural"]["stop"].count(",") + 1,
    )


@pytest.mark.slow
def test_thermal_tomography_smoke_refinement_refusal_is_reported():
    from nefi.instances.thermal_tomography import ThermalTomography

    inst = ThermalTomography(preset="smoke")
    run = inst.run(seed=0, device="cpu")
    out, rep = inst.refine(run.result, run.measurement, gt=run.gt, device="cpu")
    assert rep.fit_measure == "rmse"  # noise-free synthetic data: σ unknown
    assert rep.levels_init[0] == pytest.approx(0.01)  # the configured defect diffusivity
    assert "inst_iou" in rep.metrics_before and "inst_edge_f1" in rep.metrics_after
    if rep.accepted:
        assert out.extra["refined_from"]["mode"] == "levelset"
    else:  # a refused refinement never replaces the smooth result
        assert out is run.result and "data fit" in rep.reason
