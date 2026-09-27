"""nefi.autotune: gauges, convergence / budget, Morozov regularization, acquisition report,
held-out search, one-call orchestration, CLI and benchmark method.

The instance tests pin a fixed baseline configuration (the original smoke presets), so later
preset tuning does not move the baselines. Baseline metrics (seed 0, CPU):

* eit (200 steps):              PSNR 16.51 dB, inclusion IoU 0.46, RMSE/σ = 1.38
* poisson_source (450 steps):   PSNR 22.23 dB, RMSE/σ = 2.43 (2 of 3 sources)
* holography (400 steps):       PSNR 34.49 dB mean-subtracted, 6.77 dB raw (offset −0.69 rad)
* wave_fwi (2 × 8, 100 steps):  PSNR 14.09 dB, RMSE/σ = 2.92

With ``autotune_problem(level="standard")`` (docs/autotune.md): eit 16.51 → 23.08 dB (1600 steps,
IoU 0.92), poisson_source 22.23 → 27.34 dB (1600 steps, all 3 sources), holography raw
6.77 → 23.30 dB (37.62 dB mean-subtracted), wave_fwi 14.09 → 13.33 dB at RMSE/σ = 1.03 — the data
are fitted to the noise floor but the 2 × 8 acquisition cannot determine the field, which the
acquisition report flags.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import types

import pytest
import torch

import nefi
from nefi.autotune import (
    AutotuneReport,
    DirectionPenalty,
    LeastSquaresScale,
    MeanAnchor,
    SignConvention,
    acquisition_report,
    autotune,
    autotune_instance,
    autotune_problem,
    autotuned_method,
    border_region,
    compare_runs,
    detect_gauges,
    probe_convergence,
    repair_gauges,
    strip_gauge_fixes,
    tune_budget,
    tune_regularization,
)
from nefi.domain import Domain
from nefi.fields import Bounded, Heads, NeuralField
from nefi.fields.modifiers import Affine
from nefi.losses import MSE, LossSet, NormalizedMSE
from nefi.measurement import Measurement
from nefi.metrics.basic import psnr
from nefi.operators.base import LambdaOperator
from nefi.operators.conv import FFTConvolution, gaussian_kernel_fn
from nefi.problem import InverseProblem
from nefi.solve import Curriculum, EnergyScaleCorrection

EIT_SMOKE = dict(
    n=16, n_patterns=4, scene="single", noise_std=1e-3, supersample=2, hidden=64, depth=3,
    n_octaves=3, activation="relu", steps=(50, 100), lr=0.01, tv=1e-5, grid_lr=0.05,
)  # fmt: skip
POISSON_SMOKE = dict(
    n=32, scene="blobs", obs_mode="random", obs_fraction=0.1, noise_std=0.01, hidden=64,
    depth=3, n_octaves=5, steps=(150, 300), lr=1e-2, dd_width=32,
)  # fmt: skip
HOLOGRAPHY_SMOKE = dict(
    n=32, pixel_size=1.0, wavelength=0.5, distances=(10.0, 25.0, 50.0), scene="smooth_phase",
    phase_amplitude=1.0, absorption=0.0, noise_std=0.01, supersample=2, pad_factor=2.0,
    pad_mode="edge", hidden=64, depth=4, n_octaves=6, tv=1e-4, steps=(150, 250), lr=1e-2,
    gs_iterations=100,
)  # fmt: skip
WAVE_SMOKE = dict(
    n=24, extent=1.0, scene="smooth_anomaly", c_background=2.0, c_min=1.5, c_max=2.5,
    anomaly=0.4, geometry="transmission", n_sources=2, n_receivers=8, margin=0.08, f0=4.0,
    t_max=0.96, dt_obs=0.016, order=4, absorbing="pml", absorb_width=0.25, absorb_R=1e-3,
    courant=0.9, grad_mode="autograd", noise_std=0.005, supersample=2, hidden=64, depth=4,
    n_octaves=4, tv=1e-4, multiscale_frequency=True, f_low=5.0, steps=(40, 60), lr=5e-3,
)  # fmt: skip


def _known(cls, cfg: dict) -> dict:
    """Keep the config keys the instance still knows (robust to later config refactors)."""
    fields = cls.Config.__dataclass_fields__
    return {k: v for k, v in cfg.items() if k in fields}


def _toy(**over):
    from nefi.instances.toy1d import Toy1D

    cfg = dict(n=64, hidden=32, depth=3, n_octaves=5, steps=(40, 60), lr=1e-2, tv=1e-4)
    cfg.update(over)
    inst = Toy1D(**cfg)
    gt, meas = inst.make_measurement(0)
    torch.manual_seed(0)
    return inst, gt, inst.build_problem(meas)


def _holography():
    from nefi.instances.holography import Holography, HolographyConfig

    cfg = _known(Holography, HOLOGRAPHY_SMOKE)
    if "zero_mean" in HolographyConfig.__dataclass_fields__:
        cfg["zero_mean"] = False  # the raw (un-anchored) parameterization and phantoms
    inst = Holography(cfg)
    gt, meas = inst.make_measurement(0)
    torch.manual_seed(0)
    return inst, gt, strip_gauge_fixes(inst.build_problem(meas))


# ------------------------------------------------------------------------------------------
# gauges
# ------------------------------------------------------------------------------------------
def test_detect_and_repair_holography_constant_phase():
    inst, gt, problem = _holography()
    rep = detect_gauges(problem)
    g = rep["constant[phase]"]
    assert g.detected and g.invisibility < 1e-4 and g.handled_by is None
    assert g.fix == "mean_anchor"
    assert not rep["scale[phase]"].detected and not rep["sign[phase]"].detected
    assert "constant[phase]" in rep.summary()
    json.dumps(rep.to_dict())
    fixed = repair_gauges(problem, rep)
    assert fixed.field is not problem.field and rep.repairs
    assert fixed.meta["gauge_repairs"] == rep.repairs
    with torch.no_grad():
        phase = fixed.field(fixed.domain.coords(), 1.0)["phase"]
    assert abs(float(phase.mean())) < 1e-5  # anchored at the head's prior level (0)
    again = detect_gauges(fixed)
    assert again["constant[phase]"].detected and again["constant[phase]"].handled_by
    assert not again.unresolved
    assert repair_gauges(fixed, again) is fixed  # nothing left to do


def test_gauge_repair_improves_raw_holography_psnr():
    inst, gt, raw = _holography()
    fixed = repair_gauges(raw, detect_gauges(raw))
    cur = raw.curriculum  # 150 + 250 steps (the gallery budget)
    r0 = nefi.invert(raw, cur, device="cpu", seed=0)
    r1 = nefi.invert(fixed, cur, device="cpu", seed=0)
    raw0, raw1 = psnr(r0.fields["phase"], gt["phase"]), psnr(r1.fields["phase"], gt["phase"])
    assert raw1 > raw0 + 5.0  # recorded: 6.8 → 21.0 dB
    # the offset-invariant instance metric is not hurt (recorded 34.5 → 34.9 dB)
    assert inst.evaluate(r1, gt)["psnr"] > inst.evaluate(r0, gt)["psnr"] - 1.0


def test_scale_gauge_of_a_max_normalized_fidelity():
    inst, gt, base = _toy(steps=(100, 200))
    problem = dataclasses.replace(base, losses=LossSet({"data": NormalizedMSE(normalize="max")}))
    rep = detect_gauges(problem)
    g = rep["scale[x]"]
    assert g.detected and g.degree == 1.0 and g.fix == "energy_scale" and g.handled_by is None
    assert not rep["constant[x]"].detected
    fixed = repair_gauges(problem, rep)
    assert any(isinstance(pp, EnergyScaleCorrection) for pp in fixed.postprocess)
    assert detect_gauges(fixed)["scale[x]"].handled_by == "EnergyScaleCorrection"
    r0 = nefi.invert(problem, device="cpu", seed=0)
    r1 = nefi.invert(fixed, device="cpu", seed=0)

    def rel(r):
        return float((r.fields["x"] - gt["x"]).norm() / gt["x"].norm())

    assert "scale_factor" in r1.post_info and rel(r1) < 0.5 * rel(r0)
    # an MSE fidelity pins the scale: homogeneous of degree 1, but not a gauge
    assert not detect_gauges(base)["scale[x]"].detected


def test_scale_gauge_handled_by_netmy_scale_correction():
    from nefi.cli import load_spec, make_instance

    inst = make_instance(load_spec("nv_relaxometry", smoke=True))
    gt, meas = inst.make_measurement(0)
    rep = detect_gauges(inst.build_problem(meas))
    g = rep["scale[rho]"]
    assert g.detected and g.degree == 1.0 and g.handled_by == "EnergyScaleCorrection"
    assert not rep.unresolved


def test_sign_and_custom_gauges():
    dom = Domain.unit((48,))
    blur = FFTConvolution(gaussian_kernel_fn(0.04), dom, field="x")
    x_true = torch.sin(torch.linspace(0, 6, 48))
    y = blur({"x": x_true}) ** 2 + 0.01 * torch.randn(48)
    field = NeuralField(1, Heads({"x": Affine(init_value=0.0)}), hidden=16, depth=2, n_octaves=3)
    op = LambdaOperator(lambda f: blur({"x": f["x"]}) ** 2, primary="x")
    problem = InverseProblem(dom, field, op, LossSet({"data": MSE()}), Measurement(y, None, 0.01))
    rep = detect_gauges(problem)
    assert rep["sign[x]"].detected and rep["sign[x]"].fix == "sign_convention"
    assert rep["scale[x]"].degree == 2.0 and not rep["scale[x]"].detected  # MSE pins the scale
    fixed = repair_gauges(problem, rep)
    assert any(isinstance(pp, SignConvention) for pp in fixed.postprocess)
    res = nefi.invert(fixed, Curriculum.single(steps=20, lr=1e-2), device="cpu")
    assert "sign_flipped" in res.post_info
    # two unknowns seen only through their sum: the exchange a ↔ b is invisible
    heads = Heads({"a": Affine(init_value=0.0), "b": Affine(init_value=0.0)})
    two = NeuralField(1, heads, hidden=16, depth=2, n_octaves=3)
    op2 = LambdaOperator(lambda f: blur({"x": f["a"] + f["b"]}), primary="a")
    meas2 = Measurement(blur({"x": x_true}) + 0.01 * torch.randn(48), None, 0.01)
    p2 = InverseProblem(dom, two, op2, LossSet({"data": MSE()}), meas2)
    rep2 = detect_gauges(p2, candidates={"exchange": {"a": 1.0, "b": -1.0}, "shift": {"a": 1.0}})
    assert rep2["custom:exchange"].detected and not rep2["custom:shift"].detected
    fixed2 = repair_gauges(p2, rep2)
    assert isinstance(fixed2.losses.terms["gauge_exchange"], DirectionPenalty)
    nefi.invert(fixed2, Curriculum.single(steps=20, lr=1e-2), device="cpu")


def test_gauge_fix_building_blocks():
    head = MeanAnchor(Bounded(-3.0, 3.0), value=0.5)
    assert abs(float(head(torch.randn(8, 8, 1)).mean()) - 0.5) < 1e-5
    region = border_region((8, 8))
    assert int(region.sum()) == 28  # one-cell frame
    framed = MeanAnchor(Bounded(-3.0, 3.0), 0.0, region)
    assert abs(float(framed(torch.randn(8, 8, 1))[region].mean())) < 1e-5
    fine = framed(torch.randn(16, 16, 1))  # the region follows the grid (two-cell frame)
    assert abs(float(fine[border_region((16, 16))].mean())) < 1e-5
    y = torch.linspace(-1.0, 1.0, 10)
    fake = types.SimpleNamespace(
        measurement=Measurement(y), field=types.SimpleNamespace(primary="x")
    )
    out, info = LeastSquaresScale(homogeneity=1.0)({"x": 2 * y}, 4 * y, fake, (10,))
    assert info["scale_factor"] == pytest.approx(0.25) and torch.allclose(out["x"], 0.5 * y)
    out, info = SignConvention()({"x": -(y**3)}, y, fake, (10,))
    assert info["sign_flipped"] is False  # the largest |x| is already positive (x[0] = 1)


# ------------------------------------------------------------------------------------------
# budget / learning rate
# ------------------------------------------------------------------------------------------
def test_probe_convergence_statuses():
    inst, gt, problem = _toy()
    # a deliberately tiny budget: the misfit still falls fast -> converging, budget raised
    tiny = probe_convergence(problem, steps=(10, 20, 40), extend=0, lr_factors=())
    assert tiny.status == "converging" and tiny.recommended_steps >= 80
    assert tiny.chi > 1.5 and tiny.sigma == pytest.approx(float(problem.measurement.noise_std))
    # a generous budget reaches the noise floor: that validated schedule is recommended
    big = probe_convergence(problem, steps=(100, 200, 400), extend=2, lr_factors=())
    assert big.status == "at_noise_floor" and big.recommended_steps == big.reached_floor_at
    cur = tune_budget(problem, big)
    ref = next(p for p in big.probes if p.steps == big.reached_floor_at)
    assert cur.discrepancy_tau == pytest.approx(min(1.0, ref.chi))  # stop at the validated misfit
    assert abs(cur.total_steps - big.recommended_steps) <= 2
    assert [s.shape for s in cur.stages] == [s.shape for s in problem.curriculum.stages]
    # a crippled configuration (learning rate 1e-6) cannot move the misfit -> stalled
    _, _, frozen = _toy(lr=1e-6)
    stall = probe_convergence(frozen, steps=(20, 40, 80), lr_factors=(), lr_test=False)
    assert stall.status == "stalled" and "stalled" in stall.reasons[-1]
    assert stall.recommended_steps >= frozen.curriculum.total_steps  # never shortened
    json.dumps(stall.to_dict())
    assert "χ = RMSE/σ" in big.to_markdown()


# ------------------------------------------------------------------------------------------
# regularization
# ------------------------------------------------------------------------------------------
def test_tune_regularization_recovers_tv_weight_on_toy1d():
    # 100× too strong TV; the PSNR-best weight of a sweep on this problem is 1e-3 (34.6 dB)
    inst, gt, problem = _toy(steps=(300, 600), tv=1e-1)
    res = tune_regularization(problem, ["tv"])
    assert res.status == "discrepancy"
    assert 1e-4 <= res["tv"] <= 1e-2
    assert problem.losses.weights["tv"] == pytest.approx(1e-1)  # the input is untouched
    tuned, cur = res.apply(problem)
    r_wrong = nefi.invert(problem, device="cpu", seed=0)
    r_tuned = nefi.invert(tuned, cur, device="cpu", seed=0)
    assert psnr(r_tuned.fields["x"], gt["x"]) > psnr(r_wrong.fields["x"], gt["x"]) + 5.0
    json.dumps(res.to_dict())
    # without a noise level: GradNorm shares instead of the discrepancy principle
    blind = dataclasses.replace(
        problem, measurement=Measurement(problem.measurement.data, None, None, {})
    )
    gn = tune_regularization(blind, ["tv"], budget_scale=0.1)
    assert gn.status == "gradnorm" and gn["tv"] < 1e-1


def test_tune_regularization_poisson_l1():
    from nefi.instances.poisson_source import PoissonSource

    inst = PoissonSource(_known(PoissonSource, {**POISSON_SMOKE, "l1": 10.0}))  # 10⁴× too strong
    gt, meas = inst.make_measurement(0)
    torch.manual_seed(0)
    problem = inst.build_problem(meas)
    res = tune_regularization(problem, ["l1"], budget_scale=0.5)
    assert res["l1"] <= 0.1  # lowered ≥ 100× (recorded: 10 → 1e-3, "budget_limited")
    tuned, cur = res.apply(problem)
    r0 = nefi.invert(problem, device="cpu", seed=0)
    r1 = nefi.invert(tuned, cur, device="cpu", seed=0)
    assert inst.evaluate(r1, gt)["psnr"] > inst.evaluate(r0, gt)["psnr"] + 3.0


# ------------------------------------------------------------------------------------------
# acquisition
# ------------------------------------------------------------------------------------------
def test_acquisition_report_flags_the_sparse_wave_geometry():
    from nefi.instances.wave_fwi import WaveFWI

    def report(**over):
        inst = WaveFWI(_known(WaveFWI, {**WAVE_SMOKE, **over}))
        gt, meas = inst.make_measurement(0)
        return acquisition_report(inst.build_problem(meas), k=6, n_probes=8, match=False)

    sparse = report()  # 2 sources × 8 receivers, cross-well transmission
    assert sparse.flagged and sparse.criteria["sv_decay"] != "well-determined"
    text = sparse.to_markdown()
    assert "low-z" in text and "high-z" in text  # the sides without sensors
    assert any("sources" in r for r in sparse.recommendations)
    assert any("cannot change the acquisition" in r for r in sparse.recommendations)
    surround = report(geometry="surround", n_sources=8, n_receivers=32)
    assert not surround.flagged, surround.to_markdown()
    assert surround.sv_decay > sparse.sv_decay + 0.2
    json.dumps(sparse.to_dict())


def test_acquisition_report_data_count():
    from nefi.instances.poisson_source import PoissonSource

    inst = PoissonSource(_known(PoissonSource, POISSON_SMOKE))
    gt, meas = inst.make_measurement(0)
    rep = acquisition_report(inst.build_problem(meas), k=4, n_probes=4, match=False)
    assert rep.verdict == "severely under-determined" and rep.data_ratio < 0.25
    assert any("sparsity" in r for r in rep.recommendations)
    _, _, toy = _toy()
    ok = acquisition_report(toy, k=4, n_probes=4)
    assert not ok.flagged and not any("cannot change" in r for r in ok.recommendations)


# ------------------------------------------------------------------------------------------
# held-out search
# ------------------------------------------------------------------------------------------
def test_heldout_search_on_toy1d():
    from nefi.instances.toy1d import Toy1D

    base = dict(n=64, hidden=32, depth=3, n_octaves=5, steps=(300, 600), lr=1e-2, tv=1e-1)
    inst = Toy1D(**base)
    gt, meas = inst.make_measurement(0)

    def factory(**hp):
        return Toy1D(**{**base, **hp}).build_problem(meas)

    rep = autotune(factory, trials=6, seed=0)
    assert set(rep.space) >= {"lr", "weight.tv", "anneal_fraction", "n_octaves"}
    assert rep.trials[0].params == {} and rep.trials[0].score == rep.default_score
    assert rep.best_score < 0.5 * rep.default_score and rep.best["weight.tv"] < 0.1
    problem, cur = rep.build(factory)
    r_best = nefi.invert(problem, cur, device="cpu", seed=0)
    r_default = nefi.invert(factory(), device="cpu", seed=0)
    assert psnr(r_best.fields["x"], gt["x"]) > psnr(r_default.fields["x"], gt["x"]) + 3.0
    assert "(default)" in rep.table()
    json.dumps(rep.to_dict())
    disc = autotune(
        factory, trials=3, objective="discrepancy", space={"weight.tv": ("log", 1e-4, 1e-1)}
    )
    assert disc.objective == "discrepancy" and len(disc.trials) == 3


# ------------------------------------------------------------------------------------------
# orchestration on the smoke presets (numbers at the start: see the module docstring)
# ------------------------------------------------------------------------------------------
def test_autotune_problem_improves_the_eit_smoke_preset():
    from nefi.instances.eit import EIT

    inst = EIT(_known(EIT, EIT_SMOKE))
    gt, meas = inst.make_measurement(0)
    torch.manual_seed(0)
    problem = inst.build_problem(meas)
    base = inst.default_curriculum().scaled(200 / 150)  # the gallery budget: 67 + 133 steps
    # one doubling fewer than "quick" to keep the suite fast: probes 50 … 400 steps
    tuned, cur, report = autotune_problem(
        problem, level="quick", curriculum=base, options={"extend": 1}
    )
    comp = compare_runs(inst, gt, problem, base, tuned, cur)
    default, best = comp["default"], comp["tuned"]
    assert default["steps"] == 200
    assert best["metrics"]["psnr"] > default["metrics"]["psnr"] + 2.0
    assert best["metrics"]["psnr"] > 16.51 + 2.0  # recorded default at the start
    assert best["chi"] < default["chi"]
    assert report.convergence.recommended_steps > 200 and report.acquisition is not None
    assert isinstance(report, AutotuneReport) and "## Budget and learning rate" in (
        report.to_markdown()
    )
    json.dumps(report.to_dict())


def test_autotune_problem_improves_the_poisson_source_smoke_preset():
    from nefi.instances.poisson_source import PoissonSource

    inst = PoissonSource(_known(PoissonSource, POISSON_SMOKE))
    gt, meas = inst.make_measurement(0)
    torch.manual_seed(0)
    problem = inst.build_problem(meas)
    tuned, cur, report = autotune_problem(problem, level="quick")
    comp = compare_runs(inst, gt, problem, problem.curriculum, tuned, cur)
    assert comp["default"]["steps"] == 450
    assert comp["tuned"]["metrics"]["psnr"] > comp["default"]["metrics"]["psnr"] + 2.0
    assert comp["tuned"]["metrics"]["psnr"] > 22.23 + 2.0  # recorded default at the start
    assert report.acquisition.verdict == "severely under-determined"
    assert problem.measurement.noise_std == tuned.measurement.noise_std


@pytest.mark.slow
def test_autotune_standard_level_on_the_gallery_failures():
    """The numbers of docs/autotune.md (level="standard", seed 0)."""
    from nefi.instances.eit import EIT
    from nefi.instances.poisson_source import PoissonSource

    cases = [
        (EIT(_known(EIT, EIT_SMOKE)), 200 / 150, 16.51),
        (PoissonSource(_known(PoissonSource, POISSON_SMOKE)), 1.0, 22.23),
    ]
    for inst, scale, recorded in cases:
        gt, meas = inst.make_measurement(0)
        torch.manual_seed(0)
        problem = inst.build_problem(meas)
        base = inst.default_curriculum().scaled(scale)
        tuned, cur, report = autotune_problem(problem, level="standard", curriculum=base)
        comp = compare_runs(inst, gt, problem, base, tuned, cur)
        assert comp["tuned"]["metrics"]["psnr"] > recorded + 3.0, report.to_markdown()
        assert report.convergence.status == "at_noise_floor"
    inst, gt, raw = _holography()
    tuned, cur, report = autotune_problem(raw, level="standard")
    comp = compare_runs(inst, gt, raw, raw.curriculum, tuned, cur)
    assert comp["tuned"]["metrics"]["raw_psnr"] > comp["default"]["metrics"]["raw_psnr"] + 10.0
    assert comp["tuned"]["metrics"]["psnr"] > comp["default"]["metrics"]["psnr"]


def test_autotune_instance_and_benchmark_method():
    from nefi.bench.protocol import default_method, run_benchmark
    from nefi.instances.toy1d import Toy1D

    inst = Toy1D(n=32, hidden=16, depth=2, n_octaves=4, steps=(20, 30))
    tuned, cur, rep = autotune_instance(inst, level="quick", compare=True)
    assert set(rep.comparison) >= {"default", "tuned", "probe_steps", "tuning_seconds"}
    assert rep.meta["instance"] == "toy1d" and rep.probe_steps > 0
    assert tuned.curriculum is cur and "autotune" in tuned.meta
    assert nefi.autotune_problem is autotune_problem
    method = autotuned_method("quick", options={"extend": 0})
    res = run_benchmark(
        inst, [default_method(), method], n_samples=1, seeds=[0], progress=False,
        device="cpu", warmup=False,
    )  # fmt: skip
    rows = {r["method"]: r for r in res.rows}
    assert rows["autotuned-quick"]["error"] is None and rows["neural"]["error"] is None


# ------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------
BACKENDS = ["argparse"] + (["typer"] if importlib.util.find_spec("typer") else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_cli_autotune(backend, tmp_path, monkeypatch, capsys):
    from nefi.cli import main, read_spec_file

    monkeypatch.setenv("NEFI_CLI", backend)
    out = tmp_path / "at"
    args = ["autotune", "toy1d", "--smoke", "--level", "quick", "--out", str(out)]
    assert main([*args, "--set", "n=32", "-q"]) == 0
    text = capsys.readouterr().out
    assert "autotune[quick]" in text and "saved" in text
    for name in ("autotune.md", "autotune.json", "tuned_config.yaml"):
        assert (out / name).exists(), name
    data = json.loads((out / "autotune.json").read_text())
    assert data["level"] == "quick" and data["decisions"]
    spec = read_spec_file(out / "tuned_config.yaml")
    assert spec.instance == "toy1d" and spec.curriculum and spec.config["n"] == 32
    assert main(["run", str(out / "tuned_config.yaml"), "--out", str(tmp_path / "run"), "-q"]) == 0
    assert main(["autotune", "toy1d", "--level", "bogus", "-q"]) == 2
