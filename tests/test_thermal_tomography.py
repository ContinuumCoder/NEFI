"""Tests of the NeFTY thermal-tomography instance, its scenes, projections and metrics."""

import math

import numpy as np
import pytest
import torch

import nefi
from nefi.errors import ConfigError
from nefi.instances.thermal_tomography import (
    PRESETS,
    ThermalScenes,
    ThermalTomography,
    ThermalTomographyConfig,
    bulk_profile,
    defect_mask_2d,
    depth_map_25d,
    gt_depth_map,
    surface_downsample,
)
from nefi.metrics import (
    abs_rel,
    binary_dilation,
    delta_threshold,
    depth_rmse,
    edge_f1,
    iou_below,
    radial_power_spectrum,
)
from nefi.registry import build, list_registered
from nefi.solve import Curriculum, Stage
from nefi.utils.seed import seed_everything


@pytest.fixture(autouse=True, scope="module")
def _single_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


# ---------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------
def test_paper_config_mirrors_table5():
    inst = build("instance", {"type": "thermal_tomography"})
    c = inst.cfg
    assert (
        isinstance(c, ThermalTomographyConfig)
        and "thermal_tomography" in list_registered("instance")["instance"]
    )
    assert tuple(c.extent) == (10.0, 10.0, 1.0) and tuple(c.grid) == (64, 64, 16)
    assert inst.domain().spacing() == pytest.approx((0.15625, 0.15625, 0.0625))
    assert (c.dt, c.n_frames, c.jacobi_iters, c.face_mode) == (0.05, 100, 50, "harmonic")
    assert (c.alpha_min, c.alpha_max) == (0.003, 0.25)
    assert (c.depth, c.hidden, c.n_octaves, c.skip_at, c.activation) == (10, 512, 12, 4, "relu")
    assert (c.optimizer, c.lr, c.lr_gamma, c.lr_step_size) == ("adam", 5e-5, 0.1, 1000)
    assert (c.steps, c.anneal_steps, c.tv_weight, c.tv_eps) == (10000, 2500, 1e-3, 1e-6)
    assert tuple(c.alpha_base_range) == (0.1, 0.2)
    assert tuple(c.alpha_defect_range) == (0.005, 0.015)
    assert tuple(c.n_defects) == (1, 4) and tuple(c.n_layers) == (3, 4) and c.iou_tau == 0.03
    cur = inst.default_curriculum()
    (st,) = cur.stages
    assert st.shape == (64, 64, 16) and st.steps == 10000 and st.lr_schedule == "step"
    assert st.anneal_fraction == pytest.approx(0.25)
    assert st.lr_at(999) == pytest.approx(5e-5) and st.lr_at(1000) == pytest.approx(5e-6)
    assert cur.optim.optimizer == "adam" and cur.optim.weight_decay == 0.0
    op = inst.operator()
    assert op.n_obs == 100 and op.output_shape((64, 64, 16)) == (100, 64, 64)
    assert op.bc.kinds == ("periodic", "periodic", "neumann")
    field = inst.field()
    assert field.encoding.out_dim == 2 * 3 * 12  # Eq. (20): d_γ = 2DN, no raw coordinates
    assert inst.losses().weights == {"data": 1.0, "tv": 1e-3}


def test_presets_and_overrides():
    s = ThermalTomography(preset="smoke", steps=7)
    assert tuple(s.cfg.grid) == tuple(PRESETS["smoke"]["grid"]) and s.cfg.steps == 7
    s2 = build("instance", {"type": "thermal_tomography", "preset": "smoke", "hidden": 8})
    assert s2.cfg.hidden == 8 and tuple(s2.cfg.grid) == tuple(PRESETS["smoke"]["grid"])
    assert ThermalTomography(preset="layered").cfg.scene == "layered"
    with pytest.raises(ConfigError):
        ThermalTomography(preset="nope")
    with pytest.raises(ConfigError):
        ThermalTomography(alpha_init=0.3)
    two = ThermalTomography(preset="smoke", n_stages=2, stage_steps=(3, 4))
    cur = two.default_curriculum()
    assert [st.shape for st in cur.stages] == [(8, 8, 4), (16, 16, 6)]
    assert [st.steps for st in cur.stages] == [3, 4]
    robin = ThermalTomography(preset="smoke", bc_back="robin", robin_h=0.3, initial="flash")
    op = robin.operator()
    assert op.bc.has_robin and op.bc.robin_h == 0.3
    assert torch.allclose(op.initial_state()[0, 0], op.initial_state()[3, 5])  # uniform flash


# ---------------------------------------------------------------------------------------------
# scenes (App. E.1)
# ---------------------------------------------------------------------------------------------
def test_scene_generator_classes_ranges_and_resolution():
    inst = ThermalTomography(preset="smoke")
    gen = inst.scene_generator()
    assert isinstance(gen, ThermalScenes) and gen.classes == ("homogeneous", "layered")
    for seed in range(6):
        spec = gen.draw(np.random.default_rng(seed), "homogeneous")
        assert len(spec.layer_alpha) == 1 and 0.1 <= spec.layer_alpha[0] <= 0.2
        lo, hi = PRESETS["smoke"]["n_defects"]
        assert lo <= len(spec.defects) <= hi
        for d in spec.defects:
            assert 0.005 <= d.alpha <= 0.015 and d.kind in ("ellipsoid", "cylinder", "box")
            assert d.half_thickness <= d.center[-1] <= 1.0 - d.half_thickness
        fields, meta = gen.rasterize(spec)
        a = fields["alpha"]
        assert a.shape == (16, 16, 6) and bool(meta["defect_mask"].any())
        assert torch.allclose(a[~meta["defect_mask"]], meta["alpha_base"][~meta["defect_mask"]])
        assert float(a.min()) >= 0.005 - 1e-7 and float(a.max()) <= 0.2 + 1e-7
    lay = gen.draw(np.random.default_rng(1), "layered")
    assert 3 <= len(lay.layer_alpha) <= 4 and len(lay.layer_bounds) == len(lay.layer_alpha) - 1
    base = gen.rasterize(lay)[1]["alpha_base"]
    assert len(torch.unique(base[0, 0])) == len(lay.layer_alpha)
    # the same scene at a finer resolution (honours ``shape``); draws do not depend on the shape
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    coarse = gen.sample(r1, "homogeneous")["alpha"]
    fine = gen.sample(r2, "homogeneous", shape=(32, 32, 12))["alpha"]
    assert fine.shape == (32, 32, 12) and r1.random() == r2.random()
    frac_c, frac_f = float((coarse < 0.03).float().mean()), float((fine < 0.03).float().mean())
    assert abs(frac_c - frac_f) < 0.5 * max(frac_c, frac_f)
    with pytest.raises(ConfigError):
        gen.sample(np.random.default_rng(0), "voids")
    paper = ThermalTomography().scene_generator()
    full = paper.sample(np.random.default_rng(0), "layered")["alpha"]
    assert full.shape == (64, 64, 16)


def test_make_measurement_uses_independent_explicit_simulator():
    inst = ThermalTomography(preset="smoke", n_frames=6, noise_std=0.01)
    gt, meas = inst.make_measurement(seed=0)
    assert gt["alpha"].shape == (16, 16, 6) and meas.shape == (6, 16, 16)
    assert meas.meta["fidelity"] == "explicit-substepped-float64"
    assert meas.meta["fidelity"] != inst.operator().fidelity_tag
    assert meas.noise_std is not None and meas.noise_std > 0
    assert meas.meta["defect_mask"].shape == (16, 16, 6) and "scene" in meas.meta
    # reproducible
    gt2, meas2 = inst.make_measurement(seed=0)
    assert torch.equal(gt["alpha"], gt2["alpha"]) and torch.equal(meas.data, meas2.data)
    # supersampled simulation is area-averaged to the native measurement shape
    ss = ThermalTomography(preset="smoke", n_frames=4, sim_supersample=2)
    gt3, meas3 = ss.make_measurement(seed=0)
    assert gt3["alpha"].shape == (16, 16, 6) and meas3.shape == (4, 16, 16)


# ---------------------------------------------------------------------------------------------
# projections (App. H)
# ---------------------------------------------------------------------------------------------
def _box_field():
    a = torch.full((20, 20, 10), 0.15)
    a[5:10, 6:12, 4:7] = 0.01  # depths 0.45, 0.55, 0.65 (H = 1)
    return a


def test_defect_mask_2d_and_depth_map():
    a = _box_field()
    g = torch.Generator().manual_seed(0)
    noisy = a + 0.002 * torch.randn(a.shape, generator=g)
    gt2d = torch.zeros(20, 20, dtype=torch.bool)
    gt2d[5:10, 6:12] = True
    # k = 2 (paper): every defect pixel, ≈ 2.3 % one-sided false positives among sound pixels
    mask, info = defect_mask_2d(noisy, 0.15, return_info=True)
    assert bool(mask[gt2d].all()) and float(mask[~gt2d].float().mean()) < 0.05
    assert info["sigma"] == pytest.approx(0.002 / math.sqrt(10) / 0.15, rel=0.3)
    # the literal clipped-deficit MAD collapses and floods the sound region
    clipped = defect_mask_2d(noisy, 0.15, mad_of="clipped", min_sigma=0.0)
    assert float(clipped[~gt2d].float().mean()) > 0.2
    mask = defect_mask_2d(noisy, 0.15, k=3.0)
    assert torch.equal(mask, gt2d)
    assert torch.equal(defect_mask_2d(noisy, k=3.0), gt2d)  # label-free bulk estimate
    depth = depth_map_25d(noisy, 0.15, mask, thickness=1.0, k=3.0)
    assert torch.isnan(depth[~mask]).all()
    assert torch.allclose(depth[mask], torch.full((30,), 0.55, dtype=depth.dtype), atol=1e-6)
    top = depth_map_25d(noisy, None, None, thickness=1.0, k=3.0, reduce="min")
    assert torch.allclose(top[mask], torch.full((30,), 0.45, dtype=top.dtype), atol=1e-6)
    gtd = gt_depth_map(a < 0.03, thickness=1.0)
    assert torch.allclose(gtd[mask], depth[mask]) and torch.isnan(gtd[~mask]).all()
    # layered bulk: the per-depth median profile is recovered
    lay = torch.full((20, 20, 10), 0.12)
    lay[..., 5:] = 0.18
    lay[2:4, 2:4, 6] = 0.01
    assert torch.allclose(bulk_profile(lay)[0, 0], lay[10, 10])


# ---------------------------------------------------------------------------------------------
# metrics (App. E.2 / G.5)
# ---------------------------------------------------------------------------------------------
def test_segmentation_and_depth_metrics():
    a = _box_field()
    assert iou_below(a, a) == 1.0 and iou_below(torch.full_like(a, 0.15), a) == 0.0
    shifted = torch.roll(a, 1, 0)
    assert iou_below(shifted, a) == pytest.approx(4 / 6)
    assert edge_f1(a, a) == 1.0
    assert edge_f1(torch.full_like(a, 0.15), a) == 0.0
    assert edge_f1(shifted, a, dilate=1) == 1.0  # within the one-voxel tolerance
    assert edge_f1(torch.roll(a, 4, 0), a, dilate=1) < 0.8
    assert binary_dilation(a < 0.03, 1).sum() > (a < 0.03).sum()
    d = torch.tensor([1.0, 2.0, 4.0, 0.0])  # the 0 depth is invalid and ignored
    assert abs_rel(d * 1.1, d) == pytest.approx(0.1)
    assert depth_rmse(d + 1, d) == pytest.approx(1.0)
    assert delta_threshold(d * 1.2, d) == 1.0 and delta_threshold(d * 1.3, d) == 0.0
    assert delta_threshold(d * 1.3, d, k=2) == 1.0
    assert math.isnan(abs_rel(d, torch.zeros(4)))
    m = torch.tensor([True, True, False, False])
    assert abs_rel(torch.tensor([1.5, 2.0, 9.0, 9.0]), d, mask=m) == pytest.approx(0.25)


def test_radial_power_spectrum_peaks_at_signal_frequency():
    n = 32
    x = torch.arange(n, dtype=torch.float64)
    field = torch.cos(2 * math.pi * 4 * x / n)[:, None].expand(n, n)  # 4 cycles per 32 samples
    f, p = radial_power_spectrum(field, n_bins=16)
    assert f.shape == p.shape == (16,)
    peak = float(f[torch.nan_to_num(p, nan=-1).argmax()])
    assert abs(peak - 4 / n) < 1 / n
    # physical units: Nyquist of Δz = 1/16 is 8 cycles per unit length (App. E.2)
    f3, p3 = radial_power_spectrum(torch.rand(16, 16, 8), spacing=(10 / 64, 10 / 64, 1 / 16))
    assert float(f3.max()) <= math.sqrt(3.2**2 * 2 + 8**2) + 1e-9


# ---------------------------------------------------------------------------------------------
# problem plumbing
# ---------------------------------------------------------------------------------------------
def test_problem_multiscale_measurement_and_grid_baseline():
    inst = ThermalTomography(preset="smoke", n_frames=6, steps=4)
    gt, meas = inst.make_measurement(seed=1)
    prob = inst.build_problem(meas)
    assert prob.downsample_obs is surface_downsample
    m8 = prob.measurement_at((8, 8, 4))
    assert m8.shape == (6, 8, 8)
    avg = meas.data.reshape(6, 8, 2, 8, 2).mean(dim=(2, 4))
    assert torch.allclose(m8.data, avg, atol=1e-5)
    # the default path of InverseProblem (operator.output_shape + Measurement.resampled) agrees
    assert torch.allclose(
        meas.resampled(prob.operator.output_shape((8, 8, 4))).data, avg, atol=1e-5
    )
    fields, pred = prob.evaluate((8, 8, 4))
    assert fields["alpha"].shape == (8, 8, 4) and pred.shape == (6, 8, 8)
    total, comps = prob.loss((8, 8, 4))
    assert total.requires_grad and set(comps) == {"data", "tv"}
    # Grid Opt. baseline (App. F.2): same operator, bounded head, TV; optimizes a voxel grid
    gprob, gcur = inst.baselines()["grid"](meas)
    assert isinstance(gprob.field, nefi.GridField) and gcur.stages[0].lr == inst.cfg.grid_lr
    res = nefi.invert(
        gprob, Curriculum([Stage("g", (16, 16, 6), 6, 3e-2, "constant")]), device="cpu"
    )
    assert res.history["data"][-1] < res.history["data"][0]
    assert res.fields["alpha"].shape == (16, 16, 6)
    # coarse-to-fine NeFTY curriculum: operator.at_resolution + lateral surface downsampling
    ms = ThermalTomography(preset="smoke", n_frames=6, n_stages=2, stage_steps=(4, 4), lr=1e-2)
    seed_everything(0)
    mprob = ms.build_problem(meas)
    mres = nefi.invert(mprob, ms.default_curriculum(), device="cpu", seed=0)
    assert [st["shape"] for st in mres.stage_results] == [(8, 8, 4), (16, 16, 6)]
    assert mres.fields["alpha"].shape == (16, 16, 6) and mres.pred.shape == (6, 16, 16)
    assert all(math.isfinite(v) for v in mres.history["data"])


# ---------------------------------------------------------------------------------------------
# end-to-end smoke run (validation items 7 and 8)
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def smoke_run():
    inst = ThermalTomography(preset="smoke")
    out = inst.run(seed=0, device="cpu")
    return inst, out


def test_smoke_end_to_end(smoke_run):
    inst, out = smoke_run
    c = inst.cfg
    res = out.result
    assert res.fields["alpha"].shape == tuple(c.grid)
    assert res.pred.shape == (c.n_frames, *c.grid[:-1])
    assert len(res.history["data"]) == c.steps
    h = res.history["data"]
    assert np.mean(h[-10:]) < 0.1 * np.mean(h[:10]), (h[:3], h[-3:])
    a = res.fields["alpha"]
    assert float(a.min()) >= c.alpha_min and float(a.max()) <= c.alpha_max
    init = torch.full(tuple(c.grid), c.alpha_init)
    iou0 = iou_below(init, out.gt["alpha"], c.iou_tau)
    assert out.metrics["iou"] > iou0 + 0.1, out.metrics
    for key in ("mse", "psnr", "ssim", "iou", "edge_f1", "iou_2d", "surface_psnr"):
        assert key in out.metrics and math.isfinite(out.metrics[key])
    # reproducible field initialization and optimization
    seed_everything(0)


def test_data_fit_paradox_guard(smoke_run):
    """The recovered field re-simulates the surface far better than the initial uniform field."""
    inst, out = smoke_run
    m = out.metrics
    assert m["surface_psnr"] > m["surface_psnr_init"] + 10.0, m
    assert m["surface_mse"] < 0.1 * m["surface_mse_init"]


def test_configs_match_dataclass_and_presets():
    """``configs/thermal_tomography_{paper,smoke}.yaml`` spell out exactly the code defaults."""
    from pathlib import Path

    from nefi.config import load_config, to_dict

    root = Path(__file__).resolve().parents[1] / "configs"
    paper = load_config(root / "thermal_tomography_paper.yaml")["instance"]
    assert paper.pop("type") == "thermal_tomography"
    # the paper config compiles the Jacobi sweeps (CUDA servers); the code default stays eager
    assert paper.pop("compile_solver") is True and ThermalTomographyConfig().compile_solver is False
    assert paper == {
        k: v for k, v in to_dict(ThermalTomographyConfig()).items() if k != "compile_solver"
    }
    smoke = load_config(root / "thermal_tomography_smoke.yaml")["instance"]
    assert smoke.pop("type") == "thermal_tomography"
    assert smoke == to_dict(PRESETS["smoke"])


def test_pinn_soft_baseline_decouples_data_from_diffusivity():
    """NeFTY §3.3 / App. C.1: with a temperature surrogate, ∇_θ L_data ≡ 0 — the surface data reach
    α_θ only through the PDE residual; α_θ stays near-uniform while T_φ fits the surface."""
    inst = ThermalTomography(preset="smoke", pinn_collocation=512, pinn_steps=60)
    gt, meas = inst.make_measurement(seed=0)
    seed_everything(0)
    prob, cur = inst.baselines()["pinn_soft"](meas)
    ctx = prob.context(progress=0.5)
    theta = list(prob.field.parameters())
    for name, coupled in (("data", False), ("ic", False), ("pde", True)):
        grads = torch.autograd.grad(
            prob.losses.terms[name](ctx), theta, allow_unused=True, retain_graph=True
        )
        assert any(g is not None and float(g.abs().sum()) > 0 for g in grads) == coupled, name
    # the hard-constrained problem, in contrast, sends the data gradient to θ
    nctx = inst.build_problem(meas).context(progress=0.5)
    ntheta = [p for p in nctx.field_module.parameters()]
    g = torch.autograd.grad(inst.losses().terms["data"](nctx), ntheta, allow_unused=True)
    assert sum(float(x.abs().sum()) for x in g if x is not None) > 0
    res = nefi.invert(prob, cur, device="cpu", seed=0)
    h = res.history["data"]
    assert h[-1] < 0.2 * h[0]  # T_φ fits the surface data ...
    a = res.fields["alpha"]
    assert float(a.std()) < 0.01  # ... while α_θ stays a near-trivial constant
    assert iou_below(a, gt["alpha"], inst.cfg.iou_tau) == 0.0


@pytest.mark.slow
def test_smoke_reaches_iou():
    inst = ThermalTomography(preset="smoke", steps=200)  # ≈ 35 s on an single-threaded CPU
    ious = [inst.run(seed=s, device="cpu").metrics["iou"] for s in (0, 1)]
    assert np.mean(ious) >= 0.2 and min(ious) > 0.1, ious
