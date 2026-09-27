"""deconvolution instance: PSFs, scenes, noise models, Wiener reference, inversion, baselines."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nefi.baselines import solve
from nefi.domain import Domain
from nefi.errors import ConfigError
from nefi.instances.deconvolution import (
    OPERATOR_TAG,
    Deconvolution,
    DeconvolutionConfig,
    DeconvolutionScenes,
    make_psf,
    wiener,
)
from nefi.metrics import psnr
from nefi.operators.conv import fft_convolve, prepare_kernel
from nefi.registry import build

#: tiny but meaningful setting (≈5 s on a CPU)
SMOKE = dict(n=32, psf_sigma=0.04, hidden=64, depth=3, n_octaves=5, steps=(150, 300), lr=2e-2)


def test_psf_kernels_normalized_and_physical():
    sp = Domain.unit((32, 32)).spacing()
    for kind in ("gaussian", "motion", "disk"):
        fn = make_psf(kind, sigma=0.04, motion_length=0.2, motion_angle=0.0, disk_radius=0.08)
        k = fn(sp, (63, 63), None, torch.float64)
        assert k.shape == (63, 63) and abs(float(k.sum()) - 1.0) < 1e-9 and (k >= 0).all()
        assert float(k[31, 31]) == pytest.approx(float(k.max()))  # centered at shape // 2
    km = make_psf("motion", motion_length=0.2, motion_angle=0.0)(sp, (63, 63), None, torch.float64)
    nz = km.nonzero()
    assert set(nz[:, 1].tolist()) == {31}  # 0° motion: spread along axis 0 only
    assert int(nz[:, 0].max() - nz[:, 0].min()) >= 6  # 0.2 / (1/32) = 6.4 px long
    km90 = make_psf("motion", motion_length=0.2, motion_angle=90.0)(
        sp, (63, 63), None, torch.float64
    )
    assert torch.allclose(km90, km.T, atol=1e-12)
    kd = make_psf("disk", disk_radius=0.08)(sp, (63, 63), None, torch.float64)
    r = (kd.nonzero() - 31).double().norm(dim=1).max()
    assert 2.0 < float(r) < 3.6  # 0.08 * 32 = 2.56 px (+ anti-aliased rim)
    # physical PSF: on a 2x finer grid the Gaussian spans twice as many pixels
    kg = make_psf("gaussian", sigma=0.04)
    k32 = kg(sp, (63, 63), None, torch.float64)
    k64 = kg(Domain.unit((64, 64)).spacing(), (127, 127), None, torch.float64)
    w32 = float((k32.sum(1) * (torch.arange(63) - 31.0) ** 2).sum()) ** 0.5
    w64 = float((k64.sum(1) * (torch.arange(127) - 63.0) ** 2).sum()) ** 0.5
    assert abs(w64 / w32 - 2.0) < 0.02
    with pytest.raises(ConfigError):
        make_psf("nope")


def test_scene_classes_range_and_resolution_consistency():
    sc = DeconvolutionScenes(Domain.unit((32, 32)))
    for cls in sc.classes:
        a = sc.sample(np.random.default_rng(1), cls)["x"]
        b = sc.sample(np.random.default_rng(1), cls, (64, 64))["x"]
        assert a.shape == (32, 32) and b.shape == (64, 64)
        assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0 and float(a.max()) > 0.2
        assert torch.allclose(F.avg_pool2d(b[None, None], 2)[0, 0], a, atol=1e-5)


def test_poisson_noise_produces_counts_and_gaussian_noise_std():
    inst = Deconvolution(n=32, noise="poisson", peak_counts=50.0, background_counts=2.0)
    gt, meas = inst.make_measurement(0)
    d = meas.data
    assert (d >= 0).all() and torch.equal(d, d.round())  # non-negative integer counts
    assert meas.meta["noise"] == "poisson" and meas.meta["fidelity"] == "blur-2x-float64"
    with torch.no_grad():
        lam = inst.operator()({"x": gt["x"]})  # expected counts (native-grid physics)
    assert abs(float(d.mean()) / float(lam.mean()) - 1.0) < 0.05
    assert float(inst.measurement_image(meas).mean()) == pytest.approx(
        float((d.mean() - 2.0) / 50.0), rel=1e-6
    )
    g = Deconvolution(n=32, noise="gaussian", noise_std=0.02)
    gt, m = g.make_measurement(0)
    assert m.noise_std is not None and m.noise_std > 0
    with pytest.raises(ConfigError):
        Deconvolution(n=32, noise="gaussian", data_loss="poisson_nll").losses()


def test_inverse_crime_guard():
    inst = Deconvolution(n=32)
    gen = inst.data_generator()
    gt, meas = inst.make_measurement(0)
    prob = inst.build_problem(meas)
    assert gen.fidelity_tag == "blur-2x-float64"
    assert prob.operator.fidelity_tag == OPERATOR_TAG != gen.fidelity_tag
    assert gen.operator is not prob.operator
    assert gen.operator.domain.shape == (64, 64) and gen.dtype == torch.float64
    assert gen.supersample == 2 and meas.shape == (32, 32)


def test_wiener_reference():
    # exact inverse of a circular blur when the noise-to-signal term vanishes
    torch.manual_seed(0)
    x = torch.rand(32, 32, dtype=torch.float64)
    k = make_psf("gaussian", sigma=0.015)(
        Domain.unit((32, 32)).spacing(), (32, 32), None, torch.float64
    )
    kf, _ = prepare_kernel(k, (32, 32), periodic=True)
    y = fft_convolve(x, k, periodic=True, kernel_fft=kf, fft_shape=(32, 32))
    xr = wiener(y, k, snr=1e10, pad=0, clip=None, boundary="none")
    assert float((xr - x).abs().max()) < 1e-4
    # on instance data (non-periodic blur, 2x-grid physics) it runs, is non-negative and deblurs;
    # the SNR is chosen by the discrepancy principle
    for scene in ("smooth", "phantom"):
        inst = Deconvolution(n=32, psf_sigma=0.04, noise_std=0.01, scene=scene)
        gt, meas = inst.make_measurement(0)
        w, snr = inst.wiener_reconstruction(meas)
        assert w.shape == (32, 32) and torch.isfinite(w).all() and (w >= 0).all()
        assert 1.0 <= snr <= 1e6
        assert psnr(w, gt["x"]) > psnr(meas.data, gt["x"]) + 2.0


def test_run_smoke_recovers_phantom_better_than_blurred():
    inst = Deconvolution(**SMOKE)
    out = inst.run(seed=0, device="cpu")
    assert set(out.metrics) == {"psnr", "ssim", "mse"}
    assert out.result.fields["x"].shape == (32, 32) and out.result.pred.shape == (32, 32)
    assert [s["shape"] for s in out.result.stage_results] == [(16, 16), (32, 32)]
    blurred = psnr(inst.measurement_image(out.measurement), out.gt["x"])
    assert out.metrics["psnr"] > blurred + 2.0, (out.metrics, blurred)


def test_poisson_nll_path_runs():
    inst = Deconvolution(**{**SMOKE, "steps": (20, 40)}, noise="poisson", peak_counts=100.0)
    out = inst.run(seed=1, device="cpu")
    h = out.result.history
    assert np.isfinite(out.metrics["psnr"]) and h["data_loss"][-1] < h["data_loss"][0]
    assert float(out.result.pred.min()) >= inst.cfg.background_counts - 1e-4  # counts model


@pytest.mark.parametrize("name", ["grid", "deep_decoder", "gaussian_splat", "admm", "wiener"])
def test_baselines_build_and_solve(name):
    inst = Deconvolution(
        n=32,
        scene="sparse_dots",
        psf_sigma=0.04,
        dd_width=16,
        dd_stages=3,
        admm_outer=4,
        admm_inner=5,
        splat_primitives=16,
        splat_max_primitives=24,
    )
    gt, meas = inst.make_measurement(0)
    builders = inst.baselines()
    assert set(builders) == {"grid", "deep_decoder", "gaussian_splat", "admm", "wiener"}
    prob, cur = builders[name](meas)
    res = solve(prob, cur.scaled(0.1), device="cpu")
    x = res.fields["x"]
    assert x.shape == (32, 32) and torch.isfinite(x).all()
    if name == "wiener":
        assert torch.allclose(x, inst.wiener(meas), atol=1e-5)
    if name == "admm":
        assert float(x.min()) >= 0.0 and "primal_residual" in res.history


def test_registry_and_config():
    inst = build("instance", {"type": "deconvolution", "n": 16, "psf": "motion"})
    assert isinstance(inst.cfg, DeconvolutionConfig) and inst.cfg.psf == "motion"
    assert inst.default_curriculum().stages[0].shape == (8, 8)
    with pytest.raises(ConfigError):
        Deconvolution(n=16, head="nope").heads()
    b = Deconvolution(n=16, head="bounded")
    assert b.field().heads["x"].hi == 1.0
