"""deconvolution3d: anisotropic 3-D PSF, scenes, inverse-crime guard, classical references, run."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nefi.baselines import solve
from nefi.instances.deconvolution3d import (
    Deconvolution3D,
    Deconvolution3DConfig,
    Deconvolution3DScenes,
    richardson_lucy,
    wiener3d,
)
from nefi.metrics import psnr
from nefi.operators.conv import fft_convolve, prepare_kernel
from nefi.registry import build

DT = torch.float64


def test_psf_is_axially_elongated_and_normalized():
    inst = Deconvolution3D(preset="smoke")
    sx, sy, sz = inst.psf_sigma()
    assert sx == sy and abs(sz / sx - 3.0) < 1e-12
    blur = inst.blur()
    k = blur.kernel((32, 32, 16), dtype=DT)
    assert abs(float(k.sum()) - 1.0) < 1e-10
    # physical second moments of the sampled kernel match σ_xy / σ_z (µm)
    dom = inst.domain()
    from nefi.operators.conv import kernel_offsets

    r = kernel_offsets(dom.spacing(), tuple(k.shape), dtype=DT)
    var = [(k * r[..., a] ** 2).sum() for a in range(3)]
    assert abs(float(var[0]) ** 0.5 / sx - 1.0) < 0.05
    assert abs(float(var[2]) ** 0.5 / sz - 1.0) < 0.05
    # a point source is blurred 3× more along z (in µm) than laterally
    x = torch.zeros(32, 32, 16, dtype=DT)
    x[16, 16, 8] = 1.0
    y = blur({"x": x})
    assert abs(float(y.sum()) - 1.0) < 1e-4  # only the Gaussian tail beyond the stack is lost
    assert float(y[16, 16, 9] / y[16, 16, 8]) > float(y[18, 16, 8] / y[16, 16, 8])


def test_linearity_and_exact_adjoint_symmetry():
    inst = Deconvolution3D(preset="smoke", n=16, n_z=8)
    blur = inst.blur().to(DT)
    a, b = torch.rand(16, 16, 8, dtype=DT), torch.rand(16, 16, 8, dtype=DT)
    assert torch.allclose(blur({"x": 2 * a - 3 * b}), 2 * blur({"x": a}) - 3 * blur({"x": b}))
    # the full-extent Gaussian kernel is symmetric, so the zero-padded blur is self-adjoint
    lhs = float((blur({"x": a}) * b).sum())
    rhs = float((a * blur({"x": b})).sum())
    assert abs(lhs - rhs) < 1e-10 * abs(lhs)


def test_scenes_consistent_across_resolutions():
    inst = Deconvolution3D(preset="smoke", n=16, n_z=8)
    sc = inst.scene_generator()
    assert isinstance(sc, Deconvolution3DScenes)
    for cls in sc.classes:
        a = sc.sample(np.random.default_rng(1), cls)["x"]
        b = sc.sample(np.random.default_rng(1), cls, (32, 32, 16))["x"]
        assert a.shape == (16, 16, 8) and float(a.min()) >= 0.0 and float(a.max()) > 0.1, cls
        assert torch.allclose(F.avg_pool3d(b[None, None], 2)[0, 0], a, atol=1e-5), cls


@pytest.mark.parametrize("noise", ["gaussian", "poisson"])
def test_inverse_crime_guard_and_measurement(noise):
    inst = Deconvolution3D(preset="smoke", n=16, n_z=8, noise=noise)
    gen = inst.data_generator()
    gt, meas = inst.make_measurement(0)
    prob = inst.build_problem(meas)
    assert gen.fidelity_tag == "blur3d-2x-float64" != prob.operator.fidelity_tag
    assert gen.operator.domain.shape == (32, 32, 16) and gen.dtype == torch.float64
    assert meas.shape == (16, 16, 8)
    if noise == "poisson":
        assert float(meas.data.min()) >= 0 and float(meas.data.max()) > 50  # photon counts
        assert type(prob.losses.terms["data"]).__name__ == "PoissonNLL"
    # the model error of the inversion operator is far below the noise (2× grid, same PSF)
    fine = inst.scene_generator().sample(np.random.default_rng(0), inst.cfg.scene, (32, 32, 16))
    clean = F.avg_pool3d(gen.clean(fine)[None, None], 2)[0, 0]
    native = inst.blur().to(DT)({"x": gt["x"].double()})
    assert float((native - clean).abs().max()) < 5e-3 * float(clean.max())
    assert inst.measurement_image(meas).shape == (16, 16)


def test_classical_references_deblur():
    inst = Deconvolution3D(preset="smoke", scene="puncta")
    gt, meas = inst.make_measurement(0)
    g = gt["x"]
    blurred = psnr(meas.data, g)
    assert psnr(inst.wiener(meas), g) > blurred + 1.0
    assert psnr(inst.richardson_lucy(meas), g) > blurred + 1.0
    # the N-D Wiener filter inverts a periodic blur of a band-limited field (no boundary handling)
    blur = inst.blur().to(DT)
    k = blur.kernel((32, 32, 16), dtype=DT)
    kf, _ = prepare_kernel(k, (32, 32, 16), periodic=True)
    torch.manual_seed(0)
    smooth = fft_convolve(torch.randn(32, 32, 16, dtype=DT), k, periodic=True)
    y = torch.fft.irfftn(torch.fft.rfftn(smooth) * kf, s=(32, 32, 16))
    rec = wiener3d(y, k, snr=1e8, clip=None, boundary="none", pad=(0, 0, 0))
    assert float((rec - smooth).norm() / smooth.norm()) < 1e-3
    x = blur({"x": g.double()})
    rl = richardson_lucy(x, blur, n_iter=5)
    assert float(rl.min()) >= 0.0  # EM keeps positivity and the predicted flux Σ A x = Σ y
    assert abs(float(blur({"x": rl}).sum() / x.sum()) - 1.0) < 0.01


def test_run_smoke():
    inst = Deconvolution3D(preset="smoke")
    out = inst.run(seed=0, device="cpu")
    assert set(out.metrics) == {"psnr", "ssim", "mse"}
    x = out.result.fields["x"]
    assert x.shape == (32, 32, 16) and float(x.min()) >= 0.0  # Softplus head
    blurred = psnr(inst.measurement_stack(out.measurement), out.gt["x"])
    assert out.metrics["psnr"] > blurred + 2.0, (out.metrics, blurred)  # observed ≈ +3 dB


@pytest.mark.parametrize("name", ["grid", "wiener3d", "richardson_lucy", "deep_decoder"])
def test_baselines(name):
    inst = Deconvolution3D(preset="smoke", n=16, n_z=8, dd_width=16, dd_stages=2)
    gt, meas = inst.make_measurement(0)
    builders = inst.baselines()
    assert set(builders) == {"grid", "wiener3d", "richardson_lucy", "deep_decoder"}
    prob, cur = builders[name](meas)
    res = solve(prob, cur.scaled(0.1), device="cpu")
    assert res.fields["x"].shape == (16, 16, 8) and torch.isfinite(res.fields["x"]).all()
    if name in ("wiener3d", "richardson_lucy"):
        assert res.stage_results[0]["stop"] == "closed_form"


def test_registry_and_presets():
    inst = build("instance", {"type": "deconvolution3d", "preset": "poisson", "n": 16, "n_z": 8})
    assert isinstance(inst.cfg, Deconvolution3DConfig) and inst.cfg.noise == "poisson"
    assert Deconvolution3D(preset="puncta").cfg.scene == "puncta"
    dom = Deconvolution3D(n=16, n_z=8, pixel_size=0.1, z_step=0.25).domain()
    assert dom.spacing() == pytest.approx((0.1, 0.1, 0.25))


def test_smoke_config_file_matches_preset():
    """The smoke YAML (preferred by ``nefi run --smoke``) equals PRESETS["smoke"]."""
    from nefi.cli import make_instance, read_spec_file

    spec = read_spec_file(
        Path(__file__).resolve().parents[1] / "configs" / "deconvolution3d_smoke.yaml"
    )

    def norm(cfg):
        return {k: tuple(v) if isinstance(v, list) else v for k, v in asdict(cfg).items()}

    assert norm(make_instance(spec).cfg) == norm(Deconvolution3D(preset="smoke").cfg)
