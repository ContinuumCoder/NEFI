"""ct3d: 3-D Radon stack, adjoint, slice-wise FBP, scenes, inverse-crime guard, inversion."""

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nefi.baselines import solve
from nefi.domain import Domain
from nefi.errors import ConfigError, OperatorError
from nefi.instances.ct3d import (
    CT3D,
    PRESETS,
    CT3DConfig,
    CT3DScenes,
    Radon3DOperator,
    backproject3d,
    fbp3d,
    uniform_angles,
)
from nefi.instances.sparse_view_ct import RadonOperator, fbp
from nefi.metrics import psnr, ssim
from nefi.registry import build

DT = torch.float64


@pytest.fixture(scope="module")
def dom():
    return Domain((32, 32, 8), ((0.0, 1.0), (0.0, 1.0), (0.0, 0.5)))


def _smooth(t: torch.Tensor, k: int = 5) -> torch.Tensor:
    w = torch.ones(1, 1, k, k, k, dtype=t.dtype) / k**3
    return F.conv3d(t[None, None], w, padding=k // 2)[0, 0]


def test_radon3d_equals_slicewise_2d_radon(dom):
    torch.manual_seed(0)
    x = torch.rand(32, 32, 8, dtype=DT)
    angles = uniform_angles(10)
    op = Radon3DOperator(dom, angles=angles, det_per_pixel=1.5, samples_per_pixel=1.5)
    y = op({"mu": x})
    assert y.shape == (10, 48, 8) == op.output_shape((32, 32, 8))
    ref = RadonOperator(
        Domain.unit((32, 32)), angles=angles, det_per_pixel=1.5, samples_per_pixel=1.5
    )
    for k in range(8):
        assert torch.allclose(y[..., k], ref({"mu": x[..., k]}), atol=1e-12)
    # batch axes, view chunking, resolution change
    assert op({"mu": torch.stack([x, 2 * x])}).shape == (2, 10, 48, 8)
    chunked = Radon3DOperator(
        dom, angles=angles, det_per_pixel=1.5, samples_per_pixel=1.5, view_batch=3
    )
    assert torch.allclose(chunked({"mu": x}), y, atol=1e-12)
    coarse = op.at_resolution((16, 16, 4))
    assert coarse is op.at_resolution((16, 16, 4)) and coarse.n_det == 24
    assert coarse({"mu": x[::2, ::2, ::2]}).shape == (10, 24, 4) == op.output_shape((16, 16, 4))
    with pytest.raises(OperatorError):
        Radon3DOperator(Domain.unit((8, 16, 4)))


def test_adjoint_linearity_and_backprojector(dom):
    torch.manual_seed(1)
    op = Radon3DOperator(dom, n_views=24)
    c = dom.coords(dtype=DT)
    fov = ((c[..., :2] ** 2).sum(-1) <= 0.85**2).to(DT)
    x = _smooth(torch.rand(32, 32, 8, dtype=DT)) * fov
    y = torch.rand(24, 32, 8, dtype=DT)
    rx = op({"mu": x})
    exact = float((x * op.adjoint(y)).sum())
    assert abs(float((rx * y).sum()) - exact) < 1e-10 * float(rx.norm() * y.norm())
    approx = float((x * op.backproject(y)).sum())
    assert abs(float((rx * y).sum()) - approx) / abs(exact) < 0.02  # pixel-driven Rᵀ
    b = torch.rand(32, 32, 8, dtype=DT)
    assert torch.allclose(op({"mu": 2.5 * x - 0.7 * b}), 2.5 * rx - 0.7 * op({"mu": b}), atol=1e-12)
    assert op.homogeneity == 1.0 and op.batchable
    assert torch.allclose(backproject3d(y, op.angles, 32), op.backproject(y))


def test_fbp3d_equals_slicewise_fbp_and_recovers_phantom():
    d = Domain.unit((48, 48, 8))
    gt = CT3DScenes(d).sample(np.random.default_rng(0), "ellipsoids")["mu"].double()
    op = Radon3DOperator(d, n_views=90)
    sino = op({"mu": gt})
    rec = fbp3d(sino, op.angles, n=48)
    ref = torch.stack([fbp(sino[..., k], op.angles, n=48) for k in range(8)], -1)
    assert torch.allclose(rec, ref, atol=1e-10)
    assert psnr(rec, gt) > 20.0  # full-view FBP (observed ≈ 24 dB)
    circ = op.fbp(sino, circle=True)
    assert float(circ[0, 0].abs().max()) == 0.0


def test_scenes_in_fov_and_resolution_consistent():
    d = Domain((16, 16, 8), ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
    sc = CT3DScenes(d)
    c = d.coords()
    outside = (c[..., :2] ** 2).sum(-1) > 1.0
    for cls in sc.classes:
        a = sc.sample(np.random.default_rng(3), cls)["mu"]
        b = sc.sample(np.random.default_rng(3), cls, (32, 32, 16))["mu"]
        assert a.shape == (16, 16, 8) and float(a.min()) >= 0.0 and float(a.max()) <= 1.0
        assert float(a.max()) > 0.25, cls
        assert float(a[outside].abs().max()) < 1e-3, cls  # inside the scanner cylinder
        assert torch.allclose(F.avg_pool3d(b[None, None], 2)[0, 0], a, atol=1e-5), cls


def test_inverse_crime_guard_and_measurement():
    inst = CT3D(preset="smoke", n=16, n_z=8, n_views=6)
    gen = inst.data_generator()
    gt, meas = inst.make_measurement(0)
    prob = inst.build_problem(meas)
    assert gen.fidelity_tag == "radon3d-2x-float64" != prob.operator.fidelity_tag
    assert gen.operator is not prob.operator and gen.dtype == torch.float64
    assert gen.operator.n == 32 and gen.operator.n_det == 64 and gen.operator.n_z == 16
    assert meas.shape == (6, 16, 8) == prob.operator.output_shape((16, 16, 8))
    assert meas.noise_std is not None and meas.noise_std > 0
    assert prob.measurement_at((8, 8, 4)).shape == (6, 8, 4)  # area-averaged bins and slices
    assert inst.measurement_image(meas).shape == (6, 16)


def test_run_smoke_beats_fbp():
    inst = CT3D(preset="smoke")
    out = inst.run(seed=0, device="cpu")
    assert set(out.metrics) == {"psnr", "ssim", "mse", "iou"}
    mu = out.result.fields["mu"]
    assert mu.shape == (32, 32, 16) and out.result.pred.shape == (12, 32, 16)
    assert float(mu.min()) >= 0.0 and float(mu.max()) <= 1.0  # Bounded(0, 1) head
    rec = inst.fbp(out.measurement)
    # observed (seed 0): NF 27.3 dB / SSIM 0.85 vs FBP 25.1 dB / 0.65
    assert out.metrics["ssim"] > ssim(rec, out.gt["mu"]) + 0.05, out.metrics
    assert out.metrics["psnr"] > psnr(rec, out.gt["mu"]), out.metrics


@pytest.mark.parametrize("name", ["grid", "fbp3d", "deep_decoder"])
def test_baselines(name):
    inst = CT3D(preset="smoke", n=16, n_z=8, n_views=8, dd_width=16, dd_stages=2)
    gt, meas = inst.make_measurement(0)
    builders = inst.baselines()
    assert set(builders) == {"grid", "fbp3d", "deep_decoder"}
    prob, cur = builders[name](meas)
    res = solve(prob, cur.scaled(0.1), device="cpu")
    assert res.fields["mu"].shape == (16, 16, 8) and torch.isfinite(res.fields["mu"]).all()
    if name == "fbp3d":
        assert res.extra["method"] == "fbp3d" and res.stage_results[0]["stop"] == "closed_form"
        assert torch.allclose(res.fields["mu"], inst.fbp(meas), atol=1e-5)


def test_registry_presets_and_config():
    inst = build("instance", {"type": "ct3d", "preset": "smoke", "n_views": 8})
    assert isinstance(inst.cfg, CT3DConfig) and inst.cfg.n == 32 and inst.cfg.n_views == 8
    op = inst.operator()
    assert op.n_views == 8 and math.isclose(float(op.angles[1]), math.pi / 8)
    limited = CT3D(preset="limited_angle")
    assert limited.cfg.angle_range == PRESETS["limited_angle"]["angle_range"] < 180.0
    assert float(limited.angles().max()) < math.radians(120.0)
    with pytest.raises(ConfigError):
        CT3D(preset="nope")


def test_smoke_config_file_matches_preset():
    """The smoke YAML (preferred by ``nefi run --smoke``) equals PRESETS["smoke"]."""
    from nefi.cli import make_instance, read_spec_file

    spec = read_spec_file(Path(__file__).resolve().parents[1] / "configs" / "ct3d_smoke.yaml")

    def norm(cfg):
        return {k: tuple(v) if isinstance(v, list) else v for k, v in asdict(cfg).items()}

    assert norm(make_instance(spec).cfg) == norm(CT3D(preset="smoke").cfg)
