"""sparse_view_ct: Radon operator, adjoint, FBP, scenes, inverse-crime guard, inversion."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nefi.baselines import solve
from nefi.domain import Domain
from nefi.errors import OperatorError
from nefi.instances.sparse_view_ct import (
    CTScenes,
    RadonOperator,
    SparseViewCT,
    SparseViewCTConfig,
    backproject,
    fbp,
    uniform_angles,
)
from nefi.metrics import psnr
from nefi.registry import build

SMOKE = dict(n=32, n_views=16, hidden=64, depth=3, n_octaves=5, steps=(150, 300), lr=2e-2, tv=2e-5)


def _smooth(t: torch.Tensor, k: int = 7) -> torch.Tensor:
    w = torch.ones(1, 1, k, k, dtype=t.dtype) / k**2
    return F.conv2d(t[None, None], w, padding=k // 2)[0, 0]


@pytest.fixture(scope="module")
def dom():
    return Domain.unit((64, 64))


def test_zero_angle_is_column_sum_and_output_shapes(dom):
    x = torch.rand(64, 64, dtype=torch.float64)
    op = RadonOperator(dom, angles=[0.0])
    s = op({"mu": x})
    assert s.shape == (1, 64)
    assert torch.allclose(s[0], x.sum(0) * op.dt, atol=1e-12)  # Δt = 1/64 physical units
    op20 = RadonOperator(dom, n_views=20, det_per_pixel=1.5)
    assert op20({"mu": x}).shape == (20, 96) == op20.output_shape((64, 64))
    op10 = op20.at_resolution((32, 32))
    assert op10.n_det == 48 and op10.output_shape((32, 32)) == (20, 48)  # n_det scales
    assert torch.allclose(op10.angles, op20.angles)
    assert op20({"mu": torch.stack([x, 2 * x])}).shape == (2, 20, 96)  # batch dims
    with pytest.raises(OperatorError):
        RadonOperator(Domain.unit((8, 16)))


def test_centered_disk_sinogram_is_angle_independent(dom):
    c = dom.coords(dtype=torch.float64)
    disk = ((c**2).sum(-1) <= 0.5**2).double()
    s = RadonOperator(dom, n_views=36)({"mu": disk})
    tot = s.sum(1)
    assert float(tot.std() / tot.mean()) < 0.01
    assert float(s.max(1).values.std() / s.max(1).values.mean()) < 0.02
    assert abs(float(s.max()) - 0.5) < 0.02  # chord through the center = diameter (0.5)


def test_backproject_is_adjoint_up_to_discretization(dom):
    torch.manual_seed(0)
    op = RadonOperator(dom, n_views=30)
    c = dom.coords(dtype=torch.float64)
    fov = ((c**2).sum(-1) <= 0.9**2).double()
    for _ in range(3):
        # non-negative smooth fields: no cancellation in the inner products
        x = _smooth(torch.rand(64, 64, dtype=torch.float64)) * fov
        y = _smooth(torch.rand(30, 64, dtype=torch.float64), 5)
        lhs = float((op({"mu": x}) * y).sum())
        rhs = float((x * op.backproject(y)).sum())
        assert abs(lhs - rhs) / abs(lhs) < 0.01, (lhs, rhs)  # pixel-driven: ~1e-4 in practice
        # signed fields: compare relative to the Cauchy-Schwarz scale ‖Rx‖‖y‖
        xs = _smooth(torch.randn(64, 64, dtype=torch.float64)) * fov
        ys = _smooth(torch.randn(30, 64, dtype=torch.float64), 5)
        rx = op({"mu": xs})
        diff = float((rx * ys).sum()) - float((xs * op.backproject(ys)).sum())
        assert abs(diff) < 0.01 * float(rx.norm() * ys.norm())
        exact = float((xs * op.adjoint(ys)).sum())  # autograd adjoint: exact
        assert abs(float((rx * ys).sum()) - exact) < 1e-9 * float(rx.norm() * ys.norm())
    # the standalone function agrees with the method
    y = torch.rand(30, 64, dtype=torch.float64)
    assert torch.allclose(backproject(y, op.angles, 64, 1.0), op.backproject(y))


def test_fbp_full_view_recovers_phantom(dom):
    gt = CTScenes(dom).sample(np.random.default_rng(0), "shepp")["mu"].double()
    op = RadonOperator(dom, n_views=180)
    sino = op({"mu": gt})
    for filt in ("ramp", "shepp-logan"):
        rec = op.fbp(sino, filt)
        assert rec.shape == (64, 64)
        assert psnr(rec, gt) > 20.0, filt
    rec = fbp(sino, uniform_angles(180), "ramp", n=64, width=1.0, circle=True)
    assert psnr(rec, gt) > 20.0
    assert float(rec[0, 0]) == 0.0  # outside the inscribed circle


def test_operator_linearity_and_homogeneity(dom):
    op = RadonOperator(dom, n_views=12)
    a, b = torch.rand(64, 64, dtype=torch.float64), torch.rand(64, 64, dtype=torch.float64)
    lhs = op({"mu": 2.5 * a - 0.7 * b})
    rhs = 2.5 * op({"mu": a}) - 0.7 * op({"mu": b})
    assert torch.allclose(lhs, rhs, atol=1e-12)
    assert op.homogeneity == 1.0


def test_scenes_inside_fov_and_resolution_consistent():
    d = Domain.unit((32, 32))
    sc = CTScenes(d)
    c = d.coords()
    outside = (c**2).sum(-1) > 1.0
    for cls in sc.classes:
        a = sc.sample(np.random.default_rng(2), cls)["mu"]
        b = sc.sample(np.random.default_rng(2), cls, (64, 64))["mu"]
        assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0 and float(a.max()) > 0.2
        assert float(a[outside].abs().max()) < 1e-3  # inside the scanner's field of view
        assert torch.allclose(F.avg_pool2d(b[None, None], 2)[0, 0], a, atol=1e-5)


def test_inverse_crime_guard_and_data():
    inst = SparseViewCT(n=32, n_views=10)
    gen = inst.data_generator()
    gt, meas = inst.make_measurement(0)
    prob = inst.build_problem(meas)
    assert gen.fidelity_tag == "radon-2x-float64" != prob.operator.fidelity_tag
    assert gen.operator is not prob.operator and gen.dtype == torch.float64
    assert gen.operator.n == 64 and gen.operator.n_det == 128  # 2x grid, 2 bins per fine pixel
    assert meas.shape == (10, 32) == prob.operator.output_shape((32, 32))
    assert meas.noise_std is not None and meas.noise_std > 0
    # coarse-stage measurement is area-averaged over detector bins
    assert prob.measurement_at((16, 16)).shape == (10, 16)


def test_run_smoke_beats_fbp():
    inst = SparseViewCT(**SMOKE)
    out = inst.run(seed=0, device="cpu")
    assert set(out.metrics) == {"psnr", "ssim", "mse"}
    assert out.result.fields["mu"].shape == (32, 32) and out.result.pred.shape == (16, 32)
    mu = out.result.fields["mu"]
    assert float(mu.min()) >= 0.0 and float(mu.max()) <= 1.0  # Bounded(0, 1) head
    fbp_psnr = psnr(inst.fbp(out.measurement), out.gt["mu"])
    assert out.metrics["psnr"] > fbp_psnr + 0.5, (out.metrics, fbp_psnr)  # observed +2.8 dB


@pytest.mark.parametrize("name", ["grid", "fbp", "deep_decoder"])
def test_baselines(name):
    inst = SparseViewCT(n=32, n_views=12, dd_width=16, dd_stages=3)
    gt, meas = inst.make_measurement(0)
    builders = inst.baselines()
    assert set(builders) == {"grid", "fbp", "deep_decoder"}
    prob, cur = builders[name](meas)
    res = solve(prob, cur.scaled(0.1), device="cpu")
    assert res.fields["mu"].shape == (32, 32) and torch.isfinite(res.fields["mu"]).all()
    if name == "fbp":
        assert res.extra["method"] == "fbp" and res.stage_results[0]["stop"] == "closed_form"
        assert torch.allclose(res.fields["mu"], inst.fbp(meas), atol=1e-5)


def test_registry_and_config():
    inst = build("instance", {"type": "sparse_view_ct", "n": 16, "n_views": 8})
    assert isinstance(inst.cfg, SparseViewCTConfig)
    op = inst.operator()
    assert op.n_views == 8 and math.isclose(float(op.angles[1]), math.pi / 8)
