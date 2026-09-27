"""dot3d: diffusion physics (flux balance, dipole reflectance), IFT gradients, calibration,
scenes, depth-decaying sensitivity, inversion."""

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from nefi.baselines import solve
from nefi.domain import Domain
from nefi.errors import ConfigError
from nefi.instances.dot3d import (
    DOT3D,
    DiffuseOpticalOperator,
    DOT3DConfig,
    boundary_conductance,
    robin_A,
)
from nefi.registry import build

DT = torch.float64


def _farrell(rho: float, musp: float, mua: float, A: float) -> float:
    """Diffuse reflectance of a semi-infinite medium (Farrell, Patterson & Wilson 1992)."""
    D = 1.0 / (3.0 * musp)
    mueff = math.sqrt(mua / D)
    z0, zb = 1.0 / musp, 2.0 * A * D
    r1, r2 = math.hypot(z0, rho), math.hypot(z0 + 2.0 * zb, rho)
    t1 = z0 * (mueff + 1.0 / r1) * math.exp(-mueff * r1) / r1**2
    t2 = (z0 + 2.0 * zb) * (mueff + 1.0 / r2) * math.exp(-mueff * r2) / r2**2
    return (t1 + t2) / (4.0 * math.pi)


def test_robin_factor():
    assert robin_A(1.0) == pytest.approx(1.0, abs=0.01)
    assert robin_A(1.4) == pytest.approx(3.25, abs=0.02)  # Groenhuis fit, tissue / air
    assert boundary_conductance(1.0, 1.0 / 3.0, 1.0) == pytest.approx(1.0 / (2.0 + 1.5))


def test_flux_balance_and_dipole_reflectance():
    musp, mua, A = 1.0, 0.01, robin_A(1.4)
    D = 1.0 / (3.0 * musp)
    dom = Domain((60, 60, 30), ((0.0, 60.0), (0.0, 60.0), (0.0, 30.0)))  # 1 mm voxels
    rhos = torch.tensor([10.0, 12.0, 15.0, 20.0], dtype=DT)
    det = torch.stack([30.0 + rhos, torch.full_like(rhos, 30.0)], -1)[:, None, :]
    op = DiffuseOpticalOperator(
        dom, [[30.0, 30.0, 1.0 / musp]], det, 0.5, D=D, A=A, tol=1e-12, warm_start=False
    ).to(DT)
    mu = torch.full(dom.shape, mua, dtype=DT)
    u = op.fluence(mu)[0]
    # exact finite-volume flux balance: absorbed + escaped through the Robin faces = source power
    vol = math.prod(dom.spacing())
    absorbed = float((mua * u).sum()) * vol
    escaped = 0.0
    for ax, h in enumerate(dom.spacing()):
        faces = u.narrow(ax, 0, 1).sum() + u.narrow(ax, dom.shape[ax] - 1, 1).sum()
        escaped += boundary_conductance(h, D, A) * (vol / h) * float(faces)
    assert absorbed + escaped == pytest.approx(1.0, abs=1e-6)
    # far-field diffuse reflectance agrees with the dipole solution (observed ratio 1.02-1.06)
    y = op({"mu_a": mu})[0, :, 0]
    ref = torch.tensor([_farrell(float(r), musp, mua, A) for r in rhos], dtype=DT)
    ratio = y / ref
    assert float(ratio.min()) > 0.95 and float(ratio.max()) < 1.1, ratio


def test_ift_gradient_matches_finite_differences():
    inst = DOT3D(preset="smoke", grid=(8, 8, 4), source_grid=(2, 1), detector_grid=(3, 3))
    op = inst.operator(tol=1e-12).to(DT)
    op.warm_start = False
    torch.manual_seed(0)
    mu = (0.01 + 0.004 * torch.rand(8, 8, 4, dtype=DT)).requires_grad_(True)
    w = torch.rand(op.output_shape((8, 8, 4)), dtype=DT)
    (g,) = torch.autograd.grad((op({"mu_a": mu}) * w).sum(), mu)
    v = torch.randn(8, 8, 4, dtype=DT)
    eps = 1e-6
    with torch.no_grad():
        fp = (op({"mu_a": mu + eps * v}) * w).sum()
        fm = (op({"mu_a": mu - eps * v}) * w).sum()
    fd = float((fp - fm) / (2 * eps))
    assert float((g * v).sum()) == pytest.approx(fd, rel=1e-5)
    assert float(g.sum()) < 0  # more absorption, less light everywhere


def test_forward_and_double_backward_jvps_match_fd():
    """Forward-mode ``torch.func.jvp`` (exact implicit rule of the elliptic core) and the
    double-backward JVP (gather-based, twice-differentiable readout) both match central
    differences — on a *fresh*, warm-started operator, whose calibration reference is first
    requested inside the transform; no ``torch.func`` wrapper may survive in its caches."""
    from nefi.diagnostics._common import jvp
    from nefi.physics.elliptic import is_transformed

    inst = DOT3D(preset="smoke", grid=(8, 8, 4), source_grid=(2, 1), detector_grid=(3, 3))
    op = inst.operator(tol=1e-12).to(DT)  # warm_start=True (the instance default)
    ref = inst.operator(tol=1e-12).to(DT)
    ref.warm_start = False
    torch.manual_seed(1)
    mu = 0.01 + 0.004 * torch.rand(8, 8, 4, dtype=DT)
    v = torch.randn(8, 8, 4, dtype=DT)

    def fn(m):
        return op({"mu_a": m})

    _, jv_fwd = torch.func.jvp(fn, (mu,), (v,))
    cached = [t for val in op._cache.values() for t in (val if isinstance(val, tuple) else (val,))]
    assert not any(is_transformed(t) for t in cached if torch.is_tensor(t))
    jv_auto, mode = jvp(fn, mu, v, "auto")
    assert mode == "forward"
    jv_db, _ = jvp(fn, mu, v, "double_backward")
    eps = 1e-6
    with torch.no_grad():
        fd = (ref({"mu_a": mu + eps * v}) - ref({"mu_a": mu - eps * v})) / (2 * eps)
    for jv in (jv_fwd, jv_auto, jv_db):
        assert float((jv - fd).norm() / fd.norm()) < 1e-5
    assert torch.allclose(op({"mu_a": mu}), ref({"mu_a": mu}), rtol=1e-8)  # plain call after


def test_calibration_inverse_crime_and_mask():
    inst = DOT3D(preset="smoke")
    op = inst.operator().to(DT)
    y0 = op({"mu_a": torch.full(inst.domain().shape, inst.cfg.mua_background, dtype=DT)})
    assert torch.allclose(y0, torch.ones_like(y0), atol=1e-6)  # calibrated background = 1
    gen = inst.data_generator()
    gt, meas = inst.make_measurement(0)
    prob = inst.build_problem(meas)
    assert gen.fidelity_tag == "dot-fv-robin-2x-float64" != prob.operator.fidelity_tag
    assert gen.operator.domain.shape == (40, 40, 20) and gen.dtype == torch.float64
    assert meas.shape == (9, 10, 10) == prob.operator.output_shape((20, 20, 10))
    assert meas.mask is not None and 0.5 < float(meas.mask.mean()) < 1.0
    # native model error of the calibrated data is below the 1 % noise (≈ 0.1-0.9 % observed)
    clean = gen.clean(
        inst.scene_generator().sample(np.random.default_rng(0), "single", (40, 40, 20))
    )
    pred = op({"mu_a": gt["mu_a"].double()})
    m = meas.mask.bool()
    assert float(((pred - clean) / clean)[m].square().mean().sqrt()) < 0.012
    assert inst.measurement_image(meas).shape == (10, 10)


def test_scenes_depth_classes_and_consistency():
    inst = DOT3D(preset="smoke")
    sc = inst.scene_generator()
    rng = np.random.default_rng(0)
    single = [sc.draw(rng, "single")[0].depth for _ in range(20)]
    deep = [sc.draw(rng, "deep")[0].depth for _ in range(20)]
    assert max(single) < min(deep)
    multi = sc.draw(np.random.default_rng(1), "multi")
    assert 2 <= len(multi) <= 3
    a = sc.sample(np.random.default_rng(2), "multi")["mu_a"]
    b = sc.sample(np.random.default_rng(2), "multi", (40, 40, 20))["mu_a"]
    assert float(a.min()) == pytest.approx(inst.cfg.mua_background, rel=1e-5)
    assert float(a.max()) <= inst.cfg.inclusion_mua[1] + 1e-6
    pooled = torch.nn.functional.avg_pool3d(b[None, None], 2)[0, 0]
    assert torch.allclose(pooled, a, atol=1e-6)


def test_sensitivity_decays_with_depth():
    inst = DOT3D(preset="smoke", grid=(16, 16, 8), detector_grid=(6, 6))
    sens = inst.sensitivity(exact=False, n_probes=16)
    profile = sens.mean(dim=(0, 1))
    assert profile.shape == (8,)
    assert float(profile[0] / profile[-1]) > 10.0  # surface data barely see the bottom
    assert float(profile[1]) > float(profile[4]) > float(profile[7])


def test_run_smoke_localizes_inclusion():
    inst = DOT3D(preset="smoke")
    out = inst.run(seed=0, device="cpu")
    keys = {"psnr", "ssim", "mse", "iou", "iou_2d", "depth_error", "depth_rmse", "contrast"}
    assert set(out.metrics) == keys
    mu = out.result.fields["mu_a"]
    assert mu.shape == (20, 20, 10)
    assert float(mu.min()) >= inst.cfg.mua_min and float(mu.max()) <= inst.cfg.mua_max
    m = out.metrics
    # observed (seed 0): IoU 0.59, depth error 1.0 mm, contrast 0.90 (grid: 0.75, 0.6, 0.64)
    assert m["iou"] > 0.4 and m["depth_error"] < 2.5 and m["contrast"] > 0.25, m


@pytest.mark.parametrize("name", ["grid", "deep_decoder"])
def test_baselines(name):
    inst = DOT3D(preset="smoke", grid=(12, 12, 6), dd_width=16, dd_stages=2)
    gt, meas = inst.make_measurement(0)
    prob, cur = inst.baselines()[name](meas)
    res = solve(prob, cur.scaled(0.05), device="cpu")
    assert res.fields["mu_a"].shape == (12, 12, 6) and torch.isfinite(res.fields["mu_a"]).all()


def test_registry_and_presets():
    inst = build("instance", {"type": "dot3d", "preset": "deep", "grid": [12, 12, 6]})
    assert isinstance(inst.cfg, DOT3DConfig) and inst.cfg.scene == "deep"
    assert inst.domain().shape == (12, 12, 6)
    assert inst.sources().shape == (9, 3) and inst.D == pytest.approx(1.0 / 3.0)
    with pytest.raises(ConfigError):
        DOT3D(mua_min=0.02)  # background below the head's range


def test_smoke_config_file_matches_preset():
    """The smoke YAML (preferred by ``nefi run --smoke``) equals PRESETS["smoke"]."""
    from nefi.cli import make_instance, read_spec_file

    spec = read_spec_file(Path(__file__).resolve().parents[1] / "configs" / "dot3d_smoke.yaml")

    def norm(cfg):
        return {k: tuple(v) if isinstance(v, list) else v for k, v in asdict(cfg).items()}

    assert norm(make_instance(spec).cfg) == norm(DOT3D(preset="smoke").cfg)
