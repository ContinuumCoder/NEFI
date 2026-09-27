"""poisson_source: DSTs, spectral solver, masked observations, data generation, inversion."""

import math

import numpy as np
import pytest
import scipy.fft as sfft
import torch

from nefi.baselines import solve
from nefi.domain import Domain
from nefi.errors import ConfigError
from nefi.instances.poisson_source import (
    PoissonOperator,
    PoissonSource,
    PoissonSourceConfig,
    dst1,
    dst2,
    fd_poisson_solve,
    idst1,
    idst2,
    masked_downsample,
    observation_mask,
)
from nefi.losses import Context
from nefi.measurement import Measurement
from nefi.registry import build

SMOKE = dict(n=32, hidden=64, depth=3, n_octaves=5, steps=(150, 300), lr=1e-2)


def test_dst_matches_scipy_and_inverts():
    torch.manual_seed(0)
    x = torch.randn(3, 17, 5, dtype=torch.float64)
    for dim in (0, 1, 2, -1):
        ref2 = torch.as_tensor(sfft.dst(x.numpy(), type=2, axis=dim)) / 2
        ref1 = torch.as_tensor(sfft.dst(x.numpy(), type=1, axis=dim)) / 2
        assert torch.allclose(dst2(x, dim), ref2, atol=1e-12)
        assert torch.allclose(dst1(x, dim), ref1, atol=1e-12)
        assert torch.allclose(idst2(dst2(x, dim), dim), x, atol=1e-12)
        assert torch.allclose(idst1(dst1(x, dim), dim), x, atol=1e-12)


@pytest.mark.parametrize("extent,kappa", [(1.0, 1.0), (2.0, 0.5)])
def test_spectral_solver_reproduces_analytic_eigenfunction(extent, kappa):
    d = Domain((64, 48), ((0.0, extent), (0.0, extent)))
    c = d.physical_coords(dtype=torch.float64)
    k = math.pi / extent
    u_exact = torch.sin(k * c[..., 0]) * torch.sin(k * c[..., 1])
    f = kappa * 2 * k**2 * u_exact  # -κΔu = f  (f = 2π² sin sin on the unit square)
    op = PoissonOperator(d, conductivity=kappa)
    u = op({"f": f})
    assert float((u - u_exact).norm() / u_exact.norm()) < 1e-3
    assert torch.allclose(op.laplacian(u), f, atol=1e-9)


def test_linearity_gradient_and_fd_cross_check():
    d = Domain.unit((24, 24))
    op = PoissonOperator(d)
    a, b = torch.rand(24, 24, dtype=torch.float64), torch.rand(24, 24, dtype=torch.float64)
    assert torch.allclose(op({"f": 3 * a - b}), 3 * op({"f": a}) - op({"f": b}), atol=1e-12)
    assert op.homogeneity == 1.0 and op.output_shape((12, 12)) == (12, 12)
    # differentiable path agrees with finite differences
    small = PoissonOperator(Domain.unit((6, 5)))
    f0 = torch.rand(6, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda f: small({"f": f}), (f0,))
    # the "fd" spectrum reproduces the 5-point finite-difference solution exactly
    u_sparse = fd_poisson_solve(a.numpy(), d.spacing())
    u_fd = PoissonOperator(d, spectrum="fd")({"f": a}).numpy()
    assert np.abs(u_fd - u_sparse).max() < 1e-10 * np.abs(u_sparse).max()
    # and the continuous spectrum converges to it at second order in h
    errs = []
    for n in (16, 32, 64):
        dn = Domain.unit((n, n))
        c = dn.physical_coords(dtype=torch.float64)
        f = torch.exp(-((c - 0.5) ** 2).sum(-1) / 0.02)
        u_c = PoissonOperator(dn)({"f": f}).numpy()
        errs.append(np.abs(u_c - fd_poisson_solve(f.numpy(), dn.spacing())).max())
    assert errs[1] < errs[0] / 3 and errs[2] < errs[1] / 3


def test_masked_mse_ignores_unobserved_pixels():
    inst = PoissonSource(n=16, data_loss="mse")
    gt, meas = inst.make_measurement(0)
    prob = inst.build_problem(meas)
    dom = prob.domain
    m = meas.mask.bool()
    pred = prob.operator({"f": gt["f"]})
    base = Context({"f": gt["f"]}, pred, meas, dom)
    loss = prob.losses.terms["data"]
    v0 = float(loss(base))
    pred2 = pred.clone()
    pred2[~m] += 100.0  # arbitrary values where nothing is observed
    data2 = meas.data.clone()
    data2[~m] = -7.0
    meas2 = Measurement(data2, meas.mask, meas.noise_std)
    assert float(loss(Context({"f": gt["f"]}, pred2, meas2, dom))) == pytest.approx(v0, rel=1e-6)
    pred3 = pred.clone()
    pred3[m] += 0.1  # observed pixels do matter
    assert float(loss(Context({"f": gt["f"]}, pred3, meas, dom))) > v0 + 1e-3
    assert float(meas.data[~m].abs().max()) == 0.0


def test_masked_downsample_uses_observed_fraction():
    data = torch.arange(16.0).reshape(4, 4)
    mask = torch.zeros(4, 4)
    mask[0, 0] = mask[0, 1] = mask[3, 3] = 1.0
    coarse = masked_downsample(Measurement(data, mask, 0.1), (2, 2))
    assert torch.allclose(coarse.mask, torch.tensor([[0.5, 0.0], [0.0, 0.25]]))
    assert coarse.data[0, 0] == pytest.approx(0.5) and coarse.data[1, 1] == pytest.approx(15.0)
    assert coarse.data[0, 1] == 0.0 and coarse.noise_std == 0.1


def test_observation_masks_data_generator_and_inverse_crime_guard():
    rng = np.random.default_rng(0)
    m = observation_mask((32, 32), rng, "random", 0.1)
    assert int(m.sum()) == round(0.1 * 1024)
    b = observation_mask((32, 32), rng, "boundary", boundary_width=2)
    assert b[0].all() and b[:, -1].all() and b[1, 5] == 1 and b[2, 5] == 0 and b[16, 16] == 0
    assert observation_mask((4, 4), rng, "full").all()
    with pytest.raises(ConfigError):
        observation_mask((4, 4), rng, "nope")
    inst = PoissonSource(n=32, obs_fraction=0.2)
    gen = inst.data_generator()
    gt, meas = inst.make_measurement(0)
    prob = inst.build_problem(meas)
    assert gen.fidelity_tag == "poisson-fd-2x-float64" != prob.operator.fidelity_tag
    assert type(gen.operator).__name__ == "FDPoissonOperator" and gen.dtype == torch.float64
    assert gen.operator.domain.shape == (64, 64)
    assert meas.shape == (32, 32) and abs(float(meas.mask.mean()) - 0.2) < 0.01
    assert meas.meta["obs_mode"] == "random" and meas.noise_std > 0
    coarse = prob.measurement_at((16, 16))  # mask-aware coarse data (fractional weights)
    assert coarse.shape == (16, 16) and float(coarse.mask.max()) <= 1.0
    assert float((coarse.mask > 0).float().mean()) > 0.4


def test_run_smoke_recovers_sources():
    inst = PoissonSource(**SMOKE)
    out = inst.run(seed=0, device="cpu")
    assert set(out.metrics) == {"psnr", "mse", "relative_error"}
    assert out.result.fields["f"].shape == (32, 32) and out.result.pred.shape == (32, 32)
    assert float(out.result.fields["f"].min()) >= 0.0  # softplus head
    h = out.result.history["data_loss"]
    assert h[-1] < 0.1 * h[0]
    assert out.metrics["relative_error"] < 0.8  # the zero source has relative error 1


@pytest.mark.parametrize("name", ["grid", "deep_decoder"])
def test_baselines(name):
    inst = PoissonSource(n=16, dd_width=16, dd_stages=3)
    gt, meas = inst.make_measurement(0)
    builders = inst.baselines()
    assert set(builders) == {"grid", "deep_decoder"}
    prob, cur = builders[name](meas)
    assert prob.downsample_obs is masked_downsample
    res = solve(prob, cur.scaled(0.05), device="cpu")
    assert res.fields["f"].shape == (16, 16) and torch.isfinite(res.fields["f"]).all()


def test_registry_and_config():
    inst = build("instance", {"type": "poisson_source", "n": 16, "obs_mode": "boundary"})
    assert isinstance(inst.cfg, PoissonSourceConfig) and inst.cfg.obs_mode == "boundary"
    gt, meas = inst.make_measurement(1)
    assert float(meas.mask[8, 8]) == 0.0 and float(meas.mask[0, 8]) == 1.0
    with pytest.raises(ConfigError):
        PoissonSource(n=16, head="nope").heads()
