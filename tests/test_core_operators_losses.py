import numpy as np
import pytest
import torch
from scipy.signal import convolve as sp_convolve

from nefi.domain import Domain
from nefi.errors import ConfigError
from nefi.losses import L1, MSE, TV, Context, Laplacian, LogMSE, LossSet, NormalizedMSE
from nefi.measurement import Measurement
from nefi.operators import (
    FFTConvolution,
    LambdaOperator,
    Nuisance,
    fft_convolve,
    gaussian_kernel_fn,
)
from nefi.operators.conv import kernel_offsets
from nefi.solve.postprocess import EnergyScaleCorrection


@pytest.mark.parametrize("nd", [1, 2])
def test_fft_convolve_matches_scipy_same(nd):
    torch.manual_seed(0)
    n = (17,) if nd == 1 else (12, 15)
    m = (7,) if nd == 1 else (5, 9)
    x = torch.randn(*n, dtype=torch.float64)
    k = torch.randn(*m, dtype=torch.float64)
    y = fft_convolve(x, k, periodic=False)
    ref = sp_convolve(x.numpy(), k.numpy(), mode="same")
    assert np.allclose(y.numpy(), ref, atol=1e-10)
    # batch dims
    xb = torch.randn(3, *n, dtype=torch.float64)
    yb = fft_convolve(xb, k)
    assert yb.shape == (3, *n)
    assert np.allclose(
        yb[1].numpy(), sp_convolve(xb[1].numpy(), k.numpy(), mode="same"), atol=1e-10
    )


def test_fft_convolve_periodic_matches_numpy():
    torch.manual_seed(0)
    n = (16, 16)
    x = torch.randn(*n, dtype=torch.float64)
    k = torch.zeros(*n, dtype=torch.float64)
    k[8, 8] = 1.0  # delta at center -> identity
    k[8, 9] = 0.5  # plus a shifted copy
    y = fft_convolve(x, k, periodic=True)
    ref = x + 0.5 * torch.roll(x, shifts=1, dims=1)
    assert torch.allclose(y, ref, atol=1e-10)


def test_fft_convolution_operator_multiscale_and_homogeneity():
    dom = Domain.unit((32, 32))
    op = FFTConvolution(gaussian_kernel_fn(0.05), dom, field="x")
    x = torch.rand(32, 32)
    y = op({"x": x})
    assert y.shape == (32, 32)
    assert torch.allclose(op({"x": 3 * x}), 3 * y, atol=1e-5)
    op16 = op.at_resolution((16, 16))
    assert op16.domain.shape == (16, 16)
    y16 = op16({"x": torch.rand(16, 16)})
    assert y16.shape == (16, 16)
    # kernel normalization preserved across resolutions (integral ~ 1)
    assert abs(float(op.kernel().sum()) - 1.0) < 1e-4
    assert abs(float(op16.kernel().sum()) - 1.0) < 1e-4
    # offsets grid centered at the kernel center
    r = kernel_offsets((0.1, 0.1), (5, 5))
    assert torch.allclose(r[2, 2], torch.zeros(2))


def test_nuisance_shares_parameters_across_resolutions():
    dom = Domain.unit((16,))
    op = Nuisance(FFTConvolution(gaussian_kernel_fn(0.05), dom), gain=True, offset=True)
    op8 = op.at_resolution((8,))
    assert op8.log_gain is op.log_gain and op8.offset is op.offset
    with torch.no_grad():
        op.log_gain.fill_(np.log(2.0))
    x = torch.ones(8)
    assert torch.allclose(op8({"x": x}), 2 * op8.inner({"x": x}), atol=1e-6)


def _ctx(fields, pred, obs, dom):
    return Context(fields, pred, Measurement(obs), dom)


def test_tv_and_l1_values():
    dom = Domain.unit((10,))
    x = torch.zeros(10)
    x[5:] = 1.0  # single step of height 1 at spacing 0.1 -> gradient 10 on one cell -> mean 1.0
    ctx = _ctx({"x": x}, x, x, dom)
    assert abs(float(TV("x", isotropic=True, eps=1e-9)(ctx)) - 1.0) < 1e-6
    assert abs(float(TV("x", isotropic=False)(ctx)) - 1.0) < 1e-6
    # a constant field has zero gradient, so the smoothed TV equals its eps (≈ 1e-9 here)
    assert float(TV("x", eps=1e-9)(_ctx({"x": torch.ones(10)}, x, x, dom))) < 1e-6
    assert abs(float(L1("x")(ctx)) - 0.5) < 1e-6
    # periodic axis sees the wrap-around jump too
    assert abs(float(TV("x", isotropic=False, periodic_axes=(0,))(ctx)) - 2.0) < 1e-6
    lap0 = Laplacian("x")(_ctx({"x": torch.ones(10)}, x, x, dom))
    assert float(lap0) < 1e-10  # constant field
    # x = t^2 on a periodic axis is not periodic, so use a cosine: Δ cos(2πt) = -(2π)^2 cos(2πt)
    t = dom.physical_coords()[..., 0]
    c = torch.cos(2 * torch.pi * t)
    lap_c = Laplacian("x", periodic_axes=(0,))(_ctx({"x": c}, x, x, dom))
    expected = float(((2 * torch.pi) ** 4 * c**2).mean())
    assert abs(float(lap_c) / expected - 1.0) < 0.15  # second-order finite difference at n=10


def test_log_mse_and_normalized_mse_invariances():
    dom = Domain.unit((4, 4))
    pred = torch.rand(3, 4, 4) + 0.1
    obs = pred * 7.0  # scale-invariant after max-normalization
    ctx = _ctx({"x": pred[0]}, pred, obs, dom)
    assert float(LogMSE(normalize="max", reduce_axes=(0,))(ctx)) < 1e-10
    assert float(NormalizedMSE(normalize="mean", reduce_axes=(0,))(ctx)) < 1e-10
    assert (
        float(LogMSE(normalize="max", reduce_axes=(0,))(_ctx({"x": pred[0]}, pred, pred + 1, dom)))
        > 0
    )


def test_lossset_weights_overrides_and_data_terms():
    dom = Domain.unit((8,))
    x = torch.rand(8)
    ls = LossSet({"data": MSE(), "tv": TV("x"), "l1": L1("x")}, weights={"tv": 0.5, "l1": 0.0})
    ctx = _ctx({"x": x}, x, x + 0.1, dom)
    total, comps = ls(ctx)
    assert "l1" not in comps and set(comps) == {"data", "tv"}
    assert ls.data_terms() == ("data",)
    assert abs(ls.data_loss(comps) - comps["data"]) < 1e-12
    ls2 = ls.with_weights({"l1": 2.0})
    assert ls2.weights["l1"] == 2.0 and ls.weights["l1"] == 0.0 and ls2.terms is ls.terms
    with pytest.raises(ConfigError):
        ls.with_weights({"nope": 1.0})
    ls.auto_balance(ctx)
    total2, comps2 = ls(ctx)
    assert abs(float(total2) - 2.0) < 1e-4  # two active terms, each balanced to 1


def test_energy_scale_correction_linear_and_quadratic():
    from nefi.fields import GridField, Heads
    from nefi.problem import InverseProblem

    dom = Domain.unit((16,))
    gt = torch.rand(16) + 0.5
    for p in (1.0, 2.0):
        op = LambdaOperator(lambda f, p=p: f["x"] ** p, homogeneity=p)
        meas = Measurement(op({"x": gt}))
        field = GridField((16,), Heads({"x": "identity"}), init=0.0)
        prob = InverseProblem(
            dom, field, op, LossSet({"data": MSE()}), meas, postprocess=[EnergyScaleCorrection()]
        )
        est = gt / 3.0  # right shape, wrong scale
        pred = op({"x": est})
        out, info = EnergyScaleCorrection()({"x": est}, pred, prob, (16,))
        assert torch.allclose(out["x"], gt, atol=1e-5), p
        assert abs(info["scale_factor"] - 3.0) < 1e-5
