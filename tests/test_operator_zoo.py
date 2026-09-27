"""Operator zoo: FunctionOperator, linear / nonlinear operators, TimeStepper (+ BYOP cases c-e)."""

import math
from functools import partial

import pytest
import torch

import nefi
from nefi.domain import Domain
from nefi.errors import OperatorError, ShapeError
from nefi.measurement import Measurement
from nefi.metrics import psnr
from nefi.operators import (
    BeerLambert,
    Downsample,
    FourierSampling,
    FunctionOperator,
    Identity,
    PhaseRetrieval,
    Pointwise,
    Sampling,
    Saturation,
    Sequential,
    Stack,
    Sum,
    TimeStepper,
    UpsampleToNative,
    advection_diffusion_step,
    random_kspace_mask,
    stable_dt,
    wave_initial_state,
    wave_step,
)


# --------------------------------------------------------------------------------------------
# FunctionOperator
# --------------------------------------------------------------------------------------------
def _blur1d():
    k = torch.tensor([0.25, 0.5, 0.25]).view(1, 1, 3)
    return lambda x: torch.nn.functional.conv1d(x.view(1, 1, -1), k, padding=1).view(-1)


def test_function_operator_resolution_contract():
    blur = _blur1d()
    op = FunctionOperator(blur, native_shape=(16,), homogeneity=1.0)
    assert op.at_resolution((16,)) is op and not op.multiscale_capable
    with pytest.raises(ShapeError, match="at_resolution"):
        op.at_resolution((8,))
    up = FunctionOperator(blur, native_shape=(16,), coarse="upsample", output_shape=(16,))
    coarse = up.at_resolution((8,))
    assert isinstance(coarse, UpsampleToNative) and up.output_shape((8,)) == (16,)
    assert coarse({"x": torch.ones(8)}).shape == (16,)
    calls = []

    def at_res(shape):
        calls.append(shape)
        return blur

    ms = FunctionOperator(blur, native_shape=(16,), at_resolution=at_res, output_shape=lambda s: s)
    assert ms.at_resolution((8,))({"x": torch.ones(8)}).shape == (8,)
    assert ms.at_resolution((8,)) is ms.at_resolution((8,)) and calls == [(8,)]  # cached
    assert ms.output_shape((8,)) == (8,) and ms.multiscale_capable


def test_function_operator_dicts_modules_complex_and_errors():
    op = FunctionOperator(lambda f: f["a"] * f["b"], takes_dict=True, fields=("a", "b"), field="a")
    assert op.required_fields() == ("a", "b")
    assert torch.equal(op({"a": torch.ones(3), "b": torch.full((3,), 2.0)}), torch.full((3,), 2.0))
    with pytest.raises(OperatorError, match="needs fields"):
        op({"a": torch.ones(3)})
    lin = torch.nn.Linear(4, 2)
    frozen = FunctionOperator(lin)
    assert not any(p.requires_grad for p in frozen.parameters())
    assert all(
        p.requires_grad
        for p in FunctionOperator(torch.nn.Linear(4, 2), trainable=True).parameters()
    )
    cplx = FunctionOperator(lambda x: torch.fft.fft(x))
    assert cplx({"x": torch.rand(8)}).shape == (8, 2)
    with pytest.raises(OperatorError, match="torch.Tensor"):
        FunctionOperator(lambda x: x.detach().numpy())({"x": torch.rand(3)})


# --------------------------------------------------------------------------------------------
# linear operators
# --------------------------------------------------------------------------------------------
def test_identity_pointwise_downsample_sum_stack():
    x = torch.arange(16.0).view(4, 4)
    assert torch.equal(Identity()({"x": x}), x)
    assert torch.equal(Pointwise(torch.square, 2.0)({"x": x}), x**2)
    d = Downsample(2)({"x": x})
    assert d.shape == (2, 2) and d[0, 0] == pytest.approx(float(x[:2, :2].mean()))
    assert Downsample(2).output_shape((8, 6)) == (4, 3)
    assert torch.equal(Downsample(2, "decimate")({"x": x}), x[1::2, 1::2])
    s = Sum(Identity("a"), Pointwise(lambda v: 2 * v, 1.0, field="b"))
    assert torch.equal(s({"a": x, "b": x}), 3 * x) and s.required_fields() == ("a", "b")
    st = Stack(Identity(), Pointwise(torch.square))
    assert st({"x": x}).shape == (2, 4, 4) and st.output_shape((4, 4)) == (2, 4, 4)
    cat = Stack(Identity(), Downsample(2), mode="concat")
    assert cat({"x": x}).shape == (20,) and cat.output_shape((4, 4)) == (20,)
    with pytest.raises(ShapeError):
        Stack(Identity(), Downsample(2))({"x": x})


def test_sampling_mask_and_indices():
    m = torch.zeros(6, 6)
    m[::2, ::3] = 1
    op = Sampling(mask=m)
    x = torch.rand(6, 6)
    assert torch.equal(op({"x": x}), x * m)
    meas = op.measurement(op({"x": x}), noise_std=0.1)
    assert torch.equal(meas.mask, m) and meas.noise_std == 0.1
    assert op({"x": torch.rand(3, 3)}).shape == (6, 6)  # coarse fields are upsampled
    idx = Sampling(indices=torch.tensor([0, 7, 35]), native_shape=(6, 6))
    assert torch.equal(idx({"x": x}), x.reshape(-1)[[0, 7, 35]]) and idx.output_shape((3, 3)) == (
        3,
    )
    idx2 = Sampling(indices=torch.tensor([[1, 2], [5, 5]]), native_shape=(6, 6))
    assert torch.equal(idx2({"x": x}), torch.stack([x[1, 2], x[5, 5]]))
    with pytest.raises(OperatorError):
        Sampling()


def test_fourier_sampling_adjoint_parseval_and_noise_hook():
    torch.manual_seed(0)
    x = torch.rand(16, 16)
    full = FourierSampling(torch.ones(16, 16))
    y = full({"x": x})
    assert y.shape == (16, 16, 2)
    assert torch.allclose(full.adjoint(y), x, atol=1e-5)
    assert float((y**2).sum()) == pytest.approx(float((x**2).sum()), rel=1e-4)  # ortho = unitary
    m = random_kspace_mask((32, 32), fraction=0.3, seed=1)
    assert abs(float(m.mean()) - 0.3) < 0.02 and m[16, 16] == 1  # DC sampled
    op = FourierSampling(m)
    comp = FourierSampling(m, compact=True)
    xx = torch.rand(32, 32)
    assert comp({"x": xx}).shape == (int(m.sum()), 2)
    assert torch.allclose(comp.adjoint(comp({"x": xx})), op.adjoint(op({"x": xx})), atol=1e-5)
    assert op.measurement_mask().shape == (32, 32, 2)
    assert FourierSampling(m, imag_field="xi").required_fields() == ("x", "xi")
    # linear: homogeneity 1
    assert torch.allclose(op({"x": 3 * xx}), 3 * op({"x": xx}), atol=1e-5)
    # noise hook: pure noise in the outer k-space
    noise = 0.05 * torch.randn(32, 32, 2, generator=torch.Generator().manual_seed(2))
    meas = Measurement(
        op({"x": torch.zeros(32, 32)}) + noise * op.measurement_mask(), mask=op.measurement_mask()
    )
    assert op.estimate_noise(meas) == pytest.approx(0.05, rel=0.2)


# --------------------------------------------------------------------------------------------
# nonlinear operators
# --------------------------------------------------------------------------------------------
def test_phase_retrieval_shape_parseval_homogeneity():
    op = PhaseRetrieval(oversample=2)
    x = torch.rand(8, 8)
    y = op({"x": x})
    assert y.shape == (16, 16) and op.output_shape((8, 8)) == (16, 16) and torch.all(y >= 0)
    assert float(y.sum()) == pytest.approx(float((x**2).sum()), rel=1e-4)
    assert torch.allclose(op({"x": 2 * x}), 4 * y, rtol=1e-4, atol=1e-6)
    shifted = torch.roll(torch.nn.functional.pad(x, (0, 8, 0, 8)), (3, 2), (0, 1))[:8, :8]
    assert shifted.shape == (8, 8)  # (translation ambiguity is documented, not tested)


def test_beer_lambert_axis_cumulative_and_path_operator():
    dom = Domain.unit((32, 16))
    mu = torch.ones(32, 16)
    op = BeerLambert(axis=0, domain=dom)
    assert torch.allclose(op({"x": mu}), torch.full((16,), math.exp(-1.0)), atol=1e-6)
    assert op.output_shape((32, 16)) == (16,)
    cum = BeerLambert(axis=0, domain=dom, mode="cumulative", I0=2.0)({"x": mu})
    assert cum.shape == (32, 16) and float(cum[-1, 0]) == pytest.approx(
        2 * math.exp(-(1 - 0.5 / 32))
    )
    path = BeerLambert(path_op=Pointwise(lambda v: v.sum(-1)), I0=1.0)
    assert torch.allclose(path({"x": torch.full((4, 3), 0.1)}), torch.full((4,), math.exp(-0.3)))


def test_saturation_and_sequential_composition():
    sat = Saturation("tanh", level=2.0)
    v = sat({"x": torch.tensor([0.0, 1e-3, 100.0])})
    assert v[0] == 0 and v[1] == pytest.approx(1e-3, rel=1e-3) and v[2] == pytest.approx(2.0)
    sig = Saturation("sigmoid", level=1.0, gain=10.0, threshold=0.5)
    assert float(sig({"x": torch.tensor(0.5)})) == pytest.approx(0.5)
    cam = Sequential(Downsample(2), Saturation("tanh", 1.0))
    assert cam({"x": torch.rand(8, 8)}).shape == (4, 4)
    with pytest.raises(OperatorError):
        Saturation("relu")


# --------------------------------------------------------------------------------------------
# time stepping
# --------------------------------------------------------------------------------------------
def _wave_setup(n=32, mode="autograd", k=8, dtype=torch.float32):
    dom = Domain.unit((n,))
    h = dom.spacing()
    dt = stable_dt(h, c_max=1.0, cfl=0.5)
    op = TimeStepper(
        step=partial(wave_step, spacing=h, c=1.0, boundary="absorbing"),
        init=lambda f: wave_initial_state(f["p0"], c=1.0, dt=dt, spacing=h),
        n_steps=2 * n,
        dt=dt,
        observe=lambda s, i: s[1][[0, -1]],
        field="p0",
        grad_mode=mode,
        checkpoint_every=k,
        homogeneity=1.0,
    )
    t = dom.physical_coords(dtype=dtype)[..., 0]
    p0 = torch.exp(-0.5 * ((t - 0.35) / 0.06) ** 2) + 0.6 * torch.exp(
        -0.5 * ((t - 0.7) / 0.08) ** 2
    )
    return dom, op, p0


def test_time_stepper_checkpoint_matches_autograd_gradients():
    grads, outs = {}, {}
    w = torch.randn(65, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    for mode in ("autograd", "checkpoint"):
        _, op, p0 = _wave_setup(mode=mode, dtype=torch.float64)
        x = p0.clone().requires_grad_(True)
        y = op({"p0": x})
        (g,) = torch.autograd.grad((y * w).sum(), x)
        grads[mode], outs[mode] = g, y.detach()
    assert outs["autograd"].shape == (65, 2)
    assert torch.allclose(outs["autograd"], outs["checkpoint"], atol=1e-12)
    err = float((grads["autograd"] - grads["checkpoint"]).abs().max())
    assert err < 1e-5 * float(grads["autograd"].abs().max())
    # d'Alembert: the left sensor records ≈ p0(ct)/2 (absorbing boundary)
    assert float(outs["autograd"][:, 0].max()) == pytest.approx(0.5, abs=0.03)


def test_advection_diffusion_conserves_mass_and_stable_dt():
    n = 64
    h = (1.0 / n,)
    dt = stable_dt(h, velocity_max=1.0, diffusivity_max=1e-3, cfl=0.9)
    assert dt == pytest.approx(0.9 / (n + 2e-3 * n**2))
    u0 = torch.exp(-(((torch.arange(n) - 20.0) / 4.0) ** 2))
    op = TimeStepper(
        partial(
            advection_diffusion_step,
            spacing=h,
            velocity=(1.0,),
            diffusivity=1e-3,
            boundary="periodic",
        ),
        init=lambda f: f["u0"],
        n_steps=40,
        dt=dt,
        field="u0",
    )
    u = op({"u0": u0})
    assert float(u.sum()) == pytest.approx(float(u0.sum()), rel=1e-5)  # periodic: conservative
    assert int(u.argmax()) > 20  # advected to the right
    D = torch.full((n,), 1e-3)
    op2 = TimeStepper(
        partial(advection_diffusion_step, spacing=h, diffusivity="D", boundary="neumann"),
        init=lambda f: f["u0"],
        n_steps=10,
        dt=dt,
        field="u0",
        fields=("u0", "D"),
    )
    assert float(op2({"u0": u0, "D": D}).sum()) == pytest.approx(float(u0.sum()), rel=1e-5)
    with pytest.raises(OperatorError):
        stable_dt(h)
    bad = TimeStepper(
        lambda s, p, dt, k: s, init=lambda f: f["x"], n_steps=3, dt=0.1, observe=lambda s, k: None
    )
    with pytest.raises(OperatorError, match="None for every step"):
        bad({"x": torch.zeros(3)})


# --------------------------------------------------------------------------------------------
# BYOP end-to-end cases (c), (d), (e)
# --------------------------------------------------------------------------------------------
def _phantom(n):
    u = (torch.arange(n) + 0.5) / n * 2 - 1
    X, Y = torch.meshgrid(u, u, indexing="ij")
    x = 1.0 * ((X / 0.8) ** 2 + (Y / 0.6) ** 2 < 1).float()
    x -= 0.6 * ((X / 0.7) ** 2 + (Y / 0.5) ** 2 < 1).float()
    x += 0.5 * (((X - 0.2) / 0.2) ** 2 + ((Y + 0.1) / 0.15) ** 2 < 1).float()
    return x


def test_byop_fourier_sampling_30pct_with_hash_grid_beats_zero_filled():
    n = 32
    gt = _phantom(n)
    op = FourierSampling(random_kspace_mask((n, n), fraction=0.3, seed=0))
    y0 = op({"x": gt})
    y = y0 + 0.01 * float(y0.abs().max()) * torch.randn(
        y0.shape, generator=torch.Generator().manual_seed(1)
    )
    y = y * op.measurement_mask()
    zero_filled = psnr(op.adjoint(y), gt)
    problem = nefi.from_forward(
        op,
        Measurement(y, mask=op.measurement_mask()),
        shape=(n, n),
        prior="nonnegative + piecewise_constant",
        representation="hash",
        budget=300,
    )
    assert "hash" in problem.meta["auto"]["representation"]
    assert len(problem.curriculum.stages) == 2  # FourierSampling supports coarse grids
    res = nefi.invert(problem, device="cpu", seed=0)
    p = psnr(res.fields["x"], gt)
    assert p > zero_filled + 4.0, (p, zero_filled)


def test_byop_phase_retrieval_runs_and_decreases_loss():
    gt = torch.zeros(12, 12)
    gt[3:8, 4:9] = 1.0
    gt[5:7, 5:7] = 0.4
    op = PhaseRetrieval(oversample=2)
    problem = nefi.from_forward(
        op, op({"x": gt}), shape=(12, 12), prior="nonnegative", budget=120, noise=0.0
    )
    res = nefi.invert(problem, device="cpu", seed=0)
    assert res.history["total"][-1] < 0.1 * res.history["total"][0]
    assert res.pred.shape == (24, 24)


def test_byop_time_stepper_wave_recovers_initial_condition_with_checkpointing():
    dom, op, p0 = _wave_setup(n=32, mode="checkpoint", k=16)
    with torch.no_grad():
        y0 = op({"p0": p0})
    y = y0 + 0.005 * torch.randn(y0.shape, generator=torch.Generator().manual_seed(0))
    problem = nefi.from_forward(
        op,
        y,
        shape=(32,),
        prior="nonnegative + smooth",
        field_name="p0",
        representation="hash",
        budget=150,
    )
    assert problem.meta["auto"]["multiscale"].startswith("TimeStepper")  # single stage
    res = nefi.invert(problem, device="cpu", seed=0)
    assert psnr(res.fields["p0"], p0) > 30.0


def test_time_stepper_native_shape_params_and_complex_fourier_image():
    h = (1.0 / 32,)
    op = TimeStepper(
        partial(advection_diffusion_step, spacing=h, diffusivity="D", boundary="periodic"),
        init=lambda f: f["u0"],
        n_steps=5,
        dt=stable_dt(h, diffusivity_max=1e-3),
        params=lambda f: {"D": 1e-3 * torch.ones_like(f["u0"])},
        field="u0",
        native_shape=(32,),
    )
    assert op({"u0": torch.rand(16)}).shape == (32,)  # coarse field upsampled to the native grid
    re_, im_ = torch.rand(8, 8), torch.rand(8, 8)
    op_c = FourierSampling(torch.ones(8, 8), imag_field="xi")
    k = torch.view_as_complex(op_c({"x": re_, "xi": im_}).contiguous())
    assert torch.allclose(op_c.ifft(k), torch.complex(re_, im_), atol=1e-5)
