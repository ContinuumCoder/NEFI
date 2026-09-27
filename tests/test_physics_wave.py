"""Validation of the wave-equation physics (nefi.physics.wave / nefi.physics.timestep)."""

import math

import pytest
import torch

from nefi.domain import Domain
from nefi.physics.timestep import (
    auto_checkpoint_every,
    run_timestepping,
    stable_dt_diffusion,
    stable_dt_wave,
)
from nefi.physics.wave import (
    WaveInitialConditionOperator,
    WaveOperator,
    laplacian,
    ricker,
    time_reversal,
    wave_energy,
)

F64 = torch.float64


def _peak_time(y: torch.Tensor, dt: float) -> float:
    """Sub-sample peak time by parabolic interpolation."""
    i = int(torch.argmax(y))
    a, b, c = (float(v) for v in y[i - 1 : i + 2])
    return (i + 0.5 * (a - c) / (a - 2 * b + c)) * dt


# ---------------------------------------------------------------------------------------------
# generic time stepping
# ---------------------------------------------------------------------------------------------
def test_run_timestepping_record_and_checkpoint_gradients():
    k = torch.tensor([0.5, 1.0, 2.0], dtype=F64, requires_grad=True)

    def step(s, p, n, dt):
        return (s[0] - dt * p[0] * s[0],)  # u' = -k u, explicit Euler

    def observe(s, n):
        return s[0].clone()

    u0 = torch.ones(3, dtype=F64)
    (uT,), obs = run_timestepping(step, u0, (k,), 100, 0.01, observe, record=[0, 50, 100])
    assert len(obs) == 3 and torch.allclose(obs[0], u0)
    exact = (1 - 0.01 * k.detach()) ** 100
    assert torch.allclose(uT.detach(), exact, rtol=1e-12)
    g_plain = torch.autograd.grad(sum(o.sum() for o in obs), k)[0]
    _, obs_c = run_timestepping(
        step, u0, (k,), 100, 0.01, observe, record=[0, 50, 100], checkpoint_every=7
    )
    g_ckpt = torch.autograd.grad(sum(o.sum() for o in obs_c), k)[0]
    assert torch.allclose(g_plain, g_ckpt, rtol=1e-12, atol=0)
    assert auto_checkpoint_every(100) == 10


def test_cfl_helpers_and_leapfrog_stability_limit():
    h = (0.1, 0.1)
    assert math.isclose(stable_dt_wave(1.0, h, order=2, courant=1.0), 0.1 / math.sqrt(2))
    assert math.isclose(stable_dt_wave(1.0, h, order=4, courant=1.0), 0.1 * math.sqrt(3 / 8))
    assert math.isclose(stable_dt_diffusion(1.0, h, safety=1.0), 0.1**2 / 4)
    # leapfrog is bounded just below the limit and blows up just above it
    torch.manual_seed(0)
    p0 = torch.randn(40, 40, dtype=F64)
    for factor, bounded in ((0.99, True), (1.05, False)):
        dt = factor * stable_dt_wave(1.0, h, order=4, courant=1.0)
        prev, cur = p0.clone(), p0.clone()
        for _ in range(400):
            prev, cur = cur, 2 * cur - prev + dt**2 * laplacian(cur, h, order=4)
        assert (float(cur.abs().max()) < 1e3) == bounded


# ---------------------------------------------------------------------------------------------
# physics validation
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("order", [2, 4])
def test_pulse_travels_at_speed_c_1d(order):
    dom = Domain((400,), ((0.0, 4.0),))
    c0 = 2.0
    op = WaveOperator(
        dom,
        [[0.5]],
        [[1.5], [3.0]],
        n_t=400,
        dt_obs=0.004,
        f0=5.0,
        c_max=c0,
        order=order,
        absorbing="pml",
        absorb_width=0.5,
    )
    tr = op({"c": torch.full((400,), c0, dtype=F64)})[0]
    speed = 1.5 / (_peak_time(tr[1], op.dt_obs) - _peak_time(tr[0], op.dt_obs))
    assert abs(speed - c0) / c0 < 0.03, speed


def test_energy_conserved_without_absorption_and_decays_with_it():
    dom = Domain((40, 40), ((0.0, 1.0), (0.0, 1.0)))
    c = torch.full((40, 40), 1.0, dtype=F64)
    c[20:, :] = 1.5  # heterogeneous medium: the weighted discrete energy is still conserved
    kw = dict(n_t=150, dt_obs=0.01, f0=8.0, c_max=1.5, order=4, grad_mode="autograd")
    rec = list(range(80, 1 + 149 * 2, 20))  # dt = 0.005: t >= 0.4 s, after the Ricker (t0 = 0.15)

    closed = WaveOperator(dom, [[0.5, 0.5]], [[0.2, 0.2]], absorbing="none", **kw)

    def energy(op, crop):
        cc = crop(op.grid.pad(c, "replicate"))

        def obs(state, n):
            return wave_energy(crop(state[0]), crop(state[1]), cc, op.grid.spacing, op.dt, 4)

        return torch.stack(op.simulate(c, observe=obs, record=rec)[1]).squeeze(-1)

    e_closed = energy(closed, lambda x: x)
    assert float((e_closed.max() - e_closed.min()) / e_closed.mean()) < 1e-9
    for absorbing in ("pml", "sponge"):
        op = WaveOperator(
            dom, [[0.5, 0.5]], [[0.2, 0.2]], absorbing=absorbing, absorb_width=0.3, **kw
        )
        e = energy(op, op.grid.crop)  # energy inside the physical domain
        assert float(e[-1] / e.max()) < 0.05, (absorbing, e[-1] / e.max())


@pytest.mark.parametrize(
    "absorbing,width,tol", [("pml", 0.2, 0.01), ("sponge", 0.6, 0.05), ("cerjan", 0.6, 0.05)]
)
def test_absorbing_boundary_reflection_small(absorbing, width, tol):
    """Reflected amplitude vs a reference run in a 9x larger domain (no boundary in reach)."""
    c0, f0 = 2.0, 5.0  # λ = 0.4
    small = Domain((60, 60), ((0.0, 1.2), (0.0, 1.2)))
    big = Domain((180, 180), ((-1.2, 2.4), (-1.2, 2.4)))
    kw = dict(n_t=220, dt_obs=0.004, f0=f0, c_max=c0, grad_mode="autograd")
    src, rec = [[0.6, 0.8]], [[0.6, 1.0]]  # receiver 0.2 from the boundary: echo after 0.2 s
    ref = WaveOperator(big, src, rec, absorbing="none", **kw)
    ref_tr = ref({"c": torch.full(big.shape, c0, dtype=F64)})[0, 0]
    op = WaveOperator(small, src, rec, absorbing=absorbing, absorb_width=width, **kw)
    tr = op({"c": torch.full(small.shape, c0, dtype=F64)})[0, 0]
    refl = float((tr - ref_tr).abs().max() / ref_tr.abs().max())
    assert refl < tol, refl


def test_checkpoint_gradients_equal_autograd_and_finite_differences():
    torch.manual_seed(0)
    dom = Domain((16, 16), ((0.0, 1.0), (0.0, 1.0)))
    c = (2.0 + 0.2 * torch.randn(16, 16, dtype=F64)).clamp(1.6, 2.4)
    kw = dict(n_t=40, dt_obs=0.01, f0=5.0, c_max=2.5, absorb_width=0.2)
    src, rec = [[0.2, 0.3], [0.2, 0.7]], [[0.8, y] for y in (0.2, 0.4, 0.6, 0.8)]
    w = torch.randn(2, 4, 40, dtype=F64)
    grads = []
    for mode, ck in (("autograd", None), ("checkpoint", 5), ("checkpoint", None)):
        op = WaveOperator(dom, src, rec, grad_mode=mode, checkpoint_every=ck, **kw)
        x = c.clone().requires_grad_(True)
        (g,) = torch.autograd.grad((op({"c": x}) * w).sum(), x)
        grads.append(g)
    for g in grads[1:]:
        assert float((g - grads[0]).norm() / grads[0].norm()) < 1e-6
    op = WaveOperator(dom, src, rec, grad_mode="autograd", **kw)
    e = torch.randn(16, 16, dtype=F64)
    eps = 1e-6
    fd = float((op({"c": c + eps * e}) * w).sum()) - float((op({"c": c - eps * e}) * w).sum())
    fd /= 2 * eps
    assert abs(fd - float((grads[0] * e).sum())) < 1e-6 * max(1.0, abs(fd))


def test_wave_operator_multiresolution_and_source_batching():
    dom = Domain((32, 32), ((0.0, 1.0), (0.0, 1.0)))
    src = [[0.1, 0.3], [0.1, 0.5], [0.1, 0.7]]
    rec = [[0.9, y] for y in torch.linspace(0.1, 0.9, 6).tolist()]
    op = WaveOperator(dom, src, rec, n_t=60, dt_obs=0.016, f0=3.0, c_max=2.5, absorb_width=0.25)
    x = dom.physical_coords(dtype=F64)
    c = 2.0 + 0.2 * torch.exp(-((x - 0.5) ** 2).sum(-1) / 0.02)
    y = op({"c": c})
    assert y.shape == (3, 6, 60) == op.output_shape((32, 32))
    coarse = op.at_resolution((16, 16))
    assert coarse is op.at_resolution((16, 16)) and coarse.grid.shape == (16, 16)
    assert coarse.fidelity_tag == op.fidelity_tag
    xc = dom.at((16, 16)).physical_coords(dtype=F64)
    yc = coarse({"c": 2.0 + 0.2 * torch.exp(-((xc - 0.5) ** 2).sum(-1) / 0.02)})
    assert yc.shape == y.shape
    assert float((yc - y).norm() / y.norm()) < 0.2  # same physics on a 2x coarser grid
    batched = WaveOperator(
        dom, src, rec, n_t=60, dt_obs=0.016, f0=3.0, c_max=2.5, absorb_width=0.25, source_batch=2
    )
    assert torch.allclose(batched({"c": c}), y, atol=1e-12)
    with pytest.raises(Exception, match="exceeds"):
        op({"c": torch.full((32, 32), 3.0, dtype=F64)})


def test_initial_condition_operator_is_linear_and_time_reversal_refocuses():
    torch.manual_seed(0)
    n = 48
    dom = Domain((n, n), ((-1.0, 1.0), (-1.0, 1.0)))
    th = 2 * math.pi * torch.arange(64, dtype=F64) / 64
    rec = torch.stack([0.9 * torch.cos(th), 0.9 * torch.sin(th)], 1)
    op = WaveInitialConditionOperator(dom, rec, n_t=200, dt_obs=0.01, c=1.0, absorb_width=0.25)
    assert op.homogeneity == 1.0
    x = dom.physical_coords(dtype=F64)
    src = torch.tensor([0.3, -0.2], dtype=F64)
    p0 = torch.exp(-((x - src) ** 2).sum(-1) / (2 * 0.05**2))
    q0 = torch.randn(n, n, dtype=F64)
    y1, y2 = op({"p0": p0}), op({"p0": q0})
    y3 = op({"p0": 2 * p0 - 3 * q0})
    assert y1.shape == (64, 200)
    assert float((y3 - (2 * y1 - 3 * y2)).norm() / y3.norm()) < 1e-10
    for mode in ("dirichlet", "adjoint"):
        img = time_reversal(y1, dom, rec, dt_obs=0.01, c=1.0, mode=mode)
        i = int(torch.argmax(img))
        peak = x.reshape(-1, 2)[i]
        assert float((peak - src).norm()) < 3 * dom.spacing()[0], (mode, peak)
        corr = torch.corrcoef(torch.stack([img.flatten(), p0.flatten()]))[0, 1]
        assert float(corr) > 0.5, (mode, corr)


def test_ricker_wavelet_properties():
    t = torch.linspace(0, 1, 2001, dtype=F64)
    w = ricker(t, 5.0)
    assert abs(float(w.max()) - 1.0) < 1e-6 and abs(float(t[w.argmax()]) - 1.2 / 5.0) < 1e-3
    assert abs(float(w[0])) < 1e-4 and abs(float(w.sum() * (t[1] - t[0]))) < 1e-4  # zero mean
