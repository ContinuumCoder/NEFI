"""Tests of the differentiable heat solver, its linear solvers and its discrete adjoint (NeFTY)."""

import math

import numpy as np
import pytest
import torch
from torch import nn

from nefi.domain import Domain
from nefi.errors import ConfigError, ShapeError
from nefi.measurement import Measurement
from nefi.operators.pde import (
    BoundarySpec,
    DiffusionStencil,
    ExplicitHeatSimulator,
    GaussianFlash,
    HeatOperator,
    HeatSolveConfig,
    UniformFlash,
    apply_A,
    apply_diffusion,
    conjugate_gradient,
    diagonal_A,
    explicit_substeps,
    face_coefficients,
    implicit_euler_frames,
    jacobi,
    rollout,
)

F64 = torch.float64
PAPER_EXTENT = ((0.0, 10.0), (0.0, 10.0), (0.0, 1.0))  # NeFTY Tab. 5 slab
FACES = ("harmonic", "arithmetic")


@pytest.fixture(autouse=True, scope="module")
def _single_thread():
    # tiny stencil ops are overhead-bound; intra-op threading only slows them down on CPU
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _defect_alpha(shape, dtype=torch.float32):
    a = torch.full(shape, 0.15, dtype=dtype)
    nx, ny, nz = shape
    a[nx // 3 : nx // 2, ny // 3 : ny // 2, nz // 3 : nz // 2] = 0.01
    a[nx // 2 : 2 * nx // 3, ny // 5 : ny // 3, 1 : max(2, nz // 4)] = 0.25
    return a


def _surface_moments(frames: torch.Tensor, dom: Domain) -> np.ndarray:
    """Temperature-weighted lateral variances of every surface frame (NeFTY Eq. 32)."""
    x, y = dom.axis_coords(normalized=False, dtype=F64)[:2]
    out = []
    for f in frames.double():
        m0 = f.sum()
        px, py = f.sum(1) / m0, f.sum(0) / m0
        xb, yb = (px * x).sum(), (py * y).sum()
        out.append((float((px * (x - xb) ** 2).sum()), float((py * (y - yb) ** 2).sum())))
    return np.array(out)


# ---------------------------------------------------------------------------------------------
# 1. Gaussian variance growth (NeFTY App. I)
# ---------------------------------------------------------------------------------------------
def test_gaussian_variance_growth():
    """σ²(t) = σ0² + 2αt (Eq. 30–31): surface second moments grow with slope 2α within 1 %."""
    alpha0 = 0.1
    dom = Domain((64, 64, 8), ((0.0, 16.0), (0.0, 16.0), (0.0, 1.0)))
    ic = GaussianFlash(amplitude=1.0, width_xy=1.0, width_z=0.3, periodic=False)
    op = HeatOperator(dom, dt=0.05, n_steps=100, initial=ic, inner_iters=50).double()
    alpha = torch.full(dom.shape, alpha0, dtype=F64)
    frames = op({"alpha": alpha})
    mom = _surface_moments(frames, dom)
    t = op.frame_times.numpy()
    late = t > 1.0  # after the initial through-thickness transient (App. I.3)
    slopes = [np.polyfit(t[late], mom[late, i], 1)[0] for i in range(2)]
    rel = abs(np.mean(slopes) - 2 * alpha0) / (2 * alpha0)
    assert rel < 0.01, (slopes, rel)
    assert abs(slopes[0] - slopes[1]) < 1e-10  # isotropy
    # the volume-integrated variance of implicit Euler grows by exactly 2αΔt per step
    states = op.simulate(alpha, return_states=True)["states"]
    vol = _surface_moments(states.sum(-1), dom)
    vslope = np.polyfit(t, vol[:, 0], 1)[0]
    assert abs(vslope - 2 * alpha0) / (2 * alpha0) < 1e-4


def test_gaussian_variance_growth_2d_uniform_depth_is_exact():
    """A z-uniform Gaussian in a 2-D (x, z) slab: σ_x² grows by exactly 2αΔt per step."""
    alpha0 = 0.2
    dom = Domain((256, 4), ((0.0, 40.0), (0.0, 1.0)))
    x = dom.physical_coords(dtype=F64)[..., 0]
    T0 = torch.exp(-((x - 20.0) ** 2) / 2.0)
    op = HeatOperator(dom, dt=0.1, n_steps=50, initial=T0, solver="cg", cg_tol=1e-12).double()
    fr = op({"alpha": torch.full(dom.shape, alpha0, dtype=F64)})
    xs = dom.axis_coords(normalized=False, dtype=F64)[0]
    var = []
    for f in fr:
        p = f / f.sum()
        xb = (p * xs).sum()
        var.append(float((p * (xs - xb) ** 2).sum()))
    inc = np.diff(var)
    assert np.allclose(inc, 2 * alpha0 * 0.1, rtol=1e-6), inc[:5]


# ---------------------------------------------------------------------------------------------
# 2. discrete adjoint correctness (NeFTY Eq. 10–11 / 25–27)
# ---------------------------------------------------------------------------------------------
TINY = Domain((6, 6, 4), ((0.0, 2.0), (0.0, 2.0), (0.0, 0.5)))


def _tiny_op(grad_mode, solver="cg", bc=None, robin_h=0.0, **kw):
    return HeatOperator(
        TINY,
        dt=0.05,
        n_steps=5,
        initial=GaussianFlash(1.0, 0.6, 0.15),
        solver=solver,
        cg_tol=1e-14,
        cg_max_iter=500,
        grad_mode=grad_mode,
        bc=bc,
        robin_h=robin_h,
        **kw,
    ).double()


def _tiny_grad(op, alpha0, w):
    a = alpha0.clone().requires_grad_(True)
    fr = op({"alpha": a})
    loss = (w * fr).sum() + 0.5 * (fr**2).sum()
    (g,) = torch.autograd.grad(loss, a)
    return fr.detach(), g


@pytest.mark.parametrize(
    "bc,h", [(None, 0.0), (("periodic", "periodic", "robin"), 0.8), ("neumann", 0.0)]
)
def test_adjoint_matches_autograd(bc, h):
    g = torch.Generator().manual_seed(0)
    alpha0 = 0.05 + 0.2 * torch.rand(TINY.shape, generator=g, dtype=F64)
    w = torch.randn(5, 6, 6, generator=g, dtype=F64)
    f_ref, g_ref = _tiny_grad(_tiny_op("autograd", bc=bc, robin_h=h), alpha0, w)
    for mode, assembly in [("adjoint", "fused"), ("adjoint", "per_step"), ("checkpoint", "fused")]:
        f, gr = _tiny_grad(_tiny_op(mode, bc=bc, robin_h=h, adjoint_assembly=assembly), alpha0, w)
        assert torch.equal(f, f_ref)
        rel = float((gr - g_ref).abs().max() / g_ref.abs().max())
        assert rel < 1e-6, (mode, assembly, rel)


def test_adjoint_matches_autograd_with_jacobi():
    g = torch.Generator().manual_seed(1)
    alpha0 = 0.05 + 0.2 * torch.rand(TINY.shape, generator=g, dtype=F64)
    w = torch.randn(5, 6, 6, generator=g, dtype=F64)
    _, g_ref = _tiny_grad(_tiny_op("autograd", solver="jacobi", inner_iters=200), alpha0, w)
    _, g_adj = _tiny_grad(_tiny_op("adjoint", solver="jacobi", inner_iters=200), alpha0, w)
    assert float((g_adj - g_ref).abs().max() / g_ref.abs().max()) < 1e-6


def test_adjoint_gradcheck_alpha_and_T0():
    op = _tiny_op("adjoint")
    g = torch.Generator().manual_seed(2)
    a = (0.05 + 0.2 * torch.rand(TINY.shape, generator=g, dtype=F64)).requires_grad_(True)
    T0 = op.initial_state().double().clone().requires_grad_(True)
    w = torch.randn(5, 6, 6, generator=g, dtype=F64)

    def loss(a_, T0_):
        fr = implicit_euler_frames(a_, T0_, op.solve_config, "adjoint")
        return (w * fr).sum() + 0.5 * (fr**2).sum()

    assert torch.autograd.gradcheck(loss, (a, T0), eps=1e-6, atol=1e-7, rtol=1e-5)
    # dJ/dT0 = μ¹ (App. D.3) agrees with the unrolled reference
    for mode in ("adjoint", "autograd"):
        T0_ = T0.detach().clone().requires_grad_(True)
        fr = implicit_euler_frames(a.detach(), T0_, op.solve_config, mode)
        ((w * fr).sum()).backward()
        if mode == "adjoint":
            g_adj = T0_.grad
    assert torch.allclose(g_adj, T0_.grad, rtol=1e-9, atol=1e-12)


def test_adjoint_gradcheck_full_jacobian_with_initial_frame():
    dom = Domain((4, 3, 3), ((0.0, 1.0), (0.0, 1.0), (0.0, 0.5)))
    cfg = HeatSolveConfig(
        spacing=dom.spacing(),
        dt=0.05,
        n_steps=3,
        obs_steps=(0, 1, 3),
        bc=BoundarySpec.parse(None, 3),
        solver="cg",
        cg_tol=1e-14,
        cg_max_iter=200,
    )
    g = torch.Generator().manual_seed(3)
    a = (0.1 + 0.1 * torch.rand(dom.shape, generator=g, dtype=F64)).requires_grad_(True)
    T0 = torch.rand(dom.shape, generator=g, dtype=F64).requires_grad_(True)
    fn = lambda a_, T0_: implicit_euler_frames(a_, T0_, cfg, "adjoint")  # noqa: E731
    assert torch.autograd.gradcheck(fn, (a, T0), eps=1e-6, atol=1e-7, rtol=1e-5)


class _LearnableFlash(nn.Module):
    """Uniform flash with a learnable amplitude (exercises dJ/dT0 = μ¹ through a module)."""

    def __init__(self):
        super().__init__()
        self.log_amp = nn.Parameter(torch.tensor(0.3, dtype=F64))

    def forward(self, domain):
        return torch.exp(self.log_amp) * UniformFlash(1.0, 0.1)(domain).to(F64)


def test_learnable_initial_condition_receives_adjoint_gradient():
    grads = {}
    for mode in ("adjoint", "autograd"):
        ic = _LearnableFlash()
        op = HeatOperator(TINY, dt=0.05, n_steps=4, initial=ic, grad_mode=mode, solver="cg")
        op = op.double()
        a = torch.full(TINY.shape, 0.1, dtype=F64, requires_grad=True)
        (op({"alpha": a}) ** 2).sum().backward()
        grads[mode] = (float(ic.log_amp.grad), a.grad.clone())
        assert op.at_resolution((3, 3, 2)).initial is ic  # shared across resolutions
    assert math.isclose(grads["adjoint"][0], grads["autograd"][0], rel_tol=1e-5)
    assert torch.allclose(grads["adjoint"][1], grads["autograd"][1], rtol=1e-4, atol=1e-10)


def test_adjoint_memory_is_trajectory_sized():
    """Saved tensors: adjoint ≈ N_g·(N_t+1); unrolled autograd ≫ K·N_g·N_t (NeFTY App. D.3)."""
    dom = Domain((8, 8, 6), ((0.0, 2.0), (0.0, 2.0), (0.0, 1.0)))
    n_t, k = 10, 50
    saved = {}
    for mode in ("adjoint", "autograd", "checkpoint"):
        op = HeatOperator(dom, dt=0.05, n_steps=n_t, grad_mode=mode, inner_iters=k)
        a = torch.full(dom.shape, 0.15, requires_grad=True)
        nbytes = [0]

        def pack(t, nbytes=nbytes):
            nbytes[0] += t.numel() * t.element_size()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            fr = op({"alpha": a})
        fr.sum().backward()
        assert a.grad is not None and torch.isfinite(a.grad).all()
        saved[mode] = nbytes[0] / (dom.numel * 4)  # in units of one float32 grid
    assert saved["adjoint"] <= n_t + 1 + 1e-9
    assert saved["autograd"] > k * n_t
    assert saved["adjoint"] < saved["checkpoint"] < saved["autograd"]


# ---------------------------------------------------------------------------------------------
# 3. harmonic vs arithmetic face coefficients (Prop. 1, App. A.4)
# ---------------------------------------------------------------------------------------------
def test_face_coefficient_values():
    a = torch.tensor([0.2, 0.01, 0.2], dtype=F64)
    h = face_coefficients(a, 0, "harmonic", periodic=False)
    m = face_coefficients(a, 0, "arithmetic", periodic=False)
    assert torch.allclose(h[:2], torch.full((2,), 2 * 0.2 * 0.01 / 0.21, dtype=F64))
    assert torch.allclose(m[:2], torch.full((2,), 0.105, dtype=F64))
    assert h[-1] == 0 and m[-1] == 0  # adiabatic boundary face
    hp = face_coefficients(a, 0, "harmonic", periodic=True)
    assert math.isclose(float(hp[-1]), 0.2)  # wrap face between the two bulk cells
    with pytest.raises(ConfigError):
        face_coefficients(a, 0, "geometric")


def test_harmonic_mean_throttles_flux_through_insulating_layer():
    """An α ratio 1:20 layer transmits less heat with the harmonic mean (Prop. 1 / App. A.4)."""
    dom = Domain((4, 4, 20), ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
    alpha = torch.full(dom.shape, 0.2, dtype=F64)
    alpha[..., 10] = 0.01  # insulating cell layer (1:20)
    T = torch.zeros(dom.shape, dtype=F64)
    T[..., :10] = 1.0  # heat in front of the layer
    sp = dom.spacing()
    # instantaneous heat rate into the insulating cells, ∝ the face coefficient ᾱ_{9+1/2}
    rate = {m: float(apply_diffusion(T, alpha, sp, None, m)[..., 10].sum()) for m in FACES}
    assert 0 < rate["harmonic"] < rate["arithmetic"]
    assert rate["arithmetic"] / rate["harmonic"] == pytest.approx(0.105 / (0.004 / 0.21))
    behind = {}
    for mode in FACES:
        op = HeatOperator(dom, dt=0.01, n_steps=30, initial=T, face_mode=mode, solver="cg")
        states = op.double().simulate(alpha, return_states=True)["states"]
        behind[mode] = float(states[-1][..., 11:].sum())
    assert behind["harmonic"] < 0.5 * behind["arithmetic"], behind
    # uniform α: both means coincide exactly
    u = torch.full(dom.shape, 0.13, dtype=F64)
    assert torch.allclose(
        apply_diffusion(T, u, sp, None, "harmonic"), apply_diffusion(T, u, sp, None, "arithmetic")
    )


# ---------------------------------------------------------------------------------------------
# 4. linear solvers and the system matrix
# ---------------------------------------------------------------------------------------------
BCS = [
    (None, 0.0),
    ("neumann", 0.0),
    (("periodic", "periodic", "robin"), 0.7),
    (("neumann", "periodic", "robin"), 2.0),
]


@pytest.mark.parametrize("bc,h", BCS)
def test_system_is_symmetric_positive_definite(bc, h):
    g = torch.Generator().manual_seed(0)
    shape = (5, 4, 3)
    alpha = 0.01 + 0.24 * torch.rand(shape, generator=g, dtype=F64)
    sp, dt = (0.3, 0.2, 0.1), 0.05
    x, y = torch.randn(shape, generator=g, dtype=F64), torch.randn(shape, generator=g, dtype=F64)
    Ax, Ay = apply_A(x, alpha, dt, sp, bc, robin_h=h), apply_A(y, alpha, dt, sp, bc, robin_h=h)
    assert abs(float((Ax * y).sum() - (x * Ay).sum())) < 1e-12
    n = alpha.numel()
    eye = torch.eye(n, dtype=F64).reshape(n, *shape)
    M = torch.stack([apply_A(e, alpha, dt, sp, bc, robin_h=h).reshape(-1) for e in eye])
    assert torch.equal(M, M.T)
    assert float(torch.linalg.eigvalsh(M).min()) >= 1.0 - 1e-12  # I − Δt L with L ≤ 0
    D = diagonal_A(alpha, dt, sp, bc, robin_h=h).reshape(-1)
    assert torch.allclose(torch.diagonal(M), D)
    offsum = M.abs().sum(1) - torch.diagonal(M).abs()
    assert bool((torch.diagonal(M) - offsum >= 1.0 - 1e-12).all())  # App. D.2 dominance
    st = DiffusionStencil(alpha, sp, bc, robin_h=h)
    assert abs(float((y * st.apply(x)).sum() - st.bilinear(y, x))) < 1e-10


def test_jacobi_and_cg_agree_in_paper_regime():
    """Paper grid spacing (Δx = 0.156, Δz = 0.0625) and Δt = 0.05 with a 1:20 defect."""
    dom = Domain((64, 64, 16), PAPER_EXTENT)
    alpha = _defect_alpha(dom.shape, F64)
    sys_ = DiffusionStencil(alpha, dom.spacing()).system(0.05)
    b = GaussianFlash(100.0, 2.5, 0.1)(dom)
    x_cg, r_cg = conjugate_gradient(sys_.apply, b, b, tol=1e-12, max_iter=500, precond=sys_.diag)
    x_j, r_j = jacobi(sys_.apply, sys_.diag, b, b, iters=50, apply_offdiag=sys_.apply_offdiag)
    assert float(r_cg) / float(b.norm()) < 1e-11
    # K = 50 on the sharpest right-hand side (the first post-flash step): ≈ 3e-6 relative
    assert float(r_j) / float(b.norm()) < 1e-5
    assert float((x_j - x_cg).abs().max() / x_cg.abs().max()) < 1e-5
    # the fused sweep of the operator is the same iteration as Eq. (24)
    x_f, bD = b, b * sys_.inv_diag
    for _ in range(50):
        x_f = sys_.jacobi_sweep(bD, x_f)
    assert torch.allclose(x_f, x_j, rtol=1e-12, atol=1e-10)
    # damped Jacobi converges to the same solution
    x_d, _ = jacobi(sys_.apply, sys_.diag, b, b, iters=400, omega=0.8)
    assert float((x_d - x_cg).abs().max() / x_cg.abs().max()) < 1e-8


def test_jacobi_and_cg_rollouts_agree():
    dom = Domain((32, 32, 16), PAPER_EXTENT)
    alpha = _defect_alpha(dom.shape)
    ops = {
        s: HeatOperator(dom, dt=0.05, n_steps=30, solver=s, inner_iters=50, cg_tol=1e-6)
        for s in ("jacobi", "cg")
    }
    fj, fc = (ops[s]({"alpha": alpha}) for s in ("jacobi", "cg"))
    assert float((fj - fc).abs().max() / fc.abs().max()) < 1e-4
    res = ops["jacobi"].simulate(alpha, residuals=True)["residuals"]
    assert max(res) < 1e-4  # K = 50 reaches the camera noise floor regime (App. D.2)


def test_cg_residual_decreases_monotonically():
    dom = Domain((32, 32, 16), PAPER_EXTENT)
    alpha = _defect_alpha(dom.shape, F64)
    sys_ = DiffusionStencil(alpha, dom.spacing()).system(0.05)
    b = GaussianFlash(100.0, 2.5, 0.1)(dom)
    for precond in (None, sys_.diag):
        hist: list[float] = []
        x, r = conjugate_gradient(
            sys_.apply, b, None, tol=1e-12, max_iter=300, precond=precond, history=hist
        )
        assert np.all(np.diff(hist) < 0), hist
        assert hist[-1] <= 1e-12 * float(b.norm()) and float(r) == pytest.approx(hist[-1])
        assert float((sys_.apply(x) - b).norm()) < 1e-9 * float(b.norm())
    # zero right-hand side is an exact fixed point (no NaNs from 0/0)
    x0, r0 = conjugate_gradient(sys_.apply, torch.zeros_like(b), None, tol=0.0, max_iter=3)
    assert torch.equal(x0, torch.zeros_like(b)) and float(r0) == 0.0


# ---------------------------------------------------------------------------------------------
# 5. boundary conditions
# ---------------------------------------------------------------------------------------------
def test_periodic_adiabatic_conserves_heat_and_wraps():
    dom = Domain((16, 12, 6), ((0.0, 4.0), (0.0, 3.0), (0.0, 1.0)))
    ic = GaussianFlash(10.0, 0.5, 0.2, center=(0.0, 0.0))  # centred on the periodic corner
    alpha = _defect_alpha(dom.shape, F64)
    for solver in ("cg", "jacobi"):
        op = HeatOperator(dom, dt=0.05, n_steps=40, initial=ic, solver=solver, cg_tol=1e-13)
        out = op.double().simulate(alpha, return_states=True)
        heat = out["states"].sum(dim=(1, 2, 3))
        drift = float((heat - out["T0"].sum()).abs().max() / out["T0"].sum())
        assert drift < (1e-11 if solver == "cg" else 1e-6), (solver, drift)
    # wrap-around: the corner pulse spreads symmetrically across x = 0 in a uniform medium
    u = torch.full(dom.shape, 0.15, dtype=F64)
    st = HeatOperator(dom, dt=0.05, n_steps=10, initial=ic, solver="cg", cg_tol=1e-13).double()
    s = st.simulate(u, return_states=True)["states"][-1]
    # the corner is a cell face: cell i mirrors cell N-1-i across the periodic boundary
    assert torch.allclose(s, s.flip(0), rtol=1e-9) and torch.allclose(s, s.flip(1), rtol=1e-9)
    assert float(s[-1].sum()) > 0.1 * float(s[0].sum())  # heat really crossed the boundary


def test_robin_back_face_loses_heat_monotonically():
    dom = Domain((8, 8, 6), ((0.0, 2.0), (0.0, 2.0), (0.0, 1.0)))
    alpha = _defect_alpha(dom.shape, F64)
    op = HeatOperator(
        dom,
        dt=0.05,
        n_steps=60,
        initial=UniformFlash(1.0, 0.2),
        bc=("periodic", "periodic", "robin"),
        robin_h=0.5,
        solver="cg",
        cg_tol=1e-13,
    ).double()
    out = op.simulate(alpha, return_states=True)
    heat = torch.cat([out["T0"].sum().view(1), out["states"].sum(dim=(1, 2, 3))])
    assert bool((torch.diff(heat) < 0).all())
    assert float(heat[-1]) < 0.9 * float(heat[0])
    # h = 0 recovers the adiabatic (conservative) case
    op0 = HeatOperator(
        dom,
        dt=0.05,
        n_steps=10,
        initial=UniformFlash(1.0, 0.2),
        bc="robin",
        robin_h=0.0,
        solver="cg",
        cg_tol=1e-13,
    ).double()
    h0 = op0.simulate(alpha, return_states=True)["states"].sum(dim=(1, 2, 3))
    assert float((h0 - h0[0]).abs().max()) < 1e-10 * float(h0[0])


def test_boundary_spec_validation():
    assert BoundarySpec.parse(None, 3).kinds == ("periodic", "periodic", "neumann")
    assert BoundarySpec.parse("robin", 3, 1.0).has_robin
    assert not BoundarySpec.parse("robin", 3, 0.0).has_robin
    with pytest.raises(ConfigError):
        BoundarySpec(("robin", "periodic", "neumann"))
    with pytest.raises(ConfigError):
        BoundarySpec.parse(("periodic", "neumann"), 3)
    with pytest.raises(ConfigError):
        BoundarySpec(("dirichlet", "neumann"))


# ---------------------------------------------------------------------------------------------
# 6. multiscale plumbing
# ---------------------------------------------------------------------------------------------
def test_at_resolution_output_shape_and_measurement_resampling():
    dom = Domain((16, 16, 8), PAPER_EXTENT)
    op = HeatOperator(dom, dt=0.05, n_steps=12, obs_frames=3, inner_iters=20)
    assert op.obs_steps == (3, 6, 9, 12) and op.output_shape((16, 16, 8)) == (4, 16, 16)
    coarse = op.at_resolution((8, 8, 4))
    assert coarse is not op and op.at_resolution((16, 16, 8)) is op
    assert coarse.domain.spacing() == pytest.approx((1.25, 1.25, 0.25))
    assert (coarse.dt, coarse.n_steps, coarse.obs_steps) == (op.dt, op.n_steps, op.obs_steps)
    assert coarse.initial_state().shape == (8, 8, 4)
    for o, shape in ((op, (16, 16, 8)), (coarse, (8, 8, 4))):
        pred = o({"alpha": torch.full(shape, 0.15)})
        assert tuple(pred.shape) == o.output_shape(shape)
    with pytest.raises(ShapeError):
        op({"alpha": torch.full((8, 8, 4), 0.15)})
    fine = op({"alpha": torch.full((16, 16, 8), 0.15)})
    meas = Measurement(fine)
    m8 = meas.resampled(op.output_shape((8, 8, 4)))
    assert m8.shape == (4, 8, 8)
    avg = fine.reshape(4, 8, 2, 8, 2).mean(dim=(2, 4))  # lateral 2×2 area average
    assert torch.allclose(m8.data, avg, atol=1e-5)
    # the coarse forward model is a consistent approximation of the fine one
    cpred = coarse({"alpha": torch.full((8, 8, 4), 0.15)})
    assert float((cpred - m8.data).abs().max() / m8.data.abs().max()) < 0.2


def test_two_dimensional_slab_and_obs_frame_selection():
    dom = Domain((32, 8), ((0.0, 10.0), (0.0, 1.0)))
    op = HeatOperator(dom, dt=0.05, n_steps=10, obs_frames=[0, 5, 10])
    a = torch.full(dom.shape, 0.15, requires_grad=True)
    fr = op({"alpha": a})
    assert fr.shape == (3, 32)
    assert torch.allclose(fr[0], op.initial_state()[..., 0].float())
    fr.pow(2).sum().backward()
    assert a.grad.shape == dom.shape and float(a.grad.abs().sum()) > 0
    with pytest.raises(ConfigError):
        HeatOperator(dom, n_steps=10, obs_frames=[11])
    with pytest.raises(ConfigError):
        HeatOperator(dom, solver="gmres")


# ---------------------------------------------------------------------------------------------
# explicit data simulator (inverse-crime guard, App. E.1)
# ---------------------------------------------------------------------------------------------
def test_explicit_substep_rule():
    # App. E.1: N_sub = max(10, ceil(dt / (Δx_min² / (2 D α_max)) × 2))
    sp = (10 / 64, 10 / 64, 1 / 16)
    dt_stable = (1 / 16) ** 2 / (2 * 3 * 0.2)
    assert explicit_substeps(0.05, sp, 0.2) == math.ceil(0.05 / dt_stable * 2.0)
    assert explicit_substeps(0.05, (1.0, 1.0, 1.0), 0.2) == 10
    # always inside the exact anisotropic forward-Euler stability limit
    for a_max in (0.01, 0.15, 0.25, 1.0):
        h = 0.05 / explicit_substeps(0.05, sp, a_max)
        assert h <= 1.0 / (2 * a_max * sum(1 / s**2 for s in sp))


def test_explicit_simulator_is_independent_and_consistent():
    dom = Domain((16, 16, 6), PAPER_EXTENT)
    alpha = _defect_alpha(dom.shape, F64)
    ic = GaussianFlash(100.0, 3.0, 0.2)
    sim = ExplicitHeatSimulator(dom, dt=0.1, n_steps=10, initial=ic)
    assert sim.fidelity_tag != HeatOperator(dom).fidelity_tag
    fe = sim({"alpha": alpha.float()})
    assert fe.dtype == F64 and fe.shape == (10, 16, 16)
    # the explicit field is conservative and converges to the implicit solution as Δt → 0
    errs = []
    for m in (1, 4, 16):
        op = HeatOperator(
            dom, dt=0.1, n_steps=10, initial=ic, solver="cg", cg_tol=1e-12, substeps=m
        ).double()
        errs.append(float((op({"alpha": alpha}) - fe).abs().max() / fe.abs().max()))
    assert errs[0] > errs[1] > errs[2] and errs[2] < 0.02, errs
    # same arithmetic-mean physics option
    sim_a = ExplicitHeatSimulator(dom, dt=0.1, n_steps=10, initial=ic, face_mode="arithmetic")
    assert float((sim_a({"alpha": alpha}) - fe).abs().max()) > 1e-3
    assert sim.at_resolution((8, 8, 3)).domain.shape == (8, 8, 3)


def test_rollout_config_validation():
    with pytest.raises(ConfigError):
        HeatSolveConfig((1.0, 1.0), 0.1, 5, (6,), BoundarySpec.parse(None, 2))
    with pytest.raises(ConfigError):
        HeatSolveConfig((1.0, 1.0), 0.1, 5, (2, 1), BoundarySpec.parse(None, 2))
    cfg = HeatSolveConfig((0.5, 0.25), 0.1, 5, (1, 5), BoundarySpec.parse(None, 2))
    fr, st, res = rollout(
        torch.full((4, 4), 0.1), torch.rand(4, 4), cfg, keep_states=True, residuals=True
    )
    assert fr.shape == (2, 4) and st.shape == (5, 4, 4) and len(res) == 5


@pytest.mark.slow
def test_compiled_jacobi_matches_eager():
    dom = Domain((12, 12, 6), PAPER_EXTENT)
    alpha = _defect_alpha(dom.shape).requires_grad_(True)
    out = {}
    for flag in (False, True):
        op = HeatOperator(dom, dt=0.05, n_steps=5, inner_iters=10, compile=flag)
        fr = op({"alpha": alpha})
        (g,) = torch.autograd.grad(fr.pow(2).sum(), alpha)
        out[flag] = (fr.detach(), g)
    assert torch.allclose(out[True][0], out[False][0], rtol=1e-5, atol=1e-5)
    assert torch.allclose(out[True][1], out[False][1], rtol=1e-4, atol=1e-6)


def test_initial_condition_specs_and_cell_averages():
    dom = Domain((8, 8, 4), PAPER_EXTENT)
    by_name = HeatOperator(dom, n_steps=2, initial="flash").initial_state()
    by_dict = HeatOperator(dom, n_steps=2, initial={"type": "uniform_flash", "width_z": 0.2})
    assert torch.allclose(by_name, by_dict.initial_state())
    assert torch.allclose(by_name, UniformFlash(100.0, 0.2)(dom))
    thin = HeatOperator(dom, n_steps=2, initial={"type": "flash", "width_z": 0.05}).initial_state()
    # same amplitude, 4× thinner absorption layer: 4× less deposited heat (∫ A e^{-z²/2w²} ∝ A w)
    assert float(thin.sum() / by_name.sum()) == pytest.approx(0.25, rel=1e-6)
    t = torch.rand(16, 16, 8, dtype=F64)
    from_tensor = HeatOperator(dom, n_steps=2, initial=t).initial_state()
    assert from_tensor.shape == (8, 8, 4)  # resampled (area average) to the grid
    assert float(from_tensor.mean()) == pytest.approx(float(t.mean()), rel=1e-6)
    from_fn = HeatOperator(dom, n_steps=2, initial=lambda d: torch.ones(d.shape)).initial_state()
    assert from_fn.dtype == F64 and float(from_fn.min()) == 1.0
    with pytest.raises(ConfigError):
        HeatOperator(dom, initial="laser")
    # cell averages conserve the deposited heat across resolutions (finite-volume initialization)
    ic = GaussianFlash(100.0, 2.5, 0.1)
    heat = [float(ic(dom.at(s)).sum()) * 100.0 / math.prod(s) for s in ((64, 64, 16), (8, 8, 2))]
    assert heat[1] == pytest.approx(heat[0], rel=1e-9)
    point = GaussianFlash(100.0, 2.5, 0.1, cell_average=False)(dom.at((8, 8, 2)))
    assert float(point.sum()) * 100.0 / 128 < 0.5 * heat[0]
