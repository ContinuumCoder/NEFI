"""Validation of the variable-coefficient elliptic solver (nefi.physics.elliptic)."""

import math
from pathlib import Path

import pytest
import torch
import torch.autograd.forward_ad as fwAD
from torch.func import grad, hessian, jacfwd, jacrev, jvp, vjp, vmap

from nefi.domain import Domain
from nefi.errors import ConfigError, ShapeError
from nefi.physics.elliptic import (
    EllipticOperator,
    LogBounded,
    apply_divgrad,
    divgrad_diagonal,
    is_transformed,
    masked_mean,
    normalize_bc,
    point_source_rhs,
    solve_elliptic,
    weighted_area_downsample,
)

DT = torch.float64
BCS = [
    "dirichlet",
    "neumann",
    "periodic",
    [("dirichlet", "neumann"), "periodic"],
    [("neumann", "dirichlet"), ("dirichlet", "neumann")],
]


def _grid(n, lo=0.0, hi=1.0):
    h = (hi - lo) / n
    x = lo + (torch.arange(n, dtype=DT) + 0.5) * h
    return h, torch.meshgrid(x, x, indexing="ij")


def test_constant_sigma_reproduces_5_and_7_point_laplacians():
    torch.manual_seed(0)
    n, h = 7, 0.1
    u = torch.randn(n, n, dtype=DT)
    au = apply_divgrad(u, torch.ones(n, n, dtype=DT), h, "dirichlet")
    up = torch.nn.functional.pad(u, (1, 1, 1, 1))
    up[0, :], up[-1, :], up[:, 0], up[:, -1] = -up[1, :], -up[-2, :], -up[:, 1], -up[:, -2]
    lap = (up[2:, 1:-1] + up[:-2, 1:-1] + up[1:-1, 2:] + up[1:-1, :-2] - 4 * u) / h**2
    assert torch.allclose(au, -lap, atol=1e-10)
    u3 = torch.randn(2, 4, 5, 6, dtype=DT)  # batch of 2, periodic 3-D
    a3 = apply_divgrad(u3, torch.ones(4, 5, 6, dtype=DT), (0.5, 0.5, 0.5), "periodic")
    l3 = sum(torch.roll(u3, 1, d) + torch.roll(u3, -1, d) - 2 * u3 for d in (1, 2, 3)) / 0.25
    assert torch.allclose(a3, -l3, atol=1e-10)
    # neumann: constants are in the null space
    c = torch.full((5, 6), 3.0, dtype=DT)
    assert apply_divgrad(c, torch.rand(5, 6, dtype=DT) + 0.1, 0.2, "neumann").abs().max() < 1e-12


def test_dirichlet_poisson_is_second_order_and_cg_converges():
    errs = []
    for n in (8, 16, 32, 64):
        h, (x, y) = _grid(n)
        f = 2 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        exact = torch.sin(math.pi * x) * torch.sin(math.pi * y)
        sigma = torch.ones(n, n, dtype=DT)
        u, info = solve_elliptic(sigma, f, h, "dirichlet", tol=1e-12, return_info=True)
        assert info.converged and info.residual <= 1e-12
        true_res = (apply_divgrad(u, sigma, h, "dirichlet") - f).norm() / f.norm()
        assert true_res < 1e-10
        errs.append(float((u - exact).abs().max()))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    assert all(3.6 < r < 4.4 for r in ratios), ratios  # error ↓ ×4 per refinement


def test_variable_coefficient_manufactured_solution_second_order():
    errs = []
    for n in (16, 32, 64):
        h, (x, y) = _grid(n)
        sigma = 1.0 + 0.5 * x + 0.25 * y
        u_ex = torch.sin(math.pi * x) * torch.sin(math.pi * y)
        ux = math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)
        uy = math.pi * torch.sin(math.pi * x) * torch.cos(math.pi * y)
        f = -(0.5 * ux + 0.25 * uy) + 2 * math.pi**2 * sigma * u_ex  # −∇·(σ∇u)
        u = solve_elliptic(sigma, f, h, "dirichlet", tol=1e-12)
        errs.append(float((u - u_ex).abs().max()))
    assert errs[0] / errs[1] > 3.5 and errs[1] / errs[2] > 3.5, errs


@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("with_kappa", [False, True])
def test_operator_is_symmetric(bc, with_kappa):
    torch.manual_seed(1)
    s = torch.rand(6, 5, dtype=DT) + 0.1
    a, b = torch.randn(3, 6, 5, dtype=DT), torch.randn(3, 6, 5, dtype=DT)
    k = torch.rand(6, 5, dtype=DT) if with_kappa else None
    lhs = (apply_divgrad(a, s, (0.3, 0.2), bc, kappa=k) * b).sum()
    rhs = (a * apply_divgrad(b, s, (0.3, 0.2), bc, kappa=k)).sum()
    assert abs(float(lhs - rhs)) < 1e-10 * float(lhs.abs() + 1)
    # positive semi-definite
    assert float((apply_divgrad(a, s, (0.3, 0.2), bc, kappa=k) * a).sum()) > 0


def test_symmetry_3d_mixed_and_diagonal():
    torch.manual_seed(2)
    s = torch.rand(4, 5, 3, dtype=DT) + 0.1
    a, b = torch.randn(4, 5, 3, dtype=DT), torch.randn(4, 5, 3, dtype=DT)
    bc = ["dirichlet", "periodic", ("neumann", "dirichlet")]
    sp = (0.1, 0.2, 0.3)
    lhs = (apply_divgrad(a, s, sp, bc) * b).sum()
    assert abs(float(lhs - (a * apply_divgrad(b, s, sp, bc)).sum())) < 1e-9
    d = divgrad_diagonal(s, sp, bc)
    e = torch.zeros_like(s)
    for idx in [(0, 0, 0), (3, 4, 2), (1, 2, 1), (0, 4, 1)]:
        e.zero_()
        e[idx] = 1.0
        assert abs(float(apply_divgrad(e, s, sp, bc)[idx] - d[idx])) < 1e-10


def test_pure_neumann_compatibility_and_gauge():
    torch.manual_seed(3)
    n = 12
    s = torch.rand(n, n, dtype=DT) + 0.5
    b = torch.randn(2, n, n, dtype=DT) + 3.0  # incompatible: non-zero mean
    u, info = solve_elliptic(s, b, 1.0 / n, "neumann", tol=1e-12, return_info=True)
    assert info.converged
    proj = b - b.mean(dim=(-2, -1), keepdim=True)
    assert (apply_divgrad(u, s, 1.0 / n, "neumann") - proj).abs().max() < 1e-9
    assert u.mean(dim=(-2, -1)).abs().max() < 1e-12
    # periodic + neumann mix is singular too; with kappa > 0 it is not
    u2 = solve_elliptic(s, proj, 1.0 / n, ["periodic", "neumann"], tol=1e-12)
    assert abs(float(u2.mean())) < 1e-12
    k = torch.full((n, n), 0.5, dtype=DT)
    u3 = solve_elliptic(s, b, 1.0 / n, "neumann", kappa=k, tol=1e-12)
    assert (apply_divgrad(u3, s, 1.0 / n, "neumann", kappa=k) - b).abs().max() < 1e-9


def test_float32_tolerance_floor_and_warm_start():
    torch.manual_seed(4)
    n = 24
    s = torch.rand(n, n) + 0.5
    b = torch.randn(n, n)
    u, info = solve_elliptic(s, b, 1.0 / n, "dirichlet", tol=1e-12, return_info=True)
    assert info.converged and info.tol >= 8 * torch.finfo(torch.float32).eps
    u2, info2 = solve_elliptic(s * 1.01, b, 1.0 / n, "dirichlet", x0=u, return_info=True)
    u3, info3 = solve_elliptic(s * 1.01, b, 1.0 / n, "dirichlet", return_info=True)
    assert info2.iterations < info3.iterations
    assert (u2 - u3).abs().max() < 1e-4 * u3.abs().max()


def _grads(mode, bc, kappa_scale):
    torch.manual_seed(5)
    s = (torch.rand(6, 5, dtype=DT) + 0.3).requires_grad_(True)
    b = torch.randn(2, 6, 5, dtype=DT).requires_grad_(True)
    ins = [s, b]
    k = None
    if kappa_scale:
        k = (torch.rand(6, 5, dtype=DT) * kappa_scale).requires_grad_(True)
        ins.append(k)
    w = torch.randn(2, 6, 5, dtype=DT)
    u = solve_elliptic(s, b, (0.2, 0.3), bc, kappa=k, tol=1e-13, grad_mode=mode)
    loss = (w * u).sum() + (u**2).sum()
    return torch.autograd.grad(loss, ins)


@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("kappa_scale", [0.0, 2.0])
def test_ift_adjoint_matches_unrolled_autograd(bc, kappa_scale):
    gi = _grads("ift", bc, kappa_scale)
    ga = _grads("autograd", bc, kappa_scale)
    for a, b in zip(gi, ga):
        assert float((a - b).norm() / b.norm()) < 1e-6


def test_gradcheck_operator_wrt_sigma_and_rhs():
    torch.manual_seed(6)
    dom = Domain.unit((4, 4))
    s = (torch.rand(4, 4, dtype=DT) + 0.5).requires_grad_(True)
    b = torch.randn(2, 4, 4, dtype=DT).requires_grad_(True)
    op = EllipticOperator(dom, b.detach(), [("dirichlet", "neumann"), "periodic"], tol=1e-14)
    assert torch.autograd.gradcheck(lambda x: op({"sigma": x}), (s,), eps=1e-6, atol=1e-7)

    def f(x, rhs):
        return solve_elliptic(x, rhs, 0.25, "neumann", tol=1e-14)

    assert torch.autograd.gradcheck(f, (s, b), eps=1e-6, atol=1e-7)
    k = (torch.rand(4, 4, dtype=DT) + 0.1).requires_grad_(True)

    def g(x, rhs, kap):
        return solve_elliptic(x, rhs, 0.25, "dirichlet", kappa=kap, tol=1e-14)

    assert torch.autograd.gradcheck(g, (s, b, k), eps=1e-6, atol=1e-7)


def test_double_backward_gradgradcheck():
    torch.manual_seed(7)
    s = (torch.rand(3, 3, dtype=DT) + 0.5).requires_grad_(True)
    b = torch.randn(3, 3, dtype=DT).requires_grad_(True)
    for bc in ("dirichlet", "neumann"):

        def f(x, rhs, bc=bc):
            return solve_elliptic(x, rhs, 0.25, bc, tol=1e-14)

        assert torch.autograd.gradgradcheck(f, (s, b), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_harmonic_vs_arithmetic_face_mean():
    torch.manual_seed(8)
    n = 16
    h = 1.0 / n
    b = torch.zeros(n, n, dtype=DT)
    b[2, n // 2], b[n - 3, n // 2] = 1.0 / h**2, -1.0 / h**2  # dipole source / sink
    # identical for uniform sigma
    s1 = torch.full((n, n), 2.0, dtype=DT)
    uh = solve_elliptic(s1, b, h, "neumann", face_mode="harmonic", tol=1e-12)
    ua = solve_elliptic(s1, b, h, "neumann", face_mode="arithmetic", tol=1e-12)
    assert (uh - ua).abs().max() < 1e-10
    # thin insulating inclusion (1:100, a crack across the current path): the harmonic mean
    # throttles the flux entering it, the arithmetic mean lets it leak through
    s = torch.ones(n, n, dtype=DT)
    s[8, 3:13] = 0.01

    def flux_into_inclusion(face_mode):
        u = solve_elliptic(s, b, h, "neumann", face_mode=face_mode, tol=1e-12)
        a, c = s[7, 3:13], s[8, 3:13]
        mean = 2 * a * c / (a + c) if face_mode == "harmonic" else 0.5 * (a + c)
        return float((mean * (u[8, 3:13] - u[7, 3:13]) / h).abs().sum() * h)

    assert flux_into_inclusion("harmonic") < 0.5 * flux_into_inclusion("arithmetic")
    # series layers: harmonic reproduces the exact series resistance, arithmetic underestimates
    m = 20
    layer = torch.where((torch.arange(m) // 2) % 2 == 0, 1.0, 0.05).to(DT)
    sig = layer[:, None].expand(m, 3).contiguous()
    q = torch.zeros(m, 3, dtype=DT)
    q[0, :], q[-1, :] = 1.0 / h, -1.0 / h  # unit flux density in at x=0, out at x=1
    drops = {}
    for mode in ("harmonic", "arithmetic"):
        u = solve_elliptic(sig, q, h, ["neumann", "neumann"], face_mode=mode, tol=1e-13)
        drops[mode] = float(u[0, 1] - u[-1, 1])
    exact = (
        sum(h / float(v) for v in layer[:-1])
        - 0.5 * h / float(layer[0])
        + 0.5 * h / float(layer[-1])
    )  # centre-to-centre series resistance
    assert abs(drops["harmonic"] - exact) < 1e-8 * exact
    assert drops["arithmetic"] < 0.8 * exact


def test_elliptic_operator_multiscale_batched_transform_and_kappa_field():
    dom = Domain.unit((16, 16))

    def rhs_fn(d):
        return torch.stack([point_source_rhs(d, [[0.3, 0.5], [0.7, 0.5]], [1.0, -1.0])] * 2)

    op = EllipticOperator(dom, rhs_fn, "neumann", sigma_transform="exp", field="log_s", tol=1e-10)
    assert op.n_drives == 2 and op.output_shape((8, 8)) == (2, 8, 8)
    f = torch.zeros(16, 16, dtype=DT)
    u = op({"log_s": f})
    assert u.shape == (2, 16, 16)
    op8 = op.at_resolution((8, 8))
    assert op8 is op.at_resolution((8, 8)) and op8.domain.shape == (8, 8)
    assert op8({"log_s": torch.zeros(8, 8, dtype=DT)}).shape == (2, 8, 8)
    ident = EllipticOperator(dom, rhs_fn, "neumann", tol=1e-10)
    assert torch.allclose(ident({"sigma": torch.ones(16, 16, dtype=DT)}), u, atol=1e-8)
    # a tensor RHS is resampled; a reference mask removes the gauge constant
    ref = torch.zeros(16, 16, dtype=DT)
    ref[0, :] = 1.0
    op_t = EllipticOperator(dom, rhs_fn(dom), "neumann", reference=ref, tol=1e-10)
    v = op_t({"sigma": torch.ones(16, 16, dtype=DT)})
    assert abs(float(v[:, 0, :].mean())) < 1e-10
    assert op_t.at_resolution((8, 8))({"sigma": torch.ones(8, 8, dtype=DT)}).shape == (2, 8, 8)
    # unknown absorption field (DOT): kappa_field is differentiable
    dot = EllipticOperator(dom, rhs_fn, "dirichlet", kappa_field="mua", tol=1e-12)
    assert dot.required_fields() == ("sigma", "mua")
    mua = torch.full((16, 16), 0.1, dtype=DT, requires_grad=True)
    dot({"sigma": torch.ones(16, 16, dtype=DT), "mua": mua}).square().sum().backward()
    assert mua.grad is not None and torch.isfinite(mua.grad).all()
    with pytest.raises(ShapeError):
        op({"log_s": torch.zeros(8, 8, dtype=DT)})
    with pytest.raises(ConfigError):
        EllipticOperator(dom, rhs_fn, "dirichlet", kappa=0.1, kappa_field="mua")


def test_operator_warm_start_and_adjoint_info():
    torch.manual_seed(9)
    dom = Domain.unit((20, 20))
    b = torch.randn(3, 20, 20, dtype=DT)
    op = EllipticOperator(dom, b, "dirichlet", tol=1e-10, warm_start=True)
    s = (torch.rand(20, 20, dtype=DT) + 0.5).requires_grad_(True)
    op({"sigma": s}).square().sum().backward()
    it_cold, adj_cold = op.last_info.iterations, op.last_adjoint_info.iterations
    s2 = (s.detach() * 1.001).requires_grad_(True)
    op({"sigma": s2}).square().sum().backward()
    assert op.last_info.iterations < it_cold
    assert op.last_adjoint_info.iterations < adj_cold
    op.reset_warm_start()


def test_point_source_rhs_conserves_rate_and_helpers():
    dom = Domain((10, 8), ((0.0, 1.0), (0.0, 2.0)))
    vol = math.prod(dom.spacing())
    for spread in ("linear", "nearest"):
        r = point_source_rhs(dom, [[0.33, 0.71], [0.9, 1.9]], [2.0, -0.5], spread=spread)
        assert abs(float(r.sum()) * vol - 1.5) < 1e-12
    x = torch.randn(2, 4, 4, dtype=DT)
    w = torch.zeros(4, 4, dtype=DT)
    w[0, :] = 1.0
    assert torch.allclose(masked_mean(x, w, 2)[..., 0, 0], x[:, 0, :].mean(-1))
    d, wc = weighted_area_downsample(x, w, (2, 2))
    assert torch.allclose(d[:, 0, 0], x[:, 0, :2].mean(-1)) and float(wc[0, 1, 1]) == 0.0


def test_log_bounded_head_and_bc_validation():
    head = LogBounded(0.05, 20.0, init_value=1.0)
    raw = torch.linspace(-30, 30, 101)[:, None]
    out = head(raw)
    assert float(out.min()) >= 0.05 * (1 - 1e-6) and float(out.max()) <= 20.0 * (1 + 1e-6)
    assert abs(float(head(head.inverse(torch.tensor([3.0])))[0]) - 3.0) < 1e-5
    assert abs(float(head(torch.tensor([head.init_bias()]))[0]) - 1.0) < 1e-5
    assert normalize_bc("dirichlet", 2) == (("dirichlet", "dirichlet"),) * 2
    assert normalize_bc([("dirichlet", "neumann"), "periodic"], 2)[0] == ("dirichlet", "neumann")
    with pytest.raises(ConfigError):
        normalize_bc([("periodic", "neumann"), "dirichlet"], 2)
    with pytest.raises(ConfigError):
        normalize_bc("robin", 2)
    with pytest.raises(ConfigError):
        normalize_bc(["dirichlet"], 2)


# ---------------------------------------------------------------------------------------------
# forward mode / torch.func (regression: warm-started solves used to return truncated tangents)
# ---------------------------------------------------------------------------------------------
SP = (0.2, 0.25)


def _fwd_setup():
    torch.manual_seed(10)
    s = torch.rand(6, 5, dtype=DT) + 0.5
    b = torch.randn(2, 6, 5, dtype=DT)
    k = torch.rand(6, 5, dtype=DT) + 0.1
    vs, vb, vk = torch.randn(6, 5, dtype=DT), torch.randn(2, 6, 5, dtype=DT), torch.randn(6, 5)
    return s, b, k, vs, vb, vk.to(DT)


@pytest.mark.parametrize("bc", BCS[:4])
@pytest.mark.parametrize("with_kappa", [False, True])
def test_forward_mode_jvp_is_exact_even_when_warm_started(bc, with_kappa):
    s, b, k, vs, vb, vk = _fwd_setup()
    kk = k if with_kappa else None

    def f(s_, b_, k_, x0=None):
        return solve_elliptic(s_, b_, SP, bc, kappa=k_, tol=1e-13, x0=x0)

    eps = 1e-6
    kp = None if kk is None else kk + eps * vk
    km = None if kk is None else kk - eps * vk
    fd = (f(s + eps * vs, b + eps * vb, kp) - f(s - eps * vs, b - eps * vb, km)) / (2 * eps)
    u0 = f(s, b, kk)
    for x0 in (None, u0):  # the warm start at the solution stops CG at iteration 0
        if kk is None:
            _, t = jvp(lambda a, c, x0=x0: f(a, c, None, x0), (s, b), (vs, vb))
        else:
            _, t = jvp(lambda a, c, d, x0=x0: f(a, c, d, x0), (s, b, kk), (vs, vb, vk))
        assert float((t - fd).norm() / fd.norm()) < 1e-6
    with fwAD.dual_level():  # plain forward-AD API
        out = f(
            fwAD.make_dual(s, vs),
            fwAD.make_dual(b, vb),
            None if kk is None else fwAD.make_dual(kk, vk),
            u0,
        )
        t = fwAD.unpack_dual(out).tangent
    assert float((t - fd).norm() / fd.norm()) < 1e-6


def test_gradcheck_with_forward_ad():
    s, b, k, *_ = _fwd_setup()
    s, b, k = s[:4, :4].clone(), b[:, :4, :4].clone(), k[:4, :4].clone()
    for bc in ("neumann", [("dirichlet", "neumann"), "periodic"]):

        def f(x, rhs, kap, bc=bc):
            return solve_elliptic(x, rhs, 0.25, bc, kappa=kap, tol=1e-14)

        ins = tuple(t.requires_grad_(True) for t in (s.clone(), b.clone(), k.clone()))
        assert torch.autograd.gradcheck(f, ins, eps=1e-6, atol=1e-7, check_forward_ad=True)


@pytest.mark.parametrize("grad_mode", ["autograd", "none"])
def test_forward_mode_guard_for_unrolled_and_no_grad_modes(grad_mode):
    s, b, _, vs, _, _ = _fwd_setup()
    u0 = solve_elliptic(s, b, SP, "dirichlet", tol=1e-13)

    def f(x):
        return solve_elliptic(x, b, SP, "dirichlet", tol=1e-13, grad_mode=grad_mode, x0=u0)

    with pytest.raises(NotImplementedError, match="double_backward"):
        jvp(f, (s,), (vs,))
    with fwAD.dual_level(), pytest.raises(NotImplementedError):
        f(fwAD.make_dual(s, vs))
    assert not is_transformed(s) and not is_transformed(None)


def test_torch_func_transforms_through_the_implicit_solve():
    s, b, *_ = _fwd_setup()
    w = torch.randn(2, 6, 5, dtype=DT)

    def loss(x):
        return (w * solve_elliptic(x, b, SP, "dirichlet", tol=1e-13)).sum()

    x = s.clone().requires_grad_(True)
    (ref,) = torch.autograd.grad(loss(x), x)
    assert torch.allclose(grad(loss)(s), ref, rtol=1e-10, atol=1e-12)
    _, pullback = vjp(lambda x: solve_elliptic(x, b, SP, "dirichlet", tol=1e-13), s)
    assert torch.allclose(pullback(w)[0], ref, rtol=1e-10, atol=1e-12)
    # vmap: batch of coefficients (pure Neumann, 2 drives) and batch of right-hand sides
    S = torch.rand(3, 6, 5, dtype=DT) + 0.5
    out = vmap(lambda x: solve_elliptic(x, b, SP, "neumann", tol=1e-13))(S)
    loop = torch.stack([solve_elliptic(S[i], b, SP, "neumann", tol=1e-13) for i in range(3)])
    assert out.shape == (3, 2, 6, 5) and torch.allclose(out, loop, atol=1e-12)
    B = torch.randn(4, 6, 5, dtype=DT)
    out = vmap(lambda r: solve_elliptic(s, r, SP, "dirichlet", tol=1e-13))(B)
    loop = torch.stack([solve_elliptic(s, B[i], SP, "dirichlet", tol=1e-13) for i in range(4)])
    assert torch.allclose(out, loop, atol=1e-12)
    # Jacobians and Hessian (vmap over the VJP / JVP rules)
    s3, b3 = torch.rand(3, 3, dtype=DT) + 0.5, torch.randn(3, 3, dtype=DT)

    def F(x):
        return solve_elliptic(x, b3, 0.25, "dirichlet", tol=1e-13)

    jr, jf = jacrev(F)(s3), jacfwd(F)(s3)
    eye = torch.eye(9, dtype=DT).reshape(9, 3, 3)
    jfd = torch.stack([(F(s3 + 1e-6 * e) - F(s3 - 1e-6 * e)) / 2e-6 for e in eye], -1)
    assert torch.allclose(jr, jf, atol=1e-12)
    assert torch.allclose(jf, jfd.reshape(3, 3, 3, 3), atol=1e-8)

    def G(x):
        return solve_elliptic(x, b3, 0.25, "neumann", tol=1e-13).pow(2).sum()

    h = hessian(G)(s3).reshape(9, 9)
    assert torch.allclose(h, h.T, atol=1e-12)
    assert torch.allclose(h, torch.autograd.functional.hessian(G, s3).reshape(9, 9), atol=1e-10)


def test_operator_forward_mode_with_warm_start_keeps_caches_plain():
    s, b, _, vs, _, _ = _fwd_setup()
    op = EllipticOperator(Domain.unit((6, 5)), b, "dirichlet", tol=1e-13, warm_start=True)
    op({"sigma": s})  # populate the warm-start cache
    _, t = jvp(lambda x: op({"sigma": x}), (s,), (vs,))
    ref = solve_elliptic(s, b, op.spacing, "dirichlet", tol=1e-13)
    _, t_ref = jvp(lambda x: solve_elliptic(x, b, op.spacing, "dirichlet", tol=1e-13), (s,), (vs,))
    assert torch.allclose(t, t_ref, atol=1e-10)
    assert not is_transformed(op._warm["u"])  # no dead torch.func wrapper kept as x0
    assert torch.allclose(op({"sigma": s}), ref, atol=1e-10)


def test_eit_singular_values_forward_mode_matches_double_backward():
    import nefi.diagnostics as D
    from nefi.config import load_config
    from nefi.diagnostics._common import jvp as diag_jvp
    from nefi.diagnostics._common import operator_fn, resolve_fields
    from nefi.registry import build

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "eit_smoke.yaml")
    inst = build("instance", cfg["instance"])
    _, meas = inst.make_measurement(0)
    torch.manual_seed(0)
    problem = inst.build_problem(meas)
    auto = D.singular_values(problem, k=3, n_iter=30, jvp_mode="auto")
    exact = D.singular_values(problem, k=3, n_iter=30, jvp_mode="double_backward")
    assert float(((auto - exact).abs() / exact).max()) < 1e-4
    # "auto" must use the forward rule, not a silent fallback
    fields, dom = resolve_fields(problem, None, None, 1.0)
    fn, x0, _ = operator_fn(problem, fields, problem.field.primary, dom, True)
    assert diag_jvp(fn, x0, torch.randn_like(x0), "auto")[1] == "forward"


def test_operator_caches_do_not_keep_functorch_wrappers():
    """A fresh EllipticOperator first used inside torch.func must not cache wrapped tensors."""
    import torch

    from nefi.domain import Domain
    from nefi.physics.elliptic import EllipticOperator, is_transformed

    dom = Domain.unit((6, 6))
    op = EllipticOperator(dom, rhs=torch.ones(6, 6, dtype=torch.float64), bc="dirichlet")
    sigma = torch.ones(6, 6, dtype=torch.float64) * 1.3

    def f(s):
        return op({"sigma": s}).sum()

    _, tangent = torch.func.jvp(f, (sigma,), (torch.ones_like(sigma),))
    assert torch.isfinite(tangent)
    assert not any(is_transformed(v) for v in op._cache.values())
    out = op({"sigma": sigma})  # plain call afterwards must work
    assert torch.isfinite(out).all()
