"""Diagnostics suite (nefi.diagnostics) on toy1d and small synthetic problems."""

import json
import math

import pytest
import torch

import nefi
from nefi import diagnostics as D
from nefi.domain import Domain
from nefi.fields import GridField, Heads, Softplus
from nefi.instances.toy1d import make_problem
from nefi.losses import MSE, Context, LossSet
from nefi.measurement import Measurement
from nefi.operators import FFTConvolution, LambdaOperator, gaussian_kernel_fn
from nefi.problem import InverseProblem

TOY = {"n": 64, "seed": 0, "hidden": 32, "depth": 3, "n_octaves": 5, "steps": (40, 60)}


@pytest.fixture(scope="module")
def toy():
    torch.manual_seed(0)
    return make_problem(**TOY)


def _grid(problem):
    field = GridField(problem.domain.shape, Heads({"x": Softplus(init_value=0.1)}))
    return InverseProblem(
        problem.domain, field, problem.operator, problem.losses, problem.measurement
    )


def _loss_at(problem, x):
    pred = problem.operator({"x": x})
    ctx = Context({"x": x}, pred, problem.measurement, problem.domain)
    return float(problem.losses(ctx)[0])


def test_sensitivity_map_finite_positive_and_exact(toy):
    problem, gt, meas = toy
    s = D.sensitivity_map(problem, n_probes=16)
    assert s.shape == (64,) and torch.isfinite(s).all() and (s > 0).all()
    exact = D.sensitivity_map(problem, exact=True)
    assert (exact > 0).all()
    # interior column of a convolution = the kernel's l2 norm; truncated at the window edge (P2)
    k = problem.operator.kernel((64,))
    assert abs(float(exact[32]) / float(k.norm()) - 1.0) < 1e-3
    assert exact[0] < exact[32]
    assert D.center_to_outer_ratio(exact) > 1.0
    # both stochastic estimators are unbiased
    many = D.sensitivity_map(problem, n_probes=256)
    assert float(((many - exact).abs() / exact).mean()) < 0.1
    hutch = D.sensitivity_map(problem, n_probes=256, method="hutchinson")
    assert float(((hutch - exact).abs() / exact).mean()) < 0.15
    ratio = D.center_to_outer_ratio(s)
    assert ratio > 0 and math.isfinite(ratio)


def test_iter0_gradient_center_bias_signature():
    # a wide blur on a finite window: the uniform-init data gradient peaks at the center (P2)
    dom = Domain.unit((64,))
    op = FFTConvolution(gaussian_kernel_fn(0.1), dom)
    meas = Measurement(op({"x": torch.ones(64)}))
    field = GridField((64,), Heads({"x": "identity"}), init=0.1)
    prob = InverseProblem(dom, field, op, LossSet({"data": MSE()}), meas)
    sig = D.iter0_gradient(prob)
    assert sig.ratio > 1.5 and sig.peak_radius < 0.2, sig.to_dict()
    assert sig.grad.shape == (64,) and set(sig.grads) == {"x"}
    assert 0.0 <= sig.center_mass <= 1.0
    assert D.center_mass_ratio(torch.ones(64)) == pytest.approx(D.uniform_center_mass((64,)))
    spike = torch.zeros(32, 32)
    spike[16, 16] = 1.0
    assert D.center_mass_ratio(spike) == 1.0
    assert abs(D.uniform_center_mass((64, 64)) - math.pi * 0.15**2) < 0.01


def test_field_gradient_matches_autograd(toy):
    problem, gt, meas = toy
    x = gt["x"].clone().requires_grad_(True)
    pred = problem.operator({"x": x})
    total, _ = problem.losses(Context({"x": x}, pred, problem.measurement, problem.domain))
    (ref,) = torch.autograd.grad(total, x)
    g = D.field_gradient(problem, gt)
    assert torch.allclose(g["x"], ref, atol=1e-7)
    g_data = D.field_gradient(problem, gt, terms="data")["x"]
    assert torch.isfinite(g_data).all()


def test_filter_kernel_row_grid_delta_and_neural_bump(toy):
    problem, gt, meas = toy
    grid = _grid(problem)
    row_g = D.filter_kernel_row(grid, 32)
    assert (row_g.abs() > 1e-12).nonzero().flatten().tolist() == [32]
    assert D.kernel_spread(row_g) == 1.0
    row_n = D.filter_kernel_row(problem, "center", progress=1.0)
    assert int(row_n.abs().argmax()) == 32
    assert D.kernel_spread(row_n) > 1.0
    row_0 = D.filter_kernel_row(problem, "center", progress=0.0)
    assert D.kernel_spread(row_0) >= D.kernel_spread(row_n)  # annealing: smoother at β = 0
    assert D.effective_bandwidth(problem.field, 0.0) == 0.0
    assert D.effective_bandwidth(problem.field, 1.0) > D.effective_bandwidth(problem.field, 0.5)
    assert D.effective_bandwidth(grid.field, 1.0) is None


def test_realized_update_grid_verbatim_vs_neural(toy):
    problem, gt, meas = toy
    ru_g = D.realized_update(_grid(problem), optimizer="sgd")
    assert ru_g.alignment > 0.999 and abs(ru_g.damping - 1.0) < 1e-3  # G_θ = I
    ru_n = D.realized_update(problem)
    assert ru_n.delta.shape == problem.curriculum.stages[0].shape
    assert ru_n.optimizer == "adamw" and ru_n.progress == 0.0
    assert ru_n.alignment < 0.99  # the neural field bends the raw gradient
    assert set(ru_n.to_dict()) >= {"delta_ratio", "grad_ratio", "damping"}


def test_energy_barrier_endpoints_and_double_well(toy):
    problem, gt, meas = toy
    a = torch.full((64,), float(gt["x"].mean()))
    eb = D.energy_barrier(problem, a, gt, n=5)
    assert eb.loss.shape == (5,) and eb.t[0] == 0.0 and eb.t[-1] == 1.0
    assert abs(float(eb.loss[0]) - _loss_at(problem, a)) < 1e-6
    assert abs(float(eb.loss[-1]) - _loss_at(problem, gt["x"])) < 1e-6
    assert eb.height == 0.0  # convex loss: no barrier along a straight path
    # non-convex double well (x² = 1): the path -1 → +1 must climb over x = 0
    dom = Domain.unit((8,))
    op = LambdaOperator(lambda f: f["x"] ** 2, homogeneity=2.0, output_shape_fn=lambda s: s)
    field = GridField((8,), Heads({"x": "identity"}), init=0.5)
    prob = InverseProblem(dom, field, op, LossSet({"data": MSE()}), Measurement(torch.ones(8)))
    well = D.energy_barrier(prob, -torch.ones(8), torch.ones(8), n=11)
    assert abs(well.height - 1.0) < 1e-6 and abs(well.t_max - 0.5) < 1e-9 and not well.monotone


def test_singular_values_decay_and_operator_norm(toy):
    problem, gt, meas = toy
    sv = D.singular_values(problem, k=6, n_iter=40)
    assert sv.shape == (6,) and torch.all(sv[1:] <= sv[:-1] + 1e-9)
    # compare with a dense SVD of the 64×64 convolution matrix
    cols = problem.operator({"x": torch.eye(64)})  # row i = J e_i
    dense = torch.linalg.svdvals(cols.T.double())
    assert torch.allclose(sv, dense[:6], rtol=1e-3)
    # operator norm ≈ max |FFT(kernel)| (= 1 for a normalized Gaussian) up to window truncation
    k = problem.operator.kernel((64,)).double()
    kmax = float(torch.fft.rfft(k, n=8192).abs().max())
    assert abs(float(sv[0]) - kmax) / kmax < 0.02
    sv2, vecs = D.singular_values(problem, k=3, n_iter=40, return_vectors=True)
    assert vecs.shape == (3, 64)


def test_hessian_condition_number_quadratic(toy):
    A = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
    ev = torch.linalg.eigvalsh(A)
    kappa = D.hessian_condition_number(lambda p: 0.5 * p @ A @ p, [0.3, -0.2])
    assert abs(kappa - float(ev.max() / ev.min())) < 1e-9
    kappa2 = D.hessian_condition_number(lambda p: p[0] ** 2 + 931.0 * p[1] ** 2, [0.0, 0.0])
    assert abs(kappa2 - 931.0) < 1e-6  # NeTMY κ_F2 reference value
    # ansatz objective: Gaussian bump (log10 A, width) through the toy problem's loss
    problem, gt, meas = toy

    def bump(p, r):
        return 10 ** p[0] * torch.exp(-((r[..., 0] - 0.5) ** 2) / (2 * p[1] ** 2))

    fn = D.ansatz_objective(problem, bump)
    val = fn(torch.tensor([0.0, 0.1]))
    assert val.ndim == 0 and torch.isfinite(val)
    assert math.isfinite(D.hessian_condition_number(fn, [0.0, 0.1]))


def test_jvp_modes_and_custom_function_fallback():
    x, v = torch.randn(5, dtype=torch.float64), torch.randn(5, dtype=torch.float64)
    ref = 2 * torch.cos(x) * v
    for mode in ("forward", "double_backward", "finite_difference"):
        jv, used = D.jvp(lambda z: 2 * torch.sin(z), x, v, mode)
        assert used == mode and torch.allclose(jv, ref, atol=1e-6)

    class Square(torch.autograd.Function):  # no forward-mode rule
        @staticmethod
        def forward(ctx, z):
            ctx.save_for_backward(z)
            return z**2

        @staticmethod
        def backward(ctx, g):
            (z,) = ctx.saved_tensors
            return 2 * z * g

    jv, used = D.jvp(Square.apply, x, v)
    assert used in ("double_backward", "finite_difference")
    assert torch.allclose(jv, 2 * x * v, atol=1e-5)


def test_data_fit_paradox_and_diagnose(tmp_path):
    torch.manual_seed(0)
    problem, gt, meas = make_problem(**TOY)
    rep0 = D.diagnose(problem, gt=gt, n_probes=8, k=4, n_iter=10)
    assert not rep0.errors, rep0.errors
    assert rep0.data_fit is None and rep0.energy_barrier is not None
    res = nefi.invert(problem, problem.curriculum.scaled(0.5), device="cpu")
    d = D.data_fit_paradox(res, problem, gt)
    for key in (
        "data_mse",
        "data_rmse",
        "data_psnr",
        "data_rel_residual",
        "noise_std",
        "discrepancy_ratio",
        "field_mse",
        "field_psnr",
        "field_relative_error",
        "psnr_gap",
    ):
        assert key in d and math.isfinite(d[key]), key
    rep = D.diagnose(problem, gt=gt, result=res, n_probes=8, k=4, n_iter=10)
    assert not rep.errors, rep.errors
    md = rep.to_markdown()
    for token in ("Sensitivity", "18.29", "Data-fit paradox", "singular values", "Center-mass"):
        assert token in md, token
    paths = rep.save(tmp_path)
    assert all(p.exists() for p in paths.values())
    summary = json.loads(paths["json"].read_text())["summary"]
    assert "sensitivity_ratio" in summary and "data_psnr" in summary


def test_plots(toy, tmp_path):
    pytest.importorskip("matplotlib")
    from nefi.diagnostics import plots

    problem, gt, meas = toy
    rep = D.diagnose(problem, gt=gt, n_probes=4, k=3, n_iter=8)
    p = plots.save_figure(plots.plot_report(rep), tmp_path / "report.png")
    assert p.exists() and p.stat().st_size > 1000
    rows = {"grid": D.filter_kernel_row(_grid(problem)), "neural": D.filter_kernel_row(problem)}
    assert plots.plot_filter_kernels(rows) is not None
    assert plots.plot_sensitivity(torch.rand(8, 8)) is not None
    assert plots.show_field(torch.rand(4, 4, 3)) is not None
