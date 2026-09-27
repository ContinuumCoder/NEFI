"""Baseline parameterizations and solvers: splats, deep decoder, ADMM, L-BFGS, baseline_problem."""

import math

import pytest
import torch

import nefi
from nefi.baselines import (
    BASELINE_KINDS,
    ADMMConfig,
    ADMMSolver,
    DeepDecoderField,
    DirectSolver,
    GaussianSplatField,
    SplatControl,
    baseline_problem,
    lbfgs_curriculum,
    prox_l1_box,
    solve,
)
from nefi.domain import Domain
from nefi.fields import GridField, Heads, Identity, Softplus
from nefi.instances.toy1d import make_problem
from nefi.metrics import psnr
from nefi.registry import build, get
from nefi.solve import Curriculum, Result, Stage


@pytest.fixture(scope="module")
def toy():
    return make_problem(n=32, seed=0, hidden=16, depth=2, n_octaves=3, steps=(20, 30))


# ---- Gaussian splats ------------------------------------------------------------------------
def test_gaussian_splat_renders_single_known_primitive():
    f = GaussianSplatField(2, {"x": "identity"}, n_primitives=4, max_primitives=8, render="point")
    mu, sig, amp = torch.tensor([[0.1, -0.2]]), torch.tensor([[0.2, 0.1]]), torch.tensor([[1.5]])
    f.set_primitives(mu, sig, amp)
    assert f.n_active == 1
    for shape in ((32, 32), (17, 24)):
        c = Domain.unit(shape).coords()
        expected = 1.5 * torch.exp(
            -0.5 * (((c[..., 0] - 0.1) / 0.2) ** 2 + ((c[..., 1] + 0.2) / 0.1) ** 2)
        )
        out = f(c)["x"]
        assert out.shape == shape
        assert torch.allclose(out, expected, atol=1e-5)
    # arbitrary (non-grid) query points take the dense path
    pts = torch.rand(50, 2) * 2 - 1
    v = f(pts)["x"]
    ref = 1.5 * torch.exp(-0.5 * (((pts[:, 0] - 0.1) / 0.2) ** 2 + ((pts[:, 1] + 0.2) / 0.1) ** 2))
    assert torch.allclose(v, ref, atol=1e-5)
    # area rendering conserves the mass a (2π) σx σy (normalized cell area (2/n)²)
    f.render = "area"
    for n in (16, 64):
        with torch.no_grad():
            out = f(Domain.unit((n, n)).coords())["x"]
        mass = float(out.sum()) * (2.0 / n) ** 2
        assert abs(mass - 1.5 * 2 * math.pi * 0.2 * 0.1) < 1e-4


def test_gaussian_splat_nd_shapes_background_and_grad():
    f1 = GaussianSplatField(1, {"x": Softplus(init_value=0.1)}, n_primitives=8)
    y = f1(Domain.unit((40,)).coords())["x"]
    assert y.shape == (40,) and (y > 0).all()
    f3 = GaussianSplatField(3, {"x": "identity"}, n_primitives=27, max_primitives=40)
    y3 = f3(Domain.unit((8, 6, 4)).coords())["x"]
    assert y3.shape == (8, 6, 4) and torch.isfinite(y3).all()
    y3.sum().backward()
    assert f3._mu.grad is not None and f3._mu.grad[:27].abs().sum() > 0
    assert torch.all(f3._mu.grad[27:] == 0)  # inactive slots receive no gradient
    # "signed" amplitudes and explicit background
    fs = GaussianSplatField(2, {"x": "identity"}, amplitude="signed", background=0.5)
    fs.set_primitives(torch.zeros(1, 2), torch.full((1, 2), 0.3), torch.tensor([[-0.4]]))
    out = fs(Domain.unit((9, 9)).coords())["x"].detach()
    assert float(out.min()) < 0.2 and abs(float(out[0, 0]) - 0.5) < 0.05


def test_splat_control_prunes_near_zero_and_respects_cap():
    torch.manual_seed(0)
    f = GaussianSplatField(2, {"x": "identity"}, n_primitives=16, max_primitives=20)
    with torch.no_grad():
        amps = f.amplitudes.clone()[:16]
        amps[:4] = 1e-6  # four near-zero primitives
        f.write(torch.arange(4), f.means[:4], f.sigmas[:4], amps[:4])
    ctrl = SplatControl(every=1, start=0, prune_rel=0.01, grad_quantile=0.0)
    ctrl.field = f
    info = ctrl.apply(f)
    assert info["pruned"] == 4 and f.n_active == 12
    # strong positional gradients everywhere -> densification, but never beyond the cap
    for _ in range(5):
        g = torch.randn(f.capacity, 2)
        ctrl.accumulate(g)
        info = ctrl.apply(f)
        assert f.n_active <= f.capacity
    assert f.n_active == f.capacity
    assert info["split"] + info["cloned"] == 0  # full: nothing more can be added
    # merge collapses coincident primitives
    f2 = GaussianSplatField(2, {"x": "identity"}, n_primitives=2, max_primitives=4)
    f2.set_primitives(
        torch.tensor([[0.0, 0.0], [0.001, 0.0]]), torch.full((2, 2), 0.1), torch.ones(2, 1)
    )
    c2 = SplatControl(merge_cells=1.0)
    c2.cell = 2.0 / 32
    assert c2.apply(f2)["merged"] == 1 and f2.n_active == 1


def test_gaussian_splat_baseline_with_density_control_solves(toy):
    problem, gt, _ = toy
    bp, cur = baseline_problem(problem, "gaussian_splat", n_primitives=8, max_primitives=12)
    assert isinstance(bp.field, GaussianSplatField)
    assert isinstance(bp.field.heads["x"], Identity)  # softplus -> identity (density >= 0)
    ctrl = bp.meta["callbacks"][0]
    assert isinstance(ctrl, SplatControl)
    res = solve(bp, cur, device="cpu")
    assert res.fields["x"].shape == (32,)
    assert len(ctrl.events) >= 1 and bp.field.n_active <= 12
    assert res.history["total"][-1] < res.history["total"][0]


# ---- Deep decoder -----------------------------------------------------------------------------
def test_deep_decoder_output_follows_coordinate_grid():
    f = DeepDecoderField((32, 32), {"x": Softplus(init_value=0.1)}, n_stages=4, width=16)
    assert f.sizes[-1] == (32, 32) and f.latent_shape == (4, 4)
    for shape in ((32, 32), (16, 16), (48, 40)):
        out = f(Domain.unit(shape).coords())["x"]
        assert out.shape == shape and (out > 0).all()
    out = f(Domain.unit((32, 32)).coords())["x"]
    assert abs(float(out.detach().mean()) - 0.1) < 0.05  # near the head's init value
    out.mean().backward()
    assert all(p.grad is not None for p in f.parameters())
    assert f.latent.requires_grad is False  # fixed random input (buffer)
    f1 = DeepDecoderField((24,), {"x": "identity"}, n_stages=3, width=8)
    assert f1(Domain.unit((24,)).coords())["x"].shape == (24,)
    f3 = DeepDecoderField((8, 8, 4), {"x": "identity"}, n_stages=2, width=8)
    assert f3(Domain.unit((8, 8, 4)).coords())["x"].shape == (8, 8, 4)
    # reset keeps the latent but re-draws the convolutions
    lat = f.latent.clone()
    w = f.convs[0].weight.detach().clone()
    f.reset_parameters()
    assert torch.equal(lat, f.latent) and not torch.equal(w, f.convs[0].weight)


# ---- ADMM -----------------------------------------------------------------------------------
def test_prox_l1_box():
    v = torch.tensor([-2.0, -0.5, 0.2, 1.0, 3.0])
    assert torch.allclose(prox_l1_box(v, 0.5, None, None), torch.tensor([-1.5, 0.0, 0.0, 0.5, 2.5]))
    assert torch.allclose(prox_l1_box(v, 0.5, 0.0, 2.0), torch.tensor([0.0, 0.0, 0.0, 0.5, 2.0]))


def test_admm_decreases_objective_and_returns_result(toy):
    problem, gt, meas = toy
    bp, cur = baseline_problem(problem, "admm")
    assert isinstance(bp.field, GridField) and isinstance(bp.field.heads["x"], Identity)
    cfg = ADMMConfig(mu=0.1, l1=1e-3, lr=2e-2, n_inner=10, max_outer=15, lo=0.0, adaptive_mu=True)
    res = ADMMSolver(bp, cfg, device="cpu").run()
    assert isinstance(res, Result)
    h = res.history
    n_outer = res.extra["n_outer"]
    assert len(h["primal_residual"]) == n_outer == len(h["objective"])
    assert len(h["total"]) == len(h["data_loss"]) == n_outer * cfg.n_inner
    assert h["objective"][-1] < h["objective"][0]
    assert res.final("data_loss") is not None and res.final("primal_residual") >= 0
    assert res.fields["x"].shape == (32,) and float(res.fields["x"].min()) >= 0.0  # box
    assert res.pred.shape == meas.data.shape
    assert psnr(res.fields["x"], gt["x"]) > 20.0  # observed ≈ 29 dB
    # config from dict / from problem meta, resolution from a curriculum
    res2 = ADMMSolver(bp, {"max_outer": 2, "n_inner": 3}, device="cpu").run()
    assert res2.extra["n_outer"] == 2
    res3 = ADMMSolver(
        bp,
        ADMMConfig(max_outer=50, n_inner=2),
        curriculum=Curriculum([Stage(shape=(16,), steps=6)]),  # final shape + step budget
        device="cpu",
    ).run()
    assert res3.fields["x"].shape == (16,) and res3.extra["n_outer"] == 3
    assert get("baseline", "admm") is ADMMSolver


def test_admm_through_dispatch_uses_head_range_and_budget(toy):
    problem, _, _ = toy
    bp, cur = baseline_problem(problem, "admm", admm={"max_outer": 4, "n_inner": 5})
    assert bp.meta["solver"] == "admm" and bp.meta["solver_config"].lo == 0.0
    assert cur.total_steps == 20
    res = solve(bp, cur, device="cpu")
    assert res.extra["method"] == "admm" and len(res.history["primal_residual"]) == 4
    res = solve(bp, cur.scaled(0.5), device="cpu")  # benchmark budget scaling reaches ADMM
    assert len(res.history["primal_residual"]) == 2


# ---- L-BFGS / direct / baseline_problem ---------------------------------------------------------
def test_lbfgs_curriculum_runs(toy):
    problem, _, _ = toy
    cur = lbfgs_curriculum((32,), steps=8, history=10, max_iter=5)
    assert cur.optim.optimizer == "lbfgs" and cur.optim.lbfgs_history == 10
    assert cur.stages[0].lr_schedule == "constant" and cur.optim.grad_clip is None
    bp, _ = baseline_problem(problem, "grid")
    res = nefi.invert(bp, cur, device="cpu")
    assert len(res.history["total"]) == 8
    assert res.history["total"][-1] < res.history["total"][0]
    ms = lbfgs_curriculum((64, 64), steps=(3, 5), n_stages=2)
    assert [s.shape for s in ms.stages] == [(32, 32), (64, 64)]
    bl, cl = baseline_problem(problem, "lbfgs", steps=4)
    assert cl.optim.optimizer == "lbfgs" and cl.total_steps == 4


def test_direct_solver_packages_closed_form(toy):
    problem, gt, meas = toy
    field = GridField((32,), Heads({"x": Identity()}))
    bp = nefi.InverseProblem(
        problem.domain,
        field,
        problem.operator,
        problem.losses,
        meas,
        meta={"solver": "direct", "reconstruct": lambda obs, dom: obs.data.clamp_min(0)},
    )
    res = solve(bp, device="cpu")
    assert isinstance(res, Result) and res.stage_results[0]["stop"] == "closed_form"
    assert torch.allclose(res.fields["x"], meas.data.clamp_min(0))
    assert len(res.history["data_loss"]) == 1
    assert isinstance(DirectSolver(bp).run(), Result)


@pytest.mark.parametrize("kind", BASELINE_KINDS)
def test_baseline_problem_swaps_fields_and_solves_toy1d(toy, kind):
    problem, gt, meas = toy
    kw = {"width": 8, "n_stages": 3} if kind == "deep_decoder" else {}
    if kind == "admm":
        kw = {"admm": {"max_outer": 3, "n_inner": 5}}
    if kind == "lbfgs":
        kw = {"steps": 5}
    bp, cur = baseline_problem(problem, kind, **kw)
    assert bp.field is not problem.field
    assert bp.operator is problem.operator and bp.losses is problem.losses
    assert bp.measurement is meas and bp.domain == problem.domain
    assert bp.meta["baseline"] == kind and bp.name.endswith(kind)
    res = solve(bp, cur, device="cpu")
    assert res.fields["x"].shape == (32,) and torch.isfinite(res.fields["x"]).all()
    assert res.final("total") is not None


def test_baseline_learning_rates_and_registry(toy):
    problem, _, _ = toy
    _, cur = baseline_problem(problem, "grid")
    assert math.isclose(cur.stages[0].lr, problem.curriculum.stages[0].lr * 10)
    _, cur = baseline_problem(problem, "grid", lr=0.05, steps=100)
    assert math.isclose(cur.stages[0].lr, 0.05) and cur.total_steps == 100
    f = build("field", {"type": "gaussian_splat", "ndim": 2, "heads": {"x": "identity"}})
    assert isinstance(f, GaussianSplatField)
    d = build("field", {"type": "deep_decoder", "shape": [16, 16], "width": 8, "n_stages": 2})
    assert isinstance(d, DeepDecoderField)
    with pytest.raises(nefi.NefiError):
        baseline_problem(problem, "nope")
