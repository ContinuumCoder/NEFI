"""Validation of the Gray–Scott reaction–diffusion operator (nefi.physics.reaction_diffusion)."""

import math

import pytest
import torch

from nefi.domain import Domain
from nefi.physics.reaction_diffusion import (
    GrayScottIC,
    ReactionDiffusionOperator,
    diffusion_term,
    stable_dt,
)

F64 = torch.float64
DOM = Domain((32, 32), ((0.0, 0.64), (0.0, 0.64)))
TIMES = (25.0, 50.0, 75.0, 100.0)


def _smooth_F(dom: Domain = DOM) -> torch.Tensor:
    x = dom.physical_coords(dtype=F64) / 0.64
    return 0.045 + 0.012 * torch.sin(2 * math.pi * x[..., 0]) * torch.cos(2 * math.pi * x[..., 1])


def test_diffusion_term_conservation_and_constant_coefficient():
    torch.manual_seed(0)
    u = torch.rand(2, 16, 12, dtype=F64)
    sp = (0.1, 0.2)
    for bc in ("neumann", "periodic"):
        lap = diffusion_term(u, sp, bc)
        assert lap.shape == u.shape
        assert float(lap.sum(dim=(-1, -2)).abs().max()) < 1e-10  # zero net flux
        d = torch.full((16, 12), 3.0, dtype=F64)
        assert torch.allclose(diffusion_term(u, sp, bc, coeff=d), 3.0 * lap, atol=1e-10)
    # interior agrees with the classic 5-point stencil
    x = u[0]
    ref = (x[2:, 1:-1] - 2 * x[1:-1, 1:-1] + x[:-2, 1:-1]) / 0.01
    ref = ref + (x[1:-1, 2:] - 2 * x[1:-1, 1:-1] + x[1:-1, :-2]) / 0.04
    assert torch.allclose(diffusion_term(x, sp)[1:-1, 1:-1], ref, atol=1e-10)


def test_stability_positivity_and_mass_conservation():
    op = ReactionDiffusionOperator(DOM, TIMES)
    assert op.dt_sim <= stable_dt(2e-5, DOM.spacing(), 0.1, 0.06) + 1e-12
    y = op({"F": _smooth_F()})
    assert y.shape == (8, 32, 32) == op.output_shape((32, 32))
    assert 0.0 <= float(y.min()) and float(y.max()) <= 1.0
    long = ReactionDiffusionOperator(DOM, (500.0, 1000.0, 2000.0))
    yl = long({"F": _smooth_F()})
    assert float(yl.min()) >= -1e-9 and float(yl.max()) <= 1.0 + 1e-9  # stays in [0, 1]
    # with F = k = 0 the reactions only exchange u <-> v: Σ(u + v) is conserved (Neumann)
    op0 = ReactionDiffusionOperator(DOM, (50.0, 100.0), k=0.0)
    _, mass = op0.simulate(
        torch.zeros(32, 32, dtype=F64), observe=lambda s, n: (s[0] + s[1]).sum(), record=[0, 100]
    )
    assert abs(float(mass[1] / mass[0]) - 1.0) < 1e-12
    # a finer grid needs a smaller explicit step (dt = dt_base / m)
    fine = op.at_resolution((128, 128))
    assert fine.m > 1 and fine.n_steps == 100 * fine.m and fine.record[-1] == fine.n_steps


def test_substepped_generator_agrees_with_inversion_stepper():
    F = _smooth_F()
    y1 = ReactionDiffusionOperator(DOM, TIMES)({"F": F})
    y4 = ReactionDiffusionOperator(DOM, TIMES, substeps=4)({"F": F})
    rel = float((y4 - y1).norm() / y4.norm())
    assert 1e-4 < rel < 0.03, rel  # a small but genuine O(dt) discretization gap


def test_checkpoint_gradients_equal_autograd_and_finite_differences():
    torch.manual_seed(0)
    F = _smooth_F()
    w = torch.randn(8, 32, 32, dtype=F64)
    grads = []
    for mode, ck in (("autograd", None), ("checkpoint", 7), ("checkpoint", None)):
        op = ReactionDiffusionOperator(DOM, TIMES, grad_mode=mode, checkpoint_every=ck)
        x = F.clone().requires_grad_(True)
        (g,) = torch.autograd.grad((op({"F": x}) * w).sum(), x)
        grads.append(g)
    for g in grads[1:]:
        assert float((g - grads[0]).norm() / grads[0].norm()) < 1e-6
    op = ReactionDiffusionOperator(DOM, TIMES, grad_mode="autograd")
    e = torch.randn(32, 32, dtype=F64)
    eps = 1e-7
    fd = float((op({"F": F + eps * e}) * w).sum()) - float((op({"F": F - eps * e}) * w).sum())
    fd /= 2 * eps
    assert abs(fd - float((grads[0] * e).sum())) < 1e-5 * max(1.0, abs(fd))


@pytest.mark.parametrize("unknown", ["k", "Du"])
def test_other_unknowns_boundaries_and_coarse_stage(unknown):
    value = {"k": 0.06, "Du": 2e-5}[unknown]
    op = ReactionDiffusionOperator(
        DOM,
        (25.0, 50.0),
        field="p",
        unknown=unknown,
        observe=("u",),
        boundary="periodic",
        param_max=2 * value,
        ic=GrayScottIC(modes=(1, 2)),
    )
    p = torch.full((32, 32), value, dtype=F64, requires_grad=True)
    y = op({"p": p})
    assert y.shape == (2, 32, 32)
    y.sum().backward()
    assert p.grad is not None and float(p.grad.abs().max()) > 0
    coarse = op.at_resolution((16, 16))
    assert coarse({"p": torch.full((16, 16), value, dtype=F64)}).shape == (2, 16, 16)
    with pytest.raises(Exception, match="param_max"):
        op({"p": torch.full((32, 32), 3 * value, dtype=F64)})
