"""Physics-knowledge losses are zero on fields that satisfy the knowledge, positive otherwise."""

import logging

import pytest
import torch

from nefi.domain import Domain
from nefi.errors import ConfigError
from nefi.losses import (
    Conservation,
    Context,
    GradientL2,
    KnownSupportLoss,
    Monotone,
    PDEResidual,
    RangePenalty,
    RangeStat,
    SymmetryLoss,
)
from nefi.losses.physics import gradient, laplacian
from nefi.measurement import Measurement
from nefi.operators import Identity, LambdaOperator


def ctx_for(fields: dict, dom: Domain, operator=None) -> Context:
    first = next(iter(fields.values()))
    return Context(dict(fields), first, Measurement(first.detach()), dom, operator=operator)


def test_conservation_sum_and_mean():
    dom = Domain.unit((16, 8))
    x = torch.full((16, 8), 2.5)  # integral over the unit square = 2.5
    c = ctx_for({"rho": x}, dom)
    assert float(Conservation("rho", total=2.5)(c)) < 1e-12
    assert float(Conservation("rho", total=2.5, kind="mean")(c)) < 1e-12
    assert float(Conservation("rho", total=2.0)(c)) == pytest.approx((0.5 / 2.0) ** 2)
    assert float(Conservation("rho", total=0.0, relative=True)(c)) == pytest.approx(6.25)
    # resolution independent (integral, not raw sum)
    c2 = ctx_for({"rho": torch.full((32, 16), 2.5)}, dom.at((32, 16)))
    assert float(Conservation("rho", total=2.5)(c2)) < 1e-12
    with pytest.raises(ConfigError):
        Conservation("rho", kind="max")


@pytest.mark.parametrize("kind", ["mirror_x", "mirror_y", "mirror_xy", "radial"])
def test_symmetry_loss_zero_iff_symmetric(kind):
    dom = Domain.unit((16, 16))
    c = dom.coords()
    r = c.norm(dim=-1)
    sym = torch.exp(-4 * r**2)  # symmetric under every kind
    assert float(SymmetryLoss("x", kind)(ctx_for({"x": sym}, dom))) < 1e-10
    asym = sym + 0.3 * (c[..., 0] + 0.5 * c[..., 1])  # breaks all of them
    assert float(SymmetryLoss("x", kind)(ctx_for({"x": asym}, dom))) > 1e-4
    only_x = torch.exp(-4 * c[..., 0] ** 2) * (1 + c[..., 1])  # mirror in x only
    v = float(SymmetryLoss("x", kind)(ctx_for({"x": only_x}, dom)))
    assert (v < 1e-10) == (kind == "mirror_x")


def test_known_support_loss():
    dom = Domain.unit((8, 8))
    mask = torch.zeros(8, 8)
    mask[2:6, 2:6] = 1
    inside = torch.rand(8, 8) * mask
    assert float(KnownSupportLoss("x", mask)(ctx_for({"x": inside}, dom))) == 0.0
    leak = inside + 0.1 * (1 - mask)
    assert float(KnownSupportLoss("x", mask)(ctx_for({"x": leak}, dom))) == pytest.approx(0.01)
    # mask resampled to another resolution
    assert (
        float(KnownSupportLoss("x", mask)(ctx_for({"x": torch.zeros(16, 16)}, dom.at((16, 16)))))
        == 0.0
    )
    with pytest.raises(ConfigError):
        KnownSupportLoss("x")


def test_monotone_and_gradient_l2():
    dom = Domain.unit((10, 6))
    c = dom.physical_coords()
    inc = c[..., 0] ** 2 + c[..., 1]
    assert float(Monotone("x", axis=0)(ctx_for({"x": inc}, dom))) == 0.0
    assert float(Monotone("x", axis=1, direction="increasing")(ctx_for({"x": inc}, dom))) == 0.0
    assert float(Monotone("x", axis=0, direction="decreasing")(ctx_for({"x": inc}, dom))) > 0
    assert float(GradientL2("x")(ctx_for({"x": torch.ones(10, 6)}, dom))) == 0.0
    assert float(GradientL2("x")(ctx_for({"x": inc}, dom))) > 0
    with pytest.raises(ConfigError):
        Monotone("x", direction="sideways")


def test_range_stat_is_range_penalty():
    assert RangeStat is RangePenalty
    dom = Domain.unit((4,))
    assert float(RangeStat(0.0, 1.0, "x")(ctx_for({"x": torch.full((4,), 0.5)}, dom))) == 0.0
    assert float(RangeStat(0.0, 1.0, "x")(ctx_for({"x": torch.full((4,), 2.0)}, dom))) == 1.0


def test_pde_residual_zero_on_solution_and_fd_helpers():
    dom = Domain.unit((32,))
    t = dom.physical_coords()[..., 0]
    h = dom.spacing()

    def res(fields, ctx):  # u' = 1
        return gradient(fields["u"], ctx.domain.spacing(fields["u"].shape), axis=0) - 1.0

    loss = PDEResidual(res, fields=("u",))
    op = Identity("u")
    assert float(loss(ctx_for({"u": 2.0 + t}, dom, op))) < 1e-10
    assert float(loss(ctx_for({"u": t**2}, dom, op))) > 1e-2
    # discrete Laplacian of cos(2πx) on a periodic grid ≈ -(2π)² cos(2πx)
    u = torch.cos(2 * torch.pi * t)
    lap = laplacian(u, h, periodic_axes=(0,))
    assert torch.allclose(lap, -((2 * torch.pi) ** 2) * u, rtol=0, atol=1.0)


def test_pde_residual_warns_about_decoupled_fields(caplog):
    dom = Domain.unit((16,))
    fields = {"u": torch.rand(16), "k": torch.rand(16)}
    op = LambdaOperator(lambda f: f["u"], primary="u")  # data only sees u
    loss = PDEResidual(lambda f, ctx: f["u"] * f["k"], fields=("u", "k"))
    with caplog.at_level(logging.WARNING, logger="nefi"):
        loss(ctx_for(fields, dom, op))
        loss(ctx_for(fields, dom, op))  # warns once
    msgs = [r.getMessage() for r in caplog.records if "PDEResidual" in r.getMessage()]
    assert len(msgs) == 1 and "'k'" in msgs[0] and "NeFTY" in msgs[0]
