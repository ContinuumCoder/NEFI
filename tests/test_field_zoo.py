"""Representation zoo: hash grid, low-rank CP, level set, parametric, symmetric, head modifiers."""

import math

import pytest
import torch

from nefi.domain import Domain
from nefi.errors import ConfigError
from nefi.fields import (
    Affine,
    ExpHead,
    GridField,
    HashGridEncoding,
    HashGridField,
    Heads,
    LevelSetField,
    LevelSetHead,
    LowRankField,
    MaskedHead,
    MassNormalized,
    NeuralField,
    ParametricField,
    ScaledHead,
    Softplus,
    SymmetricField,
    ZeroMean,
    ellipses,
    gaussian_blobs,
    interface_length,
)


# --------------------------------------------------------------------------------------------
# hash grid
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("d", [1, 2, 3])
def test_hashgrid_shapes_dense_and_hashed_levels(d):
    enc = HashGridEncoding(d, n_levels=6, n_features=2, log2_table_size=9, max_resolution=64)
    assert any(enc.dense) and (d == 1 or not all(enc.dense))  # coarse dense, fine hashed
    assert enc.resolutions[0] == 4 and enc.resolutions[-1] == 64
    x = torch.rand(7, 5, d) * 2 - 1
    out = enc(x, progress=1.0)
    assert out.shape == (7, 5, d + 6 * 2) and torch.isfinite(out).all()
    assert torch.equal(out[..., :d], x)  # include_input
    # gradients reach the table
    enc(x).square().sum().backward()
    assert enc.table.grad is not None and enc.table.grad.abs().sum() > 0


def test_hashgrid_level_annealing_coarse_to_fine():
    enc = HashGridEncoding(2, n_levels=8, log2_table_size=10, min_active_levels=2)
    w0, wh, w1 = enc.level_weights(0.0), enc.level_weights(0.5), enc.level_weights(1.0)
    assert torch.equal(w0, torch.tensor([1.0, 1, 0, 0, 0, 0, 0, 0]))
    assert torch.all(w1 == 1.0)
    assert torch.all(wh[:-1] >= wh[1:])  # monotone: coarse levels first
    assert torch.all(wh >= w0) and torch.all(w1 >= wh)
    with torch.no_grad():
        enc.table.normal_()
    x = torch.rand(10, 2) * 2 - 1
    f0 = enc(x, 0.0)[:, 2:].reshape(10, 8, 2)
    assert torch.all(f0[:, 2:] == 0) and torch.any(f0[:, :2] != 0)  # gated levels contribute 0


def test_hashgrid_interpolation_is_continuous_and_exact_at_vertices():
    enc = HashGridEncoding(
        1,
        n_levels=1,
        n_features=1,
        base_resolution=4,
        max_resolution=4,
        include_input=False,
        annealed=False,
    )
    with torch.no_grad():
        enc.table.copy_(torch.arange(5.0).view(5, 1))  # vertex values 0..4 at u = 0, .25, ..., 1
    u = torch.tensor([0.0, 0.25, 0.375, 1.0])
    y = enc((2 * u - 1).view(-1, 1))[:, 0]
    assert torch.allclose(y, torch.tensor([0.0, 1.0, 1.5, 4.0]), atol=1e-6)


def test_hashgrid_field_in_neural_field_and_reset():
    heads = Heads({"x": Softplus(init_value=0.5)})
    enc = HashGridEncoding(2, n_levels=4, log2_table_size=10, max_resolution=32)
    f = NeuralField(2, heads, encoding=enc, hidden=16, depth=2)
    y = f(Domain.unit((16, 16)).coords())["x"]
    assert y.shape == (16, 16) and abs(float(y.detach().mean()) - 0.5) < 0.2
    hf = HashGridField(
        2, Heads({"x": "identity"}), n_levels=4, log2_table_size=10, max_resolution=32, hidden=16
    )
    t0 = hf.encoding.table.detach().clone()
    torch.manual_seed(5)
    hf.reset_parameters()
    assert not torch.equal(t0, hf.encoding.table)


# --------------------------------------------------------------------------------------------
# low rank
# --------------------------------------------------------------------------------------------
def test_lowrank_rank1_exact_outer_product_and_any_resolution():
    f = LowRankField((8, 6), Heads({"x": "identity"}), rank=1, bias=False)
    u, v = torch.arange(8.0), torch.linspace(0.0, 1.0, 6)
    with torch.no_grad():
        f.factors[0].copy_(u.view(1, 1, 8))
        f.factors[1].copy_(v.view(1, 1, 6))
    x = f(Domain.unit((8, 6)).coords())["x"]
    assert torch.allclose(x, torch.outer(u, v), atol=1e-6)
    assert f(Domain.unit((16, 12)).coords())["x"].shape == (16, 12)
    # scattered points agree with the grid evaluation at grid points
    c = Domain.unit((8, 6)).coords()
    pts = c.reshape(-1, 2)[[0, 5, 17, 47]]
    assert torch.allclose(f(pts)["x"], x.reshape(-1)[[0, 5, 17, 47]], atol=1e-5)
    assert f.compression_ratio() == pytest.approx(48 / 14)


def test_lowrank_rank1_fit_is_exact_and_stage_resampling():
    target = torch.outer(torch.linspace(1.0, 2.0, 12), torch.cos(torch.linspace(0, 3, 10)))
    f = LowRankField((12, 10), Heads({"x": "identity"}), rank=1, bias=False)
    coords = Domain.unit((12, 10)).coords()
    opt = torch.optim.LBFGS(f.parameters(), lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = ((f(coords)["x"] - target) ** 2).mean()
        loss.backward()
        return loss

    for _ in range(3):
        opt.step(closure)
    with torch.no_grad():
        assert float(((f(coords)["x"] - target) ** 2).mean()) < 1e-10
    f.on_stage_start(None, Domain.unit((24, 20)))
    assert tuple(f.factors[0].shape) == (1, 1, 24) and tuple(f.factors[1].shape) == (1, 1, 20)


# --------------------------------------------------------------------------------------------
# level set
# --------------------------------------------------------------------------------------------
def test_levelset_sharpening_monotone_in_progress():
    head = LevelSetHead(1.0, 3.0, eps_start=1.0, eps_end=0.01)
    phi = torch.tensor([[-0.3], [-0.05], [0.05], [0.3]])
    eps = [head.eps(p) for p in torch.linspace(0, 1, 11).tolist()]
    assert all(a > b for a, b in zip(eps, eps[1:])) and eps[-1] == pytest.approx(0.01)
    dist = [float((head(phi, progress=p) - 2.0).abs().min()) for p in torch.linspace(0, 1, 11)]
    assert all(b >= a - 1e-7 for a, b in zip(dist, dist[1:]))  # distance to midpoint grows
    x1 = head(phi, progress=1.0)
    assert torch.allclose(x1[[0, 3]], torch.tensor([1.0, 3.0]), atol=1e-6)  # binary at the end
    assert torch.all((x1 > 1.0 - 1e-6) & (x1 < 3.0 + 1e-6))
    assert torch.allclose(
        head(head.inverse(torch.tensor([1.5, 2.5])), progress=1.0),
        torch.tensor([1.5, 2.5]),
        atol=1e-4,
    )
    with pytest.raises(ConfigError):
        LevelSetHead(1.0, 0.0)


def test_levelset_field_and_interface_length():
    f = LevelSetField(2, lo=0.0, hi=1.0, hidden=16, depth=2, n_octaves=3)
    c = Domain.unit((16, 16)).coords()
    assert f(c, 0.5)["x"].shape == (16, 16) and f.phase_mask(c).dtype == torch.bool
    d = Domain.unit((128, 128))
    r = (d.physical_coords() - 0.5).norm(dim=-1)
    disk = torch.sigmoid((0.3 - r) / 0.01)  # anti-aliased disk of radius 0.3
    per = float(interface_length(disk, d.spacing()))
    assert abs(per - 2 * math.pi * 0.3) / (2 * math.pi * 0.3) < 0.05
    assert float(interface_length(torch.ones(16, 16), (1 / 16, 1 / 16))) < 1e-5


# --------------------------------------------------------------------------------------------
# parametric
# --------------------------------------------------------------------------------------------
def test_parametric_gaussian_field_reproduces_a_known_blob():
    dom = Domain.unit((32, 32))
    c = dom.coords()
    target = 1.7 * torch.exp(-((c - torch.tensor([0.2, -0.3])) ** 2).sum(-1) / (2 * 0.25**2))
    fn = gaussian_blobs(1, ndim=2, centers=[[0.0, 0.0]], sigmas=0.4)
    f = ParametricField(fn)
    assert f.params.numel() == 4
    opt = torch.optim.Adam(f.parameters(), lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        loss = ((f(c)["x"] - target) ** 2).mean()
        loss.backward()
        opt.step()
    p = f.unpack()
    assert abs(float(p["amplitude"][0]) - 1.7) < 0.02
    assert torch.allclose(p["center"][0], torch.tensor([0.2, -0.3]), atol=0.01)
    assert abs(float(p["sigma"][0]) - 0.25) < 0.01
    f.reset_parameters()
    assert torch.allclose(f.params, fn.init())


def test_parametric_ellipses_and_plain_callable():
    f = ParametricField(ellipses(2, ndim=2, sharpness=0.02))
    with torch.no_grad():
        x = f(Domain.unit((24, 24)).coords())["x"]
    assert x.shape == (24, 24) and float(x.max()) > 0.9 and float(x.min()) < 0.05
    g = ParametricField(lambda p, c: p[0] * c[..., 0], init=torch.tensor([2.0]))
    assert torch.allclose(
        g(Domain.unit((4,)).coords())["x"], 2.0 * Domain.unit((4,)).coords()[..., 0]
    )
    with pytest.raises(ConfigError):
        ParametricField(lambda p, c: c[..., 0])  # plain callable without init


# --------------------------------------------------------------------------------------------
# symmetric
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind,check", [("mirror_x", "x"), ("mirror_y", "y"), ("mirror_xy", "xy")])
def test_symmetric_field_mirror_outputs_are_symmetric(kind, check):
    inner = NeuralField(2, Heads({"x": "identity"}), hidden=16, depth=2, n_octaves=4)
    f = SymmetricField(inner, kind)
    x = f(Domain.unit((10, 12)).coords())["x"]
    if "x" in check:
        assert torch.allclose(x, x.flip(0), atol=1e-6)
    if "y" in check:
        assert torch.allclose(x, x.flip(1), atol=1e-6)
    if check == "x":
        assert not torch.allclose(x, x.flip(1), atol=1e-4)


def test_symmetric_field_radial_and_grid_average():
    f = SymmetricField(
        NeuralField(1, Heads({"x": "identity"}), hidden=16, depth=2, n_octaves=3), "radial", ndim=2
    )
    x = f(Domain.unit((16, 16)).coords())["x"]
    assert torch.allclose(x, x.flip(0), atol=1e-6) and torch.allclose(x, x.T, atol=1e-6)
    assert f.inner_ndim == 1
    g = GridField((8, 8), Heads({"x": "identity"}), init=0.0)
    with torch.no_grad():
        g.param.normal_()
    sg = SymmetricField(g, "mirror_x")
    assert sg.mode == "average"
    y = sg(Domain.unit((8, 8)).coords())["x"]
    assert torch.allclose(y, y.flip(0))
    with pytest.raises(ConfigError):
        SymmetricField(g, "radial", ndim=2)
    with pytest.raises(ConfigError):
        SymmetricField(g, "mirror_z")  # axis 2 does not exist in 2-D


# --------------------------------------------------------------------------------------------
# head modifiers
# --------------------------------------------------------------------------------------------
def test_head_modifiers():
    mask = torch.zeros(8, 8)
    mask[2:6, 2:6] = 1
    raw = torch.randn(8, 8, 1)
    m = MaskedHead(Softplus(), mask)(raw)
    assert torch.all(m[mask == 0] == 0) and torch.all(m[mask == 1] > 0)
    assert MaskedHead(Softplus(), mask)(torch.randn(16, 16, 1)).shape == (16, 16)  # resampled
    mn = MassNormalized(Softplus(), total=2.0, volume=0.5)(raw)
    assert abs(float(mn.mean()) * 0.5 - 2.0) < 1e-5
    s = ScaledHead(Softplus(init_value=1.0), scale=1e-3)
    assert torch.allclose(s(torch.full((3, 1), s.init_bias()[0])), torch.full((3,), 1e-3))
    a = Affine(scale=100.0, offset=1500.0, init_value=1600.0)
    assert a.init_bias() == [1.0] and float(a(torch.ones(1, 1))) == 1600.0
    e = ExpHead(init_value=0.25)
    assert torch.allclose(e(torch.full((1, 1), e.init_bias()[0])), torch.tensor([0.25]))


def test_zero_mean_head_fixes_the_gauge():
    from nefi.fields import Bounded, Identity
    from nefi.registry import build

    x = ZeroMean(Bounded(-1.0, 1.0))(torch.randn(6, 7, 1))
    assert x.shape == (6, 7) and abs(float(x.mean())) < 1e-6
    # n_dims: only the trailing (spatial) dims are averaged, a leading batch axis is kept apart
    xb = ZeroMean(Identity(), n_dims=2)(
        torch.randn(3, 6, 7, 1) + torch.arange(3.0)[:, None, None, None]
    )
    assert torch.allclose(xb.mean(dim=(-2, -1)), torch.zeros(3), atol=1e-6)
    # a constant offset of the inner field is invisible after the gauge
    raw = torch.randn(5, 5, 1)
    assert torch.allclose(ZeroMean(Identity())(raw), ZeroMean(Identity())(raw + 3.0), atol=1e-5)
    # warm starts (inner inverse) reproduce the zero-mean part of the target
    field = GridField((8, 8), Heads({"phase": ZeroMean(Bounded(-3.0, 3.0, init_value=0.0))}))
    v = torch.linspace(-1.0, 1.0, 64).reshape(8, 8) + 0.5
    field.set_fields({"phase": v})
    out = field(Domain((8, 8), ((0.0, 1.0), (0.0, 1.0))).coords())["phase"].detach()
    assert torch.allclose(out, v - v.mean(), atol=1e-4)
    assert isinstance(build("head", {"type": "zero_mean", "inner": "identity"}), ZeroMean)
    with pytest.raises(ConfigError):
        ZeroMean(Identity(), n_dims=0)
