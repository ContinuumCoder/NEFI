"""Geometric representations: composite, layered, shape, warp and spectral fields."""

import math

import pytest
import torch

import nefi
from nefi.domain import Domain
from nefi.errors import ConfigError, ShapeError
from nefi.fields import GatedSoftplus, GridField, Heads, NeuralField, Softplus
from nefi.fields.geometric import (
    AnomalyField,
    CompositeField,
    DeformableField,
    FourierBasisField,
    LayerCakeField,
    LayeredField,
    PolygonField,
    SensitivityWarp,
    SpectralPreconditionedField,
    StarShapeField,
    WarpedField,
    WarpRegularizer,
    contour_regularizer,
    cylindrical,
    depth_stretch,
    interface_length,
    log_depth,
    lowpass_gain,
    polar,
    progress_fn,
    sensitivity_warp,
    smooth_step,
)
from nefi.losses import MSE, LossSet
from nefi.losses.base import Context
from nefi.measurement import Measurement
from nefi.operators import FFTConvolution, gaussian_kernel_fn
from nefi.problem import InverseProblem
from nefi.registry import build
from nefi.solve import Curriculum, Stage

pytestmark = pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad")


def _small_neural(ndim=2, heads=None, **kw):
    kw = {"hidden": 16, "depth": 2, "n_octaves": 3, **kw}
    return NeuralField(ndim, heads or Heads({"x": "identity"}), **kw)


def _grads_nonzero(module):
    return any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in module.parameters())


# ------------------------------------------------------------------------------------------
# composite
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("combiner", ["sum", "product", "mean", "max"])
def test_composite_shapes_and_gradient_flow(combiner):
    dom = Domain.unit((12, 10))
    bg = FourierBasisField(2, n_modes=4, init_std=0.1)
    an = _small_neural(heads=Heads({"a": Softplus(init_value=0.5)}))
    f = CompositeField({"background": bg, "anomaly": an}, combiner)
    out = f(dom.coords(), progress=0.7)
    assert set(out) == {"x"} and out["x"].shape == (12, 10)
    out["x"].pow(2).mean().backward()
    if combiner == "max":  # the gradient reaches only the arg-max component
        assert _grads_nonzero(f.background) or _grads_nonzero(f.anomaly)
    else:
        assert _grads_nonzero(f.background) and _grads_nonzero(f.anomaly)
    with torch.no_grad():
        b = bg(dom.coords(), 0.7)["x"]
        a = an(dom.coords(), 0.7)["a"]
        ref = {"sum": b + a, "product": b * a, "mean": 0.5 * (b + a), "max": torch.maximum(b, a)}
        assert torch.allclose(f(dom.coords(), 0.7)["x"], ref[combiner], atol=1e-6)


def test_composite_blend_and_custom_combiner_and_validation():
    dom = Domain.unit((8, 8))
    base, ins = FourierBasisField(2, 2), FourierBasisField(2, 2)
    m = NeuralField(2, Heads({"m": {"type": "bounded", "lo": 0.0, "hi": 1.0}}), hidden=8, depth=1)
    f = CompositeField({"base": base, "ins": ins, "mask": m}, "blend")
    assert f(dom.coords())["x"].shape == (8, 8)
    with pytest.raises(ConfigError):
        CompositeField({"a": base, "b": ins}, "blend")  # blend needs 3 components
    g = CompositeField(
        {"a": FourierBasisField(2, 2), "b": FourierBasisField(2, 2)}, lambda ts: ts[0] - ts[1]
    )
    assert g(dom.coords())["x"].abs().max() < 1e-6
    with pytest.raises(ConfigError):
        CompositeField({"heads": FourierBasisField(2, 2)})  # reserved name
    with pytest.raises(ConfigError):
        CompositeField({"a": FourierBasisField(2, 2)}, progress_map={"zzz": "full"})


def test_composite_progress_and_weight_maps():
    dom = Domain.unit((8, 8))
    an = _small_neural(heads=Heads({"a": Softplus(init_value=0.3)}))
    f = CompositeField(
        {"background": FourierBasisField(2, 4), "anomaly": an},
        "sum",
        progress_map={"background": "full", "anomaly": ("delay", 0.5)},
        weight_map={"anomaly": ("window", 0.2, 0.6)},
    )
    assert f.component_progress("background", 0.0) == 1.0
    assert f.component_progress("anomaly", 0.25) == 0.0
    assert abs(f.component_progress("anomaly", 0.75) - 0.5) < 1e-12
    # weight 0 at progress 0.1: the anomaly contributes nothing and receives no gradient
    out = f(dom.coords(), progress=0.1)["x"]
    bg = f.background(dom.coords(), 1.0)["x"]
    assert torch.allclose(out, bg, atol=1e-6)
    out.sum().backward()
    assert not _grads_nonzero(f.anomaly)
    assert progress_fn(0.3)(0.9) == 0.3 and progress_fn("frozen")(0.9) == 0.0
    assert f.freeze_prefixes("anomaly") == ("anomaly.",)


def test_composite_freeze_with_solver_and_reset():
    dom = Domain.unit((16,))
    torch.manual_seed(0)
    x_true = torch.sin(3 * dom.coords()[..., 0]) + 1.0
    op = FFTConvolution(gaussian_kernel_fn(0.03), dom)
    meas = Measurement(op({"x": x_true}))
    bg = FourierBasisField(1, n_modes=6)
    an = _small_neural(1, heads=Heads({"a": "identity"}))
    field = CompositeField({"background": bg, "anomaly": an}, "sum")
    prob = InverseProblem(dom, field, op, LossSet({"data": MSE()}), meas)
    an0 = [p.detach().clone() for p in an.parameters()]
    bg0 = bg.coef.detach().clone()
    cur = Curriculum([Stage("s", (16,), 20, 1e-2, freeze=field.freeze_prefixes("anomaly"))])
    nefi.invert(prob, cur, device="cpu")
    assert all(torch.equal(a, b) for a, b in zip(an0, an.parameters()))
    assert not torch.equal(bg0, bg.coef)
    field.reset_parameters()
    assert torch.equal(bg.coef, bg0)


def test_anomaly_field_modes_and_support():
    dom = Domain.unit((10, 10))
    c = dom.coords()
    bg = FourierBasisField(2, 3, init_std=0.1)
    an = NeuralField(2, Heads({"a": GatedSoftplus(init_value=0.2)}), hidden=8, depth=2)
    support = torch.zeros(10, 10)
    support[3:7, 3:7] = 1.0
    with torch.no_grad():
        b = bg(c)["x"]
        a = an(c)["a"]
    f_add = AnomalyField(bg, an, "add", contrast=-2.0, support=support)
    assert torch.allclose(f_add(c)["x"], b - 2.0 * a * support, atol=1e-6)
    f_mul = AnomalyField(bg, an, "multiply", contrast=0.5)
    assert torch.allclose(f_mul(c)["x"], b * (1 + 0.5 * a), atol=1e-6)
    f_bl = AnomalyField(bg, an, "blend", inclusion_value=0.1, inclusion_bounds=(0.0, 1.0))
    v = float(f_bl.inclusion_value().detach())
    assert abs(v - 0.1) < 1e-4
    assert torch.allclose(f_bl(c)["x"], b * (1 - a) + v * a, atol=1e-5)
    # a callable support prior on a coarser grid
    f_fn = AnomalyField(bg, an, "add", support=lambda x: (x[..., 0] > 0).float())
    out = f_fn(dom.at((6, 6)).coords())["x"]
    assert out.shape == (6, 6)
    loss = f_add(c)["x"].pow(2).mean()
    loss.backward()
    assert _grads_nonzero(f_add.anomaly) and _grads_nonzero(f_add.background)
    with pytest.raises(ConfigError):
        AnomalyField(bg, an, "subtract")


# ------------------------------------------------------------------------------------------
# layered
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["constant", "grid", "neural"])
def test_layered_interfaces_monotone_2d_3d(model):
    torch.manual_seed(1)
    for ndim, shape in ((2, (16, 12)), (3, (8, 8, 6))):
        f = LayeredField(ndim, n_layers=4, interface_model=model, lateral_shape=4)
        with torch.no_grad():  # random perturbation of every parameter
            for p in f.parameters():
                p.add_(torch.randn_like(p))
        dom = Domain.unit(shape)
        assert f(dom.coords(), 0.5)["x"].shape == shape
        lat = torch.rand(50, ndim - 1) * 2 - 1
        z = f.interfaces(lat)
        assert z.shape == (50, 3)
        assert bool((z[:, 1:] > z[:, :-1]).all()) and bool((z[:, 0] > -1.0).all())


def test_layered_sharpens_with_progress_and_partial_volume():
    f = LayeredField(
        2, n_layers=2, values=[0.0, 1.0], interfaces=[0.03], eps_start=0.3, eps_end=1e-4
    )
    dom = Domain.unit((8, 40))
    with torch.no_grad():
        x0 = f(dom.coords(), 0.0)["x"][0]
        x1 = f(dom.coords(), 1.0)["x"][0]
    mixed = lambda x: int(((x > 1e-3) & (x < 1 - 1e-3)).sum())  # noqa: E731
    assert mixed(x1) < mixed(x0)
    assert mixed(x1) == 1  # only the cell containing the interface is mixed
    # partial volume: interface at z = 0.03, cells of width 0.05 → cell [0.0, 0.05] is 40% "high"
    h = 2.0 / 40
    k = int((0.03 + 1) / h)
    frac = ((k + 1) * h - 1 - 0.03) / h
    assert abs(float(x1[k]) - frac) < 1e-3
    # point rendering at the same eps is a hard step
    fp = LayeredField(2, 2, values=[0.0, 1.0], interfaces=[0.03], eps_end=1e-4, render="point")
    assert mixed(fp(dom.coords(), 1.0)["x"][0]) == 0
    assert smooth_step(torch.tensor([0.0]), 0.1).item() == 0.5


def test_layered_1d_grow_and_layer_index():
    f = LayeredField(1, n_layers=3, values=[0.0, 1.0, 2.0], n_active=2, eps_end=1e-3)
    dom = Domain.unit((30,))
    before = f(dom.coords())["x"].detach()
    assert float(before.max()) <= 1.0 + 1e-5  # third layer inactive
    assert f.can_grow() and f.grow()
    after = f(dom.coords())["x"].detach()
    assert torch.allclose(before, after, atol=1e-6)  # growth does not change the field
    assert not f.can_grow()
    idx = f.layer_index(dom.coords())
    assert int(idx.min()) == 0 and int(idx.max()) == 2
    f.reset_parameters()
    assert int(f.n_active) == 2


def test_layer_cake_field_gradients():
    dom = Domain.unit((12, 10))
    f = LayerCakeField(2, n_layers=2, layer_values=[1.0, 0.5], interface_model="grid")
    assert f.mode == "add" and float(f.contrast[0]) == -1.0  # defects lower the host value
    out = f(dom.coords(), 0.5)["x"]
    assert out.shape == (12, 10)
    assert bool((out <= f.background_map(dom.coords(), 0.5) + 1e-6).all())
    out.pow(2).mean().backward()
    assert float(f.layers.layer_values.grad.abs().sum()) > 0
    assert float(f.layers.thickness_logit.grad.abs().sum()) > 0
    assert _grads_nonzero(f.anomaly)
    shapes = StarShapeField(n_shapes=2, n_active=1, inside=1.0, outside=0.0, learn_values=False)
    g = LayerCakeField(2, 2, anomaly=shapes, mode="blend", inclusion_value=0.1)
    g(dom.coords(), 0.5)["x"].sum().backward()
    assert g.inclusion.grad is not None and _grads_nonzero(g.anomaly)
    built = build("field", {"type": "layer_cake", "ndim": 3, "n_layers": 3})
    assert built(Domain.unit((6, 6, 5)).coords())["x"].shape == (6, 6, 5)


# ------------------------------------------------------------------------------------------
# shapes
# ------------------------------------------------------------------------------------------
def test_star_shape_disk_area_and_perimeter():
    dom = Domain.unit((256, 256))
    f = StarShapeField(n_shapes=1, n_harmonics=4, radii=0.5, centers=[[0.1, -0.2]], eps_end=0.005)
    chi = f.indicator(dom.coords(), 1.0)
    area = float(chi.sum()) * (2 / 256) ** 2
    assert abs(area - math.pi * 0.25) / (math.pi * 0.25) < 5e-3
    r = (dom.coords() - torch.tensor([0.1, -0.2])).norm(dim=-1)
    disk = (r < 0.5).float()
    assert float((chi - disk).abs().mean()) < 0.01
    assert abs(float(f.areas()) - math.pi * 0.25) < 1e-4
    assert abs(float(interface_length(f)) - math.pi) < 1e-3
    # coarea perimeter of the rendered field
    x = f(dom.coords(), 1.0)["x"].detach()
    per = interface_length(x, spacing=(2 / 256, 2 / 256))
    assert abs(float(per) - math.pi) / math.pi < 0.03
    # Fourier descriptors change the shape and are regularized
    with torch.no_grad():
        f.coef[0, 2, 0] = 0.2
    assert float(f.perimeters()) > math.pi
    assert float(contour_regularizer(f, perimeter=0.0, smoothness=1.0)) > 0


def test_star_shape_values_3d_polygon_and_grow():
    dom = Domain.unit((32, 32))
    f = StarShapeField(
        n_shapes=2, n_harmonics=2, inside=[0.1, 0.9], outside=1.0, per_shape_values=True
    )
    assert f(dom.coords())["x"].shape == (32, 32)
    f3 = StarShapeField(ndim=3, n_shapes=1, half_heights=0.4, depth_centers=0.0, eps_end=0.01)
    x3 = f3.indicator(Domain.unit((24, 24, 24)).coords())
    assert float(x3[12, 12, 12]) > 0.99 and float(x3[12, 12, 1]) < 0.01
    sq = PolygonField(n_polygons=1, n_vertices=4, radius=0.5 * math.sqrt(2), eps_end=0.005)
    a = float(sq.indicator(Domain.unit((200, 200)).coords()).sum()) * (2 / 200) ** 2
    assert abs(a - 1.0) < 0.02 and abs(float(sq.areas()) - 1.0) < 1e-4
    g = StarShapeField(n_shapes=2, n_active=1, inside=1.0, outside=0.0)
    grad = torch.zeros(32, 32)
    grad[5, 20] = -3.0  # the loss decreases most when the field increases here
    assert g.grow({"gradient": grad, "coords": dom.coords()})
    assert torch.allclose(g.center[1], dom.coords()[5, 20], atol=1e-6)
    assert not g.can_grow()


# ------------------------------------------------------------------------------------------
# warps
# ------------------------------------------------------------------------------------------
def test_warps_round_trip_and_monotone():
    c2 = Domain.unit((16, 16)).coords()
    for w in (polar((0.1, -0.2)), polar(embed="circle")):
        assert torch.allclose(w.inverse(w(c2)), c2, atol=1e-5)
    c3 = Domain.unit((6, 6, 5)).coords()
    cy = cylindrical(axis=2)
    assert torch.allclose(cy.inverse(cy(c3)), c3, atol=1e-5)
    u = torch.linspace(-1, 1, 101)
    pts = torch.stack([u, u], -1)
    for w in (depth_stretch(0.5), log_depth(0.1), depth_stretch(0.4, face="high")):
        v = w(pts)[:, 1]
        assert bool((v[1:] > v[:-1]).all())  # strictly monotone
        assert abs(float(v[0]) + 1) < 1e-5 and abs(float(v[-1]) - 1) < 1e-5  # endpoints fixed
        assert torch.allclose(w.inverse(w(pts)), pts, atol=1e-5)
    # γ < 1 stretches the observed face (u = −1): the first cell is magnified
    v = depth_stretch(0.5)(pts)[:, 1]
    assert float(v[1] - v[0]) > 2 * float(u[1] - u[0])


def test_sensitivity_warp_matches_known_cumulative_profile():
    n = 64
    x = -1 + (2 * torch.arange(n) + 1) / n
    w = SensitivityWarp({0: 1.0 + x}, ndim=1, floor=0.0)
    edges = torch.linspace(-1, 1, n + 1).unsqueeze(-1)
    expected = (edges[:, 0] + 1) ** 2 / 2 - 1  # 2·C(u) − 1 with C = (u + 1)² / 4
    assert torch.allclose(w(edges)[:, 0], expected, atol=1e-5)
    assert torch.allclose(w.inverse(w(edges)), edges, atol=1e-5)
    s = torch.ones(8, 16)
    s[:, :4] = 10.0  # very sensitive near the first face of axis 1
    wm = sensitivity_warp(s, axes=(1,))
    assert float(wm.density(1, torch.tensor([-0.9]))) > float(wm.density(1, torch.tensor([0.9])))
    c = Domain.unit((8, 16)).coords()
    assert torch.equal(wm(c)[..., 0], c[..., 0])  # axis 0 untouched


def test_warped_and_deformable_fields():
    dom = Domain.unit((12, 12))
    c = dom.coords()
    t = _small_neural()
    d = DeformableField(t)
    assert torch.equal(d(c, 0.5)["x"], t(c, 1.0)["x"])  # zero displacement == template
    reg = WarpRegularizer(smoothness=1.0, magnitude=1.0, folding=1.0)
    ctx = Context(
        {"x": t(c)["x"]}, torch.zeros(12, 12), Measurement(torch.zeros(12, 12)), dom, field_module=d
    )
    assert float(reg(ctx)) == 0.0
    with torch.no_grad():
        d.warp.net.out.weight.normal_(0, 0.5)
    val = reg(ctx)
    assert float(val) > 0
    val.backward()
    assert _grads_nonzero(d.warp.net)
    wf = WarpedField(_small_neural(3), polar(embed="circle"))
    assert wf(c)["x"].shape == (12, 12)
    with pytest.raises(ConfigError):
        WarpedField(_small_neural(2), polar(embed="circle"))  # circle embedding needs 3 inputs


# ------------------------------------------------------------------------------------------
# spectral
# ------------------------------------------------------------------------------------------
def test_spectral_preconditioned_identity_and_lowpass():
    torch.manual_seed(0)
    dom = Domain.unit((32, 32))
    c = dom.coords()
    g = GridField((32, 32), Heads({"x": "identity"}), init=torch.randn(32, 32, 1))
    assert torch.equal(SpectralPreconditionedField(g)(c)["x"], g(c)["x"])
    ones = SpectralPreconditionedField(g, gain=lambda k: torch.ones_like(k))
    assert torch.allclose(ones(c)["x"], g(c)["x"], atol=1e-5)
    lp = SpectralPreconditionedField(g, gain=lowpass_gain(2.0))
    fx, fy = torch.fft.fft2(g(c)["x"]).abs(), torch.fft.fft2(lp(c)["x"]).abs()
    hi = slice(8, 25)
    assert float(fy[hi, hi].pow(2).sum()) < 1e-2 * float(fx[hi, hi].pow(2).sum())
    assert abs(float(lp(c)["x"].mean()) - float(g(c)["x"].mean())) < 1e-4  # DC preserved
    pre = SpectralPreconditionedField.from_sensitivity(
        g, [0.0, 4.0, 8.0], [1.0, 0.1, 0.01], floor=0.05
    )
    k = torch.tensor([0.0, 4.0, 8.0])
    assert torch.allclose(pre.gain_at(k), torch.tensor([1.0, 10.0, 20.0]), atol=1e-4)
    out = pre(c)["x"]
    out.sum().backward()
    assert g.param.grad is not None
    with pytest.raises(ShapeError):
        lp(torch.rand(10, 2) * 2 - 1)


def test_fourier_basis_field_annealing_growth_and_paths():
    dom = Domain.unit((16, 12))
    f = FourierBasisField(2, n_modes=5, init_std=0.3)
    x0 = f(dom.coords(), 0.0)["x"]
    assert float(x0.std()) < 1e-6  # only the constant is open at progress 0
    assert float(f(dom.coords(), 1.0)["x"].std()) > 0.01
    pts = dom.coords().reshape(-1, 2)
    assert torch.allclose(f.raw(dom.coords()).reshape(-1, 1), f.raw(pts), atol=1e-5)
    assert f.effective_bandwidth(1.0) == 1.0 and f.effective_bandwidth(0.0) == 0.0
    g = FourierBasisField(1, n_modes=8, basis="fourier", annealed=False, n_active=2)
    assert g.capacity() == {"shells": 2, "max_shells": 5}
    before = g(Domain.unit((20,)).coords())["x"].detach()
    assert g.grow() and g.capacity()["shells"] == 3
    assert torch.allclose(before, g(Domain.unit((20,)).coords())["x"].detach())
    assert float(g.sobolev_norm()) == 0.0
