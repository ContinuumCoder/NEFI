import math

import pytest
import torch

from nefi.domain import Domain
from nefi.errors import ConfigError, ShapeError
from nefi.fields import (
    Bounded,
    FourierFeatures,
    GatedSoftplus,
    GridField,
    Heads,
    NeuralField,
    Softplus,
    SupportMasked,
)
from nefi.solve.curriculum import Stage


def test_domain_coords_cell_centered_and_normalized():
    d = Domain((4, 8), ((0.0, 1.0), (-2.0, 2.0)))
    c = d.coords()
    assert c.shape == (4, 8, 2)
    assert torch.allclose(c[0, 0], torch.tensor([-0.75, -0.875]))
    assert torch.allclose(c[-1, -1], torch.tensor([0.75, 0.875]))
    p = d.physical_coords()
    assert torch.allclose(p[0, 0], torch.tensor([0.125, -1.75]))
    assert d.spacing() == (0.25, 0.5)
    assert d.at((8, 16)).spacing() == (0.125, 0.25)
    assert d.coarsen().shape == (4, 4)
    u = d.to_normalized(p)
    assert torch.allclose(u, c)
    assert torch.allclose(d.to_physical(c), p)


def test_domain_validation():
    with pytest.raises(ShapeError):
        Domain((4,), ((0, 1), (0, 1)))
    with pytest.raises(ShapeError):
        Domain((4,), ((1, 0),))
    assert Domain.from_spacing((64, 64), 20.0).extent == ((0.0, 1280.0), (0.0, 1280.0))


def test_fourier_annealing_weights():
    enc = FourierFeatures(2, n_octaves=4)
    assert enc.out_dim == 2 + 2 * 2 * 4
    w0 = enc.band_weights(0.0)
    assert torch.allclose(w0, torch.zeros(4))
    w1 = enc.band_weights(1.0)
    assert torch.allclose(w1, torch.ones(4))
    wh = enc.band_weights(0.5)  # β = 2 -> bands 0,1 fully on, 2,3 off
    assert torch.allclose(wh, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    w = enc.band_weights(0.625)  # β = 2.5 -> band 2 at half cosine ramp
    assert abs(float(w[2]) - 0.5) < 1e-6
    x = torch.rand(5, 3, 2) * 2 - 1
    assert enc(x, 0.3).shape == (5, 3, enc.out_dim)
    assert torch.allclose(enc(x, 0.0)[..., 2:], torch.zeros(5, 3, 16))
    assert torch.allclose(FourierFeatures(2, 4, annealed=False).band_weights(0.0), torch.ones(4))


def test_heads_bounded_gated_masked():
    raw = torch.randn(10, 10, 4) * 3
    h = Heads(
        {"rho": GatedSoftplus(), "w": SupportMasked(Bounded(1.5, 2.5), depends_on="rho", tau=0.3)}
    )
    assert h.n_in == 3 + 0 or h.n_in == 3
    out = h(raw[..., :3])
    assert (out["rho"] >= 0).all()
    mask = out["rho"] > 0.3 * out["rho"].max()
    assert torch.all(out["w"][~mask] == 0)
    assert torch.all((out["w"][mask] > 1.5) & (out["w"][mask] < 2.5))
    # stop-gradient through the mask: gradient of w wrt rho-channels must be zero
    r = raw[..., :3].clone().requires_grad_(True)
    o = h(r)
    o["w"].sum().backward()
    assert torch.all(r.grad[..., :2] == 0)
    with pytest.raises(ConfigError):
        Heads({"w": SupportMasked(Bounded(0, 1), depends_on="rho"), "rho": Softplus()})
    b = Bounded(0.003, 0.25, init_value=0.1)
    assert torch.allclose(b(b.init_bias()[0] * torch.ones(1, 1)), torch.full((1,), 0.1), atol=1e-6)


def test_neural_field_shapes_and_init():
    heads = Heads({"a": Softplus(init_value=0.2), "b": Bounded(0.0, 1.0, init_value=0.25)})
    f = NeuralField(2, heads, hidden=32, depth=3, skip_at=1, n_octaves=4)
    d = Domain.unit((8, 6))
    with torch.no_grad():
        out = f(d.coords(), progress=0.0)
    assert out["a"].shape == (8, 6) and out["b"].shape == (8, 6)
    # near-uniform initialization close to init values
    assert abs(float(out["a"].mean()) - 0.2) < 0.1
    assert abs(float(out["b"].mean()) - 0.25) < 0.1
    out2 = f(d.at((16, 12)).coords(), progress=1.0)
    assert out2["a"].shape == (16, 12)
    p0 = [p.clone() for p in f.parameters()]
    torch.manual_seed(1)
    f.reset_parameters()
    assert any(not torch.allclose(a, b) for a, b in zip(p0, f.parameters()))
    assert f.n_parameters() > 0


def test_neural_field_sine_and_relu_variants():
    for act in ("sine", "relu", "gelu"):
        f = NeuralField(
            3, Heads({"x": "identity"}), hidden=16, depth=2, activation=act, n_octaves=3
        )
        y = f(Domain.unit((4, 4, 4)).coords())["x"]
        assert y.shape == (4, 4, 4) and torch.isfinite(y).all()


def test_grid_field_resample_and_stage_hook():
    g = GridField((8, 8), Heads({"x": Softplus(init_value=0.5)}))
    d = Domain.unit((8, 8))
    y = g(d.coords())["x"]
    assert torch.allclose(y, torch.full((8, 8), 0.5), atol=1e-6)
    y2 = g(d.at((16, 16)).coords())["x"]  # on-the-fly evaluation at another resolution
    assert y2.shape == (16, 16)
    g.on_stage_start(Stage(shape=(16, 16)), d.at((16, 16)))
    assert tuple(g.param.shape) == (16, 16, 1)
    g.set_fields({"x": torch.full((16, 16), 2.0)})
    assert torch.allclose(g(d.at((16, 16)).coords())["x"], torch.full((16, 16), 2.0), atol=1e-5)
    assert math.isfinite(float(g.param.detach().sum()))
