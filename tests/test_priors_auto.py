"""Prior DSL and conflicts, noise estimation, auto curriculum / LR / weights, from_forward."""

import math

import pytest
import torch

import nefi
from nefi.auto import (
    auto_curriculum,
    balance_weights,
    budget_steps,
    estimate_noise,
    from_forward,
    lr_range_test,
    quick_report,
)
from nefi.errors import ConfigError, ShapeError
from nefi.fields import GatedSoftplus, LevelSetHead, MaskedHead, MassNormalized, ScaledHead
from nefi.fields.heads import Bounded as BoundedHead
from nefi.instances.toy1d import Toy1D
from nefi.measurement import Measurement
from nefi.metrics import psnr
from nefi.priors import (
    Binary,
    Conserved,
    KnownSupport,
    NonNegative,
    PiecewiseConstant,
    Prior,
    Smooth,
    Sparse,
    Symmetric,
    combine,
    describe_catalogue,
)


# --------------------------------------------------------------------------------------------
# prior DSL
# --------------------------------------------------------------------------------------------
def test_prior_dsl_parsing():
    ps = Prior.parse("nonnegative + sparse(l1=1e-2) + piecewise_constant")
    assert [type(p) for p in ps] == [NonNegative, Sparse, PiecewiseConstant]
    assert ps[1].l1 == 1e-2 and ps[2].tv == 1e-2
    ps = Prior.parse(
        "Bounded(0, 1) + TV(tv=3e-3) + piecewise-constant + symmetric(radial, hard=False)"
    )
    assert ps[0].lo == 0.0 and ps[0].hi == 1.0 and ps[1].tv == 3e-3
    assert isinstance(ps[3], Symmetric) and ps[3].kind == "radial" and ps[3].hard is False
    assert Prior.parse("bounded(lo=-inf, hi=0)".replace("-inf", "-1"))[0].lo == -1.0
    assert (
        Prior.parse("smooth")[0].laplacian == 1e-2
        and Prior.parse("smooth(tikhonov=1e-3)")[0].laplacian is None
    )
    combo = NonNegative() + "sparse" + [PiecewiseConstant()]
    assert [type(p) for p in combo] == [NonNegative, Sparse, PiecewiseConstant]
    assert [type(p) for p in ("tv" + Smooth())] == [PiecewiseConstant, Smooth]
    assert "nonnegative" in describe_catalogue()


@pytest.mark.parametrize(
    "spec,match",
    [
        ("nonnegativ", "Did you mean 'nonnegative'"),
        ("bounded(0, 1, foo=2)", "Signature"),
        ("bounded(1, 0)", "hi > lo"),
        ("sparse(l1=", "unbalanced"),
        ("bounded(0, x + 1)", "not a literal"),
        ("positive + sparse", "nonnegative \\+ sparse"),
        ("nonnegative + bounded(-1, 1)", "contradicts"),
        ("bounded(0, 1) + binary(0, 2)", "contradicts"),
        ("known_support", "mask"),
        ("symmetric(spiral)", "kind"),
    ],
)
def test_prior_dsl_errors_are_actionable(spec, match):
    with pytest.raises(ConfigError, match=match):
        combine(spec)


def test_combine_resolves_heads_and_modifiers():
    c = combine("nonnegative + sparse(l1=1e-2) + piecewise_constant", ndim=2)
    assert isinstance(c.top, Sparse)
    head = c.head()
    assert isinstance(head, ScaledHead) and isinstance(head.inner, GatedSoftplus)
    assert set(c.losses()) == {"l1", "tv"}
    assert isinstance(combine("nonnegative + bounded(0, 1)").head(), BoundedHead)
    assert isinstance(combine("bounded(0, 1) + binary(0, 1)").head(), LevelSetHead)
    assert combine("binary(0, 1)").curriculum_hints()["anneal_fraction"] == 0.8
    mask = torch.ones(4, 4)
    h = combine(["nonnegative", KnownSupport(mask), Conserved(total=2.0)], ndim=2).head()
    assert isinstance(h, MassNormalized) and isinstance(h.inner, MaskedHead)
    soft = combine(["bounded(0, 1)", Conserved(total=2.0)])  # bounded -> penalty, not rescaling
    assert "conservation" in soft.losses() and not isinstance(soft.head(), MassNormalized)
    with pytest.raises(ConfigError, match="non-negative"):
        combine(["bounded(0, 1)", Conserved(total=2.0, hard=True)]).head()
    per = combine("periodic + piecewise_constant + smooth", ndim=2)
    assert per.losses()["tv"][0].periodic_axes == (0, 1)
    assert per.field_hints()["include_input"] is False
    assert combine("nonnegative + nonnegative").priors == [NonNegative()]  # deduplicated
    assert isinstance(combine("").top, type(combine("unconstrained").top))
    assert Binary(0, 1).rank() > Sparse().rank() > NonNegative().rank()


# --------------------------------------------------------------------------------------------
# noise estimation
# --------------------------------------------------------------------------------------------
def _smooth_2d(n=64):
    u = (torch.arange(n) + 0.5) / n
    X, Y = torch.meshgrid(u, u, indexing="ij")
    return torch.exp(-((X - 0.4) ** 2 + (Y - 0.6) ** 2) / 0.02) + 0.5 * torch.sin(3 * X)


@pytest.mark.parametrize("sigma", [0.01, 0.1])
def test_estimate_noise_within_20_percent(sigma):
    g = torch.Generator().manual_seed(0)
    # 2-D (Immerkær Laplacian MAD)
    clean = _smooth_2d()
    meas = Measurement(clean + sigma * torch.randn(clean.shape, generator=g))
    est = estimate_noise(meas)
    assert abs(est / sigma - 1) < 0.2
    assert meas.noise_std == pytest.approx(est) and meas.meta["noise_std_estimated"]
    assert meas.meta["noise_estimator"] == "laplacian"
    # 1-D blurred bumps (second differences)
    toy = Toy1D(n=128, noise_std=0.0)
    gt, clean1 = toy.make_measurement(seed=0)
    y = clean1.data + sigma * torch.randn(128, generator=g)
    assert abs(estimate_noise(y) / sigma - 1) < 0.2
    # time traces (T, n_sensors): differences along time only
    t = torch.linspace(0, 1, 400)[:, None]
    traces = torch.sin(6 * t + torch.arange(3.0)) + sigma * torch.randn(400, 3, generator=g)
    assert abs(estimate_noise(traces) / sigma - 1) < 0.2
    # complex data and a random observation mask
    z = torch.complex(clean, 0.5 * clean) + sigma * torch.complex(
        torch.randn(clean.shape, generator=g), torch.randn(clean.shape, generator=g)
    )
    mask = (torch.rand(clean.shape, generator=g) < 0.7).float()
    assert abs(estimate_noise(Measurement(z, mask=mask), store=False) / sigma - 1) < 0.2


def test_estimate_noise_does_not_overwrite_known_sigma_and_validates():
    meas = Measurement(torch.randn(32, 32), noise_std=0.5)
    estimate_noise(meas)
    assert meas.noise_std == 0.5
    with pytest.raises(ConfigError):
        estimate_noise(torch.randn(16), method="median")
    with pytest.raises(ConfigError):
        estimate_noise(Measurement(torch.randn(8, 8), mask=torch.zeros(8, 8)))


# --------------------------------------------------------------------------------------------
# curriculum, LR probe, weights
# --------------------------------------------------------------------------------------------
def test_auto_curriculum_budgets():
    assert budget_steps("default", (64, 64)) == 1500
    assert budget_steps("quick", (128,)) == 150  # size factor clipped at 0.5
    assert budget_steps("thorough", (256, 256, 64)) == 10000  # clipped at 2
    assert budget_steps(123, (8,)) == 123
    with pytest.raises(ConfigError):
        budget_steps("huge", (8,))
    c = auto_curriculum((64, 64), "quick")
    assert [s.shape for s in c.stages] == [(32, 32), (64, 64)] and c.total_steps == 300
    assert c.stages[1].lr == pytest.approx(0.5 * c.stages[0].lr) and c.optim.grad_clip == 1.0
    assert c.optim.optimizer == "adamw" and c.stages[0].lr_schedule == "cosine"
    assert c.discrepancy_tau is None and c.early_stop_patience is None
    single = auto_curriculum((12, 12), 200)  # too small for a coarse stage
    assert len(single.stages) == 1 and single.stages[0].shape == (12, 12)
    assert len(auto_curriculum((64, 64), 200, multiscale=False).stages) == 1
    t = auto_curriculum((64,), "thorough", noise_known=True, hints={"anneal_fraction": 0.8})
    assert t.discrepancy_tau == 1.0 and t.early_stop_patience >= 100
    assert all(s.anneal_fraction == 0.8 for s in t.stages)
    assert auto_curriculum((64,), 100, representation="grid").stages[0].lr == 1e-1
    assert auto_curriculum((64,), 100, lr=5e-4).stages[0].lr == 5e-4


def _blur_problem(n=64, prior="nonnegative + piecewise_constant", **kw):
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16
    blur = lambda x: torch.nn.functional.conv1d(x[None, None], k[None, None], padding=2)[0, 0]  # noqa: E731
    t = (torch.arange(n) + 0.5) / n
    gt = ((t - 0.5).abs() < 0.2).float() + 0.5 * ((t - 0.2).abs() < 0.05).float()
    y = blur(gt) + 0.01 * torch.randn(n, generator=torch.Generator().manual_seed(0))
    return blur, y, gt, from_forward(blur, y, shape=(n,), prior=prior, **kw)


def test_lr_range_test_returns_finite_lr_and_restores_parameters():
    _, _, _, prob = _blur_problem(budget=100)
    before = [p.detach().clone() for p in prob.field.parameters()]
    lrs = (1e-4, 1e-3, 1e-2, 1e-1)
    lr = lr_range_test(prob, lrs=lrs, steps=10)
    assert math.isfinite(lr) and lr in lrs
    assert all(torch.equal(a, b) for a, b in zip(before, prob.field.parameters()))


def test_balance_weights_normalizes_data_and_scales_regularizers():
    _, _, _, prob = _blur_problem(budget=100, weights="raw")
    assert prob.losses.weights == {"data": 1.0, "tv": 1e-2}
    w = balance_weights(prob, {"tv": 1e-2}, probe_steps=10)
    total, comps = prob.loss()
    assert w["data"] * comps["data"] == pytest.approx(1.0, rel=1e-4)  # data = 1 at the init
    assert 0 < w["tv"] < 1e6 and math.isfinite(w["tv"])
    _, _, _, p2 = _blur_problem(budget=100, weights={"tv": 0.5})
    assert p2.losses.weights == {"data": 1.0, "tv": 0.5}
    with pytest.raises(ConfigError):
        _blur_problem(budget=100, weights={"nope": 1.0})


# --------------------------------------------------------------------------------------------
# from_forward
# --------------------------------------------------------------------------------------------
def _user_blur(n, sigma_phys):
    sp = sigma_phys * n
    R = int(math.ceil(4 * sp))
    r = torch.arange(-R, R + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (r / sp) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    return lambda x: torch.nn.functional.conv1d(x.view(1, 1, -1), k, padding=R).view(-1)


def test_from_forward_plain_blur_matches_toy1d_quality():
    """(a) a user-written blur through from_forward vs the hand-configured toy1d instance."""
    inst = Toy1D(n=64, steps=(150, 300))
    gt, meas = inst.make_measurement(seed=0)
    ref = psnr(nefi.invert(inst.build_problem(meas), device="cpu", seed=0).fields["x"], gt["x"])
    problem = from_forward(
        _user_blur(64, inst.cfg.sigma),
        meas.data,
        shape=(64,),
        prior="nonnegative + smooth",
        budget=450,
    )
    auto = problem.meta["auto"]
    assert "single stage" in auto["multiscale"] and "estimated" in auto["noise"]
    assert problem.curriculum.discrepancy_tau == 1.0
    res = nefi.invert(problem, device="cpu", seed=0)
    assert psnr(res.fields["x"], gt["x"]) > ref - 1.0
    report = quick_report(res, problem, gt=gt)
    assert "noise level" in report and "PSNR" in report and "automatic decisions" in report


def test_from_forward_nonlinear_pointwise_plus_blur():
    """(b) saturating camera: tanh(gain · blur(x))."""
    n = 32
    u = (torch.arange(n) + 0.5) / n
    X, Y = torch.meshgrid(u, u, indexing="ij")
    gt = torch.exp(-((X - 0.4) ** 2 + (Y - 0.55) ** 2) / (2 * 0.12**2))
    gt = gt + 0.8 * (((X - 0.7) ** 2 + (Y - 0.3) ** 2) < 0.1**2).float()
    kk = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
    kk = ((kk[:, None] * kk[None]) / 256)[None, None]

    def camera(x):
        return torch.tanh(1.5 * torch.nn.functional.conv2d(x[None, None], kk, padding=2)[0, 0])

    y = camera(gt) + 0.01 * torch.randn(n, n, generator=torch.Generator().manual_seed(0))
    problem = from_forward(
        camera, y, shape=(n, n), prior="nonnegative + piecewise_constant", budget=250
    )
    res = nefi.invert(problem, device="cpu", seed=0)
    assert psnr(res.fields["x"], gt) > 24.0
    assert res.history["total"][-1] < 0.05 * res.history["total"][0]


def test_from_forward_multi_field_and_parametric():
    n = 32
    t = (torch.arange(n) + 0.5) / n
    a_gt, b_gt = torch.exp(-(((t - 0.3) / 0.1) ** 2)), 0.5 + 0.3 * t
    k = torch.tensor([1.0, 2.0, 1.0]) / 4

    def fwd(f):
        blur = torch.nn.functional.conv1d(f["a"][None, None], k[None, None], padding=1)[0, 0]
        return torch.stack([blur, f["b"] * (1 + f["a"])])

    y = fwd({"a": a_gt, "b": b_gt}) + 0.005 * torch.randn(
        2, n, generator=torch.Generator().manual_seed(0)
    )
    prob = from_forward(
        fwd, y, shape=(n,), prior={"a": "nonnegative + smooth", "b": "bounded(0, 2)"}, budget=250
    )
    assert prob.losses.names == ("data", "laplacian_a") and prob.field.names == ("a", "b")
    res = nefi.invert(prob, device="cpu", seed=0)
    assert psnr(res.fields["a"], a_gt) > 25 and psnr(res.fields["b"], b_gt, data_range=1.0) > 25
    # parametric representation: identity head by default, few parameters, single stage
    c = nefi.Domain.unit((16, 16)).coords()
    blob = 1.5 * torch.exp(-((c - torch.tensor([0.1, -0.2])) ** 2).sum(-1) / (2 * 0.3**2))
    pp = from_forward(
        lambda x: x,
        blob,
        shape=(16, 16),
        representation="parametric",
        fn=nefi.fields.gaussian_blobs(1, ndim=2, centers=[[0.0, 0.0]]),
        budget=300,
        noise=0.0,
    )
    assert pp.field.n_parameters() == 4 and len(pp.curriculum.stages) == 1
    nefi.invert(pp, device="cpu", seed=0)
    got = pp.field.unpack()
    assert abs(float(got["amplitude"][0]) - 1.5) < 0.05 and abs(float(got["sigma"][0]) - 0.3) < 0.02


def test_from_forward_errors_are_actionable():
    y = torch.rand(16)
    with pytest.raises(ShapeError, match="returned shape"):
        from_forward(lambda x: x[:8], y, shape=(16,))
    with pytest.raises(ConfigError, match="differentiabl"):
        from_forward(lambda x: x.detach() * 2, y, shape=(16,))
    with pytest.raises(ConfigError, match="failed"):
        from_forward(lambda x: x @ torch.ones(3), y, shape=(16,))
    with pytest.raises(ConfigError, match="representation"):
        from_forward(lambda x: x, y, shape=(16,), representation="voxels")
    with pytest.raises(ConfigError, match="noise"):
        from_forward(lambda x: x, y, shape=(16,), noise="laplace")
    with pytest.raises(ConfigError, match="homogeneity"):
        from_forward(lambda x: x, y, shape=(16,), prior="nonnegative + scale")


def test_from_forward_decisions_overrides_and_symmetry():
    n = 32
    c = nefi.Domain.unit((n, n)).coords()
    gt = torch.exp(-4 * (c**2).sum(-1))
    prob = from_forward(
        lambda x: x,
        gt + 0.01 * torch.randn(n, n, generator=torch.Generator().manual_seed(0)),
        shape=(n, n),
        prior="nonnegative + symmetric(radial)",
        budget=120,
        noise=0.01,
        hidden=32,
        depth=2,
        n_octaves=4,
        lr=5e-3,
        multiscale=False,
    )
    auto = prob.meta["auto"]
    assert "hidden=32" in auto["representation"] and "given" in auto["noise"]
    assert len(prob.curriculum.stages) == 1 and prob.curriculum.stages[0].lr == 5e-3
    assert prob.measurement.noise_std == 0.01 and prob.curriculum.discrepancy_tau == 1.0
    res = nefi.invert(prob, device="cpu", seed=0)
    x = res.fields["x"]
    assert torch.allclose(x, x.T, atol=1e-5) and torch.allclose(x, x.flip(0), atol=1e-5)
    # upsample multiscale for a plain function, discrepancy off, Measurement mask
    m = (torch.rand(n, n, generator=torch.Generator().manual_seed(1)) < 0.5).float()
    p2 = from_forward(
        lambda x: x, gt, shape=(n, n), multiscale="upsample", discrepancy=False, mask=m, budget=60
    )
    assert len(p2.curriculum.stages) == 2 and p2.curriculum.discrepancy_tau is None
    assert torch.equal(p2.measurement.mask, m)
    nefi.invert(p2, device="cpu", seed=0)


@pytest.mark.parametrize("rep", ["splat", "deep_decoder"])
def test_from_forward_baseline_representations(rep):
    baselines = pytest.importorskip("nefi.baselines")
    cls = {"splat": "GaussianSplatField", "deep_decoder": "DeepDecoderField"}[rep]
    if not hasattr(baselines, cls):
        pytest.skip(f"nefi.baselines.{cls} not available")
    n = 16
    c = nefi.Domain.unit((n, n)).coords()
    gt = torch.exp(-8 * (c**2).sum(-1))
    prob = from_forward(lambda x: x, gt, shape=(n, n), representation=rep, budget=40, noise=0.0)
    assert cls in prob.meta["auto"]["representation"]
    res = nefi.invert(prob, device="cpu", seed=0)
    assert res.history["total"][-1] < res.history["total"][0]
