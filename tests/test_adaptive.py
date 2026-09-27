"""Adaptive tools: annealing callbacks, capacity growth, model selection, spectra."""

import dataclasses
import math
from types import SimpleNamespace

import pytest
import torch

import nefi
from nefi.domain import Domain
from nefi.errors import ConfigError
from nefi.fields import GridField, Heads, NeuralField, Softplus
from nefi.fields.adaptive import (
    BandwidthSchedule,
    GrowCapacity,
    OperatorAwareAnnealing,
    PlateauDetector,
    RepresentationEnsemble,
    ResidualDrivenAnnealing,
    bandwidth_to_progress,
    field_space_gradient,
    holdout_masks,
    masked_downsampler,
    match_report,
    operator_spectrum,
    progress_to_bandwidth,
    representation_spectrum,
    select_representation,
    softmax_weights,
)
from nefi.fields.encoding import FourierFeatures
from nefi.fields.geometric import FourierBasisField, StarShapeField
from nefi.instances.toy1d import Toy1D
from nefi.measurement import Measurement
from nefi.operators import FFTConvolution, gaussian_kernel_fn
from nefi.solve import Curriculum, Stage, StepState

pytestmark = pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad")


class FakeSolver:
    """The attributes of :class:`nefi.solve.Solver` the adaptive callbacks touch."""

    def __init__(self, field, noise_std=None, shape=(16,)):
        dom = Domain.unit(shape)
        self.problem = SimpleNamespace(
            field=field,
            domain=dom,
            measurement=Measurement(torch.zeros(shape), noise_std=noise_std),
        )
        self.progress_override = None
        self.stop_stage = None
        self.global_step = 0
        self.curriculum = SimpleNamespace(total_steps=0)


def _drive(cb, solver, losses, stage, fields=None):
    """Feed a loss sequence through a callback like the solver loop does; returns progresses."""
    cb.on_run_start(solver)
    cb.on_stage_start(solver, 0, stage)
    used = []
    for t, loss in enumerate(losses):
        p = solver.progress_override
        used.append(p)
        f = fields(t) if callable(fields) else fields
        cb.on_step(solver, StepState(0, stage, t, t, 1e-3, p, loss, {}, loss, 0.0, f, None))
    return used


# ------------------------------------------------------------------------------------------
# plateaus and residual-driven annealing
# ------------------------------------------------------------------------------------------
def test_plateau_detector_criteria():
    d = PlateauDetector(patience=10, delta=1e-3)
    assert not any(d.update(0.9**t) for t in range(100))  # steady geometric decrease
    d = PlateauDetector(patience=10, delta=1e-3)
    flags = [d.update(1.0) for _ in range(12)]
    assert flags[:10] == [False] * 10 and flags[10]  # flat: fires after the window fills
    d = PlateauDetector(patience=10, delta=1e-6, rate_fraction=0.25)
    vals = [math.exp(-0.2 * t) + math.exp(-0.01 * t) for t in range(300)]  # fast, then slow
    fired = [t for t, v in enumerate(vals) if d.update(v)]
    assert fired and 20 < fired[0] < 150  # diminishing returns detected after the fast phase
    with pytest.raises(ConfigError):
        PlateauDetector(patience=0)


def test_residual_annealing_advances_only_on_plateaus():
    field = NeuralField(1, Heads({"x": "identity"}), hidden=8, depth=1, n_octaves=4)
    stage = Stage("s", None, 400, 1e-3)
    cb = ResidualDrivenAnnealing(patience=10, ramp_steps=5)
    decreasing = [0.97**t for t in range(150)]
    used = _drive(cb, FakeSolver(field), decreasing, stage)
    assert set(used) == {0.0} and cb.events == []  # still improving: no band opened
    cb = ResidualDrivenAnnealing(patience=10, ramp_steps=5)
    losses = [0.97**t for t in range(100)] + [0.97**99] * 150
    used = _drive(cb, FakeSolver(field), losses, stage)
    assert max(used[:100]) == 0.0  # nothing opens while the loss decreases
    assert used[-1] > 0.0 and cb.events and cb.events[0]["step"] >= 100
    assert all(b >= a for a, b in zip(used, used[1:]))  # monotone within the stage
    assert cb.K == 4 and cb.level <= 4
    levels = [e["level"] for e in cb.events]
    assert levels == sorted(levels) and len(set(levels)) == len(levels)
    # the ladder restarts at each stage
    cb.on_stage_start(FakeSolver(field), 1, stage)
    assert cb.progress == 0.0


def test_residual_annealing_respects_noise_floor_and_clock():
    problem, gt, meas = nefi.instances.toy1d.make_problem(
        n=16, seed=0, hidden=8, depth=1, n_octaves=4
    )
    solver = FakeSolver(problem.field)
    solver.problem = problem
    stage = Stage("s", None, 200, 1e-3)
    sigma = float(meas.noise_std)
    pred_at_floor = meas.data + 0.5 * sigma * torch.randn(16).sign()  # RMSE = 0.5 σ
    cb = ResidualDrivenAnnealing(patience=5, ramp_steps=2, discrepancy_tau=1.1, stop_at_noise=True)
    cb.on_run_start(solver)
    cb.on_stage_start(solver, 0, stage)
    for t in range(30):
        cb.on_step(
            solver, StepState(0, stage, t, t, 1e-3, 0.0, 1.0, {}, 1.0, 0.0, None, pred_at_floor)
        )
        if solver.stop_stage:
            break
    assert cb.events == [] and cb.at_noise_floor and solver.stop_stage == "noise_floor"
    # clock: opens after max_steps_per_level even while the loss keeps decreasing
    cb = ResidualDrivenAnnealing(
        patience=10, ramp_steps=1, max_steps_per_level=20, discrepancy_tau=None
    )
    _drive(cb, FakeSolver(problem.field), [0.97**t for t in range(70)], stage)
    assert [e["reason"] for e in cb.events][:2] == ["clock", "clock"]


def test_bandwidth_progress_mapping_and_schedule():
    enc = FourierFeatures(2, n_octaves=4)  # bands at 0.5, 1, 2, 4 cycles / unit
    assert bandwidth_to_progress(enc, 0.0) == 0.0
    assert abs(bandwidth_to_progress(enc, 0.5) - 0.25) < 1e-6
    assert abs(bandwidth_to_progress(enc, 0.75) - 0.375) < 1e-6
    assert bandwidth_to_progress(enc, 4.0) == 1.0 and bandwidth_to_progress(enc, 99.0) == 1.0
    for p in (0.25, 0.5, 0.75, 1.0):
        b = progress_to_bandwidth(enc, p)
        assert abs(bandwidth_to_progress(enc, b) - p) < 1e-6
    fb = FourierBasisField(2, n_modes=5)  # shells at 0.25 … 1.0 cycles / unit
    assert abs(bandwidth_to_progress(fb, 0.5) - 0.5) < 1e-6
    field = NeuralField(1, Heads({"x": "identity"}), hidden=8, depth=1, n_octaves=4)
    cb = BandwidthSchedule([(0.0, 0.5), (1.0, 4.0)])
    used = _drive(cb, FakeSolver(field), [1.0] * 50, Stage("s", None, 50, 1e-3))
    assert abs(used[0] - 0.25) < 1e-6 and all(b >= a for a, b in zip(used, used[1:]))
    assert abs(FakeSolver(field).progress_override or 0.0) == 0.0
    with pytest.raises(ConfigError):
        bandwidth_to_progress(GridField((4,), Heads({"x": "identity"})), 1.0)


def _powerlaw_field(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, n, generator=g)
    f = torch.fft.fftfreq(n, d=2.0 / n)
    kr = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    return torch.fft.ifft2(torch.fft.fft2(w) / (1.0 + kr) ** 1.5).real


def test_operator_aware_annealing_follows_noise_level():
    field = NeuralField(2, Heads({"x": "identity"}), hidden=8, depth=1, n_octaves=6)
    freqs = torch.linspace(0, 16, 33)
    sens = torch.exp(-freqs / 2.0)
    x = _powerlaw_field()
    stage = Stage("s", None, 200, 1e-3)
    finals = {}
    for noise in (1e-6, 1e3):
        cb = OperatorAwareAnnealing(sens, noise_level=noise, freqs=freqs, update_every=10)
        used = _drive(cb, FakeSolver(field, shape=(64, 64)), [1.0] * 200, stage, {"x": x})
        assert all(b >= a - 1e-12 for a, b in zip(used, used[1:]))
        finals[noise] = used[-1]
    assert finals[1e3] == 0.0  # nothing is resolvable above that noise
    assert finals[1e-6] > 0.9  # everything is: the schedule opens all bands (rate-limited)
    cb = OperatorAwareAnnealing(sens, noise_level=1e-2, freqs=freqs)
    bw = cb.target_bandwidth(x, 1.0)
    cb._noise = 1e-2
    assert 0.0 < cb.target_bandwidth(x, 1.0) <= 16.0 and bw >= 0.0
    with pytest.raises(ConfigError):
        OperatorAwareAnnealing(sens)  # tensor profile without freqs


# ------------------------------------------------------------------------------------------
# capacity growth
# ------------------------------------------------------------------------------------------
def test_grow_capacity_opens_modes_on_plateaus():
    inst = Toy1D(n=32, hidden=8, depth=1, n_octaves=3)
    gt, meas = inst.make_measurement(0)
    basis = FourierBasisField(
        1, n_modes=12, heads=Heads({"x": Softplus(init_value=0.1)}), annealed=False, n_active=1
    )
    prob = dataclasses.replace(inst.build_problem(meas), field=basis)
    cb = GrowCapacity(patience=10, use_progress=False)
    res = nefi.invert(
        prob, Curriculum([Stage("s", (32,), 300, 5e-2)]), device="cpu", callbacks=[cb]
    )
    assert cb.n_grows >= 2 and basis.capacity()["shells"] == 1 + cb.n_grows
    assert all(e["action"] == "modules" for e in cb.events)
    h = res.history["data_loss"]
    assert h[-1] < 0.5 * h[0]


def test_grow_capacity_respects_noise_floor():
    problem, gt, meas = nefi.instances.toy1d.make_problem(
        n=16, seed=0, hidden=8, depth=1, n_octaves=3
    )
    basis = FourierBasisField(1, n_modes=6, annealed=False, n_active=1)
    problem = dataclasses.replace(problem, field=basis)
    solver = FakeSolver(basis)
    solver.problem = problem
    stage = Stage("s", None, 100, 1e-3)
    sigma = float(meas.noise_std)
    at_floor = meas.data + 0.5 * sigma * torch.randn(16).sign()
    far = meas.data + 5.0 * sigma
    for pred, expect_growth in ((at_floor, False), (far, True)):
        cb = GrowCapacity(patience=5, use_progress=False, placement="default")
        cb.on_run_start(solver)
        cb.on_stage_start(solver, 0, stage)
        for t in range(12):
            cb.on_step(solver, StepState(0, stage, t, t, 1e-3, 1.0, 1.0, {}, 1.0, 0.0, None, pred))
        assert (cb.n_grows > 0) == expect_growth
        assert cb.at_noise_floor == (not expect_growth)
        basis.reset_parameters()


def test_field_space_gradient_places_new_shapes():
    dom = Domain.unit((16, 16))
    op = FFTConvolution(gaussian_kernel_fn(0.05), dom)
    truth = torch.zeros(16, 16)
    truth[3:6, 10:13] = 1.0
    meas = Measurement(op({"x": truth}))
    shapes = StarShapeField(n_shapes=2, n_active=0, radii=0.15, inside=1.0, outside=0.0)
    prob = nefi.InverseProblem(dom, shapes, op, nefi.LossSet({"data": nefi.MSE()}), meas)
    stage = Stage("s", (16, 16), 10, 1e-3)
    with torch.no_grad():
        fields = shapes(dom.coords())
    state = StepState(0, stage, 0, 0, 1e-3, 1.0, 0.0, {}, 0.0, 0.0, fields, None)
    hint = field_space_gradient(SimpleNamespace(problem=prob), state)
    assert hint["gradient"].shape == (16, 16)
    assert shapes.grow(hint)
    c = shapes.center[0].detach()
    assert float(torch.dist(c, dom.coords()[4, 11])) < 0.3  # placed on the missing inclusion


# ------------------------------------------------------------------------------------------
# selection
# ------------------------------------------------------------------------------------------
def test_holdout_masks_and_leakage_free_downsampling():
    meas = Measurement(torch.arange(24.0).reshape(4, 6))
    ((tr, va),) = holdout_masks(meas, holdout=0.3, seed=1)
    assert torch.equal(tr + va, torch.ones(4, 6)) and 0 < int(va.sum()) < 24
    folds = holdout_masks(meas, seed=0, n_folds=3)
    assert torch.equal(sum(v for _, v in folds), torch.ones(4, 6))
    grouped = holdout_masks(Measurement(torch.zeros(5, 4, 6)), 0.3, 0, group_axes=(0,))
    assert bool((grouped[0][1] == grouped[0][1][0:1]).all())  # shared across axis 0
    dom = Domain.unit((8,))
    op = FFTConvolution(gaussian_kernel_fn(0.05), dom)
    data = torch.ones(8)
    data[3] = 1e6  # a held-out value that must not leak into the coarse data
    mask = torch.ones(8)
    mask[3] = 0.0
    coarse = masked_downsampler(op)(Measurement(data, mask), (4,))
    assert float(coarse.data.max()) == 1.0 and coarse.mask is not None
    with pytest.raises(ConfigError):
        holdout_masks(meas, holdout=1.5)


def test_select_representation_toy1d_and_ensemble():
    inst = Toy1D(n=48, hidden=32, depth=3, n_octaves=5, steps=(150, 250))
    gt, meas = inst.make_measurement(0)

    def factory(field):
        return dataclasses.replace(inst.build_problem(meas), field=field)

    heads = lambda: Heads({"x": Softplus(init_value=0.1)})  # noqa: E731
    candidates = {
        "neural": lambda: NeuralField(1, heads(), hidden=32, depth=3, skip_at=2, n_octaves=5),
        "grid_tiny_budget": lambda: GridField((48,), heads()),
        "broken": lambda: NeuralField(2, heads(), hidden=8, depth=1),  # wrong dimension
    }
    report = select_representation(
        factory, candidates, holdout=0.15, seed=0, budget_scale=0.3, device="cpu"
    )
    assert report.winner == "neural"
    assert [c.name for c in report.ranking][-1] == "broken" and report["broken"].error
    best = report.ranking[0]
    assert best.val_mse < report["grid_tiny_budget"].val_mse and best.field is not None
    assert best.n_parameters > 0 and best.steps > 0
    text = report.table()
    assert "winner" in text and "neural" in text and report.to_dict()["winner"] == "neural"
    ens = RepresentationEnsemble.from_report(report)
    assert abs(float(ens.weights.sum()) - 1.0) < 1e-6
    assert ens.weight_dict()["neural"] > ens.weight_dict()["grid_tiny_budget"]
    out = ens(Domain.unit((48,)).coords())
    assert out["x"].shape == (48,)
    assert ens.spread(Domain.unit((48,)).coords())["x"].shape == (48,)
    w = softmax_weights({"a": 1.0, "b": 2.0, "c": math.inf})
    assert abs(float(w.sum()) - 1.0) < 1e-12 and float(w[2]) == 0.0
    assert abs(float(w[1] / w[0]) - math.exp(-1.0)) < 1e-9  # relative temperature T = s_min


# ------------------------------------------------------------------------------------------
# spectra
# ------------------------------------------------------------------------------------------
def test_representation_spectrum_grid_flat_neural_lowpass():
    torch.manual_seed(0)
    dom = Domain.unit((32, 32))
    grid = GridField((32, 32), Heads({"x": "identity"}))
    s = representation_spectrum(grid, dom, n_probes=3)
    assert torch.allclose(s.values, torch.ones_like(s.values), atol=1e-5)  # G = I
    assert s.energy_bandwidth(0.9) > 0.9 * s.nyquist
    nf = NeuralField(2, Heads({"x": "identity"}), hidden=64, depth=3, n_octaves=6)
    lo = representation_spectrum(nf, dom, n_probes=4, progress=0.0)
    hi = representation_spectrum(nf, dom, n_probes=4, progress=1.0)
    v = lo.normalized().values
    f = lo.freqs
    assert float(v[f >= 4.0].mean()) < 0.1 * float(v[(f > 0) & (f <= 1.0)].mean())  # low-pass
    assert lo.energy_bandwidth(0.9) < 0.5 * lo.nyquist
    assert hi.energy_bandwidth(0.9) > lo.energy_bandwidth(0.9)  # annealing widens the band
    fb = representation_spectrum(FourierBasisField(2, n_modes=6), dom, n_probes=3)
    # hard cutoff: shells up to 5/4 cycles / unit per axis, i.e. radius √2·5/4 at the corner mode
    assert fb.energy_bandwidth(0.95) <= math.sqrt(2) * 1.25 + 0.5
    assert fb.energy_bandwidth(0.95) < 0.3 * s.energy_bandwidth(0.95)


def test_operator_spectrum_matches_gaussian_transfer():
    dom = Domain.unit((64, 64))
    sigma = 0.03
    op = FFTConvolution(gaussian_kernel_fn(sigma), dom)
    spec = operator_spectrum(op, dom, {"x": torch.ones(64, 64)}, n_probes=3, margin=0.3)
    for nu in (2.0, 4.0, 6.0):
        expected = math.exp(-2 * math.pi**2 * (2 * sigma) ** 2 * nu**2)
        assert abs(float(spec.at(nu)) - expected) < 0.05 * expected + 2e-3
    j = operator_spectrum(op, dom, {"x": torch.ones(64, 64)}, n_probes=2, margin=0.3, mode="j")
    assert abs(float(j.at(4.0)) - float(spec.at(4.0))) < 0.02


def test_match_report_verdicts():
    problem, gt, meas = nefi.instances.toy1d.make_problem(n=64, seed=0)
    grid = GridField((64,), Heads({"x": "identity"}))
    rep = match_report(grid, problem, n_probes=3)
    assert rep.verdict == "under-bandlimited" and rep.method == "data"
    assert 2.0 < rep.data_bandwidth < 12.0 and rep.rep_bandwidth > 0.8 * rep.nyquist
    smooth = FourierBasisField(1, n_modes=3)
    rep2 = match_report(smooth, problem, n_probes=3, operator_spec=rep.operator)
    assert rep2.verdict == "over-bandlimited"
    assert "OVER-BANDLIMITED" in str(rep2) and "ν_data" in rep2.text
    assert "| verdict |" in rep2.to_markdown() and rep2.to_dict()["verdict"] == "over-bandlimited"
    prior = match_report(grid, problem, n_probes=2, method="prior", operator_spec=rep.operator)
    assert prior.method == "prior" and prior.data_bandwidth > 0
