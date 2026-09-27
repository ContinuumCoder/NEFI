"""Performance-engineering pass: every optimization is checked against the previous implementation.

1. heat stencil (flat padded layout vs the ``torch.roll`` reference; kernel counts; Chebyshev);
2. whole-step compile path (compiled vs eager numbers);
3. mixed precision for the field (CPU bf16 autocast);
4. batched multi-measurement solving (batched == sequential per problem);
5. sharded benchmarks (shards merge into the unsharded table);
6. micro-optimizations (Fourier-feature caches, optimizer ``foreach``).
"""

from __future__ import annotations

import copy
import logging
import math

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import nefi
from nefi.domain import Domain
from nefi.errors import ConfigError, SolverError
from nefi.instances.toy1d import Toy1D
from nefi.operators.pde import GaussianFlash, HeatOperator, conjugate_gradient
from nefi.operators.pde.stencil import (
    BoundarySpec,
    DiffusionStencil,
    face_conductances,
    flat_layout,
    neighbors,
)
from nefi.solve.solver import normalize_autocast, normalize_compile
from nefi.utils.compat import autocast_enabled
from nefi.utils.seed import seed_everything

F64 = torch.float64
PAPER_EXTENT = ((0.0, 10.0), (0.0, 10.0), (0.0, 1.0))


class _OpCounter(TorchDispatchMode):
    """Counts aten ops below autograd; views are counted separately (no kernel launch)."""

    def __init__(self) -> None:
        super().__init__()
        self.kernels = 0
        self.views = 0
        self.names: dict[str, int] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if getattr(func, "is_view", False):
            self.views += 1
        else:
            self.kernels += 1
            name = func.overloadpacket.__name__
            self.names[name] = self.names.get(name, 0) + 1
        return func(*args, **(kwargs or {}))


def _count(fn) -> _OpCounter:
    c = _OpCounter()
    with c:
        fn()
    return c


# ---------------------------------------------------------------------------------------------
# 1. heat stencil
# ---------------------------------------------------------------------------------------------
def _legacy_apply(T, alpha, spacing, bc, dt=None, robin_h=None):
    """Verbatim copy of the previous ``DiffusionStencil.apply`` / ``ImplicitSystem.apply``
    (``torch.roll`` neighbours + ``addcmul`` chain), the reference of the new flat layout."""
    nd = len(spacing)
    spec = BoundarySpec.parse(bc, nd, robin_h)
    conds = face_conductances(alpha, spacing, spec)
    coeffs = []
    for ax, cp in enumerate(conds):
        coeffs.append(cp)
        coeffs.append(torch.roll(cp, shifts=1, dims=ax))
    diag = -sum(coeffs)
    if spec.has_robin:
        beta = torch.zeros(alpha.shape, dtype=alpha.dtype)
        beta[..., -1] = spec.robin_h / spacing[-1]
        diag = diag - beta
    if dt is not None:  # A = I - dt L
        diag = 1.0 - dt * diag
        coeffs = [-dt * c for c in coeffs]
    nbs = []
    for ax in range(nd):
        dim = T.ndim - nd + ax
        nbs.append(torch.roll(T, shifts=-1, dims=dim))
        nbs.append(torch.roll(T, shifts=1, dims=dim))
    out = diag * T
    for c, n in zip(coeffs, nbs):
        out = torch.addcmul(out, c, n)
    return out


GRIDS = [(7,), (5, 4), (6, 5, 4), (4, 3, 2), (1, 5, 3), (2, 2, 3)]


@pytest.mark.parametrize("grid", GRIDS)
def test_flat_layout_neighbours_are_the_rolled_copies(grid):
    lay = flat_layout(grid)
    g = torch.Generator().manual_seed(0)
    for batch in ((), (3,)):
        x = torch.randn(*batch, *grid, generator=g, dtype=F64)
        P = lay.pad_grid(x)
        assert torch.equal(lay.from_range(lay.center(P)), x)
        assert torch.equal(lay.from_range(lay.to_range(x)), x)
        assert torch.equal(lay.pad_range(lay.to_range(x)), P)
        for j, ref in enumerate(neighbors(x, len(grid))):
            assert torch.equal(lay.from_range(lay.neighbor(P, j)), ref), (grid, batch, j)


BCS = [
    (None, 0.0),
    ("neumann", 0.0),
    ("periodic", 0.0),
    (("periodic", "periodic", "robin"), 0.7),
    (("neumann", "periodic", "robin"), 2.0),
]


@pytest.mark.parametrize("bc,h", BCS)
@pytest.mark.parametrize("dtype,rtol", [(F64, 1e-13), (torch.float32, 1e-6)])
def test_stencil_backends_match_previous_implementation(bc, h, dtype, rtol):
    g = torch.Generator().manual_seed(1)
    shape, sp, dt = (6, 5, 4), (0.3, 0.2, 0.1), 0.05
    alpha = (0.01 + 0.24 * torch.rand(shape, generator=g, dtype=F64)).to(dtype)
    T = torch.randn(2, *shape, generator=g, dtype=F64).to(dtype)  # a leading batch axis

    def close(a, b):
        scale = b.abs().max()
        assert float((a - b).abs().max() / scale) <= rtol, float((a - b).abs().max() / scale)

    ref_L = _legacy_apply(T, alpha, sp, bc, robin_h=h)
    ref_A = _legacy_apply(T, alpha, sp, bc, dt=dt, robin_h=h)
    for backend in ("flat", "roll", "auto"):
        st = DiffusionStencil(alpha, sp, bc, robin_h=h, backend=backend)
        close(st.apply(T), ref_L)
        sysm = st.system(dt)
        close(sysm.apply(T), ref_A)
        close(sysm.apply_offdiag(T), ref_A - sysm.diag * T)
    flat = DiffusionStencil(alpha, sp, bc, robin_h=h, backend="flat").system(dt)
    roll = DiffusionStencil(alpha, sp, bc, robin_h=h, backend="roll").system(dt)
    b = T.abs() + 1.0
    close(flat.jacobi_sweep(b * flat.inv_diag, T), roll.jacobi_sweep(b * roll.inv_diag, T))
    for k in (1, 2, 7):
        close(flat.jacobi_sweeps(b, T, k), roll.jacobi_sweeps(b, T, k))
        close(flat.chebyshev_sweeps(b, T, k), roll.chebyshev_sweeps(b, T, k))
    # the fused sweeps are the classical iteration x <- D^-1 (b - R x) (NeFTY Eq. 24)
    x = T
    for _ in range(3):
        x = (b - flat.apply_offdiag(x)) * flat.inv_diag
    close(flat.jacobi_sweeps(b, T, 3), x)


def test_float32_rollout_sweeps_match_roll_to_rounding():
    """20 sweeps on the smoke grid: flat vs roll differ only by float32 rounding (≤ 1e-6)."""
    g = torch.Generator().manual_seed(2)
    shape = (16, 16, 6)
    alpha = 0.003 + 0.247 * torch.rand(shape, generator=g)
    b = torch.rand(shape, generator=g) * 100
    sp = Domain(shape, PAPER_EXTENT).spacing()
    flat = DiffusionStencil(alpha, sp, backend="flat").system(0.1)
    roll = DiffusionStencil(alpha, sp, backend="roll").system(0.1)
    x_f, x_r = flat.jacobi_sweeps(b, b, 20), roll.jacobi_sweeps(b, b, 20)
    assert float((x_f - x_r).abs().max() / x_r.abs().max()) < 1e-6


def test_heat_operator_frames_and_adjoint_gradient_match_roll_reference():
    dom = Domain((8, 8, 5), ((0.0, 4.0), (0.0, 4.0), (0.0, 1.0)))
    g = torch.Generator().manual_seed(3)
    out = {}
    for backend in ("roll", "flat"):
        for solver in ("jacobi", "chebyshev"):
            op = HeatOperator(
                dom,
                dt=0.1,
                n_steps=6,
                inner_iters=12,
                solver=solver,
                initial=GaussianFlash(10.0, 1.0, 0.2),
                stencil_backend=backend,
            )
            a = (0.05 + 0.2 * torch.rand(dom.shape, generator=g)).requires_grad_(True)
            g = torch.Generator().manual_seed(3)  # same alpha for every variant
            fr = op({"alpha": a})
            (grad,) = torch.autograd.grad((fr**2).sum(), a)
            out[(backend, solver)] = (fr.detach(), grad)
    for solver in ("jacobi", "chebyshev"):
        (f0, g0), (f1, g1) = out[("roll", solver)], out[("flat", solver)]
        assert float((f1 - f0).abs().max() / f0.abs().max()) < 1e-6
        assert float((g1 - g0).abs().max() / g0.abs().max()) < 1e-5


def _sweep_kernels(backend: str, iters: int) -> _OpCounter:
    g = torch.Generator().manual_seed(4)
    shape = (8, 8, 6)
    alpha = 0.01 + 0.2 * torch.rand(shape, generator=g)
    b = torch.rand(shape, generator=g)
    sysm = DiffusionStencil(alpha, (0.5, 0.5, 0.2), backend=backend).system(0.05)
    sysm.jacobi_sweeps(b, b, 1)  # build the cached range-layout coefficients / workspace
    sysm.chebyshev_sweeps(b, b, 2)
    return _count(lambda: sysm.jacobi_sweeps(b, b, iters))


def test_heat_stencil_kernel_counts():
    """≤ 15 kernels per stencil application; 7 per Jacobi sweep (vs 12 with torch.roll)."""
    flat = {k: _sweep_kernels("flat", k) for k in (10, 20)}
    roll = {k: _sweep_kernels("roll", k) for k in (10, 20)}
    per_sweep_flat = (flat[20].kernels - flat[10].kernels) / 10
    per_sweep_roll = (roll[20].kernels - roll[10].kernels) / 10
    assert per_sweep_flat == 7 and per_sweep_roll == 12
    assert flat[20].views == flat[10].views  # the sweeps create no views at all
    g = torch.Generator().manual_seed(5)
    shape = (8, 8, 6)
    alpha = 0.01 + 0.2 * torch.rand(shape, generator=g)
    T = torch.rand(shape, generator=g)
    st = DiffusionStencil(alpha, (0.5, 0.5, 0.2))
    sysm = st.system(0.05)
    st.apply(T), sysm.apply(T), sysm.apply_offdiag(T)  # warm the caches
    for fn in (lambda: st.apply(T), lambda: sysm.apply(T), lambda: sysm.apply_offdiag(T)):
        assert _count(fn).kernels <= 15


def _thermal_step_kernels(**overrides) -> int:
    from nefi.instances.thermal_tomography import ThermalTomography

    inst = ThermalTomography(preset="smoke", **overrides)
    gt, meas = inst.make_measurement(seed=0)
    problem = inst.build_problem(meas)
    op = problem.operator
    alpha = gt["alpha"].clone().requires_grad_(True)

    def step():
        fr = op({"alpha": alpha})
        ((fr - meas.data) ** 2).mean().backward()

    step()
    return _count(step).kernels


def test_thermal_physics_step_has_at_least_2x_fewer_kernels():
    """Operator forward + adjoint backward of the smoke problem (30 frames, K = 20)."""
    roll = _thermal_step_kernels(stencil_backend="roll")
    flat = _thermal_step_kernels()
    # Chebyshev-accelerated Jacobi at equal accuracy (K = 13 vs 20 Jacobi sweeps, see below)
    cheb = _thermal_step_kernels(solver="chebyshev", jacobi_iters=13)
    assert roll / flat >= 1.6, (roll, flat)
    assert roll / cheb >= 2.0, (roll, cheb)


@pytest.fixture(scope="module")
def paper_regime_system():
    """NeFTY Tab. 5 time step / depth spacing (Δt = 0.05, Δz = 1/16) with a 1:20 defect."""
    dom = Domain((16, 16, 16), PAPER_EXTENT)
    a = torch.full(dom.shape, 0.2, dtype=F64)
    a[5:9, 5:9, 5:9] = 0.01
    sysm = DiffusionStencil(a, dom.spacing()).system(0.05)
    b = GaussianFlash(100.0, 2.5, 0.2)(dom)
    x_ref, _ = conjugate_gradient(sysm.apply, b, b, tol=1e-14, max_iter=2000, precond=sysm.diag)
    return sysm, b, x_ref


def test_chebyshev_matches_50_jacobi_sweeps_with_20_iterations(paper_regime_system):
    sysm, b, x_ref = paper_regime_system

    def err(x):
        return float((x - x_ref).norm() / x_ref.norm())

    rho = float(sysm.spectral_bound())
    assert 0.5 < rho < 1.0
    e_jac50 = err(sysm.jacobi_sweeps(b, b, 50))
    e_cheb20 = err(sysm.chebyshev_sweeps(b, b, 20))
    assert e_cheb20 <= e_jac50, (e_cheb20, e_jac50)
    # asymptotic rate of the Chebyshev semi-iteration: ((1 - sqrt(1 - ρ²)) / ρ) per iteration
    r = (1 - math.sqrt(1 - rho**2)) / rho
    assert err(sysm.chebyshev_sweeps(b, b, 30)) < 10 * r**30 * err(b)
    # K = 1 is one plain Jacobi sweep; weights converge to 2 / (1 + sqrt(1 - ρ²))
    assert torch.equal(sysm.chebyshev_sweeps(b, b, 1), sysm.jacobi_sweeps(b, b, 1))
    om = sysm.chebyshev_omegas(40)
    assert float(om[0]) == pytest.approx(2 / (2 - rho**2), rel=1e-12)
    assert float(om[-1]) == pytest.approx(2 / (1 + math.sqrt(1 - rho**2)), rel=1e-9)


def test_chebyshev_rollout_converges_to_the_same_frames():
    dom = Domain((16, 16, 8), PAPER_EXTENT)
    a = torch.full(dom.shape, 0.15)
    a[4:8, 4:8, 2:5] = 0.01
    kw = dict(dt=0.05, n_steps=10, initial=GaussianFlash(100.0, 2.5, 0.2))
    ref = HeatOperator(dom, solver="cg", cg_tol=1e-10, cg_max_iter=2000, **kw)({"alpha": a})
    jac = HeatOperator(dom, solver="jacobi", inner_iters=50, **kw)({"alpha": a})
    che = HeatOperator(dom, solver="chebyshev", inner_iters=20, **kw)({"alpha": a})
    e_j = float((jac - ref).abs().max() / ref.abs().max())
    e_c = float((che - ref).abs().max() / ref.abs().max())
    assert e_c < 1e-5 and e_j < 1e-5, (e_c, e_j)


# ---------------------------------------------------------------------------------------------
# 2. whole-step compile path
# ---------------------------------------------------------------------------------------------
TOY = {"n": 32, "hidden": 16, "depth": 2, "n_octaves": 3, "steps": (12, 12)}


def _toy_problem(seed: int = 0, data_seed: int = 0, **cfg):
    inst = Toy1D(**{**TOY, **cfg})
    _, meas = inst.make_measurement(data_seed)
    seed_everything(seed)
    return inst.build_problem(meas)


def _solve(problem, **kw):
    return nefi.Solver(problem, problem.curriculum, device="cpu", seed=0, **kw).run()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max())


def test_compile_and_autocast_options_are_validated():
    assert normalize_compile(False) is None and normalize_compile(None) is None
    assert normalize_compile(True) == "field" and normalize_compile("step") == "step"
    assert normalize_autocast(None) is None and normalize_autocast("bf16") == torch.bfloat16
    assert normalize_autocast("fp16") == torch.float16
    with pytest.raises(ConfigError):
        normalize_compile("graph")
    with pytest.raises(ConfigError):
        normalize_autocast("int8")
    prob = _toy_problem()
    cur = copy.deepcopy(prob.curriculum)
    cur.optim.optimizer = "lbfgs"
    with pytest.raises(ConfigError, match="L-BFGS"):
        nefi.Solver(prob, cur, device="cpu", autocast="fp16")


def test_compile_failure_falls_back_to_eager_with_warning(monkeypatch, caplog):
    def broken_compile(fn, **kw):
        def run(*a, **k):
            raise RuntimeError("no compiler on this box")

        return run

    ref = _solve(_toy_problem())
    monkeypatch.setattr(torch, "compile", broken_compile)
    for mode in ("field", "step"):
        with caplog.at_level(logging.WARNING, logger="nefi"):
            res = _solve(_toy_problem(), compile=mode)
        assert "continuing eagerly" in caplog.text
        assert torch.equal(res.fields["x"], ref.fields["x"])  # the eager path is untouched


@pytest.fixture
def _inductor_single_thread(monkeypatch):
    # inductor's OpenMP C++ kernels clash with a second OpenMP runtime on some macOS setups; the
    # single-threaded code path is what CPU tests need anyway
    import torch._inductor.config as ic

    monkeypatch.setattr(ic.cpp, "threads", 1)


@pytest.mark.slow
@pytest.mark.usefixtures("_inductor_single_thread")
def test_compiled_field_and_step_match_eager_without_recompiles():
    from torch._dynamo.utils import counters

    ref = _solve(_toy_problem())
    for mode in ("field", "step"):
        counters.clear()
        res = _solve(_toy_problem(), compile=mode)
        # one graph per curriculum stage: annealing progress never triggers a recompile
        assert counters["stats"]["unique_graphs"] == len(ref.stage_results)
        assert _rel(res.fields["x"], ref.fields["x"]) < 1e-5, mode
        h, h_ref = res.history["total"], ref.history["total"]
        assert abs(h[-1] - h_ref[-1]) <= 1e-5 * abs(h_ref[-1])


@pytest.mark.slow
@pytest.mark.usefixtures("_inductor_single_thread")
def test_compiled_step_runs_non_traceable_operators_eagerly():
    from nefi.instances.thermal_tomography import ThermalTomography

    inst = ThermalTomography(preset="smoke", steps=6, n_frames=6)
    _, meas = inst.make_measurement(seed=0)
    out = {}
    for mode in (False, "step"):
        seed_everything(0)
        prob = inst.build_problem(meas)
        assert not prob.operator.traceable
        out[mode] = nefi.Solver(prob, device="cpu", seed=0, compile=mode).run()
    assert _rel(out["step"].fields["alpha"], out[False].fields["alpha"]) < 1e-4


# ---------------------------------------------------------------------------------------------
# 3. mixed precision for the field only
# ---------------------------------------------------------------------------------------------
def test_bf16_and_fp16_autocast_on_cpu_stay_close_to_fp32():
    ref = _solve(_toy_problem(steps=(40, 40)))
    for mode, tol in (("bf16", 2e-2), ("fp16", 2e-2)):
        res = _solve(_toy_problem(steps=(40, 40)), autocast=mode)
        assert res.fields["x"].dtype == torch.float32
        assert _rel(res.fields["x"], ref.fields["x"]) < tol, mode
        d, d_ref = res.history["data_loss"][-1], ref.history["data_loss"][-1]
        assert abs(d - d_ref) < 0.05 * d_ref, (mode, d, d_ref)


def test_autocast_runs_the_mlp_in_bf16_and_the_physics_in_fp32(monkeypatch):
    prob = _toy_problem()
    seen: dict[str, set] = {"linear": set(), "operator": set()}
    lin = torch.nn.functional.linear

    def spy_linear(x, w, b=None):
        out = lin(x, w, b)
        seen["linear"].add(out.dtype)
        return out

    monkeypatch.setattr(torch.nn.functional, "linear", spy_linear)
    op_forward = prob.operator.forward

    def spy_op(fields):
        seen["operator"].add(fields["x"].dtype)
        assert not autocast_enabled("cpu")
        return op_forward(fields)

    monkeypatch.setattr(prob.operator, "forward", spy_op)
    _solve(prob, autocast="bf16")
    # trunk layers in bf16, the output projection in fp32; the operator only ever sees fp32
    assert seen["linear"] == {torch.bfloat16, torch.float32}
    assert seen["operator"] == {torch.float32}


# ---------------------------------------------------------------------------------------------
# 4. batched multi-measurement solving
# ---------------------------------------------------------------------------------------------
def _pairs_problems(build, pairs):
    probs = []
    for data_seed, seed in pairs:
        probs.append(build(seed, data_seed))
    return probs


PAIRS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_batched_equals_sequential_on_toy1d():
    build = lambda s, d: _toy_problem(seed=s, data_seed=d, steps=(20, 20))  # noqa: E731
    seq = [_solve(p) for p in _pairs_problems(build, PAIRS)]
    probs = _pairs_problems(build, PAIRS)
    bat = nefi.batch_invert(probs, probs[0].curriculum, device="cpu", seeds=[s for _, s in PAIRS])
    assert len(bat) == len(seq)
    for b, s in zip(bat, seq):
        assert _rel(b.fields["x"], s.fields["x"]) < 1e-5
        assert len(b.history["total"]) == len(s.history["total"]) == 40
        for key in ("total", "data_loss", "lr", "progress"):
            for x, y in zip(b.history[key], s.history[key]):
                assert abs(x - y) <= 1e-5 * max(abs(y), 1e-12), key
        assert [st["stop"] for st in b.stage_results] == [st["stop"] for st in s.stage_results]
        assert b.extra["batched"] and b.extra["batch_size"] == 4
        assert b.extra["loss_modes"] == ["vmap", "vmap"]


def test_batched_equals_sequential_on_deconvolution():
    from nefi.instances.deconvolution import Deconvolution

    inst = Deconvolution(n=16, hidden=16, depth=2, n_octaves=3, steps=(15, 15))
    meas = [inst.make_measurement(seed=d)[1] for d in (0, 1)]

    def build(seed, data_seed):
        seed_everything(seed)
        return inst.build_problem(meas[data_seed])

    pairs = [(0, 0), (1, 0), (1, 2)]
    seq = [_solve(p) for p in _pairs_problems(build, pairs)]
    probs = _pairs_problems(build, pairs)
    bat = nefi.batch_invert(probs, device="cpu", seeds=[s for _, s in pairs])
    for b, s in zip(bat, seq):
        assert _rel(b.fields["x"], s.fields["x"]) < 1e-5


def test_batched_first_update_is_bitwise_identical_to_sequential():
    cur = nefi.Curriculum([nefi.Stage("s", (32,), steps=1, lr=1e-2)])
    seq = nefi.Solver(_toy_problem(), cur, device="cpu").run()
    bat = nefi.batch_invert([_toy_problem()], cur, device="cpu")[0]
    assert torch.equal(bat.fields["x"], seq.fields["x"])


def test_batched_problems_stop_and_fail_independently():
    """Per-problem early stopping, and a problem whose loss is always NaN fails alone."""
    cur = nefi.Curriculum(
        [nefi.Stage("s", (32,), steps=60, lr=1e-2, anneal_fraction=0.2)], early_stop_patience=3
    )
    cur.early_stop_min_delta = 1e-3
    seq = [nefi.Solver(_toy_problem(seed=s), cur, device="cpu").run() for s in (0, 1)]
    bad = _toy_problem(seed=0)
    bad.measurement.data[3] = float("nan")
    probs = [_toy_problem(seed=0), bad, _toy_problem(seed=1)]
    res = nefi.batch_invert(
        probs, cur, device="cpu", seeds=[0, 0, 1], max_bad_steps=2, raise_on_error=False
    )
    assert "non-finite" in res[1].extra["error"] and res[1].stage_results[0]["stop"] == "failed"
    for r, s in zip((res[0], res[2]), seq):
        assert r.stage_results[0]["stop"] == s.stage_results[0]["stop"] == "early_stop"
        assert len(r.history["total"]) == len(s.history["total"]) < 60
        assert _rel(r.fields["x"], s.fields["x"]) < 1e-5
    with pytest.raises(SolverError, match=r"problem\(s\) \[1\] failed"):
        nefi.batch_invert([_toy_problem(), bad], cur, device="cpu", max_bad_steps=1)


def test_batch_invert_refuses_non_batchable_problems():
    from nefi.instances.thermal_tomography import ThermalTomography

    inst = ThermalTomography(preset="smoke", steps=2, n_frames=3)
    _, meas = inst.make_measurement(seed=0)
    with pytest.raises(ConfigError, match="batch axis"):
        nefi.batch_invert([inst.build_problem(meas)] * 2, device="cpu")
    probs = [_toy_problem(), _toy_problem(n=16)]
    with pytest.raises(ConfigError, match="domain"):
        nefi.batch_invert(probs, device="cpu")
    cur = copy.deepcopy(probs[0].curriculum)
    cur.optim.optimizer = "sgd"
    with pytest.raises(ConfigError, match="optimizer"):
        nefi.batch_invert([_toy_problem()], cur, device="cpu")


BATCHABLE_INSTANCES = [
    ("toy1d", {}),
    ("deconvolution", {"noise": "poisson"}),
    ("nv_relaxometry", {"mode": "F1"}),
    ("nv_relaxometry", {}),
    ("poisson_source", {}),
    ("diffraction_tomography", {}),
    ("holography", {}),
    ("sparse_view_ct", {}),
    ("current_density", {}),
]


def _smoke_instance(name: str, **overrides):
    """The instance with ``configs/<name>_smoke.yaml`` (+ overrides), as the CLI builds it."""
    from pathlib import Path

    from nefi.config import load_config
    from nefi.registry import build

    raw = load_config(Path(__file__).resolve().parents[1] / "configs" / f"{name}_smoke.yaml")
    if isinstance(raw.get("instance"), dict):
        cfg = {k: v for k, v in raw["instance"].items() if k != "type"}
    else:
        cfg = dict(raw.get("config") or {})
    return build("instance", {"type": name, **cfg, **overrides})


@pytest.mark.parametrize("name,cfg", BATCHABLE_INSTANCES)
def test_batchable_operators_accept_a_leading_batch_axis(name, cfg):
    inst = _smoke_instance(name, **cfg)
    _, meas = inst.make_measurement(seed=0)
    prob = inst.build_problem(meas)
    op = prob.operator
    assert op.batchable and not any(q.requires_grad for q in op.parameters())
    coords = prob.domain.coords()
    fields = []
    for b in range(3):
        seed_everything(b)
        prob.field.reset_parameters()
        with torch.no_grad():
            fields.append(prob.field(coords, 1.0))
    with torch.no_grad():
        batched = op({k: torch.stack([f[k] for f in fields]) for k in fields[0]})
        for b in range(3):
            ref = op(fields[b])
            # batched and per-sample FFT paths differ by float32 rounding (platform dependent),
            # so the tolerance is relative to the output scale rather than absolute
            scale = float(ref.abs().max())
            torch.testing.assert_close(batched[b], ref, rtol=1e-5, atol=1e-6 * scale + 1e-7)


def test_run_benchmark_batched_matches_sequential():
    from nefi.bench import run_benchmark

    inst = Toy1D(**TOY)
    kw = dict(n_samples=2, seeds=(0, 1), device="cpu", progress=False)
    seq = run_benchmark(inst, "neural,grid", **kw)
    bat = run_benchmark(inst, "neural,grid", batched=True, batch_size=3, **kw)
    assert bat.meta["batched"] and all(r["batch"] in (1, 3) for r in bat.rows)
    assert [(r["method"], r["sample"], r["seed"]) for r in bat.rows] == [
        (r["method"], r["sample"], r["seed"]) for r in seq.rows
    ]
    for a, b in zip(seq.rows, bat.rows):
        assert b["error"] is None and abs(a["mse"] - b["mse"]) <= 1e-5 * a["mse"]
    assert "batch" not in bat.metric_names


# ---------------------------------------------------------------------------------------------
# 5. sharded benchmarks
# ---------------------------------------------------------------------------------------------
def test_benchmark_units_partition_the_runs():
    from nefi.bench import benchmark_units, parse_shard

    full = benchmark_units(["a", "b"], 3, (0, 1))
    shards = [benchmark_units(["a", "b"], 3, (0, 1), f"{i}/4") for i in range(4)]
    assert sorted(u for s in shards for u in s) == sorted(full) and len(full) == 12
    assert max(len(s) for s in shards) - min(len(s) for s in shards) <= 1
    assert parse_shard(None) == (0, 1) and parse_shard("2/5") == (2, 5)
    for bad in ("3", "5/5", (-1, 2), "a/b"):
        with pytest.raises(ConfigError):
            parse_shard(bad)


def test_two_shards_merge_into_the_unsharded_table(tmp_path, caplog):
    from nefi.bench import BenchmarkResult, run_benchmark

    inst = Toy1D(**TOY)
    kw = dict(n_samples=3, seeds=(0, 1), device="cpu", progress=False)
    full = run_benchmark(inst, "neural,grid", **kw)
    parts = [
        run_benchmark(inst, "neural,grid", shard=(i, 2), out_dir=tmp_path, **kw) for i in (0, 1)
    ]
    assert [p.shard for p in parts] == [(0, 2), (1, 2)]
    assert sorted(f.name for f in tmp_path.iterdir()) == [
        "shard-000-of-002.json",
        "shard-001-of-002.json",
    ]
    merged = BenchmarkResult.merge(tmp_path)
    cols = ("method", "class", "sample", "seed", "mse", "psnr", "relative_error", "steps")
    assert [tuple(r[c] for c in cols) for r in merged.rows] == [
        tuple(r[c] for c in cols) for r in full.rows
    ]
    for a, b in zip(full.table(), merged.table()):
        assert all(a[k] == b[k] for k in ("method", "n", "mse", "psnr", "mse_ci", "psnr_ci"))
    assert merged.meta["merged_shards"] == [0, 1] and merged.shard is None
    with pytest.raises(ConfigError, match="missing"):
        BenchmarkResult.merge(tmp_path / "shard-000-of-002.json")
    with caplog.at_level(logging.WARNING, logger="nefi"):
        part = BenchmarkResult.merge([tmp_path / "shard-001-of-002.json"], strict=False)
    assert "missing [0]" in caplog.text and len(part.rows) == len(parts[1].rows)


def test_cli_bench_shards_and_merge(tmp_path, monkeypatch, capsys):
    from nefi.cli import main

    monkeypatch.setenv("NEFI_CLI", "argparse")
    args = ["bench", "toy1d", "--n", "2", "--seeds", "0,1", "--smoke", "--set", "n=32", "-q"]
    for i in (0, 1):
        assert main([*args, "--batched", "--shard", f"{i}/2", "--out", str(tmp_path)]) == 0
    assert main(["bench-merge", str(tmp_path), "-q"]) == 0
    assert "merged 2 shard(s) of 2 (8 runs)" in capsys.readouterr().out
    for name in ("summary.md", "summary.csv", "rows.csv", "benchmark.json"):
        assert (tmp_path / name).exists(), name
    assert main(["bench-merge", str(tmp_path / "nope")]) == 2


# ---------------------------------------------------------------------------------------------
# 6. micro-optimizations
# ---------------------------------------------------------------------------------------------
class _PreviousFourierFeatures(torch.nn.Module):
    """Verbatim previous implementation (no caches)."""

    def __init__(self, in_dim, n_octaves=12, include_input=True, annealed=True):
        super().__init__()
        self.in_dim, self.n_octaves = in_dim, n_octaves
        self.include_input, self.annealed = include_input, annealed
        f = math.pi * (2.0 ** torch.arange(n_octaves, dtype=torch.float32))
        self.register_buffer("freqs", f, persistent=False)

    def band_weights(self, progress=1.0, device=None, dtype=None):
        k = torch.arange(self.n_octaves, device=device, dtype=dtype or torch.float32)
        if not self.annealed:
            return torch.ones_like(k)
        beta = float(progress) * self.n_octaves
        return 0.5 * (1.0 - torch.cos(math.pi * torch.clamp(beta - k, 0.0, 1.0)))

    def forward(self, coords, progress=1.0):
        ang = coords.unsqueeze(-2) * self.freqs.to(coords.dtype).view(-1, 1)
        w = self.band_weights(progress, device=coords.device, dtype=coords.dtype).view(-1, 1)
        feats = torch.cat([torch.sin(ang) * w, torch.cos(ang) * w], dim=-1).flatten(-2)
        return torch.cat([coords, feats], dim=-1) if self.include_input else feats


@pytest.mark.parametrize(
    "dim,k,inc,ann", [(1, 5, True, True), (3, 12, False, True), (2, 4, True, False)]
)
def test_fourier_feature_caches_are_bitwise_identical(dim, k, inc, ann):
    from nefi.fields.encoding import FourierFeatures

    new, old = (
        FourierFeatures(dim, k, include_input=inc, annealed=ann),
        _PreviousFourierFeatures(dim, k, inc, ann),
    )
    for dtype in (torch.float32, F64):
        c = torch.rand(29, dim, dtype=dtype) * 2 - 1
        for p in [0.0, 1 / 3, 0.61803, 1.0, 0.5, 1.0, 1.0]:  # repeated values hit the caches
            ref = old(c, p)
            assert torch.equal(new(c, p), ref)
            assert torch.equal(new(c, torch.tensor(p, dtype=F64)), ref)
            assert torch.equal(new.band_weights(p), old.band_weights(p))
    c = torch.rand(7, dim)
    new(c, 0.5)
    c.mul_(0.5)  # in-place change of the coordinates invalidates the cache
    assert torch.equal(new(c, 0.5), old(c, 0.5))
    feats = new(c, 1.0)
    feats.add_(1.0)  # an in-place change of the returned features too
    assert torch.equal(new(c, 1.0), old(c, 1.0))
    assert copy.deepcopy(new)._feat_cache is None
    new.to(F64)
    assert new._sincos_cache is None


def test_encoding_cache_saves_ops_after_annealing():
    from nefi.fields.encoding import FourierFeatures

    enc = FourierFeatures(2, 8)
    c = torch.rand(64, 2)
    enc(c, 1.0)
    assert _count(lambda: enc(c, 1.0)).kernels == 0  # constant progress: served from the cache
    assert _count(lambda: enc(c, 0.3)).kernels <= 12  # annealing: a multiply + concatenation


def test_small_cpu_models_use_bitwise_identical_foreach_adam(monkeypatch):
    import nefi.solve.solver as solver_mod

    prob = _toy_problem()
    s = nefi.Solver(prob, device="cpu")
    opt = s._build_optimizer(list(prob.field.parameters()), prob.curriculum.stages[0])
    assert opt.defaults["foreach"] is True
    ref = _solve(_toy_problem())
    monkeypatch.setattr(solver_mod, "FOREACH_MAX_NUMEL", 0)  # per-tensor path
    assert torch.equal(_solve(_toy_problem()).fields["x"], ref.fields["x"])
