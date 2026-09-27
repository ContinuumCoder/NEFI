import torch

import nefi
from nefi.instances.toy1d import make_problem
from nefi.solve import Curriculum, GradNormBalancing, Stage


def test_gradnorm_rebalances_regularizer_weights():
    problem, gt, meas = make_problem(n=48, seed=0, hidden=16, depth=2, n_octaves=3, tv=1e-4)
    # deliberately mis-scaled regularizer: TV starts 1e-6 of the data term
    problem.losses.weights["tv"] = 1e-6
    cb = GradNormBalancing(
        shares={"tv": 0.2}, every=10, warmup=10, alpha=1.0, ema=0.0, bounds=(1e-6, 1e6)
    )
    cur = Curriculum([Stage(shape=(48,), steps=45, lr=3e-3)])
    solver = nefi.Solver(problem, cur, device="cpu", callbacks=[cb])
    solver.run()
    assert len(cb.log) >= 3
    w_tv = solver.stage_losses.weights["tv"]
    assert w_tv > 1e-6  # raised towards its 20 % gradient share
    assert solver.stage_losses.weights["data"] == 1.0  # anchor untouched
    # weighted gradient norms roughly follow the shares after the last rebalance
    last = cb.log[-1]
    g_data, g_tv = last["g/data"], last["g/tv"]
    ratio = (w_tv * g_tv) / (1.0 * g_data)
    assert 0.05 < ratio < 2.0, ratio


def test_gradnorm_bounds_cap_the_update():
    problem, gt, meas = make_problem(n=48, seed=0, hidden=16, depth=2, n_octaves=3, tv=1e-4)
    problem.losses.weights["tv"] = 1e-6
    cb = GradNormBalancing(shares={"tv": 0.2}, every=10, warmup=10, alpha=1.0, ema=0.0)
    cur = Curriculum([Stage(shape=(48,), steps=45, lr=3e-3)])
    solver = nefi.Solver(problem, cur, device="cpu", callbacks=[cb])
    solver.run()
    assert 1e-6 < solver.stage_losses.weights["tv"] <= 1e-6 * 1e3 + 1e-12


def test_gradnorm_is_noop_under_compile_step_and_records_nothing_before_warmup():
    problem, gt, meas = make_problem(n=32, seed=0, hidden=8, depth=2, n_octaves=2)
    cb = GradNormBalancing(every=5, warmup=100)
    cur = Curriculum([Stage(shape=(32,), steps=12, lr=3e-3)])
    nefi.Solver(problem, cur, device="cpu", callbacks=[cb]).run()
    assert cb.log == []
    assert torch.isfinite(torch.tensor(problem.losses.weights["tv"]))
