import pytest
import torch

import nefi
from nefi.config import config_hash, from_dict, to_dict
from nefi.instances.toy1d import Toy1D, Toy1DConfig, make_problem
from nefi.metrics import psnr
from nefi.registry import build, list_registered
from nefi.solve import Curriculum, FieldSnapshots, LoggingCallback, OptimConfig, Stage


def test_curriculum_schedules():
    s = Stage(steps=100, lr=1e-2, lr_schedule="cosine", lr_min_ratio=0.1, anneal_fraction=0.5)
    assert abs(s.lr_at(0) - 1e-2) < 1e-12
    assert abs(s.lr_at(100) - 1e-3) < 1e-12
    assert s.progress_at(0) == 0.0 and s.progress_at(25) == 0.5 and s.progress_at(80) == 1.0
    st = Stage(steps=3000, lr=1.0, lr_schedule="step", lr_step_size=1000, lr_gamma=0.1)
    assert abs(st.lr_at(999) - 1.0) < 1e-12 and abs(st.lr_at(1000) - 0.1) < 1e-12
    c = Curriculum.multiscale((64, 64), n_stages=2, steps=(30, 70), lr=1e-3, lr_decay=0.5)
    assert [s.shape for s in c.stages] == [(32, 32), (64, 64)]
    assert [s.lr for s in c.stages] == [1e-3, 5e-4] and c.total_steps == 100
    assert c.scaled(0.1).total_steps == 10
    d = to_dict(c)
    c2 = from_dict(Curriculum, d)
    assert c2.stages[1].shape == (64, 64) and config_hash(c) == config_hash(c2)


def test_solver_toy1d_recovers_signal():
    problem, gt, meas = make_problem(n=64, seed=0, steps=(60, 90), hidden=48, depth=3, n_octaves=5)
    snaps = FieldSnapshots(every=50)
    res = nefi.invert(problem, callbacks=[LoggingCallback(every=1000), snaps], device="cpu", seed=0)
    assert res.fields["x"].shape == (64,)
    assert res.pred.shape == (64,)
    assert len(res.history["total"]) == 150
    assert res.history["stage"][0] == 0 and res.history["stage"][-1] == 1
    assert res.history["total"][-1] < res.history["total"][0]
    assert len(res.stage_results) == 2 and res.stage_results[0]["shape"] == (32,)
    assert len(snaps.snapshots) >= 2
    p = psnr(res.fields["x"], gt["x"])
    assert p > 15.0, p
    assert "Result" in res.summary()


@pytest.mark.slow
def test_solver_toy1d_full_quality():
    out = Toy1D(n=128, scene="bumps").run(seed=0, device="cpu")
    assert out.metrics["psnr"] > 24.0, out.metrics
    out = Toy1D(n=128, scene="mixed").run(seed=0, device="cpu")
    assert out.metrics["psnr"] > 15.0, out.metrics  # sub-cell spikes are band-limited by the blur


def test_solver_restarts_early_stop_and_lbfgs():
    problem, gt, meas = make_problem(n=32, seed=0, hidden=16, depth=2, n_octaves=3)
    cur = Curriculum(
        [Stage(shape=(32,), steps=40, lr=3e-3)],
        restarts=2,
        early_stop_patience=5,
        early_stop_min_delta=1e-12,
    )
    res = nefi.invert(problem, cur, device="cpu")
    assert res.fields["x"].shape == (32,)
    cur2 = Curriculum(
        [Stage(shape=(32,), steps=5, lr=0.5, lr_schedule="constant")],
        optim=OptimConfig(optimizer="lbfgs", lbfgs_max_iter=5, grad_clip=None),
    )
    res2 = nefi.invert(problem, cur2, device="cpu")
    assert len(res2.history["total"]) == 5
    cur3 = Curriculum(
        [Stage(shape=(32,), steps=30, lr=1e-2)], optim=OptimConfig(ema=0.9), time_budget_s=1e-9
    )
    res3 = nefi.invert(problem, cur3, device="cpu")
    assert res3.stage_results[0]["stop"] == "time_budget"


def test_discrepancy_stopping():
    problem, gt, meas = make_problem(n=32, seed=0, hidden=16, depth=2, n_octaves=3, noise_std=0.05)
    assert meas.noise_std is not None and meas.noise_std > 0
    # stops only once annealing has finished (anneal_fraction=0.25 -> step 100), never before
    cur = Curriculum(
        [Stage(shape=(32,), steps=400, lr=3e-3, anneal_fraction=0.25)], discrepancy_tau=50.0
    )
    res = nefi.invert(problem, cur, device="cpu")
    assert res.stage_results[0]["stop"] == "discrepancy"
    assert 100 <= res.stage_results[0]["steps"] < 400
    assert res.stage_results[0]["final_progress"] == 1.0


def test_early_stage_stop_evaluates_at_trained_progress():
    from nefi.solve import Callback

    class StopAt(Callback):
        def on_step(self, solver, state):
            if state.step == 49:
                solver.stop_stage = "test"

    problem, gt, meas = make_problem(n=32, seed=0, hidden=16, depth=2, n_octaves=3)
    cur = Curriculum([Stage(shape=(32,), steps=200, lr=3e-3)])
    res = nefi.invert(problem, cur, device="cpu", callbacks=[StopAt()])
    info = res.stage_results[0]
    assert info["stop"] == "test" and info["steps"] == 50
    assert abs(info["final_progress"] - 49 / 200) < 1e-9
    _, pred_trained = problem.evaluate((32,), progress=49 / 200)
    _, pred_full = problem.evaluate((32,), progress=1.0)
    assert torch.allclose(res.pred, pred_trained.cpu(), atol=1e-6)
    assert not torch.allclose(res.pred, pred_full.cpu(), atol=1e-4)


def test_instance_registry_and_run(tmp_path):
    assert "toy1d" in list_registered("instance")["instance"]
    inst = build(
        "instance",
        {"type": "toy1d", "n": 32, "steps": [20, 20], "hidden": 16, "depth": 2, "n_octaves": 3},
    )
    assert isinstance(inst.cfg, Toy1DConfig) and inst.cfg.n == 32
    out = inst.run(seed=0, device="cpu")
    assert set(out.metrics) == {"mse", "psnr", "relative_error"}
    out.result.save(tmp_path / "r.pt")
    r2 = nefi.Result.load(tmp_path / "r.pt")
    assert torch.allclose(r2.fields["x"], out.result.fields["x"])
    prob, cur = inst.baselines()["grid"](out.measurement)
    r3 = nefi.invert(prob, cur.scaled(0.5), device="cpu")
    assert r3.fields["x"].shape == (32,)


def test_measurement_resampled_and_problem_measurement_at():
    problem, gt, meas = make_problem(n=32, seed=0, hidden=16, depth=2, n_octaves=3)
    m16 = problem.measurement_at((16,))
    assert m16.shape == (16,)
    assert (
        abs(float(m16.data.mean()) - float(meas.data.mean())) < 1e-5
    )  # area averaging conserves mean
    fields, pred = problem.evaluate((16,))
    assert pred.shape == (16,) and fields["x"].shape == (16,)
    total, comps = problem.loss((16,))
    assert total.requires_grad


def test_progress_override_is_respected_at_evaluation():
    from nefi.solve import Callback

    class HoldProgress(Callback):
        def on_stage_start(self, solver, stage_idx, stage):
            solver.progress_override = 0.25

    problem, gt, meas = make_problem(n=32, seed=0, hidden=16, depth=2, n_octaves=3)
    cur = Curriculum([Stage(shape=(32,), steps=40, lr=3e-3)])
    res = nefi.invert(problem, cur, device="cpu", callbacks=[HoldProgress()])
    info = res.stage_results[0]
    assert info["stop"] == "completed" and abs(info["final_progress"] - 0.25) < 1e-12
    assert all(abs(p - 0.25) < 1e-12 for p in res.history["progress"])
    _, pred_trained = problem.evaluate((32,), progress=0.25)
    assert torch.allclose(res.pred, pred_trained.cpu(), atol=1e-6)


def test_measurement_resampled_keeps_fractional_mask_weights():
    from nefi.measurement import Measurement

    data = torch.arange(16.0).reshape(4, 4)
    mask = torch.zeros(4, 4)
    mask[0, 0] = 1.0  # a single observed pixel
    m = Measurement(data, mask).resampled((2, 2))
    assert m.mask is not None and abs(float(m.mask[0, 0]) - 0.25) < 1e-6
    assert float(m.mask.sum()) > 0  # sparse masks survive coarsening
    assert abs(float(m.data[0, 0]) - 0.0) < 1e-6  # observed-weighted average = the observed value
    assert float(m.data[1, 1]) == 0.0  # unobserved coarse cells carry no data
    assert abs(float(m.masked_mean(m.data)) - 0.0) < 1e-6


def test_lr_multipliers_by_prefix():
    problem, gt, meas = make_problem(n=32, seed=0, hidden=16, depth=2, n_octaves=3)
    cur = Curriculum(
        [Stage(shape=(32,), steps=3, lr=1e-3, lr_schedule="constant")],
        optim=OptimConfig(lr_mult={"field.out": 5.0}),
    )
    solver = nefi.Solver(problem, cur, device="cpu")
    solver.run()
    lrs = sorted({round(g["lr"], 12) for g in solver.optimizer.param_groups})
    assert lrs == [1e-3, 5e-3]


def test_measurement_resampled_batches_leading_dims_and_problem_to_keeps_complex():
    from nefi.measurement import Measurement
    from nefi.operators import LambdaOperator

    data = torch.rand(5, 3, 8, 8)  # (time, species, H, W): only the trailing 2 dims change
    m = Measurement(data).resampled((5, 3, 4, 4))
    assert m.shape == (5, 3, 4, 4)
    assert torch.allclose(m.data.mean(dim=(-1, -2)), data.mean(dim=(-1, -2)), atol=1e-6)
    from nefi.errors import ShapeError

    m2 = Measurement(data).resampled((4, 4))  # trailing shape: leading dims are batch
    assert m2.shape == (5, 3, 4, 4)
    with pytest.raises(ShapeError):
        Measurement(data).resampled((1, 5, 3, 4, 4))

    problem, gt, meas = make_problem(n=16, seed=0, hidden=8, depth=2, n_octaves=2)
    op = LambdaOperator(lambda f: f["x"], homogeneity=1)
    op.register_buffer("kernel_c", torch.ones(4, dtype=torch.complex64))
    problem.operator = op
    problem.to("cpu", torch.float64)
    assert op.kernel_c.dtype == torch.complex64
    assert next(problem.field.parameters()).dtype == torch.float64


def test_masked_mean_with_broadcast_mask():
    from nefi.measurement import Measurement

    data = torch.ones(3, 4, 4)
    mask = torch.zeros(4, 4)
    mask[0, 0] = 1.0  # broadcast over the leading (batch) dim -> 3 observed entries
    m = Measurement(data, mask)
    x = torch.zeros(3, 4, 4)
    x[:, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    assert abs(float(m.masked_mean(x)) - 2.0) < 1e-6
