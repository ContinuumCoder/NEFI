"""Benchmark protocol, inverse-crime guard, ablations, sweeps and reports (nefi.bench)."""

import logging
import math
import statistics

import pytest

from nefi.bench import (
    BenchmarkResult,
    InverseCrimeError,
    Method,
    check_inverse_crime,
    cumulative_ablation,
    drop_loss,
    mean_ci,
    metric_direction,
    no_annealing,
    no_positional_encoding,
    parse_modifier,
    parse_value,
    run_benchmark,
    runtime_table,
    scale_budget,
    set_stage,
    single_stage,
    sweep,
    variant_method,
)
from nefi.errors import ConfigError
from nefi.instances.toy1d import Toy1D, make_problem

TINY = {"n": 32, "steps": (15, 15), "hidden": 16, "depth": 2, "n_octaves": 3}
FAST = {"device": "cpu", "progress": False}


class MatchedToy1D(Toy1D):
    """Toy1D whose data generator claims the inversion operator's fidelity (inverse crime)."""

    def data_generator(self):
        gen = super().data_generator()
        gen.fidelity_tag = "FFTConvolution"
        return gen


def test_benchmark_table_ci_runtime_and_files(tmp_path):
    inst = Toy1D(**TINY)
    res = run_benchmark(inst, "neural,grid", n_samples=2, seeds=(0,), out_dir=tmp_path, **FAST)
    assert len(res.rows) == 4 and not any(r["error"] for r in res.rows)
    assert res.methods == ["neural", "grid"] and res.metric_names == [
        "mse",
        "psnr",
        "relative_error",
    ]
    assert res.regime == {"neural": "cross-fidelity", "grid": "cross-fidelity"}
    assert res.fidelity["data"] == "gaussian-blur-4x-float64"
    tab = {r["method"]: r for r in res.table()}
    row = tab["neural"]
    vals = [r["psnr"] for r in res.rows if r["method"] == "neural"]
    assert row["n"] == 2 and row["failed"] == 0
    assert abs(row["psnr"] - statistics.fmean(vals)) < 1e-9
    t975 = 12.706204736  # Student t quantile, 1 degree of freedom
    assert abs(row["psnr_ci"] - t975 * statistics.stdev(vals) / math.sqrt(2)) < 1e-6
    assert all(r["time_s"] > 0 and r["steps"] == 30 for r in res.rows)
    md = res.to_markdown()
    for token in (
        "psnr ↑",
        "mse ↓",
        " ± ",
        "time (s)",
        "peak mem (MB)",
        "cross-fidelity",
        "| grid |",
    ):
        assert token in md, token
    assert "**" in md  # best entry in bold
    assert res.best("psnr") in ("neural", "grid")
    for name in ("summary.md", "summary.csv", "rows.csv", "benchmark.json"):
        assert (tmp_path / name).exists(), name
    back = BenchmarkResult.load(tmp_path)
    assert back.methods == res.methods and len(back.rows) == 4
    assert back.table()[0]["psnr"] == pytest.approx(res.table()[0]["psnr"])
    assert "ms / step" in res.efficiency_table()
    assert res.to_csv().count("\n") == 5 and res.to_json().startswith("{")


def test_inverse_crime_guard(caplog):
    inst = MatchedToy1D(**TINY)
    with pytest.raises(InverseCrimeError, match="inverse crime"):
        run_benchmark(inst, "neural", n_samples=1, seeds=(0,), **FAST)
    with caplog.at_level(logging.WARNING, logger="nefi"):
        res = run_benchmark(
            inst, "neural", n_samples=1, seeds=(0,), allow_inverse_crime=True, **FAST
        )
    assert "MATCHED-OPERATOR" in caplog.text
    assert res.regime == {"neural": "matched"} and "MATCHED-OPERATOR" in res.to_markdown()
    problem, _, _ = make_problem(n=16, hidden=8, depth=2, n_octaves=2)
    assert check_inverse_crime("gaussian-blur-4x-float64", problem.operator) == "cross-fidelity"
    with pytest.raises(AssertionError):  # InverseCrimeError is an AssertionError
        check_inverse_crime("FFTConvolution", problem.operator)
    problem.operator.fidelity_tag = "custom-tag"
    assert check_inverse_crime("FFTConvolution", problem.operator) == "cross-fidelity"


def test_failed_runs_are_recorded():
    def broken(instance, measurement):
        raise RuntimeError("boom")

    inst = Toy1D(**TINY)
    methods = [Method("neural"), Method("broken", broken)]
    res = run_benchmark(inst, methods, n_samples=1, seeds=(0,), **FAST)
    rows = {r["method"]: r for r in res.rows}
    assert rows["neural"]["error"] is None and "boom" in rows["broken"]["error"]
    assert "0 (1 failed)" in res.to_markdown()
    with pytest.raises(RuntimeError):
        run_benchmark(inst, methods, n_samples=1, seeds=(0,), fail_fast=True, **FAST)


def test_cumulative_ablation_and_modifiers():
    inst = Toy1D(**TINY)
    res = cumulative_ablation(
        inst, [no_annealing(), ("−PE", no_positional_encoding())], n_samples=1, seeds=(0,), **FAST
    )
    assert res.methods == ["full", "−annealing", "−PE"]
    steps = res.meta["ablation"]["steps"]
    assert steps[2] == ["−PE", ["no_annealing", "no_positional_encoding"]]
    assert all(r["error"] is None for r in res.rows)
    # the modifiers really change the method
    m = variant_method(
        inst, "all", [no_annealing(), no_positional_encoding(), single_stage(), drop_loss("tv")]
    )
    _, meas = inst.make_measurement(0)
    prep = m.prepare(inst, meas)
    assert type(prep.problem.field.encoding).__name__ == "IdentityEncoding"
    assert len(prep.curriculum.stages) == 1 and not prep.curriculum.stages[0].anneal
    assert prep.curriculum.total_steps == 30 and prep.curriculum.stages[0].shape == (32,)
    assert prep.problem.losses.weights["tv"] == 0.0
    prep2 = variant_method(inst, "cfg", ["set:hidden=8", set_stage(lr=0.5)]).prepare(inst, meas)
    assert prep2.problem.field.hidden == 8 and all(s.lr == 0.5 for s in prep2.curriculum.stages)


def test_sweep_over_lr():
    inst = Toy1D(**TINY)
    res = sweep(inst, "lr", [1e-3, 1e-2], n_samples=1, seeds=(0,), **FAST)
    assert res.methods == ["lr=0.001", "lr=0.01"]
    assert res.meta["sweep"]["axis"] == "lr" and res.meta["sweep"]["values"] == [1e-3, 1e-2]
    assert all(r["error"] is None for r in res.rows)
    with pytest.raises(ConfigError, match="unknown sweep axis"):
        sweep(inst, "not_a_field", [1, 2], n_samples=1, seeds=(0,), **FAST)


def test_helpers_and_runtime_table():
    m, hw, n = mean_ci([1.0, 2.0, 3.0])
    assert m == 2.0 and n == 3 and abs(hw - 4.302652730 / math.sqrt(3)) < 1e-6
    assert math.isnan(mean_ci([5.0])[1]) and mean_ci([float("nan"), 1.0])[2] == 1
    assert metric_direction("psnr") and metric_direction("hungarian_f1")
    assert metric_direction("relative_error") is False and metric_direction("gmsd") is False
    assert parse_value("1e-3") == 1e-3 and parse_value("[1, 2]") == [1, 2]
    assert parse_value("none") is None and parse_value("true") is True and parse_value("x") == "x"
    assert parse_modifier("drop_loss:tv").name == "drop_loss(tv)"
    assert parse_modifier("set_weight:tv=0.5").label == "tv=0.5"
    assert parse_modifier("set:n_octaves=4").config(Toy1D().cfg) == {"n_octaves": 4}
    with pytest.raises(ConfigError):
        parse_modifier("bogus")
    cur = Toy1D().default_curriculum()
    small = scale_budget(cur, 1.0, max_total_steps=40, min_stage_steps=2)
    assert small.total_steps <= 41 and cur.total_steps == 900
    problem, _, _ = make_problem(n=32, hidden=8, depth=2, n_octaves=2)
    tab = runtime_table({"a": problem, "b": problem}, n_repeat=2, reference="a")
    md = tab.to_markdown()
    assert "fwd time (s)" in md and "bwd time (s)" in md and "sim. error" in md
    assert tab.rows[1]["sim_error"] == 0.0 and tab.rows[0]["bwd_s_mean"] > 0
