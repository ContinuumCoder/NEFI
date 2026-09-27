"""The ``nefi`` CLI, invoked in-process through ``nefi.cli.main`` with both front-ends."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from nefi.cli import (
    COMMANDS,
    build_parser,
    config_files,
    load_spec,
    main,
    make_instance,
    read_spec_file,
    split_list,
)
from nefi.config import from_dict
from nefi.registry import list_registered
from nefi.solve.curriculum import Curriculum

BACKENDS = ["argparse"] + (["typer"] if importlib.util.find_spec("typer") else [])
SMALL = ["--set", "n=32", "-q"]


@pytest.fixture(params=BACKENDS)
def cli(request, monkeypatch):
    monkeypatch.setenv("NEFI_CLI", request.param)
    return main


def test_list(cli, capsys):
    assert cli(["list"]) == 0
    out = capsys.readouterr().out
    assert "toy1d" in out and "fields:" in out and "operators:" in out and "losses:" in out
    assert cli(["list", "instances", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert "toy1d" in info["instance"]
    assert cli(["list", "metrics"]) == 0 and "psnr" in capsys.readouterr().out
    assert cli(["list", "configs"]) == 0 and "toy1d.yaml" in capsys.readouterr().out


def test_run_smoke_writes_files_and_reloads(cli, tmp_path, capsys):
    out = tmp_path / "run"
    assert cli(["run", "toy1d", "--smoke", "--out", str(out), *SMALL]) == 0
    for name in ("result.pt", "config.yaml", "metrics.json", "history.csv"):
        assert (out / name).exists(), name
    metrics = json.loads((out / "metrics.json").read_text())
    assert set(metrics["metrics"]) == {"mse", "psnr", "relative_error"}
    assert metrics["data_fit"]["data_psnr"] > 0
    header = (out / "history.csv").read_text().splitlines()[0].split(",")
    assert {"step", "total", "data_loss", "lr"} <= set(header)
    assert "saved" in capsys.readouterr().out
    # the saved config reproduces the run (layout A)
    spec = read_spec_file(out / "config.yaml")
    assert spec.instance == "toy1d" and spec.config["n"] == 32
    assert cli(["run", str(out / "config.yaml"), "--out", str(tmp_path / "again"), "-q"]) == 0
    m2 = json.loads((tmp_path / "again" / "metrics.json").read_text())["metrics"]
    assert m2["psnr"] == pytest.approx(metrics["metrics"]["psnr"], rel=1e-4)


def test_run_baseline_and_plot(cli, tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "grid"
    assert (
        cli(["run", "toy1d", "--smoke", "--baseline", "grid", "--plot", "-o", str(out), *SMALL])
        == 0
    )
    assert (out / "fields.png").exists() and (out / "history.png").exists()
    assert json.loads((out / "metrics.json").read_text())["method"] == "grid"


def test_bench_smoke(cli, tmp_path, capsys):
    out = tmp_path / "bench"
    args = ["bench", "toy1d", "--n", "1", "--seeds", "0", "--smoke", "--out", str(out), *SMALL]
    assert cli(args) == 0
    text = capsys.readouterr().out
    assert "| neural |" in text and "| grid |" in text
    neural_row = next(line for line in text.splitlines() if line.startswith("| neural |"))
    assert "±" not in neural_row  # a single run has no confidence interval
    for name in ("summary.md", "summary.csv", "rows.csv", "benchmark.json"):
        assert (out / name).exists(), name


def test_diagnose_smoke(cli, tmp_path, capsys):
    out = tmp_path / "diag"
    assert cli(["diagnose", "toy1d", "--smoke", "--out", str(out), *SMALL]) == 0
    assert "Diagnostics" in capsys.readouterr().out
    for name in ("diagnostics.md", "diagnostics.json", "diagnostics_tensors.pt"):
        assert (out / name).exists(), name


def test_ablate_and_sweep(cli, tmp_path, capsys):
    common = ["--n", "1", "--seeds", "0", "--smoke", *SMALL]
    args = ["ablate", "toy1d", "--variants", "no_annealing,drop_loss:tv", "-o", str(tmp_path / "a")]
    assert cli([*args, *common]) == 0
    text = capsys.readouterr().out
    assert "| full |" in text and "| −annealing |" in text and "| −tv |" in text
    args = ["sweep", "toy1d", "--axis", "lr", "--values", "1e-3,1e-2", "-o", str(tmp_path / "s")]
    assert cli([*args, *common]) == 0
    assert "lr=0.01" in capsys.readouterr().out


def test_errors_are_friendly(cli, capsys):
    assert cli(["run", "no_such_instance"]) == 2
    err = capsys.readouterr().err
    assert "unknown instance" in err and "toy1d" in err
    assert cli(["run", "missing_config.yaml"]) == 2
    assert "not found" in capsys.readouterr().err
    assert cli(["bench", "toy1d", "--methods", "bogus", "--smoke", "-q"]) == 2
    assert "unknown method" in capsys.readouterr().err
    assert cli(["run", "toy1d", "--set", "not_a_field=1", "-q"]) == 2
    assert cli(["run", "--help"]) == 0
    assert cli(["--version"]) == 0
    assert cli(["sweep", "toy1d"]) == 2  # missing required --axis / --values


def test_spec_parsing_and_configs(tmp_path):
    assert split_list("a, set:steps=[1,2],b") == ["a", "set:steps=[1,2]", "b"]
    spec = load_spec(
        "toy1d", ["n=16", "stage.lr=0.1", "solver.compile=false", "curriculum.restarts=2"]
    )
    assert spec.config["n"] == 16 and spec.solver == {"compile": False}
    assert spec.curriculum_overrides == {"stage.lr": 0.1, "curriculum.restarts": 2}
    smoke = load_spec("toy1d", [], smoke=True)
    assert smoke.smoke is not None
    # layout B (instance mapping with a type key) is accepted too
    p = tmp_path / "b.yaml"
    p.write_text(yaml.safe_dump({"instance": {"type": "toy1d", "n": 24}, "run": {"seed": 3}}))
    s = read_spec_file(p)
    assert s.instance == "toy1d" and s.config == {"n": 24} and s.run["seed"] == 3
    # every shipped config must be loadable (registered instance, valid config and curriculum)
    registered = set(list_registered("instance")["instance"])
    for f in config_files():
        spec = read_spec_file(f)
        if spec.instance not in registered:  # instance package not present in this checkout
            continue
        inst = make_instance(spec)
        assert inst.cfg is not None, f
        if spec.curriculum:
            assert isinstance(from_dict(Curriculum, spec.curriculum), Curriculum)
    assert Path("configs/toy1d.yaml").exists()


def test_backends_define_the_same_commands():
    parser = build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    assert set(sub.choices) == {c.name for c in COMMANDS}
    if "typer" in BACKENDS:
        from typer.main import get_command

        from nefi.cli import build_typer_app

        group = get_command(build_typer_app())
        assert set(group.commands) == {c.name for c in COMMANDS}
        for c in COMMANDS:
            names = {p.name for p in group.commands[c.name].params}
            assert names == {o.name for o in c.opts}, c.name


PLUGIN = """
from nefi.instances.toy1d import Toy1D
from nefi.registry import register


@register("instance", "plugin_toy")
class PluginToy(Toy1D):
    name = "plugin_toy"
    description = "instance defined outside nefi (plugin test)"
    smoke_overrides = {"n": 16, "hidden": 8, "depth": 2, "n_octaves": 2}
"""


def test_plugins_env_and_config_imports(cli, tmp_path, monkeypatch, capsys):
    plugin = tmp_path / "my_plugin.py"
    plugin.write_text(PLUGIN)
    monkeypatch.setenv("NEFI_PLUGINS", str(plugin))
    assert cli(["list", "instances"]) == 0
    assert "plugin_toy" in capsys.readouterr().out
    monkeypatch.delenv("NEFI_PLUGINS")
    cfg = tmp_path / "plugin.yaml"
    cfg.write_text("imports: [my_plugin.py]\ninstance: plugin_toy\nconfig: {steps: [5, 5]}\n")
    assert cli(["run", str(cfg), "--smoke", "-q", "-o", str(tmp_path / "p")]) == 0
    assert (tmp_path / "p" / "metrics.json").exists()
    monkeypatch.setenv("NEFI_PLUGINS", "definitely_not_a_module_xyz")
    assert cli(["list"]) == 2
