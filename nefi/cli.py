"""``nefi`` command-line interface.

Commands (``nefi <command> --help`` for the options)::

    nefi list [KIND] [-v] [--json]              registered instances / fields / operators / ...
    nefi run TARGET [--smoke] [--plot] ...      one inversion; saves result.pt, config.yaml,
                                                metrics.json, history.csv (+ PNGs with --plot)
    nefi bench TARGET --methods neural,grid     paired benchmark with 95 % CIs (markdown + files)
    nefi bench TARGET --batched --shard 0/4     batched solving / one shard of a cluster campaign
    nefi bench-merge DIR                        merge the shard files of DIR into one table
    nefi diagnose TARGET [--solve]              ill-posedness diagnostics report
    nefi autotune TARGET [--level L] [--compare] auto-tune σ, gauges, budget/LR, regularization;
                                                report + tuned_config.yaml (+ acquisition report)
    nefi ablate TARGET --variants a,b,c         cumulative ablation
    nefi sweep TARGET --axis lr --values ...    one-axis hyperparameter sweep

``TARGET`` is a registered instance name (``nefi list instances``) or a config file. Two YAML
layouts are accepted::

    instance: toy1d                  # layout A              instance:          # layout B
    config: {n: 64, lr: 3.0e-3}                                type: toy1d
    curriculum: {stages: [...]}      # optional                n: 64
    solver: {compile: true}          # optional (Solver kw)  run: {seed: 0, device: auto}
    run: {seed: 0, scene: bumps}     # optional CLI defaults

``--set key=value`` overrides config fields (``--set n=64``), curriculum settings
(``stage.lr=1e-3``, ``curriculum.restarts=2``, ``optim.weight_decay=0``) or solver options
(``solver.compile=true``). ``--smoke`` shrinks the problem for a quick local check: it applies, in
this order of preference, the instance's ``smoke_overrides`` (class attribute or method),
``configs/<instance>_smoke.yaml``, or a ``"smoke"`` entry of the instance module's ``PRESETS``, and
always caps the total number of optimization steps (``--smoke-steps``).

Instances defined outside ``nefi.instances`` are made visible by importing the module that
registers them: list it in ``NEFI_PLUGINS`` (comma-separated module names or ``.py`` paths), in
the ``imports:`` key of a config file, or expose it as a ``nefi.plugins`` entry point of an
installed package.

The interface uses ``typer`` when it is installed and falls back to ``argparse`` with the same
commands and options (force one with ``NEFI_CLI=argparse`` / ``NEFI_CLI=typer``).
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
import time
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

import nefi

from .config import config_hash, from_dict, load_config, save_config, to_dict
from .errors import NefiError
from .registry import get as registry_get
from .registry import list_registered
from .solve.curriculum import Curriculum
from .utils.seed import seed_everything

log = logging.getLogger("nefi")

REQUIRED: Any = ...
KIND_ORDER = (
    "instance",
    "field",
    "operator",
    "loss",
    "baseline",
    "metric",
    "head",
    "encoding",
    "postprocess",
    "callback",
    "scene",
    "datagen",
)


class CLIError(NefiError):
    """User-facing command-line error (bad target, missing file, malformed option)."""


# ==========================================================================================
# config / target resolution
# ==========================================================================================
@dataclass
class RunSpec:
    """Everything a command needs to build an instance and its solver settings."""

    instance: str
    config: dict[str, Any] = field(default_factory=dict)
    curriculum: dict[str, Any] | None = None
    solver: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)
    curriculum_overrides: dict[str, Any] = field(default_factory=dict)
    source: str = "registry"
    smoke: str | None = None


CLI_DESCRIPTION = (
    "nefi recovers hidden physical fields from a single measurement, with a differentiable model "
    "of the instrument and no training data."
)


def config_dirs() -> list[Path]:
    """Directories searched for ``<instance>_smoke.yaml`` and listed by ``nefi list configs``."""
    dirs = []
    env = os.environ.get("NEFI_CONFIG_DIR")
    if env:
        dirs.append(Path(env))
    dirs.append(Path.cwd() / "configs")
    dirs.append(Path(nefi.__file__).resolve().parents[1] / "configs")
    out: list[Path] = []
    for d in dirs:
        if d.is_dir() and d.resolve() not in [o.resolve() for o in out]:
            out.append(d)
    return out


def config_files() -> list[Path]:
    """All ``*.yaml`` / ``*.yml`` / ``*.json`` files in :func:`config_dirs`."""
    files: list[Path] = []
    for d in config_dirs():
        for pattern in ("*.yaml", "*.yml", "*.json"):
            files.extend(sorted(d.glob(pattern)))
    return files


_LOADED_PLUGINS: set[str] = set()


def import_plugin(ref: str) -> None:
    """Import a plugin module by dotted name or by ``.py`` path (idempotent)."""
    import importlib
    import importlib.util

    ref = ref.strip()
    if not ref or ref in _LOADED_PLUGINS:
        return
    if ref.endswith(".py"):
        path = Path(ref).expanduser().resolve()
        if not path.exists():
            raise CLIError(f"plugin file not found: {ref}")
        name = f"nefi_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise CLIError(f"cannot import plugin {ref}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    else:
        try:
            importlib.import_module(ref)
        except ImportError as e:
            raise CLIError(f"cannot import plugin module {ref!r}: {e}") from e
    _LOADED_PLUGINS.add(ref)


def load_plugins(extra: Sequence[str] = ()) -> None:
    """Import ``NEFI_PLUGINS``, ``nefi.plugins`` entry points and ``extra`` modules / files."""
    refs = [r for r in os.environ.get("NEFI_PLUGINS", "").split(",") if r.strip()]
    for ref in [*refs, *extra]:
        import_plugin(ref)
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="nefi.plugins"):
            if ep.value not in _LOADED_PLUGINS:
                try:
                    ep.load()
                    _LOADED_PLUGINS.add(ep.value)
                except Exception as e:  # a broken third-party plugin must not break the CLI
                    log.warning("nefi plugin %s failed to load: %s", ep.name, e)
    except Exception:  # pragma: no cover - importlib.metadata unavailable
        pass


def available_instances() -> list[str]:
    return list_registered("instance")["instance"]


def _check_instance(name: str) -> None:
    if name.lower() not in available_instances():
        cfgs = ", ".join(p.name for p in config_files()) or "none found"
        raise CLIError(
            f"unknown instance {name!r}. Available instances: "
            f"{', '.join(available_instances()) or '(none)'}; or pass a config file "
            f"(configs: {cfgs})"
        )


def read_spec_file(path: str | Path) -> RunSpec:
    """Parse a config file in either accepted layout (see the module docstring)."""
    path = Path(path)
    d = load_config(path)
    if not isinstance(d, Mapping):
        raise CLIError(f"{path}: expected a mapping at the top level")
    imports = d.get("imports") or []
    for ref in [imports] if isinstance(imports, str) else imports:
        ref = str(ref)
        if ref.endswith(".py") and not Path(ref).is_absolute():
            ref = str((path.parent / ref).resolve())
        import_plugin(ref)
    inst = d.get("instance")
    if isinstance(inst, Mapping):
        if "type" not in inst:
            raise CLIError(f"{path}: 'instance' mapping needs a 'type' key")
        name = str(inst["type"])
        cfg = {k: v for k, v in inst.items() if k != "type"}
        cfg.update(d.get("config") or {})
    elif isinstance(inst, str):
        name, cfg = inst, dict(d.get("config") or {})
    else:
        raise CLIError(f"{path}: needs an 'instance' key (a name, or a mapping with 'type')")
    return RunSpec(
        instance=name,
        config=cfg,
        curriculum=d.get("curriculum"),
        solver=dict(d.get("solver") or {}),
        run=dict(d.get("run") or {}),
        source=str(path),
    )


def _smoke_overrides(spec: RunSpec) -> tuple[dict[str, Any], dict[str, Any] | None, str] | None:
    cls = registry_get("instance", spec.instance)
    hook = getattr(cls, "smoke_overrides", None)
    if callable(hook):
        hook = hook()
    if isinstance(hook, Mapping):
        return dict(hook), None, f"{cls.__name__}.smoke_overrides"
    for d in config_dirs():
        for ext in (".yaml", ".yml", ".json"):
            p = d / f"{spec.instance}_smoke{ext}"
            if p.exists():
                s = read_spec_file(p)
                if s.instance.lower() == spec.instance.lower():
                    return s.config, s.curriculum, str(p)
    presets = getattr(cls, "PRESETS", None) or getattr(
        sys.modules.get(cls.__module__), "PRESETS", None
    )
    if isinstance(presets, Mapping) and isinstance(presets.get("smoke"), Mapping):
        return dict(presets["smoke"]), None, f"{cls.__module__}.PRESETS['smoke']"
    return None


def parse_override(item: str) -> tuple[str, Any]:
    """``"key=value"`` → ``(key, parsed value)``."""
    from .bench.ablation import parse_value

    if "=" not in item:
        raise CLIError(f"--set expects key=value, got {item!r}")
    k, v = item.split("=", 1)
    return k.strip(), parse_value(v)


def load_spec(target: str, overrides: Sequence[str] = (), smoke: bool = False) -> RunSpec:
    """Resolve a CLI target (instance name or config file) plus ``--smoke`` and ``--set``."""
    path = Path(target)
    if path.suffix.lower() in (".yaml", ".yml", ".json"):
        if not path.exists():
            raise CLIError(f"config file not found: {target}")
        spec = read_spec_file(path)
    elif path.is_file():
        raise CLIError(f"unsupported config file {target!r} (use .yaml/.yml/.json)")
    else:
        spec = RunSpec(target)
    _check_instance(spec.instance)
    if smoke:
        found = _smoke_overrides(spec)
        if found is not None:
            cfg, cur, src = found
            spec.config = {**spec.config, **cfg}
            if cur is not None:
                spec.curriculum = cur
            spec.smoke = src
        else:
            spec.smoke = "step cap only"
    for item in overrides or ():
        k, v = parse_override(item)
        head, dot, rest = k.partition(".")
        if dot and head in ("stage", "curriculum", "optim"):
            spec.curriculum_overrides[k] = v
        elif dot and head == "solver":
            spec.solver[rest] = v
        elif dot and head == "run":
            spec.run[rest] = v
        elif dot and head == "config":
            spec.config[rest] = v
        else:
            spec.config[k] = v
    return spec


def make_instance(spec: RunSpec) -> Any:
    """Instantiate the registered instance from the configuration file's settings."""
    cls = registry_get("instance", spec.instance)
    try:
        return cls(dict(spec.config))
    except TypeError as e:
        raise CLIError(f"cannot configure instance {spec.instance!r}: {e}") from e


def _apply_curriculum_overrides(cur: Curriculum, overrides: Mapping[str, Any]) -> Curriculum:
    for k, v in overrides.items():
        head, _, attr = k.partition(".")
        objs = {"stage": cur.stages, "curriculum": [cur], "optim": [cur.optim]}[head]
        for o in objs:
            if not hasattr(o, attr):
                raise CLIError(f"--set {k}: {type(o).__name__} has no attribute {attr!r}")
            setattr(o, attr, tuple(v) if attr in ("shape", "betas") and v is not None else v)
    return cur


def resolve_curriculum(
    spec: RunSpec, default: Curriculum, smoke: bool = False, smoke_steps: int = 40
) -> Curriculum:
    """Curriculum from the config file (or ``default``) + overrides + smoke budget cap."""
    import copy

    from .bench.protocol import scale_budget

    cur = from_dict(Curriculum, spec.curriculum) if spec.curriculum else copy.deepcopy(default)
    cur = _apply_curriculum_overrides(cur, spec.curriculum_overrides)
    if smoke:
        cur = scale_budget(cur, 1.0, max_total_steps=smoke_steps, min_stage_steps=2)
        cur.restarts = 1
    return cur


def solver_kwargs(spec: RunSpec) -> dict[str, Any]:
    """Solver keyword arguments from the ``solver:`` section (dtype strings → torch dtypes)."""
    kw = dict(spec.solver)
    for k in ("device", "seed", "callbacks"):
        kw.pop(k, None)
    if isinstance(kw.get("dtype"), str):
        kw["dtype"] = getattr(torch, kw["dtype"].replace("torch.", ""))
    return kw


def _pick(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _default_out(spec: RunSpec, kind: str, extra: Mapping[str, Any]) -> Path:
    h = config_hash(
        {
            "instance": spec.instance,
            "config": spec.config,
            "curriculum": spec.curriculum,
            "overrides": spec.curriculum_overrides,
            **extra,
        },
        n=8,
    )
    name = spec.instance if kind == "run" else f"{spec.instance}-{kind}"
    return Path("runs") / f"{name}-{h}"


def split_list(text: str | Sequence[str] | None) -> list[str]:
    """Split a comma-separated list, ignoring commas inside brackets (``a,set:s=[1,2],b``)."""
    if text is None:
        return []
    if not isinstance(text, str):
        return [t for item in text for t in split_list(item)]
    out, buf, depth = [], [], 0
    for ch in text:
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf).strip())
    return [o for o in out if o]


def _ints(text: str | Sequence[int] | None) -> list[int]:
    if text is None:
        return []
    if isinstance(text, str):
        return [int(t) for t in split_list(text)]
    return [int(t) for t in text]


def _setup_logging(verbose: bool) -> None:
    nefi.enable_logging(logging.INFO if verbose else logging.WARNING)


def _fmt(v: Any) -> str:
    from .bench.report import format_value

    return format_value(v, 4)


def _history_rows(history: Mapping[str, list]) -> tuple[list[dict[str, Any]], list[str]]:
    n = len(history.get("total", []))
    cols = [k for k, v in history.items() if len(v) == n]
    ragged = [k for k in history if k not in cols]
    return [{k: history[k][i] for k in cols} for i in range(n)], ragged


# ==========================================================================================
# commands
# ==========================================================================================
def cmd_list(kind: str | None = None, verbose: bool = False, as_json: bool = False) -> int:
    """List registered components (instances, fields, operators, losses, baselines, metrics…)."""
    reg = list_registered()
    kinds = [k for k in KIND_ORDER if k in reg] + [k for k in reg if k not in KIND_ORDER]
    want = None
    if kind:
        k = kind.lower().rstrip("s") if kind.lower() not in reg else kind.lower()
        if k == "config":
            files = config_files()
            if as_json:
                print(json.dumps([str(f) for f in files], indent=2))
            else:
                print("configs:" if files else "configs: (none found)")
                for f in files:
                    print(f"  {f}")
            return 0
        if k not in reg:
            raise CLIError(f"unknown kind {kind!r}; choose from {kinds + ['configs']}")
        want = [k]
    info: dict[str, Any] = {}
    for k in want or kinds:
        names = reg.get(k, [])
        if not names and want is None:
            continue
        if k == "instance":
            info[k] = {n: _instance_info(n, verbose) for n in names}
        else:
            info[k] = names
    if as_json:
        print(json.dumps(info, indent=2, default=str))
        return 0
    for k, v in info.items():
        if k == "instance":
            print("instances:")
            for n, d in v.items():
                cfgs = f"  [configs: {', '.join(d['configs'])}]" if d.get("configs") else ""
                print(f"  {n:<20} {d.get('description', '')}{cfgs}")
                if verbose:
                    for key in ("scene_classes", "baselines", "metrics"):
                        if d.get(key):
                            print(f"      {key}: {', '.join(map(str, d[key]))}")
                    if d.get("config"):
                        print("      config: " + ", ".join(f"{a}={b}" for a, b in d["config"]))
        else:
            plural = {"loss": "losses", "postprocess": "postprocessors", "datagen": "datagens"}
            print(f"{plural.get(k, k + 's')}: {', '.join(v) if v else '(none)'}")
    return 0


def _instance_info(name: str, verbose: bool) -> dict[str, Any]:
    cls = registry_get("instance", name)
    d: dict[str, Any] = {"description": getattr(cls, "description", "") or ""}
    d["configs"] = [p.name for p in config_files() if p.stem.split("_")[0] == name.split("_")[0]]
    d["configs"] = [c for c in d["configs"] if c.startswith(name)]
    if verbose:
        try:
            inst = cls()
            d["scene_classes"] = list(getattr(inst.scene_generator(), "classes", ()))
            d["baselines"] = sorted(inst.baselines())
            d["metrics"] = sorted(inst.metrics())
            cfg = to_dict(inst.cfg)
            d["config"] = [(k, v) for k, v in cfg.items()] if isinstance(cfg, dict) else []
        except Exception as e:  # pragma: no cover - broken instance
            d["error"] = f"{type(e).__name__}: {e}"
    return d


def cmd_run(
    target: str,
    device: str | None = None,
    seed: int | None = None,
    scene: str | None = None,
    out: str | None = None,
    smoke: bool = False,
    smoke_steps: int = 40,
    plot: bool = False,
    baseline: str | None = None,
    overrides: Sequence[str] = (),
    quiet: bool = False,
    verbose: bool = False,
) -> int:
    """Generate a measurement, invert it, evaluate, save everything and print a summary."""
    from .bench.protocol import baseline_method, default_method, evaluate_metrics
    from .bench.report import to_jsonable, write_csv, write_json
    from .solve.callbacks import ProgressBar

    _setup_logging(verbose)
    spec = load_spec(target, overrides, smoke)
    seed = int(_pick(seed, spec.run.get("seed"), 0))
    device = str(_pick(device, spec.run.get("device"), "auto"))
    scene = _pick(scene, spec.run.get("scene"))
    inst = make_instance(spec)
    gt, meas = inst.make_measurement(seed=seed, scene_class=scene)
    seed_everything(seed)
    method = baseline_method(baseline) if baseline else default_method()
    prep = method.prepare(inst, meas)
    cur = resolve_curriculum(spec, prep.resolved_curriculum(), smoke, smoke_steps)
    prep.solver_kwargs.update(solver_kwargs(spec))
    callbacks = [] if quiet else [ProgressBar()]
    t0 = time.perf_counter()
    result = prep.run(cur, device=device, seed=seed, callbacks=callbacks)
    wall = time.perf_counter() - t0
    metrics = evaluate_metrics(inst, result, gt)
    try:
        from .diagnostics import data_fit_paradox

        fit = data_fit_paradox(result, prep.problem, None)
    except Exception as e:  # pragma: no cover - exotic measurement layouts
        log.warning("data-fit summary unavailable: %s", e)
        fit = {}

    out_dir = Path(
        _pick(out, spec.run.get("out"))
        or _default_out(spec, "run", {"seed": seed, "scene": scene, "baseline": baseline})
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    result.save(out_dir / "result.pt")
    save_config(
        {
            "instance": spec.instance,
            "config": to_dict(inst.cfg),
            "curriculum": to_dict(cur),
            "solver": to_jsonable(spec.solver),
            "run": {
                "seed": seed,
                "device": device,
                "scene": scene,
                "baseline": baseline,
                "smoke": spec.smoke if smoke else None,
            },
        },
        out_dir / "config.yaml",
    )
    rows, ragged = _history_rows(result.history)
    write_csv(out_dir / "history.csv", rows)
    write_json(
        out_dir / "metrics.json",
        {
            "instance": spec.instance,
            "method": baseline or "neural",
            "seed": seed,
            "scene": scene,
            "device": result.extra.get("device"),
            "metrics": metrics,
            "data_fit": fit,
            "timing": {**result.timing, "wall_s": wall},
            "stage_results": result.stage_results,
            "post_info": result.post_info,
            "config_hash": result.config_hash,
            "n_parameters": result.extra.get("n_parameters"),
            "history_ragged_keys": ragged,
        },
    )
    files = ["result.pt", "config.yaml", "metrics.json", "history.csv"]
    if plot:
        files += _save_run_plots(out_dir, result, gt, meas)

    print(
        f"nefi run {spec.instance} · {baseline or 'neural'} · seed {seed} · "
        f"scene {scene or inst.default_scene_class() or '-'} · device "
        f"{result.extra.get('device', device)}" + (f" · smoke ({spec.smoke})" if smoke else "")
    )
    for s in result.stage_results:
        print(
            f"  {s.get('name', 'stage')} {tuple(s.get('shape', ()))}: {s.get('steps')} steps, "
            f"{s.get('stop')}, {s.get('seconds', 0.0):.2f}s"
        )
    print(f"  total {wall:.2f}s · {result.extra.get('n_parameters', '?')} parameters")
    print("  metrics  " + " · ".join(f"{k} {_fmt(v)}" for k, v in metrics.items()))
    if fit:
        line = f"  data fit PSNR {_fmt(fit.get('data_psnr'))} dB"
        if "discrepancy_ratio" in fit:
            line += f" · RMSE/σ {_fmt(fit['discrepancy_ratio'])}"
        print(line)
    print(f"  saved    {out_dir}  ({' '.join(files)})")
    return 0


def _save_run_plots(out_dir: Path, result: Any, gt: Any, meas: Any) -> list[str]:
    try:
        from .diagnostics.plots import plot_history, plot_result, save_figure
    except ImportError:  # pragma: no cover
        return []
    names = []
    try:
        save_figure(plot_result(result, gt, meas), out_dir / "fields.png")
        names.append("fields.png")
        save_figure(plot_history(result), out_dir / "history.png")
        names.append("history.png")
    except ImportError as e:
        log.warning("--plot needs matplotlib: %s", e)
    return names


def _bench_common(
    target: str,
    overrides: Sequence[str],
    smoke: bool,
    verbose: bool,
) -> tuple[RunSpec, Any]:
    _setup_logging(verbose)
    spec = load_spec(target, overrides, smoke)
    return spec, make_instance(spec)


def _with_spec_curriculum(spec: RunSpec, methods: list[Any]) -> list[Any]:
    """Apply a config-file curriculum / ``--set stage.*`` to the default (neural) method."""
    from .bench.protocol import Method, PreparedRun

    if not spec.curriculum and not spec.curriculum_overrides:
        return methods
    out = []
    for m in methods:
        if m.build is not None:
            out.append(m)
            continue

        def build(inst, meas, _m=m):
            prep = _m.prepare(inst, meas)
            cur = resolve_curriculum(spec, prep.resolved_curriculum())
            return PreparedRun(prep.problem, cur, prep.solver_kwargs, prep.runner)

        out.append(Method(m.name, build, dict(m.solver_kwargs), m.description))
    return out


def cmd_bench(
    target: str,
    methods: str | None = None,
    n: int = 4,
    seeds: str = "0,1,2",
    classes: str | None = None,
    metrics: str | None = None,
    device: str | None = None,
    out: str | None = None,
    smoke: bool = False,
    smoke_steps: int = 40,
    budget_scale: float = 1.0,
    allow_inverse_crime: bool = False,
    by_class: bool = False,
    overrides: Sequence[str] = (),
    quiet: bool = False,
    verbose: bool = False,
    batched: bool = False,
    batch_size: int | None = None,
    shard: str | None = None,
) -> int:
    """Paired benchmark of several methods; prints the markdown table and saves the reports.

    ``--batched`` solves the runs of every batchable method in one optimization loop;
    ``--shard i/n`` runs shard ``i`` of ``n`` and writes ``<out>/shard-III-of-NNN.json`` (the
    default output directory does not depend on the shard, so all shards of a campaign land in
    one directory for ``nefi bench-merge``).
    """
    from .bench.protocol import parse_shard, resolve_methods, run_benchmark

    spec, inst = _bench_common(target, overrides, smoke, verbose)
    wanted = split_list(methods) if methods else spec.run.get("methods")
    method_list = _with_spec_curriculum(spec, resolve_methods(inst, wanted))
    seed_list = _ints(seeds)
    shard_i, shard_n = parse_shard(shard)
    res = run_benchmark(
        inst,
        method_list,
        n_samples=int(n),
        seeds=seed_list,
        classes=classes if classes in (None, "all") else split_list(classes),
        metrics=split_list(metrics) if metrics else None,
        device=str(_pick(device, spec.run.get("device"), "auto")),
        budget_scale=float(budget_scale),
        max_total_steps=int(smoke_steps) if smoke else None,
        progress=not quiet,
        allow_inverse_crime=allow_inverse_crime,
        batched=batched,
        batch_size=batch_size,
        shard=(shard_i, shard_n),
    )
    out_dir = Path(
        out
        or _default_out(
            spec, "bench", {"methods": res.methods, "n": n, "seeds": seed_list, "smoke": smoke}
        )
    )
    if shard_n > 1:
        path = res.save_shard(out_dir)
        print(res.to_markdown(by_class=by_class))
        print(f"saved shard {shard_i}/{shard_n}: {path}  (merge: nefi bench-merge {out_dir})")
        return 0
    res.save(out_dir)
    print(res.to_markdown(by_class=by_class))
    print(f"saved {out_dir} (summary.md, summary.csv, rows.csv, benchmark.json)")
    return 0


def cmd_bench_merge(
    directory: str,
    out: str | None = None,
    by_class: bool = False,
    allow_missing: bool = False,
    quiet: bool = False,
    verbose: bool = False,
) -> int:
    """Merge the benchmark shards of a directory into one table (writes the usual reports)."""
    from .bench.protocol import BenchmarkResult

    _setup_logging(verbose)
    d = Path(directory)
    if not d.exists():
        raise CLIError(f"no such directory: {d}")
    try:
        res = BenchmarkResult.merge(d, strict=not allow_missing)
    except NefiError as e:
        raise CLIError(str(e.args[0] if e.args else e)) from e
    out_dir = Path(out) if out else d
    res.save(out_dir)
    if not quiet:
        print(res.to_markdown(by_class=by_class))
    shards = res.meta.get("merged_shards", [])
    print(
        f"merged {len(shards)} shard(s) of {res.meta.get('n_shards', '?')} ({len(res.rows)} runs) "
        f"into {out_dir} (summary.md, summary.csv, rows.csv, benchmark.json)"
    )
    return 0


def cmd_diagnose(
    target: str,
    device: str | None = None,
    seed: int | None = None,
    scene: str | None = None,
    out: str | None = None,
    smoke: bool = False,
    smoke_steps: int = 40,
    solve: bool = False,
    plot: bool = False,
    probes: int = 32,
    k: int = 8,
    n_iter: int = 24,
    overrides: Sequence[str] = (),
    quiet: bool = False,
    verbose: bool = False,
) -> int:
    """Diagnostics on a fresh measurement (at initialization, or at the solution with --solve)."""
    from .bench.protocol import default_method
    from .diagnostics import diagnose
    from .solve.callbacks import ProgressBar
    from .utils.device import resolve_device

    _setup_logging(verbose)
    spec = load_spec(target, overrides, smoke)
    seed = int(_pick(seed, spec.run.get("seed"), 0))
    device = str(_pick(device, spec.run.get("device"), "auto"))
    scene = _pick(scene, spec.run.get("scene"))
    inst = make_instance(spec)
    gt, meas = inst.make_measurement(seed=seed, scene_class=scene)
    seed_everything(seed)
    prep = default_method().prepare(inst, meas)
    problem = prep.problem
    result = None
    if solve:
        cur = resolve_curriculum(spec, prep.resolved_curriculum(), smoke, smoke_steps)
        prep.solver_kwargs.update(solver_kwargs(spec))
        callbacks = [] if quiet else [ProgressBar()]
        result = prep.run(cur, device=device, seed=seed, callbacks=callbacks)
    else:
        problem.to(resolve_device(device))
    if smoke:
        probes, k, n_iter = min(probes, 8), min(k, 4), min(n_iter, 12)
    report = diagnose(
        problem, gt=gt, result=result, n_probes=int(probes), k=int(k), n_iter=int(n_iter), seed=seed
    )
    out_dir = Path(
        out or _default_out(spec, "diagnose", {"seed": seed, "scene": scene, "solve": solve})
    )
    paths = report.save(out_dir)
    if plot:
        try:
            from .diagnostics.plots import plot_report, save_figure

            paths["plot"] = save_figure(plot_report(report), out_dir / "diagnostics.png")
        except ImportError as e:
            log.warning("--plot needs matplotlib: %s", e)
    print(report.to_markdown())
    print(f"saved {out_dir} ({', '.join(p.name for p in paths.values())})")
    return 0


def _tuned_curriculum_config(problem: Any, cur: Curriculum) -> dict[str, Any]:
    """The tuned curriculum as a config dict whose stages carry the tuned regularizer weights
    (``nefi run tuned_config.yaml`` rebuilds the instance with its default weights)."""
    import copy

    from .autotune import regularizer_names

    c = copy.deepcopy(cur)
    names = regularizer_names(problem, c)
    for st in c.stages:
        lw = dict(st.loss_weights or {})
        for k in names:
            lw.setdefault(k, float(problem.losses.weights[k]))
        st.loss_weights = lw or None
    return to_dict(c)


def cmd_autotune(
    target: str,
    level: str = "standard",
    device: str | None = None,
    seed: int | None = None,
    scene: str | None = None,
    out: str | None = None,
    smoke: bool = False,
    compare: bool = False,
    overrides: Sequence[str] = (),
    quiet: bool = False,
    verbose: bool = False,
) -> int:
    """Auto-tune an instance on a fresh measurement; save the report and a tuned config.

    Writes ``autotune.md`` / ``autotune.json`` (every decision with its evidence) and
    ``tuned_config.yaml`` (instance config + tuned curriculum, with the tuned loss weights as
    per-stage overrides: ``nefi run tuned_config.yaml`` reproduces the tuned solve; gauge fixes
    are applied in-process only and listed under ``autotune.applied_in_process``). ``--smoke``
    selects the small smoke preset; the budget is then *tuned*, not capped.
    """
    from .autotune import LEVELS, autotune_instance
    from .bench.report import to_jsonable

    _setup_logging(verbose)
    if level not in LEVELS:
        raise CLIError(f"--level must be one of {', '.join(LEVELS)}, got {level!r}")
    spec = load_spec(target, overrides, smoke)
    seed = int(_pick(seed, spec.run.get("seed"), 0))
    device = str(_pick(device, spec.run.get("device"), "auto"))
    scene = _pick(scene, spec.run.get("scene"))
    inst = make_instance(spec)
    base_cur = None
    if spec.curriculum or spec.curriculum_overrides:
        base_cur = resolve_curriculum(spec, inst.default_curriculum())
    tuned, cur, report = autotune_instance(
        inst,
        seed=seed,
        level=level,
        scene_class=scene,
        curriculum=base_cur,
        device=device,
        compare=compare,
    )
    out_dir = Path(
        out
        or _default_out(
            spec, "autotune", {"seed": seed, "scene": scene, "level": level, "smoke": smoke}
        )
    )
    report.save(out_dir)
    applied = list(report.gauges.repairs) if report.gauges is not None else []
    save_config(
        {
            "instance": spec.instance,
            "config": to_dict(inst.cfg),
            "curriculum": _tuned_curriculum_config(tuned, cur),
            "solver": to_jsonable(spec.solver),
            "run": {"seed": seed, "device": device, "scene": scene},
            "autotune": to_jsonable(
                {
                    "level": level,
                    "summary": report.summary(),
                    "sigma": report.sigma,
                    "sigma_source": report.sigma_source,
                    "acquisition": None
                    if report.acquisition is None
                    else report.acquisition.verdict,
                    "decisions": [d.to_dict() for d in report.decisions],
                    "applied_in_process": applied,
                    "comparison": report.comparison,
                }
            ),
        },
        out_dir / "tuned_config.yaml",
    )
    print(report.summary() if quiet else report.to_markdown())
    print(f"saved {out_dir} (autotune.md, autotune.json, tuned_config.yaml)")
    return 0


def cmd_ablate(
    target: str,
    variants: str = REQUIRED,
    n: int = 4,
    seeds: str = "0,1,2",
    independent: bool = False,
    base: str | None = None,
    classes: str | None = None,
    device: str | None = None,
    out: str | None = None,
    smoke: bool = False,
    smoke_steps: int = 40,
    budget_scale: float = 1.0,
    overrides: Sequence[str] = (),
    quiet: bool = False,
    verbose: bool = False,
) -> int:
    """Cumulative (or one-at-a-time) ablation of the instance's method."""
    from .bench.ablation import cumulative_ablation, parse_modifier

    spec, inst = _bench_common(target, overrides, smoke, verbose)
    mods = [parse_modifier(v) for v in split_list(variants)]
    if not mods:
        raise CLIError("--variants needs at least one modifier")
    seed_list = _ints(seeds)
    res = cumulative_ablation(
        inst,
        mods,
        n_samples=int(n),
        seeds=seed_list,
        cumulative=not independent,
        base=base,
        classes=classes if classes in (None, "all") else split_list(classes),
        device=str(_pick(device, spec.run.get("device"), "auto")),
        budget_scale=float(budget_scale),
        max_total_steps=int(smoke_steps) if smoke else None,
        progress=not quiet,
    )
    out_dir = Path(
        out
        or _default_out(
            spec, "ablate", {"variants": variants, "n": n, "seeds": seed_list, "smoke": smoke}
        )
    )
    res.save(out_dir)
    mode = "one-at-a-time" if independent else "cumulative (row k applies modifiers 1..k)"
    print(f"Ablation — {mode}\n")
    print(res.to_markdown())
    print(f"saved {out_dir}")
    return 0


def cmd_sweep(
    target: str,
    axis: str = REQUIRED,
    values: str = REQUIRED,
    n: int = 4,
    seeds: str = "0,1,2",
    base: str | None = None,
    classes: str | None = None,
    device: str | None = None,
    out: str | None = None,
    smoke: bool = False,
    smoke_steps: int = 40,
    budget_scale: float = 1.0,
    overrides: Sequence[str] = (),
    quiet: bool = False,
    verbose: bool = False,
) -> int:
    """One-axis hyperparameter sweep (config field, stage.*, curriculum.*, optim.*, weight.*)."""
    from .bench.ablation import parse_value
    from .bench.sweep import sweep

    spec, inst = _bench_common(target, overrides, smoke, verbose)
    vals = [parse_value(v) for v in split_list(values)]
    seed_list = _ints(seeds)
    res = sweep(
        inst,
        axis,
        vals,
        n_samples=int(n),
        seeds=seed_list,
        base=base,
        classes=classes if classes in (None, "all") else split_list(classes),
        device=str(_pick(device, spec.run.get("device"), "auto")),
        budget_scale=float(budget_scale),
        max_total_steps=int(smoke_steps) if smoke else None,
        progress=not quiet,
    )
    out_dir = Path(
        out
        or _default_out(spec, "sweep", {"axis": axis, "values": values, "n": n, "seeds": seed_list})
    )
    res.save(out_dir)
    print(res.to_markdown())
    print(f"saved {out_dir}")
    return 0


# ==========================================================================================
# declarative command table → typer / argparse front-ends
# ==========================================================================================
@dataclass(frozen=True)
class Opt:
    """One CLI parameter. ``flags=()`` makes it positional."""

    name: str
    flags: tuple[str, ...] = ()
    type: type = str
    default: Any = None
    help: str = ""
    flag: bool = False
    multiple: bool = False
    metavar: str | None = None


@dataclass(frozen=True)
class Command:
    name: str
    func: Callable[..., int]
    help: str
    opts: tuple[Opt, ...]


_TARGET = Opt("target", (), str, REQUIRED, "instance name or config file (.yaml/.json)")
_DEVICE = Opt("device", ("--device", "-d"), str, None, "auto | cpu | cuda | cuda:1 | mps")
_SEED = Opt("seed", ("--seed",), int, None, "data + optimization seed (default 0)")
_SCENE = Opt("scene", ("--scene",), str, None, "scene class (see nefi list instances -v)")
_OUT = Opt("out", ("--out", "-o"), str, None, "output directory (default runs/<name>-<hash>)")
_SMOKE = Opt("smoke", ("--smoke",), bool, False, "tiny problem + step cap (quick check)", True)
_SMOKE_STEPS = Opt("smoke_steps", ("--smoke-steps",), int, 40, "total step cap with --smoke")
_SET = Opt(
    "overrides", ("--set",), str, None, "override KEY=VALUE (repeatable)", False, True, "KEY=VALUE"
)
_QUIET = Opt("quiet", ("--quiet", "-q"), bool, False, "no progress bars", True)
_VERBOSE = Opt("verbose", ("--verbose", "-v"), bool, False, "log solver progress (INFO)", True)
_N = Opt("n", ("--n", "-n"), int, 4, "measurements per scene class")
_SEEDS = Opt("seeds", ("--seeds",), str, "0,1,2", "comma-separated optimization seeds")
_CLASSES = Opt("classes", ("--classes",), str, None, "scene classes (comma list or 'all')")
_BUDGET = Opt("budget_scale", ("--budget-scale",), float, 1.0, "multiply every stage's steps")
_BASE = Opt("base", ("--base",), str, None, "method to ablate/sweep (default: neural)")

COMMANDS: tuple[Command, ...] = (
    Command(
        "list",
        cmd_list,
        "List registered instances, fields, operators, losses, baselines, metrics (or configs).",
        (
            Opt("kind", (), str, None, "one kind: instances, fields, operators, ..., configs"),
            Opt("verbose", ("--verbose", "-v"), bool, False, "instance details", True),
            Opt("as_json", ("--json",), bool, False, "machine-readable output", True),
        ),
    ),
    Command(
        "run",
        cmd_run,
        "Generate a measurement, invert it, evaluate and save result/config/metrics/history.",
        (
            _TARGET,
            _DEVICE,
            _SEED,
            _SCENE,
            _OUT,
            _SMOKE,
            _SMOKE_STEPS,
            Opt("plot", ("--plot",), bool, False, "also save fields.png and history.png", True),
            Opt("baseline", ("--baseline", "-b"), str, None, "run a baseline instead"),
            _SET,
            _QUIET,
            _VERBOSE,
        ),
    ),
    Command(
        "bench",
        cmd_bench,
        "Paired benchmark (samples × seeds, 95% CI, runtime) of several methods.",
        (
            _TARGET,
            Opt("methods", ("--methods", "-m"), str, None, "e.g. neural,grid (default: all)"),
            _N,
            _SEEDS,
            _CLASSES,
            Opt(
                "metrics", ("--metrics",), str, None, "registered metric names (default: instance)"
            ),
            _DEVICE,
            _OUT,
            _SMOKE,
            _SMOKE_STEPS,
            _BUDGET,
            Opt(
                "allow_inverse_crime",
                ("--allow-inverse-crime",),
                bool,
                False,
                "permit the matched-operator regime (labelled in all reports)",
                True,
            ),
            Opt("by_class", ("--by-class",), bool, False, "per-class table", True),
            Opt(
                "batched",
                ("--batched",),
                bool,
                False,
                "solve the runs of batchable methods in one batched loop (nefi.batch_invert)",
                True,
            ),
            Opt("batch_size", ("--batch-size",), int, None, "max problems per batch (--batched)"),
            Opt(
                "shard",
                ("--shard",),
                str,
                None,
                "run shard i/n of the (class, sample, seed) units; merge with bench-merge",
                metavar="I/N",
            ),
            _SET,
            _QUIET,
            _VERBOSE,
        ),
    ),
    Command(
        "bench-merge",
        cmd_bench_merge,
        "Merge benchmark shards (shard-*-of-*.json in DIR) into one table and report files.",
        (
            Opt("directory", (), str, REQUIRED, "directory holding the shard files"),
            Opt("out", ("--out", "-o"), str, None, "output directory (default: DIR)"),
            Opt("by_class", ("--by-class",), bool, False, "per-class table", True),
            Opt(
                "allow_missing",
                ("--allow-missing",),
                bool,
                False,
                "merge even if shards are missing (logged)",
                True,
            ),
            _QUIET,
            _VERBOSE,
        ),
    ),
    Command(
        "diagnose",
        cmd_diagnose,
        "Ill-posedness diagnostics (sensitivity, iter-0 gradient, filter kernel, spectrum, ...).",
        (
            _TARGET,
            _DEVICE,
            _SEED,
            _SCENE,
            _OUT,
            _SMOKE,
            _SMOKE_STEPS,
            Opt("solve", ("--solve",), bool, False, "solve first; diagnose at the solution", True),
            Opt("plot", ("--plot",), bool, False, "also save diagnostics.png", True),
            Opt("probes", ("--probes",), int, 32, "sensitivity probes"),
            Opt("k", ("--k",), int, 8, "number of singular values"),
            Opt("n_iter", ("--n-iter",), int, 24, "Lanczos iterations"),
            _SET,
            _QUIET,
            _VERBOSE,
        ),
    ),
    Command(
        "autotune",
        cmd_autotune,
        "Auto-tune an instance: noise, gauges, acquisition report, budget / learning rate, "
        "regularization (report + tuned_config.yaml).",
        (
            _TARGET,
            Opt("level", ("--level", "-l"), str, "standard", "quick | standard | thorough"),
            _DEVICE,
            _SEED,
            _SCENE,
            _OUT,
            Opt(
                "smoke",
                ("--smoke",),
                bool,
                False,
                "use the instance's smoke preset (small problem; the budget is tuned, not capped)",
                True,
            ),
            Opt(
                "compare",
                ("--compare",),
                bool,
                False,
                "also solve the default and the tuned configuration and report both",
                True,
            ),
            _SET,
            _QUIET,
            _VERBOSE,
        ),
    ),
    Command(
        "ablate",
        cmd_ablate,
        "Cumulative ablation (NeTMY Tab. 3): e.g. --variants no_annealing,no_pe,single_stage.",
        (
            _TARGET,
            Opt(
                "variants",
                ("--variants",),
                str,
                REQUIRED,
                "modifiers: no_annealing, no_pe, single_stage, grid_field, no_gate, "
                "drop_loss:NAME, set_weight:NAME=W, set:KEY=V, stage:ATTR=V, optim:ATTR=V",
            ),
            _N,
            _SEEDS,
            Opt("independent", ("--independent",), bool, False, "one modifier per row", True),
            _BASE,
            _CLASSES,
            _DEVICE,
            _OUT,
            _SMOKE,
            _SMOKE_STEPS,
            _BUDGET,
            _SET,
            _QUIET,
            _VERBOSE,
        ),
    ),
    Command(
        "sweep",
        cmd_sweep,
        "One-axis sweep (NeTMY Tab. 10): e.g. --axis lr --values 1e-4,1e-3,1e-2.",
        (
            _TARGET,
            Opt("axis", ("--axis",), str, REQUIRED, "config field or stage.* / optim.* / weight.*"),
            Opt("values", ("--values",), str, REQUIRED, "comma-separated values"),
            _N,
            _SEEDS,
            _BASE,
            _CLASSES,
            _DEVICE,
            _OUT,
            _SMOKE,
            _SMOKE_STEPS,
            _BUDGET,
            _SET,
            _QUIET,
            _VERBOSE,
        ),
    ),
)


def _dispatch(func: Callable[..., int], kwargs: dict[str, Any]) -> int:
    kwargs = {k: ([] if v is None and k == "overrides" else v) for k, v in kwargs.items()}
    try:
        return int(func(**kwargs) or 0)
    except NefiError as e:
        msg = e.args[0] if e.args else str(e)
        print(f"nefi: error: {msg}", file=sys.stderr)
        return 2


# ---- argparse ---------------------------------------------------------------------------
def build_parser():
    """The argparse front-end (always available)."""
    import argparse

    p = argparse.ArgumentParser(
        prog="nefi",
        description=CLI_DESCRIPTION,
    )
    p.add_argument("--version", action="version", version=f"nefi {nefi.__version__}")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    for cmd in COMMANDS:
        sp = sub.add_parser(cmd.name, help=cmd.help, description=cmd.help)
        for o in cmd.opts:
            if not o.flags:
                kw: dict[str, Any] = {"help": o.help, "metavar": o.metavar or o.name.upper()}
                if o.default is not REQUIRED:
                    kw.update(nargs="?", default=o.default)
                if o.type is not str:
                    kw["type"] = o.type
                sp.add_argument(o.name, **kw)
            elif o.flag:
                sp.add_argument(*o.flags, dest=o.name, action="store_true", help=o.help)
            elif o.multiple:
                sp.add_argument(
                    *o.flags, dest=o.name, action="append", type=o.type, help=o.help,
                    metavar=o.metavar,
                )  # fmt: skip
            else:
                kw = {"dest": o.name, "type": o.type, "help": o.help, "metavar": o.metavar}
                if o.default is REQUIRED:
                    kw["required"] = True
                else:
                    kw["default"] = o.default
                sp.add_argument(*o.flags, **kw)
        sp.set_defaults(_func=cmd.func)
    return p


def _main_argparse(argv: list[str]) -> int:
    parser = build_parser()
    try:
        ns = parser.parse_args(argv)
    except SystemExit as e:  # --help, --version, usage errors
        return int(e.code or 0) if isinstance(e.code, int | type(None)) else 2
    func = getattr(ns, "_func", None)
    if func is None:
        parser.print_help()
        return 0
    kwargs = {k: v for k, v in vars(ns).items() if k not in ("_func", "command")}
    return _dispatch(func, kwargs)


# ---- typer ------------------------------------------------------------------------------
def _typer_available() -> bool:
    try:
        import typer  # noqa: F401
        from typer.main import get_command  # noqa: F401
    except ImportError:
        return False
    return True


def _typer_click():
    """The click implementation typer runs on.

    typer < 0.27 builds on the standalone ``click`` package; from 0.27 on it ships its own copy as
    ``typer._click`` and no longer depends on ``click``. The exceptions a command raises come from
    whichever of the two typer imported, so resolve the module through ``typer.main``.
    """
    import typer.main as typer_main

    module = getattr(typer_main, "click", None) or getattr(typer_main, "_click", None)
    if module is None:  # pragma: no cover - unknown layout: try the standalone package
        import click as module
    return module


def _typer_callback(cmd: Command):
    import typer

    params = []
    for o in cmd.opts:
        if o.multiple:
            ann: Any = typing.Optional[list[o.type]]  # noqa: UP045
        elif o.flag:
            ann = bool
        elif o.default is None:
            ann = typing.Optional[o.type]  # noqa: UP045
        else:
            ann = o.type
        if not o.flags:
            default = typer.Argument(o.default, help=o.help, metavar=o.metavar or o.name.upper())
        else:
            default = typer.Option(
                o.default, *o.flags, help=o.help, **({"metavar": o.metavar} if o.metavar else {})
            )
        params.append(
            inspect.Parameter(
                o.name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=ann
            )
        )

    def callback(**kwargs: Any) -> int:
        return _dispatch(cmd.func, kwargs)

    callback.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    callback.__name__ = cmd.name.replace("-", "_")
    callback.__doc__ = cmd.help
    return callback


def build_typer_app():
    """The typer front-end (same commands and options as :func:`build_parser`)."""
    import typer

    kw: dict[str, Any] = {
        "help": CLI_DESCRIPTION,
        "add_completion": False,
        "no_args_is_help": True,
    }
    try:
        app = typer.Typer(pretty_exceptions_enable=False, **kw)
    except TypeError:  # pragma: no cover - old typer
        app = typer.Typer(**kw)
    for cmd in COMMANDS:
        app.command(name=cmd.name, help=cmd.help)(_typer_callback(cmd))

    def _version(value: bool) -> None:
        if value:
            print(f"nefi {nefi.__version__}")
            raise typer.Exit()

    @app.callback()
    def root(
        version: bool = typer.Option(
            False, "--version", callback=_version, is_eager=True, help="Show the version."
        ),
    ) -> None:
        """nefi — Neural-Field Inversion."""

    return app


def _main_typer(argv: list[str]) -> int:
    import typer
    from typer.main import get_command

    click = _typer_click()
    command = get_command(build_typer_app())
    try:
        rv = command.main(args=argv, prog_name="nefi", standalone_mode=False)
    except click.ClickException as e:
        e.show()
        return int(e.exit_code)
    except typer.Abort:
        print("Aborted!", file=sys.stderr)
        return 1
    except typer.Exit as e:
        return int(getattr(e, "exit_code", 0) or 0)
    return int(rv or 0)


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point; returns the exit code (usable in-process)."""
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        load_plugins()
    except NefiError as e:
        print(f"nefi: error: {e.args[0] if e.args else e}", file=sys.stderr)
        return 2
    backend = os.environ.get("NEFI_CLI", "auto").lower()
    if backend == "argparse" or (backend == "auto" and not _typer_available()):
        return _main_argparse(args)
    if backend == "typer" and not _typer_available():
        raise CLIError("NEFI_CLI=typer but typer is not installed (pip install 'nefi[cli]')")
    return _main_typer(args)


def console() -> None:  # pragma: no cover - thin wrapper for `python -m nefi.cli`
    raise SystemExit(main())


__all__ = [
    "COMMANDS",
    "CLIError",
    "RunSpec",
    "build_parser",
    "build_typer_app",
    "cmd_ablate",
    "cmd_autotune",
    "cmd_bench",
    "cmd_bench_merge",
    "cmd_diagnose",
    "cmd_list",
    "cmd_run",
    "cmd_sweep",
    "config_files",
    "import_plugin",
    "load_plugins",
    "load_spec",
    "main",
    "make_instance",
    "read_spec_file",
    "resolve_curriculum",
]

if __name__ == "__main__":  # pragma: no cover
    console()
