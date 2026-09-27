"""Profile the per-step cost of an instance's optimization loop (CPU or CUDA).

Usage::

    python tools/profile_instance.py toy1d --steps 50
    python tools/profile_instance.py thermal_tomography \
        --config configs/thermal_tomography_smoke.yaml
    python tools/profile_instance.py nv_relaxometry --config configs/nv_relaxometry_smoke.yaml \
        --device cuda --steps 100 --trace runs/prof
    python tools/profile_instance.py thermal_tomography \
        --config configs/thermal_tomography_smoke.yaml --set solver=chebyshev \
        --set jacobi_iters=13 --threads 1 --json runs/prof/tt.jsonl

Runs the *real* :class:`nefi.Solver` loop (the chosen curriculum stage, a fixed number of steps,
no early stopping) and reports

* wall-clock per step (median and mean over ``--steps`` timed steps, after ``--warmup``);
* ``ops/step`` — aten operators dispatched per optimization step (forward + backward + optimizer),
  counted with a ``TorchDispatchMode``; view/metadata ops are reported separately because they
  launch no kernel. On a GPU the non-view count is a good proxy for kernel launches, which bound
  the step time of small problems;
* peak memory (CUDA) and the top ops from ``torch.profiler`` (``--table``).

``--minimal`` times the old stand-alone loop (field → operator → losses → backward → Adam,
without the solver's bookkeeping). ``--compile field|step``, ``--cuda-graphs`` and
``--autocast bf16|fp16`` forward the corresponding :class:`~nefi.solve.Solver` options.
Used by the performance-engineering pass (``docs/performance.md``); not part of the library API.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import yaml
from torch.utils._python_dispatch import TorchDispatchMode

import nefi
from nefi.config import load_config
from nefi.registry import build
from nefi.utils.device import peak_memory_mb, reset_peak_memory, resolve_device, synchronize


class OpCounter(TorchDispatchMode):
    """Count aten operators below autograd (includes backward and optimizer ops)."""

    def __init__(self) -> None:
        super().__init__()
        self.ops: Counter[str] = Counter()
        self.views: Counter[str] = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func.overloadpacket.__name__)
        (self.views if getattr(func, "is_view", False) else self.ops)[name] += 1
        return func(*args, **(kwargs or {}))


def load_instance_config(path: str | None) -> dict:
    cfg: dict = {}
    if path:
        raw = load_config(path)
        if isinstance(raw.get("instance"), dict):  # layout: instance: {type, ...}, run: {...}
            cfg = {k: v for k, v in raw["instance"].items() if k != "type"}
        else:  # layout: instance: name, config: {...}
            cfg = raw.get("config", raw)
            cfg = {
                k: v for k, v in cfg.items() if k not in ("instance", "curriculum", "run", "solver")
            }
    return cfg


class _StepClock(nefi.solve.Callback):
    """Timestamps every step; enters an op counter / profiler for a window of steps."""

    def __init__(self, count_from: int, count_n: int, prof_from: int, prof_n: int, device):
        self.t: list[float] = []
        self.count_from, self.count_n = count_from, count_n
        self.prof_from, self.prof_n = prof_from, prof_n
        self.device = device
        self.counter: OpCounter | None = None
        self.prof = None
        self.acts = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            self.acts.append(torch.profiler.ProfilerActivity.CUDA)

    def on_step(self, solver, state) -> None:
        k = len(self.t)  # index of the step that just finished
        synchronize(self.device)
        self.t.append(time.perf_counter())
        # the window [from, from + n) is entered after step from - 1 finished
        if self.count_n and k == self.count_from - 1:
            self.counter = OpCounter()
            self.counter.__enter__()
        if self.counter is not None and k == self.count_from + self.count_n - 1:
            self.counter.__exit__(None, None, None)
            self.counter_done = self.counter
            self.counter = None
        if self.prof_n and k == self.prof_from - 1:
            self.prof = torch.profiler.profile(activities=self.acts, record_shapes=False)
            self.prof.__enter__()
        if self.prof is not None and k == self.prof_from + self.prof_n - 1:
            synchronize(self.device)
            self.prof.__exit__(None, None, None)
            self.prof_done = self.prof
            self.prof = None


def solver_kwargs(args) -> dict:
    kw: dict = {}
    if args.compile != "none":
        kw["compile"] = args.compile if args.compile != "field" else "field"
    if args.cuda_graphs:
        kw["cuda_graphs"] = True
    if args.autocast != "none":
        kw["autocast"] = args.autocast
    return kw


def run_solver_loop(problem, stage, args, device) -> dict:
    """Time / count the real Solver loop on one stage."""
    n_count = args.count_steps if args.ops else 0
    n_prof = min(args.steps, 10) if args.table or args.trace else 0
    total = args.warmup + args.steps + n_count + n_prof
    st = copy.deepcopy(stage)
    st.steps = total
    cur = copy.deepcopy(problem.curriculum or nefi.Curriculum([st]))
    cur.stages = [st]
    cur.restarts = 1
    cur.early_stop_patience = None
    cur.discrepancy_tau = None
    cur.time_budget_s = None
    t_from = args.warmup
    count_from = args.warmup + args.steps
    prof_from = count_from + n_count
    clock = _StepClock(count_from, n_count, prof_from, n_prof, device)
    kw = solver_kwargs(args)
    compat = {"compile": True} if kw.get("compile") == "field" else {}
    try:
        solver = nefi.Solver(problem, cur, device=device, seed=0, callbacks=[clock], **kw)
    except TypeError:  # older library without the new keyword arguments
        if set(kw) - {"compile"}:
            raise
        solver = nefi.Solver(problem, cur, device=device, seed=0, callbacks=[clock], **compat)
    t_start = time.perf_counter()
    solver.run()
    t = clock.t
    dts = [b - a for a, b in zip(t[t_from - 1 : count_from - 1], t[t_from:count_from])]
    if t_from == 0:
        dts = [t[0] - t_start, *dts]
    out = {
        "ms_median": 1e3 * statistics.median(dts),
        "ms_mean": 1e3 * statistics.fmean(dts),
        "first_step_ms": 1e3 * (t[0] - t_start),
    }
    c = getattr(clock, "counter_done", None)
    if c is not None:
        out["ops_per_step"] = sum(c.ops.values()) / n_count
        out["view_ops_per_step"] = sum(c.views.values()) / n_count
        out["top_ops"] = {k: v / n_count for k, v in c.ops.most_common(12)}
    out["prof"] = getattr(clock, "prof_done", None)
    return out


def run_minimal_loop(problem, stage, args, device) -> dict:
    """The original stand-alone loop (no solver bookkeeping)."""
    dom = problem.domain if stage.shape is None else problem.domain.at(stage.shape)
    coords = dom.coords(device=device)
    op = problem.operator.at_resolution(dom.shape).to(device)
    obs = problem.measurement_at(dom.shape).to(device)
    problem.field.on_stage_start(stage, dom)
    problem.field.to(device)
    field = torch.compile(problem.field) if args.compile == "field" else problem.field
    losses = problem.losses.with_weights(stage.loss_weights)
    params = [p for p in problem.field.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=stage.lr)

    def step(progress: float):
        opt.zero_grad(set_to_none=True)
        fields = field(coords, progress)
        pred = op(fields)
        ctx = nefi.losses.Context(fields, pred, obs, dom, op, problem.field, stage, 0, progress)
        total, _ = losses(ctx)
        total.backward()
        opt.step()
        return total

    for i in range(args.warmup):
        step(i / max(1, args.steps))
    synchronize(device)
    dts = []
    for i in range(args.steps):
        t0 = time.perf_counter()
        step(min(1.0, i / max(1, args.steps)))
        synchronize(device)
        dts.append(time.perf_counter() - t0)
    out = {"ms_median": 1e3 * statistics.median(dts), "ms_mean": 1e3 * statistics.fmean(dts)}
    if args.ops:
        c = OpCounter()
        with c:
            for _ in range(args.count_steps):
                step(1.0)
        out["ops_per_step"] = sum(c.ops.values()) / args.count_steps
        out["view_ops_per_step"] = sum(c.views.values()) / args.count_steps
        out["top_ops"] = {k: v / args.count_steps for k, v in c.ops.most_common(12)}
    out["prof"] = None
    if args.table or args.trace:
        acts = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=acts, record_shapes=False) as prof:
            for _ in range(min(args.steps, 10)):
                step(1.0)
            synchronize(device)
        out["prof"] = prof
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--config", default=None, help="YAML with the instance config")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=20, help="timed steps")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--count-steps", type=int, default=2, help="steps for the op count")
    ap.add_argument("--stage", type=int, default=-1, help="curriculum stage index to profile")
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
    ap.add_argument(
        "--inductor-threads",
        type=int,
        default=None,
        help="torch._inductor.config.cpp.threads (1 avoids OpenMP runtime clashes of compiled CPU "
        "kernels in some conda installs)",
    )
    ap.add_argument("--compile", default="none", choices=["none", "field", "step"])
    ap.add_argument("--cuda-graphs", action="store_true")
    ap.add_argument("--autocast", default="none", choices=["none", "bf16", "fp16"])
    ap.add_argument("--minimal", action="store_true", help="old stand-alone loop")
    ap.add_argument("--no-ops", dest="ops", action="store_false", help="skip the op count")
    ap.add_argument("--table", action="store_true", help="print the torch.profiler table")
    ap.add_argument("--trace", default=None, help="directory for a chrome trace")
    ap.add_argument("--json", default=None, help="append the summary as a JSON line")
    args = ap.parse_args()
    if args.compile != "none" and args.ops:
        # compiled graphs run generated kernels outside the aten dispatcher (and an active
        # dispatch mode forces dynamo back to eager): an op count would not mean anything here
        print("note: --compile given, op counting disabled")
        args.ops = False

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.inductor_threads:
        import torch._inductor.config as inductor_config

        inductor_config.cpp.threads = args.inductor_threads
    cfg = load_instance_config(args.config)
    for item in args.overrides:
        k, _, v = item.partition("=")
        cfg[k.strip()] = yaml.safe_load(v)
    inst = build("instance", {"type": args.instance, **cfg})
    device = resolve_device(args.device)
    gt, meas = inst.make_measurement(seed=0)
    problem = inst.build_problem(meas)
    cur = problem.curriculum or inst.default_curriculum()
    stage = cur.stages[args.stage]
    problem.curriculum = cur
    reset_peak_memory(device)
    run = run_minimal_loop if args.minimal else run_solver_loop
    if args.minimal:
        problem.to(device)
    out = run(problem, stage, args, device)
    dom = problem.domain if stage.shape is None else problem.domain.at(stage.shape)
    n_params = sum(p.numel() for p in problem.field.parameters() if p.requires_grad)
    ops = out.get("ops_per_step")
    print(
        f"instance={args.instance} stage={stage.name} shape={tuple(dom.shape)} device={device} "
        f"threads={torch.get_num_threads()} loop={'minimal' if args.minimal else 'solver'} "
        f"compile={args.compile} autocast={args.autocast}"
    )
    print(
        f"params={n_params:,}  step={out['ms_median']:.2f} ms (median; mean "
        f"{out['ms_mean']:.2f})  ops/step={ops if ops is None else round(ops)}"
        f" (+{round(out.get('view_ops_per_step', 0))} views)  peak_mem={peak_memory_mb(device)} MB"
    )
    if out.get("top_ops"):
        print("top ops/step: " + ", ".join(f"{k}={v:.0f}" for k, v in out["top_ops"].items()))
    prof = out.get("prof")
    if prof is not None and args.table:
        key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
        print(prof.key_averages().table(sort_by=key, row_limit=25))
    if prof is not None and args.trace:
        Path(args.trace).mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(Path(args.trace) / f"{args.instance}_trace.json"))
    if args.json:
        rec = {
            "instance": args.instance,
            "config": args.config,
            "overrides": args.overrides,
            "shape": list(dom.shape),
            "device": str(device),
            "threads": torch.get_num_threads(),
            "loop": "minimal" if args.minimal else "solver",
            "compile": args.compile,
            "autocast": args.autocast,
            "params": n_params,
            **{k: v for k, v in out.items() if k != "prof"},
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
