"""The per-measurement optimizer: runs a :class:`Curriculum` on an :class:`InverseProblem`.

Robustness features (all optional, see :class:`Curriculum` / :class:`OptimConfig`): gradient
clipping,
NaN guard with state rollback and LR back-off, EMA of parameters, early stopping on the data loss,
Morozov discrepancy stopping, multi-restart (best final data loss), wall-clock budget, checkpoint
callbacks, deterministic seeding.

Performance switches (all off by default; see ``docs/performance.md``): ``compile="field" |
"step"`` (``torch.compile`` of the field or of the whole field → operator → loss function),
``cuda_graphs=True`` (``mode="reduce-overhead"``) and ``autocast="bf16" | "fp16"`` (mixed precision
for the field MLP only; the physics stays in the solver dtype).
"""

from __future__ import annotations

import copy
import logging
import math
import time
import types
from collections import defaultdict
from collections.abc import Callable, Sequence

import torch

from ..config import config_hash, to_dict
from ..errors import ConfigError, SolverError
from ..fields.base import Field
from ..losses.base import Context
from ..utils.compat import cudagraph_mark_step_begin, grad_scaler
from ..utils.device import resolve_device, synchronize
from ..utils.seed import seed_everything
from .callbacks import Callback, StepState
from .curriculum import Curriculum, Stage
from .result import Result

log = logging.getLogger("nefi")

#: history keys written by the solver itself; a loss component with one of these names is stored
#: under ``loss/<name>`` so every history column keeps exactly one entry per step.
RESERVED_HISTORY_KEYS = frozenset(
    {"step", "global_step", "stage", "lr", "progress", "total", "data_loss"}
)

#: accepted values of ``Solver(compile=...)`` (``True`` means ``"field"``)
COMPILE_MODES = ("field", "step")
#: accepted values of ``Solver(autocast=...)``
AUTOCAST_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
}
#: CPU optimizers with at most this many parameter elements use the multi-tensor (``foreach``)
#: Adam(W) kernels: bitwise identical updates, ≈ 30 % faster steps for small fields (measured on
#: a desktop CPU); larger models are memory-bound and keep the per-tensor path (PyTorch's CPU
#: default). CUDA keeps PyTorch's default (``foreach``).
FOREACH_MAX_NUMEL = 1 << 20


def normalize_compile(compile: bool | str | None) -> str | None:
    """``False``/``None``/``"none"`` → ``None``; ``True``/``"field"`` → ``"field"``; ``"step"``."""
    if (
        compile is None
        or compile is False
        or (isinstance(compile, str) and compile in ("", "none"))
    ):
        return None
    if compile is True:
        return "field"
    if isinstance(compile, str) and compile.lower() in COMPILE_MODES:
        return compile.lower()
    raise ConfigError(f"compile must be False, True, 'field' or 'step', got {compile!r}")


def normalize_autocast(autocast: str | torch.dtype | None) -> torch.dtype | None:
    """``None`` / ``"bf16"`` / ``"fp16"`` (or the torch dtypes) → the autocast dtype."""
    if autocast is None or autocast is False or autocast == "none":
        return None
    if isinstance(autocast, torch.dtype):
        if autocast in (torch.bfloat16, torch.float16):
            return autocast
    elif str(autocast).lower() in AUTOCAST_DTYPES:
        return AUTOCAST_DTYPES[str(autocast).lower()]
    raise ConfigError(f"autocast must be None, 'bf16' or 'fp16', got {autocast!r}")


def _fresh_function(fn: Callable, tag: str) -> Callable:
    """Copy of ``fn`` with its own code object.

    Dynamo caches compiled graphs per code object and stops recompiling after
    ``torch._dynamo.config.recompile_limit`` (8) entries — every curriculum stage, restart and
    benchmark run would otherwise add an entry to the *same* code object and silently fall back to
    eager after eight of them.
    """
    code = fn.__code__.replace(co_name=f"{fn.__code__.co_name}_{tag}")
    new = types.FunctionType(code, fn.__globals__, code.co_name, fn.__defaults__, fn.__closure__)
    new.__kwdefaults__ = fn.__kwdefaults__
    return new


class _EMA:
    def __init__(self, params: Sequence[torch.nn.Parameter], decay: float) -> None:
        self.params = list(params)
        self.decay = decay
        self.shadow = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def update(self) -> None:
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self) -> None:
        for s, p in zip(self.shadow, self.params):
            p.copy_(s)


class Solver:
    """Run a curriculum of optimization stages on an inverse problem.

    Args:
        problem: the :class:`~nefi.problem.InverseProblem`.
        curriculum: stages / optimizer settings (default: ``Curriculum.multiscale(domain.shape)``).
        device: ``"auto"`` | ``"cpu"`` | ``"cuda"`` | ``"mps"``.
        dtype: computation dtype (float32 default; float64 for stiff physics if needed).
        callbacks: list of :class:`~nefi.solve.callbacks.Callback`.
        seed: global seed (restart ``r`` uses ``seed + r``).
        keep_fields_in_state: pass field tensors to callbacks' ``StepState`` (costs nothing but
        memory refs).
        compile: ``False`` (default, eager); ``True`` / ``"field"`` — ``torch.compile`` the field
            evaluation; ``"step"`` — compile field → operator → losses as one graph
            (``dynamic=False``; forward and backward are compiled by AOTAutograd). One compile
            per curriculum stage (fixed shapes); the annealing ``progress`` is passed as a 0-dim
            tensor so it never triggers a recompile. Operators with Python time loops / custom
            autograd functions (heat, wave, elliptic IFT) run eagerly inside the step graph
            (graph break). If compilation fails the solver logs a warning and continues eagerly.
            Loss terms that read ``ctx.step`` must set ``step_dependent = True`` (their stage
            then compiles only the field). Opt-in for CUDA servers: pays a one-time compile
            (seconds to minutes) for fewer kernel launches per step.
        cuda_graphs: with ``compile``, use ``mode="reduce-overhead"`` (CUDA graphs: one graph
            launch per step instead of one launch per kernel; CUDA only, ignored elsewhere).
            Outputs of a step are only valid until the next step (callbacks must copy them).
        autocast: ``None`` (default), ``"bf16"`` or ``"fp16"``: run the field MLP under
            ``torch.autocast`` (device-appropriate); the field output layer, the heads, the
            operator (physics) and the losses stay in ``dtype``. ``"fp16"`` adds dynamic loss
            scaling (``torch.amp.GradScaler``); not with L-BFGS.
        nan_guard: on non-finite loss, roll back to the last good state and halve the LR.
    """

    def __init__(
        self,
        problem,
        curriculum: Curriculum | None = None,
        *,
        device: str | torch.device = "auto",
        dtype: torch.dtype = torch.float32,
        callbacks: Sequence[Callback] = (),
        seed: int | None = 0,
        keep_fields_in_state: bool = True,
        compile: bool | str = False,
        nan_guard: bool = True,
        checkpoint_every: int = 25,
        max_bad_steps: int = 10,
        cuda_graphs: bool = False,
        autocast: str | torch.dtype | None = None,
    ) -> None:
        self.problem = problem
        self.curriculum = (
            curriculum
            or getattr(problem, "curriculum", None)
            or Curriculum.multiscale(problem.domain.shape)
        )
        self.device = resolve_device(device)
        self.dtype = dtype
        self.callbacks = list(callbacks)
        self.seed = seed
        self.keep_fields_in_state = keep_fields_in_state
        self.compile = compile
        self.compile_mode = normalize_compile(compile)
        self.cuda_graphs = bool(cuda_graphs)
        if self.cuda_graphs and self.compile_mode is None:
            log.warning("cuda_graphs=True has no effect without compile='field' or 'step'")
        self.autocast = normalize_autocast(autocast)
        if self.autocast is not None and self.autocast == torch.float16:
            if self.curriculum.optim.optimizer.lower() == "lbfgs":
                raise ConfigError("autocast='fp16' (gradient scaling) is not supported with L-BFGS")
        self._compile_failed = False
        #: CUDA graph trees need a step marker per iteration (outputs of the previous step's
        #: replay may be overwritten by the next one)
        self._cudagraph_marks = (
            self.cuda_graphs and self.compile_mode is not None and self.device.type == "cuda"
        )
        self._scaler = None
        self._progress_cache: tuple | None = None
        self.nan_guard = nan_guard
        self.checkpoint_every = checkpoint_every
        self.max_bad_steps = max_bad_steps
        self.history: dict[str, list[float]] = defaultdict(list)
        self.stage_results: list[dict] = []
        self.global_step = 0
        self._t0 = 0.0
        self._stop_all = False
        #: Callbacks may set this (in [0, 1]) to drive frequency annealing adaptively instead of
        #: the stage's step-based schedule (e.g. residual- or operator-aware annealing).
        self.progress_override: float | None = None
        #: Current stage optimizer and its parameter list (set in ``_run_stage``).
        self.optimizer: torch.optim.Optimizer | None = None
        self.stage_params: list[torch.nn.Parameter] = []
        #: The active stage's :class:`~nefi.losses.LossSet` (per-stage weight copy).
        self.stage_losses = None
        #: Callbacks may set this to end the current stage early (reason string).
        self.stop_stage: str | None = None

    # ------------------------------------------------------------------------------------
    def run(self) -> Result:
        cur = self.curriculum
        p = self.problem
        self._t0 = time.perf_counter()
        seed_everything(self.seed)
        p.to(self.device, self.dtype)
        for cb in self.callbacks:
            cb.on_run_start(self)

        best_state = None
        best_data = math.inf
        best_hist, best_stages = None, None
        for r in range(cur.restarts):
            if r > 0:
                seed_everything(None if self.seed is None else self.seed + r)
                p.field.reset_parameters()
                p.field.to(self.device, self.dtype)
            self.history = defaultdict(list)
            self.stage_results = []
            self.global_step = 0
            self._stop_all = False
            for i, stage in enumerate(cur.stages):
                self._run_stage(i, stage)
                if self._stop_all:
                    break
            data = self._final_data_loss()
            log.info("restart %d/%d: final data loss %.4e", r + 1, cur.restarts, data)
            if data < best_data or best_state is None:
                best_data = data
                best_state = (
                    copy.deepcopy(p.field.state_dict()),
                    copy.deepcopy(p.operator.state_dict()),
                )
                best_hist, best_stages = dict(self.history), list(self.stage_results)
        if cur.restarts > 1 and best_state is not None:
            p.field.load_state_dict(best_state[0])
            p.operator.load_state_dict(best_state[1])
            self.history, self.stage_results = defaultdict(list, best_hist), best_stages

        result = self._finalize()
        for cb in self.callbacks:
            cb.on_run_end(self, result)
        return result

    # ------------------------------------------------------------------------------------
    def _trainable_named(self, stage: Stage) -> list[tuple[str, torch.nn.Parameter]]:
        """Trainable ``(qualified_name, parameter)`` pairs for a stage (honours ``freeze``)."""
        named = []
        for prefix, module in (
            ("field.", self.problem.field),
            ("operator.", self.problem.operator),
        ):
            for name, prm in module.named_parameters():
                if not prm.requires_grad:
                    continue
                full = prefix + name
                if any(full.startswith(f) or name.startswith(f) for f in stage.freeze):
                    continue
                named.append((full, prm))
        if not named:
            raise SolverError("no trainable parameters (everything frozen?)")
        return named

    def _trainable_params(self, stage: Stage) -> list[torch.nn.Parameter]:
        return [p for _, p in self._trainable_named(stage)]

    def _param_groups(self, stage: Stage) -> list[dict]:
        """Optimizer parameter groups with per-prefix learning-rate multipliers."""
        mults = self.curriculum.optim.lr_mult or {}
        groups: dict[float, list[torch.nn.Parameter]] = {}
        for full, prm in self._trainable_named(stage):
            m = 1.0
            best = -1
            for prefix, value in mults.items():
                if (full.startswith(prefix) or full.split(".", 1)[-1].startswith(prefix)) and len(
                    prefix
                ) > best:
                    m, best = float(value), len(prefix)
            groups.setdefault(m, []).append(prm)
        return [{"params": ps, "lr_mult": m} for m, ps in groups.items()]

    def _adam_kwargs(self, params) -> dict:
        """``foreach=True`` for small CPU models (see :data:`FOREACH_MAX_NUMEL`)."""
        flat = (
            [q for g in params for q in g["params"]]
            if params and isinstance(params[0], dict)
            else params
        )
        if self.device.type == "cpu" and sum(q.numel() for q in flat) <= FOREACH_MAX_NUMEL:
            return {"foreach": True}
        return {}

    def _build_optimizer(self, params, stage: Stage):
        oc = self.curriculum.optim
        name = oc.optimizer.lower()
        if isinstance(params, list) and params and isinstance(params[0], dict):
            for g in params:
                g["lr"] = stage.lr * g.get("lr_mult", 1.0)
        if name == "adamw":
            return torch.optim.AdamW(
                params,
                lr=stage.lr,
                weight_decay=oc.weight_decay,
                betas=oc.betas,
                **self._adam_kwargs(params),
            )
        if name == "adam":
            return torch.optim.Adam(
                params,
                lr=stage.lr,
                weight_decay=oc.weight_decay,
                betas=oc.betas,
                **self._adam_kwargs(params),
            )
        if name == "sgd":
            return torch.optim.SGD(params, lr=stage.lr, momentum=0.9, weight_decay=oc.weight_decay)
        if name == "lbfgs":
            return torch.optim.LBFGS(
                params,
                lr=stage.lr,
                history_size=oc.lbfgs_history,
                max_iter=oc.lbfgs_max_iter,
                line_search_fn="strong_wolfe",
            )
        raise SolverError(f"unknown optimizer {oc.optimizer!r}")

    def _run_stage(self, sidx: int, stage: Stage) -> None:
        p = self.problem
        cur = self.curriculum
        dom = p.domain if stage.shape is None else p.domain.at(stage.shape)
        coords = dom.coords(device=self.device, dtype=self.dtype)
        op = p.operator.at_resolution(dom.shape).to(self.device)
        obs = p.measurement_at(dom.shape).to(self.device, self.dtype)
        p.field.on_stage_start(stage, dom)
        p.field.to(self.device, self.dtype)
        losses = p.losses.with_weights(stage.loss_weights)
        self.stage_losses = losses  # exposed for callbacks (adaptive loss balancing)
        compiled_field, compiled_step = self._compiled_callables(
            sidx, stage, p, op, losses, obs, dom
        )
        if self.autocast == torch.float16 and self._scaler is None:
            self._scaler = grad_scaler(self.device.type)
        params = self._trainable_params(stage)
        groups = self._param_groups(stage)
        opt = self._build_optimizer(groups if len(groups) > 1 else params, stage)
        self.optimizer = opt  # exposed for callbacks (e.g. to reset state of re-used parameters)
        self.stage_params = params
        is_lbfgs = isinstance(opt, torch.optim.LBFGS)
        ema = _EMA(params, cur.optim.ema) if cur.optim.ema else None
        self.progress_override = None
        self.stop_stage = None
        for cb in self.callbacks:
            cb.on_stage_start(self, sidx, stage)

        t_stage = time.perf_counter()
        good_state = [prm.detach().clone() for prm in params] if self.nan_guard else None
        bad_steps, lr_mult = 0, 1.0
        best_data, since_best = math.inf, 0
        last_comps: dict[str, float] = {}
        steps_run = 0
        stop_reason = "completed"
        last_progress = stage.progress_at(0)
        override_used = False
        noise_std = obs.noise_std
        if torch.is_tensor(noise_std):
            noise_std = float(noise_std.mean())

        for step in range(stage.steps):
            lr = lr_mult * stage.lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr * g.get("lr_mult", 1.0)
            if self.progress_override is not None:
                progress = float(self.progress_override)
                override_used = True
            else:
                progress = stage.progress_at(step)
            holder: dict = {}

            def closure(step=step, progress=progress, holder=holder):
                opt.zero_grad(set_to_none=True)
                # the first step of a stage runs eagerly: it fills the operators' caches (FFT
                # kernels, initial states, ...) so that the compiled graph never traces a cache miss
                compiled = step > 0 and not self._compile_failed
                if compiled and self._cudagraph_marks:
                    cudagraph_mark_step_begin()
                if compiled and compiled_step is not None:
                    try:
                        total, comps, fields, pred = compiled_step(
                            coords, self._progress_arg(progress, coords)
                        )
                        if total.requires_grad:
                            self._backward(total)
                        ctx = Context(fields, pred, obs, dom, op, p.field, stage, step, progress)
                        holder["out"] = (total, comps, ctx)
                        return total
                    except Exception as e:  # compilation / tracing failure -> eager
                        self._compile_fallback(e)
                        opt.zero_grad(set_to_none=True)
                fields = None
                if compiled and compiled_field is not None and not self._compile_failed:
                    try:
                        fields = compiled_field(coords, self._progress_arg(progress, coords))
                    except Exception as e:  # compilation / tracing failure -> eager
                        self._compile_fallback(e)
                if fields is None:
                    fields = self._eval_field(p.field, coords, progress)
                pred = op(fields)
                ctx = Context(fields, pred, obs, dom, op, p.field, stage, step, progress)
                total, comps = losses.forward_tensors(ctx)
                if total.requires_grad:
                    self._backward(total)
                holder["out"] = (total, comps, ctx)
                return total

            if is_lbfgs:
                opt.step(closure)
                total, comps_t, ctx = holder["out"]
            else:
                closure()
                total, comps_t, ctx = holder["out"]
            # one device synchronization per step: total + every component together
            vals = torch.stack([total.detach(), *comps_t.values()]).tolist()
            tot, comps = vals[0], dict(zip(comps_t.keys(), vals[1:]))
            if not is_lbfgs:
                if self.nan_guard and not math.isfinite(tot):
                    bad_steps += 1
                    lr_mult *= 0.5
                    log.warning(
                        "non-finite loss at stage %d step %d; rolling back, lr×%.3g",
                        sidx,
                        step,
                        lr_mult,
                    )
                    with torch.no_grad():
                        for prm, g in zip(params, good_state):
                            prm.copy_(g)
                    opt.state.clear()
                    if bad_steps > self.max_bad_steps:
                        raise SolverError(
                            "persistent non-finite loss; lower the learning rate or "
                            "check the operator / losses for numerical issues"
                        )
                    continue
                scaler = self._scaler
                if cur.optim.grad_clip:
                    if scaler is not None:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, cur.optim.grad_clip)
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                bad_steps = 0
                if self.nan_guard and step % self.checkpoint_every == 0:
                    good_state = [prm.detach().clone() for prm in params]
            if ema is not None:
                ema.update()

            # ---- bookkeeping ----
            steps_run += 1
            last_progress = progress
            data_loss = losses.data_loss(comps)
            elapsed = time.perf_counter() - self._t0
            self.history["step"].append(step)
            self.history["global_step"].append(self.global_step)
            self.history["stage"].append(sidx)
            self.history["lr"].append(lr)
            self.history["progress"].append(progress)
            self.history["total"].append(tot)
            self.history["data_loss"].append(data_loss)
            for k, v in comps.items():
                key = f"loss/{k}" if k in RESERVED_HISTORY_KEYS else k
                self.history[key].append(v)
            last_comps = comps
            state = StepState(
                sidx,
                stage,
                step,
                self.global_step,
                lr,
                progress,
                tot,
                comps,
                data_loss,
                elapsed,
                ctx.fields if self.keep_fields_in_state else None,
                ctx.pred if self.keep_fields_in_state else None,
            )
            for cb in self.callbacks:
                cb.on_step(self, state)
            self.global_step += 1

            # ---- stopping criteria (early/discrepancy stops wait for annealing to finish,
            # otherwise never-trained high-frequency bands would be switched on at evaluation) --
            annealed = progress >= 1.0
            if cur.early_stop_patience and annealed:
                if data_loss < best_data - cur.early_stop_min_delta:
                    best_data, since_best = data_loss, 0
                else:
                    since_best += 1
                    if since_best >= cur.early_stop_patience:
                        stop_reason = "early_stop"
                        break
            if cur.discrepancy_tau and noise_std and annealed:
                with torch.no_grad():
                    rmse = float(torch.sqrt(obs.masked_mean((ctx.pred - obs.data) ** 2)))
                if rmse <= cur.discrepancy_tau * noise_std:
                    stop_reason = "discrepancy"
                    break
            if cur.time_budget_s and elapsed > cur.time_budget_s:
                stop_reason = "time_budget"
                self._stop_all = True
                break
            if self.stop_stage:
                stop_reason = self.stop_stage
                self.stop_stage = None
                break

        if ema is not None:
            ema.copy_to()
        synchronize(self.device)
        # progress the field was actually trained at: an adaptive override (callback) wins, an
        # early stop keeps the last value, a completed step clock ends at progress_at(steps)
        if override_used or stop_reason != "completed":
            final_progress = last_progress
        else:
            final_progress = stage.progress_at(stage.steps)
        info = {
            "name": stage.name,
            "shape": tuple(dom.shape),
            "steps": steps_run,
            "seconds": time.perf_counter() - t_stage,
            "final": last_comps,
            "stop": stop_reason,
            "lr_mult": lr_mult,
            "final_progress": float(final_progress),
        }
        self.stage_results.append(info)
        for cb in self.callbacks:
            cb.on_stage_end(self, sidx, stage, info)

    # ------------------------------------------------------------------------------------
    # performance switches: compile / CUDA graphs / autocast
    # ------------------------------------------------------------------------------------
    def _eval_field(self, field, coords: torch.Tensor, progress) -> dict[str, torch.Tensor]:
        """Field evaluation; with ``autocast`` only the MLP trunk runs in reduced precision."""
        if self.autocast is None:
            return field(coords, progress)
        dev = coords.device.type
        if type(field).forward is Field.forward:  # heads(raw(...)): heads stay in full precision
            with torch.autocast(dev, dtype=self.autocast):
                raw = field.raw(coords, progress)
            return field.heads(raw.to(self.dtype), progress)
        with torch.autocast(dev, dtype=self.autocast):
            out = field(coords, progress)
        return {k: v.to(self.dtype) if v.is_floating_point() else v for k, v in out.items()}

    def _backward(self, total: torch.Tensor) -> None:
        if self._scaler is not None:
            self._scaler.scale(total).backward()
        else:
            total.backward()

    def _progress_arg(self, progress: float, like: torch.Tensor) -> torch.Tensor:
        """Annealing progress as a 0-dim tensor for compiled code (a Python float would be
        specialized, i.e. recompiled at every annealing step)."""
        c = self._progress_cache
        if c is not None and c[0] == progress and c[1].device == like.device:
            return c[1]
        dt = torch.float32 if like.device.type == "mps" else torch.float64
        t = torch.tensor(float(progress), dtype=dt, device=like.device)
        self._progress_cache = (progress, t)
        return t

    def _compile_fallback(self, err: Exception) -> None:
        self._compile_failed = True
        log.warning(
            "torch.compile(%s) failed (%s: %s); continuing eagerly",
            self.compile_mode,
            type(err).__name__,
            str(err).splitlines()[0][:300] if str(err) else "",
        )

    def _torch_compile(self, fn: Callable, tag: str) -> Callable | None:
        mode = "reduce-overhead" if (self.cuda_graphs and self.device.type == "cuda") else None
        try:
            return torch.compile(_fresh_function(fn, tag), dynamic=False, mode=mode)
        except Exception as e:  # pragma: no cover - platform dependent
            self._compile_fallback(e)
            return None

    def _compiled_callables(self, sidx, stage, p, op, losses, obs, dom):
        """``(compiled_field, compiled_step)`` for this stage (``None`` when not compiling)."""
        if self.compile_mode is None or self._compile_failed:
            return None, None
        tag = f"s{sidx}_{id(self):x}"
        field = p.field
        eval_field = self._eval_field
        mode = self.compile_mode
        if mode == "step" and any(
            getattr(t, "step_dependent", False)
            for k, t in losses.terms.items()
            if losses.weights[k] != 0.0
        ):
            log.warning("a loss term reads ctx.step: compiling only the field in stage %d", sidx)
            mode = "field"
        if mode == "field":

            def field_call(coords, progress):
                return eval_field(field, coords, progress)

            return self._torch_compile(field_call, tag), None

        # operators with Python time loops / custom autograd functions run eagerly between the
        # compiled field graph and the compiled loss graph (a deliberate graph break)
        op_call = op if getattr(op, "traceable", True) else torch.compiler.disable(op)

        def step_call(coords, progress):
            fields = eval_field(field, coords, progress)
            pred = op_call(fields)
            # the step index is a Python int: it is held constant inside the graph (see the
            # ``compile`` docstring), the progress is a traced 0-dim tensor
            ctx = Context(fields, pred, obs, dom, op, field, stage, 0, progress)
            total, comps = losses.forward_tensors(ctx)
            return total, comps, fields, pred

        return None, self._torch_compile(step_call, tag)

    # ------------------------------------------------------------------------------------
    def _final_shape(self) -> tuple[int, ...]:
        s = self.curriculum.final_shape
        return tuple(self.problem.domain.shape) if s is None else tuple(s)

    @torch.no_grad()
    def _final_data_loss(self) -> float:
        h = self.history.get("data_loss")
        return float(h[-1]) if h else math.inf

    def _finalize(self) -> Result:
        p = self.problem
        shape = self._final_shape()
        # evaluate at the annealing progress that was actually trained (a stage that stopped
        # early must not switch on never-trained encoding bands)
        progress = self.stage_results[-1].get("final_progress", 1.0) if self.stage_results else 1.0
        raw_fields, pred = p.evaluate(shape, progress=progress)
        fields = dict(raw_fields)
        post_info: dict = {}
        for pp in p.postprocess:
            fields, info = pp(fields, pred, p, shape)
            post_info.update(info)
        if p.postprocess:
            with torch.no_grad():
                pred = p.operator.at_resolution(shape)(fields)
        cpu = lambda d: {k: v.detach().cpu() for k, v in d.items()}  # noqa: E731
        total_s = time.perf_counter() - self._t0
        cfg = {"curriculum": to_dict(self.curriculum), "problem": p.describe()}
        return Result(
            fields=cpu(fields),
            raw_fields=cpu(raw_fields),
            pred=pred.detach().cpu(),
            history=dict(self.history),
            stage_results=list(self.stage_results),
            timing={"total_s": total_s, "per_stage_s": [s["seconds"] for s in self.stage_results]},
            post_info=post_info,
            config_hash=config_hash(cfg),
            extra={
                "device": str(self.device),
                "dtype": str(self.dtype),
                "seed": self.seed,
                "n_parameters": p.field.n_parameters(),
            },
        )
