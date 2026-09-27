"""Batched multi-measurement solving: ``N`` independent problems in **one** optimization loop.

Benchmarks and campaigns solve the same inverse problem (same domain, field architecture,
forward operator, losses and curriculum) for many measurements and seeds. On a GPU a single small
problem is launch-bound — a 64² NeTMY step is a few hundred tiny kernels — so solving them one
after the other wastes the device. :func:`batch_invert` stacks the field parameters of all
problems (a leading problem axis, as ``torch.func.stack_module_state``; trainable parameters are
flattened into one ``(N, P)`` leaf per learning-rate group so that the optimizer, clipping, EMA and
NaN guard are a handful of kernels), evaluates all fields with ``torch.func.vmap`` over
``torch.func.functional_call``, calls the operator **once** on the stacked fields ``(N, *shape)``
and reduces the losses *per problem*:

* the objective is ``Σ_b L_b`` — each ``L_b`` is the problem's own mean-reduced objective, so the
  gradient of problem ``b``'s parameters is exactly its sequential gradient;
* Adam / AdamW run element-wise with per-problem step counts, learning-rate multipliers and state
  (the arithmetic of ``torch.optim.Adam(W)``), gradient clipping uses per-problem norms, and the
  NaN guard, EMA, early stopping and the discrepancy principle act per problem (a stopped problem
  is frozen until the stage ends).

Hence every problem follows its sequential trajectory up to float rounding (batched vs single
matrix products; ``tests/test_performance.py`` checks batched == sequential to 1e-5 on toy1d and
deconvolution).

Validity: the operator must support a leading batch axis (:attr:`Operator.batchable
<nefi.operators.Operator.batchable>`: FFT convolution, NV F1/F2, deconvolution, Poisson (DST),
Born, holography, Radon, planar magnetostatics, pointwise maps) and have no trainable parameters;
time-stepping and iterative solvers with custom ``autograd.Function`` s (heat adjoint, wave,
elliptic IFT) are not batchable and are refused with a clear message. Losses are evaluated with
``vmap`` over one shared :class:`~nefi.losses.LossSet` when all problems' loss sets are identical
(same terms, weights, hyper-parameters and buffers; verified structurally and numerically at every
stage start), otherwise problem by problem. Loss terms that *evaluate* ``ctx.field_module`` (e.g.
collocation PINN losses) are not supported. Optimizers: ``adam`` / ``adamw``; no callbacks.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from ..errors import ConfigError, SolverError
from ..losses.base import Context, LossSet
from ..measurement import Measurement
from ..utils.device import resolve_device, synchronize
from ..utils.seed import seed_everything
from .curriculum import Curriculum, Stage
from .result import Result
from .solver import RESERVED_HISTORY_KEYS, Solver

log = logging.getLogger("nefi")

SUPPORTED_OPTIMIZERS = ("adam", "adamw")
LOSS_MODES = ("auto", "vmap", "loop")

__all__ = ["LOSS_MODES", "SUPPORTED_OPTIMIZERS", "batch_invert", "batchable_reason"]


# ------------------------------------------------------------------------------------------
# validation
# ------------------------------------------------------------------------------------------
def batchable_reason(problem: Any, curriculum: Curriculum | None = None) -> str | None:
    """``None`` if :func:`batch_invert` can solve ``problem``, else the reason why not."""
    meta = getattr(problem, "meta", None) or {}
    if meta.get("solver") not in (None, "gradient", "solver") or meta.get("callbacks"):
        return "it uses a non-gradient solver or solver callbacks (baseline)"
    op = problem.operator
    if not getattr(op, "batchable", False):
        return (
            f"its operator {type(op).__name__} does not support a leading batch axis "
            "(Operator.batchable is False: time-stepping / iterative solvers with custom "
            "autograd functions — heat adjoint, wave, elliptic IFT — are solved one measurement "
            "at a time with nefi.invert / Solver)"
        )
    if any(q.requires_grad for q in op.parameters()):
        return f"its operator {type(op).__name__} has trainable (nuisance) parameters"
    cur = curriculum or getattr(problem, "curriculum", None)
    if cur is not None and cur.optim.optimizer.lower() not in SUPPORTED_OPTIMIZERS:
        return f"optimizer {cur.optim.optimizer!r} is not supported ({SUPPORTED_OPTIMIZERS} are)"
    return None


def _check_problems(problems: Sequence[Any], cur: Curriculum) -> None:
    if not problems:
        raise ConfigError("batch_invert needs at least one problem")
    for i, p in enumerate(problems):
        why = batchable_reason(p, cur)
        if why is not None:
            raise ConfigError(f"batch_invert cannot solve problem {i} ({p.name}): {why}")
    ref = problems[0]
    names = [(n, tuple(q.shape)) for n, q in ref.field.named_parameters()]
    bufs = dict(ref.field.named_buffers())
    for i, p in enumerate(problems[1:], 1):
        if p.domain != ref.domain:
            raise ConfigError(f"problem {i} has domain {p.domain}, problem 0 has {ref.domain}")
        if type(p.field) is not type(ref.field) or names != [
            (n, tuple(q.shape)) for n, q in p.field.named_parameters()
        ]:
            raise ConfigError(
                f"problem {i}'s field differs in type or parameter shapes from problem 0's; "
                "batch_invert stacks identical field architectures"
            )
        for n, b in p.field.named_buffers():
            ref_b = bufs.get(n)
            if ref_b is None or ref_b.shape != b.shape or not torch.equal(ref_b.cpu(), b.cpu()):
                raise ConfigError(f"problem {i}'s field buffer {n!r} differs from problem 0's")
        if type(p.operator) is not type(ref.operator):
            raise ConfigError(
                f"problem {i}'s operator {type(p.operator).__name__} differs from problem 0's "
                f"{type(ref.operator).__name__}"
            )


def _module_fingerprint(m: nn.Module) -> tuple:
    """Structure + hyper-parameters (plain attributes) of a module tree, without tensors."""
    items = []
    for name, mod in m.named_modules():
        attrs = []
        for k, v in sorted(vars(mod).items()):
            if k.startswith("_") or k == "training":
                continue
            if isinstance(v, int | float | str | bool | type(None)):
                attrs.append((k, v))
            elif isinstance(v, tuple | list) and all(
                isinstance(x, int | float | str | bool | type(None)) for x in v
            ):
                attrs.append((k, tuple(v)))
            elif isinstance(v, dict) and all(isinstance(x, int | float | str) for x in v.values()):
                attrs.append((k, tuple(sorted(v.items()))))
        items.append((name, type(mod).__qualname__, tuple(attrs)))
    return tuple(items)


def _module_tensors(m: nn.Module) -> list[tuple[str, torch.Tensor]]:
    out = [(n, t) for n, t in m.named_buffers()] + [(n, t) for n, t in m.named_parameters()]
    for name, mod in m.named_modules():
        for k, v in vars(mod).items():
            if torch.is_tensor(v) and not k.startswith("_"):
                out.append((f"{name}.{k}", v))
    return out


def _losses_equivalent(a: LossSet, b: LossSet) -> bool:
    """Whether two loss sets compute the same function of a context (weights, terms, tensors)."""
    if a is b:
        return True
    if a.weights != b.weights or _module_fingerprint(a) != _module_fingerprint(b):
        return False
    ta, tb = _module_tensors(a), _module_tensors(b)
    if [n for n, _ in ta] != [n for n, _ in tb]:
        return False
    return all(
        x.shape == y.shape and torch.equal(x.cpu(), y.cpu()) for (_, x), (_, y) in zip(ta, tb)
    )


class _FieldProxy:
    """Stands in for ``ctx.field_module``: metadata (``primary``, ``names``, ``heads``, …) is
    available, evaluating it is not (the module's own parameters are not the stacked ones)."""

    _FORBIDDEN = frozenset({"forward", "raw", "parameters", "named_parameters", "state_dict"})

    def __init__(self, field: nn.Module) -> None:
        object.__setattr__(self, "_field", field)

    def __getattr__(self, name: str):
        if name in self._FORBIDDEN:
            self._refuse()
        return getattr(self._field, name)

    def __call__(self, *args, **kwargs):
        self._refuse()

    @staticmethod
    def _refuse():
        raise SolverError(
            "batch_invert: a loss term evaluates ctx.field_module (e.g. collocation / PINN "
            "losses); such problems must be solved one at a time with nefi.invert / Solver"
        )


# ------------------------------------------------------------------------------------------
# per-problem Adam(W) on flat parameter groups
# ------------------------------------------------------------------------------------------
class _FlatGroup:
    """Trainable parameters sharing a learning-rate multiplier, stored as one leaf ``(B, P)``.

    Every field parameter of shape ``s`` is the view ``flat[:, o:o+n].view(B, *s)``; the Adam /
    clipping / EMA / NaN-guard updates act on the whole group with a handful of kernels.
    """

    def __init__(self, mult: float, names: list[str], stacked: dict[str, torch.Tensor]) -> None:
        self.mult = float(mult)
        self.names = list(names)
        self.shapes = [tuple(stacked[n].shape[1:]) for n in names]
        self.sizes = [math.prod(sh) for sh in self.shapes]
        b = stacked[names[0]].shape[0]
        self.flat = torch.cat([stacked[n].reshape(b, -1) for n in names], dim=1).requires_grad_()

    def views(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Parameter views of ``flat`` (``(B, P)`` or, inside ``vmap``, ``(P,)``)."""
        out, o = {}, 0
        lead = flat.shape[:-1]
        for n, sh, sz in zip(self.names, self.shapes, self.sizes):
            out[n] = flat[..., o : o + sz].view(*lead, *sh)
            o += sz
        return out


class _BatchedAdam:
    """Adam / AdamW on flat parameter groups with a leading problem axis, independent per problem.

    Reproduces the element-wise arithmetic of ``torch.optim.Adam`` / ``AdamW`` (single-tensor and
    ``foreach`` paths: bias corrections in double precision, ``p ← p + (−η/bc₁)·m / (√v/√bc₂ + ε)``)
    with a per-problem step count and learning rate, so each problem's trajectory is its
    sequential one (the first update is bitwise identical). Problems whose ``update`` flag is off
    get learning rate 0 and a zero gradient: their parameters are left bit-for-bit unchanged.
    """

    def __init__(
        self,
        groups: list[_FlatGroup],
        n: int,
        *,
        decoupled: bool,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float = 1e-8,
    ) -> None:
        self.groups = groups
        self.n = n
        self.decoupled = decoupled
        self.wd = float(weight_decay)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = eps
        self.exp_avg = [torch.zeros_like(g.flat) for g in groups]
        self.exp_avg_sq = [torch.zeros_like(g.flat) for g in groups]
        self.steps = [0] * n

    def reset(self, b: int) -> None:
        """Clear problem ``b``'s state (NaN guard: ``opt.state.clear()`` of the sequential run)."""
        with torch.no_grad():
            for m, v in zip(self.exp_avg, self.exp_avg_sq):
                m[b].zero_()
                v[b].zero_()
        self.steps[b] = 0

    @staticmethod
    def _col(values: list[float], like: torch.Tensor) -> torch.Tensor:
        """Per-problem scalars (computed in double like torch's Python scalars) as ``(B, 1)``."""
        t = torch.tensor(values, dtype=torch.float64)
        return t.to(device=like.device, dtype=like.dtype).view(-1, 1)

    @torch.no_grad()
    def step(self, lr: list[float], update: list[bool]) -> None:
        """One step; ``lr[b]`` is problem ``b``'s learning rate (before the group multiplier)."""
        for b, u in enumerate(update):
            if u:
                self.steps[b] += 1
        b1, b2 = self.beta1, self.beta2
        steps = [max(s, 1) for s in self.steps]
        keep = None
        for grp, m, v in zip(self.groups, self.exp_avg, self.exp_avg_sq):
            p, g = grp.flat, grp.flat.grad
            if g is None:
                continue
            lr_g = [lr_b * grp.mult if u else 0.0 for lr_b, u in zip(lr, update)]
            if not all(update):
                if keep is None:
                    keep = torch.tensor(update, device=p.device).view(-1, 1)
                g = torch.where(keep, g, torch.zeros((), device=p.device, dtype=p.dtype))
            if self.wd != 0:
                if self.decoupled:
                    p.mul_(self._col([1 - lr_b * self.wd for lr_b in lr_g], p))
                else:
                    g = g.add(p, alpha=self.wd)
            m.lerp_(g, 1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            neg_step = self._col([-(lr_b / (1 - b1**s)) for lr_b, s in zip(lr_g, steps)], p)
            bc2_sqrt = self._col([(1 - b2**s) ** 0.5 for s in steps], p)
            denom = (v.sqrt() / bc2_sqrt).add_(self.eps)
            # torch: param.addcdiv_(exp_avg, denom, value=-step_size) = p + (value · m) / denom
            p.add_((neg_step * m) / denom)


# ------------------------------------------------------------------------------------------
# the batched run
# ------------------------------------------------------------------------------------------
def _seeds(seeds: int | Sequence[int | None] | None, n: int) -> list[int | None]:
    if seeds is None or isinstance(seeds, int):
        return [seeds] * n
    s = list(seeds)
    if len(s) != n:
        raise ConfigError(f"got {len(s)} seeds for {n} problems")
    return s


class _BatchRun:
    """State of one restart of a batched solve (all stages)."""

    def __init__(self, problems, cur: Curriculum, device, dtype, opts: dict, t0: float) -> None:
        self.problems = list(problems)
        self.n = len(self.problems)
        self.cur = cur
        self.device, self.dtype = device, dtype
        self.opts = opts
        self.t0 = t0
        self.history: list[dict[str, list[float]]] = [defaultdict(list) for _ in problems]
        self.stage_results: list[list[dict]] = [[] for _ in problems]
        self.global_step = [0] * self.n
        self.stop_all = False
        self.failed: dict[int, str] = {}
        self.loss_modes: list[str] = []

    # -- helpers ---------------------------------------------------------------------------
    def _stack_params(self) -> dict[str, torch.Tensor]:
        mods = [dict(p.field.named_parameters()) for p in self.problems]
        return {n: torch.stack([m[n].detach() for m in mods]) for n in mods[0]}

    def _write_back(self, params: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for b, p in enumerate(self.problems):
                for n, q in p.field.named_parameters():
                    q.copy_(params[n][b])

    def _trainable(self, stage: Stage) -> list[str]:
        out = []
        for name, prm in self.problems[0].field.named_parameters():
            full = "field." + name
            if not prm.requires_grad:
                continue
            if any(full.startswith(f) or name.startswith(f) for f in stage.freeze):
                continue
            out.append(name)
        if not out:
            raise SolverError("no trainable parameters (everything frozen?)")
        return out

    def _lr_mults(self, names: list[str]) -> dict[str, float]:
        mults = self.cur.optim.lr_mult or {}
        out = {}
        for name in names:
            full = "field." + name
            m, best = 1.0, -1
            for prefix, value in mults.items():
                hit = full.startswith(prefix) or full.split(".", 1)[-1].startswith(prefix)
                if hit and len(prefix) > best:
                    m, best = float(value), len(prefix)
            out[name] = m
        return out

    # -- one stage ---------------------------------------------------------------------------
    def run_stage(self, sidx: int, stage: Stage) -> None:
        P, B, dev, dt = self.problems, self.n, self.device, self.dtype
        cur = self.cur
        dom = P[0].domain if stage.shape is None else P[0].domain.at(stage.shape)
        coords = dom.coords(device=dev, dtype=dt)
        ops = [p.operator.at_resolution(dom.shape).to(dev) for p in P]
        op = ops[0]
        if not getattr(op, "batchable", False):
            raise ConfigError(
                f"operator {type(op).__name__} at resolution {dom.shape} is not batchable"
            )
        obs = [p.measurement_at(dom.shape).to(dev, dt) for p in P]
        for p in P:
            p.field.on_stage_start(stage, dom)
            p.field.to(dev, dt)
        base = P[0].field
        stacked = self._stack_params()
        trainable = self._trainable(stage)
        mults = self._lr_mults(trainable)
        by_mult: dict[float, list[str]] = {}
        for n in trainable:
            by_mult.setdefault(mults[n], []).append(n)
        groups = [_FlatGroup(m, names, stacked) for m, names in by_mult.items()]
        frozen = {n: t for n, t in stacked.items() if n not in trainable}
        losses = [p.losses.with_weights(stage.loss_weights) for p in P]
        proxy = _FieldProxy(base)

        data = torch.stack([o.data for o in obs])
        masks = [o.mask for o in obs]
        mask = None
        if any(m is not None for m in masks):
            mask = torch.stack(
                [
                    torch.ones_like(o.data) if m is None else m.expand_as(o.data).to(o.data)
                    for o, m in zip(obs, masks)
                ]
            )
        ns_list = [o.noise_std for o in obs]
        ns_batched = None
        if any(ns is not None for ns in ns_list) and not all(
            not torch.is_tensor(ns) and ns == ns_list[0] for ns in ns_list
        ):
            ns_batched = torch.stack(
                [
                    torch.as_tensor(float("nan") if ns is None else ns, dtype=dt, device=dev)
                    .reshape(-1)
                    .mean()
                    for ns in ns_list
                ]
            )
        meta0 = dict(obs[0].meta)

        def all_params() -> dict[str, torch.Tensor]:
            """Every field parameter as a ``(B, ...)`` tensor (trainable ones are views)."""
            out = dict(frozen)
            for grp in groups:
                out.update(grp.views(grp.flat))
            return out

        def eval_fields(progress: float):
            def one(flats, frz):
                prm = dict(frz)
                for grp, f in zip(groups, flats):
                    prm.update(grp.views(f))
                return torch.func.functional_call(base, prm, (coords, progress))

            return torch.func.vmap(one)(tuple(grp.flat for grp in groups), frozen)

        def losses_vmap(fields, pred, step, progress):
            shared = losses[0]

            def one(fields_b, pred_b, data_b, mask_b, ns_b):
                ns = ns_list[0] if ns_batched is None else ns_b
                o = Measurement(data_b, mask_b, ns, meta0)
                ctx = Context(fields_b, pred_b, o, dom, op, proxy, stage, step, progress)
                return shared.forward_tensors(ctx)

            in_dims = (0, 0, 0, None if mask is None else 0, None if ns_batched is None else 0)
            return torch.func.vmap(one, in_dims=in_dims)(fields, pred, data, mask, ns_batched)

        def losses_loop(fields, pred, step, progress):
            totals, comps = [], []
            for b in range(B):
                fb = {k: v[b] for k, v in fields.items()}
                ctx = Context(fb, pred[b], obs[b], dom, ops[b], proxy, stage, step, progress)
                t, c = losses[b].forward_tensors(ctx)
                totals.append(t)
                comps.append(c)
            keys = list(comps[0])
            if any(list(c) != keys for c in comps):
                raise SolverError("problems have different active loss terms in this stage")
            return torch.stack(totals), {k: torch.stack([c[k] for c in comps]) for k in keys}

        # ---- stage-start checks: shared operator, loss evaluation mode ---------------------
        mode = self.opts["loss_mode"]
        if mode in ("auto", "vmap") and not all(_losses_equivalent(losses[0], lb) for lb in losses):
            if mode == "vmap":
                raise ConfigError("loss_mode='vmap' needs identical loss sets for all problems")
            mode = "loop"
        progress0 = stage.progress_at(0)
        if self.opts["check"] or mode == "auto":
            with torch.no_grad():
                fields0 = eval_fields(progress0)
                pred0 = op(fields0)
                if self.opts["check"]:
                    for b in range(B):
                        ref = ops[b]({k: v[b] for k, v in fields0.items()})
                        err = float((pred0[b] - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
                        if not err <= 1e-5:
                            raise SolverError(
                                f"batch_invert: problem {b}'s operator disagrees with the batched "
                                f"evaluation (relative error {err:.2e}); the problems must share "
                                "the forward operator and it must support a leading batch axis"
                            )
                if mode == "auto":
                    try:
                        tv, _ = losses_vmap(fields0, pred0, 0, progress0)
                        tl, _ = losses_loop(fields0, pred0, 0, progress0)
                        ok = bool(((tv - tl).abs() <= 1e-5 * tl.abs().clamp_min(1e-30)).all())
                        mode = "vmap" if ok else "loop"
                    except SolverError:
                        raise
                    except Exception as e:  # loss not vmap-compatible (host sync, data-dependent)
                        log.info(
                            "batch_invert: per-problem loss loop (%s: %s)", type(e).__name__, e
                        )
                        mode = "loop"
        self.loss_modes.append(mode)
        eval_losses = losses_vmap if mode == "vmap" else losses_loop

        # ---- optimizer / EMA / guards ------------------------------------------------------
        oc = cur.optim
        opt = _BatchedAdam(
            groups,
            B,
            decoupled=oc.optimizer.lower() == "adamw",
            weight_decay=oc.weight_decay,
            betas=oc.betas,
        )
        flats = [grp.flat for grp in groups]
        ema = [f.detach().clone() for f in flats] if oc.ema else None
        nan_guard = self.opts["nan_guard"]
        good = [f.detach().clone() for f in flats] if nan_guard else None
        active = [b not in self.failed for b in range(B)]
        bad_steps, lr_mult = [0] * B, [1.0] * B
        best_data, since_best = [math.inf] * B, [0] * B
        last_comps: list[dict[str, float]] = [{} for _ in range(B)]
        steps_run, stop_reason = [0] * B, ["completed"] * B
        for b in self.failed:
            stop_reason[b] = "failed"
        last_progress = [progress0] * B
        noise = []
        for o in obs:
            ns = o.noise_std
            noise.append(float(ns.mean()) if torch.is_tensor(ns) else ns)
        t_stage = time.perf_counter()

        for step in range(stage.steps):
            if not any(active):
                break
            lr_base = stage.lr_at(step)
            progress = stage.progress_at(step)
            for f in flats:
                f.grad = None
            fields = eval_fields(progress)
            pred = op(fields)
            totals, comps_t = eval_losses(fields, pred, step, progress)
            totals.sum().backward()
            # one device synchronization per step: all totals and components together
            keys = list(comps_t)
            vals = torch.stack([totals.detach(), *[comps_t[k] for k in keys]]).tolist()
            tots = vals[0]
            comps = [{k: vals[1 + i][b] for i, k in enumerate(keys)} for b in range(B)]

            update = list(active)
            for b in range(B):
                if active[b] and nan_guard and not math.isfinite(tots[b]):
                    update[b] = False
                    bad_steps[b] += 1
                    lr_mult[b] *= 0.5
                    log.warning(
                        "non-finite loss for problem %d at stage %d step %d; rolling back, lr×%.3g",
                        b,
                        sidx,
                        step,
                        lr_mult[b],
                    )
                    with torch.no_grad():
                        for f, gd in zip(flats, good):
                            f[b].copy_(gd[b])
                    opt.reset(b)
                    if bad_steps[b] > self.opts["max_bad_steps"]:
                        self.failed[b] = (
                            "persistent non-finite loss; lower the learning rate or check the "
                            "operator / losses for numerical issues"
                        )
                        active[b] = False
                        stop_reason[b] = "failed"
            if not any(update):
                continue
            if oc.grad_clip:
                self._clip(flats, oc.grad_clip)
            opt.step([lr_mult[b] * lr_base for b in range(B)], update)
            if nan_guard and step % self.opts["checkpoint_every"] == 0:
                self._masked_copy(good, flats, update)
            if ema is not None:
                self._ema_update(ema, flats, oc.ema, update)

            # ---- bookkeeping and per-problem stopping ---------------------------------------
            annealed = progress >= 1.0
            for b in range(B):
                if not update[b]:
                    continue
                bad_steps[b] = 0
                steps_run[b] += 1
                last_progress[b] = progress
                data_loss = losses[b].data_loss(comps[b])
                h = self.history[b]
                h["step"].append(step)
                h["global_step"].append(self.global_step[b])
                h["stage"].append(sidx)
                h["lr"].append(lr_mult[b] * lr_base)
                h["progress"].append(progress)
                h["total"].append(tots[b])
                h["data_loss"].append(data_loss)
                for k, v in comps[b].items():
                    h[f"loss/{k}" if k in RESERVED_HISTORY_KEYS else k].append(v)
                last_comps[b] = comps[b]
                self.global_step[b] += 1
                if cur.early_stop_patience and annealed:
                    if data_loss < best_data[b] - cur.early_stop_min_delta:
                        best_data[b], since_best[b] = data_loss, 0
                    else:
                        since_best[b] += 1
                        if since_best[b] >= cur.early_stop_patience:
                            stop_reason[b], active[b] = "early_stop", False
                            continue
                if cur.discrepancy_tau and noise[b] and annealed:
                    with torch.no_grad():
                        rmse = float(torch.sqrt(obs[b].masked_mean((pred[b] - obs[b].data) ** 2)))
                    if rmse <= cur.discrepancy_tau * noise[b]:
                        stop_reason[b], active[b] = "discrepancy", False
            if cur.time_budget_s and time.perf_counter() - self.t0 > cur.time_budget_s:
                for b in range(B):
                    if active[b]:
                        stop_reason[b], active[b] = "time_budget", False
                self.stop_all = True
                break

        with torch.no_grad():
            if ema is not None:
                for f, e in zip(flats, ema):
                    f.copy_(e)
            synchronize(dev)
            self._write_back(all_params())
        seconds = time.perf_counter() - t_stage
        for b in range(B):
            if stop_reason[b] == "completed":
                final_progress = stage.progress_at(stage.steps)
            else:
                final_progress = last_progress[b]
            self.stage_results[b].append(
                {
                    "name": stage.name,
                    "shape": tuple(dom.shape),
                    "steps": steps_run[b],
                    "seconds": seconds,
                    "final": last_comps[b],
                    "stop": stop_reason[b],
                    "lr_mult": lr_mult[b],
                    "final_progress": float(final_progress),
                    "loss_mode": mode,
                }
            )

    # -- per-problem tensor helpers -------------------------------------------------------------
    def _clip(self, flats: list[torch.Tensor], max_norm: float) -> None:
        """``clip_grad_norm_`` with one norm per problem."""
        grads = [f.grad for f in flats if f.grad is not None]
        if not grads:
            return
        with torch.no_grad():
            norms = torch.stack([torch.linalg.vector_norm(g, 2.0, dim=1) for g in grads], dim=1)
            total = torch.linalg.vector_norm(norms, 2.0, dim=1, keepdim=True)
            coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
            for g in grads:
                g.mul_(coef)

    @staticmethod
    def _masked_copy(dst: list[torch.Tensor], src: list[torch.Tensor], mask: list[bool]) -> None:
        with torch.no_grad():
            for d, s in zip(dst, src):
                if all(mask):
                    d.copy_(s)
                else:
                    keep = torch.tensor(mask, device=s.device).view(-1, 1)
                    d.copy_(torch.where(keep, s, d))

    @staticmethod
    def _ema_update(ema: list[torch.Tensor], flats, decay: float, mask: list[bool]) -> None:
        with torch.no_grad():
            for e, f in zip(ema, flats):
                new = e.mul(decay).add_(f.detach(), alpha=1.0 - decay)
                if all(mask):
                    e.copy_(new)
                else:
                    keep = torch.tensor(mask, device=f.device).view(-1, 1)
                    e.copy_(torch.where(keep, new, e))

    def final_data_loss(self, b: int) -> float:
        h = self.history[b].get("data_loss")
        return float(h[-1]) if h else math.inf


def batch_invert(
    problems: Sequence[Any],
    curriculum: Curriculum | None = None,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float32,
    seeds: int | Sequence[int | None] | None = 0,
    nan_guard: bool = True,
    checkpoint_every: int = 25,
    max_bad_steps: int = 10,
    loss_mode: str = "auto",
    check: bool = True,
    raise_on_error: bool = True,
) -> list[Result]:
    """Solve ``N`` independent inverse problems in one batched optimization loop.

    Args:
        problems: :class:`~nefi.problem.InverseProblem` s sharing the domain, the field
            architecture, a batchable operator (:attr:`Operator.batchable
            <nefi.operators.Operator.batchable>`, no trainable operator parameters) and — for the
            fast path — the loss set; typically one instance, several measurements and seeds.
        curriculum: shared curriculum (default: ``problems[0].curriculum`` or the multiscale
            default). Optimizer ``adam`` / ``adamw``.
        device / dtype: as for :class:`~nefi.solve.Solver`.
        seeds: per-problem seeds (or one for all). As in the sequential solver they seed the
            re-initialization of restarts (``seed + r``); the initial parameters are the ones the
            problems were built with.
        nan_guard / checkpoint_every / max_bad_steps: per-problem NaN guard (as in ``Solver``).
        loss_mode: ``"auto"`` (``vmap`` over a shared loss set when all loss sets are identical
            and vmap-compatible, verified at every stage start; else a per-problem loop),
            ``"vmap"`` or ``"loop"``.
        check: verify at every stage start that the batched operator reproduces every
            problem's own operator (catches problems that do not share the physics).
        raise_on_error: raise :class:`~nefi.errors.SolverError` if a problem failed (persistent
            non-finite loss); otherwise its result carries ``extra["error"]``.

    Returns:
        One :class:`~nefi.solve.Result` per problem (same content as ``Solver(problem).run()``;
        ``timing["total_s"]`` is the wall-clock of the whole batch and ``extra`` records
        ``batched``, ``batch_size``, ``batch_index`` and the loss evaluation mode per stage).

    Raises:
        ConfigError: a problem cannot be batched (the message says why) or the problems are not
            compatible with each other.
    """
    if loss_mode not in LOSS_MODES:
        raise ConfigError(f"loss_mode must be one of {LOSS_MODES}, got {loss_mode!r}")
    problems = list(problems)
    cur = (
        curriculum
        or getattr(problems[0], "curriculum", None)
        or Curriculum.multiscale(problems[0].domain.shape)
    )
    _check_problems(problems, cur)
    n = len(problems)
    seed_list = _seeds(seeds, n)
    dev = resolve_device(device)
    t0 = time.perf_counter()
    seed_everything(seed_list[0])
    for p in problems:
        p.to(dev, dtype)
    opts = {
        "nan_guard": nan_guard,
        "checkpoint_every": int(checkpoint_every),
        "max_bad_steps": int(max_bad_steps),
        "loss_mode": loss_mode,
        "check": bool(check),
    }
    best: list[tuple | None] = [None] * n
    best_data = [math.inf] * n
    run = None
    for r in range(cur.restarts):
        if r > 0:
            for b, p in enumerate(problems):
                seed_everything(None if seed_list[b] is None else seed_list[b] + r)
                p.field.reset_parameters()
                p.field.to(dev, dtype)
        run = _BatchRun(problems, cur, dev, dtype, opts, t0)
        for i, stage in enumerate(cur.stages):
            run.run_stage(i, stage)
            if run.stop_all:
                break
        for b, p in enumerate(problems):
            data = run.final_data_loss(b)
            log.info(
                "batch problem %d restart %d/%d: final data loss %.4e", b, r + 1, cur.restarts, data
            )
            if data < best_data[b] or best[b] is None:
                best_data[b] = data
                state = None
                if cur.restarts > 1:
                    state = (
                        copy.deepcopy(p.field.state_dict()),
                        copy.deepcopy(p.operator.state_dict()),
                    )
                best[b] = (state, dict(run.history[b]), list(run.stage_results[b]))
    assert run is not None
    results = []
    for b, p in enumerate(problems):
        state, hist, stages = best[b]
        if cur.restarts > 1 and state is not None:
            p.field.load_state_dict(state[0])
            p.operator.load_state_dict(state[1])
        s = Solver(p, cur, device=dev, dtype=dtype, seed=seed_list[b])
        s.history = defaultdict(list, hist)
        s.stage_results = stages
        s._t0 = t0
        res = s._finalize()
        res.extra.update(
            batched=True,
            batch_size=n,
            batch_index=b,
            loss_modes=list(run.loss_modes),
        )
        res.timing["batch_total_s"] = res.timing["total_s"]
        if b in run.failed:
            res.extra["error"] = run.failed[b]
        results.append(res)
    if run.failed and raise_on_error:
        idx = sorted(run.failed)
        raise SolverError(f"batch_invert: problem(s) {idx} failed: {run.failed[idx[0]]}")
    return results
