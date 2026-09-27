"""ADMM with an ℓ1 + box proximal split — the *ADMM* baseline (NeTMY App. E.2; Boyd et al. 2011).

The problem ``min_x D(x) + λ1 ‖x‖_1 + ι_[lo,hi](x)`` is split as ``x = z``::

    x^{k+1} = argmin_x D(x) + (μ/2) mean((x - z^k + u^k)²)       # n_inner Adam steps (warm start)
    z^{k+1} = clip(soft(x^{k+1} + u^k, λ1/μ), lo, hi)             # proximal step (closed form)
    u^{k+1} = u^k + x^{k+1} - z^{k+1}                              # scaled dual ascent

stopping when ``‖x - z‖_∞ < tol`` or after ``max_outer`` cycles. ``D`` is the problem's own
:class:`~nefi.losses.LossSet` restricted to its data terms, and — following the library's
convention that reductions are means — both the ℓ1 term and the penalty are means over pixels, so
``λ1`` and ``μ`` are resolution-independent (the prox threshold is ``λ1/μ`` per pixel).
NeTMY defaults: ``μ = 1e-3``, ``λ1 = 1e-2``, Adam lr ``5e-3``, 30 inner × 200 outer, ``tol = 1e-3``.

The primal variable is the field value of the primary head; with a :class:`~nefi.fields.GridField`
and an ``Identity`` head this is textbook ADMM on the free density (``G_θ = I``: the raw
field-space gradient is executed verbatim, NeTMY §4.4). Any other field also works (the x-step
then optimizes that parameterization). The returned field is ``z`` (feasible: sparse and inside the
box); set ``return_variable="x"`` for the data-fit iterate.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..config import from_dict
from ..errors import ConfigError, SolverError
from ..losses.base import Context
from ..problem import InverseProblem
from ..registry import register
from ..solve.callbacks import Callback, StepState
from ..solve.curriculum import Curriculum, Stage
from ..solve.result import Result
from ..utils.device import resolve_device, synchronize
from ..utils.seed import seed_everything
from ._common import assemble_result, component_key, resolve_shape

log = logging.getLogger("nefi")


@dataclass
class ADMMConfig:
    """Settings of :class:`ADMMSolver` (defaults: NeTMY App. E.2).

    Attributes:
        mu: augmented-Lagrangian penalty ``μ``.
        l1: ℓ1 weight ``λ1`` of the proximal step (0 disables shrinkage).
        lo / hi: box constraint (``None`` = unbounded side).
        lr: Adam learning rate of the inner data-fidelity step.
        n_inner: Adam steps per outer cycle.
        max_outer: maximum number of outer cycles.
        tol: stop when ``‖x - z‖_∞ < tol`` (NeTMY stopping rule).
        dual_tol: if set, additionally require the dual residual ``μ ‖z^{k+1} - z^k‖_∞ < dual_tol``
            (Boyd et al. §3.3); recommended for large ``μ``, where ``x`` is pinned to ``z`` and the
            primal residual alone stops too early.
        min_outer: never stop before this many cycles.
        shape: working resolution (``None``: curriculum final shape, else domain shape).
        field: name of the split field (``None``: primary).
        keep_terms: non-data loss terms kept in the x-step (default: data terms only).
        weight_decay / grad_clip: Adam weight decay and gradient clipping of the x-step.
        reset_optimizer: re-create the Adam state at every outer cycle.
        adaptive_mu: residual balancing (Boyd et al. §3.4.1): ``μ ← μ·tau`` if
            ``r > balance·s`` and ``μ ← μ/tau`` if ``s > balance·r`` (scaled dual rescaled).
        balance / tau: residual-balancing constants.
        return_variable: ``"z"`` (feasible split variable) or ``"x"``.
        time_budget_s: wall-clock budget.
        budget_from_curriculum: when a curriculum is passed to the solver explicitly, take the
            iteration budget from it (``max_outer = ceil(total_steps / n_inner)``), so benchmark
            budget scaling (``Curriculum.scaled``) applies to ADMM too.
    """

    mu: float = 1e-3
    l1: float = 1e-2
    lo: float | None = 0.0
    hi: float | None = None
    lr: float = 5e-3
    n_inner: int = 30
    max_outer: int = 200
    tol: float = 1e-3
    dual_tol: float | None = None
    min_outer: int = 1
    shape: tuple[int, ...] | None = None
    field: str | None = None
    keep_terms: tuple[str, ...] = ()
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    reset_optimizer: bool = False
    adaptive_mu: bool = False
    balance: float = 10.0
    tau: float = 2.0
    return_variable: str = "z"
    time_budget_s: float | None = None
    budget_from_curriculum: bool = True

    def __post_init__(self) -> None:
        if self.mu <= 0:
            raise ConfigError("ADMM penalty mu must be > 0")
        if self.n_inner < 1 or self.max_outer < 1:
            raise ConfigError("n_inner and max_outer must be >= 1")
        if self.return_variable not in ("x", "z"):
            raise ConfigError("return_variable must be 'x' or 'z'")
        if self.lo is not None and self.hi is not None and not self.hi > self.lo:
            raise ConfigError(f"box needs hi > lo, got {(self.lo, self.hi)}")
        if self.shape is not None:
            self.shape = tuple(int(s) for s in self.shape)
        self.keep_terms = tuple(self.keep_terms)


def prox_l1_box(
    v: torch.Tensor, threshold: float, lo: float | None, hi: float | None
) -> torch.Tensor:
    """``argmin_z threshold·|z| + ½(z - v)² + ι_[lo,hi](z)`` (soft threshold, then clip)."""
    z = torch.sign(v) * torch.clamp(v.abs() - threshold, min=0.0) if threshold > 0 else v
    if lo is not None or hi is not None:
        z = torch.clamp(z, min=lo, max=hi)
    return z


def _as_config(config) -> tuple[ADMMConfig, Curriculum | None]:
    if config is None:
        return ADMMConfig(), None
    if isinstance(config, ADMMConfig):
        return config, None
    if isinstance(config, Curriculum):
        return ADMMConfig(), config
    if isinstance(config, Mapping):
        return from_dict(ADMMConfig, dict(config)), None
    raise ConfigError(f"cannot build an ADMMConfig from {type(config).__name__}")


@register("baseline", "admm")
class ADMMSolver:
    """Variable-splitting ADMM solver with the same call shape as :class:`~nefi.solve.Solver`.

    ``ADMMSolver(problem, config).run() -> Result``. ``config`` may be an :class:`ADMMConfig`, a
    plain dict, a :class:`~nefi.solve.Curriculum` (its final resolution and step budget are used)
    or ``None`` (then ``problem.meta["solver_config"]`` or the defaults).

    The result's ``history`` mirrors :class:`~nefi.solve.Solver`: per inner step ``step``,
    ``global_step``, ``stage``, ``lr``, ``progress``, ``total`` (x-step objective
    ``D + penalty``), ``data_loss`` (weighted data fidelity ``D``), ``penalty`` and the individual
    data terms under their names (``loss/<name>`` on a clash with a reserved key); and per outer
    cycle ``outer``, ``outer_step``, ``objective`` (``D(z) + λ1 mean|z|``), ``data_z``
    (``D(z)``), ``primal_residual`` (``‖x - z‖_∞``), ``dual_residual``
    (``μ ‖z^{k+1} - z^k‖_∞``) and ``mu``.

    Args:
        problem: the inverse problem (field, operator, losses, measurement).
        config: see above.
        curriculum: optional curriculum whose final resolution is honoured and whose total step
            budget sets ``max_outer`` (see ``ADMMConfig.budget_from_curriculum``).
        device / dtype / callbacks / seed: as for :class:`~nefi.solve.Solver`; callbacks receive
            ``on_step`` for every inner Adam step.
        **_: other :class:`~nefi.solve.Solver` keywords are accepted and ignored.
    """

    def __init__(
        self,
        problem: InverseProblem,
        config: ADMMConfig | Mapping | Curriculum | None = None,
        *,
        curriculum: Curriculum | None = None,
        device: str | torch.device = "auto",
        dtype: torch.dtype = torch.float32,
        callbacks: Sequence[Callback] = (),
        seed: int | None = 0,
        **_,
    ) -> None:
        if config is None:
            config = problem.meta.get("solver_config") if hasattr(problem, "meta") else None
        self.config, cur = _as_config(config)
        self.problem = problem
        self.device = resolve_device(device)
        self.dtype = dtype
        self.callbacks = list(callbacks)
        self.seed = seed
        explicit = curriculum or cur
        self.shape = resolve_shape(problem, self.config.shape, explicit)
        if explicit is not None and self.config.budget_from_curriculum:
            n_outer = max(1, math.ceil(explicit.total_steps / self.config.n_inner))
            self.config = dataclasses.replace(self.config, max_outer=n_outer)
        c = self.config
        self.curriculum = Curriculum(
            [
                Stage(
                    "admm",
                    self.shape,
                    c.n_inner * c.max_outer,
                    c.lr,
                    lr_schedule="constant",
                    anneal=False,
                )
            ]
        )
        self.history: dict[str, list[float]] = defaultdict(list)
        self.stage_results: list[dict] = []
        self.global_step = 0

    # ------------------------------------------------------------------------------------
    def _params(self) -> list[torch.nn.Parameter]:
        params = [q for q in self.problem.field.parameters() if q.requires_grad]
        params += [q for q in self.problem.operator.parameters() if q.requires_grad]
        if not params:
            raise SolverError("ADMM x-step has no trainable parameters")
        return params

    def run(self) -> Result:
        """Run ADMM to convergence / budget and return a :class:`~nefi.solve.Result`."""
        c = self.config
        p = self.problem
        t0 = time.perf_counter()
        seed_everything(self.seed)
        p.to(self.device, self.dtype)
        stage = self.curriculum.stages[0]
        dom = p.domain.at(self.shape)
        coords = dom.coords(device=self.device, dtype=self.dtype)
        op = p.operator.at_resolution(dom.shape).to(self.device)
        obs = p.measurement_at(dom.shape).to(self.device, self.dtype)
        p.field.on_stage_start(stage, dom)
        p.field.to(self.device, self.dtype)
        name = c.field or p.field.primary
        if name not in p.field.names:
            raise ConfigError(f"ADMM split field {name!r} not among {p.field.names}")
        unknown = [k for k in c.keep_terms if k not in p.losses.names]
        if unknown:
            raise ConfigError(f"keep_terms {unknown} not among losses {p.losses.names}")
        keep = set(p.losses.data_terms()) | set(c.keep_terms)
        losses = p.losses.with_weights({k: 0.0 for k in p.losses.names if k not in keep})
        params = self._params()

        def make_opt():
            return torch.optim.Adam(params, lr=c.lr, weight_decay=c.weight_decay)

        opt = make_opt()
        for cb in self.callbacks:
            cb.on_run_start(self)
            cb.on_stage_start(self, 0, stage)

        with torch.no_grad():
            x0 = p.field(coords, 1.0)[name]
        z = prox_l1_box(x0, 0.0, c.lo, c.hi).detach()
        u = torch.zeros_like(z)
        mu = float(c.mu)
        h = self.history
        stop, n_outer = "max_outer", 0
        for k in range(c.max_outer):
            if c.reset_optimizer and k > 0:
                opt = make_opt()
            for j in range(c.n_inner):
                opt.zero_grad(set_to_none=True)
                fields = p.field(coords, 1.0)
                x = fields[name]
                pred = op(fields)
                ctx = Context(fields, pred, obs, dom, op, p.field, stage, self.global_step, 1.0)
                data_total, comps = losses(ctx)
                pen = 0.5 * mu * ((x - z + u) ** 2).mean()
                total = data_total + pen
                if not torch.isfinite(total):
                    raise SolverError(
                        f"non-finite ADMM x-step objective at outer {k}, inner {j}; lower lr/mu"
                    )
                total.backward()
                if c.grad_clip:
                    torch.nn.utils.clip_grad_norm_(params, c.grad_clip)
                opt.step()
                data_val = losses.data_loss(comps)
                h["step"].append(self.global_step)
                h["global_step"].append(self.global_step)
                h["stage"].append(0)
                h["lr"].append(c.lr)
                h["progress"].append(1.0)
                h["total"].append(float(total.detach()))
                h["data_loss"].append(data_val)
                h["penalty"].append(float(pen.detach()))
                for key, v in comps.items():
                    h[component_key(key)].append(v)
                state = StepState(
                    0,
                    stage,
                    self.global_step,
                    self.global_step,
                    c.lr,
                    1.0,
                    float(total.detach()),
                    comps,
                    data_val,
                    time.perf_counter() - t0,
                    fields,
                    pred,
                    {"outer": k, "inner": j},
                )
                for cb in self.callbacks:
                    cb.on_step(self, state)
                self.global_step += 1

            # ---- z- and dual updates (closed form) ----
            with torch.no_grad():
                fields = p.field(coords, 1.0)
                x = fields[name]
                z_old = z
                z = prox_l1_box(x + u, c.l1 / mu, c.lo, c.hi)
                u = u + x - z
                r = float((x - z).abs().max())
                s = mu * float((z - z_old).abs().max())
                fz = {**{kk: vv.detach() for kk, vv in fields.items()}, name: z}
                pz = op(fz)
                ctx = Context(fz, pz, obs, dom, op, p.field, stage, self.global_step, 1.0)
                _, comps_z = losses(ctx)
                data_z = losses.data_loss(comps_z)
                objective = data_z + c.l1 * float(z.abs().mean())
            n_outer = k + 1
            h["outer"].append(k)
            h["outer_step"].append(self.global_step)
            h["objective"].append(objective)
            h["data_z"].append(data_z)
            h["primal_residual"].append(r)
            h["dual_residual"].append(s)
            h["mu"].append(mu)
            if not math.isfinite(objective):
                raise SolverError(f"non-finite ADMM objective at outer {k}")
            if c.adaptive_mu:
                if r > c.balance * s:
                    mu *= c.tau
                    u = u / c.tau
                elif s > c.balance * r:
                    mu /= c.tau
                    u = u * c.tau
            dual_ok = c.dual_tol is None or s < c.dual_tol
            if n_outer >= c.min_outer and r < c.tol and dual_ok:
                stop = "converged"
                break
            if c.time_budget_s and time.perf_counter() - t0 > c.time_budget_s:
                stop = "time_budget"
                break

        synchronize(self.device)
        with torch.no_grad():
            fields = {kk: vv.detach() for kk, vv in p.field(coords, 1.0).items()}
            if c.return_variable == "z":
                fields[name] = z
        info = {
            "name": "admm",
            "shape": tuple(dom.shape),
            "steps": self.global_step,
            "outer": n_outer,
            "seconds": time.perf_counter() - t0,
            "final": {"objective": h["objective"][-1], "primal_residual": h["primal_residual"][-1]},
            "stop": stop,
        }
        self.stage_results = [info]
        for cb in self.callbacks:
            cb.on_stage_end(self, 0, stage, info)
        result = assemble_result(
            p,
            fields,
            tuple(dom.shape),
            h,
            self.stage_results,
            t0,
            dataclasses.asdict(c),
            {
                "method": "admm",
                "device": str(self.device),
                "dtype": str(self.dtype),
                "seed": self.seed,
                "n_parameters": p.field.n_parameters(),
                "n_outer": n_outer,
                "converged": stop == "converged",
                "z": z.detach().cpu(),
                "u": u.detach().cpu(),
            },
        )
        for cb in self.callbacks:
            cb.on_run_end(self, result)
        return result


__all__ = ["ADMMConfig", "ADMMSolver", "prox_l1_box"]
