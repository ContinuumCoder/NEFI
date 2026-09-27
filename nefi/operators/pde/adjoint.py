"""Implicit-Euler time stepping and its discrete adjoint (NeFTY §4.3, App. D.3).

State equation (NeFTY Eq. 8 / 25), ambient-shifted so it is affine-free::

    F^n = A(α) T^n − T^{n−1} = 0,   A(α) = I − Δt L(α),   n = 1..N_t.

Given a data term ``J = Σ_n ℓ^n(T^n)`` on the observed (surface) frames, the Lagrange multipliers
``μ^n`` of the state residuals satisfy the backward-in-time recurrence (NeFTY Eq. 10 / 26)::

    A(α)ᵀ μ^n = μ^{n+1} + ∂ℓ^n/∂T^n,    μ^{N_t+1} = 0,

solved with the *same* inner solver because ``A`` is symmetric (App. D.2), and the parameter
gradient is assembled as (NeFTY Eq. 11 / 27)::

    dJ/dα = Δt Σ_n (μ^n)ᵀ ∂(L(α) T^n)/∂α,        dJ/dT^0 = μ^1.

Because ``μᵀ L(α) T = −Σ_faces a_f(α) (Δ_f μ)(Δ_f T) − Σ β μ T`` is bilinear, the default
``assembly="fused"`` accumulates the per-face products ``S_f = Σ_n (Δ_f μ^n)(Δ_f T^n)`` during the
backward sweep and differentiates the face conductances ``a_f(α)`` (harmonic mean,
``∂ᾱ/∂α_i = 2α_{i+1}²/(α_i+α_{i+1})²``) **once**; ``assembly="per_step"`` instead runs one small
autograd graph (a VJP of ``α ↦ L(α)T^n``) per step — the literal Eq. (27), kept as a reference.

Memory: the forward stores the detached trajectory ``T^1..T^{N_t}`` (``O(N_g N_t)``); the
backward keeps ``O(N_g)`` work arrays. Unlike back-propagation through the unrolled solver
(``grad_mode="autograd"``, ``O(K N_g N_t)``), nothing from the ``K`` inner iterations is kept.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable

from ...errors import ConfigError
from ...utils.compat import cudagraph_mark_step_begin
from .linear_solvers import conjugate_gradient
from .stencil import (
    STENCIL_BACKENDS,
    BoundarySpec,
    DiffusionStencil,
    ImplicitSystem,
    _chebyshev_sweeps_roll,
    _jacobi_sweeps_roll,
    _use_flat,
    face_conductances,
)

SOLVERS = ("jacobi", "chebyshev", "cg")
GRAD_MODES = ("adjoint", "autograd", "checkpoint")
ASSEMBLY_MODES = ("fused", "per_step")

__all__ = [
    "ASSEMBLY_MODES",
    "GRAD_MODES",
    "SOLVERS",
    "HeatSolveConfig",
    "ImplicitEulerAdjoint",
    "implicit_euler_frames",
    "rollout",
    "solve_step",
    "surface",
]


@dataclass(frozen=True)
class HeatSolveConfig:
    """Static description of an implicit-Euler heat solve (hashable, no tensors).

    Args:
        spacing: physical cell size per grid axis (last axis = through-thickness z).
        dt: time step ``Δt`` (NeFTY Tab. 5: 0.05, aligned with the camera frame rate).
        n_steps: number of implicit-Euler steps ``N_t`` (Tab. 5: 100).
        obs_steps: sorted step indices whose front surface is observed; ``0`` is the initial
            state, ``n ≥ 1`` the state after ``n`` steps.
        bc: boundary conditions.
        face_mode: ``"harmonic"`` (Prop. 1) or ``"arithmetic"``.
        solver: inner linear solver, ``"jacobi"`` (Eq. 24), ``"chebyshev"`` (Chebyshev-accelerated
            Jacobi, :meth:`~nefi.operators.pde.stencil.ImplicitSystem.chebyshev_sweeps`: the same
            accuracy with ≈ 2.5× fewer iterations at the Tab. 5 time step) or ``"cg"``.
        inner_iters: Jacobi sweeps / Chebyshev iterations ``K`` per step (Tab. 5: 50 Jacobi).
        cg_tol / cg_max_iter / cg_precond: CG relative tolerance, iteration cap and Jacobi
            preconditioning.
        assembly: adjoint gradient assembly, ``"fused"`` (default) or ``"per_step"``.
        compile: run the ``K`` Jacobi sweeps of every gradient-free solve (the adjoint's forward
            and backward sweeps) through ``torch.compile`` (opt-in; one-time compile cost of tens of
            seconds, then several-fold fewer kernel launches — the setting for CUDA servers).
        compile_mode: ``mode`` passed to ``torch.compile`` (e.g. ``"reduce-overhead"`` for CUDA
            graphs, ``"max-autotune-no-cudagraphs"``); ``None`` = the default mode.
        stencil_backend: stencil implementation, ``"auto"`` (flat padded layout for the
            gradient-free adjoint solves, ``torch.roll`` reference under autograd), ``"flat"`` or
            ``"roll"`` (see :mod:`~nefi.operators.pde.stencil`).
    """

    spacing: tuple[float, ...]
    dt: float
    n_steps: int
    obs_steps: tuple[int, ...]
    bc: BoundarySpec
    face_mode: str = "harmonic"
    solver: str = "jacobi"
    inner_iters: int = 50
    cg_tol: float = 1e-6
    cg_max_iter: int = 200
    cg_precond: bool = True
    assembly: str = "fused"
    compile: bool = False
    compile_mode: str | None = None
    stencil_backend: str = "auto"

    def __post_init__(self) -> None:
        if self.solver not in SOLVERS:
            raise ConfigError(f"unknown inner solver {self.solver!r}; use one of {SOLVERS}")
        if self.stencil_backend not in STENCIL_BACKENDS:
            raise ConfigError(
                f"unknown stencil backend {self.stencil_backend!r}; use one of {STENCIL_BACKENDS}"
            )
        if self.assembly not in ASSEMBLY_MODES:
            raise ConfigError(f"unknown assembly {self.assembly!r}; use one of {ASSEMBLY_MODES}")
        if self.dt <= 0 or self.n_steps < 1:
            raise ConfigError("need dt > 0 and n_steps >= 1")
        obs = tuple(int(n) for n in self.obs_steps)
        if not obs or any(n < 0 or n > self.n_steps for n in obs):
            raise ConfigError(f"obs_steps must lie in [0, {self.n_steps}], got {obs}")
        if list(obs) != sorted(set(obs)):
            raise ConfigError("obs_steps must be sorted and unique")
        object.__setattr__(self, "obs_steps", obs)

    @property
    def ndim(self) -> int:
        return len(self.spacing)

    def stencil(self, alpha: torch.Tensor) -> DiffusionStencil:
        return DiffusionStencil(
            alpha, self.spacing, self.bc, self.face_mode, backend=self.stencil_backend
        )


def surface(T: torch.Tensor) -> torch.Tensor:
    """Observed front surface: the first slice along the last (through-thickness) axis."""
    return T[..., 0]


#: ``iters`` fused Jacobi sweeps ``x ← D⁻¹b + Σ_j W_j ⊙ shift_j(x)`` (NeFTY Eq. 24), ``torch.roll``
#: reference — the function ``compile=True`` hands to ``torch.compile``.
_jacobi_sweeps = _jacobi_sweeps_roll

_COMPILED: dict[tuple, Callable | None] = {}
log = logging.getLogger("nefi")


def _uses_cudagraphs(mode: str | None) -> bool:
    """Whether a ``torch.compile`` mode captures CUDA graphs (outputs are replay-owned)."""
    return bool(mode) and mode in ("reduce-overhead", "max-autotune")


def _compiled_sweeps(mode: str | None, solver: str = "jacobi") -> Callable | None:
    """Lazily ``torch.compile`` the reference sweeps (``None`` if compilation is unavailable)."""
    key = (mode, solver)
    if key not in _COMPILED:
        fn = _jacobi_sweeps_roll if solver == "jacobi" else _chebyshev_sweeps_roll
        try:
            _COMPILED[key] = torch.compile(fn, mode=mode, dynamic=False)
        except Exception as e:  # pragma: no cover - platform dependent
            log.warning("torch.compile unavailable (%s); using eager %s sweeps", e, solver)
            _COMPILED[key] = None
    return _COMPILED[key]


def solve_step(
    system: ImplicitSystem,
    b: torch.Tensor,
    x0: torch.Tensor | None,
    cfg: HeatSolveConfig,
) -> torch.Tensor:
    """Solve ``A x = b`` with the configured inner solver, warm-started from ``x0``.

    The Jacobi path uses the fused sweep ``x ← D⁻¹b + Σ_j W_j ⊙ shift_j(x)`` (identical to
    NeFTY Eq. 24, ``x ← D⁻¹(b − R x)``, with one fewer tensor op per sweep) on the flat padded
    layout (:meth:`ImplicitSystem.jacobi_sweeps`); ``"chebyshev"`` accelerates the same sweeps
    (:meth:`ImplicitSystem.chebyshev_sweeps`). With ``cfg.compile`` the gradient-free sweeps run
    through ``torch.compile`` (the ``torch.roll`` reference, which inductor fuses per sweep).
    """
    x = b if x0 is None else x0
    if cfg.solver in ("jacobi", "chebyshev"):
        if cfg.compile and not torch.is_grad_enabled():
            fn = _compiled_sweeps(cfg.compile_mode, cfg.solver)
            if fn is not None:
                weights = system.jacobi_sweep_weights()
                last = system.chebyshev_omegas(cfg.inner_iters) if cfg.solver != "jacobi" else None
                graphs = _uses_cudagraphs(cfg.compile_mode) and b.device.type == "cuda"
                try:
                    if graphs:  # CUDA graphs: the output buffer is reused by the next replay
                        cudagraph_mark_step_begin()
                    if last is None:
                        out = fn(b, x, system.inv_diag, weights, cfg.inner_iters)
                    elif cfg.inner_iters > 0:
                        out = fn(b, x, system.inv_diag, weights, last)
                    else:
                        return x
                    return out.clone() if graphs else out
                except Exception as e:  # pragma: no cover - platform dependent
                    log.warning(
                        "compiled %s sweeps failed (%s); falling back to eager", cfg.solver, e
                    )
                    _COMPILED[(cfg.compile_mode, cfg.solver)] = None
        if cfg.solver == "jacobi":
            return system.jacobi_sweeps(b, x, cfg.inner_iters)
        return system.chebyshev_sweeps(b, x, cfg.inner_iters)
    x, _ = conjugate_gradient(
        system.apply,
        b,
        x,
        tol=cfg.cg_tol,
        max_iter=cfg.cg_max_iter,
        precond=system.diag if cfg.cg_precond else None,
    )
    return x


def rollout(
    alpha: torch.Tensor,
    T0: torch.Tensor,
    cfg: HeatSolveConfig,
    *,
    keep_states: bool = False,
    residuals: bool = False,
    step_fn: Callable | None = None,
    n_steps: int | None = None,
    system: ImplicitSystem | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, list[float] | None]:
    """Run the implicit-Euler recurrence (NeFTY Eq. 8) and collect the observed surface frames.

    Runs under the ambient autograd mode, so with gradients enabled it is the unrolled reference
    (``grad_mode="autograd"``). ``step_fn(system, b, x0) -> x`` overrides the per-step solve (used
    by the checkpointed mode).

    Args:
        alpha / T0 / cfg: diffusivity, initial state and solve configuration.
        keep_states: also return the states ``T^1..T^{n_steps}``.
        residuals: also return the relative residual ``‖A T^n − T^{n−1}‖ / ‖T^{n−1}‖`` per step.
        step_fn: optional per-step solve override.
        n_steps: number of steps to run (default: up to the last observed step; later states
            cannot influence the observations).
        system: a prebuilt ``cfg.stencil(alpha).system(cfg.dt)`` (the adjoint reuses it for the
            backward sweep).

    Returns:
        ``(frames, states, residuals)``: frames ``(n_obs, *surface)``; states ``(n_steps, *grid)``
        or ``None``; residual list or ``None``.
    """
    if system is None:
        system = cfg.stencil(alpha).system(cfg.dt)
    obs = set(cfg.obs_steps)
    frames = [surface(T0)] if 0 in obs else []
    states = [] if keep_states else None
    res: list[float] | None = [] if residuals else None
    T = T0
    last = max(cfg.obs_steps) if n_steps is None else max(int(n_steps), max(cfg.obs_steps))
    for n in range(1, last + 1):
        b = T
        T = solve_step(system, b, b, cfg) if step_fn is None else step_fn(system, b, b)
        if res is not None:
            with torch.no_grad():
                r = torch.linalg.vector_norm(system.apply(T) - b)
                res.append(float(r / torch.linalg.vector_norm(b).clamp_min(1e-30)))
        if states is not None:
            states.append(T)
        if n in obs:
            frames.append(surface(T))
    return (
        torch.stack(frames),
        torch.stack(states) if states is not None else None,
        res,
    )


class ImplicitEulerAdjoint(torch.autograd.Function):
    """Discrete adjoint of the implicit-Euler heat solve (NeFTY Eq. 10–11, App. D.3 Eq. 25–27).

    ``ImplicitEulerAdjoint.apply(alpha, T0, cfg) -> frames`` with ``frames`` the stacked front
    surfaces at ``cfg.obs_steps``. Forward stores only the detached trajectory; backward runs
    ``A μ^n = μ^{n+1} + ∂ℓ^n/∂T^n`` backwards in time with the same inner solver and returns
    ``dJ/dα = Δt Σ_n (μ^n)ᵀ ∂(L(α)T^n)/∂α`` and ``dJ/dT^0 = μ^1``. Not twice differentiable.
    """

    @staticmethod
    def forward(ctx, alpha: torch.Tensor, T0: torch.Tensor, cfg: HeatSolveConfig) -> torch.Tensor:
        with torch.no_grad():
            alpha_d, T0_d = alpha.detach(), T0.detach()
            need_states = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
            system = cfg.stencil(alpha_d).system(cfg.dt)
            frames, states, _ = rollout(alpha_d, T0_d, cfg, keep_states=need_states, system=system)
        ctx.cfg = cfg
        # the backward solves with the same matrix (Aᵀ = A): keep the O(N_g) system tensors
        # instead of rebuilding the face conductances from α
        ctx.system = system if need_states else None
        if states is None:  # no input needs a gradient (only reachable via direct .apply)
            states = torch.empty(0, *alpha_d.shape, dtype=alpha_d.dtype, device=alpha_d.device)
        # the detached trajectory T^1..T^{n_last}: O(N_g N_t). Saved as a regular saved tensor so
        # saved-tensor hooks (e.g. torch.autograd.graph.save_on_cpu) can offload it.
        ctx.save_for_backward(alpha_d, states)
        return frames

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_frames: torch.Tensor):
        cfg: HeatSolveConfig = ctx.cfg
        alpha, states = ctx.saved_tensors
        need_alpha, need_T0 = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grid = tuple(alpha.shape)
        system = ctx.system
        if system is None:  # pragma: no cover - defensive (forward always stores it)
            system = cfg.stencil(alpha).system(cfg.dt)
        ctx.system = None
        g_step = dict(zip(cfg.obs_steps, grad_frames.unbind(0)))

        fused = cfg.assembly == "fused"
        acc: _FaceProducts | None = None
        grad_alpha = None
        a_leaf = stencil_g = None
        if need_alpha:
            if fused:
                acc = _FaceProducts(grid, alpha, cfg.stencil_backend)
            else:
                grad_alpha = torch.zeros_like(alpha)
                with torch.enable_grad():
                    a_leaf = alpha.detach().requires_grad_(True)
                    stencil_g = cfg.stencil(a_leaf)

        mu_next: torch.Tensor | None = None
        last = max(n for n in cfg.obs_steps)  # μ^n = 0 for every n after the last observation
        for n in range(last, 0, -1):
            rhs = (
                torch.zeros(grid, device=alpha.device, dtype=alpha.dtype)
                if mu_next is None
                else mu_next.clone()
            )
            if n in g_step:
                rhs[..., 0] += g_step[n]
            mu = solve_step(system, rhs, rhs, cfg)  # Aᵀ = A (App. D.2 / D.3)
            if need_alpha:
                Tn = states[n - 1]
                if acc is not None:
                    acc.add(mu, Tn)
                else:
                    with torch.enable_grad():
                        (g,) = torch.autograd.grad(
                            stencil_g.apply(Tn), a_leaf, grad_outputs=mu, retain_graph=True
                        )
                    grad_alpha.add_(g, alpha=cfg.dt)
            mu_next = mu

        if acc is not None:
            S = acc.result()
            with torch.enable_grad():
                a_leaf = alpha.detach().requires_grad_(True)
                conds = face_conductances(a_leaf, cfg.spacing, cfg.bc, cfg.face_mode)
                obj = -cfg.dt * sum((c * s).sum() for c, s in zip(conds, S))
                (grad_alpha,) = torch.autograd.grad(obj, a_leaf)

        grad_T0 = None
        if need_T0:
            grad_T0 = (
                mu_next.clone()
                if mu_next is not None
                else torch.zeros(grid, device=alpha.device, dtype=alpha.dtype)
            )
            if 0 in g_step:
                grad_T0[..., 0] += g_step[0]
        return grad_alpha, grad_T0, None


def _fdiff(x: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.roll(x, shifts=-1, dims=axis) - x


class _FaceProducts:
    """Accumulator of the per-face products ``S_a = Σ_n (Δ_a μ^n)(Δ_a T^n)`` of the fused adjoint
    assembly (forward differences with periodic wrap; the boundary faces of non-periodic axes are
    masked later by their zero conductance).

    The flat path (:class:`~nefi.operators.pde.stencil.FlatLayout`) gathers ``μ`` and ``T`` into
    persistent padded buffers and accumulates in range layout: 11 kernels per time step in 3-D
    (2 gathers + 3 × (2 differences + 1 fused multiply-add)), no allocations, instead of 15
    (3 × (2 rolls + 2 differences + 1 fused multiply-add)).
    """

    def __init__(self, grid: tuple[int, ...], like: torch.Tensor, backend: str) -> None:
        self.grid = grid
        self.flat = _use_flat(backend, like)
        kw = {"device": like.device, "dtype": like.dtype}
        if not self.flat:
            self.S = [torch.zeros(grid, **kw) for _ in grid]
            return
        from .stencil import flat_layout

        lay = self.lay = flat_layout(grid, like.device)
        self.S = [torch.zeros(lay.length, **kw) for _ in grid]
        self.Pm = torch.empty(lay.n_padded, **kw)
        self.Pt = torch.empty(lay.n_padded, **kw)
        self.dm = torch.empty(lay.length, **kw)
        self.dt = torch.empty(lay.length, **kw)
        self.cm, self.ct = lay.center(self.Pm), lay.center(self.Pt)
        self.vm = [lay.neighbor(self.Pm, 2 * ax) for ax in range(len(grid))]  # +e_ax
        self.vt = [lay.neighbor(self.Pt, 2 * ax) for ax in range(len(grid))]

    def add(self, mu: torch.Tensor, T: torch.Tensor) -> None:
        if not self.flat:
            for ax in range(len(self.S)):
                self.S[ax].addcmul_(_fdiff(mu, ax), _fdiff(T, ax))
            return
        lay = self.lay
        torch.index_select(mu.reshape(-1), 0, lay.pad_from_grid, out=self.Pm)
        torch.index_select(T.reshape(-1), 0, lay.pad_from_grid, out=self.Pt)
        for ax in range(len(self.S)):
            torch.sub(self.vm[ax], self.cm, out=self.dm)
            torch.sub(self.vt[ax], self.ct, out=self.dt)
            self.S[ax].addcmul_(self.dm, self.dt)

    def result(self) -> list[torch.Tensor]:
        """``S_a`` on the grid."""
        if not self.flat:
            return self.S
        return [self.lay.from_range(s) for s in self.S]


def implicit_euler_frames(
    alpha: torch.Tensor,
    T0: torch.Tensor,
    cfg: HeatSolveConfig,
    grad_mode: str = "adjoint",
) -> torch.Tensor:
    """Observed surface frames of the implicit-Euler heat solve with the chosen gradient mode.

    Args:
        alpha: diffusivity on the grid.
        T0: initial (ambient-shifted) temperature on the grid.
        cfg: solve configuration.
        grad_mode: ``"adjoint"`` (discrete adjoint, default), ``"autograd"`` (back-propagation
            through the unrolled inner solver, ``O(K N_g N_t)`` memory — reference) or
            ``"checkpoint"`` (``torch.utils.checkpoint`` per time step: recompute the inner
            iterations of one step during backward).
    """
    if grad_mode not in GRAD_MODES:
        raise ConfigError(f"unknown grad_mode {grad_mode!r}; use one of {GRAD_MODES}")
    if not torch.is_grad_enabled() or not (alpha.requires_grad or T0.requires_grad):
        with torch.no_grad():
            return rollout(alpha, T0, cfg)[0]
    if grad_mode == "adjoint":
        return ImplicitEulerAdjoint.apply(alpha, T0, cfg)
    if grad_mode == "autograd":
        return rollout(alpha, T0, cfg)[0]
    return _checkpointed(alpha, T0, cfg)


def _checkpointed(alpha: torch.Tensor, T0: torch.Tensor, cfg: HeatSolveConfig) -> torch.Tensor:
    from torch.utils.checkpoint import checkpoint

    def step_fn(system: ImplicitSystem, b: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        # pass every tensor the solve depends on explicitly so the recomputation is exact
        tensors = [system.diag, *system.offdiag]
        n_off = len(system.offdiag)

        def run(b_: torch.Tensor, diag_: torch.Tensor, *off: torch.Tensor) -> torch.Tensor:
            sys_ = _SystemView(system.ndim, diag_, list(off[:n_off]), system.backend)
            return solve_step(sys_, b_, b_, cfg)

        return checkpoint(run, b, *tensors, use_reentrant=False)

    return rollout(alpha, T0, cfg, step_fn=step_fn)[0]


class _SystemView(ImplicitSystem):
    """An :class:`ImplicitSystem` rebuilt from explicit tensors (for checkpoint recomputation)."""

    def __init__(
        self,
        ndim: int,
        diag: torch.Tensor,
        offdiag: Sequence[torch.Tensor],
        backend: str = "auto",
    ) -> None:
        self.ndim = ndim
        self.diag = diag
        self.offdiag = list(offdiag)
        self.inv_diag = 1.0 / diag
        self._jacobi_w = None
        self._flat = {}
        self.backend = backend
        self.dt = float("nan")
        self.stencil = None  # type: ignore[assignment]
