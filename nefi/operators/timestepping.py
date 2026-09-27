"""Generic differentiable time integration with memory-controlled adjoints.

:class:`TimeStepper` turns *any* explicit or implicit time-stepping scheme you can write in PyTorch
into a nefi operator::

    state_0     = init(fields)
    state_{n+1} = step(state_n, params, dt, n)          # your physics, any torch ops
    y           = stack([observe(state_n, n) for n in 0..N if not None])

Gradients w.r.t. the unknown fields (initial condition, coefficients, sources) come from autograd
through the unrolled scheme — the *discrete adjoint* of your discretization, so the soft-constraint
decoupling of PINNs (NeFTY §3.3) cannot occur. ``grad_mode="checkpoint"`` recomputes blocks of
``checkpoint_every`` steps during the backward pass (``torch.utils.checkpoint``), cutting memory
from O(N·state) to O(N/k·state + k·state) at the price of one extra forward. (A dedicated
implicit-Euler heat solver with an exact hand-written discrete adjoint lives in
``nefi.operators.pde``.)

Two steppers ship as plain functions: :func:`wave_step` (scalar wave equation, leapfrog, absorbing
/ Neumann / Dirichlet / periodic boundaries) and :func:`advection_diffusion_step` (upwind
advection + explicit diffusion), plus :func:`stable_dt` and :func:`wave_initial_state`.

Example (1-D photoacoustics: recover the initial pressure from two boundary sensors)::

    from functools import partial
    dom = nefi.Domain.unit((64,))
    h = dom.spacing()
    dt = stable_dt(h, c_max=1.0, cfl=0.5)
    op = TimeStepper(
        step=partial(wave_step, spacing=h, c=1.0, boundary="absorbing"),
        init=lambda f: wave_initial_state(f["p0"], c=1.0, dt=dt, spacing=h),
        n_steps=200, dt=dt,
        observe=lambda s, n: s[1][[0, -1]],             # u at both ends, every step
        field="p0", grad_mode="checkpoint", checkpoint_every=25,
    )

See also :func:`nefi.physics.timestep.run_timestepping`, the function-level counterpart used by
the physics zoo (wave, reaction–diffusion); both use the same block-checkpointing strategy.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from ..errors import OperatorError
from ..registry import register
from ..utils.tensor import shape_tuple
from .base import Fields, Operator
from .function import upsample_fields

State = torch.Tensor | tuple[torch.Tensor, ...]
StepFn = Callable[[State, Mapping[str, Any], float, int], State]


# ---------------------------------------------------------------------------------------------
# finite-difference helpers (trailing `len(spacing)` dims are spatial)
# ---------------------------------------------------------------------------------------------
_BOUNDARIES = ("periodic", "neumann", "dirichlet", "absorbing")


def _pad1(u: torch.Tensor, dim: int, boundary: str) -> torch.Tensor:
    n = u.shape[dim]
    first, last = u.narrow(dim, 0, 1), u.narrow(dim, n - 1, 1)
    if boundary == "periodic":
        return torch.cat([last, u, first], dim=dim)
    if boundary in ("neumann", "absorbing"):
        return torch.cat([first, u, last], dim=dim)
    if boundary == "dirichlet":
        return torch.cat([torch.zeros_like(first), u, torch.zeros_like(last)], dim=dim)
    raise OperatorError(f"unknown boundary {boundary!r}; choose one of {_BOUNDARIES}")


def laplacian(u: torch.Tensor, spacing: Sequence[float], boundary: str = "neumann") -> torch.Tensor:
    """Second-order finite-difference Laplacian over the trailing ``len(spacing)`` dims.

    Example::

        lap = laplacian(torch.rand(32, 32), spacing=(0.1, 0.1), boundary="periodic")
    """
    d = len(spacing)
    out = torch.zeros_like(u)
    for i, h in enumerate(spacing):
        dim = u.ndim - d + i
        n = u.shape[dim]
        up = _pad1(u, dim, boundary)
        out = out + (up.narrow(dim, 0, n) - 2.0 * u + up.narrow(dim, 2, n)) / float(h) ** 2
    return out


def _resolve(value: Any, params: Mapping[str, Any], n: int, name: str) -> Any:
    """``str`` -> ``params[value]``; callable -> ``value(params, n)``; else the value itself."""
    if isinstance(value, str):
        try:
            return params[value]
        except KeyError as e:
            raise OperatorError(
                f"stepper argument {name}={value!r} refers to a field that is not available; "
                f"fields: {tuple(params)}"
            ) from e
    if callable(value) and not torch.is_tensor(value):
        return value(params, n)
    return value


def stable_dt(
    spacing: Sequence[float],
    *,
    c_max: float | None = None,
    velocity_max: float | Sequence[float] | None = None,
    diffusivity_max: float | None = None,
    cfl: float = 0.9,
) -> float:
    """Largest stable explicit time step (times a safety factor ``cfl`` ≤ 1).

    * wave equation, leapfrog: ``dt ≤ 1 / (c_max · sqrt(Σ 1/h_a²))``;
    * advection–diffusion, upwind + explicit Euler: ``dt ≤ 1 / (Σ |v_a|/h_a + 2 D Σ 1/h_a²)``.

    Example::

        dt = stable_dt((0.01, 0.01), c_max=1500.0, cfl=0.5)
    """
    inv2 = sum(1.0 / float(h) ** 2 for h in spacing)
    bounds = []
    if c_max:
        bounds.append(1.0 / (float(c_max) * math.sqrt(inv2)))
    if velocity_max is not None or diffusivity_max:
        v = velocity_max if velocity_max is not None else 0.0
        vs = (float(v),) * len(spacing) if isinstance(v, int | float) else tuple(map(float, v))
        rate = sum(abs(vi) / float(h) for vi, h in zip(vs, spacing))
        rate += 2.0 * float(diffusivity_max or 0.0) * inv2
        if rate > 0:
            bounds.append(1.0 / rate)
    if not bounds:
        raise OperatorError("stable_dt needs c_max, velocity_max and/or diffusivity_max")
    return float(cfl) * min(bounds)


# ---------------------------------------------------------------------------------------------
# steppers
# ---------------------------------------------------------------------------------------------
def _mur(
    u_next: torch.Tensor, u: torch.Tensor, c: Any, dt: float, spacing: Sequence[float]
) -> torch.Tensor:
    """First-order Mur (Engquist–Majda) absorbing boundary on every face."""
    d = len(spacing)
    for i, h in enumerate(spacing):
        dim = u.ndim - d + i
        n = u.shape[dim]
        if n < 3:
            continue
        if torch.is_tensor(c) and c.ndim >= d:
            c_lo, c_hi = c.narrow(dim, 0, 1), c.narrow(dim, n - 1, 1)
        else:
            c_lo = c_hi = c
        k_lo = (c_lo * dt - h) / (c_lo * dt + h)
        k_hi = (c_hi * dt - h) / (c_hi * dt + h)
        lo = u.narrow(dim, 1, 1) + k_lo * (u_next.narrow(dim, 1, 1) - u.narrow(dim, 0, 1))
        hi = u.narrow(dim, n - 2, 1) + k_hi * (
            u_next.narrow(dim, n - 2, 1) - u.narrow(dim, n - 1, 1)
        )
        u_next = torch.cat([lo, u_next.narrow(dim, 1, n - 2), hi], dim=dim)
    return u_next


def wave_step(
    state: State,
    params: Mapping[str, Any],
    dt: float,
    n: int,
    *,
    spacing: Sequence[float],
    c: Any = "c",
    boundary: str = "absorbing",
    source: Any = None,
    damping: Any = None,
) -> State:
    """One leapfrog step of ``u_tt = c² Δu + s`` (optionally damped ``− γ u_t``).

    ``state = (u_prev, u)`` → ``(u, u_next)`` with
    ``u_next = 2u − u_prev + (c dt)² Δu + dt² s`` (second order in space and time; stable for
    ``dt ≤ stable_dt(spacing, c_max=max c)``).

    Args:
        state: ``(u_prev, u)`` tensors (trailing ``len(spacing)`` dims spatial).
        params: the fields dict (from :class:`TimeStepper`).
        dt, n: time step and step index.
        spacing: physical grid spacing per spatial axis.
        c: wave speed — a float/tensor, the *name* of a field in ``params`` (unknown speed),
            or a callable ``(params, n) -> tensor``.
        boundary: ``"absorbing"`` (1st-order Mur), ``"neumann"`` (rigid), ``"dirichlet"``
            (pressure-release), ``"periodic"``.
        source: optional source term ``s`` (float/tensor/field name/callable ``(params, n)``).
        damping: optional damping rate ``γ`` (float/tensor/field name/callable).

    Example::

        u0 = torch.exp(-((torch.linspace(0, 1, 100) - 0.5) / 0.05) ** 2)
        step = partial(wave_step, spacing=(0.01,), c=1.0, boundary="absorbing")
        state = step((u0, u0), {}, 0.005, 0)          # (u, u_next)
    """
    if boundary not in _BOUNDARIES:
        raise OperatorError(f"unknown boundary {boundary!r}; choose one of {_BOUNDARIES}")
    u_prev, u = state
    cv = _resolve(c, params, n, "c")
    lap = laplacian(u, spacing, "neumann" if boundary == "absorbing" else boundary)
    acc = cv**2 * lap
    if source is not None:
        acc = acc + _resolve(source, params, n, "source")
    if damping is not None:
        g = _resolve(damping, params, n, "damping")
        # centered damping: (1 + γdt/2) u_next = 2u − (1 − γdt/2) u_prev + dt² acc
        u_next = (2.0 * u - (1.0 - 0.5 * g * dt) * u_prev + dt**2 * acc) / (1.0 + 0.5 * g * dt)
    else:
        u_next = 2.0 * u - u_prev + dt**2 * acc
    if boundary == "absorbing":
        u_next = _mur(u_next, u, cv, dt, spacing)
    return (u, u_next)


def wave_initial_state(
    u0: torch.Tensor,
    v0: torch.Tensor | None = None,
    *,
    c: Any,
    dt: float,
    spacing: Sequence[float],
    boundary: str = "neumann",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Leapfrog start ``(u_{-1}, u_0)`` for initial displacement ``u0`` and velocity ``v0``.

    Uses the second-order Taylor start ``u_{-1} = u0 − dt v0 + ½ (c dt)² Δu0``.

    Example::

        p0 = torch.exp(-((torch.linspace(0, 1, 64) - 0.5) / 0.05) ** 2)
        state = wave_initial_state(p0, c=1.0, dt=0.005, spacing=(1 / 64,))
    """
    bc = "neumann" if boundary == "absorbing" else boundary
    u_prev = u0 + 0.5 * (c * dt) ** 2 * laplacian(u0, spacing, bc)
    if v0 is not None:
        u_prev = u_prev - dt * v0
    return (u_prev, u0)


def advection_diffusion_step(
    state: State,
    params: Mapping[str, Any],
    dt: float,
    n: int,
    *,
    spacing: Sequence[float],
    velocity: Any = None,
    diffusivity: Any = 0.0,
    boundary: str = "periodic",
    source: Any = None,
) -> State:
    """One explicit step of ``u_t + v·∇u = ∇·(D ∇u) + s`` (first-order upwind advection).

    Args:
        state: ``u`` or ``(u,)``; the same structure is returned.
        params: the fields dict (from :class:`TimeStepper`).
        dt, n: time step and index (``dt ≤ stable_dt(spacing, velocity_max=..,
            diffusivity_max=..)``).
        spacing: physical grid spacing per spatial axis.
        velocity: per-axis velocity — a sequence of floats/tensors, a tensor ``(d, *shape)``, a
            field name / callable returning either; ``None`` for pure diffusion.
        diffusivity: ``D`` — float, tensor (conservative form with face-averaged ``D``), field
            name (unknown diffusivity) or callable.
        boundary: ``"periodic"``, ``"neumann"`` (no flux) or ``"dirichlet"`` (zero).
        source: optional source (float/tensor/field name/callable ``(params, n)``).

    Example::

        h = 1 / 64
        step = partial(advection_diffusion_step, spacing=(h,), velocity=(1.0,), diffusivity=1e-3)
        u1 = step(torch.rand(64), {}, stable_dt((h,), velocity_max=1.0, diffusivity_max=1e-3), 0)
    """
    if boundary not in ("periodic", "neumann", "dirichlet"):
        raise OperatorError("advection_diffusion_step boundary must be periodic|neumann|dirichlet")
    single = torch.is_tensor(state)
    u = state if single else state[0]
    d = len(spacing)
    rhs = torch.zeros_like(u)
    vel = _resolve(velocity, params, n, "velocity")
    if vel is not None:
        comps = [vel[i] for i in range(d)] if not isinstance(vel, int | float) else [vel] * d
        for i, h in enumerate(spacing):
            dim = u.ndim - d + i
            nn_ = u.shape[dim]
            up = _pad1(u, dim, boundary)
            d_minus = (u - up.narrow(dim, 0, nn_)) / float(h)
            d_plus = (up.narrow(dim, 2, nn_) - u) / float(h)
            v = comps[i]
            if torch.is_tensor(v):
                rhs = rhs - (v.clamp_min(0) * d_minus + v.clamp_max(0) * d_plus)
            else:
                rhs = rhs - (v * d_minus if v >= 0 else v * d_plus)
    D = _resolve(diffusivity, params, n, "diffusivity")
    if torch.is_tensor(D) and D.ndim >= d:
        for i, h in enumerate(spacing):
            dim = u.ndim - d + i
            nn_ = u.shape[dim]
            up = _pad1(u, dim, boundary)
            Dp = _pad1(D, D.ndim - d + i, "neumann")
            Dface_hi = 0.5 * (D + Dp.narrow(D.ndim - d + i, 2, nn_))
            Dface_lo = 0.5 * (D + Dp.narrow(D.ndim - d + i, 0, nn_))
            flux_hi = Dface_hi * (up.narrow(dim, 2, nn_) - u) / float(h)
            flux_lo = Dface_lo * (u - up.narrow(dim, 0, nn_)) / float(h)
            rhs = rhs + (flux_hi - flux_lo) / float(h)
    elif D is not None and (torch.is_tensor(D) or float(D) != 0.0):
        rhs = rhs + D * laplacian(u, spacing, boundary)
    if source is not None:
        rhs = rhs + _resolve(source, params, n, "source")
    u_new = u + dt * rhs
    return u_new if single else (u_new, *tuple(state[1:]))


# ---------------------------------------------------------------------------------------------
# operator
# ---------------------------------------------------------------------------------------------
@register("operator", "time_stepper")
class TimeStepper(Operator):
    """Differentiable time integration of a user-written scheme (see module docstring).

    Args:
        step: ``step(state, params, dt, n) -> state``; ``state`` is a tensor or a tuple of tensors.
        init: ``init(fields) -> state`` (e.g. the unknown initial condition, or zeros).
        n_steps: number of steps ``N``.
        dt: time step.
        observe: ``observe(state, n) -> Tensor | None`` called for ``n = 0..N`` on the state after
            ``n`` steps; non-``None`` returns are stacked along a new leading (time) axis.
            Default: the final state's first component.
        params: optional ``params(fields) -> dict`` passed to ``step`` (default: the fields).
        field: primary unknown (homogeneity / scale correction refer to it).
        fields: all field names the scheme reads (default ``(field,)``).
        grad_mode: ``"autograd"`` (store every step) or ``"checkpoint"`` (recompute blocks of
            ``checkpoint_every`` steps in the backward pass; identical gradients, less memory).
        checkpoint_every: block length for ``"checkpoint"``.
        homogeneity: degree of homogeneity in ``field`` if known (linear PDE in the initial
            condition → 1).
        native_shape: grid the scheme is written for; coarse fields are upsampled to it.
        output_shape: fixed observation shape (enables multiscale curricula); optional.

    Example::

        h = (1 / 64,)
        dt = stable_dt(h, velocity_max=1.0)
        op = TimeStepper(partial(advection_diffusion_step, spacing=h, velocity=(1.0,)),
                         init=lambda f: f["u0"], n_steps=100, dt=dt, field="u0")
        u_final = op({"u0": torch.rand(64)})
    """

    traceable = False  # a Python time loop (whole-step compile runs it eagerly)

    def __init__(
        self,
        step: StepFn,
        init: Callable[[Fields], State],
        n_steps: int,
        dt: float,
        observe: Callable[[State, int], torch.Tensor | None] | None = None,
        *,
        params: Callable[[Fields], Mapping[str, Any]] | None = None,
        field: str = "x",
        fields: Sequence[str] | None = None,
        grad_mode: str = "autograd",
        checkpoint_every: int = 16,
        homogeneity: float | None = None,
        native_shape: Sequence[int] | None = None,
        output_shape: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if grad_mode not in ("autograd", "checkpoint"):
            raise OperatorError(f"grad_mode must be 'autograd' or 'checkpoint', got {grad_mode!r}")
        if n_steps < 0 or checkpoint_every < 1:
            raise OperatorError("n_steps must be >= 0 and checkpoint_every >= 1")
        self.step_fn = step
        self.init_fn = init
        self.n_steps = int(n_steps)
        self.dt = float(dt)
        self.observe_fn = observe
        self.params_fn = params
        self.primary = field
        self._fields = tuple(fields) if fields is not None else (field,)
        self.grad_mode = grad_mode
        self.checkpoint_every = int(checkpoint_every)
        self.homogeneity = homogeneity
        self.native_shape = None if native_shape is None else shape_tuple(native_shape)
        self._output_shape = None if output_shape is None else shape_tuple(output_shape)

    # -- helpers --------------------------------------------------------------------------------
    def _observe(self, state: tuple, single: bool, n: int) -> torch.Tensor | None:
        s = state[0] if single else state
        if self.observe_fn is None:
            return None
        return self.observe_fn(s, n)

    def _block(self, n0: int, k: int, single: bool, params, *state):
        st = tuple(state)
        obs = []
        for n in range(n0, n0 + k):
            new = self.step_fn(st[0] if single else st, params, self.dt, n)
            st = (new,) if torch.is_tensor(new) else tuple(new)
            o = self._observe(st, single, n + 1)
            if o is not None:
                obs.append(o)
        return (*st, *obs)

    # -- Operator API ---------------------------------------------------------------------------
    def forward(self, fields: Fields) -> torch.Tensor:
        if self.native_shape is not None:
            fields = upsample_fields(fields, self.native_shape, names=self._fields)
        missing = [k for k in self._fields if k not in fields]
        if missing:
            raise OperatorError(f"TimeStepper needs fields {missing}; available: {tuple(fields)}")
        params = dict(self.params_fn(fields)) if self.params_fn is not None else dict(fields)
        s0 = self.init_fn(fields)
        single = torch.is_tensor(s0)
        st = (s0,) if single else tuple(s0)
        obs: list[torch.Tensor] = []
        o = self._observe(st, single, 0)
        if o is not None:
            obs.append(o)
        use_ckpt = self.grad_mode == "checkpoint" and torch.is_grad_enabled()
        n = 0
        while n < self.n_steps:
            k = min(self.checkpoint_every if use_ckpt else self.n_steps, self.n_steps - n)
            if use_ckpt:
                out = checkpoint(self._block, n, k, single, params, *st, use_reentrant=False)
            else:
                out = self._block(n, k, single, params, *st)
            st, new_obs = tuple(out[: len(st)]), list(out[len(st) :])
            obs.extend(new_obs)
            n += k
        if self.observe_fn is None:
            return st[0]
        if not obs:
            raise OperatorError("observe() returned None for every step; nothing was measured")
        shapes = {tuple(t.shape) for t in obs}
        if len(shapes) > 1:
            raise OperatorError(
                f"observe() returned tensors of different shapes {sorted(shapes)}; return the "
                "same shape at every observed step (or None to skip a step)"
            )
        return torch.stack(obs)

    def at_resolution(self, shape):
        return self

    def output_shape(self, shape):
        return self._output_shape

    def required_fields(self):
        return self._fields

    def extra_repr(self) -> str:
        return (
            f"n_steps={self.n_steps}, dt={self.dt:.3g}, grad_mode={self.grad_mode}, "
            f"checkpoint_every={self.checkpoint_every}, fields={self._fields}"
        )


__all__ = [
    "TimeStepper",
    "advection_diffusion_step",
    "laplacian",
    "stable_dt",
    "wave_initial_state",
    "wave_step",
]
