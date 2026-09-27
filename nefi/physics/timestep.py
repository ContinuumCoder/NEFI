"""Generic explicit time integration with optional per-block gradient checkpointing.

Every time-domain operator of the wave / reaction–diffusion family is an unrolled explicit scheme

.. code-block:: text

    state_0          = given
    state_{n+1}      = step_fn(state_n, params, n, dt)          n = 0 .. n_steps-1
    observations     = [observe(state_n, n) for n in record]

and its gradient is the *discrete adjoint* of that scheme, obtained by reverse-mode autodiff through
the unrolled loop (so the physics stays a hard constraint, NeFTY §3.3/§4.3).

Memory trade-off (NeFTY §4.3 / App. D.3 make the same argument for the implicit heat solver):
plain backpropagation through time stores the intermediate tensors of *every* step, i.e.
``O(n_steps · S)`` memory for a state of size ``S``. NeFTY avoids this with a hand-written discrete
adjoint (``O(S)`` memory, one extra backward solve). A hand-written adjoint is exact and optimal but
must be re-derived for every scheme; here we use the general-purpose alternative, **per-block
checkpointing** (Griewank & Walther 2000, "Algorithm 799: revolve"; Chen et al. 2016,
arXiv:1604.06174): the loop is cut into blocks of ``k = checkpoint_every`` steps, only the state at
block boundaries is stored during the forward pass, and each block is recomputed once during the
backward pass. Peak memory becomes ``O((n_steps / k) · S + k · S)`` — minimized by
``k ≈ sqrt(n_steps)`` (:func:`auto_checkpoint_every`) — at the price of one extra forward
evaluation (≈ 1.3–1.5× wall-clock of plain autograd). Gradients are *identical* to plain autograd
(the recomputation is deterministic), which the tests check to 1e-6.

Stability helpers (:func:`stable_dt_wave`, :func:`stable_dt_diffusion`) return the largest stable
explicit step (times a safety factor) from a von Neumann analysis of the discrete operators used in
:mod:`nefi.physics.wave` and :mod:`nefi.physics.reaction_diffusion`.

Note: :class:`nefi.operators.timestepping.TimeStepper` is a parallel, operator-level abstraction of
the same idea; this module keeps a small functional core that physics modules call directly.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

import torch
from torch.utils.checkpoint import checkpoint

from ..errors import ConfigError

State = tuple[torch.Tensor, ...]
StepFn = Callable[[State, tuple[torch.Tensor, ...], int, float], State]
ObserveFn = Callable[[State, int], "torch.Tensor | None"]

__all__ = [
    "State",
    "auto_checkpoint_every",
    "laplacian_eig_max",
    "run_timestepping",
    "stable_dt_diffusion",
    "stable_dt_wave",
]


def _as_state(x: torch.Tensor | Sequence[torch.Tensor]) -> State:
    if torch.is_tensor(x):
        return (x,)
    return tuple(x)


def _as_params(p: torch.Tensor | Sequence[torch.Tensor] | None) -> tuple[torch.Tensor, ...]:
    if p is None:
        return ()
    if torch.is_tensor(p):
        return (p,)
    return tuple(p)


def auto_checkpoint_every(n_steps: int) -> int:
    """Block length ``ceil(sqrt(n_steps))`` minimizing ``n_steps/k + k`` (memory-optimal)."""
    return max(1, int(math.ceil(math.sqrt(max(1, n_steps)))))


def run_timestepping(
    step_fn: StepFn,
    state0: torch.Tensor | Sequence[torch.Tensor],
    params: torch.Tensor | Sequence[torch.Tensor] | None,
    n_steps: int,
    dt: float,
    observe: ObserveFn | None = None,
    *,
    checkpoint_every: int | None = None,
    record: Iterable[int] | None = None,
) -> tuple[State, list[torch.Tensor]]:
    """Integrate ``state_{n+1} = step_fn(state_n, params, n, dt)`` for ``n_steps`` steps.

    Args:
        step_fn: one explicit step ``(state, params, n, dt) -> new_state``; ``state`` is a tuple of
            tensors and ``n`` the index of the *current* state (time ``t_n = n·dt``). Must be a
            deterministic function of its inputs (required for exact checkpoint recomputation).
        state0: initial state (a tensor or a tuple of tensors).
        params: tensors the step depends on that may require gradients (e.g. the unknown field).
            They are passed explicitly to the checkpointed blocks so their gradients are tracked.
        n_steps: number of steps.
        dt: time step (physical units), forwarded to ``step_fn``.
        observe: ``(state, n) -> Tensor | None`` evaluated at the recorded step indices; ``None``
            results are skipped. The observation of state ``n`` is taken *after* ``n`` steps.
        checkpoint_every: if given (and autograd is recording), the loop is cut into blocks of this
            many steps that are recomputed in the backward pass (``torch.utils.checkpoint``,
            non-reentrant). Memory ``∝ n_steps/checkpoint_every + checkpoint_every`` states.
        record: step indices ``0 ≤ n ≤ n_steps`` at which ``observe`` is called
            (default: every step ``1..n_steps``).

    Returns:
        ``(final_state, observations)`` with ``observations`` in increasing ``n`` order.

    Example::

        def step(s, p, n, dt):            # u' = -k u, explicit Euler
            (u,) = s
            return (u - dt * p[0] * u,)

        (u_T,), obs = run_timestepping(step, torch.ones(3), (k,), 100, 0.01,
                                       lambda s, n: s[0], record=[50, 100])
    """
    n_steps = int(n_steps)
    if n_steps < 0:
        raise ConfigError(f"n_steps must be >= 0, got {n_steps}")
    state = _as_state(state0)
    prm = _as_params(params)
    if record is None:
        rec = set(range(1, n_steps + 1))
    else:
        rec = {int(n) for n in record}
        bad = [n for n in rec if n < 0 or n > n_steps]
        if bad:
            raise ConfigError(f"record indices {sorted(bad)} outside [0, {n_steps}]")
    obs: list[torch.Tensor] = []

    def _obs(st: State, n: int, out: list) -> None:
        if observe is not None and n in rec:
            o = observe(st, n)
            if o is not None:
                out.append(o)

    _obs(state, 0, obs)
    needs_grad = torch.is_grad_enabled() and (
        any(t.requires_grad for t in prm) or any(t.requires_grad for t in state)
    )
    k = n_steps if not (checkpoint_every and needs_grad) else max(1, int(checkpoint_every))
    if k >= n_steps or not needs_grad:
        for n in range(n_steps):
            state = _as_state(step_fn(state, prm, n, dt))
            _obs(state, n + 1, obs)
        return state, obs

    n_state = len(state)
    n0 = 0
    while n0 < n_steps:
        m = min(k, n_steps - n0)

        def block(*args: torch.Tensor, n0: int = n0, m: int = m) -> tuple[torch.Tensor, ...]:
            st: State = tuple(args[:n_state])
            pp = tuple(args[n_state:])
            out: list[torch.Tensor] = []
            for i in range(m):
                st = _as_state(step_fn(st, pp, n0 + i, dt))
                _obs(st, n0 + i + 1, out)
            return (*st, *out)

        res = checkpoint(block, *state, *prm, use_reentrant=False)
        state = tuple(res[:n_state])
        obs.extend(res[n_state:])
        n0 += m
    return state, obs


# --------------------------------------------------------------------------------------------
# stability limits (von Neumann analysis of the centered stencils used in this package)
# --------------------------------------------------------------------------------------------
#: max |symbol| · h² of the 1-D centered second-derivative stencils, per accuracy order:
#: order 2 ``[1, -2, 1]`` → 4; order 4 ``[-1/12, 4/3, -5/2, 4/3, -1/12]`` → 16/3.
_LAP_EIG = {2: 4.0, 4: 16.0 / 3.0}


def laplacian_eig_max(spacing: Sequence[float], order: int = 2) -> float:
    """Largest eigenvalue magnitude of the discrete Laplacian, ``Σ_a ρ_order / h_a²``."""
    if order not in _LAP_EIG:
        raise ConfigError(f"Laplacian order must be 2 or 4, got {order}")
    return sum(_LAP_EIG[order] / float(h) ** 2 for h in spacing)


def stable_dt_wave(
    c_max: float, spacing: Sequence[float], order: int = 2, courant: float = 0.9
) -> float:
    """Largest stable leapfrog step for ``p_tt = c² Δp`` times a safety factor ``courant``.

    Leapfrog is stable iff ``dt² c_max² λ_max ≤ 4`` with ``λ_max`` = :func:`laplacian_eig_max`.
    For order 2 this is the classic CFL condition ``c dt ≤ h / sqrt(d)``; for order 4 it is
    ``c dt ≤ h sqrt(3 / (4 d))`` (≈ 0.612 h in 2-D). A damping (sponge) term treated with the
    centered average in :mod:`nefi.physics.wave` does not tighten the bound.

    Args:
        c_max: maximum wave speed (physical units, e.g. km/s).
        spacing: grid spacing per axis (physical units, e.g. km).
        order: spatial accuracy order of the Laplacian (2 or 4).
        courant: safety factor in ``(0, 1]``.
    """
    if not 0.0 < courant <= 1.0:
        raise ConfigError(f"courant safety factor must be in (0, 1], got {courant}")
    if c_max <= 0:
        raise ConfigError(f"c_max must be positive, got {c_max}")
    return float(courant) * 2.0 / (float(c_max) * math.sqrt(laplacian_eig_max(spacing, order)))


def stable_dt_diffusion(
    d_max: float, spacing: Sequence[float], rate_max: float = 0.0, safety: float = 0.9
) -> float:
    """Largest stable explicit-Euler step for ``u_t = D Δu − r u`` (5-point / 2nd-order Laplacian).

    Explicit Euler is stable iff ``dt (D λ_max + r) ≤ 2`` with ``λ_max = Σ 4/h_a²``, i.e. the
    classic ``dt ≤ h² / (2 d D)`` for ``r = 0``. ``rate_max`` bounds the linearized reaction rate
    (e.g. ``F + k + v²`` for Gray–Scott); positive-feedback terms are not stability-limiting over
    the short observation windows used here.
    """
    if not 0.0 < safety <= 1.0:
        raise ConfigError(f"safety factor must be in (0, 1], got {safety}")
    lam = float(d_max) * laplacian_eig_max(spacing, 2) + max(0.0, float(rate_max))
    if lam <= 0:
        raise ConfigError("stable_dt_diffusion needs a positive diffusivity or rate")
    return float(safety) * 2.0 / lam
