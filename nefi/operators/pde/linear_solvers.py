"""Matrix-free linear solvers for the implicit-Euler system ``A(α) T^{n+1} = T^n``.

Both solvers only need a callable ``apply_A(x) -> A x`` (plus the diagonal for Jacobi), are built
from plain tensor operations (so they can be unrolled under autograd for the reference gradient
mode), run on any device and are **warm-started** from ``x0`` (NeFTY App. D.2: the previous frame).

* :func:`jacobi` — fixed number ``K`` of (optionally damped) Jacobi sweeps, NeFTY Eq. (24). ``A`` is
  strictly diagonally dominant (App. D.2), so the iteration converges geometrically.
* :func:`conjugate_gradient` — (Jacobi-preconditioned) conjugate gradients for SPD ``A``; much
  faster convergence for the paper's ``Δt``/spacing regime, and the solver of choice for tight
  tolerances (adjoint verification tests).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

__all__ = ["conjugate_gradient", "jacobi"]

MatVec = Callable[[torch.Tensor], torch.Tensor]


def _vdot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum()


def jacobi(
    apply_A: MatVec,
    diag: torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    iters: int = 50,
    *,
    omega: float = 1.0,
    apply_offdiag: MatVec | None = None,
    compute_residual: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Unrolled (damped) Jacobi iteration, NeFTY Eq. (24).

    ``x^{κ+1} = x^κ + ω D⁻¹ (b - A x^κ)``; with ``ω = 1`` and ``apply_offdiag`` (``R = A - D``)
    this is the classical form ``x^{κ+1} = D⁻¹ (b - R x^κ)``.

    Args:
        apply_A: ``x -> A x``.
        diag: diagonal ``D`` of ``A`` (broadcastable to ``b``).
        b: right-hand side.
        x0: initial guess (warm start; default ``b``, i.e. the previous frame for ``A ≈ I``).
        iters: number of sweeps ``K`` (NeFTY Tab. 5: 50).
        omega: damping factor (1 = plain Jacobi).
        apply_offdiag: optional ``x -> R x`` fast path used when ``omega == 1``.
        compute_residual: also return ``‖b - A x_K‖₂`` (computed without autograd).

    Returns:
        ``(x_K, residual_norm)`` (``residual_norm`` is ``None`` if not requested).
    """
    inv_d = 1.0 / diag
    x = b if x0 is None else x0
    if apply_offdiag is not None and omega == 1.0:
        for _ in range(int(iters)):
            x = (b - apply_offdiag(x)) * inv_d
    else:
        for _ in range(int(iters)):
            x = x + omega * (b - apply_A(x)) * inv_d
    res = None
    if compute_residual:
        with torch.no_grad():
            res = torch.linalg.vector_norm(b - apply_A(x))
    return x, res


def conjugate_gradient(
    apply_A: MatVec,
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    tol: float = 1e-6,
    max_iter: int = 200,
    *,
    atol: float = 0.0,
    precond: torch.Tensor | MatVec | None = None,
    history: list[float] | None = None,
    check_every: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(Preconditioned) conjugate gradients for a symmetric positive-definite ``A``.

    Stops when ``‖r‖₂ ≤ max(tol·‖b‖₂, atol)`` or after ``max_iter`` iterations. Every operation is a
    differentiable tensor op, so the iteration can be unrolled under autograd (reference gradient
    mode); the stopping test is plain Python control flow.

    Args:
        apply_A: ``x -> A x`` (SPD).
        b: right-hand side.
        x0: initial guess (warm start; default zeros).
        tol: relative residual tolerance (``0`` disables the test: exactly ``max_iter`` iterations,
            no host synchronization — useful on GPUs).
        max_iter: maximum number of iterations.
        atol: absolute residual tolerance.
        precond: optional preconditioner: a tensor ``d`` (Jacobi / diagonal, applies ``r / d``) or a
            callable ``r -> M⁻¹ r``.
        history: if a list is given, the residual norm of every iterate (including ``x0``) is
            appended (costs one host sync per iteration).
        check_every: test convergence every ``check_every`` iterations (fewer host syncs).

    Returns:
        ``(x, ‖b - A x‖₂)`` where the residual norm is the recursively updated one.
    """
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0
        r = b - apply_A(x0)

    if precond is None:

        def m_inv(v):
            return v

    elif torch.is_tensor(precond):
        inv = 1.0 / precond

        def m_inv(v):
            return v * inv

    else:
        m_inv = precond

    b_norm = float(torch.linalg.vector_norm(b.detach())) if tol > 0 else 0.0
    target = max(tol * b_norm, float(atol))
    tiny = torch.finfo(b.dtype).tiny
    z = m_inv(r)
    p = z
    rz = _vdot(r, z)
    if history is not None:
        history.append(float(torch.linalg.vector_norm(r.detach())))
    for k in range(int(max_iter)):
        if target > 0 and k % max(1, check_every) == 0:
            if float(torch.linalg.vector_norm(r.detach())) <= target:
                break
        Ap = apply_A(p)
        # rᵀM⁻¹r ≥ 0 and pᵀAp ≥ 0 for SPD A / M: the clamps only guard the exact-solution case
        step = rz / _vdot(p, Ap).clamp_min(tiny)
        x = x + step * p
        r = r - step * Ap
        z = m_inv(r)
        rz_new = _vdot(r, z)
        p = z + (rz_new / rz.clamp_min(tiny)) * p
        rz = rz_new
        if history is not None:
            history.append(float(torch.linalg.vector_norm(r.detach())))
    return x, torch.linalg.vector_norm(r.detach())
