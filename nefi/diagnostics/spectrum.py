"""Spectral diagnostics: singular values of the linearized operator and Hessian conditioning.

* :func:`singular_values` — top-``k`` singular values of ``dF/dx`` at the current field, from a
  Lanczos iteration on ``JᵀJ`` driven by jvp/vjp products (matrix-free, any operator). Their decay
  quantifies ill-posedness: NeFTY Prop. 2 proves algebraic decay ``σ_n ≲ n^(-1/3)`` for the
  linearized heat map on a slab (Cor. 1: the pseudo-inverse amplifies noise by ``1/σ_n``); NeTMY
  Lemma 1 gives an exponential envelope ``e^(-k z0)`` for the dipolar kernels.
* :func:`hessian_condition_number` — condition number of the Hessian of a scalar function of a
  few parameters (NeTMY App. E.8 step 4 / Fig. 5: ``κ_F2 = 931`` vs ``κ_F1 = 301,139`` on a
  2-parameter Gaussian ansatz; pair with :func:`~nefi.diagnostics.ansatz_objective`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import torch

from ..errors import ConfigError
from ._common import VJP, jvp, operator_fn, resolve_fields

log = logging.getLogger("nefi")


def lanczos_eigs(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    v0: torch.Tensor,
    k: int,
    n_iter: int,
    return_vectors: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Largest eigenvalues of a symmetric PSD operator by Lanczos with full reorthogonalization.

    Args:
        matvec: ``v -> A v`` for a symmetric positive semidefinite ``A``.
        v0: start vector (any nonzero tensor; its shape is the operator's domain shape).
        k: number of eigenvalues returned.
        n_iter: Krylov dimension (``>= k``; capped by ``v0.numel()``).
        return_vectors: also return the Ritz vectors, shape ``(k, *v0.shape)``.

    Returns:
        ``(eigenvalues descending (k,), vectors or None)``.
    """
    n = v0.numel()
    m = int(min(max(n_iter, k), n))
    q = v0.flatten().double()
    q = q / q.norm().clamp_min(1e-300)
    basis = [q]
    alphas: list[float] = []
    betas: list[float] = []
    for j in range(m):
        w = matvec(basis[j].view_as(v0).to(v0.dtype)).flatten().double()
        a = float(torch.dot(basis[j], w))
        alphas.append(a)
        Q = torch.stack(basis)
        for _ in range(2):  # full reorthogonalization, twice is enough
            w = w - Q.T @ (Q @ w)
        b = float(w.norm())
        if j == m - 1 or b <= 1e-12 * max(1.0, abs(a)):
            break
        betas.append(b)
        basis.append(w / b)
    T = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    if betas:
        off = torch.tensor(betas[: len(alphas) - 1], dtype=torch.float64)
        T = T + torch.diag(off, 1) + torch.diag(off, -1)
    evals, evecs = torch.linalg.eigh(T)
    order = torch.argsort(evals, descending=True)[:k]
    vals = evals[order]
    vecs = None
    if return_vectors:
        Q = torch.stack(basis[: len(alphas)])  # (m, n)
        vecs = (evecs[:, order].T @ Q).view(len(order), *v0.shape)
    return vals, vecs


def singular_values(
    problem: Any,
    k: int = 10,
    n_iter: int = 30,
    shape: Sequence[int] | None = None,
    *,
    field: str | None = None,
    fields: Any = None,
    progress: float = 1.0,
    use_mask: bool = True,
    seed: int = 0,
    jvp_mode: str = "auto",
    return_vectors: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Top-``k`` singular values of the linearized forward operator ``dF/dx`` (NeFTY Prop. 2).

    Runs ``n_iter`` Lanczos steps on ``JᵀJ`` where each product ``JᵀJ v`` is a jvp (forward mode,
    with double-backward / finite-difference fallbacks) followed by a vjp. The operator is
    linearized at the current field values (or ``fields``); other fields are held fixed.

    Args:
        problem: the inverse problem.
        k: number of singular values.
        n_iter: Lanczos iterations (Krylov dimension); the top values converge first.
        shape: grid shape (default native or the shape of ``fields``).
        field: field to differentiate with respect to (default primary).
        fields: optional linearization point.
        progress: annealing progress for the current field values.
        use_mask: restrict the output to observed entries.
        seed: seed of the random start vector.
        jvp_mode: see :func:`nefi.diagnostics._common.jvp`.
        return_vectors: also return the right singular vectors (field-shaped), e.g. to see which
            field patterns the measurement is blind to.

    Returns:
        Non-increasing tensor of ``k`` singular values (float64), optionally with vectors
        ``(k, *field_shape)``.
    """
    if k < 1:
        raise ConfigError("k must be >= 1")
    x_fields, dom = resolve_fields(problem, fields, shape, progress)
    name = field or problem.field.primary
    fn, x0, _ = operator_fn(problem, x_fields, name, dom, use_mask)
    vjp_op = VJP(fn, x0)
    state = {"mode": jvp_mode}

    def jtj(v: torch.Tensor) -> torch.Tensor:
        jv, state["mode"] = jvp(fn, x0, v, state["mode"])
        return vjp_op(jv)

    gen = torch.Generator().manual_seed(int(seed))
    v0 = torch.randn(tuple(x0.shape), generator=gen, dtype=torch.float64).to(x0)
    evals, vecs = lanczos_eigs(jtj, v0, k, n_iter, return_vectors)
    sv = evals.clamp_min(0.0).sqrt()
    if len(sv) < k:
        log.info("singular_values: Krylov space exhausted after %d vectors", len(sv))
    return (sv, vecs) if return_vectors else sv


def hessian_spectrum(
    fn: Callable[[torch.Tensor], torch.Tensor], params: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigenvalues (ascending) and the symmetrized Hessian of ``fn`` at ``params``.

    Uses :func:`torch.autograd.functional.hessian`; intended for low-dimensional parameter vectors
    (an analytic ansatz), not for network weights.
    """
    if torch.is_tensor(params):
        p = params if torch.is_floating_point(params) else params.double()
    else:  # python numbers: float64 for well-resolved large condition numbers
        p = torch.as_tensor(params, dtype=torch.float64)
    p = p.detach().flatten()
    H = torch.autograd.functional.hessian(lambda q: fn(q).reshape(()), p)
    H = 0.5 * (H + H.T)
    return torch.linalg.eigvalsh(H.double()), H


def hessian_condition_number(fn: Callable[[torch.Tensor], torch.Tensor], params: Any) -> float:
    """Condition number ``|λ|_max / |λ|_min`` of the Hessian of a scalar function (NeTMY Fig. 5).

    NeTMY App. E.8 step 4 evaluates the log-MSE loss on a Gaussian density ansatz
    ``ρ(r; A, σ) = A exp(−‖r‖²/2σ²)`` and finds ``κ_F2 = 931`` (a parabolic bowl) vs
    ``κ_F1 = 301,139`` (a degenerate valley along ``A²σ² = const``). Evaluate at a minimum; an
    indefinite Hessian (saddle) is reported with a warning. Use float64 parameters (and a
    float64 problem) for large condition numbers.

    Args:
        fn: ``params -> scalar`` (e.g. from :func:`~nefi.diagnostics.ansatz_objective`).
        params: parameter vector (tensor or sequence of floats).
    """
    evals, _ = hessian_spectrum(fn, params)
    if bool((evals < 0).any()) and bool((evals > 0).any()):
        log.warning("hessian_condition_number: indefinite Hessian (not at a minimum): %s", evals)
    a = evals.abs()
    lo = float(a.min())
    return float("inf") if lo == 0.0 else float(a.max()) / lo


__all__ = ["hessian_condition_number", "hessian_spectrum", "lanczos_eigs", "singular_values"]
