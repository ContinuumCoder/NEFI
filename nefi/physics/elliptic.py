"""Variable-coefficient elliptic PDEs ``−∇·(σ∇u) + κu = f`` with an implicit-function adjoint.

One reusable, differentiable steady-state solver powers the *elliptic family* of the physics zoo:

=============================  ==================  ============  =================  ===========
application                    σ (coefficient)     κ             u (state)          f (source)
=============================  ==================  ============  =================  ===========
electrical impedance tomog.    conductivity        —             electric potential current
Darcy flow (porous media)      permeability/visc.  —             pressure           well rates
steady-state heat conduction   conductivity        (convection)  temperature        heat source
diffuse optical tomography     diffusion D         absorption    fluence rate       light source
=============================  ==================  ============  =================  ===========

References: Calderón (1980); Cheney, Isaacson & Newell, *SIAM Rev.* 41 (1999) (EIT);
Bear, *Dynamics of Fluids in Porous Media* (1972) (Darcy); Arridge, *Inverse Problems* 15 (1999)
(DOT); Patankar, *Numerical Heat Transfer and Fluid Flow* (1980) and LeVeque, *Finite Difference
Methods for ODEs and PDEs* (2007) (finite volumes); Hestenes & Stiefel (1952), Saad, *Iterative
Methods for Sparse Linear Systems* (2003) (conjugate gradients); Giles & Pierce (2000), Plessix,
*Geophys. J. Int.* 167 (2006) (adjoint-state / implicit-function gradients).

Discretization (cell-centered finite volumes on a uniform Cartesian grid)
------------------------------------------------------------------------
The unknown lives on the trailing ``ndim = len(spacing)`` axes of a tensor (leading axes are
batch axes, e.g. several boundary drives). σ is cell-centered and strictly positive. For every
axis ``d`` with spacing ``h_d`` and every face ``i+½`` the *face conductance* is
``K_{i+½} = σ̄_{i+½} / h_d²`` with the harmonic mean ``σ̄ = 2σ_iσ_{i+1}/(σ_i+σ_{i+1})``
(NeFTY Prop. 1 / App. A.3: the unique discrete realization of flux continuity for piecewise-constant
coefficients, exact for series layers) or, for ablations, the arithmetic mean. Then::

    [A u]_i = Σ_d [ K_{i+½}(u_i − u_{i+1}) + K_{i−½}(u_i − u_{i−1}) ] + κ_i u_i  ≈ −∇·(σ∇u) + κu.

Boundary conditions per axis *and side* (``"dirichlet"`` | ``"neumann"`` | ``"periodic"``):

* ``"dirichlet"`` — homogeneous ``u = 0`` on the boundary face via the ghost value ``u_g = −u_0``
  (and ``σ_g = σ_0``): the boundary face conductance is ``2σ_0/h²`` (half-cell distance).
* ``"neumann"`` — zero flux: the boundary face conductance is 0. Non-homogeneous Neumann data
  (injected current / heat flux ``j`` through a boundary face) enter the right-hand side as the
  volumetric source ``j / h_d`` of the boundary cell (finite-volume flux balance).
* ``"periodic"`` — circular wrap (must be set on both sides).

With these conventions ``A`` is symmetric and positive semi-definite (NeFTY App. D.2 argument:
each face conductance appears symmetrically in the two rows it couples); it is positive definite
as soon as one Dirichlet face or a positive ``κ`` exists. Without Dirichlet faces and without κ
the constant vector spans the null space: :func:`solve_elliptic` then enforces the compatibility
condition by removing the mean of the right-hand side and pins the solution to zero mean (all cells
have the same volume, so the Euclidean projection is the physical one).

Derivatives (implicit-function theorem)
---------------------------------------
For ``A(σ, κ)u = b``, differentiating gives ``A du = db − (∂A/∂σ·dσ)u − dκ⊙u``.

*Reverse mode.* With the adjoint state ``λ`` solving ``Aᵀλ = ∂L/∂u`` (``A`` is symmetric, so the
*same* preconditioned CG is reused)::

    ∂L/∂b = λ,        ∂L/∂σ = −λᵀ (∂A/∂σ) u,        ∂L/∂κ = −λ ⊙ u.

Because ``λᵀA(σ)u = Σ_faces K_f(σ) (λ_{i+1}−λ_i)(u_{i+1}−u_i) + Σ_Dirichlet K_b λ_i u_i + Σ κλu``
is bilinear in ``(λ, u)``, ``∂L/∂σ`` scatters the per-face weights ``−(Δλ)(Δu)`` to the adjacent
cells with the face-mean partials (``∂σ̄/∂σ_i = 2σ_j²/(σ_i+σ_j)²`` for the harmonic mean).

*Forward mode.* One tangent solve from a cold start::

    u̇ = A⁻¹ ( ḃ − (∂A/∂σ·σ̇) u − κ̇ ⊙ u ),    (∂A/∂σ·σ̇) u = A_{K̇}u,  K̇ = (∂K/∂σ)·σ̇,

so the tangent is converged to the solver tolerance whatever warm start the primal solve used.

Memory is ``O(N)`` (no unrolled CG graph). The adjoint and tangent solves re-enter
:class:`ImplicitSolveFunction`, so the rules compose: double backward (``create_graph=True``,
``gradgradcheck``), forward mode (``torch.func.jvp``, ``torch.autograd.forward_ad``,
``gradcheck(check_forward_ad=True)``), ``torch.func.vmap`` (custom rule: batched inputs become a
leading batch axis of one batched solve) and hence ``jacrev``/``jacfwd``/``hessian``. In the
pure-Neumann case the same formulas hold with ``λ``/``u̇`` the zero-mean solutions of the projected
systems (``A1 = 0`` for every σ). For verification, ``grad_mode="autograd"`` differentiates the
unrolled CG iterations in reverse mode; forward-mode or ``torch.func`` inputs are rejected there
(``NotImplementedError``) because the early-stopped iterations would silently truncate tangents.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

from ..domain import Domain
from ..errors import ConfigError, ShapeError
from ..fields.heads import Head
from ..operators.base import Fields, Operator
from ..registry import register
from ..utils.tensor import resample, shape_tuple

log = logging.getLogger("nefi")

BC_KINDS = ("dirichlet", "neumann", "periodic")
FACE_MODES = ("harmonic", "arithmetic")
GRAD_MODES = ("ift", "autograd", "none")
PRECONDITIONERS = ("jacobi", "none")
NULLSPACE_MODES = ("auto", "constant", "none")

BCSpec = str | Sequence[str | Sequence[str]]
SpacingSpec = float | Sequence[float]

__all__ = [
    "BC_KINDS",
    "FACE_MODES",
    "AxisConductance",
    "CGInfo",
    "EllipticOperator",
    "EllipticSolveConfig",
    "ImplicitSolveFunction",
    "LogBounded",
    "apply_conductances",
    "apply_divgrad",
    "conjugate_gradient",
    "divgrad_diagonal",
    "face_conductances",
    "face_mean",
    "is_transformed",
    "masked_mean",
    "normalize_bc",
    "point_source_rhs",
    "solve_elliptic",
    "weighted_area_downsample",
]


# --------------------------------------------------------------------------------------------
# boundary conditions, spacing, face means
# --------------------------------------------------------------------------------------------
def normalize_bc(bc: BCSpec, ndim: int) -> tuple[tuple[str, str], ...]:
    """Normalize a boundary-condition spec to ``((lo, hi), ...)``, one pair per axis.

    Accepted forms: one kind for every axis (``"dirichlet"``), one kind per axis
    (``["periodic", "neumann"]``) or per-side pairs (``[("dirichlet", "neumann"), "periodic"]``).
    Kinds: ``"dirichlet"`` (homogeneous), ``"neumann"`` (zero flux), ``"periodic"`` (both sides).

    Raises:
        ConfigError: unknown kind, wrong number of axes, or one-sided periodicity.
    """
    if isinstance(bc, str):
        entries: list[Any] = [bc] * ndim
    else:
        entries = list(bc)
        if len(entries) == 2 and ndim == 1 and all(isinstance(e, str) for e in entries):
            entries = [tuple(entries)]  # 1-D shorthand ("dirichlet", "neumann")
        if len(entries) != ndim:
            raise ConfigError(f"boundary spec {bc!r} has {len(entries)} axes, grid has {ndim}")
    out = []
    for ax, entry in enumerate(entries):
        if isinstance(entry, str):
            lo = hi = entry
        else:
            pair = list(entry)
            if len(pair) != 2:
                raise ConfigError(
                    f"axis {ax}: a per-side boundary spec needs (lo, hi), got {entry}"
                )
            lo, hi = pair
        lo, hi = str(lo).lower(), str(hi).lower()
        for kind in (lo, hi):
            if kind not in BC_KINDS:
                raise ConfigError(
                    f"unknown boundary condition {kind!r} on axis {ax}; use {BC_KINDS}"
                )
        if (lo == "periodic") != (hi == "periodic"):
            raise ConfigError(f"axis {ax}: 'periodic' must be set on both sides, got {(lo, hi)}")
        out.append((lo, hi))
    return tuple(out)


def _is_normalized(bc: Any, ndim: int) -> bool:
    """Fast check for the output format of :func:`normalize_bc` (hot path of the CG loop)."""
    return (
        isinstance(bc, tuple)
        and len(bc) == ndim
        and all(isinstance(e, tuple) and len(e) == 2 and e[0] in BC_KINDS for e in bc)
    )


def _infer_ndim(spacing: SpacingSpec, bc: BCSpec | None, ref: torch.Tensor) -> int:
    if not isinstance(spacing, int | float):
        return len(tuple(spacing))
    if bc is not None and not isinstance(bc, str):
        return len(list(bc))
    return ref.ndim


def _spacing(spacing: SpacingSpec, ndim: int) -> tuple[float, ...]:
    sp = (
        (float(spacing),) * ndim if isinstance(spacing, int | float) else tuple(map(float, spacing))
    )
    if len(sp) != ndim:
        raise ShapeError(f"spacing {sp} has {len(sp)} entries, grid has {ndim} axes")
    if any(not s > 0 for s in sp):
        raise ConfigError(f"grid spacing must be positive, got {sp}")
    return sp


def face_mean(a: torch.Tensor, b: torch.Tensor, mode: str = "harmonic") -> torch.Tensor:
    """Face coefficient from the two adjacent cell values.

    ``"harmonic"``: ``2ab/(a+b)`` — exact effective conductivity of two half-cells in series
    (NeFTY Prop. 1, App. A.3–A.4; Patankar 1980 §4.2-3); dominated by the smaller value, so
    insulating cells throttle the flux. ``"arithmetic"``: ``(a+b)/2`` — first-order at jumps and
    overestimates the flux across high-contrast interfaces (ablation only).
    """
    if mode == "harmonic":
        return 2.0 * a * b / (a + b)
    if mode == "arithmetic":
        return 0.5 * (a + b)
    raise ConfigError(f"unknown face_mode {mode!r}; use {FACE_MODES}")


def _pad_axis(x: torch.Tensor, dim: int, before: int, after: int) -> torch.Tensor:
    """Zero-pad ``x`` along the negative axis index ``dim``."""
    k = -dim
    pad = [0, 0] * k
    pad[2 * (k - 1)] = before
    pad[2 * (k - 1) + 1] = after
    return F.pad(x, pad)


def _assemble(
    interior: torch.Tensor, lo: torch.Tensor | None, hi: torch.Tensor | None, dim: int
) -> torch.Tensor:
    """Concatenate ``[lo, interior, hi]`` along ``dim`` (zeros where a side is ``None``)."""
    if lo is None and hi is None:
        return _pad_axis(interior, dim, 1, 1)
    x = _pad_axis(interior, dim, 1, 0) if lo is None else torch.cat([lo, interior], dim)
    return _pad_axis(x, dim, 0, 1) if hi is None else torch.cat([x, hi], dim)


def _sum_to_shape(t: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Reduce a broadcast result ``t`` back to ``shape`` (inverse of broadcasting)."""
    shape = tuple(shape)
    if tuple(t.shape) == shape:
        return t
    lead = t.ndim - len(shape)
    if lead > 0:
        t = t.sum(dim=tuple(range(lead)))
    dims = tuple(i for i, (a, b) in enumerate(zip(t.shape, shape)) if b == 1 and a != 1)
    if dims:
        t = t.sum(dim=dims, keepdim=True)
    return t.reshape(shape)


# --------------------------------------------------------------------------------------------
# the matrix-free operator A(σ) = −∇·(σ∇·) + κ
# --------------------------------------------------------------------------------------------
@dataclass
class AxisConductance:
    """Face conductances of one axis.

    Attributes:
        interior: ``K_{i+½} = σ̄_{i+½}/h²`` for the ``n−1`` interior faces (``n`` with the
            wrap-around face on periodic axes).
        lo / hi: Dirichlet boundary conductances ``2σ_b/h²`` (slices of length 1 along the axis),
            ``None`` for Neumann or periodic sides.
    """

    interior: torch.Tensor
    lo: torch.Tensor | None = None
    hi: torch.Tensor | None = None


def face_conductances(
    sigma: torch.Tensor,
    spacing: SpacingSpec,
    bc: BCSpec = "dirichlet",
    face_mode: str = "harmonic",
    ndim: int | None = None,
) -> list[AxisConductance]:
    """Per-axis face conductances ``σ̄/h²`` of the finite-volume operator (see module docstring).

    Args:
        sigma: cell-centered coefficient ``(*batch, *grid)`` (strictly positive).
        spacing: physical cell size (scalar or one per axis).
        bc: boundary spec (:func:`normalize_bc`).
        face_mode: ``"harmonic"`` (default, NeFTY Prop. 1) or ``"arithmetic"``.
        ndim: number of trailing grid axes (default: ``len(spacing)``, ``len(bc)`` or
            ``sigma.ndim``).
    """
    ndim = _infer_ndim(spacing, bc, sigma) if ndim is None else ndim
    sp = _spacing(spacing, ndim)
    bcn = normalize_bc(bc, ndim)
    out = []
    for ax in range(ndim):
        dim = ax - ndim
        n = sigma.shape[dim]
        h2 = sp[ax] ** 2
        lo_kind, hi_kind = bcn[ax]
        if lo_kind == "periodic":
            interior = face_mean(sigma, torch.roll(sigma, -1, dim), face_mode) / h2
        else:
            interior = (
                face_mean(sigma.narrow(dim, 0, n - 1), sigma.narrow(dim, 1, n - 1), face_mode) / h2
            )
        lo = 2.0 * sigma.narrow(dim, 0, 1) / h2 if lo_kind == "dirichlet" else None
        hi = 2.0 * sigma.narrow(dim, n - 1, 1) / h2 if hi_kind == "dirichlet" else None
        out.append(AxisConductance(interior, lo, hi))
    return out


def apply_conductances(
    u: torch.Tensor,
    conds: Sequence[AxisConductance],
    bc: BCSpec,
    kappa: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """``A u`` for precomputed face conductances (used inside the CG loop)."""
    ndim = len(conds)
    bcn = bc if _is_normalized(bc, ndim) else normalize_bc(bc, ndim)
    out = None if kappa is None else kappa * u
    for ax, c in enumerate(conds):
        dim = ax - ndim
        n = u.shape[dim]
        if bcn[ax][0] == "periodic":
            if n == 1:
                continue
            flux = c.interior * (torch.roll(u, -1, dim) - u)  # F_{i+½} = K (u_{i+1} − u_i)
            div = flux - torch.roll(flux, 1, dim)
        else:
            flux = c.interior * (u.narrow(dim, 1, n - 1) - u.narrow(dim, 0, n - 1))
            lo = None if c.lo is None else c.lo * u.narrow(dim, 0, 1)  # F_{−½} = K_b (u_0 − 0)
            hi = None if c.hi is None else -c.hi * u.narrow(dim, n - 1, 1)
            faces = _assemble(flux, lo, hi, dim)  # n + 1 faces
            div = faces.narrow(dim, 1, n) - faces.narrow(dim, 0, n)
        out = -div if out is None else out - div
    if out is None:  # only degenerate periodic axes of length 1 and no κ
        out = torch.zeros_like(u)
    return out


def apply_divgrad(
    u: torch.Tensor,
    sigma: torch.Tensor,
    spacing: SpacingSpec,
    bc: BCSpec = "dirichlet",
    face_mode: str = "harmonic",
    kappa: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Matrix-free ``−∇·(σ∇u) + κu`` on a uniform 1-/2-/3-D Cartesian grid.

    Pure tensor operations (``narrow``/``roll``/``pad``), hence autodiff-compatible in ``u``,
    ``σ`` and ``κ`` and device-agnostic. With constant ``σ = 1`` and Dirichlet/periodic boundaries
    this is exactly the (2d+1)-point negative Laplacian.

    Args:
        u: state ``(*batch, *grid)``; ``grid`` are the trailing ``len(spacing)`` axes.
        sigma: cell-centered coefficient broadcastable to ``u`` (typically ``(*grid)``), > 0.
        spacing: physical cell size (scalar or one per axis).
        bc: boundary spec, see :func:`normalize_bc`.
        face_mode: ``"harmonic"`` (NeFTY Prop. 1) or ``"arithmetic"``.
        kappa: optional zeroth-order coefficient (DOT absorption μ_a, Robin-like sinks), ≥ 0.

    Returns:
        ``A u`` with the broadcast shape of ``u``, ``σ`` (and ``κ``).
    """
    ndim = _infer_ndim(spacing, bc, sigma)
    _check_grid(u, sigma, ndim)
    conds = face_conductances(sigma, spacing, bc, face_mode, ndim)
    return apply_conductances(u, conds, bc, kappa)


def _diag_from_conductances(
    ref: torch.Tensor,
    conds: Sequence[AxisConductance],
    bc: BCSpec,
    kappa: torch.Tensor | float | None,
) -> torch.Tensor:
    ndim = len(conds)
    bcn = normalize_bc(bc, ndim)
    diag = torch.zeros_like(ref) if kappa is None else kappa + torch.zeros_like(ref)
    for ax, c in enumerate(conds):
        dim = ax - ndim
        n = ref.shape[dim]
        if bcn[ax][0] == "periodic":
            if n > 1:
                diag = diag + c.interior + torch.roll(c.interior, 1, dim)
        else:
            k = _assemble(c.interior, c.lo, c.hi, dim)
            diag = diag + k.narrow(dim, 1, n) + k.narrow(dim, 0, n)
    return diag


def divgrad_diagonal(
    sigma: torch.Tensor,
    spacing: SpacingSpec,
    bc: BCSpec = "dirichlet",
    face_mode: str = "harmonic",
    kappa: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Diagonal of ``A(σ)`` (the Jacobi preconditioner), shape of ``σ`` (broadcast with ``κ``)."""
    ndim = _infer_ndim(spacing, bc, sigma)
    conds = face_conductances(sigma, spacing, bc, face_mode, ndim)
    return _diag_from_conductances(sigma, conds, bc, kappa)


def _check_grid(u: torch.Tensor, sigma: torch.Tensor, ndim: int) -> None:
    if u.ndim < ndim or sigma.ndim < ndim:
        raise ShapeError(
            f"u {tuple(u.shape)} and sigma {tuple(sigma.shape)} need >= {ndim} grid axes"
        )
    if tuple(u.shape[-ndim:]) != tuple(sigma.shape[-ndim:]):
        raise ShapeError(
            f"grid mismatch: u has grid {tuple(u.shape[-ndim:])}, sigma has "
            f"{tuple(sigma.shape[-ndim:])}"
        )


# --------------------------------------------------------------------------------------------
# preconditioned conjugate gradients (batched, matrix-free)
# --------------------------------------------------------------------------------------------
@dataclass
class CGInfo:
    """Convergence report of :func:`conjugate_gradient`.

    Attributes:
        iterations: CG iterations performed.
        residual: max over the batch of the relative (recursively updated) residual
            ``‖r‖/‖b‖``.
        converged: every batch element reached ``‖r‖ ≤ max(tol‖b‖, atol)``.
        tol: effective relative tolerance (clamped to the dtype's floor).
    """

    iterations: int
    residual: float
    converged: bool
    tol: float


def conjugate_gradient(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    ndim: int,
    *,
    x0: torch.Tensor | None = None,
    precond: Callable[[torch.Tensor], torch.Tensor] | None = None,
    project: Callable[[torch.Tensor], torch.Tensor] | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    max_iter: int = 1000,
    check_every: int = 1,
) -> tuple[torch.Tensor, CGInfo]:
    """Batched preconditioned conjugate gradients for SPD ``A`` (Hestenes & Stiefel 1952).

    Every leading (batch) index is an independent system with its own step sizes; inner products
    reduce over the trailing ``ndim`` axes. All updates are out-of-place, so the loop can be
    unrolled under autograd (``grad_mode="autograd"`` reference gradients).

    Args:
        matvec: ``x -> A x`` (same shape as ``b``).
        b: right-hand side ``(*batch, *grid)``.
        ndim: number of trailing grid axes.
        x0: initial guess (warm start); per batch element it is only used if its residual is
            smaller than ``‖b‖``.
        precond: ``r -> M⁻¹ r`` (e.g. Jacobi ``r / diag(A)``).
        project: projector onto the range of a singular ``A`` (pure Neumann: remove the mean);
            applied to ``b``, ``x0``, the initial residual and every preconditioned residual
            (so the iterates stay in the range; ``A p`` is in the range by construction).
        tol: relative tolerance on ``‖b − Ax‖₂ / ‖b‖₂`` (clamped below by ``8·eps(dtype)``).
        atol: absolute tolerance.
        max_iter: iteration cap.
        check_every: check convergence (one host sync) every this many iterations (useful on CUDA).

    Returns:
        ``(x, info)``.
    """
    dims = tuple(range(-ndim, 0))

    def dot(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return (a * c).sum(dim=dims, keepdim=True)

    tol_eff = max(float(tol), 8.0 * torch.finfo(b.dtype).eps)
    if project is not None:
        b = project(b)
    bnorm = dot(b, b).sqrt()
    thresh = torch.clamp(tol_eff * bnorm, min=float(atol))
    if x0 is None:
        x = torch.zeros_like(b)
        r = b
    else:
        x = x0.to(device=b.device, dtype=b.dtype).expand_as(b)
        if project is not None:
            x = project(x)
        r = b - matvec(x)
        if project is not None:
            r = project(r)
        use = (dot(r, r) < bnorm**2).to(b.dtype)  # fall back to zero where x0 is worse
        x = use * x
        r = use * r + (1.0 - use) * b
    z = precond(r) if precond is not None else r
    if project is not None:
        z = project(z)
    p = z
    rz = dot(r, z)
    rnorm = dot(r, r).sqrt()
    it = 0
    tiny = torch.finfo(b.dtype).tiny
    while it < max_iter:
        if it % max(1, check_every) == 0 and bool((rnorm <= thresh).all()):
            break
        # Converged batch entries keep iterating harmlessly (their steps shrink with the
        # residual); the clamped divisions keep exhausted directions (p = 0) NaN-free.
        ap = matvec(p)  # in the range of A, so r keeps its zero mean (up to round-off)
        alpha = rz / dot(p, ap).clamp_min(tiny)
        x = x + alpha * p
        r = r - alpha * ap
        z = precond(r) if precond is not None else r
        if project is not None:
            z = project(z)
        rz_new = dot(r, z)
        p = z + (rz_new / rz.clamp_min(tiny)) * p
        rz = rz_new
        rnorm = dot(r, r).sqrt()
        it += 1
    with torch.no_grad():
        rel = (rnorm / bnorm.clamp_min(torch.finfo(b.dtype).tiny)).where(bnorm > 0, rnorm)
        converged = bool((rnorm <= thresh).all())
        info = CGInfo(it, float(rel.max()) if rel.numel() else 0.0, converged, tol_eff)
    return x, info


# --------------------------------------------------------------------------------------------
# solve configuration + the implicit-function-theorem autograd Function
# --------------------------------------------------------------------------------------------
@dataclass
class EllipticSolveConfig:
    """Static settings of an elliptic solve (non-tensor argument of :class:`ImplicitSolveFunction`).

    Attributes:
        spacing: physical cell size per axis.
        bc: normalized boundary spec.
        face_mode: face-coefficient mean.
        tol / atol / max_iter / check_every: CG settings (``max_iter=None`` → ``50·max(n)+200``).
        precond: ``"jacobi"`` or ``"none"``.
        nullspace: ``"auto"`` (constant null space iff no Dirichlet face and no κ), ``"constant"``
            or ``"none"``.
        adjoint_cache: optional dict receiving the :class:`CGInfo` of the latest adjoint or
            tangent solve (``"info"``) and the previous adjoint state (``"lam"``, used as warm
            start unless ``"warm"`` is false).
        last_info / last_adjoint_info: diagnostics of the latest forward / auxiliary solve.
    """

    spacing: tuple[float, ...]
    bc: tuple[tuple[str, str], ...]
    face_mode: str = "harmonic"
    tol: float = 1e-8
    atol: float = 0.0
    max_iter: int | None = None
    precond: str = "jacobi"
    nullspace: str = "auto"
    check_every: int = 1
    adjoint_cache: dict | None = None
    last_info: CGInfo | None = field(default=None, repr=False)
    last_adjoint_info: CGInfo | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.face_mode not in FACE_MODES:
            raise ConfigError(f"unknown face_mode {self.face_mode!r}; use {FACE_MODES}")
        if self.precond not in PRECONDITIONERS:
            raise ConfigError(f"unknown precond {self.precond!r}; use {PRECONDITIONERS}")
        if self.nullspace not in NULLSPACE_MODES:
            raise ConfigError(f"unknown nullspace {self.nullspace!r}; use {NULLSPACE_MODES}")

    @property
    def ndim(self) -> int:
        return len(self.spacing)

    @property
    def has_dirichlet(self) -> bool:
        return any("dirichlet" in pair for pair in self.bc)

    def singular(self, kappa: torch.Tensor | float | None) -> bool:
        if self.nullspace == "constant":
            return True
        if self.nullspace == "none":
            return False
        return not self.has_dirichlet and kappa is None

    def iteration_cap(self, grid: Sequence[int]) -> int:
        return int(self.max_iter) if self.max_iter else 50 * max(grid) + 200

    def derived(self) -> EllipticSolveConfig:
        """Copy for an auxiliary solve (adjoint / tangent) with its own diagnostics, no cache."""
        return dataclasses.replace(self, adjoint_cache=None, last_info=None, last_adjoint_info=None)


def _zero_mean_projector(ndim: int) -> Callable[[torch.Tensor], torch.Tensor]:
    dims = tuple(range(-ndim, 0))

    def project(t: torch.Tensor) -> torch.Tensor:
        return t - t.mean(dim=dims, keepdim=True)

    return project


_WARNED: set[str] = set()


def _solve_raw(
    sigma: torch.Tensor,
    rhs: torch.Tensor,
    kappa: torch.Tensor | None,
    cfg: EllipticSolveConfig,
    x0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, CGInfo]:
    """PCG solve of ``A(σ)u = b`` under the ambient grad mode (differentiable if enabled)."""
    ndim = cfg.ndim
    _check_grid(rhs, sigma, ndim)
    shapes = [rhs.shape, sigma.shape] + ([kappa.shape] if torch.is_tensor(kappa) else [])
    out_shape = torch.broadcast_shapes(*shapes)
    b = rhs if tuple(rhs.shape) == tuple(out_shape) else rhs.expand(out_shape)
    conds = face_conductances(sigma, cfg.spacing, cfg.bc, cfg.face_mode, ndim)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        return apply_conductances(v, conds, cfg.bc, kappa)

    precond = None
    if cfg.precond == "jacobi":
        diag = _diag_from_conductances(sigma, conds, cfg.bc, kappa)
        inv = torch.where(diag > 0, 1.0 / torch.where(diag > 0, diag, 1.0), 0.0)

        def precond(r: torch.Tensor) -> torch.Tensor:
            return r * inv

    project = _zero_mean_projector(ndim) if cfg.singular(kappa) else None
    grid = tuple(out_shape[-ndim:])
    x, info = conjugate_gradient(
        matvec,
        b,
        ndim,
        x0=x0 if (x0 is not None and tuple(x0.shape[-ndim:]) == grid) else None,
        precond=precond,
        project=project,
        tol=cfg.tol,
        atol=cfg.atol,
        max_iter=cfg.iteration_cap(grid),
        check_every=cfg.check_every,
    )
    if not info.converged:
        key = f"{grid}-{b.dtype}"
        level = logging.DEBUG if key in _WARNED else logging.WARNING
        _WARNED.add(key)
        log.log(
            level,
            "elliptic CG did not converge on grid %s (%s): residual %.2e > tol %.1e after %d "
            "iterations; increase max_iter or use a better preconditioner",
            grid,
            b.dtype,
            info.residual,
            info.tol,
            info.iterations,
        )
    return x, info


def is_transformed(t: Any) -> bool:
    """True for forward-mode dual tensors and tensors wrapped by ``torch.func`` transforms.

    Such inputs carry derivative information that the plain (no-grad / unrolled) CG loop would
    propagate only through its early-stopped iterations — e.g. a warm start at the solution stops
    at iteration 0 and silently returns a zero tangent — so they are routed to the exact rules of
    :class:`ImplicitSolveFunction` (or rejected, see :func:`solve_elliptic`).
    """
    if not torch.is_tensor(t):
        return False
    try:
        if torch._C._functorch.is_functorch_wrapped_tensor(t):
            return True
    except (AttributeError, RuntimeError):  # pragma: no cover - torch without the functorch API
        pass
    try:
        return fwAD.unpack_dual(t).tangent is not None
    except RuntimeError:  # pragma: no cover
        return False


def _face_pairs(
    x: torch.Tensor, dim: int, n: int, periodic: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Values on the two sides of every interior face along ``dim`` (the face-mean arguments)."""
    if periodic:
        return x, torch.roll(x, -1, dim)
    return x.narrow(dim, 0, n - 1), x.narrow(dim, 1, n - 1)


def _face_mean_partials(
    a: torch.Tensor, b: torch.Tensor, mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(∂σ̄/∂a, ∂σ̄/∂b)`` of :func:`face_mean` (harmonic: ``2b²/(a+b)², 2a²/(a+b)²``)."""
    if mode == "harmonic":
        s2 = (a + b) ** 2
        return 2.0 * b * b / s2, 2.0 * a * a / s2
    if mode == "arithmetic":
        return torch.full_like(a, 0.5), torch.full_like(b, 0.5)
    raise ConfigError(f"unknown face_mode {mode!r}; use {FACE_MODES}")


def _conductance_jvp(
    sigma: torch.Tensor, sigma_dot: torch.Tensor, cfg: EllipticSolveConfig
) -> list[AxisConductance]:
    """Directional derivatives ``K̇ = (∂K/∂σ)·σ̇`` of the face conductances (explicit formulas).

    ``A(σ)u`` is linear in the conductances, so ``(∂A/∂σ·σ̇) u = apply_conductances(u, K̇)``.
    """
    out = []
    for ax in range(cfg.ndim):
        dim = ax - cfg.ndim
        n = sigma.shape[dim]
        h2 = cfg.spacing[ax] ** 2
        lo_kind, hi_kind = cfg.bc[ax]
        periodic = lo_kind == "periodic"
        a, b = _face_pairs(sigma, dim, n, periodic)
        at, bt = _face_pairs(sigma_dot, dim, n, periodic)
        da, db = _face_mean_partials(a, b, cfg.face_mode)
        lo = 2.0 * sigma_dot.narrow(dim, 0, 1) / h2 if lo_kind == "dirichlet" else None
        hi = 2.0 * sigma_dot.narrow(dim, n - 1, 1) / h2 if hi_kind == "dirichlet" else None
        out.append(AxisConductance((da * at + db * bt) / h2, lo, hi))
    return out


def _conductance_vjp(
    sigma: torch.Tensor, u: torch.Tensor, lam: torch.Tensor, cfg: EllipticSolveConfig
) -> torch.Tensor:
    """``∂/∂σ ⟨λ, A(σ)u⟩`` with ``λ, u`` fixed — the adjoint of :func:`_conductance_jvp`.

    ``⟨λ, A(σ)u⟩ = Σ_faces K_f(σ)(Δλ)(Δu) + Σ_Dirichlet K_b(σ) λ u (+ κ-term)`` is bilinear in
    ``(λ, u)``, so the per-face weights ``(Δλ)(Δu)`` (summed over drives) are scattered back to the
    two adjacent cells with the face-mean partials. Plain tensor algebra: differentiable (double
    backward) and valid under every ``torch.func`` transform.
    """
    g = torch.zeros_like(sigma)
    for ax in range(cfg.ndim):
        dim = ax - cfg.ndim
        n = u.shape[dim]
        h2 = cfg.spacing[ax] ** 2
        lo_kind, hi_kind = cfg.bc[ax]
        periodic = lo_kind == "periodic"
        if periodic and n == 1:
            continue
        la, lb = _face_pairs(lam, dim, n, periodic)
        ua, ub = _face_pairs(u, dim, n, periodic)
        a, b = _face_pairs(sigma, dim, n, periodic)
        da, db = _face_mean_partials(a, b, cfg.face_mode)
        w = _sum_to_shape((lb - la) * (ub - ua), a.shape) / h2
        if periodic:
            g = g + w * da + torch.roll(w * db, 1, dim)
        else:
            g = g + _pad_axis(w * da, dim, 0, 1) + _pad_axis(w * db, dim, 1, 0)
        edge = sigma.narrow(dim, 0, 1).shape
        if lo_kind == "dirichlet":
            wl = _sum_to_shape(lam.narrow(dim, 0, 1) * u.narrow(dim, 0, 1), edge)
            g = g + _pad_axis(2.0 * wl / h2, dim, 0, n - 1)
        if hi_kind == "dirichlet":
            wh = _sum_to_shape(lam.narrow(dim, n - 1, 1) * u.narrow(dim, n - 1, 1), edge)
            g = g + _pad_axis(2.0 * wh / h2, dim, n - 1, 0)
    return g


class ImplicitSolveFunction(torch.autograd.Function):
    """``u = A(σ, κ)⁻¹ b`` with exact implicit-function derivative rules (O(N) memory).

    * **forward**: batched Jacobi-PCG solve (no autograd graph);
    * **backward** (reverse mode, VJP): one adjoint solve ``Aλ = ḡ`` with the same CG (``A`` is
      symmetric), then ``∂L/∂b = λ``, ``∂L/∂σ = −λᵀ(∂A/∂σ)u`` (explicit face-conductance adjoint)
      and ``∂L/∂κ = −λ⊙u``;
    * **jvp** (forward mode): one tangent solve ``u̇ = A⁻¹(ḃ − (∂A/∂σ·σ̇)u − κ̇⊙u)`` from a cold
      start, so the tangent is converged to the solver tolerance whatever the warm start of the
      primal solve;
    * **vmap**: batched inputs are moved to a leading batch axis and solved as one batched system
      (the CG loop has data-dependent control flow and cannot be vmapped op by op).

    The adjoint and tangent solves re-enter this Function, so every rule composes with the others:
    double backward, JVPs through the double-backward trick, ``torch.func.{jvp,vjp,grad,vmap,
    jacrev,jacfwd,hessian}`` and ``gradcheck(check_forward_ad=True)``.

    Call as ``ImplicitSolveFunction.apply(sigma, rhs, kappa_or_None, cfg, x0_or_None)``.
    """

    @staticmethod
    def forward(sigma, rhs, kappa, cfg: EllipticSolveConfig, x0=None):  # type: ignore[override]
        u, info = _solve_raw(
            sigma.detach(),
            rhs.detach(),
            None if kappa is None else kappa.detach(),
            cfg,
            None if x0 is None else x0.detach(),
        )
        cfg.last_info = info
        return u

    @staticmethod
    def setup_context(ctx, inputs, output):  # type: ignore[override]
        sigma, rhs, kappa, cfg, _x0 = inputs
        ctx.cfg = cfg
        ctx.rhs_shape = tuple(rhs.shape)
        ctx.save_for_backward(sigma, kappa, output)
        ctx.save_for_forward(sigma, kappa, output)

    @staticmethod
    def backward(ctx, grad_u):  # type: ignore[override]
        sigma, kappa, u = ctx.saved_tensors
        cfg: EllipticSolveConfig = ctx.cfg
        need_sigma, need_rhs, need_kappa = ctx.needs_input_grad[:3]
        cache = cfg.adjoint_cache
        warm = cache is not None and bool(cache.get("warm", True)) and not is_transformed(grad_u)
        x0 = None
        if warm:
            prev = cache.get("lam")
            if prev is not None and prev.shape == grad_u.shape and prev.dtype == grad_u.dtype:
                x0 = prev.to(grad_u.device)
        acfg = cfg.derived()
        # re-entering the Function keeps the adjoint solve differentiable (create_graph) and
        # batchable (vmap over cotangents, jacrev); without grad mode it records nothing
        lam = ImplicitSolveFunction.apply(sigma, grad_u, kappa, acfg, x0)
        cfg.last_adjoint_info = acfg.last_info
        if cache is not None:
            cache["info"] = acfg.last_info
            if warm and not is_transformed(lam):
                cache["lam"] = lam.detach()
        g_sigma = g_rhs = g_kappa = None
        if need_rhs:
            g_rhs = _sum_to_shape(lam, ctx.rhs_shape)
        if need_sigma:
            g_sigma = -_conductance_vjp(sigma, u, lam, cfg)
        if need_kappa:
            g_kappa = -_sum_to_shape(lam * u, kappa.shape)
        return g_sigma, g_rhs, g_kappa, None, None

    @staticmethod
    def jvp(ctx, sigma_dot, rhs_dot, kappa_dot, _cfg_dot, _x0_dot):  # type: ignore[override]
        sigma, kappa, u = ctx.saved_tensors
        cfg: EllipticSolveConfig = ctx.cfg
        r = rhs_dot
        if sigma_dot is not None:
            du = apply_conductances(u, _conductance_jvp(sigma, sigma_dot, cfg), cfg.bc)
            r = -du if r is None else r - du
        if kappa_dot is not None:
            dk = kappa_dot * u
            r = -dk if r is None else r - dk
        if r is None:
            return torch.zeros_like(u)
        tcfg = cfg.derived()
        u_dot = ImplicitSolveFunction.apply(sigma, r, kappa, tcfg, None)
        cfg.last_adjoint_info = tcfg.last_info  # diagnostics of the latest auxiliary solve
        if cfg.adjoint_cache is not None:
            cfg.adjoint_cache["info"] = tcfg.last_info
        return u_dot

    @staticmethod
    def vmap(info, in_dims, sigma, rhs, kappa, cfg, x0):  # type: ignore[override]
        items = ((sigma, in_dims[0]), (rhs, in_dims[1]), (kappa, in_dims[2]))
        per_sample = []
        for t, d in items:
            if t is None:
                per_sample.append(None)
                continue
            shp = list(t.shape)
            if d is not None:
                shp.pop(d)
            per_sample.append(tuple(shp))
        rank = len(torch.broadcast_shapes(*[p for p in per_sample if p is not None]))

        def lead(t, d, shp):
            if t is None:
                return None
            t = t.unsqueeze(0) if d is None else t.movedim(d, 0)
            return t.reshape((t.shape[0],) + (1,) * (rank - len(shp)) + shp)

        args = [lead(t, d, shp) for (t, d), shp in zip(items, per_sample)]
        u = ImplicitSolveFunction.apply(args[0], args[1], args[2], cfg, None)
        return u, 0


def solve_elliptic(
    sigma: torch.Tensor,
    rhs: torch.Tensor,
    spacing: SpacingSpec,
    bc: BCSpec = "dirichlet",
    *,
    kappa: torch.Tensor | float | None = None,
    x0: torch.Tensor | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    max_iter: int | None = None,
    precond: str = "jacobi",
    face_mode: str = "harmonic",
    nullspace: str = "auto",
    grad_mode: str = "ift",
    check_every: int = 1,
    adjoint_cache: dict | None = None,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, CGInfo]:
    """Solve ``−∇·(σ∇u) + κu = f`` (matrix-free Jacobi-PCG) for one or many right-hand sides.

    Args:
        sigma: cell-centered coefficient ``(*grid)`` or ``(*batch, *grid)``, strictly positive.
        rhs: source density ``f`` with the same grid; leading axes are independent drives.
        spacing: physical cell size (scalar or one per axis); ``len(spacing)`` sets the grid rank.
        bc: boundary spec (:func:`normalize_bc`). Non-homogeneous Neumann data belong in ``rhs``.
        kappa: optional zeroth-order coefficient ``κ ≥ 0`` (scalar or tensor, differentiable).
        x0: initial guess (warm start).
        tol / atol: relative / absolute residual tolerance (relative clamped to 8·eps(dtype)).
        max_iter: CG iteration cap (default ``50·max(grid) + 200``).
        precond: ``"jacobi"`` (default) or ``"none"``.
        face_mode: ``"harmonic"`` (NeFTY Prop. 1) or ``"arithmetic"``.
        nullspace: ``"auto"`` — for pure Neumann/periodic problems without κ the mean of ``rhs``
            is removed (compatibility condition) and ``u`` is pinned to zero mean;
            ``"constant"`` forces this, ``"none"`` disables it.
        grad_mode: ``"ift"`` — exact implicit-function rules (:class:`ImplicitSolveFunction`:
            adjoint VJP, forward-mode JVP, vmap; O(N) memory); ``"autograd"`` — differentiate the
            unrolled CG in reverse mode (reference/testing); ``"none"`` — no gradients. Forward-
            mode dual or ``torch.func``-wrapped inputs require ``"ift"`` (``NotImplementedError``
            otherwise, so callers such as ``nefi.diagnostics`` fall back to double backward).
        check_every: CG convergence-check period (host syncs).
        adjoint_cache: optional dict; the adjoint solve stores its :class:`CGInfo` under
            ``"info"`` and, unless ``cache["warm"]`` is false, warm-starts from / stores the
            previous adjoint state under ``"lam"``.
        return_info: also return the :class:`CGInfo` of the forward solve.

    Returns:
        ``u`` with the broadcast shape of ``rhs``/``sigma`` (and ``(u, info)`` if requested).

    Raises:
        NotImplementedError: forward-mode / ``torch.func`` inputs with ``grad_mode != "ift"``.
    """
    if grad_mode not in GRAD_MODES:
        raise ConfigError(f"unknown grad_mode {grad_mode!r}; use {GRAD_MODES}")
    ndim = _infer_ndim(spacing, bc, sigma)
    cfg = EllipticSolveConfig(
        _spacing(spacing, ndim),
        normalize_bc(bc, ndim),
        face_mode,
        float(tol),
        float(atol),
        max_iter,
        precond,
        nullspace,
        int(check_every),
        adjoint_cache,
    )
    if kappa is not None and not torch.is_tensor(kappa):
        kappa = torch.as_tensor(float(kappa), device=sigma.device, dtype=sigma.dtype)
    transformed = any(is_transformed(t) for t in (sigma, rhs, kappa))
    if transformed and grad_mode != "ift":
        raise NotImplementedError(
            f"solve_elliptic(grad_mode={grad_mode!r}) received forward-mode dual or torch.func-"
            "wrapped inputs; differentiating the early-stopped CG iterations in forward mode "
            "silently truncates tangents (a warm start at the solution returns zero). Use "
            "grad_mode='ift' (exact implicit JVP/VJP/vmap rules) or, in nefi.diagnostics, "
            "jvp_mode='double_backward'."
        )
    needs_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (sigma, rhs, kappa)
    )
    if grad_mode == "ift" and (needs_grad or transformed):
        u = ImplicitSolveFunction.apply(sigma, rhs, kappa, cfg, x0)
        info = cfg.last_info
    elif grad_mode == "autograd" and needs_grad:
        u, info = _solve_raw(sigma, rhs, kappa, cfg, None if x0 is None else x0.detach())
    else:
        with torch.no_grad():
            u, info = _solve_raw(sigma, rhs, kappa, cfg, x0)
    if return_info:
        return u, info  # type: ignore[return-value]
    return u


# --------------------------------------------------------------------------------------------
# helpers: sources, masked statistics, observation downsampling, coefficient head
# --------------------------------------------------------------------------------------------
def point_source_rhs(
    domain: Domain,
    positions: torch.Tensor | Sequence[Sequence[float]],
    rates: torch.Tensor | Sequence[float],
    *,
    spread: str = "linear",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Source density of point sources/sinks (wells, optodes, electrodes) on ``domain``'s grid.

    A point source of rate ``q`` at ``x_s`` is ``q δ(x − x_s)``; on the cell-centered grid it is
    distributed to the ``2^d`` nearest cell centers with multilinear (cloud-in-cell) weights (or
    to the containing cell with ``spread="nearest"``) and divided by the cell volume, so the
    discrete source integrates to ``q`` at every resolution and moves smoothly with ``x_s``.

    Args:
        domain: grid.
        positions: physical coordinates ``(n_sources, ndim)``.
        rates: ``(n_sources,)`` rates (positive = injection).
        spread: ``"linear"`` (cloud-in-cell) or ``"nearest"``.

    Returns:
        ``(*domain.shape)`` source density.
    """
    pos = torch.as_tensor(positions, dtype=dtype).reshape(-1, domain.ndim)
    q = torch.as_tensor(rates, dtype=dtype).reshape(-1)
    if pos.shape[0] != q.shape[0]:
        raise ShapeError(f"{pos.shape[0]} positions but {q.shape[0]} rates")
    shape = domain.shape
    sp = domain.spacing()
    vol = math.prod(sp)
    out = torch.zeros(shape, dtype=dtype)
    lo = torch.tensor([e[0] for e in domain.extent], dtype=dtype)
    h = torch.tensor(sp, dtype=dtype)
    t = (pos - lo) / h - 0.5  # continuous cell index of each source
    for s in range(pos.shape[0]):
        if spread == "nearest":
            idx = tuple(
                int(min(max(round(float(t[s, d])), 0), shape[d] - 1)) for d in range(domain.ndim)
            )
            out[idx] += q[s] / vol
            continue
        if spread != "linear":
            raise ConfigError(f"unknown spread {spread!r}; use 'linear' or 'nearest'")
        base = torch.floor(t[s])
        frac = t[s] - base
        for corner in range(2**domain.ndim):
            w = 1.0
            idx = []
            for d in range(domain.ndim):
                bit = (corner >> d) & 1
                w = w * (float(frac[d]) if bit else 1.0 - float(frac[d]))
                i = int(base[d]) + bit
                idx.append(min(max(i, 0), shape[d] - 1))  # clamp: mass stays in the domain
            if w > 0:
                out[tuple(idx)] += q[s] * w / vol
    return out


def masked_mean(
    x: torch.Tensor, weights: torch.Tensor, ndim: int, eps: float = 1e-30
) -> torch.Tensor:
    """Weighted mean over the trailing ``ndim`` axes (``keepdim``), weights broadcastable to x."""
    dims = tuple(range(-ndim, 0))
    w = weights.to(dtype=x.dtype, device=x.device)
    num = (x * w).sum(dim=dims, keepdim=True)
    den = (w + torch.zeros_like(x)).sum(dim=dims, keepdim=True).clamp_min(eps)
    return num / den


def weighted_area_downsample(
    data: torch.Tensor, weights: torch.Tensor, shape: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted area-average of sparse observations onto a coarser (or equal) grid.

    ``data_c = area(data·w) / area(w)`` where ``area(w) > 0`` (else 0) and ``w_c = area(w)``: each
    coarse cell receives the weighted mean of the *observed* fine cells it contains, never the
    (masked, possibly zeroed) unobserved values. Used for boundary-strip (EIT) and sparse-sensor
    (Darcy) observations at coarse curriculum stages and by the supersampled data generators.

    Args:
        data: ``(*batch, *grid)``.
        weights: non-negative observation weights broadcastable to ``data``.
        shape: target grid (trailing axes).

    Returns:
        ``(data_c, weights_c)`` with trailing shape ``shape``.
    """
    shape = shape_tuple(shape)
    w = (weights.to(data) + torch.zeros_like(data)).clamp_min(0.0)
    num = resample(data * w, shape, mode="area")
    den = resample(w, shape, mode="area")
    safe = torch.where(den > 0, den, torch.ones_like(den))
    return torch.where(den > 0, num / safe, torch.zeros_like(num)), den


@register("head", "log_bounded")
class LogBounded(Head):
    """Positive coefficient in ``[lo, hi]`` with a log-uniform parameterization.

    ``σ = exp(log lo + (log hi − log lo)·sigmoid(h))`` — the log-conductivity / log-permeability
    parameterization standard in EIT and subsurface flow (coefficients spanning decades), with
    the hard bracketing of NeFTY Eq. (6).
    """

    def __init__(self, lo: float, hi: float, init_value: float | None = None) -> None:
        super().__init__()
        if not (hi > lo > 0):
            raise ConfigError(f"LogBounded needs hi > lo > 0, got {(lo, hi)}")
        self.lo, self.hi = float(lo), float(hi)
        self.log_lo, self.log_hi = math.log(self.lo), math.log(self.hi)
        self.init_value = init_value

    def transform(self, raw, others):
        return torch.exp(self.log_lo + (self.log_hi - self.log_lo) * torch.sigmoid(raw[..., 0]))

    def init_bias(self):
        if self.init_value is None:
            return None
        return [self.inverse(torch.tensor(float(self.init_value))).item()]

    def inverse(self, value):
        v = torch.log(torch.as_tensor(value).clamp_min(1e-30))
        t = ((v - self.log_lo) / (self.log_hi - self.log_lo)).clamp(1e-6, 1 - 1e-6)
        return torch.logit(t).unsqueeze(-1)


# --------------------------------------------------------------------------------------------
# the operator
# --------------------------------------------------------------------------------------------
_TRANSFORMS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "identity": lambda x: x,
    "exp": torch.exp,
    "softplus": F.softplus,
}

TensorSpec = torch.Tensor | float | Callable[[Domain], torch.Tensor] | None


@register("operator", "elliptic")
class EllipticOperator(Operator):
    """Steady-state elliptic forward model ``{σ} ↦ u`` solving ``−∇·(σ∇u) + κu = f``.

    The hard physics constraint of EIT, Darcy flow, steady heat conduction and DOT
    (see module docstring). ``σ`` is produced by the field (optionally through
    ``sigma_transform``, e.g. ``"exp"`` for a log-conductivity / log-permeability field);
    ``f`` is fixed (sources and non-homogeneous Neumann drives, one per leading index). Gradients
    use the implicit-function adjoint (:class:`ImplicitSolveFunction`) by default.

    Args:
        domain: field domain (1-, 2- or 3-D).
        rhs: source density ``(*grid)`` or ``(n_drives, *grid)`` at ``domain.shape``, or a
            callable ``rhs(domain) -> Tensor`` evaluated at every curriculum resolution (preferred:
            exact re-derivation; tensors are area/linearly resampled).
        bc: boundary spec (:func:`normalize_bc`).
        field: name of the coefficient field.
        sigma_transform: ``"identity"`` | ``"exp"`` | ``"softplus"`` | callable, applied to the
            field before the solve.
        kappa: fixed zeroth-order coefficient (scalar, tensor or callable of the domain).
        kappa_field: name of a field supplying κ (unknown absorption, DOT); excludes ``kappa``.
        face_mode: ``"harmonic"`` (NeFTY Prop. 1) or ``"arithmetic"``.
        grad_mode: ``"ift"`` (default) or ``"autograd"`` (unrolled CG, for tests).
        tol / atol / max_iter / precond / nullspace / check_every: see :func:`solve_elliptic`.
        warm_start: reuse the previous forward/adjoint solutions as CG initial guesses (large
            speed-up inside optimization loops; results then depend on the history only within
            the solver tolerance).
        reference: optional non-negative weights ``(*grid)`` / ``(n_drives, *grid)`` (or callable
            of the domain); the output is referenced to their weighted mean per drive (ground
            electrode / gauge reference), which removes the solution's additive-constant
            convention from the observable.

    Output: ``u`` of shape ``(*rhs_batch, *grid)``. Scaling: for ``κ = 0`` and the identity
    transform ``u(cσ) = u(σ)/c`` (degree −1); ``homogeneity`` is left ``None`` because
    energy-anchored scale correction does not apply to coefficient problems.
    """

    homogeneity = None
    fidelity_tag = "elliptic-fv"
    traceable = False  # PCG with host-side convergence tests + the IFT autograd.Function

    def __init__(
        self,
        domain: Domain,
        rhs: torch.Tensor | Callable[[Domain], torch.Tensor],
        bc: BCSpec = "dirichlet",
        *,
        field: str = "sigma",
        sigma_transform: str | Callable[[torch.Tensor], torch.Tensor] = "identity",
        kappa: TensorSpec = None,
        kappa_field: str | None = None,
        face_mode: str = "harmonic",
        grad_mode: str = "ift",
        tol: float = 1e-8,
        atol: float = 0.0,
        max_iter: int | None = None,
        precond: str = "jacobi",
        nullspace: str = "auto",
        check_every: int = 1,
        warm_start: bool = False,
        reference: TensorSpec = None,
    ) -> None:
        super().__init__()
        self.domain = domain
        self.primary = field
        self.bc = normalize_bc(bc, domain.ndim)
        if isinstance(sigma_transform, str) and sigma_transform not in _TRANSFORMS:
            raise ConfigError(
                f"unknown sigma_transform {sigma_transform!r}; use {list(_TRANSFORMS)}"
            )
        if grad_mode not in GRAD_MODES:
            raise ConfigError(f"unknown grad_mode {grad_mode!r}; use {GRAD_MODES}")
        if kappa is not None and kappa_field is not None:
            raise ConfigError("pass either a fixed kappa or kappa_field, not both")
        # validate solver settings early
        EllipticSolveConfig(
            domain.spacing(), self.bc, face_mode, tol, atol, max_iter, precond, nullspace
        )
        self.sigma_transform = sigma_transform
        self.kappa_field = kappa_field
        self.face_mode = face_mode
        self.grad_mode = grad_mode
        self.tol, self.atol, self.max_iter = float(tol), float(atol), max_iter
        self.precond, self.nullspace, self.check_every = precond, nullspace, int(check_every)
        self.warm_start = bool(warm_start)
        self._rhs_spec = rhs
        self._kappa_spec = kappa
        self._reference_spec = reference
        self._rhs = self._materialize(rhs, "rhs")
        self._kappa = None if kappa is None else self._materialize(kappa, "kappa")
        self._reference = None if reference is None else self._materialize(reference, "reference")
        self._cache: dict[tuple, torch.Tensor] = {}
        self._warm: dict[str, torch.Tensor] = {}
        self._adjoint_cache: dict[str, Any] = {"warm": self.warm_start}
        self._children: dict[tuple[int, ...], EllipticOperator] = {}
        self.last_info: CGInfo | None = None

    # --- construction helpers -----------------------------------------------------------
    def _materialize(self, spec: TensorSpec, name: str) -> torch.Tensor:
        if callable(spec) and not torch.is_tensor(spec):
            t = torch.as_tensor(spec(self.domain))
        else:
            t = torch.as_tensor(spec)
        if t.ndim == 0:
            return t.detach().clone()
        grid = self.domain.shape
        if t.ndim < len(grid) or tuple(t.shape[-len(grid) :]) != grid:
            raise ShapeError(
                f"{name} has shape {tuple(t.shape)}; its trailing axes must equal the domain grid "
                f"{grid}"
            )
        return t.detach().clone()

    def _resampled_spec(self, spec: TensorSpec, domain: Domain) -> TensorSpec:
        if spec is None or (callable(spec) and not torch.is_tensor(spec)):
            return spec
        t = torch.as_tensor(spec)
        if t.ndim == 0:
            return t
        return resample(t.to(torch.float64), domain.shape).to(t.dtype)

    def _rebuild(self, domain: Domain) -> EllipticOperator:
        """New operator of the same class on ``domain`` (override in subclasses)."""
        return EllipticOperator(
            domain,
            self._resampled_spec(self._rhs_spec, domain),
            self.bc,
            field=self.primary,
            sigma_transform=self.sigma_transform,
            kappa=self._resampled_spec(self._kappa_spec, domain),
            kappa_field=self.kappa_field,
            face_mode=self.face_mode,
            grad_mode=self.grad_mode,
            tol=self.tol,
            atol=self.atol,
            max_iter=self.max_iter,
            precond=self.precond,
            nullspace=self.nullspace,
            check_every=self.check_every,
            warm_start=self.warm_start,
            reference=self._resampled_spec(self._reference_spec, domain),
        )

    def _on_device(self, key: str, t: torch.Tensor, device, dtype) -> torch.Tensor:
        ck = (key, str(device), str(dtype))
        v = self._cache.get(ck)
        if v is None:
            v = t.to(device=device, dtype=dtype)
            if not is_transformed(v):  # never cache a torch.func wrapper (outlives the transform)
                self._cache[ck] = v
        return v

    # --- Operator API ---------------------------------------------------------------------
    @property
    def spacing(self) -> tuple[float, ...]:
        return self.domain.spacing()

    @property
    def n_drives(self) -> int | None:
        """Number of right-hand sides (``None`` for a single unbatched source)."""
        extra = self._rhs.ndim - self.domain.ndim
        return None if extra == 0 else int(math.prod(self._rhs.shape[:extra]))

    def rhs(self, device=None, dtype=None) -> torch.Tensor:
        """The source density at this operator's resolution."""
        return self._on_device("rhs", self._rhs, device, dtype or torch.get_default_dtype())

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``sigma_transform`` to a raw coefficient field."""
        fn = self.sigma_transform
        return _TRANSFORMS[fn](x) if isinstance(fn, str) else fn(x)

    def coefficient(self, fields: Fields) -> torch.Tensor:
        """σ (after the transform) from a field dict."""
        return self.transform(self.get_field(fields))

    def kappa_value(self, fields: Fields, ref: torch.Tensor) -> torch.Tensor | None:
        if self.kappa_field is not None:
            return self.get_field(fields, self.kappa_field)
        if self._kappa is None:
            return None
        return self._on_device("kappa", self._kappa, ref.device, ref.dtype)

    def reference_weights(self, device=None, dtype=None) -> torch.Tensor | None:
        if self._reference is None:
            return None
        return self._on_device("reference", self._reference, device, dtype)

    def solve(self, sigma: torch.Tensor, kappa: torch.Tensor | None = None) -> torch.Tensor:
        """Solve for the state ``u`` given the (transformed) coefficient ``σ``."""
        nd = self.domain.ndim
        if sigma.ndim < nd or tuple(sigma.shape[-nd:]) != self.domain.shape:
            raise ShapeError(
                f"{type(self).__name__} expects {self.primary!r} on grid {self.domain.shape}, got "
                f"{tuple(sigma.shape)}; use operator.at_resolution(shape) for other resolutions"
            )
        b = self.rhs(sigma.device, sigma.dtype)
        x0 = None
        if self.warm_start:
            prev = self._warm.get("u")
            if prev is not None and prev.dtype == sigma.dtype and prev.device == sigma.device:
                x0 = prev
        u, info = solve_elliptic(
            sigma,
            b,
            self.spacing,
            self.bc,
            kappa=kappa,
            x0=x0,
            tol=self.tol,
            atol=self.atol,
            max_iter=self.max_iter,
            precond=self.precond,
            face_mode=self.face_mode,
            nullspace=self.nullspace,
            grad_mode=self.grad_mode,
            check_every=self.check_every,
            adjoint_cache=self._adjoint_cache,
            return_info=True,
        )
        self.last_info = info
        if self.warm_start and not is_transformed(u):  # never keep torch.func wrappers alive
            self._warm["u"] = u.detach()
        return u

    def post(self, u: torch.Tensor, sigma: torch.Tensor, fields: Fields) -> torch.Tensor:
        """Map the state to the prediction (default: optional weighted-mean referencing)."""
        w = self.reference_weights(u.device, u.dtype)
        if w is None:
            return u
        return u - masked_mean(u, w, self.domain.ndim)

    def forward(self, fields: Fields) -> torch.Tensor:
        sigma = self.coefficient(fields)
        kappa = self.kappa_value(fields, sigma)
        u = self.solve(sigma, kappa)
        return self.post(u, sigma, fields)

    def at_resolution(self, shape: Sequence[int]) -> EllipticOperator:
        shape = shape_tuple(shape)
        if shape == self.domain.shape:
            return self
        child = self._children.get(shape)
        if child is None:
            child = self._rebuild(self.domain.at(shape))
            self._children[shape] = child
        return child

    def output_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        extra = tuple(self._rhs.shape[: self._rhs.ndim - self.domain.ndim])
        return extra + shape_tuple(shape)

    def required_fields(self) -> tuple[str, ...]:
        return (self.primary,) + ((self.kappa_field,) if self.kappa_field else ())

    @property
    def last_adjoint_info(self) -> CGInfo | None:
        """Convergence report of the latest auxiliary solve (adjoint or forward-mode tangent)."""
        return self._adjoint_cache.get("info")

    def reset_warm_start(self) -> None:
        """Forget cached forward/adjoint solutions (e.g. before a fresh restart)."""
        self._warm.clear()
        self._adjoint_cache.pop("lam", None)
        for c in self._children.values():
            c.reset_warm_start()

    def extra_repr(self) -> str:
        return (
            f"grid={self.domain.shape}, bc={self.bc}, drives={self.n_drives}, "
            f"face_mode={self.face_mode}, grad_mode={self.grad_mode}, tol={self.tol:g}"
        )
