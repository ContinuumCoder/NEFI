"""Finite-volume diffusion stencil ``L(α)`` with harmonic-mean face coefficients.

Implements the spatial discretization of NeFTY §4.2 / App. D.2 on an N-D Cartesian grid::

    [L(α)T]_i = Σ_d Δ_d⁻² ( ᾱ_{i+1/2} (T_{i+1} − T_i) − ᾱ_{i−1/2} (T_i − T_{i−1}) ) − β_i T_i

where ``ᾱ`` is the harmonic mean of the two neighbouring cell values (NeFTY Prop. 1, the unique
discrete realization of effective-flux continuity) or, for ablations, the arithmetic mean.

Grid convention: the unknown lives on the trailing ``ndim = len(spacing)`` axes of a tensor; the
**last axis is the through-thickness axis z** and its first slice (``z = 0``) is the observed
(laser-illuminated) front surface. Leading axes of ``T`` (if any) are batch axes.

Boundary conditions, per axis (NeFTY App. A.5 / D.2):

* ``"periodic"`` — circular wrap (lateral axes of the synthetic benchmark).
* ``"neumann"`` — adiabatic, zero flux. Implemented by giving the boundary face zero conductance,
  which is exactly equivalent to the replicate padding of App. D.2 (the cross-face temperature
  increment ``T_0 - T_{-1}`` vanishes).
* ``"robin"`` (last axis only) — adiabatic front face, convective back face
  ``n·(α∇T) = -h T`` (homogeneous form after the ambient shift). The finite-volume flux balance on
  a back-face cell adds the diagonal sink ``β = h/Δz`` (App. D.2), keeping ``L`` symmetric.

All operations are plain tensor ops, hence autodiff-compatible w.r.t. both ``T`` and ``α`` and
device-agnostic (CPU / CUDA / MPS). Two equivalent implementations of a stencil application exist
(selected by ``backend``, see :data:`STENCIL_BACKENDS`):

* ``"roll"`` — the reference: ``2·ndim`` circular shifts (``torch.roll``) + ``2·ndim`` fused
  multiply-adds (13 kernels per ``L T`` in 3-D, 12 per Jacobi sweep). Deterministic backward.
* ``"flat"`` — the fast path (:class:`FlatLayout`): the state lives in a *flat padded layout* in
  which every neighbour is a contiguous 1-D slice at a constant offset; one gather refills the
  periodic ghost cells, then ``2·ndim`` ``addcmul`` on contiguous views (8 kernels per ``L T``,
  7 per Jacobi sweep, all vectorized). Identical values up to float rounding (the same
  multiply-add chain in the same order; at most one ulp per multiply-add where the vectorized and
  scalar loop tails fuse differently).
* ``"auto"`` (default) — ``"flat"``, except when autograd records through the application on an
  accelerator while ``torch.use_deterministic_algorithms(True)`` is on: the gather's backward is a
  scatter-add, which uses atomics on CUDA, so the ``"roll"`` reference (deterministic backward) is
  used there. (On CPU the scatter-add is deterministic.)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ...errors import ConfigError, ShapeError

FACE_MODES = ("harmonic", "arithmetic")
BC_KINDS = ("periodic", "neumann", "robin")
#: implementations of a stencil application (see the module docstring)
STENCIL_BACKENDS = ("auto", "flat", "roll")

__all__ = [
    "BC_KINDS",
    "FACE_MODES",
    "STENCIL_BACKENDS",
    "BoundarySpec",
    "DiffusionStencil",
    "FlatLayout",
    "ImplicitSystem",
    "apply_A",
    "apply_diffusion",
    "diagonal_A",
    "face_coefficients",
    "face_conductances",
    "chebyshev_omegas",
    "flat_layout",
    "forward_difference",
    "neighbors",
]


# --------------------------------------------------------------------------------------------
# boundary conditions
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BoundarySpec:
    """Per-axis boundary conditions of the diffusion operator (NeFTY App. A.5 / D.2).

    Args:
        kinds: one entry per grid axis, each ``"periodic"``, ``"neumann"`` or (last axis only)
            ``"robin"`` (adiabatic front face ``z = 0``, convective back face ``z = H``).
        robin_h: convective coefficient ``h ≥ 0`` of the back-face Robin condition
            ``n·(α∇T) = -h T`` (ambient-shifted). Ignored unless the last axis is ``"robin"``.
    """

    kinds: tuple[str, ...]
    robin_h: float = 0.0

    def __post_init__(self) -> None:
        kinds = tuple(str(k).lower() for k in self.kinds)
        object.__setattr__(self, "kinds", kinds)
        object.__setattr__(self, "robin_h", float(self.robin_h))
        for i, k in enumerate(kinds):
            if k not in BC_KINDS:
                raise ConfigError(f"unknown boundary condition {k!r} on axis {i}; use {BC_KINDS}")
            if k == "robin" and i != len(kinds) - 1:
                raise ConfigError(
                    "a 'robin' boundary is only supported on the last (through-thickness) axis, "
                    f"got it on axis {i} of {kinds}"
                )
        if self.robin_h < 0:
            raise ConfigError(f"robin_h must be >= 0, got {self.robin_h}")

    @property
    def ndim(self) -> int:
        return len(self.kinds)

    def periodic(self, axis: int) -> bool:
        return self.kinds[axis] == "periodic"

    @property
    def has_robin(self) -> bool:
        return self.kinds[-1] == "robin" and self.robin_h > 0.0

    @staticmethod
    def parse(
        bc: BoundarySpec | str | Sequence[str] | None, ndim: int, robin_h: float | None = None
    ) -> BoundarySpec:
        """Normalize a boundary-condition spec.

        ``None`` gives the synthetic-benchmark default (periodic lateral axes, adiabatic
        through-thickness axis, NeFTY App. A.5); a single string applies to every axis (a
        ``"robin"`` string means periodic lateral + Robin back face); a sequence gives one kind per
        axis.
        """
        if isinstance(bc, BoundarySpec):
            spec = bc if robin_h is None else BoundarySpec(bc.kinds, robin_h)
        else:
            if bc is None:
                kinds: tuple[str, ...] = ("periodic",) * (ndim - 1) + ("neumann",)
            elif isinstance(bc, str):
                kinds = ("periodic",) * (ndim - 1) + ("robin",) if bc == "robin" else (bc,) * ndim
            else:
                kinds = tuple(bc)
            spec = BoundarySpec(kinds, 0.0 if robin_h is None else robin_h)
        if spec.ndim != ndim:
            raise ConfigError(f"boundary spec {spec.kinds} has {spec.ndim} axes, grid has {ndim}")
        return spec


# --------------------------------------------------------------------------------------------
# face coefficients (Prop. 1)
# --------------------------------------------------------------------------------------------
def _axis(x: torch.Tensor, axis: int, ndim: int) -> int:
    """Tensor dim of grid axis ``axis`` for a tensor whose trailing ``ndim`` dims are the grid."""
    if not -ndim <= axis < ndim:
        raise ShapeError(f"grid axis {axis} out of range for a {ndim}-D grid")
    return x.ndim - ndim + (axis % ndim)


def face_coefficients(
    alpha: torch.Tensor,
    axis: int,
    mode: str = "harmonic",
    periodic: bool = True,
    ndim: int | None = None,
) -> torch.Tensor:
    """Face-centred diffusivity ``ᾱ_{i+1/2}`` between cell ``i`` and ``i+1`` along ``axis``.

    NeFTY Prop. 1: for piecewise-constant cells the unique flux that is continuous across the face
    uses the harmonic mean ``ᾱ = 2 α_i α_{i+1} / (α_i + α_{i+1})``, which is dominated by the
    smaller (insulating) value — a defect cell throttles the flux like a thermal resistance
    (App. A.4). ``mode="arithmetic"`` (``(α_i + α_{i+1})/2``) is the leaky first-order
    alternative used in the HM ablation (NeFTY Tab. 3 / 8).

    Args:
        alpha: cell-centred diffusivity, grid on the trailing ``ndim`` dims.
        axis: grid axis (``0..ndim-1`` or negative).
        mode: ``"harmonic"`` | ``"arithmetic"``.
        periodic: if False the boundary face (entry ``N-1``, between the last and first cell) is
            set to zero conductance (adiabatic).
        ndim: number of grid dims (default ``alpha.ndim``).

    Returns:
        Tensor of ``alpha``'s shape; entry ``i`` along ``axis`` holds ``ᾱ_{i+1/2}`` (with index
        ``i+1`` taken modulo ``N``).
    """
    nd = alpha.ndim if ndim is None else int(ndim)
    dim = _axis(alpha, axis, nd)
    nb = torch.roll(alpha, shifts=-1, dims=dim)
    if mode == "harmonic":
        face = 2.0 * alpha * nb / (alpha + nb).clamp_min(torch.finfo(alpha.dtype).tiny)
    elif mode == "arithmetic":
        face = 0.5 * (alpha + nb)
    else:
        raise ConfigError(f"unknown face mode {mode!r}; use one of {FACE_MODES}")
    if not periodic:
        n = alpha.shape[dim]
        keep = torch.ones(n, device=alpha.device, dtype=alpha.dtype)
        keep[-1] = 0.0
        view = [1] * alpha.ndim
        view[dim] = n
        face = face * keep.view(view)
    return face


def face_conductances(
    alpha: torch.Tensor,
    spacing: Sequence[float],
    bc: BoundarySpec | str | Sequence[str] | None = None,
    mode: str = "harmonic",
) -> list[torch.Tensor]:
    """Per-axis face conductances ``a_{i+1/2} = ᾱ_{i+1/2} / Δ_d²`` (App. D.2 notation ``a_pq``).

    Adiabatic / Robin axes get zero conductance on the boundary face.
    """
    nd = len(spacing)
    spec = BoundarySpec.parse(bc, nd)
    return [
        face_coefficients(alpha, ax, mode, periodic=spec.periodic(ax), ndim=nd)
        / float(spacing[ax]) ** 2
        for ax in range(nd)
    ]


def neighbors(T: torch.Tensor, ndim: int) -> list[torch.Tensor]:
    """Circularly shifted copies ``[T_{i+1}, T_{i-1}]`` for each of the trailing ``ndim`` axes.

    Non-periodic boundaries are handled by the zero boundary-face conductance, so the wrapped
    values are always multiplied by zero there. (Reference implementation of the neighbour access;
    the fast path is :class:`FlatLayout`.)
    """
    out = []
    for ax in range(ndim):
        dim = T.ndim - ndim + ax
        out.append(torch.roll(T, shifts=-1, dims=dim))
        out.append(torch.roll(T, shifts=1, dims=dim))
    return out


def forward_difference(x: torch.Tensor, axis: int, ndim: int) -> torch.Tensor:
    """``x_{i+1} - x_i`` along grid ``axis`` (circular wrap; the boundary entry is masked by the
    zero conductance for non-periodic axes)."""
    dim = x.ndim - ndim + axis
    return torch.roll(x, shifts=-1, dims=dim) - x


# --------------------------------------------------------------------------------------------
# flat padded layout: neighbours as contiguous views (the fast path)
# --------------------------------------------------------------------------------------------
class FlatLayout:
    """Flat padded layout of a grid in which all ``2·ndim`` neighbours are contiguous 1-D views.

    The grid ``(n_0, …, n_{d-1})`` is embedded in a padded box ``(n_0+2, …, n_{d-1}+2)`` (one ghost
    cell per side), flattened in C order with strides ``s_a``. The contiguous index *range*
    ``[lo, lo + L)`` from the first to the last interior cell (``lo = Σ s_a``) contains every
    interior cell (plus the ghost cells interleaved between rows), and for every range position
    ``r`` the neighbour ``±e_a`` sits at ``r ± s_a`` of the padded array ``P``. Hence, with ``P``
    refilled by **one gather** (interior values + periodic ghost copies), a stencil application is
    ``out = base + Σ_j w_j ⊙ P[lo + o_j : lo + o_j + L]`` — ``2·ndim`` fused multiply-adds on
    contiguous, vectorizable views, instead of ``2·ndim`` materialized ``torch.roll`` copies.

    Ghost cells always hold the *periodic* source value; on non-periodic axes they are multiplied by
    the zero boundary-face conductance, exactly as the wrapped values of ``torch.roll`` (so both
    implementations agree for every boundary condition). Values at the ghost positions inside the
    range are never read back (:meth:`from_range` drops them). Leading (batch) axes are supported.

    Built once per ``(grid, device)`` by :func:`flat_layout` (cached index tensors).
    """

    def __init__(self, grid: Sequence[int], device: torch.device | str | None = None) -> None:
        self.grid = tuple(int(n) for n in grid)
        self.ndim = len(self.grid)
        if self.ndim < 1 or min(self.grid) < 1:
            raise ShapeError(f"invalid grid {self.grid}")
        pshape = tuple(n + 2 for n in self.grid)
        strides = [1] * self.ndim
        for ax in range(self.ndim - 2, -1, -1):
            strides[ax] = strides[ax + 1] * pshape[ax + 1]
        self.strides = tuple(strides)
        self.n_cells = math.prod(self.grid)
        self.n_padded = math.prod(pshape)
        self.lo = sum(strides)  # flat index of the first interior cell (1, …, 1)
        hi = sum(n * s for n, s in zip(self.grid, strides))  # last interior cell (n_0, …)
        self.length = hi - self.lo + 1
        #: neighbour offsets in the order of :func:`neighbors`: ``[+e_0, -e_0, +e_1, -e_1, …]``
        self.offsets = tuple(o for s in strides for o in (s, -s))
        # padded cell -> (periodically wrapped) interior source, as grid flat index
        pgrid = torch.meshgrid(*[torch.arange(n) for n in pshape], indexing="ij")
        src = torch.zeros(pshape, dtype=torch.long)
        gstride = 1
        for ax in range(self.ndim - 1, -1, -1):
            src += ((pgrid[ax] - 1) % self.grid[ax]) * gstride
            gstride *= self.grid[ax]
        src = src.reshape(-1)
        # grid flat index -> range position of that interior cell
        igrid = torch.meshgrid(*[torch.arange(1, n + 1) for n in self.grid], indexing="ij")
        interior = sum(g * s for g, s in zip(igrid, strides)).reshape(-1) - self.lo
        dev = torch.device(device) if device is not None else None
        self.pad_from_grid = src.contiguous().to(dev)  # (n_padded,) gather from grid layout
        self.pad_from_range = interior[src].contiguous().to(dev)  # (n_padded,) from range layout
        self.range_from_grid = src[self.lo : self.lo + self.length].contiguous().to(dev)  # (L,)
        self.interior = interior.contiguous().to(dev)  # (n_cells,) range positions, grid order

    # -- conversions ---------------------------------------------------------------------------
    def _flat(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[: x.ndim - self.ndim], self.n_cells)

    def to_range(self, x: torch.Tensor) -> torch.Tensor:
        """Grid layout ``(*batch, *grid)`` → range layout ``(*batch, L)`` (one gather)."""
        return self._flat(x).index_select(-1, self.range_from_grid)

    def from_range(self, xr: torch.Tensor) -> torch.Tensor:
        """Range layout ``(*batch, L)`` → grid layout ``(*batch, *grid)`` (one gather)."""
        return xr.index_select(-1, self.interior).reshape(*xr.shape[:-1], *self.grid)

    def pad_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Padded array ``(*batch, n_padded)`` from a grid-layout tensor (one gather)."""
        return self._flat(x).index_select(-1, self.pad_from_grid)

    def pad_range(self, xr: torch.Tensor) -> torch.Tensor:
        """Padded array ``(*batch, n_padded)`` from a range-layout tensor (one gather)."""
        return xr.index_select(-1, self.pad_from_range)

    def center(self, P: torch.Tensor) -> torch.Tensor:
        """The cells themselves (range layout) as a view of the padded array."""
        return P[..., self.lo : self.lo + self.length]

    def neighbor(self, P: torch.Tensor, j: int) -> torch.Tensor:
        """Neighbour ``j`` (order of :func:`neighbors`) of every range cell, as a view of ``P``."""
        a = self.lo + self.offsets[j]
        return P[..., a : a + self.length]

    def weighted_sum(
        self, base: torch.Tensor, weights: Sequence[torch.Tensor], P: torch.Tensor
    ) -> torch.Tensor:
        """``base + Σ_j w_j ⊙ neighbour_j`` (range layout; ``2·ndim`` fused multiply-adds)."""
        out = base
        for j, w in enumerate(weights):
            out = torch.addcmul(out, w, self.neighbor(P, j))
        return out


_LAYOUTS: dict[tuple, FlatLayout] = {}


def flat_layout(grid: Sequence[int], device: torch.device | str | None = None) -> FlatLayout:
    """Cached :class:`FlatLayout` for ``grid`` with index tensors on ``device``."""
    dev = torch.device(device) if device is not None else torch.device("cpu")
    key = (tuple(int(n) for n in grid), str(dev))
    lay = _LAYOUTS.get(key)
    if lay is None:
        if len(_LAYOUTS) > 64:  # bounded (multiscale runs use a handful of grids)
            _LAYOUTS.clear()
        lay = _LAYOUTS[key] = FlatLayout(grid, dev)
    return lay


def _check_backend(backend: str) -> str:
    if backend not in STENCIL_BACKENDS:
        raise ConfigError(f"unknown stencil backend {backend!r}; use one of {STENCIL_BACKENDS}")
    return backend


def _use_flat(backend: str, *tensors: torch.Tensor) -> bool:
    """Whether an application runs on the flat layout (see the module docstring for ``"auto"``)."""
    if backend == "flat":
        return True
    if backend == "roll":
        return False
    if torch.are_deterministic_algorithms_enabled() and torch.is_grad_enabled():
        return not any(t.requires_grad and t.device.type != "cpu" for t in tensors)
    return True


def _cached(cache: dict, key: str, fn, *sources: torch.Tensor):
    """Memoize ``fn()`` (range-layout copies of stencil tensors). An entry computed without an
    autograd graph is recomputed when a graph is needed now (so gradients w.r.t. ``α`` flow)."""
    need_graph = torch.is_grad_enabled() and any(t.requires_grad for t in sources)
    hit = cache.get(key)
    if hit is not None and (hit[1] or not need_graph):
        return hit[0]
    val = fn()
    cache[key] = (val, need_graph)
    return val


def chebyshev_omegas(rho: torch.Tensor, iters: int) -> list[torch.Tensor]:
    """Chebyshev semi-iteration weights ``ω_2 … ω_K`` for a Jacobi map of spectral radius ≤ ``rho``.

    Golub & Van Loan §11.2.8: ``y_1 = J(y_0)``, ``y_{k+1} = y_{k-1} + ω_{k+1} (J(y_k) − y_{k-1})``
    with ``ω_{k+1} = 2 μ_k / (ρ μ_{k+1})``, ``μ_k = T_k(1/ρ) = cosh(kθ)``, ``θ = arccosh(1/ρ)``,
    evaluated in the overflow-free form ``(2/ρ) e^{-θ} (1 + e^{-2kθ}) / (1 + e^{-2(k+1)θ})``.
    Computed on ``rho``'s device without a host synchronization; returns ``iters - 1`` 0-dim
    tensors in ``rho``'s dtype (the weights converge to ``2 / (1 + sqrt(1 - ρ²))``).
    """
    n = int(iters) - 1
    if n <= 0:
        return []
    wdt = torch.float32 if rho.device.type == "mps" else torch.float64
    r = rho.detach().to(wdt).clamp(1e-4, 1.0 - 1e-7)
    theta = torch.acosh(1.0 / r)
    k = torch.arange(1, n + 1, device=rho.device, dtype=wdt)
    om = (2.0 / r) * torch.exp(-theta) * (1 + torch.exp(-2 * k * theta))
    om = om / (1 + torch.exp(-2 * (k + 1) * theta))
    return list(om.to(rho.dtype).unbind())


# --------------------------------------------------------------------------------------------
# the operator L(α) and the implicit-Euler system A = I - Δt L(α)
# --------------------------------------------------------------------------------------------
class DiffusionStencil:
    """Precomputed 2·ndim+1-point stencil of ``L(α)`` (NeFTY Eq. 7 / App. D.2).

    The face conductances depend on ``α`` only and are computed once per forward solve, then
    reused by all ``N_t × K`` inner iterations. The construction is differentiable in ``α`` (the
    autograd / checkpoint gradient modes and the adjoint's gradient assembly rely on it).

    Args:
        alpha: cell-centred diffusivity of shape ``grid`` (``len(spacing)`` dims).
        spacing: physical cell size per axis ``(Δx, Δy, Δz)``.
        bc: boundary conditions (see :class:`BoundarySpec`).
        face_mode: ``"harmonic"`` (Prop. 1, default) or ``"arithmetic"``.
        robin_h: back-face convective coefficient (overrides the one stored in ``bc``).
        backend: ``"auto"`` | ``"flat"`` | ``"roll"`` (see the module docstring).

    Attributes:
        coeffs: list of ``2·ndim`` neighbour coefficient tensors ``[a_{i+1/2}, a_{i-1/2}, ...]``
            (already divided by ``Δ²``), all ``≥ 0``.
        beta: Robin sink ``β = h/Δz`` on the back-face cells, or ``None``.
        diag: diagonal of ``L`` (``-Σ coeffs - β``).
    """

    def __init__(
        self,
        alpha: torch.Tensor,
        spacing: Sequence[float],
        bc: BoundarySpec | str | Sequence[str] | None = None,
        face_mode: str = "harmonic",
        robin_h: float | None = None,
        backend: str = "auto",
    ) -> None:
        self.ndim = len(spacing)
        if alpha.ndim != self.ndim:
            raise ShapeError(
                f"alpha must have {self.ndim} grid dims (spacing {tuple(spacing)}), "
                f"got shape {tuple(alpha.shape)}"
            )
        if face_mode not in FACE_MODES:
            raise ConfigError(f"unknown face mode {face_mode!r}; use one of {FACE_MODES}")
        self.backend = _check_backend(backend)
        self.bc = BoundarySpec.parse(bc, self.ndim, robin_h)
        self.spacing = tuple(float(s) for s in spacing)
        self.face_mode = face_mode
        self.grid = tuple(alpha.shape)
        self.dtype, self.device = alpha.dtype, alpha.device
        self.conductances = face_conductances(alpha, self.spacing, self.bc, face_mode)
        coeffs: list[torch.Tensor] = []
        for ax, cp in enumerate(self.conductances):
            coeffs.append(cp)  # face i+1/2  (neighbour i+1)
            coeffs.append(torch.roll(cp, shifts=1, dims=ax))  # face i-1/2  (neighbour i-1)
        self.coeffs = coeffs
        diag = -sum(coeffs)
        self.beta: torch.Tensor | None = None
        if self.bc.has_robin:
            beta = torch.zeros(self.grid, device=self.device, dtype=self.dtype)
            beta[..., -1] = self.bc.robin_h / self.spacing[-1]
            self.beta = beta
            diag = diag - beta
        self.diag = diag
        self._flat: dict[str, tuple] = {}

    @property
    def layout(self) -> FlatLayout:
        """The :class:`FlatLayout` of this stencil's grid (on its device)."""
        return flat_layout(self.grid, self.device)

    def _flat_coeffs(self) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """``diag`` and ``coeffs`` in range layout (computed once per stencil)."""
        lay = self.layout
        diag = _cached(self._flat, "diag", lambda: lay.to_range(self.diag), self.diag)
        coeffs = _cached(
            self._flat, "coeffs", lambda: [lay.to_range(c) for c in self.coeffs], self.diag
        )
        return diag, coeffs

    def apply(self, T: torch.Tensor) -> torch.Tensor:
        """``L(α) T`` for ``T`` of shape ``(*batch, *grid)``."""
        self._check(T)
        if _use_flat(self.backend, T, self.diag):
            lay = self.layout
            diag_r, coeffs_r = self._flat_coeffs()
            P = lay.pad_grid(T)
            return lay.from_range(lay.weighted_sum(diag_r * lay.center(P), coeffs_r, P))
        out = self.diag * T
        for c, n in zip(self.coeffs, neighbors(T, self.ndim)):
            out = torch.addcmul(out, c, n)
        return out

    def bilinear(self, mu: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """``μᵀ L(α) T`` via the symmetric face form ``-Σ_faces a (Δμ)(ΔT) - Σ β μ T``."""
        total = torch.zeros((), device=T.device, dtype=T.dtype)
        for ax, cp in enumerate(self.conductances):
            total = (
                total
                - (
                    cp
                    * forward_difference(mu, ax, self.ndim)
                    * forward_difference(T, ax, self.ndim)
                ).sum()
            )
        if self.beta is not None:
            total = total - (self.beta * mu * T).sum()
        return total

    def system(self, dt: float) -> ImplicitSystem:
        """The implicit-Euler matrix ``A = I - Δt L(α)`` (NeFTY Eq. 8 / 23)."""
        return ImplicitSystem(self, dt)

    def _check(self, T: torch.Tensor) -> None:
        if tuple(T.shape[T.ndim - self.ndim :]) != self.grid:
            raise ShapeError(f"state of shape {tuple(T.shape)} does not end with grid {self.grid}")


class ImplicitSystem:
    """``A(α) = I - Δt L(α)``: symmetric, strictly diagonally dominant, SPD (NeFTY App. D.2).

    Provides the matrix-free products needed by :func:`~nefi.operators.pde.linear_solvers.jacobi`
    and :func:`~nefi.operators.pde.linear_solvers.conjugate_gradient`, and the fused multi-sweep
    solvers :meth:`jacobi_sweeps` (NeFTY Eq. 24) and :meth:`chebyshev_sweeps` used by the heat
    operator.
    """

    def __init__(self, stencil: DiffusionStencil, dt: float) -> None:
        self.stencil = stencil
        self.dt = float(dt)
        self.ndim = stencil.ndim
        self.backend = stencil.backend
        self.diag = 1.0 - self.dt * stencil.diag
        self.offdiag = [-self.dt * c for c in stencil.coeffs]  # entries of R (all <= 0)
        self.inv_diag = 1.0 / self.diag
        self._jacobi_w: list[torch.Tensor] | None = None
        self._flat: dict[str, tuple] = {}

    # -- layouts -----------------------------------------------------------------------------
    @property
    def layout(self) -> FlatLayout:
        return flat_layout(tuple(self.diag.shape[-self.ndim :]), self.diag.device)

    def _range(self, name: str):
        """Range-layout copies of the system tensors (computed once per system)."""
        lay = self.layout
        if name == "diag":
            fn = lambda: lay.to_range(self.diag)  # noqa: E731
        elif name == "inv_diag":
            fn = lambda: lay.to_range(self.inv_diag)  # noqa: E731
        elif name == "offdiag":
            fn = lambda: [lay.to_range(c) for c in self.offdiag]  # noqa: E731
        elif name == "weights":
            fn = lambda: [lay.to_range(w) for w in self.jacobi_sweep_weights()]  # noqa: E731
        else:  # pragma: no cover - internal
            raise KeyError(name)
        return _cached(self._flat, name, fn, self.diag, *self.offdiag)

    def _flat_ok(self, *tensors: torch.Tensor) -> bool:
        return _use_flat(self.backend, *tensors, self.diag)

    # -- products ------------------------------------------------------------------------------
    def apply(self, T: torch.Tensor) -> torch.Tensor:
        """``A T``."""
        if self._flat_ok(T):
            lay = self.layout
            P = lay.pad_grid(T)
            base = self._range("diag") * lay.center(P)
            return lay.from_range(lay.weighted_sum(base, self._range("offdiag"), P))
        out = self.diag * T
        for c, n in zip(self.offdiag, neighbors(T, self.ndim)):
            out = torch.addcmul(out, c, n)
        return out

    def apply_offdiag(self, T: torch.Tensor) -> torch.Tensor:
        """``R T`` with ``A = D + R`` (the Jacobi splitting of NeFTY Eq. 24)."""
        if self._flat_ok(T):
            lay = self.layout
            P = lay.pad_grid(T)
            off = self._range("offdiag")
            out = off[0] * lay.neighbor(P, 0)
            for j in range(1, len(off)):
                out = torch.addcmul(out, off[j], lay.neighbor(P, j))
            return lay.from_range(out)
        nbs = neighbors(T, self.ndim)
        out = self.offdiag[0] * nbs[0]
        for c, n in zip(self.offdiag[1:], nbs[1:]):
            out = torch.addcmul(out, c, n)
        return out

    def jacobi_sweep_weights(self) -> list[torch.Tensor]:
        """``W_j = -R_j / D`` so that one Jacobi sweep is ``x ← b/D + Σ_j W_j ⊙ shift_j(x)``."""
        if self._jacobi_w is None:
            self._jacobi_w = [-c * self.inv_diag for c in self.offdiag]
        return self._jacobi_w

    def jacobi_sweep(self, b_scaled: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """One fused Jacobi sweep ``D⁻¹(b - R x)`` given ``b_scaled = D⁻¹ b`` (Eq. 24)."""
        if self._flat_ok(b_scaled, x):
            lay = self.layout
            P = lay.pad_grid(x)
            base = lay.to_range(b_scaled)
            return lay.from_range(lay.weighted_sum(base, self._range("weights"), P))
        out = b_scaled
        for w, n in zip(self.jacobi_sweep_weights(), neighbors(x, self.ndim)):
            out = torch.addcmul(out, w, n)
        return out

    # -- fused multi-sweep solvers (warm-started, fixed iteration count) ----------------------
    def spectral_bound(self) -> torch.Tensor:
        """Gershgorin bound ``ρ = max_i Σ_j W_j[i] < 1`` for the spectral radius of the Jacobi map
        (0-dim tensor, no gradient; the eigenvalues of ``D⁻¹R`` are real since ``A`` is
        symmetric)."""
        if "rho" not in self._flat:
            with torch.no_grad():
                rho = torch.stack(self.jacobi_sweep_weights()).sum(0).amax()
            self._flat["rho"] = (rho, False)
        return self._flat["rho"][0]

    def chebyshev_omegas(self, iters: int) -> list[torch.Tensor]:
        """:func:`chebyshev_omegas` for this system (cached per ``iters``; constants for autograd,
        i.e. the ``grad_mode="autograd"`` gradient treats the iteration's weights as fixed)."""
        key = f"omega{int(iters)}"
        if key not in self._flat:
            self._flat[key] = (chebyshev_omegas(self.spectral_bound(), iters), False)
        return self._flat[key][0]

    def jacobi_sweeps(self, b: torch.Tensor, x: torch.Tensor, iters: int) -> torch.Tensor:
        """``iters`` fused Jacobi sweeps ``x ← D⁻¹b + Σ_j W_j ⊙ shift_j(x)`` (NeFTY Eq. 24).

        The flat path keeps the iterate in range layout between sweeps: ``7`` kernels per sweep in
        3-D (one ghost-refill gather + six ``addcmul``) instead of ``12`` (six rolls + six
        ``addcmul``), plus three layout conversions per call.
        """
        iters = int(iters)
        if iters <= 0:
            return x
        if not self._flat_ok(b, x):
            return _jacobi_sweeps_roll(b, x, self.inv_diag, self.jacobi_sweep_weights(), iters)
        lay = self.layout
        w = self._range("weights")
        if not self._records(b, x):  # gradient-free: preallocated buffers, in-place updates
            base, out, P, V = self._workspace(b, x)
            torch.index_select(lay._flat(b), -1, lay.range_from_grid, out=base)
            base.mul_(self._range("inv_diag"))
            torch.index_select(lay._flat(x), -1, lay.pad_from_grid, out=P)
            for k in range(iters):
                _fma_chain(out, base, w, V)
                if k < iters - 1:
                    torch.index_select(out, -1, lay.pad_from_range, out=P)
            return lay.from_range(out)
        base = lay.to_range(b) * self._range("inv_diag")
        P = lay.pad_grid(x)
        for k in range(iters):
            out = lay.weighted_sum(base, w, P)
            if k < iters - 1:
                P = lay.pad_range(out)
        return lay.from_range(out)

    def chebyshev_sweeps(self, b: torch.Tensor, x: torch.Tensor, iters: int) -> torch.Tensor:
        """``iters`` Chebyshev-accelerated Jacobi iterations (semi-iterative method).

        ``y_1 = J(y_0)``, ``y_{k+1} = lerp(y_{k-1}, J(y_k), ω_{k+1})`` with the Gershgorin bound
        of :meth:`spectral_bound`: the error after ``K`` iterations is ``≤ 1/T_K(1/ρ)`` of the
        initial one (≈ ``((1 − sqrt(1−ρ²))/ρ)^K``) instead of ``ρ^K`` for plain Jacobi, e.g.
        ≈ 20 iterations match the accuracy of the paper's 50 Jacobi sweeps at the NeFTY Tab. 5
        spacing / time step (see ``docs/performance.md``). One extra ``lerp`` per iteration.
        """
        iters = int(iters)
        if iters <= 0:
            return x
        omegas = self.chebyshev_omegas(iters)
        if not self._flat_ok(b, x):
            return _chebyshev_sweeps_roll(b, x, self.inv_diag, self.jacobi_sweep_weights(), omegas)
        lay = self.layout
        w = self._range("weights")
        if not self._records(b, x):  # gradient-free: preallocated buffers, in-place updates
            base, y, P, V = self._workspace(b, x)
            y_prev, jac = self._workspace(b, x, extra=True)
            torch.index_select(lay._flat(b), -1, lay.range_from_grid, out=base)
            base.mul_(self._range("inv_diag"))
            torch.index_select(lay._flat(x), -1, lay.range_from_grid, out=y_prev)
            torch.index_select(lay._flat(x), -1, lay.pad_from_grid, out=P)
            _fma_chain(y, base, w, V)
            for om in omegas:
                torch.index_select(y, -1, lay.pad_from_range, out=P)
                _fma_chain(jac, base, w, V)
                y_prev.lerp_(jac, om)  # y_{k+1} = y_{k-1} + ω (J(y_k) − y_{k-1})
                y, y_prev = y_prev, y
            return lay.from_range(y)
        base = lay.to_range(b) * self._range("inv_diag")
        P = lay.pad_grid(x)
        y_prev = lay.center(P)
        y = lay.weighted_sum(base, w, P)
        for om in omegas:
            P = lay.pad_range(y)
            y_prev, y = y, torch.lerp(y_prev, lay.weighted_sum(base, w, P), om)
        return lay.from_range(y)

    # -- workspace for the gradient-free fused solvers -----------------------------------------
    def _records(self, *tensors: torch.Tensor) -> bool:
        """Whether autograd records through a solve with these inputs (or this system's
        ``α``-dependent coefficients)."""
        return torch.is_grad_enabled() and any(t.requires_grad for t in (*tensors, self.diag))

    def _workspace(self, b: torch.Tensor, x: torch.Tensor, extra: bool = False):
        """Reusable buffers ``(base, out, P, neighbour views of P)`` (or two more range buffers
        with ``extra``) for the batch shape / dtype / device of ``b``; allocated once per system
        and reused by every time step of a rollout (and by the adjoint's backward sweep)."""
        lay = self.layout
        batch = tuple(b.shape[: b.ndim - self.ndim])
        dtype = torch.result_type(b, x)
        key = ("ws2" if extra else "ws", batch, dtype, b.device)
        hit = self._flat.get(key)  # type: ignore[call-overload]
        if hit is None:
            rng = (*batch, lay.length)
            if extra:
                hit = (
                    torch.empty(rng, dtype=dtype, device=b.device),
                    torch.empty(rng, dtype=dtype, device=b.device),
                )
            else:
                P = torch.empty((*batch, lay.n_padded), dtype=dtype, device=b.device)
                views = [lay.neighbor(P, j) for j in range(len(lay.offsets))]
                hit = (
                    torch.empty(rng, dtype=dtype, device=b.device),
                    torch.empty(rng, dtype=dtype, device=b.device),
                    P,
                    views,
                )
            self._flat[key] = hit  # type: ignore[index]
        return hit


def _fma_chain(
    out: torch.Tensor, base: torch.Tensor, weights: Sequence[torch.Tensor], views
) -> torch.Tensor:
    """``out ← base + Σ_j w_j ⊙ view_j`` in place (same rounding as the out-of-place chain)."""
    torch.addcmul(base, weights[0], views[0], out=out)
    for j in range(1, len(weights)):
        out.addcmul_(weights[j], views[j])
    return out


def _jacobi_sweeps_roll(
    b: torch.Tensor,
    x: torch.Tensor,
    inv_diag: torch.Tensor,
    weights: list[torch.Tensor],
    iters: int,
) -> torch.Tensor:
    """Reference ``iters`` fused Jacobi sweeps with ``torch.roll`` neighbours (also the function
    ``torch.compile`` fuses on CUDA servers)."""
    ndim = inv_diag.ndim
    b_scaled = b * inv_diag
    for _ in range(iters):
        out = b_scaled
        for w, n in zip(weights, neighbors(x, ndim)):
            out = torch.addcmul(out, w, n)
        x = out
    return x


def _chebyshev_sweeps_roll(
    b: torch.Tensor,
    x: torch.Tensor,
    inv_diag: torch.Tensor,
    weights: list[torch.Tensor],
    omegas: list[torch.Tensor],
) -> torch.Tensor:
    """Reference Chebyshev-accelerated Jacobi with ``torch.roll`` neighbours."""
    ndim = inv_diag.ndim
    b_scaled = b * inv_diag

    def sweep(y: torch.Tensor) -> torch.Tensor:
        out = b_scaled
        for w, n in zip(weights, neighbors(y, ndim)):
            out = torch.addcmul(out, w, n)
        return out

    y_prev, y = x, sweep(x)
    for om in omegas:
        y_prev, y = y, torch.lerp(y_prev, sweep(y), om)
    return y


# --------------------------------------------------------------------------------------------
# functional API
# --------------------------------------------------------------------------------------------
def apply_diffusion(
    T: torch.Tensor,
    alpha: torch.Tensor,
    spacing: Sequence[float],
    bc: BoundarySpec | str | Sequence[str] | None = None,
    face_mode: str = "harmonic",
    robin_h: float | None = None,
) -> torch.Tensor:
    """``L(α) T`` — second-order finite-volume diffusion (NeFTY Eq. 7, App. D.2).

    Args:
        T: temperature (ambient-shifted), shape ``(*batch, *grid)``.
        alpha: diffusivity, shape ``grid``.
        spacing: physical cell sizes ``(Δx, [Δy,] Δz)``.
        bc: boundary conditions (default periodic lateral + adiabatic z).
        face_mode: ``"harmonic"`` (Prop. 1) or ``"arithmetic"``.
        robin_h: back-face convective coefficient for a ``"robin"`` last axis.
    """
    return DiffusionStencil(alpha, spacing, bc, face_mode, robin_h).apply(T)


def apply_A(
    T: torch.Tensor,
    alpha: torch.Tensor,
    dt: float,
    spacing: Sequence[float],
    bc: BoundarySpec | str | Sequence[str] | None = None,
    face_mode: str = "harmonic",
    robin_h: float | None = None,
) -> torch.Tensor:
    """``A(α) T = T - Δt L(α) T`` (NeFTY Eq. 8)."""
    return DiffusionStencil(alpha, spacing, bc, face_mode, robin_h).system(dt).apply(T)


def diagonal_A(
    alpha: torch.Tensor,
    dt: float,
    spacing: Sequence[float],
    bc: BoundarySpec | str | Sequence[str] | None = None,
    face_mode: str = "harmonic",
    robin_h: float | None = None,
) -> torch.Tensor:
    """Diagonal ``D`` of ``A(α)`` (the Jacobi preconditioner, NeFTY Eq. 24)."""
    return DiffusionStencil(alpha, spacing, bc, face_mode, robin_h).system(dt).diag
