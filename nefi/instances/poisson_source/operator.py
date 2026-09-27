"""Differentiable spectral Poisson solver (discrete sine transforms via ``torch.fft``).

``u = (-κ Δ)^{-1} f`` with homogeneous Dirichlet conditions on the faces of a *cell-centered*
grid (``x_i = (i + ½) h``): the Dirichlet Laplacian's eigenfunctions ``sin(m π x / L)`` sampled
at cell centers form the DST-II basis, so

    u = IDST-II( DST-II(f) / λ ),   λ_m = κ Σ_axes (m_a π / L_a)²   ("continuous" spectrum)

which is exact for band-limited sine series and spectrally accurate otherwise. With
``spectrum="fd"`` the eigenvalues ``(4/h²) sin²(m π h / (2L))`` of the standard 5-point Laplacian
with antisymmetric ghost cells are used instead (this reproduces the finite-difference solution
exactly — a cross-check against :func:`fd_poisson_solve`).

The DSTs are implemented with real FFTs of odd extensions (no scipy in the differentiable path;
device-agnostic). DST-I (node-centered grids, interior nodes ``x_i = i L / (N+1)``) is provided
too. Normalizations follow the unnormalized sums ``X_k = Σ_n x_n sin(·)`` (scipy's DST-I/II equal
twice these).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from ...domain import Domain
from ...errors import ConfigError
from ...operators.base import Fields, Operator
from ...registry import register
from ...utils.tensor import shape_tuple


def _move(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x.movedim(dim, -1)


def dst2(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Unnormalized DST-II ``X_k = Σ_n x_n sin(π (n + ½)(k + 1) / N)`` along ``dim``."""
    xm = _move(x, dim)
    n = xm.shape[-1]
    y = torch.cat([xm, -xm.flip(-1)], dim=-1)
    Y = torch.fft.rfft(y, dim=-1)[..., 1 : n + 1]
    m = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    phase = torch.exp(torch.complex(torch.zeros_like(m), -math.pi * m / (2 * n)))
    X = (0.5j * phase * Y).real
    return X.movedim(-1, dim)


def idst2(X: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Inverse of :func:`dst2` along ``dim`` (a scaled DST-III)."""
    Xm = _move(X, dim)
    n = Xm.shape[-1]
    m = torch.arange(1, n + 1, device=X.device, dtype=X.dtype)
    phase = torch.exp(torch.complex(torch.zeros_like(m), math.pi * m / (2 * n)))
    Ym = -2j * phase * Xm.to(phase.dtype)
    zero = torch.zeros(*Ym.shape[:-1], 1, device=X.device, dtype=Ym.dtype)
    y = torch.fft.irfft(torch.cat([zero, Ym], dim=-1), n=2 * n, dim=-1)
    return y[..., :n].movedim(-1, dim)


def dst1(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Unnormalized DST-I ``X_k = Σ_n x_n sin(π (n + 1)(k + 1) / (N + 1))`` along ``dim``."""
    xm = _move(x, dim)
    n = xm.shape[-1]
    z = torch.zeros(*xm.shape[:-1], 1, device=x.device, dtype=x.dtype)
    y = torch.cat([z, xm, z, -xm.flip(-1)], dim=-1)  # odd extension, period 2(N + 1)
    Y = torch.fft.rfft(y, dim=-1)[..., 1 : n + 1]
    return (-0.5 * Y.imag).movedim(-1, dim)


def idst1(X: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Inverse of :func:`dst1` (DST-I is an involution up to ``2 / (N + 1)``)."""
    n = X.shape[dim]
    return dst1(X, dim) * (2.0 / (n + 1))


def _eig_axis(n: int, length: float, spectrum: str) -> torch.Tensor:
    """Per-axis Dirichlet eigenvalues (float64, on the host)."""
    m = torch.arange(1, n + 1, dtype=torch.float64)
    if spectrum == "continuous":
        lam = (m * math.pi / length) ** 2
    elif spectrum == "fd":
        h = length / n
        lam = (4.0 / h**2) * torch.sin(m * math.pi * h / (2.0 * length)) ** 2
    else:
        raise ConfigError(f"spectrum must be 'continuous' or 'fd', got {spectrum!r}")
    return lam


@register("operator", "poisson")
class PoissonOperator(Operator):
    """Source-to-potential map ``f ↦ u = (-κ Δ)^{-1} f`` (Dirichlet), spectral and differentiable.

    Returns the full ``u`` grid (same shape as ``f``); observation sampling is done by a
    :class:`~nefi.Measurement` mask. Works for 1-3-D domains; linear (homogeneity 1).

    Args:
        domain: the source domain (cell-centered grid; Dirichlet faces at the domain boundary).
        field: name of the source field.
        conductivity: ``κ``.
        spectrum: ``"continuous"`` (exact Laplacian eigenvalues) or ``"fd"`` (5-point stencil).
    """

    homogeneity = 1.0
    batchable = True  # DST along the trailing grid axes; leading axes are batch

    def __init__(
        self,
        domain: Domain,
        field: str = "f",
        conductivity: float = 1.0,
        spectrum: str = "continuous",
    ) -> None:
        super().__init__()
        if spectrum not in ("continuous", "fd"):
            raise ConfigError(f"spectrum must be 'continuous' or 'fd', got {spectrum!r}")
        self.domain = domain
        self.primary = field
        self.conductivity = float(conductivity)
        self.spectrum = spectrum
        self.fidelity_tag = f"poisson-dst-{spectrum}-1x"
        self._cache: dict[tuple, torch.Tensor] = {}

    def eigenvalues(self, device=None, dtype=None) -> torch.Tensor:
        """Eigenvalues ``λ`` of ``-κΔ`` on the operator's grid, shape ``domain.shape``."""
        dtype = dtype or torch.get_default_dtype()
        key = (str(device), str(dtype))
        if key not in self._cache:
            lam = None
            d = self.domain.ndim
            for a, (n, length) in enumerate(zip(self.domain.shape, self.domain.size)):
                view = [1] * d
                view[a] = n
                e = _eig_axis(n, length, self.spectrum).view(view)
                lam = e if lam is None else lam + e
            self._cache[key] = (self.conductivity * lam).to(device=device, dtype=dtype)
        return self._cache[key]

    def forward(self, fields: Fields) -> torch.Tensor:
        f = self.get_field(fields)
        d = self.domain.ndim
        if tuple(f.shape[-d:]) != tuple(self.domain.shape):
            raise ConfigError(
                f"PoissonOperator built for {self.domain.shape} got {tuple(f.shape)}; "
                "use at_resolution()"
            )
        F = f
        for a in range(-d, 0):
            F = dst2(F, a)
        U = F / self.eigenvalues(f.device, f.dtype)
        for a in range(-d, 0):
            U = idst2(U, a)
        return U

    def laplacian(self, u: torch.Tensor) -> torch.Tensor:
        """Spectral ``-κΔu`` (the inverse map), for residual checks."""
        d = self.domain.ndim
        U = u
        for a in range(-d, 0):
            U = dst2(U, a)
        U = U * self.eigenvalues(u.device, u.dtype)
        for a in range(-d, 0):
            U = idst2(U, a)
        return U

    def at_resolution(self, shape: Sequence[int]) -> PoissonOperator:
        shape = shape_tuple(shape)
        if shape == tuple(self.domain.shape):
            return self
        return PoissonOperator(
            self.domain.at(shape), self.primary, self.conductivity, self.spectrum
        )

    def output_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        return shape_tuple(shape)


def fd_poisson_solve(
    f: np.ndarray, spacing: Sequence[float], conductivity: float = 1.0
) -> np.ndarray:
    """Finite-difference solve of ``-κΔu = f`` (float64, scipy sparse direct solver).

    Standard (2d+1)-point Laplacian on a cell-centered grid with homogeneous Dirichlet conditions
    imposed on the cell faces through antisymmetric ghost cells (``u_ghost = -u_boundary``). Used
    only for data generation (independent discretization; not differentiable).
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    f = np.asarray(f, dtype=np.float64)
    shape = f.shape
    mats = []
    for n, h in zip(shape, spacing):
        main = np.full(n, 2.0)
        main[0] = main[-1] = 3.0
        off = -np.ones(n - 1)
        mats.append(sp.diags([off, main, off], [-1, 0, 1]) / float(h) ** 2)
    a = None
    for i, t in enumerate(mats):
        term = sp.identity(1, format="csr")
        for j, n in enumerate(shape):
            term = sp.kron(term, t if j == i else sp.identity(n), format="csr")
        a = term if a is None else a + term
    u = spsolve((conductivity * a).tocsc(), f.ravel())
    return np.asarray(u).reshape(shape)


class FDPoissonOperator(Operator):
    """Non-differentiable finite-difference Poisson solver used by the data generator."""

    homogeneity = 1.0

    def __init__(self, domain: Domain, field: str = "f", conductivity: float = 1.0) -> None:
        super().__init__()
        self.domain = domain
        self.primary = field
        self.conductivity = float(conductivity)
        self.fidelity_tag = "poisson-fd-float64"

    def forward(self, fields: Fields) -> torch.Tensor:
        f = self.get_field(fields)
        sp = self.domain.at(tuple(f.shape)).spacing()
        u = fd_poisson_solve(f.detach().cpu().double().numpy(), sp, self.conductivity)
        return torch.as_tensor(u, device=f.device, dtype=f.dtype)

    def at_resolution(self, shape: Sequence[int]) -> FDPoissonOperator:
        shape = shape_tuple(shape)
        if shape == tuple(self.domain.shape):
            return self
        return FDPoissonOperator(self.domain.at(shape), self.primary, self.conductivity)

    def output_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        return shape_tuple(shape)


__all__ = [
    "FDPoissonOperator",
    "PoissonOperator",
    "dst1",
    "dst2",
    "fd_poisson_solve",
    "idst1",
    "idst2",
]
