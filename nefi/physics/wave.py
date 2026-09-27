"""Scalar acoustic wave equation: differentiable time-domain simulation and wave-family operators.

Model (units are the user's; the instances use km, s, km/s):

.. math::

    \\partial_t^2 p = c(x)^2 Δp + s(x, t),
    \\qquad s(x, t) = \\sum_{src} w(t)\\, δ(x - x_{src}),

in 1-D, 2-D or 3-D. The unknown ``c`` covers only the physical domain; the simulation grid adds an
absorbing layer *around* it in which ``c`` is continued by edge replication.

Discretization
--------------
* **Time**: second-order leapfrog (central differences), ``p⁺ = 2p − p⁻ + dt² (c² Δ_h p + s)``.
  Without absorption the scheme conserves the discrete energy :func:`wave_energy` exactly (tested).
* **Space**: centered 2nd- or 4th-order Laplacian (``[1, −2, 1]/h²`` or
  ``[−1/12, 4/3, −5/2, 4/3, −1/12]/h²`` per axis), zero (pressure-release) values outside the padded
  grid; implemented as one ``convNd`` per step (fast on CPU and CUDA).
* **Stability**: ``dt ≤ courant · 2 / (c_max sqrt(λ_max))`` (:func:`stable_dt`), i.e. the CFL
  condition ``c dt ≤ h/sqrt(d)`` (order 2) or ``c dt ≤ h sqrt(3/(4d))`` (order 4).
* **Absorbing boundaries** (layer of physical width ``W``, relative depth ``ξ ∈ [0, 1]``):

  - ``"pml"`` (default, 1-D/2-D): the second-order PML of Grote & Sim (2010, arXiv:1001.0319)::

        p_tt + (ζ_x+ζ_y) p_t + ζ_x ζ_y p = c² (Δp + ∂_x ψ_x + ∂_y ψ_y) + s
        ψ_x,t = −ζ_x ψ_x + (ζ_y − ζ_x) ∂_x p,   ψ_y,t = −ζ_y ψ_y + (ζ_x − ζ_y) ∂_y p

    (complex coordinate stretching ``∂_x → ∂_x / (1 + iζ_x/ω)``), with auxiliary fields on cell
    faces at half time levels and the damping terms time-centered. Profile
    ``ζ = ζ_max ξ²``, ``ζ_max = 3 c_max ln(1/R) / (2 W)`` (Collino & Tsogka 2001).
  - ``"sponge"``: damping layer ``p_tt + σ p_t = …`` (Sochacki et al. 1987), centered damping,
    ``σ = σ_max ξ²`` with ``σ_max = 3 c_max ln(1/R) / W`` (round-trip amplitude ``R``). Cheaper but
    needs ``W ≳ 1–1.5`` dominant wavelengths (impedance mismatch of the damping itself).
  - ``"cerjan"``: multiplicative taper ``exp(−(γ ξ)²)`` on ``p`` and ``p⁻`` every step (Cerjan et
    al. 1985, *Geophysics* 50:705); its strength depends on ``dt`` (not resolution-invariant).
  - ``"none"``: rigid pressure-release box (total reflection; used for the energy tests).
* **Sources / receivers** sit at arbitrary physical positions: receivers sample ``p`` by
  multilinear interpolation, sources inject ``w(t)`` with the transposed weights divided by the cell
  volume (a grid-independent discrete δ). The source wavelet is a Ricker wavelet.
* **Observations** are sampled at fixed physical times ``t_j = j·dt_obs`` (``j < n_t``); the
  simulation step is ``dt = dt_obs / m`` with the smallest integer ``m`` satisfying the CFL limit,
  so the output shape is independent of the grid resolution — which makes every operator here
  valid at every curriculum stage (``at_resolution``).

Gradients: reverse-mode autodiff through the unrolled scheme (the discrete adjoint), with
``grad_mode="checkpoint"`` recomputing blocks of steps in the backward pass
(:func:`nefi.physics.timestep.run_timestepping`; memory ``O(n_steps/k + k)`` states instead of
``O(n_steps)``) or ``grad_mode="autograd"`` (store everything; fastest for small problems).

Operators
---------
* :class:`WaveOperator` — full-waveform inversion (FWI): unknown sound speed ``c`` → traces for a
  batch of sources (Tarantola 1984; Virieux & Operto 2009, *Geophysics* 74:WCC1).
* :class:`WaveInitialConditionOperator` — photoacoustic tomography (PAT): unknown initial pressure
  ``p0`` (zero initial velocity) → traces; linear (``homogeneity = 1``).
* :func:`time_reversal` — classical PAT reconstruction by time-reversed propagation with the
  sensors as Dirichlet points (Xu & Wang 2004, *Phys. Rev. Lett.* 92:033902; Treeby & Cox 2010,
  k-Wave), or by re-emission of the reversed traces (the adjoint, ``mode="adjoint"``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from ..domain import Domain
from ..errors import ConfigError, OperatorError, ShapeError
from ..operators.base import Fields, Operator
from ..registry import register
from ..utils.tensor import resample, shape_tuple
from .timestep import auto_checkpoint_every, run_timestepping, stable_dt_wave

log = logging.getLogger("nefi")

ABSORBING = ("pml", "sponge", "cerjan", "none")
GRAD_MODES = ("checkpoint", "autograd")

__all__ = [
    "ABSORBING",
    "GRAD_MODES",
    "WaveGrid",
    "WaveInitialConditionOperator",
    "WaveOperator",
    "laplacian",
    "laplacian_kernel",
    "ricker",
    "stable_dt",
    "time_reversal",
    "wave_energy",
]


# --------------------------------------------------------------------------------------------
# wavelets and stencils
# --------------------------------------------------------------------------------------------
def ricker(t: torch.Tensor, f0: float, t0: float | None = None) -> torch.Tensor:
    """Ricker ("Mexican hat") wavelet ``(1 − 2a²) exp(−a²)``, ``a = π f0 (t − t0)``.

    Args:
        t: sample times (s).
        f0: peak (dominant) frequency (Hz); the spectrum is negligible above ≈ 2.5 f0.
        t0: time shift (s); default ``1.2 / f0`` so that ``|w(0)| < 1e-4``.
    """
    t0 = 1.2 / f0 if t0 is None else t0
    a2 = (math.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * a2) * torch.exp(-a2)


def stable_dt(
    c_max: float, spacing: Sequence[float], order: int = 4, courant: float = 0.9
) -> float:
    """CFL-limited leapfrog step (see :func:`nefi.physics.timestep.stable_dt_wave`)."""
    return stable_dt_wave(c_max, spacing, order=order, courant=courant)


_COEFFS = {2: (-2.0, 1.0), 4: (-5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0)}
_CONV = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}


def laplacian_kernel(spacing: Sequence[float], order: int, device=None, dtype=None) -> torch.Tensor:
    """Stencil of :func:`laplacian` as a ``(1, 1, k, …, k)`` convolution kernel."""
    if order not in _COEFFS:
        raise ConfigError(f"Laplacian order must be 2 or 4, got {order}")
    d = len(spacing)
    if d not in _CONV:
        raise ShapeError(f"wave stencils support 1-3 spatial dims, got {d}")
    coeffs = _COEFFS[order]
    r = len(coeffs) - 1
    k = torch.zeros((2 * r + 1,) * d, dtype=torch.float64)
    center = (r,) * d
    for ax, h in enumerate(spacing):
        for s in range(-r, r + 1):
            idx = list(center)
            idx[ax] = r + s
            k[tuple(idx)] += coeffs[abs(s)] / float(h) ** 2
    return k.to(device=device, dtype=dtype or torch.get_default_dtype()).view(1, 1, *k.shape)


def laplacian(
    p: torch.Tensor,
    spacing: Sequence[float],
    order: int = 2,
    kernel: torch.Tensor | None = None,
) -> torch.Tensor:
    """Centered finite-difference Laplacian over the trailing ``len(spacing)`` dims.

    Values outside the grid are zero (homogeneous Dirichlet / pressure-release), which keeps the
    discrete operator symmetric negative semidefinite (exact energy conservation of leapfrog).

    Args:
        p: tensor ``(*batch, *spatial)``.
        spacing: physical grid spacing per spatial axis.
        order: accuracy order, 2 or 4.
        kernel: optional precomputed stencil (:func:`laplacian_kernel`) for speed.
    """
    d = len(spacing)
    if kernel is None:
        kernel = laplacian_kernel(spacing, order, p.device, p.dtype)
    r = (kernel.shape[-1] - 1) // 2
    spatial = tuple(p.shape[-d:])
    batch = tuple(p.shape[:-d])
    y = _CONV[d](p.reshape(-1, 1, *spatial), kernel, padding=r)
    return y.reshape(*batch, *spatial)


def wave_energy(
    p_prev: torch.Tensor,
    p: torch.Tensor,
    c: torch.Tensor,
    spacing: Sequence[float],
    dt: float,
    order: int = 2,
) -> torch.Tensor:
    """Discrete leapfrog energy ``E^{n-1/2}`` (conserved exactly without absorption and sources).

    ``E = ½ ΔV Σ_i [ (p_i − p⁻_i)² / (c_i² dt²) − p_i (Δ_h p⁻)_i ]`` — the discrete analogue of
    ``½∫ (p_t²/c² + |∇p|²) dx`` for ``p_tt = c² Δp`` (multiply the scheme by ``C⁻¹ (p⁺ − p⁻)`` and
    use the symmetry of ``Δ_h``). Leading batch dims are kept.
    """
    d = len(spacing)
    dv = math.prod(float(h) for h in spacing)
    kin = (p - p_prev) ** 2 / (c**2 * dt**2)
    pot = -p * laplacian(p_prev, spacing, order)
    return 0.5 * dv * (kin + pot).flatten(-d).sum(-1)


# --------------------------------------------------------------------------------------------
# padded grid geometry
# --------------------------------------------------------------------------------------------
class WaveGrid:
    """Simulation grid = field grid padded by an absorbing layer of ``n_abs`` cells per side.

    Args:
        domain: physical domain of the unknown field at the working resolution.
        absorb_width: physical width ``W`` of the absorbing layer (0 → no padding); rounded up to
            whole cells per axis.
    """

    def __init__(self, domain: Domain, absorb_width: float = 0.0) -> None:
        self.domain = domain
        self.spacing = domain.spacing()
        self.d = domain.ndim
        self.n_abs = tuple(
            int(math.ceil(float(absorb_width) / h - 1e-9)) if absorb_width > 0 else 0
            for h in self.spacing
        )
        self.shape = tuple(domain.shape)
        self.padded_shape = tuple(n + 2 * a for n, a in zip(self.shape, self.n_abs))
        self.lo_pad = tuple(
            lo - a * h for (lo, _), a, h in zip(domain.extent, self.n_abs, self.spacing)
        )

    @property
    def cell_volume(self) -> float:
        return math.prod(self.spacing)

    def pad(self, x: torch.Tensor, mode: str = "replicate") -> torch.Tensor:
        """Pad the trailing spatial dims (``"replicate"`` for material maps, ``"zeros"`` else)."""
        if not any(self.n_abs):
            return x
        if mode == "zeros":
            pads: list[int] = []
            for a in reversed(self.n_abs):
                pads += [a, a]
            return F.pad(x, pads)
        for ax, a in enumerate(self.n_abs):
            if a == 0:
                continue
            dim = x.ndim - self.d + ax
            n = x.shape[dim]
            size = [a if i == dim else -1 for i in range(x.ndim)]
            first = x.narrow(dim, 0, 1).expand(*size)
            last = x.narrow(dim, n - 1, 1).expand(*size)
            x = torch.cat([first, x, last], dim=dim)
        return x

    def crop(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`pad` (keeps the physical-domain cells)."""
        idx = [slice(None)] * (x.ndim - self.d)
        idx += [slice(a, a + n) for a, n in zip(self.n_abs, self.shape)]
        return x[tuple(idx)]

    def layer_depth(self, ax: int, faces: bool = False) -> torch.Tensor:
        """Relative depth ``ξ ∈ [0, 1]`` into the layer along axis ``ax`` (1-D, float64).

        Evaluated at cell centers (``faces=False``, length ``n + 2a``) or at the ``n + 2a + 1``
        cell faces; ``ξ = 0`` inside the physical domain and ``1`` at the outer grid edge.
        """
        a, n = self.n_abs[ax], self.shape[ax]
        m = n + 2 * a
        pos = torch.arange(m + (1 if faces else 0), dtype=torch.float64)
        if not faces:
            pos = pos + 0.5
        if a == 0:
            return torch.zeros_like(pos, dtype=torch.float64)
        return torch.clamp(torch.maximum(a - pos, pos - (a + n)), min=0.0) / a

    def depth(self) -> torch.Tensor:
        """Cell-center relative depth per axis, shape ``(d, *padded)``."""
        out = []
        for ax in range(self.d):
            view = [1] * self.d
            view[ax] = -1
            out.append(self.layer_depth(ax).view(view).expand(self.padded_shape))
        return torch.stack(out)

    def point_weights(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Multilinear interpolation stencils of physical ``points (n, d)`` on the padded grid.

        Returns ``(index, weight)`` of shape ``(n, 2^d)`` into the flattened padded grid.
        """
        pts = torch.as_tensor(points, dtype=torch.float64)
        if pts.ndim != 2 or pts.shape[1] != self.d:
            raise ShapeError(f"points must have shape (n, {self.d}), got {tuple(pts.shape)}")
        idx_axes, w_axes = [], []
        for ax in range(self.d):
            h, lo, m = self.spacing[ax], self.lo_pad[ax], self.padded_shape[ax]
            u = (pts[:, ax] - lo) / h - 0.5  # continuous cell index
            if bool((u < -1e-9).any()) or bool((u > m - 1 + 1e-9).any()):
                raise ConfigError(
                    f"point outside the simulation grid along axis {ax}: coordinates "
                    f"{pts[:, ax].tolist()} vs grid [{lo + 0.5 * h}, {lo + (m - 0.5) * h}]"
                )
            u = u.clamp(0.0, m - 1)
            i0 = torch.floor(u).clamp(max=max(m - 2, 0)).long()
            t = u - i0
            idx_axes.append(torch.stack([i0, (i0 + 1).clamp(max=m - 1)], dim=1))
            w_axes.append(torch.stack([1.0 - t, t], dim=1))
        strides = [math.prod(self.padded_shape[ax + 1 :]) for ax in range(self.d)]
        flat_idx = torch.zeros(pts.shape[0], 1, dtype=torch.long)
        weight = torch.ones(pts.shape[0], 1, dtype=torch.float64)
        for ax in range(self.d):
            flat_idx = (flat_idx[:, :, None] + strides[ax] * idx_axes[ax][:, None, :]).flatten(1)
            weight = (weight[:, :, None] * w_axes[ax][:, None, :]).flatten(1)
        return flat_idx, weight

    def nearest_index(self, points: torch.Tensor) -> torch.Tensor:
        """Flattened index of the padded-grid cell containing each physical point."""
        pts = torch.as_tensor(points, dtype=torch.float64)
        flat = torch.zeros(pts.shape[0], dtype=torch.long)
        for ax in range(self.d):
            h, lo, m = self.spacing[ax], self.lo_pad[ax], self.padded_shape[ax]
            i = torch.floor((pts[:, ax] - lo) / h).long().clamp(0, m - 1)
            flat = flat * m + i
        return flat

    def source_density(self, points: torch.Tensor) -> torch.Tensor:
        """Discrete δ-functions at ``points``: ``(n, *padded)`` (adjoint weights / cell volume)."""
        idx, w = self.point_weights(points)
        n = idx.shape[0]
        out = torch.zeros(n, math.prod(self.padded_shape), dtype=torch.float64)
        out.scatter_add_(1, idx, w / self.cell_volume)
        return out.view(n, *self.padded_shape)


def _sample(p: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, d: int) -> torch.Tensor:
    """Interpolate ``p (*batch, *spatial)`` at stencils ``(idx, w)`` → ``(*batch, n_points)``."""
    flat = p.reshape(*p.shape[:-d], -1)
    return (flat[..., idx] * w).sum(-1)


def _as_points(x, d: int, name: str) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float64)
    if t.ndim == 1 and d == 1:
        t = t[:, None]
    if t.ndim != 2 or t.shape[1] != d:
        raise ShapeError(f"{name} must have shape (n, {d}), got {tuple(t.shape)}")
    return t


def _face_diff(p: torch.Tensor, dim: int) -> torch.Tensor:
    """``p_{i+1} − p_i`` on all ``n+1`` faces along ``dim`` (zero outside the grid)."""
    pads = [0, 0] * (p.ndim - 1 - dim) + [1, 1]
    return torch.diff(F.pad(p, pads), dim=dim)


# --------------------------------------------------------------------------------------------
# shared leapfrog machinery
# --------------------------------------------------------------------------------------------
class _LeapfrogBase(Operator):
    """Grid / time-axis / absorbing-layer bookkeeping shared by the wave operators."""

    fidelity_tag = "wave-leapfrog"
    traceable = False  # a Python time loop of leapfrog steps

    def __init__(
        self,
        domain: Domain,
        receivers,
        *,
        n_t: int,
        dt_obs: float,
        c_max: float,
        order: int = 4,
        absorbing: str = "pml",
        absorb_width: float | None = None,
        absorb_R: float = 1e-3,
        cerjan_gamma: float = 0.3,
        courant: float = 0.9,
        grad_mode: str = "checkpoint",
        checkpoint_every: int | None = None,
    ) -> None:
        super().__init__()
        if absorbing not in ABSORBING:
            raise ConfigError(f"absorbing must be one of {ABSORBING}, got {absorbing!r}")
        if absorbing == "pml" and domain.ndim > 2:
            raise ConfigError("absorbing='pml' supports 1-D/2-D grids; use 'sponge' in 3-D")
        if grad_mode not in GRAD_MODES:
            raise ConfigError(f"grad_mode must be one of {GRAD_MODES}, got {grad_mode!r}")
        if n_t < 1 or dt_obs <= 0:
            raise ConfigError("n_t must be >= 1 and dt_obs > 0")
        if not 0.0 < absorb_R < 1.0:
            raise ConfigError(f"absorb_R must be in (0, 1), got {absorb_R}")
        self.domain = domain
        self.receivers = _as_points(receivers, domain.ndim, "receivers")
        self.n_t, self.dt_obs = int(n_t), float(dt_obs)
        self.c_max = float(c_max)
        self.order = int(order)
        self.absorbing = absorbing
        if absorb_width is None:
            absorb_width = 10.0 * max(domain.spacing()) if absorbing != "none" else 0.0
        self.absorb_width = float(absorb_width) if absorbing != "none" else 0.0
        self.absorb_R = float(absorb_R)
        self.cerjan_gamma = float(cerjan_gamma)
        self.courant = float(courant)
        self.grad_mode = grad_mode
        self.checkpoint_every = checkpoint_every
        self.grid = WaveGrid(domain, self.absorb_width)
        dt_max = stable_dt(self.c_max, self.grid.spacing, self.order, self.courant)
        self.substeps = max(1, int(math.ceil(self.dt_obs / dt_max - 1e-12)))
        self.dt = self.dt_obs / self.substeps
        self.n_steps = (self.n_t - 1) * self.substeps
        self._cache: dict = {}
        self._res: dict = {}

    @property
    def use_pml(self) -> bool:
        return self.absorbing == "pml" and any(self.grid.n_abs)

    # ---- config for at_resolution ----------------------------------------------------------
    def _kwargs(self) -> dict:
        return {
            "n_t": self.n_t,
            "dt_obs": self.dt_obs,
            "c_max": self.c_max,
            "order": self.order,
            "absorbing": self.absorbing,
            "absorb_width": self.absorb_width,
            "absorb_R": self.absorb_R,
            "cerjan_gamma": self.cerjan_gamma,
            "courant": self.courant,
            "grad_mode": self.grad_mode,
            "checkpoint_every": self.checkpoint_every,
        }

    @property
    def times(self) -> torch.Tensor:
        """Observation times ``t_j = j·dt_obs`` (s)."""
        return torch.arange(self.n_t, dtype=torch.float64) * self.dt_obs

    @property
    def record(self) -> list[int]:
        return [j * self.substeps for j in range(self.n_t)]

    # ---- absorbing profiles ----------------------------------------------------------------
    def _profile_max(self, ax: int, factor: float) -> float:
        a, h = self.grid.n_abs[ax], self.grid.spacing[ax]
        return factor * self.c_max * math.log(1.0 / self.absorb_R) / (a * h)

    def damping_profile(self) -> torch.Tensor:
        """Sponge damping rate ``σ(x)`` (1/s) on the padded grid (float64).

        ``σ = Σ_a σ_max,a ξ_a²``, ``σ_max,a = 3 c_max ln(1/R) / W_a`` (the amplitude decays as
        ``exp(−∫σ/(2c) dx)``, so ``R`` is the nominal round-trip amplitude at normal incidence).
        """
        g = self.grid
        xi = g.depth()
        sig = torch.zeros(g.padded_shape, dtype=torch.float64)
        for ax, a in enumerate(g.n_abs):
            if a:
                sig = sig + self._profile_max(ax, 3.0) * xi[ax] ** 2
        return sig

    def pml_profile(self, ax: int, faces: bool = False) -> torch.Tensor:
        """PML stretching rate ``ζ_ax`` (1/s) along axis ``ax`` (1-D, float64).

        ``ζ = ζ_max ξ²``, ``ζ_max = 3 c_max ln(1/R) / (2 W)`` (Collino & Tsogka 2001).
        """
        if self.grid.n_abs[ax] == 0:
            return self.grid.layer_depth(ax, faces) * 0.0
        return self._profile_max(ax, 1.5) * self.grid.layer_depth(ax, faces) ** 2

    # ---- cached constant tensors -----------------------------------------------------------
    def _static(self, device, dtype) -> dict:
        key = (str(device), str(dtype))
        if key in self._cache:
            return self._cache[key]
        g = self.grid
        dt = self.dt
        cast = {"device": device, "dtype": dtype}
        st: dict = {"kernel": laplacian_kernel(g.spacing, self.order, **cast)}
        ridx, rw = g.point_weights(self.receivers)
        st["rec_idx"], st["rec_w"] = ridx.to(device), rw.to(**cast)
        if self.absorbing == "sponge" and any(g.n_abs):
            half = 0.5 * self.damping_profile() * dt
            st["a"] = (1.0 / (1.0 + half)).to(**cast)
            st["b"] = (1.0 - half).to(**cast)
        elif self.absorbing == "cerjan" and any(g.n_abs):
            xi = g.depth().amax(0)
            st["taper"] = torch.exp(-((self.cerjan_gamma * xi) ** 2)).to(**cast)
        elif self.use_pml:
            d = g.d

            def bcast(v: torch.Tensor, ax: int) -> torch.Tensor:
                view = [1] * d
                view[ax] = -1
                return v.view(view)

            zc = [bcast(self.pml_profile(ax), ax) for ax in range(d)]
            zsum = sum(zc)
            st["a"] = (1.0 / (1.0 + 0.5 * dt * zsum)).expand(g.padded_shape).to(**cast)
            st["b"] = (1.0 - 0.5 * dt * zsum).expand(g.padded_shape).to(**cast)
            st["zz"] = (dt**2 * zc[0] * zc[1]).expand(g.padded_shape).to(**cast) if d == 2 else None
            alphas, betas = [], []
            for ax in range(d):
                zf = bcast(self.pml_profile(ax, faces=True), ax)
                other = sum(zc[b] for b in range(d) if b != ax) if d > 1 else 0.0 * zf
                denom = 1.0 + 0.5 * dt * zf
                fshape = list(g.padded_shape)
                fshape[ax] += 1
                alphas.append(((1.0 - 0.5 * dt * zf) / denom).expand(fshape).to(**cast))
                # 1/h of the face gradient folded into β
                betas.append(
                    (dt * (other - zf) / (denom * g.spacing[ax])).expand(fshape).to(**cast)
                )
            st["psi_alpha"], st["psi_beta"] = alphas, betas
        self._extra_static(st, device, dtype)
        self._cache[key] = st
        return st

    def _extra_static(self, st: dict, device, dtype) -> None:
        """Hook for subclasses to add constant tensors."""

    def _initial_state(self, p_prev: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not self.use_pml:
            return (p_prev, p)
        batch = tuple(p.shape[: p.ndim - self.grid.d])
        psis = []
        for ax in range(self.grid.d):
            fshape = list(self.grid.padded_shape)
            fshape[ax] += 1
            psis.append(torch.zeros(*batch, *fshape, device=p.device, dtype=p.dtype))
        return (p_prev, p, *psis)

    def _step_fn(self, st: dict, source: Callable[[int], torch.Tensor | None] | None):
        """Leapfrog step ``(p⁻, p[, ψ…]) → (p, p⁺[, ψ⁺…])``; params = ``(c² dt²,)``."""
        kernel = st["kernel"]
        a, b, taper = st.get("a"), st.get("b"), st.get("taper")
        sp, order, d = self.grid.spacing, self.order, self.grid.d
        pml = self.use_pml
        zz = st.get("zz")
        alphas, betas = st.get("psi_alpha"), st.get("psi_beta")

        def step(state, params, n, dt):
            p_prev, p = state[0], state[1]
            (c2dt2,) = params
            lap = laplacian(p, sp, order, kernel)
            new_psi: list[torch.Tensor] = []
            if pml:
                for ax in range(d):
                    dim = p.ndim - d + ax
                    psi_old = state[2 + ax]
                    # ψ^{n+1/2} = α ψ^{n-1/2} + β ∂p^n ; divergence of the time average
                    psi = torch.addcmul(alphas[ax] * psi_old, betas[ax], _face_diff(p, dim))
                    lap = lap + torch.diff(psi + psi_old, dim=dim) * (0.5 / sp[ax])
                    new_psi.append(psi)
            rhs = torch.addcmul(2.0 * p - (b * p_prev if b is not None else p_prev), c2dt2, lap)
            if zz is not None:
                rhs = rhs - zz * p
            if source is not None:
                s = source(n)
                if s is not None:
                    rhs = rhs + s
            p_next = a * rhs if a is not None else rhs
            if taper is not None:
                return (taper * p, taper * p_next)
            return (p, p_next, *new_psi)

        return step

    def _ckpt(self) -> int | None:
        if self.grad_mode != "checkpoint":
            return None
        return self.checkpoint_every or auto_checkpoint_every(self.n_steps)

    def _check_speed(self, c: torch.Tensor) -> None:
        cmax = float(c.detach().max())
        if cmax > self.c_max * (1.0 + 1e-6):
            raise OperatorError(
                f"wave speed max {cmax:.4g} exceeds the c_max={self.c_max:.4g} used for the CFL "
                "time step; raise c_max or bound the field (e.g. a Bounded head)"
            )
        if float(c.detach().min()) <= 0:
            raise OperatorError("wave speed must be positive everywhere")

    def extra_repr(self) -> str:
        g = self.grid
        return (
            f"shape={g.shape}, padded={g.padded_shape}, n_t={self.n_t}, dt_obs={self.dt_obs:.4g}, "
            f"dt={self.dt:.4g} (x{self.substeps}), order={self.order}, absorbing={self.absorbing}"
        )


# --------------------------------------------------------------------------------------------
# FWI operator
# --------------------------------------------------------------------------------------------
@register("operator", "wave")
class WaveOperator(_LeapfrogBase):
    """Acoustic FWI forward model: sound speed ``c(x)`` → receiver traces for every source.

    Args:
        domain: domain of the unknown ``c`` (physical units, e.g. km).
        sources: source positions ``(n_sources, d)`` (physical units, inside the grid).
        receivers: receiver positions ``(n_receivers, d)``, shared by all sources.
        n_t: number of trace samples; ``dt_obs``: trace sampling interval (s).
        f0: Ricker peak frequency (Hz); ``t0``: wavelet delay (s, default ``1.2/f0``).
        c_max: upper bound on ``c`` used for the CFL step (must bound the field; use the Bounded
            head's upper limit).
        field: name of the sound-speed field.
        amplitude: source amplitude (the traces are linear in it).
        order: spatial accuracy order (2 or 4).
        absorbing: ``"pml"`` | ``"sponge"`` | ``"cerjan"`` | ``"none"``; ``absorb_width``: physical
            width of the layer (default 10 native cells); ``absorb_R``: nominal reflection
            coefficient of the profile; ``cerjan_gamma``: Cerjan taper strength.
        courant: safety factor on the CFL step.
        grad_mode: ``"checkpoint"`` (recompute blocks of ``checkpoint_every`` steps, default
            ``ceil(sqrt(n_steps))``) or ``"autograd"`` (store all steps).
        source_batch: simulate sources in chunks of this size (memory); ``None`` = all at once.

    Output: ``(n_sources, n_receivers, n_t)`` at times ``t_j = j·dt_obs``, independent of the grid
    resolution (``at_resolution`` re-derives the padded grid, the CFL step and the
    source/receiver interpolation stencils at the same physical positions).
    """

    def __init__(
        self,
        domain: Domain,
        sources,
        receivers,
        *,
        n_t: int,
        dt_obs: float,
        f0: float,
        c_max: float,
        field: str = "c",
        t0: float | None = None,
        amplitude: float = 1.0,
        order: int = 4,
        absorbing: str = "pml",
        absorb_width: float | None = None,
        absorb_R: float = 1e-3,
        cerjan_gamma: float = 0.3,
        courant: float = 0.9,
        grad_mode: str = "checkpoint",
        checkpoint_every: int | None = None,
        source_batch: int | None = None,
    ) -> None:
        super().__init__(
            domain,
            receivers,
            n_t=n_t,
            dt_obs=dt_obs,
            c_max=c_max,
            order=order,
            absorbing=absorbing,
            absorb_width=absorb_width,
            absorb_R=absorb_R,
            cerjan_gamma=cerjan_gamma,
            courant=courant,
            grad_mode=grad_mode,
            checkpoint_every=checkpoint_every,
        )
        self.primary = field
        self.homogeneity = None
        self.sources = _as_points(sources, domain.ndim, "sources")
        self.f0 = float(f0)
        self.t0 = 1.2 / self.f0 if t0 is None else float(t0)
        self.amplitude = float(amplitude)
        self.source_batch = source_batch
        self.fidelity_tag = f"wave-leapfrog-o{self.order}-{self.absorbing}"

    @property
    def n_sources(self) -> int:
        return int(self.sources.shape[0])

    @property
    def n_receivers(self) -> int:
        return int(self.receivers.shape[0])

    def wavelet(self) -> torch.Tensor:
        """Source time function at the simulation steps ``t_n = n·dt`` (``n < n_steps``)."""
        t = torch.arange(self.n_steps, dtype=torch.float64) * self.dt
        return self.amplitude * ricker(t, self.f0, self.t0)

    def _extra_static(self, st, device, dtype):
        st["src"] = (self.grid.source_density(self.sources) * self.dt**2).to(
            device=device, dtype=dtype
        )
        st["wavelet"] = [float(v) for v in self.wavelet()]

    def at_resolution(self, shape):
        shape = shape_tuple(shape)
        if shape == tuple(self.domain.shape):
            return self
        if shape not in self._res:
            op = WaveOperator(
                self.domain.at(shape),
                self.sources,
                self.receivers,
                f0=self.f0,
                t0=self.t0,
                amplitude=self.amplitude,
                field=self.primary,
                source_batch=self.source_batch,
                **self._kwargs(),
            )
            op.fidelity_tag = self.fidelity_tag
            self._res[shape] = op
        return self._res[shape]

    def output_shape(self, shape):
        return (self.n_sources, self.n_receivers, self.n_t)

    def simulate(
        self,
        c: torch.Tensor,
        observe: Callable | None = None,
        record: Sequence[int] | None = None,
        sources: slice | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], list[torch.Tensor]]:
        """Run the leapfrog loop for sound speed ``c`` (field grid) and return ``(state, obs)``.

        ``observe(state, n)`` receives the padded state ``(p^{n-1}, p^n[, ψ…])`` with leading
        source dim; default: receiver samples at the observation steps.
        """
        if tuple(c.shape) != tuple(self.domain.shape):
            raise ShapeError(f"c has shape {tuple(c.shape)}, operator expects {self.domain.shape}")
        self._check_speed(c)
        st = self._static(c.device, c.dtype)
        src = st["src"] if sources is None else st["src"][sources]
        wav = st["wavelet"]
        d = self.grid.d
        c2dt2 = self.grid.pad(c, "replicate") ** 2 * self.dt**2

        def source(n: int) -> torch.Tensor:
            return src * wav[n]

        if observe is None:
            ridx, rw = st["rec_idx"], st["rec_w"]

            def observe(state, n):
                return _sample(state[1], ridx, rw, d)

            record = self.record
        zeros = torch.zeros(src.shape[0], *self.grid.padded_shape, device=c.device, dtype=c.dtype)
        return run_timestepping(
            self._step_fn(st, source),
            self._initial_state(zeros, zeros),
            (c2dt2,),
            self.n_steps,
            self.dt,
            observe,
            checkpoint_every=self._ckpt(),
            record=record,
        )

    def forward(self, fields: Fields) -> torch.Tensor:
        c = self.get_field(fields)
        n = self.n_sources
        chunk = self.source_batch or n
        outs = []
        for s0 in range(0, n, chunk):
            _, obs = self.simulate(c, sources=slice(s0, min(n, s0 + chunk)))
            outs.append(torch.stack(obs, dim=-1))  # (n_src_chunk, n_rec, n_t)
        return torch.cat(outs, dim=0)


# --------------------------------------------------------------------------------------------
# photoacoustic (initial-value) operator
# --------------------------------------------------------------------------------------------
@register("operator", "wave_initial_condition")
class WaveInitialConditionOperator(_LeapfrogBase):
    """Photoacoustic forward model: initial pressure ``p0(x)`` (zero initial velocity) → traces.

    ``p(x, 0) = p0(x)``, ``∂_t p(x, 0) = 0``, ``p_tt = c² Δp`` with known ``c`` (scalar or map on
    the domain grid). The start uses ``p⁻¹ = p¹ = p⁰ + ½ dt² c² Δ_h p⁰`` (second order). Linear in
    ``p0`` (``homogeneity = 1``). Output ``(n_receivers, n_t)`` at ``t_j = j·dt_obs`` (a leading
    batch dim of ``p0`` is kept).

    Args mirror :class:`WaveOperator` (no sources); ``c`` is the known sound speed and ``c_max``
    defaults to ``max(c)``.
    """

    def __init__(
        self,
        domain: Domain,
        receivers,
        *,
        n_t: int,
        dt_obs: float,
        c: float | torch.Tensor = 1.0,
        c_max: float | None = None,
        field: str = "p0",
        order: int = 4,
        absorbing: str = "pml",
        absorb_width: float | None = None,
        absorb_R: float = 1e-3,
        cerjan_gamma: float = 0.3,
        courant: float = 0.9,
        grad_mode: str = "checkpoint",
        checkpoint_every: int | None = None,
    ) -> None:
        c_t = torch.as_tensor(c, dtype=torch.float64)
        if c_t.ndim not in (0, domain.ndim):
            raise ShapeError("c must be a scalar or a map on the domain grid")
        if c_t.ndim > 0 and tuple(c_t.shape) != tuple(domain.shape):
            c_t = resample(c_t, domain.shape)
        c_max = float(c_t.max()) if c_max is None else float(c_max)
        super().__init__(
            domain,
            receivers,
            n_t=n_t,
            dt_obs=dt_obs,
            c_max=c_max,
            order=order,
            absorbing=absorbing,
            absorb_width=absorb_width,
            absorb_R=absorb_R,
            cerjan_gamma=cerjan_gamma,
            courant=courant,
            grad_mode=grad_mode,
            checkpoint_every=checkpoint_every,
        )
        self.primary = field
        self.homogeneity = 1.0
        self.c_known = c_t
        self.fidelity_tag = f"wave-leapfrog-ic-o{self.order}-{self.absorbing}"

    def _extra_static(self, st, device, dtype):
        c = self.c_known
        if c.ndim == 0:
            c = c.expand(self.domain.shape)
        st["c2dt2"] = (self.grid.pad(c, "replicate") ** 2 * self.dt**2).to(
            device=device, dtype=dtype
        )

    def at_resolution(self, shape):
        shape = shape_tuple(shape)
        if shape == tuple(self.domain.shape):
            return self
        if shape not in self._res:
            c = self.c_known
            if c.ndim > 0:
                c = resample(c, shape)
            op = WaveInitialConditionOperator(
                self.domain.at(shape), self.receivers, c=c, field=self.primary, **self._kwargs()
            )
            op.fidelity_tag = self.fidelity_tag
            self._res[shape] = op
        return self._res[shape]

    def output_shape(self, shape):
        return (int(self.receivers.shape[0]), self.n_t)

    def simulate(self, p0: torch.Tensor, observe=None, record=None):
        """Leapfrog run from ``p0`` (field grid); see :meth:`WaveOperator.simulate`."""
        if tuple(p0.shape[p0.ndim - self.grid.d :]) != tuple(self.domain.shape):
            raise ShapeError(
                f"p0 has shape {tuple(p0.shape)}, operator expects (..., {self.domain.shape})"
            )
        st = self._static(p0.device, p0.dtype)
        c2dt2 = st["c2dt2"]
        d = self.grid.d
        p0p = self.grid.pad(p0, "zeros")
        p_m1 = p0p + 0.5 * c2dt2 * laplacian(p0p, self.grid.spacing, self.order, st["kernel"])
        if observe is None:
            ridx, rw = st["rec_idx"], st["rec_w"]

            def observe(state, n):
                return _sample(state[1], ridx, rw, d)

            record = self.record
        return run_timestepping(
            self._step_fn(st, None),
            self._initial_state(p_m1, p0p),
            (c2dt2,),
            self.n_steps,
            self.dt,
            observe,
            checkpoint_every=self._ckpt(),
            record=record,
        )

    def forward(self, fields: Fields) -> torch.Tensor:
        p0 = self.get_field(fields)
        _, obs = self.simulate(p0)
        return torch.stack(obs, dim=-1)


# --------------------------------------------------------------------------------------------
# classical PAT baseline
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def time_reversal(
    traces: torch.Tensor,
    domain: Domain,
    receivers,
    *,
    dt_obs: float,
    c: float | torch.Tensor = 1.0,
    mode: str = "dirichlet",
    order: int = 4,
    absorbing: str = "pml",
    absorb_width: float | None = None,
    absorb_R: float = 1e-3,
    courant: float = 0.9,
) -> torch.Tensor:
    """Time-reversal reconstruction of an initial pressure from receiver traces (PAT baseline).

    ``mode="dirichlet"`` (k-Wave style, Treeby & Cox 2010): propagate from rest backwards in time
    while imposing ``p(x_r, t) = d_r(t)`` at the receiver cells (nearest grid cell, linear
    interpolation in time); the field at ``t = 0`` approximates ``p0`` inside the sensor aperture.
    Exact in odd dimensions for a closed sensor surface and long enough records; in 2-D the
    non-Huygens tail of the Green's function leaves a smooth bias.
    ``mode="adjoint"``: the exact discrete adjoint ``Aᵀ d`` of
    :class:`WaveInitialConditionOperator` (back-projection by reverse-mode autodiff; unnormalized,
    blurred — the first iterate of Landweber / CG schemes).

    Args:
        traces: ``(n_receivers, n_t)`` samples at ``t_j = j·dt_obs``.
        domain: reconstruction domain (grid of the returned image).
        receivers: physical receiver positions ``(n_receivers, d)``.
        dt_obs: trace sampling interval; ``c``: known sound speed (scalar or domain map).

    Returns:
        ``p0`` estimate on ``domain.shape``.
    """
    if mode not in ("dirichlet", "adjoint"):
        raise ConfigError(f"mode must be 'dirichlet' or 'adjoint', got {mode!r}")
    tr = torch.as_tensor(traces)
    if tr.ndim == 3 and tr.shape[0] == 1:
        tr = tr[0]
    if tr.ndim != 2:
        raise ShapeError(f"traces must be (n_receivers, n_t), got {tuple(tr.shape)}")
    device = tr.device
    dtype = tr.dtype if tr.is_floating_point() else torch.float32
    tr = tr.to(dtype)
    op = WaveInitialConditionOperator(
        domain,
        receivers,
        n_t=tr.shape[-1],
        dt_obs=dt_obs,
        c=c,
        order=order,
        absorbing=absorbing,
        absorb_width=absorb_width,
        absorb_R=absorb_R,
        courant=courant,
        grad_mode="autograd",
    )
    g = op.grid
    st = op._static(device, dtype)
    m = op.substeps
    n_t = tr.shape[-1]

    def reversed_data(n: int) -> torch.Tensor:
        """Traces at reversed time ``T − n dt`` (linear interpolation between samples)."""
        u = (n_t - 1) - n / m
        j0 = min(max(int(math.floor(u)), 0), n_t - 1)
        j1 = min(j0 + 1, n_t - 1)
        t = u - j0
        return (1.0 - t) * tr[:, j0] + t * tr[:, j1]

    if mode == "adjoint":
        with torch.enable_grad():
            p0 = torch.zeros(domain.shape, device=device, dtype=dtype, requires_grad=True)
            (grad,) = torch.autograd.grad((op({"p0": p0}) * tr).sum(), p0)
        return grad
    zeros = torch.zeros(1, *g.padded_shape, device=device, dtype=dtype)
    cells = g.nearest_index(op.receivers).to(device)
    base_step = op._step_fn(st, None)

    def step(state, params, n, dt):
        out = base_step(state, params, n, dt)
        flat = out[1].reshape(1, -1).clone()
        flat[:, cells] = reversed_data(n + 1).to(flat)
        return (out[0], flat.view_as(out[1]), *out[2:])

    p_start = zeros.clone().reshape(1, -1)
    p_start[:, cells] = reversed_data(0).to(p_start)
    state = op._initial_state(zeros, p_start.view_as(zeros))
    final, _ = run_timestepping(step, state, (st["c2dt2"],), op.n_steps, op.dt, None)
    return g.crop(final[1][0])
