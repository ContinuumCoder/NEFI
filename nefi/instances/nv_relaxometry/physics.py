"""NV-relaxometry physics: dipolar Green tensor, tensor power kernels and the Lorentzian.

Implements the measurement model of NeTMY §3.1 and App. A:

* Dipolar Green tensor (NeTMY Eq. 1 / Eq. 8)::

      G_ia(R) = (μ0 / 4π) (3 R_i R_a − |R|² δ_ia) / |R|⁵,   R = (r − r_src, z0)

* NV-projected channels ``G_nv,a = Σ_i n_i G_ia`` (``n = (0, 0, 1)`` → ``G_az``, App. A.1),
* the tensor power kernel ``P(R) = Σ_a |G_nv,a(R)|²`` used by F2 / F3 (Eq. 2, Eq. 11); for a
  z-aligned NV it has the closed form ``(μ0/4π)² (|R|² + 3 z0²) / |R|⁸``,
* the scalar NV-axis kernel ``G_nn = Σ_a n_a G_nv,a`` (``G_zz``) squared after superposition by F1,
* the Lorentzian ``L(ω; ω_L) = γ² / ((ω − ω_L)² + γ²)`` (Eq. 13).

Units. ``units="physical"`` evaluates the SI expression with ``μ0 = 4π × 1e-7`` after converting
domain lengths to metres with ``length_scale`` (default ``1e-9``: domain coordinates in nm).
``units="normalized"`` divides every channel by the peak of a z-aligned NV,
``g0 = 2 (μ0/4π) / z0³``, so that ``max P = 1`` and ``max |G_zz| = 1`` exactly. The constant is
analytic (independent of the sampling grid), hence the normalization is identical at every
curriculum resolution. Kernels carry **no** cell-area factor: they act on per-cell source weights
exactly as the direct sum of Eq. (11) (the "grid-area factor" of Eq. 9 is a global constant that the
normalized fidelities are invariant to and that energy-anchored scale correction absorbs).

Kernel factories return callables with the :class:`nefi.operators.FFTConvolution` kernel signature
``(spacing, shape, device, dtype) -> Tensor`` built on :func:`nefi.operators.conv.kernel_offsets`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ...errors import ConfigError
from ...operators.conv import KernelFn, kernel_offsets

MU0 = 4.0e-7 * math.pi
"""Vacuum permeability μ0 in T·m/A (the classical SI value used by NeTMY)."""

MU0_OVER_4PI = MU0 / (4.0 * math.pi)

_AXES = {"x": 0, "y": 1, "z": 2}
UNITS = ("normalized", "physical")


def dipolar_green_tensor(R: torch.Tensor, prefactor: float = MU0_OVER_4PI) -> torch.Tensor:
    """Dipolar Green tensor ``G_ia(R) = c (3 R_i R_a − |R|² δ_ia) / |R|⁵`` (NeTMY Eq. 1).

    Args:
        R: displacement vectors ``(..., 3)`` from source to sensor (must be non-zero).
        prefactor: ``c``; ``μ0/4π`` for SI units.

    Returns:
        Tensor ``(..., 3, 3)``; ``G[..., i, a]`` maps a unit dipole along ``a`` to the field
        component ``i``.
    """
    r2 = (R * R).sum(-1)[..., None, None]
    outer = R[..., :, None] * R[..., None, :]
    eye = torch.eye(3, device=R.device, dtype=R.dtype)
    return prefactor * (3.0 * outer - r2 * eye) / r2.pow(2.5)


def _unit_axis(nv_axis: Sequence[float], device=None, dtype=None) -> torch.Tensor:
    n = torch.as_tensor(tuple(float(v) for v in nv_axis), device=device, dtype=dtype)
    if n.shape != (3,) or float(n.norm()) == 0.0:
        raise ConfigError(f"nv_axis must be a non-zero 3-vector, got {tuple(nv_axis)}")
    return n / n.norm()


@dataclass(frozen=True)
class DipolarKernels:
    """Dipolar kernels for an NV array at standoff ``z0`` (NeTMY Eq. 1-2, App. A).

    Args:
        z0: standoff height, in domain length units (e.g. nm).
        units: ``"normalized"`` (peak of a z-aligned NV = 1) or ``"physical"`` (SI).
        length_scale: metres per domain length unit (only used by ``units="physical"``).
        nv_axis: NV quantization axis ``n`` (normalized internally); NeTMY uses ``(0, 0, 1)``.
    """

    z0: float = 20.0
    units: str = "normalized"
    length_scale: float = 1e-9
    nv_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if self.units not in UNITS:
            raise ConfigError(f"units must be one of {UNITS}, got {self.units!r}")
        if not self.z0 > 0:
            raise ConfigError(f"standoff z0 must be positive, got {self.z0}")
        object.__setattr__(self, "nv_axis", tuple(float(v) for v in self.nv_axis))
        _unit_axis(self.nv_axis)

    # ---- scalars --------------------------------------------------------------------------
    @property
    def peak(self) -> float:
        """Peak of ``|G_zz|`` for a z-aligned NV in the chosen units (``1`` when normalized)."""
        if self.units == "normalized":
            return 1.0
        return 2.0 * MU0_OVER_4PI / (self.z0 * self.length_scale) ** 3

    # ---- point evaluation -----------------------------------------------------------------
    def _displacements(self, offsets: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Scaled 3-D displacement ``R`` and prefactor for in-plane ``offsets (..., 2)``."""
        z = torch.full_like(offsets[..., :1], float(self.z0))
        R = torch.cat([offsets, z], dim=-1)
        if self.units == "normalized":
            # G(R)/g0 with g0 = 2 (μ0/4π)/z0³  ==  Ĝ(R/z0) / 2 with unit prefactor
            return R / float(self.z0), 0.5
        return R * float(self.length_scale), MU0_OVER_4PI

    def channels(self, offsets: torch.Tensor) -> torch.Tensor:
        """NV-projected channels ``G_nv,a(R) = Σ_i n_i G_ia(R)``, shape ``(..., 3)`` (a = x, y, z).

        For ``n = (0, 0, 1)`` this is ``(G_xz, G_yz, G_zz)`` (``G`` is symmetric).
        """
        R, c = self._displacements(offsets)
        G = dipolar_green_tensor(R, c)
        n = _unit_axis(self.nv_axis, device=offsets.device, dtype=offsets.dtype)
        return torch.einsum("i,...ia->...a", n, G)

    def power(self, offsets: torch.Tensor) -> torch.Tensor:
        """Tensor power kernel ``P(R) = Σ_a |G_nv,a(R)|²`` (NeTMY Eq. 2 F2, Eq. 11)."""
        return (self.channels(offsets) ** 2).sum(-1)

    def nv(self, offsets: torch.Tensor) -> torch.Tensor:
        """Scalar NV-axis kernel ``G_nn(R) = Σ_a n_a G_nv,a(R)`` (``G_zz`` for n = z; F1)."""
        n = _unit_axis(self.nv_axis, device=offsets.device, dtype=offsets.dtype)
        return (self.channels(offsets) * n).sum(-1)

    def channel(self, offsets: torch.Tensor, a: str | int) -> torch.Tensor:
        """Single channel ``G_nv,a`` for ``a ∈ {x, y, z}``."""
        idx = _AXES[a] if isinstance(a, str) else int(a)
        return self.channels(offsets)[..., idx]

    # ---- FFTConvolution kernel callables ---------------------------------------------------
    def power_fn(self) -> KernelFn:
        """``(spacing, shape, device, dtype) -> P`` sampled on :func:`kernel_offsets`."""

        def fn(spacing, shape, device, dtype):
            return self.power(kernel_offsets(spacing, shape, device=device, dtype=dtype))

        return fn

    def nv_fn(self) -> KernelFn:
        """``(spacing, shape, device, dtype) -> G_nn`` (``G_zz``) sampled on the offset grid."""

        def fn(spacing, shape, device, dtype):
            return self.nv(kernel_offsets(spacing, shape, device=device, dtype=dtype))

        return fn

    def channel_fn(self, a: str | int) -> KernelFn:
        """``(spacing, shape, device, dtype) -> G_nv,a`` sampled on the offset grid."""

        def fn(spacing, shape, device, dtype):
            return self.channel(kernel_offsets(spacing, shape, device=device, dtype=dtype), a)

        return fn


def power_kernel_fn(
    z0: float,
    units: str = "normalized",
    length_scale: float = 1e-9,
    nv_axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> KernelFn:
    """Kernel callable for the F2 tensor power kernel ``P = Σ_a |G_az|²`` (NeTMY Eq. 2)."""
    return DipolarKernels(z0, units, length_scale, tuple(nv_axis)).power_fn()


def gzz_kernel_fn(
    z0: float,
    units: str = "normalized",
    length_scale: float = 1e-9,
    nv_axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> KernelFn:
    """Kernel callable for the F1 NV-axis kernel ``G_nv = Σ_a n_a G_az`` (``G_zz``, Eq. 2)."""
    return DipolarKernels(z0, units, length_scale, tuple(nv_axis)).nv_fn()


def channel_kernel_fn(
    a: str | int,
    z0: float,
    units: str = "normalized",
    length_scale: float = 1e-9,
    nv_axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> KernelFn:
    """Kernel callable for a single channel ``G_az`` (a ∈ {x, y, z}; App. A.1, Fig. 6)."""
    return DipolarKernels(z0, units, length_scale, tuple(nv_axis)).channel_fn(a)


def lorentzian(
    omega: torch.Tensor | float, omega_L: torch.Tensor | float, gamma: float
) -> torch.Tensor:
    """Lorentzian spectral response ``L(ω; ω_L) = γ² / ((ω − ω_L)² + γ²)`` (NeTMY Eq. 13).

    Broadcasts ``omega`` against ``omega_L`` (e.g. ``(F, 1, 1)`` × ``(H, W)`` → ``(F, H, W)``).
    """
    omega = torch.as_tensor(omega)
    omega_L = torch.as_tensor(omega_L)
    g2 = float(gamma) ** 2
    return g2 / ((omega - omega_L) ** 2 + g2)


def frequency_grid(
    lo: float = 1.0, hi: float = 3.0, n: int = 50, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Uniform readout frequency grid ``W`` (NeTMY App. E.1: 50 frequencies)."""
    if n < 1 or not hi >= lo:
        raise ConfigError(f"invalid frequency grid ({lo}, {hi}, {n})")
    return torch.linspace(float(lo), float(hi), int(n), dtype=dtype)


__all__ = [
    "MU0",
    "MU0_OVER_4PI",
    "DipolarKernels",
    "channel_kernel_fn",
    "dipolar_green_tensor",
    "frequency_grid",
    "gzz_kernel_fn",
    "lorentzian",
    "power_kernel_fn",
]
