"""Magnetostatics of planar sources imaged on a parallel plane (NV / SQUID / Hall microscopy).

Forward models behind wide-field NV-magnetometry current-density and magnetization imaging
(Roth, Sepulveda & Wikswo, *J. Appl. Phys.* 65, 361 (1989); Meltzer et al., *Phys. Rev. Applied*
(2017); Broadway et al., *Phys. Rev. Applied* 14, 024076 (2020); Midha et al., *Phys. Rev.
Applied* 22, 014015 (2024) — the reconstruction problem NeTMY cites as the static-field sibling of
NV noise sensing).

Geometry and conventions
------------------------
A thin source layer lies in the plane ``z = 0`` (``domain`` axes ``(x, y)`` = tensor axes
``(-2, -1)``, ``indexing="ij"``); the sensor plane is ``z = z0 > 0``. Fourier transforms follow
the torch/numpy convention ``f̂(k) = Σ f(ρ) e^{−i k·ρ}`` with ``k = (k_x, k_y)``, ``k = |k|``.
Units are consistent: pass ``mu0`` in ``[field]·[length]/[current]`` (see
:func:`magnetic_constant`, e.g. μT·μm/mA = 400π).

**Current sheet** ``K = (K_x, K_y)`` [current/length]. From the Biot–Savart law
``B_z(ρ, z0) = (μ0/4π) ∫ [K_x(y−y′) − K_y(x−x′)] / (|ρ−ρ′|² + z0²)^{3/2} d²ρ′`` and the 2-D
transform of ``1/√(ρ²+z0²)`` (``2π e^{−kz0}/k``)::

    B̂_z(k) = (μ0/2) (e^{−k z0} / k) · i (k_x K̂_y − k_y K̂_x).

(Roth et al. 1989 write ``i(k_y ĵ_x − k_x ĵ_y)/k`` with the opposite transform sign ``e^{+ik·ρ}``;
the ``1/k`` is required dimensionally.) Check: an infinite wire along +y at ``x = 0`` gives
``B_z = −μ0 I x / (2π(x² + z0²))``, the right-hand rule. Above the source the field is a potential
field, so ``B̂_x = −i (k_x/k) B̂_z`` and ``B̂_y = −i (k_y/k) B̂_z``.

**Stream function** (divergence-free by construction): ``K = ∇×(g ẑ) = (∂_y g, −∂_x g)``, so
``i(k_x K̂_y − k_y K̂_x) = k² ĝ`` and::

    B̂_z(k) = (μ0/2) k e^{−k z0} ĝ(k).

``g = I·1_Ω`` is a counter-clockwise loop current ``I`` around ``Ω`` (magnetic moment ``+ẑ``); ``g``
is the equivalent out-of-plane magnetization (Ampère equivalence, ``g = M_z t``).

**Magnetized film** of thickness ``t`` (top surface at ``z = 0``, sensor at ``z0`` above it) with
magnetization ``M(ρ)·d̂`` (fixed unit direction ``d̂``)::

    B̂_z(k) = (μ0/2) e^{−k z0} (1 − e^{−k t}) [d_z − i (k_x d_x + k_y d_y)/k] M̂(k),

whose thin-film limit ``kt ≪ 1`` is ``(μ0 t/2) k e^{−kz0} M̂_z`` for ``d̂ = ẑ``.

**Sensor layer**: an NV ensemble of thickness ``t_nv`` averages the field over
``z ∈ [z0, z0 + t_nv]``, multiplying every transfer function by ``(1 − e^{−k t_nv})/(k t_nv)``.
**Projection** onto an NV axis ``û``: ``B̂_u = [u_z − i(u_x k_x + u_y k_y)/k] B̂_z``.

The FFT operators zero-pad the source (``pad_factor``) so the periodic images of the discrete
transform are far away, and crop back to the field of view. The data generator of the
current-density benchmark instead uses direct real-space Biot–Savart summation
(:func:`biot_savart_sheet`) in float64 on a finer grid (inverse-crime guard).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from ..domain import Domain
from ..errors import ConfigError, ShapeError
from ..operators.base import Fields, Operator
from ..registry import register
from ..utils.tensor import next_fast_len, shape_tuple

MU0_SI = 4e-7 * math.pi
"""Vacuum permeability μ0 in T·m/A (4π·10⁻⁷; CODATA-2018: 1.25663706212·10⁻⁶)."""

_LENGTH = {"m": 1.0, "mm": 1e3, "um": 1e6, "µm": 1e6, "nm": 1e9}
_CURRENT = {"A": 1.0, "mA": 1e3, "uA": 1e6, "µA": 1e6}
_FIELD = {"T": 1.0, "mT": 1e3, "uT": 1e6, "µT": 1e6, "nT": 1e9, "G": 1e4}

COMPONENTS = ("z", "x", "y", "xyz")

__all__ = [
    "COMPONENTS",
    "MU0_SI",
    "BiotSavartOperator",
    "CurrentDensityOperator",
    "MagnetizationOperator",
    "biot_savart_segments",
    "biot_savart_sheet",
    "current_divergence",
    "fourier_inversion",
    "magnetic_constant",
    "padded_shape",
    "stream_to_current",
    "upward_continuation",
    "wavenumbers",
]


def magnetic_constant(length: str = "m", current: str = "A", field: str = "T") -> float:
    """μ0 expressed in ``[field]·[length]/[current]``.

    Example: ``magnetic_constant("um", "mA", "uT") == 400π`` (≈ 1256.6 μT·μm/mA), convenient for
    NV microscopy (μm pixels, mA currents, μT fields).
    """
    try:
        return MU0_SI * _FIELD[field] * _LENGTH[length] / _CURRENT[current]
    except KeyError as e:
        raise ConfigError(
            f"unknown unit {e.args[0]!r}; lengths {list(_LENGTH)}, currents {list(_CURRENT)}, "
            f"fields {list(_FIELD)}"
        ) from e


# --------------------------------------------------------------------------------------------
# k-space helpers
# --------------------------------------------------------------------------------------------
def padded_shape(shape: Sequence[int], pad_factor: float = 2.0) -> tuple[int, ...]:
    """FFT grid for zero-padded transforms: ``shape`` itself for ``pad_factor <= 1`` (periodic),
    else the next FFT-friendly size ``>= pad_factor·n`` per axis."""
    shape = shape_tuple(shape)
    if pad_factor <= 1.0:
        return shape
    return tuple(next_fast_len(int(math.ceil(pad_factor * n))) for n in shape)


def wavenumbers(
    shape: Sequence[int], spacing: Sequence[float], device=None, dtype=None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Angular wavenumbers of a 2-D real FFT grid: ``(k_x (P0,1), k_y (1,P1//2+1), |k|)``."""
    p0, p1 = shape_tuple(shape)
    dtype = dtype or torch.get_default_dtype()
    kx = 2 * math.pi * torch.fft.fftfreq(p0, d=float(spacing[0]), device=device, dtype=dtype)
    ky = 2 * math.pi * torch.fft.rfftfreq(p1, d=float(spacing[1]), device=device, dtype=dtype)
    kx, ky = kx.view(-1, 1), ky.view(1, -1)
    return kx, ky, torch.sqrt(kx**2 + ky**2)


def _odd_factor_mask(p: tuple[int, int], kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
    """Zero the Nyquist row/column for odd (derivative-like) factors of even-length grids."""
    m = torch.ones(kx.shape[0], ky.shape[1], device=kx.device, dtype=kx.dtype)
    if p[0] % 2 == 0:
        m[p[0] // 2, :] = 0.0
    if p[1] % 2 == 0:
        m[:, -1] = 0.0
    return m


def _safe_inv(k: torch.Tensor) -> torch.Tensor:
    return torch.where(k > 0, 1.0 / torch.where(k > 0, k, torch.ones_like(k)), torch.zeros_like(k))


def _layer_average(k: torch.Tensor, thickness: float) -> torch.Tensor:
    """``(1 − e^{−kt})/(kt)`` (→ 1 at k = 0): field averaged over a sensor layer of thickness t."""
    if thickness <= 0:
        return torch.ones_like(k)
    kt = k * thickness
    small = kt < 1e-6
    safe = torch.where(small, torch.ones_like(kt), kt)
    return torch.where(small, 1.0 - 0.5 * kt, -torch.expm1(-safe) / safe)


def _component_factors(
    components: str | Sequence[float],
    kx: torch.Tensor,
    ky: torch.Tensor,
    k: torch.Tensor,
    odd_mask: torch.Tensor,
) -> list[torch.Tensor]:
    """Complex factors mapping ``B̂_z`` to the requested components (or an NV-axis projection)."""
    inv = _safe_inv(k)
    fx = (-1j) * (kx * inv * odd_mask)
    fy = (-1j) * (ky * inv * odd_mask)
    one = torch.ones_like(k) + 0j
    if isinstance(components, str):
        if components == "z":
            return [one]
        if components == "x":
            return [fx]
        if components == "y":
            return [fy]
        if components == "xyz":
            return [fx, fy, one]
        raise ConfigError(f"unknown components {components!r}; use {COMPONENTS} or a unit vector")
    u = torch.as_tensor(list(components), dtype=torch.float64)
    if u.numel() != 3 or float(u.norm()) == 0.0:
        raise ConfigError(f"a projection axis needs 3 non-zero components, got {components}")
    u = (u / u.norm()).tolist()
    return [u[2] * one + u[0] * fx + u[1] * fy]


def _pad2(x: torch.Tensor, p: tuple[int, int]) -> torch.Tensor:
    n0, n1 = x.shape[-2:]
    if (n0, n1) == tuple(p):
        return x
    return F.pad(x, (0, p[1] - n1, 0, p[0] - n0))


# --------------------------------------------------------------------------------------------
# differential helpers for current sheets
# --------------------------------------------------------------------------------------------
def stream_to_current(
    g: torch.Tensor, spacing: Sequence[float], method: str = "central", pad_factor: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sheet current ``K = ∇×(g ẑ) = (∂_y g, −∂_x g)`` of a stream function ``g (..., H, W)``.

    ``"central"`` (default): second-order central differences with ``g = 0`` outside the field of
    view (so a non-zero ``g`` at the edge produces the physical edge current). ``"spectral"``:
    exact derivatives of the trigonometric interpolant on the (optionally zero-padded) FFT grid,
    Nyquist modes removed. With the matching stencil of :func:`current_divergence` the discrete
    divergence vanishes to round-off, because the difference operators along different axes
    commute (for ``"spectral"`` this holds on the periodic grid, ``pad_factor=1``, since cropping
    a padded result discards the current outside the field of view).
    """
    dx, dy = float(spacing[0]), float(spacing[1])
    if method == "central":
        gp = F.pad(g, (1, 1, 1, 1))
        jx = (gp[..., 1:-1, 2:] - gp[..., 1:-1, :-2]) / (2 * dy)
        jy = -(gp[..., 2:, 1:-1] - gp[..., :-2, 1:-1]) / (2 * dx)
        return jx, jy
    if method == "spectral":
        n = tuple(g.shape[-2:])
        p = padded_shape(n, pad_factor)
        kx, ky, _ = wavenumbers(p, (dx, dy), g.device, g.dtype)
        odd = _odd_factor_mask(p, kx, ky)
        gh = torch.fft.rfft2(_pad2(g, p))
        jx = torch.fft.irfft2(1j * ky * odd * gh, s=p)[..., : n[0], : n[1]]
        jy = torch.fft.irfft2(-1j * kx * odd * gh, s=p)[..., : n[0], : n[1]]
        return jx, jy
    raise ConfigError(f"unknown derivative method {method!r}; use 'central' or 'spectral'")


def current_divergence(
    jx: torch.Tensor,
    jy: torch.Tensor,
    spacing: Sequence[float],
    method: str = "central",
    pad_factor: float = 1.0,
) -> torch.Tensor:
    """``∂_x K_x + ∂_y K_y`` with the stencil matching :func:`stream_to_current`."""
    dx, dy = float(spacing[0]), float(spacing[1])
    if method == "central":
        xp = F.pad(jx, (0, 0, 1, 1))
        yp = F.pad(jy, (1, 1, 0, 0))
        return (xp[..., 2:, :] - xp[..., :-2, :]) / (2 * dx) + (
            yp[..., :, 2:] - yp[..., :, :-2]
        ) / (2 * dy)
    if method == "spectral":
        n = tuple(jx.shape[-2:])
        p = padded_shape(n, pad_factor)
        kx, ky, _ = wavenumbers(p, (dx, dy), jx.device, jx.dtype)
        odd = _odd_factor_mask(p, kx, ky)
        d = 1j * kx * odd * torch.fft.rfft2(_pad2(jx, p)) + 1j * ky * odd * torch.fft.rfft2(
            _pad2(jy, p)
        )
        return torch.fft.irfft2(d, s=p)[..., : n[0], : n[1]]
    raise ConfigError(f"unknown derivative method {method!r}; use 'central' or 'spectral'")


# --------------------------------------------------------------------------------------------
# FFT forward operators
# --------------------------------------------------------------------------------------------
class _PlanarFieldOperator(Operator):
    """Shared machinery: zero padding, cached k-grids, component factors, crop.

    Subclasses implement :meth:`bz_hat` returning ``B̂_z`` on the padded rfft grid.
    """

    homogeneity = 1.0
    fidelity_tag = "magnetostatics-fft"
    batchable = True  # rfft2 along the trailing grid axes; leading axes are batch

    def __init__(
        self,
        domain: Domain,
        z0: float,
        *,
        mu0: float = MU0_SI,
        pad_factor: float = 2.0,
        components: str | Sequence[float] = "z",
        nv_axis: Sequence[float] | None = None,
        nv_layer_thickness: float = 0.0,
    ) -> None:
        super().__init__()
        if domain.ndim != 2:
            raise ShapeError(
                f"planar magnetostatic operators need a 2-D domain, got {domain.shape}"
            )
        if not z0 > 0:
            raise ConfigError(f"standoff z0 must be positive, got {z0}")
        self.domain = domain
        self.z0 = float(z0)
        self.mu0 = float(mu0)
        self.pad_factor = float(pad_factor)
        self.components = components if nv_axis is None else tuple(float(v) for v in nv_axis)
        self.nv_layer_thickness = float(nv_layer_thickness)
        _component_factors(  # validate early
            self.components,
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.ones(1, 1),
            torch.ones(1, 1),
        )
        self._kcache: dict[tuple, tuple] = {}

    # --- geometry ---------------------------------------------------------------------------
    @property
    def spacing(self) -> tuple[float, float]:
        sp = self.domain.spacing()
        return float(sp[0]), float(sp[1])

    @property
    def n_components(self) -> int:
        return 3 if self.components == "xyz" else 1

    def kgrid(self, device=None, dtype=None):
        """``(p, kx, ky, k, odd_mask, factors)`` for the padded grid, cached per device/dtype."""
        dtype = dtype or torch.get_default_dtype()
        key = (str(device), str(dtype))
        if key not in self._kcache:
            p = padded_shape(self.domain.shape, self.pad_factor)
            kx, ky, k = wavenumbers(p, self.spacing, device, dtype)
            odd = _odd_factor_mask(p, kx, ky)
            factors = _component_factors(self.components, kx, ky, k, odd)
            avg = _layer_average(k, self.nv_layer_thickness)
            factors = [f * avg for f in factors]
            self._kcache[key] = (p, kx, ky, k, odd, factors)
        return self._kcache[key]

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        p = self.kgrid(x.device, x.dtype)[0]
        return _pad2(x, p)

    def _check(self, x: torch.Tensor, name: str) -> None:
        if tuple(x.shape[-2:]) != self.domain.shape:
            raise ShapeError(
                f"{type(self).__name__}: field {name!r} has grid {tuple(x.shape[-2:])}, operator "
                f"expects {self.domain.shape}; use at_resolution(shape)"
            )

    # --- to implement -----------------------------------------------------------------------
    def bz_hat(self, fields: Fields, kgrid) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    # --- Operator API ---------------------------------------------------------------------------
    def forward(self, fields: Fields) -> torch.Tensor:
        ref = self.get_field(fields, self.required_fields()[0])
        kg = self.kgrid(ref.device, ref.dtype)
        p, factors = kg[0], kg[5]
        bz = self.bz_hat(fields, kg)
        n0, n1 = self.domain.shape
        outs = [torch.fft.irfft2(f * bz, s=p)[..., :n0, :n1] for f in factors]
        return outs[0] if len(outs) == 1 else torch.stack(outs, dim=-3)

    def output_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        shape = shape_tuple(shape)
        return shape if self.n_components == 1 else (3, *shape)

    def _kwargs(self) -> dict:
        nv_axis = None if isinstance(self.components, str) else self.components
        comps = self.components if isinstance(self.components, str) else "z"
        return {
            "mu0": self.mu0,
            "pad_factor": self.pad_factor,
            "components": comps,
            "nv_axis": nv_axis,
            "nv_layer_thickness": self.nv_layer_thickness,
        }

    def extra_repr(self) -> str:
        return (
            f"grid={self.domain.shape}, z0={self.z0:g}, mu0={self.mu0:g}, "
            f"pad_factor={self.pad_factor:g}, components={self.components}"
        )


@register("operator", "current_density")
class CurrentDensityOperator(_PlanarFieldOperator):
    """Thin current sheet → stray field at standoff ``z0`` (linear, ``homogeneity = 1``).

    ``source="stream"`` (default): fields ``{g}``, ``K = ∇×(g ẑ)`` — divergence-free by
    construction, ``B̂_z = (μ0/2) k e^{−kz0} ĝ``. ``source="current"``: fields ``{jx, jy}``,
    ``B̂_z = (μ0/2)(e^{−kz0}/k) i(k_x Ĵ_y − k_y Ĵ_x)`` (only the solenoidal part radiates).

    Args:
        domain: 2-D field of view (lengths in the unit system of ``mu0`` and ``z0``).
        z0: sensor standoff above the current sheet.
        source: ``"stream"`` or ``"current"``.
        field: stream-function field name (``source="stream"``).
        current_fields: names of ``(K_x, K_y)`` (``source="current"``).
        mu0: vacuum permeability in the chosen units (:func:`magnetic_constant`).
        pad_factor: zero-padding factor of the FFT grid (1 = periodic / no padding).
        components: ``"z"`` (default), ``"x"``, ``"y"`` or ``"xyz"`` (stacked ``(3, H, W)``).
        nv_axis: project the field on this unit vector instead (e.g. an NV axis).
        nv_layer_thickness: average over a sensor layer ``[z0, z0 + t]``.
    """

    def __init__(
        self,
        domain: Domain,
        z0: float,
        *,
        source: str = "stream",
        field: str = "g",
        current_fields: Sequence[str] = ("jx", "jy"),
        mu0: float = MU0_SI,
        pad_factor: float = 2.0,
        components: str | Sequence[float] = "z",
        nv_axis: Sequence[float] | None = None,
        nv_layer_thickness: float = 0.0,
    ) -> None:
        super().__init__(
            domain,
            z0,
            mu0=mu0,
            pad_factor=pad_factor,
            components=components,
            nv_axis=nv_axis,
            nv_layer_thickness=nv_layer_thickness,
        )
        if source not in ("stream", "current"):
            raise ConfigError(f"unknown source {source!r}; use 'stream' or 'current'")
        self.source = source
        self.current_fields = tuple(current_fields)
        self.primary = field if source == "stream" else self.current_fields[0]

    def required_fields(self) -> tuple[str, ...]:
        return (self.primary,) if self.source == "stream" else self.current_fields

    def transfer(self, device=None, dtype=None) -> torch.Tensor:
        """``B̂_z / ĝ = (μ0/2) k e^{−k z0}`` on the padded rfft grid (stream-function source)."""
        k = self.kgrid(device, dtype)[3]
        return 0.5 * self.mu0 * k * torch.exp(-k * self.z0)

    def bz_hat(self, fields, kgrid):
        p, kx, ky, k, odd, _ = kgrid
        if self.source == "stream":
            g = self.get_field(fields, self.primary)
            self._check(g, self.primary)
            return (0.5 * self.mu0 * k * torch.exp(-k * self.z0)) * torch.fft.rfft2(_pad2(g, p))
        jx = self.get_field(fields, self.current_fields[0])
        jy = self.get_field(fields, self.current_fields[1])
        self._check(jx, self.current_fields[0])
        self._check(jy, self.current_fields[1])
        curl = 1j * odd * (kx * torch.fft.rfft2(_pad2(jy, p)) - ky * torch.fft.rfft2(_pad2(jx, p)))
        return (0.5 * self.mu0 * torch.exp(-k * self.z0) * _safe_inv(k)) * curl

    def at_resolution(self, shape: Sequence[int]) -> CurrentDensityOperator:
        shape = shape_tuple(shape)
        if shape == self.domain.shape:
            return self
        return CurrentDensityOperator(
            self.domain.at(shape),
            self.z0,
            source=self.source,
            field=self.primary if self.source == "stream" else "g",
            current_fields=self.current_fields,
            **self._kwargs(),
        )


@register("operator", "magnetization")
class MagnetizationOperator(_PlanarFieldOperator):
    """Magnetized thin film (uniaxial, fixed direction ``d̂``) → stray field at standoff ``z0``.

    ``B̂_z = (μ0/2) e^{−kz0}(1 − e^{−kt}) [d_z − i(k_x d_x + k_y d_y)/k] M̂``; for ``d̂ = ẑ`` and
    ``kt ≪ 1`` this is ``(μ0 t/2) k e^{−kz0} M̂_z`` (Broadway et al. 2020; Meltzer et al. 2017).

    Args:
        domain: 2-D field of view.
        z0: standoff from the film's top surface to the sensor plane.
        thickness: film thickness ``t``.
        field: magnetization-magnitude field name.
        direction: magnetization unit vector ``d̂`` (default out-of-plane ``ẑ``).
        mu0 / pad_factor / components / nv_axis / nv_layer_thickness: as in
            :class:`CurrentDensityOperator`.
    """

    def __init__(
        self,
        domain: Domain,
        z0: float,
        thickness: float,
        *,
        field: str = "mz",
        direction: Sequence[float] = (0.0, 0.0, 1.0),
        mu0: float = MU0_SI,
        pad_factor: float = 2.0,
        components: str | Sequence[float] = "z",
        nv_axis: Sequence[float] | None = None,
        nv_layer_thickness: float = 0.0,
    ) -> None:
        super().__init__(
            domain,
            z0,
            mu0=mu0,
            pad_factor=pad_factor,
            components=components,
            nv_axis=nv_axis,
            nv_layer_thickness=nv_layer_thickness,
        )
        if not thickness > 0:
            raise ConfigError(f"film thickness must be positive, got {thickness}")
        d = torch.as_tensor(list(direction), dtype=torch.float64)
        if d.numel() != 3 or float(d.norm()) == 0.0:
            raise ConfigError(f"direction must be a non-zero 3-vector, got {direction}")
        self.direction = tuple((d / d.norm()).tolist())
        self.thickness = float(thickness)
        self.primary = field

    def transfer(self, device=None, dtype=None) -> torch.Tensor:
        """Complex ``B̂_z / M̂`` on the padded rfft grid."""
        _, kx, ky, k, odd, _ = self.kgrid(device, dtype)
        dx, dy, dz = self.direction
        depth = 0.5 * self.mu0 * torch.exp(-k * self.z0) * (-torch.expm1(-k * self.thickness))
        inplane = (-1j) * (dx * kx + dy * ky) * odd * _safe_inv(k)
        t = depth * (dz + inplane)
        t[..., 0, 0] = 0.0  # a laterally uniform film has no stray field
        return t

    def bz_hat(self, fields, kgrid):
        p = kgrid[0]
        m = self.get_field(fields, self.primary)
        self._check(m, self.primary)
        return self.transfer(m.device, m.dtype) * torch.fft.rfft2(_pad2(m, p))

    def at_resolution(self, shape: Sequence[int]) -> MagnetizationOperator:
        shape = shape_tuple(shape)
        if shape == self.domain.shape:
            return self
        return MagnetizationOperator(
            self.domain.at(shape),
            self.z0,
            self.thickness,
            field=self.primary,
            direction=self.direction,
            **self._kwargs(),
        )


# --------------------------------------------------------------------------------------------
# classical k-space tools
# --------------------------------------------------------------------------------------------
def upward_continuation(
    b: torch.Tensor,
    dz: float,
    spacing: Sequence[float],
    *,
    pad_factor: float = 2.0,
    allow_downward: bool = False,
) -> torch.Tensor:
    """Continue a potential-field map ``b (..., H, W)`` from its plane to ``dz`` further away.

    ``b̂(z + dz) = e^{−k dz} b̂(z)`` (Blakely, *Potential Theory in Gravity and Magnetic
    Applications*, 1995, §12.1). Exact and composable on the periodic grid (``pad_factor=1``);
    with zero padding it treats the field outside the map as zero. Downward continuation
    (``dz < 0``) amplifies noise exponentially and must be requested with ``allow_downward``.
    """
    if dz < 0 and not allow_downward:
        raise ConfigError("dz < 0 is downward continuation (unstable); pass allow_downward=True")
    n = tuple(b.shape[-2:])
    p = padded_shape(n, pad_factor)
    _, _, k = wavenumbers(p, spacing, b.device, b.dtype)
    out = torch.fft.irfft2(torch.exp(-k * dz) * torch.fft.rfft2(_pad2(b, p)), s=p)
    return out[..., : n[0], : n[1]]


def _window(k: torch.Tensor, kc: float, kind: str, order: int) -> torch.Tensor:
    if kind in ("none", None):
        return torch.ones_like(k)
    if kind == "hanning":
        return torch.where(k < kc, 0.5 * (1.0 + torch.cos(math.pi * k / kc)), torch.zeros_like(k))
    if kind == "butterworth":
        return 1.0 / (1.0 + (k / kc) ** (2 * order))
    raise ConfigError(f"unknown window {kind!r}; use 'hanning', 'butterworth' or 'none'")


def fourier_inversion(
    bz: torch.Tensor,
    z0: float,
    spacing: Sequence[float],
    *,
    mu0: float = MU0_SI,
    reg: float = 1e-3,
    window: str = "hanning",
    cutoff_wavelength: float | None = None,
    order: int = 4,
    pad_factor: float = 2.0,
    source: str = "stream",
    thickness: float = 1.0,
    nv_layer_thickness: float = 0.0,
    return_current: bool = False,
):
    """Classical regularized k-space inversion of a ``B_z`` map (the standard NV baseline).

    ``ŝ = W(k) · T(k) b̂ / (T(k)² + reg·max T²)`` with the transfer ``T = (μ0/2) k e^{−kz0}`` of a
    stream function (``source="stream"``, Roth et al. 1989; Broadway et al. 2020) or
    ``T = (μ0/2) e^{−kz0}(1 − e^{−kt})`` of an out-of-plane magnetization of thickness ``t``
    (``source="magnetization"``), a Tikhonov term ``reg`` (relative to the transfer peak) and a
    low-pass window ``W`` (``"hanning"``: ``½(1 + cos(πk/k_c))`` for ``k < k_c``,
    ``"butterworth"`` of ``order``, or ``"none"``). The unobservable ``k = 0`` mode is set to 0.

    Args:
        bz: measured map ``(..., H, W)``.
        z0: standoff.
        spacing: pixel size ``(dx, dy)``.
        mu0: vacuum permeability in the map's units.
        reg: relative Tikhonov weight (0 = plain inverse filter).
        window: low-pass filter.
        cutoff_wavelength: ``λ_c`` with ``k_c = 2π/λ_c`` (default ``2π z0 / 3``, i.e.
            ``k_c z0 = 3``).
        order: Butterworth order.
        pad_factor: zero padding of the FFT grid.
        source: ``"stream"`` (returns ``g``) or ``"magnetization"`` (returns ``M_z``).
        thickness: film thickness for ``source="magnetization"``.
        nv_layer_thickness: sensor-layer averaging (as in the forward operators).
        return_current: also return ``(K_x, K_y) = ∇×(g ẑ)`` (central differences).

    Returns:
        The reconstructed source map (and the current components if requested).
    """
    n = tuple(bz.shape[-2:])
    p = padded_shape(n, pad_factor)
    _, _, k = wavenumbers(p, spacing, bz.device, bz.dtype)
    if source == "stream":
        t = 0.5 * mu0 * k * torch.exp(-k * z0)
    elif source == "magnetization":
        t = 0.5 * mu0 * torch.exp(-k * z0) * (-torch.expm1(-k * thickness))
    else:
        raise ConfigError(f"unknown source {source!r}; use 'stream' or 'magnetization'")
    t = t * _layer_average(k, nv_layer_thickness)
    lam = float(reg) * float(t.max()) ** 2
    den = t**2 + lam
    filt = torch.where(den > 0, t / torch.where(den > 0, den, torch.ones_like(den)), 0.0 * den)
    kc = 3.0 / z0 if cutoff_wavelength is None else 2 * math.pi / float(cutoff_wavelength)
    filt = filt * _window(k, kc, window, order)
    filt[..., 0, 0] = 0.0
    s = torch.fft.irfft2(filt * torch.fft.rfft2(_pad2(bz, p)), s=p)[..., : n[0], : n[1]]
    if return_current:
        return s, stream_to_current(s, spacing)
    return s


# --------------------------------------------------------------------------------------------
# direct real-space Biot–Savart (data generation / validation)
# --------------------------------------------------------------------------------------------
def biot_savart_sheet(
    jx: torch.Tensor,
    jy: torch.Tensor,
    src_xy: torch.Tensor,
    obs_xy: torch.Tensor,
    z0: float,
    *,
    cell_area: float,
    mu0: float = MU0_SI,
    components: str = "z",
    chunk_elems: int = 4_000_000,
) -> torch.Tensor:
    """Direct Biot–Savart summation of a discretized current sheet (midpoint rule).

    ``B(r) = (μ0/4π) Σ_s A_s K_s × (r − r_s) / |r − r_s|³`` with ``r − r_s = (x − x_s, y − y_s,
    z0)``, i.e. ``B_x = (μ0/4π) Σ A K_y z0/R³``, ``B_y = −(μ0/4π) Σ A K_x z0/R³`` and
    ``B_z = (μ0/4π) Σ A [K_x (y − y_s) − K_y (x − x_s)]/R³`` (Jackson, *Classical
    Electrodynamics*, Eq. 5.14). ``O(N_obs N_src)`` work, chunked to bound memory; use float64.

    Args:
        jx / jy: sheet-current components at the source points, any shape (flattened).
        src_xy: source positions ``(..., 2)`` matching ``jx``.
        obs_xy: observation positions ``(M, 2)`` in the plane ``z = z0``.
        z0: standoff (> 0).
        cell_area: area represented by each source point.
        mu0: vacuum permeability in the chosen units.
        components: ``"z"`` → ``(M,)``; ``"xyz"`` → ``(3, M)``.
        chunk_elems: max elements of the (observation × source) block per chunk.
    """
    kx = jx.reshape(-1)
    ky = jy.reshape(-1)
    src = src_xy.reshape(-1, 2).to(kx)
    obs = obs_xy.reshape(-1, 2).to(kx)
    if src.shape[0] != kx.shape[0]:
        raise ShapeError(f"{src.shape[0]} source points but {kx.shape[0]} current values")
    keep = (kx != 0) | (ky != 0)
    src, kx, ky = src[keep], kx[keep], ky[keep]
    m = obs.shape[0]
    pref = mu0 / (4 * math.pi) * float(cell_area)
    step = max(1, int(chunk_elems) // max(1, src.shape[0]))
    out = torch.zeros((3 if components == "xyz" else 1, m), dtype=kx.dtype, device=kx.device)
    for s in range(0, m, step):
        o = obs[s : s + step]
        dx = o[:, None, 0] - src[None, :, 0]
        dy = o[:, None, 1] - src[None, :, 1]
        inv_r3 = (dx * dx + dy * dy + z0 * z0) ** -1.5
        bz = ((kx[None] * dy - ky[None] * dx) * inv_r3).sum(-1)
        if components == "xyz":
            out[0, s : s + step] = z0 * (ky[None] * inv_r3).sum(-1)
            out[1, s : s + step] = -z0 * (kx[None] * inv_r3).sum(-1)
            out[2, s : s + step] = bz
        elif components == "z":
            out[0, s : s + step] = bz
        else:
            raise ConfigError(f"components must be 'z' or 'xyz', got {components!r}")
    out = pref * out
    return out[0] if components == "z" else out


def biot_savart_segments(
    starts: torch.Tensor,
    ends: torch.Tensor,
    currents: torch.Tensor | float,
    points: torch.Tensor,
    *,
    mu0: float = MU0_SI,
) -> torch.Tensor:
    """Exact field ``(M, 3)`` of straight line-current segments (thin wires, polygonal loops).

    ``B = (μ0 I/4π) (cos θ₁ − cos θ₂)/d² · ê × (P − A)`` for a segment ``A → B`` with direction
    ``ê``, perpendicular distance ``d`` and ``cos θ₁ = ê·(P−A)/|P−A|``, ``cos θ₂ = ê·(P−B)/|P−B|``
    (Griffiths, *Introduction to Electrodynamics*, Ex. 5.5). Points on a segment's line give 0.

    Args:
        starts / ends: ``(S, 3)`` segment end points.
        currents: ``(S,)`` or scalar current flowing from ``start`` to ``end``.
        points: ``(M, 3)`` evaluation points.
    """
    a = starts.reshape(-1, 3)
    b = ends.reshape(-1, 3).to(a)
    pts = points.reshape(-1, 3).to(a)
    cur = torch.as_tensor(currents, dtype=a.dtype, device=a.device).reshape(-1)
    cur = cur.expand(a.shape[0])
    seg = b - a
    length = seg.norm(dim=-1).clamp_min(1e-300)
    e = seg / length[:, None]
    pa = pts[:, None, :] - a[None]  # (M, S, 3)
    pb = pts[:, None, :] - b[None]
    cos1 = (pa * e[None]).sum(-1) / pa.norm(dim=-1).clamp_min(1e-300)
    cos2 = (pb * e[None]).sum(-1) / pb.norm(dim=-1).clamp_min(1e-300)
    cross = torch.cross(e[None].expand_as(pa), pa, dim=-1)
    d2 = (cross * cross).sum(-1)
    coef = torch.where(d2 > 1e-24, (cos1 - cos2) / torch.where(d2 > 1e-24, d2, 1.0), 0.0)
    return mu0 / (4 * math.pi) * ((cur[None] * coef)[..., None] * cross).sum(1)


@register("operator", "biot_savart")
class BiotSavartOperator(Operator):
    """Direct real-space forward model ``g ↦ B`` (float64 data generation, inverse-crime guard).

    The stream function is given on any source grid covering ``domain``'s extent (typically a
    ``supersample``× finer grid); the current ``K = ∇×(g ẑ)`` is formed with central differences
    on that grid and summed with :func:`biot_savart_sheet` at the cell centers of ``obs_domain``
    (the measurement pixels) at height ``z0``. Independent of the FFT operators: real space,
    midpoint quadrature on a finer grid, no padding or periodic images.

    Args:
        obs_domain: measurement grid (its extent is also the source extent).
        z0: standoff.
        field: stream-function field name.
        mu0: vacuum permeability in the chosen units.
        components: ``"z"`` or ``"xyz"``.
        chunk_elems: memory bound of :func:`biot_savart_sheet`.
    """

    homogeneity = 1.0
    fidelity_tag = "biot-savart-direct"

    def __init__(
        self,
        obs_domain: Domain,
        z0: float,
        *,
        field: str = "g",
        mu0: float = MU0_SI,
        components: str = "z",
        chunk_elems: int = 4_000_000,
    ) -> None:
        super().__init__()
        if obs_domain.ndim != 2:
            raise ShapeError("BiotSavartOperator needs a 2-D domain")
        self.obs_domain = obs_domain
        self.z0 = float(z0)
        self.primary = field
        self.mu0 = float(mu0)
        self.components = components
        self.chunk_elems = int(chunk_elems)

    def forward(self, fields: Fields) -> torch.Tensor:
        g = self.get_field(fields)
        src_dom = self.obs_domain.at(tuple(g.shape[-2:]))
        sp = src_dom.spacing()
        jx, jy = stream_to_current(g, sp, "central")
        src = src_dom.physical_coords(device=g.device, dtype=g.dtype)
        obs = self.obs_domain.physical_coords(device=g.device, dtype=g.dtype)
        b = biot_savart_sheet(
            jx,
            jy,
            src,
            obs.reshape(-1, 2),
            self.z0,
            cell_area=sp[0] * sp[1],
            mu0=self.mu0,
            components=self.components,
            chunk_elems=self.chunk_elems,
        )
        return b.reshape(*b.shape[:-1], *self.obs_domain.shape)

    def output_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        s = self.obs_domain.shape
        return s if self.components == "z" else (3, *s)
