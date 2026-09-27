"""Scalar coherent optics: angular-spectrum propagation, thin phase objects, inline holography.

Conventions: monochromatic scalar field ``u(x, y)`` with time dependence ``e^{−iωt}``, vacuum
wavelength ``λ`` and pixel pitch ``Δ`` in the same length unit (μm in the instances); axis 0 is
``x``, axis 1 is ``y``; propagation along ``+z``.

* **Angular spectrum method** (Goodman 2005, *Introduction to Fourier Optics*, §3.10):
  ``u(z) = F⁻¹{ F{u(0)} · H_z }`` with the exact transfer function
  ``H_z(f_x, f_y) = exp(i 2π z sqrt(1/λ² − f_x² − f_y²))`` for propagating waves. Evanescent
  components are dropped (``evanescent="drop"``, which makes ``±z`` propagation exactly inverse on
  the propagating band) or attenuated (``"decay"``).
* **Band limiting** (Matsushima & Shimobaba 2009, *Opt. Express* 17:19662): the sampled transfer
  function aliases for large ``z``; frequencies beyond
  ``f_limit = 1 / (λ sqrt((2 z / S)² + 1))`` (``S`` = padded window size) are removed.
* **Padding**: the FFT computes a circular convolution; zero padding by ``pad_factor`` (default 2)
  turns it into a linear one. ``pad_mode="edge"`` continues the field by edge replication instead,
  which models an object window embedded in an infinite, uniform background (the right model for
  inline holography of a plane-wave-illuminated sample; zero padding would add an aperture).
* **Thin object** (projection approximation): transmission ``t(x) = exp(iφ(x) − a(x))`` with phase
  ``φ`` (rad) and amplitude attenuation ``a`` ≥ 0 (``I = |t|² = e^{−2a}``) — Paganin (2006),
  *Coherent X-Ray Optics*, §2.2.
* **Inline (Gabor) holography** measures ``I_i = |P_{z_i} t|²`` at one or several distances; the
  global phase offset is invisible (``I`` is invariant to ``φ → φ + const``), so phase metrics are
  computed after subtracting the mean.
* Classical baseline: multi-distance **Gerchberg–Saxton** iterative projection (Gerchberg & Saxton
  1972, *Optik* 35:237; multi-plane variant Pedrini, Osten & Zhang 2005, *Opt. Lett.* 30:833).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from ..domain import Domain
from ..errors import ConfigError, ShapeError
from ..operators.base import Fields, Operator
from ..registry import register
from ..utils.tensor import shape_tuple

log = logging.getLogger("nefi")

PAD_MODES = ("constant", "edge", "none")

__all__ = [
    "HolographyOperator",
    "PAD_MODES",
    "PhaseObject",
    "gaussian_beam",
    "gerchberg_saxton",
    "phase_object",
    "propagate",
    "rayleigh_range",
    "transfer_function",
]


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.complex64, torch.complex128):
        return dtype
    return torch.complex128 if dtype == torch.float64 else torch.complex64


def rayleigh_range(w0: float, wavelength: float) -> float:
    """``z_R = π w0² / λ`` of a Gaussian beam with waist ``w0`` (1/e² intensity radius)."""
    return math.pi * w0**2 / wavelength


def gaussian_beam(
    domain: Domain, w0: float, center: Sequence[float] | None = None, dtype=torch.float64
) -> torch.Tensor:
    """Gaussian beam waist ``exp(−r²/w0²)`` (complex) sampled on ``domain`` (at ``z = 0``)."""
    x = domain.physical_coords(dtype=torch.float64)
    c = [0.5 * (lo + hi) for lo, hi in domain.extent] if center is None else list(center)
    r2 = ((x - torch.tensor(c, dtype=torch.float64)) ** 2).sum(-1)
    return torch.exp(-r2 / w0**2).to(_complex_dtype(dtype))


def phase_object(phase: torch.Tensor, absorption: torch.Tensor | float | None = None):
    """Thin-object transmission ``exp(iφ − a)`` (complex, same shape as ``phase``)."""
    if absorption is None:
        amp = torch.ones_like(phase)
    else:
        a = torch.as_tensor(absorption, dtype=phase.dtype, device=phase.device)
        amp = torch.exp(-a).expand_as(phase)
    return torch.polar(amp, phase)


@register("operator", "phase_object")
class PhaseObject(Operator):
    """Thin-object (projection-approximation) transmission ``t = A · exp(iφ − a)``.

    Paganin (2006), *Coherent X-Ray Optics*, §2.2: a sample thin compared with the depth of field
    multiplies the incident wave by its transmission. ``φ`` (rad) is the phase field;
    ``absorption`` is ``None`` (pure phase object), a known constant ``a``, or the *name* of an
    absorption field (a second unknown). ``illumination`` is the incident amplitude ``A``.
    Output: complex transmission with the field's shape.
    """

    def __init__(
        self,
        field: str = "phase",
        absorption: str | float | None = None,
        illumination: float = 1.0,
    ) -> None:
        super().__init__()
        self.primary = field
        self.absorption = absorption
        self.illumination = float(illumination)
        self.homogeneity = None

    batchable = True  # pointwise

    def required_fields(self):
        if isinstance(self.absorption, str):
            return (self.primary, self.absorption)
        return (self.primary,)

    def output_shape(self, shape):
        return shape_tuple(shape)

    def forward(self, fields: Fields) -> torch.Tensor:
        phi = self.get_field(fields)
        if isinstance(self.absorption, str):
            a = self.get_field(fields, self.absorption)
        elif self.absorption is None:
            a = None
        else:
            a = torch.full_like(phi, float(self.absorption))
        t = phase_object(phi, a)
        return t * self.illumination if self.illumination != 1.0 else t


def transfer_function(
    shape: Sequence[int],
    spacing: Sequence[float],
    wavelength: float,
    distances: Sequence[float] | torch.Tensor | float,
    band_limit: bool = True,
    evanescent: str = "drop",
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Angular-spectrum transfer functions ``H_z`` on an FFT grid: ``(n_z, *shape)`` complex.

    Args:
        shape: FFT grid (the padded window) ``(P1, P2)``; ``spacing``: pixel pitch.
        wavelength: vacuum wavelength (same unit as ``spacing``).
        distances: propagation distance(s) ``z`` (negative = back-propagation).
        band_limit: apply the Matsushima–Shimobaba band limit.
        evanescent: ``"drop"`` (zero) or ``"decay"`` (``exp(−2π|z| sqrt(f² − 1/λ²))``).
    """
    shape = shape_tuple(shape)
    if len(shape) != 2:
        raise ShapeError("angular-spectrum propagation is 2-D (transverse plane)")
    if evanescent not in ("drop", "decay"):
        raise ConfigError(f"evanescent must be 'drop' or 'decay', got {evanescent!r}")
    z = torch.as_tensor(distances, dtype=torch.float64).reshape(-1)
    fx = torch.fft.fftfreq(shape[0], d=float(spacing[0]), dtype=torch.float64)
    fy = torch.fft.fftfreq(shape[1], d=float(spacing[1]), dtype=torch.float64)
    fx, fy = torch.meshgrid(fx, fy, indexing="ij")
    w = 1.0 / wavelength**2 - fx**2 - fy**2
    prop = w > 0
    kz = 2.0 * math.pi * torch.sqrt(w.clamp_min(0.0))
    phase = z[:, None, None] * kz
    h = torch.polar(prop.to(torch.float64).expand_as(phase).clone(), phase)
    if evanescent == "decay":
        decay = torch.exp(-2.0 * math.pi * z.abs()[:, None, None] * torch.sqrt((-w).clamp_min(0.0)))
        h = torch.where(prop, h, decay.to(h.dtype))
    if band_limit:
        sx = shape[0] * float(spacing[0])
        sy = shape[1] * float(spacing[1])
        lim_x = 1.0 / (wavelength * torch.sqrt((2.0 * z.abs() / sx) ** 2 + 1.0))
        lim_y = 1.0 / (wavelength * torch.sqrt((2.0 * z.abs() / sy) ** 2 + 1.0))
        mask = (fx.abs()[None] < lim_x[:, None, None]) & (fy.abs()[None] < lim_y[:, None, None])
        h = h * mask
    return h.to(device=device, dtype=_complex_dtype(dtype))


def _pad_amounts(shape: Sequence[int], pad_factor: float) -> list[tuple[int, int]]:
    out = []
    for n in shape:
        p = max(n, int(math.ceil(float(pad_factor) * n)))
        extra = p - n
        out.append((extra // 2, extra - extra // 2))
    return out


def _pad(u: torch.Tensor, amounts: list[tuple[int, int]], mode: str) -> torch.Tensor:
    if mode == "none" or all(a == (0, 0) for a in amounts):
        return u
    pads = [amounts[1][0], amounts[1][1], amounts[0][0], amounts[0][1]]
    if mode == "constant":
        return F.pad(u, pads)
    if mode != "edge":
        raise ConfigError(f"pad_mode must be one of {PAD_MODES}, got {mode!r}")
    batch = u.shape[:-2]

    def rep(x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x.reshape(-1, 1, *x.shape[-2:]), pads, mode="replicate")
        return y.reshape(*batch, *y.shape[-2:])

    if u.is_complex():
        return torch.complex(rep(u.real), rep(u.imag))
    return rep(u)


def _crop(u: torch.Tensor, amounts: list[tuple[int, int]], shape: Sequence[int]) -> torch.Tensor:
    (a0, _), (a1, _) = amounts
    return u[..., a0 : a0 + shape[0], a1 : a1 + shape[1]]


def propagate(
    u: torch.Tensor,
    dz: float | Sequence[float] | torch.Tensor,
    wavelength: float,
    spacing: Sequence[float] | float,
    *,
    band_limit: bool = True,
    pad_factor: float = 2.0,
    pad_mode: str = "constant",
    evanescent: str = "drop",
    transfer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Angular-spectrum propagation of ``u (*batch, H, W)`` over ``dz``.

    A scalar ``dz`` returns ``(*batch, H, W)``; a sequence of ``n_z`` distances returns
    ``(*batch, n_z, H, W)``. ``transfer`` may pass precomputed transfer functions for the padded
    grid (``(n_z, P1, P2)``).
    """
    sp = (float(spacing),) * 2 if isinstance(spacing, int | float) else tuple(spacing)
    scalar = isinstance(dz, int | float) or (torch.is_tensor(dz) and dz.ndim == 0)
    u = u if u.is_complex() else u.to(_complex_dtype(u.dtype))
    shape = tuple(u.shape[-2:])
    mode = "none" if pad_factor <= 1.0 else pad_mode
    amounts = _pad_amounts(shape, pad_factor if mode != "none" else 1.0)
    up = _pad(u, amounts, mode)
    if transfer is None:
        transfer = transfer_function(
            up.shape[-2:], sp, wavelength, dz, band_limit, evanescent, u.device, u.real.dtype
        )
    out = torch.fft.ifft2(torch.fft.fft2(up).unsqueeze(-3) * transfer)
    out = _crop(out, amounts, shape)
    return out.squeeze(-3) if scalar else out


# --------------------------------------------------------------------------------------------
# inline holography operator
# --------------------------------------------------------------------------------------------
@register("operator", "holography")
class HolographyOperator(Operator):
    """Multi-distance inline holography: phase (and absorption) → intensities ``|P_z t|²``.

    Args:
        domain: object/detector plane (2-D, physical units; the detector pixel pitch equals the
            field grid spacing at every curriculum resolution).
        distances: propagation distances ``z_i`` (same unit).
        wavelength: vacuum wavelength.
        field: name of the phase field (rad).
        absorption: ``None`` (pure phase object), a constant ``a``, or the *name* of an absorption
            field (then it is an unknown too).
        illumination: incident plane-wave amplitude.
        band_limit / pad_factor / pad_mode / evanescent: see :func:`propagate`
            (default edge padding: the sample sits in an infinite uniform background).

    Output ``(n_distances, H, W)``; ``homogeneity = None`` (nonlinear in the phase).
    """

    fidelity_tag = "angular-spectrum-1x"
    batchable = True  # leading batch axes: (*batch, H, W) -> (*batch, n_distances, H, W)

    def __init__(
        self,
        domain: Domain,
        distances: Sequence[float],
        wavelength: float,
        *,
        field: str = "phase",
        absorption: str | float | None = None,
        illumination: float = 1.0,
        band_limit: bool = True,
        pad_factor: float = 2.0,
        pad_mode: str = "edge",
        evanescent: str = "drop",
    ) -> None:
        super().__init__()
        if domain.ndim != 2:
            raise ShapeError("HolographyOperator is 2-D")
        if pad_mode not in PAD_MODES:
            raise ConfigError(f"pad_mode must be one of {PAD_MODES}, got {pad_mode!r}")
        self.domain = domain
        self.distances = tuple(float(z) for z in distances)
        if not self.distances:
            raise ConfigError("need at least one propagation distance")
        self.wavelength = float(wavelength)
        self.primary = field
        self.absorption = absorption
        self.illumination = float(illumination)
        self.band_limit, self.pad_factor = bool(band_limit), float(pad_factor)
        self.pad_mode, self.evanescent = pad_mode, evanescent
        self.homogeneity = None
        self.sample = PhaseObject(field, absorption, illumination)
        self._cache: dict = {}
        self._res: dict = {}

    @property
    def n_distances(self) -> int:
        return len(self.distances)

    def required_fields(self):
        if isinstance(self.absorption, str):
            return (self.primary, self.absorption)
        return (self.primary,)

    def output_shape(self, shape):
        return (self.n_distances, *shape_tuple(shape))

    def _kw(self) -> dict:
        return {
            "field": self.primary,
            "absorption": self.absorption,
            "illumination": self.illumination,
            "band_limit": self.band_limit,
            "pad_factor": self.pad_factor,
            "pad_mode": self.pad_mode,
            "evanescent": self.evanescent,
        }

    def at_resolution(self, shape):
        shape = shape_tuple(shape)
        if shape == tuple(self.domain.shape):
            return self
        if shape not in self._res:
            op = HolographyOperator(
                self.domain.at(shape), self.distances, self.wavelength, **self._kw()
            )
            op.fidelity_tag = self.fidelity_tag
            self._res[shape] = op
        return self._res[shape]

    def _transfer(self, device, dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        if key not in self._cache:
            mode = "none" if self.pad_factor <= 1.0 else self.pad_mode
            amounts = _pad_amounts(self.domain.shape, self.pad_factor if mode != "none" else 1.0)
            padded = tuple(n + a + b for n, (a, b) in zip(self.domain.shape, amounts))
            self._cache[key] = transfer_function(
                padded,
                self.domain.spacing(),
                self.wavelength,
                self.distances,
                self.band_limit,
                self.evanescent,
                device,
                dtype,
            )
        return self._cache[key]

    def transmission(self, fields: Fields) -> torch.Tensor:
        """Complex sample transmission (:class:`PhaseObject`)."""
        return self.sample(fields)

    def fields_at_detectors(self, fields: Fields) -> torch.Tensor:
        """Complex fields at every distance ``(n_distances, H, W)``."""
        t = self.transmission(fields)
        if tuple(t.shape[-2:]) != tuple(self.domain.shape):
            raise ShapeError(f"field shape {tuple(t.shape)} != operator grid {self.domain.shape}")
        return propagate(
            t,
            list(self.distances),
            self.wavelength,
            self.domain.spacing(),
            band_limit=self.band_limit,
            pad_factor=self.pad_factor,
            pad_mode=self.pad_mode,
            evanescent=self.evanescent,
            transfer=self._transfer(t.device, t.real.dtype),
        )

    def forward(self, fields: Fields) -> torch.Tensor:
        u = self.fields_at_detectors(fields)
        return u.real**2 + u.imag**2

    def extra_repr(self) -> str:
        return (
            f"shape={self.domain.shape}, distances={self.distances}, λ={self.wavelength}, "
            f"pad={self.pad_factor}x/{self.pad_mode}, band_limit={self.band_limit}"
        )


# --------------------------------------------------------------------------------------------
# classical baseline: multi-distance Gerchberg–Saxton
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def gerchberg_saxton(
    intensities: torch.Tensor,
    op: HolographyOperator,
    n_iter: int = 100,
    pure_phase: bool = True,
    init_phase: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Multi-distance Gerchberg–Saxton phase retrieval (sequential plane projections).

    For each iteration and each plane ``i``: propagate the object estimate to ``z_i``, replace the
    modulus by ``sqrt(I_i)`` (keeping the phase), propagate back; in the object plane enforce the
    known modulus (``pure_phase``: ``|t| = e^{−a}`` with the operator's constant absorption, 1 by
    default). Uses exactly the operator's propagation model (padding, band limit).

    Args:
        intensities: ``(n_distances, H, W)`` measured intensities (negative noise is clipped).
        op: the :class:`HolographyOperator` defining geometry and propagation.
        n_iter: outer iterations.
        pure_phase: enforce the object-plane modulus constraint.
        init_phase: optional initial phase (default 0).

    Returns:
        ``(phase, history)`` with the wrapped phase ``(H, W)`` in ``(−π, π]`` (global offset chosen
        so that the mean transmission is real) and the relative intensity mismatch
        ``‖I(t) − I‖/‖I‖`` after every iteration.
    """
    meas = torch.as_tensor(intensities)
    dtype = meas.dtype if meas.is_floating_point() else torch.float32
    amp = meas.to(dtype).clamp_min(0.0).sqrt()
    shape = tuple(op.domain.shape)
    kw = {
        "band_limit": op.band_limit,
        "pad_factor": op.pad_factor,
        "pad_mode": op.pad_mode,
        "evanescent": op.evanescent,
    }
    sp = op.domain.spacing()
    a0 = 0.0 if not isinstance(op.absorption, int | float) else float(op.absorption)
    obj_amp = math.exp(-a0) * op.illumination
    phase0 = (
        torch.zeros(shape, dtype=dtype, device=meas.device) if init_phase is None else init_phase
    )
    t = torch.polar(torch.full(shape, obj_amp, dtype=dtype, device=meas.device), phase0.to(dtype))
    if isinstance(op.absorption, str) and pure_phase:
        raise ConfigError("pure_phase=True needs a known (constant) absorption in the operator")
    fwd_all = op.at_resolution(shape)._transfer(meas.device, dtype)  # (n_z, P1, P2)
    bwd_all = fwd_all.conj()  # exact inverse on the (band-limited) propagating band
    history: list[float] = []
    target = meas.to(dtype)
    norm = torch.linalg.vector_norm(target).clamp_min(1e-30)
    for _ in range(int(n_iter)):
        for i, z in enumerate(op.distances):
            uz = propagate(t, [z], op.wavelength, sp, transfer=fwd_all[i : i + 1], **kw)[0]
            uz = torch.polar(amp[i], torch.angle(uz))
            t = propagate(uz, [-z], op.wavelength, sp, transfer=bwd_all[i : i + 1], **kw)[0]
            if pure_phase:
                t = torch.polar(torch.full_like(t.real, obj_amp), torch.angle(t))
        uz = propagate(t, list(op.distances), op.wavelength, sp, transfer=fwd_all, **kw)
        pred = uz.real**2 + uz.imag**2
        history.append(float(torch.linalg.vector_norm(pred - target) / norm))
    # fix the arbitrary global phase so that the mean transmission is real (avoids wrapping)
    m = t.mean()
    if float(m.abs()) > 0:
        t = t * (m.conj() / m.abs())
    return torch.angle(t), history
