"""NV relaxometry forward operators F1 / F2 (FFT-factorized) and F3 (direct, source-side).

* :class:`NVOperator` (``mode="F2"``, default) — tensor/incoherent operator of NeTMY Eq. (2)::

      F2(ρ, ω_L)(ω, r) = (P ∗ ρ)(r) · L(ω; ω_L(r)),     P = Σ_a |G_az|²

  ``mode="F1"`` — scalar/coherent operator ``F1 = ((G_zz ∗ ρ)(r))² · L(ω; ω_L(r))``.
  Both are built on :class:`~nefi.operators.FFTConvolution` (zero-padded linear convolution with
  the full ``2n − 1`` kernel extent), so they are exact discrete convolutions at every curriculum
  resolution (NeTMY App. D.3 "precomputed FFT kernel cache").
* :class:`NVDirectSimulator` — the source-side direct simulator F3 of NeTMY Eq. (11)::

      F3(ρ, ω_L)(ω, r) = Σ_{r_src} ρ(r_src) P(R(r, r_src)) L(ω; ω_L(r_src))

  evaluated in float64 with dense ``(n_pix × n_src)`` kernel matrices (no FFT factorization, the
  Lorentzian at the *source* pixel). It is used only for data generation (inverse-crime guard,
  NeTMY §3.1 / App. A.4).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ...domain import Domain
from ...errors import ConfigError, ShapeError
from ...operators.base import Fields, Operator
from ...operators.conv import FFTConvolution
from ...registry import register
from ...utils.tensor import shape_tuple
from .physics import DipolarKernels, lorentzian

MODES = ("F1", "F2")


def _as_freqs(freqs: torch.Tensor | Sequence[float]) -> torch.Tensor:
    f = torch.as_tensor(freqs)
    if not f.is_floating_point():
        f = f.to(torch.get_default_dtype())
    f = f.detach().reshape(-1).clone()
    if f.numel() < 1:
        raise ConfigError("the frequency grid needs at least one frequency")
    return f


@register("operator", "nv_relaxometry")
class NVOperator(Operator):
    """FFT-factorized NV relaxometry operator (NeTMY Eq. 2): ``{rho, omega_L} -> S(ω, r)``.

    Args:
        domain: field domain (2-D; physical lengths in the same units as ``z0``).
        freqs: readout frequencies ``W`` (e.g. GHz), shape ``(n_freq,)``.
        z0: NV standoff height (domain length units).
        gamma: Lorentzian linewidth (frequency units), NeTMY App. A.7 (0.5 GHz).
        mode: ``"F2"`` (tensor power-summed, linear in ρ) or ``"F1"`` (scalar coherent square,
            quadratic in ρ).
        units: kernel units, ``"normalized"`` or ``"physical"`` (see :mod:`.physics`).
        length_scale: metres per domain length unit (physical units only).
        nv_axis: NV quantization axis (NeTMY: ``(0, 0, 1)``).
        rho_field / larmor_field: names of the density and Larmor fields.

    Output shape is ``(n_freq, H, W)`` for fields sampled on ``(H, W)`` (``(B, n_freq, H, W)``
    for fields with a leading batch axis, see :attr:`~nefi.operators.Operator.batchable`).
    ``homogeneity`` is 1 for F2 and 2 for F1, which makes
    :class:`~nefi.solve.EnergyScaleCorrection` apply the linear / square-root rule of NeTMY
    Eq. (30) automatically.
    """

    batchable = True

    def __init__(
        self,
        domain: Domain,
        freqs: torch.Tensor | Sequence[float],
        z0: float = 20.0,
        gamma: float = 0.5,
        mode: str = "F2",
        units: str = "normalized",
        length_scale: float = 1e-9,
        nv_axis: Sequence[float] = (0.0, 0.0, 1.0),
        rho_field: str = "rho",
        larmor_field: str = "omega_L",
    ) -> None:
        super().__init__()
        mode = str(mode).upper()
        if mode not in MODES:
            raise ConfigError(f"NVOperator mode must be one of {MODES}, got {mode!r}")
        if domain.ndim != 2:
            raise ShapeError(f"NVOperator needs a 2-D domain, got shape {domain.shape}")
        self.domain = domain
        self.mode = mode
        self.z0, self.gamma = float(z0), float(gamma)
        self.units, self.length_scale = units, float(length_scale)
        self.nv_axis = tuple(float(v) for v in nv_axis)
        self.kernels = DipolarKernels(self.z0, units, self.length_scale, self.nv_axis)
        self.primary = rho_field
        self.larmor_field = larmor_field
        self.homogeneity = 1.0 if mode == "F2" else 2.0
        self.register_buffer("freqs", _as_freqs(freqs), persistent=False)
        kernel = self.kernels.power_fn() if mode == "F2" else self.kernels.nv_fn()
        self.conv = FFTConvolution(
            kernel,
            domain,
            field=rho_field,
            periodic=False,
            kernel_extent="full",
            post=torch.square if mode == "F1" else None,
            homogeneity=self.homogeneity,
        )

    @property
    def n_freq(self) -> int:
        return int(self.freqs.numel())

    @property
    def fidelity_tag(self) -> str:
        """Physics/discretization tag for the benchmark inverse-crime guard (``"F2-fft"``)."""
        return f"{self.mode}-fft"

    def spatial_response(self, fields: Fields) -> torch.Tensor:
        """Frequency-independent spatial factor: ``P ∗ ρ`` (F2) or ``(G_zz ∗ ρ)²`` (F1)."""
        return self.conv({self.primary: self.get_field(fields)})

    def lorentzian(self, omega_L: torch.Tensor) -> torch.Tensor:
        """``L(ω; ω_L(r))`` at the readout pixels, shape ``(n_freq, *omega_L.shape)``."""
        f = self.freqs.to(device=omega_L.device, dtype=omega_L.dtype)
        return lorentzian(f.view(-1, *([1] * omega_L.ndim)), omega_L, self.gamma)

    def forward(self, fields: Fields) -> torch.Tensor:
        rho = self.get_field(fields)
        omega = self.get_field(fields, self.larmor_field)
        if rho.shape != omega.shape or rho.ndim < 2:
            raise ShapeError(
                f"NVOperator expects 2-D fields of equal shape (optionally with leading batch "
                f"axes); got rho {tuple(rho.shape)} and {self.larmor_field} {tuple(omega.shape)}"
            )
        spatial = self.spatial_response(fields)  # (*batch, H, W)
        f = self.freqs.to(device=omega.device, dtype=omega.dtype)
        # (*batch, n_freq, H, W); for unbatched fields identical to ``self.lorentzian(omega)``
        lor = lorentzian(f.view(-1, 1, 1), omega.unsqueeze(-3), self.gamma)
        return spatial.unsqueeze(-3) * lor

    def at_resolution(self, shape: Sequence[int]) -> NVOperator:
        shape = shape_tuple(shape)
        if shape == self.domain.shape:
            return self
        return NVOperator(
            self.domain.at(shape),
            self.freqs,
            self.z0,
            self.gamma,
            self.mode,
            self.units,
            self.length_scale,
            self.nv_axis,
            self.primary,
            self.larmor_field,
        )

    def output_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        return (self.n_freq, *shape_tuple(shape))

    def required_fields(self) -> tuple[str, ...]:
        return (self.primary, self.larmor_field)

    def kernel(self, shape: Sequence[int] | None = None, device=None, dtype=None) -> torch.Tensor:
        """The sampled convolution kernel (``P`` or ``G_zz``) for a field grid of ``shape``."""
        return self.conv.kernel(shape, device=device, dtype=dtype)

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode}, shape={self.domain.shape}, z0={self.z0}, gamma={self.gamma}, "
            f"n_freq={self.n_freq}, units={self.units}"
        )


@register("operator", "nv_direct")
class NVDirectSimulator(Operator):
    """Source-side direct simulator F3 (NeTMY Eq. 11), float64, no FFT factorization.

    ``S(ω, r) = Σ_s ρ(s) P(r − s, z0) L(ω; ω_L(s))`` computed as dense matmuls
    ``K @ (ρ ⊙ L)`` where ``K[r, s] = P(r − s)`` is evaluated directly from pixel coordinates.
    Only pixels with ``ρ ≠ 0`` are used as sources (sparse scenes are cheap); sources and
    frequencies are processed in chunks to bound memory (``K`` block: ``n_pix × src_chunk``).

    Used only by the data generator (inverse-crime guard): it differs from the inversion operator
    F2 by where the Lorentzian is evaluated (source vs readout pixel), by the summation algorithm
    (direct vs FFT) and by precision (float64 vs float32). For spatially constant ``ω_L`` it
    coincides with F2 up to round-off (Eq. 9).

    Args:
        domain: field domain (2-D).
        freqs: readout frequencies.
        z0, gamma, units, length_scale, nv_axis: as in :class:`NVOperator`.
        src_chunk: number of source pixels per kernel block.
        freq_chunk: number of frequencies per matmul.
        compute_dtype: precision of the direct sum (float64).
    """

    fidelity_tag = "F3-direct-float64"

    def __init__(
        self,
        domain: Domain,
        freqs: torch.Tensor | Sequence[float],
        z0: float = 20.0,
        gamma: float = 0.5,
        units: str = "normalized",
        length_scale: float = 1e-9,
        nv_axis: Sequence[float] = (0.0, 0.0, 1.0),
        rho_field: str = "rho",
        larmor_field: str = "omega_L",
        src_chunk: int = 1024,
        freq_chunk: int = 16,
        compute_dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if domain.ndim != 2:
            raise ShapeError(f"NVDirectSimulator needs a 2-D domain, got shape {domain.shape}")
        self.domain = domain
        self.z0, self.gamma = float(z0), float(gamma)
        self.kernels = DipolarKernels(self.z0, units, float(length_scale), tuple(nv_axis))
        self.primary, self.larmor_field = rho_field, larmor_field
        self.homogeneity = 1.0
        self.src_chunk, self.freq_chunk = max(1, int(src_chunk)), max(1, int(freq_chunk))
        self.compute_dtype = compute_dtype
        self.register_buffer("freqs", _as_freqs(freqs), persistent=False)

    @property
    def n_freq(self) -> int:
        return int(self.freqs.numel())

    def forward(self, fields: Fields) -> torch.Tensor:
        rho_in = self.get_field(fields)
        omega_in = self.get_field(fields, self.larmor_field)
        if rho_in.shape != omega_in.shape or rho_in.ndim != 2:
            raise ShapeError(
                f"NVDirectSimulator expects 2-D fields of equal shape; got {tuple(rho_in.shape)} "
                f"and {tuple(omega_in.shape)}"
            )
        dt, dev = self.compute_dtype, rho_in.device
        shape = tuple(rho_in.shape)
        rho = rho_in.to(dt).reshape(-1)
        omega = omega_in.to(dt).reshape(-1)
        pix = self.domain.at(shape).physical_coords(device=dev, dtype=dt).reshape(-1, 2)
        freqs = self.freqs.to(device=dev, dtype=dt)
        out = torch.zeros(self.n_freq, pix.shape[0], device=dev, dtype=dt)
        src = torch.nonzero(rho != 0).reshape(-1)
        for s0 in range(0, src.numel(), self.src_chunk):
            idx = src[s0 : s0 + self.src_chunk]
            # K[r, s] = P(r - s): (n_pix, n_chunk)
            K = self.kernels.power(pix[:, None, :] - pix[idx][None, :, :])
            for f0 in range(0, self.n_freq, self.freq_chunk):
                f = freqs[f0 : f0 + self.freq_chunk]
                # source-side Lorentzian: (n_chunk, n_f)
                w = rho[idx, None] * lorentzian(f[None, :], omega[idx, None], self.gamma)
                out[f0 : f0 + f.numel()] += (K @ w).T
        return out.reshape(self.n_freq, *shape).to(rho_in.dtype)

    def output_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        return (self.n_freq, *shape_tuple(shape))

    def required_fields(self) -> tuple[str, ...]:
        return (self.primary, self.larmor_field)

    def extra_repr(self) -> str:
        return f"F3 direct, z0={self.z0}, gamma={self.gamma}, n_freq={self.n_freq}"


__all__ = ["MODES", "DipolarKernels", "NVDirectSimulator", "NVOperator"]
