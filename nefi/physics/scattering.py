"""2-D scalar diffraction tomography: Born / Rytov linearized scattering and Lippmann–Schwinger.

Physics and conventions (time dependence ``e^{−iωt}``; lengths in the user's unit, e.g. μm):

* Background wavenumber ``k0 = 2π n_b / λ`` (``λ`` vacuum wavelength, ``n_b`` background index).
* Object contrast ``χ(x) = n(x)²/n_b² − 1`` (dimensionless, the unknown of the instances) and
  scattering potential ``f(x) = k0² χ(x)`` (units 1/length²).
* Helmholtz equation ``Δu + k0² (1 + χ) u = 0`` ⇒ the total field ``u = u_inc + u_s`` solves the
  **Lippmann–Schwinger** equation ``u = u_inc + G ∗ (f u)`` with the outgoing free-space Green's
  function ``G(r) = (i/4) H₀⁽¹⁾(k0 r)``, ``(Δ + k0²) G = −δ``.
* **Born** approximation (first order in ``f``): ``u_s ≈ G ∗ (f u_inc)``; **Rytov**:
  ``φ = log(u/u_inc) ≈ u_s^{Born} / u_inc`` (first-order Rytov phase; better for large, smooth,
  weakly refracting objects). Both are linear in ``χ`` (``homogeneity = 1``). Kak & Slaney (1988),
  *Principles of Computerized Tomographic Imaging*, ch. 6; Devaney (1982), *Ultrasonic Imaging* 4.
* Illumination: plane waves ``u_inc = exp(i k0 d̂_j·x)``, ``d̂_j = (cos θ_j, sin θ_j)`` (axis 0 is
  ``x``, axis 1 is ``y``); receivers on a circle (default) or anywhere *outside* the object domain.

Numerics
--------
* **Receivers** are evaluated by direct quadrature with the analytic Hankel function (matrix
  precomputed once per resolution with ``scipy.special.hankel1``):
  ``u_s(x_r) = Σ_j G(x_r − y_j) q_j ΔA`` — exact up to the midpoint rule because receivers lie
  outside the domain (no singularity).
* **In-domain fields** (the multiple-scattering generator) use FFT convolution with zero padding.
  The naive k-space kernel ``Ĝ(k) = 1/(|k|² − k0² − iε)`` is *not* used by default: a small ``ε``
  leaves the periodic images of the FFT box undamped and under-resolves the singular circle
  ``|k| = k0``, a large ``ε`` damps the physical field (both give O(10 %) errors). Instead we use
  the **truncated Green's function** of Vico, Greengard & Ferrando (2016,
  *J. Comput. Phys.* 323:191): ``G_L = G·1{|x| < L}`` with ``L`` ≥ the domain diagonal has the
  smooth analytic transform

  .. math::

      \\hat G_L(s) = \\frac{1 + \\tfrac{iπ}{2} L\\,[\\,s J_1(Ls) H_0^{(1)}(k_0L)
                     - k_0 J_0(Ls) H_1^{(1)}(k_0L)\\,]}{s^2 - k_0^2}

  (removable singularity at ``s = k0``), sampled on a 4× oversampled FFT grid, transformed back,
  restricted to offsets within the 2× padded box and re-transformed — spectrally accurate free-space
  convolution of band-limited sources (agreement with ``(i/4)H₀⁽¹⁾`` to < 1 % at a few wavelengths,
  tested). ``method="regularized"`` keeps the naive kernel for comparison.
* The **Lippmann–Schwinger** equation is solved by (relaxed) fixed-point iteration, i.e. the Born
  series, which converges for the weak contrasts where diffraction tomography is meaningful.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
import torch

from ..domain import Domain
from ..errors import ConfigError, OperatorError, ShapeError
from ..operators.base import Fields, Operator
from ..registry import register
from ..utils.tensor import shape_tuple

log = logging.getLogger("nefi")

__all__ = [
    "BornOperator",
    "LippmannSchwingerOperator",
    "adjoint_reconstruction",
    "circle_receivers",
    "complex_dtype",
    "filtered_backpropagation",
    "green_2d",
    "green_convolve",
    "hankel1",
    "helmholtz_kernel_fft",
    "lippmann_schwinger",
    "plane_waves",
    "receiver_green_matrix",
]


def complex_dtype(dtype: torch.dtype) -> torch.dtype:
    """Complex dtype matching a real one (float64 → complex128, else complex64)."""
    return torch.complex128 if dtype == torch.float64 else torch.complex64


def hankel1(order: int, z: torch.Tensor) -> torch.Tensor:
    """Hankel function of the first kind ``H_ν⁽¹⁾(z) = J_ν(z) + i Y_ν(z)`` for ``ν ∈ {0, 1}``.

    Real ``z > 0``; uses ``torch.special`` Bessel functions (device-agnostic and differentiable in
    ``z``; ≈ 1e-6 relative accuracy even in float64 — the operators precompute their constant
    matrices with ``scipy.special`` instead).
    """
    z = torch.as_tensor(z)
    if order == 0:
        j, y = torch.special.bessel_j0(z), torch.special.bessel_y0(z)
    elif order == 1:
        j, y = torch.special.bessel_j1(z), torch.special.bessel_y1(z)
    else:
        raise ConfigError("hankel1 supports orders 0 and 1")
    return torch.complex(j, y)


def green_2d(r: torch.Tensor, k0: float) -> torch.Tensor:
    """Outgoing 2-D Helmholtz Green's function ``(i/4) H₀⁽¹⁾(k0 r)`` (``(Δ + k0²)G = −δ``)."""
    return 0.25j * hankel1(0, k0 * torch.as_tensor(r))


def plane_waves(coords: torch.Tensor, k0: float, angles: torch.Tensor) -> torch.Tensor:
    """``exp(i k0 d̂_j·x)`` for incidence angles ``θ_j``: ``(n_angles, *coords.shape[:-1])``."""
    ang = torch.as_tensor(angles, dtype=coords.dtype, device=coords.device)
    d = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)  # (n_angles, 2)
    phase = k0 * torch.einsum("...k,ak->a...", coords, d)
    return torch.polar(torch.ones_like(phase), phase)


def circle_receivers(n: int, radius: float, center: Sequence[float] = (0.0, 0.0)) -> torch.Tensor:
    """``n`` receivers equally spaced on a circle: ``(n, 2)`` (float64), angles ``2πm/n``."""
    phi = 2.0 * math.pi * torch.arange(n, dtype=torch.float64) / n
    return torch.stack(
        [center[0] + radius * torch.cos(phi), center[1] + radius * torch.sin(phi)], 1
    )


# --------------------------------------------------------------------------------------------
# FFT convolution with the free-space Green's function
# --------------------------------------------------------------------------------------------
def _kgrid(shape: Sequence[int], spacing: Sequence[float]) -> torch.Tensor:
    ks = [
        2.0 * math.pi * torch.fft.fftfreq(n, d=h, dtype=torch.float64)
        for n, h in zip(shape, spacing)
    ]
    kx, ky = torch.meshgrid(*ks, indexing="ij")
    return torch.sqrt(kx**2 + ky**2)


def _truncated_green_hat(s: torch.Tensor, k0: float, L: float) -> torch.Tensor:
    """Vico–Greengard–Ferrando transform of ``G·1{|x|<L}`` (float64 → complex128).

    Evaluated with ``scipy.special`` (full double precision) since it is a one-off precomputation.
    """
    from scipy import special as sp

    sn = s.numpy()
    h0, h1 = sp.hankel1(0, k0 * L), sp.hankel1(1, k0 * L)

    def num(x):
        return 1.0 + 0.5j * math.pi * L * (x * sp.j1(L * x) * h0 - k0 * sp.j0(L * x) * h1)

    den = sn**2 - k0**2
    # removable singularity at s = k0: evaluate exact / near hits slightly off the circle
    close = np.abs(den) < 1e-7 * k0**2
    s_eval = np.where(close, sn * (1.0 + 1e-5), sn)
    out = num(s_eval) / (s_eval**2 - k0**2)
    return torch.from_numpy(np.ascontiguousarray(out))


def helmholtz_kernel_fft(
    shape: Sequence[int],
    spacing: Sequence[float],
    k0: float,
    method: str = "truncated",
    eps: float | None = None,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """FFT of the discrete Green's-function convolution kernel for a zero-padded ``2N`` grid.

    ``green_convolve(q, K)`` then returns ``u_i = Σ_j g(x_i − x_j) q_j ≈ ∫ G(x_i − y) q(y) dy``
    (the cell area is included in the kernel).

    Args:
        shape: grid shape ``(N1, N2)`` of the sources/targets; ``spacing``: cell size.
        k0: background wavenumber.
        method: ``"truncated"`` (Vico et al. 2016, default) or ``"regularized"``
            (``1/(|k|² − k0² − iε)`` sampled directly on the ``2N`` grid; inaccurate, for
            comparison only).
        eps: imaginary regularization for ``"regularized"`` (default ``0.05 k0²``).
    """
    shape = shape_tuple(shape)
    if len(shape) != 2:
        raise ShapeError("helmholtz_kernel_fft is 2-D")
    n1, n2 = shape
    h1, h2 = (float(h) for h in spacing)
    if method == "regularized":
        s = _kgrid((2 * n1, 2 * n2), (h1, h2))
        e = 0.05 * k0**2 if eps is None else float(eps)
        khat = 1.0 / (s**2 - k0**2 - 1j * e)
        return khat.to(device=device, dtype=complex_dtype(dtype))
    if method != "truncated":
        raise ConfigError(f"unknown Green's kernel method {method!r}")
    L = math.hypot(n1 * h1, n2 * h2) * 1.01  # ≥ max source-target distance
    big = (4 * n1, 4 * n2)
    s = _kgrid(big, (h1, h2))
    ghat = _truncated_green_hat(s, k0, L)
    # ifft2 of the sampled transform = kernel samples g(x_j)·ΔA (FFT offset order): the
    # quadrature weight of u_i = Σ_j g(x_i − x_j) q_j ΔA is already included
    g = torch.fft.ifft2(ghat)
    # keep offsets in [-N, N) per axis -> 2N box (FFT order)
    rows = torch.cat([torch.arange(n1), torch.arange(4 * n1 - n1, 4 * n1)])
    cols = torch.cat([torch.arange(n2), torch.arange(4 * n2 - n2, 4 * n2)])
    small = g[rows][:, cols]
    return torch.fft.fft2(small).to(device=device, dtype=complex_dtype(dtype))


def green_convolve(q: torch.Tensor, kernel_fft: torch.Tensor) -> torch.Tensor:
    """``G ∗ q`` on the grid of ``q (*batch, N1, N2)`` with a kernel from
    :func:`helmholtz_kernel_fft` (zero-padded linear convolution, same-size output)."""
    n1, n2 = q.shape[-2:]
    m1, m2 = kernel_fft.shape[-2:]
    qhat = torch.fft.fft2(q, s=(m1, m2))
    return torch.fft.ifft2(qhat * kernel_fft)[..., :n1, :n2]


def lippmann_schwinger(
    f: torch.Tensor,
    u_inc: torch.Tensor,
    kernel_fft: torch.Tensor,
    n_iter: int = 100,
    tol: float = 1e-10,
    relax: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Solve ``u = u_inc + G ∗ (f u)`` by relaxed fixed-point iteration (the Born series).

    Args:
        f: scattering potential ``k0² χ`` on the grid ``(N1, N2)`` (real or complex).
        u_inc: incident fields ``(*batch, N1, N2)`` (complex).
        kernel_fft: from :func:`helmholtz_kernel_fft`.
        n_iter / tol: iteration cap and relative-update tolerance.
        relax: relaxation ``u ← (1−ω) u + ω (u_inc + G∗(f u))`` (ω < 1 stabilizes stronger
            scatterers).

    Returns:
        ``(u_total, {"iterations", "residual", "converged"})``.
    """
    u = u_inc.clone()
    res = float("inf")
    it = 0
    for _ in range(int(n_iter)):
        it += 1
        new = u_inc + green_convolve(f * u, kernel_fft)
        new = (1.0 - relax) * u + relax * new if relax != 1.0 else new
        res = float(
            torch.linalg.vector_norm(new - u) / torch.linalg.vector_norm(new).clamp_min(1e-30)
        )
        u = new
        if res < tol:
            break
    info = {"iterations": it, "residual": res, "converged": res < tol}
    if not info["converged"]:
        log.warning(
            "Lippmann-Schwinger iteration not converged after %d its (rel. update %.2e); the "
            "contrast may be too strong for the Born series (lower it or use relax < 1)",
            it,
            res,
        )
    return u, info


def receiver_green_matrix(
    receivers: torch.Tensor, coords: torch.Tensor, k0: float, cell_area: float
) -> torch.Tensor:
    """Quadrature matrix ``M_{r,j} = G(|x_r − y_j|) ΔA``: ``(n_rec, n_pixels)`` complex128.

    One-off precomputation with ``scipy.special.hankel1`` (full double precision).
    """
    from scipy.special import hankel1 as sp_hankel1

    rec = torch.as_tensor(receivers, dtype=torch.float64)
    pts = torch.as_tensor(coords, dtype=torch.float64).reshape(-1, 2)
    r = torch.cdist(rec, pts).numpy()
    if float(r.min()) <= 0:
        raise ConfigError("receivers must not coincide with grid points (place them outside)")
    return torch.from_numpy(0.25j * sp_hankel1(0, k0 * r) * cell_area)


# --------------------------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------------------------
class _ScatteringBase(Operator):
    """Geometry (grid, illumination, receivers) and caches shared by the scattering operators."""

    fidelity_tag = "scattering"

    def __init__(
        self,
        domain: Domain,
        wavelength: float,
        *,
        n_angles: int = 16,
        angles: Sequence[float] | torch.Tensor | None = None,
        receivers: torch.Tensor | None = None,
        n_receivers: int = 64,
        receiver_radius: float | None = None,
        field: str = "chi",
        contrast: str = "chi",
        background_index: float = 1.0,
        rytov: bool = False,
    ) -> None:
        super().__init__()
        if domain.ndim != 2:
            raise ShapeError("scattering operators are 2-D")
        if contrast not in ("chi", "potential"):
            raise ConfigError("contrast must be 'chi' (n²/n_b²−1) or 'potential' (k0² χ)")
        self.domain = domain
        self.wavelength = float(wavelength)
        self.background_index = float(background_index)
        self.k0 = 2.0 * math.pi * self.background_index / self.wavelength
        if angles is None:
            angles = 2.0 * math.pi * torch.arange(int(n_angles), dtype=torch.float64) / n_angles
        self.angles = torch.as_tensor(angles, dtype=torch.float64).reshape(-1)
        center = tuple(0.5 * (lo + hi) for lo, hi in domain.extent)
        self.center = center
        if receivers is None:
            if receiver_radius is None:
                receiver_radius = 0.75 * max(domain.size)
            receivers = circle_receivers(int(n_receivers), float(receiver_radius), center)
            self.receiver_radius: float | None = float(receiver_radius)
        else:
            self.receiver_radius = None
        self.receivers = torch.as_tensor(receivers, dtype=torch.float64).reshape(-1, 2)
        self.primary = field
        self.contrast = contrast
        self.rytov = bool(rytov)
        self.homogeneity = 1.0
        self._cache: dict = {}
        self._res: dict = {}
        self._check_receivers()

    def _check_receivers(self) -> None:
        lo = torch.tensor([e[0] for e in self.domain.extent], dtype=torch.float64)
        hi = torch.tensor([e[1] for e in self.domain.extent], dtype=torch.float64)
        inside = ((self.receivers > lo) & (self.receivers < hi)).all(-1)
        if bool(inside.any()):
            raise ConfigError(
                "receivers must lie outside the object domain "
                f"{self.domain.extent}; got {int(inside.sum())} inside"
            )

    @property
    def n_angles(self) -> int:
        return int(self.angles.numel())

    @property
    def n_receivers(self) -> int:
        return int(self.receivers.shape[0])

    def _geometry_kwargs(self) -> dict:
        return {
            "angles": self.angles,
            "receivers": self.receivers,
            "field": self.primary,
            "contrast": self.contrast,
            "background_index": self.background_index,
            "rytov": self.rytov,
        }

    def output_shape(self, shape):
        return (2, self.n_angles, self.n_receivers)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        """Scattering potential ``f = k0² χ`` from the field value."""
        return x * self.k0**2 if self.contrast == "chi" else x

    def _static(self, device, dtype) -> dict:
        key = (str(device), str(dtype))
        if key in self._cache:
            return self._cache[key]
        cdt = complex_dtype(dtype)
        coords = self.domain.physical_coords(dtype=torch.float64)
        area = math.prod(self.domain.spacing())
        m = receiver_green_matrix(self.receivers, coords, self.k0, area)
        st = {
            "M_T": m.T.contiguous().to(device=device, dtype=cdt),
            "u_inc": plane_waves(coords, self.k0, self.angles).to(device=device, dtype=cdt),
            "u_inc_rec": plane_waves(self.receivers, self.k0, self.angles).to(
                device=device, dtype=cdt
            ),
        }
        self._extra_static(st, device, dtype)
        self._cache[key] = st
        return st

    def _extra_static(self, st: dict, device, dtype) -> None:
        """Hook for subclasses."""

    def _measure(self, q: torch.Tensor, st: dict, u_total_rec=None) -> torch.Tensor:
        """Receiver data from induced sources ``q (*batch, n_angles, N1, N2)``:
        ``(*batch, 2, n_angles, n_rec)``."""
        us = q.flatten(-2) @ st["M_T"]
        if self.rytov:
            if u_total_rec is None:  # first-order Rytov phase
                us = us / st["u_inc_rec"]
            else:
                us = torch.log((st["u_inc_rec"] + us) / st["u_inc_rec"])
        return torch.stack([us.real, us.imag], dim=-3)

    def extra_repr(self) -> str:
        return (
            f"shape={self.domain.shape}, wavelength={self.wavelength}, k0={self.k0:.4g}, "
            f"n_angles={self.n_angles}, n_receivers={self.n_receivers}, rytov={self.rytov}"
        )


@register("operator", "born")
class BornOperator(_ScatteringBase):
    """Linearized (Born / first-order Rytov) 2-D diffraction-tomography forward model.

    ``χ(x)`` on the domain grid → scattered field at the receivers for every incidence angle,
    returned as stacked real/imaginary parts ``(2, n_angles, n_receivers)``.

    Args:
        domain: object domain (physical units, e.g. μm), 2-D.
        wavelength: vacuum wavelength (same unit); ``background_index`` ``n_b``.
        n_angles / angles: incidence directions (default ``n_angles`` equispaced over 2π).
        receivers: explicit receiver positions ``(n, 2)`` outside the domain; default
            ``n_receivers`` on a circle of radius ``receiver_radius`` (default 0.75 × the domain
            side) around the domain center.
        field: name of the unknown; ``contrast``: ``"chi"`` (field = ``n²/n_b² − 1``) or
            ``"potential"`` (field = ``k0² χ``).
        rytov: return the first-order Rytov phase ``u_s/u_inc`` instead of ``u_s``.
    """

    fidelity_tag = "born-1x"
    batchable = True  # leading batch axes: (*batch, N1, N2) -> (*batch, 2, n_angles, n_rec)

    def __init__(self, domain: Domain, wavelength: float, **kw) -> None:
        super().__init__(domain, wavelength, **kw)
        self.fidelity_tag = "rytov-1x" if self.rytov else "born-1x"

    def at_resolution(self, shape):
        shape = shape_tuple(shape)
        if shape == tuple(self.domain.shape):
            return self
        if shape not in self._res:
            op = BornOperator(self.domain.at(shape), self.wavelength, **self._geometry_kwargs())
            op.receiver_radius = self.receiver_radius
            self._res[shape] = op
        return self._res[shape]

    def scattered_field(self, x: torch.Tensor, kernel_method: str = "truncated") -> torch.Tensor:
        """Born scattered field *inside* the domain, ``(n_angles, N1, N2)`` complex.

        ``G ∗ (f u_inc)`` by zero-padded FFT convolution with the truncated (default) or naive
        regularized Green's kernel (:func:`helmholtz_kernel_fft`); for visualization and
        in-domain diagnostics (the measurement model uses the receiver quadrature).
        """
        st = self._static(x.device, x.dtype)
        key = f"K_{kernel_method}"
        if key not in st:
            st[key] = helmholtz_kernel_fft(
                self.domain.shape,
                self.domain.spacing(),
                self.k0,
                kernel_method,
                device=x.device,
                dtype=x.dtype,
            )
        return green_convolve(self.potential(x).unsqueeze(0) * st["u_inc"], st[key])

    def forward(self, fields: Fields) -> torch.Tensor:
        x = self.get_field(fields)
        if tuple(x.shape[-2:]) != tuple(self.domain.shape):
            raise ShapeError(f"field shape {tuple(x.shape)} != operator grid {self.domain.shape}")
        st = self._static(x.device, x.dtype)
        q = self.potential(x).unsqueeze(-3) * st["u_inc"]
        return self._measure(q, st)


@register("operator", "lippmann_schwinger")
class LippmannSchwingerOperator(_ScatteringBase):
    """Multiple-scattering forward model (independent data generator for diffraction tomography).

    Solves the Lippmann–Schwinger equation on the domain grid by FFT convolution with the truncated
    Green's function (:func:`helmholtz_kernel_fft`) and evaluates the scattered field of the *total*
    field at the receivers. With ``rytov=True`` it returns ``log(u/u_inc)`` (principal branch; valid
    for phase excursions below π, i.e. weak objects). Differentiable, but meant to run under
    ``torch.no_grad`` in float64 on a finer grid.

    Extra args: ``n_iter``, ``tol``, ``relax`` (see :func:`lippmann_schwinger`),
    ``kernel_method`` (``"truncated"`` | ``"regularized"``).
    """

    fidelity_tag = "lippmann-schwinger"
    traceable = False  # iterative solver (data generation)

    def __init__(
        self,
        domain: Domain,
        wavelength: float,
        *,
        n_iter: int = 200,
        tol: float = 1e-10,
        relax: float = 1.0,
        kernel_method: str = "truncated",
        **kw,
    ) -> None:
        super().__init__(domain, wavelength, **kw)
        self.n_iter, self.tol, self.relax = int(n_iter), float(tol), float(relax)
        self.kernel_method = kernel_method
        self.last_info: dict = {}

    def _extra_static(self, st, device, dtype):
        st["K"] = helmholtz_kernel_fft(
            self.domain.shape,
            self.domain.spacing(),
            self.k0,
            self.kernel_method,
            device=device,
            dtype=dtype,
        )

    def at_resolution(self, shape):
        shape = shape_tuple(shape)
        if shape == tuple(self.domain.shape):
            return self
        if shape not in self._res:
            op = LippmannSchwingerOperator(
                self.domain.at(shape),
                self.wavelength,
                n_iter=self.n_iter,
                tol=self.tol,
                relax=self.relax,
                kernel_method=self.kernel_method,
                **self._geometry_kwargs(),
            )
            op.fidelity_tag = self.fidelity_tag
            self._res[shape] = op
        return self._res[shape]

    def total_field(self, x: torch.Tensor) -> torch.Tensor:
        """Total field ``u`` on the grid for every incidence ``(n_angles, N1, N2)``."""
        st = self._static(x.device, x.dtype)
        u, info = lippmann_schwinger(
            self.potential(x), st["u_inc"], st["K"], self.n_iter, self.tol, self.relax
        )
        self.last_info = info
        return u

    def forward(self, fields: Fields) -> torch.Tensor:
        x = self.get_field(fields)
        if tuple(x.shape) != tuple(self.domain.shape):
            raise ShapeError(f"field shape {tuple(x.shape)} != operator grid {self.domain.shape}")
        st = self._static(x.device, x.dtype)
        u = self.total_field(x)
        q = self.potential(x).unsqueeze(0) * u
        return self._measure(q, st, u_total_rec=True if self.rytov else None)


# --------------------------------------------------------------------------------------------
# classical reconstructions
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def filtered_backpropagation(
    data: torch.Tensor,
    op: _ScatteringBase,
    shape: Sequence[int] | None = None,
    n_harmonics: int | None = None,
) -> torch.Tensor:
    """Filtered backpropagation (Devaney 1982) for full-view circular-receiver Born/Rytov data.

    Steps (all exact for Born data on a closed circle of radius ``R``):

    1. near- to far-field: ``u_s(R, φ) = Σ_n α_n H_n⁽¹⁾(k0 R) e^{inφ}`` → far-field pattern
       ``u_∞(φ) = Σ_n α_n (−i)^n e^{inφ}`` (``u_s ≈ sqrt(2/(πk0 r)) e^{i(k0 r − π/4)} u_∞``);
    2. Fourier diffraction theorem: ``F̂(k0(r̂ − d̂)) = −4i u_∞(φ)`` (``F̂`` = Fourier transform of
       ``f = k0² χ``);
    3. back-propagation with the Jacobian filter ``|sin(φ − θ)|`` of the map
       ``(θ, φ) → K = k0(r̂ − d̂)`` (which covers the Ewald disk ``|K| ≤ 2 k0`` twice)::

           f(x) ≈ k0²/(8π²) Σ_{θ,φ} F̂(K) e^{iK·x} |sin(φ − θ)| Δθ Δφ.

    The result is ``χ`` low-passed to ``|K| ≤ 2k0`` (resolution ≈ λ/4 half-period). Rytov data are
    first converted to the Born field via ``u_s = φ u_inc``.

    Args:
        data: ``(2, n_angles, n_receivers)`` real/imag measurement of ``op``'s geometry.
        op: the operator that defines the geometry (circle receivers, equispaced angles).
        shape: output grid (default ``op.domain.shape``).
        n_harmonics: max circular harmonic ``|n|`` (default: all resolvable, ``n_rec/2``).

    Returns:
        ``χ`` estimate on the domain grid (float, on ``data``'s device).
    """
    from scipy.special import hankel1 as sp_hankel1

    if op.receiver_radius is None:
        raise ConfigError("filtered_backpropagation needs circular receivers (receiver_radius)")
    f64, c128 = torch.float64, torch.complex128
    d = torch.as_tensor(data).detach().to("cpu", f64)
    u = torch.complex(d[0], d[1])  # (n_angles, n_rec)
    k0, R = op.k0, float(op.receiver_radius)
    rec = op.receivers.to(f64)
    c = torch.tensor(op.center, dtype=f64)
    phi = torch.atan2(rec[:, 1] - c[1], rec[:, 0] - c[0])
    theta = op.angles.to(f64)
    if op.rytov:
        u = u * plane_waves(rec, k0, theta)
    m = u.shape[1]
    nmax = m // 2 - 1 if n_harmonics is None else int(n_harmonics)
    ns = torch.arange(-nmax, nmax + 1, dtype=f64)
    # circular-harmonic coefficients (least squares on the receiver angles)
    basis = torch.polar(torch.ones(m, ns.numel(), dtype=f64), torch.outer(phi, ns))
    coef = torch.linalg.lstsq(basis, u.T.contiguous()).solution  # (n_h, n_angles)
    hn = torch.as_tensor(sp_hankel1(ns.numpy(), k0 * R), dtype=c128)
    alpha = coef / hn[:, None]
    phase = torch.polar(torch.ones_like(ns), -0.5 * math.pi * ns)  # (-i)^n
    u_inf = (basis * phase[None, :]) @ alpha  # (n_rec, n_angles)
    fhat = -4j * u_inf.T  # (n_angles, n_rec)
    # back-propagation onto the grid (coordinates relative to the domain center)
    dom = op.domain if shape is None else op.domain.at(shape)
    xy = dom.physical_coords(dtype=f64) - c
    kx = k0 * (torch.cos(phi)[None, :] - torch.cos(theta)[:, None])  # (n_angles, n_rec)
    ky = k0 * (torch.sin(phi)[None, :] - torch.sin(theta)[:, None])
    dtheta = 2.0 * math.pi / theta.numel()
    dphi = 2.0 * math.pi / m
    w = torch.abs(torch.sin(phi[None, :] - theta[:, None])) * dtheta * dphi * fhat
    # the operator uses absolute coordinates: move the phase reference to the center
    w = w * torch.polar(torch.ones_like(kx), kx * c[0] + ky * c[1])
    img = torch.zeros(xy.shape[:-1], dtype=c128)
    for j in range(theta.numel()):
        ang = xy[..., 0, None] * kx[j] + xy[..., 1, None] * ky[j]
        img = img + torch.polar(torch.ones_like(ang), ang) @ w[j]
    f = (k0**2 / (8.0 * math.pi**2)) * img.real
    chi = f / k0**2 if op.contrast == "chi" else f
    return torch.as_tensor(chi, dtype=torch.float32).to(torch.as_tensor(data).device)


def adjoint_reconstruction(op: Operator, data: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Back-projection ``Aᵀ d`` of a linear operator via autograd (unnormalized)."""
    x = torch.zeros(tuple(shape), dtype=data.dtype, device=data.device, requires_grad=True)
    with torch.enable_grad():
        (g,) = torch.autograd.grad((op({op.primary: x}) * data).sum(), x)
    return g


_ = OperatorError  # re-exported error type used in docs
