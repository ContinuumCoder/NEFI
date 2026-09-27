"""Differentiable parallel-beam Radon transform, back-projection and filtered back-projection.

Forward model (rotation-based, cf. ``skimage.transform.radon``): for every view angle ``θ`` the
image is resampled on a rotated grid with ``F.grid_sample`` (bilinear, ``align_corners=False``,
zero outside the field of view) and summed along the ray direction::

    p_θ(u_b) = Δt Σ_a f(R_θ (u_b, v_a)),     R_θ = [[cos θ, -sin θ], [sin θ, cos θ]]

in normalized coordinates (axis 1 = x = grid_sample width, axis 0 = y = height), so the detector
axis points along ``(cos θ, sin θ)``, ``θ = 0`` gives column sums, and ``Δt`` is the physical
ray step, i.e. the sinogram holds physical line integrals. Views are vectorized in one
``grid_sample`` call; autograd provides the exact adjoint (``RadonOperator.adjoint``).

:func:`backproject` is the classical *pixel-driven* back-projector (1-D linear interpolation of
each projection at ``u = x cos θ + y sin θ``), an independent discretization of ``Rᵀ`` that agrees
with the exact adjoint up to interpolation error; :func:`fbp` is filtered back-projection with the
Kak & Slaney discrete ramp filter (Kak & Slaney 1988, Eq. 61-62) and optional apodization windows.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from ...domain import Domain
from ...errors import ConfigError, OperatorError, ShapeError
from ...operators.base import Fields, Operator
from ...registry import register
from ...utils.tensor import next_fast_len, shape_tuple

FILTERS = ("ramp", "shepp-logan", "cosine", "hann")


def uniform_angles(n_views: int, angle_range: float = math.pi) -> torch.Tensor:
    """``n_views`` equispaced angles in ``[0, angle_range)`` (radians, float64)."""
    return torch.arange(int(n_views), dtype=torch.float64) * (float(angle_range) / int(n_views))


def _rotation_grid(angles: torch.Tensor, rows: int, cols: int, device, dtype) -> torch.Tensor:
    """Sampling grid ``(V, rows, cols, 2)`` of the rotated detector/ray lattice."""
    a = angles.detach().to("cpu", torch.float64)  # trig in float64 on the host (MPS: no fp64)
    c, s = torch.cos(a), torch.sin(a)
    z = torch.zeros_like(a)
    theta = torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1)], -2)
    return F.affine_grid(
        theta.to(device=device, dtype=dtype), size=(a.numel(), 1, rows, cols), align_corners=False
    )


def _smear(sino: torch.Tensor, angles: torch.Tensor, n: int) -> torch.Tensor:
    """``Σ_θ interp(sino_θ, x cos θ + y sin θ)`` on an ``n × n`` grid (no scaling)."""
    v, n_det = sino.shape
    ax = -1.0 + (2.0 * torch.arange(n, dtype=torch.float64) + 1.0) / n
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")  # rows ↔ y (axis 0), cols ↔ x (axis 1)
    a = angles.detach().to("cpu", torch.float64)
    u = torch.cos(a)[:, None, None] * xx + torch.sin(a)[:, None, None] * yy  # (V, n, n)
    grid = torch.stack([u, torch.zeros_like(u)], -1).to(device=sino.device, dtype=sino.dtype)
    out = F.grid_sample(
        sino.reshape(v, 1, 1, n_det),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return out[:, 0].sum(0)


def backproject(
    sinogram: torch.Tensor,
    angles: torch.Tensor,
    n: int | None = None,
    width: float = 1.0,
    samples_per_pixel: float = 1.0,
) -> torch.Tensor:
    """Pixel-driven back-projection, scaled to approximate the adjoint of :class:`RadonOperator`.

    ``Rᵀy(x) ≈ Δt · (n_t n_det / n²) · Σ_θ y_θ(x cos θ + y sin θ)`` where ``Δt = width / n_t``.

    Args:
        sinogram: ``(n_views, n_det)``.
        angles: view angles (radians).
        n: output image size (default ``n_det``).
        width: physical side length of the (square) image / detector span.
        samples_per_pixel: ray samples per pixel of the forward operator (``n_t / n``).
    """
    v, n_det = sinogram.shape
    n = n_det if n is None else int(n)
    n_t = max(1, round(samples_per_pixel * n))
    scale = (width / n_t) * (n_t * n_det) / float(n * n)
    return _smear(sinogram, torch.as_tensor(angles), n) * scale


def ramp_filter(sinogram: torch.Tensor, filter: str = "ramp", pad_factor: int = 2) -> torch.Tensor:
    """Convolve each projection with the Kak-Slaney ramp at unit bin spacing (± window).

    Returns ``h₁ ⊛ p`` per view; divide by the bin width ``τ`` for physical units.
    """
    if filter not in FILTERS:
        raise ConfigError(f"unknown FBP filter {filter!r}; known: {FILTERS}")
    v, n_det = sinogram.shape
    size = next_fast_len(max(64, pad_factor * n_det))
    k = torch.arange(size, dtype=torch.float64)  # filter built on the host in float64
    off = torch.where(k <= size // 2, k, k - size)
    h = torch.zeros(size, dtype=torch.float64)
    h[0] = 0.25
    odd = (off.abs() % 2) == 1
    h[odd] = -1.0 / (math.pi**2 * off[odd] ** 2)
    H = torch.fft.rfft(h).real
    f = torch.fft.rfftfreq(size, dtype=torch.float64)  # cycles/sample
    if filter == "shepp-logan":
        H = H * torch.special.sinc(f)
    elif filter == "cosine":
        H = H * torch.cos(math.pi * f)
    elif filter == "hann":
        H = H * 0.5 * (1.0 + torch.cos(2.0 * math.pi * f))
    P = torch.fft.rfft(sinogram, n=size, dim=-1)
    q = torch.fft.irfft(P * H.to(device=sinogram.device, dtype=sinogram.dtype), n=size, dim=-1)
    return q[:, :n_det]


def fbp(
    sinogram: torch.Tensor,
    angles: torch.Tensor | Sequence[float],
    filter: str = "ramp",
    *,
    n: int | None = None,
    width: float = 1.0,
    circle: bool = False,
) -> torch.Tensor:
    """Filtered back-projection ``f ≈ (π / V) Σ_θ q_θ(x cos θ + y sin θ)``, ``q = (h ⊛ p) / τ``.

    Args:
        sinogram: ``(n_views, n_det)`` physical line integrals (detector spans the image width).
        angles: view angles in radians (assumed to cover ``[0, π)`` roughly uniformly).
        filter: ``"ramp"`` | ``"shepp-logan"`` | ``"cosine"`` | ``"hann"``.
        n: output size (default ``n_det``).
        width: physical side length of the image.
        circle: zero the reconstruction outside the inscribed circle.
    """
    sinogram = torch.as_tensor(sinogram)
    if sinogram.ndim != 2:
        raise ShapeError(f"fbp expects a (n_views, n_det) sinogram, got {tuple(sinogram.shape)}")
    v, n_det = sinogram.shape
    n = n_det if n is None else int(n)
    tau = width / n_det
    q = ramp_filter(sinogram, filter) / tau
    img = _smear(q, torch.as_tensor(angles), n) * (math.pi / v)
    if circle:
        ax = -1.0 + (2.0 * torch.arange(n, device=img.device, dtype=img.dtype) + 1.0) / n
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        img = img * ((xx**2 + yy**2) <= 1.0).to(img.dtype)
    return img


@register("operator", "radon")
class RadonOperator(Operator):
    """Parallel-beam Radon transform ``μ ↦ sinogram (n_views, n_det)`` (differentiable, linear).

    Args:
        domain: square 2-D domain (square pixels) of the attenuation field.
        n_views: number of equispaced views over ``angle_range`` (ignored if ``angles`` given).
        field: name of the attenuation field.
        angles: explicit view angles (radians).
        angle_range: angular coverage (radians), ``π`` for a full parallel-beam scan.
        det_per_pixel: detector bins per image pixel (``n_det = round(det_per_pixel · n)``);
            :meth:`at_resolution` keeps this ratio, so ``n_det`` scales with the grid.
        samples_per_pixel: ray samples per pixel along each ray.
    """

    homogeneity = 1.0
    batchable = True  # leading axes are batch (evaluated per image)

    def __init__(
        self,
        domain: Domain,
        n_views: int = 20,
        field: str = "mu",
        angles: torch.Tensor | Sequence[float] | None = None,
        angle_range: float = math.pi,
        det_per_pixel: float = 1.0,
        samples_per_pixel: float = 1.0,
    ) -> None:
        super().__init__()
        if domain.ndim != 2 or domain.shape[0] != domain.shape[1]:
            raise OperatorError(f"RadonOperator needs a square 2-D grid, got {domain.shape}")
        w0, w1 = domain.size
        if abs(w0 - w1) > 1e-9 * max(w0, w1):
            raise OperatorError(f"RadonOperator needs square physical extents, got {domain.size}")
        self.domain = domain
        self.primary = field
        if angles is None:
            angles = uniform_angles(n_views, angle_range)
        # host-side float64 (only used for trigonometry; not a device buffer, so .to("mps") works)
        self.angles = torch.as_tensor(angles, dtype=torch.float64).detach().cpu().clone()
        self.angle_range = float(angle_range)
        self.n = int(domain.shape[0])
        self.width = float(w0)
        self.det_per_pixel = float(det_per_pixel)
        self.samples_per_pixel = float(samples_per_pixel)
        self.n_det = max(1, round(self.det_per_pixel * self.n))
        self.n_t = max(1, round(self.samples_per_pixel * self.n))
        self.fidelity_tag = (
            f"radon-{self.n}px-{self.det_per_pixel:g}det-{self.samples_per_pixel:g}spp"
        )
        self._cache: dict[tuple, torch.Tensor] = {}

    @property
    def n_views(self) -> int:
        return int(self.angles.numel())

    @property
    def dt(self) -> float:
        """Physical ray step Δt."""
        return self.width / self.n_t

    @property
    def tau(self) -> float:
        """Physical detector bin width."""
        return self.width / self.n_det

    def _grid(self, device, dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        if key not in self._cache:
            self._cache[key] = _rotation_grid(self.angles, self.n_t, self.n_det, device, dtype)
        return self._cache[key]

    def forward(self, fields: Fields) -> torch.Tensor:
        x = self.get_field(fields)
        if tuple(x.shape[-2:]) != (self.n, self.n):
            raise ShapeError(
                f"RadonOperator built for {self.n}x{self.n} got field {tuple(x.shape)}; "
                "use at_resolution()"
            )
        batch = tuple(x.shape[:-2])
        xb = x.reshape(-1, 1, self.n, self.n)
        grid = self._grid(x.device, x.dtype)
        v = self.n_views
        outs = []
        for b in range(xb.shape[0]):
            img = xb[b : b + 1].expand(v, 1, self.n, self.n)
            s = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            outs.append(s[:, 0].sum(dim=1) * self.dt)  # (V, n_det)
        return torch.stack(outs).reshape(*batch, v, self.n_det)

    def at_resolution(self, shape: Sequence[int]) -> RadonOperator:
        shape = shape_tuple(shape)
        if shape == tuple(self.domain.shape):
            return self
        return RadonOperator(
            self.domain.at(shape),
            field=self.primary,
            angles=self.angles,
            angle_range=self.angle_range,
            det_per_pixel=self.det_per_pixel,
            samples_per_pixel=self.samples_per_pixel,
        )

    def output_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        n = shape_tuple(shape)[0]
        return (self.n_views, max(1, round(self.det_per_pixel * n)))

    # ---- adjoint-type maps ------------------------------------------------------------
    def adjoint(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Exact adjoint ``Rᵀ y`` of this discretization (autograd VJP; the operator is linear)."""
        x = torch.zeros(
            self.n, self.n, device=sinogram.device, dtype=sinogram.dtype, requires_grad=True
        )
        with torch.enable_grad():
            y = self({self.primary: x})
            (g,) = torch.autograd.grad(y, x, grad_outputs=sinogram)
        return g.detach()

    def backproject(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Pixel-driven back-projection ≈ ``Rᵀ y`` (independent discretization)."""
        return backproject(sinogram, self.angles, self.n, self.width, self.samples_per_pixel)

    def fbp(self, sinogram: torch.Tensor, filter: str = "ramp", circle: bool = False):
        """Filtered back-projection onto this operator's grid."""
        return fbp(sinogram, self.angles, filter, n=self.n, width=self.width, circle=circle)


__all__ = [
    "FILTERS",
    "RadonOperator",
    "backproject",
    "fbp",
    "ramp_filter",
    "uniform_angles",
]
