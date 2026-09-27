"""Differentiable 3-D parallel-beam Radon transform (rotation about ``z``) and slice-wise FBP.

Geometry: the source–detector pair rotates about the ``z`` axis (the last grid axis), so every
axial slice ``μ(·, ·, z_k)`` is projected by the 2-D parallel-beam Radon transform of
:mod:`nefi.instances.sparse_view_ct.radon` with the same conventions (a slice ``f[i, j]`` has axis
0 = grid-sample height and axis 1 = width, the detector axis is ``(cos θ, sin θ)``, ``θ = 0`` gives
column sums, the sinogram holds **physical line integrals** ``Δt Σ_a μ``)::

    p_θ(u_b, z_k) = Δt Σ_a μ(R_θ (u_b, v_a), z_k)

All slices and all views are evaluated in **one** ``F.grid_sample`` call: the slices are the
channels of a single ``(n_z, n, n)`` image and the rotated ``(n_t, n_det)`` sampling lattices of all
views are stacked along the output height, so the cost is one gather of ``n_z · V · n_t · n_det``
bilinear samples (no Python loop over slices or views; ``view_batch`` chunks the views to bound
memory at paper scale). Output: the sinogram stack ``(V, n_det, n_z)``; leading batch axes are
kept. Autograd gives the exact adjoint (:meth:`Radon3DOperator.adjoint`).

:func:`fbp3d` is slice-wise filtered back-projection (Kak & Slaney 1988, Eq. 61–62 ramp filter
from :func:`~nefi.instances.sparse_view_ct.radon.ramp_filter`), batched over slices the same way;
it equals :func:`~nefi.instances.sparse_view_ct.radon.fbp` applied to every slice.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from ...domain import Domain
from ...errors import OperatorError, ShapeError
from ...operators.base import Fields, Operator
from ...registry import register
from ...utils.tensor import shape_tuple
from ..sparse_view_ct.radon import FILTERS, _rotation_grid, ramp_filter, uniform_angles

__all__ = [
    "FILTERS",
    "Radon3DOperator",
    "backproject3d",
    "fbp3d",
    "uniform_angles",
]


def _smear3d(q: torch.Tensor, angles: torch.Tensor, n: int) -> torch.Tensor:
    """``Σ_θ interp(q_θ[k], x cos θ + y sin θ)`` for every slice ``k``: ``(n_z, V, n_det)`` →
    ``(n, n, n_z)`` (no scaling). One ``grid_sample`` with the slices as channels."""
    nz, v, n_det = q.shape
    ax = -1.0 + (2.0 * torch.arange(n, dtype=torch.float64) + 1.0) / n
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")  # rows ↔ axis 0, cols ↔ axis 1
    a = torch.as_tensor(angles).detach().to("cpu", torch.float64)
    u = torch.cos(a)[:, None, None] * xx + torch.sin(a)[:, None, None] * yy  # (V, n, n)
    grid = torch.stack([u, torch.zeros_like(u)], -1).to(device=q.device, dtype=q.dtype)
    inp = q.permute(1, 0, 2).reshape(v, nz, 1, n_det)  # (V, C = n_z, H = 1, W = n_det)
    out = F.grid_sample(inp, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.sum(0).permute(1, 2, 0)  # (n, n, n_z)


def _check_stack(sinogram: torch.Tensor) -> tuple[int, int, int]:
    if sinogram.ndim != 3:
        raise ShapeError(
            f"expected a sinogram stack (n_views, n_det, n_z), got {tuple(sinogram.shape)}"
        )
    return tuple(int(s) for s in sinogram.shape)  # type: ignore[return-value]


def backproject3d(
    sinogram: torch.Tensor,
    angles: torch.Tensor,
    n: int | None = None,
    width: float = 1.0,
    samples_per_pixel: float = 1.0,
) -> torch.Tensor:
    """Pixel-driven back-projection of a sinogram stack, scaled to approximate ``Rᵀ``.

    ``Rᵀy(x, z) ≈ Δt · (n_t n_det / n²) · Σ_θ y_θ(x cos θ + y sin θ, z)`` (the 2-D
    :func:`~nefi.instances.sparse_view_ct.radon.backproject` of every slice).

    Args:
        sinogram: ``(n_views, n_det, n_z)``.
        angles: view angles (radians).
        n: lateral output size (default ``n_det``).
        width: physical lateral side length.
        samples_per_pixel: ray samples per pixel of the forward operator.
    """
    v, n_det, nz = _check_stack(sinogram)
    n = n_det if n is None else int(n)
    n_t = max(1, round(samples_per_pixel * n))
    scale = (width / n_t) * (n_t * n_det) / float(n * n)
    return _smear3d(sinogram.permute(2, 0, 1), torch.as_tensor(angles), n) * scale


def fbp3d(
    sinogram: torch.Tensor,
    angles: torch.Tensor | Sequence[float],
    filter: str = "ramp",
    *,
    n: int | None = None,
    width: float = 1.0,
    circle: bool = False,
) -> torch.Tensor:
    """Slice-wise filtered back-projection ``f(·, z) ≈ (π/V) Σ_θ q_θ(x cos θ + y sin θ, z)``.

    Args:
        sinogram: ``(n_views, n_det, n_z)`` physical line integrals.
        angles: view angles in radians (roughly uniform over ``[0, π)``).
        filter: ``"ramp"`` | ``"shepp-logan"`` | ``"cosine"`` | ``"hann"``.
        n: lateral output size (default ``n_det``).
        width: physical lateral side length.
        circle: zero the reconstruction outside the inscribed cylinder.

    Returns:
        ``(n, n, n_z)`` volume.
    """
    sinogram = torch.as_tensor(sinogram)
    v, n_det, nz = _check_stack(sinogram)
    n = n_det if n is None else int(n)
    tau = width / n_det
    rows = sinogram.permute(2, 0, 1).reshape(nz * v, n_det)
    q = (ramp_filter(rows, filter) / tau).reshape(nz, v, n_det)
    vol = _smear3d(q, torch.as_tensor(angles), n) * (math.pi / v)
    if circle:
        ax = -1.0 + (2.0 * torch.arange(n, device=vol.device, dtype=vol.dtype) + 1.0) / n
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        vol = vol * ((xx**2 + yy**2) <= 1.0).to(vol.dtype)[..., None]
    return vol


@register("operator", "radon3d")
class Radon3DOperator(Operator):
    """Parallel-beam CT about the ``z`` axis: ``μ (n, n, n_z) ↦ sinograms (V, n_det, n_z)``.

    Args:
        domain: 3-D domain with a square lateral grid and square lateral extent (``z`` = rotation
            axis, any number of slices / axial extent).
        n_views: equispaced views over ``angle_range`` (ignored if ``angles`` is given).
        field: name of the attenuation field.
        angles: explicit view angles (radians).
        angle_range: angular coverage (radians): ``π`` for a full parallel-beam scan, less for
            limited-angle tomography.
        det_per_pixel: detector bins per lateral pixel (``n_det = round(det_per_pixel · n)``);
            :meth:`at_resolution` keeps the ratio, so ``n_det`` scales with the grid.
        samples_per_pixel: ray samples per pixel along each ray.
        view_batch: evaluate the views in chunks of this size (memory); ``None`` picks the largest
            chunk whose gathered samples stay below ``max_samples``.
        max_samples: sample budget of one ``grid_sample`` call when ``view_batch`` is ``None``
            (``2**24`` ≈ 64 MB in float32).

    Linear (``homogeneity = 1``) and batchable (leading axes of the field are batch axes).
    """

    homogeneity = 1.0
    batchable = True

    def __init__(
        self,
        domain: Domain,
        n_views: int = 20,
        field: str = "mu",
        angles: torch.Tensor | Sequence[float] | None = None,
        angle_range: float = math.pi,
        det_per_pixel: float = 1.0,
        samples_per_pixel: float = 1.0,
        view_batch: int | None = None,
        max_samples: int = 1 << 24,
    ) -> None:
        super().__init__()
        if domain.ndim != 3 or domain.shape[0] != domain.shape[1]:
            raise OperatorError(
                f"Radon3DOperator needs a 3-D grid with a square lateral grid, got {domain.shape}"
            )
        w0, w1, _ = domain.size
        if abs(w0 - w1) > 1e-9 * max(w0, w1):
            raise OperatorError(
                f"Radon3DOperator needs square lateral extents, got {domain.size[:2]}"
            )
        self.domain = domain
        self.primary = field
        if angles is None:
            angles = uniform_angles(n_views, angle_range)
        self.angles = torch.as_tensor(angles, dtype=torch.float64).detach().cpu().clone()
        self.angle_range = float(angle_range)
        self.n = int(domain.shape[0])
        self.n_z = int(domain.shape[2])
        self.width = float(w0)
        self.det_per_pixel = float(det_per_pixel)
        self.samples_per_pixel = float(samples_per_pixel)
        self.view_batch = None if view_batch in (None, 0) else int(view_batch)
        self.max_samples = int(max_samples)
        self.n_det = max(1, round(self.det_per_pixel * self.n))
        self.n_t = max(1, round(self.samples_per_pixel * self.n))
        self.fidelity_tag = (
            f"radon3d-{self.n}x{self.n}x{self.n_z}-{self.det_per_pixel:g}det-"
            f"{self.samples_per_pixel:g}spp"
        )
        self._cache: dict[tuple, torch.Tensor] = {}
        self._res: dict[tuple[int, ...], Radon3DOperator] = {}

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
        """Rotated sampling lattices of all views stacked along the height: ``(1, V·n_t, n_det,
        2)``."""
        key = (str(device), str(dtype))
        if key not in self._cache:
            g = _rotation_grid(self.angles, self.n_t, self.n_det, device, dtype)
            self._cache[key] = g.reshape(1, self.n_views * self.n_t, self.n_det, 2)
        return self._cache[key]

    def forward(self, fields: Fields) -> torch.Tensor:
        x = self.get_field(fields)
        grid_shape = (self.n, self.n, self.n_z)
        if x.ndim < 3 or tuple(x.shape[-3:]) != grid_shape:
            raise ShapeError(
                f"Radon3DOperator built for {grid_shape} got field {tuple(x.shape)}; "
                "use at_resolution()"
            )
        batch = tuple(x.shape[:-3])
        xb = x.reshape(-1, *grid_shape).permute(0, 3, 1, 2)  # (B, n_z, n, n): slices = channels
        b = xb.shape[0]
        grid = self._grid(x.device, x.dtype)
        v, nt, nd = self.n_views, self.n_t, self.n_det
        chunk = self.view_batch or max(1, min(v, self.max_samples // (b * self.n_z * nt * nd)))
        outs = []
        for v0 in range(0, v, chunk):
            v1 = min(v, v0 + chunk)
            g = grid[:, v0 * nt : v1 * nt].expand(b, -1, -1, -1)
            s = F.grid_sample(xb, g, mode="bilinear", padding_mode="zeros", align_corners=False)
            outs.append(s.reshape(b, self.n_z, v1 - v0, nt, nd).sum(3))  # (B, n_z, Vc, n_det)
        s = torch.cat(outs, dim=2) * self.dt
        return s.permute(0, 2, 3, 1).reshape(*batch, v, nd, self.n_z)

    def at_resolution(self, shape: Sequence[int]) -> Radon3DOperator:
        shape = shape_tuple(shape)
        if shape == tuple(self.domain.shape):
            return self
        if shape not in self._res:
            self._res[shape] = Radon3DOperator(
                self.domain.at(shape),
                field=self.primary,
                angles=self.angles,
                angle_range=self.angle_range,
                det_per_pixel=self.det_per_pixel,
                samples_per_pixel=self.samples_per_pixel,
                view_batch=self.view_batch,
                max_samples=self.max_samples,
            )
        return self._res[shape]

    def output_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        s = shape_tuple(shape)
        return (self.n_views, max(1, round(self.det_per_pixel * s[0])), s[2])

    # ---- adjoint-type maps ------------------------------------------------------------
    def adjoint(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Exact adjoint ``Rᵀ y`` of this discretization (autograd VJP; the operator is linear)."""
        x = torch.zeros(
            self.n,
            self.n,
            self.n_z,
            device=sinogram.device,
            dtype=sinogram.dtype,
            requires_grad=True,
        )
        with torch.enable_grad():
            y = self({self.primary: x})
            (g,) = torch.autograd.grad(y, x, grad_outputs=sinogram)
        return g.detach()

    def backproject(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Pixel-driven back-projection ≈ ``Rᵀ y`` (independent discretization)."""
        return backproject3d(sinogram, self.angles, self.n, self.width, self.samples_per_pixel)

    def fbp(self, sinogram: torch.Tensor, filter: str = "ramp", circle: bool = False):
        """Slice-wise filtered back-projection onto this operator's grid."""
        return fbp3d(sinogram, self.angles, filter, n=self.n, width=self.width, circle=circle)

    def extra_repr(self) -> str:
        return (
            f"grid={tuple(self.domain.shape)}, views={self.n_views}, n_det={self.n_det}, "
            f"n_t={self.n_t}, angle_range={math.degrees(self.angle_range):.0f}°"
        )
