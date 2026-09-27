"""Shared plumbing of the elliptic-family instances (EIT, Darcy flow, NV current imaging).

* :class:`SupersampledObservationGenerator` — inverse-crime-safe data: the same finite-volume
  physics evaluated on a ``supersample``× finer grid in float64, averaged onto the *observed* native
  cells only (boundary strip, sensors), plus Gaussian noise on the observed entries.
* :func:`footprint_resample` — resample observation weights so a native cell keeps its exact
  footprint on finer grids (nearest) and its area fraction on coarser grids (area).
* Scene primitives: sub-cell coverage of analytic shapes, distances to polylines/polygons and a
  resolution-independent Gaussian random field (random Fourier features).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import torch

from ..bench.base import DataGenerator
from ..domain import Domain
from ..measurement import Measurement
from ..operators.base import Operator
from ..physics.elliptic import weighted_area_downsample
from ..utils.tensor import resample, shape_tuple

WeightsFn = Callable[[tuple[int, ...]], torch.Tensor]

__all__ = [
    "SupersampledObservationGenerator",
    "cell_coverage",
    "footprint_resample",
    "polygon_signed_distance",
    "polyline_distance",
    "random_fourier_field",
]


# --------------------------------------------------------------------------------------------
# observation weights across resolutions
# --------------------------------------------------------------------------------------------
def footprint_resample(w: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Resample observation weights ``(..., *grid)`` to ``shape`` (trailing axes).

    Finer grids replicate each native cell over its exact footprint (nearest, integer factors);
    coarser grids take area fractions — so weighted means over the observed set are consistent
    across resolutions.
    """
    shape = shape_tuple(shape)
    nd = len(shape)
    old = tuple(w.shape[-nd:])
    if old == shape:
        return w
    finer = all(n >= o for n, o in zip(shape, old))
    return resample(w.to(torch.float64), shape, mode="nearest" if finer else "area").to(w.dtype)


class SupersampledObservationGenerator(DataGenerator):
    """Data generator: fine-grid physics (float64) → observed native cells → noise.

    The instances' operators already sample their observable on the native grid when run on the
    fine grid (EIT: boundary traces interpolated to the native strip faces; Darcy: pressure
    cell-averaged to the native cells); any remaining finer output is averaged onto the observed
    native cells with the observation weights.

    Args:
        operator: forward operator defined on the *fine* grid (``supersample ×`` native).
        obs_weights: ``shape -> weights`` of the observation set at a grid resolution
            (broadcastable to the operator output); 0 = unobserved.
        native_shape: grid of the inversion / measurement.
        noise_std: Gaussian noise std relative to ``max |clean|`` over the observed entries.
        supersample: fine/native resolution ratio.
        fidelity_tag: inverse-crime tag (must differ from the inversion operator's).
        dtype / device: simulation precision / device.

    The measurement stores ``data·mask`` (unobserved entries are zeroed so no hidden state can
    leak into the inversion), the full-shape ``mask``, the absolute noise std, and
    ``meta["clean"]`` (noise-free observed data, popped into the ground truth by the instances).
    """

    def __init__(
        self,
        operator: Operator,
        obs_weights: WeightsFn,
        native_shape: Sequence[int],
        noise_std: float = 0.0,
        supersample: int = 2,
        fidelity_tag: str = "elliptic-fv-2x-float64",
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            operator,
            noise_std=noise_std,
            relative=True,
            dtype=dtype,
            device=device,
            supersample=supersample,
            fidelity_tag=fidelity_tag,
        )
        self.obs_weights = obs_weights
        self.native_shape = shape_tuple(native_shape)

    def observe(self, clean_fine: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Weighted average of the fine simulation onto the observed native cells."""
        nd = len(self.native_shape)
        w = self.obs_weights(tuple(clean_fine.shape[-nd:])).to(clean_fine)
        return weighted_area_downsample(clean_fine, w, self.native_shape)

    def generate(
        self,
        gt_fields: dict[str, torch.Tensor],
        rng: np.random.Generator,
        noise_std: float | None = None,
        target_shape: Sequence[int] | None = None,
    ) -> Measurement:
        clean, w = self.observe(self.clean(gt_fields))
        mask = (w > 0).to(clean.dtype).expand_as(clean).contiguous()
        clean = clean * mask
        ns = self.noise_std if noise_std is None else float(noise_std)
        if self.relative:
            ns = ns * float(clean.abs().max())
        data = clean
        if ns > 0:
            noise = torch.as_tensor(
                rng.standard_normal(tuple(clean.shape)), device=clean.device, dtype=clean.dtype
            )
            data = clean + ns * noise * mask
        return Measurement(
            data.float(),
            mask.float(),
            noise_std=ns if ns > 0 else None,
            meta={
                "fidelity": self.fidelity_tag,
                "supersample": self.supersample,
                "clean": clean.float(),
            },
        )


# --------------------------------------------------------------------------------------------
# scene primitives
# --------------------------------------------------------------------------------------------
def cell_coverage(
    domain: Domain,
    shape: Sequence[int] | None,
    inside: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sub: int = 4,
) -> torch.Tensor:
    """Fraction of each cell inside a 2-D analytic region (``sub × sub`` sub-samples per cell).

    ``inside(x, y) -> bool tensor`` is evaluated on physical coordinates, so the same scene renders
    consistently at every resolution (anti-aliased, area-weighted edges).
    """
    dom = domain if shape is None else domain.at(shape)
    xy = dom.physical_coords(dtype=torch.float64)
    hx, hy = dom.spacing()
    offs = (torch.arange(sub, dtype=torch.float64) + 0.5) / sub - 0.5
    acc = torch.zeros(dom.shape, dtype=torch.float64)
    for ox in offs:
        for oy in offs:
            acc += inside(xy[..., 0] + ox * hx, xy[..., 1] + oy * hy).to(torch.float64)
    return acc / sub**2


def polyline_distance(points: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Euclidean distance from ``points (..., 2)`` to an open polyline ``vertices (V, 2)``."""
    p = points.reshape(-1, 2)
    a, b = vertices[:-1].to(p), vertices[1:].to(p)
    ab = b - a
    t = ((p[:, None, :] - a[None]) * ab[None]).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-30)
    proj = a[None] + t.clamp(0.0, 1.0)[..., None] * ab[None]
    d = (p[:, None, :] - proj).norm(dim=-1).min(dim=1).values
    return d.reshape(points.shape[:-1])


def polygon_signed_distance(points: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Signed distance to a closed polygon (negative inside; even–odd rule)."""
    p = points.reshape(-1, 2)
    v = vertices.to(p)
    closed = torch.cat([v, v[:1]], dim=0)
    d = polyline_distance(p, closed)
    x, y = p[:, 0:1], p[:, 1:2]
    x1, y1 = v[None, :, 0], v[None, :, 1]
    x2, y2 = torch.roll(v, -1, 0)[None, :, 0], torch.roll(v, -1, 0)[None, :, 1]
    cond = (y1 > y) != (y2 > y)
    xint = x1 + (y - y1) * (x2 - x1) / torch.where(y2 != y1, y2 - y1, torch.ones_like(y1))
    inside = ((cond & (x < xint)).sum(dim=1) % 2) == 1
    return torch.where(inside, -d, d).reshape(points.shape[:-1])


def random_fourier_field(
    rng: np.random.Generator, n_modes: int, corr_length: float, ndim: int = 2
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Zero-mean, unit-variance Gaussian random field with covariance ``exp(−r²/2ℓ²)``.

    Randomization (spectral) method (Shinozuka 1971; Rahimi & Recht 2007):
    ``Y(x) = √(2/M) Σ_m cos(k_m·x + φ_m)``, ``k_m ~ N(0, I/ℓ²)``, ``φ_m ~ U(0, 2π)``. The field
    is a closed-form function of position, so it renders identically at every grid resolution.
    """
    k = torch.as_tensor(rng.standard_normal((n_modes, ndim)) / corr_length, dtype=torch.float64)
    phi = torch.as_tensor(rng.uniform(0.0, 2 * math.pi, n_modes), dtype=torch.float64)
    scale = math.sqrt(2.0 / n_modes)

    def field(x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float64)
        return scale * torch.cos(x @ k.T + phi).sum(-1)

    return field
