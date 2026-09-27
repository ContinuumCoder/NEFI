"""Steady single-phase Darcy flow: log-permeability → pressure at sparse sensors.

Physics (Bear, *Dynamics of Fluids in Porous Media*, 1972; Aziz & Settari, *Petroleum Reservoir
Simulation*, 1979)::

    q = −(k/μ)∇p,    ∇·q = f    ⇒    −∇·(k∇p) = μ f,

with injection/production wells as point sources/sinks ``f = Σ_w Q_w δ(x − x_w)`` and no-flow
(homogeneous Neumann) outer boundaries. The pure-Neumann problem requires balanced rates
(``Σ Q_w = 0``, compatibility) and fixes the pressure only up to a constant; the solver removes the
mean of the source and pins the mean pressure (:func:`~nefi.physics.elliptic.solve_elliptic`),
and the observable is referenced to the mean over the sensors (gauge pressures), which makes it
independent of that convention. Several well configurations (injector/producer pairs rotated
around the reservoir) are solved as a batch of right-hand sides.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ...domain import Domain
from ...errors import ConfigError, ShapeError
from ...physics.elliptic import BCSpec, EllipticOperator, masked_mean, point_source_rhs
from ...registry import register
from ...utils.tensor import resample, shape_tuple

__all__ = ["DarcyOperator", "sensor_mask", "well_configurations", "wells_rhs"]

Well = tuple[float, float, float]  # (x, y, rate)


def well_configurations(
    domain: Domain, n_configs: int, radius: float = 0.35, rate: float = 1.0, angle0: float = 0.3
) -> list[list[Well]]:
    """Injector/producer pairs on a circle, rotated over half a turn.

    Configuration ``c`` places an injector (``+rate``) at angle ``θ_c = angle0 + πc/n_configs`` and
    a producer (``−rate``) diametrically opposite, at ``radius ×`` the smaller side length from
    the centre — distinct flow directions that together illuminate the whole reservoir.
    """
    if n_configs < 1:
        raise ConfigError("Darcy flow needs at least one well configuration")
    (x0, x1), (y0, y1) = domain.extent
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    r = radius * min(x1 - x0, y1 - y0)
    out = []
    for c in range(n_configs):
        th = angle0 + math.pi * c / n_configs
        dx, dy = r * math.cos(th), r * math.sin(th)
        out.append([(cx + dx, cy + dy, float(rate)), (cx - dx, cy - dy, -float(rate))])
    return out


def wells_rhs(
    domain: Domain, wells: Sequence[Sequence[Well]], viscosity: float = 1.0, spread: str = "linear"
) -> torch.Tensor:
    """``(n_configs, *grid)`` source density ``μ Σ Q δ(x − x_w)`` (cloud-in-cell point sources)."""
    rhs = []
    for config in wells:
        pos = [[w[0], w[1]] for w in config]
        q = [viscosity * w[2] for w in config]
        rhs.append(point_source_rhs(domain, pos, q, spread=spread))
    return torch.stack(rhs)


def sensor_mask(
    domain: Domain,
    wells: Sequence[Sequence[Well]],
    n_sensors: int = 6,
    exclusion: float = 1.5,
    jitter: float = 0.0,
    rng=None,
) -> torch.Tensor:
    """``(n_configs, *grid)`` pressure-sensor mask on a regular ``n_sensors × n_sensors`` lattice.

    Sensors sit at lattice points (optionally jittered by ``jitter ×`` the lattice spacing) and
    are snapped to the containing cell; sensors closer than ``exclusion`` cells to a well of a
    configuration are dropped for it (the well-cell pressure has a grid-dependent logarithmic
    singularity, so it is not a resolution-consistent observable).
    """
    (x0, x1), (y0, y1) = domain.extent
    nx, ny = domain.shape
    hx, hy = domain.spacing()
    lat = (torch.arange(n_sensors, dtype=torch.float64) + 0.5) / n_sensors
    px = x0 + lat * (x1 - x0)
    py = y0 + lat * (y1 - y0)
    xs, ys = torch.meshgrid(px, py, indexing="ij")
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    if jitter > 0 and rng is not None:
        xs = xs + torch.as_tensor(rng.uniform(-1, 1, xs.shape)) * jitter * (x1 - x0) / n_sensors
        ys = ys + torch.as_tensor(rng.uniform(-1, 1, ys.shape)) * jitter * (y1 - y0) / n_sensors
    ii = ((xs - x0) / hx).floor().long().clamp(0, nx - 1)
    jj = ((ys - y0) / hy).floor().long().clamp(0, ny - 1)
    cxs, cys = x0 + (ii + 0.5) * hx, y0 + (jj + 0.5) * hy
    masks = []
    for config in wells:
        m = torch.zeros(domain.shape, dtype=torch.float64)
        for i, j, cx, cy in zip(ii.tolist(), jj.tolist(), cxs.tolist(), cys.tolist()):
            dmin = min(math.hypot(cx - w[0], cy - w[1]) for w in config)
            if dmin > exclusion * max(hx, hy):
                m[i, j] = 1.0
        masks.append(m)
    return torch.stack(masks)


@register("operator", "darcy")
class DarcyOperator(EllipticOperator):
    """``{log_k} → p (n_configs, *obs_shape)``: steady Darcy pressure per well configuration.

    The pressure is solved on the operator's grid and then sampled on the observation grid
    ``obs_shape`` (the native measurement grid; bilinear interpolation from coarser curriculum
    grids, cell averages from finer ones), so every resolution predicts the same observable; it is
    referenced to the weighted mean over the sensors (gauge pressure).

    Args:
        domain: 2-D reservoir grid of the forward model.
        wells: per configuration a list of ``(x, y, rate)`` wells (balanced for no-flow BCs).
        sensors: optional ``(n_configs, *obs_shape)`` sensor weights (reference of the gauge).
        obs_shape: observation grid (default ``domain.shape``).
        field: log-permeability field name (``sigma_transform="exp"``).
        viscosity: fluid viscosity ``μ`` (scales the sources).
        bc: boundary spec; default no-flow (``"neumann"``).
        well_spread: ``"linear"`` (cloud-in-cell) or ``"nearest"`` point-source discretization.
        face_mode / grad_mode / tol / max_iter / precond / check_every / warm_start: see
            :class:`~nefi.physics.elliptic.EllipticOperator`.
    """

    fidelity_tag = "elliptic-fv"

    def __init__(
        self,
        domain: Domain,
        wells: Sequence[Sequence[Well]],
        *,
        sensors: torch.Tensor | None = None,
        obs_shape: Sequence[int] | None = None,
        field: str = "log_k",
        sigma_transform: str = "exp",
        viscosity: float = 1.0,
        bc: BCSpec = "neumann",
        well_spread: str = "linear",
        face_mode: str = "harmonic",
        grad_mode: str = "ift",
        tol: float = 1e-6,
        max_iter: int | None = None,
        precond: str = "jacobi",
        check_every: int = 1,
        warm_start: bool = False,
    ) -> None:
        wells = [[tuple(map(float, w)) for w in config] for config in wells]
        obs_shape = domain.shape if obs_shape is None else shape_tuple(obs_shape)
        if sensors is not None and tuple(sensors.shape[-2:]) != obs_shape:
            raise ShapeError(
                f"sensor weights have grid {tuple(sensors.shape[-2:])}, observation grid is "
                f"{obs_shape}"
            )
        super().__init__(
            domain,
            wells_rhs(domain, wells, viscosity, well_spread),
            bc,
            field=field,
            sigma_transform=sigma_transform,
            face_mode=face_mode,
            grad_mode=grad_mode,
            tol=tol,
            max_iter=max_iter,
            precond=precond,
            check_every=check_every,
            warm_start=warm_start,
        )
        self.wells = wells
        self.obs_shape = obs_shape
        self.sensors = None if sensors is None else torch.as_tensor(sensors).detach().clone()
        self.darcy_kwargs = {
            "wells": wells,
            "sensors": self.sensors,
            "obs_shape": obs_shape,
            "field": field,
            "sigma_transform": sigma_transform,
            "viscosity": viscosity,
            "bc": bc,
            "well_spread": well_spread,
            "face_mode": face_mode,
            "grad_mode": grad_mode,
            "tol": tol,
            "max_iter": max_iter,
            "precond": precond,
            "check_every": check_every,
            "warm_start": warm_start,
        }

    def output_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        return (len(self.wells), *self.obs_shape)

    def post(self, u: torch.Tensor, sigma: torch.Tensor, fields) -> torch.Tensor:
        v = u if tuple(u.shape[-2:]) == self.obs_shape else resample(u, self.obs_shape)
        if self.sensors is None:
            return v
        w = self._on_device("sensors", self.sensors, v.device, v.dtype)
        return v - masked_mean(v, w, 2)

    def _rebuild(self, domain: Domain) -> DarcyOperator:
        return DarcyOperator(domain, **self.darcy_kwargs)
