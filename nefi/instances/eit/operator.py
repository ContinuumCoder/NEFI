"""Continuum-model EIT forward operator: conductivity → boundary potentials for current patterns.

Physics (Calderón 1980; Cheney, Isaacson & Newell, *SIAM Rev.* 41 (1999); Somersalo, Cheney &
Isaacson, *SIAM J. Appl. Math.* 52 (1992) for the gap electrode model)::

    −∇·(σ∇u_p) = 0 in Ω,   σ ∂u_p/∂ν = j_p on ∂Ω,   ∮ j_p = 0,   measure u_p on ∂Ω,

for current patterns ``p``. On a rectangle, the boundary is parameterized counter-clockwise by the
arc length ``s ∈ [0, P)`` starting at the corner ``(x_lo, y_lo)``; the trigonometric patterns are
``j_{2m} = A cos(2π(m+1)s/P)`` and ``j_{2m+1} = A sin(2π(m+1)s/P)`` (the optimal patterns for a
centred disk, Isaacson 1986). ``electrodes=L > 0`` switches to the gap model: ``L`` electrodes of
relative coverage ``c`` carry the currents ``I_l = A (P/L) trig(2π(m+1)s_l/P)`` uniformly spread
over their extent, with no current in the gaps.

Discretization: the Neumann data enter the finite-volume right-hand side as ``j/h_n`` on the
boundary cells (face-averaged pattern, so the injected current is exact at every resolution) and
:class:`~nefi.physics.elliptic.EllipticOperator` solves the pure-Neumann problem (compatibility +
zero-mean gauge). The observable is the **boundary trace** ``u_face = u_cell + (h_n/2) j/σ_cell``
(second-order extrapolation with the known flux) of every boundary face; unlike the cell value it is
resolution-consistent. The traces are mapped by periodic arc-length interpolation onto the faces
of the *native observation grid* (:func:`face_interpolation`) and averaged per strip cell over its
(electrode-covered) faces, so a coarse curriculum stage and the 2×-finer data generator predict
exactly the native observable. Potentials are referenced to their boundary-length-weighted mean
over the observed boundary (``∮ u = 0`` gauge of Cheney et al.).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ...domain import Domain
from ...errors import ConfigError, ShapeError
from ...physics.elliptic import EllipticOperator, masked_mean
from ...registry import register
from ...utils.tensor import resample, shape_tuple

__all__ = ["EITBoundary", "EITOperator", "boundary_faces", "eit_boundary", "face_interpolation"]


@dataclass
class EITBoundary:
    """Boundary data of the EIT model on one grid.

    Attributes:
        rhs: ``(n_patterns, H, W)`` finite-volume source ``j/h_n`` on boundary cells.
        offset: ``(n_patterns, H, W)`` trace offset ``(h_n/2)·j`` averaged over each strip
            cell's observed faces (divide by σ to get ``u_face − u_cell``; diagnostics — the
            operator works per face).
        weights: ``(H, W)`` observed boundary length per cell (electrode-covered length in the gap
            model); the measurement mask is ``weights > 0``.
        faces: per-face arrays (``i``, ``j``, ``s0``, ``length``, ``hn``, ``coverage``,
            ``current``).
    """

    rhs: torch.Tensor
    offset: torch.Tensor
    weights: torch.Tensor
    faces: dict[str, np.ndarray]


def boundary_faces(domain: Domain) -> dict[str, np.ndarray]:
    """Boundary faces of a 2-D rectangular grid in counter-clockwise arc-length order.

    Returns arrays ``i, j`` (cell index), ``s0`` (arc length at the face start), ``length``
    (face length), ``hn`` (cell size normal to the face) and the perimeter ``P``.
    """
    if domain.ndim != 2:
        raise ShapeError(f"EIT boundary geometry needs a 2-D domain, got {domain.shape}")
    (x0, x1), (y0, y1) = domain.extent
    nx, ny = domain.shape
    hx, hy = domain.spacing()
    lx, ly = x1 - x0, y1 - y0
    ix, jy = np.arange(nx), np.arange(ny)
    sides = [
        # bottom (y = y0), x increasing
        (ix, np.zeros(nx, int), ix * hx, np.full(nx, hx), np.full(nx, hy)),
        # right (x = x1), y increasing
        (np.full(ny, nx - 1), jy, lx + jy * hy, np.full(ny, hy), np.full(ny, hx)),
        # top (y = y1), x decreasing
        (ix[::-1], np.full(nx, ny - 1), lx + ly + (nx - 1 - ix[::-1]) * hx, np.full(nx, hx),
         np.full(nx, hy)),
        # left (x = x0), y decreasing
        (np.zeros(ny, int), jy[::-1], 2 * lx + ly + (ny - 1 - jy[::-1]) * hy, np.full(ny, hy),
         np.full(ny, hx)),
    ]  # fmt: skip
    cat = [np.concatenate([s[k] for s in sides]) for k in range(5)]
    return {
        "i": cat[0].astype(int),
        "j": cat[1].astype(int),
        "s0": cat[2].astype(float),
        "length": cat[3].astype(float),
        "hn": cat[4].astype(float),
        "P": np.array(2 * (lx + ly)),
    }


def _pattern_average(p: int, s0: np.ndarray, ell: np.ndarray, perimeter: float) -> np.ndarray:
    """Average of pattern ``p`` (cos/sin of frequency ``p//2 + 1``) over ``[s0, s0 + ell]``."""
    w = 2 * math.pi * (p // 2 + 1) / perimeter
    if p % 2 == 0:
        return (np.sin(w * (s0 + ell)) - np.sin(w * s0)) / (w * ell)
    return (np.cos(w * s0) - np.cos(w * (s0 + ell))) / (w * ell)


def _pattern_value(p: int, s: np.ndarray, perimeter: float) -> np.ndarray:
    w = 2 * math.pi * (p // 2 + 1) / perimeter
    return np.cos(w * s) if p % 2 == 0 else np.sin(w * s)


def eit_boundary(
    domain: Domain,
    n_patterns: int,
    amplitude: float = 1.0,
    electrodes: int = 0,
    coverage: float = 0.5,
    dtype: torch.dtype = torch.float64,
) -> EITBoundary:
    """Right-hand sides, trace offsets and observation weights of the EIT model on ``domain``.

    Args:
        domain: 2-D rectangular grid.
        n_patterns: number of trigonometric current patterns (cos/sin pairs of frequency
            1, 1, 2, 2, ...).
        amplitude: current-density amplitude ``A`` (continuum) / current scale (gap model).
        electrodes: 0 = continuum model; ``L > 0`` = gap model with ``L`` electrodes.
        coverage: fraction of the perimeter covered by electrodes (gap model).
    """
    if n_patterns < 1:
        raise ConfigError("EIT needs at least one current pattern")
    if electrodes and not 0.0 < coverage <= 1.0:
        raise ConfigError(f"electrode coverage must be in (0, 1], got {coverage}")
    f = boundary_faces(domain)
    per = float(f["P"])
    s0, ell, hn = f["s0"], f["length"], f["hn"]
    nf = s0.shape[0]
    if electrodes:
        centers = (np.arange(electrodes) + 0.5) * per / electrodes
        half = 0.5 * coverage * per / electrodes
        overlap = np.clip(
            np.minimum(s0[:, None] + ell[:, None], centers[None] + half)
            - np.maximum(s0[:, None], centers[None] - half),
            0.0,
            None,
        )  # (faces, electrodes)
        cov = overlap.sum(1) / ell
    else:
        cov = np.ones(nf)
    currents = np.zeros((n_patterns, nf))
    for p in range(n_patterns):
        if electrodes:
            current_l = amplitude * (per / electrodes) * _pattern_value(p, centers, per)
            j = (overlap * (current_l / (2 * half))[None]).sum(1) / ell
        else:
            j = amplitude * _pattern_average(p, s0, ell, per)
        j = j - (j * ell).sum() / (cov * ell).sum() * cov  # exact compatibility ∮ j = 0
        currents[p] = j
    nx, ny = domain.shape
    rhs = np.zeros((n_patterns, nx, ny))
    offnum = np.zeros((n_patterns, nx, ny))
    weights = np.zeros((nx, ny))
    wf = cov * ell
    np.add.at(weights, (f["i"], f["j"]), wf)
    for p in range(n_patterns):
        np.add.at(rhs[p], (f["i"], f["j"]), currents[p] / hn)
        np.add.at(offnum[p], (f["i"], f["j"]), wf * 0.5 * hn * currents[p])
    offset = np.divide(offnum, weights[None], out=np.zeros_like(offnum), where=weights[None] > 0)
    faces = dict(f)
    faces.update({"coverage": cov, "current": currents})
    return EITBoundary(
        torch.as_tensor(rhs, dtype=dtype),
        torch.as_tensor(offset, dtype=dtype),
        torch.as_tensor(weights, dtype=dtype),
        faces,
    )


def face_interpolation(
    src: dict[str, np.ndarray], dst: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Periodic linear interpolation in arc length from ``src`` to ``dst`` boundary-face centres.

    Returns ``(k0, k1, w0, w1)`` with ``t_dst = w0·t_src[k0] + w1·t_src[k1]``. The boundary trace
    is continuous along the perimeter (also around corners), so this maps traces between grid
    resolutions with an ``O(h²)`` error; for a 2× finer ``src`` it is the average of the two fine
    faces covering each coarse face.
    """
    per = float(src["P"])
    cs = src["s0"] + 0.5 * src["length"]
    cd = dst["s0"] + 0.5 * dst["length"]
    n = cs.shape[0]
    k1 = np.searchsorted(cs, cd, side="right") % n
    k0 = (k1 - 1) % n
    left = cs[k0] - np.where(cs[k0] > cd, per, 0.0)
    right = cs[k1] + np.where(cs[k1] < cd, per, 0.0)
    w1 = (cd - left) / np.maximum(right - left, 1e-300)
    return k0, k1, 1.0 - w1, w1


@register("operator", "eit")
class EITOperator(EllipticOperator):
    """Continuum / gap-model EIT: ``{σ} → V (n_patterns, *obs_shape)``, strip = boundary traces.

    Solves ``−∇·(σ∇u_p) = 0`` with the Neumann current patterns of :func:`eit_boundary` on the
    operator's grid (implicit-function adjoint, warm-startable), forms the per-face boundary
    traces ``u + (h_n/2) j/σ``, maps them by arc-length interpolation to the boundary faces of the
    *observation* grid ``obs_shape`` (the native measurement grid — so every curriculum
    resolution predicts the same observable), averages them per strip cell with the observed
    boundary length as weight, and references every pattern to the weighted mean over the observed
    boundary. Interior entries hold the (resampled, referenced) potentials; the measurement mask
    selects the strip.

    Args:
        domain: 2-D rectangular grid of the forward model.
        n_patterns / amplitude / electrodes / coverage: see :func:`eit_boundary`.
        obs_shape: observation grid (default: ``domain.shape``).
        field: conductivity field name.
        sigma_transform: e.g. ``"exp"`` when the field is a log-conductivity.
        face_mode / grad_mode / tol / max_iter / precond / check_every / warm_start: see
            :class:`~nefi.physics.elliptic.EllipticOperator`.
    """

    fidelity_tag = "elliptic-fv"

    def __init__(
        self,
        domain: Domain,
        n_patterns: int = 8,
        *,
        amplitude: float = 1.0,
        electrodes: int = 0,
        coverage: float = 0.5,
        obs_shape: Sequence[int] | None = None,
        field: str = "sigma",
        sigma_transform: str | Callable[[torch.Tensor], torch.Tensor] = "identity",
        face_mode: str = "harmonic",
        grad_mode: str = "ift",
        tol: float = 1e-6,
        max_iter: int | None = None,
        precond: str = "jacobi",
        check_every: int = 1,
        warm_start: bool = False,
    ) -> None:
        geo = eit_boundary(domain, n_patterns, amplitude, electrodes, coverage)
        obs_shape = domain.shape if obs_shape is None else shape_tuple(obs_shape)
        obs = (
            geo
            if obs_shape == domain.shape
            else eit_boundary(domain.at(obs_shape), n_patterns, amplitude, electrodes, coverage)
        )
        super().__init__(
            domain,
            geo.rhs,
            "neumann",
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
        self.eit_kwargs = {
            "n_patterns": int(n_patterns),
            "amplitude": float(amplitude),
            "electrodes": int(electrodes),
            "coverage": float(coverage),
            "obs_shape": obs_shape,
            "field": field,
            "sigma_transform": sigma_transform,
            "face_mode": face_mode,
            "grad_mode": grad_mode,
            "tol": tol,
            "max_iter": max_iter,
            "precond": precond,
            "check_every": check_every,
            "warm_start": warm_start,
        }
        self.n_patterns = int(n_patterns)
        self.obs_shape = obs_shape
        self.boundary = geo
        self.obs_boundary = obs
        f, fo = geo.faces, obs.faces
        self._face_i = torch.as_tensor(f["i"], dtype=torch.long)
        self._face_j = torch.as_tensor(f["j"], dtype=torch.long)
        self._face_coef = torch.as_tensor(0.5 * f["hn"][None] * f["current"])  # (P, F)
        if obs is geo:
            self._interp = None
        else:
            k0, k1, w0, w1 = face_interpolation(f, fo)
            self._interp = (
                torch.as_tensor(k0, dtype=torch.long),
                torch.as_tensor(k1, dtype=torch.long),
                torch.as_tensor(w0),
                torch.as_tensor(w1),
            )
        wf = fo["coverage"] * fo["length"]
        self._obs_lin = torch.as_tensor(fo["i"] * obs_shape[1] + fo["j"], dtype=torch.long)
        self._obs_wf = torch.as_tensor(wf)
        self._obs_weights = obs.weights

    @property
    def observation_weights(self) -> torch.Tensor:
        """``(*obs_shape)`` observed boundary length per strip cell (mask = ``> 0``)."""
        return self._obs_weights

    def output_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        return (self.n_patterns, *self.obs_shape)

    def face_traces(self, u: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """``(n_patterns, n_faces)`` boundary traces ``u + (h_n/2) j/σ`` on this operator's grid."""
        dev, dt = u.device, u.dtype
        fi, fj = self._face_i.to(dev), self._face_j.to(dev)
        coef = self._on_device("face_coef", self._face_coef, dev, dt)
        return u[..., fi, fj] + coef / sigma[..., fi, fj]

    def post(self, u: torch.Tensor, sigma: torch.Tensor, fields) -> torch.Tensor:
        dev, dt = u.device, u.dtype
        t = self.face_traces(u, sigma)
        if self._interp is not None:
            k0, k1, w0, w1 = self._interp
            w0 = self._on_device("w0", w0, dev, dt)
            w1 = self._on_device("w1", w1, dev, dt)
            t = w0 * t[..., k0.to(dev)] + w1 * t[..., k1.to(dev)]
        wf = self._on_device("obs_wf", self._obs_wf, dev, dt)
        wc = self._on_device("obs_weights", self._obs_weights, dev, dt)
        ho, wo = self.obs_shape
        flat = torch.zeros(*t.shape[:-1], ho * wo, device=dev, dtype=dt)
        ring = flat.index_add(-1, self._obs_lin.to(dev), t * wf).reshape(*t.shape[:-1], ho, wo)
        ring = ring / torch.where(wc > 0, wc, torch.ones_like(wc))
        interior = u if tuple(u.shape[-2:]) == self.obs_shape else resample(u, self.obs_shape)
        v = torch.where(wc > 0, ring, interior)
        return v - masked_mean(v, wc, 2)

    def _rebuild(self, domain: Domain) -> EITOperator:
        return EITOperator(domain, **self.eit_kwargs)
