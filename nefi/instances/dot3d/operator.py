"""Continuous-wave diffuse optical tomography (DOT) forward model on a 3-D slab.

Diffusion approximation of the radiative transfer equation (Arridge, *Inverse Problems* 15, 1999;
Durduran et al., *Rep. Prog. Phys.* 73, 2010)::

    −∇·(D ∇Φ) + μ_a Φ = S,        D = 1 / (3 μ_s')   (absorption-independent, Furutsu & Yamada 1994)

with the extrapolated-boundary / partial-current **Robin** condition on every face
``Φ + 2 A D ∂_n Φ = 0`` (outward exitance ``J = Φ_b / 2A``), where ``A = (1 + R_eff)/(1 − R_eff)``
accounts for the refractive-index mismatch (Groenhuis et al. 1983 fit of ``R_eff(n)``; Haskell et
al., *JOSA A* 11, 1994). Sources are isotropic point sources one transport mean free path
``1/μ_s'`` below the top face (cloud-in-cell splatted by
:func:`~nefi.physics.elliptic.point_source_rhs`, re-derived at every resolution).

Discretization: :class:`~nefi.physics.elliptic.EllipticOperator` (cell-centered finite volumes,
harmonic faces, Jacobi-PCG, implicit-function adjoint) with **Neumann** sides plus the exact
finite-volume Robin flux: the boundary face flux of a boundary cell is ``K_b Φ_0`` with the series
conductance ``K_b = 1 / (2A + h/(2D))`` of the half-cell diffusion resistance ``h/(2D)`` and the
boundary "film" ``2A``, which enters the operator as the extra sink ``κ_b = K_b / h`` of the
boundary cells (second-order accurate, no ghost state). The unknown absorption is the
zeroth-order coefficient ``κ = μ_a + κ_b``.

Observable: the diffuse reflectance (exitance) ``J = K_b Φ_0`` of the top-face cells, bilinearly
interpolated and averaged over each detector's square footprint (``S × S`` sub-samples), for
every source: ``(n_sources, n_dx, n_dy)`` — resolution independent, so every curriculum stage
compares against the same measurement. With ``reference_mua`` the readings are **calibrated**
against a homogeneous reference slab, ``J(μ_a) / J(μ_ref)`` (the normalized / Rytov data of
difference DOT, e.g. O'Leary et al. 1995, Pogue et al. 1999): unknown source and detector coupling
factors and most of the systematic discretization error cancel in the ratio (measured: the 24³
vs 48³ model mismatch drops from ≈ 6.5 % to ≤ 0.9 % rms).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ...domain import Domain
from ...errors import ConfigError, ShapeError
from ...operators.base import Fields
from ...physics.elliptic import (
    EllipticOperator,
    is_transformed,
    point_source_rhs,
    solve_elliptic,
)
from ...registry import register

__all__ = [
    "DiffuseOpticalOperator",
    "boundary_conductance",
    "detector_layout",
    "grid_positions",
    "robin_A",
    "robin_sink",
]


def robin_A(refractive_index: float) -> float:
    """Boundary mismatch factor ``A = (1 + R_eff) / (1 − R_eff)`` with the Groenhuis et al. (1983)
    fit ``R_eff = −1.440 n⁻² + 0.710 n⁻¹ + 0.668 + 0.0636 n`` (``n = 1.4`` → ``A ≈ 3.25``;
    ``n = 1`` → ``A ≈ 1``)."""
    n = float(refractive_index)
    if n < 1.0:
        raise ConfigError(f"refractive_index must be >= 1, got {n}")
    r = -1.440 / n**2 + 0.710 / n + 0.668 + 0.0636 * n
    return (1.0 + r) / (1.0 - r)


def boundary_conductance(h: float, D: float, A: float) -> float:
    """Series conductance ``K_b = 1 / (2A + h / 2D)`` of a boundary face (flux ``K_b Φ_0``)."""
    return 1.0 / (2.0 * A + h / (2.0 * D))


def robin_sink(domain: Domain, D: float, A: float, dtype=torch.float64) -> torch.Tensor:
    """Volumetric sink ``κ_b = Σ_faces K_b / h`` of the boundary cells (Robin on all six faces)."""
    shape = domain.shape
    out = torch.zeros(shape, dtype=dtype)
    for ax, h in enumerate(domain.spacing()):
        kb = boundary_conductance(h, D, A) / h
        for i in (0, shape[ax] - 1):
            idx = [slice(None)] * domain.ndim
            idx[ax] = i  # type: ignore[call-overload]
            out[tuple(idx)] += kb
    return out


def grid_positions(n: Sequence[int], lo: Sequence[float], hi: Sequence[float]) -> torch.Tensor:
    """Cell centers of a regular ``n[0] × n[1]`` layout over ``[lo, hi]`` → ``(n0, n1, 2)``."""
    axes = [
        torch.tensor(
            [lo[d] + (i + 0.5) * (hi[d] - lo[d]) / n[d] for i in range(int(n[d]))],
            dtype=torch.float64,
        )
        for d in range(2)
    ]
    a, b = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([a, b], -1)


def detector_layout(
    domain: Domain, n: Sequence[int], span: float = 0.8
) -> tuple[torch.Tensor, float]:
    """Detector centers ``(n_dx, n_dy, 2)`` covering the central ``span`` fraction of the top face
    and the detector pitch (square pixels, one per layout cell)."""
    (x0, x1), (y0, y1) = domain.extent[0], domain.extent[1]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    hx, hy = 0.5 * span * (x1 - x0), 0.5 * span * (y1 - y0)
    centers = grid_positions(n, (cx - hx, cy - hy), (cx + hx, cy + hy))
    pitch = min(2 * hx / int(n[0]), 2 * hy / int(n[1]))
    return centers, pitch


@register("operator", "diffuse_optical")
class DiffuseOpticalOperator(EllipticOperator):
    """CW DOT: absorption ``μ_a(x)`` → top-face exitance at detector footprints for every source.

    Args:
        domain: slab domain ``(x, y, z)`` (mm); the last axis is depth, ``z = extent_lo`` is the
            illuminated / imaged top face.
        sources: source positions ``(n_sources, 3)`` (physical).
        detectors: detector centers ``(n_dx, n_dy, 2)`` on the top face (physical x, y).
        detector_size: side length of the square detector footprint.
        D: diffusion coefficient ``1/(3μ_s')`` (known, homogeneous).
        A: Robin boundary factor (:func:`robin_A`).
        field: name of the absorption field.
        source_power: power of every source.
        detector_samples: ``S`` sub-samples per footprint axis.
        reference_mua: if set, divide the readings by those of the homogeneous slab with this
            absorption (calibrated data; the reference is solved once per resolution / device /
            dtype and cached).
        tol / max_iter / precond / check_every / warm_start / face_mode / grad_mode: elliptic
            solver settings (:class:`~nefi.physics.elliptic.EllipticOperator`).

    Output: ``(n_sources, n_dx, n_dy)``. The map is nonlinear in ``μ_a`` (``homogeneity = None``)
    and monotonically decreasing (more absorption, less light).
    """

    homogeneity = None

    def __init__(
        self,
        domain: Domain,
        sources: torch.Tensor | Sequence[Sequence[float]],
        detectors: torch.Tensor,
        detector_size: float,
        *,
        D: float,
        A: float,
        field: str = "mu_a",
        source_power: float = 1.0,
        detector_samples: int = 3,
        reference_mua: float | None = None,
        tol: float = 1e-8,
        max_iter: int | None = None,
        precond: str = "jacobi",
        check_every: int = 1,
        warm_start: bool = True,
        face_mode: str = "harmonic",
        grad_mode: str = "ift",
    ) -> None:
        if domain.ndim != 3:
            raise ShapeError(f"DiffuseOpticalOperator needs a 3-D slab, got {domain.shape}")
        src = torch.as_tensor(sources, dtype=torch.float64).reshape(-1, 3)
        det = torch.as_tensor(detectors, dtype=torch.float64)
        if det.ndim != 3 or det.shape[-1] != 2:
            raise ShapeError(f"detectors must be (n_dx, n_dy, 2), got {tuple(det.shape)}")
        if D <= 0 or A <= 0 or detector_size <= 0:
            raise ConfigError("D, A and detector_size must be positive")
        power = float(source_power)

        def rhs(d: Domain) -> torch.Tensor:
            return torch.stack([point_source_rhs(d, [s.tolist()], [power]) for s in src])

        super().__init__(
            domain,
            rhs,
            "neumann",
            field=field,
            face_mode=face_mode,
            grad_mode=grad_mode,
            tol=tol,
            max_iter=max_iter,
            precond=precond,
            check_every=check_every,
            warm_start=warm_start,
        )
        self.sources = src.clone()
        self.detectors = det.clone()
        self.detector_size = float(detector_size)
        self.D, self.A = float(D), float(A)
        self.source_power = power
        self.detector_samples = max(1, int(detector_samples))
        self.reference_mua = None if reference_mua is None else float(reference_mua)
        self._D_grid = torch.full(domain.shape, self.D, dtype=torch.float64)
        self._sink = robin_sink(domain, self.D, self.A)
        self._kb_top = boundary_conductance(domain.spacing()[2], self.D, self.A)
        self.fidelity_tag = "dot-fv-robin" + ("-calibrated" if reference_mua is not None else "")

    # --- configuration ------------------------------------------------------------------------
    def _rebuild(self, domain: Domain) -> DiffuseOpticalOperator:
        op = DiffuseOpticalOperator(
            domain,
            self.sources,
            self.detectors,
            self.detector_size,
            D=self.D,
            A=self.A,
            field=self.primary,
            source_power=self.source_power,
            detector_samples=self.detector_samples,
            reference_mua=self.reference_mua,
            tol=self.tol,
            max_iter=self.max_iter,
            precond=self.precond,
            check_every=self.check_every,
            warm_start=self.warm_start,
            face_mode=self.face_mode,
            grad_mode=self.grad_mode,
        )
        op.fidelity_tag = self.fidelity_tag
        return op

    @property
    def n_sources(self) -> int:
        return int(self.sources.shape[0])

    @property
    def detector_shape(self) -> tuple[int, int]:
        return int(self.detectors.shape[0]), int(self.detectors.shape[1])

    def required_fields(self) -> tuple[str, ...]:
        return (self.primary,)

    def output_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        return (self.n_sources, *self.detector_shape)

    # --- physics hooks ------------------------------------------------------------------------
    def coefficient(self, fields: Fields) -> torch.Tensor:
        """The (known, homogeneous) diffusion coefficient on the grid."""
        mua = self.get_field(fields)
        return self._on_device("D", self._D_grid, mua.device, mua.dtype)

    def kappa_value(self, fields: Fields, ref: torch.Tensor) -> torch.Tensor:
        """``κ = μ_a + κ_b`` (unknown absorption + Robin boundary sink)."""
        mua = self.get_field(fields)
        return mua + self._on_device("sink", self._sink, mua.device, mua.dtype)

    def _stencil(self, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Bilinear interpolation stencils of the detector sub-samples on the top-face cell
        centers: flat indices and weights ``(n_dx·n_dy·S², 4)`` (border-clamped, cached)."""
        key = ("readout", str(device), str(dtype))
        st = self._cache.get(key)
        if st is None:
            s = self.detector_samples
            off = ((torch.arange(s, dtype=torch.float64) + 0.5) / s - 0.5) * self.detector_size
            px = self.detectors[..., 0][:, :, None, None] + off[None, None, :, None]
            py = self.detectors[..., 1][:, :, None, None] + off[None, None, None, :]
            px, py = torch.broadcast_tensors(px, py)  # (n_dx, n_dy, S, S)
            nx, ny = self.domain.shape[0], self.domain.shape[1]
            idx, w = [], []
            for p, (lo, _hi), h, n in zip(
                (px, py), self.domain.extent[:2], self.domain.spacing()[:2], (nx, ny)
            ):
                t = ((p - lo) / h - 0.5).clamp(0.0, n - 1.0)  # continuous cell index
                i0 = torch.floor(t).clamp(max=max(n - 2, 0))
                f = t - i0
                i1 = (i0 + 1).clamp(max=n - 1)
                idx.append((i0.long(), i1.long()))
                w.append((1.0 - f, f))
            (ix0, ix1), (iy0, iy1) = idx
            (wx0, wx1), (wy0, wy1) = w
            flat = torch.stack([ix0 * ny + iy0, ix0 * ny + iy1, ix1 * ny + iy0, ix1 * ny + iy1], -1)
            wts = torch.stack([wx0 * wy0, wx0 * wy1, wx1 * wy0, wx1 * wy1], -1)
            st = (flat.reshape(-1, 4).to(device), wts.reshape(-1, 4).to(device=device, dtype=dtype))
            if not any(is_transformed(t) for t in st):
                self._cache[key] = st  # type: ignore[assignment]
        return st  # type: ignore[return-value]

    def readout(self, u: torch.Tensor) -> torch.Tensor:
        """Top-face exitance ``K_b Φ[..., 0]`` averaged over the detector footprints.

        Bilinear interpolation of the top-face cell values at ``S × S`` sub-samples per detector,
        written as an index gather so it is twice differentiable (Hessian-vector products and
        double-backward JVPs through the whole operator).
        """
        top = u[..., 0] * self._kb_top  # (n_src, nx, ny)
        flat, wts = self._stencil(u.device, u.dtype)
        vals = (top.flatten(-2)[..., flat] * wts).sum(-1)  # (n_src, n_dx·n_dy·S²)
        nd = self.detector_shape
        s2 = self.detector_samples**2
        return vals.reshape(*top.shape[:-2], nd[0], nd[1], s2).mean(-1)

    def reference_readout(self, device=None, dtype=None) -> torch.Tensor:
        """Readings of the homogeneous reference slab (``μ_a = reference_mua``), cached; the
        solve bypasses the warm-start state of the optimization loop. It uses the implicit-
        function path (``grad_mode="ift"``), whose exact forward rule gives the constant a zero
        tangent when it is first requested inside a ``torch.func`` transform (not cached then)."""
        if self.reference_mua is None:
            raise ConfigError("reference_readout needs reference_mua")
        dtype = dtype or torch.get_default_dtype()
        key = ("reference", str(device), str(dtype))
        r = self._cache.get(key)
        if r is None:
            with torch.no_grad():
                mua = torch.full(self.domain.shape, self.reference_mua, device=device, dtype=dtype)
                fields = {self.primary: mua}
                u = solve_elliptic(
                    self.coefficient(fields),
                    self.rhs(device, dtype),
                    self.spacing,
                    self.bc,
                    kappa=self.kappa_value(fields, mua),
                    tol=self.tol,
                    max_iter=self.max_iter,
                    precond=self.precond,
                    face_mode=self.face_mode,
                    grad_mode="ift",
                    check_every=self.check_every,
                )
                r = self.readout(u).detach()
            if not is_transformed(r):
                self._cache[key] = r
        return r

    def post(self, u: torch.Tensor, sigma: torch.Tensor, fields: Fields) -> torch.Tensor:
        y = self.readout(u)
        if self.reference_mua is None:
            return y
        return y / self.reference_readout(u.device, u.dtype)

    def fluence(self, mua: torch.Tensor) -> torch.Tensor:
        """Full fluence ``Φ`` ``(n_sources, *grid)`` for an absorption map (diagnostics)."""
        fields = {self.primary: mua}
        return self.solve(self.coefficient(fields), self.kappa_value(fields, mua))

    def extra_repr(self) -> str:
        return (
            f"grid={self.domain.shape}, sources={self.n_sources}, detectors={self.detector_shape}"
            f" ({self.detector_size:.3g} mm, {self.detector_samples}² samples), D={self.D:.4g}, "
            f"A={self.A:.3g}, tol={self.tol:g}"
        )
