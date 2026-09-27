"""Differentiable 3-D heat conduction: implicit-Euler forward model and an explicit data simulator.

* :class:`HeatOperator` — the NeFTY forward map ``α ↦ Π_Γ T^n(α)`` (NeFTY §4.2–4.3): finite-volume
  ``L(α)`` with harmonic-mean faces (Prop. 1), implicit Euler ``A(α)T^{n+1} = T^n`` (Eq. 8) solved
  by ``K`` warm-started Jacobi sweeps (Eq. 24) or CG, gradients by the discrete adjoint
  (Eq. 10–11). Fields ``{"alpha": (*grid)}`` → observed front-surface frames
  ``(n_obs, *grid[:-1])``.
* :class:`ExplicitHeatSimulator` — an *independent* forward-Euler finite-volume simulator in float64
  with the adaptive substepping of NeFTY App. E.1, used only to generate synthetic data (the
  inverse-crime guard: different time integrator, different BC implementation, higher precision).
* Initial conditions of NeFTY App. A.5: :class:`GaussianFlash` (synthetic benchmark) and
  :class:`UniformFlash` (real PVC). Temperatures are ambient-shifted (App. D.2).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence

import torch
from torch import nn

from ...domain import Domain
from ...errors import ConfigError, ShapeError
from ...registry import register
from ...utils.tensor import resample, shape_tuple
from ..base import Fields, Operator
from .adjoint import (
    ASSEMBLY_MODES,
    GRAD_MODES,
    SOLVERS,
    HeatSolveConfig,
    implicit_euler_frames,
    rollout,
)
from .stencil import FACE_MODES, BoundarySpec

log = logging.getLogger("nefi")

__all__ = [
    "INITIAL_CONDITIONS",
    "ExplicitHeatSimulator",
    "GaussianFlash",
    "HeatOperator",
    "InitialCondition",
    "UniformFlash",
    "explicit_substeps",
]


# --------------------------------------------------------------------------------------------
# initial conditions (NeFTY App. A.5)
# --------------------------------------------------------------------------------------------
class InitialCondition:
    """Callable ``(domain, shape=None) -> T0`` evaluated at cell centres (ambient-shifted)."""

    def __call__(self, domain: Domain, shape: Sequence[int] | None = None) -> torch.Tensor:
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {"type": type(self).__name__}


def _gauss_profile(
    lo: float, hi: float, n: int, center: float, width: float, cell_average: bool, images: int = 0
) -> torch.Tensor:
    """1-D Gaussian ``exp(-(x-c)²/(2w²))`` on the ``n`` cells of ``[lo, hi]``: cell averages (exact,
    via ``erf``) or cell-centre samples; ``images > 0`` adds periodic images ``c ± kL``."""
    L = hi - lo
    h = L / n
    edges = lo + h * torch.arange(n + 1, dtype=torch.float64)
    centres = 0.5 * (edges[1:] + edges[:-1])
    out = torch.zeros(n, dtype=torch.float64)
    for k in range(-images, images + 1):
        c = center + k * L
        if cell_average:
            s2w = math.sqrt(2.0) * width
            cdf = torch.special.erf((edges - c) / s2w)
            out = out + width * math.sqrt(math.pi / 2.0) * (cdf[1:] - cdf[:-1]) / h
        else:
            out = out + torch.exp(-((centres - c) ** 2) / (2.0 * width**2))
    return out


class GaussianFlash(InitialCondition):
    """Gaussian post-flash profile of the synthetic benchmark (NeFTY App. A.5)::

        T0 = A exp(-((x-x_c)² + (y-y_c)²) / (2 w_xy²)) · exp(-z² / (2 w_z²))

    with ``z`` the depth below the front face. By default each cell holds the exact **cell
    average** of the profile (finite-volume initialization), so the deposited heat is independent
    of the grid resolution — important for coarse multiscale stages and supersampled data, since
    ``w_z`` spans only a few cells. On periodic lateral axes the lateral Gaussian is periodized
    (sum over the nearest images), i.e. a periodic array of flashes.

    Args:
        amplitude: deposited pulse amplitude ``A`` (ambient-shifted temperature units).
        width_xy: lateral pulse width ``w_xy``.
        width_z: absorption depth ``w_z ≪ H`` (not given in the paper; 0.2 by default, see
            ``docs/instances/thermal_tomography.md`` for why not thinner).
        center: lateral footprint centre ``(x_c, y_c)`` (default: domain centre).
        periodic: periodize laterally (match periodic lateral boundary conditions).
        cell_average: cell averages (default) instead of cell-centre samples.
    """

    def __init__(
        self,
        amplitude: float = 100.0,
        width_xy: float = 2.5,
        width_z: float = 0.2,
        center: Sequence[float] | None = None,
        periodic: bool = True,
        cell_average: bool = True,
    ) -> None:
        self.amplitude, self.width_xy = float(amplitude), float(width_xy)
        self.width_z = float(width_z)
        self.center = None if center is None else tuple(float(c) for c in center)
        self.periodic = bool(periodic)
        self.cell_average = bool(cell_average)

    def __call__(self, domain: Domain, shape: Sequence[int] | None = None) -> torch.Tensor:
        shape = domain.shape if shape is None else shape_tuple(shape)
        out = torch.full((), self.amplitude, dtype=torch.float64)
        for ax in range(domain.ndim - 1):
            lo, hi = domain.extent[ax]
            xc = 0.5 * (lo + hi) if self.center is None else self.center[ax]
            images = 2 if self.periodic else 0
            prof = _gauss_profile(lo, hi, shape[ax], xc, self.width_xy, self.cell_average, images)
            view = [1] * domain.ndim
            view[ax] = shape[ax]
            out = out * prof.view(view)
        lo, hi = domain.extent[-1]
        pz = _gauss_profile(lo, hi, shape[-1], lo, self.width_z, self.cell_average)
        return (out * pz).expand(shape).contiguous()

    def to_dict(self) -> dict:
        return {
            "type": "gaussian_flash",
            "amplitude": self.amplitude,
            "width_xy": self.width_xy,
            "width_z": self.width_z,
            "center": self.center,
            "periodic": self.periodic,
            "cell_average": self.cell_average,
        }


class UniformFlash(InitialCondition):
    """Near-uniform front-face flash of the real-PVC setting (NeFTY App. A.5 / H.2)::

        T0 = A exp(-z² / (2 w_z²))

    (cell-averaged in ``z`` by default, see :class:`GaussianFlash`).
    """

    def __init__(self, amplitude: float = 100.0, width_z: float = 0.2, cell_average: bool = True):
        self.amplitude, self.width_z = float(amplitude), float(width_z)
        self.cell_average = bool(cell_average)

    def __call__(self, domain: Domain, shape: Sequence[int] | None = None) -> torch.Tensor:
        shape = domain.shape if shape is None else shape_tuple(shape)
        lo, hi = domain.extent[-1]
        pz = self.amplitude * _gauss_profile(lo, hi, shape[-1], lo, self.width_z, self.cell_average)
        return pz.expand(shape).contiguous()

    def to_dict(self) -> dict:
        return {
            "type": "uniform_flash",
            "amplitude": self.amplitude,
            "width_z": self.width_z,
            "cell_average": self.cell_average,
        }


INITIAL_CONDITIONS: dict[str, type[InitialCondition]] = {
    "gaussian_flash": GaussianFlash,
    "gaussian": GaussianFlash,
    "uniform_flash": UniformFlash,
    "flash": UniformFlash,
    "uniform": UniformFlash,
}


def _resolve_initial(initial) -> InitialCondition | nn.Module | torch.Tensor | Callable:
    """``None`` → :class:`GaussianFlash`; a name or ``{"type": name, **kw}`` → built from
    :data:`INITIAL_CONDITIONS`; tensors, callables and modules are used as given."""
    if initial is None:
        return GaussianFlash()
    if isinstance(initial, str | dict):
        spec = {"type": initial} if isinstance(initial, str) else dict(initial)
        name = str(spec.pop("type", "")).lower()
        if name not in INITIAL_CONDITIONS:
            raise ConfigError(
                f"unknown initial condition {name!r}; known: {sorted(INITIAL_CONDITIONS)}"
            )
        return INITIAL_CONDITIONS[name](**spec)
    return initial


# --------------------------------------------------------------------------------------------
# shared operator plumbing
# --------------------------------------------------------------------------------------------
def _obs_steps(obs_frames: Sequence[int] | int | None, n_steps: int) -> tuple[int, ...]:
    if obs_frames is None:
        return tuple(range(1, n_steps + 1))
    if isinstance(obs_frames, int):  # every k-th step
        if obs_frames < 1:
            raise ConfigError("an integer obs_frames is a stride and must be >= 1")
        return tuple(range(obs_frames, n_steps + 1, obs_frames))
    steps = tuple(sorted({int(n) for n in obs_frames}))
    if not steps or steps[0] < 0 or steps[-1] > n_steps:
        raise ConfigError(f"obs_frames must be step indices in [0, {n_steps}], got {obs_frames}")
    return steps


class _HeatBase(Operator):
    """Common state of the implicit and explicit heat operators."""

    primary = "alpha"
    homogeneity = None  # the parameter-to-observation map is nonlinear in α
    traceable = False  # N_t × K sweeps in a Python loop (+ the adjoint autograd.Function)

    def __init__(
        self,
        domain: Domain,
        dt: float,
        n_steps: int,
        obs_frames: Sequence[int] | int | None,
        initial,
        bc,
        robin_h: float,
        face_mode: str,
        field: str,
    ) -> None:
        super().__init__()
        if domain.ndim < 2:
            raise ConfigError("heat operators need a >= 2-D domain (lateral axes + depth)")
        if face_mode not in FACE_MODES:
            raise ConfigError(f"unknown face_mode {face_mode!r}; use one of {FACE_MODES}")
        self.domain = domain
        self.dt, self.n_steps = float(dt), int(n_steps)
        self.obs_steps = _obs_steps(obs_frames, self.n_steps)
        self._obs_arg = obs_frames
        self.bc = BoundarySpec.parse(bc, domain.ndim, robin_h)
        self.face_mode = face_mode
        self.primary = field
        self.initial = _resolve_initial(initial)
        # T^0 is kept as a float64 CPU tensor (not a buffer: MPS has no float64 and ``.to`` should
        # not round it) and cast lazily per (device, dtype). A module IC is re-evaluated at every
        # forward (it may carry learnable parameters).
        self._T0 = (
            None
            if isinstance(self.initial, nn.Module)
            else self._evaluate_initial(domain).detach().to(device="cpu", dtype=torch.float64)
        )
        self._T0_cast: dict[tuple, torch.Tensor] = {}

    # -- initial condition -------------------------------------------------------------------
    def _evaluate_initial(self, domain: Domain) -> torch.Tensor:
        ic = self.initial
        if torch.is_tensor(ic):
            t = ic.detach().to(device="cpu", dtype=torch.float64)
            if tuple(t.shape) != domain.shape:
                t = resample(t, domain.shape)
            return t
        if not callable(ic):
            raise ConfigError(f"cannot use {type(ic).__name__} as an initial condition")
        t = ic(domain)
        t = torch.as_tensor(t)
        if tuple(t.shape) != domain.shape:
            raise ShapeError(
                f"initial condition has shape {tuple(t.shape)}, domain grid is {domain.shape}"
            )
        return t

    def initial_state(self, like: torch.Tensor | None = None) -> torch.Tensor:
        """``T^0`` on the operator's grid (cast to ``like``'s dtype / device; float64 CPU
        otherwise)."""
        if self._T0 is None:
            t = self._evaluate_initial(self.domain)
            return t if like is None else t.to(device=like.device, dtype=like.dtype)
        if like is None:
            return self._T0
        key = (like.device, like.dtype)
        if key not in self._T0_cast:
            self._T0_cast[key] = self._T0.to(device=like.device, dtype=like.dtype)
        return self._T0_cast[key]

    # -- Operator API ------------------------------------------------------------------------
    @property
    def spacing(self) -> tuple[float, ...]:
        return self.domain.spacing()

    @property
    def n_obs(self) -> int:
        return len(self.obs_steps)

    @property
    def frame_times(self) -> torch.Tensor:
        """Physical times ``n Δt`` of the observed frames."""
        return torch.tensor(self.obs_steps, dtype=torch.float64) * self.dt

    def output_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        shape = shape_tuple(shape)
        return (self.n_obs, *shape[:-1])

    def required_fields(self) -> tuple[str, ...]:
        return (self.primary,)

    def _alpha(self, fields: Fields) -> torch.Tensor:
        alpha = self.get_field(fields)
        if tuple(alpha.shape) != self.domain.shape:
            raise ShapeError(
                f"{type(self).__name__} is built for grid {self.domain.shape} but got alpha of "
                f"shape {tuple(alpha.shape)}; use operator.at_resolution(shape)"
            )
        return alpha

    def _init_kwargs(self) -> dict:
        return {
            "dt": self.dt,
            "n_steps": self.n_steps,
            "obs_frames": self._obs_arg,
            "initial": self.initial,
            "bc": self.bc,
            "robin_h": self.bc.robin_h,
            "face_mode": self.face_mode,
            "field": self.primary,
        }


# --------------------------------------------------------------------------------------------
# the inversion operator
# --------------------------------------------------------------------------------------------
@register("operator", "heat")
class HeatOperator(_HeatBase):
    """NeFTY hard-constrained forward model: diffusivity → observed front-surface frames.

    Implements NeFTY Eq. (7)–(8) (finite-volume ``L(α)``, implicit Euler), Eq. (24) (``K``
    warm-started Jacobi sweeps) or CG, and the discrete adjoint of Eq. (10)–(11) / App. D.3.

    Args:
        domain: physical slab, grid ``(nx, ny, nz)`` (or ``(nx, nz)``); last axis = depth ``z``
            with the observed front surface at ``z = 0``.
        dt: time step ``Δt`` (Tab. 5: 0.05; one camera frame).
        n_steps: number of steps ``N_t`` (Tab. 5: 100 frames).
        obs_frames: observed step indices (``0`` = initial state, ``n`` = after ``n`` steps);
            ``None`` = every step ``1..n_steps``; an ``int`` ``k`` = every ``k``-th step.
        initial: initial condition: :class:`GaussianFlash` (default, App. A.5 synthetic),
            :class:`UniformFlash` (PVC), a registry name/dict, a tensor on the grid (resampled at
            other resolutions), a callable ``domain -> tensor``, or an ``nn.Module`` evaluated at
            every forward (learnable initial conditions receive ``dJ/dT^0 = μ^1``).
        bc: boundary conditions (default periodic lateral + adiabatic z, App. A.5); use
            ``("periodic", "periodic", "robin")`` with ``robin_h`` for the PVC back face.
        robin_h: back-face convective coefficient ``h`` (App. A.5, homogeneous form).
        solver: ``"jacobi"`` (Eq. 24), ``"chebyshev"`` (Chebyshev-accelerated Jacobi: the accuracy
            of 50 Jacobi sweeps with ≈ 20 iterations at the Tab. 5 time step; opt-in) or ``"cg"``
            (Jacobi-preconditioned CG).
        inner_iters: Jacobi sweeps / Chebyshev iterations ``K`` per step (Tab. 5: 50 Jacobi).
        cg_tol / cg_max_iter: CG relative tolerance and iteration cap.
        grad_mode: ``"adjoint"`` (Eq. 10–11), ``"autograd"`` (unrolled reference) or
            ``"checkpoint"``.
        face_mode: ``"harmonic"`` (Prop. 1) or ``"arithmetic"`` (ablation).
        field: name of the diffusivity field.
        adjoint_assembly: ``"fused"`` or ``"per_step"`` (see :mod:`.adjoint`).
        substeps: implicit-Euler sub-steps per frame interval (``Δt/substeps`` each). ``1``
            (default) is the paper's scheme (one step per camera frame); larger values reduce the
            first-order time-discretization error of the early post-flash transient at
            proportional cost.
        compile: ``torch.compile`` the gradient-free Jacobi sweeps of the adjoint forward /
            backward solves (opt-in, for CUDA servers; see :class:`HeatSolveConfig`).
        compile_mode: ``torch.compile`` mode (e.g. ``"reduce-overhead"``).
        stencil_backend: ``"auto"`` (default; flat padded layout, see
            :mod:`~nefi.operators.pde.stencil`), ``"flat"`` or ``"roll"`` (reference).
    """

    fidelity_tag = "implicit-euler"

    def __init__(
        self,
        domain: Domain,
        dt: float = 0.05,
        n_steps: int = 100,
        obs_frames: Sequence[int] | int | None = None,
        initial=None,
        bc: BoundarySpec | str | Sequence[str] | None = None,
        robin_h: float = 0.0,
        solver: str = "jacobi",
        inner_iters: int = 50,
        cg_tol: float = 1e-6,
        cg_max_iter: int = 200,
        grad_mode: str = "adjoint",
        face_mode: str = "harmonic",
        field: str = "alpha",
        adjoint_assembly: str = "fused",
        substeps: int = 1,
        compile: bool = False,
        compile_mode: str | None = None,
        stencil_backend: str = "auto",
    ) -> None:
        super().__init__(domain, dt, n_steps, obs_frames, initial, bc, robin_h, face_mode, field)
        if int(substeps) < 1:
            raise ConfigError(f"substeps must be >= 1, got {substeps}")
        self.substeps = int(substeps)
        if solver not in SOLVERS:
            raise ConfigError(f"unknown solver {solver!r}; use one of {SOLVERS}")
        if grad_mode not in GRAD_MODES:
            raise ConfigError(f"unknown grad_mode {grad_mode!r}; use one of {GRAD_MODES}")
        if adjoint_assembly not in ASSEMBLY_MODES:
            raise ConfigError(f"unknown adjoint_assembly {adjoint_assembly!r}")
        self.solver, self.inner_iters = solver, int(inner_iters)
        self.cg_tol, self.cg_max_iter = float(cg_tol), int(cg_max_iter)
        self.grad_mode, self.adjoint_assembly = grad_mode, adjoint_assembly
        self.compile, self.compile_mode = bool(compile), compile_mode
        self.stencil_backend = stencil_backend
        self.fidelity_tag = f"implicit-euler-{solver}"
        m = self.substeps
        self.solve_config = HeatSolveConfig(
            spacing=domain.spacing(),
            dt=self.dt / m,
            n_steps=self.n_steps * m,
            obs_steps=tuple(n * m for n in self.obs_steps),
            bc=self.bc,
            face_mode=self.face_mode,
            solver=self.solver,
            inner_iters=self.inner_iters,
            cg_tol=self.cg_tol,
            cg_max_iter=self.cg_max_iter,
            assembly=self.adjoint_assembly,
            compile=self.compile,
            compile_mode=self.compile_mode,
            stencil_backend=stencil_backend,
        )

    def forward(self, fields: Fields) -> torch.Tensor:
        alpha = self._alpha(fields)
        T0 = self.initial_state(alpha)
        return implicit_euler_frames(alpha, T0, self.solve_config, self.grad_mode)

    def simulate(
        self,
        alpha: torch.Tensor,
        T0: torch.Tensor | None = None,
        return_states: bool = False,
        residuals: bool = False,
    ) -> dict[str, torch.Tensor | list[float] | None]:
        """Diagnostic forward solve (no gradients).

        Returns a dict with ``frames`` (as :meth:`forward`), ``states`` (``T^1..T^{N_t}`` at the
        frame times, if ``return_states``), ``residuals`` (relative linear-solve residual of every
        implicit step, if ``residuals``) and ``T0``.
        """
        with torch.no_grad():
            T0 = self.initial_state(alpha) if T0 is None else T0.to(alpha)
            frames, states, res = rollout(
                alpha,
                T0,
                self.solve_config,
                keep_states=return_states,
                residuals=residuals,
                n_steps=self.solve_config.n_steps if return_states else None,
            )
        if states is not None and self.substeps > 1:
            states = states[self.substeps - 1 :: self.substeps]
        return {"frames": frames, "states": states, "residuals": res, "T0": T0}

    def at_resolution(self, shape: Sequence[int]) -> HeatOperator:
        """Same physics (``Δt``, ``N_t``, frames, BCs, solver) on grid ``shape`` of the same slab;
        spacing and the initial condition are rebuilt (a learnable IC module is shared)."""
        shape = shape_tuple(shape)
        if shape == self.domain.shape:
            return self
        return HeatOperator(self.domain.at(shape), **self._init_kwargs())

    def _init_kwargs(self) -> dict:
        kw = super()._init_kwargs()
        kw.update(
            solver=self.solver,
            inner_iters=self.inner_iters,
            cg_tol=self.cg_tol,
            cg_max_iter=self.cg_max_iter,
            grad_mode=self.grad_mode,
            adjoint_assembly=self.adjoint_assembly,
            substeps=self.substeps,
            compile=self.compile,
            compile_mode=self.compile_mode,
            stencil_backend=self.stencil_backend,
        )
        return kw

    def extra_repr(self) -> str:
        return (
            f"grid={self.domain.shape}, dt={self.dt}, n_steps={self.n_steps}, n_obs={self.n_obs}, "
            f"substeps={self.substeps}, bc={self.bc.kinds}, "
            f"solver={self.solver}(K={self.inner_iters}), "
            f"grad_mode={self.grad_mode}, face_mode={self.face_mode}"
        )


# --------------------------------------------------------------------------------------------
# the independent data simulator (inverse-crime guard, NeFTY App. E.1)
# --------------------------------------------------------------------------------------------
def explicit_substeps(
    dt: float,
    spacing: Sequence[float],
    alpha_max: float,
    min_substeps: int = 10,
    safety: float = 2.0,
) -> int:
    """Substeps per frame, NeFTY App. E.1: ``N_sub = max(10, ⌈Δt/Δt_stable × 2⌉)`` with
    ``Δt_stable = Δx_min² / (2 D α_max)`` (``D`` the spatial dimension).

    Using the smallest spacing makes ``Δt_stable`` a lower bound of the exact anisotropic
    forward-Euler limit ``1 / (2 α_max Σ_d Δ_d⁻²)``, so the scheme is always stable.
    """
    d = len(spacing)
    dt_stable = min(spacing) ** 2 / (2.0 * d * max(float(alpha_max), 1e-30))
    return max(int(min_substeps), int(math.ceil(dt / dt_stable * safety)))


def _pad_axis(x: torch.Tensor, dim: int, mode: str) -> torch.Tensor:
    """One ghost cell on each side of ``dim``: circular (periodic) or replicate (Neumann)."""
    n = x.shape[dim]
    if mode == "periodic":
        lo, hi = x.narrow(dim, n - 1, 1), x.narrow(dim, 0, 1)
    else:  # replicate padding: zero cross-face increment (App. D.2)
        lo, hi = x.narrow(dim, 0, 1), x.narrow(dim, n - 1, 1)
    return torch.cat([lo, x, hi], dim=dim)


@register("operator", "heat_explicit")
class ExplicitHeatSimulator(_HeatBase):
    """Independent explicit (forward-Euler, substepped, float64) heat simulator — data only.

    NeFTY App. E.1 generates synthetic data with an explicit finite-volume engine and adaptive
    substepping so that the reconstruction is not evaluated against its own discretization. This
    simulator deliberately shares no time-stepping or boundary code with :class:`HeatOperator`:
    ghost cells by circular / replicate padding (App. D.2), flux-difference divergence, forward
    Euler with ``N_sub`` substeps per frame (:func:`explicit_substeps`). Its ``fidelity_tag``
    differs from the inversion operator's, which the benchmark protocol checks.

    Args:
        domain / dt / n_steps / obs_frames / initial / bc / robin_h / field: as in
            :class:`HeatOperator`.
        face_mode: ``"harmonic"`` or ``"arithmetic"`` face diffusivity.
        min_substeps: lower bound on substeps per frame (App. E.1: 10).
        substep_safety: factor on ``Δt/Δt_stable`` (App. E.1: 2.0).
        sim_dtype: simulation precision (float64).
    """

    fidelity_tag = "explicit-substepped-float64"

    def __init__(
        self,
        domain: Domain,
        dt: float = 0.05,
        n_steps: int = 100,
        obs_frames: Sequence[int] | int | None = None,
        initial=None,
        bc: BoundarySpec | str | Sequence[str] | None = None,
        robin_h: float = 0.0,
        face_mode: str = "harmonic",
        min_substeps: int = 10,
        substep_safety: float = 2.0,
        field: str = "alpha",
        sim_dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(domain, dt, n_steps, obs_frames, initial, bc, robin_h, face_mode, field)
        self.min_substeps, self.substep_safety = int(min_substeps), float(substep_safety)
        self.sim_dtype = sim_dtype

    def n_substeps(self, alpha: torch.Tensor) -> int:
        return explicit_substeps(
            self.dt, self.spacing, float(alpha.max()), self.min_substeps, self.substep_safety
        )

    def _face_alpha(self, alpha: torch.Tensor) -> list[torch.Tensor]:
        """Face diffusivities on the ``N+1`` faces of every axis (ghost faces included)."""
        faces = []
        for ax in range(alpha.ndim):
            ap = _pad_axis(alpha, ax, self.bc.kinds[ax])
            left, right = ap.narrow(ax, 0, ap.shape[ax] - 1), ap.narrow(ax, 1, ap.shape[ax] - 1)
            if self.face_mode == "harmonic":
                faces.append(2.0 * left * right / (left + right))
            else:
                faces.append(0.5 * (left + right))
        return faces

    def divergence(self, T: torch.Tensor, faces: list[torch.Tensor]) -> torch.Tensor:
        """``∇·(α∇T)`` by flux differences with ghost cells (plus the Robin sink)."""
        out = torch.zeros_like(T)
        for ax, (a, h) in enumerate(zip(faces, self.spacing)):
            Tp = _pad_axis(T, ax, self.bc.kinds[ax])
            n = Tp.shape[ax]
            flux = a * (Tp.narrow(ax, 1, n - 1) - Tp.narrow(ax, 0, n - 1))  # N+1 faces
            m = flux.shape[ax]
            out = out + (flux.narrow(ax, 1, m - 1) - flux.narrow(ax, 0, m - 1)) / h**2
        if self.bc.has_robin:
            sink = torch.zeros_like(T)
            sink[..., -1] = self.bc.robin_h / self.spacing[-1]
            out = out - sink * T
        return out

    def forward(self, fields: Fields) -> torch.Tensor:
        alpha = self._alpha(fields).detach().to(self.sim_dtype)
        T = self.initial_state(alpha).detach()
        n_sub = self.n_substeps(alpha)
        h = self.dt / n_sub
        faces = self._face_alpha(alpha)
        obs = set(self.obs_steps)
        frames = [T[..., 0]] if 0 in obs else []
        with torch.no_grad():
            for n in range(1, max(self.obs_steps) + 1):
                for _ in range(n_sub):
                    T = T + h * self.divergence(T, faces)
                if n in obs:
                    frames.append(T[..., 0])
        return torch.stack(frames)

    def at_resolution(self, shape: Sequence[int]) -> ExplicitHeatSimulator:
        shape = shape_tuple(shape)
        if shape == self.domain.shape:
            return self
        return ExplicitHeatSimulator(
            self.domain.at(shape),
            min_substeps=self.min_substeps,
            substep_safety=self.substep_safety,
            sim_dtype=self.sim_dtype,
            **self._init_kwargs(),
        )

    def extra_repr(self) -> str:
        return (
            f"grid={self.domain.shape}, dt={self.dt}, n_steps={self.n_steps}, "
            f"bc={self.bc.kinds}, face_mode={self.face_mode}, dtype={self.sim_dtype}"
        )
