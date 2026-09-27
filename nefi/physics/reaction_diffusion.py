"""Gray–Scott reaction–diffusion with spatially varying reaction parameters.

Model (Gray & Scott 1984; Pearson 1993, *Science* 261:189):

.. math::

    u_t = D_u Δu − u v² + F(x)(1 − u), \\qquad
    v_t = D_v Δv + u v² − (F(x) + k(x)) v,

with zero-flux (Neumann, default) or periodic boundaries. The unknown is the feed-rate field
``F(x)`` (or the kill rate ``k(x)``); the other parameters and the initial condition
(:class:`GrayScottIC`) are known. Observations are snapshots of ``u`` and/or ``v`` at a few
physical times ``t_obs``. Units follow Pearson: time in reaction units, lengths such that
``D_u = 2·10⁻⁵``, ``D_v = 10⁻⁵``.

Discretization: conservative second-order (2d+1-point) diffusion operator in flux form
(:func:`diffusion_term`; zero boundary flux for Neumann → exact mass conservation, or wrap-around),
**explicit Euler** in time. Stability (von Neumann, linearized):

.. math::

    Δt \\,\\big(D_{max} \\textstyle\\sum_a 4/h_a^2 + r_{max}\\big) \\le 2,

``r_max = F_max + k_max + reaction_rate_bound`` bounds the reaction Jacobian
(``|∂_u(uv²)| = v²`` and ``|∂_v(uv²)| = 2uv`` are ≤ 1 for ``u, v ∈ [0, 1]``); :func:`stable_dt`
returns this bound times a safety factor. The simulation step is ``dt_sim = dt / m`` with the
smallest integer ``m`` meeting the bound (times ``substeps``), so the observation times —
integer multiples of ``dt`` — are hit exactly at every grid resolution. Explicit Euler is
first-order accurate, so the data generator's 4× substepping
(``fidelity_tag="gray-scott-substep4-float64"``) differs from the inversion model by O(dt) — a
genuine discretization gap (inverse-crime guard).

Gradients: autodiff through the unrolled loop, optionally with per-block checkpointing
(:func:`nefi.physics.timestep.run_timestepping`).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..domain import Domain
from ..errors import ConfigError, OperatorError, ShapeError
from ..operators.base import Fields, Operator
from ..registry import register
from ..utils.tensor import shape_tuple
from .timestep import auto_checkpoint_every, run_timestepping, stable_dt_diffusion

log = logging.getLogger("nefi")

BOUNDARIES = ("neumann", "periodic")
SPECIES = ("u", "v")
UNKNOWNS = ("F", "k", "Du")

__all__ = [
    "BOUNDARIES",
    "GrayScottIC",
    "ReactionDiffusionOperator",
    "UNKNOWNS",
    "diffusion_term",
    "gray_scott_step",
    "stable_dt",
]


def stable_dt(
    d_max: float,
    spacing: Sequence[float],
    F_max: float,
    k_max: float,
    reaction_rate_bound: float = 1.0,
    safety: float = 0.9,
) -> float:
    """Largest stable explicit-Euler step for Gray–Scott (see module docstring)."""
    return stable_dt_diffusion(d_max, spacing, F_max + k_max + reaction_rate_bound, safety)


def _face_mean(c: torch.Tensor, dim: int, periodic: bool) -> torch.Tensor:
    """Harmonic mean of neighbouring cell coefficients on the faces along ``dim``.

    The harmonic mean is the flux-continuity-consistent face value for a piecewise-constant
    coefficient (NeFTY Prop. 1); ``n − 1`` interior faces, or ``n`` faces when periodic.
    """
    if periodic:
        a, b = c, torch.roll(c, -1, dim)
    else:
        n = c.shape[dim]
        a, b = c.narrow(dim, 0, n - 1), c.narrow(dim, 1, n - 1)
    return 2.0 * a * b / (a + b).clamp_min(1e-30)


def diffusion_term(
    x: torch.Tensor,
    spacing: Sequence[float],
    boundary: str = "neumann",
    coeff: torch.Tensor | None = None,
) -> torch.Tensor:
    """Conservative second-order diffusion operator over the trailing ``len(spacing)`` dims.

    ``coeff=None``: the 5-point (2d+1-point) Laplacian ``Δx``; otherwise ``∇·(D ∇x)`` with the
    cell-centered ``D = coeff`` (trailing spatial shape) averaged harmonically onto faces. Written
    in flux form: face fluxes by forward differences, divergence by differences of fluxes, with
    zero boundary flux (``"neumann"``, mass-conserving) or wrap-around (``"periodic"``).
    """
    d = len(spacing)
    if boundary not in BOUNDARIES:
        raise ConfigError(f"boundary must be one of {BOUNDARIES}, got {boundary!r}")
    if x.ndim < d:
        raise ShapeError(f"tensor of shape {tuple(x.shape)} has fewer than {d} dims")
    periodic = boundary == "periodic"
    out = None
    for ax, h in enumerate(spacing):
        dim = x.ndim - d + ax
        if periodic:
            flux = torch.roll(x, -1, dim) - x
        else:
            flux = torch.diff(x, dim=dim)
        if coeff is not None:
            flux = flux * _face_mean(coeff, coeff.ndim - d + ax, periodic)
        if periodic:
            term = flux - torch.roll(flux, 1, dim)
        else:
            pads = [0, 0] * (x.ndim - 1 - dim) + [1, 1]
            term = torch.diff(F.pad(flux, pads), dim=dim)
        term = term / float(h) ** 2
        out = term if out is None else out + term
    return out


def gray_scott_step(
    u: torch.Tensor,
    v: torch.Tensor,
    F_: torch.Tensor | float,
    k: torch.Tensor | float,
    Du: torch.Tensor | float,
    Dv: float,
    dt: float,
    spacing: Sequence[float],
    boundary: str = "neumann",
) -> tuple[torch.Tensor, torch.Tensor]:
    """One explicit-Euler Gray–Scott step. ``Du`` may be a spatial field (conservative form)."""
    if torch.is_tensor(Du) and Du.ndim > 0:
        lap_u = diffusion_term(u, spacing, boundary, coeff=Du)
        lap_v = Dv * diffusion_term(v, spacing, boundary)
    else:
        lap = diffusion_term(torch.stack([u, v]), spacing, boundary)
        lap_u, lap_v = Du * lap[0], Dv * lap[1]
    uvv = u * v * v
    u_new = u + dt * (lap_u - uvv + F_ * (1.0 - u))
    v_new = v + dt * (lap_v + uvv - (F_ + k) * v)
    return u_new, v_new


@dataclass
class GrayScottIC:
    """Known smooth initial condition evaluated analytically at any resolution.

    ``g(x) = g_mean + g_amp Π_a cos(2π m_a (x_a − lo_a)/L_a + φ_a)``,
    ``u0 = u_base − a_u g``, ``v0 = v_base + a_v g``. The defaults perturb the whole domain away
    from the trivial state ``(1, 0)`` so that ``F`` influences the dynamics everywhere from
    ``t = 0`` (in ``(1, 0)`` regions ``F(1 − u) = 0`` and ``F`` would be unobservable).
    """

    u_base: float = 1.0
    v_base: float = 0.0
    a_u: float = 0.5
    a_v: float = 0.25
    g_mean: float = 0.75
    g_amp: float = 0.25
    modes: tuple[int, ...] = (2, 3)
    phases: tuple[float, ...] = (0.3, 1.1)

    def pattern(self, domain: Domain, dtype=torch.float64) -> torch.Tensor:
        x = domain.physical_coords(dtype=torch.float64)
        g = torch.ones(domain.shape, dtype=torch.float64)
        for ax in range(domain.ndim):
            lo, hi = domain.extent[ax]
            m = self.modes[ax % len(self.modes)]
            ph = self.phases[ax % len(self.phases)]
            g = g * torch.cos(2.0 * math.pi * m * (x[..., ax] - lo) / (hi - lo) + ph)
        return (self.g_mean + self.g_amp * g).to(dtype)

    def __call__(self, domain: Domain, device=None, dtype=torch.float32):
        g = self.pattern(domain)
        u0 = (self.u_base - self.a_u * g).to(device=device, dtype=dtype)
        v0 = (self.v_base + self.a_v * g).to(device=device, dtype=dtype)
        return u0, v0


@register("operator", "gray_scott")
class ReactionDiffusionOperator(Operator):
    """Gray–Scott forward model: parameter field (``F`` or ``k``) → snapshots of ``u``/``v``.

    Args:
        domain: spatial domain (1-3 D, physical units).
        obs_times: observation times (integer multiples of ``dt``).
        field: name of the unknown field; ``unknown``: which parameter it is (``"F"`` feed rate,
            ``"k"`` kill rate, or ``"Du"`` diffusivity of ``u`` in conservative form
            ``∇·(D_u ∇u)`` with harmonic-mean faces).
        F / k / Du: the known values of the parameters that are not the unknown (floats).
        Dv: diffusivity of ``v``.
        dt: base time step; the simulation uses ``dt / m`` (CFL, ``substeps``).
        observe: species to observe (``("u", "v")`` or a subset).
        boundary: ``"neumann"`` | ``"periodic"``.
        ic: known initial condition (:class:`GrayScottIC`).
        param_max: upper bound of the unknown parameter used in the stability bound (e.g. the
            Bounded head's ``hi``).
        reaction_rate_bound / safety: see :func:`stable_dt`.
        substeps: extra substepping factor (the data generator uses 4).
        grad_mode / checkpoint_every: ``"checkpoint"`` (default block ``ceil(sqrt(n_steps))``) or
            ``"autograd"``.

    Output ``(n_obs · n_species, *shape)`` ordered time-major (``[u(t1), v(t1), u(t2), …]``).
    """

    fidelity_tag = "gray-scott-euler"
    traceable = False  # a Python time loop

    def __init__(
        self,
        domain: Domain,
        obs_times: Sequence[float],
        *,
        field: str = "F",
        unknown: str = "F",
        F: float = 0.045,
        k: float = 0.06,
        Du: float = 2e-5,
        Dv: float = 1e-5,
        dt: float = 1.0,
        observe: Sequence[str] = ("u", "v"),
        boundary: str = "neumann",
        ic: GrayScottIC | dict | None = None,
        param_max: float = 0.1,
        reaction_rate_bound: float = 1.0,
        safety: float = 0.9,
        substeps: int = 1,
        grad_mode: str = "checkpoint",
        checkpoint_every: int | None = None,
    ) -> None:
        super().__init__()
        if unknown not in UNKNOWNS:
            raise ConfigError(f"unknown must be one of {UNKNOWNS}, got {unknown!r}")
        if boundary not in BOUNDARIES:
            raise ConfigError(f"boundary must be one of {BOUNDARIES}, got {boundary!r}")
        if grad_mode not in ("checkpoint", "autograd"):
            raise ConfigError(f"grad_mode must be 'checkpoint' or 'autograd', got {grad_mode!r}")
        bad = [s for s in observe if s not in SPECIES]
        if bad or not observe:
            raise ConfigError(f"observe must be a non-empty subset of {SPECIES}, got {observe}")
        self.domain = domain
        self.obs_times = tuple(float(t) for t in obs_times)
        self.dt = float(dt)
        steps = [t / self.dt for t in self.obs_times]
        if any(abs(s - round(s)) > 1e-9 or s < 0 for s in steps):
            raise ConfigError(f"obs_times {self.obs_times} must be non-negative multiples of dt")
        self.obs_steps_base = [int(round(s)) for s in steps]
        self.primary = field
        self.unknown = unknown
        self.F_const, self.k_const = float(F), float(k)
        self.Du, self.Dv = float(Du), float(Dv)
        self.observe = tuple(observe)
        self.boundary = boundary
        self.ic = GrayScottIC(**ic) if isinstance(ic, dict) else (ic or GrayScottIC())
        self.param_max = float(param_max)
        self.reaction_rate_bound = float(reaction_rate_bound)
        self.safety = float(safety)
        self.substeps = int(substeps)
        self.grad_mode = grad_mode
        self.checkpoint_every = checkpoint_every
        self.homogeneity = None
        f_max = self.param_max if unknown == "F" else self.F_const
        k_max = self.param_max if unknown == "k" else self.k_const
        d_max = max(self.param_max if unknown == "Du" else self.Du, self.Dv)
        dt_max = stable_dt(
            d_max,
            domain.spacing(),
            f_max,
            k_max,
            self.reaction_rate_bound,
            self.safety,
        )
        self.m = max(1, int(math.ceil(self.dt / dt_max - 1e-12))) * max(1, self.substeps)
        self.dt_sim = self.dt / self.m
        self.n_steps = max(self.obs_steps_base) * self.m
        self._cache: dict = {}
        self._res: dict = {}

    # -------------------------------------------------------------------------------------
    def _kw(self) -> dict:
        return {
            "field": self.primary,
            "unknown": self.unknown,
            "F": self.F_const,
            "k": self.k_const,
            "Du": self.Du,
            "Dv": self.Dv,
            "dt": self.dt,
            "observe": self.observe,
            "boundary": self.boundary,
            "ic": self.ic,
            "param_max": self.param_max,
            "reaction_rate_bound": self.reaction_rate_bound,
            "safety": self.safety,
            "substeps": self.substeps,
            "grad_mode": self.grad_mode,
            "checkpoint_every": self.checkpoint_every,
        }

    def at_resolution(self, shape):
        shape = shape_tuple(shape)
        if shape == tuple(self.domain.shape):
            return self
        if shape not in self._res:
            op = ReactionDiffusionOperator(self.domain.at(shape), self.obs_times, **self._kw())
            op.fidelity_tag = self.fidelity_tag
            self._res[shape] = op
        return self._res[shape]

    def output_shape(self, shape):
        return (len(self.obs_times) * len(self.observe), *shape_tuple(shape))

    @property
    def record(self) -> list[int]:
        return [s * self.m for s in self.obs_steps_base]

    def _static(self, device, dtype) -> dict:
        key = (str(device), str(dtype))
        if key not in self._cache:
            u0, v0 = self.ic(self.domain, device, dtype)
            self._cache[key] = {"u0": u0, "v0": v0}
        return self._cache[key]

    def simulate(self, param: torch.Tensor, observe=None, record=None):
        """Run the explicit-Euler loop; returns ``(final_state, observations)``.

        ``observe(state, n)`` receives ``(u, v)``; default: the configured species at the
        observation steps.
        """
        if tuple(param.shape) != tuple(self.domain.shape):
            raise ShapeError(
                f"{self.primary} has shape {tuple(param.shape)}, operator expects "
                f"{self.domain.shape}"
            )
        pmax = float(param.detach().max())
        if pmax > self.param_max * (1.0 + 1e-6):
            raise OperatorError(
                f"{self.unknown} max {pmax:.4g} exceeds param_max={self.param_max:.4g} used for "
                "the stability bound; raise param_max or bound the field"
            )
        if self.unknown == "Du" and float(param.detach().min()) <= 0:
            raise OperatorError("the diffusivity field must be positive")
        st = self._static(param.device, param.dtype)
        sp, bc = self.domain.spacing(), self.boundary
        which = self.unknown
        Fc, kc, Duc, Dv = self.F_const, self.k_const, self.Du, self.Dv

        def step(state, params, n, dt):
            (p,) = params
            Fv = p if which == "F" else Fc
            kv = p if which == "k" else kc
            Duv = p if which == "Du" else Duc
            return gray_scott_step(state[0], state[1], Fv, kv, Duv, Dv, dt, sp, bc)

        if observe is None:
            idx = [SPECIES.index(s) for s in self.observe]

            def observe(state, n):
                return torch.stack([state[i] for i in idx])

            record = self.record
        ck = None
        if self.grad_mode == "checkpoint":
            ck = self.checkpoint_every or auto_checkpoint_every(self.n_steps)
        return run_timestepping(
            step,
            (st["u0"], st["v0"]),
            (param,),
            self.n_steps,
            self.dt_sim,
            observe,
            checkpoint_every=ck,
            record=record,
        )

    def forward(self, fields: Fields) -> torch.Tensor:
        p = self.get_field(fields)
        _, obs = self.simulate(p)
        out = torch.stack(obs)  # (n_obs, n_species, *shape)
        return out.reshape(-1, *out.shape[2:])

    def extra_repr(self) -> str:
        return (
            f"shape={self.domain.shape}, unknown={self.unknown}, obs_times={self.obs_times}, "
            f"dt_sim={self.dt_sim:.4g} (x{self.m}), n_steps={self.n_steps}, "
            f"boundary={self.boundary}"
        )
