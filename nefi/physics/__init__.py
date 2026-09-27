"""Physics zoo: reusable differentiable forward models for many physical systems.

Modules cover the elliptic family ( variable-coefficient
Poisson with implicit-function-theorem adjoint, EIT, Darcy, magnetostatics; wave/optics family:
acoustic wave with absorbing boundaries, Born diffraction tomography, angular-spectrum optics,
reaction–diffusion). Import the submodules directly, e.g. ``from nefi.physics import elliptic``.
"""

__all__: list[str] = []

# --- wave / optics / reaction–diffusion family (nefi.physics.{timestep,wave,scattering,optics,
# reaction_diffusion}); importing registers the operators "wave", "wave_initial_condition", "born",
# "lippmann_schwinger", "phase_object", "holography" and "gray_scott".
from . import optics, reaction_diffusion, scattering, timestep, wave  # noqa: E402
from .optics import HolographyOperator, PhaseObject, gerchberg_saxton, propagate  # noqa: E402
from .reaction_diffusion import GrayScottIC, ReactionDiffusionOperator  # noqa: E402
from .scattering import (  # noqa: E402
    BornOperator,
    LippmannSchwingerOperator,
    filtered_backpropagation,
)
from .timestep import run_timestepping, stable_dt_diffusion, stable_dt_wave  # noqa: E402
from .wave import WaveInitialConditionOperator, WaveOperator, time_reversal  # noqa: E402

__all__ += [
    "BornOperator",
    "GrayScottIC",
    "HolographyOperator",
    "LippmannSchwingerOperator",
    "PhaseObject",
    "ReactionDiffusionOperator",
    "WaveInitialConditionOperator",
    "WaveOperator",
    "filtered_backpropagation",
    "gerchberg_saxton",
    "optics",
    "propagate",
    "reaction_diffusion",
    "run_timestepping",
    "scattering",
    "stable_dt_diffusion",
    "stable_dt_wave",
    "time_reversal",
    "timestep",
    "wave",
]

# --- elliptic family (nefi.physics.{elliptic,magnetostatics}); importing registers the operators
# "elliptic", "current_density", "magnetization", "biot_savart" and the head "log_bounded".
from . import elliptic, magnetostatics  # noqa: E402
from .elliptic import (  # noqa: E402
    EllipticOperator,
    ImplicitSolveFunction,
    LogBounded,
    apply_divgrad,
    solve_elliptic,
)
from .magnetostatics import (  # noqa: E402
    BiotSavartOperator,
    CurrentDensityOperator,
    MagnetizationOperator,
    fourier_inversion,
    magnetic_constant,
    upward_continuation,
)

__all__ += [
    "BiotSavartOperator",
    "CurrentDensityOperator",
    "EllipticOperator",
    "ImplicitSolveFunction",
    "LogBounded",
    "MagnetizationOperator",
    "apply_divgrad",
    "elliptic",
    "fourier_inversion",
    "magnetic_constant",
    "magnetostatics",
    "solve_elliptic",
    "upward_continuation",
]
