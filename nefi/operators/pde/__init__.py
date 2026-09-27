"""Differentiable PDE operators: finite-volume heat conduction with a discrete adjoint (NeFTY)."""

from .adjoint import (
    ASSEMBLY_MODES,
    GRAD_MODES,
    SOLVERS,
    HeatSolveConfig,
    ImplicitEulerAdjoint,
    implicit_euler_frames,
    rollout,
    solve_step,
    surface,
)
from .heat import (
    ExplicitHeatSimulator,
    GaussianFlash,
    HeatOperator,
    InitialCondition,
    UniformFlash,
    explicit_substeps,
)
from .linear_solvers import conjugate_gradient, jacobi
from .stencil import (
    BoundarySpec,
    DiffusionStencil,
    ImplicitSystem,
    apply_A,
    apply_diffusion,
    diagonal_A,
    face_coefficients,
    face_conductances,
    neighbors,
)

__all__ = [
    "ASSEMBLY_MODES",
    "GRAD_MODES",
    "SOLVERS",
    "BoundarySpec",
    "DiffusionStencil",
    "ExplicitHeatSimulator",
    "GaussianFlash",
    "HeatOperator",
    "HeatSolveConfig",
    "ImplicitEulerAdjoint",
    "ImplicitSystem",
    "InitialCondition",
    "UniformFlash",
    "apply_A",
    "apply_diffusion",
    "conjugate_gradient",
    "diagonal_A",
    "explicit_substeps",
    "face_coefficients",
    "face_conductances",
    "implicit_euler_frames",
    "jacobi",
    "neighbors",
    "rollout",
    "solve_step",
    "surface",
]
