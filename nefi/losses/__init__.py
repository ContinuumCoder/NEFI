"""Loss terms and their composition."""

from .base import Context, Loss, LossSet
from .data import L2, MSE, RMSE, DataLoss, Huber, LogMSE, NormalizedMSE, PoissonNLL, RelativeMSE
from .reg import L1, TV, FieldLoss, Laplacian, PriorMSE, RangePenalty, Tikhonov, forward_differences

__all__ = [
    "L1",
    "L2",
    "MSE",
    "RMSE",
    "TV",
    "Context",
    "DataLoss",
    "FieldLoss",
    "Huber",
    "Laplacian",
    "LogMSE",
    "Loss",
    "LossSet",
    "NormalizedMSE",
    "PoissonNLL",
    "PriorMSE",
    "RangePenalty",
    "RelativeMSE",
    "Tikhonov",
    "forward_differences",
]

# --- physics-knowledge losses (BYOP layer) -----------------------------------------------------
from .physics import (  # noqa: E402
    Conservation,
    GradientL2,
    KnownSupportLoss,
    Monotone,
    PDEResidual,
    RangeStat,
    SymmetryLoss,
)

__all__ += [
    "Conservation",
    "GradientL2",
    "KnownSupportLoss",
    "Monotone",
    "PDEResidual",
    "RangeStat",
    "SymmetryLoss",
]
