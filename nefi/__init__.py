"""nefi: Neural-Field Inversion.

nefi recovers hidden physical fields from a single measurement by optimizing a coordinate neural
field through a differentiable forward operator of the instrument. No training data is needed.

Quick start::

    import nefi
    from nefi.instances.toy1d import make_problem

    problem, gt, meas = make_problem()
    result = nefi.invert(problem)           # default: 2-stage multiscale curriculum
    print(nefi.metrics.psnr(result.fields["x"], gt["x"]))
"""

from __future__ import annotations

import logging as _logging

from . import fields, losses, metrics, operators, solve
from .config import config_hash, load_config, save_config
from .domain import Domain
from .errors import NefiError
from .fields import Bounded, GatedSoftplus, GridField, Heads, NeuralField, Softplus, SupportMasked
from .losses import L1, MSE, TV, LogMSE, LossSet, NormalizedMSE
from .measurement import Measurement
from .operators import FFTConvolution, Nuisance, Operator
from .problem import InverseProblem
from .registry import build, list_registered, register
from .solve import (
    Curriculum,
    EnergyScaleCorrection,
    OptimConfig,
    RefineReport,
    Result,
    Solver,
    Stage,
    batch_invert,
    ensemble,
    refine_edges,
)

__version__ = "0.1.0"

_logging.getLogger("nefi").addHandler(_logging.NullHandler())

from . import instances  # noqa: E402  (registers built-in instances)


def invert(problem: InverseProblem, curriculum: Curriculum | None = None, **solver_kw) -> Result:
    """One-call entry point: run ``Solver(problem, curriculum, **solver_kw).run()``."""
    return Solver(problem, curriculum, **solver_kw).run()


def enable_logging(level: int = _logging.INFO) -> None:
    """Send nefi logs to stderr (idempotent)."""
    logger = _logging.getLogger("nefi")
    if not any(
        isinstance(h, _logging.StreamHandler) and not isinstance(h, _logging.NullHandler)
        for h in logger.handlers
    ):
        h = _logging.StreamHandler()
        h.setFormatter(_logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(level)


__all__ = [
    "L1",
    "MSE",
    "TV",
    "Bounded",
    "Curriculum",
    "Domain",
    "EnergyScaleCorrection",
    "FFTConvolution",
    "GatedSoftplus",
    "GridField",
    "Heads",
    "InverseProblem",
    "LogMSE",
    "LossSet",
    "Measurement",
    "NefiError",
    "NeuralField",
    "NormalizedMSE",
    "Nuisance",
    "Operator",
    "OptimConfig",
    "RefineReport",
    "Result",
    "Softplus",
    "Solver",
    "Stage",
    "SupportMasked",
    "batch_invert",
    "build",
    "config_hash",
    "enable_logging",
    "ensemble",
    "fields",
    "instances",
    "list_registered",
    "load_config",
    "losses",
    "metrics",
    "operators",
    "register",
    "save_config",
    "invert",
    "refine_edges",
    "solve",
]

# --- bring-your-own-problem layer: prior catalogue + adaptive defaults -------------------------
from . import auto, priors  # noqa: E402
from .auto import from_forward, quick_report  # noqa: E402
from .priors import Prior  # noqa: E402

__all__ += ["Prior", "auto", "from_forward", "priors", "quick_report"]

# --- auto-tuning layer: gauges, acquisition report, budget / LR, regularization, search --------
from . import autotune  # noqa: E402
from .autotune import autotune_problem  # noqa: E402

__all__ += ["autotune", "autotune_problem"]
