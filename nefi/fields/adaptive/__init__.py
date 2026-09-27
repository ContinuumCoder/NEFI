"""Adaptive representation tools: let the data and the operator choose the geometry.

=================================  ===============================================================
Tool                               What it does
=================================  ===============================================================
:class:`ResidualDrivenAnnealing`   open the next frequency band only on data-loss plateaus
:class:`OperatorAwareAnnealing`    open band ``ν`` only if ``σ_F(ν)·|X̂(ν)|`` exceeds the noise
:class:`BandwidthSchedule`         prescribe the open bandwidth (cycles/unit) per stage fraction
:class:`GrowCapacity`              add modes / layers / shapes / bands when the residual plateaus
:func:`select_representation`      hold-out cross-validation over candidate representations
:class:`RepresentationEnsemble`    softmax(−held-out loss / T)-weighted average of fitted fields
:func:`representation_spectrum`    radial transfer of the update kernel ``G_θ = J_θ J_θᵀ``
:func:`operator_spectrum`          radial amplitude sensitivity ``σ_F`` (``JᵀJ`` probes)
:func:`match_report`               over-/under-bandlimited verdict at the noise level, in words
=================================  ===============================================================

All callbacks drive ``solver.progress_override`` / ``solver.stop_stage`` (the adaptive hooks of
:class:`~nefi.solve.Solver`). Importing this package registers the callbacks
``"residual_annealing"``, ``"operator_annealing"``, ``"bandwidth_schedule"``, ``"grow_capacity"``
and the field ``"representation_ensemble"``.
"""

from .annealing import (
    BandwidthSchedule,
    OperatorAwareAnnealing,
    PlateauDetector,
    ResidualDrivenAnnealing,
    band_frequencies,
    bandwidth_to_progress,
    find_band_module,
    n_levels,
    progress_to_bandwidth,
)
from .capacity import Growable, GrowCapacity, field_space_gradient, growable_modules
from .selection import (
    CandidateScore,
    RepresentationEnsemble,
    SelectionReport,
    build_problem,
    holdout_masks,
    masked_downsampler,
    select_representation,
    softmax_weights,
)
from .spectrum import (
    MatchReport,
    Spectrum,
    estimate_noise_std,
    field_spectrum,
    kernel_row,
    match_report,
    operator_spectrum,
    radial_average,
    representation_spectrum,
    resolvable_bandwidth,
)

__all__ = [
    "BandwidthSchedule",
    "CandidateScore",
    "GrowCapacity",
    "Growable",
    "MatchReport",
    "OperatorAwareAnnealing",
    "PlateauDetector",
    "RepresentationEnsemble",
    "ResidualDrivenAnnealing",
    "SelectionReport",
    "Spectrum",
    "band_frequencies",
    "bandwidth_to_progress",
    "build_problem",
    "estimate_noise_std",
    "field_space_gradient",
    "field_spectrum",
    "find_band_module",
    "growable_modules",
    "holdout_masks",
    "kernel_row",
    "masked_downsampler",
    "match_report",
    "n_levels",
    "operator_spectrum",
    "progress_to_bandwidth",
    "radial_average",
    "representation_spectrum",
    "resolvable_bandwidth",
    "select_representation",
    "softmax_weights",
]
