"""Ill-posedness and optimization-geometry diagnostics.

Every diagnostic takes an :class:`~nefi.problem.InverseProblem` and works for any field /
operator through autograd (forward-mode JVPs with double-backward and finite-difference
fallbacks), because papers of this kind live or die by *explaining* why a method works:

=============================  =================================================================
Function                        What it measures (paper reference)
=============================  =================================================================
:func:`sensitivity_map`         Jacobian column norms ``‖∂F/∂x_i‖`` (NeTMY Eq. 23, (P2))
:func:`center_to_outer_ratio`   center vs outer-ring magnitude (NeTMY §5.3: 18.29×)
:func:`field_gradient`          raw field-space gradient ``∇ₓL`` (what a grid solver executes)
:func:`iter0_gradient`          gradient at a uniform init + center bias (NeTMY (P2)+(P3))
:func:`center_mass_ratio`       mass fraction in a central disk (NeTMY Fig. 4c)
:func:`energy_barrier`          loss along ``(1−t)x_a + t x_b`` (NeTMY Fig. 4b: h ≈ 1.12)
:func:`filter_kernel_row`       ``G_θ e_i = J_θ J_θᵀ e_i`` (NeTMY Lemma 2, Eq. 32–34)
:func:`realized_update`         ``|Δx|`` after one step vs ``|∇ₓL|`` (NeTMY App. E.5: 1.6×/18.29×)
:func:`effective_bandwidth`     annealed encoding bandwidth ``B_β`` (NeTMY Eq. 37)
:func:`singular_values`         top-k singular values of ``dF/dx`` (NeFTY Prop. 2 / Cor. 1)
:func:`hessian_condition_number` Hessian conditioning on an ansatz (NeTMY Fig. 5: 931 vs 301,139)
:func:`data_fit_paradox`        data fit next to field accuracy (NeFTY §5.2, App. G.2)
:func:`diagnose`                all of the above → :class:`DiagnosticsReport` (``to_markdown``)
=============================  =================================================================

Plot helpers live in :mod:`nefi.diagnostics.plots` (matplotlib optional).
"""

from ._common import jvp, normalized_radius
from .filtering import (
    RealizedUpdate,
    effective_bandwidth,
    filter_kernel_row,
    kernel_spread,
    realized_update,
)
from .landscape import (
    EnergyBarrier,
    Iter0Gradient,
    ansatz_objective,
    center_mass_ratio,
    energy_barrier,
    field_gradient,
    interpolate_fields,
    iter0_gradient,
    uniform_center_mass,
    uniform_fields,
)
from .plots import (
    plot_energy_barrier,
    plot_filter_kernels,
    plot_history,
    plot_report,
    plot_result,
    plot_sensitivity,
    plot_singular_values,
    save_figure,
    show_field,
)
from .report import DIAGNOSTICS, DiagnosticsReport, data_fit_paradox, diagnose
from .sensitivity import center_to_outer_ratio, sensitivity_map
from .spectrum import (
    hessian_condition_number,
    hessian_spectrum,
    lanczos_eigs,
    singular_values,
)

__all__ = [
    "DIAGNOSTICS",
    "DiagnosticsReport",
    "EnergyBarrier",
    "Iter0Gradient",
    "RealizedUpdate",
    "ansatz_objective",
    "center_mass_ratio",
    "center_to_outer_ratio",
    "data_fit_paradox",
    "diagnose",
    "effective_bandwidth",
    "energy_barrier",
    "field_gradient",
    "filter_kernel_row",
    "hessian_condition_number",
    "hessian_spectrum",
    "interpolate_fields",
    "iter0_gradient",
    "jvp",
    "kernel_spread",
    "lanczos_eigs",
    "normalized_radius",
    "plot_energy_barrier",
    "plot_filter_kernels",
    "plot_history",
    "plot_report",
    "plot_result",
    "plot_sensitivity",
    "plot_singular_values",
    "realized_update",
    "save_figure",
    "sensitivity_map",
    "show_field",
    "singular_values",
    "uniform_center_mass",
    "uniform_fields",
]
