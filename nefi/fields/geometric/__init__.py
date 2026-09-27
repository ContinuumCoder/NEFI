"""Geometric representations: fields whose *structure* encodes the geometry of the unknown.

The representation is the geometric prior (NeTMY Lemma 2 / Eq. 7): a first-order step on the
parameters realizes ``Δx ≈ −η G_θ ∇_x L`` with ``G_θ = J_θ J_θᵀ``, so the parameterization filters
the raw field-space gradient. This package provides a family of parameterizations whose filters
match common geometry classes (see ``docs/geometry.md`` for the decision guide):

=================================  ===============================================================
Class                              Geometry / use
=================================  ===============================================================
:class:`CompositeField`            sum / product / blend of sub-fields, per-component curricula
:class:`AnomalyField`              background ⊕ gated anomaly with a support prior
:class:`LayeredField`              stratified media, monotone learnable interfaces (NeFTY layered)
:class:`LayerCakeField`            defects in a laminate (layered background ⊕ anomaly)
:class:`StarShapeField`            inclusions with Fourier-descriptor boundaries (2-D / extruded)
:class:`PolygonField`              polygonal (box-like) inclusions
:class:`WarpedField`               inner field in warped coordinates (polar, depth, sensitivity)
:class:`DeformableField`           template + learned smooth displacement
:class:`FourierBasisField`         explicit low-frequency basis (smooth background)
:class:`SpectralPreconditionedField` Fourier-gain preconditioning of any field
=================================  ===============================================================

Importing this package registers the fields (``build("field", "layered", ...)``), the losses
``"contour"`` and ``"warp_regularizer"``.
"""

from ._utils import (
    ProgressWindow,
    anneal_value,
    interp_grid,
    level_set_head,
    progress_fn,
    smooth_step,
)
from .composite import (
    COMBINERS,
    AnomalyField,
    Combiner,
    CompositeField,
    register_combiner,
    resolve_combiner,
)
from .layered import LateralModel, LayerCakeField, LayeredField, default_anomaly_field
from .shape import (
    ContourRegularizer,
    PolygonField,
    StarShapeField,
    contour_regularizer,
    hint_location,
    interface_length,
)
from .spectral import (
    FourierBasisField,
    SpectralPreconditionedField,
    inverse_sensitivity_gain,
    lowpass_gain,
    profile_gain,
)
from .warp import (
    ComposeWarp,
    CoordinateWarp,
    DeformableField,
    DepthStretchWarp,
    DisplacementWarp,
    IdentityWarp,
    LogDepthWarp,
    PolarWarp,
    SensitivityWarp,
    WarpedField,
    WarpRegularizer,
    cylindrical,
    depth_stretch,
    log_depth,
    polar,
    sensitivity_warp,
    warp_regularizer,
)

__all__ = [
    "COMBINERS",
    "AnomalyField",
    "Combiner",
    "ComposeWarp",
    "CompositeField",
    "ContourRegularizer",
    "CoordinateWarp",
    "DeformableField",
    "DepthStretchWarp",
    "DisplacementWarp",
    "FourierBasisField",
    "IdentityWarp",
    "LateralModel",
    "LayerCakeField",
    "LayeredField",
    "LogDepthWarp",
    "PolarWarp",
    "PolygonField",
    "ProgressWindow",
    "SensitivityWarp",
    "SpectralPreconditionedField",
    "StarShapeField",
    "WarpRegularizer",
    "WarpedField",
    "anneal_value",
    "contour_regularizer",
    "cylindrical",
    "default_anomaly_field",
    "depth_stretch",
    "hint_location",
    "interface_length",
    "interp_grid",
    "inverse_sensitivity_gain",
    "level_set_head",
    "log_depth",
    "lowpass_gain",
    "polar",
    "profile_gain",
    "progress_fn",
    "register_combiner",
    "resolve_combiner",
    "sensitivity_warp",
    "smooth_step",
    "warp_regularizer",
]
