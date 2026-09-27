"""Field parameterizations (the prior) and their building blocks."""

from .base import Field
from .encoding import Encoding, FourierFeatures, IdentityEncoding
from .grid import GridField
from .heads import Bounded, Exp, GatedSoftplus, Head, Heads, Identity, Softplus, SupportMasked
from .neural import NeuralField

__all__ = [
    "Bounded",
    "Encoding",
    "Exp",
    "Field",
    "FourierFeatures",
    "GatedSoftplus",
    "GridField",
    "Head",
    "Heads",
    "Identity",
    "IdentityEncoding",
    "NeuralField",
    "Softplus",
    "SupportMasked",
]

# --- representation zoo (BYOP layer) -----------------------------------------------------------
from .hashgrid import HashGridEncoding, HashGridField  # noqa: E402
from .levelset import LevelSetField, LevelSetHead, interface_length  # noqa: E402
from .lowrank import LowRankField  # noqa: E402
from .modifiers import (  # noqa: E402
    Affine,
    ExpHead,
    MaskedHead,
    MassNormalized,
    ScaledHead,
    ZeroMean,
)
from .parametric import ParametricField, ellipses, gaussian_blobs  # noqa: E402
from .symmetric import SymmetricField  # noqa: E402

__all__ += [
    "Affine",
    "ExpHead",
    "HashGridEncoding",
    "HashGridField",
    "LevelSetField",
    "LevelSetHead",
    "LowRankField",
    "MaskedHead",
    "MassNormalized",
    "ParametricField",
    "ScaledHead",
    "SymmetricField",
    "ZeroMean",
    "ellipses",
    "gaussian_blobs",
    "interface_length",
]
