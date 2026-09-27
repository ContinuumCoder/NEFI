"""Differentiable forward operators."""

from .base import Fields, LambdaOperator, Nuisance, Operator, Sequential
from .conv import FFTConvolution, fft_convolve, gaussian_kernel_fn, kernel_offsets, prepare_kernel

__all__ = [
    "FFTConvolution",
    "Fields",
    "LambdaOperator",
    "Nuisance",
    "Operator",
    "Sequential",
    "fft_convolve",
    "gaussian_kernel_fn",
    "kernel_offsets",
    "prepare_kernel",
]

# --- operator zoo (BYOP layer) -----------------------------------------------------------------
from .function import FunctionOperator, UpsampleToNative, as_operator  # noqa: E402
from .linear import (  # noqa: E402
    Downsample,
    FourierSampling,
    Identity,
    Pointwise,
    Sampling,
    Stack,
    Sum,
    random_kspace_mask,
)
from .nonlinear import BeerLambert, PhaseRetrieval, Saturation  # noqa: E402
from .timestepping import (  # noqa: E402
    TimeStepper,
    advection_diffusion_step,
    stable_dt,
    wave_initial_state,
    wave_step,
)

__all__ += [
    "BeerLambert",
    "Downsample",
    "FourierSampling",
    "FunctionOperator",
    "Identity",
    "PhaseRetrieval",
    "Pointwise",
    "Sampling",
    "Saturation",
    "Stack",
    "Sum",
    "TimeStepper",
    "UpsampleToNative",
    "advection_diffusion_step",
    "as_operator",
    "random_kspace_mask",
    "stable_dt",
    "wave_initial_state",
    "wave_step",
]
