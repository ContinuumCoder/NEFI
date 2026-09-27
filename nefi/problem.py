"""The inverse problem: domain + field + operator + losses + measurement (+ post-processing)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from .domain import Domain
from .errors import ConfigError
from .fields.base import Field
from .losses.base import Context, LossSet
from .measurement import Measurement
from .operators.base import Operator
from .solve.curriculum import Curriculum
from .solve.postprocess import Postprocess
from .utils.tensor import shape_tuple


def _cast_real_floats(module: torch.nn.Module, dtype: torch.dtype) -> None:
    """Cast real floating-point parameters and buffers of ``module`` to ``dtype`` in place."""

    def _convert(t: torch.Tensor) -> torch.Tensor:
        if t.is_floating_point() and not t.is_complex():
            return t.to(dtype)
        return t

    module._apply(_convert)


@dataclass
class InverseProblem:
    """A fully specified per-measurement inverse problem.

    Args:
        domain: physical domain of the unknown field(s).
        field: parameterization of the unknown(s) (the prior).
        operator: differentiable forward model (the hard physics constraint).
        losses: data-fidelity + regularization terms.
        measurement: the observation.
        postprocess: one-shot post-processors applied after optimization.
        curriculum: default curriculum used by :func:`nefi.invert` / :class:`Solver` when none is
            passed explicitly (instances set their paper defaults here).
        downsample_obs: optional ``(measurement, field_shape) -> Measurement`` used at coarse
            curriculum stages; default resamples to ``operator.output_shape(field_shape)``.
        name: label used in reports.
    """

    domain: Domain
    field: Field
    operator: Operator
    losses: LossSet
    measurement: Measurement
    postprocess: Sequence[Postprocess] = ()
    curriculum: Curriculum | None = None
    downsample_obs: Callable[[Measurement, tuple[int, ...]], Measurement] | None = None
    name: str = "problem"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [f for f in self.operator.required_fields() if f not in self.field.names]
        if missing:
            raise ConfigError(
                f"operator requires fields {missing} but the field provides {self.field.names}"
            )

    # --- device --------------------------------------------------------------------------
    def to(self, device=None, dtype=None) -> InverseProblem:
        """Move modules to ``device`` and cast *real* floating tensors to ``dtype``.

        Complex parameters/buffers (e.g. FFT kernels, transfer functions) keep their complex
        dtype — ``nn.Module.to(dtype)`` would silently cast them to a real dtype.
        """
        for mod in (self.field, self.operator, self.losses):
            if device is not None:
                mod.to(device=device)
            if dtype is not None:
                _cast_real_floats(mod, dtype)
        return self

    @property
    def device(self) -> torch.device:
        try:
            return next(self.field.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.field.parameters()).dtype
        except StopIteration:
            return torch.get_default_dtype()

    # --- measurement at stage resolution ---------------------------------------------------
    def measurement_at(self, shape: Sequence[int]) -> Measurement:
        shape = shape_tuple(shape)
        if shape == self.domain.shape:
            return self.measurement
        if self.downsample_obs is not None:
            return self.downsample_obs(self.measurement, shape)
        out = self.operator.output_shape(shape)
        if out is None:
            raise ConfigError(
                f"cannot infer the measurement shape for field resolution {shape}: implement "
                "Operator.output_shape or pass InverseProblem(downsample_obs=...)"
            )
        return self.measurement.resampled(out)

    # --- evaluation ------------------------------------------------------------------------
    def evaluate(
        self, shape: Sequence[int] | None = None, progress: float = 1.0, grad: bool = False
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Fields and prediction at ``shape`` (default native resolution)."""
        dom = self.domain if shape is None else self.domain.at(shape)
        coords = dom.coords(device=self.device, dtype=self.dtype)
        op = self.operator.at_resolution(dom.shape)
        with torch.set_grad_enabled(grad):
            fields = self.field(coords, progress)
            pred = op(fields)
        return fields, pred

    def context(self, shape=None, progress: float = 1.0, stage=None, step: int = 0) -> Context:
        dom = self.domain if shape is None else self.domain.at(shape)
        coords = dom.coords(device=self.device, dtype=self.dtype)
        op = self.operator.at_resolution(dom.shape)
        fields = self.field(coords, progress)
        pred = op(fields)
        obs = self.measurement_at(dom.shape).to(self.device, self.dtype)
        return Context(fields, pred, obs, dom, op, self.field, stage, step, progress)

    def loss(self, shape=None, progress: float = 1.0, stage=None, step: int = 0):
        """Total loss and components at the current parameters (with autograd)."""
        ctx = self.context(shape, progress, stage, step)
        losses = self.losses if stage is None else self.losses.with_weights(stage.loss_weights)
        return losses(ctx)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "domain": self.domain.to_dict(),
            "field": repr(self.field),
            "operator": type(self.operator).__name__,
            "losses": {k: self.losses.weights[k] for k in self.losses.names},
            "measurement_shape": list(self.measurement.shape),
            "postprocess": [type(p).__name__ for p in self.postprocess],
        }
