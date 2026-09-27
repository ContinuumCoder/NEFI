"""Forward-operator base class and generic wrappers.

An :class:`Operator` is differentiable physics: ``fields -> prediction``. It is the *hard
constraint*
of the inverse problem (NeFTY §4.2), never a soft residual. Operators must be pure functions of
their inputs and parameters so that diagnostics can differentiate through them.

Multiscale rule: :meth:`Operator.at_resolution` must return an operator that evaluates fields
sampled at the given grid shape and **shares** any ``nn.Parameter`` with the original (so nuisance
parameters keep being optimized across curriculum stages).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
from torch import nn

from ..errors import OperatorError
from ..registry import register
from ..utils.tensor import shape_tuple

Fields = Mapping[str, torch.Tensor]


class Operator(nn.Module):
    """Base class for differentiable forward models.

    Attributes:
        primary: name of the field the homogeneity statement refers to.
        homogeneity: degree ``p`` such that ``F(c·x) = c^p F(x)`` in the primary field
            (1 for linear operators such as NeTMY F2 and the heat map in the initial data;
            2 for NeTMY F1; ``None`` when not homogeneous / unknown). Used by energy-anchored scale
            correction (NeTMY Eq. 30).
        batchable: ``forward`` accepts fields with a leading batch axis ``(B, *shape)`` and
            returns ``(B, *output_shape)`` — row ``b`` equal to the unbatched output of field
            ``b`` — using only differentiable tensor ops (no custom ``autograd.Function``, no
            per-sample Python control flow on values). Required by
            :func:`nefi.solve.batch_invert`. Default ``False``; set ``True`` only after checking
            (``tests/test_performance.py`` verifies every batchable operator).
        traceable: ``torch.compile`` may trace ``forward`` into a whole-step graph
            (``Solver(compile="step")``). Operators with long Python time loops, iterative
            solvers with host-side convergence tests or custom ``autograd.Function`` s set this
            to ``False`` and are run eagerly between the compiled field and loss graphs.
    """

    primary: str = "x"
    homogeneity: float | None = None
    batchable: bool = False
    traceable: bool = True

    def forward(self, fields: Fields) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def at_resolution(self, shape: Sequence[int]) -> Operator:
        """Operator evaluating fields sampled at ``shape``. Default: resolution-agnostic (self)."""
        return self

    def output_shape(self, shape: Sequence[int]) -> tuple[int, ...] | None:
        """Prediction shape for fields sampled at ``shape`` (``None`` if not statically known)."""
        return None

    def required_fields(self) -> tuple[str, ...]:
        return (self.primary,)

    def get_field(self, fields: Fields, name: str | None = None) -> torch.Tensor:
        name = name or self.primary
        try:
            return fields[name]
        except KeyError as e:
            raise OperatorError(
                f"{type(self).__name__} needs field {name!r}; available: {tuple(fields)}"
            ) from e

    # convenience for scalar-callers / diagnostics
    def apply(self, x: torch.Tensor, **others: torch.Tensor) -> torch.Tensor:
        return self({self.primary: x, **others})


@register("operator", "lambda")
class LambdaOperator(Operator):
    """Wrap a plain function ``f(fields) -> Tensor`` (prototyping / tests)."""

    def __init__(
        self,
        fn: Callable[[Fields], torch.Tensor],
        primary: str = "x",
        homogeneity: float | None = None,
        output_shape_fn=None,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.primary = primary
        self.homogeneity = homogeneity
        self._out_shape = output_shape_fn

    def forward(self, fields: Fields) -> torch.Tensor:
        return self.fn(fields)

    def output_shape(self, shape):
        return None if self._out_shape is None else shape_tuple(self._out_shape(shape_tuple(shape)))


@register("operator", "nuisance")
class Nuisance(Operator):
    """Learnable global gain / offset around an inner operator: ``exp(log_gain)·F(x) + offset``.

    A cheap, robust way to absorb calibration mismatch on real data (unknown detector gain,
    background level) without polluting the field. Parameters are shared across resolutions.
    """

    @property
    def batchable(self) -> bool:  # type: ignore[override]
        return bool(getattr(self.inner, "batchable", False))

    @property
    def traceable(self) -> bool:  # type: ignore[override]
        return bool(getattr(self.inner, "traceable", True))

    def __init__(
        self,
        inner: Operator,
        gain: bool = True,
        offset: bool = False,
        init_gain: float = 1.0,
        init_offset: float = 0.0,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.primary = inner.primary
        self.homogeneity = inner.homogeneity
        self.log_gain = nn.Parameter(torch.tensor(float(init_gain)).log(), requires_grad=gain)
        self.offset = nn.Parameter(torch.tensor(float(init_offset)), requires_grad=offset)

    def forward(self, fields: Fields) -> torch.Tensor:
        return torch.exp(self.log_gain) * self.inner(fields) + self.offset

    def at_resolution(self, shape):
        new = Nuisance.__new__(Nuisance)
        nn.Module.__init__(new)
        new.inner = self.inner.at_resolution(shape)
        new.primary, new.homogeneity = self.primary, self.homogeneity
        new.log_gain, new.offset = self.log_gain, self.offset  # shared parameters
        return new

    def output_shape(self, shape):
        return self.inner.output_shape(shape)

    def required_fields(self):
        return self.inner.required_fields()

    @property
    def gain(self) -> float:
        return float(torch.exp(self.log_gain.detach()))


class Sequential(Operator):
    """Compose operators: the output of each becomes the primary field of the next."""

    @property
    def batchable(self) -> bool:  # type: ignore[override]
        return all(getattr(op, "batchable", False) for op in self.ops)

    @property
    def traceable(self) -> bool:  # type: ignore[override]
        return all(getattr(op, "traceable", True) for op in self.ops)

    def __init__(self, *ops: Operator) -> None:
        super().__init__()
        if not ops:
            raise OperatorError("Sequential needs at least one operator")
        self.ops = nn.ModuleList(ops)
        self.primary = ops[0].primary
        hs = [op.homogeneity for op in ops]
        self.homogeneity = None if any(h is None for h in hs) else float(torch.tensor(hs).prod())

    def forward(self, fields: Fields) -> torch.Tensor:
        y = self.ops[0](fields)
        for op in self.ops[1:]:
            y = op({op.primary: y})
        return y

    def at_resolution(self, shape):
        return Sequential(*[op.at_resolution(shape) for op in self.ops])

    def required_fields(self):
        return self.ops[0].required_fields()
