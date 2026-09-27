"""Wrap any differentiable Python callable as a nefi :class:`~nefi.operators.Operator`.

This is the entry point of "bring your own problem": your forward model is ordinary PyTorch code,

    def forward(x):                     # x: (*shape) tensor of the unknown field
        return my_physics(x)            # any differentiable torch ops

    op = FunctionOperator(forward)

and nefi treats it exactly like a built-in operator (hard physics constraint, autograd adjoint).

Multiscale curricula evaluate the field on coarser grids. A plain function usually only makes sense
at the native grid, so :class:`FunctionOperator` offers three explicit options:

* ``at_resolution=lambda shape: forward_for(shape)`` — you provide the forward for another grid
  (e.g. rebuild a kernel with the new pixel size);
* ``output_shape=lambda shape: ...`` — your function is resolution-agnostic (pointwise maps,
  physically-parameterized kernels) and the measurement is resampled to ``output_shape(shape)``;
* ``coarse="upsample"`` — coarse fields are upsampled to the native grid before calling your
  function (always correct, measurement never resampled; the coarse stage then acts as a
  smoothness curriculum for the field only).

With none of these the operator only accepts native-resolution fields and
:func:`nefi.auto.from_forward` uses a single-stage curriculum.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import torch
from torch import nn

from ..errors import OperatorError, ShapeError
from ..registry import register
from ..utils.tensor import resample, shape_tuple
from .base import Fields, Operator

log = logging.getLogger("nefi")

ShapeFn = Callable[[tuple[int, ...]], Sequence[int]]


def upsample_fields(
    fields: Fields, shape: Sequence[int], mode: str = "linear", names: Sequence[str] | None = None
) -> dict[str, torch.Tensor]:
    """Resample the trailing ``len(shape)`` dims of (selected) fields to ``shape``.

    Example::

        f = upsample_fields({"x": torch.rand(8, 8)}, (16, 16))
        assert f["x"].shape == (16, 16)
    """
    shape = shape_tuple(shape)
    d = len(shape)
    out = {}
    for k, v in fields.items():
        if (names is None or k in names) and v.ndim >= d and tuple(v.shape[-d:]) != shape:
            out[k] = resample(v, shape, mode=mode)
        else:
            out[k] = v
    return out


def field_shape(fields: Fields, name: str, ndim: int | None) -> tuple[int, ...] | None:
    """Trailing ``ndim`` dims of ``fields[name]`` (``None`` if unknown).

    Example::

        assert field_shape({"x": torch.zeros(3, 8, 8)}, "x", 2) == (8, 8)
    """
    v = fields.get(name)
    if v is None or ndim is None:
        return None
    return tuple(v.shape[-ndim:])


class UpsampleToNative(Operator):
    """Evaluate ``op`` on fields upsampled from a coarse grid to ``native_shape``.

    Generic, always-correct multiscale adapter: at coarse curriculum stages the field is sampled
    on the coarse grid (cheap, smooth) and linearly interpolated to the native grid before the
    physics runs; the measurement is used at full resolution. Returned by ``at_resolution`` of
    operators that have no cheaper coarse version.

    Example::

        fn = lambda x: x.sum(0)                    # a forward written for 64 × 64 fields
        op = UpsampleToNative(FunctionOperator(fn, native_shape=(64, 64)), (64, 64))
        y = op({"x": torch.rand(32, 32)})          # fn sees a (64, 64) field
    """

    def __init__(self, op: Operator, native_shape: Sequence[int], mode: str = "linear") -> None:
        super().__init__()
        self.op = op
        self.native_shape = shape_tuple(native_shape)
        self.mode = mode
        self.primary = op.primary
        self.homogeneity = op.homogeneity

    def forward(self, fields: Fields) -> torch.Tensor:
        names = self.op.required_fields()
        return self.op(upsample_fields(fields, self.native_shape, self.mode, names))

    def at_resolution(self, shape):
        return self.op if shape_tuple(shape) == self.native_shape else self

    def output_shape(self, shape):
        return self.op.output_shape(self.native_shape)

    def required_fields(self):
        return self.op.required_fields()


@register("operator", "function")
class FunctionOperator(Operator):
    """A differentiable callable as a forward operator.

    Args:
        fn: ``fn(x) -> Tensor`` taking the primary field ``(*shape)``, or ``fn(fields) -> Tensor``
            taking the dict of all fields when ``takes_dict=True``. May be an ``nn.Module``
            (its buffers follow ``.to(device)``; its parameters are frozen unless
            ``trainable=True``).
        field: name of the primary field (the unknown fed to ``fn``).
        takes_dict: pass the whole ``fields`` dict to ``fn``.
        fields: names of the fields ``fn`` needs (``takes_dict=True``; default ``(field,)``).
        homogeneity: degree ``p`` with ``fn(c·x) = c^p fn(x)`` if known (1 for linear physics).
        output_shape: callable ``field_shape -> measurement_shape`` (declares ``fn`` as
            resolution-agnostic) or a fixed tuple (native measurement shape).
        at_resolution: callable ``field_shape -> new_fn`` giving the forward for another grid.
        native_shape: the grid ``fn`` is written for (enables ``coarse="upsample"`` and checks).
        coarse: ``"upsample"`` to evaluate coarse fields through :class:`UpsampleToNative`;
            ``None`` (default) to require ``at_resolution``/``output_shape`` for coarse grids.
        trainable: optimize the parameters of an ``nn.Module`` ``fn`` alongside the field
            (calibration); default False freezes them.
        complex_output: ``"real_imag"`` converts complex outputs with ``torch.view_as_real``
            (trailing dim of size 2); ``"error"`` raises instead.
        name: label for reports.

    Example::

        kernel = torch.tensor([0.25, 0.5, 0.25]).view(1, 1, 3)
        blur = lambda x: torch.nn.functional.conv1d(x.view(1, 1, -1), kernel, padding=1).view(-1)
        op = FunctionOperator(blur, homogeneity=1.0, native_shape=(64,))
        y = op({"x": torch.rand(64)})
    """

    def __init__(
        self,
        fn: Callable,
        *,
        field: str = "x",
        takes_dict: bool = False,
        fields: Sequence[str] | None = None,
        homogeneity: float | None = None,
        output_shape: ShapeFn | Sequence[int] | None = None,
        at_resolution: Callable[[tuple[int, ...]], Callable] | None = None,
        native_shape: Sequence[int] | None = None,
        coarse: str | None = None,
        trainable: bool = False,
        complex_output: str = "real_imag",
        name: str | None = None,
    ) -> None:
        super().__init__()
        if not callable(fn):
            raise OperatorError(f"FunctionOperator needs a callable, got {type(fn).__name__}")
        if coarse not in (None, "upsample"):
            raise OperatorError(f"coarse must be None or 'upsample', got {coarse!r}")
        if complex_output not in ("real_imag", "error"):
            raise OperatorError("complex_output must be 'real_imag' or 'error'")
        if isinstance(fn, nn.Module):
            self.module = fn  # registered: follows .to(device); parameters frozen by default
            if not trainable:
                for p in fn.parameters():
                    p.requires_grad_(False)
        self.fn = fn
        self.primary = field
        self.takes_dict = takes_dict
        self._fields = tuple(fields) if fields is not None else (field,)
        self.homogeneity = homogeneity
        self._output_shape = output_shape
        self._at_res = at_resolution
        self.native_shape = None if native_shape is None else shape_tuple(native_shape)
        self.coarse = coarse
        self.trainable = trainable
        self.complex_output = complex_output
        self.name = name or getattr(fn, "__name__", type(fn).__name__)
        self._res_cache: dict[tuple[int, ...], FunctionOperator] = {}

    # ---- Operator API -----------------------------------------------------------------------
    def forward(self, fields: Fields) -> torch.Tensor:
        if self.takes_dict:
            missing = [k for k in self._fields if k not in fields]
            if missing:
                raise OperatorError(
                    f"forward {self.name!r} needs fields {missing}; available: {tuple(fields)}"
                )
            y = self.fn(dict(fields))
        else:
            y = self.fn(self.get_field(fields))
        if not torch.is_tensor(y):
            raise OperatorError(
                f"forward {self.name!r} returned {type(y).__name__}; it must return a torch.Tensor "
                "(use torch ops end-to-end so gradients can flow; no .numpy()/.item())"
            )
        if y.is_complex():
            if self.complex_output == "error":
                raise OperatorError(f"forward {self.name!r} returned a complex tensor")
            y = torch.view_as_real(y)
        return y

    @property
    def multiscale_capable(self) -> bool:
        """True when coarse grids are supported (``at_resolution``/``output_shape``/upsample)."""
        return self._at_res is not None or callable(self._output_shape) or self.coarse is not None

    def at_resolution(self, shape):
        shape = shape_tuple(shape)
        if self.native_shape is not None and shape == self.native_shape:
            return self
        if self._at_res is not None:
            if shape not in self._res_cache:
                new_fn = self._at_res(shape)
                self._res_cache[shape] = FunctionOperator(
                    new_fn,
                    field=self.primary,
                    takes_dict=self.takes_dict,
                    fields=self._fields,
                    homogeneity=self.homogeneity,
                    output_shape=self._output_shape if callable(self._output_shape) else None,
                    at_resolution=self._at_res,
                    native_shape=shape,
                    trainable=self.trainable,
                    complex_output=self.complex_output,
                    name=f"{self.name}@{shape}",
                )
            return self._res_cache[shape]
        if self.coarse == "upsample":
            if self.native_shape is None:
                raise OperatorError("coarse='upsample' needs native_shape=...")
            return UpsampleToNative(self, self.native_shape)
        if self.native_shape is not None and not callable(self._output_shape):
            raise ShapeError(
                f"forward {self.name!r} is defined on the native grid {self.native_shape} but a "
                f"curriculum stage asked for {shape}. Pass at_resolution=lambda shape: ..., "
                "output_shape=lambda shape: ... (resolution-agnostic function), "
                "coarse='upsample', or use a single-stage curriculum"
            )
        return self

    def output_shape(self, shape):
        shape = shape_tuple(shape)
        if callable(self._output_shape):
            return shape_tuple(self._output_shape(shape))
        if self._output_shape is not None:
            fixed = shape_tuple(self._output_shape)
            if self.native_shape is None or shape == self.native_shape or self.coarse:
                return fixed  # with coarse='upsample' the prediction always has native shape
        return None

    def required_fields(self):
        return self._fields

    def extra_repr(self) -> str:
        return f"fn={self.name}, field={self.primary!r}, homogeneity={self.homogeneity}"


def as_operator(forward: Operator | Callable, field: str = "x", **kw: object) -> Operator:
    """Return ``forward`` if it is an :class:`Operator`, else wrap it in :class:`FunctionOperator`.

    Example::

        op = as_operator(torch.square, homogeneity=2.0)
    """
    if isinstance(forward, Operator):
        return forward
    return FunctionOperator(forward, field=field, **kw)  # type: ignore[arg-type]


__all__ = ["FunctionOperator", "UpsampleToNative", "as_operator", "field_shape", "upsample_fields"]
