"""Nonlinear measurement operators: phase retrieval, Beer–Lambert attenuation, detector saturation.

Example::

    import nefi
    from nefi.operators import FFTConvolution, Sequential, gaussian_kernel_fn
    from nefi.operators.nonlinear import BeerLambert, PhaseRetrieval, Saturation

    dom = nefi.Domain.unit((32, 32))
    pr = PhaseRetrieval(oversample=2)                  # |F x|², homogeneity 2
    xray = BeerLambert(axis=0, domain=dom)            # I = exp(-∫ μ dz) along axis 0
    camera = Sequential(FFTConvolution(gaussian_kernel_fn(0.02), dom), Saturation("tanh"))
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from ..domain import Domain
from ..errors import OperatorError
from ..registry import register
from ..utils.tensor import shape_tuple
from .base import Fields, Operator
from .function import upsample_fields


@register("operator", "phase_retrieval")
class PhaseRetrieval(Operator):
    """Fourier phase retrieval (coherent diffraction imaging): ``y = |F pad(x)|²``.

    The object is zero-padded (centered) to ``oversample ×`` its size per axis — oversampling ≥ 2
    is what makes the phase recoverable in principle — and only the squared Fourier magnitude is
    measured (fftshift layout). Homogeneity 2, so :class:`~nefi.solve.EnergyScaleCorrection`
    applies. Global phase / translation / conjugate-flip ambiguities are inherent: compare
    reconstructions up to these (or break them with a support / non-negativity prior).
    Coarse fields are upsampled to the native grid.

    Args:
        oversample: padding factor per axis (int or per-axis).
        norm: FFT normalization.
        native_shape: object grid (inferred from the first call if omitted).
        field: object field name.

    Example::

        op = PhaseRetrieval(oversample=2)
        y = op({"x": torch.rand(16, 16)})          # (32, 32) diffraction pattern
    """

    homogeneity = 2.0

    def __init__(
        self,
        oversample: int | Sequence[int] = 2,
        norm: str = "ortho",
        native_shape: Sequence[int] | None = None,
        field: str = "x",
    ) -> None:
        super().__init__()
        self.oversample = oversample
        self.norm = norm
        self.native_shape = None if native_shape is None else shape_tuple(native_shape)
        self.primary = field

    def _factors(self, ndim: int) -> tuple[int, ...]:
        o = self.oversample
        fs = (int(o),) * ndim if isinstance(o, int) else tuple(int(v) for v in o)
        if len(fs) != ndim or any(v < 1 for v in fs):
            raise OperatorError(f"invalid oversample {o} for a {ndim}-D object")
        return fs

    def forward(self, fields: Fields) -> torch.Tensor:
        if self.native_shape is not None:
            fields = upsample_fields(fields, self.native_shape, names=(self.primary,))
        x = self.get_field(fields)
        n = tuple(x.shape)
        big = tuple(s * f for s, f in zip(n, self._factors(x.ndim)))
        pads: list[int] = []
        for s, b in zip(reversed(n), reversed(big)):
            lo = (b - s) // 2
            pads += [lo, b - s - lo]
        xp = torch.nn.functional.pad(x, pads)
        dims = tuple(range(x.ndim))
        k = torch.fft.fftshift(torch.fft.fftn(xp, dim=dims, norm=self.norm), dim=dims)
        return k.real**2 + k.imag**2

    def output_shape(self, shape):
        shape = shape_tuple(self.native_shape or shape)
        return tuple(s * f for s, f in zip(shape, self._factors(len(shape))))


@register("operator", "beer_lambert")
class BeerLambert(Operator):
    """Beer–Lambert attenuation ``I = I0 · exp(−∫ μ dl)`` (X-ray / optical transmission).

    The line integral is either along a grid axis (``axis``; physical spacing from ``domain``, so
    it is resolution-consistent) or computed by a user-provided linear path operator ``path_op``
    (e.g. a Radon transform, an :class:`Operator` or a callable ``μ -> ∫μ``).

    Args:
        axis: integration axis (ignored when ``path_op`` is given).
        path_op: optional linear operator returning line integrals of ``μ``.
        I0: incident intensity (scalar or tensor broadcastable to the output).
        domain: physical domain of ``μ`` (default unit cube: spacing ``1/n``).
        mode: ``"total"`` — transmitted intensity through the whole object (axis removed);
            ``"cumulative"`` — depth-resolved ``I(z) = I0 exp(−∫_0^z μ)`` (midpoint rule).
        field: attenuation-coefficient field name.

    Example::

        op = BeerLambert(axis=0, domain=nefi.Domain.unit((32, 32)))
        y = op({"x": torch.rand(32, 32)})        # (32,) transmitted intensity
    """

    def __init__(
        self,
        axis: int = 0,
        path_op: Operator | Callable[[torch.Tensor], torch.Tensor] | None = None,
        I0: float | torch.Tensor = 1.0,
        domain: Domain | None = None,
        mode: str = "total",
        field: str = "x",
    ) -> None:
        super().__init__()
        if mode not in ("total", "cumulative"):
            raise OperatorError(f"BeerLambert mode must be 'total' or 'cumulative', got {mode!r}")
        self.axis = int(axis)
        self.path_op = path_op  # an Operator is registered as a submodule (follows .to())
        self.I0 = I0
        self.domain = domain
        self.mode = mode
        self.primary = field

    def _spacing(self, shape: tuple[int, ...]) -> float:
        ax = self.axis % len(shape)
        if self.domain is not None:
            return float(self.domain.spacing(shape)[ax])
        return 1.0 / shape[ax]

    def line_integral(self, mu: torch.Tensor) -> torch.Tensor:
        if self.path_op is not None:
            if isinstance(self.path_op, Operator):
                return self.path_op({self.path_op.primary: mu})
            return self.path_op(mu)
        ax = self.axis % mu.ndim
        h = self._spacing(tuple(mu.shape))
        if self.mode == "total":
            return mu.sum(dim=ax) * h
        return torch.cumsum(mu, dim=ax) * h - 0.5 * mu * h

    def forward(self, fields: Fields) -> torch.Tensor:
        mu = self.get_field(fields)
        L = self.line_integral(mu)
        I0 = torch.as_tensor(self.I0, device=L.device, dtype=L.dtype)
        return I0 * torch.exp(-L)

    def at_resolution(self, shape):
        if isinstance(self.path_op, Operator):
            new = self.path_op.at_resolution(shape)
            if new is not self.path_op:
                return BeerLambert(self.axis, new, self.I0, self.domain, self.mode, self.primary)
        return self

    def output_shape(self, shape):
        shape = shape_tuple(shape)
        if self.path_op is not None:
            if isinstance(self.path_op, Operator):
                return self.path_op.output_shape(shape)
            return None
        if self.mode == "cumulative":
            return shape
        ax = self.axis % len(shape)
        return tuple(s for i, s in enumerate(shape) if i != ax)


@register("operator", "saturation")
class Saturation(Operator):
    """Saturating detector response (pointwise), to compose after a linear operator.

    * ``kind="tanh"``: ``y = level · tanh(gain · x / level)`` — linear with slope ``gain`` near
      zero, saturating at ``±level``.
    * ``kind="sigmoid"``: ``y = level · σ(gain · (x − threshold))`` — S-shaped response with
      threshold (film, thresholded sensors).

    Example::

        dom = nefi.Domain.unit((32, 32))
        cam = Sequential(FFTConvolution(gaussian_kernel_fn(0.02), dom), Saturation("tanh", 1.0))
        y = cam({"x": torch.rand(32, 32)})
    """

    batchable = True  # pointwise

    def __init__(
        self,
        kind: str = "tanh",
        level: float = 1.0,
        gain: float = 1.0,
        threshold: float = 0.0,
        field: str = "x",
    ) -> None:
        super().__init__()
        if kind not in ("tanh", "sigmoid"):
            raise OperatorError(f"Saturation kind must be 'tanh' or 'sigmoid', got {kind!r}")
        if level <= 0:
            raise OperatorError("Saturation level must be positive")
        self.kind, self.level, self.gain, self.threshold = (
            kind,
            float(level),
            float(gain),
            float(threshold),
        )
        self.primary = field

    def forward(self, fields: Fields) -> torch.Tensor:
        x = self.get_field(fields)
        if self.kind == "tanh":
            return self.level * torch.tanh(self.gain * x / self.level)
        return self.level * torch.sigmoid(self.gain * (x - self.threshold))

    def output_shape(self, shape):
        return shape_tuple(shape)


__all__ = ["BeerLambert", "PhaseRetrieval", "Saturation"]
