"""Linear measurement operators: resolution loss, sampling, Fourier (MRI-style) sampling, algebra.

All operators follow the nefi multiscale contract: ``at_resolution(shape)`` returns an operator for
fields sampled at ``shape`` and ``output_shape(shape)`` gives the prediction shape. Operators whose
measurement lives on the native grid (``Sampling``, ``FourierSampling``) upsample coarse fields to
the native grid (see :class:`~nefi.operators.function.UpsampleToNative`), so the measurement is
never resampled.

Example::

    import nefi
    from nefi.operators.linear import FourierSampling, random_kspace_mask

    image = torch.rand(64, 64)
    mask = random_kspace_mask((64, 64), fraction=0.3, seed=0)
    op = FourierSampling(mask)                         # x (64, 64) -> k-space (64, 64, 2)
    meas = nefi.Measurement(op({"x": image}), mask=op.measurement_mask())
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

from ..errors import OperatorError, ShapeError
from ..registry import register
from ..utils.tensor import resample, shape_tuple
from .base import Fields, Operator
from .function import upsample_fields


def _fft_dims(ndim: int) -> tuple[int, ...]:
    return tuple(range(-ndim, 0))


@register("operator", "identity")
class Identity(Operator):
    """``y = x`` (denoising / inpainting with ``Measurement.mask``). Homogeneity 1.

    Example::

        op = Identity()
        assert torch.equal(op({"x": torch.ones(3)}), torch.ones(3))
    """

    homogeneity = 1.0
    batchable = True

    def __init__(self, field: str = "x") -> None:
        super().__init__()
        self.primary = field

    def forward(self, fields: Fields) -> torch.Tensor:
        return self.get_field(fields)

    def output_shape(self, shape):
        return shape_tuple(shape)


@register("operator", "pointwise")
class Pointwise(Operator):
    """Elementwise map ``y = fn(x)`` (detector response, nonlinearity, unit conversion).

    Resolution-agnostic; typically composed with a linear operator through
    :class:`~nefi.operators.Sequential`, e.g. ``Sequential(blur, Pointwise(torch.tanh))``.

    Args:
        fn: elementwise differentiable function.
        homogeneity: degree of homogeneity of ``fn`` if any (``x**2`` → 2).
        field: input field name.

    Example::

        op = Pointwise(torch.square, homogeneity=2.0)
    """

    batchable = True  # ``fn`` is elementwise by contract

    def __init__(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        homogeneity: float | None = None,
        field: str = "x",
    ) -> None:
        super().__init__()
        self.fn = fn
        self.homogeneity = homogeneity
        self.primary = field

    def forward(self, fields: Fields) -> torch.Tensor:
        return self.fn(self.get_field(fields))

    def output_shape(self, shape):
        return shape_tuple(shape)


@register("operator", "downsample")
class Downsample(Operator):
    """Resolution loss by an integer factor per axis (detector binning / low-res imaging).

    ``mode="area"`` averages ``factor``-blocks (pixel binning), ``"decimate"`` keeps every
    ``factor``-th sample (starting at ``factor // 2``), ``"linear"`` interpolates.
    Resolution-agnostic: at a coarse stage the output is ``shape // factor`` and the measurement
    is area-resampled accordingly. Homogeneity 1.

    Example::

        op = Downsample(4)
        assert op({"x": torch.rand(64, 64)}).shape == (16, 16)
    """

    homogeneity = 1.0

    def __init__(self, factor: int | Sequence[int] = 2, mode: str = "area", field: str = "x"):
        super().__init__()
        if mode not in ("area", "decimate", "linear"):
            raise OperatorError(f"Downsample mode must be area|decimate|linear, got {mode!r}")
        self.factor = factor
        self.mode = mode
        self.primary = field

    def _factors(self, ndim: int) -> tuple[int, ...]:
        f = self.factor
        fs = (int(f),) * ndim if isinstance(f, int) else tuple(int(v) for v in f)
        if len(fs) != ndim or any(v < 1 for v in fs):
            raise ShapeError(f"Downsample factor {self.factor} invalid for a {ndim}-D field")
        return fs

    def output_shape(self, shape):
        shape = shape_tuple(shape)
        fs = self._factors(len(shape))
        return tuple(max(1, s // f) for s, f in zip(shape, fs))

    def forward(self, fields: Fields) -> torch.Tensor:
        x = self.get_field(fields)
        shape = tuple(x.shape)
        out = self.output_shape(shape)
        if self.mode == "decimate":
            fs = self._factors(len(shape))
            sl = tuple(slice(f // 2, f // 2 + o * f, f) for f, o in zip(fs, out))
            return x[sl]
        return resample(x, out, mode=self.mode)


@register("operator", "sampling")
class Sampling(Operator):
    """Point sampling of the field: inpainting, sparse sensors, scattered probes.

    Two equivalent modes:

    * ``Sampling(mask=m)`` returns the **full** field times ``m`` (zeros where unobserved). Pair
      it with ``Measurement(data, mask=m)`` (see :meth:`measurement`) so losses ignore the holes —
      the preferred, shape-preserving formulation.
    * ``Sampling(indices=idx)`` returns the compact vector of observed values: ``idx`` is a
      ``(K,)`` tensor of flat indices or a ``(K, d)`` tensor of grid indices on the native grid.

    Coarse fields are upsampled to the native grid first, so the measurement is never resampled.
    Homogeneity 1.

    Example::

        image = torch.rand(32, 32)
        m = (torch.rand(32, 32) < 0.2).float()
        op = Sampling(mask=m)
        meas = op.measurement(op({"x": image}))       # Measurement with mask=m
    """

    homogeneity = 1.0

    def __init__(
        self,
        mask: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        field: str = "x",
        native_shape: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if (mask is None) == (indices is None):
            raise OperatorError("Sampling needs exactly one of mask=... or indices=...")
        self.primary = field
        if mask is not None:
            m = torch.as_tensor(mask).float()
            self.register_buffer("mask", m, persistent=False)
            self.native_shape = tuple(m.shape)
            self.indices = None
        else:
            idx = torch.as_tensor(indices).long()
            if native_shape is None:
                raise OperatorError("Sampling(indices=...) needs native_shape=(...) of the field")
            self.native_shape = shape_tuple(native_shape)
            self.register_buffer("indices", idx, persistent=False)
            self.mask = None

    def forward(self, fields: Fields) -> torch.Tensor:
        fields = upsample_fields(fields, self.native_shape, names=(self.primary,))
        x = self.get_field(fields)
        if self.mask is not None:
            return x * self.mask.to(x)
        idx = self.indices
        if idx.ndim == 1:
            return x.reshape(*x.shape[: x.ndim - len(self.native_shape)], -1)[..., idx]
        return x[tuple(idx.T)]

    def output_shape(self, shape):
        if self.mask is not None:
            return self.native_shape
        return (int(self.indices.shape[0]),)

    def measurement_mask(self) -> torch.Tensor | None:
        """Mask to put in :class:`~nefi.measurement.Measurement` (mask mode), else ``None``."""
        return None if self.mask is None else self.mask.clone()

    def measurement(self, data: torch.Tensor, noise_std: float | None = None):
        """``Measurement(data, mask=self.measurement_mask(), noise_std=noise_std)``."""
        from ..measurement import Measurement

        return Measurement(data, mask=self.measurement_mask(), noise_std=noise_std)


def random_kspace_mask(
    shape: Sequence[int],
    fraction: float = 0.3,
    center_fraction: float = 0.08,
    seed: int = 0,
    variable_density: bool = True,
) -> torch.Tensor:
    """Random (variable-density) k-space sampling mask in centered layout (DC at ``shape//2``).

    Keeps a fully sampled square/cube of side ``center_fraction · n`` around DC and draws the rest
    with probability decaying with the distance to DC (``variable_density``) or uniformly, so that
    the overall sampling fraction is ≈ ``fraction``.

    Example::

        m = random_kspace_mask((64, 64), fraction=0.3)
        assert abs(float(m.mean()) - 0.3) < 0.03
    """
    shape = shape_tuple(shape)
    g = torch.Generator().manual_seed(int(seed))
    axes = [torch.arange(n, dtype=torch.float32) - n // 2 for n in shape]
    grids = torch.meshgrid(*axes, indexing="ij")
    r = torch.sqrt(sum((gi / (n / 2)) ** 2 for gi, n in zip(grids, shape)))
    center = torch.ones(shape, dtype=torch.bool)
    for gi, n in zip(grids, shape):
        center &= gi.abs() <= max(1.0, center_fraction * n / 2)
    n_total = int(round(fraction * r.numel()))
    n_left = max(0, n_total - int(center.sum()))
    w = (1.0 - r / r.max()).clamp_min(0) ** 2 + 1e-3 if variable_density else torch.ones(shape)
    w = w.masked_fill(center, 0.0).reshape(-1)
    mask = center.reshape(-1).clone()
    if n_left > 0:
        pick = torch.multinomial(w, min(n_left, int((w > 0).sum())), replacement=False, generator=g)
        mask[pick] = True
    return mask.reshape(shape).float()


@register("operator", "fourier_sampling")
class FourierSampling(Operator):
    """Undersampled Fourier measurements (MRI / interferometry): ``y = M ⊙ F x``.

    Args:
        mask: k-space sampling mask ``(*shape)`` (1 = sampled), in centered layout (DC at
            ``shape//2``) when ``centered=True``.
        norm: FFT normalization (``"ortho"`` keeps noise levels comparable between domains).
        real_output: return ``torch.view_as_real`` (trailing dim 2: real, imag) instead of a
            complex tensor (losses and noise estimation expect real tensors).
        centered: use the centered DFT ``fftshift(fft(ifftshift(x)))`` (standard in MRI).
        compact: return only the sampled coefficients ``(K,)`` (``(K, 2)`` if real) instead of
            the zero-filled full grid.
        field: the (real) image field; ``imag_field`` optionally names a second field used as
            the imaginary part of a complex image.

    Coarse fields are upsampled to the mask grid. Homogeneity 1.

    Example::

        mask = random_kspace_mask((32, 32), 0.3)
        op = FourierSampling(mask)
        y = op({"x": torch.rand(32, 32)})                 # (32, 32, 2)
        zf = op.adjoint(y)                                 # zero-filled reconstruction
    """

    homogeneity = 1.0

    def __init__(
        self,
        mask: torch.Tensor,
        norm: str = "ortho",
        real_output: bool = True,
        centered: bool = True,
        compact: bool = False,
        field: str = "x",
        imag_field: str | None = None,
    ) -> None:
        super().__init__()
        m = torch.as_tensor(mask).float()
        self.register_buffer("mask", m, persistent=False)
        self.native_shape = tuple(m.shape)
        self.ndim = m.ndim
        self.norm = norm
        self.real_output = real_output
        self.centered = centered
        self.compact = compact
        self.primary = field
        self.imag_field = imag_field

    def fft(self, x: torch.Tensor) -> torch.Tensor:
        dims = _fft_dims(self.ndim)
        if self.centered:
            return torch.fft.fftshift(
                torch.fft.fftn(torch.fft.ifftshift(x, dim=dims), dim=dims, norm=self.norm), dim=dims
            )
        return torch.fft.fftn(x, dim=dims, norm=self.norm)

    def ifft(self, k: torch.Tensor) -> torch.Tensor:
        dims = _fft_dims(self.ndim)
        if self.centered:
            return torch.fft.fftshift(
                torch.fft.ifftn(torch.fft.ifftshift(k, dim=dims), dim=dims, norm=self.norm),
                dim=dims,
            )
        return torch.fft.ifftn(k, dim=dims, norm=self.norm)

    def forward(self, fields: Fields) -> torch.Tensor:
        names = (self.primary,) if self.imag_field is None else (self.primary, self.imag_field)
        fields = upsample_fields(fields, self.native_shape, names=names)
        x = self.get_field(fields)
        if self.imag_field is not None:
            x = torch.complex(x, self.get_field(fields, self.imag_field))
        k = self.fft(x)
        m = self.mask.to(k.real.dtype)
        y = k[..., m > 0] if self.compact else k * m
        return torch.view_as_real(y) if self.real_output else y

    def output_shape(self, shape):
        base = (int((self.mask > 0).sum()),) if self.compact else self.native_shape
        return (*base, 2) if self.real_output else base

    def required_fields(self):
        return (self.primary,) if self.imag_field is None else (self.primary, self.imag_field)

    def measurement_mask(self) -> torch.Tensor | None:
        """Mask for :class:`~nefi.measurement.Measurement` (zero-filled layout), else ``None``."""
        if self.compact:
            return None
        m = self.mask
        return m.unsqueeze(-1).expand(*m.shape, 2).clone() if self.real_output else m.clone()

    def noise_region(self, outer: float = 0.5) -> torch.Tensor:
        """Boolean mask of sampled coefficients beyond ``outer ×`` the maximal k-space radius.

        Natural images carry little energy there, so observed values are noise dominated.
        """
        axes = [torch.arange(n, dtype=torch.float32) - n // 2 for n in self.native_shape]
        if not self.centered:
            axes = [torch.fft.ifftshift(a) for a in axes]
        grids = torch.meshgrid(*axes, indexing="ij")
        r = torch.sqrt(sum((g / (n / 2)) ** 2 for g, n in zip(grids, self.native_shape)))
        region = (r > outer * float(r.max())) & (self.mask.cpu() > 0)
        if self.compact:
            region = region[self.mask.cpu() > 0]
        return region.unsqueeze(-1).expand(*region.shape, 2) if self.real_output else region

    def estimate_noise(self, measurement, outer: float = 0.5) -> float | None:
        """Per-component noise std from the MAD of observed outer-k-space coefficients.

        Returns ``None`` when too few coefficients are observed there (the generic estimator is
        then used by :func:`nefi.auto.from_forward`).

        Example::

            sigma = op.estimate_noise(nefi.Measurement(y, mask=op.measurement_mask()))
        """
        from ..auto import estimate_noise

        region = self.noise_region(outer)
        if tuple(region.shape) != tuple(measurement.data.shape) or int(region.sum()) < 32:
            return None
        return estimate_noise(measurement, method="values", region=region, store=False)

    def adjoint(self, y: torch.Tensor, real: bool = True) -> torch.Tensor:
        """Zero-filled inverse ``F⁻¹ Mᵀ y`` (the classic baseline / a good initial guess)."""
        if self.real_output:
            y = torch.view_as_complex(y.contiguous())
        if self.compact:
            k = torch.zeros(self.native_shape, dtype=y.dtype, device=y.device)
            k[self.mask > 0] = y
        else:
            k = y * self.mask.to(y.real.dtype)
        x = self.ifft(k)
        return x.real if real else x


class _Combined(Operator):
    def __init__(self, *ops: Operator) -> None:
        super().__init__()
        if not ops:
            raise OperatorError(f"{type(self).__name__} needs at least one operator")
        self.ops = nn.ModuleList(ops)
        self.primary = ops[0].primary
        hs = {op.homogeneity for op in ops}
        self.homogeneity = hs.pop() if len(hs) == 1 else None

    def required_fields(self):
        names: list[str] = []
        for op in self.ops:
            for n in op.required_fields():
                if n not in names:
                    names.append(n)
        return tuple(names)

    def _rebuild(self, ops):  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def traceable(self) -> bool:  # type: ignore[override]
        return all(getattr(op, "traceable", True) for op in self.ops)

    def at_resolution(self, shape):
        new = [op.at_resolution(shape) for op in self.ops]
        if all(a is b for a, b in zip(new, self.ops)):
            return self
        return self._rebuild(new)


@register("operator", "sum")
class Sum(_Combined):
    """``y = Σ_i op_i(fields)`` (superposition of contributions, e.g. two field components).

    Example::

        op = Sum(Identity("a"), Identity("b"))
        y = op({"a": torch.ones(4), "b": torch.ones(4)})      # 2·ones
    """

    @property
    def batchable(self) -> bool:  # type: ignore[override]
        return all(getattr(op, "batchable", False) for op in self.ops)

    def forward(self, fields: Fields) -> torch.Tensor:
        ys = [op(fields) for op in self.ops]
        shapes = {tuple(y.shape) for y in ys}
        if len(shapes) > 1:
            raise ShapeError(f"Sum operands produced different shapes {sorted(shapes)}")
        out = ys[0]
        for y in ys[1:]:
            out = out + y
        return out

    def output_shape(self, shape):
        for op in self.ops:
            s = op.output_shape(shape)
            if s is not None:
                return s
        return None

    def _rebuild(self, ops):
        return Sum(*ops)


@register("operator", "stack")
class Stack(_Combined):
    """Several measurements of the same unknown(s) (multi-modal / multi-view data).

    ``mode="stack"``: ``torch.stack`` along a new leading axis (operands must agree in shape);
    ``mode="concat"``: flatten each prediction and concatenate (any shapes).

    Example::

        blur = FFTConvolution(gaussian_kernel_fn(0.05), nefi.Domain.unit((32, 32)))
        op = Stack(blur, Downsample(2), mode="concat")
        y = op({"x": torch.rand(32, 32)})             # (32·32 + 16·16,)
    """

    def __init__(self, *ops: Operator, mode: str = "stack") -> None:
        super().__init__(*ops)
        if mode not in ("stack", "concat"):
            raise OperatorError(f"Stack mode must be 'stack' or 'concat', got {mode!r}")
        self.mode = mode

    def forward(self, fields: Fields) -> torch.Tensor:
        ys = [op(fields) for op in self.ops]
        if self.mode == "concat":
            return torch.cat([y.reshape(-1) for y in ys])
        shapes = {tuple(y.shape) for y in ys}
        if len(shapes) > 1:
            raise ShapeError(
                f"Stack(mode='stack') operands produced different shapes {sorted(shapes)}; "
                "use mode='concat'"
            )
        return torch.stack(ys)

    def output_shape(self, shape):
        outs = [op.output_shape(shape) for op in self.ops]
        if any(o is None for o in outs):
            return None
        if self.mode == "concat":
            n = 0
            for o in outs:
                k = 1
                for v in o:
                    k *= v
                n += k
            return (n,)
        return (len(outs), *outs[0])

    def _rebuild(self, ops):
        return Stack(*ops, mode=self.mode)


__all__ = [
    "Downsample",
    "FourierSampling",
    "Identity",
    "Pointwise",
    "Sampling",
    "Stack",
    "Sum",
    "random_kspace_mask",
]
