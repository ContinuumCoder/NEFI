"""Extra heads used by the prior catalogue (:mod:`nefi.priors`).

* :class:`Affine` / :class:`ExpHead` — identity / exponential heads that accept an ``init_value``
  (so :func:`nefi.auto.from_forward` can start the field at a data-matched constant).
* :class:`MaskedHead` — multiply any head by a fixed support mask (stop-gradient by construction);
  implements the ``KnownSupport`` prior.
* :class:`MassNormalized` — rescale any non-negative head so that its integral (or mean) equals a
  prescribed total; implements the hard ``Conserved`` prior.
* :class:`ZeroMean` — subtract the spatial mean of any head (gauge fixing for unknowns defined up
  to an invisible additive constant, e.g. a phase retrieved from intensities).

Example::

    from nefi.fields import Heads, NeuralField, Softplus
    from nefi.fields.modifiers import MaskedHead, MassNormalized

    mask = torch.zeros(32, 32); mask[8:24, 8:24] = 1
    heads = Heads({"rho": MassNormalized(MaskedHead(Softplus(), mask), total=1.0)})
    field = NeuralField(2, heads, hidden=32, depth=2)
"""

from __future__ import annotations

import math

import torch

from ..errors import ConfigError
from ..registry import build, register
from ..utils.tensor import resample, shape_tuple
from .heads import Exp, Head


@register("head", "affine")
class Affine(Head):
    """Unconstrained (signed) field ``offset + scale · h``.

    ``scale``/``offset`` put the raw network output in natural units (e.g. a sound speed around
    1500 m/s: ``Affine(scale=100, offset=1500)``), which keeps the optimization well conditioned.

    Example::

        head = Affine(scale=2.0, offset=1.0, init_value=3.0)
        assert head.init_bias() == [1.0]
    """

    def __init__(
        self, scale: float = 1.0, offset: float = 0.0, init_value: float | None = None
    ) -> None:
        super().__init__()
        if scale == 0:
            raise ConfigError("Affine head needs a non-zero scale")
        self.scale, self.offset = float(scale), float(offset)
        self.init_value = init_value

    def transform(self, raw, others):
        return self.offset + self.scale * raw[..., 0]

    def init_bias(self):
        if self.init_value is None:
            return None
        return [(float(self.init_value) - self.offset) / self.scale]

    def inverse(self, value):
        return ((torch.as_tensor(value) - self.offset) / self.scale).unsqueeze(-1)


@register("head", "exp_init")
class ExpHead(Exp):
    """Strictly positive field ``exp(h)`` (log-parameterization) with an optional ``init_value``.

    Example::

        head = ExpHead(init_value=0.5)
        assert abs(head.init_bias()[0] - math.log(0.5)) < 1e-12
    """

    def __init__(self, init_value: float | None = None) -> None:
        super().__init__()
        self.init_value = init_value

    def init_bias(self):
        if self.init_value is None:
            return None
        return [math.log(max(float(self.init_value), 1e-30))]


class _WrappedHead(Head):
    """Base for heads that post-process an inner head's output."""

    uses_progress = True

    def __init__(self, inner: Head | dict | str) -> None:
        super().__init__()
        self.inner: Head = build("head", inner)
        self.n_in = self.inner.n_in
        self.depends_on = tuple(self.inner.depends_on)

    def init_bias(self):
        return self.inner.init_bias()

    def inverse(self, value):
        return self.inner.inverse(value)

    @property
    def init_value(self):
        return getattr(self.inner, "init_value", None)

    @init_value.setter
    def init_value(self, value) -> None:
        if hasattr(self.inner, "init_value"):
            self.inner.init_value = value


@register("head", "masked")
class MaskedHead(_WrappedHead):
    """``x = inner(h) · m + fill · (1 − m)`` with a fixed support mask ``m`` (known support).

    The mask is a constant, so no gradient flows through it (stop-gradient by construction) and
    the network is never asked to explain the exterior. The mask is given at the native resolution
    and resampled (area / linear) to whatever grid the field is evaluated on; with
    ``binary=True`` the resampled mask is re-thresholded at 0.5.

    Args:
        inner: the head defining the value inside the support (e.g. ``Softplus()``).
        mask: ``(*shape)`` tensor, 1 inside the support, 0 outside.
        fill: value outside the support.
        binary: re-binarize the mask after resampling.

    Example::

        mask = torch.zeros(16); mask[4:12] = 1
        head = MaskedHead(Softplus(), mask)
        x = head(torch.randn(16, 1))
        assert torch.all(x[:4] == 0)
    """

    def __init__(
        self,
        inner: Head | dict | str,
        mask: torch.Tensor,
        fill: float = 0.0,
        binary: bool = True,
    ) -> None:
        super().__init__(inner)
        m = torch.as_tensor(mask).detach().float()
        if m.numel() == 0:
            raise ConfigError("MaskedHead needs a non-empty mask")
        self.register_buffer("mask", m.clone(), persistent=False)
        self.fill = float(fill)
        self.binary = binary
        self._cache: dict[tuple, torch.Tensor] = {}

    def mask_at(self, shape, device=None, dtype=None) -> torch.Tensor:
        """The support mask resampled to ``shape`` (cached)."""
        shape = shape_tuple(shape)
        key = (shape, str(device), str(dtype))
        if key not in self._cache:
            m = self.mask
            if tuple(m.shape) != shape:
                if m.ndim != len(shape):
                    raise ConfigError(
                        f"support mask has {m.ndim} dims but the field is evaluated on a "
                        f"{len(shape)}-D grid {shape}; pass a mask with the field's shape"
                    )
                m = resample(m, shape)
                if self.binary:
                    m = (m > 0.5).float()
            self._cache[key] = m.to(device=device, dtype=dtype)
        return self._cache[key]

    def _apply(self, fn, recurse=True):  # keep the cache coherent with .to()/.cuda()
        self._cache = {}
        return super()._apply(fn, recurse)

    def transform(self, raw, others, progress=1.0):
        v = self.inner(raw, others, progress)
        m = self.mask_at(tuple(v.shape), v.device, v.dtype)
        return v * m + self.fill * (1.0 - m)


@register("head", "mass_normalized")
class MassNormalized(_WrappedHead):
    """Rescale a non-negative inner head so that its total is exactly ``total`` (conservation).

    ``kind="sum"``: the integral ``mean(x) · volume`` equals ``total`` (resolution independent;
    ``volume`` is the physical measure of the domain). ``kind="mean"``: ``mean(x) = total``.
    The normalization is global (it couples all grid points), so use it with fields evaluated on
    the full grid, which is what the solver does.

    Example::

        head = MassNormalized(Softplus(), total=2.0, volume=1.0)
        x = head(torch.randn(8, 8, 1))
        assert abs(float(x.mean()) - 2.0) < 1e-5
    """

    def __init__(
        self,
        inner: Head | dict | str,
        total: float = 1.0,
        volume: float = 1.0,
        kind: str = "sum",
        eps: float = 1e-12,
    ) -> None:
        super().__init__(inner)
        if kind not in ("sum", "mean"):
            raise ConfigError(f"MassNormalized kind must be 'sum' or 'mean', got {kind!r}")
        if volume <= 0:
            raise ConfigError("MassNormalized needs a positive domain volume")
        self.total, self.volume, self.kind, self.eps = float(total), float(volume), kind, eps

    def transform(self, raw, others, progress=1.0):
        v = self.inner(raw, others, progress)
        target_mean = self.total / self.volume if self.kind == "sum" else self.total
        return v * (target_mean / v.mean().clamp_min(self.eps))


@register("head", "zero_mean")
class ZeroMean(_WrappedHead):
    """``x = inner(h) − mean(inner(h))``: a field whose spatial mean is exactly zero (a gauge).

    For unknowns defined only up to an additive constant the data cannot see — a phase retrieved
    from intensities (inline holography, ptychography), a potential or stream function with a free
    gauge — the constant is otherwise set by the initialization and the optimizer's drift, so a
    reconstruction can sit at an arbitrary offset from the ground truth. Removing the mean *by
    construction* fixes the gauge: the field is directly comparable to a zero-mean ground truth
    and no optimization effort goes into the invisible direction. Like :class:`MassNormalized`
    the mean is global (it couples all grid points), so evaluate the field on the full grid, which
    is what the solver does.

    Args:
        inner: the head defining the field before the gauge, e.g. ``Bounded(-π, π)``.
        n_dims: number of trailing (spatial) dims averaged over; ``None`` = all dims.

    Example::

        head = ZeroMean(Bounded(-1.0, 1.0))
        x = head(torch.randn(8, 8, 1))
        assert abs(float(x.mean())) < 1e-6
    """

    def __init__(self, inner: Head | dict | str, n_dims: int | None = None) -> None:
        super().__init__(inner)
        if n_dims is not None and n_dims < 1:
            raise ConfigError(f"ZeroMean n_dims must be >= 1 or None, got {n_dims}")
        self.n_dims = n_dims

    def transform(self, raw, others, progress=1.0):
        v = self.inner(raw, others, progress)
        k = v.ndim if self.n_dims is None else min(int(self.n_dims), v.ndim)
        return v - v.mean(dim=tuple(range(v.ndim - k, v.ndim)), keepdim=True)


@register("head", "scaled")
class ScaledHead(_WrappedHead):
    """``x = scale · inner(h)``: puts the network output in *natural units*.

    With ``scale`` set to the typical magnitude of the unknown (``from_forward`` fits it to the
    data), the raw network output stays ``O(1)`` whatever the physical units (a sound speed of
    1500 m/s, an attenuation of 1e-3 /mm, 10⁴ photon counts), so learning rates, gradient clipping
    and head nonlinearities (softplus curvature, gates) behave identically across problems.

    Example::

        head = ScaledHead(Softplus(init_value=1.0), scale=1e-3)   # starts at 1e-3
        x = head(torch.zeros(4, 1) + head.init_bias()[0])
    """

    def __init__(self, inner: Head | dict | str, scale: float = 1.0) -> None:
        super().__init__(inner)
        self.scale = float(scale)

    def transform(self, raw, others, progress=1.0):
        return self.scale * self.inner(raw, others, progress)

    def inverse(self, value):
        return self.inner.inverse(torch.as_tensor(value) / self.scale)


def find_head(head: Head, cls: type) -> Head | None:
    """First head of type ``cls`` along the ``.inner`` chain of a (wrapped) head.

    Example::

        from nefi.fields import Softplus
        h = MaskedHead(ScaledHead(Softplus(), 2.0), torch.ones(4))
        assert find_head(h, ScaledHead).scale == 2.0
    """
    h: Head | None = head
    while h is not None:
        if isinstance(h, cls):
            return h
        h = getattr(h, "inner", None) if isinstance(getattr(h, "inner", None), Head) else None
    return None


def innermost(head: Head) -> Head:
    """Follow ``.inner`` links of wrapped heads down to the defining head.

    Example::

        from nefi.fields import Softplus
        assert type(innermost(ScaledHead(Softplus(), 2.0))) is Softplus
    """
    while hasattr(head, "inner") and isinstance(head.inner, Head):
        head = head.inner
    return head


def head_init_value(head: Head) -> float | None:
    """The ``init_value`` of a (possibly wrapped) head, ``None`` if unsupported/unset.

    Example::

        from nefi.fields import Softplus
        assert head_init_value(ScaledHead(Softplus(init_value=1.0))) == 1.0
    """
    return getattr(innermost(head), "init_value", None)


def set_head_init_value(head: Head, value: float | None) -> bool:
    """Set ``init_value`` on the innermost head that supports it; returns success.

    Example::

        from nefi.fields import Softplus
        h = ScaledHead(Softplus())
        assert set_head_init_value(h, 0.5) and head_init_value(h) == 0.5
    """
    h = innermost(head)
    if not hasattr(h, "init_value"):
        return False
    h.init_value = value
    return True


def value_range(head: Head) -> tuple[float, float]:
    """Interval of values a (possibly wrapped) head can produce (best effort).

    Example::

        from nefi.fields import Bounded
        assert value_range(Bounded(0.0, 2.0)) == (0.0, 2.0)
    """
    h = innermost(head)
    lo = getattr(h, "lo", None)
    hi = getattr(h, "hi", None)
    if lo is not None and hi is not None:
        rng = (float(lo), float(hi))
    elif type(h).__name__ in ("Softplus", "GatedSoftplus", "Exp", "ExpHead"):
        rng = (0.0, math.inf)
    else:
        rng = (-math.inf, math.inf)
    sc = find_head(head, ScaledHead)
    if sc is not None and sc.scale != 1.0:
        a, b = rng[0] * sc.scale, rng[1] * sc.scale
        rng = (min(a, b), max(a, b))
    return rng


__all__ = [
    "Affine",
    "ExpHead",
    "MaskedHead",
    "MassNormalized",
    "ScaledHead",
    "ZeroMean",
    "find_head",
    "head_init_value",
    "innermost",
    "set_head_init_value",
    "value_range",
]
