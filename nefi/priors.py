"""The prior catalogue: what you physically know about the unknown → heads, losses, hints.

A :class:`Prior` turns one piece of physical knowledge into the matching nefi machinery:

* a **head** (hard constraint on values: non-negativity, bounds, two phases, …),
* **losses** (soft knowledge: sparsity, piecewise constancy, smoothness, …) with a *relative
  strength* used by automatic weight balancing,
* **post-processing** (energy-anchored scale correction), **field wrappers** (exact symmetries) and
  **curriculum / field hints**.

Priors compose with ``+`` in a tiny DSL::

    import nefi
    priors = nefi.Prior.parse("nonnegative + sparse(l1=1e-2) + piecewise_constant")
    spec = nefi.priors.combine(priors)          # resolves conflicts, builds the head
    print(spec.describe())

or as objects: ``NonNegative() + Sparse(l1=1e-2) + PiecewiseConstant(tv=1e-2)``.
:func:`combine` enforces that exactly one prior defines the output head (the most specific one wins
when it implies the others, e.g. ``sparse`` ⊂ ``nonnegative``; contradictions raise a
:class:`~nefi.errors.ConfigError` explaining what to change).

DSL names (aliases in parentheses): ``nonnegative`` (``nonneg``, ``non_negative``),
``positive`` (``log``), ``bounded`` (``box``, ``range``), ``sparse``, ``piecewise_constant``
(``tv``, ``piecewise``), ``smooth``, ``binary`` (``two_phase``, ``level_set``),
``known_support`` (``support``), ``symmetric`` (``symmetry``), ``conserved`` (``mass``),
``periodic``, ``scale``, ``monotone``, ``unconstrained`` (``real``, ``signed``). Arguments are
Python literals; bare identifiers are strings (``symmetric(radial)``); ``inf`` is allowed.
"""

from __future__ import annotations

import ast
import difflib
import inspect
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from dataclasses import field as dc_field
from typing import Any, ClassVar

import torch

from .errors import ConfigError
from .fields.base import Field
from .fields.heads import Bounded as BoundedHead
from .fields.heads import GatedSoftplus, Head, Softplus
from .fields.levelset import LevelSetHead
from .fields.modifiers import Affine, ExpHead, MaskedHead, MassNormalized, ScaledHead
from .losses.base import Loss
from .losses.physics import Conservation, GradientL2, KnownSupportLoss, SymmetryLoss
from .losses.physics import Monotone as MonotoneLoss
from .losses.reg import L1, TV, Laplacian, Tikhonov
from .solve.postprocess import EnergyScaleCorrection, Postprocess

log = logging.getLogger("nefi")

INF = math.inf
LossSpec = dict[str, tuple[Loss, float]]


@dataclass
class PriorContext:
    """Information shared with priors while building losses / heads / fields.

    Example::

        ctx = PriorContext(field="rho", ndim=2, periodic_axes=(0,))
        PiecewiseConstant().losses("rho", ctx)["tv"][0].periodic_axes      # (0,)

    Attributes:
        field: name of the field the priors apply to.
        ndim: domain dimension.
        domain: the :class:`~nefi.domain.Domain` (when known).
        periodic_axes: axes declared periodic by a :class:`Periodic` prior.
        value_range: interval of values allowed by the resolved head.
    """

    field: str = "x"
    ndim: int | None = None
    domain: Any = None
    periodic_axes: tuple[int, ...] = ()
    value_range: tuple[float, float] = (-INF, INF)


class Prior:
    """Base class. Subclasses are dataclasses with a class-level ``key`` (DSL name).

    Override what applies: :meth:`head` (+ :meth:`rank`, :meth:`interval`) for head-defining
    priors, :meth:`wrap_head` for head modifiers, :meth:`losses`, :meth:`postprocess`,
    :meth:`wrap_field` (+ :meth:`inner_ndim`), :meth:`curriculum_hints`, :meth:`field_hints`.

    Example (a custom loss-only prior, usable in the DSL)::

        from dataclasses import dataclass

        @dataclass(repr=False)
        class Ridge(Prior):
            weight: float = 1e-3
            summary = "small values (ridge penalty)"

            def losses(self, field_name, context=None):
                return {"ridge": (nefi.losses.Tikhonov(field_name), self.weight)}

        register_prior("ridge", Ridge)
        spec = combine("nonnegative + ridge(weight=1e-2)")
    """

    key: ClassVar[str] = "prior"
    summary: ClassVar[str] = ""
    head_rank: ClassVar[int] = -1  # >= 0: defines the output head (higher = more specific)
    modifier_order: ClassVar[int] = -1  # >= 0: wraps the resolved head (applied in this order)
    wraps_field: ClassVar[bool] = False

    # --- head ------------------------------------------------------------------------------
    def rank(self) -> int:
        """Specificity of the head this prior defines (``-1``: no head)."""
        return self.head_rank

    def head(self) -> Head | None:
        """The output head, or ``None`` if this prior does not define one."""
        return None

    def interval(self) -> tuple[float, float]:
        """Values allowed by this prior (used to check compatibility with other priors)."""
        return (-INF, INF)

    def auto_init(self) -> bool:
        """True when the head's scale should be fitted to the data by ``from_forward``."""
        return getattr(self, "init", None) == "auto"

    def headroom(self) -> float:
        """Ratio between the head's natural scale and the fitted initial level (1 = same).

        Sparse fields have peaks far above their mean level; a headroom > 1 sets the head scale
        above the fitted constant so that peaks are reachable with ``O(1)`` raw outputs.
        """
        return 1.0

    def wrap_head(self, head: Head, context: PriorContext) -> Head:
        """Wrap the resolved head (head modifiers such as known support / conservation)."""
        return head

    # --- field -----------------------------------------------------------------------------
    def inner_ndim(self, ndim: int) -> int:
        """Coordinate dimension of the field wrapped by :meth:`wrap_field`."""
        return ndim

    def wrap_field(self, inner: Field, ndim: int) -> Field:
        return inner

    # --- losses / post / hints -------------------------------------------------------------
    def losses(self, field_name: str, context: PriorContext | None = None) -> LossSpec:
        """``{name: (Loss, relative_strength)}`` contributed by this prior."""
        return {}

    def postprocess(self, field_name: str | None = None) -> list[Postprocess]:
        return []

    def curriculum_hints(self) -> dict[str, Any]:
        """Suggestions for :func:`nefi.auto.auto_curriculum` (e.g. ``anneal_fraction``)."""
        return {}

    def field_hints(self, ndim: int | None = None) -> dict[str, Any]:
        """Suggestions for the representation (e.g. ``max_octaves``, ``include_input``)."""
        return {}

    # --- description -----------------------------------------------------------------------
    def params(self) -> dict[str, Any]:
        out = {}
        for f in fields(self):  # type: ignore[arg-type]
            v = getattr(self, f.name)
            if torch.is_tensor(v):
                v = f"<tensor {tuple(v.shape)}>"
            out[f.name] = v
        return out

    def describe(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params().items())
        return f"{type(self).__name__}({args}): {self.summary}"

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params().items())
        return f"{type(self).__name__}({args})"

    # --- composition -----------------------------------------------------------------------
    def __add__(self, other: Any) -> PriorList:
        return PriorList([self]) + other

    def __radd__(self, other: Any) -> PriorList:
        return PriorList(Prior.parse(other)) + [self]

    @staticmethod
    def parse(spec: Any) -> list[Prior]:
        """Parse the prior DSL (or pass through Prior objects / lists) into a list of priors.

        Example::

            Prior.parse("nonnegative + sparse(l1=1e-2) + piecewise_constant")
            Prior.parse(["bounded(0, 1)", Symmetric("radial")])
        """
        return parse(spec)


class PriorList(list):
    """A list of priors supporting ``+`` with priors, strings and lists.

    Example::

        priors = NonNegative() + "tv" + [Smooth(laplacian=1e-3)]     # PriorList of 3 priors
    """

    def __add__(self, other: Any) -> PriorList:  # type: ignore[override]
        return PriorList(list(self) + parse(other))


# =============================================================================================
# head-defining priors
# =============================================================================================
def _check_init(init: Any, positive: bool) -> None:
    if init is None or init == "auto":
        return
    if isinstance(init, str):
        raise ConfigError(f"init must be 'auto', None or a number, got {init!r}")
    if positive and not float(init) > 0:
        raise ConfigError(f"init must be > 0 for this prior, got {init}")


def _scaled(inner: Head, init: Any) -> Head:
    """Wrap ``inner`` (initialized at 1) in a ScaledHead when an init scale applies."""
    if init is None:
        return inner
    return ScaledHead(inner, scale=1.0 if init == "auto" else float(init))


@dataclass(repr=False)
class Unconstrained(Prior):
    """Signed, unconstrained values (identity/affine head).

    Args:
        init: ``"auto"`` (fit a constant to the data), a number, or ``None`` (start at 0).
        scale, offset: natural units ``x = offset + scale · h`` (``init="auto"`` fits ``scale``).

    Example::

        head = combine("unconstrained(init=1500, scale=100)").head()   # sound speed ~1500
    """

    init: Any = "auto"
    scale: float = 1.0
    offset: float = 0.0
    key: ClassVar[str] = "unconstrained"
    summary: ClassVar[str] = "signed values, affine head x = offset + scale·h"
    head_rank: ClassVar[int] = 0

    def __post_init__(self) -> None:
        _check_init(self.init, positive=False)

    def head(self) -> Head:
        init = None if self.init in (None, "auto") else float(self.init)
        return Affine(scale=self.scale, offset=self.offset, init_value=init)


@dataclass(repr=False)
class NonNegative(Prior):
    """``x ≥ 0`` through a softplus head (``x = s · softplus(h)``).

    Args:
        init: ``"auto"`` fits the level of a constant field that best explains the data
            (``from_forward``), a number sets it, ``None`` uses a bare softplus.
        beta: softplus sharpness.
        peak_ratio: headroom between the head scale ``s`` and the fitted level (the field starts
            at the level; values up to ``~peak_ratio ×`` the level need only ``O(1)`` raw
            outputs). Default 3 (typical peak/mean ratio of dense non-negative fields).

    Example::

        combine("nonnegative(init=0.5)").head()       # ScaledHead(Softplus), starts near 0.5
    """

    init: Any = "auto"
    beta: float = 1.0
    peak_ratio: float = 3.0
    key: ClassVar[str] = "nonnegative"
    summary: ClassVar[str] = "x >= 0 (softplus head)"
    head_rank: ClassVar[int] = 1

    def __post_init__(self) -> None:
        _check_init(self.init, positive=True)

    def headroom(self) -> float:
        return max(1.0, float(self.peak_ratio))

    def interval(self):
        return (0.0, INF)

    def head(self) -> Head:
        if self.init is None:
            return Softplus(beta=self.beta)
        return _scaled(Softplus(beta=self.beta, init_value=1.0), self.init)


@dataclass(repr=False)
class Positive(Prior):
    """``x > 0`` with a log-parameterization ``x = s · exp(h)`` (large dynamic ranges).

    Args:
        init: ``"auto"`` / number / ``None`` as for :class:`NonNegative`.

    Example::

        combine("positive(init=1e-3) + tv").head()     # x = 1e-3 · exp(h)
    """

    init: Any = "auto"
    key: ClassVar[str] = "positive"
    summary: ClassVar[str] = "x > 0, log-parameterized (exp head)"
    head_rank: ClassVar[int] = 2

    def __post_init__(self) -> None:
        _check_init(self.init, positive=True)

    def interval(self):
        return (0.0, INF)

    def head(self) -> Head:
        if self.init is None:
            return ExpHead()
        return _scaled(ExpHead(init_value=1.0), self.init)


@dataclass(repr=False)
class Bounded(Prior):
    """``lo ≤ x ≤ hi`` through a sigmoid head (NeFTY Eq. 6).

    Args:
        lo, hi: bounds (``hi > lo``).
        init: initial value in ``(lo, hi)`` (default: the midpoint).

    Example::

        combine("bounded(lo=0.003, hi=0.25)").head()   # NeFTY diffusivity bracket
    """

    lo: float = 0.0
    hi: float = 1.0
    init: float | None = None
    key: ClassVar[str] = "bounded"
    summary: ClassVar[str] = "lo <= x <= hi (sigmoid head)"
    head_rank: ClassVar[int] = 3

    def __post_init__(self) -> None:
        self.lo, self.hi = float(self.lo), float(self.hi)
        if not self.hi > self.lo:
            raise ConfigError(f"bounded needs hi > lo, got lo={self.lo}, hi={self.hi}")
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)):
            raise ConfigError(
                "bounded needs finite lo and hi; for one-sided bounds use 'nonnegative' "
                "(x >= 0) or 'unconstrained(offset=...)'"
            )
        if self.init is not None and not self.lo < float(self.init) < self.hi:
            raise ConfigError(
                f"bounded init {self.init} must lie strictly in ({self.lo}, {self.hi})"
            )

    def interval(self):
        return (self.lo, self.hi)

    def head(self) -> Head:
        return BoundedHead(self.lo, self.hi, init_value=self.init)


@dataclass(repr=False)
class Sparse(Prior):
    """Mostly-zero, non-negative field: gated softplus head (NeTMY Eq. 5) + L1 penalty.

    Args:
        l1: relative strength of the ``mean |x|`` penalty (0 disables it).
        gated: use the gated-softplus head (lets background pixels switch off cleanly); set
            False to only add the L1 penalty (e.g. together with ``positive`` or ``bounded``).
        init: head scale as for :class:`NonNegative` (only with ``gated=True``).
        peak_ratio: expected ratio between peak values and the mean level of the field; the head
            scale is set to ``peak_ratio ×`` the fitted mean level (the field still starts at the
            mean level) so that isolated peaks are reachable quickly.

    Example::

        spec = combine("nonnegative + sparse(l1=1e-2)")    # gated softplus + L1
    """

    l1: float = 1e-2
    gated: bool = True
    init: Any = "auto"
    peak_ratio: float = 30.0
    key: ClassVar[str] = "sparse"
    summary: ClassVar[str] = "mostly zero: gated softplus head + L1"

    def __post_init__(self) -> None:
        _check_init(self.init, positive=True)
        if self.l1 < 0:
            raise ConfigError("sparse l1 must be >= 0")

    def rank(self) -> int:
        return 2 if self.gated else -1

    def auto_init(self) -> bool:
        return self.gated and self.init == "auto"

    def headroom(self) -> float:
        return max(1.0, float(self.peak_ratio)) if self.gated else 1.0

    def interval(self):
        return (0.0, INF) if self.gated else (-INF, INF)

    def head(self) -> Head | None:
        if not self.gated:
            return None
        if self.init is None:
            return GatedSoftplus()
        return _scaled(GatedSoftplus(init_value=1.0), self.init)

    def losses(self, field_name, context=None):
        return {"l1": (L1(field_name), float(self.l1))} if self.l1 > 0 else {}


@dataclass(repr=False)
class Binary(Prior):
    """Two-phase field ``x ∈ {lo, hi}`` via a sharpening level-set head.

    Args:
        lo, hi: the two phase values.
        sharpen: shrink the interface softness from ``eps_start`` to ``eps_end`` with training
            progress (False keeps ``eps_start``: a smooth two-level field).
        eps_start, eps_end: interface softness in raw units.
        perimeter: relative strength of an interface-length (isotropic TV) penalty (0 = off).

    Example::

        spec = combine("binary(lo=1.0, hi=3.0, perimeter=1e-2)")   # level-set head + perimeter
    """

    lo: float = 0.0
    hi: float = 1.0
    sharpen: bool = True
    eps_start: float = 1.0
    eps_end: float = 0.05
    perimeter: float = 0.0
    key: ClassVar[str] = "binary"
    summary: ClassVar[str] = "two phases {lo, hi} (level-set head, sharpening)"
    head_rank: ClassVar[int] = 4

    def __post_init__(self) -> None:
        self.lo, self.hi = float(self.lo), float(self.hi)
        if not self.hi > self.lo:
            raise ConfigError(f"binary needs hi > lo, got lo={self.lo}, hi={self.hi}")

    def interval(self):
        return (self.lo, self.hi)

    def head(self) -> Head:
        end = self.eps_end if self.sharpen else self.eps_start
        return LevelSetHead(self.lo, self.hi, self.eps_start, end)

    def losses(self, field_name, context=None):
        if self.perimeter <= 0:
            return {}
        per = context.periodic_axes if context else ()
        return {"perimeter": (TV(field_name, isotropic=True, periodic_axes=per), self.perimeter)}

    def curriculum_hints(self):
        return {"anneal_fraction": 0.8}


# =============================================================================================
# loss-only priors
# =============================================================================================
@dataclass(repr=False)
class PiecewiseConstant(Prior):
    """Piecewise-constant field: total-variation penalty (isotropic, NeFTY Eq. 22).

    Args:
        tv: relative strength.
        eps: TV smoothing ``ε`` (field units per unit length).
        isotropic: isotropic (default) or anisotropic (NeTMY Eq. 3) TV.

    Example::

        combine("nonnegative + piecewise_constant(tv=3e-2)").losses()     # {'tv': (TV, 0.03)}
    """

    tv: float = 1e-2
    eps: float = 1e-6
    isotropic: bool = True
    key: ClassVar[str] = "piecewise_constant"
    summary: ClassVar[str] = "piecewise constant (total variation)"

    def losses(self, field_name, context=None):
        if self.tv <= 0:
            return {}
        per = context.periodic_axes if context else ()
        loss = TV(field_name, isotropic=self.isotropic, eps=self.eps, periodic_axes=per)
        return {"tv": (loss, float(self.tv))}


@dataclass(repr=False)
class Smooth(Prior):
    """Smooth field: Laplacian (second-order), gradient (first-order Tikhonov) and/or ridge
    penalties. With no argument: ``laplacian=1e-2``.

    Args:
        laplacian: relative strength of ``mean (Δx)²``.
        tikhonov: relative strength of the ridge ``mean x²``.
        gradient: relative strength of ``mean |∇x|²``.

    Example::

        combine("smooth(gradient=1e-2, tikhonov=1e-4)").losses()     # gradient + tikhonov
    """

    laplacian: float | None = None
    tikhonov: float | None = None
    gradient: float | None = None
    key: ClassVar[str] = "smooth"
    summary: ClassVar[str] = "smooth (Laplacian / gradient / ridge penalties, fewer octaves)"

    def __post_init__(self) -> None:
        if self.laplacian is None and self.tikhonov is None and self.gradient is None:
            self.laplacian = 1e-2

    def losses(self, field_name, context=None):
        per = context.periodic_axes if context else ()
        out: LossSpec = {}
        if self.laplacian:
            out["laplacian"] = (Laplacian(field_name, periodic_axes=per), float(self.laplacian))
        if self.gradient:
            out["gradient"] = (GradientL2(field_name, periodic_axes=per), float(self.gradient))
        if self.tikhonov:
            out["tikhonov"] = (Tikhonov(field_name), float(self.tikhonov))
        return out

    def field_hints(self, ndim=None):
        return {"octave_offset": -2}


@dataclass(repr=False)
class Monotone(Prior):
    """Monotone along an axis (soft penalty on violations).

    Args:
        axis: axis index.
        direction: ``"increasing"`` or ``"decreasing"``.
        weight: relative strength.

    Example::

        combine("nonnegative + monotone(axis=0, direction=decreasing)").losses()
    """

    axis: int = 0
    direction: str = "increasing"
    weight: float = 1.0
    key: ClassVar[str] = "monotone"
    summary: ClassVar[str] = "monotone along an axis (penalty)"

    def losses(self, field_name, context=None):
        return {"monotone": (MonotoneLoss(field_name, self.axis, self.direction), self.weight)}


# =============================================================================================
# modifiers / structural priors
# =============================================================================================
@dataclass(repr=False)
class KnownSupport(Prior):
    """The unknown vanishes (equals ``fill``) outside a known support ``mask`` (1 = inside).

    Args:
        mask: ``(*shape)`` tensor at the native resolution.
        fill: exterior value (default 0, or the lower bound if 0 is not allowed).
        hard: multiplicative mask on the head (stop-gradient, exact) — default; ``False`` adds
            a :class:`~nefi.losses.physics.KnownSupportLoss` instead.
        weight: relative strength of the soft version.

    Example::

        mask = torch.zeros(32, 32); mask[8:24, 8:24] = 1
        head = combine(["nonnegative", KnownSupport(mask)]).head()     # MaskedHead
    """

    mask: Any = None
    fill: float | None = None
    hard: bool = True
    weight: float = 1.0
    key: ClassVar[str] = "known_support"
    summary: ClassVar[str] = "zero outside a known support mask"
    modifier_order: ClassVar[int] = 0

    def __post_init__(self) -> None:
        if self.mask is None:
            raise ConfigError(
                "known_support needs a mask tensor, e.g. prior=['nonnegative', "
                "KnownSupport(mask)] (masks cannot be written in the DSL string)"
            )
        self.mask = torch.as_tensor(self.mask).float()

    def _fill(self, context: PriorContext | None) -> float:
        if self.fill is not None:
            return float(self.fill)
        lo, hi = context.value_range if context else (-INF, INF)
        return 0.0 if lo <= 0.0 <= hi else float(lo)

    def wrap_head(self, head, context):
        if not self.hard:
            return head
        return MaskedHead(head, self.mask, fill=self._fill(context))

    def losses(self, field_name, context=None):
        if self.hard:
            return {}
        loss = KnownSupportLoss(field_name, self.mask, fill=self._fill(context))
        return {"support": (loss, self.weight)}


@dataclass(repr=False)
class Conserved(Prior):
    """The field's total is known (mass / charge / number of particles).

    Args:
        total: the known total.
        kind: ``"sum"`` (integral over the physical domain) or ``"mean"``.
        hard: ``True`` rescales the head output to the exact total (needs a non-negative,
            upper-unbounded head); ``False`` adds a penalty; ``"auto"`` picks hard when possible.
        weight: relative strength of the soft version.

    Example::

        head = combine("nonnegative + conserved(total=2.0)").head()    # MassNormalized
    """

    total: float = 1.0
    kind: str = "sum"
    hard: Any = "auto"
    weight: float = 1.0
    key: ClassVar[str] = "conserved"
    summary: ClassVar[str] = "known total (mass normalization or penalty)"
    modifier_order: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if self.kind not in ("sum", "mean"):
            raise ConfigError(f"conserved kind must be 'sum' or 'mean', got {self.kind!r}")
        if self.hard not in (True, False, "auto"):
            raise ConfigError("conserved hard must be True, False or 'auto'")

    def _is_hard(self, context: PriorContext | None) -> bool:
        lo, hi = context.value_range if context else (-INF, INF)
        ok = lo >= 0.0 and math.isinf(hi)
        if self.hard is True and not ok:
            raise ConfigError(
                "conserved(hard=True) rescales the field and needs a non-negative, "
                "upper-unbounded head: add 'nonnegative' (or 'positive'/'sparse'), or use "
                "conserved(hard=False) for a penalty"
            )
        return bool(self.hard is True or (self.hard == "auto" and ok))

    def wrap_head(self, head, context):
        if not self._is_hard(context):
            return head
        volume = 1.0
        if context is not None and context.domain is not None:
            for s in context.domain.size:
                volume *= float(s)
        return MassNormalized(head, total=self.total, volume=volume, kind=self.kind)

    def losses(self, field_name, context=None):
        if self._is_hard(context):
            return {}
        return {"conservation": (Conservation(field_name, self.total, self.kind), self.weight)}


@dataclass(repr=False)
class Symmetric(Prior):
    """Mirror or radial symmetry, exact by coordinate folding (``hard``) or as a penalty.

    Args:
        kind: ``"mirror_x" | "mirror_y" | "mirror_z" | "mirror_xy" | "mirror" | "radial"``.
        axes: axes for ``"mirror"`` / the radial plane.
        hard: fold coordinates (:class:`~nefi.fields.symmetric.SymmetricField`) — default;
            ``False`` adds a :class:`~nefi.losses.physics.SymmetryLoss`.
        weight: relative strength of the soft version.

    Example::

        spec = combine("nonnegative + symmetric(radial)", ndim=2)   # 1-D inner field, folded r
    """

    kind: str = "mirror_x"
    axes: Any = None
    hard: bool = True
    weight: float = 1.0
    key: ClassVar[str] = "symmetric"
    summary: ClassVar[str] = "mirror / radial symmetry"

    def __post_init__(self) -> None:
        from .fields.symmetric import KINDS

        if self.kind not in KINDS:
            raise ConfigError(f"symmetric kind must be one of {KINDS}, got {self.kind!r}")

    @property
    def wraps_field(self) -> bool:  # type: ignore[override]
        return bool(self.hard)

    def inner_ndim(self, ndim: int) -> int:
        from .fields.symmetric import symmetry_axes

        if self.kind != "radial":
            return ndim
        return 1 + ndim - len(symmetry_axes(self.kind, ndim, self.axes))

    def wrap_field(self, inner, ndim):
        from .fields.symmetric import SymmetricField

        try:
            return SymmetricField(inner, self.kind, ndim=ndim, axes=self.axes)
        except ConfigError as e:
            raise ConfigError(f"{e}. Alternatively use symmetric({self.kind}, hard=False)") from e

    def losses(self, field_name, context=None):
        if self.hard:
            return {}
        return {"symmetry": (SymmetryLoss(field_name, self.kind, self.axes), self.weight)}


@dataclass(repr=False)
class Periodic(Prior):
    """Periodic domain along ``axes`` (default: all): regularizers wrap around; with all axes
    periodic the neural field uses a purely periodic Fourier encoding (no raw coordinates).

    Example::

        combine("periodic(axes=(1,)) + tv", ndim=2).losses()["tv"][0].periodic_axes   # (1,)
    """

    axes: Any = None
    key: ClassVar[str] = "periodic"
    summary: ClassVar[str] = "periodic boundaries (TV/Laplacian wrap, periodic encoding)"

    def resolved_axes(self, ndim: int | None) -> tuple[int, ...]:
        if self.axes is None:
            # all axes; with an unknown dimension cover up to 3-D (extra axes are ignored)
            return tuple(range(ndim if ndim else 3))
        ax = (self.axes,) if isinstance(self.axes, int) else tuple(self.axes)
        return tuple(a % ndim for a in ax) if ndim else ax

    def field_hints(self, ndim=None):
        if ndim and len(self.resolved_axes(ndim)) == ndim:
            return {"include_input": False}
        return {}


@dataclass(repr=False)
class Scale(Prior):
    """Recover the absolute scale after fitting (energy-anchored correction, NeTMY Eq. 30).

    Needed with scale-free fidelities; harmless (α≈1) otherwise.

    Args:
        homogeneity: degree ``p`` of the forward in this field (default: the operator's).

    Example::

        combine("nonnegative + scale(homogeneity=2)").postprocess()   # [EnergyScaleCorrection]
    """

    homogeneity: float | None = None
    key: ClassVar[str] = "scale"
    summary: ClassVar[str] = "energy-anchored scale correction after fitting"

    def postprocess(self, field_name=None):
        return [EnergyScaleCorrection(field=field_name, homogeneity=self.homogeneity)]


# =============================================================================================
# DSL
# =============================================================================================
PRIORS: dict[str, type[Prior]] = {
    "nonnegative": NonNegative,
    "nonneg": NonNegative,
    "non_negative": NonNegative,
    "positive": Positive,
    "log": Positive,
    "bounded": Bounded,
    "box": Bounded,
    "range": Bounded,
    "sparse": Sparse,
    "piecewise_constant": PiecewiseConstant,
    "piecewise": PiecewiseConstant,
    "tv": PiecewiseConstant,
    "smooth": Smooth,
    "binary": Binary,
    "two_phase": Binary,
    "level_set": Binary,
    "known_support": KnownSupport,
    "support": KnownSupport,
    "symmetric": Symmetric,
    "symmetry": Symmetric,
    "conserved": Conserved,
    "mass": Conserved,
    "periodic": Periodic,
    "scale": Scale,
    "monotone": Monotone,
    "unconstrained": Unconstrained,
    "real": Unconstrained,
    "signed": Unconstrained,
}


def register_prior(name: str, cls: type[Prior]) -> type[Prior]:
    """Make a custom :class:`Prior` subclass available in the DSL under ``name``.

    Example::

        register_prior("nonneg2", NonNegative)       # alias
        assert isinstance(Prior.parse("nonneg2")[0], NonNegative)
    """
    PRIORS[name.lower()] = cls
    return cls


def _split_top_level(s: str, sep: str = "+") -> list[str]:
    parts, depth, cur = [], 0, []
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            prev = s[i - 1] if i > 0 else ""
            if prev in "eE" and i >= 2 and (s[i - 2].isdigit() or s[i - 2] == "."):
                cur.append(ch)  # exponent sign like 1e+3
                continue
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    if depth != 0:
        raise ConfigError(f"unbalanced parentheses in prior spec {s!r}")
    return parts


def _literal(node: ast.AST, term: str) -> Any:
    if isinstance(node, ast.Name):
        if node.id.lower() in ("inf", "infinity"):
            return INF
        if node.id.lower() == "nan":
            return math.nan
        return node.id
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal(node.operand, term)
    if isinstance(node, ast.Tuple | ast.List):
        return tuple(_literal(e, term) for e in node.elts)
    try:
        return ast.literal_eval(node)
    except ValueError as e:
        raise ConfigError(
            f"prior argument {ast.unparse(node)!r} in {term!r} is not a literal; use numbers, "
            "strings, True/False/None or tuples"
        ) from e


def _lookup(name: str) -> type[Prior]:
    key = name.strip().lower().replace("-", "_")
    if key in PRIORS:
        return PRIORS[key]
    close = difflib.get_close_matches(key, list(PRIORS), n=3)
    hint = f" Did you mean {', '.join(repr(c) for c in close)}?" if close else ""
    raise ConfigError(f"unknown prior {name!r}.{hint} Known priors: {sorted(set(PRIORS))}")


def _parse_term(term: str) -> Prior:
    t = term.strip()
    head, paren, rest = t.partition("(")
    t = head.strip().replace("-", "_").replace(" ", "_") + paren + rest
    try:
        node = ast.parse(t, mode="eval").body
    except SyntaxError as e:
        raise ConfigError(
            f"cannot parse prior {term!r}; expected name or name(arg=value, ...)"
        ) from e
    if isinstance(node, ast.Name):
        cls, args, kwargs = _lookup(node.id), [], {}
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        cls = _lookup(node.func.id)
        args = [_literal(a, term) for a in node.args]
        kwargs = {k.arg: _literal(k.value, term) for k in node.keywords if k.arg}
    else:
        raise ConfigError(f"cannot parse prior {term!r}; expected name or name(arg=value, ...)")
    try:
        return cls(*args, **kwargs)
    except TypeError as e:
        sig = inspect.signature(cls)
        raise ConfigError(
            f"bad arguments for prior {term.strip()!r}: {e}. Signature: {cls.__name__}{sig}"
        ) from e


def parse(spec: Any) -> list[Prior]:
    """Parse ``spec`` (DSL string, Prior, list/tuple of those, CombinedPrior) into priors.

    Example::

        parse("nonnegative + sparse(l1=1e-2) + piecewise_constant(tv=1e-2)")
        parse("bounded(lo=1500, hi=4500) + smooth")
        parse(["binary(0, 1)", Symmetric("mirror_x")])
    """
    if spec is None:
        return []
    if isinstance(spec, Prior):
        return [spec]
    if isinstance(spec, CombinedPrior):
        return list(spec.priors)
    if isinstance(spec, str):
        return [_parse_term(t) for t in _split_top_level(spec) if t.strip()]
    if isinstance(spec, Sequence):
        out: list[Prior] = []
        for s in spec:
            out.extend(parse(s))
        return out
    raise ConfigError(
        f"cannot interpret prior spec of type {type(spec).__name__}; use a string such as "
        "'nonnegative + piecewise_constant', a Prior, or a list of those"
    )


# =============================================================================================
# combination
# =============================================================================================
_CONFLICT_HINTS: dict[frozenset, str] = {
    frozenset({"positive", "sparse"}): (
        "for a sparse non-negative field use 'nonnegative + sparse'; for a strictly positive "
        "field with a large dynamic range use 'positive + sparse(gated=False)'"
    ),
    frozenset({"binary", "sparse"}): (
        "use 'binary(lo, hi) + sparse(gated=False)' to add only the L1 penalty"
    ),
}


def _dedup_key(p: Prior) -> tuple:
    vals = []
    for f in fields(p):  # type: ignore[arg-type]
        v = getattr(p, f.name)
        vals.append((f.name, id(v) if torch.is_tensor(v) else repr(v)))
    return (type(p), tuple(vals))


def _contains(outer: tuple[float, float], inner: tuple[float, float], tol: float = 1e-12) -> bool:
    return inner[0] >= outer[0] - tol and inner[1] <= outer[1] + tol


@dataclass
class CombinedPrior:
    """The resolved set of priors for one field (see :func:`combine`).

    Example::

        spec = combine("nonnegative + tv", field="rho", ndim=2)
        spec.head(), spec.losses(), spec.curriculum_hints(), print(spec.describe())

    Attributes:
        priors: the (deduplicated) priors.
        field: field name.
        top: the head-defining prior (an :class:`Unconstrained` default if none was given).
        periodic_axes: axes declared periodic.
    """

    priors: list[Prior]
    field: str = "x"
    ndim: int | None = None
    domain: Any = None
    top: Prior = dc_field(default_factory=Unconstrained)
    periodic_axes: tuple[int, ...] = ()

    # --- resolved pieces ---------------------------------------------------------------------
    def context(self) -> PriorContext:
        from .fields.modifiers import value_range

        return PriorContext(
            self.field, self.ndim, self.domain, self.periodic_axes, value_range(self.base_head())
        )

    def base_head(self) -> Head:
        return self.top.head()  # type: ignore[return-value]

    def head(self) -> Head:
        """The output head: the top prior's head wrapped by modifiers (support, conservation)."""
        head = self.base_head()
        ctx = self.context()
        mods = sorted(
            (p for p in self.priors if p.modifier_order >= 0), key=lambda p: p.modifier_order
        )
        for p in mods:
            head = p.wrap_head(head, ctx)
        return head

    @property
    def auto_init(self) -> bool:
        return self.top.auto_init()

    def losses(self) -> LossSpec:
        """``{name: (Loss, relative_strength)}`` of every prior (names made unique)."""
        ctx = self.context()
        out: LossSpec = {}
        for p in self.priors:
            for k, v in p.losses(self.field, ctx).items():
                name, i = k, 2
                while name in out:
                    name, i = f"{k}{i}", i + 1
                out[name] = v
        return out

    def postprocess(self) -> list[Postprocess]:
        out: list[Postprocess] = []
        for p in self.priors:
            out.extend(p.postprocess(self.field))
        return out

    def curriculum_hints(self) -> dict[str, Any]:
        hints: dict[str, Any] = {}
        for p in self.priors:
            for k, v in p.curriculum_hints().items():
                if k == "anneal_fraction" and k in hints:
                    hints[k] = max(hints[k], v)
                else:
                    hints[k] = v
        return hints

    def field_hints(self) -> dict[str, Any]:
        hints: dict[str, Any] = {}
        for p in self.priors:
            for k, v in p.field_hints(self.ndim).items():
                if k == "octave_offset":
                    hints[k] = min(hints.get(k, 0), v)
                elif k == "include_input":
                    hints[k] = hints.get(k, True) and v
                else:
                    hints[k] = v
        if self.periodic_axes:
            hints["periodic_axes"] = self.periodic_axes
        return hints

    def build_field(self, builder: Callable[[int], Field], ndim: int) -> Field:
        """Build the (possibly symmetry-wrapped) field; ``builder(ndim)`` makes the inner field."""
        wrappers = [p for p in self.priors if p.wraps_field]

        def build(i: int, nd: int) -> Field:
            if i == len(wrappers):
                return builder(nd)
            w = wrappers[i]
            return w.wrap_field(build(i + 1, w.inner_ndim(nd)), nd)

        return build(0, ndim)

    def describe(self) -> str:
        lines = [f"priors for field {self.field!r}:"]
        for p in self.priors:
            mark = " [head]" if p is self.top else ""
            lines.append(f"  - {p.describe()}{mark}")
        if self.top not in self.priors:
            lines.append(f"  - {self.top.describe()} [default head]")
        return "\n".join(lines)


def combine(
    priors: Any, field: str = "x", ndim: int | None = None, domain: Any = None
) -> CombinedPrior:
    """Resolve a set of priors for one field: pick the head, check compatibility.

    Rules: exactly one prior defines the output head — the most specific one (``binary`` >
    ``bounded`` > ``positive``/``sparse`` > ``nonnegative`` > ``unconstrained``) wins if its value
    interval satisfies every other head-defining prior (e.g. ``nonnegative + sparse`` → gated
    softplus; ``nonnegative + bounded(0, 1)`` → sigmoid); otherwise a :class:`ConfigError`
    explains the contradiction. Other priors add losses, head modifiers, field wrappers,
    post-processing and hints.

    Example::

        spec = combine("nonnegative + sparse(l1=1e-2) + piecewise_constant", field="rho", ndim=2)
        head, losses = spec.head(), spec.losses()
    """
    plist: list[Prior] = []
    keys: list[tuple] = []
    for p in parse(priors):
        key = _dedup_key(p)
        if key not in keys:
            keys.append(key)
            plist.append(p)
    head_defs = [p for p in plist if p.rank() >= 0]
    top: Prior
    if not head_defs:
        top = Unconstrained()
    else:
        top = max(head_defs, key=lambda p: p.rank())
        rivals = [p for p in head_defs if p is not top and p.rank() == top.rank()]
        if rivals:
            r = rivals[0]
            hint = _CONFLICT_HINTS.get(frozenset({top.key, r.key}), "keep only one of them")
            raise ConfigError(
                f"priors {top!r} and {r!r} both define the output transform of field "
                f"{field!r}; {hint}"
            )
        for p in head_defs:
            if p is top:
                continue
            if not _contains(p.interval(), top.interval()):
                raise ConfigError(
                    f"{top!r} allows values in {list(top.interval())}, which contradicts {p!r} "
                    f"(values in {list(p.interval())}) for field {field!r}; adjust the bounds so "
                    "the more specific prior respects the others (e.g. bounded(lo=0, hi=...) "
                    "with nonnegative) or drop one of them"
                )
            log.info("prior %r is implied by the head of %r (field %r)", p, top, field)
    periodic: tuple[int, ...] = ()
    for p in plist:
        if isinstance(p, Periodic):
            periodic = tuple(sorted(set(periodic) | set(p.resolved_axes(ndim))))
    for p in plist:
        if isinstance(p, Symmetric) and p.hard and isinstance(p.axes, int):
            p.axes = (p.axes,)
    return CombinedPrior(plist, field, ndim, domain, top, periodic)


def describe_catalogue() -> str:
    """Markdown table of the DSL (name, class, meaning) — used by the docs.

    Example::

        print(describe_catalogue())
    """
    seen: dict[type, list[str]] = {}
    for k, cls in PRIORS.items():
        seen.setdefault(cls, []).append(k)
    rows = ["| DSL name | class | meaning |", "|---|---|---|"]
    for cls, names in seen.items():
        rows.append(
            f"| `{names[0]}` ({', '.join(names[1:]) or '-'}) | `{cls.__name__}` | {cls.summary} |"
        )
    return "\n".join(rows)


__all__ = [
    "PRIORS",
    "Binary",
    "Bounded",
    "CombinedPrior",
    "Conserved",
    "KnownSupport",
    "Monotone",
    "NonNegative",
    "Periodic",
    "PiecewiseConstant",
    "Positive",
    "Prior",
    "PriorContext",
    "PriorList",
    "Scale",
    "Smooth",
    "Sparse",
    "Symmetric",
    "Unconstrained",
    "combine",
    "describe_catalogue",
    "parse",
    "register_prior",
]
