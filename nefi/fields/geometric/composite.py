"""Composite fields: one unknown assembled from several sub-fields (background ⊕ anomaly, ...).

*The representation is the geometric prior.* A first-order step on the parameters of a field
``x = f_θ`` realizes ``Δx ≈ −η J_θ J_θᵀ ∇_x L = −η G_θ ∇_x L`` (NeTMY Lemma 2 / Eq. 7,
App. D.6). For a composite ``x = c(f_1, …, f_n)`` with parameters split across the components the
kernel decomposes as

    G_θ = Σ_i D_i G_i D_iᵀ,        G_i = J_i J_iᵀ,   D_i = ∂c/∂f_i   (pointwise),

so each component contributes *its own* filter. A low-octave neural or Fourier-basis background
contributes a smooth low-pass ``G_1``; a gated / level-set / shape anomaly contributes a localized
or interface-concentrated ``G_2``. The sum realizes an update that is smooth where the unknown is
smooth and sharp where it is sharp — the structure of NeFTY's "defects in a (layered) bulk"
(NeFTY §5, App. G.6) and of NeTMY's sparse sources over a background.

Per-component curricula are first-class:

* ``progress_map`` gives every component its own annealing progress as a function of the global
  progress (e.g. background ``"full"`` — all bands open from the start — and anomaly
  ``("delay", 0.5)``: *background first, anomaly second*);
* ``weight_map`` ramps a component in (towards the combiner's identity element);
* stage ``freeze`` prefixes freeze components: components are registered as direct children, so
  ``Stage(freeze=("anomaly.",))`` (see :meth:`CompositeField.freeze_prefixes`) fits the
  background alone.

Example::

    import nefi
    from nefi.fields import Heads, NeuralField, GatedSoftplus
    from nefi.fields.geometric import CompositeField, FourierBasisField

    bg = FourierBasisField(2, n_modes=6)
    an = NeuralField(2, Heads({"a": GatedSoftplus(init_value=0.05)}), hidden=32, depth=3)
    field = CompositeField({"background": bg, "anomaly": an}, "sum",
                           progress_map={"background": "full", "anomaly": ("delay", 0.25)})
    x = field(nefi.Domain.unit((32, 32)).coords(), progress=0.5)["x"]
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ...errors import ConfigError
from ...registry import register
from ..base import Field
from ..heads import Heads
from ._utils import ProgressFn, interp_grid, progress_fn

Tensors = Sequence[torch.Tensor]


# ------------------------------------------------------------------------------------------
# combiners
# ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Combiner:
    """A rule ``c(t_1, …, t_n)`` combining component tensors ``(*shape, C_i)`` (broadcastable).

    Attributes:
        fn: the combination function.
        identity: identity element used by ``weight_map`` ramps (``t_w = e + w (t − e)``): a
            float for every component, a per-component tuple (``None`` = not weightable), or
            ``None`` if no component may be weighted.
        n_components: required number of components (``None`` = any number ``>= 1``).
        description: one-line formula for reprs / docs.
    """

    fn: Callable[[Tensors], torch.Tensor]
    identity: float | tuple[float | None, ...] | None = None
    n_components: int | None = None
    description: str = ""

    def identity_for(self, i: int) -> float | None:
        if isinstance(self.identity, tuple):
            return self.identity[i] if i < len(self.identity) else None
        return self.identity


COMBINERS: dict[str, Combiner] = {}


def register_combiner(
    name: str,
    fn: Callable[[Tensors], torch.Tensor],
    identity: float | tuple[float | None, ...] | None = None,
    n_components: int | None = None,
    description: str = "",
) -> Combiner:
    """Register a named combiner usable as ``CompositeField(..., combiner=name)``.

    Example::

        register_combiner("soft_union", lambda ts: 1 - torch.prod(torch.stack(
            [1 - t for t in torch.broadcast_tensors(*ts)]), 0), identity=0.0)
    """
    comb = Combiner(fn, identity, n_components, description)
    COMBINERS[name.lower()] = comb
    return comb


def _sum(ts: Tensors) -> torch.Tensor:
    out = ts[0]
    for t in ts[1:]:
        out = out + t
    return out


def _product(ts: Tensors) -> torch.Tensor:
    out = ts[0]
    for t in ts[1:]:
        out = out * t
    return out


def _mean(ts: Tensors) -> torch.Tensor:
    return _sum(ts) / len(ts)


def _max(ts: Tensors) -> torch.Tensor:
    out = ts[0]
    for t in ts[1:]:
        out = torch.maximum(out, t)
    return out


def _min(ts: Tensors) -> torch.Tensor:
    out = ts[0]
    for t in ts[1:]:
        out = torch.minimum(out, t)
    return out


def _blend(ts: Tensors) -> torch.Tensor:
    base, insert, mask = ts
    return base + mask * (insert - base)


register_combiner("sum", _sum, 0.0, None, "x = Σ_i f_i")
register_combiner("product", _product, 1.0, None, "x = Π_i f_i")
register_combiner("mean", _mean, None, None, "x = mean_i f_i")
register_combiner("max", _max, None, None, "x = max_i f_i")
register_combiner("min", _min, None, None, "x = min_i f_i")
register_combiner(
    "blend",
    _blend,
    (None, None, 0.0),
    3,
    "x = base·(1 − m) + insert·m  (components base, insert, m)",
)


def resolve_combiner(
    combiner: str | Combiner | Callable[[Tensors], torch.Tensor],
    identity: float | None = None,
) -> tuple[str, Combiner]:
    """``(name, Combiner)`` from a registered name, a :class:`Combiner` or a plain callable."""
    if isinstance(combiner, Combiner):
        return combiner.description or "custom", combiner
    if isinstance(combiner, str):
        key = combiner.lower()
        if key not in COMBINERS:
            raise ConfigError(f"unknown combiner {combiner!r}; registered: {sorted(COMBINERS)}")
        return key, COMBINERS[key]
    if callable(combiner):
        name = getattr(combiner, "__name__", "custom")
        return name, Combiner(combiner, identity, None, name)
    raise ConfigError(f"cannot interpret {combiner!r} as a combiner")


# ------------------------------------------------------------------------------------------
# CompositeField
# ------------------------------------------------------------------------------------------
_RESERVED = {"heads", "forward", "raw", "training", "components", "combine"}


@register("field", "composite")
class CompositeField(Field):
    """Combine named sub-fields into one field; heads are applied to the combined raw tensor.

    Each component is evaluated at its own progress ``progress_map[name](progress)``; its output
    is the stack of its head outputs (``use_heads=True``, physical units) or its raw pre-head
    tensor (``use_heads=False``), shape ``(*shape, C_i)`` with ``C_i ∈ {1, heads.n_in}``. The
    combiner merges them into ``(*shape, heads.n_in)`` and the composite heads produce the named
    fields (default: identity heads named after the first component's fields).

    Filtering view: ``G_θ = Σ_i D_i G_i D_iᵀ`` (see the module docstring) — the composite's update
    kernel is the combiner-weighted sum of the components' kernels (NeTMY Lemma 2).

    Args:
        components: mapping (or sequence of pairs) ``name -> Field``; names become attribute /
            parameter prefixes (``"background.…"``) and must be valid identifiers.
        combiner: ``"sum"`` | ``"product"`` | ``"mean"`` | ``"max"`` | ``"min"`` | ``"blend"``
            (``base, insert, mask``), any name registered with :func:`register_combiner`, a
            :class:`Combiner`, or a callable ``fn(list_of_tensors) -> tensor``.
        heads: heads applied to the combined tensor (default: identity heads).
        progress_map: ``name -> progress spec`` (see :func:`~._utils.progress_fn`: ``"full"``,
            ``("delay", 0.5)``, a float, a callable, ...). Missing names use the global progress.
        weight_map: ``name -> spec`` of a multiplicative ramp ``w(progress)`` pulling the
            component towards the combiner's identity element (``t ← e + w (t − e)``); only for
            combiners with an identity (sum: 0, product: 1, blend mask: 0).
        use_heads: combine the components' head outputs (default) or their raw tensors; a bool
            or a per-component mapping.
        combiner_identity: identity element for a plain-callable combiner (enables weights).
    """

    def __init__(
        self,
        components: Mapping[str, Field] | Sequence[tuple[str, Field]],
        combiner: str | Combiner | Callable[[Tensors], torch.Tensor] = "sum",
        heads: Heads | Mapping | None = None,
        progress_map: Mapping[str, Any] | None = None,
        weight_map: Mapping[str, Any] | None = None,
        use_heads: bool | Mapping[str, bool] = True,
        combiner_identity: float | None = None,
    ) -> None:
        items = list(components.items()) if isinstance(components, Mapping) else list(components)
        if not items:
            raise ConfigError("CompositeField needs at least one component")
        names = [str(n) for n, _ in items]
        if len(set(names)) != len(names):
            raise ConfigError(f"duplicate component names: {names}")
        uh = {
            n: bool(use_heads.get(n, True)) if isinstance(use_heads, Mapping) else bool(use_heads)
            for n in names
        }
        first_name, first = items[0]
        if heads is None:
            if uh[first_name]:
                heads = Heads({k: "identity" for k in first.names})
            elif first.heads.n_in == 1:
                heads = Heads({first.primary: "identity"})
            else:
                raise ConfigError(
                    "use_heads=False with a multi-channel first component: pass `heads` explicitly"
                )
        super().__init__(heads)
        for name, comp in items:
            if not isinstance(comp, Field):
                raise ConfigError(f"component {name!r} is not a Field: {type(comp).__name__}")
            if not name.isidentifier() or name in _RESERVED or hasattr(self, name):
                raise ConfigError(
                    f"invalid component name {name!r}: use an identifier that is not an existing "
                    "attribute (e.g. 'background', 'anomaly')"
                )
            self.add_module(name, comp)
        self._names: tuple[str, ...] = tuple(names)
        self._use_heads = uh
        self.combiner_name, self.combiner = resolve_combiner(combiner, combiner_identity)
        if self.combiner.n_components is not None and len(names) != self.combiner.n_components:
            raise ConfigError(
                f"combiner {self.combiner_name!r} needs {self.combiner.n_components} components, "
                f"got {len(names)}"
            )
        for label, mp in (("progress_map", progress_map), ("weight_map", weight_map)):
            unknown = set(mp or {}) - set(names)
            if unknown:
                raise ConfigError(
                    f"{label} has unknown components {sorted(unknown)}; known {names}"
                )
        self._progress: dict[str, ProgressFn] = {
            n: progress_fn((progress_map or {}).get(n)) for n in names
        }
        self._weights: dict[str, ProgressFn | None] = {}
        for i, n in enumerate(names):
            spec = (weight_map or {}).get(n)
            if spec is None:
                self._weights[n] = None
                continue
            if self.combiner.identity_for(i) is None:
                raise ConfigError(
                    f"combiner {self.combiner_name!r} has no identity element for component "
                    f"{n!r}; weight_map is not supported there"
                )
            self._weights[n] = progress_fn(spec)
        self._check_channels()

    # --- structure ----------------------------------------------------------------------
    @property
    def component_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def components(self) -> dict[str, Field]:
        """Ordered ``name -> Field`` mapping."""
        return {n: getattr(self, n) for n in self._names}

    def component(self, name: str) -> Field:
        if name not in self._names:
            raise ConfigError(f"no component {name!r}; known {self._names}")
        return getattr(self, name)

    def component_channels(self, name: str) -> int:
        comp = self.component(name)
        return len(comp.names) if self._use_heads[name] else comp.heads.n_in

    def _check_channels(self) -> None:
        n = self.heads.n_in
        for name in self._names:
            c = self.component_channels(name)
            if c not in (1, n):
                raise ConfigError(
                    f"component {name!r} provides {c} channels but the composite heads expect "
                    f"{n} (or 1, broadcast)"
                )

    def freeze_prefixes(self, *names: str) -> tuple[str, ...]:
        """Stage ``freeze`` prefixes that freeze the given components (default: all)."""
        names = names or self._names
        for n in names:
            self.component(n)
        return tuple(f"{n}." for n in names)

    def component_progress(self, name: str, progress: float) -> float:
        """The progress a component sees at global ``progress``."""
        return float(self._progress[name](progress))

    # --- evaluation -----------------------------------------------------------------------
    def component_tensor(
        self, name: str, coords: torch.Tensor, progress: float = 1.0
    ) -> torch.Tensor:
        """Component output ``(*shape, C_i)`` at its mapped progress, weight ramp applied."""
        comp = self.component(name)
        p = self.component_progress(name, progress)
        if self._use_heads[name]:
            out = comp(coords, p)
            t = torch.stack([out[k] for k in comp.names], dim=-1)
        else:
            t = comp.raw(coords, p)
        wfn = self._weights[name]
        if wfn is not None:
            w = float(wfn(progress))
            if w != 1.0:
                e = float(self.combiner.identity_for(self._names.index(name)))  # type: ignore[arg-type]
                t = e + w * (t - e)
        return t

    def combine(
        self, tensors: list[torch.Tensor], coords: torch.Tensor, progress: float
    ) -> torch.Tensor:
        """Merge component tensors (override in subclasses for parameterized combinations)."""
        return self.combiner.fn(tensors)

    def raw(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        ts = [self.component_tensor(n, coords, progress) for n in self._names]
        out = self.combine(ts, coords, progress)
        n = self.heads.n_in
        if out.shape[-1] != n:
            out = out.expand(*out.shape[:-1], n)
        return out

    @torch.no_grad()
    def component_outputs(
        self, coords: torch.Tensor, progress: float = 1.0
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Each component's own head outputs at its mapped progress (for plots / diagnostics)."""
        return {
            n: getattr(self, n)(coords, self.component_progress(n, progress)) for n in self._names
        }

    # --- hooks ----------------------------------------------------------------------------
    def on_stage_start(self, stage, domain) -> None:
        for n in self._names:
            getattr(self, n).on_stage_start(stage, domain)

    def reset_parameters(self) -> None:
        for n in self._names:
            getattr(self, n).reset_parameters()
        self._reset_own()

    def _reset_own(self) -> None:
        """Re-initialize parameters owned by the composite itself (subclass hook)."""

    def extra_repr(self) -> str:
        maps = {n: repr(self._progress[n]) for n in self._names}
        return (
            f"combiner={self.combiner_name!r}, components={self._names}, "
            f"heads={self.heads.names}, progress_map={maps}"
        )


# ------------------------------------------------------------------------------------------
# AnomalyField
# ------------------------------------------------------------------------------------------
@register("field", "anomaly")
class AnomalyField(CompositeField):
    """``background ⊕ anomaly`` with the anomaly gated by a support prior.

    Modes (``a`` = the anomaly's output channel times the support ``s(x) ∈ [0, 1]``):

    * ``"add"``: ``x = b + c · a`` (``c`` = contrast per channel; sign encodes inclusion
      (``c > 0``) vs defect (``c < 0``); learnable with ``learn_contrast``);
    * ``"multiply"``: ``x = b · (1 + c · a)`` (relative contrast; positivity-preserving for
      ``c · a > −1``);
    * ``"blend"``: ``x = b · (1 − a) + v · a`` with an inclusion value ``v`` (learnable) and an
      indicator ``a ∈ [0, 1]`` — *replacement* semantics, natural for defects whose property does
      not depend on the host (NeFTY: air-like voids in any layer, App. E.1). With a bounded
      background and ``inclusion_bounds`` the result stays in the admissible range.

    The anomaly is typically a gated field (``GatedSoftplus`` head, NeTMY Eq. 5: sparse,
    non-negative, background driven to ~0 by the gate) for ``add``/``multiply``, or a sharpening
    level-set indicator for ``blend``. The *support prior* restricts where anomalies may appear:
    a tensor mask on the native grid (e.g. a thresholded sensitivity map — trust region,
    NeFTY App. B.4), or a callable ``coords -> [0, 1]``.

    Components are named ``"background"`` and ``"anomaly"`` (freeze prefixes ``"background."``,
    ``"anomaly."``).

    Args:
        background: any field producing the host medium (``C`` channels).
        anomaly: any field; its ``anomaly_channel`` output (default primary) is used.
        mode: ``"add"`` | ``"multiply"`` | ``"blend"``.
        heads: final heads (default: identity heads named like the background's fields).
        support: ``None`` | tensor mask ``(*native_shape)`` | callable ``coords -> (*shape)``.
        contrast: initial contrast ``c`` (float or per-channel list) for add / multiply.
        learn_contrast: optimize ``c`` (default False: fixed sign and scale, the anomaly field
            carries the amplitude).
        inclusion_value: initial inclusion value ``v`` for blend (float or per-channel list).
        learn_inclusion: optimize ``v``.
        inclusion_bounds: optional ``(lo, hi)``: ``v = lo + (hi − lo) σ(raw)`` (hard bracket,
            NeFTY Eq. 6).
        anomaly_channel: name of the anomaly output to use (default: its primary field).
        progress_map / weight_map: per-component curricula (see :class:`CompositeField`); the
            anomaly's identity element is 0 (``weight_map={"anomaly": ("delay", 0.3)}`` ramps the
            anomaly in after the background).

    Example::

        field = AnomalyField(bg, NeuralField(2, Heads({"a": GatedSoftplus(init_value=0.05)})),
                             mode="add", contrast=-1.0)            # defects lower the property
    """

    MODES = ("add", "multiply", "blend")

    def __init__(
        self,
        background: Field,
        anomaly: Field,
        mode: str = "add",
        heads: Heads | Mapping | None = None,
        support: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        contrast: float | Sequence[float] = 1.0,
        learn_contrast: bool = False,
        inclusion_value: float | Sequence[float] = 0.0,
        learn_inclusion: bool = True,
        inclusion_bounds: tuple[float, float] | None = None,
        anomaly_channel: str | None = None,
        progress_map: Mapping[str, Any] | None = None,
        weight_map: Mapping[str, Any] | None = None,
    ) -> None:
        if mode not in self.MODES:
            raise ConfigError(f"unknown anomaly mode {mode!r}; choose from {self.MODES}")
        self.mode = mode
        ch = anomaly_channel or anomaly.primary
        if ch not in anomaly.names:
            raise ConfigError(f"anomaly has no output {ch!r}; outputs: {anomaly.names}")
        self._anomaly_index = anomaly.names.index(ch)
        comb = Combiner(_sum, (None, 0.0), 2, f"anomaly[{mode}]")
        super().__init__(
            {"background": background, "anomaly": anomaly},
            combiner=comb,
            heads=heads if heads is not None else Heads({k: "identity" for k in background.names}),
            progress_map=progress_map,
            weight_map=weight_map,
        )
        c = self.heads.n_in
        self._contrast_init = _per_channel(contrast, c, "contrast")
        self._inclusion_init = _per_channel(inclusion_value, c, "inclusion_value")
        self.inclusion_bounds = None
        if inclusion_bounds is not None:
            lo, hi = (float(v) for v in inclusion_bounds)
            if not hi > lo:
                raise ConfigError(f"inclusion_bounds needs hi > lo, got {inclusion_bounds}")
            self.inclusion_bounds = (lo, hi)
        self.contrast = nn.Parameter(self._contrast_init.clone(), requires_grad=learn_contrast)
        self.inclusion = nn.Parameter(self._inclusion_raw_init(), requires_grad=learn_inclusion)
        self._support_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
        if support is None:
            self.register_buffer("support_mask", None)
        elif torch.is_tensor(support):
            self.register_buffer("support_mask", support.detach().float().clone())
        elif callable(support):
            self.register_buffer("support_mask", None)
            self._support_fn = support
        else:
            raise ConfigError("support must be None, a tensor mask or a callable coords -> mask")

    def _check_channels(self) -> None:
        n = self.heads.n_in
        c = self.component_channels("background")
        if c not in (1, n):
            raise ConfigError(f"background provides {c} channels, heads expect {n}")

    def _inclusion_raw_init(self) -> torch.Tensor:
        v = self._inclusion_init.clone()
        if self.inclusion_bounds is None:
            return v
        lo, hi = self.inclusion_bounds
        u = ((v - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
        return torch.logit(u)

    def _reset_own(self) -> None:
        with torch.no_grad():
            self.contrast.copy_(self._contrast_init.to(self.contrast))
            self.inclusion.copy_(self._inclusion_raw_init().to(self.inclusion))

    # --- pieces ---------------------------------------------------------------------------
    def inclusion_value(self) -> torch.Tensor:
        """The inclusion value ``v`` per channel (blend mode)."""
        if self.inclusion_bounds is None:
            return self.inclusion
        lo, hi = self.inclusion_bounds
        return lo + (hi - lo) * torch.sigmoid(self.inclusion)

    def support_at(self, coords: torch.Tensor) -> torch.Tensor | None:
        """Support prior ``s(x)`` of shape ``coords.shape[:-1]`` (``None`` = everywhere)."""
        if self._support_fn is not None:
            return self._support_fn(coords).to(coords.dtype)
        m = self.support_mask
        if m is None:
            return None
        shape = tuple(coords.shape[:-1])
        if tuple(m.shape) == shape:
            return m.to(coords)
        if m.ndim != coords.shape[-1]:
            raise ConfigError(
                f"support mask is {m.ndim}-D but coordinates are {coords.shape[-1]}-D; pass a "
                "mask on the field's native grid"
            )
        return interp_grid(m.to(coords).unsqueeze(0), coords)[..., 0]

    def combine(self, tensors, coords, progress):
        b, an = tensors
        a = an[..., self._anomaly_index : self._anomaly_index + 1]
        s = self.support_at(coords)
        if s is not None:
            a = a * s.unsqueeze(-1)
        if self.mode == "add":
            return b + self.contrast.to(b) * a
        if self.mode == "multiply":
            return b * (1.0 + self.contrast.to(b) * a)
        return b + a * (self.inclusion_value().to(b) - b)

    @torch.no_grad()
    def anomaly_map(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        """Effective anomaly ``a(x)·s(x)`` (the gated, support-restricted indicator/amplitude)."""
        an = self.component_tensor("anomaly", coords, progress)
        a = an[..., self._anomaly_index]
        s = self.support_at(coords)
        return a if s is None else a * s

    @torch.no_grad()
    def background_map(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        """The background's primary output at its mapped progress."""
        bg = self.component("background")
        return bg(coords, self.component_progress("background", progress))[bg.primary]

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}, " + super().extra_repr()


def _per_channel(value: float | Sequence[float], n: int, label: str) -> torch.Tensor:
    t = torch.as_tensor(value, dtype=torch.float32).flatten()
    if t.numel() == 1:
        return t.expand(n).clone()
    if t.numel() != n:
        raise ConfigError(f"{label} needs 1 or {n} values, got {t.numel()}")
    return t.clone()


__all__ = [
    "COMBINERS",
    "AnomalyField",
    "Combiner",
    "CompositeField",
    "register_combiner",
    "resolve_combiner",
]
