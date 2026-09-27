"""Stratified media: layered fields with learnable monotone interfaces (NeFTY layered setting).

A :class:`LayeredField` represents ``K`` layers stacked along a *depth axis* (default: the last
axis, NeFTY's through-thickness ``z``). Its unknowns are

* ``K − 1`` interface height fields ``z_k(ℓ)`` over the lateral coordinates ``ℓ``, kept
  monotonically ordered by construction — cumulative softplus thicknesses
  ``z_k = top + s Σ_{j ≤ k} softplus(b_j + δ_j(ℓ))`` — where ``b_j`` are learnable scalars (the
  mean layer thicknesses) and ``δ_j(ℓ)`` an optional lateral model (``"constant"`` = flat layers,
  ``"grid"`` = a coarse bilinear table, ``"neural"`` = a small annealed neural field);
* per-layer values ``v_k(ℓ)`` (learnable scalars, optionally with a lateral model).

It renders the volume by smooth Heaviside transitions whose sharpness increases with progress,

    raw(ℓ, z) = v_0 + Σ_{k=1}^{K−1} (v_k − v_{k−1}) · H_ε(z − z_k),   ε = ε(progress),

and by default *cell-averages* ``H_ε`` over each voxel (``render="area"``): as ``ε → 0`` this is
the exact partial-volume fraction, so interfaces stay differentiable with sub-voxel resolution
even when perfectly sharp.

Why a dedicated representation: the NeFTY layered benchmark (App. E.1: three to four bulk strata
along ``z``, 1–4 embedded defects) asks a generic neural field to spend capacity on sharp,
laterally extended jumps that a handful of scalars describe exactly. The layered field's update
kernel ``G_θ`` (NeTMY Lemma 2) is rank ``O(K)`` for flat layers — it cannot express defects, but it
cannot fit noise either; pair it with an anomaly (:class:`LayerCakeField`) for "defects in a
laminate" (NeFTY Fig. 5, Fig. 13).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from ...errors import ConfigError, ShapeError
from ...registry import register
from ...utils.tensor import shape_tuple
from ..base import Field
from ..heads import GatedSoftplus, Heads
from ..neural import NeuralField
from ._utils import (
    anneal_value,
    axis_spacing,
    interp_grid,
    inv_softplus,
    level_set_head,
    progress_fn,
    smooth_step,
)
from .composite import AnomalyField

LATERAL_MODELS = ("constant", "grid", "neural")


class LateralModel(nn.Module):
    """Perturbation ``δ(ℓ)`` of ``n_out`` quantities over the lateral coordinates (zero at init).

    Args:
        kind: ``"constant"`` (no lateral variation), ``"grid"`` (bilinear table of
            ``lateral_shape`` cells over ``[-1, 1]^L``), ``"neural"`` (small annealed
            :class:`~nefi.fields.NeuralField` with a zero-initialized output layer), or a
            :class:`~nefi.fields.base.Field` with ``n_out`` raw channels.
        n_out: number of perturbed quantities.
        lat_ndim: number of lateral axes ``L`` (0 for 1-D domains: only ``"constant"``).
        lateral_shape: table size for ``"grid"``.
        hidden, depth, n_octaves: network size for ``"neural"``.
    """

    def __init__(
        self,
        kind: str | Field,
        n_out: int,
        lat_ndim: int,
        lateral_shape: int | Sequence[int] = 8,
        hidden: int = 32,
        depth: int = 2,
        n_octaves: int = 4,
    ) -> None:
        super().__init__()
        self.n_out, self.lat_ndim = int(n_out), int(lat_ndim)
        if isinstance(kind, Field):
            if kind.heads.n_in != self.n_out:
                raise ConfigError(
                    f"lateral field must have {self.n_out} raw channels, got {kind.heads.n_in}"
                )
            self.kind = "field"
            self.net = kind
            return
        kind = str(kind).lower()
        if kind not in LATERAL_MODELS:
            raise ConfigError(f"unknown lateral model {kind!r}; choose from {LATERAL_MODELS}")
        if self.lat_ndim == 0 and kind != "constant":
            raise ConfigError("a 1-D layered field has no lateral axes; use the 'constant' model")
        self.kind = kind
        if kind == "grid":
            gs = shape_tuple(lateral_shape)
            if len(gs) == 1 and self.lat_ndim > 1:
                gs = gs * self.lat_ndim
            if len(gs) != self.lat_ndim:
                raise ConfigError(f"lateral_shape {gs} does not match {self.lat_ndim} lateral axes")
            self.table = nn.Parameter(torch.zeros(self.n_out, *gs))
        elif kind == "neural":
            self.net = NeuralField(
                self.lat_ndim,
                Heads({f"c{i}": "identity" for i in range(self.n_out)}),
                hidden=hidden,
                depth=depth,
                skip_at=None,
                n_octaves=n_octaves,
                out_init_scale=0.0,
            )

    def forward(self, lat: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        if self.kind == "constant":
            return lat.new_zeros(*lat.shape[:-1], self.n_out)
        if self.kind == "grid":
            return interp_grid(self.table, lat)
        return self.net.raw(lat, progress)

    def reset_parameters(self) -> None:
        if self.kind == "grid":
            with torch.no_grad():
                self.table.zero_()
        elif self.kind in ("neural", "field"):
            self.net.reset_parameters()


@register("field", "layered")
class LayeredField(Field):
    """Stratified field with ``n_layers`` layers and monotone learnable interfaces.

    Works in any dimension: 1-D (a piecewise-constant depth profile), 2-D (1-D interface curves
    ``z_k(x)``), 3-D (2-D interface surfaces ``z_k(x, y)``). Coordinates are normalized to
    ``[-1, 1]``; interface positions and softness are in normalized depth units.

    Args:
        ndim: domain dimension.
        n_layers: number of layers ``K >= 1``.
        heads: heads applied to the rendered raw volume (default identity ``"x"``).
        depth_axis: stratification axis (default last, NeFTY ``z``).
        values: initial per-layer raw values (``K`` floats, or ``K × C``); default: the heads'
            suggested initial raw value for every layer.
        learn_values: optimize the per-layer values.
        interfaces: initial interface positions (``K − 1`` increasing normalized depths ``> top``);
            default: equally spaced.
        interface_model: lateral model of the interfaces (see :class:`LateralModel`).
        value_model: lateral model of the layer values.
        lateral_shape / hidden / depth / n_octaves: lateral-model sizes.
        top: normalized depth where the first layer starts (the observed face, default ``-1``).
        thickness_scale: ``s`` in ``z_k = top + s Σ softplus(·)`` (normalized units).
        eps_start, eps_end, schedule: interface softness ``ε(progress)`` (normalized depth units;
            geometric decay by default). ``ε`` is reset at every stage start with the progress.
        render: ``"area"`` (cell-averaged transitions → partial volumes; needs grid coordinates,
            falls back to point sampling otherwise) or ``"point"``.
        lateral_progress: progress map for the lateral models (see
            :func:`~nefi.fields.geometric._utils.progress_fn`).
        n_active: number of active layers (the rest are merged into the deepest active one);
            :meth:`grow` activates them one at a time (capacity growth,
            :class:`~nefi.fields.adaptive.GrowCapacity`).

    The few geometric scalars can get their own step size with
    ``OptimConfig(lr_mult={"field.thickness_logit": ..., "field.layer_values": ...})`` (inside a
    composite: ``"field.background."``).

    Example::

        f = LayeredField(3, n_layers=3, interface_model="grid", lateral_shape=6)
        x = f(nefi.Domain.unit((32, 32, 16)).coords(), progress=1.0)["x"]
        z = f.interfaces()          # (2,) interface depths at the lateral center
    """

    def __init__(
        self,
        ndim: int,
        n_layers: int = 2,
        heads: Heads | Mapping | None = None,
        depth_axis: int = -1,
        values: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor | None = None,
        learn_values: bool = True,
        interfaces: Sequence[float] | None = None,
        interface_model: str | Field = "constant",
        value_model: str | Field = "constant",
        lateral_shape: int | Sequence[int] = 8,
        hidden: int = 32,
        depth: int = 2,
        n_octaves: int = 4,
        top: float = -1.0,
        thickness_scale: float = 1.0,
        eps_start: float = 0.25,
        eps_end: float = 0.01,
        schedule: str = "geometric",
        render: str = "area",
        lateral_progress: Any = None,
        n_active: int | None = None,
    ) -> None:
        super().__init__(heads)
        self.ndim = int(ndim)
        if self.ndim < 1:
            raise ConfigError("LayeredField needs ndim >= 1")
        self.K = int(n_layers)
        if self.K < 1:
            raise ConfigError("n_layers must be >= 1")
        self.depth_axis = int(depth_axis) % self.ndim
        self.lat_axes = tuple(a for a in range(self.ndim) if a != self.depth_axis)
        if render not in ("area", "point"):
            raise ConfigError(f"render must be 'area' or 'point', got {render!r}")
        self.render = render
        self.top = float(top)
        self.thickness_scale = float(thickness_scale)
        if self.thickness_scale <= 0:
            raise ConfigError("thickness_scale must be positive")
        self.eps_start, self.eps_end, self.schedule = float(eps_start), float(eps_end), schedule
        anneal_value(0.5, self.eps_start, self.eps_end, schedule)  # validate early
        self._lat_progress = progress_fn(lateral_progress)
        c = self.heads.n_in
        self.C = c

        # interfaces: cumulative softplus thicknesses
        if interfaces is None:
            z0 = [self.top + (1.0 - self.top) * k / self.K for k in range(1, self.K)]
        else:
            z0 = [float(v) for v in interfaces]
        if len(z0) != self.K - 1:
            raise ConfigError(f"need {self.K - 1} interface positions, got {len(z0)}")
        prev = self.top
        thick = []
        for z in z0:
            if not z > prev:
                raise ConfigError(
                    f"interfaces must be increasing and below top={self.top}: got {z0}"
                )
            thick.append((z - prev) / self.thickness_scale)
            prev = z
        self._thickness_init = inv_softplus(thick) if thick else torch.zeros(0)
        self.thickness_logit = nn.Parameter(self._thickness_init.clone())

        # values
        self._values_init = self._initial_values(values)
        self.layer_values = nn.Parameter(self._values_init.clone(), requires_grad=learn_values)

        lat = len(self.lat_axes)
        self.interface_model = LateralModel(
            interface_model, max(self.K - 1, 1), lat, lateral_shape, hidden, depth, n_octaves
        )
        self.value_model = LateralModel(
            value_model, self.K * c, lat, lateral_shape, hidden, depth, n_octaves
        )
        self._n_active_init = self.K if n_active is None else int(n_active)
        if not 1 <= self._n_active_init <= self.K:
            raise ConfigError(f"n_active must be in [1, {self.K}], got {n_active}")
        self.register_buffer("n_active", torch.tensor(self._n_active_init), persistent=True)

    # --- init -------------------------------------------------------------------------------
    def _initial_values(self, values) -> torch.Tensor:
        c = self.heads.n_in
        if values is None:
            base = self.heads.init_bias()
            return base.expand(self.K, c).clone()
        t = torch.as_tensor(values, dtype=torch.float32)
        if t.ndim == 0:
            t = t.expand(self.K)
        if t.ndim == 1:
            if t.numel() != self.K:
                raise ConfigError(f"values needs {self.K} entries (one per layer), got {t.numel()}")
            t = t.unsqueeze(-1).expand(self.K, c)
        if tuple(t.shape) != (self.K, c):
            raise ConfigError(f"values must have shape ({self.K}, {c}), got {tuple(t.shape)}")
        return t.clone()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.thickness_logit.copy_(self._thickness_init.to(self.thickness_logit))
            self.layer_values.copy_(self._values_init.to(self.layer_values))
            self.n_active.fill_(self._n_active_init)
        self.interface_model.reset_parameters()
        self.value_model.reset_parameters()

    # --- schedules --------------------------------------------------------------------------
    def eps(self, progress: float = 1.0) -> float:
        """Interface softness ``ε(progress)`` in normalized depth units."""
        return anneal_value(progress, self.eps_start, self.eps_end, self.schedule)

    # --- lateral evaluation -----------------------------------------------------------------
    def _columns(self, coords: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Lateral coordinates; deduplicated along the depth axis on tensor-product grids."""
        lat_idx = list(self.lat_axes)
        d = self.depth_axis
        if coords.ndim == self.ndim + 1 and self.ndim > 1 and coords.shape[d] > 1:
            lat_all = coords[..., lat_idx]
            first = lat_all.select(d, 0)
            if torch.equal(lat_all, first.unsqueeze(d).expand_as(lat_all)):
                return first, True
        return coords[..., lat_idx], False

    def _expand(
        self, t: torch.Tensor, coords: torch.Tensor, dedup: bool, tail: int
    ) -> torch.Tensor:
        if not dedup:
            return t
        t = t.unsqueeze(self.depth_axis)
        shape = tuple(coords.shape[:-1]) + tuple(t.shape[-tail:])
        return t.expand(*shape)

    def _interfaces_from(self, lat: torch.Tensor, progress: float) -> torch.Tensor:
        n = self.K - 1
        if n == 0:
            return lat.new_zeros(*lat.shape[:-1], 0)
        delta = self.interface_model(lat, self._lat_progress(progress))[..., :n]
        logits = self.thickness_logit.to(lat) + delta
        thick = torch.nn.functional.softplus(logits) * self.thickness_scale
        return self.top + torch.cumsum(thick, dim=-1)

    def _values_from(self, lat: torch.Tensor, progress: float) -> torch.Tensor:
        dv = self.value_model(lat, self._lat_progress(progress))
        dv = dv.reshape(*dv.shape[:-1], self.K, self.C)
        return self.layer_values.to(lat) + dv

    def interfaces(
        self, lateral_coords: torch.Tensor | None = None, progress: float = 1.0
    ) -> torch.Tensor:
        """Interface depths ``(*lateral, K − 1)`` (normalized; ``None`` → at the lateral center).

        Ordered by construction: ``top < z_1 < z_2 < … < z_{K−1}`` everywhere.
        """
        if lateral_coords is None:
            lat = self.thickness_logit.new_zeros(1, len(self.lat_axes))
            return self._interfaces_from(lat, progress)[0]
        if lateral_coords.shape[-1] != len(self.lat_axes):
            raise ShapeError(
                f"lateral coordinates need {len(self.lat_axes)} components, "
                f"got {lateral_coords.shape[-1]}"
            )
        return self._interfaces_from(lateral_coords, progress)

    def values(
        self, lateral_coords: torch.Tensor | None = None, progress: float = 1.0
    ) -> torch.Tensor:
        """Per-layer raw values ``(*lateral, K, C)`` (``None`` → base values ``(K, C)``)."""
        if lateral_coords is None:
            return self.layer_values
        return self._values_from(lateral_coords, progress)

    # --- rendering --------------------------------------------------------------------------
    def raw(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        if coords.shape[-1] != self.ndim:
            raise ShapeError(f"LayeredField expects {self.ndim}-D coordinates, got {coords.shape}")
        lat, dedup = self._columns(coords)
        v = self._expand(self._values_from(lat, progress), coords, dedup, 2)
        out = v[..., 0, :]
        n = int(self.n_active) - 1
        if n <= 0:
            return out
        z = self._expand(self._interfaces_from(lat, progress), coords, dedup, 1)[..., :n]
        u = coords[..., self.depth_axis].unsqueeze(-1)
        h = axis_spacing(coords, self.depth_axis) if self.render == "area" else None
        step = smooth_step(u - z, self.eps(progress), h)  # (*shape, n)
        jumps = v[..., 1 : n + 1, :] - v[..., :n, :]  # (*shape, n, C)
        return out + (step.unsqueeze(-1) * jumps).sum(-2)

    @torch.no_grad()
    def layer_index(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        """Hard layer assignment ``(*shape)`` (long): number of active interfaces above."""
        lat, dedup = self._columns(coords)
        n = int(self.n_active) - 1
        if n <= 0:
            return torch.zeros(coords.shape[:-1], dtype=torch.long, device=coords.device)
        z = self._expand(self._interfaces_from(lat, progress), coords, dedup, 1)[..., :n]
        u = coords[..., self.depth_axis].unsqueeze(-1)
        return (u > z).sum(-1)

    # --- capacity growth ------------------------------------------------------------------
    def can_grow(self) -> bool:
        return int(self.n_active) < self.K

    @torch.no_grad()
    def grow(self, hint: Mapping | None = None) -> bool:
        """Activate the next (deeper) layer without changing the current field.

        The new layer starts with the value of the layer it splits from (zero jump), so the
        rendered field is unchanged at the moment of growth and the new interface/value start
        receiving gradients from there on.
        """
        n = int(self.n_active)
        if n >= self.K:
            return False
        self.layer_values[n].copy_(self.layer_values[n - 1])
        self.n_active.fill_(n + 1)
        return True

    def capacity(self) -> dict[str, int]:
        return {"layers": int(self.n_active), "max_layers": self.K}

    def extra_repr(self) -> str:
        return (
            f"ndim={self.ndim}, n_layers={self.K}, depth_axis={self.depth_axis}, "
            f"interface_model={self.interface_model.kind!r}, "
            f"value_model={self.value_model.kind!r}, "
            f"eps=({self.eps_start}->{self.eps_end}), render={self.render!r}, "
            f"heads={self.heads.names}"
        )


def default_anomaly_field(
    ndim: int,
    mode: str = "blend",
    hidden: int = 32,
    depth: int = 3,
    n_octaves: int = 6,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    init_value: float | None = None,
    **neural_kw: Any,
) -> NeuralField:
    """A small anomaly network suited to an :class:`AnomalyField` mode.

    ``"blend"``: sharpening level-set indicator in ``[0, 1]`` (initially ~off, ``init_value``
    default 0.02). ``"add"`` / ``"multiply"``: gated softplus amplitude (NeTMY Eq. 5; default
    initial amplitude 0.05, gate half-closed).
    """
    if mode == "blend":
        head = level_set_head(
            0.0, 1.0, eps_start, eps_end, "geometric", 0.02 if init_value is None else init_value
        )
        heads = Heads({"m": head})
    else:
        heads = Heads(
            {
                "a": GatedSoftplus(
                    init_value=0.05 if init_value is None else init_value, gate_init=-2.0
                )
            }
        )
    neural_kw.setdefault("skip_at", None)
    return NeuralField(ndim, heads, hidden=hidden, depth=depth, n_octaves=n_octaves, **neural_kw)


@register("field", "layer_cake")
class LayerCakeField(AnomalyField):
    """Defects in a laminate: ``LayeredField`` background ⊕ gated anomaly (NeFTY layered setting).

    Equivalent to ``AnomalyField(LayeredField(...), anomaly, mode)``; components are named
    ``"background"`` (the layered medium) and ``"anomaly"``. The default is an *additive gated*
    anomaly, ``x = layers − a(x)`` with ``a = softplus(h)·σ(g) ≥ 0`` (NeTMY Eq. 5 gate; defects
    lower the host value, ``contrast = −1``; pass ``contrast=+1`` for inclusions). With
    ``mode="blend"`` defects *replace* the host value by a learnable inclusion value (NeFTY
    App. E.1 samples the defect diffusivity independently of the bulk); pair blend with a compact
    shape anomaly (e.g. :class:`~nefi.fields.geometric.StarShapeField` with ``inside=1,
    outside=0``): a free-form neural indicator can *swap roles* with the layers (cover a whole
    stratum and turn that stratum's value into the "defect").

    Args:
        ndim: domain dimension (2 or 3 typical; 1 allowed without a lateral model).
        n_layers: number of strata.
        anomaly: anomaly field (default :func:`default_anomaly_field` for ``mode``).
        mode: ``"add"`` (default) | ``"multiply"`` | ``"blend"`` (see :class:`AnomalyField`).
        name: output field name (e.g. ``"alpha"``).
        depth_axis: stratification axis.
        layer_values / interfaces / interface_model / value_model: forwarded to
            :class:`LayeredField` (``layer_values`` in field units: the layered background uses
            an identity head).
        layer_kw: extra :class:`LayeredField` kwargs (softness schedule, lateral sizes, ...).
        anomaly_kw: kwargs for :func:`default_anomaly_field` when ``anomaly`` is None.
        heads, support, contrast, learn_contrast, inclusion_value, learn_inclusion,
            inclusion_bounds, progress_map, weight_map: see :class:`AnomalyField`.

    Example::

        f = LayerCakeField(3, n_layers=3, name="alpha", layer_values=[0.15, 0.12, 0.18],
                           progress_map={"background": ("fast", 0.5), "anomaly": ("delay", 0.25)})
        shapes = LayerCakeField(3, 3, mode="blend", inclusion_value=0.01,
                                inclusion_bounds=(0.003, 0.25),
                                anomaly=StarShapeField(3, n_shapes=4, n_active=1, inside=1.0,
                                                       outside=0.0, learn_values=False))
    """

    def __init__(
        self,
        ndim: int,
        n_layers: int = 2,
        anomaly: Field | None = None,
        mode: str = "add",
        name: str = "x",
        depth_axis: int = -1,
        layer_values: Sequence[float] | None = None,
        interfaces: Sequence[float] | None = None,
        interface_model: str | Field = "constant",
        value_model: str | Field = "constant",
        layer_kw: Mapping[str, Any] | None = None,
        anomaly_kw: Mapping[str, Any] | None = None,
        heads: Heads | Mapping | None = None,
        support: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        contrast: float | Sequence[float] = -1.0,
        learn_contrast: bool = False,
        inclusion_value: float | Sequence[float] = 0.0,
        learn_inclusion: bool = True,
        inclusion_bounds: tuple[float, float] | None = None,
        progress_map: Mapping[str, Any] | None = None,
        weight_map: Mapping[str, Any] | None = None,
    ) -> None:
        background = LayeredField(
            ndim,
            n_layers,
            heads=Heads({name: "identity"}),
            depth_axis=depth_axis,
            values=layer_values,
            interfaces=interfaces,
            interface_model=interface_model,
            value_model=value_model,
            **dict(layer_kw or {}),
        )
        if anomaly is None:
            anomaly = default_anomaly_field(ndim, mode, **dict(anomaly_kw or {}))
        super().__init__(
            background,
            anomaly,
            mode=mode,
            heads=heads,
            support=support,
            contrast=contrast,
            learn_contrast=learn_contrast,
            inclusion_value=inclusion_value,
            learn_inclusion=learn_inclusion,
            inclusion_bounds=inclusion_bounds,
            progress_map=progress_map,
            weight_map=weight_map,
        )

    @property
    def layers(self) -> LayeredField:
        """The layered background component."""
        return self.component("background")  # type: ignore[return-value]


__all__ = [
    "LATERAL_MODELS",
    "LateralModel",
    "LayerCakeField",
    "LayeredField",
    "default_anomaly_field",
]
