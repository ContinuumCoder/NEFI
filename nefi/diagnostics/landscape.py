"""Loss-landscape diagnostics in *field space* (what a free-pixel solver sees).

* :func:`field_gradient` — the raw gradient ``∇_x L`` of the loss with respect to the field values
  (the update a free-density / grid solver executes verbatim, NeTMY §4.4);
* :func:`iter0_gradient` — that gradient at a uniform initialization together with its
  center/outer ratio: the (P2)+(P3) signature of NeTMY §5.3 (18.29× under F2, peak at the grid
  center rather than at any source);
* :func:`center_mass_ratio` — fraction of mass in a central disk (NeTMY Fig. 4c);
* :func:`energy_barrier` — loss along the straight path between two field states (NeTMY Fig. 4b:
  ``ρ(t) = (1−t) ρ_collapse + t ρ⋆`` crosses an ``h ≈ 1.12`` barrier at ``t = 0.20`` under F2 and is
  monotone under F1);
* :func:`ansatz_objective` — the loss as a function of a low-dimensional ansatz (feed it to
  :func:`nefi.diagnostics.hessian_condition_number`, NeTMY App. E.8 step 4).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from ..errors import ConfigError
from ._common import Fields, loss_value, normalized_radius, resolve_fields, resolve_pixel
from .sensitivity import center_to_outer_ratio


def field_gradient(
    problem: Any,
    fields: Any = None,
    shape: Sequence[int] | None = None,
    *,
    progress: float = 1.0,
    terms: Any = None,
    stage: Any = None,
) -> dict[str, torch.Tensor]:
    """Raw field-space gradient ``∇_x L`` at given field values.

    The field tensors are made autograd leaves and handed to the problem's losses through a
    :class:`~nefi.losses.base.Context`, exactly as during optimization, but without the field
    parameterization in between (``G_θ = I``).

    Args:
        problem: the inverse problem.
        fields: linearization point (``None`` = current field values; tensor = primary field;
            mapping; or a :class:`~nefi.solve.result.Result`).
        shape: grid shape (default: native or the shape of ``fields``).
        progress: annealing progress (only matters for the current field values).
        terms: ``None`` (all active terms), ``"data"``, a name, or a list of names.
        stage: optional :class:`~nefi.solve.curriculum.Stage` whose loss-weight overrides apply.

    Returns:
        ``{name: gradient}`` for every field (zeros for fields the loss does not depend on).
    """
    x_fields, dom = resolve_fields(problem, fields, shape, progress)
    with torch.enable_grad():
        leaves = {k: v.detach().clone().requires_grad_(True) for k, v in x_fields.items()}
        total, _ = loss_value(problem, leaves, dom, progress, terms, stage)
        grads = torch.autograd.grad(total, list(leaves.values()), allow_unused=True)
    return {
        k: (torch.zeros_like(v) if g is None else g.detach())
        for (k, v), g in zip(leaves.items(), grads)
    }


def uniform_fields(
    problem: Any,
    shape: Sequence[int] | None = None,
    value: float | Mapping[str, float] | None = None,
    progress: float = 0.0,
) -> Fields:
    """Spatially constant fields, by default at the mean of the current (iteration-0) values.

    Args:
        problem: the inverse problem.
        shape: grid shape (default native).
        value: constant for every field, or per-field mapping; missing entries use the mean of
            the current field values (a neural field starts near-uniform at its head's
            ``init_value``, so this mimics the papers' uniform initialization).
        progress: progress at which the current values are evaluated.
    """
    base, _ = resolve_fields(problem, None, shape, progress)
    out = {}
    for k, v in base.items():
        if isinstance(value, Mapping):
            c = value.get(k)
        else:
            c = value
        c = float(v.mean()) if c is None else float(c)
        out[k] = torch.full_like(v, c)
    return out


@dataclass
class Iter0Gradient:
    """The field-space gradient at initialization and its center-bias signature (NeTMY (P2)).

    Attributes:
        grad: gradient of the analyzed field (signed).
        grads: gradients of every field.
        ratio: center-to-outer ratio of ``|grad|`` (NeTMY F2 reference: 18.29×).
        peak_index: multi-index of ``argmax |grad|``.
        peak_radius: normalized distance of that peak from the grid center (0 = center).
        center_mass: :func:`center_mass_ratio` of ``|grad|``.
        field: name of the analyzed field.
        init: ``"uniform"`` or ``"current"``.
        terms: which loss terms were differentiated.
    """

    grad: torch.Tensor
    grads: dict[str, torch.Tensor]
    ratio: float
    peak_index: tuple[int, ...]
    peak_radius: float
    center_mass: float
    field: str
    init: str
    terms: Any = "data"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ratio": self.ratio,
            "peak_index": list(self.peak_index),
            "peak_radius": self.peak_radius,
            "center_mass": self.center_mass,
            "field": self.field,
            "init": self.init,
            "terms": self.terms if isinstance(self.terms, str | type(None)) else list(self.terms),
        }


def iter0_gradient(
    problem: Any,
    init: str = "uniform",
    *,
    shape: Sequence[int] | None = None,
    value: float | Mapping[str, float] | None = None,
    field: str | None = None,
    terms: Any = "data",
    center_radius: float = 0.2,
    ring: tuple[float, float] = (0.8, 1.0),
    radius_fraction: float = 0.15,
    axes: Sequence[int] | None = None,
) -> Iter0Gradient:
    """Field-space gradient at initialization plus its center/outer ratio (NeTMY (P2)+(P3)).

    A free-density solver executes this gradient verbatim at its first step; a peak at the grid
    center (instead of at a source) and a large center/outer ratio predict centered collapse.

    Args:
        problem: the inverse problem.
        init: ``"uniform"`` (constant fields, see :func:`uniform_fields`) or ``"current"`` (the
            field module's own iteration-0 output at ``progress=0``).
        shape: grid shape (default native).
        value: constant(s) for ``init="uniform"``.
        field: field to analyze (default primary).
        terms: loss terms to differentiate (default: data fidelity only, as in the paper).
        center_radius: see :func:`center_to_outer_ratio`.
        ring: see :func:`center_to_outer_ratio`.
        radius_fraction: see :func:`center_mass_ratio`.
        axes: axes defining the radius (default all).
    """
    if init == "uniform":
        x = uniform_fields(problem, shape, value)
    elif init == "current":
        x, _ = resolve_fields(problem, None, shape, progress=0.0)
    else:
        raise ConfigError(f"init must be 'uniform' or 'current', got {init!r}")
    grads = field_gradient(problem, x, progress=0.0, terms=terms)
    name = field or problem.field.primary
    g = grads[name]
    a = g.abs()
    peak = resolve_pixel(int(torch.argmax(a)), a.shape)
    r = normalized_radius(a.shape, axes, device=a.device, dtype=torch.float64)
    return Iter0Gradient(
        grad=g,
        grads=grads,
        ratio=center_to_outer_ratio(g, center_radius, ring, axes),
        peak_index=peak,
        peak_radius=float(r[peak]),
        center_mass=center_mass_ratio(a, radius_fraction, axes),
        field=name,
        init=init,
        terms=terms,
    )


def center_mass_ratio(
    x: torch.Tensor, radius_fraction: float = 0.15, axes: Sequence[int] | None = None
) -> float:
    """Fraction of the total mass ``Σ|x|`` inside a central disk (NeTMY Fig. 4c).

    The disk radius is ``radius_fraction`` times the grid side length (0.15 → a disk of diameter
    30 % of the side). A spatially uniform 2-D field gives ``π·0.15² ≈ 0.071``; NeTMY reports 0.00
    (NeTMY) < 0.064 (GaussianSplat) < 0.081 (L-BFGS) < 0.153 (ADMM) < 0.223 (Tikhonov) after 200
    iterations, and 0.55 → 0.78 for ADMM from F1 to F2 (App. E.5, Tab. 9).
    """
    a = x.detach().abs().double()
    total = float(a.sum())
    if total == 0.0:
        return 0.0
    r = normalized_radius(a.shape, axes, device=a.device, dtype=a.dtype)
    return float(a[r <= 2.0 * radius_fraction].sum()) / total


def uniform_center_mass(
    shape: Sequence[int], radius_fraction: float = 0.15, axes: Sequence[int] | None = None
) -> float:
    """:func:`center_mass_ratio` of a spatially uniform field (the no-bias baseline).

    ``≈ π·0.15² = 0.071`` on a 2-D grid, ``0.30`` on a 1-D grid; compare a measured ratio to it.
    """
    r = normalized_radius(shape, axes, dtype=torch.float64)
    return float((r <= 2.0 * radius_fraction).double().mean())


def interpolate_fields(a: Fields, b: Fields, t: float) -> Fields:
    """``(1 − t) a + t b`` for the fields present in both (others taken from ``a``)."""
    return {k: ((1.0 - t) * v + t * b[k]) if k in b else v for k, v in a.items()}


@dataclass
class EnergyBarrier:
    """Loss profile along ``x(t) = (1 − t) x_a + t x_b`` (NeTMY Fig. 4b).

    Attributes:
        t: interpolation parameters.
        loss: total (selected) loss at each ``t``.
        components: per-term loss values at each ``t``.
        height: escape barrier from ``x_a``: ``max(0, max_t L(t) − L(0))`` (NeTMY F2: 1.12).
        t_max: location of the maximum of ``L(t)`` (NeTMY F2: 0.20).
        monotone: ``True`` if the loss never increases along the path (NeTMY F1).
    """

    t: torch.Tensor
    loss: torch.Tensor
    components: dict[str, torch.Tensor] = field(default_factory=dict)
    height: float = 0.0
    t_max: float = 0.0
    monotone: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t.tolist(),
            "loss": self.loss.tolist(),
            "height": self.height,
            "t_max": self.t_max,
            "monotone": self.monotone,
        }


def energy_barrier(
    problem: Any,
    fields_a: Any,
    fields_b: Any,
    n: int = 21,
    *,
    shape: Sequence[int] | None = None,
    terms: Any = None,
    progress: float = 1.0,
    rtol: float = 1e-9,
) -> EnergyBarrier:
    """Losses along the straight line between two field states (NeTMY Fig. 4b).

    Typical use: ``fields_a`` = a centrally collapsed iterate (or the current solution) and
    ``fields_b`` = the ground truth. A positive ``height`` means gradient descent started at
    ``fields_a`` must climb before it can reach ``fields_b`` along this path.

    Args:
        problem: the inverse problem.
        fields_a: start state (tensor for the primary field, mapping, or Result).
        fields_b: end state (same conventions; missing fields are taken from ``fields_a``).
        n: number of points on ``t ∈ [0, 1]`` (endpoints included).
        shape: grid shape (default: shape of ``fields_a``).
        terms: loss terms (default: total loss, as in the paper).
        progress: annealing progress for fields not given explicitly.
        rtol: relative tolerance for the monotonicity test.
    """
    if n < 2:
        raise ConfigError("energy_barrier needs n >= 2")
    xa, dom = resolve_fields(problem, fields_a, shape, progress)
    xb, _ = resolve_fields(problem, fields_b, dom.shape, progress)
    ts = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    losses: list[float] = []
    comps: dict[str, list[float]] = {}
    with torch.no_grad():
        for t in ts.tolist():
            total, c = loss_value(problem, interpolate_fields(xa, xb, t), dom, progress, terms)
            losses.append(float(total))
            for k, v in c.items():
                comps.setdefault(k, []).append(float(v))
    loss = torch.tensor(losses, dtype=torch.float64)
    i = int(torch.argmax(loss))
    scale = max(1e-30, float(loss.abs().max()))
    monotone = bool(torch.all(loss[1:] <= loss[:-1] + rtol * scale))
    return EnergyBarrier(
        t=ts,
        loss=loss,
        components={k: torch.tensor(v, dtype=torch.float64) for k, v in comps.items()},
        height=max(0.0, float(loss[i] - loss[0])),
        t_max=float(ts[i]),
        monotone=monotone,
    )


def ansatz_objective(
    problem: Any,
    ansatz: Callable[[torch.Tensor, torch.Tensor], Any],
    *,
    shape: Sequence[int] | None = None,
    terms: Any = None,
    progress: float = 1.0,
    physical: bool = True,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """The problem's loss as a differentiable function of a low-dimensional ansatz.

    Example (NeTMY App. E.8 step 4, Gaussian density ansatz ``ρ = A exp(−‖r‖²/2σ²)``)::

        def gaussian(p, r):          # p = (log10 A, sigma), r = physical coords (*shape, 2)
            return 10 ** p[0] * torch.exp(-(r ** 2).sum(-1) / (2 * p[1] ** 2))

        fn = ansatz_objective(problem, gaussian)
        kappa = hessian_condition_number(fn, [0.0, 40.0])

    Args:
        problem: the inverse problem.
        ansatz: ``(params, coords) -> tensor | {name: tensor}``; a tensor is the primary field and
            other fields keep their current values.
        shape: grid shape (default native).
        terms: loss terms (default total).
        progress: progress for the current values of fields the ansatz does not produce.
        physical: pass physical coordinates (default) instead of normalized ``[-1, 1]`` ones.

    Returns:
        ``fn(params) -> scalar loss`` with autograd through ``params``.
    """
    base, dom = resolve_fields(problem, None, shape, progress)
    coords = (dom.physical_coords if physical else dom.coords)(
        device=problem.device, dtype=problem.dtype
    )
    primary = problem.field.primary

    def fn(params: torch.Tensor) -> torch.Tensor:
        out = ansatz(params, coords)
        f = dict(base)
        if isinstance(out, Mapping):
            f.update({k: v.to(problem.dtype) for k, v in out.items()})
        else:
            f[primary] = out.to(problem.dtype)
        total, _ = loss_value(problem, f, dom, progress, terms)
        return total

    return fn


__all__ = [
    "EnergyBarrier",
    "Iter0Gradient",
    "ansatz_objective",
    "center_mass_ratio",
    "energy_barrier",
    "field_gradient",
    "interpolate_fields",
    "iter0_gradient",
    "uniform_center_mass",
    "uniform_fields",
]
