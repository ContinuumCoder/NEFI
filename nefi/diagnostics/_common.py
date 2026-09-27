"""Shared autograd helpers for the diagnostics (internal).

Everything here is operator-agnostic: the diagnostics only ever touch an
:class:`~nefi.problem.InverseProblem` through ``problem.field``, ``problem.operator`` (via
``at_resolution``), ``problem.losses`` and ``problem.measurement_at``. Jacobian-vector products use
forward-mode autodiff (``torch.func.jvp``) when the operator supports it and fall back to the
double-backward trick and finally to central finite differences, so custom ``autograd.Function``
operators (e.g. discrete-adjoint PDE solvers without a forward-mode rule) still work.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from ..domain import Domain
from ..errors import ConfigError, ShapeError
from ..losses.base import Context, LossSet
from ..utils.tensor import resample, shape_tuple

log = logging.getLogger("nefi")

Fields = dict[str, torch.Tensor]
TensorFn = Callable[[torch.Tensor], torch.Tensor]

JVP_MODES = ("auto", "forward", "double_backward", "finite_difference")


# ------------------------------------------------------------------------------------------
# problem / field plumbing
# ------------------------------------------------------------------------------------------
def stage_domain(problem: Any, shape: Sequence[int] | None = None) -> Domain:
    """The problem's domain at ``shape`` (native resolution when ``None``)."""
    return problem.domain if shape is None else problem.domain.at(shape_tuple(shape))


def current_fields(
    problem: Any, shape: Sequence[int] | None = None, progress: float = 1.0
) -> Fields:
    """Detached field values produced by the problem's current parameters at ``shape``."""
    fields, _ = problem.evaluate(shape, progress=progress, grad=False)
    return {k: v.detach() for k, v in fields.items()}


def resolve_fields(
    problem: Any,
    fields: Any = None,
    shape: Sequence[int] | None = None,
    progress: float = 1.0,
) -> tuple[Fields, Domain]:
    """Normalize a field specification into a complete ``{name: Tensor}`` dict at one resolution.

    Args:
        problem: the inverse problem.
        fields: ``None`` (current field values), a tensor (the primary field; other fields are taken
            from the current values), a mapping ``name -> tensor`` (missing names are filled with
            current values, unknown names are ignored), or a :class:`~nefi.solve.result.Result`.
        shape: grid shape; inferred from the given tensors when ``None``.
        progress: annealing progress used to evaluate the current field values.

    Returns:
        ``(fields, domain)`` on the problem's device/dtype, all sampled on ``domain.shape``.
    """
    if fields is not None and not torch.is_tensor(fields) and not isinstance(fields, Mapping):
        if hasattr(fields, "fields"):  # a Result
            fields = fields.fields
        else:
            raise ConfigError(f"cannot interpret {type(fields).__name__} as fields")
    names = tuple(problem.field.names)
    if shape is None and fields is not None:
        if torch.is_tensor(fields):
            shape = tuple(fields.shape)
        else:
            ref = next((v for k, v in fields.items() if k in names), None)
            shape = None if ref is None else tuple(torch.as_tensor(ref).shape)
        if shape is not None and len(shape) != problem.domain.ndim:
            raise ShapeError(
                f"field tensor of shape {shape} does not match the {problem.domain.ndim}-D domain"
            )
    dom = stage_domain(problem, shape)
    base = current_fields(problem, dom.shape, progress)
    if fields is None:
        return base, dom
    given = {problem.field.primary: fields} if torch.is_tensor(fields) else dict(fields)
    out = dict(base)
    for k, v in given.items():
        if k not in base:
            continue
        t = torch.as_tensor(v).detach().to(device=problem.device, dtype=problem.dtype)
        if tuple(t.shape) != tuple(dom.shape):
            t = resample(t, dom.shape)
        out[k] = t
    return out, dom


def select_losses(losses: LossSet, terms: Any = None, stage: Any = None) -> LossSet:
    """Restrict a :class:`LossSet` to a subset of terms (sharing the loss modules).

    Args:
        losses: the problem's loss set.
        terms: ``None``/``"all"`` (every active term), ``"data"`` (data-fidelity terms only), a
            single term name, or a sequence of names.
        stage: optional curriculum stage whose ``loss_weights`` overrides are applied first.
    """
    ls = losses.with_weights(getattr(stage, "loss_weights", None)) if stage is not None else losses
    if terms is None or terms == "all":
        return ls
    if terms == "data":
        keep = set(ls.data_terms())
        if not keep:
            raise ConfigError("the loss set has no data-fidelity term (Loss.is_data)")
    elif isinstance(terms, str):
        keep = {terms}
    else:
        keep = set(terms)
    unknown = keep - set(ls.names)
    if unknown:
        raise ConfigError(f"unknown loss terms {sorted(unknown)}; known: {ls.names}")
    if not any(ls.weights[k] != 0.0 for k in keep):
        raise ConfigError(f"all selected loss terms {sorted(keep)} have zero weight")
    return ls.with_weights({k: 0.0 for k in ls.names if k not in keep})


def build_context(
    problem: Any,
    fields: Fields,
    dom: Domain,
    progress: float = 1.0,
    stage: Any = None,
    op: Any = None,
    field_module: Any = None,
) -> Context:
    """A loss :class:`Context` for explicit field tensors (seen by the losses as ``ctx.fields``)."""
    op = op if op is not None else problem.operator.at_resolution(dom.shape)
    pred = op(fields)
    obs = problem.measurement_at(dom.shape).to(problem.device, problem.dtype)
    fm = field_module if field_module is not None else problem.field
    return Context(fields, pred, obs, dom, op, fm, stage, 0, progress)


def loss_value(
    problem: Any,
    fields: Fields,
    dom: Domain,
    progress: float = 1.0,
    terms: Any = None,
    stage: Any = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total loss and components for explicit field values (autograd enabled)."""
    losses = select_losses(problem.losses, terms, stage)
    ctx = build_context(problem, fields, dom, progress, stage)
    return losses(ctx)


def observation_mask(problem: Any, dom: Domain) -> torch.Tensor | None:
    """The measurement mask at ``dom`` resolution (``None`` if absent / not resolvable)."""
    try:
        m = problem.measurement_at(dom.shape).mask
    except Exception:  # pragma: no cover - exotic downsamplers
        return None
    return None if m is None else m.to(problem.device, problem.dtype)


def operator_fn(
    problem: Any, fields: Fields, name: str, dom: Domain, use_mask: bool = True
) -> tuple[TensorFn, torch.Tensor, torch.Tensor]:
    """``x -> F({..., name: x})`` at ``dom`` resolution, other fields held fixed.

    The output is multiplied by the observation mask when ``use_mask`` (so the diagnostics see the
    Jacobian of what the data term actually compares). The operator is called once eagerly so any
    per-resolution cache is built outside autodiff transforms.

    Returns:
        ``(fn, x0, y0)``: the function, the current value of the field and ``fn(x0)``.
    """
    if name not in fields:
        raise ConfigError(f"no field named {name!r}; available: {tuple(fields)}")
    op = problem.operator.at_resolution(dom.shape)
    base = {k: v.detach() for k, v in fields.items()}
    mask = observation_mask(problem, dom) if use_mask else None

    def raw(x: torch.Tensor) -> torch.Tensor:
        f = dict(base)
        f[name] = x
        return op(f)

    with torch.no_grad():
        y0 = raw(base[name])
    if mask is not None:
        try:
            torch.broadcast_shapes(tuple(mask.shape), tuple(y0.shape))
        except RuntimeError:
            log.debug("measurement mask %s does not broadcast to %s", mask.shape, y0.shape)
            mask = None

    if mask is None:
        return raw, base[name], y0

    def masked(x: torch.Tensor) -> torch.Tensor:
        return raw(x) * mask

    return masked, base[name], y0 * mask


# ------------------------------------------------------------------------------------------
# products with the Jacobian
# ------------------------------------------------------------------------------------------
class VJP:
    """Reusable vector-Jacobian product ``u -> Jᵀu`` of ``fn`` at ``x`` (graph kept alive)."""

    def __init__(self, fn: TensorFn, x: torch.Tensor) -> None:
        with torch.enable_grad():
            self.x = x.detach().requires_grad_(True)
            self.output = fn(self.x)
        if not self.output.requires_grad:
            raise ConfigError("the function output does not depend on its input (no gradient)")

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        (g,) = torch.autograd.grad(
            self.output,
            self.x,
            grad_outputs=u.to(self.output),
            retain_graph=True,
            allow_unused=True,
        )
        return torch.zeros_like(self.x) if g is None else g.detach()


def jvp(
    fn: TensorFn, x: torch.Tensor, v: torch.Tensor, mode: str = "auto"
) -> tuple[torch.Tensor, str]:
    """Jacobian-vector product ``J v`` of ``fn`` at ``x``.

    Args:
        fn: tensor function.
        x: linearization point.
        v: tangent, same shape as ``x``.
        mode: ``"forward"`` (``torch.func.jvp``), ``"double_backward"`` (differentiate the VJP
            with respect to its cotangent), ``"finite_difference"`` (central differences), or
            ``"auto"`` (try them in that order).

    Returns:
        ``(Jv, mode_used)``; pass ``mode_used`` back in to skip failing attempts in loops.
    """
    if mode not in JVP_MODES:
        raise ConfigError(f"unknown jvp mode {mode!r}; choose from {JVP_MODES}")
    if mode in ("auto", "forward"):
        try:
            _, out = torch.func.jvp(fn, (x.detach(),), (v.to(x),))
            return out.detach(), "forward"
        except Exception as e:  # ops without a forward-mode rule
            if mode == "forward":
                raise
            log.debug("forward-mode jvp unavailable (%s); trying double-backward", e)
    if mode in ("auto", "double_backward"):
        try:
            return _jvp_double_backward(fn, x, v), "double_backward"
        except Exception as e:
            if mode == "double_backward":
                raise
            log.warning(
                "jvp: forward-mode and double-backward both unavailable (%s); using central "
                "finite differences (approximate for nonlinear operators)",
                e,
            )
    return _jvp_finite_difference(fn, x, v), "finite_difference"


def _jvp_double_backward(fn: TensorFn, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        xx = x.detach().requires_grad_(True)
        y = fn(xx)
        u = torch.zeros_like(y, requires_grad=True)
        (g,) = torch.autograd.grad(y, xx, grad_outputs=u, create_graph=True)
        (jv,) = torch.autograd.grad(g, u, grad_outputs=v.to(g), allow_unused=True)
    return torch.zeros_like(y).detach() if jv is None else jv.detach()


def _jvp_finite_difference(fn: TensorFn, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        vmax = float(v.abs().max())
        if vmax == 0.0:
            return torch.zeros_like(fn(x))
        eps = torch.finfo(x.dtype).eps ** (1.0 / 3.0) * max(1.0, float(x.abs().max())) / vmax
        return (fn(x + eps * v) - fn(x - eps * v)) / (2.0 * eps)


# ------------------------------------------------------------------------------------------
# geometry helpers
# ------------------------------------------------------------------------------------------
def rademacher(
    shape: Sequence[int], generator: torch.Generator, device=None, dtype=None
) -> torch.Tensor:
    """±1 probe tensor (drawn on CPU for reproducibility, then moved)."""
    r = torch.randint(0, 2, tuple(shape), generator=generator, dtype=torch.int64)
    return (2 * r - 1).to(device=device, dtype=dtype or torch.get_default_dtype())


def normalized_radius(
    shape: Sequence[int], axes: Sequence[int] | None = None, device=None, dtype=None
) -> torch.Tensor:
    """Distance of each cell center from the grid center in normalized ``[-1, 1]`` units.

    ``r = 1`` touches the middle of the grid faces (the corners of a square are at ``√2``).
    ``axes`` restricts the radius to a subset of axes (e.g. the lateral axes of a 3-D slab).
    """
    shape = shape_tuple(shape)
    coords = Domain.unit(shape).coords(device=device, dtype=dtype or torch.float32)
    ax = range(len(shape)) if axes is None else [int(a) % len(shape) for a in axes]
    r2 = sum(coords[..., a] ** 2 for a in ax)
    return torch.sqrt(torch.as_tensor(r2))


def resolve_pixel(pixel: Any, shape: Sequence[int]) -> tuple[int, ...]:
    """Multi-index of ``pixel`` (``"center"``, a flat int index, or a tuple) in ``shape``."""
    shape = shape_tuple(shape)
    if isinstance(pixel, str):
        if pixel != "center":
            raise ConfigError(f"pixel must be 'center', an int or a tuple, got {pixel!r}")
        return tuple(s // 2 for s in shape)
    if isinstance(pixel, int):
        n = math.prod(shape)
        if not -n <= pixel < n:
            raise ShapeError(f"flat pixel index {pixel} out of range for shape {shape}")
        idx = []
        rem = pixel % n
        for s in reversed(shape):
            idx.append(rem % s)
            rem //= s
        return tuple(reversed(idx))
    idx = tuple(int(i) for i in pixel)
    if len(idx) != len(shape) or any(not -s <= i < s for i, s in zip(idx, shape)):
        raise ShapeError(f"pixel {idx} is not a valid index into shape {shape}")
    return tuple(i % s for i, s in zip(idx, shape))


def tensor_summary(t: torch.Tensor | None) -> dict[str, float] | None:
    """Small JSON-able summary of a tensor (min / max / mean / abs-mean)."""
    if t is None:
        return None
    t = t.detach().float()
    return {
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "abs_mean": float(t.abs().mean()),
    }
