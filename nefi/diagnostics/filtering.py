"""The filtering view of a parameterized update (NeTMY §4.4, Lemma 2, App. D.6).

A first-order step on the parameters ``θ`` of a field ``x = f_θ`` realizes, to leading order,

    Δx ≈ −η J_θ J_θᵀ ∇_x L = −η G_θ ∇_x L,        J_θ = ∂f_θ/∂θ,

so the raw field-space gradient is *filtered* by the positive semidefinite kernel
``G_θ = J_θ J_θᵀ`` (Eq. 7 / 31–32). For a free grid (``GridField``) ``G_θ`` is the identity and the
raw gradient — including the (P2) center bias — is executed verbatim; for a coordinate MLP with
annealed Fourier features ``G_θ`` is a smooth, spatially coupled low-pass kernel whose bandwidth
grows with the annealing progress (Eq. 36–39).

* :func:`filter_kernel_row` — one row ``G_θ e_i`` (vjp then jvp through the field module);
* :func:`kernel_spread` — its half-maximum width in pixels (1 for a grid);
* :func:`realized_update` — ``|Δx|`` after one real optimizer step vs ``|∇_x L|`` with both
  center/outer ratios (NeTMY App. E.5: 1.6× realized vs 18.29× raw);
* :func:`effective_bandwidth` — ``B_β`` of the annealed encoding (Eq. 37).
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.func import functional_call

from ..errors import ConfigError
from ..solve.curriculum import OptimConfig, Stage
from ._common import build_context, resolve_pixel, select_losses, stage_domain
from .landscape import field_gradient
from .sensitivity import center_to_outer_ratio

log = logging.getLogger("nefi")


def _field_fn(module: torch.nn.Module, coords: torch.Tensor, progress: float, name: str):
    trainable = {k: p.detach() for k, p in module.named_parameters() if p.requires_grad}
    frozen = {k: p.detach() for k, p in module.named_parameters() if not p.requires_grad}
    buffers = dict(module.named_buffers())

    def f(params: dict[str, torch.Tensor]) -> torch.Tensor:
        out = functional_call(module, ({**frozen, **params}, buffers), (coords, progress))
        return out[name]

    return f, trainable


def filter_kernel_row(
    problem: Any,
    pixel: int | Sequence[int] | str = "center",
    shape: Sequence[int] | None = None,
    *,
    field: str | None = None,
    progress: float = 1.0,
    module: torch.nn.Module | None = None,
) -> torch.Tensor:
    """One row of the parameterization filter ``G_θ = J_θ J_θᵀ`` (NeTMY Lemma 2, Eq. 32–34).

    ``G_θ e_i`` is the image-space update realized by a unit field-space gradient at pixel ``i``
    under a vanilla gradient step on ``θ``. It is computed matrix-free as a vector-Jacobian
    product (``J_θᵀ e_i``, one backward pass) followed by a Jacobian-vector product (forward mode)
    through the field module, falling back to double-backward autograd when forward mode is not
    available.

    For a :class:`~nefi.fields.GridField` at its native resolution the row is a delta (identity
    kernel, scaled by the squared head derivative); for a :class:`~nefi.fields.NeuralField` it is
    a smooth bump whose width shrinks as ``progress`` (the annealing β/K) grows.

    Args:
        problem: the inverse problem (its ``field`` module is analyzed unless ``module`` is given).
        pixel: ``"center"``, a flat index, or a multi-index.
        shape: grid shape (default native).
        field: output field (default primary).
        progress: annealing progress passed to the field module.
        module: analyze this field module instead of ``problem.field``.

    Returns:
        Tensor shaped like the field.
    """
    mod = module if module is not None else problem.field
    dom = stage_domain(problem, shape)
    try:
        p0 = next(mod.parameters())
        device, dtype = p0.device, p0.dtype
    except StopIteration as e:
        raise ConfigError("the field module has no parameters") from e
    coords = dom.coords(device=device, dtype=dtype)
    name = field or mod.primary
    idx = resolve_pixel(pixel, dom.shape)
    f, params = _field_fn(mod, coords, progress, name)
    try:
        out, vjp_fn = torch.func.vjp(f, params)
        e = torch.zeros_like(out)
        e[idx] = 1.0
        (ct,) = vjp_fn(e)
        _, row = torch.func.jvp(f, (params,), (ct,))
        return row.detach()
    except Exception as err:  # modules without functorch support
        log.debug("functorch path failed for filter_kernel_row (%s); using autograd", err)
    plist = [p for p in mod.parameters() if p.requires_grad]
    with torch.enable_grad():
        out = mod(coords, progress)[name]
        ct = torch.autograd.grad(out[idx], plist, allow_unused=True)
        ct = [torch.zeros_like(p) if c is None else c for p, c in zip(plist, ct)]
        u = torch.zeros_like(out, requires_grad=True)
        g = torch.autograd.grad(out, plist, grad_outputs=u, create_graph=True, allow_unused=True)
        pairs = [(gi, ci) for gi, ci in zip(g, ct) if gi is not None]
        (row,) = torch.autograd.grad(
            [gi for gi, _ in pairs], u, grad_outputs=[ci for _, ci in pairs], allow_unused=True
        )
    return torch.zeros_like(out).detach() if row is None else row.detach()


def kernel_spread(
    row: torch.Tensor,
    level: float = 0.5,
    pixel: int | Sequence[int] | str | None = None,
    connected: bool = True,
) -> float:
    """Width of a filter-kernel row ``G_θ e_i`` in pixels (half-maximum width of its main lobe).

    Pixels with ``|row| ≥ level · |row[pixel]|`` are counted — only those connected to ``pixel``
    when ``connected`` (the main lobe; far side-lobes of a high-bandwidth kernel are ignored) —
    and ``count ** (1 / ndim)`` is returned (the full width at half maximum in 1-D). A
    :class:`~nefi.fields.GridField` gives exactly 1 (a delta); a neural field gives more.

    Args:
        row: kernel row (field-shaped).
        level: threshold relative to the value at ``pixel``.
        pixel: the pixel the row belongs to (default: the location of ``max |row|``).
        connected: restrict to the connected region containing ``pixel``.
    """
    a = row.detach().abs().double().cpu()
    idx = resolve_pixel(int(torch.argmax(a)) if pixel is None else pixel, a.shape)
    ref = float(a[idx])
    if ref == 0.0:
        return 0.0
    mask = (a >= level * ref).numpy()
    if connected:
        from scipy import ndimage

        labels, _ = ndimage.label(mask)
        mask = labels == labels[idx]
    return float(mask.sum()) ** (1.0 / max(1, a.ndim))


@dataclass
class RealizedUpdate:
    """Realized first-step field update vs the raw field-space gradient (NeTMY App. E.5).

    Attributes:
        delta: ``x₁ − x₀`` after one optimizer step (field-shaped).
        raw_grad: ``∇_x L`` at ``x₀`` (what a free-pixel solver would execute).
        delta_ratio: center/outer ratio of ``|delta|`` (NeTMY: ≈1.6×).
        grad_ratio: center/outer ratio of ``|raw_grad|`` (NeTMY F2: 18.29×).
        damping: ``grad_ratio / delta_ratio`` (NeTMY: ≈11×).
        alignment: cosine between ``delta`` and ``−raw_grad`` (1 = verbatim descent direction).
        optimizer: optimizer used for the step.
        lr: learning rate of the step.
        progress: annealing progress at the step.
        field: analyzed field.
    """

    delta: torch.Tensor
    raw_grad: torch.Tensor
    delta_ratio: float
    grad_ratio: float
    damping: float
    alignment: float
    optimizer: str
    lr: float
    progress: float
    field: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_ratio": self.delta_ratio,
            "grad_ratio": self.grad_ratio,
            "damping": self.damping,
            "alignment": self.alignment,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "progress": self.progress,
            "field": self.field,
        }


def _make_optimizer(name: str, params: list, lr: float, oc: OptimConfig):
    name = name.lower()
    if name == "sgd":  # vanilla first-order step, the setting of Lemma 2
        return torch.optim.SGD(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=oc.weight_decay, betas=oc.betas)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=oc.weight_decay, betas=oc.betas)
    if name == "lbfgs":
        return torch.optim.LBFGS(
            params,
            lr=lr,
            history_size=oc.lbfgs_history,
            max_iter=oc.lbfgs_max_iter,
            line_search_fn="strong_wolfe",
        )
    raise ConfigError(f"unknown optimizer {name!r} (sgd | adam | adamw | lbfgs)")


def realized_update(
    problem: Any,
    curriculum_stage: Stage | int | None = None,
    *,
    shape: Sequence[int] | None = None,
    optimizer: str | None = None,
    lr: float | None = None,
    progress: float | None = None,
    terms: Any = None,
    field: str | None = None,
    center_radius: float = 0.2,
    ring: tuple[float, float] = (0.8, 1.0),
    axes: Sequence[int] | None = None,
) -> RealizedUpdate:
    """``|Δx|`` after one optimizer step vs the raw gradient ``|∇_x L|`` (NeTMY App. E.5, Fig. 10).

    The field module is deep-copied, so the problem is left untouched. The step mirrors the
    :class:`~nefi.solve.Solver`: stage resolution, loss-weight overrides, learning rate and
    annealing progress at step 0, the curriculum's optimizer and gradient clipping.

    Args:
        problem: the inverse problem.
        curriculum_stage: a :class:`Stage`, an index into ``problem.curriculum.stages``, or
            ``None`` for the first stage (or a default single stage).
        shape: override the stage resolution.
        optimizer: ``None`` = the curriculum's optimizer (paper setting, AdamW), or ``"sgd"``
            for the vanilla step analyzed by Lemma 2 (``"adam"``, ``"adamw"``, ``"lbfgs"``).
        lr: override the step size (default: ``stage.lr_at(0)``).
        progress: override the annealing progress (default: ``stage.progress_at(0)``).
        terms: loss terms (default: all active terms, as optimized).
        field: analyzed field (default primary).
        center_radius: see :func:`~nefi.diagnostics.center_to_outer_ratio`.
        ring: see :func:`~nefi.diagnostics.center_to_outer_ratio`.
        axes: axes defining the radius (default all).
    """
    cur = getattr(problem, "curriculum", None)
    if curriculum_stage is None:
        stage = cur.stages[0] if cur is not None else Stage(steps=1)
    elif isinstance(curriculum_stage, int):
        if cur is None:
            raise ConfigError("an integer curriculum_stage needs problem.curriculum")
        stage = cur.stages[curriculum_stage]
    else:
        stage = curriculum_stage
    oc = cur.optim if cur is not None else OptimConfig()
    dom = stage_domain(problem, shape if shape is not None else stage.shape)
    prog = stage.progress_at(0) if progress is None else float(progress)
    step_lr = stage.lr_at(0) if lr is None else float(lr)
    opt_name = (optimizer or oc.optimizer).lower()

    device, dtype = problem.device, problem.dtype
    # The first update is tiny relative to the field (a grid moves by lr·σ'²·g ≈ 1e-8), below
    # float32 resolution of the parameters themselves; step a float64 copy of the field so the
    # measured Δx is the intended first-order update. Operator and losses keep the problem dtype.
    work = torch.float64 if device.type != "mps" else dtype
    fmod = copy.deepcopy(problem.field)
    fmod.on_stage_start(stage, dom)
    fmod.to(device=device, dtype=work)
    coords = dom.coords(device=device, dtype=work)
    name = field or fmod.primary
    losses = select_losses(problem.losses, terms, stage)
    op = problem.operator.at_resolution(dom.shape)
    params = [p for p in fmod.parameters() if p.requires_grad]
    if not params:
        raise ConfigError("the field module has no trainable parameters")

    def objective() -> tuple[torch.Tensor, dict]:
        wfields = fmod(coords, prog)
        fields = {k: v.to(dtype) for k, v in wfields.items()}
        ctx = build_context(problem, fields, dom, prog, stage, op=op, field_module=fmod)
        total, _ = losses(ctx)
        return total, wfields

    with torch.enable_grad():
        total, fields0 = objective()
        grads = torch.autograd.grad(total, params, allow_unused=True)
    x0 = {k: v.detach() for k, v in fields0.items()}
    x0_problem = {k: v.to(dtype) for k, v in x0.items()}
    raw = field_gradient(problem, x0_problem, dom.shape, progress=prog, terms=terms, stage=stage)
    raw = raw[name]

    opt = _make_optimizer(opt_name, params, step_lr, oc)
    if isinstance(opt, torch.optim.LBFGS):

        def closure():
            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                t, _ = objective()
                t.backward()
            return t

        opt.step(closure)
    else:
        for p, g in zip(params, grads):
            p.grad = torch.zeros_like(p) if g is None else g.detach().clone()
        if oc.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, oc.grad_clip)
        opt.step()
    with torch.no_grad():
        delta = (fmod(coords, prog)[name] - x0[name]).to(dtype)
    d_ratio = center_to_outer_ratio(delta, center_radius, ring, axes)
    g_ratio = center_to_outer_ratio(raw, center_radius, ring, axes)
    denom = float(delta.norm() * raw.norm())
    alignment = float(-(delta * raw).sum() / denom) if denom > 0 else float("nan")
    damping = g_ratio / d_ratio if d_ratio and d_ratio == d_ratio else float("nan")
    return RealizedUpdate(
        delta=delta,
        raw_grad=raw,
        delta_ratio=d_ratio,
        grad_ratio=g_ratio,
        damping=damping,
        alignment=alignment,
        optimizer=opt_name,
        lr=step_lr,
        progress=prog,
        field=name,
    )


def effective_bandwidth(field: torch.nn.Module, progress: float = 1.0) -> float | None:
    """Effective bandwidth ``B_β`` of the field's annealed encoding (NeTMY Eq. 36–37).

    Returns the highest active Fourier frequency (cycles per unit normalized coordinate) at
    annealing ``progress``, or ``None`` if the field has no :class:`FourierFeatures` encoding
    (e.g. a grid, whose bandwidth is the grid Nyquist limit).
    """
    enc = getattr(field, "encoding", None)
    fn = getattr(enc, "effective_bandwidth", None)
    return None if fn is None else float(fn(progress))


__all__ = [
    "RealizedUpdate",
    "effective_bandwidth",
    "filter_kernel_row",
    "kernel_spread",
    "realized_update",
]
