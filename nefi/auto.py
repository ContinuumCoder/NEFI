"""Adaptive defaults — from *your* differentiable forward model to a ready-to-solve problem.

The "bring your own problem" entry point::

    import nefi, torch

    def forward(x):                                   # any differentiable torch code
        return my_physics(x)

    problem = nefi.from_forward(forward, y, shape=(128, 128),
                                prior="nonnegative + piecewise_constant")
    result = nefi.invert(problem)
    print(nefi.auto.quick_report(result, problem))

Everything :func:`from_forward` decides automatically is recorded in ``problem.meta["auto"]``,
logged at INFO level (``nefi.enable_logging()``) and overridable by a keyword argument:

======================  ========================================================================
decision                how / override
======================  ========================================================================
domain                  isotropic spacing, longest axis = 1 (``extent=``)
representation          ``"neural"`` NeuralField sized by the grid (``representation=``,
                        ``hidden=``, ``depth=``, ``n_octaves=`` …)
head / priors           from the prior DSL (``prior=``)
initial scale           constant field that best explains the data (``prior="nonnegative(init=…)"``)
data loss               from ``noise`` (``"gaussian"`` MSE, ``"poisson"``, ``"robust"`` Huber, σ)
noise level             robust estimate (:func:`estimate_noise`) if unknown (``noise=σ``)
loss weights            balanced: data = 1 at init, each regularizer at its prior strength
                        (``weights={...}``)
curriculum              :func:`auto_curriculum` (``budget=``, ``multiscale=``, ``lr=``)
stopping                Morozov discrepancy τ = 1 when σ is known or estimated
                        (``discrepancy=``)
======================  ========================================================================
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .domain import Domain
from .errors import ConfigError, ShapeError
from .fields.base import Field
from .fields.heads import Heads
from .fields.modifiers import (
    Affine,
    MassNormalized,
    ScaledHead,
    find_head,
    innermost,
    set_head_init_value,
)
from .losses.base import LossSet
from .losses.data import MSE, Huber, PoissonNLL
from .measurement import Measurement
from .operators.base import Operator
from .operators.function import FunctionOperator
from .priors import CombinedPrior, combine
from .problem import InverseProblem
from .solve.curriculum import Curriculum, OptimConfig, Stage
from .utils.device import resolve_device
from .utils.tensor import shape_tuple

log = logging.getLogger("nefi")

#: Base step budgets for a 64² field (scaled by ``(numel / 4096) ** 0.25``, clipped to [0.5, 2]).
BUDGETS: dict[str, int] = {"quick": 300, "default": 1500, "thorough": 5000}
#: Default peak learning rate per representation (Adam/AdamW). For ``"neural"`` this is the
#: value at width 64; wider networks use ``lr · sqrt(64 / hidden)`` (see :func:`default_lr`).
DEFAULT_LR: dict[str, float] = {
    "neural": 1e-2,
    "hash": 1e-2,
    "grid": 1e-1,
    "lowrank": 1e-2,
    "parametric": 1e-2,
    "splat": 1e-2,
    "deep_decoder": 3e-3,
    "custom": 3e-3,
}
_DEFAULT_PRIOR = "nonnegative"
_MAD = 1.482602218505602  # 1 / Φ⁻¹(3/4)


# =============================================================================================
# noise estimation
# =============================================================================================
def _d2(t: torch.Tensor, ax: int) -> torch.Tensor:
    n = t.shape[ax]
    return t.narrow(ax, 0, n - 2) - 2.0 * t.narrow(ax, 1, n - 2) + t.narrow(ax, 2, n - 2)


def _and3(m: torch.Tensor, ax: int) -> torch.Tensor:
    n = m.shape[ax]
    return m.narrow(ax, 0, n - 2) & m.narrow(ax, 1, n - 2) & m.narrow(ax, 2, n - 2)


def _diff_k(y: torch.Tensor, m: torch.Tensor, ax: int, k: int):
    """Order-``k`` forward differences along ``ax`` and their complete-stencil mask."""
    for _ in range(k):
        n = y.shape[ax]
        y = y.narrow(ax, 1, n - 1) - y.narrow(ax, 0, n - 1)
        m = m.narrow(ax, 1, n - 1) & m.narrow(ax, 0, n - 1)
    return y, m


def _stencil_residuals(
    y: torch.Tensor, m: torch.Tensor, method: str, axes: Sequence[int], order: int
):
    """Residuals (pooled over axes) and noise gain of a difference stencil."""
    if method == "laplacian":
        r, v = y, m
        for ax in list(axes)[-2:]:
            r, v = _d2(r, ax), _and3(v, ax)
        return [r[v]], 6.0
    out = []
    for ax in axes:
        if y.shape[ax] > order:
            r, v = _diff_k(y, m, ax, order)
            out.append(r[v])
    return out, math.sqrt(math.comb(2 * order, order))


def _parse_method(method: str, order: int | None) -> tuple[str, int]:
    if method.startswith("diff") and method[4:].isdigit():
        return "diff", int(method[4:])
    if method == "diff":
        return "diff", int(order or 1)
    return method, int(order or 4)


def estimate_noise(
    measurement: Measurement | torch.Tensor | np.ndarray,
    *,
    axes: Sequence[int] | None = None,
    method: str = "auto",
    order: int | None = None,
    region: torch.Tensor | None = None,
    store: bool = True,
    min_samples: int = 32,
    min_coverage: float = 0.1,
) -> float:
    """Robust estimate of the additive noise standard deviation of a measurement.

    * ``"laplacian"`` (≥ 2-D, Immerkær 1996): convolve two axes with the Laplacian-difference mask
      ``[[1,−2,1],[−2,4,−2],[1,−2,1]]`` (which annihilates locally linear signal; its output has
      std ``6σ`` for white noise) and take the MAD: ``σ = 1.4826 · MAD / 6``.
    * ``"diff"`` with ``order=k``: k-th finite differences along each axis (pooled),
      ``σ = 1.4826 · MAD / sqrt(C(2k, k))``. ``order=1`` is the classic first-difference MAD;
      higher orders annihilate polynomial trends of degree ``k−1`` and are far less biased by
      sampled smooth signals. ``"diff4"`` is shorthand for ``method="diff", order=4``.
    * ``"values"``: MAD of the values themselves — valid when the signal is sparse in the
      measurement domain (e.g. the outer part of k-space, see ``region``).

    ``"auto"`` tries, in order, the Laplacian on the last two axes of length ≥ 8, differences of
    order 4, 2 and 1 along those axes, and finally ``"values"``; a stencil family is accepted when
    it yields at least ``min_samples`` complete (fully observed) stencils covering at least
    ``min_coverage`` of the observed entries — sparse masks otherwise leave only unrepresentative
    stencils (e.g. the fully sampled, signal-dominated k-space center). Masked-out entries
    (``Measurement.mask``) are excluded; complex data are split into real/imag parts. The median
    ignores the minority of stencils dominated by edges or peaks; strongly textured signals still
    bias the estimate upwards. Operators may know better: ``from_forward`` calls
    ``operator.estimate_noise(measurement)`` when defined (e.g. :class:`FourierSampling`).

    Measured accuracy (mean ``|σ̂/σ − 1|``): 2-D smooth images 2 %, 70 % random masks 3 %, pure
    noise 2–6 %; 1-D blurred bumps / spikes / wave traces (n = 32…256): 11 % with order 4 versus
    18 % (order 2) and ×1.4–3.6 over-estimates (order 1) on the sharpest signals.

    Args:
        measurement: a :class:`~nefi.measurement.Measurement` or an array.
        axes: axes to difference along (default: all axes of length ≥ 8).
        method: ``"auto" | "laplacian" | "diff" | "diffK" | "values"``.
        order: difference order for ``method="diff"`` (default 1).
        region: optional boolean tensor (broadcastable to the data) restricting the entries used.
        store: write the estimate into ``measurement.noise_std`` when it is ``None`` (enables
            discrepancy stopping) and flag ``meta["noise_std_estimated"] = True``.
        min_samples: minimum number of complete stencils.
        min_coverage: minimum fraction of observed entries covered by complete stencils.

    Returns:
        The estimated noise std (float, in data units).

    Example::

        clean = torch.sin(torch.linspace(0, 6, 256))
        sigma = nefi.auto.estimate_noise(clean + 0.05 * torch.randn(256))      # ≈ 0.05
    """
    meas = measurement if isinstance(measurement, Measurement) else None
    data = meas.data if meas is not None else torch.as_tensor(measurement)
    y = data.detach().cpu()
    m = None
    if meas is not None and meas.mask is not None:
        m = meas.mask.detach().cpu()
        m = m.real if m.is_complex() else m  # Measurement casts masks to the data dtype
    if y.is_complex():
        y = torch.view_as_real(y)
        if m is not None:
            m = m.unsqueeze(-1)
    y = y.to(torch.float64)
    m = torch.ones_like(y, dtype=torch.bool) if m is None else (m.expand_as(y) > 0.5)
    if region is not None:
        m = m & torch.as_tensor(region).bool().cpu().expand_as(y)
    n_obs = int(m.sum())
    if n_obs == 0:
        raise ConfigError("estimate_noise: no observed entries (mask/region is all zero)")
    if axes is None:
        axes = [a for a in range(y.ndim) if y.shape[a] >= 8]
    axes = [a % y.ndim for a in axes]
    if method == "auto":
        order_list: list[tuple[str, int]] = [("laplacian", 2)] if len(axes) >= 2 else []
        if axes:
            order_list += [("diff", 4), ("diff", 2), ("diff", 1)]
        order_list.append(("values", 0))
    else:
        meth, k = _parse_method(method, order)
        if meth not in ("laplacian", "diff", "values"):
            raise ConfigError(
                f"unknown noise-estimation method {method!r}; use auto|laplacian|diff|diffK|values"
            )
        if meth == "laplacian" and len(axes) < 2:
            raise ConfigError("method='laplacian' needs two axes of length >= 3")
        order_list = [(meth, k)]
    vals, norm, used = None, 1.0, "values"
    for meth, k in order_list:
        if meth == "values":
            vals, norm, used = y[m], 1.0, "values"
            break
        parts, nrm = _stencil_residuals(y, m, meth, axes, k)
        cand = torch.cat(parts) if parts else y.new_zeros(0)
        enough = cand.numel() >= min_samples and cand.numel() >= min_coverage * n_obs
        if enough or (method != "auto" and cand.numel() > 0):
            vals, norm, used = cand, nrm, (meth if meth == "laplacian" else f"diff{k}")
            break
    if method == "auto" and used == "values" and len(order_list) > 1:
        log.warning(
            "estimate_noise: the mask leaves too few complete difference stencils; using the MAD "
            "of the observed values, which assumes a signal that is sparse in the measurement "
            "domain (pass noise=σ if you know it)"
        )
    assert vals is not None
    med = vals.median()
    sigma = float(_MAD * (vals - med).abs().median() / norm)
    if meas is not None and store and meas.noise_std is None:
        meas.noise_std = sigma
        meas.meta["noise_std_estimated"] = True
        meas.meta["noise_estimator"] = used
    return sigma


# =============================================================================================
# curriculum
# =============================================================================================
def default_lr(representation: str = "neural", hidden: int | None = None) -> float:
    """Default peak learning rate: :data:`DEFAULT_LR`, scaled by ``sqrt(64 / hidden)`` for MLPs.

    Calibrated on the BYOP suite (1-D/2-D deblurring, 30 % k-space MRI, 1-D wave inversion) at
    300–1500 steps; the papers' 1e-3 suits their larger networks and 10k-step budgets.

    Example::

        default_lr("neural", hidden=256)      # 5e-3
    """
    lr = DEFAULT_LR.get(representation, DEFAULT_LR["custom"])
    if representation == "neural" and hidden:
        lr *= math.sqrt(64.0 / float(hidden))
    return lr


def budget_steps(budget: str | int, shape: Sequence[int]) -> int:
    """Total optimization steps for a budget name (scaled by field size) or an explicit int.

    Example::

        budget_steps("default", (64, 64))        # 1500
        budget_steps("quick", (128,))            # 150
    """
    if isinstance(budget, bool):
        raise ConfigError("budget must be 'quick', 'default', 'thorough' or an int")
    if isinstance(budget, int | np.integer):
        if budget < 1:
            raise ConfigError("an integer budget must be >= 1 step")
        return int(budget)
    if budget not in BUDGETS:
        raise ConfigError(f"unknown budget {budget!r}; use {sorted(BUDGETS)} or an int (steps)")
    numel = 1
    for s in shape_tuple(shape):
        numel *= s
    factor = min(2.0, max(0.5, (numel / 4096.0) ** 0.25))
    return max(10, int(round(BUDGETS[budget] * factor / 10.0)) * 10)


def auto_curriculum(
    shape: Sequence[int],
    budget: str | int = "default",
    multiscale: bool = True,
    representation: str = "neural",
    *,
    lr: float | None = None,
    noise_known: bool = False,
    hints: Mapping[str, Any] | None = None,
    coarse_fraction: float = 0.3,
    min_size: int = 8,
    anneal_fraction: float | None = None,
) -> Curriculum:
    """A robust default curriculum for a field of grid ``shape``.

    * steps: :func:`budget_steps` (``"quick"`` 300 / ``"default"`` 1500 / ``"thorough"`` 5000 for a
      64² field, scaled by ``(numel/4096)^¼`` within ×[0.5, 2]; or an explicit int);
    * two stages (``shape//2`` for ``coarse_fraction`` of the steps, then native at half the
      learning rate — NeTMY Tab. 6) when ``multiscale`` and every axis stays ≥ ``min_size``;
    * cosine LR to 1 % per stage, frequency/level annealing over the first half of each stage
      (``hints["anneal_fraction"]`` overrides, e.g. 0.8 for level-set sharpening);
    * AdamW, gradient clipping at 1; Morozov discrepancy stopping ``τ = 1`` when ``noise_known``;
      early stopping (patience ≈ 10 % of the steps) for ``budget="thorough"``.

    Args:
        shape: native field grid.
        budget: ``"quick" | "default" | "thorough"`` or an int (total steps).
        multiscale: allow a coarse stage.
        representation: selects the default LR (:data:`DEFAULT_LR`).
        lr: peak learning rate override.
        noise_known: enable discrepancy stopping.
        hints: prior hints (``anneal_fraction``, ``lr_scale``).
        coarse_fraction: fraction of the steps spent on the coarse stage.
        min_size: minimum coarse size per axis.
        anneal_fraction: fraction of each stage over which frequencies / hash levels / level-set
            sharpness are annealed (default: prior hint, else 0.5). Early and discrepancy stops
            are only checked once annealing has finished.

    Example::

        cur = auto_curriculum((64, 64), budget="quick")      # 2 stages, 300 steps
        cur = auto_curriculum((128,), budget=500, multiscale=False)
    """
    shape = shape_tuple(shape)
    hints = dict(hints or {})
    total = budget_steps(budget, shape)
    base_lr = float(lr) if lr is not None else default_lr(representation, hints.get("hidden"))
    base_lr *= float(hints.get("lr_scale", 1.0))
    if anneal_fraction is None:
        anneal_fraction = float(hints.get("anneal_fraction", 0.5))
    coarse = tuple(max(min_size, s // 2) for s in shape)
    two = (
        multiscale
        and representation != "parametric"
        and all(s // 2 >= min_size for s in shape)
        and total >= 40
    )
    common = dict(
        lr_schedule="cosine", lr_min_ratio=0.01, anneal=True, anneal_fraction=anneal_fraction
    )
    if two:
        n1 = max(10, int(round(total * coarse_fraction)))
        stages = [
            Stage("coarse", coarse, n1, base_lr, **common),  # type: ignore[arg-type]
            Stage("fine", shape, max(10, total - n1), base_lr * 0.5, **common),  # type: ignore[arg-type]
        ]
    else:
        stages = [Stage("main", shape, total, base_lr, **common)]  # type: ignore[arg-type]
    patience = None
    if budget == "thorough":
        patience = max(100, total // 10)
    return Curriculum(
        stages,
        optim=OptimConfig(optimizer="adamw", weight_decay=1e-4, grad_clip=1.0),
        discrepancy_tau=1.0 if noise_known else None,
        early_stop_patience=patience,
    )


# =============================================================================================
# probes: learning-rate range test and weight balancing
# =============================================================================================
def _snapshot(problem: InverseProblem) -> tuple[dict, dict]:
    return (
        copy.deepcopy(problem.field.state_dict()),
        copy.deepcopy(problem.operator.state_dict()),
    )


def _restore(problem: InverseProblem, snap: tuple[dict, dict]) -> None:
    problem.field.load_state_dict(snap[0])
    problem.operator.load_state_dict(snap[1])


def _trainable(problem: InverseProblem) -> list[torch.nn.Parameter]:
    params = [p for p in problem.field.parameters() if p.requires_grad]
    params += [p for p in problem.operator.parameters() if p.requires_grad]
    if not params:
        raise ConfigError("the problem has no trainable parameters")
    return params


def _probe_shape(problem: InverseProblem, shape: Sequence[int] | None) -> tuple[int, ...] | None:
    if shape is not None:
        return shape_tuple(shape)
    cur = getattr(problem, "curriculum", None)
    if cur is not None and cur.stages and cur.stages[0].shape is not None:
        return tuple(cur.stages[0].shape)
    return None


def lr_range_test(
    problem: InverseProblem,
    lrs: Sequence[float] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    steps: int = 20,
    *,
    shape: Sequence[int] | None = None,
    progress: float = 1.0,
    grad_clip: float | None = 1.0,
    divergence: float = 2.0,
    safety: float = 1.0,
) -> float:
    """Cheap learning-rate probe: pick the LR with the largest loss decrease without divergence.

    For each candidate, ``steps`` Adam steps are run from the *same* initial parameters on the
    problem's full loss (at ``shape``, default the first curriculum stage); a run diverges if the
    loss becomes non-finite or exceeds ``divergence ×`` its initial value. Parameters are restored
    afterwards. Cost: ``len(lrs) · steps`` optimization steps.

    Returns:
        ``safety ×`` the best learning rate (the smallest candidate if every run failed).

    Example::

        problem = nefi.from_forward(lambda x: x, torch.rand(32), shape=(32,), budget=50)
        lr = nefi.auto.lr_range_test(problem, steps=10)
        problem.curriculum = nefi.auto.auto_curriculum((32,), budget=50, lr=lr)
    """
    shp = _probe_shape(problem, shape)
    snap = _snapshot(problem)
    results = []
    try:
        for lr in lrs:
            _restore(problem, snap)
            params = _trainable(problem)
            opt = torch.optim.Adam(params, lr=float(lr))
            l0, diverged, last = None, False, math.inf
            for _ in range(int(steps)):
                opt.zero_grad(set_to_none=True)
                total, _ = problem.loss(shp, progress)
                v = float(total.detach())
                if l0 is None:
                    l0 = v
                if not math.isfinite(v) or (l0 is not None and v > divergence * abs(l0) + 1e-30):
                    diverged = True
                    break
                total.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()
            if not diverged:
                with torch.no_grad():
                    last = float(problem.loss(shp, progress)[0])
                if not math.isfinite(last) or last > divergence * abs(l0 or 0.0):
                    diverged = True
            dec = (
                (l0 - last) / max(abs(l0), 1e-30)
                if (l0 is not None and not diverged)
                else -math.inf
            )
            results.append((float(lr), dec, diverged))
            log.debug("lr_range_test lr=%.2e decrease=%.3f diverged=%s", lr, dec, diverged)
    finally:
        _restore(problem, snap)
    ok = [(lr, d) for lr, d, bad in results if not bad and d > 0]
    if not ok:
        log.warning("lr_range_test: no candidate decreased the loss; using %.1e", min(lrs))
        return float(min(lrs)) * safety
    best = max(ok, key=lambda t: t[1])[0]
    log.info(
        "lr_range_test: best lr %.2e (%s)",
        best,
        ", ".join(f"{lr:.0e}:{d:+.2f}" for lr, d, _ in results),
    )
    return best * safety


def _data_scale(term, ctx, value: float) -> float:
    """Distance of a data term from its minimum (PoissonNLL is offset by the saturated model)."""
    if isinstance(term, PoissonNLL):
        y = ctx.obs.data
        sat = torch.where(y > 0, y - y * torch.log(y.clamp_min(term.eps)), torch.zeros_like(y))
        return value - float(ctx.obs.masked_mean(sat))
    return value


def balance_weights(
    problem: InverseProblem,
    strengths: Mapping[str, float] | None = None,
    *,
    probe_steps: int = 25,
    lr: float = 1e-3,
    shape: Sequence[int] | None = None,
    target: float = 1.0,
) -> dict[str, float]:
    """Automatic loss weights: data term = ``target`` at the initial iterate, and each regularizer
    worth ``strength × target`` at a *reference* iterate.

    The reference is the larger of the regularizer's value at the initial iterate and after a
    short data-only probe (``probe_steps`` Adam steps, then parameters are restored). The probe
    matters because the initial field is nearly constant — TV and Laplacian are ≈ 0 there, so
    balancing at the initial iterate alone would give absurd weights. With ``probe_steps=0`` this
    reduces to :meth:`LossSet.auto_balance` at the initial iterate.

    Args:
        problem: the problem (its ``losses.weights`` are updated in place).
        strengths: ``{regularizer: relative strength}`` (default: the current weights).
        probe_steps: data-only probe length (0 disables the probe).
        lr: probe learning rate.
        shape: probe resolution (default: first curriculum stage / native).
        target: normalized data-term value at the initial iterate.

    Returns:
        The new weights.

    Example::

        problem = nefi.from_forward(lambda x: x, torch.rand(32), shape=(32,), budget=50,
                                    prior="nonnegative + tv", weights="raw")
        nefi.auto.balance_weights(problem, {"tv": 1e-2})    # {'data': ..., 'tv': ...}
    """
    losses = problem.losses
    shp = _probe_shape(problem, shape)
    data_names = set(losses.data_terms())
    regs = [k for k in losses.names if k not in data_names and losses.weights[k] != 0.0]
    strengths = {k: float((strengths or {}).get(k, losses.weights[k])) for k in regs}
    with torch.no_grad():
        ctx0 = problem.context(shp, 1.0)
        v0 = {k: float(losses.terms[k](ctx0)) for k in losses.names if losses.weights[k] != 0.0}
    new = dict(losses.weights)
    for k in data_names:
        if k not in v0:
            continue
        s = _data_scale(losses.terms[k], ctx0, v0[k])
        new[k] = target / s if (math.isfinite(s) and s > 1e-30) else 1.0
    ref = {k: v0[k] for k in regs}
    if probe_steps > 0 and regs:
        snap = _snapshot(problem)
        try:
            probe = losses.with_weights(
                {k: (new[k] if k in data_names else 0.0) for k in losses.names}
            )
            params = _trainable(problem)
            opt = torch.optim.Adam(params, lr=lr)
            for step in range(int(probe_steps)):
                opt.zero_grad(set_to_none=True)
                ctx = problem.context(shp, min(1.0, (step + 1) / probe_steps))
                total, _ = probe(ctx)
                if not torch.isfinite(total):
                    break
                total.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            with torch.no_grad():
                ctx1 = problem.context(shp, 1.0)
                for k in regs:
                    v1 = float(losses.terms[k](ctx1))
                    if math.isfinite(v1):
                        ref[k] = max(ref[k], v1)
        finally:
            _restore(problem, snap)
    floor = _synthetic_reference(problem, ctx0, regs)
    for k in regs:
        r = max(ref[k], 1e-2 * floor.get(k, 0.0))
        new[k] = strengths[k] * target / r if (math.isfinite(r) and r > 1e-30) else strengths[k]
    losses.weights.update(new)
    return dict(losses.weights)


def _synthetic_reference(problem: InverseProblem, ctx0, regs: Sequence[str]) -> dict[str, float]:
    """Regularizer values on a synthetic structured field of the right amplitude.

    Each field is replaced by ``A · (1 + 0.5 g)`` where ``A`` is its initial mean magnitude and
    ``g`` a smooth unit-variance random field (correlation length ≈ 1/8 of the domain). Used as a
    floor (1 %) for the balancing reference so that a regularizer that happens to vanish at the
    reference iterate cannot receive an absurd weight.
    """
    from .losses.base import Context
    from .utils.tensor import resample

    out: dict[str, float] = {}
    if not regs:
        return out
    gen = torch.Generator().manual_seed(1234)
    fields = {}
    for k, v in ctx0.fields.items():
        v = v.detach()
        shape = tuple(v.shape)
        coarse = tuple(max(2, min(s, 8)) for s in shape)
        g = torch.randn(coarse, generator=gen, dtype=torch.float32)
        g = (
            resample(g, shape, mode="linear")
            if len(shape) <= 3
            else torch.randn(shape, generator=gen)
        )
        g = (g - g.mean()) / g.std().clamp_min(1e-12)
        amp = float(v.abs().mean()) or 1.0
        fields[k] = (amp * (1.0 + 0.5 * g)).to(v)
    ctx = Context(fields, ctx0.pred, ctx0.obs, ctx0.domain, ctx0.operator, ctx0.field_module)
    with torch.no_grad():
        for k in regs:
            try:
                val = float(problem.losses.terms[k](ctx))
            except Exception:  # noqa: BLE001 - some losses need the real context
                continue
            if math.isfinite(val) and val > 0:
                out[k] = val
    return out


# =============================================================================================
# from_forward helpers
# =============================================================================================
def _as_measurement(measurement: Any, mask: Any) -> Measurement:
    if isinstance(measurement, Measurement):
        meas = Measurement(
            measurement.data.clone(),
            None if measurement.mask is None else measurement.mask.clone(),
            measurement.noise_std,
            dict(measurement.meta),
        )
    else:
        meas = Measurement(
            torch.as_tensor(
                np.asarray(measurement) if not torch.is_tensor(measurement) else measurement
            )
        )
    if mask is not None:
        meas.mask = torch.as_tensor(mask).float()
    if meas.data.is_complex():
        meas.data = torch.view_as_real(meas.data).contiguous()
        if meas.mask is not None:
            m = meas.mask.real if meas.mask.is_complex() else meas.mask
            m = torch.broadcast_to(m, meas.data.shape[:-1])
            meas.mask = m.unsqueeze(-1).expand(*m.shape, 2).float().clone()
        meas.meta["complex_as_real"] = True
    if meas.mask is not None and meas.mask.is_complex():
        meas.mask = meas.mask.real.float()
    if not meas.data.is_floating_point():
        meas.data = meas.data.float()
    elif meas.data.dtype == torch.float64:
        meas.data = meas.data.float()
    if meas.mask is not None:
        try:
            torch.broadcast_shapes(tuple(meas.mask.shape), tuple(meas.data.shape))
        except RuntimeError as e:
            raise ShapeError(
                f"mask shape {tuple(meas.mask.shape)} does not broadcast to the data shape "
                f"{tuple(meas.data.shape)}"
            ) from e
    return meas


def _default_extent(shape: tuple[int, ...]) -> tuple[tuple[float, float], ...]:
    m = max(shape)
    return tuple((0.0, s / m) for s in shape)


def _neural_size(shape: tuple[int, ...]) -> tuple[int, int]:
    numel = 1
    for s in shape:
        numel *= s
    if numel <= 1024:
        return 64, 3
    if numel <= 16384:
        return 128, 4
    if numel <= 262144:
        return 256, 5
    return 256, 6


def default_octaves(shape: Sequence[int]) -> int:
    """``clamp(ceil(log2(max(shape))), 4, 12)`` Fourier octaves for a grid.

    Example::

        assert default_octaves((128, 128)) == 7 and default_octaves((8,)) == 4
    """
    return int(min(12, max(4, math.ceil(math.log2(max(2, max(shape)))))))


def _load_baseline_field(kind: str):
    """Lazy import of optional baseline representations (Gaussian splats, Deep Decoder)."""
    from importlib import import_module

    from .registry import RegistryError, get

    names = {
        "splat": ("GaussianSplatField", ("gaussian_splat", "splat", "gaussian_splats")),
        "deep_decoder": ("DeepDecoderField", ("deep_decoder", "deepdecoder")),
    }[kind]
    try:
        mod = import_module(".baselines", __package__)
        if hasattr(mod, names[0]):
            return getattr(mod, names[0])
    except ImportError:
        pass
    for reg in names[1]:
        try:
            return get("field", reg)
        except RegistryError:
            continue
    raise ConfigError(
        f"representation={kind!r} needs nefi.baselines.{names[0]}, which is not available in this "
        "installation; use 'neural', 'hash', 'grid' or 'lowrank', or pass a Field instance"
    )


def _field_builder(
    representation: Any,
    shape: tuple[int, ...],
    heads: Heads,
    hints: Mapping[str, Any],
    field_kw: dict[str, Any],
    decisions: dict[str, Any],
) -> Callable[[int], Field]:
    rep = representation
    kw = dict(field_kw)

    def builder(nd: int) -> Field:
        grid = shape if nd == len(shape) else (max(shape),) * nd
        if isinstance(rep, Field):
            decisions["representation"] = f"user Field ({type(rep).__name__})"
            return rep
        if callable(rep) and not isinstance(rep, str):
            decisions["representation"] = f"user builder ({getattr(rep, '__name__', rep)})"
            return rep(nd, heads, **kw)
        if rep == "neural":
            hidden, depth = _neural_size(grid)
            octaves = default_octaves(grid) + int(hints.get("octave_offset", 0))
            args: dict[str, Any] = dict(
                hidden=hidden,
                depth=depth,
                skip_at=depth // 2 if depth >= 4 else None,
                n_octaves=max(2, octaves),
                activation="tanh",
                include_input=bool(hints.get("include_input", True)),
            )
            args.update(kw)
            decisions["representation"] = (
                f"neural (hidden={args['hidden']}, depth={args['depth']}, "
                f"n_octaves={args['n_octaves']}, activation={args['activation']})"
            )
            from .fields.neural import NeuralField

            return NeuralField(nd, heads, **args)
        if rep == "hash":
            from .fields.hashgrid import HashGridField

            numel = 1
            for s in grid:
                numel *= s
            max_res = max(grid)
            base = 4 if max_res >= 32 else 2
            levels = int(min(16, max(4, math.ceil(math.log2(max_res / base)) * 2 + 1)))
            args = dict(
                hidden=64,
                depth=2,
                n_levels=levels,
                n_features=2,
                log2_table_size=int(min(19, max(10, math.ceil(math.log2(numel)) + 1))),
                base_resolution=base,
                max_resolution=max_res,
            )
            args.update(kw)
            decisions["representation"] = (
                f"hash (levels={args['n_levels']}, T=2^{args['log2_table_size']}, "
                f"res={args['base_resolution']}..{args['max_resolution']}, "
                f"mlp={args['hidden']}x{args['depth']})"
            )
            return HashGridField(nd, heads, **args)
        if rep == "grid":
            from .fields.grid import GridField

            decisions["representation"] = f"grid {grid}"
            return GridField(grid, heads, **kw)
        if rep == "lowrank":
            from .fields.lowrank import LowRankField

            args = dict(rank=8)
            args.update(kw)
            decisions["representation"] = f"lowrank (rank={args['rank']})"
            return LowRankField(grid, heads, **args)
        if rep == "parametric":
            from .fields.parametric import ParametricField

            if "fn" not in kw:
                raise ConfigError(
                    "representation='parametric' needs fn=... "
                    "(e.g. fn=nefi.fields.gaussian_blobs(2))"
                )
            args = dict(kw)
            fn = args.pop("fn")
            decisions["representation"] = f"parametric ({type(fn).__name__})"
            return ParametricField(fn, heads=heads, **args)
        if rep in ("splat", "deep_decoder"):
            cls = _load_baseline_field(rep)
            decisions["representation"] = f"{rep} ({cls.__name__})"
            # GaussianSplatField(ndim, heads, ...); DeepDecoderField(shape, heads, ...)
            return cls(nd, heads, **kw) if rep == "splat" else cls(grid, heads, **kw)
        raise ConfigError(
            f"unknown representation {rep!r}; use 'neural', 'hash', 'grid', 'lowrank', "
            "'parametric', 'splat', 'deep_decoder', a Field instance or a builder callable"
        )

    return builder


def _estimate(meas: Measurement, op: Operator | None, store: bool = True) -> float:
    """Noise estimate, preferring an operator-specific estimator when the operator has one."""
    hook = getattr(op, "estimate_noise", None)
    if callable(hook):
        sigma = hook(meas)
        if sigma is not None and math.isfinite(float(sigma)):
            if store and meas.noise_std is None:
                meas.noise_std = float(sigma)
                meas.meta["noise_std_estimated"] = True
                meas.meta["noise_estimator"] = f"{type(op).__name__}.estimate_noise"
            return float(sigma)
    return estimate_noise(meas, store=store)


def _data_loss(
    noise: Any, meas: Measurement, decisions: dict[str, Any], op: Operator | None = None
):
    """Data-fidelity term from the noise model; returns ``(loss, sigma_known_or_estimated)``."""
    if isinstance(noise, bool):
        raise ConfigError("noise must be 'auto', 'gaussian', 'poisson', 'robust' or a float σ")
    if isinstance(noise, int | float):
        if not noise >= 0:
            raise ConfigError(f"noise std must be >= 0, got {noise}")
        meas.noise_std = float(noise)
        decisions["noise"] = f"gaussian, σ = {float(noise):.3g} (given)"
        return MSE(), float(noise) > 0
    key = str(noise).lower()
    if key == "auto":
        if meas.noise_std is None:
            sigma = _estimate(meas, op)
            decisions["noise"] = (
                f"gaussian, σ ≈ {sigma:.3g} (estimated, {meas.meta.get('noise_estimator')})"
            )
        else:
            ns = meas.noise_std
            ns = float(ns.mean()) if torch.is_tensor(ns) else float(ns)
            decisions["noise"] = f"gaussian, σ = {ns:.3g} (from the measurement)"
        return MSE(), True
    if key in ("gaussian", "mse", "l2"):
        decisions["noise"] = "gaussian (MSE), σ not used"
        return MSE(), meas.noise_std is not None
    if key == "poisson":
        decisions["noise"] = (
            "poisson (PoissonNLL); predictions must be > 0 (use a non-negative prior)"
        )
        return PoissonNLL(), False
    if key in ("robust", "huber"):
        sigma = (
            _estimate(meas, op, store=False)
            if meas.noise_std is None
            else float(torch.as_tensor(meas.noise_std).float().mean())
        )
        delta = max(1.345 * sigma, 1e-12)
        decisions["noise"] = f"robust (Huber, δ = 1.345σ = {delta:.3g})"
        return Huber(delta=delta), False
    raise ConfigError(
        f"unknown noise model {noise!r}; use 'auto', 'gaussian', 'poisson', 'robust' or σ"
    )


def _validate(problem: InverseProblem) -> None:
    """Dry-run the forward on the initial field: shape, finiteness and differentiability."""
    dom = problem.domain
    coords = dom.coords(device=problem.device, dtype=problem.dtype)
    field_desc = None
    try:
        with torch.no_grad():
            fields = problem.field(coords, 1.0)
            field_desc = {k: tuple(v.shape) for k, v in fields.items()}
            pred = problem.operator(fields)
    except Exception as e:  # noqa: BLE001 - re-raised with guidance
        raise ConfigError(
            f"calling the forward model on the initial field failed with {type(e).__name__}: {e}. "
            f"The forward receives fields {field_desc or '(field evaluation failed)'} on device "
            f"{problem.device} (dtype {problem.dtype}); if it closes over tensors, create them on "
            "that device or wrap the forward in an nn.Module."
        ) from e
    if not torch.is_tensor(pred):
        raise ConfigError(f"the forward returned {type(pred).__name__}, expected a torch.Tensor")
    meas_shape = tuple(problem.measurement.data.shape)
    if tuple(pred.shape) != meas_shape:
        raise ShapeError(
            f"the forward returned shape {tuple(pred.shape)} but the measurement has shape "
            f"{meas_shape}. Make the forward return exactly the measured quantity (select the "
            "observed entries / sensors / time steps), or reshape the measurement to match."
        )
    if not torch.isfinite(pred).all():
        raise ConfigError(
            "the forward returned non-finite values (NaN/inf) on the initial field; check for "
            "divisions by zero, logs of non-positive values or overflowing exponentials"
        )
    fields = problem.field(coords, 1.0)
    pred = problem.operator(fields)
    params = [p for p in problem.field.parameters() if p.requires_grad]
    if not pred.requires_grad:
        raise ConfigError(
            "the forward output does not depend differentiably on the unknown field (no autograd "
            "graph). Use torch operations end-to-end: no .detach(), .numpy(), .item(), "
            "torch.no_grad() or in-place writes into leaf tensors inside the forward."
        )
    grads = torch.autograd.grad((pred * torch.randn_like(pred)).sum(), params, allow_unused=True)
    if all(g is None or not bool(g.abs().max() > 0) for g in grads):
        raise ConfigError(
            "the gradient of the forward w.r.t. the field parameters is identically zero; the "
            "forward may ignore its input or use non-differentiable ops (argmax, rounding, "
            "comparisons)"
        )


def _masked_mse(pred: torch.Tensor, meas: Measurement) -> float:
    y = meas.data.to(pred)
    r = (pred - y) ** 2
    if meas.mask is not None:
        m = meas.mask.to(pred)
        return float((r * m).sum() / m.expand_as(r).sum().clamp_min(1.0))
    return float(r.mean())


def _auto_init(
    problem: InverseProblem,
    name: str,
    seed: int,
    decisions: dict[str, Any],
    headroom: float = 1.0,
) -> None:
    """Fit the initial constant level of the field so that it best explains the data (MSE).

    The head's natural scale is set to ``headroom ×`` that level (the inner head starting at
    ``1/headroom``), so the initial field equals the fitted level either way.
    """
    head = problem.field.heads[name]
    if find_head(head, MassNormalized) is not None:
        decisions["init"] = "fixed by conservation (mass-normalized head)"
        return
    sc = find_head(head, ScaledHead)
    aff = innermost(head) if isinstance(innermost(head), Affine) else None
    if sc is None and aff is None:
        return
    meas = problem.measurement

    def setv(v: float) -> None:
        if sc is not None:
            sc.scale = float(v) * headroom
            set_head_init_value(sc.inner, 1.0 / headroom)
        else:
            aff.scale = abs(float(v)) if v != 0 else 1.0  # type: ignore[union-attr]
            aff.init_value = float(v)  # type: ignore[union-attr]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            problem.field.reset_parameters()
        problem.field.to(problem.device)

    def loss_at(v: float) -> float:
        setv(v)
        try:
            with torch.no_grad():
                _, pred = problem.evaluate()
        except Exception:  # noqa: BLE001 - overflow in exotic forwards at extreme scales
            return math.inf
        if not torch.isfinite(pred).all():
            return math.inf
        return _masked_mse(pred, meas)

    signed = sc is None
    p = problem.operator.homogeneity
    cands: list[float] = []
    if p:
        setv(1.0)
        with torch.no_grad():
            _, a = problem.evaluate()
        y = meas.data.to(a)
        m = meas.mask.to(a).expand_as(a) if meas.mask is not None else torch.ones_like(a)
        num, den = float((a * y * m).sum()), float((a * a * m).sum())
        if den > 0:
            c = num / den
            if signed and p == 1:
                cands.append(c)
            elif c > 0:
                cands.append(c ** (1.0 / p))
    best_v, best_l = 1.0, math.inf
    for v in cands:
        lv = loss_at(v)
        if lv < best_l:
            best_v, best_l = v, lv
    note = ""
    if not cands or not math.isfinite(best_l):
        grid = [10.0**k for k in range(-8, 9)]
        if signed:
            grid = grid + [-g for g in grid]
        losses_at = {}
        for v in grid:
            lv = loss_at(v)
            losses_at[v] = lv
            if lv < best_l:
                best_v, best_l = v, lv
        at_edge = abs(best_v) in (grid[0], grid[len(grid) // (2 if signed else 1) - 1])
        finite = [lv for lv in losses_at.values() if math.isfinite(lv)]
        flat = bool(finite) and (max(finite) - min(finite)) <= 1e-6 * max(abs(min(finite)), 1e-30)
        if at_edge or flat:
            y = meas.data.float()
            ref = float(torch.sqrt((y**2).mean())) or 1.0
            note = (
                " — the constant-field fit was degenerate (optimum at the search boundary or a "
                "flat misfit, e.g. other unknowns dominate the data), so the data RMS is used; "
                "pass init=... in the prior to override"
            )
            log.warning("from_forward: initial level of %r fitted at %.3g%s", name, best_v, note)
            best_v, best_l = ref, loss_at(ref)
        else:
            for f in (10**0.5, 10**-0.5, 10**0.25, 10**-0.25):
                v = best_v * f
                lv = loss_at(v)
                if lv < best_l:
                    best_v, best_l = v, lv
    if not math.isfinite(best_l):
        log.warning("from_forward: could not find a finite initial scale; keeping 1.0")
        best_v = 1.0
    setv(best_v)
    extra = (
        f", head scale {best_v * headroom:.4g} (peak headroom x{headroom:g})"
        if headroom != 1
        else ""
    )
    prev = decisions.get("init")
    msg = f"{name}: level {best_v:.4g} (constant field fitted to the data, MSE {best_l:.3g}){extra}"
    decisions["init"] = f"{prev}; {msg}{note}" if prev else f"{msg}{note}"


def _innermost_field(field: Field) -> Field:
    while hasattr(field, "inner") and isinstance(field.inner, Field):
        field = field.inner
    return field


def _multiscale_decision(
    op: Operator, shape: tuple[int, ...], multiscale: Any, representation: Any
) -> tuple[bool, str]:
    if multiscale is False:
        return False, "disabled (multiscale=False)"
    if representation == "parametric":
        return False, "parametric representation (a few global parameters): single stage"
    if any(s // 2 < 8 for s in shape):
        return False, f"grid {shape} too small for a coarse stage (needs >= 16 per axis)"
    if isinstance(op, FunctionOperator):
        if op.multiscale_capable:
            how = "coarse='upsample'" if op.coarse else "at_resolution/output_shape given"
            return True, f"2 stages ({how})"
        return False, (
            "the forward has no at_resolution/output_shape, so it only runs at the native "
            "resolution: single stage (pass at_resolution=..., output_shape=... or "
            "multiscale='upsample' for coarse-to-fine)"
        )
    coarse = tuple(max(8, s // 2) for s in shape)
    try:
        out = op.output_shape(coarse)
    except Exception:  # noqa: BLE001
        out = None
    if out is None:
        return (
            False,
            f"{type(op).__name__} does not declare output_shape for coarse grids: single stage",
        )
    return True, f"2 stages ({type(op).__name__} supports coarse grids, output {tuple(out)})"


# =============================================================================================
# from_forward
# =============================================================================================
def from_forward(
    forward: Callable | Operator,
    measurement: Measurement | torch.Tensor | np.ndarray,
    shape: Sequence[int] | int,
    *,
    extent: Sequence[tuple[float, float]] | None = None,
    prior: Any = _DEFAULT_PRIOR,
    representation: Any = "neural",
    noise: Any = "auto",
    homogeneity: float | None = None,
    output_shape: Callable | Sequence[int] | None = None,
    at_resolution: Callable | None = None,
    field_name: str = "x",
    weights: Any = "auto",
    budget: str | int = "default",
    name: str = "custom",
    takes_dict: bool = False,
    lr: float | str | None = None,
    multiscale: Any = "auto",
    mask: Any = None,
    discrepancy: Any = "auto",
    probe_steps: int = 25,
    anneal_fraction: float | None = None,
    device: str | torch.device = "auto",
    seed: int = 0,
    **field_kw: Any,
) -> InverseProblem:
    """Build a complete :class:`~nefi.problem.InverseProblem` from a differentiable forward model.

    Args:
        forward: ``fn(x) -> y`` on the unknown ``x`` of shape ``shape`` (any differentiable torch
            code), ``fn(fields) -> y`` with ``takes_dict=True`` (several unknowns, see ``prior``),
            or an existing :class:`~nefi.operators.Operator` (used as is; its ``homogeneity`` is
            set from ``homogeneity=`` only if it was ``None``).
        measurement: observed data (tensor / array / :class:`~nefi.measurement.Measurement`);
            complex data are converted to stacked real/imag parts.
        shape: grid of the unknown field.
        extent: physical ``(lo, hi)`` per axis (default: isotropic, longest axis = 1).
        prior: physical knowledge as a DSL string / :class:`~nefi.priors.Prior` / list (see
            :mod:`nefi.priors`), or ``{field_name: spec}`` for several unknowns (one network,
            several heads; the forward then receives the dict).
        representation: ``"neural"`` (default), ``"hash"``, ``"grid"``, ``"lowrank"``,
            ``"parametric"`` (with ``fn=...``), ``"splat"``/``"deep_decoder"`` (from
            ``nefi.baselines``), a :class:`~nefi.fields.Field`, or a builder
            ``(ndim, heads, **kw) -> Field``. With the default prior, ``"parametric"`` and
            ``"splat"`` use an identity head (their parameterization already fixes the values).
        noise: ``"auto"`` (MSE + :func:`estimate_noise` when σ is unknown), ``"gaussian"``,
            ``"poisson"``, ``"robust"`` (Huber) or a float σ.
        homogeneity: degree of homogeneity of ``forward`` in the field (1 = linear) — enables an
            exact initial-scale fit and scale correction.
        output_shape, at_resolution: multiscale support for a plain function (see
            :class:`~nefi.operators.function.FunctionOperator`).
        field_name: name of the (primary) unknown.
        weights: ``"auto"`` (:func:`balance_weights`), ``"raw"`` (prior strengths used as
            absolute weights) or ``{loss_name: weight}`` (absolute; unlisted terms use their
            prior strength as an absolute weight; the data term defaults to 1).
        budget: ``"quick" | "default" | "thorough"`` or total steps (:func:`auto_curriculum`).
        name: problem label.
        takes_dict: pass the fields dict to ``forward``.
        lr: peak learning rate, ``"auto"`` (:func:`lr_range_test`) or ``None`` (per
            representation, :data:`DEFAULT_LR`).
        multiscale: ``"auto"``, ``True``/``"upsample"`` (force two stages; plain functions are
            fed upsampled coarse fields) or ``False``.
        mask: observed-entry mask for the measurement (1 = observed).
        discrepancy: ``"auto"`` (τ = 1 when σ is known or estimated and the noise model is
            Gaussian), ``True`` (τ = 1, requires σ), ``False`` or a float τ.
        probe_steps: length of the data-only probe used by weight balancing.
        anneal_fraction: annealing fraction of each stage (see :func:`auto_curriculum`).
        device: device for the setup (dry run, initial-scale fit, probes): ``"auto"`` = the
            solver's default (``NEFI_DEVICE``, else CUDA if available, else CPU), so device
            problems in the forward surface here; the solver moves the problem as needed.
        seed: seed for the field initialization and probes.
        **field_kw: forwarded to the representation (``hidden``, ``depth``, ``n_octaves``,
            ``activation``, ``rank``, ``fn``, ``max_resolution`` …).

    Returns:
        An :class:`~nefi.problem.InverseProblem` with ``problem.curriculum`` set and every
        automatic decision in ``problem.meta["auto"]``.

    Example::

        import torch, nefi
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16
        blur = lambda x: torch.nn.functional.conv1d(x[None, None], k[None, None], padding=2)[0, 0]
        x_true = torch.zeros(128); x_true[40:80] = 1.0
        y = blur(x_true) + 0.01 * torch.randn(128)
        problem = nefi.from_forward(blur, y, shape=(128,), prior="nonnegative + piecewise_constant",
                                    homogeneity=1.0, budget="quick")
        result = nefi.invert(problem)
    """
    shape = shape_tuple(shape)
    ndim = len(shape)
    decisions: dict[str, Any] = {}
    dev = resolve_device(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        domain = Domain(shape, tuple(extent) if extent is not None else _default_extent(shape))
        meas = _as_measurement(measurement, mask)

        # ---- priors -> heads -------------------------------------------------------------------
        rep_key = representation if isinstance(representation, str) else "custom"
        if rep_key in ("parametric", "splat") and prior is _DEFAULT_PRIOR:
            # the family / splat density already encodes the values (splats are non-negative
            # by construction): identity head, as in NeTMY App. E.2
            prior = "unconstrained(init=None)"
        if isinstance(prior, Mapping):
            specs: dict[str, CombinedPrior] = {
                k: combine(v, k, ndim, domain) for k, v in prior.items()
            }
            if not specs:
                raise ConfigError("prior dict is empty")
            field_name = next(iter(specs))
        else:
            specs = {field_name: combine(prior, field_name, ndim, domain)}
        multi = len(specs) > 1
        heads = Heads({k: s.head() for k, s in specs.items()}, primary=field_name)
        decisions["priors"] = "; ".join(s.describe().replace("\n", " ") for s in specs.values())
        if any(p.wraps_field for k, s in specs.items() if k != field_name for p in s.priors):
            raise ConfigError(
                "symmetry priors can only be given for the primary field when several unknowns "
                "share one network; use symmetric(..., hard=False) for the others"
            )

        # ---- operator -----------------------------------------------------------------------
        if isinstance(forward, Operator):
            op = forward
            if homogeneity is not None and op.homogeneity is None:
                op.homogeneity = float(homogeneity)
            missing = [k for k in op.required_fields() if k not in specs]
            if missing:
                raise ConfigError(
                    f"the operator requires fields {missing} but the prior defines "
                    f"{tuple(specs)}; pass field_name={missing[0]!r} or prior={{...}} with these "
                    "names"
                )
        else:
            forced = multiscale in (True, "upsample")
            has_ms = at_resolution is not None or callable(output_shape)
            op = FunctionOperator(
                forward,
                field=field_name,
                takes_dict=takes_dict or multi,
                fields=tuple(specs),
                homogeneity=homogeneity,
                output_shape=(
                    output_shape
                    if output_shape is not None
                    else (tuple(meas.data.shape) if forced and not has_ms else None)
                ),
                at_resolution=at_resolution,
                native_shape=shape,
                coarse="upsample" if (forced and not has_ms) else None,
            )
        decisions["operator"] = f"{type(op).__name__} (homogeneity={op.homogeneity})"

        # ---- field --------------------------------------------------------------------------
        fhints: dict[str, Any] = {}
        for s in specs.values():
            fhints.update(s.field_hints())
        builder = _field_builder(representation, shape, heads, fhints, dict(field_kw), decisions)
        field = specs[field_name].build_field(builder, ndim)
        if isinstance(representation, Field) and any(
            p.rank() >= 0 for s in specs.values() for p in s.priors
        ):
            log.warning(
                "from_forward: a Field instance was given, so its own heads are used; "
                "head-defining priors (%s) only contribute their losses",
                decisions["priors"],
            )

        # ---- losses -------------------------------------------------------------------------
        data_loss, sigma_known = _data_loss(noise, meas, decisions, op)
        terms: dict[str, Any] = {"data": data_loss}
        strengths: dict[str, float] = {}
        for fname, s in specs.items():
            for k, (loss, strength) in s.losses().items():
                key = f"{k}_{fname}" if multi else k
                terms[key] = loss
                strengths[key] = float(strength)
        losses = LossSet(terms, weights={"data": 1.0, **strengths})
        post = [pp for s in specs.values() for pp in s.postprocess()]
        if (
            post
            and op.homogeneity is None
            and not any(getattr(pp, "homogeneity", None) for pp in post)
        ):
            raise ConfigError(
                "the 'scale' prior needs the homogeneity degree of the forward: pass "
                "homogeneity=... (1 for linear forwards) or scale(homogeneity=...)"
            )
        problem = InverseProblem(domain, field, op, losses, meas, post, name=name)
        problem.to(dev)
        _validate(problem)

        # ---- initial scale ------------------------------------------------------------------
        if not isinstance(representation, Field):
            for fname in [field_name] + [k for k in specs if k != field_name]:
                if specs[fname].auto_init:
                    _auto_init(problem, fname, seed, decisions, specs[fname].top.headroom())

        # ---- curriculum ---------------------------------------------------------------------
        ms, reason = _multiscale_decision(op, shape, multiscale, representation)
        decisions["multiscale"] = reason
        (log.warning if (multiscale in (True, "upsample") and not ms) else log.info)(
            "from_forward: %s", reason
        )
        gaussian = isinstance(data_loss, MSE)
        if discrepancy == "auto":
            # Morozov stopping (checked by the solver only once annealing has finished)
            tau = 1.0 if (gaussian and sigma_known and meas.noise_std is not None) else None
        elif discrepancy is True:
            if meas.noise_std is None:
                raise ConfigError("discrepancy=True needs a noise level: pass noise=σ or 'auto'")
            tau = 1.0
        elif discrepancy is False or discrepancy is None:
            tau = None
        else:
            tau = float(discrepancy)
        chints: dict[str, Any] = {}
        for s in specs.values():
            chints.update(s.curriculum_hints())
        width = getattr(_innermost_field(field), "hidden", None)
        if width:
            chints.setdefault("hidden", int(width))
        cur = auto_curriculum(
            shape,
            budget,
            ms,
            rep_key,
            lr=float(lr) if isinstance(lr, int | float) and not isinstance(lr, bool) else None,
            noise_known=False,
            hints=chints,
            anneal_fraction=anneal_fraction,
        )
        cur.discrepancy_tau = tau
        problem.curriculum = cur

        # ---- weights ------------------------------------------------------------------------
        if isinstance(weights, Mapping):
            unknown = [k for k in weights if k not in losses.weights]
            if unknown:
                raise ConfigError(f"weights for unknown losses {unknown}; losses: {losses.names}")
            losses.weights.update({k: float(v) for k, v in weights.items()})
            decisions["weights"] = f"user (absolute): {dict(losses.weights)}"
        elif weights in ("raw", None):
            decisions["weights"] = f"raw prior strengths: {dict(losses.weights)}"
        elif weights == "auto":
            w = balance_weights(problem, strengths, probe_steps=probe_steps, lr=cur.stages[0].lr)
            decisions["weights"] = "auto: " + ", ".join(f"{k}={v:.3g}" for k, v in w.items())
        else:
            raise ConfigError("weights must be 'auto', 'raw' or a dict {loss_name: weight}")

        # ---- learning rate ------------------------------------------------------------------
        if lr == "auto":
            best = lr_range_test(problem)
            f = best / cur.stages[0].lr
            for st in cur.stages:
                st.lr *= f
            decisions["lr"] = f"auto (lr_range_test): {best:.2e}"
        else:
            origin = "given" if lr is not None else f"default for {rep_key}"
            decisions["lr"] = f"{cur.stages[0].lr:.2e} ({origin})"
    decisions["curriculum"] = ", ".join(
        f"{s.name} {s.shape} x{s.steps} lr={s.lr:.1e}" for s in cur.stages
    ) + (f", discrepancy tau={tau}" if tau else "")
    decisions["n_parameters"] = problem.field.n_parameters()
    problem.meta["auto"] = decisions
    for k, v in decisions.items():
        log.info("from_forward[%s] %s: %s", name, k, v)
    return problem


# =============================================================================================
# report
# =============================================================================================
def _fmt(v: float) -> str:
    return f"{v:.3g}" if isinstance(v, int | float) and math.isfinite(v) else str(v)


def _fit_verdict(chi: float, estimated: bool) -> str:
    lo, hi = (0.75, 1.3) if estimated else (0.9, 1.2)
    if chi < lo:
        return (
            "below the noise level — likely fitting noise; strengthen the priors, lower the "
            "budget or keep discrepancy stopping on"
        )
    if chi <= hi:
        return "at the noise floor ✓"
    if chi <= 2.0:
        return "slightly above the noise floor — more steps or weaker regularization may help"
    return "underfit — check the forward model / units, raise the budget, or relax the priors"


def quick_report(result, problem: InverseProblem, gt: Any = None) -> str:
    """Compact markdown summary: does the fit reach the noise floor, and what happened.

    Reports the data RMSE against the noise level (``χ = RMSE / σ`` with a verdict; the band
    counted as "at the noise floor" is wider when σ was estimated), per-stage steps / stop
    reasons / time, the scale-correction factor, the parameter count, timing, the automatic
    decisions of :func:`from_forward`, and ground-truth metrics (PSNR, SSIM, relative error) when
    ``gt`` (tensor or ``{name: tensor}``) is given.

    Example::

        import torch, nefi
        x_true = torch.zeros(64); x_true[20:40] = 1.0
        y = x_true + 0.05 * torch.randn(64)
        problem = nefi.from_forward(lambda x: x, y, shape=(64,), prior="nonnegative + tv",
                                    budget=100)
        result = nefi.invert(problem)
        print(nefi.quick_report(result, problem, gt=x_true))
    """
    from .metrics.basic import psnr, relative_error, ssim

    meas = problem.measurement
    pred = result.pred.detach().cpu().float()
    y = meas.data.detach().cpu().float()
    n_par = result.extra.get("n_parameters", problem.field.n_parameters())
    rows: list[tuple[str, str]] = [
        ("problem", f"`{problem.name}`: field {problem.domain.shape}, data {tuple(y.shape)}"),
        ("representation", f"{type(problem.field).__name__}, {n_par} parameters"),
        ("operator", type(problem.operator).__name__),
    ]
    if tuple(pred.shape) == tuple(y.shape):
        m = torch.ones_like(y)
        if meas.mask is not None:
            mk = meas.mask.detach().cpu()
            m = (mk.real if mk.is_complex() else mk).float().expand_as(y)
        sq = (pred - y) ** 2 * m
        rmse = float(torch.sqrt(sq.sum() / m.sum().clamp_min(1)))
        rel = float(torch.sqrt(sq.sum() / (y**2 * m).sum().clamp_min(1e-30)))
        sigma = None if meas.noise_std is None else float(torch.as_tensor(meas.noise_std).mean())
        if sigma is not None and sigma > 0:
            estimated = bool(meas.meta.get("noise_std_estimated"))
            chi = rmse / sigma
            est = " (estimated; typically within ±20 %)" if estimated else ""
            rows.append(("noise level σ", f"{_fmt(sigma)}{est}"))
            verdict = _fit_verdict(chi, estimated)
            rows.append(("data fit", f"RMSE {_fmt(rmse)} = **{chi:.2f} σ** → {verdict}"))
        elif sigma is not None:
            rows.append(("data fit", f"RMSE {_fmt(rmse)} (noiseless data, σ = 0)"))
        else:
            msg = f"RMSE {_fmt(rmse)} (noise level unknown: pass noise=σ or noise='auto')"
            rows.append(("data fit", msg))
        rows.append(("relative residual", _fmt(rel)))
    else:
        msg = f"prediction {tuple(pred.shape)} vs data {tuple(y.shape)}: not comparable"
        rows.append(("data fit", msg))
    stages = [
        f"{s.get('name')} {tuple(s.get('shape', ()))}: {s.get('steps')} steps, "
        f"stop={s.get('stop')}, {s.get('seconds', 0):.1f}s"
        for s in result.stage_results
    ]
    rows.append(("stages", "; ".join(stages) or "-"))
    if "scale_factor" in result.post_info:
        rows.append(("scale correction α", _fmt(result.post_info["scale_factor"])))
    total_s = result.timing.get("total_s", float("nan"))
    rows.append(("time", f"{total_s:.1f} s on {result.extra.get('device', '?')}"))
    if gt is not None:
        g = gt[problem.field.primary] if isinstance(gt, Mapping) else gt
        g = torch.as_tensor(g).detach().cpu().float()
        x = result.fields[problem.field.primary].detach().cpu().float()
        if tuple(x.shape) == tuple(g.shape):
            parts = [f"PSNR {psnr(x, g):.2f} dB", f"rel. error {relative_error(x, g):.3f}"]
            if 1 <= g.ndim <= 3 and min(g.shape[: min(g.ndim, 2)]) >= 3:
                try:
                    parts.insert(1, f"SSIM {ssim(x, g):.3f}")
                except Exception:  # noqa: BLE001 - SSIM is best effort
                    pass
            rows.append(("vs ground truth", ", ".join(parts)))
        else:
            rows.append(("vs ground truth", f"shape {tuple(x.shape)} vs {tuple(g.shape)}"))
    out = [f"### nefi quick report: {problem.name}", "", "| item | value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    dec = problem.meta.get("auto") if isinstance(problem.meta, dict) else None
    if dec:
        out += ["", "<details><summary>automatic decisions</summary>", ""]
        out += [f"- **{k}**: {v}" for k, v in dec.items()]
        out += ["", "</details>"]
    return "\n".join(out)


__all__ = [
    "BUDGETS",
    "DEFAULT_LR",
    "auto_curriculum",
    "balance_weights",
    "budget_steps",
    "default_lr",
    "default_octaves",
    "estimate_noise",
    "from_forward",
    "lr_range_test",
    "quick_report",
]
