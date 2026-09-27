"""Reconstruction-side edge refinement: a short, physics-constrained sharpening stage.

Smooth reconstructions of piecewise-constant unknowns come out with soft edges and reduced
contrast (``docs/refinement.md``): the forward map is smoothing (singular values decay, NeFTY
Prop. 2 — edges are the first thing the data stop constraining), the prior prefers gentle
transitions (a band-limited coordinate MLP filters the gradient, NeTMY Lemma 2; TV with a finite
``ε`` and small weight tolerates ramps), and a finite budget stops before fine detail converges.
:func:`refine_edges` starts from such a reconstruction and runs a *second, short* optimization
under the **same** forward operator and the **same** data, with a prior that favours crisp
interfaces:

* ``mode="levelset"`` — a multi-phase level set over **one** function ``φ`` (Chung & Vese's
  multilayer level set; two phases: ``lo + (hi − lo) σ(φ/ε)``, the :class:`LevelSetHead`
  formula)::

      x = v₀ + Σ_{j=1}^{k−1} (v_j − v_{j−1}) · H_ε(φ − c_j),      ε: ε_start → ε_end

  φ is initialized from the smooth result, ``φ₀ = (x_s − τ₁)/(v_{k−1} − v₀)`` with value-space
  thresholds ``τ_j`` (Otsu / multi-Otsu, volume- or mass-matched, or given) and level offsets
  ``c_j = (τ_j − τ₁)/(v_{k−1} − v₀)``; the phase values ``v_j`` start at robust percentiles of the
  classes and may be learnable scalars — or *free* per voxel (``free_phases``: crisp interfaces
  around smooth interiors). ``render="area"`` evaluates the *cell average* of the Heaviside over
  each voxel (partial volumes: sub-voxel interface positions, like the cell-averaged ground truths
  of the instances). φ is a :class:`~nefi.fields.GridField` (default) or a coordinate MLP
  warm-started by regression on ``φ₀`` (``representation="neural"``).
* ``mode="phasefield"`` — the original problem continued on a voxel grid (warm-started through the
  head inverse) with a Cahn–Hilliard-style multi-well penalty
  ``W(x) = Π_j ((x − v_j)/(v_{k−1} − v₀))²`` whose weight grows with progress; the field stays
  continuous but is pushed to the phases.
* ``mode="tv_sharpen"`` — the original problem continued with a stronger isotropic TV and, when
  the head is bracketed (``Bounded`` / ``LogBounded``), a binarizing
  :class:`~nefi.fields.LevelSetHead` swap (the :class:`nefi.priors.Binary` head). The cheapest
  option; the swap binarizes the *current* state, so it suits a converged smooth solution (an
  under-converged one needs values to move continuously, which a saturating sigmoid resists).
* ``mode="continue"`` — the **control**: the same voxel-grid continuation and budget with the
  problem's own losses and head, no sharpening prior. A smooth result stopped early (``χ > 1``)
  improves under any continuation; compare against this mode before crediting a prior.

**Strength of the added prior.** Penalties are scaled so that their energy at the smooth solution
is ``γ ×`` a reference data loss with ``γ = fit_tol² − 1`` (0.21 for the default
``fit_tol = 1.1``): a descent method started at a converged smooth solution can then raise the data
loss by at most that much (``L_data ≤ (1 + γ) L_data(x_s)``), i.e. the RMSE by at most
``fit_tol`` — the tolerance the acceptance test uses. The reference is the smooth solution's data
loss, brought down to the noise floor (``× 1/χ²``) when the smooth solution stopped early. Changes
of representation (level set, head swap) have no such bound; the test catches them.

**Reporting rules.** The refined field is a *separate* :class:`~nefi.solve.Result`
(``extra["refined_from"]``); it never replaces the smooth one silently. The
:class:`RefineReport` shows the data fit before and after (RMSE, relative RMSE and
``χ = RMSE/σ`` when the noise level is known) next to the thresholds, phase values and — given a
ground truth — segmentation metrics. The refinement is **refused** (the smooth result is returned
and the report says why) when the fit degrades beyond ``fit_tol × max(χ_ref, 1)`` (σ known) or
``fit_tol × RMSE_ref`` (σ unknown) — the reference being the smooth result, or, when that did not
reach the noise floor, the better of it and the ``continue`` control with the same budget (an
under-converged smooth start would otherwise make the test vacuous) —, when sharpening costs more
than ``fit_tol`` in misfit relative to the best fit the same stage reached before it sharpened, or
when a refined phase volume leaves the smooth
solution's transition band by more than ``max_volume_change`` (default 30 %,
:func:`transition_band_check`): refinement is a prior (a piecewise-constant world), and it must
not invent interfaces the data cannot support.

Example::

    import nefi
    from nefi.solve.refine import refine_edges

    result = nefi.invert(problem)
    refined, report = refine_edges(problem, result, mode="levelset", levels=(0.0, 1.0), gt=gt)
    print(report.to_markdown())          # data fit before → after, IoU / Edge-F1 before → after
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..errors import ConfigError
from ..fields.base import Field
from ..fields.grid import GridField
from ..fields.heads import Bounded as BoundedHead
from ..fields.heads import Head, Heads, Identity
from ..fields.heads import Softplus as SoftplusHead
from ..fields.levelset import LevelSetHead
from ..fields.modifiers import innermost, value_range
from ..fields.neural import NeuralField
from ..losses.base import Context, Loss, LossSet
from ..losses.reg import TV, RangePenalty
from ..metrics.basic import psnr, ssim
from ..metrics.segmentation import edge_f1
from ..utils.device import resolve_device
from ..utils.seed import seed_everything
from ..utils.tensor import resample
from .curriculum import Curriculum, OptimConfig, Stage
from .result import Result
from .solver import Solver

log = logging.getLogger("nefi")

#: refinement modes accepted by :func:`refine_edges` (``"continue"`` is the control: the same
#: grid continuation and budget without any sharpening prior)
REFINE_MODES = ("levelset", "phasefield", "tv_sharpen", "continue")
#: loss classes dropped by the sharpening modes: quadratic smoothness priors fight sharp edges
SMOOTHNESS_LOSSES = ("Laplacian", "GradientL2")
#: :func:`refine_edges` arguments that only apply to ``mode="levelset"``
LEVELSET_ONLY = (
    "representation",
    "free_phases",
    "perimeter",
    "learn_levels",
    "level_lr_mult",
    "eps",
    "render",
    "neural",
    "warm_steps",
)

__all__ = [
    "LEVELSET_ONLY",
    "REFINE_DEFAULTS",
    "REFINE_MODES",
    "SMOOTHNESS_LOSSES",
    "DoubleWell",
    "MultiPhaseHead",
    "RefineReport",
    "cell_heaviside",
    "data_fit",
    "edge_metrics",
    "estimate_levels",
    "mass_matched_fraction",
    "multi_otsu",
    "noise_level",
    "otsu_threshold",
    "refine_defaults",
    "refine_edges",
    "refine_instance",
    "refine_method",
    "refine_run_output",
    "transition_band_check",
    "volume_matched_threshold",
]


# ============================================================================================
# thresholds and phase values
# ============================================================================================
def _flat(x: Any) -> torch.Tensor:
    t = torch.as_tensor(x).detach().reshape(-1).to("cpu", torch.float64)
    t = t[torch.isfinite(t)]
    if t.numel() == 0:
        raise ConfigError("cannot compute thresholds of an empty / non-finite field")
    return t


def volume_matched_threshold(x: Any, fraction: float, above: bool = True) -> float:
    """Threshold ``τ`` such that a ``fraction`` of the voxels lies above it (``below`` if
    ``above=False``) — the level whose super-level set has a prescribed volume.

    ``τ`` is the midpoint between the ``m``-th and ``(m+1)``-th ordered values,
    ``m = round(fraction · N)``, so ``{x > τ}`` holds exactly ``m`` voxels (ties aside).

    Example::

        x = torch.linspace(0, 1, 101)
        tau = volume_matched_threshold(x, 0.25)          # 0.755: 25 values lie above
    """
    v = _flat(x)
    n = v.numel()
    m = int(round(min(max(float(fraction), 0.0), 1.0) * n))
    s = torch.sort(v, descending=above).values
    if m <= 0:
        return float(s[0]) + (1.0 if above else -1.0) * 1e-12 * max(1.0, abs(float(s[0])))
    if m >= n:
        return float(s[-1]) - (1.0 if above else -1.0) * 1e-12 * max(1.0, abs(float(s[-1])))
    return 0.5 * float(s[m - 1] + s[m])


def mass_matched_fraction(x: Any, lo: float, hi: float) -> float:
    """Volume fraction of the ``hi`` phase that conserves the mean of ``x``:
    ``f = clip((mean x − lo)/(hi − lo), 0, 1)``.

    Smoothing forward maps (blur, diffusion) preserve low-order moments, so the integral of a
    smooth reconstruction is usually well determined even when its edges are not; thresholding at
    ``volume_matched_threshold(x, f)`` then gives a two-phase field with the same integral.
    """
    if not hi > lo:
        raise ConfigError(f"mass_matched_fraction needs hi > lo, got {(lo, hi)}")
    m = float(_flat(x).mean())
    return min(max((m - float(lo)) / (float(hi) - float(lo)), 0.0), 1.0)


def multi_otsu(x: Any, k: int = 2, bins: int = 256) -> list[float]:
    """Globally optimal ``k``-class thresholds of a histogram (multi-level Otsu).

    Maximizing the between-class variance is minimizing the within-class variance, i.e. optimal
    1-D k-means on the histogram, solved exactly by dynamic programming over the ``bins`` bins
    (``O(k · bins²)``). ``k = 2`` is Otsu's method.

    Returns:
        ``k − 1`` increasing thresholds, each centred in the (possibly empty) gap between the
        last occupied bin of one class and the first occupied bin of the next.

    Example::

        x = torch.cat([torch.zeros(90), torch.ones(10)])
        multi_otsu(x, 2)                    # [0.5]
    """
    k = int(k)
    if k < 2:
        raise ConfigError(f"multi_otsu needs k >= 2 classes, got {k}")
    v = _flat(x).numpy()
    lo, hi = float(v.min()), float(v.max())
    if not hi > lo:
        raise ConfigError("cannot threshold a constant field (all values are equal)")
    counts, edges = np.histogram(v, bins=int(bins), range=(lo, hi))
    w = counts.astype(np.float64)
    c = 0.5 * (edges[1:] + edges[:-1])
    n = w.size
    cw = np.concatenate([[0.0], np.cumsum(w)])
    cx = np.concatenate([[0.0], np.cumsum(w * c)])
    cxx = np.concatenate([[0.0], np.cumsum(w * c * c)])

    def sse(i: np.ndarray | int, j: int) -> np.ndarray:  # bins i..j-1 (half-open), vectorized in i
        ww = cw[j] - cw[i]
        s1 = cx[j] - cx[i]
        s2 = cxx[j] - cxx[i]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = s2 - np.where(ww > 0, s1 * s1 / np.where(ww > 0, ww, 1.0), 0.0)
        return np.maximum(out, 0.0)

    inf = np.inf
    # D[m, j]: best SSE of the first j bins split into m classes; A: argmin split point
    D = np.full((k + 1, n + 1), inf)
    A = np.zeros((k + 1, n + 1), dtype=np.int64)
    D[0, 0] = 0.0
    for m in range(1, k + 1):
        for j in range(m, n + 1):
            i = np.arange(m - 1, j)
            cand = D[m - 1, i] + sse(i, j)
            b = int(np.argmin(cand))
            D[m, j], A[m, j] = cand[b], i[b]
    cuts = []
    j = n
    for m in range(k, 1, -1):
        j = int(A[m, j])
        cuts.append(j)
    out = []
    nonempty = np.flatnonzero(w > 0)
    for j in sorted(cuts):  # centre each threshold in the empty gap between the classes
        below, above = nonempty[nonempty < j], nonempty[nonempty >= j]
        if below.size and above.size:
            out.append(float(0.5 * (c[below[-1]] + c[above[0]])))
        else:
            out.append(float(edges[j]))
    return out


def otsu_threshold(x: Any, bins: int = 256) -> float:
    """Otsu's threshold of a field (maximal between-class variance of the value histogram)."""
    return multi_otsu(x, 2, bins)[0]


def estimate_levels(
    x: Any,
    levels: Any = "auto",
    threshold: Any = "auto",
    quantiles: tuple[float, float] = (0.5, 0.9),
    value_range: tuple[float, float] = (-math.inf, math.inf),
) -> tuple[list[float], list[float], str]:
    """Phase values ``v₀ < … < v_{k−1}`` and value-space thresholds ``τ₁ < … < τ_{k−1}`` of a
    smooth reconstruction.

    Args:
        x: the smooth field.
        levels: ``"auto"`` (two phases), an int ``k`` (``k`` phases), or a sequence of ``k``
            phase values in which ``None`` entries are estimated.
        threshold: ``"auto"`` (``"midpoint"`` when every value is given, else ``"otsu"``),
            ``"otsu"`` (multi-level Otsu), ``"midpoint"`` (between the given values),
            ``"mass"`` (two phases: the threshold whose volume conserves the mean of ``x`` given
            the phase values, :func:`mass_matched_fraction`), a float (two phases) or a sequence
            of ``k − 1`` floats.
        quantiles: ``(q_majority, q_extreme)``: the majority class takes its ``q_majority``
            quantile (the median), the other *extreme* classes the ``q_extreme`` quantile
            counted away from their threshold (softened blobs under-shoot their true value, so
            their core is a better estimate than their mean); middle classes take the median.
        value_range: phase values are clamped to this range (the original head's range).

    Returns:
        ``(values, thresholds, method)``.
    """
    v = _flat(x)
    if isinstance(levels, str):
        if levels != "auto":
            raise ConfigError(f"levels must be 'auto', an int or a sequence, got {levels!r}")
        given: list[float | None] = [None, None]
    elif isinstance(levels, bool):
        raise ConfigError("levels must be 'auto', an int or a sequence of values")
    elif isinstance(levels, int):
        if levels < 2:
            raise ConfigError(f"levels needs at least two phases, got {levels}")
        given = [None] * int(levels)
    else:
        given = [None if g is None else float(g) for g in levels]
        if len(given) < 2:
            raise ConfigError(f"levels needs at least two phase values, got {levels!r}")
    k = len(given)
    known = [g for g in given if g is not None]
    if any(b <= a for a, b in zip(known, known[1:])):
        raise ConfigError(f"given phase values must be strictly increasing, got {levels!r}")
    rng = (float(value_range[0]), float(value_range[1]))
    method = str(threshold) if isinstance(threshold, str) else "given"
    if isinstance(threshold, str) and threshold == "auto":
        method = "midpoint" if len(known) == k else "otsu"
    # ---- thresholds ------------------------------------------------------------------------
    if method == "given":
        taus = [float(threshold)] if isinstance(threshold, int | float) else list(threshold)
        taus = [float(t) for t in taus]
        if len(taus) != k - 1 or any(b <= a for a, b in zip(taus, taus[1:])):
            raise ConfigError(f"need {k - 1} increasing thresholds for {k} phases, got {taus}")
    elif method == "otsu":
        taus = multi_otsu(v, k)
    elif method == "midpoint":
        if len(known) != k:
            raise ConfigError("threshold='midpoint' needs every phase value")
        taus = [0.5 * (a + b) for a, b in zip(known, known[1:])]
    elif method == "mass":
        if k != 2:
            raise ConfigError("threshold='mass' is defined for two phases")
        lo_hi = list(given)
        if None in lo_hi:  # estimate the missing value from an Otsu split first
            vals, _, _ = estimate_levels(v, given, "otsu", quantiles, rng)
            lo_hi = vals
        f = mass_matched_fraction(v, float(lo_hi[0]), float(lo_hi[1]))
        taus = [volume_matched_threshold(v, f, above=True)]
    else:
        raise ConfigError(
            f"unknown threshold {threshold!r}; use 'auto', 'otsu', 'midpoint', 'mass', a float "
            "or a sequence of floats"
        )
    # ---- values ----------------------------------------------------------------------------
    edges = [-math.inf, *taus, math.inf]
    classes = [v[(v >= edges[j]) & (v < edges[j + 1])] for j in range(k)]
    counts = [int(c.numel()) for c in classes]
    major = int(np.argmax(counts))
    q_major, q_ext = float(quantiles[0]), float(quantiles[1])
    values: list[float] = []
    for j, (g, c) in enumerate(zip(given, classes)):
        if g is not None:
            values.append(g)
            continue
        if c.numel() == 0:
            raise ConfigError(
                f"cannot estimate phase {j} of {k}: no voxel lies in [{edges[j]:.4g}, "
                f"{edges[j + 1]:.4g}); pass levels=(...) explicitly or use fewer phases"
            )
        if j == major:
            q = q_major
        elif j == k - 1:
            q = q_ext
        elif j == 0:
            q = 1.0 - q_ext
        else:
            q = 0.5
        values.append(float(torch.quantile(c, q)))
    values = [min(max(val, rng[0]), rng[1]) for val in values]
    span = max(float(v.max() - v.min()), 1e-12)
    for j in range(1, k):
        if not values[j] - values[j - 1] > 1e-6 * span:
            raise ConfigError(
                f"phase {j} value {values[j]:.4g} is not above phase {j - 1} value "
                f"{values[j - 1]:.4g}: the phases {[round(a, 6) for a in values]} do not bracket "
                f"the field (range [{float(v.min()):.4g}, {float(v.max()):.4g}]) — e.g. a given "
                "background above inclusions that lie below it; pass levels in increasing "
                "order around the field's values, or levels='auto'"
            )
    return values, taus, method


# ============================================================================================
# heads and losses of the refinement stage
# ============================================================================================
def _softplus_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``softplus(a) − softplus(b)`` without cancellation when both arguments are large."""
    pos = (a > 0) & (b > 0)
    stable = (a - b) + F.softplus(-a) - F.softplus(-b)
    return torch.where(pos, stable, F.softplus(a) - F.softplus(b))


def _cell_range(phi: torch.Tensor) -> torch.Tensor:
    """Range of a locally linear ``φ`` over each cell: ``Σ_d |∂_d φ|`` (index units)."""
    h = torch.zeros_like(phi)
    for ax in range(phi.ndim):
        if phi.shape[ax] > 1:
            h = h + torch.gradient(phi, dim=ax)[0].abs()
    return h


def cell_heaviside(s: torch.Tensor, eps: float, h: torch.Tensor | None = None) -> torch.Tensor:
    """Smooth Heaviside ``σ(s/ε)``, or its average over cells spanning ``h`` units of ``s``.

    ``(1/h)∫_{s−h/2}^{s+h/2} σ(t/ε) dt = (ε/h)[softplus((s + h/2)/ε) − softplus((s − h/2)/ε)]``
    tends to the partial-volume fraction ``clip(s/h + ½, 0, 1)`` as ``ε → 0``: a perfectly sharp
    interface still has a differentiable, sub-voxel position (cf.
    :func:`nefi.fields.geometric.smooth_step` for a scalar cell width).
    """
    eps = max(float(eps), 1e-12)
    point = torch.sigmoid(s / eps)
    if h is None:
        return point
    hc = h.clamp_min(1e-6)
    avg = (eps / hc) * _softplus_diff((s + 0.5 * hc) / eps, (s - 0.5 * hc) / eps)
    return torch.where(h > 1e-3 * eps, avg, point)


def _anneal(p: float, a: float, b: float, schedule: str = "geometric") -> float:
    p = min(max(float(p), 0.0), 1.0)
    if schedule == "geometric":
        return a * (b / a) ** p
    if schedule == "linear":
        return a + (b - a) * p
    if schedule == "cosine":
        return b + (a - b) * 0.5 * (1.0 + math.cos(math.pi * p))
    raise ConfigError(f"unknown schedule {schedule!r}; use 'geometric', 'linear' or 'cosine'")


class MultiPhaseHead(Head):
    """``k``-phase head over one level-set function (multilayer level set).

    ``x = v₀ + Σ_{j=1}^{k−1} (v_j − v_{j−1}) H_ε(φ − c_j)`` with ``ε`` shrinking from
    ``eps_start`` (progress 0) to ``eps_end`` (progress 1); two phases give the
    :class:`~nefi.fields.LevelSetHead` formula ``lo + (hi − lo) σ(φ/ε)``. The phases are nested
    super-level sets of φ (Chung & Vese 2005), so one function represents any ordered
    ``k``-phase field. In general form ``x = Σ_j V_j χ_j`` with the phase indicators
    ``χ_j = H_ε(φ − c_j) − H_ε(φ − c_{j+1})`` (a partition of unity) and ``V_j`` either a scalar
    (a piecewise-*constant* phase) or, for the ``free`` phases, a per-point value read from an
    extra raw channel (a piecewise-*smooth* world: crisp interfaces, free interiors).

    Args:
        values: initial phase values ``v₀ < … < v_{k−1}`` (field units; for free phases the
            representative value used for thresholds and reports).
        offsets: level offsets ``c₁ < … < c_{k−1}`` in φ units.
        eps_start, eps_end, schedule: interface softness ``ε(progress)`` in φ units.
        render: ``"area"`` — cell average of the Heaviside over each voxel using the local
            φ-range ``Σ_d|∂_d φ|`` (stop-gradient): partial volumes, sub-voxel interfaces; used
            when the head sees the full grid, point sampling otherwise — or ``"point"``.
        learn: learnable scalar phase values: ``True`` (all), ``False`` (none) or indices.
        free: indices of phases with per-point values: raw channels ``1, 2, …`` (in this order)
            hold ``(V_j − v₀)/(v_{k−1} − v₀)``, i.e. values in units of the contrast like φ.
        value_range: hard clamp of the phase values (e.g. the original head's bracket).
        ndim: spatial dimension of the grid (enables ``render="area"``).

    Example::

        head = MultiPhaseHead([0.0, 0.5, 1.0], [-0.2, 0.2], eps_end=1e-3, render="point")
        head(torch.tensor([[-1.0], [0.0], [1.0]]), {}, 1.0)        # ≈ [0.0, 0.5, 1.0]
    """

    uses_progress = True

    def __init__(
        self,
        values: Sequence[float],
        offsets: Sequence[float] | None = None,
        eps_start: float = 0.25,
        eps_end: float = 0.02,
        schedule: str = "geometric",
        render: str = "area",
        learn: bool | Sequence[int] = True,
        free: Sequence[int] = (),
        value_range: tuple[float, float] = (-math.inf, math.inf),
        ndim: int | None = None,
    ) -> None:
        super().__init__()
        v = [float(a) for a in values]
        k = len(v)
        if k < 2 or any(b <= a for a, b in zip(v, v[1:])):
            raise ConfigError(f"MultiPhaseHead needs >= 2 increasing values, got {v}")
        if offsets is None:
            offsets = [(b - v[0]) / (v[-1] - v[0]) - 0.5 for b in v[1:]]
        c = [float(a) for a in offsets]
        if len(c) != k - 1 or any(b <= a for a, b in zip(c, c[1:])):
            raise ConfigError(f"need {k - 1} increasing level offsets, got {c}")
        if not (eps_start > 0 and eps_end > 0):
            raise ConfigError("MultiPhaseHead needs positive eps_start / eps_end")
        if render not in ("area", "point"):
            raise ConfigError(f"render must be 'area' or 'point', got {render!r}")
        self.free = tuple(sorted({int(j) % k for j in free}))
        if len(self.free) == k:
            raise ConfigError("at least one phase must have a constant value (free has all phases)")
        self.n_in = 1 + len(self.free)
        self.k = k
        self.scale = v[-1] - v[0]
        self.register_buffer("v_init", torch.tensor(v, dtype=torch.float32))
        self.register_buffer("offsets", torch.tensor(c, dtype=torch.float32))
        if isinstance(learn, bool):
            mask = [1.0 if learn else 0.0] * k
        else:
            idx = {int(i) % k for i in learn}
            mask = [1.0 if j in idx else 0.0 for j in range(k)]
        mask = [0.0 if j in self.free else m for j, m in enumerate(mask)]
        self.register_buffer("learn_mask", torch.tensor(mask, dtype=torch.float32))
        #: normalized value offsets (units of the initial contrast ``v_{k−1} − v₀``)
        self.delta = nn.Parameter(torch.zeros(k), requires_grad=any(m > 0 for m in mask))
        self.eps_start, self.eps_end, self.schedule = float(eps_start), float(eps_end), schedule
        self.render = render
        self.value_range = (float(value_range[0]), float(value_range[1]))
        self.ndim = ndim

    def eps(self, progress: float = 1.0) -> float:
        """Interface softness ``ε`` (φ units) at ``progress``."""
        return _anneal(progress, self.eps_start, self.eps_end, self.schedule)

    def _clamp(self, v: torch.Tensor) -> torch.Tensor:
        lo, hi = self.value_range
        if math.isfinite(lo) or math.isfinite(hi):
            v = v.clamp(
                min=lo if math.isfinite(lo) else None, max=hi if math.isfinite(hi) else None
            )
        return v

    def values(self) -> torch.Tensor:
        """Current scalar phase values ``(k,)`` (clamped; free phases: their initial value)."""
        return self._clamp(self.v_init + self.scale * self.delta * self.learn_mask)

    def _area(self, phi: torch.Tensor) -> bool:
        return self.render == "area" and self.ndim is not None and phi.ndim == self.ndim

    def indicators(self, phi: torch.Tensor, progress: float = 1.0) -> list[torch.Tensor]:
        """Smoothed Heavisides ``H_ε(φ − c_j)``, ``j = 1 … k−1`` (cell-averaged on grids)."""
        e = self.eps(progress)
        h = _cell_range(phi.detach()) if self._area(phi) else None
        return [cell_heaviside(phi - self.offsets[j].to(phi), e, h) for j in range(self.k - 1)]

    def transform(self, raw, others, progress: float = 1.0):
        phi = raw[..., 0]
        v = self.values().to(phi)
        hs = self.indicators(phi, progress)
        if not self.free:  # telescoped piecewise-constant form
            out = v[0].expand_as(phi)
            for j in range(1, self.k):
                out = out + (v[j] - v[j - 1]) * hs[j - 1]
            return out
        out = torch.zeros_like(phi)
        for j in range(self.k):
            chi = (hs[j - 1] if j > 0 else 1.0) - (hs[j] if j < self.k - 1 else 0.0)
            if j in self.free:  # raw channel in contrast units: V = v₀ + (v_{k−1} − v₀)·r
                r = raw[..., 1 + self.free.index(j)]
                val = self._clamp(self.v_init[0].to(r) + self.scale * r)
            else:
                val = v[j]
            out = out + val * chi
        return out

    def phase(self, raw: torch.Tensor) -> torch.Tensor:
        """Integer phase index ``Σ_j 1{φ > c_j}`` of every point."""
        phi = raw[..., 0]
        return (phi.unsqueeze(-1) > self.offsets.to(phi)).sum(-1)

    def extra_repr(self) -> str:
        return (
            f"k={self.k}, eps=({self.eps_start:g} → {self.eps_end:g}), render={self.render}, "
            f"learn={self.learn_mask.tolist()}, free={list(self.free)}"
        )


class _FixedHead(Head):
    """A field held fixed during refinement (a buffer; consumes no raw channel)."""

    n_in = 0

    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("value", torch.as_tensor(value).detach().clone())

    def transform(self, raw, others):
        shape = tuple(raw.shape[:-1])
        v = self.value
        if tuple(v.shape) != shape and v.ndim == len(shape) and 1 <= v.ndim <= 3:
            v = resample(v, shape)
        return v.to(device=raw.device, dtype=raw.dtype if v.is_floating_point() else v.dtype)


class DoubleWell(Loss):
    """Cahn–Hilliard-style multi-well penalty ``mean Π_j ((x − v_j)/(v_{k−1} − v₀))²``.

    Zero exactly at the phase values, ``(1/16)`` in the middle of a two-phase gap; the weight is
    multiplied by ``ramp(progress)`` (``ramp[0] → ramp[1]``, geometric), so the field is pushed to
    the phases gradually while the data term keeps it where the data need it.
    """

    def __init__(
        self,
        field: str,
        values: Sequence[float],
        ramp: tuple[float, float] = (0.01, 1.0),
        name: str | None = None,
    ) -> None:
        super().__init__(name or "double_well")
        v = [float(a) for a in values]
        if len(v) < 2 or not v[-1] > v[0]:
            raise ConfigError(f"DoubleWell needs >= 2 increasing values, got {v}")
        self.field_name = field
        self.register_buffer("wells", torch.tensor(v, dtype=torch.float32))
        self.scale = v[-1] - v[0]
        self.ramp = (float(ramp[0]), float(ramp[1]))

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.ones_like(x)
        for v in self.wells.to(x):
            w = w * ((x - v) / self.scale) ** 2
        return w.mean()

    def forward(self, ctx: Context) -> torch.Tensor:
        r = _anneal(ctx.progress, max(self.ramp[0], 1e-12), max(self.ramp[1], 1e-12))
        return r * self.energy(ctx.field(self.field_name))


# ============================================================================================
# data fit and metrics
# ============================================================================================
def noise_level(noise_std: Any) -> float | None:
    """A measurement's noise level as one number: the scalar, or the mean of the positive
    entries of a per-entry ``noise_std`` (``None`` when unknown)."""
    if noise_std is None:
        return None
    s = torch.as_tensor(noise_std).detach().cpu()
    s = (s.abs() if s.is_complex() else s).double().reshape(-1)
    s = s[s > 0]
    return float(s.mean()) if s.numel() else None


def data_fit(pred: torch.Tensor, measurement: Any) -> dict[str, float | None]:
    """Data misfit of a prediction: ``rmse``, ``rel_rmse`` and ``chi`` over observed entries.

    ``chi = sqrt(mean ((pred − obs)/σ)²)`` uses the measurement's noise level (scalar or
    per-entry — entries with ``σ = 0`` excluded; fractional mask weights respected): ≈ 1 means a
    fit at the noise floor (Morozov's discrepancy principle), ``None`` when σ is unknown.
    Complex data are compared on their real and imaginary parts.
    """
    y = torch.as_tensor(measurement.data).detach().cpu()
    p = torch.as_tensor(pred).detach().cpu()
    if tuple(p.shape) != tuple(y.shape):
        raise ConfigError(
            f"prediction {tuple(p.shape)} and measurement {tuple(y.shape)} differ in shape"
        )
    m = measurement.mask
    m = None if m is None else torch.as_tensor(m).detach().cpu()
    ns = measurement.noise_std
    s = None if ns is None else torch.as_tensor(ns).detach().cpu()
    if s is not None and s.is_complex():
        s = s.abs()
    if m is not None and m.is_complex():
        m = m.real
    if p.is_complex() or y.is_complex():
        p = torch.view_as_real(p.to(torch.complex128))
        y = torch.view_as_real(y.to(torch.complex128))
        if m is not None:
            m = torch.broadcast_to(m, y.shape[:-1]).unsqueeze(-1)
        if s is not None and s.numel() > 1:
            s = torch.broadcast_to(s, y.shape[:-1]).unsqueeze(-1)
    p, y = p.double(), y.double()
    w = torch.ones_like(y) if m is None else torch.broadcast_to(m.double(), y.shape)
    den = float(w.sum())
    if den <= 0:
        raise ConfigError("the measurement mask has no observed entry")
    r2 = (p - y) ** 2
    mse = float((r2 * w).sum()) / den
    ms = float((y * y * w).sum()) / den
    rmse = math.sqrt(max(mse, 0.0))
    out: dict[str, float | None] = {
        "rmse": rmse,
        "rel_rmse": rmse / math.sqrt(ms) if ms > 0 else math.inf,
        "chi": None,
    }
    if s is None:
        return out
    s = s.double()
    if s.numel() == 1:
        sv = float(s.reshape(()))
        out["chi"] = rmse / sv if sv > 0 else None
        return out
    s = torch.broadcast_to(s, y.shape)
    ok = (w > 0) & (s > 0)
    if int(ok.sum()) == 0:
        return out
    wo = w[ok]
    out["chi"] = math.sqrt(float((wo * r2[ok] / s[ok] ** 2).sum() / wo.sum()))
    return out


def _misfit(fit: Mapping[str, Any]) -> tuple[float, str]:
    """The misfit used by the acceptance test: ``χ`` when known, else the RMSE."""
    if fit.get("chi") is not None:
        return float(fit["chi"]), "chi"
    return float(fit["rmse"]), "rmse"


def _mask(x: torch.Tensor, tau: float, above: bool) -> torch.Tensor:
    return x > tau if above else x < tau


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    union = int((a | b).sum())
    return 1.0 if union == 0 else float((a & b).sum()) / union


def edge_metrics(
    pred: Any,
    gt: Any,
    *,
    tau: float | str | None = None,
    above: bool | str = "auto",
    data_range: float | None = None,
    edge_threshold: float = 0.5,
    edge_dilate: int = 1,
) -> dict[str, float]:
    """Contrast, segmentation and edge metrics of a reconstruction against a ground truth.

    * ``psnr`` / ``ssim`` (range: ``data_range`` or the GT range);
    * ``iou`` / ``dice`` of the masks ``{x > τ}`` (``{x < τ}`` with ``above=False``; ``"auto"``:
      the GT's minority side, i.e. the features) at the GT threshold ``τ`` (a float, ``"otsu"`` /
      ``None`` — Otsu of the GT — or ``"half_max"`` — halfway between the GT's extremes);
    * ``iou_vm`` — *volume-matched* IoU: the reconstruction is thresholded at the level whose
      mask has the GT mask's volume (shape agreement, independent of contrast calibration);
    * ``edge_f1`` — :func:`nefi.metrics.segmentation.edge_f1` (NeFTY App. G.5);
    * ``volume`` / ``volume_gt`` — mask volume fractions.
    """
    g = torch.as_tensor(gt).detach().cpu().double()
    p = torch.as_tensor(pred).detach().cpu().double()
    if tuple(p.shape) != tuple(g.shape):
        p = resample(p, g.shape) if 1 <= g.ndim <= 3 else p.reshape(g.shape)
    if tau is None or tau == "otsu":
        t = otsu_threshold(g)
    elif tau == "half_max":
        t = float(g.min() + 0.5 * (g.max() - g.min()))
    elif isinstance(tau, str):
        raise ConfigError(f"unknown GT threshold {tau!r}; use a float, 'otsu' or 'half_max'")
    else:
        t = float(tau)
    if above == "auto":
        above = bool(float((g > t).double().mean()) <= 0.5)
    mg, mp = _mask(g, t, above), _mask(p, t, above)
    denom = int(mg.sum() + mp.sum())
    frac = float(mg.double().mean())
    t_vm = volume_matched_threshold(p, frac, above)
    dr = float(g.max() - g.min()) if data_range is None else float(data_range)
    return {
        "psnr": float(psnr(p, g, data_range=dr or 1.0)),
        "ssim": float(ssim(p.float(), g.float(), data_range=dr or 1.0)),
        "iou": _iou(mp, mg),
        "dice": 1.0 if denom == 0 else 2.0 * float((mp & mg).sum()) / denom,
        "iou_vm": _iou(_mask(p, t_vm, above), mg),
        "edge_f1": float(edge_f1(p, g, threshold=edge_threshold, dilate=edge_dilate)),
        "volume": float(mp.double().mean()),
        "volume_gt": frac,
        "tau": t,
        "above": float(above),
    }


def _phase_fractions(x: torch.Tensor, thresholds: Sequence[float]) -> list[float]:
    v = x.detach().reshape(-1).double().cpu()
    idx = torch.zeros_like(v, dtype=torch.long)
    for t in thresholds:
        idx += (v > float(t)).long()
    n = max(1, v.numel())
    return [float((idx == j).sum()) / n for j in range(len(thresholds) + 1)]


def transition_band_check(
    smooth: torch.Tensor,
    refined: torch.Tensor,
    values: Sequence[float],
    band: float = 0.25,
    refined_above: Sequence[float] | None = None,
) -> tuple[float, list[dict[str, float]]]:
    """Does every refined interface lie inside the smooth field's transition band?

    For the interface between phases ``j − 1`` and ``j`` (values ``a < b``) the refined volume
    ``V_r = |{x_r > (a + b)/2}|`` is compared with the smooth field's volumes at its
    ``(½ + band)`` and ``(½ − band)`` contrast levels, ``V_core = |{x_s > a + (½ + band)(b − a)}|``
    ≤ ``V_outer = |{x_s > a + (½ − band)(b − a)}|``: the soft transition of the smooth
    reconstruction is exactly where the data leave the interface position uncertain, so a
    refinement may move the interface *within* it, not beyond. Volumes are taken on the minority
    side (complements when the super-level set is the majority) so the deviation is relative to
    the smaller region; a vanishing phase gives deviation 1.

    ``refined_above`` gives the refined super-level volumes ``|{phase ≥ j}|`` directly (e.g. read
    from a level-set function) instead of thresholding ``refined`` at the midpoints.

    Returns:
        ``(deviation, per_interface)``: the largest relative distance of ``V_r`` outside
        ``[V_core, V_outer]`` (0 inside) and ``{"core", "outer", "refined", "deviation"}`` per
        interface (volume fractions).
    """
    xs = smooth.detach().reshape(-1).double().cpu()
    xr = refined.detach().reshape(-1).double().cpu()
    floor = 5.0 / max(xs.numel(), 1)
    worst, rows = 0.0, []
    for j, (a, b) in enumerate(zip(values, values[1:])):
        a, b = float(a), float(b)
        v_core = float((xs > a + (0.5 + band) * (b - a)).double().mean())
        v_out = float((xs > a + (0.5 - band) * (b - a)).double().mean())
        if refined_above is not None:
            v_r = float(refined_above[j])
        else:
            v_r = float((xr > 0.5 * (a + b)).double().mean())
        if v_r > 0.5 and v_out > 0.5:  # measure on the minority side
            v_core, v_out, v_r = 1.0 - v_out, 1.0 - v_core, 1.0 - v_r
        if v_r < v_core:
            dev = (v_core - v_r) / max(v_core, floor)
        elif v_r > v_out:
            dev = (v_r - v_out) / max(v_out, floor)
        else:
            dev = 0.0
        rows.append({"core": v_core, "outer": v_out, "refined": v_r, "deviation": dev})
        worst = max(worst, dev)
    return worst, rows


# ============================================================================================
# report
# ============================================================================================
def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "∞" if v > 0 else "-∞"
        if v != 0 and (abs(v) >= 1e4 or abs(v) < 1e-3):
            return f"{v:.{digits - 1}e}"
        return f"{v:.{digits}g}"
    if isinstance(v, list | tuple):
        return "[" + ", ".join(_fmt(a, digits) for a in v) + "]"
    if isinstance(v, Mapping):
        return "{" + ", ".join(f"{k}: {_fmt(a, digits)}" for k, a in v.items()) + "}"
    return str(v)


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += [
        "| " + " | ".join(c if isinstance(c, str) else _fmt(c) for c in r) + " |" for r in rows
    ]
    return "\n".join(lines)


@dataclass
class RefineReport:
    """Everything :func:`refine_edges` measured and decided (``to_markdown()``, ``to_dict()``).

    Attributes:
        problem / mode / field / representation / render: what was refined, how.
        accepted: whether the refined field passed the acceptance test; ``reason`` says why.
        levels_init / levels_final: phase values before / after the stage (learnable values
            move); ``free_phases``: phases with per-voxel values (their entry is the initial
            representative value).
        thresholds / threshold_method: value-space thresholds used to initialize the phases.
        steps / seconds / settings: cost and settings of the stage (ε schedule, lr, weights, ...).
        fit_before / fit_after: :func:`data_fit` of the smooth and the refined field
            (``rmse``, ``rel_rmse``, ``chi``); ``fit_ratio`` = after / before of the misfit used
            by the test, ``fit_limit`` = the largest misfit accepted; ``sharpening_cost`` =
            ``sqrt(L_final / max(L_best, L_noise))``: the final data loss against the best one
            reached during the stage (its soft part), floored at the noise level;
            ``fit_control``: the data fit of the ``continue`` control with the same budget (run
            when the smooth start is not at the noise floor, ``reference``), else ``None``.
        volume_before / volume_after: phase volume fractions of the smooth and the refined
            field at the midpoints of the final phase values.
        band / band_deviation / band_limit: :func:`transition_band_check` per interface (the
            smooth field's core / outer volumes and the refined volume), the largest relative
            deviation outside the band and the refusal threshold (``max_volume_change``).
        metrics_before / metrics_after: :func:`edge_metrics` (+ instance metrics) against a
            ground truth (``None`` without one).
        notes: dropped loss terms, fallbacks, caveats.
        candidate: the refined :class:`~nefi.solve.Result` (also when refused; not serialized).
    """

    problem: str
    mode: str
    field: str
    representation: str
    render: str
    accepted: bool
    reason: str
    levels_init: list[float]
    levels_final: list[float]
    free_phases: list[int]
    thresholds: list[float]
    threshold_method: str
    steps: int
    seconds: float
    fit_before: dict[str, Any]
    fit_after: dict[str, Any]
    fit_ratio: float
    fit_limit: float
    fit_measure: str
    fit_control: dict[str, Any] | None
    sharpening_cost: float
    sigma: float | None
    volume_before: list[float]
    volume_after: list[float]
    band: list[dict[str, float]]
    band_deviation: float
    band_limit: float
    settings: dict[str, Any] = dc_field(default_factory=dict)
    metrics_before: dict[str, float] | None = None
    metrics_after: dict[str, float] | None = None
    notes: list[str] = dc_field(default_factory=list)
    candidate: Result | None = dc_field(default=None, repr=False, compare=False)

    def summary(self) -> str:
        """One line: verdict, data fit and (with a GT) IoU / Edge-F1 before → after."""
        verdict = "accepted" if self.accepted else "REFUSED"
        lab = "χ" if self.fit_measure == "chi" else "RMSE"
        b, a = _misfit(self.fit_before)[0], _misfit(self.fit_after)[0]
        s = f"refine[{self.mode}] {self.problem}: {verdict} — {lab} {_fmt(b)} → {_fmt(a)}"
        if self.metrics_before and self.metrics_after:
            for k in ("iou", "iou_vm", "edge_f1", "psnr"):
                if k in self.metrics_before and k in self.metrics_after:
                    b, a = self.metrics_before[k], self.metrics_after[k]
                    s += f", {k} {_fmt(b, 3)} → {_fmt(a, 3)}"
        return s + f" ({self.steps} steps, {self.seconds:.1f} s)"

    def rows(self) -> list[list[Any]]:
        """``[quantity, smooth, refined]`` rows of the comparison table."""
        rows: list[list[Any]] = [
            ["data RMSE", self.fit_before.get("rmse"), self.fit_after.get("rmse")],
            ["relative RMSE", self.fit_before.get("rel_rmse"), self.fit_after.get("rel_rmse")],
        ]
        if self.fit_before.get("chi") is not None:
            rows.append(["χ = RMSE/σ", self.fit_before.get("chi"), self.fit_after.get("chi")])
        if self.fit_control is not None:
            m, lab = _misfit(self.fit_control)[0], "χ" if self.fit_measure == "chi" else "RMSE"
            rows.append([f"{lab} of the control (continue, same budget)", "—", m])
        rows.append(["phase volumes", self.volume_before, self.volume_after])
        for j, b in enumerate(self.band):
            rows.append(
                [f"interface {j + 1}: band [core, outer]", [b["core"], b["outer"]], b["refined"]]
            )
        tag = "".join(f", phase {j} free" for j in self.free_phases)
        rows.append([f"phase values{tag}", self.levels_init, self.levels_final])
        if self.metrics_before and self.metrics_after:
            for k, v in self.metrics_before.items():
                if k in ("tau", "volume_gt", "above"):
                    continue
                rows.append([k, v, self.metrics_after.get(k)])
        return rows

    def to_markdown(self) -> str:
        verdict = "accepted" if self.accepted else "**refused** — the smooth result is kept"
        lab = "χ" if self.fit_measure == "chi" else "RMSE"
        lines = [
            f"# Edge refinement — {self.problem} ({self.mode})",
            "",
            f"Verdict: {verdict}. {self.reason}",
            "",
            f"Acceptance: refined {lab} {_fmt(_misfit(self.fit_after)[0])} (× "
            f"{_fmt(self.fit_ratio)} of the smooth {lab}) against the limit "
            f"{_fmt(self.fit_limit)}; sharpening cost {self.sharpening_cost:.2f}× (limit: the "
            f"same tolerance); refined phase volumes within the smooth transition band up to "
            f"{_fmt(self.band_limit)} relative deviation (got {_fmt(self.band_deviation)}).",
            "",
            _md_table(["quantity", "smooth", "refined"], self.rows()),
            "",
            f"Field `{self.field}` · representation `{self.representation}` · render "
            f"`{self.render}` · thresholds {_fmt(self.thresholds)} ({self.threshold_method}) · "
            f"{self.steps} steps, {self.seconds:.1f} s.",
        ]
        if self.settings:
            lines += [
                "",
                "Settings: " + ", ".join(f"{k}={_fmt(v)}" for k, v in self.settings.items()),
            ]
        world = "piecewise-smooth" if self.free_phases else "piecewise-constant"
        notes = [
            *self.notes,
            f"Refinement is a prior (a {world} world): read the data-fit rows — a refined field "
            "that explains the data as well as the smooth one is *consistent* with the "
            "measurement, not proven by it.",
        ]
        lines += ["", "Notes:", ""] + [f"- {n}" for n in notes]
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        d = {
            f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "candidate"
        }
        d["summary"] = self.summary()
        return _jsonable(d)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, str | int | float | bool):
        return obj
    return str(obj)


# ============================================================================================
# the refinement stage
# ============================================================================================
def _field_value_range(problem: Any, name: str) -> tuple[float, float]:
    try:
        return value_range(problem.field.heads[name])
    except Exception:  # pragma: no cover - exotic heads
        return (-math.inf, math.inf)


def _copy_heads(problem: Any, name: str, refined: Head, fixed: Mapping[str, torch.Tensor]) -> Heads:
    """Heads in the problem's field order: ``refined`` for ``name``, fixed buffers otherwise."""
    order = [n for n in problem.field.names if n == name or n in fixed]
    items = [(n, refined if n == name else _FixedHead(fixed[n])) for n in order]
    return Heads(items, primary=name)


def _warm_start_raw(head: Head, x: torch.Tensor) -> torch.Tensor | None:
    """Raw channels reproducing ``x`` through ``head`` (``None`` if the head is not invertible)."""
    try:
        raw = head.inverse(x)
    except NotImplementedError:
        return None
    with torch.no_grad():
        back = head(raw, {}, 1.0)
    tol = 1e-3 * max(float(x.max() - x.min()), 1e-12)
    if not torch.isfinite(back).all() or float((back - x).abs().max()) > tol:
        return None
    return raw


def _data_loss(losses: LossSet, ctx: Context) -> float:
    with torch.no_grad():
        _, comps = losses(ctx)
    return float(losses.data_loss(comps))


def _term_value(term: Loss, ctx: Context) -> float:
    with torch.no_grad():
        return float(term(ctx))


def _merge_history(src: Result, ref: Result) -> tuple[dict[str, list[float]], list[dict]]:
    """Source history followed by the refinement history (stage indices / global steps offset)."""
    n_src = len(src.history.get("total", []))
    n_ref = len(ref.history.get("total", []))
    n_stage = len(src.stage_results)
    g0 = int(max(src.history.get("global_step", [-1]) or [-1])) + 1
    out: dict[str, list[float]] = {}
    for k in list(src.history) + [k for k in ref.history if k not in src.history]:
        a = list(src.history.get(k, [math.nan] * n_src))
        b = list(ref.history.get(k, [math.nan] * n_ref))
        if k == "stage":
            b = [s + n_stage for s in b]
        elif k == "global_step":
            b = [s + g0 for s in b]
        out[k] = a + b
    return out, list(src.stage_results) + list(ref.stage_results)


@dataclass
class _Plan:
    """What a mode builds: the refined field's head, its initial raw channels, extra losses."""

    head: Head
    raw0: torch.Tensor
    sharpen: bool
    settings: dict[str, Any] = dc_field(default_factory=dict)
    notes: list[str] = dc_field(default_factory=list)
    terms: dict[str, Loss] = dc_field(default_factory=dict)
    weights: dict[str, float] = dc_field(default_factory=dict)


def _plan_levelset(
    x_s: torch.Tensor,
    values: Sequence[float],
    taus: Sequence[float],
    *,
    eps: tuple[float, float],
    render: str,
    learn_levels: bool | Sequence[int],
    free_phases: Sequence[int],
    value_range_: tuple[float, float],
    ndim: int,
) -> _Plan:
    """Multi-phase level set: φ₀ = (x_s − τ₁)/(v_{k−1} − v₀), offsets c_j = (τ_j − τ₁)/(…)."""
    scale = values[-1] - values[0]
    offsets = [(t - taus[0]) / scale for t in taus]
    head = MultiPhaseHead(
        values,
        offsets,
        eps_start=eps[0],
        eps_end=eps[1],
        render=render,
        learn=learn_levels,
        free=free_phases,
        value_range=value_range_,
        ndim=ndim,
    )
    u_s = (x_s - values[0]) / scale  # the smooth field in contrast units (free phases)
    raw0 = torch.stack([(x_s - taus[0]) / scale, *([u_s] * len(head.free))], dim=-1)
    settings: dict[str, Any] = {"eps": list(eps), "offsets": offsets}
    if head.free:
        settings["free_phases"] = list(head.free)
    return _Plan(head, raw0, True, settings)


def _plan_continuation(
    problem: Any,
    name: str,
    x_s: torch.Tensor,
    values: Sequence[float],
    *,
    mode: str,
    head_swap: bool,
    swap_eps: tuple[float, float],
    value_range_: tuple[float, float],
) -> _Plan:
    """Grid continuation with the problem's head (warm start through its inverse) or, for
    ``tv_sharpen`` with a bracketed head and two phases, a binarizing level-set head swap."""
    orig = copy.deepcopy(problem.field.heads[name])
    scale = values[-1] - values[0]
    plan = _Plan(orig, x_s.unsqueeze(-1), mode == "phasefield")
    if mode == "tv_sharpen" and head_swap:
        inner = innermost(orig)
        if len(values) == 2 and hasattr(inner, "lo") and hasattr(inner, "hi"):
            plan.head = LevelSetHead(values[0], values[1], swap_eps[0], swap_eps[1])
            # a margin keeps voxels sitting at a phase value out of deep sigmoid saturation
            # (logit ±3.9 instead of ±9): the optimizer can still move them
            u = ((x_s - values[0]) / scale).clamp(0.02, 0.98)
            plan.raw0 = (torch.logit(u) * swap_eps[0]).unsqueeze(-1)
            plan.sharpen = True
            plan.settings["head_swap"] = f"LevelSetHead({_fmt(values[0])}, {_fmt(values[1])})"
            plan.settings["swap_eps"] = list(swap_eps)
            return plan
        why = (
            "more than two phases" if len(values) != 2 else f"{type(inner).__name__} is unbracketed"
        )
        plan.notes.append(f"no binarizing head swap ({why}): TV continuation only")
    raw = _warm_start_raw(orig, x_s)
    if raw is not None:
        plan.raw0 = raw.to(torch.float32)
        return plan
    # a head without an inverse (e.g. GatedSoftplus, masked / normalized wrappers): continue with
    # an invertible head of the same value range, so the field cannot leave it
    lo, hi = value_range_
    if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
        substitute: Head = BoundedHead(lo, hi)
    elif lo == 0.0 and not math.isfinite(hi):
        substitute = SoftplusHead()
    else:
        substitute = Identity()
    raw = _warm_start_raw(substitute, x_s)
    if raw is not None:
        plan.head, plan.raw0 = substitute, raw.to(torch.float32)
        plan.notes.append(
            f"the head {type(orig).__name__} is not invertible: continued with "
            f"{type(substitute).__name__} (same value range {_fmt([lo, hi])})"
        )
        return plan
    plan.notes.append(
        f"the head {type(orig).__name__} is not invertible: continued with an identity head and a "
        "soft range penalty"
    )
    plan.head = Identity()
    lo_r = lo if math.isfinite(lo) else None
    hi_r = hi if math.isfinite(hi) else None
    if lo_r is not None or hi_r is not None:
        plan.terms["range"] = RangePenalty(lo_r, hi_r, field=name)
        plan.weights["range"] = 1.0 / max(abs(scale), 1e-12) ** 2
    return plan


def _add_penalties(
    mode: str,
    name: str,
    terms: dict[str, Loss],
    weights: dict[str, float],
    ctx0: Context,
    l_data0: float,
    *,
    values: Sequence[float],
    gamma: float,
    perimeter: float | str,
    well_ramp: tuple[float, float],
    tv_weight: float | None,
    raw0: torch.Tensor,
) -> dict[str, Any]:
    """Mode penalties scaled to ``γ ×`` the smooth data loss (mutates ``terms`` / ``weights``)."""
    scale = values[-1] - values[0]
    x0 = ctx0.fields[name]
    out: dict[str, Any] = {}

    def relative(energy: float) -> float:
        return gamma * l_data0 / max(energy, 1e-30) if l_data0 > 0 else 0.0

    if mode == "levelset" and perimeter:
        term = TV(name, isotropic=True, eps=1e-3 * abs(scale))
        if perimeter == "auto":  # perimeter energy of the sharp initial phases
            sharp = values[0] + scale * (raw0[..., 0] > 0).to(x0)
            w = relative(
                _term_value(term, dataclasses.replace(ctx0, fields={**ctx0.fields, name: sharp}))
            )
        else:
            w = float(perimeter) / abs(scale)
        terms["perimeter"], weights["perimeter"] = term, w
        out["perimeter_weight"] = w
    elif mode == "phasefield":
        well = DoubleWell(name, values, ramp=well_ramp).to(x0.device)
        w = relative(float(well.energy(x0)))
        terms["double_well"], weights["double_well"] = well, w
        out["double_well_weight"] = w
    elif mode == "tv_sharpen":
        keys = [k for k, t in terms.items() if isinstance(t, TV) and weights[k] > 0]
        if tv_weight is not None:
            if not keys:
                terms["tv"], keys = TV(name, isotropic=True), ["tv"]
            for k in keys:
                weights[k] = float(tv_weight)
        else:
            if not keys:
                terms["tv_sharpen"], weights["tv_sharpen"] = TV(name, isotropic=True), 0.0
                keys = ["tv_sharpen"]
            weights[keys[0]] += relative(_term_value(terms[keys[0]], ctx0))
        out["tv_weights"] = {k: weights[k] for k, t in terms.items() if isinstance(t, TV)}
    return out


def refine_edges(
    problem: Any,
    result: Result,
    *,
    mode: str = "levelset",
    levels: Any = "auto",
    field: str | None = None,
    threshold: Any = "auto",
    quantiles: tuple[float, float] = (0.5, 0.9),
    steps: int | None = None,
    lr: float | None = None,
    representation: str = "grid",
    eps: tuple[float, float] = (0.25, 0.02),
    sharpen_fraction: float = 0.6,
    render: str = "area",
    learn_levels: bool | Sequence[int] = True,
    free_phases: Sequence[int] = (),
    level_lr_mult: float = 0.25,
    perimeter: float | str = 0.0,
    gamma: float | None = None,
    well_ramp: tuple[float, float] = (0.01, 1.0),
    tv_weight: float | None = None,
    head_swap: bool = True,
    swap_eps: tuple[float, float] = (1.0, 0.1),
    drop: Sequence[str] = SMOOTHNESS_LOSSES,
    neural: Mapping[str, Any] | None = None,
    warm_steps: int = 300,
    gt: Any = None,
    metrics: Mapping[str, Callable] | None = None,
    iou_tau: float | str | None = None,
    iou_above: bool | str = "auto",
    data_range: float | None = None,
    sigma: float | None = None,
    fit_tol: float = 1.1,
    reference: str = "auto",
    max_volume_change: float = 0.3,
    band: float = 0.25,
    refuse: bool = True,
    merge_history: bool = False,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float32,
    seed: int | None = 0,
    callbacks: Sequence[Any] = (),
) -> tuple[Result, RefineReport]:
    """Sharpen the interfaces of a smooth reconstruction under the same operator and data.

    A second, short optimization stage (see the module docstring for the modes and their
    equations) that starts from ``result`` and keeps the problem's forward operator, measurement
    and data terms; only the representation / prior of one field changes. Neither the source
    result nor the problem is modified (the operator's trainable nuisance parameters are frozen,
    loss weights live in a private :class:`~nefi.losses.LossSet`).

    Args:
        problem: the :class:`~nefi.problem.InverseProblem` that produced ``result`` (its field
            need not hold the trained weights: the refinement warm-starts from ``result``).
        result: the smooth reconstruction.
        mode: ``"levelset"`` (default), ``"phasefield"``, ``"tv_sharpen"`` or ``"continue"``
            (the control: grid continuation without a sharpening prior).
        levels: ``"auto"`` (two phases), ``k`` (``k`` phases), ``(lo, hi)`` or ``k`` values with
            ``None`` entries estimated (:func:`estimate_levels`).
        field: field to refine (default: the primary field); other fields stay fixed.
        threshold / quantiles: phase initialization (:func:`estimate_levels`).
        steps / lr: stage length and learning rate (defaults: 200 steps at 2e-2 on grids — φ and
            free phase values are in units of the contrast —, 300 steps at 2e-3 for a neural φ).
        representation: ``"grid"`` (default: φ, or the continued field, on the voxel grid) or
            ``"neural"`` (``levelset`` only: φ as a coordinate MLP warm-started by
            ``warm_steps`` regression steps on φ₀; ``neural`` = ``{"hidden", "depth",
            "n_octaves", "activation", "lr"}``).
        eps / sharpen_fraction / render: level-set softness ``(ε_start, ε_end)`` in φ units
            (φ₀ changes by ≈ 1 across an interface; ``ε_start = 0.25`` reproduces the smooth
            transition), the fraction of the stage over which ε (and the double-well ramp /
            head-swap sharpening) progresses — the rest runs at the final value — and
            ``"area"`` (partial volumes) / ``"point"`` rendering (:class:`MultiPhaseHead`).
        learn_levels / level_lr_mult: learnable scalar phase values (bool or indices) and their
            LR multiplier (values move in units of the initial contrast).
        free_phases: ``levelset``: phases whose values stay free per voxel (initialized at the
            smooth field) — crisp interfaces around smooth interiors, e.g. ``(1,)`` for
            structures of varying intensity on a constant background.
        perimeter: ``levelset``: extra interface penalty ``mean|∇u|``,
            ``u = (x − v₀)/(v_{k−1} − v₀)``: an absolute weight, ``"auto"`` (``γ ×`` the smooth
            data loss at the sharp initial phases) or 0 (default: the problem's own TV, kept,
            already penalizes the perimeter of a phase field).
        gamma: strength of the added penalties (double well, extra TV, auto perimeter) as a
            fraction of the reference data loss — the smooth solution's, times ``min(1, 1/χ²)``
            when σ is known (its value at the noise floor) — default ``fit_tol² − 1``, so a
            descent from a converged smooth solution raises the data loss by at most
            ``fit_tol²``.
        well_ramp: ``phasefield`` weight ramp ``(start, end)`` factors.
        tv_weight: ``tv_sharpen``: explicit total TV weight (default: the problem's TV weight plus
            ``γ × L_data / TV`` at the smooth solution).
        head_swap / swap_eps: ``tv_sharpen``: swap a bracketed head for a sharpening
            :class:`~nefi.fields.LevelSetHead` between the two phase values (ε from
            ``swap_eps[0]`` to ``swap_eps[1]`` in logit units).
        drop: loss classes removed from the problem's losses (quadratic smoothness priors).
        gt: ground truth (tensor or fields dict) → ``metrics_before`` / ``metrics_after``.
        metrics: extra ``{name: fn(pred, gt)}`` evaluated before / after (instance metrics).
        iou_tau / iou_above / data_range: GT threshold and orientation of the segmentation
            metrics (:func:`edge_metrics`) and the PSNR range.
        sigma: noise (or model-error) level used for ``χ`` in the acceptance test, overriding
            ``measurement.noise_std`` (not used by the losses).
        fit_tol / reference / max_volume_change / band / refuse: acceptance test — the misfit
            must stay below ``fit_tol × max(χ_ref, 1)`` (``fit_tol × RMSE_ref`` without σ), where
            the reference is the smooth result (``reference="smooth"``) or the better of it and
            the ``continue`` control run with the same budget (``"control"``; ``"auto"``, the
            default: whenever the smooth start is not at the noise floor, ``χ_smooth > fit_tol``,
            or σ is unknown — an under-converged start would make the test vacuous); the
            *sharpening cost* ``sqrt(L_final / max(L_best, L_noise))`` — the final data loss
            against the best one the same stage reached before it sharpened — must stay below
            ``fit_tol``; every refined phase volume must lie within
            ``max_volume_change`` (relative) of the smooth field's transition band
            ``(½ ∓ band)`` (:func:`transition_band_check`). ``refuse=False`` returns the refined
            result even when it fails (flagged).
        merge_history: prepend the source history (stages, steps, time) to the refined result's.
        device / dtype / seed / callbacks: solver settings of the stage.

    Returns:
        ``(result, report)``: the refined :class:`~nefi.solve.Result` (``extra["refined_from"]``
        set) — or, when refused and ``refuse=True``, the unmodified source ``result`` — and the
        :class:`RefineReport` (``report.candidate`` always holds the refined result).
    """
    t_start = time.perf_counter()
    if mode not in REFINE_MODES:
        raise ConfigError(f"unknown refinement mode {mode!r}; use one of {REFINE_MODES}")
    if representation not in ("grid", "neural"):
        raise ConfigError(f"representation must be 'grid' or 'neural', got {representation!r}")
    if representation == "neural" and mode != "levelset":
        raise ConfigError("representation='neural' is available for mode='levelset' only")
    if not fit_tol >= 1.0:
        raise ConfigError(f"fit_tol must be >= 1, got {fit_tol}")
    if reference not in ("auto", "smooth", "control"):
        raise ConfigError(f"reference must be 'auto', 'smooth' or 'control', got {reference!r}")
    name = field or problem.field.primary
    if name not in result.fields:
        raise ConfigError(f"result has no field {name!r}; available: {tuple(result.fields)}")
    if result.fields[name].is_complex():
        raise ConfigError(f"field {name!r} is complex: edge refinement needs a real field")
    x_s = result.fields[name].detach().to("cpu", torch.float32)
    if not torch.isfinite(x_s).all():
        raise ConfigError(f"the smooth field {name!r} contains non-finite values")
    shape = tuple(int(s) for s in x_s.shape)
    ndim = problem.domain.ndim
    if len(shape) != ndim:
        raise ConfigError(f"field {name!r} has shape {shape}; the domain is {ndim}-D")
    missing = [n for n in problem.field.names if n != name and n not in result.fields]
    if missing:
        raise ConfigError(f"result lacks fields {missing} needed to hold them fixed")
    fixed = {n: result.fields[n].detach().cpu() for n in problem.field.names if n != name}
    gamma = float(fit_tol**2 - 1.0) if gamma is None else float(gamma)
    vr = _field_value_range(problem, name)
    dev = resolve_device(device)
    seed_everything(seed)

    # ---- phases, losses and the refined representation --------------------------------------
    values, taus, tmethod = estimate_levels(x_s, levels, threshold, quantiles, vr)
    terms: dict[str, Loss] = {}
    weights: dict[str, float] = {}
    notes: list[str] = []
    for key, term in problem.losses.terms.items():
        if type(term).__name__ in drop and problem.losses.weights[key] != 0.0:
            notes.append(f"dropped the smoothness term {key!r} ({type(term).__name__})")
            continue
        terms[key], weights[key] = term, problem.losses.weights[key]
    if mode == "levelset":
        plan = _plan_levelset(
            x_s, values, taus, eps=eps, render=render, learn_levels=learn_levels,
            free_phases=free_phases, value_range_=vr, ndim=ndim,
        )  # fmt: skip
    else:
        plan = _plan_continuation(
            problem, name, x_s, values, mode=mode, head_swap=head_swap, swap_eps=swap_eps,
            value_range_=vr,
        )  # fmt: skip
    notes += plan.notes
    terms.update(plan.terms)
    weights.update(plan.weights)
    heads = _copy_heads(problem, name, plan.head, fixed)
    settings: dict[str, Any] = {"gamma": gamma, **plan.settings}
    if representation == "grid":
        refine_field: Field = GridField(shape, heads, init=plan.raw0.to(torch.float32))
    else:
        nk = {"hidden": 64, "depth": 3, "n_octaves": 6, "activation": "tanh", **dict(neural or {})}
        warm_lr = float(nk.pop("lr", 5e-3))
        refine_field = NeuralField(
            ndim,
            heads,
            hidden=int(nk["hidden"]),
            depth=int(nk["depth"]),
            skip_at=nk.get("skip_at"),
            activation=str(nk["activation"]),
            n_octaves=int(nk["n_octaves"]),
            annealed=False,
        )
        settings["neural"] = {**nk, "warm_steps": int(warm_steps), "warm_lr": warm_lr}
    rp = dataclasses.replace(
        problem,
        field=refine_field,
        losses=LossSet(dict(terms), weights=dict(weights)),
        postprocess=(),
        curriculum=None,
        name=f"{problem.name}+refine[{mode}]",
        meta={**dict(problem.meta), "refine": mode},
    )
    if problem.postprocess:
        notes.append(
            "the problem's post-processing ("
            + ", ".join(type(p).__name__ for p in problem.postprocess)
            + ") is not applied to the refined field"
        )
    rp.to(dev, dtype)

    # ---- the smooth reference: prediction, data loss, data fit --------------------------------
    dom = problem.domain.at(shape)
    coords = dom.coords(device=dev, dtype=dtype)
    op = rp.operator.at_resolution(shape)
    meas = rp.measurement_at(shape).to(dev, dtype)
    meas_fit = meas if sigma is None else dataclasses.replace(meas, noise_std=float(sigma))
    smooth = {name: x_s, **fixed}
    smooth = {
        n: v.to(dev, dtype) if v.is_floating_point() and not v.is_complex() else v.to(dev)
        for n, v in smooth.items()
    }
    with torch.no_grad():
        pred0 = op(smooth)
    ctx0 = Context(smooth, pred0, meas, dom, op, rp.field, None, 0, 1.0)
    fit_before = data_fit(pred0, meas_fit)
    # reference data loss for the penalty strengths: the smooth solution's, brought down to the
    # noise floor when σ is known (an under-converged start, χ > 1, would otherwise over-weight
    # every penalty by χ² once the fit converges)
    chi0 = fit_before.get("chi")
    l_smooth = _data_loss(rp.losses, ctx0)
    l_data0 = l_smooth * (min(1.0, 1.0 / chi0**2) if chi0 else 1.0)
    settings["reference_data_loss"] = l_data0
    settings.update(
        _add_penalties(
            mode, name, terms, weights, ctx0, l_data0, values=values, gamma=gamma,
            perimeter=perimeter, well_ramp=well_ramp, tv_weight=tv_weight, raw0=plan.raw0.to(dev),
        )
    )  # fmt: skip
    rp.losses = LossSet(dict(terms), weights=dict(weights)).to(dev)

    # ---- neural φ: warm start by regression on the initial raw channels ------------------------
    if representation == "neural":
        target = plan.raw0.to(dev, dtype)
        params = [p for n_, p in refine_field.named_parameters() if not n_.startswith("heads.")]
        opt = torch.optim.Adam(params, lr=settings["neural"]["warm_lr"])
        for _ in range(int(warm_steps)):
            opt.zero_grad(set_to_none=True)
            loss = ((refine_field.raw(coords, 1.0) - target) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            r = refine_field.raw(coords, 1.0)[..., 0] - target[..., 0]
            err = float((r**2).mean().sqrt())
        settings["neural"]["warm_rmse"] = err
        notes.append(f"neural φ warm start: RMSE(φ − φ₀) = {_fmt(err, 3)} (φ₀ spans ≈ 1)")

    # ---- the stage ---------------------------------------------------------------------------
    steps = int(steps if steps is not None else (300 if representation == "neural" else 200))
    lr = float(lr if lr is not None else (2e-3 if representation == "neural" else 2e-2))
    stage = Stage(
        f"refine[{mode}]",
        shape,
        steps,
        lr,
        lr_schedule="cosine",
        lr_min_ratio=0.1,
        anneal=plan.sharpen,
        anneal_fraction=min(max(float(sharpen_fraction), 1e-6), 1.0),
        freeze=("operator.",),
    )
    mult = {"field.heads.": float(level_lr_mult)} if mode == "levelset" else {}
    optim = OptimConfig(optimizer="adam", weight_decay=0.0, grad_clip=None, lr_mult=mult)
    settings.update(steps=steps, lr=lr, sharpen_fraction=sharpen_fraction)
    settings["weights"] = dict(rp.losses.weights)
    solver = Solver(rp, Curriculum([stage], optim), device=dev, dtype=dtype, seed=seed,
                    callbacks=list(callbacks))  # fmt: skip
    refined = solver.run()

    # ---- assessment ---------------------------------------------------------------------------
    x_r = refined.fields[name]
    fit_after = data_fit(refined.pred, meas_fit)
    # sharpening cost: the final data loss against the best one the stage reached (its soft,
    # unsharpened part), floored at the noise level — what the prior costs in data fit
    final = {n: v.to(dev) for n, v in refined.fields.items()}
    ctx_f = Context(final, refined.pred.to(dev), meas, dom, op, rp.field, None, 0, 1.0)
    l_final = _data_loss(rp.losses, ctx_f)
    hist = [float(v) for v in refined.history.get("data_loss", []) if math.isfinite(v)]
    l_best = min([*hist, l_final])
    floor = l_smooth / chi0**2 if chi0 else 0.0
    cost = math.sqrt(l_final / max(l_best, floor, 1e-300)) if l_final > 0 else 1.0
    head = plan.head
    is_ls = isinstance(head, MultiPhaseHead)
    levels_final = [float(v) for v in head.values().detach().cpu()] if is_ls else list(values)
    free = list(head.free) if is_ls else []
    ordered = all(b > a for a, b in zip(levels_final, levels_final[1:]))
    mid = [0.5 * (a + b) for a, b in zip(levels_final, levels_final[1:])]
    # value-based phase volumes (for free phases the φ-support is not identifiable: where the
    # free value approaches the neighbouring phase, moving the support boundary costs nothing)
    vol_before = _phase_fractions(x_s, mid)
    vol_after = _phase_fractions(x_r, mid)
    if ordered:
        deviation, band_rows = transition_band_check(x_s, x_r, levels_final, band)
    else:
        deviation, band_rows = math.inf, []
    m_b, measure = _misfit(fit_before)
    m_a, _ = _misfit(fit_after)
    lab = "χ" if measure == "chi" else "RMSE"
    # the reference misfit: the smooth solution's — or, when it did not reach the noise floor
    # (or σ is unknown), the better of it and the continuation control with the same budget, so
    # an under-converged start cannot make the test vacuous
    fit_control = None
    use_control = reference == "control" or (
        reference == "auto" and mode != "continue" and (measure != "chi" or m_b > fit_tol)
    )
    if use_control:
        _, crep = refine_edges(
            problem, result, mode="continue", field=name, levels=values, steps=steps,
            drop=drop, sigma=sigma, fit_tol=fit_tol, reference="smooth", refuse=False,
            device=dev, dtype=dtype, seed=seed,
        )  # fmt: skip
        fit_control = crep.fit_after
        settings["control_seconds"] = crep.seconds
    m_c = _misfit(fit_control)[0] if fit_control is not None else math.inf
    m_ref = min(m_b, m_c)
    limit = float(fit_tol) * (max(m_ref, 1.0) if measure == "chi" else m_ref)
    ratio = m_a / m_b if m_b > 0 else (1.0 if m_a == 0 else math.inf)
    reasons = []
    if not math.isfinite(m_a) or m_a > limit:
        who = "the continuation control" if m_c < m_b else "the smooth result"
        base = f"max({lab} of {who}, 1)" if measure == "chi" else f"{lab} of {who}"
        reasons.append(
            f"the data fit degrades: {lab} {_fmt(m_b)} → {_fmt(m_a)} exceeds {fit_tol:g} × {base} "
            f"= {_fmt(limit)}"
        )
    if not math.isfinite(cost) or cost > fit_tol:
        reasons.append(
            f"sharpening costs {cost:.2f}× in misfit relative to the best fit of the same stage "
            f"before it sharpened (> {fit_tol:g}): the prior fights the data"
        )
    if not ordered:
        reasons.append("the learned phase values crossed (the phase ordering broke down)")
    elif deviation > max_volume_change:
        reasons.append(
            f"a phase volume leaves the smooth solution's transition band by "
            f"{100 * deviation:.0f} % (> {100 * max_volume_change:.0f} %): the interfaces moved "
            "beyond where the smooth reconstruction placed them"
        )
    accepted = not reasons
    if accepted:
        reason = (
            f"The refined field explains the data within tolerance ({lab} {_fmt(m_b)} → "
            f"{_fmt(m_a)}, limit {_fmt(limit)}; sharpening cost {cost:.2f}×) and its "
            f"interfaces lie within the smooth solution's transition band (deviation "
            f"{100 * deviation:.0f} %)."
        )
    else:
        reason = "Refused: " + "; ".join(reasons) + "."
    m_before = m_after = None
    if gt is not None:
        g = torch.as_tensor(gt[name] if isinstance(gt, Mapping) else gt).detach().cpu()
        kw = {"tau": iou_tau, "above": iou_above, "data_range": data_range}
        m_before, m_after = edge_metrics(x_s, g, **kw), edge_metrics(x_r, g, **kw)
        for key, fn in (metrics or {}).items():
            try:
                m_before[key] = float(fn(x_s, g))
                m_after[key] = float(fn(x_r.to(x_s), g))
            except Exception as e:  # noqa: BLE001 - an optional metric must not abort
                notes.append(f"metric {key!r} failed: {type(e).__name__}: {e}")

    # ---- the refined result -------------------------------------------------------------------
    refined.extra["refined_from"] = {
        "config_hash": result.config_hash,
        "mode": mode,
        "field": name,
        "source_steps": len(result.history.get("total", [])),
        "source_seconds": result.timing.get("total_s"),
    }
    refined.extra["refine_accepted"] = accepted
    refined.timing["refine_s"] = refined.timing.get("total_s")
    if merge_history:
        refined.history, refined.stage_results = _merge_history(result, refined)
        refined.timing["total_s"] = float(result.timing.get("total_s") or 0.0) + float(
            refined.timing.get("refine_s") or 0.0
        )
        refined.timing["per_stage_s"] = [s.get("seconds", 0.0) for s in refined.stage_results]
    report = RefineReport(
        problem=str(problem.name),
        mode=mode,
        field=name,
        representation=representation,
        render=render if is_ls else "point",
        accepted=accepted,
        reason=reason,
        levels_init=[float(v) for v in values],
        levels_final=levels_final,
        free_phases=free,
        thresholds=[float(t) for t in taus],
        threshold_method=tmethod,
        steps=len(solver.history.get("total", [])),
        seconds=time.perf_counter() - t_start,
        fit_before=fit_before,
        fit_after=fit_after,
        fit_ratio=ratio,
        fit_limit=limit,
        fit_measure=measure,
        fit_control=fit_control,
        sharpening_cost=cost,
        sigma=noise_level(meas_fit.noise_std),
        volume_before=vol_before,
        volume_after=vol_after,
        band=band_rows,
        band_deviation=deviation,
        band_limit=float(max_volume_change),
        settings=settings,
        metrics_before=m_before,
        metrics_after=m_after,
        notes=notes,
        candidate=refined,
    )
    refined.extra["refine"] = report.summary()
    log.info("%s", report.summary())
    if not accepted and refuse:
        return result, report
    return refined, report


# ============================================================================================
# instances, runs, benchmarks
# ============================================================================================
def _mean(r: Sequence[float]) -> float:
    return 0.5 * (float(r[0]) + float(r[-1]))


def _thermal_defaults(c: Any) -> dict[str, Any]:
    """NeFTY: defects (``alpha_defect_range``) in a bulk whose diffusivity the data fix."""
    return {
        "levels": (_mean(c.alpha_defect_range), None),
        "iou_tau": float(c.iou_tau),
        "iou_above": False,
        "data_range": float(c.alpha_max - c.alpha_min),
    }


def _ct3d_defaults(c: Any) -> dict[str, Any]:
    """Attenuation phantoms: a crisp air / object boundary around a free (multi-tissue) interior
    (no binarizing head swap for ``tv_sharpen``: the tissues are not two-phase)."""
    return {
        "levels": 2,
        "free_phases": (1,),
        "head_swap": False,
        "iou_tau": float(c.iou_tau),
        "iou_above": True,
    }


def _dot3d_defaults(c: Any) -> dict[str, Any]:
    """Absorbing inclusions (``inclusion_mua``) in a background (``mua_background``): both
    phases start at the configured values and are learned (the grid φ was the robust choice over
    five smoke scenes; an MLP φ gains more on some and fails on others)."""
    return {
        "levels": (float(c.mua_background), _mean(c.inclusion_mua)),
        "iou_tau": "half_max",
        "iou_above": True,
    }


def _deconvolution3d_defaults(c: Any) -> dict[str, Any]:
    """Fluorescence: labelled structures of varying brightness (free) on a dark background."""
    return {"levels": "auto", "free_phases": (1,), "iou_tau": "otsu", "iou_above": True}


def _photoacoustic3d_defaults(c: Any) -> dict[str, Any]:
    """Initial pressure: absorbers of varying amplitude (free) on a zero background."""
    return {
        "levels": (0.0, None),
        "free_phases": (1,),
        "perimeter": "auto",
        "iou_tau": "otsu",
        "iou_above": True,
    }


def _eit_defaults(c: Any) -> dict[str, Any]:
    """Conductive or resistive inclusions (sign unknown): two estimated phases."""
    return {"levels": "auto", "iou_tau": "otsu", "iou_above": "auto"}


def _deconvolution_defaults(c: Any) -> dict[str, Any]:
    """2-D deblurring: objects of varying brightness (free) on a dark background."""
    return {"levels": "auto", "free_phases": (1,), "iou_tau": "otsu", "iou_above": True}


#: Per-instance keyword defaults of :func:`refine_edges` (``mode="levelset"`` unless set):
#: ``name -> fn(cfg) -> kwargs`` — phase values from the instance configuration, free phases for
#: structures of varying intensity, the GT threshold / orientation of the segmentation metrics.
#: Chosen on the smoke presets (see ``docs/refinement.md`` for the numbers).
REFINE_DEFAULTS: dict[str, Callable[[Any], dict[str, Any]]] = {
    "thermal_tomography": _thermal_defaults,
    "ct3d": _ct3d_defaults,
    "dot3d": _dot3d_defaults,
    "deconvolution3d": _deconvolution3d_defaults,
    "photoacoustic3d": _photoacoustic3d_defaults,
    "eit": _eit_defaults,
    "deconvolution": _deconvolution_defaults,
}


def refine_defaults(instance: Any) -> dict[str, Any]:
    """Refinement defaults of an instance: its ``refine_defaults`` attribute (dict or
    ``fn(cfg)``) if set, else :data:`REFINE_DEFAULTS` by name, else ``{}``."""
    own = getattr(instance, "refine_defaults", None)
    cfg = getattr(instance, "cfg", None)
    if callable(own):
        return dict(own(cfg))
    if isinstance(own, Mapping):
        return dict(own)
    fn = REFINE_DEFAULTS.get(str(getattr(instance, "name", "")))
    return dict(fn(cfg)) if fn is not None else {}


def refine_instance(
    instance: Any,
    result: Result,
    measurement: Any,
    *,
    gt: Any = None,
    problem: Any = None,
    **kw: Any,
) -> tuple[Result, RefineReport]:
    """:func:`refine_edges` with the instance's defaults (:func:`refine_defaults`) and metrics.

    Args:
        instance: an :class:`~nefi.instances.Instance`.
        result: its smooth reconstruction of ``measurement``.
        measurement: the measurement (a problem is built with ``instance.build_problem`` unless
            ``problem`` is given).
        gt: optional ground truth (metrics before / after, including ``instance.metrics()``).
        **kw: overrides of the defaults (any :func:`refine_edges` argument); with a ``mode``
            other than ``"levelset"`` the level-set-only defaults (:data:`LEVELSET_ONLY`) are
            dropped.
    """
    defaults = refine_defaults(instance)
    if kw.get("mode", defaults.get("mode", "levelset")) != "levelset":
        defaults = {k: v for k, v in defaults.items() if k not in LEVELSET_ONLY}
    opts = {**defaults, **kw}
    if gt is not None:
        opts.setdefault("gt", gt)
        try:
            opts.setdefault("metrics", {f"inst_{k}": fn for k, fn in instance.metrics().items()})
        except Exception:  # noqa: BLE001 - instance metrics are optional
            pass
    if problem is None:
        problem = instance.build_problem(measurement)
    return refine_edges(problem, result, **opts)


def refine_run_output(
    run: Any,
    *,
    instance: Any = None,
    problem: Any = None,
    **kw: Any,
) -> Any:
    """Refine a finished run: an :class:`~nefi.instances.RunOutput` (pass ``instance``), a
    gallery entry / run (``nefi.viz`` ``GalleryEntry`` / ``GalleryRun``: instance and problem are
    taken from it) or anything with ``result``, ``measurement``, ``gt`` attributes.

    Returns:
        A new :class:`~nefi.instances.RunOutput` whose ``result`` is the refined field (or the
        smooth one when the refinement is refused), ``metrics`` the instance metrics of that
        result, and ``extra``: ``refine_report`` (:class:`RefineReport`), ``smooth_result``,
        ``smooth_metrics``, ``refine_accepted``.

    Example (the gallery hook)::

        refined = refine_run_output(entry)        # entry = nefi.viz.run_instance_smoke(...)
        compare_fields(refined.gt, {"smooth": entry.run.result, "refined": refined.result})
    """
    from ..instances.base import RunOutput

    src = getattr(run, "run", None) or run  # GalleryEntry -> GalleryRun
    if not hasattr(src, "result") or not hasattr(src, "measurement"):
        raise ConfigError(
            f"cannot refine {type(run).__name__}: need a RunOutput or a gallery entry / run with "
            "result and measurement (a failed gallery entry has no run)"
        )
    inst = instance or getattr(src, "instance", None)
    if inst is None:
        raise ConfigError("refine_run_output needs the instance (pass instance=...)")
    prob = problem or getattr(src, "problem", None)
    res, meas, gt = src.result, src.measurement, getattr(src, "gt", None)
    out, report = refine_instance(inst, res, meas, gt=gt, problem=prob, **kw)

    def evaluate(r: Result) -> dict[str, float]:
        if gt is None:
            return {}
        try:
            vals = inst.evaluate(r, gt, meas)
        except TypeError:
            vals = inst.evaluate(r, gt)
        return {k: float(v) for k, v in vals.items()}

    smooth_metrics = dict(getattr(src, "metrics", None) or evaluate(res))
    metrics = evaluate(out) if out is not res else dict(smooth_metrics)
    extra = dict(getattr(src, "extra", None) or {})
    extra.update(
        refine_report=report,
        smooth_result=res,
        smooth_metrics=smooth_metrics,
        refine_accepted=report.accepted,
    )
    return RunOutput(out, metrics, gt, meas, extra)


def refine_method(mode: str | None = None, name: str | None = None, **refine_kw: Any) -> Any:
    """A benchmark :class:`~nefi.bench.protocol.Method`: the instance's own method followed by
    :func:`refine_instance` (defaults from :data:`REFINE_DEFAULTS`, ``refine_kw`` override).

    The returned result carries the merged history (smooth stages + the refinement stage), so the
    benchmark's ``steps`` / ``time_s`` columns include the refinement; refused refinements return
    the smooth result (``extra["refine_accepted"] = False``).

    Example::

        from nefi.bench import run_benchmark
        res = run_benchmark(inst, ["neural", refine_method()], n_samples=2, seeds=(0,))
    """
    from ..bench.protocol import Method, PreparedRun, solve_problem

    opts = dict(refine_kw)
    if mode is not None:
        opts["mode"] = mode
    label = name or ("neural+refine" if mode is None else f"neural+refine[{mode}]")

    def build(instance: Any, measurement: Any) -> Any:
        problem = instance.build_problem(measurement)
        cur = getattr(problem, "curriculum", None) or instance.default_curriculum()

        def runner(prob: Any, curriculum: Curriculum, **solver_kw: Any) -> Result:
            res = solve_problem(prob, curriculum, **solver_kw)
            kw = {"device": solver_kw.get("device", "auto"), "seed": solver_kw.get("seed", 0)}
            out, rep = refine_instance(
                instance, res, measurement, problem=prob, merge_history=True, **{**kw, **opts}
            )
            if out is res:  # refused: keep the smooth result, flag it
                res.extra["refine_accepted"] = False
                res.extra["refine"] = rep.summary()
            return out

        return PreparedRun(problem, cur, {}, runner)

    return Method(label, build, description="instance default + edge refinement")
