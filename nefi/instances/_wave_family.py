"""Private helpers shared by the wave / optics / reaction–diffusion instances.

Kept out of the core so the instances stay self-contained: a trailing-dims measurement downsampler
for coarse curriculum stages, a closed-form ("direct") baseline wrapper that plugs a classical
reconstruction into the generic benchmark machinery, mean-subtracted metrics, and array layouts for
sources/receivers.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch

from ..domain import Domain
from ..fields import GridField, Heads
from ..measurement import Measurement
from ..metrics.basic import psnr, ssim
from ..problem import InverseProblem
from ..solve.curriculum import Curriculum, Stage
from ..utils.tensor import resample, shape_tuple

__all__ = [
    "direct_baseline_problem",
    "line_positions",
    "mean_subtracted",
    "phase_psnr",
    "phase_ssim",
    "ring_positions",
    "spatial_downsample_obs",
    "uniform_like",
]


def spatial_downsample_obs(
    n_spatial: int,
) -> Callable[[Measurement, tuple[int, ...]], Measurement]:
    """``downsample_obs`` hook resampling only the trailing ``n_spatial`` dims of the data.

    Used for operators whose output is ``(channels, *field_shape)`` (holography, reaction–diffusion)
    so coarse curriculum stages compare against area-averaged observations, whatever the number of
    leading channel dims.
    """

    def fn(meas: Measurement, field_shape: tuple[int, ...]) -> Measurement:
        shape = shape_tuple(field_shape)[-n_spatial:]
        if tuple(meas.data.shape[-n_spatial:]) == shape:
            return meas
        data = resample(meas.data, shape)
        mask = None
        if meas.mask is not None:
            m = meas.mask.expand_as(meas.data) if meas.mask.shape != meas.data.shape else meas.mask
            mask = (resample(m, shape) > 0.5).to(meas.data.dtype)
        return Measurement(data, mask, meas.noise_std, dict(meas.meta))

    return fn


def direct_baseline_problem(
    problem: InverseProblem,
    heads: Heads,
    reconstruct: Callable[[Measurement, Domain], torch.Tensor],
    name: str,
) -> tuple[InverseProblem, Curriculum]:
    """Wrap a closed-form reconstruction as a ``(problem, curriculum)`` baseline.

    The problem carries ``meta = {"solver": "direct", "reconstruct": fn, "baseline": name}`` so
    :func:`nefi.baselines.solve` dispatches it to the closed-form solver; its field is a
    :class:`~nefi.fields.GridField` warm-started at the reconstruction and the curriculum is a
    single zero-learning-rate step, so running it with the plain :class:`~nefi.solve.Solver` also
    returns the reconstruction unchanged (projected onto the head's range).
    """
    shape = tuple(problem.domain.shape)
    field = GridField(shape, heads)
    with torch.no_grad():
        rec = reconstruct(problem.measurement, problem.domain)
        field.set_fields({heads.primary: torch.as_tensor(rec, dtype=torch.float32).cpu()})
    cur = Curriculum([Stage("direct", shape, 1, 0.0, lr_schedule="constant", anneal=False)])
    meta = dict(problem.meta)
    meta.update({"solver": "direct", "reconstruct": reconstruct, "baseline": name})
    new = InverseProblem(
        problem.domain,
        field,
        problem.operator,
        problem.losses,
        problem.measurement,
        postprocess=list(problem.postprocess),
        curriculum=cur,
        downsample_obs=problem.downsample_obs,
        name=f"{problem.name}-{name}",
        meta=meta,
    )
    return new, cur


def mean_subtracted(metric: Callable[..., float]) -> Callable[..., float]:
    """Metric evaluated on ``pred − mean(pred)`` vs ``gt − mean(gt)`` (global-offset invariant)."""

    def fn(pred, gt, **kw) -> float:
        p = torch.as_tensor(pred).detach().double()
        g = torch.as_tensor(gt).detach().double().to(p.device)
        return metric(p - p.mean(), g - g.mean(), **kw)

    fn.__name__ = f"mean_subtracted_{getattr(metric, '__name__', 'metric')}"
    return fn


phase_psnr = mean_subtracted(psnr)
phase_ssim = mean_subtracted(ssim)


def uniform_like(gt: torch.Tensor, value: float) -> torch.Tensor:
    """The spatially uniform initial field (reference point for "better than the start")."""
    return torch.full_like(torch.as_tensor(gt, dtype=torch.float32), float(value))


def line_positions(
    n: int, fixed_axis: int, fixed_value: float, lo: float, hi: float
) -> torch.Tensor:
    """``n`` points evenly spread (cell-centred) along ``[lo, hi]`` on the line
    ``x_{fixed_axis} = fixed_value`` in 2-D: ``(n, 2)`` float64."""
    t = lo + (hi - lo) * (torch.arange(n, dtype=torch.float64) + 0.5) / n
    fixed = torch.full_like(t, float(fixed_value))
    cols = [fixed, t] if fixed_axis == 0 else [t, fixed]
    return torch.stack(cols, dim=1)


def ring_positions(n: int, center: Sequence[float], radius: float, phase: float = 0.0):
    """``n`` points on a circle: ``(n, 2)`` float64."""
    a = phase + 2.0 * math.pi * torch.arange(n, dtype=torch.float64) / n
    return torch.stack([center[0] + radius * torch.cos(a), center[1] + radius * torch.sin(a)], 1)
