"""Post-processing applied once after optimization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from ..errors import ConfigError
from ..registry import register

if TYPE_CHECKING:  # pragma: no cover
    from ..problem import InverseProblem


class Postprocess:
    """Maps ``(fields, pred, problem, shape) -> (new_fields, info)``."""

    def __call__(
        self,
        fields: Mapping[str, torch.Tensor],
        pred: torch.Tensor,
        problem: InverseProblem,
        shape: tuple[int, ...],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        raise NotImplementedError


@register("postprocess", "energy_scale")
class EnergyScaleCorrection(Postprocess):
    """Energy-anchored scale correction (NeTMY Eq. 30, Prop. 1, Eq. 21).

    Resolves the scale ambiguity of normalized fidelities: ``x ← α x`` with
    ``α = (E_obs / E_pred)^(1/p)`` where ``E = Σ`` of the (masked) measurement / prediction and
    ``p`` is
    the operator's degree of homogeneity in ``field`` (1 → NeTMY F2, 2 → F1). Exact in the noiseless
    limit when the reconstruction has the right shape; applied one-shot so it never reweights the
    data-fidelity gradient during optimization (NeTMY App. D.5).
    """

    def __init__(self, field: str | None = None, homogeneity: float | None = None) -> None:
        self.field, self.homogeneity = field, homogeneity

    def __call__(self, fields, pred, problem, shape):
        name = self.field or problem.field.primary
        p = self.homogeneity if self.homogeneity is not None else problem.operator.homogeneity
        if p is None:
            raise ConfigError(
                "EnergyScaleCorrection needs the operator's homogeneity degree; set "
                "Operator.homogeneity or pass homogeneity=..."
            )
        obs = problem.measurement.to(pred.device, pred.dtype)
        if tuple(obs.data.shape) != tuple(pred.shape):
            obs = obs.resampled(pred.shape)
        e_obs = obs.energy()
        e_pred = (pred * obs.mask).sum() if obs.mask is not None else pred.sum()
        alpha = (e_obs / e_pred.clamp_min(1e-30)).clamp_min(0.0) ** (1.0 / p)
        out = dict(fields)
        out[name] = fields[name] * alpha
        return out, {
            "scale_factor": float(alpha),
            "energy_obs": float(e_obs),
            "energy_pred": float(e_pred),
        }


@register("postprocess", "clip")
class Clip(Postprocess):
    def __init__(self, field: str | None = None, lo: float | None = 0.0, hi: float | None = None):
        self.field, self.lo, self.hi = field, lo, hi

    def __call__(self, fields, pred, problem, shape):
        name = self.field or problem.field.primary
        out = dict(fields)
        out[name] = fields[name].clamp(self.lo, self.hi)
        return out, {}


@register("postprocess", "threshold_mask")
class ThresholdMask(Postprocess):
    """Add a binary mask field ``<field>_mask = 1{x < thr}`` (``mode="below"``) or ``> thr``."""

    def __init__(
        self,
        threshold: float,
        field: str | None = None,
        mode: str = "below",
        relative: bool = False,
        out_name: str | None = None,
    ) -> None:
        self.threshold, self.field, self.mode, self.relative, self.out_name = (
            threshold,
            field,
            mode,
            relative,
            out_name,
        )

    def __call__(self, fields, pred, problem, shape):
        name = self.field or problem.field.primary
        x = fields[name]
        thr = self.threshold * float(x.max()) if self.relative else self.threshold
        m = (x < thr) if self.mode == "below" else (x > thr)
        out = dict(fields)
        out[self.out_name or f"{name}_mask"] = m.to(x.dtype)
        return out, {"threshold": thr}
