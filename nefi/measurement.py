"""Observed data container."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from .errors import ShapeError
from .utils.tensor import resample, shape_tuple


@dataclass
class Measurement:
    """An observation produced by the forward operator plus noise.

    Args:
        data: observed tensor (any shape the operator produces).
        mask: optional 1/0 tensor marking observed entries (broadcastable to ``data``).
        noise_std: known noise standard deviation (scalar or tensor). Enables Morozov
            discrepancy stopping and noise-aware weighting when provided.
        meta: free-form metadata (units, acquisition parameters, ...).
    """

    data: torch.Tensor
    mask: torch.Tensor | None = None
    noise_std: float | torch.Tensor | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = torch.as_tensor(self.data)
        if self.mask is not None:
            self.mask = torch.as_tensor(self.mask).to(self.data.dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    def to(self, device=None, dtype=None) -> Measurement:
        data = self.data.to(device=device, dtype=dtype)
        mask = self.mask.to(device=device, dtype=dtype) if self.mask is not None else None
        ns = self.noise_std
        if torch.is_tensor(ns):
            ns = ns.to(device=device, dtype=dtype)
        return Measurement(data, mask, ns, dict(self.meta))

    def resampled(self, shape: Sequence[int], mode: str = "auto") -> Measurement:
        """Resample the trailing dims of ``data`` (and ``mask``) to ``shape``.

        Used to compare against operator outputs at coarse curriculum stages. Extra leading dims
        (e.g. frequency or time) are kept as batch dims; if ``shape`` includes them they must match.

        With a mask, the coarse data are the *observed-weighted* averages of the fine data and the
        coarse mask keeps **fractional weights** (the observed fraction of each coarse cell), so
        sparse observation patterns survive coarsening and :meth:`masked_mean` weights them
        correctly.
        """
        shape = shape_tuple(shape)
        cur = tuple(self.data.shape)
        if cur == shape or cur[len(cur) - len(shape) :] == shape:
            return self
        if len(shape) > len(cur):
            raise ShapeError(
                f"cannot resample measurement of shape {cur} to {shape} (rank differs)"
            )
        if len(shape) < len(cur):
            target = shape  # a trailing shape: leading dims are batch
        else:
            first = min(i for i in range(len(shape)) if cur[i] != shape[i])
            target = shape[first:]  # trailing block that changes; leading dims are batch
        if len(target) > 3:
            raise ShapeError(
                f"resampling {cur} -> {shape} would change {len(target)} trailing dims; at most 3 "
                "are supported (pass InverseProblem(downsample_obs=...) for custom layouts)"
            )
        if self.mask is None:
            data = resample(self.data, target, mode=mode)
            return Measurement(data, None, self.noise_std, dict(self.meta))
        m = self.mask.expand_as(self.data) if self.mask.shape != self.data.shape else self.mask
        w = resample(m, target, mode=mode).clamp(0.0, 1.0)
        num = resample(self.data * m, target, mode=mode)
        data = torch.where(w > 0, num / w.clamp_min(1e-12), torch.zeros_like(num))
        return Measurement(data, w, self.noise_std, dict(self.meta))

    def masked_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Mean of ``x`` over observed entries."""
        if self.mask is None:
            return x.mean()
        m = self.mask.expand_as(x) if self.mask.shape != x.shape else self.mask
        return (x * m).sum() / m.sum().clamp_min(1e-12)

    def energy(self) -> torch.Tensor:
        """Total observed energy ∑ data (masked), used by energy-anchored scale correction."""
        return (self.data * self.mask).sum() if self.mask is not None else self.data.sum()
