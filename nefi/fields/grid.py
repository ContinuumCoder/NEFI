"""Free pixel/voxel parameterization (the "Tikhonov" / "Grid Opt." baseline of both papers).

Because the parameterization Jacobian is the identity (G_θ = I), a grid field executes the raw
field-space gradient verbatim (NeTMY §4.4) — this is the reference point for the filtering-view
diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from ..registry import register
from ..utils.tensor import resample, shape_tuple
from .base import Field
from .heads import Heads


@register("field", "grid")
class GridField(Field):
    """Raw values stored on a grid ``(*shape, n_in)``; heads applied pointwise.

    Args:
        shape: grid shape.
        heads: output heads.
        init: initial raw value(s): scalar, per-channel list, or a tensor ``(*shape, n_in)``;
            ``None`` uses the heads' suggested biases (e.g. mid-range for :class:`Bounded`).
        resample_mode: interpolation used when the curriculum changes resolution.
    """

    def __init__(
        self,
        shape: Sequence[int],
        heads: Heads | Mapping | None = None,
        init: float | Sequence[float] | torch.Tensor | None = None,
        resample_mode: str = "auto",
    ) -> None:
        super().__init__(heads)
        self.shape = shape_tuple(shape)
        self.resample_mode = resample_mode
        self._init = init
        self.param = nn.Parameter(self._initial(self.shape))

    def _initial(self, shape: tuple[int, ...]) -> torch.Tensor:
        n = self.heads.n_in
        if torch.is_tensor(self._init):
            t = self._init.detach().clone().float()
            if t.shape[-1] != n or tuple(t.shape[:-1]) != shape:
                t = resample(t.movedim(-1, 0), shape).movedim(0, -1)
            return t
        if self._init is None:
            base = self.heads.init_bias()
        elif isinstance(self._init, int | float):
            base = torch.full((n,), float(self._init))
        else:
            base = torch.tensor(list(self._init), dtype=torch.float32)
        return base.expand(*shape, n).clone()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.param.copy_(self._initial(tuple(self.param.shape[:-1])).to(self.param))

    def raw(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        shape = tuple(coords.shape[:-1])
        if shape == tuple(self.param.shape[:-1]):
            return self.param
        # evaluate at another resolution (differentiable)
        return resample(self.param.movedim(-1, 0), shape, mode=self.resample_mode).movedim(0, -1)

    def on_stage_start(self, stage, domain) -> None:
        if tuple(domain.shape) != tuple(self.param.shape[:-1]):
            with torch.no_grad():
                new = resample(self.param.movedim(-1, 0), domain.shape, mode=self.resample_mode)
            self.param = nn.Parameter(new.movedim(0, -1).contiguous())

    def set_fields(self, fields: Mapping[str, torch.Tensor]) -> None:
        """Warm start from field values (through head inverses where available)."""
        with torch.no_grad():
            raw = self.heads.inverse(fields).to(self.param)
            if tuple(raw.shape[:-1]) != tuple(self.param.shape[:-1]):
                raw = resample(raw.movedim(-1, 0), self.param.shape[:-1]).movedim(0, -1)
            self.param.copy_(raw)

    def extra_repr(self) -> str:
        return f"shape={tuple(self.param.shape[:-1])}, heads={self.heads.names}"
