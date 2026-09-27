"""Field base class: a parameterization of the unknown(s) queried on a coordinate grid."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch import nn

from .heads import Heads

if TYPE_CHECKING:  # pragma: no cover
    from ..domain import Domain
    from ..solve.curriculum import Stage


class Field(nn.Module):
    """Maps normalized coordinates ``(*shape, ndim)`` to ``{name: Tensor(*shape)}``.

    Subclasses implement :meth:`raw` returning ``(*shape, heads.n_in)``; :class:`Heads` applies the
    physical transforms. ``progress`` in ``[0, 1]`` drives frequency annealing.
    """

    def __init__(self, heads: Heads | Mapping | None) -> None:
        super().__init__()
        self.heads: Heads = heads if isinstance(heads, Heads) else Heads(heads or {"x": "identity"})

    # --- to implement -------------------------------------------------------------------
    def raw(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    # --- API ----------------------------------------------------------------------------
    def forward(self, coords: torch.Tensor, progress: float = 1.0) -> dict[str, torch.Tensor]:
        return self.heads(self.raw(coords, progress), progress)

    @property
    def primary(self) -> str:
        return self.heads.primary

    @property
    def names(self) -> tuple[str, ...]:
        return self.heads.names

    def on_stage_start(self, stage: Stage, domain: Domain) -> None:
        """Hook called by the solver when a curriculum stage starts (e.g. to resample a grid)."""

    def reset_parameters(self) -> None:
        """Re-initialize all parameters (used for multi-restart)."""
        for m in self.modules():
            if m is not self and hasattr(m, "reset_parameters") and not isinstance(m, Field):
                m.reset_parameters()

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
