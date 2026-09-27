"""Loss composition: a :class:`Context` per step, :class:`Loss` terms, and a weighted
:class:`LossSet`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ..errors import ConfigError
from ..registry import build

if TYPE_CHECKING:  # pragma: no cover
    from ..domain import Domain
    from ..fields.base import Field
    from ..measurement import Measurement
    from ..operators.base import Operator
    from ..solve.curriculum import Stage


@dataclass
class Context:
    """Everything a loss term may look at during one optimization step."""

    fields: dict[str, torch.Tensor]
    pred: torch.Tensor
    obs: Measurement
    domain: Domain
    operator: Operator | None = None
    field_module: Field | None = None
    stage: Stage | None = None
    step: int = 0
    progress: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def field(self, name: str | None = None) -> torch.Tensor:
        if name is None:
            name = (
                self.field_module.primary
                if self.field_module is not None
                else next(iter(self.fields))
            )
        try:
            return self.fields[name]
        except KeyError as e:
            raise ConfigError(f"no field named {name!r}; available: {tuple(self.fields)}") from e


class Loss(nn.Module):
    """A scalar loss term. ``is_data`` marks data-fidelity terms (used for stopping / restarts).

    ``step_dependent`` must be set by terms whose value depends on ``ctx.step`` (e.g. a warm-up
    ramp): ``Solver(compile="step")`` holds the step index constant inside the compiled graph and
    therefore compiles only the field for stages with such a term.
    """

    is_data: bool = False
    step_dependent: bool = False

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self.name = name or type(self).__name__.lower()

    def forward(self, ctx: Context) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


class LossSet(nn.Module):
    """Weighted sum of named loss terms with per-stage weight overrides.

    Args:
        losses: mapping ``name -> Loss`` (or registry configs) or a sequence of losses (named by
            ``loss.name``).
        weights: mapping ``name -> weight`` (missing names default to 1.0; 0 disables a term).
    """

    def __init__(
        self,
        losses: Mapping[str, Loss | dict | str] | Sequence[Loss],
        weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(losses, Mapping):
            items = [(k, build("loss", v)) for k, v in losses.items()]
        else:
            items = [(loss.name, loss) for loss in losses]
        names = [k for k, _ in items]
        if len(set(names)) != len(names):
            raise ConfigError(f"duplicate loss names: {names}")
        self.terms = nn.ModuleDict(dict(items))
        for k, t in items:
            t.name = k
        self.weights: dict[str, float] = {k: 1.0 for k in names}
        if weights:
            for k, w in weights.items():
                if k not in self.weights:
                    raise ConfigError(f"weight for unknown loss {k!r}; known: {names}")
                self.weights[k] = float(w)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.terms.keys())

    def data_terms(self) -> tuple[str, ...]:
        return tuple(k for k, t in self.terms.items() if getattr(t, "is_data", False))

    def with_weights(self, overrides: Mapping[str, float] | None) -> LossSet:
        """Shallow copy sharing the loss modules with (partially) overridden weights."""
        if not overrides:
            return self
        new = LossSet.__new__(LossSet)
        nn.Module.__init__(new)
        new.terms = self.terms
        new.weights = {**self.weights}
        for k, w in overrides.items():
            if k not in new.weights:
                raise ConfigError(f"override for unknown loss {k!r}; known: {self.names}")
            new.weights[k] = float(w)
        return new

    def forward_tensors(self, ctx: Context) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Total loss and *detached tensor* components (no device synchronization).

        The solver uses this and converts all components with a single ``tolist()`` per step;
        :meth:`forward` returns python floats for convenience (one sync per term).
        """
        total = None
        comps: dict[str, torch.Tensor] = {}
        for k, term in self.terms.items():
            w = self.weights[k]
            if w == 0.0:
                continue
            v = term(ctx)
            comps[k] = v.detach()
            total = w * v if total is None else total + w * v
        if total is None:
            raise ConfigError("all loss weights are zero")
        return total, comps

    def forward(self, ctx: Context) -> tuple[torch.Tensor, dict[str, float]]:
        total, comps = self.forward_tensors(ctx)
        if comps:
            vals = torch.stack(list(comps.values())).tolist()
            return total, dict(zip(comps.keys(), vals))
        return total, {}

    def data_loss(self, comps: Mapping[str, float]) -> float:
        """Weighted sum of the data-fidelity components of a ``comps`` dict."""
        return float(sum(self.weights[k] * comps[k] for k in self.data_terms() if k in comps))

    @torch.no_grad()
    def auto_balance(self, ctx: Context, target: float = 1.0, only: Sequence[str] | None = None):
        """Rescale weights so each active term contributes ``target`` at the current iterate.

        A pragmatic starting point when porting to a new problem; the papers use hand-tuned fixed
        weights (NeTMY Tab. 7, NeFTY Tab. 5), which you should still prefer for reported results.
        """
        for k, term in self.terms.items():
            if self.weights[k] == 0.0 or (only is not None and k not in only):
                continue
            v = float(term(ctx))
            if v > 0:
                self.weights[k] = target / v
        return dict(self.weights)
