"""Coordinate neural field: annealed Fourier features -> MLP with skip -> heads.

Defaults follow NeTMY Table 5 (6 layers × 320, tanh, skip at layer 3, K=12 octaves) and can be set
to NeFTY Table 5 (10 layers × 512, ReLU, skip at 4).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import build, register
from ..utils.compat import autocast_enabled
from .base import Field
from .encoding import Encoding, FourierFeatures
from .heads import Heads


class Sine(nn.Module):
    def __init__(self, w0: float = 30.0) -> None:
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


_ACTS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "softplus": nn.Softplus,
}


@register("field", "neural")
class NeuralField(Field):
    """Coordinate MLP with annealed Fourier features and a mid-network skip connection.

    Args:
        ndim: coordinate dimension of the domain.
        heads: output heads (mapping name -> Head / config) or :class:`Heads`.
        hidden: hidden width.
        depth: number of hidden layers.
        skip_at: index of the hidden layer whose input is concatenated with the encoding
            (``None`` disables the skip).
        activation: ``"tanh"`` | ``"relu"`` | ``"gelu"`` | ``"silu"`` | ``"sine"`` (SIREN).
        encoding: an :class:`Encoding`, a registry config, or ``None`` for Fourier features.
        n_octaves / annealed / include_input: Fourier-feature settings used when ``encoding`` is
        None.
        out_init_scale: scale applied to the output layer's initial weights so the initial field is
            close to the head's ``init_value`` (a near-uniform start, as in the papers).
        out_bias: optional explicit output bias vector (overrides head suggestions).
    """

    def __init__(
        self,
        ndim: int,
        heads: Heads | Mapping | None = None,
        hidden: int = 256,
        depth: int = 6,
        skip_at: int | None = 3,
        activation: str = "tanh",
        encoding: Encoding | dict | str | None = None,
        n_octaves: int = 12,
        annealed: bool = True,
        include_input: bool = True,
        out_init_scale: float = 0.1,
        out_bias: list[float] | None = None,
    ) -> None:
        super().__init__(heads)
        self.ndim = int(ndim)
        if encoding is None:
            self.encoding: Encoding = FourierFeatures(
                self.ndim, n_octaves=n_octaves, annealed=annealed, include_input=include_input
            )
        else:
            self.encoding = build("encoding", encoding, in_dim=self.ndim)
        self.hidden, self.depth = int(hidden), int(depth)
        self.skip_at = skip_at if (skip_at is not None and 0 < skip_at < depth) else None
        self.activation = activation
        self.out_init_scale = float(out_init_scale)
        self._out_bias = None if out_bias is None else torch.tensor(out_bias, dtype=torch.float32)

        e = self.encoding.out_dim
        layers = []
        in_dim = e
        for i in range(self.depth):
            if self.skip_at is not None and i == self.skip_at:
                in_dim = self.hidden + e
            layers.append(nn.Linear(in_dim, self.hidden))
            in_dim = self.hidden
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(self.hidden, self.heads.n_in)
        self.act = Sine() if activation == "sine" else _ACTS[activation]()
        self.reset_parameters()

    # --- init ---------------------------------------------------------------------------
    def reset_parameters(self) -> None:
        for i, layer in enumerate(self.layers):
            fan_in = layer.in_features
            if self.activation == "sine":
                bound = 1.0 / fan_in if i == 0 else math.sqrt(6.0 / fan_in) / 30.0
                nn.init.uniform_(layer.weight, -bound, bound)
            elif self.activation in ("relu", "gelu", "silu"):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.out.weight)
        with torch.no_grad():
            self.out.weight.mul_(self.out_init_scale)
            b = self._out_bias if self._out_bias is not None else self.heads.init_bias()
            self.out.bias.copy_(b.to(self.out.bias.dtype))

    # --- forward ------------------------------------------------------------------------
    def raw(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        shape = coords.shape[:-1]
        x = coords.reshape(-1, self.ndim)
        e = self.encoding(x, progress)
        h = e
        for i, layer in enumerate(self.layers):
            if self.skip_at is not None and i == self.skip_at:
                h = torch.cat([h, e], dim=-1)
            h = self.act(layer(h))
        if autocast_enabled(h.device.type):
            # mixed precision (Solver(autocast=...)): the trunk runs in bf16/fp16, the output
            # projection in full precision so the heads see an unquantized pre-activation
            with torch.autocast(h.device.type, enabled=False):
                out = self.out(h.to(self.out.weight.dtype))
        else:
            out = self.out(h)
        return out.reshape(*shape, self.heads.n_in)

    def extra_repr(self) -> str:
        return (
            f"ndim={self.ndim}, hidden={self.hidden}, depth={self.depth}, skip_at={self.skip_at}, "
            f"activation={self.activation}, heads={self.heads.names}"
        )


def mlp_jacobian_rank_hint(field: NeuralField) -> int:
    """Upper bound on rank(G_θ) = min(|Ω|, P) — see NeTMY Eq. (35)."""
    return field.n_parameters()


__all__ = ["NeuralField", "Sine", "F"]
