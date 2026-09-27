"""Multiresolution hash-grid encoding (Instant-NGP-lite, Müller et al. 2022).

``L`` levels with resolutions growing geometrically from ``base_resolution`` to
``max_resolution``; each level stores ``F`` learnable features per grid vertex in a table of at most
``T = 2**log2_table_size`` entries (dense indexing when the level fits, spatial hashing otherwise).
A coordinate is encoded by N-linear interpolation of the ``2**d`` surrounding vertex features on
every level, concatenated over levels. Pure ``torch`` (gather + weights), 1-D/2-D/3-D, runs on any
device.

Coarse-to-fine: like the annealed Fourier features, the levels are switched on progressively with
the training ``progress`` (cosine gate per level), so a curriculum starts with a smooth field.

Example::

    import nefi
    from nefi.fields import Heads, NeuralField, Softplus
    from nefi.fields.hashgrid import HashGridEncoding

    enc = HashGridEncoding(2, n_levels=8, log2_table_size=12, max_resolution=64)
    field = NeuralField(2, Heads({"x": Softplus()}), encoding=enc, hidden=64, depth=2)
    x = field(nefi.Domain.unit((32, 32)).coords(), progress=1.0)["x"]
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

from ..errors import ConfigError
from ..registry import register
from .encoding import Encoding
from .heads import Heads
from .neural import NeuralField

# Instant-NGP spatial-hash primes (the first is 1 for memory coherence along the first axis).
_PRIMES = (1, 2654435761, 805459861, 3674653429)


@register("encoding", "hashgrid")
class HashGridEncoding(Encoding):
    """Multiresolution hash-grid encoding with progress-driven level annealing.

    Args:
        in_dim: coordinate dimension (1, 2 or 3; up to 4 supported).
        n_levels: number of resolution levels ``L``.
        n_features: features per level ``F``.
        log2_table_size: ``log2`` of the maximum table size ``T`` per level.
        base_resolution: grid resolution of the coarsest level.
        max_resolution: grid resolution of the finest level (≈ the finest detail you want;
            ~1–2× the field's grid size is a good default).
        include_input: prepend the raw coordinates to the features.
        annealed: switch levels on progressively with ``progress``.
        min_active_levels: levels active at ``progress = 0`` (coarsest first).
        init_scale: features are initialized ``U(-init_scale, init_scale)`` (Instant-NGP: 1e-4).

    Example::

        enc = HashGridEncoding(3, n_levels=6, log2_table_size=14, max_resolution=32)
        feats = enc(torch.rand(100, 3) * 2 - 1, progress=0.5)   # (100, 3 + 6·2)
    """

    def __init__(
        self,
        in_dim: int,
        n_levels: int = 12,
        n_features: int = 2,
        log2_table_size: int = 15,
        base_resolution: int = 4,
        max_resolution: int = 256,
        include_input: bool = True,
        annealed: bool = True,
        min_active_levels: int = 2,
        init_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        d = int(in_dim)
        if not 1 <= d <= len(_PRIMES):
            raise ConfigError(f"HashGridEncoding supports 1-{len(_PRIMES)} dims, got {d}")
        if n_levels < 1 or n_features < 1:
            raise ConfigError("HashGridEncoding needs n_levels >= 1 and n_features >= 1")
        if max_resolution < base_resolution:
            raise ConfigError(
                f"max_resolution ({max_resolution}) must be >= base_resolution ({base_resolution})"
            )
        self.in_dim = d
        self.n_levels = int(n_levels)
        self.n_features = int(n_features)
        self.table_size = 2 ** int(log2_table_size)
        self.include_input = include_input
        self.annealed = annealed
        self.min_active_levels = max(1, min(int(min_active_levels), self.n_levels))
        self.init_scale = float(init_scale)
        if self.n_levels > 1:
            growth = math.exp(
                (math.log(max_resolution) - math.log(base_resolution)) / (self.n_levels - 1)
            )
        else:
            growth = 1.0
        self.growth = growth
        self.resolutions: list[int] = [
            max(1, int(math.floor(base_resolution * growth**lvl + 1e-9)))
            for lvl in range(self.n_levels)
        ]
        self.sizes: list[int] = []
        self.dense: list[bool] = []
        self.offsets: list[int] = []
        total = 0
        for res in self.resolutions:
            n_vert = (res + 1) ** d
            dense = n_vert <= self.table_size
            size = n_vert if dense else self.table_size
            self.offsets.append(total)
            self.sizes.append(size)
            self.dense.append(dense)
            total += size
        self.table = nn.Parameter(torch.empty(total, self.n_features))
        corners = torch.tensor(
            [[(c >> i) & 1 for i in range(d)] for c in range(2**d)], dtype=torch.long
        )
        self.register_buffer("corners", corners, persistent=False)
        self.out_dim = (d if include_input else 0) + self.n_levels * self.n_features
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.table.uniform_(-self.init_scale, self.init_scale)

    # ---- annealing ------------------------------------------------------------------------
    def level_weights(self, progress: float = 1.0, device=None, dtype=None) -> torch.Tensor:
        """Per-level gates ``(L,)``: coarse levels first, all active at ``progress = 1``."""
        lv = torch.arange(self.n_levels, device=device, dtype=dtype or torch.float32)
        if not self.annealed:
            return torch.ones_like(lv)
        m = self.min_active_levels
        beta = m + float(progress) * (self.n_levels - m)
        return 0.5 * (1.0 - torch.cos(math.pi * torch.clamp(beta - lv, 0.0, 1.0)))

    # ---- indexing ---------------------------------------------------------------------------
    def _index(self, c: torch.Tensor, level: int) -> torch.Tensor:
        """Table indices ``(...,)`` for integer vertex coordinates ``c`` ``(..., d)``."""
        res = self.resolutions[level]
        if self.dense[level]:
            idx = c[..., 0].clone()
            stride = res + 1
            for i in range(1, self.in_dim):
                idx = idx + c[..., i] * stride
                stride *= res + 1
            return idx + self.offsets[level]
        h = c[..., 0] * _PRIMES[0]
        for i in range(1, self.in_dim):
            h = torch.bitwise_xor(h, c[..., i] * _PRIMES[i])
        return torch.remainder(h, self.sizes[level]) + self.offsets[level]

    # ---- forward ----------------------------------------------------------------------------
    def forward(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        lead = coords.shape[:-1]
        if coords.shape[-1] != self.in_dim:
            raise ConfigError(
                f"HashGridEncoding built for {self.in_dim}-D coordinates, got {coords.shape[-1]}"
            )
        x = coords.reshape(-1, self.in_dim)
        u = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
        gates = self.level_weights(progress)
        table = self.table.to(x.dtype)
        feats = []
        for lvl, res in enumerate(self.resolutions):
            g = float(gates[lvl])
            if g == 0.0:
                feats.append(x.new_zeros(x.shape[0], self.n_features))
                continue
            pos = u * res
            p0 = torch.floor(pos).clamp(0, res - 1)
            frac = pos - p0
            c = p0.long().unsqueeze(1) + self.corners.unsqueeze(0)  # (P, 2^d, d)
            w = torch.where(
                self.corners.unsqueeze(0).bool(), frac.unsqueeze(1), 1.0 - frac.unsqueeze(1)
            )
            w = w.prod(-1)  # (P, 2^d)
            idx = self._index(c, lvl)  # (P, 2^d)
            f = table[idx]  # (P, 2^d, F)
            feats.append((w.unsqueeze(-1) * f).sum(1) * g)
        out = torch.cat(feats, dim=-1)
        if self.include_input:
            out = torch.cat([x, out], dim=-1)
        return out.reshape(*lead, self.out_dim)

    def n_parameters(self) -> int:
        return self.table.numel()

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, levels={self.n_levels}, features={self.n_features}, "
            f"T=2^{int(math.log2(self.table_size))}, res={self.resolutions[0]}..."
            f"{self.resolutions[-1]}"
        )


@register("field", "hash")
class HashGridField(NeuralField):
    """A small MLP on a :class:`HashGridEncoding` (the Instant-NGP recipe) with nefi heads.

    Identical to ``NeuralField(ndim, heads, encoding=HashGridEncoding(...), hidden, depth)`` but
    also re-initializes the hash tables in :meth:`reset_parameters` (multi-restart).

    Args:
        ndim: coordinate dimension.
        heads: output heads.
        hidden, depth, activation: MLP settings (Instant-NGP uses 64 × 2, ReLU).
        n_levels, n_features, log2_table_size, base_resolution, max_resolution, annealed,
            min_active_levels: see :class:`HashGridEncoding`.
        **neural_kw: other :class:`~nefi.fields.NeuralField` options (``out_init_scale``...).

    Example::

        field = HashGridField(2, Heads({"x": Softplus()}), max_resolution=64, log2_table_size=12)
    """

    def __init__(
        self,
        ndim: int,
        heads: Heads | Mapping | None = None,
        hidden: int = 64,
        depth: int = 2,
        activation: str = "relu",
        n_levels: int = 12,
        n_features: int = 2,
        log2_table_size: int = 15,
        base_resolution: int = 4,
        max_resolution: int = 256,
        annealed: bool = True,
        min_active_levels: int = 2,
        **neural_kw,
    ) -> None:
        enc = HashGridEncoding(
            ndim,
            n_levels=n_levels,
            n_features=n_features,
            log2_table_size=log2_table_size,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            annealed=annealed,
            min_active_levels=min_active_levels,
        )
        neural_kw.setdefault("skip_at", None)
        super().__init__(
            ndim,
            heads,
            hidden=hidden,
            depth=depth,
            activation=activation,
            encoding=enc,
            **neural_kw,
        )

    def reset_parameters(self) -> None:
        super().reset_parameters()
        enc = getattr(self, "encoding", None)
        if isinstance(enc, HashGridEncoding):
            enc.reset_parameters()


__all__ = ["HashGridEncoding", "HashGridField"]
