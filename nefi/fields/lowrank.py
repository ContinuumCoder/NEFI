"""Low-rank (CP / canonical-polyadic) tensor field.

``x(i_1, …, i_d) = b + Σ_{r<R} Π_a v_r^a(i_a)`` per raw channel, with one learnable factor
vector per axis and rank. Memory is ``R · Σ_a N_a`` instead of ``Π_a N_a`` — e.g. a 256³ volume at
rank 32 needs 25k parameters instead of 16.7M — which makes it a cheap, strongly regularizing
representation for large 3-D problems whose unknown is (approximately) separable (layered media,
axis-aligned structures, smooth backgrounds). Factors are 1-D and are linearly interpolated at the
query coordinates, so the field can be evaluated at any resolution; ``on_stage_start`` resamples
the factors to the stage grid (like :class:`~nefi.fields.GridField`).

Example::

    import nefi
    from nefi.fields import Heads, Softplus
    from nefi.fields.lowrank import LowRankField

    field = LowRankField((64, 64, 32), Heads({"x": Softplus(init_value=0.1)}), rank=8)
    x = field(nefi.Domain.unit((64, 64, 32)).coords())["x"]      # (64, 64, 32)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from ..errors import ConfigError, ShapeError
from ..registry import register
from ..utils.tensor import resample, shape_tuple
from .base import Field
from .heads import Heads

_LETTERS = "abcdefgh"


def _interp_1d(v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate ``v (..., N)`` sampled at cell centers of [-1, 1] at points ``t (M,)``.

    Constant extrapolation beyond the first/last cell center. Returns ``(..., M)``.
    """
    n = v.shape[-1]
    if n == 1:
        return v.expand(*v.shape[:-1], t.shape[0])
    pos = (t + 1.0) * (n / 2.0) - 0.5
    j0 = torch.floor(pos).clamp(0, n - 2)
    frac = (pos - j0).clamp(0.0, 1.0).to(v.dtype)
    j0 = j0.long()
    return v[..., j0] * (1.0 - frac) + v[..., j0 + 1] * frac


@register("field", "lowrank")
class LowRankField(Field):
    """Rank-``R`` CP decomposition on a grid with per-axis factor vectors.

    Args:
        shape: initial grid shape (the factors have these lengths).
        heads: output heads (each raw channel gets its own factors).
        rank: CP rank ``R``.
        bias: learn a constant offset per channel (initialized to the heads' suggested bias).
        init_std: per-factor init std; default makes the rank sum ≈ ``N(0, 0.05²)``.
        resample_on_stage: resample the factors to each curriculum stage's resolution.

    Example::

        f = LowRankField((8, 8), rank=1, bias=False)
        with torch.no_grad():
            f.factors[0].copy_(torch.arange(8.0).view(1, 1, 8))
            f.factors[1].copy_(torch.ones(1, 1, 8))
        x = f(nefi.Domain.unit((8, 8)).coords())["x"]      # outer(arange(8), ones(8))
    """

    def __init__(
        self,
        shape: Sequence[int],
        heads: Heads | Mapping | None = None,
        rank: int = 8,
        bias: bool = True,
        init_std: float | None = None,
        resample_on_stage: bool = True,
    ) -> None:
        super().__init__(heads)
        self.shape = shape_tuple(shape)
        self.ndim = len(self.shape)
        if not 1 <= self.ndim <= len(_LETTERS):
            raise ConfigError(f"LowRankField supports 1-{len(_LETTERS)} dims, got {self.ndim}")
        if rank < 1:
            raise ConfigError("LowRankField needs rank >= 1")
        self.rank = int(rank)
        self.init_std = init_std
        self.resample_on_stage = resample_on_stage
        n = self.heads.n_in
        self.factors = nn.ParameterList(
            [nn.Parameter(torch.empty(n, self.rank, s)) for s in self.shape]
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(n))
        else:
            self.register_buffer("bias", torch.zeros(n))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d, r = self.ndim, self.rank
        std = self.init_std
        if std is None:
            std = (0.05 / r**0.5) ** (1.0 / d)
        with torch.no_grad():
            for f in self.factors:
                f.normal_(0.0, std)
            self.bias.copy_(self.heads.init_bias().to(self.bias))

    # ---- evaluation ---------------------------------------------------------------------------
    def _factor_at(self, a: int, t: torch.Tensor) -> torch.Tensor:
        v = self.factors[a].to(t.dtype)
        n = v.shape[-1]
        if t.shape[0] == n:
            centers = -1.0 + (2.0 * torch.arange(n, device=t.device, dtype=t.dtype) + 1.0) / n
            if torch.allclose(t, centers, atol=1e-6):
                return v
        return _interp_1d(v, t)

    def raw(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        d = self.ndim
        if coords.shape[-1] != d:
            raise ShapeError(f"LowRankField is {d}-D but coordinates have {coords.shape[-1]} dims")
        n = self.heads.n_in
        if coords.ndim == d + 1:
            # tensor-product grid (as produced by Domain.coords): extract the axis coordinates
            ts = []
            for a in range(d):
                idx = [0] * d
                idx[a] = slice(None)
                ts.append(coords[tuple(idx) + (a,)])
            vals = [self._factor_at(a, ts[a]) for a in range(d)]
            letters = _LETTERS[:d]
            expr = ",".join(f"nr{c}" for c in letters) + "->n" + letters
            out = torch.einsum(expr, *vals)
            out = out + self.bias.to(out.dtype).view(n, *([1] * d))
            return out.movedim(0, -1)
        # scattered points (P, d)
        pts = coords.reshape(-1, d)
        prod = None
        for a in range(d):
            va = _interp_1d(self.factors[a].to(pts.dtype), pts[:, a])  # (n, R, P)
            prod = va if prod is None else prod * va
        out = prod.sum(1) + self.bias.to(pts.dtype).view(n, 1)  # (n, P)
        return out.transpose(0, 1).reshape(*coords.shape[:-1], n)

    def on_stage_start(self, stage, domain) -> None:
        if not self.resample_on_stage:
            return
        target = tuple(domain.shape)
        if len(target) != self.ndim:
            raise ShapeError(f"stage grid {target} does not match LowRankField ndim {self.ndim}")
        for a, s in enumerate(target):
            f = self.factors[a]
            if f.shape[-1] != s:
                with torch.no_grad():
                    new = resample(f.detach(), (s,))
                self.factors[a] = nn.Parameter(new.contiguous())

    def dense(self, shape: Sequence[int] | None = None) -> torch.Tensor:
        """Raw tensor ``(*shape, n_in)`` on the cell-centered grid of ``shape``."""
        from ..domain import Domain

        shape = self.shape if shape is None else shape_tuple(shape)
        coords = Domain.unit(shape).coords(device=self.bias.device, dtype=self.bias.dtype)
        return self.raw(coords)

    def compression_ratio(self) -> float:
        """Dense parameter count divided by the CP parameter count."""
        dense = self.heads.n_in
        for f in self.factors:
            dense *= f.shape[-1]
        return dense / max(1, self.n_parameters())

    def extra_repr(self) -> str:
        return (
            f"shape={tuple(f.shape[-1] for f in self.factors)}, rank={self.rank}, "
            f"heads={self.heads.names}"
        )


__all__ = ["LowRankField"]
