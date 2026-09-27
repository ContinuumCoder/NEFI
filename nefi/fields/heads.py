"""Output heads: bounded / gated / masked transforms that encode physical priors on the unknowns.

Each :class:`Head` consumes ``n_in`` raw channels of the field output and produces one named
field. Heads may depend on *other* heads' outputs (``depends_on``), e.g. the NeTMY Larmor head is
masked by the density support (Eq. 26).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..errors import ConfigError, ShapeError
from ..registry import build, register


class Head(nn.Module):
    """Base class for output transforms. Subclasses set ``n_in`` and implement ``transform``.

    Heads that change with training progress (e.g. a level-set sharpening schedule) set
    ``uses_progress = True`` and implement ``transform(raw, others, progress)``.
    """

    n_in: int = 1
    depends_on: tuple[str, ...] = ()
    uses_progress: bool = False

    def transform(self, raw: torch.Tensor, others: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        raw: torch.Tensor,
        others: Mapping[str, torch.Tensor] | None = None,
        progress: float = 1.0,
    ):
        if raw.shape[-1] != self.n_in:
            raise ShapeError(
                f"{type(self).__name__} expects {self.n_in} raw channels, got {raw.shape[-1]}"
            )
        if self.uses_progress:
            return self.transform(raw, others or {}, progress)  # type: ignore[call-arg]
        return self.transform(raw, others or {})

    def init_bias(self) -> list[float] | None:
        """Suggested output-layer bias for this head's raw channels (``None`` = zeros)."""
        return None

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        """Map a field value back to raw space where well-defined (used for warm starts)."""
        raise NotImplementedError(f"{type(self).__name__} has no inverse")


@register("head", "identity")
class Identity(Head):
    def transform(self, raw, others):
        return raw[..., 0]

    def inverse(self, value):
        return value.unsqueeze(-1)


@register("head", "softplus")
class Softplus(Head):
    """Non-negative field ``softplus(h) / beta``."""

    def __init__(self, beta: float = 1.0, init_value: float | None = None) -> None:
        super().__init__()
        self.beta = float(beta)
        self.init_value = init_value

    def transform(self, raw, others):
        return F.softplus(raw[..., 0], beta=self.beta)

    def init_bias(self):
        if self.init_value is None:
            return None
        return [self.inverse(torch.tensor(float(self.init_value))).item()]

    def inverse(self, value):
        v = torch.as_tensor(value).clamp_min(1e-12)
        return (torch.log(torch.expm1(self.beta * v)) / self.beta).unsqueeze(-1)


@register("head", "exp")
class Exp(Head):
    """Strictly positive field ``exp(h)`` (log-parameterization; good for large dynamic range)."""

    def transform(self, raw, others):
        return torch.exp(raw[..., 0])

    def inverse(self, value):
        return torch.log(torch.as_tensor(value).clamp_min(1e-30)).unsqueeze(-1)


@register("head", "bounded")
class Bounded(Head):
    """Hard-bracketed field ``lo + (hi - lo) σ(h)`` (NeFTY Eq. 6, NeTMY Larmor band)."""

    def __init__(self, lo: float, hi: float, init_value: float | None = None) -> None:
        super().__init__()
        if not hi > lo:
            raise ConfigError(f"Bounded head needs hi > lo, got {(lo, hi)}")
        self.lo, self.hi = float(lo), float(hi)
        self.init_value = init_value

    def transform(self, raw, others):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(raw[..., 0])

    def init_bias(self):
        if self.init_value is None:
            return None
        return [self.inverse(torch.tensor(float(self.init_value))).item()]

    def inverse(self, value):
        u = ((torch.as_tensor(value) - self.lo) / (self.hi - self.lo)).clamp(1e-6, 1 - 1e-6)
        return torch.logit(u).unsqueeze(-1)


@register("head", "gated_softplus")
class GatedSoftplus(Head):
    """``ρ = softplus(h) · σ(g)`` (NeTMY Eq. 5): non-negative with a pixelwise on/off gate.

    The gate lets the network drive background pixels to ~0 without saturating the softplus,
    which the NeTMY ablation identifies as important for sparse localization.
    """

    n_in = 2

    def __init__(self, beta: float = 1.0, init_value: float | None = None, gate_init: float = 0.0):
        super().__init__()
        self.beta = float(beta)
        self.init_value = init_value
        self.gate_init = float(gate_init)

    def transform(self, raw, others):
        return F.softplus(raw[..., 0], beta=self.beta) * torch.sigmoid(raw[..., 1])

    def gate(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(raw[..., 1])

    def init_bias(self):
        if self.init_value is None:
            return [0.0, self.gate_init]
        target = float(self.init_value) / (1.0 / (1.0 + math.exp(-self.gate_init)))
        h = math.log(math.expm1(self.beta * max(target, 1e-12))) / self.beta
        return [h, self.gate_init]


@register("head", "support_masked")
class SupportMasked(Head):
    """Multiply an inner head by a hard support mask of another field (NeTMY Eq. 26).

    ``mask = 1{ref > tau · max(ref)}`` is treated as a stop-gradient constant, so the inner head
    only
    receives gradients where the reference field is appreciable. Outside the support the value is
    ``fill`` (NeTMY uses 0, which is outside the Larmor band and ignored by support-weighted
    metrics).
    """

    def __init__(
        self, inner: Head | dict | str, depends_on: str, tau: float = 0.3, fill: float = 0.0
    ):
        super().__init__()
        self.inner: Head = build("head", inner)
        self.n_in = self.inner.n_in
        self.depends_on = (depends_on,)
        self.tau = float(tau)
        self.fill = float(fill)

    def support_mask(self, ref: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            thr = self.tau * ref.max()
            return (ref > thr).to(ref.dtype)

    def transform(self, raw, others):
        ref = others.get(self.depends_on[0])
        if ref is None:
            raise ConfigError(
                f"SupportMasked head depends on {self.depends_on[0]!r}, which must be declared "
                "before it in the Heads ordering"
            )
        m = self.support_mask(ref)
        return self.inner(raw, others) * m + self.fill * (1.0 - m)

    def init_bias(self):
        return self.inner.init_bias()

    def inverse(self, value):
        return self.inner.inverse(value)


class Heads(nn.Module):
    """Ordered collection of named heads consuming consecutive slices of a raw output tensor.

    Args:
        heads: mapping ``name -> Head`` (or registry config), applied in insertion order.
        primary: name of the primary field (defaults to the first head).
    """

    def __init__(
        self,
        heads: Mapping[str, Head | dict | str] | Sequence[tuple[str, Head]],
        primary: str | None = None,
    ) -> None:
        super().__init__()
        items = list(heads.items()) if isinstance(heads, Mapping) else list(heads)
        if not items:
            raise ConfigError("Heads needs at least one head")
        self._heads = nn.ModuleDict(OrderedDict((k, build("head", v)) for k, v in items))
        self.names: tuple[str, ...] = tuple(k for k, _ in items)
        self.primary: str = primary or self.names[0]
        if self.primary not in self.names:
            raise ConfigError(f"primary {self.primary!r} not among heads {self.names}")
        self.slices: dict[str, slice] = {}
        start = 0
        for k in self.names:
            n = self._heads[k].n_in
            self.slices[k] = slice(start, start + n)
            start += n
        self.n_in = start
        # validate dependencies
        seen: set[str] = set()
        for k in self.names:
            for dep in self._heads[k].depends_on:
                if dep not in seen:
                    raise ConfigError(
                        f"head {k!r} depends on {dep!r} which is not declared before it"
                    )
            seen.add(k)

    def __getitem__(self, name: str) -> Head:
        return self._heads[name]

    def __len__(self) -> int:
        return len(self.names)

    def items(self):
        return [(k, self._heads[k]) for k in self.names]

    def forward(self, raw: torch.Tensor, progress: float = 1.0) -> dict[str, torch.Tensor]:
        if raw.shape[-1] != self.n_in:
            raise ShapeError(f"Heads expect {self.n_in} raw channels, got {raw.shape[-1]}")
        out: dict[str, torch.Tensor] = {}
        for k in self.names:
            out[k] = self._heads[k](raw[..., self.slices[k]], out, progress)
        return out

    def init_bias(self) -> torch.Tensor:
        """Concatenated suggested output biases (zeros where a head has no preference)."""
        b = torch.zeros(self.n_in)
        for k in self.names:
            ib = self._heads[k].init_bias()
            if ib is not None:
                b[self.slices[k]] = torch.tensor(ib, dtype=b.dtype)
        return b

    def inverse(self, fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Raw tensor reproducing ``fields`` where heads are invertible (else zeros)."""
        ref = next(iter(fields.values()))
        raw = torch.zeros(*ref.shape, self.n_in, device=ref.device, dtype=ref.dtype)
        for k in self.names:
            if k in fields:
                try:
                    raw[..., self.slices[k]] = self._heads[k].inverse(fields[k])
                except NotImplementedError:
                    pass
        return raw
