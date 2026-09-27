"""Coordinate encodings.

:class:`FourierFeatures` implements annealed Fourier positional encoding (Tancik et al. 2020 with
the
Nerfies cosine gate of Park et al. 2021), used by both NeTMY (Eq. 27) and NeFTY (Eq. 21)::

    γ_β(x) = [x, w_k(β) sin(2^k π x), w_k(β) cos(2^k π x)]_{k=0..K-1}
    w_k(β) = (1 - cos(π clip(β - k, 0, 1))) / 2,   β = progress · K

At ``progress=0`` only the raw coordinate passes; at ``progress=1`` all K octaves are active.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..registry import register
from ..utils.compat import is_compiling, is_functorch_wrapped


class Encoding(nn.Module):
    """Base class: maps ``(..., in_dim)`` coordinates to ``(..., out_dim)`` features."""

    in_dim: int
    out_dim: int

    def forward(
        self, coords: torch.Tensor, progress: float = 1.0
    ) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


@register("encoding", "identity")
class IdentityEncoding(Encoding):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim

    def forward(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        return coords


@register("encoding", "fourier")
class FourierFeatures(Encoding):
    """Annealed Fourier features.

    Args:
        in_dim: coordinate dimension.
        n_octaves: number of frequency bands K (NeTMY default 12, NeFTY default 12).
        base: geometric base of the frequencies (2 -> octaves).
        include_input: prepend the raw coordinate.
        annealed: enable the cosine band gate; if False all bands are always active.
        scale: multiplies the frequencies (π for coordinates in [-1, 1]).

    ``progress`` may be a Python float or a 0-dim tensor (the compiled solver paths pass a tensor
    so that annealing never triggers a recompilation; both give identical gates).

    Performance: the features depend only on the coordinates and ``progress`` (no parameters), so
    the eager path memoizes ``[sin(2^k π x), cos(2^k π x)]`` for the coordinate tensor it was last
    called with and the gates for the last ``progress`` values: during annealing a forward costs
    one multiply (plus the input concatenation) instead of recomputing all transcendentals, and
    once ``progress`` stays constant (after annealing) the features are returned from the cache.
    The caches are keyed on the coordinate storage and its version counter (in-place changes
    invalidate them), bypassed for coordinates that require grad and under ``torch.compile``.
    """

    def __init__(
        self,
        in_dim: int,
        n_octaves: int = 12,
        base: float = 2.0,
        include_input: bool = True,
        annealed: bool = True,
        scale: float = math.pi,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_octaves = int(n_octaves)
        self.include_input = include_input
        self.annealed = annealed
        freqs = scale * (base ** torch.arange(self.n_octaves, dtype=torch.float32))
        self.register_buffer("freqs", freqs, persistent=False)
        self.out_dim = (self.in_dim if include_input else 0) + 2 * self.in_dim * self.n_octaves
        self._gate_cache: dict[tuple, torch.Tensor] = {}
        self._sincos_cache: tuple | None = None
        self._feat_cache: tuple | None = None

    # -- gates --------------------------------------------------------------------------------
    def band_weights(self, progress=1.0, device=None, dtype=None) -> torch.Tensor:
        """Gate weights ``w_k(β)`` of shape ``(K,)`` (``progress``: float or 0-dim tensor)."""
        dtype = dtype or torch.float32
        if torch.is_tensor(progress):
            return self._band_weights_tensor(progress, device, dtype)
        caching = not is_compiling()
        key = (float(progress), str(device), dtype)
        if caching:
            hit = self._gate_cache.get(key)
            if hit is not None:
                return hit
        k = torch.arange(self.n_octaves, device=device, dtype=dtype)
        if not self.annealed:
            w = torch.ones_like(k)
        else:
            beta = float(progress) * self.n_octaves
            w = 0.5 * (1.0 - torch.cos(math.pi * torch.clamp(beta - k, 0.0, 1.0)))
        if caching:
            if len(self._gate_cache) >= 8:
                self._gate_cache.clear()
            self._gate_cache[key] = w
        return w

    def _band_weights_tensor(self, progress: torch.Tensor, device, dtype) -> torch.Tensor:
        """Same gates from a tensor ``progress`` without a host synchronization: ``β = progress·K``
        is formed in the progress dtype (float64 → the same double as the float path) and rounded
        to ``dtype`` once, exactly like the Python-scalar operand of the float path."""
        k = torch.arange(self.n_octaves, device=device, dtype=dtype)
        if not self.annealed:
            return torch.ones_like(k)
        beta = (progress.to(device=device) * self.n_octaves).to(dtype)
        return 0.5 * (1.0 - torch.cos(math.pi * torch.clamp(beta - k, 0.0, 1.0)))

    # -- features -----------------------------------------------------------------------------
    def _sincos(self, coords: torch.Tensor) -> torch.Tensor:
        """``[sin(ang), cos(ang)]`` of shape ``(..., K, 2D)``, ``ang = x · 2^k π``."""
        freqs = self.freqs.to(coords.dtype)
        ang = coords.unsqueeze(-2) * freqs.view(-1, 1)  # (..., K, D)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    @staticmethod
    def _key(coords: torch.Tensor) -> tuple:
        return (
            coords.data_ptr(),
            coords._version,
            tuple(coords.shape),
            coords.stride(),
            coords.dtype,
            coords.device,
        )

    def forward(self, coords: torch.Tensor, progress=1.0) -> torch.Tensor:
        if (
            torch.is_tensor(progress)
            or coords.requires_grad
            or is_compiling()
            or is_functorch_wrapped(coords)
        ):
            return self._features(self._sincos(coords), coords, progress)
        key = self._key(coords)
        fc = self._feat_cache
        if fc is not None and fc[1] == key and fc[2] == float(progress) and fc[3]._version == fc[4]:
            return fc[3]
        sc_hit = self._sincos_cache
        if sc_hit is not None and sc_hit[1] == key and sc_hit[2]._version == sc_hit[3]:
            sc = sc_hit[2]
        else:
            sc = self._sincos(coords)
            # hold a reference to ``coords``: its storage (hence data_ptr) cannot be reused
            self._sincos_cache = (coords, key, sc, sc._version)
        feats = self._features(sc, coords, progress)
        self._feat_cache = (coords, key, float(progress), feats, feats._version)
        return feats

    def _features(self, sc: torch.Tensor, coords: torch.Tensor, progress) -> torch.Tensor:
        w = self.band_weights(progress, device=coords.device, dtype=coords.dtype).view(-1, 1)
        feats = (sc * w).flatten(-2)  # (..., K·2D): [w sin, w cos] per band
        if self.include_input:
            feats = torch.cat([coords, feats], dim=-1)
        return feats

    def clear_cache(self) -> None:
        """Drop the memoized features / gates (they are rebuilt on demand)."""
        self._gate_cache.clear()
        self._sincos_cache = None
        self._feat_cache = None

    def _apply(self, fn, recurse=True):
        self.clear_cache()  # .to(device / dtype): cached tensors would be on the old device
        return super()._apply(fn, recurse)

    def __getstate__(self):
        # copies / pickles never inherit the caches (their keys refer to the original storage)
        state = super().__getstate__()
        state.update(_gate_cache={}, _sincos_cache=None, _feat_cache=None)
        return state

    def __setstate__(self, state):
        state.setdefault("_gate_cache", {})
        state.setdefault("_sincos_cache", None)
        state.setdefault("_feat_cache", None)
        super().__setstate__(state)

    def effective_bandwidth(self, progress: float = 1.0) -> float:
        """Approximate highest active frequency (NeTMY Eq. 37), in cycles per unit coordinate."""
        w = self.band_weights(progress)
        active = (w > 1e-3).nonzero()
        if active.numel() == 0:
            return 0.0
        return float(self.freqs[int(active.max())] / (2 * math.pi))
