"""Spectral representations: explicit Fourier bases and Fourier-domain preconditioning.

**FourierBasisField** — ``x(u) = Σ_k c_k φ_k(u)`` with a few low-frequency modes (cosine/DCT basis
by default: complete on ``[-1, 1]^d``, no periodicity assumption, zero normal derivative at the
boundary). Its update kernel is exactly the projector onto the retained modes,
``G = Φ Φᵀ`` (NeTMY Lemma 2 with ``J_θ = Φ``): an ideal low-pass filter with a *hard* bandwidth.
It is the natural "smooth background" component of a composite field, and the modes can be
annealed shell by shell (``progress``) or grown on demand (:meth:`FourierBasisField.grow`).

**SpectralPreconditionedField** — wraps any field ``f_θ`` and applies a fixed (or low-parametric
learnable) radial Fourier gain to its raw output, ``x = P f_θ``, ``P = F⁻¹ diag(g) F``. Then

    J_x = P J_θ,     Δx ≈ −η P G_θ Pᵀ ∇_x L,

so gradient descent on ``θ`` realizes a *preconditioned* image-space step. For a free grid
(``G_θ = I``) and a convolution operator with transfer ``a(k)``, ``g = 1/max(|a|, floor)`` turns the
Landweber step ``Aᵀ(Ax − y)`` into a (Tikhonov-floored) Gauss–Newton step: all resolvable
frequencies converge at the same rate (the classic Fourier preconditioner, fused with a neural
field). The floor is exactly where the data stop resolving (NeFTY Cor. 1: ``1/σ_n`` noise
amplification) — pick it from :func:`nefi.fields.adaptive.match_report`.
Boundary handling uses the even (half-sample symmetric) extension, i.e. DCT-II filtering, so a
non-periodic field does not wrap around.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from ...errors import ConfigError, ShapeError
from ...registry import register
from ...utils.tensor import shape_tuple
from ..base import Field
from ..heads import Heads
from ._utils import grid_axes

GainFn = Callable[[torch.Tensor], torch.Tensor]


# ------------------------------------------------------------------------------------------
# FourierBasisField
# ------------------------------------------------------------------------------------------
@register("field", "fourier_basis")
class FourierBasisField(Field):
    """Explicit low-frequency basis with learnable coefficients (the smooth-background prior).

    Bases (per axis, ``u ∈ [-1, 1]``):

    * ``"cosine"`` (default): ``φ_k(u) = cos(π k (u + 1) / 2)``, ``k = 0 … K−1`` (DCT-II modes;
      frequency ``k/4`` cycles per unit);
    * ``"fourier"``: ``1, cos(π m u), sin(π m u)`` for ``m = 1 …`` (periodic; frequency ``m/2``).

    Multi-dimensional modes are tensor products; the *shell* of a mode is its largest per-axis
    harmonic index. With ``annealed=True`` shells open one by one with the cosine gate of the
    annealed Fourier features (NeTMY Eq. 27): at ``progress = 0`` only the mean is free.
    ``n_active`` limits the open shells (capacity pool for :meth:`grow`).

    Args:
        ndim: coordinate dimension.
        n_modes: basis size per axis (int or per-axis tuple).
        heads: heads (default identity ``"x"``); the constant mode starts at the heads' suggested
            initial raw value.
        basis: ``"cosine"`` | ``"fourier"``.
        annealed: gate shells by progress.
        n_active: number of open shells including the constant (default: all).
        init_std: std of random initial non-constant coefficients (default 0: a constant field).

    Example::

        bg = FourierBasisField(2, n_modes=6)            # 36 coefficients, bandwidth 1.25 cyc/unit
        x = bg(nefi.Domain.unit((32, 32)).coords())["x"]
    """

    def __init__(
        self,
        ndim: int,
        n_modes: int | Sequence[int] = 8,
        heads: Heads | Mapping | None = None,
        basis: str = "cosine",
        annealed: bool = True,
        n_active: int | None = None,
        init_std: float = 0.0,
    ) -> None:
        super().__init__(heads)
        self.ndim = int(ndim)
        k = shape_tuple(n_modes)
        if len(k) == 1 and self.ndim > 1:
            k = k * self.ndim
        if len(k) != self.ndim or min(k) < 1:
            raise ConfigError(f"n_modes must give {self.ndim} positive sizes, got {n_modes}")
        if basis not in ("cosine", "fourier"):
            raise ConfigError(f"basis must be 'cosine' or 'fourier', got {basis!r}")
        self.n_modes, self.basis, self.annealed = k, basis, bool(annealed)
        self.init_std = float(init_std)
        harm = [self._harmonics(n) for n in k]
        mesh = torch.meshgrid(*harm, indexing="ij") if self.ndim > 1 else (harm[0],)
        shell = torch.stack(mesh, dim=0).amax(dim=0) if self.ndim > 1 else harm[0]
        self.register_buffer("shell", shell.long(), persistent=False)
        self.n_levels = int(shell.max())
        self._n_active_init = self.n_levels + 1 if n_active is None else int(n_active)
        if not 1 <= self._n_active_init <= self.n_levels + 1:
            raise ConfigError(f"n_active must be in [1, {self.n_levels + 1}]")
        self.register_buffer("n_active", torch.tensor(self._n_active_init))
        self.coef = nn.Parameter(torch.zeros(*k, self.heads.n_in))
        self.reset_parameters()

    def _harmonics(self, n: int) -> torch.Tensor:
        j = torch.arange(n)
        return j if self.basis == "cosine" else (j + 1) // 2

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.init_std > 0:
                self.coef.normal_(0.0, self.init_std)
            else:
                self.coef.zero_()
            self.coef[(0,) * self.ndim] = self.heads.init_bias().to(self.coef)
            self.n_active.fill_(self._n_active_init)

    # --- spectral bookkeeping ----------------------------------------------------------------
    @property
    def cycles_per_shell(self) -> float:
        """Frequency increment per shell (cycles per unit normalized coordinate)."""
        return 0.25 if self.basis == "cosine" else 0.5

    def band_frequencies(self) -> list[float]:
        """Frequencies of shells ``1 … n_levels`` (for bandwidth ↔ progress mapping)."""
        return [s * self.cycles_per_shell for s in range(1, self.n_levels + 1)]

    def shell_weights(self, progress: float = 1.0) -> torch.Tensor:
        """Gate per shell ``(n_levels + 1,)``: constant always on, shell ``s`` ramps for
        ``β ∈ [s − 1, s]``, ``β = progress · n_levels``; shells ``≥ n_active`` closed."""
        s = torch.arange(self.n_levels + 1, dtype=torch.float32, device=self.coef.device)
        if self.annealed and self.n_levels > 0:
            beta = float(progress) * self.n_levels
            w = 0.5 * (1.0 - torch.cos(math.pi * torch.clamp(beta - s + 1.0, 0.0, 1.0)))
            w[0] = 1.0
        else:
            w = torch.ones_like(s)
        return w * (s < self.n_active.to(s.device)).to(w)

    def effective_bandwidth(self, progress: float = 1.0) -> float:
        """Highest open frequency (cycles per unit normalized coordinate)."""
        w = self.shell_weights(progress)
        on = (w > 1e-3).nonzero()
        return 0.0 if on.numel() == 0 else float(on.max()) * self.cycles_per_shell

    def mode_weights(self, progress: float = 1.0) -> torch.Tensor:
        return self.shell_weights(progress)[self.shell]

    def basis_1d(self, u: torch.Tensor, n: int) -> torch.Tensor:
        """Per-axis basis matrix ``(len(u), n)``."""
        j = torch.arange(n, device=u.device, dtype=u.dtype)
        if self.basis == "cosine":
            return torch.cos(math.pi * j * (u.unsqueeze(-1) + 1.0) * 0.5)
        m = torch.div(j + 1, 2, rounding_mode="floor")
        ang = math.pi * m * u.unsqueeze(-1)
        out = torch.where((j % 2) == 1, torch.cos(ang), torch.sin(ang))
        return torch.where(j == 0, torch.ones_like(out), out)

    # --- evaluation ----------------------------------------------------------------------
    def raw(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        if coords.shape[-1] != self.ndim:
            raise ShapeError(f"FourierBasisField expects {self.ndim}-D coordinates")
        w = self.mode_weights(progress).to(coords)
        coef = self.coef.to(coords) * w.unsqueeze(-1)
        axes = grid_axes(coords)
        letters = "abcdefgh"[: self.ndim]
        if axes is not None:  # separable fast path
            mats = [self.basis_1d(axes[i], self.n_modes[i]) for i in range(self.ndim)]
            outs = "ijklmnop"[: self.ndim]
            expr = ",".join(f"{o}{a}" for o, a in zip(outs, letters))
            return torch.einsum(f"{expr},{letters}z->{outs}z", *mats, coef)
        pts = coords.reshape(-1, self.ndim)
        mats = [self.basis_1d(pts[:, i], self.n_modes[i]) for i in range(self.ndim)]
        expr = ",".join(f"p{a}" for a in letters)
        out = torch.einsum(f"{expr},{letters}z->pz", *mats, coef)
        return out.reshape(*coords.shape[:-1], self.heads.n_in)

    def sobolev_norm(self, s: float = 1.0) -> torch.Tensor:
        """``Σ_k (1 + |k|²)^s |c_k|²`` over non-constant modes (a smoothness penalty)."""
        harm = [self._harmonics(n).to(self.coef) for n in self.n_modes]
        k2 = sum(
            h.view(*([1] * i), -1, *([1] * (self.ndim - i - 1))) ** 2 for i, h in enumerate(harm)
        )
        wgt = (1.0 + k2) ** s
        wgt = wgt.clone()
        wgt[(0,) * self.ndim] = 0.0
        return (wgt.unsqueeze(-1) * self.coef**2).sum()

    # --- capacity growth -----------------------------------------------------------------
    def can_grow(self) -> bool:
        return int(self.n_active) <= self.n_levels

    @torch.no_grad()
    def grow(self, hint: Mapping | None = None) -> bool:
        """Open the next shell of modes (coefficients start at their current values, zero by
        default, so the field is unchanged at the moment of growth)."""
        if not self.can_grow():
            return False
        self.n_active.add_(1)
        return True

    def capacity(self) -> dict[str, int]:
        return {"shells": int(self.n_active), "max_shells": self.n_levels + 1}

    def extra_repr(self) -> str:
        return (
            f"ndim={self.ndim}, n_modes={self.n_modes}, basis={self.basis!r}, "
            f"annealed={self.annealed}, heads={self.heads.names}"
        )


# ------------------------------------------------------------------------------------------
# gains
# ------------------------------------------------------------------------------------------
def profile_gain(freqs: Sequence[float] | torch.Tensor, values: Sequence[float] | torch.Tensor):
    """Radial gain from a sampled profile (piecewise linear, constant beyond the ends)."""
    f = torch.as_tensor(freqs, dtype=torch.float64).flatten()
    v = torch.as_tensor(values, dtype=torch.float64).flatten()
    if f.numel() != v.numel() or f.numel() < 1:
        raise ConfigError("freqs and values must be non-empty and of equal length")
    order = torch.argsort(f)
    f, v = f[order], v[order]

    def gain(kr: torch.Tensor) -> torch.Tensor:
        ff, vv = f.to(kr.device, kr.dtype), v.to(kr.device, kr.dtype)
        if ff.numel() == 1:
            return vv.expand_as(kr).clone()
        x = kr.clamp(float(ff[0]), float(ff[-1]))
        idx = torch.searchsorted(ff, x.contiguous().reshape(-1)).clamp(1, ff.numel() - 1)
        idx = idx.reshape(x.shape)
        x0, x1, v0, v1 = ff[idx - 1], ff[idx], vv[idx - 1], vv[idx]
        t = (x - x0) / (x1 - x0).clamp_min(1e-12)
        return v0 + t * (v1 - v0)

    return gain


def inverse_sensitivity_gain(
    freqs: Sequence[float] | torch.Tensor,
    sensitivity: Sequence[float] | torch.Tensor,
    floor: float = 1e-2,
    power: float = 1.0,
) -> GainFn:
    """``g(k) = (σ_max / max(σ(k), floor · σ_max))^power`` — a floored inverse of the operator's
    amplitude sensitivity ``σ(k)`` (``power = 1`` ≈ Gauss–Newton for a free grid, since the
    image-space step scales with ``g²``)."""
    s = torch.as_tensor(sensitivity, dtype=torch.float64).flatten().abs()
    smax = float(s.max())
    if smax <= 0:
        raise ConfigError("sensitivity profile is identically zero")
    g = (smax / s.clamp_min(floor * smax)) ** power
    return profile_gain(freqs, g)


def lowpass_gain(cutoff: float, order: int = 4) -> GainFn:
    """Butterworth-like radial low-pass ``g = 1 / sqrt(1 + (k/cutoff)^{2·order})``."""

    def gain(kr: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.sqrt(1.0 + (kr / cutoff) ** (2 * order))

    return gain


def _radial_frequency(shape: Sequence[int], spacing: Sequence[float], device, dtype):
    """``|k|`` in cycles per unit coordinate for an ``rfftn`` over all axes of ``shape``."""
    fs = [
        torch.fft.fftfreq(n, d=h, device=device, dtype=dtype) for n, h in zip(shape[:-1], spacing)
    ]
    fs.append(torch.fft.rfftfreq(shape[-1], d=spacing[-1], device=device, dtype=dtype))
    mesh = torch.meshgrid(*fs, indexing="ij")
    return torch.sqrt(sum(m**2 for m in mesh))


@register("field", "spectral_preconditioned")
class SpectralPreconditionedField(Field):
    """``x = F⁻¹(g(|k|) · F f_θ)`` — a field whose raw output passes through a radial Fourier gain.

    See the module docstring for why this realizes a preconditioned image-space update. Evaluated
    on tensor-product grids (the FFT acts on the grid samples; the gain is defined in cycles per
    unit normalized coordinate, so it is consistent across curriculum resolutions).

    Args:
        inner: the wrapped field (its heads are applied after the gain).
        gain: ``None`` (identity, exact pass-through), a callable ``|k| -> g`` (cycles per unit),
            or a 1-D tensor of values at ``freqs``.
        freqs: sample frequencies for a tensor ``gain``.
        boundary: ``"reflect"`` (even extension, DCT-II; default) or ``"periodic"``.
        channels: raw channels to filter (default all).
        learnable: multiply the gain by a learnable log-profile on ``n_knots`` knots over
            ``[0, knot_max]`` (initialized at 1).
        normalize_dc: rescale so that ``g(0) = 1`` (the mean is not amplified).
        max_gain: optional clip of the gain.

    Example::

        op_sigma = operator_spectrum(problem.operator, problem.domain, fields)   # adaptive
        field = SpectralPreconditionedField.from_sensitivity(
            NeuralField(2, heads), op_sigma.freqs, op_sigma.values, floor=0.05)
    """

    def __init__(
        self,
        inner: Field,
        gain: GainFn | torch.Tensor | Sequence[float] | None = None,
        freqs: torch.Tensor | Sequence[float] | None = None,
        boundary: str = "reflect",
        channels: Sequence[int] | None = None,
        learnable: bool = False,
        n_knots: int = 8,
        knot_max: float = 16.0,
        normalize_dc: bool = True,
        max_gain: float | None = None,
    ) -> None:
        super().__init__(inner.heads)
        self.inner = inner
        if boundary not in ("reflect", "periodic"):
            raise ConfigError(f"boundary must be 'reflect' or 'periodic', got {boundary!r}")
        self.boundary = boundary
        if gain is not None and not callable(gain):
            if freqs is None:
                raise ConfigError("a sampled gain needs `freqs`")
            gain = profile_gain(freqs, gain)
        self._gain: GainFn | None = gain  # type: ignore[assignment]
        self.channels = None if channels is None else tuple(int(c) for c in channels)
        self.normalize_dc = bool(normalize_dc)
        self.max_gain = None if max_gain is None else float(max_gain)
        self.learnable = bool(learnable)
        if self.learnable:
            self.knot_max = float(knot_max)
            self.log_gain = nn.Parameter(torch.zeros(int(n_knots)))
        self._cache: dict[tuple, torch.Tensor] = {}

    @property
    def ndim(self) -> int | None:
        return getattr(self.inner, "ndim", None)

    @classmethod
    def from_sensitivity(
        cls,
        inner: Field,
        freqs: Sequence[float] | torch.Tensor,
        sensitivity: Sequence[float] | torch.Tensor,
        floor: float = 1e-2,
        power: float = 1.0,
        **kw: Any,
    ) -> SpectralPreconditionedField:
        """Preconditioner ``g = (σ_max / max(σ, floor σ_max))^power`` from a sensitivity profile
        (e.g. ``nefi.fields.adaptive.operator_spectrum(...)``: ``.freqs``, ``.values``)."""
        return cls(inner, inverse_sensitivity_gain(freqs, sensitivity, floor, power), **kw)

    @classmethod
    def from_operator(
        cls,
        inner: Field,
        operator: Any,
        domain: Any,
        fields: Mapping[str, torch.Tensor] | None = None,
        floor: float = 1e-2,
        power: float = 1.0,
        n_probes: int = 4,
        **kw: Any,
    ) -> SpectralPreconditionedField:
        """Measure the operator's radial sensitivity (JᵀJ probes) and build the preconditioner."""
        from ..adaptive.spectrum import operator_spectrum  # lazy: avoids an import cycle

        if fields is None:
            with torch.no_grad():
                fields = inner(domain.coords(), 1.0)
        spec = operator_spectrum(operator, domain, fields, n_probes=n_probes)
        return cls.from_sensitivity(inner, spec.freqs, spec.values, floor, power, **kw)

    # --- gain ----------------------------------------------------------------------------
    def gain_at(self, kr: torch.Tensor) -> torch.Tensor:
        """The (normalized, clipped, possibly learnable) gain at radial frequencies ``kr``."""
        g = torch.ones_like(kr) if self._gain is None else self._gain(kr).to(kr)
        if self.normalize_dc and self._gain is not None:
            g = g / self._gain(torch.zeros(1, device=kr.device, dtype=kr.dtype)).to(kr)
        if self.learnable:
            lg = self.log_gain.to(kr)
            x = kr.clamp(0.0, self.knot_max)
            pos = x / self.knot_max * (lg.numel() - 1)
            i0 = pos.floor().long().clamp(0, lg.numel() - 2)
            t = pos - i0.to(kr)
            g = g * torch.exp(lg[i0] * (1 - t) + lg[i0 + 1] * t)
        if self.max_gain is not None:
            g = g.clamp(max=self.max_gain)
        return g

    def _grid_gain(self, shape: tuple[int, ...], spacing: tuple[float, ...], ref: torch.Tensor):
        if self.learnable:
            return self.gain_at(_radial_frequency(shape, spacing, ref.device, ref.dtype))
        key = (shape, spacing, str(ref.device), str(ref.dtype))
        if key not in self._cache:
            kr = _radial_frequency(shape, spacing, ref.device, ref.dtype)
            self._cache[key] = self.gain_at(kr).detach()
        return self._cache[key]

    def _apply(self, fn, recurse=True):  # keep the gain cache coherent with .to()/.cuda()
        self._cache = {}
        return super()._apply(fn, recurse)

    # --- evaluation ----------------------------------------------------------------------
    def raw(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        r = self.inner.raw(coords, progress)
        if self._gain is None and not self.learnable:
            return r
        axes = grid_axes(coords)
        if axes is None:
            raise ShapeError(
                "SpectralPreconditionedField needs tensor-product grid coordinates "
                "(Domain.coords()); the Fourier gain is defined on grid samples"
            )
        d = coords.shape[-1]
        spacing = tuple(float(a[1] - a[0]) if a.numel() > 1 else 2.0 for a in axes)
        chans = range(r.shape[-1]) if self.channels is None else self.channels
        x = r.movedim(-1, 0)  # (C, *shape)
        sel = x[list(chans)]
        n = tuple(sel.shape[1:])
        if self.boundary == "reflect":
            for i in range(d):
                sel = torch.cat([sel, sel.flip(i + 1)], dim=i + 1)
        dims = tuple(range(1, d + 1))
        shape = tuple(sel.shape[1:])
        spec = torch.fft.rfftn(sel, dim=dims)
        g = self._grid_gain(shape, spacing, r)
        y = torch.fft.irfftn(spec * g, s=shape, dim=dims)
        y = y[(slice(None),) + tuple(slice(0, m) for m in n)]
        if self.channels is None:
            return y.movedim(0, -1)
        out = x.clone()
        out[list(chans)] = y
        return out.movedim(0, -1)

    def on_stage_start(self, stage, domain) -> None:
        self.inner.on_stage_start(stage, domain)

    def reset_parameters(self) -> None:
        self.inner.reset_parameters()
        if self.learnable:
            with torch.no_grad():
                self.log_gain.zero_()

    def extra_repr(self) -> str:
        return (
            f"gain={'identity' if self._gain is None else 'custom'}, boundary={self.boundary!r}, "
            f"learnable={self.learnable}, channels={self.channels}"
        )


__all__ = [
    "FourierBasisField",
    "SpectralPreconditionedField",
    "inverse_sensitivity_gain",
    "lowpass_gain",
    "profile_gain",
]
