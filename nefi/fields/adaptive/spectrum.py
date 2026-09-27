"""Radial spectra of representations and operators, and the match report that compares them.

The filtering view (NeTMY Lemma 2, App. D.6) says a representation acts on the field-space
gradient through ``G_θ = J_θ J_θᵀ``; linearizing the data term around the solution, the error at
spatial frequency ``ν`` of a translation-invariant problem evolves as

    ê_{t+1}(ν) ≈ (1 − η · g(ν) · σ_F(ν)²) ê_t(ν),

where ``g`` is the transfer function of ``G_θ`` and ``σ_F`` the operator's amplitude sensitivity
(NeTMY Lemma 1: ``σ_F ~ e^{−k z0}``; NeFTY Prop. 2: ``σ_n ≲ n^{−1/3}``). Two questions decide
whether a representation fits a system:

1. *Band*: does ``g`` pass the frequencies the data can resolve at the noise level — and only
   those? (Beyond the resolvable band, whatever ``g`` lets through is fitted to noise, NeFTY
   Cor. 1.)
2. *Conditioning*: is ``g · σ_F²`` flat on the resolvable band (fast, uniform convergence), or does
   it inherit the operator's decay (the free-grid case ``g ≡ 1``)?

:func:`representation_spectrum` measures ``g`` (rows ``G_θ e_i`` by a vjp followed by a jvp through
the field, Fourier-transformed and radially averaged over random pixels), :func:`operator_spectrum`
measures ``σ_F`` (rows ``JᵀJ e_i``), and :func:`match_report` compares them at the noise level and
says in words what to change. Frequencies are in cycles per unit *normalized* coordinate
(``[-1, 1]`` per axis; a grid of ``n`` cells has Nyquist ``n/4``), the unit used by
``nefi.fields.encoding.FourierFeatures.effective_bandwidth``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.func import functional_call

from ...errors import ConfigError, ShapeError
from ...utils.tensor import resample, shape_tuple

log = logging.getLogger("nefi")

TensorFn = Callable[[torch.Tensor], torch.Tensor]


# ------------------------------------------------------------------------------------------
# containers and radial averaging
# ------------------------------------------------------------------------------------------
@dataclass
class Spectrum:
    """A radially averaged spectrum.

    Attributes:
        freqs: ring centers ``(B,)`` in cycles per unit normalized coordinate.
        values: per-ring value ``(B,)``; meaning depends on ``kind``:
            ``"representation"`` — transfer ``|FFT(G_θ e_i)|`` of the update kernel;
            ``"operator"`` — amplitude sensitivity ``σ_F = sqrt|FFT(JᵀJ e_i)|`` (``"jtj"``) or
            ``|FFT(J e_i)|`` (``"j"``);
            ``"field"`` / ``"data"`` — amplitude ``|X̂|`` (orthonormal FFT, mean removed).
        power: per-ring power ``(B,)`` (``|FFT(G e_i)|²``, ``σ_F²``, ``|X̂|²``).
        counts: number of Fourier modes per ring ``(B,)``.
        kind: see above.
        meta: provenance (progress, shape, probes, Nyquist, ...).
    """

    freqs: torch.Tensor
    values: torch.Tensor
    power: torch.Tensor
    counts: torch.Tensor
    kind: str = "field"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def nyquist(self) -> float:
        return float(self.meta.get("nyquist", float(self.freqs.max())))

    def reference(self, ref: str = "max") -> float:
        """Normalization value: max over non-DC rings (``"max"``), the DC ring (``"dc"``) or the
        first non-DC ring (``"low"``). The DC ring is special for update kernels: a bias
        parameter adds a constant column to ``J_θ`` and hence an ``N``-fold spike at ``k = 0``."""
        v = self.values
        nz = v[self.freqs > 0] if bool((self.freqs > 0).any()) else v
        if ref == "max":
            return float(nz.max())
        if ref == "dc":
            return float(v[0])
        if ref == "low":
            return float(nz[0])
        raise ConfigError(f"unknown reference {ref!r}; use 'max', 'dc' or 'low'")

    def normalized(self, ref: str = "max") -> Spectrum:
        """Copy with ``values`` divided by :meth:`reference` (``power`` by its square)."""
        r = self.reference(ref)
        if r <= 0:
            return self
        return Spectrum(
            self.freqs, self.values / r, self.power / r**2, self.counts, self.kind, dict(self.meta)
        )

    def at(self, f: float | torch.Tensor) -> torch.Tensor:
        """Linear interpolation of ``values`` at frequencies ``f`` (clamped to the range)."""
        return _interp(torch.as_tensor(f, dtype=self.freqs.dtype), self.freqs, self.values)

    def bandwidth(
        self, level: float = 0.1, ref: str = "max", max_freq: float | None = None
    ) -> float:
        """First non-DC frequency where ``values / reference`` drops below ``level``."""
        top = self.nyquist if max_freq is None else float(max_freq)
        keep = (self.freqs <= top + 1e-9) & (self.freqs > 0)
        f, v = self.freqs[keep], self.values[keep]
        r = self.reference(ref)
        if f.numel() == 0 or r <= 0:
            return 0.0
        rel = v / r
        below = (rel < level).nonzero().flatten()
        if below.numel() == 0:
            return float(f[-1])
        b = int(below[0])
        if b == 0:
            return float(f[0])
        return _crossing(f[b - 1], f[b], rel[b - 1], rel[b], level)

    def energy_bandwidth(self, fraction: float = 0.9, max_freq: float | None = None) -> float:
        """Radius containing ``fraction`` of the (non-DC) energy ``Σ counts · power``.

        For an update kernel this is where a white field-space gradient's *realized update*
        (``G_θ g``) has put ``fraction`` of its energy (Parseval): a free grid gives ≈ Nyquist, a
        ``K``-mode Fourier basis its cutoff, an annealed neural field a radius growing with
        progress (NeTMY Eq. 36–39).
        """
        if not 0.0 < fraction <= 1.0:
            raise ConfigError("fraction must be in (0, 1]")
        top = self.nyquist if max_freq is None else float(max_freq)
        keep = (self.freqs > 0) & (self.freqs <= top + 1e-9)
        f = self.freqs[keep].double()
        e = (self.power[keep].double() * self.counts[keep].double()).clamp_min(0)
        if f.numel() == 0 or float(e.sum()) <= 0:
            return 0.0
        c = torch.cumsum(e, 0) / e.sum()
        i = int((c >= fraction - 1e-12).nonzero()[0])
        if i == 0:
            return float(f[0])
        c0, c1 = float(c[i - 1]), float(c[i])
        t = (fraction - c0) / max(c1 - c0, 1e-300)
        return float(f[i - 1] + min(max(t, 0.0), 1.0) * (f[i] - f[i - 1]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "freqs": self.freqs.tolist(),
            "values": self.values.tolist(),
            "power": self.power.tolist(),
            "counts": self.counts.tolist(),
            "meta": {
                k: v for k, v in self.meta.items() if isinstance(v, int | float | str | tuple)
            },
        }

    def table(self, max_rows: int = 12) -> str:
        """Compact text table (log-spaced subset of the rings)."""
        n = self.freqs.numel()
        idx = sorted({int(round(i)) for i in torch.linspace(0, n - 1, min(n, max_rows)).tolist()})
        lines = [f"{'freq':>8} {'value':>12} {'power':>12}"]
        for i in idx:
            lines.append(
                f"{float(self.freqs[i]):8.3f} {float(self.values[i]):12.4e} "
                f"{float(self.power[i]):12.4e}"
            )
        return "\n".join(lines)


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    xp, fp = xp.to(torch.float64), fp.to(torch.float64)
    xx = x.to(torch.float64).clamp(float(xp[0]), float(xp[-1]))
    if xp.numel() == 1:
        return fp.expand_as(xx).clone()
    idx = torch.searchsorted(xp, xx.contiguous().reshape(-1)).clamp(1, xp.numel() - 1)
    idx = idx.reshape(xx.shape)
    x0, x1, f0, f1 = xp[idx - 1], xp[idx], fp[idx - 1], fp[idx]
    t = (xx - x0) / (x1 - x0).clamp_min(1e-12)
    return f0 + t * (f1 - f0)


def _crossing(f0, f1, r0, r1, level: float) -> float:
    """Frequency where a curve crosses ``level`` between two samples (log interpolation)."""
    f0, f1, r0, r1 = float(f0), float(f1), float(r0), float(r1)
    if r0 <= 0 or r1 <= 0 or r0 == r1:
        return f0
    t = (math.log(level) - math.log(r0)) / (math.log(r1) - math.log(r0))
    return f0 + min(max(t, 0.0), 1.0) * (f1 - f0)


def _float_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "mps" else torch.float64


def _hann_window(shape: Sequence[int], axes: Sequence[int], device, dtype) -> torch.Tensor:
    """Separable Hann window over ``axes``, normalized to unit mean power (``mean w² = 1``), so
    white noise keeps its per-mode power after windowing."""
    w = torch.ones(tuple(shape), device=device, dtype=dtype)
    for a in axes:
        n = shape[a]
        if n < 3:
            continue
        h = torch.hann_window(n, periodic=False, device=device, dtype=dtype)
        view = [1] * len(shape)
        view[a] = n
        w = w * h.view(view)
    return w / torch.sqrt((w**2).mean())


def radial_average(
    x: torch.Tensor,
    spacing: Sequence[float] | None = None,
    axes: Sequence[int] | None = None,
    norm: str = "ortho",
    center: bool = False,
    crop: bool = True,
    window: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Radially averaged ``|FFT|`` and ``|FFT|²`` of a field-shaped tensor.

    Args:
        x: tensor ``(*shape)``.
        spacing: cell size per axis in normalized units (default ``2 / n``: ``[-1, 1]`` axes).
        axes: axes to transform (others are averaged), default all.
        norm: FFT normalization (``"ortho"`` for amplitudes of signals, ``"backward"`` for
            transfer functions of kernel rows).
        center: subtract the mean first.
        crop: drop rings beyond the smallest per-axis Nyquist (partially covered corners).
        window: ``None`` or ``"hann"`` — taper non-periodic signals so that the wrap-around
            jump of the FFT's periodic extension (e.g. different layers at opposite faces) does
            not leak ``1/ν²`` power into every ring; normalized to preserve white-noise power.

    Returns:
        ``(freqs, mean_abs, mean_power, counts, nyquist)``.
    """
    d = x.ndim
    axes = tuple(range(d)) if axes is None else tuple(int(a) % d for a in axes)
    spacing = tuple(2.0 / n for n in x.shape) if spacing is None else tuple(spacing)
    t = x.detach().to(_float_dtype(x.device))
    if center:
        t = t - t.mean()
    if window == "hann":
        wnd = _hann_window(tuple(t.shape), axes, t.device, t.dtype)
        if center:
            t = t - (t * wnd**2).sum() / (wnd**2).sum()  # weighted mean under the window
        t = t * wnd
    elif window is not None:
        raise ConfigError(f"unknown window {window!r}; use None or 'hann'")
    spec = torch.fft.fftn(t, dim=axes, norm=norm)
    mag = spec.abs()
    pw = mag**2
    other = tuple(i for i in range(d) if i not in axes)
    if other:
        mag, pw = mag.mean(dim=other), pw.mean(dim=other)
    sub_n = [x.shape[a] for a in axes]
    sub_h = [float(spacing[a]) for a in axes]
    fs = [torch.fft.fftfreq(n, d=h, device=x.device, dtype=t.dtype) for n, h in zip(sub_n, sub_h)]
    mesh = torch.meshgrid(*fs, indexing="ij") if len(fs) > 1 else (fs[0],)
    kr = torch.sqrt(sum(m**2 for m in mesh))
    bw = min(1.0 / (n * h) for n, h in zip(sub_n, sub_h))
    nyq = min(1.0 / (2.0 * h) for h in sub_h)
    idx = torch.round(kr / bw).long().flatten()
    nb = int(idx.max()) + 1
    counts = torch.bincount(idx, minlength=nb)
    s_mag = torch.bincount(idx, weights=mag.flatten(), minlength=nb)
    s_pw = torch.bincount(idx, weights=pw.flatten(), minlength=nb)
    freqs = torch.arange(nb, device=x.device, dtype=t.dtype) * bw
    keep = counts > 0
    if crop:
        keep = keep & (freqs <= nyq + 1e-9)
    c = counts[keep].to(t.dtype)
    return (
        freqs[keep].cpu(),
        (s_mag[keep] / c).cpu(),
        (s_pw[keep] / c).cpu(),
        counts[keep].cpu(),
        nyq,
    )


def field_spectrum(
    x: torch.Tensor,
    spacing: Sequence[float] | None = None,
    axes: Sequence[int] | None = None,
    center: bool = True,
    kind: str = "field",
    window: str | None = "hann",
) -> Spectrum:
    """Radial amplitude spectrum of a field (orthonormal FFT, mean removed, Hann-windowed by
    default — images of layered media are far from periodic)."""
    f, mag, pw, cnt, nyq = radial_average(x, spacing, axes, "ortho", center, window=window)
    return Spectrum(f, mag, pw, cnt, kind, {"nyquist": nyq, "shape": tuple(x.shape)})


# ------------------------------------------------------------------------------------------
# autograd helpers
# ------------------------------------------------------------------------------------------
def _module_device_dtype(module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    for p in module.parameters():
        return p.device, p.dtype
    for b in module.buffers():
        if torch.is_floating_point(b):
            return b.device, b.dtype
    return torch.device("cpu"), torch.get_default_dtype()


def kernel_row(
    field_module: torch.nn.Module,
    coords: torch.Tensor,
    index: tuple[int, ...],
    progress: float = 1.0,
    name: str | None = None,
) -> torch.Tensor:
    """``G_θ e_i = J_θ J_θᵀ e_i`` for the pixel ``index`` (NeTMY Lemma 2, Eq. 32–34).

    Computed as a vjp (``J_θᵀ e_i``) followed by a jvp through the field (``torch.func``), with a
    double-backward fallback for modules that functorch cannot trace.
    """
    name = name or field_module.primary
    trainable = {k: p.detach() for k, p in field_module.named_parameters() if p.requires_grad}
    frozen = {k: p.detach() for k, p in field_module.named_parameters() if not p.requires_grad}
    if not trainable:
        raise ConfigError("the field has no trainable parameters")
    buffers = dict(field_module.named_buffers())

    def f(params: dict[str, torch.Tensor]) -> torch.Tensor:
        out = functional_call(field_module, ({**frozen, **params}, buffers), (coords, progress))
        return out[name]

    try:
        out, vjp_fn = torch.func.vjp(f, trainable)
        e = torch.zeros_like(out)
        e[index] = 1.0
        (ct,) = vjp_fn(e)
        _, row = torch.func.jvp(f, (trainable,), (ct,))
        return row.detach()
    except Exception as err:  # modules without functorch support
        log.debug("functorch path failed in kernel_row (%s); using double-backward", err)
    plist = [p for p in field_module.parameters() if p.requires_grad]
    with torch.enable_grad():
        out = field_module(coords, progress)[name]
        ct = torch.autograd.grad(out[index], plist, retain_graph=True, allow_unused=True)
        ct = [torch.zeros_like(p) if c is None else c for p, c in zip(plist, ct)]
        u = torch.zeros_like(out, requires_grad=True)
        g = torch.autograd.grad(out, plist, grad_outputs=u, create_graph=True, allow_unused=True)
        pairs = [(gi, ci) for gi, ci in zip(g, ct) if gi is not None]
        (row,) = torch.autograd.grad(
            [gi for gi, _ in pairs], u, grad_outputs=[ci for _, ci in pairs], allow_unused=True
        )
    return torch.zeros_like(out).detach() if row is None else row.detach()


def jvp(
    fn: TensorFn, x: torch.Tensor, v: torch.Tensor, mode: str = "auto"
) -> tuple[torch.Tensor, str]:
    """``J v`` by forward mode, falling back to double-backward, then central differences."""
    if mode in ("auto", "forward"):
        try:
            _, out = torch.func.jvp(fn, (x.detach(),), (v.to(x),))
            return out.detach(), "forward"
        except Exception as e:
            if mode == "forward":
                raise
            log.debug("forward-mode jvp unavailable (%s)", e)
    if mode in ("auto", "double_backward"):
        try:
            with torch.enable_grad():
                xx = x.detach().requires_grad_(True)
                y = fn(xx)
                u = torch.zeros_like(y, requires_grad=True)
                (g,) = torch.autograd.grad(y, xx, grad_outputs=u, create_graph=True)
                (jv,) = torch.autograd.grad(g, u, grad_outputs=v.to(g), allow_unused=True)
            return (torch.zeros_like(y) if jv is None else jv).detach(), "double_backward"
        except Exception as e:
            if mode == "double_backward":
                raise
            log.debug("double-backward jvp unavailable (%s); using finite differences", e)
    with torch.no_grad():
        vmax = float(v.abs().max())
        if vmax == 0:
            return torch.zeros_like(fn(x)), "finite_difference"
        eps = torch.finfo(x.dtype).eps ** (1.0 / 3.0) * max(1.0, float(x.abs().max())) / vmax
        return (fn(x + eps * v) - fn(x - eps * v)) / (2.0 * eps), "finite_difference"


class _VJP:
    def __init__(self, fn: TensorFn, x: torch.Tensor) -> None:
        with torch.enable_grad():
            self.x = x.detach().requires_grad_(True)
            self.y = fn(self.x)
        if not self.y.requires_grad:
            raise ConfigError("the operator output does not depend on the probed field")

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        (g,) = torch.autograd.grad(
            self.y, self.x, grad_outputs=u.to(self.y), retain_graph=True, allow_unused=True
        )
        return torch.zeros_like(self.x) if g is None else g.detach()


def _choose_pixels(
    shape: tuple[int, ...],
    n: int,
    seed: int,
    margin: float,
    pixels: Sequence[Sequence[int]] | None,
) -> list[tuple[int, ...]]:
    if pixels is not None:
        return [tuple(int(i) for i in p) for p in pixels]
    if n < 1:
        raise ConfigError("n_probes must be >= 1")
    gen = torch.Generator().manual_seed(int(seed))
    out = []
    for _ in range(n):
        idx = []
        for s in shape:
            m = int(math.floor(margin * s))
            lo, hi = m, max(m + 1, s - m)
            idx.append(int(torch.randint(lo, hi, (1,), generator=gen)))
        out.append(tuple(idx))
    return out


# ------------------------------------------------------------------------------------------
# representation / operator spectra
# ------------------------------------------------------------------------------------------
def representation_spectrum(
    field: torch.nn.Module,
    domain: Any,
    n_probes: int = 8,
    progress: float = 1.0,
    *,
    shape: Sequence[int] | None = None,
    name: str | None = None,
    seed: int = 0,
    pixels: Sequence[Sequence[int]] | None = None,
    axes: Sequence[int] | None = None,
    margin: float = 0.1,
) -> Spectrum:
    """Radial transfer function of the representation's update kernel ``G_θ`` (NeTMY Lemma 2).

    For ``n_probes`` random interior pixels ``i`` (a ``margin`` fraction of each axis is avoided so
    rows are not truncated by the boundary) the row ``G_θ e_i`` is computed by a vjp then a jvp
    through the field, Fourier transformed (unnormalized FFT: a delta has transfer 1) and radially
    averaged; the result is averaged over probes. A :class:`~nefi.fields.GridField` gives a flat
    spectrum (``G = I``); a :class:`~nefi.fields.NeuralField` at low annealing progress a low-pass
    one whose cutoff grows with progress (NeTMY Eq. 36–39).

    Args:
        field: the field module (its current parameters are the linearization point).
        domain: the domain (grid on which the kernel acts).
        n_probes: number of random pixels.
        progress: annealing progress passed to the field.
        shape: evaluate at another resolution of the domain.
        name: output field (default primary).
        seed: pixel seed.
        pixels: explicit pixel multi-indices (overrides the random choice).
        axes: transform only these axes (others averaged), e.g. lateral axes of a slab.
        margin: fraction of each axis excluded at both ends when drawing pixels.

    Returns:
        :class:`Spectrum` with ``kind="representation"`` (``values`` = mean ``|FFT(G e_i)|``).
    """
    dom = domain if shape is None else domain.at(shape_tuple(shape))
    device, dtype = _module_device_dtype(field)
    coords = dom.coords(device=device, dtype=dtype)
    name = name or field.primary
    idxs = _choose_pixels(tuple(dom.shape), n_probes, seed, margin, pixels)
    spacing = tuple(2.0 / n for n in dom.shape)
    acc_m = acc_p = None
    for idx in idxs:
        row = kernel_row(field, coords, idx, progress, name)
        f, m, p, cnt, nyq = radial_average(row, spacing, axes, norm="backward")
        acc_m = m if acc_m is None else acc_m + m
        acc_p = p if acc_p is None else acc_p + p
    k = len(idxs)
    meta = {
        "progress": float(progress),
        "n_probes": k,
        "shape": tuple(dom.shape),
        "nyquist": nyq,
        "name": name,
        "field": type(field).__name__,
    }
    return Spectrum(f, acc_m / k, acc_p / k, cnt, "representation", meta)


def _linearization_point(
    fields: Any, dom: Any, name: str, progress: float = 1.0
) -> dict[str, torch.Tensor]:
    if isinstance(fields, torch.nn.Module):
        device, dtype = _module_device_dtype(fields)
        with torch.no_grad():
            fields = fields(dom.coords(device=device, dtype=dtype), progress)
    if torch.is_tensor(fields):
        fields = {name: fields}
    if not isinstance(fields, Mapping):
        raise ConfigError("fields must be a mapping name -> tensor, a tensor or a Field")
    out = {}
    for k, v in fields.items():
        t = torch.as_tensor(v).detach()
        if tuple(t.shape) != tuple(dom.shape):
            t = resample(t, dom.shape)
        out[k] = t
    if name not in out:
        raise ConfigError(f"no field {name!r} in the linearization point {tuple(out)}")
    return out


def operator_spectrum(
    operator: Any,
    domain: Any,
    fields: Any,
    n_probes: int = 8,
    *,
    name: str | None = None,
    shape: Sequence[int] | None = None,
    seed: int = 0,
    mode: str = "jtj",
    pixels: Sequence[Sequence[int]] | None = None,
    axes: Sequence[int] | None = None,
    mask: torch.Tensor | None = None,
    margin: float = 0.1,
    jvp_mode: str = "auto",
    progress: float = 1.0,
) -> Spectrum:
    """Radial amplitude sensitivity ``σ_F(ν)`` of the (linearized) forward operator.

    ``mode="jtj"`` (default, any operator): rows ``JᵀJ e_i`` (a jvp then a vjp) live on the field
    grid; their transfer is ``σ_F²`` (the normal operator), and ``values = sqrt`` of it. For a
    convolution with transfer ``a(k)`` this recovers ``|a(k)|`` (up to boundary truncation).
    ``mode="j"``: power spectrum of ``J e_i`` over the trailing ``ndim`` axes of the measurement
    (meaningful when the measurement lives on a grid of the same domain).

    Args:
        operator: the forward operator (``at_resolution`` is honoured).
        domain: field domain.
        fields: linearization point — mapping ``name -> tensor``, a tensor (the probed field), or
            a :class:`~nefi.fields.base.Field` (evaluated at ``progress``).
        n_probes: number of random interior pixels.
        name: probed field (default ``operator.primary``).
        shape: grid resolution (default the domain's).
        seed / pixels / axes / margin: see :func:`representation_spectrum`.
        mask: optional measurement mask (the Jacobian of what the data term compares).
        jvp_mode: ``"auto"`` | ``"forward"`` | ``"double_backward"`` | ``"finite_difference"``.
        progress: progress used when ``fields`` is a field module.

    Returns:
        :class:`Spectrum` with ``kind="operator"`` (``values`` = ``σ_F``, ``power`` = ``σ_F²``).
    """
    if mode not in ("jtj", "j"):
        raise ConfigError(f"mode must be 'jtj' or 'j', got {mode!r}")
    dom = domain if shape is None else domain.at(shape_tuple(shape))
    name = name or getattr(operator, "primary", "x")
    base = _linearization_point(fields, dom, name, progress)
    op = operator.at_resolution(tuple(dom.shape))
    x0 = base[name]
    m = None if mask is None else torch.as_tensor(mask).to(x0)

    def fn(x: torch.Tensor) -> torch.Tensor:
        y = op({**base, name: x})
        return y * m if m is not None else y

    idxs = _choose_pixels(tuple(dom.shape), n_probes, seed, margin, pixels)
    spacing = tuple(2.0 / n for n in dom.shape)
    vjp_op = _VJP(fn, x0) if mode == "jtj" else None
    jm = jvp_mode
    acc = None
    nyq = 0.0
    for idx in idxs:
        e = torch.zeros_like(x0)
        e[idx] = 1.0
        jv, jm = jvp(fn, x0, e, jm)
        if mode == "jtj":
            row = vjp_op(jv)  # type: ignore[misc]
            f, mag, _, cnt, nyq = radial_average(row, spacing, axes, norm="backward")
            acc = mag if acc is None else acc + mag
        else:
            d = dom.ndim
            if jv.ndim < d:
                raise ShapeError(f"measurement of shape {tuple(jv.shape)} has fewer than {d} dims")
            lead = jv.shape[:-d] if jv.ndim > d else ()
            y = jv.reshape(-1, *jv.shape[-d:]) if lead else jv.unsqueeze(0)
            sp = tuple(2.0 / n for n in y.shape[1:])
            pw_acc = None
            for yy in y:
                f, _, pw, cnt, nyq = radial_average(yy, sp, None, norm="backward")
                pw_acc = pw if pw_acc is None else pw_acc + pw
            acc = pw_acc if acc is None else acc + pw_acc
    k = len(idxs)
    power = acc / k  # jtj: transfer of JᵀJ = σ_F²; j: ring power of J e_i = σ_F²
    values = power.clamp_min(0).sqrt()
    meta = {
        "mode": mode,
        "n_probes": k,
        "shape": tuple(dom.shape),
        "nyquist": nyq,
        "name": name,
        "jvp_mode": jm,
        "operator": type(operator).__name__,
    }
    return Spectrum(f, values, power, cnt, "operator", meta)


# ------------------------------------------------------------------------------------------
# match report
# ------------------------------------------------------------------------------------------
def estimate_noise_std(data: torch.Tensor) -> float:
    """Robust white-noise estimate from first differences along the last axis (MAD / √2)."""
    y = torch.as_tensor(data).detach().double()
    if y.shape[-1] < 3:
        return float(y.std())
    dy = (y[..., 1:] - y[..., :-1]).flatten()
    return float(dy.abs().median() / 0.6745 / math.sqrt(2.0))


def resolvable_bandwidth(
    operator: Spectrum,
    noise_std: float,
    data_norm: float,
    prior_decay: float = 0.0,
    margin: float = 1.0,
) -> float:
    """Resolvable band under a power-law field prior ``|X̂(ν)| = c (1 + ν/ν₁)^{−p}``.

    ``c`` is fixed by the data energy ``‖y‖² ≈ Σ_modes σ_F² |X̂|²``; frequencies where the signal
    power per mode ``σ_F² |X̂|²`` exceeds ``margin · σ_ε²`` are resolvable (a Picard-type rule,
    NeFTY Cor. 1). Returns the end of the contiguous resolvable band.
    """
    f, s = operator.freqs.double(), operator.values.double()
    cnt = operator.counts.double()
    nu1 = float(f[1]) if f.numel() > 1 else 1.0
    w = (1.0 + f / nu1) ** (-float(prior_decay))
    denom = float((cnt * s**2 * w**2).sum())
    if denom <= 0 or data_norm <= 0:
        return 0.0
    c2 = data_norm**2 / denom
    snr = s**2 * w**2 * c2 / max(noise_std**2, 1e-300)
    return _band_end(f, snr, margin, operator.counts)


def smooth_octave(
    f: torch.Tensor, r: torch.Tensor, counts: torch.Tensor | None = None, width: float = 0.5
) -> torch.Tensor:
    """Counts-weighted average of ``r`` over ``[ν 2^{−width}, ν 2^{width}]`` around each ring.

    Averages out interference ripples of multi-object spectra and the ``χ²₂`` fluctuation of
    single Fourier modes (rings with few modes), so band edges are not triggered by one dip.
    """
    f = f.double()
    r = r.double()
    c = torch.ones_like(r) if counts is None else counts.double()
    out = r.clone()
    lo_f, hi_f = 2.0 ** (-width), 2.0**width
    for b in range(1, f.numel()):
        win = (f >= f[b] * lo_f - 1e-12) & (f <= f[b] * hi_f + 1e-12) & (f > 0)
        out[b] = (r[win] * c[win]).sum() / c[win].sum()
    return out


def _band_end(
    f: torch.Tensor,
    ratio: torch.Tensor,
    level: float,
    counts: torch.Tensor | None = None,
    width: float = 0.5,
) -> float:
    """End of the contiguous run (from the first non-DC ring) where the octave-smoothed
    ``ratio >= level``."""
    r = smooth_octave(f, ratio, counts, width) if width > 0 else ratio.double()
    start = 1 if r.numel() > 1 else 0
    fail = (r[start:] < level).nonzero().flatten()
    if fail.numel() == 0:
        return float(f[-1])
    b = int(fail[0]) + start
    if b == 0:
        return 0.0
    if b == start:
        return float(f[b - 1]) if b > 0 else 0.0
    return _crossing(f[b - 1], f[b], r[b - 1], r[b], level)


@dataclass
class MatchReport:
    """Representation vs. operator at the noise level (see :func:`match_report`)."""

    representation: Spectrum
    operator: Spectrum
    data: Spectrum | None
    noise_std: float
    noise_source: str
    data_bandwidth: float
    rep_bandwidth: float
    nyquist: float
    verdict: str
    ratio: float
    kappa_rep: float
    kappa_grid: float
    method: str
    progress: float
    text: str = ""

    def __str__(self) -> str:
        return self.text

    def to_markdown(self) -> str:
        rows = [
            ("verdict", self.verdict),
            (
                "resolvable bandwidth ν_data [cyc/unit]",
                f"{self.data_bandwidth:.3g} ({self.method})",
            ),
            ("representation bandwidth ν_rep", f"{self.rep_bandwidth:.3g}"),
            ("ν_rep / ν_data", f"{self.ratio:.3g}"),
            ("grid Nyquist", f"{self.nyquist:.3g}"),
            ("noise σ", f"{self.noise_std:.3g} ({self.noise_source})"),
            ("κ(g·σ_F²) on band", f"{self.kappa_rep:.3g}"),
            ("κ(σ_F²) free grid", f"{self.kappa_grid:.3g}"),
        ]
        body = "\n".join(f"| {k} | {v} |" for k, v in rows)
        return f"| quantity | value |\n|---|---|\n{body}\n\n{self.text}\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "data_bandwidth": self.data_bandwidth,
            "rep_bandwidth": self.rep_bandwidth,
            "ratio": self.ratio,
            "nyquist": self.nyquist,
            "noise_std": self.noise_std,
            "noise_source": self.noise_source,
            "kappa_rep": self.kappa_rep,
            "kappa_grid": self.kappa_grid,
            "method": self.method,
            "progress": self.progress,
            "text": self.text,
        }


def match_report(
    field: torch.nn.Module | None,
    problem: Any,
    *,
    progress: float = 1.0,
    noise_std: float | None = None,
    n_probes: int = 8,
    energy: float = 0.9,
    tol: float = 0.25,
    margin: float = 1.0,
    method: str = "auto",
    prior_decay: float = 0.0,
    shape: Sequence[int] | None = None,
    seed: int = 0,
    operator_spec: Spectrum | None = None,
) -> MatchReport:
    """Compare a representation's pass band with what the data can resolve at the noise level.

    1. ``σ_F(ν)`` — :func:`operator_spectrum` at the current field values (linearization point).
    2. ``g(ν)`` — :func:`representation_spectrum` of ``field`` at ``progress``.
    3. The *resolvable band* ``ν_data``: ``method="data"`` uses the measurement's own radial power
       spectrum against the noise floor (``|Ŷ|² ≥ (1 + margin) σ²``, a data-driven Picard rule;
       needs the measurement on the field grid), ``method="prior"`` uses ``σ_F`` with a
       power-law field prior (:func:`resolvable_bandwidth`); ``"auto"`` picks ``"data"`` when
       possible.
    4. ``ν_rep``: the radius holding ``energy`` of the realized update energy of a white
       gradient through ``G_θ`` (:meth:`Spectrum.energy_bandwidth`).
    5. Verdict: ``ν_rep < (1 − tol) ν_data`` → *over-bandlimited* (too smooth: cannot express
       resolvable detail); ``ν_rep > (1 + tol) ν_data`` → *under-bandlimited* (moves frequencies
       the data cannot constrain: noise fitting unless annealed / regularized / stopped early);
       else *matched*. Plus the effective conditioning ``κ(g σ_F²)`` on the resolvable band vs the
       free-grid ``κ(σ_F²)``.

    Args:
        field: the representation to assess (default ``problem.field``).
        problem: the :class:`~nefi.problem.InverseProblem` (operator, measurement, domain).
        progress: annealing progress at which ``G_θ`` is measured.
        noise_std: noise level (default: ``measurement.noise_std``, else a MAD estimate).
        n_probes: probes per spectrum.
        energy: energy fraction defining ``ν_rep``.
        tol: relative tolerance of the "matched" verdict.
        margin: SNR margin of the resolvability rule.
        method: ``"auto"`` | ``"data"`` | ``"prior"``.
        prior_decay: power-law exponent ``p`` of the ``"prior"`` rule (0 = white field).
        shape: resolution (default native).
        seed: probe seed.
        operator_spec: reuse a precomputed operator spectrum.

    Returns:
        :class:`MatchReport` (``str(report)`` is the explanation in words).
    """
    fld = problem.field if field is None else field
    dom = problem.domain if shape is None else problem.domain.at(shape_tuple(shape))
    meas = problem.measurement_at(dom.shape)
    device, dtype = _module_device_dtype(problem.field)
    with torch.no_grad():
        fields0 = problem.field(dom.coords(device=device, dtype=dtype), 1.0)
    fields0 = {k: v.detach() for k, v in fields0.items()}
    op_spec = operator_spec
    if op_spec is None:
        mask = meas.mask.to(device, dtype) if meas.mask is not None else None
        op_spec = operator_spectrum(problem.operator, dom, fields0, n_probes, seed=seed, mask=mask)
    rep = representation_spectrum(fld, dom, n_probes, progress, seed=seed)
    nyq = min(rep.nyquist, op_spec.nyquist)

    # noise level
    if noise_std is not None:
        sigma, src = float(noise_std), "given"
    elif meas.noise_std is not None:
        ns = meas.noise_std
        sigma, src = float(ns.mean()) if torch.is_tensor(ns) else float(ns), "measurement"
    else:
        sigma, src = estimate_noise_std(meas.data), "estimated (MAD of differences)"
    sigma = max(sigma, 1e-12)

    # resolvable band
    data = meas.data.detach()
    on_grid = tuple(data.shape) == tuple(dom.shape) and (
        meas.mask is None or bool((meas.mask > 0).all())
    )
    use = method
    if method == "auto":
        use = "data" if on_grid else "prior"
    data_spec = None
    if use == "data":
        if not on_grid:
            raise ConfigError("method='data' needs a fully observed measurement on the field grid")
        data_spec = field_spectrum(data, kind="data")
        ratio_curve = data_spec.power / sigma**2
        nu_data = _band_end(data_spec.freqs, ratio_curve, 1.0 + margin, data_spec.counts)
        how = "data spectrum vs noise floor (|Ŷ|² ≥ (1+margin)σ²)"
    elif use == "prior":
        yc = data - data.mean()
        mask = meas.mask
        if mask is not None:
            yc = yc * mask.expand_as(yc)
        norm_sig = float(
            torch.sqrt(torch.clamp((yc.double() ** 2).sum() - sigma**2 * yc.numel(), min=0))
        )
        nu_data = resolvable_bandwidth(op_spec, sigma, norm_sig, prior_decay, margin)
        how = f"operator sensitivity with a (1+ν/ν₁)^-{prior_decay:g} field prior"
    else:
        raise ConfigError(f"method must be 'auto', 'data' or 'prior', got {method!r}")
    nu_data = min(nu_data, nyq)
    nu_rep = rep.energy_bandwidth(energy, nyq)
    ratio = nu_rep / nu_data if nu_data > 0 else math.inf

    if ratio < 1.0 - tol:
        verdict = "over-bandlimited"
    elif ratio > 1.0 + tol:
        verdict = "under-bandlimited"
    else:
        verdict = "matched"

    # effective conditioning on the resolvable band
    f = rep.freqs
    sig = _interp(f, op_spec.freqs, op_spec.values)
    g = rep.values.double() / max(rep.reference("max"), 1e-300)
    band = (f > 0) & (f <= max(nu_data, float(f[1]) if f.numel() > 1 else 0.0) + 1e-9)
    if bool(band.any()):
        eff = (g * sig**2)[band].clamp_min(1e-300)
        s2 = (sig**2)[band].clamp_min(1e-300)
        kappa_rep = float(eff.max() / eff.min())
        kappa_grid = float(s2.max() / s2.min())
    else:
        kappa_rep = kappa_grid = float("nan")

    shape_txt = "×".join(str(s) for s in dom.shape)
    lines = [
        f"Representation vs. operator at the noise level ({type(fld).__name__}, "
        f"progress={progress:.2f}, grid {shape_txt}, {rep.meta['n_probes']} probes):",
        f"- the data resolve frequencies up to ν_data ≈ {nu_data:.3g} cycles/unit "
        f"[{how}]; noise σ = {sigma:.3g} ({src}); grid Nyquist {nyq:.3g}.",
        f"- the representation passes frequencies up to ν_rep ≈ {nu_rep:.3g} cycles/unit "
        f"({energy:.0%} of a white gradient's realized update energy through "
        "G_θ = J_θ J_θᵀ lies below it; NeTMY Lemma 2).",
    ]
    if verdict == "over-bandlimited":
        lines.append(
            f"- verdict: OVER-BANDLIMITED (ν_rep/ν_data = {ratio:.2f}): the parameterization is "
            "smoother than what the data can resolve, so resolvable detail is lost. Raise the "
            "annealing progress / octaves / basis size, or use a geometry that carries sharp "
            "structure explicitly (level set, star shapes, layered interfaces)."
        )
    elif verdict == "under-bandlimited":
        lines.append(
            f"- verdict: UNDER-BANDLIMITED (ν_rep/ν_data = {ratio:.2f}): the parameterization "
            "moves frequencies the data cannot constrain at this noise level (NeFTY Cor. 1: they "
            "are amplified like 1/σ_n). Expect noise fitting above ν_data unless you anneal "
            "(OperatorAwareAnnealing), regularize (TV), stop by the discrepancy principle, or "
            "use a smoother / more structured geometry."
        )
    else:
        lines.append(
            f"- verdict: MATCHED (ν_rep/ν_data = {ratio:.2f}): the pass band of G_θ coincides "
            "with the data-resolvable band."
        )
    if math.isfinite(kappa_rep):
        gain = kappa_grid / kappa_rep if kappa_rep > 0 else math.inf
        if gain >= 1.2:
            what = (
                f"flattens the effective spectrum by {gain:.3g}× (preconditioning: resolvable "
                "detail converges at more uniform rates)"
            )
        elif gain <= 1 / 1.2:
            what = (
                f"steepens the effective spectrum by {1 / gain:.3g}× (spectral bias: fine "
                "resolvable detail converges slowly — implicit regularization, but a slow fit)"
            )
        else:
            what = "leaves the effective spectrum essentially unchanged"
        lines.append(
            f"- conditioning on the resolvable band: κ(g·σ_F²) = {kappa_rep:.3g} vs "
            f"κ(σ_F²) = {kappa_grid:.3g} for a free grid — the representation {what}."
        )
    return MatchReport(
        representation=rep,
        operator=op_spec,
        data=data_spec,
        noise_std=sigma,
        noise_source=src,
        data_bandwidth=nu_data,
        rep_bandwidth=nu_rep,
        nyquist=nyq,
        verdict=verdict,
        ratio=ratio,
        kappa_rep=kappa_rep,
        kappa_grid=kappa_grid,
        method=use,
        progress=float(progress),
        text="\n".join(lines),
    )


__all__ = [
    "MatchReport",
    "Spectrum",
    "estimate_noise_std",
    "field_spectrum",
    "jvp",
    "kernel_row",
    "match_report",
    "operator_spectrum",
    "radial_average",
    "representation_spectrum",
    "resolvable_bandwidth",
    "smooth_octave",
]
