"""Explicit Gaussian-splat parameterization — the *GaussianSplat* baseline (NeTMY App. E.2).

The unknown is a sum of ``K`` axis-aligned anisotropic Gaussian primitives (Kerbl et al. 2023,
"3D Gaussian Splatting", adapted from radiance to density fields)::

    raw(x) = b + Σ_k a_k Π_i exp(-(x_i - μ_ki)² / (2 σ_ki²))

with positions ``μ_k`` in normalized coordinates ``[-1, 1]^d``, per-axis widths
``σ_k = σ_min + exp(s_k)`` and amplitudes ``a_k = softplus(α_k) ≥ 0`` (one per raw head channel).
Heads are applied after the sum, so the representation is resolution-free: the same primitives
render on every curriculum grid (no ``on_stage_start`` work is needed).

Rendering is separable on tensor-product grids (``O(K Σ_i n_i)`` Gaussian evaluations plus one
matrix product) and by default **area-averaged** over each cell with the erf closed form, which
conserves the rendered mass across resolutions and keeps gradients alive for sub-pixel primitives.
Arbitrary point sets fall back to (chunked) point sampling.

Filtering view (NeTMY §4.4, App. D.6): the parameterization Jacobian has at most ``K (2d + C)``
columns, each a localized Gaussian or one of its derivatives, so ``G_θ = J_θ J_θᵀ`` is a low-rank,
strongly *localized* smoother. It cannot represent dense fields with few primitives, but it damps
isolated single-pixel updates on sparse scenes — which is why the NeTMY center-mass ranking places
GaussianSplat between free-density solvers (``G_θ = I``) and the coordinate MLP.

The primitive set has a fixed *capacity* (``max_primitives``, NeTMY cap 128) with an ``active``
mask, so adaptive density control (:class:`SplatControl`) can prune / split / clone primitives
in place without rebuilding the optimizer that the :class:`~nefi.solve.Solver` owns.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..errors import ConfigError, ShapeError
from ..fields.base import Field
from ..fields.heads import Heads
from ..registry import register
from ..solve.callbacks import Callback

log = logging.getLogger("nefi")

_SQRT2 = math.sqrt(2.0)
_SQRT_PI_2 = math.sqrt(math.pi / 2.0)


def _inv_softplus(a: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of ``softplus`` for positive inputs."""
    a = a.clamp_min(1e-12)
    return a + torch.log(-torch.expm1(-a))


def grid_axes(coords: torch.Tensor) -> list[torch.Tensor] | None:
    """Return the 1-D axis vectors if ``coords`` is a tensor-product (``meshgrid``) grid, else None.

    Args:
        coords: ``(*shape, d)`` coordinates with ``len(shape) == d``.
    """
    d = coords.shape[-1]
    if coords.ndim != d + 1:
        return None
    axes = []
    for i in range(d):
        idx: list = [0] * d
        idx[i] = slice(None)
        axes.append(coords[tuple(idx) + (i,)])
    mesh = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    if mesh.shape != coords.shape or not torch.equal(mesh, coords):
        return None
    return axes


def _uniform_spacing(v: torch.Tensor) -> float | None:
    """Cell size of a uniformly spaced axis vector (2.0 for a single cell of ``[-1, 1]``)."""
    if v.numel() == 1:
        return 2.0
    dv = v[1:] - v[:-1]
    h = float(dv.mean())
    if h <= 0 or float((dv - h).abs().max()) > 1e-4 * abs(h):
        return None
    return h


def axis_factor(
    v: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, h: float | None
) -> torch.Tensor:
    """Per-axis Gaussian factor ``(n, K)``: point-sampled (``h=None``) or cell-averaged.

    The cell average of ``exp(-(t-μ)²/(2σ²))`` over ``[v - h/2, v + h/2]`` is
    ``σ √(π/2) / h · [erf(z_hi) - erf(z_lo)]`` with ``z = (v ± h/2 - μ) / (√2 σ)``; it is evaluated
    with ``erfc`` on the far side of the peak to avoid catastrophic cancellation in the tails.
    """
    diff = v[:, None] - mu[None, :]
    if h is None:
        return torch.exp(-0.5 * (diff / sigma) ** 2)
    s = sigma * _SQRT2
    z_hi = (diff + 0.5 * h) / s
    z_lo = (diff - 0.5 * h) / s
    flip = (z_hi + z_lo) < 0  # mirror so the cell midpoint is on the non-negative side
    a = torch.where(flip, -z_hi, z_lo)
    b = torch.where(flip, -z_lo, z_hi)
    tail = torch.erfc(a) - torch.erfc(b)  # exact when a >= 0 (both in the upper tail)
    core = torch.erf(b) - torch.erf(a)  # no cancellation when the cell straddles the peak
    val = torch.where(a >= 0, tail, core)
    return val * (sigma * _SQRT_PI_2 / h)


def _combine(factors: list[torch.Tensor], amp: torch.Tensor) -> torch.Tensor:
    """``out[p0, ..., p_{d-1}, c] = Σ_k Π_i factors[i][p_i, k] · amp[k, c]``."""
    k = amp.shape[0]
    g0 = factors[0]
    if len(factors) == 1:
        return g0 @ amp
    rest = factors[1]
    for f in factors[2:]:
        rest = (rest[:, None, :] * f[None, :, :]).reshape(-1, k)
    ga = g0.unsqueeze(-1) * amp.unsqueeze(0)  # (n0, K, C)
    out = torch.einsum("pkc,mk->pmc", ga, rest)  # (n0, M, C)
    shape = [f.shape[0] for f in factors]
    return out.reshape(*shape, amp.shape[1])


@register("field", "gaussian_splat")
class GaussianSplatField(Field):
    """Sum of anisotropic Gaussian primitives rendered onto the query grid (NeTMY App. E.2).

    Args:
        ndim: coordinate dimension (1, 2 or 3; any ``d`` works for point sets).
        heads: output heads applied to the summed density (``Identity`` reproduces the paper's
            non-negative splat density; positivity heads such as ``Softplus`` are redundant).
        n_primitives: initial number of active primitives (NeTMY: 64).
        max_primitives: capacity / cap on the number of primitives (NeTMY: 128).
        init: ``"grid"`` (regular lattice of ``round(K^(1/d))`` per axis, remainder random) or
            ``"random"`` (uniform in ``[-1, 1]^d``) initial positions.
        init_sigma: initial width in normalized units (``None``: half the lattice spacing).
        init_amplitude: initial amplitude of every primitive.
        min_sigma: width floor ``σ_min`` in normalized units (``σ = σ_min + exp(s)``).
        amplitude: ``"softplus"`` (non-negative, paper) or ``"signed"`` (free sign; useful when a
            bounded head's background must be pushed both up and down).
        background: constant raw offset per channel; ``None`` uses the heads' suggested initial
            biases (0 for ``Identity``), so the empty field equals the heads' initial value.
        learn_background: optimize the background offset.
        render: ``"area"`` (cell-averaged, default) or ``"point"`` (sampled at cell centers).
        pos_lr_mult / scale_lr_mult / amp_lr_mult: per-group learning-rate multipliers implemented
            by reparameterization (``value = mult · raw``), because the core solver uses one LR.
        chunk_size: number of points per chunk on the non-separable (point-set) path.
    """

    def __init__(
        self,
        ndim: int,
        heads: Heads | Mapping | None = None,
        n_primitives: int = 64,
        max_primitives: int = 128,
        init: str = "grid",
        init_sigma: float | None = None,
        init_amplitude: float = 0.1,
        min_sigma: float = 1e-3,
        amplitude: str = "softplus",
        background: float | Sequence[float] | None = None,
        learn_background: bool = False,
        render: str = "area",
        pos_lr_mult: float = 1.0,
        scale_lr_mult: float = 1.0,
        amp_lr_mult: float = 1.0,
        chunk_size: int = 65536,
    ) -> None:
        super().__init__(heads)
        if n_primitives < 1 or max_primitives < n_primitives:
            raise ConfigError(
                f"need 1 <= n_primitives <= max_primitives, got {n_primitives}, {max_primitives}"
            )
        if amplitude not in ("softplus", "signed"):
            raise ConfigError(f"amplitude must be 'softplus' or 'signed', got {amplitude!r}")
        if render not in ("area", "point"):
            raise ConfigError(f"render must be 'area' or 'point', got {render!r}")
        if init not in ("grid", "random"):
            raise ConfigError(f"init must be 'grid' or 'random', got {init!r}")
        self.ndim = int(ndim)
        self.n_init = int(n_primitives)
        self.capacity = int(max_primitives)
        self.init = init
        self.init_sigma = init_sigma
        self.init_amplitude = float(init_amplitude)
        self.min_sigma = float(min_sigma)
        self.amplitude_mode = amplitude
        self.render = render
        self.pos_mult = float(pos_lr_mult)
        self.scale_mult = float(scale_lr_mult)
        self.amp_mult = float(amp_lr_mult)
        self.chunk_size = int(chunk_size)
        c = self.heads.n_in
        k = self.capacity
        self._mu = nn.Parameter(torch.zeros(k, self.ndim))
        self._log_sigma = nn.Parameter(torch.zeros(k, self.ndim))
        self._amp = nn.Parameter(torch.zeros(k, c))
        if background is None:
            bg = self.heads.init_bias().float()
        elif isinstance(background, int | float):
            bg = torch.full((c,), float(background))
        else:
            bg = torch.tensor(list(background), dtype=torch.float32)
        if bg.shape != (c,):
            raise ConfigError(f"background needs {c} values, got {tuple(bg.shape)}")
        if learn_background:
            self.background = nn.Parameter(bg.clone())
        else:
            self.register_buffer("background", bg.clone())
        self.register_buffer("active", torch.zeros(k, dtype=torch.bool))
        self.register_buffer("age", torch.zeros(k, dtype=torch.long))  # control rounds inactive
        self.reset_parameters()

    # ---- parameter transforms -----------------------------------------------------------
    @property
    def means(self) -> torch.Tensor:
        """Positions ``(K_max, d)`` in normalized coordinates."""
        return self._mu * self.pos_mult

    @property
    def sigmas(self) -> torch.Tensor:
        """Per-axis widths ``(K_max, d)`` in normalized units."""
        return self.min_sigma + torch.exp(self._log_sigma * self.scale_mult)

    @property
    def amplitudes(self) -> torch.Tensor:
        """Amplitudes ``(K_max, C)`` (inactive slots included; see :attr:`active`)."""
        a = self._amp * self.amp_mult
        return F.softplus(a) if self.amplitude_mode == "softplus" else a

    @property
    def n_active(self) -> int:
        return int(self.active.sum())

    def _raw_mu(self, mu: torch.Tensor) -> torch.Tensor:
        return mu / self.pos_mult

    def _raw_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.log((sigma - self.min_sigma).clamp_min(1e-12)) / self.scale_mult

    def _raw_amp(self, amp: torch.Tensor) -> torch.Tensor:
        if self.amplitude_mode == "softplus":
            return _inv_softplus(amp) / self.amp_mult
        return amp / self.amp_mult

    # ---- initialization -----------------------------------------------------------------
    def _initial_means(self, k: int) -> tuple[torch.Tensor, float]:
        d = self.ndim
        if self.init == "grid":
            m = max(1, round(k ** (1.0 / d)))
            ax = -1.0 + (2.0 * torch.arange(m, dtype=torch.float32) + 1.0) / m
            lattice = torch.stack(torch.meshgrid(*([ax] * d), indexing="ij"), -1).reshape(-1, d)
            lattice = lattice[:k]
            if lattice.shape[0] < k:
                extra = torch.rand(k - lattice.shape[0], d) * 2.0 - 1.0
                lattice = torch.cat([lattice, extra], 0)
            return lattice, 1.0 / m
        spacing = 2.0 / max(1.0, k ** (1.0 / d))
        return torch.rand(k, d) * 2.0 - 1.0, 0.5 * spacing

    def reset_parameters(self) -> None:
        """Re-initialize the first ``n_primitives`` slots (active) and clear the rest."""
        k0 = self.n_init
        mu0, auto_sigma = self._initial_means(k0)
        sigma0 = float(self.init_sigma) if self.init_sigma is not None else auto_sigma
        sigma0 = max(sigma0, 2.0 * self.min_sigma)
        c = self.heads.n_in
        with torch.no_grad():
            dev, dt = self._mu.device, self._mu.dtype
            self._mu.zero_()
            self._log_sigma.copy_(self._raw_sigma(torch.full_like(self._log_sigma, sigma0)))
            amp0 = torch.full((k0, c), self.init_amplitude)
            self._amp.zero_()
            self._amp[:k0] = self._raw_amp(amp0).to(device=dev, dtype=dt)
            if self.amplitude_mode == "softplus":  # inactive slots: tiny amplitude
                self._amp[k0:] = self._raw_amp(torch.full((1,), 1e-6)).to(device=dev, dtype=dt)
            self._mu[:k0] = self._raw_mu(mu0).to(device=dev, dtype=dt)
            self.active.zero_()
            self.active[:k0] = True
            self.age.zero_()

    # ---- rendering ----------------------------------------------------------------------
    def raw(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        if coords.shape[-1] != self.ndim:
            raise ShapeError(
                f"GaussianSplatField(ndim={self.ndim}) got coords with last dim {coords.shape[-1]}"
            )
        shape = tuple(coords.shape[:-1])
        mu, sig = self.means.to(coords.dtype), self.sigmas.to(coords.dtype)
        amp = self.amplitudes.to(coords.dtype) * self.active.unsqueeze(-1).to(coords.dtype)
        axes = grid_axes(coords)
        if axes is not None:
            factors = []
            for i, v in enumerate(axes):
                h = _uniform_spacing(v) if self.render == "area" else None
                factors.append(axis_factor(v, mu[:, i], sig[:, i], h))
            dens = _combine(factors, amp)
        else:
            flat = coords.reshape(-1, self.ndim)
            chunks = []
            for s in range(0, flat.shape[0], self.chunk_size):
                x = flat[s : s + self.chunk_size]
                z = ((x[:, None, :] - mu[None]) / sig[None]) ** 2
                chunks.append(torch.exp(-0.5 * z.sum(-1)) @ amp)
            dens = torch.cat(chunks, 0).reshape(*shape, amp.shape[1])
        return dens + self.background.to(dens.dtype)

    # ---- primitive management (used by SplatControl; all in-place, no optimizer rebuild) ----
    def free_slots(self) -> torch.Tensor:
        """Indices of inactive slots, longest-inactive first (limits stale optimizer moments)."""
        idx = (~self.active).nonzero().flatten()
        order = torch.argsort(self.age[idx], descending=True, stable=True)
        return idx[order]

    @torch.no_grad()
    def deactivate(self, idx: torch.Tensor) -> None:
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.active.device).flatten()
        self.active[idx] = False
        self.age[idx] = 0
        if self.amplitude_mode == "softplus":
            self._amp[idx] = self._raw_amp(torch.full((1,), 1e-6)).to(self._amp)
        else:
            self._amp[idx] = 0.0

    @torch.no_grad()
    def write(
        self, idx: torch.Tensor, means: torch.Tensor, sigmas: torch.Tensor, amps: torch.Tensor
    ) -> None:
        """Set slots ``idx`` to the given primitives and mark them active."""
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.active.device).flatten()
        self._mu[idx] = self._raw_mu(means.to(self._mu))
        self._log_sigma[idx] = self._raw_sigma(sigmas.to(self._log_sigma))
        self._amp[idx] = self._raw_amp(amps.to(self._amp))
        self.active[idx] = True
        self.age[idx] = 0

    @torch.no_grad()
    def add(self, means: torch.Tensor, sigmas: torch.Tensor, amps: torch.Tensor) -> torch.Tensor:
        """Place new primitives into free slots (as many as fit); returns their slot indices."""
        free = self.free_slots()
        n = min(int(free.numel()), int(means.shape[0]))
        slots = free[:n]
        if n > 0:
            self.write(slots, means[:n], sigmas[:n], amps[:n])
        return slots

    @torch.no_grad()
    def set_primitives(self, means: torch.Tensor, sigmas: torch.Tensor, amps: torch.Tensor) -> None:
        """Replace the whole primitive set (``len(means) <= max_primitives``)."""
        k = means.shape[0]
        if k > self.capacity:
            raise ConfigError(f"{k} primitives exceed max_primitives={self.capacity}")
        self.deactivate(torch.arange(self.capacity))
        amps = amps.reshape(k, -1).expand(k, self.heads.n_in)
        self.write(torch.arange(k), means.reshape(k, self.ndim), sigmas.reshape(k, self.ndim), amps)

    def extra_repr(self) -> str:
        return (
            f"ndim={self.ndim}, active={self.n_active}/{self.capacity}, render={self.render}, "
            f"amplitude={self.amplitude_mode}, heads={self.heads.names}"
        )


@register("callback", "splat_control")
class SplatControl(Callback):
    """Adaptive density control for :class:`GaussianSplatField` (Kerbl et al. 2023 §5).

    Every ``every`` steps between ``start`` and ``stop_fraction · total_steps`` it

    1. **prunes** primitives whose amplitude is tiny (``< max(prune_abs, prune_rel · max_k a_k)``)
       or that drifted outside the domain (``|μ| > 1 + 3σ`` on some axis);
    2. selects under-reconstructed primitives by their *average positional gradient norm* over the
       window (``>= max(grad_threshold, quantile(grad_quantile))``) and
       **splits** the large ones (``max σ > split_sigma_cells`` cells) into two children with
       widths ``σ / split_factor`` sampled from the parent (mass preserving), or
       **clones** the small ones (half the mass moved one σ along the descent direction);
    3. optionally **merges** primitives closer than ``merge_cells`` cells (NeTMY's "merge").

    Growth never exceeds the field capacity (``max_primitives``, NeTMY cap 128) and at most
    ``max_new_fraction · n_active`` primitives are added per round. All edits are in place, so the
    solver's optimizer keeps working (new slots start from the longest-unused Adam moments).

    Attach it to the :class:`~nefi.solve.Solver` callbacks, or put it in
    ``problem.meta["callbacks"]`` and run through :func:`nefi.baselines.solve`.
    """

    def __init__(
        self,
        every: int = 100,
        start: int = 100,
        stop_fraction: float = 0.8,
        grad_quantile: float = 0.8,
        grad_threshold: float = 0.0,
        split_sigma_cells: float = 2.0,
        split_factor: float = 1.6,
        prune_rel: float = 0.01,
        prune_abs: float = 0.0,
        prune_outside: bool = True,
        merge_cells: float | None = None,
        max_new_fraction: float = 0.5,
        min_primitives: int = 1,
    ) -> None:
        self.every, self.start = int(every), int(start)
        self.stop_fraction = float(stop_fraction)
        self.grad_quantile, self.grad_threshold = float(grad_quantile), float(grad_threshold)
        self.split_sigma_cells, self.split_factor = float(split_sigma_cells), float(split_factor)
        self.prune_rel, self.prune_abs = float(prune_rel), float(prune_abs)
        self.prune_outside = prune_outside
        self.merge_cells = merge_cells
        self.max_new_fraction = float(max_new_fraction)
        self.min_primitives = int(min_primitives)
        self.field: GaussianSplatField | None = None
        self.cell = 2.0 / 64
        self.stop_step: int | None = None
        self.events: list[dict] = []
        self._acc: torch.Tensor | None = None
        self._cnt: torch.Tensor | None = None
        self._dir: torch.Tensor | None = None

    # ---- callback hooks -----------------------------------------------------------------
    def on_run_start(self, solver) -> None:
        field = solver.problem.field
        self.field = field if isinstance(field, GaussianSplatField) else None
        if self.field is None:
            log.warning("SplatControl: problem field is not a GaussianSplatField; disabled")
        total = getattr(getattr(solver, "curriculum", None), "total_steps", None)
        self.stop_step = None if total is None else int(self.stop_fraction * total)
        self.events = []

    def on_stage_start(self, solver, stage_idx, stage) -> None:
        shape = stage.shape or tuple(solver.problem.domain.shape)
        self.cell = 2.0 / max(1, min(shape))
        self._reset_stats()

    def on_step(self, solver, state) -> None:
        f = self.field
        if f is None:
            return
        g = f._mu.grad
        if g is not None:
            self.accumulate(g.detach() / f.pos_mult)
        gs = state.global_step
        if gs < self.start or (self.stop_step is not None and gs >= self.stop_step):
            return
        if (gs - self.start) % self.every == 0 and gs > 0:
            info = self.apply(f)
            info["global_step"] = gs
            self.events.append(info)
            log.debug("SplatControl @%d: %s", gs, info)

    # ---- statistics ---------------------------------------------------------------------
    def _reset_stats(self) -> None:
        self._acc = self._cnt = self._dir = None

    def accumulate(self, grad_mu: torch.Tensor) -> None:
        """Accumulate per-slot positional gradients ``∂L/∂μ`` ``(K_max, d)`` of one step."""
        f = self.field
        if f is None:
            return
        act = f.active.to(grad_mu.device)
        norm = grad_mu.norm(dim=-1) * act
        if self._acc is None:
            self._acc = torch.zeros_like(norm)
            self._cnt = torch.zeros_like(norm)
            self._dir = torch.zeros_like(grad_mu)
        self._acc += norm
        self._cnt += act.to(norm.dtype)
        self._dir += grad_mu * act.unsqueeze(-1)

    # ---- the density-control round ------------------------------------------------------
    @torch.no_grad()
    def apply(self, field: GaussianSplatField | None = None) -> dict:
        """Run one prune / split / clone (/ merge) round now; returns counts."""
        f = field if field is not None else self.field
        if f is None:
            return {}
        self.field = f
        dev = f.active.device
        info = {"pruned": 0, "split": 0, "cloned": 0, "merged": 0}
        f.age[~f.active] += 1

        # 1) prune tiny / escaped primitives
        amp = f.amplitudes.abs().amax(-1)
        act = f.active.clone()
        if act.any():
            thr = max(self.prune_abs, self.prune_rel * float(amp[act].max()))
            bad = act & (amp < thr)
            if self.prune_outside:
                out = (f.means.abs() > 1.0 + 3.0 * f.sigmas).any(-1)
                bad |= act & out
            keep_min = max(0, int(act.sum()) - self.min_primitives)
            if int(bad.sum()) > keep_min:  # never prune below min_primitives
                cand = bad.nonzero().flatten()
                cand = cand[torch.argsort(amp[cand])][:keep_min]
                bad = torch.zeros_like(bad)
                bad[cand] = True
            if bad.any():
                f.deactivate(bad.nonzero().flatten())
                info["pruned"] = int(bad.sum())

        # 2) densify by average positional gradient
        if self._acc is not None and f.n_active > 0:
            cnt = self._cnt.to(dev)
            avg = torch.where(cnt > 0, self._acc.to(dev) / cnt.clamp_min(1), torch.zeros_like(cnt))
            act = f.active & (cnt > 0)
            if act.any():
                q = float(torch.quantile(avg[act], self.grad_quantile))
                thr = max(self.grad_threshold, q)
                cand = (act & (avg >= thr) & (avg > 0)).nonzero().flatten()
                cand = cand[torch.argsort(avg[cand], descending=True)]
                budget = min(
                    f.capacity - f.n_active,
                    max(1, int(self.max_new_fraction * f.n_active)),
                )
                cand = cand[: max(0, budget)]
                if cand.numel() > 0:
                    info["split"], info["cloned"] = self._densify(f, cand)

        # 3) optional merge of near-duplicates
        if self.merge_cells is not None and f.n_active > 1:
            info["merged"] = self._merge(f, float(self.merge_cells) * self.cell)

        info["n_active"] = f.n_active
        self._reset_stats()
        return info

    def _densify(self, f: GaussianSplatField, cand: torch.Tensor) -> tuple[int, int]:
        """Split large / clone small candidates; each adds exactly one primitive."""
        mu, sig, amp = f.means[cand], f.sigmas[cand], f.amplitudes[cand]
        large = sig.amax(-1) > self.split_sigma_cells * self.cell
        n_split = n_clone = 0
        # split: parent slot -> child 1, a free slot -> child 2; 2 a_c Π σ_c = a_p Π σ_p
        sp = large.nonzero().flatten()[: max(0, f.capacity - f.n_active)]
        if sp.numel() > 0:
            p_mu, p_sig, p_amp = mu[sp], sig[sp], amp[sp]
            c_sig = (p_sig / self.split_factor).clamp_min(1.01 * f.min_sigma)
            c_amp = p_amp * (torch.prod(p_sig / c_sig, dim=-1, keepdim=True) / 2.0)
            off = torch.randn_like(p_mu) * p_sig
            f.write(cand[sp], p_mu + off, c_sig, c_amp)
            f.add(p_mu - off, c_sig, c_amp)
            n_split = int(sp.numel())
        # clone: half the mass stays, half moves one sigma along the descent direction
        cl = (~large).nonzero().flatten()[: max(0, f.capacity - f.n_active)]
        if cl.numel() > 0:
            c_mu, c_sig, c_amp = mu[cl], sig[cl], amp[cl]
            if self._dir is not None:
                gdir = -self._dir.to(c_mu)[cand[cl]]
            else:
                gdir = torch.randn_like(c_mu)
            gdir = gdir / gdir.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            half = 0.5 * c_amp
            f.write(cand[cl], c_mu, c_sig, half)
            f.add(c_mu + gdir * c_sig, c_sig, half)
            n_clone = int(cl.numel())
        return n_split, n_clone

    def _merge(self, f: GaussianSplatField, radius: float) -> int:
        idx = f.active.nonzero().flatten()
        mu, sig, amp = f.means[idx], f.sigmas[idx], f.amplitudes[idx]
        mass = amp.abs().amax(-1) * torch.prod(sig, -1)
        dist = torch.cdist(mu, mu)
        dist.fill_diagonal_(float("inf"))
        merged = 0
        alive = torch.ones(idx.numel(), dtype=torch.bool, device=mu.device)
        pairs = (dist < radius).nonzero()
        for i, j in pairs.tolist():
            if i >= j or not (alive[i] and alive[j]):
                continue
            w = mass[i] / (mass[i] + mass[j]).clamp_min(1e-30)
            new_mu = w * mu[i] + (1 - w) * mu[j]
            new_sig = torch.maximum(sig[i], sig[j])
            new_amp = (amp[i] * torch.prod(sig[i]) + amp[j] * torch.prod(sig[j])) / torch.prod(
                new_sig
            )
            f.write(idx[i : i + 1], new_mu[None], new_sig[None], new_amp[None])
            f.deactivate(idx[j : j + 1])
            alive[j] = False
            merged += 1
        return merged


__all__ = ["GaussianSplatField", "SplatControl", "axis_factor", "grid_axes"]
