"""Adaptive annealing: open frequency bands when the data ask for them.

The papers anneal on a fixed clock: ``β = K · t / T`` per stage (NeTMY Eq. 27, Tab. 6) or over the
first ``T_FA`` steps (NeFTY Eq. 21, ``T_FA = 2500``). Two quantities decide when a band *should*
open instead:

* the **residual** — a band is worth opening once the currently open bands have explained what
  they can (the data loss plateaus): :class:`ResidualDrivenAnnealing` ("earn your frequencies");
* the **operator and the noise** — a band carries information only if the operator's sensitivity
  times the field's amplitude at that frequency exceeds the noise (NeTMY Lemma 1 ``e^{−k z0}``,
  NeFTY Prop. 2 / Cor. 1): :class:`OperatorAwareAnnealing` never opens bands the data cannot
  resolve, which is the representation-level version of the discrepancy principle.

Both are :class:`~nefi.solve.callbacks.Callback` s that drive ``solver.progress_override`` (the
global annealing progress; composite fields map it per component). :class:`BandwidthSchedule`
prescribes the open bandwidth (cycles per unit normalized coordinate) as a function of the stage
fraction, and :func:`bandwidth_to_progress` / :func:`progress_to_bandwidth` convert between the two
for any field with an annealed :class:`~nefi.fields.encoding.FourierFeatures` encoding or a
:class:`~nefi.fields.geometric.FourierBasisField`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn

from ...errors import ConfigError
from ...registry import register
from ...solve.callbacks import Callback, StepState
from ..encoding import FourierFeatures
from .spectrum import Spectrum, _band_end, _interp, field_spectrum

log = logging.getLogger("nefi")


# ------------------------------------------------------------------------------------------
# plateau detection
# ------------------------------------------------------------------------------------------
class PlateauDetector:
    """Detect when a monitored loss stops paying off (windowed, scale-free).

    The improvement over the last ``patience`` updates, ``imp = (L_{t−p} − L_t) / |L_{t−p}|``
    (``relative``) or ``L_{t−p} − L_t`` (absolute), is compared with

        threshold = max(delta, rate_fraction · peak),

    where ``peak`` is the largest windowed improvement seen since the last :meth:`reset`.
    ``rate_fraction = 0`` is the classic criterion ("less than ``delta`` improvement in
    ``patience`` steps"); ``rate_fraction > 0`` adds *diminishing returns* ("the current
    capacity now improves the fit at less than a fraction of its best rate"), which works across
    loss scales and optimizer speeds — a tanh MLP with few open bands keeps improving slowly
    forever, so an absolute threshold alone would stall.

    Args:
        patience: window length (updates); also the minimum number of updates after a reset.
        delta: improvement below which the window counts as a plateau.
        relative: relative (default) or absolute improvements.
        min_steps: minimum number of updates before any plateau can be signalled (at least
            ``patience``).
        smoothing: EMA factor applied to the monitored value (0 = raw values).
        rate_fraction: diminishing-returns fraction of the peak windowed improvement.
    """

    def __init__(
        self,
        patience: int = 50,
        delta: float = 1e-3,
        relative: bool = True,
        min_steps: int = 0,
        smoothing: float = 0.0,
        rate_fraction: float = 0.0,
    ) -> None:
        if patience < 1:
            raise ConfigError("patience must be >= 1")
        if not 0.0 <= smoothing < 1.0:
            raise ConfigError("smoothing must be in [0, 1)")
        if not 0.0 <= rate_fraction < 1.0:
            raise ConfigError("rate_fraction must be in [0, 1)")
        self.patience, self.delta, self.relative = int(patience), float(delta), bool(relative)
        self.min_steps, self.smoothing = max(int(min_steps), int(patience)), float(smoothing)
        self.rate_fraction = float(rate_fraction)
        self.reset()

    def reset(self) -> None:
        self.values: list[float] = []
        self.peak = 0.0
        self.last_improvement = math.nan
        self.steps = 0
        self._ema: float | None = None

    def update(self, value: float) -> bool:
        """Feed one value; returns True when a plateau is detected."""
        value = float(value)
        if not math.isfinite(value):
            return False
        self.steps += 1
        if self.smoothing > 0:
            self._ema = (
                value
                if self._ema is None
                else self.smoothing * self._ema + (1 - self.smoothing) * value
            )
            value = self._ema
        self.values.append(value)
        if len(self.values) > self.patience + 1:
            self.values.pop(0)
        if len(self.values) <= self.patience:
            return False
        old = self.values[0]
        imp = old - value
        if self.relative:
            imp = imp / max(abs(old), 1e-300)
        self.last_improvement = imp
        self.peak = max(self.peak, imp)
        threshold = max(self.delta, self.rate_fraction * self.peak)
        return self.steps >= self.min_steps and imp < threshold


# ------------------------------------------------------------------------------------------
# bandwidth <-> progress
# ------------------------------------------------------------------------------------------
def find_band_module(field: nn.Module) -> nn.Module | None:
    """First module with an annealed band structure (annealed ``FourierFeatures`` or any module
    exposing ``band_frequencies()`` and ``effective_bandwidth(progress)``)."""
    for m in field.modules():
        if isinstance(m, FourierFeatures):
            if m.annealed and m.n_octaves > 0:
                return m
        elif callable(getattr(m, "band_frequencies", None)) and callable(
            getattr(m, "effective_bandwidth", None)
        ):
            return m
    return None


def _band_module(obj: Any) -> nn.Module:
    if isinstance(obj, FourierFeatures) or callable(getattr(obj, "band_frequencies", None)):
        return obj
    if isinstance(obj, nn.Module):
        m = find_band_module(obj)
        if m is not None:
            return m
    raise ConfigError(
        f"{type(obj).__name__} has no annealed band structure (FourierFeatures or "
        "FourierBasisField); bandwidth <-> progress mapping is undefined"
    )


def band_frequencies(obj: Any) -> list[float]:
    """Frequency (cycles per unit normalized coordinate) opened by each successive band."""
    m = _band_module(obj)
    if isinstance(m, FourierFeatures):
        return [float(f) / (2.0 * math.pi) for f in m.freqs.tolist()]
    return [float(f) for f in m.band_frequencies()]  # type: ignore[operator]


def n_levels(obj: Any, default: int = 8) -> int:
    """Number of annealing levels (bands) of a field / encoding (``default`` if none)."""
    try:
        return len(band_frequencies(obj))
    except ConfigError:
        return int(default)


def progress_to_bandwidth(obj: Any, progress: float) -> float:
    """Highest open frequency at ``progress`` (``effective_bandwidth``, NeTMY Eq. 37)."""
    m = _band_module(obj)
    return float(m.effective_bandwidth(progress))  # type: ignore[operator]


def bandwidth_to_progress(obj: Any, bandwidth: float) -> float:
    """Smallest progress at which the band containing ``bandwidth`` is fully open.

    Piecewise linear through ``(0, 0), (f_0, 1/K), …, (f_{K−1}, 1)`` where band ``k`` (frequency
    ``f_k``) is fully open at ``β = k + 1`` (cosine gate, NeTMY Eq. 27 / NeFTY Eq. 21).
    """
    f = band_frequencies(obj)
    k = len(f)
    if k == 0 or bandwidth <= 0:
        return 0.0
    if bandwidth >= f[-1] * (1.0 - 1e-6):
        return 1.0
    xs = torch.tensor([0.0] + f, dtype=torch.float64)
    ys = torch.arange(k + 1, dtype=torch.float64)
    beta = float(_interp(torch.tensor(float(bandwidth), dtype=torch.float64), xs, ys))
    if abs(beta - round(beta)) < 1e-5:  # band edges given in float32 buffers
        beta = float(round(beta))
    return min(max(beta / k, 0.0), 1.0)


# ------------------------------------------------------------------------------------------
# residual-driven annealing
# ------------------------------------------------------------------------------------------
def stage_noise_std(problem: Any, shape: Sequence[int] | None, scale: bool = True) -> float | None:
    """The measurement noise std at a curriculum stage's resolution.

    Coarse stages compare against area-averaged data, whose noise std is smaller by
    ``sqrt(#native entries / #stage entries)``; ``None`` if the noise level is unknown.
    """
    ns = problem.measurement.noise_std
    if ns is None:
        return None
    sigma = float(ns.mean()) if torch.is_tensor(ns) else float(ns)
    if scale and shape is not None and tuple(shape) != tuple(problem.domain.shape):
        n_stage = problem.measurement_at(tuple(shape)).data.numel()
        sigma *= math.sqrt(n_stage / max(problem.measurement.data.numel(), 1))
    return sigma


@register("callback", "residual_annealing")
class ResidualDrivenAnnealing(Callback):
    """Open the next frequency band only when the data loss plateaus ("earn your frequencies").

    Progress moves on a discrete ladder ``L/K`` (``L`` open bands of the field's annealed
    encoding). A band is opened when the monitored loss plateaus — improves by less than
    ``max(delta, rate_fraction × its best rate at this level)`` over ``patience`` steps — *and*
    the residual is still above the noise floor: once ``RMSE ≤ discrepancy_tau · σ`` (Morozov's
    discrepancy principle, with ``σ`` the measurement noise at the stage resolution) no further
    band is opened, because anything the new band could fit is noise (NeFTY Cor. 1). Newly
    opened bands ramp in over ``ramp_steps`` and the plateau test pauses meanwhile: switching a
    band on abruptly feeds the untrained first-layer weights of its features and injects a
    random high-frequency perturbation (the shock the cosine gate avoids, NeTMY App. D.2). The
    ladder
    restarts at ``start_level`` every stage (the per-stage β reset of NeTMY Tab. 6) unless
    ``reset_each_stage=False``. ``max_steps_per_level`` adds a clock (open after at most that many
    steps even without a plateau) for fixed budgets.

    Args:
        patience: plateau window (steps).
        delta: minimum relative improvement over the window.
        n_levels: number of bands ``K`` (default: from the field's encoding / Fourier basis).
        start_level: open bands at stage start.
        ramp_steps: steps over which a newly opened band ramps in (long ramps matter: see
            above).
        relative: relative (default) or absolute ``delta``.
        min_steps_per_level: minimum steps between openings (default ``patience``).
        max_steps_per_level: open the next band after at most this many steps (``None`` = pure
            residual criterion).
        discrepancy_tau: noise-floor factor ``τ`` (``None`` disables the noise floor; inactive
            when the measurement has no ``noise_std``).
        stop_at_noise: end the stage once the noise floor is reached and the loss plateaus.
        reset_each_stage: restart the ladder at each curriculum stage.
        stop_at_full: end the stage (``solver.stop_stage``) on a plateau with all bands open.
        metric: ``"data"`` (``StepState.data_loss``) or ``"total"``.
        smoothing: EMA factor on the monitored loss.
        rate_fraction: diminishing-returns criterion — a level is exhausted once its windowed
            improvement falls below this fraction of its best windowed improvement (see
            :class:`PlateauDetector`; ``0`` = pure ``delta`` plateau).
        module: the band module to count levels from (default: auto-detected).

    Attributes:
        events: list of ``{"stage", "step", "global_step", "level", "loss", "reason"}``.
    """

    def __init__(
        self,
        patience: int = 25,
        delta: float = 1e-3,
        n_levels: int | None = None,
        start_level: int = 0,
        ramp_steps: int = 50,
        relative: bool = True,
        min_steps_per_level: int | None = None,
        max_steps_per_level: int | None = None,
        discrepancy_tau: float | None = 1.1,
        stop_at_noise: bool = False,
        reset_each_stage: bool = True,
        stop_at_full: bool = False,
        metric: str = "data",
        smoothing: float = 0.0,
        rate_fraction: float = 0.25,
        module: nn.Module | None = None,
    ) -> None:
        if metric not in ("data", "total"):
            raise ConfigError("metric must be 'data' or 'total'")
        self.detector = PlateauDetector(
            patience,
            delta,
            relative,
            min_steps=patience if min_steps_per_level is None else min_steps_per_level,
            smoothing=smoothing,
            rate_fraction=rate_fraction,
        )
        self.n_levels_arg, self.start_level = n_levels, int(start_level)
        self.ramp_steps = max(0, int(ramp_steps))
        self.max_steps_per_level = max_steps_per_level
        self.discrepancy_tau = discrepancy_tau
        self.stop_at_noise = bool(stop_at_noise)
        self.reset_each_stage, self.stop_at_full = bool(reset_each_stage), bool(stop_at_full)
        self.metric, self.module = metric, module
        self.K = 1
        self.level = self.start_level
        self.progress = 0.0
        self.at_noise_floor = False
        self._ramp: tuple[float, float, int] | None = None
        self._started = False
        self._since_open = 0
        self._sigma: float | None = None
        self._obs: Any = None
        self.events: list[dict[str, Any]] = []

    def _count_levels(self, solver) -> int:
        if self.n_levels_arg is not None:
            return max(1, int(self.n_levels_arg))
        src = self.module if self.module is not None else solver.problem.field
        return max(1, n_levels(src))

    def on_run_start(self, solver) -> None:
        self.K = self._count_levels(solver)
        self.events = []
        self._started = False

    def on_stage_start(self, solver, stage_idx, stage) -> None:
        if not self._started:
            self.K = self._count_levels(solver)
        if self.reset_each_stage or not self._started:
            self.level = min(max(self.start_level, 0), self.K)
            self.progress = self.level / self.K
        self._started = True
        self._ramp = None
        self._since_open = 0
        self.at_noise_floor = False
        self.detector.reset()
        p = solver.problem
        shape = p.domain.shape if stage.shape is None else tuple(stage.shape)
        self._sigma = None if self.discrepancy_tau is None else stage_noise_std(p, shape)
        self._obs = p.measurement_at(shape) if self._sigma is not None else None
        solver.progress_override = self.progress

    def _noise_floor(self, state: StepState) -> bool:
        if self._sigma is None or self._obs is None or state.pred is None:
            return False
        obs = self._obs
        pred = state.pred.detach()
        if tuple(obs.data.shape) != tuple(pred.shape):
            return False
        obs = obs.to(pred.device, pred.dtype)
        rmse = float(torch.sqrt(obs.masked_mean((pred - obs.data) ** 2)))
        return rmse <= float(self.discrepancy_tau) * self._sigma  # type: ignore[arg-type]

    def _open(self, state: StepState, value: float, reason: str) -> None:
        self.level += 1
        target = self.level / self.K
        if self.ramp_steps > 0:
            self._ramp = (self.progress, target, self.ramp_steps)
        else:
            self.progress = target
        self._since_open = 0
        self.events.append(
            {
                "stage": state.stage_idx,
                "step": state.step,
                "global_step": state.global_step,
                "level": self.level,
                "loss": float(value),
                "reason": reason,
            }
        )
        log.debug("ResidualDrivenAnnealing: opened band %d/%d (%s)", self.level, self.K, reason)

    def on_step(self, solver, state: StepState) -> None:
        if self._ramp is not None:
            a, b, left = self._ramp
            left -= 1
            self.progress = b - (b - a) * left / max(1, self.ramp_steps)
            self._ramp = None if left <= 0 else (a, b, left)
        self._since_open += 1
        value = state.data_loss if self.metric == "data" else state.total
        if self._ramp is not None:  # a band is still ramping in: judge the level afterwards
            self.detector.reset()
            solver.progress_override = self.progress
            return
        plateau = self.detector.update(value)
        clock = self.max_steps_per_level is not None and self._since_open >= int(
            self.max_steps_per_level
        )
        if plateau or clock:
            self.at_noise_floor = self._noise_floor(state)
            if self.at_noise_floor:
                if self.stop_at_noise and plateau:
                    solver.stop_stage = "noise_floor"
            elif self.level < self.K:
                self._open(state, value, "plateau" if plateau else "clock")
            elif self.stop_at_full and plateau:
                solver.stop_stage = "residual_plateau_full_band"
            self.detector.reset()
            self._since_open = 0 if clock else self._since_open
        solver.progress_override = self.progress


# ------------------------------------------------------------------------------------------
# operator-aware annealing
# ------------------------------------------------------------------------------------------
@register("callback", "operator_annealing")
class OperatorAwareAnnealing(Callback):
    """Open frequency ``ν`` only when ``σ_F(ν) × |X̂(ν)|`` exceeds the noise.

    Every ``update_every`` steps the current field's radial amplitude spectrum ``|X̂(ν)|`` is
    measured on the stage grid; above the open band it is extrapolated by a power law fitted to
    the open band (a field cannot show energy in bands that are still closed). With the operator's
    amplitude sensitivity ``σ_F(ν)`` (e.g. ``operator_spectrum(...)`` or a precomputed 1-D radial
    profile, NeTMY Lemma 1 / NeFTY Prop. 2), the predicted data signal per mode is
    ``D(ν) = σ_F(ν) |X̂(ν)|``; the target bandwidth ``ν*`` is the end of the contiguous band where
    ``D ≥ margin · σ_ε`` (per-mode noise std of the orthonormal FFT = the entry noise std). The
    progress follows ``bandwidth_to_progress(ν*)``, monotonically and at most ``max_rate`` per
    step. ``signal="data"`` instead uses the measurement spectrum (needs the measurement on the
    field grid): ``|Ŷ(ν)| ≥ margin · σ_ε`` — a fixed, data-driven Picard cutoff.

    Coarse curriculum stages see area-averaged data: the noise level is scaled by
    ``sqrt(#stage entries / #native entries)`` (``scale_noise=True``), consistent with the
    orthonormal field spectrum on the coarse grid.

    Args:
        sensitivity: a :class:`~.spectrum.Spectrum` (``values`` = ``σ_F``) or a 1-D tensor of
            ``σ_F`` samples at ``freqs`` (cycles per unit normalized coordinate).
        noise_level: entry noise std (default: the measurement's ``noise_std``).
        freqs: frequencies of a tensor ``sensitivity``.
        field: monitored field name (default primary).
        signal: ``"field"`` (adaptive) or ``"data"`` (fixed Picard cutoff).
        update_every: steps between target updates.
        margin: SNR margin (amplitude ratio).
        max_rate: maximum progress increase per step (default ``2 / stage.steps``).
        decay_exponent: default power-law slope when the fit is under-determined.
        start: progress at stage start.
        min_progress: floor of the progress (e.g. ``1/K`` keeps the first band open).
        reset_each_stage: restart at ``start`` each stage.
        scale_noise: scale the noise with the stage resolution (see above).
        module: band module for the progress mapping (default: auto-detected).

    Attributes:
        history: list of ``{"step", "global_step", "target_bandwidth", "progress"}``.
    """

    def __init__(
        self,
        sensitivity: Spectrum | torch.Tensor | Sequence[float],
        noise_level: float | None = None,
        freqs: torch.Tensor | Sequence[float] | None = None,
        *,
        field: str | None = None,
        signal: str = "field",
        update_every: int = 25,
        margin: float = 1.0,
        max_rate: float | None = None,
        decay_exponent: float = 1.0,
        start: float = 0.0,
        min_progress: float = 0.0,
        reset_each_stage: bool = True,
        scale_noise: bool = True,
        module: nn.Module | None = None,
    ) -> None:
        if isinstance(sensitivity, Spectrum):
            f, s = sensitivity.freqs, sensitivity.values
        else:
            if freqs is None:
                raise ConfigError("a tensor sensitivity profile needs `freqs`")
            f = torch.as_tensor(freqs, dtype=torch.float64).flatten()
            s = torch.as_tensor(sensitivity, dtype=torch.float64).flatten()
        if f.numel() != s.numel() or f.numel() < 2:
            raise ConfigError("sensitivity profile needs >= 2 samples and matching freqs")
        order = torch.argsort(f)
        self.sens_freqs, self.sens_values = f[order].double(), s[order].double().abs()
        if signal not in ("field", "data"):
            raise ConfigError("signal must be 'field' or 'data'")
        self.noise_level = noise_level
        self.field, self.signal = field, signal
        self.update_every = max(1, int(update_every))
        self.margin, self.max_rate = float(margin), max_rate
        self.decay_exponent = float(decay_exponent)
        self.start, self.min_progress = float(start), float(min_progress)
        self.reset_each_stage, self.scale_noise = bool(reset_each_stage), bool(scale_noise)
        self.module = module
        self.progress = self.start
        self.history: list[dict[str, float]] = []
        self._rate = 0.0
        self._noise = 1.0
        self._band: nn.Module | None = None
        self._data_target: float | None = None
        self._started = False

    # --- helpers ------------------------------------------------------------------------
    def sensitivity_at(self, f: torch.Tensor) -> torch.Tensor:
        return _interp(f, self.sens_freqs, self.sens_values)

    def target_bandwidth(self, x: torch.Tensor, progress: float) -> float:
        """Resolvable bandwidth predicted from the current field ``x`` (stage grid)."""
        spec = field_spectrum(x)
        f, a = spec.freqs.double(), spec.values.double()
        b_open = progress_to_bandwidth(self._band, progress) if self._band is not None else 0.0
        bw = float(f[1]) if f.numel() > 1 else 1.0
        fit = (f > 0) & (f <= max(b_open, 2 * bw) + 1e-9) & (a > 0)
        est = a.clone()
        beyond = f > max(b_open, 2 * bw) + 1e-9
        if int(fit.sum()) >= 3:
            lx, ly = torch.log(f[fit]), torch.log(a[fit])
            slope = float(
                ((lx - lx.mean()) * (ly - ly.mean())).sum() / ((lx - lx.mean()) ** 2).sum()
            )
            slope = min(slope, 0.0)
            inter = float(ly.mean() - slope * lx.mean())
            est[beyond] = torch.exp(inter + slope * torch.log(f[beyond]))
        elif bool(fit.any()):
            i = int(fit.nonzero().max())
            est[beyond] = a[i] * (f[beyond] / f[i]) ** (-self.decay_exponent)
        d = self.sensitivity_at(f) * est
        return _band_end(f, d / self._noise, self.margin, spec.counts)

    # --- callback API ---------------------------------------------------------------------
    def on_stage_start(self, solver, stage_idx, stage) -> None:
        p = solver.problem
        src = self.module if self.module is not None else p.field
        try:
            self._band = _band_module(src)
        except ConfigError:
            self._band = None
            log.warning(
                "OperatorAwareAnnealing: no annealed band structure found; progress moves "
                "linearly with the target bandwidth fraction of Nyquist"
            )
        dom = p.domain if stage.shape is None else p.domain.at(stage.shape)
        if self.noise_level is not None:
            noise = float(self.noise_level)
            if self.scale_noise and tuple(dom.shape) != tuple(p.domain.shape):
                n_stage = p.measurement_at(dom.shape).data.numel()
                noise *= math.sqrt(n_stage / max(p.measurement.data.numel(), 1))
        else:
            noise = stage_noise_std(p, dom.shape, self.scale_noise)
            if noise is None:
                raise ConfigError(
                    "OperatorAwareAnnealing needs a noise level (noise_level=... or "
                    "Measurement.noise_std)"
                )
        self._noise = max(float(noise), 1e-300)
        self._rate = self.max_rate if self.max_rate is not None else 2.0 / max(1, stage.steps)
        if self.reset_each_stage or not self._started:
            self.progress = max(self.start, self.min_progress)
        self._started = True
        self._data_target = None
        if self.signal == "data":
            obs = p.measurement_at(dom.shape).data.detach()
            if tuple(obs.shape) != tuple(dom.shape):
                raise ConfigError("signal='data' needs the measurement on the field grid")
            spec = field_spectrum(obs, kind="data")
            self._data_target = _band_end(
                spec.freqs, spec.values / self._noise, self.margin, spec.counts
            )
        solver.progress_override = self.progress

    def _progress_for(self, bandwidth: float, nyquist: float) -> float:
        if self._band is not None:
            return bandwidth_to_progress(self._band, bandwidth)
        return min(max(bandwidth / max(nyquist, 1e-12), 0.0), 1.0)

    def on_step(self, solver, state: StepState) -> None:
        if state.step % self.update_every == 0:
            if self._data_target is not None:
                target_bw = self._data_target
                nyq = target_bw
            else:
                if state.fields is None:
                    log.warning(
                        "OperatorAwareAnnealing needs StepState.fields (keep_fields_in_state)"
                    )
                    return
                name = self.field or solver.problem.field.primary
                x = state.fields[name].detach()
                target_bw = self.target_bandwidth(x, state.progress)
                nyq = min(n / 4.0 for n in x.shape)
            target = max(self._progress_for(target_bw, nyq), self.min_progress)
            step_cap = self.progress + self._rate * self.update_every
            self.progress = max(self.progress, min(target, step_cap, 1.0))
            self.history.append(
                {
                    "step": float(state.step),
                    "global_step": float(state.global_step),
                    "target_bandwidth": float(target_bw),
                    "progress": float(self.progress),
                }
            )
        solver.progress_override = self.progress


# ------------------------------------------------------------------------------------------
# prescribed bandwidth schedules
# ------------------------------------------------------------------------------------------
@register("callback", "bandwidth_schedule")
class BandwidthSchedule(Callback):
    """Prescribe the open bandwidth (cycles per unit normalized coordinate) over each stage.

    ``schedule`` maps the stage fraction ``t ∈ [0, 1]`` to a target bandwidth — a callable or
    breakpoints ``[(t0, B0), (t1, B1), …]`` (linear interpolation) — and the callback sets
    ``solver.progress_override = bandwidth_to_progress(field, B(t))``. Physical frequencies ``f``
    (cycles per unit length) convert as ``B = f · L / 2`` for an axis of length ``L``.

    Example::

        BandwidthSchedule([(0.0, 0.5), (0.5, 4.0), (1.0, 8.0)])   # 0.5 -> 4 -> 8 cycles/unit
    """

    def __init__(
        self,
        schedule: Callable[[float], float] | Sequence[tuple[float, float]],
        module: nn.Module | None = None,
        per_stage: bool = True,
    ) -> None:
        if callable(schedule):
            self._fn = schedule
        else:
            pts = sorted((float(t), float(b)) for t, b in schedule)
            if not pts:
                raise ConfigError("empty bandwidth schedule")
            ts = torch.tensor([t for t, _ in pts], dtype=torch.float64)
            bs = torch.tensor([b for _, b in pts], dtype=torch.float64)
            self._fn = lambda t: float(_interp(torch.tensor(float(t), dtype=torch.float64), ts, bs))
        self.module, self.per_stage = module, bool(per_stage)
        self._band: nn.Module | None = None
        self._steps = 1
        self._total = 1

    @staticmethod
    def progress_for(field: nn.Module, bandwidth: float) -> float:
        return bandwidth_to_progress(field, bandwidth)

    @staticmethod
    def bandwidth_for(field: nn.Module, progress: float) -> float:
        return progress_to_bandwidth(field, progress)

    def on_run_start(self, solver) -> None:
        self._total = max(1, solver.curriculum.total_steps)

    def on_stage_start(self, solver, stage_idx, stage) -> None:
        self._band = _band_module(self.module if self.module is not None else solver.problem.field)
        self._steps = max(1, stage.steps)
        t = 0.0 if self.per_stage else solver.global_step / self._total
        solver.progress_override = bandwidth_to_progress(self._band, self._fn(t))

    def on_step(self, solver, state: StepState) -> None:
        if self.per_stage:
            t = min(1.0, (state.step + 1) / self._steps)
        else:
            t = min(1.0, (state.global_step + 1) / self._total)
        solver.progress_override = bandwidth_to_progress(self._band, self._fn(t))


__all__ = [
    "BandwidthSchedule",
    "OperatorAwareAnnealing",
    "PlateauDetector",
    "ResidualDrivenAnnealing",
    "band_frequencies",
    "bandwidth_to_progress",
    "find_band_module",
    "n_levels",
    "progress_to_bandwidth",
    "stage_noise_std",
]
