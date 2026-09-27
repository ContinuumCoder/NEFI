"""wave_fwi — acoustic full-waveform inversion (FWI) of a 2-D sound-speed map.

Unknown: sound speed ``c(x, z)`` (km/s) on ``[0, L]²`` km, parameterized by a neural field with a
``Bounded(c_min, c_max)`` head started at the known background ``c_background``. Data: pressure
traces ``(n_sources, n_receivers, n_t)`` from Ricker point sources, simulated by the
leapfrog / PML wave solver of :mod:`nefi.physics.wave` (hard physics constraint; gradients by the
discrete adjoint = autodiff through the unrolled scheme, optional block checkpointing).

Acquisition geometries: ``"transmission"`` (cross-well: sources on the left, receivers on the
right), ``"reflection"`` (surface: sources and receivers near ``z = 0``), ``"surround"`` (both on
all four sides, the default). ``min_offset`` mutes near-offset traces (source–receiver distance
below it; :meth:`WaveFWI.offset_mask`): next to a point source the direct wave is dominated by
the near field, which the coarse inversion grid represents differently from the 2×-finer data
grid, so unmuted surround data are fitted with artifacts around the sources and receivers.
Scene classes: ``smooth_anomaly`` (Gaussian velocity anomalies), ``layered`` (dipping layers with
smooth interfaces), ``scatterers`` (small smooth-edged inclusions).

Multiscale: two curriculum stages — a coarse grid fitting **low-pass filtered traces**
(``fit_lo``, cutoff ``f_low``; frequency continuation of Bunks et al. 1995 against cycle skipping),
then the full grid with the full band (``fit``); see :mod:`.losses` for the cycle-skipping
discussion. Inverse-crime guard: data come from a 2× finer grid in float64
(``fidelity_tag="wave-leapfrog-o4-2x-float64"`` vs the inversion operator's
``"wave-leapfrog-o{order}-pml"``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from ...bench.base import DataGenerator, SceneGenerator
from ...domain import Domain
from ...errors import ConfigError
from ...fields import Bounded, GridField, Heads, NeuralField
from ...losses import TV, LossSet
from ...measurement import Measurement
from ...metrics.basic import psnr, relative_error, ssim
from ...physics.wave import WaveOperator
from ...problem import InverseProblem
from ...registry import register
from ...solve.curriculum import Curriculum, Stage
from .._wave_family import line_positions
from ..base import Instance
from .losses import BandLimitedMSE, lowpass

GEOMETRIES = ("transmission", "reflection", "surround")


@dataclass
class WaveFWIConfig:
    """Configuration (defaults = CPU smoke / gallery preset: 24², surround acquisition with
    6 sources and 32 receivers, near offsets < 0.3 km muted, 150 + 250 steps)."""

    # domain / scene
    n: int = 24  # field grid (n × n)
    extent: float = 1.0  # km, domain [0, extent]²
    scene: str = "smooth_anomaly"  # smooth_anomaly | layered | scatterers
    c_background: float = 2.0  # km/s, background and initial model
    c_min: float = 1.5  # km/s, Bounded head lower limit
    c_max: float = 2.5  # km/s, Bounded head upper limit (also the CFL bound)
    anomaly: float = 0.4  # km/s, max |c − c_background| of anomalies / inclusions
    # acquisition
    geometry: str = "surround"  # transmission | reflection | surround
    n_sources: int = 6
    n_receivers: int = 32
    margin: float = 0.08  # km, distance of the arrays from the domain edge
    min_offset: float = 0.3  # km, near-offset mute: traces with |src − rec| < min_offset masked
    f0: float = 4.0  # Hz, Ricker peak frequency
    t_max: float = 0.96  # s, record length
    dt_obs: float = 0.016  # s, trace sampling
    # solver
    order: int = 4  # spatial stencil order (2 | 4)
    absorbing: str = "pml"  # pml | sponge | cerjan
    absorb_width: float = 0.25  # km (6 cells at n = 24)
    absorb_R: float = 1e-3  # nominal PML reflection coefficient
    courant: float = 0.9
    grad_mode: str = "autograd"  # autograd (small) | checkpoint (large runs)
    checkpoint_every: int | None = None  # None -> ceil(sqrt(n_steps))
    source_batch: int | None = None  # simulate sources in chunks (memory)
    # data generation (inverse-crime guard)
    noise_std: float = 0.005  # relative to max |trace|
    supersample: int = 2  # generator grid refinement
    gen_order: int = 4
    # neural field
    hidden: int = 64
    depth: int = 4
    n_octaves: int = 3  # 4 cycles over the domain: smooth updates (FWI resolves ≈ λ/2)
    activation: str = "tanh"
    # losses / curriculum
    tv: float = 1e-3  # isotropic TV weight on c (physical spacing)
    multiscale_frequency: bool = True  # stage 1: coarse grid + low-pass data
    f_low: float = 5.0  # Hz, low-pass cutoff of stage 1
    lowpass_taper: float = 0.3
    coarse_factor: int = 2  # stage-1 grid = n / coarse_factor
    steps: tuple[int, int] = (150, 250)
    lr: float = 2e-2
    lr_decay: float = 0.5
    grid_lr_mult: float = 10.0  # learning-rate multiplier of the "grid" baseline


class WaveFWIScenes(SceneGenerator):
    """Analytic sound-speed models (evaluated at any resolution; km/s)."""

    classes = ("smooth_anomaly", "layered", "scatterers")

    def __init__(self, domain: Domain, cfg: WaveFWIConfig) -> None:
        super().__init__(domain)
        self.cfg = cfg

    def sample(self, rng: np.random.Generator, cls: str | None = None, shape=None):
        cls = self.check_class(cls)
        c = self.cfg
        L = self.domain.size[0]
        xz = self.domain.physical_coords(shape, dtype=torch.float64) / L
        x, z = xz[..., 0], xz[..., 1]
        v = torch.full_like(x, c.c_background)
        if cls == "smooth_anomaly":
            for _ in range(int(rng.integers(1, 3))):
                cx, cz = rng.uniform(0.3, 0.7, size=2)
                s = rng.uniform(0.08, 0.14)
                a = rng.choice([-1.0, 1.0]) * rng.uniform(0.6, 1.0) * c.anomaly
                v = v + a * torch.exp(-((x - cx) ** 2 + (z - cz) ** 2) / (2 * s**2))
        elif cls == "layered":
            n_if = int(rng.integers(2, 4))
            depths = np.sort(rng.uniform(0.2, 0.85, size=n_if))
            dips = rng.uniform(-0.15, 0.15, size=n_if)
            lo, hi = c.c_background - c.anomaly, c.c_background + c.anomaly
            vels = np.sort(rng.uniform(lo, hi, size=n_if + 1))
            v = torch.full_like(x, float(vels[0]))
            for k in range(n_if):
                zk = depths[k] + dips[k] * (x - 0.5)
                v = v + float(vels[k + 1] - vels[k]) * 0.5 * (1 + torch.tanh((z - zk) / 0.015))
        else:  # scatterers
            for _ in range(int(rng.integers(2, 5))):
                cx, cz = rng.uniform(0.25, 0.75, size=2)
                r0 = rng.uniform(0.05, 0.09)
                a = rng.choice([-1.0, 1.0]) * rng.uniform(0.6, 1.0) * c.anomaly
                r = torch.sqrt((x - cx) ** 2 + (z - cz) ** 2)
                v = v + a * 0.5 * (1 + torch.tanh((r0 - r) / 0.01))
        v = v.clamp(c.c_min + 0.02, c.c_max - 0.02)
        return {"c": v.float()}


class TraceDataGenerator(DataGenerator):
    """:class:`~nefi.bench.DataGenerator` with an optional trace mask (near-offset mute).

    Muted traces are zeroed and masked out (``Measurement.mask``), so the misfit never sees them;
    the noise level stays relative to the maximum of the full (unmuted) recording.
    """

    def __init__(self, operator, trace_mask: torch.Tensor | None = None, **kw) -> None:
        super().__init__(operator, **kw)
        self.trace_mask = trace_mask

    def generate(self, gt_fields, rng, noise_std=None, target_shape=None) -> Measurement:
        meas = super().generate(gt_fields, rng, noise_std, target_shape)
        if self.trace_mask is None:
            return meas
        m = self.trace_mask.to(meas.data).expand_as(meas.data).contiguous()
        meta = {**meas.meta, "muted_traces": int((self.trace_mask == 0).sum())}
        return Measurement(meas.data * m, m, meas.noise_std, meta)


@register("instance", "wave_fwi")
class WaveFWI(Instance):
    """Acoustic FWI instance (see module docstring)."""

    name = "wave_fwi"
    Config = WaveFWIConfig
    description = "2-D acoustic full-waveform inversion of sound speed (leapfrog + PML)"

    # ---- geometry -------------------------------------------------------------------------
    def domain(self) -> Domain:
        L = self.cfg.extent
        return Domain((self.cfg.n, self.cfg.n), ((0.0, L), (0.0, L)), axes=("x", "z"))

    def acquisition(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(sources, receivers)`` physical positions (km)."""
        c = self.cfg
        L, m = c.extent, c.margin
        if c.geometry not in GEOMETRIES:
            raise ConfigError(f"geometry must be one of {GEOMETRIES}, got {c.geometry!r}")
        if c.geometry == "transmission":
            src = line_positions(c.n_sources, 0, m, m, L - m)
            rec = line_positions(c.n_receivers, 0, L - m, m, L - m)
        elif c.geometry == "reflection":
            src = line_positions(c.n_sources, 1, m, m, L - m)
            rec = line_positions(c.n_receivers, 1, m, m, L - m)
        else:
            src = _perimeter(c.n_sources, L, m, phase=0.5)
            rec = _perimeter(c.n_receivers, L, m, phase=0.0)
        return src, rec

    @property
    def n_t(self) -> int:
        return int(round(self.cfg.t_max / self.cfg.dt_obs)) + 1

    def offset_mask(self) -> torch.Tensor | None:
        """Near-offset mute: ``(n_sources, n_receivers, 1)`` with 0 where the source–receiver
        distance is below ``min_offset`` (``None`` when nothing is muted).

        Close to a point source the direct wave is dominated by the near field, which a coarse
        grid represents poorly (the source/receiver stencils differ between the inversion grid and
        the 2× finer data grid): with the ``surround`` geometry the near-offset traces carry most
        of the energy and most of the modeling error, and an unmuted misfit is explained by
        artifacts around the sources and receivers. Muting them is standard FWI practice.
        """
        c = self.cfg
        if c.min_offset <= 0:
            return None
        src, rec = self.acquisition()
        keep = (torch.cdist(src, rec) >= c.min_offset).float()
        if not bool(keep.any()):
            raise ConfigError(
                f"min_offset = {c.min_offset} km mutes every trace of the {c.geometry!r} geometry"
            )
        return keep[..., None]

    def operator(self, domain: Domain | None = None, order: int | None = None) -> WaveOperator:
        c = self.cfg
        src, rec = self.acquisition()
        return WaveOperator(
            domain or self.domain(),
            src,
            rec,
            n_t=self.n_t,
            dt_obs=c.dt_obs,
            f0=c.f0,
            c_max=c.c_max,
            field="c",
            order=c.order if order is None else order,
            absorbing=c.absorbing,
            absorb_width=c.absorb_width,
            absorb_R=c.absorb_R,
            courant=c.courant,
            grad_mode=c.grad_mode,
            checkpoint_every=c.checkpoint_every,
            source_batch=c.source_batch,
        )

    # ---- Instance hooks ---------------------------------------------------------------------
    def scene_generator(self) -> SceneGenerator:
        return WaveFWIScenes(self.domain(), self.cfg)

    def data_generator(self) -> DataGenerator:
        c = self.cfg
        fine = self.domain().refine(c.supersample) if c.supersample > 1 else self.domain()
        op = self.operator(fine, order=c.gen_order)
        op.grad_mode = "autograd"
        return TraceDataGenerator(
            op,
            trace_mask=self.offset_mask(),
            noise_std=c.noise_std,
            relative=True,
            supersample=c.supersample,
            fidelity_tag=f"wave-leapfrog-o{c.gen_order}-{c.supersample}x-float64",
        )

    def build_problem_measurement_shape(self, field_shape):
        src, rec = self.acquisition()
        return (src.shape[0], rec.shape[0], self.n_t)

    def heads(self) -> Heads:
        c = self.cfg
        return Heads({"c": Bounded(c.c_min, c.c_max, init_value=c.c_background)})

    def field(self) -> NeuralField:
        c = self.cfg
        return NeuralField(
            2,
            self.heads(),
            hidden=c.hidden,
            depth=c.depth,
            skip_at=c.depth // 2,
            activation=c.activation,
            n_octaves=c.n_octaves,
        )

    def losses(self) -> LossSet:
        c = self.cfg
        return LossSet(
            {
                "fit": BandLimitedMSE(None, c.dt_obs),
                "fit_lo": BandLimitedMSE(c.f_low, c.dt_obs, taper=c.lowpass_taper),
                "tv": TV("c", isotropic=True),
            },
            weights={"fit": 1.0, "fit_lo": 0.0, "tv": c.tv},
        )

    def build_problem(self, measurement: Measurement) -> InverseProblem:
        return InverseProblem(
            self.domain(),
            self.field(),
            self.operator(),
            self.losses(),
            measurement,
            curriculum=self.default_curriculum(),
            name="wave_fwi",
        )

    def default_curriculum(self, lr_mult: float = 1.0) -> Curriculum:
        c = self.cfg
        n = c.n
        coarse = (max(8, n // c.coarse_factor),) * 2
        lr = c.lr * lr_mult
        w_lo = {"fit": 0.0, "fit_lo": 1.0, "tv": c.tv}
        w_full = {"fit": 1.0, "fit_lo": 0.0, "tv": c.tv}
        if not c.multiscale_frequency:
            w_lo = w_full
        return Curriculum(
            [
                Stage("low", coarse, int(c.steps[0]), lr, loss_weights=w_lo),
                Stage("full", (n, n), int(c.steps[1]), lr * c.lr_decay, loss_weights=w_full),
            ]
        )

    def metrics(self) -> dict[str, Callable]:
        c0 = self.cfg.c_background

        def anomaly_error(pred, gt) -> float:
            """``‖c − c_gt‖ / ‖c_gt − c_background‖`` (1.0 for the initial uniform model)."""
            p, g = torch.as_tensor(pred).double(), torch.as_tensor(gt).double()
            return float(torch.linalg.vector_norm(p - g) / torch.linalg.vector_norm(g - c0))

        return {
            "psnr": psnr,
            "ssim": ssim,
            "relative_error": relative_error,
            "anomaly_error": anomaly_error,
        }

    def initial_model(self) -> torch.Tensor:
        """The uniform starting model ``c_background`` on the native grid."""
        return torch.full((self.cfg.n, self.cfg.n), float(self.cfg.c_background))

    def baselines(self):
        def grid(measurement: Measurement):
            field = GridField((self.cfg.n, self.cfg.n), self.heads())
            cur = self.default_curriculum(lr_mult=self.cfg.grid_lr_mult)
            prob = InverseProblem(
                self.domain(),
                field,
                self.operator(),
                self.losses(),
                measurement,
                curriculum=cur,
                name="wave_fwi-grid",
                meta={"baseline": "grid"},
            )
            return prob, cur

        return {"grid": grid}


def _perimeter(n: int, L: float, m: float, phase: float = 0.0) -> torch.Tensor:
    """``n`` points evenly spaced along the square ``[m, L−m]²`` perimeter."""
    side = L - 2 * m
    s = (torch.arange(n, dtype=torch.float64) + phase) / n * 4 * side
    pts = []
    for si in s.tolist():
        k, t = int(si // side) % 4, si % side
        pts.append([(m + t, m), (L - m, m + t), (L - m - t, L - m), (m, L - m - t)][k])
    return torch.tensor(pts, dtype=torch.float64)


PRESETS: dict[str, dict] = {
    "smoke": {},  # the dataclass defaults
    "full": {
        "n": 96,
        "n_sources": 8,
        "n_receivers": 48,
        "geometry": "surround",
        "min_offset": 0.0,
        "tv": 1e-4,
        "f0": 8.0,
        "t_max": 1.2,
        "dt_obs": 0.004,
        "absorb_width": 0.12,
        "grad_mode": "checkpoint",
        "hidden": 256,
        "depth": 6,
        "n_octaves": 6,
        "f_low": 6.0,
        "steps": (1500, 3500),
        "lr": 1e-3,
    },
}


def make_problem(
    seed: int = 0, **cfg
) -> tuple[InverseProblem, dict[str, torch.Tensor], Measurement]:
    """Convenience: ``(problem, gt, measurement)`` for docs and tests."""
    inst = WaveFWI(**cfg)
    gt, meas = inst.make_measurement(seed)
    return inst.build_problem(meas), gt, meas


def run(cfg: dict | WaveFWIConfig | None = None, seed: int = 0, **kw):
    """End-to-end demo (DESIGN §3.10): generate → invert → evaluate → ``(result, metrics)``."""
    out = WaveFWI(cfg).run(seed=seed, **kw)
    return out.result, out.metrics


__all__ = [
    "GEOMETRIES",
    "PRESETS",
    "BandLimitedMSE",
    "TraceDataGenerator",
    "WaveFWI",
    "WaveFWIConfig",
    "WaveFWIScenes",
    "lowpass",
    "make_problem",
    "run",
]
