"""darcy_flow — permeability recovery from steady pressure data (subsurface flow).

The Darcy instance of the elliptic physics family. Several injector/producer well configurations
drive steady single-phase flow in a closed (no-flow) reservoir; sparse pressure gauges record the
pressure field. The task is to recover the log-permeability ``Y = log k`` without labels — the
classic hydraulic-tomography / history-matching inverse problem (Yeh & Liu 2000; Oliver, Reynolds
& Liu, *Inverse Theory for Petroleum Reservoir Characterization*, 2008).

Pipeline::

    scene (DarcyScenes: log-normal GRF or channels) ──DarcyOperator on a 2× finer grid, float64,
        + noise──► p_obs (n_configs, H, W), mask = sensors (minus well cells)
    coords ─► annealed PE ─► MLP ─► Y = Bounded(Y_min, Y_max) ─► k = exp(Y) ─► DarcyOperator
        (harmonic-mean FV, pure-Neumann PCG, implicit-function adjoint) ─► masked MSE + TV
    two-stage curriculum H/2 → H (every stage predicts the native gauge pressures)

Quick start::

    from nefi.instances.darcy_flow import DarcyFlow
    out = DarcyFlow(n=16, n_configs=4, steps=(50, 100)).run(seed=0, device="cpu")
    print(out.metrics)          # psnr, ssim, mse, relative_error, high_k_iou (on log k)

See ``docs/instances/darcy_flow.md``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from ...bench.base import SceneGenerator
from ...domain import Domain
from ...errors import ConfigError
from ...fields import Bounded, GridField, Heads, NeuralField
from ...losses import MSE, TV, LossSet, RelativeMSE
from ...measurement import Measurement
from ...metrics.basic import mse, psnr, relative_error, ssim
from ...problem import InverseProblem
from ...registry import register
from ...solve.curriculum import Curriculum, Stage
from .._elliptic_common import SupersampledObservationGenerator, footprint_resample
from ..base import Instance
from .operator import DarcyOperator, sensor_mask, well_configurations, wells_rhs
from .scenes import DarcyScenes, high_k_iou

__all__ = [
    "DarcyConfig",
    "DarcyFlow",
    "DarcyOperator",
    "DarcyScenes",
    "high_k_iou",
    "make_problem",
    "run",
    "sensor_mask",
    "well_configurations",
    "wells_rhs",
]


@dataclass
class DarcyConfig:
    """Darcy-flow instance configuration (every number used by the pipeline)."""

    # geometry & physics
    n: int = 32
    length: float = 1.0
    viscosity: float = 1.0
    bc: str = "neumann"
    n_configs: int = 6
    well_radius: float = 0.3
    well_rate: float = 1.0
    well_angle: float = 0.3
    well_spread: str = "linear"
    face_mode: str = "harmonic"
    # sensors
    n_sensors: int = 8
    sensor_jitter: float = 0.0
    sensor_seed: int = 12345
    well_exclusion: float = 1.5
    # scenes (log k)
    scene: str = "smooth"
    log_k_mean: float = 0.0
    log_k_std: float = 0.8
    corr_length: float = 0.15
    n_modes: int = 128
    channel_contrast: float = 2.5
    channel_bg_std: float = 0.4
    n_channels: tuple[int, int] = (1, 3)
    channel_width: tuple[float, float] = (0.06, 0.09)
    channel_amplitude: tuple[float, float] = (0.05, 0.12)
    channel_wavelength: tuple[float, float] = (0.4, 0.8)
    # field bounds on log k
    log_k_min: float = -3.0
    log_k_max: float = 4.0
    clip_margin: float = 0.2  # scenes are clamped to [log_k_min + m, log_k_max - m]
    # data (inverse-crime guard: same physics on a supersample× finer grid in float64)
    noise_std: float = 2e-3
    supersample: int = 2
    # PDE solver
    tol: float = 1e-6
    max_iter: int | None = None
    precond: str = "jacobi"
    grad_mode: str = "ift"
    warm_start: bool = True
    check_every: int = 1  # CG convergence-check period (host syncs); use 8-16 on CUDA
    # field
    hidden: int = 128
    depth: int = 4
    skip_at: int | None = 2
    n_octaves: int = 4
    activation: str = "relu"
    out_init_scale: float = 0.1
    # curriculum
    multiscale: bool = True
    min_coarse: int = 12
    steps: tuple[int, int] = (500, 1500)
    lr: float = 5e-3
    lr_decay: float = 0.5
    lr_min_ratio: float = 0.05
    anneal_fraction: float = 0.5
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    # losses
    data_loss: str = "mse"
    tv: float = 1e-5
    tv_eps: float = 1e-3
    # metrics
    iou_n_std: float = 1.0
    # grid baseline
    grid_lr: float = 5e-2
    grid_tv: float | None = None


@register("instance", "darcy_flow")
class DarcyFlow(Instance):
    """Steady Darcy flow: log-permeability from sparse pressure gauges, several well patterns."""

    name = "darcy_flow"
    Config = DarcyConfig
    description = "Darcy flow: log-permeability from sparse steady pressures (well configurations)"

    # --- geometry / physics -------------------------------------------------------------
    def domain(self) -> Domain:
        c = self.cfg
        return Domain((c.n, c.n), ((0.0, c.length), (0.0, c.length)), ("x", "y"))

    def wells(self) -> list[list[tuple[float, float, float]]]:
        c = self.cfg
        return well_configurations(
            self.domain(), c.n_configs, c.well_radius, c.well_rate, c.well_angle
        )

    def sensors(self) -> torch.Tensor:
        """``(n_configs, n, n)`` native sensor mask (deterministic unless ``sensor_jitter > 0``)."""
        c = self.cfg
        rng = np.random.default_rng(c.sensor_seed) if c.sensor_jitter > 0 else None
        return sensor_mask(
            self.domain(), self.wells(), c.n_sensors, c.well_exclusion, c.sensor_jitter, rng
        )

    def observation_weights(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Sensor weights at grid ``shape`` (exact native-cell footprints)."""
        return footprint_resample(self.sensors(), shape)

    def operator(self, domain: Domain | None = None, **overrides) -> DarcyOperator:
        c = self.cfg
        kw = {
            "sensors": self.sensors(),
            "obs_shape": (c.n, c.n),
            "viscosity": c.viscosity,
            "bc": c.bc,
            "well_spread": c.well_spread,
            "face_mode": c.face_mode,
            "grad_mode": c.grad_mode,
            "tol": c.tol,
            "max_iter": c.max_iter,
            "precond": c.precond,
            "check_every": c.check_every,
            "warm_start": c.warm_start,
        }
        kw.update(overrides)
        return DarcyOperator(domain or self.domain(), self.wells(), **kw)

    def scene_generator(self) -> SceneGenerator:
        c = self.cfg
        return DarcyScenes(
            self.domain(),
            mean=c.log_k_mean,
            std=c.log_k_std,
            corr_length=c.corr_length,
            n_modes=c.n_modes,
            channel_contrast=c.channel_contrast,
            channel_bg_std=c.channel_bg_std,
            n_channels=c.n_channels,
            channel_width=c.channel_width,
            channel_amplitude=c.channel_amplitude,
            channel_wavelength=c.channel_wavelength,
            clip=(c.log_k_min + c.clip_margin, c.log_k_max - c.clip_margin),
        )

    def data_generator(self) -> SupersampledObservationGenerator:
        c = self.cfg
        fine = self.domain().refine(c.supersample)
        op = self.operator(fine, tol=1e-10, warm_start=False, grad_mode="none")
        return SupersampledObservationGenerator(
            op,
            self.observation_weights,
            self.domain().shape,
            noise_std=c.noise_std,
            supersample=c.supersample,
            fidelity_tag=f"elliptic-fv-{c.supersample}x-float64",
        )

    def build_problem_measurement_shape(self, field_shape):
        return (self.cfg.n_configs, *tuple(field_shape))

    def make_measurement(self, seed=0, scene_class=None, gt=None):
        gt_out, meas = super().make_measurement(seed, scene_class, gt)
        meas.meta.pop("clean", None)
        return gt_out, meas

    # --- inversion ----------------------------------------------------------------------
    def heads(self) -> Heads:
        c = self.cfg
        return Heads({"log_k": Bounded(c.log_k_min, c.log_k_max, init_value=c.log_k_mean)})

    def field(self) -> NeuralField:
        c = self.cfg
        return NeuralField(
            2,
            self.heads(),
            hidden=c.hidden,
            depth=c.depth,
            skip_at=c.skip_at,
            activation=c.activation,
            n_octaves=c.n_octaves,
            out_init_scale=c.out_init_scale,
        )

    def losses(self, tv: float | None = None) -> LossSet:
        c = self.cfg
        data = {"mse": MSE, "relative_mse": RelativeMSE}.get(c.data_loss)
        if data is None:
            raise ConfigError(f"unknown data_loss {c.data_loss!r}; use 'mse' or 'relative_mse'")
        return LossSet(
            {"data": data(), "tv": TV("log_k", isotropic=True, eps=c.tv_eps)},
            weights={"data": 1.0, "tv": c.tv if tv is None else tv},
        )

    def _curriculum(self, lr: float) -> Curriculum:
        c = self.cfg
        steps = [int(s) for s in c.steps]
        kw = {"lr_min_ratio": c.lr_min_ratio, "anneal_fraction": c.anneal_fraction}
        if c.multiscale and len(steps) > 1 and c.n // 2 ** (len(steps) - 1) >= c.min_coarse:
            cur = Curriculum.multiscale(
                (c.n, c.n), n_stages=len(steps), steps=steps, lr=lr, lr_decay=c.lr_decay, **kw
            )
        else:  # all stages at the native resolution: LR decay + annealing restart per stage
            cur = Curriculum(
                [
                    Stage(f"stage{i + 1}", (c.n, c.n), s, lr * c.lr_decay**i, **kw)
                    for i, s in enumerate(steps)
                ]
            )
        cur.optim.weight_decay = c.weight_decay
        cur.optim.grad_clip = c.grad_clip
        return cur

    def default_curriculum(self) -> Curriculum:
        return self._curriculum(self.cfg.lr)

    def _problem(self, field, measurement: Measurement, tv=None, name="darcy_flow"):
        # the operator predicts the native observation grid at every curriculum resolution, so
        # the measurement is used as is (no observation downsampling)
        return InverseProblem(
            self.domain(), field, self.operator(), self.losses(tv), measurement, name=name
        )

    def build_problem(self, measurement: Measurement) -> InverseProblem:
        problem = self._problem(self.field(), measurement)
        problem.curriculum = self.default_curriculum()
        return problem

    def metrics(self) -> dict[str, Callable]:
        return {
            "psnr": psnr,
            "ssim": ssim,
            "mse": mse,
            "relative_error": relative_error,
            "high_k_iou": functools.partial(high_k_iou, n_std=self.cfg.iou_n_std),
        }

    def baselines(self):
        def grid(measurement: Measurement):
            c = self.cfg
            field = GridField((c.n, c.n), self.heads())
            problem = self._problem(
                field, measurement, c.grid_tv if c.grid_tv is not None else c.tv, "darcy-grid"
            )
            cur = self._curriculum(c.grid_lr)
            problem.curriculum = cur
            return problem, cur

        return {"grid": grid}


def make_problem(seed: int = 0, scene_class: str | None = None, **cfg):
    """Convenience: ``(problem, gt, measurement)`` for docs, tests and notebooks."""
    inst = DarcyFlow(**cfg)
    gt, meas = inst.make_measurement(seed, scene_class)
    return inst.build_problem(meas), gt, meas


def run(cfg: DarcyConfig | dict | None = None, seed: int = 0, **kw):
    """End-to-end demo: generate → invert → evaluate. Returns ``(Result, metrics)``."""
    out = DarcyFlow(cfg).run(seed=seed, **kw)
    return out.result, out.metrics
