"""sparse_view_ct — parallel-beam CT from a few views with a differentiable Radon transform.

Recover an attenuation map ``μ ∈ [0, 1]`` on the unit square from ``n_views`` (e.g. 20) noisy
parallel-beam projections over 180°. The forward model is :class:`RadonOperator` (rotation by
``grid_sample`` + summation, physical line integrals) evaluated at every curriculum resolution
with the detector count scaled accordingly. Data are simulated with the same Radon physics on a 2×
finer image grid with 2× more detector bins per pixel, in float64, then area-averaged onto the
detector (``fidelity_tag="radon-2x-float64"``, inverse-crime guard). The prior is a coordinate
neural field with a ``Bounded(0, 1)`` head (NeFTY Eq. 6) and isotropic TV (NeFTY Eq. 22), solved
with the two-stage multiscale curriculum (NeTMY Tab. 6).

Baselines: ``grid`` (free pixels + TV), ``fbp`` (filtered back-projection, closed form, reported
through the same metrics) and ``deep_decoder``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ...baselines import baseline_problem
from ...bench.base import DataGenerator, SceneGenerator
from ...domain import Domain
from ...errors import ConfigError
from ...fields import Bounded, GridField, Heads, Identity, NeuralField, Softplus
from ...losses import L1, MSE, TV, LossSet
from ...measurement import Measurement
from ...metrics.basic import mse, psnr, ssim
from ...problem import InverseProblem
from ...registry import register
from ...solve.curriculum import Curriculum
from ..base import Instance
from .radon import FILTERS, RadonOperator, backproject, fbp, ramp_filter, uniform_angles
from .scenes import SHEPP_LOGAN, CTScenes


@dataclass
class SparseViewCTConfig:
    """Configuration of the :class:`SparseViewCT` instance (all numbers are config fields).

    Physics: ``n × n`` image on ``[0, extent]²``, ``n_views`` equispaced views over
    ``angle_range`` degrees, ``det_per_pixel`` detector bins per pixel and ``samples_per_pixel``
    ray samples; relative Gaussian sinogram noise ``noise_std``; data simulated on a
    ``supersample``× grid with ``data_det_per_pixel`` bins per fine pixel.
    Prior / solver: neural field (``hidden``, ``depth``, ``skip_at``, ``n_octaves``,
    ``activation``), ``head`` ∈ {bounded, softplus}, isotropic ``tv`` (``tv_eps``), ``l1``,
    two-stage curriculum (``steps``, ``lr``, ``lr_decay``, ``anneal_fraction``).
    Baselines: ``fbp_filter`` / ``fbp_clip``, ``grid_lr``, ``dd_*``.
    """

    n: int = 64
    extent: float = 1.0
    n_views: int = 20
    angle_range: float = 180.0
    det_per_pixel: float = 1.0
    samples_per_pixel: float = 1.0
    scene: str = "shepp"
    noise_std: float = 0.01
    supersample: int = 2
    data_det_per_pixel: float = 2.0
    head: str = "bounded"
    init_value: float = 0.1
    tv: float = 1e-5
    tv_eps: float = 1e-3
    l1: float = 0.0
    hidden: int = 128
    depth: int = 4
    skip_at: int = 2
    n_octaves: int = 7
    activation: str = "tanh"
    steps: tuple[int, int] = (400, 800)
    lr: float = 1e-2
    lr_decay: float = 0.5
    anneal_fraction: float = 1.0
    fbp_filter: str = "ramp"
    fbp_clip: bool = True
    grid_lr: float = 5e-2
    dd_lr: float = 5e-3
    dd_width: int = 64
    dd_stages: int = 5


@register("instance", "sparse_view_ct")
class SparseViewCT(Instance):
    """Sparse-view parallel-beam CT instance (see module docstring)."""

    name = "sparse_view_ct"
    Config = SparseViewCTConfig
    description = "Sparse-view parallel-beam CT with a differentiable Radon transform"

    def domain(self) -> Domain:
        c = self.cfg
        return Domain((c.n, c.n), ((0.0, c.extent), (0.0, c.extent)), axes=("x", "y"))

    def angles(self) -> torch.Tensor:
        c = self.cfg
        return uniform_angles(c.n_views, math.radians(c.angle_range))

    def operator(self, domain: Domain | None = None) -> RadonOperator:
        c = self.cfg
        return RadonOperator(
            domain or self.domain(),
            field="mu",
            angles=self.angles(),
            angle_range=math.radians(c.angle_range),
            det_per_pixel=c.det_per_pixel,
            samples_per_pixel=c.samples_per_pixel,
        )

    def scene_generator(self) -> SceneGenerator:
        return CTScenes(self.domain())

    def data_generator(self) -> DataGenerator:
        c = self.cfg
        fine = RadonOperator(
            self.domain().refine(c.supersample),
            field="mu",
            angles=self.angles(),
            angle_range=math.radians(c.angle_range),
            det_per_pixel=c.data_det_per_pixel,
            samples_per_pixel=c.samples_per_pixel,
        )
        return DataGenerator(
            fine,
            noise_std=c.noise_std,
            relative=True,
            supersample=c.supersample,
            fidelity_tag="radon-2x-float64",
        )

    def build_problem_measurement_shape(self, field_shape):
        return self.operator().output_shape(field_shape)

    def heads(self) -> Heads:
        c = self.cfg
        if c.head == "bounded":
            return Heads({"mu": Bounded(0.0, 1.0, init_value=c.init_value)})
        if c.head == "softplus":
            return Heads({"mu": Softplus(init_value=c.init_value)})
        raise ConfigError(f"head must be 'bounded' or 'softplus', got {c.head!r}")

    def field(self) -> NeuralField:
        c = self.cfg
        return NeuralField(
            2,
            self.heads(),
            hidden=c.hidden,
            depth=c.depth,
            skip_at=c.skip_at,
            n_octaves=c.n_octaves,
            activation=c.activation,
        )

    def losses(self) -> LossSet:
        c = self.cfg
        return LossSet(
            {"data": MSE(), "tv": TV("mu", isotropic=True, eps=c.tv_eps), "l1": L1("mu")},
            weights={"data": 1.0, "tv": c.tv, "l1": c.l1},
        )

    def default_curriculum(self) -> Curriculum:
        c = self.cfg
        return Curriculum.multiscale(
            (c.n, c.n),
            n_stages=2,
            steps=tuple(c.steps),
            lr=c.lr,
            lr_decay=c.lr_decay,
            anneal_fraction=c.anneal_fraction,
        )

    def build_problem(self, measurement: Measurement) -> InverseProblem:
        return InverseProblem(
            self.domain(),
            self.field(),
            self.operator(),
            self.losses(),
            measurement,
            curriculum=self.default_curriculum(),
            name="sparse_view_ct",
            meta={"scene": self.cfg.scene, "n_views": self.cfg.n_views},
        )

    def metrics(self) -> dict[str, Callable]:
        return {"psnr": psnr, "ssim": ssim, "mse": mse}

    # ---- classical reference & baselines ------------------------------------------------
    def fbp(self, measurement: Measurement, filter: str | None = None) -> torch.Tensor:
        """Filtered back-projection of ``measurement`` on the native grid."""
        c = self.cfg
        rec = self.operator().fbp(measurement.data, filter or c.fbp_filter)
        if c.fbp_clip:
            hi = 1.0 if c.head == "bounded" else None
            rec = rec.clamp(min=0.0, max=hi)
        return rec

    def baselines(self) -> dict[str, Callable[[Measurement], tuple[InverseProblem, Curriculum]]]:
        c = self.cfg

        def grid(m: Measurement):
            return baseline_problem(self.build_problem(m), "grid", lr=c.grid_lr)

        def deep_decoder(m: Measurement):
            return baseline_problem(
                self.build_problem(m),
                "deep_decoder",
                lr=c.dd_lr,
                width=c.dd_width,
                n_stages=c.dd_stages,
            )

        def fbp_baseline(m: Measurement):
            def reconstruct(obs: Measurement, dom: Domain) -> torch.Tensor:
                return self.fbp(obs)

            shape = (c.n, c.n)
            x0 = reconstruct(m, self.domain())
            field = GridField(shape, Heads({"mu": Identity()}), init=x0[..., None])
            cur = Curriculum.single(shape, steps=1, lr=0.0, anneal=False)
            prob = InverseProblem(
                self.domain(),
                field,
                self.operator(),
                self.losses(),
                m,
                curriculum=cur,
                name="sparse_view_ct-fbp",
                meta={"solver": "direct", "reconstruct": reconstruct, "baseline": "fbp"},
            )
            return prob, cur

        return {"grid": grid, "fbp": fbp_baseline, "deep_decoder": deep_decoder}


def make_problem(
    n: int = 64, seed: int = 0, scene: str | None = None, **cfg
) -> tuple[InverseProblem, dict[str, torch.Tensor], Measurement]:
    """Convenience: ``(problem, gt, measurement)`` for docs and tests."""
    inst = SparseViewCT(n=n, **({"scene": scene} if scene else {}), **cfg)
    gt, meas = inst.make_measurement(seed)
    return inst.build_problem(meas), gt, meas


def run(cfg: SparseViewCTConfig | dict | None = None, seed: int = 0, **kw):
    """End-to-end demo (generate → invert → evaluate); returns a ``RunOutput``."""
    return SparseViewCT(cfg).run(seed=seed, **kw)


METRICS: dict[str, Callable] = {"psnr": psnr, "ssim": ssim, "mse": mse}

__all__ = [
    "FILTERS",
    "METRICS",
    "SHEPP_LOGAN",
    "CTScenes",
    "RadonOperator",
    "SparseViewCT",
    "SparseViewCTConfig",
    "backproject",
    "fbp",
    "make_problem",
    "ramp_filter",
    "run",
    "uniform_angles",
]
