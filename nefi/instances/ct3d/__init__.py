"""ct3d — 3-D parallel-beam sparse-view CT (rotation about ``z``) with a differentiable Radon stack.

Recover a volumetric attenuation map ``μ(x, y, z) ∈ [0, 1]`` from ``n_views`` noisy parallel-beam
projections of every axial slice (``n_views`` equispaced angles over ``angle_range`` degrees; 180°
= full scan, less = limited-angle tomography, preset ``"limited_angle"``). The forward model is
:class:`~nefi.instances.ct3d.radon3d.Radon3DOperator`: the 2-D rotation-based Radon transform of
:mod:`~nefi.instances.sparse_view_ct` applied to all slices and all views in one batched
``grid_sample`` call; the measurement is the sinogram stack ``(n_views, n_det, n_z)``.

* **field** — coordinate MLP on annealed 3-D Fourier features with a ``Bounded(0, 1)`` head
  (NeFTY Eq. 6);
* **losses** — sinogram MSE + isotropic 3-D TV (NeFTY Eq. 22) + optional ℓ1;
* **curriculum** — two-stage multiscale (NeTMY Tab. 6): half resolution in every axis (coarse
  detector bins and slices are area averages of the native ones, i.e. exact coarse data), then
  native;
* **data** — :class:`CT3DScenes` (3-D Shepp–Logan-like ellipsoids, blobs, piecewise-constant
  inserts) projected on a 2× finer grid (2× slices, 2 detector bins per fine pixel, i.e. 4 sub-rays
  per native bin) in float64 and area-averaged (``fidelity_tag="radon3d-2x-float64"``, inverse-
  crime guard);
* **metrics** — PSNR, slice-averaged SSIM, MSE and the feature IoU of ``{μ > iou_tau}``;
* **baselines** — ``grid`` (free voxels + the same objective), ``fbp3d`` (slice-wise filtered
  back-projection, closed form through the ``DirectSolver``) and ``deep_decoder`` (3-D).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

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
from .._volumetric import PresetInstance, iou_above
from .radon3d import FILTERS, Radon3DOperator, backproject3d, fbp3d, uniform_angles
from .scenes import SHEPP_LOGAN_3D, CT3DScenes


@dataclass
class CT3DConfig:
    """Configuration of the :class:`CT3D` instance (every number is a field).

    Physics: ``n × n × n_z`` voxels on ``[0, extent]² × [0, height]``; ``n_views`` equispaced
    views over ``angle_range`` degrees about ``z``; ``det_per_pixel`` detector bins and
    ``samples_per_pixel`` ray samples per lateral pixel; ``view_batch`` chunks the views (memory);
    relative Gaussian sinogram noise ``noise_std``; data simulated on a ``supersample``× grid with
    ``data_det_per_pixel`` bins per fine pixel. Scenes: ``scene`` ∈ {ellipsoids, blobs,
    piecewise}, ``render_factor`` sub-samples per voxel and axis.
    Prior / solver: neural field (``hidden``, ``depth``, ``skip_at``, ``n_octaves``,
    ``activation``), ``head`` ∈ {bounded, softplus}, isotropic ``tv`` (``tv_eps``), ``l1``,
    two-stage curriculum (``steps``, ``lr``, ``lr_decay``, ``anneal_fraction``, ``min_coarse``).
    Metrics: ``iou_tau``. Baselines: ``fbp_filter`` / ``fbp_clip``, ``grid_lr``, ``dd_*``.
    """

    n: int = 64
    n_z: int = 32
    extent: float = 1.0
    height: float = 1.0
    n_views: int = 30
    angle_range: float = 180.0
    det_per_pixel: float = 1.0
    samples_per_pixel: float = 1.0
    view_batch: int | None = None
    scene: str = "ellipsoids"
    render_factor: int = 2
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
    n_octaves: int = 6
    activation: str = "tanh"
    steps: tuple[int, int] = (600, 1200)
    lr: float = 1e-2
    lr_decay: float = 0.5
    anneal_fraction: float = 0.3
    min_coarse: int = 8
    iou_tau: float = 0.25
    fbp_filter: str = "ramp"
    fbp_clip: bool = True
    grid_lr: float = 5e-2
    dd_lr: float = 5e-3
    dd_width: int = 64
    dd_stages: int = 4


#: Named presets (partial configs applied before user overrides).
PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    # CPU smoke run: 32×32×16 volume, 12 views, 16×16×8 → 32×32×16 curriculum (≈ 6 s, one core).
    "smoke": {
        "n": 32,
        "n_z": 16,
        "n_views": 12,
        "hidden": 64,
        "depth": 3,
        "skip_at": 2,
        "n_octaves": 5,
        "steps": (150, 250),
        "lr": 3e-2,
        "tv": 2e-5,
        "dd_width": 32,
        "dd_stages": 3,
    },
    # limited-angle tomography (the "missing wedge" class): 120° coverage
    "limited_angle": {"angle_range": 120.0},
}


@register("instance", "ct3d")
class CT3D(PresetInstance):
    """3-D sparse-view parallel-beam CT instance (see module docstring).

    Args:
        cfg: :class:`CT3DConfig`, a dict, or ``None``.
        preset: optional preset from :data:`PRESETS` (``"smoke"``, ``"limited_angle"``).
        **overrides: individual config fields.
    """

    name = "ct3d"
    Config = CT3DConfig
    PRESETS = PRESETS
    description = "3-D sparse-view parallel-beam CT (slice-stacked differentiable Radon)"
    #: measurement display: the central sinogram of the (views, detector, slice) stack
    viz_hints = {
        "layout": "sinogram",
        "row": "view",
        "col": "detector",
        "axis_values": "angles",
        "measurement_image_label": "sinogram of the most structured slice",
    }

    def _validate(self) -> None:
        c = self.cfg
        if c.n < 4 or c.n_z < 1 or c.n_views < 1:
            raise ConfigError("need n >= 4, n_z >= 1 and n_views >= 1")
        if not 0.0 < c.angle_range <= 360.0:
            raise ConfigError("angle_range must be in (0, 360] degrees")
        if c.fbp_filter not in FILTERS:
            raise ConfigError(f"fbp_filter must be one of {FILTERS}")

    # ---- physics ------------------------------------------------------------------------
    def domain(self) -> Domain:
        c = self.cfg
        return Domain(
            (c.n, c.n, c.n_z),
            ((0.0, c.extent), (0.0, c.extent), (0.0, c.height)),
            axes=("x", "y", "z"),
        )

    def angles(self) -> torch.Tensor:
        c = self.cfg
        return uniform_angles(c.n_views, math.radians(c.angle_range))

    def operator(self, domain: Domain | None = None, det_per_pixel: float | None = None):
        c = self.cfg
        return Radon3DOperator(
            domain or self.domain(),
            field="mu",
            angles=self.angles(),
            angle_range=math.radians(c.angle_range),
            det_per_pixel=c.det_per_pixel if det_per_pixel is None else det_per_pixel,
            samples_per_pixel=c.samples_per_pixel,
            view_batch=c.view_batch,
        )

    def scene_generator(self) -> SceneGenerator:
        return CT3DScenes(self.domain(), render_factor=self.cfg.render_factor)

    def data_generator(self) -> DataGenerator:
        c = self.cfg
        fine = self.operator(self.domain().refine(c.supersample), c.data_det_per_pixel)
        return DataGenerator(
            fine,
            noise_std=c.noise_std,
            relative=True,
            supersample=c.supersample,
            fidelity_tag=f"radon3d-{c.supersample}x-float64",
        )

    def build_problem_measurement_shape(self, field_shape):
        return self.operator().output_shape(field_shape)

    # ---- prior / objective --------------------------------------------------------------
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
            3,
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
            (c.n, c.n, c.n_z),
            n_stages=2,
            steps=tuple(c.steps),
            lr=c.lr,
            lr_decay=c.lr_decay,
            min_size=c.min_coarse,
            anneal_fraction=c.anneal_fraction,
        )

    def build_problem(self, measurement: Measurement) -> InverseProblem:
        c = self.cfg
        return InverseProblem(
            self.domain(),
            self.field(),
            self.operator(),
            self.losses(),
            measurement,
            curriculum=self.default_curriculum(),
            name="ct3d",
            meta={"scene": c.scene, "n_views": c.n_views, "angle_range": c.angle_range},
        )

    # ---- evaluation ---------------------------------------------------------------------
    def metrics(self) -> dict[str, Callable]:
        return {
            "psnr": psnr,
            "ssim": ssim,
            "mse": mse,
            "iou": partial(iou_above, tau=self.cfg.iou_tau),
        }

    def measurement_image(self, measurement: Measurement) -> torch.Tensor:
        """Compact 2-D view: the sinogram ``(n_views, n_det)`` of the most structured slice."""
        d = torch.as_tensor(measurement.data)
        if d.ndim != 3:
            return d
        k = int(d.flatten(0, 1).var(dim=0).argmax())
        return d[..., k]

    # ---- classical reference & baselines ------------------------------------------------
    def fbp(self, measurement: Measurement, filter: str | None = None) -> torch.Tensor:
        """Slice-wise filtered back-projection of ``measurement`` on the native grid."""
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

            shape = (c.n, c.n, c.n_z)
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
                name="ct3d-fbp3d",
                meta={"solver": "direct", "reconstruct": reconstruct, "baseline": "fbp3d"},
            )
            return prob, cur

        return {"grid": grid, "fbp3d": fbp_baseline, "deep_decoder": deep_decoder}


def make_problem(
    seed: int = 0, scene: str | None = None, **cfg: Any
) -> tuple[InverseProblem, dict[str, torch.Tensor], Measurement]:
    """Convenience: ``(problem, gt, measurement)`` for docs and tests."""
    inst = CT3D(**({"scene": scene} if scene else {}), **cfg)
    gt, meas = inst.make_measurement(seed)
    return inst.build_problem(meas), gt, meas


def run(cfg: CT3DConfig | dict | None = None, seed: int = 0, **kw: Any):
    """End-to-end demo (generate → invert → evaluate); returns a ``RunOutput``."""
    return CT3D(cfg).run(seed=seed, **kw)


METRICS: dict[str, Callable] = {"psnr": psnr, "ssim": ssim, "mse": mse}

__all__ = [
    "FILTERS",
    "METRICS",
    "PRESETS",
    "SHEPP_LOGAN_3D",
    "CT3D",
    "CT3DConfig",
    "CT3DScenes",
    "Radon3DOperator",
    "backproject3d",
    "fbp3d",
    "make_problem",
    "run",
    "uniform_angles",
]
