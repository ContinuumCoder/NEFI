"""current_density — NV-magnetometry current imaging: sheet current from its stray field B_z.

A wide-field NV-diamond microscope images the out-of-plane stray field ``B_z`` of a current
distribution flowing in a thin film at standoff ``z0`` (Tetienne et al., *Phys. Rev. B* 99,
014436 (2019); Broadway et al., *Phys. Rev. Applied* 14, 024076 (2020); Midha et al., *Phys.
Rev. Applied* 22, 014015 (2024) — cited by NeTMY as the static-field sibling of NV noise
sensing). The unknown is the stream function ``g`` of the sheet current ``K = ∇×(g ẑ)``, so charge
conservation (``∇·K = 0``) is built into the parameterization.

Pipeline::

    scene (CurrentScenes: wires / loops / branching, Gaussian current profiles)
        ──direct Biot–Savart summation on a 2× finer source grid, float64, + noise──► B_z (H, W)
    coords ─► annealed PE ─► MLP ─► g (identity head) ─► CurrentDensityOperator
        (B̂_z = (μ0/2) k e^{−kz0} ĝ, zero-padded FFT) ─► relative MSE + isotropic TV on g
    two-stage curriculum H/2 → H; metrics on K = ∇×g and on the recovered B_z

Baselines: ``grid`` (free pixels + TV, same operator) and ``fourier`` (classical Tikhonov/Hanning
k-space inversion, reported through the same metrics). Units: μm, mA, μT by default
(``μ0 = 400π μT·μm/mA``).

Quick start::

    from nefi.instances.current_density import CurrentDensity
    out = CurrentDensity(n=32, steps=(100, 200)).run(seed=0, device="cpu")
    print(out.metrics)          # j_psnr, j_relative_error, bz_psnr, bz_relative_error

See ``docs/instances/current_density.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch

from ...bench.base import DataGenerator, SceneGenerator
from ...domain import Domain
from ...errors import ConfigError
from ...fields import GridField, Heads, Identity, NeuralField
from ...losses import MSE, TV, LossSet, RelativeMSE
from ...measurement import Measurement
from ...metrics.basic import psnr, relative_error
from ...physics.magnetostatics import (
    BiotSavartOperator,
    CurrentDensityOperator,
    fourier_inversion,
    magnetic_constant,
    stream_to_current,
)
from ...problem import InverseProblem
from ...registry import register
from ...solve.curriculum import Curriculum, Stage
from ...solve.result import Result
from ..base import Instance
from .scenes import CurrentScenes

__all__ = [
    "BiotSavartDataGenerator",
    "CurrentDensity",
    "CurrentDensityConfig",
    "CurrentScenes",
    "current_metrics",
    "make_problem",
    "run",
]


@dataclass
class CurrentDensityConfig:
    """NV current-imaging instance configuration (every number used by the pipeline).

    Lengths are in ``length_unit`` (default μm), currents in ``current_unit`` (mA), fields in
    ``field_unit`` (μT); ``μ0`` is derived from these units.
    """

    # geometry & physics
    n: int = 64
    pixel: float = 0.1
    z0: float = 0.2
    nv_layer: float = 0.0
    length_unit: str = "um"
    current_unit: str = "mA"
    field_unit: str = "uT"
    pad_factor: float = 2.0
    # scenes
    scene: str = "wires"
    current: tuple[float, float] = (0.5, 1.5)
    width_px: tuple[float, float] = (1.0, 1.5)
    margin: float = 0.15
    loop_radius: tuple[float, float] = (0.08, 0.2)
    wire_separation: tuple[float, float] = (0.08, 0.14)
    branch_width: tuple[float, float] = (0.04, 0.07)
    # data (inverse-crime guard: direct Biot–Savart on a supersample× finer grid in float64)
    noise_std: float = 0.01
    supersample: int = 2
    # field
    hidden: int = 128
    depth: int = 4
    skip_at: int | None = 2
    n_octaves: int = 8
    activation: str = "tanh"
    out_init_scale: float = 0.1
    # curriculum
    multiscale: bool = True
    min_coarse: int = 12
    steps: tuple[int, int] = (500, 1500)
    lr: float = 3e-3
    lr_decay: float = 0.5
    lr_min_ratio: float = 0.05
    anneal_fraction: float = 0.5
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    # losses
    data_loss: str = "relative_mse"
    tv: float = 1e-4
    tv_eps: float = 1e-3
    # baselines
    grid_lr: float = 5e-2
    grid_tv: float | None = None
    fourier_reg: float = 1e-3
    fourier_window: str = "hanning"
    fourier_cutoff: float | None = None  # cutoff wavelength λ_c; None → k_c z0 = 3


class BiotSavartDataGenerator(DataGenerator):
    """``g`` on a fine grid → direct Biot–Savart ``B_z`` at the native pixels (float64) + noise.

    The measurement carries ``meta["clean"]`` (noise-free field), moved into the ground truth by
    :meth:`CurrentDensity.make_measurement` for data-space metrics.
    """

    fidelity_tag = "biot-savart-direct-float64"

    def generate(self, gt_fields, rng: np.random.Generator, noise_std=None, target_shape=None):
        clean = self.clean(gt_fields)
        data, ns = self.add_noise(clean, rng, noise_std)
        return Measurement(
            data.float(),
            noise_std=ns if ns > 0 else None,
            meta={
                "fidelity": self.fidelity_tag,
                "supersample": self.supersample,
                "clean": clean.float(),
            },
        )


def current_metrics(
    pred_g: torch.Tensor,
    gt_g: torch.Tensor,
    spacing,
    pred_bz: torch.Tensor | None = None,
    gt_bz: torch.Tensor | None = None,
) -> dict[str, float]:
    """PSNR / relative error of ``K = ∇×g`` (central differences, both components) and of B_z."""
    jp = torch.stack(stream_to_current(torch.as_tensor(pred_g).double(), spacing))
    jg = torch.stack(stream_to_current(torch.as_tensor(gt_g).double(), spacing))
    out = {"j_psnr": psnr(jp, jg), "j_relative_error": relative_error(jp, jg)}
    if pred_bz is not None and gt_bz is not None:
        out["bz_psnr"] = psnr(pred_bz, gt_bz)
        out["bz_relative_error"] = relative_error(pred_bz, gt_bz)
    return out


@register("instance", "current_density")
class CurrentDensity(Instance):
    """NV magnetometry current-density imaging (stream-function parameterization)."""

    name = "current_density"
    Config = CurrentDensityConfig
    description = "NV magnetometry: divergence-free sheet current from its stray field B_z"
    #: Display hints (:mod:`nefi.viz.hints`): the measured ``B_z`` is signed (diverging map centred
    #: at 0); the unknown is the stream function ``g``, but the physically meaningful map — and the
    #: quantity the ``j_*`` metrics score — is the sheet-current magnitude ``|K| = |∇×(g ẑ)|``.
    viz_hints: ClassVar[dict[str, Any]] = {
        "layout": "image",
        "measurement_cmap": "signed",
        "measurement_label": "B_z",
        "field_transform": {"g": "curl_magnitude"},
        "field_label": {"g": "|J| = |∇×g|"},
    }

    # --- geometry / physics -------------------------------------------------------------
    @property
    def mu0(self) -> float:
        c = self.cfg
        return magnetic_constant(c.length_unit, c.current_unit, c.field_unit)

    def domain(self) -> Domain:
        c = self.cfg
        half = 0.5 * c.n * c.pixel
        return Domain.from_spacing((c.n, c.n), c.pixel, origin=-half, axes=("x", "y"))

    def operator(self, domain: Domain | None = None) -> CurrentDensityOperator:
        c = self.cfg
        return CurrentDensityOperator(
            domain or self.domain(),
            c.z0,
            field="g",
            mu0=self.mu0,
            pad_factor=c.pad_factor,
            nv_layer_thickness=c.nv_layer,
        )

    def scene_generator(self) -> SceneGenerator:
        c = self.cfg
        return CurrentScenes(
            self.domain(),
            current=c.current,
            width=tuple(w * c.pixel for w in c.width_px),
            margin=c.margin,
            loop_radius=c.loop_radius,
            wire_separation=c.wire_separation,
            branch_width=c.branch_width,
        )

    def data_generator(self) -> BiotSavartDataGenerator:
        c = self.cfg
        if c.nv_layer > 0:
            raise ConfigError(
                "nv_layer > 0 is not supported by the direct Biot–Savart data generator; "
                "use nv_layer = 0 (the FFT operator supports layer averaging)"
            )
        op = BiotSavartOperator(self.domain(), c.z0, field="g", mu0=self.mu0)
        return BiotSavartDataGenerator(
            op,
            noise_std=c.noise_std,
            relative=True,
            supersample=c.supersample,
            fidelity_tag="biot-savart-direct-float64",
        )

    def build_problem_measurement_shape(self, field_shape):
        return tuple(field_shape)

    def make_measurement(self, seed=0, scene_class=None, gt=None):
        gt_out, meas = super().make_measurement(seed, scene_class, gt)
        clean = meas.meta.pop("clean", None)
        if clean is not None:
            gt_out = {**gt_out, "bz": clean}
        return gt_out, meas

    # --- inversion ----------------------------------------------------------------------
    def heads(self) -> Heads:
        return Heads({"g": Identity()})

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
            {"data": data(), "tv": TV("g", isotropic=True, eps=c.tv_eps)},
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

    def build_problem(self, measurement: Measurement) -> InverseProblem:
        return InverseProblem(
            self.domain(),
            self.field(),
            self.operator(),
            self.losses(),
            measurement,
            curriculum=self.default_curriculum(),
            name="current_density",
        )

    # --- evaluation ---------------------------------------------------------------------
    def metrics(self) -> dict[str, Callable]:
        sp = self.domain().spacing()
        return {
            "j_psnr": lambda p, g: current_metrics(p, g, sp)["j_psnr"],
            "j_relative_error": lambda p, g: current_metrics(p, g, sp)["j_relative_error"],
        }

    def evaluate(self, result: Result, gt: Mapping[str, torch.Tensor]) -> dict[str, float]:
        g = result.fields["g"]
        sp = self.domain().spacing()
        bz = gt.get("bz")
        pred = (
            result.pred if bz is not None and tuple(result.pred.shape) == tuple(bz.shape) else None
        )
        return current_metrics(g, gt["g"], sp, pred, bz if pred is not None else None)

    def fourier_reconstruction(self, measurement: Measurement) -> torch.Tensor:
        """Classical regularized k-space inversion of the measured ``B_z`` (stream function)."""
        c = self.cfg
        return fourier_inversion(
            measurement.data.double(),
            c.z0,
            self.domain().spacing(),
            mu0=self.mu0,
            reg=c.fourier_reg,
            window=c.fourier_window,
            cutoff_wavelength=c.fourier_cutoff,
            pad_factor=c.pad_factor,
        ).float()

    def baselines(self):
        c = self.cfg

        def grid(measurement: Measurement):
            field = GridField((c.n, c.n), self.heads(), init=0.0)
            problem = InverseProblem(
                self.domain(),
                field,
                self.operator(),
                self.losses(c.grid_tv if c.grid_tv is not None else c.tv),
                measurement,
                name="current_density-grid",
            )
            cur = self._curriculum(c.grid_lr)
            problem.curriculum = cur
            return problem, cur

        def fourier(measurement: Measurement):
            g0 = self.fourier_reconstruction(measurement)
            field = GridField((c.n, c.n), self.heads(), init=g0[..., None])
            problem = InverseProblem(
                self.domain(),
                field,
                self.operator(),
                self.losses(),
                measurement,
                name="current_density-fourier",
                meta={"baseline": "fourier", "closed_form": True},
            )
            # closed-form estimate: one zero-learning-rate step just evaluates it
            cur = Curriculum(
                [Stage("fourier", (c.n, c.n), 1, 0.0, lr_schedule="constant", anneal=False)]
            )
            problem.curriculum = cur
            return problem, cur

        return {"grid": grid, "fourier": fourier}


def make_problem(seed: int = 0, scene_class: str | None = None, **cfg):
    """Convenience: ``(problem, gt, measurement)`` for docs, tests and notebooks."""
    inst = CurrentDensity(**cfg)
    gt, meas = inst.make_measurement(seed, scene_class)
    return inst.build_problem(meas), gt, meas


def run(cfg: CurrentDensityConfig | dict | None = None, seed: int = 0, **kw):
    """End-to-end demo: generate → invert → evaluate. Returns ``(Result, metrics)``."""
    out = CurrentDensity(cfg).run(seed=seed, **kw)
    return out.result, out.metrics
