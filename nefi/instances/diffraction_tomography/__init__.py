"""diffraction_tomography — 2-D Born / Rytov inversion of a weak scattering contrast.

Unknown: contrast ``χ(x) = n(x)²/n_b² − 1`` (dimensionless) on a square object domain of side
``extent`` (μm; ``λ = 1 μm`` by default, so lengths are in wavelengths), parameterized by a neural
field with a ``Bounded(chi_min, chi_max)`` head started at 0 (homogeneous background).
Data: complex scattered fields (stacked real / imaginary parts) on a ring of receivers for
``n_angles`` plane-wave illuminations — ``(2, n_angles, n_receivers)`` — predicted by the linear
Born (or first-order Rytov) operator of :mod:`nefi.physics.scattering` (``homogeneity = 1``).

Inverse-crime guard: measurements are generated with **multiple scattering** (Lippmann–Schwinger
fixed-point iterations with the truncated free-space Green's function) on a 2× finer grid in
float64 (``fidelity_tag="lippmann-schwinger-iterated-2x-float64"`` vs ``"born-1x"``); at the default
contrast the Born model error is a few percent of the data, comparable to the noise.

Scene classes: ``blobs`` (Gaussian inclusions) and ``cells`` (elliptical cells with nuclei, smooth
edges). Baselines: ``"grid"`` (pixel field, same losses) and ``"backpropagation"`` (Devaney's
filtered backpropagation, closed form).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from ...bench.base import DataGenerator, SceneGenerator
from ...domain import Domain
from ...fields import Bounded, GridField, Heads, NeuralField
from ...losses import TV, LossSet, RelativeMSE
from ...measurement import Measurement
from ...metrics.basic import psnr, relative_error, ssim
from ...physics.scattering import BornOperator, LippmannSchwingerOperator, filtered_backpropagation
from ...problem import InverseProblem
from ...registry import register
from ...solve.curriculum import Curriculum
from .._wave_family import direct_baseline_problem
from ..base import Instance


@dataclass
class DiffractionTomographyConfig:
    """Configuration (defaults = CPU smoke preset: 32², 16 angles, 64 receivers)."""

    n: int = 32  # object grid (n × n)
    extent: float = 4.0  # μm, object domain side (centered at the origin)
    wavelength: float = 1.0  # μm (vacuum)
    background_index: float = 1.0
    scene: str = "blobs"  # blobs | cells
    contrast: float = 0.02  # peak χ of the phantom (weak scattering)
    n_angles: int = 16  # plane-wave illuminations over 360°
    n_receivers: int = 64  # receivers on a ring
    receiver_radius: float = 3.0  # μm (must exceed the half-diagonal extent/√2)
    rytov: bool = False  # first-order Rytov data model
    noise_std: float = 0.01  # relative to max |data|
    supersample: int = 2  # generator grid refinement
    ls_iterations: int = 200
    ls_tol: float = 1e-10
    ls_relax: float = 1.0
    chi_min: float = -0.05
    chi_max: float = 0.1
    hidden: int = 64
    depth: int = 4
    n_octaves: int = 5  # 16 cycles over the domain = Nyquist at 32 px
    activation: str = "tanh"
    tv: float = 1e-3
    steps: tuple[int, int] = (200, 300)
    lr: float = 2e-2
    lr_decay: float = 0.5
    grid_lr_mult: float = 1.0  # the base lr is already large (tuned: 1 > 2.5 > 5 for the grid)


class DiffractionTomographyScenes(SceneGenerator):
    """Weak-contrast phantoms ``χ(x)`` (analytic, any resolution)."""

    classes = ("blobs", "cells")

    def __init__(self, domain: Domain, contrast: float) -> None:
        super().__init__(domain)
        self.contrast = float(contrast)

    def sample(self, rng: np.random.Generator, cls: str | None = None, shape=None):
        cls = self.check_class(cls)
        D = max(self.domain.size)
        xy = self.domain.physical_coords(shape, dtype=torch.float64) / D  # [-0.5, 0.5]²
        x, y = xy[..., 0], xy[..., 1]
        chi = torch.zeros_like(x)
        if cls == "blobs":
            for _ in range(int(rng.integers(2, 5))):
                r, a = rng.uniform(0.0, 0.25), rng.uniform(0, 2 * math.pi)
                cx, cy = r * math.cos(a), r * math.sin(a)
                s = rng.uniform(0.05, 0.1)
                amp = rng.uniform(0.4, 1.0) * (1.0 if rng.uniform() > 0.25 else -0.5)
                chi = chi + amp * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s**2))
            chi = chi * self.contrast
        else:  # cells
            cell = torch.zeros_like(x)
            nucleus = torch.zeros_like(x)
            for _ in range(int(rng.integers(2, 4))):
                cx, cy = rng.uniform(-0.2, 0.2, size=2)
                a = rng.uniform(0.1, 0.17)
                b = a * rng.uniform(0.6, 1.0)
                th = rng.uniform(0, math.pi)
                u = ((x - cx) * math.cos(th) + (y - cy) * math.sin(th)) / a
                v = (-(x - cx) * math.sin(th) + (y - cy) * math.cos(th)) / b
                rr = torch.sqrt(u**2 + v**2)
                cell = torch.maximum(cell, 0.5 * (1 + torch.tanh((1.0 - rr) / 0.12)))
                ox, oy = rng.uniform(-0.25, 0.25, size=2)
                rn = torch.sqrt((u - ox) ** 2 + (v - oy) ** 2) / 0.45
                nucleus = torch.maximum(nucleus, 0.5 * (1 + torch.tanh((1.0 - rn) / 0.15)))
            chi = self.contrast * (0.5 * cell + 0.5 * nucleus * cell)
        return {"chi": chi.float()}


@register("instance", "diffraction_tomography")
class DiffractionTomography(Instance):
    """Born diffraction tomography instance (see module docstring)."""

    name = "diffraction_tomography"
    Config = DiffractionTomographyConfig
    description = "2-D diffraction tomography (Born/Rytov) of a weak scattering contrast"

    def domain(self) -> Domain:
        h = 0.5 * self.cfg.extent
        return Domain((self.cfg.n, self.cfg.n), ((-h, h), (-h, h)), axes=("x", "y"))

    def _geometry(self) -> dict:
        c = self.cfg
        return {
            "n_angles": c.n_angles,
            "n_receivers": c.n_receivers,
            "receiver_radius": c.receiver_radius,
            "background_index": c.background_index,
            "rytov": c.rytov,
            "field": "chi",
            "contrast": "chi",
        }

    def operator(self, domain: Domain | None = None) -> BornOperator:
        return BornOperator(domain or self.domain(), self.cfg.wavelength, **self._geometry())

    def scene_generator(self) -> SceneGenerator:
        return DiffractionTomographyScenes(self.domain(), self.cfg.contrast)

    def data_generator(self) -> DataGenerator:
        c = self.cfg
        fine = self.domain().refine(c.supersample) if c.supersample > 1 else self.domain()
        op = LippmannSchwingerOperator(
            fine,
            c.wavelength,
            n_iter=c.ls_iterations,
            tol=c.ls_tol,
            relax=c.ls_relax,
            **self._geometry(),
        )
        return DataGenerator(
            op,
            noise_std=c.noise_std,
            relative=True,
            supersample=c.supersample,
            fidelity_tag=f"lippmann-schwinger-iterated-{c.supersample}x-float64",
        )

    def build_problem_measurement_shape(self, field_shape):
        return (2, self.cfg.n_angles, self.cfg.n_receivers)

    def heads(self) -> Heads:
        return Heads({"chi": Bounded(self.cfg.chi_min, self.cfg.chi_max, init_value=0.0)})

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
        return LossSet(
            {"fit": RelativeMSE(), "tv": TV("chi", isotropic=True)},
            weights={"fit": 1.0, "tv": self.cfg.tv},
        )

    def build_problem(self, measurement: Measurement) -> InverseProblem:
        return InverseProblem(
            self.domain(),
            self.field(),
            self.operator(),
            self.losses(),
            measurement,
            curriculum=self.default_curriculum(),
            name="diffraction_tomography",
        )

    def default_curriculum(self, lr_mult: float = 1.0) -> Curriculum:
        c = self.cfg
        return Curriculum.multiscale(
            (c.n, c.n), n_stages=2, steps=c.steps, lr=c.lr * lr_mult, lr_decay=c.lr_decay
        )

    def metrics(self) -> dict[str, Callable]:
        return {"psnr": psnr, "ssim": ssim, "relative_error": relative_error}

    def backpropagation(self, measurement: Measurement, domain: Domain | None = None):
        """Filtered backpropagation reconstruction of ``χ`` (closed form)."""
        op = self.operator(domain)
        return filtered_backpropagation(measurement.data, op)

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
                name="diffraction_tomography-grid",
                meta={"baseline": "grid"},
            )
            return prob, cur

        def backpropagation(measurement: Measurement):
            prob = self.build_problem(measurement)

            def reconstruct(meas: Measurement, domain: Domain) -> torch.Tensor:
                return self.backpropagation(meas, domain)

            return direct_baseline_problem(prob, self.heads(), reconstruct, "backpropagation")

        return {"grid": grid, "backpropagation": backpropagation}


PRESETS: dict[str, dict] = {
    "smoke": {},
    "full": {
        "n": 96,
        "extent": 8.0,
        "receiver_radius": 6.0,
        "n_angles": 64,
        "n_receivers": 192,
        "hidden": 256,
        "depth": 6,
        "n_octaves": 7,
        "steps": (1500, 3500),
        "lr": 1e-3,
    },
}


def make_problem(seed: int = 0, **cfg):
    """Convenience: ``(problem, gt, measurement)``."""
    inst = DiffractionTomography(**cfg)
    gt, meas = inst.make_measurement(seed)
    return inst.build_problem(meas), gt, meas


def run(cfg=None, seed: int = 0, **kw):
    """End-to-end demo (DESIGN §3.10): ``(result, metrics)``."""
    out = DiffractionTomography(cfg).run(seed=seed, **kw)
    return out.result, out.metrics


__all__ = [
    "PRESETS",
    "DiffractionTomography",
    "DiffractionTomographyConfig",
    "DiffractionTomographyScenes",
    "make_problem",
    "run",
]
