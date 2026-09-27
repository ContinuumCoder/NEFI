"""holography — multi-distance inline (Gabor) holography phase retrieval.

Unknown: the phase ``φ(x)`` (rad) of a thin, transparent sample (transmission ``exp(iφ − a)`` with a
known constant absorption ``a``) on an ``n × n`` pixel grid (pixel pitch ``pixel_size`` μm),
parameterized by a neural field with a ``ZeroMean(Bounded(−phase_bound, phase_bound))`` head started
at 0.
Data: intensities ``|P_{z_i} t|²`` at ``len(distances)`` propagation distances — output
``(n_distances, n, n)`` — computed by band-limited angular-spectrum propagation
(:mod:`nefi.physics.optics`; the sample sits in an infinite uniform background → edge padding).

The global phase offset is invisible in intensities, so the phase is **zero-mean by construction**
(:class:`~nefi.fields.ZeroMean` head, ``zero_mean=True``) and the phantoms are zero-mean too: the
reconstruction and the ground truth share one gauge and are directly comparable (figures included).
The metrics (``psnr``, ``ssim``, ``rmse``) still compare mean-subtracted phases, so they are
offset-invariant for any parameterization. Multiple distances fill the zeros of the single-distance
phase-contrast transfer function ``sin(πλz|f|²)`` (the twin-image / low-frequency ambiguity).

Inverse-crime guard: data are simulated on a 2× finer grid in float64 and the intensities are
area-averaged onto the detector pixels (``fidelity_tag="angular-spectrum-2x-float64"`` vs
``"angular-spectrum-1x"``). Baselines: ``"grid"`` (pixel phase, same losses) and
``"gerchberg_saxton"`` (multi-plane iterative projections, closed-form-style direct method).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from ...bench.base import DataGenerator, SceneGenerator
from ...domain import Domain
from ...fields import Bounded, GridField, Heads, NeuralField, ZeroMean
from ...losses import TV, LossSet, RelativeMSE
from ...measurement import Measurement
from ...metrics.basic import rmse
from ...physics.optics import HolographyOperator, gerchberg_saxton
from ...problem import InverseProblem
from ...registry import register
from ...solve.curriculum import Curriculum
from .._wave_family import (
    direct_baseline_problem,
    mean_subtracted,
    phase_psnr,
    phase_ssim,
    spatial_downsample_obs,
)
from ..base import Instance


@dataclass
class HolographyConfig:
    """Configuration (defaults = CPU smoke preset: 32², 3 distances)."""

    n: int = 32  # detector / object grid (n × n pixels)
    pixel_size: float = 1.0  # μm
    wavelength: float = 0.5  # μm
    distances: tuple[float, ...] = (10.0, 25.0, 50.0)  # μm, propagation distances
    scene: str = "smooth_phase"  # smooth_phase | cells
    phase_amplitude: float = 1.0  # rad, peak |φ| of the phantom (before the zero-mean shift)
    absorption: float = 0.0  # known constant amplitude attenuation a (t = e^{iφ − a})
    noise_std: float = 0.01  # Gaussian, relative to max intensity
    supersample: int = 2  # generator grid refinement (intensities area-averaged)
    pad_factor: float = 2.0
    pad_mode: str = "edge"  # edge | constant | none
    band_limit: bool = True
    phase_bound: float = math.pi  # Bounded head range [−b, b]
    zero_mean: bool = True  # ZeroMean head: fix the invisible global phase (gauge) to mean 0
    hidden: int = 64
    depth: int = 4
    n_octaves: int = 6
    activation: str = "tanh"
    tv: float = 1e-4
    steps: tuple[int, int] = (150, 250)
    lr: float = 1e-2
    lr_decay: float = 0.5
    grid_lr_mult: float = 10.0
    gs_iterations: int = 100  # Gerchberg–Saxton baseline iterations


class HolographyScenes(SceneGenerator):
    """Phase phantoms ``φ(x)`` in rad (analytic, any resolution)."""

    classes = ("smooth_phase", "cells")

    def __init__(self, domain: Domain, amplitude: float, zero_mean: bool = True) -> None:
        super().__init__(domain)
        self.amplitude = float(amplitude)
        self.zero_mean = bool(zero_mean)

    def sample(self, rng: np.random.Generator, cls: str | None = None, shape=None):
        cls = self.check_class(cls)
        L = max(self.domain.size)
        lo = torch.tensor([e[0] for e in self.domain.extent], dtype=torch.float64)
        xy = (self.domain.physical_coords(shape, dtype=torch.float64) - lo) / L  # [0, 1]²
        x, y = xy[..., 0], xy[..., 1]
        phi = torch.zeros_like(x)
        if cls == "smooth_phase":
            for _ in range(int(rng.integers(3, 6))):
                cx, cy = rng.uniform(0.2, 0.8, size=2)
                s = rng.uniform(0.06, 0.14)
                a = rng.uniform(0.3, 1.0) * (1.0 if rng.uniform() > 0.3 else -1.0)
                phi = phi + a * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s**2))
            phi = phi / phi.abs().max().clamp_min(1e-12) * self.amplitude
        else:  # cells: smooth plateaus (cytoplasm) with a denser nucleus
            for _ in range(int(rng.integers(2, 5))):
                cx, cy = rng.uniform(0.25, 0.75, size=2)
                a = rng.uniform(0.08, 0.14)
                b = a * rng.uniform(0.6, 1.0)
                th = rng.uniform(0, math.pi)
                u = ((x - cx) * math.cos(th) + (y - cy) * math.sin(th)) / a
                v = (-(x - cx) * math.sin(th) + (y - cy) * math.cos(th)) / b
                r = torch.sqrt(u**2 + v**2)
                cell = 0.5 * (1 + torch.tanh((1.0 - r) / 0.15))
                nuc = torch.exp(-((u - 0.2) ** 2 + (v + 0.1) ** 2) / (2 * 0.3**2))
                phi = torch.maximum(phi, self.amplitude * (0.5 * cell + 0.5 * nuc * cell))
        if self.zero_mean:  # the gauge of the ZeroMean head (a global phase is not observable)
            phi = phi - phi.mean()
        return {"phase": phi.float()}


@register("instance", "holography")
class Holography(Instance):
    """Multi-distance inline holography instance (see module docstring)."""

    name = "holography"
    Config = HolographyConfig
    description = "Multi-distance inline holography phase retrieval (angular spectrum)"

    @property
    def viz_hints(self) -> dict:
        """Display hints (:mod:`nefi.viz.hints`): with the :class:`~nefi.fields.ZeroMean` head
        the phase and the phantoms share one gauge, so the viewer's ``zero_mean`` display
        transform (the fallback for ``zero_mean=False``) is switched off."""
        return {"field_transform": None} if self.cfg.zero_mean else {}

    def domain(self) -> Domain:
        c = self.cfg
        return Domain.from_spacing((c.n, c.n), c.pixel_size, axes=("x", "y"))

    def operator(self, domain: Domain | None = None) -> HolographyOperator:
        c = self.cfg
        return HolographyOperator(
            domain or self.domain(),
            c.distances,
            c.wavelength,
            field="phase",
            absorption=c.absorption if c.absorption else None,
            band_limit=c.band_limit,
            pad_factor=c.pad_factor,
            pad_mode=c.pad_mode,
        )

    def scene_generator(self) -> SceneGenerator:
        return HolographyScenes(self.domain(), self.cfg.phase_amplitude, self.cfg.zero_mean)

    def data_generator(self) -> DataGenerator:
        c = self.cfg
        fine = self.domain().refine(c.supersample) if c.supersample > 1 else self.domain()
        return DataGenerator(
            self.operator(fine),
            noise_std=c.noise_std,
            relative=True,
            supersample=c.supersample,
            fidelity_tag=f"angular-spectrum-{c.supersample}x-float64",
        )

    def build_problem_measurement_shape(self, field_shape):
        return (len(self.cfg.distances), *tuple(field_shape))

    def heads(self) -> Heads:
        b = self.cfg.phase_bound
        head = Bounded(-b, b, init_value=0.0)
        return Heads({"phase": ZeroMean(head) if self.cfg.zero_mean else head})

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
            {"fit": RelativeMSE(), "tv": TV("phase", isotropic=True)},
            weights={"fit": 1.0, "tv": self.cfg.tv},
        )

    def _problem(self, field, measurement, curriculum, name, meta=None) -> InverseProblem:
        return InverseProblem(
            self.domain(),
            field,
            self.operator(),
            self.losses(),
            measurement,
            curriculum=curriculum,
            downsample_obs=spatial_downsample_obs(2),
            name=name,
            meta=meta or {},
        )

    def build_problem(self, measurement: Measurement) -> InverseProblem:
        return self._problem(self.field(), measurement, self.default_curriculum(), "holography")

    def default_curriculum(self, lr_mult: float = 1.0) -> Curriculum:
        c = self.cfg
        return Curriculum.multiscale(
            (c.n, c.n), n_stages=2, steps=c.steps, lr=c.lr * lr_mult, lr_decay=c.lr_decay
        )

    def metrics(self) -> dict[str, Callable]:
        return {"psnr": phase_psnr, "ssim": phase_ssim, "rmse": mean_subtracted(rmse)}

    def gerchberg_saxton(self, measurement: Measurement, domain: Domain | None = None):
        """Multi-distance Gerchberg–Saxton phase (closed-form-style baseline)."""
        op = self.operator(domain)
        phase, _ = gerchberg_saxton(measurement.data, op, n_iter=self.cfg.gs_iterations)
        return phase

    def baselines(self):
        def grid(measurement: Measurement):
            cur = self.default_curriculum(lr_mult=self.cfg.grid_lr_mult)
            field = GridField((self.cfg.n, self.cfg.n), self.heads())
            return (
                self._problem(field, measurement, cur, "holography-grid", {"baseline": "grid"}),
                cur,
            )

        def gs(measurement: Measurement):
            prob = self.build_problem(measurement)

            def reconstruct(meas: Measurement, domain: Domain) -> torch.Tensor:
                return self.gerchberg_saxton(meas, domain)

            return direct_baseline_problem(prob, self.heads(), reconstruct, "gerchberg_saxton")

        return {"grid": grid, "gerchberg_saxton": gs}


PRESETS: dict[str, dict] = {
    "smoke": {},
    "full": {
        "n": 128,
        "pixel_size": 0.5,
        "distances": (20.0, 45.0, 80.0, 130.0),
        "hidden": 256,
        "depth": 6,
        "n_octaves": 7,
        "steps": (1500, 3500),
        "lr": 2e-3,
        "gs_iterations": 300,
    },
}


def make_problem(seed: int = 0, **cfg):
    """Convenience: ``(problem, gt, measurement)``."""
    inst = Holography(**cfg)
    gt, meas = inst.make_measurement(seed)
    return inst.build_problem(meas), gt, meas


def run(cfg=None, seed: int = 0, **kw):
    """End-to-end demo (DESIGN §3.10): ``(result, metrics)``."""
    out = Holography(cfg).run(seed=seed, **kw)
    return out.result, out.metrics


__all__ = ["PRESETS", "Holography", "HolographyConfig", "HolographyScenes", "make_problem", "run"]
