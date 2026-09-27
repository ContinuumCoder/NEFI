"""photoacoustic3d — 3-D photoacoustic tomography (PAT) from a planar sensor array.

Recover the initial pressure ``p0(x, y, z) ≥ 0`` (the absorbed optical energy after a short laser
pulse) of a tissue slab from the pressure traces recorded by a planar ``n_sx × n_sy`` array of
point sensors on the **top face** — the limited-view geometry of planar / linear-array PAT
(structures parallel to the view direction are invisible; the classical time-reversal image
suffers from limited-view artifacts). The acoustic propagation is the hard physics constraint:

    ∂_t² p = c² Δp,   p(x, 0) = p0(x),   ∂_t p(x, 0) = 0     (homogeneous, known c)

solved by :class:`~nefi.physics.wave.WaveInitialConditionOperator` in 3-D: leapfrog in time,
centered 2nd/4th-order Laplacian (one ``conv3d`` per step), a sponge absorbing layer around the
slab (the 3-D solver has no PML), sensors sampled by trilinear interpolation at fixed physical
times, so the output ``(n_sensors, n_t)`` is resolution independent. The sensors' finite
bandwidth is part of the model (:class:`~nefi.instances.photoacoustic3d.sensor.SensorBandwidth`,
a zero-phase Gaussian low-pass at ``sensor_fc``). The map is **linear** (``homogeneity = 1``);
gradients by autodiff through the unrolled scheme (the discrete adjoint).

* **field** — coordinate MLP on annealed 3-D Fourier features with a ``Softplus`` head;
* **losses** — trace MSE + isotropic 3-D TV (NeFTY Eq. 22) + optional ℓ1;
* **curriculum** — two-stage multiscale (half resolution: cheaper, smoother wave solves, the
  same traces), then native;
* **data** — :class:`PAT3DScenes` (``vessels``, ``spheres``) simulated on a 2× finer grid in float64
  (``gen_order`` stencil; ``fidelity_tag="wave-ic-o2-2x-float64-lp1.0"`` vs the inversion's
  native operator — set ``gen_order=4`` for a different stencil too), plus relative Gaussian
  noise;
* **metrics** — PSNR, slice-averaged SSIM, MSE;
* **baselines** — ``grid`` (free voxels + the same objective) and ``time_reversal`` (closed form:
  k-Wave-style time reversal with the sensors as Dirichlet points, Treeby & Cox 2010; or the
  adjoint back-projection with ``tr_mode="adjoint"``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from ...baselines import baseline_problem
from ...bench.base import DataGenerator, SceneGenerator
from ...domain import Domain
from ...errors import ConfigError
from ...fields import GridField, Heads, Identity, NeuralField, Softplus
from ...losses import L1, MSE, TV, LossSet
from ...measurement import Measurement
from ...metrics.basic import mse, psnr, ssim
from ...physics.wave import WaveInitialConditionOperator, time_reversal
from ...problem import InverseProblem
from ...registry import register
from ...solve.curriculum import Curriculum
from .._volumetric import PresetInstance
from .scenes import PAT3DScenes
from .sensor import SensorBandwidth, gaussian_lowpass


@dataclass
class Photoacoustic3DConfig:
    """Configuration of the :class:`Photoacoustic3D` instance (every number is a field; mm, µs).

    Physics: slab ``extent`` on ``grid`` voxels (last axis = depth), sound speed ``sound_speed``
    (mm/µs); ``sensor_grid`` point sensors over the central ``sensor_span`` of the top face at
    depth ``sensor_depth`` with a Gaussian bandwidth ``sensor_fc`` (MHz; ``None`` = ideal); traces
    sampled every ``dt_obs`` up to ``t_max``; leapfrog with stencil
    ``order``, ``absorbing`` layer (``absorb_width`` mm, ``absorb_R``), ``courant``,
    ``grad_mode`` (``checkpoint_every``); relative noise ``noise_std``; data on a
    ``supersample``× grid with stencil ``gen_order``. Scenes: ``scene`` ∈ {vessels, spheres},
    ``vessel_radius``, ``sphere_radius``, ``edge_width``, ``depth_range``, ``render_factor``.
    Prior / solver: neural field (``hidden``, ``depth``, ``skip_at``, ``n_octaves``,
    ``activation``), ``init_value``, ``tv`` (``tv_eps``), ``l1``, two-stage curriculum
    (``steps``, ``lr``, ``lr_decay``, ``anneal_fraction``, ``min_coarse``). Baselines:
    ``grid_lr``, ``tr_mode`` ∈ {dirichlet, adjoint}.
    """

    extent: tuple[float, float, float] = (12.0, 12.0, 8.0)
    grid: tuple[int, int, int] = (48, 48, 32)
    sound_speed: float = 1.5
    sensor_grid: tuple[int, int] = (16, 16)
    sensor_span: float = 0.9
    sensor_depth: float = 0.0
    sensor_fc: float | None = 1.0
    t_max: float = 12.0
    dt_obs: float = 0.25
    order: int = 2
    absorbing: str = "sponge"
    absorb_width: float = 1.0
    absorb_R: float = 1e-3
    courant: float = 0.9
    grad_mode: str = "autograd"
    checkpoint_every: int | None = None
    noise_std: float = 0.02
    supersample: int = 2
    gen_order: int = 2
    scene: str = "vessels"
    vessel_radius: tuple[float, float] = (0.3, 0.6)
    sphere_radius: tuple[float, float] = (0.6, 1.5)
    edge_width: float = 0.15
    depth_range: tuple[float, float] = (0.15, 0.8)
    render_factor: int = 2
    init_value: float = 0.02
    tv: float = 1e-4
    tv_eps: float = 1e-3
    l1: float = 0.0
    hidden: int = 128
    depth: int = 4
    skip_at: int = 2
    n_octaves: int = 5
    activation: str = "tanh"
    steps: tuple[int, int] = (300, 500)
    lr: float = 1e-2
    lr_decay: float = 0.5
    anneal_fraction: float = 0.3
    min_coarse: int = 8
    grid_lr: float = 5e-2
    tr_mode: str = "dirichlet"


#: Named presets (partial configs applied before user overrides).
PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    # CPU smoke run: a 5 × 5 × 3.5 mm crop at the same 0.25 mm voxels (20×20×14), 8×8 sensors,
    # 21 samples of 0.25 µs (60 leapfrog steps); most steps on the 16× cheaper coarse grid.
    "smoke": {
        "extent": (5.0, 5.0, 3.5),
        "grid": (20, 20, 14),
        "t_max": 5.0,
        "sensor_grid": (8, 8),
        "hidden": 64,
        "depth": 3,
        "skip_at": 2,
        "n_octaves": 4,
        "steps": (150, 50),
        "lr": 2e-2,
    },
    "spheres": {"scene": "spheres"},
}


@register("instance", "photoacoustic3d")
class Photoacoustic3D(PresetInstance):
    """3-D planar-array photoacoustic tomography instance (see module docstring).

    Args:
        cfg: :class:`Photoacoustic3DConfig`, a dict, or ``None``.
        preset: optional preset from :data:`PRESETS` (``"smoke"``, ``"spheres"``).
        **overrides: individual config fields.
    """

    name = "photoacoustic3d"
    Config = Photoacoustic3DConfig
    PRESETS = PRESETS
    description = "3-D photoacoustic tomography from a planar sensor array (wave equation)"
    #: sensor traces (n_sensors, n_t): drawn as a time-down gather
    viz_hints = {"layout": "traces"}

    def _validate(self) -> None:
        c = self.cfg
        if len(c.grid) != 3 or len(c.extent) != 3:
            raise ConfigError("photoacoustic3d needs a 3-D grid and extent")
        if c.absorbing == "pml":
            raise ConfigError("the 3-D wave solver has no PML; use absorbing='sponge' or 'cerjan'")
        if c.tr_mode not in ("dirichlet", "adjoint"):
            raise ConfigError("tr_mode must be 'dirichlet' or 'adjoint'")
        if c.sound_speed <= 0 or c.dt_obs <= 0 or c.t_max <= 0:
            raise ConfigError("sound_speed, dt_obs and t_max must be positive")

    # ---- geometry / physics -----------------------------------------------------------------
    def domain(self) -> Domain:
        c = self.cfg
        return Domain(
            tuple(int(n) for n in c.grid),
            tuple((0.0, float(e)) for e in c.extent),
            axes=("x", "y", "z"),
        )

    @property
    def n_t(self) -> int:
        return int(round(self.cfg.t_max / self.cfg.dt_obs)) + 1

    def sensors(self) -> torch.Tensor:
        """Sensor positions ``(n_sx · n_sy, 3)``: a regular grid on the top face."""
        c = self.cfg
        (x0, x1), (y0, y1), (z0, _) = self.domain().extent
        nx, ny = (int(v) for v in c.sensor_grid)
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        hx, hy = 0.5 * c.sensor_span * (x1 - x0), 0.5 * c.sensor_span * (y1 - y0)
        xs = cx - hx + (torch.arange(nx, dtype=torch.float64) + 0.5) * (2 * hx / nx)
        ys = cy - hy + (torch.arange(ny, dtype=torch.float64) + 0.5) * (2 * hy / ny)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        Z = torch.full_like(X, z0 + c.sensor_depth)
        return torch.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)

    def wave_operator(
        self, domain: Domain | None = None, order: int | None = None
    ) -> WaveInitialConditionOperator:
        """The bare wave solver (ideal point sensors)."""
        c = self.cfg
        return WaveInitialConditionOperator(
            domain or self.domain(),
            self.sensors(),
            n_t=self.n_t,
            dt_obs=c.dt_obs,
            c=c.sound_speed,
            field="p0",
            order=c.order if order is None else int(order),
            absorbing=c.absorbing,
            absorb_width=c.absorb_width,
            absorb_R=c.absorb_R,
            courant=c.courant,
            grad_mode=c.grad_mode,
            checkpoint_every=c.checkpoint_every,
        )

    def operator(self, domain: Domain | None = None, order: int | None = None) -> SensorBandwidth:
        """Inversion operator: the wave solver followed by the sensors' bandwidth."""
        c = self.cfg
        return SensorBandwidth(self.wave_operator(domain, order), c.dt_obs, c.sensor_fc)

    def scene_generator(self) -> SceneGenerator:
        c = self.cfg
        return PAT3DScenes(
            self.domain(),
            vessel_radius=tuple(c.vessel_radius),
            sphere_radius=tuple(c.sphere_radius),
            edge_width=c.edge_width,
            depth_range=tuple(c.depth_range),
            render_factor=c.render_factor,
        )

    def data_generator(self) -> DataGenerator:
        c = self.cfg
        fine = self.domain().refine(c.supersample) if c.supersample > 1 else self.domain()
        op = self.operator(fine, order=c.gen_order)
        op.inner.grad_mode = "autograd"
        return DataGenerator(
            op,
            noise_std=c.noise_std,
            relative=True,
            supersample=c.supersample,
            fidelity_tag=f"wave-ic-o{c.gen_order}-{c.supersample}x-float64-lp{c.sensor_fc}",
        )

    def build_problem_measurement_shape(self, field_shape):
        return (int(self.sensors().shape[0]), self.n_t)

    # ---- prior / objective ------------------------------------------------------------------
    def heads(self) -> Heads:
        return Heads({"p0": Softplus(init_value=self.cfg.init_value)})

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
            {"data": MSE(), "tv": TV("p0", isotropic=True, eps=c.tv_eps), "l1": L1("p0")},
            weights={"data": 1.0, "tv": c.tv, "l1": c.l1},
        )

    def default_curriculum(self) -> Curriculum:
        c = self.cfg
        return Curriculum.multiscale(
            tuple(int(n) for n in c.grid),
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
            name="photoacoustic3d",
            meta={"scene": c.scene, "n_sensors": int(self.sensors().shape[0]), "n_t": self.n_t},
        )

    def metrics(self) -> dict[str, Callable]:
        return {"psnr": psnr, "ssim": ssim, "mse": mse}

    # ---- classical reference & baselines ----------------------------------------------------
    def time_reversal(self, measurement: Measurement, mode: str | None = None) -> torch.Tensor:
        """Time-reversal (or adjoint) reconstruction on the native grid, clipped to ``≥ 0``."""
        c = self.cfg
        data = torch.as_tensor(measurement.data)
        rec = time_reversal(
            data,
            self.domain(),
            self.sensors(),
            dt_obs=c.dt_obs,
            c=c.sound_speed,
            mode=mode or c.tr_mode,
            order=c.order,
            absorbing=c.absorbing,
            absorb_width=c.absorb_width,
            absorb_R=c.absorb_R,
            courant=c.courant,
        )
        return rec.clamp_min(0.0).to(data.dtype)

    def baselines(self) -> dict[str, Callable[[Measurement], tuple[InverseProblem, Curriculum]]]:
        c = self.cfg

        def grid(m: Measurement):
            return baseline_problem(self.build_problem(m), "grid", lr=c.grid_lr)

        def tr_baseline(m: Measurement):
            def reconstruct(obs: Measurement, dom: Domain) -> torch.Tensor:
                return self.time_reversal(obs)

            shape = tuple(int(n) for n in c.grid)
            field = GridField(
                shape, Heads({"p0": Identity()}), init=reconstruct(m, None)[..., None]
            )
            cur = Curriculum.single(shape, steps=1, lr=0.0, anneal=False)
            prob = InverseProblem(
                self.domain(),
                field,
                self.operator(),
                self.losses(),
                m,
                curriculum=cur,
                name="photoacoustic3d-time_reversal",
                meta={"solver": "direct", "reconstruct": reconstruct, "baseline": "time_reversal"},
            )
            return prob, cur

        return {"grid": grid, "time_reversal": tr_baseline}


def make_problem(
    seed: int = 0, scene: str | None = None, **cfg: Any
) -> tuple[InverseProblem, dict[str, torch.Tensor], Measurement]:
    """Convenience: ``(problem, gt, measurement)`` for docs and tests."""
    inst = Photoacoustic3D(**({"scene": scene} if scene else {}), **cfg)
    gt, meas = inst.make_measurement(seed)
    return inst.build_problem(meas), gt, meas


def run(cfg: Photoacoustic3DConfig | dict | None = None, seed: int = 0, **kw: Any):
    """End-to-end demo (generate → invert → evaluate); returns a ``RunOutput``."""
    return Photoacoustic3D(cfg).run(seed=seed, **kw)


METRICS: dict[str, Callable] = {"psnr": psnr, "ssim": ssim, "mse": mse}

__all__ = [
    "METRICS",
    "PRESETS",
    "PAT3DScenes",
    "Photoacoustic3D",
    "Photoacoustic3DConfig",
    "SensorBandwidth",
    "gaussian_lowpass",
    "make_problem",
    "run",
]
