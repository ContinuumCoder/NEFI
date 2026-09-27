"""thermal_tomography — NeFTY: 3-D inverse heat conduction from pulsed-thermography surface frames.

Recovers the volumetric diffusivity ``α(x, y, z)`` of a slab from the transient front-surface
temperature after a laser flash (NeFTY, arXiv 2603.11045):

* **field** — coordinate MLP (10 × 512, ReLU, skip at 4) on annealed Fourier features (K = 12)
  with the bounded head ``α = α_min + (α_max − α_min) σ(f)`` (Eq. 6, App. D.1);
* **operator** — :class:`~nefi.operators.pde.HeatOperator`: harmonic-mean finite volumes (Prop. 1),
  implicit Euler (Eq. 8), 50 warm-started Jacobi sweeps (Eq. 24), discrete adjoint (Eq. 10–11);
* **losses** — surface MSE + isotropic TV (Eq. 22), λ = 1e-3, ε = 1e-6;
* **data** — :class:`ThermalScenes` (App. E.1) simulated by the independent explicit, substepped,
  float64 :class:`~nefi.operators.pde.ExplicitHeatSimulator` (inverse-crime guard);
* **metrics** — MSE / PSNR (range α_max − α_min) / slice-wise SSIM / IoU at τ = 0.03 / Edge F1
  (App. E.2, G.5), plus the surface-fit PSNR of the re-simulated recovered field (data-fit paradox,
  App. G.2) and the App. H 2-D / 2.5-D projections.

Every number of NeFTY Tab. 5 is a field of :class:`ThermalTomographyConfig`; ``preset="smoke"`` is a
CPU-sized variant (16×16×6 grid, 30 frames, 150 steps, ≈ 4 s on a single CPU core).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import torch

from ...bench.base import DataGenerator
from ...domain import Domain
from ...errors import ConfigError
from ...fields import Bounded, GridField, Heads, Identity, NeuralField
from ...losses import MSE, TV, LossSet
from ...measurement import Measurement
from ...metrics.basic import mse, psnr, ssim
from ...metrics.segmentation import (
    abs_rel,
    delta_threshold,
    depth_rmse,
    edge_f1,
    iou_below,
)
from ...operators.pde import ExplicitHeatSimulator, GaussianFlash, HeatOperator, UniformFlash
from ...problem import InverseProblem
from ...registry import register
from ...solve.curriculum import Curriculum, OptimConfig, Stage
from ...solve.result import Result
from ...solve.solver import Solver
from ...utils.seed import seed_everything
from ...utils.tensor import resample
from ..base import Instance, RunOutput
from .pinn import (
    InitialConditionLoss,
    NormalizedSurfaceMSE,
    PDEResidualLoss,
    SoftPINNSurface,
    TemperatureNet,
)
from .projection import bulk_profile, defect_mask_2d, depth_map_25d, gt_depth_map
from .scenes import DEFECT_SHAPES, DefectSpec, SceneSpec, ThermalScenes

__all__ = [
    "PRESETS",
    "DefectSpec",
    "SceneSpec",
    "ThermalScenes",
    "ThermalTomography",
    "ThermalTomographyConfig",
    "bulk_profile",
    "defect_mask_2d",
    "depth_map_25d",
    "gt_depth_map",
    "surface_downsample",
]


@dataclass
class ThermalTomographyConfig:
    """NeFTY configuration; defaults reproduce NeFTY Tab. 5 / App. A.5 / E.1 (see the docs page
    ``docs/instances/thermal_tomography.md`` for the field-by-field mapping)."""

    # --- domain and grid (Tab. 5: 10 × 10 × 1, 64 × 64 × 16 → Δx = Δy = 0.156, Δz = 0.0625) ---
    extent: tuple[float, ...] = (10.0, 10.0, 1.0)
    grid: tuple[int, ...] = (64, 64, 16)
    # --- forward solver (Tab. 5, App. D.2) ---
    dt: float = 0.05
    n_frames: int = 100
    first_frame: int = 1  # first observed step (1 = the frame after the flash)
    frame_stride: int = 1
    solver: str = "jacobi"
    jacobi_iters: int = 50
    cg_tol: float = 1e-6
    cg_max_iter: int = 200
    implicit_substeps: int = 1
    face_mode: str = "harmonic"
    grad_mode: str = "adjoint"
    adjoint_assembly: str = "fused"
    compile_solver: bool = False  # torch.compile the Jacobi sweeps (CUDA servers; App. D.2)
    compile_mode: str | None = None
    stencil_backend: str = "auto"  # auto | flat | roll (implementation detail, same numbers)
    alpha_min: float = 0.003
    alpha_max: float = 0.25
    # --- boundary conditions (App. A.5) ---
    bc_lateral: str = "periodic"  # periodic | neumann
    bc_back: str = "neumann"  # neumann (adiabatic, synthetic) | robin (PVC)
    robin_h: float = 0.0
    # --- initial condition (App. A.5; amplitude / widths are not given in the paper) ---
    initial: str = "gaussian"  # gaussian (synthetic) | flash (PVC near-uniform)
    flash_amplitude: float = 100.0
    flash_width_xy: float = 2.5
    flash_width_z: float = 0.2  # 0.1 triples the implicit-vs-explicit model error (see docs)
    flash_center: tuple[float, ...] | None = None
    # --- scenes (App. E.1) ---
    scene: str = "homogeneous"  # homogeneous | layered
    alpha_base_range: tuple[float, float] = (0.1, 0.2)
    alpha_defect_range: tuple[float, float] = (0.005, 0.015)
    n_defects: tuple[int, int] = (1, 4)
    n_layers: tuple[int, int] = (3, 4)
    defect_shapes: tuple[str, ...] = DEFECT_SHAPES
    defect_radius_range: tuple[float, float] = (0.6, 1.6)
    defect_half_thickness_range: tuple[float, float] = (0.08, 0.2)
    defect_depth_range: tuple[float, float] = (0.25, 0.75)
    lateral_margin: float = 1.0
    min_layer_fraction: float = 0.15
    # --- data generation (App. E.1: explicit, substepped, independent) ---
    sim_face_mode: str = "harmonic"
    sim_min_substeps: int = 10
    sim_substep_safety: float = 2.0
    sim_supersample: int = 1
    noise_std: float = 0.0  # relative to max|T| (the paper's synthetic data are noise-free)
    # --- neural field (Tab. 5, App. D.1) ---
    hidden: int = 512
    depth: int = 10
    skip_at: int = 4
    activation: str = "relu"
    n_octaves: int = 12
    include_input: bool = False  # Eq. (20): γ(x) has no raw-coordinate channel
    out_init_scale: float = 0.1
    alpha_init: float = 0.15  # initial uniform field (bulk mean)
    # --- optimization (Tab. 5) ---
    steps: int = 10000
    lr: float = 5e-5
    lr_schedule: str = "step"
    lr_step_size: int = 1000
    lr_gamma: float = 0.1
    lr_min_ratio: float = 0.01
    anneal_steps: int = 2500
    optimizer: str = "adam"
    weight_decay: float = 0.0
    grad_clip: float | None = None
    n_stages: int = 1  # the paper is single-stage; >1 = coarse-to-fine multiscale
    stage_steps: tuple[int, ...] | None = None
    stage_lr_decay: float = 1.0
    # --- regularization (Tab. 5, Eq. 22) ---
    tv_weight: float = 1e-3
    tv_eps: float = 1e-6
    # --- evaluation (App. E.2, G.5, H.3) ---
    iou_tau: float = 0.03
    edge_threshold: float = 0.5
    edge_dilate: int = 1
    projection_k: float = 2.0
    # --- Grid Opt. baseline (App. F.2: same schedule; None = same as NeFTY) ---
    grid_lr: float | None = None
    grid_steps: int | None = None
    # --- soft-constrained PINN baseline (Eq. 4, App. F.1; fixed weights instead of GradNorm) ---
    pinn_hidden: int = 128
    pinn_depth: int = 5
    pinn_n_octaves: int = 0
    pinn_collocation: int = 24576
    pinn_lambda_pde: float = 1.0
    pinn_lambda_ic: float = 1.0
    pinn_lr: float = 1e-3
    pinn_steps: int = 22000


#: Named presets (partial configs applied before user overrides).
PRESETS: dict[str, dict[str, Any]] = {
    "paper": {},
    "layered": {"scene": "layered"},
    # CPU smoke test: ≈ 4 s inversion on a single CPU thread.
    "smoke": {
        "grid": (16, 16, 6),
        "n_frames": 30,
        "dt": 0.1,
        "jacobi_iters": 20,  # contraction ≈ 0.55 per sweep at this Δt / spacing (vs ≈ 0.85)
        "flash_width_xy": 4.0,
        "n_defects": (1, 2),
        "defect_radius_range": (1.6, 2.4),
        "defect_half_thickness_range": (0.15, 0.25),
        "defect_depth_range": (0.3, 0.6),
        "lateral_margin": 2.5,
        "hidden": 64,
        "depth": 4,
        "skip_at": 2,
        "n_octaves": 4,
        "steps": 150,
        "lr": 3e-3,
        "lr_schedule": "cosine",
        "lr_min_ratio": 0.1,
        "anneal_steps": 60,
        "grid_lr": 3e-2,
        "pinn_hidden": 64,
        "pinn_depth": 4,
        "pinn_collocation": 1024,
        "pinn_steps": 150,
    },
}


def surface_downsample(measurement: Measurement, field_shape: Sequence[int]) -> Measurement:
    """Measurement at a coarse curriculum stage: surface frames area-averaged laterally.

    The frame axis is kept; only the lateral axes are resampled to ``field_shape[:-1]``.
    """
    lateral = tuple(int(s) for s in field_shape[:-1])
    if tuple(measurement.data.shape[1:]) == lateral:
        return measurement
    data = resample(measurement.data, lateral)
    mask = None
    if measurement.mask is not None:
        m = measurement.mask.expand_as(measurement.data)
        mask = (resample(m, lateral) > 0.5).to(data.dtype)
    return Measurement(data, mask, measurement.noise_std, dict(measurement.meta))


@register("instance", "thermal_tomography")
class ThermalTomography(Instance):
    """NeFTY thermal tomography (pulsed thermography, 3-D inverse heat conduction).

    Args:
        cfg: :class:`ThermalTomographyConfig`, a dict, or ``None``.
        preset: optional preset name from :data:`PRESETS` (``"paper"``, ``"layered"``,
            ``"smoke"``) applied before ``cfg`` and ``overrides``.
        **overrides: individual config fields.
    """

    name = "thermal_tomography"
    Config = ThermalTomographyConfig
    description = "NeFTY: 3-D diffusivity from pulsed-thermography surface frames (heat adjoint)"

    def __init__(self, cfg: Any = None, preset: str | None = None, **overrides: Any) -> None:
        if isinstance(cfg, Mapping) and "preset" in cfg:
            cfg = dict(cfg)
            preset = cfg.pop("preset") or preset
        if preset is not None:
            if preset not in PRESETS:
                raise ConfigError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}")
            if dataclasses.is_dataclass(cfg):
                cfg = dataclasses.asdict(cfg)
            cfg = {**PRESETS[preset], **dict(cfg or {})}
        super().__init__(cfg, **overrides)
        self.preset = preset
        self._validate()

    def _validate(self) -> None:
        c = self.cfg
        if len(c.extent) != len(c.grid) or len(c.grid) < 2:
            raise ConfigError("extent and grid must have the same length (>= 2)")
        if not 0 < c.alpha_min < c.alpha_max:
            raise ConfigError("need 0 < alpha_min < alpha_max")
        if not c.alpha_min < c.alpha_init < c.alpha_max:
            raise ConfigError("alpha_init must lie strictly inside (alpha_min, alpha_max)")
        if not 1 <= c.first_frame <= c.n_frames:
            raise ConfigError("first_frame must be in [1, n_frames]")

    # ---- building blocks ---------------------------------------------------------------
    def domain(self) -> Domain:
        c = self.cfg
        axes = {2: ("x", "z"), 3: ("x", "y", "z")}.get(len(c.grid))
        return Domain(tuple(c.grid), tuple((0.0, float(L)) for L in c.extent), axes)

    def initial_condition(self):
        """Post-flash temperature (App. A.5): Gaussian (synthetic) or near-uniform flash (PVC)."""
        c = self.cfg
        if c.initial in ("gaussian", "gaussian_flash"):
            return GaussianFlash(
                c.flash_amplitude,
                c.flash_width_xy,
                c.flash_width_z,
                c.flash_center,
                periodic=c.bc_lateral == "periodic",
            )
        if c.initial in ("flash", "uniform", "uniform_flash"):
            return UniformFlash(c.flash_amplitude, c.flash_width_z)
        raise ConfigError(f"unknown initial condition {c.initial!r}; use 'gaussian' or 'flash'")

    def boundary_conditions(self) -> tuple[str, ...]:
        c = self.cfg
        if c.bc_lateral not in ("periodic", "neumann"):
            raise ConfigError("bc_lateral must be 'periodic' or 'neumann'")
        if c.bc_back not in ("neumann", "robin"):
            raise ConfigError("bc_back must be 'neumann' or 'robin'")
        return (c.bc_lateral,) * (len(c.grid) - 1) + (c.bc_back,)

    def obs_frames(self) -> tuple[int, ...]:
        c = self.cfg
        return tuple(range(c.first_frame, c.n_frames + 1, max(1, c.frame_stride)))

    def operator(self, domain: Domain | None = None, **overrides: Any) -> HeatOperator:
        """The inversion operator (implicit Euler + Jacobi/CG + discrete adjoint)."""
        c = self.cfg
        kw = dict(
            dt=c.dt,
            n_steps=c.n_frames,
            obs_frames=self.obs_frames(),
            initial=self.initial_condition(),
            bc=self.boundary_conditions(),
            robin_h=c.robin_h,
            solver=c.solver,
            inner_iters=c.jacobi_iters,
            cg_tol=c.cg_tol,
            cg_max_iter=c.cg_max_iter,
            grad_mode=c.grad_mode,
            face_mode=c.face_mode,
            adjoint_assembly=c.adjoint_assembly,
            substeps=c.implicit_substeps,
            compile=c.compile_solver,
            compile_mode=c.compile_mode,
            stencil_backend=c.stencil_backend,
        )
        kw.update(overrides)
        return HeatOperator(domain or self.domain(), **kw)

    def simulator(self, domain: Domain | None = None) -> ExplicitHeatSimulator:
        """The independent data simulator (explicit, substepped, float64; App. E.1)."""
        c = self.cfg
        return ExplicitHeatSimulator(
            domain or self.domain(),
            dt=c.dt,
            n_steps=c.n_frames,
            obs_frames=self.obs_frames(),
            initial=self.initial_condition(),
            bc=self.boundary_conditions(),
            robin_h=c.robin_h,
            face_mode=c.sim_face_mode,
            min_substeps=c.sim_min_substeps,
            substep_safety=c.sim_substep_safety,
        )

    def scene_generator(self) -> ThermalScenes:
        c = self.cfg
        return ThermalScenes(
            self.domain(),
            alpha_base_range=c.alpha_base_range,
            alpha_defect_range=c.alpha_defect_range,
            n_defects=c.n_defects,
            n_layers=c.n_layers,
            shapes=c.defect_shapes,
            radius_range=c.defect_radius_range,
            half_thickness_range=c.defect_half_thickness_range,
            depth_range=c.defect_depth_range,
            lateral_margin=c.lateral_margin,
            min_layer_fraction=c.min_layer_fraction,
            periodic=c.bc_lateral == "periodic",
        )

    def data_generator(self) -> DataGenerator:
        c = self.cfg
        sim = self.simulator()
        return DataGenerator(
            sim,
            noise_std=c.noise_std,
            relative=True,
            dtype=torch.float64,
            supersample=c.sim_supersample,
            fidelity_tag=sim.fidelity_tag,
        )

    def build_problem_measurement_shape(self, field_shape: Sequence[int]) -> tuple[int, ...]:
        return (len(self.obs_frames()), *tuple(field_shape)[:-1])

    def heads(self) -> Heads:
        c = self.cfg
        return Heads({"alpha": Bounded(c.alpha_min, c.alpha_max, init_value=c.alpha_init)})

    def field(self) -> NeuralField:
        """NeFTY neural diffusivity field (App. D.1, Tab. 5)."""
        c = self.cfg
        return NeuralField(
            len(c.grid),
            self.heads(),
            hidden=c.hidden,
            depth=c.depth,
            skip_at=c.skip_at,
            activation=c.activation,
            n_octaves=c.n_octaves,
            annealed=True,
            include_input=c.include_input,
            out_init_scale=c.out_init_scale,
        )

    def losses(self) -> LossSet:
        """Surface MSE + isotropic TV (Eq. 22: lateral periodic differences, zero δz on the last
        slice, physical spacing)."""
        c = self.cfg
        periodic = tuple(range(len(c.grid) - 1)) if c.bc_lateral == "periodic" else ()
        return LossSet(
            {
                "data": MSE(),
                "tv": TV("alpha", isotropic=True, eps=c.tv_eps, periodic_axes=periodic),
            },
            weights={"data": 1.0, "tv": c.tv_weight},
        )

    def _problem(self, measurement: Measurement, field, name: str, curriculum) -> InverseProblem:
        return InverseProblem(
            self.domain(),
            field,
            self.operator(),
            self.losses(),
            measurement,
            curriculum=curriculum,
            downsample_obs=surface_downsample,
            name=name,
            meta={"instance": self.name, "preset": self.preset},
        )

    def build_problem(self, measurement: Measurement) -> InverseProblem:
        return self._problem(
            measurement, self.field(), "thermal_tomography", self.default_curriculum()
        )

    # ---- curriculum ---------------------------------------------------------------------
    def _stage(self, name: str, shape, steps: int, lr: float) -> Stage:
        c = self.cfg
        return Stage(
            name,
            tuple(shape),
            int(steps),
            float(lr),
            lr_schedule=c.lr_schedule,
            lr_min_ratio=c.lr_min_ratio,
            lr_step_size=c.lr_step_size,
            lr_gamma=c.lr_gamma,
            anneal=True,
            anneal_fraction=min(1.0, max(1e-6, c.anneal_steps / max(1, steps))),
        )

    def _curriculum(self, steps: int, lr: float) -> Curriculum:
        c = self.cfg
        n = max(1, int(c.n_stages))
        if n == 1:
            stages = [self._stage("main", c.grid, steps, lr)]
        else:
            per = list(c.stage_steps) if c.stage_steps else [steps // n] * n
            if len(per) != n:
                raise ConfigError("stage_steps must have one entry per stage")
            stages = []
            for i in range(n):
                f = 2 ** (n - 1 - i)
                shape = tuple(max(4, s // f) for s in c.grid)
                stages.append(self._stage(f"stage{i + 1}", shape, per[i], lr * c.stage_lr_decay**i))
        optim = OptimConfig(
            optimizer=c.optimizer, weight_decay=c.weight_decay, grad_clip=c.grad_clip
        )
        return Curriculum(stages, optim)

    def default_curriculum(self) -> Curriculum:
        """Single stage (paper): Adam, step decay ×0.1 / 1000, annealing over the first
        ``anneal_steps`` (Tab. 5); ``n_stages > 1`` gives a coarse-to-fine curriculum."""
        return self._curriculum(self.cfg.steps, self.cfg.lr)

    # ---- baselines ----------------------------------------------------------------------
    def baselines(self) -> dict[str, Callable[[Measurement], tuple[InverseProblem, Curriculum]]]:
        """Baselines on the same measurement.

        * ``"grid"`` — Grid Opt. (App. F.2): free voxel grid with the same bounded head, solver,
          adjoint, TV and schedule, initialized at the bulk guess.
        * ``"pinn_soft"`` — soft-constrained PINN (Eq. 4, App. F.1): temperature surrogate
          ``T_φ(x, t)`` + the NeFTY diffusivity field, surface data + PDE residual + initial
          condition penalties (fixed weights). ``∇_θ L_data ≡ 0`` (§3.3 decoupling).
        * ``"ppt"`` / ``"tsr"`` — the classical pixel-wise thermography heuristics (App. F.4),
          :mod:`nefi.baselines.thermography`, lifted to a volume through
          :func:`~nefi.baselines.thermography.depth_to_alpha_volume` (closed form, one call).
        """

        def grid(measurement: Measurement) -> tuple[InverseProblem, Curriculum]:
            c = self.cfg
            field_ = GridField(tuple(c.grid), self.heads())
            cur = self._curriculum(c.grid_steps or c.steps, c.grid_lr or c.lr)
            return self._problem(measurement, field_, "thermal_tomography-grid", cur), cur

        def pinn_soft(measurement: Measurement) -> tuple[InverseProblem, Curriculum]:
            c = self.cfg
            dom = self.domain()
            tnet = TemperatureNet(
                len(c.grid), c.pinn_hidden, c.pinn_depth, c.pinn_n_octaves, c.flash_amplitude
            )
            times = [n * c.dt for n in self.obs_frames()]
            op = SoftPINNSurface(dom, times, c.n_frames * c.dt, tnet)
            losses = LossSet(
                {
                    "data": NormalizedSurfaceMSE(c.flash_amplitude),
                    "pde": PDEResidualLoss(c.pinn_collocation),
                    "ic": InitialConditionLoss(self.initial_condition()),
                },
                weights={"data": 1.0, "pde": c.pinn_lambda_pde, "ic": c.pinn_lambda_ic},
            )
            stage = Stage(
                "pinn",
                tuple(c.grid),
                c.pinn_steps,
                c.pinn_lr,
                lr_schedule="constant",
                anneal=True,
                anneal_fraction=min(1.0, max(1e-6, c.anneal_steps / max(1, c.steps))),
            )
            cur = Curriculum(
                [stage], OptimConfig(optimizer="adam", weight_decay=0.0, grad_clip=None)
            )
            prob = InverseProblem(
                dom,
                self.field(),
                op,
                losses,
                measurement,
                curriculum=cur,
                downsample_obs=surface_downsample,
                name="thermal_tomography-pinn_soft",
                meta={"instance": self.name, "preset": self.preset},
            )
            return prob, cur

        def thermography(kind: str):
            """PPT / TSR heuristics (App. F.4) lifted to a volume, as closed-form baselines."""
            from ...baselines.thermography import depth_to_alpha_volume, ppt, tsr

            def build(measurement: Measurement) -> tuple[InverseProblem, Curriculum]:
                c = self.cfg
                grid = tuple(c.grid)
                dt_obs = c.dt * max(1, c.frame_stride)
                t_first = c.first_frame * c.dt

                def reconstruct(obs: Measurement, dom: Domain) -> torch.Tensor:
                    frames = obs.data
                    if kind == "ppt":
                        out = ppt(frames, dt_obs, c.alpha_init)
                    else:
                        out = tsr(frames, dt_obs, c.alpha_init, t0=t_first)
                    vol = depth_to_alpha_volume(
                        out["depth"],
                        out["mask"],
                        dom.shape[2],
                        c.extent[2],
                        c.alpha_init,
                        c.alpha_min,
                    )
                    return vol.to(frames.device, torch.float32)

                x0 = reconstruct(measurement, self.domain())
                field_ = GridField(grid, Heads({"alpha": Identity()}), init=x0[..., None])
                cur = Curriculum.single(grid, steps=1, lr=0.0, anneal=False)
                prob = self._problem(measurement, field_, f"thermal_tomography-{kind}", cur)
                prob.meta.update({"solver": "direct", "reconstruct": reconstruct, "baseline": kind})
                return prob, cur

            return build

        return {
            "grid": grid,
            "pinn_soft": pinn_soft,
            "ppt": thermography("ppt"),
            "tsr": thermography("tsr"),
        }

    # ---- data -----------------------------------------------------------------------------
    def make_measurement(
        self,
        seed: int = 0,
        scene_class: str | None = None,
        gt: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], Measurement]:
        """Draw a scene, simulate it with the explicit float64 simulator (optionally on a
        ``sim_supersample``× finer grid), add noise. Scene metadata (bulk field, defect mask,
        parameters) is stored in ``measurement.meta``."""
        rng = np.random.default_rng(seed)
        scene_class = scene_class or self.cfg.scene
        scenes = self.scene_generator()
        gen = self.data_generator()
        dom = self.domain()
        meta: dict[str, Any] = {}
        if gt is None:
            spec = scenes.draw(rng, scene_class)
            gt_fields, meta = scenes.rasterize(spec)
            if gen.supersample > 1:
                fine = tuple(s * gen.supersample for s in dom.shape)
                sim_fields = scenes.rasterize(spec, fine)[0]
            else:
                sim_fields = gt_fields
        else:
            gt_fields = {k: torch.as_tensor(v) for k, v in gt.items()}
            sim_fields = gt_fields
        target = self.build_problem_measurement_shape(dom.shape)
        meas = gen.generate(sim_fields, rng, target_shape=target)
        meas.meta.update(
            {
                "scene_class": scene_class,
                "seed": seed,
                "frame_times": [n * self.cfg.dt for n in self.obs_frames()],
            }
        )
        if meta:
            meas.meta.update(
                {
                    "alpha_base": meta["alpha_base"],
                    "defect_mask": meta["defect_mask"],
                    "scene": meta["spec"].to_dict(),
                }
            )
        return {k: torch.as_tensor(v) for k, v in gt_fields.items()}, meas

    # ---- metrics ------------------------------------------------------------------------
    def metrics(self) -> dict[str, Callable]:
        """Volumetric metrics of NeFTY App. E.2 / G.5 on the diffusivity field."""
        c = self.cfg
        rng_ = c.alpha_max - c.alpha_min
        return {
            "mse": mse,
            "psnr": partial(psnr, data_range=rng_),
            "ssim": partial(ssim, data_range=rng_),
            "iou": partial(iou_below, tau=c.iou_tau),
            "edge_f1": partial(edge_f1, threshold=c.edge_threshold, dilate=c.edge_dilate),
        }

    @torch.no_grad()
    def surface_metrics(
        self, pred: torch.Tensor, measurement: Measurement, reference: torch.Tensor | None = None
    ) -> dict[str, float]:
        """Surface-fit MSE / PSNR of re-simulated frames against the observed frames (App. G.2);
        PSNR uses the observed data range. ``reference`` overrides the comparison target."""
        obs = (measurement.data if reference is None else reference).detach().double().cpu()
        p = pred.detach().double().cpu()
        if p.shape != obs.shape:
            p = resample(p, obs.shape[1:])
        m = float(((p - obs) ** 2).mean())
        rng_ = float(obs.max() - obs.min()) or 1.0
        return {"surface_mse": m, "surface_psnr": 10.0 * math.log10(rng_**2 / max(m, 1e-300))}

    @torch.no_grad()
    def uniform_prediction(self, value: float | None = None, dtype=torch.float64) -> torch.Tensor:
        """Frames predicted by the inversion operator for a uniform field (the initial iterate)."""
        c = self.cfg
        a = torch.full(tuple(c.grid), float(c.alpha_init if value is None else value), dtype=dtype)
        return self.operator().to(dtype)({"alpha": a})

    def projection_metrics(
        self, alpha: torch.Tensor, gt: torch.Tensor, gt_mask: torch.Tensor | None = None
    ) -> dict[str, float]:
        """App. H projections evaluated against the ground truth: 2-D footprint IoU / Dice and
        2.5-D depth AbsRel / RMSE / δ<1.25 on the correctly detected pixels."""
        c = self.cfg
        gt_def = gt_mask if gt_mask is not None else (gt < c.iou_tau)
        gt2d = gt_def.bool().any(-1)
        pred2d = defect_mask_2d(alpha, None, k=c.projection_k)
        inter, union = float((pred2d & gt2d).sum()), float((pred2d | gt2d).sum())
        denom = float(pred2d.sum() + gt2d.sum())
        out = {
            "iou_2d": inter / union if union > 0 else 1.0,
            "dice_2d": 2 * inter / denom if denom > 0 else 1.0,
        }
        H = float(c.extent[-1])
        d_pred = depth_map_25d(alpha, None, pred2d, thickness=H, k=c.projection_k)
        d_gt = gt_depth_map(gt_def, thickness=H)
        both = pred2d & gt2d & torch.isfinite(d_pred) & torch.isfinite(d_gt)
        if bool(both.any()):
            out.update(
                {
                    "depth_abs_rel": abs_rel(d_pred, d_gt, both),
                    "depth_rmse": depth_rmse(d_pred, d_gt, both),
                    "depth_delta1": delta_threshold(d_pred, d_gt, 1, both),
                }
            )
        return out

    def evaluate(
        self,
        result: Result,
        gt: Mapping[str, torch.Tensor],
        measurement: Measurement | None = None,
    ) -> dict[str, float]:
        """Volumetric metrics, App. H projections and — given the measurement — the surface-fit
        PSNR of the recovered field's re-simulation next to that of the initial uniform field
        (the data-fit-paradox diagnostic, NeFTY §5.2 / App. G.2)."""
        a = result.fields["alpha"].detach().cpu().float()
        g = torch.as_tensor(gt["alpha"]).detach().cpu().float()
        if a.shape != g.shape:
            a = resample(a, g.shape)
        out = {k: float(fn(a, g)) for k, fn in self.metrics().items()}
        gt_mask = None
        if measurement is not None and "defect_mask" in measurement.meta:
            gt_mask = torch.as_tensor(measurement.meta["defect_mask"])
            if gt_mask.shape != g.shape:
                gt_mask = None
        out.update(self.projection_metrics(a, g, gt_mask))
        if measurement is not None:
            out.update(self.surface_metrics(result.pred, measurement))
            init = self.surface_metrics(self.uniform_prediction(), measurement)
            out["surface_psnr_init"] = init["surface_psnr"]
            out["surface_mse_init"] = init["surface_mse"]
        return out

    def run(
        self,
        seed: int = 0,
        device: str = "auto",
        scene_class: str | None = None,
        curriculum: Curriculum | None = None,
        callbacks: Sequence = (),
        **solver_kw: Any,
    ) -> RunOutput:
        """Generate → invert → evaluate (with the surface-fit diagnostics).

        The field is initialized after ``seed_everything(seed)``, so a run is reproducible.
        """
        gt, meas = self.make_measurement(seed, scene_class)
        seed_everything(seed)
        problem = self.build_problem(meas)
        cur = curriculum or self.default_curriculum()
        result = Solver(
            problem, cur, device=device, seed=seed, callbacks=callbacks, **solver_kw
        ).run()
        metrics = self.evaluate(result, gt, meas)
        extra = {"scene_class": scene_class or self.cfg.scene, "seed": seed}
        return RunOutput(result, metrics, gt, meas, extra)
