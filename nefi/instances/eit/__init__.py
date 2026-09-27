"""eit — electrical impedance tomography: conductivity from boundary voltages.

The continuum-model EIT instance of the elliptic physics family. Trigonometric current patterns
``j_p`` are injected through the boundary of a square body; the task is to recover the conductivity
``σ(x) ∈ [σ_min, σ_max]`` from the boundary potentials (Neumann-to-Dirichlet data), without labels.

Pipeline::

    scene (EITScenes: σ_bg + 1–3 ellipses) ──EITOperator on a 2× finer grid, float64, + noise──►
        V_obs (n_patterns, H, W), mask = boundary strip
    coords ─► annealed PE ─► MLP ─► σ = LogBounded(σ_min, σ_max) ─► EITOperator (harmonic-mean FV,
        pure-Neumann PCG, implicit-function adjoint) ─► masked MSE + isotropic TV
    two-stage curriculum H/2 → H (every stage predicts the native boundary traces)

Quick start::

    from nefi.instances.eit import EIT
    out = EIT(n=16, n_patterns=4, steps=(50, 100)).run(seed=0, device="cpu")
    print(out.metrics)          # psnr, ssim, mse, relative_error, inclusion_iou

See ``docs/instances/eit.md``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ...bench.base import SceneGenerator
from ...domain import Domain
from ...errors import ConfigError
from ...fields import Bounded, GridField, Heads, NeuralField
from ...losses import MSE, TV, LossSet, RelativeMSE
from ...measurement import Measurement
from ...metrics.basic import mse, psnr, relative_error, ssim
from ...physics.elliptic import LogBounded
from ...problem import InverseProblem
from ...registry import register
from ...solve.curriculum import Curriculum, Stage
from .._elliptic_common import SupersampledObservationGenerator
from ..base import Instance
from .operator import EITBoundary, EITOperator, boundary_faces, eit_boundary
from .scenes import EITScenes, inclusion_iou

__all__ = [
    "EIT",
    "EITBoundary",
    "EITConfig",
    "EITOperator",
    "EITScenes",
    "boundary_faces",
    "eit_boundary",
    "inclusion_iou",
    "make_problem",
    "run",
]


@dataclass
class EITConfig:
    """EIT instance configuration (every number used by the pipeline).

    Geometry/physics: ``n`` cells per axis on ``[-half_width, half_width]²``; ``n_patterns``
    trigonometric current patterns of amplitude ``amplitude``; ``electrodes = 0`` for the
    continuum model or ``L`` for the gap model with ``electrode_coverage``.
    """

    # geometry & physics
    n: int = 32
    half_width: float = 1.0
    sigma_bg: float = 1.0
    sigma_min: float = 0.05
    sigma_max: float = 20.0
    n_patterns: int = 8
    amplitude: float = 1.0
    electrodes: int = 0
    electrode_coverage: float = 0.5
    face_mode: str = "harmonic"
    # scenes
    scene: str = "single"
    contrast: tuple[float, float] = (3.0, 6.0)
    high_contrast: tuple[float, float] = (10.0, 20.0)
    radius: tuple[float, float] = (0.2, 0.4)
    center_radius: float = 0.5
    inclusion_gap: float = 0.1
    subsample: int = 4
    # data (inverse-crime guard: same physics on a supersample× finer grid in float64)
    noise_std: float = 1e-3
    supersample: int = 2
    # PDE solver
    tol: float = 1e-6
    max_iter: int | None = None
    precond: str = "jacobi"
    grad_mode: str = "ift"
    warm_start: bool = True
    check_every: int = 1  # CG convergence-check period (host syncs); use 8-16 on CUDA
    # field
    log_param: bool = True
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
    iou_fraction: float = 0.5
    # grid baseline
    grid_lr: float = 5e-2
    grid_tv: float | None = None


@register("instance", "eit")
class EIT(Instance):
    """Electrical impedance tomography (continuum or gap electrode model)."""

    name = "eit"
    Config = EITConfig
    description = "EIT: conductivity from boundary voltages of trigonometric current patterns"

    # --- geometry / physics -------------------------------------------------------------
    def domain(self) -> Domain:
        c = self.cfg
        hw = float(c.half_width)
        return Domain((c.n, c.n), ((-hw, hw), (-hw, hw)), ("x", "y"))

    def operator(self, domain: Domain | None = None, **overrides) -> EITOperator:
        c = self.cfg
        kw = {
            "obs_shape": (c.n, c.n),
            "amplitude": c.amplitude,
            "electrodes": c.electrodes,
            "coverage": c.electrode_coverage,
            "face_mode": c.face_mode,
            "grad_mode": c.grad_mode,
            "tol": c.tol,
            "max_iter": c.max_iter,
            "precond": c.precond,
            "check_every": c.check_every,
            "warm_start": c.warm_start,
        }
        kw.update(overrides)
        return EITOperator(domain or self.domain(), c.n_patterns, **kw)

    def observation_weights(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Boundary length observed per strip cell at grid ``shape`` (mask = ``> 0``)."""
        c = self.cfg
        geo = eit_boundary(
            self.domain().at(shape), 1, c.amplitude, c.electrodes, c.electrode_coverage
        )
        return geo.weights

    def scene_generator(self) -> SceneGenerator:
        c = self.cfg
        return EITScenes(
            self.domain(),
            sigma_bg=c.sigma_bg,
            contrast=c.contrast,
            high_contrast=c.high_contrast,
            radius=c.radius,
            center_radius=c.center_radius,
            gap=c.inclusion_gap,
            subsample=c.subsample,
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
        return (self.cfg.n_patterns, *tuple(field_shape))

    def make_measurement(self, seed=0, scene_class=None, gt=None):
        gt_out, meas = super().make_measurement(seed, scene_class, gt)
        meas.meta.pop("clean", None)
        return gt_out, meas

    def difference_data(self, measurement: Measurement) -> torch.Tensor:
        """Boundary signature of the inclusions (difference EIT): per observed strip cell, the RMS
        over the current patterns of ``V − V_bg`` in % of the RMS boundary voltage, where ``V_bg``
        is the prediction for the homogeneous body ``σ ≡ σ_bg``; ``NaN`` off the observed strip.

        A display quantity (the viewers' compact measurement view, :meth:`measurement_image`): the
        raw patterns are smooth trigonometric profiles around the ring whatever the interior, so
        averaging them shows nothing; the difference to the homogeneous body shows how strongly,
        and where, the inclusions perturb the boundary voltages.
        """
        with torch.no_grad():
            op = self.operator(warm_start=False)
            v_bg = op({"sigma": torch.full(self.domain().shape, self.cfg.sigma_bg).double()})
            data = measurement.data.to(v_bg)
            m = (
                torch.ones_like(data)
                if measurement.mask is None
                else measurement.mask.to(v_bg).expand_as(data)
            )
            rms = torch.sqrt(((data - v_bg) ** 2 * m).sum(0) / m.sum(0).clamp_min(1.0))
            ref = torch.sqrt((v_bg**2 * m).sum() / m.sum().clamp_min(1.0)).clamp_min(1e-30)
            out = 100.0 * rms / ref
            return torch.where(m.amax(0) > 0, out, torch.full_like(out, float("nan"))).float()

    def measurement_image(self, measurement: Measurement) -> tuple[torch.Tensor, str]:
        """Compact measurement view for the viewers (:mod:`nefi.viz`): :meth:`difference_data`."""
        n = int(measurement.data.shape[0])
        return self.difference_data(measurement), f"|V − V_bg| %, {n} patterns"

    # --- inversion ----------------------------------------------------------------------
    def heads(self) -> Heads:
        c = self.cfg
        if c.log_param:
            head = LogBounded(c.sigma_min, c.sigma_max, init_value=c.sigma_bg)
        else:
            head = Bounded(c.sigma_min, c.sigma_max, init_value=c.sigma_bg)
        return Heads({"sigma": head})

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
            {"data": data(), "tv": TV("sigma", isotropic=True, eps=c.tv_eps)},
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
        # the operator predicts the native observation grid at every curriculum resolution, so
        # the measurement is used as is (no observation downsampling)
        return InverseProblem(
            self.domain(),
            self.field(),
            self.operator(),
            self.losses(),
            measurement,
            curriculum=self.default_curriculum(),
            name="eit",
        )

    def metrics(self) -> dict[str, Callable]:
        c = self.cfg
        return {
            "psnr": psnr,
            "ssim": ssim,
            "mse": mse,
            "relative_error": relative_error,
            "inclusion_iou": functools.partial(
                inclusion_iou, sigma_bg=c.sigma_bg, fraction=c.iou_fraction
            ),
        }

    def baselines(self):
        def grid(measurement: Measurement):
            c = self.cfg
            field = GridField((c.n, c.n), self.heads())
            problem = InverseProblem(
                self.domain(),
                field,
                self.operator(),
                self.losses(tv=c.grid_tv if c.grid_tv is not None else c.tv),
                measurement,
                name="eit-grid",
            )
            cur = self._curriculum(c.grid_lr)
            problem.curriculum = cur
            return problem, cur

        return {"grid": grid}


def make_problem(seed: int = 0, scene_class: str | None = None, **cfg):
    """Convenience: ``(problem, gt, measurement)`` for docs, tests and notebooks."""
    inst = EIT(**cfg)
    gt, meas = inst.make_measurement(seed, scene_class)
    return inst.build_problem(meas), gt, meas


def run(cfg: EITConfig | dict | None = None, seed: int = 0, **kw):
    """End-to-end demo: generate → invert → evaluate. Returns ``(Result, metrics)``."""
    out = EIT(cfg).run(seed=seed, **kw)
    return out.result, out.metrics
