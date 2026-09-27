"""dot3d — continuous-wave diffuse optical tomography: absorption from surface reflectance.

Recover the absorption coefficient ``μ_a(x, y, z)`` (mm⁻¹) of a scattering slab (tissue-like
``μ_s' = 1 mm⁻¹``) from the diffuse reflectance measured by a camera-like grid of detector pixels on
the **top face**, for several point sources on the same face — the surface-to-volume elliptic
inverse problem, structurally NeFTY's twin (surface data, a smoothing forward map, depth-decaying
sensitivity; see ``nefi.diagnostics.sensitivity_map`` in ``docs/instances/dot3d.md``).

* **operator** — :class:`~nefi.instances.dot3d.operator.DiffuseOpticalOperator`: the diffusion
  equation ``−∇·(D∇Φ) + μ_a Φ = S`` with exact finite-volume Robin (partial-current) boundaries,
  built on :class:`~nefi.physics.elliptic.EllipticOperator` (Jacobi-PCG, implicit-function adjoint,
  warm starts); the unknown is the zeroth-order coefficient. Output ``(n_sources, n_dx, n_dy)``
  (resolution independent), calibrated by the homogeneous-reference readings (``calibrate``);
* **field** — coordinate ReLU MLP (as NeFTY's volumetric field) on annealed 3-D Fourier features
  with a ``Bounded(μ_min, μ_max)`` head (NeFTY Eq. 6) or its log-uniform variant ``LogBounded``
  (``log_param=True``), started at the known background; ``μ_min`` just below the background
  makes the sigmoid's flat lower tail an *absorbers-only* prior (inclusions add absorption);
* **losses** — masked relative-residual MSE (source–detector pairs closer than
  ``min_separation`` are excluded) + isotropic 3-D TV (NeFTY Eq. 22);
* **curriculum** — two-stage multiscale (half resolution in every axis, then native) against the
  same resolution-independent measurement;
* **data** — :class:`DOT3DScenes` (``single``, ``multi``, ``deep`` inclusions) simulated on a 2×
  finer grid in float64 (``fidelity_tag="dot-fv-robin-2x-float64"``, inverse-crime guard) with 1 %
  multiplicative noise;
* **metrics** — PSNR / slice-averaged SSIM / MSE on ``μ_a``; inclusion IoU of the half-maximum
  masks of the absorption excess (contrast independent), the lateral footprint IoU (``iou_2d``),
  the depth error of the inclusion centroid (``depth_error``, mm), the RMSE of the 2.5-D
  excess-weighted depth maps (``depth_rmse``, mm; NeFTY App. H-style) and the contrast recovery
  (``contrast``);
* **baselines** — ``grid`` (free voxels + the same objective) and ``deep_decoder``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from ...baselines import baseline_problem
from ...bench.base import SceneGenerator
from ...domain import Domain
from ...errors import ConfigError
from ...fields import Bounded, Heads, NeuralField
from ...losses import TV, LossSet
from ...measurement import Measurement
from ...metrics.basic import mse, psnr, ssim
from ...physics.elliptic import LogBounded
from ...problem import InverseProblem
from ...registry import register
from ...solve.curriculum import Curriculum
from ...solve.result import Result
from ...utils.tensor import resample
from .._volumetric import PresetInstance, depth_centroid, depth_map, excess
from .data import DOTDataGenerator, RelativeResidualMSE, separation_mask
from .operator import (
    DiffuseOpticalOperator,
    boundary_conductance,
    detector_layout,
    grid_positions,
    robin_A,
    robin_sink,
)
from .scenes import DOT3DScenes, Inclusion


@dataclass
class DOT3DConfig:
    """Configuration of the :class:`DOT3D` instance (every number is a field; lengths in mm,
    coefficients in mm⁻¹).

    Physics: slab ``extent`` on ``grid`` voxels (last axis = depth), reduced scattering ``musp``
    (``D = 1/(3 μ_s')``), refractive index ``refractive_index`` (Robin factor ``A``), background
    absorption ``mua_background``; ``source_grid`` sources over the central ``source_span`` of the
    top face at depth ``1/μ_s'``; ``detector_grid`` square detector pixels over ``detector_span``
    (``detector_samples``² sub-samples each); pairs closer than ``min_separation`` are unobserved;
    ``calibrate`` divides the readings by those of the homogeneous background slab (reference-
    phantom normalization); relative noise ``noise_std``; data on a ``supersample``× grid.
    Solver: ``tol``, ``max_iter``, ``check_every``, ``warm_start``. Scenes: ``scene`` ∈ {single,
    multi, deep}, ``inclusion_mua``, ``inclusion_radius``, ``depth_*``, ``lateral_span``,
    ``render_factor``.
    Prior / solver: ``mua_min`` / ``mua_max`` head bounds, ``log_param``, neural field
    (``hidden``, ``depth``, ``skip_at``, ``n_octaves``, ``activation``), ``tv`` (``tv_eps``),
    two-stage curriculum (``steps``, ``lr``, ``lr_decay``, ``anneal_fraction``, ``min_coarse``).
    Metrics: ``iou_fraction`` (inclusion masks at this fraction of the peak excess). Baselines:
    ``grid_lr``, ``dd_*``.
    """

    extent: tuple[float, float, float] = (40.0, 40.0, 20.0)
    grid: tuple[int, int, int] = (32, 32, 16)
    musp: float = 1.0
    refractive_index: float = 1.4
    mua_background: float = 0.01
    source_grid: tuple[int, int] = (3, 3)
    source_span: float = 0.6
    detector_grid: tuple[int, int] = (12, 12)
    detector_span: float = 0.85
    detector_samples: int = 3
    min_separation: float = 6.0
    calibrate: bool = True
    noise_std: float = 0.01
    supersample: int = 2
    tol: float = 1e-7
    max_iter: int | None = None
    check_every: int = 1
    warm_start: bool = True
    scene: str = "single"
    inclusion_mua: tuple[float, float] = (0.03, 0.05)
    inclusion_radius: tuple[float, float] = (3.0, 5.0)
    depth_single: tuple[float, float] = (5.0, 8.0)
    depth_multi: tuple[float, float] = (4.0, 11.0)
    depth_deep: tuple[float, float] = (10.0, 13.0)
    lateral_span: float = 0.5
    render_factor: int = 2
    mua_min: float = 0.008
    mua_max: float = 0.1
    log_param: bool = False
    hidden: int = 128
    depth: int = 4
    skip_at: int = 2
    n_octaves: int = 4
    activation: str = "relu"
    tv: float = 1.0
    tv_eps: float = 1e-4
    steps: tuple[int, int] = (300, 600)
    lr: float = 1e-2
    lr_decay: float = 0.5
    anneal_fraction: float = 0.3
    min_coarse: int = 6
    iou_fraction: float = 0.5
    grid_lr: float = 5e-2
    dd_lr: float = 5e-3
    dd_width: int = 64
    dd_stages: int = 3


#: Named presets (partial configs applied before user overrides).
PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    # CPU smoke run: 20×20×10 slab (2 mm voxels), 3×3 sources, 10×10 detectors (≈ 7 s on one
    # core: ≈ 25 ms per native step, two batched PCG solves of 9 sources each).
    "smoke": {
        "grid": (20, 20, 10),
        "detector_grid": (10, 10),
        "hidden": 64,
        "depth": 3,
        "skip_at": 2,
        "steps": (150, 200),
        "lr": 1e-2,
        "dd_width": 32,
    },
    "multi": {"scene": "multi"},
    "deep": {"scene": "deep"},
}


@register("instance", "dot3d")
class DOT3D(PresetInstance):
    """CW diffuse optical tomography instance (see module docstring).

    Args:
        cfg: :class:`DOT3DConfig`, a dict, or ``None``.
        preset: optional preset from :data:`PRESETS` (``"smoke"``, ``"multi"``, ``"deep"``).
        **overrides: individual config fields.
    """

    name = "dot3d"
    Config = DOT3DConfig
    PRESETS = PRESETS
    description = "3-D diffuse optical tomography: absorption from surface reflectance (elliptic)"
    #: detector maps (n_sources, n_dx, n_dy): one surface image per source
    viz_hints = {
        "layout": "stack",
        "word": "sources",
        "measurement_image_label": "Rytov data 100·log(y/y₀), source mean [%]",
    }

    def _validate(self) -> None:
        c = self.cfg
        if len(c.grid) != 3 or len(c.extent) != 3:
            raise ConfigError("dot3d needs a 3-D grid and extent")
        if not 0 < c.mua_min < c.mua_background < c.mua_max:
            raise ConfigError("need 0 < mua_min < mua_background < mua_max")
        if c.inclusion_mua[1] >= c.mua_max:
            raise ConfigError("inclusion_mua must stay below mua_max (the head's upper bound)")
        if c.musp <= 0:
            raise ConfigError("musp must be positive")

    # ---- geometry / physics -----------------------------------------------------------------
    def domain(self) -> Domain:
        c = self.cfg
        return Domain(
            tuple(int(n) for n in c.grid),
            tuple((0.0, float(e)) for e in c.extent),
            axes=("x", "y", "z"),
        )

    @property
    def D(self) -> float:
        """Diffusion coefficient ``1/(3 μ_s')`` (mm)."""
        return 1.0 / (3.0 * self.cfg.musp)

    @property
    def A(self) -> float:
        """Robin boundary factor from the refractive index (Groenhuis fit)."""
        return robin_A(self.cfg.refractive_index)

    def sources(self) -> torch.Tensor:
        """Source positions ``(n_sources, 3)``: a grid on the top face, depth ``1/μ_s'``."""
        c = self.cfg
        (x0, x1), (y0, y1), (z0, _) = self.domain().extent
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        hx, hy = 0.5 * c.source_span * (x1 - x0), 0.5 * c.source_span * (y1 - y0)
        n = tuple(int(v) for v in c.source_grid)
        if n == (1, 1):
            xy = torch.tensor([[cx, cy]], dtype=torch.float64)
        else:
            lo, hi = (cx - hx, cy - hy), (cx + hx, cy + hy)
            pitch = ((hi[0] - lo[0]) / max(n[0] - 1, 1), (hi[1] - lo[1]) / max(n[1] - 1, 1))
            xy = grid_positions(
                n,
                (lo[0] - 0.5 * pitch[0], lo[1] - 0.5 * pitch[1]),
                (hi[0] + 0.5 * pitch[0], hi[1] + 0.5 * pitch[1]),
            ).reshape(-1, 2)
        z = torch.full((xy.shape[0], 1), z0 + 1.0 / c.musp, dtype=torch.float64)
        return torch.cat([xy, z], -1)

    def detectors(self) -> tuple[torch.Tensor, float]:
        """Detector centers ``(n_dx, n_dy, 2)`` and pixel size (mm)."""
        c = self.cfg
        return detector_layout(self.domain(), c.detector_grid, c.detector_span)

    def operator(self, domain: Domain | None = None, tol: float | None = None):
        c = self.cfg
        det, size = self.detectors()
        return DiffuseOpticalOperator(
            domain or self.domain(),
            self.sources(),
            det,
            size,
            D=self.D,
            A=self.A,
            field="mu_a",
            detector_samples=c.detector_samples,
            reference_mua=c.mua_background if c.calibrate else None,
            tol=c.tol if tol is None else tol,
            max_iter=c.max_iter,
            check_every=c.check_every,
            warm_start=c.warm_start,
        )

    def observation_mask(self) -> torch.Tensor:
        """``(n_sources, n_dx, n_dy)`` mask of the observed source–detector pairs."""
        det, _ = self.detectors()
        return separation_mask(self.sources(), det, self.cfg.min_separation)

    def scene_generator(self) -> SceneGenerator:
        c = self.cfg
        return DOT3DScenes(
            self.domain(),
            mua_background=c.mua_background,
            inclusion_mua=tuple(c.inclusion_mua),
            radius=tuple(c.inclusion_radius),
            depth_single=tuple(c.depth_single),
            depth_multi=tuple(c.depth_multi),
            depth_deep=tuple(c.depth_deep),
            lateral_span=c.lateral_span,
            render_factor=c.render_factor,
        )

    def data_generator(self) -> DOTDataGenerator:
        c = self.cfg
        fine = self.operator(self.domain().refine(c.supersample), tol=1e-10)
        fine.warm_start = False
        return DOTDataGenerator(
            fine,
            noise_std=c.noise_std,
            min_separation=c.min_separation,
            supersample=c.supersample,
            fidelity_tag=f"dot-fv-robin-{c.supersample}x-float64",
        )

    def build_problem_measurement_shape(self, field_shape):
        c = self.cfg
        return (int(math.prod(c.source_grid)), *(int(v) for v in c.detector_grid))

    # ---- prior / objective ------------------------------------------------------------------
    def heads(self) -> Heads:
        c = self.cfg
        head = LogBounded if c.log_param else Bounded
        return Heads({"mu_a": head(c.mua_min, c.mua_max, init_value=c.mua_background)})

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
            {"data": RelativeResidualMSE(), "tv": TV("mu_a", isotropic=True, eps=c.tv_eps)},
            weights={"data": 1.0, "tv": c.tv},
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
            name="dot3d",
            meta={"scene": c.scene, "D": self.D, "A": self.A},
        )

    # ---- evaluation -------------------------------------------------------------------------
    def depths(self) -> torch.Tensor:
        """Cell-center depths (mm) of the native grid."""
        return self.domain().axis_coords(normalized=False, dtype=torch.float64)[2]

    def inclusion_mask(self, mua: torch.Tensor) -> torch.Tensor:
        """Half-maximum inclusion mask ``{e > iou_fraction · max e}`` of the absorption excess
        ``e = [μ_a − μ_bg]_+`` (contrast independent: DOT under-recovers contrast by design)."""
        e = excess(mua, self.cfg.mua_background)
        peak = float(e.max())
        if peak <= 0:
            return torch.zeros_like(e, dtype=torch.bool)
        return e > self.cfg.iou_fraction * peak

    def metrics(self) -> dict[str, Callable]:
        return {"psnr": psnr, "ssim": ssim, "mse": mse}

    def inclusion_metrics(self, pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
        """Inclusion localization, depth and contrast recovery (half-maximum masks).

        * ``iou`` — volumetric IoU of the half-maximum inclusion masks;
        * ``iou_2d`` — lateral footprint IoU (columns containing an inclusion voxel);
        * ``depth_error`` — ``|z̄_pred − z̄_gt|`` (mm) of the excess-weighted centroids of the
          masked inclusions ``z̄ = Σ_m e z / Σ_m e``;
        * ``depth_rmse`` — RMSE (mm) of the 2.5-D excess-weighted depth maps over the columns
          detected in both (NaN if none; NeFTY App. H-style projection);
        * ``contrast`` — recovered / true peak excess (1 = full contrast recovery).
        """
        c = self.cfg
        z = self.depths()
        ep, eg = excess(pred, c.mua_background), excess(gt, c.mua_background)
        mp, mg = self.inclusion_mask(pred), self.inclusion_mask(gt)
        union = int((mp | mg).sum())
        out = {"iou": float((mp & mg).sum()) / union if union else 1.0}
        fp, fg = mp.any(-1), mg.any(-1)
        union2 = int((fp | fg).sum())
        out["iou_2d"] = float((fp & fg).sum()) / union2 if union2 else 1.0
        out["depth_error"] = abs(depth_centroid(ep * mp, z) - depth_centroid(eg * mg, z))
        both = fp & fg
        if bool(both.any()):
            diff = depth_map(ep * mp, z, both) - depth_map(eg * mg, z, both)
            diff = diff[torch.isfinite(diff)]
            out["depth_rmse"] = float(diff.square().mean().sqrt()) if diff.numel() else math.nan
        else:
            out["depth_rmse"] = math.nan
        peak = float(eg.max())
        out["contrast"] = float(ep.max()) / peak if peak > 0 else math.nan
        return out

    def evaluate(self, result: Result, gt: Mapping[str, torch.Tensor]) -> dict[str, float]:
        p = result.fields["mu_a"].detach().cpu().double()
        g = torch.as_tensor(gt["mu_a"]).detach().cpu().double()
        if p.shape != g.shape:
            p = resample(p, g.shape)
        out = {k: float(fn(p, g)) for k, fn in self.metrics().items()}
        out.update(self.inclusion_metrics(p, g))
        return out

    def measurement_image(self, measurement: Measurement) -> torch.Tensor:
        """Compact 2-D view: source-averaged Rytov data ``log(y / y₀)`` (%, observed pairs), with
        ``y₀`` the homogeneous-background reading (1 for calibrated data) — the inclusions'
        shadow."""
        y = torch.as_tensor(measurement.data, dtype=torch.float64)
        if self.cfg.calibrate:
            y0 = torch.ones_like(y)
        else:
            y0 = self.background_prediction().to(y)
        m = measurement.mask if measurement.mask is not None else torch.ones_like(y)
        m = torch.as_tensor(m, dtype=torch.float64).expand_as(y)
        ok = (m > 0) & (y > 0) & (y0 > 0)
        r = torch.where(ok, torch.log(torch.where(ok, y / y0, torch.ones_like(y))), 0.0)
        num, den = (r * ok).sum(0), ok.sum(0)
        return torch.where(den > 0, 100.0 * num / den.clamp_min(1), torch.nan)

    @torch.no_grad()
    def background_prediction(self) -> torch.Tensor:
        """Readings of the homogeneous background slab (inversion operator, float64, cached)."""
        cached = getattr(self, "_y0", None)
        if cached is None:
            op = self.operator(tol=1e-10).to(torch.float64)
            mua = torch.full(self.domain().shape, self.cfg.mua_background, dtype=torch.float64)
            cached = op({"mu_a": mua}).detach()
            self._y0 = cached
        return cached

    def sensitivity(self, problem: InverseProblem | None = None, **kw: Any) -> torch.Tensor:
        """Depth-resolved sensitivity ``‖∂y/∂μ_a(x)‖`` of the observed data at the background
        (:func:`nefi.diagnostics.sensitivity_map`: stochastic output probes by default, one adjoint
        solve each; ``exact=True`` computes every column, one solve per voxel)."""
        from ...diagnostics import sensitivity_map

        if problem is None:
            y0 = self.background_prediction().float()
            problem = self.build_problem(Measurement(y0, self.observation_mask().float()))
        mua = torch.full(self.domain().shape, self.cfg.mua_background)
        return sensitivity_map(problem, field="mu_a", fields={"mu_a": mua}, **kw)

    # ---- baselines --------------------------------------------------------------------------
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

        return {"grid": grid, "deep_decoder": deep_decoder}


def make_problem(
    seed: int = 0, scene: str | None = None, **cfg: Any
) -> tuple[InverseProblem, dict[str, torch.Tensor], Measurement]:
    """Convenience: ``(problem, gt, measurement)`` for docs and tests."""
    inst = DOT3D(**({"scene": scene} if scene else {}), **cfg)
    gt, meas = inst.make_measurement(seed)
    return inst.build_problem(meas), gt, meas


def run(cfg: DOT3DConfig | dict | None = None, seed: int = 0, **kw: Any):
    """End-to-end demo (generate → invert → evaluate); returns a ``RunOutput``."""
    return DOT3D(cfg).run(seed=seed, **kw)


METRICS: dict[str, Callable] = {"psnr": psnr, "ssim": ssim, "mse": mse}

__all__ = [
    "METRICS",
    "PRESETS",
    "DOT3D",
    "DOT3DConfig",
    "DOT3DScenes",
    "DOTDataGenerator",
    "DiffuseOpticalOperator",
    "Inclusion",
    "RelativeResidualMSE",
    "boundary_conductance",
    "detector_layout",
    "make_problem",
    "robin_A",
    "robin_sink",
    "run",
    "separation_mask",
]
