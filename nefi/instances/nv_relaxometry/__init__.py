"""nv_relaxometry — NeTMY: sparse spin-density recovery from NV-center relaxometry spectra.

The NV instance of nefi (Zhao, Zhong et al., *Neural Fields for NV-Center Inverse Sensing*,
arXiv 2605.13988). A widefield NV array at standoff ``z0`` reads the magnetic-noise spectrum
``S(ω, r)`` produced by sparse fluctuating spins with density ``ρ ≥ 0`` and local Larmor field
``ω_L``; the task is to recover ``(ρ, ω_L)`` from one noisy spectrum, without labels.

Pipeline (NeTMY Algorithm 1)::

    scene (NVScenes) ──F3 direct float64 + noise──► S_obs (n_freq, H, W)
    coords ─► annealed PE (K=12) ─► MLP 5×320 tanh (+skip) ─► {ρ = softplus(h)σ(g),
              ω_L = 1{ρ>τ max ρ}·Bounded(h_ω)} ─► F2 = (P∗ρ)·L(ω; ω_L) ─► losses (Tab. 7)
    two-stage curriculum 32² → 64² (Tab. 6) ─► energy-anchored scale correction (Eq. 30)

Quick start::

    from nefi.instances.nv_relaxometry import NVRelaxometry
    inst = NVRelaxometry(n=32, steps=(300, 600), hidden=128, depth=4, n_octaves=8)
    out = inst.run(seed=0, scene_class="few/far")
    print(out.metrics)          # gmsd, hungarian_f1, swd, mse, masked_ssim, larmor_mae

See ``docs/instances/nv_relaxometry.md`` for the paper-to-code map and run instructions.
"""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from ...domain import Domain
from ...fields import Bounded, GatedSoftplus, GridField, Heads, NeuralField, Softplus, SupportMasked
from ...losses import L1, TV, LogMSE, LossSet, NormalizedMSE, Tikhonov
from ...measurement import Measurement
from ...metrics.basic import evaluate as _evaluate
from ...metrics.basic import masked_ssim, mse
from ...metrics.localization import gmsd, hungarian_f1, sliced_wasserstein
from ...problem import InverseProblem
from ...registry import register
from ...solve.curriculum import Curriculum, OptimConfig, Stage
from ...solve.postprocess import EnergyScaleCorrection
from ...solve.result import Result
from ...solve.solver import Solver
from ..base import Instance, RunOutput
from .data import NVDataGenerator, downsample_spectrum
from .losses import DirectDensityLoss, noise_map, normalized_noise_map
from .operator import NVDirectSimulator, NVOperator
from .physics import DipolarKernels, frequency_grid, lorentzian, power_kernel_fn
from .scenes import PAPER_CLASSES, NVScenes

log = logging.getLogger("nefi")

LOSS_NAMES = ("log_mse", "noise_map", "direct_density", "sparsity", "l1", "tv", "spectrum")


@dataclass
class NVRelaxometryConfig:
    """Configuration of the NV-relaxometry instance; defaults follow NeTMY.

    Geometry / physics (NeTMY §3.1, App. A.7, App. E.1):
        n: grid size ``n × n`` (64). spacing: pixel size in nm (20). z0: NV standoff in nm (20).
        gamma: Lorentzian linewidth in GHz (0.5). larmor_band: ω_L band in GHz ([1.5, 2.5]).
        freq_range / n_freq: readout grid W; the paper only states "50 frequencies", we use
        [1.0, 3.0] GHz = band ± 2γ so every in-band Lorentzian is resolved to its half-width
        on both sides. units: kernel units ("normalized": peak = 1; "physical": SI).
        length_scale: metres per domain unit (1e-9). nv_axis: NV axis n ((0, 0, 1)).
    Data (App. E.1):
        scene: default scene class. data_operator: "F3" (cross-fidelity, default), "F2"/"F1"
        (matched-operator, inverse crime by construction). noise_std: σ relative to each
        sample's dynamic range (1 %). amp_range, margin_z0, min_sep_px, source_width, counts,
        separations: scene generator settings (see :class:`NVScenes`).
    Inversion:
        mode: inversion operator ("F2"; "F1" for the scalar baseline).
    Field (Tab. 5, Eq. 5, Eq. 26, Eq. 27):
        hidden 320, depth 5 hidden layers (+ output = 6 fully-connected layers, 4.4e5 parameters
        as in Tab. 5), skip_at 3, activation "tanh", n_octaves 12 (base 2, annealed), tau 0.3
        (support mask). larmor_fill: ω_L outside the predicted support; ``None`` = band centre
        (default, see note), ``0.0`` reproduces Eq. (26) literally. out_init_scale: output-layer
        init scale (near-uniform initial density). init_seed: seed of the network
        initialization when none is given (``run``/``compare`` use the run seed).
    Optimizer (Tab. 5): AdamW, weight_decay 1e-4, grad_clip 1.0.
    Curriculum (Tab. 6): steps (3000, 7000), lr 1e-3, lr_decay 0.5 (stage 2 at 5e-4),
        cosine to lr_min_ratio 0.01 of each stage's lr, coarse_factor 2 (32² → 64²), β annealed
        over anneal_fraction of each stage; anneal_reset (True, Tab. 6) resets β to 0 at every
        stage start, False keeps all bands on after stage 1. stage_loss_weights: per-stage
        :class:`LossSet` overrides; default stage 2 replaces D by R_nm as primary fidelity
        (App. D.4). restarts: multi-restart (App. E.10 centered-minimum trapping).
    Losses (Tab. 7, App. D.4): w_log_mse 2.0 (D, log-MSE on max-normalized noise maps, Eq. 19),
        w_noise_map 0.5 (R_nm), w_direct_density 0.1 (R_ds), w_sparsity 1e-2 + w_l1 1e-3 (two
        L1 terms), w_tv 1e-3 (anisotropic TV, unit grid steps), w_spectrum 0 (optional
        per-frequency mean-normalized MSE — not in the paper; makes ω_L identifiable).
        log_eps: the additive constant of Eq. (19); ``None`` = 1e-10 (paper) for noiseless data,
        else ``log_eps_noise_mult`` × the noise level of the max-normalized noise map.
    Post-processing / metrics (Eq. 30, App. E.3): scale_correction, match_radius 2 px,
        peak_threshold 5 %, swd_projections 128, gmsd_c 0.0026.
    Baselines (App. E.2, Tikhonov): grid_steps 5000, grid_lr 5e-3 (Adam), grid_weight_decay
        1e-5, grid_l2 1e-3, grid_tv 1e-3, grid_init 0.35 (uniform initial density).
    """

    # geometry & physics
    n: int = 64
    spacing: float = 20.0
    z0: float = 20.0
    gamma: float = 0.5
    larmor_band: tuple[float, float] = (1.5, 2.5)
    freq_range: tuple[float, float] = (1.0, 3.0)
    n_freq: int = 50
    units: str = "normalized"
    length_scale: float = 1e-9
    nv_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    # data generation
    scene: str = "medium/medium"
    data_operator: str = "F3"
    noise_std: float = 0.01
    amp_range: tuple[float, float] = (0.5, 1.0)
    margin_z0: float = 3.0
    min_sep_px: float = 2.0
    source_width: float = 0.0
    counts: dict[str, list[int]] | None = None
    separations: dict[str, list[float]] | None = None
    # inversion operator
    mode: str = "F2"
    # neural field
    hidden: int = 320
    depth: int = 5
    skip_at: int | None = 3
    activation: str = "tanh"
    n_octaves: int = 12
    tau: float = 0.3
    larmor_fill: float | None = None
    out_init_scale: float = 0.1
    init_seed: int | None = 0
    # optimizer
    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0
    # curriculum
    steps: tuple[int, ...] = (3000, 7000)
    lr: float = 1e-3
    lr_decay: float = 0.5
    lr_min_ratio: float = 0.01
    coarse_factor: int = 2
    anneal_fraction: float = 1.0
    anneal_reset: bool = True
    restarts: int = 1
    stage_loss_weights: list[dict[str, float] | None] = field(
        default_factory=lambda: [None, {"log_mse": 0.0, "noise_map": 2.0}]
    )
    # losses
    w_log_mse: float = 2.0
    w_noise_map: float = 0.5
    w_direct_density: float = 0.1
    w_sparsity: float = 1e-2
    w_l1: float = 1e-3
    w_tv: float = 1e-3
    w_spectrum: float = 0.0
    log_eps: float | None = None
    log_eps_noise_mult: float = 1.0
    # post-processing & metrics
    scale_correction: bool = True
    match_radius: float = 2.0
    peak_threshold: float = 0.05
    swd_projections: int = 128
    gmsd_c: float = 0.0026
    # baselines
    grid_steps: int = 5000
    grid_lr: float = 5e-3
    grid_weight_decay: float = 1e-5
    grid_l2: float = 1e-3
    grid_tv: float = 1e-3
    grid_init: float = 0.35


def _support_masked_ssim(pred, gt) -> float:
    """SSIM on the ground-truth support ``ρ⋆ > 0`` (NeTMY App. E.3 "masked SSIM")."""
    gt_t = torch.as_tensor(gt)
    return masked_ssim(pred, gt_t, gt_t > 0)


@register("instance", "nv_relaxometry")
class NVRelaxometry(Instance):
    """NeTMY NV-relaxometry instance (arXiv 2605.13988).

    Hooks: :meth:`domain`, :meth:`scene_generator` (:class:`NVScenes`, 8 paper classes),
    :meth:`data_generator` (F3 direct float64 + noise), :meth:`build_problem` (neural field +
    :class:`NVOperator` F2/F1 + Tab. 7 losses + energy scale correction), :meth:`default_curriculum`
    (Tab. 6), :meth:`metrics` (GMSD, Hungarian F1, SWD, density MSE, masked SSIM) and
    :meth:`baselines` (``grid`` Tikhonov, ``grid_f1``, ``f1``, ``f2``). :meth:`compare` runs the
    cross-fidelity comparison of Tab. 1 on one sample in a single call.
    """

    name = "nv_relaxometry"
    Config = NVRelaxometryConfig
    description = "NeTMY: spin density + Larmor field from NV relaxometry spectra (F3 → F2)"
    #: Display hints (:mod:`nefi.viz.hints`): the measurement is a spectral stack whose compact view
    #: is the noise map ``Σ_ω S``, drawn log-scaled (a few bright sources dominate it linearly).
    viz_hints: ClassVar[dict[str, Any]] = {
        "layout": "spectra",
        "word": "frequencies",
        "axis_values": "frequencies",
        "measurement_transform": "log",
    }

    # ---- geometry / physics ----------------------------------------------------------------
    def domain(self) -> Domain:
        c = self.cfg
        return Domain.from_spacing((c.n, c.n), float(c.spacing), axes=("x", "y"))

    def frequencies(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        lo, hi = self.cfg.freq_range
        return frequency_grid(lo, hi, self.cfg.n_freq, dtype=dtype)

    def kernels(self) -> DipolarKernels:
        c = self.cfg
        return DipolarKernels(c.z0, c.units, c.length_scale, tuple(c.nv_axis))

    @property
    def band_center(self) -> float:
        lo, hi = self.cfg.larmor_band
        return 0.5 * (float(lo) + float(hi))

    def operator(self, mode: str | None = None, domain: Domain | None = None) -> NVOperator:
        """Inversion operator ``F2`` / ``F1`` (NeTMY Eq. 2) on ``domain`` (default native)."""
        c = self.cfg
        return NVOperator(
            domain or self.domain(),
            self.frequencies(),
            z0=c.z0,
            gamma=c.gamma,
            mode=mode or c.mode,
            units=c.units,
            length_scale=c.length_scale,
            nv_axis=tuple(c.nv_axis),
        )

    def direct_simulator(self) -> NVDirectSimulator:
        """Source-side direct simulator F3 (NeTMY Eq. 11), float64."""
        c = self.cfg
        return NVDirectSimulator(
            self.domain(),
            self.frequencies(torch.float64),
            z0=c.z0,
            gamma=c.gamma,
            units=c.units,
            length_scale=c.length_scale,
            nv_axis=tuple(c.nv_axis),
        )

    # ---- data ------------------------------------------------------------------------------
    def scene_generator(self) -> NVScenes:
        c = self.cfg
        return NVScenes(
            self.domain(),
            z0=c.z0,
            larmor_band=tuple(c.larmor_band),
            amp_range=tuple(c.amp_range),
            counts=c.counts,
            separations=c.separations,
            margin_z0=c.margin_z0,
            min_sep_px=c.min_sep_px,
            source_width=c.source_width,
        )

    def data_generator(self) -> NVDataGenerator:
        """F3 (default) or F2/F1 (matched-operator) data generator with relative noise."""
        c = self.cfg
        kind = str(c.data_operator).upper()
        if kind == "F3":
            op, tag = self.direct_simulator(), NVDirectSimulator.fidelity_tag
        elif kind in ("F1", "F2"):  # matched-operator regime: same tag as the inversion operator
            op = self.operator(kind)
            tag = op.fidelity_tag
        else:
            raise ValueError(f"data_operator must be 'F3', 'F2' or 'F1', got {c.data_operator!r}")
        meta = {"z0": c.z0, "spacing": c.spacing, "gamma": c.gamma, "units": c.units}
        return NVDataGenerator(
            op, noise_std=c.noise_std, relative=True, fidelity_tag=tag, meta=meta
        )

    def build_problem_measurement_shape(self, field_shape: Sequence[int]) -> tuple[int, ...]:
        return (int(self.cfg.n_freq), *tuple(field_shape))

    # ---- problem ---------------------------------------------------------------------------
    def heads(self) -> Heads:
        """``ρ`` = gated softplus (Eq. 5); ``ω_L`` = support-masked bounded head (Eq. 26)."""
        c = self.cfg
        lo, hi = (float(v) for v in c.larmor_band)
        fill = self.band_center if c.larmor_fill is None else float(c.larmor_fill)
        return Heads(
            {
                "rho": GatedSoftplus(),
                "omega_L": SupportMasked(Bounded(lo, hi), depends_on="rho", tau=c.tau, fill=fill),
            }
        )

    def field(self, seed: int | None = None) -> NeuralField:
        """Coordinate MLP with annealed Fourier features (Tab. 5, Eq. 27).

        The weights are drawn from a private RNG stream seeded with ``seed`` (default
        ``cfg.init_seed``) so that a run is reproducible independently of anything else that
        consumed the global torch RNG; ``init_seed=None`` uses the global RNG.
        """
        c = self.cfg
        seed = c.init_seed if seed is None else seed
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(int(seed))
            return NeuralField(
                2,
                self.heads(),
                hidden=c.hidden,
                depth=c.depth,
                skip_at=c.skip_at,
                activation=c.activation,
                n_octaves=c.n_octaves,
                annealed=True,
                include_input=True,
                out_init_scale=c.out_init_scale,
            )

    def log_eps(self, measurement: Measurement | None = None) -> float:
        """Additive constant of the log-MSE (Eq. 19): paper 1e-10, or the noise floor of N̂."""
        c = self.cfg
        if c.log_eps is not None:
            return float(c.log_eps)
        ns = None if measurement is None else measurement.noise_std
        if ns is None:
            return 1e-10
        ns = float(torch.as_tensor(ns).float().mean())
        if ns <= 0:
            return 1e-10
        n_obs = noise_map(measurement.data.double())
        sigma_n = ns * math.sqrt(measurement.data.shape[0]) / max(float(n_obs.max()), 1e-30)
        eps = max(1e-10, c.log_eps_noise_mult * sigma_n)
        log.debug("log-MSE eps = %.3g (noise floor of the max-normalized noise map)", eps)
        return eps

    def losses(self, measurement: Measurement | None = None) -> LossSet:
        """NeTMY objective Eq. (6), Tab. 7 weights (``log_mse`` = D, ``noise_map`` = R_nm, ...)."""
        c = self.cfg
        terms = {
            "log_mse": LogMSE(normalize="max", reduce_axes=(0,), eps=self.log_eps(measurement)),
            "noise_map": NormalizedMSE(normalize="mean", reduce_axes=(0,)),
            "direct_density": DirectDensityLoss("rho"),
            "sparsity": L1("rho"),
            "l1": L1("rho"),
            "tv": TV("rho", isotropic=False, use_spacing=False),
            "spectrum": NormalizedMSE(normalize="mean", reduce_axes=None),
        }
        weights = {
            "log_mse": c.w_log_mse,
            "noise_map": c.w_noise_map,
            "direct_density": c.w_direct_density,
            "sparsity": c.w_sparsity,
            "l1": c.w_l1,
            "tv": c.w_tv,
            "spectrum": c.w_spectrum,
        }
        return LossSet(terms, weights)

    def postprocess(self) -> list:
        return [EnergyScaleCorrection(field="rho")] if self.cfg.scale_correction else []

    def build_problem(
        self,
        measurement: Measurement,
        mode: str | None = None,
        field=None,
        seed: int | None = None,
    ) -> InverseProblem:
        """NeTMY problem for ``measurement`` under inversion operator ``mode`` ("F2" | "F1").

        Args:
            measurement: observed spectrum ``(n_freq, n, n)``.
            mode: inversion operator (default ``cfg.mode``).
            field: optional replacement parameterization (default: the NeTMY neural field).
            seed: network-initialization seed (default ``cfg.init_seed``).
        """
        mode = (mode or self.cfg.mode).upper()
        return InverseProblem(
            self.domain(),
            field if field is not None else self.field(seed),
            self.operator(mode),
            self.losses(measurement),
            measurement,
            postprocess=self.postprocess(),
            curriculum=self.default_curriculum(),
            downsample_obs=downsample_spectrum,
            name=f"nv_relaxometry-netmy-{mode}",
            meta={"mode": mode, "method": "netmy"},
        )

    def default_curriculum(self) -> Curriculum:
        """Tab. 6: 32² (3000 steps, η) → 64² (7000 steps, η/2), cosine to 0.01η, β reset."""
        c = self.cfg
        steps = [int(s) for s in (c.steps if isinstance(c.steps, list | tuple) else [c.steps])]
        weights = list(c.stage_loss_weights or [])
        stages = []
        k = len(steps)
        for i, s in enumerate(steps):
            f = int(c.coarse_factor) ** (k - 1 - i)
            shape = (max(4, c.n // f), max(4, c.n // f))
            w = weights[i] if i < len(weights) else None
            stages.append(
                Stage(
                    name=f"stage{i + 1}",
                    shape=shape,
                    steps=s,
                    lr=c.lr * c.lr_decay**i,
                    lr_schedule="cosine",
                    lr_min_ratio=c.lr_min_ratio,
                    anneal=bool(c.anneal_reset) or i == 0,
                    anneal_fraction=c.anneal_fraction,
                    loss_weights=dict(w) if w else None,
                )
            )
        optim = OptimConfig(
            optimizer=c.optimizer, weight_decay=c.weight_decay, grad_clip=c.grad_clip
        )
        return Curriculum(stages, optim, restarts=int(c.restarts))

    # ---- baselines -------------------------------------------------------------------------
    def build_grid_problem(
        self, measurement: Measurement, mode: str = "F2"
    ) -> tuple[InverseProblem, Curriculum]:
        """Tikhonov free-density baseline (NeTMY App. E.2): grid ρ ≥ 0 (softplus) + ℓ2 + TV.

        Adam, lr 5e-3, weight decay 1e-5, grad clip 1, ``grid_steps`` (5000) steps at native
        resolution; ω_L is a free bounded grid. Same fidelity D and scale correction as NeTMY.
        """
        c = self.cfg
        lo, hi = (float(v) for v in c.larmor_band)
        heads = Heads(
            {"rho": Softplus(init_value=c.grid_init), "omega_L": Bounded(lo, hi, self.band_center)}
        )
        field_ = GridField((c.n, c.n), heads)
        losses = LossSet(
            {
                "log_mse": LogMSE(normalize="max", reduce_axes=(0,), eps=self.log_eps(measurement)),
                "l2": Tikhonov("rho"),
                "tv": TV("rho", isotropic=False, use_spacing=False),
            },
            {"log_mse": c.w_log_mse, "l2": c.grid_l2, "tv": c.grid_tv},
        )
        cur = Curriculum(
            [
                Stage(
                    "grid",
                    (c.n, c.n),
                    int(c.grid_steps),
                    c.grid_lr,
                    lr_schedule="constant",
                    anneal=False,
                )
            ],
            OptimConfig(optimizer="adam", weight_decay=c.grid_weight_decay, grad_clip=c.grad_clip),
        )
        mode = mode.upper()
        prob = InverseProblem(
            self.domain(),
            field_,
            self.operator(mode),
            losses,
            measurement,
            postprocess=self.postprocess(),
            curriculum=cur,
            downsample_obs=downsample_spectrum,
            name=f"nv_relaxometry-grid-{mode}",
            meta={"mode": mode, "method": "grid"},
        )
        return prob, cur

    def baselines(self) -> dict[str, Callable[[Measurement], tuple[InverseProblem, Curriculum]]]:
        """``grid`` (Tikhonov, F2), ``grid_f1``, ``f1`` (NeTMY under F1), ``f2`` (NeTMY, F2)."""

        def netmy(mode):
            def build(measurement, seed=None):
                prob = self.build_problem(measurement, mode, seed=seed)
                return prob, prob.curriculum

            return build

        def grid(mode):
            def build(measurement, seed=None):  # constant init: the seed is irrelevant
                return self.build_grid_problem(measurement, mode)

            return build

        return {
            "grid": grid("F2"),
            "grid_f1": grid("F1"),
            "f1": netmy("F1"),
            "f2": netmy("F2"),
        }

    # ---- metrics ---------------------------------------------------------------------------
    def metrics(self) -> dict[str, Callable]:
        """GMSD, Hungarian F1, SWD (primary) + density MSE, masked SSIM (App. E.3)."""
        c = self.cfg
        return {
            "gmsd": functools.partial(gmsd, c=c.gmsd_c),
            "hungarian_f1": functools.partial(
                hungarian_f1, radius=c.match_radius, threshold=c.peak_threshold
            ),
            "swd": functools.partial(sliced_wasserstein, n_proj=c.swd_projections, seed=0),
            "mse": mse,
            "masked_ssim": _support_masked_ssim,
        }

    def evaluate(self, result: Result, gt: Mapping[str, torch.Tensor]) -> dict[str, float]:
        """Density metrics on ``result.fields["rho"]`` (scale-corrected) + Larmor MAE on support."""
        out = _evaluate(result.fields["rho"], gt["rho"], self.metrics())
        if "omega_L" in result.fields and "omega_L" in gt:
            support = torch.as_tensor(gt["rho"]) > 0
            if bool(support.any()):
                err = (
                    result.fields["omega_L"].float() - torch.as_tensor(gt["omega_L"]).float()
                ).abs()
                out["larmor_mae"] = float(err[support].mean())
        return out

    # ---- end-to-end ----------------------------------------------------------------------------
    def run(
        self,
        seed: int = 0,
        device: str = "auto",
        scene_class: str | None = None,
        curriculum: Curriculum | None = None,
        callbacks: Sequence = (),
        **solver_kw: Any,
    ) -> RunOutput:
        """Generate one sample (F3 + noise), invert it with NeTMY and evaluate (Algorithm 1).

        ``seed`` selects the scene, the noise, the network initialization and the solver seed.
        """
        gt, meas = self.make_measurement(seed, scene_class)
        problem = self.build_problem(meas, seed=seed)
        cur = curriculum or problem.curriculum
        result = Solver(
            problem, cur, device=device, seed=seed, callbacks=callbacks, **solver_kw
        ).run()
        extra = {"scene_class": scene_class or self.cfg.scene, "seed": seed, "method": "netmy"}
        return RunOutput(result, self.evaluate(result, gt), gt, meas, extra)

    def compare(
        self,
        seed: int = 0,
        scene_class: str | None = None,
        methods: Sequence[str] = ("netmy", "f1", "grid", "grid_f1"),
        device: str = "auto",
        step_scale: float = 1.0,
        callbacks: Sequence = (),
        measurement: tuple[dict[str, torch.Tensor], Measurement] | None = None,
        **solver_kw: Any,
    ) -> dict[str, RunOutput]:
        """Run several methods on the *same* F3-generated sample (NeTMY Tab. 1 in one call).

        ``methods`` may contain ``"netmy"`` (F2, the default model) and any key of
        :meth:`baselines`. ``step_scale`` scales every curriculum (smoke runs). The run ``seed``
        selects the sample, the network initialization and the solver seed.
        """
        gt, meas = (
            measurement if measurement is not None else self.make_measurement(seed, scene_class)
        )
        builders: dict[str, Callable] = {
            "netmy": lambda m, seed=None: (lambda p: (p, p.curriculum))(
                self.build_problem(m, seed=seed)
            ),
            **self.baselines(),
        }
        out: dict[str, RunOutput] = {}
        for name in methods:
            if name not in builders:
                raise KeyError(f"unknown method {name!r}; known: {sorted(builders)}")
            prob, cur = builders[name](meas, seed=seed)
            if step_scale != 1.0:
                cur = cur.scaled(step_scale)
            res = Solver(
                prob, cur, device=device, seed=seed, callbacks=callbacks, **solver_kw
            ).run()
            extra = {"method": name, "seed": seed, "scene_class": scene_class}
            out[name] = RunOutput(res, self.evaluate(res, gt), gt, meas, extra)
        return out


# ---- module-level convenience ----------------------------------------------------------------
def build_problem(
    measurement: Measurement,
    mode: str = "F2",
    cfg: NVRelaxometryConfig | Mapping | None = None,
    **overrides: Any,
) -> InverseProblem:
    """NeTMY :class:`InverseProblem` for ``measurement`` under ``mode`` ("F2" | "F1")."""
    return NVRelaxometry(cfg, **overrides).build_problem(measurement, mode)


def default_curriculum(cfg: NVRelaxometryConfig | Mapping | None = None, **overrides) -> Curriculum:
    """Tab. 6 curriculum for ``cfg``."""
    return NVRelaxometry(cfg, **overrides).default_curriculum()


def make_problem(
    seed: int = 0, scene: str | None = None, mode: str = "F2", **cfg: Any
) -> tuple[InverseProblem, dict[str, torch.Tensor], Measurement]:
    """Convenience: ``(problem, gt, measurement)`` for docs, tests and notebooks."""
    inst = NVRelaxometry(**cfg)
    gt, meas = inst.make_measurement(seed, scene)
    return inst.build_problem(meas, mode), gt, meas


def run(
    cfg: NVRelaxometryConfig | Mapping | None = None, seed: int = 0, **run_kw: Any
) -> tuple[Result, dict[str, float]]:
    """End-to-end demo (DESIGN §3.10): generate (F3) → invert (NeTMY) → evaluate."""
    out = NVRelaxometry(cfg).run(seed=seed, **run_kw)
    return out.result, out.metrics


METRICS: dict[str, Callable] = NVRelaxometry().metrics()

__all__ = [
    "LOSS_NAMES",
    "METRICS",
    "PAPER_CLASSES",
    "DipolarKernels",
    "DirectDensityLoss",
    "NVDataGenerator",
    "NVDirectSimulator",
    "NVOperator",
    "NVRelaxometry",
    "NVRelaxometryConfig",
    "NVScenes",
    "build_problem",
    "default_curriculum",
    "downsample_spectrum",
    "frequency_grid",
    "lorentzian",
    "make_problem",
    "noise_map",
    "normalized_noise_map",
    "power_kernel_fn",
    "run",
]
