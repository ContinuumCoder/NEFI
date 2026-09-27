"""deconvolution3d — fluorescence-microscopy z-stack deblurring with an anisotropic 3-D PSF.

Recover the non-negative fluorophore density ``x(x, y, z)`` from a widefield z-stack
``y = h ∗ x + n`` (Gaussian noise) or ``y ~ Poisson(peak · (h ∗ x) + background)`` (photon
counts). The PSF is the Gaussian approximation of the widefield PSF (Zhang, Zerubia & Olivo-Marin,
*Appl. Opt.* 46, 2007) with the classic **axial elongation** ``σ_z = psf_axial_ratio · σ_xy``
(default 3), in physical µm, sampled as a *function of the voxel spacing* by the N-D
:class:`~nefi.operators.FFTConvolution` (zero-padded linear convolution), so it is rebuilt
consistently at every curriculum resolution and on the data-generation grid. The lateral pixel
(``pixel_size``) and the z-step (``z_step``) are independent: voxels are anisotropic like a real
stack.

* **field** — coordinate MLP on annealed 3-D Fourier features with a ``Softplus`` head
  (non-negative intensity);
* **losses** — MSE (Gaussian) or Poisson NLL (counts) + isotropic 3-D TV (NeFTY Eq. 22) + ℓ1
  (sparsity, for ``puncta``);
* **curriculum** — two-stage multiscale (NeTMY Tab. 6): the coarse stage fits area-averaged data
  with the PSF re-sampled at the coarse spacing;
* **data** — :class:`Deconvolution3DScenes` (``filaments``, ``puncta``, ``cells``) blurred on a
  2× finer grid in float64 and area-averaged onto the camera voxels
  (``fidelity_tag="blur3d-2x-float64"``, inverse-crime guard), plus Gaussian or Poisson noise;
* **metrics** — PSNR, slice-averaged SSIM, MSE;
* **baselines** — ``grid`` (free voxels + the same objective), ``wiener3d`` (closed form, SNR by
  the discrepancy principle), ``richardson_lucy`` (the classical ML-EM of fluorescence
  microscopy, fixed iteration count) and ``deep_decoder`` (3-D).
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
from ...fields import Bounded, GridField, Heads, Identity, NeuralField, Softplus
from ...losses import L1, MSE, TV, LossSet, PoissonNLL
from ...measurement import Measurement
from ...metrics.basic import mse, psnr, ssim
from ...operators.base import Nuisance, Operator
from ...operators.conv import FFTConvolution, gaussian_kernel_fn
from ...problem import InverseProblem
from ...registry import register
from ...solve.curriculum import Curriculum
from .._volumetric import PresetInstance
from ..deconvolution.data import BlurDataGenerator
from .classical import richardson_lucy, wiener3d, wiener3d_discrepancy
from .scenes import Deconvolution3DScenes

#: inversion-operator discretization tag (differs from the data generator's ``fidelity_tag``)
OPERATOR_TAG = "blur3d-1x-float32"


@dataclass
class Deconvolution3DConfig:
    """Configuration of the :class:`Deconvolution3D` instance (every number is a field).

    Physics: ``n × n × n_z`` voxels of ``pixel_size × pixel_size × z_step`` µm; Gaussian PSF with
    lateral std ``psf_sigma_xy`` µm and axial std ``psf_axial_ratio · psf_sigma_xy``;
    ``psf_extent`` ∈ {full (exact linear support), same (kernel truncated to the stack size)};
    ``noise`` ∈ {gaussian (relative ``noise_std``), poisson (``peak_counts`` at unit intensity,
    ``background_counts``)}; data simulated on a ``supersample``× finer grid. Scenes: ``scene`` ∈
    {filaments, puncta, cells}, ``render_factor``, ``filament_sigma``, ``puncta_sigma``,
    ``membrane_width`` (µm).
    Prior / solver: neural field (``hidden``, ``depth``, ``skip_at``, ``n_octaves``,
    ``activation``), ``head`` ∈ {softplus, bounded}, ``data_loss`` ∈ {auto, mse, poisson_nll},
    isotropic ``tv`` (``tv_eps``), ``l1``, two-stage curriculum (``steps``, ``lr``, ``lr_decay``,
    ``anneal_fraction``, ``min_coarse``). Baselines: ``grid_lr``, ``dd_*``, ``wiener_snr`` (≤ 0:
    discrepancy principle with ``wiener_tau``), ``rl_iters``.
    """

    n: int = 64
    n_z: int = 32
    pixel_size: float = 0.1
    z_step: float = 0.2
    scene: str = "filaments"
    render_factor: int = 2
    filament_sigma: tuple[float, float] = (0.06, 0.09)
    puncta_sigma: tuple[float, float] = (0.04, 0.07)
    membrane_width: float = 0.06
    psf_sigma_xy: float = 0.12
    psf_axial_ratio: float = 3.0
    psf_extent: str = "full"
    noise: str = "gaussian"
    noise_std: float = 0.01
    peak_counts: float = 200.0
    background_counts: float = 2.0
    supersample: int = 2
    head: str = "softplus"
    init_value: float = 0.05
    data_loss: str = "auto"
    tv: float = 1e-4
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
    grid_lr: float = 5e-2
    dd_lr: float = 5e-3
    dd_width: int = 64
    dd_stages: int = 4
    wiener_snr: float = 0.0
    wiener_tau: float = 1.0
    rl_iters: int = 30


#: Named presets (partial configs applied before user overrides).
PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    # CPU smoke run: 32×32×16 stack (3.2 × 3.2 × 3.2 µm), 16²×8 → 32²×16 curriculum.
    "smoke": {
        "n": 32,
        "n_z": 16,
        "hidden": 64,
        "depth": 3,
        "skip_at": 2,
        "n_octaves": 5,
        "steps": (150, 250),
        "lr": 2e-2,
        "dd_width": 32,
        "dd_stages": 3,
    },
    # photon-limited widefield stack
    "poisson": {"noise": "poisson", "peak_counts": 100.0, "background_counts": 2.0},
    # sparse point sources with an ℓ1 prior
    "puncta": {"scene": "puncta", "l1": 1e-3},
}


@register("instance", "deconvolution3d")
class Deconvolution3D(PresetInstance):
    """3-D widefield fluorescence deconvolution instance (see module docstring).

    Args:
        cfg: :class:`Deconvolution3DConfig`, a dict, or ``None``.
        preset: optional preset from :data:`PRESETS` (``"smoke"``, ``"poisson"``, ``"puncta"``).
        **overrides: individual config fields.
    """

    name = "deconvolution3d"
    Config = Deconvolution3DConfig
    PRESETS = PRESETS
    description = "3-D fluorescence z-stack deconvolution (anisotropic PSF, Gaussian/Poisson)"
    #: the measurement is a field-shaped stack: compact view = its most structured z-slice
    viz_hints = {"layout": "image", "measurement_image_label": "most structured z-slice"}

    def _validate(self) -> None:
        c = self.cfg
        if c.noise not in ("gaussian", "poisson"):
            raise ConfigError(f"noise must be 'gaussian' or 'poisson', got {c.noise!r}")
        if c.psf_extent not in ("full", "same"):
            raise ConfigError(f"psf_extent must be 'full' or 'same', got {c.psf_extent!r}")
        if c.psf_sigma_xy <= 0 or c.psf_axial_ratio <= 0:
            raise ConfigError("psf_sigma_xy and psf_axial_ratio must be positive")

    # ---- physics ------------------------------------------------------------------------
    def domain(self) -> Domain:
        c = self.cfg
        lx = c.n * c.pixel_size
        return Domain(
            (c.n, c.n, c.n_z),
            ((0.0, lx), (0.0, lx), (0.0, c.n_z * c.z_step)),
            axes=("x", "y", "z"),
        )

    def psf_sigma(self) -> tuple[float, float, float]:
        """Physical PSF standard deviations ``(σ_xy, σ_xy, σ_z)`` in µm."""
        c = self.cfg
        return (c.psf_sigma_xy, c.psf_sigma_xy, c.psf_axial_ratio * c.psf_sigma_xy)

    def blur(self, domain: Domain | None = None) -> FFTConvolution:
        """The PSF convolution in image units (no count scaling)."""
        return FFTConvolution(
            gaussian_kernel_fn(self.psf_sigma(), normalize=True),
            domain or self.domain(),
            field="x",
            periodic=False,
            kernel_extent=self.cfg.psf_extent,
        )

    def operator(self, domain: Domain | None = None) -> Operator:
        """Inversion operator: blur, times ``peak_counts`` plus background for Poisson data."""
        c = self.cfg
        op: Operator = self.blur(domain)
        if c.noise == "poisson":
            op = Nuisance(
                op,
                gain=False,
                offset=False,
                init_gain=c.peak_counts,
                init_offset=c.background_counts,
            )
        op.fidelity_tag = OPERATOR_TAG
        return op

    def scene_generator(self) -> SceneGenerator:
        c = self.cfg
        return Deconvolution3DScenes(
            self.domain(),
            render_factor=c.render_factor,
            filament_sigma=tuple(c.filament_sigma),
            puncta_sigma=tuple(c.puncta_sigma),
            membrane_width=c.membrane_width,
        )

    def data_generator(self) -> DataGenerator:
        c = self.cfg
        return BlurDataGenerator(
            self.blur(self.domain().refine(c.supersample)),
            noise=c.noise,
            noise_std=c.noise_std,
            peak_counts=c.peak_counts,
            background_counts=c.background_counts,
            supersample=c.supersample,
            fidelity_tag=f"blur3d-{c.supersample}x-float64",
        )

    def build_problem_measurement_shape(self, field_shape):
        return tuple(field_shape)

    def measurement_stack(self, measurement: Measurement) -> torch.Tensor:
        """Measurement in image units (counts → intensity for Poisson data)."""
        c = self.cfg
        if c.noise == "poisson":
            return (measurement.data - c.background_counts) / c.peak_counts
        return measurement.data

    def measurement_image(self, measurement: Measurement) -> torch.Tensor:
        """Compact 2-D view: the z-slice (image units) with the largest in-slice variance."""
        vol = self.measurement_stack(measurement)
        if vol.ndim != 3:
            return vol
        k = int(vol.flatten(0, 1).var(dim=0).argmax())
        return vol[..., k]

    # ---- prior / objective --------------------------------------------------------------
    def heads(self) -> Heads:
        c = self.cfg
        if c.head == "softplus":
            return Heads({"x": Softplus(init_value=c.init_value)})
        if c.head == "bounded":
            return Heads({"x": Bounded(0.0, 1.0, init_value=c.init_value)})
        raise ConfigError(f"head must be 'softplus' or 'bounded', got {c.head!r}")

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
        kind = c.data_loss
        if kind == "auto":
            kind = "poisson_nll" if c.noise == "poisson" else "mse"
        if kind == "mse":
            data, w = MSE(), 1.0
            if c.noise == "poisson":  # counts: rescale to image units²
                w = 1.0 / c.peak_counts**2
        elif kind == "poisson_nll":
            if c.noise != "poisson":
                raise ConfigError("poisson_nll needs noise='poisson' (count data)")
            data, w = PoissonNLL(), 1.0 / c.peak_counts  # per-count NLL -> image units
        else:
            raise ConfigError(f"data_loss must be auto | mse | poisson_nll, got {kind!r}")
        return LossSet(
            {"data": data, "tv": TV("x", isotropic=True, eps=c.tv_eps), "l1": L1("x")},
            weights={"data": w, "tv": c.tv, "l1": c.l1},
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
            name="deconvolution3d",
            meta={"scene": c.scene, "noise": c.noise, "psf_sigma_um": self.psf_sigma()},
        )

    def metrics(self) -> dict[str, Callable]:
        return {"psnr": psnr, "ssim": ssim, "mse": mse}

    # ---- classical references & baselines ---------------------------------------------
    def noise_level(self, measurement: Measurement) -> float:
        """Noise std in image units (known Gaussian std, or ``√λ / peak`` for Poisson counts)."""
        c = self.cfg
        if c.noise == "poisson":
            return float(torch.sqrt(measurement.data.clamp_min(1.0).mean())) / c.peak_counts
        ns = measurement.noise_std
        if ns is not None:
            return float(torch.as_tensor(ns).mean())
        return 0.01 * float(measurement.data.abs().max())

    def wiener_reconstruction(self, measurement: Measurement) -> tuple[torch.Tensor, float]:
        """Closed-form 3-D Wiener reconstruction (image units, ≥ 0) and the SNR used."""
        c = self.cfg
        img = self.measurement_stack(measurement)
        blur = self.blur()
        if c.wiener_snr > 0:
            return wiener3d(img, blur, c.wiener_snr), float(c.wiener_snr)
        return wiener3d_discrepancy(img, blur, self.noise_level(measurement), tau=c.wiener_tau)

    def wiener(self, measurement: Measurement) -> torch.Tensor:
        """Closed-form 3-D Wiener reconstruction of ``measurement``."""
        return self.wiener_reconstruction(measurement)[0]

    def richardson_lucy(self, measurement: Measurement, n_iter: int | None = None) -> torch.Tensor:
        """Richardson–Lucy reconstruction (``rl_iters`` EM iterations) in image units."""
        c = self.cfg
        n_iter = c.rl_iters if n_iter is None else int(n_iter)
        blur = self.blur().to(dtype=measurement.data.dtype)
        if c.noise == "poisson":
            return richardson_lucy(
                measurement.data, blur, n_iter, c.peak_counts, c.background_counts
            )
        return richardson_lucy(measurement.data, blur, n_iter)

    def _direct(self, m: Measurement, name: str, fn: Callable[[Measurement], torch.Tensor]):
        c = self.cfg
        shape = (c.n, c.n, c.n_z)

        def reconstruct(obs: Measurement, dom: Domain) -> torch.Tensor:
            return fn(obs)

        field = GridField(shape, Heads({"x": Identity()}), init=fn(m)[..., None].float())
        cur = Curriculum.single(shape, steps=1, lr=0.0, anneal=False)
        prob = InverseProblem(
            self.domain(),
            field,
            self.operator(),
            self.losses(),
            m,
            curriculum=cur,
            name=f"deconvolution3d-{name}",
            meta={"solver": "direct", "reconstruct": reconstruct, "baseline": name},
        )
        return prob, cur

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

        def wiener_baseline(m: Measurement):
            _, snr = self.wiener_reconstruction(m)
            blur = self.blur()
            return self._direct(
                m,
                "wiener3d",
                lambda obs: wiener3d(self.measurement_stack(obs), blur.to(obs.data.dtype), snr),
            )

        def rl_baseline(m: Measurement):
            return self._direct(m, "richardson_lucy", self.richardson_lucy)

        return {
            "grid": grid,
            "wiener3d": wiener_baseline,
            "richardson_lucy": rl_baseline,
            "deep_decoder": deep_decoder,
        }


def make_problem(
    seed: int = 0, scene: str | None = None, **cfg: Any
) -> tuple[InverseProblem, dict[str, torch.Tensor], Measurement]:
    """Convenience: ``(problem, gt, measurement)`` for docs and tests."""
    inst = Deconvolution3D(**({"scene": scene} if scene else {}), **cfg)
    gt, meas = inst.make_measurement(seed)
    return inst.build_problem(meas), gt, meas


def run(cfg: Deconvolution3DConfig | dict | None = None, seed: int = 0, **kw: Any):
    """End-to-end demo (generate → invert → evaluate); returns a ``RunOutput``."""
    return Deconvolution3D(cfg).run(seed=seed, **kw)


METRICS: dict[str, Callable] = {"psnr": psnr, "ssim": ssim, "mse": mse}

__all__ = [
    "METRICS",
    "OPERATOR_TAG",
    "PRESETS",
    "Deconvolution3D",
    "Deconvolution3DConfig",
    "Deconvolution3DScenes",
    "make_problem",
    "richardson_lucy",
    "run",
    "wiener3d",
    "wiener3d_discrepancy",
]
