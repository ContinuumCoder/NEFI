"""NeTMY-specific loss terms (App. D.4) and noise-map helpers.

The canonical fidelity D (Eq. 19) and the mean-normalized companion R_nm are generic and live in
the core (:class:`nefi.losses.LogMSE` with ``normalize="max", reduce_axes=(0,)`` and
:class:`nefi.losses.NormalizedMSE` with ``normalize="mean", reduce_axes=(0,)``, both acting on the
frequency-summed noise map ``N(r) = Σ_ω S(ω, r)``). This module adds the direct-density proxy
R_ds, which is specific to NV relaxometry.
"""

from __future__ import annotations

import torch

from ...losses.base import Context, Loss
from ...registry import register


def noise_map(spectrum: torch.Tensor, freq_axis: int = 0) -> torch.Tensor:
    """Frequency-summed noise map ``N(r; S) = Σ_{ω∈W} S(ω, r)`` (NeTMY §3.2, App. B.1)."""
    return spectrum.sum(dim=freq_axis)


def normalized_noise_map(
    spectrum: torch.Tensor, how: str = "max", freq_axis: int = 0, eps: float = 1e-30
) -> torch.Tensor:
    """``N̂ = N / max N`` (``how="max"``, Eq. 19) or ``N / mean N`` (``how="mean"``, R_nm)."""
    n = noise_map(spectrum, freq_axis)
    if how == "max":
        return n / n.max().clamp_min(eps)
    if how == "mean":
        return n / n.mean().clamp_min(eps)
    raise ValueError(f"unknown normalization {how!r} (use 'max' or 'mean')")


@register("loss", "nv_direct_density")
class DirectDensityLoss(Loss):
    """Direct-density proxy ``R_ds = MSE(ρ² / mean ρ², N_obs / mean N_obs)`` (NeTMY App. D.4).

    A soft, oracle-free support regularizer: it nudges large predicted densities toward bright
    pixels of the observed, frequency-summed noise map ``N_obs = Σ_ω S_obs``. No ground-truth
    labels are used.

    Note:
        The identity "spectrum integral ∝ ρ²" behind this term is exact only for the coherent
        point-source operator F1. Under the thermal operator F2 the spectrum is *linear* in ρ, so
        R_ds is a heuristic (the paper keeps it because the ablation shows it accelerates support
        discovery; weight 0.1 in Tab. 7). It compares against the *blurred* observed map, so a
        large weight biases ρ toward the blurred support.

    Args:
        field: density field name.
        freq_axis: frequency axis of the observed spectrum.
        eps: guard for the mean normalizations.
    """

    is_data = False

    def __init__(
        self, field: str = "rho", freq_axis: int = 0, eps: float = 1e-12, name: str | None = None
    ) -> None:
        super().__init__(name or "direct_density")
        self.field_name = field
        self.freq_axis = int(freq_axis)
        self.eps = float(eps)

    def forward(self, ctx: Context) -> torch.Tensor:
        rho = ctx.field(self.field_name)
        n_obs = noise_map(ctx.obs.data, self.freq_axis).to(rho)
        if n_obs.shape != rho.shape:
            raise ValueError(
                f"direct-density loss: noise map {tuple(n_obs.shape)} does not match the density "
                f"grid {tuple(rho.shape)} (is the observation downsampled to the stage shape?)"
            )
        r2 = rho * rho
        a = r2 / r2.mean().clamp_min(self.eps)
        b = n_obs / n_obs.mean().clamp_min(self.eps)
        return ((a - b) ** 2).mean()


__all__ = ["DirectDensityLoss", "noise_map", "normalized_noise_map"]
