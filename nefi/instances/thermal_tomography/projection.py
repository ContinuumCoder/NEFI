"""Projections of a reconstructed 3-D diffusivity to 2-D defect masks and 2.5-D depth maps.

NeFTY App. H.3 (used for the real-PVC benchmarks, which only provide 2-D masks and discrete
depths):

* **2-D defect mask** — depth-averaged diffusivity ``ᾱ(x, y) = N_z⁻¹ Σ_z α̂(x, y, z)``, one-sided
  normalized deficit ``d = [1 − ᾱ/α_base]_+``, robust noise floor ``σ̂ = 1.4826 · MAD`` and mask
  ``1{d > 2σ̂}``.
* **2.5-D depth** — for each masked pixel, the median depth of the z-voxels whose diffusivity falls
  below the bulk by more than the noise floor of the defect-free region.

Noise floor: by default the MAD is taken of the *signed* deficit ``1 − ᾱ/α_base``, which is
symmetric around zero on sound pixels. The MAD of the clipped deficit (the literal reading of
App. H.3, ``mad_of="clipped"``) collapses towards zero whenever more than half of the pixels are
sound, because the clipping maps all of their negative fluctuations to exactly zero.

Grids follow the heat operator convention: the last axis is depth ``z`` (front face first).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ...errors import ShapeError

__all__ = ["bulk_profile", "defect_mask_2d", "depth_map_25d", "gt_depth_map", "mad_sigma"]


def mad_sigma(x: torch.Tensor) -> torch.Tensor:
    """Robust standard deviation ``1.4826 · median(|x − median(x)|)``."""
    x = x.reshape(-1)
    med = x.median()
    return 1.4826 * (x - med).abs().median()


def bulk_profile(
    alpha: torch.Tensor, alpha_base: float | torch.Tensor | None = None
) -> torch.Tensor:
    """Bulk (defect-free) diffusivity broadcast to ``alpha``'s shape.

    ``alpha_base`` may be a scalar (homogeneous bulk), a depth profile of length ``n_z`` (layered
    bulk), a full field, or ``None`` — then the bulk is estimated label-free as the lateral median
    of every depth slice (robust as long as defects cover less than half of each slice).
    """
    a = torch.as_tensor(alpha)
    if alpha_base is None:
        prof = a.reshape(-1, a.shape[-1]).median(dim=0).values
        return prof.expand_as(a)
    b = torch.as_tensor(alpha_base, dtype=a.dtype, device=a.device)
    if b.ndim == 0 or tuple(b.shape) == (a.shape[-1],):
        return b.expand_as(a)
    if tuple(b.shape) == tuple(a.shape):
        return b
    raise ShapeError(
        f"alpha_base must be a scalar, a depth profile of length {a.shape[-1]} or a field of "
        f"shape {tuple(a.shape)}; got {tuple(b.shape)}"
    )


def _noise_sigma(signed: torch.Tensor, mad_of: str) -> torch.Tensor:
    if mad_of == "signed":
        return mad_sigma(signed)
    if mad_of == "clipped":
        return mad_sigma(signed.clamp_min(0.0))
    raise ValueError(f"mad_of must be 'signed' or 'clipped', got {mad_of!r}")


def defect_mask_2d(
    alpha: torch.Tensor,
    alpha_base: float | torch.Tensor | None = None,
    k: float = 2.0,
    min_sigma: float = 1e-3,
    return_info: bool = False,
    mad_of: str = "signed",
):
    """2-D defect mask from a 3-D diffusivity (NeFTY App. H.3).

    Args:
        alpha: reconstructed diffusivity ``(nx, [ny,] nz)``.
        alpha_base: bulk diffusivity (see :func:`bulk_profile`; ``None`` = label-free estimate).
        k: noise-floor multiplier (paper: 2).
        min_sigma: floor on ``σ̂`` (relative deficit units).
        return_info: also return ``{"deficit", "sigma", "threshold"}``.
        mad_of: ``"signed"`` (default, robust) or ``"clipped"`` (literal App. H.3) deficit for the
            MAD noise floor (see the module docstring).

    Returns:
        Boolean mask of shape ``alpha.shape[:-1]`` (and the info dict if requested).
    """
    a = torch.as_tensor(alpha).detach().double()
    base = bulk_profile(a, None if alpha_base is None else torch.as_tensor(alpha_base).double())
    abar, bbar = a.mean(-1), base.mean(-1)
    signed = 1.0 - abar / bbar
    deficit = signed.clamp_min(0.0)
    sigma = _noise_sigma(signed, mad_of)
    thr = max(k * float(sigma), float(min_sigma))
    mask = deficit > thr
    if return_info:
        return mask, {"deficit": deficit, "sigma": float(sigma), "threshold": thr}
    return mask


def _depth_coords(nz: int, depth_coords, thickness: float | None, like: torch.Tensor):
    if depth_coords is not None:
        z = torch.as_tensor(depth_coords, dtype=like.dtype, device=like.device)
        if z.shape != (nz,):
            raise ShapeError(f"depth_coords must have length {nz}, got {tuple(z.shape)}")
        return z
    H = 1.0 if thickness is None else float(thickness)
    return (torch.arange(nz, dtype=like.dtype, device=like.device) + 0.5) * (H / nz)


def _masked_reduce(values: torch.Tensor, flags: torch.Tensor, how: str) -> torch.Tensor:
    """Per-column median / min / mean of ``values`` (last axis) where ``flags``; NaN if none."""
    count = flags.sum(-1)
    inf = torch.full_like(values, float("inf"))
    if how == "min":
        out = torch.where(flags, values, inf).min(-1).values
    elif how == "mean":
        out = (values * flags).sum(-1) / count.clamp_min(1)
    elif how == "median":
        srt = torch.where(flags, values, inf).sort(-1).values
        lo = ((count - 1).clamp_min(0) // 2).unsqueeze(-1)
        hi = (count // 2).clamp_max(values.shape[-1] - 1).unsqueeze(-1)
        out = 0.5 * (srt.gather(-1, lo) + srt.gather(-1, hi)).squeeze(-1)
    else:
        raise ValueError(f"unknown reduction {how!r}; use 'median', 'min' or 'mean'")
    return torch.where(count > 0, out, torch.full_like(out, float("nan")))


def depth_map_25d(
    alpha: torch.Tensor,
    alpha_base: float | torch.Tensor | None = None,
    mask2d: torch.Tensor | None = None,
    depth_coords: Sequence[float] | torch.Tensor | None = None,
    thickness: float | None = None,
    k: float = 2.0,
    min_sigma: float = 1e-3,
    reduce: str = "median",
    mad_of: str = "signed",
) -> torch.Tensor:
    """2.5-D defect depth map (NeFTY App. H.3).

    A voxel is a defect voxel if its one-sided deficit ``[1 − α̂/α_base]_+`` exceeds ``k`` times the
    robust noise floor of the defect-free region (pixels outside ``mask2d``). The depth of a masked
    pixel is the ``reduce`` (default: median) of the depths of its defect voxels.

    Args:
        alpha: reconstructed diffusivity ``(nx, [ny,] nz)``.
        alpha_base: bulk diffusivity (``None`` = label-free estimate, :func:`bulk_profile`).
        mask2d: pixels assigned to a defect (default: :func:`defect_mask_2d`).
        depth_coords: physical depth of each z cell centre (default: uniform cells of
            ``thickness`` or of a unit thickness).
        thickness: slab thickness ``H`` used when ``depth_coords`` is not given.
        k / min_sigma / mad_of: noise-floor rule as in :func:`defect_mask_2d`.
        reduce: ``"median"`` (paper), ``"min"`` (top of the defect) or ``"mean"``.

    Returns:
        Depth map of shape ``alpha.shape[:-1]``; NaN outside the mask or where no voxel is flagged.
    """
    a = torch.as_tensor(alpha).detach().double()
    base = bulk_profile(a, None if alpha_base is None else torch.as_tensor(alpha_base).double())
    if mask2d is None:
        mask2d = defect_mask_2d(a, base, k=k, min_sigma=min_sigma, mad_of=mad_of)
    mask2d = torch.as_tensor(mask2d).bool()
    signed = 1.0 - a / base
    sound = ~mask2d
    ref = signed[sound] if bool(sound.any()) else signed
    thr = max(k * float(_noise_sigma(ref, mad_of)), float(min_sigma))
    flags = (signed > thr) & mask2d.unsqueeze(-1)
    z = _depth_coords(a.shape[-1], depth_coords, thickness, a)
    return _masked_reduce(z.expand_as(a), flags, reduce)


def gt_depth_map(
    defect_mask: torch.Tensor,
    depth_coords: Sequence[float] | torch.Tensor | None = None,
    thickness: float | None = None,
    reduce: str = "median",
) -> torch.Tensor:
    """Ground-truth 2.5-D depth labels from a 3-D defect mask (same reduction as the prediction)."""
    m = torch.as_tensor(defect_mask).bool()
    like = torch.zeros((), dtype=torch.float64, device=m.device)
    z = _depth_coords(m.shape[-1], depth_coords, thickness, like)
    return _masked_reduce(z.expand(m.shape), m, reduce)
