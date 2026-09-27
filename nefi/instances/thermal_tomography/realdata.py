"""Real pulsed-thermography data: loading, ambient shift, physical ↔ unitless scaling (NeFTY App.
A.5 / H).

The unitless solver evolves in ``(t / t_total, x / L0)`` coordinates with the effective
diffusivity ``α_sim = α_phys · t_total / L0²`` (the Fourier number), so one set of solver
parameters serves micron/microsecond and centimetre/second specimens alike. This module turns a
recorded surface sequence into a :class:`~nefi.measurement.Measurement` plus the matching
:class:`ThermalTomographyConfig` overrides (uniform front-face flash, convective Robin back face,
lateral Neumann boundaries for cropped specimens — the three PVC adaptations of App. H.2), and
converts recovered fields / depths back to physical units. Nothing here downloads data: pass a
local file (``.npy`` / ``.npz`` / ``.pt`` / ``.mat`` / a TIFF stack / a directory of images).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...errors import ConfigError
from ...measurement import Measurement
from ...utils.tensor import resample

__all__ = [
    "PhysicalScales",
    "config_for_frames",
    "load_frames",
    "measurement_from_frames",
    "physical_depth_map",
    "pvc_overrides",
]


@dataclass(frozen=True)
class PhysicalScales:
    """Dimensional scales of a specimen and its recording (App. A.5, H.2).

    Args:
        alpha_phys: bulk thermal diffusivity in m²/s (PVC ≈ 1.2e-7).
        L0: lateral characteristic length in m (e.g. the imaged footprint, 0.1 for 100 mm).
        thickness: specimen thickness in m.
        t_total: recording window used for inversion in s.
        lateral_size: imaged lateral size in m ``(Lx, Ly)``; defaults to ``(L0, L0)``.
    """

    alpha_phys: float
    L0: float
    thickness: float
    t_total: float
    lateral_size: tuple[float, float] | None = None

    @property
    def fourier_number(self) -> float:
        """``Fo = α_phys · t_total / L0²`` — the unitless bulk diffusivity ``α_sim``."""
        return self.alpha_phys * self.t_total / self.L0**2

    def unitless_alpha(self, alpha_phys: float | torch.Tensor | None = None):
        a = self.alpha_phys if alpha_phys is None else alpha_phys
        return a * self.t_total / self.L0**2

    def physical_alpha(self, alpha_sim: float | torch.Tensor):
        return alpha_sim * self.L0**2 / self.t_total

    def unitless_time(self, t_phys: float) -> float:
        return t_phys / self.t_total

    def physical_time(self, t_sim: float) -> float:
        return t_sim * self.t_total

    def unitless_length(self, x_phys: float) -> float:
        return x_phys / self.L0

    def physical_length(self, x_sim: float | torch.Tensor):
        return x_sim * self.L0

    def extent(self) -> tuple[float, float, float]:
        lx, ly = self.lateral_size or (self.L0, self.L0)
        return (lx / self.L0, ly / self.L0, self.thickness / self.L0)


def load_frames(path: str | Path, key: str | None = None, dtype=torch.float32) -> torch.Tensor:
    """Load a surface temperature sequence ``(n_t, H, W)`` from a local file or directory.

    Supported: ``.npy``, ``.npz`` (``key`` or the first array), ``.pt`` / ``.pth`` (tensor or dict
    with ``key``), ``.mat`` (``key`` or the first non-private variable), multi-page
    ``.tif`` / ``.tiff`` (via ``tifffile`` or PIL), and a directory of image files sorted by name.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"no such file or directory: {p}")
    if p.is_dir():
        from PIL import Image  # optional dependency, only for image folders

        files = sorted(
            f for f in p.iterdir() if f.suffix.lower() in (".png", ".tif", ".tiff", ".jpg")
        )
        if not files:
            raise ConfigError(f"directory {p} contains no png/tif/jpg frames")
        arr = np.stack([np.asarray(Image.open(f), dtype=np.float64) for f in files])
    elif p.suffix == ".npy":
        arr = np.load(p)
    elif p.suffix == ".npz":
        z = np.load(p)
        arr = z[key] if key else z[list(z.keys())[0]]
    elif p.suffix in (".pt", ".pth"):
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            obj = obj[key] if key else next(iter(obj.values()))
        arr = torch.as_tensor(obj).numpy()
    elif p.suffix == ".mat":
        from scipy.io import loadmat

        m = loadmat(p)
        names = [k for k in m if not k.startswith("__")]
        arr = m[key] if key else m[names[0]]
    elif p.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile

            arr = tifffile.imread(p)
        except ImportError:
            from PIL import Image

            im = Image.open(p)
            pages = []
            try:
                while True:
                    pages.append(np.asarray(im, dtype=np.float64))
                    im.seek(im.tell() + 1)
            except EOFError:
                pass
            arr = np.stack(pages)
    else:
        raise ConfigError(f"unsupported frame file type {p.suffix!r}")
    t = torch.as_tensor(np.asarray(arr), dtype=dtype)
    if t.ndim == 2:
        t = t[None]
    if t.ndim != 3:
        raise ConfigError(f"expected frames of shape (n_t, H, W), got {tuple(t.shape)}")
    return t


def measurement_from_frames(
    frames: torch.Tensor,
    *,
    dt: float,
    t_ambient: float | torch.Tensor | None = None,
    pre_flash_frames: int = 0,
    first_frame: int = 1,
    frame_stride: int = 1,
    crop: tuple[slice, slice] | None = None,
    resample_to: Sequence[int] | None = None,
    max_frames: int | None = None,
    noise_std: float | str | None = "auto",
    dtype=torch.float32,
) -> tuple[Measurement, dict[str, Any]]:
    """Ambient-shift, crop, subsample and package recorded frames as a :class:`Measurement`.

    Args:
        frames: ``(n_t, H, W)`` temperatures (any units) at spacing ``dt`` seconds.
        dt: physical frame spacing in seconds.
        t_ambient: ambient temperature (scalar or per-pixel map). Default: the mean of the
            ``pre_flash_frames`` leading frames if given, else the per-pixel minimum over time.
        pre_flash_frames: number of leading frames recorded before the flash (dropped).
        first_frame / frame_stride: index (1-based, after the flash) of the first used frame and
            the temporal stride (both mirror :class:`ThermalTomographyConfig`).
        crop: ``(rows, cols)`` slices applied to every frame.
        resample_to: lateral shape to area-average the frames to (e.g. ``(64, 64)``).
        max_frames: keep at most this many frames after subsampling.
        noise_std: ``"auto"`` estimates the noise level from the data
            (:func:`nefi.auto.estimate_noise`), a float sets it, ``None`` leaves it unknown.

    Returns:
        ``(measurement, meta)`` where ``meta`` records ``dt_phys``, ``t_total_phys``, ``n_frames``,
        ``first_frame``, ``frame_stride`` and the ambient level.
    """
    x = torch.as_tensor(frames).to(torch.float64)
    if x.ndim != 3:
        raise ConfigError(f"frames must be (n_t, H, W), got {tuple(x.shape)}")
    if pre_flash_frames > 0:
        pre = x[:pre_flash_frames]
        x = x[pre_flash_frames:]
        if t_ambient is None:
            t_ambient = pre.mean(0)
    if t_ambient is None:
        t_ambient = x.amin(dim=0)
    amb = torch.as_tensor(t_ambient, dtype=torch.float64)
    x = x - amb
    if crop is not None:
        x = x[:, crop[0], crop[1]]
    x = x[first_frame - 1 :: max(1, frame_stride)]
    if max_frames is not None:
        x = x[:max_frames]
    if resample_to is not None:
        x = resample(x, tuple(resample_to))
    data = x.to(dtype)
    n_used = data.shape[0]
    dt_used = dt * max(1, frame_stride)
    meta: dict[str, Any] = {
        "dt_phys": float(dt),
        "dt_used_phys": float(dt_used),
        "first_frame": int(first_frame),
        "frame_stride": int(frame_stride),
        "n_frames_used": int(n_used),
        "t_total_phys": float(dt * (first_frame - 1) + dt_used * n_used),
        "ambient": float(amb.mean()),
        "source": "real",
    }
    meas = Measurement(data, noise_std=None if noise_std in (None, "auto") else float(noise_std))
    if noise_std == "auto":
        try:
            from ...auto import estimate_noise

            meas.noise_std = float(estimate_noise(meas))
        except Exception:  # pragma: no cover - estimator is best effort
            meas.noise_std = None
    meas.meta.update(meta)
    return meas, meta


def pvc_overrides(
    scales: PhysicalScales,
    *,
    robin_h: float = 0.0,
    lateral: str = "neumann",
    alpha_contrast: float = 50.0,
    flash_width_z: float | None = None,
) -> dict[str, Any]:
    """The three PVC adaptations of App. H.2 as config overrides.

    Uniform front-face flash, convective Robin back face (``robin_h`` in unitless form, i.e.
    ``h_phys · L0 / k``-style calibration is the caller's job), lateral Neumann boundaries for
    cropped recordings, and diffusivity bounds ``[α_sim / alpha_contrast, 2 α_sim]`` around the
    unitless bulk value.
    """
    a = scales.fourier_number
    out: dict[str, Any] = {
        "initial": "flash",
        "bc_back": "robin" if robin_h > 0 else "neumann",
        "robin_h": float(robin_h),
        "bc_lateral": lateral,
        "alpha_min": a / alpha_contrast,
        "alpha_max": 2.0 * a,
        "alpha_init": a,
        "alpha_base_range": (0.9 * a, 1.1 * a),
        "alpha_defect_range": (a / alpha_contrast, a / (alpha_contrast / 3)),
    }
    if flash_width_z is not None:
        out["flash_width_z"] = float(flash_width_z)
    return out


def config_for_frames(
    measurement: Measurement,
    scales: PhysicalScales,
    *,
    nz: int = 16,
    base: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Config overrides for :class:`ThermalTomography` matching a real measurement.

    Sets the unitless extent from ``scales``, the lateral grid from the frame shape, ``nz``
    through-thickness cells, and the unitless time step / frame bookkeeping from the measurement
    meta written by :func:`measurement_from_frames` (``dt = dt_used_phys / t_total``,
    ``n_frames = first_frame − 1 + n_used·stride`` in solver steps). ``base`` is merged first,
    keyword ``overrides`` last.
    """
    meta = measurement.meta
    if "dt_used_phys" not in meta:
        raise ConfigError("measurement lacks real-data meta; build it with measurement_from_frames")
    n_used = int(meta["n_frames_used"])
    stride = int(meta["frame_stride"])
    first = int(meta["first_frame"])
    dt_sim = scales.unitless_time(float(meta["dt_phys"]))
    n_steps = (first - 1) + n_used * stride
    h, w = tuple(measurement.data.shape[1:])
    cfg: dict[str, Any] = dict(base or {})
    cfg.update(
        {
            "extent": scales.extent(),
            "grid": (int(h), int(w), int(nz)),
            "dt": float(dt_sim),
            "n_frames": int(n_steps),
            "first_frame": first,
            "frame_stride": stride,
            "noise_std": 0.0,
        }
    )
    cfg.update(overrides)
    return cfg


def physical_depth_map(depth_sim: torch.Tensor, scales: PhysicalScales, unit: str = "mm"):
    """Convert a unitless depth map (fraction of ``L0``) to physical units."""
    d = torch.as_tensor(depth_sim) * scales.L0
    if unit == "m":
        return d
    if unit == "mm":
        return d * 1e3
    if unit == "um":
        return d * 1e6
    raise ConfigError("unit must be 'm', 'mm' or 'um'")
