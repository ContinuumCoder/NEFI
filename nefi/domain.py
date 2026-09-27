"""Physical domain: extents, resolution and normalized coordinate grids.

Coordinates are **cell-centered** and normalized to ``[-1, 1]`` per axis (NeTMY App. D.1,
NeFTY App. D.1), so a field can be queried at any resolution of the same physical domain — the
basis of multiscale curricula.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from .errors import ShapeError
from .utils.tensor import shape_tuple


@dataclass(frozen=True)
class Domain:
    """A rectangular physical domain discretized on a uniform grid.

    Args:
        shape: native grid shape, e.g. ``(64, 64)`` or ``(64, 64, 16)``.
        extent: physical ``(lo, hi)`` per axis, same length as ``shape``.
        axes: optional axis names, e.g. ``("x", "y", "z")``.
    """

    shape: tuple[int, ...]
    extent: tuple[tuple[float, float], ...]
    axes: tuple[str, ...] | None = field(default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", shape_tuple(self.shape))
        ext = tuple((float(lo), float(hi)) for lo, hi in self.extent)
        object.__setattr__(self, "extent", ext)
        if len(ext) != len(self.shape):
            raise ShapeError(
                f"extent has {len(ext)} axes but shape {self.shape} has {len(self.shape)}"
            )
        for lo, hi in ext:
            if not hi > lo:
                raise ShapeError(f"extent must satisfy hi > lo, got {(lo, hi)}")
        if self.axes is not None:
            object.__setattr__(self, "axes", tuple(self.axes))
            if len(self.axes) != len(self.shape):
                raise ShapeError("axes must have one name per dimension")

    # ---- constructors -------------------------------------------------------------------
    @staticmethod
    def unit(shape: Sequence[int] | int, axes: Sequence[str] | None = None) -> Domain:
        """Domain ``[0, 1]^d``."""
        shape = shape_tuple(shape)
        return Domain(shape, tuple((0.0, 1.0) for _ in shape), tuple(axes) if axes else None)

    @staticmethod
    def from_spacing(
        shape: Sequence[int] | int,
        spacing: float | Sequence[float],
        origin: float | Sequence[float] = 0.0,
        axes: Sequence[str] | None = None,
    ) -> Domain:
        """Domain from grid shape and physical cell size (e.g. NeTMY: 64 px × 20 nm)."""
        shape = shape_tuple(shape)
        sp = (spacing,) * len(shape) if isinstance(spacing, int | float) else tuple(spacing)
        og = (origin,) * len(shape) if isinstance(origin, int | float) else tuple(origin)
        ext = tuple((float(o), float(o + n * s)) for n, s, o in zip(shape, sp, og))
        return Domain(shape, ext, tuple(axes) if axes else None)

    # ---- geometry -----------------------------------------------------------------------
    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> tuple[float, ...]:
        """Physical side lengths."""
        return tuple(hi - lo for lo, hi in self.extent)

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def spacing(self, shape: Sequence[int] | None = None) -> tuple[float, ...]:
        """Physical cell size per axis at resolution ``shape`` (default native)."""
        shape = self.shape if shape is None else shape_tuple(shape)
        if len(shape) != self.ndim:
            raise ShapeError(f"shape {shape} does not match domain ndim {self.ndim}")
        return tuple((hi - lo) / n for (lo, hi), n in zip(self.extent, shape))

    def at(self, shape: Sequence[int] | int) -> Domain:
        """Same physical extent at a different resolution."""
        shape = shape_tuple(shape)
        if len(shape) != self.ndim:
            raise ShapeError(f"shape {shape} does not match domain ndim {self.ndim}")
        return Domain(shape, self.extent, self.axes)

    def coarsen(self, factor: int = 2, min_size: int = 4) -> Domain:
        return self.at(tuple(max(min_size, s // factor) for s in self.shape))

    def refine(self, factor: int = 2) -> Domain:
        return self.at(tuple(s * factor for s in self.shape))

    # ---- coordinate grids ---------------------------------------------------------------
    def axis_coords(self, shape=None, normalized: bool = True, device=None, dtype=None):
        """List of 1-D cell-center coordinate tensors, one per axis."""
        shape = self.shape if shape is None else shape_tuple(shape)
        out = []
        for (lo, hi), n in zip(self.extent, shape):
            i = torch.arange(n, device=device, dtype=dtype or torch.get_default_dtype())
            if normalized:
                out.append(-1.0 + (2.0 * i + 1.0) / n)
            else:
                out.append(lo + (i + 0.5) * (hi - lo) / n)
        return out

    def coords(self, shape=None, device=None, dtype=None) -> torch.Tensor:
        """Normalized cell-centered grid, shape ``(*shape, ndim)`` with values in ``[-1, 1]``."""
        axes = self.axis_coords(shape, normalized=True, device=device, dtype=dtype)
        mesh = torch.meshgrid(*axes, indexing="ij")
        return torch.stack(mesh, dim=-1)

    def physical_coords(self, shape=None, device=None, dtype=None) -> torch.Tensor:
        """Physical cell-centered grid, shape ``(*shape, ndim)``."""
        axes = self.axis_coords(shape, normalized=False, device=device, dtype=dtype)
        mesh = torch.meshgrid(*axes, indexing="ij")
        return torch.stack(mesh, dim=-1)

    def to_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """Map physical coordinates ``(..., ndim)`` to ``[-1, 1]``."""
        lo = torch.tensor([e[0] for e in self.extent], device=x.device, dtype=x.dtype)
        hi = torch.tensor([e[1] for e in self.extent], device=x.device, dtype=x.dtype)
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    def to_physical(self, u: torch.Tensor) -> torch.Tensor:
        lo = torch.tensor([e[0] for e in self.extent], device=u.device, dtype=u.dtype)
        hi = torch.tensor([e[1] for e in self.extent], device=u.device, dtype=u.dtype)
        return lo + (u + 1.0) * 0.5 * (hi - lo)

    def to_dict(self) -> dict:
        return {
            "shape": list(self.shape),
            "extent": [list(e) for e in self.extent],
            "axes": list(self.axes) if self.axes else None,
        }

    @staticmethod
    def from_dict(d: dict) -> Domain:
        return Domain(
            tuple(d["shape"]),
            tuple(tuple(e) for e in d["extent"]),
            tuple(d["axes"]) if d.get("axes") else None,
        )
