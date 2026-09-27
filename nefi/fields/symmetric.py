"""Exact symmetries by coordinate folding: mirror planes and radial symmetry.

:class:`SymmetricField` wraps any *coordinate-based* field (neural, hash-grid, low-rank,
parametric) and feeds it folded coordinates, so the output is symmetric **by construction** (no
penalty to tune, half the unknowns):

* ``"mirror_x"`` / ``"mirror_y"`` / ``"mirror_z"``: ``x_a ↦ c + |x_a − c|`` on axis 0 / 1 / 2;
  ``"mirror_xy"`` both; ``"mirror"`` with explicit ``axes``.
* ``"radial"``: the inner field only sees ``r = ‖x − c‖`` (rescaled to ``[-1, 1]``) plus any axes
  not included in ``axes`` — so a radial 2-D field wraps a **1-D** inner field (``inner_ndim``).

Raster fields (:class:`~nefi.fields.GridField`) ignore coordinate values, so for them the
wrapper switches to ``mode="average"``: the raw output is averaged with its mirror image (mirror
kinds only).

Example::

    import nefi
    from nefi.fields import Heads, NeuralField, Softplus
    from nefi.fields.symmetric import SymmetricField

    heads = Heads({"x": Softplus()})
    radial = SymmetricField(NeuralField(1, heads, hidden=32, depth=2), kind="radial", ndim=2)
    x = radial(nefi.Domain.unit((32, 32)).coords())["x"]       # rotationally symmetric
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ..errors import ConfigError
from ..registry import register
from .base import Field

_MIRROR_AXES = {"mirror_x": (0,), "mirror_y": (1,), "mirror_z": (2,), "mirror_xy": (0, 1)}
KINDS = (*_MIRROR_AXES, "mirror", "radial")


def symmetry_axes(kind: str, ndim: int, axes: Sequence[int] | None = None) -> tuple[int, ...]:
    """Axes involved in a symmetry ``kind`` for an ``ndim``-D domain (validated).

    Example::

        assert symmetry_axes("mirror_xy", 3) == (0, 1) and symmetry_axes("radial", 2) == (0, 1)
    """
    if kind not in KINDS:
        raise ConfigError(f"unknown symmetry kind {kind!r}; choose one of {KINDS}")
    if kind in _MIRROR_AXES:
        ax = _MIRROR_AXES[kind] if axes is None else tuple(axes)
    elif axes is None:
        if kind == "mirror":
            raise ConfigError("kind='mirror' needs explicit axes=(...)")
        ax = tuple(range(ndim))
    else:
        ax = tuple(axes)
    if not ax or any(not -ndim <= a < ndim for a in ax):
        raise ConfigError(
            f"symmetry {kind!r} uses axes {ax}, which do not exist in a {ndim}-D domain "
            f"(valid: 0..{ndim - 1} or negative indices)"
        )
    return tuple(sorted({a % ndim for a in ax}))


@register("field", "symmetric")
class SymmetricField(Field):
    """Wrap a field so that its output has an exact mirror or radial symmetry.

    Args:
        inner: the wrapped field (for ``"radial"`` it must accept ``inner_ndim``-D coordinates).
        kind: ``"mirror_x" | "mirror_y" | "mirror_z" | "mirror_xy" | "mirror" | "radial"``.
        ndim: domain dimension (default: inferred from ``inner.ndim`` for mirror kinds; required
            for ``"radial"`` unless the wrapped field exposes it).
        axes: axes for ``"mirror"`` (required) or the radial plane (default: all axes).
        center: mirror plane / symmetry center in normalized coordinates (scalar or per axis).
        mode: ``"fold"`` (coordinate folding), ``"average"`` (mirror-average of the raw output, for
            raster fields; center must be 0) or ``"auto"``.

    Example::

        inner = NeuralField(2, Heads({"x": "identity"}), hidden=16, depth=2)
        f = SymmetricField(inner, "mirror_x")
        x = f(nefi.Domain.unit((8, 8)).coords())["x"]
        assert torch.allclose(x, x.flip(0))
    """

    def __init__(
        self,
        inner: Field,
        kind: str = "mirror_x",
        ndim: int | None = None,
        axes: Sequence[int] | None = None,
        center: float | Sequence[float] = 0.0,
        mode: str = "auto",
    ) -> None:
        super().__init__(inner.heads)
        self.inner = inner
        self.kind = kind
        if ndim is None:
            ndim = getattr(inner, "ndim", None)
            if ndim is None and getattr(inner, "shape", None) is not None:
                ndim = len(inner.shape)
            if kind == "radial" or ndim is None:
                raise ConfigError(
                    "SymmetricField needs ndim=<domain dimension> (it cannot be inferred for "
                    f"kind={kind!r} from {type(inner).__name__})"
                )
        self.ndim = int(ndim)
        self.axes = symmetry_axes(kind, self.ndim, axes)
        c = torch.as_tensor(center, dtype=torch.float32)
        self.center = c.expand(self.ndim).clone() if c.ndim == 0 else c.reshape(self.ndim).clone()
        coordinate_free = type(inner).__name__ == "GridField" or getattr(
            inner, "coordinate_free", False
        )
        if mode == "auto":
            mode = "average" if coordinate_free else "fold"
        if mode not in ("fold", "average"):
            raise ConfigError(
                f"SymmetricField mode must be 'fold', 'average' or 'auto', got {mode!r}"
            )
        if mode == "average":
            if kind == "radial":
                raise ConfigError(
                    "radial symmetry of a raster field cannot be imposed by averaging; use a "
                    "coordinate-based representation ('neural', 'hash') or the soft SymmetryLoss"
                )
            if bool((self.center != 0).any()):
                raise ConfigError("mode='average' needs the mirror plane at the domain center")
        elif coordinate_free:
            raise ConfigError(
                f"{type(inner).__name__} ignores coordinate values, so folding cannot impose a "
                "symmetry; use mode='average' (mirror kinds) or a coordinate-based field"
            )
        self.mode = mode
        others = [a for a in range(self.ndim) if a not in self.axes]
        self.inner_ndim = (1 + len(others)) if kind == "radial" else self.ndim
        self._others = tuple(others)
        # max radius over the folded axes in normalized coords (corner of the box)
        self._rmax = math.sqrt(sum((1.0 + abs(float(self.center[a]))) ** 2 for a in self.axes))

    def fold(self, coords: torch.Tensor) -> torch.Tensor:
        """Folded coordinates fed to the inner field."""
        c = self.center.to(coords)
        if self.kind == "radial":
            rel = torch.stack([coords[..., a] - c[a] for a in self.axes], dim=-1)
            r = rel.norm(dim=-1)
            rn = 2.0 * r / self._rmax - 1.0
            cols = [rn] + [coords[..., a] for a in self._others]
            return torch.stack(cols, dim=-1)
        cols = []
        for a in range(self.ndim):
            x = coords[..., a]
            cols.append(c[a] + (x - c[a]).abs() if a in self.axes else x)
        return torch.stack(cols, dim=-1)

    def raw(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        if self.mode == "fold":
            return self.inner.raw(self.fold(coords), progress)
        r = self.inner.raw(coords, progress)
        spatial = r.ndim - 1
        for a in self.axes:
            dim = a - self.ndim - 1 if spatial >= self.ndim else a
            r = 0.5 * (r + r.flip(dim))
        return r

    def on_stage_start(self, stage, domain) -> None:
        if self.mode == "fold" and self.kind == "radial":
            return  # the inner 1-D field has no grid of its own
        self.inner.on_stage_start(stage, domain)

    def reset_parameters(self) -> None:
        self.inner.reset_parameters()

    def extra_repr(self) -> str:
        return f"kind={self.kind}, axes={self.axes}, mode={self.mode}"


__all__ = ["KINDS", "SymmetricField", "symmetry_axes"]
