"""Two-phase (binary) fields through a sharpening level-set head.

A level-set parameterization represents a two-material object by the sign of a smooth function
``φ`` (the raw network output): ``x = lo + (hi − lo) · σ(φ / ε)``. A large ``ε`` gives a smooth,
easily optimized field; as ``ε → 0`` the field becomes piecewise constant with values in
``{lo, hi}``. :class:`LevelSetHead` shrinks ``ε`` with the training ``progress`` (the same scalar
that anneals the Fourier features), which is the classic continuation strategy for shape
reconstruction (e.g. inclusions / defects in NDE, NeFTY §5 two-phase phantoms).

Example::

    import nefi
    from nefi.fields.levelset import LevelSetField, interface_length

    field = LevelSetField(2, lo=1.0, hi=3.0, hidden=32, depth=3)   # a NeuralField
    x = field(nefi.Domain.unit((32, 32)).coords(), progress=1.0)["x"]
    per = interface_length(x, spacing=(1 / 32, 1 / 32), lo=1.0, hi=3.0)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ..errors import ConfigError
from ..losses.reg import forward_differences
from ..registry import register
from .heads import Head, Heads
from .neural import NeuralField


@register("head", "level_set")
class LevelSetHead(Head):
    """Two-phase head ``lo + (hi − lo) · σ(φ / ε(progress))`` with a sharpening schedule.

    ``ε`` decays from ``eps_start`` (at ``progress = 0``) to ``eps_end`` (at ``progress = 1``);
    final evaluation (``progress = 1``) uses ``eps_end``. Because the solver resets ``progress``
    at the start of every curriculum stage, the field re-softens briefly at each stage start, which
    helps the fine stage move interfaces.

    Args:
        lo, hi: the two phase values (``hi > lo``).
        eps_start, eps_end: interface softness at the start / end of a stage (in raw units).
        schedule: ``"geometric"`` (default), ``"linear"`` or ``"cosine"`` interpolation of ε.
        init_value: optional initial value in ``(lo, hi)`` (default: the midpoint).

    Example::

        head = LevelSetHead(0.0, 1.0, eps_start=1.0, eps_end=0.01)
        phi = torch.tensor([[-0.1], [0.1]])
        soft, sharp = head(phi, progress=0.0), head(phi, progress=1.0)
        assert (sharp - 0.5).abs().min() > (soft - 0.5).abs().max()
    """

    n_in = 1
    uses_progress = True

    def __init__(
        self,
        lo: float = 0.0,
        hi: float = 1.0,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        schedule: str = "geometric",
        init_value: float | None = None,
    ) -> None:
        super().__init__()
        if not hi > lo:
            raise ConfigError(f"LevelSetHead needs hi > lo, got {(lo, hi)}")
        if not (eps_start > 0 and eps_end > 0):
            raise ConfigError("LevelSetHead needs positive eps_start and eps_end")
        if schedule not in ("geometric", "linear", "cosine"):
            raise ConfigError(
                f"unknown level-set schedule {schedule!r}; use 'geometric', 'linear' or 'cosine'"
            )
        self.lo, self.hi = float(lo), float(hi)
        self.eps_start, self.eps_end = float(eps_start), float(eps_end)
        self.schedule = schedule
        self.init_value = init_value

    def eps(self, progress: float = 1.0) -> float:
        """Interface softness ``ε`` at training ``progress`` in ``[0, 1]``."""
        p = min(max(float(progress), 0.0), 1.0)
        a, b = self.eps_start, self.eps_end
        if self.schedule == "geometric":
            return a * (b / a) ** p
        if self.schedule == "linear":
            return a + (b - a) * p
        return b + (a - b) * 0.5 * (1.0 + math.cos(math.pi * p))

    def transform(self, raw, others, progress: float = 1.0):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(raw[..., 0] / self.eps(progress))

    def phase(self, raw: torch.Tensor) -> torch.Tensor:
        """Boolean phase indicator ``φ > 0`` (True where the field is ``hi``)."""
        return raw[..., 0] > 0

    def init_bias(self):
        if self.init_value is None:
            return None
        u = (float(self.init_value) - self.lo) / (self.hi - self.lo)
        u = min(max(u, 1e-6), 1 - 1e-6)
        return [math.log(u / (1 - u)) * self.eps_start]

    def inverse(self, value):
        u = ((torch.as_tensor(value) - self.lo) / (self.hi - self.lo)).clamp(1e-6, 1 - 1e-6)
        return (torch.logit(u) * self.eps_end).unsqueeze(-1)


@register("field", "level_set")
class LevelSetField(NeuralField):
    """A :class:`~nefi.fields.NeuralField` with a single :class:`LevelSetHead`.

    Args:
        ndim: coordinate dimension.
        lo, hi: phase values.
        name: field name.
        eps_start, eps_end, schedule: see :class:`LevelSetHead`.
        **neural_kw: forwarded to :class:`~nefi.fields.NeuralField` (``hidden``, ``depth``, ...).

    Example::

        f = LevelSetField(2, lo=0.0, hi=1.0, hidden=32, depth=2, n_octaves=4)
        coords = nefi.Domain.unit((16, 16)).coords()
        mask = f.phase_mask(coords)        # (16, 16) bool
    """

    def __init__(
        self,
        ndim: int,
        lo: float = 0.0,
        hi: float = 1.0,
        name: str = "x",
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        schedule: str = "geometric",
        **neural_kw,
    ) -> None:
        heads = Heads({name: LevelSetHead(lo, hi, eps_start, eps_end, schedule)})
        super().__init__(ndim, heads, **neural_kw)

    def phi(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        """The level-set function (raw output) ``(*shape)``."""
        return self.raw(coords, progress)[..., 0]

    def phase_mask(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        """Boolean mask of the ``hi`` phase."""
        return self.phi(coords, progress) > 0


def interface_length(
    field: torch.Tensor,
    spacing: Sequence[float] | None = None,
    lo: float = 0.0,
    hi: float = 1.0,
    periodic_axes: Sequence[int] = (),
    eps: float = 1e-8,
) -> torch.Tensor:
    """Interface measure of a two-phase field (perimeter in 2-D, area in 3-D, #jumps in 1-D).

    Uses the coarea formula ``Per({u > 1/2}) ≈ ∫ |∇u| dV`` with ``u = (x − lo) / (hi − lo)``
    and isotropic forward differences, so it is differentiable and usable as a shape regularizer
    (a perimeter penalty). ``spacing`` is the physical cell size (default: unit cells).

    Example::

        d = nefi.Domain.unit((128, 128))
        r = (d.physical_coords() - 0.5).norm(dim=-1)
        disk = (r < 0.3).float()
        per = interface_length(disk, d.spacing())      # ≈ 2π·0.3 up to metrication error
    """
    x = torch.as_tensor(field)
    if spacing is None:
        spacing = (1.0,) * x.ndim
    if len(spacing) != x.ndim:
        raise ConfigError(f"spacing has {len(spacing)} entries but the field has {x.ndim} dims")
    u = (x - lo) / (hi - lo)
    diffs = forward_differences(u, spacing, periodic_axes)
    g = torch.sqrt(sum(d**2 for d in diffs) + eps**2)
    cell = 1.0
    for h in spacing:
        cell *= float(h)
    return g.sum() * cell


__all__ = ["LevelSetField", "LevelSetHead", "interface_length"]
