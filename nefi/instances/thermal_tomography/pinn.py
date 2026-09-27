"""Soft-constrained PINN baseline for thermal tomography (NeFTY Eq. 4, §3.3, App. C / F.1).

Two networks are optimized jointly: a temperature surrogate ``T_φ(x, t)`` and the diffusivity field
``α_θ(x)`` (the same neural field as NeFTY), against::

    L_PINN = ‖T_φ − T̂‖²_Γobs + λ_PDE ‖∂_t T_φ − ∇·(α_θ ∇T_φ)‖²_Ω + λ_IC ‖T_φ(·, 0) − T0‖²_Ω.

In ``nefi`` terms the surrogate is the *operator* (:class:`SoftPINNSurface`, whose parameters the
solver optimizes alongside the field) and the residual / initial-condition penalties are losses
(:class:`PDEResidualLoss`, :class:`InitialConditionLoss`). Because the predicted surface frames do
not depend on ``θ`` at all, ``∇_θ L_data ≡ 0``: the surface data can reach ``α_θ`` only through the
residual term — the structural decoupling of NeFTY §3.3 / App. C.1 that this baseline exists to
demonstrate. Loss weights are fixed (the paper balances them with GradNorm, App. F.1).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

from ...domain import Domain
from ...errors import ConfigError
from ...fields.encoding import FourierFeatures
from ...losses.base import Context, Loss
from ...losses.data import DataLoss
from ...operators.base import Fields, Operator
from ...utils.tensor import shape_tuple

__all__ = [
    "InitialConditionLoss",
    "NormalizedSurfaceMSE",
    "PDEResidualLoss",
    "SoftPINNSurface",
    "TemperatureNet",
]


class TemperatureNet(nn.Module):
    """``T_φ(x, t)``: tanh MLP on (optionally Fourier-encoded) normalized space-time coordinates.

    Inputs are normalized to ``[-1, 1]`` (space: the domain box; time: ``[0, t_end]``); the output
    is multiplied by ``scale`` (the flash amplitude) so the network works with ``O(1)`` values.
    ``tanh`` keeps the second spatial derivatives needed by the residual non-trivial.
    """

    def __init__(
        self,
        ndim: int,
        hidden: int = 128,
        depth: int = 5,
        n_octaves: int = 0,
        scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.in_dim = int(ndim) + 1
        self.scale = float(scale)
        self.encoding = (
            FourierFeatures(self.in_dim, n_octaves=n_octaves, annealed=False)
            if n_octaves > 0
            else None
        )
        d = self.encoding.out_dim if self.encoding is not None else self.in_dim
        layers: list[nn.Module] = []
        for _ in range(int(depth)):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        """``xt``: ``(..., ndim + 1)`` normalized ``(x, [y,] z, t)`` → temperature ``(...)``."""
        h = self.encoding(xt) if self.encoding is not None else xt
        return self.scale * self.net(h)[..., 0]


class SoftPINNSurface(Operator):
    """The PINN's "forward model": ``T_φ`` read off at the observed surface and frame times.

    The prediction ignores ``α`` (it only declares it as a required field so the problem wiring is
    identical to NeFTY's) — this is exactly the soft-constraint decoupling ``∂T_φ/∂θ ≡ 0``.

    Args:
        domain: slab (last axis = depth); the surface is sampled at the first cell centres, like
            the frames of the finite-volume solvers.
        frame_times: physical times of the observed frames.
        t_end: time normalization horizon.
        tnet: the temperature surrogate (shared across resolutions).
        field: name of the diffusivity field.
    """

    homogeneity = None

    def __init__(
        self,
        domain: Domain,
        frame_times: Sequence[float],
        t_end: float,
        tnet: TemperatureNet,
        field: str = "alpha",
    ) -> None:
        super().__init__()
        self.domain = domain
        self.frame_times = tuple(float(t) for t in frame_times)
        self.t_end = float(t_end)
        self.tnet = tnet
        self.primary = field

    def normalize(self, x_phys: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Physical ``(…, ndim)`` coordinates and times ``(…)`` → normalized ``(…, ndim + 1)``."""
        return torch.cat(
            [self.domain.to_normalized(x_phys), (2.0 * t / self.t_end - 1.0)[..., None]], -1
        )

    def temperature(self, x_phys: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.tnet(self.normalize(x_phys, t))

    def forward(self, fields: Fields) -> torch.Tensor:
        ref = self.get_field(fields)
        c = self.domain.physical_coords(device=ref.device, dtype=ref.dtype)[..., 0, :]  # surface
        t = torch.tensor(self.frame_times, device=ref.device, dtype=ref.dtype)
        cc = c.unsqueeze(0).expand(len(self.frame_times), *c.shape)
        tt = t.view(-1, *([1] * (c.ndim - 1))).expand(cc.shape[:-1])
        return self.temperature(cc, tt)

    def at_resolution(self, shape: Sequence[int]) -> SoftPINNSurface:
        shape = shape_tuple(shape)
        if shape == self.domain.shape:
            return self
        return SoftPINNSurface(
            self.domain.at(shape), self.frame_times, self.t_end, self.tnet, self.primary
        )

    def output_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        return (len(self.frame_times), *shape_tuple(shape)[:-1])


def _pinn_operator(ctx: Context) -> SoftPINNSurface:
    op = ctx.operator
    while op is not None and not isinstance(op, SoftPINNSurface):
        op = getattr(op, "inner", None)
    if op is None:
        raise ConfigError("PINN losses need a SoftPINNSurface operator in the context")
    return op


class PDEResidualLoss(Loss):
    """``mean r²`` with ``r = ∂_t T_φ − α_θ ΔT_φ − ∇α_θ·∇T_φ`` at random collocation points
    (NeFTY Eq. 4 / App. C.1, Eq. 19). Space is sampled uniformly in the slab, time in
    ``(0, t_end]``; derivatives are taken by autograd in physical units.

    Args:
        n_points: collocation points per evaluation (NeFTY App. F.1: 24 576).
        normalize: divide by ``scale²/t_end²`` (the natural size of ``(∂_t T)²``) so the weight is
            dimensionless.
    """

    def __init__(self, n_points: int = 24576, normalize: bool = True, name: str | None = None):
        super().__init__(name)
        self.n_points = int(n_points)
        self.normalize = bool(normalize)

    def forward(self, ctx: Context) -> torch.Tensor:
        op = _pinn_operator(ctx)
        dom = op.domain
        field = ctx.field_module
        if field is None:
            raise ConfigError("PDEResidualLoss needs ctx.field_module (the α network)")
        ref = next(iter(ctx.fields.values()))
        dev, dt = ref.device, ref.dtype
        lo = torch.tensor([e[0] for e in dom.extent], device=dev, dtype=dt)
        hi = torch.tensor([e[1] for e in dom.extent], device=dev, dtype=dt)
        x = lo + (hi - lo) * torch.rand(self.n_points, dom.ndim, device=dev, dtype=dt)
        t = op.t_end * torch.rand(self.n_points, device=dev, dtype=dt).clamp_min(1e-3)
        x.requires_grad_(True)
        t.requires_grad_(True)
        T = op.temperature(x, t)
        gx, gt = torch.autograd.grad(T.sum(), (x, t), create_graph=True)
        lap = torch.zeros_like(T)
        for d in range(dom.ndim):
            lap = lap + torch.autograd.grad(gx[:, d].sum(), x, create_graph=True)[0][:, d]
        alpha = field(dom.to_normalized(x), ctx.progress)[op.primary]
        (ga,) = torch.autograd.grad(alpha.sum(), x, create_graph=True)
        r = gt - alpha * lap - (ga * gx).sum(-1)
        loss = (r**2).mean()
        if self.normalize:
            loss = loss / (op.tnet.scale / op.t_end) ** 2
        return loss


class InitialConditionLoss(Loss):
    """``mean (T_φ(x, 0) − T0(x))²`` on the cell centres of the current grid (NeFTY Eq. 4).

    Args:
        initial: callable ``domain -> T0`` (e.g. :class:`~nefi.operators.pde.GaussianFlash`).
        normalize: divide by ``scale²`` (dimensionless weight).
    """

    def __init__(
        self, initial: Callable[[Domain], torch.Tensor], normalize: bool = True, name=None
    ):
        super().__init__(name)
        self.initial = initial
        self.normalize = bool(normalize)
        self._cache: dict[tuple, torch.Tensor] = {}

    def forward(self, ctx: Context) -> torch.Tensor:
        op = _pinn_operator(ctx)
        dom = op.domain
        ref = next(iter(ctx.fields.values()))
        key = (dom.shape, ref.device, ref.dtype)
        if key not in self._cache:
            self._cache[key] = torch.as_tensor(self.initial(dom)).to(ref.device, ref.dtype)
        T0 = self._cache[key]
        x = dom.physical_coords(device=ref.device, dtype=ref.dtype)
        T = op.temperature(x, torch.zeros(x.shape[:-1], device=ref.device, dtype=ref.dtype))
        loss = ((T - T0) ** 2).mean()
        return loss / op.tnet.scale**2 if self.normalize else loss


class NormalizedSurfaceMSE(DataLoss):
    """``mean (T_φ|_Γ − T̂)² / scale²`` — the PINN data term in the same dimensionless units as the
    residual and initial-condition penalties."""

    def __init__(self, scale: float = 100.0, name: str | None = None) -> None:
        super().__init__(name)
        self.scale = float(scale)

    def forward(self, ctx: Context) -> torch.Tensor:
        return ctx.obs.masked_mean((ctx.pred - ctx.obs.data) ** 2) / self.scale**2
