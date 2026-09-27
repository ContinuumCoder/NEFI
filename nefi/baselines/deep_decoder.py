"""Untrained convolutional decoder prior — the *DeepDecoder* baseline (Heckel & Hand 2018).

A fixed random latent tensor ``B_0`` of shape ``(c_0, *latent)`` is mapped to the field by
``n_stages`` blocks ``B_{i+1} = cn(ReLU(U_i(C_i B_i)))`` — a 1×1 convolution ``C_i`` (channel
mixing only), (bi/tri)linear upsampling ``U_i``, ReLU and channel normalization ``cn`` — followed
by a final 1×1 convolution to the heads' raw channels. Only the 1×1 convolutions and the norm's
affine parameters are optimized; the architecture itself is the prior (an under-parameterized
image model with no training data, Heckel & Hand, ICLR 2019).

NeTMY App. E.2 setting: 5 upsampling stages, channel widths ``[128]*4 + [1]`` (four hidden widths
plus the one-channel output), bilinear upsampling, Adam lr 1e-3 for 5000 steps. Here ``width``
sets every hidden width and the output width is ``heads.n_in``.

Resolution handling: the stage sizes grow geometrically from the latent shape to the field's
native ``shape`` (exactly ``×2`` per stage when ``shape = latent · 2^n_stages``); queries at any
other grid are served by resampling the native output (area when coarser, linear when finer),
so ``raw(coords)`` always matches ``coords.shape[:-1]`` and multiscale curricula see a consistent
function. Works in 1-D/2-D/3-D via ``nn.Conv{1,2,3}d``.

Filtering view (NeTMY §4.4): the decoder Jacobian columns are smooth, spatially extended
upsampling patterns, so ``G_θ = J_θ J_θᵀ`` is a *global low-pass* filter (strong smoothness bias,
poor for point-like sources).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..errors import ConfigError, ShapeError
from ..fields.base import Field
from ..fields.heads import Heads
from ..registry import register
from ..utils.tensor import resample, shape_tuple

_CONV = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
_INTERP = {1: "linear", 2: "bilinear", 3: "trilinear"}


class ChannelNorm(nn.Module):
    """Per-channel standardization over the spatial dims with a learnable affine map.

    Equivalent to ``BatchNorm`` with a batch of one in training mode (Heckel & Hand), but without
    running statistics, so the field is a pure function of its parameters (train/eval agnostic).
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.fill_(1.0)
            self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(2, x.ndim))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        xn = (x - mean) / torch.sqrt(var + self.eps)
        view = (1, -1) + (1,) * (x.ndim - 2)
        return xn * self.weight.view(view) + self.bias.view(view)


def stage_sizes(
    latent: Sequence[int], shape: Sequence[int], n_stages: int
) -> list[tuple[int, ...]]:
    """Geometric size schedule from ``latent`` to ``shape`` over ``n_stages`` upsamplings."""
    out = []
    for i in range(1, n_stages + 1):
        t = i / n_stages
        out.append(tuple(max(1, round(lt * (s / lt) ** t)) for lt, s in zip(latent, shape)))
    out[-1] = tuple(shape)
    return out


@register("field", "deep_decoder")
class DeepDecoderField(Field):
    """Deep Decoder field (Heckel & Hand 2018; NeTMY App. E.2 baseline).

    Args:
        shape: native output grid (usually the final curriculum resolution).
        heads: output heads applied to the decoder output channels.
        n_stages: number of [1×1 conv → upsample → ReLU → channel norm] blocks (NeTMY: 5).
        width: hidden channel width of every block (NeTMY: 128).
        channels: explicit per-block output widths (overrides ``width``; length ``n_stages``).
        latent_shape: spatial shape of the fixed random input; default
            ``max(min_latent, round(n / 2^n_stages))`` per axis.
        min_latent: lower bound on the latent size per axis (a 1-pixel latent cannot vary).
        upsample: ``"linear"`` (bi/trilinear, ``align_corners=False``) or ``"nearest"``.
        out_init_scale: scale on the initial output-layer weights so the initial field is close to
            the heads' initial value (the bias is set to ``heads.init_bias()``).
        latent_scale: the latent is ``U[0, latent_scale)`` (Heckel & Hand use 0.1).
        norm_eps: channel-norm epsilon.
        seed: seed of the fixed latent (independent of the global RNG).
    """

    def __init__(
        self,
        shape: Sequence[int],
        heads: Heads | Mapping | None = None,
        n_stages: int = 5,
        width: int = 128,
        channels: Sequence[int] | None = None,
        latent_shape: Sequence[int] | None = None,
        min_latent: int = 4,
        upsample: str = "linear",
        out_init_scale: float = 0.1,
        latent_scale: float = 0.1,
        norm_eps: float = 1e-5,
        seed: int = 0,
    ) -> None:
        super().__init__(heads)
        self.shape = shape_tuple(shape)
        self.ndim = len(self.shape)
        if self.ndim not in _CONV:
            raise ShapeError(f"DeepDecoderField supports 1-3 dims, got shape {self.shape}")
        if n_stages < 1:
            raise ConfigError("n_stages must be >= 1")
        if upsample not in ("linear", "nearest"):
            raise ConfigError(f"upsample must be 'linear' or 'nearest', got {upsample!r}")
        chans = [int(width)] * n_stages if channels is None else [int(c) for c in channels]
        if len(chans) != n_stages:
            raise ConfigError(f"channels needs {n_stages} entries, got {len(chans)}")
        self.n_stages = int(n_stages)
        self.channels = chans
        self.upsample = upsample
        self.out_init_scale = float(out_init_scale)
        if latent_shape is None:
            f = 2**self.n_stages
            latent_shape = tuple(max(int(min_latent), round(s / f)) for s in self.shape)
        self.latent_shape = shape_tuple(latent_shape)
        if len(self.latent_shape) != self.ndim:
            raise ShapeError(f"latent_shape {self.latent_shape} does not match shape {self.shape}")
        self.sizes = stage_sizes(self.latent_shape, self.shape, self.n_stages)
        gen = torch.Generator().manual_seed(int(seed))
        latent = torch.rand((1, chans[0], *self.latent_shape), generator=gen) * float(latent_scale)
        self.register_buffer("latent", latent)
        conv = _CONV[self.ndim]
        c_in = chans[0]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for c_out in chans:
            self.convs.append(conv(c_in, c_out, kernel_size=1))
            self.norms.append(ChannelNorm(c_out, norm_eps))
            c_in = c_out
        self.out = conv(c_in, self.heads.n_in, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-draw the 1×1 convolutions (global RNG); the latent input stays fixed."""
        for c in self.convs:
            c.reset_parameters()
        for n in self.norms:
            n.reset_parameters()
        self.out.reset_parameters()
        with torch.no_grad():
            self.out.weight.mul_(self.out_init_scale)
            self.out.bias.copy_(self.heads.init_bias().to(self.out.bias))

    def decode(self) -> torch.Tensor:
        """Native decoder output ``(n_in, *shape)``."""
        x = self.latent
        mode = _INTERP[self.ndim] if self.upsample == "linear" else "nearest"
        for conv, norm, size in zip(self.convs, self.norms, self.sizes):
            x = conv(x)
            if mode == "nearest":
                x = F.interpolate(x, size=size, mode="nearest")
            else:
                x = F.interpolate(x, size=size, mode=mode, align_corners=False)
            x = norm(F.relu(x))
        return self.out(x)[0]

    def raw(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        target = tuple(coords.shape[:-1])
        if len(target) != self.ndim:
            raise ShapeError(
                f"DeepDecoderField({self.shape}) needs a {self.ndim}-D coordinate grid, got "
                f"coords of shape {tuple(coords.shape)}"
            )
        y = self.decode().to(coords.dtype)
        if target != self.shape:
            y = resample(y, target)
        return y.movedim(0, -1)

    def extra_repr(self) -> str:
        return (
            f"shape={self.shape}, latent={self.latent_shape}, channels={self.channels}, "
            f"heads={self.heads.names}"
        )


__all__ = ["ChannelNorm", "DeepDecoderField", "stage_sizes"]
