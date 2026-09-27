"""Low-dimensional parametric fields (analytic ansätze with a handful of learnable numbers).

When the physics or the experiment says the unknown is "a Gaussian blob", "a few ellipses", or any
closed-form family, a :class:`ParametricField` optimizes the few shape parameters directly through
the same forward operator and losses as a neural field. This is the NeTMY §5.5 setting (Gaussian
density ansatz ``ρ(r; A, σ) = A exp(−‖r‖²/2σ²)`` for the α-RuCl₃ cross-check) and the usual way to
get a well-conditioned first answer — or a Hessian-conditioning diagnostic — before going
nonparametric.

Parameter families are :class:`ParamFn` objects: ``fn(params, coords) -> (*shape)`` (or
``(*shape, n_raw)``) with ``init()`` / ``unpack(params)`` helpers. Two are shipped:
:func:`gaussian_blobs` and :func:`ellipses`. Any callable ``fn(params, coords)`` plus an initial
parameter vector works too.

Example::

    import nefi
    from nefi.fields.parametric import ParametricField, gaussian_blobs

    field = ParametricField(gaussian_blobs(2, ndim=2))
    x = field(nefi.Domain.unit((32, 32)).coords())["x"]
    print(field.unpack())          # {'amplitude': ..., 'center': ..., 'sigma': ...}
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
from torch import nn

from ..errors import ConfigError, ShapeError
from ..registry import register
from .base import Field
from .heads import Heads


class ParamFn:
    """A parametric family ``fn(params, coords) -> Tensor``.

    Subclasses define ``n_params``, ``__call__``, ``init`` (default parameter vector) and
    optionally ``unpack`` (a readable dict).

    Example::

        class Plane(ParamFn):                         # x = a + b·x0 + c·x1
            n_params = 3

            def __call__(self, params, coords):
                return params[0] + (params[1:] * coords).sum(-1)

            def init(self):
                return torch.zeros(3)

        field = ParametricField(Plane())
    """

    n_params: int = 0

    def __call__(self, params: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def init(self) -> torch.Tensor:
        raise NotImplementedError

    def unpack(self, params: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"params": params.detach().clone()}


def _as_rows(x, n: int, d: int, name: str) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim == 0:
        t = t.expand(n, d) if d > 0 else t.expand(n)
    elif t.ndim == 1 and d > 0 and t.shape[0] == d:
        t = t.expand(n, d)
    t = t.reshape(n, d) if d > 0 else t.reshape(n)
    if not torch.isfinite(t).all():
        raise ConfigError(f"{name} contains non-finite values")
    return t.clone()


class GaussianBlobs(ParamFn):
    """Sum of isotropic Gaussians ``Σ_k A_k exp(−‖r − c_k‖² / 2σ_k²)`` in normalized coordinates.

    Parameter layout per blob: ``[log A, c_1..c_d, log σ]`` (``[log A, log σ]`` if
    ``centered=True``), so amplitudes and widths stay positive. Coordinates are the normalized
    ``[-1, 1]`` grid coordinates (σ = 0.1 is 5 % of the domain width).

    Example::

        fn = GaussianBlobs(2, ndim=2, sigmas=[0.1, 0.2])
        x = fn(fn.init(), nefi.Domain.unit((16, 16)).coords())      # (16, 16)
    """

    def __init__(
        self,
        n: int = 1,
        ndim: int = 2,
        centered: bool = False,
        amplitudes: float | Sequence[float] = 1.0,
        centers: Sequence | None = None,
        sigmas: float | Sequence[float] = 0.2,
        seed: int = 0,
    ) -> None:
        if n < 1 or ndim < 1:
            raise ConfigError("gaussian_blobs needs n >= 1 and ndim >= 1")
        self.n, self.ndim, self.centered = int(n), int(ndim), bool(centered)
        self.per = 2 + (0 if centered else self.ndim)
        self.n_params = self.n * self.per
        self._amp = _as_rows(amplitudes, self.n, 0, "amplitudes")
        self._sig = _as_rows(sigmas, self.n, 0, "sigmas")
        if centers is None:
            g = torch.Generator().manual_seed(seed)
            self._ctr = (torch.rand(self.n, self.ndim, generator=g) - 0.5) * 1.0
        else:
            self._ctr = _as_rows(centers, self.n, self.ndim, "centers")

    def init(self) -> torch.Tensor:
        rows = [torch.log(self._amp.clamp_min(1e-12)).unsqueeze(1)]
        if not self.centered:
            rows.append(self._ctr)
        rows.append(torch.log(self._sig.clamp_min(1e-6)).unsqueeze(1))
        return torch.cat(rows, dim=1).reshape(-1)

    def _split(self, params: torch.Tensor):
        p = params.reshape(self.n, self.per)
        amp = torch.exp(p[:, 0])
        if self.centered:
            ctr = torch.zeros(self.n, self.ndim, device=p.device, dtype=p.dtype)
        else:
            ctr = p[:, 1 : 1 + self.ndim]
        sig = torch.exp(p[:, -1])
        return amp, ctr, sig

    def __call__(self, params, coords):
        if coords.shape[-1] != self.ndim:
            raise ShapeError(f"gaussian_blobs(ndim={self.ndim}) got {coords.shape[-1]}-D coords")
        amp, ctr, sig = self._split(params.to(coords.dtype))
        r2 = ((coords.unsqueeze(-2) - ctr) ** 2).sum(-1)  # (*shape, n)
        return (amp * torch.exp(-0.5 * r2 / sig**2)).sum(-1)

    def unpack(self, params):
        amp, ctr, sig = self._split(params.detach())
        return {"amplitude": amp.clone(), "center": ctr.clone(), "sigma": sig.clone()}


class Ellipses(ParamFn):
    """Sum of smooth ellipse indicators ``Σ_k A_k σ((1 − q_k(r)) / sharpness)``.

    ``q_k(r) = Σ_i ((R_k (r − c_k))_i / a_{k,i})²`` with a rotation ``R_k`` in 2-D (axis-aligned
    ellipsoids / intervals in 3-D / 1-D). Parameter layout per ellipse:
    ``[A, c_1..c_d, log a_1..log a_d, (θ in 2-D)]``. Amplitudes are signed (phantoms such as
    Shepp–Logan use negative inner ellipses).

    Example::

        fn = Ellipses(1, ndim=2, radii=[0.5, 0.3])
        x = fn(fn.init(), nefi.Domain.unit((16, 16)).coords())      # (16, 16) smooth ellipse
    """

    def __init__(
        self,
        n: int = 1,
        ndim: int = 2,
        sharpness: float = 0.05,
        amplitudes: float | Sequence[float] = 1.0,
        centers: Sequence | None = None,
        radii: float | Sequence = 0.3,
        seed: int = 0,
    ) -> None:
        if n < 1 or ndim < 1:
            raise ConfigError("ellipses needs n >= 1 and ndim >= 1")
        self.n, self.ndim = int(n), int(ndim)
        self.sharpness = float(sharpness)
        self.rotate = self.ndim == 2
        self.per = 1 + 2 * self.ndim + (1 if self.rotate else 0)
        self.n_params = self.n * self.per
        self._amp = _as_rows(amplitudes, self.n, 0, "amplitudes")
        self._rad = _as_rows(radii, self.n, self.ndim, "radii")
        if centers is None:
            g = torch.Generator().manual_seed(seed)
            self._ctr = (torch.rand(self.n, self.ndim, generator=g) - 0.5) * 0.8
        else:
            self._ctr = _as_rows(centers, self.n, self.ndim, "centers")

    def init(self) -> torch.Tensor:
        rows = [self._amp.unsqueeze(1), self._ctr, torch.log(self._rad.clamp_min(1e-6))]
        if self.rotate:
            rows.append(torch.zeros(self.n, 1))
        return torch.cat(rows, dim=1).reshape(-1)

    def _split(self, params):
        p = params.reshape(self.n, self.per)
        d = self.ndim
        amp = p[:, 0]
        ctr = p[:, 1 : 1 + d]
        rad = torch.exp(p[:, 1 + d : 1 + 2 * d])
        theta = p[:, -1] if self.rotate else None
        return amp, ctr, rad, theta

    def __call__(self, params, coords):
        if coords.shape[-1] != self.ndim:
            raise ShapeError(f"ellipses(ndim={self.ndim}) got {coords.shape[-1]}-D coords")
        amp, ctr, rad, theta = self._split(params.to(coords.dtype))
        r = coords.unsqueeze(-2) - ctr  # (*shape, n, d)
        if theta is not None:
            c, s = torch.cos(theta), torch.sin(theta)
            r = torch.stack([c * r[..., 0] + s * r[..., 1], -s * r[..., 0] + c * r[..., 1]], -1)
        q = ((r / rad) ** 2).sum(-1)
        return (amp * torch.sigmoid((1.0 - q) / self.sharpness)).sum(-1)

    def unpack(self, params):
        amp, ctr, rad, theta = self._split(params.detach())
        out = {"amplitude": amp.clone(), "center": ctr.clone(), "radii": rad.clone()}
        if theta is not None:
            out["theta"] = theta.clone()
        return out


def gaussian_blobs(n: int = 1, ndim: int = 2, **kw) -> GaussianBlobs:
    """``n`` isotropic Gaussian blobs (NeTMY §5.5 ansatz when ``n=1, centered=True``).

    Example::

        fn = gaussian_blobs(1, ndim=2, centered=True, sigmas=0.2)
        field = ParametricField(fn)          # 2 parameters: log A, log σ
    """
    return GaussianBlobs(n, ndim, **kw)


def ellipses(n: int = 1, ndim: int = 2, **kw) -> Ellipses:
    """``n`` smooth ellipses (rotated in 2-D, axis-aligned ellipsoids in 3-D).

    Example::

        field = ParametricField(ellipses(3, ndim=2, sharpness=0.05))
    """
    return Ellipses(n, ndim, **kw)


@register("field", "parametric")
class ParametricField(Field):
    """A field given by a closed-form family with a learnable parameter vector.

    Args:
        fn: a :class:`ParamFn` or any callable ``fn(params, coords) -> (*shape)`` /
            ``(*shape, n_raw)``.
        init: initial parameter vector (required when ``fn`` is a plain callable).
        heads: output heads applied to ``fn``'s output (default identity: the family already
            encodes the physics, e.g. positive amplitudes).
        bounds: optional ``(lo, hi)`` tensors/floats clamping the parameters after each forward
            (soft safety; use the family's log-parameterization for hard positivity).

    Example::

        field = ParametricField(lambda p, c: p[0] * torch.exp(-(c**2).sum(-1) / p[1] ** 2),
                                init=torch.tensor([1.0, 0.3]))
    """

    def __init__(
        self,
        fn: ParamFn | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        init: torch.Tensor | Sequence[float] | None = None,
        heads: Heads | Mapping | None = None,
        bounds: tuple | None = None,
    ) -> None:
        super().__init__(heads)
        self.fn = fn
        if init is None:
            if not hasattr(fn, "init"):
                raise ConfigError(
                    "ParametricField(fn) needs init=... when fn is a plain callable "
                    "(or use a ParamFn such as gaussian_blobs(n))"
                )
            init = fn.init()
        p0 = torch.as_tensor(init, dtype=torch.float32).reshape(-1).clone()
        self.register_buffer("_init", p0.clone())
        self.params = nn.Parameter(p0.clone())
        self.bounds = bounds

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.params.copy_(self._init.to(self.params))

    def raw(self, coords: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        p = self.params
        if self.bounds is not None:
            lo, hi = (torch.as_tensor(b).to(p) for b in self.bounds)
            p = torch.maximum(torch.minimum(p, hi), lo)
        y = self.fn(p, coords)
        n = self.heads.n_in
        if y.shape == coords.shape[:-1]:
            y = y.unsqueeze(-1)
        if tuple(y.shape) != (*coords.shape[:-1], n):
            raise ShapeError(
                f"parametric fn returned shape {tuple(y.shape)}; expected "
                f"{(*coords.shape[:-1],)} or {(*coords.shape[:-1], n)} for heads {self.heads.names}"
            )
        return y

    def unpack(self) -> dict[str, torch.Tensor]:
        """Readable parameters (via ``fn.unpack`` when available)."""
        if hasattr(self.fn, "unpack"):
            return self.fn.unpack(self.params)
        return {"params": self.params.detach().clone()}

    def extra_repr(self) -> str:
        return f"fn={type(self.fn).__name__}, n_params={self.params.numel()}"


__all__ = ["Ellipses", "GaussianBlobs", "ParamFn", "ParametricField", "ellipses", "gaussian_blobs"]
