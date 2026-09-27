"""Gauge / null-space detection and repair: transformations of the unknown the data cannot see.

A *gauge* is a transformation of the unknown that leaves the fitted data unchanged:

* a **global additive constant** — the mean phase of an intensity measurement
  (``|P_z e^{i(φ + c)}|² = |P_z e^{iφ}|²``, inline holography), a potential defined up to a
  constant;
* a **global scale** under a scale-free fidelity — ``F(c·x) = c^p F(x)`` with a max- or
  mean-normalized data term (NeTMY's log-MSE on max-normalized maps; fixed there by the
  energy-anchored scale correction, Eq. 30), or an operator that normalizes internally
  (``p = 0``);
* a **sign flip** of an even operator (``F(−x) = F(x)``, e.g. ``|A x|²`` of a signed field);
* any **candidate direction** the user suspects (per-field constants of a multi-field problem,
  a known null vector of the physics).

Along a gauge the objective is flat, so the reconstruction lands wherever the initialization and
the optimizer's drift put it: correct *up to the gauge*, wrong in absolute terms (holography smoke
run: 34.5 dB mean-subtracted PSNR, 6.8 dB raw PSNR — the phase is offset by −0.69 rad).

:func:`detect_gauges` probes the forward model numerically, at a *generic* point near the current
field (a symmetric initialization such as ``φ ≡ 0`` would make every sign flip look invisible):
the data change along each candidate direction (central differences through the masked
operator) is compared with the largest data change along random smooth / white directions of the
same size. A ratio below ``tol`` means the data cannot see the direction. Scale gauges combine a
homogeneity test of the operator (degree ``p`` measured from ``F(c·x)`` for two values of ``c``)
with a scale-invariance test of the data terms of the final curriculum stage.

:func:`repair_gauges` fixes each unresolved gauge with the cheapest *hard* mechanism: a mean
anchor on the field's head (:class:`~nefi.fields.modifiers.ZeroMean` when the library provides
it, else :class:`MeanAnchor`, which can also anchor to a prior value or to a border region), an
:class:`~nefi.solve.postprocess.EnergyScaleCorrection` with the measured degree (or a
least-squares scale fit for signed data), a mass anchor when the total is known, a sign convention
applied after the fit, or a soft penalty along a custom direction. It never touches the caller's
problem: the returned problem has a deep-copied field and new postprocess / loss containers, and
``problem.meta["gauge_repairs"]`` lists what was done.

Limits: the probe is local (a direction that is invisible at the probe point but visible
elsewhere is reported as a gauge only if it is invisible *there*); discrete gauges (``φ + 2π``),
gauges that act on the grid (translations) and gauges hidden inside nuisance parameters are not
probed; a scale gauge is only detected for data terms that are exactly scale-free.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from ..errors import ConfigError
from ..fields.heads import Head, Heads
from ..losses.base import Context, Loss, LossSet
from ..solve.postprocess import EnergyScaleCorrection, Postprocess
from ..utils.tensor import resample, shape_tuple
from ._common import fmt, md_table

log = logging.getLogger("nefi")

GAUGE_KINDS = ("constant", "scale", "sign", "custom")
FIXES = ("mean_anchor", "energy_scale", "lsq_scale", "mass_anchor", "sign_convention", "penalty")


# ------------------------------------------------------------------------------------------
# fixes: head wrapper, postprocessors, penalty
# ------------------------------------------------------------------------------------------
def border_region(shape: Sequence[int], width: float = 0.1) -> torch.Tensor:
    """Boolean frame of ``max(1, round(width · n))`` cells along every axis of ``shape``."""
    shape = shape_tuple(shape)
    m = torch.zeros(shape, dtype=torch.bool)
    for ax, n in enumerate(shape):
        w = max(1, int(round(width * n)))
        idx = [slice(None)] * len(shape)
        idx[ax] = slice(0, w)
        m[tuple(idx)] = True
        idx[ax] = slice(n - w, n)
        m[tuple(idx)] = True
    return m


class MeanAnchor(Head):
    """``x = inner(h) − mean_R(inner(h)) + value``: fixes an invisible additive constant.

    The mean over the region ``R`` (default: the whole evaluation grid; e.g. a border frame for
    an object in a known uniform background, see :func:`border_region`) is pinned to ``value``
    (a prior level; 0 for a phase). Like :class:`~nefi.fields.modifiers.MassNormalized` it couples
    all grid points, so evaluate the field on full grids (what the solver does). The region is
    given at the native resolution and resampled (nearest) to whatever grid the field is evaluated
    on.

    Args:
        inner: the head defining the field before the gauge fix.
        value: the anchored mean.
        region: optional boolean mask (native grid) over which the mean is taken.

    Example::

        from nefi.fields import Bounded
        head = MeanAnchor(Bounded(-3.0, 3.0), value=0.5)
        x = head(torch.randn(8, 8, 1))
        assert abs(float(x.mean()) - 0.5) < 1e-5
    """

    uses_progress = True

    def __init__(self, inner: Head, value: float = 0.0, region: torch.Tensor | None = None) -> None:
        super().__init__()
        self.inner = inner
        self.n_in = inner.n_in
        self.depends_on = tuple(inner.depends_on)
        self.value = float(value)
        if region is not None:
            r = torch.as_tensor(region).detach().float()
            if float(r.sum()) <= 0:
                raise ConfigError("MeanAnchor region is empty")
            self.register_buffer("region", r, persistent=False)
        else:
            self.region = None
        self._cache: dict[tuple, torch.Tensor] = {}

    def _region_at(self, shape, device, dtype) -> torch.Tensor | None:
        if self.region is None:
            return None
        key = (tuple(shape), str(device), str(dtype))
        if key not in self._cache:
            r = self.region
            if tuple(r.shape) != tuple(shape):
                if r.ndim != len(shape):
                    raise ConfigError(
                        f"MeanAnchor region has {r.ndim} dims, the field is evaluated on {shape}"
                    )
                r = (resample(r, shape, mode="nearest") > 0.5).float()
                if float(r.sum()) <= 0:
                    r = torch.ones(tuple(shape))
            self._cache[key] = r.to(device=device, dtype=dtype)
        return self._cache[key]

    def _apply(self, fn, recurse=True):  # keep the region cache coherent with .to()
        self._cache = {}
        return super()._apply(fn, recurse)

    def transform(self, raw, others, progress=1.0):
        v = self.inner(raw, others, progress)
        r = self._region_at(tuple(v.shape), v.device, v.dtype)
        mean = v.mean() if r is None else (v * r).sum() / r.sum()
        return v - mean + self.value

    def init_bias(self):
        return self.inner.init_bias()

    def inverse(self, value):
        return self.inner.inverse(value)

    @property
    def init_value(self):
        return getattr(self.inner, "init_value", None)


class LeastSquaresScale(Postprocess):
    """Scale correction for signed data: ``α^p = ⟨y, ŷ⟩ / ⟨ŷ, ŷ⟩`` over observed entries.

    The least-squares companion of :class:`~nefi.solve.postprocess.EnergyScaleCorrection`
    (NeTMY Eq. 30), for homogeneous operators of degree ``p`` whose data do not have a positive
    total (e.g. zero-mean traces), where an energy ratio is ill-defined.
    """

    def __init__(self, field: str | None = None, homogeneity: float = 1.0) -> None:
        if not homogeneity:
            raise ConfigError("LeastSquaresScale needs a non-zero homogeneity degree")
        self.field, self.homogeneity = field, float(homogeneity)

    def __call__(self, fields, pred, problem, shape):
        name = self.field or problem.field.primary
        obs = problem.measurement.to(pred.device, pred.dtype)
        if tuple(obs.data.shape) != tuple(pred.shape):
            obs = obs.resampled(pred.shape)
        m = obs.mask.expand_as(pred) if obs.mask is not None else torch.ones_like(pred)
        num = float((obs.data * pred * m).sum())
        den = float((pred * pred * m).sum())
        ratio = num / den if den > 0 else 1.0
        if ratio <= 0 and self.homogeneity % 2 == 0:
            ratio = 1.0
        alpha = math.copysign(abs(ratio) ** (1.0 / self.homogeneity), ratio)
        out = dict(fields)
        out[name] = fields[name] * alpha
        return out, {"scale_factor": float(alpha)}


class SignConvention(Postprocess):
    """Resolve an invisible global sign by a convention applied after the fit.

    Rules: ``"max_abs_positive"`` (the entry of largest magnitude is positive) or
    ``"mean_positive"`` (the mean is non-negative). A convention, not information: pick the one
    that matches your physics.
    """

    RULES = ("max_abs_positive", "mean_positive")

    def __init__(self, field: str | None = None, rule: str = "max_abs_positive") -> None:
        if rule not in self.RULES:
            raise ConfigError(f"unknown sign rule {rule!r}; use {self.RULES}")
        self.field, self.rule = field, rule

    def __call__(self, fields, pred, problem, shape):
        name = self.field or problem.field.primary
        x = fields[name]
        if self.rule == "mean_positive":
            flip = float(x.mean()) < 0
        else:
            flip = float(x.flatten()[x.abs().argmax()]) < 0
        out = dict(fields)
        if flip:
            out[name] = -x
        return out, {"sign_flipped": bool(flip)}


class DirectionPenalty(Loss):
    """Soft gauge fix along a custom direction: ``((⟨x, v⟩ − c₀) / scale)²``.

    ``⟨x, v⟩ = Σ_k mean(x_k · v_k)`` over the fields of the direction (resampled to the current
    grid), ``c₀`` its value at the initial field. The data cannot see the direction, so any
    positive weight pins it without biasing the fit.
    """

    def __init__(
        self,
        direction: Mapping[str, torch.Tensor],
        target: float = 0.0,
        scale: float = 1.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name or "gauge_penalty")
        self.direction = {k: torch.as_tensor(v).detach().float() for k, v in direction.items()}
        self.target, self.scale = float(target), float(scale) or 1.0

    def coefficient(self, fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
        total = None
        for k, v in self.direction.items():
            x = fields[k]
            vv = v if tuple(v.shape) == tuple(x.shape) else resample(v, x.shape)
            c = (x * vv.to(x)).mean()
            total = c if total is None else total + c
        assert total is not None
        return total

    def forward(self, ctx: Context) -> torch.Tensor:
        return ((self.coefficient(ctx.fields) - self.target) / self.scale) ** 2


def _zero_mean_head(inner: Head) -> Head | None:
    """The library's ``ZeroMean`` head (``nefi.fields.modifiers``) when present and working."""
    try:
        from ..fields import modifiers
    except ImportError:  # pragma: no cover
        return None
    cls = getattr(modifiers, "ZeroMean", None)
    if cls is None:
        return None
    try:
        head = cls(inner)
        test = torch.randn(4, 5, inner.n_in)
        if inner.depends_on:
            return head  # cannot test without the other fields; trust the class
        if abs(float(head(test).mean())) > 1e-4:
            return None
        return head
    except Exception as e:  # noqa: BLE001 - incompatible signature -> private wrapper
        log.debug("ZeroMean unavailable (%s); using MeanAnchor", e)
        return None


# ------------------------------------------------------------------------------------------
# report
# ------------------------------------------------------------------------------------------
@dataclass
class Gauge:
    """One probed direction.

    Attributes:
        kind: ``"constant"``, ``"scale"``, ``"sign"`` or ``"custom"``.
        field: field name (``"+"``-joined names for multi-field custom directions).
        name: label, e.g. ``"constant[phase]"``.
        invisibility: data change along the direction / largest data change along random
            directions of the same size (≈ 0: the data cannot see it); for ``"scale"`` the
            relative change of the data terms under ``pred → c·pred`` (or of ``F`` when the
            operator is scale-invariant).
        detected: ``invisibility < tol`` (and, for ``"scale"``, a consistent homogeneity test).
        degree: measured homogeneity degree ``p`` (``"scale"``).
        handled_by: existing mechanism that already fixes it (head, postprocess, loss).
        fix: recommended repair (one of :data:`FIXES`) or ``"none"`` / ``"unresolved"``.
        note: explanation in words.
        direction: the probed direction for ``"custom"`` gauges (``{field: tensor}``).
    """

    kind: str
    field: str
    name: str
    invisibility: float
    detected: bool
    degree: float | None = None
    handled_by: str | None = None
    fix: str = "none"
    note: str = ""
    direction: dict[str, torch.Tensor] | None = field(default=None, repr=False)

    @property
    def unresolved(self) -> bool:
        return self.detected and self.handled_by is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "field": self.field,
            "name": self.name,
            "invisibility": float(self.invisibility),
            "detected": bool(self.detected),
            "degree": self.degree,
            "handled_by": self.handled_by,
            "fix": self.fix,
            "note": self.note,
        }


@dataclass
class GaugeReport:
    """Outcome of :func:`detect_gauges` (``report.unresolved`` is what :func:`repair_gauges`
    fixes; ``report.repairs`` lists what it did)."""

    gauges: list[Gauge]
    tol: float
    probe: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    evaluations: int = 0
    repairs: list[str] = field(default_factory=list)

    @property
    def detected(self) -> list[Gauge]:
        return [g for g in self.gauges if g.detected]

    @property
    def unresolved(self) -> list[Gauge]:
        return [g for g in self.gauges if g.unresolved]

    def __getitem__(self, name: str) -> Gauge:
        for g in self.gauges:
            if g.name == name:
                return g
        raise KeyError(name)

    def summary(self) -> str:
        """One line: what the data cannot see and what fixes it."""
        det = self.detected
        if not det:
            return "no gauge freedom detected (every probed direction is visible in the data)"
        parts = []
        for g in det:
            state = f"handled by {g.handled_by}" if g.handled_by else f"fix: {g.fix}"
            parts.append(f"{g.name} ({state})")
        return "invisible to the data: " + "; ".join(parts)

    def to_markdown(self) -> str:
        rows = [
            [
                g.name,
                g.invisibility,
                "**yes**" if g.detected else "no",
                "—" if g.degree is None else fmt(g.degree),
                g.handled_by or "—",
                g.fix if g.detected else "—",
                g.note,
            ]
            for g in self.gauges
        ]
        head = ["direction", "invisibility", "gauge", "degree", "handled by", "fix", "note"]
        out = [md_table(head, rows), "", f"tol = {self.tol:g}; {self.summary()}."]
        if self.repairs:
            out += ["", "Repairs applied:"] + [f"- {r}" for r in self.repairs]
        return "\n".join(out)

    def __str__(self) -> str:
        return self.to_markdown()

    def to_dict(self) -> dict[str, Any]:
        return {
            "gauges": [g.to_dict() for g in self.gauges],
            "tol": self.tol,
            "probe": {k: v for k, v in self.probe.items() if isinstance(v, int | float | str)},
            "seconds": self.seconds,
            "evaluations": self.evaluations,
            "repairs": list(self.repairs),
            "summary": self.summary(),
        }


# ------------------------------------------------------------------------------------------
# detection
# ------------------------------------------------------------------------------------------
def _head_chain(head: Head):
    h: Any = head
    while isinstance(h, Head):
        yield h
        h = getattr(h, "inner", None)


def _value_range(head: Head) -> tuple[float, float]:
    try:
        from ..fields.modifiers import value_range

        return value_range(head)
    except Exception:  # noqa: BLE001 - exotic heads
        return (-math.inf, math.inf)


def _natural_scale(head: Head, x0: torch.Tensor) -> float:
    """Typical magnitude of a field.

    Half the range of a signed bounded head (a phase in ``[−π, π]``), else the size of the
    current field, else half a finite range, else the head's ``scale`` attribute, else 1.
    """
    lo, hi = _value_range(head)
    if lo < 0 < hi and math.isfinite(lo) and math.isfinite(hi):
        return 0.5 * (hi - lo)
    m = max(float(x0.abs().mean()), float(x0.std()) if x0.numel() > 1 else 0.0)
    if m > 1e-8:
        return m
    if math.isfinite(lo) and math.isfinite(hi):
        return 0.5 * (hi - lo)
    for h in _head_chain(head):
        s = getattr(h, "scale", None)
        if isinstance(s, int | float) and s:
            return abs(float(s))
    return 1.0


def _smooth(shape: tuple[int, ...], gen: torch.Generator, corr: int) -> torch.Tensor:
    coarse = tuple(max(2, min(s, corr)) for s in shape)
    g = torch.randn(coarse, generator=gen, dtype=torch.float64)
    if 1 <= len(shape) <= 3:
        g = resample(g[None], shape, mode="linear")[0]
    else:
        g = torch.randn(shape, generator=gen, dtype=torch.float64)
    return (g - g.mean()) / g.std().clamp_min(1e-12)


def _unit(t: torch.Tensor) -> torch.Tensor:
    return t / t.double().pow(2).mean().sqrt().clamp_min(1e-300).to(t)


class _Probe:
    """The masked forward model at a generic point, with per-field scales."""

    def __init__(self, problem, shape, progress, amplitude, seed) -> None:
        dom = problem.domain if shape is None else problem.domain.at(shape_tuple(shape))
        self.problem, self.dom = problem, dom
        dev, dt = problem.device, problem.dtype
        with torch.no_grad():
            x0 = problem.field(dom.coords(device=dev, dtype=dt), progress)
        self.x0 = {k: v.detach() for k, v in x0.items()}
        self.op = problem.operator.at_resolution(dom.shape)
        meas = problem.measurement_at(dom.shape)
        self.meas = meas.to(dev, dt)
        self.gen = torch.Generator().manual_seed(int(seed))
        self.heads = {k: problem.field.heads[k] for k in problem.field.names}
        self.scale: dict[str, float] = {}
        self.bounds: dict[str, tuple[float, float]] = {}
        self.x: dict[str, torch.Tensor] = {}
        for k, v in self.x0.items():
            head = self.heads[k]
            lo, hi = _value_range(head)
            self.bounds[k] = (lo, hi)
            s = _natural_scale(head, v)
            self.scale[k] = s
            g = _smooth(tuple(v.shape), self.gen, 8).to(v)
            if lo >= 0 and float(v.min()) > 0:  # positive field: multiplicative perturbation
                x = v * torch.exp(amplitude * 3.0 * g)
            else:
                x = v + amplitude * s * g
            if math.isfinite(lo) and math.isfinite(hi):
                pad = 0.02 * (hi - lo)
                x = x.clamp(lo + pad, hi - pad)
            elif math.isfinite(lo):
                x = x.clamp_min(lo)
            self.x[k] = x
        self.mask = None
        if self.meas.mask is not None:
            self.mask = self.meas.mask
        self.evaluations = 0
        self.y = self.F(self.x)

    def clamp(self, k: str, t: torch.Tensor) -> torch.Tensor:
        """Keep a perturbed field inside its head's range (operators may require it, e.g. a
        CFL bound on a sound speed or a positive conductivity)."""
        lo, hi = self.bounds[k]
        if math.isfinite(lo) or math.isfinite(hi):
            return t.clamp(lo if math.isfinite(lo) else None, hi if math.isfinite(hi) else None)
        return t

    def F(self, fields: Mapping[str, torch.Tensor], masked: bool = True) -> torch.Tensor:
        self.evaluations += 1
        with torch.no_grad():
            y = self.op({**self.x, **fields})
        if y.is_complex():
            y = torch.view_as_real(y)
            m = self.mask
            if masked and m is not None:
                m = torch.broadcast_to(m.real if m.is_complex() else m, y.shape[:-1])
                return y * m.unsqueeze(-1).to(y.dtype)
            return y
        if masked and self.mask is not None:
            try:
                return y * self.mask.to(y.dtype)
            except RuntimeError:
                return y
        return y

    def change(self, direction: Mapping[str, torch.Tensor], h: float) -> float:
        """``‖F(x + hΔ) − F(x − hΔ)‖ / 2h`` for a direction in normalized units."""
        plus, minus = {}, {}
        for k, v in direction.items():
            dv = h * self.scale[k] * v.to(self.x[k])
            plus[k] = self.clamp(k, self.x[k] + dv)
            minus[k] = self.clamp(k, self.x[k] - dv)
        d = self.F(plus) - self.F(minus)
        return float(d.double().norm()) / (2.0 * h)

    def reference(self, names: Sequence[str], h: float, n: int = 3) -> float:
        """Largest data change along random directions (smooth ×2, white ×1) on ``names``."""
        best = 0.0
        for i in range(n):
            direction = {}
            for k in names:
                shp = tuple(self.x[k].shape)
                if i < n - 1:
                    g = _smooth(shp, self.gen, 8 if i == 0 else 3)
                else:
                    g = torch.randint(0, 2, shp, generator=self.gen).double() * 2 - 1
                direction[k] = _unit(g) / math.sqrt(len(names))
            best = max(best, self.change(direction, h))
        return best


def _final_losses(problem, curriculum) -> LossSet:
    cur = curriculum or getattr(problem, "curriculum", None)
    if cur is not None and cur.stages and cur.stages[-1].loss_weights:
        return problem.losses.with_weights(cur.stages[-1].loss_weights)
    return problem.losses


def _data_value(losses: LossSet, probe: _Probe, pred: torch.Tensor) -> float:
    ctx = Context(
        dict(probe.x), pred, probe.meas, probe.dom, probe.op, probe.problem.field, None, 0, 1.0
    )
    total = 0.0
    with torch.no_grad():
        for k in losses.data_terms():
            w = losses.weights[k]
            if w != 0.0:
                total += w * float(losses.terms[k](ctx))
    return total


def _soft_pins(losses: LossSet, probe: _Probe, name: str, h: float) -> list[str]:
    """Active non-data terms that change when the field ``name`` is shifted by a constant."""
    data = set(losses.data_terms())
    pins = []
    shifted = dict(probe.x)
    shifted[name] = probe.x[name] + h * probe.scale[name]
    pred = probe.F(probe.x, masked=False)
    for k, term in losses.terms.items():
        if k in data or losses.weights[k] == 0.0:
            continue
        try:
            with torch.no_grad():
                c0 = Context(dict(probe.x), pred, probe.meas, probe.dom, probe.op, None, None)
                c1 = Context(shifted, pred, probe.meas, probe.dom, probe.op, None, None)
                v0, v1 = float(term(c0)), float(term(c1))
        except Exception:  # noqa: BLE001 - terms that need a full context
            continue
        if math.isfinite(v0) and math.isfinite(v1) and abs(v1 - v0) > 1e-3 * max(abs(v0), 1e-12):
            pins.append(k)
    return pins


def _mean_fixed_by(problem, name: str) -> str | None:
    for h in _head_chain(problem.field.heads[name]):
        cls = type(h).__name__
        if cls in ("ZeroMean", "MeanAnchor"):
            return f"{cls} head"
        if cls == "MassNormalized":
            return "MassNormalized head (fixed mean)"
    return _conservation(problem, name)


def _conservation(problem, name: str) -> str | None:
    try:
        from ..losses.physics import Conservation
    except ImportError:  # pragma: no cover
        return None
    for k, t in problem.losses.terms.items():
        if isinstance(t, Conservation) and problem.losses.weights[k] != 0.0:
            fname = getattr(t, "field_name", None)
            if fname in (None, name):
                return f"Conservation loss {k!r}"
    return None


def _scale_fixed_by(problem, name: str) -> str | None:
    for pp in problem.postprocess:
        if isinstance(pp, EnergyScaleCorrection | LeastSquaresScale):
            if getattr(pp, "field", None) in (None, name):
                return type(pp).__name__
    for h in _head_chain(problem.field.heads[name]):
        if type(h).__name__ == "MassNormalized":
            return "MassNormalized head"
    return _conservation(problem, name)


def _sign_fixed_by(problem, name: str) -> str | None:
    for pp in problem.postprocess:
        if isinstance(pp, SignConvention) and getattr(pp, "field", None) in (None, name):
            return "SignConvention"
    return None


def _as_direction(
    spec: Any, probe: _Probe, name: str
) -> dict[str, torch.Tensor]:  # field units -> normalized units
    if callable(spec) and not torch.is_tensor(spec):
        spec = spec(dict(probe.x))
    if torch.is_tensor(spec):
        spec = {probe.problem.field.primary: spec}
    if not isinstance(spec, Mapping) or not spec:
        raise ConfigError(f"candidate {name!r}: expected a tensor, a mapping or a callable")
    out = {}
    for k, v in spec.items():
        if k not in probe.x:
            raise ConfigError(f"candidate {name!r}: unknown field {k!r}; fields: {tuple(probe.x)}")
        t = torch.as_tensor(v).detach().double()
        shp = tuple(probe.x[k].shape)
        if t.numel() == 1:
            t = t.reshape(()).expand(shp).clone()
        elif tuple(t.shape) != shp:
            t = resample(t, shp)
        out[k] = t / probe.scale[k]
    norm = math.sqrt(sum(float(t.pow(2).mean()) for t in out.values()))
    if norm <= 0:
        raise ConfigError(f"candidate {name!r} is the zero direction")
    return {k: t / norm for k, t in out.items()}


def detect_gauges(
    problem: Any,
    *,
    tol: float = 1e-3,
    candidates: Mapping[str, Any] | None = None,
    fields: Sequence[str] | None = None,
    kinds: Sequence[str] = ("constant", "scale", "sign"),
    amplitude: float = 0.1,
    step: float = 0.02,
    curriculum: Any = None,
    shape: Sequence[int] | None = None,
    progress: float = 1.0,
    seed: int = 0,
) -> GaugeReport:
    """Probe the forward model for invariances the data cannot resolve.

    For every field (or those in ``fields``): the additive constant, the global scale and — for
    signed fields — the sign flip; plus every direction in ``candidates``. Cost: a few dozen
    forward evaluations (no gradients).

    Args:
        problem: the :class:`~nefi.problem.InverseProblem` (not modified).
        tol: a direction whose data change is below ``tol ×`` the largest change along random
            directions of the same size is a gauge.
        candidates: extra directions ``name -> direction``: a tensor (primary field), a mapping
            ``{field: tensor}`` (several fields at once, e.g. per-field constants of a
            multi-field problem; scalars broadcast) or a callable ``fields -> mapping``.
            Directions are in field units.
        fields: fields to probe (default: all).
        kinds: built-in probes to run (subset of ``("constant", "scale", "sign")``).
        amplitude: size of the random perturbation that moves the probe point away from the
            (often symmetric) current field, relative to each field's natural scale.
        step: finite-difference step relative to the natural scale.
        curriculum: curriculum whose final stage defines the active data terms (default: the
            problem's).
        shape: probe resolution (default native).
        progress: annealing progress at which the current field is evaluated.
        seed: seed of the probe point and random reference directions.

    Returns:
        :class:`GaugeReport`.

    Example::

        report = detect_gauges(problem)
        print(report.summary())
        fixed = repair_gauges(problem, report)
    """
    t0 = time.perf_counter()
    bad = set(kinds) - set(GAUGE_KINDS[:3])
    if bad:
        raise ConfigError(f"unknown gauge kinds {sorted(bad)}; use {GAUGE_KINDS[:3]}")
    probe = _Probe(problem, shape, progress, float(amplitude), seed)
    names = list(fields or problem.field.names)
    unknown = [k for k in names if k not in probe.x]
    if unknown:
        raise ConfigError(f"unknown fields {unknown}; the field provides {tuple(probe.x)}")
    losses = _final_losses(problem, curriculum)
    gauges: list[Gauge] = []
    h = float(step)
    refs: dict[str, float] = {}

    def ref_for(ks: Sequence[str]) -> float:
        key = "+".join(ks)
        if key not in refs:
            refs[key] = probe.reference(list(ks), h)
        return refs[key]

    for k in names:
        head = problem.field.heads[k]
        lo, hi = _value_range(head)
        ref = ref_for([k])
        if ref <= 0 or not math.isfinite(ref):
            gauges.append(
                Gauge("constant", k, f"field[{k}]", 0.0, False, note="the data do not depend "
                      "on this field at the probe point (random directions are invisible too)")
            )  # fmt: skip
            continue
        # ---- additive constant -------------------------------------------------------------------
        if "constant" in kinds:
            ones = _unit(torch.ones_like(probe.x[k]).double())
            inv = probe.change({k: ones}, h) / ref
            det = inv < tol
            g = Gauge("constant", k, f"constant[{k}]", inv, det)
            if det:
                g.handled_by = _mean_fixed_by(problem, k)
                pins = [] if g.handled_by else _soft_pins(losses, probe, k, h)
                if pins:
                    g.handled_by = f"regularizer {', '.join(pins)} (soft)"
                g.fix = "mean_anchor" if g.handled_by is None else "none"
                g.note = (
                    f"F(x + c) = F(x): the level of {k!r} is set by the initialization and the "
                    "optimizer's drift"
                )
            gauges.append(g)
        # ---- global scale ------------------------------------------------------------------------
        if "scale" in kinds:
            gauges.append(_scale_gauge(problem, probe, losses, k, tol))
        # ---- sign flip (signed fields only) ------------------------------------------------------
        if "sign" in kinds and lo < 0 < hi:
            x = probe.x[k]
            nrm = float(x.double().norm())
            if nrm > 0:
                y0 = probe.y
                flip = float((probe.F({k: probe.clamp(k, -x)}) - y0).double().norm())
                u = _smooth(tuple(x.shape), probe.gen, 8).to(x)
                d = u / float(u.double().norm()) * 2.0 * nrm
                refd = float((probe.F({k: probe.clamp(k, x + d)}) - y0).double().norm())
                inv = flip / refd if refd > 0 else math.inf
                det = inv < tol
                g = Gauge("sign", k, f"sign[{k}]", inv, det)
                if det:
                    g.handled_by = _sign_fixed_by(problem, k)
                    g.fix = "sign_convention" if g.handled_by is None else "none"
                    g.note = "F(−x) = F(x): the sign is a convention"
                gauges.append(g)
    # ---- user candidates -------------------------------------------------------------------------
    for cname, spec in (candidates or {}).items():
        direction = _as_direction(spec, probe, cname)
        ks = list(direction)
        inv = probe.change(direction, h) / max(ref_for(ks), 1e-300)
        det = inv < tol
        g = Gauge("custom", "+".join(ks), f"custom:{cname}", inv, det, direction=direction)
        if det:
            g.fix = "penalty"
            g.note = "user direction invisible to the data"
        gauges.append(g)
    report = GaugeReport(
        gauges,
        float(tol),
        probe={"amplitude": float(amplitude), "step": h, "seed": int(seed)},
        seconds=time.perf_counter() - t0,
        evaluations=probe.evaluations,
    )
    for g in report.detected:
        log.info("gauge %s: invisibility %.2e, %s", g.name, g.invisibility, g.handled_by or g.fix)
    return report


def _scale_gauge(problem, probe: _Probe, losses: LossSet, k: str, tol: float) -> Gauge:
    name = f"scale[{k}]"
    x = probe.x[k]
    y0 = probe.F(probe.x)
    n0 = float(y0.double().norm())
    if float(x.double().norm()) == 0 or n0 == 0:
        return Gauge("scale", k, name, math.inf, False, note="zero field or zero data at probe")
    degrees, resid, changes = [], [], []
    try:
        for c in (0.9, 1.1):
            yc = probe.F({k: c * x})
            if not bool(torch.isfinite(yc).all()):
                raise FloatingPointError("non-finite output")
            dc = float((yc - y0).double().norm())
            changes.append(dc / n0)
            nc = float(yc.double().norm())
            p = math.log(max(nc, 1e-300) / n0) / math.log(c)
            degrees.append(p)
            resid.append(float((yc - (c**p) * y0).double().norm()) / max(dc, 1e-300))
    except Exception as e:  # noqa: BLE001 - e.g. a CFL bound or a solver failure
        return Gauge("scale", k, name, math.inf, False, note=f"scale probe failed ({e})")
    if max(changes) < tol:  # F(c x) = F(x): the operator itself ignores the scale
        g = Gauge("scale", k, name, max(changes), True, degree=0.0)
        g.handled_by = _scale_fixed_by(problem, k)
        g.fix = "mass_anchor" if g.handled_by is None else "none"
        g.note = "F(c·x) = F(x): the operator normalizes the scale away (needs a known total)"
        return g
    p = sum(degrees) / len(degrees)
    homogeneous = max(resid) < 2e-2 and abs(degrees[0] - degrees[1]) < 5e-2 * max(1.0, abs(p))
    if not homogeneous:
        return Gauge(
            "scale", k, name, math.inf, False, note="not homogeneous (the data fix the scale)"
        )
    if abs(p - round(p)) < 5e-2:
        p = float(round(p))
    pred = probe.F(probe.x, masked=False)
    v0 = _data_value(losses, probe, pred)
    rel = max(abs(_data_value(losses, probe, c * pred) - v0) for c in (0.5, 2.0))
    rel /= max(abs(v0), 1e-30)
    det = rel < tol
    g = Gauge("scale", k, name, rel, det, degree=p)
    if det:
        g.handled_by = _scale_fixed_by(problem, k)
        g.fix = "energy_scale" if g.handled_by is None else "none"
        g.note = f"F(c·x) = c^{fmt(p)} F(x) and the data terms are scale-free"
    else:
        g.note = f"homogeneous of degree {fmt(p)}, but the data terms fix the scale"
    return g


# ------------------------------------------------------------------------------------------
# repair
# ------------------------------------------------------------------------------------------
def _prior_level(head: Head, x0: torch.Tensor) -> float:
    """The prior level of a field: its head's ``init_value`` if set, else the initial mean."""
    try:
        from ..fields.modifiers import head_init_value

        v = head_init_value(head)
    except Exception:  # noqa: BLE001 - exotic heads
        v = None
    return float(v) if isinstance(v, int | float) else float(x0.mean())


def _replace_head(fld, name: str, new_head: Head) -> None:
    items = [(k, new_head if k == name else h) for k, h in fld.heads.items()]
    fld.heads = Heads(items, primary=fld.heads.primary)


def _energy_like(problem) -> bool:
    y = problem.measurement.data.detach().double()
    if y.is_complex():
        return False
    m = problem.measurement.mask
    if m is not None:
        w = m.detach().double().expand_as(y)
        s, a = float((y * w).sum()), float((y.abs() * w).sum())
    else:
        s, a = float(y.sum()), float(y.abs().sum())
    return a > 0 and s > 0.5 * a


def repair_gauges(
    problem: Any,
    report: GaugeReport,
    *,
    fixes: Sequence[str] | None = None,
    mean_prior: float | Mapping[str, float] | None = None,
    anchor: str | torch.Tensor | Mapping[str, Any] = "mean",
    mass: Mapping[str, float] | None = None,
    sign_rule: str = "max_abs_positive",
    penalty_weight: float | None = None,
) -> Any:
    """Return a copy of ``problem`` with every unresolved gauge of ``report`` fixed.

    Args:
        problem: the problem (not modified).
        report: from :func:`detect_gauges` (its ``repairs`` list is filled in).
        fixes: restrict to these fix kinds (default: all of :data:`FIXES`).
        mean_prior: anchored mean per field (float for all, or ``{field: value}``); default the
            prior level: the head's ``init_value`` when set (0 for a phase started at 0), else
            the mean of the initial field.
        anchor: ``"mean"`` (spatial mean), ``"border"`` (mean over a 10 % frame: an object in a
            known uniform background) or a boolean region mask; per field as a mapping.
        mass: ``{field: total}`` for scale gauges of operators that normalize the scale away
            (installs a :class:`~nefi.fields.modifiers.MassNormalized` head).
        sign_rule: :class:`SignConvention` rule.
        penalty_weight: weight of :class:`DirectionPenalty` terms for custom gauges (default:
            the data term's value at the initial field, so one natural-scale step costs as much
            as the initial misfit).

    Returns:
        A new :class:`~nefi.problem.InverseProblem` (``meta["gauge_repairs"]`` lists the
        repairs).
    """
    allowed = set(FIXES if fixes is None else fixes)
    bad = allowed - set(FIXES)
    if bad:
        raise ConfigError(f"unknown fixes {sorted(bad)}; use {FIXES}")
    todo = [g for g in report.unresolved if g.fix in allowed]
    if not todo:
        return problem
    fld = copy.deepcopy(problem.field)
    post = list(problem.postprocess)
    terms = dict(problem.losses.terms.items())
    weights = dict(problem.losses.weights)
    done: list[str] = []
    dom = problem.domain
    with torch.no_grad():
        x0 = problem.field(dom.coords(device=problem.device, dtype=problem.dtype), 1.0)
    for g in todo:
        k = g.field
        if g.fix == "mean_anchor":
            default = _prior_level(fld.heads[k], x0[k])
            if isinstance(mean_prior, Mapping):
                value = float(mean_prior.get(k, default))
            elif mean_prior is not None:
                value = float(mean_prior)
            else:
                value = default
            spec = anchor.get(k, "mean") if isinstance(anchor, Mapping) else anchor
            region = None
            if isinstance(spec, str):
                if spec == "border":
                    region = border_region(dom.shape)
                elif spec != "mean":
                    raise ConfigError(f"anchor must be 'mean', 'border' or a mask, got {spec!r}")
            else:
                region = torch.as_tensor(spec).bool()
            inner = fld.heads[k]
            head = _zero_mean_head(inner) if (region is None and value == 0.0) else None
            head = head or MeanAnchor(inner, value, region)
            _replace_head(fld, k, head)
            with torch.no_grad():
                v = fld(dom.coords(device=problem.device, dtype=problem.dtype), 1.0)[k]
            got = float(v.mean()) if region is None else float(v[region].mean())
            if abs(got - value) > 1e-3 * max(1.0, abs(value)):
                raise ConfigError(
                    f"{type(fld).__name__} does not route its output through its heads; the "
                    f"mean anchor of {k!r} had no effect (mean {got:.4g} ≠ {value:.4g})"
                )
            where = "border mean" if isinstance(spec, str) and spec == "border" else "mean"
            if not isinstance(spec, str):
                where = "region mean"
            done.append(f"{g.name}: {type(head).__name__} head ({where} of {k!r} = {value:.4g})")
        elif g.fix in ("energy_scale", "lsq_scale"):
            p = float(g.degree or 1.0)
            if g.fix == "energy_scale" and _energy_like(problem):
                post.append(EnergyScaleCorrection(field=k, homogeneity=p))
                done.append(f"{g.name}: EnergyScaleCorrection(homogeneity={fmt(p)}) postprocess")
            else:
                post.append(LeastSquaresScale(field=k, homogeneity=p))
                done.append(
                    f"{g.name}: LeastSquaresScale(homogeneity={fmt(p)}) postprocess (the data "
                    "have no positive total, so an energy ratio is ill-defined)"
                )
        elif g.fix == "mass_anchor":
            total = (mass or {}).get(k)
            if total is None:
                done.append(
                    f"{g.name}: UNRESOLVED — the operator ignores the scale; pass "
                    f"mass={{'{k}': total}} to anchor it"
                )
                continue
            from ..fields.modifiers import MassNormalized

            lo, _ = _value_range(fld.heads[k])
            if lo < 0:
                done.append(f"{g.name}: UNRESOLVED — a mass anchor needs a non-negative head")
                continue
            vol = 1.0
            for a, b in dom.extent:
                vol *= float(b) - float(a)
            _replace_head(fld, k, MassNormalized(fld.heads[k], total=float(total), volume=vol))
            done.append(f"{g.name}: MassNormalized head (total {float(total):.4g})")
        elif g.fix == "sign_convention":
            post.append(SignConvention(field=k, rule=sign_rule))
            done.append(f"{g.name}: SignConvention({sign_rule!r}) postprocess")
        elif g.fix == "penalty" and g.direction is not None:
            direction = {
                f: t * _natural_scale(fld.heads[f], x0[f]) for f, t in g.direction.items()
            }  # back to field units
            pen = DirectionPenalty(direction)
            with torch.no_grad():
                pen.target = float(pen.coefficient(dict(x0)))
            pen.scale = max(float(sum(float(t.abs().mean()) for t in direction.values())), 1e-12)
            key = f"gauge_{g.name.split(':', 1)[-1]}"
            w = penalty_weight
            if w is None:
                with torch.no_grad():
                    ctx = problem.context()
                    w = max(float(problem.losses.data_loss(problem.losses(ctx)[1])), 1e-12)
            terms[key], weights[key] = pen, float(w)
            done.append(f"{g.name}: DirectionPenalty loss {key!r} (weight {float(w):.3g})")
    new_losses = problem.losses
    if len(terms) != len(problem.losses.terms):
        new_losses = LossSet(terms, weights)
    meta = dict(problem.meta)
    meta["gauge_repairs"] = list(meta.get("gauge_repairs", [])) + done
    new = dataclasses.replace(problem, field=fld, postprocess=post, losses=new_losses, meta=meta)
    for d in done:
        log.info("repair_gauges: %s", d)
    report.repairs.extend(done)
    return new


def strip_gauge_fixes(problem: Any, names: Sequence[str] | None = None) -> Any:
    """Copy of ``problem`` without mean-anchor heads (``ZeroMean`` / :class:`MeanAnchor`).

    For experiments that compare an anchored parameterization with the raw one.
    """
    fld = copy.deepcopy(problem.field)
    for k in names or fld.names:
        head = fld.heads[k]
        if type(head).__name__ in ("ZeroMean", "MeanAnchor"):
            _replace_head(fld, k, head.inner)
    return dataclasses.replace(problem, field=fld, meta=dict(problem.meta))


__all__ = [
    "FIXES",
    "GAUGE_KINDS",
    "DirectionPenalty",
    "Gauge",
    "GaugeReport",
    "LeastSquaresScale",
    "MeanAnchor",
    "SignConvention",
    "border_region",
    "detect_gauges",
    "repair_gauges",
    "strip_gauge_fixes",
]
