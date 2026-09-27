"""Acquisition / identifiability report: what the measurement can and cannot determine.

No hyper-parameter can recover what the data do not contain. :func:`acquisition_report`
linearizes the forward model at the current field and measures, in data units against the noise:

* **coverage** — per-pixel sensitivity ``s_i = ‖∂F/∂x_i‖`` (the estimator of
  :func:`nefi.diagnostics.sensitivity_map`: ``E_u[(Jᵀu)²]`` over Rademacher probes ``u``) and, from
  the *same* probes, the sensitivity of block perturbations ``s_B = ‖J 1_B‖`` (``blocks`` cells
  per axis: the gradient is summed over the block before squaring). A block is *above the noise
  floor* when a perturbation of amplitude ``a`` on it changes the data by more than ``snr``
  noise standard deviations, ``a · s_B ≥ snr · σ``. The amplitude ``a`` is data-implied by
  default: the block amplitude that would explain the residual at the current field,
  ``a = ‖y − F(x₀)‖ / ‖s_B‖₂``. Relative coverage is the fraction of pixels whose sensitivity is
  at least ``rel_tol`` × the 95th percentile;
* **conditioning** — the top-``k`` singular values of ``dF/dx`` (Lanczos,
  :func:`nefi.diagnostics.singular_values`): a flat head (``σ_k/σ_1 ≥ 0.5``) means at least ``k``
  combinations of the unknown are constrained within a factor 2 of the best one; a steep head
  means the data pin few combinations — sparse acquisition (few sources / receivers / patterns,
  e.g. 2 × 8 cross-well FWI: ``σ_8/σ_1 = 0.35``; an 8 × 32 surround array: 0.93) or strong
  smoothing;
* **data count** — observed entries against unknowns (10 % random observations of a 32² Poisson
  source give 102 data for 1024 unknowns);
* **where** — boundary bands whose mean sensitivity is lowest (the sides without sensors) and
  contiguous regions below the noise floor (a deep slab under a surface array, the center of an
  EIT body);
* **representation vs. operator** — :func:`nefi.fields.adaptive.match_report` (when available):
  does the field's update kernel pass what the data resolve at the noise level?

The verdict — *well-determined*, *moderately* or *severely under-determined* — is the worst of
the individual criteria, each stated with its number and a recommendation. The report changes
nothing: acquisition is physics and hardware. It says so, recommends where more measurements
would help, and which prior / representation fits what the data support (the tuners then set
that prior's strength by the discrepancy principle).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from ..errors import ConfigError
from ._common import fmt, md_table, resolve_sigma

log = logging.getLogger("nefi")

VERDICTS = ("well-determined", "moderately under-determined", "severely under-determined")
_RANK = {v: i for i, v in enumerate(VERDICTS)}

#: criteria thresholds: (moderate below, severe below)
THRESHOLDS: dict[str, tuple[float, float]] = {
    "data_ratio": (1.0, 0.25),
    "coverage_noise": (0.85, 0.5),
    "coverage_rel": (0.8, 0.5),
    "sv_decay": (0.5, 0.05),
    "modes_fraction": (1.0, 0.5),
}


@dataclass
class AcquisitionReport:
    """Outcome of :func:`acquisition_report` (``str(report)`` is the explanation)."""

    verdict: str
    field: str
    shape: tuple[int, ...]
    n_unknowns: int
    n_data: int
    data_ratio: float
    sigma: float | None
    sigma_source: str
    amplitude: float | None
    amplitude_source: str
    coverage_noise: float | None
    coverage_rel: float
    sv: torch.Tensor | None
    sv_decay: float | None
    sv_flat: int | None
    modes_above_noise: int | None
    sides: dict[str, float] = field(default_factory=dict)
    weak_regions: list[str] = field(default_factory=list)
    criteria: dict[str, str] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    match: dict[str, Any] | None = None
    sensitivity: torch.Tensor | None = field(default=None, repr=False)
    block_snr: torch.Tensor | None = field(default=None, repr=False)
    seconds: float = 0.0

    @property
    def flagged(self) -> bool:
        """True unless the verdict is *well-determined*."""
        return self.verdict != VERDICTS[0]

    def summary(self) -> str:
        return f"{self.verdict}: " + ("; ".join(self.findings[:2]) or "no issue found")

    def to_markdown(self) -> str:
        rows = [
            ["observed data / unknowns", f"{self.n_data} / {self.n_unknowns}", self.data_ratio,
             self.criteria.get("data_ratio", "—")],
            ["blocks above the noise floor", _pct(self.coverage_noise), "",
             self.criteria.get("coverage_noise", "—")],
            ["pixels ≥ 10 % of the p95 sensitivity", _pct(self.coverage_rel), "",
             self.criteria.get("coverage_rel", "—")],
            ["σ_k / σ_1 of dF/dx (top-k)", fmt(self.sv_decay),
             f"{self.sv_flat} within ×2 of σ_1" if self.sv_flat is not None else "",
             self.criteria.get("sv_decay", "—")],
            ["top-k modes above the noise", fmt(self.modes_above_noise), "",
             self.criteria.get("modes_fraction", "—")],
        ]  # fmt: skip
        lines = [
            f"**{self.verdict}** — field `{self.field}` on {tuple(self.shape)}; noise σ = "
            f"{fmt(self.sigma)} ({self.sigma_source}); perturbation amplitude a = "
            f"{fmt(self.amplitude)} ({self.amplitude_source})",
            "",
            md_table(["criterion", "value", "detail", "reading"], rows),
        ]
        if self.sides:
            lines += [
                "",
                "Mean sensitivity of the boundary bands (relative to the best side): "
                + ", ".join(f"{k} {fmt(v)}" for k, v in self.sides.items()),
            ]
        if self.findings:
            lines += ["", "Findings:"] + [f"- {f}" for f in self.findings]
        if self.recommendations:
            lines += ["", "Recommendations:"] + [f"- {r}" for r in self.recommendations]
        if self.match and self.match.get("text"):
            lines += ["", "Representation vs. operator (match report):", "", self.match["text"]]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_markdown()

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "field": self.field,
            "shape": list(self.shape),
            "n_unknowns": self.n_unknowns,
            "n_data": self.n_data,
            "data_ratio": self.data_ratio,
            "sigma": self.sigma,
            "sigma_source": self.sigma_source,
            "amplitude": self.amplitude,
            "amplitude_source": self.amplitude_source,
            "coverage_noise": self.coverage_noise,
            "coverage_rel": self.coverage_rel,
            "singular_values": None if self.sv is None else [float(v) for v in self.sv],
            "sv_decay": self.sv_decay,
            "sv_flat": self.sv_flat,
            "modes_above_noise": self.modes_above_noise,
            "sides": dict(self.sides),
            "weak_regions": list(self.weak_regions),
            "criteria": dict(self.criteria),
            "findings": list(self.findings),
            "recommendations": list(self.recommendations),
            "match": self.match,
            "seconds": self.seconds,
        }


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{100 * v:.0f} %"


def _grade(value: float | None, key: str) -> str:
    if value is None or not math.isfinite(value):
        return VERDICTS[0]
    moderate, severe = THRESHOLDS[key]
    if value < severe:
        return VERDICTS[2]
    if value < moderate:
        return VERDICTS[1]
    return VERDICTS[0]


def _jvp_mode(operator: Any) -> str:
    """Forward-mode AD through long Python loops / custom autograd is slow or unavailable:
    use central differences for non-traceable operators."""
    return "finite_difference" if getattr(operator, "traceable", True) is False else "auto"


def _block_sum(g: torch.Tensor, blocks: Sequence[int]) -> torch.Tensor:
    """Sum of ``g`` over a grid of ``blocks`` (per axis), uneven edges handled by index maps."""
    out = g
    for ax, nb in enumerate(blocks):
        n = g.shape[ax]
        idx = torch.div(torch.arange(n, device=g.device) * nb, n, rounding_mode="floor")
        shape = list(out.shape)
        shape[ax] = nb
        acc = torch.zeros(shape, dtype=out.dtype, device=out.device)
        out = acc.index_add_(ax, idx, out)
    return out


def _axis_names(dom) -> list[str]:
    axes = getattr(dom, "axes", None)
    if axes and len(axes) == dom.ndim:
        return [str(a) for a in axes]
    return [f"axis {i}" for i in range(dom.ndim)]


def _side_profile(s: torch.Tensor, names: Sequence[str]) -> dict[str, float]:
    """Mean sensitivity of the boundary bands (1/8 of each axis), relative to the best side."""
    out: dict[str, float] = {}
    for ax, name in enumerate(names):
        n = s.shape[ax]
        w = max(1, n // 8)
        lo = s.narrow(ax, 0, w).mean()
        hi = s.narrow(ax, n - w, w).mean()
        out[f"low-{name}"] = float(lo)
        out[f"high-{name}"] = float(hi)
    best = max(out.values()) if out else 0.0
    return {k: v / best for k, v in out.items()} if best > 0 else out


def _weak_regions(low: torch.Tensor, names: Sequence[str], extents) -> list[tuple[str, float]]:
    """Contiguous low-coverage bands from each side (≥ 10 % of the axis) and a central region."""
    found: list[tuple[str, float]] = []
    d = low.ndim
    for ax, name in enumerate(names):
        n = low.shape[ax]
        for side in ("low", "high"):
            depth = 0
            for i in range(n):
                j = i if side == "low" else n - 1 - i
                if float(low.narrow(ax, j, 1).float().mean()) >= 0.5:
                    depth += 1
                else:
                    break
            frac = depth / n
            if frac >= 0.1 and depth < n:
                a, b = extents[ax]
                cut = a + (b - a) * (frac if side == "low" else 1.0 - frac)
                op = "<" if side == "low" else ">"
                found.append(
                    (f"the {side}-{name} {100 * frac:.0f} % of the domain ({name} {op} "
                     f"{cut:.3g})", frac)
                )  # fmt: skip
    if not found and float(low.float().mean()) > 0.05:
        grids = torch.meshgrid(*[torch.linspace(-1, 1, s) for s in low.shape], indexing="ij")
        r = torch.sqrt(sum(g**2 for g in grids))
        inner = r < 0.5
        frac_in = float((low & inner).sum()) / max(1, int(low.sum()))
        if frac_in >= 0.6:
            found.append(
                ("the central region (normalized radius < 0.5)", float(low.float().mean()))
            )
        else:
            found.append(("scattered pixels", float(low.float().mean())))
    del d
    return found


def acquisition_report(
    problem: Any,
    k: int = 8,
    *,
    field: str | None = None,
    n_probes: int = 16,
    n_iter: int | None = None,
    blocks: int = 8,
    rel_tol: float = 0.1,
    snr: float = 1.0,
    amplitude: float | None = None,
    sigma: float | None = None,
    match: bool = True,
    jvp_mode: str | None = None,
    seed: int = 0,
) -> AcquisitionReport:
    """Classify how well the acquisition determines the unknown, and say what would help.

    Args:
        problem: the problem, linearized at its current field (not modified).
        k: number of singular values (``0`` skips the spectrum).
        field: field to analyze (default: primary).
        n_probes: Rademacher probes of the sensitivity estimator.
        n_iter: Lanczos iterations (default ``max(3k, k + 16)``: the k-th Ritz value needs
            a Krylov space well beyond k to converge).
        blocks: blocks per axis for the noise-relative coverage.
        rel_tol: relative-coverage threshold (fraction of the 95th-percentile sensitivity).
        snr: noise multiple a block perturbation must exceed to count as resolved.
        amplitude: field perturbation amplitude in field units (default: data-implied).
        sigma: noise level (default: measurement, else estimated).
        match: include :func:`nefi.fields.adaptive.match_report` (skipped on failure).
        jvp_mode: Jacobian-vector product mode (default: central differences for
            non-traceable operators, forward mode otherwise).
        seed: probe seed.

    Returns:
        :class:`AcquisitionReport` (``report.verdict``, ``report.recommendations``).

    Example::

        report = acquisition_report(problem)
        print(report)            # verdict, evidence, what would help
    """
    from ..diagnostics._common import VJP, operator_fn, rademacher, resolve_fields
    from ..diagnostics.spectrum import singular_values

    t0 = time.perf_counter()
    name = field or problem.field.primary
    if name not in problem.field.names:
        raise ConfigError(f"unknown field {name!r}; fields: {problem.field.names}")
    fields, dom = resolve_fields(problem, None, None, 1.0)
    fn, x0, y0 = operator_fn(problem, fields, name, dom, True)
    meas = problem.measurement_at(dom.shape)
    s_val, s_src = resolve_sigma(problem, sigma)
    mode = jvp_mode or _jvp_mode(problem.operator)
    n = int(x0.numel())
    if meas.mask is not None:
        m_obs = int((meas.mask.detach().expand_as(meas.data) > 0).sum())
    else:
        m_obs = int(meas.data.numel())
    if meas.data.is_complex():
        m_obs *= 2
    n_unknowns = sum(int(v.numel()) for v in fields.values())
    # ---- sensitivity: pixels and blocks from the same VJP probes ---------------------------------
    nb = [max(1, min(int(blocks), s)) for s in x0.shape]
    gen = torch.Generator().manual_seed(int(seed))
    vjp = VJP(fn, x0)
    acc = torch.zeros_like(x0, dtype=torch.float64)
    acc_b = torch.zeros(nb, dtype=torch.float64, device=x0.device)
    for _ in range(int(n_probes)):
        u = rademacher(vjp.output.shape, gen, x0.device, vjp.output.dtype)
        g = vjp(u).double()
        acc += g**2
        acc_b += _block_sum(g, nb) ** 2
    sens = (acc / n_probes).sqrt()
    sens_b = (acc_b / n_probes).sqrt()
    q95 = float(torch.quantile(sens.flatten().cpu(), 0.95))
    coverage_rel = float((sens >= rel_tol * q95).double().mean()) if q95 > 0 else 0.0
    # ---- amplitude and noise-relative coverage ---------------------------------------------------
    y = meas.data.to(y0.device, y0.dtype)
    if y.is_complex() or y0.is_complex():
        y, y0r = torch.view_as_real(y), torch.view_as_real(y0)
        resid = y0r - y
    else:
        resid = y0 - y * (meas.mask.to(y) if meas.mask is not None else 1.0)
    r0 = float(resid.double().norm())
    if amplitude is not None:
        a, a_src = float(amplitude), "given"
    else:
        den = float(sens_b.norm())
        a = r0 / den if den > 0 else None
        a_src = "data-implied: explains the residual at the current field"
    coverage_noise = None
    block_snr = None
    if s_val is not None and a is not None and a > 0:
        block_snr = a * sens_b / s_val
        coverage_noise = float((block_snr >= snr).double().mean())
    # ---- singular values -------------------------------------------------------------------------
    sv = sv_decay = None
    sv_flat = modes = None
    if k and k > 0:
        try:
            sv = singular_values(
                problem,
                k=int(k),
                n_iter=int(n_iter or max(3 * k, k + 16)),
                field=name,
                jvp_mode=mode,
                seed=seed,
            )
            sv = sv.detach().double().cpu()
            if float(sv[0]) > 0:
                sv_decay = float(sv[-1] / sv[0])
                sv_flat = int((sv >= 0.5 * sv[0]).sum())
            if s_val is not None and a is not None:
                modes = int((sv * a * math.sqrt(n) >= snr * s_val).sum())
        except Exception as e:  # noqa: BLE001 - reported, the other criteria still apply
            log.warning("acquisition_report: singular values failed (%s)", e)
    # ---- where -----------------------------------------------------------------------------------
    names = _axis_names(dom)
    sides = _side_profile(sens.cpu(), names) if dom.ndim >= 1 else {}
    low = sens.cpu() < rel_tol * q95
    if block_snr is not None:
        up = block_snr.cpu() < snr
        for ax, b in enumerate(nb):
            reps = torch.div(torch.arange(x0.shape[ax]) * b, x0.shape[ax], rounding_mode="floor")
            up = up.index_select(ax, reps)
        low = low | up
    regions = _weak_regions(low, names, dom.extent) if bool(low.any()) else []
    # ---- verdict ---------------------------------------------------------------------------------
    criteria = {
        "data_ratio": _grade(m_obs / max(1, n_unknowns), "data_ratio"),
        "coverage_noise": _grade(coverage_noise, "coverage_noise"),
        "coverage_rel": _grade(coverage_rel, "coverage_rel"),
        "sv_decay": _grade(sv_decay if (sv is not None and len(sv) >= 4) else None, "sv_decay"),
        "modes_fraction": _grade(
            None if modes is None or sv is None else modes / len(sv), "modes_fraction"
        ),
    }
    verdict = max(criteria.values(), key=lambda v: _RANK[v])
    rep = AcquisitionReport(
        verdict=verdict,
        field=name,
        shape=tuple(dom.shape),
        n_unknowns=n_unknowns,
        n_data=m_obs,
        data_ratio=m_obs / max(1, n_unknowns),
        sigma=s_val,
        sigma_source=s_src,
        amplitude=a,
        amplitude_source=a_src,
        coverage_noise=coverage_noise,
        coverage_rel=coverage_rel,
        sv=sv,
        sv_decay=sv_decay,
        sv_flat=sv_flat,
        modes_above_noise=modes,
        sides=sides,
        weak_regions=[r for r, _ in regions],
        criteria=criteria,
        sensitivity=sens.detach().cpu(),
        block_snr=None if block_snr is None else block_snr.detach().cpu(),
    )
    if match:
        rep.match = _match(problem, dom, fields, name, mode, seed)
    _explain(rep, regions, k)
    rep.seconds = time.perf_counter() - t0
    log.info("acquisition_report: %s", rep.summary())
    return rep


def _match(problem, dom, fields, name, mode, seed) -> dict[str, Any] | None:
    try:
        from ..fields.adaptive.spectrum import match_report, operator_spectrum
    except ImportError:  # pragma: no cover - adaptive tools not installed
        return None
    try:
        meas = problem.measurement_at(dom.shape)
        mask = None
        if meas.mask is not None:
            mask = meas.mask.to(problem.device, problem.dtype)
        spec = operator_spectrum(
            problem.operator, dom, fields, 4, name=name, seed=seed, mask=mask, jvp_mode=mode
        )
        mr = match_report(None, problem, n_probes=4, seed=seed, operator_spec=spec)
        return {"verdict": mr.verdict, "ratio": mr.ratio, "text": mr.text}
    except Exception as e:  # noqa: BLE001 - informative only
        log.debug("acquisition_report: match_report skipped (%s)", e)
        return None


def _explain(rep: AcquisitionReport, regions: list[tuple[str, float]], k: int) -> None:
    f, r = rep.findings, rep.recommendations
    bad = {key: v for key, v in rep.criteria.items() if v != VERDICTS[0]}
    if "data_ratio" in bad:
        f.append(
            f"only {rep.n_data} observed values for {rep.n_unknowns} unknowns (ratio "
            f"{rep.data_ratio:.2f}): the data cannot pin every pixel"
        )
        r.append(
            "observe more (a denser mask, more sensors / sources / patterns / views) or rely on "
            "a prior matched to the unknown — sparsity (ℓ1, a sparse prior) for localized "
            "sources, TV for piecewise-constant media, smoothness otherwise — with its strength "
            "set by the discrepancy principle (nefi.autotune.tune_regularization)"
        )
    if "sv_decay" in bad and rep.sv_decay is not None:
        f.append(
            f"the top-{k} singular values of dF/dx fall to {rep.sv_decay:.2f}·σ₁ (only "
            f"{rep.sv_flat} within a factor 2 of the largest): few independent combinations of "
            "the unknown are well constrained"
        )
        r.append(
            "add independent illuminations / views (more sources, receivers, patterns, angles) to "
            "widen the well-determined subspace; with the current acquisition prefer a smoother "
            "or lower-dimensional representation (fewer octaves, a layered / parametric field) "
            "and stronger regularization"
        )
    if "modes_fraction" in bad and rep.modes_above_noise is not None and rep.sv is not None:
        f.append(
            f"only {rep.modes_above_noise} of the top-{len(rep.sv)} modes rise above the noise "
            f"for perturbations of amplitude {fmt(rep.amplitude)}"
        )
        r.append("lower the noise (averaging, longer exposure) or accept a coarser reconstruction")
    below = [(d, frac) for d, frac in regions]
    if ("coverage_noise" in bad or "coverage_rel" in bad) and below:
        desc = "; ".join(d for d, _ in below[:3])
        f.append(f"weakly constrained or below the noise floor: {desc}")
        layered = any(" of the domain (" in d for d, _ in below)
        r.append(
            f"{desc}: no data resolve it — prefer stronger smoothness"
            + (" or a layered representation" if layered else "")
            + " there and treat the reconstruction as prior-driven (report the sensitivity map "
            "as a trust mask)"
        )
    if rep.sides and rep.flagged:
        vals = sorted(rep.sides.items(), key=lambda t: t[1])
        weakest = [s for s, v in vals if v < 0.85]
        if weakest:
            f.append(
                "coverage is weakest near the "
                + ", ".join(f"{s} side ({100 * (1 - rep.sides[s]):.0f} % below the best)"
                            for s in weakest[:2])
            )  # fmt: skip
            r.append(
                "if you can, add sources / receivers (sensors) near the "
                + " and ".join(f"{s} side" for s in weakest[:2])
                + " — e.g. a surround geometry instead of a one-sided array"
            )
    if rep.match and rep.match.get("verdict") in ("over-bandlimited", "under-bandlimited"):
        verdict = rep.match["verdict"]
        f.append(f"match report: the representation is {verdict} for this operator and noise")
        r.append(
            "raise the representation's bandwidth (octaves / basis size) or use a geometry that "
            "carries sharp structure explicitly"
            if verdict == "over-bandlimited"
            else "anneal by the operator (OperatorAwareAnnealing), regularize (TV / smoothness) "
            "or stop by the discrepancy principle: the representation moves frequencies the data "
            "cannot constrain"
        )
    if rep.flagged:
        r.append(
            "auto-tuning cannot change the acquisition: these are design recommendations; the "
            "tuners only adapt the prior (regularization, representation, budget) to what the "
            "data support"
        )
    else:
        f.append(
            f"{rep.n_data} data for {rep.n_unknowns} unknowns, "
            + (f"flat top-{k} spectrum (σ_k/σ₁ = {rep.sv_decay:.2f}), "
               if rep.sv_decay is not None else "")
            + f"{_pct(rep.coverage_noise)} of the blocks above the noise floor"
        )  # fmt: skip


__all__ = ["THRESHOLDS", "VERDICTS", "AcquisitionReport", "acquisition_report"]
