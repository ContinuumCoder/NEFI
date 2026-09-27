"""Forward sensitivity: per-pixel Jacobian column norms of the operator (NeTMY Eq. 23, (P2)).

For a field ``x`` (flattened to ``n`` pixels) and the operator output ``y = F(x)`` (``m`` values),
the sensitivity of pixel ``i`` is the norm of the ``i``-th Jacobian column,
``s_i = ‖∂F/∂x_i‖ = sqrt((JᵀJ)_ii)``. It tells how strongly the measurement "sees" each pixel:

* on a bounded sensing window it is larger at the center than at the boundary (the finite-window
  center bias (P2) of NeTMY §3.3 / App. C.2), which is why free-density solvers receive a centered
  descent direction even from a uniform initialization;
* deep voxels of a heat-conduction slab have exponentially small sensitivity (NeFTY App. B.4),
  so it doubles as a trust mask for the reconstruction (report uncertainty where it is small).

The diagonal of ``JᵀJ`` is estimated matrix-free with random probes:

* ``method="output"`` (default): ``E_u[(Jᵀu)²] = diag(JᵀJ)`` for Rademacher ``u`` in measurement
  space — reverse-mode only (works with custom adjoints), always non-negative, and its relative
  error ``~ sqrt(2 / n_probes)`` does not grow with the operator's blur width;
* ``method="hutchinson"``: ``E_v[v ⊙ JᵀJ v]`` for Rademacher ``v`` in field space, with
  ``JᵀJ v`` computed as jvp then vjp (the classic Hutchinson diagonal estimator);
* ``exact=True``: one Jacobian-vector product per pixel (small grids only).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ..errors import ConfigError
from ._common import VJP, jvp, normalized_radius, operator_fn, rademacher, resolve_fields


def sensitivity_map(
    problem: Any,
    field: str | None = None,
    shape: Sequence[int] | None = None,
    n_probes: int = 32,
    exact: bool = False,
    *,
    fields: Any = None,
    progress: float = 1.0,
    method: str = "output",
    use_mask: bool = True,
    squared: bool = False,
    seed: int = 0,
    jvp_mode: str = "auto",
    max_exact: int = 16384,
) -> torch.Tensor:
    """Per-pixel Jacobian column norms ``‖∂F/∂x_i‖`` of the operator (NeTMY Eq. 23).

    The operator is linearized at the current field values (or at ``fields``); all other fields
    are held fixed (e.g. the Larmor map when differentiating with respect to the density).

    Args:
        problem: the :class:`~nefi.problem.InverseProblem`.
        field: name of the field to differentiate with respect to (default: primary).
        shape: grid shape (default: native, or the shape of ``fields``).
        n_probes: number of random probes for the stochastic estimators.
        exact: compute every column explicitly (one JVP per pixel; ``numel <= max_exact``).
        fields: optional linearization point (tensor for the primary field, mapping, or Result).
        progress: annealing progress used when evaluating the current field values.
        method: ``"output"`` (squared VJPs of measurement-space probes, default) or
            ``"hutchinson"`` (field-space probes through ``JᵀJ v`` via jvp∘vjp).
        use_mask: restrict the operator output to observed entries (``Measurement.mask``).
        squared: return ``diag(JᵀJ)`` instead of its square root.
        seed: probe seed (the estimate is reproducible).
        jvp_mode: see :func:`nefi.diagnostics._common.jvp`.
        max_exact: safety limit on the number of pixels for ``exact=True``.

    Returns:
        Non-negative tensor shaped like the field.
    """
    x_fields, dom = resolve_fields(problem, fields, shape, progress)
    name = field or problem.field.primary
    fn, x0, _ = operator_fn(problem, x_fields, name, dom, use_mask)
    n = x0.numel()
    if exact:
        if n > max_exact:
            raise ConfigError(
                f"exact sensitivity needs {n} JVPs (> max_exact={max_exact}); use the stochastic "
                "estimator (exact=False) or raise max_exact"
            )
        diag = torch.empty(n, device=x0.device, dtype=x0.dtype)
        mode = jvp_mode
        e = torch.zeros(n, device=x0.device, dtype=x0.dtype)
        for i in range(n):
            e.zero_()
            e[i] = 1.0
            col, mode = jvp(fn, x0, e.view_as(x0), mode)
            diag[i] = (col.double() ** 2).sum().to(diag.dtype)
        diag = diag.view_as(x0)
    else:
        if n_probes < 1:
            raise ConfigError("n_probes must be >= 1")
        gen = torch.Generator().manual_seed(int(seed))
        vjp_op = VJP(fn, x0)
        acc = torch.zeros_like(x0, dtype=torch.float64)
        if method == "output":
            for _ in range(n_probes):
                u = rademacher(vjp_op.output.shape, gen, x0.device, x0.dtype)
                acc += vjp_op(u).double() ** 2
        elif method == "hutchinson":
            mode = jvp_mode
            for _ in range(n_probes):
                v = rademacher(x0.shape, gen, x0.device, x0.dtype)
                jv, mode = jvp(fn, x0, v, mode)
                acc += v.double() * vjp_op(jv).double()
        else:
            raise ConfigError(f"unknown method {method!r}; use 'output' or 'hutchinson'")
        diag = (acc / n_probes).clamp_min(0.0).to(x0.dtype)
    return diag if squared else diag.sqrt()


def center_to_outer_ratio(
    x: torch.Tensor,
    center_radius: float = 0.2,
    ring: tuple[float, float] = (0.8, 1.0),
    axes: Sequence[int] | None = None,
) -> float:
    """Ratio of the mean magnitude in a central disk to the mean magnitude in an outer ring.

    Radii are in normalized units (``r = 1`` touches the middle of the grid faces). The NeTMY
    iter-0 data gradient under F2 has a center-to-outer ratio of 18.29× (§5.3) while the realized
    neural-field update has 1.6× (App. E.5); a ratio near 1 means no center bias.

    Args:
        x: field-shaped tensor (e.g. a sensitivity map or a gradient); magnitudes are used.
        center_radius: radius of the central disk.
        ring: ``(r_in, r_out)`` of the outer ring.
        axes: axes that define the radius (default: all), e.g. ``(0, 1)`` for lateral axes.

    Returns:
        ``mean|x|_center / mean|x|_ring`` (``nan`` if the ring is empty or zero).
    """
    a = x.detach().abs().double()
    r = normalized_radius(a.shape, axes, device=a.device, dtype=a.dtype)
    center = r <= center_radius
    if not bool(center.any()):
        center = r <= r.min() + 1e-12
    outer = (r >= ring[0]) & (r <= ring[1])
    if not bool(outer.any()):
        return float("nan")
    den = float(a[outer].mean())
    if den == 0.0:
        return float("inf") if float(a[center].mean()) > 0 else float("nan")
    return float(a[center].mean()) / den


__all__ = ["center_to_outer_ratio", "sensitivity_map"]
