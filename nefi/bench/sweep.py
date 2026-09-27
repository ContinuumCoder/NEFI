"""One-axis hyperparameter sweeps (NeTMY Tab. 10: learning rate, hidden width, PE octaves).

Every value becomes one method of a :class:`~nefi.bench.protocol.BenchmarkResult` named
``"<axis>=<value>"``; all values share the same measurements and seeds (paired comparison).

Axes:

* an instance-config field, e.g. ``"lr"``, ``"hidden"``, ``"n_octaves"`` (validated up front);
* ``"stage.<attr>"`` — every curriculum stage (``stage.lr``, ``stage.anneal_fraction``);
* ``"curriculum.<attr>"`` — curriculum attributes (``curriculum.restarts``);
* ``"optim.<attr>"`` — optimizer settings (``optim.weight_decay``, ``optim.optimizer``);
* ``"weight.<loss>"`` — a loss weight (``weight.tv``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from ..errors import ConfigError
from .ablation import Modifier, _SetAttrs, set_config, set_weight, variant_method
from .report import format_value


def axis_modifier(instance: Any, axis: str, value: Any) -> Modifier:
    """The :class:`~nefi.bench.ablation.Modifier` that sets ``axis`` to ``value``."""
    head, dot, rest = axis.partition(".")
    if dot and head in ("stage", "curriculum", "optim"):
        return _SetAttrs(head, **{rest: value})
    if dot and head in ("weight", "loss"):
        return set_weight(rest, float(value))
    cfg = getattr(instance, "cfg", None)
    if dataclasses.is_dataclass(cfg):
        known = {f.name for f in dataclasses.fields(cfg)}
        if axis not in known:
            raise ConfigError(
                f"unknown sweep axis {axis!r} for {type(cfg).__name__}; config fields: "
                f"{sorted(known)} (or stage.<attr>, curriculum.<attr>, optim.<attr>, "
                "weight.<loss>)"
            )
    return set_config(**{axis: value})


def _label(axis: str, value: Any) -> str:
    v = format_value(value) if isinstance(value, int | float) else str(value)
    return f"{axis}={v}"


def sweep(
    instance: Any,
    axis: str,
    values: Sequence[Any],
    n_samples: int = 4,
    seeds: Sequence[int] = (0, 1, 2),
    *,
    base: Any = None,
    **bench_kw: Any,
):
    """One-axis sweep: one benchmark method per value of ``axis`` (NeTMY Tab. 10).

    Args:
        instance: the instance (its config provides the defaults of the other axes).
        axis: config field or ``stage.* / curriculum.* / optim.* / weight.*`` (module docs).
        values: values to try.
        n_samples: measurements (shared by all values).
        seeds: optimization seeds (shared by all values).
        base: method being swept (default: the instance's neural method).
        **bench_kw: forwarded to :func:`~nefi.bench.protocol.run_benchmark`.

    Returns:
        A :class:`~nefi.bench.protocol.BenchmarkResult` with methods ``"<axis>=<value>"``.
    """
    from .protocol import run_benchmark

    values = list(values)
    if not values:
        raise ConfigError("sweep needs at least one value")
    labels = [_label(axis, v) for v in values]
    if len(set(labels)) != len(labels):
        raise ConfigError(f"sweep values must be distinct: {values}")
    methods = [
        variant_method(instance, lab, [axis_modifier(instance, axis, v)], base)
        for lab, v in zip(labels, values)
    ]
    res = run_benchmark(instance, methods, n_samples, seeds, **bench_kw)
    res.meta["sweep"] = {"axis": axis, "values": values, "labels": labels}
    out_dir = bench_kw.get("out_dir")
    if out_dir is not None:  # re-save with the sweep metadata
        res.save(out_dir)
    return res


__all__ = ["axis_modifier", "sweep"]
