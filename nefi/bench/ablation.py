"""Component ablations (NeTMY Tab. 3 / App. E.6, NeFTY Tab. 3 / Tab. 8).

A *modifier* changes one component of a method — the instance config, the built
:class:`~nefi.problem.InverseProblem`, or the :class:`~nefi.solve.Curriculum`. A cumulative
ablation applies modifiers ``1..k`` for row ``k`` (NeTMY: "each row removes one component
cumulatively"), reusing the same measurements and seeds for every row so differences are paired.

Ready-made modifiers for the generic components:

==============================  ===============================================================
``no_annealing()``              β = K from step 0 in every stage (NeTMY "−annealed PE", NeFTY FA)
``no_positional_encoding()``    raw coordinates into the MLP (NeTMY "−PE", NeFTY PE)
``single_stage()``              one stage at the final resolution (NeTMY "−multiscale")
``drop_loss(name)``             weight 0 for a loss term everywhere (NeTMY "−ℓ1", "−R_ds", "−TV")
``set_weight(name, w)``         set a loss weight everywhere
``grid_field(lr_scale=1)``      free-pixel field instead of the neural field (Grid Opt.)
``no_gate()``                   gated softplus → softplus (NeTMY "−gate")
``set_config(**kw)``            instance-config overrides (inversion-side settings only)
``set_stage(**kw)``             attributes of every curriculum stage
``set_curriculum(**kw)``        curriculum attributes (restarts, early stopping, ...)
``set_optim(**kw)``             optimizer settings (optimizer, weight_decay, grad_clip, ema)
==============================  ===============================================================

Strings are parsed by :func:`parse_modifier` (``"drop_loss:tv"``, ``"set:n_octaves=4"``,
``"stage:anneal_fraction=0.25"``, ...), which is what the CLI ``nefi ablate --variants`` uses.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import yaml

from ..errors import ConfigError
from ..fields.encoding import IdentityEncoding
from ..fields.grid import GridField
from ..fields.heads import GatedSoftplus, Heads, Softplus
from ..fields.neural import NeuralField
from ..solve.curriculum import Curriculum

log = logging.getLogger("nefi")


def parse_value(text: str) -> Any:
    """Parse a CLI value: int, float (``1e-3`` too), bool, ``none``, YAML list/dict, or string."""
    s = text.strip()
    low = s.lower()
    if low in ("none", "null", "~"):
        return None
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    for conv in (int, float):
        try:
            return conv(s)
        except ValueError:
            pass
    if s[:1] in "[{(":
        try:
            v = yaml.safe_load(s.replace("(", "[").replace(")", "]"))
            return _numberize(v)
        except yaml.YAMLError:
            pass
    return s


def _numberize(v: Any) -> Any:
    if isinstance(v, list):
        return [_numberize(x) for x in v]
    if isinstance(v, dict):
        return {k: _numberize(x) for k, x in v.items()}
    if isinstance(v, str):
        return parse_value(v)
    return v


# ------------------------------------------------------------------------------------------
# modifiers
# ------------------------------------------------------------------------------------------
class Modifier:
    """Changes one component of a method. Override any of the hooks.

    :meth:`apply` (called by the benchmark) runs :meth:`problem` then :meth:`curriculum`;
    :meth:`config` returns instance-config overrides applied before the problem is built.

    Attributes:
        name: identifier (used in descriptions and JSON).
        label: display label for table rows (NeTMY-style, e.g. ``"−PE"``).
    """

    name: str = "modifier"
    label: str = "modifier"

    def config(self, cfg: Any) -> dict[str, Any]:
        """Instance-config overrides (applied before the problem is built)."""
        return {}

    def problem(self, problem: Any) -> Any:
        """Return the (possibly new) problem; in-place changes may return ``None``."""
        return problem

    def curriculum(self, curriculum: Curriculum) -> Curriculum:
        """Return the (possibly new) curriculum; in-place changes may return ``None``."""
        return curriculum

    def apply(self, problem: Any, curriculum: Curriculum) -> tuple[Any, Curriculum]:
        """Apply the problem and curriculum hooks."""
        problem = self.problem(problem) or problem
        curriculum = self.curriculum(curriculum) or curriculum
        return problem, curriculum

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


class FnModifier(Modifier):
    """Wrap ``fn(problem, curriculum) -> (problem, curriculum) | None`` (in-place allowed)."""

    def __init__(self, fn: Callable[[Any, Curriculum], Any], name: str | None = None) -> None:
        self.fn = fn
        self.name = self.label = name or getattr(fn, "__name__", "custom")

    def apply(self, problem, curriculum):
        out = self.fn(problem, curriculum)
        if out is None:
            return problem, curriculum
        if isinstance(out, tuple) and len(out) == 2:
            return out
        raise ConfigError(f"{self.name}: must return (problem, curriculum) or None")


class _NoAnnealing(Modifier):
    name, label = "no_annealing", "−annealing"

    def curriculum(self, curriculum):
        for s in curriculum.stages:
            s.anneal = False
        return curriculum


def _neural_kwargs(field: NeuralField) -> dict[str, Any]:
    return {
        "ndim": field.ndim,
        "heads": field.heads,
        "hidden": field.hidden,
        "depth": field.depth,
        "skip_at": field.skip_at,
        "activation": field.activation,
        "encoding": copy.deepcopy(field.encoding),
        "out_init_scale": field.out_init_scale,
        "out_bias": None if field._out_bias is None else field._out_bias.tolist(),
    }


def rebuild_field(field: Any, heads: Heads | None = None, encoding: Any = None) -> Any:
    """A fresh field of the same kind/settings with new ``heads`` and/or ``encoding``.

    Supports :class:`NeuralField` and :class:`GridField`; other fields raise ``ConfigError``.
    """
    if isinstance(field, NeuralField):
        kw = _neural_kwargs(field)
        if heads is not None:
            kw["heads"] = heads
            kw["out_bias"] = None
        if encoding is not None:
            kw["encoding"] = encoding
        return NeuralField(**kw)
    if isinstance(field, GridField):
        if encoding is not None:
            raise ConfigError("a GridField has no encoding")
        return GridField(field.shape, heads or field.heads, resample_mode=field.resample_mode)
    raise ConfigError(f"cannot rebuild a {type(field).__name__}")


class _NoPE(Modifier):
    name, label = "no_positional_encoding", "−PE"

    def problem(self, problem):
        f = problem.field
        if not isinstance(f, NeuralField):
            log.warning("no_positional_encoding: %s has no encoding; unchanged", type(f).__name__)
            return problem
        problem.field = rebuild_field(f, encoding=IdentityEncoding(f.ndim))
        return problem


class _SingleStage(Modifier):
    name, label = "single_stage", "−multiscale"

    def curriculum(self, curriculum):
        if len(curriculum.stages) == 1:
            return curriculum
        first, last = curriculum.stages[0], curriculum.stages[-1]
        stage = dataclasses.replace(last, name="single", steps=curriculum.total_steps, lr=first.lr)
        new = copy.copy(curriculum)
        new.stages = [stage]
        return new


class _DropLoss(Modifier):
    def __init__(self, loss: str) -> None:
        self.loss = loss
        self.name, self.label = f"drop_loss({loss})", f"−{loss}"

    def problem(self, problem):
        if self.loss not in problem.losses.weights:
            raise ConfigError(f"drop_loss: no loss {self.loss!r}; known: {problem.losses.names}")
        problem.losses.weights[self.loss] = 0.0
        return problem

    def curriculum(self, curriculum):
        for s in curriculum.stages:
            if s.loss_weights and self.loss in s.loss_weights:
                s.loss_weights = {**s.loss_weights, self.loss: 0.0}
        return curriculum


class _SetWeight(Modifier):
    def __init__(self, loss: str, weight: float) -> None:
        self.loss, self.weight = loss, float(weight)
        self.name = f"set_weight({loss}={weight:g})"
        self.label = f"{loss}={weight:g}"

    def problem(self, problem):
        if self.loss not in problem.losses.weights:
            raise ConfigError(f"set_weight: no loss {self.loss!r}; known: {problem.losses.names}")
        problem.losses.weights[self.loss] = self.weight
        return problem

    def curriculum(self, curriculum):
        for s in curriculum.stages:
            if s.loss_weights and self.loss in s.loss_weights:
                s.loss_weights = {**s.loss_weights, self.loss: self.weight}
        return curriculum


class _GridField(Modifier):
    def __init__(self, lr_scale: float = 1.0) -> None:
        self.lr_scale = float(lr_scale)
        self.name = "grid_field" + (f"(lr×{lr_scale:g})" if lr_scale != 1.0 else "")
        self.label = "grid field"

    def problem(self, problem):
        problem.field = GridField(problem.domain.shape, problem.field.heads)
        return problem

    def curriculum(self, curriculum):
        if self.lr_scale != 1.0:
            for s in curriculum.stages:
                s.lr *= self.lr_scale
        return curriculum


class _NoGate(Modifier):
    name, label = "no_gate", "−gate"

    def problem(self, problem):
        items = []
        changed = False
        for k, h in problem.field.heads.items():
            if isinstance(h, GatedSoftplus):
                h = Softplus(beta=h.beta, init_value=h.init_value)
                changed = True
            items.append((k, h))
        if not changed:
            log.warning("no_gate: the field has no GatedSoftplus head; unchanged")
            return problem
        heads = Heads(items, primary=problem.field.heads.primary)
        problem.field = rebuild_field(problem.field, heads=heads)
        return problem


class _SetConfig(Modifier):
    def __init__(self, **overrides: Any) -> None:
        self.overrides = overrides
        self.name = "set(" + ",".join(f"{k}={v}" for k, v in overrides.items()) + ")"
        self.label = ",".join(f"{k}={v}" for k, v in overrides.items())

    def config(self, cfg):
        if dataclasses.is_dataclass(cfg):
            known = {f.name for f in dataclasses.fields(cfg)}
            unknown = set(self.overrides) - known
            if unknown:
                raise ConfigError(
                    f"unknown config keys {sorted(unknown)} for {type(cfg).__name__}; "
                    f"known: {sorted(known)}"
                )
        return dict(self.overrides)


class _SetAttrs(Modifier):
    def __init__(self, target: str, **attrs: Any) -> None:
        self.target, self.attrs = target, attrs
        items = ",".join(f"{k}={v}" for k, v in attrs.items())
        self.name = f"{target}({items})"
        self.label = f"{target}.{items}" if target != "stage" else items

    def curriculum(self, curriculum):
        objs = {
            "stage": list(curriculum.stages),
            "curriculum": [curriculum],
            "optim": [curriculum.optim],
        }[self.target]
        for o in objs:
            for k, v in self.attrs.items():
                if not hasattr(o, k):
                    raise ConfigError(f"{type(o).__name__} has no attribute {k!r}")
                setattr(o, k, tuple(v) if isinstance(v, list) and k == "shape" else v)
        return curriculum


def no_annealing() -> Modifier:
    """Disable frequency annealing: every stage starts with all Fourier bands on."""
    return _NoAnnealing()


def no_positional_encoding() -> Modifier:
    """Replace the Fourier-feature encoding of a :class:`NeuralField` by raw coordinates."""
    return _NoPE()


def single_stage() -> Modifier:
    """Collapse a multiscale curriculum to one stage at the final resolution.

    The stage copies the last stage's settings with the total step budget and the first
    stage's learning rate.
    """
    return _SingleStage()


def drop_loss(name: str) -> Modifier:
    """Set loss ``name`` to weight 0 (also in per-stage overrides)."""
    return _DropLoss(name)


def set_weight(name: str, weight: float) -> Modifier:
    """Set loss ``name`` to ``weight`` everywhere (base weight and per-stage overrides)."""
    return _SetWeight(name, weight)


def grid_field(lr_scale: float = 1.0) -> Modifier:
    """Replace the field by a free :class:`GridField` with the same heads (Grid Opt. / Tikhonov).

    ``lr_scale`` multiplies every stage's learning rate (grids usually want a larger one; the
    papers' Grid Opt. keeps the neural-field schedule, which is the default here).
    """
    return _GridField(lr_scale)


def no_gate() -> Modifier:
    """Replace every :class:`GatedSoftplus` head by a plain :class:`Softplus` (NeTMY "−gate")."""
    return _NoGate()


def set_config(**overrides: Any) -> Modifier:
    """Instance-config overrides for the *inversion* (must not change data generation)."""
    return _SetConfig(**overrides)


def set_stage(**attrs: Any) -> Modifier:
    """Set attributes on every curriculum stage (e.g. ``anneal_fraction=0.25``)."""
    return _SetAttrs("stage", **attrs)


def set_curriculum(**attrs: Any) -> Modifier:
    """Set curriculum attributes (e.g. ``restarts=3``, ``early_stop_patience=200``)."""
    return _SetAttrs("curriculum", **attrs)


def set_optim(**attrs: Any) -> Modifier:
    """Set optimizer settings (e.g. ``optimizer="adam"``, ``grad_clip=None``, ``ema=0.99``)."""
    return _SetAttrs("optim", **attrs)


MODIFIERS: dict[str, Callable[..., Modifier]] = {
    "no_annealing": no_annealing,
    "no_positional_encoding": no_positional_encoding,
    "no_pe": no_positional_encoding,
    "single_stage": single_stage,
    "grid_field": grid_field,
    "no_gate": no_gate,
}


def _kv(spec: str) -> tuple[str, Any]:
    if "=" not in spec:
        raise ConfigError(f"expected key=value, got {spec!r}")
    k, v = spec.split("=", 1)
    return k.strip(), parse_value(v)


def parse_modifier(spec: str) -> Modifier:
    """Parse a modifier string (used by the CLI).

    Accepted forms: ``no_annealing``, ``no_positional_encoding`` (``no_pe``), ``single_stage``,
    ``grid_field[:lr_scale]``, ``no_gate``, ``drop_loss:<name>``, ``set_weight:<name>=<w>``,
    ``set:<key>=<value>`` (instance config), ``stage:<attr>=<value>``,
    ``curriculum:<attr>=<value>``, ``optim:<attr>=<value>``.
    """
    s = spec.strip()
    head, _, arg = s.partition(":")
    head = head.strip().lower()
    if head in MODIFIERS and not arg:
        return MODIFIERS[head]()
    if head == "grid_field":
        return grid_field(float(arg))
    if head in ("drop_loss", "drop"):
        return drop_loss(arg.strip())
    if head in ("set_weight", "weight"):
        k, v = _kv(arg)
        return set_weight(k, float(v))
    if head in ("set", "config"):
        k, v = _kv(arg)
        return set_config(**{k: v})
    if head in ("stage", "curriculum", "optim"):
        k, v = _kv(arg)
        return _SetAttrs(head, **{k: v})
    known = sorted(MODIFIERS) + [
        "grid_field:<lr_scale>",
        "drop_loss:<name>",
        "set_weight:<name>=<w>",
        "set:<key>=<value>",
        "stage:<attr>=<value>",
        "curriculum:<attr>=<value>",
        "optim:<attr>=<value>",
    ]
    raise ConfigError(f"unknown modifier {spec!r}; known: {known}")


def as_modifier(obj: Any) -> Modifier:
    """Coerce a :class:`Modifier`, a spec string, a config-override dict or a callable."""
    if isinstance(obj, Modifier):
        return obj
    if isinstance(obj, str):
        return parse_modifier(obj)
    if isinstance(obj, Mapping):
        return set_config(**obj)
    if callable(obj):
        return FnModifier(obj)
    raise ConfigError(f"cannot interpret {obj!r} as a modifier")


# ------------------------------------------------------------------------------------------
# variants and the cumulative ablation
# ------------------------------------------------------------------------------------------
def _variant_instance(instance: Any, mods: Sequence[Modifier]) -> Any:
    inst = instance
    for m in mods:
        over = m.config(inst.cfg)
        if over:
            try:
                inst = type(inst)(inst.cfg, **over)
            except TypeError as e:
                raise ConfigError(f"{m.name}: cannot apply config overrides {over}: {e}") from e
    return inst


def variant_method(instance: Any, label: str, modifiers: Sequence[Any], base: Any = None):
    """A :class:`~nefi.bench.protocol.Method` = ``base`` method with ``modifiers`` applied.

    Config modifiers create a variant instance for building the problem; the benchmark still
    generates the data with the *original* instance, so every variant sees the same measurement.
    """
    from .protocol import Method, PreparedRun, default_method

    mods = [as_modifier(m) for m in modifiers]
    base = base if base is not None else default_method()
    if isinstance(base, str):
        from .protocol import resolve_methods

        base = resolve_methods(instance, [base])[0]
    state: dict[str, Any] = {}

    def build(_instance: Any, measurement: Any) -> PreparedRun:
        if "inst" not in state:
            state["inst"] = _variant_instance(instance, mods)
        prep = base.prepare(state["inst"], measurement)
        problem = prep.problem
        cur = copy.deepcopy(prep.resolved_curriculum())
        for m in mods:
            problem, cur = m.apply(problem, cur)
        if hasattr(problem, "curriculum"):
            problem.curriculum = cur
        return PreparedRun(problem, cur, prep.solver_kwargs, prep.runner)

    desc = " + ".join(m.name for m in mods) or "unmodified"
    return Method(label, build, description=desc)


def _normalize_variants(variants: Sequence[Any]) -> list[tuple[str, Modifier]]:
    out = []
    for v in variants:
        if isinstance(v, tuple | list) and len(v) == 2 and isinstance(v[0], str):
            out.append((v[0], as_modifier(v[1])))
        else:
            m = as_modifier(v)
            out.append((m.label, m))
    return out


def cumulative_ablation(
    instance: Any,
    variants: Sequence[Any],
    n_samples: int = 4,
    seeds: Sequence[int] = (0, 1, 2),
    *,
    cumulative: bool = True,
    include_full: bool = True,
    full_name: str = "full",
    base: Any = None,
    **bench_kw: Any,
):
    """Ablation table with one method per (cumulative) step (NeTMY Tab. 3, NeFTY Tab. 3).

    Args:
        instance: the instance.
        variants: ``[(label, modifier), ...]`` or bare modifiers / spec strings (their ``label``
            is used). Row ``k`` applies modifiers ``1..k`` when ``cumulative`` (NeTMY: "each row
            removes one component cumulatively"), only modifier ``k`` otherwise (one-at-a-time).
            For an additive table (NeFTY Tab. 3, "each row adds one component") list the
            removals in reverse order and read the table bottom-up.
        n_samples: measurements (shared by all rows).
        seeds: optimization seeds (shared by all rows).
        cumulative: cumulative (default) or independent single-modifier rows.
        include_full: prepend the unmodified method as ``full_name``.
        full_name: label of the unmodified row.
        base: the method being ablated (default: the instance's neural method).
        **bench_kw: forwarded to :func:`~nefi.bench.protocol.run_benchmark` (``classes``,
            ``metrics``, ``device``, ``budget_scale``, ``out_dir``, ``progress``, ...).

    Returns:
        A :class:`~nefi.bench.protocol.BenchmarkResult` whose methods are the ablation rows.
    """
    from .protocol import run_benchmark

    pairs = _normalize_variants(variants)
    steps: list[tuple[str, list[Modifier]]] = [(full_name, [])] if include_full else []
    for k, (label, mod) in enumerate(pairs):
        mods = [m for _, m in pairs[: k + 1]] if cumulative else [mod]
        steps.append((label, mods))
    labels = [s[0] for s in steps]
    if len(set(labels)) != len(labels):
        raise ConfigError(f"ablation labels must be unique: {labels}")
    methods = [variant_method(instance, label, mods, base) for label, mods in steps]
    res = run_benchmark(instance, methods, n_samples, seeds, **bench_kw)
    res.meta["ablation"] = {
        "cumulative": cumulative,
        "steps": [[label, [m.name for m in mods]] for label, mods in steps],
    }
    out_dir = bench_kw.get("out_dir")
    if out_dir is not None:  # re-save with the ablation metadata
        res.save(out_dir)
    return res


__all__ = [
    "MODIFIERS",
    "FnModifier",
    "Modifier",
    "as_modifier",
    "cumulative_ablation",
    "drop_loss",
    "grid_field",
    "no_annealing",
    "no_gate",
    "no_positional_encoding",
    "parse_modifier",
    "parse_value",
    "rebuild_field",
    "set_config",
    "set_curriculum",
    "set_optim",
    "set_stage",
    "set_weight",
    "single_stage",
    "variant_method",
]
