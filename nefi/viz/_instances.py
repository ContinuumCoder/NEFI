"""Instance resolution for the demo layer: registry lookup, smoke presets and step budgets.

Smoke presets are found exactly like ``nefi run --smoke`` does (the CLI is the single source of
truth when importable): an instance ``smoke_overrides`` attribute / method, then
``configs/<name>_smoke.yaml``, then a ``"smoke"`` entry of the instance module's ``PRESETS``.
"""

from __future__ import annotations

import copy
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

log = logging.getLogger("nefi")


def registered_instances() -> list[str]:
    """Registered instance names, in the order of :data:`nefi.instances.available` first."""
    import nefi.instances as inst_pkg

    from ..registry import list_registered

    names = list(list_registered("instance")["instance"])
    order = [n for n in getattr(inst_pkg, "available", []) if n in names]
    return order + [n for n in names if n not in order]


def _own_smoke_lookup(cls: type, name: str) -> tuple[dict, dict | None, str] | None:
    hook = getattr(cls, "smoke_overrides", None)
    if callable(hook):
        try:
            hook = hook()
        except TypeError:
            hook = None
    if isinstance(hook, Mapping):
        return dict(hook), None, f"{cls.__name__}.smoke_overrides"
    import nefi

    from ..config import load_config

    for d in (Path.cwd() / "configs", Path(nefi.__file__).resolve().parents[1] / "configs"):
        for ext in (".yaml", ".yml", ".json"):
            p = d / f"{name}_smoke{ext}"
            if not p.exists():
                continue
            raw = load_config(p)
            inst = raw.get("instance")
            if isinstance(inst, Mapping):
                cfg = {k: v for k, v in inst.items() if k != "type"}
            else:
                cfg = {}
            cfg.update(raw.get("config") or {})
            return cfg, raw.get("curriculum"), display_source(str(p))
    presets = getattr(cls, "PRESETS", None) or getattr(
        sys.modules.get(cls.__module__), "PRESETS", None
    )
    if isinstance(presets, Mapping) and isinstance(presets.get("smoke"), Mapping):
        return dict(presets["smoke"]), None, f"{cls.__module__}.PRESETS['smoke']"
    return None


def display_source(src: str | None) -> str | None:
    """A preset source for reports: file paths relative to the repository (or working
    directory) so shared reports do not expose local absolute paths."""
    if not src:
        return src
    p = Path(src)
    if not p.is_absolute():
        return src
    import nefi

    for root in (Path(nefi.__file__).resolve().parents[1], Path.cwd()):
        try:
            return str(p.resolve().relative_to(root.resolve()))
        except ValueError:
            continue
    return p.name


def smoke_config(name: str) -> tuple[dict[str, Any], dict | None, str | None]:
    """``(config_overrides, curriculum_dict | None, source | None)`` of an instance's smoke preset.

    Uses :func:`nefi.cli.load_spec` when available so the gallery and ``nefi run --smoke`` agree;
    falls back to the same lookup order implemented here.
    """
    try:
        from ..cli import load_spec

        spec = load_spec(name, smoke=True)
        src = spec.smoke if spec.smoke not in (None, "step cap only") else None
        if src is None:
            return {}, None, None
        return dict(spec.config), spec.curriculum, display_source(str(src))
    except ImportError:
        pass
    except Exception as e:  # CLI present but refused (e.g. not registered): use our lookup
        log.debug("cli smoke lookup failed for %s: %s", name, e)
    from ..registry import get

    try:
        cls = get("instance", name)
    except Exception:
        return {}, None, None
    found = _own_smoke_lookup(cls, name)
    if found is None:
        return {}, None, None
    return found


def resolve_instance(
    spec: Any, *, smoke: bool = True, overrides: Mapping[str, Any] | None = None
) -> tuple[Any, dict[str, Any]]:
    """Instantiate an instance from a registered name, an ``Instance`` subclass or an object.

    Args:
        spec: registered name (``"toy1d"``), ``Instance`` subclass, or instance object.
        smoke: apply the smoke preset (names only).
        overrides: config overrides (win over the smoke preset).

    Returns:
        ``(instance, info)`` with ``info = {"name", "smoke", "config", "curriculum", "cls"}``:
        ``smoke`` is the preset source (``None`` if none was applied), ``config`` the overrides
        actually applied and ``curriculum`` a smoke curriculum dict (or ``None``).
    """
    overrides = dict(overrides or {})
    info: dict[str, Any] = {"smoke": None, "config": {}, "curriculum": None}
    if isinstance(spec, str):
        from ..registry import get

        cls = get("instance", spec)
        cfg: dict[str, Any] = {}
        if smoke:
            cfg, cur, src = smoke_config(spec)
            info.update(smoke=src, curriculum=cur)
        cfg = {**cfg, **overrides}
        inst = cls(dict(cfg)) if cfg else cls()
        info.update(name=spec, config=cfg, cls=cls, user=overrides)
        return inst, info
    if isinstance(spec, type):
        inst = spec(dict(overrides)) if overrides else spec()
        info.update(name=getattr(spec, "name", spec.__name__), config=overrides, cls=spec)
        return inst, info
    if overrides:
        log.warning("config overrides ignored for an already-built instance %r", spec)
    info.update(name=str(getattr(spec, "name", type(spec).__name__)), cls=type(spec))
    return spec, info


def default_total_steps(info: Mapping[str, Any], fallback: int) -> int:
    """Total steps of the instance's *non-smoke* default curriculum (budget reference)."""
    cls = info.get("cls")
    if cls is None or info.get("smoke") is None:
        return int(fallback)
    try:
        user = dict(info.get("user") or {})
        full = cls(user) if user else cls()
        return int(full.default_curriculum().total_steps)
    except Exception as e:  # pragma: no cover - exotic constructors
        log.debug("default curriculum unavailable: %s", e)
        return int(fallback)


def budget_curriculum(
    curriculum: Any,
    info: Mapping[str, Any],
    budget_scale: float,
    *,
    max_steps: int | None = None,
    min_steps: int = 20,
    time_budget_s: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Scale a (smoke-configured) curriculum for a demo run.

    The target is ``budget_scale ×`` the steps of the instance's *default* (paper) curriculum,
    floored by the smoke preset's own step count (validated by the instance author) — or by
    ``min_steps`` without a preset — and capped by ``max_steps``. Stage structure, resolutions,
    learning rates and annealing fractions are kept; restarts are set to 1.

    Returns:
        ``(curriculum, budget_info)``.
    """
    from ..config import from_dict
    from ..solve.curriculum import Curriculum

    base = copy.deepcopy(curriculum)
    smoke_cur = info.get("curriculum")
    if isinstance(smoke_cur, Mapping) and smoke_cur:
        try:
            base = from_dict(Curriculum, dict(smoke_cur))
        except Exception as e:  # pragma: no cover - malformed smoke curriculum
            log.debug("smoke curriculum ignored: %s", e)
    base_total = max(1, int(base.total_steps))
    ref_total = default_total_steps(info, base_total)
    target = float(budget_scale) * ref_total
    floor = base_total if info.get("smoke") else int(min_steps)
    target = max(target, float(floor))
    if max_steps is not None:
        target = min(target, float(max_steps))
    target = max(1.0, target)
    cur = base.scaled(target / base_total) if abs(target - base_total) > 0.5 else base
    cur.restarts = 1
    if time_budget_s is not None:
        cur.time_budget_s = (
            float(time_budget_s)
            if cur.time_budget_s is None
            else min(float(cur.time_budget_s), float(time_budget_s))
        )
    budget = {
        "budget_scale": float(budget_scale),
        "reference_steps": int(ref_total),
        "preset_steps": int(base_total),
        "steps": int(cur.total_steps),
        "stages": [
            {"name": s.name, "shape": list(s.shape) if s.shape else None, "steps": s.steps}
            for s in cur.stages
        ],
        "time_budget_s": cur.time_budget_s,
    }
    return cur, budget


def mse_noise_floor(problem: Any, measurement: Any) -> float | None:
    """``w · σ²`` when the only data term is a plain :class:`~nefi.losses.MSE` (weight ``w``) and
    the noise level is known — the expected data loss at the truth. ``None`` otherwise."""
    try:
        ns = getattr(measurement, "noise_std", None)
        if ns is None:
            return None
        import torch

        sigma = float(torch.as_tensor(ns).float().mean())
        losses = problem.losses
        data = list(losses.data_terms())
        if len(data) != 1 or type(losses.terms[data[0]]).__name__ != "MSE":
            return None
        return float(losses.weights[data[0]]) * sigma**2
    except Exception:
        return None


def primary_field(result: Any, gt: Mapping | None) -> str | None:
    """The field compared in tiles: the first result field that also exists in ``gt``."""
    fields = list(getattr(result, "fields", {}) or {})
    if gt:
        for k in fields:
            if k in gt:
                return k
        return next(iter(gt))
    return fields[0] if fields else None


def shapes(d: Mapping[str, Any] | None) -> dict[str, list[int]]:
    """``{name: shape}`` of a tensor dict (JSON-friendly)."""
    out = {}
    for k, v in (d or {}).items():
        try:
            out[str(k)] = [int(s) for s in v.shape]
        except AttributeError:
            continue
    return out


def as_list(x: Any) -> list[Any]:
    """``None`` → ``[]``; comma-separated string → list; sequence → list."""
    if x is None:
        return []
    if isinstance(x, str):
        return [s.strip() for s in x.split(",") if s.strip()]
    if isinstance(x, Sequence):
        return list(x)
    return [x]
