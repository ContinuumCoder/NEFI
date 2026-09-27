"""A tiny registry so every component is constructible from config (``{"type": name, ...}``)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .errors import ConfigError, RegistryError

KINDS = (
    "field",
    "encoding",
    "head",
    "operator",
    "loss",
    "postprocess",
    "instance",
    "baseline",
    "metric",
    "callback",
    "scene",
    "datagen",
)

_REGISTRY: dict[str, dict[str, Any]] = defaultdict(dict)


def register(kind: str, name: str | None = None) -> Callable:
    """Decorator registering a class / factory under ``kind`` and ``name``.

    Example::

        @register("head", "gated_softplus")
        class GatedSoftplus(Head): ...
    """
    if kind not in KINDS:
        raise RegistryError(f"unknown registry kind {kind!r}; known: {KINDS}")

    def deco(obj):
        key = name or getattr(obj, "__name__", None)
        if key is None:
            raise RegistryError("cannot infer a registry name; pass one explicitly")
        _REGISTRY[kind][key.lower()] = obj
        return obj

    return deco


def get(kind: str, name: str) -> Any:
    try:
        return _REGISTRY[kind][name.lower()]
    except KeyError as e:
        known = ", ".join(sorted(_REGISTRY[kind])) or "(none)"
        raise RegistryError(f"no {kind} registered as {name!r}; known: {known}") from e


def list_registered(kind: str | None = None) -> dict[str, list[str]]:
    kinds = [kind] if kind else list(KINDS)
    return {k: sorted(_REGISTRY[k]) for k in kinds}


def build(kind: str, cfg: str | dict | Any, **extra: Any) -> Any:
    """Instantiate a registered component.

    ``cfg`` may be a name, a dict ``{"type": name, **kwargs}``, or an already-built object
    (returned unchanged). ``extra`` kwargs are merged (config wins on conflicts).
    """
    if isinstance(cfg, str):
        return get(kind, cfg)(**extra)
    if isinstance(cfg, dict):
        cfg = dict(cfg)
        name = cfg.pop("type", None)
        if name is None:
            raise ConfigError(f"config for {kind} needs a 'type' key: {cfg}")
        kwargs = {**extra, **cfg}
        return get(kind, name)(**kwargs)
    return cfg
