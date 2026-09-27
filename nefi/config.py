"""Config helpers: dataclass <-> dict / YAML, and stable config hashing."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .errors import ConfigError

T = TypeVar("T")


def to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses / tuples / tensors to JSON-serializable objects."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_dict(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def from_dict(cls: type[T], d: dict) -> T:
    """Build a (possibly nested) dataclass from a dict, ignoring unknown keys with an error."""
    if not dataclasses.is_dataclass(cls):
        raise ConfigError(f"{cls} is not a dataclass")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for k, v in d.items():
        if k not in fields:
            raise ConfigError(
                f"unknown config key {k!r} for {cls.__name__}; known: {sorted(fields)}"
            )
        ft = fields[k].type
        target = _resolve_type(ft, cls)
        if dataclasses.is_dataclass(target) and isinstance(v, dict):
            v = from_dict(target, v)
        elif isinstance(v, list) and _list_inner_dataclass(ft, cls) is not None:
            inner = _list_inner_dataclass(ft, cls)
            v = [from_dict(inner, x) if isinstance(x, dict) else x for x in v]
        kwargs[k] = v
    return cls(**kwargs)  # type: ignore[return-value]


def _resolve_type(ft, owner):
    if isinstance(ft, str):
        import sys

        mod = sys.modules.get(owner.__module__)
        try:
            return eval(ft, vars(mod) if mod else {})  # noqa: S307 - trusted annotations
        except Exception:
            return None
    return ft


def _list_inner_dataclass(ft, owner):
    import typing

    ft = _resolve_type(ft, owner)
    origin = typing.get_origin(ft)
    if origin in (list, tuple):
        args = typing.get_args(ft)
        if args and dataclasses.is_dataclass(args[0]):
            return args[0]
    return None


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        if path.suffix in (".yml", ".yaml"):
            return yaml.safe_load(f) or {}
        if path.suffix == ".json":
            return json.load(f)
    raise ConfigError(f"unsupported config extension: {path.suffix}")


def save_config(cfg: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = to_dict(cfg)
    with open(path, "w") as f:
        if path.suffix == ".json":
            json.dump(d, f, indent=2)
        else:
            yaml.safe_dump(d, f, sort_keys=False)


def config_hash(cfg: Any, n: int = 10) -> str:
    """Stable short hash of a config (dataclass or dict)."""
    s = json.dumps(to_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:n]
