"""Benchmarking: scene/data generators, protocols, ablations, reports."""

from .base import DataGenerator, SceneGenerator

__all__ = ["DataGenerator", "SceneGenerator"]

try:  # protocol / ablation / sweep / report modules
    from .protocol import *  # noqa: F401,F403
    from .protocol import __all__ as _p_all

    __all__ += _p_all
except ImportError:  # pragma: no cover
    pass
