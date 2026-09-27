"""Version shims for the PyTorch APIs used by the performance switches (``torch >= 2.1``).

Resolved once at import time (no ``try`` blocks on hot paths, so ``torch.compile`` can trace the
callers).
"""

from __future__ import annotations

import torch

__all__ = [
    "autocast_enabled",
    "cudagraph_mark_step_begin",
    "grad_scaler",
    "is_compiling",
    "is_functorch_wrapped",
]


def _resolve_is_compiling():
    fn = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if fn is not None:
        return fn
    try:  # torch 2.1 / 2.2
        from torch._dynamo import is_compiling as fn  # type: ignore[attr-defined]

        return fn
    except Exception:  # pragma: no cover - very old torch
        return lambda: False


#: ``True`` while ``torch.compile`` traces the caller.
is_compiling = _resolve_is_compiling()


def _autocast_takes_device() -> bool:
    try:
        torch.is_autocast_enabled("cpu")
        return True
    except TypeError:  # pragma: no cover - torch < 2.4
        return False


if _autocast_takes_device():

    def autocast_enabled(device_type: str) -> bool:
        """Whether ``torch.autocast`` is active for ``device_type``."""
        return torch.is_autocast_enabled(device_type)

else:  # pragma: no cover - torch < 2.4

    def autocast_enabled(device_type: str) -> bool:
        """Whether ``torch.autocast`` is active for ``device_type``."""
        if device_type == "cpu":
            return torch.is_autocast_cpu_enabled()
        return torch.is_autocast_enabled()


_WRAPPED = getattr(torch._C._functorch, "is_functorch_wrapped_tensor", None)


def is_functorch_wrapped(t: torch.Tensor) -> bool:
    """Whether ``t`` is a ``vmap`` / ``grad`` wrapper (batched) tensor."""
    return bool(_WRAPPED(t)) if _WRAPPED is not None else False


def cudagraph_mark_step_begin() -> None:
    """Mark a new iteration for CUDA-graph trees (no-op when unavailable)."""
    fn = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
    if fn is not None:
        fn()


def grad_scaler(device_type: str):
    """Dynamic loss scaler for fp16 autocast on ``device_type``."""
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        try:
            return amp.GradScaler(device_type)
        except TypeError:  # pragma: no cover - torch < 2.3
            pass
    return torch.cuda.amp.GradScaler(enabled=device_type == "cuda")  # pragma: no cover
