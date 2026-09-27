"""Device resolution. CUDA-first; MPS only when explicitly requested (no float64, partial FFT)."""

from __future__ import annotations

import os

import torch


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve a device spec.

    ``"auto"`` (default) picks ``NEFI_DEVICE`` from the environment if set, else CUDA if
    available, else CPU. Apple MPS is *not* auto-selected because it lacks float64 and some FFT
    kernels; pass ``"mps"`` explicitly to use it.
    """
    if device is None or device == "auto":
        env = os.environ.get("NEFI_DEVICE")
        if env:
            return torch.device(env)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def synchronize(device: torch.device) -> None:
    """Block until all queued kernels on ``device`` finished (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def peak_memory_mb(device: torch.device) -> float | None:
    """Peak allocated memory in MB for CUDA devices, ``None`` otherwise."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    return None


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
