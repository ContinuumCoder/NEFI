"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int | None, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (all devices).

    Args:
        seed: seed value; ``None`` leaves RNG state untouched.
        deterministic: if True, ask torch for deterministic algorithms (may be slower and may
            raise for ops without a deterministic implementation).
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
