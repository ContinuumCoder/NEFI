"""Wall-clock timing helper."""

from __future__ import annotations

import time


class Timer:
    """Context manager measuring wall-clock seconds; ``Timer().elapsed`` after the block."""

    def __init__(self) -> None:
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self.start
