"""Result container with save / load and a compact summary."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class Result:
    """Output of :class:`nefi.solve.Solver`.

    Attributes:
        fields: final post-processed fields (detached, on CPU).
        raw_fields: fields before post-processing.
        pred: operator output for ``fields``.
        history: per-step logs (``step``, ``global_step``, ``stage``, ``lr``, ``progress``,
        ``total`` and
            one entry per loss component).
        stage_results: per-stage summaries (final losses, steps run, seconds).
        timing: ``{"total_s": ..., "per_stage_s": [...]}``.
        post_info: information returned by post-processors (e.g. ``scale_factor``).
        config_hash: hash of the curriculum/problem config.
        extra: free-form (restart index, device, ...).
    """

    fields: dict[str, torch.Tensor]
    raw_fields: dict[str, torch.Tensor]
    pred: torch.Tensor
    history: dict[str, list[float]]
    stage_results: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)
    post_info: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def primary(self) -> torch.Tensor:
        return next(iter(self.fields.values()))

    def final(self, key: str) -> float | None:
        h = self.history.get(key)
        return None if not h else float(h[-1])

    def summary(self) -> str:
        lines = [f"Result: fields={ {k: tuple(v.shape) for k, v in self.fields.items()} }"]
        lines.append(
            f"  total time {self.timing.get('total_s', float('nan')):.1f}s, "
            f"steps {len(self.history.get('total', []))}"
        )
        for i, s in enumerate(self.stage_results):
            comps = ", ".join(f"{k}={v:.3g}" for k, v in s.get("final", {}).items())
            lines.append(
                f"  stage {i} ({s.get('name')}, {s.get('shape')}): {s.get('steps')} steps, "
                f"{s.get('seconds', 0):.1f}s | {comps}"
            )
        if self.post_info:
            lines.append(f"  post: {self.post_info}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "fields": self.fields,
                "raw_fields": self.raw_fields,
                "pred": self.pred,
                "history": self.history,
                "stage_results": self.stage_results,
                "timing": self.timing,
                "post_info": self.post_info,
                "config_hash": self.config_hash,
                "extra": self.extra,
            },
            path,
        )

    @staticmethod
    def load(path: str | Path) -> Result:
        d = torch.load(Path(path), map_location="cpu", weights_only=False)
        return Result(**d)
