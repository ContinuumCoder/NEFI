"""Report writers (markdown / CSV / JSON), confidence intervals and operator runtime tables.

Used by :class:`~nefi.bench.protocol.BenchmarkResult`, the ablation and sweep helpers and the CLI.
:func:`runtime_table` times the forward and backward pass of a problem's operator on its current
field values — the solver-level efficiency table of NeFTY Tab. 4 (fwd/bwd time, peak memory,
simulation error against a reference).
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..utils.device import peak_memory_mb, reset_peak_memory, resolve_device, synchronize

# ------------------------------------------------------------------------------------------
# statistics
# ------------------------------------------------------------------------------------------
_HIGHER = ("psnr", "ssim", "iou", "dice", "f1", "acc", "delta", "precision", "recall", "corr")
_LOWER = ("mse", "mae", "error", "gmsd", "swd", "wasserstein", "abs_rel", "loss", "time", "mem")


def metric_direction(name: str) -> bool | None:
    """``True`` if higher is better, ``False`` if lower is better, ``None`` if unknown."""
    n = name.lower()
    if any(t in n for t in _HIGHER):
        return True
    if any(t in n for t in _LOWER):
        return False
    return None


def mean_ci(values: Iterable[float], ci: float = 0.95) -> tuple[float, float, int]:
    """Mean and confidence-interval half-width with the Student-t distribution.

    Non-finite values are dropped. Returns ``(mean, half_width, n)``; the half-width is ``nan``
    for ``n < 2`` (a single run has no interval).
    """
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = statistics.fmean(vals)
    if n < 2:
        return m, float("nan"), n
    from scipy.stats import t as student_t

    sd = statistics.stdev(vals)
    return m, float(student_t.ppf(0.5 + ci / 2.0, n - 1)) * sd / math.sqrt(n), n


# ------------------------------------------------------------------------------------------
# formatting
# ------------------------------------------------------------------------------------------
def format_value(x: Any, precision: int = 4) -> str:
    """Compact number formatting (``—`` for missing, scientific for very small/large)."""
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return "—"
    if math.isinf(v):
        return "∞" if v > 0 else "-∞"
    if v != 0 and (abs(v) >= 10 ** (precision + 1) or abs(v) < 10 ** -(precision - 1)):
        return f"{v:.{max(1, precision - 2)}e}"
    return f"{v:.{precision}g}"


def format_mean_ci(mean: float, half_width: float, precision: int = 4) -> str:
    """``mean ± hw`` (just ``mean`` when the half-width is undefined)."""
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "—"
    if half_width is None or math.isnan(half_width):
        return format_value(mean, precision)
    return f"{format_value(mean, precision)} ± {format_value(half_width, max(2, precision - 1))}"


def markdown_table(
    headers: Sequence[str], rows: Sequence[Sequence[Any]], align: Sequence[str] | None = None
) -> str:
    """GitHub-flavoured markdown table (``align`` entries: ``"l"``, ``"r"`` or ``"c"``)."""
    align = align or ["l"] + ["r"] * (len(headers) - 1)
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    out = ["| " + " | ".join(str(h) for h in headers) + " |"]
    out.append("|" + "|".join(sep.get(a, "---") for a in align) + "|")
    for r in rows:
        cells = [str(c).replace("|", "\\|").replace("\n", " ") for c in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------
# CSV / JSON
# ------------------------------------------------------------------------------------------
def to_jsonable(obj: Any) -> Any:
    """Recursively convert tensors / tuples / paths / non-finite floats to JSON-safe values."""
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set):
        return [to_jsonable(v) for v in obj]
    if torch.is_tensor(obj):
        return to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return to_jsonable(obj.item())
        except (TypeError, ValueError):
            pass
    if obj is None or isinstance(obj, str | int | bool):
        return obj
    return str(obj)


def rows_to_csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> str:
    """CSV text for a list of dict rows (union of keys, in first-seen order)."""
    if columns is None:
        columns = []
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: _csv_cell(r.get(k)) for k in columns})
    return buf.getvalue()


def _csv_cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float) and not math.isfinite(v):
        return "" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    if isinstance(v, list | tuple):
        return json.dumps(to_jsonable(v))
    return v


def write_text(path: str | Path, text: str) -> Path:
    """Write ``text`` (creating parent directories) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def write_json(path: str | Path, obj: Any) -> Path:
    """Write a JSON file (tensors, tuples and non-finite floats converted)."""
    return write_text(path, json.dumps(to_jsonable(obj), indent=2) + "\n")


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write dict rows as CSV."""
    return write_text(path, rows_to_csv(rows))


# ------------------------------------------------------------------------------------------
# operator runtime table (NeFTY Tab. 4)
# ------------------------------------------------------------------------------------------
@dataclass
class RuntimeTable:
    """Per-operator forward/backward timings and peak memory (NeFTY Tab. 4 style)."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def to_markdown(self, precision: int = 3) -> str:
        has_err = any(r.get("sim_error") is not None for r in self.rows)
        headers = ["Operator", "grid", "fwd time (s) ↓", "bwd time (s) ↓", "peak mem (MB) ↓"]
        if has_err:
            headers.append("sim. error ↓")
        body = []
        for r in self.rows:
            line = [
                r["name"],
                "×".join(str(s) for s in r["shape"]),
                format_mean_ci(r["fwd_s_mean"], r["fwd_s_std"], precision),
                format_mean_ci(r["bwd_s_mean"], r["bwd_s_std"], precision),
                format_value(r.get("peak_mem_mb"), precision),
            ]
            if has_err:
                line.append(format_value(r.get("sim_error"), precision))
            body.append(line)
        note = (
            f"\n\nmean ± std over {self.rows[0]['n_repeat']} calls on {self.rows[0]['device']}"
            if self.rows
            else ""
        )
        return markdown_table(headers, body) + note + "\n"

    def to_csv(self) -> str:
        return rows_to_csv(self.rows)

    def to_json(self) -> str:
        return json.dumps(to_jsonable(self.rows), indent=2)


def _prepare_entry(entry: Any, shape: Sequence[int] | None):
    """(operator, fields) for an InverseProblem or an (operator, fields) pair."""
    if hasattr(entry, "operator") and hasattr(entry, "evaluate"):
        fields, _ = entry.evaluate(shape, progress=1.0)
        dom = entry.domain if shape is None else entry.domain.at(shape)
        return entry.operator.at_resolution(dom.shape), fields, tuple(dom.shape)
    op, fields = entry
    first = next(iter(fields.values()))
    return op, fields, tuple(first.shape)


def runtime_table(
    problems: Any,
    shape: Sequence[int] | None = None,
    *,
    n_repeat: int = 5,
    warmup: int = 1,
    device: str | torch.device | None = None,
    grad_fields: Sequence[str] | None = None,
    reference: str | torch.Tensor | None = None,
) -> RuntimeTable:
    """Time the forward and backward pass of one or several operators (NeFTY Tab. 4).

    Args:
        problems: an :class:`~nefi.problem.InverseProblem`, an ``(operator, fields)`` pair, or a
            mapping ``name -> either`` to compare implementations (e.g. autograd vs adjoint).
        shape: grid shape (default native).
        n_repeat: timed calls (mean ± std reported).
        warmup: untimed calls first (kernel caches, cuDNN autotuning, compilation).
        device: device to time on (default: the problem's current device).
        grad_fields: fields that receive gradients in the backward pass (default: all).
        reference: name of an entry, or a tensor, to compute the relative simulation error
            ``‖y − y_ref‖ / ‖y_ref‖`` of every entry.

    Returns:
        A :class:`RuntimeTable` (``.to_markdown()``); peak memory is reported on CUDA only.
    """
    entries = dict(problems) if isinstance(problems, Mapping) else {"operator": problems}
    rows: list[dict[str, Any]] = []
    outputs: dict[str, torch.Tensor] = {}
    for name, entry in entries.items():
        op, fields, fshape = _prepare_entry(entry, shape)
        dev = resolve_device(device) if device is not None else next(iter(fields.values())).device
        op = op.to(dev)
        wanted = set(grad_fields) if grad_fields is not None else set(fields)
        leaves = {
            k: v.detach().to(dev).clone().requires_grad_(k in wanted and v.is_floating_point())
            for k, v in fields.items()
        }
        for _ in range(max(0, warmup)):
            y = op(leaves)
            if y.requires_grad:
                y.backward(torch.ones_like(y))
        fwd, bwd = [], []
        reset_peak_memory(dev)
        for _ in range(max(1, n_repeat)):
            for v in leaves.values():
                v.grad = None
            synchronize(dev)
            t0 = time.perf_counter()
            y = op(leaves)
            synchronize(dev)
            fwd.append(time.perf_counter() - t0)
            if y.requires_grad:
                g = torch.ones_like(y)
                synchronize(dev)
                t0 = time.perf_counter()
                y.backward(g)
                synchronize(dev)
                bwd.append(time.perf_counter() - t0)
        outputs[name] = y.detach()
        rows.append(
            {
                "name": name,
                "shape": list(fshape),
                "out_shape": list(y.shape),
                "fwd_s_mean": statistics.fmean(fwd),
                "fwd_s_std": statistics.stdev(fwd) if len(fwd) > 1 else float("nan"),
                "bwd_s_mean": statistics.fmean(bwd) if bwd else float("nan"),
                "bwd_s_std": statistics.stdev(bwd) if len(bwd) > 1 else float("nan"),
                "peak_mem_mb": peak_memory_mb(dev),
                "device": str(dev),
                "n_repeat": max(1, n_repeat),
                "sim_error": None,
            }
        )
    if reference is not None:
        ref = outputs[reference] if isinstance(reference, str) else torch.as_tensor(reference)
        for r in rows:
            y = outputs[r["name"]]
            if tuple(y.shape) == tuple(ref.shape):
                y64, ref64 = y.double().cpu(), ref.double().cpu()
                r["sim_error"] = float((y64 - ref64).norm() / ref64.norm().clamp_min(1e-300))
    return RuntimeTable(rows)


__all__ = [
    "RuntimeTable",
    "format_mean_ci",
    "format_value",
    "markdown_table",
    "mean_ci",
    "metric_direction",
    "rows_to_csv",
    "runtime_table",
    "to_jsonable",
    "write_csv",
    "write_json",
    "write_text",
]
