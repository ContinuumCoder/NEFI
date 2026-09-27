"""Plot helpers for diagnostics and results (matplotlib is optional).

All helpers use matplotlib's object-oriented API (``matplotlib.figure.Figure``), never the pyplot
state machine, so they are safe on headless servers and inside libraries. Each accepts an optional
``ax``; without one a new :class:`~matplotlib.figure.Figure` is created and returned. Use
:func:`save_figure` to write PNGs (as the CLI ``--plot`` flag does).

Field-shaped tensors are drawn as lines (1-D), images (2-D) or the central slice along the last
axis (3-D).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def _mpl():
    try:
        from matplotlib.figure import Figure
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "plotting needs matplotlib: pip install 'nefi[viz]' (or pip install matplotlib)"
        ) from e
    return Figure


def new_figure(nrows: int = 1, ncols: int = 1, figsize: tuple[float, float] | None = None):
    """A pyplot-free figure and its axes array (``squeeze=False``)."""
    Figure = _mpl()
    fig = Figure(figsize=figsize or (4.0 * ncols, 3.2 * nrows), layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False)
    return fig, axes


def _axes(ax):
    if ax is not None:
        return ax.figure, ax
    fig, axes = new_figure()
    return fig, axes[0, 0]


def save_figure(fig: Any, path: str | Path, dpi: int = 120) -> Path:
    """Save a figure (creating parent directories) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return path


def _np(t: Any):
    return torch.as_tensor(t).detach().float().cpu().numpy()


def show_field(t: Any, ax=None, title: str | None = None, cmap: str = "viridis", **plot_kw):
    """Draw a field-shaped tensor: line (1-D), image with colorbar (2-D), mid-slice (3-D)."""
    fig, ax = _axes(ax)
    a = torch.as_tensor(t).detach().float().cpu()
    note = ""
    if a.ndim == 3:
        a = a[..., a.shape[-1] // 2]
        note = " (mid z-slice)"
    elif a.ndim > 3:
        a = a.flatten()
    if a.ndim <= 1:
        ax.plot(a.numpy(), **plot_kw)
        ax.set_xlabel("pixel")
    else:
        im = ax.imshow(a.numpy().T, origin="lower", cmap=cmap, **plot_kw)
        fig.colorbar(im, ax=ax, shrink=0.8)
    if title:
        ax.set_title(title + note, fontsize=9)
    return fig


def plot_sensitivity(sens: torch.Tensor, ax=None, title: str = "sensitivity ‖∂F/∂x_i‖"):
    """Sensitivity map (Jacobian column norms, NeTMY Eq. 23)."""
    return show_field(sens, ax, title, cmap="magma")


def plot_filter_kernels(
    rows: Mapping[str, torch.Tensor] | torch.Tensor,
    ax=None,
    normalize: bool = True,
    title: str = "filter-kernel rows G_θ e_i",
):
    """Overlay 1-D filter-kernel rows (or show one 2-D row) — NeTMY Lemma 2."""
    fig, ax = _axes(ax)
    items = {"row": rows} if torch.is_tensor(rows) else dict(rows)
    for label, r in items.items():
        r = torch.as_tensor(r).detach().float().cpu()
        if r.ndim >= 2:
            return show_field(r, ax, f"{title} [{label}]", cmap="coolwarm")
        if normalize and float(r.abs().max()) > 0:
            r = r / r.abs().max()
        ax.plot(r.numpy(), label=label)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("pixel")
    ax.legend(fontsize=7)
    return fig


def plot_energy_barrier(barriers: Any, ax=None, title: str = "loss along x(t)"):
    """Loss profile(s) along the interpolation path (NeTMY Fig. 4b)."""
    fig, ax = _axes(ax)
    items = barriers if isinstance(barriers, Mapping) else {"path": barriers}
    for label, eb in items.items():
        lab = f"{label} (h={eb.height:.3g})"
        ax.plot(eb.t.numpy(), eb.loss.numpy(), marker="o", ms=3, label=lab)
    ax.set_xlabel("interpolation t (a → b)")
    ax.set_ylabel("loss")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    return fig


def plot_singular_values(sv: Any, ax=None, title: str = "singular values of dF/dx"):
    """Semilog plot of singular-value decay (NeFTY Prop. 2)."""
    fig, ax = _axes(ax)
    items = sv if isinstance(sv, Mapping) else {"σ": sv}
    for label, s in items.items():
        s = _np(s)
        ax.semilogy(range(1, len(s) + 1), s, marker="o", ms=3, label=label)
    ax.set_xlabel("index n")
    ax.set_title(title, fontsize=9)
    if isinstance(sv, Mapping):
        ax.legend(fontsize=7)
    return fig


def plot_history(
    history: Any,
    ax=None,
    keys: Sequence[str] | None = None,
    logy: bool = True,
    title: str = "loss history",
):
    """Loss curves from ``Result.history`` (or a Result) with stage boundaries."""
    fig, ax = _axes(ax)
    h = history.history if hasattr(history, "history") else history
    skip = {"step", "global_step", "stage", "lr", "progress"}  # non-loss bookkeeping columns
    keys = keys or [k for k in h if k not in skip]
    x = h.get("global_step") or list(range(len(h.get("total", []))))
    for k in keys:
        v = h.get(k)
        if v and len(v) == len(x):
            ax.plot(x, v, label=k, lw=1)
    stages = h.get("stage")
    if stages:
        for i in range(1, len(stages)):
            if stages[i] != stages[i - 1]:
                ax.axvline(x[i], color="gray", ls=":", lw=0.8)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    return fig


def plot_result(result: Any, gt: Mapping[str, torch.Tensor] | None = None, measurement: Any = None):
    """Overview figure: every field (vs ground truth), prediction vs measurement, loss history."""
    names = list(result.fields)
    ncols = len(names) + 2
    fig, axes = new_figure(1, ncols, figsize=(3.6 * ncols, 3.2))
    for i, n in enumerate(names):
        ax = axes[0, i]
        f = result.fields[n]
        if gt is not None and n in gt and f.ndim == 1:
            ax.plot(_np(gt[n]), "k--", lw=1, label="gt")
            ax.plot(_np(f), lw=1, label="recon")
            ax.legend(fontsize=7)
            ax.set_title(n, fontsize=9)
        else:
            show_field(f, ax, f"{n} (recon)")
    ax = axes[0, len(names)]
    pred = result.pred
    if pred.ndim == 1:
        if measurement is not None:
            ax.plot(_np(measurement.data), "k.", ms=2, label="measurement")
        ax.plot(_np(pred), lw=1, label="prediction")
        ax.legend(fontsize=7)
        ax.set_title("data fit", fontsize=9)
    else:
        p = pred
        while p.ndim > 2:
            p = p[p.shape[0] // 2]
        show_field(p, ax, "prediction")
    plot_history(result, axes[0, len(names) + 1])
    return fig


def plot_report(report: Any):
    """Multi-panel figure for a :class:`~nefi.diagnostics.DiagnosticsReport`."""
    panels = []
    if report.sensitivity is not None:
        panels.append(("sens", report.sensitivity))
    if report.iter0 is not None:
        panels.append(("iter0", report.iter0))
    if report.filter_rows:
        panels.append(("filter", report.filter_rows))
    if report.singular_values is not None:
        panels.append(("sv", report.singular_values))
    if report.energy_barrier is not None:
        panels.append(("barrier", report.energy_barrier))
    if report.realized is not None:
        panels.append(("realized", report.realized))
    n = max(1, len(panels))
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = new_figure(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows))
    for i, (kind, obj) in enumerate(panels):
        ax = axes[i // ncols, i % ncols]
        if kind == "sens":
            plot_sensitivity(obj, ax)
        elif kind == "iter0":
            show_field(obj.grad.abs(), ax, f"|iter-0 gradient| (ratio {obj.ratio:.3g}×)", "magma")
        elif kind == "filter":
            plot_filter_kernels(obj, ax)
        elif kind == "sv":
            plot_singular_values(obj, ax)
        elif kind == "barrier":
            plot_energy_barrier(obj, ax)
        elif kind == "realized":
            show_field(obj.delta.abs(), ax, f"|Δx| after 1 step (ratio {obj.delta_ratio:.3g}×)")
    for j in range(len(panels), nrows * ncols):
        axes[j // ncols, j % ncols].set_axis_off()
    fig.suptitle(f"diagnostics — {report.name}", fontsize=10)
    return fig


__all__ = [
    "new_figure",
    "plot_energy_barrier",
    "plot_filter_kernels",
    "plot_history",
    "plot_report",
    "plot_result",
    "plot_sensitivity",
    "plot_singular_values",
    "save_figure",
    "show_field",
]
