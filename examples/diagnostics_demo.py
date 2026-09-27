"""Diagnostics tour on toy1d: why a neural field is not a free grid (NeTMY §4.4, Lemma 2).

Produces ``runs/diagnostics_demo/diagnostics_demo.png`` (six panels) and prints the
``nefi.diagnostics.diagnose`` markdown report. Runs in a few seconds on a CPU::

    python examples/diagnostics_demo.py [--out runs/diagnostics_demo] [--n 128]

Panels:
  1. filter-kernel rows G_θ e_i: a GridField realizes a delta (G_θ = I), a NeuralField a smooth
     bump that sharpens as the annealing progress β/K opens higher Fourier bands (Eq. 36-39);
  2. sensitivity ‖∂F/∂x_i‖ (NeTMY Eq. 23) for a narrow and a wide blur: the window truncates the
     footprint of edge pixels — the finite-window center bias (P2);
  3. iter-0 data gradient at a uniform initialization (what a grid solver executes first);
  4. realized first update |Δx| of a grid (SGD: verbatim gradient) vs the neural field (AdamW);
  5. loss along the straight path from a "centrally collapsed" field to the ground truth
     (NeTMY Fig. 4b; this toy loss is convex, so no barrier);
  6. singular values of dF/dx for both blurs (NeFTY Prop. 2: faster decay = more ill-posed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nefi import diagnostics as D
from nefi.diagnostics import plots
from nefi.fields import GridField, Heads, Softplus
from nefi.instances.toy1d import Toy1D
from nefi.problem import InverseProblem


def grid_version(problem: InverseProblem) -> InverseProblem:
    """Same problem with a free-pixel field (the Tikhonov / Grid Opt. parameterization)."""
    field = GridField(problem.domain.shape, Heads({"x": Softplus(init_value=0.1)}))
    return InverseProblem(
        problem.domain, field, problem.operator, problem.losses, problem.measurement
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="runs/diagnostics_demo")
    ap.add_argument("--n", type=int, default=128)
    args = ap.parse_args()
    torch.manual_seed(0)

    inst = Toy1D(n=args.n, scene="bumps")
    gt, meas = inst.make_measurement(seed=0)
    problem = inst.build_problem(meas)
    grid = grid_version(problem)
    wide = Toy1D(n=args.n, sigma=0.05)
    wide_problem = wide.build_problem(wide.make_measurement(seed=0)[1])
    n = args.n

    # 1. filter-kernel rows
    rows = {"GridField": D.filter_kernel_row(grid, n // 2)}
    for p in (0.0, 0.5, 1.0):
        rows[f"NeuralField β/K={p:g}"] = D.filter_kernel_row(problem, n // 2, progress=p)
    spreads = {k: D.kernel_spread(v, pixel=n // 2) for k, v in rows.items()}

    # 2. sensitivity maps (exact column norms; n JVPs)
    sens = D.sensitivity_map(problem, exact=True)
    sens_wide = D.sensitivity_map(wide_problem, exact=True)

    # 3. iter-0 gradient at a uniform initialization
    it0 = D.iter0_gradient(problem)

    # 4. realized first updates
    stage1 = problem.curriculum.stages[0]  # the iteration-0 stage (coarse grid, β = 0)
    ru_grid = D.realized_update(grid, stage1, optimizer="sgd")
    ru_nf = D.realized_update(problem, stage1)

    # 5. energy profile from a centrally collapsed field (same total mass) to the ground truth
    x = problem.domain.physical_coords()[..., 0]
    bump = torch.exp(-0.5 * ((x - 0.5) / 0.03) ** 2)
    collapsed = bump / bump.sum() * gt["x"].sum()
    eb = D.energy_barrier(problem, collapsed, gt, n=21)

    # 6. singular values for the narrow and the wide blur
    sv = D.singular_values(problem, k=24, n_iter=60)
    sv_wide = D.singular_values(wide_problem, k=24, n_iter=60)

    fig, axes = plots.new_figure(2, 3, figsize=(15, 8))
    plots.plot_filter_kernels(rows, axes[0, 0], title="G_θ e_i (center pixel), normalized")
    ax = axes[0, 1]
    ax.plot((sens / sens.max()).numpy(), label="σ=0.02 (toy1d)")
    ax.plot((sens_wide / sens_wide.max()).numpy(), label="σ=0.05")
    ax.set_title("sensitivity ‖∂F/∂x_i‖ / max (P2 window effect)", fontsize=9)
    ax.set_xlabel("pixel")
    ax.legend(fontsize=7)
    ax = axes[0, 2]
    g = it0.grad.abs()
    ax.plot((g / g.max()).numpy(), label="|∇ₓL| at uniform init")
    ax.plot((gt["x"] / gt["x"].max()).numpy(), "k--", lw=1, label="ground truth (scaled)")
    ax.set_title(f"iter-0 data gradient (center/outer {it0.ratio:.2f}×)", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    for label, ru in (("grid + SGD", ru_grid), ("neural field + AdamW", ru_nf)):
        d = ru.delta.abs()
        ax.plot((d / d.max()).numpy(), label=f"{label} (cos {ru.alignment:.2f})")
    ax.set_title("realized first update |Δx| (normalized, stage 1)", fontsize=9)
    ax.set_xlabel("pixel (stage-1 grid)")
    ax.legend(fontsize=7)
    plots.plot_energy_barrier({"collapse → GT": eb}, axes[1, 1])
    plots.plot_singular_values({"σ=0.02": sv, "σ=0.05": sv_wide}, axes[1, 2])

    out = Path(args.out)
    path = plots.save_figure(fig, out / "diagnostics_demo.png")
    report = D.diagnose(problem, gt=gt)
    report.save(out)
    print(report.to_markdown())
    print("filter-kernel half-max widths (px):", {k: round(v, 1) for k, v in spreads.items()})
    print(
        f"realized update: grid cos={ru_grid.alignment:.3f} damping={ru_grid.damping:.3f} | "
        f"neural cos={ru_nf.alignment:.3f} damping={ru_nf.damping:.2f}"
    )
    decay, decay_wide = float(sv[-1] / sv[0]), float(sv_wide[-1] / sv_wide[0])
    print(f"σ_24/σ_1: narrow blur {decay:.3g}, wide blur {decay_wide:.3g}")
    print(f"energy profile collapse → GT: height {eb.height:.3g}, monotone {eb.monotone}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
