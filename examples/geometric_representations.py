"""Geometric representations on a synthetic "defect in a two-layer medium" deblurring toy.

A 2-D cross-section ``(x, z)`` of a laminate — a top layer over a bottom layer separated by an
undulating interface — contains an elliptical low-value defect (NeFTY's layered setting in
miniature). The forward model is the core :class:`~nefi.operators.FFTConvolution` Gaussian blur
(no PDE); data are simulated on a 2× finer grid in float64 and area-averaged (inverse-crime guard),
with 2 % noise. Four representations are fitted at the same small budget (600 steps):

* ``NeuralField``   — generic coordinate MLP with annealed Fourier features (NeTMY / NeFTY);
* ``GridField``     — free pixels (``G_θ = I``, NeTMY §4.4);
* ``LayerCakeField`` — layered background (monotone interfaces) ⊕ gated anomaly (NeTMY Eq. 5);
* ``Composite``     — ``CompositeField`` blend of a ``FourierBasisField`` background, a learnable
  inclusion value and a level-set-like indicator.

The script prints an accuracy table (MSE, PSNR, defect IoU — NeFTY's segmentation metric), the
parameters the layered representation recovers, the ``match_report`` (representation pass band
vs. what the data resolve at the noise level) for each representation, a held-out-entries model
selection (``select_representation``), and saves a figure to
``runs/geometric_representations.png``.

Run::

    python examples/geometric_representations.py            # ~1-2 min on a CPU
    python examples/geometric_representations.py --quick    # smaller budget
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import nefi
from nefi.bench.base import DataGenerator
from nefi.fields import GatedSoftplus, GridField, Heads, NeuralField
from nefi.fields.adaptive import match_report, operator_spectrum, select_representation
from nefi.fields.geometric import (
    CompositeField,
    FourierBasisField,
    LayerCakeField,
    level_set_head,
)
from nefi.losses import MSE, TV, LossSet
from nefi.metrics import iou, mse, psnr
from nefi.operators import FFTConvolution, gaussian_kernel_fn
from nefi.solve import Curriculum, Stage
from nefi.utils.seed import seed_everything

N = 48  # native grid
SIGMA = 0.035  # blur std (physical units, unit square): ≈1.7 cells
NOISE = 0.02  # relative to the clean signal's max
V_TOP, V_BOTTOM, V_DEFECT = 1.0, 0.55, 0.1
DEFECT_THRESHOLD = 0.3  # defect mask x < 0.3 (NeFTY App. E.2 style IoU)


def scene(shape: tuple[int, int]) -> torch.Tensor:
    """Ground truth on a grid of ``shape`` over ``[0, 1]²`` (axes x, z; z = depth)."""
    dom = nefi.Domain.unit(shape, axes=("x", "z"))
    p = dom.physical_coords().double()
    x, z = p[..., 0], p[..., 1]
    interface = 0.45 + 0.06 * torch.sin(2 * np.pi * x)
    out = torch.where(z < interface, V_TOP, V_BOTTOM)
    ellipse = ((x - 0.62) / 0.13) ** 2 + ((z - 0.66) / 0.075) ** 2 < 1.0
    return torch.where(ellipse, torch.full_like(out, V_DEFECT), out)


def make_data(seed: int = 0):
    dom = nefi.Domain.unit((N, N), axes=("x", "z"))
    fine = dom.refine(2)
    gen = DataGenerator(
        FFTConvolution(gaussian_kernel_fn(SIGMA), fine),
        noise_std=NOISE,
        relative=True,
        supersample=2,
        fidelity_tag="gaussian-blur-2x-float64",
    )
    meas = gen.generate({"x": scene(fine.shape)}, np.random.default_rng(seed), target_shape=(N, N))
    return dom, scene((N, N)).float(), meas


def problem_for(field, dom, meas, tv: float = 2e-4):
    op = FFTConvolution(gaussian_kernel_fn(SIGMA), dom)
    losses = LossSet({"data": MSE(), "tv": TV("x", isotropic=True)}, {"data": 1.0, "tv": tv})
    return nefi.InverseProblem(dom, field, op, losses, meas, name="layered-defect")


# --- the four representations ---------------------------------------------------------------
def neural_field():
    return NeuralField(
        2, Heads({"x": "identity"}), hidden=64, depth=3, skip_at=2, n_octaves=6, out_bias=[0.7]
    )


def grid_field():
    return GridField((N, N), Heads({"x": "identity"}), init=0.7)


def layer_cake():
    anomaly = NeuralField(
        2,
        Heads({"a": GatedSoftplus(init_value=0.02, gate_init=-2.0)}),  # NeTMY Eq. 5 gate
        hidden=32,
        depth=3,
        n_octaves=5,
    )
    return LayerCakeField(
        2,
        n_layers=2,
        anomaly=anomaly,
        mode="add",
        contrast=-1.0,  # defects lower the host value
        depth_axis=1,
        layer_values=[0.75, 0.65],  # uninformative start (no oracle values)
        interface_model="grid",
        layer_kw={"lateral_shape": 8, "eps_start": 0.3, "eps_end": 0.01},
        progress_map={"background": ("fast", 0.5), "anomaly": ("delay", 0.1)},
    )


def fourier_composite():
    indicator = NeuralField(
        2,
        Heads({"m": level_set_head(0.0, 1.0, 1.0, 0.1, "geometric", 0.05)}),
        hidden=32,
        depth=3,
        n_octaves=5,
    )
    return CompositeField(
        {
            "background": FourierBasisField(2, n_modes=8, heads=Heads({"x": {"type": "identity"}})),
            "insert": FourierBasisField(2, n_modes=1),  # a learnable constant (inclusion value)
            "mask": indicator,
        },
        "blend",
        progress_map={"background": ("fast", 0.5), "insert": "full", "mask": ("delay", 0.1)},
    )


CANDIDATES = {  # name -> (factory, learning rate from a small sweep in {3e-3, 1e-2, 2e-2, 4e-2})
    "NeuralField": (neural_field, 1e-2),
    "GridField": (grid_field, 1e-2),
    "LayerCakeField": (layer_cake, 4e-2),
    "Composite(Fourier+levelset)": (fourier_composite, 1e-2),
}


def curriculum(lr: float, steps: tuple[int, int]) -> Curriculum:
    return Curriculum(
        [
            Stage("coarse", (N // 2, N // 2), steps[0], lr),
            Stage("fine", (N, N), steps[1], 0.5 * lr),
        ]
    )


def init_fb_background(field: CompositeField) -> None:
    """Start the Fourier background at the data mean level (no oracle information)."""
    with torch.no_grad():
        field.background.coef[0, 0, 0] = 0.7
        field.insert.coef[0, 0, 0] = 0.4


def main(quick: bool = False, out: str = "runs") -> None:
    nefi.enable_logging(30)
    steps = (100, 150) if quick else (200, 400)
    dom, gt, meas = make_data(0)
    print(f"domain {dom.shape}, blur σ={SIGMA}, noise σ={meas.noise_std:.4f} ({NOISE:.0%} of max)")
    rows, fits = [], {}
    op_spec = None
    for name, (make, lr) in CANDIDATES.items():
        seed_everything(0)
        field = make()
        if name.startswith("Composite"):
            init_fb_background(field)
        prob = problem_for(field, dom, meas)
        t0 = time.perf_counter()
        res = nefi.invert(prob, curriculum(lr, steps), device="cpu", seed=0)
        secs = time.perf_counter() - t0
        x = res.fields["x"]
        if op_spec is None:  # operator sensitivity at the fitted solution (shared by all reports)
            op_spec = operator_spectrum(prob.operator, dom, {"x": x}, n_probes=6)
        rep = match_report(field, prob, n_probes=6, operator_spec=op_spec)
        fits[name] = (x, field, rep)
        defect_iou = iou(x < DEFECT_THRESHOLD, gt < DEFECT_THRESHOLD)
        rows.append(
            (
                name,
                field.n_parameters(),
                mse(x, gt),
                psnr(x, gt, data_range=1.0),
                defect_iou,
                secs,
                rep.verdict,
                rep.rep_bandwidth,
                rep.data_bandwidth,
            )
        )

    print(
        "\n| representation | params | MSE | PSNR [dB] | defect IoU | time [s] | match verdict "
        "| ν_rep | ν_data |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r[0]} | {r[1]} | {r[2]:.2e} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.1f} | {r[6]} | "
            f"{r[7]:.2f} | {r[8]:.2f} |"
        )
    cake = fits["LayerCakeField"][1]
    with torch.no_grad():
        vals = [round(float(v), 3) for v in cake.layers.layer_values[:, 0]]
        z = float(cake.layers.interfaces()[0])
    print(
        f"\nLayerCakeField recovered layer values {vals} (truth {[V_TOP, V_BOTTOM]}), mean "
        f"interface depth {(z + 1) / 2:.3f} (truth 0.450)"
    )
    for name, (_, _, rep) in fits.items():
        print(f"\n### match_report — {name}\n{rep.text}")

    # --- measurement-driven model selection (held-out entries of the same measurement) -------
    print("\n### select_representation (10 % held-out measurement entries, budget ×0.3)")
    report = select_representation(
        lambda f: problem_for(f, dom, meas),
        {n: (lambda m=m, n=n: _prepared(m, n)) for n, (m, _) in CANDIDATES.items()},
        holdout=0.1,
        seed=0,
        budget_scale=0.3,
        curricula={n: curriculum(lr, steps) for n, (_, lr) in CANDIDATES.items()},
        device="cpu",
        keep_fields=False,
    )
    print(report.table())

    _figure(dom, gt, meas, fits, Path(out) / "geometric_representations.png")


def _prepared(make, name):
    f = make()
    if name.startswith("Composite"):
        init_fb_background(f)
    return f


def _figure(dom, gt, meas, fits, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        print("matplotlib not installed; skipping the figure")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [("ground truth", gt), (f"observation (blur + {NOISE:.0%} noise)", meas.data)]
    panels += [(f"{n}\n{psnr(x, gt, data_range=1.0):.1f} dB", x) for n, (x, _, _) in fits.items()]
    fig, axes = plt.subplots(2, len(panels), figsize=(3.0 * len(panels), 6.0))
    for j, (title, img) in enumerate(panels):
        ax = axes[0, j]
        ax.imshow(img.T.numpy(), cmap="viridis", vmin=0.0, vmax=1.1, origin="upper")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax2 = axes[1, j]
        if j < 2:
            ax2.axis("off")
            continue
        ax2.imshow((img - gt).abs().T.numpy(), cmap="magma", vmin=0.0, vmax=0.5, origin="upper")
        ax2.set_title("|error|", fontsize=9)
        ax2.set_xticks([])
        ax2.set_yticks([])
    # spectra: representation transfer vs operator sensitivity
    ax = axes[1, 1]
    ax.axis("on")
    first = next(iter(fits.values()))[2]
    op = first.operator
    ax.semilogy(op.freqs, (op.values / op.values.max()) ** 2, "k-", lw=2, label="σ_F² (operator)")
    for n, (_, _, rep) in fits.items():
        r = rep.representation
        v = r.values / r.reference("max")
        ax.semilogy(r.freqs[1:], v[1:].clamp_min(1e-6), label=n.split("(")[0])
    ax.axvline(first.data_bandwidth, color="gray", ls="--", lw=1)
    ax.set_xlabel("cycles / unit (normalized)", fontsize=8)
    ax.set_title("update-kernel transfer vs σ_F²", fontsize=9)
    ax.legend(fontsize=6)
    ax.set_ylim(1e-5, 2)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print(f"\nfigure saved to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="smaller budget")
    ap.add_argument("--out", default="runs", help="output directory")
    args = ap.parse_args()
    main(args.quick, args.out)
