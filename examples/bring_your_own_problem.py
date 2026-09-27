"""Bring your own problem: three inverse problems solved from a plain forward model.

1. **Saturating camera** — ``y = tanh(1.5 · blur(x))``, written as a lambda (2-D, 48²).
2. **Accelerated MRI** — 30 % variable-density k-space samples (``FourierSampling``) with a
   hash-grid field (2-D, 64²), compared with the zero-filled inverse FFT.
3. **1-D photoacoustics** — recover the initial pressure ``p0`` of the wave equation from two
   boundary sensors through a differentiable :class:`~nefi.operators.TimeStepper` with
   gradient checkpointing.

Each case prints :func:`nefi.quick_report` and saves a PNG to ``runs/byop/`` (matplotlib
optional). Run: ``python examples/bring_your_own_problem.py [--quick] [--out runs/byop]``.
"""

from __future__ import annotations

import argparse
import math
from functools import partial
from pathlib import Path

import torch

import nefi
from nefi.metrics import psnr
from nefi.operators import (
    FourierSampling,
    TimeStepper,
    random_kspace_mask,
    stable_dt,
    wave_initial_state,
    wave_step,
)


def gaussian_blur_2d(sigma_px: float):
    """A user-side forward model: separable Gaussian blur with zero padding."""
    r = torch.arange(-math.ceil(4 * sigma_px), math.ceil(4 * sigma_px) + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (r / sigma_px) ** 2)
    k = (k / k.sum())[None, None]
    pad = k.shape[-1] // 2

    def blur(x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.conv2d(x[None, None], k[..., None], padding=(pad, 0))
        return torch.nn.functional.conv2d(x, k[:, :, None, :], padding=(0, pad))[0, 0]

    return blur


def shapes_image(n: int) -> torch.Tensor:
    u = (torch.arange(n) + 0.5) / n
    X, Y = torch.meshgrid(u, u, indexing="ij")
    img = 0.8 * (((X - 0.35) ** 2 + (Y - 0.4) ** 2) < 0.18**2).float()
    img += 0.5 * (((X - 0.7).abs() < 0.12) & ((Y - 0.65).abs() < 0.2)).float()
    img += 0.6 * torch.exp(-((X - 0.72) ** 2 + (Y - 0.25) ** 2) / (2 * 0.06**2))
    return img


def mri_phantom(n: int) -> torch.Tensor:
    u = (torch.arange(n) + 0.5) / n * 2 - 1
    X, Y = torch.meshgrid(u, u, indexing="ij")
    img = 1.0 * ((X / 0.85) ** 2 + (Y / 0.65) ** 2 < 1).float()
    img -= 0.6 * ((X / 0.75) ** 2 + (Y / 0.55) ** 2 < 1).float()
    img += 0.5 * (((X - 0.25) / 0.2) ** 2 + ((Y + 0.1) / 0.15) ** 2 < 1).float()
    img += 0.3 * (((X + 0.3) / 0.1) ** 2 + ((Y - 0.2) / 0.25) ** 2 < 1).float()
    img += 0.2 * (((X + 0.05) / 0.06) ** 2 + ((Y + 0.3) / 0.06) ** 2 < 1).float()
    return img


def save_png(path: Path, panels: dict[str, torch.Tensor]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        print(f"(matplotlib not installed: skipping {path})")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.0))
    for ax, (title, img) in zip(axes, panels.items()):
        img = img.detach().cpu()
        if img.ndim == 1:
            ax.plot(img.numpy())
        else:
            ax.imshow(img.numpy(), cmap="gray")
            ax.axis("off")
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"saved {path}")


def case_camera(out: Path, budget: int) -> None:
    n = 48
    blur = gaussian_blur_2d(1.5)
    camera = lambda x: torch.tanh(1.5 * blur(x))  # noqa: E731 - "any differentiable function"
    x_true = shapes_image(n)
    y = camera(x_true) + 0.01 * torch.randn(n, n, generator=torch.Generator().manual_seed(0))

    problem = nefi.from_forward(
        camera,
        y,
        shape=(n, n),
        prior="nonnegative + piecewise_constant",
        budget=budget,
        name="saturating-camera",
    )
    result = nefi.invert(problem)
    print(nefi.quick_report(result, problem, gt=x_true), "\n")
    save_png(out / "camera.png", {"truth": x_true, "measurement": y, "nefi": result.fields["x"]})


def case_mri(out: Path, budget: int) -> None:
    n = 64
    x_true = mri_phantom(n)
    op = FourierSampling(random_kspace_mask((n, n), fraction=0.3, seed=0))
    k = op({"x": x_true})
    k = k + 0.01 * float(k.abs().max()) * torch.randn(
        k.shape, generator=torch.Generator().manual_seed(1)
    )
    meas = nefi.Measurement(k * op.measurement_mask(), mask=op.measurement_mask())
    zero_filled = op.adjoint(meas.data)

    # 30 % k-space is strongly undersampled: a 3x stronger TV than the default 1e-2 pays off
    problem = nefi.from_forward(
        op,
        meas,
        shape=(n, n),
        prior="nonnegative + piecewise_constant(tv=3e-2)",
        representation="hash",
        budget=budget,
        name="mri-30pct",
    )
    result = nefi.invert(problem)
    print(nefi.quick_report(result, problem, gt=x_true))
    print(f"zero-filled IFFT baseline: PSNR {psnr(zero_filled, x_true):.2f} dB\n")
    save_png(
        out / "mri.png",
        {
            "truth": x_true,
            "k-space mask": op.mask,
            "zero-filled": zero_filled,
            "nefi (hash grid)": result.fields["x"],
        },
    )


def case_wave(out: Path, budget: int) -> None:
    n = 64
    dom = nefi.Domain.unit((n,))
    h, c = dom.spacing(), 1.0
    dt = stable_dt(h, c_max=c, cfl=0.5)
    op = TimeStepper(
        step=partial(wave_step, spacing=h, c=c, boundary="absorbing"),
        init=lambda f: wave_initial_state(f["p0"], c=c, dt=dt, spacing=h),
        n_steps=2 * n,  # the wave crosses the domain once
        dt=dt,
        observe=lambda state, i: state[1][[0, -1]],  # pressure at both ends, every step
        field="p0",
        grad_mode="checkpoint",
        checkpoint_every=32,
        homogeneity=1.0,
    )
    t = dom.physical_coords()[..., 0]
    p0 = torch.exp(-0.5 * ((t - 0.3) / 0.03) ** 2) + 0.6 * ((t - 0.65).abs() < 0.08).float()
    with torch.no_grad():
        traces = op({"p0": p0})
    traces = traces + 0.005 * torch.randn(traces.shape, generator=torch.Generator().manual_seed(2))

    problem = nefi.from_forward(
        op,
        traces,
        shape=(n,),
        field_name="p0",
        prior="nonnegative",
        representation="hash",
        budget=budget,
        name="photoacoustic-1d",
    )
    result = nefi.invert(problem)
    print(nefi.quick_report(result, problem, gt=p0), "\n")
    save_png(
        out / "wave.png",
        {"true p0": p0, "sensor traces": traces[:, 0], "recovered p0": result.fields["p0"]},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="runs/byop", help="output directory for PNGs")
    ap.add_argument("--quick", action="store_true", help="smaller budgets (CI smoke run)")
    args = ap.parse_args()
    out = Path(args.out)
    torch.manual_seed(0)
    case_camera(out, 150 if args.quick else 400)
    case_mri(out, 150 if args.quick else 400)
    case_wave(out, 100 if args.quick else 200)


if __name__ == "__main__":
    main()
