"""Poisson source recovery: generate → invert (neural field) → evaluate → save a PNG in ``runs/``.

Recovers the source ``f`` of ``-Δu = f`` (Dirichlet) from ``u`` observed on a sparse random pixel
subset (or a boundary strip) through the differentiable spectral solver.

Usage::

    python examples/poisson_source.py                                  # smoke config (~5 s CPU)
    python examples/poisson_source.py --config configs/poisson_source_full.yaml --device cuda
    python examples/poisson_source.py --set obs_mode=boundary --set scene=smooth
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from nefi.config import load_config  # noqa: E402
from nefi.instances.poisson_source import PoissonSource  # noqa: E402


def load(path: str, overrides: list[str]) -> tuple[dict, dict]:
    """(instance config, run options) from a YAML file in either nefi layout, plus overrides."""
    d = load_config(path)
    inst = d.get("instance")
    cfg = {k: v for k, v in inst.items() if k != "type"} if isinstance(inst, dict) else {}
    cfg.update(d.get("config") or {})
    for item in overrides:
        k, v = item.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg, dict(d.get("run") or {})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/poisson_source_smoke.yaml")
    ap.add_argument("--set", action="append", default=[], help="config override key=value")
    ap.add_argument("--out", default=None, help="output directory (default: run.out)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg, run = load(args.config, args.set)
    seed = run.get("seed", 0) if args.seed is None else args.seed
    device = args.device or run.get("device", "auto")
    out = Path(args.out or run.get("out", "runs/poisson_source"))
    out.mkdir(parents=True, exist_ok=True)

    inst = PoissonSource(cfg)
    t0 = time.perf_counter()
    run_out = inst.run(seed=seed, device=device)  # 5-point FD on a 2x grid + mask + noise
    t_nf = time.perf_counter() - t0
    gt, meas, result = run_out.gt, run_out.measurement, run_out.result
    m = run_out.metrics
    c = inst.cfg
    frac = float(meas.mask.mean())
    print(f"poisson source ({c.scene}, {c.n}x{c.n}, {c.obs_mode} observations, {frac:.1%})")
    print(
        f"neural field: PSNR {m['psnr']:.2f} dB, relative error {m['relative_error']:.3f}, "
        f"{t_nf:.1f} s"
    )

    obs = meas.data.clone()
    obs[meas.mask == 0] = float("nan")
    u_true = inst.operator()({"f": gt["f"]})
    vmax = float(gt["f"].max())
    panels = [
        ("ground-truth source f", gt["f"], "viridis", (0, vmax)),
        (f"observed u ({frac:.0%} of pixels)", obs, "magma", None),
        (
            f"recovered f\n{m['psnr']:.2f} dB / rel.err {m['relative_error']:.3f}",
            result.fields["f"],
            "viridis",
            (0, vmax),
        ),
        ("predicted u (full grid)", result.pred, "magma", (0, float(u_true.max()))),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    for ax, (title, img, cmap, lim) in zip(axes, panels):
        kw = {} if lim is None else {"vmin": lim[0], "vmax": lim[1]}
        ax.imshow(torch.as_tensor(img).T, origin="lower", cmap=cmap, **kw)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    path = out / "poisson_source.png"
    fig.savefig(path, dpi=110)
    result.save(out / "result.pt")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
