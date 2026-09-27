import math

import torch

from nefi.baselines.thermography import contrast_mask, depth_to_alpha_volume, ppt, tsr


def slab_decay(t: torch.Tensor, L: float, alpha: float, n_terms: int = 60) -> torch.Tensor:
    """Finite slab after a surface flash (adiabatic back): T ∝ (1/L)(1 + 2Σ e^{-n²π²αt/L²})."""
    n = torch.arange(1, n_terms + 1, dtype=torch.float64)[:, None]
    return (1.0 / L) * (
        1.0 + 2.0 * torch.exp(-(n**2) * math.pi**2 * alpha * t[None, :] / L**2).sum(0)
    )


def _two_region_frames(alpha=0.1, L_shallow=0.3, L_deep=0.8, n_t=200, dt=0.01, noise=1e-4):
    t = dt * torch.arange(1, n_t + 1, dtype=torch.float64)
    shallow = slab_decay(t, L_shallow, alpha)
    deep = slab_decay(t, L_deep, alpha)
    frames = torch.empty(n_t, 8, 8, dtype=torch.float64)
    frames[:, :, :4] = deep[:, None, None]  # "sound" majority = the thick (deep) region
    frames[:, :, 4:] = shallow[:, None, None]
    frames[:, :, 4:6] = deep[:, None, None]  # make the sound region the majority (3/4)
    g = torch.Generator().manual_seed(0)
    frames += noise * torch.randn(frames.shape, generator=g, dtype=torch.float64)
    return frames, t


def test_tsr_recovers_slab_depth_ordering_and_scale():
    alpha = 0.1
    frames, t = _two_region_frames(alpha=alpha)
    out = tsr(frames, dt=0.01, alpha=alpha, t0=0.01)
    d = out["depth"]
    assert d.shape == (8, 8) and torch.isfinite(d).all()
    d_shallow = float(d[:, 6:].mean())
    d_deep = float(d[:, :4].mean())
    assert d_shallow < d_deep
    # TSR's calibration constant is approximate; expect the right order of magnitude
    assert 0.5 * 0.3 < d_shallow < 2.0 * 0.3, d_shallow
    mask = out["mask"]
    assert mask[:, 6:].float().mean() > 0.9 and mask[:, :4].float().mean() < 0.1


def test_ppt_phase_contrast_marks_shallow_region():
    alpha = 0.1
    frames, t = _two_region_frames(alpha=alpha)
    out = ppt(frames, dt=0.01, alpha=alpha)
    assert out["depth"].shape == (8, 8) and torch.isfinite(out["depth"]).all()
    c = out["contrast"]
    assert float(c[:, 6:].mean()) > 5 * float(c[:, :4].mean())
    assert out["mask"][:, 6:].all()
    assert out["mask"][:, :4].float().mean() < 0.1  # MAD floor on near-identical sound pixels
    blind = ppt(frames, dt=0.01, alpha=alpha, mode="blind")
    assert torch.isfinite(blind["depth"]).all()


def test_contrast_mask_and_volume_lift():
    c = torch.zeros(6, 6)
    c[1, 1] = 5.0
    m = contrast_mask(c, k=2.0)
    assert m[1, 1] and m.sum() == 1
    depth = torch.full((6, 6), 0.4)
    vol = depth_to_alpha_volume(depth, m, nz=10, thickness=1.0, alpha_bulk=0.15, alpha_defect=0.003)
    assert vol.shape == (6, 6, 10)
    col = vol[1, 1]
    assert (col[:4] == 0.15).all() and (col[4:6] == 0.003).all() and (col[6:] == 0.15).all()
    assert (vol[0, 0] == 0.15).all()


def test_thermal_instance_ppt_tsr_baselines_run():
    import nefi
    from nefi.baselines import solve as baseline_solve
    from nefi.instances.thermal_tomography import ThermalTomography

    inst = ThermalTomography(preset="smoke")
    gt, meas = inst.make_measurement(seed=0)
    for kind in ("ppt", "tsr"):
        prob, cur = inst.baselines()[kind](meas)
        res = baseline_solve(prob, cur, device="cpu")
        assert isinstance(res, nefi.Result)
        assert tuple(res.fields["alpha"].shape) == tuple(inst.cfg.grid)
        metrics = inst.evaluate(res, gt, meas)
        assert "iou" in metrics and math.isfinite(metrics["iou"])
