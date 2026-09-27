import numpy as np
import pytest
import torch

import nefi
from nefi.instances.thermal_tomography import ThermalTomography
from nefi.instances.thermal_tomography.realdata import (
    PhysicalScales,
    config_for_frames,
    load_frames,
    measurement_from_frames,
    physical_depth_map,
    pvc_overrides,
)


def test_physical_scales_round_trip():
    s = PhysicalScales(alpha_phys=1.2e-7, L0=0.1, thickness=0.005, t_total=20.0)
    a_sim = s.fourier_number
    assert abs(a_sim - 1.2e-7 * 20.0 / 0.01) < 1e-15
    back = float(s.physical_alpha(torch.tensor(a_sim, dtype=torch.float64)))
    assert abs(back / 1.2e-7 - 1.0) < 1e-12
    ext = s.extent()
    assert ext[:2] == (1.0, 1.0) and abs(ext[2] - 0.05) < 1e-12
    assert abs(float(physical_depth_map(torch.tensor(0.02), s, "mm")) - 2.0) < 1e-9


def test_load_frames_and_measurement_from_frames(tmp_path):
    rng = np.random.default_rng(0)
    frames = 20.0 + rng.random((12, 8, 8)).astype(np.float32)  # ambient 20 + decay-ish noise
    np.save(tmp_path / "seq.npy", frames)
    x = load_frames(tmp_path / "seq.npy")
    assert x.shape == (12, 8, 8)
    meas, meta = measurement_from_frames(
        x, dt=0.1, pre_flash_frames=2, first_frame=1, frame_stride=2, resample_to=(4, 4)
    )
    assert meas.shape == (5, 4, 4)
    assert meta["n_frames_used"] == 5 and meta["frame_stride"] == 2
    assert abs(meta["ambient"] - float(x[:2].mean())) < 1e-4
    assert meas.noise_std is None or meas.noise_std >= 0
    torch.save({"frames": torch.as_tensor(frames)}, tmp_path / "seq.pt")
    assert load_frames(tmp_path / "seq.pt", key="frames").shape == (12, 8, 8)
    from nefi.errors import ConfigError

    with pytest.raises(ConfigError):
        load_frames(tmp_path / "missing.npy")


def test_config_for_frames_builds_a_consistent_problem():
    inst = ThermalTomography(preset="smoke")
    gt, meas_syn = inst.make_measurement(seed=0)
    frames = meas_syn.data + 20.0  # pretend the camera recorded absolute temperatures
    scales = PhysicalScales(alpha_phys=1.2e-7, L0=0.1, thickness=0.005, t_total=10.0)
    meas, meta = measurement_from_frames(frames, dt=10.0 / frames.shape[0], t_ambient=20.0)
    cfg = config_for_frames(
        meas, scales, nz=4, base=inst.config_dict(), **pvc_overrides(scales, robin_h=0.1)
    )
    cfg.update({"steps": 3, "hidden": 8, "depth": 2, "n_octaves": 2, "anneal_steps": 1})
    real = ThermalTomography(cfg)
    assert real.cfg.initial == "flash" and real.cfg.bc_back == "robin"
    assert real.cfg.grid[:2] == tuple(frames.shape[1:]) and real.cfg.grid[2] == 4
    assert real.cfg.n_frames == frames.shape[0]
    problem = real.build_problem(meas)
    assert tuple(problem.operator.output_shape(real.cfg.grid)) == tuple(meas.shape)
    res = nefi.invert(problem, device="cpu")
    assert tuple(res.fields["alpha"].shape) == tuple(real.cfg.grid)
    assert float(scales.physical_alpha(res.fields["alpha"]).mean()) > 0
