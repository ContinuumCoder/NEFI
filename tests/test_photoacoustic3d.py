"""photoacoustic3d: 3-D initial-value wave operator, sensor bandwidth, adjoint, scenes,
inverse-crime guard, time reversal, inversion."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nefi.baselines import solve
from nefi.errors import ConfigError
from nefi.instances.photoacoustic3d import (
    Photoacoustic3D,
    Photoacoustic3DConfig,
    SensorBandwidth,
    gaussian_lowpass,
)
from nefi.metrics import psnr
from nefi.registry import build

DT = torch.float64
TINY = dict(extent=(3.0, 3.0, 2.0), grid=(12, 12, 8), t_max=3.0, sensor_grid=(4, 4))


def test_geometry_and_output_shapes():
    inst = Photoacoustic3D(preset="smoke")
    sensors = inst.sensors()
    assert sensors.shape == (64, 3) and float(sensors[:, 2].abs().max()) == 0.0  # top face
    op = inst.operator()
    assert isinstance(op, SensorBandwidth) and op.homogeneity == 1.0
    assert op.output_shape((20, 20, 14)) == (64, inst.n_t) == op.output_shape((10, 10, 7))
    coarse = op.at_resolution((10, 10, 7))
    assert coarse is op.at_resolution((10, 10, 7)) and coarse.inner.domain.shape == (10, 10, 7)
    with pytest.raises(ConfigError):
        Photoacoustic3D(absorbing="pml")  # the 3-D wave solver has no PML


def test_lowpass_is_zero_phase_and_attenuates():
    t = torch.arange(200, dtype=DT) * 0.05
    slow, fast = torch.sin(2 * torch.pi * 0.2 * t), torch.sin(2 * torch.pi * 4.0 * t)
    out = gaussian_lowpass(torch.stack([slow, fast]), 0.05, 1.0)
    assert float((out[0] - slow)[20:-20].abs().max()) < 0.05  # passband, no phase shift
    assert float(out[1][20:-20].abs().max()) < 1e-3  # 4 MHz ≫ 1 MHz cutoff
    assert gaussian_lowpass(slow, 0.05, None) is slow


def test_linearity_and_exact_adjoint():
    inst = Photoacoustic3D(**TINY)
    op = inst.operator().to(DT)
    torch.manual_seed(0)
    a, b = torch.rand(12, 12, 8, dtype=DT), torch.rand(12, 12, 8, dtype=DT)
    ya, yb = op({"p0": a}), op({"p0": b})
    assert torch.allclose(op({"p0": 2 * a - 0.5 * b}), 2 * ya - 0.5 * yb, atol=1e-10)
    y = torch.randn_like(ya)
    x = torch.zeros(12, 12, 8, dtype=DT, requires_grad=True)
    (adj,) = torch.autograd.grad((op({"p0": x}) * y).sum(), x)
    assert float((ya * y).sum()) == pytest.approx(float((a * adj).sum()), rel=1e-9)


def test_scenes_consistent_and_inside_slab():
    inst = Photoacoustic3D(preset="smoke")
    sc = inst.scene_generator()
    for cls in sc.classes:
        a = sc.sample(np.random.default_rng(4), cls)["p0"]
        b = sc.sample(np.random.default_rng(4), cls, (40, 40, 28))["p0"]
        assert a.shape == (20, 20, 14) and float(a.min()) >= 0.0 and float(a.max()) <= 1.0
        assert float(a.max()) > 0.3, cls
        assert torch.allclose(F.avg_pool3d(b[None, None], 2)[0, 0], a, atol=1e-5), cls


def test_inverse_crime_guard_and_model_error():
    inst = Photoacoustic3D(preset="smoke", noise_std=0.0)
    gen = inst.data_generator()
    gt, meas = inst.make_measurement(0)
    prob = inst.build_problem(meas)
    assert gen.fidelity_tag.startswith("wave-ic-o2-2x-float64")
    assert gen.fidelity_tag != prob.operator.fidelity_tag
    assert gen.operator.inner.domain.shape == (40, 40, 28) and gen.dtype == torch.float64
    assert meas.shape == (64, inst.n_t)
    # band-limited sensors keep the native-vs-2× model error at ≈ 2 % of the peak trace
    pred = inst.operator().to(DT)({"p0": gt["p0"].double()})
    y = meas.data.double()
    assert float((pred - y).square().mean().sqrt() / y.abs().max()) < 0.035


def test_time_reversal_baselines():
    inst = Photoacoustic3D(preset="smoke")
    gt, meas = inst.make_measurement(0)
    tr = inst.time_reversal(meas)
    assert tr.shape == (20, 20, 14) and float(tr.min()) >= 0.0
    assert float(torch.corrcoef(torch.stack([tr.flatten(), gt["p0"].flatten()]))[0, 1]) > 0.3
    builders = inst.baselines()
    assert set(builders) == {"grid", "time_reversal"}
    prob, cur = builders["time_reversal"](meas)
    res = solve(prob, cur, device="cpu")
    assert res.stage_results[0]["stop"] == "closed_form"
    assert torch.allclose(res.fields["p0"], tr.float(), atol=1e-5)
    prob, cur = builders["grid"](meas)
    res = solve(prob, cur.scaled(0.05), device="cpu")
    assert torch.isfinite(res.fields["p0"]).all()


def test_run_smoke_beats_time_reversal():
    inst = Photoacoustic3D(preset="smoke")
    out = inst.run(seed=0, device="cpu")
    assert set(out.metrics) == {"psnr", "ssim", "mse"}
    p0 = out.result.fields["p0"]
    assert p0.shape == (20, 20, 14) and float(p0.min()) >= 0.0  # Softplus head
    tr = inst.time_reversal(out.measurement)
    # observed (seed 0): NF ≈ 18 dB / SSIM 0.58 vs time reversal ≈ 12 dB / 0.36
    assert out.metrics["psnr"] > psnr(tr, out.gt["p0"]) + 2.0, out.metrics


def test_registry_and_presets():
    inst = build("instance", {"type": "photoacoustic3d", "preset": "spheres", **TINY})
    assert isinstance(inst.cfg, Photoacoustic3DConfig) and inst.cfg.scene == "spheres"
    assert inst.n_t == 13 and inst.sensors().shape == (16, 3)


def test_smoke_config_file_matches_preset():
    """The smoke YAML (preferred by ``nefi run --smoke``) equals PRESETS["smoke"]."""
    from nefi.cli import make_instance, read_spec_file

    spec = read_spec_file(
        Path(__file__).resolve().parents[1] / "configs" / "photoacoustic3d_smoke.yaml"
    )

    def norm(cfg):
        return {k: tuple(v) if isinstance(v, list) else v for k, v in asdict(cfg).items()}

    assert norm(make_instance(spec).cfg) == norm(Photoacoustic3D(preset="smoke").cfg)
