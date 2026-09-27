"""Elliptic-family instances (eit, darcy_flow, current_density): smoke runs and physics checks."""

from pathlib import Path

import numpy as np
import pytest
import torch

import nefi
from nefi.config import load_config
from nefi.registry import build, list_registered
from nefi.solve.result import Result

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["eit", "darcy_flow", "current_density"]
PRIMARY = {"eit": "sigma", "darcy_flow": "log_k", "current_density": "g"}
# (higher-is-better metric, lower-is-better metric) that must beat the initial uniform field
KEY_METRICS = {
    "eit": ("psnr", "mse"),
    "darcy_flow": ("psnr", "mse"),
    "current_density": ("j_psnr", "j_relative_error"),
}


def _smoke(name, **overrides):
    cfg = load_config(ROOT / "configs" / f"{name}_smoke.yaml")
    inst = build("instance", {**cfg["instance"], **overrides})
    return inst, cfg.get("run", {})


INITIAL = {
    "eit": lambda c: c.sigma_bg,
    "darcy_flow": lambda c: c.log_k_mean,
    "current_density": lambda c: 0.0,
}


def _uniform(inst, name, gt, meas):
    """The initial field of the default inversion (uniform background / zero current)."""
    key = PRIMARY[name]
    f = torch.full_like(gt[key], INITIAL[name](inst.cfg))
    with torch.no_grad():
        pred = inst.operator()({key: f})
    return Result({key: f}, {key: f}, pred, {})


def test_instances_registered_and_available():
    reg = list_registered("instance")["instance"]
    from nefi.instances import available

    for name in NAMES:
        assert name in reg and name in available


@pytest.mark.parametrize("name", NAMES)
def test_smoke_run_shapes_loss_and_metrics(name):
    inst, run = _smoke(name)
    out = inst.run(seed=run.get("seed", 0), device="cpu")
    res, gt, meas = out.result, out.gt, out.measurement
    n = inst.cfg.n
    key = PRIMARY[name]
    assert res.fields[key].shape == (n, n)
    assert res.pred.shape == meas.data.shape
    assert tuple(meas.data.shape) == inst.operator().output_shape((n, n))
    if name != "current_density":
        assert meas.mask is not None and meas.mask.shape == meas.data.shape
        assert float((meas.data * (1 - meas.mask)).abs().max()) == 0.0  # no hidden data leaks
    data = res.history["data"]
    assert all(np.isfinite(data))
    assert data[-1] < 0.2 * data[0], (data[0], data[-1])
    up, down = KEY_METRICS[name]
    base = inst.evaluate(_uniform(inst, name, gt, meas), gt)
    assert out.metrics[up] > base[up] + 0.5, (out.metrics, base)
    assert out.metrics[down] < base[down], (out.metrics, base)
    # inverse-crime guard: data generator and inversion operator are different models
    gen = inst.data_generator()
    problem = inst.build_problem(meas)
    assert gen.fidelity_tag != problem.operator.fidelity_tag
    assert gen.operator is not problem.operator
    try:
        from nefi.bench.protocol import check_inverse_crime
    except ImportError:  # pragma: no cover - bench package in flux
        return
    assert check_inverse_crime(gen, problem.operator) == "cross-fidelity"


@pytest.mark.parametrize("name", NAMES)
def test_grid_baseline_runs(name):
    inst, _ = _smoke(name)
    gt, meas = inst.make_measurement(0)
    problem, cur = inst.baselines()["grid"](meas)
    res = nefi.invert(problem, cur.scaled(0.2), device="cpu")
    assert res.fields[PRIMARY[name]].shape == (inst.cfg.n, inst.cfg.n)
    assert np.isfinite(list(inst.evaluate(res, gt).values())).all()


def test_current_density_fourier_baseline_and_data_metrics():
    # the classical k-space inversion keeps wavelengths above 2π z0 / 3 only: at the smoke preset's
    # 4 px standoff its current map is too blurred for a 3 dB margin, so test it at z0 = 2 px
    inst, _ = _smoke("current_density", z0=0.5)
    gt, meas = inst.make_measurement(0)
    assert "bz" in gt and "clean" not in meas.meta
    problem, cur = inst.baselines()["fourier"](meas)
    res = nefi.invert(problem, cur, device="cpu")
    assert torch.allclose(res.fields["g"], inst.fourier_reconstruction(meas), atol=1e-5)
    m = inst.evaluate(res, gt)
    base = inst.evaluate(_uniform(inst, "current_density", gt, meas), gt)
    assert m["j_psnr"] > base["j_psnr"] + 3.0 and m["bz_relative_error"] < 1.0
    # the FFT operator at the true field matches the direct Biot–Savart data to a few %
    with torch.no_grad():
        bz = inst.operator()({"g": gt["g"].double()})
    assert float((bz - gt["bz"].double()).norm() / gt["bz"].norm()) < 0.05


def test_eit_boundary_currents_and_resolution_consistency():
    from nefi.instances.eit import eit_boundary

    inst, _ = _smoke("eit")
    dom = inst.domain()
    for electrodes in (0, 8):
        geo = eit_boundary(dom, 6, 1.0, electrodes, 0.5)
        injected = (geo.faces["current"] * geo.faces["length"][None]).sum(1)
        assert np.abs(injected).max() < 1e-12  # compatibility ∮ j = 0 per pattern
        assert float(geo.rhs.sum(dim=(-2, -1)).abs().max()) < 1e-9
        ring = geo.weights > 0
        assert int(ring.sum()) <= 4 * dom.shape[0] - 4
    # the native model at the true σ reproduces the 2×-finer data far below the inclusion signal
    gt, meas = inst.make_measurement(0)
    op = inst.operator(tol=1e-10, warm_start=False)
    m = meas.mask.double()

    def rms(x):
        return float(torch.sqrt((x**2 * m).sum() / m.sum()))

    with torch.no_grad():
        v = op({"sigma": gt["sigma"].double()})
        v0 = op({"sigma": torch.ones_like(gt["sigma"]).double()})
        vc = op.at_resolution((8, 8))({"sigma": torch.ones(8, 8, dtype=torch.float64)})
    assert rms(v - meas.data.double()) < 0.15 * rms(v - v0)
    assert vc.shape == v.shape  # coarse stages predict the native observation grid
    gap = build("instance", {"type": "eit", "n": 12, "n_patterns": 2, "electrodes": 8})
    _, gmeas = gap.make_measurement(0)
    assert 0 < int(gmeas.mask[0].sum()) < 44
    # display view: difference to the homogeneous body on the observed ring, NaN elsewhere; the
    # inclusion perturbs the boundary voltages far above the noise level
    diff, label = inst.measurement_image(meas)
    ring = meas.mask[0] > 0
    assert diff.shape == gt["sigma"].shape and "patterns" in label
    assert bool(torch.isnan(diff[~ring]).all()) and bool(torch.isfinite(diff[ring]).all())
    _, flat = inst.make_measurement(0, gt={"sigma": torch.full_like(gt["sigma"], 1.0)})
    assert float(diff[ring].mean()) > 5 * float(inst.difference_data(flat)[ring].mean())


def test_darcy_wells_sensors_and_multiscale_observable():
    inst, _ = _smoke("darcy_flow")
    wells = inst.wells()
    assert len(wells) == inst.cfg.n_configs
    assert all(abs(sum(w[2] for w in config)) < 1e-12 for config in wells)  # balanced
    sensors = inst.sensors()
    assert sensors.shape == (inst.cfg.n_configs, 16, 16) and bool((sensors.sum((1, 2)) > 10).all())
    op = inst.operator()
    x = torch.zeros(8, 8)
    assert op.at_resolution((8, 8))({"log_k": x}).shape == (inst.cfg.n_configs, 16, 16)
    gt, meas = inst.make_measurement(0, "channels")
    assert gt["log_k"].shape == (16, 16) and float(gt["log_k"].max()) > 1.0
    with torch.no_grad():
        p = op({"log_k": torch.zeros(16, 16)})
    w = inst.sensors()
    assert float((p * w).sum((1, 2)).abs().max()) < 1e-4  # gauge: zero mean over the sensors
