"""Smoke tests of the wave / optics / reaction–diffusion instances (tiny budgets, CPU)."""

import math

import pytest
import torch

import nefi
from nefi.instances.diffraction_tomography import DiffractionTomography
from nefi.instances.holography import Holography
from nefi.instances.reaction_diffusion import ReactionDiffusion
from nefi.instances.wave_fwi import WaveFWI, lowpass
from nefi.metrics import evaluate
from nefi.registry import list_registered

NAMES = ("wave_fwi", "diffraction_tomography", "holography", "reaction_diffusion")


def _data_history(result) -> list[float]:
    h = result.history
    return h["data_loss"] if "data_loss" in h else h["data"]


def _check_common(inst, out, field: str, init_value: float):
    """Shapes, decreasing data loss, better-than-initial metrics, inverse-crime tags."""
    res, gt, meas = out.result, out.gt, out.measurement
    n = inst.cfg.n
    assert res.fields[field].shape == (n, n)
    assert tuple(res.pred.shape) == tuple(meas.data.shape)
    assert tuple(meas.data.shape) == tuple(inst.build_problem_measurement_shape((n, n)))
    data = _data_history(res)
    assert data[-1] < 0.5 * data[0], (data[0], data[-1])
    init = evaluate(torch.full((n, n), init_value), gt[field], inst.metrics())
    assert out.metrics["psnr"] > init["psnr"], (out.metrics, init)
    gen = inst.data_generator()
    problem = inst.build_problem(meas)
    assert gen.fidelity_tag != problem.operator.fidelity_tag
    assert meas.meta["fidelity"] == gen.fidelity_tag
    return init


def test_instances_registered():
    names = list_registered("instance")["instance"]
    for name in NAMES:
        assert name in names
        assert name in nefi.instances.available


def test_lowpass_filter_and_band_limited_loss():
    dt = 0.004
    t = torch.arange(500, dtype=torch.float64) * dt
    low, high = torch.sin(2 * math.pi * 2.0 * t), torch.sin(2 * math.pi * 20.0 * t)
    y = lowpass(low + high, dt, f_max=6.0)
    core = slice(100, 400)  # away from the record ends
    assert float((y[core] - low[core]).abs().max()) < 0.05


def test_wave_fwi_smoke():
    inst = WaveFWI(steps=(20, 30))
    out = inst.run(seed=0, device="cpu")
    init = _check_common(inst, out, "c", inst.cfg.c_background)
    assert out.metrics["anomaly_error"] < init["anomaly_error"] == pytest.approx(1.0)
    # near-offset mute (surround smoke preset): short source-receiver pairs are masked and zeroed
    meas, keep = out.measurement, inst.offset_mask()
    src, rec = inst.acquisition()
    assert keep.shape == (inst.cfg.n_sources, inst.cfg.n_receivers, 1)
    assert torch.equal(keep[..., 0] > 0, torch.cdist(src, rec) >= inst.cfg.min_offset)
    assert meas.mask.shape == meas.data.shape and 0 < float(meas.mask.mean()) < 1
    assert float((meas.data * (1 - meas.mask)).abs().max()) == 0.0
    assert WaveFWI(min_offset=0.0).offset_mask() is None
    assert out.result.stage_results[0]["shape"] == (12, 12)
    # frequency continuation: stage 1 fits the low band only, stage 2 the full band
    h = out.result.history
    assert h["fit_lo"][0] > 0 and len(h["fit"]) >= 30
    assert set(inst.baselines()) == {"grid"}
    prob, cur = inst.baselines()["grid"](out.measurement)
    assert isinstance(prob.field, nefi.GridField) and cur.total_steps == 50


def test_diffraction_tomography_smoke_and_backpropagation():
    inst = DiffractionTomography(steps=(60, 90))
    out = inst.run(seed=0, device="cpu")
    _check_common(inst, out, "chi", 0.0)
    assert out.metrics["relative_error"] < 0.6
    assert set(inst.baselines()) == {"grid", "backpropagation"}
    fbp = inst.backpropagation(out.measurement)
    assert evaluate(fbp, out.gt["chi"], inst.metrics())["psnr"] > 20.0
    prob, cur = inst.baselines()["backpropagation"](out.measurement)
    assert prob.meta["solver"] == "direct" and callable(prob.meta["reconstruct"])
    res = nefi.invert(prob, cur, device="cpu")  # zero-lr single step keeps the reconstruction
    assert torch.allclose(res.fields["chi"], prob.field(prob.domain.coords())["chi"].detach())


def test_holography_smoke_and_gerchberg_saxton():
    inst = Holography(steps=(60, 90))
    out = inst.run(seed=0, device="cpu")
    _check_common(inst, out, "phase", 0.0)
    # metrics are invariant to a global phase offset (mean-subtracted)
    shifted = evaluate(out.result.fields["phase"] + 0.3, out.gt["phase"], inst.metrics())
    assert shifted["psnr"] == pytest.approx(out.metrics["psnr"], rel=1e-4)
    # ... and the offset is fixed in the model: zero-mean head (ZeroMean) and zero-mean phantoms
    assert abs(float(out.result.fields["phase"].mean())) < 1e-5
    assert abs(float(out.gt["phase"].mean())) < 1e-5
    with torch.no_grad():  # intensities do not see the offset
        op, phi = inst.operator(), out.gt["phase"].double()
        assert torch.allclose(op({"phase": phi}), op({"phase": phi + 0.7}), atol=1e-9)
    assert set(inst.baselines()) == {"grid", "gerchberg_saxton"}
    inst_gs = Holography(gs_iterations=40)
    gs = inst_gs.gerchberg_saxton(out.measurement)
    zero = evaluate(torch.zeros_like(gs), out.gt["phase"], inst.metrics())
    assert evaluate(gs, out.gt["phase"], inst.metrics())["psnr"] > zero["psnr"] + 5.0


def test_reaction_diffusion_smoke():
    inst = ReactionDiffusion(steps=(40, 60))
    out = inst.run(seed=0, device="cpu")
    init = _check_common(inst, out, "F", inst.cfg.F_background)
    assert out.metrics["relative_error"] < init["relative_error"]
    assert out.result.stage_results[0]["shape"] == (16, 16)
    assert set(inst.baselines()) == {"grid"}


@pytest.mark.slow
@pytest.mark.parametrize(
    "cls,scene,min_psnr",
    [
        (WaveFWI, "smooth_anomaly", 13.0),
        (WaveFWI, "layered", 9.0),
        (DiffractionTomography, "blobs", 30.0),
        (DiffractionTomography, "cells", 24.0),
        (Holography, "smooth_phase", 30.0),
        (Holography, "cells", 27.0),
        (ReactionDiffusion, "blobs", 27.0),
        (ReactionDiffusion, "stripes", 17.0),
    ],
)
def test_default_budget_quality(cls, scene, min_psnr):
    out = cls(scene=scene).run(seed=0, device="cpu")
    assert out.metrics["psnr"] > min_psnr, out.metrics
