"""NV-relaxometry instance (NeTMY, arXiv 2605.13988): physics, operators, metrics, end-to-end."""

import dataclasses
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import nefi
from nefi.config import config_hash, from_dict, to_dict
from nefi.domain import Domain
from nefi.instances.nv_relaxometry import (
    PAPER_CLASSES,
    DirectDensityLoss,
    NVDirectSimulator,
    NVOperator,
    NVRelaxometry,
    NVRelaxometryConfig,
    NVScenes,
    downsample_spectrum,
    make_problem,
    noise_map,
)
from nefi.instances.nv_relaxometry import run as nv_run
from nefi.instances.nv_relaxometry.physics import (
    MU0,
    DipolarKernels,
    frequency_grid,
    gzz_kernel_fn,
    lorentzian,
    power_kernel_fn,
)
from nefi.losses import Context
from nefi.measurement import Measurement
from nefi.metrics import gmsd, hungarian_f1, peak_match, peak_positions, sliced_wasserstein
from nefi.operators import FFTConvolution
from nefi.registry import build, get
from nefi.solve import EnergyScaleCorrection

Z0, GAMMA = 20.0, 0.5
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _dom(n: int) -> Domain:
    return Domain.from_spacing((n, n), 20.0)


def _f2(n: int, mode: str = "F2") -> NVOperator:
    return NVOperator(
        _dom(n), frequency_grid(1.0, 3.0, 50, torch.float64), Z0, GAMMA, mode
    ).double()


def _f3(n: int) -> NVDirectSimulator:
    return NVDirectSimulator(_dom(n), frequency_grid(1.0, 3.0, 50, torch.float64), Z0, GAMMA)


def _sparse_rho(n: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    rho = torch.zeros(n, n, dtype=torch.float64)
    idx = torch.randint(3, n - 3, (k, 2), generator=g)
    rho[idx[:, 0], idx[:, 1]] = 0.5 + torch.rand(k, generator=g, dtype=torch.float64)
    return rho


# ---------------------------------------------------------------------------------------------
# 1. kernels
# ---------------------------------------------------------------------------------------------
def test_power_kernel_sanity():
    n, c = 63, 31
    K = power_kernel_fn(Z0)((20.0, 20.0), (n, n), None, torch.float64)
    assert (K >= 0).all()
    assert torch.allclose(K, torch.flip(K, (0, 1)))  # symmetric under r -> -r
    assert torch.allclose(K, K.T)  # isotropic for a z-aligned NV
    assert float(K[c, c]) == pytest.approx(1.0) and float(K.max()) == pytest.approx(1.0)
    assert torch.all(torch.diff(K[c, c:]) < 0)  # decays along an axis ...
    assert torch.all(torch.diff(torch.diagonal(K)[c:]) < 0)  # ... and along the diagonal
    # closed form P = (|R|² + 3 z0²)/|R|⁸, normalized by its peak 4/z0⁶; P = Σ_a G_az²
    k = DipolarKernels(Z0)
    off = torch.randn(200, 2, dtype=torch.float64) * 40
    R2 = (off**2).sum(-1) + Z0**2
    assert torch.allclose(k.power(off), (R2 + 3 * Z0**2) / R2**4 * Z0**6 / 4, rtol=1e-10)
    assert torch.allclose((k.channels(off) ** 2).sum(-1), k.power(off))
    # F1 kernel G_zz: peak 1 at the origin, sign change at |r| = √2 z0
    G = gzz_kernel_fn(Z0)((20.0, 20.0), (n, n), None, torch.float64)
    assert float(G[c, c]) == pytest.approx(1.0) and float(G.abs().max()) == pytest.approx(1.0)
    g = k.nv(torch.tensor([[1.3 * Z0, 0.0], [1.5 * Z0, 0.0]], dtype=torch.float64))
    assert g[0] > 0 > g[1]
    # physical (SI) units: same shape, scaled by the squared peak (2 μ0/4π / z0³)²
    kp = DipolarKernels(Z0, units="physical", length_scale=1e-9)
    peak2 = (2 * MU0 / (4 * math.pi) / (Z0 * 1e-9) ** 3) ** 2
    assert kp.peak**2 == pytest.approx(peak2)
    assert torch.allclose(kp.power(off) / peak2, k.power(off), rtol=1e-10)
    # the kernel is resolution-consistent: the peak stays 1 on a coarser grid
    K2 = power_kernel_fn(Z0)((40.0, 40.0), (31, 31), None, torch.float64)
    assert float(K2.max()) == pytest.approx(1.0) and float(K2[15, 16]) < float(K[c, c + 1])


def test_power_kernel_fourier_decay_lemma1():
    """NeTMY Lemma 1: |F[P](k)| <= C (1 + k z0)^m exp(-k z0), monotone decay in k."""
    h, n = Z0 / 4, 257  # sample finely so the DFT approximates the continuous transform
    K = power_kernel_fn(Z0)((h, h), (n, n), None, torch.float64)
    F = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(K))).abs().reshape(-1)
    k1 = torch.fft.fftshift(torch.fft.fftfreq(n, d=h)) * 2 * math.pi
    kr = torch.sqrt(k1[:, None] ** 2 + k1[None, :] ** 2).reshape(-1)
    dk = 2 * math.pi / (n * h)
    bins = torch.round(kr / dk).long()
    sums = torch.zeros(int(bins.max()) + 1, dtype=torch.float64).index_add_(0, bins, F)
    cnt = torch.zeros_like(sums).index_add_(0, bins, torch.ones_like(F))
    nb = int(math.pi / h / dk)  # radial bins inside the Nyquist circle
    prof = (sums / cnt)[:nb]
    kz = torch.arange(nb, dtype=torch.float64) * dk * Z0  # k·z0
    assert torch.all(torch.diff(prof[3:]) < 0)  # monotone beyond the first few modes
    env = prof * torch.exp(kz) / (1 + kz) ** 3  # Eq. (22) with m = 3, C = P̂(0)
    assert float(env.max()) <= float(env[0]) * (1 + 1e-9)
    sel = (kz > 6) & (kz < 12)
    A = torch.stack([kz[sel], torch.ones_like(kz[sel])], 1)
    slope = float(torch.linalg.lstsq(A, torch.log(prof[sel])[:, None]).solution[0, 0])
    assert -1.2 < slope < -0.5, slope  # exponential envelope e^{-k z0} up to algebraic factors
    assert float(prof[-1] / prof[0]) < 1e-3


def test_lorentzian_and_frequency_grid():
    f = frequency_grid(1.0, 3.0, 50)
    assert f.shape == (50,) and float(f[0]) == 1.0 and float(f[-1]) == 3.0
    L = lorentzian(f.view(-1, 1, 1), torch.full((2, 3), 2.0), GAMMA)
    assert L.shape == (50, 2, 3) and float(L.max()) <= 1.0
    assert float(lorentzian(2.5, 2.0, GAMMA)) == pytest.approx(0.5)  # half maximum at ω_L ± γ


# ---------------------------------------------------------------------------------------------
# 2-4. operators
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode,p", [("F2", 1.0), ("F1", 2.0)])
def test_operator_homogeneity(mode, p):
    n = 16
    op = _f2(n, mode)
    assert op.homogeneity == p and op.required_fields() == ("rho", "omega_L")
    g = torch.Generator().manual_seed(0)
    rho = torch.rand(n, n, generator=g, dtype=torch.float64)
    w = 1.5 + torch.rand(n, n, generator=g, dtype=torch.float64)
    s = op({"rho": rho, "omega_L": w})
    assert s.shape == (50, n, n) and op.output_shape((8, 8)) == (50, 8, 8) and (s >= 0).all()
    for c in (0.5, 3.0):
        assert torch.allclose(op({"rho": c * rho, "omega_L": w}), c**p * s, rtol=1e-10, atol=0)
    if mode == "F2":
        f3 = _f3(n)
        s3 = f3({"rho": rho, "omega_L": w})
        assert torch.allclose(f3({"rho": 3 * rho, "omega_L": w}), 3 * s3, rtol=1e-12)


def test_f2_matches_f3_for_constant_larmor():
    """Eq. (9): the FFT factorization is exact for spatially constant ω_L."""
    n = 24
    f3 = _f3(n)
    f2_64 = _f2(n)
    f2_32 = NVOperator(_dom(n), frequency_grid(1.0, 3.0, 50), Z0, GAMMA, "F2")
    w = torch.full((n, n), 2.1, dtype=torch.float64)
    for rho in (_sparse_rho(n, 8), torch.rand(n, n, dtype=torch.float64)):
        s3 = f3({"rho": rho, "omega_L": w})
        assert s3.dtype == torch.float64
        rel64 = float((f2_64({"rho": rho, "omega_L": w}) - s3).norm() / s3.norm())
        s2 = f2_32({"rho": rho.float(), "omega_L": w.float()})
        rel32 = float((s2.double() - s3).norm() / s3.norm())
        assert rel64 < 1e-10, rel64
        assert rel32 < 1e-5, rel32


def test_f2_departs_from_f3_when_larmor_varies_within_kernel():
    """App. A.8: the F2-vs-F3 error grows with χ = Δω_L^(kernel)/γ."""
    n = 24
    f3, f2 = _f3(n), _f2(n)
    rho = torch.rand(n, n, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    xy = _dom(n).physical_coords(dtype=torch.float64)
    errs = []
    for chi in (0.1, 0.3, 1.0):
        amp = 0.5  # half the Larmor band: ω_L stays inside [1.5, 2.5] GHz
        lam = 2 * math.pi * amp * 2 * Z0 / (chi * GAMMA)  # steepest change over 2 z0 = χγ
        w = 2.0 + amp * torch.sin(2 * math.pi * xy[..., 0] / lam) * torch.cos(
            2 * math.pi * xy[..., 1] / lam
        )
        s3 = f3({"rho": rho, "omega_L": w})
        errs.append(float((f2({"rho": rho, "omega_L": w}) - s3).norm() / s3.norm()))
    assert errs[0] < errs[1] < errs[2], errs
    assert errs[0] < 5e-3 and errs[2] > 1e-2 and errs[2] > 3 * errs[1], errs


def test_nonperiodic_convolution_has_no_wraparound():
    n = 16
    rho = torch.zeros(n, n, dtype=torch.float64)
    rho[0, 7] = 1.0  # source on the top boundary row
    w = torch.full((n, n), 2.0, dtype=torch.float64)
    nm = noise_map(_f2(n)({"rho": rho, "omega_L": w}))
    assert float(nm[-1].max()) < 1e-6 * float(nm.max())  # opposite edge sees ~P(15 z0) only
    assert torch.allclose(nm, noise_map(_f3(n)({"rho": rho, "omega_L": w})), rtol=1e-10)
    periodic = FFTConvolution(power_kernel_fn(Z0), _dom(n), field="rho", periodic=True)
    wrapped = periodic({"rho": rho})
    assert float(wrapped[-1, 7]) == pytest.approx(5 / 64, rel=1e-6)  # P(1 z0) = 5/64 wraps


def test_operator_at_resolution_and_measurement_downsampling():
    inst = NVRelaxometry(n=16, scene="few/far", margin_z0=2.0)
    gt, meas = inst.make_measurement(0)
    prob = inst.build_problem(meas)
    op8 = prob.operator.at_resolution((8, 8))
    assert op8.domain.shape == (8, 8) and op8.domain.spacing() == (40.0, 40.0)
    assert prob.operator.at_resolution((16, 16)) is prob.operator
    m8 = prob.measurement_at((8, 8))
    assert m8.shape == (50, 8, 8)
    assert float(m8.data.mean()) == pytest.approx(float(meas.data.mean()), rel=1e-5)  # area avg
    assert m8.noise_std == pytest.approx(meas.noise_std / 2)
    assert downsample_spectrum(meas, (50, 8, 8)).shape == (50, 8, 8)
    fields, pred = prob.evaluate((8, 8))
    assert pred.shape == (50, 8, 8) and fields["rho"].shape == (8, 8)
    total, comps = prob.loss((8, 8))
    assert total.requires_grad and {"log_mse", "noise_map", "direct_density"} <= set(comps)


# ---------------------------------------------------------------------------------------------
# 5. scale correction
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["F2", "F1"])
def test_energy_scale_correction_is_exact(mode):
    """Prop. 1 / Eq. (21) / Eq. (30): ρ⋆/3 is rescaled to ρ⋆ (square-root rule under F1)."""
    inst = NVRelaxometry(n=16, data_operator=mode, noise_std=0.0, scene="few/far", margin_z0=2.0)
    gt, meas = inst.make_measurement(0)  # noiseless, same operator: the identity is exact
    assert meas.noise_std is None
    prob = inst.build_problem(meas, mode)
    assert prob.operator.homogeneity == (1.0 if mode == "F2" else 2.0)
    est = {"rho": gt["rho"] / 3.0, "omega_L": gt["omega_L"]}
    pred = prob.operator(est)
    out, info = EnergyScaleCorrection(field="rho")(est, pred, prob, (16, 16))
    assert info["scale_factor"] == pytest.approx(3.0, rel=1e-4)
    assert torch.allclose(out["rho"], gt["rho"], atol=1e-5)


# ---------------------------------------------------------------------------------------------
# 6. metrics
# ---------------------------------------------------------------------------------------------
def test_localization_metrics():
    x = torch.zeros(32, 32)
    x[5, 5], x[20, 10], x[25, 28] = 1.0, 0.7, 0.3
    assert peak_positions(x).tolist() == [[5, 5], [20, 10], [25, 28]]
    assert hungarian_f1(x, x) == 1.0 and gmsd(x, x) == 0.0 and sliced_wasserstein(x, x) == 0.0
    y = torch.zeros(32, 32)
    y[10, 20] = 1.0
    assert hungarian_f1(y, x) == 0.0
    assert sliced_wasserstein(y, x) > 0.0 and gmsd(y, x) > 0.0
    assert hungarian_f1(torch.roll(x, 1, 0), x) == 1.0  # within the 2-px matching radius
    assert hungarian_f1(torch.roll(x, 3, 0), x) == 0.0
    assert hungarian_f1(torch.zeros(32, 32), x) == 0.0  # degenerate case
    # scale invariance of the primary metrics
    assert hungarian_f1(5 * x, x) == 1.0 and gmsd(5 * x, x) == pytest.approx(0.0, abs=1e-12)
    # (float32 rounding of 5·x perturbs the normalized masses by ~1e-8; W2 ~ sqrt(Δm)·d)
    assert sliced_wasserstein(5 * x, x) < 1e-3
    assert sliced_wasserstein(5 * x.double(), x.double()) < 1e-6
    # 5 % threshold, plateau de-duplication, counts
    z = x.clone()
    z[12, 12] = 0.04
    assert [12, 12] not in peak_positions(z).tolist()
    p = torch.zeros(8, 8)
    p[3, 3] = p[3, 4] = 1.0
    assert peak_positions(p).tolist() == [[3, 3]]
    m = peak_match(x + y, x)
    assert (m["tp"], m["fp"], m["fn"]) == (3.0, 1.0, 0.0) and m["f1"] == pytest.approx(6 / 7)
    # SWD of two unit masses at distance d (pixel units) = d · E|cos θ| = 2d/π
    a, b = torch.zeros(32, 32), torch.zeros(32, 32)
    a[4, 4], b[4, 14] = 1.0, 1.0
    assert sliced_wasserstein(a, b, n_proj=4000, pixel_units=True) == pytest.approx(
        20 / math.pi, rel=0.03
    )
    assert get("metric", "swd") is sliced_wasserstein and get("metric", "gmsd") is gmsd
    assert get("metric", "hungarian_f1") is hungarian_f1


# ---------------------------------------------------------------------------------------------
# scenes, data generation, losses, configuration
# ---------------------------------------------------------------------------------------------
def test_scene_classes_counts_separations_and_shape():
    dom = _dom(64)
    scenes = NVScenes(dom, z0=Z0)
    assert scenes.classes == PAPER_CLASSES and len(PAPER_CLASSES) == 8
    rng = np.random.default_rng(0)
    bands = {"close": (2.0, 3.0), "medium": (3.0, 5.0), "far": (5.0, math.inf)}
    counts = {"few": (1, 3), "medium": (4, 8), "many": (9, 15)}
    for cls in PAPER_CLASSES:
        c, s = cls.split("/")
        src = scenes.sample_sources(rng, cls)
        pos = src["positions"] / Z0
        assert counts[c][0] <= len(pos) <= counts[c][1]
        if len(pos) > 1:
            d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
            np.fill_diagonal(d, np.inf)
            nn = d.min(1)
            assert np.all(nn >= bands[s][0] - 1e-9) and np.all(nn < bands[s][1])
        f = scenes.render(src)
        assert f["rho"].shape == (64, 64) and int((f["rho"] > 0).sum()) == len(pos)
        on = f["rho"] > 0
        assert torch.all((f["omega_L"][on] >= 1.5) & (f["omega_L"][on] <= 2.5))
        assert torch.all(f["omega_L"][~on] == 2.0)  # background = band centre
    fine = scenes.sample(np.random.default_rng(1), "few_far", shape=(128, 128))
    assert fine["rho"].shape == (128, 128)
    gauss = NVScenes(dom, source_width=10.0).sample(np.random.default_rng(2), "medium/far")
    assert (gauss["rho"] > 0).sum() > 50
    with pytest.raises(nefi.NefiError):
        NVScenes(_dom(16), max_tries=20).sample(np.random.default_rng(0), "many/far")
    with pytest.raises(nefi.NefiError):
        scenes.check_class("lots/near")


def test_data_generator_is_f3_with_relative_noise():
    inst = NVRelaxometry(n=16, scene="few/far", margin_z0=2.0, noise_std=0.02)
    gen = inst.data_generator()
    assert isinstance(gen.operator, NVDirectSimulator) and gen.fidelity_tag == "F3-direct-float64"
    gt, meas = inst.make_measurement(3)
    gt2, meas2 = inst.make_measurement(3)
    assert torch.equal(meas.data, meas2.data) and torch.equal(gt["rho"], gt2["rho"])
    clean = gen.clean(gt)
    assert meas.noise_std == pytest.approx(0.02 * float(clean.max() - clean.min()), rel=1e-6)
    assert meas.meta["fidelity"] == "F3-direct-float64" and meas.meta["z0"] == Z0
    matched = NVRelaxometry(n=16, data_operator="F2").data_generator()
    assert isinstance(matched.operator, NVOperator) and matched.fidelity_tag == "F2-fft"
    prob = inst.build_problem(meas)
    assert prob.operator.fidelity_tag == "F2-fft" != gen.fidelity_tag
    try:  # the benchmark protocol's inverse-crime guard (NeTMY §3.1) sees F3 → F2 as cross-fidelity
        from nefi.bench.protocol import InverseCrimeError, check_inverse_crime
    except ImportError:  # pragma: no cover - protocol module not available
        return
    assert check_inverse_crime(gen, prob.operator) == "cross-fidelity"
    with pytest.raises(InverseCrimeError):
        check_inverse_crime(matched, prob.operator)


def test_direct_density_loss_and_log_eps():
    dom = _dom(8)
    obs = torch.rand(5, 8, 8, dtype=torch.float64) + 0.1
    rho_match = noise_map(obs).sqrt()  # ρ² ∝ N_obs  ->  R_ds = 0
    loss = DirectDensityLoss("rho")
    ctx = Context({"rho": 7.0 * rho_match}, obs, Measurement(obs), dom)
    assert float(loss(ctx)) == pytest.approx(0.0, abs=1e-12)
    ctx2 = Context({"rho": torch.ones(8, 8, dtype=torch.float64)}, obs, Measurement(obs), dom)
    assert float(loss(ctx2)) > 0
    inst = NVRelaxometry(n=16, scene="few/far", margin_z0=2.0)
    _, noisy = inst.make_measurement(0)
    assert 1e-4 < inst.log_eps(noisy) < 1e-1  # noise floor of the max-normalized noise map
    assert inst.log_eps(Measurement(noisy.data)) == 1e-10  # paper value without noise
    assert NVRelaxometry(log_eps=1e-6).log_eps(noisy) == 1e-6


def test_config_registry_problem_and_baselines():
    inst = build("instance", {"type": "nv_relaxometry", "n": 16, "steps": [4, 6], "hidden": 16})
    assert isinstance(inst, NVRelaxometry) and inst.cfg.n == 16
    c2 = from_dict(NVRelaxometryConfig, to_dict(inst.cfg))
    assert to_dict(c2) == to_dict(inst.cfg) and config_hash(c2) == config_hash(inst.cfg)
    d = NVRelaxometryConfig()  # paper defaults (Tab. 5-7)
    assert (d.n, d.spacing, d.z0, d.gamma, d.n_freq) == (64, 20.0, 20.0, 0.5, 50)
    assert (d.hidden, d.depth, d.skip_at, d.n_octaves, d.tau) == (320, 5, 3, 12, 0.3)
    assert (d.w_log_mse, d.w_noise_map, d.w_direct_density, d.w_tv) == (2.0, 0.5, 0.1, 1e-3)
    assert d.w_sparsity + d.w_l1 == pytest.approx(1.1e-2)
    full = NVRelaxometry()
    cur = full.default_curriculum()
    assert [(s.shape, s.steps, s.lr) for s in cur.stages] == [
        ((32, 32), 3000, 1e-3),
        ((64, 64), 7000, 5e-4),
    ]
    assert cur.stages[1].loss_weights == {"log_mse": 0.0, "noise_map": 2.0}
    assert cur.optim.optimizer == "adamw" and cur.optim.weight_decay == 1e-4
    assert full.field().n_parameters() == 444163  # ≈ 4.5e5 (Tab. 5)
    gt, meas = inst.make_measurement(0)
    prob = inst.build_problem(meas)
    assert prob.field.names == ("rho", "omega_L") and prob.operator.mode == "F2"
    assert inst.build_problem(meas, mode="F1").operator.homogeneity == 2.0
    assert isinstance(prob.postprocess[0], EnergyScaleCorrection)
    bl = inst.baselines()
    assert {"grid", "grid_f1", "f1", "f2"} <= set(bl)
    gp, gcur = bl["grid"](meas)
    assert gcur.stages[0].steps == inst.cfg.grid_steps and gcur.optim.optimizer == "adam"
    assert bl["f1"](meas)[0].operator.mode == "F1"
    # Larmor head: masked to the predicted support, band centre outside (0 with larmor_fill=0)
    fields, _ = prob.evaluate()
    rho, w = fields["rho"], fields["omega_L"]
    off = rho <= 0.3 * rho.max()
    assert torch.all(w[off] == 2.0) and torch.all((w[~off] >= 1.5) & (w[~off] <= 2.5))
    lit = NVRelaxometry(n=16, larmor_fill=0.0, hidden=16).build_problem(meas)
    f_lit, _ = lit.evaluate()
    assert torch.all(f_lit["omega_L"][f_lit["rho"] <= 0.3 * f_lit["rho"].max()] == 0.0)
    for path in ("nv_relaxometry_smoke.yaml", "nv_relaxometry_paper.yaml"):
        cfg = nefi.load_config(CONFIGS / path)
        assert isinstance(build("instance", cfg["instance"]), NVRelaxometry)
    paper = build("instance", nefi.load_config(CONFIGS / "nv_relaxometry_paper.yaml")["instance"])
    assert dataclasses.asdict(paper.cfg)["steps"] == [3000, 7000]


# ---------------------------------------------------------------------------------------------
# 8. (P2) center bias of the iter-0 free-density gradient
# ---------------------------------------------------------------------------------------------
def test_iter0_center_bias_signature():
    """App. C.2 / §5.3: on a uniform density the F2 data gradient peaks at the window center."""
    n = 32
    inst = NVRelaxometry(n=n, scene="medium/medium")
    gt, meas = inst.make_measurement(0)
    ratios = {}
    for mode in ("F2", "F1"):
        prob, _ = inst.build_grid_problem(meas, mode)
        ctx = prob.context()
        assert float(ctx.fields["rho"].detach().std()) < 1e-6  # uniform init (GridField)
        prob.losses.terms["log_mse"](ctx).backward()
        g = prob.field.param.grad[..., 0].abs()  # ∝ |∂D/∂ρ| (uniform softplus slope)
        center = g[n // 2 - 1 : n // 2 + 1, n // 2 - 1 : n // 2 + 1].mean()
        ring = torch.cat([g[0], g[-1], g[1:-1, 0], g[1:-1, -1]]).mean()
        ratios[mode] = float(center / ring)
        if mode == "F2":
            i, j = divmod(int(torch.argmax(g)), n)
            assert abs(i - n / 2) <= 2 and abs(j - n / 2) <= 2  # peak at the center, not a source
            assert float(gt["rho"][i, j]) == 0.0
    assert ratios["F2"] > 1.0, ratios
    assert ratios["F2"] > 10 * ratios["F1"], ratios  # the scalar operator has no such bias


# ---------------------------------------------------------------------------------------------
# 7. end-to-end
# ---------------------------------------------------------------------------------------------
def _log_mse_native(problem, fields) -> float:
    pred = problem.operator(fields)
    ctx = Context(fields, pred, problem.measurement, problem.domain)
    return float(problem.losses.terms["log_mse"](ctx))


def test_smoke_end_to_end():
    """24×24 at z0 = 3 px, 3-5 sources 1-1.5 z0 apart whose blobs merge in the noise map,
    40 + 60 steps (configs/nv_relaxometry_smoke.yaml).

    A 100-step budget separates most merged sources but can miss one or add a faint spurious
    peak: Hungarian F1 is 0.75-1.0 over seeds 0-9 (mean 0.86; 1.0 on every seed at the gallery's
    600 steps), so the bar is 0.5.
    """
    cfg = nefi.load_config(CONFIGS / "nv_relaxometry_smoke.yaml")
    inst = build("instance", cfg["instance"])
    assert inst.cfg.z0 == 3 * inst.cfg.spacing  # the standoff blurs sources over ~3 px
    for seed in (0, 1):
        out = inst.run(seed=seed, device="cpu")
        res = out.result
        assert res.fields["rho"].shape == (24, 24) and res.fields["omega_L"].shape == (24, 24)
        assert res.pred.shape == (50, 24, 24) and out.measurement.shape == (50, 24, 24)
        assert [s["shape"] for s in res.stage_results] == [(12, 12), (24, 24)]
        assert len(res.history["total"]) == 100
        assert 3 <= int((out.gt["rho"] > 0).sum()) <= 5
        # the data losses decrease: D over stage 1, R_nm (the stage-2 primary fidelity) over
        # stage 2, and D at the native resolution relative to a uniform density
        d = res.history["log_mse"]
        assert np.mean(d[-5:]) < 0.5 * np.mean(d[:5])
        r = res.history["noise_map"][40:]
        assert np.mean(r[-5:]) < np.mean(r[:5])
        prob = inst.build_problem(out.measurement)
        uniform = {"rho": torch.ones(24, 24), "omega_L": torch.full((24, 24), 2.0)}
        assert _log_mse_native(prob, res.raw_fields) < 0.5 * _log_mse_native(prob, uniform)
        assert res.post_info["scale_factor"] > 0
        keys = {"gmsd", "hungarian_f1", "swd", "mse", "masked_ssim", "larmor_mae"}
        assert keys <= set(out.metrics)
        assert out.metrics["hungarian_f1"] >= 0.5, (seed, out.metrics)


def test_compare_runs_methods_on_one_sample():
    inst = NVRelaxometry(
        n=16,
        scene="few/far",
        margin_z0=2.0,
        hidden=16,
        depth=2,
        n_octaves=3,
        steps=(6, 6),
        grid_steps=20,
    )
    outs = inst.compare(seed=1, methods=("netmy", "grid", "f1"), device="cpu", step_scale=0.5)
    assert set(outs) == {"netmy", "grid", "f1"}
    assert all(o.measurement is outs["netmy"].measurement for o in outs.values())
    assert outs["grid"].result.fields["rho"].shape == (16, 16)
    assert len(outs["grid"].result.history["total"]) == inst.cfg.grid_steps // 2
    assert all(math.isfinite(o.metrics["swd"]) for o in outs.values())


def test_module_level_convenience():
    prob, gt, meas = make_problem(seed=0, scene="few/far", n=16, margin_z0=2.0, hidden=16)
    assert meas.shape == (50, 16, 16) and set(gt) == {"rho", "omega_L"}
    assert prob.name.endswith("F2")
    cfg = {"n": 16, "margin_z0": 2.0, "scene": "few/far", "hidden": 16, "depth": 2, "steps": [3, 3]}
    res, metrics = nv_run(cfg, seed=0, device="cpu")
    assert res.fields["rho"].shape == (16, 16) and "hungarian_f1" in metrics


@pytest.mark.slow
def test_bigger_budget_localizes_sources():
    """32×32, 400 + 800 steps: Hungarian F1 >= 0.8 on average over three scene classes.

    Measured: few/far 0.86, medium/medium 0.93, many/close 0.96 (≈ 25 s idle, 51 s on a loaded
    10-core CPU).
    """
    t0 = time.perf_counter()
    inst = NVRelaxometry(n=32, hidden=128, depth=4, n_octaves=10, steps=(400, 800))
    f1s = []
    for cls in ("few/far", "medium/medium", "many/close"):
        out = inst.run(seed=0, device="cpu", scene_class=cls)
        f1s.append(out.metrics["hungarian_f1"])
    elapsed = time.perf_counter() - t0
    assert np.mean(f1s) >= 0.8 and min(f1s) >= 0.6, f1s
    # the wall-clock budget is only meaningful on a machine that is not oversubscribed
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    if load <= (os.cpu_count() or 1):
        assert elapsed <= 90.0, elapsed
    else:
        warnings.warn(
            f"budget not asserted: load {load:.1f} > {os.cpu_count()} cores ({elapsed:.0f}s)"
        )
