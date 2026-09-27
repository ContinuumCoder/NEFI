"""Validation of diffraction tomography (nefi.physics.scattering) and optics (physics.optics)."""

import math

import numpy as np
import pytest
import torch
from scipy.special import hankel1

from nefi.domain import Domain
from nefi.metrics import psnr
from nefi.physics.optics import (
    HolographyOperator,
    gaussian_beam,
    gerchberg_saxton,
    propagate,
    rayleigh_range,
)
from nefi.physics.scattering import (
    BornOperator,
    LippmannSchwingerOperator,
    filtered_backpropagation,
    green_2d,
    green_convolve,
    helmholtz_kernel_fft,
)

F64 = torch.float64
K0 = 2 * math.pi  # wavelength 1


def _blobs(x: torch.Tensor, amp: float) -> torch.Tensor:
    chi = torch.zeros(x.shape[:-1], dtype=x.dtype)
    for cx, cy, s, a in ((-0.5, 0.3, 0.4, 1.0), (0.6, -0.4, 0.3, 0.7), (0.2, 0.8, 0.25, -0.5)):
        chi = chi + a * torch.exp(-((x[..., 0] - cx) ** 2 + (x[..., 1] - cy) ** 2) / (2 * s**2))
    return amp * chi


# ---------------------------------------------------------------------------------------------
# scattering
# ---------------------------------------------------------------------------------------------
def test_green_function_torch_matches_scipy():
    r = torch.linspace(0.05, 10.0, 200, dtype=F64)
    ref = 0.25j * hankel1(0, K0 * r.numpy())
    rel = np.abs(green_2d(r, K0).numpy() - ref) / np.abs(ref)
    assert rel.max() < 1e-5  # torch.special Bessel accuracy (operators use scipy precomputation)


@pytest.mark.parametrize("n", [64, 96])
def test_fft_green_convolution_of_point_scatterer_matches_hankel(n):
    """Truncated-kernel FFT convolution reproduces (i/4) H0(k0 r) h² beyond two wavelengths."""
    D = 8.0
    h = D / n
    dom = Domain((n, n), ((-D / 2, D / 2), (-D / 2, D / 2)))
    kernel = helmholtz_kernel_fft((n, n), (h, h), K0, "truncated", dtype=F64)
    for i0, j0 in ((n // 2, n // 2), (3, 5)):
        q = torch.zeros(n, n, dtype=torch.complex128)
        q[i0, j0] = 1.0
        u = green_convolve(q, kernel)
        x = dom.physical_coords(dtype=F64)
        r = (x - x[i0, j0]).norm(dim=-1)
        ref = 0.25j * hankel1(0, K0 * r.numpy()) * h * h
        far = (r > 2.0).numpy()
        err = np.abs(u.numpy()[far] - ref[far]).max() / np.abs(ref[far]).max()
        assert err < 0.03, err
    # the naive regularized k-space kernel is not accurate (documented design choice)
    naive = helmholtz_kernel_fft((n, n), (h, h), K0, "regularized", dtype=F64)
    q = torch.zeros(n, n, dtype=torch.complex128)
    q[n // 2, n // 2] = 1.0
    u = green_convolve(q, naive)
    r = (dom.physical_coords(dtype=F64) - dom.physical_coords(dtype=F64)[n // 2, n // 2]).norm(
        dim=-1
    )
    ref = 0.25j * hankel1(0, K0 * r.numpy()) * h * h
    far = (r > 2.0).numpy()
    assert np.abs(u.numpy()[far] - ref[far]).max() / np.abs(ref[far]).max() > 0.1


def test_born_operator_point_scatterer_matches_analytic_green():
    n, D = 32, 4.0
    dom = Domain((n, n), ((-D / 2, D / 2), (-D / 2, D / 2)))
    op = BornOperator(dom, 1.0, n_angles=4, n_receivers=16, receiver_radius=3.0)
    chi = torch.zeros(n, n, dtype=F64)
    chi[20, 11] = 1e-3
    y = op({"chi": chi})
    assert y.shape == (2, 4, 16) == op.output_shape((n, n))
    xp = dom.physical_coords(dtype=F64)[20, 11]
    h2 = (D / n) ** 2
    rec, ang = op.receivers, op.angles
    u_inc = np.exp(
        1j * K0 * (np.cos(ang.numpy()) * float(xp[0]) + np.sin(ang.numpy()) * float(xp[1]))
    )
    dist = (rec - xp).norm(dim=-1).numpy()
    ref = (0.25j * hankel1(0, K0 * dist))[None, :] * u_inc[:, None] * K0**2 * 1e-3 * h2
    got = y[0].numpy() + 1j * y[1].numpy()
    assert np.allclose(got, ref, rtol=1e-8, atol=1e-14)


def test_born_linearity_precision_and_resolution():
    torch.manual_seed(0)
    n, D = 32, 4.0
    dom = Domain((n, n), ((-D / 2, D / 2), (-D / 2, D / 2)))
    op = BornOperator(dom, 1.0, n_angles=8, n_receivers=32)
    a, b = torch.randn(n, n, dtype=F64), torch.randn(n, n, dtype=F64)
    lhs = op({"chi": 2 * a - 3 * b})
    assert float((lhs - (2 * op({"chi": a}) - 3 * op({"chi": b}))).norm() / lhs.norm()) < 1e-12
    assert op.homogeneity == 1.0
    chi = _blobs(dom.physical_coords(dtype=F64), 0.02)
    y64 = op({"chi": chi})
    y32 = op({"chi": chi.float()})
    assert y32.dtype == torch.float32 and float((y32.double() - y64).norm() / y64.norm()) < 1e-5
    coarse = op.at_resolution((16, 16))
    yc = coarse({"chi": _blobs(dom.at((16, 16)).physical_coords(dtype=F64), 0.02)})
    assert yc.shape == y64.shape and float((yc - y64).norm() / y64.norm()) < 0.01
    rytov = BornOperator(dom, 1.0, n_angles=8, n_receivers=32, rytov=True)
    assert rytov.fidelity_tag == "rytov-1x" and rytov({"chi": chi}).shape == y64.shape


def test_lippmann_schwinger_generator_and_filtered_backpropagation():
    n, D = 32, 4.0
    dom = Domain((n, n), ((-D / 2, D / 2), (-D / 2, D / 2)))
    fine = dom.refine(2)
    born = BornOperator(dom, 1.0, n_angles=16, n_receivers=64)
    rel = []
    for amp in (0.01, 0.02):
        ls = LippmannSchwingerOperator(fine, 1.0, n_angles=16, n_receivers=64)
        yl = ls({"chi": _blobs(fine.physical_coords(dtype=F64), amp)})
        assert ls.last_info["converged"] and ls.last_info["iterations"] < 50
        yb = born({"chi": _blobs(dom.physical_coords(dtype=F64), amp)})
        rel.append(float((yb - yl).norm() / yl.norm()))
    # multiple scattering is a genuine model gap that grows ~linearly with the contrast
    assert 1e-3 < rel[0] < 0.1 and 1.5 < rel[1] / rel[0] < 2.5, rel
    chi = _blobs(dom.physical_coords(dtype=F64), 0.02)
    rec = filtered_backpropagation(yl, born)
    assert rec.shape == (n, n)
    assert psnr(rec.double(), chi) > 15.0
    # Rytov data path
    ls_r = LippmannSchwingerOperator(fine, 1.0, n_angles=16, n_receivers=64, rytov=True)
    y_r = ls_r({"chi": _blobs(fine.physical_coords(dtype=F64), 0.02)})
    born_r = BornOperator(dom, 1.0, n_angles=16, n_receivers=64, rytov=True)
    assert psnr(filtered_backpropagation(y_r, born_r).double(), chi) > 15.0


# ---------------------------------------------------------------------------------------------
# optics
# ---------------------------------------------------------------------------------------------
def test_angular_spectrum_round_trip_is_unitary():
    torch.manual_seed(0)
    n = 64
    dom = Domain.from_spacing((n, n), 1.0, origin=-n / 2)
    u = gaussian_beam(dom, 6.0)
    for pad, bl in ((1.0, False), (2.0, True)):
        v = propagate(u, 150.0, 0.5, dom.spacing(), pad_factor=pad, band_limit=bl)
        w = propagate(v, -150.0, 0.5, dom.spacing(), pad_factor=pad, band_limit=bl)
        assert float((w - u).abs().norm() / u.abs().norm()) < 1e-8
        assert abs(float(v.abs().pow(2).sum() / u.abs().pow(2).sum()) - 1.0) < 1e-8
    r = torch.randn(n, n, dtype=F64) + 1j * torch.randn(n, n, dtype=F64)
    v = propagate(r, 40.0, 0.5, dom.spacing(), pad_factor=1.0, band_limit=False)
    w = propagate(v, -40.0, 0.5, dom.spacing(), pad_factor=1.0, band_limit=False)
    assert float((w - r).abs().norm() / r.abs().norm()) < 1e-12  # all waves propagate (Δ > λ/√2)
    assert abs(float(v.abs().pow(2).sum() / r.abs().pow(2).sum()) - 1.0) < 1e-12


def test_gaussian_beam_width_follows_rayleigh_formula():
    n, lam, w0 = 256, 0.5, 6.0
    dom = Domain.from_spacing((n, n), 0.5, origin=-n / 4)
    zr = rayleigh_range(w0, lam)
    u = gaussian_beam(dom, w0)
    x = dom.physical_coords(dtype=F64)
    for z in (0.5 * zr, zr, 2 * zr):
        intensity = propagate(u, z, lam, dom.spacing(), pad_factor=2.0).abs() ** 2
        w = 2 * math.sqrt(float((intensity * x[..., 0] ** 2).sum() / intensity.sum()))
        w_th = w0 * math.sqrt(1 + (z / zr) ** 2)
        assert abs(w - w_th) / w_th < 0.02, (z / zr, w, w_th)


def test_holography_operator_phase_offset_invariance_and_gerchberg_saxton():
    n = 32
    dom = Domain.from_spacing((n, n), 1.0)
    x = dom.physical_coords(dtype=F64)
    phi = 1.2 * torch.exp(-((x - 16) ** 2).sum(-1) / (2 * 4**2))
    phi = phi - 0.6 * torch.exp(-((x - torch.tensor([8.0, 22.0], dtype=F64)) ** 2).sum(-1) / 18)
    op = HolographyOperator(dom, [10.0, 25.0, 50.0], 0.5)
    intens = op({"phase": phi})
    assert intens.shape == (3, n, n) == op.output_shape((n, n)) and op.homogeneity is None
    assert float((op({"phase": phi + 0.7}) - intens).abs().max()) < 1e-12  # global phase
    assert float(intens.std()) > 0.01  # phase contrast is visible
    coarse = op.at_resolution((16, 16))
    assert coarse({"phase": torch.zeros(16, 16, dtype=F64)}).shape == (3, 16, 16)
    phase, hist = gerchberg_saxton(intens, op, n_iter=60)
    assert hist[-1] < 0.3 * hist[0], hist[::10]
    err = (phase - phase.mean()) - (phi - phi.mean())
    assert float(err.pow(2).mean().sqrt()) < 0.5 * float((phi - phi.mean()).pow(2).mean().sqrt())
    # absorption as a second unknown field
    op2 = HolographyOperator(dom, [10.0, 25.0], 0.5, absorption="absorption")
    assert op2.required_fields() == ("phase", "absorption")
    i2 = op2({"phase": phi, "absorption": torch.full_like(phi, 0.1)})
    assert torch.allclose(i2, math.exp(-0.2) * op({"phase": phi})[:2], rtol=1e-10)
