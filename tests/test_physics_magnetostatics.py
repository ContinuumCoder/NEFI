"""Validation of the planar magnetostatic operators (nefi.physics.magnetostatics)."""

import math

import pytest
import torch

from nefi.domain import Domain
from nefi.errors import ConfigError
from nefi.metrics import psnr
from nefi.physics.magnetostatics import (
    MU0_SI,
    BiotSavartOperator,
    CurrentDensityOperator,
    MagnetizationOperator,
    biot_savart_segments,
    biot_savart_sheet,
    current_divergence,
    fourier_inversion,
    magnetic_constant,
    stream_to_current,
    upward_continuation,
)

DT = torch.float64
MU0 = magnetic_constant("um", "mA", "uT")  # 400π μT·μm/mA


def _dom(n=64, h=0.1):
    return Domain.from_spacing((n, n), h, origin=-n * h / 2)


def _loop(dom, radius=1.0, current=1.0, width=0.05, center=(0.0, 0.0)):
    xy = dom.physical_coords(dtype=DT)
    r = torch.sqrt((xy[..., 0] - center[0]) ** 2 + (xy[..., 1] - center[1]) ** 2)
    return current * 0.5 * torch.erfc((r - radius) / (math.sqrt(2) * width))


def _obs(dom, z0):
    xy = dom.physical_coords(dtype=DT).reshape(-1, 2)
    return torch.cat([xy, torch.full((xy.shape[0], 1), z0, dtype=DT)], -1)


def test_magnetic_constant_units():
    assert abs(magnetic_constant() - MU0_SI) < 1e-18
    assert abs(MU0 - 400 * math.pi) < 1e-9
    assert abs(magnetic_constant("mm", "A", "mT") - 4 * math.pi * 1e-7 * 1e6) < 1e-12
    with pytest.raises(ConfigError):
        magnetic_constant("furlong")


def test_fft_current_operator_matches_biot_savart_for_a_loop():
    dom, z0, radius = _dom(), 0.4, 1.0
    g = _loop(dom, radius)
    bz = CurrentDensityOperator(dom, z0, mu0=MU0)({"g": g})
    # exact thin-loop Biot–Savart (1024-gon line integral)
    th = torch.linspace(0, 2 * math.pi, 1025, dtype=DT)
    pts = torch.stack([radius * torch.cos(th), radius * torch.sin(th), torch.zeros_like(th)], -1)
    ref = biot_savart_segments(pts[:-1], pts[1:], 1.0, _obs(dom, z0), mu0=MU0)[:, 2]
    ref = ref.reshape(dom.shape)
    xy = dom.physical_coords(dtype=DT)
    inner = xy.norm(dim=-1) < 2.2  # away from the zero-padding boundary
    rel = float((bz - ref)[inner].norm() / ref[inner].norm())
    assert rel < 0.03, rel
    # on-axis analytic value μ0 I R² / (2 (R² + z0²)^{3/2}) (counter-clockwise ⇒ +z)
    analytic = MU0 * radius**2 / (2 * (radius**2 + z0**2) ** 1.5)
    centre = float(bz[32:34, 32:34].mean())
    assert abs(centre / analytic - 1) < 0.02


def test_fft_matches_direct_sheet_summation_all_components():
    dom, z0 = _dom(48, 0.1), 0.35
    g = _loop(dom, 0.8, 1.5, 0.12, (0.3, -0.2)) - _loop(dom, 0.5, 1.0, 0.12, (-0.9, 0.8))
    b_fft = CurrentDensityOperator(dom, z0, mu0=MU0, components="xyz")({"g": g})
    assert b_fft.shape == (3, 48, 48)
    fine = dom.at((96, 96))  # direct summation on a 2× finer source grid
    b_dir = BiotSavartOperator(dom, z0, mu0=MU0, components="xyz")(
        {"g": _loop(fine, 0.8, 1.5, 0.12, (0.3, -0.2)) - _loop(fine, 0.5, 1.0, 0.12, (-0.9, 0.8))}
    )
    for c in range(3):
        rel = float((b_fft[c] - b_dir[c]).norm() / b_dir[c].norm())
        assert rel < 0.03, (c, rel)


def test_straight_wire_sign_convention_and_hairpin():
    # hairpin: wire along +y at x = -1 and back along -y at x = +1 (clockwise loop, g = -I inside)
    dom, z0, current = Domain.from_spacing((64, 128), 0.1, origin=(-3.2, -6.4)), 0.3, 1.0
    xy = dom.physical_coords(dtype=DT)
    w = 0.06
    inside_x = (
        0.25
        * torch.erfc((xy[..., 0] - 1.0) / (math.sqrt(2) * w))
        * torch.erfc(-(xy[..., 0] + 1.0) / (math.sqrt(2) * w))
    )
    inside_y = (
        0.25
        * torch.erfc((xy[..., 1] - 5.0) / (math.sqrt(2) * w))
        * torch.erfc(-(xy[..., 1] + 5.0) / (math.sqrt(2) * w))
    )
    g = -current * inside_x * inside_y
    bz = CurrentDensityOperator(dom, z0, mu0=MU0)({"g": g})
    corners = torch.tensor(
        [[-1.0, -5.0, 0.0], [-1.0, 5.0, 0.0], [1.0, 5.0, 0.0], [1.0, -5.0, 0.0], [-1.0, -5.0, 0.0]],
        dtype=DT,
    )
    ref = biot_savart_segments(corners[:-1], corners[1:], current, _obs(dom, z0), mu0=MU0)
    ref = ref[:, 2].reshape(dom.shape)
    band = xy[..., 1].abs() < 3.0
    assert float((bz - ref)[band].norm() / ref[band].norm()) < 0.03
    # sign convention: right of the +y wire at x = −1 the infinite-wire law gives
    # B_z = −μ0 I (x − x_w) / (2π((x − x_w)² + z0²)) < 0 (plus the return wire at x = +1)
    x = xy[:, 64, 0]
    i = int(torch.argmin((x - (-1.0 + z0)).abs()))
    xi = float(x[i])
    law = -MU0 * current * (xi + 1.0) / (2 * math.pi * ((xi + 1.0) ** 2 + z0**2))
    law += MU0 * current * (xi - 1.0) / (2 * math.pi * ((xi - 1.0) ** 2 + z0**2))
    assert float(bz[i, 64]) < 0 and abs(float(bz[i, 64]) / law - 1) < 0.05


def test_upward_continuation_composes():
    dom, z0 = _dom(64, 0.1), 0.3
    g = _loop(dom, 1.0)
    sp = dom.spacing()
    b1 = CurrentDensityOperator(dom, z0, mu0=MU0, pad_factor=1)({"g": g})
    b2 = CurrentDensityOperator(dom, z0 + 0.3, mu0=MU0, pad_factor=1)({"g": g})
    up = upward_continuation(b1, 0.3, sp, pad_factor=1)
    assert float((up - b2).abs().max() / b2.abs().max()) < 1e-12  # exact on the periodic grid
    two = upward_continuation(upward_continuation(b1, 0.1, sp, pad_factor=1), 0.2, sp, pad_factor=1)
    assert float((two - b2).abs().max() / b2.abs().max()) < 1e-12
    b1p = CurrentDensityOperator(dom, z0, mu0=MU0)({"g": g})
    b2p = CurrentDensityOperator(dom, z0 + 0.3, mu0=MU0)({"g": g})
    assert float((upward_continuation(b1p, 0.3, sp) - b2p).abs().max() / b2p.abs().max()) < 0.03
    with pytest.raises(ConfigError):
        upward_continuation(b1, -0.1, sp)


@pytest.mark.parametrize("method", ["central", "spectral"])
def test_stream_function_current_is_divergence_free(method):
    torch.manual_seed(0)
    for shape in ((33, 40), (32, 40)):
        g = torch.randn(*shape, dtype=DT)
        jx, jy = stream_to_current(g, (0.1, 0.2), method)
        div = current_divergence(jx, jy, (0.1, 0.2), method)
        assert float(div.abs().max()) < 1e-12 * float(jx.abs().max() + jy.abs().max())


def test_current_input_matches_stream_input_and_projection():
    dom, z0 = _dom(64, 0.1), 0.3
    g = _loop(dom, 1.0, 1.0, 0.12)
    jx, jy = stream_to_current(g, dom.spacing(), "spectral")
    b_g = CurrentDensityOperator(dom, z0, mu0=MU0)({"g": g})
    b_j = CurrentDensityOperator(dom, z0, mu0=MU0, source="current")({"jx": jx, "jy": jy})
    assert float((b_g - b_j).norm() / b_g.norm()) < 1e-3
    xyz = CurrentDensityOperator(dom, z0, mu0=MU0, components="xyz")({"g": g})
    u = torch.tensor([0.3, -0.5, 0.8], dtype=DT)
    u = u / u.norm()
    proj = CurrentDensityOperator(dom, z0, mu0=MU0, nv_axis=u.tolist())({"g": g})
    assert torch.allclose(proj, (u[:, None, None] * xyz).sum(0), atol=1e-9 * float(xyz.abs().max()))


def test_magnetization_operator_thin_film_and_inplane_dipoles():
    dom, z0 = _dom(48, 0.1), 0.3
    g = _loop(dom, 0.9, 1.0, 0.08)
    b_stream = CurrentDensityOperator(dom, z0, mu0=MU0)({"g": g})
    t = 1e-4  # Ampère equivalence: M_z t = g
    b_mag = MagnetizationOperator(dom, z0, t, mu0=MU0)({"mz": g / t})
    assert float((b_mag - b_stream).norm() / b_stream.norm()) < 1e-3
    # in-plane (x) magnetization vs a direct sum of point dipoles
    xy = dom.physical_coords(dtype=DT)
    m = torch.exp(-(xy[..., 0] ** 2 + xy[..., 1] ** 2) / (2 * 0.3**2))
    b = MagnetizationOperator(dom, z0, 1e-3, direction=(1.0, 0.0, 0.0), mu0=MU0)({"mz": m})
    src = xy.reshape(-1, 2)
    moment = (m * 1e-3 * 0.01).reshape(-1)
    dx = src[:, None, 0] - src[None, :, 0]
    dy = src[:, None, 1] - src[None, :, 1]
    r2 = dx**2 + dy**2 + z0**2
    ref = (MU0 / (4 * math.pi) * 3 * dx * z0 / r2**2.5 * moment[None]).sum(-1).reshape(dom.shape)
    inner = xy.norm(dim=-1) < 1.6
    assert float((b - ref)[inner].norm() / ref[inner].norm()) < 0.01


def test_linearity_homogeneity_and_multiscale():
    torch.manual_seed(1)
    dom = _dom(32, 0.2)
    op = CurrentDensityOperator(dom, 0.4, mu0=MU0)
    assert op.homogeneity == 1.0 and op.required_fields() == ("g",)
    g1, g2 = torch.randn(32, 32, dtype=DT), torch.randn(32, 32, dtype=DT)
    lhs = op({"g": 2.0 * g1 - 3.0 * g2})
    rhs = 2.0 * op({"g": g1}) - 3.0 * op({"g": g2})
    assert float((lhs - rhs).abs().max()) < 1e-10 * float(rhs.abs().max())
    op16 = op.at_resolution((16, 16))
    assert op16.domain.spacing() == (0.4, 0.4) and op16.output_shape((16, 16)) == (16, 16)
    assert op16({"g": torch.randn(16, 16, dtype=DT)}).shape == (16, 16)
    mop = MagnetizationOperator(dom, 0.4, 0.05, mu0=MU0, components="xyz")
    assert mop.output_shape((32, 32)) == (3, 32, 32) and mop.homogeneity == 1.0
    assert mop.at_resolution((16, 16))({"mz": torch.randn(16, 16, dtype=DT)}).shape == (3, 16, 16)


def test_fourier_inversion_recovers_smooth_current_pattern():
    dom, z0 = _dom(64, 0.1), 0.4
    xy = dom.physical_coords(dtype=DT)
    x, y = xy[..., 0], xy[..., 1]
    g = torch.exp(-((x - 0.5) ** 2 + y**2) / (2 * 0.5**2)) - torch.exp(
        -((x + 0.7) ** 2 + (y - 0.3) ** 2) / (2 * 0.4**2)
    )
    bz = CurrentDensityOperator(dom, z0, mu0=MU0)({"g": g})
    gen = torch.Generator().manual_seed(0)
    noisy = bz + 0.002 * float(bz.abs().max()) * torch.randn(bz.shape, dtype=DT, generator=gen)
    rec, (jx, jy) = fourier_inversion(noisy, z0, dom.spacing(), mu0=MU0, return_current=True)
    jgx, jgy = stream_to_current(g, dom.spacing())
    assert psnr(torch.stack([jx, jy]), torch.stack([jgx, jgy])) > 20.0
    assert psnr(rec, g) > 20.0
    rec_m = fourier_inversion(
        MagnetizationOperator(dom, z0, 0.05, mu0=MU0)({"mz": g}),
        z0,
        dom.spacing(),
        mu0=MU0,
        source="magnetization",
        thickness=0.05,
        window="butterworth",
    )
    assert psnr(rec_m, g) > 20.0


def test_biot_savart_sheet_chunking_consistency():
    torch.manual_seed(2)
    src = torch.rand(50, 2, dtype=DT)
    jx, jy = torch.randn(50, dtype=DT), torch.randn(50, dtype=DT)
    obs = torch.rand(40, 2, dtype=DT)
    a = biot_savart_sheet(jx, jy, src, obs, 0.2, cell_area=0.01, components="xyz")
    b = biot_savart_sheet(jx, jy, src, obs, 0.2, cell_area=0.01, components="xyz", chunk_elems=7)
    assert a.shape == (3, 40) and torch.allclose(a, b, atol=1e-14)
