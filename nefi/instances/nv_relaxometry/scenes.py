"""Sparse spin-source scenes for NV relaxometry (NeTMY App. E.1).

Scene classes are ``<count>/<separation>``:

* count — ``few`` (1-3 sources), ``medium`` (4-8), ``many`` (9-15);
* separation — nearest-neighbour distance of every source, in units of the standoff ``z0``:
  ``close`` ∈ [2, 3) z0, ``medium`` ∈ [3, 5) z0, ``far`` ≥ 5 z0.

The paper lists eight classes (``many/medium`` is not among them, App. E.1); all nine
combinations are accepted by :meth:`NVScenes.sample`, ``classes`` holds the paper's eight.

Why these separation bands. At the paper geometry (20 nm pixels, z0 = 20 nm) one pixel equals
one standoff, so a literal "< 1 z0" band would put sources in the same or adjacent pixels, where
3×3 local-maximum peak detection (Hungarian F1, App. E.3) cannot see two peaks and the scene is
not resolvable even in the noiseless ground truth. The bands are therefore shifted to start at
2 z0 (≥ 2 px, the smallest separation with two distinct 3×3 maxima); "close" sits at the matching
radius (2 px ≈ ½ w_psf, App. E.3) where the (P4) merging pathology bites. All bands and counts are
constructor arguments.

Sources are placed analytically in physical coordinates (snapped to native pixel centres so the
native ground truth is exactly single-pixel) and rendered at any requested ``shape``: single-pixel
masses (``source_width = 0``) or tiny Gaussians (``source_width > 0``, physical std). Each source
carries a constant Larmor frequency drawn from the band; the background Larmor value is the band
centre (it is irrelevant to F3, which only reads ω_L on the source support).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from ...bench.base import SceneGenerator
from ...domain import Domain
from ...errors import ConfigError
from ...registry import register

COUNTS: dict[str, tuple[int, int]] = {"few": (1, 3), "medium": (4, 8), "many": (9, 15)}
SEPARATIONS: dict[str, tuple[float, float]] = {
    "close": (2.0, 3.0),
    "medium": (3.0, 5.0),
    "far": (5.0, math.inf),
}
PAPER_CLASSES: tuple[str, ...] = (
    "few/close",
    "few/medium",
    "few/far",
    "medium/close",
    "medium/medium",
    "medium/far",
    "many/close",
    "many/far",
)


def normalize_class(cls: str) -> str:
    """Canonical ``"<count>/<separation>"`` form (accepts ``_``, ``-`` or ``/`` separators)."""
    c = str(cls).strip().lower().replace("_", "/").replace("-", "/")
    return c


@register("scene", "nv_relaxometry")
class NVScenes(SceneGenerator):
    """Ground-truth ``{"rho", "omega_L"}`` scenes with sparse point sources (NeTMY App. E.1).

    Args:
        domain: 2-D field domain (physical units, e.g. nm).
        z0: standoff height in domain units (separation bands are multiples of ``z0``).
        larmor_band: ``(ω_min, ω_max)`` of the per-source Larmor frequencies.
        background_larmor: ω_L off the sources (default: band centre).
        amp_range: uniform range of the source amplitudes.
        counts: override of :data:`COUNTS`.
        separations: override of :data:`SEPARATIONS` (units of ``z0``; ``inf`` allowed).
        margin_z0: minimum distance of a source from the domain boundary, in ``z0``.
        min_sep_px: absolute floor of the pairwise separation, in native pixels.
        source_width: 0 → single-pixel sources; > 0 → Gaussian std in physical units.
        support_frac: Gaussian sources only — pixels where a source exceeds this fraction of its
            amplitude take its Larmor frequency.
        snap: snap source positions to native pixel centres.
        max_tries: rejection-sampling budget per source and per scene.
    """

    classes = PAPER_CLASSES

    def __init__(
        self,
        domain: Domain,
        z0: float = 20.0,
        larmor_band: Sequence[float] = (1.5, 2.5),
        background_larmor: float | None = None,
        amp_range: Sequence[float] = (0.5, 1.0),
        counts: Mapping[str, Sequence[int]] | None = None,
        separations: Mapping[str, Sequence[float]] | None = None,
        margin_z0: float = 3.0,
        min_sep_px: float = 2.0,
        source_width: float = 0.0,
        support_frac: float = 0.1,
        snap: bool = True,
        max_tries: int = 200,
    ) -> None:
        super().__init__(domain)
        if domain.ndim != 2:
            raise ConfigError(f"NVScenes needs a 2-D domain, got {domain.shape}")
        self.z0 = float(z0)
        lo, hi = (float(v) for v in larmor_band)
        if not hi >= lo:
            raise ConfigError(f"larmor_band must satisfy hi >= lo, got {(lo, hi)}")
        self.larmor_band = (lo, hi)
        self.background_larmor = 0.5 * (lo + hi) if background_larmor is None else background_larmor
        self.amp_range = tuple(float(v) for v in amp_range)
        self.counts = {k: (int(v[0]), int(v[1])) for k, v in (counts or COUNTS).items()}
        self.separations = {
            k: (float(v[0]), float(v[1])) for k, v in (separations or SEPARATIONS).items()
        }
        self.margin_z0 = float(margin_z0)
        self.min_sep_px = float(min_sep_px)
        self.source_width = float(source_width)
        self.support_frac = float(support_frac)
        self.snap = bool(snap)
        self.max_tries = int(max_tries)

    # ---- classes ------------------------------------------------------------------------
    def check_class(self, cls: str | None) -> str:
        cls = normalize_class(cls or self.classes[0])
        parts = cls.split("/")
        if len(parts) != 2 or parts[0] not in self.counts or parts[1] not in self.separations:
            raise ConfigError(
                f"unknown scene class {cls!r}; expected '<count>/<separation>' with count in "
                f"{tuple(self.counts)} and separation in {tuple(self.separations)} "
                f"(paper classes: {self.classes})"
            )
        return cls

    # ---- geometry helpers ---------------------------------------------------------------
    def _snap(self, p: np.ndarray) -> np.ndarray:
        if not self.snap:
            return p
        lo = np.array([e[0] for e in self.domain.extent])
        sp = np.array(self.domain.spacing())
        n = np.array(self.domain.shape)
        i = np.clip(np.floor((p - lo) / sp), 0, n - 1)
        return lo + (i + 0.5) * sp

    def _box(self) -> tuple[np.ndarray, np.ndarray]:
        m = self.margin_z0 * self.z0
        lo = np.array([e[0] + m for e in self.domain.extent])
        hi = np.array([e[1] - m for e in self.domain.extent])
        if np.any(hi <= lo):
            raise ConfigError(
                f"margin {self.margin_z0} z0 leaves no room for sources in extent "
                f"{self.domain.extent}"
            )
        return lo, hi

    # ---- sampling -------------------------------------------------------------------------
    def sample_sources(self, rng: np.random.Generator, cls: str | None = None) -> dict:
        """Sample source positions (physical), amplitudes and Larmor frequencies for ``cls``."""
        cls = self.check_class(cls)
        count_key, sep_key = cls.split("/")
        n_lo, n_hi = self.counts[count_key]
        n = int(rng.integers(n_lo, n_hi + 1))
        d_lo, d_hi = self.separations[sep_key]
        d_min = max(d_lo * self.z0, self.min_sep_px * min(self.domain.spacing()))
        d_max = d_hi * self.z0
        box_lo, box_hi = self._box()
        tol = 1e-9 * min(self.domain.spacing())

        def inside(p):
            return bool(np.all(p >= box_lo - tol) and np.all(p <= box_hi + tol))

        for _ in range(self.max_tries):
            pts = [self._snap(rng.uniform(box_lo, box_hi))]
            ok = inside(pts[0])
            while ok and len(pts) < n:
                ok = False
                for _ in range(self.max_tries):
                    if math.isfinite(d_max):
                        anchor = pts[int(rng.integers(len(pts)))]
                        d = rng.uniform(d_min, max(d_min, d_max))
                        ang = rng.uniform(0.0, 2.0 * math.pi)
                        cand = anchor + d * np.array([math.cos(ang), math.sin(ang)])
                    else:
                        cand = rng.uniform(box_lo, box_hi)
                    cand = self._snap(cand)
                    if not inside(cand):
                        continue
                    dist = np.linalg.norm(np.asarray(pts) - cand, axis=1)
                    if dist.min() < d_min - tol:
                        continue
                    if math.isfinite(d_max) and dist.min() >= d_max - tol:
                        continue
                    pts.append(cand)
                    ok = True
                    break
            if ok:
                return {
                    "class": cls,
                    "positions": np.asarray(pts, dtype=np.float64),
                    "amplitudes": rng.uniform(*self.amp_range, size=n),
                    "larmor": rng.uniform(*self.larmor_band, size=n),
                }
        raise ConfigError(
            f"could not place {n} sources for class {cls!r} on a {self.domain.shape} grid "
            f"(min separation {d_min:g}, margin {self.margin_z0} z0); use a larger grid, a "
            "smaller margin or a sparser class"
        )

    def render(self, sources: Mapping, shape: Sequence[int] | None = None) -> dict:
        """Rasterize ``sources`` (from :meth:`sample_sources`) on a grid of ``shape``."""
        dom = self.domain if shape is None else self.domain.at(shape)
        shape = dom.shape
        pos = np.asarray(sources["positions"], dtype=np.float64).reshape(-1, 2)
        amp = np.asarray(sources["amplitudes"], dtype=np.float64).reshape(-1)
        lar = np.asarray(sources["larmor"], dtype=np.float64).reshape(-1)
        rho = torch.zeros(shape, dtype=torch.float64)
        omega = torch.full(shape, float(self.background_larmor), dtype=torch.float64)
        if self.source_width <= 0:
            lo = np.array([e[0] for e in dom.extent])
            sp = np.array(dom.spacing())
            idx = np.clip(np.floor((pos - lo) / sp), 0, np.array(shape) - 1).astype(int)
            best = {}
            for (i, j), a, w in zip(map(tuple, idx), amp, lar):
                rho[i, j] += float(a)
                if a > best.get((i, j), -1.0):
                    best[(i, j)] = a
                    omega[i, j] = float(w)
        else:
            xy = dom.physical_coords(dtype=torch.float64)
            strongest = torch.zeros(shape, dtype=torch.float64)
            for p, a, w in zip(pos, amp, lar):
                d2 = ((xy - torch.as_tensor(p)) ** 2).sum(-1)
                g = float(a) * torch.exp(-0.5 * d2 / self.source_width**2)
                rho += g
                take = (g > self.support_frac * float(a)) & (g > strongest)
                omega[take] = float(w)
                strongest = torch.where(take, g, strongest)
        return {"rho": rho.float(), "omega_L": omega.float()}

    def sample(
        self, rng: np.random.Generator, cls: str | None = None, shape: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        return self.render(self.sample_sources(rng, cls), shape)


__all__ = ["COUNTS", "PAPER_CLASSES", "SEPARATIONS", "NVScenes", "normalize_class"]
