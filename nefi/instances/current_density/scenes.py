"""Current-density scenes for NV magnetometry: wires, loops and branching current networks.

Every scene is a stream function ``g = Σ_s I_s Φ(−d_s(x)/w_s)`` — a sum of smoothed indicators of
regions ``Ω_s`` (``d_s`` = signed distance to ``∂Ω_s``, ``Φ`` = standard normal CDF). Its current
``K = ∇×(g ẑ)`` is divergence-free by construction and concentrated on the region boundaries with
a Gaussian cross-section of standard deviation ``w_s`` carrying the current ``I_s``
(counter-clockwise for ``I_s > 0``) — i.e. rasterized current paths with Gaussian width:

* ``loops``     — 1–3 elliptical current loops (96-gons);
* ``wires``     — 1–2 hairpin circuits: the region within ``s/2`` of a random polyline, so the
  current runs out along one side of the polyline and back along the other (a go-and-return wire
  pair of separation ``s``);
* ``branching`` — a trunk and 2–3 branches sharing a junction, with different currents: where the
  regions overlap the currents add, so current splits and merges at the junctions.

All sources keep a margin from the field of view so ``g → 0`` at its edges (no hidden return
currents outside the image). Closed-form in position, so any grid resolution renders the same scene.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from ...bench.base import SceneGenerator
from ...domain import Domain
from .._elliptic_common import polygon_signed_distance, polyline_distance

__all__ = ["CurrentScenes"]


def _phi(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.erfc(-x / math.sqrt(2.0))


class CurrentScenes(SceneGenerator):
    """Stream-function scenes (see module docstring). Lengths in domain units.

    Args:
        domain: 2-D field of view.
        current: ``(lo, hi)`` magnitude of each path current (sign random).
        width: ``(lo, hi)`` Gaussian std of the current profile.
        margin: sources stay this fraction of the field of view away from the edges.
        loop_radius: ``(lo, hi)`` loop semi-axes (fraction of the field of view).
        wire_separation: ``(lo, hi)`` go/return separation of hairpin wires (fraction of the FOV).
        branch_width: ``(lo, hi)`` half-width of branching regions (fraction of the FOV).
    """

    classes = ("wires", "loops", "branching")

    def __init__(
        self,
        domain: Domain,
        current: Sequence[float] = (0.5, 1.5),
        width: Sequence[float] = (0.2, 0.3),
        margin: float = 0.15,
        loop_radius: Sequence[float] = (0.08, 0.2),
        wire_separation: Sequence[float] = (0.08, 0.14),
        branch_width: Sequence[float] = (0.04, 0.07),
    ) -> None:
        super().__init__(domain)
        self.current = tuple(map(float, current))
        self.width = tuple(map(float, width))
        self.margin = float(margin)
        self.loop_radius = tuple(map(float, loop_radius))
        self.wire_separation = tuple(map(float, wire_separation))
        self.branch_width = tuple(map(float, branch_width))
        (x0, x1), (y0, y1) = domain.extent
        self.center = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
        self.fov = min(x1 - x0, y1 - y0)
        self.half_box = (0.5 - self.margin) * self.fov  # usable half-width

    # --- random primitives (resolution independent) ------------------------------------------
    def _current(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(*self.current) * (1 if rng.uniform() < 0.5 else -1))

    def _point(self, rng: np.random.Generator, pad: float) -> np.ndarray:
        lim = max(self.half_box - pad, 0.0)
        return self.center + rng.uniform(-lim, lim, 2)

    def _polyline(self, rng: np.random.Generator, n_seg: int, pad: float, start=None) -> np.ndarray:
        """Random walk polyline with segment length ~0.2–0.35 FOV kept inside the box."""
        pts = [self._point(rng, pad) if start is None else np.asarray(start, float)]
        heading = rng.uniform(0, 2 * math.pi)
        lim = self.half_box - pad
        for _ in range(n_seg):
            for _attempt in range(50):
                ang = heading + rng.uniform(-0.6 * math.pi, 0.6 * math.pi)
                step = rng.uniform(0.2, 0.35) * self.fov
                nxt = pts[-1] + step * np.array([math.cos(ang), math.sin(ang)])
                if np.all(np.abs(nxt - self.center) <= lim):
                    heading = ang
                    break
            else:
                nxt = np.clip(nxt, self.center - lim, self.center + lim)
            pts.append(nxt)
        return np.stack(pts)

    def components(self, rng: np.random.Generator, cls: str) -> list[dict]:
        """Sample the scene's current paths: dicts with ``kind``, geometry, ``I`` and ``w``."""
        out: list[dict] = []
        if cls == "loops":
            for _ in range(int(rng.integers(1, 4))):
                a = rng.uniform(*self.loop_radius) * self.fov
                b = rng.uniform(*self.loop_radius) * self.fov
                c = self._point(rng, max(a, b))
                ang = rng.uniform(0, math.pi)
                t = np.linspace(0, 2 * math.pi, 96, endpoint=False)
                u, v = a * np.cos(t), b * np.sin(t)
                poly = np.stack(
                    [
                        c[0] + u * math.cos(ang) - v * math.sin(ang),
                        c[1] + u * math.sin(ang) + v * math.cos(ang),
                    ],
                    -1,
                )
                w = rng.uniform(*self.width)
                out.append({"kind": "polygon", "vertices": poly, "I": self._current(rng), "w": w})
        elif cls == "wires":
            for _ in range(int(rng.integers(1, 3))):
                sep = rng.uniform(*self.wire_separation) * self.fov
                line = self._polyline(rng, int(rng.integers(2, 5)), 0.5 * sep)
                w = rng.uniform(*self.width)
                out.append(
                    {
                        "kind": "tube",
                        "vertices": line,
                        "r": 0.5 * sep,
                        "I": self._current(rng),
                        "w": w,
                    }
                )
        elif cls == "branching":
            half = rng.uniform(*self.branch_width) * self.fov
            trunk = self._polyline(rng, 2, half)
            w = rng.uniform(*self.width)
            out.append(
                {"kind": "tube", "vertices": trunk, "r": half, "I": self._current(rng), "w": w}
            )
            for _ in range(int(rng.integers(2, 4))):
                br = self._polyline(rng, int(rng.integers(1, 3)), half, start=trunk[-1])
                out.append(
                    {"kind": "tube", "vertices": br, "r": half, "I": self._current(rng), "w": w}
                )
        return out

    # --- rendering ----------------------------------------------------------------------------
    def render(self, comps: list[dict], shape: Sequence[int] | None = None) -> torch.Tensor:
        """Stream function ``g`` (float64) on the grid ``shape``."""
        dom = self.domain if shape is None else self.domain.at(shape)
        xy = dom.physical_coords(dtype=torch.float64)
        g = torch.zeros(dom.shape, dtype=torch.float64)
        for c in comps:
            verts = torch.as_tensor(c["vertices"], dtype=torch.float64)
            if c["kind"] == "polygon":
                d = polygon_signed_distance(xy, verts)
            else:
                d = polyline_distance(xy, verts) - c["r"]
            g = g + c["I"] * _phi(-d / c["w"])
        return g

    def sample(self, rng, cls=None, shape=None):
        cls = self.check_class(cls)
        return {"g": self.render(self.components(rng, cls), shape).float()}
