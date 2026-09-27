"""Non-iterative reference reconstructions (FBP, Wiener, ...) reported as a regular ``Result``.

Classical closed-form inverses are the natural "zeroth baseline" of every imaging instance; the
:class:`DirectSolver` evaluates such a reconstruction once and packages it exactly like an
iterative result (same post-processing, metrics, history keys), so benchmark tables treat all
methods uniformly.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from ..errors import ConfigError
from ..fields.grid import GridField
from ..losses.base import Context
from ..measurement import Measurement
from ..problem import InverseProblem
from ..registry import register
from ..solve.callbacks import Callback
from ..solve.curriculum import Curriculum, Stage
from ..solve.result import Result
from ..utils.device import resolve_device
from ._common import assemble_result, component_key, resolve_shape

Reconstruct = Callable[[Measurement, Any], "torch.Tensor | Mapping[str, torch.Tensor]"]


@register("baseline", "direct")
class DirectSolver:
    """Evaluate a closed-form reconstruction ``reconstruct(measurement, domain)``.

    Same call shape as :class:`~nefi.solve.Solver`: ``DirectSolver(problem, config).run()``.
    ``config`` is the reconstruction callable (default ``problem.meta["reconstruct"]``); it receives
    the measurement at the working resolution and the working :class:`~nefi.domain.Domain` and
    returns the primary field (or a dict of fields). If the problem's field is a
    :class:`~nefi.fields.GridField` it is set to the reconstruction, so ``problem.evaluate()``
    agrees with the result.
    """

    def __init__(
        self,
        problem: InverseProblem,
        config: Reconstruct | None = None,
        *,
        curriculum: Curriculum | None = None,
        device: str | torch.device = "auto",
        dtype: torch.dtype = torch.float32,
        callbacks: Sequence[Callback] = (),
        seed: int | None = 0,
        **_,
    ) -> None:
        fn = config if callable(config) else problem.meta.get("reconstruct")
        if fn is None:
            raise ConfigError("DirectSolver needs a reconstruct callable (problem.meta)")
        self.problem = problem
        self.reconstruct = fn
        self.device = resolve_device(device)
        self.dtype = dtype
        self.callbacks = list(callbacks)
        self.seed = seed
        self.shape = resolve_shape(problem, None, curriculum)
        self.curriculum = Curriculum([Stage("direct", self.shape, 1, 0.0, anneal=False)])
        self.history: dict[str, list[float]] = {}
        self.stage_results: list[dict] = []

    def run(self) -> Result:
        """Evaluate the reconstruction once and package it as a :class:`~nefi.solve.Result`."""
        p = self.problem
        t0 = time.perf_counter()
        p.to(self.device, self.dtype)
        dom = p.domain.at(self.shape)
        obs = p.measurement_at(dom.shape).to(self.device, self.dtype)
        for cb in self.callbacks:
            cb.on_run_start(self)
        with torch.no_grad():
            out = self.reconstruct(obs, dom)
            recon = dict(out) if isinstance(out, Mapping) else {p.field.primary: out}
            recon = {k: torch.as_tensor(v).to(self.device, self.dtype) for k, v in recon.items()}
            if isinstance(p.field, GridField):
                p.field.on_stage_start(self.curriculum.stages[0], dom)
                p.field.set_fields(recon)
            coords = dom.coords(device=self.device, dtype=self.dtype)
            fields = {k: v.detach() for k, v in p.field(coords, 1.0).items()}
            fields.update(recon)
            op = p.operator.at_resolution(dom.shape).to(self.device)
            pred = op(fields)
            ctx = Context(fields, pred, obs, dom, op, p.field, self.curriculum.stages[0], 0, 1.0)
            total, comps = p.losses(ctx)
        seconds = time.perf_counter() - t0
        h: dict[str, list[float]] = {
            "step": [0],
            "global_step": [0],
            "stage": [0],
            "lr": [0.0],
            "progress": [1.0],
            "total": [float(total)],
            "data_loss": [p.losses.data_loss(comps)],
        }
        for k, v in comps.items():
            h[component_key(k)] = [v]
        self.history = h
        info = {
            "name": "direct",
            "shape": tuple(dom.shape),
            "steps": 0,
            "seconds": seconds,
            "final": comps,
            "stop": "closed_form",
        }
        self.stage_results = [info]
        result = assemble_result(
            p,
            fields,
            tuple(dom.shape),
            h,
            self.stage_results,
            t0,
            {"reconstruct": getattr(self.reconstruct, "__name__", repr(self.reconstruct))},
            {
                "method": p.meta.get("baseline", "direct"),
                "device": str(self.device),
                "dtype": str(self.dtype),
                "seed": self.seed,
                "n_parameters": 0,
            },
        )
        for cb in self.callbacks:
            cb.on_run_end(self, result)
        return result


__all__ = ["DirectSolver"]
