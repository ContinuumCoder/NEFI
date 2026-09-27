<!-- --8<-- [start:guide] -->
# Contributing to nefi

Thanks for helping! nefi is a small library with a strict contract (`docs/DESIGN.md`): every
problem is *field → differentiable operator → losses → curriculum → solver*, physics is a hard
constraint, and every paper number is a config field. This page explains how to add the four
kinds of components people usually need, and the rules every change must follow.

## Setup

```bash
git clone https://github.com/ContinuumCoder/NEFI.git && cd NEFI
pip install -e ".[dev]"      # torch, numpy, scipy, pyyaml, tqdm + pytest, ruff, matplotlib, typer
pre-commit install           # optional: ruff on every commit
make test                    # fast CPU suite (pytest -q -m "not slow")
make lint                    # ruff format --check + ruff check
make smoke                   # every registered instance end-to-end with --smoke
```

Everything in the test suite and the examples must run on a CPU in seconds: keep them tiny
(≤ 32² / 16³ grids, ≤ 300 steps), never download data, and write outputs only to `runs/`
(git-ignored). Paper-scale runs belong on CUDA machines (`docs/tutorials/07_running_on_gpu_servers.md`).

## Adding a forward operator

1. Subclass `nefi.operators.Operator` (or wrap a function with
   `nefi.operators.function.FunctionOperator` for a quick start) and register it:

   ```python
   from nefi.operators import Operator
   from nefi.registry import register

   @register("operator", "my_blur")
   class MyBlur(Operator):
       primary = "x"            # field the homogeneity statement refers to
       homogeneity = 1.0        # F(c·x) = c^p F(x); None if not homogeneous
       fidelity_tag = "my-blur-1x-float32"   # what the inverse-crime guard compares

       def __init__(self, domain, sigma):
           super().__init__()
           self.domain, self.sigma = domain, sigma

       def forward(self, fields):          # pure function of its inputs: no hidden state
           return blur(self.get_field(fields), self.sigma, self.domain.spacing())

       def at_resolution(self, shape):     # multiscale curricula evaluate coarse grids
           return MyBlur(self.domain.at(shape), self.sigma)

       def output_shape(self, shape):
           return tuple(shape)
   ```

2. Rules: differentiable in every field it consumes; `at_resolution` must *share* any
   `nn.Parameter` with the original (nuisance parameters keep training across stages); device and
   dtype agnostic (no hard-coded `cuda`, no float64 unless the physics needs it); memory-heavy
   time stepping should offer an adjoint or checkpointing (see `nefi/operators/pde/`).
3. Write an *independent* data generator (a finer grid, a different discretization or a
   higher-fidelity model) with a different `fidelity_tag` — the benchmark refuses matched
   operators (`docs/tutorials/03_new_forward_operator.md`).
4. Tests: adjoint / gradient check (`torch.autograd.gradcheck` in float64 on a tiny grid), an
   analytic case, `at_resolution` shape consistency, and homogeneity if declared.

## Adding a field (representation)

Subclass `nefi.fields.Field`, implement `raw(coords, progress) -> (*shape, heads.n_in)` and, if
needed, `on_stage_start(stage, domain)` (e.g. resample a grid) and `reset_parameters()` (used by
multi-restart). Register with `@register("field", "name")`. The physical transform of the raw
output belongs in a `Head` (`Softplus`, `Bounded`, `GatedSoftplus`, `SupportMasked`, ...), not in
the field, so every representation can reuse every prior. Check it with the diagnostics:
`nefi.diagnostics.filter_kernel_row` shows the update kernel `G_θ = J_θ J_θᵀ` it induces.

## Adding a loss

Subclass `nefi.losses.DataLoss` for a data-fidelity term (it sets `is_data = True`, which the
solver uses for early stopping, restarts and the `data_loss` history) or `nefi.losses.FieldLoss`
for a regularizer on a named field; implement `forward(ctx) -> scalar` reading `ctx.pred`,
`ctx.obs`, `ctx.fields`, `ctx.domain`, and register with `@register("loss", "name")`. Reduce by **means** (not sums) so weights are resolution
independent; use `ctx.domain.spacing()` for derivatives.

## Adding an instance

An instance is a complete, registered exemplar problem that the CLI, the benchmark protocol and the
tutorials drive through `nefi.instances.Instance`:

```python
@register("instance", "my_problem")
class MyProblem(Instance):
    name = "my_problem"
    Config = MyConfig                  # dataclass: every paper number is a field
    description = "one line for `nefi list`"

    def domain(self): ...
    def scene_generator(self): ...     # bench.SceneGenerator with difficulty `classes`
    def data_generator(self): ...      # bench.DataGenerator with an independent operator
    def build_problem(self, measurement): ...   # -> InverseProblem (set curriculum=...)
    def default_curriculum(self): ...
    def metrics(self): ...             # {name: fn(pred, gt)}
    def baselines(self): ...           # {name: measurement -> (problem, curriculum)}
    def build_problem_measurement_shape(self, field_shape): ...
```

Then: add the module name to `_INSTANCE_MODULES` in `nefi/instances/__init__.py`; ship
`configs/<name>.yaml` (and `configs/<name>_paper.yaml` for paper-scale settings) plus a tiny
`configs/<name>_smoke.yaml` (or a class attribute `smoke_overrides = {...}`) so that
`nefi run <name> --smoke` finishes in seconds on a CPU; add `docs/instances/<name>.md`; and add a
test that runs the smoke configuration. Baselines built with `nefi.baselines.baseline_problem`
automatically get the right solver dispatch (`problem.meta["solver"]`). See
`docs/tutorials/04_new_instance.md`.

## Documentation

The site at <https://continuumcoder.github.io/NEFI/> is built from `docs/` with MkDocs Material;
every page is plain markdown that also reads well on GitHub.

```bash
pip install -r docs/requirements.txt
mkdocs serve                     # live preview at http://127.0.0.1:8000/NEFI/
mkdocs build --strict            # broken links and missing pages fail the build
```

- A new instance gets `docs/instances/<name>.md` with the "At a glance" card at the top (problem,
  unknown, physics, measurement, difficulty classes, baselines, metrics, run command) and an entry
  in the `nav` of `mkdocs.yml`; the [instance catalogue](https://continuumcoder.github.io/NEFI/instances/) lists it.
- Figures come from runs, never from hand-made images: `python examples/gallery.py --budget 0.2
  --out runs/gallery --html` and then `python tools/build_site_assets.py` refresh the gallery
  images and the interactive 3-D viewers under `docs/assets/` (each image stays below 300 kB);
  `python tools/build_brand_assets.py` redraws the logo, favicon, social card and recipe diagram.
- Numbers quoted in the documentation must come from a test, a run or a report in the
  repository, and CUDA timings that were not measured are labelled as estimates.
- `docs/api/` is generated from the docstrings (`make docs`); do not edit it by hand.

## Testing rules

- `python -m pytest -q -m "not slow"` must pass on CPU in a few minutes; each test file < 60 s.
  Anything heavier is `@pytest.mark.slow`.
- Tests are deterministic (seed everything; tolerances justified by the numerics).
- Test the math, not only the shapes: adjoints against autograd, operators against analytic
  solutions, estimators against exact values (see `tests/test_diagnostics.py`).
- No network, no files outside `tmp_path` / `runs/`, nothing > 200 kB in the repository.

## Style

- `ruff format` + `ruff check` (line length 100); type hints everywhere; Google-style docstrings.
- Cite the paper for every paper-derived component in its docstring (e.g. "NeTMY Eq. (5)",
  "NeFTY App. D.3") and keep `docs/paper_mapping.md` current.
- No `print` in library code — use the `nefi` logger (`logging.getLogger("nefi")`); the CLI may
  print.
- Errors: raise `nefi.errors.NefiError` subclasses with actionable messages (say which shapes /
  names / values were expected).
- Every public class is constructible from a plain dict (registry + dataclass configs); public
  API changes go through `docs/DESIGN.md` first.
- Keep the CHANGELOG up to date under "Unreleased".
<!-- --8<-- [end:guide] -->

## Maintainers

The release procedure is in [RELEASING.md](RELEASING.md).
