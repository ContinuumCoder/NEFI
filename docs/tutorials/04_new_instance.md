# 4 · A new instance

An **instance** packages a complete, reproducible inverse problem — config, domain, ground-truth
scenes, an independent data generator, the problem builder, a default curriculum, metrics and
baselines — behind the `nefi.instances.Instance` protocol. Everything else in nefi talks to that
protocol: the CLI (`nefi run/bench/diagnose/ablate/sweep`), the benchmark harness, the
diagnostics, the smoke tests. This tutorial turns the operator of
[tutorial 3](03_new_forward_operator.md) into the instance `sensor_blur`.

## The protocol

| hook | returns | notes |
|---|---|---|
| `Config` (class attribute) | a dataclass | every number (paper settings!) is a field; `Instance(cfg_dict_or_dataclass, **overrides)` validates keys |
| `domain()` | `Domain` | |
| `scene_generator()` | `bench.SceneGenerator` | `classes = (...)` difficulty classes; `sample(rng, cls, shape)` must honour `shape` (supersampled data) |
| `data_generator()` | `bench.DataGenerator` | independent model, distinct `fidelity_tag` |
| `build_problem(measurement)` | `InverseProblem` | set `curriculum=self.default_curriculum()` |
| `default_curriculum()` | `Curriculum` | the paper schedule |
| `build_problem_measurement_shape(field_shape)` | shape | measurement shape at native resolution (used to resample supersampled data) |
| `metrics()` | `{name: fn(pred, gt)}` | or override `evaluate(result, gt)` for multi-field metrics |
| `baselines()` | `{name: measurement -> (problem, curriculum)}` | optional |
| `default_scene_class()` | str | defaults to `cfg.scene` |

`Instance.make_measurement(seed, scene_class)` and `Instance.run(seed=..., device=...)` are
provided by the base class.

## The code

```python
"""sensor_blur — recover an image from a blurred, 2x-downsampled sensor image (tutorial 4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import nefi
from nefi.baselines import baseline_problem
from nefi.bench import DataGenerator, SceneGenerator
from nefi.instances import Instance
from nefi.metrics import psnr, ssim
from nefi.registry import register

from sensor_blur_operator import SensorBlur  # the operator of tutorial 3


@dataclass
class SensorBlurConfig:
    n: int = 64                  # native field grid (n × n) on [0, 1]²
    sigma: float = 0.03          # optics blur (physical units)
    factor: int = 2              # sensor pixels are factor× coarser
    noise_std: float = 0.01      # relative to the clean maximum
    supersample: int = 4         # data simulated on a 4× finer float64 grid
    scene: str = "few"           # scene class: few | many
    hidden: int = 64
    depth: int = 3
    n_octaves: int = 6
    steps: tuple[int, int] = (150, 300)
    lr: float = 3e-3
    tv: float = 1e-5


class DiskScenes(SceneGenerator):
    """Random disks with random amplitudes; analytic, so any resolution can be rendered."""

    classes = ("few", "many")

    def sample(self, rng, cls=None, shape=None):
        cls = self.check_class(cls)
        r = self.domain.physical_coords(shape, dtype=torch.float64)
        x = torch.zeros(r.shape[:-1], dtype=torch.float64)
        for _ in range(int(rng.integers(2, 4) if cls == "few" else rng.integers(6, 10))):
            c = rng.uniform(0.2, 0.8, size=2)
            rad, amp = rng.uniform(0.04, 0.12), rng.uniform(0.4, 1.0)
            x = x + amp * (((r[..., 0] - c[0]) ** 2 + (r[..., 1] - c[1]) ** 2) < rad**2)
        return {"x": x.float()}


@register("instance", "sensor_blur")
class SensorBlurInstance(Instance):
    name = "sensor_blur"
    Config = SensorBlurConfig
    description = "image from a blurred, 2x-downsampled sensor (tutorial 4)"
    smoke_overrides = {"n": 32, "hidden": 32, "depth": 2, "steps": (20, 30)}  # `--smoke`

    def domain(self):
        return nefi.Domain.unit((self.cfg.n, self.cfg.n))

    def operator(self, domain=None):
        c = self.cfg
        return SensorBlur(domain or self.domain(), c.sigma, c.factor)

    def scene_generator(self):
        return DiskScenes(self.domain())

    def data_generator(self):
        c = self.cfg
        fine = self.domain().refine(c.supersample)
        tag = f"sensorblur-{c.supersample}x-float64"
        return DataGenerator(
            SensorBlur(fine, c.sigma, c.factor, fidelity_tag=tag),
            noise_std=c.noise_std, supersample=c.supersample, fidelity_tag=tag,
        )

    def build_problem_measurement_shape(self, field_shape):
        return self.operator().output_shape(field_shape)

    def build_problem(self, measurement):
        c = self.cfg
        field = nefi.NeuralField(
            2, nefi.Heads({"x": nefi.Softplus(init_value=0.1)}),
            hidden=c.hidden, depth=c.depth, n_octaves=c.n_octaves,
        )
        losses = nefi.LossSet({"data": nefi.MSE(), "tv": nefi.TV("x")}, weights={"tv": c.tv})
        return nefi.InverseProblem(
            self.domain(), field, self.operator(), losses, measurement,
            curriculum=self.default_curriculum(), name=self.name,
        )

    def default_curriculum(self):
        c = self.cfg
        return nefi.Curriculum.multiscale((c.n, c.n), n_stages=2, steps=c.steps, lr=c.lr)

    def metrics(self):
        return {"psnr": psnr, "ssim": ssim}

    def baselines(self):
        # free-pixel grid with the same losses (Tikhonov / Grid Opt.), via the baselines package
        return {"grid": lambda m: baseline_problem(self.build_problem(m), "grid")}
```

Notes:

* **Data are generated once per sample by the instance and shared by all methods.** Scene classes
  are difficulty strata (NeTMY: few/medium/many × close/medium/far; NeFTY: homogeneous/layered).
* `nefi.baselines.baseline_problem(problem, kind)` gives any instance the classical family —
  `"grid"`, `"lbfgs"`, `"admm"`, `"gaussian_splat"`, `"deep_decoder"` — by swapping only the
  parameterization / solver while keeping the operator, losses and measurement, so comparisons
  isolate the prior. Non-gradient baselines request their solver through
  `problem.meta["solver"]`; the CLI and benchmark dispatch automatically.
* `smoke_overrides` (a class attribute or method returning config overrides) makes
  `nefi run sensor_blur --smoke` a seconds-long check; alternatively ship
  `configs/sensor_blur_smoke.yaml` or a `PRESETS["smoke"]` dict in the module.

## Registering it

`@register("instance", "sensor_blur")` adds the class to the registry when the module is imported.
Built-in instances live in `nefi/instances/<name>/` and are listed in `_INSTANCE_MODULES` in
`nefi/instances/__init__.py`. Your own instances can stay in your own code; tell the CLI to import
the module in any of three ways:

```bash
NEFI_PLUGINS=sensor_blur_instance nefi list instances     # module name (on PYTHONPATH) or a .py path
nefi run my_config.yaml                                    # with `imports: [sensor_blur_instance.py]`
```

or declare an entry point in your package's `pyproject.toml`:

```toml
[project.entry-points."nefi.plugins"]
sensor_blur = "my_package.sensor_blur_instance"
```

## A config file

```yaml
# configs/sensor_blur.yaml
imports: [sensor_blur_instance.py]   # relative to this file; registers the instance
instance: sensor_blur
config:                              # fields of SensorBlurConfig (unknown keys are an error)
  n: 64
  scene: many
  steps: [150, 300]
run:                                 # defaults for CLI flags
  seed: 0
  methods: [neural, grid]            # used by `nefi bench`
# optional:
# curriculum: {stages: [{shape: [32, 32], steps: 150, lr: 3.0e-3}, ...], restarts: 1}
# solver: {compile: false, dtype: float32}
```

`instance: {type: sensor_blur, n: 64, ...}` (the registry form) is accepted too.

## Using it

```bash
nefi run sensor_blur.yaml --plot
```

```text
nefi run sensor_blur · neural · seed 0 · scene many · device cpu
  stage1 (32, 32): 150 steps, completed, 2.25s
  stage2 (64, 64): 300 steps, completed, 10.37s
  total 14.13s · 10113 parameters
  metrics  psnr 20.47 · ssim 0.3679
  data fit PSNR 27.58 dB · RMSE/σ 4.321
  saved    out_run  (result.pt config.yaml metrics.json history.csv)
```

```bash
nefi bench sensor_blur.yaml --n 8 --seeds 0,1,2          # neural vs grid (run.methods)
nefi diagnose sensor_blur.yaml                            # ill-posedness report
nefi ablate sensor_blur.yaml --variants no_annealing,no_pe,single_stage --n 4
nefi sweep sensor_blur.yaml --axis n_octaves --values 3,6,9 --n 4
nefi run sensor_blur.yaml --set n=128 --set stage.lr=1e-3 --device cuda
```

and from Python:

```python
inst = SensorBlurInstance(n=64, scene="many")
out = inst.run(seed=0, device="cpu")        # RunOutput(result, metrics, gt, measurement)
print(out.metrics)
```

## Checklist

- [ ] Config dataclass with every number (cite the paper table in comments).
- [ ] Scenes honour `shape`; data generator independent (different `fidelity_tag`).
- [ ] `build_problem` sets the default curriculum; metrics appropriate to the science.
- [ ] Baselines through `nefi.baselines.baseline_problem` (or explicit builders).
- [ ] `configs/<name>.yaml` (+ `_paper.yaml`) and a smoke path (`smoke_overrides`,
      `configs/<name>_smoke.yaml` or `PRESETS["smoke"]`).
- [ ] A test running the smoke configuration end to end; `docs/instances/<name>.md`.
