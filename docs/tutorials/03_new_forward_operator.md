# 3 · A new forward operator

The operator is the physics of your instrument and the one piece nefi cannot provide. This
tutorial writes a small but complete operator — Gaussian optics followed by a sensor with 2×
coarser pixels — makes it work at every curriculum resolution, checks its gradient, and pairs it
with an **inverse-crime-safe** data generator.

> Quick path: if you only need to invert one measurement, wrap any differentiable function with
> `nefi.operators.function.FunctionOperator(forward)` or let `nefi.from_forward(forward, y,
> shape=...)` build the whole problem. Write an `Operator` subclass when you want multiscale
> curricula, benchmarks and reuse.

## The contract

```python
class Operator(nn.Module):
    primary: str = "x"                  # field the homogeneity statement refers to
    homogeneity: float | None = None    # F(c·x) = c^p F(x); enables EnergyScaleCorrection
    def forward(self, fields) -> Tensor                 # {name: (*shape)} -> prediction
    def at_resolution(self, shape) -> Operator          # the same physics for fields on `shape`
    def output_shape(self, shape) -> tuple | None       # prediction shape for fields on `shape`
    def required_fields(self) -> tuple[str, ...]        # default: (primary,)
```

Rules (DESIGN §3.5): the operator is a **pure function** of its inputs and parameters (no state
that changes between calls — the diagnostics differentiate through it many times); it is device and
dtype agnostic; `at_resolution` shares any `nn.Parameter` with the original so nuisance parameters
keep training across stages.

## Writing the operator

```python
from collections.abc import Sequence

import numpy as np
import torch

import nefi
from nefi.bench import DataGenerator, check_inverse_crime
from nefi.operators import Operator
from nefi.operators.conv import fft_convolve, kernel_offsets
from nefi.registry import register
from nefi.utils.tensor import resample


@register("operator", "sensor_blur")
class SensorBlur(Operator):
    """Gaussian optics blur followed by a sensor with ``factor``× coarser pixels."""

    primary = "x"
    homogeneity = 1.0  # linear: F(c·x) = c·F(x)

    def __init__(self, domain, sigma=0.03, factor=2, fidelity_tag="sensorblur-1x-float32"):
        super().__init__()
        self.domain, self.sigma, self.factor = domain, float(sigma), int(factor)
        self.fidelity_tag = fidelity_tag
        self._kernels = {}  # cache per (grid, device, dtype); the kernel does not depend on x

    def kernel(self, device, dtype):
        key = (self.domain.shape, str(device), dtype)
        if key not in self._kernels:
            full = tuple(2 * n - 1 for n in self.domain.shape)  # exact linear-convolution support
            r = kernel_offsets(self.domain.spacing(), full, device=device, dtype=dtype)
            k = torch.exp(-0.5 * (r**2).sum(-1) / self.sigma**2)
            self._kernels[key] = k / k.sum()
        return self._kernels[key]

    def forward(self, fields):
        x = self.get_field(fields)
        y = fft_convolve(x, self.kernel(x.device, x.dtype))  # zero-padded linear convolution
        return resample(y, self.output_shape(x.shape), mode="area")  # sensor integration

    def at_resolution(self, shape: Sequence[int]):
        return SensorBlur(self.domain.at(shape), self.sigma, self.factor, self.fidelity_tag)

    def output_shape(self, shape):
        return tuple(max(1, s // self.factor) for s in shape)
```

Three details carry the multiscale design:

* **Physical units.** The blur width `sigma` is physical, and the kernel is sampled with
  `domain.spacing()` of the *current* grid, so the physics is the same at 32² and 64² (NeTMY App.
  D.3 rebuilds its dipolar kernels per stage the same way; `FFTConvolution` does this automatically when
  given a `kernel_fn(spacing, shape, device, dtype)`).
* **`at_resolution`** returns a new operator on `domain.at(shape)`. A curriculum stage at 32² calls
  it once and reuses the result for every step of that stage.
* **`output_shape`** lets `InverseProblem.measurement_at` resample the observation for coarse
  stages (here a 32² field predicts a 16² sensor image, and the 32² measurement is area-averaged to
  16²). If the measurement cannot be resampled that way (e.g. time frames on a surface), pass
  `InverseProblem(downsample_obs=...)` instead.

```python
domain = nefi.Domain.unit((64, 64))
op = SensorBlur(domain)
op({"x": torch.rand(64, 64)}).shape                           # (32, 32)
op.at_resolution((32, 32))({"x": torch.rand(32, 32)}).shape   # (16, 16)
```

## Check the gradient

Every new operator gets a float64 `gradcheck` on a tiny grid (and, for adjoint implementations, a
comparison against autograd — see `tests/test_heat_solver.py`):

```python
tiny = SensorBlur(nefi.Domain.unit((6, 6)))
x = torch.rand(6, 6, dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(lambda z: tiny({"x": z}), (x,))
```

## An inverse-crime-safe data generator

Simulating the data with the very operator used for the inversion hides every discretization error
and makes results look better than they are (the *inverse crime*, Kaipio & Somersalo 2007). Both
papers avoid it: NeTMY simulates spectra with a direct source-side simulator F3 and inverts with
the FFT-factorized F2 (or F1); NeFTY simulates with an independent explicit finite-volume engine and
inverts with implicit Euler. The cheapest independent model is the same physics on a **finer grid in
float64**, area-averaged to the sensor:

```python
def disks(coords):
    """Analytic scene evaluated at any resolution (physical coords (..., 2))."""
    x, y = coords[..., 0], coords[..., 1]
    a = ((x - 0.35) ** 2 + (y - 0.5) ** 2 < 0.15**2).double()
    b = 0.6 * ((x - 0.7) ** 2 + (y - 0.35) ** 2 < 0.08**2).double()
    return a + b


fine = domain.refine(4)                                   # 256² simulation grid
gen = DataGenerator(
    SensorBlur(fine, fidelity_tag="sensorblur-4x-float64"),
    noise_std=0.01, relative=True, supersample=4, fidelity_tag="sensorblur-4x-float64",
)
rng = np.random.default_rng(0)
gt_fine = {"x": disks(fine.physical_coords(dtype=torch.float64))}
gt = {"x": disks(domain.physical_coords(dtype=torch.float64)).float()}
meas = gen.generate(gt_fine, rng, target_shape=op.output_shape(domain.shape))
meas.shape, meas.meta        # (32, 32), {'fidelity': 'sensorblur-4x-float64', 'supersample': 4}
```

`DataGenerator(operator, noise_std, relative=True, dtype=float64, supersample, fidelity_tag)`
evaluates the clean measurement without gradients, resamples it to `target_shape`, adds Gaussian
noise (relative to the clean maximum when `relative=True`) and records `noise_std` in the
`Measurement`, which enables discrepancy stopping. Subclass it for other noise models (Poisson
counts, noise relative to the dynamic range as in `NVDataGenerator`, …).

The benchmark protocol compares the generator's `fidelity_tag` with the inversion operator's
(`operator.fidelity_tag`, else its class name) and refuses to run when they coincide:

```python
torch.manual_seed(0)
field = nefi.NeuralField(2, nefi.Heads({"x": nefi.Softplus(init_value=0.1)}),
                         hidden=64, depth=3, n_octaves=6)
losses = nefi.LossSet({"data": nefi.MSE(), "tv": nefi.TV("x")}, weights={"tv": 1e-5})
problem = nefi.InverseProblem(domain, field, op, losses, meas, name="sensor_blur")

check_inverse_crime(gen, problem.operator)              # 'cross-fidelity'
try:
    check_inverse_crime("sensorblur-1x-float32", op)    # data tagged like the inversion model
except AssertionError as err:                           # InverseCrimeError
    print(err)                                          # "inverse crime: the data generator ..."
```

Give data and inversion operators *different, descriptive* tags (`"<model>-<grid>-<dtype>"`), and
make them differ in substance: a finer grid, a different integrator, a higher-fidelity physics.

## Invert

```python
cur = nefi.Curriculum.multiscale((64, 64), n_stages=2, steps=(150, 300), lr=3e-3)
result = nefi.invert(problem, cur, device="cpu", seed=0)
print(result.summary())
print(nefi.metrics.psnr(result.fields["x"], gt["x"]))
```

```text
Result: fields={'x': (64, 64)}
  total time 14.4s, steps 450
  stage 0 (stage1, (32, 32)): 150 steps, 2.7s | data=0.00433, tv=1.66
  stage 1 (stage2, (64, 64)): 300 steps, 11.1s | data=0.00161, tv=6.31
15.888299942016602
```

The coarse stage inverted a 16² sensor image with a 32² field, the fine stage the full 32² image
with a 64² field — the field is super-resolved beyond the sensor by the prior, which is exactly
where the [diagnostics](05_diagnostics.md) (singular values, sensitivity) tell you what the data
can and cannot support.

## Checklist for production operators

- [ ] `forward` is pure, differentiable in every consumed field, device/dtype agnostic.
- [ ] `at_resolution` rebuilds grids/kernels from physical units and shares parameters.
- [ ] `output_shape` (or `InverseProblem.downsample_obs`) defined for multiscale curricula.
- [ ] `homogeneity` set when the operator is homogeneous in its primary field (scale correction).
- [ ] `fidelity_tag` set; an independent `DataGenerator` with a different tag.
- [ ] float64 `gradcheck` on a tiny grid; an analytic test case; for time-stepping physics, an
      adjoint or `grad_mode="checkpoint"` (see `nefi.operators.timestepping.TimeStepper`) so memory
      does not grow with the number of steps (NeFTY Tab. 4: 18.6 GB autograd vs 21.9 MB adjoint).
- [ ] Registered with `@register("operator", "name")` so configs can build it.

Next: package operator, scenes, data generator and defaults as an [instance](04_new_instance.md).
