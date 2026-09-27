"""Auto-tuning: detect and fix the common failure modes of a per-measurement inversion.

A demo gallery run can fail in four typical ways, each with its own remedy — and its own limits:

* **too-conservative budget / learning rate** — detected by :func:`probe_convergence` (doubling
  budgets against the noise floor), fixed by :func:`tune_budget` (measured budget, learning rate,
  annealing length, Morozov stop);
* **wrong regularization strength** — a misfit below / above ``τσ``, fixed by
  :func:`tune_regularization` (discrepancy principle; GradNorm shares when σ is unknown);
* **an unconstrained gauge** (offset, scale, sign, user directions) — detected by
  :func:`detect_gauges` (directions invisible to the data), fixed by :func:`repair_gauges` (mean
  anchor, scale correction, sign convention, penalty);
* **insufficient acquisition** — measured by :func:`acquisition_report` (coverage, singular
  values, data count, match report) and *not* fixed: it is physics; the report says what would
  help and which prior fits what the data support.

:func:`autotune` is a held-out (or discrepancy) Sobol search over a small hyper-parameter space,
and :func:`autotune_problem` chains everything (``level="quick" | "standard" | "thorough"``) with
every decision in an :class:`AutotuneReport`; :func:`autotune_instance` does it for a registered
instance, :func:`autotuned_method` makes it a benchmark method, and ``nefi autotune`` is the CLI.

Example::

    import nefi
    from nefi.instances.eit import EIT

    problem, curriculum, report = nefi.autotune.autotune_instance(EIT(n=16), level="quick")
    print(report.to_markdown())
    result = nefi.invert(problem, curriculum)

See ``docs/autotune.md``.
"""

from ._common import Decision, FitOutcome, fit, resolve_sigma, with_sigma
from .acquisition import THRESHOLDS, VERDICTS, AcquisitionReport, acquisition_report
from .convergence import ConvergenceReport, ProbeRecord, probe_convergence, tune_budget
from .gauge import (
    FIXES,
    GAUGE_KINDS,
    DirectionPenalty,
    Gauge,
    GaugeReport,
    LeastSquaresScale,
    MeanAnchor,
    SignConvention,
    border_region,
    detect_gauges,
    repair_gauges,
    strip_gauge_fixes,
)
from .orchestrate import (
    LEVELS,
    AutotuneReport,
    autotune_instance,
    autotune_problem,
    autotuned_method,
    compare_runs,
    instance_factory,
)
from .regularization import RegularizationResult, regularizer_names, tune_regularization
from .search import (
    Dimension,
    Trial,
    TuneReport,
    apply_params,
    as_dimension,
    autotune,
    default_space,
)

__all__ = [
    "FIXES",
    "GAUGE_KINDS",
    "LEVELS",
    "THRESHOLDS",
    "VERDICTS",
    "AcquisitionReport",
    "AutotuneReport",
    "ConvergenceReport",
    "Decision",
    "Dimension",
    "DirectionPenalty",
    "FitOutcome",
    "Gauge",
    "GaugeReport",
    "LeastSquaresScale",
    "MeanAnchor",
    "ProbeRecord",
    "RegularizationResult",
    "SignConvention",
    "Trial",
    "TuneReport",
    "acquisition_report",
    "apply_params",
    "as_dimension",
    "autotune",
    "autotune_instance",
    "autotune_problem",
    "autotuned_method",
    "border_region",
    "compare_runs",
    "default_space",
    "detect_gauges",
    "fit",
    "instance_factory",
    "probe_convergence",
    "regularizer_names",
    "repair_gauges",
    "resolve_sigma",
    "strip_gauge_fixes",
    "tune_budget",
    "tune_regularization",
    "with_sigma",
]
