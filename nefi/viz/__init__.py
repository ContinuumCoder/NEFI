"""nefi.viz — figures, animations, dashboards and shareable reports for neural-field inversion.

matplotlib is optional and imported lazily inside the plotting functions: ``import nefi`` and
``import nefi.viz`` never need it. Every function returns a ``matplotlib.figure.Figure`` built
without ``pyplot`` (safe on headless GPU servers); save it with :func:`savefig`.

Modules:

* :mod:`~nefi.viz.style` — house style: colorblind-validated palette, per-quantity colormaps
  (density → magma, diffusivity / conductivity → viridis, signed → RdBu_r centred, phase →
  twilight), light / dark themes, figure sizes, :func:`savefig` with a size budget.
* :mod:`~nefi.viz.fields` — 1-D / 2-D / 3-D / complex field viewers and :func:`compare_fields`
  (measurement | GT | reconstruction(s) | signed error, GT overlay or line profile; metrics in
  titles).
* :mod:`~nefi.viz.volume` — 3-D views: depth mosaics, cross-sections through the anomaly
  centroid, projections, :func:`voxel_compare`.
* :mod:`~nefi.viz.isosurface` — threshold rules (the instance's IoU rule), volume-matched and
  Otsu iso-levels, display transforms (smooth / sharpen / edge-preserving), marching-cubes
  meshes, shaded isosurfaces and multi-level contours (:func:`iso_compare`).
* :mod:`~nefi.viz.interactive` — the dependency-free drag-to-rotate 3-D viewer embedded in
  HTML reports (:func:`volume_viewer_html`, :func:`save_volume_viewer`) and the optional
  CDN-backed plotly mode.
* :mod:`~nefi.viz.hints` — display hints (``viz_hints``) and field / measurement transforms.
* :mod:`~nefi.viz.measurement` — spectra, frames, sinograms, traces, k-space, sparse points,
  volumes and :func:`plot_fit` (data vs prediction vs residual with the noise floor).
* :mod:`~nefi.viz.training` — loss history with stages and annealing, stage summaries, the
  multiscale view and snapshot animations.
* :mod:`~nefi.viz.qualitative` — scenes × methods grids, failure modes, baselines panel.
* :mod:`~nefi.viz.performance` — step time, scaling, memory, wall-clock, benchmark CIs.
* :mod:`~nefi.viz.multiphysics` — the multi-physics gallery (one tile per instance, a full-width
  row per 3-D system, ``gallery_3d.png``).
* :mod:`~nefi.viz.report` — self-contained HTML and Markdown reports.

Quick start::

    import matplotlib
    matplotlib.use("Agg")
    import nefi
    from nefi import viz
    from nefi.instances.toy1d import Toy1D

    out = Toy1D().run(seed=0)
    fig = viz.compare_fields(out.gt, {"nefi": out.result}, out.measurement)
    viz.savefig(fig, "runs/toy1d/compare")
    viz.savefig(viz.plot_history(out.result), "runs/toy1d/history")
"""

from __future__ import annotations

from .fields import (
    PANELS,
    anomaly_centroid,
    anomaly_levels,
    compare_fields,
    depth_mosaic,
    detect_kind,
    draw_levels,
    field_metrics,
    image_panels,
    orthoslices,
    plot_complex,
    plot_field,
    projections,
    representative_slice,
    show_image,
    voxel_threshold,
    voxel_view,
)
from .hints import (
    HINT_KEYS,
    INSTANCE_HINTS,
    TRANSFORMS,
    apply_transform,
    field_hint,
    instance_hints,
)
from .interactive import (
    plotly_volume_html,
    save_volume_viewer,
    viewer_payload,
    viewer_size_bytes,
    volume_viewer_html,
)
from .isosurface import (
    draw_isosurfaces,
    draw_level_contours,
    iso_compare,
    iso_levels,
    iso_mesh,
    matched_level,
    otsu_level,
    volume_transform,
    voxel_rule,
)
from .measurement import (
    LAYOUTS,
    detect_layout,
    draw_measurement,
    measurement_view,
    plot_fit,
    plot_frames,
    plot_kspace,
    plot_measurement,
    plot_signal,
    plot_sinogram,
    plot_spectra,
    plot_traces,
    residual_stats,
    stack_axis,
)
from .multiphysics import (
    GalleryEntry,
    GalleryManifest,
    draw_tile,
    draw_volume_block,
    draw_volume_tile,
    gallery_sections,
    instance_figures,
    physics_gallery,
    physics_overview,
    physics_volumes,
    run_instance_smoke,
    tile_data,
)
from .performance import (
    collect_performance,
    collect_scaling,
    load_profiles,
    load_rows,
    loglog_slope,
    plot_bench_table,
    plot_memory,
    plot_scaling,
    plot_speedup,
    plot_step_time,
    plot_wallclock_breakdown,
    profile_problem,
    profile_speedups,
)
from .qualitative import (
    axis_excess,
    baselines_panel,
    failure_modes,
    failure_stats,
    method_grid,
    run_methods,
)
from .report import Section, environment_info, html_report, markdown_report
from .style import (
    DARK,
    LIGHT,
    PALETTE,
    PALETTE_DARK,
    CmapSpec,
    cmap_for,
    figsize,
    is_signed,
    new_figure,
    palette,
    paper_figsize,
    quantity_of,
    savefig,
    set_default_style,
    spec_from_hint,
    theme,
    use_style,
)
from .training import (
    StageSnapshots,
    animate_snapshots,
    history_arrays,
    plot_history,
    plot_multiscale,
    plot_stage_summary,
)
from .volume import (
    Outline,
    compare_volume,
    draw_projection,
    draw_sections,
    draw_slices,
    draw_voxels,
    mosaic_shape,
    mosaic_slices,
    project,
    projection_mode,
    section,
    voxel_compare,
)

__all__ = [
    "DARK",
    "HINT_KEYS",
    "INSTANCE_HINTS",
    "LAYOUTS",
    "LIGHT",
    "PALETTE",
    "PALETTE_DARK",
    "PANELS",
    "TRANSFORMS",
    "CmapSpec",
    "GalleryEntry",
    "GalleryManifest",
    "Outline",
    "Section",
    "StageSnapshots",
    "animate_snapshots",
    "anomaly_centroid",
    "anomaly_levels",
    "apply_transform",
    "axis_excess",
    "baselines_panel",
    "cmap_for",
    "collect_performance",
    "collect_scaling",
    "compare_fields",
    "compare_volume",
    "depth_mosaic",
    "detect_kind",
    "detect_layout",
    "draw_levels",
    "draw_measurement",
    "draw_projection",
    "draw_sections",
    "draw_slices",
    "draw_isosurfaces",
    "draw_level_contours",
    "draw_tile",
    "draw_volume_block",
    "draw_volume_tile",
    "draw_voxels",
    "environment_info",
    "failure_modes",
    "failure_stats",
    "field_hint",
    "field_metrics",
    "figsize",
    "gallery_sections",
    "history_arrays",
    "html_report",
    "image_panels",
    "instance_figures",
    "instance_hints",
    "is_signed",
    "iso_compare",
    "iso_levels",
    "iso_mesh",
    "load_profiles",
    "load_rows",
    "loglog_slope",
    "markdown_report",
    "matched_level",
    "measurement_view",
    "method_grid",
    "mosaic_shape",
    "mosaic_slices",
    "new_figure",
    "orthoslices",
    "otsu_level",
    "palette",
    "paper_figsize",
    "physics_gallery",
    "physics_overview",
    "physics_volumes",
    "plot_bench_table",
    "plot_complex",
    "plot_field",
    "plot_fit",
    "plot_frames",
    "plot_history",
    "plot_kspace",
    "plot_measurement",
    "plot_memory",
    "plot_multiscale",
    "plot_scaling",
    "plot_signal",
    "plot_sinogram",
    "plot_spectra",
    "plot_speedup",
    "plot_stage_summary",
    "plot_step_time",
    "plot_traces",
    "plot_wallclock_breakdown",
    "plotly_volume_html",
    "profile_problem",
    "profile_speedups",
    "project",
    "projection_mode",
    "projections",
    "quantity_of",
    "representative_slice",
    "residual_stats",
    "run_instance_smoke",
    "run_methods",
    "save_volume_viewer",
    "savefig",
    "section",
    "set_default_style",
    "show_image",
    "spec_from_hint",
    "stack_axis",
    "theme",
    "tile_data",
    "use_style",
    "viewer_payload",
    "viewer_size_bytes",
    "volume_transform",
    "volume_viewer_html",
    "voxel_compare",
    "voxel_rule",
    "voxel_threshold",
    "voxel_view",
]
