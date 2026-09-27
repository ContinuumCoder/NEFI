"""Strong exemplar instances. Importing this package registers every instance."""

from importlib import import_module

from .base import Instance, RunOutput

_INSTANCE_MODULES = (
    "toy1d",
    "nv_relaxometry",
    "thermal_tomography",
    "deconvolution",
    "sparse_view_ct",
    "poisson_source",
    # physics zoo (elliptic family)
    "eit",
    "darcy_flow",
    "current_density",
    # physics zoo (wave / optics / reaction family)
    "wave_fwi",
    "diffraction_tomography",
    "holography",
    "reaction_diffusion",
    # 3-D exemplars
    "ct3d",
    "deconvolution3d",
    "dot3d",
    "photoacoustic3d",
)

available: list[str] = []
for _m in _INSTANCE_MODULES:
    try:
        import_module(f"{__name__}.{_m}")
        available.append(_m)
    except ModuleNotFoundError as e:  # instance package not present yet
        if _m not in str(e):
            raise

__all__ = ["Instance", "RunOutput", "available"]
