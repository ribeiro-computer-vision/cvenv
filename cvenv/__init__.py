"""cvenv — à-la-carte environment installer for CV/ML courses and projects.

Install components individually or in bulk, from a notebook or the CLI:

    import cvenv
    cvenv.setup(["pytorch3d", "mast3r", "sam2"])          # bulk
    cvenv.get_component("sam2").install()                 # one component
    cvenv.get_component("pytorch3d").verify()             # sanity check

    # or from a shell / a Colab `!` cell, before opening the notebook:
    #   cvenv install pytorch3d mast3r sam2
"""

from .platform import PlatformManager
from .doctor import run_doctor
from .components.pytorch3d import (
    read_wheel_metadata,
    wheel_compatibility,
    wheel_sidecar,
)
from .base import (
    Component,
    REGISTRY,
    register,
    get_component,
    list_components,
    setup,
)

__version__ = "0.1.18"

__all__ = [
    "run_doctor",
    "PlatformManager",
    "Component",
    "REGISTRY",
    "register",
    "get_component",
    "list_components",
    "setup",
    "read_wheel_metadata",
    "wheel_compatibility",
    "wheel_sidecar",
    "__version__",
]
