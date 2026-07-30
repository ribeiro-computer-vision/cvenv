"""Importing this package registers all built-in components.

Import order controls the order in which same-priority components appear; actual
install order is resolved from each component's ``requires`` in ``base.setup``.
"""

from . import science  # noqa: F401
from . import opengl  # noqa: F401
from . import pytorch3d  # noqa: F401
from . import mast3r  # noqa: F401
from . import sam2  # noqa: F401
