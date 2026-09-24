"""The base scientific-Python stack, pinned so it stays ABI-consistent.

Lesson baked in: on modern Colab / Lightning Studio the whole stack is numpy-2
native, and their preinstalled compiled packages (cv2, scipy, …) are built
against numpy 2.x. Downgrading numpy below 2.0 then breaks them with
"numpy.dtype size changed, Expected 96 ... got 88". ``numpy>=2.0,<2.1`` satisfies
numba (needs <2.1 on older Colab), the numpy-2 packages, and torch/pytorch3d.

That upper bound is unusable on Python 3.13, though: NumPy only gained 3.13
support in 2.1.0, so ``<2.1`` has no cp313 wheel and pip silently falls back to
compiling NumPy from source — minutes of build, or a failure. The pin is
therefore chosen per interpreter.

This component is also the base for pure-numpy/scipy course material (e.g. Kalman
filtering, Lie groups) — those need nothing beyond this stack.
"""

from __future__ import annotations

import sys

from ..base import Component, register

# NumPy 2.0.x has no cp313 wheel — 3.13 support arrived in 2.1.0. Asking for
# <2.1 there triggers a source build, so only apply the upper bound where a
# wheel actually exists.
NUMPY_PIN = "numpy>=2.0,<2.1" if sys.version_info < (3, 13) else "numpy>=2.1"

# (pip requirement, module to import when checking it is there). The two names
# differ often enough — scikit-image/skimage, scikit-learn/sklearn, pillow/PIL,
# opencv-python/cv2 — that checking the wrong one is how a package goes missing
# without anyone noticing.
STACK = [
    (NUMPY_PIN,       "numpy"),
    ("scipy",         "scipy"),
    ("matplotlib",    "matplotlib"),
    ("pandas",        "pandas"),
    ("scikit-image",  "skimage"),
    ("scikit-learn",  "sklearn"),
    ("opencv-python", "cv2"),
    ("pillow",        "PIL"),
    ("tqdm",          "tqdm"),
    ("imageio",       "imageio"),
    ("colorama",      "colorama"),
]
REQUIREMENTS = [req for req, _ in STACK]
MODULES = [mod for _, mod in STACK]


def _missing_modules() -> list[str]:
    """Which of the stack's modules are not importable, without importing them."""
    import importlib.util
    missing = []
    for mod in MODULES:
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        except (ImportError, ValueError):
            missing.append(mod)
    return missing


class Science(Component):
    name = "science"
    summary = "Core scientific-Python stack (numpy 2.0.x pinned, scipy, matplotlib, opencv, …)."
    teaching_note = (
        "numpy pinned to >=2.0,<2.1: modern Colab/Studio ship numpy-2, and their "
        "compiled cv2/scipy are built against it. Installing numpy<2 triggers "
        "'numpy.dtype size changed' ABI errors. If numpy changes in a live kernel, "
        "restart the runtime once — never force-reinstall numpy repeatedly."
    )

    def is_installed(self) -> bool:
        # Every module the component installs must be present, not just the three
        # compiled ones. Colab and Lightning Studio ship numpy, scipy and cv2
        # preinstalled, so checking only those declared the component already
        # installed and skipped the rest — scikit-image, scikit-learn, pandas and
        # the others were never installed, and the failure surfaced much later as
        # "No module named 'skimage'" in a tutorial's import cell.
        return not _missing_modules()

    def _install(self, platform=None, **opts) -> None:
        from .._pip import pip_install
        # numpy first (and alone) so the pin is resolved before the packages that
        # depend on its ABI get (re)built/checked against it.
        pip_install(NUMPY_PIN, check=False)
        pip_install(*REQUIREMENTS[1:], check=False)

    def verify(self) -> bool:
        # Genuinely import every module rather than only locating it: an ABI
        # break ("numpy.dtype size changed") appears on import and nowhere else.
        import importlib
        failed = {}
        for mod in MODULES:
            try:
                importlib.import_module(mod)
            except Exception as exc:
                failed[mod] = exc
        if failed:
            print(f"❌ science: {len(failed)} of {len(MODULES)} modules unusable:")
            for mod, exc in failed.items():
                print(f"      • {mod}: {type(exc).__name__}: {exc}")
            print("   Re-run with force=True / --force to install them:\n"
                  '       cvenv.get_component("science").install(force=True)')
            return False

        import numpy, cv2  # noqa: F401
        # numpy<2 is fine locally; it only bites on numpy-2-native Colab/Studio,
        # so warn rather than fail.
        if int(numpy.__version__.split(".")[0]) < 2:
            print(f"⚠️  science: numpy {numpy.__version__} is <2.0 — fine locally, "
                  "but on Colab/Studio pin >=2.0,<2.1 (else cv2/scipy ABI breaks).")
        print(f"✅ science: all {len(MODULES)} modules import "
              f"(numpy {numpy.__version__}, cv2 {cv2.__version__})")
        return True


register(Science())
