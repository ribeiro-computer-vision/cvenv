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

The mismatch runs both ways. Lightning Studio shipped pandas 2.1.4 and
scikit-learn 1.3.2, both requiring numpy<2, beside the numpy>=2.0 this component
pins — and installing them by bare name changed nothing, because pip counts an
already-installed package as satisfying a bare requirement. Hence the numpy-2
lower bounds in STACK, and hence checking the stack by importing it rather than
merely locating it: a package built against the wrong numpy is present and
importable-looking right up until it raises.

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
#
# The lower bounds are the first releases BUILT against numpy 2, and they are the
# point of this list rather than decoration. `pip install pandas` does not upgrade
# an already-installed pandas — pip treats the requirement as satisfied — so
# pinning numpy>=2.0 on an image that ships a numpy-1-era pandas leaves the two
# mismatched and produces exactly the error this component exists to prevent:
#
#     ValueError: numpy.dtype size changed, may indicate binary incompatibility.
#                 Expected 96 from C header, got 88 from PyObject
#
# Lightning Studio shipped pandas 2.1.4 and scikit-learn 1.3.2, both of which
# require numpy<2; installing them by bare name changed nothing. A lower bound
# makes the installed version *unsatisfying*, so pip actually upgrades it. Only
# packages that embed numpy's C ABI need one.
STACK = [
    (NUMPY_PIN,              "numpy"),
    ("scipy>=1.13",          "scipy"),
    ("matplotlib>=3.9",      "matplotlib"),
    ("pandas>=2.2.2",        "pandas"),
    ("scikit-image>=0.24",   "skimage"),
    ("scikit-learn>=1.5",    "sklearn"),
    ("opencv-python>=4.10",  "cv2"),
    ("pillow",               "PIL"),
    ("tqdm",                 "tqdm"),
    ("imageio",              "imageio"),
    ("colorama",             "colorama"),
]
REQUIREMENTS = [req for req, _ in STACK]
MODULES = [mod for _, mod in STACK]


def _broken_modules() -> "dict[str, Exception]":
    """Stack modules that do not import, mapped to why.

    This genuinely imports rather than calling find_spec. A package whose compiled
    extension disagrees with the installed numpy is *present* — find_spec finds it
    happily — and only raises on import:

        ValueError: numpy.dtype size changed ...

    Locating it therefore reported the stack as fine, install() skipped, and the
    mismatch surfaced later inside whatever first imported pandas.
    """
    import importlib
    broken = {}
    for mod in MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            broken[mod] = exc
    return broken


class Science(Component):
    name = "science"
    summary = "Core scientific-Python stack (numpy 2.0.x pinned, scipy, matplotlib, opencv, …)."
    teaching_note = (
        "numpy pinned to >=2.0,<2.1: modern Colab/Studio ship numpy-2, and their "
        "compiled cv2/scipy are built against it. Installing numpy<2 triggers "
        "'numpy.dtype size changed' ABI errors. The converse bites too: an image "
        "may ship a numpy-1-era pandas or scikit-learn, and `pip install pandas` "
        "will NOT upgrade an already-installed one, so the stack carries explicit "
        "numpy-2 lower bounds. If numpy changes in a live kernel, restart the "
        "runtime once — never force-reinstall numpy repeatedly."
    )

    def is_installed(self) -> bool:
        # Every module the component installs must be present, not just the three
        # compiled ones. Colab and Lightning Studio ship numpy, scipy and cv2
        # preinstalled, so checking only those declared the component already
        # installed and skipped the rest — scikit-image, scikit-learn, pandas and
        # the others were never installed, and the failure surfaced much later as
        # "No module named 'skimage'" in a tutorial's import cell.
        return not _broken_modules()

    def _install(self, platform=None, **opts) -> None:
        from .._pip import pip_install
        # numpy first (and alone) so the pin is resolved before the packages that
        # depend on its ABI get (re)built/checked against it.
        pip_install(NUMPY_PIN, check=False)
        pip_install(*REQUIREMENTS[1:], check=False)

    def verify(self) -> bool:
        # Genuinely import every module rather than only locating it: an ABI
        # break ("numpy.dtype size changed") appears on import and nowhere else.
        failed = _broken_modules()
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
