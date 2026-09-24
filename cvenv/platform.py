"""Compute-platform detection (Colab / RunPod / Lightning AI / WSL / LocalPC).

Genericized from a per-project setup helper into a standalone, reusable class so
components can adapt install behavior to where they run.

WSL2 is reported separately from a plain Linux box: the two install identically
once CUDA is working, but they fail differently, and a student on WSL needs to
know the NVIDIA driver belongs on *Windows* while the CUDA toolkit belongs
*inside* the distro.
"""

from __future__ import annotations

import os
from typing import Tuple


_warned_missing_cwd = False


def _safe_cwd(default: str | None = None) -> str:
    """``os.getcwd()``, tolerant of a working directory that no longer exists.

    A kernel keeps running after its cwd is deleted — a cleaned-up temp build
    dir, a removed clone, a studio restart — and ``os.getcwd()`` then raises
    ``FileNotFoundError``. Platform detection must not be what breaks: it is the
    first thing every notebook calls, so the traceback reads as "cvenv is
    broken" when the real problem is that the directory is gone. Warn once and
    carry on; every caller here only needs a plausible base path.
    """
    global _warned_missing_cwd
    if default is None:
        # Home, not "/": callers derive writable paths from this (a wheel cache,
        # a clone directory), and root is not writable.
        default = os.path.expanduser("~")
    try:
        return os.getcwd()
    except OSError:
        if not _warned_missing_cwd:
            _warned_missing_cwd = True
            print("⚠️  this process's working directory no longer exists, so "
                  "paths\n    derived from it may be wrong. Fix it with "
                  "os.chdir(os.path.expanduser('~'))\n    — or any directory "
                  "that does exist — then re-run.")
        return default


def _is_colab() -> bool:
    """True only in a real Colab runtime.

    The cheap test used to be ``"content" in os.getcwd()``, which matches any
    path merely *containing* that word — ``~/course_content`` on a student's own
    laptop was detected as Colab, sending built wheels to
    ``/content/drive/MyDrive`` and prompting a Drive mount that cannot work.
    Colab's own environment variables and the importable ``google.colab`` are
    definitive; the path test is kept but anchored.
    """
    if os.getenv("COLAB_RELEASE_TAG") or os.getenv("COLAB_GPU"):
        return True
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except Exception:
        pass
    cwd = _safe_cwd("")
    return cwd == "/content" or cwd.startswith("/content/")


def _is_wsl() -> bool:
    """True inside a WSL distribution (WSL2 or WSL1)."""
    if os.getenv("WSL_DISTRO_NAME") or os.getenv("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


class PlatformManager:
    """Detect the compute platform and a sensible working root for it.

    Attributes
    ----------
    platform : str
        One of "Colab", "RunPod", "LightningAI", "WSL", "LocalPC".
    local_path : str
        A platform-appropriate working root (trailing slash), e.g. "/content/".
    """

    def __init__(self):
        self.platform, self.local_path = self.detect_platform()

    @staticmethod
    def detect_platform() -> Tuple[str, str]:
        if os.getenv("RUNPOD_POD_ID"):
            return "RunPod", "/workspace/"
        if _is_colab():
            return "Colab", "/content/"
        if os.getenv("LIGHTNING_ARTIFACTS_DIR"):
            return "LightningAI", os.getenv("LIGHTNING_ARTIFACTS_DIR") + "/"
        cwd = _safe_cwd().rstrip("/") + "/"
        if _is_wsl():
            return "WSL", cwd
        return "LocalPC", cwd

    @staticmethod
    def mount_gdrive():
        """Mount Google Drive in Colab (no-op elsewhere)."""
        if _is_colab():
            try:
                from google.colab import drive  # type: ignore
                drive.mount("/content/drive")
            except Exception as e:
                print(f"Failed to mount Google Drive: {e}")
        else:
            print("Google Drive mount is only applicable in Colab.")
