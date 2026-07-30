"""Compute-platform detection (Colab / RunPod / Lightning AI Studio / LocalPC).

Genericized from a per-project setup helper into a standalone, reusable class so
components can adapt install behavior to where they run.
"""

from __future__ import annotations

import os
from typing import Tuple


def _is_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except Exception:
        return "content" in os.getcwd()


class PlatformManager:
    """Detect the compute platform and a sensible working root for it.

    Attributes
    ----------
    platform : str
        One of "Colab", "RunPod", "LightningAI", "LocalPC".
    local_path : str
        A platform-appropriate working root (trailing slash), e.g. "/content/".
    """

    def __init__(self):
        self.platform, self.local_path = self.detect_platform()

    @staticmethod
    def detect_platform() -> Tuple[str, str]:
        if os.getenv("RUNPOD_POD_ID"):
            return "RunPod", "/workspace/"
        if "content" in str(os.getcwd()) or _is_colab():
            return "Colab", "/content/"
        if os.getenv("LIGHTNING_ARTIFACTS_DIR"):
            return "LightningAI", os.getenv("LIGHTNING_ARTIFACTS_DIR") + "/"
        return "LocalPC", os.getcwd() + "/"

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
