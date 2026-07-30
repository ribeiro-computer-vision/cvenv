"""Thin subprocess / pip helpers.

All installs go through the *current* interpreter (``sys.executable``), so when
cvenv runs inside a notebook kernel the packages land in that same kernel's
environment.
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def run(cmd, check: bool = True) -> int:
    """Run a command (list form), echoing it for transparency."""
    print("$", " ".join(cmd))
    try:
        return subprocess.run(cmd, check=check).returncode
    except subprocess.CalledProcessError as e:
        print(f"Command failed with code {e.returncode}: {' '.join(cmd)}")
        if check:
            raise
        return e.returncode


def pip_install(*pkgs: str, extra_args=None, check: bool = True) -> int:
    """pip install ``pkgs`` into the current interpreter's environment."""
    args = [sys.executable, "-m", "pip", "install"]
    if extra_args:
        args += list(extra_args)
    args += list(pkgs)
    return run(args, check=check)


def apt_available() -> bool:
    return shutil.which("apt-get") is not None


def sudo_prefix() -> list:
    return ["sudo"] if shutil.which("sudo") else []
