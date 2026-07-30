"""MASt3R installer component (public NAVER repo + checkpoint).

MASt3R is used as a source-checkout (cloned, not pip-installed), so this clones
the repo, installs its + dust3r's requirements, downloads the public checkpoint,
and puts the repo root on ``sys.path`` so ``import mast3r`` works.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..base import Component, register
from .._pip import run
from ..download import download_resumable

REPO_URL = "https://github.com/naver/mast3r"
CKPT_NAME = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
CKPT_URL = ("https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
            "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")


def _repo_root(base_dir=None) -> Path:
    return Path(base_dir or os.getcwd()) / "mast3r"


def _ensure_on_path(repo_root: Path):
    p = str(repo_root)
    if repo_root.is_dir() and p not in sys.path:
        sys.path.insert(0, p)


class MASt3R(Component):
    name = "mast3r"
    summary = "NAVER MASt3R matcher: clone repo, install deps, fetch checkpoint."
    teaching_note = (
        "MASt3R is used as a source checkout, not a pip package — the repo root "
        "must be on sys.path for 'import mast3r'. Its checkpoint is multi-GB, so "
        "the download resumes on failure. A local copy can be reused via "
        "checkpoint_dir= to skip the download."
    )

    def is_installed(self, base_dir=None) -> bool:
        repo_root = _repo_root(base_dir)
        ckpt = repo_root / "checkpoints" / CKPT_NAME
        _ensure_on_path(repo_root)
        try:
            import importlib.util as ilu
            deps_ok = all(ilu.find_spec(m) is not None for m in ("roma", "einops"))
            return repo_root.is_dir() and ckpt.exists() and deps_ok
        except Exception:
            return False

    def _install(self, platform=None, base_dir=None, checkpoint_dir=None,
                 force_pip=False, **opts) -> None:
        repo_root = _repo_root(base_dir)

        if not repo_root.exists():
            run(["git", "clone", "--recursive", REPO_URL, str(repo_root)], check=True)

        _ensure_on_path(repo_root)

        # Requirements (mast3r + dust3r submodule). Skip if already importable.
        import importlib.util as ilu
        deps_ok = all(ilu.find_spec(m) is not None for m in ("roma", "einops"))
        if force_pip or not deps_ok:
            req = repo_root / "requirements.txt"
            dreq = repo_root / "dust3r" / "requirements.txt"
            dopt = repo_root / "dust3r" / "requirements_optional.txt"
            if req.exists():
                run([sys.executable, "-m", "pip", "install", "-r", str(req)], check=False)
            if dreq.exists():
                run([sys.executable, "-m", "pip", "install", "-r", str(dreq)], check=False)
            if dopt.exists():
                run([sys.executable, "-m", "pip", "install", "-r", str(dopt)], check=False)

        # Checkpoint: pre-staged local copy first, else resumable download.
        ckpt_dir = repo_root / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / CKPT_NAME
        if ckpt_path.exists():
            print(f"✅ checkpoint present: {ckpt_path}")
        elif checkpoint_dir and (Path(checkpoint_dir) / CKPT_NAME).exists():
            import shutil
            src = Path(checkpoint_dir) / CKPT_NAME
            print(f"📄 copying checkpoint from {src}")
            shutil.copy2(src, ckpt_path)
        else:
            print(f"⬇️  downloading MASt3R checkpoint to {ckpt_path}")
            download_resumable(CKPT_URL, ckpt_path)

    def verify(self, base_dir=None) -> bool:
        _ensure_on_path(_repo_root(base_dir))
        import mast3r  # noqa: F401
        from mast3r.model import AsymmetricMASt3R  # noqa: F401
        print("✅ mast3r importable (repo on sys.path, checkpoint present)")
        return True


register(MASt3R())
