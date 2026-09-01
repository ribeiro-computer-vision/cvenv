"""MASt3R installer component (public NAVER repo + checkpoint).

MASt3R is used as a source-checkout (cloned, not pip-installed), so this clones
the repo, installs its + dust3r's requirements, obtains the public checkpoint,
and puts the repo root on ``sys.path`` so ``import mast3r`` works.

Each phase is timed and the totals printed at the end, because on Colab the
cost is dominated by whichever of clone / pip / checkpoint happens to be slow
that session, and that is not obvious from watching the cell.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from ..base import Component, register
from .._pip import run
from ..download import download_resumable

REPO_URL = "https://github.com/naver/mast3r"
CKPT_NAME = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
CKPT_URL = ("https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
            "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")


@contextmanager
def _timed(label: str, timings: list):
    """Time one install phase and record it for the closing summary."""
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        timings.append((label, dt))
        print(f"⏱  {label}: {dt:.1f}s")


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
        "the download resumes on failure. A local copy is reused via "
        "checkpoint_dir=, linked rather than copied so the model load is "
        "the only pass over the file; link_checkpoint=False forces a copy."
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
                 force_pip=False, link_checkpoint=True, **opts) -> None:
        repo_root = _repo_root(base_dir)
        timings: list = []

        if not repo_root.exists():
            with _timed("clone repo", timings):
                run(["git", "clone", "--recursive", REPO_URL, str(repo_root)], check=True)
        else:
            print("✅ repo already cloned")

        _ensure_on_path(repo_root)

        # Requirements (mast3r + dust3r submodule). Skip if already importable.
        import importlib.util as ilu
        deps_ok = all(ilu.find_spec(m) is not None for m in ("roma", "einops"))
        if force_pip or not deps_ok:
            with _timed("install requirements", timings):
                req = repo_root / "requirements.txt"
                dreq = repo_root / "dust3r" / "requirements.txt"
                dopt = repo_root / "dust3r" / "requirements_optional.txt"
                if req.exists():
                    run([sys.executable, "-m", "pip", "install", "-r", str(req)], check=False)
                if dreq.exists():
                    run([sys.executable, "-m", "pip", "install", "-r", str(dreq)], check=False)
                if dopt.exists():
                    run([sys.executable, "-m", "pip", "install", "-r", str(dopt)], check=False)
        else:
            print("✅ requirements already satisfied")

        # Checkpoint.
        ckpt_dir = repo_root / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / CKPT_NAME

        # A link from an earlier session points into a mount that may not be
        # mounted now. Path.exists() follows the link, so a dangling one reads
        # as absent — drop it so the branches below can redo the work.
        if ckpt_path.is_symlink() and not ckpt_path.exists():
            print(f"⚠\ufe0f  stale checkpoint link -> {os.readlink(ckpt_path)}")
            ckpt_path.unlink()

        src = (Path(checkpoint_dir) / CKPT_NAME) if checkpoint_dir else None

        if ckpt_path.exists():
            print(f"✅ checkpoint present: {ckpt_path}")
        elif src is not None and src.exists():
            with _timed("checkpoint from local copy", timings):
                if link_checkpoint:
                    # Link rather than copy. Copying reads the whole multi-GB
                    # file across the mount and writes it to local disk, and
                    # torch.load then reads it again; a link makes the model
                    # load the only pass over the data. Pass
                    # link_checkpoint=False to force a real copy.
                    try:
                        ckpt_path.symlink_to(src)
                        print(f"🔗 linked checkpoint -> {src}")
                    except OSError as exc:
                        import shutil
                        print(f"📄 link unavailable ({exc}); copying from {src}")
                        shutil.copy2(src, ckpt_path)
                else:
                    import shutil
                    print(f"📄 copying checkpoint from {src}")
                    shutil.copy2(src, ckpt_path)
        else:
            if src is not None:
                print(f"ℹ\ufe0f  no checkpoint at {src} — downloading instead")
            with _timed("checkpoint download", timings):
                print(f"⬇\ufe0f  downloading MASt3R checkpoint to {ckpt_path}")
                download_resumable(CKPT_URL, ckpt_path)

        if timings:
            total = sum(dt for _, dt in timings)
            print("")
            print("⏱  install phases")
            for label, dt in timings:
                print(f"     {label:<28} {dt:7.1f}s  ({100 * dt / total:4.1f}%)")
            print(f"     {'total':<28} {total:7.1f}s")

    def verify(self, base_dir=None) -> bool:
        _ensure_on_path(_repo_root(base_dir))
        import mast3r  # noqa: F401
        from mast3r.model import AsymmetricMASt3R  # noqa: F401
        print("✅ mast3r importable (repo on sys.path, checkpoint present)")
        return True


register(MASt3R())
