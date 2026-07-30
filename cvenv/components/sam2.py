"""SAM2 installer component (Meta Segment Anything 2).

Pip-installs SAM2 from its public GitHub repo and downloads a public checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..base import Component, register
from .._pip import pip_install
from ..download import download_resumable

REPO = "git+https://github.com/facebookresearch/sam2.git"
CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
CKPT_NAME = "sam2.1_hiera_large.pt"


class SAM2(Component):
    name = "sam2"
    summary = "Meta SAM2 video segmentation: pip install + checkpoint download."
    teaching_note = (
        "SAM2 is a normal pip install from GitHub, but its checkpoint (~900 MB) "
        "is downloaded separately — store it under a persistent/mounted path so "
        "it survives kernel or container restarts, not in an ephemeral cwd."
    )

    def is_installed(self, checkpoint_dir=None) -> bool:
        try:
            import sam2  # noqa: F401
        except Exception:
            return False
        ckpt = Path(checkpoint_dir or "checkpoints") / CKPT_NAME
        return ckpt.exists()

    def _install(self, platform=None, checkpoint_dir=None, prestaged_path=None, **opts) -> None:
        try:
            import sam2  # noqa: F401
            print("✅ sam2 package already importable — skipping pip install.")
        except Exception:
            pip_install(REPO, extra_args=["--no-cache-dir"], check=False)

        ckpt_dir = Path(checkpoint_dir or "checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / CKPT_NAME

        if ckpt_path.exists():
            print(f"✅ checkpoint present: {ckpt_path}")
        elif prestaged_path and os.path.exists(prestaged_path):
            import shutil
            print(f"📄 copying pre-staged checkpoint from {prestaged_path}")
            shutil.copy2(prestaged_path, ckpt_path)
        else:
            print(f"⬇️  downloading SAM2 checkpoint to {ckpt_path}")
            download_resumable(CKPT_URL, ckpt_path)

    def verify(self, checkpoint_dir=None) -> bool:
        import sam2  # noqa: F401
        from sam2.build_sam import build_sam2_video_predictor  # noqa: F401
        ckpt = Path(checkpoint_dir or "checkpoints") / CKPT_NAME
        status = "present" if ckpt.exists() else "MISSING"
        print(f"✅ sam2 importable; checkpoint {status} ({ckpt})")
        return True


register(SAM2())
