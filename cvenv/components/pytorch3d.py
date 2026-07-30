"""PyTorch3D installer component.

Strategy: a user-provided prebuilt wheel (downloaded locally, then pip-installed)
-> the official FB wheel index (Linux, when a CUDA build tag can be derived) ->
build from source. Carries the project's hard-won lessons: verify
``import pytorch3d._C`` (not just ``pytorch3d``), import ``torch`` before
``pytorch3d._C`` so libc10 loads first, and never verify from inside a
``pytorch3d`` source checkout (cwd shadows the install).

Key behaviour: when you *explicitly* pass ``wheel_url``, a failure is raised with
a diagnosis — it does NOT silently fall back to a slow source build (that
silent fallback is exactly what makes "I gave it a wheel but it built from
source" so confusing). The source fallback only kicks in when no wheel was given.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

from ..base import Component, register


def _wheel_tag() -> str | None:
    """FB packaging wheel tag, e.g. 'py312_cu121_pyt231'. None if CPU-only."""
    import torch
    pyt = torch.__version__.split("+")[0].replace(".", "")
    cuda = getattr(torch.version, "cuda", None)
    if not cuda:
        return None
    return f"py3{sys.version_info.minor}_cu{cuda.replace('.', '')}_pyt{pyt}"


def _runtime_desc() -> str:
    try:
        import torch
        return (f"python=cp3{sys.version_info.minor}, torch={torch.__version__}, "
                f"cuda={getattr(torch.version, 'cuda', None)}")
    except Exception:
        return f"python=cp3{sys.version_info.minor}, torch=<not importable>"


def _normalize_dropbox(url: str) -> str:
    """Force dl=1 on Dropbox share links (harmless for other hosts)."""
    p = urlparse(url)
    if "dropbox.com" not in p.netloc.lower():
        return url
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    q["dl"] = "1"
    return urlunparse(p._replace(query=urlencode(q, doseq=True)))


def _c_import_error() -> str:
    try:
        import torch  # noqa: F401
        import pytorch3d._C  # noqa: F401
        return ""
    except Exception as e:
        return f"{type(e).__name__}: {e}"


class PyTorch3D(Component):
    name = "pytorch3d"
    summary = "Facebook PyTorch3D (differentiable 3D). Wheel if possible, else source build."
    teaching_note = (
        "The single hardest install here. A CUDA-matched prebuilt wheel is far "
        "faster than a source build. Always test 'import pytorch3d._C' — plain "
        "'import pytorch3d' succeeds even when the compiled _C extension is "
        "broken. A wheel whose filename tag (cp312/linux) matches will pip-install "
        "cleanly yet still fail _C at runtime if its torch/CUDA build differs from "
        "the runtime's — so verify _C, not the pip exit code."
    )
    requires = ["opengl"]

    def is_installed(self) -> bool:
        return _c_import_error() == ""

    # -- helpers -----------------------------------------------------------

    def _fetch_wheel(self, wheel_url: str) -> Path:
        """Return a local .whl path. Downloads remote URLs first (reliable for
        Dropbox), validating the file is really a zip/wheel and not an HTML
        error page."""
        # already a local file?
        if os.path.exists(wheel_url):
            return Path(wheel_url)

        from ..download import download_resumable
        url = _normalize_dropbox(wheel_url)
        name = os.path.basename(urlparse(url).path) or "pytorch3d.whl"
        if not name.endswith(".whl"):
            raise ValueError(f"URL does not point to a .whl: {wheel_url}")
        dest = Path(tempfile.gettempdir()) / name
        print(f"⬇️  downloading wheel: {name}")
        download_resumable(url, dest)

        # sanity: a wheel is a zip → starts with 'PK'; an HTML error page is not
        with open(dest, "rb") as f:
            head = f.read(2)
        if head != b"PK":
            size = dest.stat().st_size
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"downloaded file is not a valid wheel (got {size} bytes, not a "
                f"zip). The share link probably returned an HTML page — check the "
                f"URL is a direct download (Dropbox: keep '?...&dl=1'). URL: {url}")
        return dest

    def _source_build(self) -> None:
        from .._pip import pip_install
        print("Building PyTorch3D from source (this can take several minutes)…")
        pip_install("ninja", extra_args=["--root-user-action", "ignore"], check=False)
        pip_install("git+https://github.com/facebookresearch/pytorch3d.git@stable",
                    check=False)

    # -- install -----------------------------------------------------------

    def _install(self, platform=None, wheel_url=None, from_source=False, **opts) -> None:
        from .._pip import pip_install

        # Keep numpy-2 ABI intact (see the `science` component's note).
        pip_install("numpy>=2.0,<2.1", check=False)
        pip_install("iopath", check=False)

        if from_source:
            self._source_build()
            return

        if wheel_url:
            local = self._fetch_wheel(wheel_url)
            print(f"Installing PyTorch3D from wheel: {local.name}")
            pip_install(str(local), extra_args=["--force-reinstall", "--no-deps"],
                        check=False)
            if self.is_installed():
                print("✅ installed from provided wheel.")
                return
            # Explicit wheel failed → tell the user WHY instead of silently
            # falling back to a source build.
            raise RuntimeError(
                "the provided wheel installed but 'import pytorch3d._C' failed:\n"
                f"    {_c_import_error()}\n"
                f"Runtime: {_runtime_desc()}\n"
                f"Wheel:   {local.name}\n"
                "Most likely the wheel's torch/CUDA build doesn't match this "
                "runtime. Rebuild the wheel against the runtime's torch/CUDA, or "
                "pass from_source=True to build here.")

        # No wheel provided: best-effort official index (Linux), else source.
        if sys.platform.startswith("linux"):
            tag = _wheel_tag()
            if tag:
                index = (f"https://dl.fbaipublicfiles.com/pytorch3d/packaging/"
                         f"wheels/{tag}/download.html")
                print(f"Trying official PyTorch3D wheel index (tag {tag}).")
                pip_install("pytorch3d",
                            extra_args=["--no-index", "--no-cache-dir", "-f", index],
                            check=False)
        if self.is_installed():
            return
        self._source_build()

    def verify(self) -> bool:
        import torch  # noqa: F401  (load libc10 first)
        import pytorch3d
        import pytorch3d._C  # noqa: F401
        print(f"✅ pytorch3d {pytorch3d.__version__} (_C OK)")
        return True


register(PyTorch3D())
