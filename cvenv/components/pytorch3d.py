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

import glob
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


def _default_wheel_dir(platform=None) -> str:
    """A persistent, platform-appropriate directory to save built wheels so they
    survive session/runtime resets and can be reused next time."""
    if platform is None:
        from ..platform import PlatformManager
        platform = PlatformManager().platform

    if platform == "Colab":
        drive = "/content/drive/MyDrive"
        if os.path.isdir(drive):
            return os.path.join(drive, "cvenv_wheels")
        print("⚠️  Google Drive is not mounted at /content/drive — the built wheel "
              "will go to /content/cvenv_wheels, which is LOST on runtime reset.\n"
              "    To persist it: from google.colab import drive; "
              "drive.mount('/content/drive'), then rebuild.")
        return "/content/cvenv_wheels"
    if platform == "RunPod":
        return "/workspace/cvenv_wheels"                 # persistent volume
    if platform == "LightningAI":
        studio = "/teamspace/studios/this_studio"
        base = studio if os.path.isdir(studio) else os.getcwd()
        return os.path.join(base, "cvenv_wheels")
    # LocalPC / unknown: a stable spot in the user's home
    return os.path.join(os.path.expanduser("~"), ".cvenv", "wheels")


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

    def build_wheel(self, out_dir=None, platform=None, ref="stable",
                    force=False, cuda_home=None, arch_list=None, **opts) -> str:
        """Build a reusable PyTorch3D wheel from source and save it, WITHOUT
        installing. Returns the wheel path.

        Idempotent: if a ``pytorch3d-*.whl`` already exists in ``out_dir`` it is
        reused (no rebuild) unless ``force=True``. A wheel is valid only where
        python (cp), torch, and CUDA match the box it was built on — so if the
        existing wheel came from a different runtime, rebuild with ``force=True``.

        ``cuda_home`` overrides ``CUDA_HOME`` (must match torch's CUDA build, or
        pulsar fails to link). ``arch_list`` sets ``TORCH_CUDA_ARCH_LIST``; if
        unset it defaults to the **running GPU's** compute capability (a single
        arch — building for many ``-gencode`` targets is a common cause of the
        pulsar ``undefined reference … <true>`` link failure).
        """
        from .._pip import pip_install, run

        out_dir = out_dir or _default_wheel_dir(platform)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        existing = glob.glob(os.path.join(out_dir, "pytorch3d-*.whl"))
        if existing and not force:
            whl = max(existing, key=os.path.getmtime)
            print(f"✅ reusing existing wheel: {whl}\n"
                  "   (a wheel only matches the torch/CUDA/python it was built "
                  "against — pass force=True / --force to rebuild for this runtime.)")
            return whl

        print(f"Building a PyTorch3D wheel from source (ref={ref}); "
              "this can take several minutes…")
        # Build-time prerequisites (harmless if already satisfied).
        pip_install("numpy>=2.0,<2.1", check=False)
        pip_install("iopath", check=False)
        pip_install("ninja", extra_args=["--root-user-action", "ignore"], check=False)

        # A CUDA-enabled build needs FORCE_CUDA=1; without it the compile can
        # silently produce a CPU-only extension. CUDA_HOME must point at the
        # toolkit matching the runtime's torch build (a mismatch also breaks
        # pulsar's link).
        os.environ.setdefault("FORCE_CUDA", "1")
        if cuda_home:
            os.environ["CUDA_HOME"] = cuda_home
        elif os.path.isdir("/usr/local/cuda"):
            os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

        # Build for a SINGLE GPU arch by default. Compiling for many -gencode
        # targets (torch's default when TORCH_CUDA_ARCH_LIST is unset) is a
        # common cause of the pulsar "undefined reference … <true>" link failure;
        # restricting to the running GPU's compute capability avoids it and is
        # much faster. Explicit arch_list / a pre-set env var win.
        arch = arch_list or os.environ.get("TORCH_CUDA_ARCH_LIST")
        if not arch:
            try:
                import torch
                if torch.cuda.is_available():
                    maj, mn = torch.cuda.get_device_capability(0)
                    arch = f"{maj}.{mn}"
            except Exception:
                arch = None
        if arch:
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch
            print(f"TORCH_CUDA_ARCH_LIST = {arch}  (CUDA_HOME = "
                  f"{os.environ.get('CUDA_HOME')})")

        spec = f"git+https://github.com/facebookresearch/pytorch3d.git@{ref}"
        before = set(existing)

        # torch/iopath/ninja are already present, so skip build isolation (an
        # isolated env wouldn't have torch, which PyTorch3D imports at build time).
        run([sys.executable, "-m", "pip", "wheel", "--no-deps",
             "--no-build-isolation", spec, "-w", out_dir], check=False)

        whls = glob.glob(os.path.join(out_dir, "pytorch3d-*.whl"))
        if not whls:
            raise RuntimeError(f"wheel build produced no .whl in {out_dir}")
        # prefer a freshly-produced wheel; fall back to newest overall
        fresh = [w for w in whls if w not in before]
        whl = max(fresh or whls, key=os.path.getmtime)
        print(f"💾 saved reusable wheel: {whl}")
        return whl

    def _source_build(self, wheel_out_dir=None, platform=None, **kw) -> None:
        """Build the wheel (saved to a persistent dir), then install it. Reusing
        the saved wheel next session takes seconds, not a fresh compile."""
        from .._pip import pip_install
        # from_source is the recovery path (wheels didn't work), so force a
        # genuine rebuild rather than reusing a possibly-mismatched saved wheel.
        try:
            whl = self.build_wheel(out_dir=wheel_out_dir, platform=platform,
                                   force=True, **kw)
        except RuntimeError as e:
            print(f"⚠️  {e}; installing directly from source as a fallback.")
            pip_install("git+https://github.com/facebookresearch/pytorch3d.git@stable",
                        extra_args=["--no-build-isolation"], check=False)
            return
        pip_install(whl, extra_args=["--force-reinstall", "--no-deps"], check=False)
        print("   ↻ next time, skip the build with:\n"
              f'       cvenv.get_component("pytorch3d").install(wheel_url="{whl}")\n'
              f"     or:  cvenv install pytorch3d --wheel-url {whl}")

    # -- install -----------------------------------------------------------

    def _install(self, platform=None, wheel_url=None, from_source=False,
                 wheel_out_dir=None, **opts) -> None:
        from .._pip import pip_install

        # Keep numpy-2 ABI intact (see the `science` component's note).
        pip_install("numpy>=2.0,<2.1", check=False)
        pip_install("iopath", check=False)

        if from_source:
            self._source_build(wheel_out_dir=wheel_out_dir, platform=platform,
                               **{k: opts[k] for k in ("cuda_home", "arch_list")
                                  if k in opts})
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
        self._source_build(wheel_out_dir=wheel_out_dir, platform=platform)

    def verify(self) -> bool:
        import torch  # noqa: F401  (load libc10 first)
        import pytorch3d
        import pytorch3d._C  # noqa: F401
        print(f"✅ pytorch3d {pytorch3d.__version__} (_C OK)")
        return True


register(PyTorch3D())
