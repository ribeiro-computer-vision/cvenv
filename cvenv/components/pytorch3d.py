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

import contextlib
import glob
import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

from ..base import Component, register


@contextlib.contextmanager
def _system_linker():
    """Temporarily neutralize conda's bundled ``compiler_compat/ld`` so the
    final link uses the system linker.

    Conda's old ``compiler_compat/ld`` can't link CUDA-13 / sm_90 objects and
    fails with ``final link failed: bad value`` plus spurious pulsar
    ``undefined reference … <true>`` errors. The exact same PyTorch3D source
    builds fine outside conda (e.g. Colab). No-op when not in a conda env.
    """
    prefix = os.environ.get("CONDA_PREFIX")
    ld = os.path.join(prefix, "compiler_compat", "ld") if prefix else None
    moved = None
    if ld and os.path.exists(ld):
        try:
            os.rename(ld, ld + ".cvenv-bak")
            moved = ld
            print(f"↪ neutralized conda linker → using system ld ({ld})")
        except OSError as e:
            print(f"⚠️  could not neutralize conda ld ({e}); build may fail to link.")
    try:
        yield
    finally:
        if moved and os.path.exists(moved + ".cvenv-bak"):
            try:
                os.rename(moved + ".cvenv-bak", moved)
            except OSError:
                pass


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


# -- wheel provenance ------------------------------------------------------
#
# A wheel filename records the python tag and platform, but NOT torch or CUDA
# — yet a pytorch3d ``_C`` extension is linked against both. A wheel that
# survives in Drive while the runtime's torch moves under it still *installs*,
# then fails at ``import pytorch3d._C`` with an undefined symbol. So each built
# wheel gets a sidecar recording what it was built against.

SIDECAR_SUFFIX = ".build.json"


def wheel_sidecar(whl: str) -> str:
    """Path of the metadata file recorded beside ``whl``."""
    return whl + SIDECAR_SUFFIX


def build_env() -> dict:
    """What the current runtime would build against (or needs to match)."""
    env = {
        "python_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "python": ".".join(str(v) for v in sys.version_info[:3]),
        "torch": None,
        "cuda": None,
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
    except Exception:
        pass
    return env


def read_wheel_metadata(whl: str):
    """Metadata recorded beside ``whl``, or ``None`` if there is none."""
    import json
    try:
        with open(wheel_sidecar(whl)) as fh:
            return json.load(fh)
    except Exception:
        return None


def _write_wheel_metadata(whl: str, **extra) -> None:
    import json
    from datetime import datetime, timezone
    meta = build_env()
    meta["built"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta.update({k: v for k, v in extra.items() if v is not None})
    try:
        with open(wheel_sidecar(whl), "w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
    except Exception as e:                                   # never fail a build
        print(f"   (could not record build metadata: {e})")


def _short(v):
    """major.minor of a version string; extensions are tied to that, not the patch."""
    return ".".join(str(v).split("+")[0].split(".")[:2]) if v else v


def wheel_compatibility(whl: str):
    """Is ``whl`` usable in this runtime?

    Returns ``(verdict, reasons)`` where verdict is True (matches), False
    (provably not), or None (no metadata recorded — unknowable from the
    filename alone). Compares torch and CUDA at major.minor, since that is the
    granularity at which a compiled extension stays loadable.
    """
    meta = read_wheel_metadata(whl)
    now = build_env()
    if not meta:
        return None, ["no build metadata recorded beside this wheel"]

    reasons, unchecked = [], []
    for key, label in (("python_tag", "python"), ("torch", "torch"), ("cuda", "CUDA")):
        was, is_ = meta.get(key), now.get(key)
        if was is None or is_ is None:
            unchecked.append(label)
            continue
        same = was == is_ if key == "python_tag" else _short(was) == _short(is_)
        if not same:
            reasons.append(f"{label}: built against {was}, this runtime has {is_}")
    if reasons:
        return False, reasons
    if unchecked:
        # Everything comparable matched, but something could not be compared —
        # say so rather than claim a clean match.
        return None, [f"could not compare {', '.join(unchecked)} "
                      "(not recorded, or not importable here)"]
    return True, []


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
            verdict, reasons = wheel_compatibility(whl)
            if verdict is False:
                # Recorded metadata proves this wheel cannot load here. Reusing
                # it would install cleanly and then fail at `import
                # pytorch3d._C`, so rebuild rather than hand back a dud.
                print(f"⚠️  not reusing {os.path.basename(whl)} — built for a "
                      "different runtime:")
                for r in reasons:
                    print(f"      • {r}")
                print("   Rebuilding for this runtime.")
            elif verdict is None:
                print(f"✅ reusing existing wheel: {whl}\n"
                      f"   ({reasons[0]}; it is only valid where python, torch and "
                      "CUDA all match the machine it was built on — pass "
                      "force=True / --force to rebuild.)")
                return whl
            else:
                meta = read_wheel_metadata(whl) or {}
                print(f"✅ reusing existing wheel: {whl}\n"
                      f"   (built {meta.get('built', '?')} against torch "
                      f"{meta.get('torch')} / CUDA {meta.get('cuda')} — matches "
                      "this runtime.)")
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

        # Build for a SINGLE GPU arch. Prefer an explicit arch_list, else the
        # running GPU's compute capability. This intentionally OVERRIDES any
        # ambient TORCH_CUDA_ARCH_LIST — Lightning Studio presets it to every
        # arch, which slows the build and aggravates the pulsar link issue.
        arch = arch_list
        if not arch:
            try:
                import torch
                if torch.cuda.is_available():
                    maj, mn = torch.cuda.get_device_capability(0)
                    arch = f"{maj}.{mn}"
            except Exception:
                arch = None
        if not arch:
            arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        if arch:
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch

        # THE key fix for CUDA 13: nvcc now defaults
        # -static-global-template-stub=true in whole-program mode (-rdc=false),
        # which turns pulsar's cross-file __global__ template specializations
        # (calc_signature<true>, render<true>, …) into stubs with no definition
        # → "undefined reference … <true>" at final link. nvcc's own warning
        # (#20280) says to set it false. Apply to every nvcc call in the build.
        stub_flag = "-static-global-template-stub=false"
        cur = os.environ.get("NVCC_APPEND_FLAGS", "")
        if stub_flag not in cur:
            os.environ["NVCC_APPEND_FLAGS"] = (cur + " " + stub_flag).strip()

        print(f"TORCH_CUDA_ARCH_LIST = {os.environ.get('TORCH_CUDA_ARCH_LIST')} | "
              f"CUDA_HOME = {os.environ.get('CUDA_HOME')} | "
              f"NVCC_APPEND_FLAGS = {os.environ.get('NVCC_APPEND_FLAGS')}")

        spec = f"git+https://github.com/facebookresearch/pytorch3d.git@{ref}"
        before = set(existing)

        # torch/iopath/ninja are already present, so skip build isolation (an
        # isolated env wouldn't have torch, which PyTorch3D imports at build time).
        # _system_linker() works around conda's compiler_compat/ld failing to
        # link CUDA-13/sm_90 objects (the pulsar "undefined reference" link error).
        with _system_linker():
            run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                 "--no-build-isolation", spec, "-w", out_dir], check=False)

        whls = glob.glob(os.path.join(out_dir, "pytorch3d-*.whl"))
        if not whls:
            raise RuntimeError(f"wheel build produced no .whl in {out_dir}")
        # prefer a freshly-produced wheel; fall back to newest overall
        fresh = [w for w in whls if w not in before]
        whl = max(fresh or whls, key=os.path.getmtime)
        _write_wheel_metadata(whl, ref=ref,
                              arch_list=os.environ.get("TORCH_CUDA_ARCH_LIST"),
                              cuda_home=os.environ.get("CUDA_HOME"))
        print(f"💾 saved reusable wheel: {whl}\n"
              f"   provenance recorded in {os.path.basename(wheel_sidecar(whl))}")
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
