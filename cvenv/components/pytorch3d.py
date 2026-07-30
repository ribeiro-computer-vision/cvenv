"""PyTorch3D installer component.

Strategy (in order): a user-provided prebuilt wheel URL -> the official FB wheel
index (Linux, when a CUDA build tag can be derived) -> build from source. Carries
the project's hard-won lessons: verify ``import pytorch3d._C`` (not just
``pytorch3d``), import ``torch`` before ``pytorch3d._C`` so libc10 loads first,
and never verify from inside a ``pytorch3d`` source checkout (cwd shadows the
install).
"""

from __future__ import annotations

import sys

from ..base import Component, register


def _wheel_tag() -> str | None:
    """FB packaging wheel tag, e.g. 'py312_cu121_pyt231'. None if CPU-only
    (no CUDA) -> must build from source."""
    import torch
    pyt = torch.__version__.split("+")[0].replace(".", "")
    cuda = getattr(torch.version, "cuda", None)
    if not cuda:
        return None
    return f"py3{sys.version_info.minor}_cu{cuda.replace('.', '')}_pyt{pyt}"


class PyTorch3D(Component):
    name = "pytorch3d"
    summary = "Facebook PyTorch3D (differentiable 3D). Wheel if possible, else source build."
    teaching_note = (
        "The single hardest install here. A CUDA-matched prebuilt wheel is far "
        "faster than a source build. Always test 'import pytorch3d._C' — plain "
        "'import pytorch3d' succeeds even when the compiled _C extension is "
        "broken. Import torch before pytorch3d._C (libc10), and don't run the "
        "check from inside a pytorch3d/ source dir (cwd shadows the install)."
    )
    requires = ["opengl"]

    def is_installed(self) -> bool:
        try:
            import torch  # noqa: F401
            import pytorch3d._C  # noqa: F401
            return True
        except Exception:
            return False

    def _install(self, platform=None, wheel_url=None, from_source=False, **opts) -> None:
        from .._pip import pip_install

        # Keep numpy-2 ABI intact (see the `science` component's note).
        pip_install("numpy>=2.0,<2.1", check=False)
        pip_install("iopath", check=False)

        if not from_source:
            if wheel_url:
                if not str(wheel_url).lower().split("?")[0].endswith(".whl"):
                    raise ValueError(f"wheel_url does not point to a .whl: {wheel_url}")
                print(f"Installing PyTorch3D from provided wheel: {wheel_url}")
                pip_install(str(wheel_url), check=False)
            elif sys.platform.startswith("linux"):
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

        # Fall back to source build.
        print("Building PyTorch3D from source (this can take several minutes)…")
        pip_install("ninja", extra_args=["--root-user-action", "ignore"], check=False)
        pip_install("git+https://github.com/facebookresearch/pytorch3d.git@stable",
                    check=False)

    def verify(self) -> bool:
        import torch  # noqa: F401  (load libc10 first)
        import pytorch3d
        import pytorch3d._C  # noqa: F401
        print(f"✅ pytorch3d {pytorch3d.__version__} (_C OK)")
        return True


register(PyTorch3D())
