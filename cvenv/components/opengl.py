"""OpenGL / GLUT support (PyOpenGL + system GL dev libs).

Needed by mesh renderers (pyrender, and PyTorch3D's rasterizer build). Kept as a
standalone component so it can be installed on its own and reused by an
offscreen-rendering path.
"""

from __future__ import annotations

from ..base import Component, register


class OpenGL(Component):
    name = "opengl"
    summary = "PyOpenGL + system GL/GLUT dev libraries (for pyrender / rendering)."
    teaching_note = (
        "Offscreen GL on a headless server needs a context: set "
        "PYOPENGL_PLATFORM=egl (GPU, fast) or osmesa (CPU, robust). The apt libs "
        "here (freeglut3-dev, libglew-dev, libsdl2-dev) only install on "
        "Debian/Ubuntu images; they're skipped elsewhere."
    )

    def is_installed(self) -> bool:
        try:
            import OpenGL  # noqa: F401
            return True
        except Exception:
            return False

    def _install(self, platform=None, **opts) -> None:
        from .._pip import pip_install, apt_available, sudo_prefix, run

        if apt_available():
            apt = sudo_prefix() + ["apt-get"]
            run(apt + ["-qq", "update"], check=False)
            run(apt + ["install", "-y", "freeglut3-dev", "libglew-dev", "libsdl2-dev"],
                check=False)
        else:
            print("apt-get not available; skipping system GL/GLUT dev packages.")

        pip_install("PyOpenGL", "PyOpenGL_accelerate", check=False)

    def verify(self) -> bool:
        import OpenGL  # noqa: F401
        print(f"✅ opengl: PyOpenGL {OpenGL.__version__}")
        return True


register(OpenGL())
