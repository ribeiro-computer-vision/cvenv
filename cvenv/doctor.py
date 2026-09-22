"""Preflight check for a machine that is about to build or install PyTorch3D.

Building PyTorch3D from source takes tens of minutes and fails late: the
interesting errors come out of nvcc, long after the point where the real cause
(a CPU-only torch, an invisible GPU, a toolkit that does not match torch) could
have been spotted in a second. On Colab the runtime is fixed and known-good, so
that rarely bit anyone. On a student's own Linux box or WSL2 distro every one of
those assumptions can be wrong, and the resulting error message points nowhere
near the cause.

``cvenv doctor`` checks the things the build actually depends on, in the order
they fail, and says what to do about each one. It installs nothing and changes
nothing.

WSL2 gets its own advice throughout: the NVIDIA driver belongs on the *Windows*
side (installing a Linux driver inside the distro is the classic way to break a
working setup), while the CUDA toolkit belongs *inside* the distro.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

OK, WARN, BAD = "✅", "⚠️ ", "❌"


def _run(cmd: list[str]) -> str | None:
    """Return a command's stdout, or None if it is missing or fails."""
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


def _cuda_major(version: str | None) -> str | None:
    return version.split(".")[0] if version else None


# Headers a torch CUDA extension needs from the toolkit, and the apt package
# (suffixed with the toolkit version) that supplies each. torch's own ATen
# headers pull in cuSPARSE, cuBLAS and cuSOLVER -- ATen/cuda/CUDAContextLight.h
# includes cusparse.h directly -- so a toolkit carrying only nvcc and cudart
# compiles nothing. That failure arrives ~1 minute into a build as
# "fatal error: cusparse.h: No such file or directory", which reads like a
# PyTorch3D problem and is not one.
_REQUIRED_HEADERS = [
    ("cuda_runtime.h", "cuda-cudart-dev"),
    ("cusparse.h",     "libcusparse-dev"),
    ("cublas_v2.h",    "libcublas-dev"),
    ("cusolverDn.h",   "libcusolver-dev"),
    ("thrust/version.h", "cuda-cccl"),
    ("cub/version.cuh",  "cuda-cccl"),
]


def _missing_headers(include_dir: str) -> list[tuple[str, str]]:
    """Which required headers are absent from a toolkit's include dir."""
    if not include_dir or not os.path.isdir(include_dir):
        return []
    return [(h, pkg) for h, pkg in _REQUIRED_HEADERS
            if not os.path.exists(os.path.join(include_dir, h))]


def _matching_toolkit(torch_cuda: str | None) -> str | None:
    """Delegate to the same finder the build uses, so advice and behaviour agree."""
    from .components.pytorch3d import find_matching_toolkit
    return find_matching_toolkit(torch_cuda)


def _report(rows: list[tuple[str, str, str]], title: str) -> None:
    print(f"\n{title}")
    width = max(len(label) for _, label, _ in rows) if rows else 0
    for mark, label, detail in rows:
        print(f"  {mark} {label:<{width}}  {detail}")


def run_doctor(cuda_home: str | None = None) -> int:
    """Print a preflight report. Return 0 if a CUDA build can proceed, else 1.

    ``cuda_home`` mirrors the ``--cuda-home`` flag on install/build-wheel, so the
    machine can be checked against the same toolkit the build will use without
    exporting CUDA_HOME first.
    """
    from .platform import PlatformManager

    pm = PlatformManager()
    is_wsl = pm.platform == "WSL"
    problems: list[str] = []
    advice: list[str] = []

    # ---------------------------------------------------------------- machine
    rows = [
        (OK, "platform", pm.platform + (" (Windows Subsystem for Linux)" if is_wsl else "")),
        (OK, "python", f"{sys.version.split()[0]}  ({sys.executable})"),
    ]
    from . import __version__
    rows.append((OK, "cvenv", __version__))
    _report(rows, "Machine")

    # ----------------------------------------------------------------- driver
    rows = []
    smi = _run(["nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader"])
    if smi:
        first = smi.strip().splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        name = parts[0] if parts else "?"
        driver = parts[1] if len(parts) > 1 else "?"
        memory = parts[2] if len(parts) > 2 else "?"
        rows.append((OK, "GPU", name))
        rows.append((OK, "driver", driver))
        rows.append((OK, "GPU memory", memory))
    else:
        rows.append((BAD, "nvidia-smi", "not found, or reported no GPU"))
        problems.append("no usable NVIDIA driver")
        if is_wsl:
            advice.append(
                "WSL2: install the NVIDIA driver on the WINDOWS side (the normal\n"
                "     Windows GeForce/Studio driver includes WSL support). Do NOT\n"
                "     install a Linux NVIDIA driver inside the distro — that breaks\n"
                "     the passthrough. After installing, run `wsl --shutdown` in\n"
                "     PowerShell and reopen the distro.\n"
                "     Also confirm you are on WSL2, not WSL1: `wsl -l -v` in\n"
                "     PowerShell. WSL1 cannot see the GPU at all.")
        else:
            advice.append(
                "Install the NVIDIA driver for your distribution, then reboot.\n"
                "     `nvidia-smi` must list your GPU before anything else can work.")
    _report(rows, "NVIDIA driver")

    # ------------------------------------------------------------------ torch
    rows = []
    torch_cuda = None
    try:
        import torch
        rows.append((OK, "torch", torch.__version__))
        torch_cuda = torch.version.cuda
        if torch_cuda is None:
            rows.append((BAD, "torch CUDA", "CPU-only build"))
            problems.append("torch is a CPU-only build")
            advice.append(
                "Install a CUDA build of torch:\n"
                "         pip install --force-reinstall torch torchvision \\\n"
                "             --index-url https://download.pytorch.org/whl/cu124\n"
                "     (a plain `pip install torch` can resolve to a +cpu build).")
        else:
            rows.append((OK, "torch CUDA", torch_cuda))
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                rows.append((OK, "GPU visible to torch",
                             f"{torch.cuda.get_device_name(0)}  "
                             f"(compute capability {cap[0]}.{cap[1]})"))
            else:
                rows.append((BAD, "GPU visible to torch",
                             "no — torch.cuda.is_available() is False"))
                problems.append("torch cannot see the GPU")
                advice.append(
                    "torch is a CUDA build but finds no GPU. Usually the driver is\n"
                    "     missing or older than torch's CUDA runtime requires"
                    + (", or the distro is WSL1 rather than WSL2." if is_wsl else "."))
    except ImportError:
        rows.append((BAD, "torch", "not installed in this Python"))
        problems.append("torch is not installed")
        advice.append(
            "Install a CUDA build of torch first:\n"
            "         pip install torch torchvision \\\n"
            "             --index-url https://download.pytorch.org/whl/cu124")
    _report(rows, "PyTorch")

    # ----------------------------------------------------- CUDA toolkit (nvcc)
    # Only needed to BUILD a wheel. Installing a prebuilt wheel does not need
    # nvcc at all, so a missing toolkit is a warning rather than a failure.
    rows = []
    # Resolve nvcc exactly the way torch does: cpp_extension reads
    # $CUDA_HOME/bin/nvcc, NOT whatever `nvcc` is first on PATH. Reading PATH
    # here reported the wrong compiler on an image that ships nvcc 13 as
    # /usr/local/cuda while CUDA_HOME points at a 12.8 toolkit -- the version
    # torch would use matched, but the report claimed a mismatch and refused.
    cuda_home = (cuda_home or os.environ.get("CUDA_HOME")
                 or os.environ.get("CUDA_PATH"))
    toolkit_root = cuda_home or ("/usr/local/cuda"
                                 if os.path.isdir("/usr/local/cuda") else None)
    nvcc_bin, nvcc_from = None, ""
    if toolkit_root:
        candidate = os.path.join(toolkit_root, "bin", "nvcc")
        if os.path.isfile(candidate):
            nvcc_bin, nvcc_from = candidate, " (from CUDA_HOME)" if cuda_home else ""
    if nvcc_bin is None:
        nvcc_bin = shutil.which("nvcc")
        nvcc_from = " (from PATH)" if nvcc_bin and cuda_home else ""

    nvcc_out = _run([nvcc_bin, "--version"]) if nvcc_bin else None
    nvcc_ver = None
    if nvcc_out:
        m = re.search(r"release (\d+\.\d+)", nvcc_out)
        nvcc_ver = m.group(1) if m else "?"
        rows.append((OK, "nvcc", f"{nvcc_ver}{nvcc_from}"))
    else:
        rows.append((WARN, "nvcc", "not found (only needed to BUILD a wheel)"))
        advice.append(
            "No CUDA toolkit found. You do not need one to INSTALL a prebuilt\n"
            "     wheel. To build one, install the toolkit matching torch's CUDA\n"
            + ("     INSIDE the WSL distro, not on Windows — the toolkit and the\n"
               "     driver live on opposite sides of the boundary."
               if is_wsl else "     version shown above."))

    if cuda_home:
        rows.append((OK if os.path.isdir(cuda_home) else BAD, "CUDA_HOME", cuda_home))
    elif os.path.isdir("/usr/local/cuda"):
        rows.append((OK, "CUDA_HOME", "unset — will default to /usr/local/cuda"))
    else:
        rows.append((WARN, "CUDA_HOME", "unset, and /usr/local/cuda does not exist"))

    # A toolkit that exists but is missing library headers fails the build a
    # minute in, so check it here rather than letting nvcc discover it.
    if nvcc_out and toolkit_root:
        missing = _missing_headers(os.path.join(toolkit_root, "include"))
        if missing:
            names = ", ".join(h for h, _ in missing)
            rows.append((BAD, "toolkit headers", f"missing: {names}"))
            problems.append("the CUDA toolkit is missing headers a torch "
                            "extension needs")
            suffix = (nvcc_ver or "").replace(".", "-")
            pkgs = " ".join(sorted({f"{pkg}-{suffix}" for _, pkg in missing}))
            advice.append(
                "The toolkit has nvcc but not the library headers torch's ATen\n"
                "     includes, so every CUDA source fails to compile. Install them:\n"
                f"         sudo apt-get install -y {pkgs}\n"
                f"     Or install the lot in one go: sudo apt-get install -y "
                f"cuda-toolkit-{suffix}")
        else:
            rows.append((OK, "toolkit headers",
                         "cudart, cuBLAS, cuSPARSE, cuSOLVER, thrust/cub all present"))

    if nvcc_ver and torch_cuda:
        if _cuda_major(nvcc_ver) == _cuda_major(torch_cuda):
            note = "match" if nvcc_ver == torch_cuda else "same major version — fine"
            rows.append((OK, "nvcc vs torch CUDA", f"{nvcc_ver} vs {torch_cuda} — {note}"))
        else:
            # This blocks a source build — and `cvenv install` falls back to a
            # source build whenever no prebuilt wheel matches the runtime, which
            # for PyTorch3D is the common case. So it blocks the install too, and
            # reporting it as a mere warning would be misleading: torch's own
            # cpp_extension refuses the build outright with CUDA_MISMATCH.
            want_major = _cuda_major(torch_cuda)
            rows.append((BAD, "nvcc vs torch CUDA",
                         f"{nvcc_ver} vs {torch_cuda} — major versions differ"))
            problems.append(f"CUDA toolkit {nvcc_ver} does not match torch's "
                            f"CUDA {torch_cuda}")
            alt = _matching_toolkit(torch_cuda)
            fix = (f"     A matching toolkit is already on this machine — use it:\n"
                   f"         cvenv install pytorch3d --from-source "
                   f"--cuda-home {alt}"
                   if alt else
                   f"     No CUDA {want_major}.x toolkit found under /usr/local. Two ways out:\n"
                   f"       (a) move torch to the toolkit you already have:\n"
                   f"           pip install --force-reinstall torch \\\n"
                   f"               --index-url https://download.pytorch.org/whl/cu{nvcc_ver.replace('.','')}\n"
                   f"           (check that index has a wheel for your python first)\n"
                   f"       (b) install a CUDA {want_major}.x toolkit and pass --cuda-home.\n"
                   f"     Avoid `conda install cuda-toolkit` inside a managed env: it pulls a\n"
                   f"     cross-compiler whose post-link script expects the BASE conda prefix\n"
                   f"     and fails (seen on Lightning Studio).")
            advice.append(
                f"torch refuses to compile against a different CUDA major version:\n"
                f"     nvcc is {nvcc_ver}, torch was built with {torch_cuda}. This stops\n"
                f"     `cvenv install` too, because it falls back to a source build\n"
                f"     when no prebuilt wheel matches this runtime.\n" + fix)
    _report(rows, "CUDA toolkit")

    # ------------------------------------------------------------ build tools
    rows = []
    for tool, why in (("git", "fetching the pytorch3d source"),
                      ("c++", "compiling the extension")):
        path = shutil.which(tool)
        rows.append((OK, tool, path) if path
                    else (WARN, tool, f"not found — needed for {why}"))
    if not shutil.which("c++") or not shutil.which("git"):
        advice.append("Install build tools:  sudo apt install -y build-essential git")
    _report(rows, "Build tools")

    # --------------------------------------------------------------- wheel dir
    rows = []
    try:
        from .components.pytorch3d import (_default_wheel_dir, wheel_compatibility)
        import glob
        wheel_dir = _default_wheel_dir(pm.platform)
        rows.append((OK, "wheel directory", wheel_dir))
        wheels = sorted(glob.glob(os.path.join(wheel_dir, "pytorch3d-*.whl")))
        if not wheels:
            rows.append((WARN, "wheels present", "none yet"))
        for whl in wheels:
            verdict, reasons = wheel_compatibility(whl)
            mark = {True: OK, False: BAD, None: WARN}[verdict]
            detail = ("matches this runtime" if verdict is True
                      else reasons[0] if reasons else "unverified")
            rows.append((mark, os.path.basename(whl), detail))
    except Exception as exc:
        rows.append((WARN, "wheel directory", f"could not inspect ({exc})"))
    _report(rows, "Wheel cache")

    # ---------------------------------------------------------------- verdict
    print()
    if problems:
        print(f"{BAD} Not ready — " + "; ".join(problems) + ".")
        print("\nWhat to do:")
        for i, item in enumerate(advice, 1):
            print(f"  {i}. {item}")
        return 1

    if advice:                       # warnings only: installing a wheel is fine
        print(f"{OK} Ready to INSTALL a prebuilt PyTorch3D wheel.")
        print(f"{WARN} Building one from source needs a little more:")
        for i, item in enumerate(advice, 1):
            print(f"  {i}. {item}")
        return 0

    print(f"{OK} Ready — this machine can build and install PyTorch3D with CUDA.")
    print("\n  cvenv install pytorch3d        # reuses a cached wheel when it matches")
    print("  cvenv build-wheel pytorch3d   # build one without installing")
    return 0
