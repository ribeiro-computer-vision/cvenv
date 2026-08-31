# cvenv

À-la-carte environment installer for computer-vision / ML courses and projects.
Set up **PyTorch3D**, **MASt3R**, **SAM2**, and the base scientific-Python stack —
each an independent, idempotent, verifiable component — from a notebook or the
command line. It carries the hard-won install lessons (numpy-2 ABI pinning,
`pytorch3d._C` verification, resumable multi-GB checkpoint downloads) as
documented behaviour so students don't have to rediscover them.

`cvenv` itself has **no heavy dependencies** — it stays importable on a bare
machine and installs the heavy things into whatever environment it runs in
(notebook kernel, Colab, Lightning Studio, RunPod, or a local box).

## Install

```bash
pip install "git+https://github.com/ribeiro-computer-vision/cvenv@v0.1.12"
```

(Pin a tag so a tutorial keeps working across semesters; bump it when you
re-validate against the current Colab stack.)

## Use it as a pre-notebook script (students)

Run once before opening the notebook — installs land in the active kernel/env:

```bash
cvenv install pytorch3d mast3r sam2
cvenv verify  pytorch3d mast3r sam2
```

In a Colab / Jupyter cell:

```python
!pip install "git+https://github.com/ribeiro-computer-vision/cvenv@v0.1.12"
!cvenv install pytorch3d mast3r sam2
# If numpy was changed, Runtime -> Restart, then continue.
```

## Use it from Python (notebooks)

```python
import cvenv

cvenv.setup(["pytorch3d", "mast3r"])          # bulk (deps resolved first)
cvenv.get_component("sam2").install(checkpoint_dir="/content/drive/MyDrive/ckpts")
cvenv.get_component("pytorch3d").verify()     # sanity check
```

## Components

| name        | what it installs |
|-------------|------------------|
| `science`   | numpy (pinned `>=2.0,<2.1`, or `>=2.1` on Python 3.13+), scipy, matplotlib, pandas, scikit-image, scikit-learn, opencv, pillow, tqdm, imageio, colorama. Base for pure-numpy/scipy material (Kalman, Lie groups). |
| `opengl`    | PyOpenGL + system GL/GLUT dev libs (for pyrender / rendering). |
| `pytorch3d` | Facebook PyTorch3D — prebuilt wheel if possible, else source build. `verify()` rasterizes a triangle on the GPU, so an architecture mismatch shows up here rather than at your first render. Depends on `opengl`. |
| `mast3r`    | NAVER MASt3R (clone repo, install deps, fetch checkpoint). |
| `sam2`      | Meta Segment Anything 2 (pip install + checkpoint download). |

`cvenv list -v` prints each component's "why this is tricky" teaching note.

## Build PyTorch3D once, reuse forever

A prebuilt wheel is the fast path. If none is available and `cvenv` builds
PyTorch3D from source, it **saves the resulting wheel to a persistent directory**
so you never pay the multi-minute compile twice:

| platform     | default save dir            |
|--------------|-----------------------------|
| Colab        | `/content/drive/MyDrive/cvenv_wheels` (if Drive mounted; else ephemeral `/content/...` with a warning) |
| RunPod       | `/workspace/cvenv_wheels`   |
| Lightning AI | `…/this_studio/cvenv_wheels`|
| local        | `~/.cvenv/wheels`           |

```python
# first time (builds + saves the wheel, then installs it)
cvenv.get_component("pytorch3d").install(from_source=True)
#   → 💾 saved reusable wheel: /content/drive/MyDrive/cvenv_wheels/pytorch3d-….whl

# every session after (seconds, not minutes)
cvenv.get_component("pytorch3d").install(
    wheel_url="/content/drive/MyDrive/cvenv_wheels/pytorch3d-….whl")
```

Override the location with `wheel_out_dir=...` (Python) or `--wheel-out-dir DIR`
(CLI). On Colab, mount Drive **before** building so the wheel persists.

**Build the wheel without installing** (e.g. produce an artifact on a build box to
download or hand to students) — `build_wheel` / `cvenv build-wheel`:

```python
whl = cvenv.get_component("pytorch3d").build_wheel(out_dir="./wheels")   # returns the path
```
```bash
cvenv build-wheel pytorch3d --wheel-out-dir ./wheels
```

A wheel is valid **only** where python (cp), torch, and CUDA match the machine it was
built on — so build it on (or identically to) the runtime you'll use it on.
`build_wheel` is **idempotent**: it reuses an existing wheel in the output dir; pass
`force=True` / `--force` to rebuild it anyway.

### GPU architecture, and why the default ends in `+PTX`

A wheel also carries **native GPU code only for the architectures it was compiled
for**. That is a separate axis from python/torch/CUDA, and it bites on any platform
that hands out different hardware between sessions: a wheel built when Colab gave
you an L4 (`sm_89`) dies on the next session's T4 (`sm_75`) with

```
CUDA error: no kernel image is available for execution on the device
(cudaErrorNoKernelImageForDevice)
```

`build_wheel` therefore targets **one architecture plus `+PTX`**:

- **one** `-gencode` target, because several is what provokes pulsar's
  `undefined reference … <true>` link failure;
- **`+PTX`**, which adds a portable image the driver JIT-compiles on demand.

PTX is **forward-only** — `compute_75` PTX runs on `sm_80/86/89/90`, but nothing
older. A wheel is portable across a fleet only if built for the fleet's *oldest*
member, so where the platform's floor is known cvenv targets that rather than
whatever card is attached right now:

| Platform | Default target | Covers |
|----------|----------------|--------|
| Colab    | `7.5+PTX` (T4) | T4 natively; A100 / L4 / H100 by JIT |
| other    | detected capability `+PTX` | that GPU natively; newer by JIT |

When the floor differs from the attached GPU the build says so. Override with
`arch_list=...`, which is used verbatim:

```python
whl = cvenv.get_component("pytorch3d").build_wheel(arch_list="8.0+PTX", force=True)
```

Check what actually went into a built extension:

```python
import glob, os, pytorch3d
so = glob.glob(os.path.dirname(pytorch3d.__file__) + "/_C*.so")[0]
!cuobjdump --list-elf {so}    # native architectures
!cuobjdump --list-ptx {so}    # the PTX that makes newer GPUs work
```

JIT costs a few seconds at the first kernel launch on a non-native GPU, and the
session is ephemeral, so you may pay it once per session. That is the price of one
wheel that works everywhere.

### Wheel provenance

A wheel filename records the python tag and platform but **not** torch or CUDA,
even though the compiled `_C` extension is linked against both. So a wheel left in
Drive still installs cleanly after a runtime's torch moves under it, then fails at
`import pytorch3d._C` with an undefined symbol.

Each built wheel therefore gets a `<wheel>.build.json` sidecar recording what it was
compiled against, and cvenv checks it:

```python
cvenv.wheel_compatibility(whl)   # (True | False | None, reasons)
cvenv.read_wheel_metadata(whl)   # python / torch / CUDA / arch / timestamp
```

- `build_wheel` will **not** hand back a wheel the sidecar proves cannot load here —
  it explains which of python/torch/CUDA differs and rebuilds instead.
- `install(wheel_url=...)` reports compatibility *before* installing, and fetches the
  sidecar alongside a remote wheel.
- Wheels built before this existed have no sidecar and are reported as *unverified* —
  reused as before, since absence of evidence isn't evidence of a mismatch.

**If you copy or share a wheel, take the sidecar with it**; without it the check
degrades to *unverified*.

## CLI reference

```
cvenv list [-v]                                   # components (+ teaching notes)
cvenv platform                                    # detected platform + paths
cvenv install <components...> [options]
    --wheel-url URL        prebuilt PyTorch3D wheel
    --checkpoint-dir DIR   reuse a locally staged checkpoint (skip download)
    --from-source          force a PyTorch3D source build
    --wheel-out-dir DIR    where to save a source-built wheel (default: persistent per-platform dir)
    --force                reinstall even if already present
cvenv verify <components...>                       # import/probe sanity checks
cvenv build-wheel [pytorch3d] [options]            # build a reusable wheel, do NOT install
    --wheel-out-dir DIR    output dir (default: persistent per-platform dir)
    --ref REF              git ref/branch/tag to build (default: stable)
    --force                rebuild even if a wheel already exists in the output dir
    --cuda-home DIR        CUDA toolkit dir (must match torch's CUDA; default /usr/local/cuda)
    --arch-list LIST       TORCH_CUDA_ARCH_LIST, e.g. 8.0 (default: the running GPU's arch)
```

## Notebooks

Ready-to-run notebooks live in [`notebooks/`](notebooks/). To use one: open it in
Colab (**Runtime → Change runtime type → GPU** for the GPU components), then run the
cells top to bottom. Each installs `cvenv` in its first cell, so nothing else needs to
be set up first.

- **`cvenv_demo.ipynb`** — one cell per component (`install` + `verify`). Use it as a
  menu: copy the single cell you need (e.g. just `sam2`) into your own tutorial
  notebook.
- **`cvenv_create_pytorch3d_wheel.ipynb`** — build a PyTorch3D wheel matching the
  current runtime, save it to Drive, and install it. Run this **once** when a runtime
  has no working wheel; afterwards every session reuses the saved `.whl` in seconds.
  A `FORCE_REBUILD` flag chooses between reusing and recompiling, and the cell above
  it reports any wheel already saved, with what it was built against.

## Notes

- **numpy**: modern Colab/Studio are numpy-2 native, and installing `numpy<2`
  breaks the ABI of their preinstalled cv2/scipy. `cvenv` pins `numpy>=2.0,<2.1`
  — the window that also satisfies numba — and warns you to restart the kernel if
  numpy changed mid-session. **On Python 3.13+ the pin is `numpy>=2.1`**: NumPy
  gained 3.13 support in 2.1.0, so asking for `<2.1` there has no wheel and pip
  silently compiles NumPy from source, which looks exactly like the slow source
  build a prebuilt wheel is meant to avoid.
- **torch** is intentionally *not* a default component (it's preinstalled on
  Colab/Studio/RunPod, and reinstalling it is risky).
- **PyTorch3D** verification always tests `import pytorch3d._C`, not just
  `import pytorch3d` — the latter succeeds even when the compiled extension is
  broken.
- **PyTorch3D source build**: by default it compiles for the **running GPU's**
  compute capability only. Building for many arches at once is a common trigger of
  the pulsar `undefined reference … <true>` link error; single-arch avoids it and is
  faster. Override with `arch_list=` / `--arch-list` (e.g. `8.0`), and point
  `cuda_home=` / `--cuda-home` at a toolkit matching `torch.version.cuda` if the
  default `/usr/local/cuda` doesn't. (A CUDA/torch mismatch is the *other* pulsar
  link cause.)
- **CUDA 13 / pulsar link error:** CUDA 13's nvcc defaults
  `-static-global-template-stub=true`, which stubs out pulsar's cross-file
  `__global__` template specializations and fails the final link with
  `undefined reference to pulsar::Renderer::…<true>`. The source build sets
  `NVCC_APPEND_FLAGS=-static-global-template-stub=false` (nvcc's own suggested
  fix) so pulsar links on CUDA 13. This is why the same source builds on CUDA
  12.x but not 13 until this flag is set.
- **Conda envs (e.g. Lightning AI Studio):** conda's bundled `compiler_compat/ld`
  can't link CUDA-13 / sm_90 objects and fails the build with `final link failed:
  bad value` + spurious pulsar `undefined reference … <true>` errors — the *same*
  source builds fine outside conda. The source build detects this and temporarily
  switches to the system `ld` (reversible; no-op when not in a conda env).
