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
pip install "git+https://github.com/ribeiro-computer-vision/cvenv@v0.1.3"
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
!pip install "git+https://github.com/ribeiro-computer-vision/cvenv@v0.1.3"
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
| `science`   | numpy (pinned `>=2.0,<2.1`), scipy, matplotlib, pandas, scikit-image, scikit-learn, opencv, pillow, tqdm, imageio, colorama. Base for pure-numpy/scipy material (Kalman, Lie groups). |
| `opengl`    | PyOpenGL + system GL/GLUT dev libs (for pyrender / rendering). |
| `pytorch3d` | Facebook PyTorch3D — prebuilt wheel if possible, else source build. Depends on `opengl`. |
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
  Shows both the Python (`build_wheel`) and CLI (`cvenv build-wheel`) paths.

## Notes

- **numpy**: modern Colab/Studio are numpy-2 native. `cvenv` pins `numpy>=2.0,<2.1`
  and warns you to restart the kernel if numpy changed mid-session — installing
  `numpy<2` breaks the ABI of preinstalled cv2/scipy.
- **torch** is intentionally *not* a default component (it's preinstalled on
  Colab/Studio/RunPod, and reinstalling it is risky).
- **PyTorch3D** verification always tests `import pytorch3d._C`, not just
  `import pytorch3d` — the latter succeeds even when the compiled extension is
  broken.
