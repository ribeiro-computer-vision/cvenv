#!/usr/bin/env python3
"""Check that the version agrees everywhere before a release.

cvenv writes its version by hand in several places — ``pyproject.toml``,
``cvenv/__init__.py``, the ``cvenv@vX.Y.Z`` install pin in each notebook, and the
same pin in the README — and nothing else keeps them in step. Both halves of that
have already gone wrong: v0.1.8 was bumped in ``__init__.py`` only, and the
notebook pins were left at v0.1.7 for two releases, quietly installing old code.

Usage
-----
    python3 tools/check_version.py            # every version agrees
    python3 tools/check_version.py v0.1.8     # ...and the tag matches too

With no argument, a tag pointing at HEAD is checked if there is one.

Reads both files *without importing cvenv*, so it works on a bare machine
with no dependencies installed.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    try:                                   # tomllib is 3.11+; cvenv supports 3.9
        import tomllib
        return tomllib.loads(text)["project"]["version"]
    except ModuleNotFoundError:
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
        if not m:
            raise SystemExit("could not find a version in pyproject.toml")
        return m.group(1)


def dunder_version() -> str:
    """Read ``__version__`` from the source, without importing the package."""
    tree = ast.parse((ROOT / "cvenv" / "__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__version__":
                    return ast.literal_eval(node.value)
    raise SystemExit("could not find __version__ in cvenv/__init__.py")


def install_pins() -> list[tuple[str, str]]:
    """``(file, pinned_version)`` for every ``cvenv@vX.Y.Z`` that tells someone
    which version to install.

    The notebooks and the README all install cvenv from GitHub at a pinned tag,
    so a release that forgets to bump them silently hands people the previous
    version — which is how v0.1.7 kept being installed after v0.1.8 shipped.
    """
    files = sorted((ROOT / "notebooks").glob("*.ipynb")) + [ROOT / "README.md"]
    pins = []
    for f in files:
        if not f.exists():
            continue
        for m in re.finditer(r"cvenv@v(\d+\.\d+\.\d+)", f.read_text()):
            pins.append((f.name, m.group(1)))
    # A file may pin the same version several times; collapse those to one row,
    # while still surfacing a file whose own pins disagree with each other.
    return sorted(set(pins))


def tag_at_head() -> str | None:
    """A tag pointing at HEAD, but only when the tree is clean.

    Between tagging one release and committing the next bump, HEAD still
    carries the previous tag while the files already say the new version.
    Flagging that is noise — it is exactly what being mid-release looks like.
    CI always checks out clean, so the tag comparison still runs where it
    matters, and passing a tag explicitly always forces the check.
    """
    try:
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        if dirty:
            return None
        out = subprocess.run(["git", "describe", "--tags", "--exact-match"],
                             cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip() or None
    except Exception:
        return None


def main(argv: list[str]) -> int:
    pyproj, dunder = pyproject_version(), dunder_version()
    tag = argv[1] if len(argv) > 1 else tag_at_head()
    pins = install_pins()

    width = max([18] + [len(n) for n, _ in pins])
    print(f"{'pyproject.toml':<{width}}  {pyproj}")
    print(f"{'cvenv/__init__.py':<{width}}  {dunder}")
    for name, ver in pins:
        print(f"{name:<{width}}  {ver}")
    if tag:
        print(f"{'git tag':<{width}}  {tag}")

    problems = []
    if pyproj != dunder:
        problems.append(f"pyproject.toml says {pyproj}, "
                        f"cvenv/__init__.py says {dunder}")
    for name, ver in pins:
        if ver != pyproj:
            problems.append(f"{name} tells people to install cvenv@v{ver}, "
                            f"but the code is {pyproj}")
    if tag and tag.lstrip("v") != pyproj:
        problems.append(f"tag {tag} does not match the code version {pyproj}")

    if problems:
        print()
        for p in problems:
            print(f"MISMATCH: {p}")
        print("\nBump every place before tagging.")
        return 1

    print("\nversions agree" + (" (including the tag)" if tag else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
