"""Component interface + registry.

Every installable thing (pytorch3d, mast3r, sam2, the science stack, opengl) is a
``Component`` with a uniform, self-describing shape so it can be used on its own
in a notebook cell or composed via the CLI:

    from cvenv import get_component, setup
    get_component("pytorch3d").install(wheel_url=MY_WHEEL)
    get_component("pytorch3d").verify()
    setup(["pytorch3d", "mast3r", "sam2"])
"""

from __future__ import annotations

from typing import Dict, List, Optional


class Component:
    """Base class for an installable component.

    Subclasses set ``name`` / ``summary`` / ``teaching_note`` and implement
    ``is_installed`` / ``_install`` / ``verify``.
    """

    name: str = ""
    summary: str = ""
    # A short "why this install is tricky" note — surfaced by the CLI and reused
    # as lecture material.
    teaching_note: str = ""
    # Names of components that should be installed first.
    requires: List[str] = []

    def is_installed(self) -> bool:
        """Cheap probe (usually an import test). Override."""
        raise NotImplementedError

    def _install(self, platform: Optional[str] = None, **opts) -> None:
        """Do the install. Override."""
        raise NotImplementedError

    def install(self, platform: Optional[str] = None, force: bool = False, **opts) -> None:
        """Idempotent install: skips when already present unless ``force``."""
        if not force and self.is_installed():
            print(f"✅ {self.name}: already installed — skipping.")
            return
        print(f"▶️  {self.name}: installing…")
        self._install(platform=platform, **opts)

    def verify(self) -> bool:
        """Post-install sanity check. Override. Return True / raise on failure."""
        raise NotImplementedError

    def build_wheel(self, out_dir=None, platform=None, **opts) -> str:
        """Build a reusable wheel WITHOUT installing it, returning its path.

        Only components whose install compiles from source (e.g. pytorch3d)
        support this. Override there; the default rejects it."""
        raise NotImplementedError(
            f"{self.name} has no wheel to build (it's not a compiled-from-source "
            "component).")


REGISTRY: Dict[str, Component] = {}


def register(component: Component) -> Component:
    REGISTRY[component.name] = component
    return component


def get_component(name: str) -> Component:
    _load_components()
    if name not in REGISTRY:
        raise KeyError(f"Unknown component: {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def list_components() -> List[Component]:
    _load_components()
    return [REGISTRY[k] for k in sorted(REGISTRY)]


def _resolve_order(names: List[str]) -> List[str]:
    """Expand `requires` and de-duplicate, dependencies first (simple DFS)."""
    _load_components()
    ordered: List[str] = []

    def visit(n: str, stack: tuple):
        if n in ordered:
            return
        if n in stack:
            raise ValueError(f"Circular dependency involving {n!r}")
        comp = REGISTRY.get(n)
        if comp is None:
            raise KeyError(f"Unknown component: {n!r}")
        for dep in comp.requires:
            visit(dep, stack + (n,))
        ordered.append(n)

    for name in names:
        visit(name, ())
    return ordered


def setup(names, platform: Optional[str] = None, force: bool = False, **opts) -> None:
    """Install several components (dependencies first)."""
    if isinstance(names, str):
        names = [names]
    for name in _resolve_order(list(names)):
        get_component(name).install(platform=platform, force=force, **opts)


_loaded = False


def _load_components():
    """Import the component modules once to populate the registry."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    from . import components  # noqa: F401  (import side effect: registration)
