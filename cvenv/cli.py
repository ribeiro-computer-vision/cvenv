"""cvenv command-line interface.

    cvenv list
    cvenv platform
    cvenv install pytorch3d mast3r sam2 [--wheel-url URL] [--checkpoint-dir DIR] [--from-source] [--force]
    cvenv verify pytorch3d mast3r sam2
"""

from __future__ import annotations

import argparse
import sys

from .platform import PlatformManager
from .base import get_component, list_components, setup


def _cmd_list(args) -> int:
    print("Available cvenv components:\n")
    for comp in list_components():
        print(f"  {comp.name:<10} {comp.summary}")
        if args.verbose and comp.teaching_note:
            print(f"             ↳ {comp.teaching_note}\n")
    return 0


def _cmd_platform(args) -> int:
    pm = PlatformManager()
    print(f"platform   : {pm.platform}")
    print(f"local_path : {pm.local_path}")
    print(f"python     : {sys.version.split()[0]} ({sys.executable})")
    return 0


def _install_opts(args) -> dict:
    opts = {}
    if args.wheel_url:
        opts["wheel_url"] = args.wheel_url
    if args.checkpoint_dir:
        opts["checkpoint_dir"] = args.checkpoint_dir
    if args.from_source:
        opts["from_source"] = True
    if args.wheel_out_dir:
        opts["wheel_out_dir"] = args.wheel_out_dir
    return opts


def _cmd_install(args) -> int:
    pm = PlatformManager()
    print(f"Platform: {pm.platform}\n")
    numpy_before = _numpy_version()
    try:
        setup(args.components, platform=pm.platform, force=args.force, **_install_opts(args))
    except Exception as e:
        print(f"\n❌ install failed: {e}")
        return 1
    _warn_if_numpy_changed(numpy_before)
    print("\n✅ install step complete. Run 'cvenv verify ...' to sanity-check.")
    return 0


def _cmd_verify(args) -> int:
    rc = 0
    for name in args.components:
        try:
            get_component(name).verify()
        except Exception as e:
            print(f"❌ {name}: {e}")
            rc = 1
    return rc


def _cmd_build_wheel(args) -> int:
    pm = PlatformManager()
    print(f"Platform: {pm.platform}\n")
    rc = 0
    for name in (args.components or ["pytorch3d"]):
        try:
            path = get_component(name).build_wheel(
                out_dir=args.wheel_out_dir, platform=pm.platform, ref=args.ref)
            print(f"✅ {name}: {path}")
        except Exception as e:
            print(f"❌ {name}: {e}")
            rc = 1
    return rc


def _numpy_version():
    try:
        import numpy
        return numpy.__version__
    except Exception:
        return None


def _warn_if_numpy_changed(before):
    after = _numpy_version()
    if before and after and before != after:
        print("\n" + "=" * 68)
        print(f"⚠️  numpy changed in this session ({before} -> {after}).")
        print("   RESTART the kernel/runtime before importing numpy-dependent")
        print("   packages, or you'll hit 'numpy.dtype size changed' ABI errors.")
        print("=" * 68)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cvenv",
                                description="À-la-carte CV/ML environment installer.")
    sub = p.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("list", help="list available components")
    pl.add_argument("-v", "--verbose", action="store_true", help="show teaching notes")
    pl.set_defaults(func=_cmd_list)

    pp = sub.add_parser("platform", help="show detected platform")
    pp.set_defaults(func=_cmd_platform)

    pi = sub.add_parser("install", help="install one or more components")
    pi.add_argument("components", nargs="+")
    pi.add_argument("--wheel-url", help="prebuilt wheel URL (pytorch3d)")
    pi.add_argument("--checkpoint-dir", help="local dir holding a pre-staged checkpoint")
    pi.add_argument("--from-source", action="store_true", help="force source build (pytorch3d)")
    pi.add_argument("--wheel-out-dir",
                    help="dir to save a source-built pytorch3d wheel (default: a "
                         "persistent per-platform dir, e.g. Drive on Colab)")
    pi.add_argument("--force", action="store_true", help="reinstall even if present")
    pi.set_defaults(func=_cmd_install)

    pv = sub.add_parser("verify", help="verify one or more components")
    pv.add_argument("components", nargs="+")
    pv.set_defaults(func=_cmd_verify)

    pw = sub.add_parser("build-wheel",
                        help="build a reusable wheel WITHOUT installing (pytorch3d)")
    pw.add_argument("components", nargs="*", default=["pytorch3d"],
                    help="component(s) to build (default: pytorch3d)")
    pw.add_argument("--wheel-out-dir",
                    help="output dir (default: persistent per-platform dir)")
    pw.add_argument("--ref", default="stable",
                    help="git ref/branch/tag to build (pytorch3d; default: stable)")
    pw.set_defaults(func=_cmd_build_wheel)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
