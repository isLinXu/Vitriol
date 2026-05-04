#!/usr/bin/env python3
# Generated: 2026-03-31T00:00Z
# Purpose: Mount large external test assets into tests/ via symlink or copy

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional


DEFAULT_SUBDIRS = ("offload", "offload_inference", "output")


def _env_assets_dir() -> Optional[Path]:
    raw = os.getenv("VITRIOL_TEST_ASSETS_DIR")
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p


def _remove_path(p: Path) -> None:
    if not p.exists() and not p.is_symlink():
        return
    if p.is_symlink() or p.is_file():
        p.unlink(missing_ok=True)
        return
    shutil.rmtree(p)


def _symlink_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if copy:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return

    # Prefer symlink
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except (OSError, NotImplementedError):
        # Fallback to copy if symlink fails (e.g. Windows without privileges)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mount large external test assets into tests/ (default: symlink; optionally: copy)."
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        default=None,
        help="Local test assets root directory (defaults to env var VITRIOL_TEST_ASSETS_DIR).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Use copy instead of symlink (not recommended; large).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If the target path under tests/ already exists, delete it first and recreate.",
    )
    parser.add_argument(
        "--subdir",
        action="append",
        default=[],
        help="Mount only specified subdirectories; can be repeated (default: offload/offload_inference/output).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    tests_dir = repo_root / "tests"

    assets_dir = Path(args.assets_dir).expanduser() if args.assets_dir else _env_assets_dir()
    if assets_dir is None:
        print("[ERROR] Test assets directory is not set. Set VITRIOL_TEST_ASSETS_DIR or pass --assets-dir.")
        return 2
    if not assets_dir.exists():
        print(f"[ERROR] Test assets directory does not exist: {assets_dir}")
        return 2

    subdirs = tuple(args.subdir) if args.subdir else DEFAULT_SUBDIRS

    print(f"[INFO] Repo: {repo_root}")
    print(f"[INFO] Tests: {tests_dir}")
    print(f"[INFO] Assets: {assets_dir}")
    print(f"[INFO] Mode: {'copy' if args.copy else 'symlink (fallback to copy)'}")

    mounted_any = False
    for name in subdirs:
        src = assets_dir / name
        dst = tests_dir / name
        if not src.exists():
            print(f"[WARN] Missing asset subdir, skipping: {src}")
            continue

        if (dst.exists() or dst.is_symlink()) and args.force:
            print(f"[INFO] --force: removing existing target: {dst}")
            _remove_path(dst)
        elif dst.exists() or dst.is_symlink():
            print(f"[INFO] Target already exists, skipping: {dst} (use --force to overwrite)")
            mounted_any = True
            continue

        _symlink_or_copy(src, dst, copy=bool(args.copy))
        print(f"[OK] {dst} -> {src}")
        mounted_any = True

    if not mounted_any:
        print("[WARN] 未挂载任何资产。请检查 assets 目录结构或使用 --subdir 指定。")
        return 1

    print("\nNext:")
    print('  pytest -m "not slow and not network" tests/ --ignore=tests/integration -v')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
