"""Safely report or remove reproducible DataGoblin-local cache files."""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIRECTORY_NAMES = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
}


def _within_repository(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return False
    return path.resolve() != REPOSITORY_ROOT.resolve()


def discover_targets(*, include_runs: bool) -> tuple[Path, ...]:
    targets: set[Path] = set()
    for current, directories, files in os.walk(REPOSITORY_ROOT, topdown=True, followlinks=False):
        current_path = Path(current)
        directories[:] = [name for name in directories if name != ".git"]
        for name in tuple(directories):
            path = current_path / name
            if name in CACHE_DIRECTORY_NAMES or name.endswith(".egg-info"):
                targets.add(path)
                directories.remove(name)
        for name in files:
            if name.endswith((".pyc", ".pyo")):
                targets.add(current_path / name)
    if include_runs and (REPOSITORY_ROOT / "runs").exists():
        targets.add(REPOSITORY_ROOT / "runs")
    return tuple(sorted(targets, key=lambda path: (len(path.parts), path.as_posix()), reverse=True))


def target_size(path: Path) -> int:
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size
    return sum(child.lstat().st_size for child in path.rglob("*") if child.is_file() and not child.is_symlink())


def clear_targets(targets: Sequence[Path]) -> int:
    reclaimed = 0
    for path in targets:
        if not _within_repository(path):
            raise ValueError(f"refusing path outside repository: {path}")
        reclaimed += target_size(path)
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.exists():
            shutil.rmtree(path)
    return reclaimed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Remove discovered caches; default is report-only.")
    parser.add_argument(
        "--include-runs",
        action="store_true",
        help="Also remove runs/. This is opt-in because run ledgers and reports may be valuable.",
    )
    args = parser.parse_args(argv)
    targets = discover_targets(include_runs=args.include_runs)
    size = sum(target_size(path) for path in targets)
    for path in reversed(targets):
        print(path.relative_to(REPOSITORY_ROOT).as_posix())
    if args.apply:
        clear_targets(targets)
        print(f"removed {len(targets)} targets; reclaimed approximately {size} bytes")
    else:
        print(f"would remove {len(targets)} targets; approximately {size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
