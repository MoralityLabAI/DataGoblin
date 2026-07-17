from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_file
from datagoblin.serialization import load_structured


@dataclass(frozen=True)
class UpstreamStatus:
    logical_name: str
    discovered_root: str | None
    branch: str | None
    commit: str | None
    available: bool
    read_only: bool
    note: str

    def to_jsonable(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class UpstreamEntry:
    logical_name: str
    local_root: Path
    expected_branch: str | None
    expected_commit: str | None
    include_globs: tuple[str, ...]
    exclude_globs: tuple[str, ...]
    adapter_type: str
    read_only: bool

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> UpstreamEntry:
        return cls(
            logical_name=str(data["logical_name"]),
            local_root=Path(str(data["local_root"])),
            expected_branch=str(data["expected_branch"]) if data.get("expected_branch") else None,
            expected_commit=str(data["expected_commit"]) if data.get("expected_commit") else None,
            include_globs=tuple(str(item) for item in data.get("include_globs", ())),
            exclude_globs=tuple(str(item) for item in data.get("exclude_globs", ())),
            adapter_type=str(data["adapter_type"]),
            read_only=bool(data.get("read_only", True)),
        )


def _git_root(candidate: Path) -> Path | None:
    if not candidate.exists():
        return None
    if (candidate / ".git").exists():
        return candidate.resolve()
    roots = sorted(path.parent for path in candidate.glob("**/.git") if path.is_dir())
    return roots[0].resolve() if roots else None


def _git_value(root: Path, *args: str) -> str | None:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, shell=False, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _candidate_paths(logical_name: str, repository_root: Path) -> list[Path]:
    if logical_name == "small_control_harness":
        env = os.environ.get("SMALL_CONTROL_HARNESS_PATH")
        values = [
            env,
            repository_root.parent / "SmallControlHarness",
            repository_root.parent / "Control-Harness",
            Path("C:/projects/SmallControlHarness"),
            Path("C:/projects/Control-Harness"),
        ]
    else:
        env = os.environ.get("HYBRID_TRM_LDT_PATH")
        values = [
            env,
            repository_root.parent / "HybridTRMLDT",
            repository_root.parent / "Hybrid-LDT-TRM",
            Path("C:/projects/HybridTRMLDT"),
        ]
    return [Path(value) for value in values if value]


def discover_upstreams(repository_root: Path | None = None) -> tuple[UpstreamStatus, ...]:
    repository_root = (repository_root or Path.cwd()).resolve()
    statuses = []
    for logical_name in ("small_control_harness", "hybrid_trm_ldt"):
        root = next(
            (found for candidate in _candidate_paths(logical_name, repository_root) if (found := _git_root(candidate))),
            None,
        )
        if root:
            statuses.append(
                UpstreamStatus(
                    logical_name,
                    str(root),
                    _git_value(root, "branch", "--show-current"),
                    _git_value(root, "rev-parse", "HEAD"),
                    True,
                    True,
                    "local read-only reference detected",
                )
            )
        else:
            statuses.append(
                UpstreamStatus(
                    logical_name,
                    None,
                    None,
                    None,
                    False,
                    True,
                    "not found; adapter remains optional",
                )
            )
    return tuple(statuses)


def _excluded(relative: str, patterns: Iterable[str]) -> bool:
    normalized = relative.replace("\\", "/")
    path = Path(normalized)
    blocked_parts = {".git", "__pycache__", "wandb", "checkpoints", "models", "private", "secrets"}
    blocked_suffixes = {".ckpt", ".pt", ".pth", ".safetensors", ".bin"}
    secret_name = path.name == ".env" or path.name.startswith(".env.")
    return (
        bool(blocked_parts.intersection(path.parts))
        or path.suffix.lower() in blocked_suffixes
        or secret_name
        or any(fnmatch.fnmatch(normalized, pattern) for pattern in patterns)
    )


class UpstreamRegistry:
    def __init__(self, entries: tuple[UpstreamEntry, ...]) -> None:
        self.entries = entries

    @classmethod
    def load(cls, path: Path) -> UpstreamRegistry:
        value = load_structured(path)
        return cls(tuple(UpstreamEntry.from_jsonable(item) for item in value["upstreams"]))

    def status(self) -> list[dict[str, Any]]:
        output = []
        for entry in self.entries:
            root = _git_root(entry.local_root)
            output.append(
                {
                    "logical_name": entry.logical_name,
                    "available": bool(root),
                    "root": str(root) if root else None,
                    "branch": _git_value(root, "branch", "--show-current") if root else None,
                    "commit": _git_value(root, "rev-parse", "HEAD") if root else None,
                    "expected_branch": entry.expected_branch,
                    "expected_commit": entry.expected_commit,
                    "read_only": entry.read_only,
                    "adapter_type": entry.adapter_type,
                }
            )
        return output

    def ingest(self, output_root: Path, *, mode: str = "copy") -> dict[str, Any]:
        if mode not in {"copy", "symlink"}:
            raise ValueError("ingestion mode must be copy or symlink")
        receipt: dict[str, Any] = {"mode_requested": mode, "upstreams": {}}
        for entry in self.entries:
            if not entry.read_only:
                raise ValueError(f"upstream must be read-only: {entry.logical_name}")
            root = _git_root(entry.local_root)
            if root is None:
                receipt["upstreams"][entry.logical_name] = {"available": False, "files": {}}
                continue
            selected: dict[str, Path] = {}
            for pattern in entry.include_globs:
                for source in root.glob(pattern):
                    if not source.is_file() or source.is_symlink():
                        continue
                    relative = source.relative_to(root).as_posix()
                    if source.stat().st_size <= 16 * 1024 * 1024 and not _excluded(
                        relative, entry.exclude_globs
                    ):
                        selected[relative] = source
            files = {}
            actual_modes = set()
            destination_root = output_root / entry.logical_name
            for relative, source in sorted(selected.items()):
                destination = destination_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                actual = mode
                if mode == "symlink":
                    try:
                        destination.symlink_to(source)
                    except OSError:
                        shutil.copy2(source, destination)
                        actual = "copy_fallback"
                else:
                    shutil.copy2(source, destination)
                actual_modes.add(actual)
                files[relative] = {"sha256": sha256_file(source), "mode": actual}
            receipt["upstreams"][entry.logical_name] = {
                "available": True,
                "root": str(root),
                "branch": _git_value(root, "branch", "--show-current"),
                "commit": _git_value(root, "rev-parse", "HEAD"),
                "actual_modes": sorted(actual_modes),
                "files": files,
            }
        return receipt
