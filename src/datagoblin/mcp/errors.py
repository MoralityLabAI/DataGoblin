from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class McpToolError(Exception):
    """A tool failure with a stable machine-readable code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# Directories under the root that tools may read from, and the only directory they may write to.
READ_DIRS = ("examples", "runs", "configs")
WRITE_DIR = "runs"


class Sandbox:
    """Resolves tool paths against a root and enforces the read and write directories."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _resolve(self, value: Any, field: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise McpToolError("invalid_arguments", f"{field} must be a non-empty path string")
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        return candidate.resolve()

    def rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()

    def read_path(self, value: Any, field: str, *, must_exist: bool = True) -> Path:
        path = self._resolve(value, field)
        if not any(path.is_relative_to(self.root / name) for name in READ_DIRS):
            raise McpToolError(
                "path_not_allowed", f"{field} must resolve under one of: {', '.join(READ_DIRS)} (got {value})"
            )
        if must_exist and not path.exists():
            raise McpToolError("not_found", f"{field} does not exist: {self.rel(path)}")
        return path

    def write_path(self, value: Any, field: str) -> Path:
        path = self._resolve(value, field)
        if not path.is_relative_to(self.root / WRITE_DIR) or path == self.root / WRITE_DIR:
            raise McpToolError("path_not_allowed", f"{field} must resolve under {WRITE_DIR}/ (got {value})")
        return path


def arg_str(args: dict[str, Any], key: str, default: str | None = None) -> str:
    value = args.get(key, default)
    if not isinstance(value, str) or not value:
        raise McpToolError("invalid_arguments", f"{key} is required and must be a non-empty string")
    return value


def arg_int(args: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    value = args.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise McpToolError("invalid_arguments", f"{key} must be an integer in [{low}, {high}]")
    return value


def arg_choice(args: dict[str, Any], key: str, choices: tuple[str, ...], default: str | None = None) -> str:
    value = args.get(key, default)
    if value not in choices:
        raise McpToolError("invalid_arguments", f"{key} must be one of: {', '.join(choices)}")
    return str(value)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise McpToolError("invalid_input", f"{path.name} is not valid JSON: {exc}") from exc
