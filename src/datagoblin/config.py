from __future__ import annotations

from pathlib import Path
from typing import Any

from .serialization import load_structured


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path | None, default_name: str) -> dict[str, Any]:
    return load_structured(path or repository_root() / "configs" / default_name)
