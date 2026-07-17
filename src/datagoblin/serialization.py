from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


def to_jsonable(value: Any) -> Any:
    """Recursively convert supported records to deterministic JSON values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported serialization type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def pretty_json(value: Any) -> str:
    return json.dumps(to_jsonable(value), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def load_structured(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(text)
    else:
        raise ValueError(f"unsupported structured file: {path}")
    if not isinstance(value, dict):
        raise ValueError(f"expected object at {path}")
    return value


def dump_structured(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = to_jsonable(value)
    if path.suffix.lower() == ".json":
        path.write_text(pretty_json(data), encoding="utf-8")
    elif path.suffix.lower() in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        raise ValueError(f"unsupported structured file: {path}")


def require_keys(data: Mapping[str, Any], *names: str) -> None:
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
