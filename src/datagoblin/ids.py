from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .hashing import sha256_json


def content_id(prefix: str, value: Any, length: int = 16) -> str:
    return f"{prefix}_{sha256_json(value)[:length]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
