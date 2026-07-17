from __future__ import annotations

from pathlib import Path
from typing import Any

from datagoblin.config import load_config
from datagoblin.hashing import sha256_json


class Charter:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.version = str(data["constitution_version"])
        self.hash = sha256_json(data)

    @classmethod
    def load(cls, path: Path | None = None) -> Charter:
        return cls(load_config(path, "constitution.yaml"))
