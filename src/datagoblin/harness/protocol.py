from __future__ import annotations

from typing import Any, Protocol


class Harness(Protocol):
    def run(self, seed: int) -> dict[str, Any]: ...
