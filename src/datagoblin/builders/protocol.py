from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from datagoblin.domain.builders import BuilderTicket


@dataclass(frozen=True)
class BuildResult:
    accepted: bool
    build_root: Path
    outputs: tuple[Path, ...]
    receipt: dict[str, Any]


class BuilderBackend(Protocol):
    def build(self, ticket: BuilderTicket, build_root: Path) -> BuildResult: ...
