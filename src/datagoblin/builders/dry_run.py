from __future__ import annotations

from pathlib import Path

from datagoblin.domain.builders import BuilderTicket
from datagoblin.hashing import sha256_json
from datagoblin.serialization import dump_structured

from .protocol import BuildResult


class DryRunBuilder:
    def build(self, ticket: BuilderTicket, build_root: Path) -> BuildResult:
        build_root.mkdir(parents=True, exist_ok=True)
        path = build_root / f"{ticket.ticket_id}.manifest.json"
        payload = {
            "mode": "dry_run",
            "ticket": ticket.to_jsonable(),
            "would_build": list(ticket.declared_outputs),
        }
        payload["receipt_hash"] = sha256_json(payload)
        dump_structured(path, payload)
        return BuildResult(True, build_root, (path,), payload)
