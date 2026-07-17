from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_json
from datagoblin.ids import utc_now
from datagoblin.serialization import canonical_json, to_jsonable

from .migrations import TABLES, migrate


class ExperimentLedger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        migrate(self.connection)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ExperimentLedger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def insert(self, table: str, record_id: str, payload: Any) -> str:
        if table not in TABLES:
            raise ValueError(f"unknown ledger table: {table}")
        jsonable = to_jsonable(payload)
        digest = sha256_json(jsonable)
        encoded = canonical_json(jsonable)
        row = self.connection.execute(f"SELECT content_hash FROM {table} WHERE record_id = ?", (record_id,)).fetchone()
        if row:
            if row["content_hash"] != digest:
                raise ValueError(f"immutable record differs: {table}/{record_id}")
            return digest
        self.connection.execute(
            f"INSERT INTO {table}(record_id, content_hash, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (record_id, digest, encoded, utc_now()),
        )
        self.connection.commit()
        return digest

    def get(self, table: str, record_id: str) -> dict[str, Any] | None:
        if table not in TABLES:
            raise ValueError(f"unknown ledger table: {table}")
        row = self.connection.execute(f"SELECT payload_json FROM {table} WHERE record_id = ?", (record_id,)).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def annotate_experiment(self, experiment_id: str, annotation: dict[str, Any]) -> None:
        if self.get("experiments", experiment_id) is None:
            raise KeyError(experiment_id)
        self.connection.execute(
            "INSERT INTO experiment_annotations(experiment_id, annotation_json, created_at) VALUES (?, ?, ?)",
            (experiment_id, canonical_json(annotation), utc_now()),
        )
        self.connection.commit()

    def analogous_experiments(self, capability: str, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT payload_json FROM experiments WHERE payload_json LIKE ? ORDER BY created_at DESC LIMIT ?",
            (f"%{capability}%", limit),
        ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]
