from __future__ import annotations

import sqlite3

TABLES = (
    "learner_snapshots",
    "capability_observations",
    "failure_clusters",
    "hypotheses",
    "recipes",
    "constitutional_decisions",
    "builder_tickets",
    "builds",
    "experiments",
    "metrics",
    "artifacts",
    "membrane_decisions",
    "research_episodes",
)


def migrate(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    for table in TABLES:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                record_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS experiment_annotations (
            annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL,
            annotation_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES experiments(record_id)
        )
        """
    )
    connection.execute("INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (1, datetime('now'))")
    connection.commit()
