# Experiment ledger

The ledger uses SQLite with foreign keys enabled and one envelope table for each public record family: snapshots, observations, failures, hypotheses, recipes, constitutional decisions, tickets, builds, experiments, metrics, artifacts, membrane decisions, and research episodes.

Canonical JSON is hashed before insertion. Re-inserting an identical record is idempotent; changing a record at the same ID fails. Evaluated experiments therefore cannot be silently rewritten. Later notes use an append-only annotations table.

Artifact bytes live under `artifacts/sha256/<prefix>/<hash>`. Store reads recompute the hash. Model weights, checkpoints, secret-like files, and oversized artifacts are rejected. A run manifest records artifact references and the ledger path rather than cloning all underlying build products.

This is local tamper evidence and immutability discipline, not a distributed transparency log. Filesystem or database administrators remain within the threat model.

