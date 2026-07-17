# ADR 0003: Ledger and content-addressed artifacts

Status: accepted

SQLite stores typed record envelopes and relationships. Bytes live in a SHA-256 content-addressed artifact store; run manifests point to those artifacts instead of duplicating them. Canonical JSON is used before hashing.

Evaluated experiment records are immutable. Later commentary is append-only. The ledger is tamper-evident within a local-process threat model, not a cryptographic transparency service.

