"""Independent declarative reference semantics for owner/scope/temporal/revocation state.

Designed cleanly from first principles as an immutable event-sourced relation
interpreter, independent of datagoblin.research_v2.suite.persona_memory and
scripts/research_v2/jinn_beast_persona_exp._reconstruct_store.

Semantics (declarative per-key causal projection):
  Given an unordered set/sequence of events E = {(actor, scope, revision, op, key, value?)}:
  1. Filter E by predicate P(e):
       - owner binding (if enforce_owner): e.actor == owner
       - scope binding (if enforce_scope): e.scope == scope
       - temporal horizon (if enforce_temporal): e.revision <= at_revision
       - key allowlist (if allowed_keys is not None): e.key in allowed_keys
     Every excluded event is categorized into an explicit rejection ledger.
  2. Partition admitted events by key k.
  3. For each key k, order its admitted history by causal revision clock:
       - If enforce_temporal is True: sort ascending by (revision, arrival_index).
       - If enforce_temporal is False: retain raw arrival_index order only.
  4. Resolve effective key state via explicit tombstone fold:
       - op == "set": active value becomes (revision, str(value))
       - op == "revoke": if enforce_revocation is True, state becomes TOMBSTONE(revision)
  5. Project active store {k: val} containing ONLY keys whose final state is live
     (not TOMBSTONE).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuditRecord:
    arrival_index: int
    actor: str
    scope: str
    revision: int
    op: str
    key: str
    value: str | None


@dataclass(frozen=True)
class StoreProjection:
    active: dict[str, str]
    tombstones: dict[str, int]
    admitted_count: int
    rejected_cross_owner: int
    rejected_cross_scope: int
    rejected_future_revision: int
    rejected_disallowed_key: int
    rejected_unknown_op: int


def _parse_event(raw: dict[str, Any], idx: int) -> AuditRecord:
    if not isinstance(raw, dict):
        raise ValueError("event must be a mapping")
    actor = str(raw["actor"])
    scope = str(raw["scope"])
    rev = raw["revision"]
    if type(rev) is not int or rev < 0:
        raise ValueError("event revision must be a non-negative integer")
    op = str(raw["op"])
    key = str(raw["key"])
    val = None if raw.get("value") is None else str(raw["value"])
    return AuditRecord(
        arrival_index=idx,
        actor=actor,
        scope=scope,
        revision=rev,
        op=op,
        key=key,
        value=val,
    )


def project_persona_store(
    events: list[dict[str, Any]],
    *,
    owner: str,
    scope: str,
    at_revision: int,
    allowed_keys: list[str] | set[str] | tuple[str, ...] | None = None,
    enforce_owner: bool = True,
    enforce_temporal: bool = True,
    enforce_scope: bool = True,
    enforce_revocation: bool = True,
) -> StoreProjection:
    """Independently project the active preference store and rejection audit ledger."""
    if type(at_revision) is not int or at_revision < 0:
        raise ValueError("at_revision must be a non-negative integer")
    allow = None if allowed_keys is None else frozenset(str(k) for k in allowed_keys)
    records = [_parse_event(ev, i) for i, ev in enumerate(events)]

    per_key: dict[str, list[AuditRecord]] = {}
    rej_owner = 0
    rej_scope = 0
    rej_future = 0
    rej_key = 0
    rej_op = 0
    admitted = 0

    for rec in records:
        if rec.op not in {"set", "revoke"}:
            rej_op += 1
            continue
        if enforce_owner and rec.actor != owner:
            rej_owner += 1
            continue
        if enforce_scope and rec.scope != scope:
            rej_scope += 1
            continue
        if enforce_temporal and rec.revision > at_revision:
            rej_future += 1
            continue
        if allow is not None and rec.key not in allow:
            rej_key += 1
            continue
        admitted += 1
        per_key.setdefault(rec.key, []).append(rec)

    active: dict[str, str] = {}
    tombstones: dict[str, int] = {}

    for key, history in sorted(per_key.items()):
        ordered = (
            sorted(history, key=lambda r: (r.revision, r.arrival_index))
            if enforce_temporal
            else sorted(history, key=lambda r: r.arrival_index)
        )
        live_value: str | None = None
        tombstone_rev: int | None = None
        for rec in ordered:
            if rec.op == "set":
                if rec.value is None:
                    raise ValueError(f"set event for key {key!r} missing value")
                live_value = rec.value
                tombstone_rev = None
            elif rec.op == "revoke" and enforce_revocation:
                live_value = None
                tombstone_rev = rec.revision
        if live_value is not None:
            active[key] = live_value
        elif tombstone_rev is not None:
            tombstones[key] = tombstone_rev

    return StoreProjection(
        active=active,
        tombstones=tombstones,
        admitted_count=admitted,
        rejected_cross_owner=rej_owner,
        rejected_cross_scope=rej_scope,
        rejected_future_revision=rej_future,
        rejected_disallowed_key=rej_key,
        rejected_unknown_op=rej_op,
    )


def reference_persona_memory(world: dict[str, Any]) -> dict[str, str]:
    """Drop-in independent reference resolver matching Research v2/v2.1 persona worlds."""
    proj = project_persona_store(
        world["events"],
        owner=world["owner"],
        scope=world["scope"],
        at_revision=world["at_revision"],
        allowed_keys=world.get("allowed_keys"),
    )
    return proj.active
