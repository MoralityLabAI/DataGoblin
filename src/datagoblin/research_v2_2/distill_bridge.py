"""Public research-decision records for a future Distill adapter.

The teacher provides concise rationale and proposed decisions. A separately
verified receipt determines eligibility; teacher prose is never a grade.
"""

from __future__ import annotations

import hashlib
import json


def _digest(record: dict) -> str:
    raw = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def admit_record(record: dict, *, public_packet: dict, verified_grade: dict) -> str:
    required = {"schema", "packet_id", "packet_sha256", "decision",
                "hypothesis", "control", "falsifier", "expected_observation",
                "budget", "evidence_ids", "failure_recovery", "rationale"}
    if set(record) != required or record["schema"] != "research-decision.v1":
        raise ValueError("invalid research-decision record")
    if public_packet.get("split") != "train":
        raise ValueError("only train packets may be admitted")
    if record["packet_id"] != public_packet["packet_id"] or record["packet_sha256"] != _digest(public_packet):
        raise ValueError("packet identity mismatch")
    text_fields = ("decision", "hypothesis", "control", "falsifier", "expected_observation",
                   "failure_recovery", "rationale")
    if any(not isinstance(record[k], str) or not 0 < len(record[k]) <= 2048 for k in text_fields):
        raise ValueError("missing or oversized research rationale")
    ids = record["evidence_ids"]
    if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
        raise ValueError("invalid evidence references")
    if not set(ids) <= set(public_packet["allowed_evidence_ids"]):
        raise ValueError("unregistered evidence reference")
    budget = record["budget"]
    if type(budget) is not int or not 0 <= budget <= public_packet["max_budget"]:
        raise ValueError("invalid proposed budget")
    sha = _digest(record)
    if (verified_grade.get("record_sha256") != sha
            or verified_grade.get("packet_sha256") != record["packet_sha256"]
            or verified_grade.get("exact_grade_pass") is not True):
        raise ValueError("independent grade missing or rejected")
    return sha
