"""Reference-only bridge to Compactification-Theory Distill research v0.4.

Run with the reviewed Distill checkout on PYTHONPATH. Native grade_research
performs its own packet validation and exact certificate check.
"""

from __future__ import annotations

import hashlib
import json


def canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                     allow_nan=False).encode("ascii")
    if len(raw) >= 1_048_576:
        raise ValueError("Distill canonical byte cap")
    return hashlib.sha256(raw).hexdigest()


def reference_record(packet, response, *, adapters=None):
    from Distill.research_oracle import grade_research

    if packet.get("library_version") != "0.4.0" or "research" not in packet:
        raise ValueError("expected research v0.4 packet")
    if response.get("schema_version") != "distill.response.v2":
        raise ValueError("expected native JSON decision response")
    receipt = grade_research(packet, response, adapters=adapters)
    if (receipt.get("packet_id") != packet["packet_id"]
            or receipt.get("response_sha256") != canonical_digest(response)
            or receipt.get("custody") != "reference_only"
            or receipt.get("training_admission") is not False):
        raise ValueError("invalid native receipt binding or custody")
    answers = {a["question_id"]: a for a in response["answers"]}
    checks = {c["question_id"]: c for c in receipt["checks"]}
    decision = answers.get("case.solve")
    check = checks.get("case.solve")
    if (not decision or decision["status"] != "answered"
            or not check or check["exact_check"] != "pass"):
        raise ValueError("missing or unverified native decision certificate")
    if packet["public_case"].get("classification") != "public_development":
        raise ValueError("only reviewed public development cases")
    return {
        "schema": "datagoblin.distill_reference.v1",
        "root_id": packet["research"]["content_sha256"],
        "packet_id": packet["packet_id"],
        "source": packet["source"],
        "response_sha256": receipt["response_sha256"],
        "receipt_sha256": canonical_digest(receipt),
        "decision": decision["fields"]["answer"],
        "decision_fields": decision["fields"],
        "review_questions": [a for q, a in answers.items() if q != "case.solve"],
        "explanation_review": "pending",
        "provider_identity": "claimed_only",
        "custody": "reference_only",
        "training_admission": False,
    }
