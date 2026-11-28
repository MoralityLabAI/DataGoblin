"""Audit a steward-reviewed candidate pack with the exact Aleph tokenizer.

This is a pre-packing unique-content census, not a trainer or rights oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


PIN = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
HEX = re.compile(r"^[0-9a-f]{64}$")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_json(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"review/custody manifest hash mismatch: {path.name}")
    return json.loads(raw)


def count_rows(rows, approved, prior, encode, quotas):
    """Separate operator-approved eligibility from tokenizer measurements."""
    if approved.get("schema") != "dg.research20m.steward-approval.v1":
        raise ValueError("missing steward approval contract")
    if prior.get("schema") != "dg.research20m.prior-custody.v1":
        raise ValueError("missing prior custody contract")
    if not prior.get("prior_snapshot_digests") or not prior.get("reserved_family_digest"):
        raise ValueError("prior and reserved snapshot identity required")
    prior_roots = set(prior["roots"])
    prior_content = set(prior["content_sha256"])
    blocked_families = set(prior["reserved_families"])
    decisions = approved["records"]
    seen_roots, seen_hashes, lanes = set(), set(), Counter()
    rejections = Counter()
    count = 0
    for row in rows:
        root = row["canonical_root"]
        family = row["family"]
        lane = row["lane"]
        content = row["canonical_text"]
        if not all(isinstance(x, str) and x.strip() for x in (root, family, lane, content)):
            raise ValueError("empty or non-string record identity/content")
        h = digest(content.encode("utf-8"))
        approval = decisions.get(root)
        if not approval or approval.get("content_sha256") != h:
            rejections["unapproved_or_content_changed"] += 1
            continue
        required = ("rights_evidence", "lineage_evidence", "near_duplicate_review",
                    "split_review", "objective_review", "source_identity")
        if not all(isinstance(approval.get(k), str) and approval[k].strip() for k in required):
            rejections["incomplete_review"] += 1
            continue
        if (approval.get("decision") != "admit_candidate"
                or approval.get("split") != "train"
                or approval.get("family") != family
                or approval.get("lane") != lane
                or family in blocked_families):
            rejections["split_or_admission"] += 1
            continue
        if root in prior_roots or h in prior_content or root in seen_roots or h in seen_hashes:
            rejections["repeated_root_or_content"] += 1
            continue
        if lane not in quotas:
            raise ValueError("unknown accounting lane")
        # Count one canonical rendering per root. Do not credit prompt replays,
        # alternate views, padding, or special tokens inserted during packing.
        ids = encode(content)
        n = len(ids)
        if n == 0:
            rejections["empty_tokens"] += 1
            continue
        if any(not isinstance(i, int) or not 0 <= i < 49152 for i in ids):
            raise ValueError("token ID outside frozen vocabulary")
        if lanes[lane] + n > quotas[lane]:
            rejections["lane_budget_overflow"] += 1
            continue
        seen_roots.add(root)
        seen_hashes.add(h)
        lanes[lane] += n
        count += 1
    return {
        "schema": "dg.research20m.unique-token-census.v1",
        "training_ready": False,
        "accepted_candidate_roots": count,
        "measured_unique_content_tokens": sum(lanes.values()),
        "lane_tokens": {k: lanes[k] for k in quotas},
        "lane_shortfalls": {k: quotas[k] - lanes[k] for k in quotas},
        "rejected": dict(rejections),
        "target_unique_content_tokens": sum(quotas.values()),
        "counting_scope": "one reviewed canonical rendering per root, before packing; no padding or inserted special tokens; not a loss-mask or training-release receipt",
    }


def selftest():
    row = {"canonical_root": "r", "family": "f", "lane": "a", "canonical_text": "x y"}
    h = digest(b"x y")
    approved = {"schema": "dg.research20m.steward-approval.v1", "records": {
        "r": {"content_sha256": h, "family": "f", "lane": "a", "decision": "admit_candidate",
              "split": "train", **{k: "fixture" for k in (
                  "rights_evidence", "lineage_evidence", "near_duplicate_review",
                  "split_review", "objective_review", "source_identity")}}}}
    prior = {"schema": "dg.research20m.prior-custody.v1", "prior_snapshot_digests": ["fixture"],
             "reserved_family_digest": "fixture", "roots": [], "content_sha256": [],
             "reserved_families": []}
    encode = lambda text: [1, 2]
    result = count_rows([row, row], approved, prior, encode, {"a": 2})
    assert result["measured_unique_content_tokens"] == 2
    assert result["rejected"]["repeated_root_or_content"] == 1
    assert count_rows([row], approved, {**prior, "reserved_families": ["f"]},
                      encode, {"a": 2})["accepted_candidate_roots"] == 0
    assert count_rows([row], approved, {**prior, "content_sha256": [h]},
                      encode, {"a": 2})["accepted_candidate_roots"] == 0
    assert count_rows([{**row, "canonical_text": "changed"}], approved, prior,
                      encode, {"a": 2})["accepted_candidate_roots"] == 0
    assert count_rows([row], approved, prior, encode, {"a": 1})["accepted_candidate_roots"] == 0
    print("5 fixture checks passed (stub tokenizer only)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    for name in ("records", "approvals", "prior", "tokenizer", "out"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--approvals-sha256")
    p.add_argument("--prior-sha256")
    a = p.parse_args()
    if a.selftest:
        selftest()
        return
    if not all((a.records, a.approvals, a.prior, a.tokenizer, a.out,
                a.approvals_sha256, a.prior_sha256)):
        p.error("all paths and the two externally pinned review hashes required")
    if not HEX.fullmatch(a.approvals_sha256) or not HEX.fullmatch(a.prior_sha256):
        p.error("expected full SHA-256 review hashes")
    if digest(a.tokenizer.read_bytes()) != PIN:
        raise ValueError("wrong tokenizer bytes; use exact pilot tokenizer.json")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(a.tokenizer))
    if tok.get_vocab_size() != 49152:
        raise ValueError("wrong vocabulary")
    approved = checked_json(a.approvals, a.approvals_sha256)
    prior = checked_json(a.prior, a.prior_sha256)
    plan = json.loads(Path(__file__).with_name("sources.json").read_text(encoding="utf-8"))
    quotas = {r["id"]: r["target_unique_tokens"] for r in plan["lanes"]}
    with a.records.open(encoding="utf-8") as f:
        result = count_rows((json.loads(line) for line in f if line.strip()),
                            approved, prior, lambda s: tok.encode(s, add_special_tokens=False).ids, quotas)
    result.update({"tokenizer_sha256": PIN, "approvals_sha256": a.approvals_sha256,
                   "prior_sha256": a.prior_sha256})
    with a.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
