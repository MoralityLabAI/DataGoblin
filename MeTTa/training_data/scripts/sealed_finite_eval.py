#!/usr/bin/env python3
"""Keyed finite-MeTTa confirmation harness with private reference answers.

A real evaluation requires an independently held, high-entropy key and
unseen model responses. Public CI runs exercise only a published dummy key.
This tool computes scoped reference answers without calling Hyperon.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import random
from pathlib import Path
from typing import Any

from finite_semantic_oracle import (
    UnsupportedSemantics,
    finite_match,
    parse_one,
    result_signature,
)

SCHEMA = "metta07-keyed-finite-eval-v1"
MAX_ROOTS = 256
MIN_KEY_BYTES = 32
QUERIES = (
    "(match &self (Group $agent blue) (Selected $agent))",
    "(match &self (Bridge $node $node) $node)",
    "(match &self (Status (Pair $node $tag) ready) ($tag $node))",
    "(match &self (Group $agent impossible) $agent)",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mac(key: bytes, label: str) -> bytes:
    return hmac.new(key, label.encode(), hashlib.sha256).digest()


def read_key(key_file: Path) -> bytes:
    key = key_file.read_bytes()
    if len(key) < MIN_KEY_BYTES:
        raise ValueError("Evaluation key must contain at least 32 random bytes")
    return key


def private_world(key: bytes, index: int) -> dict[str, Any]:
    """Make one new fact root from a private pseudorandom stream."""
    if len(key) < MIN_KEY_BYTES or not 0 <= index < MAX_ROOTS:
        raise ValueError("Invalid secret or world index")
    seed = int.from_bytes(mac(key, f"{SCHEMA}/rng/{index}"), "big")
    rng = random.Random(seed)
    nodes = [f"n{n:02d}" for n in range(10)]
    facts: list[str] = []
    for n, node in enumerate(nodes):
        color = rng.choice(("red", "blue", "green"))
        target = node if n % 4 == 0 else nodes[rng.randrange(len(nodes))]
        state = rng.choice(("ready", "waiting", "ready"))
        facts.extend((
            f"(Group {node} {color})",
            f"(Bridge {node} {target})",
            f"(Status (Pair {node} tag{n:02d}) {state})",
        ))
    source = "; Private-key generated finite-ground MeTTa evaluation\n"
    source += "\n".join(facts) + "\n"
    parsed = [parse_one(fact) for fact in facts]
    outputs = [finite_match(parsed, parse_one(query)) for query in QUERIES]
    return {
        "root_id": mac(key, f"{SCHEMA}/root/{index}").hex(),
        "source_prefix": source,
        "queries": QUERIES,
        "reference": outputs,
    }


def challenge_bundle(key: bytes, roots: int) -> tuple[bytes, dict[str, Any], dict[str, list[Any]]]:
    if not 1 <= roots <= MAX_ROOTS:
        raise ValueError("Root count exceeds declared bound")
    rows: list[dict[str, Any]] = []
    private_answers: dict[str, list[Any]] = {}
    seen_ids: set[str] = set()
    for index in range(roots):
        world = private_world(key, index)
        for query_index, (query, expected) in enumerate(
            zip(world["queries"], world["reference"], strict=True)
        ):
            case_id = mac(key, f"{SCHEMA}/case/{index}/{query_index}").hex()
            if case_id in seen_ids:
                raise ValueError("Duplicate keyed challenge identity")
            seen_ids.add(case_id)
            rows.append({
                "schema_version": SCHEMA,
                "challenge_id": case_id,
                "root_id": world["root_id"],
                "source_prefix": world["source_prefix"],
                "query": query,
                "task_kind": "finite_ground_pattern_match",
                "training_ready": False,
            })
            private_answers[case_id] = expected
    raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    manifest = {
        "schema_version": SCHEMA,
        "roots": roots,
        "challenges": len(rows),
        "challenge_bundle_sha256": digest(raw),
        "secret_commitment_sha256": digest(b"MeTTa-07-secret-commitment\0" + key),
        "generator": "keyed_hmac_sha256_finite_ground_v1",
        "references_in_challenge_bundle": False,
        "all_reference_answers_private": "requires_external_custody_attestation",
        "public_ci_is_dummy_key_only": True,
        "root_split": "keyed_exact_root_isolation",
        "source_training_ready": False,
        "sealed_eval_eligible": "requires_external_custody_attestation",
    }
    return raw, manifest, private_answers


def prepare(
    key_file: Path,
    challenge_path: Path,
    commitment_path: Path,
    roots: int,
) -> dict[str, Any]:
    key = read_key(key_file)
    raw, manifest, _ = challenge_bundle(key, roots)
    challenge_path.parent.mkdir(parents=True, exist_ok=True)
    commitment_path.parent.mkdir(parents=True, exist_ok=True)
    challenge_path.write_bytes(raw)
    commitment_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def score(
    key_file: Path,
    challenge_path: Path,
    commitment_path: Path,
    predictions_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    key = read_key(key_file)
    stored_manifest = json.loads(commitment_path.read_text(encoding="utf-8"))
    if not isinstance(stored_manifest, dict):
        raise ValueError("Invalid evaluation commitment")
    roots = stored_manifest.get("roots")
    if type(roots) is not int:
        raise ValueError("Invalid root count in commitment")
    expected_bytes, manifest, answers = challenge_bundle(key, roots)
    if stored_manifest != manifest or challenge_path.read_bytes() != expected_bytes:
        raise ValueError("Challenge/commitment identity or keyed source mismatch")
    submissions: dict[str, list[str]] = {}
    invalid_format: set[str] = set()
    for lineno, line in enumerate(predictions_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Submission line {lineno} is not a JSON object")
        cid = row.get("challenge_id")
        if not isinstance(cid, str) or cid not in answers or cid in submissions:
            raise ValueError(f"Unknown or duplicated challenge on line {lineno}")
        proposed = row.get("answers")
        if (
            not isinstance(proposed, list)
            or len(proposed) > 256
            or any(not isinstance(item, str) or len(item) > 4096 for item in proposed)
        ):
            invalid_format.add(cid)
            submissions[cid] = []
            continue
        submissions[cid] = proposed
    correct = 0
    invalid_syntax = len(invalid_format)
    for case_id, proposed in submissions.items():
        if case_id in invalid_format:
            continue
        try:
            expressions = [parse_one(text) for text in proposed]
        except UnsupportedSemantics:
            invalid_syntax += 1
            continue
        if result_signature(expressions) == result_signature(answers[case_id]):
            correct += 1
    total = len(answers)
    report = {
        "schema_version": SCHEMA,
        "challenge_bundle_sha256": manifest["challenge_bundle_sha256"],
        "predictions_sha256": digest(predictions_path.read_bytes()),
        "roots": roots,
        "challenges": total,
        "submitted": len(submissions),
        "missing": total - len(submissions),
        "invalid_syntax": invalid_syntax,
        "correct": correct,
        "incorrect_or_missing": total - correct,
        "accuracy": correct / total,
        "reference_semantics": "finite_ground_match_only",
        "answers_written_to_report": False,
        "secret_written_to_report": False,
        "external_custody_attested": False,
        "model_evaluated": "requires_valid_model_submissions",
        "training_ready": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "score"))
    parser.add_argument("--secret-file", type=Path, required=True)
    parser.add_argument("--challenge-file", type=Path, required=True)
    parser.add_argument("--commitment-file", type=Path, required=True)
    parser.add_argument("--predictions-file", type=Path)
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--roots", type=int, default=64)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.secret_file, args.challenge_file, args.commitment_file, args.roots
        )
    else:
        if args.predictions_file is None or args.summary_file is None:
            parser.error("Scoring requires predictions-file and summary-file")
        result = score(
            args.secret_file, args.challenge_file, args.commitment_file,
            args.predictions_file, args.summary_file,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
