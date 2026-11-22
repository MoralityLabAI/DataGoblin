#!/usr/bin/env python3
"""Build an open-development MeTTa root partition and exact-leakage audit.

Every source-query label here is already public. The output is a partition
proposal for development experiments, never a sealed evaluation set or a
training-release approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from corpus import ROOT, validate_corpus

PARTITIONS = ("train_candidate", "dev_public", "test_public")
FRACTIONS = (0.70, 0.15, 0.15)
DEFAULT_SEED = "metta-public-root-splits-v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(record)
    return rows


def check_identity(
    record: dict[str, Any], source_map: dict[str, dict[str, Any]]
) -> str:
    meta = record.get("metadata", record)
    if not isinstance(meta, dict):
        raise ValueError("missing source metadata")
    sid = meta.get("lineage_root")
    if not isinstance(sid, str) or meta.get("source_id") != sid:
        raise ValueError("source_id/lineage_root mismatch")
    src = source_map.get(sid)
    if src is None:
        raise ValueError(f"unknown pinned source root: {sid}")
    for field, source_field in (
        ("source_repo", "source_repo"),
        ("source_commit", "source_commit"),
        ("source_path", "source_path"),
        ("source_blob_sha1", "upstream_blob_sha1"),
        ("license_spdx", "license_spdx"),
        ("task_family", "task_family"),
    ):
        if meta.get(field) != src[source_field]:
            raise ValueError(f"{sid}: {field} differs from pinned source")
    runtime = meta.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("package") != "hyperon"
        or runtime.get("version") != "0.2.10"
    ):
        raise ValueError(f"{sid}: unexpected native runtime")
    if meta.get("training_ready") is not False or meta.get("split") != "unassigned":
        raise ValueError(f"{sid}: input was promoted before the audit")
    return sid


def build_plan(
    packets_path: Path,
    exceptions_path: Path,
    out: Path,
    *,
    corpus_root: Path = ROOT,
    source_map: dict[str, dict[str, Any]] | None = None,
    seed: str = DEFAULT_SEED,
) -> dict[str, Any]:
    """Check original provenance and issue a root-only, public diagnostic plan."""
    if not seed:
        raise ValueError("seed must be nonempty")
    if source_map is None:
        sources, _, errors = validate_corpus(corpus_root)
        if errors:
            raise ValueError("Invalid MeTTa corpus: " + "; ".join(errors))
        source_map = {source["id"]: source for source in sources}
    packets = read_jsonl(packets_path)
    exceptions = read_jsonl(exceptions_path)
    if not packets:
        raise ValueError("No native-agreed packets to partition")
    seen_packet_ids: set[str] = set()
    seen_observations: set[str] = set()
    root_counts: Counter[str] = Counter()
    exception_counts: Counter[str] = Counter()
    queries: dict[str, set[str]] = defaultdict(set)
    pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    prefixes: dict[str, set[str]] = defaultdict(set)

    for record in packets:
        sid = check_identity(record, source_map)
        meta = record["metadata"]
        if (
            record.get("schema_version") != "metta-native-query-candidate-v1"
            or meta.get("native_observation") != "native_assertion_pass"
            or meta.get("native_agreement_observed") is not True
            or meta.get("source_author_expected") is not True
            or meta.get("independent_semantic_oracle") is not False
            or meta.get("is_generated_rollout") is not False
            or meta.get("approval_scope") != "review_candidate_only"
        ):
            raise ValueError(f"{sid}: unsupported native or release claim")
        observation = meta.get("assertion_observation_id")
        if not isinstance(observation, str) or observation in seen_observations:
            raise ValueError(f"{sid}: missing or duplicated assertion observation")
        seen_observations.add(observation)
        input_text = record.get("input")
        answer = record.get("output")
        if not isinstance(input_text, str) or not isinstance(answer, str):
            raise ValueError(f"{sid}: invalid input/output")
        parsed = json.loads(input_text)
        if (
            not isinstance(parsed, dict)
            or parsed.get("runtime") != "hyperon==0.2.10"
            or not isinstance(parsed.get("query"), str)
            or not isinstance(parsed.get("source_prefix"), str)
        ):
            raise ValueError(f"{sid}: invalid source-conditioned query")
        prefix = parsed["source_prefix"].encode("utf-8")
        if (
            meta.get("source_context_sha256") != sha256(prefix)
            or meta.get("prefix_bytes") != len(prefix)
            or meta.get("answer_text_sha256") != sha256(answer.encode("utf-8"))
        ):
            raise ValueError(f"{sid}: prefix or answer digest mismatch")
        ordinal = meta.get("evaluation_ordinal")
        if not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError(f"{sid}: invalid evaluation ordinal")
        packet_id = sha256(
            f"{sid}:{ordinal}:{sha256(input_text.encode('utf-8'))}".encode()
        )
        if record.get("id") != packet_id or packet_id in seen_packet_ids:
            raise ValueError(f"{sid}: packet ID mismatch or duplicate")
        seen_packet_ids.add(packet_id)
        root_counts[sid] += 1
        queries[parsed["query"]].add(sid)
        pairs[(parsed["query"], answer)].add(sid)
        prefixes[meta["source_context_sha256"]].add(sid)

    for record in exceptions:
        sid = check_identity(record, source_map)
        observation = record.get("assertion_observation_id")
        if not isinstance(observation, str) or observation in seen_observations:
            raise ValueError(f"{sid}: duplicate or missing exception observation")
        seen_observations.add(observation)
        if not isinstance(record.get("reason"), str) or not record["reason"]:
            raise ValueError(f"{sid}: missing exception reason")
        exception_counts[sid] += 1

    # Joining roots on exact queries is conservative when context changes the
    # expected result. Exact source prefixes and query-answer pairs also join.
    parents = {sid: sid for sid in root_counts}

    def find(sid: str) -> str:
        while parents[sid] != sid:
            parents[sid] = parents[parents[sid]]
            sid = parents[sid]
        return sid

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)

    for groups in (queries.values(), pairs.values(), prefixes.values()):
        for roots in groups:
            ordered = sorted(roots)
            for sid in ordered[1:]:
                union(ordered[0], sid)

    components: dict[str, list[str]] = defaultdict(list)
    for sid in sorted(root_counts):
        components[find(sid)].append(sid)
    grouped = sorted(
        components.values(),
        key=lambda roots: (
            -sum(root_counts[sid] for sid in roots),
            sha256((seed + "\0" + "\0".join(roots)).encode("utf-8")),
        ),
    )

    total = len(packets)
    targets = [total * fraction for fraction in FRACTIONS]
    current = [0, 0, 0]
    assignments: dict[str, str] = {}
    for ix, roots in enumerate(grouped):
        weight = sum(root_counts[sid] for sid in roots)
        if ix < len(PARTITIONS):
            choice = ix
        else:
            choice = min(
                range(len(PARTITIONS)),
                key=lambda k: (
                    sum(
                        (current[j] + (weight if j == k else 0) - targets[j]) ** 2
                        for j in range(len(PARTITIONS))
                    ),
                    k,
                ),
            )
        current[choice] += weight
        for sid in roots:
            assignments[sid] = PARTITIONS[choice]
    for sid in exception_counts:
        if sid not in assignments:
            assignments[sid] = "exception_quarantine"

    for groups in (queries.values(), pairs.values(), prefixes.values()):
        for roots in groups:
            if len({assignments[sid] for sid in roots}) > 1:
                raise ValueError("Cross-partition exact-source or query leakage")

    summary = {
        "schema_version": "metta-public-root-split-plan-v1",
        "seed": seed,
        "source_packets_sha256": sha256(packets_path.read_bytes()),
        "source_exceptions_sha256": sha256(exceptions_path.read_bytes()),
        "native_query_packets": len(packets),
        "exception_observations": len(exceptions),
        "candidate_source_roots": len(root_counts),
        "exception_only_roots": sum(sid not in root_counts for sid in exception_counts),
        "root_components": len(grouped),
        "shared_exact_query_keys_across_roots": sum(len(v) > 1 for v in queries.values()),
        "shared_exact_query_answer_keys_across_roots": sum(len(v) > 1 for v in pairs.values()),
        "shared_exact_prefixes_across_roots": sum(len(v) > 1 for v in prefixes.values()),
        "packets_by_partition": dict(zip(PARTITIONS, current, strict=True)),
        "roots_by_partition": {
            name: sum(partition == name for partition in assignments.values())
            for name in (*PARTITIONS, "exception_quarantine")
        },
        "root_assignments": dict(sorted(assignments.items())),
        "labels_embedded_in_plan": False,
        "candidate_source_labels_already_public": True,
        "sealed_eval_eligible": False,
        "training_ready": False,
        "independent_semantic_oracle": False,
        "scope": "public_development_diagnostic_only",
        "caveat": "Exact-query/root grouping does not establish near-duplicate or semantic disjointness.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packets", type=Path,
        default=ROOT / "training_candidates/hyperon_source_query_pairs.jsonl",
    )
    parser.add_argument(
        "--exceptions", type=Path,
        default=ROOT / "training_candidates/hyperon_query_exceptions.jsonl",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "reports/public_root_split_plan.json",
    )
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    result = build_plan(args.packets, args.exceptions, args.out, seed=args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
