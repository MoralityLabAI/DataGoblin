#!/usr/bin/env python3
"""Make unapproved source-conditioned MeTTa query learning packets from native pass receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from corpus import ROOT, safe_file, validate_corpus
from native_assertions import scan_eval_sites

INSTRUCTION = (
    "Given the preceding MeTTa program and its sequential state changes, "
    "predict the expected result expression for the next MeTTa query. "
    "The provided runtime is Hyperon 0.2.10. "
    "Keep expression syntax exact."
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_packets(
    root: Path,
    native_path: Path,
    out: Path,
    exceptions_path: Path,
    report_path: Path,
    max_context_bytes: int = 16_384,
) -> dict[str, Any]:
    source_rows, _, errors = validate_corpus(root)
    if errors:
        raise ValueError("Corpus provenance invalid: " + "; ".join(errors))
    manifest = {row["id"]: row for row in source_rows}
    verdicts = [json.loads(line) for line in native_path.read_text(encoding="utf-8").splitlines() if line]
    packets: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    roots: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for observation in verdicts:
        if observation["id"] in seen_ids:
            raise ValueError("Duplicate assertion observation: " + observation["id"])
        seen_ids.add(observation["id"])
        sid = observation["source_id"]
        source = manifest.get(sid)
        if source is None or source["upstream_blob_sha1"] != observation["source_blob_sha1"]:
            raise ValueError("Source file identity differs: " + sid)
        program = safe_file(root, source["raw_path"]).read_text(encoding="utf-8")
        sites = scan_eval_sites(program)
        index = observation["evaluation_ordinal"]
        if not 0 <= index < len(sites):
            raise ValueError("Evaluation ordinal outside source: " + sid)
        site = sites[index]
        if (
            site["eval_source"] != observation["assertion_source"]
            or site["line_start"] != observation["line_start"]
            or site["line_end"] != observation["line_end"]
            or site["operator"] != observation["operator"]
        ):
            raise ValueError("Source evaluation span mismatch: " + sid)
        status = observation["native_result"]
        counts[status] += 1
        roots[sid] += 1
        identity = {
            "assertion_observation_id": observation["id"],
            "source_id": sid,
            "source_repo": source["source_repo"],
            "source_commit": source["source_commit"],
            "source_path": source["source_path"],
            "source_blob_sha1": source["upstream_blob_sha1"],
            "license_spdx": source["license_spdx"],
            "runtime": observation["runtime"],
            "line_start": site["line_start"],
            "evaluation_ordinal": index,
            "task_family": source["task_family"],
            "lineage_root": sid,
            "split": "unassigned",
            "training_ready": False,
            "native_observation": status,
            "native_stdout_sha256": observation["stdout_sha256"],
        }
        if status != "native_assertion_pass" or observation["native_output"] != "[()]":
            exceptions.append({**identity, "reason": "nonpassing_or_unobserved_native_assertion"})
            continue
        if site["query_expression"] is None or site["author_expected_expression"] is None:
            exceptions.append({**identity, "reason": "no_parseable_query_expected_pair"})
            continue
        prefix = program[:site["start_offset"]]
        prefix_bytes = len(prefix.encode("utf-8"))
        if prefix_bytes > max_context_bytes:
            exceptions.append({**identity, "reason": "context_over_byte_budget", "prefix_bytes": prefix_bytes})
            continue
        # The supervised completion comes from the author's expected-expression
        # field, and the runtime success establishes only native agreement.
        input_text = json.dumps({
            "runtime": "hyperon==0.2.10",
            "source_prefix": prefix,
            "query": site["query_expression"],
        }, sort_keys=True, ensure_ascii=False)
        output_text = site["author_expected_expression"]
        packet = {
            "schema_version": "metta-native-query-candidate-v1",
            "id": sha256((sid + ":" + str(index) + ":" + sha256(input_text.encode())).encode()),
            "instruction": INSTRUCTION,
            "input": input_text,
            "output": output_text,
            "metadata": {
                **identity,
                "prefix_bytes": prefix_bytes,
                "source_context_sha256": sha256(prefix.encode("utf-8")),
                "answer_text_sha256": sha256(output_text.encode("utf-8")),
                "assertion_type": observation["operator"],
                "source_author_expected": True,
                "native_agreement_observed": True,
                "independent_semantic_oracle": False,
                "is_generated_rollout": False,
                "approval_scope": "review_candidate_only",
            },
        }
        packets.append(packet)
    packets.sort(key=lambda x: (x["metadata"]["source_id"], x["metadata"]["evaluation_ordinal"]))
    exceptions.sort(key=lambda x: (x["source_id"], x["evaluation_ordinal"]))
    report = {
        "schema_version": "metta-native-query-candidates-v1",
        "source_assertion_observations": len(verdicts),
        "source_lineage_roots": len(roots),
        "packets": len(packets),
        "exceptions": len(exceptions),
        "native_observations_by_status": dict(sorted(counts.items())),
        "max_context_bytes": max_context_bytes,
        "heldout_labels_written": 0,
        "root_splits_assigned": 0,
        "training_ready": False,
        "independent_semantic_oracle": False,
        "approval_scope": "source_query_candidates_pending_policy_and_leakage_review",
    }
    for path in (out, exceptions_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n" for x in packets),
                   encoding="utf-8")
    exceptions_path.write_text(
        "".join(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n" for x in exceptions),
        encoding="utf-8",
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--native", type=Path, default=ROOT / "native_assertion_verdicts/hyperon_0_2_10.jsonl"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "training_candidates/hyperon_source_query_pairs.jsonl"
    )
    parser.add_argument(
        "--exceptions", type=Path, default=ROOT / "training_candidates/hyperon_query_exceptions.jsonl"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/hyperon_query_candidates.json"
    )
    parser.add_argument("--max-context-bytes", type=int, default=16_384)
    args = parser.parse_args()
    if args.max_context_bytes < 1:
        parser.error("max-context-bytes must be positive")
    result = build_packets(args.root, args.native, args.out, args.exceptions, args.report, args.max_context_bytes)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
