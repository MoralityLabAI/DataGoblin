"""Export the same checked synthetic trace presentations as SFT and RLT views.

The two files are mutually exclusive training views over the same identities.
They are research candidates, never general corpus release approvals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .ram_hillclimb import canonical


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def export_views(
    traces_path: Path,
    summary_path: Path,
    sft_path: Path,
    rlt_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    raw = traces_path.read_bytes()
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != "metta-ram-hillclimb-v1"
        or report.get("training_ready") is not False
        or report.get("public_test_roots_materialized") != 0
        or report.get("train_repair_traces_sha256") != sha256(raw)
    ):
        raise ValueError("Trace source disagrees with CPU research manifest")
    records = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
    if not records or len(records) != report.get("train_repair_traces"):
        raise ValueError("Trace row count mismatch")
    views: set[tuple[str, int]] = set()
    root_exposures: Counter[str] = Counter()
    sft: list[dict[str, Any]] = []
    rlt: list[dict[str, Any]] = []
    for record in records:
        if (
            record.get("schema_version") != "metta-ram10-checked-repair-trace-v1"
            or record.get("training_ready") is not False
            or record.get("synthetic_research_fit_eligible") is not True
            or record.get("split") != "train_candidate"
            or record.get("teacher_source") != "independent_python_oracle"
        ):
            raise ValueError("Unverified or reserved root entered trace export")
        root = record["root"]
        step = record["presentation"]
        if not isinstance(root, str) or not root.startswith("metta-ram10-v1:"):
            raise ValueError("Invalid source lineage identity")
        if type(step) is not int or not 0 <= step <= 3:
            raise ValueError("Presentation index beyond four-view cap")
        key = (root, step)
        if key in views:
            raise ValueError("Duplicate source-root presentation")
        views.add(key)
        root_exposures[root] += 1
        source = record["metta_program"]
        if sha256(source.encode()) != record["source_sha256"]:
            raise ValueError("Source MeTTa bytes changed")
        target = record["verified_canonical_next_edit"]
        if (
            not isinstance(target, dict)
            or target.get("op") not in {"add", "remove", "stop"}
            or not isinstance(target.get("id"), str)
        ):
            raise ValueError("Unsupported teacher action")
        view_id = sha256(f"{root}:{step}:{record['source_sha256']}".encode())
        prompt = canonical({
            "metta_program": source,
            "current_selection": record["observation"]["selected"],
            "objective": "Emit the next checked add/remove/stop edit to repair the evidence set.",
        })
        common = {
            "root": root,
            "presentation": step,
            "source_sha256": record["source_sha256"],
            "trace_view_id": view_id,
            "training_ready": False,
            "synthetic_research_fit_eligible": True,
            "alternate_view_single_exposure": True,
            "teacher_authority": "bounded_independent_synthetic_oracle",
        }
        sft.append({
            "schema_version": "metta-ram10-looped-sft-alternate-v1",
            "instruction": "Propose one verified MeTTa evidence-set repair action.",
            "input": prompt,
            "output": canonical(target),
            "metadata": common,
        })
        rlt.append({
            "schema_version": "metta-ram10-rlt-alternate-v1",
            **common,
            "observation": prompt,
            "model_proposal": record["model_proposal"],
            "oracle_next_action": target,
            "verifier_feedback": {
                "model_action_matches_oracle": record["exact_teacher_match"],
                "public_constraint_valid": record["proposed_transition_public_constraint_valid"],
                "new_violations": record["proposed_transition_introduced_violations"],
                "verifier_calls": record["host_verifier_calls"],
                "verifier_vetoes": record["host_verifier_vetoes"],
            },
            "reward": record["reward"],
            "chosen": target,
            "rejected": record["rejected"],
            "source_trace_view_id": view_id,
        })
    if any(count > 4 for count in root_exposures.values()):
        raise ValueError("One source root received more than four presentations")
    sft_raw = "".join(canonical(row) + "\n" for row in sft).encode("utf-8")
    rlt_raw = "".join(canonical(row) + "\n" for row in rlt).encode("utf-8")
    for path, data in ((sft_path, sft_raw), (rlt_path, rlt_raw)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = {
        "schema_version": "metta-ram10-training-view-manifest-v1",
        "source_trace_sha256": sha256(raw),
        "sft_sha256": sha256(sft_raw),
        "rlt_sha256": sha256(rlt_raw),
        "rows_per_view": len(records),
        "unique_lineage_roots": len(root_exposures),
        "max_presentations_per_root": max(root_exposures.values()),
        "alternate_views_same_identity": True,
        "count_training_exposure_once_per_root_presentation": True,
        "consume_only_one_view_per_training_run": True,
        "development_labels_exported": False,
        "public_test_materialized": False,
        "general_corpus_training_ready": False,
        "synthetic_research_fit_eligible": True,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    result = export_views(
        args.bundle / "train_repair_traces.jsonl",
        args.bundle / "report.json",
        args.bundle / "train_sft_alternate.jsonl",
        args.bundle / "train_rlt_alternate.jsonl",
        args.bundle / "training_views_manifest.json",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
