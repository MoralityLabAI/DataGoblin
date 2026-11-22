"""Hash and oracle replay of paired MeTTa AIRIS-style research receipts.

Explicitly excludes promotion into training. Internal confirmation labels are
held in a separate research-only artifact; this auditor validates their exact
source-root replay and preserves the original four-presentation SFT cap.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .airis_counterfactual import (
    SCHEMA,
    _confirmation_cases,
    canonical,
    discover_policy,
    evaluate_pair,
)
from .collect import sha256_bytes
from .looped_sft import root_id, split_for
from .pattern_miner import _internal_partition
from .sft_oracle import FAMILIES, OOD_FAMILY


def audit_bundle(directory: Path, *, full_replay: bool = True) -> dict[str, Any]:
    """Validate hashes, root custody, matched arms and optional exact oracle replay."""
    manifest = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    raw = (directory / "internal_confirmation_only.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("training_ready") is not False
        or manifest.get("new_sft_presentations") != 0
        or manifest.get("internal_confirmation_labels_used_for_policy") is not False
        or manifest.get("reserved_dev_test_ood_roots_accessed") != 0
        or manifest.get("native_airis_executed") is not False
        or manifest.get("learned_model_evaluated") is not False
    ):
        raise ValueError("unsafe or incompatible AIRIS research custody")
    n = manifest.get("actual_paired_cases")
    if type(n) is not int or n < 1 or len(rows) != n:
        raise ValueError("paired case count mismatch")
    digest_actual = sha256_bytes("\n".join(canonical(row) for row in rows).encode())
    if manifest.get("source_root_receipts_sha256") != digest_actual:
        raise ValueError("paired source-root receipts digest mismatch")
    counts = Counter()
    seen = set()
    root_limit = manifest.get("root_horizon_per_family")
    requested = manifest.get("requested_case_limit")
    if type(root_limit) is not int or not 1 <= root_limit <= 512:
        raise ValueError("unbounded source root horizon")
    if type(requested) is not int or not 1 <= requested <= 512:
        raise ValueError("invalid paired case limit")
    expected_pool = {
        root_id(family, seed)
        for family, seed in _confirmation_cases(root_limit, requested)
    }
    if n > requested or manifest["available_confirmation_roots_selected"] != len(expected_pool):
        raise ValueError("unmatched confirmation pool")
    motifs = [tuple(x) for x in manifest.get("policy_motifs_frozen", [])]
    for row in rows:
        root = row.get("root")
        if not isinstance(root, str) or root in seen:
            raise ValueError("repeated or malformed root")
        seen.add(root)
        if root not in expected_pool or row.get("split") != "train_candidate":
            raise ValueError("root crosses internal confirmation boundary")
        pieces = root.split(":")
        if len(pieces) != 3 or pieces[0] != "metta-sft-v1":
            raise ValueError("noncanonical source identity")
        family, index_str = pieces[1:]
        if family not in FAMILIES or family == OOD_FAMILY or not index_str.isdigit():
            raise ValueError("reserved family root")
        seed = int(index_str)
        if (
            row["family"] != family
            or split_for(family, seed) != "train_candidate"
            or _internal_partition(root) != "confirmation"
            or row.get("internal_partition") != "confirmation"
        ):
            raise ValueError("reserved source-root label accessed")
        if (
            row.get("training_ready") is not False
            or row.get("new_sft_presentations") != 0
            or row.get("ticket_custody") != "shadow_only_not_admitted"
            or row.get("zero_change_sham_passed") is not True
        ):
            raise ValueError("case-level admission or sham violation")
        for arm in ("guided", "uniform"):
            verdict = row.get(arm)
            if (
                not isinstance(verdict, dict)
                or verdict.get("training_ready") is not False
                or verdict.get("one_fact_delta") != 1
                or verdict.get("native_metta_executed") is not False
            ):
                raise ValueError("counterfactual arm invalid")
            if not isinstance(verdict.get("world_sha256"), str) or len(verdict["world_sha256"]) != 64:
                raise ValueError("missing changed-world digest")
            counts[(arm, "targets")] += int(verdict["target_violation_changed"])
            counts[(arm, "signatures")] += int(verdict["violation_signature_changed"])
            counts[(arm, "first_edits")] += int(verdict["oracle_first_edit_changed"])
        if full_replay:
            rebuilt = evaluate_pair(family, seed, motifs, control_seed=manifest["control_seed"])
            if rebuilt != row:
                raise ValueError(f"independent source-root replay differs: {root}")
    for arm in ("guided", "uniform"):
        scores = manifest["paired_policy"][arm]
        if (
            scores["paired_roots"] != n
            or scores["targeted_violation_flips"] != counts[(arm, "targets")]
            or scores["full_violation_signature_changes"] != counts[(arm, "signatures")]
            or scores["oracle_next_edit_changes"] != counts[(arm, "first_edits")]
        ):
            raise ValueError("aggregated counterfactual metrics drift")
    if full_replay:
        frozen, source_counts = discover_policy(
            roots_per_family=root_limit,
            min_support=manifest.get("min_support", 20),
            top_k=manifest.get("top_k", 24),
        )
        if frozen != motifs or source_counts["discovery"] != manifest["discovery_root_count"]:
            raise ValueError("discovery-policy replay differs")
    return {
        "schema_version": "metta-airis-bundle-audit-v1",
        "valid": True,
        "oracle_replayed": full_replay,
        "paired_cases": n,
        "public_source_roots_only": True,
        "reserved_root_labels_touched": 0,
        "cross_arm_root_ids_matched": True,
        "training_ready": False,
        "new_sft_presentations": 0,
        "receipt_sha256": digest_actual,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--fast", action="store_true", help="hash/custody only; skip oracle replay")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    receipt = audit_bundle(args.bundle, full_replay=not args.fast)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
