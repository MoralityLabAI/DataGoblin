from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.adapter_verifier.architecture_tournament import (
    load_architecture_tournament_contract,
)
from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured, load_structured


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    config = load_architecture_tournament_contract(args.config)
    manifest = load_structured(args.dataset_manifest)
    custody = load_structured(args.custody_manifest)
    development = load_structured(args.development_evaluation)
    packet = load_structured(args.packet_audit)
    heldout = load_structured(args.heldout_evaluation)
    receipt = load_structured(args.heldout_receipt)
    shadow = _read_jsonl(args.shadow_predictions)
    arrays = heldout["array_results"]
    exact_status = float(np.mean([row["exact_status_vector"] for row in arrays]))
    exact_selected = float(np.mean([row["exact_selected_set"] for row in arrays]))
    per_surface = {
        pressure: {
            "exact_status_vector_accuracy": float(
                np.mean([row["exact_status_vector"] for row in arrays if row["pressure"] == pressure])
            ),
            "exact_selected_set_accuracy": float(
                np.mean([row["exact_selected_set"] for row in arrays if row["pressure"] == pressure])
            ),
        }
        for pressure in sorted({str(row["pressure"]) for row in arrays})
    }
    worst_status = min(value["exact_status_vector_accuracy"] for value in per_surface.values())
    worst_selected = min(value["exact_selected_set_accuracy"] for value in per_surface.values())
    observed_metrics = heldout["whole_array_metrics"]
    gates = config["heldout_gates"]
    reproduced_ldt_checks = {
        "minimum_exact_status_vector_accuracy": exact_status
        >= float(gates["minimum_ldt_exact_status_vector_accuracy"]),
        "minimum_exact_selected_set_accuracy": exact_selected
        >= float(gates["minimum_ldt_exact_selected_set_accuracy"]),
        "minimum_worst_surface_exact_status_vector_accuracy": worst_status
        >= float(gates["minimum_worst_surface_exact_status_vector_accuracy"]),
        "minimum_worst_surface_exact_selected_set_accuracy": worst_selected
        >= float(gates["minimum_worst_surface_exact_selected_set_accuracy"]),
    }
    resource_checks = []
    candidate_hashes = {}
    for run_dir in args.candidate_run:
        candidate_path = run_dir / "candidate_manifest.json"
        candidate = load_structured(candidate_path)
        candidate_id = str(candidate["candidate_id"])
        seed = int(candidate["seed"])
        candidate_hashes[f"{candidate_id}:{seed}"] = sha256_file(candidate_path)
        if candidate["trainable"]:
            run_receipt = load_structured(run_dir / "run_receipt.json")
            resource_checks.append(
                run_receipt.get("candidate_manifest_sha256") == sha256_file(candidate_path)
                and bool(run_receipt.get("caps_enforced"))
                and bool(run_receipt.get("cleanup_passed"))
                and not bool(run_receipt.get("heldout_labels_staged"))
                and not bool(run_receipt.get("development_evaluation_labels_staged"))
                and not bool(run_receipt.get("external_checkpoint_loaded"))
            )
        else:
            resource_checks.append(True)
    custody_hashes = {
        str(entry["surface_id"]): sha256_file(
            args.custody_manifest.parent / str(entry["ground_truth_path"])
        )
        for entry in custody["entries"]["heldout"]
    }
    shadow_by_id = {str(row["array_id"]): row for row in shadow}
    shadow_contract = all(
        row.get("authority") == "MODEL_RECOMMENDATION_ONLY"
        and row.get("source_model_sha256") == config["source_evidence"]["v15_model_sha256"]
        and not bool(row.get("heldout_metrics_seen"))
        and not bool(row.get("heldout_targets_seen"))
        for row in shadow
    )
    candidate_ids = [str(arm["candidate_id"]) for arm in config["arms"]]
    array_contract = all(
        row["candidate_ids"] == candidate_ids
        and shadow_by_id[row["array_id"]]["candidate_ids"] == candidate_ids
        and row["predicted_statuses"] == shadow_by_id[row["array_id"]]["statuses"]
        and row["exact_status_vector"]
        == (row["predicted_statuses"] == row["actual_statuses"])
        and row["exact_selected_set"]
        == (set(row["predicted_selected"]) == set(row["actual_selected"]))
        for row in arrays
    )
    advancement_reproduced = True
    advancing = []
    minimum_fraction = float(
        config["candidate_advancement"]["minimum_heldout_selected_array_fraction"]
    )
    for candidate_id in candidate_ids:
        statuses = [
            row["actual_statuses"][row["candidate_ids"].index(candidate_id)] for row in arrays
        ]
        selected_fraction = statuses.count("SELECTED") / len(statuses)
        arm = next(arm for arm in config["arms"] if arm["candidate_id"] == candidate_id)
        checks = {
            "trainable": bool(arm["trainable"]),
            "development_qualified": bool(
                development["candidate_summaries"][candidate_id]["qualified"]
            ),
            "heldout_qualified": bool(heldout["candidate_summaries"][candidate_id]["qualified"]),
            "minimum_selected_fraction": selected_fraction >= minimum_fraction,
            "no_unsafe_status": "UNSAFE" not in statuses,
            "global_ldt_whole_array_gates": all(reproduced_ldt_checks.values()),
        }
        reported = heldout["candidate_advancement"][candidate_id]
        advancement_reproduced &= (
            reported["heldout_status_counts"] == dict(sorted(Counter(statuses).items()))
            and _close(reported["heldout_selected_array_fraction"], selected_fraction)
            and reported["checks"] == checks
            and bool(reported["advances"]) == all(checks.values())
        )
        if all(checks.values()):
            advancing.append(candidate_id)
    checks = {
        "config_bound": heldout.get("tournament_config_sha256") == sha256_file(args.config),
        "dataset_bound": heldout.get("tournament_manifest_sha256")
        == sha256_file(args.dataset_manifest),
        "development_bound": heldout.get("development_evaluation_sha256")
        == sha256_file(args.development_evaluation),
        "packet_bound": heldout.get("confirmation_packet_audit_sha256")
        == sha256_file(args.packet_audit),
        "shadow_bound": heldout.get("shadow_predictions_sha256")
        == sha256_file(args.shadow_predictions),
        "heldout_receipt_bound": receipt.get("heldout_evaluation_sha256")
        == sha256_file(args.heldout_evaluation),
        "packet_passed_before_labels": bool(packet.get("passed"))
        and not bool(packet.get("authoritative_labels_read")),
        "development_did_not_read_heldout": not bool(development.get("heldout_labels_read"))
        and int(development.get("heldout_reveal_count", -1)) == 0,
        "single_join_and_no_retune": int(heldout.get("heldout_reveal_count", 0)) == 1
        and not bool(heldout.get("post_reveal_tuning"))
        and int(receipt.get("heldout_reveal_count", 0)) == 1
        and not bool(receipt.get("post_reveal_tuning")),
        "lineage_contamination_zero": int(manifest["lineage_audit"]["cross_lane_overlap"]) == 0,
        "record_contamination_zero": int(manifest["record_audit"]["cross_lane_overlap"]) == 0,
        "heldout_label_hashes_bound": heldout.get("heldout_label_hashes") == custody_hashes,
        "candidate_matrix_complete": len(candidate_hashes) == 10,
        "resource_and_cleanup_receipts_pass": all(resource_checks),
        "shadow_contract_pass": shadow_contract,
        "array_contract_pass": array_contract and len(arrays) == 12,
        "exact_status_reproduced": _close(
            observed_metrics["exact_status_vector_accuracy"], exact_status
        ),
        "exact_selected_reproduced": _close(
            observed_metrics["exact_selected_set_accuracy"], exact_selected
        ),
        "worst_status_reproduced": _close(
            observed_metrics["worst_surface_exact_status_vector_accuracy"], worst_status
        ),
        "worst_selected_reproduced": _close(
            observed_metrics["worst_surface_exact_selected_set_accuracy"], worst_selected
        ),
        "ldt_gates_reproduced": heldout.get("ldt_gate_checks") == reproduced_ldt_checks,
        "advancement_reproduced": advancement_reproduced
        and sorted(heldout.get("advancing_candidates", ())) == sorted(advancing),
        "no_universal_winner": heldout.get("winner") is None
        and heldout.get("universal_scalar_winner") is False,
        "reported_final_checks_pass": all(heldout.get("final_checks", {}).values()),
    }
    passed = all(checks.values())
    expected_confirmation = passed and bool(advancing)
    checks["confirmation_conclusion_consistent"] = bool(
        heldout.get("confirmed_for_next_preregistration_only")
    ) == expected_confirmation and bool(receipt.get("confirmed_for_next_preregistration_only")) == expected_confirmation
    passed = all(checks.values())
    result = {
        "schema_version": "game-adapter-architecture-independent-audit.v2",
        "checks": checks,
        "passed": passed,
        "reproduced_whole_array_metrics": {
            "exact_status_vector_accuracy": exact_status,
            "exact_selected_set_accuracy": exact_selected,
            "worst_surface_exact_status_vector_accuracy": worst_status,
            "worst_surface_exact_selected_set_accuracy": worst_selected,
        },
        "reproduced_advancing_candidates": sorted(advancing),
        "conclusion": (
            "confirmed_for_next_preregistration_only"
            if passed and advancing
            else "not_confirmed_no_promotion"
        ),
    }
    dump_structured(args.out, result)
    if not passed:
        failed = sorted(name for name, value in checks.items() if not value)
        raise ValueError(f"architecture tournament evidence audit failed: {failed}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit v2 adapter architecture evidence")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--custody-manifest", required=True, type=Path)
    parser.add_argument("--development-evaluation", required=True, type=Path)
    parser.add_argument("--packet-audit", required=True, type=Path)
    parser.add_argument("--shadow-predictions", required=True, type=Path)
    parser.add_argument("--heldout-evaluation", required=True, type=Path)
    parser.add_argument("--heldout-receipt", required=True, type=Path)
    parser.add_argument("--candidate-run", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.dataset_manifest = args.dataset_manifest.resolve()
    args.custody_manifest = args.custody_manifest.resolve()
    args.development_evaluation = args.development_evaluation.resolve()
    args.packet_audit = args.packet_audit.resolve()
    args.shadow_predictions = args.shadow_predictions.resolve()
    args.heldout_evaluation = args.heldout_evaluation.resolve()
    args.heldout_receipt = args.heldout_receipt.resolve()
    args.candidate_run = [path.resolve() for path in args.candidate_run]
    args.out = args.out.resolve()
    result = audit(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
