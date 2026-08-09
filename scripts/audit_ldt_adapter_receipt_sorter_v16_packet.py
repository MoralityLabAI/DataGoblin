#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_sorter import sha256_file
from datagoblin.serialization import dump_structured, load_structured

FORBIDDEN_TARGET_KEYS = frozenset(
    {
        "actual",
        "authoritative_label",
        "gold",
        "ground_truth",
        "label",
        "labels",
        "target",
        "targets",
    }
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), abs_tol=1e-12)


def _individual_qualified(
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> bool:
    mean = receipt["cv_mean_metrics"]
    folds = receipt["cv_fold_metrics"]
    calibration = receipt["calibration"]["selected_metrics"]
    return (
        _close(
            mean["safety_gate_accuracy"],
            contract["required_mean_safety_gate_accuracy"],
        )
        and _close(
            mean["swap_consistency"],
            contract["required_mean_swap_consistency"],
        )
        and float(mean["preference_macro_f1"])
        >= float(contract["minimum_mean_preference_macro_f1"])
        and float(mean["abstain_recall"])
        >= float(contract["minimum_mean_abstain_recall"])
        and float(mean["status_macro_f1"])
        >= float(contract["minimum_mean_status_macro_f1"])
        and float(mean["exact_array_accuracy"])
        >= float(contract["minimum_mean_exact_array_accuracy"])
        and float(mean["selected_set_exact_accuracy"])
        >= float(contract["minimum_mean_selected_set_exact_accuracy"])
        and min(float(row["exact_array_accuracy"]) for row in folds)
        >= float(contract["minimum_worst_fold_exact_array_accuracy"])
        and min(
            float(row["selected_set_exact_accuracy"]) for row in folds
        )
        >= float(
            contract["minimum_worst_fold_selected_set_exact_accuracy"]
        )
        and float(calibration["expected_calibration_error"])
        <= float(contract["maximum_oof_expected_calibration_error"])
    )


def _rank(receipt: Mapping[str, Any]) -> tuple[float, ...]:
    folds = receipt["cv_fold_metrics"]
    mean = receipt["cv_mean_metrics"]
    calibration = receipt["calibration"]["selected_metrics"]
    return (
        -min(float(row["selected_set_exact_accuracy"]) for row in folds),
        -min(float(row["exact_array_accuracy"]) for row in folds),
        -min(float(row["status_macro_f1"]) for row in folds),
        -min(float(row["preference_macro_f1"]) for row in folds),
        -float(mean["selected_set_exact_accuracy"]),
        -float(mean["exact_array_accuracy"]),
        -float(mean["status_macro_f1"]),
        -float(mean["preference_macro_f1"]),
        float(calibration["multiclass_negative_log_likelihood"]),
        float(receipt["trainable_parameters"]),
        float(receipt["seed"]),
    )


def _reproduce_selection(
    config: Mapping[str, Any],
    trainer: Mapping[str, Any],
) -> dict[str, Any]:
    contract = config["development_selection"]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for receipt in trainer["candidate_receipts"]:
        grouped[str(receipt["candidate_id"])].append(receipt)
    selected = {}
    qualified_seeds = {}
    for candidate_id, receipts in grouped.items():
        qualified = [
            row for row in receipts if _individual_qualified(row, contract)
        ]
        selected[candidate_id] = min(
            qualified if qualified else receipts,
            key=_rank,
        )
        qualified_seeds[candidate_id] = sorted(
            int(row["seed"]) for row in qualified
        )
    primary_id = str(contract["primary_candidate_ids"][0])
    comparison = contract["comparative_primary_gates"]
    baseline_id = str(comparison["baseline_candidate_id"])
    primary = selected[primary_id]
    baseline = selected[baseline_id]
    primary_folds = primary["cv_fold_metrics"]
    baseline_folds = baseline["cv_fold_metrics"]
    comparative = {
        "mean_exact_array_accuracy_delta": (
            float(primary["cv_mean_metrics"]["exact_array_accuracy"])
            - float(baseline["cv_mean_metrics"]["exact_array_accuracy"])
            >= float(
                comparison["minimum_mean_exact_array_accuracy_delta"]
            )
        ),
        "mean_selected_set_exact_accuracy_delta": (
            float(
                primary["cv_mean_metrics"][
                    "selected_set_exact_accuracy"
                ]
            )
            - float(
                baseline["cv_mean_metrics"][
                    "selected_set_exact_accuracy"
                ]
            )
            >= float(
                comparison[
                    "minimum_mean_selected_set_exact_accuracy_delta"
                ]
            )
        ),
        "worst_fold_exact_array_accuracy_delta": (
            min(
                float(row["exact_array_accuracy"])
                for row in primary_folds
            )
            - min(
                float(row["exact_array_accuracy"])
                for row in baseline_folds
            )
            >= float(
                comparison[
                    "minimum_worst_fold_exact_array_accuracy_delta"
                ]
            )
        ),
        "worst_fold_selected_set_exact_accuracy_delta": (
            min(
                float(row["selected_set_exact_accuracy"])
                for row in primary_folds
            )
            - min(
                float(row["selected_set_exact_accuracy"])
                for row in baseline_folds
            )
            >= float(
                comparison[
                    "minimum_worst_fold_selected_set_exact_accuracy_delta"
                ]
            )
        ),
    }
    authorized = _individual_qualified(primary, contract) and all(
        comparative.values()
    )
    return {
        "selected_seeds": {
            candidate_id: int(row["seed"])
            for candidate_id, row in selected.items()
        },
        "qualified_seeds": qualified_seeds,
        "primary_id": primary_id if authorized else None,
        "primary_seed": int(primary["seed"]) if authorized else None,
        "comparative_checks": comparative,
        "authorized": authorized,
    }


def _target_free(value: Any) -> bool:
    if isinstance(value, Mapping):
        return not (set(value) & FORBIDDEN_TARGET_KEYS) and all(
            _target_free(item) for item in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return all(_target_free(item) for item in value)
    return True


def _caps_receipt_valid(receipt: Mapping[str, Any]) -> bool:
    telemetry = (
        float(receipt["peak_ram_mb"]),
        float(receipt["cpu_pct"]),
        float(receipt["peak_io_mb_s"]),
    )
    return (
        bool(receipt["caps_enforced"])
        and int(receipt["exit_code"]) == 0
        and all(math.isfinite(value) and value >= 0.0 for value in telemetry)
    )


def _public_splits_valid(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> bool:
    public_root = manifest_path.parent.resolve()
    for name, split in manifest["splits"].items():
        path = (public_root / str(split["path"])).resolve()
        try:
            path.relative_to(public_root)
        except ValueError:
            return False
        if not path.is_file():
            return False
        rows = _read_jsonl(path)
        if (
            sha256_file(path) != split["sha256"]
            or len(rows) != int(split["count"])
        ):
            return False
        heldout = str(name).startswith("heldout_")
        if heldout and (
            bool(split["targets_in_file"]) or not _target_free(rows)
        ):
            return False
    return True


def _resource_receipts_valid(
    config: Mapping[str, Any],
    config_sha256: str,
    manifest_sha256: str,
    plan_sha256: str,
    shard_root: Path,
) -> tuple[bool, list[dict[str, Any]]]:
    expected = {
        (str(model["model_id"]), int(seed))
        for model in config["models"]["candidates"]
        for seed in config["training"]["seeds"]
    }
    observed = set()
    receipts = []
    valid = True
    for candidate_id, seed in sorted(expected):
        run_dir = shard_root / f"{candidate_id}-seed-{seed}"
        receipt_path = run_dir / "run_receipt.json"
        if not receipt_path.is_file():
            valid = False
            continue
        receipt = load_structured(receipt_path)
        trainer = receipt["trainer_receipt"]
        cap_receipt = receipt["caps_receipt"]
        cleanup = receipt["cleanup_receipt"]
        candidate_receipts = trainer["candidate_receipts"]
        checkpoints = list(
            (run_dir / "capped_run" / "checkpoints").rglob(
                "epoch-*.json"
            )
        )
        cell_valid = (
            int(receipt["capped_process_exit_code"]) == 0
            and int(receipt["wsl_stage_cleanup_exit_code"]) == 0
            and receipt["trainer_plan_sha256"] == plan_sha256
            and _caps_receipt_valid(cap_receipt)
            and bool(cleanup["cleanup_passed"])
            and not cleanup["lingering_owned_pids"]
            and not receipt["heldout_pair_labels_staged"]
            and not receipt["heldout_array_labels_staged"]
            and not receipt["external_checkpoint_loaded"]
            and trainer["status"] == "completed"
            and trainer["preregistration_sha256"] == config_sha256
            and trainer["dataset_manifest_sha256"] == manifest_sha256
            and not trainer["heldout_pair_targets_seen"]
            and not trainer["heldout_array_targets_seen"]
            and bool(trainer["hard_caps_verified"])
            and trainer["shard_filter"]
            == {"candidate_id": candidate_id, "seed": seed}
            and len(candidate_receipts) == 1
            and candidate_receipts[0]["candidate_id"] == candidate_id
            and int(candidate_receipts[0]["seed"]) == seed
            and bool(candidate_receipts[0]["trained"])
            and int(candidate_receipts[0]["checkpoint_count"])
            == len(checkpoints)
            and len(checkpoints) > 0
        )
        valid = valid and cell_valid
        observed.add((candidate_id, seed))
        receipts.append(
            {
                "candidate_id": candidate_id,
                "seed": seed,
                "run_receipt_sha256": sha256_file(receipt_path),
                "valid": cell_valid,
            }
        )
    return valid and observed == expected, receipts


def audit(args: argparse.Namespace) -> dict[str, Any]:
    config = load_structured(args.preregistration)
    manifest = load_structured(args.dataset_manifest)
    generation = load_structured(args.generation_report)
    plan = load_structured(args.trainer_plan)
    trainer = load_structured(args.merged_trainer_summary)
    selection = load_structured(args.selection_receipt)
    config_sha256 = sha256_file(args.preregistration)
    manifest_sha256 = sha256_file(args.dataset_manifest)
    plan_sha256 = sha256_file(args.trainer_plan)
    pair_sha256 = sha256_file(args.merged_pair_predictions)
    array_sha256 = sha256_file(args.merged_array_predictions)
    reproduced = _reproduce_selection(config, trainer)
    expected_cells = len(config["models"]["candidates"]) * len(
        config["training"]["seeds"]
    )
    custody = manifest["heldout_label_custody"]
    checks: dict[str, bool] = {
        "v16_contract_frozen": config["schema_version"] == "16.0"
        and config["status"] == "frozen_before_dataset_generation",
        "generation_bound": generation["preregistration_sha256"]
        == config_sha256
        and generation["dataset_manifest_sha256"] == manifest_sha256,
        "generation_passed": bool(generation["generation_passed"])
        and all(generation["checks"].values()),
        "manifest_passed": all(manifest["checks"].values()),
        "public_splits_hash_bound_and_heldout_target_free": (
            _public_splits_valid(args.dataset_manifest, manifest)
        ),
        "lineage_firewall_passed": manifest["lineage_audit"][
            "cross_split_groups"
        ]
        == 0
        and manifest["lineage_audit"]["prior_overlap"] == 0
        and manifest["lineage_audit"]["forbidden_prior_groups"]
        == len(config["surface"]["forbidden_prior_lineage_sha256"]),
        "sealed_custody_intact": custody["authority"]
        == "ENVIRONMENT_OWNED"
        and bool(custody["sealed_before_training"])
        and not bool(custody["paths_disclosed_to_candidate"])
        and int(custody["reveal_count_budget"]) == 1,
        "trainer_plan_bound_and_target_free": plan[
            "preregistration_sha256"
        ]
        == config_sha256
        and plan["dataset_manifest_sha256"] == manifest_sha256
        and not plan["heldout_pair_label_access"]
        and not plan["heldout_array_label_access"]
        and plan["external_checkpoint"] is None
        and not plan["network"]
        and plan["hard_caps"] == config["hard_caps"],
        "merged_trainer_complete_and_target_free": trainer["status"]
        == "completed"
        and trainer["preregistration_sha256"] == config_sha256
        and trainer["dataset_manifest_sha256"] == manifest_sha256
        and not trainer["heldout_pair_targets_seen"]
        and not trainer["heldout_array_targets_seen"]
        and len(trainer["candidate_receipts"]) == expected_cells,
        "merged_predictions_bound": trainer["predictions_sha256"]
        == pair_sha256
        and trainer["array_predictions_sha256"] == array_sha256,
        "merged_prediction_rows_target_free": _target_free(
            _read_jsonl(args.merged_pair_predictions)
        )
        and _target_free(_read_jsonl(args.merged_array_predictions)),
        "selection_sources_bound": selection[
            "source_trainer_summary_sha256"
        ]
        == sha256_file(args.merged_trainer_summary)
        and selection["source_pair_predictions_sha256"] == pair_sha256
        and selection["source_array_predictions_sha256"] == array_sha256,
        "selection_frozen_without_confirmation_labels": bool(
            selection["selection_frozen_before_confirmation_join"]
        )
        and not selection["confirmation_labels_read"],
        "selection_reproduced": selection[
            "confirmation_join_authorized"
        ]
        == reproduced["authorized"]
        and selection["primary_candidate_id"] == reproduced["primary_id"]
        and selection["primary_seed"] == reproduced["primary_seed"]
        and selection["comparative_primary_checks"]
        == reproduced["comparative_checks"]
        and {
            candidate_id: int(row["selected_seed"])
            for candidate_id, row in selection[
                "candidate_selection"
            ].items()
        }
        == reproduced["selected_seeds"],
    }
    resource_valid, resource_receipts = _resource_receipts_valid(
        config,
        config_sha256,
        manifest_sha256,
        plan_sha256,
        args.shard_root,
    )
    checks["resource_and_cleanup_receipts_pass"] = resource_valid

    primary_files = {
        "pairs": args.selected_dir
        / "heldout_pair_predictions.primary.jsonl",
        "arrays": args.selected_dir
        / "heldout_array_predictions.primary.jsonl",
        "summary": args.selected_dir / "trainer_summary.primary.json",
    }
    if reproduced["authorized"]:
        packet = selection["confirmation_packet"]
        if packet is None or not all(path.is_file() for path in primary_files.values()):
            packet_valid = False
        else:
            primary_trainer = load_structured(primary_files["summary"])
            primary_pairs = _read_jsonl(primary_files["pairs"])
            primary_arrays = _read_jsonl(primary_files["arrays"])
            packet_valid = (
                packet["candidate_id"] == reproduced["primary_id"]
                and int(packet["seed"]) == reproduced["primary_seed"]
                and int(packet["candidate_count"]) == 1
                and not packet["private_labels_read"]
                and packet["pair_predictions_sha256"]
                == sha256_file(primary_files["pairs"])
                and packet["array_predictions_sha256"]
                == sha256_file(primary_files["arrays"])
                and packet["trainer_summary_sha256"]
                == sha256_file(primary_files["summary"])
                and primary_trainer["confirmation_packet_candidate_count"]
                == 1
                and not primary_trainer["heldout_pair_targets_seen"]
                and not primary_trainer["heldout_array_targets_seen"]
                and len(primary_trainer["candidate_receipts"]) == 1
                and primary_trainer["candidate_receipts"][0]["candidate_id"]
                == reproduced["primary_id"]
                and int(primary_trainer["candidate_receipts"][0]["seed"])
                == reproduced["primary_seed"]
                and len(primary_pairs)
                == int(manifest["splits"]["heldout_pairs"]["count"])
                and len(primary_arrays)
                == int(manifest["splits"]["heldout_arrays"]["count"])
                and _target_free(primary_pairs)
                and _target_free(primary_arrays)
                and all(
                    row["authority"] == "MODEL_RECOMMENDATION_ONLY"
                    and row["candidate_id"] == reproduced["primary_id"]
                    and int(row["seed"]) == reproduced["primary_seed"]
                    and row["dataset_manifest_sha256"] == manifest_sha256
                    for row in primary_pairs + primary_arrays
                )
            )
    else:
        packet_valid = (
            selection["confirmation_packet"] is None
            and selection["primary_candidate_id"] is None
            and selection["primary_seed"] is None
            and selection["no_qualified_primary"]
            and not any(path.exists() for path in primary_files.values())
        )
    checks["primary_packet_or_development_stop_consistent"] = packet_valid

    integrity_passed = all(checks.values())
    authorized = integrity_passed and reproduced["authorized"]
    report = {
        "schema_version": "ldt-adapter-receipt-packet-audit.v16",
        "passed": integrity_passed,
        "checks": checks,
        "authoritative_labels_read": False,
        "label_paths_accepted_by_interface": False,
        "reproduced_development_selection": reproduced,
        "resource_receipts": resource_receipts,
        "heldout_join_authorized": authorized,
        "heldout_join_budget": 1 if authorized else 0,
        "operational_decision": (
            "authorize_single_environment_join"
            if authorized
            else (
                "stop_without_join_development_gate_not_met"
                if integrity_passed
                else "reject_packet_integrity_failure"
            )
        ),
        "bindings": {
            "preregistration_sha256": config_sha256,
            "dataset_manifest_sha256": manifest_sha256,
            "generation_report_sha256": sha256_file(
                args.generation_report
            ),
            "trainer_plan_sha256": plan_sha256,
            "merged_trainer_summary_sha256": sha256_file(
                args.merged_trainer_summary
            ),
            "merged_pair_predictions_sha256": pair_sha256,
            "merged_array_predictions_sha256": array_sha256,
            "selection_receipt_sha256": sha256_file(
                args.selection_receipt
            ),
        },
    }
    dump_structured(args.out, report)
    if not integrity_passed:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"v16 packet integrity audit failed: {failed}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the v16 development packet without accepting label paths"
        )
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--generation-report", type=Path, required=True)
    parser.add_argument("--trainer-plan", type=Path, required=True)
    parser.add_argument("--merged-trainer-summary", type=Path, required=True)
    parser.add_argument("--merged-pair-predictions", type=Path, required=True)
    parser.add_argument("--merged-array-predictions", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--selected-dir", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    result = audit(build_parser().parse_args())
    print(json.dumps(result, sort_keys=True))
