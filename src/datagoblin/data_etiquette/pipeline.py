from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.domain.evidence import EvidenceType
from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import dump_structured, load_structured

from .evidence_receipts import (
    EvidenceReceiptStore,
    EvidenceVerificationReceipt,
    EvidenceVerifierRegistry,
    issue_evidence_receipt,
)
from .policy import EtiquettePolicy
from .records import CollectionDisposition, DataCollectionRecord, EtiquetteDecision
from .sorter import DeterministicMockLDTSorter

CATEGORIES = (
    "clean_verified",
    "model_generated",
    "secret",
    "hidden_reasoning",
    "checkpoint",
    "evaluation_material",
    "sealed_evaluation",
    "unknown_license",
    "consent_denied",
    "unredacted_personal",
    "unknown_lawful_basis",
    "deletion_requested",
    "missing_provenance",
    "cross_split_duplicate",
)
SPLITS = ("train", "calibration", "ordinary_test", "ood_test", "sealed_poison")


def run_etiquette_demo(policy_path: Path, out_dir: Path, *, seed: int = 7) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = policy_path.parent / "evidence_verifiers.yaml"
    registry = EvidenceVerifierRegistry(registry_path)
    provisional_records = tuple(
        _make_record(category, split, seed, index)
        for index, (split, category) in enumerate((split, category) for split in SPLITS for category in CATEGORIES)
    )
    records, receipts = _bind_demo_evidence(provisional_records, registry)
    policy = EtiquettePolicy(policy_path, evidence_store=EvidenceReceiptStore(registry, receipts))
    shutil.copyfile(policy_path, out_dir / "frozen_policy.yaml")
    steward_dir = out_dir / "private_steward"
    steward_dir.mkdir()
    _write_evidence_receipts(steward_dir, registry_path, receipts)
    decisions = policy.decide_batch(records)
    dataset_dir = out_dir / "public_dataset"
    dataset_dir.mkdir()
    split_entries: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = dataset_dir / f"{split}.jsonl"
        selected = [
            (record, decision)
            for record, decision in zip(records, decisions, strict=True)
            if record.intended_split == split
        ]
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record, decision in selected:
                row = {
                    "schema_version": "1.0",
                    "item_id": record.item_id,
                    "split": split,
                    "category": record.public_metadata["category"],
                    "record": record.to_jsonable(),
                    "record_hash": sha256_json(record),
                }
                if split != "sealed_poison":
                    row["authoritative_label"] = decision.to_jsonable()
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        split_entries[split] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "count": len(selected),
            "labels_in_file": split != "sealed_poison",
        }
    manifest = {
        "schema_version": "1.0",
        "dataset_id": f"data-etiquette-v1-{seed}-{policy.hash[:12]}",
        "seed": seed,
        "policy_hash": policy.hash,
        "content_policy": "public metadata and content hashes only; no raw bytes or filesystem paths",
        "splits": split_entries,
        "categories": list(CATEGORIES),
        "sealed_policy": "labels withheld until predictions are complete",
    }
    manifest_path = dataset_dir / "dataset_manifest.json"
    dump_structured(manifest_path, manifest)
    sorter = DeterministicMockLDTSorter()
    predictions_path = out_dir / "sorter_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(sorter.sort(record), sort_keys=True, separators=(",", ":")) + "\n")
    sealed_labels_path = steward_dir / "sealed_labels.jsonl"
    with sealed_labels_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record, decision in zip(records, decisions, strict=True):
            if record.intended_split == "sealed_poison":
                handle.write(
                    json.dumps(
                        {
                            "item_id": record.item_id,
                            "record_hash": sha256_json(record),
                            "authoritative_label": decision.to_jsonable(),
                            "steward_id": "independent-etiquette-policy-steward-v1",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    report = _evaluate(records, decisions, predictions_path, policy, manifest_path, sealed_labels_path, out_dir)
    _write_handoff(records, decisions, policy, manifest_path, report, out_dir / "larger_model_handoff")
    _write_training_plan(policy, out_dir)
    return report


def build_etiquette_training_dataset(policy_path: Path, out_dir: Path, *, seed: int = 7) -> dict[str, Any]:
    """Build ML partitions independently from each item's enterprise-use disposition."""
    out_dir.mkdir(parents=True, exist_ok=True)
    indexed_records: list[tuple[str, DataCollectionRecord]] = []
    index = 0
    for outer_split in SPLITS:
        for category in CATEGORIES:
            operational_split = "train"
            if category == "evaluation_material":
                operational_split = "evaluation"
            elif category == "sealed_evaluation":
                operational_split = "sealed_poison"
            record = _make_record(category, operational_split, seed, index)
            if record.evaluation_material or record.sealed_evaluation:
                record = replace(record, allowed_uses=("EVALUATION",))
            record = replace(
                record,
                item_id=f"etq-fit-{seed}-{outer_split}-{index:04d}",
                source_kind=(
                    "held_out_enterprise_source"
                    if outer_split == "ood_test"
                    else "synthetic_research_episode"
                ),
                public_metadata={
                    "category": category,
                    "outer_split": outer_split,
                    "raw_content_present": False,
                },
            )
            indexed_records.append((outer_split, record))
            index += 1
    registry_path = policy_path.parent / "evidence_verifiers.yaml"
    registry = EvidenceVerifierRegistry(registry_path)
    records, receipts = _bind_demo_evidence(
        tuple(record for _, record in indexed_records),
        registry,
    )
    indexed_records = [
        (outer_split, record)
        for (outer_split, _), record in zip(indexed_records, records, strict=True)
    ]
    policy = EtiquettePolicy(policy_path, evidence_store=EvidenceReceiptStore(registry, receipts))
    decisions = policy.decide_batch(
        records,
        dataset_partitions=tuple(outer_split for outer_split, _ in indexed_records),
    )
    split_entries: dict[str, dict[str, Any]] = {}
    for outer_split in SPLITS:
        path = out_dir / f"{outer_split}.jsonl"
        selected = [
            (record, decision)
            for (row_split, record), decision in zip(indexed_records, decisions, strict=True)
            if row_split == outer_split
        ]
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record, decision in selected:
                row = {
                    "schema_version": "1.0",
                    "item_id": record.item_id,
                    "split": outer_split,
                    "category": record.public_metadata["category"],
                    "record": record.to_jsonable(),
                    "record_hash": sha256_json(record),
                }
                if outer_split != "sealed_poison":
                    row["authoritative_label"] = decision.to_jsonable()
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        split_entries[outer_split] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "count": len(selected),
            "labels_in_file": outer_split != "sealed_poison",
        }
    manifest = {
        "schema_version": "1.0",
        "dataset_id": f"data-etiquette-fit-v2-{seed}-{policy.hash[:12]}",
        "seed": seed,
        "policy_hash": policy.hash,
        "content_policy": "public metadata and content hashes only; no raw bytes or filesystem paths",
        "split_contract": "outer ML partitions are independent of record.intended_split enterprise use",
        "splits": split_entries,
        "categories": list(CATEGORIES),
        "sealed_policy": "outer sealed labels withheld until predictions are complete",
    }
    manifest_path = out_dir / "dataset_manifest.json"
    dump_structured(manifest_path, manifest)
    steward_dir = out_dir.parent / "private_steward"
    steward_dir.mkdir()
    _write_evidence_receipts(steward_dir, registry_path, receipts)
    sealed_path = steward_dir / "sealed_labels.jsonl"
    with sealed_path.open("w", encoding="utf-8", newline="\n") as handle:
        for (outer_split, record), decision in zip(indexed_records, decisions, strict=True):
            if outer_split == "sealed_poison":
                handle.write(
                    json.dumps(
                        {
                            "item_id": record.item_id,
                            "record_hash": sha256_json(record),
                            "authoritative_label": decision.to_jsonable(),
                            "steward_id": "independent-etiquette-policy-steward-v1",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    return manifest


def evaluate_sorter_predictions(
    dataset_manifest_path: Path,
    predictions_path: Path,
    sealed_labels_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Independently score a trained sorter without exposing sealed labels to its workspace."""
    manifest = load_structured(dataset_manifest_path)
    records: list[tuple[str, DataCollectionRecord]] = []
    labels: dict[str, dict[str, Any]] = {}
    sealed_item_ids: set[str] = set()
    for split_name, split in manifest["splits"].items():
        split_path = dataset_manifest_path.parent / str(split["path"])
        if sha256_file(split_path) != split["sha256"]:
            raise ValueError(f"split hash mismatch: {split_name}")
        for line in split_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            record = DataCollectionRecord.from_jsonable(row["record"])
            if row["record_hash"] != sha256_json(record):
                raise ValueError(f"record hash mismatch: {record.item_id}")
            if split_name == "sealed_poison":
                if "authoritative_label" in row:
                    raise ValueError("sealed label leaked into public dataset")
                sealed_item_ids.add(record.item_id)
            else:
                labels[record.item_id] = row["authoritative_label"]
            records.append((split_name, record))
    record_by_id = {record.item_id: record for _, record in records}
    sealed_count = 0
    for line in sealed_labels_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        sealed = json.loads(line)
        item_id = str(sealed["item_id"])
        if item_id not in record_by_id:
            raise ValueError(f"unknown sealed label item: {item_id}")
        record = record_by_id[item_id]
        if item_id not in sealed_item_ids:
            raise ValueError("sealed label file contains a non-sealed item")
        if sealed["record_hash"] != sha256_json(record):
            raise ValueError(f"sealed label record hash mismatch for {item_id}")
        labels[item_id] = sealed["authoritative_label"]
        sealed_count += 1
    expected_sealed = len(sealed_item_ids)
    if sealed_count != expected_sealed:
        raise ValueError("sealed label file is incomplete")
    predictions = {
        row["item_id"]: row
        for row in (json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line)
    }
    if set(predictions) != set(record_by_id):
        raise ValueError("prediction item set does not match dataset")
    rows: list[dict[str, Any]] = []
    sorter_ids: set[str] = set()
    for outer_split, record in records:
        prediction = predictions[record.item_id]
        sorter_ids.add(str(prediction["sorter_id"]))
        if prediction["record_hash"] != sha256_json(record):
            raise ValueError(f"prediction record hash mismatch for {record.item_id}")
        if prediction.get("authority") != "MODEL_RECOMMENDATION_ONLY":
            raise ValueError("trained sorter asserted forbidden label authority")
        label = labels[record.item_id]
        rows.append(
            {
                "item_id": record.item_id,
                "split": outer_split,
                "category": record.public_metadata["category"],
                "actual": label["disposition"],
                "predicted": prediction["recommended_disposition"],
                "correct": prediction["recommended_disposition"] == label["disposition"],
                "confidence": float(prediction["confidence"]),
                "abstained": bool(prediction["abstained"]),
                "actual_flags": set(label["violations"]),
                "predicted_flags": set(prediction["flags"]),
            }
        )
    if len(sorter_ids) != 1:
        raise ValueError("predictions mix multiple sorter IDs")
    out_dir.mkdir(parents=True, exist_ok=True)
    split_results = {split: _score([row for row in rows if row["split"] == split]) for split in SPLITS}
    disagreements = [row for row in rows if not row["correct"] or row["abstained"]]
    queue_path = out_dir / "adjudication_queue.jsonl"
    with queue_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in disagreements:
            handle.write(
                json.dumps(
                    {
                        "item_id": row["item_id"],
                        "split": row["split"],
                        "category": row["category"],
                        "sorter_recommendation": row["predicted"],
                        "sorter_confidence": row["confidence"],
                        "sorter_abstained": row["abstained"],
                        "reason": "sorter_error_or_abstention",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    promotion_checks = {
        "sealed_critical_false_release_zero": (
            split_results["sealed_poison"]["critical_false_release_rate"] == 0.0
        ),
        "ordinary_critical_false_release_zero": (
            split_results["ordinary_test"]["critical_false_release_rate"] == 0.0
        ),
        "ood_critical_false_release_zero": split_results["ood_test"]["critical_false_release_rate"] == 0.0,
        "independent_steward_signoff": False,
        "resource_and_cleanup_receipts_verified": False,
    }
    report = {
        "schema_version": "1.0",
        "sorter_id": next(iter(sorter_ids)),
        "evidence_status": "trained_candidate_quality_evidence_pending_resource_and_steward_receipts",
        "policy_hash": manifest["policy_hash"],
        "dataset_id": manifest["dataset_id"],
        "split_results": split_results,
        "adjudication_queue_count": len(disagreements),
        "sorter_authority": "recommendation_only",
        "sealed_labels_sha256": sha256_file(sealed_labels_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "predictions_sha256": sha256_file(predictions_path),
        "promotion_checks": promotion_checks,
        "promotion_eligible": all(promotion_checks.values()),
        "claim_boundary": "Learned proposal-head evidence only; release remains an independent policy decision.",
    }
    dump_structured(out_dir / "trained_sorter_evaluation.json", report)
    return report


def audit_sorter_candidate(
    evaluation_report_path: Path,
    run_receipt_path: Path,
    policy_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Bind independent quality evidence to environment-owned execution receipts."""
    report = load_structured(evaluation_report_path)
    receipt = load_structured(run_receipt_path)
    policy = EtiquettePolicy(policy_path)
    trainer = receipt["trainer_receipt"]
    caps = receipt["caps_receipt"]
    cleanup = receipt["cleanup_receipt"]
    evaluation_splits = ("ordinary_test", "ood_test", "sealed_poison")
    gates = policy.config["sorter_training"]["promotion_gates"]
    audit_thresholds = {
        "minimum_evaluation_coverage": 0.5,
        "minimum_evaluation_selective_accuracy": 0.85,
        "minimum_evaluation_flag_micro_f1": 0.8,
    }
    receipt_checks = {
        "policy_hash_matches": report["policy_hash"] == policy.hash,
        "dataset_hash_matches_trainer": (
            report["dataset_manifest_sha256"] == trainer["dataset_manifest_sha256"]
        ),
        "predictions_hash_matches_trainer": report["predictions_sha256"] == trainer["predictions_sha256"],
        "cgroup_caps_enforced": bool(caps["caps_enforced"]),
        "capped_process_completed": receipt["capped_process_exit_code"] == 0 and caps["status"] == "completed",
        "trainer_completed": trainer["status"] == "completed" and trainer["steps_completed"] > 0,
        "checkpoints_present": len(trainer["checkpoints"]) > 0,
        "cross_split_fit_rows_excluded": trainer["cross_split_rows_excluded_from_fit"] > 0,
        "wsl_stage_removed": receipt["wsl_stage_cleanup_exit_code"] == 0,
        "cleanup_passed": bool(cleanup["cleanup_passed"]),
        "no_lingering_owned_processes": not cleanup["lingering_owned_pids"],
        "no_lingering_gpu_apps": not cleanup["gpu_after"]["compute_apps"],
        "critical_false_release_zero": all(
            report["split_results"][split]["critical_false_release_rate"]
            <= gates["sealed_critical_false_release_rate"]
            for split in evaluation_splits
        ),
    }
    post_hoc_diagnostics = {
        "minimum_coverage": min(
            report["split_results"][split]["coverage"] for split in evaluation_splits
        )
        >= audit_thresholds["minimum_evaluation_coverage"],
        "minimum_selective_accuracy": min(
            report["split_results"][split]["selective_accuracy"] for split in evaluation_splits
        )
        >= audit_thresholds["minimum_evaluation_selective_accuracy"],
        "minimum_flag_micro_f1": min(
            report["split_results"][split]["flag_micro_f1"] for split in evaluation_splits
        )
        >= audit_thresholds["minimum_evaluation_flag_micro_f1"],
    }
    evidence_checks = {
        "selection_thresholds_preregistered": False,
        "held_out_distribution_confirmation_complete": False,
        "independent_steward_signoff": False,
    }
    promoted = all(receipt_checks.values()) and all(evidence_checks.values())
    decision = {
        "schema_version": "1.0",
        "candidate_id": report["sorter_id"],
        "decision": "PROMOTE_RECOMMENDATION_ONLY" if promoted else "HOLD_FOR_PREREGISTERED_CONFIRMATION",
        "promotion_eligible": promoted,
        "authority_granted": "MODEL_RECOMMENDATION_ONLY" if promoted else "NONE",
        "receipt_and_safety_checks": receipt_checks,
        "evidence_checks": evidence_checks,
        "post_hoc_diagnostic_thresholds": audit_thresholds,
        "post_hoc_diagnostic_checks": post_hoc_diagnostics,
        "evaluation_report_sha256": sha256_file(evaluation_report_path),
        "run_receipt_sha256": sha256_file(run_receipt_path),
        "policy_hash": policy.hash,
        "verified_by": "datagoblin-environment-candidate-auditor-v1",
        "larger_model_training_permitted": False,
        "larger_model_blockers": [
            "released handoff contains hashes and receipts, not steward-materialized training content",
            "no approved local larger-model checkpoint or architecture recipe",
            "no separate hard-cap plan sized for the larger model",
            "sorter requires preregistered held-out distribution confirmation",
        ],
        "claim_boundary": (
            "Any future promotion permits triage recommendations only; deterministic policy and independent "
            "stewardship remain the training-data authorities."
        ),
    }
    decision["decision_receipt_hash"] = sha256_json(decision)
    dump_structured(out_path, decision)
    return decision


def _make_record(category: str, split: str, seed: int, index: int) -> DataCollectionRecord:
    content_sha256 = sha256_json({"synthetic_metadata_only": True, "seed": seed, "index": index})
    values: dict[str, Any] = {
        "item_id": f"etq-{seed}-{split}-{index:04d}",
        "schema_version": "1.0",
        "content_sha256": content_sha256,
        "source_kind": "held_out_enterprise_source" if split == "ood_test" else "synthetic_research_episode",
        "purpose_id": "research-corpus-etiquette-v1",
        "lawful_basis": "SYNTHETIC_NOT_PERSONAL",
        "allowed_uses": ("EVALUATION",) if split in {"ordinary_test", "ood_test", "sealed_poison"} else ("TRAINING",),
        "data_residency_region": "LOCAL_OFFLINE_DEMO",
        "sensitivity_class": "PUBLIC_METADATA_ONLY",
        "license_status": "ALLOWED",
        "consent_status": "NOT_REQUIRED",
        "personal_data_status": "NONE",
        "redacted": True,
        "contains_secret": False,
        "contains_hidden_reasoning": False,
        "contains_model_weight": False,
        "evaluation_material": split in {"ordinary_test", "ood_test"},
        "sealed_evaluation": split == "sealed_poison",
        "model_generated": False,
        "generator_id": None,
        "evidence_type": EvidenceType.INDEPENDENT_HARNESS,
        "asserted_by": "synthetic-etiquette-fixture-v1",
        "verified_by": None,
        "verification_receipt_sha256": None,
        "verification_subject_sha256": None,
        "verification_custody": "NONE",
        "provenance_refs": (f"synthetic://etiquette/{seed}/{index}",),
        "dedup_group": f"dedup-{seed}-{split}-{index}",
        "intended_split": split,
        "retention_days": 365,
        "deletion_requested": False,
        "public_metadata": {"category": category, "raw_content_present": False},
    }
    if category == "model_generated":
        values.update(model_generated=True, generator_id="mock-generator", evidence_type=EvidenceType.MODEL_INFERRED)
    elif category == "secret":
        values["contains_secret"] = True
    elif category == "hidden_reasoning":
        values["contains_hidden_reasoning"] = True
    elif category == "checkpoint":
        values["contains_model_weight"] = True
    elif category == "evaluation_material":
        values["evaluation_material"] = True
    elif category == "sealed_evaluation":
        values.update(evaluation_material=True, sealed_evaluation=True)
    elif category == "unknown_license":
        values["license_status"] = "UNKNOWN"
    elif category == "consent_denied":
        values["consent_status"] = "DENIED"
    elif category == "unredacted_personal":
        values.update(personal_data_status="DIRECT_IDENTIFIER", redacted=False)
    elif category == "unknown_lawful_basis":
        values["lawful_basis"] = "UNKNOWN"
    elif category == "deletion_requested":
        values["deletion_requested"] = True
    elif category == "missing_provenance":
        values["provenance_refs"] = ()
    elif category == "cross_split_duplicate":
        values["dedup_group"] = f"cross-split-{seed}"
    return DataCollectionRecord(**values)


def _bind_demo_evidence(
    records: tuple[DataCollectionRecord, ...],
    registry: EvidenceVerifierRegistry,
) -> tuple[tuple[DataCollectionRecord, ...], tuple[EvidenceVerificationReceipt, ...]]:
    bound: list[DataCollectionRecord] = []
    receipts: list[EvidenceVerificationReceipt] = []
    for record in records:
        if not record.evidence_type.may_hard_promote:
            bound.append(record)
            continue
        category = str(record.public_metadata["category"])
        claim_types = (
            ("CONTENT_INTEGRITY", "POLICY_ELIGIBILITY")
            if category in {"clean_verified", "cross_split_duplicate"}
            else ("CONTENT_INTEGRITY",)
        )
        receipt = issue_evidence_receipt(
            registry,
            subject_sha256=record.content_sha256,
            evidence_type=record.evidence_type,
            verifier_id="independent-etiquette-policy-steward-v2",
            method_id="synthetic-metadata-exact-verifier-v1",
            asserted_by=record.asserted_by,
            verdict="PASS",
            challenge_sha256=sha256_json(
                {
                    "registry_sha256": registry.sha256,
                    "item_id": record.item_id,
                    "purpose_id": record.purpose_id,
                }
            ),
            observation_sha256=sha256_json(
                {
                    "subject_sha256": record.content_sha256,
                    "category": record.public_metadata["category"],
                    "synthetic_metadata_only": True,
                }
            ),
            claim_types=claim_types,
            artifact_sha256s=(
                sha256_json(
                    {
                        "fixture_item_id": record.item_id,
                        "subject_sha256": record.content_sha256,
                        "method": "synthetic-metadata-exact-verifier-v1",
                    }
                ),
            ),
        )
        receipts.append(receipt)
        bound.append(
            replace(
                record,
                verified_by=receipt.verifier_id,
                verification_receipt_sha256=receipt.receipt_sha256,
                verification_subject_sha256=receipt.subject_sha256,
                verification_custody=receipt.custody,
            )
        )
    return tuple(bound), tuple(receipts)


def _write_evidence_receipts(
    steward_dir: Path,
    registry_path: Path,
    receipts: tuple[EvidenceVerificationReceipt, ...],
) -> None:
    shutil.copyfile(registry_path, steward_dir / "frozen_evidence_verifiers.yaml")
    path = steward_dir / "evidence_receipts.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for receipt in sorted(receipts, key=lambda value: value.receipt_sha256):
            handle.write(json.dumps(receipt.to_jsonable(), sort_keys=True, separators=(",", ":")) + "\n")


def _evaluate(
    records: tuple[DataCollectionRecord, ...],
    decisions: tuple[EtiquetteDecision, ...],
    predictions_path: Path,
    policy: EtiquettePolicy,
    manifest_path: Path,
    sealed_labels_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    predictions = {
        row["item_id"]: row
        for row in (json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line)
    }
    labels = {
        decision.item_id: decision.to_jsonable()
        for record, decision in zip(records, decisions, strict=True)
        if record.intended_split != "sealed_poison"
    }
    record_by_id = {record.item_id: record for record in records}
    sealed_count = 0
    for line in sealed_labels_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        sealed = json.loads(line)
        item_id = str(sealed["item_id"])
        record = record_by_id[item_id]
        if record.intended_split != "sealed_poison":
            raise ValueError("sealed label file contains a non-sealed item")
        if sealed["record_hash"] != sha256_json(record):
            raise ValueError(f"sealed label record hash mismatch for {item_id}")
        labels[item_id] = sealed["authoritative_label"]
        sealed_count += 1
    expected_sealed = sum(record.intended_split == "sealed_poison" for record in records)
    if sealed_count != expected_sealed:
        raise ValueError("sealed label file is incomplete")
    rows = []
    for record in records:
        label = labels[record.item_id]
        prediction = predictions[record.item_id]
        correct = prediction["recommended_disposition"] == label["disposition"]
        rows.append(
            {
                "item_id": record.item_id,
                "split": record.intended_split,
                "category": record.public_metadata["category"],
                "actual": label["disposition"],
                "predicted": prediction["recommended_disposition"],
                "correct": correct,
                "confidence": float(prediction["confidence"]),
                "actual_flags": set(label["violations"]),
                "predicted_flags": set(prediction["flags"]),
            }
        )
    split_results = {split: _score([row for row in rows if row["split"] == split]) for split in SPLITS}
    disagreements = [row for row in rows if not row["correct"] or row["confidence"] < 0.75]
    queue_path = out_dir / "adjudication_queue.jsonl"
    with queue_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in disagreements:
            handle.write(
                json.dumps(
                    {
                        "item_id": row["item_id"],
                        "split": row["split"],
                        "category": row["category"],
                        "sorter_recommendation": row["predicted"],
                        "sorter_confidence": row["confidence"],
                        "reason": "sorter_policy_disagreement_or_low_confidence",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    report = {
        "schema_version": "1.0",
        "sorter_id": "deterministic-mock-ldt-etiquette-v1",
        "evidence_status": "pipeline_smoke_only_not_trained_ldt_evidence",
        "policy_version": policy.version,
        "policy_hash": policy.hash,
        "split_results": split_results,
        "adjudication_queue_count": len(disagreements),
        "sorter_authority": "recommendation_only",
        "sealed_labels_sha256": sha256_file(sealed_labels_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "predictions_sha256": sha256_file(predictions_path),
        "claim_boundary": "The deterministic mock validates sorting and stewardship plumbing; no LDT was trained.",
    }
    dump_structured(out_dir / "sorter_evaluation.json", report)
    return report


def _score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual = [row["actual"] for row in rows]
    predicted = [row["predicted"] for row in rows]
    correct = [float(row["correct"]) for row in rows]
    confidence = [row["confidence"] for row in rows]
    unsafe_actual = {
        CollectionDisposition.REJECT.value,
        CollectionDisposition.QUARANTINE.value,
        CollectionDisposition.EVALUATION_ONLY.value,
        CollectionDisposition.SEALED_EVALUATION.value,
    }
    unsafe_indices = [index for index, value in enumerate(actual) if value in unsafe_actual]
    released = [
        predicted[index] == CollectionDisposition.VERIFIED_TRAIN.value and not rows[index].get("abstained", False)
        for index in unsafe_indices
    ]
    false_release = sum(released)
    tp = sum(len(row["actual_flags"] & row["predicted_flags"]) for row in rows)
    fp = sum(len(row["predicted_flags"] - row["actual_flags"]) for row in rows)
    fn = sum(len(row["actual_flags"] - row["predicted_flags"]) for row in rows)
    return {
        "examples": len(rows),
        "disposition_accuracy": float(np.mean(correct)),
        "abstention_rate": sum(bool(row.get("abstained", False)) for row in rows) / len(rows) if rows else 0.0,
        "coverage": sum(not bool(row.get("abstained", False)) for row in rows) / len(rows) if rows else 0.0,
        "selective_accuracy": (
            sum(row["correct"] for row in rows if not row.get("abstained", False))
            / sum(not row.get("abstained", False) for row in rows)
            if any(not row.get("abstained", False) for row in rows)
            else 0.0
        ),
        "flag_micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "brier_score": float(np.mean((np.asarray(confidence) - np.asarray(correct)) ** 2)),
        "critical_false_release_rate": false_release / len(unsafe_indices) if unsafe_indices else 0.0,
        "disposition_counts": dict(Counter(actual)),
    }


def _write_handoff(
    records: tuple[DataCollectionRecord, ...],
    decisions: tuple[EtiquetteDecision, ...],
    policy: EtiquettePolicy,
    manifest_path: Path,
    report: dict[str, Any],
    out_dir: Path,
) -> None:
    out_dir.mkdir()
    lanes: dict[str, list[dict[str, Any]]] = {"verified_supervision": [], "soft_discovery": []}
    evaluation_commitments: list[str] = []
    for record, decision in zip(records, decisions, strict=True):
        entry = {
            "item_id": record.item_id,
            "content_sha256": record.content_sha256,
            "policy_receipt_hash": decision.decision_receipt_hash,
            "provenance_refs": list(record.provenance_refs),
            "dedup_group": record.dedup_group,
        }
        if decision.disposition == CollectionDisposition.VERIFIED_TRAIN:
            lanes["verified_supervision"].append(entry)
        elif decision.disposition == CollectionDisposition.DISCOVERY_ONLY:
            lanes["soft_discovery"].append(entry)
        elif decision.disposition in {
            CollectionDisposition.EVALUATION_ONLY,
            CollectionDisposition.SEALED_EVALUATION,
        }:
            evaluation_commitments.append(sha256_json(entry))
    artifacts: dict[str, dict[str, Any]] = {}
    for lane, entries in lanes.items():
        path = out_dir / f"{lane}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
        artifacts[lane] = {"path": path.name, "sha256": sha256_file(path), "count": len(entries)}
    handoff = {
        "schema_version": "1.0",
        "handoff_id": f"larger-model-corpus-{policy.hash[:12]}",
        "policy_version": policy.version,
        "policy_hash": policy.hash,
        "source_dataset_manifest_sha256": sha256_file(manifest_path),
        "sorter_evaluation_hash": sha256_json(report),
        "sorter_output_is_authority": False,
        "released_lanes": artifacts,
        "evaluation_item_count": len(evaluation_commitments),
        "evaluation_commitment_hash": sha256_json(evaluation_commitments),
        "evaluation_items_exported_to_training": False,
        "raw_content_in_manifest": False,
        "dedup_audit": "batch policy quarantined groups crossing training and evaluation roles",
        "deletion_audit": "deletion-requested records excluded; private deletion requires operator receipt",
        "larger_model_training_permitted": False,
        "next_gate": "independent steward materializes authorized content and approves a separate capped trainer plan",
    }
    dump_structured(out_dir / "corpus_handoff_manifest.json", handoff)


def _write_training_plan(policy: EtiquettePolicy, out_dir: Path) -> None:
    plan = dict(policy.config["sorter_training"])
    plan.update(
        {
            "schema_version": "1.0",
            "status": "scaffold_only_no_training_started",
            "weights_loaded": False,
            "network_used": False,
            "abort_is_valid": True,
            "post_run_cleanup": "scripts/post_run_hrm_cleanup.ps1",
        }
    )
    dump_structured(out_dir / "ldt_sorter_trainer_plan.json", plan)
    events = [
        {
            "ts": "example-only",
            "event": "start",
            "run_id": plan["training_task_id"],
            "caps": plan["caps"],
        },
        {
            "ts": "example-only",
            "event": "checkpoint",
            "step": plan["checkpoint_interval"]["steps"],
            "path": "checkpoints/example/step-100",
        },
        {
            "ts": "example-only",
            "event": "abort",
            "reason": "example_cap_breach",
            "weights_loaded": False,
        },
    ]
    events_path = out_dir / "events.example.jsonl"
    with events_path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    dump_structured(
        out_dir / "cleanup_summary.json",
        {
            "schema_version": "1.0",
            "run_id": plan["training_task_id"],
            "mode": "mock_sorter_no_weight_load",
            "owned_pids": [],
            "stopped_owned_pids": [],
            "wsl_action": "not_started",
            "memory_before": {"available_mb": None, "committed_gb": None, "cache_gb": None, "standby_gb": None},
            "memory_after": {"available_mb": None, "committed_gb": None, "cache_gb": None, "standby_gb": None},
            "lingering_owned_pids": [],
            "gpu_before": {"available": False, "compute_apps": []},
            "gpu_after": {"available": False, "compute_apps": []},
            "top_by_private_commit": [],
            "top_by_working_set": [],
            "global_standby_cache_purged": False,
            "memory_audit": "not_sampled_because_no_training_process_started",
            "cleanup_passed": True,
        },
    )
    dump_structured(
        out_dir / "ldt_sorter_summary.json",
        {
            "schema_version": "1.0",
            "run_id": plan["training_task_id"],
            "status": "mock_sorter_evaluation_completed",
            "abort_reason": None,
            "weights_loaded": False,
            "steps_completed": 0,
            "peak_ram_mb": 0.0,
            "avg_ram_mb": 0.0,
            "peak_io_mb_s": 0.0,
            "cpu_pct": 0.0,
            "checkpoints": [],
            "cleanup_passed": True,
            "events_sha256": sha256_file(events_path),
            "trainer_plan_hash": sha256_json(plan),
            "claim_boundary": "Metadata-only plumbing control; no trainable LDT or checkpoint was executed.",
        },
    )
