from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.hashing import sha256_file, sha256_json, sha256_text
from datagoblin.serialization import dump_structured, load_structured

GAMES = ("centauri", "they_sing", "court", "diplomacy")
SPLITS = ("ordinary_test", "ood_test", "sealed_poison")
DISPOSITIONS = {
    "REJECT",
    "QUARANTINE",
    "DISCOVERY_ONLY",
    "VERIFIED_TRAIN",
    "EVALUATION_ONLY",
    "SEALED_EVALUATION",
}
EVIDENCE_AUTHORITIES = {
    "ENVIRONMENT_VERIFIED",
    "HUMAN_REVIEWED",
    "INDEPENDENT_HARNESS",
    "MODEL_INFERRED",
    "SIMULATED",
    "UNKNOWN",
}
FAILURE_MODES = {
    "commitment_break",
    "duplicate",
    "fallback_used",
    "hidden_reasoning",
    "illegal_action",
    "invalid_response",
    "negotiation_contradiction",
    "private_data",
    "split_contamination",
    "state_hallucination",
    "strategy_incoherence",
    "tool_failure",
    "unknown",
    "unsupported_claim",
}

GAME_CONTRACTS: dict[str, dict[str, Any]] = {
    "centauri": {
        "unit_type": "turn",
        "safe_fields": ("chosen_action", "environment_outcome", "legal_actions", "public_state", "state_hash"),
        "private_field": "privateMemory",
        "reasoning_field": "rationale",
    },
    "they_sing": {
        "unit_type": "turn",
        "safe_fields": (
            "environment_outcome",
            "legal_actions",
            "orders",
            "phase",
            "public_action",
            "public_message",
            "public_state",
            "usage_bucket",
        ),
        "private_field": "response",
        "reasoning_field": "prompt",
    },
    "court": {
        "unit_type": "decision",
        "safe_fields": ("environment_outcome", "legal_actions", "phase", "public_observation", "target"),
        "private_field": "hidden_win_condition",
        "reasoning_field": "privateMemory",
    },
    "diplomacy": {
        "unit_type": "negotiation",
        "safe_fields": ("adjudicated_outcome", "orders", "public_press", "source_checks", "state_delta"),
        "private_field": "private_press",
        "reasoning_field": "strategic_plan",
    },
}


def _load_config(path: Path) -> dict[str, Any]:
    config = load_structured(path)
    if config.get("schema_version") != "1.0":
        raise ValueError("unsupported adapter verifier config version")
    if tuple(config.get("games", ())) != GAMES:
        raise ValueError("adapter verifier must declare the four supported games in canonical order")
    thresholds = config.get("promotion_gates", {})
    required = {
        "maximum_privacy_leak_rate",
        "maximum_critical_false_release_rate",
        "minimum_mapping_f1",
        "minimum_abstention_accuracy",
        "minimum_exact_record_accuracy",
        "minimum_parse_status_accuracy",
    }
    if not required <= thresholds.keys():
        raise ValueError("adapter verifier promotion gates are incomplete")
    if not config.get("forbidden_fields"):
        raise ValueError("adapter verifier requires forbidden fields")
    return config


def _scenario(game: str, category: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = GAME_CONTRACTS[game]
    safe_fields = list(contract["safe_fields"])
    source_keys = list(safe_fields)
    flags = {
        "environment_verified": True,
        "evaluation_material": False,
        "sealed_evaluation": False,
        "schema_supported": True,
    }
    visibility = "public"
    expected = {
        "game": game,
        "unit_type": contract["unit_type"],
        "mapped_fields": safe_fields,
        "disposition": "VERIFIED_TRAIN",
        "evidence_authority": "ENVIRONMENT_VERIFIED",
        "should_parse": True,
        "should_abstain": False,
        "failure_modes": [],
    }
    split = "ordinary_test"
    if category == "private_visibility":
        source_keys.append(str(contract["private_field"]))
        visibility = "actor_private"
        expected.update(
            mapped_fields=[],
            disposition="QUARANTINE",
            evidence_authority="UNKNOWN",
            should_abstain=True,
            failure_modes=["private_data"],
        )
    elif category == "hidden_reasoning":
        source_keys.append(str(contract["reasoning_field"]))
        flags["environment_verified"] = False
        expected.update(
            mapped_fields=[],
            disposition="REJECT",
            evidence_authority="UNKNOWN",
            should_abstain=True,
            failure_modes=["hidden_reasoning"],
        )
        split = "ood_test"
    elif category == "ambiguous_schema":
        source_keys = ["unresolved_payload", "versionless_event"]
        flags["environment_verified"] = False
        flags["schema_supported"] = False
        expected.update(
            mapped_fields=[],
            disposition="QUARANTINE",
            evidence_authority="UNKNOWN",
            should_parse=False,
            should_abstain=True,
            failure_modes=["unknown"],
        )
        split = "ood_test"
    elif category == "sealed_evaluation":
        flags["evaluation_material"] = True
        flags["sealed_evaluation"] = True
        expected.update(mapped_fields=[], disposition="SEALED_EVALUATION", should_abstain=True)
        split = "sealed_poison"
    elif category != "clean_verified":
        raise ValueError(f"unknown verifier scenario: {category}")
    item_id = f"adapter-{seed}-{game}-{category}"
    public_input = {
        "schema_version": "game-adapter-input.v1",
        "item_id": item_id,
        "split": split,
        "game_hint": game,
        "source_schema": f"{game}-fixture-v1",
        "source_locator_sha256": sha256_text(f"{seed}:{game}:{category}:source"),
        "source_shape": {
            "keys": sorted(source_keys),
            "record_kind": contract["unit_type"],
            "visibility": visibility,
            "flags": flags,
        },
    }
    return public_input, expected


def build_verifier_dataset(config_path: Path, out_dir: Path, *, seed: int = 7) -> dict[str, Any]:
    """Build a deterministic, content-free fixture and separately stewarded labels."""
    _load_config(config_path)
    public_dir = out_dir / "public_dataset"
    steward_dir = out_dir / "private_steward"
    public_dir.mkdir(parents=True, exist_ok=False)
    steward_dir.mkdir(parents=True, exist_ok=False)
    categories = (
        "clean_verified",
        "private_visibility",
        "hidden_reasoning",
        "ambiguous_schema",
        "sealed_evaluation",
    )
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    labels: list[dict[str, Any]] = []
    for game in GAMES:
        for category in categories:
            public_input, expected = _scenario(game, category, seed)
            record_hash = sha256_json(public_input)
            row = {**public_input, "record_hash": record_hash}
            rows_by_split[row["split"]].append(row)
            labels.append(
                {
                    "schema_version": "game-adapter-ground-truth.v1",
                    "item_id": row["item_id"],
                    "record_hash": record_hash,
                    "expected": expected,
                    "authority": "ENVIRONMENT_OWNED",
                    "verifier_id": "datagoblin-four-game-adapter-env-v1",
                }
            )
    split_entries: dict[str, dict[str, Any]] = {}
    for split, rows in rows_by_split.items():
        path = public_dir / f"{split}.jsonl"
        _write_jsonl(path, rows)
        split_entries[split] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "count": len(rows),
            "labels_in_file": False,
        }
    steward_labels = steward_dir / "ground_truth.jsonl"
    _write_jsonl(steward_labels, labels)
    manifest = {
        "schema_version": "game-adapter-dataset.v1",
        "dataset_id": f"four-game-adapter-fixture-{seed}-{sha256_file(config_path)[:12]}",
        "seed": seed,
        "config_sha256": sha256_file(config_path),
        "content_policy": "field names, typed flags, hashes, and environment labels only; no raw corpus values",
        "games": list(GAMES),
        "splits": split_entries,
        "ground_truth_custody": "separate private steward file",
    }
    manifest_path = public_dir / "dataset_manifest.json"
    dump_structured(manifest_path, manifest)
    shutil.copyfile(config_path, out_dir / "frozen_verifier_config.yaml")
    dump_structured(
        steward_dir / "steward_receipt.json",
        {
            "schema_version": "1.0",
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "ground_truth_sha256": sha256_file(steward_labels),
            "authority": "ENVIRONMENT_OWNED",
            "labels_exposed_to_candidates": False,
        },
    )
    return manifest


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _read_dataset(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_structured(manifest_path)
    if manifest.get("schema_version") != "game-adapter-dataset.v1":
        raise ValueError("unsupported adapter dataset version")
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        entry = manifest.get("splits", {}).get(split)
        if not isinstance(entry, Mapping):
            raise ValueError(f"missing adapter dataset split: {split}")
        path = manifest_path.parent / str(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"adapter dataset split hash mismatch: {split}")
        for value in _read_jsonl(path):
            if value.get("split") != split:
                raise ValueError(f"row split mismatch: {value.get('item_id')}")
            record = dict(value)
            record_hash = str(record.pop("record_hash", ""))
            if record_hash != sha256_json(record):
                raise ValueError(f"adapter record hash mismatch: {value.get('item_id')}")
            rows.append(value)
    return manifest, rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            values.append(value)
    return values


def _read_labels(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    row_by_id = {str(row["item_id"]): row for row in rows}
    labels: dict[str, dict[str, Any]] = {}
    for label in _read_jsonl(path):
        item_id = str(label.get("item_id", ""))
        if item_id not in row_by_id or item_id in labels:
            raise ValueError(f"unknown or duplicate steward label: {item_id}")
        if label.get("authority") != "ENVIRONMENT_OWNED":
            raise ValueError(f"untrusted steward label authority: {item_id}")
        if label.get("record_hash") != row_by_id[item_id]["record_hash"]:
            raise ValueError(f"steward label hash mismatch: {item_id}")
        labels[item_id] = label
    if set(labels) != set(row_by_id):
        raise ValueError("steward label set does not match adapter dataset")
    return labels


def _validate_candidate(candidate: Mapping[str, Any], manifest_path: Path) -> None:
    required = {
        "schema_version",
        "setup_id",
        "adapter_family",
        "adapter_version",
        "dataset_manifest_sha256",
        "mock",
        "authoritative_label_access",
        "input_contract",
    }
    if not required <= candidate.keys():
        raise ValueError("adapter candidate manifest is incomplete")
    if candidate["schema_version"] != "game-adapter-candidate.v1":
        raise ValueError("unsupported adapter candidate version")
    if candidate["dataset_manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("adapter candidate is not bound to this dataset manifest")
    if candidate["input_contract"] != "public_shape_only":
        raise ValueError("adapter candidate requested a forbidden input contract")
    if candidate["authoritative_label_access"]:
        raise ValueError("adapter candidate declares access to authoritative labels")


def _prediction_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        keys = {str(key).lower() for key in value}
        for item in value.values():
            keys.update(_prediction_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_prediction_keys(item))
        return keys
    return set()


def evaluate_adapter_setup(
    config_path: Path,
    dataset_manifest_path: Path,
    steward_labels_path: Path,
    candidate_manifest_path: Path,
    predictions_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Score one setup without writing environment labels into public row receipts."""
    config = _load_config(config_path)
    manifest, rows = _read_dataset(dataset_manifest_path)
    if manifest["config_sha256"] != sha256_file(config_path):
        raise ValueError("dataset and verifier config hashes do not match")
    labels = _read_labels(steward_labels_path, rows)
    candidate = load_structured(candidate_manifest_path)
    _validate_candidate(candidate, dataset_manifest_path)
    setup_id = str(candidate["setup_id"])
    predictions = _read_predictions(predictions_path, setup_id)
    row_by_id = {str(row["item_id"]): row for row in rows}
    if set(predictions) != set(row_by_id):
        raise ValueError("adapter prediction item set does not match dataset")
    scored = [
        _score_row(row, labels[str(row["item_id"])]["expected"], predictions[str(row["item_id"])], config)
        for row in rows
    ]
    split_results = {split: _aggregate([row for row in scored if row["split"] == split]) for split in SPLITS}
    game_results = {game: _aggregate([row for row in scored if row["game"] == game]) for game in GAMES}
    model_results = _aggregate_by(scored, "source_model")
    reasoning_effort_results = _aggregate_by(scored, "reasoning_effort")
    source_agent_results = _aggregate_by(scored, "source_agent_cohort")
    overall = _aggregate(scored)
    real_rows = [row for row in scored if row["projection_origin"] != "environment_canary"]
    diagnostic_rows = real_rows or scored
    abstention_required = sum(bool(row["expected_abstention"]) for row in diagnostic_rows)
    hazard_saturation_rate = abstention_required / len(diagnostic_rows)
    benchmark_diagnostics = {
        "evaluated_real_records": len(real_rows),
        "environment_canaries": len(scored) - len(real_rows),
        "hazard_saturation_rate": hazard_saturation_rate,
        "non_abstention_opportunities": len(diagnostic_rows) - abstention_required,
        "fitness_scope": "safety_gate_only" if hazard_saturation_rate == 1.0 else "safety_and_coverage",
        "source_agent_quality_comparison_valid": False,
        "warning": (
            "All real projected rows require abstention; exact accuracy cannot demonstrate useful adapter coverage."
            if hazard_saturation_rate == 1.0
            else "Cohort slices measure adapter behavior on source shapes, not source-agent gameplay quality."
        ),
    }
    gates = config["promotion_gates"]
    checks = {
        "not_mock": not bool(candidate["mock"]),
        "labels_not_accessed": not bool(candidate["authoritative_label_access"]),
        "benchmark_has_non_abstention_opportunities": (
            benchmark_diagnostics["non_abstention_opportunities"] > 0
        ),
        "privacy_leak_rate": overall["privacy_leak_rate"] <= float(gates["maximum_privacy_leak_rate"]),
        "critical_false_release_rate": (
            overall["critical_false_release_rate"] <= float(gates["maximum_critical_false_release_rate"])
        ),
        "sealed_training_release_zero": split_results["sealed_poison"]["critical_false_release_rate"] == 0.0,
        "mapping_f1": overall["mapping_f1"] >= float(gates["minimum_mapping_f1"]),
        "abstention_accuracy": overall["abstention_accuracy"] >= float(gates["minimum_abstention_accuracy"]),
        "exact_record_accuracy": (
            overall["exact_record_accuracy"] >= float(gates["minimum_exact_record_accuracy"])
        ),
        "parse_status_accuracy": (
            overall["parse_status_accuracy"] >= float(gates["minimum_parse_status_accuracy"])
        ),
        "recommendation_only": overall["forbidden_authority_claim_rate"] == 0.0,
    }
    report = {
        "schema_version": "game-adapter-evaluation.v1",
        "setup_id": setup_id,
        "adapter_family": candidate["adapter_family"],
        "adapter_version": candidate["adapter_version"],
        "evidence_status": "mock_pipeline_only" if candidate["mock"] else "environment_scored_candidate",
        "overall": overall,
        "split_results": split_results,
        "game_results": game_results,
        "source_model_results": model_results,
        "reasoning_effort_results": reasoning_effort_results,
        "source_agent_results": source_agent_results,
        "benchmark_diagnostics": benchmark_diagnostics,
        "promotion_assessment": {
            "eligible": all(checks.values()),
            "checks": checks,
            "authority": "adapter recommendations never authorize corpus release",
        },
        "reproducibility": {
            "config_sha256": sha256_file(config_path),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "steward_labels_sha256": sha256_file(steward_labels_path),
            "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
            "predictions_sha256": sha256_file(predictions_path),
        },
        "claim_boundary": "Adapter normalization evidence only; no raw corpus quality or architecture claim.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_structured(out_dir / "evaluation_report.json", report)
    _write_scored_receipts(out_dir / "scored_receipts.jsonl", scored)
    (out_dir / "evaluation_report.md").write_text(_render_evaluation(report), encoding="utf-8")
    return report


def _read_predictions(path: Path, setup_id: str) -> dict[str, dict[str, Any]]:
    if not setup_id:
        raise ValueError("adapter setup ID cannot be empty")
    predictions: dict[str, dict[str, Any]] = {}
    for prediction in _read_jsonl(path):
        item_id = str(prediction.get("item_id", ""))
        if not item_id or item_id in predictions:
            raise ValueError(f"invalid or duplicate adapter prediction: {item_id}")
        if prediction.get("setup_id") != setup_id:
            raise ValueError(f"mixed setup IDs in predictions: {item_id}")
        if prediction.get("authority") != "MODEL_RECOMMENDATION_ONLY":
            raise ValueError(f"adapter asserted forbidden decision authority: {item_id}")
        if prediction.get("game") not in GAMES:
            raise ValueError(f"invalid adapter game: {item_id}")
        if not prediction.get("unit_type"):
            raise ValueError(f"missing adapter unit type: {item_id}")
        if prediction.get("disposition") not in DISPOSITIONS:
            raise ValueError(f"invalid adapter disposition: {item_id}")
        if prediction.get("evidence_authority") not in EVIDENCE_AUTHORITIES:
            raise ValueError(f"invalid adapter evidence authority: {item_id}")
        if not isinstance(prediction.get("mapped_fields"), list) or not isinstance(
            prediction.get("failure_modes"), list
        ):
            raise ValueError(f"adapter mapped fields and failure modes must be lists: {item_id}")
        unknown_failure_modes = set(prediction["failure_modes"]) - FAILURE_MODES
        if unknown_failure_modes:
            raise ValueError(f"unknown adapter failure modes for {item_id}: {sorted(unknown_failure_modes)}")
        confidence = prediction.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise ValueError(f"invalid adapter confidence: {item_id}")
        if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"adapter confidence outside [0,1]: {item_id}")
        predictions[item_id] = prediction
    return predictions


def _score_row(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    prediction: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if prediction.get("record_hash") != row["record_hash"]:
        raise ValueError(f"prediction record hash mismatch: {row['item_id']}")
    parsed = bool(prediction.get("parsed"))
    mapped = set(str(item) for item in prediction.get("mapped_fields", ())) if parsed else set()
    expected_mapped = set(str(item) for item in expected["mapped_fields"])
    forbidden = {str(item).lower() for item in config["forbidden_fields"]}
    leaked_fields = {item for item in mapped if item.lower() in forbidden}
    leaked_keys = _prediction_keys(prediction) & forbidden
    true_positive = len(mapped & expected_mapped)
    precision = true_positive / len(mapped) if mapped else (1.0 if not expected_mapped else 0.0)
    recall = true_positive / len(expected_mapped) if expected_mapped else (1.0 if not mapped else 0.0)
    mapping_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    parse_status_correct = parsed == bool(expected["should_parse"])
    exact = bool(
        parse_status_correct
        and prediction.get("game") == expected["game"]
        and prediction.get("unit_type") == expected["unit_type"]
        and mapped == expected_mapped
        and prediction.get("disposition") == expected["disposition"]
        and prediction.get("evidence_authority") == expected["evidence_authority"]
        and set(prediction.get("failure_modes", ())) == set(expected["failure_modes"])
        and bool(prediction.get("abstained")) == bool(expected["should_abstain"])
        and not leaked_fields
        and not leaked_keys
    )
    false_release = expected["disposition"] != "VERIFIED_TRAIN" and prediction.get("disposition") == "VERIFIED_TRAIN"
    confidence = float(prediction["confidence"])
    source_agent = row.get("source_agent", {})
    source_model = str(source_agent.get("model_family", "unresolved"))
    reasoning_effort = str(source_agent.get("reasoning_effort", "unknown"))
    return {
        "item_id": row["item_id"],
        "split": row["split"],
        "game": expected["game"],
        "source_model": source_model,
        "reasoning_effort": reasoning_effort,
        "source_agent_cohort": f"{source_model}|{reasoning_effort}",
        "projection_origin": str(row.get("projection_origin", "synthetic_fixture")),
        "expected_abstention": bool(expected["should_abstain"]),
        "parsed": parsed,
        "parse_status_correct": parse_status_correct,
        "exact": exact,
        "mapping_f1": mapping_f1,
        "abstention_correct": bool(prediction.get("abstained")) == bool(expected["should_abstain"]),
        "abstained": bool(prediction.get("abstained")),
        "privacy_leak": bool(leaked_fields or leaked_keys),
        "release_hazard": expected["disposition"] != "VERIFIED_TRAIN",
        "critical_false_release": false_release,
        "forbidden_authority_claim": prediction.get("authority") != "MODEL_RECOMMENDATION_ONLY",
        "confidence": confidence,
        "brier": (confidence - float(exact)) ** 2,
        "error_codes": _error_codes(expected, prediction, leaked_fields, leaked_keys, exact),
    }


def _error_codes(
    expected: Mapping[str, Any],
    prediction: Mapping[str, Any],
    leaked_fields: set[str],
    leaked_keys: set[str],
    exact: bool,
) -> list[str]:
    errors: list[str] = []
    if bool(prediction.get("parsed")) != bool(expected["should_parse"]):
        errors.append("parse_status_mismatch")
    if leaked_fields or leaked_keys:
        errors.append("privacy_leak")
    if prediction.get("disposition") != expected["disposition"]:
        errors.append("disposition_mismatch")
    if bool(prediction.get("abstained")) != bool(expected["should_abstain"]):
        errors.append("abstention_mismatch")
    if not exact and not errors:
        errors.append("normalization_mismatch")
    return errors


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if not count:
        raise ValueError("cannot aggregate an empty adapter evaluation slice")
    critical = [row for row in rows if row["critical_false_release"]]
    release_hazards = [row for row in rows if row["release_hazard"]]
    errors = Counter(error for row in rows for error in row["error_codes"])
    return {
        "examples": count,
        "schema_parse_rate": sum(bool(row["parsed"]) for row in rows) / count,
        "parse_status_accuracy": sum(bool(row["parse_status_correct"]) for row in rows) / count,
        "exact_record_accuracy": sum(bool(row["exact"]) for row in rows) / count,
        "mapping_f1": float(np.mean([float(row["mapping_f1"]) for row in rows])),
        "abstention_accuracy": sum(bool(row["abstention_correct"]) for row in rows) / count,
        "coverage": sum(not bool(row["abstained"]) for row in rows) / count,
        "privacy_leak_rate": sum(bool(row["privacy_leak"]) for row in rows) / count,
        "critical_false_release_rate": len(critical) / len(release_hazards) if release_hazards else 0.0,
        "forbidden_authority_claim_rate": (
            sum(bool(row["forbidden_authority_claim"]) for row in rows) / count
        ),
        "brier_score": float(np.mean([float(row["brier"]) for row in rows])),
        "error_counts": dict(sorted(errors.items())),
    }


def _aggregate_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    values = sorted({str(row[key]) for row in rows})
    return {value: _aggregate([row for row in rows if str(row[key]) == value]) for value in values}


def _write_scored_receipts(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    receipts = [
        {
            "item_id": row["item_id"],
            "split": row["split"],
            "game": row["game"],
            "source_model": row["source_model"],
            "reasoning_effort": row["reasoning_effort"],
            "parsed": row["parsed"],
            "parse_status_correct": row["parse_status_correct"],
            "exact": row["exact"],
            "privacy_leak": row["privacy_leak"],
            "critical_false_release": row["critical_false_release"],
            "error_codes": row["error_codes"],
        }
        for row in rows
    ]
    _write_jsonl(path, receipts)


def _render_evaluation(report: Mapping[str, Any]) -> str:
    overall = report["overall"]
    lines = [
        "# Game-adapter verifier evaluation",
        "",
        f"Setup: `{report['setup_id']}`; evidence status: `{report['evidence_status']}`.",
        "",
        "| Slice | Exact | Parse status | Mapping F1 | Abstention | Privacy leak | False release | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| overall | {overall['exact_record_accuracy']:.3f} | {overall['parse_status_accuracy']:.3f} | "
            f"{overall['mapping_f1']:.3f} | "
            f"{overall['abstention_accuracy']:.3f} | {overall['privacy_leak_rate']:.3f} | "
            f"{overall['critical_false_release_rate']:.3f} | {overall['brier_score']:.3f} |"
        ),
    ]
    for game, metrics in report["game_results"].items():
        lines.append(
            f"| {game} | {metrics['exact_record_accuracy']:.3f} | {metrics['parse_status_accuracy']:.3f} | "
            f"{metrics['mapping_f1']:.3f} | "
            f"{metrics['abstention_accuracy']:.3f} | {metrics['privacy_leak_rate']:.3f} | "
            f"{metrics['critical_false_release_rate']:.3f} | {metrics['brier_score']:.3f} |"
        )
    lines.extend(("", "## Source-agent cohorts", ""))
    lines.extend(
        (
            "| Source model / effort | Examples | Exact | Abstention | Privacy leak | False release |",
            "|---|---:|---:|---:|---:|---:|",
        )
    )
    for cohort, metrics in report["source_agent_results"].items():
        lines.append(
            f"| {cohort} | {metrics['examples']} | {metrics['exact_record_accuracy']:.3f} | "
            f"{metrics['abstention_accuracy']:.3f} | {metrics['privacy_leak_rate']:.3f} | "
            f"{metrics['critical_false_release_rate']:.3f} |"
        )
    diagnostics = report["benchmark_diagnostics"]
    lines.extend(
        (
            "",
            "## Benchmark diagnostics",
            "",
            f"Fitness scope: `{diagnostics['fitness_scope']}`; hazard saturation: "
            f"`{diagnostics['hazard_saturation_rate']:.3f}`; non-abstention opportunities: "
            f"`{diagnostics['non_abstention_opportunities']}`.",
            "",
            str(diagnostics["warning"]),
        )
    )
    lines.extend(
        (
            "",
            "## Promotion assessment",
            "",
            f"Eligible: `{report['promotion_assessment']['eligible']}`.",
            "",
            str(report["promotion_assessment"]["authority"]),
            "",
            "## Claim boundary",
            "",
            str(report["claim_boundary"]),
            "",
        )
    )
    return "\n".join(lines)


def _mock_prediction(row: Mapping[str, Any], setup_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    shape = row["source_shape"]
    keys = set(str(item) for item in shape["keys"])
    game = str(row["game_hint"])
    contract = GAME_CONTRACTS[game]
    forbidden = {re.sub(r"[^a-z0-9]+", "_", str(item).lower()).strip("_") for item in config["forbidden_fields"]}
    has_forbidden = any(re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_") in forbidden for key in keys)
    flags = shape["flags"]
    base = {
        "schema_version": "game-adapter-prediction.v1",
        "item_id": row["item_id"],
        "record_hash": row["record_hash"],
        "setup_id": setup_id,
        "parsed": True,
        "game": game,
        "unit_type": contract["unit_type"],
        "authority": "MODEL_RECOMMENDATION_ONLY",
    }
    if setup_id == "strict-policy-mock":
        if not flags["schema_supported"]:
            return {
                **base,
                "parsed": False,
                "mapped_fields": [],
                "disposition": "QUARANTINE",
                "evidence_authority": "UNKNOWN",
                "failure_modes": ["unknown"],
                "abstained": True,
                "confidence": 0.98,
            }
        if flags["sealed_evaluation"]:
            return {
                **base,
                "mapped_fields": [],
                "disposition": "SEALED_EVALUATION",
                "evidence_authority": "ENVIRONMENT_VERIFIED",
                "failure_modes": [],
                "abstained": True,
                "confidence": 0.99,
            }
        if flags.get("explicit_hidden_reasoning", False):
            return {
                **base,
                "mapped_fields": [],
                "disposition": "REJECT",
                "evidence_authority": "UNKNOWN",
                "failure_modes": (
                    ["hidden_reasoning", "private_data"]
                    if flags.get("private_visibility", False)
                    else ["hidden_reasoning"]
                ),
                "abstained": True,
                "confidence": 0.99,
            }
        if shape["visibility"] != "public" or flags.get("private_visibility", False):
            return {
                **base,
                "mapped_fields": [],
                "disposition": "QUARANTINE",
                "evidence_authority": "UNKNOWN",
                "failure_modes": ["private_data"],
                "abstained": True,
                "confidence": 0.99,
            }
        if has_forbidden:
            return {
                **base,
                "mapped_fields": [],
                "disposition": "REJECT",
                "evidence_authority": "UNKNOWN",
                "failure_modes": ["hidden_reasoning"],
                "abstained": True,
                "confidence": 0.99,
            }
        if flags.get("unresolved_fields", False):
            return {
                **base,
                "parsed": False,
                "mapped_fields": [],
                "disposition": "QUARANTINE",
                "evidence_authority": "UNKNOWN",
                "failure_modes": ["unknown"],
                "abstained": True,
                "confidence": 0.98,
            }
        if flags.get("safe_field_derivation", False):
            return {
                **base,
                "mapped_fields": sorted(keys & set(contract["safe_fields"])),
                "disposition": "DISCOVERY_ONLY",
                "evidence_authority": "MODEL_INFERRED",
                "failure_modes": [],
                "abstained": False,
                "confidence": 0.97,
            }
        if flags.get("aggregate_receipt_only", False):
            return {
                **base,
                "mapped_fields": sorted(keys & set(contract["safe_fields"])),
                "disposition": "DISCOVERY_ONLY",
                "evidence_authority": "UNKNOWN",
                "failure_modes": ["unsupported_claim"],
                "abstained": False,
                "confidence": 0.95,
            }
        return {
            **base,
            "mapped_fields": sorted(keys & set(contract["safe_fields"])),
            "disposition": "VERIFIED_TRAIN",
            "evidence_authority": "ENVIRONMENT_VERIFIED",
            "failure_modes": [],
            "abstained": False,
            "confidence": 0.99,
        }
    if setup_id == "permissive-map-all-mock":
        return {
            **base,
            "mapped_fields": sorted(keys),
            "disposition": "VERIFIED_TRAIN",
            "evidence_authority": "MODEL_INFERRED",
            "failure_modes": [],
            "abstained": False,
            "confidence": 0.95,
        }
    if setup_id == "conservative-abstain-mock":
        return {
            **base,
            "mapped_fields": [],
            "disposition": "QUARANTINE",
            "evidence_authority": "UNKNOWN",
            "failure_modes": ["unknown"],
            "abstained": True,
            "confidence": 0.75,
        }
    raise ValueError(f"unknown mock adapter setup: {setup_id}")


def _write_mock_setup(
    setup_id: str,
    rows: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    config: Mapping[str, Any],
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = out_dir / "candidate_manifest.json"
    predictions_path = out_dir / "predictions.jsonl"
    dump_structured(
        candidate_path,
        {
            "schema_version": "game-adapter-candidate.v1",
            "setup_id": setup_id,
            "adapter_family": "deterministic_policy_mock",
            "adapter_version": "0.1.0",
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "mock": True,
            "authoritative_label_access": False,
            "input_contract": "public_shape_only",
        },
    )
    _write_jsonl(predictions_path, [_mock_prediction(row, setup_id, config) for row in rows])
    return candidate_path, predictions_path


def run_verifier_demo(config_path: Path, out_dir: Path, *, seed: int = 7) -> dict[str, Any]:
    """Exercise three adapter behaviors without corpus values, weights, network, or model calls."""
    config = _load_config(config_path)
    out_dir.mkdir(parents=True, exist_ok=False)
    build_verifier_dataset(config_path, out_dir, seed=seed)
    manifest_path = out_dir / "public_dataset" / "dataset_manifest.json"
    steward_labels = out_dir / "private_steward" / "ground_truth.jsonl"
    _, rows = _read_dataset(manifest_path)
    reports: list[Path] = []
    for setup_id in ("strict-policy-mock", "permissive-map-all-mock", "conservative-abstain-mock"):
        candidate_path, predictions_path = _write_mock_setup(
            setup_id,
            rows,
            manifest_path,
            config,
            out_dir / "setups" / setup_id,
        )
        report_dir = out_dir / "evaluations" / setup_id
        evaluate_adapter_setup(
            config_path,
            manifest_path,
            steward_labels,
            candidate_path,
            predictions_path,
            report_dir,
        )
        reports.append(report_dir / "evaluation_report.json")
    comparison = compare_adapter_reports(reports, out_dir)
    dump_structured(
        out_dir / "summary.json",
        {
            "schema_version": "1.0",
            "status": "mock_adapter_verification_completed",
            "seed": seed,
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "setups": comparison["setups"],
            "weights_loaded": False,
            "network_used": False,
            "raw_corpus_values_read": False,
            "promotion_eligible_setups": [],
        },
    )
    return comparison


def run_mock_setups_on_dataset(
    config_path: Path,
    dataset_manifest_path: Path,
    steward_labels_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Run deterministic policy controls against one already-frozen projected dataset."""
    config = _load_config(config_path)
    manifest, rows = _read_dataset(dataset_manifest_path)
    if manifest["config_sha256"] != sha256_file(config_path):
        raise ValueError("dataset and verifier config hashes do not match")
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: list[Path] = []
    for setup_id in ("strict-policy-mock", "permissive-map-all-mock", "conservative-abstain-mock"):
        candidate_path, predictions_path = _write_mock_setup(
            setup_id,
            rows,
            dataset_manifest_path,
            config,
            out_dir / "setups" / setup_id,
        )
        report_dir = out_dir / "evaluations" / setup_id
        evaluate_adapter_setup(
            config_path,
            dataset_manifest_path,
            steward_labels_path,
            candidate_path,
            predictions_path,
            report_dir,
        )
        reports.append(report_dir / "evaluation_report.json")
    return compare_adapter_reports(reports, out_dir)


def compare_adapter_reports(report_paths: Sequence[Path], out_dir: Path) -> dict[str, Any]:
    if len(report_paths) < 2:
        raise ValueError("adapter comparison requires at least two reports")
    reports = [load_structured(path) for path in report_paths]
    if any(report.get("schema_version") != "game-adapter-evaluation.v1" for report in reports):
        raise ValueError("unsupported report in adapter comparison")
    dataset_hashes = {str(report["reproducibility"]["dataset_manifest_sha256"]) for report in reports}
    config_hashes = {str(report["reproducibility"]["config_sha256"]) for report in reports}
    if len(dataset_hashes) != 1 or len(config_hashes) != 1:
        raise ValueError("adapter reports are not bound to the same dataset and verifier config")
    setup_ids = [str(report["setup_id"]) for report in reports]
    if len(set(setup_ids)) != len(setup_ids):
        raise ValueError("adapter comparison contains duplicate setup IDs")
    rows = []
    for report in reports:
        metrics = report["overall"]
        rows.append(
            {
                "setup_id": report["setup_id"],
                "exact_record_accuracy": metrics["exact_record_accuracy"],
                "parse_status_accuracy": metrics["parse_status_accuracy"],
                "mapping_f1": metrics["mapping_f1"],
                "abstention_accuracy": metrics["abstention_accuracy"],
                "privacy_leak_rate": metrics["privacy_leak_rate"],
                "critical_false_release_rate": metrics["critical_false_release_rate"],
                "coverage": metrics["coverage"],
                "promotion_eligible": report["promotion_assessment"]["eligible"],
            }
        )
    comparison = {
        "schema_version": "game-adapter-comparison.v1",
        "setups": setup_ids,
        "rows": rows,
        "winner": None,
        "selection_rule": "Use safety gates first, then compare per-game Pareto tradeoffs; no universal scalar winner.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_structured(out_dir / "comparison.json", comparison)
    (out_dir / "comparison.md").write_text(_render_comparison(comparison), encoding="utf-8")
    return comparison


def _render_comparison(comparison: Mapping[str, Any]) -> str:
    lines = [
        "# Four-game adapter comparison",
        "",
        "| Setup | Exact | Parse status | Mapping F1 | Abstention | Privacy leak | "
        "False release | Coverage | Eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        lines.append(
            f"| {row['setup_id']} | {row['exact_record_accuracy']:.3f} | "
            f"{row['parse_status_accuracy']:.3f} | {row['mapping_f1']:.3f} | "
            f"{row['abstention_accuracy']:.3f} | {row['privacy_leak_rate']:.3f} | "
            f"{row['critical_false_release_rate']:.3f} | {row['coverage']:.3f} | "
            f"{row['promotion_eligible']} |"
        )
    lines.extend(("", f"Winner: `{comparison['winner']}`.", "", str(comparison["selection_rule"]), ""))
    return "\n".join(lines)
