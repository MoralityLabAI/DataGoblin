#!/usr/bin/env python3
"""Train a small public-envelope proposal head behind the LDT policy membrane.

This trainer is deliberately standard-library-only so the capped WSL run does not
download packages. The learned head recommends; the deterministic policy and
independent steward remain the release authority.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DISPOSITIONS = (
    "DISCOVERY_ONLY",
    "EVALUATION_ONLY",
    "QUARANTINE",
    "REJECT",
    "SEALED_EVALUATION",
    "VERIFIED_TRAIN",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_event(path: Path, event: str, **fields: Any) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _read_dataset(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for split_name, split in manifest["splits"].items():
        path = manifest_path.parent / str(split["path"])
        if _sha256_file(path) != split["sha256"]:
            raise ValueError(f"split hash mismatch: {split_name}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                if row["split"] != split_name:
                    raise ValueError(f"row split mismatch: {row['item_id']}")
                rows.append(row)
    return manifest, rows


def _cross_split_groups(rows: list[dict[str, Any]]) -> set[str]:
    roles: dict[str, set[str]] = {}
    for row in rows:
        record = row["record"]
        role = "training" if row["split"] in {"train", "calibration"} else "evaluation"
        roles.setdefault(str(record["dedup_group"]), set()).add(role)
    return {group for group, values in roles.items() if values == {"training", "evaluation"}}


def _features(record: dict[str, Any], cross_split_groups: set[str]) -> tuple[str, ...]:
    intended_use = "EVALUATION" if record["evaluation_material"] or record["sealed_evaluation"] else "TRAINING"
    uses = tuple(sorted(str(item) for item in record["allowed_uses"]))
    features = {
        "bias",
        f"source_kind={record['source_kind']}",
        f"lawful_basis={record['lawful_basis']}",
        f"license_status={record['license_status']}",
        f"consent_status={record['consent_status']}",
        f"personal_data_status={record['personal_data_status']}",
        f"evidence_type={record['evidence_type']}",
        f"verification_custody={record['verification_custody']}",
        f"intended_split={record['intended_split']}",
        f"residency={record['data_residency_region']}",
        f"sensitivity={record['sensitivity_class']}",
        f"retention_positive={record['retention_days'] > 0}",
        f"redacted={record['redacted']}",
        f"contains_secret={record['contains_secret']}",
        f"contains_hidden_reasoning={record['contains_hidden_reasoning']}",
        f"contains_model_weight={record['contains_model_weight']}",
        f"evaluation_material={record['evaluation_material']}",
        f"sealed_evaluation={record['sealed_evaluation']}",
        f"model_generated={record['model_generated']}",
        f"generator_present={bool(record.get('generator_id'))}",
        f"verified_by_present={bool(record.get('verified_by'))}",
        f"verification_receipt_present={bool(record.get('verification_receipt_sha256'))}",
        f"verification_subject_matches={record.get('verification_subject_sha256') == record['content_sha256']}",
        f"provenance_present={bool(record['provenance_refs'])}",
        f"deletion_requested={record['deletion_requested']}",
        f"purpose_present={bool(record['purpose_id'])}",
        f"intended_use_permitted={intended_use in uses}",
        f"cross_split_duplicate={record['dedup_group'] in cross_split_groups}",
    }
    features.update(f"allowed_use={use}" for use in uses)
    return tuple(sorted(features))


def _score(weights: dict[str, dict[str, float]], label: str, features: tuple[str, ...]) -> float:
    row = weights[label]
    return sum(row.get(feature, 0.0) for feature in features)


def _predict_label(weights: dict[str, dict[str, float]], features: tuple[str, ...]) -> str:
    return max(DISPOSITIONS, key=lambda label: (_score(weights, label, features), label))


def _softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    peak = max(scores.values())
    values = {label: math.exp((score - peak) / temperature) for label, score in scores.items()}
    total = sum(values.values())
    return {label: value / total for label, value in values.items()}


def _calibration_loss(
    weights: dict[str, dict[str, float]],
    examples: list[tuple[tuple[str, ...], str, tuple[str, ...]]],
    temperature: float,
) -> float:
    if not examples:
        return 0.0
    total = 0.0
    for features, actual, _ in examples:
        probabilities = _softmax(
            {label: _score(weights, label, features) for label in DISPOSITIONS},
            temperature,
        )
        total += sum((probabilities[label] - float(label == actual)) ** 2 for label in DISPOSITIONS)
    return total / len(examples)


def _accuracy(
    weights: dict[str, dict[str, float]],
    examples: list[tuple[tuple[str, ...], str, tuple[str, ...]]],
) -> float:
    if not examples:
        return 0.0
    return sum(_predict_label(weights, features) == actual for features, actual, _ in examples) / len(examples)


def _checkpoint(
    out_dir: Path,
    step: int,
    weights: dict[str, dict[str, float]],
    flag_weights: dict[str, dict[str, float]],
    events_path: Path,
) -> Path:
    path = out_dir / "checkpoints" / f"step-{step:06d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "step": step,
        "architecture": "learned proposal head plus explicit LDT policy membrane",
        "disposition_weights": weights,
        "violation_weights": flag_weights,
    }
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    _json_event(events_path, "checkpoint", step=step, checkpoint_path=str(path), sha256=_sha256_file(path))
    return path


def train(args: argparse.Namespace) -> int:
    if args.batch_size < 1 or args.batch_size > 32:
        raise ValueError("batch size must be between 1 and 32")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint cadence must be positive")
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = Path(os.environ.get("DATAGOBLIN_GYM_EVENTS", out_dir / "events.jsonl"))
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.touch(exist_ok=True)
    summary_path = out_dir / "trainer_summary.json"
    weights: dict[str, dict[str, float]] = {label: {} for label in DISPOSITIONS}
    flag_weights: dict[str, dict[str, float]] = {}
    checkpoints: list[str] = []
    step = 0
    started = datetime.now(timezone.utc)
    try:
        manifest, rows = _read_dataset(args.dataset_manifest.resolve())
        cross_groups = _cross_split_groups(rows)
        train_rows_all = [row for row in rows if row["split"] == "train"]
        calibration_rows_all = [row for row in rows if row["split"] == "calibration"]
        train_rows = [row for row in train_rows_all if row["record"]["dedup_group"] not in cross_groups]
        calibration_rows = [
            row for row in calibration_rows_all if row["record"]["dedup_group"] not in cross_groups
        ]
        if any("authoritative_label" not in row for row in train_rows + calibration_rows):
            raise ValueError("training and calibration rows require steward labels")
        if any("authoritative_label" in row for row in rows if row["split"] == "sealed_poison"):
            raise ValueError("sealed labels leaked into the trainer dataset")
        examples = [
            (
                _features(row["record"], cross_groups),
                str(row["authoritative_label"]["disposition"]),
                tuple(str(item) for item in row["authoritative_label"]["violations"]),
            )
            for row in train_rows
        ]
        calibration = [
            (
                _features(row["record"], cross_groups),
                str(row["authoritative_label"]["disposition"]),
                tuple(str(item) for item in row["authoritative_label"]["violations"]),
            )
            for row in calibration_rows
        ]
        violation_labels = sorted({flag for _, _, flags in examples for flag in flags})
        flag_weights = {flag: {} for flag in violation_labels}
        rng = random.Random(args.seed)
        _json_event(
            events_path,
            "trainer_start",
            run_id=args.run_id,
            examples=len(examples),
            calibration_examples=len(calibration),
            epochs=args.epochs,
            batch_size=args.batch_size,
            sealed_labels_available=False,
            external_checkpoint_loaded=False,
        )
        order = list(range(len(examples)))
        for epoch in range(args.epochs):
            rng.shuffle(order)
            for offset in range(0, len(order), args.batch_size):
                batch = order[offset : offset + args.batch_size]
                for index in batch:
                    features, actual, actual_flags = examples[index]
                    predicted = _predict_label(weights, features)
                    if predicted != actual:
                        for feature in features:
                            weights[actual][feature] = weights[actual].get(feature, 0.0) + args.learning_rate
                            weights[predicted][feature] = weights[predicted].get(feature, 0.0) - args.learning_rate
                    actual_flag_set = set(actual_flags)
                    for flag in violation_labels:
                        predicted_present = _score(flag_weights, flag, features) > 0.0
                        actual_present = flag in actual_flag_set
                        if predicted_present != actual_present:
                            direction = args.learning_rate if actual_present else -args.learning_rate
                            for feature in features:
                                flag_weights[flag][feature] = flag_weights[flag].get(feature, 0.0) + direction
                    step += 1
                    if step % args.checkpoint_every == 0:
                        checkpoint = _checkpoint(out_dir, step, weights, flag_weights, events_path)
                        checkpoints.append(str(checkpoint))
                _json_event(events_path, "batch", epoch=epoch + 1, step=step, examples=len(batch))
        if not checkpoints or step % args.checkpoint_every:
            checkpoint = _checkpoint(out_dir, step, weights, flag_weights, events_path)
            checkpoints.append(str(checkpoint))
        temperatures = (0.5, 1.0, 2.0, 4.0)
        temperature = min(temperatures, key=lambda value: (_calibration_loss(weights, calibration, value), value))
        model = {
            "schema_version": "1.0",
            "model_id": f"{args.run_id}-step-{step}",
            "architecture": "learned public-envelope proposal head plus explicit LDT policy membrane",
            "authority": "MODEL_RECOMMENDATION_ONLY",
            "external_checkpoint_loaded": False,
            "seed": args.seed,
            "steps": step,
            "temperature": temperature,
            "dispositions": list(DISPOSITIONS),
            "disposition_weights": weights,
            "violation_weights": flag_weights,
            "forbidden_features": ["category", "item_id", "content_sha256", "provenance_values", "sealed_labels"],
        }
        model_path = out_dir / "model.json"
        model_path.write_text(json.dumps(model, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        predictions_path = out_dir / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                features = _features(row["record"], cross_groups)
                scores = {label: _score(weights, label, features) for label in DISPOSITIONS}
                probabilities = _softmax(scores, temperature)
                predicted = max(DISPOSITIONS, key=lambda label: (probabilities[label], label))
                flags = sorted(flag for flag in violation_labels if _score(flag_weights, flag, features) > 0.0)
                prediction = {
                    "item_id": row["item_id"],
                    "sorter_id": model["model_id"],
                    "recommended_disposition": predicted,
                    "flags": flags,
                    "confidence": probabilities[predicted],
                    "abstained": probabilities[predicted] < args.abstain_threshold,
                    "authority": "MODEL_RECOMMENDATION_ONLY",
                    "record_hash": row["record_hash"],
                }
                handle.write(json.dumps(prediction, sort_keys=True, separators=(",", ":")) + "\n")
        summary = {
            "schema_version": "1.0",
            "run_id": args.run_id,
            "status": "completed",
            "abort_reason": None,
            "external_checkpoint_loaded": False,
            "weights_trained": True,
            "steps_completed": step,
            "epochs_completed": args.epochs,
            "batch_size": args.batch_size,
            "cross_split_rows_excluded_from_fit": len(train_rows_all) - len(train_rows),
            "cross_split_rows_excluded_from_calibration": len(calibration_rows_all) - len(calibration_rows),
            "checkpoint_interval_steps": args.checkpoint_every,
            "checkpoints": checkpoints,
            "train_accuracy": _accuracy(weights, examples),
            "calibration_accuracy": _accuracy(weights, calibration),
            "calibration_temperature": temperature,
            "dataset_id": manifest["dataset_id"],
            "dataset_manifest_sha256": _sha256_file(args.dataset_manifest.resolve()),
            "model_sha256": _sha256_file(model_path),
            "predictions_sha256": _sha256_file(predictions_path),
            "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
            "claim_boundary": "Trained proposal head only; the LDT policy membrane and steward remain authoritative.",
        }
        summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        _json_event(events_path, "trainer_complete", step=step, model_sha256=summary["model_sha256"])
        return 0
    except Exception as exc:
        summary = {
            "schema_version": "1.0",
            "run_id": args.run_id,
            "status": "aborted",
            "abort_reason": f"{type(exc).__name__}: {exc}",
            "external_checkpoint_loaded": False,
            "weights_trained": step > 0,
            "steps_completed": step,
            "checkpoints": checkpoints,
            "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        }
        summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        _json_event(events_path, "trainer_abort", step=step, reason=summary["abort_reason"])
        return 1
    finally:
        weights.clear()
        flag_weights.clear()
        gc.collect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default="ldt-etiquette-sorter-v1-smoke")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=14)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--abstain-threshold", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=7)
    return parser


if __name__ == "__main__":
    sys.exit(train(build_parser().parse_args()))
