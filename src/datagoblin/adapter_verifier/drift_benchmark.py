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

from .environment import GAME_CONTRACTS, GAMES

DRIFT_SPLITS = ("train", "calibration", "ordinary_test", "ood_test", "sealed_poison")
EVALUATION_SPLITS = ("ordinary_test", "ood_test", "sealed_poison")
_SYNONYMS = {
    "chosen_action": ("selected_action", "selected_move"),
    "environment_outcome": ("observed_outcome", "observed_result"),
    "legal_actions": ("permitted_actions", "permitted_moves"),
    "public_state": ("visible_state", "visible_snapshot"),
    "state_hash": ("state_digest", "state_fingerprint"),
    "orders": ("submitted_orders", "submitted_commands"),
    "phase": ("turn_phase", "turn_stage"),
    "public_action": ("visible_action", "visible_move"),
    "public_message": ("visible_message", "visible_note"),
    "usage_bucket": ("usage_group", "usage_cluster"),
    "public_observation": ("visible_observation", "visible_snapshot"),
    "target": ("selected_target", "selected_entity"),
    "adjudicated_outcome": ("resolved_outcome", "resolved_result"),
    "public_press": ("visible_press", "visible_broadcast"),
    "source_checks": ("provenance_checks", "provenance_validations"),
    "state_delta": ("state_change", "state_difference"),
}


def _load_contract(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_structured(config_path)
    if config.get("schema_version") != "1.0":
        raise ValueError("unsupported schema-drift benchmark version")
    if tuple(config.get("games", ())) != GAMES:
        raise ValueError("schema-drift benchmark must declare the four games in canonical order")
    split_contract = config.get("split_contract", {})
    if tuple(split_contract.get("fitting_splits", ())) != ("train",):
        raise ValueError("schema-drift fitting split must be train only")
    if tuple(split_contract.get("calibration_splits", ())) != ("calibration",):
        raise ValueError("schema-drift calibration split must be calibration only")
    if tuple(split_contract.get("evaluation_splits", ())) != EVALUATION_SPLITS:
        raise ValueError("schema-drift evaluation splits are not frozen")
    policy_path = Path(str(config["policy_config"]))
    if not policy_path.is_absolute():
        policy_path = config_path.parents[2] / policy_path
    policy = load_structured(policy_path)
    if policy.get("schema_version") != "1.0" or not policy.get("forbidden_fields"):
        raise ValueError("schema-drift benchmark requires a valid curation policy")
    return config, policy


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _snake(value: str) -> str:
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return re.sub(r"[^a-z0-9]+", "_", separated.lower()).strip("_")


def _camel(value: str) -> str:
    parts = _snake(value).split("_")
    return parts[0] + "".join(part.title() for part in parts[1:])


def _choice(seed: int, field: str, split: str, values: Sequence[str]) -> str:
    index = int(sha256_text(f"{seed}:{field}:{split}")[:8], 16) % len(values)
    return str(values[index])


def _aliases(field: str, split: str, seed: int) -> tuple[str, ...]:
    synonym, ood_synonym = _SYNONYMS.get(field, (f"observed_{field}", f"alternate_{field}"))
    prefix = _choice(seed, field, split, ("src", "incoming", "input"))
    suffix = _choice(seed + 1, field, split, ("field", "entry", "slot"))
    if split == "train":
        values = (f"{prefix}_{field}", f"{field}_{suffix}", synonym)
    elif split == "calibration":
        separator = _choice(seed, field, split, ("-", "."))
        values = (synonym.replace("_", separator), f"{_camel(synonym)}{suffix.title()}")
    elif split == "ordinary_test":
        separator = _choice(seed, field, split, (".", "-"))
        dotted = field.replace("_", separator) if "_" in field else f"{field}{separator}{suffix}"
        camel = f"{_camel(field)}{suffix.title()}"
        values = (dotted, camel)
    elif split == "ood_test":
        values = (f"{ood_synonym}_{suffix}",)
    else:
        raise ValueError(f"aliases are not defined for split: {split}")
    if field in values or len(set(values)) != len(values):
        raise ValueError(f"invalid alias family for {field}/{split}")
    return values


def _expected(game: str, category: str) -> dict[str, Any]:
    contract = GAME_CONTRACTS[game]
    expected = {
        "game": game,
        "unit_type": contract["unit_type"],
        "mapped_fields": list(contract["safe_fields"]),
        "disposition": "VERIFIED_TRAIN",
        "evidence_authority": "ENVIRONMENT_VERIFIED",
        "should_parse": True,
        "should_abstain": False,
        "failure_modes": [],
    }
    if category == "private_visibility":
        expected.update(
            mapped_fields=[],
            disposition="QUARANTINE",
            evidence_authority="UNKNOWN",
            should_abstain=True,
            failure_modes=["private_data"],
        )
    elif category == "hidden_reasoning":
        expected.update(
            mapped_fields=[],
            disposition="REJECT",
            evidence_authority="UNKNOWN",
            should_abstain=True,
            failure_modes=["hidden_reasoning"],
        )
    elif category == "ambiguous_schema":
        expected.update(
            mapped_fields=[],
            disposition="QUARANTINE",
            evidence_authority="UNKNOWN",
            should_parse=False,
            should_abstain=True,
            failure_modes=["unknown"],
        )
    elif category == "sealed_evaluation":
        expected.update(mapped_fields=[], disposition="SEALED_EVALUATION", should_abstain=True)
    elif category != "clean_drift":
        raise ValueError(f"unknown schema-drift category: {category}")
    return expected


def _make_row(
    *,
    game: str,
    split: str,
    category: str,
    keys: Sequence[str],
    field_targets: Mapping[str, str | None],
    family: str,
    seed: int,
    ordinal: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = GAME_CONTRACTS[game]
    flags = {
        "environment_verified": True,
        "evaluation_material": split == "sealed_poison",
        "sealed_evaluation": split == "sealed_poison",
        "schema_supported": category != "ambiguous_schema",
        "explicit_hidden_reasoning": category == "hidden_reasoning",
        "private_visibility": category == "private_visibility",
    }
    visibility = "actor_private" if category == "private_visibility" else "public"
    opaque = sha256_text(f"{seed}:{split}:{game}:{category}:{family}:{ordinal}:{'|'.join(keys)}")
    public = {
        "schema_version": "game-adapter-drift-input.v1",
        "item_id": f"drift-{split}-{game}-{ordinal:04d}-{opaque[:10]}",
        "split": split,
        "game_hint": game,
        "source_schema": f"{game}-drift-fixture-v1",
        "source_locator_sha256": sha256_text(f"source:{opaque}"),
        "lineage_group_sha256": sha256_text(f"lineage:{opaque}"),
        "transformation_family": family,
        "source_shape": {
            "keys": sorted(keys),
            "record_kind": contract["unit_type"],
            "visibility": visibility,
            "flags": flags,
        },
    }
    record_hash = sha256_json(public)
    row = {**public, "record_hash": record_hash}
    label = {
        "schema_version": "game-adapter-drift-label.v1",
        "item_id": row["item_id"],
        "record_hash": record_hash,
        "split": split,
        "expected": _expected(game, category),
        "field_targets": dict(sorted(field_targets.items())),
        "authority": "ENVIRONMENT_OWNED",
    }
    return row, label


def _clean_rows(game: str, split: str, seed: int, start: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    safe_fields = tuple(str(field) for field in GAME_CONTRACTS[game]["safe_fields"])
    family = {
        "train": "fit_lexical_affix_v1",
        "calibration": "calibration_separator_v1",
        "ordinary_test": "holdout_orthographic_v1",
        "ood_test": "holdout_semantic_v1",
    }[split]
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    ordinal = start
    for target in safe_fields:
        for alias in _aliases(target, split, seed):
            keys = [alias if field == target else field for field in safe_fields]
            field_targets = {key: (target if key == alias else key) for key in keys}
            rows.append(
                _make_row(
                    game=game,
                    split=split,
                    category="clean_drift",
                    keys=keys,
                    field_targets=field_targets,
                    family=family,
                    seed=seed,
                    ordinal=ordinal,
                )
            )
            ordinal += 1
    return rows


def _hazard_rows(game: str, split: str, seed: int, start: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    categories = {
        "train": ("private_visibility", "hidden_reasoning", "ambiguous_schema"),
        "calibration": ("private_visibility",),
        "ordinary_test": ("private_visibility",),
        "ood_test": ("hidden_reasoning", "ambiguous_schema"),
        "sealed_poison": ("sealed_evaluation", "sealed_evaluation"),
    }[split]
    contract = GAME_CONTRACTS[game]
    safe_fields = list(str(field) for field in contract["safe_fields"])
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for offset, category in enumerate(categories):
        keys = list(safe_fields)
        if category == "private_visibility":
            keys.append(str(contract["private_field"]))
        elif category == "hidden_reasoning":
            keys.append(str(contract["reasoning_field"]))
        elif category == "ambiguous_schema":
            keys = ["unresolved_payload", "versionless_event"]
        elif offset == 1:
            first = safe_fields[0]
            keys[0] = _aliases(first, "ood_test", seed)[0]
        targets = {key: (key if key in safe_fields else None) for key in keys}
        rows.append(
            _make_row(
                game=game,
                split=split,
                category=category,
                keys=keys,
                field_targets=targets,
                family=f"{split}_hazard_{category}_{offset}_v1",
                seed=seed,
                ordinal=start + offset,
            )
        )
    return rows


def _training_target(expected: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "parsed": bool(expected["should_parse"]),
        "game": expected["game"],
        "unit_type": expected["unit_type"],
        "mapped_fields": list(expected["mapped_fields"]),
        "disposition": expected["disposition"],
        "evidence_authority": expected["evidence_authority"],
        "failure_modes": list(expected["failure_modes"]),
        "abstained": bool(expected["should_abstain"]),
        "authority": "MODEL_RECOMMENDATION_ONLY",
    }


def _write_architecture_recipe_sets(
    config: Mapping[str, Any],
    candidate_dir: Path,
    training_labels: Sequence[Mapping[str, Any]],
    row_by_id: Mapping[str, Mapping[str, Any]],
    split_entries: Mapping[str, Mapping[str, Any]],
    training_labels_sha256: str,
) -> dict[str, Any]:
    recipe_dir = candidate_dir / "architecture_recipes"
    recipe_dir.mkdir()
    lora_rows: list[dict[str, Any]] = []
    hrm_rows: list[dict[str, Any]] = []
    recurrence = next(
        recipe
        for recipe in config["architecture_recipes"]
        if recipe["recipe_id"] == "hrm-text-schema-normalizer-v1"
    )["recurrence_grid"]
    for label in training_labels:
        row = row_by_id[str(label["item_id"])]
        input_record = {
            "game_hint": row["game_hint"],
            "source_schema": row["source_schema"],
            "source_shape": row["source_shape"],
        }
        target = _training_target(label["expected"])
        shared = {
            "example_id": row["item_id"],
            "record_hash": row["record_hash"],
            "split": row["split"],
            "label_authority": label["authority"],
        }
        lora_rows.append(
            {
                "schema_version": "game-adapter-lora-sft.v1",
                **shared,
                "instruction": (
                    "Normalize this public game-record shape. Fail closed on private, hidden, "
                    "ambiguous, or sealed material. Return only the target JSON contract."
                ),
                "input": input_record,
                "target": target,
            }
        )
        hrm_rows.append(
            {
                "schema_version": "game-adapter-hrm-text-sft.v1",
                **shared,
                "input_text": json.dumps(input_record, sort_keys=True, separators=(",", ":")),
                "target_text": json.dumps(target, sort_keys=True, separators=(",", ":")),
                "recurrence_budget_options": list(recurrence),
                "reasoning_trace_target_present": False,
            }
        )
    lora_path = recipe_dir / "lora_sft.jsonl"
    hrm_path = recipe_dir / "hrm_text_sft.jsonl"
    _write_jsonl(lora_path, lora_rows)
    _write_jsonl(hrm_path, hrm_rows)
    allowed_splits = sorted({str(label["split"]) for label in training_labels})
    manifest = {
        "schema_version": "game-adapter-architecture-recipe-pack.v1",
        "source_training_label_release_sha256": training_labels_sha256,
        "source_split_hashes": {
            split: split_entries[split]["sha256"] for split in allowed_splits
        },
        "allowed_splits": allowed_splits,
        "evaluation_splits_included": [],
        "examples_per_file": len(training_labels),
        "files": {
            "lora_schema_normalizer_sft": {
                "path": "lora_sft.jsonl",
                "sha256": sha256_file(lora_path),
                "status": "proposed_not_executed",
            },
            "hrm_text_schema_normalizer_sft": {
                "path": "hrm_text_sft.jsonl",
                "sha256": sha256_file(hrm_path),
                "status": "proposed_not_executed",
            },
        },
        "content_policy": config["generator"]["content_policy"],
        "claim_boundary": config["hypothesis"]["claim_scope"],
    }
    manifest_path = recipe_dir / "recipe_manifest.json"
    dump_structured(manifest_path, manifest)
    return {
        "path": "../candidate_workspace/architecture_recipes/recipe_manifest.json",
        "sha256": sha256_file(manifest_path),
        "evaluation_splits_included": [],
    }


def build_drift_benchmark(config_path: Path, out_dir: Path, *, seed: int = 101) -> dict[str, Any]:
    """Build shape-only rows, candidate-visible fit labels, and separately held evaluation labels."""
    config, _ = _load_contract(config_path)
    public_dir = out_dir / "public_dataset"
    candidate_dir = out_dir / "candidate_workspace"
    steward_dir = out_dir / "private_steward"
    public_dir.mkdir(parents=True, exist_ok=False)
    candidate_dir.mkdir(parents=True, exist_ok=False)
    steward_dir.mkdir(parents=True, exist_ok=False)
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in DRIFT_SPLITS}
    labels: list[dict[str, Any]] = []
    ordinal = 0
    for split in DRIFT_SPLITS:
        for game in GAMES:
            pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            if split != "sealed_poison":
                pairs.extend(_clean_rows(game, split, seed, ordinal))
                ordinal += len(pairs)
            hazards = _hazard_rows(game, split, seed, ordinal)
            ordinal += len(hazards)
            pairs.extend(hazards)
            for row, label in pairs:
                rows_by_split[split].append(row)
                labels.append(label)
    for rows in rows_by_split.values():
        rows.sort(key=lambda row: str(row["item_id"]))
    labels.sort(key=lambda row: str(row["item_id"]))
    _assert_split_isolation(config, rows_by_split)
    split_entries: dict[str, dict[str, Any]] = {}
    for split in DRIFT_SPLITS:
        path = public_dir / f"{split}.jsonl"
        _write_jsonl(path, rows_by_split[split])
        split_entries[split] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "count": len(rows_by_split[split]),
            "labels_in_file": False,
        }
    training_labels = [
        {**label, "authority": "ENVIRONMENT_OWNED_TRAINING_LABEL"}
        for label in labels
        if label["split"] in {"train", "calibration"}
    ]
    training_path = candidate_dir / "training_labels.jsonl"
    _write_jsonl(training_path, training_labels)
    row_by_id = {
        str(row["item_id"]): row
        for split in DRIFT_SPLITS
        for row in rows_by_split[split]
    }
    recipe_pack = _write_architecture_recipe_sets(
        config,
        candidate_dir,
        training_labels,
        row_by_id,
        split_entries,
        sha256_file(training_path),
    )
    steward_path = steward_dir / "ground_truth.jsonl"
    _write_jsonl(steward_path, labels)
    surface_rows = [
        {
            "split": split,
            "game_hint": row["game_hint"],
            "transformation_family": row["transformation_family"],
            "source_shape": row["source_shape"],
        }
        for split in DRIFT_SPLITS
        for row in rows_by_split[split]
    ]
    manifest = {
        "schema_version": "game-adapter-drift-dataset.v1",
        "benchmark_id": config["benchmark_id"],
        "dataset_id": f"{config['benchmark_id']}-seed-{seed}",
        "seed": seed,
        "config_sha256": sha256_file(config_path),
        "generator_version": config["generator"]["version"],
        "schema_surface_sha256": sha256_json(surface_rows),
        "games": list(GAMES),
        "splits": split_entries,
        "split_contract": config["split_contract"],
        "training_label_release": {
            "path": "../candidate_workspace/training_labels.jsonl",
            "sha256": sha256_file(training_path),
            "allowed_splits": ["train", "calibration"],
            "evaluation_labels_included": False,
        },
        "architecture_recipe_pack": recipe_pack,
        "ground_truth_custody": "separate private steward file; path not disclosed in manifest",
        "content_policy": config["generator"]["content_policy"],
    }
    manifest_path = public_dir / "dataset_manifest.json"
    dump_structured(manifest_path, manifest)
    shutil.copyfile(config_path, out_dir / "frozen_preregistration.yaml")
    dump_structured(
        steward_dir / "steward_receipt.json",
        {
            "schema_version": "game-adapter-drift-steward-receipt.v1",
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "ground_truth_sha256": sha256_file(steward_path),
            "training_label_release_sha256": sha256_file(training_path),
            "evaluation_labels_exposed_to_candidate": False,
            "authority": "ENVIRONMENT_OWNED",
        },
    )
    return manifest


def _assert_split_isolation(
    config: Mapping[str, Any], rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    lineages = {
        split: {str(row["lineage_group_sha256"]) for row in rows}
        for split, rows in rows_by_split.items()
    }
    for index, left in enumerate(DRIFT_SPLITS):
        for right in DRIFT_SPLITS[index + 1 :]:
            if lineages[left] & lineages[right]:
                raise ValueError(f"schema-drift lineage contamination: {left}/{right}")
    families = {
        split: {str(row["transformation_family"]) for row in rows}
        for split, rows in rows_by_split.items()
    }
    for pair in config["split_contract"]["require_disjoint_transformation_families"]:
        left, right = (str(value) for value in pair)
        if families[left] & families[right]:
            raise ValueError(f"schema-drift transformation-family contamination: {left}/{right}")


def _read_dataset(manifest_path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest = load_structured(manifest_path)
    if manifest.get("schema_version") != "game-adapter-drift-dataset.v1":
        raise ValueError("unsupported schema-drift dataset version")
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for split in DRIFT_SPLITS:
        entry = manifest.get("splits", {}).get(split)
        if not isinstance(entry, Mapping):
            raise ValueError(f"missing schema-drift split: {split}")
        path = manifest_path.parent / str(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"schema-drift split hash mismatch: {split}")
        rows = _read_jsonl(path)
        if len(rows) != int(entry["count"]):
            raise ValueError(f"schema-drift split count mismatch: {split}")
        for row in rows:
            item_id = str(row.get("item_id", ""))
            if row.get("split") != split or not item_id or item_id in seen:
                raise ValueError(f"invalid or duplicate schema-drift row: {item_id}")
            record = dict(row)
            record_hash = str(record.pop("record_hash", ""))
            if record_hash != sha256_json(record):
                raise ValueError(f"schema-drift record hash mismatch: {item_id}")
            seen.add(item_id)
        rows_by_split[split] = rows
    return manifest, rows_by_split


def _feature_vector(game: str, key: str, dimension: int, ngram_min: int, ngram_max: int) -> np.ndarray:
    normalized = _snake(key)
    padded = f"^{normalized}$"
    tokens = [f"game={game}", "bias", *(f"token={token}" for token in normalized.split("_"))]
    for size in range(ngram_min, ngram_max + 1):
        tokens.extend(f"g={padded[index:index + size]}" for index in range(len(padded) - size + 1))
    vector = np.zeros(dimension, dtype=np.float64)
    for token in tokens:
        digest = sha256_text(token)
        index = int(digest[:16], 16) % dimension
        sign = 1.0 if int(digest[16:18], 16) % 2 == 0 else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _load_training_labels(
    manifest_path: Path, manifest: Mapping[str, Any], row_by_id: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    release = manifest["training_label_release"]
    path = manifest_path.parent / str(release["path"])
    if sha256_file(path) != release["sha256"]:
        raise ValueError("training-label release hash mismatch")
    labels = _read_jsonl(path)
    allowed = set(str(value) for value in release["allowed_splits"])
    for label in labels:
        item_id = str(label.get("item_id", ""))
        row = row_by_id.get(item_id)
        if row is None or label.get("record_hash") != row["record_hash"]:
            raise ValueError(f"unbound training label: {item_id}")
        if label.get("split") not in allowed or row["split"] not in allowed:
            raise ValueError(f"evaluation label leaked into training release: {item_id}")
        if label.get("authority") != "ENVIRONMENT_OWNED_TRAINING_LABEL":
            raise ValueError(f"invalid training-label authority: {item_id}")
    if any(label["split"] in EVALUATION_SPLITS for label in labels):
        raise ValueError("evaluation labels are present in candidate training release")
    return labels


def _training_samples(
    labels: Sequence[Mapping[str, Any]], split: str
) -> list[tuple[str, str, str]]:
    samples: set[tuple[str, str, str]] = set()
    for label in labels:
        expected = label["expected"]
        if label["split"] != split or expected["should_abstain"] or not expected["should_parse"]:
            continue
        game = str(expected["game"])
        safe = set(str(value) for value in GAME_CONTRACTS[game]["safe_fields"])
        for key, target in label["field_targets"].items():
            if target in safe:
                samples.add((game, str(key), str(target)))
    return sorted(samples)


def _fit_linear_mapper(
    config: Mapping[str, Any], samples: Sequence[tuple[str, str, str]]
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    candidate = config["candidate"]
    dimension = int(candidate["feature_dimension"])
    classes = [f"{game}:{field}" for game in GAMES for field in GAME_CONTRACTS[game]["safe_fields"]]
    class_index = {label: index for index, label in enumerate(classes)}
    weights = np.zeros((len(classes), dimension), dtype=np.float64)
    initial_loss = 0.0
    final_loss = 0.0
    for epoch in range(int(candidate["epochs"])):
        epoch_loss = 0.0
        for game in GAMES:
            game_samples = [sample for sample in samples if sample[0] == game]
            indices = [index for index, label in enumerate(classes) if label.startswith(f"{game}:")]
            x = np.vstack(
                [
                    _feature_vector(
                        game,
                        key,
                        dimension,
                        int(candidate["character_ngram_min"]),
                        int(candidate["character_ngram_max"]),
                    )
                    for _, key, _ in game_samples
                ]
            )
            local_targets = np.asarray(
                [indices.index(class_index[f"{game}:{target}"]) for _, _, target in game_samples],
                dtype=np.int64,
            )
            local_weights = weights[indices]
            logits = x @ local_weights.T
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            epoch_loss += float(-np.log(probabilities[np.arange(len(x)), local_targets] + 1e-12).mean())
            probabilities[np.arange(len(x)), local_targets] -= 1.0
            gradient = probabilities.T @ x / len(x)
            gradient += float(candidate["l2"]) * local_weights
            weights[indices] -= float(candidate["learning_rate"]) * gradient
        if epoch == 0:
            initial_loss = epoch_loss / len(GAMES)
        final_loss = epoch_loss / len(GAMES)
    fit = {
        "samples": len(samples),
        "epochs": int(candidate["epochs"]),
        "steps_completed": int(candidate["epochs"]) * len(GAMES),
        "initial_cross_entropy": initial_loss,
        "final_cross_entropy": final_loss,
        "converged": final_loss < initial_loss,
    }
    return classes, weights, fit


def _map_field(
    config: Mapping[str, Any],
    classes: Sequence[str],
    weights: np.ndarray,
    game: str,
    key: str,
) -> tuple[str, float]:
    candidate = config["candidate"]
    indices = [index for index, label in enumerate(classes) if label.startswith(f"{game}:")]
    vector = _feature_vector(
        game,
        key,
        int(candidate["feature_dimension"]),
        int(candidate["character_ngram_min"]),
        int(candidate["character_ngram_max"]),
    )
    logits = weights[indices] @ vector
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    local = int(np.argmax(probabilities))
    return classes[indices[local]].split(":", 1)[1], float(probabilities[local])


def _calibrate_threshold(
    config: Mapping[str, Any],
    classes: Sequence[str],
    weights: np.ndarray,
    samples: Sequence[tuple[str, str, str]],
) -> tuple[float, dict[str, Any]]:
    scored = [
        (*_map_field(config, classes, weights, game, key), target)
        for game, key, target in samples
    ]
    candidates = [value / 100 for value in range(0, 96, 5)]
    ranked: list[tuple[int, int, float]] = []
    for threshold in candidates:
        correct = sum(predicted == target and confidence >= threshold for predicted, confidence, target in scored)
        incorrect = sum(predicted != target and confidence >= threshold for predicted, confidence, target in scored)
        ranked.append((correct - 2 * incorrect, correct + incorrect, -threshold))
    _, _, negative_threshold = max(ranked)
    threshold = -negative_threshold
    accepted = [(predicted, target) for predicted, confidence, target in scored if confidence >= threshold]
    return threshold, {
        "examples": len(scored),
        "threshold": threshold,
        "coverage": len(accepted) / len(scored) if scored else 0.0,
        "accepted_accuracy": (
            sum(predicted == target for predicted, target in accepted) / len(accepted) if accepted else 0.0
        ),
    }


def _normalized_forbidden(policy: Mapping[str, Any]) -> set[str]:
    return {_snake(str(value)) for value in policy["forbidden_fields"]}


def _policy_prediction(
    row: Mapping[str, Any],
    setup_id: str,
    policy: Mapping[str, Any],
    *,
    mapper: Any | None = None,
    threshold: float = 0.0,
) -> dict[str, Any]:
    game = str(row["game_hint"])
    contract = GAME_CONTRACTS[game]
    shape = row["source_shape"]
    flags = shape["flags"]
    keys = [str(value) for value in shape["keys"]]
    base = {
        "schema_version": "game-adapter-drift-prediction.v1",
        "item_id": row["item_id"],
        "record_hash": row["record_hash"],
        "setup_id": setup_id,
        "parsed": True,
        "game": game,
        "unit_type": contract["unit_type"],
        "authority": "MODEL_RECOMMENDATION_ONLY",
    }
    if not flags["schema_supported"]:
        return {
            **base,
            "parsed": False,
            "mapped_fields": [],
            "disposition": "QUARANTINE",
            "evidence_authority": "UNKNOWN",
            "failure_modes": ["unknown"],
            "abstained": True,
            "confidence": 0.99,
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
    if flags["explicit_hidden_reasoning"]:
        return {
            **base,
            "mapped_fields": [],
            "disposition": "REJECT",
            "evidence_authority": "UNKNOWN",
            "failure_modes": ["hidden_reasoning"],
            "abstained": True,
            "confidence": 0.99,
        }
    if shape["visibility"] != "public" or flags["private_visibility"]:
        return {
            **base,
            "mapped_fields": [],
            "disposition": "QUARANTINE",
            "evidence_authority": "UNKNOWN",
            "failure_modes": ["private_data"],
            "abstained": True,
            "confidence": 0.99,
        }
    if any(_snake(key) in _normalized_forbidden(policy) for key in keys):
        return {
            **base,
            "mapped_fields": [],
            "disposition": "REJECT",
            "evidence_authority": "UNKNOWN",
            "failure_modes": ["hidden_reasoning"],
            "abstained": True,
            "confidence": 0.99,
        }
    safe = set(str(value) for value in contract["safe_fields"])
    mapped = {key for key in keys if key in safe}
    confidences: list[float] = [1.0 for key in mapped]
    if mapper is not None:
        for key in keys:
            if key in safe:
                continue
            target, confidence = mapper(game, key)
            confidences.append(confidence)
            if confidence >= threshold:
                mapped.add(target)
    return {
        **base,
        "mapped_fields": sorted(mapped),
        "disposition": "VERIFIED_TRAIN",
        "evidence_authority": "ENVIRONMENT_VERIFIED",
        "failure_modes": [],
        "abstained": False,
        "confidence": min(confidences, default=1.0),
    }


def fit_drift_adapter(
    config_path: Path,
    dataset_manifest_path: Path,
    out_dir: Path,
    *,
    setup_id: str = "sparse-char-ngram-v1",
) -> dict[str, Any]:
    """Fit only on the released train labels, calibrate on calibration, then predict public evaluation rows."""
    config, policy = _load_contract(config_path)
    manifest, rows_by_split = _read_dataset(dataset_manifest_path)
    if manifest["config_sha256"] != sha256_file(config_path):
        raise ValueError("schema-drift dataset is not bound to this preregistration")
    out_dir.mkdir(parents=True, exist_ok=False)
    all_rows = [row for split in DRIFT_SPLITS for row in rows_by_split[split]]
    row_by_id = {str(row["item_id"]): row for row in all_rows}
    labels = _load_training_labels(dataset_manifest_path, manifest, row_by_id)
    train_samples = _training_samples(labels, "train")
    calibration_samples = _training_samples(labels, "calibration")
    classes, weights, fit = _fit_linear_mapper(config, train_samples)
    threshold, calibration = _calibrate_threshold(
        config, classes, weights, calibration_samples
    )
    model = {
        "schema_version": "sparse-character-ngram-adapter.v1",
        "classes": classes,
        "feature_dimension": int(config["candidate"]["feature_dimension"]),
        "character_ngram_range": [
            int(config["candidate"]["character_ngram_min"]),
            int(config["candidate"]["character_ngram_max"]),
        ],
        "threshold": threshold,
        "weights": np.round(weights, 12).tolist(),
    }
    model_path = out_dir / "model_weights.json"
    dump_structured(model_path, model)
    def mapper(game: str, key: str) -> tuple[str, float]:
        return _map_field(config, classes, weights, game, key)
    predictions = [
        _policy_prediction(row, setup_id, policy, mapper=mapper, threshold=threshold)
        for row in all_rows
    ]
    predictions_path = out_dir / "predictions.jsonl"
    _write_jsonl(predictions_path, predictions)
    fit_receipt = {
        "schema_version": "game-adapter-drift-fit-receipt.v1",
        "setup_id": setup_id,
        "status": "completed",
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "training_label_release_sha256": manifest["training_label_release"]["sha256"],
        "fitting_splits": ["train"],
        "calibration_splits": ["calibration"],
        "evaluation_splits_read_as_labels": [],
        "evaluation_public_shapes_read_for_inference": list(EVALUATION_SPLITS),
        "fit": fit,
        "calibration": calibration,
        "trainable_parameters": int(weights.size),
        "maximum_trainable_parameters": int(config["candidate"]["maximum_trainable_parameters"]),
        "model_weights_sha256": sha256_file(model_path),
        "predictions_sha256": sha256_file(predictions_path),
        "network_used": False,
        "raw_corpus_values_read": False,
        "checkpoint_downloaded": False,
    }
    if fit_receipt["trainable_parameters"] > fit_receipt["maximum_trainable_parameters"]:
        raise ValueError("schema-drift candidate exceeded preregistered parameter cap")
    fit_path = out_dir / "fit_receipt.json"
    dump_structured(fit_path, fit_receipt)
    candidate_manifest = {
        "schema_version": "game-adapter-drift-candidate.v1",
        "setup_id": setup_id,
        "adapter_family": config["candidate"]["family"],
        "adapter_version": "1.0.0",
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "config_sha256": sha256_file(config_path),
        "mock": False,
        "weights_trained": True,
        "authorized_training_label_access": True,
        "evaluation_label_access": False,
        "input_contract": config["candidate"]["model_input_contract"],
        "training_label_release_sha256": manifest["training_label_release"]["sha256"],
        "model_weights_sha256": sha256_file(model_path),
        "fit_receipt_sha256": sha256_file(fit_path),
        "training_splits": ["train"],
        "calibration_splits": ["calibration"],
    }
    dump_structured(out_dir / "candidate_manifest.json", candidate_manifest)
    return candidate_manifest


def run_exact_key_baseline(
    config_path: Path,
    dataset_manifest_path: Path,
    out_dir: Path,
    *,
    setup_id: str = "frozen-exact-key-policy-v1",
) -> dict[str, Any]:
    config, policy = _load_contract(config_path)
    manifest, rows_by_split = _read_dataset(dataset_manifest_path)
    if manifest["config_sha256"] != sha256_file(config_path):
        raise ValueError("schema-drift dataset is not bound to this preregistration")
    out_dir.mkdir(parents=True, exist_ok=False)
    rows = [row for split in DRIFT_SPLITS for row in rows_by_split[split]]
    predictions_path = out_dir / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [_policy_prediction(row, setup_id, policy) for row in rows],
    )
    candidate_manifest = {
        "schema_version": "game-adapter-drift-candidate.v1",
        "setup_id": setup_id,
        "adapter_family": "frozen_exact_key_policy",
        "adapter_version": "1.0.0",
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "config_sha256": sha256_file(config_path),
        "mock": False,
        "weights_trained": False,
        "authorized_training_label_access": False,
        "evaluation_label_access": False,
        "input_contract": config["candidate"]["model_input_contract"],
        "training_label_release_sha256": None,
        "model_weights_sha256": None,
        "fit_receipt_sha256": None,
        "training_splits": [],
        "calibration_splits": [],
    }
    dump_structured(out_dir / "candidate_manifest.json", candidate_manifest)
    return candidate_manifest


def _read_steward_labels(
    path: Path, row_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for label in _read_jsonl(path):
        item_id = str(label.get("item_id", ""))
        if item_id not in row_by_id or item_id in labels:
            raise ValueError(f"unknown or duplicate drift steward label: {item_id}")
        if label.get("authority") != "ENVIRONMENT_OWNED":
            raise ValueError(f"untrusted drift steward label: {item_id}")
        if label.get("record_hash") != row_by_id[item_id]["record_hash"]:
            raise ValueError(f"drift steward label hash mismatch: {item_id}")
        labels[item_id] = label
    if set(labels) != set(row_by_id):
        raise ValueError("drift steward label set does not match dataset")
    return labels


def _read_predictions(
    path: Path, setup_id: str, row_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for prediction in _read_jsonl(path):
        item_id = str(prediction.get("item_id", ""))
        if item_id not in row_by_id or item_id in predictions:
            raise ValueError(f"unknown or duplicate drift prediction: {item_id}")
        if prediction.get("setup_id") != setup_id:
            raise ValueError(f"mixed drift prediction setup: {item_id}")
        if prediction.get("authority") != "MODEL_RECOMMENDATION_ONLY":
            raise ValueError(f"drift candidate asserted decision authority: {item_id}")
        if prediction.get("record_hash") != row_by_id[item_id]["record_hash"]:
            raise ValueError(f"drift prediction record hash mismatch: {item_id}")
        confidence = prediction.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise ValueError(f"invalid drift confidence: {item_id}")
        if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"drift confidence outside [0,1]: {item_id}")
        predictions[item_id] = prediction
    if set(predictions) != set(row_by_id):
        raise ValueError("drift prediction set does not match dataset")
    return predictions


def _score(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    prediction: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    parsed = bool(prediction.get("parsed"))
    mapped = set(str(value) for value in prediction.get("mapped_fields", ())) if parsed else set()
    expected_mapped = set(str(value) for value in expected["mapped_fields"])
    forbidden = _normalized_forbidden(policy)
    leaked = {value for value in mapped if _snake(value) in forbidden}
    true_positive = len(mapped & expected_mapped)
    precision = true_positive / len(mapped) if mapped else (1.0 if not expected_mapped else 0.0)
    recall = true_positive / len(expected_mapped) if expected_mapped else (1.0 if not mapped else 0.0)
    mapping_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = bool(
        parsed == bool(expected["should_parse"])
        and prediction.get("game") == expected["game"]
        and prediction.get("unit_type") == expected["unit_type"]
        and mapped == expected_mapped
        and prediction.get("disposition") == expected["disposition"]
        and prediction.get("evidence_authority") == expected["evidence_authority"]
        and set(prediction.get("failure_modes", ())) == set(expected["failure_modes"])
        and bool(prediction.get("abstained")) == bool(expected["should_abstain"])
        and not leaked
    )
    release_hazard = expected["disposition"] != "VERIFIED_TRAIN"
    false_release = release_hazard and prediction.get("disposition") == "VERIFIED_TRAIN"
    confidence = float(prediction["confidence"])
    errors: list[str] = []
    if parsed != bool(expected["should_parse"]):
        errors.append("parse_status_mismatch")
    if mapped != expected_mapped:
        errors.append("field_mapping_mismatch")
    if prediction.get("disposition") != expected["disposition"]:
        errors.append("disposition_mismatch")
    if leaked:
        errors.append("privacy_leak")
    if false_release:
        errors.append("critical_false_release")
    return {
        "item_id": row["item_id"],
        "split": row["split"],
        "game": expected["game"],
        "exact": exact,
        "mapping_f1": mapping_f1,
        "abstained": bool(prediction.get("abstained")),
        "abstention_correct": bool(prediction.get("abstained")) == bool(expected["should_abstain"]),
        "privacy_leak": bool(leaked),
        "release_hazard": release_hazard,
        "critical_false_release": false_release,
        "confidence": confidence,
        "brier": (confidence - float(exact)) ** 2,
        "error_codes": errors,
    }


def _aggregate(scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not scored:
        raise ValueError("cannot aggregate empty schema-drift slice")
    count = len(scored)
    hazards = [row for row in scored if row["release_hazard"]]
    errors = Counter(error for row in scored for error in row["error_codes"])
    return {
        "examples": count,
        "exact_record_accuracy": sum(bool(row["exact"]) for row in scored) / count,
        "mapping_f1": float(np.mean([float(row["mapping_f1"]) for row in scored])),
        "abstention_accuracy": sum(bool(row["abstention_correct"]) for row in scored) / count,
        "coverage": sum(not bool(row["abstained"]) for row in scored) / count,
        "privacy_leak_rate": sum(bool(row["privacy_leak"]) for row in scored) / count,
        "critical_false_release_rate": (
            sum(bool(row["critical_false_release"]) for row in hazards) / len(hazards) if hazards else 0.0
        ),
        "brier_score": float(np.mean([float(row["brier"]) for row in scored])),
        "error_counts": dict(sorted(errors.items())),
    }


def evaluate_drift_setup(
    config_path: Path,
    dataset_manifest_path: Path,
    steward_labels_path: Path,
    candidate_manifest_path: Path,
    predictions_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Score a bound setup; the steward labels enter only this evaluator."""
    config, policy = _load_contract(config_path)
    manifest, rows_by_split = _read_dataset(dataset_manifest_path)
    all_rows = [row for split in DRIFT_SPLITS for row in rows_by_split[split]]
    row_by_id = {str(row["item_id"]): row for row in all_rows}
    labels = _read_steward_labels(steward_labels_path, row_by_id)
    candidate = load_structured(candidate_manifest_path)
    if candidate.get("schema_version") != "game-adapter-drift-candidate.v1":
        raise ValueError("unsupported schema-drift candidate version")
    if candidate.get("dataset_manifest_sha256") != sha256_file(dataset_manifest_path):
        raise ValueError("schema-drift candidate is not bound to this dataset")
    if candidate.get("config_sha256") != sha256_file(config_path):
        raise ValueError("schema-drift candidate is not bound to this preregistration")
    if candidate.get("input_contract") != "public_shape_only":
        raise ValueError("schema-drift candidate requested a forbidden input contract")
    if candidate.get("evaluation_label_access"):
        raise ValueError("schema-drift candidate declares evaluation-label access")
    steward_hash = sha256_file(steward_labels_path)
    if steward_hash in json.dumps(candidate, sort_keys=True):
        raise ValueError("schema-drift candidate manifest contains the steward label hash")
    setup_id = str(candidate["setup_id"])
    predictions = _read_predictions(predictions_path, setup_id, row_by_id)
    scored = [
        _score(row, labels[str(row["item_id"])]["expected"], predictions[str(row["item_id"])], policy)
        for row in all_rows
    ]
    split_results = {
        split: _aggregate([row for row in scored if row["split"] == split])
        for split in DRIFT_SPLITS
    }
    evaluation_rows = [row for row in scored if row["split"] in EVALUATION_SPLITS]
    game_results = {
        game: _aggregate([row for row in evaluation_rows if row["game"] == game])
        for game in GAMES
    }
    report = {
        "schema_version": "game-adapter-drift-evaluation.v1",
        "setup_id": setup_id,
        "adapter_family": candidate["adapter_family"],
        "evidence_status": (
            "trained_shape_adapter_evidence"
            if candidate.get("weights_trained")
            else "frozen_policy_baseline"
        ),
        "diagnostic_results": {
            split: split_results[split] for split in ("train", "calibration")
        },
        "split_results": {
            split: split_results[split] for split in EVALUATION_SPLITS
        },
        "evaluation_overall": _aggregate(evaluation_rows),
        "game_results": game_results,
        "custody_checks": {
            "dataset_bound": True,
            "config_bound": manifest["config_sha256"] == sha256_file(config_path),
            "evaluation_labels_declared_unseen": not bool(candidate["evaluation_label_access"]),
            "public_shape_only": candidate["input_contract"] == "public_shape_only",
        },
        "reproducibility": {
            "config_sha256": sha256_file(config_path),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "steward_labels_sha256": steward_hash,
            "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
            "predictions_sha256": sha256_file(predictions_path),
        },
        "claim_boundary": config["hypothesis"]["claim_scope"],
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    dump_structured(out_dir / "evaluation_report.json", report)
    _write_jsonl(
        out_dir / "scored_receipts.jsonl",
        [
            {
                "item_id": row["item_id"],
                "split": row["split"],
                "game": row["game"],
                "exact": row["exact"],
                "privacy_leak": row["privacy_leak"],
                "critical_false_release": row["critical_false_release"],
                "error_codes": row["error_codes"],
            }
            for row in scored
        ],
    )
    return report


def _compare_reports(
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["promotion_gates"]
    ordinary_gain = (
        candidate["split_results"]["ordinary_test"]["exact_record_accuracy"]
        - baseline["split_results"]["ordinary_test"]["exact_record_accuracy"]
    )
    ood_gain = (
        candidate["split_results"]["ood_test"]["exact_record_accuracy"]
        - baseline["split_results"]["ood_test"]["exact_record_accuracy"]
    )
    per_game_gain = {
        game: (
            candidate["game_results"][game]["exact_record_accuracy"]
            - baseline["game_results"][game]["exact_record_accuracy"]
        )
        for game in GAMES
    }
    overall = candidate["evaluation_overall"]
    checks = {
        "weights_trained": candidate["evidence_status"] == "trained_shape_adapter_evidence",
        "evaluation_labels_unseen": candidate["custody_checks"]["evaluation_labels_declared_unseen"],
        "privacy_leak_rate": overall["privacy_leak_rate"] <= float(gates["maximum_privacy_leak_rate"]),
        "critical_false_release_rate": (
            overall["critical_false_release_rate"]
            <= float(gates["maximum_critical_false_release_rate"])
        ),
        "sealed_exact": (
            candidate["split_results"]["sealed_poison"]["exact_record_accuracy"]
            >= float(gates["minimum_sealed_exact_record_accuracy"])
        ),
        "ordinary_exact": (
            candidate["split_results"]["ordinary_test"]["exact_record_accuracy"]
            >= float(gates["minimum_ordinary_exact_record_accuracy"])
        ),
        "ood_exact": (
            candidate["split_results"]["ood_test"]["exact_record_accuracy"]
            >= float(gates["minimum_ood_exact_record_accuracy"])
        ),
        "ordinary_gain": ordinary_gain
        >= float(gates["minimum_ordinary_exact_gain_over_exact_key"]),
        "ood_gain": ood_gain >= float(gates["minimum_ood_exact_gain_over_exact_key"]),
        "no_game_regression": (
            not bool(gates["require_nonnegative_exact_gain_every_game"])
            or all(gain >= 0.0 for gain in per_game_gain.values())
        ),
    }
    return {
        "schema_version": "game-adapter-drift-comparison.v1",
        "baseline_setup_id": baseline["setup_id"],
        "candidate_setup_id": candidate["setup_id"],
        "ordinary_exact_gain": ordinary_gain,
        "ood_exact_gain": ood_gain,
        "per_game_evaluation_exact_gain": per_game_gain,
        "promotion_assessment": {
            "eligible_for_next_architecture_gate": all(checks.values()),
            "checks": checks,
            "authority": "benchmark advancement only; never corpus release authority",
        },
        "claim_boundary": config["hypothesis"]["claim_scope"],
    }


def _render_comparison(
    comparison: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> str:
    lines = [
        "# Schema-drift adapter gym",
        "",
        "| Setup | Ordinary exact | OOD exact | Sealed exact | Privacy leak | False release |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for report in (baseline, candidate):
        ordinary = report["split_results"]["ordinary_test"]
        ood = report["split_results"]["ood_test"]
        sealed = report["split_results"]["sealed_poison"]
        overall = report["evaluation_overall"]
        lines.append(
            f"| {report['setup_id']} | {ordinary['exact_record_accuracy']:.3f} | "
            f"{ood['exact_record_accuracy']:.3f} | {sealed['exact_record_accuracy']:.3f} | "
            f"{overall['privacy_leak_rate']:.3f} | {overall['critical_false_release_rate']:.3f} |"
        )
    lines.extend(
        (
            "",
            f"Ordinary exact gain: `{comparison['ordinary_exact_gain']:.3f}`; "
            f"OOD exact gain: `{comparison['ood_exact_gain']:.3f}`.",
            "",
            "Eligible for the next architecture gate: "
            f"`{comparison['promotion_assessment']['eligible_for_next_architecture_gate']}`.",
            "",
            str(comparison["promotion_assessment"]["authority"]),
            "",
            "This benchmark measures synthetic shape normalization only. It does not authorize "
            "training on private game records.",
            "",
        )
    )
    return "\n".join(lines)


def run_drift_gym(config_path: Path, out_dir: Path, *, seed: int = 101) -> dict[str, Any]:
    """Build, fit, score, and compare one deterministic schema-drift gym run."""
    config, _ = _load_contract(config_path)
    out_dir.mkdir(parents=True, exist_ok=False)
    build_drift_benchmark(config_path, out_dir, seed=seed)
    manifest_path = out_dir / "public_dataset" / "dataset_manifest.json"
    steward_path = out_dir / "private_steward" / "ground_truth.jsonl"
    baseline_dir = out_dir / "setups" / "frozen-exact-key-policy-v1"
    candidate_dir = out_dir / "setups" / "sparse-char-ngram-v1"
    run_exact_key_baseline(config_path, manifest_path, baseline_dir)
    fit_drift_adapter(config_path, manifest_path, candidate_dir)
    baseline_report = evaluate_drift_setup(
        config_path,
        manifest_path,
        steward_path,
        baseline_dir / "candidate_manifest.json",
        baseline_dir / "predictions.jsonl",
        out_dir / "evaluations" / "frozen-exact-key-policy-v1",
    )
    candidate_report = evaluate_drift_setup(
        config_path,
        manifest_path,
        steward_path,
        candidate_dir / "candidate_manifest.json",
        candidate_dir / "predictions.jsonl",
        out_dir / "evaluations" / "sparse-char-ngram-v1",
    )
    comparison = _compare_reports(config, baseline_report, candidate_report)
    comparison["seed"] = seed
    comparison["config_sha256"] = sha256_file(config_path)
    comparison["schema_surface_sha256"] = load_structured(manifest_path)["schema_surface_sha256"]
    dump_structured(out_dir / "comparison.json", comparison)
    (out_dir / "comparison.md").write_text(
        _render_comparison(comparison, baseline_report, candidate_report),
        encoding="utf-8",
    )
    dump_structured(
        out_dir / "run_receipt.json",
        {
            "schema_version": "game-adapter-drift-run-receipt.v1",
            "seed": seed,
            "config_sha256": sha256_file(config_path),
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "schema_surface_sha256": comparison["schema_surface_sha256"],
            "comparison_sha256": sha256_file(out_dir / "comparison.json"),
            "network_used": False,
            "raw_corpus_values_read": False,
            "private_backend_workspace_read": False,
            "evaluation_labels_entered_fit_process": False,
            "generated_artifacts_must_not_be_committed": True,
        },
    )
    return comparison


def run_drift_confirmation(config_path: Path, out_dir: Path) -> dict[str, Any]:
    """Run the preregistered pilot and fresh-seed confirmation without changing the contract."""
    config, _ = _load_contract(config_path)
    out_dir.mkdir(parents=True, exist_ok=False)
    seeds = {
        "pilot": int(config["confirmation"]["pilot_seed"]),
        "fresh_confirmation": int(config["confirmation"]["fresh_confirmation_seed"]),
    }
    runs = {
        name: run_drift_gym(config_path, out_dir / name, seed=seed)
        for name, seed in seeds.items()
    }
    config_hashes = {str(report["config_sha256"]) for report in runs.values()}
    schema_surfaces = {str(report["schema_surface_sha256"]) for report in runs.values()}
    checks = {
        "config_unchanged": len(config_hashes) == 1,
        "schema_surface_differs": (
            not bool(config["confirmation"]["schema_surface_must_differ"])
            or len(schema_surfaces) == len(runs)
        ),
        "pilot_passed": runs["pilot"]["promotion_assessment"]["eligible_for_next_architecture_gate"],
        "fresh_confirmation_passed": runs["fresh_confirmation"]["promotion_assessment"][
            "eligible_for_next_architecture_gate"
        ],
    }
    report = {
        "schema_version": "game-adapter-drift-confirmation.v1",
        "benchmark_id": config["benchmark_id"],
        "config_sha256": sha256_file(config_path),
        "runs": {
            name: {
                "seed": report["seed"],
                "comparison_sha256": sha256_file(out_dir / name / "comparison.json"),
                "schema_surface_sha256": report["schema_surface_sha256"],
                "ordinary_exact_gain": report["ordinary_exact_gain"],
                "ood_exact_gain": report["ood_exact_gain"],
                "passed": report["promotion_assessment"]["eligible_for_next_architecture_gate"],
            }
            for name, report in runs.items()
        },
        "checks": checks,
        "confirmed_for_next_architecture_gate": all(checks.values()),
        "claim_boundary": config["hypothesis"]["claim_scope"],
    }
    dump_structured(out_dir / "confirmation_report.json", report)
    return report
