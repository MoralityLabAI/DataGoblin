from __future__ import annotations

import copy
import json
import math
import random
from array import array
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_sorter import (
    BUCKET_RANK,
    METRIC_FIELDS,
    sha256_file,
    sha256_json,
)
from datagoblin.hashing import sha256_text
from datagoblin.serialization import dump_structured, load_structured

HARD_PREFERENCES = ("LEFT", "RIGHT", "TRADEOFF", "ABSTAIN")
HARD_RATIONALES = (
    "safety_gate",
    "pareto_dominance",
    "parameter_efficiency",
    "irreducible_tradeoff",
    "insufficient_evidence",
)
ARRAY_STATUSES = ("SELECTED", "DOMINATED", "UNRESOLVED", "UNSAFE")
HARD_FEATURE_WHITELIST = (
    "game",
    "fixture_pressure",
    "context_mode",
    "optimization_pressure",
    "trainable_parameter_bucket",
    "safety_qualified",
    *METRIC_FIELDS,
)
HARD_INVERT_PREFERENCE = {
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
    "TRADEOFF": "TRADEOFF",
    "ABSTAIN": "ABSTAIN",
}


def _interval(value: float, width: float = 0.0) -> dict[str, float]:
    return {
        "lower": max(0.0, value - width),
        "upper": min(1.0, value + width),
    }


def _missing_interval() -> dict[str, None]:
    return {"lower": None, "upper": None}


def _validate_interval(value: Mapping[str, Any]) -> None:
    if set(value) != {"lower", "upper"}:
        raise ValueError("metric interval must contain exactly lower and upper")
    lower = value["lower"]
    upper = value["upper"]
    if lower is None or upper is None:
        if lower is not None or upper is not None:
            raise ValueError("partially missing intervals are forbidden")
        return
    lower_value = float(lower)
    upper_value = float(upper)
    if (
        not math.isfinite(lower_value)
        or not math.isfinite(upper_value)
        or not 0.0 <= lower_value <= upper_value <= 1.0
    ):
        raise ValueError("metric interval bounds are invalid")


def validate_hard_side(side: Mapping[str, Any]) -> None:
    if set(side) != set(HARD_FEATURE_WHITELIST):
        raise ValueError("hard-surface side differs from the public feature contract")
    if not isinstance(side["safety_qualified"], bool):
        raise ValueError("safety_qualified must be boolean")
    if str(side["trainable_parameter_bucket"]) not in BUCKET_RANK:
        raise ValueError("unknown trainable parameter bucket")
    for field in METRIC_FIELDS:
        if not isinstance(side[field], Mapping):
            raise ValueError(f"{field} must be an interval object")
        _validate_interval(side[field])


def _known_bounds(side: Mapping[str, Any]) -> list[tuple[float, float]] | None:
    values: list[tuple[float, float]] = []
    for field in METRIC_FIELDS:
        interval = side[field]
        if interval["lower"] is None or interval["upper"] is None:
            return None
        values.append((float(interval["lower"]), float(interval["upper"])))
    return values


def interval_taste_policy(
    input_: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float = 0.01,
) -> tuple[str, str]:
    left = input_["left"]
    right = input_["right"]
    left_safe = bool(left["safety_qualified"])
    right_safe = bool(right["safety_qualified"])
    if left_safe != right_safe:
        return ("LEFT" if left_safe else "RIGHT"), "safety_gate"
    left_bounds = _known_bounds(left)
    right_bounds = _known_bounds(right)
    if left_bounds is None or right_bounds is None:
        return "ABSTAIN", "insufficient_evidence"
    left_dominates = all(
        left_lower >= right_upper - tolerance
        for (left_lower, _), (_, right_upper) in zip(
            left_bounds, right_bounds, strict=True
        )
    ) and any(
        left_lower > right_upper + tolerance
        for (left_lower, _), (_, right_upper) in zip(
            left_bounds, right_bounds, strict=True
        )
    )
    right_dominates = all(
        right_lower >= left_upper - tolerance
        for (_, left_upper), (right_lower, _) in zip(
            left_bounds, right_bounds, strict=True
        )
    ) and any(
        right_lower > left_upper + tolerance
        for (_, left_upper), (right_lower, _) in zip(
            left_bounds, right_bounds, strict=True
        )
    )
    if left_dominates != right_dominates:
        return ("LEFT" if left_dominates else "RIGHT"), "pareto_dominance"
    equivalent = all(
        max(abs(left_lower - right_upper), abs(left_upper - right_lower))
        <= tolerance
        for (left_lower, left_upper), (right_lower, right_upper) in zip(
            left_bounds, right_bounds, strict=True
        )
    )
    left_cost = BUCKET_RANK[str(left["trainable_parameter_bucket"])]
    right_cost = BUCKET_RANK[str(right["trainable_parameter_bucket"])]
    if equivalent and left_cost != right_cost:
        return ("LEFT" if left_cost < right_cost else "RIGHT"), "parameter_efficiency"
    interval_width_present = any(
        lower != upper for lower, upper in (*left_bounds, *right_bounds)
    )
    if interval_width_present:
        return "ABSTAIN", "insufficient_evidence"
    return "TRADEOFF", "irreducible_tradeoff"


def swap_hard_pair(row: Mapping[str, Any]) -> dict[str, Any]:
    swapped = copy.deepcopy(dict(row))
    swapped["input"] = {
        "left": copy.deepcopy(row["input"]["right"]),
        "right": copy.deepcopy(row["input"]["left"]),
    }
    if "left_index" in swapped:
        swapped["left_index"], swapped["right_index"] = (
            swapped["right_index"],
            swapped["left_index"],
        )
    if "target" in swapped:
        swapped["target"]["preference"] = HARD_INVERT_PREFERENCE[
            str(row["target"]["preference"])
        ]
    return swapped


def _candidate(
    *,
    slot: int,
    pressure: str,
    game: str,
    split: str,
    jitter: float,
    context_prefix: str,
) -> dict[str, Any]:
    base = (0.78 + jitter, 0.80 + jitter, 0.78 + jitter, 0.80 + jitter)
    dominant = tuple(min(0.97, value + 0.08) for value in base)
    values: tuple[dict[str, Any], ...]
    if slot == 0:
        metrics = tuple(_interval(value) for value in base)
        mode = "baseline"
        optimization = "balanced"
        bucket = "medium"
        safe = True
    elif slot == 1:
        metrics = tuple(_interval(value) for value in dominant)
        mode = "dominant"
        optimization = "quality"
        bucket = "large"
        safe = True
    elif slot == 2:
        vector = (0.93 + jitter, 0.92 + jitter, 0.69 + jitter, 0.70 + jitter)
        metrics = tuple(_interval(max(0.0, min(1.0, value))) for value in vector)
        mode = "ordinary_specialist"
        optimization = "ordinary_quality"
        bucket = "small"
        safe = True
    elif slot == 3:
        metrics = tuple(_interval(0.99) for _ in METRIC_FIELDS)
        mode = "unsafe_high_coverage"
        optimization = "coverage_without_safety"
        bucket = "zero"
        safe = False
    elif pressure == "tolerance_edge":
        metrics = tuple(_interval(value + 0.004) for value in dominant)
        mode = "tolerance_neighbor"
        optimization = "parameter_efficiency"
        bucket = "small"
        safe = True
    elif pressure == "interval_uncertainty":
        metrics = tuple(_interval(value, 0.025) for value in dominant)
        mode = "interval_candidate"
        optimization = "bounded_uncertainty"
        bucket = "medium"
        safe = True
    elif pressure == "missing_evidence":
        provisional = [_interval(value + 0.02) for value in dominant]
        provisional[(len(game) + len(split)) % len(METRIC_FIELDS)] = _missing_interval()
        metrics = tuple(provisional)
        mode = "missing_metric_candidate"
        optimization = "incomplete_evidence"
        bucket = "medium"
        safe = True
    elif pressure == "mixed_pareto":
        vector = (0.68 + jitter, 0.70 + jitter, 0.95 + jitter, 0.94 + jitter)
        metrics = tuple(_interval(max(0.0, min(1.0, value))) for value in vector)
        mode = "robustness_specialist"
        optimization = "ood_robustness"
        bucket = "zero"
        safe = True
    else:
        raise ValueError(f"unsupported hard-surface pressure: {pressure}")
    values = metrics
    result: dict[str, Any] = {
        "game": game,
        "fixture_pressure": pressure,
        "context_mode": f"{context_prefix}:{mode}",
        "optimization_pressure": optimization,
        "trainable_parameter_bucket": bucket,
        "safety_qualified": safe,
    }
    result.update(
        {field: value for field, value in zip(METRIC_FIELDS, values, strict=True)}
    )
    validate_hard_side(result)
    return result


def derive_array_target(
    candidates: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    incoming: set[int] = set()
    uncertain: set[int] = set()
    safe_indices = [
        index for index, candidate in enumerate(candidates) if candidate["safety_qualified"]
    ]
    for offset, left_index in enumerate(safe_indices):
        for right_index in safe_indices[offset + 1 :]:
            preference, _ = interval_taste_policy(
                {
                    "left": candidates[left_index],
                    "right": candidates[right_index],
                },
                tolerance=tolerance,
            )
            if preference == "LEFT":
                incoming.add(right_index)
            elif preference == "RIGHT":
                incoming.add(left_index)
            elif preference == "ABSTAIN":
                uncertain.update((left_index, right_index))
    statuses = []
    for index, candidate in enumerate(candidates):
        if not candidate["safety_qualified"]:
            status = "UNSAFE"
        elif index in incoming:
            status = "DOMINATED"
        elif index in uncertain:
            status = "UNRESOLVED"
        else:
            status = "SELECTED"
        statuses.append(status)
    return {
        "statuses": statuses,
        "selected_indices": [
            index for index, status in enumerate(statuses) if status == "SELECTED"
        ],
        "dominated_indices": [
            index for index, status in enumerate(statuses) if status == "DOMINATED"
        ],
        "unresolved_indices": [
            index for index, status in enumerate(statuses) if status == "UNRESOLVED"
        ],
        "unsafe_indices": [
            index for index, status in enumerate(statuses) if status == "UNSAFE"
        ],
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _make_split(
    config: Mapping[str, Any],
    *,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    surface = config["surface"]
    count = int(surface[f"{split}_arrays_per_pressure"])
    seed = int(surface[f"{split}_seed"])
    context_prefix = str(surface[f"{split}_context_prefix"])
    tolerance = float(surface["metric_equivalence_tolerance"])
    games = tuple(str(value) for value in surface["games"])
    pair_rows: list[dict[str, Any]] = []
    array_rows: list[dict[str, Any]] = []
    for pressure_index, pressure_value in enumerate(surface["pressures"]):
        pressure = str(pressure_value)
        lineage = sha256_text(f"hard-surface-v2:{split}:{pressure}:{seed}")
        for array_offset in range(count):
            local_seed = seed + pressure_index * 1009 + array_offset * 37
            rng = random.Random(local_seed)
            game = games[(pressure_index + array_offset) % len(games)]
            jitter = rng.uniform(-0.012, 0.012)
            candidates = [
                _candidate(
                    slot=slot,
                    pressure=pressure,
                    game=game,
                    split=split,
                    jitter=jitter,
                    context_prefix=context_prefix,
                )
                for slot in range(int(surface["candidates_per_array"]))
            ]
            array_identity = f"{split}:{pressure}:{array_offset}:{local_seed}"
            array_id = f"array-{sha256_text(array_identity)[:20]}"
            pair_ids: list[str] = []
            for left_index in range(len(candidates)):
                for right_index in range(len(candidates)):
                    if left_index == right_index:
                        continue
                    pair_identity = f"{array_identity}:{left_index}:{right_index}"
                    example_id = f"hard-pair-{sha256_text(pair_identity)[:20]}"
                    pair_ids.append(example_id)
                    target = interval_taste_policy(
                        {
                            "left": candidates[left_index],
                            "right": candidates[right_index],
                        },
                        tolerance=tolerance,
                    )
                    pair_rows.append(
                        {
                            "schema_version": "curation-taste-uncertain-pair.v2",
                            "example_id": example_id,
                            "array_id": array_id,
                            "left_index": left_index,
                            "right_index": right_index,
                            "meta_split": split,
                            "lineage_group_sha256": lineage,
                            "source_kind": "environment_hard_surface",
                            "input": {
                                "left": candidates[left_index],
                                "right": candidates[right_index],
                            },
                            "target": {
                                "preference": target[0],
                                "rationale_code": target[1],
                            },
                            "authority": "ENVIRONMENT_OWNED",
                        }
                    )
            array_rows.append(
                {
                    "schema_version": "curation-taste-candidate-array.v2",
                    "array_id": array_id,
                    "meta_split": split,
                    "lineage_group_sha256": lineage,
                    "game": game,
                    "fixture_pressure": pressure,
                    "candidates": candidates,
                    "pair_example_ids": sorted(pair_ids),
                    "target": derive_array_target(candidates, tolerance=tolerance),
                    "authority": "ENVIRONMENT_OWNED",
                }
            )
    return (
        sorted(pair_rows, key=lambda row: str(row["example_id"])),
        sorted(array_rows, key=lambda row: str(row["array_id"])),
    )


def generate_hard_surface_dataset(config_path: Path, out_dir: Path) -> dict[str, Any]:
    config = load_structured(config_path)
    if config["status"] != "frozen_before_dataset_generation":
        raise ValueError("hard-surface preregistration is not frozen")
    out_dir.mkdir(parents=True, exist_ok=False)
    public_dir = out_dir / "public_dataset"
    steward_dir = out_dir / "private_steward"
    public_dir.mkdir()
    steward_dir.mkdir()
    train_pairs, train_arrays = _make_split(config, split="train")
    heldout_pairs, heldout_arrays = _make_split(config, split="heldout")
    contract = config["dataset_contract"]
    expected = {
        "train_pairs": int(contract["required_train_pairs"]),
        "heldout_pairs": int(contract["required_heldout_pairs"]),
        "train_arrays": int(contract["required_train_arrays"]),
        "heldout_arrays": int(contract["required_heldout_arrays"]),
    }
    observed = {
        "train_pairs": len(train_pairs),
        "heldout_pairs": len(heldout_pairs),
        "train_arrays": len(train_arrays),
        "heldout_arrays": len(heldout_arrays),
    }
    if observed != expected:
        raise ValueError(f"hard-surface counts differ from preregistration: {observed}")
    train_lineages = {str(row["lineage_group_sha256"]) for row in train_pairs}
    heldout_lineages = {str(row["lineage_group_sha256"]) for row in heldout_pairs}
    if train_lineages & heldout_lineages:
        raise ValueError("hard-surface train and heldout lineages overlap")
    train_contexts = {
        str(row["input"][side]["context_mode"])
        for row in train_pairs
        for side in ("left", "right")
    }
    heldout_contexts = {
        str(row["input"][side]["context_mode"])
        for row in heldout_pairs
        for side in ("left", "right")
    }
    if train_contexts & heldout_contexts:
        raise ValueError("heldout context categories must be novel")
    train_pair_path = public_dir / "train_pairs.jsonl"
    train_array_path = public_dir / "train_arrays.jsonl"
    heldout_pair_path = public_dir / "heldout_pairs.jsonl"
    heldout_array_path = public_dir / "heldout_arrays.jsonl"
    heldout_pair_labels_path = steward_dir / "heldout_pair_labels.jsonl"
    heldout_array_labels_path = steward_dir / "heldout_array_labels.jsonl"
    _write_jsonl(train_pair_path, train_pairs)
    _write_jsonl(train_array_path, train_arrays)
    _write_jsonl(
        heldout_pair_path,
        [
            {key: value for key, value in row.items() if key not in {"target", "authority"}}
            for row in heldout_pairs
        ],
    )
    _write_jsonl(
        heldout_array_path,
        [
            {key: value for key, value in row.items() if key not in {"target", "authority"}}
            for row in heldout_arrays
        ],
    )
    _write_jsonl(
        heldout_pair_labels_path,
        [
            {
                "schema_version": "curation-taste-uncertain-pair-label.v2",
                "example_id": row["example_id"],
                "array_id": row["array_id"],
                "lineage_group_sha256": row["lineage_group_sha256"],
                "target": row["target"],
                "authority": row["authority"],
            }
            for row in heldout_pairs
        ],
    )
    _write_jsonl(
        heldout_array_labels_path,
        [
            {
                "schema_version": "curation-taste-candidate-array-label.v2",
                "array_id": row["array_id"],
                "lineage_group_sha256": row["lineage_group_sha256"],
                "target": row["target"],
                "authority": row["authority"],
            }
            for row in heldout_arrays
        ],
    )
    preference_distribution = Counter(
        str(row["target"]["preference"]) for row in train_pairs
    )
    rationale_distribution = Counter(
        str(row["target"]["rationale_code"]) for row in train_pairs
    )
    status_distribution = Counter(
        str(status)
        for row in train_arrays
        for status in row["target"]["statuses"]
    )
    orientation_complete = all(
        {
            (int(row["left_index"]), int(row["right_index"]))
            for row in train_pairs
            if row["array_id"] == array_row["array_id"]
        }
        == {
            (left, right)
            for left in range(len(array_row["candidates"]))
            for right in range(len(array_row["candidates"]))
            if left != right
        }
        for array_row in train_arrays
    )
    checks = {
        "counts_match_preregistration": observed == expected,
        "cross_split_lineages_zero": not bool(train_lineages & heldout_lineages),
        "orientation_complete": orientation_complete,
        "all_pair_preferences_present": set(preference_distribution)
        == set(HARD_PREFERENCES),
        "all_rationales_present": set(rationale_distribution) == set(HARD_RATIONALES),
        "all_array_statuses_present": set(status_distribution) == set(ARRAY_STATUSES),
        "context_categories_disjoint": not bool(train_contexts & heldout_contexts),
        "heldout_pair_targets_separate": True,
        "heldout_array_targets_separate": True,
    }
    manifest = {
        "schema_version": "curation-taste-array-hard-surface.v2",
        "experiment_id": config["experiment_id"],
        "task": "uncertain_pairwise_and_setwise_curation_selection",
        "feature_whitelist": list(HARD_FEATURE_WHITELIST),
        "preference_labels": list(HARD_PREFERENCES),
        "rationale_labels": list(HARD_RATIONALES),
        "array_status_labels": list(ARRAY_STATUSES),
        "metric_equivalence_tolerance": config["surface"][
            "metric_equivalence_tolerance"
        ],
        "splits": {
            "train_pairs": {
                "path": train_pair_path.name,
                "count": len(train_pairs),
                "sha256": sha256_file(train_pair_path),
                "targets_in_file": True,
            },
            "train_arrays": {
                "path": train_array_path.name,
                "count": len(train_arrays),
                "sha256": sha256_file(train_array_path),
                "targets_in_file": True,
            },
            "heldout_pairs": {
                "path": heldout_pair_path.name,
                "count": len(heldout_pairs),
                "sha256": sha256_file(heldout_pair_path),
                "targets_in_file": False,
            },
            "heldout_arrays": {
                "path": heldout_array_path.name,
                "count": len(heldout_arrays),
                "sha256": sha256_file(heldout_array_path),
                "targets_in_file": False,
            },
        },
        "heldout_label_custody": {
            "pair_labels_sha256": sha256_file(heldout_pair_labels_path),
            "array_labels_sha256": sha256_file(heldout_array_labels_path),
            "paths_disclosed_to_candidate": False,
            "authority": "ENVIRONMENT_OWNED",
        },
        "lineage_audit": {
            "train_groups": len(train_lineages),
            "heldout_groups": len(heldout_lineages),
            "cross_split_groups": 0,
        },
        "category_shift_audit": {
            "train_context_count": len(train_contexts),
            "heldout_context_count": len(heldout_contexts),
            "overlap": 0,
        },
        "orientation_audit": {
            "contract": config["surface"]["orientation_contract"],
            "complete": orientation_complete,
        },
        "train_preference_distribution": dict(sorted(preference_distribution.items())),
        "train_rationale_distribution": dict(sorted(rationale_distribution.items())),
        "train_array_status_distribution": dict(sorted(status_distribution.items())),
        "checks": checks,
        "larger_model_training_authorized": False,
    }
    dump_structured(public_dir / "dataset_manifest.json", manifest)
    report = {
        "schema_version": "ldt-array-hard-surface-generation-report.v2",
        "experiment_id": config["experiment_id"],
        "preregistration_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file(public_dir / "dataset_manifest.json"),
        "counts": observed,
        "checks": checks,
        "generation_passed": all(checks.values()),
        "network_used": False,
        "raw_game_records_read": False,
        "heldout_targets_exported_to_candidate": False,
        "claim_boundary": config["claim_scope"],
    }
    dump_structured(out_dir / "generation_report.json", report)
    return report


def validate_hard_pair_row(row: Mapping[str, Any], *, targets_required: bool) -> None:
    expected = {
        "schema_version",
        "example_id",
        "array_id",
        "left_index",
        "right_index",
        "meta_split",
        "lineage_group_sha256",
        "source_kind",
        "input",
    }
    if targets_required:
        expected.update({"target", "authority"})
    if set(row) != expected:
        raise ValueError("hard pair has missing or extra fields")
    if row["schema_version"] != "curation-taste-uncertain-pair.v2":
        raise ValueError("unsupported hard pair schema")
    expected_split = "train" if targets_required else "heldout"
    if row["meta_split"] != expected_split:
        raise ValueError("hard pair split mismatch")
    input_ = row["input"]
    if set(input_) != {"left", "right"}:
        raise ValueError("hard pair input must contain left and right")
    validate_hard_side(input_["left"])
    validate_hard_side(input_["right"])
    if targets_required:
        if row["target"]["preference"] not in HARD_PREFERENCES:
            raise ValueError("unknown hard preference")
        if row["target"]["rationale_code"] not in HARD_RATIONALES:
            raise ValueError("unknown hard rationale")
        if row["authority"] != "ENVIRONMENT_OWNED":
            raise ValueError("hard pair target lacks environment authority")


def load_hard_surface_dataset(
    manifest_path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != "curation-taste-array-hard-surface.v2":
        raise ValueError("unsupported hard-surface manifest")
    if tuple(manifest["feature_whitelist"]) != HARD_FEATURE_WHITELIST:
        raise ValueError("hard-surface feature whitelist mismatch")
    if not all(bool(value) for value in manifest["checks"].values()):
        raise ValueError("hard-surface generation checks did not pass")
    loaded: dict[str, list[dict[str, Any]]] = {}
    for split_name, split in manifest["splits"].items():
        path = manifest_path.parent / str(split["path"])
        if sha256_file(path) != split["sha256"]:
            raise ValueError(f"hard-surface split hash mismatch: {split_name}")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(rows) != int(split["count"]):
            raise ValueError(f"hard-surface split count mismatch: {split_name}")
        loaded[split_name] = rows
    for row in loaded["train_pairs"]:
        validate_hard_pair_row(row, targets_required=True)
    for row in loaded["heldout_pairs"]:
        validate_hard_pair_row(row, targets_required=False)
    train_lineages = {
        str(row["lineage_group_sha256"]) for row in loaded["train_pairs"]
    }
    heldout_lineages = {
        str(row["lineage_group_sha256"]) for row in loaded["heldout_pairs"]
    }
    if train_lineages & heldout_lineages:
        raise ValueError("hard-surface split contamination")
    return (
        manifest,
        loaded["train_pairs"],
        loaded["heldout_pairs"],
        loaded["train_arrays"],
        loaded["heldout_arrays"],
    )


@dataclass(frozen=True)
class HardPairEncoder:
    games: tuple[str, ...]
    pressures: tuple[str, ...]
    context_modes: tuple[str, ...]
    optimization_pressures: tuple[str, ...]
    tolerance: float = 0.01

    @classmethod
    def fit(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        tolerance: float = 0.01,
    ) -> HardPairEncoder:
        sides = [row["input"][side] for row in rows for side in ("left", "right")]
        return cls(
            games=tuple(sorted({str(side["game"]) for side in sides})),
            pressures=tuple(
                sorted({str(side["fixture_pressure"]) for side in sides})
            ),
            context_modes=tuple(
                sorted({str(side["context_mode"]) for side in sides})
            ),
            optimization_pressures=tuple(
                sorted({str(side["optimization_pressure"]) for side in sides})
            ),
            tolerance=tolerance,
        )

    def feature_names(self, view: str) -> tuple[str, ...]:
        names = ["bias"]
        for field in METRIC_FIELDS:
            names.extend(
                (
                    f"{field}:center_delta",
                    f"{field}:absolute_center_delta",
                    f"{field}:left_width",
                    f"{field}:right_width",
                    f"{field}:left_missing",
                    f"{field}:right_missing",
                )
            )
        names.extend(
            (
                "safety:delta",
                "safety:mismatch",
                "parameter_bucket:delta",
                "parameter_bucket:absolute_delta",
            )
        )
        names.extend(f"game={value}" for value in self.games)
        names.extend(f"fixture_pressure={value}" for value in self.pressures)
        names.extend(
            f"context_mode={value}:signed" for value in self.context_modes
        )
        names.extend(
            f"optimization_pressure={value}:signed"
            for value in self.optimization_pressures
        )
        if view == "typed":
            names.extend(
                (
                    "lattice:safety_direction",
                    "lattice:left_robustly_dominates",
                    "lattice:right_robustly_dominates",
                    "lattice:metrics_equivalent",
                    "lattice:efficiency_direction",
                    "lattice:evidence_missing",
                    "lattice:interval_present",
                    "lattice:insufficient_evidence",
                    "lattice:irreducible_tradeoff",
                )
            )
        elif view != "raw":
            raise ValueError(f"unknown hard feature view: {view}")
        return tuple(names)

    @staticmethod
    def _summary(value: Mapping[str, Any]) -> tuple[float, float, float]:
        if value["lower"] is None or value["upper"] is None:
            return 0.0, 1.0, 1.0
        lower = float(value["lower"])
        upper = float(value["upper"])
        return (lower + upper) / 2.0, upper - lower, 0.0

    def transform(
        self,
        input_: Mapping[str, Mapping[str, Any]],
        *,
        view: str,
    ) -> array:
        left = input_["left"]
        right = input_["right"]
        values = [1.0]
        for field in METRIC_FIELDS:
            left_center, left_width, left_missing = self._summary(left[field])
            right_center, right_width, right_missing = self._summary(right[field])
            delta = left_center - right_center
            values.extend(
                (
                    delta,
                    abs(delta),
                    left_width,
                    right_width,
                    left_missing,
                    right_missing,
                )
            )
        safety_delta = float(bool(left["safety_qualified"])) - float(
            bool(right["safety_qualified"])
        )
        parameter_delta = (
            BUCKET_RANK[str(left["trainable_parameter_bucket"])]
            - BUCKET_RANK[str(right["trainable_parameter_bucket"])]
        )
        values.extend(
            (
                safety_delta,
                float(safety_delta != 0.0),
                parameter_delta / 3.0,
                abs(parameter_delta) / 3.0,
            )
        )
        values.extend(float(left["game"] == value) for value in self.games)
        values.extend(
            float(left["fixture_pressure"] == value) for value in self.pressures
        )
        values.extend(
            float(left["context_mode"] == value)
            - float(right["context_mode"] == value)
            for value in self.context_modes
        )
        values.extend(
            float(left["optimization_pressure"] == value)
            - float(right["optimization_pressure"] == value)
            for value in self.optimization_pressures
        )
        if view == "typed":
            preference, rationale = interval_taste_policy(
                input_, tolerance=self.tolerance
            )
            values.extend(
                (
                    safety_delta,
                    float(
                        preference == "LEFT"
                        and rationale == "pareto_dominance"
                    ),
                    float(
                        preference == "RIGHT"
                        and rationale == "pareto_dominance"
                    ),
                    float(rationale == "parameter_efficiency"),
                    (
                        1.0
                        if preference == "LEFT"
                        and rationale == "parameter_efficiency"
                        else -1.0
                        if preference == "RIGHT"
                        and rationale == "parameter_efficiency"
                        else 0.0
                    ),
                    float(
                        _known_bounds(left) is None
                        or _known_bounds(right) is None
                    ),
                    float(
                        any(
                            bound[0] != bound[1]
                            for bound in (
                                *(_known_bounds(left) or ()),
                                *(_known_bounds(right) or ()),
                            )
                        )
                    ),
                    float(rationale == "insufficient_evidence"),
                    float(rationale == "irreducible_tradeoff"),
                )
            )
        elif view != "raw":
            raise ValueError(f"unknown hard feature view: {view}")
        result = array("d", values)
        if len(result) != len(self.feature_names(view)):
            raise AssertionError("hard feature names and values differ")
        return result

    def receipt(self) -> dict[str, Any]:
        return {
            "games": list(self.games),
            "pressures": list(self.pressures),
            "context_modes": list(self.context_modes),
            "optimization_pressures": list(self.optimization_pressures),
            "tolerance": self.tolerance,
            "fit_split": "train_only",
        }


def hard_input_sha256(row: Mapping[str, Any]) -> str:
    return sha256_json(row["input"])
