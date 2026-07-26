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

from datagoblin.experiments.ldt_array_hard_surface import (
    ARRAY_STATUSES,
    HARD_INVERT_PREFERENCE,
    HARD_PREFERENCES,
    HARD_RATIONALES,
)
from datagoblin.experiments.ldt_array_sorter import (
    BUCKET_RANK,
    METRIC_FIELDS,
    sha256_file,
    sha256_json,
)
from datagoblin.hashing import sha256_text
from datagoblin.serialization import dump_structured, load_structured

CONFIRMATION_FEATURE_WHITELIST = (
    "game",
    "fixture_pressure",
    "context_mode",
    "optimization_pressure",
    "trainable_parameter_bucket",
    "safety_qualified",
    *METRIC_FIELDS,
)


def _interval(
    lower: float | None,
    upper: float | None,
) -> dict[str, float | None]:
    return {
        "lower": None if lower is None else max(0.0, min(1.0, lower)),
        "upper": None if upper is None else max(0.0, min(1.0, upper)),
    }


def _point(value: float) -> dict[str, float | None]:
    bounded = max(0.0, min(1.0, value))
    return _interval(bounded, bounded)


def _validate_interval(value: Mapping[str, Any]) -> None:
    if set(value) != {"lower", "upper"}:
        raise ValueError("confirmation metric must contain exactly lower and upper")
    lower = value["lower"]
    upper = value["upper"]
    for bound in (lower, upper):
        if bound is not None and (
            not math.isfinite(float(bound)) or not 0.0 <= float(bound) <= 1.0
        ):
            raise ValueError("confirmation interval bound is invalid")
    if lower is not None and upper is not None and float(lower) > float(upper):
        raise ValueError("confirmation lower bound exceeds upper bound")


def validate_confirmation_side(side: Mapping[str, Any]) -> None:
    if set(side) != set(CONFIRMATION_FEATURE_WHITELIST):
        raise ValueError("confirmation side differs from the feature contract")
    if not isinstance(side["safety_qualified"], bool):
        raise ValueError("confirmation safety_qualified must be boolean")
    if str(side["trainable_parameter_bucket"]) not in BUCKET_RANK:
        raise ValueError("confirmation parameter bucket is unknown")
    for field in METRIC_FIELDS:
        interval = side[field]
        if not isinstance(interval, Mapping):
            raise ValueError(f"{field} must be an interval")
        _validate_interval(interval)


def conservative_bounds(
    side: Mapping[str, Any],
) -> tuple[tuple[float, float], ...]:
    result = []
    for field in METRIC_FIELDS:
        value = side[field]
        lower = 0.0 if value["lower"] is None else float(value["lower"])
        upper = 1.0 if value["upper"] is None else float(value["upper"])
        result.append((lower, upper))
    return tuple(result)


def evidence_is_complete(side: Mapping[str, Any]) -> bool:
    return all(
        side[field]["lower"] is not None and side[field]["upper"] is not None
        for field in METRIC_FIELDS
    )


def evidence_has_interval_width(side: Mapping[str, Any]) -> bool:
    return any(lower != upper for lower, upper in conservative_bounds(side))


def evidence_has_partial_bound(side: Mapping[str, Any]) -> bool:
    return any(
        (side[field]["lower"] is None) != (side[field]["upper"] is None)
        for field in METRIC_FIELDS
    )


def partial_interval_taste_policy(
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
    left_bounds = conservative_bounds(left)
    right_bounds = conservative_bounds(right)
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
    complete = evidence_is_complete(left) and evidence_is_complete(right)
    equivalent = complete and all(
        max(abs(left_lower - right_upper), abs(left_upper - right_lower))
        <= tolerance
        for (left_lower, left_upper), (right_lower, right_upper) in zip(
            left_bounds, right_bounds, strict=True
        )
    )
    left_cost = BUCKET_RANK[str(left["trainable_parameter_bucket"])]
    right_cost = BUCKET_RANK[str(right["trainable_parameter_bucket"])]
    if equivalent and left_cost != right_cost:
        return ("LEFT" if left_cost < right_cost else "RIGHT"), (
            "parameter_efficiency"
        )
    if not complete or evidence_has_interval_width(left) or evidence_has_interval_width(
        right
    ):
        return "ABSTAIN", "insufficient_evidence"
    return "TRADEOFF", "irreducible_tradeoff"


def swap_confirmation_pair(row: Mapping[str, Any]) -> dict[str, Any]:
    swapped = copy.deepcopy(dict(row))
    swapped["input"] = {
        "left": copy.deepcopy(row["input"]["right"]),
        "right": copy.deepcopy(row["input"]["left"]),
    }
    for left_key, right_key in (
        ("left_index", "right_index"),
        ("left_tournament_report_sha256", "right_tournament_report_sha256"),
    ):
        if left_key in swapped:
            swapped[left_key], swapped[right_key] = (
                swapped[right_key],
                swapped[left_key],
            )
    if "target" in swapped:
        preference = str(row["target"]["preference"])
        swapped["target"]["preference"] = HARD_INVERT_PREFERENCE[preference]
    return swapped


def _candidate(
    *,
    slot: int,
    pressure: str,
    game: str,
    jitter: float,
    context_prefix: str,
) -> dict[str, Any]:
    base = tuple(
        max(0.0, min(1.0, value + jitter))
        for value in (0.74, 0.76, 0.73, 0.75)
    )
    dominant = tuple(min(0.97, value + 0.09) for value in base)
    if slot == 0:
        metrics = tuple(_point(value) for value in base)
        mode = "baseline"
        optimization = "balanced"
        bucket = "medium"
        safe = True
    elif slot == 1:
        metrics = tuple(_point(value) for value in dominant)
        mode = "dominant"
        optimization = "quality"
        bucket = "large"
        safe = True
    elif slot == 2:
        metrics = tuple(_point(0.99) for _ in METRIC_FIELDS)
        mode = "unsafe_high_coverage"
        optimization = "coverage_without_safety"
        bucket = "zero"
        safe = False
    elif pressure == "one_sided_bounds" and slot == 3:
        metrics = tuple(_interval(value + 0.03, None) for value in dominant)
        mode = "lower_bound_only"
        optimization = "conservative_dominance"
        bucket = "small"
        safe = True
    elif pressure == "one_sided_bounds" and slot == 4:
        metrics = tuple(_interval(None, value + 0.025) for value in dominant)
        mode = "upper_bound_only"
        optimization = "partial_evidence"
        bucket = "small"
        safe = True
    elif pressure == "one_sided_bounds" and slot == 5:
        vector = (0.94 + jitter, 0.93 + jitter, 0.65 + jitter, 0.67 + jitter)
        metrics = tuple(_point(value) for value in vector)
        mode = "ordinary_specialist"
        optimization = "ordinary_quality"
        bucket = "small"
        safe = True
    elif pressure == "asymmetric_intervals" and slot == 3:
        metrics = tuple(
            _interval(value - 0.004 * (index + 1), value + 0.018 + 0.004 * index)
            for index, value in enumerate(dominant)
        )
        mode = "asymmetric_interval"
        optimization = "bounded_uncertainty"
        bucket = "medium"
        safe = True
    elif pressure == "asymmetric_intervals" and slot == 4:
        vector = (0.94 + jitter, 0.93 + jitter, 0.65 + jitter, 0.67 + jitter)
        metrics = tuple(_point(value) for value in vector)
        mode = "ordinary_specialist"
        optimization = "ordinary_quality"
        bucket = "small"
        safe = True
    elif pressure == "asymmetric_intervals" and slot == 5:
        provisional = [_point(value + 0.02) for value in dominant]
        missing_index = (len(game) + len(pressure)) % len(METRIC_FIELDS)
        provisional[missing_index] = _interval(None, None)
        metrics = tuple(provisional)
        mode = "missing_axis"
        optimization = "incomplete_evidence"
        bucket = "small"
        safe = True
    elif pressure == "tolerance_calibration" and slot == 3:
        metrics = tuple(_point(value + 0.004) for value in dominant)
        mode = "equivalent_compact"
        optimization = "parameter_efficiency"
        bucket = "small"
        safe = True
    elif pressure == "tolerance_calibration" and slot == 4:
        metrics = tuple(_point(value + 0.014) for value in dominant)
        mode = "just_dominant"
        optimization = "tolerance_edge"
        bucket = "large"
        safe = True
    elif pressure == "tolerance_calibration" and slot == 5:
        metrics = tuple(
            _interval(value - 0.006, value + 0.012 + 0.002 * index)
            for index, value in enumerate(dominant)
        )
        mode = "calibration_interval"
        optimization = "calibration_boundary"
        bucket = "medium"
        safe = True
    elif pressure == "cross_game_pareto" and slot == 3:
        vector = (0.94 + jitter, 0.93 + jitter, 0.65 + jitter, 0.67 + jitter)
        metrics = tuple(_point(value) for value in vector)
        mode = "ordinary_specialist"
        optimization = "ordinary_quality"
        bucket = "small"
        safe = True
    elif pressure == "cross_game_pareto" and slot == 4:
        vector = (0.64 + jitter, 0.66 + jitter, 0.95 + jitter, 0.94 + jitter)
        metrics = tuple(_point(value) for value in vector)
        mode = "ood_specialist"
        optimization = "ood_robustness"
        bucket = "zero"
        safe = True
    elif pressure == "cross_game_pareto" and slot == 5:
        vector = (0.88 + jitter, 0.87 + jitter, 0.89 + jitter, 0.88 + jitter)
        metrics = tuple(
            _interval(value - 0.01, None if index % 2 else value + 0.02)
            for index, value in enumerate(vector)
        )
        mode = "partial_balanced"
        optimization = "partial_evidence"
        bucket = "medium"
        safe = True
    else:
        raise ValueError(
            f"unsupported confirmation candidate: {pressure} slot {slot}"
        )
    result: dict[str, Any] = {
        "game": game,
        "fixture_pressure": pressure,
        "context_mode": f"{context_prefix}:{mode}",
        "optimization_pressure": optimization,
        "trainable_parameter_bucket": bucket,
        "safety_qualified": safe,
    }
    result.update(
        {field: value for field, value in zip(METRIC_FIELDS, metrics, strict=True)}
    )
    validate_confirmation_side(result)
    return result


def reduce_confirmation_array(
    candidates: Sequence[Mapping[str, Any]],
    pair_preferences: Mapping[tuple[int, int], str],
) -> dict[str, Any]:
    incoming: set[int] = set()
    uncertain: set[int] = set()
    safe = [
        index for index, candidate in enumerate(candidates) if candidate["safety_qualified"]
    ]
    for offset, left_index in enumerate(safe):
        for right_index in safe[offset + 1 :]:
            forward = str(pair_preferences[(left_index, right_index)])
            reverse = str(pair_preferences[(right_index, left_index)])
            if forward != HARD_INVERT_PREFERENCE[reverse]:
                uncertain.update((left_index, right_index))
            elif forward == "LEFT":
                incoming.add(right_index)
            elif forward == "RIGHT":
                incoming.add(left_index)
            elif forward == "ABSTAIN":
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


def derive_confirmation_array_target(
    candidates: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    pair_preferences = {
        (left, right): partial_interval_taste_policy(
            {"left": candidates[left], "right": candidates[right]},
            tolerance=tolerance,
        )[0]
        for left in range(len(candidates))
        for right in range(len(candidates))
        if left != right
    }
    return reduce_confirmation_array(candidates, pair_preferences)


def _report(
    *,
    split: str,
    array_id: str,
    slot: int,
    lineage_group_sha256: str,
    features: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": "content-free-tournament-summary.v3",
        "report_id": f"report-{sha256_text(f'{array_id}:{slot}')[:20]}",
        "array_id": array_id,
        "candidate_slot": slot,
        "meta_split": split,
        "lineage_group_sha256": lineage_group_sha256,
        "feature_summary": copy.deepcopy(dict(features)),
        "contains_row_level_game_records": False,
        "contains_hidden_reasoning": False,
        "authority": "ENVIRONMENT_OWNED_SUMMARY",
    }
    return {
        **payload,
        "tournament_report_sha256": sha256_json(payload),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _make_split(
    config: Mapping[str, Any],
    *,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    surface = config["surface"]
    sizes = [int(value) for value in surface[f"{split}_array_sizes"]]
    seed = int(surface[f"{split}_seed"])
    context_prefix = str(surface[f"{split}_context_prefix"])
    tolerance = float(surface["metric_equivalence_tolerance"])
    games = tuple(str(value) for value in surface["games"])
    pairs: list[dict[str, Any]] = []
    arrays: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for pressure_index, pressure_value in enumerate(surface["pressures"]):
        pressure = str(pressure_value)
        lineage = sha256_text(f"confirmation-v3:{split}:{pressure}:{seed}")
        for array_offset, size in enumerate(sizes):
            local_seed = seed + pressure_index * 1709 + array_offset * 61
            rng = random.Random(local_seed)
            game = games[(pressure_index + array_offset) % len(games)]
            jitter = rng.uniform(-0.009, 0.009)
            array_identity = f"{split}:{pressure}:{array_offset}:{size}:{local_seed}"
            array_id = f"confirmation-array-{sha256_text(array_identity)[:20]}"
            features = [
                _candidate(
                    slot=slot,
                    pressure=pressure,
                    game=game,
                    jitter=jitter,
                    context_prefix=context_prefix,
                )
                for slot in range(size)
            ]
            array_reports = [
                _report(
                    split=split,
                    array_id=array_id,
                    slot=slot,
                    lineage_group_sha256=lineage,
                    features=candidate,
                )
                for slot, candidate in enumerate(features)
            ]
            reports.extend(array_reports)
            report_hashes = [
                str(report["tournament_report_sha256"])
                for report in array_reports
            ]
            pair_ids = []
            for left_index in range(size):
                for right_index in range(size):
                    if left_index == right_index:
                        continue
                    identity = f"{array_identity}:{left_index}:{right_index}"
                    example_id = f"confirmation-pair-{sha256_text(identity)[:20]}"
                    pair_ids.append(example_id)
                    target = partial_interval_taste_policy(
                        {
                            "left": features[left_index],
                            "right": features[right_index],
                        },
                        tolerance=tolerance,
                    )
                    pairs.append(
                        {
                            "schema_version": "curation-taste-confirmation-pair.v3",
                            "example_id": example_id,
                            "array_id": array_id,
                            "left_index": left_index,
                            "right_index": right_index,
                            "left_tournament_report_sha256": report_hashes[left_index],
                            "right_tournament_report_sha256": report_hashes[right_index],
                            "meta_split": split,
                            "lineage_group_sha256": lineage,
                            "source_kind": "content_free_tournament_summary",
                            "input": {
                                "left": features[left_index],
                                "right": features[right_index],
                            },
                            "target": {
                                "preference": target[0],
                                "rationale_code": target[1],
                            },
                            "authority": "ENVIRONMENT_OWNED",
                        }
                    )
            arrays.append(
                {
                    "schema_version": "curation-taste-confirmation-array.v3",
                    "array_id": array_id,
                    "meta_split": split,
                    "lineage_group_sha256": lineage,
                    "game": game,
                    "fixture_pressure": pressure,
                    "candidate_count": size,
                    "candidates": [
                        {
                            "tournament_report_sha256": report_hash,
                            "features": candidate,
                        }
                        for report_hash, candidate in zip(
                            report_hashes, features, strict=True
                        )
                    ],
                    "pair_example_ids": sorted(pair_ids),
                    "target": derive_confirmation_array_target(
                        features, tolerance=tolerance
                    ),
                    "authority": "ENVIRONMENT_OWNED",
                }
            )
    return (
        sorted(pairs, key=lambda row: str(row["example_id"])),
        sorted(arrays, key=lambda row: str(row["array_id"])),
        sorted(reports, key=lambda row: str(row["report_id"])),
    )


def _public_without_target(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in row.items() if key not in {"target", "authority"}
    }


def generate_confirmation_dataset(config_path: Path, out_dir: Path) -> dict[str, Any]:
    config = load_structured(config_path)
    if config["status"] != "frozen_before_dataset_generation":
        raise ValueError("confirmation preregistration is not frozen")
    out_dir.mkdir(parents=True, exist_ok=False)
    public_dir = out_dir / "public_dataset"
    steward_dir = out_dir / "private_steward"
    public_dir.mkdir()
    steward_dir.mkdir()
    train_pairs, train_arrays, train_reports = _make_split(config, split="train")
    heldout_pairs, heldout_arrays, heldout_reports = _make_split(
        config, split="heldout"
    )
    contract = config["dataset_contract"]
    observed = {
        "train_pairs": len(train_pairs),
        "heldout_pairs": len(heldout_pairs),
        "train_arrays": len(train_arrays),
        "heldout_arrays": len(heldout_arrays),
        "train_reports": len(train_reports),
        "heldout_reports": len(heldout_reports),
    }
    expected = {
        name: int(contract[f"required_{name}"])
        for name in observed
    }
    if observed != expected:
        raise ValueError(
            f"confirmation counts differ from preregistration: {observed}"
        )
    train_lineages = {
        str(row["lineage_group_sha256"]) for row in train_arrays
    }
    heldout_lineages = {
        str(row["lineage_group_sha256"]) for row in heldout_arrays
    }
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
    if train_lineages & heldout_lineages or train_contexts & heldout_contexts:
        raise ValueError("confirmation split isolation failed")
    paths = {
        "train_pairs": public_dir / "train_pairs.jsonl",
        "train_arrays": public_dir / "train_arrays.jsonl",
        "train_reports": public_dir / "train_tournament_reports.jsonl",
        "heldout_pairs": public_dir / "heldout_pairs.jsonl",
        "heldout_arrays": public_dir / "heldout_arrays.jsonl",
        "heldout_reports": public_dir / "heldout_tournament_reports.jsonl",
    }
    _write_jsonl(paths["train_pairs"], train_pairs)
    _write_jsonl(paths["train_arrays"], train_arrays)
    _write_jsonl(paths["train_reports"], train_reports)
    _write_jsonl(
        paths["heldout_pairs"],
        [_public_without_target(row) for row in heldout_pairs],
    )
    _write_jsonl(
        paths["heldout_arrays"],
        [_public_without_target(row) for row in heldout_arrays],
    )
    _write_jsonl(paths["heldout_reports"], heldout_reports)
    pair_labels_path = steward_dir / "heldout_pair_labels.jsonl"
    array_labels_path = steward_dir / "heldout_array_labels.jsonl"
    _write_jsonl(
        pair_labels_path,
        [
            {
                "schema_version": "curation-taste-confirmation-pair-label.v3",
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
        array_labels_path,
        [
            {
                "schema_version": "curation-taste-confirmation-array-label.v3",
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
    array_sizes = {
        int(row["candidate_count"]) for row in train_arrays + heldout_arrays
    }
    report_hashes = {
        str(row["tournament_report_sha256"])
        for row in train_reports + heldout_reports
    }
    bound_hashes = {
        str(row[key])
        for row in train_pairs + heldout_pairs
        for key in (
            "left_tournament_report_sha256",
            "right_tournament_report_sha256",
        )
    }
    orientation_complete = all(
        {
            (int(pair["left_index"]), int(pair["right_index"]))
            for pair in train_pairs + heldout_pairs
            if pair["array_id"] == array_row["array_id"]
        }
        == {
            (left, right)
            for left in range(int(array_row["candidate_count"]))
            for right in range(int(array_row["candidate_count"]))
            if left != right
        }
        for array_row in train_arrays + heldout_arrays
    )
    checks = {
        "counts_match_preregistration": observed == expected,
        "cross_split_lineages_zero": not bool(train_lineages & heldout_lineages),
        "orientation_complete": orientation_complete,
        "required_array_sizes_present": array_sizes
        == {int(value) for value in contract["require_array_sizes"]},
        "all_pair_preferences_present": set(preference_distribution)
        == set(HARD_PREFERENCES),
        "all_rationales_present": set(rationale_distribution)
        == set(HARD_RATIONALES),
        "all_array_statuses_present": set(status_distribution)
        == set(ARRAY_STATUSES),
        "context_categories_disjoint": not bool(train_contexts & heldout_contexts),
        "report_hash_bindings_complete": bound_hashes == report_hashes,
        "heldout_pair_targets_separate": True,
        "heldout_array_targets_separate": True,
    }
    manifest_splits = {}
    for name, path in paths.items():
        manifest_splits[name] = {
            "path": path.name,
            "count": observed[name],
            "sha256": sha256_file(path),
            "targets_in_file": name in {"train_pairs", "train_arrays"},
        }
    manifest = {
        "schema_version": "curation-taste-array-confirmation.v3",
        "experiment_id": config["experiment_id"],
        "task": "variable_array_partial_interval_tournament_summary_selection",
        "feature_whitelist": list(CONFIRMATION_FEATURE_WHITELIST),
        "preference_labels": list(HARD_PREFERENCES),
        "rationale_labels": list(HARD_RATIONALES),
        "array_status_labels": list(ARRAY_STATUSES),
        "metric_equivalence_tolerance": config["surface"][
            "metric_equivalence_tolerance"
        ],
        "splits": manifest_splits,
        "heldout_label_custody": {
            "pair_labels_sha256": sha256_file(pair_labels_path),
            "array_labels_sha256": sha256_file(array_labels_path),
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
        "array_size_audit": {
            "sizes": sorted(array_sizes),
            "train_distribution": dict(
                sorted(
                    Counter(
                        int(row["candidate_count"]) for row in train_arrays
                    ).items()
                )
            ),
            "heldout_distribution": dict(
                sorted(
                    Counter(
                        int(row["candidate_count"]) for row in heldout_arrays
                    ).items()
                )
            ),
        },
        "report_binding_audit": {
            "report_count": len(report_hashes),
            "bound_report_count": len(bound_hashes),
            "complete": bound_hashes == report_hashes,
        },
        "train_preference_distribution": dict(sorted(preference_distribution.items())),
        "train_rationale_distribution": dict(sorted(rationale_distribution.items())),
        "train_array_status_distribution": dict(sorted(status_distribution.items())),
        "checks": checks,
        "larger_model_training_authorized": False,
    }
    manifest_path = public_dir / "dataset_manifest.json"
    dump_structured(manifest_path, manifest)
    report = {
        "schema_version": "ldt-array-confirmation-generation-report.v3",
        "experiment_id": config["experiment_id"],
        "preregistration_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def validate_confirmation_pair(
    row: Mapping[str, Any],
    *,
    targets_required: bool,
) -> None:
    expected = {
        "schema_version",
        "example_id",
        "array_id",
        "left_index",
        "right_index",
        "left_tournament_report_sha256",
        "right_tournament_report_sha256",
        "meta_split",
        "lineage_group_sha256",
        "source_kind",
        "input",
    }
    if targets_required:
        expected.update({"target", "authority"})
    if set(row) != expected:
        raise ValueError("confirmation pair has missing or extra fields")
    if row["schema_version"] != "curation-taste-confirmation-pair.v3":
        raise ValueError("unsupported confirmation pair schema")
    if row["meta_split"] != ("train" if targets_required else "heldout"):
        raise ValueError("confirmation pair split mismatch")
    if set(row["input"]) != {"left", "right"}:
        raise ValueError("confirmation pair input must contain left and right")
    validate_confirmation_side(row["input"]["left"])
    validate_confirmation_side(row["input"]["right"])
    if targets_required:
        if row["target"]["preference"] not in HARD_PREFERENCES:
            raise ValueError("unknown confirmation preference")
        if row["target"]["rationale_code"] not in HARD_RATIONALES:
            raise ValueError("unknown confirmation rationale")
        if row["authority"] != "ENVIRONMENT_OWNED":
            raise ValueError("confirmation target lacks environment authority")


def load_confirmation_dataset(
    manifest_path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != "curation-taste-array-confirmation.v3":
        raise ValueError("unsupported confirmation manifest")
    if tuple(manifest["feature_whitelist"]) != CONFIRMATION_FEATURE_WHITELIST:
        raise ValueError("confirmation feature whitelist mismatch")
    if not all(bool(value) for value in manifest["checks"].values()):
        raise ValueError("confirmation generation checks did not pass")
    loaded = {}
    for split_name, split in manifest["splits"].items():
        path = manifest_path.parent / str(split["path"])
        if sha256_file(path) != split["sha256"]:
            raise ValueError(f"confirmation split hash mismatch: {split_name}")
        rows = _read_jsonl(path)
        if len(rows) != int(split["count"]):
            raise ValueError(f"confirmation split count mismatch: {split_name}")
        loaded[split_name] = rows
    for row in loaded["train_pairs"]:
        validate_confirmation_pair(row, targets_required=True)
    for row in loaded["heldout_pairs"]:
        validate_confirmation_pair(row, targets_required=False)
    train_lineages = {
        str(row["lineage_group_sha256"]) for row in loaded["train_pairs"]
    }
    heldout_lineages = {
        str(row["lineage_group_sha256"]) for row in loaded["heldout_pairs"]
    }
    if train_lineages & heldout_lineages:
        raise ValueError("confirmation split contamination")
    reports = loaded["train_reports"] + loaded["heldout_reports"]
    report_map = {}
    for report in reports:
        report_hash = str(report["tournament_report_sha256"])
        payload = {
            key: value
            for key, value in report.items()
            if key != "tournament_report_sha256"
        }
        if report_hash != sha256_json(payload) or report_hash in report_map:
            raise ValueError("confirmation report hash binding is invalid")
        report_map[report_hash] = report
    pair_hashes = {
        str(row[key])
        for row in loaded["train_pairs"] + loaded["heldout_pairs"]
        for key in (
            "left_tournament_report_sha256",
            "right_tournament_report_sha256",
        )
    }
    if pair_hashes != set(report_map):
        raise ValueError("confirmation pair/report coverage differs")
    return (
        manifest,
        loaded["train_pairs"],
        loaded["heldout_pairs"],
        loaded["train_arrays"],
        loaded["heldout_arrays"],
        loaded["train_reports"],
        loaded["heldout_reports"],
    )


@dataclass(frozen=True)
class ConfirmationPairEncoder:
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
    ) -> ConfirmationPairEncoder:
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
                    f"{field}:lower_delta",
                    f"{field}:upper_delta",
                    f"{field}:left_width",
                    f"{field}:right_width",
                    f"{field}:left_lower_missing",
                    f"{field}:left_upper_missing",
                    f"{field}:right_lower_missing",
                    f"{field}:right_upper_missing",
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
                    "lattice:evidence_incomplete",
                    "lattice:partial_bound_present",
                    "lattice:interval_width_present",
                    "lattice:insufficient_evidence",
                    "lattice:irreducible_tradeoff",
                )
            )
        elif view != "raw":
            raise ValueError(f"unknown confirmation feature view: {view}")
        return tuple(names)

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
            left_value = left[field]
            right_value = right[field]
            left_lower = (
                0.0
                if left_value["lower"] is None
                else float(left_value["lower"])
            )
            left_upper = (
                1.0
                if left_value["upper"] is None
                else float(left_value["upper"])
            )
            right_lower = (
                0.0
                if right_value["lower"] is None
                else float(right_value["lower"])
            )
            right_upper = (
                1.0
                if right_value["upper"] is None
                else float(right_value["upper"])
            )
            values.extend(
                (
                    left_lower - right_lower,
                    left_upper - right_upper,
                    left_upper - left_lower,
                    right_upper - right_lower,
                    float(left_value["lower"] is None),
                    float(left_value["upper"] is None),
                    float(right_value["lower"] is None),
                    float(right_value["upper"] is None),
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
            preference, rationale = partial_interval_taste_policy(
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
                        not evidence_is_complete(left)
                        or not evidence_is_complete(right)
                    ),
                    float(
                        evidence_has_partial_bound(left)
                        or evidence_has_partial_bound(right)
                    ),
                    float(
                        evidence_has_interval_width(left)
                        or evidence_has_interval_width(right)
                    ),
                    float(rationale == "insufficient_evidence"),
                    float(rationale == "irreducible_tradeoff"),
                )
            )
        elif view != "raw":
            raise ValueError(f"unknown confirmation feature view: {view}")
        result = array("d", values)
        if len(result) != len(self.feature_names(view)):
            raise AssertionError("confirmation feature names and values differ")
        return result

    def receipt(self) -> dict[str, Any]:
        return {
            "games": list(self.games),
            "pressures": list(self.pressures),
            "context_modes": list(self.context_modes),
            "optimization_pressures": list(self.optimization_pressures),
            "tolerance": self.tolerance,
            "fit_split": "train_only",
            "partial_interval_completion": {"lower": 0.0, "upper": 1.0},
        }


def confirmation_input_sha256(row: Mapping[str, Any]) -> str:
    return sha256_json(row["input"])
