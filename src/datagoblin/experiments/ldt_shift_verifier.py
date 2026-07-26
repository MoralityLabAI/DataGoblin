from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_hard_surface import (
    ARRAY_STATUSES,
    HARD_PREFERENCES,
    HARD_RATIONALES,
)
from datagoblin.experiments.ldt_array_sorter import (
    BUCKET_RANK,
    METRIC_FIELDS,
    sha256_file,
)
from datagoblin.experiments.ldt_shift_builder import load_shift_fixture
from datagoblin.serialization import load_structured


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )


def _conservative_bounds(
    side: Mapping[str, Any],
) -> tuple[tuple[float, float], ...]:
    bounds = []
    for field in METRIC_FIELDS:
        value = side[field]
        lower = 0.0 if value["lower"] is None else float(value["lower"])
        upper = 1.0 if value["upper"] is None else float(value["upper"])
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower > upper
        ):
            raise ValueError("verifier received an invalid interval")
        bounds.append((lower, upper))
    return tuple(bounds)


def _complete(side: Mapping[str, Any]) -> bool:
    return all(
        side[field]["lower"] is not None
        and side[field]["upper"] is not None
        for field in METRIC_FIELDS
    )


def _has_interval_width(side: Mapping[str, Any]) -> bool:
    return any(
        lower != upper for lower, upper in _conservative_bounds(side)
    )


def shift_verifier_target(
    input_: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float,
) -> tuple[str, str]:
    left = input_["left"]
    right = input_["right"]
    left_safe = bool(left["safety_qualified"])
    right_safe = bool(right["safety_qualified"])
    if left_safe != right_safe:
        return ("LEFT" if left_safe else "RIGHT"), "safety_gate"

    left_bounds = _conservative_bounds(left)
    right_bounds = _conservative_bounds(right)
    left_dominates = all(
        left_lower >= right_upper - tolerance
        for (left_lower, _), (_, right_upper) in zip(
            left_bounds,
            right_bounds,
            strict=True,
        )
    ) and any(
        left_lower > right_upper + tolerance
        for (left_lower, _), (_, right_upper) in zip(
            left_bounds,
            right_bounds,
            strict=True,
        )
    )
    right_dominates = all(
        right_lower >= left_upper - tolerance
        for (_, left_upper), (right_lower, _) in zip(
            left_bounds,
            right_bounds,
            strict=True,
        )
    ) and any(
        right_lower > left_upper + tolerance
        for (_, left_upper), (right_lower, _) in zip(
            left_bounds,
            right_bounds,
            strict=True,
        )
    )
    if left_dominates != right_dominates:
        return ("LEFT" if left_dominates else "RIGHT"), "pareto_dominance"

    complete = _complete(left) and _complete(right)
    equivalent = complete and all(
        max(
            abs(left_lower - right_upper),
            abs(left_upper - right_lower),
        )
        <= tolerance
        for (left_lower, left_upper), (
            right_lower,
            right_upper,
        ) in zip(left_bounds, right_bounds, strict=True)
    )
    left_cost = BUCKET_RANK[str(left["trainable_parameter_bucket"])]
    right_cost = BUCKET_RANK[str(right["trainable_parameter_bucket"])]
    if equivalent and left_cost != right_cost:
        return (
            ("LEFT" if left_cost < right_cost else "RIGHT"),
            "parameter_efficiency",
        )
    if (
        not complete
        or _has_interval_width(left)
        or _has_interval_width(right)
    ):
        return "ABSTAIN", "insufficient_evidence"
    return "TRADEOFF", "irreducible_tradeoff"


def shift_verifier_array_target(
    candidates: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    incoming: set[int] = set()
    uncertain: set[int] = set()
    safe = [
        index
        for index, candidate in enumerate(candidates)
        if candidate["safety_qualified"]
    ]
    for offset, left_index in enumerate(safe):
        for right_index in safe[offset + 1 :]:
            forward = shift_verifier_target(
                {
                    "left": candidates[left_index],
                    "right": candidates[right_index],
                },
                tolerance=tolerance,
            )[0]
            reverse = shift_verifier_target(
                {
                    "left": candidates[right_index],
                    "right": candidates[left_index],
                },
                tolerance=tolerance,
            )[0]
            expected_reverse = {
                "LEFT": "RIGHT",
                "RIGHT": "LEFT",
                "TRADEOFF": "TRADEOFF",
                "ABSTAIN": "ABSTAIN",
            }[reverse]
            if forward != expected_reverse:
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
            index
            for index, status in enumerate(statuses)
            if status == "SELECTED"
        ],
        "dominated_indices": [
            index
            for index, status in enumerate(statuses)
            if status == "DOMINATED"
        ],
        "unresolved_indices": [
            index
            for index, status in enumerate(statuses)
            if status == "UNRESOLVED"
        ],
        "unsafe_indices": [
            index
            for index, status in enumerate(statuses)
            if status == "UNSAFE"
        ],
    }


def seal_shift_verifier_labels(
    *,
    contract_path: Path,
    manifest_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    contract = load_structured(contract_path)
    if (
        contract["schema_version"] != "6.0"
        or contract["status"] != "frozen_before_shift_generation"
    ):
        raise ValueError("shift challenge contract is not frozen v6")
    manifest, pairs, arrays, _ = load_shift_fixture(manifest_path)
    if manifest["contract_sha256"] != sha256_file(contract_path):
        raise ValueError("shift manifest belongs to another contract")
    if (
        contract["independent_builder"]["implementation_id"]
        == contract["independent_verifier"]["implementation_id"]
    ):
        raise ValueError("fixture builder and verifier are not independent")
    out_dir.mkdir(parents=True, exist_ok=False)
    tolerance = float(
        contract["shift_surface"]["metric_equivalence_tolerance"]
    )
    pair_labels = []
    for row in pairs:
        preference, rationale = shift_verifier_target(
            row["input"],
            tolerance=tolerance,
        )
        pair_labels.append(
            {
                "schema_version": "ldt-shift-verifier-pair-label.v6",
                "example_id": row["example_id"],
                "array_id": row["array_id"],
                "lineage_group_sha256": row["lineage_group_sha256"],
                "target": {
                    "preference": preference,
                    "rationale_code": rationale,
                },
                "authority": "ENVIRONMENT_OWNED",
            }
        )
    array_labels = []
    for row in arrays:
        array_labels.append(
            {
                "schema_version": "ldt-shift-verifier-array-label.v6",
                "array_id": row["array_id"],
                "lineage_group_sha256": row["lineage_group_sha256"],
                "target": shift_verifier_array_target(
                    [
                        candidate["features"]
                        for candidate in row["candidates"]
                    ],
                    tolerance=tolerance,
                ),
                "authority": "ENVIRONMENT_OWNED",
            }
        )
    pair_labels = sorted(
        pair_labels,
        key=lambda row: str(row["example_id"]),
    )
    array_labels = sorted(
        array_labels,
        key=lambda row: str(row["array_id"]),
    )
    pair_path = out_dir / "verifier_pair_labels.jsonl"
    array_path = out_dir / "verifier_array_labels.jsonl"
    _write_jsonl(pair_path, pair_labels)
    _write_jsonl(array_path, array_labels)
    preference_counts = Counter(
        str(row["target"]["preference"]) for row in pair_labels
    )
    rationale_counts = Counter(
        str(row["target"]["rationale_code"]) for row in pair_labels
    )
    status_counts = Counter(
        str(status)
        for row in array_labels
        for status in row["target"]["statuses"]
    )
    required = contract["sealed_verifier_custody"]
    checks = {
        "all_preferences_present": set(preference_counts)
        == set(HARD_PREFERENCES),
        "all_rationales_present": set(rationale_counts)
        == set(HARD_RATIONALES),
        "all_array_statuses_present": set(status_counts)
        == set(ARRAY_STATUSES),
        "pair_count_matches": len(pair_labels)
        == int(contract["shift_surface"]["required_pairs"]),
        "array_count_matches": len(array_labels)
        == int(contract["shift_surface"]["required_arrays"]),
        "sealed_before_candidate_inference_declared": (
            required["seal_stage"] == "before_candidate_inference"
        ),
        "labels_excluded_from_candidate_workspace": bool(
            required["candidate_workspace_receives_label_paths"] is False
        ),
    }
    receipt = {
        "schema_version": "ldt-shift-verifier-seal.v6",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "shift_manifest_sha256": sha256_file(manifest_path),
        "verifier_implementation": contract["independent_verifier"],
        "pair_labels_sha256": sha256_file(pair_path),
        "array_labels_sha256": sha256_file(array_path),
        "pair_count": len(pair_labels),
        "array_count": len(array_labels),
        "preference_distribution": dict(sorted(preference_counts.items())),
        "rationale_distribution": dict(sorted(rationale_counts.items())),
        "array_status_distribution": dict(sorted(status_counts.items())),
        "checks": checks,
        "seal_passed": all(checks.values()),
        "sealed_before_candidate_inference": True,
        "candidate_workspace_receives_label_paths": False,
        "authority": "ENVIRONMENT_OWNED",
        "training_lane": "SEALED_EVALUATION",
        "training_authorized": False,
        "model_selection_authorized": False,
    }
    _write_json(out_dir / "seal_receipt.json", receipt)
    return receipt
