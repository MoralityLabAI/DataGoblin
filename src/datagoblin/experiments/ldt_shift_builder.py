from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_confirmation import (
    CONFIRMATION_FEATURE_WHITELIST,
    validate_confirmation_pair,
    validate_confirmation_side,
)
from datagoblin.experiments.ldt_array_sorter import (
    METRIC_FIELDS,
    sha256_file,
    sha256_json,
)
from datagoblin.hashing import sha256_text
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _interval(
    lower: float | None,
    upper: float | None,
) -> dict[str, float | None]:
    return {
        "lower": None if lower is None else max(0.0, min(1.0, lower)),
        "upper": None if upper is None else max(0.0, min(1.0, upper)),
    }


def _point(value: float) -> dict[str, float | None]:
    return _interval(value, value)


def _base_vector(jitter: float) -> tuple[float, ...]:
    return (
        0.740 + jitter,
        0.752 - jitter / 2,
        0.736 + jitter / 3,
        0.748 - jitter / 4,
    )


def _candidate(
    *,
    slot: int,
    pressure: str,
    game: str,
    jitter: float,
    context_prefix: str,
) -> dict[str, Any]:
    base = _base_vector(jitter)
    safe = slot != 2
    bucket = "medium"
    optimization = "balanced"
    mode = "baseline"
    metrics = tuple(_point(value) for value in base)

    receipt_pressures = {
        "receipt_clear_dominance",
        "receipt_seed_interval_overlap",
        "receipt_mixed_frontier",
        "receipt_parameter_equivalence",
        "receipt_cross_neighbor_frontier",
        "receipt_safety_sentinel",
    }
    if pressure in receipt_pressures:
        receipt_base = (
            0.24 + jitter,
            0.89 - jitter / 2,
            0.34 + jitter / 3,
            0.91 - jitter / 4,
        )
        safe = True
        bucket = "medium"
        optimization = "adapter_receipt_quality"
        mode = "receipt_baseline"
        metrics = tuple(_point(value) for value in receipt_base)
        if pressure == "receipt_clear_dominance":
            vectors = (
                receipt_base,
                (0.96, 0.995, 0.80, 0.985),
                (0.84, 0.975, 0.51, 0.955),
                (0.88, 0.982, 0.61, 0.965),
                (0.90, 0.986, 0.68, 0.972),
                (0.82, 0.968, 0.49, 0.948),
            )
            vector = tuple(value + jitter for value in vectors[slot])
            metrics = tuple(_point(value) for value in vector)
            bucket = ("zero", "medium", "small", "small", "large", "zero")[slot]
            optimization = "clear_conservative_dominance"
            mode = ("control", "robust", "one_pass", "looped", "wide", "compact")[slot]
        elif pressure == "receipt_seed_interval_overlap":
            bounds = (
                tuple((value, value) for value in receipt_base),
                ((0.93, 1.0), (0.982, 1.0), (0.60, 0.88), (0.955, 0.995)),
                ((0.94, 1.0), (0.981, 1.0), (0.55, 0.91), (0.952, 0.996)),
                ((0.84, 0.99), (0.965, 0.998), (0.47, 0.80), (0.938, 0.982)),
                ((0.86, 1.0), (0.968, 1.0), (0.45, 0.83), (0.940, 0.985)),
                ((0.89, 0.99), (0.974, 0.998), (0.52, 0.86), (0.947, 0.989)),
            )
            metrics = tuple(
                _interval(lower + jitter, upper + jitter)
                for lower, upper in bounds[slot]
            )
            bucket = ("zero", "medium", "medium", "small", "small", "large")[slot]
            optimization = "seed_interval_uncertainty"
            mode = ("control", "low_rank_a", "low_rank_b", "one_pass", "looped", "wide")[slot]
        elif pressure == "receipt_mixed_frontier":
            vectors = (
                receipt_base,
                (0.98, 0.995, 0.62, 0.958),
                (0.87, 0.975, 0.93, 0.995),
                (0.94, 0.988, 0.79, 0.978),
                (0.79, 0.960, 0.48, 0.941),
                (0.83, 0.968, 0.58, 0.950),
            )
            metrics = tuple(_point(value + jitter) for value in vectors[slot])
            bucket = ("zero", "small", "small", "medium", "zero", "large")[slot]
            optimization = "mixed_pareto_frontier"
            mode = ("control", "ordinary", "ood", "balanced", "weak_a", "weak_b")[slot]
        elif pressure == "receipt_parameter_equivalence":
            vectors = (
                receipt_base,
                (0.915, 0.985, 0.745, 0.975),
                (0.918, 0.987, 0.749, 0.978),
                (0.921, 0.989, 0.752, 0.981),
                (0.82, 0.965, 0.55, 0.948),
                (0.80, 0.958, 0.51, 0.942),
            )
            metrics = tuple(_point(value + jitter) for value in vectors[slot])
            bucket = ("zero", "large", "medium", "small", "zero", "medium")[slot]
            optimization = "equivalent_parameter_tiebreak"
            mode = ("control", "expensive", "mid_cost", "compact", "weak_a", "weak_b")[slot]
        elif pressure == "receipt_cross_neighbor_frontier":
            vectors = (
                receipt_base,
                (0.96, 0.990, 0.64, 0.950),
                (0.66, 0.955, 0.96, 0.992),
                (0.83, 0.975, 0.82, 0.973),
                (0.90, 0.982, 0.73, 0.962),
                (0.78, 0.960, 0.52, 0.940),
            )
            metrics = tuple(_point(value + jitter) for value in vectors[slot])
            bucket = ("zero", "small", "small", "medium", "large", "zero")[slot]
            optimization = "neighbor_coupled_pareto"
            mode = ("control", "ordinary", "ood", "middle", "tilted", "weak")[slot]
        else:
            vectors = (
                receipt_base,
                (0.94, 0.990, 0.79, 0.980),
                (0.995, 0.999, 0.995, 0.999),
                (0.82, 0.968, 0.54, 0.948),
                (0.90, 0.982, 0.70, 0.965),
                (0.86, 0.974, 0.62, 0.956),
            )
            metrics = tuple(_point(value + jitter) for value in vectors[slot])
            safe = slot != 2
            bucket = ("zero", "medium", "zero", "small", "large", "small")[slot]
            optimization = "fixed_safety_membrane_canary"
            mode = ("control", "robust", "unsafe_high", "weak", "wide", "compact")[slot]
    elif slot == 2:
        metrics = tuple(_point(0.99) for _ in METRIC_FIELDS)
        bucket = "zero"
        optimization = "coverage_without_safety"
        mode = "unsafe_high_coverage"
    elif pressure == "near_tolerance_dominance":
        if slot == 1:
            metrics = tuple(_point(value + 0.012) for value in base)
            bucket = "large"
            optimization = "just_over_tolerance"
            mode = "near_dominant"
        elif slot == 3:
            vector = (
                base[0] + 0.028,
                base[1] + 0.028,
                base[2] - 0.028,
                base[3] - 0.028,
            )
            metrics = tuple(_point(value) for value in vector)
            bucket = "small"
            optimization = "ordinary_specialist"
            mode = "ordinary_specialist"
        elif slot == 4:
            metrics = tuple(_point(value + 0.005) for value in base)
            bucket = "small"
            optimization = "equivalent_compact"
            mode = "equivalent_compact"
        elif slot == 5:
            vector = (
                base[0] - 0.028,
                base[1] - 0.028,
                base[2] + 0.028,
                base[3] + 0.028,
            )
            metrics = tuple(_point(value) for value in vector)
            bucket = "zero"
            optimization = "ood_specialist"
            mode = "ood_specialist"
    elif pressure == "near_tolerance_equivalence":
        if slot == 1:
            metrics = tuple(_point(value + 0.009) for value in base)
            bucket = "large"
            optimization = "equivalent_expensive"
            mode = "upper_equivalent"
        elif slot == 3:
            metrics = tuple(_point(value - 0.009) for value in base)
            bucket = "small"
            optimization = "equivalent_compact"
            mode = "lower_equivalent"
        elif slot == 4:
            vector = tuple(
                value + (0.007 if index % 2 == 0 else -0.007)
                for index, value in enumerate(base)
            )
            metrics = tuple(_point(value) for value in vector)
            bucket = "zero"
            optimization = "equivalent_zero_parameter"
            mode = "mixed_equivalent"
        elif slot == 5:
            metrics = tuple(_point(value + 0.012) for value in base)
            bucket = "large"
            optimization = "just_over_tolerance"
            mode = "near_dominant"
    elif pressure == "controlled_missingness":
        if slot == 1:
            metrics = tuple(
                _interval(value + 0.025, None) for value in base
            )
            bucket = "small"
            optimization = "lower_bound_only"
            mode = "robust_lower_bound"
        elif slot == 3:
            metrics = tuple(
                _interval(None, value - 0.025) for value in base
            )
            bucket = "small"
            optimization = "upper_bound_only"
            mode = "robust_upper_bound"
        elif slot == 4:
            provisional = [_point(value + 0.015) for value in base]
            provisional[1] = _interval(None, None)
            metrics = tuple(provisional)
            bucket = "zero"
            optimization = "missing_axis"
            mode = "one_axis_missing"
        elif slot == 5:
            metrics = tuple(
                _interval(value - 0.008, value + 0.008)
                for value in base
            )
            bucket = "medium"
            optimization = "bounded_interval"
            mode = "narrow_interval"
    elif pressure == "boundary_pareto_tradeoff":
        if slot == 1:
            vector = (
                base[0] + 0.030,
                base[1] + 0.030,
                base[2] - 0.030,
                base[3] - 0.030,
            )
            metrics = tuple(_point(value) for value in vector)
            bucket = "small"
            optimization = "ordinary_specialist"
            mode = "ordinary_specialist"
        elif slot == 3:
            vector = (
                base[0] - 0.030,
                base[1] - 0.030,
                base[2] + 0.030,
                base[3] + 0.030,
            )
            metrics = tuple(_point(value) for value in vector)
            bucket = "zero"
            optimization = "ood_specialist"
            mode = "ood_specialist"
        elif slot == 4:
            metrics = tuple(_point(value + 0.012) for value in base)
            bucket = "large"
            optimization = "just_over_tolerance"
            mode = "near_dominant"
        elif slot == 5:
            metrics = tuple(_point(value + 0.005) for value in base)
            bucket = "small"
            optimization = "equivalent_compact"
            mode = "equivalent_compact"
    else:
        raise ValueError(f"unsupported shift pressure: {pressure}")

    result: dict[str, Any] = {
        "game": game,
        "fixture_pressure": pressure,
        "context_mode": f"{context_prefix}:{pressure}:{mode}",
        "optimization_pressure": optimization,
        "trainable_parameter_bucket": bucket,
        "safety_qualified": safe,
    }
    result.update(
        {
            field: value
            for field, value in zip(METRIC_FIELDS, metrics, strict=True)
        }
    )
    validate_confirmation_side(result)
    return result


def _report(
    *,
    array_id: str,
    slot: int,
    lineage: str,
    features: Mapping[str, Any],
    meta_split: str = "heldout",
) -> dict[str, Any]:
    payload = {
        "schema_version": "content-free-tournament-summary.v6",
        "report_id": f"shift-report-{sha256_text(f'{array_id}:{slot}')[:20]}",
        "array_id": array_id,
        "candidate_slot": slot,
        "meta_split": meta_split,
        "lineage_group_sha256": lineage,
        "feature_summary": dict(features),
        "contains_row_level_game_records": False,
        "contains_hidden_reasoning": False,
        "authority": "INDEPENDENT_FIXTURE_BUILDER",
    }
    return {**payload, "tournament_report_sha256": sha256_json(payload)}


def _materialize(
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    surface = contract["shift_surface"]
    seed = int(surface["seed"])
    meta_split = str(surface.get("meta_split", "heldout"))
    if meta_split not in {"train", "heldout"}:
        raise ValueError("shift surface meta_split must be train or heldout")
    games = [str(value) for value in surface["games"]]
    pairs = []
    arrays = []
    reports = []
    for pressure_index, pressure_value in enumerate(surface["pressures"]):
        pressure = str(pressure_value)
        lineage = sha256_text(
            f"independent-shift-builder-v6:{pressure}:{seed}"
        )
        for array_offset, size_value in enumerate(surface["array_sizes"]):
            size = int(size_value)
            local_seed = seed + pressure_index * 2081 + array_offset * 79
            jitter = random.Random(local_seed).uniform(-0.0025, 0.0025)
            game = games[(pressure_index + array_offset) % len(games)]
            identity = (
                f"shift-v6:{pressure}:{array_offset}:{size}:{local_seed}"
            )
            array_id = f"shift-array-{sha256_text(identity)[:20]}"
            features = [
                _candidate(
                    slot=slot,
                    pressure=pressure,
                    game=game,
                    jitter=jitter,
                    context_prefix=str(surface["context_prefix"]),
                )
                for slot in range(size)
            ]
            array_reports = [
                _report(
                    array_id=array_id,
                    slot=slot,
                    lineage=lineage,
                    features=candidate,
                    meta_split=meta_split,
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
                    example_id = (
                        "shift-pair-"
                        + sha256_text(
                            f"{identity}:{left_index}:{right_index}"
                        )[:20]
                    )
                    pair_ids.append(example_id)
                    pairs.append(
                        {
                            "schema_version": (
                                "curation-taste-confirmation-pair.v3"
                            ),
                            "example_id": example_id,
                            "array_id": array_id,
                            "left_index": left_index,
                            "right_index": right_index,
                            "left_tournament_report_sha256": (
                                report_hashes[left_index]
                            ),
                            "right_tournament_report_sha256": (
                                report_hashes[right_index]
                            ),
                            "meta_split": meta_split,
                            "lineage_group_sha256": lineage,
                            "source_kind": (
                                "content_free_tournament_summary"
                            ),
                            "input": {
                                "left": features[left_index],
                                "right": features[right_index],
                            },
                        }
                    )
            arrays.append(
                {
                    "schema_version": (
                        "curation-taste-confirmation-array.v3"
                    ),
                    "array_id": array_id,
                    "meta_split": meta_split,
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
                            report_hashes,
                            features,
                            strict=True,
                        )
                    ],
                    "pair_example_ids": sorted(pair_ids),
                }
            )
    return (
        sorted(pairs, key=lambda row: str(row["example_id"])),
        sorted(arrays, key=lambda row: str(row["array_id"])),
        sorted(reports, key=lambda row: str(row["report_id"])),
    )


def _surface_counts(
    pairs: Sequence[Mapping[str, Any]],
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    missing_bounds = 0
    interval_metrics = 0
    for report in reports:
        features = report["feature_summary"]
        for field in METRIC_FIELDS:
            metric = features[field]
            missing_bounds += int(metric["lower"] is None)
            missing_bounds += int(metric["upper"] is None)
            interval_metrics += int(
                metric["lower"] is not None
                and metric["upper"] is not None
                and float(metric["lower"]) != float(metric["upper"])
            )
    near_tolerance = 0
    for row in pairs:
        for field in METRIC_FIELDS:
            left = row["input"]["left"][field]
            right = row["input"]["right"][field]
            if (
                left["lower"] is not None
                and left["lower"] == left["upper"]
                and right["lower"] is not None
                and right["lower"] == right["upper"]
                and 0.008
                <= abs(float(left["lower"]) - float(right["lower"]))
                <= 0.013
            ):
                near_tolerance += 1
    return {
        "missing_bound_count": missing_bounds,
        "complete_interval_metric_count": interval_metrics,
        "near_tolerance_directed_metric_count": near_tolerance,
    }


def generate_shift_fixture(
    contract_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    contract = load_structured(contract_path)
    if (
        contract["schema_version"] != "6.0"
        or contract["status"] != "frozen_before_shift_generation"
    ):
        raise ValueError("shift challenge contract is not frozen v6")
    out_dir.mkdir(parents=True, exist_ok=False)
    pairs, arrays, reports = _materialize(contract)
    public_dir = out_dir / "public_shift"
    public_dir.mkdir()
    pair_path = public_dir / "shift_pairs.jsonl"
    array_path = public_dir / "shift_arrays.jsonl"
    report_path = public_dir / "shift_tournament_reports.jsonl"
    _write_jsonl(pair_path, pairs)
    _write_jsonl(array_path, arrays)
    _write_jsonl(report_path, reports)

    surface = contract["shift_surface"]
    observed = {
        "pairs": len(pairs),
        "arrays": len(arrays),
        "reports": len(reports),
    }
    expected = {
        "pairs": int(surface["required_pairs"]),
        "arrays": int(surface["required_arrays"]),
        "reports": int(surface["required_reports"]),
    }
    surface_counts = _surface_counts(pairs, reports)
    lineages = {str(row["lineage_group_sha256"]) for row in pairs}
    forbidden = {
        str(value) for value in surface["forbidden_prior_lineage_sha256"]
    }
    report_hashes = {
        str(row["tournament_report_sha256"]) for row in reports
    }
    referenced_hashes = {
        str(row[key])
        for row in pairs
        for key in (
            "left_tournament_report_sha256",
            "right_tournament_report_sha256",
        )
    }
    checks = {
        "counts_match_contract": observed == expected,
        "fresh_lineages": not bool(lineages & forbidden),
        "lineage_count_matches": len(lineages)
        == int(surface["required_lineage_groups"]),
        "report_bindings_complete": report_hashes == referenced_hashes,
        "all_pairs_target_free": all(
            "target" not in row and "authority" not in row for row in pairs
        ),
        "all_arrays_target_free": all(
            "target" not in row and "authority" not in row for row in arrays
        ),
        "all_reports_content_free": all(
            not row["contains_row_level_game_records"]
            and not row["contains_hidden_reasoning"]
            for row in reports
        ),
        "missingness_count_matches": (
            surface_counts["missing_bound_count"]
            == int(surface["required_missing_bound_count"])
        ),
        "interval_pressure_present": (
            surface_counts["complete_interval_metric_count"]
            >= int(surface["minimum_complete_interval_metric_count"])
        ),
        "boundary_density_present": (
            surface_counts["near_tolerance_directed_metric_count"]
            >= int(surface["minimum_near_tolerance_directed_metric_count"])
        ),
    }
    manifest = {
        "schema_version": "ldt-independent-shift-fixture.v6",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "builder": contract["independent_builder"],
        "feature_whitelist": list(CONFIRMATION_FEATURE_WHITELIST),
        "splits": {
            "shift_pairs": {
                "path": pair_path.name,
                "count": len(pairs),
                "sha256": sha256_file(pair_path),
                "targets_in_file": False,
            },
            "shift_arrays": {
                "path": array_path.name,
                "count": len(arrays),
                "sha256": sha256_file(array_path),
                "targets_in_file": False,
            },
            "shift_reports": {
                "path": report_path.name,
                "count": len(reports),
                "sha256": sha256_file(report_path),
                "targets_in_file": False,
            },
        },
        "counts": observed,
        "surface_counts": surface_counts,
        "lineage_audit": {
            "shift_groups": len(lineages),
            "forbidden_prior_groups": len(forbidden),
            "overlap": len(lineages & forbidden),
        },
        "preference_targets_materialized": False,
        "array_targets_materialized": False,
        "verifier_state_present": False,
        "training_lane": "SEALED_EVALUATION",
        "checks": checks,
        "training_authorized": False,
        "model_selection_authorized": False,
    }
    manifest_path = public_dir / "shift_manifest.json"
    _write_json(manifest_path, manifest)
    report = {
        "schema_version": "ldt-independent-shift-generation-report.v6",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "shift_manifest_sha256": sha256_file(manifest_path),
        "counts": observed,
        "surface_counts": surface_counts,
        "pressure_distribution": dict(
            sorted(
                Counter(
                    str(row["fixture_pressure"]) for row in arrays
                ).items()
            )
        ),
        "checks": checks,
        "generation_passed": all(checks.values()),
        "targets_generated": False,
        "raw_game_records_read": False,
        "network_used": False,
    }
    _write_json(out_dir / "generation_report.json", report)
    return report


def load_shift_fixture(
    manifest_path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["schema_version"] != "ldt-independent-shift-fixture.v6"
        or manifest["verifier_state_present"]
        or manifest["preference_targets_materialized"]
        or manifest["array_targets_materialized"]
        or not all(bool(value) for value in manifest["checks"].values())
    ):
        raise ValueError("independent shift manifest is invalid")
    loaded = {}
    for name, spec in manifest["splits"].items():
        path = manifest_path.parent / str(spec["path"])
        if sha256_file(path) != spec["sha256"]:
            raise ValueError(f"shift split hash mismatch: {name}")
        rows = _read_jsonl(path)
        if len(rows) != int(spec["count"]) or spec["targets_in_file"]:
            raise ValueError(f"shift split count or custody differs: {name}")
        loaded[name] = rows
    pairs = loaded["shift_pairs"]
    arrays = loaded["shift_arrays"]
    reports = loaded["shift_reports"]
    for row in pairs:
        validate_confirmation_pair(row, targets_required=False)
    report_map = {}
    for report in reports:
        payload = {
            key: value
            for key, value in report.items()
            if key != "tournament_report_sha256"
        }
        report_hash = str(report["tournament_report_sha256"])
        if (
            report["schema_version"]
            != "content-free-tournament-summary.v6"
            or report_hash != sha256_json(payload)
            or report_hash in report_map
            or report["contains_row_level_game_records"]
            or report["contains_hidden_reasoning"]
        ):
            raise ValueError("shift report is invalid")
        validate_confirmation_side(report["feature_summary"])
        report_map[report_hash] = report
    referenced = {
        str(row[key])
        for row in pairs
        for key in (
            "left_tournament_report_sha256",
            "right_tournament_report_sha256",
        )
    }
    if referenced != set(report_map):
        raise ValueError("shift report coverage differs")
    for row in arrays:
        if "target" in row or "authority" in row:
            raise ValueError("shift array contains verifier state")
        if int(row["candidate_count"]) != len(row["candidates"]):
            raise ValueError("shift array candidate count differs")
        for candidate in row["candidates"]:
            validate_confirmation_side(candidate["features"])
            if candidate["tournament_report_sha256"] not in report_map:
                raise ValueError("shift array report binding is unknown")
    return manifest, pairs, arrays, reports
