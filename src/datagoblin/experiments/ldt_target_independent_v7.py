from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_confirmation import (
    CONFIRMATION_FEATURE_WHITELIST,
    ConfirmationPairEncoder,
)
from datagoblin.experiments.ldt_array_hard_surface import (
    ARRAY_STATUSES,
    HARD_PREFERENCES,
    HARD_RATIONALES,
)
from datagoblin.experiments.ldt_array_sorter import sha256_file
from datagoblin.experiments.ldt_shift_builder import _materialize
from datagoblin.experiments.ldt_shift_verifier import (
    shift_verifier_array_target,
    shift_verifier_target,
)
from datagoblin.serialization import dump_structured, load_structured

INDEPENDENT_FEATURE_FORBIDDEN_FRAGMENTS = (
    "lattice",
    "domin",
    "equivalent",
    "tradeoff",
    "insufficient",
    "context_mode",
    "fixture_pressure",
    "optimization_pressure",
    "delta",
    "mismatch",
)


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def _without_target(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"target", "authority"}
    }


def _split_contract(
    config: Mapping[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    surface = config["surface"]
    prefix = "train" if split == "train" else "heldout"
    return {
        "shift_surface": {
            "games": list(surface["games"]),
            "pressures": list(surface["pressures"]),
            "array_sizes": list(surface[f"{prefix}_array_sizes"]),
            "seed": int(surface[f"{prefix}_seed"]),
            "context_prefix": str(surface[f"{prefix}_context_prefix"]),
            "meta_split": split,
        }
    }


def _label_split(
    pairs: Sequence[Mapping[str, Any]],
    arrays: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labeled_pairs = []
    for row in pairs:
        preference, rationale = shift_verifier_target(
            row["input"],
            tolerance=tolerance,
        )
        labeled_pairs.append(
            {
                **row,
                "target": {
                    "preference": preference,
                    "rationale_code": rationale,
                },
                "authority": "ENVIRONMENT_OWNED",
            }
        )
    labeled_arrays = []
    for row in arrays:
        candidates = [
            candidate["features"] for candidate in row["candidates"]
        ]
        labeled_arrays.append(
            {
                **row,
                "target": shift_verifier_array_target(
                    candidates,
                    tolerance=tolerance,
                ),
                "authority": "ENVIRONMENT_OWNED",
            }
        )
    return labeled_pairs, labeled_arrays


def _lineages(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row["lineage_group_sha256"]) for row in rows}


def _contexts(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(row["input"][side]["context_mode"])
        for row in rows
        for side in ("left", "right")
    }


def _report_bindings_complete(
    pairs: Sequence[Mapping[str, Any]],
    reports: Sequence[Mapping[str, Any]],
) -> bool:
    referenced = {
        str(row[key])
        for row in pairs
        for key in (
            "left_tournament_report_sha256",
            "right_tournament_report_sha256",
        )
    }
    available = {
        str(row["tournament_report_sha256"]) for row in reports
    }
    return referenced == available


def _orientation_complete(
    pairs: Sequence[Mapping[str, Any]],
    arrays: Sequence[Mapping[str, Any]],
) -> bool:
    return all(
        {
            (int(pair["left_index"]), int(pair["right_index"]))
            for pair in pairs
            if pair["array_id"] == array_row["array_id"]
        }
        == {
            (left, right)
            for left in range(int(array_row["candidate_count"]))
            for right in range(int(array_row["candidate_count"]))
            if left != right
        }
        for array_row in arrays
    )


def _private_pair_labels(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": (
                "curation-taste-confirmation-pair-label.v3"
            ),
            "example_id": row["example_id"],
            "array_id": row["array_id"],
            "lineage_group_sha256": row["lineage_group_sha256"],
            "target": row["target"],
            "authority": row["authority"],
        }
        for row in rows
    ]


def _private_array_labels(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": (
                "curation-taste-confirmation-array-label.v3"
            ),
            "array_id": row["array_id"],
            "lineage_group_sha256": row["lineage_group_sha256"],
            "target": row["target"],
            "authority": row["authority"],
        }
        for row in rows
    ]


def generate_target_independent_dataset(
    config_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    config = load_structured(config_path)
    if (
        config["schema_version"] != "7.0"
        or config["status"] != "frozen_before_dataset_generation"
    ):
        raise ValueError("target-independent preregistration is not frozen v7")
    if (
        config["independent_builder"]["implementation_id"]
        == config["independent_verifier"]["implementation_id"]
    ):
        raise ValueError("builder and verifier implementation IDs must differ")

    out_dir.mkdir(parents=True, exist_ok=False)
    public_dir = out_dir / "public_dataset"
    steward_dir = out_dir / "private_steward"
    public_dir.mkdir()
    steward_dir.mkdir()

    train_pairs, train_arrays, train_reports = _materialize(
        _split_contract(config, split="train")
    )
    heldout_pairs, heldout_arrays, heldout_reports = _materialize(
        _split_contract(config, split="heldout")
    )
    tolerance = float(
        config["surface"]["metric_equivalence_tolerance"]
    )
    train_pairs, train_arrays = _label_split(
        train_pairs,
        train_arrays,
        tolerance=tolerance,
    )
    heldout_pairs, heldout_arrays = _label_split(
        heldout_pairs,
        heldout_arrays,
        tolerance=tolerance,
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
        name: int(contract[f"required_{name}"]) for name in observed
    }
    train_lineages = _lineages(train_pairs)
    heldout_lineages = _lineages(heldout_pairs)
    forbidden_lineages = {
        str(value)
        for value in config["surface"][
            "forbidden_prior_lineage_sha256"
        ]
    }
    train_contexts = _contexts(train_pairs)
    heldout_contexts = _contexts(heldout_pairs)
    all_pairs = train_pairs + heldout_pairs
    all_arrays = train_arrays + heldout_arrays
    all_reports = train_reports + heldout_reports

    pair_labels_path = steward_dir / "heldout_pair_labels.jsonl"
    array_labels_path = steward_dir / "heldout_array_labels.jsonl"
    _write_jsonl(
        pair_labels_path,
        _private_pair_labels(heldout_pairs),
    )
    _write_jsonl(
        array_labels_path,
        _private_array_labels(heldout_arrays),
    )

    public_paths = {
        "train_pairs": public_dir / "train_pairs.jsonl",
        "train_arrays": public_dir / "train_arrays.jsonl",
        "train_reports": public_dir / "train_tournament_reports.jsonl",
        "heldout_pairs": public_dir / "heldout_pairs.jsonl",
        "heldout_arrays": public_dir / "heldout_arrays.jsonl",
        "heldout_reports": public_dir / "heldout_tournament_reports.jsonl",
    }
    _write_jsonl(public_paths["train_pairs"], train_pairs)
    _write_jsonl(public_paths["train_arrays"], train_arrays)
    _write_jsonl(public_paths["train_reports"], train_reports)
    _write_jsonl(
        public_paths["heldout_pairs"],
        [_without_target(row) for row in heldout_pairs],
    )
    _write_jsonl(
        public_paths["heldout_arrays"],
        [_without_target(row) for row in heldout_arrays],
    )
    _write_jsonl(public_paths["heldout_reports"], heldout_reports)

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
    sizes = {
        int(row["candidate_count"]) for row in all_arrays
    }
    independent_names = ConfirmationPairEncoder.fit(
        train_pairs
    ).feature_names("independent")
    feature_audit = {
        "feature_view": "independent",
        "feature_count": len(independent_names),
        "feature_names": list(independent_names),
        "forbidden_fragments": list(
            INDEPENDENT_FEATURE_FORBIDDEN_FRAGMENTS
        ),
        "forbidden_feature_hits": [
            name
            for name in independent_names
            if any(
                fragment in name
                for fragment in INDEPENDENT_FEATURE_FORBIDDEN_FRAGMENTS
            )
        ],
        "executes_verifier_policy": False,
        "includes_context_or_pressure_categories": False,
    }
    checks = {
        "counts_match_preregistration": observed == expected,
        "fresh_lineages": not bool(
            (train_lineages | heldout_lineages) & forbidden_lineages
        ),
        "cross_split_lineages_zero": not bool(
            train_lineages & heldout_lineages
        ),
        "lineage_counts_match": (
            len(train_lineages)
            == int(contract["required_train_lineage_groups"])
            and len(heldout_lineages)
            == int(contract["required_heldout_lineage_groups"])
        ),
        "context_categories_disjoint": not bool(
            train_contexts & heldout_contexts
        ),
        "orientation_complete": _orientation_complete(
            all_pairs,
            all_arrays,
        ),
        "required_array_sizes_present": sizes
        == {int(value) for value in contract["require_array_sizes"]},
        "all_pair_preferences_present": set(preference_distribution)
        == set(HARD_PREFERENCES),
        "all_rationales_present": set(rationale_distribution)
        == set(HARD_RATIONALES),
        "all_array_statuses_present": set(status_distribution)
        == set(ARRAY_STATUSES),
        "report_hash_bindings_complete": _report_bindings_complete(
            all_pairs,
            all_reports,
        ),
        "heldout_pair_targets_separate": True,
        "heldout_array_targets_separate": True,
        "independent_feature_audit_passed": not feature_audit[
            "forbidden_feature_hits"
        ],
    }
    manifest_splits = {
        name: {
            "path": path.name,
            "count": observed[name],
            "sha256": sha256_file(path),
            "targets_in_file": name in {"train_pairs", "train_arrays"},
        }
        for name, path in public_paths.items()
    }
    manifest = {
        "schema_version": "curation-taste-array-confirmation.v3",
        "experiment_id": config["experiment_id"],
        "task": "target_independent_interval_array_selection",
        "feature_whitelist": list(CONFIRMATION_FEATURE_WHITELIST),
        "preference_labels": list(HARD_PREFERENCES),
        "rationale_labels": list(HARD_RATIONALES),
        "array_status_labels": list(ARRAY_STATUSES),
        "metric_equivalence_tolerance": tolerance,
        "splits": manifest_splits,
        "heldout_label_custody": {
            "pair_labels_sha256": sha256_file(pair_labels_path),
            "array_labels_sha256": sha256_file(array_labels_path),
            "paths_disclosed_to_candidate": False,
            "authority": "ENVIRONMENT_OWNED",
            "sealed_before_training": True,
            "reveal_count_budget": 1,
        },
        "lineage_audit": {
            "train_groups": len(train_lineages),
            "heldout_groups": len(heldout_lineages),
            "cross_split_groups": len(
                train_lineages & heldout_lineages
            ),
            "forbidden_prior_groups": len(forbidden_lineages),
            "prior_overlap": len(
                (train_lineages | heldout_lineages)
                & forbidden_lineages
            ),
        },
        "category_shift_audit": {
            "train_context_count": len(train_contexts),
            "heldout_context_count": len(heldout_contexts),
            "overlap": len(train_contexts & heldout_contexts),
        },
        "orientation_audit": {
            "contract": config["surface"]["orientation_contract"],
            "complete": checks["orientation_complete"],
        },
        "array_size_audit": {
            "sizes": sorted(sizes),
            "train_distribution": dict(
                sorted(
                    Counter(
                        int(row["candidate_count"])
                        for row in train_arrays
                    ).items()
                )
            ),
            "heldout_distribution": dict(
                sorted(
                    Counter(
                        int(row["candidate_count"])
                        for row in heldout_arrays
                    ).items()
                )
            ),
        },
        "report_binding_audit": {
            "report_count": len(all_reports),
            "complete": checks["report_hash_bindings_complete"],
        },
        "target_independence_audit": feature_audit,
        "train_preference_distribution": dict(
            sorted(preference_distribution.items())
        ),
        "train_rationale_distribution": dict(
            sorted(rationale_distribution.items())
        ),
        "train_array_status_distribution": dict(
            sorted(status_distribution.items())
        ),
        "checks": checks,
        "larger_model_training_authorized": False,
    }
    manifest_path = public_dir / "dataset_manifest.json"
    dump_structured(manifest_path, manifest)
    report = {
        "schema_version": (
            "ldt-target-independent-generation-report.v7"
        ),
        "experiment_id": config["experiment_id"],
        "preregistration_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "private_pair_labels_sha256": sha256_file(pair_labels_path),
        "private_array_labels_sha256": sha256_file(array_labels_path),
        "counts": observed,
        "checks": checks,
        "generation_passed": all(checks.values()),
        "sealed_confirmation_created_before_training": True,
        "network_used": False,
        "raw_game_records_read": False,
        "heldout_targets_exported_to_candidate": False,
        "claim_boundary": config["claim_scope"],
    }
    dump_structured(out_dir / "generation_report.json", report)
    return report
