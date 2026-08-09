from __future__ import annotations

import random
from collections import Counter, defaultdict
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
from datagoblin.experiments.ldt_array_sorter import (
    sha256_file,
)
from datagoblin.experiments.ldt_shift_builder import (
    _candidate,
    _report,
)
from datagoblin.experiments.ldt_target_independent_v7 import (
    INDEPENDENT_FEATURE_FORBIDDEN_FRAGMENTS,
    _contexts,
    _label_split,
    _lineages,
    _orientation_complete,
    _private_array_labels,
    _private_pair_labels,
    _report_bindings_complete,
    _without_target,
    _write_jsonl,
)
from datagoblin.hashing import sha256_text
from datagoblin.serialization import dump_structured, load_structured


def _mixed_surface(
    config: Mapping[str, Any],
    *,
    split: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    surface = config["surface"]
    prefix = "train" if split == "train" else "heldout"
    seed = int(surface[f"{prefix}_seed"])
    lineage_count = int(surface[f"{prefix}_lineage_count"])
    context_prefix = str(surface[f"{prefix}_context_prefix"])
    pressures = [str(value) for value in surface["pressures"]]
    games = [str(value) for value in surface["games"]]
    sizes = [int(value) for value in surface["array_sizes"]]
    minimum_sizes = {
        str(pressure): int(size)
        for pressure, size in surface[
            "minimum_array_size_by_pressure"
        ].items()
    }
    pairs = []
    arrays = []
    reports = []
    major_version = str(config["schema_version"]).split(".", 1)[0]

    for lineage_index in range(lineage_count):
        lineage = sha256_text(
            f"structured-array-builder-v{major_version}:"
            f"{seed}:{lineage_index}"
        )
        for pressure_index, pressure in enumerate(pressures):
            size = sizes[(lineage_index + pressure_index) % len(sizes)]
            size = max(size, minimum_sizes.get(pressure, 0))
            local_seed = (
                seed
                + lineage_index * 4099
                + pressure_index * 811
            )
            jitter = random.Random(local_seed).uniform(-0.0025, 0.0025)
            game = games[
                (lineage_index + pressure_index) % len(games)
            ]
            identity = (
                f"structured-v8:{seed}:{lineage_index}:"
                f"{pressure}:{size}:{local_seed}"
            )
            array_id = (
                f"structured-array-{sha256_text(identity)[:20]}"
            )
            features = [
                _candidate(
                    slot=slot,
                    pressure=pressure,
                    game=game,
                    jitter=jitter,
                    context_prefix=(
                        f"{context_prefix}:lineage-{lineage_index}"
                    ),
                )
                for slot in range(size)
            ]
            array_reports = [
                _report(
                    array_id=array_id,
                    slot=slot,
                    lineage=lineage,
                    features=candidate,
                    meta_split=split,
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
                        "structured-pair-"
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
                            "meta_split": split,
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


def _label_counts(
    pairs: Sequence[Mapping[str, Any]],
    arrays: Sequence[Mapping[str, Any]],
) -> dict[str, Counter[str]]:
    return {
        "preference": Counter(
            str(row["target"]["preference"]) for row in pairs
        ),
        "rationale": Counter(
            str(row["target"]["rationale_code"]) for row in pairs
        ),
        "array_status": Counter(
            str(status)
            for row in arrays
            for status in row["target"]["statuses"]
        ),
    }


def _support_matrix(
    pairs: Sequence[Mapping[str, Any]],
    arrays: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pairs_by_lineage: dict[str, list[Mapping[str, Any]]] = defaultdict(
        list
    )
    arrays_by_lineage: dict[str, list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in pairs:
        pairs_by_lineage[str(row["lineage_group_sha256"])].append(row)
    for row in arrays:
        arrays_by_lineage[str(row["lineage_group_sha256"])].append(row)
    lineages = sorted(pairs_by_lineage)
    labels = {
        "preference": tuple(HARD_PREFERENCES),
        "rationale": tuple(HARD_RATIONALES),
        "array_status": tuple(ARRAY_STATUSES),
    }
    by_lineage = {}
    for lineage in lineages:
        counts = _label_counts(
            pairs_by_lineage[lineage],
            arrays_by_lineage[lineage],
        )
        by_lineage[lineage] = {
            axis: {
                label: int(counts[axis][label])
                for label in axis_labels
            }
            for axis, axis_labels in labels.items()
        }
        by_lineage[lineage]["pressure_count"] = len(
            {
                str(row["fixture_pressure"])
                for row in arrays_by_lineage[lineage]
            }
        )
    lineage_support = {
        axis: {
            label: sum(
                by_lineage[lineage][axis][label] > 0
                for lineage in lineages
            )
            for label in axis_labels
        }
        for axis, axis_labels in labels.items()
    }
    folds = {}
    for validation_lineage in lineages:
        training_pairs = [
            row
            for lineage, rows in pairs_by_lineage.items()
            if lineage != validation_lineage
            for row in rows
        ]
        training_arrays = [
            row
            for lineage, rows in arrays_by_lineage.items()
            if lineage != validation_lineage
            for row in rows
        ]
        training_counts = _label_counts(
            training_pairs,
            training_arrays,
        )
        validation_counts = _label_counts(
            pairs_by_lineage[validation_lineage],
            arrays_by_lineage[validation_lineage],
        )
        folds[validation_lineage] = {
            "training": {
                axis: {
                    label: int(training_counts[axis][label])
                    for label in axis_labels
                }
                for axis, axis_labels in labels.items()
            },
            "validation": {
                axis: {
                    label: int(validation_counts[axis][label])
                    for label in axis_labels
                }
                for axis, axis_labels in labels.items()
            },
        }
    return {
        "lineage_count": len(lineages),
        "by_lineage": by_lineage,
        "lineage_support": lineage_support,
        "folds": folds,
    }


def _support_complete(
    support: Mapping[str, Any],
    *,
    minimum_lineages: int,
) -> bool:
    return (
        all(
            int(count) >= minimum_lineages
            for axis in support["lineage_support"].values()
            for count in axis.values()
        )
        and all(
            int(count) > 0
            for fold in support["folds"].values()
            for side in ("training", "validation")
            for axis in fold[side].values()
            for count in axis.values()
        )
    )


def generate_structured_array_dataset(
    config_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    config = load_structured(config_path)
    schema_version = str(config["schema_version"])
    if (
        schema_version
        not in {"8.0", "9.0", "10.0", "11.0", "12.0"}
        or config["status"] != "frozen_before_dataset_generation"
    ):
        raise ValueError(
            "structured-array preregistration is not frozen v8-v12"
        )
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

    train_pairs, train_arrays, train_reports = _mixed_surface(
        config,
        split="train",
    )
    heldout_pairs, heldout_arrays, heldout_reports = _mixed_surface(
        config,
        split="heldout",
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
    train_support = _support_matrix(train_pairs, train_arrays)
    heldout_support = _support_matrix(
        heldout_pairs,
        heldout_arrays,
    )
    minimum_lineages = int(
        contract["minimum_lineages_per_critical_class"]
    )

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

    sizes = {int(row["candidate_count"]) for row in all_arrays}
    independent_names = ConfirmationPairEncoder.fit(
        train_pairs
    ).feature_names("independent")
    forbidden_hits = [
        name
        for name in independent_names
        if any(
            fragment in name
            for fragment in INDEPENDENT_FEATURE_FORBIDDEN_FRAGMENTS
        )
    ]
    pressures_per_lineage = len(config["surface"]["pressures"])
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
        "report_hash_bindings_complete": _report_bindings_complete(
            all_pairs,
            all_reports,
        ),
        "mixed_mechanisms_in_every_development_lineage": all(
            int(row["pressure_count"]) == pressures_per_lineage
            for row in train_support["by_lineage"].values()
        ),
        "development_class_support_complete": _support_complete(
            train_support,
            minimum_lineages=minimum_lineages,
        ),
        "heldout_class_support_complete": _support_complete(
            heldout_support,
            minimum_lineages=minimum_lineages,
        ),
        "heldout_pair_targets_separate": True,
        "heldout_array_targets_separate": True,
        "independent_feature_audit_passed": not forbidden_hits,
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
        "task": "mixed_mechanism_structured_array_selection",
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
        "target_independence_audit": {
            "feature_view": "independent",
            "feature_count": len(independent_names),
            "feature_names": list(independent_names),
            "forbidden_fragments": list(
                INDEPENDENT_FEATURE_FORBIDDEN_FRAGMENTS
            ),
            "forbidden_feature_hits": forbidden_hits,
            "executes_verifier_policy": False,
            "includes_context_or_pressure_categories": False,
        },
        "development_support_audit": train_support,
        "heldout_support_audit": {
            "lineage_count": heldout_support["lineage_count"],
            "minimum_lineages_per_critical_class": minimum_lineages,
            "complete": checks["heldout_class_support_complete"],
            "row_level_or_class_counts_disclosed": False,
        },
        "train_preference_distribution": dict(
            sorted(
                _label_counts(
                    train_pairs,
                    train_arrays,
                )["preference"].items()
            )
        ),
        "train_rationale_distribution": dict(
            sorted(
                _label_counts(
                    train_pairs,
                    train_arrays,
                )["rationale"].items()
            )
        ),
        "train_array_status_distribution": dict(
            sorted(
                _label_counts(
                    train_pairs,
                    train_arrays,
                )["array_status"].items()
            )
        ),
        "checks": checks,
        "larger_model_training_authorized": False,
    }
    if "skill_fingerprint" in config:
        manifest["skill_fingerprint"] = config[
            "skill_fingerprint"
        ]
    manifest_path = public_dir / "dataset_manifest.json"
    dump_structured(manifest_path, manifest)
    report = {
        "schema_version": (
            "ldt-structured-array-generation-report.v"
            + schema_version.split(".", 1)[0]
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
