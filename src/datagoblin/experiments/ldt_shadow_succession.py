from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_sorter import sha256_file
from datagoblin.experiments.ldt_shadow import (
    evaluate_shadow,
    generate_shadow_fixture,
    load_shadow_fixture,
    run_shadow,
)
from datagoblin.serialization import dump_structured, load_structured


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


def _load_parent_contract(contract_path: Path) -> dict[str, Any]:
    contract = load_structured(contract_path)
    if (
        contract["schema_version"] != "5.0"
        or contract["status"] != "frozen_before_shadow_generation"
    ):
        raise ValueError("shadow succession contract is not frozen v5")
    surface = contract["succession_surface"]
    rounds = surface["rounds"]
    if len(rounds) < 2:
        raise ValueError("shadow succession requires multiple rounds")
    ids = [str(round_spec["shadow_run_id"]) for round_spec in rounds]
    seeds = [int(round_spec["seed"]) for round_spec in rounds]
    contexts = [str(round_spec["context_prefix"]) for round_spec in rounds]
    if (
        len(set(ids)) != len(ids)
        or len(set(seeds)) != len(seeds)
        or len(set(contexts)) != len(contexts)
    ):
        raise ValueError("shadow succession round bindings must be unique")
    forbidden_seeds = {
        int(value) for value in surface["forbidden_prior_seeds"]
    }
    if set(seeds) & forbidden_seeds:
        raise ValueError("shadow succession reuses a forbidden seed")
    if not surface["require_cross_round_lineage_disjointness"]:
        raise ValueError("cross-round lineage disjointness must be required")
    custody = contract["custody"]
    boundary = contract["decision_boundary"]
    if (
        not custody["one_independent_join_per_round"]
        or not custody["aggregate_contains_comparison_receipts_only"]
        or boundary["shadow_decision_effect"] != "NONE"
        or not boundary["independent_verifier_remains_decision_owner"]
        or boundary["automatic_corpus_promotion_authorized"]
        or boundary["training_from_disagreements_authorized"]
        or boundary["larger_model_training_authorized"]
        or boundary["adapter_selection_authorized"]
    ):
        raise ValueError("shadow succession authority boundary is invalid")
    return contract


def _derived_round_contract(
    parent: Mapping[str, Any],
    *,
    parent_sha256: str,
    round_spec: Mapping[str, Any],
    forbidden_lineages: Sequence[str],
    ordinal: int,
) -> dict[str, Any]:
    surface = parent["succession_surface"]
    monitoring = parent["monitoring"]
    return {
        "schema_version": "4.0",
        "experiment_id": parent["experiment_id"],
        "status": "frozen_before_shadow_generation",
        "shadow_run_id": round_spec["shadow_run_id"],
        "purpose": parent["purpose"],
        "claim_scope": parent["claim_scope"],
        "source_confirmation": parent["source_confirmation"],
        "frozen_model": parent["frozen_model"],
        "approved_input_scope": parent["approved_input_scope"],
        "fresh_shadow_surface": {
            "games": surface["games"],
            "pressures": surface["pressures"],
            "array_sizes": round_spec["array_sizes"],
            "seed": round_spec["seed"],
            "forbidden_prior_seeds": surface["forbidden_prior_seeds"],
            "forbidden_prior_lineage_sha256": list(forbidden_lineages),
            "context_prefix": round_spec["context_prefix"],
            "metric_equivalence_tolerance": surface[
                "metric_equivalence_tolerance"
            ],
            "orientation_contract": surface["orientation_contract"],
            "required_pairs": surface["required_pairs_per_round"],
            "required_arrays": surface["required_arrays_per_round"],
            "required_reports": surface["required_reports_per_round"],
        },
        "custody": {
            key: parent["custody"][key]
            for key in (
                "public_shadow_inputs_include_targets",
                "verifier_pair_targets_available_to_shadow_runner",
                "verifier_array_targets_available_to_shadow_runner",
                "verifier_authority",
                "shadow_authority",
                "evaluated_records_immutable",
                "corrections_append_superseding_receipts",
            )
        },
        "monitoring": {
            "calibration_bins": monitoring["calibration_bins"],
            "maximum_ece": monitoring["maximum_ece_per_round"],
            "minimum_pair_preference_macro_f1": monitoring[
                "minimum_pair_preference_macro_f1_per_round"
            ],
            "minimum_abstain_recall": monitoring[
                "minimum_abstain_recall_per_round"
            ],
            "minimum_selected_set_exact_accuracy": monitoring[
                "minimum_selected_set_exact_accuracy_per_round"
            ],
            "require_safety_accuracy": monitoring[
                "require_safety_accuracy_per_round"
            ],
            "require_swap_consistency": monitoring[
                "require_swap_consistency_per_round"
            ],
            "report_by_game": monitoring["report_by_game"],
            "report_by_array_size": monitoring["report_by_array_size"],
            "disagreement_review": monitoring["disagreement_review"],
            "universal_scalar_champion_permitted": monitoring[
                "universal_scalar_champion_permitted"
            ],
        },
        "decision_boundary": parent["decision_boundary"],
        "succession_binding": {
            "parent_contract_sha256": parent_sha256,
            "round_ordinal": ordinal,
            "derived_before_round_generation": True,
            "earlier_round_lineages_added_to_exclusions": ordinal > 1,
        },
    }


def generate_shadow_succession(
    contract_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    contract = _load_parent_contract(contract_path)
    contract_sha256 = sha256_file(contract_path)
    out_dir.mkdir(parents=True, exist_ok=False)
    rounds_dir = out_dir / "rounds"
    rounds_dir.mkdir()
    public_dir = out_dir / "public_succession"
    public_dir.mkdir()

    surface = contract["succession_surface"]
    original_forbidden = {
        str(value) for value in surface["forbidden_prior_lineage_sha256"]
    }
    accumulated_forbidden = set(original_forbidden)
    observed_lineages: set[str] = set()
    round_records = []
    round_checks = []
    totals = {"pairs": 0, "arrays": 0, "reports": 0}

    for ordinal, round_spec in enumerate(surface["rounds"], start=1):
        round_id = str(round_spec["shadow_run_id"])
        round_dir = rounds_dir / round_id
        round_dir.mkdir()
        derived_contract_path = round_dir / "derived_contract.json"
        derived = _derived_round_contract(
            contract,
            parent_sha256=contract_sha256,
            round_spec=round_spec,
            forbidden_lineages=sorted(accumulated_forbidden),
            ordinal=ordinal,
        )
        dump_structured(derived_contract_path, derived)
        fixture_dir = round_dir / "fixture"
        generation = generate_shadow_fixture(
            derived_contract_path,
            fixture_dir,
        )
        manifest_path = fixture_dir / "public_shadow/shadow_manifest.json"
        manifest, pairs, arrays, reports = load_shadow_fixture(manifest_path)
        round_lineages = {
            str(row["lineage_group_sha256"]) for row in pairs
        }
        prior_overlap = round_lineages & accumulated_forbidden
        cross_round_overlap = round_lineages & observed_lineages
        checks = {
            "generation_passed": bool(generation["generation_passed"]),
            "prior_lineage_overlap_zero": not prior_overlap,
            "cross_round_lineage_overlap_zero": not cross_round_overlap,
            "public_pairs_target_free": all(
                "target" not in row and "authority" not in row
                for row in pairs
            ),
            "public_arrays_target_free": all(
                "target" not in row and "authority" not in row
                for row in arrays
            ),
            "reports_content_free": all(
                not row["contains_row_level_game_records"]
                and not row["contains_hidden_reasoning"]
                for row in reports
            ),
        }
        if not all(checks.values()):
            raise ValueError(f"shadow succession generation failed: {round_id}")
        counts = {
            "pairs": len(pairs),
            "arrays": len(arrays),
            "reports": len(reports),
        }
        for key, value in counts.items():
            totals[key] += value
        round_root_relative = round_dir.relative_to(out_dir)
        round_records.append(
            {
                "round_ordinal": ordinal,
                "shadow_run_id": round_id,
                "seed": int(round_spec["seed"]),
                "context_prefix": round_spec["context_prefix"],
                "array_sizes": list(round_spec["array_sizes"]),
                "derived_contract_path": (
                    round_root_relative / derived_contract_path.name
                ).as_posix(),
                "derived_contract_sha256": sha256_file(
                    derived_contract_path
                ),
                "shadow_manifest_path": (
                    round_root_relative
                    / "fixture/public_shadow/shadow_manifest.json"
                ).as_posix(),
                "shadow_manifest_sha256": sha256_file(manifest_path),
                "generation_report_sha256": sha256_file(
                    fixture_dir / "generation_report.json"
                ),
                "counts": counts,
                "lineage_group_count": len(round_lineages),
                "prior_lineage_exclusion_count": len(
                    accumulated_forbidden
                ),
                "checks": checks,
                "verifier_label_paths_disclosed": False,
            }
        )
        round_checks.append(checks)
        observed_lineages.update(round_lineages)
        accumulated_forbidden.update(round_lineages)

    succession_checks = {
        "all_rounds_generated": len(round_records)
        == len(surface["rounds"]),
        "all_round_checks_passed": all(
            all(check.values()) for check in round_checks
        ),
        "all_round_lineages_disjoint": sum(
            int(record["lineage_group_count"]) for record in round_records
        )
        == len(observed_lineages),
        "all_prior_lineages_excluded": not bool(
            observed_lineages & original_forbidden
        ),
        "public_aggregate_discloses_no_verifier_paths": all(
            not record["verifier_label_paths_disclosed"]
            for record in round_records
        ),
    }
    manifest = {
        "schema_version": "ldt-shadow-succession-fixture.v5",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": contract_sha256,
        "source_shadow": contract["source_shadow"],
        "frozen_model_binding": {
            key: contract["frozen_model"][key]
            for key in (
                "candidate_id",
                "seed",
                "trainable_parameters",
                "model_sha256",
                "trainer_summary_sha256",
            )
        },
        "rounds": round_records,
        "totals": totals,
        "prior_lineage_group_count": len(original_forbidden),
        "succession_lineage_group_count": len(observed_lineages),
        "checks": succession_checks,
        "verifier_label_paths_disclosed": False,
        "shadow_decision_effect": "NONE",
        "independent_verifier_remains_decision_owner": True,
        "training_authorized": False,
        "corpus_promotion_authorized": False,
        "adapter_selection_authorized": False,
    }
    manifest_path = public_dir / "succession_manifest.json"
    _write_json(manifest_path, manifest)
    report = {
        "schema_version": "ldt-shadow-succession-generation-report.v5",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": contract_sha256,
        "succession_manifest_sha256": sha256_file(manifest_path),
        "round_count": len(round_records),
        "totals": totals,
        "checks": succession_checks,
        "generation_passed": all(succession_checks.values()),
        "verifier_targets_exported_to_shadow_runner": False,
        "raw_game_records_read": False,
        "network_used": False,
        "claim_boundary": contract["claim_scope"],
    }
    _write_json(out_dir / "generation_report.json", report)
    return report


def _load_succession_manifest(
    contract_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    contract = _load_parent_contract(contract_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["schema_version"] != "ldt-shadow-succession-fixture.v5"
        or manifest["contract_sha256"] != sha256_file(contract_path)
        or not all(bool(value) for value in manifest["checks"].values())
        or manifest["verifier_label_paths_disclosed"]
        or manifest["shadow_decision_effect"] != "NONE"
        or manifest["training_authorized"]
        or manifest["corpus_promotion_authorized"]
        or manifest["adapter_selection_authorized"]
    ):
        raise ValueError("shadow succession manifest is invalid")
    expected_ids = [
        str(round_spec["shadow_run_id"])
        for round_spec in contract["succession_surface"]["rounds"]
    ]
    if [str(row["shadow_run_id"]) for row in manifest["rounds"]] != expected_ids:
        raise ValueError("shadow succession round order differs")
    fixture_root = manifest_path.parent.parent
    for row in manifest["rounds"]:
        derived_path = fixture_root / str(row["derived_contract_path"])
        shadow_manifest_path = (
            fixture_root / str(row["shadow_manifest_path"])
        )
        if (
            sha256_file(derived_path) != row["derived_contract_sha256"]
            or sha256_file(shadow_manifest_path)
            != row["shadow_manifest_sha256"]
        ):
            raise ValueError("shadow succession artifact binding differs")
        derived = load_structured(derived_path)
        binding = derived["succession_binding"]
        if (
            binding["parent_contract_sha256"]
            != manifest["contract_sha256"]
            or int(binding["round_ordinal"]) != int(row["round_ordinal"])
        ):
            raise ValueError("derived round contract binding differs")
    return contract, manifest, fixture_root


def run_shadow_succession(
    *,
    contract_path: Path,
    succession_manifest_path: Path,
    model_path: Path,
    trainer_summary_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    contract, manifest, fixture_root = _load_succession_manifest(
        contract_path,
        succession_manifest_path,
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    rounds_out = out_dir / "rounds"
    rounds_out.mkdir()
    round_receipts = []
    totals = {
        "pair_count": 0,
        "array_count": 0,
        "abstain_count": 0,
        "unresolved_candidate_count": 0,
    }
    for row in manifest["rounds"]:
        round_id = str(row["shadow_run_id"])
        round_out = rounds_out / round_id
        receipt = run_shadow(
            contract_path=fixture_root / str(row["derived_contract_path"]),
            manifest_path=fixture_root / str(row["shadow_manifest_path"]),
            model_path=model_path,
            trainer_summary_path=trainer_summary_path,
            out_dir=round_out,
        )
        if (
            receipt["verifier_pair_targets_seen"]
            or receipt["verifier_array_targets_seen"]
            or receipt["decision_effect"] != "NONE"
            or receipt["authority"] != "MODEL_RECOMMENDATION_ONLY"
        ):
            raise ValueError("shadow succession runner crossed custody boundary")
        for key in totals:
            totals[key] += int(receipt[key])
        receipt_path = round_out / "shadow_run_receipt.json"
        round_receipts.append(
            {
                "round_ordinal": row["round_ordinal"],
                "shadow_run_id": round_id,
                "shadow_manifest_sha256": row["shadow_manifest_sha256"],
                "shadow_run_receipt_path": (
                    Path("rounds") / round_id / receipt_path.name
                ).as_posix(),
                "shadow_run_receipt_sha256": sha256_file(receipt_path),
                "pair_recommendations_sha256": receipt[
                    "pair_recommendations_sha256"
                ],
                "array_recommendations_sha256": receipt[
                    "array_recommendations_sha256"
                ],
                "pair_count": receipt["pair_count"],
                "array_count": receipt["array_count"],
                "abstain_count": receipt["abstain_count"],
                "unresolved_candidate_count": receipt[
                    "unresolved_candidate_count"
                ],
            }
        )
    aggregate = {
        "schema_version": "ldt-shadow-succession-run-receipt.v5",
        "experiment_id": contract["experiment_id"],
        "status": "completed",
        "contract_sha256": sha256_file(contract_path),
        "succession_manifest_sha256": sha256_file(
            succession_manifest_path
        ),
        "model_sha256": contract["frozen_model"]["model_sha256"],
        "trainer_summary_sha256": contract["frozen_model"][
            "trainer_summary_sha256"
        ],
        "calibration_temperature": contract["frozen_model"]["calibration"][
            "temperature"
        ],
        "round_count": len(round_receipts),
        "rounds": round_receipts,
        "totals": totals,
        "verifier_pair_targets_seen": False,
        "verifier_array_targets_seen": False,
        "raw_game_records_read": False,
        "network_used": False,
        "model_weights_persisted_to_repository": False,
        "authority": "MODEL_RECOMMENDATION_ONLY",
        "decision_effect": "NONE",
        "training_authorized": False,
        "corpus_promotion_authorized": False,
        "adapter_selection_authorized": False,
        "claim_boundary": contract["claim_scope"],
    }
    _write_json(out_dir / "succession_run_receipt.json", aggregate)
    return aggregate


def _metric_range(
    values: Sequence[float],
) -> dict[str, float]:
    return {"minimum": min(values), "maximum": max(values)}


def aggregate_shadow_comparisons(
    contract: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_rounds = len(contract["succession_surface"]["rounds"])
    if len(comparisons) != expected_rounds:
        raise ValueError("shadow succession comparison count differs")
    games = [str(value) for value in contract["approved_input_scope"]["allowed_games"]]
    round_summaries = []
    for comparison in comparisons:
        if (
            comparison["schema_version"] != "ldt-shadow-comparison.v4"
            or int(comparison["verifier_join_count"]) != 1
            or comparison["shadow_decision_effect"] != "NONE"
            or comparison["verifier_decisions_mutated"]
            or comparison["training_from_disagreements_authorized"]
            or comparison["corpus_promotion_authorized"]
            or comparison["larger_model_training_authorized"]
        ):
            raise ValueError("round comparison violates succession boundary")
        round_summaries.append(
            {
                "shadow_run_id": comparison["shadow_run_id"],
                "comparison_revision": comparison["comparison_revision"],
                "monitoring_passed": comparison["monitoring_passed"],
                "pair_aggregate": comparison["pair_aggregate"],
                "array_aggregate": comparison["array_aggregate"],
                "pair_by_game": comparison["pair_by_game"],
                "array_by_game": comparison["array_by_game"],
                "array_by_size": comparison["array_by_size"],
                "review_queue_count": comparison["review_queue_count"],
                "verifier_join_count": comparison["verifier_join_count"],
            }
        )
    pair_aggregates = [row["pair_aggregate"] for row in round_summaries]
    array_aggregates = [row["array_aggregate"] for row in round_summaries]
    stability_by_game = {}
    for game in games:
        pair_slices = [row["pair_by_game"][game] for row in round_summaries]
        array_slices = [
            row["array_by_game"][game] for row in round_summaries
        ]
        stability_by_game[game] = {
            "preference_macro_f1": _metric_range(
                [float(row["preference_macro_f1"]) for row in pair_slices]
            ),
            "expected_calibration_error": _metric_range(
                [
                    float(row["expected_calibration_error"])
                    for row in pair_slices
                ]
            ),
            "exact_array_accuracy": _metric_range(
                [float(row["exact_array_accuracy"]) for row in array_slices]
            ),
            "selected_set_exact_accuracy": _metric_range(
                [
                    float(row["selected_set_exact_accuracy"])
                    for row in array_slices
                ]
            ),
        }
    all_rounds_passed = all(
        bool(row["monitoring_passed"]) for row in round_summaries
    )
    return {
        "round_count": len(round_summaries),
        "all_rounds_passed": all_rounds_passed,
        "verifier_join_count": sum(
            int(row["verifier_join_count"]) for row in round_summaries
        ),
        "pair_count": sum(
            int(row["pair_aggregate"]["count"]) for row in round_summaries
        ),
        "array_count": sum(
            int(row["array_aggregate"]["array_count"])
            for row in round_summaries
        ),
        "review_queue_count": sum(
            int(row["review_queue_count"]) for row in round_summaries
        ),
        "minimum_pair_preference_macro_f1": min(
            float(row["preference_macro_f1"]) for row in pair_aggregates
        ),
        "minimum_abstain_recall": min(
            float(row["abstain_recall"]) for row in pair_aggregates
        ),
        "maximum_expected_calibration_error": max(
            float(row["expected_calibration_error"])
            for row in pair_aggregates
        ),
        "maximum_multiclass_brier": max(
            float(row["multiclass_brier"]) for row in pair_aggregates
        ),
        "maximum_multiclass_negative_log_likelihood": max(
            float(row["multiclass_negative_log_likelihood"])
            for row in pair_aggregates
        ),
        "minimum_exact_array_accuracy": min(
            float(row["exact_array_accuracy"]) for row in array_aggregates
        ),
        "minimum_selected_set_exact_accuracy": min(
            float(row["selected_set_exact_accuracy"])
            for row in array_aggregates
        ),
        "stability_by_game": stability_by_game,
        "rounds": round_summaries,
    }


def evaluate_shadow_succession(
    *,
    contract_path: Path,
    succession_manifest_path: Path,
    shadow_run_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    contract, manifest, fixture_root = _load_succession_manifest(
        contract_path,
        succession_manifest_path,
    )
    aggregate_run_path = shadow_run_dir / "succession_run_receipt.json"
    aggregate_run = json.loads(
        aggregate_run_path.read_text(encoding="utf-8")
    )
    if (
        aggregate_run["status"] != "completed"
        or aggregate_run["contract_sha256"] != sha256_file(contract_path)
        or aggregate_run["succession_manifest_sha256"]
        != sha256_file(succession_manifest_path)
        or aggregate_run["verifier_pair_targets_seen"]
        or aggregate_run["verifier_array_targets_seen"]
        or aggregate_run["decision_effect"] != "NONE"
        or aggregate_run["authority"] != "MODEL_RECOMMENDATION_ONLY"
        or aggregate_run["training_authorized"]
        or aggregate_run["corpus_promotion_authorized"]
        or aggregate_run["adapter_selection_authorized"]
    ):
        raise ValueError("shadow succession run receipt is invalid")
    run_by_id = {
        str(row["shadow_run_id"]): row for row in aggregate_run["rounds"]
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    rounds_out = out_dir / "rounds"
    rounds_out.mkdir()
    comparisons = []
    comparison_bindings = []
    combined_review_rows = []
    for row in manifest["rounds"]:
        round_id = str(row["shadow_run_id"])
        if round_id not in run_by_id:
            raise ValueError("shadow succession run is missing a round")
        round_run_dir = shadow_run_dir / "rounds" / round_id
        run_receipt_path = round_run_dir / "shadow_run_receipt.json"
        run_binding = run_by_id[round_id]
        if (
            sha256_file(run_receipt_path)
            != run_binding["shadow_run_receipt_sha256"]
        ):
            raise ValueError("round run receipt hash differs")
        round_root = (
            fixture_root / str(row["derived_contract_path"])
        ).parent
        comparison_out = rounds_out / round_id
        comparison = evaluate_shadow(
            contract_path=round_root / "derived_contract.json",
            manifest_path=fixture_root / str(row["shadow_manifest_path"]),
            pair_recommendations_path=(
                round_run_dir / "shadow_pair_recommendations.jsonl"
            ),
            array_recommendations_path=(
                round_run_dir / "shadow_array_recommendations.jsonl"
            ),
            shadow_run_receipt_path=run_receipt_path,
            verifier_pair_labels_path=(
                round_root
                / "fixture/private_verifier/verifier_pair_labels.jsonl"
            ),
            verifier_array_labels_path=(
                round_root
                / "fixture/private_verifier/verifier_array_labels.jsonl"
            ),
            out_dir=comparison_out,
        )
        comparison_path = comparison_out / "shadow_comparison.json"
        review_path = comparison_out / "hash_only_review_queue.jsonl"
        comparisons.append(comparison)
        comparison_bindings.append(
            {
                "round_ordinal": row["round_ordinal"],
                "shadow_run_id": round_id,
                "comparison_revision": comparison["comparison_revision"],
                "shadow_comparison_sha256": sha256_file(comparison_path),
                "hash_only_review_queue_sha256": sha256_file(review_path),
                "verifier_join_count": comparison["verifier_join_count"],
            }
        )
        for review in _read_jsonl(review_path):
            if review["contains_verifier_label"]:
                raise ValueError("review queue exposed a verifier label")
            combined_review_rows.append(
                {
                    **review,
                    "shadow_run_id": round_id,
                }
            )
    summary = aggregate_shadow_comparisons(contract, comparisons)
    checks = {
        "expected_round_count": summary["round_count"]
        == len(contract["succession_surface"]["rounds"]),
        "one_verifier_join_per_round": summary["verifier_join_count"]
        == summary["round_count"],
        "every_round_monitoring_passed": summary["all_rounds_passed"],
        "review_queue_contains_no_verifier_labels": all(
            not row["contains_verifier_label"]
            for row in combined_review_rows
        ),
        "verifier_decisions_unmutated": all(
            not comparison["verifier_decisions_mutated"]
            for comparison in comparisons
        ),
        "model_binding_stable": all(
            comparison["model_sha256"]
            == contract["frozen_model"]["model_sha256"]
            and comparison["trainer_summary_sha256"]
            == contract["frozen_model"]["trainer_summary_sha256"]
            for comparison in comparisons
        ),
    }
    report = {
        "schema_version": "ldt-shadow-succession-comparison.v5",
        "comparison_revision": 1,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "succession_manifest_sha256": sha256_file(
            succession_manifest_path
        ),
        "succession_run_receipt_sha256": sha256_file(aggregate_run_path),
        "model_sha256": contract["frozen_model"]["model_sha256"],
        "trainer_summary_sha256": contract["frozen_model"][
            "trainer_summary_sha256"
        ],
        "comparison_bindings": comparison_bindings,
        "summary": summary,
        "checks": checks,
        "succession_passed": all(checks.values()),
        "aggregate_contains_row_level_verifier_labels": False,
        "review_queue_contains_verifier_labels": False,
        "shadow_authority": "MODEL_RECOMMENDATION_ONLY",
        "verifier_authority": "ENVIRONMENT_OWNED",
        "shadow_decision_effect": "NONE",
        "verifier_decisions_mutated": False,
        "training_from_disagreements_authorized": False,
        "corpus_promotion_authorized": False,
        "larger_model_training_authorized": False,
        "adapter_selection_authorized": False,
        "universal_scalar_champion": None,
        "claim_boundary": contract["claim_scope"],
    }
    report_path = out_dir / "shadow_succession_comparison.json"
    review_path = out_dir / "hash_only_succession_review_queue.jsonl"
    _write_json(report_path, report)
    _write_jsonl(
        review_path,
        sorted(
            combined_review_rows,
            key=lambda item: (
                str(item["shadow_run_id"]),
                str(item["review_key_sha256"]),
            ),
        ),
    )
    receipt = {
        "schema_version": "ldt-shadow-succession-comparison-receipt.v5",
        "shadow_succession_comparison_sha256": sha256_file(report_path),
        "comparison_revision": 1,
        "round_comparison_sha256": [
            row["shadow_comparison_sha256"]
            for row in comparison_bindings
        ],
        "verifier_join_count": summary["verifier_join_count"],
        "hash_only_review_queue_sha256": sha256_file(review_path),
        "review_queue_contains_verifier_labels": False,
        "aggregate_contains_row_level_verifier_labels": False,
        "shadow_decision_effect": "NONE",
        "verifier_decisions_mutated": False,
        "post_comparison_training_authorized": False,
        "adapter_selection_authorized": False,
    }
    _write_json(out_dir / "comparison_receipt.json", receipt)
    return report
