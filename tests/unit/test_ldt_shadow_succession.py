from __future__ import annotations

import json
from pathlib import Path

from datagoblin.experiments.ldt_shadow import load_shadow_fixture
from datagoblin.experiments.ldt_shadow_succession import (
    aggregate_shadow_comparisons,
    generate_shadow_succession,
)
from datagoblin.serialization import load_structured


def test_succession_generator_is_disjoint_target_free_and_append_only(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    fixture_root = tmp_path / "fixture"
    report = generate_shadow_succession(
        root
        / "configs/preregistrations/ldt_shadow_succession_v5.yaml",
        fixture_root,
    )
    assert report["generation_passed"]
    assert report["round_count"] == 3
    assert report["totals"] == {
        "pairs": 744,
        "arrays": 36,
        "reports": 180,
    }

    manifest_path = (
        fixture_root / "public_succession/succession_manifest.json"
    )
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert "private_verifier" not in manifest_text
    assert manifest["checks"]["all_round_lineages_disjoint"]
    assert manifest["checks"]["all_prior_lineages_excluded"]
    assert not manifest["training_authorized"]
    assert not manifest["adapter_selection_authorized"]

    observed: set[str] = set()
    previous_lineages: set[str] = set()
    for index, round_record in enumerate(manifest["rounds"]):
        round_contract = load_structured(
            fixture_root / round_record["derived_contract_path"]
        )
        exclusions = set(
            round_contract["fresh_shadow_surface"][
                "forbidden_prior_lineage_sha256"
            ]
        )
        assert previous_lineages <= exclusions
        shadow_manifest = (
            fixture_root / round_record["shadow_manifest_path"]
        )
        _, pairs, arrays, reports = load_shadow_fixture(shadow_manifest)
        lineages = {
            str(row["lineage_group_sha256"]) for row in pairs
        }
        assert not lineages & observed
        assert all("target" not in row for row in pairs + arrays)
        assert all(
            not row["contains_row_level_game_records"]
            and not row["contains_hidden_reasoning"]
            for row in reports
        )
        if index:
            assert round_contract["succession_binding"][
                "earlier_round_lineages_added_to_exclusions"
            ]
        previous_lineages.update(lineages)
        observed.update(lineages)


def _comparison(round_id: str, ece: float) -> dict[str, object]:
    pair = {
        "count": 248,
        "preference_macro_f1": 1.0,
        "abstain_recall": 1.0,
        "expected_calibration_error": ece,
        "multiclass_brier": ece / 10,
        "multiclass_negative_log_likelihood": ece * 1.1,
    }
    array = {
        "array_count": 12,
        "exact_array_accuracy": 1.0,
        "selected_set_exact_accuracy": 1.0,
    }
    games = ("centauri", "they_sing", "court", "diplomacy")
    pair_by_game = {
        game: {
            "preference_macro_f1": 1.0,
            "expected_calibration_error": ece,
        }
        for game in games
    }
    array_by_game = {
        game: {
            "exact_array_accuracy": 1.0,
            "selected_set_exact_accuracy": 1.0,
        }
        for game in games
    }
    return {
        "schema_version": "ldt-shadow-comparison.v4",
        "comparison_revision": 1,
        "shadow_run_id": round_id,
        "verifier_join_count": 1,
        "shadow_decision_effect": "NONE",
        "verifier_decisions_mutated": False,
        "training_from_disagreements_authorized": False,
        "corpus_promotion_authorized": False,
        "larger_model_training_authorized": False,
        "monitoring_passed": True,
        "pair_aggregate": pair,
        "array_aggregate": array,
        "pair_by_game": pair_by_game,
        "array_by_game": array_by_game,
        "array_by_size": {},
        "review_queue_count": 0,
    }


def test_comparison_aggregation_preserves_rounds_and_worst_case_metrics() -> None:
    root = Path(__file__).resolve().parents[2]
    contract = load_structured(
        root / "configs/preregistrations/ldt_shadow_succession_v5.yaml"
    )
    comparisons = [
        _comparison("ldt-shadow-v5-round-2", 0.02),
        _comparison("ldt-shadow-v5-round-3", 0.04),
        _comparison("ldt-shadow-v5-round-4", 0.03),
    ]
    summary = aggregate_shadow_comparisons(contract, comparisons)
    assert summary["all_rounds_passed"]
    assert summary["verifier_join_count"] == 3
    assert summary["pair_count"] == 744
    assert summary["array_count"] == 36
    assert summary["maximum_expected_calibration_error"] == 0.04
    assert summary["minimum_pair_preference_macro_f1"] == 1.0
    assert summary["stability_by_game"]["diplomacy"][
        "expected_calibration_error"
    ] == {"minimum": 0.02, "maximum": 0.04}
