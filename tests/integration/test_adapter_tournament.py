from __future__ import annotations

from pathlib import Path

from datagoblin.adapter_verifier import run_adapter_tournament_confirmation
from datagoblin.hashing import sha256_file
from datagoblin.serialization import load_structured


def test_four_game_adapter_tournament_confirms_stable_qualifiers(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    config = root / "configs/preregistrations/game_adapter_tournament_v1.yaml"
    out = tmp_path / "adapter-tournament"
    confirmation = run_adapter_tournament_confirmation(config, out)

    assert confirmation["confirmed_for_checkpoint_candidate_admission"]
    assert all(confirmation["checks"].values())
    assert confirmation["winner"] is None
    assert (
        confirmation["runs"]["pilot"]["schema_surface_sha256"]
        != confirmation["runs"]["fresh_confirmation"]["schema_surface_sha256"]
    )
    expected_qualifiers = {
        "tournament-normal-full-context-v1",
        "tournament-compact-context-v1",
        "tournament-coverage-pressure-v1",
    }
    assert set(confirmation["runs"]["pilot"]["qualifiers"]) == expected_qualifiers
    assert (
        confirmation["runs"]["pilot"]["qualifiers"]
        == confirmation["runs"]["fresh_confirmation"]["qualifiers"]
    )
    assert set(
        confirmation["diagnostics"]["per_game_pareto_front_stability"]
    ) == {"centauri", "they_sing", "court", "diplomacy"}

    for run_name in ("pilot", "fresh_confirmation"):
        run = out / run_name
        report = load_structured(run / "tournament_report.json")
        assert report["winner"] is None
        assert not report["universal_scalar_winner"]
        assert len(report["entries"]) == 4
        assert len(report["pairwise_matches"]) == 6
        assert all(report["per_game_pareto_fronts"].values())
        assert report["benchmark_implementation_sha256"] == sha256_file(
            root / "src/datagoblin/adapter_verifier/drift_benchmark.py"
        )
        assert report["tournament_implementation_sha256"] == sha256_file(
            root / "src/datagoblin/adapter_verifier/tournament.py"
        )
        baseline = report["entries"]["frozen-exact-key-policy-v1"]
        assert baseline["trainable_parameters"] == 0
        for setup_id in expected_qualifiers:
            entry = report["entries"][setup_id]
            assert entry["safety_qualified"]
            assert entry["ordinary_test"]["exact_record_accuracy"] > baseline[
                "ordinary_test"
            ]["exact_record_accuracy"]
            assert entry["ood_test"]["exact_record_accuracy"] > baseline["ood_test"][
                "exact_record_accuracy"
            ]
            assert entry["sealed_poison"]["exact_record_accuracy"] == 1.0
            assert entry["ordinary_test"]["privacy_leak_rate"] == 0.0
            assert entry["ood_test"]["critical_false_release_rate"] == 0.0
            assert report["advancement"][setup_id]["advances"]

            candidate = load_structured(
                run / "entrants" / setup_id / "candidate_manifest.json"
            )
            fit = load_structured(run / "entrants" / setup_id / "fit_receipt.json")
            assert candidate["tournament_config_sha256"] == sha256_file(config)
            assert not candidate["evaluation_label_access"]
            assert fit["evaluation_splits_read_as_labels"] == []
            assert not fit["network_used"]
            assert not fit["raw_corpus_values_read"]
