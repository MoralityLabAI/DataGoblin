from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datagoblin.adapter_verifier import (
    build_drift_benchmark,
    run_curation_skill_tournament_succession,
    run_exact_key_baseline,
)
from datagoblin.adapter_verifier import skill_tournament as succession_module
from datagoblin.serialization import dump_structured, load_structured


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_boundary_pressure_applies_precedence_and_cumulative_diagnostics(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    config = root / "configs/preregistrations/game_adapter_schema_drift_v1.yaml"
    out = tmp_path / "boundary-pressure"
    build_drift_benchmark(
        config,
        out,
        seed=601,
        pressure_mode="boundary_precedence",
    )
    manifest = out / "public_dataset/dataset_manifest.json"
    setup = out / "exact-key"
    run_exact_key_baseline(config, manifest, setup)
    predictions = {
        row["item_id"]: row for row in _jsonl(setup / "predictions.jsonl")
    }
    labels = {
        row["item_id"]: row
        for row in _jsonl(out / "private_steward/ground_truth.jsonl")
    }
    rows = {
        row["item_id"]: row
        for split in (
            "train",
            "calibration",
            "ordinary_test",
            "ood_test",
            "sealed_poison",
        )
        for row in _jsonl(out / "public_dataset" / f"{split}.jsonl")
    }

    def category_rows(category: str) -> list[str]:
        return [
            item_id
            for item_id, row in rows.items()
            if f"_hazard_{category}_" in row["transformation_family"]
        ]

    for item_id in category_rows("sealed_hidden_private"):
        assert predictions[item_id]["disposition"] == "SEALED_EVALUATION"
        assert predictions[item_id]["failure_modes"] == []
        assert predictions[item_id]["abstained"]
    for item_id in category_rows("hidden_private"):
        assert predictions[item_id]["disposition"] == "REJECT"
        assert set(predictions[item_id]["failure_modes"]) == {
            "hidden_reasoning",
            "private_data",
        }
        assert set(labels[item_id]["expected"]["failure_modes"]) == {
            "hidden_reasoning",
            "private_data",
        }
    for item_id in category_rows("private_unsupported"):
        assert predictions[item_id]["parsed"]
        assert predictions[item_id]["disposition"] == "QUARANTINE"
        assert predictions[item_id]["failure_modes"] == ["private_data"]
    for item_id in category_rows("hidden_unsupported"):
        assert predictions[item_id]["parsed"]
        assert predictions[item_id]["disposition"] == "REJECT"
        assert predictions[item_id]["failure_modes"] == ["hidden_reasoning"]


def test_succession_exports_all_taste_labels_without_heldout_targets(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    root = Path(__file__).parents[2]
    config = (
        root
        / "configs/preregistrations/curation_skill_tournament_succession_v1.yaml"
    )
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    dump_structured(
        before,
        {
            "schema_version": "skill-fingerprint.v1",
            "skill_name": "curate-game-agent-data",
            "skill_sha256": "b08756184391a73e748e478c7c0aa1c08152189a7d96977c980db417841fa31e",
        },
    )
    dump_structured(
        after,
        {
            "schema_version": "skill-fingerprint.v1",
            "skill_name": "curate-game-agent-data",
            "skill_sha256": "a" * 64,
        },
    )
    validator = tmp_path / "validator.py"
    validator.write_text("# test double; tournament runner is monkeypatched\n", encoding="utf-8")
    calls = 0

    def fake_tournament(
        config_path: Path,
        out_dir: Path,
        *,
        seed: int,
        fixture_pressure: str,
        candidate_validator_path: Path | None,
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        out_dir.mkdir(parents=True)

        def entry(
            context: str,
            parameters: int,
            ordinary: float,
            ood: float,
            profile_id: str | None,
        ) -> dict[str, Any]:
            def split_metrics(value: float) -> dict[str, float]:
                return {
                    "exact_record_accuracy": value,
                    "mapping_f1": value,
                    "privacy_leak_rate": 0.0,
                    "critical_false_release_rate": 0.0,
                }

            return {
                "profile_id": profile_id,
                "context_mode": context,
                "optimization_pressure": context,
                "trainable_parameters": parameters,
                "safety_qualified": True,
                "per_game": {
                    game: {
                        "ordinary_test": split_metrics(ordinary),
                        "ood_test": split_metrics(ood),
                        "sealed_poison": split_metrics(1.0),
                    }
                    for game in ("centauri", "they_sing", "court", "diplomacy")
                },
            }

        entries = {
            "baseline": entry("baseline", 0, 0.1, 0.1, None),
            "compact": entry("compact", 5888, 0.8, 0.8, "compact"),
            "coverage": entry("coverage", 11776, 0.8, 0.8, "coverage"),
            "normal": entry("normal", 23552, 0.9, 0.9, "normal"),
        }
        report = {
            "seed": seed,
            "fixture_pressure": fixture_pressure,
            "schema_surface_sha256": f"{calls:064x}",
            "qualifiers": ["compact", "coverage", "normal"],
            "entries": entries,
            "public_preflight": {
                "mode": "external_skill_validator",
                "all_candidates_valid": candidate_validator_path == validator,
            },
        }
        dump_structured(out_dir / "tournament_report.json", report)
        return report

    monkeypatch.setattr(succession_module, "run_adapter_tournament", fake_tournament)
    out = tmp_path / "succession"
    report = run_curation_skill_tournament_succession(
        config,
        before,
        after,
        validator,
        out,
    )

    assert report["succession_passed"]
    assert report["ldt_foundation"]["train_examples"] == 180
    assert report["ldt_foundation"]["heldout_examples"] == 180
    manifest = load_structured(out / "public_ldt_dataset/dataset_manifest.json")
    assert set(manifest["train_preference_distribution"]) == {
        "LEFT",
        "RIGHT",
        "TRADEOFF",
    }
    assert set(manifest["train_rationale_distribution"]) == {
        "safety_gate",
        "pareto_dominance",
        "parameter_efficiency",
        "irreducible_tradeoff",
    }
    heldout = _jsonl(out / "public_ldt_dataset/heldout.jsonl")
    assert heldout
    assert all("target" not in row and "authority" not in row for row in heldout)
    labels = _jsonl(out / "private_steward/heldout_labels.jsonl")
    assert len(labels) == len(heldout)
    assert manifest["lineage_audit"]["cross_split_groups"] == 0
