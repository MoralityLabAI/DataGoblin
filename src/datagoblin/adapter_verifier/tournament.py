from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured, load_structured

from .drift_benchmark import (
    EVALUATION_SPLITS,
    build_drift_benchmark,
    evaluate_drift_setup,
    fit_drift_adapter,
    run_exact_key_baseline,
)
from .environment import GAMES


def _load_tournament_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    config = load_structured(config_path)
    if config.get("schema_version") != "1.0":
        raise ValueError("unsupported adapter tournament config version")
    if tuple(config.get("games", ())) != GAMES:
        raise ValueError("adapter tournament must declare the four games in canonical order")
    if config.get("event_metrics", {}).get("universal_scalar_winner") is not False:
        raise ValueError("adapter tournament forbids a universal scalar winner")
    profiles = config.get("profiles")
    if not isinstance(profiles, list) or len(profiles) < 3:
        raise ValueError("adapter tournament requires normal, compact, and coverage profiles")
    profile_ids = [str(profile.get("profile_id", "")) for profile in profiles]
    setup_ids = [str(profile.get("setup_id", "")) for profile in profiles]
    if not all(profile_ids) or len(set(profile_ids)) != len(profile_ids):
        raise ValueError("adapter tournament profile IDs must be nonempty and unique")
    if not all(setup_ids) or len(set(setup_ids)) != len(setup_ids):
        raise ValueError("adapter tournament setup IDs must be nonempty and unique")
    modes = {str(profile.get("context_mode", "")) for profile in profiles}
    required_modes = {"normal_full_context", "compact_context", "coverage_pressure"}
    if not required_modes <= modes:
        raise ValueError("adapter tournament is missing a required context mode")
    benchmark_path = Path(str(config["benchmark_config"]))
    if not benchmark_path.is_absolute():
        benchmark_path = config_path.resolve().parents[2] / benchmark_path
    if not benchmark_path.is_file():
        raise ValueError(f"adapter tournament benchmark config is missing: {benchmark_path}")
    return config, benchmark_path


def _safety_qualified(report: Mapping[str, Any], gates: Mapping[str, Any]) -> bool:
    overall = report["evaluation_overall"]
    sealed = report["split_results"]["sealed_poison"]
    return bool(
        overall["privacy_leak_rate"] <= float(gates["maximum_privacy_leak_rate"])
        and overall["critical_false_release_rate"]
        <= float(gates["maximum_critical_false_release_rate"])
        and sealed["exact_record_accuracy"]
        >= float(gates["minimum_sealed_exact_record_accuracy"])
        and report["custody_checks"]["evaluation_labels_declared_unseen"]
    )


def _entry(
    report: Mapping[str, Any],
    gates: Mapping[str, Any],
    *,
    profile: Mapping[str, Any] | None,
    trainable_parameters: int,
) -> dict[str, Any]:
    return {
        "setup_id": report["setup_id"],
        "adapter_family": report["adapter_family"],
        "profile_id": profile["profile_id"] if profile is not None else None,
        "context_mode": profile["context_mode"] if profile is not None else "frozen_baseline",
        "optimization_pressure": (
            profile["optimization_pressure"] if profile is not None else "exact_key_reference"
        ),
        "trainable_parameters": trainable_parameters,
        "safety_qualified": _safety_qualified(report, gates),
        "ordinary_test": report["split_results"]["ordinary_test"],
        "ood_test": report["split_results"]["ood_test"],
        "sealed_poison": report["split_results"]["sealed_poison"],
        "per_game": {
            game: {
                split: report["game_split_results"][game][split]
                for split in EVALUATION_SPLITS
            }
            for game in GAMES
        },
        "evaluation_report_sha256": report["_report_sha256"],
    }


def _advancement_checks(
    entry: Mapping[str, Any],
    baseline: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    ordinary_gain = (
        entry["ordinary_test"]["exact_record_accuracy"]
        - baseline["ordinary_test"]["exact_record_accuracy"]
    )
    ood_gain = (
        entry["ood_test"]["exact_record_accuracy"]
        - baseline["ood_test"]["exact_record_accuracy"]
    )
    per_game = {
        game: {
            "ordinary_exact_gain": (
                entry["per_game"][game]["ordinary_test"]["exact_record_accuracy"]
                - baseline["per_game"][game]["ordinary_test"]["exact_record_accuracy"]
            ),
            "ood_exact_gain": (
                entry["per_game"][game]["ood_test"]["exact_record_accuracy"]
                - baseline["per_game"][game]["ood_test"]["exact_record_accuracy"]
            ),
        }
        for game in GAMES
    }
    checks = {
        "safety_qualified": bool(entry["safety_qualified"]),
        "ordinary_gain": ordinary_gain
        >= float(gates["minimum_ordinary_exact_gain_over_baseline"]),
        "ood_gain": ood_gain >= float(gates["minimum_ood_exact_gain_over_baseline"]),
        "nonnegative_per_game_ordinary_gain": (
            not bool(gates["require_nonnegative_per_game_ordinary_gain"])
            or all(value["ordinary_exact_gain"] >= 0.0 for value in per_game.values())
        ),
        "nonnegative_per_game_ood_gain": (
            not bool(gates["require_nonnegative_per_game_ood_gain"])
            or all(value["ood_exact_gain"] >= 0.0 for value in per_game.values())
        ),
    }
    deltas = {
        "ordinary_exact_gain": ordinary_gain,
        "ood_exact_gain": ood_gain,
        "per_game": per_game,
    }
    return checks, deltas


def _event_vector(entry: Mapping[str, Any], game: str) -> tuple[float, ...]:
    ordinary = entry["per_game"][game]["ordinary_test"]
    ood = entry["per_game"][game]["ood_test"]
    return (
        float(ordinary["exact_record_accuracy"]),
        float(ordinary["mapping_f1"]),
        float(ood["exact_record_accuracy"]),
        float(ood["mapping_f1"]),
    )


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a >= b for a, b in zip(left, right, strict=True)) and any(
        a > b for a, b in zip(left, right, strict=True)
    )


def _pareto_fronts(entries: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    qualified = {
        setup_id: entry
        for setup_id, entry in entries.items()
        if entry["profile_id"] is not None and entry["safety_qualified"]
    }
    fronts: dict[str, list[str]] = {}
    for game in GAMES:
        fronts[game] = sorted(
            setup_id
            for setup_id, entry in qualified.items()
            if not any(
                other_id != setup_id
                and _dominates(_event_vector(other, game), _event_vector(entry, game))
                for other_id, other in qualified.items()
            )
        )
    return fronts


def _pairwise_matches(entries: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    setup_ids = sorted(entries)
    matches: list[dict[str, Any]] = []
    for left_index, left_id in enumerate(setup_ids):
        for right_id in setup_ids[left_index + 1 :]:
            per_game: dict[str, str] = {}
            for game in GAMES:
                left = _event_vector(entries[left_id], game)
                right = _event_vector(entries[right_id], game)
                if _dominates(left, right):
                    outcome = "LEFT_DOMINATES"
                elif _dominates(right, left):
                    outcome = "RIGHT_DOMINATES"
                elif left == right:
                    outcome = "TIE"
                else:
                    outcome = "TRADEOFF"
                per_game[game] = outcome
            matches.append(
                {
                    "left_setup_id": left_id,
                    "right_setup_id": right_id,
                    "per_game_outcome": per_game,
                }
            )
    return matches


def _render_tournament(report: Mapping[str, Any]) -> str:
    lines = [
        "# Four-game adapter tournament",
        "",
        f"Seed: `{report['seed']}`; dataset: `{report['dataset_manifest_sha256'][:12]}`.",
        "",
        "| Entrant | Context | Parameters | Safe | Ordinary exact | OOD exact | Sealed exact | Advances |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for setup_id, entry in report["entries"].items():
        advances = report["advancement"].get(setup_id, {}).get("advances", False)
        lines.append(
            f"| {setup_id} | {entry['context_mode']} | {entry['trainable_parameters']} | "
            f"{entry['safety_qualified']} | {entry['ordinary_test']['exact_record_accuracy']:.3f} | "
            f"{entry['ood_test']['exact_record_accuracy']:.3f} | "
            f"{entry['sealed_poison']['exact_record_accuracy']:.3f} | {advances} |"
        )
    lines.extend(("", "## Per-game Pareto fronts", ""))
    for game, setup_ids in report["per_game_pareto_fronts"].items():
        lines.append(f"- `{game}`: {', '.join(f'`{setup_id}`' for setup_id in setup_ids) or 'none'}")
    lines.extend(
        (
            "",
            "No universal scalar winner is computed. Advancement is qualification for the next "
            "adapter gate only; it is not corpus-release or gameplay authority.",
            "",
            "This stage measures schema normalization, not game-playing ability.",
            "",
        )
    )
    return "\n".join(lines)


def run_adapter_tournament(
    config_path: Path,
    out_dir: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    """Run normal, compact, and coverage-pressure profiles on one frozen four-game fixture."""
    config, benchmark_config = _load_tournament_config(config_path)
    out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, out_dir / "frozen_tournament_config.yaml")
    benchmark_root = out_dir / "benchmark"
    build_drift_benchmark(benchmark_config, benchmark_root, seed=seed)
    manifest_path = benchmark_root / "public_dataset" / "dataset_manifest.json"
    steward_path = benchmark_root / "private_steward" / "ground_truth.jsonl"
    dataset_manifest = load_structured(manifest_path)
    tournament_hash = sha256_file(config_path)
    baseline_setup_id = str(config["baseline"]["setup_id"])
    baseline_dir = out_dir / "entrants" / baseline_setup_id
    run_exact_key_baseline(
        benchmark_config,
        manifest_path,
        baseline_dir,
        setup_id=baseline_setup_id,
    )
    baseline_report_path = out_dir / "evaluations" / baseline_setup_id
    baseline_report = evaluate_drift_setup(
        benchmark_config,
        manifest_path,
        steward_path,
        baseline_dir / "candidate_manifest.json",
        baseline_dir / "predictions.jsonl",
        baseline_report_path,
    )
    baseline_report["_report_sha256"] = sha256_file(
        baseline_report_path / "evaluation_report.json"
    )
    reports: dict[str, dict[str, Any]] = {baseline_setup_id: baseline_report}
    fit_receipts: dict[str, dict[str, Any]] = {}
    profiles_by_setup: dict[str, dict[str, Any]] = {}
    for profile in config["profiles"]:
        setup_id = str(profile["setup_id"])
        profile_id = str(profile["profile_id"])
        entrant_dir = out_dir / "entrants" / setup_id
        fit_drift_adapter(
            benchmark_config,
            manifest_path,
            entrant_dir,
            setup_id=setup_id,
            candidate_profile=profile["candidate"],
            profile_id=profile_id,
            tournament_config_sha256=tournament_hash,
        )
        candidate_manifest = load_structured(entrant_dir / "candidate_manifest.json")
        if candidate_manifest.get("profile_id") != profile_id:
            raise ValueError(f"adapter tournament profile binding mismatch: {setup_id}")
        if candidate_manifest.get("tournament_config_sha256") != tournament_hash:
            raise ValueError(f"adapter tournament config binding mismatch: {setup_id}")
        fit_receipt = load_structured(entrant_dir / "fit_receipt.json")
        for key, value in profile["candidate"].items():
            if fit_receipt["candidate_profile"].get(key) != value:
                raise ValueError(f"adapter tournament fit profile drift: {setup_id}/{key}")
        evaluation_dir = out_dir / "evaluations" / setup_id
        report = evaluate_drift_setup(
            benchmark_config,
            manifest_path,
            steward_path,
            entrant_dir / "candidate_manifest.json",
            entrant_dir / "predictions.jsonl",
            evaluation_dir,
        )
        report["_report_sha256"] = sha256_file(evaluation_dir / "evaluation_report.json")
        reports[setup_id] = report
        fit_receipts[setup_id] = fit_receipt
        profiles_by_setup[setup_id] = profile
    gates = config["advancement_gates"]
    baseline_entry = _entry(
        baseline_report,
        gates,
        profile=None,
        trainable_parameters=0,
    )
    entries: dict[str, dict[str, Any]] = {baseline_setup_id: baseline_entry}
    for setup_id, profile in profiles_by_setup.items():
        entries[setup_id] = _entry(
            reports[setup_id],
            gates,
            profile=profile,
            trainable_parameters=int(fit_receipts[setup_id]["trainable_parameters"]),
        )
    advancement: dict[str, dict[str, Any]] = {}
    for setup_id in profiles_by_setup:
        checks, deltas = _advancement_checks(entries[setup_id], baseline_entry, gates)
        advancement[setup_id] = {
            "advances": all(checks.values()),
            "checks": checks,
            "deltas": deltas,
            "authority": gates["advancement_authority"],
        }
    report = {
        "schema_version": "game-adapter-tournament-report.v1",
        "tournament_id": config["tournament_id"],
        "seed": seed,
        "tournament_config_sha256": tournament_hash,
        "tournament_implementation_sha256": sha256_file(Path(__file__)),
        "benchmark_config_sha256": sha256_file(benchmark_config),
        "benchmark_implementation_sha256": dataset_manifest[
            "generator_implementation_sha256"
        ],
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "schema_surface_sha256": dataset_manifest["schema_surface_sha256"],
        "entries": entries,
        "advancement": advancement,
        "qualifiers": sorted(
            setup_id
            for setup_id, value in advancement.items()
            if value["advances"]
        ),
        "per_game_pareto_fronts": _pareto_fronts(entries),
        "pairwise_matches": _pairwise_matches(entries),
        "winner": None,
        "universal_scalar_winner": False,
        "future_gameplay_arena": config["future_gameplay_arena"],
        "claim_boundary": config["claim_scope"],
    }
    dump_structured(out_dir / "tournament_report.json", report)
    (out_dir / "tournament_report.md").write_text(
        _render_tournament(report),
        encoding="utf-8",
    )
    dump_structured(
        out_dir / "tournament_run_receipt.json",
        {
            "schema_version": "game-adapter-tournament-run-receipt.v1",
            "tournament_config_sha256": tournament_hash,
            "tournament_report_sha256": sha256_file(out_dir / "tournament_report.json"),
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "schema_surface_sha256": dataset_manifest["schema_surface_sha256"],
            "network_used": False,
            "raw_corpus_values_read": False,
            "evaluation_labels_entered_fit_process": False,
            "generated_artifacts_must_not_be_committed": True,
        },
    )
    return report


def run_adapter_tournament_confirmation(
    config_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Run the frozen tournament on pilot and different-surface confirmation seeds."""
    config, _ = _load_tournament_config(config_path)
    out_dir.mkdir(parents=True, exist_ok=False)
    seeds = {
        "pilot": int(config["confirmation"]["pilot_seed"]),
        "fresh_confirmation": int(config["confirmation"]["fresh_confirmation_seed"]),
    }
    reports = {
        name: run_adapter_tournament(config_path, out_dir / name, seed=seed)
        for name, seed in seeds.items()
    }
    config_hashes = {
        str(report["tournament_config_sha256"]) for report in reports.values()
    }
    surfaces = {str(report["schema_surface_sha256"]) for report in reports.values()}
    qualifier_sets = {
        tuple(str(value) for value in report["qualifiers"])
        for report in reports.values()
    }
    pareto_front_stability = {
        game: len(
            {
                tuple(report["per_game_pareto_fronts"][game])
                for report in reports.values()
            }
        )
        == 1
        for game in GAMES
    }
    checks = {
        "tournament_config_unchanged": (
            not bool(config["confirmation"]["tournament_config_must_be_unchanged"])
            or len(config_hashes) == 1
        ),
        "schema_surface_differs": (
            not bool(config["confirmation"]["schema_surface_must_differ"])
            or len(surfaces) == len(reports)
        ),
        "qualifier_set_stable": (
            not bool(config["confirmation"]["qualifier_set_must_be_stable"])
            or len(qualifier_sets) == 1
        ),
        "at_least_one_profile_qualified": (
            not bool(config["confirmation"]["at_least_one_profile_must_qualify"])
            or all(bool(report["qualifiers"]) for report in reports.values())
        ),
    }
    confirmation = {
        "schema_version": "game-adapter-tournament-confirmation.v1",
        "tournament_id": config["tournament_id"],
        "tournament_config_sha256": sha256_file(config_path),
        "runs": {
            name: {
                "seed": report["seed"],
                "schema_surface_sha256": report["schema_surface_sha256"],
                "tournament_report_sha256": sha256_file(
                    out_dir / name / "tournament_report.json"
                ),
                "qualifiers": report["qualifiers"],
                "per_game_pareto_fronts": report["per_game_pareto_fronts"],
            }
            for name, report in reports.items()
        },
        "checks": checks,
        "diagnostics": {
            "per_game_pareto_front_stability": pareto_front_stability,
            "all_per_game_pareto_fronts_stable": all(pareto_front_stability.values()),
            "interpretation": (
                "Qualifier stability is the gate. Pareto-front movement across fresh schema "
                "surfaces is reported as robustness evidence, not hidden by an aggregate rank."
            ),
        },
        "confirmed_for_checkpoint_candidate_admission": all(checks.values()),
        "winner": None,
        "claim_boundary": config["claim_scope"],
    }
    dump_structured(out_dir / "tournament_confirmation.json", confirmation)
    return confirmation
