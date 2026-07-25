from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_file, sha256_text
from datagoblin.serialization import dump_structured, load_structured

from .environment import GAMES
from .tournament import run_adapter_tournament


def _load_succession_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    config = load_structured(config_path)
    if config.get("schema_version") != "1.0":
        raise ValueError("unsupported curation-skill succession config version")
    rounds = config.get("rounds")
    if not isinstance(rounds, list) or len(rounds) < 6:
        raise ValueError("curation-skill succession requires at least six rounds")
    round_ids = [str(round_.get("round_id", "")) for round_ in rounds]
    if not all(round_ids) or len(set(round_ids)) != len(round_ids):
        raise ValueError("curation-skill succession round IDs must be nonempty and unique")
    if {str(round_["meta_split"]) for round_ in rounds} != {"train", "heldout"}:
        raise ValueError("curation-skill succession requires train and heldout meta splits")
    pressures_by_split = {
        split: {
            str(round_["fixture_pressure"])
            for round_ in rounds
            if round_["meta_split"] == split
        }
        for split in ("train", "heldout")
    }
    required_pressures = {"balanced", "boundary_precedence", "authority_provenance"}
    if any(pressures != required_pressures for pressures in pressures_by_split.values()):
        raise ValueError("each meta split must contain the three frozen pressure families")
    seeds = [int(round_["seed"]) for round_ in rounds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("curation-skill succession seeds must be unique")
    tournament_path = Path(str(config["tournament_config"]))
    if not tournament_path.is_absolute():
        tournament_path = config_path.resolve().parents[2] / tournament_path
    if not tournament_path.is_file():
        raise ValueError(f"curation-skill tournament config is missing: {tournament_path}")
    return config, tournament_path


def _load_skill_fingerprint(
    path: Path,
    *,
    skill_name: str,
) -> dict[str, Any]:
    value = load_structured(path)
    if value.get("schema_version") != "skill-fingerprint.v1":
        raise ValueError(f"unsupported skill fingerprint: {path}")
    if value.get("skill_name") != skill_name:
        raise ValueError(f"skill fingerprint name mismatch: {path}")
    digest = str(value.get("skill_sha256", ""))
    if len(digest) != 64:
        raise ValueError(f"invalid skill fingerprint hash: {path}")
    return value


def _parameter_bucket(parameters: int) -> str:
    if parameters == 0:
        return "zero"
    if parameters <= 8192:
        return "small"
    if parameters <= 16384:
        return "medium"
    return "large"


def _entry_features(
    entry: Mapping[str, Any],
    *,
    game: str,
    fixture_pressure: str,
) -> dict[str, Any]:
    ordinary = entry["per_game"][game]["ordinary_test"]
    ood = entry["per_game"][game]["ood_test"]
    return {
        "game": game,
        "fixture_pressure": fixture_pressure,
        "context_mode": entry["context_mode"],
        "optimization_pressure": entry["optimization_pressure"],
        "trainable_parameter_bucket": _parameter_bucket(
            int(entry["trainable_parameters"])
        ),
        "safety_qualified": bool(entry["safety_qualified"]),
        "ordinary_exact_record_accuracy": float(ordinary["exact_record_accuracy"]),
        "ordinary_mapping_f1": float(ordinary["mapping_f1"]),
        "ood_exact_record_accuracy": float(ood["exact_record_accuracy"]),
        "ood_mapping_f1": float(ood["mapping_f1"]),
    }


def _metric_vector(features: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(features["ordinary_exact_record_accuracy"]),
        float(features["ordinary_mapping_f1"]),
        float(features["ood_exact_record_accuracy"]),
        float(features["ood_mapping_f1"]),
    )


def _preference(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    tolerance: float,
) -> tuple[str, str]:
    left_safe = bool(left["safety_qualified"])
    right_safe = bool(right["safety_qualified"])
    if left_safe != right_safe:
        return ("LEFT" if left_safe else "RIGHT"), "safety_gate"
    left_metrics = _metric_vector(left)
    right_metrics = _metric_vector(right)
    left_dominates = all(
        left_value + tolerance >= right_value
        for left_value, right_value in zip(left_metrics, right_metrics, strict=True)
    ) and any(
        left_value > right_value + tolerance
        for left_value, right_value in zip(left_metrics, right_metrics, strict=True)
    )
    right_dominates = all(
        right_value + tolerance >= left_value
        for left_value, right_value in zip(left_metrics, right_metrics, strict=True)
    ) and any(
        right_value > left_value + tolerance
        for left_value, right_value in zip(left_metrics, right_metrics, strict=True)
    )
    if left_dominates != right_dominates:
        return ("LEFT" if left_dominates else "RIGHT"), "pareto_dominance"
    equivalent = all(
        abs(left_value - right_value) <= tolerance
        for left_value, right_value in zip(left_metrics, right_metrics, strict=True)
    )
    bucket_rank = {"zero": 0, "small": 1, "medium": 2, "large": 3}
    left_cost = bucket_rank[str(left["trainable_parameter_bucket"])]
    right_cost = bucket_rank[str(right["trainable_parameter_bucket"])]
    if equivalent and left_cost != right_cost:
        return ("LEFT" if left_cost < right_cost else "RIGHT"), "parameter_efficiency"
    return "TRADEOFF", "irreducible_tradeoff"


def _counterfactuals(
    features: Mapping[str, Any],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    unsafe = {
        **features,
        "context_mode": "counterfactual_unsafe_high_coverage",
        "optimization_pressure": "coverage_without_safety",
        "trainable_parameter_bucket": "zero",
        "safety_qualified": False,
        "ordinary_exact_record_accuracy": 1.0,
        "ordinary_mapping_f1": 1.0,
        "ood_exact_record_accuracy": 1.0,
        "ood_mapping_f1": 1.0,
    }
    zero_coverage = {
        **features,
        "context_mode": "counterfactual_safe_zero_coverage",
        "optimization_pressure": "safety_without_utility",
        "trainable_parameter_bucket": "zero",
        "safety_qualified": True,
        "ordinary_exact_record_accuracy": 0.0,
        "ordinary_mapping_f1": 0.0,
        "ood_exact_record_accuracy": 0.0,
        "ood_mapping_f1": 0.0,
    }
    return (
        ("unsafe_high_coverage", unsafe),
        ("safe_zero_coverage", zero_coverage),
    )


def _preference_rows(
    report: Mapping[str, Any],
    round_spec: Mapping[str, Any],
    *,
    tolerance: float,
    include_counterfactuals: bool,
) -> list[dict[str, Any]]:
    round_id = str(round_spec["round_id"])
    meta_split = str(round_spec["meta_split"])
    fixture_pressure = str(round_spec["fixture_pressure"])
    source_hash = str(report["_report_sha256"])
    lineage_hash = sha256_text(f"ldt-lineage:{round_id}:{source_hash}")
    rows: list[dict[str, Any]] = []
    entries = report["entries"]
    for game in GAMES:
        features = {
            setup_id: _entry_features(
                entry,
                game=game,
                fixture_pressure=fixture_pressure,
            )
            for setup_id, entry in entries.items()
        }
        for left_id, right_id in combinations(sorted(features), 2):
            preference, rationale = _preference(
                features[left_id],
                features[right_id],
                tolerance=tolerance,
            )
            identity = f"{round_id}:{game}:{left_id}:{right_id}"
            rows.append(
                {
                    "schema_version": "curation-taste-pair.v1",
                    "example_id": f"taste-{sha256_text(identity)[:20]}",
                    "meta_split": meta_split,
                    "lineage_group_sha256": lineage_hash,
                    "source_tournament_report_sha256": source_hash,
                    "source_kind": "observed_tournament_pair",
                    "input": {
                        "left": features[left_id],
                        "right": features[right_id],
                    },
                    "target": {
                        "preference": preference,
                        "rationale_code": rationale,
                    },
                    "authority": "ENVIRONMENT_OWNED",
                }
            )
        if include_counterfactuals:
            for setup_id, real in features.items():
                for control_id, control in _counterfactuals(real):
                    preference, rationale = _preference(
                        real,
                        control,
                        tolerance=tolerance,
                    )
                    identity = f"{round_id}:{game}:{setup_id}:{control_id}"
                    rows.append(
                        {
                            "schema_version": "curation-taste-pair.v1",
                            "example_id": f"taste-{sha256_text(identity)[:20]}",
                            "meta_split": meta_split,
                            "lineage_group_sha256": lineage_hash,
                            "source_tournament_report_sha256": source_hash,
                            "source_kind": "environment_counterfactual_control",
                            "input": {"left": real, "right": control},
                            "target": {
                                "preference": preference,
                                "rationale_code": rationale,
                            },
                            "authority": "ENVIRONMENT_OWNED",
                        }
                    )
            template = next(iter(features.values()))
            ordinary_specialist = {
                **template,
                "context_mode": "counterfactual_ordinary_specialist",
                "optimization_pressure": "ordinary_quality",
                "trainable_parameter_bucket": "medium",
                "safety_qualified": True,
                "ordinary_exact_record_accuracy": 1.0,
                "ordinary_mapping_f1": 1.0,
                "ood_exact_record_accuracy": 0.6,
                "ood_mapping_f1": 0.6,
            }
            robustness_specialist = {
                **template,
                "context_mode": "counterfactual_robustness_specialist",
                "optimization_pressure": "ood_robustness",
                "trainable_parameter_bucket": "medium",
                "safety_qualified": True,
                "ordinary_exact_record_accuracy": 0.6,
                "ordinary_mapping_f1": 0.6,
                "ood_exact_record_accuracy": 1.0,
                "ood_mapping_f1": 1.0,
            }
            preference, rationale = _preference(
                ordinary_specialist,
                robustness_specialist,
                tolerance=tolerance,
            )
            identity = f"{round_id}:{game}:ordinary-specialist:robustness-specialist"
            rows.append(
                {
                    "schema_version": "curation-taste-pair.v1",
                    "example_id": f"taste-{sha256_text(identity)[:20]}",
                    "meta_split": meta_split,
                    "lineage_group_sha256": lineage_hash,
                    "source_tournament_report_sha256": source_hash,
                    "source_kind": "environment_counterfactual_control",
                    "input": {
                        "left": ordinary_specialist,
                        "right": robustness_specialist,
                    },
                    "target": {
                        "preference": preference,
                        "rationale_code": rationale,
                    },
                    "authority": "ENVIRONMENT_OWNED",
                }
            )
    return sorted(rows, key=lambda row: str(row["example_id"]))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _render_succession(report: Mapping[str, Any]) -> str:
    lines = [
        "# Curation-skill tournament succession",
        "",
        "| Round | Meta split | Pressure | Qualifiers | Safety regression |",
        "|---|---|---|---:|---:|",
    ]
    for round_id, round_ in report["rounds"].items():
        lines.append(
            f"| {round_id} | {round_['meta_split']} | {round_['fixture_pressure']} | "
            f"{len(round_['qualifiers'])} | {round_['safety_regression']} |"
        )
    foundation = report["ldt_foundation"]
    lines.extend(
        (
            "",
            "## LDT foundation",
            "",
            f"Training pairs: `{foundation['train_examples']}`; held-out inputs: "
            f"`{foundation['heldout_examples']}`.",
            "",
            f"Training authorized for the synthetic meta-task: "
            f"`{foundation['ldt_training_permitted_for_synthetic_meta_task']}`.",
            "",
            "Held-out targets remain steward-only. No row-level game records or sealed metrics are features.",
            "",
            "## Skill mutation",
            "",
            f"Before: `{report['skill_mutation']['before_sha256']}`",
            "",
            f"After: `{report['skill_mutation']['after_sha256']}`",
            "",
            str(report["claim_boundary"]),
            "",
        )
    )
    return "\n".join(lines)


def run_curation_skill_tournament_succession(
    config_path: Path,
    skill_before_fingerprint_path: Path,
    skill_after_fingerprint_path: Path,
    candidate_validator_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Run pressure-varied tournaments and emit a split-safe LDT preference curriculum."""
    config, tournament_config = _load_succession_config(config_path)
    before = _load_skill_fingerprint(
        skill_before_fingerprint_path,
        skill_name=str(config["skill_name"]),
    )
    after = _load_skill_fingerprint(
        skill_after_fingerprint_path,
        skill_name=str(config["skill_name"]),
    )
    if before["skill_sha256"] != config["skill_before_sha256"]:
        raise ValueError("skill-before fingerprint does not match the frozen succession contract")
    if not candidate_validator_path.is_file():
        raise ValueError(
            f"curation-skill candidate validator is missing: {candidate_validator_path}"
        )
    out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, out_dir / "frozen_succession_config.yaml")
    round_reports: dict[str, dict[str, Any]] = {}
    preference_rows: list[dict[str, Any]] = []
    tolerance = float(config["taste_contract"]["metric_equivalence_tolerance"])
    include_counterfactuals = bool(
        config["ldt_foundation"]["include_environment_counterfactual_controls"]
    )
    for round_spec in config["rounds"]:
        round_id = str(round_spec["round_id"])
        report = run_adapter_tournament(
            tournament_config,
            out_dir / "rounds" / round_id,
            seed=int(round_spec["seed"]),
            fixture_pressure=str(round_spec["fixture_pressure"]),
            candidate_validator_path=candidate_validator_path,
        )
        report_path = out_dir / "rounds" / round_id / "tournament_report.json"
        report["_report_sha256"] = sha256_file(report_path)
        round_reports[round_id] = report
        preference_rows.extend(
            _preference_rows(
                report,
                round_spec,
                tolerance=tolerance,
                include_counterfactuals=include_counterfactuals,
            )
        )
    train_rows = [row for row in preference_rows if row["meta_split"] == "train"]
    heldout_rows = [row for row in preference_rows if row["meta_split"] == "heldout"]
    train_lineages = {str(row["lineage_group_sha256"]) for row in train_rows}
    heldout_lineages = {str(row["lineage_group_sha256"]) for row in heldout_rows}
    if train_lineages & heldout_lineages:
        raise ValueError("LDT foundation train/heldout lineage contamination")
    public_dir = out_dir / "public_ldt_dataset"
    steward_dir = out_dir / "private_steward"
    public_dir.mkdir()
    steward_dir.mkdir()
    train_path = public_dir / "train.jsonl"
    heldout_path = public_dir / "heldout.jsonl"
    heldout_labels_path = steward_dir / "heldout_labels.jsonl"
    _write_jsonl(train_path, train_rows)
    public_heldout = [
        {
            key: value
            for key, value in row.items()
            if key not in {"target", "authority"}
        }
        for row in heldout_rows
    ]
    _write_jsonl(heldout_path, public_heldout)
    _write_jsonl(
        heldout_labels_path,
        [
            {
                "schema_version": "curation-taste-heldout-label.v1",
                "example_id": row["example_id"],
                "lineage_group_sha256": row["lineage_group_sha256"],
                "target": row["target"],
                "authority": row["authority"],
            }
            for row in heldout_rows
        ],
    )
    train_labels = Counter(
        str(row["target"]["preference"]) for row in train_rows
    )
    train_rationales = Counter(
        str(row["target"]["rationale_code"]) for row in train_rows
    )
    manifest = {
        "schema_version": "curation-taste-ldt-foundation.v1",
        "succession_id": config["succession_id"],
        "task": config["ldt_foundation"]["task"],
        "feature_whitelist": config["ldt_foundation"]["public_feature_whitelist"],
        "target_fields": config["ldt_foundation"]["target_fields"],
        "splits": {
            "train": {
                "path": "train.jsonl",
                "sha256": sha256_file(train_path),
                "count": len(train_rows),
                "targets_in_file": True,
            },
            "heldout": {
                "path": "heldout.jsonl",
                "sha256": sha256_file(heldout_path),
                "count": len(public_heldout),
                "targets_in_file": False,
            },
        },
        "heldout_label_custody": {
            "path_disclosed_to_candidate": False,
            "sha256": sha256_file(heldout_labels_path),
            "authority": "ENVIRONMENT_OWNED",
        },
        "lineage_audit": {
            "train_groups": len(train_lineages),
            "heldout_groups": len(heldout_lineages),
            "cross_split_groups": 0,
        },
        "train_preference_distribution": dict(sorted(train_labels.items())),
        "train_rationale_distribution": dict(sorted(train_rationales.items())),
        "row_level_game_records_included": False,
        "sealed_metrics_in_features": False,
        "larger_model_corpus_training_authorized": False,
    }
    dump_structured(public_dir / "dataset_manifest.json", manifest)
    pressure_by_split = {
        split: {
            str(round_["fixture_pressure"])
            for round_ in config["rounds"]
            if round_["meta_split"] == split
        }
        for split in ("train", "heldout")
    }
    round_summaries = {
        str(round_spec["round_id"]): {
            "meta_split": round_spec["meta_split"],
            "fixture_pressure": round_spec["fixture_pressure"],
            "seed": round_spec["seed"],
            "schema_surface_sha256": round_reports[str(round_spec["round_id"])][
                "schema_surface_sha256"
            ],
            "tournament_report_sha256": round_reports[str(round_spec["round_id"])][
                "_report_sha256"
            ],
            "qualifiers": round_reports[str(round_spec["round_id"])]["qualifiers"],
            "safety_regression": any(
                not entry["safety_qualified"]
                for entry in round_reports[str(round_spec["round_id"])]["entries"].values()
                if entry["profile_id"] is not None
            ),
        }
        for round_spec in config["rounds"]
    }
    gates = config["succession_gates"]
    checks = {
        "all_rounds_have_qualifier": (
            not bool(gates["require_all_rounds_have_qualifier"])
            or all(summary["qualifiers"] for summary in round_summaries.values())
        ),
        "zero_safety_regressions": (
            not bool(gates["require_zero_safety_regressions"])
            or not any(summary["safety_regression"] for summary in round_summaries.values())
        ),
        "distinct_round_lineages": (
            not bool(gates["require_distinct_round_lineages"])
            or len(
                {
                    summary["schema_surface_sha256"]
                    for summary in round_summaries.values()
                }
            )
            == len(round_summaries)
        ),
        "pressure_matched_train_and_heldout": (
            not bool(gates["require_pressure_matched_train_and_heldout"])
            or pressure_by_split["train"] == pressure_by_split["heldout"]
        ),
        "skill_fingerprint_changed": (
            not bool(gates["require_skill_fingerprint_change"])
            or before["skill_sha256"] != after["skill_sha256"]
        ),
        "heldout_targets_separate": (
            not bool(gates["require_heldout_targets_separate"])
            or all("target" not in row for row in public_heldout)
        ),
        "all_preference_labels_present": (
            not bool(gates["require_all_preference_labels"])
            or set(train_labels) == set(config["taste_contract"]["preference_labels"])
        ),
        "all_rationale_codes_present": (
            not bool(gates["require_all_rationale_codes"])
            or set(train_rationales)
            == set(config["taste_contract"]["rationale_codes"])
        ),
        "external_candidate_preflight": (
            not bool(gates["require_external_candidate_preflight"])
            or all(
                round_reports[str(round_spec["round_id"])]["public_preflight"][
                    "mode"
                ]
                == "external_skill_validator"
                and round_reports[str(round_spec["round_id"])]["public_preflight"][
                    "all_candidates_valid"
                ]
                for round_spec in config["rounds"]
            )
        ),
    }
    report = {
        "schema_version": "curation-skill-tournament-succession-report.v1",
        "succession_id": config["succession_id"],
        "succession_config_sha256": sha256_file(config_path),
        "succession_implementation_sha256": sha256_file(Path(__file__)),
        "tournament_config_sha256": sha256_file(tournament_config),
        "candidate_validator_sha256": sha256_file(candidate_validator_path),
        "rounds": round_summaries,
        "skill_mutation": {
            "before_sha256": before["skill_sha256"],
            "after_sha256": after["skill_sha256"],
            "changed": before["skill_sha256"] != after["skill_sha256"],
            "hypothesis": config["mutation_hypothesis"],
        },
        "ldt_foundation": {
            "dataset_manifest_sha256": sha256_file(
                public_dir / "dataset_manifest.json"
            ),
            "train_examples": len(train_rows),
            "heldout_examples": len(public_heldout),
            "ldt_training_permitted_for_synthetic_meta_task": all(checks.values()),
            "larger_model_corpus_training_authorized": False,
        },
        "checks": checks,
        "succession_passed": all(checks.values()),
        "claim_boundary": config["claim_scope"],
    }
    dump_structured(out_dir / "succession_report.json", report)
    (out_dir / "succession_report.md").write_text(
        _render_succession(report),
        encoding="utf-8",
    )
    dump_structured(
        out_dir / "succession_receipt.json",
        {
            "schema_version": "curation-skill-succession-receipt.v1",
            "succession_report_sha256": sha256_file(
                out_dir / "succession_report.json"
            ),
            "ldt_dataset_manifest_sha256": sha256_file(
                public_dir / "dataset_manifest.json"
            ),
            "skill_before_fingerprint_sha256": sha256_file(
                skill_before_fingerprint_path
            ),
            "skill_after_fingerprint_sha256": sha256_file(
                skill_after_fingerprint_path
            ),
            "network_used": False,
            "raw_corpus_values_read": False,
            "row_level_steward_labels_exported": False,
            "generated_artifacts_must_not_be_committed": True,
        },
    )
    return report
