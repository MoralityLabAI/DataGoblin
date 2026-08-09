from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.experiments.ldt_array_confirmation import (
    derive_confirmation_array_target,
)
from datagoblin.experiments.ldt_edge_set_array_v10 import (
    EdgeDifferenceCandidateEncoder,
)
from datagoblin.experiments.ldt_exact_array_likelihood_v12 import (
    ExactArrayLikelihoodModel,
    predict_exact_array_statuses,
)
from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured, load_structured

from .architecture_models import (
    calibrate_mapping_threshold,
    finalize_model_artifact,
    train_lora_like_mapper,
    train_recurrent_readout_mapper,
    write_architecture_predictions,
)
from .drift_benchmark import (
    DRIFT_SPLITS,
    EVALUATION_SPLITS,
    GAMES,
    _aggregate,
    _load_contract,
    _read_dataset,
    _read_predictions,
    _read_steward_labels,
    _score,
    _training_samples,
    build_drift_benchmark,
)

LANES = ("development", "heldout")
METRIC_FIELDS = (
    "ordinary_exact_record_accuracy",
    "ordinary_mapping_f1",
    "ood_exact_record_accuracy",
    "ood_mapping_f1",
)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path.name}:{line_number}")
            values.append(value)
    return values


def _resolve_from_repo(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return config_path.resolve().parents[2] / path


def load_architecture_tournament_contract(config_path: Path) -> dict[str, Any]:
    config = load_structured(config_path)
    if config.get("schema_version") != "2.0":
        raise ValueError("unsupported architecture tournament schema version")
    if config.get("status") != "frozen_before_dataset_generation":
        raise ValueError("architecture tournament is not frozen before generation")
    if tuple(config.get("games", ())) != GAMES:
        raise ValueError("architecture tournament must declare the four games in canonical order")
    fixtures = config.get("fixture_plan", {})
    for lane in LANES:
        surfaces = fixtures.get(lane)
        if not isinstance(surfaces, list) or len(surfaces) != 3:
            raise ValueError(f"architecture tournament requires three {lane} surfaces")
        ids = [str(row.get("surface_id", "")) for row in surfaces]
        seeds = [int(row.get("generator_seed", -1)) for row in surfaces]
        pressures = [str(row.get("pressure", "")) for row in surfaces]
        if not all(ids) or len(set(ids)) != len(ids):
            raise ValueError(f"architecture tournament {lane} surface IDs must be unique")
        if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f"architecture tournament {lane} seeds must be unique")
        if set(pressures) != {"balanced", "boundary_precedence", "authority_provenance"}:
            raise ValueError(f"architecture tournament {lane} pressures are incomplete")
    dev_seeds = {int(row["generator_seed"]) for row in fixtures["development"]}
    heldout_seeds = {int(row["generator_seed"]) for row in fixtures["heldout"]}
    forbidden = {int(seed) for seed in fixtures["forbidden_prior_generator_seeds"]}
    if dev_seeds & heldout_seeds or (dev_seeds | heldout_seeds) & forbidden:
        raise ValueError("architecture tournament fixture seed firewall failed")
    expected_arrays = len(fixtures["development"]) * len(GAMES)
    if int(fixtures["required_arrays_per_lane"]) != expected_arrays:
        raise ValueError("architecture tournament array count differs from fixture plan")
    arms = config.get("arms")
    if not isinstance(arms, list) or len(arms) != int(fixtures["required_candidates_per_array"]):
        raise ValueError("architecture tournament requires exactly four declared arms")
    ids = [str(arm.get("candidate_id", "")) for arm in arms]
    if len(set(ids)) != len(ids) or not all(ids):
        raise ValueError("architecture tournament candidate IDs must be unique")
    roles = {str(arm.get("role", "")) for arm in arms}
    required_roles = {
        "unchanged_deterministic_control",
        "low_rank_adapter",
        "recurrence_matched_one_pass_control",
        "hrm_text_style_recurrent_surrogate",
    }
    if roles != required_roles:
        raise ValueError("architecture tournament arm matrix is incomplete")
    model_seeds = [int(seed) for seed in config["model_seed_plan"]["seeds"]]
    if len(model_seeds) < 3 or len(set(model_seeds)) != len(model_seeds):
        raise ValueError("architecture tournament requires three distinct model seeds")
    caps = config["resource_contract"]["caps"]
    for name in ("ram_mb", "cpu_percent", "io_mb_per_second", "wall_clock_seconds"):
        if int(caps.get(name, 0)) <= 0:
            raise ValueError(f"architecture tournament cap {name} must be positive")
    if int(caps["cpu_percent"]) > 100 or int(caps.get("swap_bytes", -1)) != 0:
        raise ValueError("architecture tournament CPU or swap cap is invalid")
    if config["metrics"].get("universal_scalar_winner") is not False:
        raise ValueError("architecture tournament forbids a universal scalar winner")
    if not config["source_evidence"].get("environment_verifier_remains_decision_owner"):
        raise ValueError("architecture tournament cannot delegate decision authority to the LDT")
    benchmark_path = _resolve_from_repo(config_path, str(config["benchmark_config"]))
    if not benchmark_path.is_file():
        raise ValueError(f"architecture tournament benchmark config is missing: {benchmark_path}")
    _load_contract(benchmark_path)
    return config


def _surface_rows(manifest_path: Path) -> list[dict[str, Any]]:
    _, rows_by_split = _read_dataset(manifest_path)
    return [row for split in DRIFT_SPLITS for row in rows_by_split[split]]


def build_architecture_tournament_fixture(
    config_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    config = load_architecture_tournament_contract(config_path)
    benchmark_path = _resolve_from_repo(config_path, str(config["benchmark_config"]))
    out_dir.mkdir(parents=True, exist_ok=False)
    public_root = out_dir / "public_dataset"
    release_root = out_dir / "candidate_release"
    private_root = out_dir / "private_steward"
    public_root.mkdir()
    release_root.mkdir()
    private_root.mkdir()
    fixture_entries: dict[str, list[dict[str, Any]]] = {lane: [] for lane in LANES}
    private_entries: dict[str, list[dict[str, Any]]] = {lane: [] for lane in LANES}
    release_entries: list[dict[str, Any]] = []
    lane_lineages: dict[str, set[str]] = {lane: set() for lane in LANES}
    lane_records: dict[str, set[str]] = {lane: set() for lane in LANES}
    with tempfile.TemporaryDirectory(prefix="datagoblin-architecture-v2-") as temporary:
        staging = Path(temporary)
        for lane in LANES:
            for surface in config["fixture_plan"][lane]:
                surface_id = str(surface["surface_id"])
                staged = staging / surface_id
                build_drift_benchmark(
                    benchmark_path,
                    staged,
                    seed=int(surface["generator_seed"]),
                    pressure_mode=str(surface["pressure"]),
                )
                source_manifest = staged / "public_dataset" / "dataset_manifest.json"
                destination = public_root / lane / surface_id
                shutil.copytree(staged / "public_dataset", destination)
                manifest_path = destination / "dataset_manifest.json"
                rows = _surface_rows(manifest_path)
                lineages = {str(row["lineage_group_sha256"]) for row in rows}
                records = {str(row["record_hash"]) for row in rows}
                if lane_lineages[lane] & lineages or lane_records[lane] & records:
                    raise ValueError(f"duplicate lineage or record across {lane} surfaces")
                lane_lineages[lane].update(lineages)
                lane_records[lane].update(records)
                fixture_entries[lane].append(
                    {
                        "surface_id": surface_id,
                        "generator_seed": int(surface["generator_seed"]),
                        "pressure": str(surface["pressure"]),
                        "manifest_path": manifest_path.relative_to(public_root).as_posix(),
                        "manifest_sha256": sha256_file(manifest_path),
                        "schema_surface_sha256": load_structured(source_manifest)["schema_surface_sha256"],
                        "record_count": len(rows),
                        "lineage_count": len(lineages),
                    }
                )
                ground_truth_source = staged / "private_steward" / "ground_truth.jsonl"
                ground_truth_path = private_root / lane / f"{surface_id}.jsonl"
                ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ground_truth_source, ground_truth_path)
                private_entries[lane].append(
                    {
                        "surface_id": surface_id,
                        "ground_truth_path": ground_truth_path.relative_to(private_root).as_posix(),
                        "ground_truth_sha256": sha256_file(ground_truth_path),
                    }
                )
                if lane == "development":
                    training_source = staged / "candidate_workspace" / "training_labels.jsonl"
                    training_path = release_root / f"{surface_id}.jsonl"
                    shutil.copyfile(training_source, training_path)
                    release_entries.append(
                        {
                            "surface_id": surface_id,
                            "path": training_path.name,
                            "sha256": sha256_file(training_path),
                            "allowed_splits": ["train", "calibration"],
                            "evaluation_labels_included": False,
                        }
                    )
    checks = {
        "development_surface_count_matches": len(fixture_entries["development"]) == 3,
        "heldout_surface_count_matches": len(fixture_entries["heldout"]) == 3,
        "development_heldout_lineage_overlap_zero": not bool(
            lane_lineages["development"] & lane_lineages["heldout"]
        ),
        "development_heldout_record_overlap_zero": not bool(
            lane_records["development"] & lane_records["heldout"]
        ),
        "development_training_release_only": len(release_entries) == 3,
        "heldout_labels_private": all(
            entry["surface_id"].startswith("heldout-") for entry in private_entries["heldout"]
        ),
        "fixture_seeds_fresh": not bool(
            {
                int(entry["generator_seed"])
                for lane in LANES
                for entry in fixture_entries[lane]
            }
            & {int(seed) for seed in config["fixture_plan"]["forbidden_prior_generator_seeds"]}
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"architecture tournament generation checks failed: {checks}")
    tournament_manifest = {
        "schema_version": "game-adapter-architecture-dataset.v2",
        "tournament_id": config["tournament_id"],
        "tournament_config_sha256": sha256_file(config_path),
        "benchmark_config_sha256": sha256_file(benchmark_path),
        "games": list(GAMES),
        "fixtures": fixture_entries,
        "independent_unit": config["fixture_plan"]["independent_unit"],
        "arrays_per_lane": int(config["fixture_plan"]["required_arrays_per_lane"]),
        "candidates_per_array": int(config["fixture_plan"]["required_candidates_per_array"]),
        "lineage_audit": {
            "development_count": len(lane_lineages["development"]),
            "heldout_count": len(lane_lineages["heldout"]),
            "cross_lane_overlap": len(lane_lineages["development"] & lane_lineages["heldout"]),
        },
        "record_audit": {
            "development_count": len(lane_records["development"]),
            "heldout_count": len(lane_records["heldout"]),
            "cross_lane_overlap": len(lane_records["development"] & lane_records["heldout"]),
        },
        "heldout_label_custody": {
            "sealed_before_candidate_training": True,
            "paths_disclosed_to_candidate": False,
            "reveal_count_budget": int(config["fixture_plan"]["heldout_reveal_budget"]),
            "authority": "ENVIRONMENT_OWNED",
        },
        "candidate_input_contract": "public_shape_only",
        "larger_model_training_authorized": False,
        "checks": checks,
    }
    tournament_manifest_path = public_root / "tournament_manifest.json"
    dump_structured(tournament_manifest_path, tournament_manifest)
    release_manifest = {
        "schema_version": "game-adapter-architecture-training-release.v2",
        "tournament_manifest_sha256": sha256_file(tournament_manifest_path),
        "tournament_config_sha256": sha256_file(config_path),
        "entries": release_entries,
        "allowed_lane": "development",
        "allowed_splits": ["train", "calibration"],
        "evaluation_labels_included": False,
        "heldout_labels_included": False,
        "authority": "ENVIRONMENT_OWNED_TRAINING_LABEL",
    }
    release_manifest_path = release_root / "release_manifest.json"
    dump_structured(release_manifest_path, release_manifest)
    custody_manifest = {
        "schema_version": "game-adapter-architecture-custody.v2",
        "tournament_manifest_sha256": sha256_file(tournament_manifest_path),
        "tournament_config_sha256": sha256_file(config_path),
        "entries": private_entries,
        "heldout_reveal_budget": int(config["fixture_plan"]["heldout_reveal_budget"]),
        "heldout_reveal_count": 0,
        "heldout_labels_exposed_to_candidate": False,
        "authority": "ENVIRONMENT_OWNED",
    }
    custody_manifest_path = private_root / "custody_manifest.json"
    dump_structured(custody_manifest_path, custody_manifest)
    report = {
        "schema_version": "game-adapter-architecture-generation.v2",
        "tournament_id": config["tournament_id"],
        "tournament_config_sha256": sha256_file(config_path),
        "tournament_manifest_sha256": sha256_file(tournament_manifest_path),
        "candidate_release_manifest_sha256": sha256_file(release_manifest_path),
        "private_custody_manifest_sha256": sha256_file(custody_manifest_path),
        "checks": checks,
        "generation_passed": all(checks.values()),
        "network_used": False,
        "raw_game_records_read": False,
        "heldout_labels_exported_to_candidate": False,
        "claim_boundary": config["claim_scope"],
    }
    dump_structured(out_dir / "generation_report.json", report)
    return report


def _load_public_fixture(
    config_path: Path,
    tournament_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_architecture_tournament_contract(config_path)
    manifest = load_structured(tournament_manifest_path)
    if manifest.get("schema_version") != "game-adapter-architecture-dataset.v2":
        raise ValueError("unsupported architecture tournament dataset schema")
    if manifest.get("tournament_config_sha256") != sha256_file(config_path):
        raise ValueError("architecture tournament dataset is not bound to the config")
    if tuple(manifest.get("games", ())) != GAMES or not all(manifest.get("checks", {}).values()):
        raise ValueError("architecture tournament dataset contract checks failed")
    for lane in LANES:
        expected = {str(row["surface_id"]) for row in config["fixture_plan"][lane]}
        observed = {str(row["surface_id"]) for row in manifest["fixtures"][lane]}
        if observed != expected:
            raise ValueError(f"architecture tournament {lane} fixture IDs differ")
        for entry in manifest["fixtures"][lane]:
            path = tournament_manifest_path.parent / str(entry["manifest_path"])
            if sha256_file(path) != entry["manifest_sha256"]:
                raise ValueError(f"architecture tournament surface manifest hash mismatch: {entry['surface_id']}")
            _read_dataset(path)
    return config, manifest


def _arm(config: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = [dict(arm) for arm in config["arms"] if arm["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"unknown architecture tournament candidate: {candidate_id}")
    return matches[0]


def _load_released_labels(
    release_manifest_path: Path,
    tournament_manifest_path: Path,
    config_path: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    release = load_structured(release_manifest_path)
    if release.get("schema_version") != "game-adapter-architecture-training-release.v2":
        raise ValueError("unsupported architecture training release")
    if release.get("tournament_manifest_sha256") != sha256_file(tournament_manifest_path):
        raise ValueError("architecture training release is not bound to the public dataset")
    if release.get("tournament_config_sha256") != sha256_file(config_path):
        raise ValueError("architecture training release is not bound to the config")
    if release.get("evaluation_labels_included") or release.get("heldout_labels_included"):
        raise ValueError("architecture training release contains evaluation labels")
    expected = {str(entry["surface_id"]) for entry in manifest["fixtures"]["development"]}
    observed = {str(entry["surface_id"]) for entry in release["entries"]}
    if observed != expected:
        raise ValueError("architecture training release surface set differs")
    labels: list[dict[str, Any]] = []
    for entry in release["entries"]:
        path = release_manifest_path.parent / str(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("architecture training-label release hash mismatch")
        for row in _read_jsonl(path):
            if row.get("split") not in {"train", "calibration"}:
                raise ValueError("architecture training release contains a forbidden split")
            labels.append(row)
    return labels


def train_architecture_candidate(
    config_path: Path,
    tournament_manifest_path: Path,
    release_manifest_path: Path,
    out_dir: Path,
    *,
    candidate_id: str,
    seed: int,
) -> dict[str, Any]:
    config, manifest = _load_public_fixture(config_path, tournament_manifest_path)
    arm = _arm(config, candidate_id)
    trainable = bool(arm["trainable"])
    allowed_seeds = {int(value) for value in config["model_seed_plan"]["seeds"]}
    if trainable and seed not in allowed_seeds:
        raise ValueError("architecture candidate seed is outside the frozen plan")
    if not trainable and seed != 0:
        raise ValueError("the deterministic control must use seed zero")
    if out_dir.exists():
        existing = {path.name for path in out_dir.iterdir()}
        if existing - {"events.jsonl"}:
            raise ValueError("architecture candidate output directory is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    labels = (
        _load_released_labels(
            release_manifest_path,
            tournament_manifest_path,
            config_path,
            manifest,
        )
        if trainable
        else []
    )
    train_samples = _training_samples(labels, "train") if trainable else []
    calibration_samples = _training_samples(labels, "calibration") if trainable else []
    checkpoint = config["resource_contract"]["checkpoint_interval"]
    model_state: dict[str, Any] | None = None
    mapper = None
    threshold = 0.0
    if arm["role"] == "low_rank_adapter":
        model_state, mapper, fit = train_lora_like_mapper(
            arm,
            train_samples,
            seed=seed,
            checkpoint_root=out_dir / "checkpoints",
            checkpoint_steps=int(checkpoint["steps"]),
            checkpoint_seconds=int(checkpoint["seconds"]),
        )
    elif arm["role"] in {
        "recurrence_matched_one_pass_control",
        "hrm_text_style_recurrent_surrogate",
    }:
        model_state, mapper, fit = train_recurrent_readout_mapper(
            arm,
            train_samples,
            seed=seed,
            checkpoint_root=out_dir / "checkpoints",
            checkpoint_steps=int(checkpoint["steps"]),
            checkpoint_seconds=int(checkpoint["seconds"]),
        )
    elif arm["role"] == "unchanged_deterministic_control":
        fit = {
            "samples": 0,
            "epochs": 0,
            "steps_completed": 0,
            "initial_cross_entropy": None,
            "final_cross_entropy": None,
            "converged": None,
            "trainable_parameters": 0,
            "frozen_parameters": 0,
            "checkpoints": [],
        }
    else:
        raise ValueError(f"unsupported architecture role: {arm['role']}")
    if mapper is not None:
        threshold, calibration = calibrate_mapping_threshold(
            mapper,
            calibration_samples,
            grid=[float(value) for value in config["calibration"]["threshold_grid"]],
            incorrect_penalty=float(config["calibration"]["incorrect_mapping_penalty"]),
        )
    else:
        calibration = {
            "examples": 0,
            "selected_threshold": 0.0,
            "source": "not_applicable_deterministic_control",
        }
    benchmark_path = _resolve_from_repo(config_path, str(config["benchmark_config"]))
    _, policy = _load_contract(benchmark_path)
    prediction_entries: dict[str, dict[str, dict[str, Any]]] = {lane: {} for lane in LANES}
    for lane in LANES:
        for surface in manifest["fixtures"][lane]:
            surface_id = str(surface["surface_id"])
            surface_manifest_path = tournament_manifest_path.parent / str(surface["manifest_path"])
            rows = _surface_rows(surface_manifest_path)
            prediction_path = out_dir / "predictions" / lane / f"{surface_id}.jsonl"
            write_architecture_predictions(
                rows,
                prediction_path,
                setup_id=candidate_id,
                policy=policy,
                mapper=mapper,
                threshold=threshold,
            )
            prediction_entries[lane][surface_id] = {
                "path": prediction_path.relative_to(out_dir).as_posix(),
                "sha256": sha256_file(prediction_path),
                "count": len(rows),
                "surface_manifest_sha256": surface["manifest_sha256"],
            }
    model_hash = None
    if model_state is not None:
        model_state = {
            **model_state,
            "candidate_id": candidate_id,
            "seed": seed,
            "authority": "MODEL_RECOMMENDATION_ONLY",
            "heldout_targets_seen": False,
        }
        model_hash = finalize_model_artifact(out_dir / "model.json", model_state)
    fit_receipt = {
        "schema_version": "game-adapter-architecture-fit.v2",
        "candidate_id": candidate_id,
        "seed": seed,
        "architecture_family": arm["architecture_family"],
        "tournament_config_sha256": sha256_file(config_path),
        "tournament_manifest_sha256": sha256_file(tournament_manifest_path),
        "training_release_manifest_sha256": sha256_file(release_manifest_path),
        "training_splits": ["development/train"] if trainable else [],
        "calibration_splits": ["development/calibration"] if trainable else [],
        "development_evaluation_labels_seen": False,
        "heldout_labels_seen": False,
        "fit": fit,
        "calibration": calibration,
        "threshold": threshold,
        "model_sha256": model_hash,
        "network_used": False,
        "external_checkpoint_loaded": False,
        "raw_corpus_values_read": False,
    }
    fit_path = out_dir / "fit_receipt.json"
    dump_structured(fit_path, fit_receipt)
    candidate_manifest = {
        "schema_version": "game-adapter-architecture-candidate.v2",
        "candidate_id": candidate_id,
        "architecture_family": arm["architecture_family"],
        "role": arm["role"],
        "seed": seed,
        "trainable": trainable,
        "trainable_parameters": int(fit["trainable_parameters"]),
        "trainable_parameter_bucket": arm["trainable_parameter_bucket"],
        "recurrence_steps": int(arm.get("recurrence_steps", 0)),
        "tournament_config_sha256": sha256_file(config_path),
        "tournament_manifest_sha256": sha256_file(tournament_manifest_path),
        "training_release_manifest_sha256": sha256_file(release_manifest_path),
        "input_contract": "public_shape_only",
        "training_splits": ["development/train"] if trainable else [],
        "calibration_splits": ["development/calibration"] if trainable else [],
        "evaluation_label_access": False,
        "heldout_label_access": False,
        "model_sha256": model_hash,
        "fit_receipt_sha256": sha256_file(fit_path),
        "predictions": prediction_entries,
        "authority": "MODEL_RECOMMENDATION_ONLY",
        "claim_boundary": config["claim_scope"],
    }
    candidate_path = out_dir / "candidate_manifest.json"
    dump_structured(candidate_path, candidate_manifest)
    trainer_events = [
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "complete",
            "candidate_id": candidate_id,
            "seed": seed,
            "steps_completed": int(fit["steps_completed"]),
        }
    ]
    _write_jsonl(out_dir / "trainer_events.jsonl", trainer_events)
    trainer_summary = {
        "schema_version": "game-adapter-architecture-trainer-summary.v2",
        "candidate_id": candidate_id,
        "seed": seed,
        "status": "completed",
        "steps_completed": int(fit["steps_completed"]),
        "trainable_parameters": int(fit["trainable_parameters"]),
        "checkpoint_count": len(fit["checkpoints"]),
        "checkpoint_interval_steps": int(checkpoint["steps"]),
        "checkpoint_interval_seconds": int(checkpoint["seconds"]),
        "elapsed_seconds": time.monotonic() - started,
        "object_cleanup_passed": True,
        "cuda_loaded": False,
        "weights_loaded": model_state is not None,
        "network_used": False,
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "fit_receipt_sha256": sha256_file(fit_path),
    }
    dump_structured(out_dir / "trainer_summary.json", trainer_summary)
    return candidate_manifest


def _expected_bucket(parameters: int) -> str:
    if parameters == 0:
        return "zero"
    if parameters <= 1024:
        return "small"
    if parameters <= 4096:
        return "medium"
    return "large"


def validate_architecture_candidate(
    config_path: Path,
    tournament_manifest_path: Path,
    candidate_run_dir: Path,
    *,
    require_resource_receipt: bool = True,
) -> dict[str, Any]:
    config, manifest = _load_public_fixture(config_path, tournament_manifest_path)
    candidate_path = candidate_run_dir / "candidate_manifest.json"
    fit_path = candidate_run_dir / "fit_receipt.json"
    trainer_path = candidate_run_dir / "trainer_summary.json"
    candidate = load_structured(candidate_path)
    fit = load_structured(fit_path)
    trainer = load_structured(trainer_path)
    arm = _arm(config, str(candidate.get("candidate_id", "")))
    checks = {
        "candidate_schema_valid": candidate.get("schema_version")
        == "game-adapter-architecture-candidate.v2",
        "config_bound": candidate.get("tournament_config_sha256") == sha256_file(config_path),
        "dataset_bound": candidate.get("tournament_manifest_sha256")
        == sha256_file(tournament_manifest_path),
        "training_release_bound": candidate.get("training_release_manifest_sha256")
        == fit.get("training_release_manifest_sha256"),
        "public_shape_only": candidate.get("input_contract") == "public_shape_only",
        "development_evaluation_labels_unseen": not bool(
            fit.get("development_evaluation_labels_seen")
        ),
        "heldout_labels_unseen": not bool(fit.get("heldout_labels_seen"))
        and not bool(candidate.get("heldout_label_access")),
        "evaluation_labels_unseen": not bool(candidate.get("evaluation_label_access")),
        "authority_recommendation_only": candidate.get("authority")
        == "MODEL_RECOMMENDATION_ONLY",
        "candidate_role_bound": candidate.get("role") == arm["role"],
        "candidate_family_bound": candidate.get("architecture_family")
        == arm["architecture_family"],
        "parameter_bucket_bound": candidate.get("trainable_parameter_bucket")
        == _expected_bucket(int(candidate.get("trainable_parameters", -1))),
        "fit_receipt_bound": candidate.get("fit_receipt_sha256") == sha256_file(fit_path),
        "trainer_candidate_bound": trainer.get("candidate_manifest_sha256")
        == sha256_file(candidate_path),
        "training_converged_or_control": (
            not bool(arm["trainable"]) or fit.get("fit", {}).get("converged") is True
        ),
        "checkpoint_cadence_satisfied": (
            not bool(arm["trainable"])
            or (
                int(trainer.get("checkpoint_count", 0)) > 0
                and int(trainer.get("checkpoint_interval_steps", 0))
                == int(config["resource_contract"]["checkpoint_interval"]["steps"])
            )
        ),
        "network_unused": not bool(fit.get("network_used")),
        "external_checkpoint_unused": not bool(fit.get("external_checkpoint_loaded")),
    }
    expected_surfaces = {
        lane: {str(entry["surface_id"]): entry for entry in manifest["fixtures"][lane]}
        for lane in LANES
    }
    prediction_identity_complete = True
    prediction_hashes_bound = True
    for lane in LANES:
        entries = candidate.get("predictions", {}).get(lane, {})
        if set(entries) != set(expected_surfaces[lane]):
            prediction_identity_complete = False
            continue
        for surface_id, surface in expected_surfaces[lane].items():
            entry = entries[surface_id]
            path = candidate_run_dir / str(entry["path"])
            if not path.is_file() or sha256_file(path) != entry.get("sha256"):
                prediction_hashes_bound = False
                continue
            surface_manifest_path = tournament_manifest_path.parent / str(surface["manifest_path"])
            rows = _surface_rows(surface_manifest_path)
            row_by_id = {str(row["item_id"]): row for row in rows}
            try:
                predictions = _read_predictions(path, str(candidate["candidate_id"]), row_by_id)
            except ValueError:
                prediction_identity_complete = False
                continue
            if len(predictions) != int(entry["count"]) or len(predictions) != len(rows):
                prediction_identity_complete = False
    checks["prediction_identity_complete"] = prediction_identity_complete
    checks["prediction_hashes_bound"] = prediction_hashes_bound
    if require_resource_receipt and bool(arm["trainable"]):
        run_receipt_path = candidate_run_dir / "run_receipt.json"
        if run_receipt_path.is_file():
            run_receipt = load_structured(run_receipt_path)
            checks.update(
                {
                    "run_receipt_candidate_bound": run_receipt.get("candidate_manifest_sha256")
                    == sha256_file(candidate_path),
                    "hard_caps_enforced": bool(run_receipt.get("caps_enforced")),
                    "cleanup_passed": bool(run_receipt.get("cleanup_passed")),
                    "heldout_labels_not_staged": not bool(
                        run_receipt.get("heldout_labels_staged")
                    ),
                    "development_evaluation_labels_not_staged": not bool(
                        run_receipt.get("development_evaluation_labels_staged")
                    ),
                }
            )
        else:
            checks.update(
                {
                    "run_receipt_candidate_bound": False,
                    "hard_caps_enforced": False,
                    "cleanup_passed": False,
                    "heldout_labels_not_staged": False,
                    "development_evaluation_labels_not_staged": False,
                }
            )
    elif not bool(arm["trainable"]):
        checks.update(
            {
                "run_receipt_candidate_bound": True,
                "hard_caps_enforced": True,
                "cleanup_passed": True,
                "heldout_labels_not_staged": True,
                "development_evaluation_labels_not_staged": True,
            }
        )
    report = {
        "schema_version": "game-adapter-architecture-preflight.v2",
        "candidate_id": candidate["candidate_id"],
        "seed": int(candidate["seed"]),
        "checks": checks,
        "valid": all(checks.values()),
        "authoritative_labels_read": False,
    }
    if not report["valid"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"architecture candidate preflight failed: {failed}")
    return report


def _ece(rows: Sequence[Mapping[str, Any]], bins: int = 10) -> float:
    if not rows:
        return 0.0
    total = len(rows)
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            row
            for row in rows
            if lower <= float(row["confidence"]) < upper
            or (index == bins - 1 and float(row["confidence"]) == 1.0)
        ]
        if selected:
            confidence = float(np.mean([float(row["confidence"]) for row in selected]))
            accuracy = float(np.mean([float(bool(row["exact"])) for row in selected]))
            result += len(selected) / total * abs(confidence - accuracy)
    return result


def _aggregate_with_ece(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    value = _aggregate(rows)
    value["expected_calibration_error"] = _ece(rows)
    return value


def _custody_entries(
    custody_manifest_path: Path,
    tournament_manifest_path: Path,
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    custody = load_structured(custody_manifest_path)
    if custody.get("schema_version") != "game-adapter-architecture-custody.v2":
        raise ValueError("unsupported architecture tournament custody manifest")
    if custody.get("tournament_manifest_sha256") != sha256_file(tournament_manifest_path):
        raise ValueError("architecture custody manifest is not bound to the dataset")
    if custody.get("tournament_config_sha256") != sha256_file(config_path):
        raise ValueError("architecture custody manifest is not bound to the config")
    entries = {
        lane: {str(entry["surface_id"]): entry for entry in custody["entries"][lane]}
        for lane in LANES
    }
    return custody, entries


def _score_candidate_surface(
    benchmark_path: Path,
    surface_manifest_path: Path,
    label_path: Path,
    prediction_path: Path,
    *,
    candidate_id: str,
) -> dict[str, Any]:
    _, policy = _load_contract(benchmark_path)
    _, rows_by_split = _read_dataset(surface_manifest_path)
    rows = [row for split in DRIFT_SPLITS for row in rows_by_split[split]]
    row_by_id = {str(row["item_id"]): row for row in rows}
    labels = _read_steward_labels(label_path, row_by_id)
    predictions = _read_predictions(prediction_path, candidate_id, row_by_id)
    scored = [
        _score(
            row,
            labels[str(row["item_id"])]["expected"],
            predictions[str(row["item_id"])],
            policy,
        )
        for row in rows
    ]
    split_results = {
        split: _aggregate_with_ece([row for row in scored if row["split"] == split])
        for split in EVALUATION_SPLITS
    }
    game_split_results = {
        game: {
            split: _aggregate_with_ece(
                [row for row in scored if row["game"] == game and row["split"] == split]
            )
            for split in EVALUATION_SPLITS
        }
        for game in GAMES
    }
    evaluation_rows = [row for row in scored if row["split"] in EVALUATION_SPLITS]
    return {
        "split_results": split_results,
        "game_split_results": game_split_results,
        "evaluation_overall": _aggregate_with_ece(evaluation_rows),
        "label_sha256": sha256_file(label_path),
        "prediction_sha256": sha256_file(prediction_path),
    }


def _interval(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot create an empty architecture metric interval")
    return {"lower": float(min(values)), "upper": float(max(values))}


def _summarize_lane(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    *,
    lane: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arms = [dict(arm) for arm in config["arms"]]
    baseline_id = next(
        str(arm["candidate_id"])
        for arm in arms
        if arm["role"] == "unchanged_deterministic_control"
    )
    summaries: dict[str, Any] = {}
    feature_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for arm in arms:
        candidate_id = str(arm["candidate_id"])
        candidate_cells = [cell for cell in cells if cell["candidate_id"] == candidate_id]
        if not candidate_cells:
            raise ValueError(f"missing architecture evaluation cells for {candidate_id}")
        safety_checks = []
        ordinary_values = []
        ood_values = []
        per_game = {}
        for game in GAMES:
            ordinary_game = []
            ood_game = []
            for cell in candidate_cells:
                ordinary = cell["metrics"]["game_split_results"][game]["ordinary_test"]
                ood = cell["metrics"]["game_split_results"][game]["ood_test"]
                sealed = cell["metrics"]["game_split_results"][game]["sealed_poison"]
                ordinary_game.append(float(ordinary["exact_record_accuracy"]))
                ood_game.append(float(ood["exact_record_accuracy"]))
                ordinary_values.append(float(ordinary["exact_record_accuracy"]))
                ood_values.append(float(ood["exact_record_accuracy"]))
                safety_checks.append(
                    ordinary["privacy_leak_rate"] == 0.0
                    and ood["privacy_leak_rate"] == 0.0
                    and ordinary["critical_false_release_rate"] == 0.0
                    and ood["critical_false_release_rate"] == 0.0
                    and sealed["exact_record_accuracy"] == 1.0
                )
            per_game[game] = {
                "ordinary_exact_interval": _interval(ordinary_game),
                "ood_exact_interval": _interval(ood_game),
            }
        summaries[candidate_id] = {
            "candidate_id": candidate_id,
            "architecture_family": arm["architecture_family"],
            "role": arm["role"],
            "trainable_parameters": int(candidate_cells[0]["trainable_parameters"]),
            "trainable_parameter_bucket": arm["trainable_parameter_bucket"],
            "safety_qualified": all(safety_checks),
            "ordinary_exact_record_accuracy": float(np.mean(ordinary_values)),
            "ood_exact_record_accuracy": float(np.mean(ood_values)),
            "minimum_ordinary_exact_record_accuracy": min(ordinary_values),
            "minimum_ood_exact_record_accuracy": min(ood_values),
            "per_game": per_game,
        }
    baseline = summaries[baseline_id]
    gates = config[f"{lane}_gates"]
    for candidate_id, summary in summaries.items():
        arm = _arm(config, candidate_id)
        gains = {
            game: {
                "ordinary": summary["per_game"][game]["ordinary_exact_interval"]["lower"]
                - baseline["per_game"][game]["ordinary_exact_interval"]["upper"],
                "ood": summary["per_game"][game]["ood_exact_interval"]["lower"]
                - baseline["per_game"][game]["ood_exact_interval"]["upper"],
            }
            for game in GAMES
        }
        checks = {
            "trainable": bool(arm["trainable"]),
            "safety": bool(summary["safety_qualified"]),
            "ordinary_minimum": summary["ordinary_exact_record_accuracy"]
            >= float(gates["minimum_ordinary_exact_record_accuracy"]),
            "ood_minimum": summary["ood_exact_record_accuracy"]
            >= float(gates["minimum_ood_exact_record_accuracy"]),
            "ordinary_gain_every_game": (
                not bool(gates["require_nonnegative_per_game_ordinary_gain_over_control"])
                or all(value["ordinary"] >= 0.0 for value in gains.values())
            ),
            "ood_gain_every_game": (
                not bool(gates["require_nonnegative_per_game_ood_gain_over_control"])
                or all(value["ood"] >= 0.0 for value in gains.values())
            ),
        }
        summary["gains_over_control"] = gains
        summary["qualification_checks"] = checks
        summary["qualified"] = all(checks.values())
    surfaces = {str(entry["surface_id"]): entry for entry in manifest["fixtures"][lane]}
    arrays: list[dict[str, Any]] = []
    ordered_ids = [str(arm["candidate_id"]) for arm in arms]
    for surface_id, surface in surfaces.items():
        for game in GAMES:
            candidates = []
            for arm in arms:
                candidate_id = str(arm["candidate_id"])
                selected_cells = [
                    cell
                    for cell in cells
                    if cell["candidate_id"] == candidate_id and cell["surface_id"] == surface_id
                ]
                metrics = {}
                for field, split, metric in (
                    ("ordinary_exact_record_accuracy", "ordinary_test", "exact_record_accuracy"),
                    ("ordinary_mapping_f1", "ordinary_test", "mapping_f1"),
                    ("ood_exact_record_accuracy", "ood_test", "exact_record_accuracy"),
                    ("ood_mapping_f1", "ood_test", "mapping_f1"),
                ):
                    metrics[field] = _interval(
                        [
                            float(cell["metrics"]["game_split_results"][game][split][metric])
                            for cell in selected_cells
                        ]
                    )
                feature = {
                    "game": game,
                    "fixture_pressure": surface["pressure"],
                    "context_mode": arm["context_mode"],
                    "optimization_pressure": arm["optimization_pressure"],
                    "trainable_parameter_bucket": arm["trainable_parameter_bucket"],
                    "safety_qualified": bool(summaries[candidate_id]["safety_qualified"]),
                    **metrics,
                }
                feature_lookup[(surface_id, game, candidate_id)] = feature
                candidates.append({"candidate_id": candidate_id, "features": feature})
            target = derive_confirmation_array_target(
                [candidate["features"] for candidate in candidates],
                tolerance=float(config["decision_policy"]["tolerance"]),
            )
            arrays.append(
                {
                    "schema_version": "game-adapter-architecture-array.v2",
                    "array_id": f"architecture-array-{surface['pressure']}-{game}",
                    "lane": lane,
                    "surface_id": surface_id,
                    "fixture_pressure": surface["pressure"],
                    "game": game,
                    "candidate_ids": ordered_ids,
                    "candidates": candidates,
                    "target": target,
                }
            )
    return summaries, sorted(arrays, key=lambda row: str(row["array_id"]))


def evaluate_architecture_development(
    config_path: Path,
    tournament_manifest_path: Path,
    custody_manifest_path: Path,
    candidate_run_dirs: Sequence[Path],
    out_dir: Path,
    *,
    require_resource_receipts: bool = True,
) -> dict[str, Any]:
    config, manifest = _load_public_fixture(config_path, tournament_manifest_path)
    custody, custody_entries = _custody_entries(
        custody_manifest_path,
        tournament_manifest_path,
        config_path,
    )
    if int(custody.get("heldout_reveal_count", -1)) != 0:
        raise ValueError("heldout custody was already consumed before development evaluation")
    expected_cells = {
        (str(arm["candidate_id"]), seed)
        for arm in config["arms"]
        for seed in (
            config["model_seed_plan"]["seeds"] if arm["trainable"] else [0]
        )
    }
    observed: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    preflights = []
    for run_dir in candidate_run_dirs:
        preflight = validate_architecture_candidate(
            config_path,
            tournament_manifest_path,
            run_dir,
            require_resource_receipt=require_resource_receipts,
        )
        candidate = load_structured(run_dir / "candidate_manifest.json")
        key = (str(candidate["candidate_id"]), int(candidate["seed"]))
        if key in observed:
            raise ValueError(f"duplicate architecture tournament cell: {key}")
        observed[key] = (run_dir, candidate)
        preflights.append(preflight)
    if set(observed) != expected_cells:
        missing = sorted(expected_cells - set(observed))
        extra = sorted(set(observed) - expected_cells)
        raise ValueError(f"architecture tournament matrix mismatch; missing={missing} extra={extra}")
    benchmark_path = _resolve_from_repo(config_path, str(config["benchmark_config"]))
    surfaces = {
        str(entry["surface_id"]): entry for entry in manifest["fixtures"]["development"]
    }
    cells = []
    for (candidate_id, seed), (run_dir, candidate) in sorted(observed.items()):
        for surface_id, surface in surfaces.items():
            label_entry = custody_entries["development"][surface_id]
            label_path = custody_manifest_path.parent / str(label_entry["ground_truth_path"])
            if sha256_file(label_path) != label_entry["ground_truth_sha256"]:
                raise ValueError(f"development steward label hash mismatch: {surface_id}")
            surface_manifest_path = tournament_manifest_path.parent / str(surface["manifest_path"])
            prediction_entry = candidate["predictions"]["development"][surface_id]
            prediction_path = run_dir / str(prediction_entry["path"])
            metrics = _score_candidate_surface(
                benchmark_path,
                surface_manifest_path,
                label_path,
                prediction_path,
                candidate_id=candidate_id,
            )
            cells.append(
                {
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "surface_id": surface_id,
                    "pressure": surface["pressure"],
                    "trainable_parameters": int(candidate["trainable_parameters"]),
                    "metrics": metrics,
                }
            )
    summaries, arrays = _summarize_lane(
        config,
        manifest,
        cells,
        lane="development",
    )
    public_arrays = [
        {
            key: value
            for key, value in row.items()
            if key not in {"target", "lane", "surface_id"}
        }
        for row in arrays
    ]
    arrays_path = out_dir / "development_public_arrays.jsonl"
    out_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(arrays_path, public_arrays)
    report = {
        "schema_version": "game-adapter-architecture-development-evaluation.v2",
        "tournament_id": config["tournament_id"],
        "tournament_config_sha256": sha256_file(config_path),
        "tournament_manifest_sha256": sha256_file(tournament_manifest_path),
        "custody_manifest_sha256": sha256_file(custody_manifest_path),
        "preflight_valid": all(row["valid"] for row in preflights),
        "cell_count": len(cells),
        "cells": cells,
        "candidate_summaries": summaries,
        "environment_array_targets": [
            {
                "array_id": row["array_id"],
                "candidate_ids": row["candidate_ids"],
                "target": row["target"],
            }
            for row in arrays
        ],
        "public_arrays_sha256": sha256_file(arrays_path),
        "heldout_labels_read": False,
        "heldout_reveal_count": 0,
        "qualified_candidates": sorted(
            candidate_id
            for candidate_id, summary in summaries.items()
            if summary["qualified"]
        ),
        "winner": None,
        "universal_scalar_winner": False,
        "conclusion_layers": {
            "task_result": "development_evaluated",
            "measurement_reliability": "deterministic_environment_labels_and_complete_matrix",
            "claim_support": "development_only_not_heldout",
            "operational_decision": "prepare_v15_shadow_packet_before_single_heldout_join",
        },
        "claim_boundary": config["claim_scope"],
    }
    dump_structured(out_dir / "development_evaluation.json", report)
    return report


def _load_v15_model(
    model_path: Path,
) -> tuple[ExactArrayLikelihoodModel, EdgeDifferenceCandidateEncoder, dict[str, Any]]:
    payload = load_structured(model_path)
    if payload.get("schema_version") != "ldt-array-confirmation-model.v3":
        raise ValueError("unsupported retained LDT model schema")
    if payload.get("candidate_id") != "exact-edge-h4-global-v15" or int(payload.get("seed", -1)) != 167:
        raise ValueError("retained LDT model identity differs from v15")
    if payload.get("authority") != "MODEL_RECOMMENDATION_ONLY" or payload.get("heldout_targets_seen"):
        raise ValueError("retained LDT model authority or custody declaration is invalid")
    state = payload.get("listwise_state")
    if not isinstance(state, Mapping):
        raise ValueError("retained v15 model has no direct listwise state")
    if state.get("architecture") != "exact_globally_normalized_array_status":
        raise ValueError("retained v15 model is not the exact global decoder")
    encoder = EdgeDifferenceCandidateEncoder()
    model = ExactArrayLikelihoodModel(
        input_width=int(state["input_width"]),
        hidden_width=int(state["hidden_width"]),
        loop_count=int(state["loop_count"]),
        seed=167,
        max_candidates=int(state["max_candidates"]),
        decoder=str(state["decoder"]),
    )
    if model.input_width != len(encoder.feature_names()):
        raise ValueError("retained v15 model input width differs from the edge encoder")
    for name in (
        "input_weights",
        "hidden_bias",
        "status_weights",
        "status_pool_weights",
        "status_bias",
        "selected_count_potential",
    ):
        setattr(model, name, np.asarray(state[name], dtype=np.float64))
    model.self_weights = np.asarray(state["self_weights"], dtype=np.float64)
    model.pool_weights = np.asarray(state["pool_weights"], dtype=np.float64)
    return model, encoder, payload


def apply_v15_shadow_sorter(
    config_path: Path,
    development_evaluation_path: Path,
    development_arrays_path: Path,
    retained_model_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    config = load_architecture_tournament_contract(config_path)
    expected_hash = str(config["source_evidence"]["v15_model_sha256"])
    if sha256_file(retained_model_path) != expected_hash:
        raise ValueError("retained v15 model hash differs from the frozen tournament contract")
    development = load_structured(development_evaluation_path)
    if development.get("tournament_config_sha256") != sha256_file(config_path):
        raise ValueError("development evaluation is not bound to the tournament config")
    if development.get("public_arrays_sha256") != sha256_file(development_arrays_path):
        raise ValueError("development array packet hash differs")
    if development.get("heldout_labels_read") or int(development.get("heldout_reveal_count", -1)) != 0:
        raise ValueError("shadow sorting must occur before the heldout join")
    model, encoder, payload = _load_v15_model(retained_model_path)
    arrays = _read_jsonl(development_arrays_path)
    expected_candidates = [str(arm["candidate_id"]) for arm in config["arms"]]
    predictions = []
    for row in arrays:
        if row.get("candidate_ids") != expected_candidates or len(row.get("candidates", ())) != 4:
            raise ValueError(f"shadow array candidate contract differs: {row.get('array_id')}")
        candidate_ids = [str(value["candidate_id"]) for value in row["candidates"]]
        if candidate_ids != expected_candidates:
            raise ValueError(f"shadow array candidate order differs: {row.get('array_id')}")
        prediction = predict_exact_array_statuses(
            model,
            encoder,
            {"candidates": [{"features": value["features"]} for value in row["candidates"]]},
        )
        predictions.append(
            {
                "schema_version": "game-adapter-architecture-shadow-prediction.v2",
                "array_id": row["array_id"],
                "candidate_ids": candidate_ids,
                "statuses": prediction["statuses"],
                "status_probabilities": prediction["status_probabilities"],
                "authority": "MODEL_RECOMMENDATION_ONLY",
                "source_model_id": payload["candidate_id"],
                "source_model_seed": int(payload["seed"]),
                "source_model_sha256": expected_hash,
                "input_arrays_sha256": sha256_file(development_arrays_path),
                "heldout_metrics_seen": False,
                "heldout_targets_seen": False,
            }
        )
    out_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = out_dir / "shadow_predictions.jsonl"
    _write_jsonl(predictions_path, predictions)
    receipt = {
        "schema_version": "game-adapter-architecture-shadow-receipt.v2",
        "tournament_config_sha256": sha256_file(config_path),
        "development_evaluation_sha256": sha256_file(development_evaluation_path),
        "development_arrays_sha256": sha256_file(development_arrays_path),
        "retained_model_sha256": expected_hash,
        "prediction_sha256": sha256_file(predictions_path),
        "array_count": len(predictions),
        "authority": "MODEL_RECOMMENDATION_ONLY",
        "heldout_metrics_seen": False,
        "heldout_targets_seen": False,
        "next_gate": "public packet audit then one environment-owned heldout join",
    }
    dump_structured(out_dir / "shadow_receipt.json", receipt)
    return receipt


def audit_architecture_confirmation_packet(
    config_path: Path,
    tournament_manifest_path: Path,
    development_evaluation_path: Path,
    development_arrays_path: Path,
    shadow_predictions_path: Path,
    shadow_receipt_path: Path,
    candidate_run_dirs: Sequence[Path],
    out_path: Path,
    *,
    require_resource_receipts: bool = True,
) -> dict[str, Any]:
    config, manifest = _load_public_fixture(config_path, tournament_manifest_path)
    development = load_structured(development_evaluation_path)
    shadow_receipt = load_structured(shadow_receipt_path)
    predictions = _read_jsonl(shadow_predictions_path)
    arrays = _read_jsonl(development_arrays_path)
    checks = {
        "development_config_bound": development.get("tournament_config_sha256")
        == sha256_file(config_path),
        "development_dataset_bound": development.get("tournament_manifest_sha256")
        == sha256_file(tournament_manifest_path),
        "heldout_not_read": not bool(development.get("heldout_labels_read"))
        and int(development.get("heldout_reveal_count", -1)) == 0,
        "shadow_config_bound": shadow_receipt.get("tournament_config_sha256")
        == sha256_file(config_path),
        "shadow_model_bound": shadow_receipt.get("retained_model_sha256")
        == config["source_evidence"]["v15_model_sha256"],
        "shadow_prediction_bound": shadow_receipt.get("prediction_sha256")
        == sha256_file(shadow_predictions_path),
        "shadow_inputs_bound": shadow_receipt.get("development_arrays_sha256")
        == sha256_file(development_arrays_path),
        "shadow_has_no_heldout_access": not bool(shadow_receipt.get("heldout_metrics_seen"))
        and not bool(shadow_receipt.get("heldout_targets_seen")),
        "array_count_matches": len(arrays)
        == int(config["fixture_plan"]["required_arrays_per_lane"]),
        "prediction_count_matches": len(predictions) == len(arrays),
        "array_ids_match": {str(row["array_id"]) for row in arrays}
        == {str(row["array_id"]) for row in predictions},
        "shadow_authority_is_recommendation_only": all(
            row.get("authority") == "MODEL_RECOMMENDATION_ONLY" for row in predictions
        ),
        "shadow_rows_have_no_heldout_access": all(
            not bool(row.get("heldout_metrics_seen")) and not bool(row.get("heldout_targets_seen"))
            for row in predictions
        ),
        "lineage_contamination_zero": int(manifest["lineage_audit"]["cross_lane_overlap"]) == 0,
        "record_contamination_zero": int(manifest["record_audit"]["cross_lane_overlap"]) == 0,
    }
    candidate_preflights = []
    for run_dir in candidate_run_dirs:
        candidate_preflights.append(
            validate_architecture_candidate(
                config_path,
                tournament_manifest_path,
                run_dir,
                require_resource_receipt=require_resource_receipts,
            )
        )
    expected_cells = 1 + sum(
        len(config["model_seed_plan"]["seeds"])
        for arm in config["arms"]
        if arm["trainable"]
    )
    checks["candidate_cell_count_matches"] = len(candidate_preflights) == expected_cells
    checks["candidate_preflights_pass"] = all(row["valid"] for row in candidate_preflights)
    report = {
        "schema_version": "game-adapter-architecture-confirmation-packet-audit.v2",
        "checks": checks,
        "passed": all(checks.values()),
        "authoritative_labels_read": False,
        "heldout_join_authorized": all(checks.values()),
        "heldout_join_budget": 1,
    }
    dump_structured(out_path, report)
    if not report["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"architecture confirmation packet audit failed: {failed}")
    return report


def evaluate_architecture_heldout(
    config_path: Path,
    tournament_manifest_path: Path,
    custody_manifest_path: Path,
    development_evaluation_path: Path,
    confirmation_packet_audit_path: Path,
    shadow_predictions_path: Path,
    candidate_run_dirs: Sequence[Path],
    out_dir: Path,
    *,
    require_resource_receipts: bool = True,
) -> dict[str, Any]:
    config, manifest = _load_public_fixture(config_path, tournament_manifest_path)
    custody, custody_entries = _custody_entries(
        custody_manifest_path,
        tournament_manifest_path,
        config_path,
    )
    if int(custody.get("heldout_reveal_count", -1)) != 0:
        raise ValueError("heldout architecture labels have already been joined")
    if int(custody.get("heldout_reveal_budget", 0)) != 1:
        raise ValueError("heldout architecture reveal budget differs from one")
    packet_audit = load_structured(confirmation_packet_audit_path)
    if (
        packet_audit.get("schema_version")
        != "game-adapter-architecture-confirmation-packet-audit.v2"
        or not packet_audit.get("heldout_join_authorized")
        or packet_audit.get("authoritative_labels_read")
    ):
        raise ValueError("architecture heldout join lacks a passing no-label packet audit")
    development = load_structured(development_evaluation_path)
    if development.get("tournament_config_sha256") != sha256_file(config_path):
        raise ValueError("architecture development report is not bound to the config")
    expected_cells = {
        (str(arm["candidate_id"]), seed)
        for arm in config["arms"]
        for seed in (
            config["model_seed_plan"]["seeds"] if arm["trainable"] else [0]
        )
    }
    observed: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    for run_dir in candidate_run_dirs:
        validate_architecture_candidate(
            config_path,
            tournament_manifest_path,
            run_dir,
            require_resource_receipt=require_resource_receipts,
        )
        candidate = load_structured(run_dir / "candidate_manifest.json")
        key = (str(candidate["candidate_id"]), int(candidate["seed"]))
        if key in observed:
            raise ValueError(f"duplicate architecture tournament heldout cell: {key}")
        observed[key] = (run_dir, candidate)
    if set(observed) != expected_cells:
        raise ValueError("architecture heldout candidate matrix differs from the frozen plan")
    benchmark_path = _resolve_from_repo(config_path, str(config["benchmark_config"]))
    surfaces = {str(entry["surface_id"]): entry for entry in manifest["fixtures"]["heldout"]}
    cells = []
    heldout_label_hashes = {}
    for (candidate_id, seed), (run_dir, candidate) in sorted(observed.items()):
        for surface_id, surface in surfaces.items():
            label_entry = custody_entries["heldout"][surface_id]
            label_path = custody_manifest_path.parent / str(label_entry["ground_truth_path"])
            if sha256_file(label_path) != label_entry["ground_truth_sha256"]:
                raise ValueError(f"heldout steward label hash mismatch: {surface_id}")
            heldout_label_hashes[surface_id] = sha256_file(label_path)
            surface_manifest_path = tournament_manifest_path.parent / str(surface["manifest_path"])
            prediction_entry = candidate["predictions"]["heldout"][surface_id]
            prediction_path = run_dir / str(prediction_entry["path"])
            metrics = _score_candidate_surface(
                benchmark_path,
                surface_manifest_path,
                label_path,
                prediction_path,
                candidate_id=candidate_id,
            )
            cells.append(
                {
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "surface_id": surface_id,
                    "pressure": surface["pressure"],
                    "trainable_parameters": int(candidate["trainable_parameters"]),
                    "metrics": metrics,
                }
            )
    summaries, arrays = _summarize_lane(config, manifest, cells, lane="heldout")
    shadow_rows = _read_jsonl(shadow_predictions_path)
    shadow = {str(row["array_id"]): row for row in shadow_rows}
    if len(shadow) != len(shadow_rows) or set(shadow) != {str(row["array_id"]) for row in arrays}:
        raise ValueError("architecture shadow prediction array set differs from heldout targets")
    array_results = []
    for row in arrays:
        prediction = shadow[str(row["array_id"])]
        if prediction.get("candidate_ids") != row["candidate_ids"]:
            raise ValueError(f"architecture shadow candidate order differs: {row['array_id']}")
        actual_statuses = [str(value) for value in row["target"]["statuses"]]
        predicted_statuses = [str(value) for value in prediction["statuses"]]
        actual_selected = {
            row["candidate_ids"][index]
            for index, status in enumerate(actual_statuses)
            if status == "SELECTED"
        }
        predicted_selected = {
            row["candidate_ids"][index]
            for index, status in enumerate(predicted_statuses)
            if status == "SELECTED"
        }
        array_results.append(
            {
                "array_id": row["array_id"],
                "surface_id": row["surface_id"],
                "pressure": row["fixture_pressure"],
                "game": row["game"],
                "candidate_ids": row["candidate_ids"],
                "actual_statuses": actual_statuses,
                "predicted_statuses": predicted_statuses,
                "actual_selected": sorted(actual_selected),
                "predicted_selected": sorted(predicted_selected),
                "exact_status_vector": predicted_statuses == actual_statuses,
                "exact_selected_set": predicted_selected == actual_selected,
                "safety_status_accuracy": float(
                    np.mean(
                        [
                            (predicted == "UNSAFE") == (actual == "UNSAFE")
                            for predicted, actual in zip(
                                predicted_statuses, actual_statuses, strict=True
                            )
                        ]
                    )
                ),
            }
        )
    exact_status = float(np.mean([row["exact_status_vector"] for row in array_results]))
    exact_selected = float(np.mean([row["exact_selected_set"] for row in array_results]))
    per_surface = {
        pressure: {
            "array_count": len(selected),
            "exact_status_vector_accuracy": float(
                np.mean([row["exact_status_vector"] for row in selected])
            ),
            "exact_selected_set_accuracy": float(
                np.mean([row["exact_selected_set"] for row in selected])
            ),
        }
        for pressure in sorted({str(row["pressure"]) for row in array_results})
        for selected in [[row for row in array_results if row["pressure"] == pressure]]
    }
    worst_status = min(value["exact_status_vector_accuracy"] for value in per_surface.values())
    worst_selected = min(value["exact_selected_set_accuracy"] for value in per_surface.values())
    gates = config["heldout_gates"]
    ldt_checks = {
        "minimum_exact_status_vector_accuracy": exact_status
        >= float(gates["minimum_ldt_exact_status_vector_accuracy"]),
        "minimum_exact_selected_set_accuracy": exact_selected
        >= float(gates["minimum_ldt_exact_selected_set_accuracy"]),
        "minimum_worst_surface_exact_status_vector_accuracy": worst_status
        >= float(gates["minimum_worst_surface_exact_status_vector_accuracy"]),
        "minimum_worst_surface_exact_selected_set_accuracy": worst_selected
        >= float(gates["minimum_worst_surface_exact_selected_set_accuracy"]),
    }
    global_ldt_passed = all(ldt_checks.values())
    advancement = {}
    minimum_selected_fraction = float(
        config["candidate_advancement"]["minimum_heldout_selected_array_fraction"]
    )
    development_summaries = development["candidate_summaries"]
    for candidate_id, summary in summaries.items():
        statuses = [
            row["actual_statuses"][row["candidate_ids"].index(candidate_id)]
            for row in array_results
        ]
        selected_fraction = statuses.count("SELECTED") / len(statuses)
        arm = _arm(config, candidate_id)
        checks = {
            "trainable": bool(arm["trainable"]),
            "development_qualified": bool(development_summaries[candidate_id]["qualified"]),
            "heldout_qualified": bool(summary["qualified"]),
            "minimum_selected_fraction": selected_fraction >= minimum_selected_fraction,
            "no_unsafe_status": "UNSAFE" not in statuses,
            "global_ldt_whole_array_gates": global_ldt_passed,
        }
        advancement[candidate_id] = {
            "heldout_status_counts": dict(sorted(Counter(statuses).items())),
            "heldout_selected_array_fraction": selected_fraction,
            "checks": checks,
            "advances": all(checks.values()),
        }
    advancing = sorted(
        candidate_id for candidate_id, value in advancement.items() if value["advances"]
    )
    final_checks = {
        "packet_audit_passed_before_labels": bool(packet_audit["passed"]),
        "single_heldout_join": int(config["fixture_plan"]["heldout_reveal_budget"]) == 1,
        "lineage_contamination_zero": int(manifest["lineage_audit"]["cross_lane_overlap"]) == 0,
        "record_contamination_zero": int(manifest["record_audit"]["cross_lane_overlap"]) == 0,
        "all_trainable_cells_resource_qualified": all(
            validate_architecture_candidate(
                config_path,
                tournament_manifest_path,
                run_dir,
                require_resource_receipt=require_resource_receipts,
            )["valid"]
            for run_dir in candidate_run_dirs
        ),
        "ldt_whole_array_gates": global_ldt_passed,
        "at_least_one_trainable_adapter_advances": bool(advancing),
        "no_post_reveal_tuning": bool(gates["no_post_reveal_tuning"]),
    }
    confirmed = all(final_checks.values())
    out_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": "game-adapter-architecture-heldout-evaluation.v2",
        "tournament_id": config["tournament_id"],
        "tournament_config_sha256": sha256_file(config_path),
        "tournament_manifest_sha256": sha256_file(tournament_manifest_path),
        "development_evaluation_sha256": sha256_file(development_evaluation_path),
        "confirmation_packet_audit_sha256": sha256_file(confirmation_packet_audit_path),
        "shadow_predictions_sha256": sha256_file(shadow_predictions_path),
        "heldout_label_hashes": heldout_label_hashes,
        "heldout_reveal_count": 1,
        "post_reveal_tuning": False,
        "candidate_summaries": summaries,
        "array_results": array_results,
        "whole_array_metrics": {
            "exact_status_vector_accuracy": exact_status,
            "exact_selected_set_accuracy": exact_selected,
            "worst_surface_exact_status_vector_accuracy": worst_status,
            "worst_surface_exact_selected_set_accuracy": worst_selected,
            "mean_safety_status_accuracy": float(
                np.mean([row["safety_status_accuracy"] for row in array_results])
            ),
            "per_surface": per_surface,
        },
        "ldt_gate_checks": ldt_checks,
        "candidate_advancement": advancement,
        "advancing_candidates": advancing,
        "winner": None,
        "universal_scalar_winner": False,
        "final_checks": final_checks,
        "confirmed_for_next_preregistration_only": confirmed,
        "conclusion_layers": {
            "metric_robustness": "v15_exact_whole_array_metrics_previously_confirmed",
            "task_result": "heldout_confirmed" if confirmed else "heldout_not_confirmed",
            "measurement_reliability": "single_join_environment_owned_exact_labels",
            "claim_support": "supported_for_declared_tiny_surrogates" if confirmed else "not_supported",
            "operational_decision": (
                "admit_advancing_recipes_to_separate_preregistration"
                if confirmed
                else "no_architecture_promotion"
            ),
        },
        "mandatory_non_claims": config["mandatory_non_claims"],
        "claim_boundary": config["claim_scope"],
    }
    report_path = out_dir / "heldout_evaluation.json"
    dump_structured(report_path, report)
    receipt = {
        "schema_version": "game-adapter-architecture-heldout-receipt.v2",
        "heldout_evaluation_sha256": sha256_file(report_path),
        "heldout_reveal_count": 1,
        "post_reveal_tuning": False,
        "confirmed_for_next_preregistration_only": confirmed,
        "advancing_candidates": advancing,
    }
    dump_structured(out_dir / "heldout_evaluation_receipt.json", receipt)
    return report
