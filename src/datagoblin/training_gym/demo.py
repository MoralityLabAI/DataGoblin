from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from datagoblin.experiments.recipe_verdict_dataset import (
    generate_verdict_dataset,
    read_dataset_rows,
    steward_sealed_labels,
)
from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import dump_structured

from .candidate import write_mock_candidate
from .comparison import compare_candidates
from .contract import load_gym_contract
from .evaluation import evaluate_candidate


def run_training_gym_demo(
    contract_path: Path,
    recipe_path: Path,
    out_dir: Path,
    *,
    seed: int = 7,
) -> dict[str, Any]:
    contract = load_gym_contract(contract_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(contract_path, out_dir / "frozen_contract.yaml")
    dataset_dir = out_dir / "public_dataset"
    manifest = generate_verdict_dataset(recipe_path, dataset_dir, seed=seed)
    manifest_path = dataset_dir / "dataset_manifest.json"
    rows = read_dataset_rows(manifest_path)
    candidates = {
        "constant_accept": "constant-accept-negative-control",
        "structural_smoke": "structural-two-loop-smoke-control",
    }
    candidate_paths: dict[str, tuple[Path, Path]] = {}
    for kind, candidate_id in candidates.items():
        candidate_paths[kind] = write_mock_candidate(
            rows,
            out_dir / "candidates" / kind,
            candidate_id=candidate_id,
            kind=kind,
            dataset_manifest_path=manifest_path,
        )
    # The prediction files are complete before the independent validator writes sealed labels.
    steward_dir = out_dir / "private_steward"
    steward_sealed_labels(manifest_path, steward_dir)
    sealed_labels_path = steward_dir / "sealed_labels.jsonl"
    report_paths: list[Path] = []
    for kind, (candidate_manifest, predictions) in candidate_paths.items():
        evaluation_dir = out_dir / "evaluations" / kind
        evaluate_candidate(
            out_dir / "frozen_contract.yaml",
            manifest_path,
            candidate_manifest,
            predictions,
            evaluation_dir,
            sealed_labels_path=sealed_labels_path,
        )
        report_paths.append(evaluation_dir / "evaluation_report.json")
    trainer_plan = {
        "schema_version": "1.0",
        "training_task_id": contract["trainer"]["training_task_id"],
        "status": "scaffold_only_no_training_started",
        "caps": contract["trainer"]["caps"],
        "chunk_strategy": contract["trainer"]["chunk_strategy"],
        "checkpoint_interval": contract["trainer"]["checkpoint_interval"],
        "candidate_families": list(contract["candidate_families"]),
        "hard_cap_wrapper": contract["trainer"]["wrapper"],
        "post_run_cleanup": contract["trainer"]["cleanup"],
        "weights_loaded": False,
        "network_used": False,
        "abort_is_valid": True,
        "next_gate": "operator approval of checkpoint, candidate recipe, and any cap change before real training",
    }
    dump_structured(out_dir / "trainer_plan.json", trainer_plan)
    events = [
        {
            "ts": "example-only",
            "event": "start",
            "run_id": contract["trainer"]["training_task_id"],
            "caps": contract["trainer"]["caps"],
        },
        {
            "ts": "example-only",
            "event": "checkpoint",
            "step": contract["trainer"]["checkpoint_interval"]["steps"],
            "path": "checkpoints/example/step-100",
        },
        {"ts": "example-only", "event": "abort", "reason": "example_cap_breach", "weights_loaded": False},
    ]
    events_path = out_dir / "events.example.jsonl"
    with events_path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    cleanup = {
        "schema_version": "1.0",
        "run_id": contract["trainer"]["training_task_id"],
        "mode": "mock_evaluation_no_weight_load",
        "owned_pids": [],
        "stopped_owned_pids": [],
        "wsl_action": "not_started",
        "memory_before": {"available_mb": None, "committed_gb": None, "cache_gb": None, "standby_gb": None},
        "memory_after": {"available_mb": None, "committed_gb": None, "cache_gb": None, "standby_gb": None},
        "memory_audit": "not_sampled_because_no_training_process_or_weight_load_started",
        "gpu_before": {"available": False, "compute_apps": []},
        "gpu_after": {"available": False, "compute_apps": []},
        "top_by_private_commit": [],
        "top_by_working_set": [],
        "lingering_owned_pids": [],
        "global_standby_cache_purged": False,
        "cleanup_passed": True,
    }
    dump_structured(out_dir / "cleanup_summary.json", cleanup)
    summary = {
        "schema_version": "1.0",
        "run_id": contract["trainer"]["training_task_id"],
        "status": "mock_evaluation_completed",
        "abort_reason": None,
        "weights_loaded": False,
        "steps_completed": 0,
        "peak_ram_mb": 0.0,
        "avg_ram_mb": 0.0,
        "peak_io_mb_s": 0.0,
        "cpu_pct": 0.0,
        "checkpoints": [],
        "cleanup_passed": True,
        "events_sha256": sha256_file(events_path),
        "trainer_plan_hash": sha256_json(trainer_plan),
        "dataset_id": manifest.dataset_id,
        "claim_boundary": "Evaluation plumbing only; no trainable model or checkpoint was executed.",
    }
    dump_structured(out_dir / "summary.json", summary)
    return compare_candidates(
        out_dir / "frozen_contract.yaml",
        report_paths,
        candidates["constant_accept"],
        out_dir,
    )
