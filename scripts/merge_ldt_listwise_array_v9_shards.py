#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_sorter import sha256_file
from datagoblin.serialization import dump_structured, load_structured


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def merge(
    *,
    preregistration_path: Path,
    dataset_manifest_path: Path,
    shard_runs: Sequence[Path],
    out_dir: Path,
) -> dict[str, Any]:
    preregistration = load_structured(preregistration_path)
    if preregistration["schema_version"] != "9.0" or preregistration["status"] != "frozen_before_dataset_generation":
        raise ValueError("shard merge requires the frozen v9 contract")
    expected = {
        (str(model["model_id"]), int(seed))
        for model in preregistration["models"]["candidates"]
        for seed in preregistration["training"]["seeds"]
    }
    if not shard_runs or len(shard_runs) > len(expected):
        raise ValueError("v9 shard count is empty or excessive")
    manifest = load_structured(dataset_manifest_path)
    heldout_pair_path = dataset_manifest_path.parent / str(manifest["splits"]["heldout_pairs"]["path"])
    heldout_array_path = dataset_manifest_path.parent / str(manifest["splits"]["heldout_arrays"]["path"])
    if (
        sha256_file(heldout_pair_path) != manifest["splits"]["heldout_pairs"]["sha256"]
        or sha256_file(heldout_array_path) != manifest["splits"]["heldout_arrays"]["sha256"]
    ):
        raise ValueError("public heldout inputs differ from the manifest")
    heldout_pair_ids = {str(row["example_id"]) for row in _read_jsonl(heldout_pair_path)}
    heldout_array_ids = {str(row["array_id"]) for row in _read_jsonl(heldout_array_path)}
    if len(heldout_pair_ids) != int(manifest["splits"]["heldout_pairs"]["count"]) or len(heldout_array_ids) != int(
        manifest["splits"]["heldout_arrays"]["count"]
    ):
        raise ValueError("public heldout inputs contain duplicate identities")

    summaries = []
    pair_rows = []
    array_rows = []
    source_receipts = []
    observed = set()
    dataset_hash = None
    expected_dataset_hash = sha256_file(dataset_manifest_path)
    preregistration_hash = sha256_file(preregistration_path)
    for run_dir in shard_runs:
        run_receipt_path = run_dir / "run_receipt.json"
        trainer_path = run_dir / "capped_run/trainer_summary.json"
        pair_path = run_dir / "capped_run/heldout_pair_predictions.jsonl"
        array_path = run_dir / "capped_run/heldout_array_predictions.jsonl"
        run_receipt = _read_json(run_receipt_path)
        trainer = _read_json(trainer_path)
        if (
            int(run_receipt["capped_process_exit_code"]) != 0
            or int(run_receipt["wsl_stage_cleanup_exit_code"]) != 0
            or not bool(run_receipt["caps_receipt"]["caps_enforced"])
            or run_receipt["caps_receipt"]["status"] != "completed"
            or int(run_receipt["caps_receipt"]["exit_code"]) != 0
            or not bool(run_receipt["cleanup_receipt"]["cleanup_passed"])
            or bool(run_receipt["heldout_pair_labels_staged"])
            or bool(run_receipt["heldout_array_labels_staged"])
            or bool(run_receipt["external_checkpoint_loaded"])
            or trainer["status"] != "completed"
            or not trainer["hard_caps_verified"]
            or trainer["heldout_pair_targets_seen"]
            or trainer["heldout_array_targets_seen"]
            or trainer["external_checkpoint_loaded"]
        ):
            raise ValueError(f"shard lacks cap, cleanup, or custody proof: {run_dir}")
        if (
            run_receipt["artifacts"]["trainer_summary.json"] != sha256_file(trainer_path)
            or run_receipt["artifacts"]["heldout_pair_predictions.jsonl"] != sha256_file(pair_path)
            or run_receipt["artifacts"]["heldout_array_predictions.jsonl"] != sha256_file(array_path)
            or trainer["predictions_sha256"] != sha256_file(pair_path)
            or trainer["array_predictions_sha256"] != sha256_file(array_path)
            or trainer["preregistration_sha256"] != preregistration_hash
        ):
            raise ValueError(f"shard artifact hash differs: {run_dir}")
        shard_filter = trainer.get("shard_filter")
        receipts = trainer["candidate_receipts"]
        if not shard_filter or not receipts:
            raise ValueError("trainer output is not a filtered shard")
        candidate_id = str(shard_filter["candidate_id"])
        filtered_seed = shard_filter["seed"]
        receipt_keys = {
            (
                str(receipt["candidate_id"]),
                int(receipt["seed"]),
            )
            for receipt in receipts
        }
        expected_run_keys = {
            key
            for key in expected
            if key[0] == candidate_id and (filtered_seed is None or key[1] == int(filtered_seed))
        }
        if (
            len(receipts) != len(receipt_keys)
            or receipt_keys != expected_run_keys
            or not receipt_keys
            or not receipt_keys.isdisjoint(observed)
        ):
            raise ValueError("trainer shard keys are duplicate or unexpected")
        observed.update(receipt_keys)
        if dataset_hash is None:
            dataset_hash = trainer["dataset_manifest_sha256"]
        elif dataset_hash != trainer["dataset_manifest_sha256"]:
            raise ValueError("trainer shards used different datasets")
        shard_pair_rows = _read_jsonl(pair_path)
        shard_array_rows = _read_jsonl(array_path)
        shard_pair_keys = {
            (
                str(row["candidate_id"]),
                int(row["seed"]),
                str(row["example_id"]),
            )
            for row in shard_pair_rows
        }
        shard_array_keys = {
            (
                str(row["candidate_id"]),
                int(row["seed"]),
                str(row["array_id"]),
            )
            for row in shard_array_rows
        }
        expected_pair_keys = {(*key, example_id) for key in receipt_keys for example_id in heldout_pair_ids}
        expected_array_keys = {(*key, array_id) for key in receipt_keys for array_id in heldout_array_ids}
        if (
            len(shard_pair_rows) != len(shard_pair_keys)
            or shard_pair_keys != expected_pair_keys
            or len(shard_array_rows) != len(shard_array_keys)
            or shard_array_keys != expected_array_keys
        ):
            raise ValueError(f"shard predictions are incomplete or duplicated: {run_dir}")
        if any(
            row["schema_version"] != "ldt-array-confirmation-prediction.v3"
            or row["authority"] != "MODEL_RECOMMENDATION_ONLY"
            or row["dataset_manifest_sha256"] != expected_dataset_hash
            or any(
                key in row
                for key in (
                    "preference_label",
                    "rationale_label",
                    "target",
                )
            )
            for row in shard_pair_rows
        ) or any(
            row["schema_version"] != "ldt-listwise-array-prediction.v9"
            or row["authority"] != "MODEL_RECOMMENDATION_ONLY"
            or row["dataset_manifest_sha256"] != expected_dataset_hash
            or any(key in row for key in ("status_labels", "targets", "target"))
            for row in shard_array_rows
        ):
            raise ValueError(f"shard predictions violate the public claim boundary: {run_dir}")
        summaries.append(trainer)
        pair_rows.extend(shard_pair_rows)
        array_rows.extend(shard_array_rows)
        source_receipts.append(
            {
                "candidate_id": candidate_id,
                "seeds": sorted(key[1] for key in receipt_keys),
                "run_receipt_sha256": sha256_file(run_receipt_path),
                "trainer_summary_sha256": sha256_file(trainer_path),
                "cleanup_passed": True,
                "caps_enforced": True,
            }
        )
    if observed != expected:
        raise ValueError("v9 shard coverage is incomplete")
    if dataset_hash != expected_dataset_hash:
        raise ValueError("v9 shards are bound to another public dataset")

    out_dir.mkdir(parents=True, exist_ok=False)
    pair_path = out_dir / "heldout_pair_predictions.jsonl"
    array_path = out_dir / "heldout_array_predictions.jsonl"
    _write_jsonl(
        pair_path,
        sorted(
            pair_rows,
            key=lambda row: (
                str(row["candidate_id"]),
                int(row["seed"]),
                str(row["example_id"]),
            ),
        ),
    )
    _write_jsonl(
        array_path,
        sorted(
            array_rows,
            key=lambda row: (
                str(row["candidate_id"]),
                int(row["seed"]),
                str(row["array_id"]),
            ),
        ),
    )
    combined = {
        "schema_version": "ldt-array-confirmation-trainer-summary.v3",
        "experiment_id": preregistration["experiment_id"],
        "training_task_id": preregistration["training_task_id"],
        "status": "completed",
        "abort_reason": None,
        "hard_caps_verified": True,
        "dataset_manifest_sha256": dataset_hash,
        "preregistration_sha256": preregistration_hash,
        "encoder": summaries[0]["encoder"],
        "candidate_receipts": sorted(
            (receipt for summary in summaries for receipt in summary["candidate_receipts"]),
            key=lambda row: (
                str(row["candidate_id"]),
                int(row["seed"]),
            ),
        ),
        "steps_completed": sum(int(summary["steps_completed"]) for summary in summaries),
        "checkpoints": [],
        "shard_checkpoint_count": sum(len(summary["checkpoints"]) for summary in summaries),
        "predictions_sha256": sha256_file(pair_path),
        "array_predictions_sha256": sha256_file(array_path),
        "heldout_pair_targets_seen": False,
        "heldout_array_targets_seen": False,
        "external_checkpoint_loaded": False,
        "elapsed_seconds_sum": sum(float(summary["elapsed_seconds"]) for summary in summaries),
        "shard_filter": None,
        "source_shards": sorted(
            source_receipts,
            key=lambda row: (
                str(row["candidate_id"]),
                tuple(int(seed) for seed in row["seeds"]),
            ),
        ),
        "claim_boundary": (
            "Merged target-free v9 shard evidence only; every source ran "
            "under verified caps and passed PID-scoped cleanup."
        ),
    }
    summary_path = out_dir / "trainer_summary.json"
    dump_structured(summary_path, combined)
    receipt = {
        "schema_version": "ldt-listwise-array-shard-merge.v9",
        "experiment_id": preregistration["experiment_id"],
        "source_shard_count": len(source_receipts),
        "complete_candidate_seed_coverage": True,
        "all_caps_enforced": True,
        "all_cleanup_passed": True,
        "heldout_pair_targets_seen": False,
        "heldout_array_targets_seen": False,
        "trainer_summary_sha256": sha256_file(summary_path),
        "pair_predictions_sha256": sha256_file(pair_path),
        "array_predictions_sha256": sha256_file(array_path),
    }
    dump_structured(out_dir / "merge_receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge all hard-capped v9 candidate/seed shards")
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--shard-run",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    result = merge(
        preregistration_path=arguments.preregistration.resolve(),
        dataset_manifest_path=(arguments.dataset_manifest.resolve()),
        shard_runs=[path.resolve() for path in arguments.shard_run],
        out_dir=arguments.out.resolve(),
    )
    print(json.dumps(result, sort_keys=True))
