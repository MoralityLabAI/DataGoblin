"""Re-evaluate the four frozen interaction arms on fresh revision roots."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import torch

from MeTTa.ram_hillclimb import digest, install_memory_ceiling
from MeTTa.ram_hillclimb_trajectory import check_memory

from ..experiment import write_json
from ..taskset import REVISION_REPLICATION_ROOTS, REVISION_REPLICATION_SPLIT
from .__main__ import LIMIT, evaluate, revision
from .ablation import load_model, read_reference
from .ablation_stats import contrasts
from .interaction import ARMS, JOINT, PROTOCOL, factorial_metrics

REPLICATION = {
    "split": REVISION_REPLICATION_SPLIT,
    "roots": REVISION_REPLICATION_ROOTS,
    "primary_contrast": PROTOCOL["primary_contrast"],
    "secondary_contrast": "good - without_lineage_revision",
    "bootstrap_draws": PROTOCOL["bootstrap_draws"],
    "sign_flip_draws": PROTOCOL["sign_flip_draws"],
    "interaction_lock_sha256": "336af481a214adc26dcfcb9c2f77cbf6783843b91bbf20cc8247622cb27821cb",
    "interaction_report_sha256": "bb37f90a381bed89419ea0e1c40bf641589a8f897e97aa0dc242d44515a4f4f0",
}


async def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    commit, dirty = revision()
    if dirty:
        raise ValueError("Commit the replication protocol before opening the fresh reserve")
    lock_raw = (args.interaction / "locked_interaction.json").read_bytes()
    report_raw = (args.interaction / "frozen_interaction_report.json").read_bytes()
    if (digest(lock_raw), digest(report_raw)) != (
        REPLICATION["interaction_lock_sha256"], REPLICATION["interaction_report_sha256"],
    ):
        raise ValueError("Replication requires the exact archived interaction run")
    lock, archived = json.loads(lock_raw), json.loads(report_raw)
    original, original_raw = read_reference(args.reference)
    if lock["reference_lock_sha256"] != digest(original_raw) or lock["matched_protocol"] != original["protocol"]:
        raise ValueError("Original runtime and matched training protocol must agree with the archive")

    seeds = lock["matched_protocol"]["seeds"]
    models, expected = {}, {}
    for row in lock["fits"]:
        key = (row["curator"], row["seed"])
        models[key] = load_model(getattr(args, row["location"]), row)
        expected[key] = row["checkpoint_sha256"]
    if set(models) != {(arm, seed) for arm in ARMS for seed in seeds} or len(models) != len(lock["fits"]):
        raise ValueError("Every arm must use all of its archived checkpoints")
    excluded = set(lock["excluded_source_hashes"])
    for split in (archived["primary"], archived["conflict_transfer"]):
        excluded.update(row["source_sha256"] for row in split["sources"])
    preflight = {
        "commit": commit, "protocol": REPLICATION, "fit_receipts": lock["fits"],
        "matched_training": lock["matched_protocol"], "architecture": lock["architecture"],
        "optimizer": lock["optimizer"], "evaluation": lock["evaluation"],
        "excluded_source_hashes": sorted(excluded), "new_optimizer_updates": 0,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    lock_file = args.out / "locked_revision_replication.json"
    write_json(lock_file, preflight)

    def analyze(rows: list[dict]) -> dict:
        result = factorial_metrics(rows, seeds)
        if set(result["family_safe_completions"]) != {"revision"}:
            raise ValueError("Replication must score revision roots only")
        result["full_minus_joint"] = contrasts(
            rows, {"good": "good", JOINT: JOINT}, {**REPLICATION, "original_difference_rules": []},
        )[JOINT]
        return result

    result, _ = await evaluate(args.out, REPLICATION["split"], REPLICATION["roots"],
                               models, expected, excluded, analyze=analyze)
    report = {
        "locked_replication_sha256": digest(lock_file.read_bytes()), "revision": result,
        "evaluation_peak_mib": check_memory(LIMIT), "wall_seconds": time.perf_counter() - started,
    }
    write_json(args.out / "frozen_revision_replication.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("reference", "reference-checkpoints", "interaction", "prior-checkpoints", "checkpoints", "out"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args()
    install_memory_ceiling(LIMIT)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    report = asyncio.run(run(args))
    print(json.dumps({key: report["revision"][key] for key in ("interaction", "full_minus_joint")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
