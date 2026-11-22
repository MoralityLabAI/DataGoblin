"""Re-evaluate the four frozen interaction arms on fresh revision roots."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Callable
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


async def run_replication(
    args: argparse.Namespace, family: str, protocol: dict, analyze: Callable[[list[dict], list[int]], dict],
    *, extra_excluded: set[str] | None = None,
) -> dict:
    started = time.perf_counter()
    commit, dirty = revision()
    if dirty:
        raise ValueError("Commit the replication protocol before opening the fresh reserve")
    lock_raw = (args.interaction / "locked_interaction.json").read_bytes()
    report_raw = (args.interaction / "frozen_interaction_report.json").read_bytes()
    if (digest(lock_raw), digest(report_raw)) != (
        protocol["interaction_lock_sha256"], protocol["interaction_report_sha256"],
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
    if extra_excluded is not None:
        excluded.update(extra_excluded)
    preflight = {
        "commit": commit, "protocol": protocol, "fit_receipts": lock["fits"],
        "matched_training": lock["matched_protocol"], "architecture": lock["architecture"],
        "optimizer": lock["optimizer"], "evaluation": lock["evaluation"],
        "excluded_source_hashes": sorted(excluded), "new_optimizer_updates": 0,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    lock_file = args.out / f"locked_{family}_replication.json"
    write_json(lock_file, preflight)

    def analyze_reserve(rows: list[dict]) -> dict:
        if {row["family"] for row in rows} != {family}:
            raise ValueError(f"Replication must score {family} roots only")
        return analyze(rows, seeds)

    result, _ = await evaluate(args.out, protocol["split"], protocol["roots"],
                               models, expected, excluded, analyze=analyze_reserve)
    report = {
        "locked_replication_sha256": digest(lock_file.read_bytes()), family: result,
        "evaluation_peak_mib": check_memory(LIMIT), "wall_seconds": time.perf_counter() - started,
    }
    write_json(args.out / f"frozen_{family}_replication.json", report)
    return report


def revision_metrics(rows: list[dict], seeds: list[int]) -> dict:
    result = factorial_metrics(rows, seeds)
    result["full_minus_joint"] = contrasts(
        rows, {"good": "good", JOINT: JOINT}, {**REPLICATION, "original_difference_rules": []},
    )[JOINT]
    return result


def replication_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    for argument in ("reference", "reference-checkpoints", "interaction", "prior-checkpoints", "checkpoints", "out"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    return parser


def configure_inference() -> None:
    install_memory_ceiling(LIMIT)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


async def run(args: argparse.Namespace) -> dict:
    return await run_replication(args, "revision", REPLICATION, revision_metrics)


def main() -> int:
    args = replication_parser(__doc__).parse_args()
    configure_inference()
    report = asyncio.run(run(args))
    print(json.dumps({key: report["revision"][key] for key in ("interaction", "full_minus_joint")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
