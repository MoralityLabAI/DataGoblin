"""Reconcile the archived native courier pilot without fitting or scoring a policy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .experiment import file_sha
from .native import load_storyworld

HISTORICAL_PROTOCOL = "5697d37f0adb782d1829ecd545e2afa92263434a"
ARCHIVE_HASHES = {
    "locked_protocol.json": "3690627b0139d4dc3de4d823427b5cbcb0137e94d898a7a067cb0d31a6102030",
    "locked_fits.json": "8cce365056baf9d1484064dfa7f4398d93f11a180987bf52bc75f70e22161b3b",
    "training_frames.jsonl": "afdf248db150394f90c364c6a29ec437a00fb15df9c5c93b418fa21fd5d8fd9f",
    "training_trajectories.jsonl": "392f3d398de0be558d76640ca982606a92b7ab13f9d7219d5b56490c25d3758a",
    "frozen_receipts.jsonl": "88419586c30ce3d750f8be2bc81b07b635747242c5620ad19f39f7d3d2be1b79",
}


def audit(archive: Path, storyworld: Path) -> dict:
    from MeTTa.storyworld_courier_pilot import analyze

    upstream, hashes = load_storyworld(storyworld)
    if any(file_sha(archive / name) != checksum for name, checksum in ARCHIVE_HASHES.items()):
        raise ValueError("Historical archive differs from the published receipt hashes")

    def read(name: str):
        return json.loads((archive / name).read_text(encoding="utf-8"))

    def rows(name: str):
        return [json.loads(line) for line in (archive / name).read_text(encoding="utf-8").splitlines()]

    lock = read("locked_protocol.json")
    if lock["commit"] != HISTORICAL_PROTOCOL or lock["storyworld_sources"] != {
        k.removeprefix("metta-storyworld/"): v for k, v in hashes.items()
    }:
        raise ValueError("Historical source provenance differs")
    expected = {"train": 200, "frozen": 144}
    native = upstream.NativeCourier()
    worlds = {}
    for split, size in expected.items():
        if len(lock["cohorts"][split]) != size:
            raise ValueError("Historical split sizes differ")
        worlds[split] = {}
        for root in lock["cohorts"][split]:
            raw = root["world"]
            world = upstream.World(
                tuple(map(tuple, raw["arcs"])), tuple(raw["offices"]), tuple(raw["depots"]), raw["desk"], raw["entry"]
            )
            if world.source_sha256 != root["source_sha256"] or world.source_sha256 in worlds[split]:
                raise ValueError("Historical world serialization or duplicate check failed")
            worlds[split][world.source_sha256] = world
    if worlds["train"].keys() & worlds["frozen"].keys():
        raise ValueError("Historical roots overlap")
    replays = {"train": rows("training_trajectories.jsonl"), "frozen": rows("frozen_receipts.jsonl")}
    if len(replays["train"]) != 400 or len(replays["frozen"]) != 1440:
        raise ValueError("Historical replay totals differ")
    for split, records in replays.items():
        for row in records:
            checked = native.verify_plan(worlds[split][row["source_sha256"]], row["moves"])
            if any(row[key] != value for key, value in checked.items()):
                raise ValueError("Historical native proof receipt differs")
            if split == "frozen" and row["verified_completion"] != (row["stopped"] and checked["delivered"]):
                raise ValueError("Historical completion endpoint differs")

    fits = read("locked_fits.json")
    if (
        len(fits) != 9
        or len({r["checkpoint_sha256"] for r in fits}) != 9
        or any(
            r["parameters"] != 44417 or r["optimizer_updates"] != 180 or r["supervised_action_tokens"] != 8640
            for r in fits
        )
        or any(len({r["initial_sha256"] for r in fits if r["seed"] == seed}) != 1 for seed in (1101, 1102, 1103))
    ):
        raise ValueError("Historical fit identities or matched budgets differ")
    frames = rows("training_frames.jsonl")
    tally = Counter(len(row["labels"]["action_set"]) > 1 for row in frames)
    if len(frames) != 1452 or tally[True] != 132:
        raise ValueError("Historical training accounting differs")
    results = analyze(replays["frozen"])
    archived = read("courier_report.json")
    if any(archived[key] != value for key, value in results.items()):
        raise ValueError("Historical outcome or interval calculation differs")
    return {
        "archive_hashes": ARCHIVE_HASHES,
        "source_hashes": hashes,
        "replays": native.replay_calls,
        "fit_digests": {f"{row['arm']}/{row['seed']}": row["checkpoint_sha256"] for row in fits},
        "checkpoint_files_available": False,
        **results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--storyworld", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive, args.storyworld), indent=2))
