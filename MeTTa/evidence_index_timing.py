"""Repeat matched host CPU timings without the native audit or file I/O."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median

from .evidence_index import ProofIndex
from .evidence_index_pilot import PILOT_FAMILIES, stress_episode
from .evidence_proofs import Snapshot


def timed(episodes: list[tuple[Snapshot, ...]], carry: bool, repeats: int) -> tuple[float, int]:
    start = time.process_time()
    proof_count = 0
    for _ in range(repeats):
        for snapshots in episodes:
            retained = ProofIndex() if carry else None
            for snapshot in snapshots:
                index = retained if retained is not None else ProofIndex()
                summary, _ = index.evaluate(snapshot)
                proof_count += len(summary["proofs"])
    return time.process_time() - start, proof_count


def measure(episodes: list[tuple[Snapshot, ...]]) -> dict:
    times: dict[str, list[float]] = {"warm": [], "cold": []}
    repetitions, rounds = 80, 7
    for run in range(rounds):
        counts = set()
        for carry in (bool(run % 2), not run % 2):
            seconds, count = timed(episodes, carry, repetitions)
            times["warm" if carry else "cold"].append(seconds)
            counts.add(count)
        if len(counts) != 1:
            raise ValueError("Timed arms returned different proof counts")
    return {"episodes": len(episodes), "rounds": rounds, "repetitions_per_round": repetitions,
            "cpu_seconds": times, "median_seconds": {arm: median(values) for arm, values in times.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    episodes = {family: [stress_episode(family, seed) for seed in range(6)] for family in PILOT_FAMILIES}
    result = {"all_families": measure([episode for rows in episodes.values() for episode in rows]),
              "wide_alternatives": measure(episodes["wide_alternatives"])}
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({name: row["median_seconds"] for name, row in result.items()}))


if __name__ == "__main__":
    main()
