"""Replay neural rule-change proposals through the upstream strict LDT membrane."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured, load_structured
from datagoblin.training_gym.rlt_revision import NODES, SPLITS, follow, read_rows


def replay(pack: Path, evidence_dir: Path):
    from research_gym.core import hybrid

    source = load_structured(evidence_dir / "evidence.json")
    protocol = source["protocol"]
    outcomes = evidence_dir / "item-outcomes.jsonl"
    manifest = pack / "manifest.json"
    if sha256_file(manifest) != protocol["rule_manifest_sha256"]:
        raise ValueError("rule pack does not match scored evidence")
    if sha256_file(outcomes) != source["artifact_sha256"][outcomes.name]:
        raise ValueError("outcome hash differs from scored evidence")
    fixture = load_structured(manifest)["splits"]
    tasks = {}
    for split in SPLITS[1:]:
        task_path = pack / f"{split}.tasks.jsonl"
        if sha256_file(task_path) != fixture[split]["tasks_sha256"]:
            raise ValueError(f"rule tasks changed for {split}")
        tasks[split] = read_rows(task_path, {"id", "lineage", "links", "start", "steps", "patch"})
    counts = defaultdict(Counter)
    top = hybrid.CandidateState({"endpoint": frozenset(NODES)})
    soundness = hybrid.SoundnessType

    def verify(proposal, context):
        task, epoch, count = context
        if proposal.metadata["epoch"] != epoch:
            return soundness.UNKNOWN
        count["checker_route_hops"] += task["steps"]
        patch = task["patch"]
        links = dict(task["links"]) | {patch["source"]: patch["destination"]}
        endpoint = follow(links, task["start"], task["steps"])
        candidates = proposal.proposed_state.domains["endpoint"]
        return soundness.ENV_SOUND_DEAD if candidates == {endpoint} else soundness.UNKNOWN

    with outcomes.open(encoding="utf-8") as fh:
        for item in map(json.loads, fh):
            task = tasks[item["split"]][item["id"]]
            for mode in protocol["modes"]:
                result = item["results"][f"{mode}/{mode}"]
                count = counts[item["seed"], item["depth"], item["split"], mode]
                for turn, epoch in (("before", 0), ("after", 1)):
                    proposed = hybrid.LatticeProposal(
                        hybrid.CandidateState({"endpoint": frozenset({result[turn]})}),
                        soundness.MODEL_SOUND_DEAD,
                        metadata={"epoch": epoch},
                    )
                    decision = hybrid.certify_and_apply_verified(
                        top, proposed, provenance_verifier=verify, verifier_context=(task, 1, count),
                    )
                    expected = epoch == 1 and result["after_ok"]
                    result_state = proposed.proposed_state if expected else top
                    if decision.accepted != expected or decision.after != result_state:
                        raise ValueError("membrane update disagrees with the independent scored outcome")
                    count["previous_epoch_rejected" if epoch == 0 else "checked_current"] += 1
                    if epoch == 1:
                        count["committed" if decision.accepted else "abstained"] += 1
    return {
        "placement": protocol.get("placement", "reasoning"),
        "outcomes_sha256": sha256_file(outcomes),
        "rule_manifest_sha256": sha256_file(manifest),
        "upstream_membrane_sha256": sha256_file(Path(hybrid.__file__)),
        "cells": [
            {"seed": seed, "depth": depth, "split": split, "mode": mode, **count}
            for (seed, depth, split, mode), count in sorted(counts.items())
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, nargs="+", required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream.resolve()))
    dump_structured(args.out, {"schema": "datagoblin.rlt-ldt-boundary.v1",
                              "replays": [replay(args.pack, directory) for directory in args.evidence]})


if __name__ == "__main__":
    main()
