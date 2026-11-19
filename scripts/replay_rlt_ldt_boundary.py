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
from datagoblin.training_gym.rlt_route_certificate import observe_route, patched_endpoint


def replay(pack: Path, evidence_dir: Path, *, reuse_route=False):
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
    old_graphs = {}
    for split in SPLITS[1:]:
        task_path = pack / f"{split}.tasks.jsonl"
        if sha256_file(task_path) != fixture[split]["tasks_sha256"]:
            raise ValueError(f"rule tasks changed for {split}")
        rows = read_rows(task_path, {"id", "lineage", "links", "start", "steps", "patch"})
        tasks[split] = {}
        for item_id, task in rows.items():
            key = (task["lineage"], task["start"], task["steps"])
            links = dict(task["links"])
            old_rules = old_graphs.setdefault(key, links)
            if old_rules != links:
                raise ValueError("inconsistent prior graph for the same witness key")
            patch = task["patch"]
            tasks[split][item_id] = key, old_rules, {patch["source"]: patch["destination"]}
    counts = defaultdict(Counter)
    witness_cache = {}
    top = hybrid.CandidateState({"endpoint": frozenset(NODES)})
    soundness = hybrid.SoundnessType

    def verify(proposal, context):
        graph, epoch, count = context
        if proposal.metadata["epoch"] != epoch:
            return soundness.UNKNOWN
        key, links, edits = graph
        _, start, steps = key
        count["checker_route_hops"] += steps
        if reuse_route:
            if key not in witness_cache:
                witness_cache[key] = observe_route(links, start, steps)
            endpoint, suffix_steps = patched_endpoint(links, witness_cache[key], edits)
        else:
            endpoint, suffix_steps = follow(links | edits, start, steps), steps
        count["checker_traversed_hops"] += suffix_steps
        candidates = proposal.proposed_state.domains["endpoint"]
        return soundness.ENV_SOUND_DEAD if candidates == {endpoint} else soundness.UNKNOWN

    with outcomes.open(encoding="utf-8") as fh:
        for item in map(json.loads, fh):
            graph = tasks[item["split"]][item["id"]]
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
                        top, proposed, provenance_verifier=verify, verifier_context=(graph, 1, count),
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
        "route_verification": "cached_prefix" if reuse_route else "full_traversal",
        "witness_count": len(witness_cache),
        "witness_build_hops": sum(len(witness.nodes) - 1 for witness in witness_cache.values()),
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
    parser.add_argument("--reuse-route", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream.resolve()))
    dumps = [replay(args.pack, directory, reuse_route=args.reuse_route) for directory in args.evidence]
    dump_structured(args.out, {"schema": "datagoblin.rlt-ldt-boundary.v1", "replays": dumps})


if __name__ == "__main__":
    main()
