"""Export Lean proposal episodes with separated public inputs and verified labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import canonical_json, load_structured

from .rlt_lean_labels import atomic_record
from .rlt_lean_ldt import AXIOMS

SCHEMA = "datagoblin.checked-candidate-pack.v1"
HARD_CAP = 256 * 1024 * 1024


def checked_rows(trace, environment):
    goal = trace["goal"]
    epoch = sha256_json([goal["id"], environment])
    outcomes = trace["outcomes"]
    proposals = {}
    receipts = {}
    for row in outcomes:
        tactic = row["tactic"]
        verified = (
            row.get("closed") is True and row.get("kernel_ok") is True
            and row["kernel_calls"] == 1 and set(row["axioms"]) <= AXIOMS
        )
        label = row["label"]
        if label not in (0, 1, None) or (label == 1) != verified or tactic in proposals:
            raise ValueError("inconsistent or duplicate Lean proposal result")
        candidate_id = sha256_json([epoch, tactic])
        proposals[tactic] = {"id": candidate_id, "action": "lean_tactic", "tactic": tactic}
        receipts[candidate_id] = {k: v for k, v in row.items() if k != "tactic"}
    pool_union = {tactic for pool in trace["pools"].values() for tactic in pool}
    if pool_union != proposals.keys():
        raise ValueError("proposal pool and independent checks disagree")
    public = {
        "id": goal["id"], "lineage": goal["lineage"], "family": goal["family"], "epoch": epoch,
        "task": "lean_candidate_proof", "statement": goal["statement"], "context": trace["goal_pp"],
        "retrieved": trace["retrieved"], "candidates": list(proposals.values()),
        "orders": {arm: [proposals[t]["id"] for t in order] for arm, order in trace["pools"].items()},
    }
    private = {
        "id": goal["id"], "epoch": epoch, "outcomes": receipts, "native": trace["native"],
        "retrieval_comparisons": trace["retrieval_comparisons"], "generation_wall_s": trace["generation_wall_s"],
    }
    return public, private


def export(run, out):
    evidence = load_structured(run / "evidence.json")
    protocol = load_structured(run / "protocol.json")
    if evidence["protocol_sha256"] != sha256_file(run / "protocol.json"):
        raise ValueError("the proof audit is for a different protocol")
    if protocol["fixture_sha256"] != sha256_file(run / "fixture.json"):
        raise ValueError("the frozen goal pack changed")
    history = [json.loads(line) for line in (run / "resource-history.jsonl").read_text(encoding="utf-8").splitlines()]
    if any(row["hard_commit_cap_bytes"] != HARD_CAP or row["job_peak_commit_bytes"] > HARD_CAP for row in history):
        raise ValueError("resource history exceeds the registered cap")
    terminal = {row["phase"]: row["status"] for row in history}
    if terminal.get("evaluate") != "complete" or terminal.get("certify") != "complete":
        raise ValueError("evaluation and replay must complete before export")
    environment = {key: protocol[key] for key in ("lean_sha256", "repl_sha256", "gate_sha256", "membrane_sha256")}
    splits = load_structured(run / "fixture.json")["splits"]
    all_goals, all_lineages, receipts = set(), set(), {}
    for split, goals in splits.items():
        ids = {goal["id"] for goal in goals}
        lineages = {goal["lineage"] for goal in goals}
        if len(ids) != len(goals) or ids & all_goals or lineages & all_lineages:
            raise ValueError("duplicate goal or crossed lineage boundary")
        all_goals.update(ids)
        all_lineages.update(lineages)
        out.mkdir(parents=True, exist_ok=True)
        public = out / f"{split}.public.jsonl"
        labels = out / f"{split}.receipts.jsonl"
        pending = [path.with_suffix(".part.jsonl") for path in (public, labels)]
        with pending[0].open("w", encoding="utf-8", newline="\n") as inputs, pending[1].open(
            "w", encoding="utf-8", newline="\n",
        ) as checked:
            for goal in goals:
                path = run / "traces" / f"{goal['id']}.json"
                if sha256_file(path) != evidence["traces_sha256"][goal["id"]]:
                    raise ValueError("a certified trace changed")
                trace = load_structured(path)
                if goal != trace["goal"] or split != trace["split"]:
                    raise ValueError("trace split or identity differs from the fixture")
                scan, labels_row = checked_rows(trace, environment)
                inputs.write(canonical_json(scan) + "\n")
                checked.write(canonical_json(labels_row) + "\n")
        for temp, final in zip(pending, (public, labels), strict=True):
            temp.replace(final)
        receipts[split] = {
            "goals": len(goals), "lineages": len(lineages), "public": public.name,
            "public_sha256": sha256_file(public), "receipts": labels.name, "receipts_sha256": sha256_file(labels),
        }
    if all_goals != evidence["traces_sha256"].keys():
        raise ValueError("unregistered traces in proof evidence")
    atomic_record(out / "manifest.json", {
        "schema": SCHEMA, "splits": receipts, "environment": environment,
        "source_evidence_sha256": sha256_file(run / "evidence.json"),
        "resource_history_sha256": sha256_file(run / "resource-history.jsonl"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    export(args.run, args.out)


if __name__ == "__main__":
    main()
