"""Purchase six generated repair labels after the frozen finite-pool experiment."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from corpus import generate_repair, load_corpus
from gym import file_sha
from repair_transfer import key, scores, target_trials, transition
from study import dense, tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    history = json.loads(args.transfer.read_bytes())
    if file_sha(args.cell / "checkpoint.pt") != history["teacher_checkpoint_sha256"]:
        raise ValueError("different teacher")
    train, dev, source = load_corpus(args.pack)
    if source != history["source"]:
        raise ValueError("changed corpus")
    source_rows = {r.root: r for r in dev["repair"]}
    excluded = set(source_rows) | {r.root for r in train["repair"]}
    rng = random.Random(405113)
    roster = {}
    for _ in range(10000):
        row = generate_repair(rng)
        if row.root not in excluded:
            roster.setdefault(transition(row.prompt), row)
        if len(roster) == 32:
            break
    if len(roster) != 32 or len({r.root for r in roster.values()}) != 32:
        raise RuntimeError("failed to generate distinct full transition panel")
    rows = list(roster.values())
    state = torch.load(args.cell / "checkpoint.pt", map_location="cpu", weights_only=True)
    cfg = state["cfg"]
    teacher = dense(cfg["width"])
    teacher.load_state_dict(state["model"])
    teacher.eval()
    with torch.inference_mode():
        x, lengths = tensor(rows)
        logits = teacher(x, cfg["arm"], cfg["r"])
        predictions = {r.root: int(logits[i, int(lengths[i]) - 1, :4].argmax()) for i, r in enumerate(rows)}
    targets = target_trials()
    target_labels = dict(targets)
    out = {}
    for seed, original in history["results"]["shared_set"]["coverage"].items():
        baseline = defaultdict(Counter)
        for receipt in original["16"]["records"]:
            baseline[key(transition(source_rows[receipt["root"]].prompt), False)][receipt["verified"]] += 1
        baseline_score = scores(baseline, targets, False)
        if baseline_score != original["16"]["checked"]:
            raise ValueError("student reconstruction failed")
        order = list(rows)
        random.Random(int(seed)).shuffle(order)
        out[seed] = {}
        for policy in ("uniform", "uncovered"):
            chosen = (
                order[:6]
                if policy == "uniform"
                else [row for row in order if key(transition(row.prompt), False) not in baseline]
            )
            if len(chosen) != 6:
                raise ValueError("changed support of original pool")
            learned = defaultdict(Counter, {k: v.copy() for k, v in baseline.items()})
            records = []
            for row in chosen:
                t = transition(row.prompt)
                correction = target_labels[t]
                if row.good != (correction,):
                    raise ValueError("independent transition verifier disagrees")
                learned[key(t, False)][correction] += 1
                records.append(
                    {
                        "root": row.root,
                        "transition": t,
                        "prompt": row.prompt,
                        "proposal": predictions[row.root],
                        "verified": correction,
                    }
                )
            out[seed][policy] = {
                "score": scores(learned, targets, False),
                "records": records,
                "teacher_errors": sum(r["proposal"] != r["verified"] for r in records),
            }
            print(seed, policy, out[seed][policy]["score"], "errors", out[seed][policy]["teacher_errors"])
    args.out.write_text(
        json.dumps(
            {
                "scope": "new generated challenges in the same finite transition language",
                "pinned_teacher": history["teacher_checkpoint_sha256"],
                "original_pack": source,
                "generated_proposals": len(rows),
                "new_checks_per_policy": 6,
                "original_checks": 16,
                "results": out,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
