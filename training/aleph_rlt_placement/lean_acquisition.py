"""Public development replay of three cost-matched agonist data policies."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from corpus import SEP, identity, load_corpus


def bucket(row):
    tactic = row.prompt[row.prompt.index(SEP) + 1 : -1]
    return (tactic[0] if tactic else -1, min(3, len(tactic) // 4))


def evaluate(rows, counts):
    goals = defaultdict(list)
    squared = []
    for row in rows:
        positive, mass = counts[bucket(row)]
        p = positive / mass
        goals[row.root].append((-p, row.order, row.good))
        if row.good:
            squared.append((p - row.good[0]) ** 2)
    solved = sum(any(grade == (1,) for _, _, grade in sorted(pool)[:4]) for pool in goals.values())
    return {"brier": sum(squared) / len(squared), "graded": len(squared), "proved": solved, "goals": len(goals)}


def acquire(train, dev, seed, policy):
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    keys = [bucket(row) for row in train]
    weights = Counter(keys)
    counts = defaultdict(lambda: [1, 2])
    queried = []
    rejected = 0
    unknown = 0
    snapshots = {"0": evaluate(dev, counts)}
    for i in range(1, 97):
        if policy == "uniform":
            offset = 0
        else:
            scores = {}
            for key, weight in weights.items():
                positive, mass = counts[key]
                p = positive / mass
                scores[key] = 1 - p if policy == "rejection" else weight * p * (1 - p) / (mass + 1) ** 2
            offset = max(range(len(order)), key=lambda j: scores[keys[order[j]]])
        index = order.pop(offset)
        row = train[index]
        queried.append((row.root, row.order))
        if row.good:
            counts[keys[index]][0] += row.good[0]
            counts[keys[index]][1] += 1
            rejected += row.good == (0,)
        else:
            unknown += 1
        if i in (8, 32, 96):
            snapshots[str(i)] = {
                **evaluate(dev, counts),
                "train_rejected": rejected,
                "train_unknown": unknown,
                "query_digest": identity(queried),
                "query_ids": list(queried),
            }
    return snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    train, held, source = load_corpus(args.pack)
    train, held = train["lean"], held["lean"]
    if {row.root for row in train} & {row.root for row in held}:
        raise ValueError("overlapping proof goals")
    results = {
        policy: {str(seed): acquire(train, held, seed, policy) for seed in (137, 163, 191)}
        for policy in ("uniform", "rejection", "target")
    }
    out = {
        "source": source,
        "scope": "public development with cached kernel outcomes",
        "student": "Beta(1,1) per tactic-prefix and length bin, goal-independent",
        "seeds": [137, 163, 191],
        "families": len({row.group for row in held}),
        "train_goals": len({row.root for row in train}),
        "results": results,
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for policy, trials in results.items():
        for budget in ("0", "8", "32", "96"):
            rows = [trial[budget] for trial in trials.values()]
            print(
                policy,
                budget,
                "Brier",
                round(sum(r["brier"] for r in rows) / len(rows), 5),
                "proved",
                [r["proved"] for r in rows],
                "rejected",
                [r.get("train_rejected", 0) for r in rows],
            )


if __name__ == "__main__":
    main()
