"""Finite student-transfer replay of verified proposals from a real Aleph checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path

from corpus import SEP, load_corpus


def key(transition, literal):
    op, old, value = transition
    return op, old if op or literal else None, value


def transition(prompt):
    marker, old, encoded_op, value, end = prompt[-5:]
    if marker != SEP or end != SEP or encoded_op not in (16, 17):
        raise ValueError("unrecognized public query")
    return encoded_op - 16, old, value


def target_trials():
    """Generate longer traces with a single disputed transition and independently replay them."""
    rng = random.Random(70613)
    cases = []
    for op, old, argument in product(range(2), range(4), range(4)):
        prefix = [
            ("replace", rng.randrange(4)) if rng.randrange(2) else ("advance", rng.randrange(4)) for _ in range(8)
        ]
        prefix.append(("replace", old))
        disputed = ("advance" if op else "replace", argument)
        state = rng.randrange(4)
        for action, operand in prefix:
            state = operand if action == "replace" else (state + operand) % 4
        correct = disputed[1] if disputed[0] == "replace" else (state + disputed[1]) % 4
        cases.append(((int(disputed[0] == "advance"), state, disputed[1]), correct))
    return cases


def scores(counts, targets, literal):
    acc = loss = Fraction()
    for t, gold in targets:
        votes = counts.get(key(t, literal), {})
        total = sum(votes.values())
        p = [Fraction(votes.get(i, 0), total) if total else Fraction(1, 4) for i in range(4)]
        best = [i for i, prob in enumerate(p) if prob == max(p)]
        acc += Fraction(int(gold in best), len(best))
        loss += sum((prob - int(i == gold)) ** 2 for i, prob in enumerate(p))
    return {"accuracy": float(acc / len(targets)), "brier": float(loss / len(targets))}


def run(pool, grader, targets, seed, policy, literal):
    remaining = list(pool)
    random.Random(seed).shuffle(remaining)
    weights = Counter(key(t, literal) for t, _ in targets)
    checked, pseudo = defaultdict(Counter), defaultdict(Counter)
    error_rate = defaultdict(lambda: [1, 2])
    log = []
    snapshots = {}
    for i in range(1, 33):

        def score(root):
            t, prediction = pool[root]
            if policy == "coverage":
                student_key = key(t, literal)
                return 0 if student_key in checked else weights[student_key]
            if policy == "error":
                n, mass = error_rate[t[0], prediction]
                return Fraction(n, mass)
            return 0

        root = remaining.pop(max(range(len(remaining)), key=lambda j: score(remaining[j])))
        t, prediction = pool[root]
        gold = grader(root)
        wrong = int(prediction != gold)
        stats = error_rate[t[0], prediction]
        stats[0] += wrong
        stats[1] += 1
        student_key = key(t, literal)
        checked[student_key][gold] += 1
        pseudo[student_key][prediction] += 1
        log.append({"root": root, "proposal": prediction, "verified": gold})
        if i in (8, 16, 32):
            snapshots[str(i)] = {
                "checked": scores(checked, targets, literal),
                "pseudo": scores(pseudo, targets, literal),
                "distinct_keys": len(checked),
                "teacher_errors": sum(r["proposal"] != r["verified"] for r in log),
                "records": list(log),
            }
    return snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.cell / "receipt.json").read_bytes())
    if data["checkpoint_sha256"] != "e71935b9bd9479249d48c31c0defec49ace4ed833e42c6fc0b99002ad5288c82":
        raise ValueError("unregistered teacher checkpoint")
    _, dev, source = load_corpus(args.pack)
    if data["config"]["source"] != source:
        raise ValueError("different evidence")
    predictions = {r["root"]: r["pred"] for r in data["rows"] if r["lane"] == "repair"}
    rows = {r.root: r for r in dev["repair"] if r.root in predictions}
    pool = {root: (transition(row.prompt), predictions[root]) for root, row in rows.items()}
    labels = {root: row.good[0] for root, row in rows.items()}
    targets = target_trials()
    assert len(pool) == 48 and len(set(targets)) == 32
    reference = dict(targets)
    assert all(reference[t] == labels[root] for root, (t, _) in pool.items())
    results = {}
    for literal in (False, True):
        student = "literal" if literal else "shared_set"
        results[student] = {}
        for policy in ("uniform", "error", "coverage"):
            trials = {str(s): run(pool, labels.__getitem__, targets, s, policy, literal) for s in (137, 163, 191)}
            results[student][policy] = trials
            print(
                student,
                policy,
                [
                    (r["16"]["checked"]["accuracy"], r["16"]["pseudo"]["accuracy"], r["16"]["teacher_errors"])
                    for r in trials.values()
                ],
            )
        # Coverage selection cannot inspect predictions. Substitute a constant
        # proposer and independently replay its choices and verified student score.
        blank_pool = {root: (t, 0) for root, (t, _) in pool.items()}
        for seed, original in results[student]["coverage"].items():
            blank = run(blank_pool, labels.__getitem__, targets, int(seed), "coverage", literal)
            for budget, observed in original.items():
                assert observed["checked"] == blank[budget]["checked"]
                assert [r["root"] for r in observed["records"]] == [r["root"] for r in blank[budget]["records"]]
    args.out.write_text(
        json.dumps(
            {
                "scope": "public finite transfer across renderers using frozen development proposals",
                "teacher_checkpoint_sha256": data["checkpoint_sha256"],
                "teacher_parameters": data["params"],
                "source": source,
                "available_teacher_proposals": len(pool),
                "target_transitions": len(targets),
                "no_data": scores({}, targets, False),
                "interpreter_accuracy": 1.0,
                "teacher_blind_coverage_equivalence": True,
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
