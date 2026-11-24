"""Additional public input-only controls for the verified development tasks."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from corpus import COUNTER, NONE, REPAIR, SEP, identity, load_corpus


def counter_predictions(prompt):
    """Return a cheap cue and a constructive witness verified independently by the gym."""
    tag, a, forbidden_a, sep, b, forbidden_b, end = prompt
    if (tag, sep, end) != (COUNTER, SEP, SEP):
        raise ValueError("unexpected counterexample encoding")
    cue = NONE if (a, forbidden_a) == (b, forbidden_b) else a
    # Each positive mask is a legal assignment of its own rule. Try both before
    # adding a single bit on which the forbidden masks differ.
    if (a & (b | forbidden_b)) != b:
        return cue, a
    if (b & (a | forbidden_a)) != a:
        return cue, b
    assert a == b
    difference = forbidden_a ^ forbidden_b
    exact = a | (difference & -difference) if difference else NONE
    return cue, exact


def repair_prediction(prompt):
    if prompt[0] != REPAIR or prompt[-5] != SEP or prompt[-1] != SEP:
        raise ValueError("unexpected repair encoding")
    previous, action, argument = prompt[-4:-1]
    if action not in (16, 17):
        raise ValueError("unexpected repair opcode")
    return argument if action == 16 else (previous + argument) % 4


def summarize(rows):
    counters = [row for row in rows if row.lane == "counter"]
    repairs = [row for row in rows if row.lane == "repair"]
    lean = defaultdict(list)
    for row in rows:
        if row.lane == "lean":
            lean[row.root].append(row)
    cue_count = exact_count = 0
    for row in counters:
        cue, exact = counter_predictions(row.public())
        cue_count += cue in row.good
        exact_count += exact in row.good
    result = {"root_counts": {lane: len({r.root for r in rows if r.lane == lane})
                              for lane in sorted({r.lane for r in rows})},
              "counter_equal_else_first_mask": [cue_count, len(counters)],
              "counter_constructive": [exact_count, len(counters)],
              "repair_marked_query_only": [sum(repair_prediction(row.public()) in row.good for row in repairs),
                                           len(repairs)]}
    if lean:
        result["lean"] = {
            "goals": len(lean), "structural_families": len({r.group for group in lean.values() for r in group}),
            "lexical_top4": sum(any(row.good == (1,) for row in sorted(group, key=lambda r: r.order)[:4])
                                for group in lean.values()),
            "pool_ceiling": sum(any(row.good == (1,) for row in group) for group in lean.values()),
            "unknown_candidates": sum(row.good == () for group in lean.values() for row in group),
        }
    return result


def main():
    from study import select_panel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=48)
    args = parser.parse_args()
    if args.roots < 1:
        parser.error("roots must be positive")
    _, dev, source = load_corpus(args.pack)
    native = select_panel(dev, args.roots)
    print(json.dumps({
        "scope": "Additional descriptive public development controls outside the promotion rule",
        "data": source, "native_panel_digest": identity([(r.lane, r.root, r.order) for r in native]),
        "native": summarize(native), "all_development": summarize([r for group in dev.values() for r in group]),
    }, indent=2))


if __name__ == "__main__":
    main()
