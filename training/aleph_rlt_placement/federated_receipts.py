"""Exact mergeable summaries for noiseless binary linear research receipts."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def project(equations, variables, boundary):
    """Eliminate local variables first. Return canonical boundary equations or an inconsistency."""
    if len(set(variables)) != len(variables) or len(set(boundary)) != len(boundary):
        raise ValueError("duplicate variable")
    if not set(boundary) <= set(variables):
        raise ValueError("boundary must be in the declared world")
    private = [v for v in variables if v not in boundary]
    ordered = private + list(boundary)
    positions = {v: i for i, v in enumerate(ordered)}
    rows = []
    for names, value in equations:
        if type(value) is not int or value not in (0, 1):
            raise ValueError("nonbinary label")
        mask = 0
        for v in names:
            if v not in positions:
                raise ValueError("undeclared variable")
            mask ^= 1 << positions[v]
        rows.append([mask, value])
    cursor = 0
    for column in range(len(ordered)):
        bit = 1 << column
        pivot = next((i for i in range(cursor, len(rows)) if rows[i][0] & bit), None)
        if pivot is None:
            continue
        rows[cursor], rows[pivot] = rows[pivot], rows[cursor]
        m, y = rows[cursor]
        for j, row in enumerate(rows):
            if j != cursor and row[0] & bit:
                row[0] ^= m
                row[1] ^= y
        cursor += 1
    if [0, 1] in rows:
        return {"consistent": False, "equations": []}
    kept = []
    for mask, y in rows[:cursor]:
        if mask & ((1 << len(private)) - 1):
            continue
        kept.append(([v for v in boundary if mask & (1 << positions[v])], y))
    return {"consistent": True, "equations": kept}


def federate(instance):
    boundary, labs = instance["boundary"], instance["labs"]
    used = set()
    summaries = []
    for lab in labs:
        private = set(lab["variables"]) - set(boundary)
        if private & used:
            raise ValueError("shared variable omitted from boundary")
        used |= private
        # A lab with no local measurement on a boundary variable leaves it free.
        variables = list(dict.fromkeys([*lab["variables"], *boundary]))
        summaries.append(project(lab["equations"], variables, boundary))
    if any(not summary["consistent"] for summary in summaries):
        merged = {"consistent": False, "equations": []}
    else:
        merged = project([e for s in summaries for e in s["equations"]], boundary, boundary)
    return {"boundary": boundary, "summaries": summaries, "merged": merged}


def affine_sets(d):
    """Independent truth-table oracle, including the empty feasible set."""
    universe = range(1 << d)
    sets = []
    for mask in range(1 << (1 << d)):
        subset = frozenset(x for x in universe if mask & (1 << x))
        if subset:
            pivot = min(subset)
            directions = {v ^ pivot for v in subset}
            if any(a ^ b not in directions for a in directions for b in directions):
                continue
        sets.append(subset)
    return sets


def encode(table, variables):
    if not table:
        return [([], 1)]
    out = []
    for m in range(1, 1 << len(variables)):
        labels = {(h & m).bit_count() % 2 for h in table}
        if len(labels) == 1:
            out.append(([name for i, name in enumerate(variables) if m & (1 << i)], labels.pop()))
    return out


def decode(summary, variables):
    if not summary["consistent"]:
        return set()
    rows = [(sum(1 << variables.index(v) for v in names), y) for names, y in summary["equations"]]
    return {h for h in range(1 << len(variables)) if all((h & m).bit_count() % 2 == y for m, y in rows)}


def census():
    sets = [affine_sets(d) for d in range(4)]
    counts = [len(x) for x in sets]
    assert counts == [2, 4, 12, 52]
    boundary = ["x", "y"]
    checks = 0
    for left, right in itertools.product(sets[3], repeat=2):
        labs = []
        for private, table in (("p", left), ("q", right)):
            variables = [*boundary, private]
            labs.append({"variables": variables, "equations": encode(table, variables)})
        observed = federate({"boundary": boundary, "labs": labs})["merged"]
        # Solve both original records on all global worlds independently of elimination.
        truth = {h & 3 for h in range(16) if h & 7 in left and ((h & 3) | ((h >> 1) & 4)) in right}
        assert decode(observed, boundary) == truth
        checks += 1
    def project_target(relation):
        return {h & 1 for h in relation}

    false_merges = compatible_but_lossy = 0
    for left, right in itertools.product(sets[2], repeat=2):
        exact = project_target(left & right)
        naive = project_target(left) & project_target(right)
        if exact != naive:
            false_merges += 1
            compatible_but_lossy += bool(exact)
    example = {
        "boundary": ["s", "t"],
        "labs": [
            {"variables": ["s"], "equations": [[["s"], 1]]},
            {"variables": ["s", "t"], "equations": [[["s", "t"], 0]]},
        ],
    }
    result = federate(example)
    target = project(result["merged"]["equations"], example["boundary"], ["t"])
    assert decode(target, ["t"]) == {1}
    return {
        "scope": "finite exact noiseless binary-linear interface, contradiction observable",
        "feasible_boundary_relations_d0_to_3": counts,
        "minimum_fixed_bits": [(n - 1).bit_length() for n in counts],
        "disjoint_private_merge_checks": checks,
        "target_only_two_lab_pairs": len(sets[2]) ** 2,
        "target_only_wrong": false_merges,
        "wrong_with_compatible_records": compatible_but_lossy,
        "two_lab_example": example,
        "messages": result,
        "target_after_merge": target,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", type=Path, help="JSON instance with boundary and labs")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = federate(json.loads(args.merge.read_bytes())) if args.merge else census()
    body = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(body, encoding="utf-8")
    print(body, end="")
