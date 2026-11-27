"""ZKL-08 finite-domain circuit gym. No model calls or promotion authority."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

DEVELOPMENT = ("bit", "multiply", "affine")
HOLDOUT = ("range2", "private_select", "composed")
MAX_ASSIGNMENTS = 50_000


def lc(p, *terms):
    acc = {}
    for i, c in terms:
        acc[i] = (acc.get(i, 0) + c) % p
    return [[i, str(c)] for i, c in sorted(acc.items()) if c]


def shape(p, family, k):
    one = lc(p, (0, 1))

    def var(i):
        return lc(p, (i, 1))

    def bit(i):
        return [var(i), lc(p, (i, 1), (0, -1)), []]

    if family == "bit":
        n, public, rows = 2, [1], [bit(1)]
    elif family == "multiply":
        n, public, rows = 4, [1, 2, 3], [[var(1), var(2), var(3)]]
    elif family == "affine":
        n, public, rows = 3, [1, 2], [[lc(p, (1, k), (0, 1)), one, var(2)]]
    elif family == "range2":
        n, public, rows = 4, [1], [
            bit(2), bit(3), [lc(p, (2, 1), (3, 2)), one, var(1)],
        ]
    elif family == "private_select":
        n, public, rows = 5, [1, 2, 3], [
            bit(4),
            [lc(p, (1, 1), (2, -1)), var(4), lc(p, (3, 1), (2, -1))],
        ]
    elif family == "composed":
        n, public, rows = 5, [1, 2, 3], [
            [var(1), var(2), var(4)],
            [lc(p, (4, 1), (1, 1)), one, var(3)],
        ]
    else:
        raise ValueError(family)
    return {"format": "zkl-r1cs-v1", "modulus": str(p), "wires": n,
            "public": public, "rows": rows}


def spec(family, p, k, public):
    """Independent public relation, without consulting circuit or witness generation."""
    if family == "bit":
        return public[0] in (0, 1)
    if family == "multiply":
        a, b, c = public
        return c == a * b % p
    if family == "affine":
        a, b = public
        return b == (k * a + 1) % p
    if family == "range2":
        return public[0] < 4
    if family == "private_select":
        a, b, y = public
        return y in (a, b)
    if family == "composed":
        a, b, c = public
        return c == (a * b + a) % p
    raise ValueError(family)


def shape_digest(s):
    return hashlib.sha256(json.dumps(s, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def public_relation(s):
    p, n = int(s["modulus"]), s["wires"]
    if p not in (7, 11) or n < 1 or p ** (n - 1) > MAX_ASSIGNMENTS:
        raise ValueError("outside exhaustive fixture cap")
    public = s["public"]
    if (any(type(i) is not int or not 0 < i < n for i in public)
            or len(set(public)) != len(public)):
        raise ValueError("invalid public map")
    rows = []
    for row in s["rows"]:
        if len(row) != 3:
            raise ValueError("bad row arity")
        parsed = []
        for combination in row:
            terms = [(i, int(c)) for i, c in combination]
            if any(type(i) is not int or not 0 <= i < n or not 0 <= c < p for i, c in terms):
                raise ValueError("bad coefficient or wire")
            parsed.append(terms)
        rows.append(parsed)
    accepted = {}
    for aux in itertools.product(range(p), repeat=n - 1):
        w = (1, *aux)

        def ev(combination, assignment=w):
            return sum(assignment[i] * c for i, c in combination) % p

        if all(ev(a) * ev(b) % p == ev(c) for a, b, c in rows):
            x = tuple(w[i] for i in public)
            accepted.setdefault(x, w)
    return accepted


def evaluate(s, family, k):
    p = int(s["modulus"])
    valid = {x for x in itertools.product(range(p), repeat=len(s["public"]))
             if spec(family, p, k, x)}
    accepted = public_relation(s)
    false_positive = sorted(accepted.keys() - valid)
    missing = sorted(valid - accepted.keys())
    return {
        "status": "EXHAUSTIVE_FINITE_DOMAIN",
        "sound": not false_positive, "complete": not missing,
        "accepted_public": len(accepted), "expected_public": len(valid),
        "false_positive_count": len(false_positive), "missing_count": len(missing),
        "counterexample": None if not false_positive else {
            "public": false_positive[0], "assignment": accepted[false_positive[0]],
        },
        "missing_public": missing[0] if missing else None,
        "shape_sha256": shape_digest(s), "constraints": len(s["rows"]),
        "lean_replayed": False, "production_accepted": False,
    }


def variant(s, name, rng):
    rows = list(s["rows"])
    i = rng.randrange(len(rows))
    if name == "drop":
        rows.pop(i)
    elif name == "duplicate":
        rows.insert(i, rows[i])
    elif name == "coefficient":
        # Change C by the constant one; may make relation incomplete or unsound.
        p = int(s["modulus"])
        a, b, c = rows[i]
        rows[i] = [a, b, lc(p, *((j, int(v)) for j, v in c), (0, 1))]
    elif name != "clean":
        raise ValueError(name)
    return dict(s, rows=rows)


def optimize(s, family, k):
    baseline = evaluate(s, family, k)
    if not baseline["sound"] or not baseline["complete"]:
        return s, []
    accepted = set(public_relation(s))
    out = dict(s, rows=list(s["rows"]))
    edits = []
    for i in reversed(range(len(out["rows"]))):
        rows = out["rows"]
        candidate = dict(out, rows=rows[:i] + rows[i + 1:])
        if set(public_relation(candidate)) == accepted:
            edits.append({
                "delete_row": i, "before": shape_digest(out),
                "after": shape_digest(candidate),
                "certificate_kind": "duplicate" if rows[i] in candidate["rows"] else "finite_only",
                "lean_replayed": False,
            })
            out = candidate
    return out, edits


def campaign(seed, split, per_family, with_optimization=False):
    if per_family < 1:
        raise ValueError("per_family must be positive")
    if split not in ("development", "holdout"):
        raise ValueError("unknown split")
    families = DEVELOPMENT if split == "development" else HOLDOUT
    rng = random.Random(seed)
    for family in families:
        for root in range(per_family):
            p = rng.choice((7, 11))
            k = rng.randrange(1, p)
            clean = shape(p, family, k)
            for kind in ("clean", "drop", "duplicate", "coefficient"):
                case = variant(clean, kind, rng)
                result = evaluate(case, family, k)
                record = {
                    "schema": "zkl-gym-v1", "seed": seed, "root": root,
                    "split": split, "family": family, "parameter": k,
                    "mutation": kind, "shape": case, "receipt": result,
                }
                if with_optimization:
                    repaired, edits = optimize(case, family, k)
                    record["optimization"] = {"shape": repaired, "edits": edits,
                                              "receipt": evaluate(repaired, family, k)}
                yield record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split", choices=("development", "holdout"), required=True)
    parser.add_argument("--per-family", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"cases": 0, "sound_complete": 0, "unsound": 0, "incomplete": 0}
    with args.out.open("x", encoding="utf-8") as f:
        for rec in campaign(args.seed, args.split, args.per_family, args.optimize):
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            result = rec["receipt"]
            counts["cases"] += 1
            counts["sound_complete"] += int(result["sound"] and result["complete"])
            counts["unsound"] += int(not result["sound"])
            counts["incomplete"] += int(not result["complete"])
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
