"""Custody separate mathematical assessment families and private reproducible draws."""

import argparse
import hashlib
import itertools
import json
import random
import secrets
import tempfile
from functools import lru_cache
from pathlib import Path

from production import digest

FAMILIES = {
    "confirmation": ("independent_sets", "subset_sum"),
    "adjudication": ("linear_extensions", "perfect_matchings"),
    "final": ("proper_colorings", "domino_tilings"),
}


def sample(family, rng):
    if family in ("independent_sets", "linear_extensions", "proper_colorings"):
        n = rng.randrange(5, 8)
        edges = [list(e) for e in itertools.combinations(range(n), 2) if rng.random() < 0.34]
        return {"n": n, "edges": edges}
    if family == "subset_sum":
        values = sorted(rng.randrange(1, 18) for _ in range(rng.randint(6, 12)))
        return {"values": values, "target": rng.randrange(4, sum(values))}
    if family == "perfect_matchings":
        n = rng.randrange(4, 7)
        return {"n": n, "allowed": [[int(rng.random() < 0.57) for _ in range(n)] for _ in range(n)]}
    w, h = rng.choice(((4, 3), (4, 4), (6, 2)))
    cells = sorted(rng.sample(range(w * h), rng.choice((2, 4, 6))))
    return {"width": w, "height": h, "blocked": cells}


def counts(family, obj):
    """Two different finite enumerations per environment-owned target."""
    if family == "subset_sum":
        v, t = obj["values"], obj["target"]
        independent = sum(sum(vals) == t for n in range(len(v) + 1) for vals in itertools.combinations(v, n))
        ways = [1] + [0] * t
        for a in v:
            for b in range(t, a - 1, -1):
                ways[b] += ways[b - a]
        return independent, ways[t]
    if family in ("perfect_matchings", "domino_tilings"):
        if family == "perfect_matchings":
            opts = [{j for j, x in enumerate(row) if x} for row in obj["allowed"]]
            right = list(range(obj["n"]))
        else:
            w, h = obj["width"], obj["height"]
            cells = set(range(w * h)) - set(obj["blocked"])
            left = sorted(i for i in cells if (i // w + i % w) % 2 == 0)
            right = sorted(cells - set(left))
            opts = [{j for j in right if abs(i // w - j // w) + abs(i % w - j % w) == 1} for i in left]
        if len(opts) != len(right):
            # A domino or matching covers exactly one cell on each side.
            return 0, 0
        brute = sum(all(v in vs for v, vs in zip(p, opts, strict=True)) for p in itertools.permutations(right))

        @lru_cache(None)
        def extend(i, used):
            if i == len(opts):
                return 1
            return sum(extend(i + 1, used | 1 << v) for v in opts[i] if not used & (1 << v))

        return brute, extend(0, 0)
    n, edges = obj["n"], obj["edges"]
    if family == "linear_extensions":
        brute = sum(all(p.index(u) < p.index(v) for u, v in edges) for p in itertools.permutations(range(n)))
        pred = [sum(1 << u for u, v in edges if v == i) for i in range(n)]

        @lru_cache(None)
        def grow(mask):
            if mask == (1 << n) - 1:
                return 1
            return sum(grow(mask | 1 << v) for v in range(n) if not (mask >> v) & 1 and pred[v] & mask == pred[v])

        return brute, grow(0)
    options = 2 if family == "independent_sets" else 3

    def valid(a, b):
        return not (a and b) if options == 2 else a != b

    brute = sum(
        all(valid(values[a], values[b]) for a, b in edges) for values in itertools.product(range(options), repeat=n)
    )
    neighbors = [{a for a, b in edges if b == v} for v in range(n)]

    def assign(prefix):
        if len(prefix) == n:
            return 1
        return sum(
            assign((*prefix, c)) for c in range(options) if all(valid(prefix[u], c) for u in neighbors[len(prefix)])
        )

    return brute, assign(())


def freeze(out):
    if out.exists():
        raise FileExistsError("custody directory already exists; use verify for the existing draw")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp) / "reserve"
        stage.mkdir()
        key = secrets.token_bytes(32)
        (stage / "private.key").write_bytes(key)
        summary = {}
        for role, families in FAMILIES.items():
            folder = stage / role
            folder.mkdir()
            public, private = [], []
            for family in families:
                rng = random.Random(hashlib.sha256(key + family.encode()).digest())
                roots = set()
                while len(roots) < 24:
                    obj = sample(family, rng)
                    root = hashlib.sha256(json.dumps([family, obj], sort_keys=True).encode()).hexdigest()
                    if root in roots:
                        continue
                    roots.add(root)
                    a, b = counts(family, obj)
                    if a != b:
                        raise ValueError("independent counters disagree")
                    public.append({"root": root, "family": family, "input": obj})
                    private.append({"root": root, "count": a})
            for filename, rows in (("questions.jsonl", public), ("targets.jsonl", private)):
                (folder / filename).write_bytes(b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in rows))
            summary[role] = {
                "families": families,
                "roots": len(public),
                "questions_sha256": digest(folder / "questions.jsonl"),
                "targets_sha256": digest(folder / "targets.jsonl"),
            }
        manifest = {
            "roles": summary,
            "draw_commitment": digest(stage / "private.key"),
            "scored_models": 0,
            "training_rows": 0,
        }
        (stage / "manifest.json").write_bytes(json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
        stage.rename(out)
    return manifest


def verify(out):
    manifest = json.loads((out / "manifest.json").read_bytes())
    if digest(out / "private.key") != manifest["draw_commitment"]:
        raise ValueError("draw key changed")
    for role, families in FAMILIES.items():
        meta = manifest["roles"][role]
        if list(families) != meta["families"] or meta["roots"] != 48:
            raise ValueError("held family commitments changed")
        for kind in ("questions", "targets"):
            if digest(out / role / (kind + ".jsonl")) != meta[kind + "_sha256"]:
                raise ValueError("held shard changed")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("freeze", "verify"))
    p.add_argument("out", type=Path)
    a = p.parse_args()
    print(json.dumps((freeze if a.mode == "freeze" else verify)(a.out), indent=2))
