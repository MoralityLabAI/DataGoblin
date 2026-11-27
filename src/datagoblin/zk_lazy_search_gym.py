"""ZKL-14A: paired eager/lazy allocation control, identical search policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from neighborhood_search import BN254_FR

from datagoblin.zk_search_gym import TEST, fixture, run


def campaign(test_start=13301, roots=8):
    if test_start != 13301 or not 1 <= roots <= 32:
        raise ValueError("reserved evaluation roots only")
    tasks = []
    for family in TEST:
        for seed in range(test_start, test_start + roots):
            p = 11 if seed % 2 else BN254_FR
            a, b = fixture(p, family, seed), fixture(p, family, seed, eager=False)
            assert a.hashes == b.hashes
            assert b.prepared_columns == b.pool_terms == 0
            eager = run(a, "peel_local", 24, seed=seed)
            lazy = run(b, "peel_local", 24, seed=seed)
            for key in ("certificate", "columns", "input_terms", "reductions", "coefficient_work",
                        "inversions", "pool_columns", "score_evals", "graph_row_scans"):
                if eager[key] != lazy[key]:
                    raise AssertionError(f"paired search changed: {key}")
            assert lazy["prepared_columns"] == lazy["columns"]
            assert lazy["pool_terms"] == lazy["input_terms"]
            assert eager["exact_identity_checked"] == (family != "essential")
            tasks.append({"family": family, "seed": seed, "modulus": str(p), "eager": eager, "lazy": lazy})
    return {"schema": "zkl-lazy-search-v1", "tasks": tasks, "lean_replayed": False,
            "production_accepted": False}


def summarize(result):
    keys = ("columns", "prepared_columns", "pool_terms", "coefficient_work", "score_evals", "graph_row_scans")
    return {arm: {**{key: sum(t[arm][key] for t in result["tasks"]) for key in keys},
                  "certified": sum(t[arm]["exact_identity_checked"] for t in result["tasks"])}
            for arm in ("eager", "lazy")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=8)
    args = parser.parse_args()
    result = campaign(roots=args.roots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, sort_keys=True)
    print(json.dumps(summarize(result), sort_keys=True))


if __name__ == "__main__":
    main()
