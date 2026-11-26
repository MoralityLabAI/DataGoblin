"""MOLT-GOBLIN-01: deterministic planning and array curation fixtures.

Offline reference only. No checkpoint loading, execution or promotion authority.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any

FAMILIES = ("parallel_array", "pareto_array", "prefix_dag", "matched_control")
SPLITS = ("train", "calibration", "ordinary_test", "ood_test")


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify(task: dict[str, Any]) -> dict[str, Any]:
    """Environment-owned answer, computed only from immutable public input."""
    family = task["family"]
    p = task["problem"]
    if family == "parallel_array":
        best = (0, ())
        eligible = [c for c in p["candidates"] if c["qualified"] and c["prefix_ok"]]
        for k in range(1, len(eligible) + 1):
            for subset in itertools.combinations(eligible, k):
                if any(sum(c[m] for c in subset) > p["caps"][m] for m in p["caps"]):
                    continue
                ids = tuple(sorted(c["id"] for c in subset))
                score = sum(c["utility"] for c in subset)
                if score > best[0] or (score == best[0] and ids < best[1]):
                    best = (score, ids)
        return {"selected": list(best[1]), "utility": best[0]}
    if family == "pareto_array":
        cs = p["candidates"]
        out = {}
        for b in cs:
            if not b["qualified"]:
                out[b["id"]] = "INELIGIBLE"
                continue
            others = [a for a in cs if a["id"] != b["id"] and a["qualified"]]
            # All axes maximize; cost has already been negated.
            def dominates(a: dict, x: dict, bound: int) -> bool:
                pairs = list(zip(a["intervals"], x["intervals"], strict=True))
                ds = [u[bound] - v[1 - bound] for u, v in pairs]
                return all(d >= 0 for d in ds) and any(d > 0 for d in ds)
            if any(dominates(a, b, 0) for a in others):
                out[b["id"]] = "DOMINATED"
            elif any(dominates(a, b, 1) for a in others):
                out[b["id"]] = "UNRESOLVED"
            else:
                out[b["id"]] = "SELECTED"
        return {"statuses": out}
    if family == "prefix_dag":
        bad = sorted([e for e in p["edges"] if e[0] >= e[1]])
        if bad:
            return {"invalid_edges": bad, "stabilization_bound": None}
        depth = [1] * p["positions"]
        for j in range(p["positions"]):
            for i, target in p["edges"]:
                if target == j:
                    depth[j] = max(depth[j], depth[i] + 1)
        return {"invalid_edges": [], "stabilization_bound": max(depth)}
    if family == "matched_control":
        ref = p["treatment"]
        keys = ("root", "params", "tokens", "visits", "verifier_calls", "dose")
        eligible_ids = sorted(
            c["id"] for c in p["candidates"]
            if c["kind"] == "sham" and all(c[k] == ref[k] for k in keys)
        )
        return {"matched_shams": eligible_ids, "status": "READY" if eligible_ids else "REDESIGN"}
    raise ValueError(f"unknown family: {family}")


def make_task(split: str, seed: int, family: str, index: int) -> dict[str, Any]:
    if split not in SPLITS or family not in FAMILIES or index < 0:
        raise ValueError("invalid split, family or index")
    root = digest(["molt-goblin-01", split, seed, family, index])
    rng = random.Random(int(root, 16))
    n = rng.randint(7, 9) if split == "ood_test" else rng.randint(4, 6)
    if family == "parallel_array":
        p = {"caps": {"ram_mb": 8192, "cpu_slots": 8, "visits": 64}, "candidates": [
            {"id": f"c{i}", "qualified": rng.randrange(5) != 0, "prefix_ok": rng.randrange(5) != 0,
             "ram_mb": rng.choice([512, 1024, 2048, 4096]), "cpu_slots": rng.randint(1, 4),
             "visits": rng.choice([4, 8, 16, 32]), "utility": rng.randint(1, 30)}
            for i in range(n)
        ]}
    elif family == "pareto_array":
        cs = []
        for i in range(n):
            lo = [rng.randint(0, 90), -rng.randint(10, 90)]
            cs.append({"id": f"c{i}", "qualified": rng.randrange(6) != 0,
                       "intervals": [[x, x + rng.randint(0, 10)] for x in lo]})
        p = {"axes": ["verified_utility", "negative_cost"], "candidates": cs}
    elif family == "prefix_dag":
        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if rng.randrange(3) == 0]
        if rng.randrange(3) == 0:
            i = rng.randrange(n)
            edges.append([i, rng.randrange(i + 1)])
        p = {"positions": n, "edges": sorted(edges)}
    else:
        ref = {"root": "r0", "params": 4096, "tokens": 512, "visits": 8,
               "verifier_calls": 32, "dose": 20}
        cs = []
        for i in range(n):
            c = dict(ref, id=f"c{i}", kind=rng.choice(["sham", "sham", "treatment"]))
            if rng.randrange(3) != 0:
                key = rng.choice(sorted(ref))
                c[key] = "r1" if key == "root" else c[key] + 1
            cs.append(c)
        p = {"treatment": ref, "candidates": cs}
    return {"schema": "molt-goblin-task-v1", "task_id": root, "root_id": root,
            "split": split, "family": family, "problem": p}


INSTRUCTIONS = {
    "parallel_array": "Select qualified prefix-valid concurrent experiments maximizing total utility "
                      "within every summed cap. Break ties by lexicographically smallest sorted ID list.",
    "pareto_array": "Classify each qualified candidate using conservative and possible interval dominance "
                    "on all maximizing axes. Strict improvement on at least one axis is required. "
                    "Conservative dominance gives DOMINATED, possible-only gives UNRESOLVED, "
                    "otherwise SELECTED; unqualified gives INELIGIBLE.",
    "prefix_dag": "Edges are previous-pass source to next-pass target. Identify all non-prefix edges "
                  "(source >= target), or compute the maximum vertex count on a directed path.",
    "matched_control": "Select all sham controls exactly matching root, parameters, tokens, visits, "
                       "verifier calls and dose. READY if at least one exists; otherwise REDESIGN.",
}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def generate(split: str, seed: int, count: int, public: Path, steward: Path) -> None:
    if not 1 <= count <= 2000:
        raise ValueError("count per family must be in 1..2000")
    public, steward = public.resolve(), steward.resolve()
    if public == steward or public in steward.parents or steward in public.parents:
        raise ValueError("public and steward directories must be disjoint")
    if public.exists() or steward.exists():
        raise FileExistsError("refuse to overwrite an existing release")
    public.mkdir(parents=True)
    steward.mkdir(parents=True, mode=0o700)
    tasks = [make_task(split, seed, family, i) for family in FAMILIES for i in range(count)]
    rows = [dict(t, instruction=INSTRUCTIONS[t["family"]]) for t in tasks]
    labels = [{"task_id": t["task_id"], "answer": verify(t), "input_digest": digest(t)} for t in tasks]
    write_rows(public / "tasks.jsonl", rows)
    write_rows(steward / "labels.jsonl", labels)
    if split == "train":
        sft = [{"messages": [
            {"role": "user", "content": t["instruction"] + "\n" + json.dumps(t["problem"], sort_keys=True)},
            {"role": "assistant", "content": json.dumps(a["answer"], sort_keys=True)},
        ], "task_id": t["task_id"], "root_id": t["root_id"], "split": split}
            for t, a in zip(rows, labels, strict=True)]
        write_rows(public / "train_sft.jsonl", sft)
    manifest = {"schema": "molt-goblin-release-v1", "split": split, "count": len(rows),
                "input_digest": digest(rows), "model_executed": False}
    (public / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(public: Path, steward: Path, predictions: Path) -> dict[str, Any]:
    """Steward-only exact answer score. Missing predictions count as failures."""
    rows = read_rows(public / "tasks.jsonl")
    labels = read_rows(steward / "labels.jsonl")
    manifest = json.loads((public / "manifest.json").read_text(encoding="utf-8"))
    if digest(rows) != manifest["input_digest"] or len(rows) != manifest["count"]:
        raise ValueError("public release mismatch")
    gold = {a["task_id"]: a for a in labels}
    pred = {}
    for row in read_rows(predictions):
        tid = row["task_id"]
        if tid in pred or tid not in gold:
            raise ValueError("duplicate or unknown prediction")
        pred[tid] = row["answer"]
    if len(gold) != len(labels) or set(gold) != {r["task_id"] for r in rows}:
        raise ValueError("label roster mismatch")
    totals = {f: {"correct": 0, "total": 0} for f in FAMILIES}
    for row in rows:
        tid = row["task_id"]
        task = {k: v for k, v in row.items() if k != "instruction"}
        if digest(task) != gold[tid]["input_digest"] or digest(verify(task)) != digest(gold[tid]["answer"]):
            raise ValueError("invalid label binding")
        bucket = totals[row["family"]]
        bucket["total"] += 1
        bucket["correct"] += int(tid in pred and digest(pred[tid]) == digest(gold[tid]["answer"]))
    return {"split": manifest["split"], "families": totals, "missing": len(rows) - len(pred),
            "promotion_authorized": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--split", choices=SPLITS, required=True)
    gen.add_argument("--seed", type=int, required=True)
    gen.add_argument("--per-family", type=int, default=32)
    for command in (gen, (sc := sub.add_parser("score"))):
        command.add_argument("--public", type=Path, required=True)
        command.add_argument("--steward", type=Path, required=True)
    sc.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.split, args.seed, args.per_family, args.public, args.steward)
    else:
        print(json.dumps(score(args.public, args.steward, args.predictions), indent=2))


if __name__ == "__main__":
    main()
