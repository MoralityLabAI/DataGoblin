"""ZKL-21: cache invariant symbolic score factors; replay actual learned certificates."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from neighborhood_search import BN254_FR
from poly_certificate import generate

from datagoblin.zk_advantage_gym import FAMILIES, SHIFT
from datagoblin.zk_exploration_gym import Hybrid
from datagoblin.zk_exploration_gym import run as run_gate
from datagoblin.zk_neighborhood_gym import PROTOCOLS, fixture, full_reference
from datagoblin.zk_row_heap_gym import HeapHybrid, RowFrontier, check_pair, train

ROOT_BLOCKS = (21101, 21201)
ARMS = ("indexed", "row_random", "periodic", "random_gate",
        "heap_recurrent", "cached_recurrent", "heap_reset", "cached_reset")


def prepare_cached(problem, seed):
    """Same supports, float operation order and random state as original prepare."""
    rows = [{m for m, c in poly.items() if c % problem.p} for poly in problem.polys]
    target = problem.target_support
    target_wires = {i for m in target for i in m}
    rw = [{i for m in support for i in m} for support in rows]
    freq = Counter(i for wires in rw for i in wires)
    row_info = [
        (wires, len(wires & target_wires)/max(1, len(wires)),
         sum(freq[j] == 1 for j in wires)/max(1, len(wires)))
        for wires in rw
    ]
    mon_info = {}
    supports, base = {}, {}
    for key in problem.keys:
        i, mon = key
        if mon not in mon_info:
            mw = set(mon)
            mon_info[mon] = (mw, len(mw & target_wires)/max(1, len(mon)),
                             float(len(mon) == 0), float(len(mon) == 1))
        mw, f7, f1, f2 = mon_info[mon]
        row_wires, f5, f8 = row_info[i]
        support = frozenset(tuple(sorted((*m, *mon))) for m in rows[i])
        supports[key] = support
        wires = row_wires | mw if support else set()
        f4 = len(support & target)/max(1, len(support))
        f6 = len(wires & target_wires)/max(1, len(wires))
        # Preserve the original expression, including float association.
        base[key] = 3*f4 + f5 + f6 + 0.5*f7 - f8 + 0.7*f1 + 0.2*f2
    rng = random.Random(seed)
    tie = {key: rng.random() for key in problem.keys}
    if len(set(tie.values())) != len(tie):
        raise ValueError("tie collision")
    return supports, base, tie, rng


class CachedHeapHybrid(HeapHybrid):
    def __init__(self, problem, seed):
        Hybrid.__init__(self, problem, seed, index_factory=RowFrontier,
                        prepare_factory=prepare_cached)


def evaluate(p, family, seed, arm, budget, limit, model, linear):
    if arm.startswith(("heap_", "cached_")):
        kind, variant = arm.split("_", 1)
        factory = CachedHeapHybrid if kind == "cached" else HeapHybrid
        return run_gate(p, family, seed, variant, budget, limit, model, linear,
                        hybrid_factory=factory, record_trace=True)
    return run_gate(p, family, seed, arm, budget, limit, model, linear)


def campaign(roots=8, test_start=21101, train_roots=8):
    if not 1 <= roots <= 16 or not 1 <= train_roots <= 16 or test_start not in ROOT_BLOCKS:
        raise ValueError("reserved root blocks and caps")
    model, linear, training = train(range(18001, 18001+train_roots))
    tasks = []
    for family in FAMILIES:
        for seed in range(test_start, test_start+roots):
            p = 11 if seed % 2 else BN254_FR
            essential = family.startswith("essential_")
            ref, ref_cols = full_reference(p, family, seed)
            assert (ref is not None) == (not essential)
            arms = {}
            order = list(ARMS)
            random.Random(seed).shuffle(order)
            for protocol, (budget, limit) in PROTOCOLS.items():
                recs = {arm: evaluate(p, family, seed, arm, budget, limit, model, linear)
                        for arm in order}
                for variant in ("recurrent", "reset"):
                    a, b = recs[f"heap_{variant}"], recs[f"cached_{variant}"]
                    if limit is None:
                        check_pair(a, b)
                        assert a["column_scores"] == b["column_scores"]
                        assert a["row_index_metrics"] == b["row_index_metrics"]
                    else:
                        n = min(len(a["trace"]), len(b["trace"]))
                        assert a["trace"][:n] == b["trace"][:n]
                for rec in recs.values():
                    assert rec["columns"] <= budget
                    assert rec["prepared_columns"] == rec["columns"]
                    assert not rec["lean_replayed"] and not rec["production_accepted"]
                    if essential:
                        assert not rec["exact_identity_checked"]
                    if limit is not None and rec["credited"]:
                        assert rec["cpu_ns"] <= limit
                arms[protocol] = recs
            tasks.append({"family": family, "seed": seed, "modulus": str(p),
                          "shift": family in SHIFT, "essential": essential,
                          "reference": ref, "reference_columns": ref_cols, "arms": arms})
    return {"schema": "zkl-cached-prep-v1", "test_start": test_start, "training": training,
            "model": {k: v.tolist() for k, v in model.items()}, "linear": linear.tolist(),
            "tasks": tasks, "lean_replayed": False, "production_accepted": False}


def summarize(result):
    fields = ("columns", "column_scores", "gate_decisions", "explore_actions",
              "coefficient_work", "setup_ns", "search_ns", "verify_ns", "cpu_ns")
    out = []
    for protocol in PROTOCOLS:
        for arm in ARMS:
            records = [t["arms"][protocol][arm] for t in result["tasks"]]
            out.append({"protocol": protocol, "arm": arm,
                        "certified": sum(q["exact_identity_checked"] for q in records),
                        "credited": sum(q["credited"] for q in records),
                        "deadline_hits": sum(q["deadline_hit"] for q in records),
                        "shift_credited": sum(t["arms"][protocol][arm]["credited"]
                                              for t in result["tasks"] if t["shift"]),
                        **{key: sum(q[key] for q in records) for key in fields}})
    return out


def closed_source(shape):
    # Each membership hypothesis reduces to a right-associated disjunction
    # of equalities. Generate the exact number of alternatives from the shape.
    def cases(n):
        if n < 1:
            raise ValueError("empty membership proof requires a separate branch")
        return " | ".join(["h"]*n)

    proof = """
set_option maxRecDepth 4096 in
set_option maxHeartbeats 4000000 in
theorem original_well_formed : wellFormed originalShape := by
  unfold wellFormed
  refine ⟨by decide, by decide, ?_, by decide, ?_⟩
  · intro i hi
    simp only [originalShape, List.mem_cons, List.not_mem_nil, or_false] at hi
"""
    if len(shape["public"]) == 1:
        proof += "    subst i\n    decide\n"
    else:
        proof += f"    rcases hi with {cases(len(shape['public']))} <;> subst i <;> decide\n"
    proof += """  · intro r hr
    simp only [originalShape, List.mem_cons, List.not_mem_nil, or_false] at hr
"""
    proof += f"    rcases hr with {cases(len(shape['rows']))}\n"
    for row in shape["rows"]:
        n = sum(map(len, row))
        proof += "    · subst r\n      intro t ht\n"
        # Reduce the concrete append by definitional equality before membership cases.
        terms = [t for comb in row for t in comb]
        literal = ", ".join(f"⟨{i}, {v}⟩" for i, v in terms)
        proof += f"      change t ∈ ([{literal}] : List Term) at ht\n"
        proof += "      simp only [List.mem_cons, List.not_mem_nil, or_false] at ht\n"
        if n == 1:
            proof += "      subst t\n      decide\n"
        elif n > 1:
            proof += f"      rcases ht with {cases(n)} <;> subst t <;> decide\n"
        # For a zero-term row, simp closes the contradictory hypothesis.
    return proof + """
theorem exact_poly_delete_closed (x : List Nat) :
    accepts originalShape x ↔ accepts replacementShape x :=
  exact_poly_delete original_well_formed x

#print axioms original_well_formed
#print axioms exact_poly_delete_closed
"""


def emit_replay(results, out):
    """First successful learned certificate per family/field, no reference substitution."""
    out.mkdir(parents=True, exist_ok=False)
    tasks = sorted((t for result in results for t in result["tasks"]), key=lambda t: t["seed"])
    manifest = []
    for family in FAMILIES:
        if family.startswith("essential_"):
            continue
        for p in (11, BN254_FR):
            candidates = [t for t in tasks if t["family"] == family and int(t["modulus"]) == p]
            eligible = [t for t in candidates
                        if t["arms"]["c96"]["cached_recurrent"]["exact_identity_checked"]]
            if not eligible:
                manifest.append({"family": family, "modulus": str(p), "status": "abstained",
                                 "candidate_seeds": [t["seed"] for t in candidates]})
                continue
            task = eligible[0]
            problem = fixture(p, family, task["seed"])
            cert = task["arms"]["c96"]["cached_recurrent"]["certificate"]
            source, receipt = generate(problem.original, problem.replacement, cert, *problem.hashes)
            if not receipt["exact_identity_checked"]:
                raise AssertionError("generator did not reverify certificate")
            source += closed_source(problem.original)
            stem = f"{family}_{'f11' if p == 11 else 'bn254'}"
            path = out / f"{stem}.lean"
            path.write_text(source, encoding="utf-8")
            case = {"family": family, "modulus": str(p), "seed": task["seed"],
                    "candidate_seeds": [t["seed"] for t in candidates],
                    "successful_seeds": [t["seed"] for t in eligible],
                    "status": "generated", "file": path.name,
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "original_sha256": problem.hashes[0], "replacement_sha256": problem.hashes[1],
                    "certificate": cert, "arm": "cached_recurrent", "protocol": "c96"}
            manifest.append(case)
    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2)+"\n",
                                       encoding="utf-8")
    print(json.dumps({"generated": sum(m["status"] == "generated" for m in manifest),
                      "abstained": sum(m["status"] == "abstained" for m in manifest)}))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--train-roots", type=int, default=8)
    parser.add_argument("--test-start", type=int, choices=ROOT_BLOCKS, default=ROOT_BLOCKS[0])
    parser.add_argument("--replay-input", type=Path, nargs="+")
    args = parser.parse_args()
    if args.replay_input:
        emit_replay([json.loads(p.read_text(encoding="utf-8")) for p in args.replay_input], args.out)
        return
    result = campaign(args.roots, args.test_start, args.train_roots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, sort_keys=True)
    print(json.dumps({"training": result["training"], "summary": summarize(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
