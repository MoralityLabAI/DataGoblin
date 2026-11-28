"""RV23-C: dependency-union relaxations and independently replayable proofs."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import random
import subprocess
import sys
import time
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np

from .skill_transfer import UPSTREAM, world_id

KIND = "robust_discrimination"
SEED = 2026831000
ARMS = ("cover", "forced", "closure_one", "closure_all")
LIMITS = (64, 512, 65536)
COHORTS = ("native_frontier", "shared_controls", "transitive_controls")
MAX_TERMS = 20000


def validate(d):
    """Strict finite positive-cost model; no untrusted Python object protocol."""
    if type(d) is not dict or set(d) != {"mechanisms", "experiments", "error_bits", "budget"}:
        raise ValueError("world shape")
    hs, rows = d["mechanisms"], d["experiments"]
    if (type(hs) is not list or not 2 <= len(hs) <= 8 or type(rows) is not list
            or not 1 <= len(rows) <= 16):
        raise ValueError("world bounds")
    if type(d["error_bits"]) is not int or not 0 <= d["error_bits"] <= 2:
        raise ValueError("error bound")
    if type(d["budget"]) is not int or not 0 <= d["budget"] <= 16000:
        raise ValueError("budget")

    def name(s):
        return type(s) is str and 0 < len(s) <= 64 and s.isascii() and all(
            c.isalnum() or c in "_.-" for c in s)

    if not all(map(name, hs)) or len(set(hs)) != len(hs):
        raise ValueError("mechanism IDs")
    ids = []
    for r in rows:
        if type(r) is not dict or set(r) != {"id", "cost", "requires", "predictions"}:
            raise ValueError("experiment shape")
        if not name(r["id"]) or type(r["cost"]) is not int or not 1 <= r["cost"] <= 1000:
            raise ValueError("experiment ID or cost")
        deps, bits = r["requires"], r["predictions"]
        if (type(deps) is not list or len(deps) > 16 or not all(map(name, deps))
                or len(set(deps)) != len(deps)):
            raise ValueError("dependencies")
        if (type(bits) is not list or len(bits) != len(hs)
                or any(type(b) is not int or b not in (0, 1) for b in bits)):
            raise ValueError("predictions")
        ids.append(r["id"])
    if len(set(ids)) != len(ids) or any(not set(r["requires"]) <= set(ids) for r in rows):
        raise ValueError("unknown or duplicate ID")
    return d


class Cutoff(Exception):
    pass


class Search:
    def __init__(self, data, arm, cap, term_cap=MAX_TERMS):
        self.d = validate(data)
        if arm not in ARMS or type(cap) is not int or not 1 <= cap <= 65536:
            raise ValueError("arm or inspection cap")
        if type(term_cap) is not int or not 0 <= term_cap <= MAX_TERMS:
            raise ValueError("combination cap")
        self.arm, self.cap, self.term_cap = arm, cap, term_cap
        self.rows = data["experiments"]
        self.n = len(self.rows)
        self.full = (1 << self.n)-1
        self.costs = [r["cost"] for r in self.rows]
        ids = {r["id"]: i for i, r in enumerate(self.rows)}
        deps = [sum(1 << ids[x] for x in r["requires"]) for r in self.rows]
        self.closures = []
        for i in range(self.n):
            m = 1 << i
            while True:
                nxt = m
                for j in self.indices(m):
                    nxt |= deps[j]
                if nxt == m:
                    break
                m = nxt
            self.closures.append(m)
        self.pairs = list(combinations(range(len(data["mechanisms"])), 2))
        self.seps = [tuple(i for i, r in enumerate(self.rows) if r["predictions"][a] != r["predictions"][b])
                     for a, b in self.pairs]
        self.smasks = [sum(1 << i for i in s) for s in self.seps]
        self.need = 2*data["error_bits"]+1
        self.best, self.best_cost = None, data["budget"]+1
        self.proof, self.queue = {}, []
        self.stats = dict(inspected=0, expanded=0, pruned=0, forced_steps=0, bound_calls=0,
                          pair_terms=0, closure_terms=0, closure_skips=0, generated=0)

    def indices(self, m):
        return [i for i in range(self.n) if m >> i & 1]

    def cost(self, m):
        return sum(self.costs[i] for i in self.indices(m))

    def analyze(self, m, deficits):
        active = []
        forced = 0
        for j, need in enumerate(deficits):
            if not need:
                continue
            remain = tuple(i for i in self.seps[j] if not m >> i & 1)
            self.stats["pair_terms"] += len(self.seps[j])
            if len(remain) < need:
                return None, ["missing", j], (), 0
            active.append((len(remain)-need, len(remain), j, need, remain))
            if len(remain) == need:
                for i in remain:
                    forced |= self.closures[i]
        active.sort()
        chosen = active[0]
        branch = tuple(sorted({m | self.closures[i] for i in chosen[4]}))
        lower, witness = 0, ["cost"]
        if self.arm == "cover":
            return lower, witness, branch, forced
        self.stats["bound_calls"] += 1
        for _, _, j, need, remain in active:
            cost = sum(sorted(self.costs[i] for i in remain)[:need])
            if cost > lower:
                lower, witness = cost, ["pair", j]
        # Aggregate fractional-cover relaxation, with a dual-price witness.
        target = sum(deficits)
        gains = [(self.costs[i], sum(bool(self.smasks[j] >> i & 1) for j, d in enumerate(deficits) if d))
                 for i in self.indices(self.full ^ m)]
        gains = [(Fraction(c, g), g) for c, g in gains if g]
        value = Fraction(0)
        for price, gain in sorted(gains):
            take = min(gain, target)
            value += price*take
            target -= take
            if not target:
                if math.ceil(value) > lower:
                    lower = math.ceil(value)
                    witness = ["fractional", price.numerator, price.denominator]
                break
        # Exact single-pair union-cost relaxation. Skip rather than truncate
        # a minimization if the remaining work allowance is insufficient.
        todo = active if self.arm == "closure_all" else active[:1] if self.arm == "closure_one" else []
        for _, _, j, need, remain in todo:
            count = math.comb(len(remain), need)
            if self.stats["closure_terms"]+count > self.term_cap:
                self.stats["closure_skips"] += 1
                continue
            best = self.d["budget"]+sum(self.costs)+1
            for combo in combinations(remain, need):
                union = 0
                for i in combo:
                    union |= self.closures[i]
                best = min(best, self.cost(union & ~m))
            self.stats["closure_terms"] += count
            if best > lower:
                lower, witness = best, ["closure", j]
        return lower, witness, branch, forced

    def push(self, m):
        if m in self.proof:
            return
        if self.stats["inspected"] >= self.cap:
            raise Cutoff
        self.stats["inspected"] += 1
        cost = self.cost(m)
        deficits = tuple(max(0, self.need-(m&s).bit_count()) for s in self.smasks)
        if cost >= self.best_cost:
            self.proof[m] = ["cost"]
            return
        if not any(deficits):
            self.best, self.best_cost = m, cost
            self.proof[m] = ["cost"]
            return
        lb, witness, branch, forced = self.analyze(m, deficits)
        if lb is None or cost+lb >= self.best_cost:
            self.proof[m] = witness
            self.stats["pruned"] += 1
            return
        self.proof[m] = None
        heapq.heappush(self.queue, (cost+lb, sum(deficits), cost, m, witness, branch, forced))

    def run(self):
        certified = False
        try:
            self.push(0)
            while self.queue:
                lower, _, _, m, witness, branch, forced = heapq.heappop(self.queue)
                if lower >= self.best_cost:
                    self.proof[m] = witness
                    self.stats["pruned"] += 1
                    continue
                self.stats["expanded"] += 1
                if self.arm != "cover" and forced & ~m:
                    self.proof[m] = ["force"]
                    children = (m | forced,)
                    self.stats["forced_steps"] += 1
                else:
                    # Keep the explicit pair index in the proof, independent
                    # of the search queue's ordering.
                    eligible = [(len(s)-need, len(s), j) for j, sep in enumerate(self.seps)
                                for need in [max(0, self.need-(m&self.smasks[j]).bit_count())]
                                for s in [tuple(i for i in sep if not m >> i & 1)] if need]
                    self.proof[m] = ["branch", min(eligible)[2]]
                    children = branch
                for nxt in children:
                    self.stats["generated"] += 1
                    self.push(nxt)
            certified = True
        except Cutoff:
            pass
        if self.best is not None:
            answer = dict(status="plan", experiments=sorted(self.rows[i]["id"] for i in self.indices(self.best)),
                          cost=self.best_cost)
        else:
            answer = dict(status="infeasible") if certified else None
        proof = (dict(schema="rv23c.cover_proof.v1", root_id=world_id(KIND, self.d), answer=answer,
                      nodes=[[m, *v] for m, v in sorted(self.proof.items())]) if certified else None)
        return answer, dict(**self.stats, certified=int(certified)), proof


def solve(d, arm, cap, term_cap=MAX_TERMS):
    return Search(d, arm, cap, term_cap).run()


def verify_plan(d, answer):
    """Direct set-based feasibility check, independent of Search."""
    if type(answer) is not dict or set(answer) != {"status", "experiments", "cost"}:
        raise ValueError("plan shape")
    ids = answer["experiments"]
    lookup = {r["id"]: r for r in d["experiments"]}
    if (answer["status"] != "plan" or type(ids) is not list or not all(type(x) is str for x in ids)
            or ids != sorted(set(ids)) or not set(ids) <= set(lookup)):
        raise ValueError("plan IDs")
    rows = [lookup[x] for x in ids]
    cost = sum(r["cost"] for r in rows)
    if (type(answer["cost"]) is not int or cost != answer["cost"] or cost > d["budget"]
            or any(not set(r["requires"]) <= set(ids) for r in rows)):
        raise ValueError("plan resources")
    if any(sum(r["predictions"][i] != r["predictions"][j] for r in rows) < 2*d["error_bits"]+1
           for i, j in combinations(range(len(d["mechanisms"])), 2)):
        raise ValueError("plan separation")
    return cost


def verify(d, proof):
    """Check a finite proof DAG; does not call Search, native oracle, or fit."""
    validate(d)
    if (type(proof) is not dict or set(proof) != {"schema", "root_id", "answer", "nodes"}
            or proof["schema"] != "rv23c.cover_proof.v1" or proof["root_id"] != world_id(KIND, d)):
        raise ValueError("proof binding")
    answer = proof["answer"]
    cutoff = d["budget"]+1 if answer == {"status": "infeasible"} else verify_plan(d, answer)
    rows = d["experiments"]
    n = len(rows)
    full = (1 << n)-1
    lookup = {r["id"]: i for i, r in enumerate(rows)}
    costs = [r["cost"] for r in rows]
    deps = [{lookup[x] for x in r["requires"]} for r in rows]
    pairs = list(combinations(range(len(d["mechanisms"])), 2))
    separators = [{i for i, r in enumerate(rows) if r["predictions"][a] != r["predictions"][b]}
                  for a, b in pairs]
    need = 2*d["error_bits"]+1

    def subset(m):
        return {i for i in range(n) if m >> i & 1}

    def closure(s):
        out = set(s)
        while True:
            nxt = out.union(*(deps[i] for i in out))
            if nxt == out:
                return out
            out = nxt

    def price(s):
        return sum(costs[i] for i in s)

    def encode(s):
        return sum(1 << i for i in s)

    nodes = proof["nodes"]
    if type(nodes) is not list or not 1 <= len(nodes) <= 1 << n:
        raise ValueError("proof node bound")
    records = {}
    for row in nodes:
        if (type(row) is not list or not 2 <= len(row) <= 4 or type(row[0]) is not int
                or not 0 <= row[0] <= full or row[0] in records or type(row[1]) is not str):
            raise ValueError("proof node shape")
        records[row[0]] = row[1:]
    stack, seen = [0], set()
    combinations_checked = 0
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        if m not in records:
            raise ValueError("missing branch")
        seen.add(m)
        sel = subset(m)
        if closure(sel) != sel:
            raise ValueError("unclosed proof node")
        c = price(sel)
        deficits = [max(0, need-len(sel&s)) for s in separators]
        reason, *args = records[m]
        kids = []
        if reason == "cost":
            ok = not args and c >= cutoff
        elif reason in ("missing", "pair", "closure", "branch"):
            if len(args) != 1 or type(args[0]) is not int or not 0 <= args[0] < len(pairs):
                raise ValueError("pair witness")
            j = args[0]
            left, req = separators[j]-sel, deficits[j]
            if not req:
                raise ValueError("nondeficient pair")
            if reason == "missing":
                ok = len(left) < req
            elif len(left) < req:
                raise ValueError("unavailable pair witness")
            elif reason == "pair":
                ok = c+sum(sorted(costs[i] for i in left)[:req]) >= cutoff
            elif reason == "closure":
                combinations_checked += math.comb(len(left), req)
                if combinations_checked > MAX_TERMS:
                    raise ValueError("proof combination bound")
                lower = min(price(closure(set(xs))-sel) for xs in combinations(sorted(left), req))
                ok = c+lower >= cutoff
            else:
                kids = [encode(closure(sel | {i})) for i in left]
                ok = bool(kids)
        elif reason == "force":
            mandatory = set()
            for j, req in enumerate(deficits):
                left = separators[j]-sel
                if req and len(left) == req:
                    mandatory |= left
            kids = [encode(closure(sel | mandatory))] if mandatory else []
            ok = not args and bool(kids)
        elif reason == "fractional":
            if (len(args) != 2 or any(type(x) is not int for x in args)
                    or not 0 < args[0] <= 1000 or not 0 < args[1] <= len(pairs)):
                raise ValueError("dual price")
            lam = Fraction(*args)
            # A feasible fractional-cover dual objective, independently
            # verified without reproducing the search's sorting operation.
            lower = lam*sum(deficits)
            for i in range(n):
                if i not in sel:
                    gain = sum(bool(req) and i in s for req, s in zip(deficits, separators, strict=True))
                    lower -= max(0, lam*gain-costs[i])
            ok = c+math.ceil(lower) >= cutoff
        else:
            raise ValueError("unknown proof reason")
        if not ok or any(k == m or k&m != m for k in kids):
            raise ValueError("invalid pruning or branching witness")
        stack.extend(kids)
    if seen != set(records):
        raise ValueError("orphan proof node")
    return dict(nodes=len(seen), closure_terms=combinations_checked, verified=True)


def exhaustive(d):
    """Independent finite subset oracle, for evaluation only."""
    rows = d["experiments"]
    n = len(rows)
    ids = {r["id"]: i for i, r in enumerate(rows)}
    deps = [sum(1 << ids[x] for x in r["requires"]) for r in rows]
    prices, requirements = [0]*(1 << n), [0]*(1 << n)
    masks = [sum(1 << i for i, r in enumerate(rows) if r["predictions"][a] != r["predictions"][b])
             for a, b in combinations(range(len(d["mechanisms"])), 2)]
    best = d["budget"]+1
    for m in range(1, 1 << n):
        b = m & -m
        i = b.bit_length()-1
        prices[m] = prices[m^b]+rows[i]["cost"]
        requirements[m] = requirements[m^b] | deps[i]
        if prices[m] < best and not requirements[m] & ~m and all(
                (m&s).bit_count() >= 2*d["error_bits"]+1 for s in masks):
            best = prices[m]
    return None if best > d["budget"] else best


def structured(seed, transitive):
    rng = random.Random(seed)
    n = 6
    rows = []
    for i in range(4):
        req = [f"c{i-1}"] if transitive and i else []
        rows.append(dict(id=f"c{i}", cost=rng.randint(2, 6), requires=req, predictions=[0]*n))
    for i in range(12):
        req = [f"c{rng.randrange(4)}"]
        if transitive and i and rng.random() < .4:
            req.append(f"e{rng.randrange(i)}")
        bits = [rng.randrange(2) for _ in range(n)]
        rows.append(dict(id=f"e{i}", cost=rng.randint(1, 4), requires=req, predictions=bits))
    return dict(mechanisms=[f"h{i}" for i in range(n)], experiments=rows, error_bits=1,
                budget=rng.randint(18, 42))


def campaign(generator, reference, n_native=24, n_structured=12):
    previous = {world_id(KIND, generator(KIND, seed, i, ("standard", "frontier")[i%2]))
                for seed, count in ((2026811000, 24), (2026812000, 32)) for i in range(count)}
    previous.update(world_id(KIND, generator(KIND, 2026821000+1000*j, i, diff))
                    for j, diff in enumerate(("foundation", "standard", "frontier")) for i in range(24))
    seen = set(previous)
    rng = random.Random(SEED+9999)
    report = dict(schema="rv23c.closure_proofs.v1", upstream=UPSTREAM, prior_excluded=len(previous),
                  limits=list(LIMITS), closure_term_cap=MAX_TERMS, cohorts={}, training_admission=False)
    searches = 0
    for ci, kind in enumerate(COHORTS):
        count = n_native if ci == 0 else n_structured
        samples = {str(cap): {arm: [] for arm in ARMS} for cap in LIMITS}
        ids = []
        infeasible = 0
        for j in range(count):
            d = (generator(KIND, SEED, j, "frontier") if ci == 0
                 else structured(SEED+1000*ci+j, ci == 2))
            rid = world_id(KIND, d)
            if rid in seen:
                raise ValueError("prior or cross-cohort root overlap")
            seen.add(rid)
            ids.append(rid)
            optimum = exhaustive(d)
            if ci == 0 and reference(dict(kind=KIND, data=d)).get("cost") != optimum:
                raise ValueError("native oracle parity")
            infeasible += int(optimum is None)
            order = [(cap, arm) for cap in LIMITS for arm in ARMS]
            rng.shuffle(order)
            for cap, arm in order:
                start = time.process_time_ns()
                answer, stats, proof = solve(d, arm, cap)
                search_ms = (time.process_time_ns()-start)/1e6
                cost = verify_plan(d, answer) if answer is not None and answer["status"] == "plan" else None
                success = (answer is not None and ((optimum is None and answer["status"] == "infeasible")
                           or (cost is not None and cost == optimum)))
                start = time.process_time_ns()
                receipt = verify(d, proof) if proof is not None else None
                check_ms = (time.process_time_ns()-start)/1e6 if proof is not None else 0.
                if proof is not None and not success:
                    raise ValueError("unsound certification")
                if cap == LIMITS[-1] and not (proof is not None and success):
                    raise ValueError("full-cap exactness")
                if stats["inspected"] > cap or stats["closure_terms"] > MAX_TERMS:
                    raise ValueError("meter cap")
                samples[str(cap)][arm].append(dict(**stats, optimal=int(success), answered=int(answer is not None),
                    search_ms=search_ms, check_ms=check_ms, combined_ms=search_ms+check_ms,
                    proof_bytes=len(json.dumps(proof,separators=(",",":")).encode()) if proof else 0,
                    check_closure_terms=receipt["closure_terms"] if receipt else 0))
                searches += 1
        results = {}
        idx = np.random.default_rng(217+ci).integers(count, size=(10000,count))
        for cap, arms in samples.items():
            means = {a: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} for a, rs in arms.items()}
            paired = {}
            for a in ("closure_one", "closure_all"):
                paired[a] = {}
                for metric in ("optimal", "inspected", "search_ms", "combined_ms"):
                    delta = np.asarray([u[metric]-v[metric] for u,v in zip(arms[a],arms["forced"],strict=True)])
                    paired[a][metric] = dict(delta=float(delta.mean()),
                        ci95=np.quantile(delta[idx].mean(axis=1),[.025,.975]).tolist())
            results[cap] = dict(means=means, closure_vs_forced=paired)
        report["cohorts"][kind] = dict(roots=count, infeasible=infeasible,
            inventory_sha256=hashlib.sha256("".join(sorted(ids)).encode()).hexdigest(), limits=results)
        print("RV23C_COHORT "+json.dumps(dict(cohort=kind, **report["cohorts"][kind]),separators=(",",":")),
              flush=True)
    report["searches"] = searches
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--distill-checkout",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a = p.parse_args()
    path = a.distill_checkout.resolve()
    sha = subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()
    dirty = subprocess.check_output(["git","-C",str(path),"status","--porcelain"],text=True).strip()
    if sha != UPSTREAM or dirty or a.out.exists():
        raise ValueError("clean pinned checkout and fresh output required")
    sys.path.insert(0,str(path))
    from Distill.research_cases import _data
    from Distill.research_oracle import reference
    report = campaign(_data,reference)
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("RV23C_RECEIPT "+json.dumps(report,separators=(",",":"),allow_nan=False),flush=True)


if __name__ == "__main__":
    main()
