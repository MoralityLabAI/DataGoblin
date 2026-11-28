"""RV23-D: dependency-mask quotient, union atlas and cheap control lower bound."""
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

from .closure_proof import KIND, MAX_TERMS, Search, exhaustive, structured, validate, verify_plan
from .skill_transfer import UPSTREAM, world_id

ARMS = ("cover", "forced", "closure_all", "control", "quotient", "quotient_cached", "atlas")
COHORTS = ("native", "shared", "transitive", "bundled")
CAPS = (64, 65536)
SEED = 2026841000
CACHE_ENTRIES = 256
ATLAS_MASKS = 4096


class Exhausted(Exception):
    pass


class CompactSearch(Search):
    def __init__(self, d, mode, cap, term_cap=MAX_TERMS):
        if mode not in ARMS:
            raise ValueError("unknown mode")
        super().__init__(d, mode if mode in ARMS[:3] else "forced", cap, term_cap)
        self.mode = mode
        self.memo = {}
        self.atlas = {}
        self.atlas_masks = 0
        self.stats.update(relax_terms=0, quotient_pops=0, quotient_duplicates=0, residual_hits=0,
                          atlas_hits=0, atlas_duplicates=0, cheap_terms=0, proof_terms_reserved=0,
                          fallback=0)

    def tick(self, n=1):
        if self.stats["relax_terms"]+n > self.term_cap:
            raise Exhausted
        self.stats["relax_terms"] += n

    def control(self, m, remain, req):
        """Costs of req distinct separators plus a mandatory nonseparator floor."""
        residual = [self.closures[i] & ~m for i in remain]
        direct = sum(sorted(self.costs[i] for i in remain)[:req])
        sep = sum(1 << i for i in remain)
        overhead = sorted(self.cost(x & ~sep) for x in residual)[req-1]
        union = sorted(self.cost(x) for x in residual)[req-1]
        self.stats["cheap_terms"] += 3*len(remain)
        return max(direct+overhead,union)

    def quotient(self, m, j, req, cached):
        sep = self.smasks[j] & ~m
        generators = tuple(sorted({self.closures[i] & ~m for i in self.seps[j] if sep >> i & 1}))
        signature = sep,req,generators
        if cached and signature in self.memo:
            self.stats["residual_hits"] += 1
            return self.memo[signature]
        # Every residual union is a state; duplicate choices leading to the
        # same union have identical price and future possible extensions.
        fringe, seen = [(0,0)], {0}
        while fringe:
            cost,u = heapq.heappop(fringe)
            self.stats["quotient_pops"] += 1
            if (u&sep).bit_count() >= req:
                if cached and len(self.memo) < CACHE_ENTRIES:
                    self.memo[signature] = cost
                return cost
            for g in generators:
                self.tick()
                v = u | g
                if v in seen:
                    self.stats["quotient_duplicates"] += 1
                    continue
                seen.add(v)
                heapq.heappush(fringe,(self.cost(v),v))
        raise ValueError("available pair has no closure union")

    def atlas_bound(self, m, j):
        sep = self.smasks[j]
        if sep in self.atlas:
            self.stats["atlas_hits"] += 1
            masks = self.atlas[sep]
        else:
            count = math.comb(len(self.seps[j]),self.need)
            self.tick(count)
            unions = set()
            for combo in combinations(self.seps[j],self.need):
                u = 0
                for i in combo:
                    u |= self.closures[i]
                unions.add(u)
            masks = tuple(sorted(unions))
            self.stats["atlas_duplicates"] += count-len(masks)
            if self.atlas_masks+len(masks) <= ATLAS_MASKS:
                self.atlas[sep] = masks
                self.atlas_masks += len(masks)
        self.tick(len(masks))
        return min(self.cost(u & ~m) for u in masks)

    def analyze(self, m, deficits):
        lower,witness,branch,forced = super().analyze(m,deficits)
        if lower is None or self.mode in ARMS[:3]:
            return lower,witness,branch,forced
        active = []
        for j,req in enumerate(deficits):
            if not req:
                continue
            remain = tuple(i for i in self.seps[j] if not m >> i & 1)
            active.append((len(remain)-req,len(remain),j,req,remain))
            floor = self.control(m,remain,req)
            if floor > lower:
                lower,witness = floor,["control",j]
        # Skip exact relaxation when the cheap bound already closes the node.
        if self.mode == "control" or self.cost(m)+lower >= self.best_cost:
            return lower,witness,branch,forced
        cheap = lower,witness
        for _,_,j,req,remain in sorted(active):
            # The independently enumerating checker has its own fixed cap.
            # Reserve every chosen exact witness, even if it later branches.
            proof_cost = math.comb(len(remain),req)
            if self.stats["proof_terms_reserved"]+proof_cost > MAX_TERMS:
                self.stats["fallback"] += 1
                continue
            try:
                exact = (self.atlas_bound(m,j) if self.mode == "atlas"
                         else self.quotient(m,j,req,self.mode == "quotient_cached"))
            except Exhausted:
                self.stats["fallback"] += 1
                break
            if exact > lower:
                lower,witness = exact,["closure",j]
                self.stats["proof_terms_reserved"] += proof_cost
            if self.cost(m)+lower >= self.best_cost:
                break
        if lower < cheap[0]:
            raise ValueError("weakened bound")
        return lower,witness,branch,forced

    def run(self):
        answer,stats,proof = super().run()
        if proof is not None:
            proof["schema"] = "rv23d.cover_proof.v2"
        # Shallow retained cache estimate (includes key/value objects, may
        # double-count shared integers; excludes search/proof/scratch/RSS).
        stats["cache_bytes_estimate"] = (
            sys.getsizeof(self.memo)+sys.getsizeof(self.atlas)
            +sum(sys.getsizeof(k)+sys.getsizeof(k[2])+sum(sys.getsizeof(x) for x in k[2])
                 +sys.getsizeof(v) for k,v in self.memo.items())
            +sum(sys.getsizeof(k)+sys.getsizeof(v)+sum(sys.getsizeof(x) for x in v)
                 for k,v in self.atlas.items()))
        stats["cache_entries"] = len(self.memo)
        stats["atlas_masks"] = self.atlas_masks
        return answer,stats,proof


def solve(d, mode, cap, term_cap=MAX_TERMS):
    return CompactSearch(d,mode,cap,term_cap).run()


def verify(d, proof):
    """Check a finite proof DAG; does not call Search, native oracle, or fit."""
    validate(d)
    if (type(proof) is not dict or set(proof) != {"schema", "root_id", "answer", "nodes"}
            or proof["schema"] not in ("rv23c.cover_proof.v1", "rv23d.cover_proof.v2")
            or proof["root_id"] != world_id(KIND, d)):
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
        elif reason in ("missing", "pair", "closure", "branch", "control"):
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
            elif reason == "control":
                if proof["schema"] != "rv23d.cover_proof.v2":
                    raise ValueError("control witness version")
                # Independent set-based recomputation: choose req separators.
                residual = [closure({i})-sel for i in left]
                direct = sum(sorted(costs[i] for i in left)[:req])
                overhead = sorted(price(x-left) for x in residual)[req-1]
                union_floor = sorted(price(x) for x in residual)[req-1]
                ok = c+max(direct+overhead, union_floor) >= cutoff
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



def bundled(seed):
    """Co-selection cycles intentionally create equal closure masks."""
    rng = random.Random(seed)
    d = structured(seed,False)
    for j in range(0,12,3):
        for i in range(j,j+3):
            d["experiments"][4+i]["requires"].append(f"e{j+(i-j+1)%3}")
    d["budget"] = rng.randint(24,60)
    return d


def campaign(generator, reference, n_native=24, n_structured=16):
    # Freeze all prior raw content identities, including C's structured roots.
    previous = {world_id(KIND,generator(KIND,seed,i,("standard","frontier")[i%2]))
                for seed,count in ((2026811000,24),(2026812000,32)) for i in range(count)}
    previous.update(world_id(KIND,generator(KIND,2026821000+1000*j,i,diff))
                    for j,diff in enumerate(("foundation","standard","frontier")) for i in range(24))
    previous.update(world_id(KIND,generator(KIND,2026831000,i,"frontier")) for i in range(24))
    previous.update(world_id(KIND,structured(2026831000+1000*j+i,j==2))
                    for j in (1,2) for i in range(12))
    seen = set(previous)
    rng = random.Random(SEED+9999)
    result = dict(schema="rv23d.dependency_compact.v1",upstream=UPSTREAM,prior_excluded=len(previous),
                  term_cap=MAX_TERMS,caps=list(CAPS),cohorts={},training_admission=False)
    searches = 0
    for ci,kind in enumerate(COHORTS):
        count = n_native if ci == 0 else n_structured
        rows = {str(cap):{a:[] for a in ARMS} for cap in CAPS}
        roots = []
        for i in range(count):
            seed = SEED+1000*ci+i
            d = (generator(KIND,SEED,i,"frontier") if ci == 0 else
                 bundled(seed) if ci == 3 else structured(seed,ci==2))
            rid = world_id(KIND,d)
            if rid in seen:
                raise ValueError("root overlap")
            seen.add(rid)
            roots.append(rid)
            optimum = exhaustive(d)
            if ci == 0 and reference(dict(kind=KIND,data=d)).get("cost") != optimum:
                raise ValueError("native parity")
            order = [(cap,a) for cap in CAPS for a in ARMS]
            rng.shuffle(order)
            for cap,arm in order:
                start = time.process_time_ns()
                answer,stats,proof = solve(d,arm,cap)
                search_ms = (time.process_time_ns()-start)/1e6
                cost = verify_plan(d,answer) if answer is not None and answer["status"]=="plan" else None
                optimal = answer is not None and (
                    (answer["status"]=="infeasible" and optimum is None) or
                    (cost is not None and cost==optimum))
                start = time.process_time_ns()
                receipt = verify(d,proof) if proof is not None else None
                check_ms = (time.process_time_ns()-start)/1e6 if proof is not None else 0.
                if (proof is not None and not optimal) or (cap==CAPS[-1] and not (proof and optimal)):
                    raise ValueError("unsound or incomplete full search")
                if (stats["inspected"]>cap or stats["relax_terms"]>MAX_TERMS
                        or stats["closure_terms"]>MAX_TERMS or stats["cache_entries"]>CACHE_ENTRIES
                        or stats["atlas_masks"]>ATLAS_MASKS):
                    raise ValueError("meter cap")
                rows[str(cap)][arm].append(dict(**stats, optimal=int(optimal),
                    search_ms=search_ms,check_ms=check_ms,combined_ms=search_ms+check_ms,
                    proof_bytes=len(json.dumps(proof,separators=(",",":")).encode()) if proof else 0,
                    check_terms=receipt["closure_terms"] if receipt else 0,
                    infeasible=int(optimum is None)))
                searches += 1
        idx = np.random.default_rng(991+ci).integers(count,size=(10000,count))
        limits = {}
        for cap,arms in rows.items():
            means = {a:{k:float(np.mean([r[k] for r in rs])) for k in rs[0]} for a,rs in arms.items()}
            paired = {}
            for a in ARMS[2:]:
                paired[a] = {}
                for b in ("forced","closure_all","control","quotient"):
                    if a == b:
                        continue
                    paired[a][b] = {}
                    for metric in ("inspected","combined_ms","search_ms","optimal"):
                        delta = np.asarray([u[metric]-v[metric] for u,v in zip(arms[a],arms[b],strict=True)])
                        paired[a][b][metric] = dict(delta=float(delta.mean()),
                            ci95=np.quantile(delta[idx].mean(axis=1),[.025,.975]).tolist())
            limits[cap] = dict(means=means,paired=paired)
        result["cohorts"][kind] = dict(roots=count,inventory_sha256=hashlib.sha256(
            "".join(sorted(roots)).encode()).hexdigest(),limits=limits)
        print("RV23D_COHORT "+json.dumps(dict(cohort=kind,**result["cohorts"][kind]),separators=(",",":")),
              flush=True)
    result["searches"] = searches
    return result


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
    print("RV23D_RECEIPT "+json.dumps(report,separators=(",",":"),allow_nan=False),flush=True)


if __name__ == "__main__":
    main()
