"""RV23-B: bounded dependency-aware robust experiment design."""
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

from .skill_transfer import EVAL as OLD_EVAL
from .skill_transfer import TRAIN as OLD_TRAIN
from .skill_transfer import UPSTREAM, Problem, search, world_id

KIND = 'robust_discrimination'
SEED = 2026821000
LIMITS = (64, 256, 4096)
ARMS = ('legacy', 'greedy', 'cover', 'bound', 'forced', 'warm_forced')
DIFFICULTIES = ('foundation', 'standard', 'frontier')


class LimitReached(Exception):
    pass


class Design:
    """Positive-cost dependency-closed binary separation multicover."""
    def __init__(self, d):
        self.d = d
        self.items = d['experiments']
        self.n = len(self.items)
        if not 1 <= self.n <= 12:
            raise ValueError('native subset domain')
        self.full = (1 << self.n)-1
        self.ids = {r['id']: i for i,r in enumerate(self.items)}
        self.costs = [r['cost'] for r in self.items]
        self.deps = [sum(1 << self.ids[x] for x in r['requires']) for r in self.items]
        self.closures = [self.close(1 << i) for i in range(self.n)]
        self.budget = d['budget']
        self.need = 2*d['error_bits']+1
        pairs = list(combinations(range(len(d['mechanisms'])),2))
        self.cover = [sum(1 << j for j,(a,b) in enumerate(pairs) if r['predictions'][a]!=r['predictions'][b])
                      for r in self.items]
        self.separators = [tuple(i for i in range(self.n) if self.cover[i] >> j & 1)
                           for j in range(len(pairs))]
        self.width = len(pairs)
        if not self.width or min(self.costs) <= 0:
            raise ValueError('unsupported separation model')

    def indices(self, m):
        return [i for i in range(self.n) if m >> i & 1]

    def close(self, m):
        while True:
            old = m
            for i in self.indices(m):
                m |= self.deps[i]
            if m == old:
                return m

    def record(self, m):
        """Cost and exact capped deficits; candidates are dependency-closed."""
        ix = self.indices(m)
        cost = sum(self.costs[i] for i in ix)
        deficits = tuple(max(0,self.need-sum(self.cover[i] >> j & 1 for i in ix))
                         for j in range(self.width))
        return cost,deficits

    def analyze(self, m, deficits, bounded=True):
        """Necessary availability, relaxed remaining-cost bound, and branching options.

        The fractional bound ignores dependencies and per-pair allocation,
        so it can underestimate the cost of any feasible completion.
        """
        active = [(j,need) for j,need in enumerate(deficits) if need]
        if not active:
            return 0,(),0,0
        options = []
        pair_lb, mandatory, terms = 0,0,0
        for j,need in active:
            remaining = tuple(i for i in self.separators[j] if not m >> i & 1)
            terms += len(self.separators[j])
            if len(remaining) < need:
                return None,(),0,terms
            if len(remaining) == need:
                for i in remaining:
                    mandatory |= self.closures[i]
            if bounded:
                pair_lb = max(pair_lb,sum(sorted(self.costs[i] for i in remaining)[:need]))
            options.append((len(remaining)-need,len(remaining),j,remaining))
        choices = min(options)[3]
        if not bounded:
            return 0,choices,mandatory,terms
        # Each item contributes at most one unit per presently deficient pair.
        target = sum(deficits)
        active_bits = sum(1 << j for j,_ in active)
        gains = []
        for i in self.indices(self.full ^ m):
            g = (self.cover[i]&active_bits).bit_count()
            terms += 1
            if g:
                gains.append((Fraction(self.costs[i],g),g))
        lower = Fraction(0)
        for price,gain in sorted(gains):
            take = min(gain,target)
            lower += price*take
            target -= take
            if not target:
                break
        if target:
            return None,(),0,terms
        return max(pair_lb,math.ceil(lower)),choices,mandatory,terms

    def answer(self, mask, certified):
        if mask is None:
            return {'status':'infeasible'} if certified else None
        return dict(status='plan',experiments=sorted(self.items[i]['id'] for i in self.indices(mask)),
                    cost=sum(self.costs[i] for i in self.indices(mask)))


class Meter:
    def __init__(self, p, cap):
        self.p,self.cap = p,cap
        self.records = {}
        self.best = None
        self.best_cost = p.budget+1
        self.stats = dict(expanded=0, bound_calls=0, bound_terms=0, pruned=0,
                          forced_steps=0, generated=0)

    def inspect(self, m):
        if m not in self.records:
            if len(self.records) >= self.cap:
                raise LimitReached
            cost,deficits = self.p.record(m)
            self.records[m] = cost,deficits
            if cost < self.best_cost and not any(deficits):
                self.best,self.best_cost = m,cost
        return self.records[m]


def greedy(p, meter):
    m = 0
    cost,deficits = meter.inspect(m)
    while any(deficits):
        best = None
        for nxt in sorted({m | c for c in p.closures} - {m}):
            meter.stats['generated'] += 1
            price,ds = meter.inspect(nxt)
            gain = sum(deficits)-sum(ds)
            if price <= p.budget and gain > 0:
                rank = Fraction(gain,price-cost),gain,-price,-nxt
                if best is None or rank > best[0]:
                    best = rank,nxt,price,ds
        if best is None:
            return
        _,m,cost,deficits = best


def branch(p, meter, bounded, forced):
    fringe,seen = [],set()

    def push(m):
        if m in seen:
            return
        seen.add(m)
        cost,deficits = meter.inspect(m)
        if cost >= meter.best_cost or not any(deficits):
            return
        lb,choices,mandatory,terms = p.analyze(m,deficits,bounded)
        meter.stats['bound_calls'] += 1
        meter.stats['bound_terms'] += terms
        if lb is None or cost+lb >= meter.best_cost:
            meter.stats['pruned'] += 1
            return
        heapq.heappush(fringe,(cost+lb,sum(deficits),cost,m,choices,mandatory))

    push(0)
    while fringe:
        lower,_,_,m,choices,mandatory = heapq.heappop(fringe)
        if lower >= meter.best_cost:
            meter.stats['pruned'] += 1
            continue
        meter.stats['expanded'] += 1
        if forced and mandatory & ~m:
            meter.stats['forced_steps'] += 1
            successors = (m | mandatory,)
        else:
            successors = sorted({m | p.closures[i] for i in choices})
        for nxt in successors:
            meter.stats['generated'] += 1
            push(nxt)


def solve(d, arm, cap):
    if arm not in ARMS or cap < 1:
        raise ValueError('unknown arm or inspection cap')
    if arm == 'legacy':
        answer,_,work,certified = search(Problem(KIND,d),'analytic',cap)
        return answer,dict(inspected=work,certified=int(certified),expanded=0,bound_calls=0,
                           bound_terms=0,pruned=0,forced_steps=0,generated=0)
    p = Design(d)
    meter = Meter(p,cap)
    certified = False
    try:
        if arm in ('greedy','warm_forced'):
            greedy(p,meter)
        if arm != 'greedy':
            branch(p,meter,arm!='cover',arm in ('forced','warm_forced'))
            certified = True
    except LimitReached:
        pass
    return p.answer(meter.best,certified),dict(inspected=len(meter.records),certified=int(certified),**meter.stats)


def campaign(generator, reference, assess, n=24):
    # Reconstruct prior A's robust train/eval roots and exclude by full content ID.
    previous = {world_id(KIND,generator(KIND,seed,i,('standard','frontier')[i%2]))
                for seed,count in ((OLD_TRAIN,24),(OLD_EVAL,32)) for i in range(count)}
    seen = set(previous)
    report = dict(schema='rv23b.deficit_cover.v1',upstream=UPSTREAM,seed=SEED,roots_per_difficulty=n,
                  limits=list(LIMITS),prior_excluded=len(previous),cohorts={},training_admission=False)
    rng = random.Random(2026829999)
    for ci,difficulty in enumerate(DIFFICULTIES):
        rows = {str(cap):{a:[] for a in ARMS} for cap in LIMITS}
        ids = []
        for i in range(n):
            d = generator(KIND,SEED+ci*1000,i,difficulty)
            rid = world_id(KIND,d)
            if rid in seen:
                raise ValueError('prior or cross-cohort content overlap')
            seen.add(rid)
            ids.append(rid)
            case = dict(kind=KIND,data=d)
            ref = reference(case)
            optimum = ref.get('cost')
            grades = {}
            order = [(cap,a) for cap in LIMITS for a in ARMS]
            rng.shuffle(order)
            for cap,arm in order:
                start = time.process_time_ns()
                answer,stats = solve(d,arm,cap)
                elapsed = (time.process_time_ns()-start)/1e6
                key = json.dumps(answer,sort_keys=True)
                if key not in grades:
                    grades[key] = assess(case,answer) if answer is not None else {}
                g = grades[key]
                success = int(g.get('optimal',False))
                if answer is not None and not g['feasible']:
                    raise ValueError('invalid returned certificate')
                if stats['certified'] and not success:
                    raise ValueError('unsound search certification')
                if stats['inspected'] > cap:
                    raise ValueError('inspection budget exceeded')
                if cap == max(LIMITS) and arm in ('cover','bound','forced','warm_forced'):
                    if not stats['certified'] or not success:
                        raise ValueError('exact full-budget parity')
                plan = answer is not None and answer.get('status') == 'plan'
                rows[str(cap)][arm].append(dict(**stats,cpu_ms=elapsed,optimal=success,
                    answered=int(answer is not None),plan=int(plan),oracle_infeasible=int(optimum is None),
                    cost_gap=(answer['cost']-optimum if plan else None)))
        out = {}
        idx = np.random.default_rng(629+ci).integers(n,size=(10000,n))
        for cap,arms in rows.items():
            means = {a:{k:float(np.mean([r[k] for r in rs])) for k in rs[0] if k!='cost_gap'}
                     for a,rs in arms.items()}
            paired = {}
            for a in ARMS[1:]:
                pairs = {}
                for b in ('legacy','cover','bound'):
                    if a == b:
                        continue
                    pair = {}
                    for metric in ('optimal','inspected','cpu_ms'):
                        delta = np.asarray([u[metric]-v[metric] for u,v in zip(arms[a],arms[b],strict=True)])
                        pair[metric] = dict(delta=float(delta.mean()),
                            ci95=np.quantile(delta[idx].mean(axis=1),[.025,.975]).tolist())
                    pairs[b] = pair
                paired[a] = pairs
            out[cap] = dict(means=means,paired=paired)
        report['cohorts'][difficulty] = dict(roots=n,inventory_sha256=hashlib.sha256(
            ''.join(sorted(ids)).encode()).hexdigest(),limits=out)
        print('RV23B_COHORT '+json.dumps(dict(difficulty=difficulty,**report['cohorts'][difficulty]),
                                           separators=(',',':')),flush=True)
    report['searches'] = n*len(DIFFICULTIES)*len(ARMS)*len(LIMITS)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distill-checkout',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    path = args.distill_checkout.resolve()
    sha = subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip()
    if sha!=UPSTREAM or dirty or args.out.exists():
        raise ValueError('clean pinned checkout and fresh output required')
    sys.path.insert(0,str(path))
    from Distill.research_cases import _data
    from Distill.research_oracle import assess, reference
    report = campaign(_data,reference,assess)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('RV23B_RECEIPT '+json.dumps(report,separators=(',',':'),allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
