"""RV23-A: leave-one-family-out transfer of bounded subset-search ranking."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np

UPSTREAM = '7fa19c90372b13d71676e6fef13819d2a18b9be3'
FAMILIES = ('robust_discrimination', 'research_portfolio', 'proof_obligations')
TRAIN, EVAL = 2026811000, 2026812000
ARMS = ('analytic', 'random', 'in_family', 'transfer', 'shuffled')
LIMITS = (64, 256)


class Problem:
    """Family-specific public semantics, shared search and learned ranker."""
    def __init__(self, kind, data):
        if kind not in FAMILIES:
            raise ValueError('unsupported subset family')
        self.kind, self.d = kind, data
        self.items = data['experiments'] if kind == FAMILIES[0] else data['jobs']
        self.n = len(self.items)
        if not 1 <= self.n <= 12:
            raise ValueError('bounded subset domain')
        self.full = (1 << self.n)-1
        self.ids = {r['id']: i for i, r in enumerate(self.items)}
        self.cost = [r['cpu'] if kind == FAMILIES[1] else r['cost'] for r in self.items]
        self.budget = data['cpu_budget'] if kind == FAMILIES[1] else data['budget']
        self.deps = [sum(1 << self.ids[x] for x in r.get('requires', [])) for r in self.items]
        self.closures = [self.close(1 << i) for i in range(self.n)]
        self.memo = {}
        self.current = self.claims(0) if kind == FAMILIES[2] else ([], 0)
        if kind == FAMILIES[0]:
            self.pairs = list(combinations(range(len(data['mechanisms'])), 2))
            self.need = 2*data['error_bits']+1
            self.cover = [[int(r['predictions'][i] != r['predictions'][j])
                           for i,j in self.pairs] for r in self.items]
        elif kind == FAMILIES[1]:
            self.upper = max(1, min(sum(max(0,r['gains'][j]) for r in self.items)
                                    for j in range(len(data['scenarios']))))
        else:
            self.upper = max(1, sum(c['value'] for c in data['claims'])-self.current[1])

    def indices(self, m):
        return [i for i in range(self.n) if m >> i & 1]

    def close(self, m):
        while True:
            old = m
            for i in self.indices(m):
                m |= self.deps[i]
            if m == old:
                return m

    def claims(self, m):
        resolved = {x for i in self.indices(m) for x in self.items[i]['resolves']}
        checked = set()
        # Native generator orders claims topologically.
        for c in self.d['claims']:
            if ((c['state'] == 'checked' or c['state'] == 'open' and c['id'] in resolved)
                    and set(c['requires']) <= checked):
                checked.add(c['id'])
        return sorted(checked), sum(c['value'] for c in self.d['claims'] if c['id'] in checked)

    def inspect(self, m):
        """Return (objective, progress, raw value, complete) or None if infeasible."""
        if m in self.memo:
            return self.memo[m]
        ix = self.indices(m)
        cost = sum(self.cost[i] for i in ix)
        if cost > self.budget or self.close(m) != m:
            out = None
        elif self.kind == FAMILIES[0]:
            counts = [sum(self.cover[i][j] for i in ix) for j in range(len(self.pairs))]
            progress = sum(min(self.need,c) for c in counts)/(self.need*len(counts))
            done = all(c >= self.need for c in counts)
            out = (1-cost/(self.budget+1) if done else 0., progress, cost, done)
        elif self.kind == FAMILIES[1]:
            jobs = [self.items[i] for i in ix]
            groups = [r['group'] for r in jobs if r['group'] is not None]
            if (len(groups) != len(set(groups))
                    or any(not r['permitted'] or r['ram'] > self.d['ram_cap'] for r in jobs)):
                out = None
            else:
                gains = [sum(r['gains'][j] for r in jobs) for j in range(len(self.d['scenarios']))]
                value = min(gains)
                progress = max(0., min(1., sum(gains)/len(gains)/self.upper))
                out = (max(0.,value)/self.upper,progress,value,True)
        else:
            _,value = self.claims(m)
            gain = (value-self.current[1])/self.upper
            resolved = {x for i in ix for x in self.items[i]['resolves']}
            potential = sum(c['value'] for c in self.d['claims']
                            if c['state'] == 'open' and c['id'] in resolved)/self.upper
            out = (gain,min(1.,max(gain,potential)),value,True)
        self.memo[m] = out
        return out

    def features(self, m, rec):
        score, progress, _, _ = rec
        cost = sum(self.cost[i] for i in self.indices(m))/self.budget
        fraction = m.bit_count()/self.n
        return (1.,score,progress,cost,fraction,1-cost,progress*(1-cost),
                score*(1-cost),progress**2,score**2,cost**2,self.n/12)

    def successors(self, m):
        return sorted({m | x for x in self.closures if (m | x) != m})

    def answer(self, m, exhausted):
        rec = self.inspect(m)
        ids = sorted(self.items[i]['id'] for i in self.indices(m))
        if self.kind == FAMILIES[0]:
            if rec[3]:
                return dict(status='plan', experiments=ids, cost=rec[2])
            return dict(status='infeasible') if exhausted else None
        if self.kind == FAMILIES[1]:
            return dict(jobs=ids,worst_gain=rec[2])
        claims,value = self.claims(m)
        return dict(current=self.current[0],jobs=ids,conditional=claims,conditional_value=value)

    def labels(self, seed):
        # Exact superset envelope. Feasibility of intermediate masks is not
        # needed for the transform: every feasible superset can be reached
        # by adding its dependency closures within its final resource budget.
        scores = np.full(self.full+1,-1.)
        for m in range(self.full+1):
            rec = self.inspect(m)
            if rec is not None:
                scores[m] = rec[0]
        best = scores.copy()
        for i in range(self.n):
            for m in range(self.full+1):
                if not m >> i & 1:
                    best[m] = max(best[m],best[m | 1 << i])
        feasible = np.flatnonzero(scores >= 0)
        rng = np.random.default_rng(seed)
        ids = rng.choice(feasible,size=64,replace=len(feasible)<64)
        x = [self.features(int(m),self.inspect(int(m))) for m in ids]
        y = best[ids]-scores[ids]
        return np.asarray(x),y


def search(p, arm, limit, weights=None, seed=0):
    """Charge every distinct inspected mask, feasible or not, to the same budget."""
    rng = np.random.default_rng(seed)
    seen, fringe = {0}, []
    root = p.inspect(0)
    best, best_score = 0,root[0]
    exhausted = True

    def rank(m, rec):
        f = p.features(m,rec)
        if arm == 'random':
            return float(rng.random())
        if weights is None:
            return f[1]+f[2]-.25*f[3]
        return rec[0]+max(0.,float(np.asarray(f)@weights))

    fringe.append((rank(0,root),0))
    while fringe:
        _,m = max(fringe,key=lambda x:(x[0],-x[1]))
        fringe.remove((_,m))
        if p.kind == FAMILIES[0] and p.inspect(m)[3]:
            continue
        for nxt in p.successors(m):
            if nxt in seen:
                continue
            if len(seen) >= limit:
                exhausted = False
                break
            seen.add(nxt)
            rec = p.inspect(nxt)
            if rec is None:
                continue
            # Empty is a feasible incumbent for maximization; never report
            # a negative portfolio when the score is tied with the empty set.
            if rec[0] > best_score or (rec[0] == best_score and rec[2] > p.inspect(best)[2]
                                       and p.kind == FAMILIES[1]):
                best,best_score = nxt,rec[0]
            fringe.append((rank(nxt,rec),nxt))
        if not exhausted:
            break
    return p.answer(best,exhausted),best_score,len(seen),exhausted


def fit(panels, seed, shuffle=False):
    rows = [Problem(k,d).labels(seed+i) for i,(k,d) in enumerate(panels)]
    x = np.concatenate([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    if shuffle:
        y = np.random.default_rng(seed).permutation(y)
    return np.linalg.solve(x.T@x+.5*np.eye(x.shape[1]),x.T@y)


def paired_answer(d, pooled=False):
    out, eligible = [],[]
    for arm in d['arms']:
        rows = arm['roots']
        delta = [Fraction(sum(r['active']),len(r['active']))-Fraction(sum(r['sham']),len(r['sham'])) for r in rows]
        mean = sum(delta,Fraction(0))/len(delta)
        if pooled:
            mean = (Fraction(sum(sum(r['active']) for r in rows),sum(len(r['active']) for r in rows))
                    -Fraction(sum(sum(r['sham']) for r in rows),sum(len(r['sham']) for r in rows)))
        positive = sum(x>0 for x in delta)
        matched = all(r['active_visits']==r['sham_visits'] and r['active_dose']==r['sham_dose'] for r in rows)
        out.append(dict(id=arm['id'],mean_delta=f'{mean.numerator}/{mean.denominator}',positive_roots=positive,
                        matched=matched))
        if matched and positive>=d['min_positive'] and mean>=Fraction(d['min_mean']):
            eligible.append(arm['id'])
    return dict(arms=out,eligible=sorted(eligible))


def world_id(kind,d):
    return hashlib.sha256(json.dumps(dict(kind=kind,data=d),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def campaign(generator, assess, reference, ntrain=24, neval=32):
    train = {k:[(k,generator(k,TRAIN,i,('standard','frontier')[i%2])) for i in range(ntrain)] for k in FAMILIES}
    ev = {k:[generator(k,EVAL,i,('standard','frontier')[i%2]) for i in range(neval)]
          for k in (*FAMILIES,'paired_causal_gate')}
    ids = [world_id(k,d) for rows in train.values() for k,d in rows]
    ids += [world_id(k,d) for k,rows in ev.items() for d in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError('split content overlap')
    out = dict(schema='rv23a.family_transfer.v1',upstream=UPSTREAM,train_roots_per_family=ntrain,
               eval_roots_per_family=neval,root_ids_sha256=hashlib.sha256(''.join(sorted(ids)).encode()).hexdigest(),
               limits=list(LIMITS),families={},training_admission=False)
    for k in FAMILIES:
        others = [j for j in FAMILIES if j != k]
        pool = [r for j in others for r in train[j][:ntrain//2]]
        heads = dict(in_family=fit(train[k],421),transfer=fit(pool,421),shuffled=fit(pool,421,True))
        metrics = {lim:{arm:[] for arm in ARMS} for lim in LIMITS}
        infeasible = 0
        for i,d in enumerate(ev[k]):
            case = dict(kind=k,data=d)
            oracle = reference(case)
            p = Problem(k,d)
            if k == FAMILIES[0]:
                opt = 0. if oracle['status']=='infeasible' else 1-oracle['cost']/(p.budget+1)
                infeasible += int(oracle['status']=='infeasible')
            elif k == FAMILIES[1]:
                opt = oracle['worst_gain']/p.upper
            else:
                opt = (oracle['conditional_value']-p.current[1])/p.upper
            for lim in LIMITS:
                for arm in ARMS:
                    answer,score,work,exhausted = search(Problem(k,d),arm,lim,heads.get(arm),seed=1000+i)
                    result = assess(case,answer) if answer is not None else {}
                    if answer is not None and not result['feasible']:
                        raise ValueError('infeasible returned certificate')
                    if score>opt+1e-9 or work>lim:
                        raise ValueError('objective or work bound')
                    metrics[lim][arm].append(dict(optimal=int(result.get('optimal',False)),
                        answered=int(answer is not None),regret=max(0.,opt-score),inspected=work,
                        exhausted=int(exhausted)))
        family = dict(oracle_infeasible=infeasible,limits={})
        for lim,arms in metrics.items():
            means = {a:{m:float(np.mean([r[m] for r in rows])) for m in rows[0]} for a,rows in arms.items()}
            idx = np.random.default_rng(427).integers(neval,size=(10000,neval))
            paired = {}
            for control in ('analytic','in_family','shuffled'):
                delta = np.asarray([u['optimal']-v['optimal'] for u,v in zip(
                    arms['transfer'],arms[control],strict=True)])
                paired[control] = dict(delta=float(delta.mean()),
                    ci95=np.quantile(delta[idx].mean(axis=1),[.025,.975]).tolist())
            family['limits'][str(lim)] = dict(means=means,transfer_vs=paired)
        out['families'][k] = family
        print('RV23A_FAMILY '+json.dumps(dict(kind=k,**family),separators=(',',':')),flush=True)
    paired = dict(roots=neval,exact=0,pooled=0,pooled_gate_mismatches=0)
    for d in ev['paired_causal_gate']:
        case = dict(kind='paired_causal_gate',data=d)
        a,b = paired_answer(d),paired_answer(d,True)
        paired['exact'] += int(assess(case,a)['optimal'])
        paired['pooled'] += int(assess(case,b)['optimal'])
        paired['pooled_gate_mismatches'] += int(a['eligible']!=b['eligible'])
    out['paired_causal_gate'] = paired
    return out


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
    report = campaign(_data,assess,reference)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('RV23A_RECEIPT '+json.dumps(report,separators=(',',':'),allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
