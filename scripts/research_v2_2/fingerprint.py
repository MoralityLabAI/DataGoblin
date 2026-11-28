"""RV22-R: structural prefilters; canonical equality authorizes reuse."""
from __future__ import annotations

import argparse
import json
import math
import random
import struct
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from activation import Machine, make_stream
from cold_bypass import CAP, DP, Monitor, canonical, check, enc, replay, variant, world

RESERVE, H, SLOTS = 384, 12, 16
ENTRY = struct.Struct('<QI')
KINDS = ('burst', 'mixed', 'churn', 'phase', 'recovery', 'scan', 'delayed', 'alias', 'hidden', 'collision')
ARMS = ('cold', 'trusted_lru', 'checked_lru', 'always_second', 'jitter1',
        'trusted_mass', 'trusted_marginal', 'checked_mass', 'checked_marginal', 'checked_marginal_confirm')
EVAL = 2026310000


def fingerprint(d, marginal=False):
    """Normalized prior masses, optionally one stable legal test's class marginals."""
    hs = d['hypotheses']
    a = sum(h['weight'] for h in hs if h['class'] == 'A')
    b = sum(h['weight'] for h in hs if h['class'] == 'B')
    base = d['budget'], d['correct_reward'], d['wrong_reward']
    if marginal:
        t = min((t for t in d['tests'] if t['cost'] <= d['budget']
                 and 0 in t['bits'] and 1 in t['bits']), key=lambda t: t['id'], default=None)
        if t is not None:
            a1 = b1 = 0
            for h, bit in zip(hs, t['bits'], strict=True):
                if bit:
                    if h['class'] == 'A':
                        a1 += h['weight']
                    else:
                        b1 += h['weight']
            g = math.gcd(a, b, a1, b1)
            return (*base, t['cost'], a//g, b//g, a1//g, b1//g)
    g = math.gcd(a, b)
    return (*base, 0, a//g, b//g)


def fingerprint_tag(d, marginal=False):
    return hash(fingerprint(d, marginal)) & ((1 << 64)-1)


class Hint:
    """Packed ring of tags, never an authentication mechanism."""
    def __init__(self):
        self.buf = bytearray(SLOTS*ENTRY.size)
        self.pos = 0

    def observe(self, tag, t):
        hit = any(old == tag and last and 0 < t+1-last <= H
                  for old, last in ENTRY.iter_unpack(self.buf))
        ENTRY.pack_into(self.buf, self.pos*ENTRY.size, tag, t+1)
        self.pos = (self.pos+1)%SLOTS
        return hit


def hidden_variant(d, rng):
    out = variant(d, rng)
    n = len(out['hypotheses'])
    out['tests'].append(dict(id='constant', cost=1, bits=[rng.randrange(2)]*n))
    out['tests'].append(dict(id='unaffordable', cost=out['budget']+1,
                             bits=[rng.randrange(2) for _ in range(n)]))
    rng.shuffle(out['tests'])
    return out


def generate(seed, kind, n):
    if kind not in ('hidden', 'collision'):
        return make_stream(seed, kind, n)
    rng = random.Random(seed)
    if kind == 'hidden':
        roots = [world(rng, 12 if j%2 == 0 else 16) for j in range(5)]
        return [hidden_variant(roots[t%5], rng) for t in range(n)]
    out = []
    # Same cheap invariants; different joint distributions.
    for _ in range(n):
        d = world(rng, 12)
        for h in d['hypotheses']:
            h['weight'] = 1
        a = b = 0
        bits = []
        for h in d['hypotheses']:
            if h['class'] == 'A':
                bits.append(int(a < 3))
                a += 1
            else:
                bits.append(int(b < 3))
                b += 1
        d['tests'][0].update(cost=1, bits=bits)
        out.append(variant(d, rng))
    return out


def evaluate(ds, arm, seed=0, cap=CAP):
    if arm in ARMS[:5]:
        machine = Machine(arm, seed=seed, cap=cap)
        vals = [machine.step(d,t) for t,d in enumerate(ds)]
        stats = machine.m
        stats.update(fingerprints=0, candidates=0, exact_hints=0, fingerprint_ms=0., canonical_ms=0.)
        return stats, vals
    if arm not in ARMS:
        raise ValueError('unknown arm')
    if cap < RESERVE:
        raise ValueError('metadata capacity')
    checked = arm.startswith('checked')
    marginal = 'marginal' in arm
    confirm = arm.endswith('confirm')
    hint, exact = Hint(), Monitor()
    store = OrderedDict()
    used = 0
    vals = []
    m = dict(cpu_ms=0., fingerprints=0, candidates=0, keys=0, exact_hints=0,
             hits=0, solves=0, checks=0, admissions=0, evictions=0, solve_edges=0,
             verify_edges=0, peak_accounted=RESERVE, fingerprint_ms=0., canonical_ms=0.)
    for t,d in enumerate(ds):
        start = time.process_time_ns()
        need = hint.observe(fingerprint_tag(d,marginal),t)
        m['fingerprint_ms'] += (time.process_time_ns()-start)/1e6
        m['fingerprints'] += 1
        m['candidates'] += int(need)
        begin = time.process_time_ns()
        k, scale = canonical(d) if need else (None,1)
        m['canonical_ms'] += (time.process_time_ns()-begin)/1e6 if need else 0.
        m['keys'] += int(need)
        f = exact.observe(k,t) if need and confirm else None
        m['exact_hints'] += int(f is not None and f[1])
        if k is not None and k in store:
            v,p = json.loads(store[k])
            store.move_to_end(k)
            v *= scale
            if replay(d,p) != v:
                raise ValueError('cached replay mismatch')
            m['hits'] += 1
        else:
            s = DP(d)
            v,p = s.run()
            m['solves'] += 1
            m['solve_edges'] += s.edges
            if need and (not confirm or f[1]):
                if v % scale:
                    raise ValueError('nonintegral canonical value')
                payload = enc((v//scale,p))
                size = len(k)+len(payload)
                if size <= cap-RESERVE:
                    if checked:
                        m['verify_edges'] += check(s)
                        m['checks'] += 1
                    while used+size > cap-RESERVE:
                        a,b = store.popitem(last=False)
                        used -= len(a)+len(b)
                        m['evictions'] += 1
                    store[k] = payload
                    used += size
                    m['admissions'] += 1
        m['cpu_ms'] += (time.process_time_ns()-start)/1e6
        m['peak_accounted'] = max(m['peak_accounted'],used+RESERVE)
        if m['peak_accounted'] > cap:
            raise ValueError('capacity')
        vals.append((v,sum(h['weight'] for h in d['hypotheses'])))
    if m['hits']+m['solves'] != len(ds) or (checked and m['checks'] != m['admissions']):
        raise ValueError('accounting')
    return m,vals


def audit_stream(ds):
    """Offline exact recent-repeat coverage; excluded from measured CPU."""
    keys = [canonical(d)[0] for d in ds]
    report = {}
    for marginal in (False,True):
        h = Hint()
        tp = fp = fn = tn = 0
        for t,d in enumerate(ds):
            repeat = keys[t] in keys[max(0,t-H):t]
            predicted = h.observe(fingerprint_tag(d,marginal),t)
            tp += int(repeat and predicted)
            fp += int(not repeat and predicted)
            fn += int(repeat and not predicted)
            tn += int(not repeat and not predicted)
        report['marginal' if marginal else 'mass'] = dict(tp=tp,fp=fp,fn=fn,tn=tn)
    return report


def campaign(n=6,length=48,repeats=2,seed=EVAL):
    rng = random.Random(seed+9999)
    inventory = set()
    report = dict(schema='rv22r.fingerprint.v1',streams=n,length=length,repeats=repeats,
                  seed=seed,cap=CAP,reserve=RESERVE,cohorts={})
    for ci,kind in enumerate(KINDS):
        rows = {a:[] for a in ARMS}
        cohort = set()
        coverage = {f:dict(tp=0,fp=0,fn=0,tn=0) for f in ('mass','marginal')}
        for j in range(n):
            ds = generate(seed+100*ci+j,kind,length)
            cohort.update(canonical(d)[0] for d in ds)
            for f,c in audit_stream(ds).items():
                for k,v in c.items():
                    coverage[f][k] += v
            samples,ref = {a:[] for a in ARMS},None
            for _ in range(repeats):
                order = list(ARMS)
                rng.shuffle(order)
                for a in order:
                    stats,vals = evaluate(ds,a,seed=991+j)
                    if ref is None:
                        ref = vals
                    if ref != vals or stats['peak_accounted'] > CAP:
                        raise ValueError('exact parity or capacity')
                    if (a not in ('cold', 'trusted_lru','trusted_mass','trusted_marginal')
                            and stats['checks'] != stats['admissions']):
                        raise ValueError('unchecked admission')
                    samples[a].append(stats)
            for a,rs in samples.items():
                rows[a].append({k:float(np.mean([r[k] for r in rs])) for k in rs[0]})
        if inventory.intersection(cohort):
            raise ValueError('cross-cohort canonical overlap')
        inventory.update(cohort)
        means = {a:{k:float(np.mean([r[k] for r in rs])) for k in rs[0]} for a,rs in rows.items()}
        idx = np.random.default_rng(891+ci).integers(n,size=(10000,n))
        paired = {}
        for a in ARMS:
            for b in sorted({'cold','always_second','jitter1','trusted_lru'}-{a}):
                delta = np.array([u['cpu_ms']-v['cpu_ms'] for u,v in zip(rows[a],rows[b],strict=True)])
                paired[a+'_vs_'+b] = dict(delta_ms=float(delta.mean()),
                    ci95=np.quantile(delta[idx].mean(axis=1),[.025,.975]).tolist())
        report['cohorts'][kind] = dict(keys=len(cohort),means=means,paired=paired,coverage=coverage)
        print(kind,{a:round(v['cpu_ms'],3) for a,v in means.items()},coverage,flush=True)
    report['arm_queries'] = n*length*repeats*len(KINDS)*len(ARMS)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--streams',type=int,default=6)
    p.add_argument('--length',type=int,default=48)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--seed',type=int,default=EVAL)
    a = p.parse_args()
    if a.out.exists() or not 1 <= a.streams <= 32 or not 24 <= a.length <= 128 or not 1 <= a.repeats <= 4:
        raise ValueError('fresh output and bounded campaign required')
    a.out.write_text(json.dumps(campaign(a.streams,a.length,a.repeats,a.seed),indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
