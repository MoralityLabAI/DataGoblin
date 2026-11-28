"""RV22-S: marginal collision witnesses, compact exact keys, lazy refinement."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import OrderedDict, defaultdict
from itertools import combinations, product
from pathlib import Path

import numpy as np
from cold_bypass import CAP, DP, canonical, check, enc, replay, variant
from fingerprint import Hint, fingerprint_tag, generate

RESERVE = 1024
KINDS = ('burst', 'mixed', 'churn', 'alias', 'hidden', 'collision', 'pair_alias')
ARMS = ('cold', 'json_second', 'packed_second', 'packed_trusted', 'marginal_eager',
        'packed_marginal', 'ambiguous_guard', 'progressive_pair')
EVAL = 2026420000


def legal_tests(d):
    return sorted((t for t in d['tests'] if t['cost'] <= d['budget']
                   and 0 in t['bits'] and 1 in t['bits']), key=lambda t: t['id'])


def packed_key(d):
    """Injective re-encoding of the normalized legal joint distribution."""
    ts = legal_tests(d)
    hs = d['hypotheses']
    codes = [int(h['class'] == 'B') for h in hs]
    for j, t in enumerate(ts, 1):
        for i, bit in enumerate(t['bits']):
            codes[i] |= bit << j
    counts = defaultdict(int)
    for code, h in zip(codes, hs, strict=True):
        counts[code] += h['weight']
    scale = math.gcd(*counts.values())
    return enc((d['budget'], d['correct_reward'], d['wrong_reward'],
                [(t['id'], t['cost']) for t in ts],
                [(k, v//scale) for k, v in sorted(counts.items())])), scale


def moments(d, order):
    """All class-conditioned positive-bit intersection masses through order."""
    ts = legal_tests(d)
    hs = d['hypotheses']
    masks = [sum(bit << i for i, bit in enumerate(t['bits'])) for t in ts]
    full = (1 << len(hs))-1
    weights = [h['weight'] for h in hs]
    classes = [h['class'] == 'B' for h in hs]
    cache = {}
    rows = []
    for size in range(min(order, len(ts))+1):
        for js in combinations(range(len(ts)), size):
            mask = full
            for j in js:
                mask &= masks[j]
            if mask not in cache:
                counts = [0, 0]
                rest = mask
                while rest:
                    bit = rest & -rest
                    i = bit.bit_length()-1
                    counts[classes[i]] += weights[i]
                    rest ^= bit
                cache[mask] = counts
            rows.extend(cache[mask])
    scale = math.gcd(*rows)
    return (d['budget'], d['correct_reward'], d['wrong_reward'],
            tuple((t['id'], t['cost']) for t in ts), tuple(v//scale for v in rows))


def tag(k):
    return int.from_bytes(hashlib.blake2b(k, digest_size=8).digest(), 'little')


def parity_world(m, coeff, base):
    """A/B masses base +/- signed Walsh characters."""
    bits = list(product((0, 1), repeat=m))
    hs, cols = [], [[] for _ in range(m)]
    for x in bits:
        s = sum(a*(-1 if sum(x[j] for j in js)%2 else 1) for js, a in coeff)
        for c, weight in (('A', base+s), ('B', base-s)):
            if weight:
                hs.append({'class': c, 'weight': weight})
                for j in range(m):
                    cols[j].append(x[j])
    return dict(hypotheses=hs, tests=[dict(id=f't{j}', cost=1, bits=b) for j,b in enumerate(cols)],
                budget=m, correct_reward=12, wrong_reward=-16)


def witness(m):
    return parity_world(m, [], 1), parity_world(m, [(tuple(range(m)), 1)], 1)


def stream(seed, kind, length):
    if kind != 'pair_alias':
        return generate(seed, kind, length)
    rng = random.Random(seed)
    subsets = [js for j in (3,4) for js in combinations(range(4),j)]
    return [variant(parity_world(4, [(js,rng.randrange(1,10000)) for js in subsets], 100000),rng)
            for _ in range(length)]


def evaluate(ds, arm, cap=CAP):
    if arm not in ARMS or cap < RESERVE:
        raise ValueError('unsupported arm or capacity')
    reserve = 0 if arm == 'cold' else RESERVE
    cache, buckets = OrderedDict(), OrderedDict()
    coarse, pair, exact = Hint(), Hint(), Hint()
    used = 0
    stats = dict(cpu_ms=0., keys=0, pair_probes=0, ambiguous=0, coarse_candidates=0,
                 hits=0, solves=0, checks=0, admissions=0, evictions=0,
                 solve_edges=0, verify_edges=0, peak_accounted=reserve)
    values = []
    for t,d in enumerate(ds):
        start = time.process_time_ns()
        need = arm != 'cold'
        h, ambiguous = None, False
        if arm in ARMS[4:]:
            h = fingerprint_tag(d, True)
            need = coarse.observe(h,t)
            stats['coarse_candidates'] += int(need)
            if need and arm in ('ambiguous_guard','progressive_pair'):
                _, ambiguous = buckets.get(h, (None, False))
                if ambiguous:
                    stats['ambiguous'] += 1
                    if arm == 'progressive_pair':
                        stats['pair_probes'] += 1
                        need = pair.observe(hash(moments(d,2)) & ((1<<64)-1),t)
        k, scale = None, 1
        repeat = False
        if need:
            k, scale = canonical(d) if arm in ('json_second','marginal_eager') else packed_key(d)
            stats['keys'] += 1
            if arm in ('json_second', 'packed_second', 'ambiguous_guard', 'progressive_pair'):
                digest = tag(k)
                repeat = exact.observe(digest,t)
            if h is not None and arm in ('ambiguous_guard','progressive_pair'):
                prev, flagged = buckets.pop(h,(None,False))
                ambiguous = flagged or (prev is not None and prev != digest)
                buckets[h] = digest, ambiguous
                if len(buckets) > 16:
                    buckets.popitem(last=False)
        if need and k in cache:
            v,p = json.loads(cache[k])
            v *= scale
            cache.move_to_end(k)
            if replay(d,p) != v:
                raise ValueError('cached replay mismatch')
            stats['hits'] += 1
        else:
            s = DP(d)
            v,p = s.run()
            stats['solves'] += 1
            stats['solve_edges'] += s.edges
            admit = need and (arm not in ('json_second','packed_second') or repeat)
            if arm in ('ambiguous_guard','progressive_pair') and ambiguous:
                admit = admit and repeat
            if admit:
                if v%scale:
                    raise ValueError('invalid scale')
                data = enc((v//scale,p))
                size = len(k)+len(data)
                if size <= cap-reserve:
                    if arm != 'packed_trusted':
                        stats['verify_edges'] += check(s)
                        stats['checks'] += 1
                    while used+size > cap-reserve:
                        a,b = cache.popitem(last=False)
                        used -= len(a)+len(b)
                        stats['evictions'] += 1
                    cache[k] = data
                    used += size
                    stats['admissions'] += 1
        stats['cpu_ms'] += (time.process_time_ns()-start)/1e6
        stats['peak_accounted'] = max(stats['peak_accounted'],used+reserve)
        values.append((v,sum(h['weight'] for h in d['hypotheses'])))
    if stats['peak_accounted'] > cap or stats['solves']+stats['hits'] != len(ds):
        raise ValueError('capacity or work accounting')
    if arm != 'packed_trusted' and stats['checks'] != stats['admissions']:
        raise ValueError('unchecked admission')
    return stats,values


def campaign(n=6,length=48,repeats=2,seed=EVAL):
    rng = random.Random(seed+9999)
    report = dict(schema='rv22s.refinement.v1',streams=n,length=length,repeats=repeats,
                  seed=seed,reserve=RESERVE,cap=CAP,cohorts={})
    inventory = set()
    old_to_new, new_to_old = {},{}
    for ci,kind in enumerate(KINDS):
        rows, cohort = {a:[] for a in ARMS},set()
        for j in range(n):
            ds = stream(seed+100*ci+j,kind,length)
            for d in ds:
                old,s = canonical(d)
                new,q = packed_key(d)
                if s != q or old_to_new.setdefault(old,new) != new or new_to_old.setdefault(new,old) != old:
                    raise ValueError('exact key parity')
                cohort.add(old)
            samples,ref = {a:[] for a in ARMS},None
            for _ in range(repeats):
                order = list(ARMS)
                rng.shuffle(order)
                for a in order:
                    stats,vals = evaluate(ds,a)
                    if ref is None:
                        ref = vals
                    if vals != ref:
                        raise ValueError('exact utility mismatch')
                    samples[a].append(stats)
            for a,rs in samples.items():
                rows[a].append({k:float(np.mean([r[k] for r in rs])) for k in rs[0]})
        if inventory.intersection(cohort):
            raise ValueError('cross-cohort overlap')
        inventory.update(cohort)
        means = {a:{k:float(np.mean([r[k] for r in rs])) for k in rs[0]} for a,rs in rows.items()}
        idx = np.random.default_rng(991+ci).integers(n,size=(10000,n))
        paired = {}
        for a in ARMS:
            for b in sorted({'cold','json_second','packed_second','packed_marginal','ambiguous_guard'}-{a}):
                delta = np.asarray([u['cpu_ms']-v['cpu_ms'] for u,v in zip(rows[a],rows[b],strict=True)])
                paired[a+'_vs_'+b] = dict(delta_ms=float(delta.mean()),
                    ci95=np.quantile(delta[idx].mean(axis=1),[.025,.975]).tolist())
        report['cohorts'][kind] = dict(keys=len(cohort),means=means,paired=paired)
        print(kind,{a:round(v['cpu_ms'],2) for a,v in means.items()},flush=True)
    report['arm_queries'] = n*length*repeats*len(KINDS)*len(ARMS)
    report['unique_exact_keys'] = len(old_to_new)
    report['key_bytes'] = {name: sum(map(len, keys)) for name, keys in
                           (('json', old_to_new), ('packed', new_to_old))}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--streams',type=int,default=6)
    p.add_argument('--length',type=int,default=48)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--seed',type=int,default=EVAL)
    a = p.parse_args()
    if a.out.exists() or not 1<=a.streams<=32 or not 24<=a.length<=128 or not 1<=a.repeats<=4:
        raise ValueError('fresh output and bounded campaign required')
    a.out.write_text(json.dumps(campaign(a.streams,a.length,a.repeats,a.seed),indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
