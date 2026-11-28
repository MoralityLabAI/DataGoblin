"""RV22-U: scan-resistant checked binary cache under working-set overflow."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, OrderedDict, deque
from pathlib import Path

import numpy as np
from binary_keys import binary_key, digest
from cold_bypass import DP, check, enc, replay, variant, world

CAPS = (2048, 4096, 8192)
RESERVE, H, META = 1024, 32, 24
KINDS = ('cycle4', 'cycle12', 'cycle24', 'scan', 'phase')
ARMS = ('cold', 'eager', 'second', 'freeze', 'frequency', 'economic')
SEED = 2026600000


def stream(seed, kind, length):
    rng = random.Random(seed)
    n = int(kind[5:]) if kind.startswith('cycle') else 8
    roots = [world(rng, 16) for _ in range(n)]
    out = []
    for t in range(length):
        if kind == 'scan':
            d = roots[(t//3)%4] if t%3 == 0 else world(rng, 16)
        elif kind == 'phase':
            d = roots[t%4+4*int(t >= length//2)]
        else:
            d = roots[t%n]
        out.append(variant(d, rng))
    return out


def calibration():
    rng = random.Random(SEED-10000)
    ratios, hits = [], []
    for _ in range(32):
        d = world(rng, 16)
        start = time.process_time_ns()
        s = DP(d)
        v,p = s.run()
        miss = (time.process_time_ns()-start)/1e6
        start = time.process_time_ns()
        check(s)
        ratios.append((time.process_time_ns()-start)/1e6/max(miss, 1e-9))
        payload = enc((v,p))
        start = time.process_time_ns()
        u,tree = json.loads(payload)
        if replay(d,tree) != u:
            raise ValueError('calibration replay')
        hits.append((time.process_time_ns()-start)/1e6)
    return float(np.median(ratios)), float(np.median(hits))


def evaluate(ds, arm, cap, cost=(1.,.02)):
    if arm not in ARMS or cap < RESERVE:
        raise ValueError('unsupported arm or capacity')
    reserve = 0 if arm == 'cold' else RESERVE
    cache, ring, counts = OrderedDict(), deque(), Counter()
    used = 0
    stats = dict(cpu_ms=0., check_ms=0., solve_ms=0., hits=0, solves=0, checks=0,
                 solve_edges=0, verify_edges=0, admissions=0, rejections=0, evictions=0,
                 peak_accounted=reserve, peak_retained_estimate=0, first_hits=0, second_hits=0)
    values = []
    for t,d in enumerate(ds):
        start = time.process_time_ns()
        k,scale = binary_key(d) if arm != 'cold' else (None,1)
        h = digest(k) if k is not None else None
        prior = counts[h]
        if h is not None:
            if len(ring) == H:
                old = ring.popleft()
                counts[old] -= 1
                if not counts[old]:
                    del counts[old]
            ring.append(h)
            counts[h] += 1
        if k in cache:
            payload = cache[k][0]
            cache.move_to_end(k)
            v,p = json.loads(payload)
            v *= scale
            if replay(d,p) != v:
                raise ValueError('cached replay')
            stats['hits'] += 1
            stats['first_hits' if t < len(ds)//2 else 'second_hits'] += 1
        else:
            begin = time.process_time_ns()
            s = DP(d)
            v,p = s.run()
            miss = (time.process_time_ns()-begin)/1e6
            stats['solve_ms'] += miss
            stats['solves'] += 1
            stats['solve_edges'] += s.edges
            if k is not None and (arm == 'eager' or prior):
                if v%scale:
                    raise ValueError('normalization')
                payload = enc((v//scale,p))
                size = len(k)+len(payload)+META
                saving = max(0., miss-cost[1])
                need = max(0, used+size-(cap-reserve))
                victims = []
                admit = size <= cap-reserve
                if need and admit:
                    if arm == 'freeze':
                        admit = False
                    else:
                        entries = list(cache.items())
                        if arm == 'frequency':
                            entries.sort(key=lambda kv: counts[kv[1][1]])
                        elif arm == 'economic':
                            entries.sort(key=lambda kv: counts[kv[1][1]]*kv[1][2]/
                                         (len(kv[0])+len(kv[1][0])+META))
                        for old,(data,tag,sv) in entries:
                            victims.append((old,data,tag,sv))
                            need -= len(old)+len(data)+META
                            if need <= 0:
                                break
                if admit and arm == 'frequency' and victims:
                    admit = prior > sum(counts[tag] for _,_,tag,_ in victims)
                if admit and arm == 'economic':
                    admit = prior*saving > cost[0]*miss+sum(counts[tag]*sv for _,_,tag,sv in victims)
                if admit:
                    begin = time.process_time_ns()
                    stats['verify_edges'] += check(s)
                    stats['check_ms'] += (time.process_time_ns()-begin)/1e6
                    stats['checks'] += 1
                    for old,data,_,_ in victims:
                        del cache[old]
                        used -= len(old)+len(data)+META
                        stats['evictions'] += 1
                    cache[k] = payload,h,saving
                    used += size
                    stats['admissions'] += 1
                else:
                    stats['rejections'] += 1
        stats['cpu_ms'] += (time.process_time_ns()-start)/1e6
        stats['peak_accounted'] = max(stats['peak_accounted'],used+reserve)
        retained = (sys.getsizeof(cache)+sys.getsizeof(ring)+sys.getsizeof(counts)
                    +sum(sys.getsizeof(k)+sys.getsizeof(e)+sum(sys.getsizeof(a) for a in e)
                         for k,e in cache.items())
                    +sum(sys.getsizeof(a) for a in ring)
                    +sum(sys.getsizeof(k)+sys.getsizeof(v) for k,v in counts.items()))
        stats['peak_retained_estimate'] = max(stats['peak_retained_estimate'],retained)
        values.append((v,sum(h['weight'] for h in d['hypotheses'])))
    if (stats['peak_accounted'] > cap or stats['hits']+stats['solves'] != len(ds)
            or stats['checks'] != stats['admissions']):
        raise ValueError('accounting failure')
    return stats,values


def campaign(n=3, length=72, seed=SEED):
    cost = calibration()
    rng = random.Random(seed+9999)
    report = dict(schema='rv22u.overflow.v1',streams=n,length=length,seed=seed,cost=cost,
                  caps=CAPS,reserve=RESERVE,entry_metadata=META,cohorts={})
    seen = set()
    for ci,kind in enumerate(KINDS):
        rows = {cap:{a:[] for a in ARMS} for cap in CAPS}
        keys = set()
        for j in range(n):
            ds = stream(seed+100*ci+j,kind,length)
            inventory = {binary_key(d)[0] for d in ds}
            if seen.intersection(inventory):
                raise ValueError('cross-stream key overlap')
            seen.update(inventory)
            keys.update(inventory)
            reference = None
            order = [(cap,a) for cap in CAPS for a in ARMS]
            rng.shuffle(order)
            for cap,a in order:
                stats,vals = evaluate(ds,a,cap,cost)
                if reference is None:
                    reference = vals
                if vals != reference:
                    raise ValueError('exact utility mismatch')
                rows[cap][a].append(stats)
        result = {}
        idx = np.random.default_rng(121+ci).integers(n,size=(10000,n))
        for cap,arms in rows.items():
            means = {a:{k:float(np.mean([r[k] for r in rs])) for k in rs[0]} for a,rs in arms.items()}
            paired = {}
            for a in ARMS[1:]:
                for b in ('cold','second'):
                    if a == b:
                        continue
                    d = np.array([u['cpu_ms']-v['cpu_ms'] for u,v in zip(arms[a],arms[b],strict=True)])
                    paired[a+'_vs_'+b] = dict(delta_ms=float(d.mean()),
                        ci95=np.quantile(d[idx].mean(axis=1),[.025,.975]).tolist())
            result[str(cap)] = dict(means=means,paired=paired)
        report['cohorts'][kind] = dict(unique_keys=len(keys),caps=result)
        print(kind,{cap:{a:round(v['cpu_ms'],2) for a,v in c['means'].items()} for cap,c in result.items()},flush=True)
    report['arm_queries'] = n*length*len(KINDS)*len(CAPS)*len(ARMS)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--streams',type=int,default=3)
    p.add_argument('--length',type=int,default=72)
    p.add_argument('--seed',type=int,default=SEED)
    a = p.parse_args()
    if a.out.exists() or not 1<=a.streams<=16 or not 48<=a.length<=128:
        raise ValueError('fresh output and bounded campaign required')
    a.out.write_text(json.dumps(campaign(a.streams,a.length,a.seed),indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
