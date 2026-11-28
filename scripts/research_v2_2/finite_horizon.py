"""RV22-W: survival-weighted admission for the fixed checked-cache arena."""
from __future__ import annotations

import argparse
import json
import math
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np
from binary_keys import binary_key
from cold_bypass import DP, check, enc, replay
from reuse_distance import HEADER, Arena, History, generate, tag_of

MAX_AGE, HORIZON = 256, 96
COUNT = struct.Struct('<H')
CAPS = (4096, 8192)
ARMS = ('cold', 'second', 'third', 'freeze', 'fixed', 'fixed_age', 'fixed_reset', 'survival', 'lower', 'known_end')
COHORTS = (('short12', 'cycle12', (24, 36, 48)), ('long12', 'cycle12', (120, 168, 216)),
           ('cycle40', 'cycle40', (96, 128, 160)), ('cycle56', 'cycle56', (120, 144, 192)),
           ('phase', 'phase', (96, 144, 192)), ('mixed_scan', 'mixed_scan', (72, 108, 156)))
TRAIN, SEED = 2026720000, 2026730000
CHECK_RATIO = 1.2


def lifetime_table(seed=TRAIN):
    rng = random.Random(seed)
    lengths = rng.choices((48, 72, 96, 144, 192), weights=(1, 2, 3, 3, 1), k=256)
    return b''.join(COUNT.pack(sum(n >= t for n in lengths)) for t in range(MAX_AGE+1))


def count_at(table, age):
    return COUNT.unpack_from(table, age*COUNT.size)[0] if 0 <= age <= MAX_AGE else 0


def forecast(table, now, interval, first=None, lower=False):
    """Conditional survival sum; periodic future reuse is an assumption."""
    n = count_at(table, now)
    if not n:
        return 0.
    penalty = math.sqrt(math.log(2*(MAX_AGE+1)**2/.05)/(2*n)) if lower else 0.
    step = max(1, interval)
    first = step if first is None else max(1, first)
    return sum(max(0., count_at(table, t)/n-penalty) for t in range(now+first, MAX_AGE+1, step))


def future_hits(arm, table, now, interval, age=0, end=None):
    d = max(1, interval)
    decay = max(0., min(1., 2-age/d)) if arm in ('fixed_age', 'fixed_reset', 'survival', 'lower', 'known_end') else 1.
    if arm in ('survival', 'lower'):
        return decay*forecast(table, now, d, max(1, d-age), arm == 'lower')
    if arm == 'known_end':
        return decay*max(0, 1+(end-now-max(1,d-age))//d)
    return decay*HORIZON/d


def evaluate(ds, arm, cap, table):
    if arm not in ARMS or len(ds) > MAX_AGE:
        raise ValueError('unsupported arm or stream length')
    hist = History() if arm != 'cold' else None
    reserve = sys.getsizeof(table) if hist is not None else 0
    store = Arena(cap-hist.retained()-reserve) if hist is not None else None
    retained = hist.retained()+store.retained()+reserve if store is not None else 0
    stats = dict(cpu_ms=0., hits=0, solves=0, checks=0, admissions=0, evictions=0, resets=0,
                 reset_evictions=0, repeat_hints=0, rejections=0, solve_edges=0, verify_edges=0,
                 retained_bytes=retained, peak_record_bytes=0, first_hits=0, second_hits=0,
                 exhausted_prior=0, projected_hits=0.)
    vals = []
    economic = arm in ARMS[4:]
    for t,d in enumerate(ds):
        start = time.process_time_ns()
        now = t+1
        k,scale = binary_key(d) if store is not None else (None,1)
        rows = store.rows() if store is not None else []
        prior,gap,prev,reset = hist.observe(tag_of(k),t,True) if hist else (0,0,0,False)
        stats['repeat_hints'] += int(bool(prior))
        if arm == 'fixed_reset' and reset:
            stats['resets'] += 1
            survivors = [r for r in rows if now-r[2] <= 2*max(1,r[3])]
            stats['reset_evictions'] += len(rows)-len(survivors)
            stats['evictions'] += len(rows)-len(survivors)
            rows = survivors
            store.write(rows)
        found = next((i for i,r in enumerate(rows) if r[0] == k),None)
        if found is not None:
            key,payload,last,old_gap,saving = rows.pop(found)
            v,p = json.loads(payload)
            v *= scale
            if replay(d,p) != v:
                raise ValueError('cached replay mismatch')
            rows.append((key,payload,now,max(1,gap,prev) if gap else max(old_gap,now-last),saving))
            store.write(rows)
            stats['hits'] += 1
            stats['first_hits' if t < len(ds)//2 else 'second_hits'] += 1
        else:
            solver = DP(d)
            v,p = solver.run()
            stats['solves'] += 1
            stats['solve_edges'] += solver.edges
            if store is not None and prior >= (2 if arm == 'third' else 1):
                if v%scale:
                    raise ValueError('invalid normalization')
                row = (k,enc((v//scale,p)),now,max(1,gap,prev),float(solver.edges))
                size = Arena.size(row)
                need = max(0,sum(map(Arena.size,rows))+size-store.available())
                victims = []
                admit = size <= store.available()
                end = len(ds) if arm == 'known_end' else None

                def resident(r, now=now, end=end):
                    return r[4]*future_hits(arm,table,now,r[3],now-r[2],end)

                if need and admit:
                    if arm == 'freeze':
                        admit = False
                    else:
                        ordered = sorted(rows,key=lambda r:resident(r)/Arena.size(r)) if economic else rows
                        for r in ordered:
                            victims.append(r)
                            need -= Arena.size(r)
                            if need <= 0:
                                break
                if economic and admit:
                    expected = future_hits(arm,table,now,row[3],end=end)
                    stats['projected_hits'] += expected
                    stats['exhausted_prior'] += int(arm in ('survival','lower') and not count_at(table,now))
                    admit = expected*row[4] > CHECK_RATIO*solver.edges+sum(map(resident,victims))
                if admit:
                    stats['verify_edges'] += check(solver)
                    stats['checks'] += 1
                    stats['admissions'] += 1
                    stats['evictions'] += len(victims)
                    rows = [r for r in rows if r not in victims]
                    rows.append(row)
                    store.write(rows)
                else:
                    stats['rejections'] += 1
        stats['cpu_ms'] += (time.process_time_ns()-start)/1e6
        if store is not None:
            stats['peak_record_bytes'] = max(stats['peak_record_bytes'],HEADER.unpack_from(store.buf)[0])
            if hist.retained()+store.retained()+reserve != retained or retained > cap:
                raise ValueError('retained cap changed')
        vals.append((v,sum(h['weight'] for h in d['hypotheses'])))
    if stats['checks'] != stats['admissions'] or stats['hits']+stats['solves'] != len(ds):
        raise ValueError('accounting failure')
    stats['weighted_edges'] = 5*stats['solve_edges']+6*stats['verify_edges']
    return stats,vals


def campaign(seed=SEED, selected=None):
    table = lifetime_table()
    rng = random.Random(seed+9999)
    report = dict(schema='rv22w.finite_horizon.v1',seed=seed,train_seed=TRAIN,train_lifetimes=256,
                  lifetime_counts={str(t):count_at(table,t) for t in (1,48,49,72,73,96,97,144,145,192,193)},
                  table_bytes=sys.getsizeof(table),caps=CAPS,check_ratio=CHECK_RATIO,cohorts={})
    seen = set()
    queries = 0
    for ci,(kind,profile,lengths) in enumerate(COHORTS):
        if selected is not None and kind != selected:
            continue
        rows = {cap:{a:[] for a in ARMS} for cap in CAPS}
        for j,length in enumerate(lengths):
            ds = generate(seed+100*ci+j,profile,length)
            keys = {binary_key(d)[0] for d in ds}
            if seen.intersection(keys):
                raise ValueError('cross-stream key overlap')
            seen.update(keys)
            order = [(cap,a) for cap in CAPS for a in ARMS]
            rng.shuffle(order)
            ref = None
            for cap,a in order:
                stats,vals = evaluate(ds,a,cap,table)
                if ref is None:
                    ref = vals
                if ref != vals:
                    raise ValueError('exact utility mismatch')
                rows[cap][a].append(stats)
                queries += len(ds)
        idx = np.random.default_rng(1201+ci).integers(len(lengths),size=(10000,len(lengths)))
        result = {}
        for cap,arms in rows.items():
            means = {a:{k:float(np.mean([r[k] for r in rs])) for k in rs[0]} for a,rs in arms.items()}
            paired = {}
            for a in ARMS[1:]:
                for b in ('cold','second','third','freeze','fixed_age'):
                    if a == b:
                        continue
                    delta = np.array([u['cpu_ms']-v['cpu_ms'] for u,v in zip(arms[a],arms[b],strict=True)])
                    paired[a+'_vs_'+b] = dict(delta_ms=float(delta.mean()),
                        ci95=np.quantile(delta[idx].mean(axis=1),[.025,.975]).tolist())
            result[str(cap)] = dict(means=means,paired=paired,streams=arms)
        report['cohorts'][kind] = dict(lengths=lengths,caps=result)
        print(kind,{c:{a:round(v['cpu_ms'],2) for a,v in x['means'].items()} for c,x in result.items()},flush=True)
    report['arm_queries'] = queries
    report['unique_keys'] = len(seen)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--cohort',choices=[c[0] for c in COHORTS])
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('fresh output required')
    a.out.write_text(json.dumps(campaign(a.seed,a.cohort),indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
