"""RV22-O: fixed-buffer reuse detection and exact root-cache admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from economic_memory import DP, check, enc, key, replay, stream, variant, world

CAP, H = 8192, 12
SLOTS = 64
ENTRY = struct.Struct('<QIB3x')
HEADER = struct.Struct('<H6x')
NFEATURES = 6
RESERVE = HEADER.size + SLOTS * ENTRY.size + 8 * NFEATURES
KINDS = ('burst', 'mixed', 'churn', 'phase', 'recovery', 'scan')
ARMS = ('cold', 'trusted_lru', 'trusted_matched', 'trusted_second', 'trusted_gate',
        'trusted_ldt', 'checked_lru', 'checked_matched', 'checked_second',
        'checked_gate', 'checked_ldt', 'checked_shuffled')
TRAIN_BASE, EVAL_BASE = 2026131000, 2026150000


class Sketch:
    """Direct-mapped 64-bit digest hints; never used to authorize cache hits."""
    def __init__(self, slots=SLOTS):
        if slots < 1:
            raise ValueError('empty sketch')
        self.slots = slots
        self.buf = bytearray(HEADER.size + slots * ENTRY.size)
        HEADER.pack_into(self.buf, 0, 64)  # Q8 initial repeat-rate prior .25

    def observe(self, k, t):
        tag = int.from_bytes(hashlib.blake2b(k, digest_size=8).digest(), 'little')
        offset = HEADER.size + (tag % self.slots) * ENTRY.size
        old, last, count = ENTRY.unpack_from(self.buf, offset)
        match = bool(last and old == tag and 0 < t+1-last <= H)
        gap = t+1-last if match else H+1
        count = min(count+1, 255) if match else 1
        ENTRY.pack_into(self.buf, offset, tag, t+1, count)
        rate, = HEADER.unpack_from(self.buf, 0)
        rate = (7*rate+256*int(match)+4)//8
        HEADER.pack_into(self.buf, 0, rate)
        return int(match), count-1, gap, rate/256


def features(obs):
    repeat, count, gap, rate = obs
    return (1., float(repeat), min(count, 6)/6, min(gap, H)/H,
            rate, float(repeat)*rate)


def make_stream(seed, kind, n):
    if kind in ('burst', 'mixed', 'churn', 'phase'):
        return stream(seed, kind, n)
    rng = random.Random(seed)
    projects = [world(rng, 12 if i%2 == 0 else 16) for i in range(12)]
    out = []
    for t in range(n):
        if kind == 'recovery':
            d = (world(rng, 12 if t%2 == 0 else 16) if t < n//2 and rng.random() < .8
                 else projects[(t//8)%12])
        elif kind == 'scan':
            # Repeated hot roots interleaved with nonrepeating scans; unseen family.
            d = projects[t%3] if t%5 < 2 else world(rng, 12 if t%2 == 0 else 16)
        else:
            raise ValueError('unknown cohort')
        out.append(variant(d, rng))
    return out


def fit():
    rows, targets = [], []
    inventory = set()
    for ci, kind in enumerate(KINDS[:3]):
        for j in range(16):
            ds = make_stream(TRAIN_BASE+100*ci+j, kind, 72)
            keys = [key(d)[0] for d in ds]
            inventory.update(keys)
            sketch = Sketch()
            for t, k in enumerate(keys[:-H]):
                rows.append(features(sketch.observe(k, t)))
                targets.append(keys[t+1:t+1+H].count(k))
    x, y = np.asarray(rows), np.asarray(targets)
    gram = x.T@x + np.eye(NFEATURES)
    return (np.linalg.solve(gram, x.T@y).tolist(),
            np.linalg.solve(gram, x.T@np.random.default_rng(419).permutation(y)).tolist(),
            len(y), inventory)


def want(arm, obs, weights):
    repeat, count, gap, rate = obs
    checked = arm.startswith('checked')
    if arm.endswith('second'):
        return bool(repeat)
    if arm.endswith('gate'):
        if checked:
            return bool(repeat and (count >= 2 or (gap <= 4 and rate >= .55)))
        return bool(repeat or rate >= .50)
    return sum(a*b for a,b in zip(weights, features(obs), strict=True)) >= (1.5 if checked else .25)


def evaluate(ds, arm, weights, cap=CAP):
    checked = arm.startswith('checked')
    adaptive = arm.endswith(('second', 'gate', 'ldt', 'shuffled'))
    reserve = RESERVE if adaptive or arm.endswith('matched') else 0
    if cap < reserve:
        raise ValueError('capacity below reserved metadata')
    cache_cap = cap-reserve
    sketch = Sketch() if adaptive else None
    cache = OrderedDict()
    used = 0
    values = []
    stats = dict(cpu_ms=0., hits=0, misses=0, solves=0, checks=0, solve_edges=0,
                 verify_edges=0, evictions=0, admissions=0, bypass=0, sketch_repeats=0,
                 peak_accounted_bytes=reserve, peak_retained_estimate=0,
                 first_cpu_ms=0., second_cpu_ms=0., first_hits=0, second_hits=0)
    for t, d in enumerate(ds):
        start = time.process_time_ns()
        if arm == 'cold':
            k, scale = None, 1
        else:
            k, scale = key(d)
        obs = sketch.observe(k,t) if sketch is not None else None
        if obs is not None:
            stats['sketch_repeats'] += obs[0]
        hit = k in cache
        if hit:
            payload = cache[k]
            cache.move_to_end(k)
            v, p = json.loads(payload)
            v *= scale
            if replay(d,p) != v:
                raise ValueError('cached replay mismatch')
            stats['hits'] += 1
        else:
            s = DP(d)
            v,p = s.run()
            stats['solves'] += 1
            stats['misses'] += 1
            stats['solve_edges'] += s.edges
            admit = arm != 'cold' and (not adaptive or want(arm, obs, weights))
            if admit:
                if v % scale:
                    raise ValueError('nonintegral canonical value')
                payload = enc((v//scale,p))
                size = len(k)+len(payload)
                if size <= cache_cap:
                    if checked:
                        stats['verify_edges'] += check(s)
                        stats['checks'] += 1
                    while used+size > cache_cap:
                        a,b = cache.popitem(last=False)
                        used -= len(a)+len(b)
                        stats['evictions'] += 1
                    cache[k] = payload
                    used += size
                    stats['admissions'] += 1
                else:
                    stats['bypass'] += 1
            elif arm != 'cold':
                stats['bypass'] += 1
        elapsed = (time.process_time_ns()-start)/1e6
        stats['cpu_ms'] += elapsed
        half = 'first' if t < len(ds)//2 else 'second'
        stats[half+'_cpu_ms'] += elapsed
        stats[half+'_hits'] += int(hit)
        if used+reserve > cap:
            raise ValueError('accounted payload cap')
        stats['peak_accounted_bytes'] = max(stats['peak_accounted_bytes'],used+reserve)
        # Excludes solver scratch, process allocator, shared static code/model and stack.
        retained = sys.getsizeof(cache)+sum(sys.getsizeof(a)+sys.getsizeof(b) for a,b in cache.items())
        if sketch is not None:
            retained += sys.getsizeof(sketch)+sys.getsizeof(sketch.__dict__)+sys.getsizeof(sketch.buf)
            retained += sys.getsizeof(weights)+sum(sys.getsizeof(a) for a in weights)
        stats['peak_retained_estimate'] = max(stats['peak_retained_estimate'],retained)
        values.append((v,sum(h['weight'] for h in d['hypotheses'])))
    return stats,values


def campaign(n=12, length=64, repeats=2):
    w, sham, examples, inventory = fit()
    report = dict(schema='rv22o.streaming_memory.v1',weights=w,shuffled_weights=sham,
                  train_examples=examples,train_keys=len(inventory),cap=CAP,
                  reserve=RESERVE,sketch_bytes=HEADER.size+SLOTS*ENTRY.size,
                  streams=n,length=length,timing_repeats=repeats,cohorts={})
    rng = random.Random(2026149901)
    for ci,kind in enumerate(KINDS):
        rows = {a:[] for a in ARMS}
        cohort_keys = set()
        for j in range(n):
            ds = make_stream(EVAL_BASE+100*ci+j,kind,length)
            cohort_keys.update(key(d)[0] for d in ds)
            ref = None
            samples = {a:[] for a in ARMS}
            for _ in range(repeats):
                order = list(ARMS)
                rng.shuffle(order)
                for a in order:
                    stats, values = evaluate(ds,a,sham if a.endswith('shuffled') else w)
                    if ref is None:
                        ref = values
                    if values != ref:
                        raise ValueError('cross-arm exact utility mismatch')
                    samples[a].append(stats)
            for a,rs in samples.items():
                rows[a].append({k:float(np.mean([r[k] for r in rs])) for k in rs[0]})
        if inventory.intersection(cohort_keys):
            raise ValueError('overlapping canonical split')
        inventory.update(cohort_keys)
        idx = np.random.default_rng(771+ci).integers(n,size=(10000,n))
        means = {a:{k:float(np.mean([r[k] for r in rs])) for k in rs[0]} for a,rs in rows.items()}
        paired = {}
        for a in ARMS[1:]:
            baseline = 'checked_matched' if a.startswith('checked') else 'trusted_matched'
            controls = {'cold',baseline}
            if a == 'checked_ldt':
                controls.update(('checked_gate','checked_shuffled'))
            if a == 'trusted_ldt':
                controls.add('trusted_gate')
            for b in sorted(controls-{a}):
                delta = np.array([u['cpu_ms']-v['cpu_ms'] for u,v in zip(rows[a],rows[b],strict=True)])
                paired[a+'_vs_'+b] = dict(delta_ms=float(delta.mean()),
                    ci95=np.quantile(delta[idx].mean(axis=1),[.025,.975]).tolist())
        report['cohorts'][kind] = dict(keys=len(cohort_keys),means=means,paired=paired)
        print(kind,{a:round(v['cpu_ms'],2) for a,v in means.items()},flush=True)
    report['arm_queries'] = n*length*repeats*len(KINDS)*len(ARMS)
    report['scope'] = ('Exact synthetic root reuse; fixed metadata payload and LRU; timing includes gate and check, '
                       'excludes offline fitting, evaluator and retained-size estimation; trust modes separated.')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--streams',type=int,default=12)
    p.add_argument('--length',type=int,default=64)
    p.add_argument('--repeats',type=int,default=2)
    a = p.parse_args()
    if a.out.exists() or not 1 <= a.streams <= 64 or not 24 <= a.length <= 256 or not 1 <= a.repeats <= 4:
        raise ValueError('fresh output and bounded campaign required')
    a.out.write_text(json.dumps(campaign(a.streams,a.length,a.repeats),indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
