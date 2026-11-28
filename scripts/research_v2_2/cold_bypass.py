"""RV22-P: sampled cold/cache switching in finite deterministic research worlds."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import struct
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

CAP, RESERVE, H = 8192, 256, 12
KINDS = ('burst', 'mixed', 'churn', 'phase', 'recovery', 'scan', 'delayed')
ARMS = ('cold', 'trusted_lru', 'trusted_matched', 'trusted_sample', 'trusted_ldt',
        'checked_lru', 'checked_matched', 'checked_sample', 'checked_ldt')
TRAIN, CAL, EVAL = 2026191000, 2026192000, 2026193000
ENTRY = struct.Struct('<QI')


def enc(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def canonical(d):
    ts = sorted(d['tests'], key=lambda t: t['id'])
    ts = [t for t in ts if t['cost'] <= d['budget'] and len(set(t['bits'])) > 1]
    rows = defaultdict(int)
    for i, h in enumerate(d['hypotheses']):
        rows[h['class'], tuple(t['bits'][i] for t in ts)] += h['weight']
    scale = math.gcd(*rows.values())
    return enc((d['budget'], d['correct_reward'], d['wrong_reward'],
                [(t['id'], t['cost']) for t in ts],
                [(a, b, w//scale) for (a, b), w in sorted(rows.items())])), scale


def replay(d, tree):
    ts = {t['id']: t for t in d['tests']}
    total = 0
    for i, h in enumerate(d['hypotheses']):
        node, cost, used = tree, 0, set()
        while len(node) == 3:
            a = node[0]
            if a in used:
                raise ValueError('repeated test')
            used.add(a)
            cost += ts[a]['cost']
            if cost > d['budget']:
                raise ValueError('budget')
            node = node[1+ts[a]['bits'][i]]
        if len(node) != 1 or node[0] not in ('A', 'B', 'defer'):
            raise ValueError('terminal')
        r = 0 if node[0] == 'defer' else d['correct_reward'] if node[0] == h['class'] else d['wrong_reward']
        total += h['weight']*(r-cost)
    return total


class DP:
    def __init__(self, d):
        self.d, self.hs = d, d['hypotheses']
        self.ts = sorted(d['tests'], key=lambda t: t['id'])
        self.ms = [sum(b << i for i, b in enumerate(t['bits'])) for t in self.ts]
        self.full = (1 << len(self.hs))-1
        self.memo, self.counts, self.edges = {}, {0: (0, 0)}, 0

    def count(self, m):
        if m not in self.counts:
            bit = m & -m
            h = self.hs[bit.bit_length()-1]
            w, a = self.count(m^bit)
            self.counts[m] = w+h['weight'], a+h['weight']*int(h['class'] == 'A')
        return self.counts[m]

    def solve(self, m, b):
        if (m, b) not in self.memo:
            w, a = self.count(m)
            c, r = self.d['correct_reward'], self.d['wrong_reward']
            v, p = max((a*c+(w-a)*r, 'A'), ((w-a)*c+a*r, 'B'), (0, 'defer'))
            if v < c*w:
                for t, mask in zip(self.ts, self.ms, strict=True):
                    o, cost = m & mask, t['cost']
                    if not o or o == m or cost > b:
                        continue
                    self.edges += 1
                    q = self.solve(m^o, b-cost)[0]+self.solve(o, b-cost)[0]-cost*w
                    if q > v:
                        v, p = q, t['id']
            self.memo[m, b] = v, p
        return self.memo[m, b]

    def run(self):
        value, _ = self.solve(self.full, self.d['budget'])

        def tree(m, b):
            _, a = self.memo[m, b]
            if a in ('A', 'B', 'defer'):
                return (a,)
            j = next(j for j, t in enumerate(self.ts) if t['id'] == a)
            o = m & self.ms[j]
            b -= self.ts[j]['cost']
            return a, tree(m^o, b), tree(o, b)

        p = tree(self.full, self.d['budget'])
        if replay(self.d, p) != value:
            raise ValueError('cold replay mismatch')
        return value, p


def check(s):
    d, hs = s.d, s.hs
    ts = sorted(d['tests'], key=lambda t: t['id'])
    masks = [sum(bit << i for i, bit in enumerate(t['bits'])) for t in ts]
    done, counts, edges = {}, {}, 0
    c, r = d['correct_reward'], d['wrong_reward']
    for (m, b), rec in sorted(s.memo.items(), key=lambda kv: kv[0][1]):
        if m not in counts:
            rows = [h for i, h in enumerate(hs) if m >> i & 1]
            counts[m] = sum(h['weight'] for h in rows), sum(h['weight'] for h in rows if h['class'] == 'A')
        w, a = counts[m]
        v, p = max((a*c+(w-a)*r, 'A'), ((w-a)*c+a*r, 'B'), (0, 'defer'))
        if v < c*w:
            for t, mask in zip(ts, masks, strict=True):
                o, cost = m & mask, t['cost']
                if not o or o == m or cost > b:
                    continue
                edges += 1
                q = done[o, b-cost][0]+done[m^o, b-cost][0]-cost*w
                if q > v:
                    v, p = q, t['id']
        if rec != (v, p):
            raise ValueError('invalid Bellman record')
        done[m, b] = rec
    if (s.full, d['budget']) not in done:
        raise ValueError('missing checked root')
    return edges


def world(rng, n):
    labels = list('A'*(n//2)+'B'*(n-n//2))
    rng.shuffle(labels)
    cols = []
    while len(cols) < (6 if n == 12 else 8):
        bits = [rng.randrange(2) for _ in labels]
        if len(set(bits)) == 2 and bits not in cols and [1-b for b in bits] not in cols:
            cols.append(bits)
    return dict(hypotheses=[{'class': a, 'weight': rng.randrange(1, 6)} for a in labels],
                tests=[dict(id=f't{j}', cost=rng.randrange(1, 4), bits=b) for j,b in enumerate(cols)],
                budget=7 if n == 12 else 9, correct_reward=12, wrong_reward=-16)


def variant(d, rng):
    ids = list(range(len(d['hypotheses'])))
    rng.shuffle(ids)
    out = copy.deepcopy(d)
    split = rng.random() < .5
    if split:
        ids.insert(0, ids[0])
    out['hypotheses'] = [copy.deepcopy(d['hypotheses'][i]) for i in ids]
    for h in out['hypotheses']:
        h['weight'] *= 3
    if split:
        out['hypotheses'][0]['weight'] = 1
        out['hypotheses'][1]['weight'] -= 1
    for t in out['tests']:
        t['bits'] = [t['bits'][i] for i in ids]
    rng.shuffle(out['tests'])
    return out


def stream(seed, kind, n):
    rng = random.Random(seed)
    projects = [world(rng, 12 if i%2 == 0 else 16) for i in range(12)]
    out = []
    for t in range(n):
        mode = ('burst' if t < n//2 else 'churn') if kind == 'phase' else kind
        if kind == 'recovery':
            mode = 'churn' if t < n//2 else 'burst'
        i = (t//8)%12 if mode == 'burst' else t%12
        if mode == 'mixed':
            i = rng.choices(range(12), weights=[8,6,4,3,2,2,1,1,1,1,1,1])[0]
        if mode == 'scan':
            d = projects[t%3] if t%5 < 2 else world(rng, 12 if t%2 == 0 else 16)
        elif mode == 'delayed':
            d = projects[t%9] if t%7 != 6 else world(rng, 12 if t%2 == 0 else 16)
        elif mode == 'churn' and rng.random() < .8:
            d = world(rng, 12 if t%2 == 0 else 16)
        else:
            d = projects[i]
        out.append(variant(d, rng))
    return out


class Monitor:
    """Six-feature controller; 16 packed tag/time observations, no full-key history."""
    def __init__(self, period=8, idle=10):
        self.period, self.idle = period, idle
        self.buf = bytearray(16*ENTRY.size)
        self.pos = 0
        self.active = False
        self.last_repeat = -100
        self.rate = 0.

    def probe(self, t):
        return self.active or (t % self.period - (t//self.period) % (self.period-1)) % self.period < 2

    def observe(self, k, t):
        tag = int.from_bytes(hashlib.blake2b(k, digest_size=8).digest(), 'little')
        prior = [p for old,p in ENTRY.iter_unpack(self.buf) if p and old == tag and t+1-p <= H]
        repeat = bool(prior)
        gap = t+1-max(prior) if prior else H+1
        ENTRY.pack_into(self.buf, self.pos*ENTRY.size, tag,t+1)
        self.pos = (self.pos+1)%16
        self.rate = .875*self.rate+.125*repeat
        if repeat:
            self.last_repeat = t
        return (1., float(repeat), min(len(prior),4)/4, min(gap,H)/H, self.rate, float(repeat)*self.rate)

    def advance(self, t, feat, threshold, weights=None):
        if self.active:
            if t-self.last_repeat >= self.idle:
                self.active = False
        elif feat is not None and (feat[1] if weights is None else sum(
                a*b for a,b in zip(weights,feat,strict=True))) >= threshold:
            self.active = True
            self.last_repeat = t


def fit():
    rows, targets, keys = [], [], set()
    for i,kind in enumerate(KINDS[:3]):
        for j in range(12):
            ks = [canonical(d)[0] for d in stream(TRAIN+100*i+j,kind,72)]
            keys.update(ks)
            mon = Monitor()
            for t,k in enumerate(ks[:-H]):
                if mon.probe(t):
                    f = mon.observe(k,t)
                    recent = set(ks[max(0,t-H):t+1])
                    future = ks[t+1:t+1+H]
                    rows.append(f)
                    targets.append(sum(q in recent or q in future[:i] for i,q in enumerate(future))/H)
                    mon.advance(t,f,1.)
    x,y = np.asarray(rows),np.asarray(targets)
    return np.linalg.solve(x.T@x+np.eye(6),x.T@y).tolist(),len(y),keys


def evaluate(ds, arm, weights, params=(8,10,.35), cap=CAP):
    switch = arm.endswith(('sample','ldt'))
    checked = arm.startswith('checked')
    reserve = RESERVE if switch or arm.endswith('matched') else 0
    if cap < reserve:
        raise ValueError('capacity below metadata reserve')
    mon = Monitor(params[0],params[1]) if switch else None
    cache = OrderedDict()
    used = 0
    m = dict(cpu_ms=0., keys=0, bypass=0, probes=0, hits=0, solves=0, checks=0, solve_edges=0,
             verify_edges=0, admissions=0, evictions=0, active=0, activations=0, deactivations=0,
             peak_accounted=reserve, first_ms=0., second_ms=0.)
    values = []
    for t,d in enumerate(ds):
        start = time.process_time_ns()
        before = mon.active if mon else False
        need_key = arm != 'cold' and (not switch or mon.probe(t))
        k, scale = canonical(d) if need_key else (None,1)
        m['keys'] += int(need_key)
        feat = mon.observe(k,t) if mon and need_key else None
        if mon:
            m['probes'] += int(need_key and not before)
            mon.advance(t,feat,params[2] if arm.endswith('ldt') else 1.,weights if arm.endswith('ldt') else None)
            m['activations'] += int(mon.active and not before)
            m['deactivations'] += int(before and not mon.active)
            m['active'] += int(mon.active)
            m['bypass'] += int(not need_key)
        if k is not None and k in cache:
            v,p = json.loads(cache[k])
            v *= scale
            if replay(d,p) != v:
                raise ValueError('cache replay mismatch')
            cache.move_to_end(k)
            m['hits'] += 1
        else:
            s = DP(d)
            v,p = s.run()
            m['solves'] += 1
            m['solve_edges'] += s.edges
            admit = need_key and (not switch or mon.active)
            if admit and checked and switch:
                admit = bool(feat[1])
            if admit:
                if v % scale:
                    raise ValueError('invalid normalization')
                payload = enc((v//scale,p))
                size = len(k)+len(payload)
                if size <= cap-reserve:
                    if checked:
                        m['verify_edges'] += check(s)
                        m['checks'] += 1
                    while used+size > cap-reserve:
                        a,b = cache.popitem(last=False)
                        used -= len(a)+len(b)
                        m['evictions'] += 1
                    cache[k] = payload
                    used += size
                    m['admissions'] += 1
        elapsed = (time.process_time_ns()-start)/1e6
        m['cpu_ms'] += elapsed
        m['first_ms' if t < len(ds)//2 else 'second_ms'] += elapsed
        m['peak_accounted'] = max(m['peak_accounted'],used+reserve)
        values.append((v,sum(h['weight'] for h in d['hypotheses'])))
    return m,values


def campaign(n=8, length=48, repeats=2):
    w, examples, inv = fit()
    train_keys = len(inv)
    cal_rows = [(i,j,stream(CAL+100*i+j,kind,48)) for i,kind in enumerate(KINDS[:3]) for j in range(2)]
    cal_keys = {canonical(d)[0] for _,_,ds in cal_rows for d in ds}
    if inv.intersection(cal_keys):
        raise ValueError('train/cal overlap')
    inv.update(cal_keys)
    # Freeze learned threshold and probe cadence on development only.
    cal = []
    for params in ((4,8,.25),(8,10,.25),(8,10,.40),(12,12,.40)):
        times = sum(evaluate(ds,a,w,params)[0]['cpu_ms'] for _,_,ds in cal_rows
                    for a in ('trusted_ldt','checked_ldt'))
        cal.append({'params':list(params),'cpu_ms':times})
    params = tuple(min(cal,key=lambda r:r['cpu_ms'])['params'])
    report = dict(schema='rv22p.cold_bypass.v1',weights=w,train_examples=examples,train_keys=train_keys,
                  calibration_keys=len(cal_keys),calibration=cal,chosen=params,cap=CAP,reserve=RESERVE,
                  streams=n,length=length,repeats=repeats,cohorts={})
    rng = random.Random(2026194999)
    for ci,kind in enumerate(KINDS):
        rows, cohort = {a:[] for a in ARMS},set()
        for j in range(n):
            ds = stream(EVAL+100*ci+j,kind,length)
            cohort.update(canonical(d)[0] for d in ds)
            samples,ref = {a:[] for a in ARMS},None
            for _ in range(repeats):
                order = list(ARMS)
                rng.shuffle(order)
                for a in order:
                    m,values = evaluate(ds,a,w,params if a.endswith('ldt') else (8,10,.35))
                    if ref is None:
                        ref = values
                    if ref != values:
                        raise ValueError('cross-arm exact mismatch')
                    if m['peak_accounted'] > CAP or m['hits']+m['solves'] != len(ds):
                        raise ValueError('accounting failure')
                    if a.startswith('checked') and m['checks'] != m['admissions']:
                        raise ValueError('unchecked admission')
                    samples[a].append(m)
            for a,rs in samples.items():
                rows[a].append({k:float(np.mean([r[k] for r in rs])) for k in rs[0]})
        if inv.intersection(cohort):
            raise ValueError('cross-split canonical key overlap')
        inv.update(cohort)
        means = {a:{k:float(np.mean([r[k] for r in rs])) for k in rs[0]} for a,rs in rows.items()}
        idx = np.random.default_rng(811+ci).integers(n,size=(10000,n))
        paired = {}
        for a in ARMS[1:]:
            bset = {'cold','checked_matched' if a.startswith('checked') else 'trusted_matched'}
            if a.endswith('ldt'):
                bset.add(a.replace('ldt','sample'))
            for b in sorted(bset-{a}):
                d = np.asarray([u['cpu_ms']-v['cpu_ms'] for u,v in zip(rows[a],rows[b],strict=True)])
                paired[a+'_vs_'+b] = dict(delta_ms=float(d.mean()),
                                          ci95=np.quantile(d[idx].mean(axis=1),[.025,.975]).tolist())
        report['cohorts'][kind] = dict(unique_keys=len(cohort),means=means,paired=paired)
        print(kind,{a:round(v['cpu_ms'],2) for a,v in means.items()},flush=True)
    report['arm_queries'] = n*length*repeats*len(KINDS)*len(ARMS)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--streams',type=int,default=8)
    p.add_argument('--length',type=int,default=48)
    p.add_argument('--repeats',type=int,default=2)
    a = p.parse_args()
    if a.out.exists() or not 1 <= a.streams <= 64 or not 24 <= a.length <= 256 or not 1 <= a.repeats <= 4:
        raise ValueError('fresh output and bounded campaign required')
    a.out.write_text(json.dumps(campaign(a.streams,a.length,a.repeats),indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
