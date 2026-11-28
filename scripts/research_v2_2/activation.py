"""RV22-Q: isolate activation from admission and learn measured CPU savings."""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from cold_bypass import CAP, DP, Monitor, canonical, check, enc, replay, stream, variant, world

RESERVE, H = 384, 12
KINDS = ('burst', 'mixed', 'churn', 'phase', 'recovery', 'scan', 'delayed', 'alias')
ARMS = ('cold', 'trusted_lru', 'checked_lru', 'always_second', 'sample_second',
        'switch1', 'switch2', 'cost', 'sham', 'jitter1')
TRAIN, CAL, EVAL = 2026210000, 2026220000, 2026240000


def probe(t, seed=None):
    block, offset = divmod(t, 8)
    start = block % 7 if seed is None else ((seed ^ (block*2654435761)) * 2246822519 >> 13) % 8
    return (offset-start) % 8 < 2


def make_stream(seed, kind, n):
    if kind != 'alias':
        return stream(seed, kind, n)
    rng = random.Random(seed)
    hot = world(rng, 16)
    return [variant(world(rng, 12 if t%2 == 0 else 16) if probe(t) else hot, rng) for t in range(n)]


def features(f, d):
    return (*f, len(d['hypotheses'])/16, len(d['tests'])/8, d['budget']/9)


class Machine:
    def __init__(self, arm, weights=(), threshold=0., seed=0, cap=CAP):
        self.arm, self.weights, self.threshold, self.seed = arm, weights, threshold, seed
        self.reserve = 0 if arm == 'cold' else RESERVE
        if cap < self.reserve:
            raise ValueError('capacity below metadata reserve')
        self.cap = cap-self.reserve
        self.mon = Monitor()
        self.cache = OrderedDict()
        self.used = 0
        self.m = dict(cpu_ms=0., keys=0, hits=0, solves=0, checks=0, admissions=0,
                      solve_edges=0, verify_edges=0, activations=0, deactivations=0,
                      evictions=0, active=0, peak_accounted=self.reserve)

    def scheduled(self, t):
        return probe(t, self.seed if self.arm == 'jitter1' else None)

    def step(self, d, t, forced=None):
        start = time.process_time_ns()
        a, m, mon = self.arm, self.m, self.mon
        static = a in ('trusted_lru', 'checked_lru', 'always_second')
        switching = a in ('switch1', 'switch2', 'cost', 'sham', 'jitter1')
        need = a != 'cold' and (static or mon.active or self.scheduled(t))
        if forced is not None:
            need = forced == 'full' or self.scheduled(t)
        k, scale = canonical(d) if need else (None, 1)
        m['keys'] += int(need)
        f = mon.observe(k, t) if need and a not in ('trusted_lru', 'checked_lru') else None
        if switching and forced is None:
            before = mon.active
            if before:
                if t-mon.last_repeat >= 10:
                    mon.active = False
            elif f is not None and f[1]:
                if a == 'switch2':
                    mon.active = f[2] >= .5
                elif a in ('cost', 'sham'):
                    score = sum(x*y for x, y in zip(self.weights, features(f, d), strict=True))
                    mon.active = score > self.threshold
                else:
                    mon.active = True
            m['activations'] += int(mon.active and not before)
            m['deactivations'] += int(before and not mon.active)
        m['active'] += int(mon.active)
        if need and k in self.cache:
            v, p = json.loads(self.cache[k])
            self.cache.move_to_end(k)
            v *= scale
            if replay(d, p) != v:
                raise ValueError('cached replay mismatch')
            m['hits'] += 1
        else:
            s = DP(d)
            v, p = s.run()
            m['solves'] += 1
            m['solve_edges'] += s.edges
            # Delayed-admission arms share the same observed-repeat requirement.
            admit = need and (a in ('trusted_lru', 'checked_lru') or (f is not None and f[1]))
            if admit:
                if v % scale:
                    raise ValueError('nonintegral canonical value')
                data = enc((v//scale, p))
                size = len(k)+len(data)
                if size <= self.cap:
                    if a != 'trusted_lru':
                        m['verify_edges'] += check(s)
                        m['checks'] += 1
                    while self.used+size > self.cap:
                        old, val = self.cache.popitem(last=False)
                        self.used -= len(old)+len(val)
                        m['evictions'] += 1
                    self.cache[k] = data
                    self.used += size
                    m['admissions'] += 1
        m['cpu_ms'] += (time.process_time_ns()-start)/1e6
        if self.used > self.cap:
            raise ValueError('payload overflow')
        m['peak_accounted'] = max(m['peak_accounted'], self.reserve+self.used)
        return v, sum(h['weight'] for h in d['hypotheses'])


def evaluate(ds, arm, w=(), tau=0., seed=0, cap=CAP):
    machine = Machine(arm, w, tau, seed, cap)
    vals = [machine.step(d,t) for t,d in enumerate(ds)]
    m = machine.m
    if m['solves']+m['hits'] != len(ds) or (arm != 'trusted_lru' and m['checks'] != m['admissions']):
        raise ValueError('accounting mismatch')
    return m, vals


def fit():
    x, y, keys = [], [], set()
    rng = random.Random(422)
    for i,kind in enumerate(KINDS[:3]):
        for j in range(8):
            ds = make_stream(TRAIN+100*i+j,kind,64)
            keys.update(canonical(d)[0] for d in ds)
            machine = Machine('sample_second')
            for t,d in enumerate(ds):
                if t+H <= len(ds) and machine.scheduled(t):
                    f = copy.deepcopy(machine.mon).observe(canonical(d)[0],t)
                    if f[1]:
                        times = {'full': [], 'sample': []}
                        for _ in range(2):
                            order = ['full','sample']
                            rng.shuffle(order)
                            for mode in order:
                                clone = copy.deepcopy(machine)
                                clone.m['cpu_ms'] = 0.
                                for k in range(t,t+H):
                                    clone.step(ds[k],k,forced=mode)
                                times[mode].append(clone.m['cpu_ms'])
                        x.append(features(f,d))
                        y.append(np.mean(times['sample'])-np.mean(times['full']))
                machine.step(d,t)
    x,y = np.asarray(x),np.asarray(y)
    gram = x.T@x + .5*np.eye(x.shape[1])
    return (np.linalg.solve(gram,x.T@y).tolist(),
            np.linalg.solve(gram,x.T@np.random.default_rng(423).permutation(y)).tolist(),
            dict(examples=len(y),positive=int((y>0).sum()),mean_ms=float(y.mean())),keys)


def campaign(n=4, length=48, repeats=2):
    w,sham,training,inv = fit()
    training['keys'] = len(inv)
    print('trained',training,flush=True)
    cal = [make_stream(CAL+100*i+j,k,length) for i,k in enumerate(KINDS[:3]) for j in range(3)]
    calkeys = {canonical(d)[0] for ds in cal for d in ds}
    if inv.intersection(calkeys):
        raise ValueError('training/calibration overlap')
    inv.update(calkeys)
    grid = {}
    for arm,weights in (('cost',w),('sham',sham)):
        grid[arm] = [(tau,sum(evaluate(ds,arm,weights,tau)[0]['cpu_ms'] for ds in cal))
                     for tau in (0.,.25,.5,1.)]
    tau = {a:min(rows,key=lambda r:(r[1],r[0]))[0] for a,rows in grid.items()}
    print('calibration',grid,tau,flush=True)
    report = dict(schema='rv22q.activation.v1',train=training,calibration_keys=len(calkeys),weights=w,
                  shuffled_weights=sham,calibration=grid,thresholds=tau,cap=CAP,reserve=RESERVE,
                  streams=n,length=length,repeats=repeats,cohorts={})
    rng = random.Random(424)
    for i,kind in enumerate(KINDS):
        rows = {a:[] for a in ARMS}
        cohort = set()
        for j in range(n):
            ds = make_stream(EVAL+100*i+j,kind,length)
            cohort.update(canonical(d)[0] for d in ds)
            samples,ref = {a:[] for a in ARMS},None
            for _ in range(repeats):
                order = list(ARMS)
                rng.shuffle(order)
                for a in order:
                    m,vals = evaluate(ds,a,sham if a=='sham' else w,tau.get(a,0.),seed=913+j)
                    if ref is None:
                        ref = vals
                    if ref != vals or m['peak_accounted'] > CAP:
                        raise ValueError('exact parity or capacity')
                    samples[a].append(m)
            for a,rs in samples.items():
                rows[a].append({k:float(np.mean([r[k] for r in rs])) for k in rs[0]})
        if inv.intersection(cohort):
            raise ValueError('canonical split overlap')
        inv.update(cohort)
        means = {a:{k:float(np.mean([r[k] for r in rs])) for k in rs[0]} for a,rs in rows.items()}
        idx = np.random.default_rng(425+i).integers(n,size=(10000,n))
        paired = {}
        for a in ARMS:
            for b in sorted({'cold','always_second','sample_second','switch1','sham'}-{a}):
                d = np.array([u['cpu_ms']-v['cpu_ms'] for u,v in zip(rows[a],rows[b],strict=True)])
                paired[a+'_vs_'+b] = dict(delta_ms=float(d.mean()),
                                          ci95=np.quantile(d[idx].mean(1),[.025,.975]).tolist())
        report['cohorts'][kind] = dict(keys=len(cohort),means=means,paired=paired)
        print(kind,{a:round(v['cpu_ms'],3) for a,v in means.items()},flush=True)
    report['arm_queries'] = n*length*repeats*len(KINDS)*len(ARMS)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--streams',type=int,default=4)
    p.add_argument('--length',type=int,default=48)
    p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args()
    if a.out.exists() or not 1<=a.streams<=32 or not 24<=a.length<=128 or not 1<=a.repeats<=4:
        raise ValueError('fresh output and bounded campaign required')
    a.out.write_text(json.dumps(campaign(a.streams,a.length,a.repeats),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
