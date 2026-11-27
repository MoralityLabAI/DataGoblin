"""RV22-N: synthetic cost-aware exact root memory."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

CAP, H = 8192, 12
KINDS = ('burst', 'mixed', 'churn', 'phase')
ARMS = ('cold', 'trusted_lru', 'trusted_analytic', 'trusted_learned', 'checked_lru',
        'checked_second', 'checked_analytic', 'checked_learned', 'checked_shuffled')


def enc(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def key(d):
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
        reward = 0 if node[0] == 'defer' else d['correct_reward'] if node[0] == h['class'] else d['wrong_reward']
        total += h['weight']*(reward-cost)
    return total


class DP:
    def __init__(self, d):
        self.d = d
        self.hs = d['hypotheses']
        self.ts = sorted(d['tests'], key=lambda t: t['id'])
        self.ms = [sum(b << i for i, b in enumerate(t['bits'])) for t in self.ts]
        self.full = (1 << len(self.hs))-1
        self.counts, self.memo, self.edges = {0: (0, 0)}, {}, 0

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
            raise ValueError('replay')
        return value, p


def check(s):
    """Independent bottom-up check of all legal Bellman alternatives."""
    d = s.d
    ts = sorted(d['tests'], key=lambda t: t['id'])
    masks = [sum(bit << i for i, bit in enumerate(t['bits'])) for t in ts]
    checked, counts, edges = {}, {}, 0
    c, r = d['correct_reward'], d['wrong_reward']
    for (m, b), rec in sorted(s.memo.items(), key=lambda kv: kv[0][1]):
        if m not in counts:
            rows = [h for i, h in enumerate(d['hypotheses']) if m >> i & 1]
            counts[m] = sum(h['weight'] for h in rows), sum(h['weight'] for h in rows if h['class'] == 'A')
        w, a = counts[m]
        v, p = max((a*c+(w-a)*r, 'A'), ((w-a)*c+a*r, 'B'), (0, 'defer'))
        if v < c*w:
            for t, mask in zip(ts, masks, strict=True):
                o, cost = m & mask, t['cost']
                if not o or o == m or cost > b:
                    continue
                edges += 1
                if (o, b-cost) not in checked or (m^o, b-cost) not in checked:
                    raise ValueError('missing child')
                q = checked[o, b-cost][0]+checked[m^o, b-cost][0]-cost*w
                if q > v:
                    v, p = q, t['id']
        if rec != (v, p):
            raise ValueError('invalid Bellman record')
        checked[m, b] = rec
    if (s.full, d['budget']) not in checked:
        raise ValueError('missing root')
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


def stream(seed, kind, length):
    rng = random.Random(seed)
    projects = [world(rng, 12 if j%2 == 0 else 16) for j in range(10)]
    out = []
    for t in range(length):
        phase = ('burst' if t < length//2 else 'churn') if kind == 'phase' else kind
        i = (t//8)%10 if phase == 'burst' else t%10
        if phase == 'mixed':
            i = rng.choices(range(10), weights=[8,6,4,3,2,2,1,1,1,1])[0]
        d = world(rng, 12 if t%2 == 0 else 16) if phase == 'churn' and rng.random() < .7 else projects[i]
        out.append(variant(d, rng))
    return out


def observe(hist, k, t):
    count, first, last = hist.pop(k, (0, t, t-H))
    hist[k] = count+1, first, t
    if len(hist) > 128:
        hist.popitem(last=False)
    return count, min(H, t-last), max(1, t-first)


def features(obs, d, saving):
    count, gap, span = obs
    c, g = min(count, 8)/8, gap/H
    return [1., c, g, float(gap == 1), min(count/span, 1.), len(d['hypotheses'])/24,
            len(d['tests'])/8, d['budget']/9, saving, saving*c, saving*(1-g)]


def train():
    x, y, ratios, hit_times = [], [], [], []
    for ki, kind in enumerate(KINDS[:3]):
        for j in range(12):
            ds = stream(2026120100+100*ki+j, kind, 60)
            keys, cold, hits = [], [], []
            for d in ds:
                k, scale = key(d)
                keys.append(k)
                start = time.process_time_ns()
                s = DP(d)
                v, p = s.run()
                miss = (time.process_time_ns()-start)/1e6
                payload = enc((v//scale, p))
                start = time.process_time_ns()
                u, tree = json.loads(payload)
                if replay(d, tree) != u*scale or v != u*scale:
                    raise ValueError('training replay')
                hit = (time.process_time_ns()-start)/1e6
                start = time.process_time_ns()
                check(s)
                ratios.append((time.process_time_ns()-start)/1e6/max(miss, 1e-6))
                cold.append(miss)
                hits.append(hit)
                hit_times.append(hit)
            hist = OrderedDict()
            for t, d in enumerate(ds[:-H]):
                x.append(features(observe(hist, keys[t], t), d, max(0., cold[t]-hits[t])))
                y.append(sum(max(0., cold[k]-hits[k]) for k in range(t+1, t+1+H) if keys[k] == keys[t]))
    x, y = np.asarray(x), np.asarray(y)
    ridge = x.T@x + .5*np.eye(x.shape[1])
    return dict(weights=np.linalg.solve(ridge, x.T@y).tolist(),
                shuffled=np.linalg.solve(ridge, x.T@np.random.default_rng(91).permutation(y)).tolist(), count=len(y),
                hit_ms=float(np.median(hit_times)), check_ratio=float(np.median(ratios)), target_ms=float(y.mean()))


def evaluate(ds, arm, model, cap=CAP):
    cache, hist = OrderedDict(), OrderedDict()
    occupied = 0
    stats = dict(cpu_ms=0., hits=0, admissions=0, evictions=0, checks=0, solve_edges=0,
                 verify_edges=0, peak_payload=0, peak_estimated_bytes=0, rejected=0)
    values = []
    economic = arm.endswith(('analytic','learned','shuffled'))
    checked = arm.startswith('checked')

    def gross(obs, d, saving):
        if arm.endswith('analytic'):
            c, g, span = obs
            return saving*min(8., .25+H*c/max(span+g,H))
        w = model['shuffled'] if arm.endswith('shuffled') else model['weights']
        return max(0., sum(a*b for a,b in zip(w, features(obs,d,saving), strict=True)))

    for t, d in enumerate(ds):
        start = time.process_time_ns()
        k, scale = key(d) if arm != 'cold' else (None,1)
        obs = observe(hist,k,t) if economic or arm == 'checked_second' else None
        if k in cache:
            payload, saving, val, _last = cache[k]
            v, p = json.loads(payload)
            v *= scale
            if replay(d,p) != v:
                raise ValueError('cached replay')
            cache.move_to_end(k)
            if economic:
                cache[k] = payload,saving,gross(obs,d,saving),t
            stats['hits'] += 1
        else:
            begin = time.process_time_ns()
            s = DP(d)
            v,p = s.run()
            miss = (time.process_time_ns()-begin)/1e6
            stats['solve_edges'] += s.edges
            if arm != 'cold' and (arm != 'checked_second' or obs[0] >= 1):
                if v%scale:
                    raise ValueError('scale')
                payload = enc((v//scale,p))
                size = len(k)+len(payload)
                saving = max(0.,miss-model['hit_ms'])
                val = gross(obs,d,saving) if economic else 0.
                victims, opportunity = [],0.
                need = max(0,occupied+size-cap)
                def remaining(e, now=t):
                    return e[2]*max(0.,1-(now-e[3])/H)
                if need and size <= cap:
                    entries = list(cache.items())
                    if economic:
                        entries.sort(key=lambda kv: remaining(kv[1])/(len(kv[0])+len(kv[1][0])))
                    for old,e in entries:
                        victims.append(old)
                        opportunity += remaining(e)
                        need -= len(old)+len(e[0])
                        if need <= 0:
                            break
                expense = .01 + (miss*model['check_ratio'] if checked else 0.)
                if size <= cap and (not economic or val > expense+opportunity):
                    if checked:
                        stats['verify_edges'] += check(s)
                        stats['checks'] += 1
                    for old in victims:
                        occupied -= len(old)+len(cache.pop(old)[0])
                        stats['evictions'] += 1
                    cache[k] = payload,saving,val,t
                    occupied += size
                    stats['admissions'] += 1
                    if occupied > cap:
                        raise ValueError('payload capacity')
                else:
                    stats['rejected'] += 1
        stats['cpu_ms'] += (time.process_time_ns()-start)/1e6
        stats['peak_payload'] = max(stats['peak_payload'],occupied)
        # Shallow Python object estimate, shared objects may be double counted; not RSS.
        resident = sum(sys.getsizeof(m)+sum(sys.getsizeof(k)+sys.getsizeof(e)+sum(sys.getsizeof(z) for z in e)
                                           for k,e in m.items()) for m in (cache,hist))
        stats['peak_estimated_bytes'] = max(stats['peak_estimated_bytes'],resident)
        values.append((v,sum(h['weight'] for h in d['hypotheses'])))
    return stats,values


def campaign(n=16):
    model = train()
    report = dict(schema='rv22n.economic_memory.v1',model=model,cap=CAP,horizon=H,streams=n,cohorts={})
    rng = random.Random(2026121100)
    for ki,kind in enumerate(KINDS):
        results = {a:[] for a in ARMS}
        for j in range(n):
            ds = stream(2026128000+100*ki+j,kind,48)
            arms = list(ARMS)
            rng.shuffle(arms)
            ref = None
            for a in arms:
                stats,vals = evaluate(ds,a,model)
                if ref is None:
                    ref = vals
                if vals != ref:
                    raise ValueError('cross-arm exact mismatch')
                results[a].append(stats)
        idx = np.random.default_rng(991+ki).integers(n,size=(10000,n))
        means = {a:{k:float(np.mean([r[k] for r in rows])) for k in rows[0]} for a,rows in results.items()}
        paired = {}
        for a in ARMS[1:]:
            ref = 'checked_lru' if a.startswith('checked') else 'trusted_lru'
            for b in sorted({'cold',ref}-{a}):
                d = np.array([u['cpu_ms']-v['cpu_ms'] for u,v in zip(results[a],results[b],strict=True)])
                paired[a+'_vs_'+b] = dict(delta_ms=float(d.mean()),
                                          ci95=np.quantile(d[idx].mean(1),[.025,.975]).tolist())
        report['cohorts'][kind] = dict(means=means,paired=paired)
        print(kind,{a:round(r['cpu_ms'],2) for a,r in means.items()},flush=True)
    report['arm_queries'] = n*len(KINDS)*48*len(ARMS)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--streams',type=int,default=16)
    a = p.parse_args()
    if a.out.exists() or not 1 <= a.streams <= 64:
        raise ValueError('fresh output and bounded streams required')
    a.out.write_text(json.dumps(campaign(a.streams),indent=2,sort_keys=True,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
