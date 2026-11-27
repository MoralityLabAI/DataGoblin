"""RV22-L: full exact DP and verified canonical continuation reuse."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path


# A key is (budget, correct reward, wrong reward, ((test ID,cost),...),
#             ((class, bits tuple, normalized mass),...)).
def canonical(budget, correct, wrong, tests, rows):
    keep = [j for j, (_, c) in enumerate(tests)
            if c <= budget and len({bits[j] for _, bits, _ in rows}) > 1]
    grouped = defaultdict(int)
    for label, bits, w in rows:
        grouped[label, tuple(bits[j] for j in keep)] += w
    scale = math.gcd(*grouped.values())
    return (budget, correct, wrong, tuple(tests[j] for j in keep),
            tuple((a, b, w // scale) for (a, b), w in sorted(grouped.items()))), scale


def root(d):
    tests = sorted(d['tests'], key=lambda t: t['id'])
    if (len({t['id'] for t in tests}) != len(tests) or not d['hypotheses']
            or type(d['budget']) is not int or d['budget'] < 0
            or type(d['correct_reward']) is not int or type(d['wrong_reward']) is not int
            or d['correct_reward'] < max(0, d['wrong_reward'])):
        raise ValueError('unsupported world')
    for t in tests:
        if (not isinstance(t['id'], str) or not t['id'] or t['id'] in ('A', 'B', 'defer')
                or type(t['cost']) is not int or t['cost'] <= 0 or len(t['bits']) != len(d['hypotheses'])
                or any(type(b) is not int or b not in (0, 1) for b in t['bits'])):
            raise ValueError('unsupported test')
    rows = []
    for i, h in enumerate(d['hypotheses']):
        if h['class'] not in ('A', 'B') or type(h['weight']) is not int or h['weight'] <= 0:
            raise ValueError('unsupported hypothesis')
        rows.append((h['class'], tuple(t['bits'][i] for t in tests), h['weight']))
    return canonical(d['budget'], d['correct_reward'], d['wrong_reward'],
                     tuple((t['id'], t['cost']) for t in tests), rows)


def mass(k):
    return sum(w for _, _, w in k[4])


def stop(k):
    _, c, r, _, rows = k
    a = sum(w for label, _, w in rows if label == 'A')
    b = mass(k) - a
    return max((a*c+b*r, 'A'), (b*c+a*r, 'B'), (0, 'defer'))


def children(k, j):
    budget, c, r, tests, rows = k
    cost = tests[j][1]
    return [canonical(budget-cost, c, r, tests, [x for x in rows if x[1][j] == bit])
            for bit in (0, 1)]


def replay(d, tree):
    """Independent path replay using the original (unnormalized) hypotheses."""
    tests = {t['id']: t for t in d['tests']}
    total = 0
    for i, h in enumerate(d['hypotheses']):
        node, spent, used = tree, 0, set()
        while len(node) == 3:
            name = node[0]
            if name in used:
                raise ValueError('repeated test')
            used.add(name)
            t = tests[name]
            spent += t['cost']
            if spent > d['budget']:
                raise ValueError('path budget exceeded')
            node = node[1+t['bits'][i]]
        action = node[0]
        reward = 0 if action == 'defer' else d['correct_reward'] if action == h['class'] else d['wrong_reward']
        total += h['weight']*(reward-spent)
    return total


class MaskDP:
    """Cold exact baseline with mask/budget memo and integer masses."""
    def __init__(self, d):
        self.d = d
        self.h = d['hypotheses']
        self.tests = sorted(d['tests'], key=lambda t: t['id'])
        self.masks = [sum(b << i for i, b in enumerate(t['bits'])) for t in self.tests]
        self.counts = {0: (0, 0)}
        self.memo = {}
        self.work = 0

    def count(self, m):
        if m not in self.counts:
            bit = m & -m
            h = self.h[bit.bit_length()-1]
            w, a = self.count(m ^ bit)
            self.counts[m] = w+h['weight'], a+(h['weight'] if h['class']=='A' else 0)
        return self.counts[m]

    def solve(self, m, b):
        key = m, b
        if key not in self.memo:
            w, a = self.count(m)
            c, r = self.d['correct_reward'], self.d['wrong_reward']
            value, action = max((a*c+(w-a)*r, 'A'), ((w-a)*c+a*r, 'B'), (0, 'defer'))
            if value < c*w:
                for t, tm in zip(self.tests, self.masks, strict=True):
                    o = m & tm
                    cost = t['cost']
                    if not o or o == m or cost > b:
                        continue
                    self.work += 1
                    q = self.solve(m^o,b-cost)[0]+self.solve(o,b-cost)[0]-cost*w
                    if q > value:
                        value, action = q, t['id']
            self.memo[key] = value, action
        return self.memo[key]

    def run(self):
        m = (1 << len(self.h))-1
        value = self.solve(m, self.d['budget'])[0]

        def policy(mask, budget):
            _, action = self.memo[mask, budget]
            if action in ('A', 'B', 'defer'):
                return (action,)
            j = next(j for j, t in enumerate(self.tests) if t['id'] == action)
            one = mask & self.masks[j]
            b = budget-self.tests[j]['cost']
            return action, policy(mask ^ one, b), policy(one, b)

        if replay(self.d, policy(m, self.d['budget'])) != value:
            raise ValueError('mask replay mismatch')
        return value, self.count(m)[0]


class CertifiedDP:
    """Trusted in-process memo; only verified records enter persistent memory."""
    def __init__(self, capacity=50_000):
        self.verified = {}
        self.policies = {}
        self.capacity = capacity

    def run(self, d, retain=True):
        k, scale = root(d)
        new = {}
        stats = dict(hits=0, new_states=0, solve_edges=0, verify_edges=0,
                     root_hit=int(k in self.verified), policy_hit=int(k in self.policies))

        def solve(key):
            if key in self.verified:
                stats['hits'] += 1
                return self.verified[key]
            if key not in new:
                value, decision = stop(key)
                w = mass(key)
                if value < key[1]*w:
                    for j, (name, cost) in enumerate(key[3]):
                        stats['solve_edges'] += 1
                        q = sum(s*solve(child)[0] for child,s in children(key,j))-cost*w
                        if q > value:
                            value, decision = q, name
                new[key] = value, decision
            return new[key]

        value, _ = solve(k)
        # Bottom-up exact Bellman check of all new records. Existing records were
        # checked earlier in this process. New records cannot depend on unchecked ones.
        checked = {}
        for key, record in sorted(new.items(), key=lambda p: p[0][0]):
            best, decision = stop(key)
            w = mass(key)
            if best < key[1]*w:
                for j, (name, cost) in enumerate(key[3]):
                    stats['verify_edges'] += 1
                    q = -cost*w
                    for child, s in children(key,j):
                        known = checked.get(child, self.verified.get(child))
                        if known is None:
                            raise ValueError('missing verified dependency')
                        q += s*known[0]
                    if q > best:
                        best, decision = q, name
            if record != (best,decision):
                raise ValueError('invalid Bellman record')
            checked[key] = record
        stats['new_states'] = len(checked)
        if retain and len(self.verified)+len(checked) <= self.capacity:
            self.verified.update(checked)
        # Compile once, then replay using the original rows and stable test IDs.
        def policy(key):
            if key in self.policies:
                return self.policies[key]
            rec = checked.get(key, self.verified.get(key))
            if rec is None:
                raise ValueError('missing replay record')
            _, action = rec
            if action in ('A', 'B', 'defer'):
                tree = (action,)
            else:
                j = next(j for j, (name, _) in enumerate(key[3]) if name == action)
                zero, one = children(key, j)
                tree = action, policy(zero[0]), policy(one[0])
            if key in self.verified:
                self.policies[key] = tree
            return tree

        total = replay(d, policy(k))
        if total != value*scale:
            raise ValueError('certificate replay mismatch')
        return total, mass(k)*scale, stats


def world(rng,n,m):
    labels = list('A'*(n//2)+'B'*(n-n//2))
    rng.shuffle(labels)
    cols = []
    while len(cols)<m:
        bits = [rng.randrange(2) for _ in labels]
        if 0 in bits and 1 in bits and bits not in cols and [1-b for b in bits] not in cols:
            cols.append(bits)
    return dict(hypotheses=[dict(weight=rng.randrange(1,6), **{'class':a}) for a in labels],
                tests=[dict(id=f't{j}',cost=rng.randrange(1,4),bits=bits) for j,bits in enumerate(cols)],
                budget=7 if n==12 else 9,correct_reward=12,wrong_reward=-16)


def subset(d, indices):
    out = copy.deepcopy(d)
    out['hypotheses'] = [copy.deepcopy(d['hypotheses'][i]) for i in indices]
    for t in out['tests']:
        t['bits'] = [t['bits'][i] for i in indices]
    return out


def variants(d,rng):
    order = list(range(len(d['hypotheses'])))
    rng.shuffle(order)
    perm = subset(d,order)
    rng.shuffle(perm['tests'])
    for h in perm['hypotheses']:
        h['weight'] *= 3
    split = subset(perm,[0,0,*range(1,len(order))])
    w = perm['hypotheses'][0]['weight']
    split['hypotheses'][0]['weight'],split['hypotheses'][1]['weight'] = 1,w-1
    mutated = copy.deepcopy(d)
    mutated['tests'][0]['cost'] += 1
    t = d['tests'][0]
    conditioned = subset(d,[i for i,b in enumerate(t['bits']) if b==1])
    conditioned['budget'] -= t['cost']
    return [('anchor',d),('perm_scale',perm),('split_duplicate',split),('cost_mutation',mutated),
            ('conditioned',conditioned),('unrelated',world(rng,len(order),len(d['tests'])))]


def campaign(roots=64):
    rng = random.Random(2026112813)
    arm_rng = random.Random(2026112814)
    rows = defaultdict(list)
    for j in range(roots):
        memory = CertifiedDP()
        d = world(rng,12 if j%2==0 else 16,6 if j%2==0 else 8)
        for name,w in variants(d,rng):
            arms = ['mask', 'cold', 'warm']
            arm_rng.shuffle(arms)
            for arm in arms:
                t = time.process_time_ns()
                if arm == 'mask':
                    m = MaskDP(w)
                    a, b = m.run()
                    mask_ms = (time.process_time_ns()-t)/1e6
                elif arm == 'cold':
                    c, e, cold = CertifiedDP().run(w, retain=False)
                    cold_ms = (time.process_time_ns()-t)/1e6
                else:
                    f, g, warm = memory.run(w)
                    warm_ms = (time.process_time_ns()-t)/1e6
            if a*e != c*b or a*g != f*b:
                raise ValueError('exact value mismatch')
            rows[name].append(dict(mask_ms=mask_ms,cold_ms=cold_ms,warm_ms=warm_ms,mask_states=len(m.memo),
                                   mask_edges=m.work,utility=a/b,cold=cold,warm=warm,
                                   retained=len(memory.verified)))
    result={}
    for name,rs in rows.items():
        n=len(rs)
        result[name]={k:sum(r[k] for r in rs)/n for k in
                      ('mask_ms','cold_ms','warm_ms','mask_states','mask_edges','utility','retained')}
        for lane in ('cold','warm'):
            result[name][lane]={k:sum(r[lane][k] for r in rs) for k in rs[0][lane]}
    return dict(schema='rv22l.certified_dp.v1',roots=roots,queries=roots*6,results=result,
                scope=('Synthetic exact deterministic worlds; randomized arm order within query; '
                       'local CPU timing includes policy compilation and replay; verified reuse within trajectory.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--roots',type=int,default=64)
    args=p.parse_args()
    if args.out.exists() or not 1<=args.roots<=256:
        raise ValueError('fresh output and bounded roots required')
    args.out.write_text(json.dumps(campaign(args.roots),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
