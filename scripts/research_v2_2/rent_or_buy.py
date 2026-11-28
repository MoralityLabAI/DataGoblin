"""RV22-X: rent-or-buy checked admission and finite offline cache oracle."""
from __future__ import annotations

import argparse
import json
import random
import struct
import sys
import time
from functools import cache
from pathlib import Path

import numpy as np
from binary_keys import binary_key
from cold_bypass import DP, check, enc, replay, variant, world
from reuse_distance import Arena, generate, tag_of

SLOTS = 64
OBS = struct.Struct('<QHHI')  # tag, last query + 1, observations, accumulated rental work
ARMS = ('cold', 'eager', 'second', 'third', 'freeze', 'payback', 'payback_freeze', 'payback_guard')
SEED = 2026740000
CAPS = (4096, 8192)
COHORTS = (('short12', 'cycle12', (24, 36, 48)), ('long12', 'cycle12', (96, 120, 144)),
           ('cycle40', 'cycle40', (96, 120, 144)), ('phase', 'phase', (96, 120, 144)),
           ('scan', 'mixed_scan', (72, 96, 120)), ('terminating', 'terminating', (54, 78, 102)))


class Ledger:
    """Bounded digest hints, never evidence of exact identity."""
    __slots__ = ('buf',)

    def __init__(self):
        self.buf = bytearray(SLOTS*OBS.size)

    def retained(self):
        return sys.getsizeof(self)+sys.getsizeof(self.buf)

    def observe(self, tag, t):
        now = t+1
        oldest = 0
        for i in range(SLOTS):
            h,last,n,debt = OBS.unpack_from(self.buf,i*OBS.size)
            if last and h == tag:
                OBS.pack_into(self.buf,i*OBS.size,tag,now,min(n+1,65535),debt)
                return i,n,now-last,debt
            if last < OBS.unpack_from(self.buf,oldest*OBS.size)[1]:
                oldest = i
        OBS.pack_into(self.buf,oldest*OBS.size,tag,now,1,0)
        return oldest,0,0,0

    def charge(self, slot, rental, purchased=False):
        h,last,n,debt = OBS.unpack_from(self.buf,slot*OBS.size)
        OBS.pack_into(self.buf,slot*OBS.size,h,last,n,0 if purchased else min(2**32-1,debt+rental))


def make_stream(seed, profile, length):
    if profile != 'terminating':
        return generate(seed,profile,length)
    rng = random.Random(seed)
    roots = [world(rng,12 if i%2 == 0 else 16) for i in range(12)]
    # Sustained reuse, then a long sequence of distinct roots ending abruptly.
    return [variant(roots[t%12] if t<length//2 else world(rng,12 if t%2 == 0 else 16),rng)
            for t in range(length)]


def evaluate(ds, arm, cap):
    if arm not in ARMS or len(ds) >= 65535:
        raise ValueError('unsupported arm or length')
    hist = Ledger() if arm != 'cold' else None
    store = Arena(cap-hist.retained()) if hist is not None else None
    retained = hist.retained()+store.retained() if store is not None else 0
    stats = dict(cpu_ms=0.,hits=0,solves=0,checks=0,admissions=0,evictions=0,rejections=0,
                 solve_edges=0,verify_edges=0,retained_bytes=retained,first_hits=0,second_hits=0)
    vals, decisions = [], []
    for t,d in enumerate(ds):
        start = time.process_time_ns()
        now = t+1
        k,scale = binary_key(d) if store is not None else (None,1)
        rows = store.rows() if store is not None else []
        slot,prior,gap,debt = hist.observe(tag_of(k),t) if hist else (0,0,0,0)
        found = next((i for i,r in enumerate(rows) if r[0] == k),None)
        decision = 'cold'
        if found is not None:
            key,payload,last,old_gap,rental = rows.pop(found)
            v,p = json.loads(payload)
            v *= scale
            if replay(d,p) != v:
                raise ValueError('cached replay mismatch')
            rows.append((key,payload,now,gap or max(1,now-last,old_gap),rental))
            store.write(rows)
            stats['hits'] += 1
            stats['first_hits' if t < len(ds)//2 else 'second_hits'] += 1
            decision = 'hit'
        else:
            solver = DP(d)
            v,p = solver.run()
            rental,buy = 5*solver.edges,6*solver.edges
            stats['solves'] += 1
            stats['solve_edges'] += solver.edges
            admit = False
            if store is not None:
                eligible = (arm == 'eager' or (arm in ('second','freeze') and prior >= 1)
                            or (arm == 'third' and prior >= 2)
                            or (arm.startswith('payback') and rental > 0 and debt >= buy))
                if eligible:
                    if v%scale:
                        raise ValueError('invalid normalization')
                    row = (k,enc((v//scale,p)),now,max(1,gap),float(rental))
                    size = Arena.size(row)
                    need = max(0,sum(map(Arena.size,rows))+size-store.available())
                    admit = size <= store.available()
                    victims = []
                    def hurdle(r, now=now):
                        # Replacement hysteresis: decay the resident's original buy cost.
                        return 1.2*r[4]*max(0.,min(1.,2-(now-r[2])/max(1,r[3])))
                    if need and admit:
                        if arm in ('freeze','payback_freeze'):
                            admit = False
                        else:
                            ordered = (sorted(rows,key=lambda r:hurdle(r)/Arena.size(r))
                                       if arm == 'payback_guard' else rows)
                            for r in ordered:
                                victims.append(r)
                                need -= Arena.size(r)
                                if need <= 0:
                                    break
                    if admit and arm == 'payback_guard' and victims:
                        admit = debt >= buy+sum(map(hurdle,victims))
                    if admit:
                        stats['verify_edges'] += check(solver)
                        stats['checks'] += 1
                        stats['admissions'] += 1
                        stats['evictions'] += len(victims)
                        rows = [r for r in rows if r not in victims]
                        rows.append(row)
                        store.write(rows)
                        decision = 'buy'
                    else:
                        stats['rejections'] += 1
                hist.charge(slot,rental,admit)
        stats['cpu_ms'] += (time.process_time_ns()-start)/1e6
        if store is not None and (hist.retained()+store.retained() != retained or retained > cap):
            raise ValueError('retained cap changed')
        vals.append((v,sum(h['weight'] for h in d['hypotheses'])))
        decisions.append(decision)
    if stats['checks'] != stats['admissions'] or stats['hits']+stats['solves'] != len(ds):
        raise ValueError('work accounting')
    stats['weighted_edges'] = 5*stats['solve_edges']+6*stats['verify_edges']
    return stats,vals,decisions


def offline(requests, capacity):
    """Exact finite weighted cache DP; each request is (identity,size,rent,buy).

    On a miss: rent; optionally buy after evicting any subset of current entries.
    No prefetch, hit costs zero in this logical-work model. Small streams only.
    """
    ids = {k:i for i,k in enumerate(dict.fromkeys(k for k,_,_,_ in requests))}
    if len(ids)>12 or len(requests)>96 or capacity<0:
        raise ValueError('offline oracle bounds')
    sizes = [0]*len(ids)
    seq = []
    for k,size,r,b in requests:
        i = ids[k]
        if size <= 0 or min(r,b)<0 or sizes[i] not in (0,size):
            raise ValueError('invalid oracle request')
        sizes[i] = size
        seq.append((i,r,b))
    occupied = [sum(s for i,s in enumerate(sizes) if mask>>i&1) for mask in range(1<<len(ids))]
    @cache
    def dp(t,mask):
        if t == len(seq):
            return 0
        i,r,b = seq[t]
        bit = 1<<i
        if mask&bit:
            return dp(t+1,mask)
        best = r+dp(t+1,mask)
        sub = mask
        while True:
            new = sub|bit
            if occupied[new]<=capacity:
                best = min(best,r+b+dp(t+1,new))
            if sub == 0:
                break
            sub = (sub-1)&mask
        return best
    result = dp(0,0)
    return result,dp.cache_info().currsize


def offline_audit(seed=SEED+10000):
    rng = random.Random(seed)
    result = []
    for kind in ('cycle','burst','scan','shift'):
        for j in range(8):
            roots = [world(rng,12 if i%2==0 else 16) for i in range(8)]
            if kind == 'cycle':
                ids = list(range(8))*4
            elif kind == 'burst':
                ids = [i for i in range(8) for _ in range(rng.randrange(2,7))]
            elif kind == 'scan':
                ids = [i%3 if i%2==0 else 3+(i//2)%5 for i in range(32)]
            else:
                ids = list(range(4))*4+list(range(4,8))*4
            ds = [variant(roots[i],rng) for i in ids]
            requests = []
            for d in ds:
                k,scale = binary_key(d)
                s = DP(d)
                v,p = s.run()
                requests.append((k,Arena.size((k,enc((v//scale,p)),0,1,0.)),5*s.edges,6*s.edges))
            # Deliberately limit capacity to 2-4 typical entries, plus identical history cost.
            cap = 2048 if j%2==0 else 2304
            avail = Arena(cap-Ledger().retained()).available()
            opt,states = offline(requests,avail)
            stats = {a:evaluate(ds,a,cap)[0] for a in ARMS}
            if any(s['weighted_edges']<opt for s in stats.values()):
                raise ValueError('online cost below exact offline optimum')
            result.append(dict(kind=kind,cap=cap,queries=len(ds),opt=opt,oracle_states=states,
                               costs={a:s['weighted_edges'] for a,s in stats.items()}))
    return result


def campaign(selected=None):
    rng = random.Random(SEED+9999)
    out = dict(schema='rv22x.rent_or_buy.v1',seed=SEED,caps=CAPS,cohorts={},arm_queries=0)
    seen = set()
    for ci,(kind,profile,lengths) in enumerate(COHORTS):
        if selected is not None and selected != kind:
            continue
        rows = {cap:{a:[] for a in ARMS} for cap in CAPS}
        for j,length in enumerate(lengths):
            ds = make_stream(SEED+100*ci+j,profile,length)
            keys = {binary_key(d)[0] for d in ds}
            if seen.intersection(keys):
                raise ValueError('cross-stream overlap')
            seen.update(keys)
            order = [(cap,a) for cap in CAPS for a in ARMS]
            rng.shuffle(order)
            ref = None
            for cap,a in order:
                stats,vals,_ = evaluate(ds,a,cap)
                if ref is None:
                    ref = vals
                if ref != vals:
                    raise ValueError('exact utility mismatch')
                rows[cap][a].append(stats)
                out['arm_queries'] += len(ds)
        results = {}
        for cap,arms in rows.items():
            means = {a:{k:float(np.mean([r[k] for r in rs])) for k in rs[0]} for a,rs in arms.items()}
            results[str(cap)] = dict(means=means,streams=arms)
        out['cohorts'][kind] = dict(lengths=lengths,caps=results)
        print(kind,{c:{a:round(v['cpu_ms'],2) for a,v in r['means'].items()} for c,r in results.items()},flush=True)
    out['unique_keys'] = len(seen)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cohort',choices=[c[0] for c in COHORTS])
    p.add_argument('--offline',action='store_true')
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('fresh output required')
    result = offline_audit() if a.offline else campaign(a.cohort)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
