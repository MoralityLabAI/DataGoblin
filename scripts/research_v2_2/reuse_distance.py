"""RV22-V: bounded reuse-distance tracking with a packed checked-cache arena."""
from __future__ import annotations

import argparse
import json
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np
from binary_keys import binary_key, digest
from cold_bypass import DP, check, enc, replay, variant, world

SLOTS, WINDOW, HORIZON = 64, 32, 96
OBS = struct.Struct('<QHHB3x')  # digest, last query + 1, previous interval, count
STATE = struct.Struct('<BBB5x')  # warm streak, novel streak, armed
HEADER = struct.Struct('<I')  # arena used bytes after header
REC = struct.Struct('<HHHHd')  # key bytes, payload bytes, last + 1, gap, saving ms
CAPS = (4096, 8192)
KINDS = ('cycle12', 'cycle40', 'cycle56', 'phase', 'mixed_scan')
ARMS = ('cold', 'short_second', 'short_frequency', 'long_second', 'long_freeze', 'distance', 'distance_reset')
SEED = 2026710000


class History:
    """Same retained allocation for short event ring and long distinct-tag table."""
    __slots__ = ('buf',)

    def __init__(self):
        self.buf = bytearray(STATE.size+SLOTS*OBS.size)

    def retained(self):
        return sys.getsizeof(self)+sys.getsizeof(self.buf)

    def observe(self, tag, t, long):
        now = t+1
        prior = 0
        gap = prev = 0
        oldest = 0
        oldest_time = now
        match = None
        for i in range(SLOTS):
            h,last,g,n = OBS.unpack_from(self.buf, STATE.size+i*OBS.size)
            if last < oldest_time:
                oldest,oldest_time = i,last
            if last and h == tag and (long or now-last <= WINDOW):
                if long:
                    match = i
                    prior,gap,prev = n,now-last,g
                    break
                prior += 1
                gap = min(gap,now-last) if gap else now-last
        if long:
            i = oldest if match is None else match
            OBS.pack_into(self.buf,STATE.size+i*OBS.size,tag,now,gap,min(prior+1,255))
        else:
            OBS.pack_into(self.buf,STATE.size+(t%SLOTS)*OBS.size,tag,now,gap,1)
        warm,novel,armed = STATE.unpack_from(self.buf)
        warm = min(255,warm+1) if prior else 0
        novel = 0 if prior else min(255,novel+1)
        armed = int(bool(armed or warm >= 4))
        reset = bool(armed and novel >= 8)
        if reset:
            warm,novel,armed = 0,0,0
        STATE.pack_into(self.buf,0,warm,novel,armed)
        return prior,gap,prev,reset

    def frequency(self, tag, t):
        return sum(h == tag and last and 0 <= t+1-last <= WINDOW
                   for h,last,_,_ in OBS.iter_unpack(memoryview(self.buf)[STATE.size:]))


class Arena:
    """Retains one fixed bytearray; parsed records are transient query scratch."""
    __slots__ = ('buf',)

    def __init__(self, budget):
        # CPython bytearray allocation includes an object header and terminator.
        n = budget-sys.getsizeof(self)-(sys.getsizeof(bytearray(1))-1)
        if n < HEADER.size:
            raise ValueError('cap below packed metadata')
        self.buf = bytearray(n)
        if self.retained() > budget:
            raise ValueError('retained allocation cap')

    def retained(self):
        return sys.getsizeof(self)+sys.getsizeof(self.buf)

    def rows(self):
        end = HEADER.size+HEADER.unpack_from(self.buf)[0]
        pos = HEADER.size
        out = []
        while pos < end:
            kl,pl,last,gap,saving = REC.unpack_from(self.buf,pos)
            pos += REC.size
            k = bytes(self.buf[pos:pos+kl])
            pos += kl
            payload = bytes(self.buf[pos:pos+pl])
            pos += pl
            out.append((k,payload,last,gap,saving))
        if pos != end:
            raise ValueError('arena record boundary')
        return out

    @staticmethod
    def size(row):
        return REC.size+len(row[0])+len(row[1])

    def available(self):
        return len(self.buf)-HEADER.size

    def write(self, rows):
        used = sum(map(self.size,rows))
        if used > self.available():
            raise ValueError('arena overflow')
        old_end = HEADER.size+HEADER.unpack_from(self.buf)[0]
        pos = HEADER.size
        for k,p,last,gap,saving in rows:
            REC.pack_into(self.buf,pos,len(k),len(p),last,gap,saving)
            pos += REC.size
            self.buf[pos:pos+len(k)] = k
            pos += len(k)
            self.buf[pos:pos+len(p)] = p
            pos += len(p)
        if pos < old_end:
            self.buf[pos:old_end] = bytes(old_end-pos)
        HEADER.pack_into(self.buf,0,used)


def tag_of(k):
    return int.from_bytes(digest(k),'little')


def generate(seed,kind,length):
    rng = random.Random(seed)
    if kind.startswith('cycle'):
        n = int(kind[5:])
        roots = [world(rng,12 if i%3 == 0 else 16) for i in range(n)]
        return [variant(roots[t%n],rng) for t in range(length)]
    if kind == 'phase':
        roots = [world(rng,12 if i%3 == 0 else 16) for i in range(24)]
        return [variant(roots[t%12+12*int(t >= length//2)],rng) for t in range(length)]
    if kind == 'mixed_scan':
        roots = [world(rng,12 if i%2 == 0 else 16) for i in range(12)]
        return [variant(roots[(t//3)%12] if t%3 != 2 else world(rng,12 if t%2 == 0 else 16),rng)
                for t in range(length)]
    raise ValueError('unknown workload')


def calibrate():
    rng = random.Random(SEED-10000)
    ratios,hits,keys = [],[],set()
    for i in range(32):
        d = world(rng,12 if i%2 == 0 else 16)
        keys.add(binary_key(d)[0])
        start = time.process_time_ns()
        s = DP(d)
        v,p = s.run()
        miss = (time.process_time_ns()-start)/1e6
        start = time.process_time_ns()
        check(s)
        ratios.append((time.process_time_ns()-start)/1e6/max(miss,1e-9))
        payload = enc((v,p))
        start = time.process_time_ns()
        u,p = json.loads(payload)
        if replay(d,p) != u:
            raise ValueError('calibration replay')
        hits.append((time.process_time_ns()-start)/1e6)
    return (float(np.median(ratios)),float(np.median(hits))),keys


def evaluate(ds,arm,cap,cost=(1.15,.02)):
    if arm not in ARMS or len(ds) >= 65535:
        raise ValueError('unsupported arm or stream length')
    hist = History() if arm != 'cold' else None
    store = Arena(cap-hist.retained()) if hist is not None else None
    retained = hist.retained()+store.retained() if store is not None else 0
    stats = dict(cpu_ms=0.,hits=0,solves=0,checks=0,admissions=0,evictions=0,
                 resets=0,reset_evictions=0,rejections=0,repeat_hints=0,long_repeat_hints=0,
                 solve_edges=0,verify_edges=0,retained_bytes=retained,peak_record_bytes=0,
                 first_hits=0,second_hits=0)
    vals = []
    for t,d in enumerate(ds):
        start = time.process_time_ns()
        now = t+1
        k,scale = binary_key(d) if store is not None else (None,1)
        rows = store.rows() if store is not None else []
        prior,gap,prev,reset = hist.observe(tag_of(k),t,not arm.startswith('short')) if hist else (0,0,0,False)
        stats['repeat_hints'] += int(bool(prior))
        stats['long_repeat_hints'] += int(gap>WINDOW)
        if arm == 'distance_reset' and reset:
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
            rows.append((key,payload,now,gap or max(1,now-last) or old_gap,saving))
            store.write(rows)
            stats['hits'] += 1
            stats['first_hits' if t < len(ds)//2 else 'second_hits'] += 1
        else:
            begin = time.process_time_ns()
            solver = DP(d)
            v,p = solver.run()
            miss = (time.process_time_ns()-begin)/1e6
            stats['solves'] += 1
            stats['solve_edges'] += solver.edges
            if store is not None and prior:
                if v%scale:
                    raise ValueError('invalid normalization')
                payload = enc((v//scale,p))
                interval = max(1,gap,prev)
                saving = max(0.,miss-cost[1])
                row = (k,payload,now,interval,saving)
                size = Arena.size(row)
                need = max(0,sum(map(Arena.size,rows))+size-store.available())
                victims = []
                admit = size <= store.available()
                def resident(r):
                    value = HORIZON/max(1,r[3])*r[4]
                    if arm == 'distance_reset':
                        value *= max(0.,min(1.,2-(now-r[2])/max(1,r[3])))
                    return value
                if need and admit:
                    if arm == 'long_freeze':
                        admit = False
                    else:
                        ordered = list(rows)
                        if arm == 'short_frequency':
                            ordered.sort(key=lambda r:hist.frequency(tag_of(r[0]),t))
                        elif arm.startswith('distance'):
                            ordered.sort(key=lambda r:resident(r)/Arena.size(r))
                        for r in ordered:
                            victims.append(r)
                            need -= Arena.size(r)
                            if need <= 0:
                                break
                if admit and arm == 'short_frequency' and victims:
                    admit = prior > sum(hist.frequency(tag_of(r[0]),t) for r in victims)
                if admit and arm.startswith('distance'):
                    admit = HORIZON/interval*saving > cost[0]*miss+sum(map(resident,victims))
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
            if store.retained()+hist.retained() != retained or retained > cap:
                raise ValueError('retained capacity changed')
        vals.append((v,sum(h['weight'] for h in d['hypotheses'])))
        # Parsed rows/policy/solver are per-query scratch, not cache state.
        del rows
    if stats['checks'] != stats['admissions'] or stats['hits']+stats['solves'] != len(ds):
        raise ValueError('accounting failure')
    return stats,vals


def campaign(n=3,length=144,seed=SEED):
    cost,seen = calibrate()
    rng = random.Random(seed+9999)
    report = dict(schema='rv22v.reuse_distance.v1',streams=n,length=length,seed=seed,cost=cost,
                  caps=CAPS,history_slots=SLOTS,horizon=HORIZON,cohorts={})
    for ci,kind in enumerate(KINDS):
        rows = {cap:{a:[] for a in ARMS} for cap in CAPS}
        keys = set()
        for j in range(n):
            ds = generate(seed+100*ci+j,kind,length)
            inventory = {binary_key(d)[0] for d in ds}
            if seen.intersection(inventory):
                raise ValueError('cross-stream key overlap')
            seen.update(inventory)
            keys.update(inventory)
            order = [(cap,a) for cap in CAPS for a in ARMS]
            rng.shuffle(order)
            reference = None
            for cap,a in order:
                stats,vals = evaluate(ds,a,cap,cost)
                if reference is None:
                    reference = vals
                if vals != reference:
                    raise ValueError('exact utility mismatch')
                rows[cap][a].append(stats)
        results = {}
        idx = np.random.default_rng(821+ci).integers(n,size=(10000,n))
        for cap,arms in rows.items():
            means = {a:{k:float(np.mean([r[k] for r in rs])) for k in rs[0]} for a,rs in arms.items()}
            paired = {}
            for a in ARMS[1:]:
                for b in ('cold','short_frequency','long_second','long_freeze','distance'):
                    if a == b:
                        continue
                    delta = np.asarray([u['cpu_ms']-v['cpu_ms'] for u,v in zip(arms[a],arms[b],strict=True)])
                    paired[a+'_vs_'+b] = dict(delta_ms=float(delta.mean()),
                        ci95=np.quantile(delta[idx].mean(axis=1),[.025,.975]).tolist())
            results[str(cap)] = dict(means=means,paired=paired)
        report['cohorts'][kind] = dict(unique_keys=len(keys),caps=results)
        print(kind,{c:{a:round(v['cpu_ms'],2) for a,v in x['means'].items()} for c,x in results.items()},flush=True)
    report['arm_queries'] = n*length*len(KINDS)*len(CAPS)*len(ARMS)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--streams',type=int,default=3)
    p.add_argument('--length',type=int,default=144)
    p.add_argument('--seed',type=int,default=SEED)
    a = p.parse_args()
    if a.out.exists() or not 1<=a.streams<=16 or not 72<=a.length<=256:
        raise ValueError('fresh output and bounded campaign required')
    a.out.write_text(json.dumps(campaign(a.streams,a.length,a.seed),indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
