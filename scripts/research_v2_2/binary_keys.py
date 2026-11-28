"""RV22-T: bounded binary canonical keys and checked cache capacity curves."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import sys
import time
from collections import OrderedDict, defaultdict
from itertools import product
from pathlib import Path

import numpy as np
from cold_bypass import DP, canonical, check, enc, replay, variant, world
from refinement import legal_tests, packed_key

MAGIC = b'DGK\x01'
HEADER = struct.Struct('>4sIiiHHB')
U32 = struct.Struct('>I')
MAX_TESTS, MAX_ROWS, MAX_ID, MAX_BYTES = 32, 4096, 128, 65536
H, RESERVE = 64, 1024
CAPS = (2048, 4096, 8192, 16384)
DISTANCES = (4, 12, 24)
PROFILES = ('sparse8', 'dense4')
ARMS = ('cold', 'json', 'integer_json', 'binary')
SEED = 2026550000


def record(d):
    """Integer-coded joint distribution for valid synthetic fixture worlds."""
    ts, hs = legal_tests(d), d['hypotheses']
    codes = [int(h['class'] == 'B') for h in hs]
    for j, t in enumerate(ts, 1):
        for i, bit in enumerate(t['bits']):
            codes[i] |= bit << j
    mass = defaultdict(int)
    for code, h in zip(codes, hs, strict=True):
        mass[code] += h['weight']
    scale = math.gcd(*mass.values())
    return (d['budget'], d['correct_reward'], d['wrong_reward'],
            tuple((t['id'], t['cost']) for t in ts),
            tuple((code, w//scale) for code, w in sorted(mass.items()))), scale


def integer_key(d):
    rec, scale = record(d)
    return enc(rec), scale


def encode_record(rec):
    """Encode a normalized record from the trusted fixture canonicalizer."""
    budget, correct, wrong, tests, rows = rec
    if len(tests) > MAX_TESTS or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError('binary inventory bound')
    width = max(1, (max(w for _, w in rows).bit_length()+7)//8)
    if width > 4:
        raise ValueError('binary mass bound')
    cw = (len(tests)+8)//8
    out = bytearray(HEADER.pack(MAGIC, budget, correct, wrong, len(tests), len(rows), width))
    for name, cost in tests:
        b = name.encode('utf-8')
        if not 1 <= len(b) <= MAX_ID:
            raise ValueError('test ID bound')
        out.append(len(b))
        out.extend(b)
        out.extend(U32.pack(cost))
    for code, w in rows:
        out.extend(code.to_bytes(cw, 'big'))
        out.extend(w.to_bytes(width, 'big'))
    if len(out) > MAX_BYTES:
        raise ValueError('key byte bound')
    return bytes(out)


def binary_key(d):
    rec, scale = record(d)
    return encode_record(rec), scale


def decode(data):
    """Strict bounded decoder for canonical keys, not a certificate importer."""
    if type(data) is not bytes or not HEADER.size <= len(data) <= MAX_BYTES:
        raise ValueError('invalid key length/type')
    magic, budget, correct, wrong, nt, nr, width = HEADER.unpack_from(data)
    if (magic != MAGIC or nt > MAX_TESTS or not 1 <= nr <= MAX_ROWS
            or not 1 <= width <= 4 or correct < max(0, wrong)):
        raise ValueError('invalid header')
    pos = HEADER.size
    tests = []
    for _ in range(nt):
        if pos >= len(data):
            raise ValueError('truncated ID')
        n = data[pos]
        pos += 1
        if not 1 <= n <= MAX_ID or pos+n+4 > len(data):
            raise ValueError('invalid ID length')
        try:
            name = data[pos:pos+n].decode('utf-8', errors='strict')
        except UnicodeDecodeError as exc:
            raise ValueError('invalid ID encoding') from exc
        pos += n
        cost = int.from_bytes(data[pos:pos+4], 'big')
        pos += 4
        if (name in ('A', 'B', 'defer') or (tests and tests[-1][0] >= name)
                or not 0 < cost <= budget):
            raise ValueError('invalid legal test')
        tests.append((name, cost))
    cw = (nt+8)//8
    if len(data)-pos != nr*(cw+width):
        raise ValueError('invalid row length')
    rows, seen_one, seen_zero = [], 0, 0
    mask = ((1 << (nt+1))-1) ^ 1
    for _ in range(nr):
        code = int.from_bytes(data[pos:pos+cw], 'big')
        pos += cw
        w = int.from_bytes(data[pos:pos+width], 'big')
        pos += width
        if code >= 1 << (nt+1) or w == 0 or (rows and rows[-1][0] >= code):
            raise ValueError('invalid row')
        seen_one |= code & mask
        seen_zero |= (~code) & mask
        rows.append((code, w))
    if seen_one != mask or seen_zero != mask:
        raise ValueError('constant test in canonical key')
    if math.gcd(*(w for _, w in rows)) != 1:
        raise ValueError('unnormalized masses')
    if width != max(1, (max(w for _, w in rows).bit_length()+7)//8):
        raise ValueError('nonminimal mass width')
    return budget, correct, wrong, tuple(tests), tuple(rows)


def to_world(rec, scale=1):
    budget, correct, wrong, ts, rows = rec
    return dict(budget=budget, correct_reward=correct, wrong_reward=wrong,
                hypotheses=[{'class': 'AB'[c&1], 'weight': w*scale} for c,w in rows],
                tests=[dict(id=name, cost=cost, bits=[(c >> j)&1 for c,_ in rows])
                       for j,(name,cost) in enumerate(ts,1)])


def generate(seed, profile, distance, length):
    rng = random.Random(seed)
    roots = []
    for _ in range(distance):
        if profile == 'sparse8':
            d = world(rng,16)
        elif profile == 'dense4':
            rows = [(c, bits) for c in 'AB' for bits in product((0,1), repeat=4)]
            d = dict(hypotheses=[{'class': c, 'weight': rng.randrange(1,6)} for c,_ in rows],
                     tests=[dict(id=f't{j}', cost=rng.randrange(1,4), bits=[b[j] for _,b in rows])
                            for j in range(4)], budget=7, correct_reward=12, wrong_reward=-16)
        else:
            raise ValueError('unknown profile')
        roots.append(d)
    return [variant(roots[t%distance],rng) for t in range(length)]


def digest(k):
    return hashlib.blake2b(k,digest_size=8).digest()


def evaluate(ds, arm, cap):
    if arm not in ARMS or cap < RESERVE:
        raise ValueError('invalid arm or cap')
    cache, history = OrderedDict(), OrderedDict()
    used, reserve = 0, 0 if arm == 'cold' else RESERVE
    stats = dict(cpu_ms=0., key_ms=0., key_bytes=0, hits=0, solves=0, checks=0,
                 admissions=0, evictions=0, solve_edges=0, verify_edges=0,
                 peak_accounted=reserve, peak_retained_estimate=0)
    values = []
    for t,d in enumerate(ds):
        start = time.process_time_ns()
        k,scale = None,1
        repeat = False
        if arm != 'cold':
            begin = time.process_time_ns()
            k,scale = {'json':canonical,'integer_json':integer_key,'binary':binary_key}[arm](d)
            stats['key_ms'] += (time.process_time_ns()-begin)/1e6
            stats['key_bytes'] += len(k)
            h = digest(k)
            prior = history.pop(h,None)
            repeat = prior is not None and t-prior <= H
            history[h] = t
            if len(history) > H:
                history.popitem(last=False)
        if k is not None and k in cache:
            v,p = json.loads(cache[k])
            cache.move_to_end(k)
            v *= scale
            if replay(d,p) != v:
                raise ValueError('cached replay mismatch')
            stats['hits'] += 1
        else:
            s = DP(d)
            v,p = s.run()
            stats['solves'] += 1
            stats['solve_edges'] += s.edges
            if repeat:
                if v%scale:
                    raise ValueError('invalid normalization')
                payload = enc((v//scale,p))
                size = len(k)+len(payload)
                if size <= cap-reserve:
                    stats['verify_edges'] += check(s)
                    stats['checks'] += 1
                    while used+size > cap-reserve:
                        old,data = cache.popitem(last=False)
                        used -= len(old)+len(data)
                        stats['evictions'] += 1
                    cache[k] = payload
                    used += size
                    stats['admissions'] += 1
        stats['cpu_ms'] += (time.process_time_ns()-start)/1e6
        stats['peak_accounted'] = max(stats['peak_accounted'],used+reserve)
        retained = sum(sys.getsizeof(m)+sum(sys.getsizeof(a)+sys.getsizeof(b) for a,b in m.items())
                       for m in (cache,history))
        stats['peak_retained_estimate'] = max(stats['peak_retained_estimate'],retained)
        values.append((v,sum(h['weight'] for h in d['hypotheses'])))
    if (stats['peak_accounted'] > cap or stats['hits']+stats['solves'] != len(ds)
            or stats['checks'] != stats['admissions']):
        raise ValueError('accounting failure')
    return stats,values


def cohort(profile, distance, n=4, length=72, repeats=2, seed=SEED):
    pi,di = PROFILES.index(profile),DISTANCES.index(distance)
    rng = random.Random(seed+9999+1000*pi+100*di)
    roots, old_new, new_old = set(), {}, {}
    byte_counts = dict(json=0,integer_json=0,binary=0)
    rows = {cap:{a:[] for a in ARMS} for cap in CAPS}
    for j in range(n):
        ds = generate(seed+1000*pi+100*di+j,profile,distance,length)
        stream_keys = set()
        for d in ds:
            old,s = canonical(d)
            integer,q = integer_key(d)
            binary,r = binary_key(d)
            decoded = decode(binary)
            if (s != q or q != r or packed_key(d) != (integer,q)
                    or integer != enc(decoded) or binary != encode_record(decoded)
                    or canonical(to_world(decoded,r)) != (old,s)
                    or old_new.setdefault(old,binary) != binary or new_old.setdefault(binary,old) != old):
                raise ValueError('key equivalence audit')
            if old not in stream_keys:
                for a,k in (('json',old),('integer_json',integer),('binary',binary)):
                    byte_counts[a] += len(k)
            stream_keys.add(old)
        if len(stream_keys) != distance or roots.intersection(stream_keys):
            raise ValueError('project overlap')
        roots.update(stream_keys)
        ref = None
        samples = {cap:{a:[] for a in ARMS} for cap in CAPS}
        for _ in range(repeats):
            order = [(cap,a) for cap in CAPS for a in ARMS]
            rng.shuffle(order)
            for cap,a in order:
                stats,vals = evaluate(ds,a,cap)
                if ref is None:
                    ref = vals
                if vals != ref:
                    raise ValueError('exact utility mismatch')
                samples[cap][a].append(stats)
        for cap,arms in samples.items():
            for a,rs in arms.items():
                rows[cap][a].append({k:float(np.mean([r[k] for r in rs])) for k in rs[0]})
    idx = np.random.default_rng(911+pi*100+di).integers(n,size=(10000,n))
    data = {}
    for cap,arms in rows.items():
        means = {a:{k:float(np.mean([r[k] for r in rs])) for k in rs[0]} for a,rs in arms.items()}
        paired = {}
        for b in ('cold','json','integer_json'):
            delta = np.array([u['cpu_ms']-v['cpu_ms'] for u,v in zip(arms['binary'],arms[b],strict=True)])
            paired[b] = dict(delta_ms=float(delta.mean()),
                             ci95=np.quantile(delta[idx].mean(axis=1),[.025,.975]).tolist())
        data[str(cap)] = dict(means=means,paired_binary=paired)
    return dict(profile=profile,distance=distance,streams=n,length=length,repeats=repeats,seed=seed,
                root_digests=sorted(hashlib.sha256(k).hexdigest() for k in roots),
                key_bytes=byte_counts,caps=data,arm_queries=n*length*repeats*len(CAPS)*len(ARMS))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--profile',choices=PROFILES,required=True)
    p.add_argument('--distance',type=int,choices=DISTANCES,required=True)
    p.add_argument('--streams',type=int,default=4)
    p.add_argument('--length',type=int,default=72)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--seed',type=int,default=SEED)
    a = p.parse_args()
    if a.out.exists() or not 1 <= a.streams <= 16 or not 48 <= a.length <= 128 or not 1 <= a.repeats <= 4:
        raise ValueError('fresh output and bounded campaign required')
    r = cohort(a.profile,a.distance,a.streams,a.length,a.repeats,a.seed)
    a.out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(a.profile,a.distance,r['arm_queries'],r['key_bytes'],flush=True)


if __name__ == '__main__':
    main()
