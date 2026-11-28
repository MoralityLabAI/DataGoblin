"""Exact deterministic evidence engine extracted from RV22-P/T (CPU only)."""
from __future__ import annotations

import json
import math
import struct
from collections import defaultdict

MAGIC = b'DGK\x01'
HEADER = struct.Struct('>4sIiiHHB')
U32 = struct.Struct('>I')
MAX_TESTS, MAX_ROWS, MAX_ID, MAX_BYTES = 32, 4096, 128, 65536


def enc(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


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
                    if self.edges > 100_000:
                        raise ValueError('exact solver work bound')
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


def legal_tests(d):
    return sorted((t for t in d['tests'] if t['cost'] <= d['budget']
                   and 0 in t['bits'] and 1 in t['bits']), key=lambda t: t['id'])


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


