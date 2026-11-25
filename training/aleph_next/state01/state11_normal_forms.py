"""Independent denotational normal forms for the synthetic STATE11 grammar."""
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from state11 import SEED, generate


def polynomial(e):
    if isinstance(e, str):
        if e == '0':
            return {}
        if e == '1':
            return {(0, 0, 0): 1}
        return {tuple(int(k == e) for k in 'xyz'): 1}
    op, l, r = e
    a, b = polynomial(l), polynomial(r)
    out = Counter(a)
    if op == '+':
        out.update(b)
    else:
        assert op == '*'
        out = Counter()
        for u, cu in a.items():
            for v, cv in b.items():
                out[tuple(x + y for x, y in zip(u, v))] += cu * cv
    return dict(+out)


def list_word(e):
    if isinstance(e, str):
        return () if e == 'nil' else ((e, False, ()),)
    op, *xs = e
    if op == '++':
        return list_word(xs[0]) + list_word(xs[1])
    word = list_word(xs[0])
    if op == 'rev':
        return tuple((v, not direction, maps) for v, direction, maps in reversed(word))
    assert op in ('mf', 'mg')
    return tuple((v, direction, maps + (op,)) for v, direction, maps in word)


def run():
    root = Path(__file__).resolve().parent
    receipt = json.loads((root / 'STATE11_RECEIPTS.json').read_text())
    r = random.Random(SEED)
    seen = set()
    train, held = [], []
    for d, g in [('nat', {'struct', 'id'}), ('nat', {'dist', 'id'}), ('list', {'append', 'rev'}), ('list', {'append', 'map'})]:
        train += generate(r, d, g, 120, False, seen)
    for d, g in [('nat', {'struct', 'id', 'dist'}), ('list', {'append', 'rev', 'map'})]:
        held += generate(r, d, g, 32, True, seen)
    assert hashlib.sha256(repr(held).encode()).hexdigest() == receipt['held_sha256']
    start = time.process_time()
    counts = Counter()
    for group, rows in [('train', train), ('held', held)]:
        for d, x, y, path in rows:
            nf = polynomial if d == 'nat' else list_word
            source = nf(x)
            assert source == nf(y)
            for z, _, _ in path:
                assert source == nf(z)
                counts[group + '_steps'] += 1
            counts[group + '_' + d] += 1
            # Mutate the target by an operation guaranteed to change its meaning.
            bad = ('+', y, '1') if d == 'nat' else ('++', y, 'xs')
            assert source != nf(bad)
            counts[group + '_negative'] += 1
    result = dict(counts, cpu_seconds=time.process_time() - start,
                  held_sha256=receipt['held_sha256'], kernel_verified=False)
    (root / 'STATE11_NORMAL_FORMS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
