"""Post-hoc semantic gate and two-frontier -> normalization portfolio."""
import csv
import gzip
import hashlib
import json
import time
from pathlib import Path

import state12 as s


def compact_path(x, path):
    out = []
    for z, i, p, rev in path:
        # Reversed erasing rules need a witness for the larger subterm.
        out.append([i, p, rev, s.s.get(z, p)] if rev else [i, p, rev])
        x = z
    return out


def restore(x):
    return tuple(restore(a) for a in x) if isinstance(x, list) else x


def expand_path(x, compact):
    out = []
    for i, p, rev, *tail in compact:
        p = tuple(p)
        if rev:
            z = s.s.replace(x, p, restore(tail[0]))
            assert s.apply(z, i, p) == x
        else:
            z = s.apply(x, i, p)
        out.append((z, i, p, rev))
        x = z
    return out


def run(out):
    rows = s.panel()
    prior = json.loads((out / 'STATE12_RECEIPTS.json').read_text())
    assert hashlib.sha256(repr(rows).encode()).hexdigest() == prior['panel_sha256']
    records, certs, decl = [], [], []
    s.moves.cache_clear()
    s.s.distance.cache_clear()
    start = time.process_time()
    for j, (d, x, y, truth, kind) in enumerate(rows):
        nf = s.polynomial if d == 'nat' else s.list_word
        equal = nf(x) == nf(y)
        assert equal == truth
        n = edges = steps = tests = visits = 0
        if equal:
            path, n, edges = s.search(d, x, y, 'two_frontier', 64)
            status = 'two_frontier'
            if path is None:
                path, steps, visits, tests, status = s.normal(d, x, y)
                status = 'normal:' + status
            assert path is not None and s.verify(d, x, y, path)
            assert all(nf(z) == nf(x) for z, *_ in path)
            cert = compact_path(x, path)
            restored = expand_path(x, json.loads(json.dumps(cert)))
            assert restored == path
            certs.append(dict(id=j, domain=d, x=x, y=y, steps=cert))
        else:
            path, status = None, 'semantic_inequality'
        records.append(dict(id=j, domain=d, kind=kind, truth=truth, solved=path is not None,
                            expansions=n, edges=edges, normal_steps=steps, node_visits=visits,
                            match_tests=tests, proof_steps=len(path) if path is not None else 0,
                            status=status))
    cpu = time.process_time() - start
    render_start = time.process_time()
    for c in certs:
        x, y = c['x'], c['y']
        path = expand_path(x, c['steps'])
        decl.append(s.declaration('portfolio_'+str(c['id']), c['domain'], x, y, path))
    lean = 'set_option autoImplicit false\n\n'+'\n'.join(decl)+'\n'+'\n'.join('#print axioms portfolio_'+str(c['id']) for c in certs)+'\n'
    (out / 'STATE12_PORTFOLIO.lean').write_text(lean)
    render = time.process_time()-render_start
    with (out / 'STATE12_CERTIFICATES.json.gz').open('wb') as f:
        f.write(gzip.compress(json.dumps(certs,separators=(',',':')).encode(),mtime=0))
    # Keep the numerical receipt small. Full normalizer paths are represented by the
    # original reproducible source, and the portfolio certificates are independently replayable.
    prior.pop('certificates')
    prior['portfolio'] = dict(records=records, cpu_seconds=cpu, lean_render_cpu_seconds=render,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        certificate_sha256=hashlib.sha256((out/'STATE12_CERTIFICATES.json.gz').read_bytes()).hexdigest(),
        lean_sha256=hashlib.sha256(lean.encode()).hexdigest(), lean_kernel_verified=False)
    (out / 'STATE12_COMPACT_RECEIPTS.json').write_text(json.dumps(prior,indent=2)+'\n')
    from collections import Counter
    print('portfolio',Counter(r['status'] for r in records),'cpu',cpu,'render',render)
    print('proof steps',sum(r['proof_steps'] for r in records),'max',max(r['proof_steps'] for r in records),
          'expansions',sum(r['expansions'] for r in records),'normal',sum(r['normal_steps'] for r in records))


if __name__ == '__main__':
    run(Path(__file__).resolve().parent)
