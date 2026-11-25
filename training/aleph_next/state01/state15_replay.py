"""Replay STATE10--14 archived proofs with a real Lean executable; no training admission."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PINS = {
    10: {'STATE10_POSITIVE.lean': '35b37487de75ab682c322d8aae65d05101de58b7036b4cae6fd172e3b0dd8c75', 'STATE10_NEGATIVE.json': 'ac19c1508ddf101e6710f3b7781552f3b04b8b86c3e5061f5733f18ffd8d548d'},
    11: {'STATE11_POSITIVE.lean': '9702ca64b5a04240fc6043c53ad1a71f41e164e745d725bf3e335972f48f78d6', 'STATE11_NEGATIVE.json': '981ea058617281877414448924e39f77a73481ed226fb978724beb035b7bcd9f'},
    12: {'STATE12_PORTFOLIO.lean': '282e09f6abc02cc0d0bc2fd7e5f60c2f2c44e87b17e10e23bb8bb522ac1c8302', 'STATE12_CERTIFICATES.json.gz': '5ac8ddf8be0ab32b1e3b2be1c973298e98746c228dda888e914eaa5edcac610b'},
    13: {'STATE13_CERTIFICATES.json.gz': '6c73346a16aa69ceec1e7f5ee02e8ca02e525f619a5cbbaa83336d9bb307a2c0'},
    14: {'STATE14_CERTIFICATES.json.gz': 'cbccdf054947c34040f06e04d09674698a53bd359ae3a7cf977ff1a425cb0a68'},
}
ROW_PINS = {13: 'cb113671b481f399d3b1124ca4f1a6f3eb7be04279728fc16cfb3884a23e2d38', 14: '6766ea1684e375703af73de694a48b94440dd9a31df4c3e46668e2b10cc0650e'}
EXPECTED = {10: 168, 11: 353, 12: 192, 13: 319, 14: 332}
ALLOWED_AXIOMS = {'propext', 'Quot.sound', 'Classical.choice'}
HEADER = 'set_option autoImplicit false\n'
NAME = re.compile(r'^theorem ([A-Za-z0-9_]+)\b', re.M)
AXIOMS = re.compile(r"^'([^']+)' (?:does not depend on any axioms|depends on axioms: \[(.*)\])$", re.M)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_checked(path, expected):
    data = path.read_bytes()
    if path.suffix == '.gz':
        data = gzip.decompress(data)
    if digest(data) != expected:
        raise ValueError(f'archive hash mismatch: {path}')
    return data


def tuples(v):
    return tuple(tuples(a) for a in v) if isinstance(v, list) else v


def prepare(root, out):
    import state12
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    negatives = []
    for n, paths in PINS.items():
        stage = root / f'aleph_state{n}'
        for name, expected in paths.items():
            read_checked(stage / name, expected)
        if n < 13:
            name = f'STATE{n}_POSITIVE.lean' if n < 12 else 'STATE12_PORTFOLIO.lean'
            source = (stage / name).read_text(encoding='utf-8')
        else:
            receipt = json.loads((stage / f'STATE{n}_RECEIPTS.json').read_text())
            rows = receipt['records' if n == 13 else 'rows']
            slim = [{k: row[k] for k in ('domain', 'x', 'y', 'truth', 'split')} for row in rows]
            if digest(json.dumps(slim, separators=(',', ':')).encode()) != ROW_PINS[n]:
                raise ValueError(f'STATE{n} row identity mismatch')
            certificates = json.loads(read_checked(stage / f'STATE{n}_CERTIFICATES.json.gz', paths[f'STATE{n}_CERTIFICATES.json.gz']))
            terms = []
            for c in certificates:
                row = rows[c['row']]
                assert row['split'] == 'held' and row['truth']
                x, y, p = tuples(row['x']), tuples(row['y']), tuples(c['path'])
                assert state12.verify(row['domain'], x, y, p)
                terms.append(state12.declaration(f's{n}_{c["row"]}_{c["action"]}', row['domain'], x, y, p))
            source = HEADER + '\n'.join(terms)
        # Each theorem is closed and independent of the other generated theorems.
        source = re.sub(r'^#print axioms .*$\n?', '', source, flags=re.M)
        names = NAME.findall(source)
        assert len(names) == EXPECTED[n] and len(set(names)) == len(names)
        assert not re.search(r'\b(sorry|admit|axiom|unsafe)\b', source)
        source += '\n' + '\n'.join('#print axioms ' + x for x in names) + '\n'
        path = out / f'STATE{n}_REPLAY.lean'
        path.write_text(source, encoding='utf-8')
        counts[n] = {'positive': len(names), 'source_sha256': digest(path.read_bytes()), 'source_bytes': path.stat().st_size}
        if n < 12:
            for row in json.loads((stage / f'STATE{n}_NEGATIVE.json').read_text()):
                negatives.append({'stage': n, 'id': row['id'], 'context': row['context'] if n == 10 else row['source'], 'invalid': row['invalid']})
    assert len(negatives) == 136
    (out / 'NEGATIVES.json').write_text(json.dumps(negatives, indent=2), encoding='utf-8')
    result = {'schema': 'datagoblin.state15.replay.v1', 'archive_pins': PINS, 'prepared': counts, 'negative_cases': len(negatives), 'kernel_verified': False}
    (out / 'PREPARED.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def regenerate(root):
    # Exact hash equality is mandatory, so numerical/platform drift cannot silently replace archived proofs.
    import state10
    import state11
    import state12
    import state12_portfolio
    import state13
    import state14
    root.mkdir(parents=True, exist_ok=True)
    for n, module in [(10, state10), (11, state11), (13, state13), (14, state14)]:
        folder = root / f'aleph_state{n}'
        (module.run if n < 12 else module.main)(folder)
    folder = root / 'aleph_state12'
    folder.mkdir(exist_ok=True)
    panel_hash = digest(repr(state12.panel()).encode())
    assert panel_hash == 'eaaeb9e75887d826134c77b7665933bcaec2f4fc090ed3b38e3aff568798c09e'
    # The portfolio runner only needs the panel hash and presence of the certificates key.
    (folder / 'STATE12_RECEIPTS.json').write_text(json.dumps({'panel_sha256': panel_hash, 'certificates': []}))
    state12_portfolio.run(folder)


def replay(out, batch_size):
    lean = shutil.which('lean')
    if not lean:
        raise RuntimeError('Lean executable unavailable; no kernel verification claimed')
    report = json.loads((out / 'PREPARED.json').read_text())
    version = subprocess.check_output([lean, '--version'], text=True).strip()
    executable = subprocess.check_output([lean, '--print-prefix'], text=True).strip()
    report.update(lean_version=version, lean_prefix=executable, kernel_verified=False, positive_batches=[], negative_checks=[])
    batch_dir = out / 'batches'
    batch_dir.mkdir(exist_ok=True)

    def checkpoint():
        (out / 'KERNEL_RECEIPTS.json').write_text(json.dumps(report, indent=2) + '\n')

    def compile_file(file):
        start = time.perf_counter()
        p = subprocess.run([lean, '-j', '1', str(file)], capture_output=True, text=True, timeout=180)
        log = p.stdout + p.stderr
        file.with_suffix('.log').write_text(log)
        return p.returncode, log, time.perf_counter() - start

    checkpoint()
    for n in EXPECTED:
        source = (out / f'STATE{n}_REPLAY.lean').read_text()
        assert digest(source.encode()) == report['prepared'][str(n)]['source_sha256']
        source = re.sub(r'^#print axioms .*$\n?', '', source, flags=re.M)
        starts = list(NAME.finditer(source))
        terms = [(m.group(1), source[m.start():starts[i+1].start() if i+1 < len(starts) else len(source)]) for i, m in enumerate(starts)]
        for i in range(0, len(terms), batch_size):
            batch = terms[i:i+batch_size]
            file = batch_dir / f's{n}_{i:04d}.lean'
            text = HEADER + '\n'.join(v for _, v in batch) + '\n' + '\n'.join('#print axioms ' + name for name, _ in batch) + '\n'
            file.write_text(text)
            code, log, seconds = compile_file(file)
            ax = {name: {a.strip() for a in vals.split(',') if a.strip()} for name, vals in AXIOMS.findall(log)}
            valid = code == 0 and set(ax) == {name for name, _ in batch} and all(a <= ALLOWED_AXIOMS for a in ax.values())
            report['positive_batches'].append(dict(stage=n, start=i, declarations=len(batch), wall_seconds=seconds, passed=valid, exit_code=code, source_sha256=digest(text.encode()), log_sha256=digest(log.encode()), axioms={k: sorted(v) for k, v in ax.items()}))
            checkpoint()
    for i, row in enumerate(json.loads((out / 'NEGATIVES.json').read_text())):
        path = batch_dir / f'negative_{i:03d}_context.lean'
        path.write_text(HEADER + row['context'])
        c1, l1, t1 = compile_file(path)
        bad = path.with_name(f'negative_{i:03d}_invalid.lean')
        bad.write_text(HEADER + row['context'] + '\n' + row['invalid'])
        c2, l2, t2 = compile_file(bad)
        passed = c1 == 0 and c2 != 0 and 'type mismatch' in l2.lower()
        report['negative_checks'].append(dict(stage=row['stage'], id=row['id'], passed=passed, context_exit=c1, invalid_exit=c2, context_wall_seconds=t1, invalid_wall_seconds=t2, context_log_sha256=digest(l1.encode()), invalid_log_sha256=digest(l2.encode())))
        checkpoint()
    report['positive_declarations_passed'] = sum(r['declarations'] for r in report['positive_batches'] if r['passed'])
    report['negative_rejections_passed'] = sum(r['passed'] for r in report['negative_checks'])
    report['positive_compile_wall_seconds'] = sum(r['wall_seconds'] for r in report['positive_batches'])
    report['negative_compile_wall_seconds'] = sum(r['context_wall_seconds'] + r['invalid_wall_seconds'] for r in report['negative_checks'])
    report['kernel_verified'] = report['positive_declarations_passed'] == sum(EXPECTED.values()) and report['negative_rejections_passed'] == 136
    checkpoint()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--regenerate', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    assert args.batch_size > 0
    if args.regenerate:
        regenerate(args.archive_root)
    result = prepare(args.archive_root, args.out)
    print(json.dumps(result, indent=2), flush=True)
    if not args.prepare_only:
        report = replay(args.out, args.batch_size)
        if not report['kernel_verified']:
            sys.exit(1)


if __name__ == '__main__':
    main()
