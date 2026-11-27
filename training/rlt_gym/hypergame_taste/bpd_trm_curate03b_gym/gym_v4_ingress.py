"""Fail-closed GYM-02 V4 ingestion; stage-local inputs, separate behavior archive."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


SCHEMA = 'bpd-trm-gym02-stage-v4'
SHAMS = {'F_shuffled_k4', 'H_rotating_sham', 'J_balanced_sham'}
VERIFIED_ARMS = {'C_analytic_full16', 'D_joint_early12'}
STATE_KEYS = {'stage', 'scores', 'update_norms', 'feedback_loss', 'feedback_age',
              'charged_cpu_before', 'cpu_to_stage', 'selected_indices', 'cpu_to_proposal'}


def canonical(row):
    return (json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def audit(rows, roots):
    require(len(roots) >= 3 and len(roots) == len(set(roots)), 'need distinct root groups')
    ordered = sorted(roots)
    partition = {root: 'train' for root in ordered[:-2]}
    partition[ordered[-2]] = 'calibration'
    partition[ordered[-1]] = 'test'
    groups = defaultdict(list)
    seen = set()
    counts = Counter()
    for row in rows:
        require(row['schema'] == SCHEMA and row['split'] == 'train'
                and row['evaluation_leakage'] is False
                and row['evidence_class'] == 'DEVELOPMENT_ONLY', 'bad source scope')
        p, s = row['provenance'], row['state']
        root, arm, k = p['root'], p['arm'], s['stage']
        require(root in partition and k in (0, 4, 8, 12, 16)
                and set(s) <= STATE_KEYS and len(s['scores']) == len(s['update_norms']) == k,
                'future/invalid stage features')
        require(all(math.isfinite(v) for v in s['scores'] + s['update_norms']), 'nonfinite prefix')
        require(all(p.get(v) for v in ('parent_model_sha256', 'parent_custody_sha256',
                                       'feedback_cohort_id')), 'missing parent custody')
        key = (p['task'], root, arm, p['window'])
        rid = hashlib.sha256(canonical([*key, k])).hexdigest()
        require(row['record_id'] == rid and rid not in seen, 'duplicate/invalid stage ID')
        seen.add(rid)
        groups[key].append(row)
        if row['training_eligible']:
            o = row['outcome']
            pref = row['preference']
            require(arm in VERIFIED_ARMS and arm not in SHAMS and row['behavior_action'] == 'SELECT_BATCH'
                    and row['label_kind'] == 'verified_terminal_preference'
                    and o and o['status'] in ('committed', 'rolled_back')
                    and o['actual_gain'] is not None and pref is not None
                    and s.get('selected_indices'), 'unverified positive supervision')
            require(math.isfinite(o['actual_gain'])
                    and abs(o['cal_before']-o['cal_after']-o['actual_gain']) <= 1e-8,
                    'invalid gain receipt')
            want = 'SELECT_BATCH' if o['actual_gain'] > 0 and o['committed'] else 'ABSTAIN'
            require(pref['chosen'] == want and pref['rejected'] != want, 'incorrect preference')
        else:
            require(row['preference'] is None and row['label_kind'] == 'behavior_only',
                    'unverified preference')
        counts[f"{partition[root]}:{row['label_kind']}"] += 1
    for seq in groups.values():
        seq.sort(key=lambda r: r['state']['stage'])
        sizes = [r['state']['stage'] for r in seq]
        require(sizes in ([0, 4], [0, 4, 8, 12], [0, 4, 8, 12, 16]), 'incomplete stages')
        for a, b in zip(seq[:-1], seq[1:], strict=True):
            sa, sb = a['state'], b['state']
            k = sa['stage']
            require(a['behavior_action'] == f"ACQUIRE_{sb['stage']}"
                    and a['outcome'] is None and a['preference'] is None
                    and not a['training_eligible'] and 'selected_indices' not in sa
                    and 'cpu_to_proposal' not in sa, 'future outcome at acquisition')
            require(sa['scores'] == sb['scores'][:k]
                    and sa['update_norms'] == sb['update_norms'][:k]
                    and sa['cpu_to_stage'] <= sb['cpu_to_stage']
                    and a['provenance'] == b['provenance'], 'prefix/custody mismatch')
        terminal = seq[-1]
        require(terminal['behavior_action'] in ('SELECT_BATCH', 'ABSTAIN')
                and terminal['outcome'] is not None, 'missing terminal outcome')
        if terminal['outcome']['status'] == 'missing_unaudited':
            require(terminal['outcome']['actual_gain'] is None
                    and terminal['preference'] is None, 'imputed abstention outcome')
    return partition, dict(counts)


def run(export, out):
    manifest = json.loads((export / 'manifest.json').read_text())
    require(manifest['schema'] == SCHEMA, 'incorrect manifest schema')
    with gzip.open(export / 'stages_gym_v4.jsonl.gz', 'rb') as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == manifest['files']['stages_gym_v4.jsonl.gz']['canonical_uncompressed_sha256'],
            'stage source hash mismatch')
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) == manifest['counts']['stage_rows'], 'stage count mismatch')
    require(all(r['provenance']['task'] in manifest['tasks'] for r in rows), 'unknown task')
    partition, counts = audit(rows, manifest['roots'])
    out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split in ('train', 'calibration', 'test'):
        for kind in ('behavior_only', 'verified_terminal_preference'):
            name = f'{split}_{kind}.jsonl.gz'
            selected = [r for r in rows if partition[r['provenance']['root']] == split
                        and r['label_kind'] == kind]
            data = b''.join(canonical(r) for r in selected)
            with (out / name).open('wb') as f, gzip.GzipFile(filename='', fileobj=f, mode='wb', mtime=0) as gz:
                gz.write(data)
            outputs[name] = {'records': len(selected), 'sha256': hashlib.sha256(data).hexdigest()}
    result = {'schema': 'datagoblin-gym02-ingress-v4', 'source_stage_sha256': digest,
              'root_partition': partition, 'counts': counts, 'outputs': outputs,
              'scope': 'development root-grouped dataset; only verified terminal preferences admitted to preference supervision; behavior separate'}
    (out / 'ingress_receipt.json').write_bytes(canonical(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.export, args.out), indent=2))


if __name__ == '__main__':
    main()
