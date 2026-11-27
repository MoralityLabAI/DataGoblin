"""CURATE-03B paid-acquisition controller: root-held-out offline policy evaluation + Inky traces.

Input: the BPD CURATE-03B DataGoblin V2 export directory (TRAIN-only). Hashes are
checked against source_manifest.json before anything is read.

Decision process per window: acquire candidates in the frozen prospective order in
stages of 4; after stages 4, 8 and 12 choose ACQUIRE_next or ABSTAIN; after 16
choose SELECT_BATCH (attempt) or ABSTAIN. Controller value:
    Q(s, a) = E[dU + gamma V(s') | s, a] - lambda C(s, a)
with dU the committed positive feedback gain and C the measured CPU of the
acquisitions, attempt and controller inference.

Evaluation set: D_analytic_mech windows (no gate, so every window carries all
16 scores and its attempted outcome). Replay is per-window on recorded parents;
skipping a window would have changed later checkpoints, which this cannot see.
Policies: constant ACQUIRE_16, constant ABSTAIN, fixed zero gates (full-16,
early-4/8/12), root-held-out adaptive threshold grid, learned logistic stage
model, and the learned model fit on shuffled labels. Fitting constraint
(all fitted policies): pooled retained positive committed gain >= 99.5% and
every training root >= 98%; then minimize estimated CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL_ARM = 'D_analytic_mech'
GRID = (None, -0.1, -0.05, -0.02, -0.01, -0.005, 0.0)
FEATURES = ('max_score', 'n_positive', 'sum_positive', 'mean_score', 'max_update_norm', 'delay')


def _bytes(p: Path) -> bytes:
    """Canonical LF file bytes; a missing ``x.jsonl`` is read from its deterministic ``x.jsonl.gz``."""
    if p.is_file():
        return p.read_bytes().replace(b'\r\n', b'\n')
    import gzip
    return gzip.decompress(p.with_name(p.name + '.gz').read_bytes()).replace(b'\r\n', b'\n')


def _sha(p: Path) -> str:
    return hashlib.sha256(_bytes(p)).hexdigest()


def load(export: Path) -> tuple[list, dict]:
    pinned = json.loads((HERE / 'source_manifest.json').read_text(encoding='utf-8'))
    for name, meta in pinned['files'].items():
        got = _sha(export / name)
        if got != meta['sha256']:
            raise SystemExit(f'hash mismatch for {name}: {got} != {meta["sha256"]}')
    manifest = json.loads((export / 'manifest.json').read_text(encoding='utf-8'))
    traj = [json.loads(x) for x in _bytes(export / 'trajectories_v2.jsonl').decode('utf-8').splitlines() if x]
    for t in traj:
        assert t['split'] == 'train' and t['training_eligible'] and not t['evaluation_leakage']
    return traj, manifest


def windows_for(traj, arm=EVAL_ARM):
    out = []
    for t in traj:
        if t['provenance']['arm'] != arm:
            continue
        pool = sorted(t['candidate_pool'], key=lambda c: c['acquisition_position'])
        if not all(c['acquired'] for c in pool) or len(pool) != 16:
            continue
        s = [c['features_after_acquisition']['predicted_gain'] for c in pool]
        un = [c['features_after_acquisition']['update_norm'] for c in pool]
        j = t['joint_outcome'] or {}
        committed = bool(t['committed'])
        gain = j.get('actual_gain') or 0.0
        out.append({'root': t['provenance']['root'], 'window': t['provenance']['window'], 'scores': s,
                    'update_norms': un, 'delay': t['state']['delay'], 'committed': committed,
                    'pos_gain': max(gain, 0.0) if committed else 0.0, 'gain': gain,
                    'cpu': t['cost']['window_cpu_seconds']})
    return out


def stage_features(w, k):
    s = w['scores'][:k]
    return np.array([max(s), sum(x > 0 for x in s), sum(max(x, 0) for x in s), sum(s) / k,
                     max(w['update_norms'][:k]), w['delay']], dtype=np.float64)


class Costs:
    def __init__(self, acq: float, attempt: float, base: float):
        self.acq, self.attempt, self.base = acq, attempt, base


def evaluate(ws, decide, costs: Costs) -> dict:
    """decide(w) -> (acquired, attempt). Inference CPU is measured, not assumed."""
    acq = att = miss = acc = 0
    gain = missed = 0.0
    t0 = time.process_time()
    decisions = [decide(w) for w in ws]
    inf = time.process_time() - t0
    for w, (a, attempt) in zip(ws, decisions, strict=True):
        acq += a
        att += int(attempt)
        acc += int(w['committed'])
        gain += w['pos_gain']
        if w['committed'] and not attempt:
            miss += 1
            missed += w['pos_gain']
    cpu = len(ws) * costs.base + acq * costs.acq + att * costs.attempt + inf
    return {'windows': len(ws), 'candidates_acquired': acq, 'attempts': att, 'committed_in_trajectory': acc,
            'committed_missed': miss, 'positive_gain': gain, 'missed_positive_gain': missed,
            'retained_fraction': 1 - missed / gain if gain else 1.0,
            'controller_inference_cpu_seconds': inf, 'estimated_cpu_seconds': cpu}


def threshold_policy(t):
    def decide(w):
        for j, k in enumerate((4, 8, 12)):
            if t[j] is not None and max(w['scores'][:k]) <= t[j]:
                return k, False
        return 16, max(w['scores']) > 0
    return decide


def _feasible(ws, decide, costs, roots):
    pooled = evaluate(ws, decide, costs)
    if pooled['retained_fraction'] < 0.995:
        return None
    for r in roots:
        sub = [w for w in ws if w['root'] == r]
        if evaluate(sub, decide, costs)['retained_fraction'] < 0.98:
            return None
    return pooled


def fit_thresholds(train, costs, roots):
    best = None
    for t in itertools.product(GRID, repeat=3):
        f = _feasible(train, threshold_policy(t), costs, roots)
        if f is None:
            continue
        key = (f['estimated_cpu_seconds'] - f['controller_inference_cpu_seconds'], f['attempts'], str(t))
        if best is None or key < best[0]:
            best = (key, t)
    return best[1] if best else (None, None, None)


def _logistic(X, y, l2=1.0, iters=50):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(H + 1e-9 * np.eye(len(w)), g)
    return w, mu, sd


def fit_learned(train, costs, roots, *, shuffle_seed=None):
    models = {}
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    for k in (4, 8, 12):
        X = np.stack([stage_features(w, k) for w in train])
        y = np.array([float(w['committed'] and w['pos_gain'] > 0) for w in train])
        if rng is not None:
            y = rng.permutation(y)
        if y.min() == y.max():
            models[k] = None
            continue
        models[k] = _logistic(X, y)

    def prob(w, k):
        m = models[k]
        if m is None:
            return 1.0
        wt, mu, sd = m
        z = np.r_[1.0, (stage_features(w, k) - mu) / sd] @ wt
        return 1 / (1 + math.exp(-z))

    taus = (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3)
    best = None
    for tau in itertools.product(taus, repeat=3):
        def decide(w, tau=tau):
            for j, k in enumerate((4, 8, 12)):
                if prob(w, k) < tau[j]:
                    return k, False
            return 16, max(w['scores']) > 0
        f = _feasible(train, decide, costs, roots)
        if f is None:
            continue
        key = (f['estimated_cpu_seconds'] - f['controller_inference_cpu_seconds'], f['attempts'], tau)
        if best is None or key < best[0]:
            best = (key, tau)
    tau = best[1] if best else (0.0, 0.0, 0.0)

    def decide(w):
        for j, k in enumerate((4, 8, 12)):
            if prob(w, k) < tau[j]:
                return k, False
        return 16, max(w['scores']) > 0
    return decide, {'tau': tau, 'coefficients': {k: (None if m is None else [float(x) for x in m[0]])
                                                 for k, m in models.items()}}


def run(export: Path, out: Path) -> dict:
    traj, _manifest = load(export)
    ws = windows_for(traj)
    roots = sorted({w['root'] for w in ws})
    # Measured unit costs from the same export (D arm): per-candidate and per-attempt CPU.
    d_rows = [t for t in traj if t['provenance']['arm'] == EVAL_ARM]
    mean_window = float(np.mean([t['cost']['window_cpu_seconds'] for t in d_rows]))
    costs = Costs(acq=mean_window * 0.6 / 16, attempt=mean_window * 0.2, base=mean_window * 0.2)
    unit_path = export / 'unit_costs.json'
    unit = json.loads(unit_path.read_text(encoding='utf-8')) if unit_path.is_file() else None
    if unit:
        costs = Costs(acq=unit['per_candidate'], attempt=unit['per_attempt'], base=unit['per_window_base'])
    fixed = {'constant_ACQUIRE_16': lambda w: (16, True), 'constant_ABSTAIN_at_4': lambda w: (4, False),
             'full16_gate': threshold_policy((None, None, None)), 'early4_gate': threshold_policy((0.0, None, None)),
             'early8_gate': threshold_policy((None, 0.0, None)), 'early12_gate': threshold_policy((None, None, 0.0))}
    folds = []
    pooled = defaultdict(lambda: defaultdict(float))
    for r in roots:
        train = [w for w in ws if w['root'] != r]
        test = [w for w in ws if w['root'] == r]
        tr_roots = [x for x in roots if x != r]
        t = fit_thresholds(train, costs, tr_roots)
        learned, linfo = fit_learned(train, costs, tr_roots)
        shuffled, sinfo = fit_learned(train, costs, tr_roots, shuffle_seed=r)
        pol = dict(fixed)
        pol['adaptive_threshold'] = threshold_policy(t)
        pol['learned_logistic'] = learned
        pol['learned_shuffled_labels'] = shuffled
        res = {name: evaluate(test, f, costs) for name, f in pol.items()}
        folds.append({'heldout_root': r, 'adaptive_thresholds': t, 'learned': linfo, 'shuffled': sinfo, 'test': res})
        for name, m in res.items():
            for k, v in m.items():
                if k != 'retained_fraction':
                    pooled[name][k] += v
    summary = {}
    for name, m in pooled.items():
        m = dict(m)
        m['retained_fraction'] = 1 - m['missed_positive_gain'] / m['positive_gain'] if m['positive_gain'] else 1.0
        m['meets_99_5_retention'] = m['retained_fraction'] >= 0.995
        summary[name] = m
    base = summary['constant_ACQUIRE_16']['estimated_cpu_seconds']
    for m in summary.values():
        m['estimated_cpu_vs_always'] = m['estimated_cpu_seconds'] / base
    scope_note = (
        'per-window replay on recorded parents; '
        'changed-trajectory effects of skipping are not observed'
    )
    receipt = {'schema': 'datagoblin-bpd-curate03b-acquisition-policy-v1', 'evidence_class': 'DEVELOPMENT_ONLY',
               'export_manifest_sha256': _sha(export / 'manifest.json'), 'eval_arm': EVAL_ARM,
               'windows': len(ws), 'roots': roots,
               'unit_costs_cpu_seconds': {'per_candidate': costs.acq, 'per_attempt': costs.attempt,
                                          'per_window_base': costs.base,
                                          'source': 'unit_costs.json' if unit else 'window-mean split'},
               'scope': scope_note,
               'pooled_heldout': summary, 'folds': folds}
    out.write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str) + '\n', encoding='utf-8', newline='\n')
    return receipt


def inky_traces(export: Path, out_dir: Path) -> dict:
    """SFT, preference and verifier-grounded repair traces for later Inky Mini training."""
    traj, _ = load(export)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = defaultdict(int)
    by_run = defaultdict(list)
    for t in traj:
        by_run[(t['provenance']['root'], t['provenance']['arm'])].append(t)
    with (out_dir / 'inky_sft.jsonl').open('w', encoding='utf-8', newline='\n') as sft, \
            (out_dir / 'inky_preferences.jsonl').open('w', encoding='utf-8', newline='\n') as pref, \
            (out_dir / 'inky_repair_traces.jsonl').open('w', encoding='utf-8', newline='\n') as rep:
        for (root, arm), rows in sorted(by_run.items()):
            rows.sort(key=lambda t: t['provenance']['window'])
            for i, t in enumerate(rows):
                if t['decision'] == 'REINVEST_CPU':
                    continue
                prompt = _prompt(t)
                j = t['joint_outcome'] or {}
                sft.write(json.dumps({'record_id': t['record_id'], 'root': root, 'arm': arm, 'prompt': prompt,
                                      'completion': ' '.join(t['actions']),
                                      'label_kind': 'policy_imitation', 'verified': j.get('status') in (
                                          'committed', 'rolled_back', 'audited_counterfactual')}) + '\n')
                n['sft'] += 1
                if arm == EVAL_ARM and j.get('status') in ('committed', 'rolled_back'):
                    good = j['status'] == 'committed' and (j.get('actual_gain') or 0) > 0
                    chosen, rejected = (('SELECT_BATCH', 'ABSTAIN') if good else ('ABSTAIN', 'SELECT_BATCH'))
                    pref.write(json.dumps({'record_id': t['record_id'], 'root': root, 'prompt': prompt,
                                           'chosen': chosen, 'rejected': rejected,
                                           'verifier': {'test': 'matched-RNG TRAIN feedback acceptance',
                                                        'cal_before': j.get('cal_before'),
                                                        'cal_after': j.get('cal_after'),
                                                        'actual_gain': j.get('actual_gain')},
                                           'label_kind': 'joint_minibatch_outcome'}) + '\n')
                    n['preferences'] += 1
                if j.get('status') == 'rolled_back' and i + 1 < len(rows):
                    nxt = rows[i + 1]
                    rep.write(json.dumps({'record_id': t['record_id'], 'root': root, 'arm': arm,
                                          'trace': [{'actions': t['actions'] + ['ROLLBACK'],
                                                     'verifier': {'cal_before': j.get('cal_before'),
                                                                  'cal_after': j.get('cal_after')}},
                                                    {'actions': nxt['actions'],
                                                     'outcome': (nxt['joint_outcome'] or {}).get('status')}],
                                          'label_kind': 'verifier_grounded_repair'}) + '\n')
                    n['repairs'] += 1
    files = {p.name: _sha(p) for p in sorted(out_dir.glob('inky_*.jsonl'))}
    note = (
        'state prompts contain TRAIN-only features available before the decision; '
        'horizon outcomes are excluded'
    )
    m = {'counts': dict(n), 'files': files, 'note': note}
    manifest_text = json.dumps(m, indent=1, sort_keys=True) + '\n'
    (out_dir / 'inky_manifest.json').write_text(manifest_text, encoding='utf-8', newline='\n')
    return m


def _prompt(t) -> str:
    st = t['state']
    stage_parts = [
        f"s{x['stage']}:max={x['max_score']:.4g},pos={x['n_positive']}"
        for x in (st.get('stages') or [])
    ]
    stages = ', '.join(stage_parts)
    fb = (st.get('feedback') or {}).get('loss0')
    return (f"BPD-TRM curation window {t['provenance']['window']} delay={st['delay']} "
            f"feedback_loss={fb if fb is None else round(fb, 4)} stages[{stages}] "
            f"budget_used={st['cumulative_cpu_seconds_before']:.1f}s. Choose actions.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('export', type=Path)
    ap.add_argument('--out', type=Path, default=HERE / 'acquisition_policy_receipt.json')
    ap.add_argument('--traces', type=Path, default=None)
    a = ap.parse_args()
    r = run(a.export, a.out)
    cols = ('candidates_acquired', 'retained_fraction', 'estimated_cpu_vs_always')
    summary = {k: {kk: r['pooled_heldout'][k][kk] for kk in cols} for k in r['pooled_heldout']}
    print(json.dumps(summary, indent=1))
    if a.traces:
        print(json.dumps(inky_traces(a.export, a.traces)['counts']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
