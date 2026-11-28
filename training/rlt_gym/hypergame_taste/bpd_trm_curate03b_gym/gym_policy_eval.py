"""BPD-TRM-GYM-01 DataGoblin multi-task paid-acquisition controller + Inky traces.

Reads the strictly TRAIN-only BPD-TRM-GYM-01 DataGoblin V3 export, verifies
canonical LF SHA-256 hashes against ``source_manifest.json``, evaluates
root-held-out paid-acquisition policies per task and pooled over all 5 tasks
on ``C_analytic_full16`` windows (charging measured inference CPU), and emits
multi-task Inky Mini SFT, verifier-grounded preference, and rollback-repair
traces in deterministic ``gzip -n`` form.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL_ARM = 'C_analytic_full16'
GRID = (None, -0.1, -0.05, -0.02, -0.01, -0.005, 0.0)


def _bytes(p: Path) -> bytes:
    """Canonical LF file bytes; falls back to deterministic ``.gz`` twin."""
    if p.is_file():
        return p.read_bytes().replace(b'\r\n', b'\n')
    return gzip.decompress(p.with_name(p.name + '.gz').read_bytes()).replace(b'\r\n', b'\n')


def _sha(p: Path) -> str:
    return hashlib.sha256(_bytes(p)).hexdigest()


def _write_gz(path: Path, raw: bytes) -> str:
    with path.open('wb') as f, gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0) as gz:
        gz.write(raw)
    return hashlib.sha256(raw).hexdigest()


def load_export(export: Path, verify_manifest: bool = True) -> tuple[list[dict], dict, dict]:
    if verify_manifest:
        pinned = json.loads((HERE / 'source_manifest.json').read_text(encoding='utf-8'))
        for name, meta in pinned['files'].items():
            got = _sha(export / name)
            if got != meta['sha256']:
                raise SystemExit(f'hash mismatch for {name}: {got} != {meta["sha256"]}')
    manifest = json.loads((export / 'manifest.json').read_text(encoding='utf-8'))
    unit_costs = json.loads((export / 'unit_costs_gym_v3.json').read_text(encoding='utf-8'))
    raw_lines = _bytes(export / 'trajectories_gym_v3.jsonl').decode('utf-8').splitlines()
    traj = [json.loads(x) for x in raw_lines if x]
    for t in traj:
        if not (t['split'] == 'train' and t['training_eligible'] and not t['evaluation_leakage']):
            raise ValueError('evaluation leakage or non-train record in gym V3 export')
    return traj, manifest, unit_costs


def windows_for(traj: list[dict], *, arm: str = EVAL_ARM, task: str | None = None) -> list[dict]:
    out = []
    for t in traj:
        prov = t['provenance']
        if prov['arm'] != arm:
            continue
        if task is not None and prov['task'] != task:
            continue
        scores = t.get('scores_prefix') or []
        unorms = t.get('update_norms_prefix') or []
        if len(scores) != 16 or len(unorms) != 16:
            continue
        j = t.get('joint_outcome') or {}
        committed = bool(t['committed'])
        gain = j.get('actual_gain') or 0.0
        out.append({
            'task': prov['task'],
            'root': prov['root'],
            'window': prov['window'],
            'scores': scores,
            'update_norms': unorms,
            'loss0': t['state'].get('feedback_loss0') or 0.0,
            'committed': committed,
            'pos_gain': max(gain, 0.0) if committed else 0.0,
            'gain': gain,
            'cpu': t['cost']['window_cpu_seconds'],
        })
    return out


def stage_features(w: dict, k: int) -> np.ndarray:
    s = w['scores'][:k]
    return np.array([
        max(s),
        sum(x > 0 for x in s),
        sum(max(x, 0.0) for x in s),
        sum(s) / k,
        max(w['update_norms'][:k]),
        w['loss0'],
    ], dtype=np.float64)


def evaluate(ws: list[dict], decide, unit_costs: dict) -> dict:
    acq = att = miss = acc = 0
    gain = missed = cpu_work = 0.0
    t0 = time.process_time()
    decisions = [decide(w) for w in ws]
    inf = time.process_time() - t0
    for w, (a, attempt) in zip(ws, decisions, strict=True):
        uc = unit_costs[w['task']]
        acq += a
        att += int(attempt)
        acc += int(w['committed'])
        gain += w['pos_gain']
        cpu_work += uc['per_window_base'] + a * uc['per_candidate'] + int(attempt) * uc['per_attempt']
        if w['committed'] and not attempt:
            miss += 1
            missed += w['pos_gain']
    return {
        'windows': len(ws),
        'candidates_acquired': acq,
        'attempts': att,
        'committed_in_trajectory': acc,
        'committed_missed': miss,
        'positive_gain': gain,
        'missed_positive_gain': missed,
        'retained_fraction': 1.0 - missed / gain if gain > 0 else 1.0,
        'controller_inference_cpu_seconds': inf,
        'estimated_cpu_seconds': cpu_work + inf,
    }


def threshold_policy(t: tuple[float | None, ...]):
    def decide(w: dict) -> tuple[int, bool]:
        for j, k in enumerate((4, 8, 12)):
            if t[j] is not None and max(w['scores'][:k]) <= t[j]:
                return k, False
        return 16, max(w['scores']) > 0
    return decide


def _feasible(ws: list[dict], decide, unit_costs: dict, roots: list[int]) -> dict | None:
    pooled = evaluate(ws, decide, unit_costs)
    if pooled['retained_fraction'] < 0.995:
        return None
    for r in roots:
        sub = [w for w in ws if w['root'] == r]
        if sub and evaluate(sub, decide, unit_costs)['retained_fraction'] < 0.98:
            return None
    return pooled


def fit_thresholds(train: list[dict], unit_costs: dict, roots: list[int]) -> tuple:
    best = None
    for t in itertools.product(GRID, repeat=3):
        f = _feasible(train, threshold_policy(t), unit_costs, roots)
        if f is None:
            continue
        key = (f['estimated_cpu_seconds'] - f['controller_inference_cpu_seconds'], f['attempts'], str(t))
        if best is None or key < best[0]:
            best = (key, t)
    return best[1] if best else (None, None, None)


def _logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30.0, 30.0)))
        g = Z.T @ (p - y) + l2 * np.r_[0.0, w[1:]]
        H = (Z * (p * (1.0 - p))[:, None]).T @ Z + l2 * np.diag(np.r_[0.0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(H + 1e-9 * np.eye(len(w)), g)
    return w, mu, sd


def fit_learned(train: list[dict], unit_costs: dict, roots: list[int], *, shuffle_seed: int | None = None):
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

    # Precompute probabilities once on train for fast grid search over tau
    train_probs: dict[int, list[float]] = {}
    for k in (4, 8, 12):
        m = models[k]
        if m is None:
            train_probs[k] = [1.0] * len(train)
        else:
            wt, mu, sd = m
            X = np.stack([stage_features(w, k) for w in train])
            Z = np.c_[np.ones(len(X)), (X - mu) / sd]
            train_probs[k] = (1.0 / (1.0 + np.exp(-np.clip(Z @ wt, -30.0, 30.0)))).tolist()

    taus = (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3)
    best = None
    by_root_idx = {r: [i for i, w in enumerate(train) if w['root'] == r] for r in roots}
    p4, p8, p12 = train_probs[4], train_probs[8], train_probs[12]

    for tau in itertools.product(taus, repeat=3):
        t4, t8, t12 = tau
        gain_all = missed_all = cpu_all = 0.0
        att_all = 0
        root_gain = dict.fromkeys(roots, 0.0)
        root_miss = dict.fromkeys(roots, 0.0)
        for i, w in enumerate(train):
            if p4[i] < t4:
                a, att = 4, False
            elif p8[i] < t8:
                a, att = 8, False
            elif p12[i] < t12:
                a, att = 12, False
            else:
                a, att = 16, max(w['scores']) > 0
            uc = unit_costs[w['task']]
            cpu_all += uc['per_window_base'] + a * uc['per_candidate'] + int(att) * uc['per_attempt']
            att_all += int(att)
            g = w['pos_gain']
            gain_all += g
            root_gain[w['root']] += g
            if w['committed'] and not att:
                missed_all += g
                root_miss[w['root']] += g
        ret_all = 1.0 - missed_all / gain_all if gain_all > 0 else 1.0
        if ret_all < 0.995:
            continue
        if any(
            by_root_idx[r] and (1.0 - root_miss[r] / root_gain[r] if root_gain[r] > 0 else 1.0) < 0.98
            for r in roots
        ):
            continue
        key = (cpu_all, att_all, tau)
        if best is None or key < best[0]:
            best = (key, tau)
    tau = best[1] if best else (0.0, 0.0, 0.0)

    def prob_one(w: dict, k: int) -> float:
        m = models[k]
        if m is None:
            return 1.0
        wt, mu, sd = m
        z = float(np.r_[1.0, (stage_features(w, k) - mu) / sd] @ wt)
        return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))

    def decide(w: dict) -> tuple[int, bool]:
        for j, k in enumerate((4, 8, 12)):
            if prob_one(w, k) < tau[j]:
                return k, False
        return 16, max(w['scores']) > 0

    info = {
        'tau': list(tau),
        'coefficients': {str(k): (None if m is None else [float(x) for x in m[0]]) for k, m in models.items()},
    }
    return decide, info


def evaluate_subset(ws: list[dict], unit_costs: dict) -> dict:
    roots = sorted({w['root'] for w in ws})
    fixed = {
        'constant_ACQUIRE_16': lambda w: (16, True),
        'constant_ABSTAIN_at_4': lambda w: (4, False),
        'full16_gate': threshold_policy((None, None, None)),
        'early4_gate': threshold_policy((0.0, None, None)),
        'early8_gate': threshold_policy((None, 0.0, None)),
        'early12_gate': threshold_policy((None, None, 0.0)),
    }
    folds = []
    pooled: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in roots:
        train = [w for w in ws if w['root'] != r]
        test = [w for w in ws if w['root'] == r]
        tr_roots = [x for x in roots if x != r]
        t = fit_thresholds(train, unit_costs, tr_roots)
        learned, linfo = fit_learned(train, unit_costs, tr_roots)
        shuffled, sinfo = fit_learned(train, unit_costs, tr_roots, shuffle_seed=r)
        pol = dict(fixed)
        pol['adaptive_threshold'] = threshold_policy(t)
        pol['learned_logistic'] = learned
        pol['learned_shuffled_labels'] = shuffled
        res = {name: evaluate(test, fn, unit_costs) for name, fn in pol.items()}
        folds.append({
            'heldout_root': r,
            'adaptive_thresholds': list(t),
            'learned': linfo,
            'shuffled': sinfo,
            'test': res,
        })
        for name, m in res.items():
            for k, v in m.items():
                if k != 'retained_fraction':
                    pooled[name][k] += v

    summary = {}
    for name, m in pooled.items():
        d = dict(m)
        d['retained_fraction'] = 1.0 - d['missed_positive_gain'] / d['positive_gain'] if d['positive_gain'] > 0 else 1.0
        d['meets_99_5_retention'] = d['retained_fraction'] >= 0.995
        summary[name] = d
    base = summary['constant_ACQUIRE_16']['estimated_cpu_seconds']
    for d in summary.values():
        d['estimated_cpu_vs_always'] = d['estimated_cpu_seconds'] / max(base, 1e-12)
    return {'windows': len(ws), 'roots': roots, 'pooled_heldout': summary, 'folds': folds}


def _prompt(t: dict) -> str:
    prov = t['provenance']
    st = t['state']
    stages = ', '.join(
        f"s{x['stage']}:max={x['max_score']:.4g},pos={x['n_positive']}"
        for x in (st.get('stages') or [])
    )
    fb = st.get('feedback_loss0')
    fb_str = 'none' if fb is None else f'{fb:.4f}'
    return (
        f"BPD-TRM-GYM task={prov['task']} window={prov['window']} "
        f"feedback_loss={fb_str} stages[{stages}] "
        f"cpu_used={st['charged_cpu_before']:.1f}s. Choose actions."
    )


def write_inky_gym_traces(traj: list[dict], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_run: dict[tuple, list[dict]] = defaultdict(list)
    for t in traj:
        p = t['provenance']
        by_run[(p['task'], p['root'], p['arm'])].append(t)

    sft_lines: list[str] = []
    pref_lines: list[str] = []
    rep_lines: list[str] = []

    for (task, root, arm), rows in sorted(by_run.items()):
        rows.sort(key=lambda r: r['provenance']['window'])
        for i, t in enumerate(rows):
            prompt = _prompt(t)
            j = t.get('joint_outcome') or {}
            status = j.get('status')
            sft_lines.append(json.dumps({
                'record_id': t['record_id'],
                'task': task,
                'root': root,
                'arm': arm,
                'prompt': prompt,
                'completion': ' '.join(t.get('actions') or []),
                'label_kind': 'policy_imitation',
                'verified': status in ('committed', 'rolled_back', 'audited_counterfactual'),
            }, sort_keys=True) + '\n')

            if arm == EVAL_ARM and status in ('committed', 'rolled_back'):
                good = status == 'committed' and (j.get('actual_gain') or 0.0) > 0.0
                chosen, rejected = ('SELECT_BATCH', 'ABSTAIN') if good else ('ABSTAIN', 'SELECT_BATCH')
                pref_lines.append(json.dumps({
                    'record_id': t['record_id'],
                    'task': task,
                    'root': root,
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected,
                    'verifier': {
                        'test': 'matched-RNG TRAIN feedback acceptance',
                        'cal_before': j.get('cal_before'),
                        'cal_after': j.get('cal_after'),
                        'actual_gain': j.get('actual_gain'),
                    },
                    'label_kind': 'joint_minibatch_outcome',
                }, sort_keys=True) + '\n')

            if status == 'rolled_back' and i + 1 < len(rows):
                nxt = rows[i + 1]
                rep_lines.append(json.dumps({
                    'record_id': t['record_id'],
                    'task': task,
                    'root': root,
                    'arm': arm,
                    'trace': [
                        {
                            'actions': (t.get('actions') or []) + ['ROLLBACK'],
                            'verifier': {'cal_before': j.get('cal_before'), 'cal_after': j.get('cal_after')},
                        },
                        {
                            'actions': nxt.get('actions') or [],
                            'outcome': (nxt.get('joint_outcome') or {}).get('status'),
                        },
                    ],
                    'label_kind': 'verifier_grounded_repair',
                }, sort_keys=True) + '\n')

    files = {
        'inky_gym_sft.jsonl': _write_gz(out_dir / 'inky_gym_sft.jsonl.gz', ''.join(sft_lines).encode('utf-8')),
        'inky_gym_preferences.jsonl': _write_gz(
            out_dir / 'inky_gym_preferences.jsonl.gz', ''.join(pref_lines).encode('utf-8'),
        ),
        'inky_gym_repair_traces.jsonl': _write_gz(
            out_dir / 'inky_gym_repair_traces.jsonl.gz', ''.join(rep_lines).encode('utf-8'),
        ),
    }
    manifest = {
        'schema': 'datagoblin-bpd-trm-gym01-inky-v3',
        'counts': {'sft': len(sft_lines), 'preferences': len(pref_lines), 'repairs': len(rep_lines)},
        'files': files,
        'note': 'TRAIN-only multi-task gym prompts; horizon EVAL outcomes excluded.',
    }
    (out_dir / 'inky_manifest.json').write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + '\n', encoding='utf-8', newline='\n',
    )
    return manifest


def run(export: Path, out: Path, *, traces_dir: Path | None = None, verify_manifest: bool = True) -> dict:
    traj, manifest, unit_costs = load_export(export, verify_manifest=verify_manifest)
    all_ws = windows_for(traj)
    tasks = sorted({w['task'] for w in all_ws})
    per_task = {task: evaluate_subset(windows_for(traj, task=task), unit_costs) for task in tasks}
    pooled = evaluate_subset(all_ws, unit_costs)
    receipt = {
        'schema': 'datagoblin-bpd-trm-gym01-policy-v1',
        'evidence_class': 'DEVELOPMENT_ONLY',
        'export_manifest_sha256': _sha(export / 'manifest.json'),
        'eval_arm': EVAL_ARM,
        'tasks': tasks,
        'unit_costs_cpu_seconds': unit_costs,
        'scope': 'per-window replay on recorded C_analytic_full16 parents across 5 gym tasks',
        'pooled_all_tasks': pooled,
        'per_task': per_task,
        'export_counts': manifest['counts'],
    }
    out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    if traces_dir is not None:
        write_inky_gym_traces(traj, traces_dir)
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('export', type=Path)
    ap.add_argument('--out', type=Path, default=HERE / 'gym_policy_receipt.json')
    ap.add_argument('--traces', type=Path, default=None)
    ap.add_argument('--skip-manifest-check', action='store_true')
    args = ap.parse_args()
    r = run(args.export, args.out, traces_dir=args.traces, verify_manifest=not args.skip_manifest_check)
    cols = ('candidates_acquired', 'retained_fraction', 'estimated_cpu_vs_always')
    ph = r['pooled_all_tasks']['pooled_heldout']
    print(json.dumps({k: {c: ph[k][c] for c in cols} for k in ph}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
