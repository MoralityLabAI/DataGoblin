"""RV22-Z: native curriculum audit and isolated synthetic decision-value probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from datagoblin.research_v2_2._exact_engine import binary_key
from datagoblin.research_v2_2.curriculum_audit import SPLITS, UPSTREAM, audit_directory
from datagoblin.research_v2_2.exact_research import ExactResearch, action_values, policy_value, validate_world

TRAIN, CAL, FRESH, FOUNDATION = 2026770000, 2026780000, 2026790000, 2026800000
SEEDS = (29, 31, 37)
NFEATURES = 16


def state(d, indices, budget, skip):
    return dict(hypotheses=[d['hypotheses'][i] for i in indices],
                tests=[dict(id=t['id'], cost=t['cost'], bits=[t['bits'][i] for i in indices])
                       for t in d['tests'] if t['id'] != skip],
                budget=budget, correct_reward=d['correct_reward'], wrong_reward=d['wrong_reward'])


def mass(d):
    w = sum(h['weight'] for h in d['hypotheses'])
    a = sum(h['weight'] for h in d['hypotheses'] if h['class'] == 'A')
    return w, a


def stop(d):
    w, a = mass(d)
    c, r = d['correct_reward'], d['wrong_reward']
    return max((0., 'defer'), ((a*c+(w-a)*r)/w, 'A'), (((w-a)*c+a*r)/w, 'B'))


def candidates(d):
    w, a = mass(d)
    b = d['budget']
    s, _ = stop(d)
    scale = max(1, d['correct_reward'], abs(d['wrong_reward']))
    legal = []
    for t in sorted(d['tests'], key=lambda t: t['id']):
        cost = t['cost']
        if cost > b:
            continue
        z = [i for i, bit in enumerate(t['bits']) if not bit]
        o = [i for i, bit in enumerate(t['bits']) if bit]
        if z and o:
            children = state(d, z, b-cost, t['id']), state(d, o, b-cost, t['id'])
            legal.append((t, children))
    for t, children in legal:
        z, o = children
        wz, az = mass(z)
        wo, ao = mass(o)
        p, pz, po = a/w, az/wz, ao/wo
        q1 = (wz*stop(z)[0]+wo*stop(o)[0])/w-t['cost']
        features = (1., b/12, t['cost']/12, (b-t['cost'])/12, len(d['hypotheses'])/64,
                    len(d['tests'])/8, len(legal)/8, abs(p-.5)*2, 2*min(wz,wo)/w, abs(pz-po),
                    (wz*abs(pz-.5)+wo*abs(po-.5))*2/w, s/scale, q1/scale, (q1-s)/scale,
                    d['correct_reward']/scale, d['wrong_reward']/scale)
        yield t['id'], children, q1, features


def examples(worlds):
    x, y = [], []
    for d in worlds:
        # Targets are recomputed from fresh synthetic worlds, never imported referee labels.
        states = [d]
        states.extend(s for _, children, _, _ in candidates(d) for s in children)
        for sub in states:
            q = action_values(sub)
            scale = max(1, sub['correct_reward'], abs(sub['wrong_reward']))
            for name, _, q1, f in candidates(sub):
                label = q['test:'+name]
                if abs(float(label['one_step'])-q1) > 1e-9:
                    raise ValueError('analytic one-step mismatch')
                x.append(f)
                y.append(float(label['value']-label['one_step'])/scale)
    return np.asarray(x), np.asarray(y)


class Head:
    """Small tied tanh recurrence with explicit backpropagation; no actor endpoint."""
    def __init__(self, seed, width, depth):
        rng = np.random.default_rng(seed)
        self.depth = depth
        self.params = [rng.normal(0,.2,(NFEATURES,width)), np.zeros(width),
                       rng.normal(0,.08,(width,width)), rng.normal(0,.2,width), np.zeros(1)]

    def forward(self, x):
        w,b,r,o,c = self.params
        h = np.zeros((len(x),len(b)))
        states = [h]
        proj = x@w+b
        for step in range(self.depth):
            h = np.tanh(proj if step == 0 else proj+h@r)
            states.append(h)
        return h@o+c, states

    def predict(self, x):
        return self.forward(x)[0]

    def fit(self, x, y):
        rng = np.random.default_rng(991)
        p = self.params
        m, v = [np.zeros_like(a) for a in p], [np.zeros_like(a) for a in p]
        step = 0
        for _ in range(40):
            for batch in np.array_split(rng.permutation(len(x)), max(1,len(x)//128)):
                xb,yb = x[batch],y[batch]
                pred,h = self.forward(xb)
                e = 2*(pred-yb)/len(batch)
                g = [np.zeros_like(a) for a in p]
                g[3],g[4][0] = h[-1].T@e,e.sum()
                dh = e[:,None]*p[3]
                for j in range(self.depth,0,-1):
                    a = dh*(1-h[j]**2)
                    g[0] += xb.T@a
                    g[1] += a.sum(axis=0)
                    g[2] += h[j-1].T@a
                    dh = a@p[2].T
                step += 1
                for i in range(len(p)):
                    m[i] = .9*m[i]+.1*g[i]
                    v[i] = .999*v[i]+.001*g[i]**2
                    p[i] -= .003*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)
        return self


def tree(d, mode, model=None, tau=0.):
    """Public-only actor policy. Exact DP is never called here."""
    calls = 0
    def visit(sub):
        nonlocal calls
        best, decision = stop(sub)
        if mode == 'stop':
            return {'decision': decision}
        opts = list(candidates(sub))
        if not opts:
            return {'decision': decision}
        scale = max(1, sub['correct_reward'], abs(sub['wrong_reward']))
        scores = np.asarray([q for _,_,q,_ in opts])
        if mode == 'two_step':
            w,_ = mass(sub)
            for i,(_,children,_,_) in enumerate(opts):
                scores[i] += sum(mass(ch)[0]/w*(max([stop(ch)[0], *(q for _,_,q,_ in candidates(ch))])
                                                -stop(ch)[0]) for ch in children)
        elif model is not None:
            calls += len(opts)
            scores += scale*np.maximum(model.predict(np.asarray([f for _,_,_,f in opts])),0.)
        j = int(np.argmax(scores))
        if scores[j] <= best+scale*tau+1e-10:
            return {'decision': decision}
        name, (z,o), _, _ = opts[j]
        return {'test':name, 'zero':visit(z), 'one':visit(o)}
    return visit(d),calls


def evaluate(worlds, mode, model=None, tau=0.):
    values, calls = [], []
    for d in worlds:
        p,n = tree(d,mode,model,tau)
        values.append(float(policy_value(d,p)))
        calls.append(n)
    return np.asarray(values),float(np.mean(calls))


def panel(generator, seed, count, difficulties):
    return [validate_world(generator('adaptive_evidence',seed,i,difficulties[i%len(difficulties)]))
            for i in range(count)]


def run_probe(panels, audit_keys=()):
    # Exact keys of the native reference export are prohibited from all new panels.
    reference = set(audit_keys)
    seen = set()
    hashes = {}
    for name, worlds in panels.items():
        keys = [binary_key(d)[0] for d in worlds]
        distinct = set(keys)
        if len(distinct) != len(worlds) or seen.intersection(distinct):
            raise ValueError('probe split overlap')
        if name == 'native_eval':
            if not distinct <= reference:
                raise ValueError('native evaluation binding')
        elif reference.intersection(distinct):
            raise ValueError('reference/probe overlap')
        seen.update(distinct)
        hashes[name] = hashlib.sha256(b''.join(sorted(hashlib.sha256(k).digest() for k in keys))).hexdigest()
    x,y = examples(panels['train'])
    if not len(y) or not np.all(np.isfinite(y)) or np.any(y < -1e-9):
        raise ValueError('invalid synthetic labels')
    outputs, thresholds = defaultdict(dict), defaultdict(list)
    work = defaultdict(dict)
    models = []
    for mode,width,depth in (('ff',29,1),('trm',12,3)):
        for seed in SEEDS:
            model = Head(seed,width,depth).fit(x,y)
            tau = max((evaluate(panels['cal'],mode,model,t)[0].mean(),-t,t) for t in (0.,.025,.05,.1))[2]
            thresholds[mode].append(tau)
            models.append((mode,model,tau))
    # Freeze all fits and calibration decisions before evaluating any held-out policy.
    for name, worlds in panels.items():
        if name in ('train','cal'):
            continue
        for mode in ('stop','one_step','two_step'):
            values,calls = evaluate(worlds,mode)
            outputs[name][mode] = [values]
            work[name][mode] = [calls]
        for mode,model,tau in models:
            values,calls = evaluate(worlds,mode,model,tau)
            outputs[name].setdefault(mode,[]).append(values)
            work[name].setdefault(mode,[]).append(calls)
        outputs[name]['exact'] = [np.asarray([float(Fraction(ExactResearch().solve(d)['value'])) for d in worlds])]
    result = dict(train_roots=len(panels['train']), calibration_roots=len(panels['cal']), train_examples=len(y),
                  thresholds=dict(thresholds), seeds=list(SEEDS), head_macs={'ff':493,'trm':492},
                  panel_key_hashes=hashes, panels={})
    for name, arms in outputs.items():
        means = {a:np.mean(v,axis=0) for a,v in arms.items()}
        optimum = means['exact']
        if any(np.any(v>optimum+1e-9) for v in means.values()):
            raise ValueError('policy beats exact bound')
        idx = np.random.default_rng(20261128).integers(len(optimum),size=(10000,len(optimum)))
        pairs = {}
        for a,b in (('trm','one_step'),('trm','two_step'),('trm','ff'),('ff','one_step')):
            delta = means[a]-means[b]
            pairs[a+'_vs_'+b] = dict(delta=float(delta.mean()),
                                     ci95=np.quantile(delta[idx].mean(axis=1),[.025,.975]).tolist())
        result['panels'][name] = dict(roots=len(optimum), means={a:dict(utility=float(v.mean()),
            regret=float((optimum-v).mean()), mean_neural_candidate_calls=float(np.mean(work[name].get(a,[0.]))))
            for a,v in means.items()},paired=pairs)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--distill-checkout',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    path = a.distill_checkout.resolve()
    sha = subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip()
    if sha != UPSTREAM or dirty:
        raise ValueError('expected clean pinned Distill checkout')
    sys.path.insert(0,str(path))
    from Distill import research_cases, research_curriculum
    if not Path(research_curriculum.__file__).resolve().is_relative_to(path):
        raise ValueError('native source not loaded from pinned checkout')
    a.out.mkdir(parents=True,exist_ok=False)
    export = a.out/'native_reference'
    research_curriculum.build(export,counts=(48,16,32))
    audit = audit_directory(export)
    (a.out/'native_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('RV22Z_NATIVE_AUDIT '+json.dumps(audit,separators=(',',':')),flush=True)
    docs = {s:json.loads((export/'public'/f'{s}.json').read_text())['rows'] for s in SPLITS}
    audit_keys = {binary_key(r['public_case']['data'])[0] for rows in docs.values() for r in rows}
    panels = dict(train=panel(research_cases._data,TRAIN,128,('standard','frontier')),
                  cal=panel(research_cases._data,CAL,32,('standard','frontier')),
                  native_eval=[validate_world(r['public_case']['data']) for r in docs['public_eval']],
                  fresh=panel(research_cases._data,FRESH,64,('standard','frontier')),
                  foundation=panel(research_cases._data,FOUNDATION,64,('foundation',)))
    result = run_probe(panels,audit_keys)
    receipt = dict(schema='rv22z.native_probe.v1',upstream=sha,audit=audit,probe=result,
                   custody='synthetic_probe_only',training_admission=False,
                   scope='Fresh synthetic train labels recomputed locally; reference referee never used for fitting.')
    (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print('RV22Z_PROBE '+json.dumps(receipt,separators=(',',':'),allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
