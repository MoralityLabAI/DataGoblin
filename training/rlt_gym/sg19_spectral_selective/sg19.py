"""SG-19: selective multiplier retrieval on no-repeat task trajectories."""
import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np
import sg07
import sg18

ARMS = ('cold', 'nearest2', 'nearest1', 'fixed16', 'tuned', 'learned', 'shuffled')
PHASES = ('base', 'small', 'medium', 'large')
THRESHOLDS = (0, 4, 8, 16, 32, 64, 96, 128, 1000000)


def trajectory(seed, regime, drift):
    rng = random.Random(seed)
    d, t = 32, 12
    anchors = [sg07.task(seed*10+j, regime, d, t) for j in range(3)]
    for q in anchors:
        q.update(x=anchors[0]['x'][:], bits=anchors[0]['bits'][:],
                 k=[7+(seed+i)%2 for i in range(t)], **{'lambda': anchors[0]['lambda']})
    current = [q['u'] for q in anchors]
    seen = set()
    for phase, n in zip(PHASES, (0, 4, 16, 48), strict=True):
        order = list(range(3)); rng.shuffle(order)
        for j in order:
            q = {**anchors[j], 'u': [row[:] for row in (current[j] if drift else anchors[j]['u'])]}
            for p in rng.sample(range(d*t), n):
                q['u'][p//d][p%d] += rng.choice((-1, 1)) * (1 if phase != 'large' else rng.randint(1, 3))
            current[j] = q['u']
            sig = sg18.signature(q)
            assert sig not in seen
            seen.add(sig)
            yield phase, j, q


def distance(sig, e):
    return sum(abs(a-b) for a, b in zip(sig[-1], e.sig[-1], strict=True))


def features(sig, e, stamp, root_gap, dist):
    return (1., min(4., dist/64), math.exp(-dist/8), math.exp(-dist/32),
            math.exp(-dist/128), (stamp-e.stamp)/12, min(4., root_gap/32),
            math.exp(-dist/32)*min(4., root_gap/32))


def retrieve(q, mem, arm, stamp, gap, model):
    if arm == 'cold':
        return []
    sig = sg18.signature(q)
    rows = [(e, distance(sig, e)) for e in mem.compatible(sig)]
    rows.sort(key=lambda v: v[1])
    if arm == 'nearest2':
        return [e for e, _ in rows[:2]]
    if arm == 'nearest1':
        return [e for e, _ in rows[:1]]
    if arm in ('fixed16', 'tuned'):
        limit = 16 if arm == 'fixed16' else model['threshold']
        return [e for e, dist in rows[:1] if dist <= limit]
    if arm in ('learned', 'shuffled'):
        if not rows:
            return []
        def value(v):
            e, dist = v
            return sum(a*b for a,b in zip(model[arm], features(sig,e,stamp,gap,dist),strict=True))
        best = max(rows, key=value)
        return [best[0]] if value(best) > 0 else []
    raise ValueError('unknown policy')


def run(q, mem, arm, stamp, model):
    tick = time.process_time_ns()
    s = sg18.Acquisition(q)
    root = s.cert['certified']
    chosen = [] if root else retrieve(q,mem,arm,stamp,s.cert['gap'],model)
    for e in chosen:
        s.attempt(e.eta,e.den,'transfer',e.stamp)
        if s.cert['certified']:
            break
    transferred = not root and s.cert['certified']
    s.finish()
    r = s.receipt()
    if arm != 'cold':
        mem.put(q,r['schedule'],r['eta'],r['den'],stamp)
    elapsed = time.process_time_ns()-tick
    sg18.audit(q,r)
    return dict(**r,cpu_ns=elapsed,root_certified=root,transfer_certified=transferred,
                lp_calls=sum(e['kind'] in ('chain','direct') for e in r['log']),
                probes=sum(e['kind']=='transfer' for e in r['log']),resident=len(mem.entries),
                selected=len(chosen))


def train():
    rows = []
    for j,reg in enumerate(sg18.REGIMES):
        for i in range(6):
            seed = 1900000+j*1000+i; mem = sg18.Memory(); data = []
            for stamp, (phase, anchor, q) in enumerate(trajectory(seed,reg,bool(i%2))):
                init = sg18.Acquisition(q); sig = sg18.signature(q); gap = init.cert['gap']
                tick = time.process_time_ns(); init.finish(); finish_ns = time.process_time_ns()-tick
                r = init.receipt(); sg18.audit(q,r)
                if gap:
                    pool = sorted(mem.compatible(sig),key=lambda e:distance(sig,e))
                    for rank,e in enumerate(pool):
                        s = sg18.Acquisition(q); tick = time.process_time_ns()
                        s.attempt(e.eta,e.den,'transfer',e.stamp); probe_ns = time.process_time_ns()-tick
                        sg18.audit(q,s.receipt()); closed = s.cert['certified']; dist = distance(sig,e)
                        # Immediate acquisition value proxy. Failure is charged the probe cost;
                        # subsequent solver effects and future memory contents are not modeled.
                        net = (finish_ns*int(closed)-probe_ns)/1e6
                        data.append(dict(stamp=stamp,phase=phase,rank=rank,dist=dist,closed=closed,
                                         probe_ns=probe_ns,finish_ns=finish_ns,net=net,
                                         x=features(sig,e,stamp,gap,dist)))
                mem.put(q,r['schedule'],r['eta'],r['den'],stamp)
            rows.append(dict(seed=seed,drift=bool(i%2),data=data))
    a=[]; y=[]; null=[]; w=[]
    for r in rows:
        z=r['data']; v=[e['net'] for e in z]; shuffle=v[:];random.Random(r['seed']).shuffle(shuffle)
        a.extend(e['x'] for e in z);y.extend(v);null.extend(shuffle);w.extend([1/max(1,len(z))]*len(z))
    a,w=np.asarray(a),np.asarray(w);g=a.T@(a*w[:,None])+0.1*np.eye(a.shape[1])
    def fit(v):return np.linalg.solve(g,a.T@(w*np.asarray(v))).tolist()
    # Select a one-probe threshold on training roots, using nearest candidate only.
    scores={t:sum(sum(e['net'] for e in r['data'] if e['rank']==0 and e['dist']<=t) for r in rows) for t in THRESHOLDS}
    return dict(learned=fit(y),shuffled=fit(null),threshold=max(scores,key=lambda t:(scores[t],-t)),
                threshold_scores=scores,labels=len(y),successes=sum(e['closed'] for r in rows for e in r['data']),rows=rows)


def summarize(roots,phase=None):
    out={}
    for arm in ARMS:
        groups=[[z for rep in r['arms'][arm] for z in rep if phase is None or z['phase']==phase] for r in roots]
        flat=[z for group in groups for z in group]
        ms=[statistics.mean(z['cpu_ns'] for z in group)/1e6 for group in groups]
        cold=[statistics.mean(z['cpu_ns'] for rep in r['arms']['cold'] for z in rep if phase is None or z['phase']==phase)/1e6 for r in roots]
        out[arm]=dict(cpu_ms=statistics.mean(ms),runs=len(flat),certified=sum(z['gap']==0 for z in flat),
                      mean_gap=statistics.mean(z['gap'] for z in flat),lp_calls=sum(z['lp_calls'] for z in flat),
                      probes=sum(z['probes'] for z in flat),transfers=sum(z['transfer_certified'] for z in flat),
                      roots_certified=sum(z['root_certified'] for z in flat),wins=sum(a<b for a,b in zip(ms,cold,strict=True)),
                      peak_resident=max(z['resident'] for z in flat))
    return out


def campaign(resume=False):
    if resume:
        model=json.loads(Path('sg19_model.json').read_text()); roots=json.loads(Path('sg19_progress.json').read_text())
    else:
        model=train(); roots=[]
        Path('sg19_model.json').write_text(json.dumps(model,indent=2)+'\n')
    done={r['seed'] for r in roots}
    for j,reg in enumerate(sg18.REGIMES):
        for i in range(8):
            seed=1910000+j*1000+i
            if seed in done: continue
            tasks=list(trajectory(seed,reg,bool(i%2))); arms={a:[] for a in ARMS}
            for rep in range(2):
                order=list(ARMS);random.Random(seed+31*rep).shuffle(order)
                for arm in order:
                    mem=sg18.Memory();rows=[]
                    for stamp,(phase,anchor,q) in enumerate(tasks):
                        z=run(q,mem,arm,stamp,model);rows.append(dict(**z,phase=phase,anchor=anchor))
                    arms[arm].append(rows)
            for stamp in range(len(tasks)):
                rr=[rep[stamp] for reps in arms.values() for rep in reps]
                assert max(z['objective'] for z in rr)<=min(z['upper'] for z in rr)
            roots.append(dict(seed=seed,regime=reg,drift=bool(i%2),tasks=[dict(phase=p,anchor=a,task=q) for p,a,q in tasks],arms=arms))
            Path('sg19_progress.json').write_text(json.dumps(roots))
            print(seed,flush=True)
    return dict(version='SG-19/v0.1',model=model,roots=roots,summary=summarize(roots),
                by_phase={p:summarize(roots,p) for p in PHASES},
                by_drift={str(d):summarize([r for r in roots if r['drift']==d]) for d in (False,True)})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    z=campaign(args.resume);Path('sg19_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k not in ('roots','model')},indent=2))
