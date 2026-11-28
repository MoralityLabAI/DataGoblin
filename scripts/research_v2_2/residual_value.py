"""RV22-J: continuation residuals and selective exact search."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from decision_value import Head, World, make_world

SEEDS = (41, 42, 43)
SPLITS = {'train': (181, 128), 'cal': (182, 32), 'iid': (183, 96), 'parity3': (184, 96), 'parity4': (185, 96)}
MODES = ('direct', 'residual1', 'residual1_pair', 'residual2_pair')
FIELDS = ('utility', 'regret', 'credits', 'cpu_ms', 'dp_states', 'head_calls', 'expanded')


def parity4(rng):
    latent = np.array([[i >> j & 1 for j in range(4)] for i in range(16)])
    bits = [latent[:, j] for j in range(4)]
    while len(bits) < 8:
        b = rng.integers(0, 2, 16)
        if len(set(b)) == 2 and not any(np.array_equal(b, x) or np.array_equal(1-b, x) for x in bits):
            bits.append(b)
    order = rng.permutation(16)
    weights = rng.integers(1, 6, 16)
    return {'hypotheses': [{'weight': int(weights[i]), 'class': 'AB'[latent[i].sum()%2]} for i in order],
            'tests': [{'id': f't{j}', 'cost': int(rng.integers(1, 4)),
                       'bits': [int(bits[k][i]) for i in order]} for j,k in enumerate(rng.permutation(8))],
            'budget': 9, 'correct_reward': 12, 'wrong_reward': -16}


def clear():
    World.mass.cache_clear()
    World.a_mass.cache_clear()
    World.value.cache_clear()


class ResidualWorld(World):
    def __init__(self, d):
        super().__init__(d)
        self.head_calls = self.expanded = 0

    def local(self, mask, b, used, i, z, o, depth):
        return (self.q(mask,b,used,i,z,o,depth)-self.stop(mask)[0])/self.mass(mask)/12

    def pair(self, mask, b, used, i, z, o, q1):
        c, u = self.costs[i], used | 1 << i
        # Exact one-step best improvement in each branch. These features compute Q2.
        a = [(self.value(m,b-c,u,1)-self.stop(m)[0])/self.mass(m)/12 for m in (z,o)]
        p = self.mass(z)/self.mass(mask)
        gain = p*a[0]+(1-p)*a[1]
        return [gain,abs(a[0]-a[1]),min(a),max(a),float(a[0]>0),float(a[1]>0)], q1+gain

    def examples2(self):
        states = {(self.full,self.d['budget'],0)}
        for i,z,o in self.legal(self.full,self.d['budget'],0):
            states.update((m,self.d['budget']-self.costs[i],1 << i) for m in (z,o))
        x,y,base = [],[],[]
        for mask,b,used in sorted(states):
            for i,z,o in self.legal(mask,b,used):
                q1 = self.local(mask,b,used,i,z,o,1)
                extra,q2 = self.pair(mask,b,used,i,z,o,q1)
                truth = self.local(mask,b,used,i,z,o,len(self.tests))
                if abs(q2-self.local(mask,b,used,i,z,o,2))>1e-12 or q2>truth+1e-12:
                    raise ValueError('depth-two parity or bound failed')
                x.append(self.features(mask,b,used,i,z,o)+extra)
                y.append(truth)
                base.append((q1,q2))
        return x,y,base

    def tree(self, mode, model=None, tau=0.):
        def build(mask,b,used):
            _,stop = self.stop(mask)
            legal = list(self.legal(mask,b,used))
            if not legal:
                return ('stop',stop)
            q1 = np.array([self.local(mask,b,used,i,z,o,1) for i,z,o in legal])
            if mode=='one_step':
                scores = q1
            elif mode in ('two_step','oracle'):
                depth = 2 if mode=='two_step' else len(self.tests)
                self.expanded += len(legal)
                scores = np.array([self.local(mask,b,used,i,z,o,depth) for i,z,o in legal])
            else:
                x = [self.features(mask,b,used,i,z,o) for i,z,o in legal]
                q2 = None
                if mode.endswith('_pair'):
                    pairs = [self.pair(mask,b,used,i,z,o,v) for (i,z,o),v in zip(legal,q1,strict=True)]
                    x = [a+p[0] for a,p in zip(x,pairs,strict=True)]
                    q2 = np.array([p[1] for p in pairs])
                    self.expanded += len(legal)
                self.head_calls += len(legal)
                scores = model.predict(np.array(x))
                if mode in ('residual1','residual1_pair','shortlist2'):
                    scores = q1+np.maximum(scores,0.)
                elif mode=='residual2_pair':
                    scores = q2+np.maximum(scores,0.)
                if mode=='shortlist2':
                    ix = np.argsort(-scores,kind='stable')[:2]
                    scores = np.full(len(legal),-np.inf)
                    self.expanded += len(ix)
                    for j in ix:
                        scores[j] = self.local(mask,b,used,*legal[j],2)
            j = int(np.argmax(np.round(scores,12)))
            if scores[j]<=tau+1e-12:
                return ('stop',stop)
            i,z,o = legal[j]
            return ('test',i,build(z,b-self.costs[i],used | 1 << i),build(o,b-self.costs[i],used | 1 << i))
        return build(self.full,self.d['budget'],0)


def evaluate(worlds,mode,model=None,tau=0.):
    rows=[]
    for d in worlds:
        start=time.process_time_ns()
        w=ResidualWorld(d)
        tree=w.tree(mode,model,tau)
        elapsed=(time.process_time_ns()-start)/1e6
        work=w.work
        utility,credits=w.replay(tree)
        optimum=w.value(w.full,d['budget'],0,len(w.tests))/w.mass(w.full)
        if utility>optimum+1e-10 or (mode=='oracle' and abs(utility-optimum)>1e-10):
            raise ValueError('invalid replay')
        rows.append((utility,optimum-utility,credits,elapsed,work,w.head_calls,w.expanded))
        clear()
    return np.asarray(rows)


def run():
    worlds={}
    for name,(seed,n) in SPLITS.items():
        rng=np.random.default_rng(seed)
        worlds[name]=[parity4(rng) if name=='parity4' else make_world(rng,name=='parity3') for _ in range(n)]
    seen,hashes=set(),{}
    for name,rows in worlds.items():
        ids=[hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest() for d in rows]
        if len(set(ids))!=len(ids) or seen.intersection(ids):
            raise ValueError('world collision')
        seen.update(ids)
        hashes[name]=hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    x,y,bases=[],[],[]
    for d in worlds['train']:
        a,b,c=ResidualWorld(d).examples2()
        x.extend(a)
        y.extend(b)
        bases.extend(c)
        clear()
    x,y,bases=np.asarray(x),np.asarray(y),np.asarray(bases)
    results={s:{} for s in ('iid','parity3','parity4')}
    report={'schema':'rv22j.residual.v1','splits':{k:len(v) for k,v in worlds.items()},'split_hashes':hashes,
            'train_examples':len(x),'seeds':list(SEEDS),'arms':{},'paired':{},'thresholds':{}}
    for mode in ('one_step','two_step','oracle'):
        for split in results:
            results[split][mode]=[evaluate(worlds[split],mode)]
    for mode in MODES:
        cols=20 if mode.endswith('_pair') else 14
        target=y if mode=='direct' else y-bases[:,int(mode=='residual2_pair')]
        report['thresholds'][mode]=[]
        for seed in SEEDS:
            model=Head(seed,3,16)
            if cols!=14:
                model.params[0]=np.random.default_rng(seed).normal(0,.2,(cols,16))
            model.fit(x[:,:cols],target)
            tau=max((evaluate(worlds['cal'],mode,model,t)[:,0].mean(),-t,t)
                    for t in (0.,.025,.05,.1,.2))[2]
            report['thresholds'][mode].append(tau)
            for split in results:
                results[split].setdefault(mode,[]).append(evaluate(worlds[split],mode,model,tau))
                if mode=='residual1':
                    results[split].setdefault('shortlist2',[]).append(evaluate(worlds[split],'shortlist2',model))
        print('finished '+mode,flush=True)
    for split,arms in results.items():
        means={mode:np.asarray(rows).mean(axis=0) for mode,rows in arms.items()}
        report['arms'][split]={mode:dict(zip(FIELDS,v.mean(axis=0).tolist(),strict=True)) for mode,v in means.items()}
        rng=np.random.default_rng(20261128)
        idx=rng.integers(len(worlds[split]),size=(10000,len(worlds[split])))
        report['paired'][split]={}
        for mode in (*MODES,'shortlist2'):
            for control in ('direct','two_step'):
                if mode==control:
                    continue
                d=means[mode][:,0]-means[control][:,0]
                b=d[idx].mean(axis=1)
                report['paired'][split][mode+'_vs_'+control]={'delta':float(d.mean()),
                    'ci95':np.quantile(b,[.025,.975]).tolist(),'lower95':float(np.quantile(b,.05))}
    report['head_macs']={'14_features':752,'20_features':848}
    report['scope']=('Synthetic worlds; CPU process time includes construction and policy tree; '
                     'excludes replay and grading.')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise ValueError('fresh output required')
    args.out.write_text(json.dumps(run(),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
