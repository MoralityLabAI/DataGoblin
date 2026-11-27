"""RV22-K: cached integer depth-two search and learned width controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from decision_value import Head, World, make_world
from residual_value import clear, parity4

SEEDS = (61, 62, 63)
SPLITS = {'train': (281,128), 'cal': (282,48), 'iid': (283,96), 'parity3': (284,96),
          'iid16': (285,96), 'parity4': (286,96)}
ARMS = ('legacy','exact','bounded','random2','analytic2','ff2','trm2','adaptive')
FIELDS = ('utility','regret','credits','cpu_ms','depth2_candidates','branch_tests','head_calls')


def generate(rng, kind):
    if kind == 'iid16':
        d = parity4(rng)
        for h,c in zip(d['hypotheses'],rng.permutation(list('A'*8+'B'*8)),strict=True):
            h['class'] = str(c)
        return d
    return parity4(rng) if kind == 'parity4' else make_world(rng,kind == 'parity3')


class FastWorld(World):
    """Used deterministic tests are constant on every reachable descendant mask."""
    def __init__(self,d):
        super().__init__(d)
        self.weights = [h['weight'] for h in self.h]
        self.aweights = [h['weight'] if h['class']=='A' else 0 for h in self.h]
        self.counts,self.stops,self.branch = {0:(0,0)}, {}, {}
        self.expanded = self.branch_tests = self.head_calls = 0
        self.c,self.r = d['correct_reward'],d['wrong_reward']

    def counts_for(self,m):
        if m not in self.counts:
            bit=m & -m
            j=bit.bit_length()-1
            w,a=self.counts_for(m^bit)
            self.counts[m]=w+self.weights[j],a+self.aweights[j]
        return self.counts[m]

    def mass(self,m):
        return self.counts_for(m)[0]

    def a_mass(self,m):
        return self.counts_for(m)[1]

    def stop(self,m):
        if m not in self.stops:
            w,a=self.counts_for(m)
            self.stops[m]=max((a*self.c+(w-a)*self.r,'A'),((w-a)*self.c+a*self.r,'B'),(0,'defer'))
        return self.stops[m]

    def legal(self,m,b,used=0):
        for i,c in enumerate(self.costs):
            o=m & self.masks[i]
            if c<=b and o and o!=m:
                yield i,m^o,o

    def q1(self,m,i,z,o):
        return self.stop(z)[0]+self.stop(o)[0]-self.costs[i]*self.mass(m)

    def branch1(self,m,b):
        key=m,b
        if key not in self.branch:
            v=self.stop(m)[0]
            if v<self.c*self.mass(m):
                for a in self.legal(m,b):
                    self.branch_tests+=1
                    v=max(v,self.q1(m,*a))
            self.branch[key]=v
        return self.branch[key]

    def q2(self,m,b,i,z,o):
        self.expanded+=1
        c=self.costs[i]
        return self.branch1(z,b-c)+self.branch1(o,b-c)-c*self.mass(m)

    def upper1(self,m,b):
        v,w=self.stop(m)[0],self.mass(m)
        if v==self.c*w:
            return v
        cost=min((self.costs[i] for i,_,_ in self.legal(m,b)),default=None)
        return v if cost is None else max(v,(self.c-cost)*w)

    def upper2(self,m,b,i,z,o):
        c=self.costs[i]
        return self.upper1(z,b-c)+self.upper1(o,b-c)-c*self.mass(m)

    def tree(self,mode,model=None,params=None,seed=0):
        rng=np.random.default_rng(seed) if mode=='random2' else None
        def build(m,b,u):
            stop,decision=self.stop(m)
            legal=list(self.legal(m,b))
            if not legal or stop==self.c*self.mass(m):
                return ('stop',decision)
            q1=np.array([self.q1(m,*a) for a in legal])
            order=np.argsort(-q1,kind='stable')
            if mode=='random2':
                order=rng.permutation(len(legal))[:2]
            elif mode=='analytic2':
                order=order[:2]
            elif mode in {'trm2','ff2','adaptive'}:
                x=np.array([self.features(m,b,u,*a) for a in legal])
                scores=q1/self.mass(m)/12+np.maximum(model.predict(x),0.)
                self.head_calls+=len(legal)
                order=np.argsort(-scores,kind='stable')
                width=2
                if mode=='adaptive':
                    gap,wide=params
                    if len(order)>1 and scores[order[0]]-scores[order[1]]>=gap:
                        width=1
                    elif len(order)>2 and scores[order[1]]-scores[order[2]]<gap:
                        width=wide
                order=order[:width]
            best,chosen=stop,None
            for j in order:
                a=legal[int(j)]
                if mode=='bounded':
                    ub=self.upper2(m,b,*a)
                    if ub<best or (ub==best and (chosen is None or a[0]>=chosen[0])):
                        continue
                val=self.q2(m,b,*a)
                if val>best or (val==best and chosen is not None and a[0]<chosen[0]):
                    best,chosen=val,a
            if chosen is None:
                return ('stop',decision)
            i,z,o=chosen
            return ('test',i,build(z,b-self.costs[i],u|1<<i),build(o,b-self.costs[i],u|1<<i))
        return build(self.full,self.d['budget'],0)


def train(worlds):
    x,y=[],[]
    for d in worlds:
        a,b=World(d).examples()
        x.extend(a)
        y.extend(b)
        clear()
    x,y=np.asarray(x),np.asarray(y)
    models={name:[Head(s,depth,width).fit(x,y-x[:,13]) for s in SEEDS]
            for name,depth,width in (('trm',3,16),('ff',1,50))}
    return models,len(x)


def measure(d,mode,model=None,params=None,seed=0):
    start=time.process_time_ns()
    if mode=='legacy':
        w=World(d)
        tree,_=w.policy('two_step')
        counts=(0,0,0)
    else:
        w=FastWorld(d)
        tree=w.tree(mode,model,params,seed)
        counts=w.expanded,w.branch_tests,w.head_calls
    ms=(time.process_time_ns()-start)/1e6
    u,c=w.replay(tree)
    clear()
    return tree,np.array([u,0,c,ms,*counts],dtype=float)


def optimum(d):
    w=World(d)
    v=w.value(w.full,d['budget'],0,len(w.tests))/w.mass(w.full)
    clear()
    return v


def campaign():
    worlds={}
    for name,(seed,n) in SPLITS.items():
        rng=np.random.default_rng(seed)
        worlds[name]=[generate(rng,'iid16' if name=='cal' and i%2 else name) for i in range(n)]
    ids={k:[hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest() for d in v]
         for k,v in worlds.items()}
    flat=[x for v in ids.values() for x in v]
    if len(set(flat))!=len(flat):
        raise ValueError('duplicate world')
    models,n=train(worlds['train'])
    print('trained',n,flush=True)
    baseline=np.mean([measure(d,'exact')[1][0] for d in worlds['cal']])
    cal=[]
    for p in ((g,w) for g in (.025,.05,.10) for w in (3,8)):
        a=np.array([measure(d,'adaptive',m,p)[1] for m in models['trm'] for d in worlds['cal']])
        cal.append({'params':p,'utility':float(a[:,0].mean()),'expanded':float(a[:,4].mean())})
    feasible=[r for r in cal if r['utility']>=baseline-.05]
    params=min(feasible,key=lambda r:(r['expanded'],-r['utility'],r['params']))['params'] if feasible else None
    report={'schema':'rv22k.adaptive_width.v1','train_examples':n,'seeds':SEEDS,
            'splits':{k:len(v) for k,v in worlds.items()},
            'split_hashes':{k:hashlib.sha256('\n'.join(v).encode()).hexdigest() for k,v in ids.items()},
            'calibration':{'exact_utility':float(baseline),'grid':cal,'selected':params},'cohorts':{},'paired':{}}
    print('calibration',report['calibration'],flush=True)
    rng=np.random.default_rng(771)
    for name in ('iid','parity3','iid16','parity4'):
        data=worlds[name]
        rows={a:[] for a in ARMS}
        for j,d in enumerate(data):
            opt=optimum(d)
            root={a:[] for a in ARMS}
            for s in range(3):
                trees={}
                for a in rng.permutation(ARMS):
                    m=models['ff'][s] if a=='ff2' else models['trm'][s] if a in {'trm2','adaptive'} else None
                    mode='exact' if a=='adaptive' and params is None else a
                    tree,r=measure(d,mode,m,params,seed=SEEDS[s]*1000+j)
                    if r[0]>opt+1e-10:
                        raise ValueError('utility exceeds optimum')
                    r[1]=opt-r[0]
                    root[a].append(r)
                    trees[a]=tree
                if trees['exact']!=trees['bounded'] or abs(root['legacy'][-1][0]-root['exact'][-1][0])>1e-10:
                    raise ValueError('exact policy parity failure')
            for a in ARMS:
                rows[a].append(np.mean(root[a],axis=0))
        rows={a:np.array(v) for a,v in rows.items()}
        report['cohorts'][name]={a:dict(zip(FIELDS,v.mean(axis=0).tolist(),strict=True)) for a,v in rows.items()}
        idx=rng.integers(len(data),size=(10000,len(data)))
        report['paired'][name]={}
        for a in ARMS:
            d=rows[a][:,0]-rows['exact'][:,0]
            cpu=rows[a][:,3]-rows['exact'][:,3]
            b=d[idx].mean(axis=1)
            report['paired'][name][a]={'delta':float(d.mean()),'ci95':np.quantile(b,[.025,.975]).tolist(),
                'lower95':float(np.quantile(b,.05)),'cpu_delta_ms':float(cpu.mean()),
                'cpu_ci95_ms':np.quantile(cpu[idx].mean(axis=1),[.025,.975]).tolist()}
        print(name,json.dumps(report['cohorts'][name]),flush=True)
    report['scope']='Synthetic; CPU construction and policy tree only; fixed-seed paired root bootstrap.'
    report['head_macs']={'ff':750,'trm':752}
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise ValueError('fresh output required')
    args.out.write_text(json.dumps(campaign(),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
