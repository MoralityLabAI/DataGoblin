"""SG-18: bounded multiplier transfer on interleaved task trajectories."""
import json
import random
import statistics
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import sg07
import sg11
from sg15_lp import propose
from sg17_lp import chain_lp
from datagoblin.training_gym.spectral_blocks import BlockSearch, root_vector
from datagoblin.training_gym.spectral_block_replay import verify
from datagoblin.training_gym.spectral_face import face, recover_face, verify_certificate
from datagoblin.training_gym.spectral_primal import dual_round, recover, row_sweep
from datagoblin.training_gym.spectral_quotient import score, validate

ARMS = ('cold', 'identity', 'recency', 'nearest', 'learned', 'shuffled', 'bound_rank')
REGIMES = ('persistent', 'rotating', 'mixed')
PHASES = ('base', 'repeat', 'mild', 'large')
CAP, PROBES, FACE_CAP = 4, 2, 512


def trajectory(seed, regime, d=32, t=12):
    r = random.Random(seed)
    anchors = [sg07.task(seed * 10 + j, regime, d, t) for j in range(3)]
    for q in anchors:
        q.update(x=anchors[0]['x'][:], bits=anchors[0]['bits'][:],
                 k=[min(d, 7 + (seed+i)%2) for i in range(t)], **{'lambda': anchors[0]['lambda']})
    for phase in PHASES:
        order = list(range(3)); r.shuffle(order)
        for j in order:
            q = {**anchors[j], 'u': [row[:] for row in anchors[j]['u']]}
            n = min(d*t, {'mild': 8, 'large': 48}.get(phase, 0))
            for p in r.sample(range(d*t), n):
                q['u'][p//d][p%d] += r.choice((-1,1)) * (1 if phase=='mild' else r.randint(1,3))
            yield phase, j, q


def signature(q):
    # Include the entire task, not just the objective utility matrix.
    return (q['d'], q['T'], tuple(q['k']), tuple(q['x']), tuple(q['bits']), q['lambda'],
            tuple(v for row in q['u'] for v in row))


@dataclass(frozen=True)
class Entry:
    sig: tuple
    eta: tuple
    den: int
    stamp: int


class Memory:
    def __init__(self, cap=CAP):
        if type(cap) is not int or not 0 <= cap <= 64:
            raise ValueError('invalid capacity')
        self.cap = cap
        self.entries = []

    def put(self, q, schedule, eta, den, stamp):
        # Only exact, integer-certified results can become reusable records.
        if not verify_certificate(q, schedule, list(eta), den)['certified']:
            return False
        sig = signature(q)
        self.entries = [e for e in self.entries if e.sig != sig]
        if self.cap:
            self.entries.insert(0, Entry(sig, tuple(eta), den, stamp))
            del self.entries[self.cap:]
        return True

    def compatible(self, sig):
        # Restrict transfer to matching dimensions, read budgets and prices.
        return [e for e in self.entries if e.sig[:-1] == sig[:-1]]


def features(sig, entry, stamp, root):
    diff = [abs(a-b) for a,b in zip(sig[-1],entry.sig[-1],strict=True)]
    d = len(diff)
    return (1., sum(diff)/d, sum(v!=0 for v in diff)/d, max(diff,default=0)/16,
            (stamp-entry.stamp)/12, sum(abs(a-b/entry.den) for a,b in zip(root,entry.eta,strict=True))/len(root)/16,
            float(entry.sig==sig))


def retrieve(q, memory, arm, root, stamp, beta=()):
    sig = signature(q); pool = memory.compatible(sig)
    if arm=='cold': return []
    if arm=='identity': return [e for e in pool if e.sig==sig][:1]
    if arm=='recency': return pool[:PROBES]
    if arm=='bound_rank':
        return sorted(pool,key=lambda e: Fraction(face(q,list(e.eta),e.den)[0],e.den))[:PROBES]
    if arm=='nearest':
        return sorted(pool,key=lambda e:sum(abs(a-b) for a,b in zip(sig[-1],e.sig[-1],strict=True)))[:PROBES]
    if arm in ('learned','shuffled'):
        return sorted(pool,key=lambda e:-sum(a*b for a,b in zip(beta,features(sig,e,stamp,root),strict=True)))[:PROBES]
    raise ValueError('unknown policy')


class Acquisition:
    def __init__(self,q):
        self.q=validate(q); self.root=root_vector(q)
        self.schedule,_=recover(q,sg11.initial(q),self.root,'dual')
        self.eta,self.den=list(self.root),1
        self.cert=verify_certificate(q,self.schedule,self.eta,self.den)
        self.log=[]; self.block=None

    def attempt(self,eta,den,kind,stamp=None):
        q=self.q; e=list(eta)
        tick=time.process_time_ns()
        scaled={**q,'u':[[v*den for v in row] for row in q['u']], 'lambda':q['lambda']*den}
        a,_=row_sweep(q,dual_round(scaled,e))
        if score(q,a)['objective']>self.cert['objective']: self.schedule=a
        c=verify_certificate(q,self.schedule,e,den)
        nodes=0
        if not c['certified']:
            f=recover_face(q,e,den,node_cap=FACE_CAP);nodes=f['nodes']
            if f['schedule'] is not None:
                self.schedule=f['schedule'];c=verify_certificate(q,self.schedule,e,den)
        if c['bound_num']*self.den < self.cert['bound_num']*den:
            self.eta,self.den=e,den
        self.cert=verify_certificate(q,self.schedule,self.eta,self.den)
        self.log.append(dict(kind=kind,stamp=stamp,eta=e,den=den,certificate=c,
                             schedule=list(self.schedule),best=self.cert.copy(),nodes=nodes,
                             cpu_ns=time.process_time_ns()-tick))

    def finish(self):
        q=self.q
        for kind,fn in [('chain',chain_lp),('direct',propose)]:
            if self.cert['certified']: break
            p,info=fn(q)
            if p is not None:
                self.attempt(*p,kind);self.log[-1]['info']=info
            else:self.log.append(dict(kind=kind,failed=True,info=info))
        if not self.cert['certified']:
            init=list(self.schedule);s=BlockSearch(q,init,event_cap=2048,extra=[self.root]);r=s.run()
            self.schedule=r['schedule'];self.cert=verify_certificate(q,self.schedule,self.eta,self.den)
            self.block=dict(initial=init,extra=[self.root],receipt=r)

    def receipt(self):
        upper=min(self.cert['integer_upper'],self.block['receipt']['upper']) if self.block else self.cert['integer_upper']
        return dict(schedule=list(self.schedule),eta=list(self.eta),den=self.den,certificate=self.cert.copy(),
                    objective=self.cert['objective'],upper=upper,gap=upper-self.cert['objective'],log=list(self.log),block=self.block)


def audit(q,r):
    assert verify_certificate(q,r['schedule'],r['eta'],r['den'])==r['certificate']
    for e in r['log']:
        if 'certificate' in e:
            assert verify_certificate(q,e['schedule'],e['eta'],e['den'])==e['certificate']
    upper=r['certificate']['integer_upper']
    if r['block']:
        b=r['block'];z=b['receipt']
        assert verify(q,b['initial'],b['extra'],z['events'],event_cap=2048)=={k:v for k,v in z.items() if k!='peak_stored_blocks'}
        assert r['schedule']==z['schedule'];upper=min(upper,z['upper'])
    assert r['objective']==r['certificate']['objective'] and r['upper']==upper and r['gap']==upper-r['objective']>=0


def run(q,memory,arm,stamp,beta=()):
    tick=time.process_time_ns();s=Acquisition(q)
    root_certified=s.cert['certified']
    if not root_certified:
        for e in retrieve(q,memory,arm,s.root,stamp,beta):
            s.attempt(e.eta,e.den,'transfer',e.stamp)
            if s.cert['certified']:break
    transfer_certified=not root_certified and s.cert['certified']
    s.finish()
    r=s.receipt()
    if arm!='cold':memory.put(q,r['schedule'],r['eta'],r['den'],stamp)
    elapsed=time.process_time_ns()-tick
    audit(q,r)
    return dict(**r,cpu_ns=elapsed,root_certified=root_certified,transfer_certified=transfer_certified,
                probes=sum(e['kind']=='transfer' for e in r['log']),
                lp_calls=sum(e['kind'] in ('chain','direct') for e in r['log']),resident=len(memory.entries))


def train():
    x,y,null,counts=[],[],[],[]
    for j,reg in enumerate(REGIMES):
        for i in range(6):
            seed=1800000+j*1000+i;mem=Memory();xx,yy=[],[]
            for stamp,(_,_,q) in enumerate(trajectory(seed,reg)):
                init=Acquisition(q)
                if not init.cert['certified']:
                    sig=signature(q)
                    for e in mem.compatible(sig):
                        s=Acquisition(q);s.attempt(e.eta,e.den,'transfer',e.stamp)
                        audit(q,s.receipt());xx.append(features(sig,e,stamp,init.root));yy.append(float(s.cert['certified']))
                init.finish();r=init.receipt();audit(q,r)
                mem.put(q,r['schedule'],r['eta'],r['den'],stamp)
            ys=yy.copy();random.Random(seed).shuffle(ys)
            x.extend(xx);y.extend(yy);null.extend(ys);counts.append(dict(seed=seed,labels=len(yy),positive=sum(yy)))
    a,b=np.asarray(x),np.asarray(y);g=a.T@a+2*np.eye(a.shape[1]);
    return dict(learned=np.linalg.solve(g,a.T@b).tolist(),shuffled=np.linalg.solve(g,a.T@null).tolist(),counts=counts,labels=len(y),positive=sum(y))


def summarize(roots,phase=None):
    out={}
    for arm in ARMS:
        byroot=[[z for rep in r['arms'][arm] for z in rep if phase is None or z['phase']==phase] for r in roots]
        flat=[z for zs in byroot for z in zs]
        ms=[statistics.mean(z['cpu_ns'] for z in zs)/1e6 for zs in byroot]
        base=[statistics.mean(z['cpu_ns'] for rep in r['arms']['cold'] for z in rep if phase is None or z['phase']==phase)/1e6 for r in roots]
        out[arm]=dict(runs=len(flat),cpu_ms=statistics.mean(ms),certified=sum(z['gap']==0 for z in flat),
                      gap=statistics.mean(z['gap'] for z in flat),lp_calls=sum(z['lp_calls'] for z in flat),
                      transfers=sum(z['transfer_certified'] for z in flat),probes=sum(z['probes'] for z in flat),
                      root_certified=sum(z['root_certified'] for z in flat),wins=sum(a<b for a,b in zip(ms,base,strict=True)),
                      peak_resident=max(z['resident'] for z in flat))
    return out


def campaign():
    model=train();roots=[]
    Path('sg18_model.json').write_text(json.dumps(model,indent=2))
    for j,reg in enumerate(REGIMES):
        for i in range(8):
            seed=1810000+j*1000+i;tasks=list(trajectory(seed,reg));arms={a:[] for a in ARMS}
            for rep in range(2):
                order=list(ARMS);random.Random(seed+31*rep).shuffle(order)
                for arm in order:
                    mem=Memory();rows=[]
                    for stamp,(phase,anchor,q) in enumerate(tasks):
                        z=run(q,mem,arm,stamp,model.get(arm,()))
                        rows.append(dict(**z,phase=phase,anchor=anchor))
                    arms[arm].append(rows)
            for stamp in range(len(tasks)):
                rr=[rep[stamp] for reps in arms.values() for rep in reps]
                assert max(z['objective'] for z in rr)<=min(z['upper'] for z in rr)
            roots.append(dict(seed=seed,regime=reg,tasks=[dict(phase=p,anchor=a,task=q) for p,a,q in tasks],arms=arms))
            Path('sg18_progress.json').write_text(json.dumps(roots))
            print(seed,flush=True)
    return dict(version='SG-18/v0.1',training_roots=len(model['counts']),eval_roots=len(roots),model=model,
                summary=summarize(roots),by_phase={p:summarize(roots,p) for p in PHASES},roots=roots)


if __name__=='__main__':
    z=campaign();Path('sg18_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k not in ('roots','model')},indent=2))
