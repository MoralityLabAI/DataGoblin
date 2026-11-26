"""Bijective MOD7 public-development fixture with a competence gate."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from datagoblin.training_gym.resource_plan import paired_outcomes

class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.init=nn.Linear(7,64)
        self.cell=nn.GRUCell(14,64)
        self.out=nn.Linear(64,7)
        self.probe=nn.Linear(64,7)
    def step(self,h,ab):
        return self.cell(F.one_hot(ab,7).flatten(1).float(),h)
    def states(self,r,ops):
        h=torch.tanh(self.init(F.one_hot(r,7).float()))
        hs=[]
        for j in range(ops.shape[1]):
            h=self.step(h,ops[:,j])
            hs.append(h)
        return torch.stack(hs,1)

def sample(g,n,length):
    r=torch.randint(7,(n,),generator=g)
    ops=torch.stack((
        torch.randint(1,7,(n,length),generator=g), torch.randint(7,(n,length),generator=g)
    ),-1)
    z=r
    path=[]
    for j in range(length):
        z=(ops[:,j,0]*z+ops[:,j,1])%7
        path.append(z)
    return r,ops,torch.stack(path,1)

def run(updates):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cal=sample(torch.Generator().manual_seed(3011),512,5)
    dev=sample(torch.Generator().manual_seed(4011),512,6)
    j=2
    z=dev[2][:,j]
    donors=[]
    shams=[]
    for i in range(len(z)):
        d=next(k%len(z) for k in range(i+1,i+len(z)) if z[k%len(z)]!=z[i])
        donors.append(d)
        shams.append(next(k%len(z) for k in range(d+1,d+len(z)) if z[k%len(z)]!=z[d] and z[k%len(z)]!=z[i]))
    donor=torch.tensor(donors)
    sham=torch.tensor(shams)
    def target(r):
        for k in range(j+1,6):
            r=(dev[1][:,k,0]*r+dev[1][:,k,1])%7
        return r
    cf=target(z[donor].clone())
    changed=cf!=dev[2][:,-1]
    records=[]
    comparisons=[]
    for seed in (11,23,37):
        outcomes={}
        for arm in ('A_task','B_concept','S_shuffled'):
            torch.manual_seed(seed)
            net=Core()
            opt=torch.optim.AdamW(net.parameters(),lr=.003)
            g=torch.Generator().manual_seed(10000+seed)
            t=time.monotonic()
            visits=0
            for i in range(updates):
                length=1 if i<updates//4 else 2 if i<updates//2 else 4
                visits+=96*length
                r,ops,path=sample(g,96,length)
                hs=net.states(r,ops)
                loss=F.cross_entropy(net.out(hs[:,-1]),path[:,-1])
                if arm!='A_task':
                    labels=path if arm=='B_concept' else path.roll(1,0)
                    loss=loss+.5*F.cross_entropy(net.probe(hs).reshape(-1,7),labels.reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            elapsed=time.monotonic()-t
            t=time.monotonic()
            heuristic=max(
                int((dev[1][:,-1,1]==dev[2][:,-1]).sum()),
                int(torch.bincount(dev[2][:,-1],minlength=7).max()),
            )
            with torch.no_grad():
                ordinary=net.out(net.states(dev[0],dev[1])[:,-1]).argmax(-1)==dev[2][:,-1]
            if int(ordinary.sum())<max(256,heuristic+52):
                rec={
                    'seed':seed, 'arm':arm, 'dev_correct':int(ordinary.sum()), 'dev_n':512,
                    'competent':False, 'heuristic_correct':heuristic, 'updates':updates,
                    'train_seconds':elapsed, 'token_block_visits':visits, 'root_draws':updates*96,
                    'counterfactual_n':None,
                }
                records.append(rec)
                outcomes[arm]=ordinary.int().tolist()
                print(json.dumps(rec),flush=True)
                continue
            with torch.no_grad():
                hc=net.states(cal[0],cal[1])[:,j]
                hd=net.states(dev[0],dev[1])
                h=hd[:,j]
                ordinary=net.out(hd[:,-1]).argmax(-1)==dev[2][:,-1]
                mean=hc.mean(0)
                X=hc-mean
                Y=F.one_hot(cal[2][:,j],7).float()
                lm=Y.mean(0)
                W=torch.linalg.solve(X.T@X+.1*torch.eye(64),X.T@(Y-lm))
                U,sv,_=torch.linalg.svd(W,full_matrices=False)
                rank=int((sv>1e-5).sum())
                U=U[:,:rank]
                probe=((h-mean)@W+lm).argmax(-1)==z
                def suffix(h0, net=net):
                    for k in range(j+1,6):
                        h0=net.step(h0,dev[1][:,k])
                    return net.out(h0).argmax(-1)
                delta=(h[donor]-h)@U@U.T
                sd=(h[sham]-h)@U@U.T
                norm=delta.norm(dim=1,keepdim=True)
                sn=sd.norm(dim=1,keepdim=True)
                valid=changed&(sn[:,0]>1e-8)
                sd=sd*norm/sn.clamp_min(1e-8)
                rd=torch.randn(h.shape,generator=torch.Generator().manual_seed(9000+seed))
                rd=rd*norm/rd.norm(dim=1,keepdim=True).clamp_min(1e-8)
                edits = [('identity', torch.zeros_like(h)), ('concept_patch', delta),
                         ('semantic_sham', sd), ('random_norm_matched', rd)]
                cfcounts = {name: int(((suffix(h+edit)==cf)&valid).sum()) for name,edit in edits}
                rec={
                    'seed':seed,
                    'arm':arm,
                    'parameters':sum(p.numel() for p in net.parameters()),
                    'updates':updates,
                    'root_draws':updates*96,
                    'token_block_visits':visits,
                    'competent':True,
                    'heuristic_correct':heuristic,
                    'train_seconds':elapsed,
                    'eval_seconds':time.monotonic()-t,
                    'dev_correct':int(ordinary.sum()),
                    'dev_n':len(z),
                    'probe_dev_correct':int(probe.sum()),
                    'probe_rank':rank,
                    'counterfactual_n':int(valid.sum()),
                    'counterfactual_correct':cfcounts,
                }
                records.append(rec)
                outcomes[arm]=ordinary.int().tolist()
                print(json.dumps(rec),flush=True)
        for arm in ('B_concept','S_shuffled'):
            comparisons.append({'seed':seed,'treatment':arm,**paired_outcomes(outcomes['A_task'],outcomes[arm])})
    return {
        'status':'PUBLIC_DEVELOPMENT_ONLY',
        'budget_estimand':'fixed exposure; not matched wall-clock',
        'train_lengths':[1,2,4],
        'competence_gate':'>=256/512 and >=heuristic+52 roots',
        'nonzero_multipliers':True,
        'calibration_length':5,
        'dev_length':6,
        'training_seeds':[11,23,37],
        'cpu_threads':2,
        'device':'cpu',
        'torch_version':torch.__version__,
        'platform':platform.platform(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'arms':records,
        'paired':comparisons,
        'nonclaims': [
            'No sealed test or paid compute', 'No fixed-wall-time conclusion',
            'Oracle-assisted donor selection for diagnostic only', 'No promotion or corpus admission',
            'No arm-specific peak RAM or GPU throughput measured',
        ],
    }

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--updates',type=int,default=1200)
    p.add_argument('--out',type=Path,default=Path('pilot_receipt.json'))
    a=p.parse_args()
    if not 1<=a.updates<=2000:
        p.error('pilot bounded to 1..2000 updates')
    a.out.write_text(json.dumps(run(a.updates),indent=2)+'\n')
