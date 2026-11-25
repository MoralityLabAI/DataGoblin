"""Fixed-dose matched miniature Aleph-A CPU diagnostic."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import torch
from torch.nn import functional as F
from state02 import words, expand, metrics, REGIMES
SEEDS=(229,233,239)
ARMS=('terminal','state','shuffled')
WIDTH=24
STEPS=384
N=32
PANEL_SEED=2026112505
LR=.001
LENGTHS=(12,24,48)

def model_factory(standalone):
    if standalone:
        from direct_excerpt import Aleph
    else:
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'aleph_rlt_placement'))
        from gym import Aleph
    return Aleph(width=WIDTH,vocab=512,heads=4,kv=1,core_ff=35*WIDTH//8)

def actor(ops):
    starts=torch.arange(4).repeat(len(ops)//4)[:,None]
    return torch.cat([torch.full_like(starts,260),torch.full_like(starts,262),starts,ops+32],1)

def targets(z,y,arm,sg):
    end=F.cross_entropy(z[:,-1],y[:,-1])
    if arm=='terminal': return end
    aux=y[:,:-1]
    if arm=='shuffled':
        aux=torch.stack([v[torch.randperm(len(v),generator=sg)] for v in aux.unbind(1)],1)
    return .5*(end+F.cross_entropy(z[:,:-1].reshape(-1,4),aux.reshape(-1)))

def eval_panel():
    g=torch.Generator().manual_seed(PANEL_SEED)
    return {f'{l}/{r}':expand(words(g,N,l,r)) for l in LENGTHS for r in REGIMES}

@torch.no_grad()
def evaluate(m,panel):
    m.eval(); out={}
    for key,(ops,y) in panel.items():
        captures=[]; pred=[]; first=[]; second=[]
        def hook(_mod,_args,h):
            captures.append(F.linear(m.norm(h),m.embed.weight)[:,3:,:4].argmax(-1))
        handle=m.blocks[10].register_forward_hook(hook)
        try:
            for x in ops.split(16):
                captures.clear(); z=m(actor(x),'direct',2,1)[:,3:,:4]
                assert len(captures)==2
                pred.append(z.argmax(-1));first.append(captures[0]);second.append(captures[1])
        finally: handle.remove()
        out[key]={}
        for name,p in [('final',pred),('core_pass1',first),('core_pass2',second)]:
            z=torch.cat(p); row=metrics(z,y); row['trace_hits']=int(z.eq(y).sum()); row['trace_total']=y.numel(); out[key][name]=row
    return out

def run(seed,arm,out,standalone):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
    m=model_factory(standalone); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=.01,foreach=False)
    g=torch.Generator().manual_seed(seed+10000);sg=torch.Generator().manual_seed(seed+20000)
    panel=eval_panel(); pos=0; labels=0; aux_labels=0; hist=[]; start=time.process_time();wall=time.perf_counter()
    for step in range(STEPS):
        l=4 if step%2==0 else 8
        ops,y=expand(torch.cat([words(g,2,l,r) for r in REGIMES[:3]]))
        x=actor(ops);pos+=x.numel();labels+=len(y);aux_labels+=0 if arm=='terminal' else y[:,:-1].numel()
        m.train(); z=m(x,'direct',2,1)[:,3:,:4];loss=targets(z,y,arm,sg)
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        if (step+1)%64==0:
            hist.append(dict(step=step+1,loss=float(loss.detach()),terminal_accuracy=float(z[:,-1].argmax(-1).eq(y[:,-1]).float().mean()),preclip=float(norm)))
            print(seed,arm,hist[-1],flush=True)
    cpu=time.process_time()-start; train_wall=time.perf_counter()-wall
    held=evaluate(m,panel); out.mkdir(parents=True,exist_ok=True)
    ck=out/f'{seed}_{arm}.pt';torch.save(m.state_dict(),ck)
    r=dict(seed=seed,arm=arm,width=WIDTH,parameters=sum(p.numel() for p in m.parameters()),steps=STEPS,positions=pos,terminal_labels=labels,auxiliary_labels=aux_labels,train_cpu_seconds=cpu,train_wall_seconds=train_wall,torch=torch.__version__,standalone=standalone,panel_seed=PANEL_SEED,words_per_cell=N,lr=LR,history=hist,held=held,checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest())
    (out/f'{seed}_{arm}.json').write_text(json.dumps(r,indent=2)+'\n');print('DONE',seed,arm,cpu,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--standalone',action='store_true');a=p.parse_args();run(a.seed,a.arm,a.out,a.standalone)
