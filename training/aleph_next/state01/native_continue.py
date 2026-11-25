"""CPU-only mixed continuation from an explicitly hashed Aleph A checkpoint.

One arm per invocation; never overwrite historical weights or outputs.
Public diagnostic panel only. Register an independent confirmation separately.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from native_bridge import actor_input, state_logits, objective
from state02 import ARMS, REGIMES, expand, words, metrics

STATE_MASS=.2
PANEL_SEED=2026112504
N_WORDS_PER_REGIME=2


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def state_packet(seed,step):
    g=torch.Generator().manual_seed(seed*1000003+step+2026112504)
    length=4 if step%2==0 else 8
    return expand(torch.cat([words(g,N_WORDS_PER_REGIME,length,r) for r in REGIMES[:3]]))


def mixed_step(net,opt,native,rows,ops,y,arm,seed,step):
    count=Counter(tag for tag,_ in rows)
    if set(count)!=set(native.MASS): raise ValueError('native loss lanes changed')
    opt.param_groups[0]['lr']=.00035*min((step+1)/4,1)
    opt.zero_grad(set_to_none=True)
    loss_sum=0.; padded=0
    for group in native.microbatches(rows):
        x,lengths=native.tensor([r for _,r in group])
        z=net(x,'direct',2,1); terms=[]
        for i,(tag,r) in enumerate(group):
            if r.lane=='text':
                loss=F.cross_entropy(z[i,:lengths[i]],torch.tensor(r.good))
            else:
                scores=z[i,lengths[i]-1,:r.choices]
                loss=scores.logsumexp(0)-scores[list(r.good)].logsumexp(0)
            terms.append(loss*((1-STATE_MASS)*native.MASS[tag]/count[tag]))
        loss=torch.stack(terms).sum();loss.backward()
        loss_sum+=float(loss.detach());padded+=x.numel()
    # Terminal-only uses the full lane on terminal CE; auxiliary arms divide
    # its fixed mass equally between final and intermediate CE.
    sg=torch.Generator().manual_seed(seed*1000033+step+987654321)
    z=state_logits(net,ops)
    loss=objective(z,y,arm,sg)*(STATE_MASS if arm=='terminal' else STATE_MASS/2)
    loss.backward();loss_sum+=float(loss.detach())
    if any(p.grad is None for p in net.parameters()):
        raise ValueError('disconnected model parameter')
    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
    opt.step()
    return dict(loss=loss_sum,preclip=float(norm),
                native_positions=sum(len(r.prompt) for _,r in rows),
                native_padded=padded,state_positions=actor_input(ops).numel(),
                state_final_labels=len(y),state_auxiliary_labels=0 if arm=='terminal' else y[:,:-1].numel())


@torch.no_grad()
def diagnostics(net):
    net.eval(); g=torch.Generator().manual_seed(PANEL_SEED); out={}
    for length in (12,24,48):
        for regime in REGIMES:
            ops,y=expand(words(g,16,length,regime))
            chunks=[state_logits(net,v).argmax(-1) for v in ops.split(16)]
            out[f'{length}/{regime}']=metrics(torch.cat(chunks),y)
    return out


def main():
    import anchor14_train as native
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--pack',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--steps',type=int,choices=(1,128,512,2048),required=True)
    args=p.parse_args()
    if sha(args.checkpoint)!=args.sha256: raise ValueError('checkpoint digest mismatch')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(args.seed)
    train,dev,source,buckets=native.load_inputs(args.pack)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if 'model' in ck:
        key=ck.get('key',{})
        if key.get('arm','direct')!='direct' or key.get('seed',args.seed)!=args.seed:
            raise ValueError('expected matched direct parent')
        ck=ck['model']
    model=native.dense(256)
    if sum(p.numel() for p in model.parameters())!=native.PARAMS: raise ValueError('recipe mismatch')
    model.load_state_dict(ck,strict=True);del ck
    opt=torch.optim.AdamW(model.parameters(),lr=.00035,weight_decay=.01,foreach=False)
    args.out.mkdir(parents=True,exist_ok=False)
    meta=dict(parent_sha256=args.sha256,seed=args.seed,arm=args.arm,steps=args.steps,
              pack_source=source,optimizer_restart=True,state_mass=STATE_MASS,
              panel_seed=PANEL_SEED,source_sha256=sha(Path(__file__)),
              bridge_sha256=sha(HERE/'native_bridge.py'),generator_sha256=sha(HERE/'state02.py'))
    (args.out/'manifest.json').write_text(json.dumps(meta,indent=2,default=str))
    model.train();start=time.process_time()
    with (args.out/'steps.jsonl').open('w') as log:
        for step in range(args.steps):
            rows=native.logical_rows(train,buckets,args.seed,step)
            ops,y=state_packet(args.seed,step)
            record=mixed_step(model,opt,native,rows,ops,y,args.arm,args.seed,step)
            log.write(json.dumps(dict(step=step+1,**record))+'\n');log.flush()
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'manifest':meta},args.out/'endpoint.pt')
    result=dict(train_cpu_seconds=time.process_time()-start,
                endpoint_sha256=sha(args.out/'endpoint.pt'),
                public_diagnostic=diagnostics(model))
    (args.out/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
