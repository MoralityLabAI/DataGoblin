"""Post-hoc finite-alphabet calibration. All local transition labels are training data."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from state04 import actor,model_factory
from state02 import MAPS
from state04_diagnostics import evaluate

@torch.no_grad()
def local_count(m,x,y):
    m.eval();p=m(x,'direct',2,1)[:,-1,:4].argmax(-1)
    return int(p.eq(y).sum())

def run(parent,out,standalone):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    m=model_factory(standalone);m.load_state_dict(torch.load(parent,map_location='cpu',weights_only=True),strict=True)
    u=torch.arange(len(MAPS)).repeat_interleave(4)[:,None];x=actor(u);y=MAPS.flatten()
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,foreach=False)
    hist={0:local_count(m,x,y)};start=time.process_time()
    for step in range(1,65):
        m.train();loss=F.cross_entropy(m(x,'direct',2,1)[:,-1,:4],y)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step in (8,16,32,64):hist[step]=local_count(m,x,y)
    seconds=time.process_time()-start
    out.mkdir(exist_ok=True,parents=True);ck=out/parent.name;torch.save(m.state_dict(),ck)
    r=dict(parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),endpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),positions=64*x.numel(),labels=64*len(y),optimizer_restart=True,local_accuracy_history=hist,train_cpu_seconds=seconds,**evaluate(m))
    (out/(parent.stem+'.json')).write_text(json.dumps(r,indent=2)+'\n');print(parent.stem,hist,seconds,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--standalone',action='store_true');a=p.parse_args();run(a.checkpoint,a.out,a.standalone)
