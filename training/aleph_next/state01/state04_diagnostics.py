"""Diagnostic-only endpoint readout and autoregressive state feedback.
No gold state is fed into the rollout after the supplied initial state.
"""
import argparse,json
from pathlib import Path
import torch
from state04 import actor,model_factory,eval_panel
from state02 import MAPS,metrics

@torch.no_grad()
def evaluate(m):
    m.eval()
    # Each operation with every possible start. No test-label lookup in actor.
    u=torch.arange(len(MAPS)).repeat_interleave(4)[:,None]
    local=m(actor(u),'direct',2,1)[:,-1,:4].argmax(1).reshape(-1,4)
    results={}
    for key,(ops,y) in eval_panel().items():
        direct=torch.cat([m(actor(chunk),'direct',2,1)[:,3:,:4].argmax(-1) for chunk in ops.split(16)])
        # Predicted-state-only feedback is equivalent to repeated one-step model calls.
        start=torch.arange(4).repeat(len(ops)//4); s=start; z=[]
        for op in ops.unbind(1):
            s=local[op,s]; z.append(s)
        feedback=torch.stack(z,1)
        row={}
        for name,p in [('direct',direct),('feedback',feedback)]:
            r=metrics(p,y); ok=p.eq(y)
            r['trace_hits']=int(ok.sum());r['trace_total']=ok.numel()
            r['prefix_hits']=int(ok[:,:8].sum());r['prefix_total']=ok[:,:8].numel()
            r['tail_hits']=int(ok[:,8:].sum());r['tail_total']=ok[:,8:].numel()
            prev=torch.cat([start[:,None],p[:,:-1]],1)
            expected=MAPS[ops,prev];r['transition_hits']=int(p.eq(expected).sum());r['transition_total']=p.numel()
            row[name]=r
        results[key]=row
    return dict(local_predicted_map=local.tolist(),local_correct=int(local.eq(MAPS).sum()),local_total=MAPS.numel(),held=results)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--standalone',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);m=model_factory(a.standalone);m.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True),strict=True)
    a.out.write_text(json.dumps(evaluate(m),indent=2)+'\n')
