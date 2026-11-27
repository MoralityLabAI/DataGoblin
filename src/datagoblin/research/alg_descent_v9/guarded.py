"""V9b: fail closed on valid products that do not bind indexed children."""
import hashlib, time
import torch
import experiment as exp
try:
    from alg_descent_v8.experiment import prob,bucket,structural,verify
except ImportError:
    from v8.experiment import prob,bucket,structural,verify

@torch.no_grad()
def route(r,prior,model=None,edit=None,reset=False):
    work=calls=rejects=binding_rejects=0;first=[];trace=[]
    def visit(s):
        nonlocal work,calls,rejects,binding_rejects
        parent=r.poly(s)
        if len(parent)==2:return True
        bank=r.bank(s);xs=r.x[s];history=[];remain=set(range(len(bank)));base=torch.tensor([prob(prior,bucket(r,a)) for a in bank]).logit();xt=torch.tensor(xs);st=[structural(a) for a in bank]
        while remain:
            h=torch.zeros(len(bank),2,20);m=torch.zeros(len(bank),2);hist=history[-2:]
            if hist:h[:,:len(hist)]=torch.tensor(hist);m[:,:len(hist)]=1
            q=base if model is None else base+(model(xt,h,m,edit,reset) if isinstance(model,exp.Dense) else model(xt,h,m));ps=q.sigmoid().clamp(.05,.995).tolist()
            i=min(remain,key=lambda j:(st[j]+bank[j][2]*(1/ps[j]-1),j))
            if not history:first.append((s,i))
            remain.remove(i);spans,cs,cost,_=bank[i]
            if work+cost>3*r.n*r.n or calls>=8*r.n:return False
            ok,paid=verify(parent,cs,r.p);assert paid==cost;work+=paid;calls+=1
            bound=ok and all(c==r.poly(t) for c,t in zip(cs,spans))
            binding_rejects+=int(ok and not bound);trace.append((s,i,bound))
            if not bound:rejects+=1;history.append(xs[i]);continue
            return all(visit(t) for t in spans)
        return False
    start=time.perf_counter_ns();closed=visit((0,r.n));elapsed=time.perf_counter_ns()-start
    assert work<=3*r.n*r.n and calls<=8*r.n
    return dict(closed=closed,work=work,calls=calls,rejects=rejects,binding_rejects=binding_rejects,elapsed_ns=elapsed,first=first[0][1],trace=hashlib.sha256(repr(trace).encode()).hexdigest())
if __name__=='__main__':
    exp.route=route
    exp.main()
