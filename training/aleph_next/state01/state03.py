"""Cross-prefix state interchange; fixed dose, CPU only."""
import json,time,torch
from torch.nn import functional as F
from state02 import Executor, MAPS, ARMS, SEEDS, WORDS, STEPS, REGIMES, words, expand
N=128
CUTS=(4,12)
TAIL=24
PANEL_SEED=2026112503

def exact(s,u):
    for op in u.unbind(1): s=MAPS[op,s]
    return s

def advance(m,h,u):
    for op in F.one_hot(u,len(MAPS)).float().unbind(1): h=m.cell(op,h)
    return h

def hidden(m,a,u): return advance(m,m.init(F.one_hot(a,4).float()),u)

def panel():
    g=torch.Generator().manual_seed(PANEL_SEED); out={}
    for c in CUTS:
        rp=words(g,N,c,'survival'); dp=words(g,N,c,'survival')
        eq=(rp==dp).all(1); dp[eq,0]=(dp[eq,0]+1)%3
        assert not (rp==dp).all(1).any()
        rp,rt=expand(rp); dp,dt=expand(dp)
        rs=rt[:,-1]; inv=dt[:,-1].reshape(N,4).argsort(1)
        a=torch.arange(4).repeat(N); group=torch.arange(N).repeat_interleave(12)
        r=rs.repeat_interleave(3); target=((rs[:,None]+torch.arange(1,4))%4).flatten()
        candidates=torch.arange(4).expand(len(r),4)
        for regime in REGIMES:
            u=words(g,N,TAIL,regime).repeat_interleave(12,0)
            wrong=candidates[(candidates!=r[:,None])&(candidates!=target[:,None])].reshape(-1,2)
            wrong=wrong[torch.arange(len(r)),torch.randint(0,2,(len(r),),generator=g)]
            indices=[4*group+inv[group,s] for s in (r,target,wrong)]
            y=[exact(s,u) for s in (r,target,wrong)]
            out[f'{c}/{regime}']=(rp,dp,a,u,indices,y)
    return out

@torch.no_grad()
def score(m,panel,seed):
    w=m.readout.weight; _,s,v=torch.linalg.svd(w-w.mean(0),full_matrices=False)
    rank=int((s>s[0]*1e-5).sum()); b=v[:rank].T
    g=torch.Generator().manual_seed(seed+33333)
    q=torch.linalg.qr(torch.randn(w.shape[1],rank,generator=g)).Q
    out={}
    for key,(rp,dp,a,u,idx,y) in panel.items():
        r=hidden(m,a,rp).repeat_interleave(3,0); d=hidden(m,a,dp)
        same,target,wrong=(d[i] for i in idx); delta=target-r
        rd=(delta@b)@b.T; rnd=(delta@q)@q.T
        rnd=rnd*(rd.norm(dim=1,keepdim=True)/rnd.norm(dim=1,keepdim=True).clamp_min(1e-12))
        patch=dict(self=r,full=target,same=same,wrong=wrong,readout=r+rd,random=r+rnd,complement=target-rd)
        pred={k:m.readout(advance(m,h,u)).argmax(1) for k,h in patch.items()}
        y0,y1,yw=y; base=pred['self']; cells={}
        for regime,mask0 in [('surviving',y0!=y1),('erased',y0==y1)]:
            for tag,mask in [('all',mask0),('base_correct',mask0&base.eq(y0))]:
                row=dict(n=int(mask.sum()),wrong_oracle_target=int((mask&yw.eq(y1)).sum()))
                for k,z in pred.items():
                    row[k]=dict(target=int((mask&z.eq(y1)).sum()),original=int((mask&z.eq(y0)).sum()),changed=int((mask&z.ne(base)).sum()),own=int((mask&z.eq(yw if k=='wrong' else y0 if k in ('same','self') else y1)).sum()))
                cells[f'{regime}/{tag}']=row
        out[key]=cells
    return dict(rank=rank,cells=out)

def train(seed,arm):
    torch.manual_seed(seed); m=Executor(); opt=torch.optim.AdamW(m.parameters(),lr=.005,weight_decay=.01)
    g=torch.Generator().manual_seed(seed+10000); sg=torch.Generator().manual_seed(seed+20000)
    start=time.process_time()
    for i in range(STEPS):
        l=4 if i%2==0 else 8
        ops,y=expand(torch.cat([words(g,WORDS//3,l,r) for r in REGIMES[:3]])); z=m(ops)
        loss=F.cross_entropy(z[:,-1],y[:,-1])
        if arm!='terminal':
            aux=y[:,:-1]
            if arm=='shuffled': aux=torch.stack([v[torch.randperm(len(v),generator=sg)] for v in aux.unbind(1)],1)
            loss=loss+F.cross_entropy(z[:,:-1].reshape(-1,4),aux.reshape(-1))
        opt.zero_grad();loss.backward();opt.step()
    return m,time.process_time()-start

def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    p=panel(); runs=[]
    for seed in SEEDS:
        for arm in ARMS:
            m,seconds=train(seed,arm); runs.append(dict(seed=seed,arm=arm,seconds=seconds,**score(m,p,seed)))
            print(seed,arm,round(seconds,2),flush=True,file=__import__('sys').stderr)
    return dict(panel_seed=PANEL_SEED,n_words=N,cuts=CUTS,tail=TAIL,steps=STEPS,seeds=SEEDS,torch=torch.__version__,runs=runs)

if __name__=='__main__': print(json.dumps(run(),indent=2))
