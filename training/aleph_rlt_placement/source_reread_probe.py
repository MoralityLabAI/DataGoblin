"""Independent, public synthetic prelude-reread screen; no mixed archive."""
import argparse, json, math, time, hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ARMS=('direct','prelude_reread','site_damp','global_full_damp','direct3','pass_rlt3','pass_damp3')
SEEDS=(137,163,191)

class Block(nn.Module):
    def __init__(self,d):
        super().__init__(); self.d=d; self.k=d//4
        self.n1=nn.RMSNorm(d); self.n2=nn.RMSNorm(d)
        self.q=nn.Linear(d,d,bias=False); self.kp=nn.Linear(d,self.k,bias=False); self.v=nn.Linear(d,self.k,bias=False); self.o=nn.Linear(d,d,bias=False)
        self.up=nn.Linear(d,4*d,bias=False); self.gate=nn.Linear(d,4*d,bias=False); self.down=nn.Linear(4*d,d,bias=False)
    def rope(self,x):
        t,k=x.shape[-2:]; f=10000.**(-torch.arange(0,k,2,dtype=x.dtype)/k); a=torch.arange(t,dtype=x.dtype)[:,None]*f
        u,v=x[...,::2],x[...,1::2]; return torch.stack((u*a.cos()-v*a.sin(),u*a.sin()+v*a.cos()),-1).flatten(-2)
    def forward(self,x):
        b,t,d=x.shape; z=self.n1(x)
        q=self.rope(self.q(z).reshape(b,t,4,self.k).transpose(1,2)); k=self.rope(self.kp(z).reshape(b,t,1,self.k).transpose(1,2)).expand(-1,4,-1,-1); v=self.v(z).reshape(b,t,1,self.k).transpose(1,2).expand(-1,4,-1,-1)
        x=x+self.o(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(b,t,d)); z=self.n2(x)
        return x+self.down(F.silu(self.gate(z))*self.up(z))

class Model(nn.Module):
    def __init__(self,d=16):
        super().__init__(); self.embed=nn.Embedding(32,d); nn.init.normal_(self.embed.weight,std=d**-.5); self.blocks=nn.ModuleList(Block(d) for _ in range(14)); self.norm=nn.RMSNorm(d)
    def forward(self,tokens,arm):
        r=3 if arm.endswith('3') else 2; word=[0]+list(range(1,11))*r+[11,12,13]; x=self.embed(tokens); p=x; start=x
        damp=sum(1/i for i in range(2,25))/23
        for i,j in enumerate(word):
            if j==1: start=x
            if i==16 and arm in ('prelude_reread','site_damp'): x=torch.lerp(x,p if arm=='prelude_reread' else start,.25)
            y=self.blocks[j](x)
            if arm=='global_full_damp' and i<len(word)-1: y=torch.lerp(x,y,damp)
            if j==10 and arm in ('pass_rlt3','pass_damp3'):
                k=(i-1)//10; a=(1/(3-k) if arm=='pass_rlt3' else (5/12 if k<2 else 1.)); y=torch.lerp(start,y,a) if a<1 else y
            x=y
            if j==0: p=x
        return F.linear(self.norm(x),self.embed.weight)

def op(a,b,k): return torch.where(k==0,a^b,torch.where(k==1,a&b,a|b))

def data(n,seed,held=False, exhaustive=False):
    g=torch.Generator().manual_seed(seed); b=torch.randint(0,2,(n,4),generator=g); a,c,d,e=b.T; k=torch.randint(0,3,(n,),generator=g); l=torch.randint(0,3,(n,),generator=g)
    # Exclude XOR-after-OR and OR-after-XOR combinations from training.
    bad=((k==0)&(l==2))|((k==2)&(l==0)); l=torch.where(bad,torch.ones_like(l),l) if not held else torch.where(k==0,2,0); k=torch.where((k==1)&held,torch.full_like(k,2),k)
    if exhaustive:
        import itertools
        rows=[(*bits,i,j) for bits in itertools.product(range(2),repeat=4) for i in range(3) for j in range(3) if ((i,j) in ((0,2),(2,0)))==held]
        a,c,d,e,k,l=torch.tensor(rows).T
    # Three paired views: unchanged, relevant edit of first premise, irrelevant edit of fourth bit.
    old=op(op(a,c,k),d,l); y1=op(op(1-a,c,k),d,l); n0=torch.zeros_like(a); n1=torch.ones_like(a)
    def prompt(edit,val): return torch.stack((n0+8,a+2,c+2,d+2,e+2,k+12,l+16,old+20,edit+24,val+2,n0+31),1)
    x=torch.stack((prompt(n0,a),prompt(n1,1-a),prompt(n1+1,1-e)),1); y=torch.stack((old,y1,old),1)
    return x,y

@torch.no_grad()
def score(m,arm,x,y):
    out=[]; loss=0.
    for i in range(0,len(x),64):
        z=m(x[i:i+64].flatten(0,1),arm)[:,-1,:2]; yy=y[i:i+64].flatten(); loss+=float(F.cross_entropy(z,yy,reduction='sum')); out.append(z.argmax(-1).reshape(-1,3))
    pred=torch.cat(out); hits=pred==y; ch=y[:,0]!=y[:,1]; correct=hits.all(1)
    return dict(nll=loss/y.numel(),accuracy=float(hits.float().mean()),joint=float(correct.float().mean()),changed_joint=float(correct[ch].float().mean()),unchanged=float(hits[:,0].float().mean()),edited=float(hits[:,1].float().mean()),irrelevant=float(hits[:,2].float().mean()),pred=pred.tolist(),roots=len(x),copy_old_joint=float((y[:,0]==y[:,1]).float().mean()),copy_old_accuracy=float((y==y[:,0,None]).float().mean()))

def run(seed,arm,steps,width):
    torch.manual_seed(seed); m=Model(width); init=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in m.parameters())).hexdigest(); opt=torch.optim.AdamW(m.parameters(),lr=.001); t=time.monotonic()
    for step in range(steps):
        x,y=data(8,seed*100000+step); opt.zero_grad(set_to_none=True); z=m(x.flatten(0,1),arm)[:,-1,:2]; loss=F.cross_entropy(z,y.flatten()); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True); opt.step()
    fit=time.monotonic()-t; x,y=data(256,987654,True,True); xn,yn=data(256,987655,False,True); native=score(m,arm,x,y); native_in=score(m,arm,xn,yn)
    swaps={a:score(m,a,x,y) for a in ('direct','prelude_reread','site_damp','global_full_damp')} if not arm.endswith('3') else {a:score(m,a,x,y) for a in ('direct3','pass_rlt3','pass_damp3')}
    return dict(seed=seed,arm=arm,width=width,steps=steps,params=sum(p.numel() for p in m.parameters()),init_sha=init,fit_s=fit,train_loss=float(loss.detach()),positions=steps*8*3*11,block_positions=steps*8*3*11*(34 if arm.endswith('3') else 24),held=native,in_distribution=native_in,swaps=swaps)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--arm',choices=ARMS,required=True); p.add_argument('--steps',type=int,default=96); p.add_argument('--width',type=int,default=16); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); result=run(a.seed,a.arm,a.steps,a.width); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,sort_keys=True)); print(a.seed,a.arm,result['fit_s'],result['held']['joint'],result['held']['nll'],flush=True)
