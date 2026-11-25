"""Two-register held-edge CPU probe using a direct Aleph transcription."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
Q=8; NS=64; D=16; STEPS=256; B=20
SEEDS=(241,251,257); ARMS=('terminal','state','local')
OPS=('A+1','A+3','A+5','A+7','B+1','B+3','B+5','B+7','A+=B','B+=A','swap','A-=B','B-=A','A=0','B=0','A=B','B=A','A*=2','B*=2')
U=len(OPS)
def oracle(s,u):
 a,b=divmod(s,Q)
 if u<4:a+=2*u+1
 elif u<8:b+=2*(u-4)+1
 elif u==8:a+=b
 elif u==9:b+=a
 elif u==10:a,b=b,a
 elif u==11:a-=b
 elif u==12:b-=a
 elif u==13:a=0
 elif u==14:b=0
 elif u==15:a=b
 elif u==16:b=a
 elif u==17:a*=2
 elif u==18:b*=2
 else:raise ValueError(u)
 return (a%Q)*Q+b%Q
TABLE=np.array([[oracle(s,u) for u in range(U)] for s in range(NS)])
HELD=np.zeros((NS,U),bool);r=np.random.default_rng(2026112507)
for u in range(U):HELD[r.permutation(NS)[:16],u]=True
EDGES=np.argwhere(~HELD);ALLOWED=[np.flatnonzero(~x) for x in HELD]
class Block(nn.Module):
 def __init__(self,d,ff):
  super().__init__();self.n1=nn.RMSNorm(d);self.n2=nn.RMSNorm(d);self.q=nn.Linear(d,d,bias=False);self.k=nn.Linear(d,d//4,bias=False);self.v=nn.Linear(d,d//4,bias=False);self.o=nn.Linear(d,d,bias=False);self.up=nn.Linear(d,ff,bias=False);self.gate=nn.Linear(d,ff,bias=False);self.down=nn.Linear(ff,d,bias=False)
 @staticmethod
 def rope(x):
  t,k=x.shape[-2:];f=10000.**(-torch.arange(0,k,2,dtype=torch.float32)/k);p=torch.arange(t,dtype=torch.float32)[:,None]*f;c,s=p.cos(),p.sin();a,b=x[...,::2],x[...,1::2];return torch.stack((a*c-b*s,a*s+b*c),-1).flatten(-2)
 def forward(self,x):
  b,t,d=x.shape;z=self.n1(x);q=self.rope(self.q(z).view(b,t,4,d//4).transpose(1,2));k=self.rope(self.k(z).view(b,t,1,d//4).transpose(1,2));v=self.v(z).view(b,t,1,d//4).transpose(1,2);x=x+self.o(F.scaled_dot_product_attention(q,k.repeat_interleave(4,1),v.repeat_interleave(4,1),is_causal=True).transpose(1,2).reshape(b,t,d));z=self.n2(x);return x+self.down(F.silu(self.gate(z))*self.up(z))
class Aleph(nn.Module):
 def __init__(self):
  super().__init__();self.embed=nn.Embedding(512,D);nn.init.normal_(self.embed.weight,std=D**-.5);self.blocks=nn.ModuleList(Block(D,35*D//8 if 1<=i<=10 else 4*D) for i in range(14));self.norm=nn.RMSNorm(D)
 def forward(self,x):
  h=self.embed(x)
  for i in [0]+list(range(1,11))*2+[11,12,13]:h=self.blocks[i](h)
  return F.linear(self.norm(h),self.embed.weight)
def actor(s,u):
 s=torch.as_tensor(s,dtype=torch.long);u=torch.as_tensor(u,dtype=torch.long);return torch.cat([torch.full((len(s),1),260),torch.full((len(s),1),262),(s//Q)[:,None],(s%Q+Q)[:,None],u+32],1)
def logits(m,x):
 z=m(x)[:,4:];return z[:,:,:Q],z[:,:,Q:2*Q]
def loss(z,y):
 y=torch.as_tensor(y,dtype=torch.long);a,b=z;return .5*(F.cross_entropy(a.reshape(-1,Q),(y//Q).reshape(-1))+F.cross_entropy(b.reshape(-1,Q),(y%Q).reshape(-1)))
def pred(z):return (z[0].argmax(-1)*Q+z[1].argmax(-1)).cpu().numpy()
def paths(r,l):
 s=r.integers(NS,size=B);cur=s.copy();ops=[];ys=[]
 for _ in range(l):
  u=np.array([r.choice(ALLOWED[v]) for v in cur]);assert not HELD[cur,u].any();cur=TABLE[cur,u];ops.append(u);ys.append(cur.copy())
 return s,np.stack(ops,1),np.stack(ys,1)
def panel():
 r=np.random.default_rng(2026112506);out={};n=16
 for l in (12,24,48):
  for regime in ('permutation','single','double'):
   u=r.integers(13,size=(n,l))
   if regime!='permutation':u[np.arange(n),r.integers(1,l//2,size=n)]=r.integers(13,U,size=n)
   if regime=='double':u[np.arange(n),r.integers(l//2,l-1,size=n)]=r.integers(13,U,size=n)
   s=(np.arange(Q)[None,:]*Q+r.integers(Q,size=(n,1))).ravel();u=np.repeat(u,Q,0);cur=s.copy();ys=[];seen=[]
   for col in u.T:seen.append(HELD[cur,col]);cur=TABLE[cur,col];ys.append(cur.copy())
   out[f'{l}/{regime}']=(s,u,np.stack(ys,1),np.stack(seen,1))
 return out
@torch.no_grad()
def evaluate(m):
 m.eval();s,u=np.indices(TABLE.shape);s=s.ravel();u=u.ravel();local=np.concatenate([pred(logits(m,actor(a,b[:,None])))[:,0] for a,b in zip(np.array_split(s,32),np.array_split(u,32))]).reshape(TABLE.shape);ok=local==TABLE
 out=dict(train_correct=int(ok[~HELD].sum()),train_total=int((~HELD).sum()),held_correct=int(ok[HELD].sum()),held_total=int(HELD.sum()),local_map=local.tolist(),cells={})
 for key,(s,u,y,seen) in panel().items():
  p=np.concatenate([pred(logits(m,actor(a,b))) for a,b in zip(np.array_split(s,8),np.array_split(u,8))]);cur=s.copy();trace=[]
  for col in u.T:cur=local[cur,col];trace.append(cur.copy())
  feedback=np.stack(trace,1);row=dict(held_edge_traces=int(seen.any(1).sum()),suffix_ceiling=int(sum(np.bincount(v,minlength=NS).max() for v in y[:,-1].reshape(-1,Q))))
  for name,z in [('direct',p),('feedback',feedback),('symbolic',y)]:
   hit=z==y;prev=np.concatenate([s[:,None],z[:,:-1]],1);row[name]=dict(answer=int(hit[:,-1].sum()),total=len(s),orbit=int(hit[:,-1].reshape(-1,Q).all(1).sum()),orbit_total=len(s)//Q,trace=int(hit.all(1).reshape(-1,Q).all(1).sum()),state_hits=int(hit.sum()),state_total=int(hit.size),transition_hits=int((TABLE[prev,u]==z).sum()))
  out['cells'][key]=row
 return out
def run(seed,arm,out):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed);m=Aleph();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,foreach=False);r=np.random.default_rng(seed+10000);lr=np.random.default_rng(seed+20000);pos=labels=0;hist=[];start=time.process_time()
 for i in range(STEPS):
  l=4 if i%2==0 else 8;s,u,y=paths(r,l)
  if arm=='local':
   e=EDGES[lr.integers(len(EDGES),size=B*(4+l)//5)];s=e[:,0];u=e[:,1,None];y=TABLE[s,u[:,0]][:,None]
  x=actor(s,u);pos+=x.numel();z=logits(m,x)
  if arm=='state':v=.5*(loss(tuple(a[:,-1:] for a in z),y[:,-1:])+loss(tuple(a[:,:-1] for a in z),y[:,:-1]));labels+=y.size
  else:v=loss(tuple(a[:,-1:] for a in z),y[:,-1:]);labels+=len(y)
  opt.zero_grad(set_to_none=True);v.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
  if (i+1)%64==0:hist.append(dict(step=i+1,loss=float(v.detach())));print(seed,arm,hist[-1],flush=True)
 cpu=time.process_time()-start;res=dict(seed=seed,arm=arm,steps=STEPS,positions=pos,state_label_pairs=labels,params=sum(p.numel() for p in m.parameters()),cpu_seconds=cpu,history=hist,held=evaluate(m));out.mkdir(parents=True,exist_ok=True);(out/f'{seed}_{arm}.json').write_text(json.dumps(res,indent=2)+'\n');print('DONE',seed,arm,cpu,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.seed,a.arm,a.out)
