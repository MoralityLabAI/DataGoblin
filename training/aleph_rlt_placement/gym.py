"""Aleph RLT placement tournament: bounded, synthetic development Transformer.
Run from repo root: python training/aleph_rlt_placement/gym.py --smoke
Requires PyTorch. No network, teacher, verifier service, or paid backend calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ARMS = ("direct", "terminal", "first_last", "penultimate",
        "first_penultimate_last", "first_penultimate_damp", "global_full",
        "core_edges", "core_edges_damp", "pass_rlt", "pass_damp")


def schedule(r, s):
    if r < 1 or s < 1:
        raise ValueError("positive planned pass counts required")
    return [0] + list(range(1, 11)) * r + [11, 12] * s + [13]


def damping(weights):
    active = [a for a in weights if a < 1.0]
    mean = math.fsum(active) / len(active) if active else 1.0
    return [mean if a < 1.0 else 1.0 for a in weights]


def coefficients(arm, r, s):
    """Maps zero-based executed visits to (alpha, anchor).
    anchor 'local' is block input; 'pass' is input to the whole core pass.
    Unlisted visits use the native residual endpoint.
    """
    word = schedule(r, s)
    n = len(word)
    if arm not in ARMS:
        raise ValueError(arm)
    if arm == "direct":
        return {}
    if arm in ("core_edges", "core_edges_damp", "pass_rlt", "pass_damp"):
        weights = [1 / (r - j) for j in range(r)]
        if arm.endswith("damp"):
            weights = damping(weights)
        sites = (0, 9) if arm.startswith("core_edges") else (9,)
        anchor = "local" if arm.startswith("core_edges") else "pass"
        return {1 + 10*j + k: (weights[j], anchor)
                for j in range(r) for k in sites}
    sites = {
        "terminal": [n-1], "first_last": [0, n-1],
        "penultimate": [n-2],
        "first_penultimate_last": [0, n-2, n-1],
        "first_penultimate_damp": [0, n-2, n-1],
        "global_full": list(range(n)),
    }[arm]
    weights = [1 / (n - i) for i in sites]
    if arm.endswith("damp"):
        weights = damping(weights)
    return dict(zip(sites, [(a, "local") for a in weights], strict=True))


def blend(x, y, a):
    return y if a == 1.0 else x + a * (y - x)


class Block(nn.Module):
    def __init__(self, d, heads=4, kv=2):
        super().__init__()
        if d % (2*heads) or heads % kv:
            raise ValueError("invalid grouped-query dimensions")
        self.h, self.kv, self.k = heads, kv, d // heads
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.kp = nn.Linear(d, kv*self.k, bias=False)
        self.v = nn.Linear(d, kv*self.k, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 4*d, bias=False)
        self.gate = nn.Linear(d, 4*d, bias=False)
        self.down = nn.Linear(4*d, d, bias=False)

    @staticmethod
    def rope(x):
        t, k = x.shape[-2:]
        freq = 10000.0 ** (-torch.arange(0, k, 2, device=x.device, dtype=torch.float32)/k)
        phase = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * freq
        c, s = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
        a, b = x[..., ::2], x[..., 1::2]
        return torch.stack((a*c-b*s, a*s+b*c), -1).flatten(-2)

    def forward(self, x):
        b, t, d = x.shape
        z = self.n1(x)
        q = self.rope(self.q(z).view(b,t,self.h,self.k).transpose(1,2))
        k = self.rope(self.kp(z).view(b,t,self.kv,self.k).transpose(1,2))
        v = self.v(z).view(b,t,self.kv,self.k).transpose(1,2)
        k = k.repeat_interleave(self.h//self.kv, dim=1)
        v = v.repeat_interleave(self.h//self.kv, dim=1)
        attn = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        x = x + self.o(attn.transpose(1,2).reshape(b,t,d))
        z = self.n2(x)
        return x + self.down(F.silu(self.gate(z))*self.up(z))


class Aleph(nn.Module):
    def __init__(self, width=64, vocab=64):
        super().__init__()
        self.embed = nn.Embedding(vocab,width)
        nn.init.normal_(self.embed.weight,std=width**-0.5)
        self.blocks = nn.ModuleList(Block(width) for _ in range(14))
        self.norm = nn.RMSNorm(width)

    def forward(self, tokens, arm="direct", r=2, s=1):
        x = self.embed(tokens)
        word, coeff = schedule(r,s), coefficients(arm,r,s)
        pass_input = x
        for i, physical in enumerate(word):
            if physical == 1:
                pass_input = x
            y = self.blocks[physical](x)
            a, anchor = coeff.get(i, (1.0,"local"))
            x = blend(pass_input if anchor == "pass" else x, y, a)
        return F.linear(self.norm(x), self.embed.weight)


def example(rng, family, ood=False):
    # 16-node permutation graph, or modular accumulation; 20-token prompts.
    # The answer is never appended to the input. Family marker 48/49.
    if family == 0:
        p = rng.sample(range(16),16)
        start = rng.randrange(16)
        hops = rng.randrange(6,10) if ood else rng.randrange(1,5)
        ans = start
        for _ in range(hops):
            ans = p[ans]
        return [48,*p,start,32+hops,63],ans
    p = [rng.randrange(16) for _ in range(16)]
    start = rng.randrange(16)
    hops = rng.randrange(6,10) if ood else rng.randrange(1,5)
    return [49,*p,start,32+hops,63],(start+sum(p[:hops]))%16


def batch(seed, count, device, ood=False, family=None):
    rng = random.Random(seed)
    rows = [example(rng, i%2 if family is None else family, ood) for i in range(count)]
    x,y = zip(*rows,strict=True)
    return torch.tensor(x,device=device),torch.tensor(y,device=device)


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    tmp.replace(path)


def checkpoint(path, model, opt, step, cfg):
    tmp = path.with_suffix(".tmp")
    torch.save({"model":model.state_dict(),"opt":opt.state_dict(),
                "step":step,"cfg":cfg,"rng":torch.random.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() and next(model.parameters()).is_cuda else []},tmp)
    tmp.replace(path)


@torch.no_grad()
def evaluate(model, arm, r, s, device, n):
    model.eval()
    out = []
    for family in (0,1):
        for ood in (False,True):
            x,y = batch(f"dev:{family}:{int(ood)}",n,device,ood,family)
            logits = model(x,arm,r,s)[:,-1,:]
            rows = [{"root":digest(row),"prediction":pred,"target":target}
                    for row,pred,target in zip(x.cpu().tolist(),logits.argmax(-1).cpu().tolist(),y.cpu().tolist(),strict=True)]
            out.append({"family":family,"ood":ood,"correct":int((logits.argmax(-1)==y).sum()),
                        "n":n,"nll":float(F.cross_entropy(logits,y)),"rows":rows,
                        "prompt_tokens":int(x.numel()),"input_sha":digest(x.cpu().tolist())})
    model.train()
    return out


def fit(args, seed, arm, deadline):
    device = torch.device(args.device)
    torch.manual_seed(seed)
    model = Aleph(args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=args.lr)
    folder = args.out/f"{seed}-{arm}"
    folder.mkdir(parents=True,exist_ok=True)
    cfg = dict(seed=seed,arm=arm,width=args.width,r=args.r,s=args.s,
               steps=args.steps,batch=args.batch,lr=args.lr,device=args.device,eval_n=args.eval_n)
    meta = folder/"config.json"
    ckpt = folder/"checkpoint.pt"
    if meta.exists():
        if not args.resume or json.loads(meta.read_text()) != cfg:
            raise ValueError(f"existing run or changed config: {folder}")
    else:
        atomic_json(meta,cfg)
    if ckpt.exists():
        state = torch.load(ckpt,map_location="cpu",weights_only=True)
        if state["cfg"] != cfg:
            raise ValueError("checkpoint config mismatch")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        torch.random.set_rng_state(state["rng"])
        if state["cuda_rng"] and device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        step = state["step"]
    else:
        step = 0
    start = step
    t0 = time.monotonic()
    last_loss = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while step < args.steps and time.monotonic() < deadline:
        # Per-update data RNG is independent of model RNG and placement.
        x,y = batch(f"train:{seed}:{step}",args.batch,device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x,arm,args.r,args.s)[:,-1,:],y)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        opt.step()
        step += 1
        last_loss = float(loss.detach())
        if step % args.checkpoint_every == 0:
            checkpoint(ckpt,model,opt,step,cfg)
    checkpoint(ckpt,model,opt,step,cfg)
    fit_s = time.monotonic()-t0
    result = dict(config=cfg,params=sum(p.numel() for p in model.parameters()),
                  step=step,resume_from=start,complete=step==args.steps,
                  processed_positions=step*args.batch*20,loss_positions=step*args.batch,
                  executed_block_token_visits=step*args.batch*20*len(schedule(args.r,args.s)),
                  session_fit_s=fit_s,last_loss=last_loss,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
                  checkpoint_sha256=file_sha(ckpt))
    if result["complete"] and time.monotonic()<deadline:
        result["dev_native"] = evaluate(model,arm,args.r,args.s,device,args.eval_n)
        if args.swaps:
            result["dev_runtime_swap"] = {a:evaluate(model,a,args.r,args.s,device,args.eval_n)
                                          for a in ("direct","first_penultimate_last","pass_rlt") if time.monotonic()<deadline}
            result["dev_depth"] = {str(k):evaluate(model,arm,k,args.s,device,args.eval_n)
                                   for k in (1,3,4) if time.monotonic()<deadline}
    atomic_json(folder/"result.json",result)
    print(json.dumps(result),flush=True)
    return result


def self_test():
    assert len(schedule(2,1))==24 and len(set(schedule(2,1)))==14
    assert schedule(2,1)[22]==12
    assert math.isclose(coefficients("first_penultimate_damp",2,1)[0][0],13/48)
    assert coefficients("core_edges",2,1)==coefficients("core_edges_damp",2,1)
    assert coefficients("core_edges",3,1)!=coefficients("core_edges_damp",3,1)
    assert coefficients("first_last",2,1)=={0:(1/24,"local"),23:(1.0,"local")}
    assert coefficients("penultimate",2,1)=={22:(0.5,"local")}
    assert coefficients("core_edges",2,1)[1]==(0.5,"local")
    for arm in ARMS:
        assert all(0<a<=1 for a,_ in coefficients(arm,2,1).values())
    torch.manual_seed(11)
    m = Aleph(16)
    x,y = batch(23,2,"cpu")
    direct = m(x)
    assert torch.equal(direct,m(x,"terminal"))
    assert torch.equal(m(x,"direct",1,1),m(x,"pass_rlt",1,1))
    assert torch.equal(m(x,"direct",1,1),m(x,"core_edges",1,1))
    x2=x.clone()
    x2[:,-1]=62
    assert torch.allclose(direct[:,:-1],m(x2)[:,:-1],atol=1e-6,rtol=1e-6)
    F.cross_entropy(m(x,"first_penultimate_last")[:,-1,:],y).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    print("PASS: placement clocks, terminal identity, one-pass identity, causal mask, finite gradients")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--out",type=Path)
    p.add_argument("--device",default="cpu")
    p.add_argument("--width",type=int,default=64)
    p.add_argument("--r",type=int,default=2)
    p.add_argument("--s",type=int,default=1)
    p.add_argument("--steps",type=int,default=64)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--lr",type=float,default=0.001)
    p.add_argument("--seeds",type=int,nargs="+",default=[137,163])
    p.add_argument("--arms",nargs="+",choices=ARMS,default=["direct","first_last","penultimate","first_penultimate_last","first_penultimate_damp","pass_rlt"])
    p.add_argument("--seconds",type=int,default=600,help="soft inter-step deadline; checkpoint/eval may overrun")
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--eval-n",type=int,default=32)
    p.add_argument("--checkpoint-every",type=int,default=16)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--swaps",action="store_true")
    a=p.parse_args()
    if min(a.r,a.s,a.width,a.steps,a.batch,a.seconds,a.threads,a.eval_n,a.checkpoint_every)<1 or a.width%8 or not math.isfinite(a.lr) or a.lr<=0:
        p.error("positive bounds and width divisible by eight required")
    if len(set(a.arms))!=len(a.arms) or len(set(a.seeds))!=len(a.seeds):
        p.error("duplicate arms or seeds")
    torch.set_num_threads(a.threads)
    if a.smoke:
        self_test()
        if a.out is None:
            return
    if a.out is None:
        p.error("--out required for training")
    if a.out.exists() and not a.resume:
        p.error("output exists; choose new path or --resume")
    a.out.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+a.seconds
    for seed in a.seeds:
        for arm in a.arms:
            if time.monotonic()>=deadline:
                print("TIME BUDGET: remaining arms not started",flush=True)
                return
            fit(a,seed,arm,deadline)

if __name__=="__main__":
    main()
