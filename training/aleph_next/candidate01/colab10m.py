"""Direct-A single-GPU engineering pilot on the immutable first-10M corpus.

No runtime is provisioned here. --execute is a deliberate operator action.
The process wall limit is checked between microbatches; terminate the Colab
runtime externally to stop billing. No TPU/distributed qualification is claimed.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import time
from collections import defaultdict
from contextlib import ExitStack, nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from production import HERE, digest, read
from restart import Aleph, adam, lr, restore_rng, rng_state, schedule
from packing import reference_logits, sdpa_mask, token_loss_sum
from milestone_mix import audit_pack
from snapshot import MixedStream

PIN = "be25522ded491e0c0c3f81d87d786034b46733e1ccbe2b9cea2dafc47b48abf4"
RECIPE_SHA = "58fd8c2805f04ed918b4547d5a177624f839648c6be98dc749426ba39cd32e87"
RECIPE = HERE / "RECIPE.json"
RECEIPT = HERE / "MILESTONE10M.json"


def rotary(x, positions):
    k = x.shape[-1]
    freq = 10000.0 ** (-torch.arange(0, k, 2, device=x.device, dtype=torch.float32) / k)
    phase = positions.float()[:, :, None] * freq
    c, s = phase.cos().to(x.dtype)[:, None], phase.sin().to(x.dtype)[:, None]
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((a*c-b*s, a*s+b*c), -1).flatten(-2)


def packed_block(block, h, positions, mask):
    b, t, d = h.shape
    z = block.n1(h)
    q = rotary(block.q(z).view(b,t,block.h,block.k).transpose(1,2), positions)
    k = rotary(block.kp(z).view(b,t,block.kv,block.k).transpose(1,2), positions)
    v = block.v(z).view(b,t,block.kv,block.k).transpose(1,2)
    k = k.repeat_interleave(block.h//block.kv, dim=1)
    v = v.repeat_interleave(block.h//block.kv, dim=1)
    a = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=False)
    h = h + block.o(a.transpose(1,2).reshape(b,t,d))
    z = block.n2(h)
    return h + block.down(F.silu(block.gate(z))*block.up(z))


def precision(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False) if device.type=="cuda" else nullcontext()


def query_logits(net, pack, remat=True):
    """Use native parameters and direct visit schedule; skip padded and masked logits."""
    device = net.embed.weight.device
    n = pack["nonpadding"]
    ids, positions, segments, labels = (
        torch.tensor(pack[k][:n], device=device, dtype=torch.long)[None]
        for k in ("input_ids","position_ids","segment_ids","labels_at_query")
    )
    valid = labels != -100
    if not bool(valid.any()):
        raise ValueError("skip empty-supervision packs before calling")
    with precision(device):
        h = net.embed(ids)
        mask = sdpa_mask(segments)
        for i in schedule(2,1):
            block = net.blocks[i]
            if remat:
                h = checkpoint(packed_block, block, h, positions, mask,
                               use_reentrant=False, preserve_rng_state=True)
            else:
                h = packed_block(block,h,positions,mask)
        logits = F.linear(net.norm(h[valid]), net.embed.weight)
    return logits.float(), labels[valid]


def fixture():
    def pack(ids,segments,positions,labels,n):
        return dict(input_ids=ids,segment_ids=segments,position_ids=positions,
                    labels_at_query=labels,nonpadding=n)
    a = pack([1,2,3,4,5,6,0,0],[0,0,0,1,1,1,-1,-1],
             [0,1,2,0,1,2,0,0],[-100,3,-100,5,6,-100,-100,-100],6)
    b = pack([7,8,9,0],[0,0,0,-1],[0,1,2,0],[8,-100,-100,-100],3)
    empty = dict(b,labels_at_query=[-100]*4)
    return [a,b,empty]


def self_test(device):
    torch.set_num_threads(1)
    torch.manual_seed(137)
    native = Aleph(width=16,vocab=64,heads=4,kv=1,core_ff=70).to(device)
    other = copy.deepcopy(native)
    packs = fixture()
    # FP32 CPU test uses strict tolerances; CUDA BF16 comparison is exploratory
    # and separately reports its declared numerical tolerances.
    atol, rtol = (2e-4,2e-4) if device.type=="cpu" else (0.15,0.08)
    ga, gr = (3e-4,3e-4) if device.type=="cpu" else (0.25,0.15)
    total = sum(v!=-100 for p in packs for v in p["labels_at_query"])
    ref_loss = packed_loss = 0.
    for p in packs:
        labels = torch.tensor(p["labels_at_query"],device=device)
        with precision(device):
            z = reference_logits(native,p,"direct",2,1)
        numerator,count = token_loss_sum(z.float(),p["labels_at_query"])
        (numerator/total).backward()
        ref_loss += float(numerator.detach())
        if not int(count):continue
        zz,yy = query_logits(other,p,True)
        torch.testing.assert_close(zz,z[0,labels!=-100].float(),atol=atol,rtol=rtol)
        num = F.cross_entropy(zz,yy,reduction="sum")
        (num/total).backward()
        packed_loss += float(num.detach())
    errors=[]
    for (n,p),(m,q) in zip(native.named_parameters(),other.named_parameters(),strict=True):
        assert n==m and p.grad is not None and q.grad is not None
        assert torch.isfinite(q.grad).all()
        torch.testing.assert_close(p.grad,q.grad,atol=ga,rtol=gr)
        errors.append(float((p.grad-q.grad).abs().max()))
    p = packs[0]
    q = dict(p,input_ids=[10,11,12,*p["input_ids"][3:]])
    with torch.no_grad():
        z,_ = query_logits(other,p,False)
        zz,_ = query_logits(other,q,False)
    torch.testing.assert_close(z[1:],zz[1:],atol=atol,rtol=rtol)
    assert len({id(p) for p in native.parameters()})==len(list(native.parameters()))
    result=dict(device=str(device),labels=total,ref_loss=ref_loss/total,
                packed_loss=packed_loss/total,max_gradient_abs_error=max(errors),
                logit_tolerances=[atol,rtol],gradient_tolerances=[ga,gr],
                document_isolation=True,all_masked_pack=True,unequal_label_counts=True,
                fullsize_steps=0)
    print(json.dumps(result),flush=True)
    return result


def atomic_json(path,obj):
    tmp=path.with_suffix(".partial")
    with tmp.open("w") as f:
        json.dump(obj,f,indent=2,allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)


def collect_group(stream,expected):
    sha=hashlib.sha256(); packs=[]; labels=positions=padding=0
    while (item:=stream.next_pack(expected["stop"])) is not None:
        lane,p=item
        n,k=audit_pack(p,lane,sha)
        packs.append((lane,p,k))
        labels+=k;positions+=n;padding+=len(p["input_ids"])-n
    got=dict(sha256=sha.hexdigest(),positions=positions,labels=labels,
             packs=len(packs),padding=padding,stop=expected["stop"])
    if got!=expected or labels==0:
        raise ValueError(f"frozen update-group mismatch: {got}")
    return packs,labels


def save(out,identity,net,opt,stream,group,records):
    # Publish pointer only after checkpoint file has been written and hashed.
    # Store on persistent operator-controlled storage; local Colab disk is ephemeral.
    state=dict(identity=identity,model=net.state_dict(),optimizer=opt.state_dict(),
               group=group,cursor=stream.checkpoint(),rng=rng_state(),
               cuda_rng=torch.cuda.get_rng_state(),records=records)
    path=out/f"group_{group:03d}.pt"
    tmp=path.with_suffix(".partial")
    if path.exists():raise FileExistsError(path)
    with tmp.open("wb") as f:
        torch.save(state,f)
        f.flush()
        os.fsync(f.fileno())
    stamp=digest(tmp)
    os.replace(tmp,path)
    atomic_json(out/"LATEST.json",dict(file=path.name,sha256=stamp,group=group))
    return stamp


def main_run(args):
    started=time.monotonic()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("a BF16 CUDA accelerator is required")
    device=torch.device("cuda:0")
    prop=torch.cuda.get_device_properties(device)
    free,total=torch.cuda.mem_get_info(device)
    if total<35*2**30 or free<30*2**30:
        raise RuntimeError(f"need a mostly empty >=40 GB class accelerator; free={free}")
    if digest(RECIPE)!=RECIPE_SHA or digest(args.base/"index.json")!=PIN:
        raise ValueError("frozen recipe/index differs")
    cfg=read(RECIPE); gold=read(RECEIPT); groups=gold["bounded_replay"]["steps"]
    if len(groups)!=39 or not 1<=args.max_groups<=len(groups):
        raise ValueError("invalid group limit")
    if args.resume:
        if not (args.out/"LATEST.json").is_file():raise ValueError("no published checkpoint")
    else:
        args.out.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(args.out).free<26_000_000_000:
        raise RuntimeError("insufficient checkpoint storage; retain two complete generations")
    torch.set_num_threads(1);torch.manual_seed(137);random.seed(137);np.random.seed(137)
    torch.backends.cuda.matmul.allow_tf32=False
    self_test(device)
    torch.manual_seed(137);random.seed(137);np.random.seed(137)
    a=cfg["construction"]
    net=Aleph(a["width"],a["vocab"],heads=a["query_heads"],kv=a["kv_heads"],core_ff=a["core_ff"])
    assert sum(p.numel() for p in net.parameters())==a["parameters"]==999352320
    assert list(schedule(2,1))==[0,*range(1,11),*range(1,11),11,12,13]
    net=net.to(device);opt=adam(net,cfg)
    identity=dict(recipe=RECIPE_SHA,index=PIN,code=digest(Path(__file__)),
                  model=digest(HERE.parents[1]/"aleph_rlt_placement/gym.py"),
                  packer=digest(HERE.parents[1]/"aleph_rlt_placement/packing.py"),
                  torch=str(torch.__version__),cuda=torch.version.cuda,device=prop.name,
                  capacity=total,world_size=1)
    records=[];cursor=None;first=0
    if args.resume:
        pointer=read(args.out/"LATEST.json");path=args.out/pointer["file"]
        if digest(path)!=pointer["sha256"]:raise ValueError("checkpoint hash mismatch")
        ck=torch.load(path,map_location="cpu",weights_only=True)
        if ck["identity"]!=identity or ck["group"]!=pointer["group"]:raise ValueError("resume identity mismatch")
        first=ck["group"];records=ck["records"];cursor=ck["cursor"]
        assert first==len(records) and records[-1]["stop"]==groups[first-1]["stop"]
        net.load_state_dict(ck["model"],strict=True);opt.load_state_dict(ck["optimizer"])
        restore_rng(ck["rng"]);torch.cuda.set_rng_state(ck["cuda_rng"]);del ck
    atomic_json(args.out/"IDENTITY.json",identity)
    with ExitStack() as stack:
        stream=MixedStream(args.base,args.base/"index.json",PIN,stack,cursor)
        assert sum(stream.totals.values())==(groups[first-1]["stop"] if first else 0)
        for i in range(first,args.max_groups):
            if time.monotonic()-started>=args.wall_seconds:break
            t=time.monotonic();group=i+1;expected=groups[i]
            packs,denom=collect_group(stream,expected)
            opt.zero_grad(set_to_none=True)
            numer=defaultdict(float);counts=defaultdict(int)
            torch.cuda.reset_peak_memory_stats(device)
            complete=True
            for lane,pack,k in packs:
                if time.monotonic()-started>=args.wall_seconds:
                    complete=False;break
                if not k:continue
                z,y=query_logits(net,pack,True)
                loss=F.cross_entropy(z,y,reduction="sum")
                if not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite loss")
                (loss/denom).backward()
                numer[lane]+=float(loss.detach());counts[lane]+=k
                del z,y,loss
            if not complete:
                opt.zero_grad(set_to_none=True)
                print("wall cap hit during accumulation; discard partial group and restore last published checkpoint",flush=True)
                break
            if sum(counts.values())!=denom:raise ValueError("label accounting differs")
            norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True,foreach=False))
            for g in opt.param_groups:g["lr"]=lr(cfg,expected["stop"])
            opt.step();torch.cuda.synchronize(device)
            row=dict(group=group,**expected,loss=sum(numer.values())/denom,grad_norm=norm,
                     lr=opt.param_groups[0]["lr"],lane_loss_sums=dict(numer),
                     lane_label_counts=dict(counts),update_wall_seconds=time.monotonic()-t,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            records.append(row);print(json.dumps(row),flush=True)
            # Avoid losing first real update; keep periodic and final checkpoints.
            if group in (1,19,39,args.max_groups):
                save(args.out,identity,net,opt,stream,group,records)
            atomic_json(args.out/"PROGRESS.json",dict(records=records,session_wall_seconds=time.monotonic()-started,
                        complete_10m=group==39,topology="single CUDA GPU",distributed_qualified=False))
    print("Return control to operator; disconnect/delete Colab runtime to stop billing.",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--base",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--wall-seconds",type=int)
    p.add_argument("--max-groups",type=int,default=1)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test(torch.device(args.device));return
    if not (args.execute and args.base and args.out and args.wall_seconds and 0<args.wall_seconds<=5400):
        p.error("explicit --execute, --base, --out and 1..5400 --wall-seconds required")
    main_run(args)


if __name__=="__main__":
    main()
