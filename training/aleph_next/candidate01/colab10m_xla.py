"""Single-device TPU qualification pilot for the frozen direct Aleph A.

CPU and TPU share the same static-shape math. No runtime provisioning or
billing control is performed by this script. The operator must disconnect
the runtime at the externally enforced spend/time cap.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import time
import uuid
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from colab10m import PIN, RECEIPT, RECIPE, RECIPE_SHA, collect_group, fixture, rotary
from packing import reference_logits, sdpa_mask, token_loss_sum
from production import HERE, digest, read
from restart import Aleph, adam, lr, restore_rng, rng_state, schedule
from snapshot import MixedStream
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

FIELDS = ("input_ids", "position_ids", "segment_ids", "labels_at_query")
CHECKPOINT_SCHEMA = "aleph.direct-a.xla.single.v1"


class Backend:
    def __init__(self, kind):
        self.kind = kind
        self.xla = self.xm = self.metrics = None
        if kind == "cpu":
            self.device = torch.device("cpu")
            self.identity = {"backend": "cpu", "world_size": 1}
            return
        if any(os.environ.get(k, "").lower() not in ("", "0", "false") for k in
               ("XLA_USE_BF16", "XLA_DOWNCAST_BF16", "XLA_USE_SPMD")):
            raise RuntimeError("global BF16 conversion and SPMD are forbidden")
        expected = "CPU" if kind == "xla-cpu" else "TPU"
        os.environ.setdefault("PJRT_DEVICE", expected)
        if os.environ["PJRT_DEVICE"] != expected:
            raise RuntimeError("PJRT_DEVICE differs from requested backend")
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as met
        import torch_xla.runtime as xr

        if xr.device_type() != expected or xr.world_size() != 1 or torch_xla.device_count() != 1:
            raise RuntimeError("require exactly one addressable device and one process")
        self.xla, self.xm, self.metrics = torch_xla, xm, met
        self.device = torch_xla.device()
        hw = xm.xla_device_hw(self.device)
        self.identity = {
            "backend": kind, "hardware": hw, "world_size": 1,
            "torch_xla": str(torch_xla.__version__),
            "memory_limit": (int(xm.get_memory_info(self.device)["bytes_limit"])
                             if kind == "xla" else None),
        }

    def sync(self):
        if self.xla is not None:
            self.xla.sync(wait=True)

    def memory(self):
        return {} if self.kind != "xla" else {
            k: int(v) for k, v in self.xm.get_memory_info(self.device).items()
        }

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.xm is not None:
            self.xm.set_rng_state(seed, device=str(self.device))

    def get_rng(self):
        return {"host": rng_state(), "xla": None if self.xm is None else
                int(self.xm.get_rng_state(device=str(self.device)))}

    def set_rng(self, state):
        restore_rng(state["host"])
        if self.xm is not None:
            self.xm.set_rng_state(state["xla"], device=str(self.device))


def tensors(pack, device, context):
    # All packs retain their original padded length, including the final update.
    if any(len(pack[k]) != context for k in FIELDS):
        raise ValueError("unexpected static shape")
    return tuple(torch.tensor(pack[k], dtype=torch.long, device=device)[None] for k in FIELDS)


def dense(x, w, dtype):
    # Casts participate in autograd; the unique master Parameter stays FP32.
    return F.linear(x.to(dtype), w.to(dtype))


def norm(x, layer):
    # Native Aleph residuals are FP32; match its default RMSNorm epsilon.
    x = x.float()
    eps = layer.eps if layer.eps is not None else torch.finfo(torch.float32).eps
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * layer.weight.float()


def block_forward(block, h, pos, mask, dtype):
    b, t, d = h.shape
    z = norm(h, block.n1)
    q = rotary(dense(z, block.q.weight, dtype).view(b,t,block.h,block.k).transpose(1,2), pos)
    k = rotary(dense(z, block.kp.weight, dtype).view(b,t,block.kv,block.k).transpose(1,2), pos)
    v = dense(z, block.v.weight, dtype).view(b,t,block.kv,block.k).transpose(1,2)
    # Explicit GQA expansion and matmuls avoid reliance on a CUDA-only fused kernel.
    k = k.repeat_interleave(block.h // block.kv, dim=1)
    v = v.repeat_interleave(block.h // block.kv, dim=1)
    scores = (q @ k.transpose(-1,-2)).float() * (block.k ** -0.5)
    probs = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1).to(dtype)
    a = (probs @ v).transpose(1,2).reshape(b,t,d)
    h = h + dense(a, block.o.weight, dtype).float()
    z = norm(h, block.n2)
    g = F.silu(dense(z, block.gate.weight, dtype))
    u = dense(z, block.up.weight, dtype)
    return h + dense(g*u, block.down.weight, dtype).float()


class CheckpointVisit(torch.nn.Module):
    def __init__(self, block, dtype):
        super().__init__()
        self.block = block
        self.dtype = dtype

    def forward(self, h, pos, mask):
        return block_forward(self.block, h, pos, mask, self.dtype)


def logits(net, ids, pos, seg, dtype, remat=True):
    h = net.embed(ids).float()
    mask = sdpa_mask(seg)
    for i in schedule(2,1):
        b = net.blocks[i]
        if remat:
            # No random operators occur in the frozen block. Bind the actual
            # shared block as an argument to avoid late-binding closure errors.
            if h.device.type == "xla":
                from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint

                # XLA 2.9 requires its own reentrant implementation, which
                # inserts optimization barriers. Use a bound module method so
                # the barrier sees the block weights as well as input tensors.
                visit = CheckpointVisit(b, dtype)
                h = xla_checkpoint(visit.forward, h, pos, mask,
                                   use_reentrant=True, preserve_rng_state=False)
            else:
                h = checkpoint(block_forward, b, h, pos, mask, dtype,
                               use_reentrant=False, preserve_rng_state=False)
        else:
            h = block_forward(b, h, pos, mask, dtype)
    return dense(norm(h, net.norm), net.embed.weight, dtype).float()


def loss_sum(z, labels):
    # Fixed-shape gather, including all-masked inputs; no nonzero/boolean indexing.
    valid = labels != -100
    safe = torch.where(valid, labels, torch.zeros_like(labels))
    correct = z.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    each = torch.logsumexp(z, dim=-1) - correct
    return torch.where(valid, each, torch.zeros_like(each)).sum(dtype=torch.float32)


class FP32AdamW:
    """Static tensor-scalar AdamW update with the exact frozen decay groups."""
    def __init__(self, net, cfg, backend):
        self.named = list(net.named_parameters())
        if len({id(p) for _, p in self.named}) != len(self.named):
            raise ValueError("duplicate optimizer identity")
        self.cfg = cfg["optimizer"]
        self.backend = backend
        self.moments = {}
        self.step = 0
        for name, p in self.named:
            if p.dtype != torch.float32:
                raise ValueError("master parameter is not FP32")
            self.moments[name] = (torch.zeros_like(p), torch.zeros_like(p))
        backend.sync()

    def zero(self):
        for _, p in self.named:
            p.grad = None

    @torch.no_grad()
    def update(self, rate):
        ps = [p for _, p in self.named]
        if any(p.grad is None or p.grad.dtype != torch.float32 for p in ps):
            raise ValueError("missing or non-FP32 gradient")
        b1, b2, eps = self.cfg["beta1"], self.cfg["beta2"], self.cfg["eps"]
        grad_norm = torch.stack([p.grad.float().square().sum() for p in ps]).sum().sqrt()
        self.backend.sync()
        n = float(grad_norm.detach().cpu())
        if not math.isfinite(n):
            raise FloatingPointError("nonfinite global gradient norm")
        t = self.step + 1
        # Device scalar inputs, rather than Python literals that vary per group.
        rate, bc1, bc2, clip = torch.tensor(
            [rate, 1-b1**t, math.sqrt(1-b2**t),
             min(1., self.cfg["clip_global_norm"]/(n+1e-6))],
            dtype=torch.float32, device=ps[0].device,
        ).unbind()
        finite = []
        for name, p in self.named:
            m, v = self.moments[name]
            g = p.grad * clip
            m.lerp_(g, 1-b1)
            v.mul_(b2).addcmul_(g,g,value=1-b2)
            decay = self.cfg["weight_decay_matrices"] if p.ndim >= 2 else self.cfg["weight_decay_norms"]
            p.mul_(1-rate*decay)
            p.add_(-(rate/bc1)*m/(v.sqrt()/bc2+eps))
            finite.append(torch.isfinite(p).all() & torch.isfinite(m).all() & torch.isfinite(v).all())
        ok = torch.stack(finite).all()
        self.backend.sync()
        if not bool(ok.cpu()):
            raise FloatingPointError("nonfinite update; no checkpoint published")
        self.step = t
        self.zero()
        return n


def mini_model():
    return Aleph(width=16, vocab=64, heads=4, kv=1, core_ff=70)


def static_fixture():
    result = []
    for p in fixture():
        q = copy.deepcopy(p)
        for k, val in zip(FIELDS, (0,0,-1,-100), strict=True):
            q[k] += [val]*(8-len(q[k]))
        result.append(q)
    return result


def accumulate(net, packs, backend, dtype, remat=True):
    count = sum(v != -100 for p in packs for v in p["labels_at_query"])
    if count <= 0:
        raise ValueError("zero global labels")
    den = torch.tensor(float(count), device=backend.device, dtype=torch.float32)
    losses = []
    for p in packs:
        ids,pos,seg,y = tensors(p,backend.device,len(p["input_ids"]))
        value = loss_sum(logits(net,ids,pos,seg,dtype,remat),y)
        (value/den).backward()
        losses.append(value.detach())
        backend.sync()
    return torch.stack(losses).sum(), count


def atomic_json(path, obj):
    tmp = path.with_name(path.name+".partial")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj,f,indent=2,allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)


def save_checkpoint(out, identity, net, opt, backend, cursor, records):
    """Stream master and moment tensors to host in per-parameter shards."""
    backend.sync()
    nbytes = sum(p.numel()*12 for _,p in opt.named)
    if shutil.disk_usage(out).free < nbytes+64*1024*1024:
        raise RuntimeError("insufficient storage for another complete checkpoint")
    tag = f"group_{opt.step:03d}_{uuid.uuid4().hex[:12]}"
    partial = out/(tag+".partial")
    partial.mkdir()
    entries = []
    for i,(name,p) in enumerate(opt.named):
        m,v = opt.moments[name]
        file = partial/f"p{i:04d}.pt"
        with file.open("wb") as f:
            torch.save({"weight":p.detach().cpu(),"m":m.cpu(),"v":v.cpu()},f)
            f.flush()
            os.fsync(f.fileno())
        entries.append({"name":name,"shape":list(p.shape),"file":file.name,"sha256":digest(file)})
    meta = partial/"state.pt"
    with meta.open("wb") as f:
        torch.save({"rng":backend.get_rng(),"cursor":cursor,"records":records},f)
        f.flush()
        os.fsync(f.fileno())
    manifest = {"schema":CHECKPOINT_SCHEMA,"identity":identity,"step":opt.step,
                "parameters":entries,"state_sha256":digest(meta)}
    atomic_json(partial/"manifest.json",manifest)
    # Publish only after all contents have been written and verified.
    for e in entries:
        if digest(partial/e["file"])!=e["sha256"]:
            raise ValueError("checkpoint shard changed")
    os.replace(partial,out/tag)
    atomic_json(out/"LATEST.json",{"directory":tag,"sha256":digest(out/tag/"manifest.json")})
    return tag


@torch.no_grad()
def load_checkpoint(out, identity, net, opt, backend):
    pointer = read(out/"LATEST.json")
    folder = (out/pointer["directory"]).resolve()
    if folder.parent!=out.resolve() or folder.name.endswith(".partial"):
        raise ValueError("invalid checkpoint pointer")
    manifest = read(folder/"manifest.json")
    if digest(folder/"manifest.json")!=pointer["sha256"] or manifest["schema"]!=CHECKPOINT_SCHEMA:
        raise ValueError("invalid checkpoint manifest")
    if manifest["identity"]!=identity:
        raise ValueError("checkpoint identity/topology differs")
    entries = manifest["parameters"]
    if len(entries)!=len(opt.named):
        raise ValueError("checkpoint parameter count differs")
    if digest(folder/"state.pt")!=manifest["state_sha256"]:
        raise ValueError("checkpoint state changed")
    # Verify all hashes before modifying any parameter.
    for i,(e,(name,p)) in enumerate(zip(entries,opt.named,strict=True)):
        if e["name"]!=name or e["shape"]!=list(p.shape) or e["file"]!=f"p{i:04d}.pt":
            raise ValueError("checkpoint geometry changed")
        if digest(folder/e["file"])!=e["sha256"]:
            raise ValueError("checkpoint shard changed")
    for e,(name,p) in zip(entries,opt.named,strict=True):
        blob = torch.load(folder/e["file"],map_location="cpu",weights_only=True)
        m,v = opt.moments[name]
        for target,key in ((p,"weight"),(m,"m"),(v,"v")):
            if blob[key].dtype!=torch.float32 or blob[key].shape!=target.shape:
                raise ValueError("checkpoint tensor mismatch")
            target.copy_(blob[key].to(backend.device))
        backend.sync()
        del blob
    state = torch.load(folder/"state.pt",map_location="cpu",weights_only=True)
    opt.step = manifest["step"]
    opt.zero()
    backend.set_rng(state["rng"])
    return state


def self_test(backend, dtype):
    import tempfile

    cfg = read(RECIPE)
    backend.seed(137)
    native = mini_model()
    model = copy.deepcopy(native).to(backend.device)
    packs = static_fixture()
    count = sum(v != -100 for p in packs for v in p["labels_at_query"])
    reference = adam(native,cfg)
    current = FP32AdamW(model,cfg,backend)
    # Predeclared tolerances: FP32 math vs native SDPA, or BF16 vs CPU FP32.
    atol,rtol,grad_rms_limit = (2e-4,2e-4,1e-4) if dtype==torch.float32 else (.12,.10,.12)
    loss_limit = 2e-4 if dtype==torch.float32 else .04
    ref_sum = 0.
    for p in packs:
        z = reference_logits(native,p)
        num,_ = token_loss_sum(z,p["labels_at_query"])
        (num/count).backward()
        ref_sum += float(num.detach())
        with torch.no_grad():
            ids,pos,seg,y = tensors(p,backend.device,8)
            got = logits(model,ids,pos,seg,dtype,False).cpu()
            valid = torch.tensor(p["labels_at_query"])!=-100
            torch.testing.assert_close(got[0,valid],z.detach()[0,valid],atol=atol,rtol=rtol)
    got_sum,got_count = accumulate(model,packs,backend,dtype)
    assert got_count==count
    measured = float(got_sum.cpu())
    assert abs(measured-ref_sum)/count<=loss_limit
    error=scale=0.
    max_error=0.
    for (name,a),(other,b) in zip(native.named_parameters(),model.named_parameters(),strict=True):
        assert name==other and b.grad is not None
        gb=b.grad.cpu()
        assert torch.isfinite(gb).all()
        error+=float((a.grad-gb).square().sum())
        scale+=float(a.grad.square().sum())
        max_error=max(max_error,float((a.grad-gb).abs().max()))
    rel=math.sqrt(error/max(scale,1e-30))
    if rel>grad_rms_limit:
        raise ValueError(f"gradient relative L2 error {rel}")
    # Check both logits and the gradient contribution of the second document.
    p = copy.deepcopy(packs[0])
    p["labels_at_query"][:3]=[-100]*3
    q = dict(p,input_ids=[10,11,12,*p["input_ids"][3:]])
    def isolated(pack):
        model.zero_grad(set_to_none=True)
        ids,pos,seg,y=tensors(pack,backend.device,8)
        z=logits(model,ids,pos,seg,dtype,False)
        loss_sum(z,y).backward()
        backend.sync()
        return z.detach().cpu()[0,3:6], [g.grad.detach().cpu() for g in model.parameters()]
    left,gl=isolated(p)
    right,gr=isolated(q)
    torch.testing.assert_close(left,right,atol=2e-5,rtol=2e-5)
    for a,b in zip(gl,gr,strict=True):
        torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
    model.zero_grad(set_to_none=True)
    empty=packs[-1]
    ids,pos,seg,y=tensors(empty,backend.device,8)
    zero=loss_sum(logits(model,ids,pos,seg,dtype,True),y)
    zero.backward()
    backend.sync()
    assert float(zero.cpu())==0.
    assert all(int(torch.count_nonzero(p.grad.cpu()))==0 for p in model.parameters())
    model.zero_grad(set_to_none=True)
    accumulate(model,packs,backend,dtype)
    rate=lr(cfg,262144)
    torch.nn.utils.clip_grad_norm_(native.parameters(),1.)
    for g in reference.param_groups:
        g["lr"]=rate
    reference.step()
    current.update(rate)
    # FP32 optimizer parity uses the identical reference AdamW update.
    if dtype==torch.float32:
        for a,b in zip(native.parameters(),model.parameters(),strict=True):
            torch.testing.assert_close(a,b.cpu(),atol=3e-6,rtol=3e-6)
        for (_,b),a in zip(current.named,native.parameters(),strict=True):
            name=next(n for n,p in current.named if p is b)
            m,v=current.moments[name]
            torch.testing.assert_close(m.cpu(),reference.state[a]["exp_avg"],atol=3e-6,rtol=3e-6)
            torch.testing.assert_close(v.cpu(),reference.state[a]["exp_avg_sq"],atol=3e-6,rtol=3e-6)
    # Execute, save, advance, restore and repeat an identical update.
    def state():
        return [(p.detach().cpu().clone(),*(v.cpu().clone() for v in current.moments[n]))
                for n,p in current.named]
    with tempfile.TemporaryDirectory() as folder:
        out=Path(folder)
        ident={"fixture":True,**backend.identity}
        save_checkpoint(out,ident,model,current,backend,{"fixture_cursor":1},[{"group":1}])
        accumulate(model,packs,backend,dtype)
        current.update(lr(cfg,524288))
        expected=state()
        restored=load_checkpoint(out,ident,model,current,backend)
        assert restored["cursor"]=={"fixture_cursor":1} and current.step==1
        accumulate(model,packs,backend,dtype)
        current.update(lr(cfg,524288))
        for a,b in zip(expected,state(),strict=True):
            for x,y in zip(a,b,strict=True):
                torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-6)
    result={"backend":backend.identity,"dtype":str(dtype),"labels":count,
            "reference_loss":ref_sum/count,"static_loss":measured/count,
            "max_gradient_abs_error":max_error,"gradient_relative_l2":rel,
            "gradient_relative_l2_limit":grad_rms_limit,"logit_tolerance":[atol,rtol],
            "isolated_document_gradient":True,"all_masked_zero_gradient":True,
            "unequal_label_counts":True,"fixture_restart":True,"fullsize_steps":0}
    print(json.dumps(result),flush=True)
    return result


def run(args,backend):
    started=time.monotonic()
    if backend.kind!="xla":
        raise RuntimeError("full-size execution requires TPU")
    if backend.identity["memory_limit"]<28_000_000_000:
        raise RuntimeError("need a 32 GB HBM class TPU")
    if digest(RECIPE)!=RECIPE_SHA or digest(args.base/"index.json")!=PIN:
        raise ValueError("frozen recipe/index differs")
    cfg=read(RECIPE)
    groups=read(RECEIPT)["bounded_replay"]["steps"]
    if len(groups)!=39:
        raise ValueError("unexpected frozen groups")
    if args.resume:
        if not (args.out/"LATEST.json").is_file():
            raise ValueError("no completed checkpoint")
    else:
        args.out.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(args.out).free<13_000_000_000:
        raise RuntimeError("need capacity for at least one full checkpoint generation")
    parity=self_test(backend,torch.bfloat16)
    atomic_json(args.out/"TPU_FIXTURE.json",parity)
    backend.seed(137)
    a=cfg["construction"]
    net=Aleph(a["width"],a["vocab"],heads=a["query_heads"],kv=a["kv_heads"],core_ff=a["core_ff"])
    assert sum(p.numel() for p in net.parameters())==999352320
    assert list(schedule(2,1))==[0,*range(1,11),*range(1,11),11,12,13]
    net=net.to(backend.device)
    opt=FP32AdamW(net,cfg,backend)
    identity={"recipe":RECIPE_SHA,"index":PIN,"receipt":digest(RECEIPT),
              "code":digest(Path(__file__)),"cuda_reference":digest(HERE/"colab10m.py"),
              "native":digest(HERE.parents[1]/"aleph_rlt_placement/gym.py"),
              "packer":digest(HERE.parents[1]/"aleph_rlt_placement/packing.py"),
              "torch":str(torch.__version__),"microbatch":1,"context":2048,
              "precision":"explicit-bf16-dense-fp32-master",**backend.identity}
    cursor=None
    records=[]
    if args.resume:
        ck=load_checkpoint(args.out,identity,net,opt,backend)
        cursor,records=ck["cursor"],ck["records"]
        assert opt.step==len(records) and records[-1]["stop"]==groups[opt.step-1]["stop"]
        if args.max_groups<=opt.step:
            raise ValueError("resume target must exceed completed step")
    planned = {g for g in (1,19,39,args.max_groups) if opt.step<g<=args.max_groups}
    required = 13_000_000_000 * max(len(planned), 2 if not args.resume else 1)
    if shutil.disk_usage(args.out).free<required:
        raise RuntimeError(f"need {required} free bytes for planned checkpoint generations")
    atomic_json(args.out/"IDENTITY.json",identity)
    durable_step = opt.step
    last_cursor = cursor
    train_deadline = started+args.wall_seconds-args.checkpoint_reserve_seconds
    with ExitStack() as stack:
        stream=MixedStream(args.base,args.base/"index.json",PIN,stack,cursor)
        assert sum(stream.totals.values())==(groups[opt.step-1]["stop"] if opt.step else 0)
        for i in range(opt.step,args.max_groups):
            if time.monotonic()>=train_deadline:
                break
            t=time.monotonic()
            expected=groups[i]
            packs,denom=collect_group(stream,expected)
            den=torch.tensor(float(denom),device=backend.device,dtype=torch.float32)
            opt.zero()
            totals=defaultdict(list)
            counts=defaultdict(int)
            complete=True
            micro_times=[]
            for lane,p,k in packs:
                if time.monotonic()>=train_deadline:
                    complete=False
                    break
                ts=time.monotonic()
                ids,pos,seg,y=tensors(p,backend.device,2048)
                numerator=loss_sum(logits(net,ids,pos,seg,torch.bfloat16,True),y)
                (numerator/den).backward()
                totals[lane].append(numerator.detach())
                counts[lane]+=k
                # Flush each microbatch so 128 accumulated graphs cannot grow
                # without bound. Do not materialize logits or gradients on host.
                backend.sync()
                micro_times.append(time.monotonic()-ts)
            if not complete:
                opt.zero()
                backend.sync()
                print("time cap: discarded incomplete update; resume last published checkpoint",flush=True)
                break
            numer={k:float(torch.stack(v).sum().cpu()) for k,v in totals.items()}
            if sum(counts.values())!=denom or not all(math.isfinite(v) for v in numer.values()):
                raise FloatingPointError("loss or valid-label accounting failed")
            rate=lr(cfg,expected["stop"])
            grad_norm=opt.update(rate)
            elapsed=time.monotonic()-t
            row={"group":i+1,**expected,"loss":sum(numer.values())/denom,
                 "gradient_norm":grad_norm,"lr":rate,"lane_loss_sums":numer,
                 "lane_label_counts":dict(counts),"update_wall_seconds":elapsed,
                 "first_microbatch_seconds":micro_times[0],
                 "median_microbatch_seconds":float(np.median(micro_times)),
                 "useful_positions_per_second":expected["positions"]/elapsed,
                 "hbm":backend.memory()}
            records.append(row)
            last_cursor=stream.checkpoint()
            print(json.dumps(row),flush=True)
            if i+1 in (1,19,39,args.max_groups):
                ts=time.monotonic()
                tag=save_checkpoint(args.out,identity,net,opt,backend,last_cursor,records)
                durable_step=opt.step
                atomic_json(args.out/f"CHECKPOINT_{i+1:03d}.json",{
                    "directory":tag,"wall_seconds":time.monotonic()-ts,"group":i+1})
            atomic_json(args.out/"PROGRESS.json",{
                "records":records,"session_wall_seconds":time.monotonic()-started,
                "complete_10m":i+1==39,"durable_group":durable_step,"distributed_qualified":False,
                "production_launch_authorized":False,
            })
            if backend.metrics is not None:
                (args.out/"XLA_METRICS.txt").write_text(backend.metrics.metrics_report())
    if opt.step>durable_step:
        # An interrupted accumulation advanced the live stream; last_cursor is
        # the state immediately after the last successful optimizer update.
        ts=time.monotonic()
        tag=save_checkpoint(args.out,identity,net,opt,backend,last_cursor,records)
        durable_step=opt.step
        atomic_json(args.out/f"CHECKPOINT_{opt.step:03d}.json",{
            "directory":tag,"wall_seconds":time.monotonic()-ts,"group":opt.step})
    atomic_json(args.out/"PROGRESS.json",{
        "records":records,"session_wall_seconds":time.monotonic()-started,
        "complete_10m":durable_step==39,"durable_group":durable_step,
        "distributed_qualified":False,"production_launch_authorized":False})
    print("Stop/delete the Colab runtime externally to stop billing.",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--device",choices=("cpu","xla","xla-cpu"),default="cpu")
    p.add_argument("--bf16",action="store_true",help="also exercise BF16 on CPU fixture")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--base",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--max-groups",type=int,default=1)
    p.add_argument("--wall-seconds",type=int)
    p.add_argument("--checkpoint-reserve-seconds",type=int,default=120)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(1)
    if args.self_test:
        b=Backend(args.device)
        self_test(b,torch.bfloat16 if args.device!="cpu" or args.bf16 else torch.float32)
        return
    if not (args.execute and args.device=="xla" and args.base and args.out and
            args.wall_seconds and 0<=args.checkpoint_reserve_seconds<args.wall_seconds<=5400 and
            1<=args.max_groups<=39):
        p.error("explicit --execute --device xla --base --out --wall-seconds (1..5400) required")
    run(args,Backend("xla"))


if __name__=="__main__":
    main()
