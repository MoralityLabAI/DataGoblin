"""AFG-01: native Direct-A logical-visit cache and draft-only masked suffix.

The target uses the actual frozen training source, not a converted Llama
model. Full-size checkpoint loading is explicit and hash pinned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "training" / "aleph_next" / "candidate01"
SOURCE_COMMIT = "b078ab75274a2f95449b1daf99b80529049663e7"
SOURCE_BLOBS = {
    "training/aleph_rlt_placement/gym.py": "3553cfc77a86ab6957816aaac82565c5da68275e",
    "training/aleph_rlt_placement/packing.py": "cd7b016d1899910a4ca50a16ed7e95cdae43e909",
    "training/aleph_next/candidate01/colab10m_xla.py": "613665a0642e1f82b009fad8cbdc927457f73db2",
    "training/aleph_next/candidate01/colab10m.py": "724728ccf7ed4d745fc61720aeadda392d518f3d",
    "training/aleph_next/candidate01/restart.py": "43d4840b389b6589819c0ba4a53e3f1100e0ed42",
    "training/aleph_next/candidate01/production.py": "3da6d379d0db4a50b47dafa4a6131f894f7baf4e",
    "training/aleph_next/candidate01/RECIPE.json": "aa3932a21cfef9890548924f2521ba425dbf47f3",
}
CHECKPOINT_SHA = "2e40e32da6a6d975fbd7dc7bd56cd440321531e2e755b9bcc215a289c64232a8"
CHECKPOINT_BYTES = 3997421432
SCHEMA = "aleph.direct_a.floodgates.v1"
TARGET_WORD = (0, *range(1, 11), *range(1, 11), 11, 12, 13)
DRAFT_WORD = (0, *range(1, 11), 11, 12, 13)
MODES = ("first", "deep")
BLOCK = 4
MAX_CONTEXT = 2048


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_sources():
    manifest = {}
    for name, expected in SOURCE_BLOBS.items():
        raw = (ROOT / name).read_bytes()
        git_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if git_blob != expected:
            raise ValueError(f"frozen source changed: {name}")
        manifest[name] = hashlib.sha256(raw).hexdigest()
    for name in ("native_cache.py", "PROTOCOL.md"):
        manifest[str((HERE / name).relative_to(ROOT))] = file_sha(HERE / name)
    return manifest


# Validate the actual source before importing the training reference.
SOURCE_MANIFEST = check_sources()
sys.path.insert(0, str(SOURCE))
import colab10m_xla as native  # noqa: E402

if tuple(native.schedule(2, 1)) != TARGET_WORD:
    raise ValueError("Direct A schedule changed")


def digest(x):
    raw = json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def tensor_hash(model):
    # Used only for compact engineering fixtures and adapters.
    h = hashlib.sha256()
    for name, t in model.state_dict().items():
        x = t.detach().cpu().contiguous()
        h.update(str((name, x.dtype, tuple(x.shape))).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def versions(model):
    return tuple((name, id(p), p._version) for name, p in model.named_parameters())


def mini_model(seed=137):
    torch.manual_seed(seed)
    return native.Aleph(width=32, vocab=64, heads=4, kv=1, core_ff=140).eval().requires_grad_(False)


def full_model():
    return native.Aleph(width=2048, vocab=49152, heads=16, kv=4, core_ff=8960)


def geometry():
    with torch.device("meta"):
        model = full_model()
    count = sum(p.numel() for p in model.parameters())
    if count != 999352320 or len(model.blocks) != 14 or len(TARGET_WORD) != 24:
        raise ValueError("full geometry mismatch")
    if any(model.blocks[TARGET_WORD[i]] is not model.blocks[TARGET_WORD[i+10]]
           for i in range(1, 11)):
        raise ValueError("physical weight tying changed")
    return {"parameters": count, "physical_blocks": len(model.blocks),
            "target_word": list(TARGET_WORD), "draft_word": list(DRAFT_WORD),
            "fp32_kv_bytes_per_token": 24*2*4*128*4,
            "bf16_kv_bytes_per_token": 24*2*4*128*2}


def load_native_checkpoint(path):
    from safetensors import safe_open

    path = Path(path).resolve()
    if path.stat().st_size != CHECKPOINT_BYTES or file_sha(path) != CHECKPOINT_SHA:
        raise ValueError("checkpoint is not the pinned native FP32 10M export")
    geometry()
    with torch.device("meta"):
        model = full_model()
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        if set(f.keys()) != set(expected):
            raise ValueError("native state_dict key mismatch; no alias conversion")
        state = {}
        for k in expected:
            v = f.get_tensor(k)
            if tuple(v.shape) != expected[k] or v.dtype != torch.float32:
                raise ValueError(f"native tensor geometry/precision mismatch: {k}")
            state[k] = v
        model.load_state_dict(state, strict=True, assign=True)
    del state
    if sum(p.numel() for p in model.parameters()) != 999352320:
        raise ValueError("native parameter count")
    return model.eval().requires_grad_(False)


@dataclass(frozen=True)
class VisitKV:
    visit: int
    physical: int
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class Cache:
    owner: object
    tokens: tuple[int, ...]
    visits: tuple[VisitKV, ...]

    @property
    def bytes(self):
        return sum(t.numel()*t.element_size() for v in self.visits for t in (v.key, v.value))


class LoRA(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super().__init__()
        self.a = nn.Linear(input_dim, rank, bias=False)
        self.b = nn.Linear(rank, output_dim, bias=False)
        nn.init.zeros_(self.b.weight)
        self.scale = 1./rank

    def forward(self, x):
        return self.b(self.a(x.float()))*self.scale


class BlockAdapter(nn.Module):
    def __init__(self, block, rank):
        super().__init__()
        d = block.q.in_features
        self.q = LoRA(d, d, rank)
        self.v = LoRA(d, block.kv*block.k, rank)
        self.o = LoRA(d, d, rank)


class DraftAdapter(nn.Module):
    def __init__(self, model, rank=4):
        super().__init__()
        if type(rank) is not int or not 1 <= rank <= 32:
            raise ValueError("invalid rank")
        self.rank = rank
        self.mask = nn.Parameter(torch.empty(model.embed.embedding_dim))
        nn.init.normal_(self.mask, std=model.embed.embedding_dim**-.5)
        self.core = nn.ModuleDict({str(i): BlockAdapter(model.blocks[i], rank) for i in range(1, 11)})


def visit_block(block, h, positions, past, dtype, adapter=None):
    """Explicit native training math with offset RoPE and unexpanded GQA KV."""
    b, t, d = h.shape
    z = native.norm(h, block.n1)
    q = native.dense(z, block.q.weight, dtype)
    k = native.dense(z, block.kp.weight, dtype)
    v = native.dense(z, block.v.weight, dtype)
    if adapter is not None:
        q = q + adapter.q(z).to(dtype)
        v = v + adapter.v(z).to(dtype)
    q = native.rotary(q.view(b, t, block.h, block.k).transpose(1, 2), positions)
    k = native.rotary(k.view(b, t, block.kv, block.k).transpose(1, 2), positions)
    v = v.view(b, t, block.kv, block.k).transpose(1, 2)
    old = 0
    if past is not None:
        old = past.key.shape[-2]
        k = torch.cat((past.key, k), dim=-2)
        v = torch.cat((past.value, v), dim=-2)
    keys = k.repeat_interleave(block.h//block.kv, dim=1)
    values = v.repeat_interleave(block.h//block.kv, dim=1)
    query_positions = torch.arange(old, old+t, device=h.device)
    key_positions = torch.arange(old+t, device=h.device)
    mask = key_positions[None, :] <= query_positions[:, None]
    scores = (q @ keys.transpose(-1, -2)).float() * (block.k**-.5)
    probs = torch.softmax(scores.masked_fill(~mask[None, None], float("-inf")), dim=-1).to(dtype)
    a = (probs @ values).transpose(1, 2).reshape(b, t, d)
    update = native.dense(a, block.o.weight, dtype)
    if adapter is not None:
        update = update + adapter.o(a).to(dtype)
    h = h + update.float()
    z = native.norm(h, block.n2)
    g = F.silu(native.dense(z, block.gate.weight, dtype))
    u = native.dense(z, block.up.weight, dtype)
    h = h + native.dense(g*u, block.down.weight, dtype).float()
    return h, k, v


class Decoder:
    """Single trusted decoder instance, one unpadded token stream."""
    def __init__(self, model, dtype=torch.float32):
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("unsupported compute precision")
        if (len(model.blocks) != 14 or any(p.dtype != torch.float32 or p.requires_grad
                                          for p in model.parameters())):
            raise ValueError("require frozen native FP32 parameters")
        self.model = model
        self.dtype = dtype
        self.device = model.embed.weight.device
        self.owner = object()
        self.versions = versions(model)
        self.vocab = model.embed.num_embeddings

    def frozen(self):
        if versions(self.model) != self.versions or any(
            p.requires_grad or p.grad is not None for p in self.model.parameters()
        ):
            raise ValueError("target parameter identity/version changed")

    def ids(self, tokens):
        if not isinstance(tokens, (tuple, list)) or not tokens or any(
            type(t) is not int or not 0 <= t < self.vocab for t in tokens
        ):
            raise ValueError("invalid token IDs")
        return torch.tensor([tokens], device=self.device, dtype=torch.long)

    def validate_cache(self, cache, expected=None):
        if not isinstance(cache, Cache) or cache.owner is not self.owner:
            raise ValueError("foreign cache")
        if expected is not None and tuple(expected) != cache.tokens:
            raise ValueError("stale or wrong-prefix cache")
        n = len(cache.tokens)
        if not 0 <= n <= MAX_CONTEXT or len(cache.visits) != len(TARGET_WORD):
            raise ValueError("invalid cache length or inventory")
        for i, (v, physical) in enumerate(zip(cache.visits, TARGET_WORD, strict=True)):
            b = self.model.blocks[physical]
            shape = (1, b.kv, n, b.k)
            if (v.visit != i or v.physical != physical or
                any(t.shape != shape or t.dtype != self.dtype or t.device != self.device
                    for t in (v.key, v.value))):
                raise ValueError("cache visit/order/shape mismatch")
        return n

    @torch.no_grad()
    def append(self, tokens, cache=None):
        self.frozen()
        ids = self.ids(tokens)
        old = 0 if cache is None else self.validate_cache(cache)
        if old + len(tokens) > MAX_CONTEXT:
            raise ValueError("target context limit")
        pos = torch.arange(old, old+len(tokens), device=self.device)[None]
        h = self.model.embed(ids).float()
        visits = []
        for i, physical in enumerate(TARGET_WORD):
            past = None if cache is None else cache.visits[i]
            h, k, v = visit_block(self.model.blocks[physical], h, pos, past, self.dtype)
            visits.append(VisitKV(i, physical, k, v))
        z = native.dense(native.norm(h, self.model.norm), self.model.embed.weight, self.dtype).float()
        prefix = () if cache is None else cache.tokens
        out = Cache(self.owner, prefix+tuple(tokens), tuple(visits))
        self.validate_cache(out)
        return z, out

    def crop(self, cache, n):
        length = self.validate_cache(cache)
        if type(n) is not int or not 0 <= n <= length:
            raise ValueError("invalid committed crop")
        return Cache(self.owner, cache.tokens[:n], tuple(
            VisitKV(v.visit, v.physical, v.key[:, :, :n].clone(), v.value[:, :, :n].clone())
            for v in cache.visits
        ))

    @torch.no_grad()
    def cold(self, tokens):
        """Independent original training forward, single document, no remat."""
        self.frozen()
        ids = self.ids(tokens)
        if len(tokens) > MAX_CONTEXT:
            raise ValueError("cold context cap")
        pos = torch.arange(len(tokens), device=self.device)[None]
        seg = torch.zeros_like(ids)
        return native.logits(self.model, ids, pos, seg, self.dtype, remat=False)

    def draft(self, anchor, prefix, cache, adapter, k, mode):
        self.frozen()
        old = self.validate_cache(cache, prefix)
        if mode not in MODES or type(k) is not int or not 1 <= k <= BLOCK:
            raise ValueError("invalid draft policy")
        if old+1+k > MAX_CONTEXT:
            raise ValueError("draft context cap")
        ids = self.ids([anchor])
        h = torch.cat((self.model.embed(ids).float(),
                       adapter.mask[None, None].expand(1, k, -1)), dim=1)
        pos = torch.arange(old, old+1+k, device=self.device)[None]
        for physical in DRAFT_WORD:
            if physical == 0:
                source = 0
            elif 1 <= physical <= 10:
                source = physical+(10 if mode=="deep" else 0)
            else:
                source = physical+10
            a = adapter.core[str(physical)] if 1 <= physical <= 10 else None
            h, _, _ = visit_block(self.model.blocks[physical], h, pos, cache.visits[source], self.dtype, a)
        return native.dense(native.norm(h[:, 1:], self.model.norm), self.model.embed.weight, self.dtype).float()

    @torch.no_grad()
    def verify(self, tokens, cache, draft, remaining):
        if len(tokens)<2 or type(remaining) is not int or remaining<1 or not 1<=len(draft)<=BLOCK:
            raise ValueError("invalid verification request")
        self.validate_cache(cache, tokens[:-1])
        z, provisional = self.append([tokens[-1], *draft], cache)
        predictions = z[0].argmax(-1).tolist()
        accepted = 0
        while accepted<len(draft) and draft[accepted]==predictions[accepted]:
            accepted+=1
        emitted = (list(draft[:accepted])+[predictions[accepted]])[:remaining]
        newtokens = list(tokens)+emitted
        keep = len(newtokens)-1
        committed = self.crop(provisional, keep)
        self.validate_cache(committed, newtokens[:-1])
        return newtokens, committed, {"draft":list(draft),"target_predictions":predictions,
             "accepted_prefix":accepted,"emitted":emitted,"cache_before":len(tokens)-1,
             "cache_provisional":len(provisional.tokens),"cache_after":keep,
             "discarded_positions":len(provisional.tokens)-keep,
             "target_block_token_visits":len(TARGET_WORD)*(len(draft)+1),
             "provisional_kv_bytes":provisional.bytes,"committed_kv_bytes":committed.bytes}


class Diagnostics:
    def __init__(self, dtype):
        self.atol, self.rtol = (2e-4, 2e-4) if dtype==torch.float32 else (.12, .10)
        self.n = 0
        self.positions = 0
        self.max_error = 0.
        self.sum_error = 0.

    def compare(self, a, b, *, logits=False):
        if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise AssertionError("invalid parity tensors")
        delta=(a.float()-b.float()).abs()
        self.max_error=max(self.max_error, float(delta.max()))
        self.sum_error+=float(delta.double().sum())
        self.n+=delta.numel()
        if logits:
            self.positions+=a.numel()//a.shape[-1]
            if not torch.equal(a.argmax(-1), b.argmax(-1)):
                raise AssertionError("exact greedy ID mismatch")
        torch.testing.assert_close(a.float(), b.float(), atol=self.atol, rtol=self.rtol)

    def receipt(self):
        return {"compared_values":self.n,"exact_logit_argmax_positions":self.positions,
                "max_abs_error":self.max_error,"mean_abs_error":self.sum_error/max(self.n,1),
                "tolerances":[self.atol,self.rtol],"greedy_mismatches":0}


@torch.no_grad()
def serial(decoder, prefix, n, diag):
    tokens=list(prefix)
    _,cache=decoder.append(tokens[:-1])
    for _ in range(n):
        z,cache=decoder.append([tokens[-1]],cache)
        cold=decoder.cold(tokens)
        diag.compare(z[:, -1],cold[:, -1],logits=True)
        tokens.append(int(z[0,-1].argmax()))
    return tokens


def check_cache(decoder, actual, tokens, diag):
    decoder.validate_cache(actual,tokens)
    _,fresh=decoder.append(list(tokens))
    for a,b in zip(actual.visits,fresh.visits,strict=True):
        diag.compare(a.key,b.key)
        diag.compare(a.value,b.value)


def expect_reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("invalid cache accepted")


def qualify(decoder,prefixes,n=8):
    if not 5<=n<=16:
        raise ValueError("qualification generation limit")
    diag=Diagnostics(decoder.dtype)
    cases=[]
    for prefix in prefixes:
        if not 2<=len(prefix)<=32 or len(prefix)+n+1>MAX_CONTEXT:
            raise ValueError("bounded prefix required")
        z,full=decoder.append(prefix)
        diag.compare(z,decoder.cold(prefix),logits=True)
        # Multi-token append, including nonzero RoPE offset.
        cut=max(1,len(prefix)//2)
        x,c=decoder.append(prefix[:cut])
        y,c=decoder.append(prefix[cut:],c)
        diag.compare(torch.cat((x,y),dim=1),z,logits=True)
        check_cache(decoder,c,prefix,diag)
        # Check prefix causality with a changed suffix.
        changed=list(prefix)
        changed[-1]=(changed[-1]+1)%decoder.vocab
        _,other=decoder.append(changed)
        for a,b in zip(full.visits,other.visits,strict=True):
            diag.compare(a.key[:,:,:-1],b.key[:,:,:-1])
            diag.compare(a.value[:,:,:-1],b.value[:,:,:-1])
        _,cache=decoder.append(prefix[:-1])
        for i in range(1,11):
            if cache.visits[i].key.data_ptr()==cache.visits[i+10].key.data_ptr():
                raise AssertionError("shared physical weights aliased logical K/V")
        ref=serial(decoder,prefix,n,diag)
        trials=[]
        for j in range(BLOCK+1):
            proposed=ref[len(prefix):len(prefix)+BLOCK]
            if j<BLOCK:
                proposed[j]=(proposed[j]+1)%decoder.vocab
            out,kv,r=decoder.verify(prefix,cache,proposed,n)
            if r["accepted_prefix"]!=j or out!=ref[:len(out)]:
                raise AssertionError("forced first mismatch or full-acceptance parity")
            check_cache(decoder,kv,out[:-1],diag)
            zz,_=decoder.append([out[-1]],kv)
            diag.compare(zz[:,-1],decoder.cold(out)[:,-1],logits=True)
            trials.append(r)
        out,kv,r=decoder.verify(prefix,cache,ref[len(prefix):len(prefix)+BLOCK],1)
        if out!=ref[:len(prefix)+1]:
            raise AssertionError("generation cap mismatch")
        check_cache(decoder,kv,out[:-1],diag)
        wrong=list(prefix)
        wrong[0]=(wrong[0]+1)%decoder.vocab
        expect_reject(lambda:decoder.verify(wrong,cache,[0],1))
        foreign=Decoder(decoder.model,decoder.dtype)
        expect_reject(lambda:foreign.append([0],cache))
        bad=Cache(cache.owner,cache.tokens,(cache.visits[1],cache.visits[0],*cache.visits[2:]))
        expect_reject(lambda:decoder.append([0],bad))
        expect_reject(lambda:decoder.verify(prefix,decoder.crop(cache,len(cache.tokens)-1),[0],1))
        cases.append({"prefix":list(prefix),"reference_output":ref[len(prefix):],"forced_trials":trials,
                      "cap_trial":r,"prefix_kv_bytes":cache.bytes})
    return {"precision":str(decoder.dtype),"cases":cases,"parity":diag.receipt(),
            "forced_rejection_and_acceptance":len(cases)*(BLOCK+1),
            "stale_foreign_misordered_rejected":True,"physical_cache_nonalias":True}


def fixture_prefixes(split,n,vocab=64):
    rng=random.Random(f"{SCHEMA}:{split}")
    return [tuple(rng.randrange(vocab) for _ in range(8)) for _ in range(n)]


def fixture_train(decoder,mode,train,steps=48):
    before=tensor_hash(decoder.model)
    torch.manual_seed(837)
    a=DraftAdapter(decoder.model,4)
    initial=tensor_hash(a)
    diag=Diagnostics(decoder.dtype)
    teacher=[serial(decoder,list(p),8,diag) for p in train]
    rng=random.Random(438)
    opt=torch.optim.AdamW(a.parameters(),lr=.01)
    batches=[]
    for _ in range(steps):
        i=rng.randrange(len(teacher))
        cut=rng.randrange(8,13)
        seq=teacher[i]
        _,cache=decoder.append(seq[:cut-1])
        z=decoder.draft(seq[cut-1],seq[:cut-1],cache,a,BLOCK,mode)
        target=torch.tensor(seq[cut:cut+BLOCK],dtype=torch.long)
        loss=F.cross_entropy(z.reshape(-1,decoder.vocab),target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        batches.append([i,cut])
    a.eval().requires_grad_(False)
    if before!=tensor_hash(decoder.model) or initial==tensor_hash(a):
        raise AssertionError("target mutation or unchanged adapter")
    decoder.frozen()
    return a,{"mode":mode,"initial_hash":initial,"weights_hash":tensor_hash(a),
              "target_hash":before,"updates":steps,"parameters":sum(p.numel() for p in a.parameters()),
              "batch_sha256":digest(batches),"last_loss":float(loss.detach()),
              "draft_train_block_token_visits":steps*(BLOCK+1)*len(DRAFT_WORD),
              "teacher_prefix_and_serial_visits":len(train)*(7+8)*len(TARGET_WORD),
              "teacher_cold_reference_visits":len(train)*sum((8+t)*len(TARGET_WORD) for t in range(8)),
              "fit_prefix_recompute_visits":sum((cut-1)*len(TARGET_WORD) for _,cut in batches),
              "teacher_parity":diag.receipt()}


@torch.no_grad()
def fixture_rollout(decoder,a,mode,prefix):
    diag=Diagnostics(decoder.dtype)
    ref=serial(decoder,list(prefix),8,diag)
    tokens=list(prefix)
    _,cache=decoder.append(tokens[:-1])
    rows=[]
    work=Counter()
    before=versions(decoder.model)
    while len(tokens)<len(prefix)+8:
        remaining=len(prefix)+8-len(tokens)
        k=min(BLOCK,remaining)
        z=decoder.draft(tokens[-1],tokens[:-1],cache,a,k,mode)
        guesses=z[0].argmax(-1).tolist()
        tokens,cache,r=decoder.verify(tokens,cache,guesses,remaining)
        if tokens!=ref[:len(tokens)]:
            raise AssertionError("fixture greedy mismatch")
        r["draft_block_token_visits"]=(k+1)*len(DRAFT_WORD)
        rows.append(r)
        work["verify_calls"]+=1
        work["draft_calls"]+=1
        for key in ("target_block_token_visits","draft_block_token_visits","accepted_prefix","discarded_positions"):
            work[key]+=r[key]
        work["emitted_tokens"]+=len(r["emitted"])
        work["provisional_kv_peak_bytes"]=max(work["provisional_kv_peak_bytes"],r["provisional_kv_bytes"])
    if tokens!=ref or versions(decoder.model)!=before:
        raise AssertionError("frozen verifier parity")
    return {"prefix":list(prefix),"mode":mode,"steps":rows,"output":tokens[len(prefix):],
            "reference":ref[len(prefix):],"work":dict(work),"greedy_parity":True,
            "training_ready":False}


def save_fixture(out):
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    model=mini_model()
    d=Decoder(model)
    qualification=qualify(d,[list(p) for p in fixture_prefixes("parity",4)])
    # BF16 is a separately qualified precision. A failure is reported and
    # blocks that precision; it never relaxes the FP32 gate.
    try:
        bf16={"status":"passed",**qualify(Decoder(model,torch.bfloat16),
                                        [list(p) for p in fixture_prefixes("parity",4)])}
    except (AssertionError,RuntimeError,ValueError) as error:
        bf16={"status":"blocked","reason":str(error)[:1500]}
    train=fixture_prefixes("adapter_train",16)
    dev=fixture_prefixes("adapter_dev",8)
    if len(set(train+dev))!=len(train)+len(dev):
        raise ValueError("fixture lineage collision")
    target_hash=tensor_hash(model)
    path=out/"target.pt"
    torch.save(model.state_dict(),path)
    report={"schema":SCHEMA,"status":"native_source_fixture_only","source_manifest":SOURCE_MANIFEST,
            "geometry":geometry(),"checkpoint_loaded":False,"fixture_target_hash":target_hash,
            "target_artifact_sha256":file_sha(path),"fp32":qualification,"bf16":bf16,
            "train_roots_sha256":digest(train),"dev_roots_sha256":digest(dev),"adapters":[]}
    trace=out/"fixture_rollouts.jsonl"
    with trace.open("x",encoding="utf-8") as f:
        for mode in MODES:
            a,info=fixture_train(d,mode,train)
            p=out/f"{mode}-adapter.pt"
            torch.save(a.state_dict(),p)
            info.update(artifact=p.name,artifact_sha256=file_sha(p))
            for prefix in dev:
                r=fixture_rollout(d,a,mode,prefix)
                r.update(schema=SCHEMA+".fixture_rollout",root=digest(["adapter_dev",prefix]),
                         target_hash=target_hash,adapter_hash=info["weights_hash"])
                r["receipt_hash"]=digest(r)
                f.write(json.dumps(r,sort_keys=True,allow_nan=False)+"\n")
            report["adapters"].append(info)
    if len({a["initial_hash"] for a in report["adapters"]})!=1 or len(
        {a["batch_sha256"] for a in report["adapters"]})!=1:
        raise AssertionError("adapter pairing")
    report["trace_sha256"]=file_sha(trace)
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def audit_fixture(out):
    torch.set_num_threads(1)
    r=json.loads((out/"report.json").read_text(encoding="utf-8"))
    if (r["schema"]!=SCHEMA or r["source_manifest"]!=SOURCE_MANIFEST or r["geometry"]!=geometry()
        or r["checkpoint_loaded"] is not False or file_sha(out/"target.pt")!=r["target_artifact_sha256"]
        or file_sha(out/"fixture_rollouts.jsonl")!=r["trace_sha256"]):
        raise ValueError("fixture inventory mismatch")
    model=mini_model()
    model.load_state_dict(torch.load(out/"target.pt",map_location="cpu",weights_only=True))
    if tensor_hash(model)!=r["fixture_target_hash"]:
        raise ValueError("fixture weight mismatch")
    d=Decoder(model)
    if qualify(d,[list(p) for p in fixture_prefixes("parity",4)])!=r["fp32"]:
        raise ValueError("FP32 qualification replay")
    train=fixture_prefixes("adapter_train",16)
    dev=fixture_prefixes("adapter_dev",8)
    if r["train_roots_sha256"]!=digest(train) or r["dev_roots_sha256"]!=digest(dev):
        raise ValueError("fixture split mismatch")
    rows=[json.loads(line) for line in (out/"fixture_rollouts.jsonl").read_text(encoding="utf-8").splitlines()]
    expected=[]
    if [a["mode"] for a in r["adapters"]]!=list(MODES):
        raise ValueError("adapter coverage")
    for info in r["adapters"]:
        mode=info["mode"]
        a,replayed=fixture_train(d,mode,train)
        if any(info[k]!=v for k,v in replayed.items()) or file_sha(out/info["artifact"])!=info["artifact_sha256"]:
            raise ValueError("adapter refit or artifact mismatch")
        a.load_state_dict(torch.load(out/info["artifact"],map_location="cpu",weights_only=True))
        if tensor_hash(a)!=info["weights_hash"]:
            raise ValueError("adapter weight mismatch")
        for prefix in dev:
            obj=fixture_rollout(d,a,mode,prefix)
            obj.update(schema=SCHEMA+".fixture_rollout",root=digest(["adapter_dev",prefix]),
                       target_hash=r["fixture_target_hash"],adapter_hash=info["weights_hash"])
            obj["receipt_hash"]=digest(obj)
            expected.append(obj)
    if expected!=rows:
        raise ValueError("fixture rollout replay")
    return {"status":"audited_native_source_fixture_only","rollouts":len(rows),
            "fp32_parity":r["fp32"]["parity"],"bf16_status":r["bf16"]["status"],
            "full_checkpoint_loaded":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture-out",type=Path)
    p.add_argument("--audit",type=Path)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--ids-json",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--precision",choices=("fp32","bf16"),default="fp32")
    p.add_argument("--max-new",type=int,default=8)
    p.add_argument("--threads",type=int,default=1)
    a=p.parse_args()
    if not 1<=a.threads<=8:
        p.error("thread cap")
    torch.set_num_threads(a.threads)
    if a.fixture_out:
        r=save_fixture(a.fixture_out)
        print(json.dumps({"status":r["status"],"fp32":r["fp32"]["parity"],
                          "bf16_status":r["bf16"]["status"],"geometry":r["geometry"]},sort_keys=True))
    elif a.audit:
        print(json.dumps(audit_fixture(a.audit),sort_keys=True))
    elif a.checkpoint and a.ids_json and a.out:
        if a.out.exists():
            raise ValueError("fresh report required")
        prefixes=json.loads(a.ids_json.read_text(encoding="utf-8"))
        if not isinstance(prefixes,list) or not 1<=len(prefixes)<=4 or any(
            not isinstance(p,list) or not 2<=len(p)<=32 for p in prefixes
        ):
            raise ValueError("bounded prompt panel (1..4 prefixes, 2..32 tokens)")
        start=time.monotonic()
        m=load_native_checkpoint(a.checkpoint)
        dtype=torch.float32 if a.precision=="fp32" else torch.bfloat16
        d=Decoder(m,dtype)
        result=qualify(d,prefixes,a.max_new)
        d.frozen()
        r={"schema":SCHEMA,"status":"native_checkpoint_qualification_passed",
           "checkpoint_sha256":CHECKPOINT_SHA,"source_manifest":SOURCE_MANIFEST,
           "geometry":geometry(),"qualification":result,"elapsed_seconds":time.monotonic()-start,
           "threads":a.threads,"torch":str(torch.__version__),"adapter_fitting":False}
        with a.out.open("x",encoding="utf-8") as f:
            json.dump(r,f,indent=2,allow_nan=False)
        print(json.dumps({"status":r["status"],"parity":result["parity"]}))
    else:
        p.error("choose --fixture-out, --audit, or --checkpoint --ids-json --out")


if __name__=="__main__":
    main()
