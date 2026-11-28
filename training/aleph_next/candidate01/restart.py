"""CPU acceptance fixture for the direct-A optimizer, token clock and packed checkpoint contract."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from production import HERE, digest, read

sys.path.insert(0, str(HERE.parents[1] / "aleph_rlt_placement"))
from gym import Aleph, schedule
from packing import PackedStream, reference_logits, token_loss_sum

RECIPE = HERE / "RECIPE.json"


def boundaries(cfg):
    size = cfg["stream"]["batch_nonpadding"]
    last = 0
    for stop in cfg["stream"]["checkpoints"]:
        while last < stop:
            last = min(stop, last + size)
            yield last


def lr(cfg, positions):
    opt = cfg["optimizer"]
    if positions < opt["warmup_tokens"]:
        return opt["peak_lr"] * positions / opt["warmup_tokens"]
    if positions <= opt["stable_until_tokens"]:
        return opt["peak_lr"]
    fraction = min(
        1.0, (positions - opt["stable_until_tokens"]) / (opt["cosine_decay_ends_tokens"] - opt["stable_until_tokens"])
    )
    return opt["ending_lr"] + (opt["peak_lr"] - opt["ending_lr"]) * (1 + math.cos(math.pi * fraction)) / 2


def adam(net, cfg):
    opt = cfg["optimizer"]
    params = list(net.parameters())
    groups = [
        {"params": [p for p in params if p.ndim >= 2], "weight_decay": opt["weight_decay_matrices"]},
        {"params": [p for p in params if p.ndim < 2], "weight_decay": opt["weight_decay_norms"]},
    ]
    return torch.optim.AdamW(
        groups, betas=(opt["beta1"], opt["beta2"]), eps=opt["eps"], lr=opt["peak_lr"], foreach=False
    )


def geometry(cfg):
    a = cfg["construction"]
    with torch.device("meta"):
        model = Aleph(a["width"], a["vocab"], heads=a["query_heads"], kv=a["kv_heads"], core_ff=a["core_ff"])
        ids = torch.empty(1, a["target_context"], dtype=torch.long)
        result = model(ids, "direct", 2, 1)
    word = schedule(2, 1)
    count = sum(p.numel() for p in model.parameters())
    if (
        count != a["parameters"]
        or list(word) != [0, *range(1, 11), *range(1, 11), 11, 12, 13]
        or list(result.shape) != [1, 2048, 49152]
    ):
        raise ValueError("geometry changed")
    return dict(
        trainable_parameters=count,
        physical_blocks=len(model.blocks),
        logical_schedule=list(word),
        shared_core_identity=all(model.blocks[word[i]] is model.blocks[word[i + 10]] for i in range(1, 11)),
        output_shape=list(result.shape),
        numerical_fullsize_steps=0,
    )


def state_hash(obj):
    h = hashlib.sha256()

    def visit(value):
        if isinstance(value, torch.Tensor):
            x = value.detach().cpu().contiguous()
            h.update(str((x.dtype, tuple(x.shape))).encode())
            h.update(x.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(key)
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            h.update(str(len(value)).encode())
            for v in value:
                visit(v)
        else:
            h.update(repr(value).encode() + b"\0")

    visit(obj)
    return h.hexdigest()


def rng_state():
    name, vec, n, g, val = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(), numpy=(name, vec.tolist(), n, g, val))


def restore_rng(s):
    random.setstate(s["python"])
    torch.set_rng_state(s["torch"])
    name, vec, n, g, val = s["numpy"]
    np.random.set_state((name, np.array(vec, dtype=np.uint32), n, g, val))


def update(net, opt, stream, step, totals, cfg):
    target = totals[step]
    pack = stream.next_pack(target)
    if pack is None or stream.state["processed"] != target:
        raise ValueError("fixture requires more real training positions")
    # Removing the padded tail leaves every attended source and supervised query unchanged.
    n = pack["nonpadding"]
    batch = {k: (v[:n] if isinstance(v, list) else v) for k, v in pack.items()}
    opt.zero_grad(set_to_none=True)
    z = reference_logits(net, batch, "direct", 2, 1)
    numerator, denom = token_loss_sum(z.float(), batch["labels_at_query"])
    if denom.item() < 1:
        raise ValueError("restart test needs a supervised training batch")
    (numerator / denom).backward()
    norm = float(
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["optimizer"]["clip_global_norm"], error_if_nonfinite=True)
    )
    for g in opt.param_groups:
        g["lr"] = lr(cfg, target)
    opt.step()
    return dict(
        processed=target,
        targets=denom.item(),
        grad_norm=norm,
        batch_sha256=state_hash(batch),
        loss_sum=numerator.detach().item(),
        lr=opt.param_groups[0]["lr"],
    )


def probe(manifest, expected, work):
    cfg = read(RECIPE)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(137)
    torch.manual_seed(137)
    np.random.seed(137)
    totals = (32, 64, 96)
    frozen = {
        "recipe": digest(RECIPE),
        "tokenizer": cfg["tokenizer"]["sha256"],
        "manifest": expected,
        "code": {
            p.name: digest(p)
            for p in (
                Path(__file__),
                HERE.parents[1] / "aleph_rlt_placement/gym.py",
                HERE.parents[1] / "aleph_rlt_placement/packing.py",
            )
        },
        "seed": 137,
        "fixture_width": 16,
        "world_size": 1,
        "backend": "cpu",
        "stops": list(totals),
    }

    def model():
        net = Aleph(width=16, vocab=49152)
        return net, adam(net, cfg)

    net, opt = model()
    work.mkdir(parents=True, exist_ok=True)
    path = work / "step1.pt"
    with PackedStream(manifest, expected, "train") as stream:
        first = update(net, opt, stream, 0, totals, cfg)
        cursor = stream.checkpoint()
        checkpoint = dict(
            identity=frozen,
            step=1,
            processed=32,
            rng=rng_state(),
            cursor=cursor,
            optimizer=opt.state_dict(),
            model=net.state_dict(),
        )
        tmp = path.with_suffix(".partial")
        torch.save(checkpoint, tmp)
        os.replace(tmp, path)
        blob = digest(path)
        expected_trace = [update(net, opt, stream, i, totals, cfg) for i in (1, 2)]
        final_cursor = stream.checkpoint()
    baseline = (state_hash(net.state_dict()), state_hash(opt.state_dict()), state_hash(rng_state()))
    other, nextopt = model()
    if digest(path) != blob:
        raise ValueError("checkpoint bytes changed")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["identity"] != frozen or ck["processed"] != ck["cursor"]["payload"]["state"]["processed"]:
        raise ValueError("checkpoint source/cursor mismatch")
    other.load_state_dict(ck["model"])
    nextopt.load_state_dict(ck["optimizer"])
    restore_rng(ck["rng"])
    with PackedStream(manifest, expected, "train", ck["cursor"]) as stream:
        resumed = [update(other, nextopt, stream, i, totals, cfg) for i in range(ck["step"], 3)]
        if resumed != expected_trace or stream.checkpoint() != final_cursor:
            raise ValueError("batch, optimizer schedule or cursor diverged")
    parity = (state_hash(other.state_dict()), state_hash(nextopt.state_dict()), state_hash(rng_state()))
    if parity != baseline:
        raise ValueError("restart state differs")
    return dict(
        schema="aleph.direct-a.cpu-restart.v1",
        geometry=geometry(cfg),
        contract=frozen,
        exact_cpu_resume=True,
        fixture_dtype="float32",
        checkpoint_sha256=blob,
        terminal_hashes=parity,
        trace=[first, *resumed],
        processed=96,
        checkpoint_update=1,
        terminal_update=3,
        cpu_only=True,
        same_rng_and_optimizer=True,
        accelerator_parity_measured=False,
        ten_b_optimizer_updates=len(list(boundaries(cfg))),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if (a.out / "restart.json").exists():
        p.error("use an empty fixture directory")
    result = probe(a.manifest, a.sha256, a.out)
    (a.out / "restart.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["geometry"]))
