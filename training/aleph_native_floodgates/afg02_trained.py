"""AFG-02: train a native Direct-A algorithmic target and evaluate masked drafters."""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import native_cache as N
import torch
from torch.nn import functional as F

SCHEMA = "aleph.direct_a.afg02.v1"
PREFIX = 8
NEW = 8
PANELS = ("cal", "dev", "shift", "trace_train")
POLICIES = ("serial", "repeat", "first_initial", "deep_initial", "first_trained", "deep_trained")
DEFAULT = dict(seeds=[137, 163], base_n=384, adapter_n=64, eval_n=16,
               base_steps=320, batch=16, adapter_steps=128)


def manifest():
    return {**N.SOURCE_MANIFEST, "afg02_trained.py": N.file_sha(Path(__file__)),
            "AFG02_PROTOCOL.md": N.file_sha(Path(__file__).with_name("AFG02_PROTOCOL.md"))}


def validate(cfg):
    limits = {"base_n": (8, 384), "adapter_n": (8, 64), "eval_n": (4, 16),
              "base_steps": (1, 320), "batch": (1, 16), "adapter_steps": (1, 128)}
    if set(cfg) != set(limits) | {"seeds"}:
        raise ValueError("unexpected config")
    if any(type(cfg[k]) is not int or not lo <= cfg[k] <= hi for k, (lo, hi) in limits.items()):
        raise ValueError("resource cap")
    if cfg["seeds"] not in ([137], [137, 163]):
        raise ValueError("unregistered seeds")


def worlds(split, n):
    rng = random.Random(f"{SCHEMA}:{split}")
    rows = []
    for _ in range(n):
        marker = 2 if split == "shift" else rng.randrange(2)
        prefix = [marker, *[rng.randrange(3, 19) for _ in range(PREFIX-1)]]
        period = marker+1
        truth = [prefix[PREFIX-period+i % period] for i in range(NEW)]
        rows.append({"root": N.digest([SCHEMA, prefix]), "prefix": prefix, "truth": truth})
    return rows


def partitions(cfg):
    sets = {"base_train": worlds("base_train", cfg["base_n"]),
            "adapter_train": worlds("adapter_train", cfg["adapter_n"])}
    sets.update({s: worlds(s, cfg["eval_n"]) for s in PANELS})
    roots = [r["root"] for rs in sets.values() for r in rs]
    if len(set(roots)) != len(roots):
        raise ValueError("prefix collision")
    return sets


def split_hashes(sets):
    return {k: N.digest(v) for k, v in sets.items()}


def train_target(seed, ws, cfg):
    m = N.mini_model(seed).train().requires_grad_(True)
    before = N.tensor_hash(m)
    opt = torch.optim.AdamW(m.parameters(), lr=.0015)
    rng = random.Random(seed+4200)
    draws = []
    data = torch.tensor([w["prefix"]+w["truth"] for w in ws])
    positions = torch.arange(PREFIX+NEW-1)[None].expand(cfg["batch"], -1)
    seg = torch.zeros_like(positions)
    for _ in range(cfg["base_steps"]):
        ids = [rng.randrange(len(ws)) for _ in range(cfg["batch"])]
        draws.append(ids)
        seq = data[ids]
        z = N.native.logits(m, seq[:, :-1], positions, seg, torch.float32, remat=False)
        loss = F.cross_entropy(z[:, PREFIX-1:].reshape(-1, 64), seq[:, PREFIX:].reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite target loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
        opt.step()
    m.zero_grad(set_to_none=True)
    m.eval().requires_grad_(False)
    after = N.tensor_hash(m)
    if after == before:
        raise AssertionError("target weights unchanged")
    return m, {"initial_hash": before, "weights_hash": after, "updates": cfg["base_steps"],
               "parameters": sum(p.numel() for p in m.parameters()), "batch_sha256": N.digest(draws),
               "train_token_block_visits": cfg["base_steps"]*cfg["batch"]*(PREFIX+NEW-1)*len(N.TARGET_WORD),
               "last_loss": float(loss.detach())}


@torch.no_grad()
def reference(d, ws):
    refs = {}
    diag = N.Diagnostics(d.dtype)
    forced = 0
    free = 0
    exact = 0
    for w in ws:
        seq = w["prefix"]+w["truth"]
        z = d.cold(seq[:-1])
        forced += sum(a == b for a, b in zip(z[0, PREFIX-1:].argmax(-1).tolist(), w["truth"], strict=True))
        out = N.serial(d, w["prefix"], NEW, diag)
        refs[w["root"]] = out
        free += sum(a == b for a, b in zip(out[PREFIX:], w["truth"], strict=True))
        exact += out[PREFIX:] == w["truth"]
    n = len(ws)*NEW
    return refs, {"teacher_forced_accuracy": forced/n, "free_running_accuracy": free/n,
                  "exact_sequence_accuracy": exact/len(ws), "parity": diag.receipt(),
                  "reference_token_block_visits": len(ws)*len(N.TARGET_WORD)*(
                      PREFIX+NEW-1+PREFIX-1+NEW+sum(PREFIX+i for i in range(NEW)))}


def initial_adapter(model):
    torch.manual_seed(837)
    return N.DraftAdapter(model, 4).eval().requires_grad_(False)


def repeat(tokens, k):
    # Discover the shortest repeated suffix from the committed prefix.
    p = next((p for p in range(1, 5) if len(tokens) >= 2*p and tokens[-p:] == tokens[-2*p:-p]), 1)
    return [tokens[-p+i % p] for i in range(k)]


def tariff(work):
    return 4*work["serialized_block_calls"]+work["target_token_block_visits"]+work["draft_token_block_visits"]


@torch.no_grad()
def rollout(d, adapter, policy, k, w, ref):
    tokens = list(w["prefix"])
    _, cache = d.append(tokens[:-1])
    steps = []
    work = Counter()
    work["prefill_token_block_visits"] = (PREFIX-1)*len(N.TARGET_WORD)
    started = time.perf_counter()
    cpu_start = time.process_time()
    while len(tokens) < PREFIX+NEW:
        remaining = PREFIX+NEW-len(tokens)
        if policy == "serial":
            z, cache = d.append([tokens[-1]], cache)
            emitted = [int(z[0, -1].argmax())]
            tokens.extend(emitted)
            r = {"draft": [], "accepted_prefix": 0, "emitted": emitted, "discarded_positions": 0,
                 "provisional_kv_bytes": cache.bytes, "committed_kv_bytes": cache.bytes,
                 "target_block_token_visits": len(N.TARGET_WORD)}
            draft_visits = 0
        else:
            length = min(k, remaining)
            if policy == "repeat":
                guesses = repeat(tokens, length)
                work["host_repeat_calls"] += 1
                draft_visits = 0
            else:
                mode = policy.split("_")[0]
                z = d.draft(tokens[-1], tokens[:-1], cache, adapter, length, mode)
                guesses = z[0].argmax(-1).tolist()
                work["draft_calls"] += 1
                draft_visits = len(N.DRAFT_WORD)*(length+1)
            tokens, cache, r = d.verify(tokens, cache, guesses, remaining)
        if tokens != ref[:len(tokens)]:
            raise AssertionError("native trained target greedy mismatch")
        r["draft_block_token_visits"] = draft_visits
        steps.append(r)
        work["target_calls"] += 1
        work["target_token_block_visits"] += r["target_block_token_visits"]
        work["draft_token_block_visits"] += draft_visits
        work["serialized_block_calls"] += len(N.TARGET_WORD)+(len(N.DRAFT_WORD) if draft_visits else 0)
        work["emitted_tokens"] += len(r["emitted"])
        work["draft_tokens"] += len(r["draft"])
        work["accepted_prefix"] += r["accepted_prefix"]
        work["discarded_positions"] += r["discarded_positions"]
        work["peak_provisional_kv_bytes"] = max(work["peak_provisional_kv_bytes"], r["provisional_kv_bytes"])
    wall = time.perf_counter()-started
    cpu = time.process_time()-cpu_start
    if tokens != ref or work["emitted_tokens"] != NEW:
        raise AssertionError("greedy or token conservation")
    d.frozen()
    row = {"root": w["root"], "prefix": w["prefix"], "truth": w["truth"],
           "verified_output": tokens[PREFIX:], "policy": policy, "draft_length": k,
           "steps": steps, "work": dict(work), "tariff": tariff(work), "greedy_parity": True}
    return row, {"wall_seconds": wall, "cpu_seconds": cpu}


def evaluate(d, adapter, policy, k, ws, refs):
    rows = []
    timing = Counter()
    for w in ws:
        r, t = rollout(d, adapter, policy, k, w, refs[w["root"]])
        rows.append(r)
        timing.update(t)
    work = Counter()
    for r in rows:
        for key, value in r["work"].items():
            work[key] = max(work[key], value) if key.startswith("peak_") else work[key]+value
    stable = {"n": len(rows), "work": dict(work), "tariff": sum(r["tariff"] for r in rows),
              "tariff_per_token": sum(r["tariff"] for r in rows)/(len(rows)*NEW),
              "emitted_per_verification": work["emitted_tokens"]/work["target_calls"],
              "target_call_reduction": len(rows)*NEW/work["target_calls"],
              "greedy_mismatches": 0}
    return stable, rows, dict(timing)


def save(out, name, m):
    p = out/name
    torch.save(m.state_dict(), p)
    return {"artifact": name, "artifact_sha256": N.file_sha(p)}


def load(out, m, receipt):
    p = out/receipt["artifact"]
    if N.file_sha(p) != receipt["artifact_sha256"]:
        raise ValueError("weight artifact digest")
    m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True), strict=True)
    if N.tensor_hash(m) != receipt["weights_hash"]:
        raise ValueError("weight identity")
    return m.eval().requires_grad_(False)


def trace_record(seed, th, ah, row):
    r = {"schema": SCHEMA+".rollout", "split": "trace_train", "seed": seed,
         "target_hash": th, "adapter_hash": ah, **row, "training_ready": False}
    r["receipt_hash"] = N.digest(r)
    return r


def build(out, cfg, previous=None):
    validate(cfg)
    torch.set_num_threads(1)
    sets = partitions(cfg)
    report = {"schema": SCHEMA, "status": "public_native_development_only",
              "config": cfg, "source_manifest": manifest(), "split_hashes": split_hashes(sets),
              "targets": [], "adapters": [], "cells": [], "paired": []}
    traces = []
    for seed in cfg["seeds"]:
        model, t = train_target(seed, sets["base_train"], cfg)
        if previous is None:
            t.update(seed=seed, **save(out, f"{seed}-target.pt", model))
        else:
            old = next(x for x in previous["targets"] if x["seed"] == seed)
            if any(old[k] != v for k, v in t.items()):
                raise ValueError("target refit")
            model = load(out, model, old)
            t = {k: old[k] for k in (*t, "seed", "artifact", "artifact_sha256")}
        d = N.Decoder(model)
        refs, quality = {}, {}
        for split in PANELS:
            refs[split], quality[split] = reference(d, sets[split])
        t["quality"] = quality
        t["competence_gate"] = all(quality["cal"][key] >= .8 for key in
                                   ("teacher_forced_accuracy", "free_running_accuracy"))
        report["targets"].append(t)
        adapters = {}
        ah = {}
        train = [tuple(w["prefix"]) for w in sets["adapter_train"]]
        for mode in N.MODES:
            initial = initial_adapter(model)
            adapters[mode+"_initial"] = initial
            ah[mode+"_initial"] = N.tensor_hash(initial)
            a, r = N.fixture_train(d, mode, train, steps=cfg["adapter_steps"])
            if r["initial_hash"] != ah[mode+"_initial"]:
                raise AssertionError("unpaired initial adapter")
            if previous is None:
                r.update(seed=seed, **save(out, f"{seed}-{mode}-adapter.pt", a))
            else:
                old = next(x for x in previous["adapters"] if x["seed"] == seed and x["mode"] == mode)
                if any(old[k] != v for k, v in r.items()):
                    raise ValueError("adapter refit")
                a = load(out, a, old)
                r = old
            adapters[mode+"_trained"] = a
            ah[mode+"_trained"] = r["weights_hash"]
            report["adapters"].append(r)
        current = [a for a in report["adapters"] if a["seed"] == seed]
        if len({x["initial_hash"] for x in current}) != 1 or len({x["batch_sha256"] for x in current}) != 1:
            raise AssertionError("adapter pairing")
        paired = {}
        for policy in POLICIES:
            a = adapters.get(policy)
            lengths = (0,) if policy == "serial" else (1, 2, 4)
            cal = {str(k): evaluate(d, a, policy, k, sets["cal"], refs["cal"])[0] for k in lengths}
            k = min(lengths, key=lambda n: (cal[str(n)]["tariff"], n))
            cell = {"seed": seed, "policy": policy, "draft_length": k, "target_hash": t["weights_hash"],
                    "adapter_hash": ah.get(policy), "competence_gate": t["competence_gate"],
                    "calibration": cal, "panels": {}, "timing": {}}
            paired[policy] = {}
            for split in PANELS:
                metrics, rows, timing = evaluate(d, a, policy, k, sets[split], refs[split])
                cell["panels"][split] = metrics
                cell["timing"][split] = timing
                paired[policy][split] = rows
                if split == "trace_train":
                    traces.extend(trace_record(seed, t["weights_hash"], ah.get(policy), row) for row in rows)
            report["cells"].append(cell)
        for policy in POLICIES:
            controls = ["serial"] if policy != "serial" else []
            if policy.endswith("_trained"):
                controls.extend(["repeat", policy.replace("_trained", "_initial")])
            for control in controls:
                for split in PANELS:
                    ds = [b["tariff"]-a["tariff"] for a, b in
                          zip(paired[policy][split], paired[control][split], strict=True)]
                    report["paired"].append({"seed": seed, "policy": policy, "control": control,
                                             "split": split, "tariff_saved": sum(ds),
                                             "wins": sum(x > 0 for x in ds), "ties": sum(x == 0 for x in ds),
                                             "losses": sum(x < 0 for x in ds)})
        if N.tensor_hash(model) != t["weights_hash"]:
            raise AssertionError("target mutation")
    return report, traces


def without_timing(report):
    return {**report, "cells": [{k: v for k, v in row.items() if k != "timing"} for row in report["cells"]]}


def run(out, cfg):
    out.mkdir(parents=True, exist_ok=False)
    report, traces = build(out, cfg)
    path = out/"afg02_rollouts.jsonl"
    with path.open("x", encoding="utf-8") as f:
        for row in traces:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
    report["trace_sha256"] = N.file_sha(path)
    (out/"afg02_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def audit(out):
    report = json.loads((out/"afg02_report.json").read_text(encoding="utf-8"))
    path = out/"afg02_rollouts.jsonl"
    if N.file_sha(path) != report["trace_sha256"]:
        raise ValueError("trace inventory")
    replay, rows = build(out, report["config"], previous=report)
    expected = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    replay["trace_sha256"] = report["trace_sha256"]
    if without_timing(replay) != without_timing(report) or rows != expected:
        raise ValueError("exact target, adapter, calibration or rollout replay")
    return {"status": "audited_public_native_development_only", "targets": len(report["targets"]),
            "competent_targets": sum(t["competence_gate"] for t in report["targets"]),
            "adapters": len(report["adapters"]), "cells": len(report["cells"]), "rollouts": len(rows),
            "greedy_mismatches": 0, "full_checkpoint_loaded": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path)
    p.add_argument("--audit", type=Path)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.audit:
        print(json.dumps(audit(a.audit), sort_keys=True))
    elif a.out:
        cfg = (dict(DEFAULT, seeds=[137], base_n=16, adapter_n=8, eval_n=4,
                    base_steps=4, batch=4, adapter_steps=4) if a.smoke else DEFAULT)
        r = run(a.out, cfg)
        print(json.dumps({"status": r["status"], "targets": len(r["targets"]),
                          "cells": len(r["cells"])}, sort_keys=True))
    else:
        p.error("choose --out or --audit")


if __name__ == "__main__":
    main()
