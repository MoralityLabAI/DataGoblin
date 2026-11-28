"""AFG-03: explicit local checkpoint qualification, adapter fits and timing."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import native_cache as N
import torch
from torch.nn import functional as F

SCHEMA = "aleph.direct_a.afg03.v1"
PANEL_SCHEMA = "aleph.afg03.token_panel.v1"
TOKENIZER = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
SPLITS = ("train", "cal", "dev")
POLICIES = ("serial", "repeat", "first_initial", "deep_initial", "first_trained", "deep_trained")
HOST_CACHE_CAP = 256 * 2**20
DEFAULT = dict(steps=64, max_new=8, repeats=3, threads=2, wall_seconds=1800,
               allocator_gib=10., precision="bf16", device="cuda:0")


class Deadline:
    def __init__(self, seconds):
        self.end = time.monotonic()+seconds

    def check(self):
        if time.monotonic() >= self.end:
            raise TimeoutError("AFG-03 wall limit at operation boundary")


def source_manifest():
    here = Path(__file__).resolve().parent
    return {**N.SOURCE_MANIFEST,
            "afg03_checkpoint.py": N.file_sha(here/"afg03_checkpoint.py"),
            "AFG03_PROTOCOL.md": N.file_sha(here/"AFG03_PROTOCOL.md")}


def validate_config(cfg):
    if set(cfg) != set(DEFAULT):
        raise ValueError("invalid run config")
    for key, lo, hi in (("steps", 1, 256), ("max_new", 8, 16), ("repeats", 1, 3),
                        ("threads", 1, 8), ("wall_seconds", 60, 7200)):
        if type(cfg[key]) is not int or not lo <= cfg[key] <= hi:
            raise ValueError(f"invalid {key} cap")
    if cfg["precision"] not in ("fp32", "bf16"):
        raise ValueError("invalid precision")
    if cfg["device"] not in ("cpu", "cuda:0"):
        raise ValueError("select CPU or exactly CUDA device zero")
    if not isinstance(cfg["allocator_gib"], (int, float)) or not 5 <= cfg["allocator_gib"] <= 12:
        raise ValueError("allocator bound must be 5..12 GiB")


def parse_panel(data, vocab, *, fixture=False):
    if (not isinstance(data, dict) or set(data) != {"schema", "tokenizer_sha256", "rows"}
            or data["schema"] != PANEL_SCHEMA or data["tokenizer_sha256"] != TOKENIZER
            or not isinstance(data["rows"], list) or len(data["rows"]) > 64):
        raise ValueError("token panel schema")
    by = {s: [] for s in SPLITS}
    groups = {}
    seen = set()
    for r in data["rows"]:
        if (not isinstance(r, dict) or set(r) != {"split", "group", "ids"}
                or r["split"] not in SPLITS or not isinstance(r["group"], str)
                or not 1 <= len(r["group"]) <= 128 or not isinstance(r["ids"], list)
                or not 8 <= len(r["ids"]) <= 32 or any(
                    type(t) is not int or not 0 <= t < vocab for t in r["ids"])):
            raise ValueError("invalid token panel row")
        split, group, ids = r["split"], r["group"], r["ids"]
        if group in groups and groups[group] != split:
            raise ValueError("source group crosses split")
        groups[group] = split
        key = tuple(ids)
        if key in seen:
            raise ValueError("duplicate prompt")
        seen.add(key)
        by[split].append({"root": N.digest([PANEL_SCHEMA, ids]), "group": group, "ids": ids})
    for split, lo, hi in (("train", 8, 32), ("cal", 4, 16), ("dev", 4, 16)):
        if not lo <= len(by[split]) <= hi:
            raise ValueError(f"{split} row count")
    for i, s in enumerate(SPLITS):
        for other in SPLITS[i+1:]:
            for a in by[s]:
                for b in by[other]:
                    n = min(len(a["ids"]), len(b["ids"]))
                    if a["ids"][:n] == b["ids"][:n]:
                        raise ValueError("cross-split prefix containment")
    return by, {"panel_hash": N.digest(data), "split_hashes": {s: N.digest(by[s]) for s in SPLITS},
                "root_counts": {s: len(by[s]) for s in SPLITS}, "groups": len(groups),
                "fixture": fixture}


def fixture_panel():
    rows = []
    for split, n in (("train", 8), ("cal", 4), ("dev", 4)):
        rng = random.Random(f"{SCHEMA}:{split}")
        for i in range(n):
            rows.append({"split": split, "group": f"fixture-{split}-{i}",
                         "ids": [rng.randrange(64) for _ in range(8)]})
    return {"schema": PANEL_SCHEMA, "tokenizer_sha256": TOKENIZER, "rows": rows}


def setup_device(cfg):
    torch.set_num_threads(cfg["threads"])
    if cfg["device"] == "cpu":
        return torch.device("cpu"), {"type": "cpu", "torch": str(torch.__version__)}
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA unavailable; explicit CPU rerun required")
    d = torch.device("cuda:0")
    torch.cuda.set_device(d)
    if cfg["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA unsupported")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    free, total = torch.cuda.mem_get_info(d)
    limit = int(cfg["allocator_gib"] * 2**30)
    if limit >= total or free < limit+512*2**20:
        raise RuntimeError(f"insufficient CUDA headroom: free={free}, cap={limit}")
    torch.cuda.set_per_process_memory_fraction(limit/total, d)
    prop = torch.cuda.get_device_properties(d)
    return d, {"type": "cuda", "name": prop.name, "device": str(d), "torch": str(torch.__version__),
               "cuda_runtime": str(torch.version.cuda), "compute_capability": [prop.major, prop.minor],
               "total_bytes": total, "initial_free_bytes": free, "allocator_cap_bytes": limit,
               "allocator_only_cap": True}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peaks(device):
    if device.type == "cuda":
        sync(device)
        torch.cuda.reset_peak_memory_stats(device)


def windows_peak_memory():
    """Process-lifetime peak working set and private commit from psapi."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    *((name, ctypes.c_size_t) for name in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                        "PagefileUsage", "PeakPagefileUsage"))]

    c = Counters()
    c.cb = ctypes.sizeof(c)
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not kernel.K32GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return {}
    return {"peak_host_rss_bytes": int(c.PeakWorkingSetSize), "peak_host_private_bytes": int(c.PeakPagefileUsage)}


def memory(device):
    result = {}
    if device.type == "cuda":
        sync(device)
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    if sys.platform == "win32":
        result.update(windows_peak_memory())
        return result
    try:
        import resource
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform.startswith("linux"):
            result["peak_host_rss_bytes"] = int(raw)*1024
        elif sys.platform == "darwin":
            result["peak_host_rss_bytes"] = int(raw)
    except ImportError:
        pass
    return result


def overall_memory(observations):
    """Maximum of every per-phase memory record kept in the observations."""
    peaks = Counter()

    def visit(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k.startswith("peak_") and k.endswith("_bytes") and isinstance(v, int):
                    peaks[k] = max(peaks[k], v)
                else:
                    visit(v)
        elif isinstance(x, list):
            for v in x:
                visit(v)
    visit(observations)
    return dict(peaks)


def first_divergence(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))


@torch.no_grad()
def parity_probe(d, prompts, n):
    """Diagnostic after a parity failure: cached vs cold logits along the cold greedy path."""
    diag = N.Diagnostics(d.dtype)
    rows = []
    for ids in prompts:
        tokens = list(ids)
        _, cache = d.append(tokens[:-1])
        for step in range(n):
            z, cache = d.append([tokens[-1]], cache)
            a, b = z[0, -1].float(), d.cold(tokens)[0, -1].float()
            top = torch.topk(b, 2).values
            rows.append({"root": N.digest([PANEL_SCHEMA, list(ids)]), "step": step,
                         "cached_argmax": int(a.argmax()), "cold_argmax": int(b.argmax()),
                         "cold_top2_margin": float(top[0]-top[1]),
                         "max_abs_logit_difference": float((a-b).abs().max()),
                         "within_tolerance": bool(torch.allclose(a, b, atol=diag.atol, rtol=diag.rtol))})
            tokens.append(int(b.argmax()))
    return {"precision": str(d.dtype), "tolerances": [diag.atol, diag.rtol], "steps": rows,
            "argmax_disagreements": sum(r["cached_argmax"] != r["cold_argmax"] for r in rows),
            "tolerance_failures": sum(not r["within_tolerance"] for r in rows)}


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix+".partial")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    tmp.replace(path)


def repeat(tokens, k):
    period = next((p for p in range(1, 5)
                   if len(tokens) >= 2*p and tokens[-p:] == tokens[-2*p:-p]), 1)
    return [tokens[-period+i % period] for i in range(k)]


def tariff(work):
    return 4*work["serialized_block_calls"]+work["target_token_block_visits"]+work["draft_token_block_visits"]


@torch.no_grad()
def teacher_references(d, by, n, deadline):
    diag = N.Diagnostics(d.dtype)
    refs = {}
    ledger = Counter()
    for split in SPLITS:
        refs[split] = {}
        for w in by[split]:
            deadline.check()
            ids = w["ids"]
            try:
                seq = N.serial(d, ids, n, diag)
            except AssertionError as error:
                error.failed_prompt = {"split": split, "root": w["root"], "ids": ids}
                raise
            refs[split][w["root"]] = seq
            ledger["reference_roots"] += 1
            ledger["reference_prefill_token_block_visits"] += (len(ids)-1)*len(N.TARGET_WORD)
            ledger["reference_serial_token_block_visits"] += n*len(N.TARGET_WORD)
            ledger["reference_cold_token_block_visits"] += sum(
                (len(ids)+i)*len(N.TARGET_WORD) for i in range(n))
    return refs, {"work": dict(ledger), "parity": diag.receipt()}


@torch.no_grad()
def teacher_bank(d, rows, refs, deadline):
    bank = []
    work = Counter()
    transfer = 0.
    expected = sum((len(refs[w["root"]])-1)*sum(
        2*d.model.blocks[i].kv*d.model.blocks[i].k for i in N.TARGET_WORD)
        * torch.empty((), dtype=d.dtype).element_size() for w in rows)
    if expected > HOST_CACHE_CAP:
        raise ValueError("teacher host KV bank exceeds registered tensor cap")
    for w in rows:
        deadline.check()
        seq = refs[w["root"]]
        _, cache = d.append(seq[:-1])
        sync(d.device)
        started = time.perf_counter()
        kv = tuple(N.VisitKV(v.visit, v.physical, v.key.detach().to("cpu"),
                             v.value.detach().to("cpu")) for v in cache.visits)
        transfer += time.perf_counter()-started
        size = sum(t.numel()*t.element_size() for v in kv for t in (v.key, v.value))
        if size != cache.bytes:
            raise AssertionError("teacher bank byte conservation")
        work["host_kv_bytes"] += size
        if d.device.type == "cuda":
            work["device_to_host_kv_bytes"] += size
        work["teacher_cache_token_block_visits"] += (len(seq)-1)*len(N.TARGET_WORD)
        bank.append({"root": w["root"], "prompt_length": len(w["ids"]), "seq": seq, "kv": kv})
        del cache
    if work["host_kv_bytes"] != expected:
        raise AssertionError("teacher cache estimate differs")
    return bank, dict(work), {"device_to_host_seconds": transfer}


def prefix_cache(d, entry, cut, timer=None):
    n = cut-1
    if not entry["prompt_length"] <= cut <= len(entry["seq"])-N.BLOCK:
        raise ValueError("invalid adapter cut")
    sync(d.device)
    started = time.perf_counter()
    kv = tuple(N.VisitKV(v.visit, v.physical, v.key[:, :, :n].contiguous().to(d.device),
                         v.value[:, :, :n].contiguous().to(d.device)) for v in entry["kv"])
    sync(d.device)
    if timer is not None:
        timer["host_to_device_seconds"] += time.perf_counter()-started
    cache = N.Cache(d.owner, tuple(entry["seq"][:n]), kv)
    d.validate_cache(cache, entry["seq"][:n])
    return cache


def make_adapter(d):
    torch.manual_seed(837)
    return N.DraftAdapter(d.model, 4).to(d.device)


def fit_adapter(d, mode, bank, cfg, deadline):
    a = make_adapter(d)
    before = N.tensor_hash(a)
    opt = torch.optim.AdamW(a.parameters(), lr=.01)
    rng = random.Random(438)
    draws = []
    work = Counter()
    timer = Counter()
    losses = []
    reset_peaks(d.device)
    start = time.monotonic()
    for _ in range(cfg["steps"]):
        deadline.check()
        i = rng.randrange(len(bank))
        e = bank[i]
        cut = rng.randrange(e["prompt_length"], len(e["seq"])-N.BLOCK+1)
        cache = prefix_cache(d, e, cut, timer)
        z = d.draft(e["seq"][cut-1], e["seq"][:cut-1], cache, a, N.BLOCK, mode)
        y = torch.tensor(e["seq"][cut:cut+N.BLOCK], device=d.device, dtype=torch.long)
        loss = F.cross_entropy(z.reshape(-1, d.vocab), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite draft training loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(a.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        d.frozen()
        losses.append(float(loss.detach()))
        draws.append([i, cut])
        work["draft_train_token_block_visits"] += (N.BLOCK+1)*len(N.DRAFT_WORD)
        work["supervised_labels"] += N.BLOCK
        work["logical_prefix_kv_bytes"] += cache.bytes
        if d.device.type == "cuda":
            work["host_to_device_kv_bytes"] += cache.bytes
    sync(d.device)
    elapsed = time.monotonic()-start
    a.eval().requires_grad_(False)
    after = N.tensor_hash(a)
    if before == after:
        raise AssertionError("adapter unchanged")
    return a, {"mode": mode, "initial_hash": before, "weights_hash": after,
               "parameters": sum(p.numel() for p in a.parameters()), "updates": cfg["steps"],
               "draws_sha256": N.digest(draws), "last_loss": float(loss.detach()), "work": dict(work)}, {
               "seconds": elapsed, "host_to_device_seconds": timer["host_to_device_seconds"],
               "loss_first8_mean": statistics.fmean(losses[:8]), "loss_last8_mean": statistics.fmean(losses[-8:]),
               "memory": memory(d.device)}


@torch.no_grad()
def decode(d, adapter, policy, k, w, n):
    tokens = list(w["ids"])
    _, cache = d.append(tokens[:-1])
    work = Counter()
    work["prefill_token_block_visits"] = (len(tokens)-1)*len(N.TARGET_WORD)
    work["peak_committed_kv_bytes"] = cache.bytes
    steps = []
    limit = len(tokens)+n
    while len(tokens) < limit:
        remaining = limit-len(tokens)
        if policy == "serial":
            z, cache = d.append([tokens[-1]], cache)
            emitted = [int(z[0, -1].argmax())]
            tokens.extend(emitted)
            r = {"draft": [], "accepted_prefix": 0, "emitted": emitted, "discarded_positions": 0,
                 "target_block_token_visits": len(N.TARGET_WORD), "provisional_kv_bytes": cache.bytes,
                 "committed_kv_bytes": cache.bytes}
            draft_work = 0
        else:
            width = min(k, remaining)
            if policy == "repeat":
                guesses = repeat(tokens, width)
                work["host_repeat_calls"] += 1
                draft_work = 0
            else:
                z = d.draft(tokens[-1], tokens[:-1], cache, adapter, width, policy.split("_")[0])
                guesses = z[0].argmax(-1).tolist()
                work["draft_calls"] += 1
                draft_work = (width+1)*len(N.DRAFT_WORD)
            tokens, cache, r = d.verify(tokens, cache, guesses, remaining)
        r["draft_block_token_visits"] = draft_work
        steps.append(r)
        work["target_calls"] += 1
        work["target_token_block_visits"] += r["target_block_token_visits"]
        work["draft_token_block_visits"] += draft_work
        work["serialized_block_calls"] += len(N.TARGET_WORD)+(len(N.DRAFT_WORD) if draft_work else 0)
        work["emitted_tokens"] += len(r["emitted"])
        work["draft_tokens"] += len(r["draft"])
        work["accepted_prefix"] += r["accepted_prefix"]
        work["discarded_positions"] += r["discarded_positions"]
        work["peak_provisional_kv_bytes"] = max(work["peak_provisional_kv_bytes"], r["provisional_kv_bytes"])
        work["peak_committed_kv_bytes"] = max(work["peak_committed_kv_bytes"], r["committed_kv_bytes"])
    if work["emitted_tokens"] != n or cache.tokens != tuple(tokens[:-1]):
        raise AssertionError("decode token or cache conservation")
    return {"root": w["root"], "policy": policy, "draft_length": k, "steps": steps,
            "output": tokens[len(w["ids"]):], "work": dict(work), "tariff": tariff(work)}


def aggregate(rows):
    work = Counter()
    eligible, accepted = [0]*N.BLOCK, [0]*N.BLOCK
    for r in rows:
        for key, value in r["work"].items():
            work[key] = max(work[key], value) if key.startswith("peak_") else work[key]+value
        for step in r["steps"]:
            for j in range(len(step["draft"])):
                eligible[j] += 1
                accepted[j] += step["accepted_prefix"] >= j+1
    total = sum(r["tariff"] for r in rows)
    if total != tariff(work):
        raise AssertionError("tariff conservation")
    return {"requests": len(rows), "tariff": total, "tariff_per_token": total/work["emitted_tokens"],
            "emitted_per_verification": work["emitted_tokens"]/work["target_calls"],
            "work": dict(work), "acceptance_survival": {"eligible": eligible, "accepted": accepted},
            "greedy_mismatches": 0}


def evaluate(d, adapter, policy, k, rows, refs, n, deadline, *, timed=False):
    result = []
    reset_peaks(d.device)
    sync(d.device)
    wall, cpu = time.perf_counter(), time.process_time()
    for w in rows:
        deadline.check()
        result.append(decode(d, adapter, policy, k, w, n))
    sync(d.device)
    wall, cpu = time.perf_counter()-wall, time.process_time()-cpu
    # Compare against independent reference outside the timed region.
    for r, w in zip(result, rows, strict=True):
        expected = refs[w["root"]][len(w["ids"]):]
        if r["root"] != w["root"] or r["output"] != expected:
            error = AssertionError(f"exact greedy mismatch: {policy}, {r['root']}")
            error.receipt = {"policy": policy, "draft_length": k, "root": w["root"], "output": r["output"],
                             "reference": expected, "first_divergence": first_divergence(r["output"], expected),
                             "steps": r["steps"]}
            raise error
        r["greedy_parity"] = True
    d.frozen()
    s = aggregate(result)
    timing = None
    if timed:
        timing = {"wall_seconds": wall, "cpu_seconds": cpu,
                  "tokens_per_second": s["work"]["emitted_tokens"]/wall, "memory": memory(d.device)}
    return s, result, timing


def save_adapter(out, a, info):
    p = out/f"{info['mode']}-adapter.pt"
    torch.save(a.state_dict(), p)
    return {**info, "artifact": p.name, "artifact_sha256": N.file_sha(p)}


def run(out, data, data_sha, cfg, *, checkpoint=None, fixture=False):
    validate_config(cfg)
    deadline = Deadline(cfg["wall_seconds"])
    out.mkdir(parents=True, exist_ok=False)
    stage = "input"
    observations = {}
    d = qualification_prompts = None
    started = time.perf_counter()
    try:
        vocab = 64 if fixture else 49152
        by, panel_receipt = parse_panel(data, vocab, fixture=fixture)
        atomic_json(out/"input_panel.json", data)
        stage = "device"
        device, hardware = setup_device(cfg)
        observations["hardware"] = hardware
        stage = "checkpoint"
        deadline.check()
        reset_peaks(device)
        t = time.perf_counter()
        model = N.mini_model() if fixture else N.load_native_checkpoint(checkpoint, device)
        model = model.to(device).eval().requires_grad_(False)
        d = N.Decoder(model, torch.bfloat16 if cfg["precision"] == "bf16" else torch.float32)
        target_hash = N.tensor_hash(model)
        observations["load_seconds"] = time.perf_counter()-t
        observations["load_memory"] = memory(device)
        stage = "parity"
        deadline.check()
        reset_peaks(device)
        t = time.perf_counter()
        qualification_prompts = [w["ids"] for w in by["cal"][:2]]
        qualification = N.qualify(d, qualification_prompts, n=cfg["max_new"])
        observations["qualification"] = {"seconds": time.perf_counter()-t, "parity": qualification["parity"],
                                         "memory": memory(device)}
        stage = "teacher"
        reset_peaks(device)
        t = time.perf_counter()
        refs, reference_receipt = teacher_references(d, by, cfg["max_new"], deadline)
        reference_seconds = time.perf_counter()-t
        t = time.perf_counter()
        bank, bank_work, bank_timing = teacher_bank(d, by["train"], refs["train"], deadline)
        observations["teacher"] = {"reference_seconds": reference_seconds, "parity": reference_receipt["parity"],
                                   "bank_seconds": time.perf_counter()-t, **bank_timing,
                                   "memory": memory(device)}
        stable = {"schema": SCHEMA, "status": "fixture_only" if fixture else "native_checkpoint_complete",
                  "fixture": fixture, "checkpoint_sha256": None if fixture else N.CHECKPOINT_SHA,
                  "target_weights_hash": target_hash, "source_manifest": source_manifest(),
                  "data_sha256": data_sha, "panel": panel_receipt, "config": cfg,
                  "qualification": qualification, "teacher": reference_receipt, "teacher_bank": bank_work,
                  "teacher_streams_sha256": N.digest(refs), "adapters": [], "cells": []}
        atomic_json(out/"progress.json", {"status": "qualified", "stage": stage,
                                         "target_hash": target_hash, "panel": panel_receipt})
        stage = "adapter_fit"
        adapters, hashes = {}, {}
        fit_observations = {}
        for mode in N.MODES:
            deadline.check()
            initial = make_adapter(d).eval().requires_grad_(False)
            adapters[mode+"_initial"] = initial
            hashes[mode+"_initial"] = N.tensor_hash(initial)
            a, receipt, metrics = fit_adapter(d, mode, bank, cfg, deadline)
            if receipt["initial_hash"] != hashes[mode+"_initial"]:
                raise AssertionError("unpaired initialization")
            adapters[mode+"_trained"] = a
            hashes[mode+"_trained"] = receipt["weights_hash"]
            stable["adapters"].append(save_adapter(out, a, receipt))
            fit_observations[mode] = metrics
        if len({r["initial_hash"] for r in stable["adapters"]}) != 1 or len(
                {r["draws_sha256"] for r in stable["adapters"]}) != 1:
            raise AssertionError("adapter arms differ in initialization or data draws")
        del bank
        observations["fit"] = fit_observations
        stage = "calibration"
        selections = {}
        for policy in POLICIES:
            lengths = (0,) if policy == "serial" else (1, 2, 4)
            cal = {}
            for k in lengths:
                s, _, _ = evaluate(d, adapters.get(policy), policy, k, by["cal"], refs["cal"],
                                   cfg["max_new"], deadline)
                cal[str(k)] = s
            chosen = min(lengths, key=lambda k: (cal[str(k)]["tariff"], k))
            selections[policy] = chosen
            stable["cells"].append({"policy": policy, "adapter_hash": hashes.get(policy),
                                    "calibration": cal, "selected_length": chosen})
        best = min(stable["cells"], key=lambda c: (c["calibration"][str(c["selected_length"])]["tariff"],
                                                   POLICIES.index(c["policy"])))
        stable["cal_selected_policy"] = best["policy"]
        stage = "development"
        trace = []
        observations["decode"] = {}
        for cell in stable["cells"]:
            policy, k = cell["policy"], cell["selected_length"]
            a = adapters.get(policy)
            evaluate(d, a, policy, k, by["cal"][:1], refs["cal"], cfg["max_new"], deadline)
            timings = []
            reference = None
            for _ in range(cfg["repeats"]):
                s, rows, t = evaluate(d, a, policy, k, by["dev"], refs["dev"],
                                      cfg["max_new"], deadline, timed=True)
                if reference is None:
                    reference = (s, rows)
                elif reference != (s, rows):
                    raise AssertionError("timed repeat changed decode decisions or work")
                timings.append(t)
            cell["development"] = reference[0]
            for row in reference[1]:
                r = {"schema": SCHEMA+".development", "split": "dev", "target_hash": target_hash,
                     "adapter_hash": hashes.get(policy), **row, "training_ready": False}
                r["receipt_hash"] = N.digest(r)
                trace.append(r)
            observations["decode"][policy] = {"samples": timings,
                "median_tokens_per_second": statistics.median(t["tokens_per_second"] for t in timings),
                "median_wall_seconds": statistics.median(t["wall_seconds"] for t in timings)}
        stage = "final_identity"
        deadline.check()
        d.frozen()
        if N.tensor_hash(model) != target_hash or any(N.tensor_hash(a) != hashes[p] for p, a in adapters.items()):
            raise AssertionError("target or adapter mutated")
        baseline = observations["decode"]["serial"]["median_tokens_per_second"]
        for p in POLICIES:
            observations["decode"][p]["speedup_over_serial"] = (
                observations["decode"][p]["median_tokens_per_second"]/baseline)
        observations["final_memory"] = memory(device)
        observations["overall_peak_memory"] = overall_memory(observations)
        observations["elapsed_seconds"] = time.perf_counter()-started
        path = out/"development_rollouts.jsonl"
        with path.open("x", encoding="utf-8") as f:
            for row in trace:
                f.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
        stable["trace_sha256"] = N.file_sha(path)
        result = {"stable": stable, "observations": observations}
        atomic_json(out/"report.json", result)
        atomic_json(out/"progress.json", {"status": "complete", "stage": stage})
        return result
    except Exception as error:
        failure = {"status": "blocked", "stage": stage, "error_type": type(error).__name__,
                   "reason": str(error)[:2000], "elapsed_seconds": time.perf_counter()-started,
                   "receipt": getattr(error, "receipt", None),
                   "failed_prompt": getattr(error, "failed_prompt", None)}
        if isinstance(error, AssertionError) and d is not None and stage in ("parity", "teacher"):
            probe_prompts = ([failure["failed_prompt"]["ids"]] if failure["failed_prompt"]
                             else qualification_prompts)
            try:
                failure["parity_probe"] = parity_probe(d, probe_prompts, cfg["max_new"])
            except Exception as probe_error:  # the probe is diagnostic only
                failure["parity_probe"] = {"error": f"{type(probe_error).__name__}: {str(probe_error)[:500]}"}
        if d is not None:
            observations["overall_peak_memory"] = overall_memory(
                {**observations, "at_failure": memory(d.device)})
        atomic_json(out/"progress.json", {**failure, "observations": observations})
        raise


def stable_for_replay(result):
    s = result["stable"]
    return {**s, "adapters": [{k: v for k, v in a.items() if k != "artifact_sha256"} for a in s["adapters"]]}


def audit_fixture(out):
    report = json.loads((out/"report.json").read_text(encoding="utf-8"))
    s = report["stable"]
    if not s["fixture"] or s["status"] != "fixture_only":
        raise ValueError("automatic refit audit is only for the small fixture")
    if s["source_manifest"] != source_manifest():
        raise ValueError("source mismatch")
    data = json.loads((out/"input_panel.json").read_text(encoding="utf-8"))
    if N.digest(data) != s["data_sha256"]:
        raise ValueError("fixture data hash")
    m = N.mini_model()
    if N.tensor_hash(m) != s["target_weights_hash"]:
        raise ValueError("fixture target identity")
    for a in s["adapters"]:
        path = out/a["artifact"]
        if N.file_sha(path) != a["artifact_sha256"]:
            raise ValueError("adapter file digest")
        model = N.DraftAdapter(m, 4)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        if N.tensor_hash(model) != a["weights_hash"]:
            raise ValueError("adapter tensor digest")
    path = out/"development_rollouts.jsonl"
    if N.file_sha(path) != s["trace_sha256"]:
        raise ValueError("trace digest")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != len(POLICIES)*s["panel"]["root_counts"]["dev"]:
        raise ValueError("trace coverage")
    for r in rows:
        obj = dict(r)
        bound = obj.pop("receipt_hash")
        if N.digest(obj) != bound or r["training_ready"] is not False or not r["greedy_parity"]:
            raise ValueError("trace receipt")
    for obs in report["observations"]["decode"].values():
        samples = obs["samples"]
        if len(samples) != s["config"]["repeats"] or any(
                not math.isfinite(t["tokens_per_second"]) or t["tokens_per_second"] <= 0 for t in samples):
            raise ValueError("timing coverage")
        if statistics.median(t["tokens_per_second"] for t in samples) != obs["median_tokens_per_second"]:
            raise ValueError("timing summary")
    with tempfile.TemporaryDirectory(prefix="afg03-audit-") as folder:
        replay = run(Path(folder)/"replay", data, s["data_sha256"], s["config"], fixture=True)
        if stable_for_replay(replay) != stable_for_replay(report):
            raise ValueError("fixture refit/calibration/greedy replay")
    return {"status": "audited_fixture_only", "policies": len(s["cells"]),
            "development_rollouts": len(rows), "greedy_mismatches": 0, "full_checkpoint_loaded": False}


def audit_report(out):
    """Weight-free integrity audit of a completed run (real checkpoint or fixture).

    Checks digests, trace receipts, aggregates, CAL selection and timing
    summaries against the saved receipts. It does not reload the target,
    so it cannot replay decoding; that needs the checkpoint and device.
    """
    report = json.loads((out/"report.json").read_text(encoding="utf-8"))
    progress = json.loads((out/"progress.json").read_text(encoding="utf-8"))
    s, obs = report["stable"], report["observations"]
    if progress["status"] != "complete" or s["source_manifest"] != source_manifest():
        raise ValueError("incomplete run or source mismatch")
    data = json.loads((out/"input_panel.json").read_text(encoding="utf-8"))
    if N.digest(data) != s["panel"]["panel_hash"]:
        raise ValueError("panel hash")
    by, panel = parse_panel(data, 64 if s["fixture"] else 49152, fixture=s["fixture"])
    if panel != s["panel"]:
        raise ValueError("panel receipt")
    with torch.device("meta"):
        target = N.mini_model() if s["fixture"] else N.full_model()
    for a in s["adapters"]:
        path = out/a["artifact"]
        if N.file_sha(path) != a["artifact_sha256"]:
            raise ValueError("adapter file digest")
        with torch.device("cpu"):
            model = N.DraftAdapter(target, 4)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        if N.tensor_hash(model) != a["weights_hash"] or a["updates"] != s["config"]["steps"]:
            raise ValueError("adapter tensor digest or update count")
    path = out/"development_rollouts.jsonl"
    if N.file_sha(path) != s["trace_sha256"]:
        raise ValueError("trace digest")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    dev_roots = [w["root"] for w in by["dev"]]
    for cell in s["cells"]:
        mine = [r for r in rows if r["policy"] == cell["policy"]]
        if [r["root"] for r in mine] != dev_roots:
            raise ValueError("trace coverage")
        for r in mine:
            obj = dict(r)
            bound = obj.pop("receipt_hash")
            if (N.digest(obj) != bound or r["training_ready"] is not False or not r["greedy_parity"]
                    or r["draft_length"] != cell["selected_length"] or r["target_hash"] != s["target_weights_hash"]
                    or r["adapter_hash"] != cell["adapter_hash"]):
                raise ValueError("trace receipt")
        strip = ("schema", "split", "target_hash", "adapter_hash", "training_ready", "receipt_hash", "greedy_parity")
        if aggregate([{k: v for k, v in r.items() if k not in strip} for r in mine]) != cell["development"]:
            raise ValueError("development aggregate")
        lengths = [int(k) for k in cell["calibration"]]
        if cell["selected_length"] != min(lengths, key=lambda k: (cell["calibration"][str(k)]["tariff"], k)):
            raise ValueError("CAL length selection")
    best = min(s["cells"], key=lambda c: (c["calibration"][str(c["selected_length"])]["tariff"],
                                          POLICIES.index(c["policy"])))
    if best["policy"] != s["cal_selected_policy"]:
        raise ValueError("CAL policy selection")
    references = {}
    for r in rows:
        if references.setdefault(r["root"], r["output"]) != r["output"]:
            raise ValueError("policies disagree on greedy output")
    for policy, o in obs["decode"].items():
        samples = o["samples"]
        if len(samples) != s["config"]["repeats"] or any(
                not math.isfinite(t["tokens_per_second"]) or t["tokens_per_second"] <= 0 for t in samples):
            raise ValueError("timing coverage")
        if statistics.median(t["tokens_per_second"] for t in samples) != o["median_tokens_per_second"]:
            raise ValueError("timing summary")
        ratio = o["median_tokens_per_second"]/obs["decode"]["serial"]["median_tokens_per_second"]
        if not math.isclose(ratio, o["speedup_over_serial"], rel_tol=1e-12) or policy not in POLICIES:
            raise ValueError("speedup summary")
    return {"status": "audited_receipts", "fixture": s["fixture"], "policies": len(s["cells"]),
            "development_rollouts": len(rows), "greedy_mismatches": 0,
            "cal_selected_policy": s["cal_selected_policy"], "target_reloaded": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--data-sha256")
    p.add_argument("--out", type=Path)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--audit-fixture", type=Path)
    p.add_argument("--audit-report", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default=DEFAULT["device"])
    p.add_argument("--precision", choices=("fp32", "bf16"), default=DEFAULT["precision"])
    for k in ("steps", "max_new", "repeats", "threads", "wall_seconds"):
        p.add_argument("--"+k.replace("_", "-"), type=int, default=DEFAULT[k])
    p.add_argument("--allocator-gib", type=float, default=DEFAULT["allocator_gib"])
    a = p.parse_args()
    if a.audit_fixture:
        print(json.dumps(audit_fixture(a.audit_fixture), sort_keys=True))
        return
    if a.audit_report:
        print(json.dumps(audit_report(a.audit_report), sort_keys=True))
        return
    if not a.out:
        p.error("--out is required")
    cfg = {k: getattr(a, k) for k in DEFAULT}
    if a.fixture:
        if a.execute or a.checkpoint or a.data:
            p.error("fixture does not accept production inputs")
        cfg.update(device="cpu", precision="fp32", steps=4, repeats=2, threads=1, max_new=8)
        data = fixture_panel()
        sha = N.digest(data)
    else:
        if not a.execute or not a.checkpoint or not a.data or not a.data_sha256:
            p.error("real checkpoint requires --execute, --checkpoint, --data and --data-sha256")
        if a.data.stat().st_size > 1024*1024:
            raise ValueError("token panel file exceeds 1 MiB")
        sha = N.file_sha(a.data)
        if sha != a.data_sha256:
            raise ValueError("operator token-panel hash mismatch")
        data = json.loads(a.data.read_text(encoding="utf-8"))
    r = run(a.out, data, sha, cfg, checkpoint=a.checkpoint, fixture=a.fixture)
    print(json.dumps({"status": r["stable"]["status"], "cal_selected_policy": r["stable"]["cal_selected_policy"],
                      "decode": {p: v["median_tokens_per_second"]
                                 for p, v in r["observations"]["decode"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
