"""Offline matched feedback pilot. The native launcher applies process limits before importing this module."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import autograd.numpy as anp
import numpy as np
from autograd import value_and_grad
from autograd.tracer import getval

from datagoblin.hashing import sha256_file
from datagoblin.serialization import canonical_json, dump_structured, load_structured

from .rlt_pilot_data import (
    BOS,
    CONTEXT,
    NODE,
    PAD,
    RULE_SPLITS,
    TEXT,
    VOCAB,
    batch_plan,
    initialization_digest,
    jsonl,
    rule_input,
    sample_rule_turns,
)

MODES = ("direct", "rlt", "matched_damping")
DEPTHS = (2, 4, 8)
TRAIN_DEPTH = 4
WIDTH = 32
HEADS = 2
TEXT_BATCH = 2
RULE_BATCH = 2
WINDOW_CONTEXT = 96
OPTION_STRIDE = 32


def feedback_weight(mode: str, visit: int, depth: int) -> float:
    if mode == "direct" or visit == depth - 1:
        return 1.0
    if mode == "rlt":
        return 1 / (depth - visit)
    if mode == "matched_damping":
        return math.fsum(1 / j for j in range(2, depth + 1)) / (depth - 1)
    raise ValueError(f"unknown feedback {mode!r}")


def initialize(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shapes = {
        "tokens": (VOCAB, WIDTH), "positions": (CONTEXT, WIDTH), "time": (2, WIDTH),
        "qkv": (WIDTH, WIDTH * 3), "projection": (WIDTH, WIDTH),
        "up": (WIDTH, WIDTH * 4), "down": (WIDTH * 4, WIDTH),
    }
    return {name: rng.normal(0, 0.06, size=shape).astype(np.float32) for name, shape in shapes.items()}


def normalize(x):
    centered = x - anp.mean(x, axis=-1, keepdims=True)
    return centered / anp.sqrt(anp.mean(centered**2, axis=-1, keepdims=True) + 1e-5)


def logits_and_states(
    weights, ids, mode, depth, *, all_visits=False, delta=None, trace=None,
    clock_depths=None, feedback_pulse=None, final_endpoint=None,
):
    batch, length = ids.shape
    x = weights["tokens"][ids] + weights["positions"][:length]
    if delta is not None:
        x = x + delta
    causal = np.triu(np.full((length, length), -1e4, dtype=np.float32), k=1)
    outputs = []
    phase_depth, step_depth = (depth, depth) if clock_depths is None else clock_depths
    for visit in range(depth):
        clock = np.array([visit / phase_depth, 1 / step_depth], dtype=np.float32)
        conditioned = normalize(x) + anp.dot(clock, weights["time"])
        packed = anp.matmul(conditioned, weights["qkv"])
        q, k, v = anp.transpose(anp.reshape(packed, (batch, length, 3, HEADS, WIDTH // HEADS)), (2, 0, 3, 1, 4))
        scores = anp.matmul(q, anp.swapaxes(k, -1, -2)) / math.sqrt(WIDTH // HEADS) + causal
        e = anp.exp(scores - anp.max(scores, axis=-1, keepdims=True))
        attn = anp.matmul(e / anp.sum(e, axis=-1, keepdims=True), v)
        attn = anp.reshape(anp.transpose(attn, (0, 2, 1, 3)), (batch, length, WIDTH))
        attended = x + anp.matmul(attn, weights["projection"])
        hidden = anp.matmul(normalize(attended), weights["up"])
        gelu = 0.5 * hidden * (1 + anp.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * hidden**3)))
        h = attended + anp.matmul(gelu, weights["down"])
        eta = feedback_weight(mode, visit, depth)
        advanced = h if eta == 1 else x + eta * (h - x)
        if feedback_pulse is not None and visit == feedback_pulse[0]:
            advanced = advanced + eta * feedback_pulse[1]
        if trace is not None:
            trace.append(tuple(float(np.sqrt(np.mean(t**2))) for t in (advanced, h, advanced - x)))
        if all_visits or visit == depth - 1:
            outputs.append(anp.matmul(normalize(h), weights["tokens"].T))
        if final_endpoint is not None and visit == depth - 1:
            final_endpoint.append(h)
        x = advanced
    return outputs


def target_nll(logits, targets):
    top = anp.max(logits, axis=-1)
    logsum = top + anp.log(anp.sum(anp.exp(logits - top[..., None]), axis=-1))
    chosen = logits[anp.arange(len(targets)), targets]
    return logsum - chosen


def make_batch(text, rules, text_idx, rule_idx):
    xs, ys = [], []
    for i in text_idx:
        seq = text[i]
        xs.append(seq[:-1])
        ys.append(seq[1:])
    width = len(xs[0])
    for i in rule_idx:
        seq, answer = rules[i]
        xs.append(seq + [PAD] * (width - len(seq)))
        ys.append([-100] * (len(seq) - 1) + [answer] + [-100] * (width - len(seq)))
    return np.asarray(xs), np.asarray(ys)


def losses(weights, x, y, mode):
    active_rows, active_positions = np.where(y[TEXT_BATCH:] >= 0)
    text_losses, rule_losses = [], []
    for logits in logits_and_states(weights, x, mode, TRAIN_DEPTH, all_visits=True):
        text = logits[:TEXT_BATCH].reshape((-1, VOCAB))
        rule = logits[active_rows + TEXT_BATCH, active_positions]
        text_losses.append(anp.mean(target_nll(text, y[:TEXT_BATCH].reshape(-1))))
        rule_losses.append(anp.mean(target_nll(rule, y[active_rows + TEXT_BATCH, active_positions])))
    return sum(text_losses) / TRAIN_DEPTH, sum(rule_losses) / TRAIN_DEPTH


def loss_and_parts(weights, x, y, mode, parts):
    text_loss, rule_loss = losses(weights, x, y, mode)
    # `value_and_grad` differentiates this scalar. Save plain scalars for the receipt.
    parts.extend((float(getval(text_loss)), float(getval(rule_loss))))
    return text_loss + rule_loss


def save_optimizer(path, step, weights, first, second, elapsed):
    state = {"step": step, "elapsed": elapsed}
    for prefix, arrays in (("w", weights), ("m", first), ("v", second)):
        state.update({f"{prefix}_{name}": value for name, value in arrays.items()})
    pending = path.with_suffix(".partial.npz")
    np.savez(pending, **state)
    os.replace(pending, path)


def restore_optimizer(path, weights, first, second):
    with np.load(path, allow_pickle=False) as state:
        for prefix, arrays in (("w", weights), ("m", first), ("v", second)):
            for name in arrays:
                arrays[name] = state[f"{prefix}_{name}"]
        return int(state["step"]), float(state["elapsed"])


def train(
    root: Path, out: Path, mode: str, seed: int, steps: int, guard: Callable[[], None],
) -> dict[str, Any]:
    text_path = root / "frozen" / "text_train.json"
    text = json.loads(text_path.read_text(encoding="utf-8"))
    rules = sample_rule_turns(root)
    weights = initialize(seed)
    init_hash = initialization_digest(weights)
    first = {key: np.zeros_like(w) for key, w in weights.items()}
    second = {key: np.zeros_like(w) for key, w in weights.items()}
    differentiate = value_and_grad(loss_and_parts)
    checkpoint = out / "weights.npz"
    if (out / "train.json").exists():
        finished = load_structured(out / "train.json")
        if finished["checkpoint_sha256"] != sha256_file(checkpoint) or finished["steps"] != steps:
            raise ValueError("finished training artifact changed")
        return finished

    progress = out / "optimizer.npz"
    saved, elapsed = restore_optimizer(progress, weights, first, second) if progress.exists() else (0, 0.0)
    started = time.perf_counter()
    updates = out / "train.jsonl"
    lines = updates.read_text(encoding="utf-8").splitlines()[:saved] if updates.exists() else []
    if len(lines) != saved or (lines and json.loads(lines[-1])["update"] != saved):
        raise ValueError("saved optimizer does not match training history")
    # A process can stop after writing the log but before committing the optimizer snapshot.
    # Retain only the committed prefix, replacing the log atomically before appending.
    committed = updates.with_suffix(".partial.jsonl")
    committed.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")
    os.replace(committed, updates)
    saved_step = saved
    with updates.open("a", encoding="utf-8", newline="\n") as handle:
        plan = batch_plan(seed, steps, len(text), len(rules), TEXT_BATCH, RULE_BATCH)
        for step, (ti, ri) in enumerate(plan, 1):
            if step <= saved:
                continue
            try:
                guard()
            except Exception:
                if saved_step > saved:
                    handle.flush()
                    duration = elapsed + time.perf_counter() - started
                    save_optimizer(progress, saved_step, weights, first, second, duration)
                raise
            x, y = make_batch(text, rules, ti, ri)
            parts = []
            loss, grads = differentiate(weights, x, y, mode, parts)
            norm = math.sqrt(sum(float(np.sum(g**2)) for g in grads.values()))
            if not math.isfinite(loss) or not math.isfinite(norm):
                raise FloatingPointError("training loss became nonfinite")
            for key, param in weights.items():
                grad = grads[key] / max(1, norm)
                first[key] = 0.9 * first[key] + 0.1 * grad
                second[key] = 0.999 * second[key] + 0.001 * grad**2
                mean = first[key] / (1 - 0.9**step)
                var = second[key] / (1 - 0.999**step)
                param -= 0.002 * (mean / (np.sqrt(var) + 1e-8) + 0.01 * param)
            row = {"update": step, "text_loss": parts[0], "rule_loss": parts[1],
                   "elapsed_s": elapsed + time.perf_counter() - started}
            handle.write(canonical_json(row) + "\n")
            saved_step = step
            if step % 32 == 0 or step == steps:
                handle.flush()
                save_optimizer(progress, step, weights, first, second, row["elapsed_s"])
                print(canonical_json({"mode": mode, **row}), flush=True)
    np.savez(checkpoint, **weights)
    receipt = {
        "feedback": mode, "seed": seed, "steps": steps, "init_sha256": init_hash,
        "checkpoint_sha256": sha256_file(checkpoint), "train_depth": TRAIN_DEPTH,
        "unique_parameters": sum(p.size for p in weights.values()),
        "text_target_bytes": steps * TEXT_BATCH * WINDOW_CONTEXT, "rule_targets": steps * RULE_BATCH,
        "shared_block_token_visits_including_padding": steps * (TEXT_BATCH + RULE_BATCH) * WINDOW_CONTEXT * TRAIN_DEPTH,
        "elapsed_s": elapsed + time.perf_counter() - started,
        "train_trace_sha256": sha256_file(updates),
    }
    dump_structured(out / "train.json", receipt)
    return receipt


@contextmanager
def pending_rows(path):
    pending = path.with_suffix(".part")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        yield handle
    os.replace(pending, path)


def continuation(
    weights: dict[str, np.ndarray], mode: str, depth: int, prefix: list[int], target: list[int],
    *, clock_depths=None,
) -> tuple[float, bool]:
    total = 0.0
    greedy_correct = True
    history = prefix
    for start in range(0, len(target), OPTION_STRIDE):
        chunk = target[start : start + OPTION_STRIDE]
        context = history[-WINDOW_CONTEXT:]
        seq = context + chunk
        logits = logits_and_states(
            weights, np.asarray([seq[:-1]]), mode, depth, clock_depths=clock_depths,
        )[0][0, len(context) - 1 :]
        gold = np.asarray(chunk)
        total += float(target_nll(logits, gold).sum())
        greedy_correct &= bool((logits.argmax(-1) == gold).all())
        history = history + chunk
    return total, greedy_correct


def eval_known(model, mode, depth, name, item, *, clock_depths=None):
    if name in TEXT:
        seq = item["bytes"]
        logits = logits_and_states(
            model, np.asarray([[BOS, *seq[:-1]]]), mode, depth, clock_depths=clock_depths,
        )[0][0]
        nll = float(target_nll(logits, np.asarray(seq)).sum())
        return {"nll_nats": nll, "bytes": len(seq)}
    prefix = [BOS, *item["prompt"].encode()]
    if name == "lambada":
        target = list(item["target"].encode())
        nll, correct = continuation(model, mode, depth, prefix, target, clock_depths=clock_depths)
        return {"correct": correct, "nll_nats": nll, "target_bytes": len(target)}
    nlls = [
        continuation(model, mode, depth, prefix, list(choice.encode()), clock_depths=clock_depths)[0]
        for choice in item["choices"]
    ]
    lengths = [len(c.strip()) for c in item["choices"]]
    raw = min(range(len(nlls)), key=nlls.__getitem__)
    normalized = min(range(len(nlls)), key=lambda i: nlls[i] / max(1, lengths[i]))
    return {
        "correct": raw == item["gold"], "correct_normalized": normalized == item["gold"],
        "choice_nll_nats": nlls, "choice_chars": lengths, "gold": item["gold"],
        "predicted": raw, "predicted_normalized": normalized,
    }


def eval_rule(model, mode, depth, task):
    first = logits_and_states(model, np.asarray([rule_input(task)]), mode, depth)[0][0, -1, 256:288]
    index = first.argmax().item()
    second_ids = np.asarray([rule_input(task, prior=256 + index)])
    second = logits_and_states(model, second_ids, mode, depth)[0][0, -1, 256:288]
    nodes = tuple(NODE)
    return {"id": task["id"], "before": nodes[index], "after": nodes[second.argmax().item()]}


def evaluate(root: Path, out: Path, mode: str, guard: Callable[[], None]) -> None:
    with np.load(out / "weights.npz", allow_pickle=False) as checkpoint:
        model = dict(checkpoint)
    known = load_structured(root / "frozen" / "known.json")
    for depth in DEPTHS:
        known_path = out / f"known-k{depth}.jsonl"
        if not known_path.exists():
            with pending_rows(known_path) as handle:
                for name, items in known.items():
                    for item in items:
                        guard()
                        result = eval_known(model, mode, depth, name, item)
                        handle.write(canonical_json({"benchmark": name, "id": item["id"], **result}) + "\n")
        for split in RULE_SPLITS:
            path = out / f"rule-{split}-k{depth}.jsonl"
            if not path.exists():
                with pending_rows(path) as handle:
                    for task in jsonl(root / "rules" / f"{split}.tasks.jsonl"):
                        guard()
                        handle.write(canonical_json(eval_rule(model, mode, depth, task)) + "\n")
        print(canonical_json({"mode": mode, "evaluated_depth": depth}), flush=True)
    write_diagnostics(model, out, mode, known, guard)


def write_diagnostics(model, out, mode, known, guard):
    measurements = []
    for depth in DEPTHS:
        for item in known[TEXT[0]][:8]:
            guard()
            trace = []
            ids = np.asarray([[BOS, *item["bytes"][:-1]]])
            clean = logits_and_states(model, ids, mode, depth, trace=trace)[0]
            rng = np.random.default_rng(int(item["id"][:8], 16))
            delta = rng.normal(0, 0.01, (1, ids.shape[1], WIDTH)).astype(np.float32)
            shifted = logits_and_states(model, ids, mode, depth, delta=delta)[0]
            measurements.append({
                "id": item["id"], "depth": depth,
                "visits": [
                    {"visit": i + 1, "state_rms": state, "endpoint_rms": endpoint, "update_rms": update}
                    for i, (state, endpoint, update) in enumerate(trace)
                ],
                "logit_rms_over_initial_noise_rms": float(np.linalg.norm(shifted - clean) * math.sqrt(WIDTH / VOCAB)
                                                         / np.linalg.norm(delta)),
            })
    dump_structured(out / "diagnostics.json", measurements)
