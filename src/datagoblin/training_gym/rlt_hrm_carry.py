"""Matched RLT placements in a learned two-timescale binary addition loop."""

from __future__ import annotations

import hashlib
import time
from itertools import islice, product
from pathlib import Path

import autograd.numpy as anp
import numpy as np
from autograd import value_and_grad

from datagoblin.hashing import sha256_file
from datagoblin.serialization import canonical_json, dump_structured, load_structured

from .rlt_pilot import feedback_weight, pending_rows, restore_optimizer, save_optimizer
from .rlt_pilot_data import initialization_digest

ARMS = {
    "direct": ("direct", "direct"),
    "slow_rlt": ("direct", "rlt"),
    "slow_damping": ("direct", "matched_damping"),
    "fast_rlt": ("rlt", "direct"),
    "fast_damping": ("matched_damping", "direct"),
}
SEEDS = (131, 149)
TRAIN_WIDTH = 8
FAST_PER_SLOW = 2
TRAIN_SLOW = 4
SLOW_DEPTHS = (2, 4, 6, 8)
UPDATES = 400
BATCH = 32
LR = 0.07


def sigmoid(x):
    return 0.5 + 0.5 * anp.tanh(x / 2)


def shifted(state):
    return anp.concatenate((anp.zeros_like(state[:, :1]), state[:, :-1]), axis=1)


def encode(a, b, width):
    shifts = np.arange(width + 1, dtype=np.int64)
    return (2 * ((a[:, None] >> shifts) & 1) + ((b[:, None] >> shifts) & 1)).astype(np.intp)


def refine(w, inputs, arm, slow, prior=None, trace=None):
    fast_mode, slow_mode = ARMS[arm]
    z = np.zeros(inputs.shape, dtype=np.float32)
    h = np.full(inputs.shape, 0.5, dtype=np.float32) if prior is None else prior
    low, high = w["low"][inputs], w["high"][inputs]
    total_fast = slow * FAST_PER_SLOW
    for macro in range(slow):
        low_cross = low[..., 2] * (h - 0.5)
        generate = sigmoid(low[..., 0] + low_cross)
        pass_through = sigmoid(low[..., 1] + low_cross)
        for visit in range(FAST_PER_SLOW):
            incoming = shifted(z)
            endpoint = generate + (1 - generate) * pass_through * incoming
            eta = feedback_weight(fast_mode, macro * FAST_PER_SLOW + visit, total_fast)
            z = endpoint if eta == 1 else z + eta * (endpoint - z)
        incoming = shifted(z)
        high_cross = high[..., 2] * (h - 0.5)
        endpoint = (1 - incoming) * sigmoid(high[..., 0] + high_cross)
        endpoint += incoming * sigmoid(high[..., 1] + high_cross)
        eta = feedback_weight(slow_mode, macro, slow)
        h = endpoint if eta == 1 else h + eta * (endpoint - h)
        if trace is not None:
            trace.append((z, h))
    return z, h


def loss(w, inputs, targets, arm):
    _, p = refine(w, inputs, arm, TRAIN_SLOW)
    p = anp.clip(p, 1e-6, 1 - 1e-6)
    return -anp.mean(targets * anp.log(p) + (1 - targets) * anp.log1p(-p))


def operands(rng, width, count, *, chain):
    a = rng.integers(1 << width, size=count, dtype=np.int64)
    b = rng.integers(1 << width, size=count, dtype=np.int64)
    if not chain:
        return a, b, np.full(count, -1, dtype=np.int64)
    anchors = rng.integers(0, 4, size=count, dtype=np.int64)
    anchor = 1 << anchors
    lower = anchor - 1
    # Set the column below the anchor to 00, preventing an incoming carry there.
    usable = lower ^ (anchor >> 1)
    a = (a & ~lower) | (a & usable)
    a |= anchor
    high = ((1 << width) - 1) ^ (lower | anchor)
    b = (b & (usable | anchor)) | (~a & high)
    return a, b, anchors


def batches(seed):
    rng = np.random.default_rng(seed + 12209)
    plan = []
    for _ in range(UPDATES):
        a, b, _ = operands(rng, TRAIN_WIDTH, BATCH // 2, chain=False)
        c, d, _ = operands(rng, TRAIN_WIDTH, BATCH // 2, chain=True)
        plan.append(np.stack((np.concatenate((a, c)), np.concatenate((b, d)))))
    return np.asarray(plan)


def train(out: Path, seed: int, arm: str, plan, guard):
    rng = np.random.default_rng(seed)
    w = {name: rng.normal(0, 0.1, (4, 3)).astype(np.float32) for name in ("low", "high")}
    receipt = {
        "seed": seed, "arm": arm, "params": sum(v.size for v in w.values()), "updates": UPDATES,
        "batch": BATCH, "training_width": TRAIN_WIDTH, "slow_visits": TRAIN_SLOW, "fast_per_slow": FAST_PER_SLOW,
        "init_sha256": initialization_digest(w), "batches_sha256": hashlib.sha256(plan.tobytes()).hexdigest(),
    }
    weights_path, summary_path = out / "weights.npz", out / "fit.json"
    if summary_path.exists():
        previous = load_structured(summary_path)
        if any(previous[k] != v for k, v in receipt.items()) or previous["weights_sha256"] != sha256_file(weights_path):
            raise ValueError("finished arm differs from its training receipt")
        return previous
    moment1 = {k: np.zeros_like(v) for k, v in w.items()}
    moment2 = {k: np.zeros_like(v) for k, v in w.items()}
    state = out / "optimizer.npz"
    last, wall = restore_optimizer(state, w, moment1, moment2) if state.exists() else (0, 0.0)
    start = time.monotonic()
    gradient = value_and_grad(loss)
    shifts = np.arange(TRAIN_WIDTH + 1)
    for i in range(last, UPDATES):
        try:
            guard()
        except Exception:
            save_optimizer(state, i, w, moment1, moment2, wall + time.monotonic() - start)
            raise
        a, b = plan[i]
        inputs = encode(a, b, TRAIN_WIDTH)
        labels = (((a + b)[:, None] >> shifts) & 1).astype(np.float32)
        value, grads = gradient(w, inputs, labels, arm)
        if not np.isfinite(value) or any(not np.isfinite(g).all() for g in grads.values()):
            raise FloatingPointError("nonfinite loss or gradient")
        t = i + 1
        for name, param in w.items():
            g = grads[name]
            moment1[name] = 0.9 * moment1[name] + 0.1 * g
            moment2[name] = 0.999 * moment2[name] + 0.001 * g**2
            factor = np.sqrt(1 - 0.999**t) / (1 - 0.9**t)
            param -= LR * factor * moment1[name] / (np.sqrt(moment2[name]) + 1e-8)
        if t % 50 == 0:
            save_optimizer(state, t, w, moment1, moment2, wall + time.monotonic() - start)
            print(canonical_json({"arm": arm, "seed": seed, "update": t, "bit_loss": float(value)}), flush=True)
    temporary = out / "weights.part.npz"
    np.savez(temporary, **w)
    temporary.replace(weights_path)
    receipt["weights_sha256"] = sha256_file(weights_path)
    dump_structured(summary_path, receipt)
    return receipt


def cases(seed: int, exclude=()):
    rng = np.random.default_rng(seed)
    for family, width, count in (("iid", 8, 192), ("width", 16, 192), ("chain", 16, 192)):
        if family == "iid":
            seen = set(exclude)
            candidates = rng.permutation(1 << (2 * width))
            pairs = (divmod(int(x), 1 << width) for x in candidates)
            a, b = np.asarray(list(islice((pair for pair in pairs if pair not in seen), count))).T
            starts = np.full(count, -1)
        else:
            a, b, starts = operands(rng, width, count, chain=family == "chain")
        for i in range(count):
            variants = ("changed", "no_op") if family == "chain" else ("single",)
            for status in variants:
                edited = int(b[i]) ^ (1 << int(starts[i])) if status == "changed" else int(b[i])
                yield {
                    "id": f"{family}-{i:03d}-{status}", "family": family, "status": status,
                    "width": width, "anchor": int(starts[i]), "a": int(a[i]), "b": int(b[i]), "edited": edited,
                }


def words(probs):
    return (probs >= 0.5) @ (1 << np.arange(probs.shape[1]))


def evaluate(out, arm, tasks, guard):
    with np.load(out / "weights.npz", allow_pickle=False) as state:
        weights = dict(state)
    cells = [(arm, slow) for slow in SLOW_DEPTHS]
    if arm == "direct":
        cells += list(product((a for a in ARMS if a != "direct"), (4, 8)))
    for runtime, slow in cells:
        dest = out / f"{runtime}-{slow}.jsonl"
        if dest.exists():
            continue
        with pending_rows(dest) as fh:
            for width in (8, 16):
                subset = [task for task in tasks if task["width"] == width]
                for i in range(0, len(subset), BATCH):
                    guard()
                    batch = subset[i : i + BATCH]
                    a, b, edited = (np.asarray([task[key] for task in batch]) for key in ("a", "b", "edited"))
                    _, before = refine(weights, encode(a, b, width), runtime, slow)
                    patch = encode(a, edited, width)
                    _, reset = refine(weights, patch, runtime, slow)
                    _, carry = refine(weights, patch, runtime, slow, prior=before)
                    change = np.sqrt(np.mean((reset - carry) ** 2, axis=1))
                    outcomes = zip(batch, words(before), words(reset), words(carry), change, strict=True)
                    for item, x, y, z, delta in outcomes:
                        fh.write(canonical_json({
                            "id": item["id"], "before": int(x), "reset": int(y), "carry": int(z),
                            "carry_delta_rms": float(delta),
                        }) + "\n")
