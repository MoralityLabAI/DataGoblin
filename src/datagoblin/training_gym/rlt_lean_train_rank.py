"""Fit a compact recurrent ranker on verified Lean tactic outcomes."""

from __future__ import annotations

import re
import time
from pathlib import Path

import autograd.numpy as anp
import numpy as np
from autograd import value_and_grad

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import load_structured

from . import rlt_lean_grounded as grounded
from .rlt_lean_labels import atomic_record, label_counts, training_labels
from .rlt_pilot import MODES, feedback_weight, restore_optimizer, save_optimizer
from .rlt_pilot_data import initialization_digest

TACTIC_FORMS = ("simp", "rw", "exact", "apply")
LEMMAS = re.compile(r"(?:List|Nat)\.[A-Za-z_][A-Za-z0-9_]*")


def vocabulary(entries):
    return tuple(sorted({piece for entry in entries for piece in grounded.words(entry["name"].split(".", 1)[1])}))


def features(pp, tactics, hits, words):
    target = pp.rsplit("⊢", 1)[-1]
    lhs, _, rhs = target.partition("=")
    left, right = grounded.words(lhs), grounded.words(rhs)
    goal = left | right
    rank = {name: i + 1 for i, name in enumerate(hits)}
    rows = []
    for place, tactic in enumerate(tactics):
        form, *_ = tactic.split()
        names = LEMMAS.findall(tactic)
        operations = [grounded.words(name.split(".", 1)[1]) for name in names]
        union = set().union(*operations)
        active = {name.split(".", 1)[1].split("_", 1)[0].lower() for name in names}
        ranks = [rank.get(name, len(hits) + 1) for name in names]
        row = [form == candidate for candidate in TACTIC_FORMS]
        row += [
            len(tactic.encode("utf-8")) / 96, min(len(names), 3) / 3,
            place / len(tactics), min(ranks, default=0) / max(1, len(hits)),
            max(ranks, default=0) / max(1, len(hits)),
            len(union & goal) / max(1, len(union)), len(union & left) / max(1, len(union)),
            len(union & right) / max(1, len(union)), len(active & left) / max(1, len(active)),
            len(active & right) / max(1, len(active)),
        ]
        for terms in (left, right, union, active):
            row.extend(word in terms for word in words)
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def initialize(seed, dimension, width):
    rng = np.random.default_rng(seed)
    return {
        "input": rng.normal(0, dimension**-0.5, (dimension, width)),
        "local": rng.normal(0, 0.3 * width**-0.5, (width, width)),
        "pool": rng.normal(0, 0.3 * width**-0.5, (width, width)),
        "bias": np.zeros(width),
        "readout": rng.normal(0, width**-0.5, (width,)),
    }


def scores(weights, x, present, mode, depth):
    encoded = anp.dot(x, weights["input"]) + weights["bias"]
    state = anp.zeros(encoded.shape)
    mask = present[..., None]
    count = anp.sum(mask, axis=1, keepdims=True)
    for visit in range(depth):
        pool = anp.sum(state * mask, axis=1, keepdims=True) / count
        endpoint = anp.tanh(encoded + anp.dot(state, weights["local"]) + anp.dot(pool, weights["pool"]))
        eta = feedback_weight(mode, visit, depth)
        state = endpoint if eta == 1 else state + eta * (endpoint - state)
        yield anp.dot(state, weights["readout"])


def masked_logsumexp(logits, mask):
    permitted = anp.where(mask, logits, -10000.0)
    shift = anp.max(permitted, axis=1, keepdims=True)
    return shift[:, 0] + anp.log(anp.sum(anp.exp(permitted - shift), axis=1))


def objective(weights, x, present, labels, mode, depth):
    loss = 0
    for logits in scores(weights, x, present, mode, depth):
        loss += anp.mean(masked_logsumexp(logits, labels >= 0) - masked_logsumexp(logits, labels == 1))
    return loss / depth


class CheckedRanker:
    def __init__(self, weights, words, hits):
        self.weights = weights
        self.words = words
        self.hits = hits

    def predict(self, pp, mode, depth, tactics):
        x = features(pp, tactics, self.hits, self.words)
        for output in scores(self.weights, x[None, ...], np.ones((1, len(tactics)), bool), mode, depth):
            yield output[0]


def prepare_training(rows, words):
    grouped = []
    for row in rows:
        outcomes = row["outcomes"]
        labels = np.array([-1 if a["label"] is None else a["label"] for a in outcomes])
        if 1 not in labels or 0 not in labels:
            continue
        tactics = [a["tactic"] for a in outcomes]
        grouped.append((features(row["goal_pp"], tactics, row["hits"], words), labels))
    if not grouped:
        raise ValueError("no eligible positive/negative training pools")
    max_size = max(len(x) for x, _ in grouped)
    width = grouped[0][0].shape[-1]
    x = np.zeros((len(grouped), max_size, width))
    present = np.zeros(x.shape[:2], bool)
    labels = np.full(x.shape[:2], -1)
    for i, (examples, y) in enumerate(grouped):
        x[i, : len(y)] = examples
        present[i, : len(y)] = True
        labels[i, : len(y)] = y
    return x, present, labels


def train_arm(directory, data, mode, seed, plan, spec, guard):
    directory.mkdir(parents=True, exist_ok=True)
    x, present, labels = data
    depth = spec["depth"]
    weights = initialize(seed, x.shape[-1], spec["width"])
    init = initialization_digest(weights)
    first = {k: np.zeros_like(v) for k, v in weights.items()}
    second = {k: np.zeros_like(v) for k, v in weights.items()}
    initial_loss = float(objective(weights, *data, mode, depth))
    weight_path, receipt_path = directory / "weights.npz", directory / "fit.json"
    if receipt_path.exists():
        receipt = load_structured(receipt_path)
        if sha256_file(weight_path) != receipt["weights_sha256"]:
            raise ValueError("trained ranker changed")
        return receipt
    pending = directory / "optimizer.npz"
    done, elapsed = restore_optimizer(pending, weights, first, second) if pending.exists() else (0, 0.0)
    current = done
    started = time.monotonic()
    differentiate = value_and_grad(objective)
    try:
        for step in range(done + 1, spec["steps"] + 1):
            guard()
            ix = plan[step - 1]
            loss, grads = differentiate(weights, x[ix], present[ix], labels[ix], mode, depth)
            if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in grads.values()):
                raise FloatingPointError("nonfinite grounded ranker loss or gradient")
            for key, gradient in grads.items():
                first[key] = 0.9 * first[key] + 0.1 * gradient
                second[key] = 0.999 * second[key] + 0.001 * gradient**2
                mean = first[key] / (1 - 0.9**step)
                variance = second[key] / (1 - 0.999**step)
                weights[key] -= spec["lr"] * mean / (np.sqrt(variance) + 1e-8)
            current = step
            if current % 50 == 0:
                save_optimizer(pending, current, weights, first, second, elapsed + time.monotonic() - started)
    finally:
        if current > done:
            save_optimizer(pending, current, weights, first, second, elapsed + time.monotonic() - started)
    temp = weight_path.with_suffix(".part.npz")
    np.savez(temp, **weights)
    temp.replace(weight_path)
    receipt = {
        "mode": mode, "seed": seed, "steps": spec["steps"], "batch": spec["batch"], "depth": depth,
        "initialization_sha256": init, "batch_plan_sha256": initialization_digest({"plan": plan}),
        "weights_sha256": sha256_file(weight_path), "parameters": sum(w.size for w in weights.values()),
        "initial_loss": initial_loss, "final_loss": float(objective(weights, *data, mode, depth)),
        "fit_wall_s": elapsed + time.monotonic() - started,
    }
    atomic_record(receipt_path, receipt)
    print(f"Fit seed {seed} {mode}: {receipt['final_loss']:.5f} listwise loss", flush=True)
    return receipt


def fit(root, guard):
    frozen = root / "protocol.json"
    protocol = load_structured(frozen)
    spec = protocol["fit"]
    if any(sha256_file(root / file) != protocol[key] for file, key in (
        ("fixtures.json", "fixtures_sha256"), ("library.json", "library_sha256"),
    )):
        raise ValueError("frozen training inputs changed")
    entries = load_structured(root / "library.json")["entries"]
    words = vocabulary(entries)
    labels = training_labels(root)
    data = prepare_training(labels, words)
    fit_protocol = {
        "protocol_sha256": sha256_file(frozen), "label_manifest_sha256": sha256_file(root / "label-manifest.json"),
        "source_sha256": sha256_file(Path(__file__)),
        "feedback_sha256": sha256_file(Path(__file__).with_name("rlt_pilot.py")),
        "feature_vocabulary": words, "numpy": np.__version__, "labeled": label_counts(labels),
        "eligible_pools": len(data[0]), "spec": spec, "modes": MODES,
    }
    fit_path = root / "fit-protocol.json"
    if fit_path.exists() and sha256_json(load_structured(fit_path)) != sha256_json(fit_protocol):
        raise ValueError("frozen training recipe changed")
    atomic_record(fit_path, fit_protocol)
    receipts = []
    for seed in spec["seeds"]:
        rng = np.random.default_rng(seed + 91253)
        plan = rng.integers(len(data[0]), size=(spec["steps"], spec["batch"]))
        matched = [train_arm(root / "fits" / str(seed) / mode, data, mode, seed, plan, spec, guard) for mode in MODES]
        if len({(r["initialization_sha256"], r["batch_plan_sha256"], r["parameters"]) for r in matched}) != 1:
            raise ValueError("training arms were not matched")
        receipts.extend(matched)
    atomic_record(root / "fit-receipts.json", {"fit_protocol_sha256": sha256_file(fit_path), "arms": receipts})
