"""Matched feedback arms in a small graph-conditioned TRM-style loop."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable

import autograd.numpy as anp
import numpy as np
from autograd import value_and_grad

from datagoblin.hashing import sha256_file
from datagoblin.serialization import canonical_json, dump_structured, load_structured

from .rlt_pilot import MODES, feedback_weight, pending_rows, restore_optimizer, save_optimizer, target_nll
from .rlt_pilot_data import RULE_SPLITS, initialization_digest, jsonl
from .rlt_revision import NODES

NODE_ID = {node: i for i, node in enumerate(NODES)}
WIDTH = len(NODES)
HIDDEN = 12
TRAIN_DEPTH = 6
DEPTHS = (4, 6, 10)
SEEDS = (131, 149)
STEPS = 240
BATCH = 12
LR = 0.008


def encode_tasks(tasks):
    start = np.asarray([NODE_ID[t["start"]] for t in tasks])
    hops = np.asarray([t["steps"] for t in tasks], dtype=np.float32)
    maps = (dict(task["links"]) for task in tasks)
    before = np.asarray([[NODE_ID[edges[n]] for n in NODES] for edges in maps])
    after = before.copy()
    for i, t in enumerate(tasks):
        after[i, NODE_ID[t["patch"]["source"]]] = NODE_ID[t["patch"]["destination"]]
    eye = np.eye(WIDTH, dtype=np.float32)
    return eye[start], hops, before, after


def prepare(tasks):
    start, hops, before, after = encode_tasks(tasks)
    eye = np.eye(WIDTH, dtype=np.float32)
    return start, hops, eye[before], eye[after]


def init_weights(seed):
    rng = np.random.default_rng(seed)
    return {
        "up": rng.normal(0, 0.09, (9, HIDDEN)).astype(np.float32),
        "bias": np.zeros(HIDDEN, dtype=np.float32),
        "down": rng.normal(0, 0.09, HIDDEN).astype(np.float32),
        "scale": np.array(2.0, dtype=np.float32),
    }


def local_cell(w, z, message, y, start, clock, phase):
    # One shared nonlinear operator serves the reasoning and answer phases.
    inputs = anp.stack((z, message, y, start, *clock, np.full(z.shape, phase, dtype=np.float32)), axis=-1)
    return anp.dot(anp.tanh(anp.dot(inputs, w["up"]) + w["bias"]), w["down"])


def refine(
    w, edges, start, hops, prior_answer, mode, depth, trace=None, *, placement="reasoning", clock_depths=None,
    stop_when: Callable[[int, np.ndarray], bool] | None = None,
):
    z = start
    y = prior_answer
    ones = anp.ones_like(start)
    phase_depth, step_depth = (depth, depth) if clock_depths is None else clock_depths
    for visit in range(depth):
        clock = (
            ones * (hops[:, None] / 10),
            ones * (visit / phase_depth),
            ones * (1 / step_depth),
            ones * ((hops[:, None] - visit - 1) / 10),
        )
        # Route one edge of latent mass, with scatter-add semantics for the patched graph.
        routed = anp.matmul(z[:, None, :], edges)[:, 0, :]
        fraction = 0.5 + 0.5 * anp.tanh(local_cell(w, z, routed, y, start, clock, -1))
        proposed = z + fraction * (routed - z)
        eta = feedback_weight(mode, visit, depth)
        updated = z + eta * (proposed - z) if placement == "reasoning" and eta != 1 else proposed
        # The answer phase reads the routed reasoning state without traversing an extra edge.
        endpoint = updated + 0.1 * local_cell(w, updated, updated, y, start, clock, 1)
        y = y + eta * (endpoint - y) if placement == "answer" and eta != 1 else endpoint
        if trace is not None:
            trace.append((updated, updated - z, y))
        z = updated
        if stop_when is not None and stop_when(visit + 1, y):
            break
    logits = anp.exp(anp.clip(w["scale"], -3, 6)) * y
    probabilities = anp.exp(logits - anp.max(logits, axis=1, keepdims=True))
    return logits, probabilities / anp.sum(probabilities, axis=1, keepdims=True)


def two_turns(w, data, mode, depth, traces=(None, None), *, placement="reasoning"):
    start, hops, before, after = data
    first, answer = refine(
        w, before, start, hops, anp.zeros_like(start), mode, depth, traces[0], placement=placement,
    )
    second, _ = refine(w, after, start, hops, answer, mode, depth, traces[1], placement=placement)
    return first, second


def objective(w, data, gold, mode, placement):
    first, second = two_turns(w, data, mode, TRAIN_DEPTH, placement=placement)
    return 0.5 * anp.mean(target_nll(first, gold[:, 0]) + target_nll(second, gold[:, 1]))


def load_training(pack):
    tasks = jsonl(pack / "train.tasks.jsonl")
    labels = jsonl(pack / "train.targets.jsonl")
    if [t["id"] for t in tasks] != [r["id"] for r in labels]:
        raise ValueError("training items do not match their targets")
    targets = np.array([[NODE_ID[r["before"]], NODE_ID[r["after"]]] for r in labels])
    return encode_tasks(tasks), targets


def train(pack, out, mode, seed, training, guard: Callable[[], None], *, placement="reasoning"):
    receipt_path = out / "train.json"
    weights_path = out / "weights.npz"
    if receipt_path.exists():
        receipt = load_structured(receipt_path)
        if receipt["steps"] != STEPS or receipt["weights_sha256"] != sha256_file(weights_path):
            raise ValueError("trained weights no longer match the run receipt")
        if receipt.get("placement", "reasoning") != placement:
            raise ValueError("completed arm used a different feedback placement")
        return receipt
    data, gold = training
    weights = init_weights(seed)
    initial = initialization_digest(weights)
    first = {k: np.zeros_like(w) for k, w in weights.items()}
    second = {k: np.zeros_like(w) for k, w in weights.items()}
    progress = out / "optimizer.npz"
    completed, elapsed = restore_optimizer(progress, weights, first, second) if progress.exists() else (0, 0.0)
    history = out / "updates.jsonl"
    lines = history.read_text(encoding="utf-8").splitlines()[:completed] if history.exists() else []
    if len(lines) != completed or (lines and json.loads(lines[-1])["step"] != completed):
        raise ValueError("optimizer checkpoint and update log disagree")
    with pending_rows(history) as fh:
        fh.writelines(line + "\n" for line in lines)
    eye = np.eye(WIDTH, dtype=np.float32)
    rng = np.random.default_rng(seed + 20261120)
    plan = rng.integers(len(gold), size=(STEPS, BATCH))
    schedule_hash = hashlib.sha256(plan.tobytes()).hexdigest()
    started = time.monotonic()
    differentiate = value_and_grad(objective)
    step_done = completed
    with history.open("a", encoding="utf-8", newline="\n") as fh:
        for step in range(completed + 1, STEPS + 1):
            try:
                guard()
            except Exception:
                if step_done > completed:
                    fh.flush()
                    save_optimizer(progress, step_done, weights, first, second, elapsed + time.monotonic() - started)
                raise
            ix = plan[step - 1]
            batch = (data[0][ix], data[1][ix], eye[data[2][ix]], eye[data[3][ix]])
            loss, grads = differentiate(weights, batch, gold[ix], mode, placement)
            norm = math.sqrt(sum(float(np.sum(g**2)) for g in grads.values()))
            if not math.isfinite(loss) or not math.isfinite(norm):
                raise FloatingPointError("nonfinite refinement loss or gradient")
            for key, param in weights.items():
                grad = grads[key] / max(1, norm)
                first[key] = 0.9 * first[key] + 0.1 * grad
                second[key] = 0.999 * second[key] + 0.001 * grad**2
                mean = first[key] / (1 - 0.9**step)
                var = second[key] / (1 - 0.999**step)
                param -= LR * mean / (np.sqrt(var) + 1e-8)
            row = {"step": step, "loss": float(loss), "elapsed_s": elapsed + time.monotonic() - started}
            fh.write(canonical_json(row) + "\n")
            step_done = step
            if step % 40 == 0 or step == STEPS:
                fh.flush()
                save_optimizer(progress, step, weights, first, second, row["elapsed_s"])
                print(canonical_json({"seed": seed, "mode": mode, **row}), flush=True)
    pending = weights_path.with_suffix(".part.npz")
    np.savez(pending, **weights)
    os.replace(pending, weights_path)
    receipt = {
        "mode": mode, "seed": seed, "steps": STEPS, "batch": BATCH, "depth": TRAIN_DEPTH,
        "init_sha256": initial, "schedule_sha256": schedule_hash, "weights_sha256": sha256_file(weights_path),
        "updates_sha256": sha256_file(history), "parameters": sum(w.size for w in weights.values()),
        "cell_calls": STEPS * BATCH * TRAIN_DEPTH * 4, "training_manifest_sha256": sha256_file(pack / "manifest.json"),
        "placement": placement,
    }
    dump_structured(receipt_path, receipt)
    return receipt


def diagnostic_row(states):
    return [
        {"latent_rms": float(np.sqrt(np.mean(z**2))), "update_rms": float(np.sqrt(np.mean(dz**2))),
         "answer_rms": float(np.sqrt(np.mean(y**2))), "argmax": int(y[0].argmax())}
        for z, dz, y in states
    ]


def probe_answer_prior(w, data, mode, depth, item_id, placement):
    start, hops, before, after = data
    _, prior = refine(w, before, start, hops, np.zeros_like(start), mode, depth, placement=placement)
    rng = np.random.default_rng(int(item_id[:16], 16))
    direction = rng.normal(size=start.shape).astype(np.float32)
    direction -= direction.mean()
    perturbation = 0.01 * direction / np.sqrt(np.mean(direction**2))
    input_rms = float(np.sqrt(np.mean(perturbation**2)))
    clean_states, perturbed_states = [], []
    clean, _ = refine(w, after, start, hops, prior, mode, depth, clean_states, placement=placement)
    noisy, _ = refine(w, after, start, hops, prior + perturbation, mode, depth, perturbed_states, placement=placement)
    return {
        "input_rms": input_rms,
        "answer_gain": [
            float(np.sqrt(np.mean((second[2] - first[2]) ** 2)) / input_rms)
            for first, second in zip(clean_states, perturbed_states, strict=True)
        ],
        "flipped": bool(clean.argmax() != noisy.argmax()),
    }


def evaluate(pack, out, trained_mode, guard, *, placement="reasoning", probe_prior=False):
    with np.load(out / "weights.npz", allow_pickle=False) as state:
        weights = dict(state)
    # Native-depth sweep plus a fixed-weight schedule crossover at six visits.
    evaluations = [(trained_mode, k) for k in DEPTHS] + [(m, TRAIN_DEPTH) for m in MODES if m != trained_mode]
    for mode, depth in evaluations:
        for split in RULE_SPLITS:
            prediction = out / f"{split}-{mode}-k{depth}.jsonl"
            trace_path = out / f"{split}-{mode}-k{depth}.trace.jsonl"
            if prediction.exists():
                if not trace_path.exists():
                    raise ValueError("completed scores are missing their trace")
                continue
            tasks = jsonl(pack / f"{split}.tasks.jsonl")
            with pending_rows(prediction) as pred, pending_rows(trace_path) as trace:
                for i in range(0, len(tasks), BATCH):
                    guard()
                    batch = tasks[i : i + BATCH]
                    first, second = two_turns(weights, prepare(batch), mode, depth, placement=placement)
                    for task, p1, p2 in zip(batch, first.argmax(1), second.argmax(1), strict=True):
                        pred.write(canonical_json({"id": task["id"], "before": NODES[p1], "after": NODES[p2]}) + "\n")
                for task in tasks[:8]:
                    guard()
                    stages: tuple[list, list] = ([], [])
                    data = prepare([task])
                    two_turns(weights, data, mode, depth, traces=stages, placement=placement)
                    row = {"id": task["id"], "before": diagnostic_row(stages[0]), "after": diagnostic_row(stages[1])}
                    if probe_prior:
                        row["prior_probe"] = probe_answer_prior(weights, data, mode, depth, task["id"], placement)
                    trace.write(canonical_json(row) + "\n")
        print(
            canonical_json({"trained_mode": trained_mode, "runtime_mode": mode, "evaluated_depth": depth}),
            flush=True,
        )
