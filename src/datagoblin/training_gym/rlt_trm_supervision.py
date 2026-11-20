"""Matched supervision and feedback schedules in a two-state TRM-style loop."""

from __future__ import annotations

import time
from pathlib import Path

import autograd.numpy as anp
import numpy as np
from autograd import value_and_grad

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import load_structured

from . import rlt_pilot
from . import rlt_pretrain_cell as cell
from .rlt_lean_labels import atomic_record
from .rlt_pilot_data import initialization_digest

TASKS = ("successor", "predecessor", "alternating")
DEPTHS = (4, 6, 8)
TRAIN_DEPTH = 6
NODES = 12
WIDTH = 12
SEEDS = (117, 163, 229)
STEPS = 160
BATCH = 24
LR = 0.008
CHECKPOINTS = (0, 16, 32, 64, 96, 128, STEPS)
SUPERVISION = {
    "final": tuple(cell.ARMS),
    "stepwise": tuple(cell.ARMS),
    "final_at_every_visit": ("direct", "answer_rlt"),
}
SPLITS = {"train": 256, "development": 48, "spectral_fit": 64, "held": 96}


def make_split(rng, per_task, seen):
    permutations, starts, tasks = [], [], []
    for task in range(len(TASKS)):
        for _ in range(per_task):
            pair = []
            while len(pair) < 2:
                cycle = rng.permutation(NODES)
                edges = np.empty(NODES, dtype=int)
                edges[cycle] = np.roll(cycle, -1)
                fingerprint = {tuple(edges.tolist()), tuple(np.argsort(edges).tolist())}
                if fingerprint.isdisjoint(seen):
                    seen.update(fingerprint)
                    pair.append(edges)
            permutations.append(pair)
            starts.append(rng.integers(NODES))
            tasks.append(task)
    return np.asarray(permutations), np.asarray(starts), np.asarray(tasks)


def transitions(permutations, tasks, depth):
    successors = permutations[:, 0]
    backwards = np.argsort(successors, axis=1)
    first = np.where((tasks == 1)[:, None], backwards, successors)
    second = np.where((tasks == 2)[:, None], permutations[:, 1], first)
    return np.stack([first if i % 2 == 0 else second for i in range(depth)], axis=1)


def targets(permutations, starts, tasks, depth):
    """Reference walk using source-to-destination dictionaries."""
    labels = []
    for pair, start, task in zip(permutations, starts, tasks, strict=True):
        first, second = (dict(enumerate(p.tolist())) for p in pair)
        if task == 1:
            first = {v: k for k, v in first.items()}
        active = (first, second if task == 2 else first)
        locations = []
        location = int(start)
        for step in range(depth):
            location = active[step % 2][location]
            locations.append(location)
        labels.append(locations)
    return np.asarray(labels)


def batch(permutations, starts, tasks, depth):
    links = transitions(permutations, tasks, depth)
    return np.argsort(links, axis=-1), starts, tasks, targets(permutations, starts, tasks, depth)


def initialize(seed):
    weights = cell.initialize(seed, WIDTH, 1)
    kept = {key: weights[key] for key in ("self", "message", "partner", "phase", "clock")}
    rng = np.random.default_rng(seed + 100_007)
    return kept | {
        "start": rng.normal(0, 0.55, WIDTH),
        "task": rng.normal(0, 0.10, (len(TASKS), WIDTH)),
        "readout": rng.normal(0, WIDTH**-0.5, WIDTH),
    }


def roll(weights, inputs, arm, intervene=None):
    predecessors, starts, tasks, _ = inputs
    count, depth, nodes = predecessors.shape
    rows = np.arange(count)[:, None]
    reasoning_mode, answer_mode = cell.ARMS[arm]
    reason = (np.arange(nodes) == starts[:, None])[..., None] * weights["start"]
    answer = anp.zeros_like(reason)
    context = weights["task"][tasks, None, :]
    for index in range(depth):
        clock = np.array([index / depth, 1 / depth, 1 - index / depth])
        encoded = context + anp.dot(clock, weights["clock"])
        old_answer = answer
        message = reason[rows, predecessors[:, index]]
        endpoint = cell.recurrent_endpoint(weights, encoded, reason, message, answer, 0)
        reason = cell.mix(reason, endpoint, reasoning_mode, index, depth)
        proposal = cell.recurrent_endpoint(weights, encoded, answer, reason, reason, 1)
        answer = cell.mix(answer, proposal, answer_mode, index, depth)
        if intervene is not None:
            reason, answer = intervene(index + 1, reason, answer)
        yield anp.dot(answer, weights["readout"]), reason, answer, answer - old_answer


def objective(weights, inputs, arm, supervision):
    labels = inputs[-1]
    intermediate = 0.0
    for index, (logits, _, _, _) in enumerate(roll(weights, inputs, arm)):
        endpoint = index == labels.shape[1] - 1
        if endpoint or supervision != "final":
            target = labels[:, index if supervision == "stepwise" else -1]
            loss = anp.mean(rlt_pilot.target_nll(logits, target))
            if endpoint:
                return loss + intermediate
            intermediate += 0.5 * loss / (labels.shape[1] - 1)
    raise ValueError("supervision requires at least one visit")


def score(inputs, frames):
    correct = frames.argmax(axis=-1) == inputs[-1].T
    final = frames.argmax(axis=-1) == inputs[-1][:, -1]
    return {
        name: {
            "items": int(mask.sum()),
            "step_correct": correct[:, mask].sum(axis=1).tolist(),
            "endpoint_correct": final[:, mask].sum(axis=1).tolist(),
            "endpoint_ever_correct": int(final[:, mask].any(axis=0).sum()),
        }
        for i, name in enumerate(TASKS)
        for mask in [inputs[2] == i]
    }


def freeze(root, data_seed):
    manifest = root / "dataset.json"
    fixture = root / "fixtures.npz"
    if manifest.exists():
        registered = load_structured(manifest)
        if registered["seed_fingerprint"] != sha256_json(data_seed) or registered["sha256"] != sha256_file(fixture):
            raise ValueError("dataset seed or fixture changed")
    else:
        seen = set()
        payload = {}
        rng = np.random.default_rng(data_seed)
        for split, count in SPLITS.items():
            values = make_split(rng, count, seen)
            for key, value in zip(("graph", "start", "task"), values, strict=True):
                payload[f"{split}_{key}"] = value
        pending = fixture.with_suffix(".part.npz")
        np.savez(pending, **payload)
        pending.replace(fixture)
        atomic_record(manifest, {"schema": "datagoblin.trm-supervision-data.v1", "tasks": TASKS,
                                 "seed_fingerprint": sha256_json(data_seed), "sha256": sha256_file(fixture),
                                 "per_task": SPLITS, "nodes": NODES})
    with np.load(fixture, allow_pickle=False) as stored:
        return {split: tuple(stored[f"{split}_{key}"] for key in ("graph", "start", "task")) for split in SPLITS}


def fit(root, train, dev, seed, arm, supervision, guard):
    folder = root / "fits" / f"{seed}-{supervision}-{arm}"
    folder.mkdir(parents=True, exist_ok=True)
    weights, plan = initialize(seed), np.random.default_rng(seed + 199_933).integers(len(train[0]), size=(STEPS, BATCH))
    init_sha, plan_sha = initialization_digest(weights), initialization_digest({"batch_plan": plan})
    result, history, checkpoint = (folder / name for name in ("fit.json", "cycles.json", "optimizer.npz"))
    final_weights = folder / "weights.npz"
    if result.exists():
        receipt = load_structured(result)
        if (
            receipt["weights_sha256"] != sha256_file(final_weights)
            or receipt["cycles_sha256"] != sha256_file(history)
            or receipt["initial_sha256"] != init_sha or receipt["batch_plan_sha256"] != plan_sha
        ):
            raise ValueError("a completed fit changed")
        return receipt
    first = {k: np.zeros_like(v) for k, v in weights.items()}
    second = {k: np.zeros_like(v) for k, v in weights.items()}
    done, previous_s = (0, 0)
    if checkpoint.exists():
        done, previous_s = rlt_pilot.restore_optimizer(checkpoint, weights, first, second)
    records = load_structured(history) if history.exists() else {}
    current, began = done, time.monotonic()
    differentiate = value_and_grad(objective)

    def record(step, loss):
        frames = np.stack([logits for logits, _, _, _ in roll(weights, dev, arm)])
        records[str(step)] = {"last_batch_loss": loss, "development": score(dev, frames)}
        atomic_record(history, records)

    try:
        if done in CHECKPOINTS and str(done) not in records:
            record(done, None)
        for step in range(done + 1, STEPS + 1):
            guard()
            indices = plan[step - 1]
            loss, grads = differentiate(weights, tuple(v[indices] for v in train), arm, supervision)
            if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in grads.values()):
                raise FloatingPointError("nonfinite loss or gradient")
            for key, grad in grads.items():
                first[key] = 0.9 * first[key] + 0.1 * grad
                second[key] = 0.999 * second[key] + 0.001 * grad**2
                weights[key] -= LR * (first[key] / (1 - 0.9**step)) / (np.sqrt(second[key] / (1 - 0.999**step)) + 1e-8)
            current = step
            if step in CHECKPOINTS:
                rlt_pilot.save_optimizer(
                    checkpoint, step, weights, first, second, previous_s + time.monotonic() - began,
                )
                record(step, float(loss))
    finally:
        if current > done:
            rlt_pilot.save_optimizer(checkpoint, current, weights, first, second, previous_s + time.monotonic() - began)
    pending = final_weights.with_suffix(".part.npz")
    np.savez(pending, **weights)
    pending.replace(final_weights)
    receipt = {"seed": seed, "arm": arm, "supervision": supervision,
               "parameters": sum(v.size for v in weights.values()),
               "updates": current, "examples_seen": current * BATCH, "cell_calls": 2 * TRAIN_DEPTH * current * BATCH,
               "weights_sha256": sha256_file(final_weights), "cycles_sha256": sha256_file(history),
               "initial_sha256": init_sha, "batch_plan_sha256": plan_sha,
               "wall_s": previous_s + time.monotonic() - began}
    atomic_record(result, receipt)
    print(f"Fit {seed} {supervision} {arm}", flush=True)
    return receipt


def fit_run(root, data_seed, guard):
    inputs = freeze(root, data_seed)
    protocol = {
        "schema": "datagoblin.trm-supervision-fit.v1", "dataset_sha256": sha256_file(root / "dataset.json"),
        "source_sha256": {Path(s.__file__).name: sha256_file(Path(s.__file__)) for s in (cell, rlt_pilot)},
        "experiment_sha256": sha256_file(Path(__file__)), "supervision": SUPERVISION, "schedules": cell.ARMS,
        "seeds": SEEDS, "steps": STEPS, "batch": BATCH, "width": WIDTH, "depth": TRAIN_DEPTH,
        "depths": DEPTHS, "checkpoints": CHECKPOINTS, "lr": LR, "numpy": np.__version__,
    }
    path = root / "fit-protocol.json"
    if path.exists() and sha256_json(load_structured(path)) != sha256_json(protocol):
        raise ValueError("fit protocol changed after registration")
    atomic_record(path, protocol)
    train, dev = (batch(*inputs[name], TRAIN_DEPTH) for name in ("train", "development"))
    fits = []
    for seed in SEEDS:
        group = [fit(root, train, dev, seed, arm, labels, guard)
                 for labels, arms in SUPERVISION.items() for arm in arms]
        if len({(r["initial_sha256"], r["batch_plan_sha256"], r["cell_calls"]) for r in group}) != 1:
            raise ValueError("feedback and supervision arms must share initialization and work")
        fits.extend(group)
    atomic_record(root / "fit-receipts.json", {"protocol_sha256": sha256_file(path), "fits": fits})
