"""MOLT-04: actual tied utility-head training and matched work development panels."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from datagoblin.training_gym.molt_goblin import digest
from datagoblin.training_gym.molt_retention import (
    VARIANTS,
    baseline_scores,
    episode,
    make_world,
    public_features,
)

CAPACITIES = (1, 2, 4, 6)
BASELINES = ("fifo", "random", "scope", "generator_rank", "hindsight")
ARMS = ("single", "tied3", "single_reinvest", "tied3_shuffled")


def features(world) -> np.ndarray:
    return 2.0 * np.asarray(public_features(world), dtype=np.float64) - 1.0


def dataset(worlds, shuffle_seed: int | None = None):
    if not worlds or any(w.split != "train" for w in worlds):
        raise ValueError("fit data must contain TRAIN worlds only")
    xs, ys = [], []
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    for world in worlds:
        xs.append(features(world))
        labels = np.asarray([float(i in world.query) for i in range(len(world.entries))])
        if rng is not None:
            labels = rng.permutation(labels)
        ys.append(labels)
    return np.concatenate(xs), np.concatenate(ys)


class TiedScorer:
    """Shared tanh cell with exact BPTT; independent of native TRM/MoLT code."""

    def __init__(self, seed: int, width: int = 16):
        rng = np.random.default_rng(seed)
        self.params = {
            "x": rng.normal(0, 0.3, (5, width)),
            "h": rng.normal(0, 0.15, (width, width)),
            "b": np.zeros(width),
            "o": rng.normal(0, 0.3, width),
            "c": np.zeros(1),
        }

    def forward(self, x: np.ndarray, loops: int):
        p = self.params
        states = [np.zeros((x.shape[0], p["b"].size))]
        fixed = x @ p["x"] + p["b"]
        for _ in range(loops):
            states.append(np.tanh(fixed + states[-1] @ p["h"]))
        return states[-1] @ p["o"] + p["c"][0], states

    def loss_grad(self, x: np.ndarray, y: np.ndarray, loops: int):
        p = self.params
        logits, states = self.forward(x, loops)
        loss = float(np.mean(np.logaddexp(0.0, logits) - y * logits))
        dlogits = (1 / (1 + np.exp(-np.clip(logits, -60, 60))) - y) / len(y)
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        grads["o"] = states[-1].T @ dlogits
        grads["c"][0] = dlogits.sum()
        dh = dlogits[:, None] * p["o"]
        for r in range(loops, 0, -1):
            dz = dh * (1 - states[r] ** 2)
            grads["x"] += x.T @ dz
            grads["h"] += states[r - 1].T @ dz
            grads["b"] += dz.sum(axis=0)
            dh = dz @ p["h"].T
        return loss, grads

    def identity(self) -> str:
        return digest({k: v.tolist() for k, v in self.params.items()})


def fit(worlds, arm: str, seed: int, steps: int, batch_size: int = 64):
    if arm not in ARMS or not 1 <= steps <= 1000 or not 1 <= batch_size <= 256:
        raise ValueError("invalid fit")
    loops = 3 if arm.startswith("tied3") else 1
    updates = 3 * steps if arm == "single_reinvest" else steps
    x, y = dataset(worlds, seed + 918273 if arm == "tied3_shuffled" else None)
    model = TiedScorer(seed)
    initial = model.identity()
    initial_loss, _ = model.loss_grad(x, y, loops)
    rng = np.random.default_rng(seed + 100000)
    start = time.perf_counter()
    for _ in range(updates):
        ids = rng.integers(0, len(y), size=batch_size)
        _, grads = model.loss_grad(x[ids], y[ids], loops)
        for k, g in grads.items():
            model.params[k] -= 0.06 * g
    elapsed = time.perf_counter() - start
    final_loss, _ = model.loss_grad(x, y, loops)
    if not np.isfinite(final_loss) or model.identity() == initial:
        raise RuntimeError("invalid fitted head")
    return model, {"arm": arm, "training_seed": seed, "loops": loops, "updates": updates,
                   "processed_examples": updates * batch_size,
                   "forward_cell_example_visits": updates * batch_size * loops,
                   "parameters_total": sum(v.size for v in model.params.values()),
                   "parameters_active": sum(v.size for k, v in model.params.items()
                                            if k != "h" or loops > 1),
                   "initial_weights_digest": initial, "weights_digest": model.identity(),
                   "initial_loss": initial_loss, "final_loss": final_loss, "wall_seconds": elapsed,
                   "label_control": arm == "tied3_shuffled"}


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = (row["policy"], row["split"], row["variant"], row["capacity"], row["verifier_budget"])
        groups[key].append(row)
    result = []
    for (policy, split, variant, capacity, budget), group in sorted(groups.items()):
        roots = defaultdict(list)
        for row in group:
            roots[row["root_id"]].append(int(row["verified"]))
        result.append({"policy": policy, "split": split, "variant": variant, "capacity": capacity,
                       "verifier_budget": budget, "rows": len(group), "world_roots": len(roots),
                       "model_seeds": len({r["training_seed"] for r in group if r["training_seed"] is not None}),
                       "verified": sum(r["verified"] for r in group),
                       "equal_world_macro": sum(sum(v) / len(v) for v in roots.values()) / len(roots),
                       "mean_verifier_calls": sum(r["verifier_calls"] for r in group) / len(group),
                       "mean_hits": sum(r["hits"] for r in group) / len(group)})
    return result


def run(train_roots: int, dev_roots: int, steps: int, seeds: tuple[int, ...]):
    if not 1 <= train_roots <= 4096 or not 1 <= dev_roots <= 256:
        raise ValueError("root count outside development bounds")
    if not 1 <= len(seeds) <= 8 or len(set(seeds)) != len(seeds) or not all(0 <= s <= 100000 for s in seeds):
        raise ValueError("invalid training seeds")
    train = [make_world("train", 41, i) for i in range(train_roots)]
    dev = [make_world(split, 713, i)
           for split in ("calibration_short", "calibration_long") for i in range(dev_roots)]
    train_ids = {w.root_id for w in train}
    assert not train_ids.intersection(w.root_id for w in dev)
    trained, rows, receipts = [], [], []
    for seed in seeds:
        for arm in ARMS:
            model, receipt = fit(train, arm, seed, steps)
            receipts.append(receipt)
            trained.append((arm, seed, receipt["loops"], model))
    for world in dev:
        x = features(world)
        frozen_scores = [(arm, seed, tuple(float(s) for s in model.forward(x, loops)[0]))
                         for arm, seed, loops, model in trained]
        for variant in VARIANTS:
            policies = [(p, None, baseline_scores(world, p, variant=variant)) for p in BASELINES] + frozen_scores
            for policy, seed, scores in policies:
                for capacity in CAPACITIES:
                    for budget in (0, 1):
                        result = episode(world, scores, capacity, budget, variant)
                        rows.append({"policy": policy, "training_seed": seed, **result})
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in ("molt_goblin.py", "molt_retention.py", "molt_retention_train.py")}
    summary = summarize(rows)
    return {"schema": "molt-retention-v4", "status": "development_only",
            "train_root_count": len(train), "dev_root_count": len(dev), "training_seeds": list(seeds),
            "train_manifest_digest": digest([asdict(w) for w in train]),
            "dev_manifest_digest": digest([asdict(w) for w in dev]),
            "numpy_version": np.__version__, "sources": sources, "fits": receipts, "summary": summary,
            "rows_digest": digest(rows), "rows": rows, "actual_flops": None,
            "model_kind": "tiny_numpy_tied_utility_head", "checkpoint_rank_eligible": False,
            "promotion_authorized": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-roots", type=int, default=1024)
    parser.add_argument("--dev-roots", type=int, default=128)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 37, 41, 43])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refuse to overwrite")
    report = run(args.train_roots, args.dev_roots, args.steps, tuple(args.seeds))
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
    # Fixed compact view for CI logs; raw report retains all capacities and variants.
    selected = [r for r in report["summary"]
                if r["capacity"] == 2 and r["verifier_budget"] == 1]
    print(json.dumps({"status": report["status"], "fits": report["fits"], "summary": selected,
                      "rows": len(report["rows"]), "rows_digest": report["rows_digest"]}, sort_keys=True))


if __name__ == "__main__":
    main()
