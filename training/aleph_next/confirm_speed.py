"""Same-core paired training throughput for the dense direct and constant policies."""

import argparse
import json
import random
import time
from pathlib import Path
from statistics import median

import torch
from corpus import load_corpus
from gym import atomic_json, file_sha
from study import SEEDS, dense, objective, tensor

ARMS = ("direct", "global_full_damp")


def batches(train, seed, step):
    rng = random.Random(f"aleph14-throughput/{seed}/{step}")
    return [[rng.choice(train["text"])], rng.choices(train["repair"], k=4), rng.choices(train["rule"], k=2)]


def updates(model, opt, arm, groups):
    opt.zero_grad(set_to_none=True)
    positions = padded = 0
    for rows in groups:
        x, lengths = tensor(rows)
        (objective(model(x, arm, 2), rows) / len(groups)).backward()
        positions += int(lengths.sum())
        padded += x.numel()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    opt.step()
    return positions, padded


def measure(prefix, train, seed):
    # A common trained starting tensor isolates policy overhead within seed.
    path = prefix / f"w256-direct-{seed}.pt"
    meta = json.loads(path.with_suffix(".json").read_bytes())
    if file_sha(path) != meta["checkpoint_sha256"]:
        raise ValueError("changed calibration weights")
    nets, opts = {}, {}
    for arm in ARMS:
        net = dense(256)
        net.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        nets[arm] = net
        opts[arm] = torch.optim.AdamW(net.parameters(), lr=0.0003)
    if any(sum(p.numel() for p in m.parameters()) != 14_179_584 for m in nets.values()):
        raise ValueError("unexpected calibration model")
    rows = []
    for step in range(5):
        group = batches(train, seed, step)
        order = ARMS if (step + SEEDS.index(seed)) % 2 == 0 else ARMS[::-1]
        obs = {}
        for arm in order:
            wall, cpu = time.perf_counter(), time.process_time()
            positions, padded = updates(nets[arm], opts[arm], arm, group)
            obs[arm] = dict(
                cpu=time.process_time() - cpu, wall=time.perf_counter() - wall, positions=positions, padded=padded
            )
        if step:  # Warm-up includes Adam allocation and the first backward.
            rows.append(obs)
    ratios = {}
    for clock in ("cpu", "wall"):
        ratios[clock] = sum(r["direct"][clock] for r in rows) / sum(r["global_full_damp"][clock] for r in rows)
    return dict(seed=seed, calibration_parent=file_sha(path), ratios=ratios, timed_pairs=rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("pack", "prefix", "out"):
        cli.add_argument("--" + key, type=Path, required=True)
    args = cli.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    train, _, source = load_corpus(args.pack)
    records = [measure(args.prefix, train, s) for s in SEEDS]
    ratios = {t: median(x["ratios"][t] for x in records) for t in ("cpu", "wall")}
    atomic_json(
        args.out,
        dict(
            records=records,
            median_ratios=ratios,
            passed=ratios["cpu"] >= 0.9,
            corpus=source,
            timing="matched positions / single-process seconds",
            source_sha256=file_sha(Path(__file__)),
        ),
    )
    print(ratios)


if __name__ == "__main__":
    main()
