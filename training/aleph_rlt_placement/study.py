"""Matched dense placement fits on public development cohorts. Run from the repository root."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from corpus import PAD, VOCAB, identity, load_corpus
from gym import ARMS, Aleph, atomic_json, checkpoint, file_sha, schedule
from torch.nn import functional as F

SEEDS = (137, 163, 191)
# Constant source roles: 3 text, 4 Lean, 3 verified-task examples.
# Divide the third lane equally among revision, repair and counterexamples.
COUNTS = {"text": 3, "lean": 4, "rule": 1, "repair": 1, "counter": 1}
WEIGHTS = {"text": 0.5, "lean": 0.25, "rule": 1 / 12, "repair": 1 / 12, "counter": 1 / 12}


def dense(width, vocab=VOCAB, heads=4):
    """Allocate the recipe's optional bridge budget to dense shared core FFNs."""
    return Aleph(width, vocab, heads=heads, kv=heads // 4, core_ff=35 * width // 8)


def tensor(rows):
    sizes = [len(r.prompt) for r in rows]
    x = torch.full((len(sizes), max(sizes)), PAD, dtype=torch.long)
    for i, row in enumerate(rows):
        x[i, : sizes[i]] = torch.tensor(row.prompt)
    return x, torch.tensor(sizes)


def choose(train, seed, step, multiplier):
    rng = random.Random(f"aleph-mix-v1:{seed}:{step}")
    out = []
    for lane, count in COUNTS.items():
        pool = train.get(lane)
        if not pool:
            continue
        if lane == "lean":
            # Balance *training* grades. Preserve natural candidate prevalence at eval.
            for i in range(count * multiplier):
                out.append(rng.choice(train["lean_pos"] if i % 2 else train["lean_neg"]))
        else:
            out.extend(rng.choices(pool, k=count * multiplier))
    return out


def objective(logits, rows):
    parts = defaultdict(list)
    for i, row in enumerate(rows):
        if row.lane == "text":
            loss = F.cross_entropy(logits[i, : len(row.prompt)], torch.tensor(row.good))
        else:
            scores = logits[i, len(row.prompt) - 1, : row.choices]
            loss = scores.logsumexp(0) - scores[list(row.good)].logsumexp(0)
        parts[row.lane].append(loss)
    mass = sum(WEIGHTS[k] for k in parts)
    return sum(WEIGHTS[k] / mass * torch.stack(losses).mean() for k, losses in parts.items())


def select_panel(dev, limit):
    selected = []
    for lane, rows in dev.items():
        groups = defaultdict(list)
        for row in rows:
            groups[row.group if lane == "text" else lane].append(row)
        for group in groups.values():
            keys = sorted({r.root for r in group}, key=lambda k: identity(["panel", k]))[:limit]
            keep = set(keys)
            selected.extend(r for r in group if r.root in keep)
    return selected


@torch.no_grad()
def score(model, rows, arm, r, s, trace=False):
    model.eval()
    result, telemetry = [], []
    # Similar lengths share a padded batch; the last real position is the readout.
    ordered = sorted(rows, key=lambda row: (row.lane, len(row.prompt), row.root, row.order))
    for j in range(0, len(ordered), 12):
        group = ordered[j : j + 12]
        x, lengths = tensor(group)
        logits = model(x, arm, r, s)
        for i, row in enumerate(group):
            n = int(lengths[i])
            rec = {"lane": row.lane, "root": row.root, "group": row.group, "order": row.order}
            if row.lane == "text":
                rec.update(
                    nll=float(F.cross_entropy(logits[i, :n], torch.tensor(row.good), reduction="sum")), positions=n
                )
            else:
                z = logits[i, n - 1, : row.choices]
                pred = int(z.argmax())
                rec.update(pred=pred, hit=pred in row.good, known=bool(row.good))
                if row.lane == "lean":
                    rec.update(score=float(z[1] - z[0]), grade=row.good[0] if row.good else None)
            result.append(rec)
    if trace:
        # No padding in the telemetry input. One fixed public root per lane.
        for lane in sorted({row.lane for row in rows}):
            row = next(row for row in ordered if row.lane == lane)
            visits = []
            model(torch.tensor([row.prompt]), arm, r, s, trace=visits)
            telemetry.append({"lane": lane, "root": row.root, "visits": visits})
    model.train()
    return result, telemetry


def metrics(rows):
    out = {}
    for lane in sorted({r["lane"] for r in rows}):
        data = [r for r in rows if r["lane"] == lane]
        if lane == "text":
            out["text_nll"] = sum(r["nll"] for r in data) / sum(r["positions"] for r in data)
            for group in sorted({r["group"] for r in data}):
                sub = [r for r in data if r["group"] == group]
                out["nll_" + group] = sum(r["nll"] for r in sub) / sum(r["positions"] for r in sub)
        elif lane == "lean":
            goals = defaultdict(list)
            for d in data:
                goals[d["root"]].append(d)
            for method in ("model", "lexical", "pool"):
                proved = 0
                for candidates in goals.values():
                    ranked = sorted(candidates, key=lambda x: ((-x["score"] if method == "model" else 0), x["order"]))
                    prefix = ranked if method == "pool" else ranked[:4]
                    proved += any(x["grade"] == 1 for x in prefix)
                out["lean_" + method] = proved / len(goals)
        elif lane == "rule":
            episodes = defaultdict(list)
            for d in data:
                episodes[d["group"].rsplit("/", 1)[0]].append(d["hit"])
            assert all(len(x) == 2 for x in episodes.values())
            out["rule_joint"] = sum(all(x) for x in episodes.values()) / len(episodes)
        else:
            out[lane] = sum(d["hit"] for d in data) / len(data)
            for group in sorted({r["group"] for r in data}):
                sub = [r for r in data if r["group"] == group]
                out[lane + "_" + group] = sum(r["hit"] for r in sub) / len(sub)
    return out


def baselines(train, panel):
    out = {}
    texts = train.get("text", [])
    if texts:
        counts = Counter(t for row in texts for t in row.good)
        total = sum(counts.values()) + 256
        windows = [row for row in panel if row.lane == "text"]
        out["text_unigram_nll"] = -sum(math.log((counts[t] + 1) / total) for row in windows for t in row.good) / sum(
            len(row.good) for row in windows
        )
    for lane, choices in (("repair", 4), ("counter", 33)):
        # Learn a fixed constant on TRAIN; the transparent cue is an additional strong control.
        fixed = max(range(choices), key=lambda x: sum(x in row.good for row in train[lane]))
        sub = [row for row in panel if row.lane == lane]
        out[lane + "_constant"] = sum(fixed in row.good for row in sub) / len(sub)
        cue_index = -2 if lane == "repair" else 1
        out[lane + "_cue"] = sum(row.prompt[cue_index] in row.good for row in sub) / len(sub)
    return out


def train_cell(args, train, dev, source, seed, arm):
    panel = select_panel(dev, args.roots)
    probe = select_panel(dev, args.probe_roots)
    cell = args.out / f"w{args.width}-r{args.r}-s{seed}-{arm}"
    cell.mkdir(parents=True, exist_ok=True)
    cfg = {
        "schema": 2,
        "width": args.width,
        "r": args.r,
        "s": 1,
        "seed": seed,
        "arm": arm,
        "steps": args.steps,
        "lr": args.lr,
        "multiplier": args.multiplier,
        "roots": args.roots,
        "probe_roots": args.probe_roots,
        "torch": str(torch.__version__),
        "source": source,
        "counts": COUNTS,
        "weights": WEIGHTS,
        "code": {name: file_sha(Path(__file__).with_name(name)) for name in ("study.py", "gym.py", "corpus.py")},
    }
    ck = cell / "checkpoint.pt"
    receipt = cell / "receipt.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old["config"] != cfg or not ck.exists() or file_sha(ck) != old["checkpoint_sha256"]:
            raise ValueError("existing receipt differs from requested cell")
        return
    torch.manual_seed(seed)
    model = dense(args.width)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ledger = {
        "input_positions": 0,
        "loss_positions": 0,
        "padded_positions": 0,
        "lane_examples": {},
        "plan_hash": "",
        "train_s": 0.0,
        "grad": [],
    }
    start = 0
    if ck.exists():
        state = torch.load(ck, map_location="cpu", weights_only=True)
        if state["cfg"] != cfg:
            raise ValueError("refusing changed configuration on resume")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        torch.random.set_rng_state(state["rng"])
        start, ledger = state["step"], state["ledger"]
    for i in range(start, args.steps):
        t = time.monotonic()
        rows = choose(train, seed, i, args.multiplier)
        x, lens = tensor(rows)
        opt.zero_grad(set_to_none=True)
        loss = objective(model(x, arm, args.r), rows)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if i in (0, args.steps // 2, args.steps - 1):
            ledger["grad"].append(
                {
                    "step": i + 1,
                    "loss": float(loss.detach()),
                    "preclip_l2": float(norm),
                    "blocks_postclip_l2": [
                        math.sqrt(sum(float(p.grad.square().sum()) for p in block.parameters()))
                        for block in model.blocks
                    ],
                }
            )
        opt.step()
        ledger["plan_hash"] = identity([ledger["plan_hash"], [(r.root, r.group, r.order, r.prompt) for r in rows]])
        ledger["input_positions"] += int(lens.sum())
        ledger["loss_positions"] += sum(len(row.good) if row.lane == "text" else 1 for row in rows)
        ledger["padded_positions"] += x.numel()
        for row in rows:
            counts = ledger["lane_examples"]
            counts[row.lane] = counts.get(row.lane, 0) + 1
        ledger["train_s"] += time.monotonic() - t
        if (i + 1) % 48 == 0 or i + 1 == args.steps:
            checkpoint(ck, model, opt, i + 1, cfg, ledger)
    t = time.monotonic()
    native, telemetry = score(model, panel, arm, args.r, 1, trace=True)
    sweep = {}
    for runtime, depth in dict.fromkeys([(a, args.r) for a in ARMS] + [(arm, k) for k in (1, 2, 3, 4, 6)]):
        key = f"{runtime}@{depth}"
        path = cell / f"sweep-{key}.json"
        if path.exists():
            part = json.loads(path.read_text())
            if part["cfg"] != identity(cfg):
                raise ValueError("stale crossover")
        else:
            probe_rows, _ = score(model, probe, runtime, depth, 1)
            part = {"cfg": identity(cfg), "metrics": metrics(probe_rows), "rows": probe_rows}
            atomic_json(path, part)
        sweep[key] = part["metrics"]
    detail = {
        "config": cfg,
        "params": sum(p.numel() for p in model.parameters()),
        "ledger": ledger,
        "train_block_positions": ledger["padded_positions"] * len(schedule(args.r, 1)),
        "eval_session_s": time.monotonic() - t,
        "checkpoint_sha256": file_sha(ck),
        "native": metrics(native),
        "baselines": baselines(train, panel),
        "rows": native,
        "trace": telemetry,
        "sweep": sweep,
        "root_counts": {k: len({r.root for r in panel if r.lane == k}) for k in dev},
    }
    atomic_json(receipt, detail)
    print(json.dumps({"cell": cell.name, "metrics": detail["native"], "train_s": ledger["train_s"]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pack", type=Path)
    p.add_argument("--width", type=int, default=80)
    p.add_argument("--steps", type=int, default=192)
    p.add_argument("--multiplier", type=int, default=1)
    p.add_argument("--roots", type=int, default=48)
    p.add_argument("--probe-roots", type=int, default=6)
    p.add_argument("--r", type=int, choices=(2, 3), default=2)
    p.add_argument("--lr", type=float, default=0.0007)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--arms", choices=ARMS, nargs="+", default=["direct"])
    args = p.parse_args()
    if (
        min(args.width, args.steps, args.multiplier, args.roots, args.probe_roots) < 1
        or args.width % 8
        or not 0 < args.lr < 1
    ):
        p.error("invalid shape or training budget")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    train, dev, source = load_corpus(args.pack)
    if train.get("lean"):
        for key, i in (("lean_pos", 1), ("lean_neg", 0)):
            train[key] = [r for r in train["lean"] if r.good == (i,)]
            if not train[key]:
                raise ValueError("training Lean pack must include both outcomes")
    for seed in args.seeds:
        for arm in args.arms:
            train_cell(args, train, dev, source, seed, arm)


if __name__ == "__main__":
    main()
