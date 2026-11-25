"""Short, token-matched successive-halving study on the pinned mixed corpus."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from statistics import mean

import psutil
import torch
from corpus import VOCAB, load_corpus
from gym import atomic_json, file_sha, schedule
from study import SEEDS, baselines, choose, dense, metrics, objective, score, select_panel, tensor


def macro(m):
    return mean(m[k] for k in ("lean_model", "rule_joint", "repair", "counter"))


def fit(width, arm, seed, train, panel, out, source, contract):
    label = f"w{width}-{arm}-{seed}"
    rec, ck = out / (label + ".json"), out / (label + ".pt")
    if rec.exists():
        row = json.loads(rec.read_text())
        if row["checkpoint_sha256"] != file_sha(ck) or row["contract"] != contract:
            raise ValueError("replay contract mismatch")
        return row
    torch.manual_seed(seed)
    net = dense(width)
    opt = torch.optim.AdamW(net.parameters(), lr=0.0007)
    visits = schedule(2, 1)
    logical = sum(p.numel() for i in visits for p in net.blocks[i].parameters())
    weights = sum(p.numel() for p in net.parameters())
    matrices = sum(p.numel() for i in visits for p in net.blocks[i].parameters() if p.ndim == 2)
    npos = padded = ops = 0
    elapsed = 0.0
    process = psutil.Process()
    peak = 0
    stream = hashlib.sha256()
    trace = []
    for step in range(32):
        examples = choose(train, seed, step, 1)
        x, lengths = tensor(examples)
        stream.update(x.numpy().tobytes())
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = objective(net(x, arm, 2), examples)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        elapsed += time.perf_counter() - start
        npos += int(lengths.sum())
        padded += x.numel()
        ops += 6 * x.numel() * (matrices + VOCAB * width + 2 * len(visits) * x.shape[1] * width)
        peak = max(peak, process.memory_info().rss)
        if step in (15, 31):
            rows, _ = score(net, panel, arm, 2, 1)
            trace.append(
                dict(
                    step=step + 1,
                    tokens=npos,
                    metrics=metrics(rows),
                    train_loss=float(loss.detach()),
                    preclip_norm=float(norm),
                    train_flops_est=ops,
                )
            )
    tmp = ck.with_suffix(".tmp")
    torch.save(net.state_dict(), tmp)
    tmp.replace(ck)
    row = dict(
        id=label,
        arm=arm,
        seed=seed,
        width=width,
        params=weights,
        logical_block_parameters=logical,
        visits=len(visits),
        steps=32,
        tokens=npos,
        padded=padded,
        flops=ops,
        train_seconds=elapsed,
        tokens_per_second=npos / elapsed,
        peak_sampled_rss=peak,
        trace=trace,
        stream=stream.hexdigest(),
        checkpoint_sha256=file_sha(ck),
        corpus=source,
        contract=contract,
    )
    atomic_json(rec, row)
    print(label, npos, round(elapsed, 1), flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pack", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=True)
    train, dev, source = load_corpus(args.pack)
    for v, k in ((1, "lean_pos"), (0, "lean_neg")):
        train[k] = [row for row in train["lean"] if row.good == (v,)]
    panel = select_panel(dev, 8)
    fixed = baselines(train, panel)
    contract = file_sha(Path(__file__)) + "-" + file_sha(Path(__file__).with_name("SCALE_PROTOCOL.md"))
    arms = ["direct", "global_full_damp", "prelude_reread", "site_damp"]
    decision = {}
    for width in (80, 256, 480, 680):
        data = {}
        for a in arms:
            data[a] = [fit(width, a, s, train, panel, args.out, source, contract) for s in SEEDS]
        if any(len({rows[i]["stream"] for rows in data.values()}) != 1 for i in range(len(SEEDS))):
            raise ValueError("unmatched token order")
        scores = {a: [x["trace"][-1]["metrics"] for x in rows] for a, rows in data.items()}
        good = any(
            sum(m["text_nll"] < fixed["text_unigram_nll"] for m in scores[a]) >= 2
            and sum(m["lean_model"] > m["lean_lexical"] or m["repair"] > 0.25 for m in scores[a]) >= 2
            for a in ("direct", "global_full_damp")
        )
        extras = []
        for a in arms[2:]:
            against = scores["direct"]
            win = sum(macro(x) > macro(y) for x, y in zip(scores[a], against, strict=True)) >= 2
            if a == "prelude_reread" and "site_damp" in scores:
                win &= sum(macro(x) > macro(y) for x, y in zip(scores[a], scores["site_damp"], strict=True)) >= 2
            if win and mean(m["text_nll"] for m in scores[a]) <= 1.01 * mean(m["text_nll"] for m in against):
                extras.append(a)
        extras.sort(key=lambda a: -mean(macro(m) for m in scores[a]))
        arms = ["direct", "global_full_damp", *extras[:1]]
        decision[str(width)] = {
            "trained": list(data),
            "controls_competent": good,
            "survivors": arms,
            "baselines": fixed,
            "params": data["direct"][0]["params"],
        }
        atomic_json(args.out / "gates.json", decision)
        if width >= 256 and not good:
            break


if __name__ == "__main__":
    main()
