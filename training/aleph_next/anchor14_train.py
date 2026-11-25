"""Resume paired 14M dense checkpoints. All training and horizon selection use public data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "aleph_rlt_placement"
sys.path.insert(0, str(LEGACY))
from confirm_reason import train_repair_rows, train_rule_rows  # noqa: E402
from corpus import REPAIR, SEP, Row, identity, load_corpus  # noqa: E402
from gym import atomic_json, file_sha, schedule  # noqa: E402
from study import SEEDS, choose, dense, metrics, score, tensor  # noqa: E402

ARMS = ("direct", "global_full_damp")
HORIZONS = (64, 128, 256, 384, 512)
MASS = {"text": 0.38, "full": 0.25, "suffix": 0.10, "compact": 0.22, "other": 0.05}
PARAMS = 14_179_584
PREFIX = 32


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_tensor(path, state):
    tmp = path.with_suffix(".saving")
    torch.save(state, tmp)
    tmp.replace(path)


def recipe():
    files = [Path(__file__), HERE / "ANCHOR14_PROTOCOL.md", HERE / "confirm_reason.py"]
    files += [LEGACY / f"{stem}.py" for stem in ("study", "gym", "corpus")]
    return identity({p.name: file_sha(p) for p in files})


def load_inputs(pack):
    train, dev, source = load_corpus(pack)
    for val, key in ((1, "lean_pos"), (0, "lean_neg")):
        train[key] = [r for r in train["lean"] if r.good == (val,)]
    buckets = defaultdict(list)
    for row in train["repair"]:
        prev, op, arg = row.prompt[-4:-1]
        target = arg if op == 16 else (prev + arg) % 4
        if row.good != (target,) or op not in (16, 17):
            raise ValueError("invalid native repair target")
        buckets[prev, op, arg].append(row)
    if len(buckets) != 32:
        raise ValueError("native train misses a transition")
    return train, dev, source, buckets


def logical_rows(train, buckets, seed, step):
    phase = step % 4
    packet = choose(train, seed, PREFIX + step // 4, 1)
    if [r.lane for r in packet] != ["text"] * 3 + ["lean"] * 4 + ["rule", "repair", "counter"]:
        raise ValueError("original mix layout changed")
    rng = random.Random(f"aleph-anchor14-v1:{seed}:{step}")
    text = packet[phase] if phase < 3 else rng.choice(train["text"])
    rows = [("text", text), ("other", packet[3 + phase])]
    if phase < 3:
        rows.append(("full" if phase == 1 else "other", packet[7 + phase]))
    else:
        rows.extend(("other", r) for r in train_rule_rows(seed, step, 4))
    cycle = sorted(buckets)
    random.Random(f"aleph-anchor14-transition:{seed}:{step // 8}").shuffle(cycle)
    for key in cycle[4 * (step % 8) : 4 * (step % 8 + 1)]:
        row = rng.choice(buckets[key])
        rows.append(("full", row))
        suffix = (REPAIR, SEP, *key, SEP)
        rows.append(("suffix", Row("repair", identity(["suffix", key]), suffix, row.good, 4, "suffix")))
    rows.extend(("compact", r) for r in train_repair_rows(seed, step, 8))
    return rows


def microbatches(tagged):
    ordered = sorted(tagged, key=lambda pair: len(pair[1].prompt))
    current = []
    for item in ordered:
        n = len(item[1].prompt)
        if current and (len(current) >= 8 or n * (len(current) + 1) > 192):
            yield current
            current = []
        current.append(item)
    if current:
        yield current


def input_digest(tagged):
    return identity([[tag, r.lane, r.root, r.prompt, r.good, r.choices, r.group, r.order] for tag, r in tagged])


def earn_step(net, opt, rows, arm, step):
    count = Counter(tag for tag, row in rows)
    if set(count) != set(MASS):
        raise ValueError("missing loss lane")
    opt.param_groups[0]["lr"] = 0.00035 * min((step + 1) / 4, 1)
    opt.zero_grad(set_to_none=True)
    loss_sum, padded = 0.0, 0
    for group in microbatches(rows):
        x, lengths = tensor([r for _, r in group])
        z = net(x, arm, 2, 1)
        terms = []
        for i, (tag, r) in enumerate(group):
            if r.lane == "text":
                loss = F.cross_entropy(z[i, : lengths[i]], torch.tensor(r.good))
            else:
                scores = z[i, lengths[i] - 1, : r.choices]
                loss = scores.logsumexp(0) - scores[list(r.good)].logsumexp(0)
            terms.append(loss * (MASS[tag] / count[tag]))
        loss = torch.stack(terms).sum()
        loss.backward()
        loss_sum += float(loss.detach())
        padded += x.numel()
    if any(p.grad is None for p in net.parameters()):
        raise ValueError("a physical parameter is disconnected from the objective")
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
    opt.step()
    return loss_sum, float(norm), padded


def anchors(net, arm, train, dev):
    text, repairs = dev["text"], dev["repair"]
    scored, _ = score(net, [*text, *repairs], arm, 2, 1)
    result = metrics(scored)
    freq = Counter(t for row in train["text"] for t in row.good)
    denom = sum(freq.values()) + 256
    positions = sum(len(r.good) for r in text)
    unigram = -sum(math.log((freq[t] + 1) / denom) for r in text for t in r.good) / positions
    cue = sum(r.prompt[-2] in r.good for r in repairs) / len(repairs)
    result.update(
        text_unigram_nll=unigram, repair_cue=cue, repair_chance=0.25, text_positions=positions, repair_rows=len(repairs)
    )
    result["competent"] = bool(result["text_nll"] < unigram and result["repair"] > max(cue, 0.25))
    return result


def root_table(rows, roots):
    for row in rows:
        if row.lane != "text":
            roots.setdefault(row.lane, set()).add(row.root)


def prefix_checked(args, train, source):
    name = f"w256-{args.arm}-{args.seed}"
    meta, ck = read(args.prefix_dir / (name + ".json")), args.prefix_dir / (name + ".pt")
    mate = ARMS[1 - ARMS.index(args.arm)]
    peer = read(args.prefix_dir / f"w256-{mate}-{args.seed}.json")
    h, roots = hashlib.sha256(), {}
    for j in range(PREFIX):
        rows = choose(train, args.seed, j, 1)
        h.update(tensor(rows)[0].numpy().tobytes())
        root_table(rows, roots)
    if (
        meta["stream"] != h.hexdigest()
        or meta["stream"] != peer["stream"]
        or meta["tokens"] != peer["tokens"]
        or meta["padded"] != peer["padded"]
        or meta["params"] != PARAMS
        or meta["steps"] != PREFIX
        or meta["visits"] != 24
        or meta["corpus"] != source
        or meta["checkpoint_sha256"] != file_sha(ck)
    ):
        raise ValueError("archived prefix mismatch")
    return meta, ck, roots


def cell(args):
    return args.out / "cells" / f"{args.arm}-{args.seed}"


def latest_payload(key, ledger, net, opt, roots, logs):
    return dict(
        key=key,
        ledger=ledger,
        model=net.state_dict(),
        optimizer=opt.state_dict(),
        roots={k: sorted(v) for k, v in roots.items()},
        trace=logs,
    )


def fit(args):
    train, dev, source, buckets = load_inputs(args.pack)
    meta, ck, roots = prefix_checked(args, train, source)
    folder = cell(args)
    folder.mkdir(parents=True, exist_ok=True)
    contract = recipe()
    key = dict(arm=args.arm, seed=args.seed, recipe=contract, corpus=source, prefix_sha=meta["checkpoint_sha256"])
    torch.manual_seed(args.seed)
    model = dense(256)
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != PARAMS or len(schedule(2, 1)) != 24:
        raise ValueError("architecture mismatch")
    opt = torch.optim.AdamW(model.parameters(), lr=0.00035, weight_decay=0.01, foreach=False)
    latest = folder / "latest.pt"
    ledger = dict(step=0, tokens=0, padded=0, labels=0, seconds=0.0, stream=meta["stream"])
    logs = []
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        if saved["key"] != key:
            raise ValueError("continuation recipe or prefix changed")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        ledger, logs = saved["ledger"], saved["trace"]
        roots = {k: set(v) for k, v in saved["roots"].items()}
        del saved
    else:
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    end = args.until
    if ledger["step"] > end:
        raise ValueError("requested horizon precedes resumable state")
    print(json.dumps(dict(start=ledger["step"], end=end, **key)), flush=True)
    for j in range(ledger["step"], end):
        rows = logical_rows(train, buckets, args.seed, j)
        before = time.perf_counter()
        loss, norm, padded = earn_step(model, opt, rows, args.arm, j)
        ledger["seconds"] += time.perf_counter() - before
        root_table((r for _, r in rows), roots)
        ledger["tokens"] += sum(len(r.prompt) for _, r in rows)
        ledger["labels"] += sum(len(r.good) if r.lane == "text" else 1 for _, r in rows)
        ledger["padded"] += padded
        ledger["step"] = j + 1
        ledger["stream"] = identity([ledger["stream"], input_digest(rows)])
        if (j + 1) % 16 == 0:
            logs.append(
                dict(step=j + 1, loss=loss, preclip_norm=norm, tokens=ledger["tokens"], padded=ledger["padded"])
            )
            save_tensor(latest, latest_payload(key, ledger, model, opt, roots, logs))
            print(json.dumps({**logs[-1], "training_s": round(ledger["seconds"], 2)}), flush=True)
    horizon = f"h{end:04d}"
    weight, receipt = folder / f"{horizon}.pt", folder / f"{horizon}.json"
    if receipt.exists():
        if read(receipt)["checkpoint_sha256"] != file_sha(weight):
            raise ValueError("immutable horizon checksum mismatch")
        return
    save_tensor(weight, model.state_dict())
    root_path = folder / f"{horizon}-roots.json"
    atomic_json(root_path, {k: sorted(values) for k, values in roots.items()})
    out = dict(
        **key,
        **ledger,
        params=PARAMS,
        visits=24,
        prefix_steps=PREFIX,
        prefix_tokens=meta["tokens"],
        prefix_padded=meta["padded"],
        prefix_stream=meta["stream"],
        prefix_receipt_sha=file_sha(ck.with_suffix(".json")),
        checkpoint=weight.name,
        checkpoint_sha256=file_sha(weight),
        roots=root_path.name,
        roots_sha256=file_sha(root_path),
        trace=logs,
        optimizer_restart=True,
    )
    if args.arm == "direct":
        out["development"] = anchors(model, args.arm, train, dev)
    atomic_json(receipt, out)
    print(
        json.dumps(
            {"finish": out["step"], "development": out.get("development"), "checksum": out["checkpoint_sha256"]}
        ),
        flush=True,
    )


def select(out, write=True):
    for h in HORIZONS:
        paths = {(a, s): out / "cells" / f"{a}-{s}" / f"h{h:04d}.json" for a in ARMS for s in SEEDS}
        if not all(p.exists() for p in paths.values()):
            return None
        records = {k: read(p) for k, p in paths.items()}
        contracts = {r["recipe"] for r in records.values()}
        if contracts != {recipe()}:
            raise ValueError("source/horizon contract mismatch")
        for s in SEEDS:
            a, b = (records[k, s] for k in ARMS)
            for k in (
                "step",
                "stream",
                "tokens",
                "labels",
                "padded",
                "prefix_stream",
                "prefix_tokens",
                "prefix_padded",
                "params",
                "visits",
                "roots_sha256",
            ):
                if a[k] != b[k]:
                    raise ValueError(f"unmatched {s}: {k}")
            if "development" in b:
                raise ValueError("B evaluation leaked into selection")
        ok = [s for s in SEEDS if records["direct", s]["development"]["competent"]]
        if len(ok) >= 2:
            certificate = dict(
                horizon=h,
                competent_direct_seeds=ok,
                recipe=contracts.pop(),
                receipts={f"{a}-{s}": file_sha(p) for (a, s), p in paths.items()},
            )
            if write:
                dest = out / "selection.json"
                if dest.exists() and read(dest) != certificate:
                    raise ValueError("selection certificate changed")
                atomic_json(dest, certificate)
            return certificate
    return None


def locked_b(args):
    cert = read(args.score_locked)
    if args.arm != ARMS[1] or cert != select(args.out, write=False):
        raise ValueError("B anchor check requires locked direct selection")
    h = cert["horizon"]
    path = cell(args) / f"h{h:04d}.json"
    info = read(path)
    weight = path.with_suffix(".pt")
    if file_sha(weight) != info["checkpoint_sha256"]:
        raise ValueError("frozen weights mismatch")
    train, dev, _, _ = load_inputs(args.pack)
    net = dense(256)
    net.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    result = dict(
        horizon=h,
        seed=args.seed,
        checkpoint_sha256=info["checkpoint_sha256"],
        certificate_sha256=file_sha(args.score_locked),
        development=anchors(net, args.arm, train, dev),
    )
    atomic_json(cell(args) / f"h{h:04d}-locked-dev.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default="direct")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--until", type=int, choices=HORIZONS, default=64)
    parser.add_argument("--pack", type=Path, default=Path("runs/placement/mixed"))
    parser.add_argument("--prefix-dir", type=Path, default=Path("../aleph-next/scale"))
    parser.add_argument("--out", type=Path, default=Path("../aleph-confirm14"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--select", action="store_true")
    mode.add_argument("--score-locked", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.select:
        print(json.dumps(select(args.out)), flush=True)
    elif args.score_locked:
        locked_b(args)
    else:
        fit(args)


if __name__ == "__main__":
    main()
