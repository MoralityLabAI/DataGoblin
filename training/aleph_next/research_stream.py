"""Export verified training-only posterior repairs from heterogeneous frozen research partners."""

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "aleph_rlt_placement"))
from decoder_lab import (  # noqa: E402
    build_sender,
    cells,
    ckpt_path,
    digest,
    model_id,
    predictions,
    uploads,
    world_labels,
)
from packing import INDEX_SCHEMA, PackedStream  # noqa: E402
from receipt_relations import make_relations, verified_side_sha256  # noqa: E402
from research_options import options  # noqa: E402

PUBLIC_TRAIN_SEED = 2026112503


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def examples(v2, decoder):
    fits = json.loads((decoder / "training.json").read_bytes())
    parents = json.loads((v2 / "training.json").read_bytes())
    games = make_relations("train", PUBLIC_TRAIN_SEED)
    frames = list(games.values())
    for frame in frames:
        verified_side_sha256(frame)
        if not np.array_equal(world_labels(frame[1]), frame[2]):
            raise ValueError("independent query verifier disagrees")
    for seed in (137, 163, 191):
        for sender, receiver in (("C", "D"), ("D", "MLP"), ("MLP", "structured")):
            name = f"{sender}-{seed}"
            actor, stamp = build_sender(v2, sender, seed, parents)
            if stamp != fits["senders"][name]:
                raise ValueError("sender identity mismatch")
            label = model_id(name, receiver, "valid")
            path = ckpt_path(decoder, label, receiver)
            if digest(path) != fits["fits"][label]["sha256"]:
                raise ValueError("receiver identity mismatch")
            codes = uploads(actor, frames)
            x = np.array([cells(frame, code, "valid")[0] for frame, code in zip(frames, codes, strict=True)])
            probs = predictions(receiver, seed, path, x)
            for idx, (key, frame) in enumerate(games.items()):
                _, query, target, side, _ = frame
                # One uniformly rotating late query per game. The public policy table is
                # supplied only to the audit student after the receiver has committed.
                q = (idx + seed) % len(query)
                for s in range(2):
                    for m in range(2):
                        mask = (codes[idx] == m) & (side[:, q] == s)
                        n = int(mask.sum())
                        if not n:
                            continue
                        yes = int(target[mask, q].sum())
                        exact = str(Fraction(yes, n))
                        proposal = float(probs[idx, (m * 2 + s) * len(query) + q])
                        facts = dict(
                            policy=codes[idx].tolist(),
                            rivals=query[q, :6].astype(int).tolist(),
                            side_axis=int(query[q, 6:9].argmax()),
                            side=s,
                            receipt=m,
                        )
                        prompt = (
                            "Uniform worlds 0..7; bit j is (world >> j) & 1. Two rival masks each need "
                            "at least one set bit. Condition on the side-axis value and the posted "
                            "policy's receipt. Audit the receiver forecast and return the exact "
                            "probability as a reduced fraction. "
                            + canonical(facts).decode()
                            + f" Forecast {proposal:.4f}. Answer "
                        )
                        record = dict(
                            split="train",
                            family="ldt-train-" + key.split("/")[0],
                            prompt=prompt,
                            answer=exact,
                            sender=stamp["sha256"],
                            receiver=fits["fits"][label]["sha256"],
                            positive=yes,
                            eligible=n,
                            receiver_absolute_error=abs(proposal - yes / n),
                            verifier="eight-world enumeration",
                        )
                        yield record
            yield from options(fits, decoder, name, seed, frames, games, codes, stamp["sha256"])


def pack(records, out):
    out.mkdir(parents=True, exist_ok=False)
    tok = out / "tokenizer.json"
    tok.write_bytes(canonical({"pilot_encoding": "UTF-8 byte + 1", "pad": 0}))
    shard, checked = out / "train.jsonl", out / "audit-receipts.jsonl"
    view_index = []
    families = set()
    counts = Counter()
    errors = rejected = rows = targets = total = 0
    seen = {}
    with shard.open("wb") as stream, checked.open("wb") as receipts:
        for r in records:
            if r["split"] != "train" or not r["family"].startswith("ldt-train-"):
                raise ValueError("unsupported training provenance")
            root = sha(r["prompt"].encode())
            if root in seen:
                if seen[root] != r["answer"]:
                    raise ValueError("conflicting verified answers for one prompt")
                continue
            seen[root] = r["answer"]
            payload = canonical(r)
            receipt = sha(payload)
            receipts.write(payload + b"\n")
            left, right = r["prompt"].encode(), (r["answer"] + "\n").encode()
            ids = [1 + b for b in left + right]
            mask = [False] * len(left) + [True] * len(right)
            item = dict(
                root=root,
                family=r["family"],
                split="train",
                presentation=1,
                lane=r.get("objective", "researcher-posterior-repair"),
                objective="checked_response",
                token_ids=ids,
                target_mask=mask,
                response_start=len(left),
                receipt_sha256=receipt,
            )
            row = canonical(item)
            stream.write(row + b"\n")
            view_index.append((root, r["family"], sha(row), receipt))
            families.add(r["family"])
            rows += 1
            total += len(ids)
            targets += len(right)
            counts[item["lane"]] += 1
            errors += int(r.get("receiver_absolute_error", 0) > 0.1)
            rejected += int(r.get("proposed_excess_microloss", 0) > 0)
    if not rows:
        raise ValueError("empty training corpus")
    db = out / "registry.sqlite"
    shard_sha = digest(shard)
    with sqlite3.connect(db) as registry:
        registry.executescript(INDEX_SCHEMA)
        registry.executemany("INSERT INTO families VALUES (?, 'train')", [(f,) for f in sorted(families)])
        for i, (root, family, row_sha, receipt) in enumerate(view_index):
            registry.execute("INSERT INTO roots VALUES (?, ?)", (root, family))
            registry.execute("INSERT INTO views VALUES (?, 1, ?, ?, ?, ?)", (root, shard_sha, i, row_sha, receipt))

    def member(p):
        return {"path": p.name, "sha256": digest(p)}

    manifest = out / "manifest.json"
    manifest.write_bytes(
        canonical(
            dict(
                schema="aleph.packed-stream.v1",
                context=2048,
                vocab_size=257,
                pad_id=0,
                tokenizer=member(tok),
                registry=member(db),
                shards=[{**member(shard), "records": rows, "split": "train"}],
            )
        )
    )
    with PackedStream(manifest, digest(manifest), "train") as corpus:
        while corpus.next_pack(total) is not None:
            pass
        if corpus.state["processed"] != total or corpus.state["started_views"] != rows:
            raise ValueError("incomplete corpus round trip")
        effective = corpus.state["targets"]
    result = dict(
        rows=rows,
        lane_counts=dict(counts),
        rejected_costed_proposals=rejected,
        families=len(families),
        hypotheses_over_10pp_wrong=errors,
        positions=total,
        response_positions=targets,
        packed_targets=effective,
        manifest_sha256=digest(manifest),
        receipts_sha256=digest(checked),
        public_training_generator_seed=PUBLIC_TRAIN_SEED,
    )
    (out / "inventory.json").write_bytes(canonical(result) + b"\n")
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("v2", "decoder", "out"):
        cli.add_argument("--" + arg, type=Path, required=True)
    args = cli.parse_args()
    torch.set_num_threads(1)
    print(pack(examples(args.v2, args.decoder), args.out))
