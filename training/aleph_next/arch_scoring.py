"""Partition the registered held scoring over independent model checkpoints."""

import argparse
import hashlib
import json
import secrets
from pathlib import Path

import torch
from architecture_lab import ARMS, MiniAleph, contract, evaluate, evaluation_rows, export_receipts
from gym import atomic_json, file_sha


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inventory(out):
    manifest = out / "training.json"
    record = load(manifest)
    cfg = record["contract"]["recipe"]
    if record["contract"] != contract(cfg):
        raise ValueError("architecture or protocol changed since fitting")
    expected = {f"{a}-{s}" for s in cfg["seeds"] for a in cfg["arms"]}
    if set(record["fits"]) != expected:
        raise ValueError("complete all registered fits before the held draw")
    for key, row in record["fits"].items():
        if file_sha(out / f"{key}.pt") != row["checkpoint_sha256"]:
            raise ValueError(f"invalid frozen checkpoint {key}")
    for seed in cfg["seeds"]:
        rows = [r for r in record["fits"].values() if r["seed"] == seed]
        if any(len({r[k] for r in rows}) != 1 for k in ("stream_sha256", "backbone_init", "positions")):
            raise ValueError("paired fit mismatch")
    return record


def prepare(out):
    inventory(out)
    seed = out / "private.seed"
    if not seed.exists():
        with seed.open("x", encoding="ascii") as fp:
            fp.write(str(secrets.randbits(128)) + "\n")
    source = Path(__file__)
    custody = {
        "training_sha256": file_sha(out / "training.json"),
        "seed_sha256": file_sha(seed),
        "dispatch_sources": {
            p.name: hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
            for p in (source, source.with_name("ARCH_SCORE_PROTOCOL.md"))
        },
    }
    path = out / "score-custody.json"
    if path.exists() and load(path) != custody:
        raise ValueError("scoring contract mismatch")
    atomic_json(path, custody)
    (out / "score-parts").mkdir(exist_ok=True)


def verified(out):
    record = inventory(out)
    custody = load(out / "score-custody.json")
    if custody["training_sha256"] != file_sha(out / "training.json") or custody["seed_sha256"] != file_sha(
        out / "private.seed"
    ):
        raise ValueError("scorer input mismatch")
    for name, sha in custody["dispatch_sources"].items():
        normalized = Path(__file__).with_name(name).read_text(encoding="utf-8").encode()
        if hashlib.sha256(normalized).hexdigest() != sha:
            raise ValueError("dispatch source changed after draw")
    return record, custody


def run(out, seeds):
    record, custody = verified(out)
    cfg = record["contract"]["recipe"]
    if not set(seeds) <= set(cfg["seeds"]):
        raise ValueError("unregistered seed")
    panel = evaluation_rows(int((out / "private.seed").read_text()), cfg["width"], 16)
    for key, fit in record["fits"].items():
        if fit["seed"] not in seeds:
            continue
        path = out / "score-parts" / (key + ".json")
        verified_tag = {"key": key, "checkpoint": fit["checkpoint_sha256"], **custody}
        if path.exists():
            if load(path)["custody"] != verified_tag:
                raise ValueError("existing partial result has different provenance")
            continue
        arm = fit["arm"]
        model = MiniAleph(cfg["width"], arm).eval()
        model.load_state_dict(torch.load(out / f"{key}.pt", map_location="cpu", weights_only=True))
        native = evaluate(model, panel, arm, True)
        swaps = ("rlt3", "damp3", "direct3") if ARMS[arm] == 3 else ("direct2", "source2", "local2")
        cross = {other: evaluate(model, panel, other, False) for other in swaps if other != arm}
        atomic_json(path, {"custody": verified_tag, "examples": len(panel[0]), "native": native, "crossovers": cross})
        print("scored", key, flush=True)


def collect(out):
    record, custody = verified(out)
    results = {
        "training_sha256": custody["training_sha256"],
        "seed_sha256": custody["seed_sha256"],
        "native": {},
        "crossovers": {},
    }
    for key, row in record["fits"].items():
        part = load(out / "score-parts" / (key + ".json"))
        if part["custody"] != {"key": key, "checkpoint": row["checkpoint_sha256"], **custody}:
            raise ValueError("unbound partial result")
        if results.setdefault("examples", part["examples"]) != part["examples"]:
            raise ValueError("partial panels have different sizes")
        results["native"][key] = part["native"]
        results["crossovers"][key] = part["crossovers"]
    target = out / "scores.json"
    if target.exists() and results != load(target):
        raise ValueError("refusing to replace a different evaluation")
    atomic_json(target, results)
    export_receipts(record, results, out)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("prepare", "run", "collect"))
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--seeds", nargs="+", type=int)
    args = cli.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.phase == "prepare":
        prepare(args.out)
    elif args.phase == "run":
        if not args.seeds:
            cli.error("select a disjoint set of seeds for this worker")
        run(args.out, args.seeds)
    else:
        collect(args.out)
