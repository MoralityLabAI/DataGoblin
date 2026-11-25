"""Descriptive train-count controls on the original native development lanes."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from confirm_text_check import raw_ngram_nll

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aleph_rlt_placement"))
from corpus import load_corpus


def lookup(train, dev, indices):
    def key(row):
        return tuple(row.prompt[i] for i in indices)

    table = defaultdict(Counter)
    for row in train:
        table[key(row)][row.good[0]] += 1
    fitted = {k: min(counts, key=lambda y: (-counts[y], y)) for k, counts in table.items()}
    covered = sum(key(r) in fitted for r in dev)
    hits = sum(fitted.get(key(r)) in r.good for r in dev)
    return {"training_keys": len(fitted), "held_keys_seen": covered, "hits": hits, "total": len(dev)}


def controls(pack):
    train, dev, source = load_corpus(pack)
    raw_train = [bytes((*r.prompt, r.good[-1])) for r in train["text"]]
    held = [{"doc": r.group, "hex": bytes((*r.prompt, r.good[-1])).hex()} for r in dev["text"]]
    pooled = [{**r, "doc": "pooled"} for r in held]
    keys = {
        "constant": (),
        "operand": (-2,),
        "op_operand": (-3, -2),
        "previous_operand": (-4, -2),
        "previous_op_operand": (-4, -3, -2),
    }
    return {
        "scope": "existing public native development",
        "source": source,
        "text_pooled": raw_ngram_nll(raw_train, pooled),
        "text_by_source": raw_ngram_nll(raw_train, held),
        "repair_lookup": {name: lookup(train["repair"], dev["repair"], idx) for name, idx in keys.items()},
        "repair_operand_copy": {
            "hits": sum(r.prompt[-2] in r.good for r in dev["repair"]),
            "total": len(dev["repair"]),
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", type=Path, default=Path("runs/placement/mixed"))
    p.add_argument("--out", type=Path, default=Path("../aleph-confirm14/native-controls.json"))
    args = p.parse_args()
    result = controls(args.pack)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
