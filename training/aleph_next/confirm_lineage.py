"""Bind the three 14M A/B pairs to the archived, published 32-update prefix identities."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ("direct", "global_full_damp")
SEEDS = (137, 163, 191)


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def prefix_inventory(prefix, catalog=HERE / "catalog"):
    frozen = read(catalog / "inventory.json")
    roles = catalog / "fit_roles.jsonl"
    if digest(roles) != frozen["digests"]["fit_roles.jsonl"]:
        raise ValueError("canonical fit-role catalog differs from published inventory")
    registered = {}
    with roles.open(encoding="utf-8") as fp:
        for line in fp:
            r = json.loads(line)
            if r["study"] == "ALEPH-06":
                registered[r["key"]] = r
    output = {}
    for seed in SEEDS:
        pair = []
        for arm in ARMS:
            key = f"w256-{arm}-{seed}"
            fit = read(prefix / (key + ".json"))
            expected = registered[key]
            actual = digest(prefix / (key + ".pt"))
            if (
                fit["checkpoint_sha256"] != actual
                or expected["sha256"] != actual
                or (fit["params"], fit["width"], fit["seed"], fit["arm"], fit["steps"], fit["visits"])
                != (14179584, 256, seed, arm, 32, 24)
            ):
                raise ValueError(f"archived prefix mismatch for {key}")
            pair.append(
                {k: fit[k] for k in ("tokens", "padded", "steps", "visits", "corpus", "stream", "contract", "flops")}
            )
            output[key] = dict(sha256=actual, seed=seed, policy=arm, **pair[-1])
        if pair[0] != pair[1]:
            raise ValueError(f"unmatched original prefix for seed {seed}")
    return output


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--prefix", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    result = prefix_inventory(args.prefix)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(result)} published prefix checkpoints")
