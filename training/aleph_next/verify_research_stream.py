"""Independently solve every published training exercise from its prompt and verify its packing."""

import argparse
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path


def solve(prompt):
    if prompt.startswith("Uniform worlds"):
        facts = json.loads(prompt.split("fraction. ", 1)[1].split(" Forecast ", 1)[0])
        n = positive = 0
        for world, upload in enumerate(facts["policy"]):
            side = (world >> facts["side_axis"]) & 1
            if side != facts["side"] or upload != facts["receipt"]:
                continue
            a, b = facts["rivals"][:3], facts["rivals"][3:]
            yes_a = any(v and ((world >> axis) & 1) for axis, v in enumerate(a))
            yes_b = any(v and ((world >> axis) & 1) for axis, v in enumerate(b))
            positive += yes_a and yes_b
            n += 1
        return str(Fraction(positive, n))
    if not prompt.startswith("Choose a reader"):
        raise ValueError("unknown public exercise")
    charge = int(re.search(r"Reading the bit costs (\d+) millionths", prompt)[1])
    costs = {}
    for reader, side, valid in re.findall(r"([\w-]+)=(\d+),(\d+)", prompt):
        costs[reader + "/side"] = int(side)
        costs[reader + "/valid"] = int(valid) + charge
    if len(costs) != 6:
        raise ValueError("expected three reader alternatives")
    return min(costs, key=lambda k: (costs[k], k))


def verify(directory):
    rows = 0
    with (directory / "audit-receipts.jsonl").open("rb") as src, (directory / "train.jsonl").open("rb") as dest:
        for claim, packed in zip(src, dest, strict=True):
            row, tokens = json.loads(claim), json.loads(packed)
            if solve(row["prompt"]) != row["answer"] or row["split"] != "train":
                raise ValueError("invalid verified training answer or split")
            digest = hashlib.sha256(claim.rstrip(b"\r\n")).hexdigest()
            data = bytes(i - 1 for i in tokens["token_ids"])
            start = len(row["prompt"].encode())
            target = (row["answer"] + "\n").encode()
            if (
                data != row["prompt"].encode() + target
                or tokens["receipt_sha256"] != digest
                or tokens["target_mask"] != [False] * start + [True] * len(target)
            ):
                raise ValueError("packing changed condition, answer, receipt or supervised span")
            rows += 1
    if rows != json.loads((directory / "inventory.json").read_bytes())["rows"]:
        raise ValueError("incomplete corpus")
    return rows


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path)
    print("Independently verified training rows", verify(cli.parse_args().directory))
