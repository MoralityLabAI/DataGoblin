"""Verify the selected public-development horizon and bind six endpoints for sealed scoring."""

import argparse
import json
from pathlib import Path

from confirm_lineage import ARMS, HERE, SEEDS, digest, prefix_inventory, read


def checked_development(value):
    expected = value["text_nll"] < value["text_unigram_nll"] and value["repair"] > max(
        value["repair_cue"], value["repair_chance"]
    )
    if value["competent"] != expected or value["text_positions"] <= 0 or value["repair_rows"] <= 0:
        raise ValueError("inconsistent development gate")
    return bool(expected)


def manifest(directory, prefix, path):
    selected_path = directory / "selection.json"
    selected = read(selected_path)
    selection_digest = digest(selected_path)
    parents = prefix_inventory(prefix)
    fits = []
    development = {arm: {} for arm in ARMS}
    paired = {}
    h = selected["horizon"]
    horizons = (64, 128, 256, 384, 512)
    if h not in horizons:
        raise ValueError("unregistered confirmation horizon")
    for prior in horizons[: horizons.index(h)]:
        earlier = [read(directory / "cells" / f"direct-{s}" / f"h{prior:04d}.json") for s in SEEDS]
        if sum(checked_development(r["development"]) for r in earlier) >= 2:
            raise ValueError("an earlier reference horizon already qualified")
    for seed in SEEDS:
        for arm in ARMS:
            key = f"{arm}-{seed}"
            folder = directory / "cells" / key
            base = f"h{h:04d}"
            receipt = folder / (base + ".json")
            stage = read(receipt)
            if digest(receipt) != selected["receipts"][key] or stage["recipe"] != selected["recipe"]:
                raise ValueError("receipt differs from selected stage")
            if (stage["arm"], stage["seed"], stage["step"], stage["params"], stage["visits"]) != (
                arm,
                seed,
                h,
                14179584,
                24,
            ):
                raise ValueError("selection geometry mismatch")
            if stage["prefix_sha"] != parents["w256-" + key]["sha256"]:
                raise ValueError("continuation weight lineage mismatch")
            ck = folder / (base + ".pt")
            root = folder / (base + "-roots.json")
            if digest(ck) != stage["checkpoint_sha256"] or digest(root) != stage["roots_sha256"]:
                raise ValueError("checkpoint or training-root identity mismatch")
            footprint = {
                k: stage[k]
                for k in (
                    "tokens",
                    "padded",
                    "labels",
                    "stream",
                    "roots_sha256",
                    "prefix_stream",
                    "prefix_tokens",
                    "prefix_padded",
                )
            }
            if seed in paired and footprint != paired[seed]:
                raise ValueError("unmatched position, supervision or root stream")
            paired[seed] = footprint
            if arm == "direct":
                dev = stage["development"]
            else:
                if "development" in stage:
                    raise ValueError("candidate was scored before horizon selection")
                locked = read(folder / (base + "-locked-dev.json"))
                if (locked["horizon"], locked["seed"], locked["certificate_sha256"], locked["checkpoint_sha256"]) != (
                    h,
                    seed,
                    selection_digest,
                    stage["checkpoint_sha256"],
                ):
                    raise ValueError("candidate development result is stale")
                dev = locked["development"]
            checked_development(dev)
            development[arm][str(seed)] = dev
            fits.append(
                dict(
                    arm=arm,
                    seed=seed,
                    step=h,
                    checkpoint=str(ck.resolve()),
                    sha256=stage["checkpoint_sha256"],
                    parent_sha256=stage["prefix_sha"],
                    receipt_sha256=digest(receipt),
                )
            )
    qualified = {a: [s for s in SEEDS if development[a][str(s)]["competent"]] for a in ARMS}
    if qualified["direct"] != selected["competent_direct_seeds"] or len(qualified["direct"]) < 2:
        raise ValueError("direct baseline has not earned confirmation")
    contract_files = ("CONFIRM14_PROTOCOL.md", "ANCHOR14_PROTOCOL.md", "CONFIRM_REASON_PROTOCOL.md", "confirm_lock.py")
    result = dict(
        fits=fits,
        policy="ALEPH-14M fresh A/B v1",
        source_sha256={f: digest(HERE / f) for f in contract_files},
        reference_competent=True,
        candidate_competent=len(qualified[ARMS[1]]) >= 2,
        development_competence=development,
        qualifying_seeds=qualified,
        streams=paired,
        selected_step=h,
        selection_sha256=selection_digest,
    )
    data = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("an incompatible fit roster is already frozen")
    else:
        path.write_bytes(data)
    return digest(path)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "prefix", "out"):
        cli.add_argument("--" + key, type=Path, required=True)
    args = cli.parse_args()
    print(manifest(args.run, args.prefix, args.out))
