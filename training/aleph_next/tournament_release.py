"""Create the public gate ledger and receipt digest inventory from complete frozen cohorts."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(root, name):
    return json.loads((root / name).read_bytes())


def public(root):
    a = read(root, "arch/summary.json")
    d = read(root, "decoder/results.json")["comparisons"]
    n = read(root, "nonlinear/public-summary.json")
    c = read(root, "continual/summary.json")
    s = read(root, "scale/summary.json")
    mixed = read(root, "mixed192/summary.json")
    if s["status"] != "complete" or not mixed or len(a["cells"]) != 33 or len(d) != 18:
        raise ValueError("incomplete tournament")
    decision = {
        "schema": "datagoblin.aleph-next.gates.v1",
        "paid_run": {
            "status": "HOLD",
            "credits_spent": 0,
            "config": "strict_1b_pair.json",
            "reason": "Missing author-independent production-quality confirmation and priced hardware replay.",
        },
        "ALEPH-01-02": a["gates"],
        "ALEPH-03": {
            key: {
                k: item[k]
                for k in (
                    "passed",
                    "registered_primary",
                    "gates",
                    "learned_value",
                    "oracle_value",
                    "fraction_recovered",
                    "paired_seed_panel_bootstrap_95",
                )
            }
            for key, item in d.items()
        },
        "ALEPH-04": {
            "all_three_required": True,
            "four_bit_by_seed": n["gate_pass_by_seed"],
            "passed": {key: all(x) for key, x in n["gate_pass_by_seed"].items()},
        },
        "ALEPH-05": {
            k: {t: v[t] for t in ("all_passes", "seeds", "p3_code_pass", "p4_code_pass")}
            for k, v in c["stages"]["p4"].items()
        },
        "ALEPH-06": s["rungs"],
        "mixed_192": {"status": "completed public development replay", "new_fits": 6, "archived_controls": 6},
        "population_gym": read(root, "research-stream-v2/inventory.json"),
    }
    inputs = [
        "arch/training.json",
        "arch/scores.json",
        "arch/summary.json",
        "arch/spectra.json",
        "arch/edit-depth.json",
        "decoder/training.json",
        "decoder/fit-scores.json",
        "decoder/results.json",
        "decoder/pooled-oracle.json",
        "nonlinear/training.json",
        "nonlinear/certificate.json",
        "nonlinear/transfer-certificate.json",
        "nonlinear/results.json",
        "nonlinear/public-summary.json",
        "continual/training.json",
        "continual/results.json",
        "continual/summary.json",
        "scale/summary.json",
        "scale/gates.json",
        "mixed192/summary.json",
        "research-stream-v2/inventory.json",
        "research-stream-v2/gradient-probe.json",
        "atlas/inventory.json",
    ]
    inventory = dict(
        schema="datagoblin.aleph-next.receipt-manifest.v1",
        model_atlas=read(root, "atlas/inventory.json"),
        receipts={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in inputs},
    )
    return decision, inventory


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in zip(("gates.json", "receipt-manifest.json"), public(args.runs), strict=True):
        (args.out / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
