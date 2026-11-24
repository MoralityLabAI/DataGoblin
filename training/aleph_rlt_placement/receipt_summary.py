"""Aggregate one-bit evaluations with equal weight per family and per paired seed."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def average(rows):
    return {
        field: mean(r[field] for r in rows)
        for field in ("brier", "code_bayes", "side_only", "majority", "best_linear", "oracle", "corrected", "spoiled")
    }


def families(rows):
    groups = defaultdict(list)
    for name, row in rows.items():
        groups[name.rsplit("/", 1)[0]].append(row)
    grouped = {family: average(items) for family, items in groups.items()}
    return {"families": grouped, "overall": average(list(grouped.values()))}


def summarize(result):
    pairs = {}
    for key, row in result["scores"].items():
        parity = row["parity_to_parity"]
        near = [parity[str(v)] for v in (0.07, 0.08)]
        pairs[key] = dict(
            bracket=all(r["optimal"] for r in near),
            usable_bit=all(r["brier"] < r["side_only"] for r in near),
            parity={k: [r["code"], r["brier"], r["code_bayes"], r["optimal"]] for k, r in parity.items()},
            zero_shot=families(row["parity_to_ldt"]),
            adapted=families(row["ldt_to_ldt"]),
        )
    arms = {}
    for arm in ("C", "D", "MLP"):
        by_seed = {k.split("-")[1]: v for k, v in pairs.items() if k.startswith(arm + "-")}
        adapted = [v["adapted"]["overall"] for v in by_seed.values()]
        family_names = next(iter(by_seed.values()))["adapted"]["families"]
        fam = {f: average([v["adapted"]["families"][f] for v in by_seed.values()]) for f in family_names}
        gates = dict(
            bracket=sum(v["bracket"] for v in by_seed.values()),
            usable_bit=sum(v["usable_bit"] for v in by_seed.values()),
            beats_linear=sum(r["brier"] < r["best_linear"] for r in adapted),
            families_with_positive_gain=sum(r["brier"] < r["side_only"] for r in fam.values()),
            family_count=len(fam),
        )
        if arm in ("C", "D"):
            for other in ("MLP", "D" if arm == "C" else "C"):
                gates[f"beats_{other}"] = sum(
                    v["adapted"]["overall"]["brier"] < pairs[f"{other}-{s}"]["adapted"]["overall"]["brier"]
                    for s, v in by_seed.items()
                )
        qualifies = (
            arm in ("C", "D")
            and all(
                gates[k] == len(by_seed)
                for k in ("bracket", "usable_bit", "beats_linear", "beats_MLP", f"beats_{'D' if arm == 'C' else 'C'}")
            )
            and all(r["brier"] <= r["side_only"] for r in fam.values())
        )
        arms[arm] = dict(
            gates=gates,
            qualifies=qualifies,
            adapted=average(adapted),
            families=fam,
            zero_shot=average([v["zero_shot"]["overall"] for v in by_seed.values()]),
        )
    return {"arms": arms, "paired": pairs}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("results", type=Path)
    cli.add_argument("--out", type=Path)
    args = cli.parse_args()
    result = summarize(json.loads(args.results.read_text(encoding="utf-8")))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["arms"], indent=2))
