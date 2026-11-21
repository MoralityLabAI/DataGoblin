"""Score outer-actor belief predictions and intervention identifiability by cohort."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .hypergame_taste import _file_sha256
from .research_campaign import _write_json, records
from .research_world import DENOMINATOR, Game, message_likelihood


def measure(archive):
    manifest = json.loads((archive / "manifest.json").read_bytes())
    public_path = archive / "public.jsonl"
    if _file_sha256(public_path) != manifest["artifacts"][public_path.name]["sha256"]:
        raise ValueError("public task archive digest differs")
    cells = defaultdict(lambda: defaultdict(float))
    for task in records(public_path):
        if task["split"] not in {"ordinary_test", "ood_test"}:
            continue
        public = task["public"]
        game = Game(public)
        prior = public["prior_weights"]
        positives = message_likelihood({**public, "endorsement": 1})
        denom = DENOMINATOR ** (public["K"] + 1)
        p = sum(weight * chance for weight, chance in zip(prior, positives, strict=True)) / (sum(prior) * denom)
        y = public["endorsement"]
        excluded = public["extension"]["mode"]
        extension = public["experiments"][f"intervene:{excluded}"]["p16"]
        worst_other = max(extension[i] for i in range(3) if i != excluded)
        values = game.values(game.initial)
        terminal = max(value for action, value in values.items() if action not in game.experiments)
        best_probe = max(value for action, value in values.items() if action in game.experiments)
        for axis in ("K", "condition"):
            row = cells[task["split"], axis, str(public[axis])]
            row["n"] += 1
            row["endorsement_accuracy"] += int((p >= 0.5) == y)
            row["endorsement_brier"] += (p - y) ** 2
            row["endorsement_rate"] += y
            row["mean_predicted_endorsement"] += p
            row["aliased_modes_1_2"] += positives[1] == positives[2]
            row["mode_1_2_probe_separation"] += any(spec["p16"][1] != spec["p16"][2]
                                                  for spec in public["experiments"].values())
            row["extension_sensor_total_variation_floor"] += (extension[excluded] - worst_other) / DENOMINATOR
            row["best_probe_value_vs_terminal"] += best_probe - terminal
        game._solve.cache_clear()
    return [
        {"split": split, "axis": axis, "value": value, "roots": int(row["n"]),
         **{name: total / row["n"] for name, total in row.items() if name != "n"}}
        for (split, axis, value), row in sorted(cells.items())
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _write_json(args.out, measure(args.archive))


if __name__ == "__main__":
    main()
