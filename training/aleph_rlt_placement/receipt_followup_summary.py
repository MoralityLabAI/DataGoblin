"""Summarize the frozen receipt follow-up at matched seed and structural family level."""

import argparse
import json
from pathlib import Path
from statistics import mean

from receipt_summary import average, families

BRACKET = ("0.07", "0.08", "0.072", "0.079")
ARMS = ("C", "D", "MLP")


def pareval(rows):
    near = [rows[p] for p in BRACKET]
    return {
        "passes": all(r["optimal"] and r["brier"] < r["side_only"] for r in near),
        "codes": [r["code"] for r in near],
        "optimal": [r["optimal"] for r in near],
        "brier": [r["brier"] for r in near],
        "mean_code_regret": mean(r["code_bayes"] - r["oracle"] for r in near),
    }


def summarize(result):
    paired = {}
    for key, run in result["scores"].items():
        row = {name: {"parity": pareval(branch["parity"])} for name, branch in run.items()}
        for name in ("primary", "original", "majority"):
            panels = run[name]
            for domain in ("held", "familiar"):
                row[name][domain] = families(panels[domain])
            examples = list(panels["held"].values())
            row[name]["constant_frames"] = sum(r["code"] == 0 for r in examples)
            row[name]["n_frames"] = len(examples)
            if name == "majority":
                row[name]["clamped_brier"] = mean(r["clamped_majority_brier"] for r in examples)
                row[name]["bit_accuracy"] = mean(r["majority_bit_accuracy"] for r in examples)
        paired[key] = row
    arms = {}
    for arm in ARMS:
        runs = {k.split("-")[1]: v for k, v in paired.items() if k.startswith(arm + "-")}
        main = [v["primary"]["held"]["overall"] for v in runs.values()]
        fs = next(iter(runs.values()))["primary"]["held"]["families"]
        family_scores = {f: average([v["primary"]["held"]["families"][f] for v in runs.values()]) for f in fs}
        gates = {
            "parity": sum(v["coordinated"]["parity"]["passes"] for v in runs.values()),
            "linear": sum(r["brier"] < r["best_linear"] - 1e-10 for r in main),
            "family_floor": all(r["brier"] <= r["side_only"] for r in family_scores.values()),
        }
        for other in ARMS:
            if other != arm:
                gates[f"beats_{other}"] = sum(
                    v["primary"]["held"]["overall"]["brier"]
                    < paired[f"{other}-{seed}"]["primary"]["held"]["overall"]["brier"]
                    for seed, v in runs.items()
                )
        eligible = (
            arm != "MLP"
            and gates["family_floor"]
            and all(
                gates[name] == len(runs)
                for name in ("parity", "linear", "beats_MLP", "beats_D" if arm == "C" else "beats_C")
            )
        )
        arms[arm] = {
            "qualifies": eligible,
            "gates": gates,
            "families": family_scores,
            "baseline_parity_passes": sum(v["baseline"]["parity"]["passes"] for v in runs.values()),
            "retained_parity_passes": sum(v["primary"]["parity"]["passes"] for v in runs.values()),
        }
        for name in ("primary", "original", "majority"):
            g = [v[name] for v in runs.values()]
            arms[arm][name] = {
                "held": average([x["held"]["overall"] for x in g]),
                "familiar": average([x["familiar"]["overall"] for x in g]),
                "constant_frames": sum(x["constant_frames"] for x in g),
                "frames": sum(x["n_frames"] for x in g),
            }
            if name == "majority":
                arms[arm][name].update(
                    clamped_brier=mean(x["clamped_brier"] for x in g), bit_accuracy=mean(x["bit_accuracy"] for x in g)
                )
    ex = next(iter(result["scores"].values()))["primary"]["held"]
    return {
        "arms": arms,
        "paired": paired,
        "n_held_frames": len(ex),
        "linear_optimal_frames": sum(abs(x["best_linear"] - x["oracle"]) < 1e-10 for x in ex.values()),
        "solver_max_regret": max(x["regret"] for x in result["affine_solver"].values()),
    }


def scan_checkpoints(directory, result):
    """Describe the learned transition on a 0.00025 grid, after the frozen evaluation."""
    import numpy as np
    import torch
    from receipt_gym import digest, parity
    from receipt_models import ReceiptModel

    torch.set_num_threads(1)
    manifest = directory / "training.json"
    if digest(manifest) != result["manifest_sha256"]:
        raise ValueError("result and training manifest mismatch")
    runs = json.loads(manifest.read_text())["runs"]
    priors = np.linspace(0.05, 0.10, 201)
    xs = parity(0)[0]
    inp = np.broadcast_to(xs, (len(priors), *xs.shape)).copy()
    inp[:, :, 3] = priors[:, None]
    batch = torch.tensor(inp, dtype=torch.float32).reshape(-1, 12)
    scans = {}
    with torch.no_grad():
        for key, run in runs.items():
            cell = run["coordinated"]
            path = directory / f"{key}-coordinated.pt"
            if digest(path) != cell["sha256"]:
                raise ValueError("transition checkpoint mismatch")
            model = ReceiptModel(key.split("-")[0]).eval()
            model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
            bits = (model.encode(batch).reshape(-1, 8) >= 0).int().numpy()
            codes = (bits ^ bits[:, :1]) @ (1 << np.arange(8))
            starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
            ends = np.r_[starts[1:] - 1, len(codes) - 1]
            scans[key] = [
                [round(float(priors[i]), 5), round(float(priors[j]), 5), int(codes[i])]
                for i, j in zip(starts, ends, strict=True)
            ]
    return scans


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("results", type=Path)
    cli.add_argument("--out", type=Path)
    cli.add_argument("--checkpoints", type=Path, help="also scan the frozen parity checkpoints")
    args = cli.parse_args()
    result = json.loads(args.results.read_text())
    summary = summarize(result)
    if args.checkpoints:
        summary["transition_grid"] = scan_checkpoints(args.checkpoints, result)
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name, a in summary["arms"].items():
        print(name, a["gates"], a["primary"]["held"]["brier"], a["qualifies"])
