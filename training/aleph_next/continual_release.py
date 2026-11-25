"""Collapse fresh stage checkpoints to a public failure-preserving retention ledger."""

import argparse
import json
from pathlib import Path
from statistics import mean


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def release(out):
    fits = json.loads((out / "training.json").read_bytes())
    result = json.loads((out / "results.json").read_bytes())
    if len(fits) != 42 or set(fits) != set(result["cells"]):
        raise ValueError("incomplete continual tournament")
    atlas, groups = [], {}
    for name, fit in sorted(fits.items()):
        cell = result["cells"][name]
        stat = dict(pass_all=cell["pass_all"], **cell["ldt"])
        for d in ("p3", "p4"):
            stat.update({d + "_" + k: cell[d][k] for k in ("mean_regret", "optimal_fraction", "code_pass", "pass_all")})
            stat[d + "_bracket_brier"] = mean(x["brier"] for x in cell[d]["bracket"])
            stat[d + "_bracket_ideal"] = mean(x["bit_bayes"] for x in cell[d]["bracket"])
        groups.setdefault((fit["stage"], fit["arm"]), []).append(stat)
        trajectory = [
            dict(step=point["step"], **{d: point[d]["optimal_fraction"] if point[d] else None for d in ("p3", "p4")})
            for point in fit["trace"]
        ]
        atlas.append(
            dict(
                key=name,
                checkpoint=name + ".pt",
                sha256=fit["checkpoint_sha256"],
                architecture="C-" + fit["arm"],
                params=fit["params"],
                passes=2,
                seed=fit["seed"],
                steps=fit["updates"],
                parent=fit["parent_sha256"],
                metrics=stat,
                mechanism=dict(
                    trajectory=trajectory,
                    trainable=fit["trainable"],
                    stage=fit["stage"],
                    starting_p3_fraction=fit["initial_p3"]["optimal_fraction"],
                ),
                failures="Both brackets and deployment utility required. Exact side-only is frame-aware.",
            )
        )
    summary = {}
    for (stage, method), cells in groups.items():
        summary.setdefault(stage, {})[method] = {
            "seeds": len(cells),
            "all_passes": sum(x["pass_all"] for x in cells),
            **{k: mean(x[k] for x in cells) for k in cells[0] if k != "pass_all"},
            "fresh_ldt_ideal_side_minus_actual": [x["side_only"] - x["brier"] for x in cells],
        }
    save(out / "atlas-input.json", atlas)
    save(out / "summary.json", dict(stages=summary, split_commitment=result["private_commitment"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("out", type=Path)
    release(cli.parse_args().out)
