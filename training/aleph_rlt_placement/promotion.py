"""Write a small-model development decision and exact follow-up command, without starting jobs."""

import argparse
import json
from pathlib import Path

EQUIVALENT = {"first_last", "penultimate"}


def decide(review):
    cohorts = {}
    for key, data in review["mixed"].items():
        for r in (2, 3):
            if key.startswith(f"w80-r{r}-u192-"):
                if r in cohorts:
                    raise ValueError("multiple incompatible screens at one depth")
                cohorts[r] = data
    if len(cohorts) != 2 or not all(c["complete_registered_screen"] for c in cohorts.values()):
        return {"status": "pending", "reason": "waiting for both registered 80-wide screens"}
    contenders = []
    for r, c in cohorts.items():
        ranked = c["ranked_eligible"]
        if ranked:
            arm = ranked[0]
            effect = c["arms"][arm]["versus_direct"]["paired_delta"]["text_relative"]
            contenders.append((effect, r, arm))
    if not contenders:
        return {"status": "hold", "reason": "no registered candidate qualified", "command": None}
    best_effect = min(t[0] for t in contenders)
    # At most 0.5 percentage point of relative NLL separates tied paired effects.
    effect, r, arm = min((t for t in contenders if t[0] <= best_effect + 0.005), key=lambda t: (t[1], t[0]))
    control = {
        "first_penultimate_last": "first_penultimate_damp",
        "global_full": "global_full_damp",
        "core_edges": "core_edges_damp",
        "pass_rlt": "pass_damp",
    }.get(arm)
    aliases = arm in EQUIVALENT or (r == 2 and arm in {"pass_rlt", "core_edges"})
    arms = ["direct", arm]
    if control and not aliases:
        arms.append(control)
    command = (
        "python training/aleph_rlt_placement/cpu_worker.py --wall 21600 study.py "
        "--pack runs/placement/mixed --out runs/placement/promotion --width 256 "
        f"--r {r} --steps 192 --roots 48 --probe-roots 6 --seeds 137 163 191 --arms " + " ".join(arms)
    )
    return {
        "status": "development_promotion",
        "r": r,
        "arm": arm,
        "arms": arms,
        "paired_relative_text": effect,
        "same_coefficient_as_constant": aliases,
        "command": command,
        "scope": "development replication at larger width; paid arms still require new confirmation",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    report = decide(json.loads(a.review.read_text()))
    body = json.dumps(report, indent=2) + "\n"
    a.out.write_text(body, encoding="utf-8")
    print(body)
