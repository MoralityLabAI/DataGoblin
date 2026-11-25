"""Collect score-only ALEPH-04 aggregates and per-checkpoint atlas records."""

import argparse
import json
from pathlib import Path
from statistics import fmean

ARMS = ("C", "D", "MLP")
BUDGETS = (1, 2, 4, 8)


def mean_rows(rows, names):
    return {k: fmean(r[k] for r in rows) for k in names}


def aggregate(root):
    train = json.loads((root / "training.json").read_bytes())
    res = json.loads((root / "results.json").read_bytes())
    cert = json.loads((root / "certificate.json").read_bytes())
    results = {}
    for name, source, field in (
        ("trained_programs", res, "pair"),
        ("fresh_programs", res["fresh_program_transfer"], "arm"),
    ):
        panels, measures = source["panels"], source["measured"]
        controls, models = {}, {}
        for b in BUDGETS:
            br = str(b)
            controls[br] = mean_rows(
                [p[br] for p in panels],
                (
                    "side_only",
                    "linear_optimum",
                    "nonlinear_optimum",
                    "native_ldt",
                    "uniform_schedule",
                    "round_robin",
                    "objective_blind",
                    "oracle_utility_per_bit",
                ),
            )
            for label in sorted({row[field] for row in measures}):
                subset = [r for r in measures if r[field] == label and r["bits"] == b]
                key = ("brier", "matched_no_message", "utility_per_bit", "optimal_senders")
                models.setdefault(label, {})[br] = {
                    **mean_rows(subset, key),
                    "seed_brier": [fmean(r["brier"] for r in subset if r["seed"] == s) for s in (137, 163, 191)],
                }
        results[name] = dict(controls=controls, models=models)
    atlas = []
    for key, run in train["runs"].items():
        arm, seed = run["architecture"], run["seed"]
        vals = [r for r in res["measured"] if r["seed"] == seed and r["pair"] == f"{arm}->{arm}"]
        transfer = [r for r in res["fresh_program_transfer"]["measured"] if r["seed"] == seed and r["arm"] == arm]
        local = [
            c
            for rec in res["local_measurements"]
            if rec["seed"] == seed and rec["sender"] == rec["receiver"] == arm
            for c in rec["cells"]
        ]
        entry = dict(
            key=f"aleph04-{arm.lower()}-{seed}",
            checkpoint=f"{key}.pt",
            sha256=run["checkpoint_sha256"],
            architecture=arm,
            params=run["params"],
            passes=run["logical_visits"],
            seed=seed,
            steps=run["steps"],
            metrics={},
            mechanism=dict(
                optimal_local_code_fraction=fmean(c["optimal"] for c in local),
                constant_local_code_fraction=fmean(c["code_mask"] == 0 for c in local),
                sender_supervision="canonical exact optimum for one public site/objective",
                sender_receipt_width=1,
            ),
            failures="",
        )
        for b in BUDGETS:
            e = mean_rows(
                [r for r in vals if r["bits"] == b],
                ("brier", "matched_no_message", "utility_per_bit", "ideal_for_learned_bits"),
            )
            f = mean_rows([r for r in transfer if r["bits"] == b], ("brier", "matched_no_message", "utility_per_bit"))
            entry["metrics"][f"budget_{b}"] = dict(trained_programs=e, fresh_programs=f)
        passed = res["gate_pass_by_seed"][arm][(137, 163, 191).index(seed)]
        entry["failures"] = (
            "Passed this seed's four-bit comparator."
            if passed
            else "Missed the four-bit optimal-linear/Bayes-reader threshold on this seed."
        )
        atlas.append(entry)
    margins = [p["min_shortcut_margin"] for g in cert["rows"] for p in g["profiles"]]
    central = {}
    for b in BUDGETS:
        central[str(b)] = {
            k: fmean(p[k][str(b)] for g in cert["rows"] for p in g["profiles"])
            for k in ("central_midpoint_brier", "linear_midpoint_brier")
        }
    return dict(
        schema="aleph04-public-summary-v1",
        gate_pass_by_seed=res["gate_pass_by_seed"],
        local_shortcut_margin=[min(margins), max(margins)],
        central_local=central,
        **results,
    ), atlas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary, records = aggregate(args.out)
    training = json.loads((args.out / "training.json").read_bytes())
    gym = {
        "key": "aleph04-independent-nonlinear-uplinks-v1",
        "training_sources": training["sources"],
        "training_certificate_sha256": training["certificate"],
        "transfer_certificate_sha256": training["transfer_certificate"],
        "worker_observations": "Eight independent research sites, three uniform private measurements each",
        "future_query": "Conditional proof obligations or native intervened two-rival LDT status",
        "verifier": "Exact eight-world conditional Brier cell counts and repository interval LDT",
        "uplink_bits": list(BUDGETS),
        "public_training_programs": 8,
        "objectives_per_program": 2,
        "transfer_boundary": "Unused program tables and fresh mixture seeds excluded from optimization",
        "gate_pass_by_seed": summary["gate_pass_by_seed"],
    }
    for name, payload in (("public-summary.json", summary), ("atlas-input.json", records), ("gym-input.json", gym)):
        (args.out / name).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
