"""Replay a lower-work Bayes tie-break on the opened evidence-check development roots."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .evidence_budget import ROUTES, bayes_check, check_run, make_episode
from .evidence_budget_pilot import EVAL_ROOTS_PER_FAMILY, gate_rules, verified_proofs
from .ram_hillclimb import canonical, digest, install_memory_ceiling, peak_rss_mib


def run(checkout: Path, prior: Path, out: Path) -> dict:
    limits = install_memory_ceiling(8192)
    rules = gate_rules(checkout)
    report = json.loads((prior / "budget_report.json").read_bytes())
    archived = (prior / "check_receipts.jsonl").read_bytes()
    if report["receipts_sha256"] != digest(archived) or report["native_rule_sha256"] != digest(rules.encode()):
        raise ValueError("Archived evidence and native source must match the original pilot")
    baselines = {}
    for line in archived.splitlines():
        row = json.loads(line)
        if row["arm"] == "bayes_public":
            baselines[row["source_root_sha256"]] = row
    receipts = []
    counts: dict[str, Counter] = defaultdict(Counter)
    for family in ROUTES:
        for seed in range(EVAL_ROOTS_PER_FAMILY):
            episode = make_episode(family, "fresh_pilot", seed)
            baseline = baselines.pop(episode.catalog.source_root)
            original = check_run(episode, bayes_check(episode.catalog))
            if canonical(original) != canonical({key: baseline[key] for key in original}):
                raise ValueError("Original Bayes outcomes do not reproduce the archived receipts")
            row = check_run(episode, bayes_check(episode.catalog, minimize_work=True))
            proofs, native_hash = verified_proofs(rules, episode, row["final_state"])
            if bool(proofs) != row["completed"]:
                raise ValueError("Work tie-break disagrees with native receipt verification")
            counts[family].update(trials=1, earlier_checks=baseline["checks"], current_checks=row["checks"],
                                  earlier_completed=int(baseline["completed"]), current_completed=int(row["completed"]))
            receipts.append({**row, "family": family, "native_input_sha256": native_hash,
                             "native_accepted_proofs": sorted(proofs)})
    if baselines:
        raise ValueError("The cost pilot omitted archived Bayes roots")
    payload = "".join(canonical(row) + "\n" for row in receipts).encode()
    out.mkdir(parents=True, exist_ok=False)
    (out / "cost_receipts.jsonl").write_bytes(payload)
    result = {"families": counts, "baseline_receipts_sha256": digest(archived), "receipts_sha256": digest(payload),
              "peak_mib": peak_rss_mib(), "memory": limits, "evidence_class": "opened_root_diagnostic"}
    (out / "cost_report.json").write_text(canonical(result) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("storyworld", "prior", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    print(canonical(run(args.storyworld, args.prior, args.out)))


if __name__ == "__main__":
    main()
