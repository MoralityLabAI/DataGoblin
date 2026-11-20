"""Aggregate checked episode, stopping, and intervention comparisons."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import dump_structured, load_structured

from .corpus import records


def pair_key(episode):
    cfg = episode["config"]
    return (episode["task_digest"], episode["backend"]["model_digest"], cfg["mode"],
            cfg["planned_depth"], cfg["readout_site"])


def episode_metrics(episode, elapsed):
    outcome, steps = episode["outcome"], episode["steps"]
    first = outcome["first_verified_visit"]
    passed_seen = False
    regression = False
    changed = 0
    previous = None
    dynamics = Counter()
    previous_scores = None
    for step in steps:
        passed = step["verification"]["passed"]
        regression |= passed_seen and not passed
        passed_seen |= passed
        candidate = step["verification"]["candidate_digest"]
        changed += int(previous is not None and previous != candidate)
        previous = candidate
        if "state_summaries" in step:
            name = f"visit_{step['visit'] + 1}"
            state = step["state_summaries"]
            dynamics[f"{name}_observations"] += 1
            dynamics[f"{name}_answer_rms_sum"] += state["carried_answer"]["rms"]
            dynamics[f"{name}_update_rms_sum"] += state["feedback_update"]["rms"]
            scores = step["candidate_scores"]
            if previous_scores is not None:
                dynamics[f"{name}_score_change_l1_sum"] += sum(
                    abs(a - b) for a, b in zip(scores, previous_scores, strict=True)
                )
            previous_scores = scores
    return Counter({
        "episodes": 1, "verified": int(outcome["passed"]), "ever_verified": int(first is not None),
        "regressed_after_success": int(regression),
        "final_regressions": int(first is not None and not outcome["passed"]),
        "censored": int(outcome["censored"]), "empty_episodes": int(not steps),
        "invalid_format": sum(
            s["verification"]["code"] in {"candidate_shape", "action_budget_or_shape"} for s in steps
        ),
        "verifier_calls": outcome["verifier_calls"], "adapter_calls": outcome["adapter_calls"],
        "readout_candidate_changes": changed, "elapsed_episode_seconds": elapsed,
        **dynamics,
        **({f"first_verified_visit_{first + 1}": 1} if first is not None else {}),
    })


def audit(root):
    runs, terminal, comparisons = {}, {}, defaultdict(Counter)
    for name in ("terminal", "early", "pulse"):
        directory = root / name
        replay = load_structured(root / f"{name}-replay.json")
        manifest = load_structured(directory / "manifest.json")
        trace = directory / "rollouts.jsonl"
        if (replay["passed"] is not True or replay["rollouts_sha256"] != sha256_file(trace)
                or replay["run_manifest_digest"] != manifest["manifest_digest"]):
            raise ValueError("campaign requires completed bound replay")
        measurement = directory / "measurements.jsonl"
        if sha256_file(measurement) != manifest["files"][measurement.name]:
            raise ValueError("campaign measurements changed")
        totals, groups, codes = Counter(), defaultdict(Counter), Counter()
        for episode, wall in zip(records(trace), records(measurement), strict=True):
            if episode["episode_id"] != wall["episode_id"]:
                raise ValueError("episode timing identity mismatch")
            metrics = episode_metrics(episode, wall["elapsed_seconds"])
            totals.update(metrics)
            codes.update(s["verification"]["code"] for s in episode["steps"])
            backend, cfg = episode["backend"], episode["config"]
            fit = backend.get("training", {}).get("training_mode", "none")
            group = (episode["split"], episode["family"], backend["kind"], fit, cfg["mode"], cfg["planned_depth"])
            groups[group].update(metrics)
            if backend["kind"] != "frozen_random_reservoir":
                continue
            key = pair_key(episode)
            steps = episode["steps"]
            if name == "terminal":
                terminal[key] = (episode["outcome"], [sha256_json(step) for step in steps],
                                 [step["verification"]["candidate_digest"] for step in steps])
                continue
            baseline, step_digests, candidates = terminal[key]
            summary = comparisons[(name, episode["split"], episode["family"], cfg["mode"], cfg["planned_depth"])]
            summary.update({"pairs": 1, "verified_delta": metrics["verified"] - int(baseline["passed"]),
                            "verifier_call_delta": metrics["verifier_calls"] - baseline["verifier_calls"]})
            if name == "early":
                if [sha256_json(step) for step in steps] != step_digests[:len(steps)]:
                    raise ValueError("early-stop prefix differs from terminal replay")
            else:
                pulse = next(step["intervention"] for step in steps if step.get("intervention"))
                summary["perturbed_visits"] += 1
                for step, candidate in zip(steps, candidates, strict=True):
                    summary[f"changed_candidate_visit_{step['visit'] + 1}"] += int(
                        step["verification"]["candidate_digest"] != candidate
                    )
                summary["total_realized_relative_dose"] += pulse["realized_relative_dose"]
                summary["total_realized_absolute_dose"] += pulse["realized_absolute_dose"]
        if totals["episodes"] != replay["episodes"]:
            raise ValueError("campaign episode count differs from replay")
        runs[name] = {"totals": dict(totals), "verification_codes": dict(codes),
                      "replay": replay, "resources": load_structured(directory / "resources.json"),
                      "trace_bytes": trace.stat().st_size,
                      "groups": [
                          {"split": s, "family": f, "backend": b, "fit": fit, "runtime": mode, "depth": k, **dict(v),
                           "verifier_calls_per_verified": v["verifier_calls"] / v["verified"]
                           if v["verified"] else None,
                           "episode_seconds_per_verified": v["elapsed_episode_seconds"] / v["verified"]
                           if v["verified"] else None}
                          for (s, f, b, fit, mode, k), v in sorted(groups.items())
                      ]}
    return {"schema_version": "rlt_gym.campaign_audit.v1", "runs": runs,
            "matched": [{"comparison": n, "split": s, "family": f, "runtime": m, "depth": k, **dict(v)}
                        for (n, s, f, m, k), v in sorted(comparisons.items())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    dump_structured(args.out, audit(args.root))


if __name__ == "__main__":
    main()
