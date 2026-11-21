"""Select recurrent proxy configurations on calibration and replay fresh holdout tests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import dump_structured

from ..rlt_feedback import MODES
from .runner import read_jsonl, replay_run, run_suite, source_fingerprint


def _episodes(run: Path) -> list[dict]:
    return [row for row in read_jsonl(run / "rollouts.jsonl")
            if row["backend"]["kind"] == "frozen_random_reservoir"]


def _metrics(rows: list[dict], *, early: bool = False, measurements: dict | None = None) -> dict:
    if not rows or any(row["outcome"]["censored"] for row in rows):
        raise ValueError("selection requires complete, definitive episodes")
    outcomes = [row["outcome"] for row in rows]
    first = [o["first_verified_visit"] + 1 for o in outcomes if o["first_verified_visit"] is not None]
    regressions = sum(any(
        not step["verification"]["passed"] for step in row["steps"][index + 1:]
    ) for row in rows if (index := row["outcome"]["first_verified_visit"]) is not None)
    saved = sum(o["realized_visits"] - o["first_verified_visit"] - 1 for o in outcomes
                if early and o["first_verified_visit"] is not None)
    passed = len(first) if early else sum(o["passed"] for o in outcomes)
    visits = sum(o["realized_visits"] for o in outcomes) - saved
    checks = sum(o["verifier_calls"] for o in outcomes) - saved
    elapsed = (sum(measurements[row["episode_id"]] for row in rows)
               if measurements is not None and not early else None)
    return {
        "episodes": len(rows), "passes": passed, "accuracy": passed / len(rows),
        "first_valid_episodes": len(first),
        "first_valid_visit_mean_one_based": sum(first) / len(first) if first else None,
        "regression_episodes": 0 if early else regressions,
        "terminal_failure_after_success": 0 if early else len(first) - passed,
        "visits": visits, "verifier_calls": checks,
        "checks_per_solution": checks / passed if passed else None,
        "episode_ms_mean": 1000 * elapsed / len(rows) if elapsed is not None else None,
    }


def evaluate(
    suite: Path, calibration: Path, out: Path, *, max_wall_seconds: float = 600,
    calibration_replay: Path | None = None,
) -> dict:
    """Replay the frozen full-depth calibration matrix, then confirm preselected arms."""
    suite = suite / "manifest.json" if suite.is_dir() else suite
    manifest = json.loads((calibration / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["configuration"]
    if (cfg["splits"] != ["calibration"] or cfg["verified_stop"] or cfg["intervention"] is not None
            or set(cfg["training_modes"]) != set(MODES) or set(cfg["modes"]) != set(MODES)):
        raise ValueError("expected the full calibration-only crossed schedule matrix")
    if calibration_replay is None:
        audit = replay_run(suite, calibration)
    else:
        audit = json.loads(calibration_replay.read_text(encoding="utf-8"))
        digest = sha256_json({key: value for key, value in manifest.items() if key != "manifest_digest"})
        if (audit.get("schema_version") != "rlt_gym.replay.v1" or audit.get("passed") is not True
                or digest != manifest["manifest_digest"] or digest != audit["run_manifest_digest"]
                or manifest["suite_manifest_sha256"] != sha256_file(suite)
                or audit["source_fingerprint"] != source_fingerprint()
                or audit["exact_trajectory_replays"] != manifest["expected_episodes"]
                or audit["receipt_only_aborted_episodes"] != 0
                or any(sha256_file(calibration / filename) != manifest["files"][filename]
                       for filename in ("rollouts.jsonl", "measurements.jsonl"))
                or audit["rollouts_sha256"] != manifest["files"]["rollouts.jsonl"]):
            raise ValueError("calibration replay receipt does not bind the complete matrix")
    cells = defaultdict(list)
    for row in _episodes(calibration):
        cells[(row["backend"]["training"]["training_mode"], row["config"]["mode"],
               row["config"]["planned_depth"])].append(row)
    timings = {row["episode_id"]: row["elapsed_seconds"]
               for row in read_jsonl(calibration / "measurements.jsonl")}
    scores = [{
        "fit": fit, "runtime": mode, "depth": depth, "stop_on_verified": early,
        "extrapolates_fit_depth": depth > cfg["training_depth"],
        **_metrics(rows, early=early, measurements=timings),
    } for (fit, mode, depth), rows in sorted(cells.items()) for early in (False, True)]

    def rank(row):
        return (-row["passes"], row["verifier_calls"], row["depth"], row["fit"], row["runtime"])

    selected = {
        f"{mode}-{'early' if early else 'fixed'}": min(
            (row for row in scores if row["fit"] == row["runtime"] == mode and row["stop_on_verified"] == early),
            key=rank,
        )
        for mode in MODES for early in (False, True)
    }
    for early in (False, True):
        group = [row for row in scores if row["stop_on_verified"] == early]
        winner = min(group, key=rank)
        if winner["fit"] != winner["runtime"] and rank(winner)[:2] < min(
            rank(row)[:2] for row in group if row["fit"] == row["runtime"]
        ):
            selected[f"crossed-{'early' if early else 'fixed'}"] = winner
    out.mkdir(parents=True, exist_ok=False)
    dump_structured(out / "selection.json", {
        "suite_sha256": sha256_file(suite), "calibration_replay": audit,
        "calibration_early_stop_is_exact_prefix": True, "scores": scores, "selected": selected,
    })
    confirmation = {}
    for tag, arm in selected.items():
        run = out / tag
        result = run_suite(
            suite, run, seeds=tuple(cfg["seeds"]), training_modes=(arm["fit"],),
            modes=(arm["runtime"],), depths=(arm["depth"],), splits=("ordinary_test", "ood_test"),
            training_depth=cfg["training_depth"], readout_site=cfg["readout_site"],
            verified_stop=arm["stop_on_verified"], include_controls=False,
            max_wall_seconds=max_wall_seconds / len(selected),
        )
        receipt = replay_run(suite, run)
        rows = _episodes(run)
        timings = {row["episode_id"]: row["elapsed_seconds"] for row in read_jsonl(run / "measurements.jsonl")}
        confirmation[tag] = {
            "replay": receipt, "wall_seconds": result["resources"]["wall_seconds"],
            "splits": {split: _metrics([row for row in rows if row["split"] == split], measurements=timings)
                       for split in ("ordinary_test", "ood_test")},
        }
        dump_structured(out / "confirmation.json", confirmation)
    return {"confirmed_arms": len(confirmation),
            "episodes": sum(row["replay"]["episodes"] for row in confirmation.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=600)
    parser.add_argument("--calibration-replay", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(
        args.suite, args.calibration, args.out, max_wall_seconds=args.max_wall_seconds,
        calibration_replay=args.calibration_replay,
    )))
