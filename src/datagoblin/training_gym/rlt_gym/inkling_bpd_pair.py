"""Verify a frozen Inkling geometry pair and score matched native readouts."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured

from .inkling import _jsonl, _manifest, _new_dir, _suite_file
from .inkling_feature_trial import ACTIVATION_OPERATOR, ARMS, score_trial
from .inkling_native import _finished_rows
from .inkling_trace import SHA256

GEOMETRIES = ("aligned", "orthogonal")
RUNTIME_DIGESTS = (
    "native_program_sha256", "placement_sha256", "runtime_archive_sha256", "runtime_llama_sha256",
    "suite_sha256", "weight_sha256",
)


def audit_geometry_pair(trials: Path) -> dict:
    """Check that the two staged plans differ only in intervention geometry."""
    selection_path = trials / "geometry.json"
    selection = _manifest(selection_path)
    selection_sha = sha256_file(selection_path)
    dose = selection["requested_l2"]
    if (selection.get("schema_version") != "rlt_gym.inkling_bpd_matched_geometry.v1"
            or selection.get("native_mechanism_verified") is not False
            or not math.isfinite(dose) or dose <= 0):
        raise ValueError("expected a frozen equal-dose geometry pair")

    plans = {}
    for name in GEOMETRIES:
        plan = _manifest(trials / name / "manifest.json")
        if (plan.get("schema_version") != "rlt_gym.inkling_feature_trial.v1"
                or plan.get("geometry") != name or plan.get("selection_receipt_sha256") != selection_sha
                or plan.get("intervention_pass") != 2 or plan.get("intervention_operator") != ACTIVATION_OPERATOR
                or any(plan[key] != selection[key] for key in ("seed", "feature"))
                or plan["requested_dose"] != dose):
            raise ValueError(f"{name} trial does not match the frozen geometry selection")
        plans[name] = plan

    aligned, orthogonal = (plans[name] for name in GEOMETRIES)
    common_keys = ("suite_sha256", "weights_sha256", "dictionary_sha256")
    if any(aligned[key] != orthogonal[key] for key in common_keys):
        raise ValueError("geometry trials use different suite, checkpoint, or SAE")
    rows = aligned["rows"]
    contexts = [
        [{key: value for key, value in row.items() if key != "delta_sha256"} for row in plan["rows"]]
        for plan in plans.values()
    ]
    if not rows or contexts[0] != contexts[1]:
        raise ValueError("geometry trials use different ordered roots, prompts, or arms")
    if len({row["sample_key"] for row in rows}) != len(rows):
        raise ValueError("geometry trial sample keys must be unique")

    by_root = defaultdict(list)
    for row in rows:
        by_root[row["lineage_id"]].append(row)
    if any([row["arm"] for row in arms] != list(ARMS)
           or len({row["prompt_sha256"] for row in arms}) != 1 for arms in by_root.values()):
        raise ValueError("each geometry root requires four ordered arms with the same prompt")
    return {
        "selection_sha256": selection_sha,
        "feature": selection["feature"],
        "requested_dose": dose,
        "planned_roots": len(by_root),
    }


def _command(suite: Path, trial: Path, run: Path) -> dict:
    command = _manifest(run / "command.json")
    receipt = _manifest(run / "resource-receipt.json")
    plan = _manifest(trial / "manifest.json")
    argv = command["argv"]
    if (any(not isinstance(command.get(key), str) or not SHA256.fullmatch(command[key])
            for key in RUNTIME_DIGESTS)
            or command.get("trial_manifest_sha256") != sha256_file(trial / "manifest.json")
            or command["suite_sha256"] != sha256_file(_suite_file(suite))
            or command["suite_sha256"] != plan["suite_sha256"]
            or command["weight_sha256"] != plan["weights_sha256"]
            or command.get("operator") != ACTIVATION_OPERATOR
            or type(receipt.get("native_returncode")) is not int
            or len(argv) != 13 or argv[4] != "3" or argv[12] != "2"
            or Path(argv[7]).resolve() != (trial / "prompts").resolve()
            or Path(argv[8]).resolve() != (run / "raw").resolve()
            or Path(argv[11]).resolve() != (trial / "deltas").resolve()):
        raise ValueError("geometry capture does not bind to its frozen native trial")
    if receipt["native_returncode"] == 0 and len(_finished_rows(run / "raw" / "completed.tsv")) != len(plan["rows"]):
        raise ValueError("successful geometry capture has an incomplete prompt list")
    return command


def _runtime_identity(command: dict) -> tuple:
    argv = tuple(arg for index, arg in enumerate(command["argv"]) if index not in (7, 8, 11))
    return (*(command[key] for key in RUNTIME_DIGESTS), argv,
            command.get("placement_plan_sha256"), command.get("whole_file_prefetch_disabled"),
            command.get("stream_gpu_weights_requested"))


def _effects(comparison: dict) -> tuple[int, float]:
    arms = comparison["arms"]
    distances = comparison["final_pass_state_l2_vs_untreated"]
    if set(distances) != set(ARMS[1:]) or any(not math.isfinite(x) or x < 0 for x in distances.values()):
        raise ValueError("geometry pair requires finite final-pass distances for the intervention arms")
    return (
        int(arms["component"]["verdict"]["passed"]) - int(arms["matched_random"]["verdict"]["passed"]),
        distances["component"] - distances["matched_random"],
    )


def score_geometry_pair(suite: Path, trials: Path, aligned_run: Path, orthogonal_run: Path, out: Path) -> dict:
    """Independently score both native captures and compare only shared complete roots."""
    selection = audit_geometry_pair(trials)
    runs = dict(zip(GEOMETRIES, (aligned_run, orthogonal_run), strict=True))
    commands = {name: _command(suite, trials / name, run) for name, run in runs.items()}
    if _runtime_identity(commands["aligned"]) != _runtime_identity(commands["orthogonal"]):
        raise ValueError("geometry captures use different runtime, placement, or decoding settings")

    _new_dir(out)
    scored = {}
    comparisons = {}
    for name, run in runs.items():
        report = out / name
        scored[name] = score_trial(
            suite, trials / name, run / "raw", report,
            native_program_sha256=commands[name]["native_program_sha256"],
        )
        with (report / "comparisons.jsonl").open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        comparisons[name] = {row["arms"]["untreated"]["lineage_id"]: row for row in rows}
        if len(comparisons[name]) != scored[name]["matched_roots"]:
            raise ValueError("duplicate completed geometry root")

    aligned, orthogonal = (comparisons[name] for name in GEOMETRIES)
    pairs = []
    for root in sorted(aligned.keys() & orthogonal.keys()):
        a, b = aligned[root], orthogonal[root]
        if len({arm["prefill_token"] for case in (a, b) for arm in case["arms"].values()}) != 1:
            raise ValueError("geometry comparison uses different native token positions")
        a_solve, a_state = _effects(a)
        b_solve, b_state = _effects(b)
        controls = {arm: a["arms"][arm]["reply_sha256"] == b["arms"][arm]["reply_sha256"] for arm in ARMS[:2]}
        pairs.append({
            "lineage_id": root,
            "control_replies_match": controls,
            "component_minus_random": {
                "aligned": {"solve": a_solve, "final_state_l2": a_state},
                "orthogonal": {"solve": b_solve, "final_state_l2": b_state},
            },
            "aligned_minus_orthogonal_solve_contrast": a_solve - b_solve,
            "aligned_minus_orthogonal_state_l2_contrast": a_state - b_state,
        })

    paired_path = out / "pairs.jsonl"
    _jsonl(paired_path, pairs)
    result = {
        "schema_version": "rlt_gym.inkling_bpd_geometry_pair_score.v1",
        **selection,
        "observed_answers": {name: value["answers"] for name, value in scored.items()},
        "complete_roots": {name: len(value) for name, value in comparisons.items()},
        "paired_roots": len(pairs),
        "complete_pairing": len(pairs) == selection["planned_roots"],
        "paired_roots_with_matching_control_replies": sum(all(row["control_replies_match"].values()) for row in pairs),
        "mean_solve_contrast": math.fsum(row["aligned_minus_orthogonal_solve_contrast"] for row in pairs) / len(pairs)
        if pairs else None,
        "mean_final_state_l2_contrast": math.fsum(
            row["aligned_minus_orthogonal_state_l2_contrast"] for row in pairs
        ) / len(pairs) if pairs else None,
        "source_command_sha256": {name: sha256_file(run / "command.json") for name, run in runs.items()},
        "source_resource_sha256": {name: sha256_file(run / "resource-receipt.json") for name, run in runs.items()},
        "verified_geometry_manifest_sha256": {name: sha256_file(out / name / "manifest.json") for name in GEOMETRIES},
        "pairs_sha256": sha256_file(paired_path),
        "bpd_mechanism_verified": False,
    }
    dump_structured(out / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect").add_argument("--trials", type=Path, required=True)
    score = commands.add_parser("score")
    for field in ("suite", "trials", "aligned-run", "orthogonal-run", "out"):
        score.add_argument(f"--{field}", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = audit_geometry_pair(**args) if command == "inspect" else score_geometry_pair(**args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
