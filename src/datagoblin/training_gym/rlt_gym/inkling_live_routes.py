"""Stage and verify a prospective Inkling prompt-routing comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured

from .inkling import _jsonl, _manifest, _new_dir
from .inkling_holdout import PROVENANCE, SPLITS, _metrics, _read_arm, audit_holdout_plan, capture_segments
from .inkling_prompt_router import ROUTES_SCHEMA
from .inkling_trace import MAX_PROMPTS, SHA256
from .runner import read_jsonl

REQUEST_SCHEMA = "rlt_gym.inkling_live_route_comparison.v1"
GROUPS = (
    ("routed-p01", "routed", 1),
    ("routed-p03", "routed", 3),
    ("fixed-p03", "fixed_3", 3),
)


def _request_plan(suite: Path, holdout: Path, routes: Path, prior: Path) -> tuple[dict, dict, dict]:
    holdout_plan, tasks, _ = audit_holdout_plan(suite, holdout)
    holdout_sha = sha256_file(holdout / "manifest.json")
    routes_path, prior_path = routes / "manifest.json", prior / "manifest.json"
    policy, excluded = _manifest(routes_path), _manifest(prior_path)
    choices_path, outcomes_path = routes / "choices.jsonl", prior / "outcomes.jsonl"
    if (policy["schema_version"] != ROUTES_SCHEMA or policy["holdout_plan_sha256"] != holdout_sha
            or excluded["schema_version"] != "rlt_gym.inkling_native_holdout_score.v1"
            or excluded["holdout_plan_sha256"] != holdout_sha
            or policy["choices_sha256"] != sha256_file(choices_path)
            or excluded["files"]["outcomes.jsonl"] != sha256_file(outcomes_path)):
        raise ValueError("live routing inputs do not share the frozen holdout plan")
    cases = {row["lineage_id"]: row for row in holdout_plan["rows"]}
    observed = {row["lineage_id"] for row in read_jsonl(outcomes_path)}
    choices = read_jsonl(choices_path)
    if (len(choices) != len(cases) or {row["lineage_id"] for row in choices} != cases.keys()
            or observed - cases.keys()):
        raise ValueError("live routing inputs have missing or unexpected held-out roots")
    for choice in choices:
        expected = cases[choice["lineage_id"]]
        if any(choice[k] != v for k, v in expected.items()) or choice["chosen_depth"] not in (1, 3):
            raise ValueError("a route differs from its frozen public case")
    pending = [choice for choice in choices if choice["lineage_id"] not in observed]
    if not 1 <= len(pending) <= MAX_PROMPTS:
        raise ValueError("live routing needs an unobserved cohort within the native prompt limit")
    groups = []
    for name, policy_name, depth in GROUPS:
        selected = [row for row in pending if policy_name == "fixed_3" or row["chosen_depth"] == depth]
        if selected:
            groups.append({
                "group": name, "policy": policy_name, "passes": depth,
                "rows": [{**row, "original_sample_key": row["sample_key"], "sample_key": f"{i:03d}"}
                         for i, row in enumerate(selected)],
            })
    return {
        "schema_version": REQUEST_SCHEMA, "holdout_plan_sha256": holdout_sha,
        "routes_manifest_sha256": sha256_file(routes_path), "excluded_audit_sha256": sha256_file(prior_path),
        "model_new_token_cap": holdout_plan["model_new_token_cap"], "planned_roots": len(pending),
        "selector_wall_ms_all_routes": policy["selection_wall_ms"], "groups": groups,
    }, tasks, holdout_plan


def stage_live_requests(suite: Path, holdout: Path, routes: Path, prior: Path, out: Path) -> dict:
    manifest, _, _ = _request_plan(suite, holdout, routes, prior)
    _new_dir(out)
    for group in manifest["groups"]:
        directory = out / group["group"]
        directory.mkdir()
        for row in group["rows"]:
            source = holdout / row["batch"] / (row["original_sample_key"] + ".txt")
            (directory / (row["sample_key"] + ".txt")).write_bytes(source.read_bytes())
    dump_structured(out / "manifest.json", manifest)
    return manifest


def audit_live_requests(
    suite: Path, holdout: Path, routes: Path, prior: Path, requests: Path,
) -> tuple[dict, dict, dict]:
    expected, tasks, holdout_plan = _request_plan(suite, holdout, routes, prior)
    if _manifest(requests / "manifest.json") != expected:
        raise ValueError("live request manifest differs from the frozen prompt choices")
    for group in expected["groups"]:
        directory = requests / group["group"]
        if {path.stem for path in directory.glob("*.txt")} != {row["sample_key"] for row in group["rows"]}:
            raise ValueError("unexpected live routing prompt")
        for row in group["rows"]:
            if sha256_file(directory / (row["sample_key"] + ".txt")) != row["prompt_sha256"]:
                raise ValueError("live routing prompt changed after staging")
    return expected, tasks, holdout_plan


def _capture_group(path: Path, group: dict, tasks: dict, cap: int, request_sha: str, sealed: dict):
    captured, receipts = [], {}
    identity = None
    for segment in capture_segments(path):
        command_path = segment / "command.json"
        if not command_path.is_file():
            continue
        command = _manifest(command_path)
        resource = _manifest(segment / "resource-receipt.json")
        if type(resource["launched"]) is not bool or not isinstance(resource["status"], str):
            raise ValueError("invalid terminal live resource receipt")
        observed = (*(command[key] for key in PROVENANCE), command["load_mode"])
        if (command["request_plan_sha256"] != request_sha or command["group"] != group["group"]
                or command["passes"] != group["passes"] or command["prompt_offset"] != len(captured)
                or any(SHA256.fullmatch(command[key]) is None for key in PROVENANCE)
                or command["weights_sha256"] != sealed["weights_sha256"]
                or command["runtime_archive_sha256"] != sealed["runtime_sha256"]
                or (identity is not None and observed != identity)):
            raise ValueError("live capture differs from its scheduled depth, model, or resume position")
        identity = observed
        wall = float(resource["wall_seconds"]) if resource["launched"] else 0.0
        if not math.isfinite(wall) or wall < 0:
            raise ValueError("invalid live resource timing")
        summary = {
            "wall_seconds": wall, "status": resource["status"], "launched": resource["launched"],
            "command_sha256": sha256_file(command_path),
            "resource_sha256": sha256_file(segment / "resource-receipt.json"),
        }
        raw = segment / "raw"
        remaining = group["rows"][len(captured):]
        if (raw / "completed.tsv").is_file():
            if not resource["launched"] or not remaining:
                raise ValueError("unexpected live rollout trace")
            scored, config = _read_arm(raw, remaining, tasks, group["passes"], cap)
            if config["requested_load_mode"] != command["load_mode"]:
                raise ValueError("live arm used a different load mode")
            captured.extend(scored)
            summary["completed_sha256"] = sha256_file(raw / "completed.tsv")
        if resource["status"] == "complete" and (len(captured) != len(group["rows"])
                                                or not (raw / "native-complete.txt").is_file()):
            raise ValueError("live execution claimed completion without covering its prompts")
        receipts[segment.relative_to(path.parent).as_posix()] = summary
    return captured, receipts, identity


def score_live_requests(
    suite: Path, holdout: Path, routes: Path, prior: Path, requests: Path, runs: Path, reference: Path, out: Path,
) -> dict:
    plan, tasks, holdout_plan = audit_live_requests(suite, holdout, routes, prior, requests)
    trace_path = reference / "p03-raw" / "manifest.json"
    if sha256_file(trace_path) != holdout_plan["prior_trace_sha256"]:
        raise ValueError("live runtime must match the original checkpoint and backend release")
    sealed = _manifest(trace_path)
    request_sha = sha256_file(requests / "manifest.json")
    outputs = {"routed": {}, "fixed_3": {}}
    runtime, resources = None, {}
    for group in plan["groups"]:
        scored, receipts, identity = _capture_group(
            runs / group["group"], group, tasks, plan["model_new_token_cap"], request_sha, sealed,
        )
        if identity is not None:
            if runtime is not None and identity != runtime:
                raise ValueError("live routes and fixed-depth control need identical runtime provenance")
            runtime = identity
        outputs[group["policy"]].update((row["lineage_id"], row) for row in scored)
        resources[group["group"]] = receipts
    paired = outputs["routed"].keys() & outputs["fixed_3"].keys()
    if any(outputs["routed"][root]["prompt_tokens"] != outputs["fixed_3"][root]["prompt_tokens"] for root in paired):
        raise ValueError("paired live routes used different prompt token counts")
    results = {}
    for split in (*SPLITS, "all"):
        roots = sorted(root for root in paired if split == "all" or outputs["routed"][root]["split"] == split)
        results[split] = {}
        for policy, by_root in outputs.items():
            rows = [by_root[root] for root in roots]
            results[split][policy] = _metrics(rows, rows)
    _new_dir(out)
    outcomes = out / "outcomes.jsonl"
    _jsonl(outcomes, [{**row, "policy": policy} for policy, by_root in outputs.items() for row in by_root.values()])
    report = {
        "schema_version": "rlt_gym.inkling_live_route_score.v1",
        "request_plan_sha256": request_sha, "planned_roots": plan["planned_roots"], "paired_roots": len(paired),
        "observed_attempts": {policy: len(rows) for policy, rows in outputs.items()},
        "results": results, "resources": resources, "outcomes_sha256": sha256_file(outcomes),
        "selector_wall_ms_all_routes": plan["selector_wall_ms_all_routes"],
        "complete_pairing": len(paired) == plan["planned_roots"],
        "model_wall_seconds_including_startup_and_retries": {
            policy: sum(receipt["wall_seconds"] for group in plan["groups"] if group["policy"] == policy
                        for receipt in resources[group["group"]].values())
            for policy in outputs
        },
    }
    dump_structured(out / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    for name, handler, additional in (
        ("stage", stage_live_requests, ()),
        ("score", score_live_requests, ("requests", "runs", "reference")),
    ):
        command = commands.add_parser(name)
        command.set_defaults(handler=handler)
        for field in ("suite", "holdout", "routes", "prior", *additional, "out"):
            command.add_argument(f"--{field}", type=Path, required=True)
    args = vars(parser.parse_args())
    handler = args.pop("handler")
    result = handler(**args)
    print(json.dumps({field: result[field] for field in ("schema_version", "planned_roots")}))


if __name__ == "__main__":
    main()
