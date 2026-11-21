"""Audit paired native Inkling test rollouts and replay a frozen depth selector."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured
from datagoblin.training_gym.rlt_feedback import MODES

from .inkling import MAX_RESPONSE_BYTES, _decode_candidate, _jsonl, _manifest, _new_dir, _suite_file
from .inkling_native import _finished_rows
from .inkling_routing import routing_strength
from .inkling_sae import _encode
from .inkling_trace import MAX_PROMPTS, MAX_TOKENS, NATIVE_ANSWER, SHA256, WIDTH, native_prompt
from .tasks import load_suite, public_task, verify

DEPTHS = (1, 3, 6)
SPLITS = ("ordinary_test", "ood_test")
PROVENANCE = ("weights_sha256", "runtime_archive_sha256", "native_program_sha256", "placement_sha256")


def audit_holdout_plan(suite: Path, holdout: Path) -> tuple[dict, dict, dict]:
    suite_file = _suite_file(suite)
    plan = _manifest(holdout / "manifest.json")
    if (plan["schema_version"] != "rlt_gym.inkling_native_holdout_plan.v1"
            or plan["suite_sha256"] != sha256_file(suite_file) or tuple(plan["fixed_depths"]) != DEPTHS
            or not 1 <= plan["model_new_token_cap"] <= 384):
        raise ValueError("invalid native holdout plan or suite")
    _, suite_tasks = load_suite(suite_file)
    tasks = {task["task_id"]: task for task in suite_tasks}
    batches, roots = defaultdict(list), set()
    for row in plan["rows"]:
        batch = row["batch"]
        task = tasks[row["task_id"]]
        if (not batch.startswith("batch-") or not batch[6:].isascii() or not batch[6:].isdigit()
                or row["sample_key"] != f"{len(batches[batch]):03d}" or len(batches[batch]) >= MAX_PROMPTS
                or task["split"] not in SPLITS or task["lineage_id"] in roots
                or any(row[k] != task[k] for k in ("split", "family", "task_digest", "lineage_id"))):
            raise ValueError("holdout contains a duplicate, mismatched or unsupported task")
        path = holdout / batch / f"{row['sample_key']}.txt"
        expected = native_prompt({"context": {"task": public_task(task), "history": []}}).encode("utf-8")
        if sha256_file(path) != row["prompt_sha256"] or path.read_bytes() != expected:
            raise ValueError("holdout prompt differs from its original public task")
        roots.add(task["lineage_id"])
        batches[batch].append(row)
    if not batches:
        raise ValueError("empty native holdout")
    for batch, rows in batches.items():
        if {p.stem for p in (holdout / batch).glob("*.txt")} != {row["sample_key"] for row in rows}:
            raise ValueError("unexpected staged holdout prompt")
    return plan, tasks, dict(batches)


def _policy(plan: dict, reference: Path, fit_prompts: Path) -> tuple[dict, dict, Path, Path]:
    trace = reference / "p03-raw" / "manifest.json"
    policy = reference / "audited" / "routing" / "manifest.json"
    head_receipt = reference / "answer-heads" / "manifest.json"
    dictionary = reference / "answer-sae" / "sae_dictionary.npz"
    heads = head_receipt.parent / "heads.npz"
    sealed, frozen, transition = (_manifest(path) for path in (trace, policy, head_receipt))
    prompts = _manifest(fit_prompts / "manifest.json")
    fitted_roots = {row["lineage_id"] for row in prompts["samples"]}
    if (sha256_file(trace) != plan["prior_trace_sha256"] or sha256_file(policy) != plan["frozen_depth_policy_sha256"]
            or sealed["input_format"] != NATIVE_ANSWER or sealed["passes"] != 3
            or sealed["source_prompt_manifest_sha256"] != sha256_file(fit_prompts / "manifest.json")
            or prompts["suite_manifest_sha256"] != plan["suite_sha256"]
            or any(row["split"] not in {"train", "calibration"} for row in prompts["samples"])
            or fitted_roots.intersection(row["lineage_id"] for row in plan["rows"])
            or frozen["trace_feedback_sha256"] != sha256_file(head_receipt)
            or transition["trace_manifest_sha256"] != plan["prior_trace_sha256"]
            or transition["dictionary_sha256"] != sha256_file(dictionary)
            or transition["heads_sha256"] != sha256_file(heads)):
        raise ValueError("held-out routing needs the original frozen fit and disjoint roots")
    return sealed, frozen, dictionary, heads


def _read_arm(run: Path, rows: list[dict], tasks: dict, depth: int, cap: int) -> tuple[list[dict], dict]:
    metadata = _finished_rows(run / "native-config.tsv")
    if (len(metadata) != 1 or int(metadata[0]["passes"]) != depth
            or int(metadata[0]["context"]) != MAX_TOKENS or int(metadata[0]["max_new_tokens"]) != cap
            or int(metadata[0].get("intervention_pass", 0)) != 0):
        raise ValueError("native holdout configuration differs from the plan")
    completed = _finished_rows(run / "completed.tsv")
    expected_keys = [row["sample_key"] for row in rows]
    if ([row["sample_key"] for row in completed] != expected_keys[:len(completed)]
            or ((run / "native-complete.txt").exists() and len(completed) != len(expected_keys))):
        raise ValueError("holdout receipts must cover a durable ordered prefix")
    outcomes = []
    for case, status in zip(rows, completed, strict=False):
        n_input, n_output, elapsed = (int(status[k]) for k in ("prompt_tokens", "generated_tokens", "elapsed_ms"))
        reason = status["finish_reason"]
        if (not 1 <= n_input <= MAX_TOKENS - cap or not 1 <= n_output <= cap or elapsed < 0
                or reason not in ("stop", "length") or (reason == "length" and n_output != cap)):
            raise ValueError("invalid native holdout usage")
        reply = run / f"{case['sample_key']}.reply"
        if reply.stat().st_size > MAX_RESPONSE_BYTES:
            raise ValueError("oversized native holdout reply")
        try:
            parsed, candidate, _ = _decode_candidate(reply.read_bytes().decode("utf-8"))
        except UnicodeDecodeError:
            parsed, candidate = False, None
        verdict = verify(tasks[case["task_id"]], candidate) if parsed else {"passed": False, "status": "parse_failure"}
        outcomes.append({
            **case, "passes": depth, "prompt_tokens": n_input, "generated_tokens": n_output,
            "finish_reason": reason, "elapsed_ms": elapsed, "passed": verdict["passed"], "verdict": verdict,
            "verifier_calls": int(parsed), "reply_sha256": sha256_file(reply),
        })
    return outcomes, metadata[0]


def _prelude(run: Path, rows: list[dict]) -> dict[str, np.ndarray]:
    tokens = {row["sample_key"]: str(row["prompt_tokens"] - 1) for row in rows}
    result = {}
    if not tokens:
        return result
    with (run / "rows.tsv").open(encoding="utf-8", newline="") as index, (run / "activations.f32").open("rb") as data:
        for offset, row in enumerate(csv.DictReader(index, delimiter="\t")):
            key = row["sample_key"]
            if key not in tokens or row["pass_index"] != "0" or row["token_index"] != tokens[key]:
                continue
            if row["tensor"] != "l_out-5" or key in result:
                raise ValueError("invalid holdout prelude trace")
            data.seek(offset * WIDTH * 4)
            vector = np.frombuffer(data.read(WIDTH * 4), dtype="<f4")
            if vector.shape != (WIDTH,) or not np.isfinite(vector).all():
                raise ValueError("invalid holdout prelude state")
            result[key] = vector
            if result.keys() == tokens.keys():
                break
    if result.keys() != tokens.keys():
        raise ValueError("a completed holdout is missing its last-token prelude state")
    return result


def _metrics(outputs: list[dict], work: list[dict]) -> dict:
    calls = sum(row["verifier_calls"] for row in outputs)
    solved = sum(row["passed"] for row in outputs)
    proxy = sum((12 + 12 * row["passes"]) * (row["prompt_tokens"] + row["generated_tokens"]) for row in work)
    return {
        "roots": len(outputs), "verified_correct": solved, "verifier_calls": calls, "model_calls": len(work),
        "parse_failures": len(outputs) - calls, "observed_elapsed_ms": sum(row["elapsed_ms"] for row in work),
        "layer_visit_token_proxy": proxy, "proxy_per_verified_solution": proxy / solved if solved else None,
    }


def _paired_metrics(groups: list[dict], choices: dict, depths: tuple[int, ...]) -> dict:
    matched = [group for group in groups if all(depth in group for depth in depths)]
    fixed = {
        str(depth): _metrics([group[depth] for group in matched], [group[depth] for group in matched])
        for depth in depths
    }
    regressions = sum(
        any(group[depths[i]]["passed"] and not group[late]["passed"]
            for i in range(len(depths)) for late in depths[i + 1:])
        for group in matched
    )
    first = {str(depth): 0 for depth in depths}
    for group in matched:
        valid = next((depth for depth in depths if group[depth]["passed"]), None)
        if valid is not None:
            first[str(valid)] += 1
    result = {"matched_roots": len(matched), "fixed": fixed, "first_valid_depth": first, "regressed_roots": regressions}
    if depths == (1, 3):
        result["routing"] = {}
        for mode in MODES:
            outputs = [group[choices[group[1]["lineage_id"]][mode]["chosen_depth"]] for group in matched]
            work = [group[1] for group in matched] + [row for row in outputs if row["passes"] == 3]
            result["routing"][mode] = _metrics(outputs, work)
    return result


def score_holdout(suite: Path, holdout: Path, runs: Path, reference: Path, fit_prompts: Path, out: Path) -> dict:
    plan, tasks, batches = audit_holdout_plan(suite, holdout)
    sealed, frozen, dictionary, heads = _policy(plan, reference, fit_prompts)
    plan_sha = sha256_file(holdout / "manifest.json")
    outcomes, initial, sources = [], [], {}
    per_root = defaultdict(dict)
    provenance = None
    for batch, tasks_in_batch in batches.items():
        for depth in DEPTHS:
            path = runs / batch / f"p{depth:02d}"
            raw = path / "raw"
            if not (raw / "completed.tsv").is_file():
                continue
            receipt = _manifest(path / "command.json")
            arm, config = _read_arm(raw, tasks_in_batch, tasks, depth, plan["model_new_token_cap"])
            identity = (*(receipt[key] for key in PROVENANCE), config["requested_load_mode"])
            if (receipt["holdout_plan_sha256"] != plan_sha or receipt["batch"] != batch or receipt["passes"] != depth
                    or any(SHA256.fullmatch(receipt[k]) is None for k in PROVENANCE)
                    or receipt["weights_sha256"] != sealed["weights_sha256"]
                    or receipt["runtime_archive_sha256"] != sealed["runtime_sha256"]
                    or (provenance is not None and identity != provenance)):
                raise ValueError("holdout arms must use the same frozen checkpoint, runtime and placement")
            provenance = identity
            for row in arm:
                group = per_root[row["lineage_id"]]
                if group and row["prompt_tokens"] != next(iter(group.values()))["prompt_tokens"]:
                    raise ValueError("a matched root used different prompt token counts across depths")
                group[depth] = row
            outcomes.extend(arm)
            if depth == 1:
                states = _prelude(raw, arm)
                initial.extend((row["lineage_id"], states[row["sample_key"]]) for row in arm)
            sources[f"{batch}/p{depth:02d}"] = {
                "command": sha256_file(path / "command.json"), "completed": sha256_file(raw / "completed.tsv"),
                "config": sha256_file(raw / "native-config.tsv"),
            }
    choices = defaultdict(dict)
    if initial:
        with np.load(dictionary, allow_pickle=False) as w:
            latent = _encode(np.stack([state for _, state in initial]) - w["pre_bias"],
                             w["encoder_weight"], w["encoder_bias"], int(w["top_k"]))
        with np.load(heads, allow_pickle=False) as w:
            for mode in MODES:
                cut = frozen["fitted"][mode]["threshold"]
                for (root, _), strength in zip(initial, routing_strength(latent, w[mode], mode), strict=True):
                    choices[root][mode] = {"strength": float(strength), "chosen_depth": 3 if strength >= cut else 1}
    results = {}
    for split in SPLITS:
        for family in ("all", *sorted({row["family"] for row in plan["rows"] if row["split"] == split})):
            roots = {row["lineage_id"] for row in plan["rows"]
                     if row["split"] == split and (family == "all" or row["family"] == family)}
            groups = [per_root[root] for root in sorted(roots & per_root.keys())]
            results[f"{split}/{family}"] = {
                "paired_1_3": _paired_metrics(groups, choices, (1, 3)),
                "paired_1_3_6": _paired_metrics(groups, choices, DEPTHS),
            }
    _new_dir(out)
    _jsonl(out / "outcomes.jsonl", outcomes)
    _jsonl(out / "routing.jsonl", [{
        "lineage_id": root, "prelude_sha256": hashlib.sha256(state).hexdigest(),
        "prompt_sha256": per_root[root][1]["prompt_sha256"],
        "policies": choices[root],
    } for root, state in initial])
    report = {
        "schema_version": "rlt_gym.inkling_native_holdout_score.v1",
        "holdout_plan_sha256": plan_sha, "frozen_depth_policy_sha256": plan["frozen_depth_policy_sha256"],
        "planned_roots": len(plan["rows"]), "planned_attempts": len(plan["rows"]) * len(DEPTHS),
        "observed_attempts": len(outcomes), "verifier_calls": sum(row["verifier_calls"] for row in outcomes),
        "scope": "native_fixed_depth_with_frozen_restart_routing_replay",
        "sources": sources, "results": results,
        "files": {name: sha256_file(out / name) for name in ("outcomes.jsonl", "routing.jsonl")},
    }
    dump_structured(out / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("suite", "holdout", "runs", "reference", "fit-prompts", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    result = score_holdout(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("planned_attempts", "observed_attempts", "verifier_calls")}))


if __name__ == "__main__":
    main()
