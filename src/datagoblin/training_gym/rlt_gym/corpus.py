"""Assemble deduplicated Aleph supervision from replayed rollout streams."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import canonical_json, dump_structured, load_structured

from .runner import verify_episode
from .tasks import _strings, load_suite, public_task
from .teachers import _history, _validate_episode

KINDS = ("sft", "repairs", "transitions", "outcomes", "preferences", "research")


def records(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def state_features(summary):
    return {key: summary[key] for key in ("shape", "rms", "l2", "mean", "max_abs")}


def plan_issues(world, candidate):
    if not isinstance(candidate, dict):
        return ["plan_shape"]
    keys = {"arms", "replicates", "tokens_per_run", "training_split", "selection_split",
            "confirmation_split", "independent_verifier", "metrics"}
    if candidate.keys() != keys:
        return ["plan_shape"]
    arms, metrics = candidate["arms"], candidate["metrics"]
    replicates, tokens = candidate["replicates"], candidate["tokens_per_run"]
    checks = {
        "matched_arms": _strings(arms, maximum=8) and set(arms) == set(world["required_arms"]),
        "repetitions_and_run_budget": type(replicates) is int and isinstance(arms, list)
        and world["minimum_replicates"] <= replicates <= 100 and len(arms) * replicates <= world["max_runs"],
        "token_budget": type(tokens) is int and 1 <= tokens <= world["max_tokens_per_run"],
        "split_custody": [candidate[f"{stage}_split"] for stage in ("training", "selection", "confirmation")]
        == ["train", "calibration", "ordinary_test"],
        "independent_verifier": candidate["independent_verifier"] is True,
        "required_metrics": _strings(metrics, maximum=8) and set(metrics) == set(world["required_metrics"]),
    }
    return [name for name, passed in checks.items() if not passed]


def capture_receipts(capture, imported):
    manifest = load_structured(capture / "manifest.json")
    for name, field in (("responses.jsonl", "responses_sha256"), ("invocations.jsonl", "invocations_sha256")):
        if sha256_file(capture / name) != manifest[field]:
            raise ValueError("Codex capture artifact changed")
    if (manifest["responses_sha256"] != imported["responses_sha256"]
            or manifest["request_file_sha256"] != imported["requests_sha256"]):
        raise ValueError("provider capture is for a different import")
    by_request = {}
    for call in records(capture / "invocations.jsonl"):
        event_file = (capture / call["event_file"]).resolve()
        if (event_file.parent != capture.resolve() or sha256_file(event_file) != call["event_sha256"]
                or call["request_file_sha256"] != imported["requests_sha256"]):
            raise ValueError("provider event receipt mismatch")
        if call["status"] != "complete":
            continue
        receipt = {"capture_manifest_sha256": sha256_file(capture / "manifest.json"),
                   "event_sha256": call["event_sha256"], "provider": call["provider"],
                   "requested_model": call["requested_model"], "thread_id": call["thread_id"]}
        for request_id in call["request_ids"]:
            if request_id in by_request:
                raise ValueError("provider capture repeats a request")
            by_request[request_id] = receipt
    return by_request


def bound_source(source):
    path = Path(source["rollouts"])
    digest = sha256_file(path)
    receipt = load_structured(Path(source["receipt"]))
    suite_path = Path(source["suite"])
    if suite_path.is_dir():
        suite_path /= "manifest.json"
    capture = {}
    if receipt.get("schema_version") == "rlt_gym.replay.v1":
        manifest = load_structured(path.parent / "manifest.json")
        if (receipt["passed"] is not True or receipt["rollouts_sha256"] != digest
                or manifest["suite_manifest_sha256"] != sha256_file(suite_path)
                or receipt["run_manifest_digest"] != manifest["manifest_digest"]
                or receipt["exact_trajectory_replays"] + receipt["receipt_only_aborted_episodes"]
                != receipt["episodes"]):
            raise ValueError("native export requires a bound successful replay")
        is_native = True
    else:
        if (receipt.get("schema_version") != "rlt_gym.teacher_import.v1" or receipt["rollouts_sha256"] != digest
                or receipt["suite_manifest_sha256"] != sha256_file(suite_path)):
            raise ValueError("teacher export requires a bound checked import")
        is_native = False
        if "capture" in source:
            capture = capture_receipts(Path(source["capture"]), receipt)
    return path, is_native, capture, digest


def build(spec, out):
    suites = {}
    roots = defaultdict(set)
    for value in spec["suites"]:
        _, tasks = load_suite(Path(value))
        suites[value] = {task["task_id"]: task for task in tasks}
        for task in tasks:
            roots[task["lineage_id"]].add(task["split"])
    eligible = {root for root, splits in roots.items() if splits == {"train"}}
    streams = {kind: {} for kind in KINDS}
    preferences = defaultdict(dict)
    source_tallies, held_out, failure_codes = [], Counter(), Counter()
    teacher_roots = set()

    def add(kind, meta, context, target, *, eligible=True):
        example_id = sha256_json([kind, meta["root"], context, target])
        body = {"id": example_id, **meta, "input": context, "target": target,
                "loss": {"boundary": "response_only", "field": "target", "eligible": eligible}}
        # Keep one representative with intact provenance. The source traces retain repetitions.
        priority = bool(meta.get("provider_capture"))
        old = streams[kind].get(example_id)
        if old is None or (priority and not old[0]):
            streams[kind][example_id] = (
                priority, canonical_json(body), meta["root"], meta["family"],
                len(canonical_json(context).encode("utf-8")),
                len(canonical_json(target).encode("utf-8")) if eligible else 0,
            )

    for source in spec["sources"]:
        path, native, capture, digest = bound_source(source)
        tasks = suites[source["suite"]]
        checks = 0
        tally = Counter()
        for episode in records(path):
            task = tasks[episode["task_id"]]
            if native:
                verify_episode(task, episode)
            else:
                _validate_episode(episode, task)
            if task["lineage_id"] not in eligible:
                held_out[task["split"] if task["split"] != "train" else "cross_suite_overlap"] += 1
                continue
            world = public_task(task)
            prior = []
            if not episode["steps"]:
                tally["missing_or_censored_episodes"] += 1
            for step in episode["steps"]:
                receipt = step["verification"]
                status = receipt["status"]
                tally[status] += 1
                checked = status in {"passed", "failed"}
                checks += int(checked)
                source_model = step.get("producer", episode["backend"])
                provider_capture = capture.get(step.get("request_id"))
                meta = {
                    "root": task["lineage_id"], "split": "train", "family": task["family"],
                    "task_digest": task["task_digest"], "generator": task["generator_id"],
                    "episode_id": episode["episode_id"], "episode_digest": episode["episode_digest"],
                    "source_rollouts_sha256": digest, "visit": step["visit"], "source_model": source_model,
                    "source_receipt": step.get("provenance"), "provider_capture": provider_capture,
                    "verification": receipt,
                }
                context = {"task": world, "history": _history(prior)}
                candidate = {"candidate": step.get("candidate")}
                verdict = {"passed": receipt["passed"], "checker_code": receipt["code"]}
                if status != "passed":
                    failure_codes[receipt["code"]] += 1
                if not step.get("context_only"):
                    add("outcomes", meta, {**context, **candidate}, verdict if checked else None, eligible=checked)
                    if checked:
                        transition_input = context
                        transition = {**candidate, "checker": verdict}
                        state = step.get("state_summaries")
                        if state is not None:
                            transition_input = {**context, "state_before": state_features(state["before"]),
                                                "schedule": episode["config"], "eta": step["eta"],
                                                "visit": step["visit"]}
                            transition["next_state"] = {
                                name: state_features(state[name])
                                for name in ("endpoint", "carried_answer", "feedback_update")
                            }
                            transition["selected_score"] = step["selected_score"]
                            transition["score_margin"] = step["score_margin"]
                        add("transitions", meta, transition_input, transition)
                        if task["family"] == "experiment_plan":
                            issues = plan_issues(world["input"], candidate["candidate"])
                            add("research", meta, {"exercise": "critique", **context, **candidate},
                                {**verdict, "issues": issues})
                    if receipt["passed"]:
                        add("sft", meta, context, candidate)
                        if any(s["verification"]["status"] == "failed" for s in prior):
                            add("repairs", meta, context, candidate)
                        if provider_capture:
                            teacher_roots.add(task["lineage_id"])
                if checked:
                    pair_key = sha256_json([task["lineage_id"], context])
                    options = preferences[pair_key]
                    option_key = receipt["candidate_digest"]
                    old_option = options.get(option_key)
                    if old_option is None or (provider_capture and not old_option[0].get("provider_capture")):
                        options[option_key] = (meta, context, candidate)
                prior.append(step)
        source_tallies.append({"rollouts_sha256": digest, "kind": "native" if native else "teacher",
                               "checks_on_eligible_roots": checks, "step_counts": dict(tally),
                               "receipt_sha256": sha256_file(Path(source["receipt"]))})
    for options in preferences.values():
        good = [row for row in options.values() if row[0]["verification"]["passed"]]
        bad = [row for row in options.values() if not row[0]["verification"]["passed"]]
        for winner, context, chosen in good:
            for loser, _, rejected in bad:
                meta = {**winner, "rejected_provenance": loser}
                add("preferences", meta, context, {"chosen": chosen, "rejected": rejected})
                if winner["family"] == "experiment_plan":
                    plan_options = sorted([rejected, chosen], key=sha256_json)
                    add("research", meta, {"exercise": "selection", **context, "options": plan_options},
                        {"selected": chosen, "rejected_checker_code": loser["verification"]["code"],
                         "rejected_issues": plan_issues(context["task"]["input"], rejected["candidate"])})

    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, rows in streams.items():
        path = out / f"{name}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for _, body, *_ in rows.values():
                stream.write(body + "\n")
        files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size, "unique_examples": len(rows),
                       "roots": len({row[2] for row in rows.values()}),
                       "families": dict(Counter(row[3] for row in rows.values())),
                       "prompt_payload_utf8_bytes": sum(row[4] for row in rows.values()),
                       "loss_payload_utf8_bytes": sum(row[5] for row in rows.values())}
    manifest = {"schema_version": "rlt_gym.aleph_corpus.v1", "split": "train", "streams": files,
                "sources": source_tallies, "teacher_verified_roots": len(teacher_roots),
                "roots_by_custody": dict(Counter("+".join(sorted(splits)) for splits in roots.values())),
                "excluded_rollouts": dict(held_out), "failure_codes": dict(failure_codes),
                "aleph_tokenizer": None, "aleph_loss_tokens": None,
                "sampling_plan": {"max_presentations_per_root_across_streams": 4,
                                  "accounting_lane_by_family": {"experiment_plan": "research_mechinterp",
                                                               "default": "games_procedural"}}}
    dump_structured(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result = build(load_structured(args.spec), args.out)
    print(canonical_json({"streams": result["streams"], "teacher_verified_roots": result["teacher_verified_roots"]}))


if __name__ == "__main__":
    main()
