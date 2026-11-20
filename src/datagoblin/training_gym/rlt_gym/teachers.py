"""Offline teacher requests, independently checked replies, and training exports.

Providers receive public task state and visible feedback. The evaluator retains
split custody and recomputes every answer receipt before training export.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datagoblin.serialization import canonical_json, dump_structured

ALLOWED_SPLITS = frozenset({"train", "calibration"})
CONTROL_KINDS = frozenset(
    {
        "frozen_random_reservoir",
        "fixture",
        "mock",
        "oracle",
        "reference",
        "reference_solver",
        "candidate_order",
    }
)
UNVERIFIED_STATUSES = frozenset({"parse_failure", "verifier_error", "unavailable", "timeout", "resource_abort"})
MAX_LINE_BYTES = 131072
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
RESPONSE_FIELDS = frozenset(
    {
        "request_id",
        "task_digest",
        "candidate",
        "public_steps",
        "checks_requested",
        "revision_summary",
        "confidence",
        "usage",
    }
)
RECEIPT_FIELDS = (
    "status",
    "passed",
    "code",
    "task_digest",
    "candidate_digest",
    "evidence_epoch",
    "verifier_id",
)
RESPONSE_SCHEMA = {
    "required": ["request_id", "task_digest", "candidate"],
    "optional": {
        "public_steps": "At most 12 brief, visible work artifacts or explanations; strings.",
        "checks_requested": "At most 12 descriptions of checks; strings. These are requests only.",
        "revision_summary": "Brief description of the visible revision, at most 2000 characters.",
        "confidence": "A finite number from 0 to 1, representing the teacher's own estimate.",
        "usage": "Optional object with input_tokens and output_tokens, nonnegative integers or null.",
    },
    "instructions": (
        "Return one JSON object per request, copying request_id and task_digest exactly. "
        "Use the candidate JSON shape in the task prompt. Supply public work artifacts only. "
        "The environment will check the candidate independently."
    ),
}


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _check_json_bounds(value: Any) -> None:
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if depth > 32 or count > 50000:
            raise ValueError("JSON structure exceeds response bounds")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
    canonical_json(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > MAX_LINE_BYTES:
                raise ValueError(f"oversized JSONL row {index}")
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("expected an object")
                _check_json_bounds(value)
            except (ValueError, TypeError, RecursionError) as error:
                raise ValueError(f"invalid JSONL row {index}: {type(error).__name__}") from error
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def _new_output(out: Path) -> None:
    if out.exists():
        raise ValueError(f"output already exists: {out}")
    out.mkdir(parents=True)


def _suite(path: Path) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    from .tasks import load_suite

    manifest_path = path / "manifest.json" if path.is_dir() else path
    manifest, tasks = load_suite(manifest_path)
    return manifest_path, manifest, {task["task_id"]: task for task in tasks}


def _public(task: dict[str, Any]) -> dict[str, Any]:
    return {key: task[key] for key in ("family", "prompt", "input", "evidence_epoch")}


def _history(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate": step.get("candidate"),
            "feedback": {
                "status": step["verification"]["status"],
                "code": step["verification"]["code"],
            },
            "public_steps": step.get("public_steps", []),
            "revision_summary": step.get("revision_summary", ""),
            **(
                {
                    "producer": {
                        key: step["producer"][key]
                        for key in ("kind", "provider", "model", "dynamics_kind")
                        if key in step["producer"]
                    }
                }
                if isinstance(step.get("producer"), dict)
                else {}
            ),
        }
        for step in steps
    ]


def _initial_episode(task: dict[str, Any], provider: str, model: str, max_visits: int) -> dict[str, Any]:
    identity = {"task": task["task_digest"], "provider": provider, "model": model, "depth": max_visits}
    return {
        "schema_version": "rlt_gym.rollout.v1",
        "episode_id": _digest(identity),
        **{
            key: task[key]
            for key in (
                "task_id",
                "task_digest",
                "family",
                "generator_id",
                "lineage_id",
                "split",
                "evidence_epoch",
            )
        },
        "backend": {
            "kind": "teacher",
            "provider": provider,
            "model": model,
            "dynamics_kind": "black_box_refinement",
            "hidden_state_access": False,
        },
        "config": {
            "planned_depth": max_visits,
            "mode": "teacher_refinement",
            "readout_site": "candidate",
            "seed": None,
        },
        "steps": [],
        "outcome": {
            "status": "missing_response",
            "passed": False,
            "realized_visits": 0,
            "stop_reason": "awaiting_teacher",
        },
        "bpd_mechanism_verified": False,
    }


def _request(task: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    value = {
        "schema_version": "rlt_gym.teacher_request.v1",
        "task_id": task["task_id"],
        "task_digest": task["task_digest"],
        "evidence_epoch": task["evidence_epoch"],
        "provider": episode["backend"]["provider"],
        "model": episode["backend"]["model"],
        "visit": len(episode["steps"]),
        "planned_depth": episode["config"]["planned_depth"],
        "context": {"task": _public(task), "history": _history(episode["steps"])},
        "response_schema": RESPONSE_SCHEMA,
    }
    value["request_id"] = _digest(value)
    return value


def _write_batch(
    suite_path: Path,
    out: Path,
    tasks: dict[str, dict[str, Any]],
    episodes: list[dict[str, Any]],
    *,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requests = [_request(tasks[episode["task_id"]], episode) for episode in episodes]
    if len({item["request_id"] for item in requests}) != len(requests):
        raise ValueError("duplicate teacher request contexts")
    _new_output(out)
    _write_jsonl(out / "requests.jsonl", requests)
    _write_jsonl(out / "contexts.jsonl", episodes)
    manifest = {
        "schema_version": "rlt_gym.teacher_batch.v1",
        "suite_manifest_sha256": _file_digest(suite_path),
        "requests_file": "requests.jsonl",
        "requests_sha256": _file_digest(out / "requests.jsonl"),
        "contexts_file": "contexts.jsonl",
        "contexts_sha256": _file_digest(out / "contexts.jsonl"),
        "request_count": len(requests),
        "split_counts": dict(sorted(Counter(item["split"] for item in episodes).items())),
        "network_calls": 0,
        "instructions": (
            "Give requests.jsonl to the named teacher; keep contexts.jsonl and this manifest with the evaluator."
        ),
    }
    if selection is not None:
        manifest["selection"] = selection
    dump_structured(out / "manifest.json", manifest)
    return {**manifest, "requests_path": str(out / "requests.jsonl"), "manifest_path": str(out / "manifest.json")}


def prepare_teacher_batch(
    suite_manifest: Path,
    out: Path,
    *,
    provider: str,
    model: str,
    split: str = "train",
    max_tasks: int | None = None,
    max_visits: int = 4,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Write an offline batch. Evidence splits cannot enter a teacher request."""
    if split not in ALLOWED_SPLITS:
        raise ValueError("teacher requests require train or calibration split")
    if provider not in {"codex", "opus", "local"}:
        raise ValueError("provider must be codex, opus, or local")
    if not isinstance(model, str) or not model.strip() or len(model) > 200:
        raise ValueError("supply a nonempty model identifier")
    _bounded_int(max_visits, "max_visits", 1, 64)
    if max_tasks is not None:
        _bounded_int(max_tasks, "max_tasks", 1, 100000)
    suite_path, _, tasks = _suite(Path(suite_manifest))
    if task_ids is not None:
        if len(set(task_ids)) != len(task_ids) or any(task_id not in tasks for task_id in task_ids):
            raise ValueError("task_ids must contain unique known task identifiers")
        if any(tasks[task_id]["split"] != split for task_id in task_ids):
            raise ValueError("requested tasks must belong to the selected development split")
        selected = [tasks[task_id] for task_id in task_ids]
    else:
        selected = [task for task in tasks.values() if task["split"] == split]
    selected = _balanced_tasks(selected)
    if max_tasks is not None:
        selected = selected[:max_tasks]
    episodes = [_initial_episode(task, provider, model, max_visits) for task in selected]
    return _write_batch(suite_path, Path(out), tasks, episodes)


def _balanced_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        groups[task["family"]].append(task)
    ordered = [sorted(groups[key], key=lambda task: task["task_id"]) for key in sorted(groups)]
    return [
        group[index]
        for index in range(max((len(group) for group in ordered), default=0))
        for group in ordered
        if index < len(group)
    ]


def prepare_teacher_repairs(
    suite_manifest: Path,
    rollouts_path: Path,
    out: Path,
    *,
    provider: str,
    model: str,
    max_tasks: int = 14,
    max_visits: int = 4,
) -> dict[str, Any]:
    """Give a teacher checked native failures as visible repair context.

    The total visit budget includes one native context visit. Every later reply
    records its actual producer; native control targets need explicit export opt-in.
    """
    if provider not in {"codex", "opus", "local"}:
        raise ValueError("provider must be codex, opus, or local")
    if not isinstance(model, str) or not model.strip() or len(model) > 200:
        raise ValueError("supply a nonempty model identifier")
    _bounded_int(max_tasks, "max_tasks", 1, 100000)
    _bounded_int(max_visits, "max_visits", 2, 64)
    suite_path, _, tasks = _suite(Path(suite_manifest))
    source_digest = _file_digest(Path(rollouts_path))
    episodes = []
    selected_tasks: set[str] = set()
    excluded: Counter[str] = Counter()
    for source in _read_jsonl(Path(rollouts_path)):
        task = tasks.get(source.get("task_id"))
        if task is None:
            raise ValueError("unknown source rollout task")
        _validate_episode(source, task)
        if task["split"] != "train":
            excluded["split:" + task["split"]] += 1
            continue
        backend = source["backend"]
        if (
            backend.get("kind") in {"teacher", "reference_solver", "reference", "oracle", "fixture", "mock"}
            or backend.get("dynamics_kind") not in {"continuous_recurrent", "discrete_search"}
            or backend.get("mock") is True
        ):
            excluded["source_backend"] += 1
            continue
        if source["outcome"]["status"] not in {"failed", "budget_exhausted"}:
            excluded["outcome:" + source["outcome"]["status"]] += 1
            continue
        if not source["steps"] or source["steps"][-1]["verification"]["status"] != "failed":
            excluded["no_checked_failure"] += 1
            continue
        if task["task_id"] in selected_tasks:
            excluded["duplicate_task"] += 1
            continue
        selected_tasks.add(task["task_id"])
        failed = source["steps"][-1]
        source_row_digest = _digest(source)
        origin = {
            "kind": "native_failure_repair",
            "source_rollouts_sha256": source_digest,
            "source_episode_digest": source.get("episode_digest", source_row_digest),
            "source_row_sha256": source_row_digest,
            "source_episode_id": source["episode_id"],
            "source_task_digest": task["task_digest"],
            "source_visit": failed["visit"],
        }
        episode = _initial_episode(task, provider, model, max_visits)
        episode["episode_id"] = _digest({"teacher_episode": episode["episode_id"], "origin": origin})
        episode["origin"] = origin
        episode["config"]["native_context_visits"] = 1
        episode["config"]["teacher_reply_budget"] = max_visits - 1
        episode["steps"] = [
            {
                "visit": 0,
                "candidate": failed["candidate"],
                "verification": failed["verification"],
                "producer": failed.get("producer", backend),
                "context_only": True,
                "provenance": origin,
                "public_steps": failed.get("public_steps", []),
                "revision_summary": failed.get("revision_summary", ""),
                "usage": {"input_tokens": None, "output_tokens": None},
            }
        ]
        episode["outcome"] = {
            "status": "failed",
            "passed": False,
            "realized_visits": 1,
            "stop_reason": "awaiting_teacher_repair",
        }
        _validate_episode(episode, task)
        episodes.append(episode)
    episodes = _balanced_tasks(episodes)[:max_tasks]
    return _write_batch(
        suite_path,
        Path(out),
        tasks,
        episodes,
        selection={
            "kind": "native_failure_repair",
            "source_rollouts_sha256": source_digest,
            "policy": "First completed, independently checked native failure per training task; balanced by family.",
            "excluded": dict(sorted(excluded.items())),
            "native_context_visits_per_episode": 1,
            "maximum_teacher_replies_per_episode": max_visits - 1,
        },
    )


def _validate_episode(episode: dict[str, Any], task: dict[str, Any]) -> None:
    from .tasks import verify

    if episode.get("schema_version") != "rlt_gym.rollout.v1":
        raise ValueError("unsupported rollout schema")
    for key in ("task_id", "task_digest", "family", "generator_id", "lineage_id", "split", "evidence_epoch"):
        if episode.get(key) != task[key]:
            raise ValueError(f"rollout {key} does not match the frozen task")
    if "episode_digest" in episode:
        unsigned = {key: value for key, value in episode.items() if key != "episode_digest"}
        if episode["episode_digest"] != _digest(unsigned):
            raise ValueError("rollout digest mismatch")
    backend = episode.get("backend", {})
    teacher = backend.get("kind") == "teacher"
    if teacher and (backend.get("dynamics_kind") != "black_box_refinement" or backend.get("hidden_state_access")):
        raise ValueError("teacher state access must describe observable black-box refinement")
    steps = episode.get("steps")
    if not isinstance(steps, list):
        raise ValueError("rollout steps must be a list")
    depth = _bounded_int(episode.get("config", {}).get("planned_depth"), "planned_depth", 1, 64)
    if len(steps) > depth:
        raise ValueError("rollout exceeds its planned depth")
    for index, step in enumerate(steps):
        if step.get("visit") != index:
            raise ValueError("rollout visits must be contiguous and zero-based")
        if teacher and any(
            key in step
            for key in ("hidden_states", "activations", "reasoning_trace", "chain_of_thought", "state_summaries")
        ):
            raise ValueError("teacher traces cannot assert hidden-state observations")
        receipt = step.get("verification", {})
        if receipt.get("status") in UNVERIFIED_STATUSES:
            if receipt.get("passed") is not False or not isinstance(receipt.get("code"), str):
                raise ValueError("malformed unverified step")
            if (
                receipt.get("task_digest") != task["task_digest"]
                or receipt.get("evidence_epoch") != task["evidence_epoch"]
            ):
                raise ValueError("unverified step task binding mismatch")
            if receipt["status"] == "parse_failure" and step.get("candidate") is not None:
                raise ValueError("malformed parse-failure candidate")
            continue
        expected = verify(task, step.get("candidate"))
        if any(receipt.get(key) != expected[key] for key in RECEIPT_FIELDS):
            raise ValueError("rollout verifier receipt does not replay")
    outcome = episode.get("outcome", {})
    status = outcome.get("status")
    allowed_statuses = {"passed", "failed", "budget_exhausted", "missing_response"} | UNVERIFIED_STATUSES
    if status not in allowed_statuses:
        raise ValueError("unsupported rollout outcome status")
    if teacher and any(step["verification"].get("passed") is True for step in steps[:-1]):
        raise ValueError("teacher rollout continued after verified stop")
    last_passed = bool(steps and steps[-1]["verification"].get("passed") is True)
    # Native terminal-only runs may regress after an earlier correct answer.
    # An aborted run stays censored even when its latest completed visit passed.
    success = last_passed if teacher or status in {"passed", "failed"} else False
    if outcome.get("passed") is not success or outcome.get("realized_visits") != len(steps):
        raise ValueError("rollout outcome is inconsistent with its attempts")
    if success != (status == "passed"):
        raise ValueError("verified success must have passed outcome")


def _load_batch(
    suite_manifest: Path,
    requests_path: Path,
) -> tuple[Path, dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    suite_path, _, tasks = _suite(suite_manifest)
    manifest = json.loads((requests_path.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "rlt_gym.teacher_batch.v1":
        raise ValueError("teacher batch manifest required")
    if manifest.get("suite_manifest_sha256") != _file_digest(suite_path):
        raise ValueError("teacher batch suite manifest changed")
    if requests_path.name != "requests.jsonl" or manifest.get("requests_sha256") != _file_digest(requests_path):
        raise ValueError("teacher requests changed")
    contexts_path = requests_path.parent / "contexts.jsonl"
    if manifest.get("contexts_sha256") != _file_digest(contexts_path):
        raise ValueError("teacher contexts changed")
    requests, episodes = _read_jsonl(requests_path), _read_jsonl(contexts_path)
    if len(requests) != len(episodes) or len(requests) != manifest.get("request_count"):
        raise ValueError("teacher batch count mismatch")
    if len({item["request_id"] for item in requests}) != len(requests):
        raise ValueError("duplicate teacher requests")
    for request, episode in zip(requests, episodes, strict=True):
        task = tasks.get(episode.get("task_id"))
        if task is None or task["split"] not in ALLOWED_SPLITS:
            raise ValueError("teacher task absent or in an evidence split")
        _validate_episode(episode, task)
        if episode["outcome"]["passed"] or len(episode["steps"]) >= episode["config"]["planned_depth"]:
            raise ValueError("teacher request follows a terminal episode")
        if request != _request(task, episode):
            raise ValueError("teacher request does not match its task and visible history")
    return suite_path, tasks, requests, episodes


def _validate_response(response: dict[str, Any], request: dict[str, Any]) -> None:
    if set(response) - RESPONSE_FIELDS:
        raise ValueError("unsupported_response_fields")
    if "candidate" not in response or response.get("task_digest") != request["task_digest"]:
        raise ValueError("missing_candidate_or_task_mismatch")
    canonical_json(response)
    for key in ("public_steps", "checks_requested"):
        values = response.get(key, [])
        if not isinstance(values, list) or len(values) > 12:
            raise ValueError(f"invalid_{key}")
        if any(not isinstance(value, str) or len(value) > 2000 for value in values):
            raise ValueError(f"invalid_{key}")
    revision = response.get("revision_summary", "")
    if not isinstance(revision, str) or len(revision) > 2000:
        raise ValueError("invalid_revision_summary")
    if "confidence" in response:
        confidence = response["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("invalid_confidence")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid_confidence")
    usage = response.get("usage", {})
    if not isinstance(usage, dict) or set(usage) - {"input_tokens", "output_tokens"}:
        raise ValueError("invalid_usage")
    for value in usage.values():
        if value is not None and (type(value) is not int or not 0 <= value <= 1000000000):
            raise ValueError("invalid_usage")


def _parse_responses(raw: bytes) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = []
    for index, line in enumerate(raw.splitlines(keepends=True), 1):
        if not line.strip():
            continue
        raw_digest = hashlib.sha256(line).hexdigest()
        value = None
        error_code = None
        try:
            if len(line) > MAX_LINE_BYTES:
                raise ValueError("oversized_response")
            value = json.loads(line.decode("utf-8"))
            if not isinstance(value, dict) or not isinstance(value.get("request_id"), str):
                raise ValueError("missing_request_id")
            _check_json_bounds(value)
        except (ValueError, TypeError, RecursionError) as error:
            error_code = type(error).__name__ + ": " + str(error)[:80]
        if isinstance(value, dict) and isinstance(value.get("request_id"), str):
            by_request[value["request_id"]].append(
                {
                    "line": index,
                    "raw_sha256": raw_digest,
                    "error_code": error_code,
                    "response": value if error_code is None else None,
                }
            )
        else:
            rejected.append({"line": index, "raw_sha256": raw_digest, "code": error_code})
    return dict(by_request), rejected


def import_teacher_responses(
    suite_manifest: Path,
    requests_path: Path,
    responses_path: Path,
    out: Path,
) -> dict[str, Any]:
    """Import one revision per request, preserving failures and missing replies."""
    from .tasks import verify

    suite_path, tasks, requests, episodes = _load_batch(Path(suite_manifest), Path(requests_path))
    with Path(responses_path).open("rb") as handle:
        raw_responses = handle.read(MAX_RESPONSE_BYTES + 1)
    if len(raw_responses) > MAX_RESPONSE_BYTES:
        raise ValueError("teacher response file exceeds the 64 MiB import bound")
    responses_digest = hashlib.sha256(raw_responses).hexdigest()
    request_batch_digest = _file_digest(Path(requests_path))
    responses, rejected = _parse_responses(raw_responses)
    known = {request["request_id"] for request in requests}
    for request_id in sorted(set(responses) - known):
        for reply in responses[request_id]:
            rejected.append({"line": reply["line"], "raw_sha256": reply["raw_sha256"], "code": "unknown_request"})
    accepted = 0
    imported_usage = []
    for request, episode in zip(requests, episodes, strict=True):
        episode.pop("episode_digest", None)
        task = tasks[episode["task_id"]]
        matches = responses.get(request["request_id"], [])
        code = None
        response = None
        if len(matches) > 1:
            code = "duplicate_response"
            for reply in matches:
                rejected.append({"line": reply["line"], "raw_sha256": reply["raw_sha256"], "code": code})
        elif matches:
            reply = matches[0]
            response = reply["response"]
            code = reply["error_code"]
            if code is None:
                try:
                    _validate_response(response, request)
                    receipt = verify(task, response["candidate"])
                except (ValueError, TypeError, OverflowError, RecursionError) as error:
                    code = str(error)[:100]
            if code is not None:
                rejected.append({"line": reply["line"], "raw_sha256": reply["raw_sha256"], "code": code})
        if matches:
            if code is not None:
                step = {
                    "visit": request["visit"],
                    "candidate": None,
                    "verification": {
                        "status": "parse_failure",
                        "passed": False,
                        "code": code,
                        "task_digest": task["task_digest"],
                        "evidence_epoch": task["evidence_epoch"],
                    },
                    "request_id": request["request_id"],
                    "usage": {"input_tokens": None, "output_tokens": None},
                }
            else:
                assert response is not None
                accepted += 1
                step = {
                    "visit": request["visit"],
                    "candidate": response["candidate"],
                    "verification": receipt,
                    "request_id": request["request_id"],
                    "public_steps": response.get("public_steps", []),
                    "checks_requested": response.get("checks_requested", []),
                    "revision_summary": response.get("revision_summary", ""),
                    "confidence": response.get("confidence"),
                    "usage": {key: response.get("usage", {}).get(key) for key in ("input_tokens", "output_tokens")},
                }
            step["producer"] = episode["backend"]
            step["provenance"] = {
                "request_batch_sha256": request_batch_digest,
                "responses_sha256": responses_digest,
                "raw_reply_sha256": [reply["raw_sha256"] for reply in matches],
                "provider_execution_attested": False,
            }
            episode["steps"].append(step)
            imported_usage.append(step["usage"])
        passed = bool(episode["steps"] and episode["steps"][-1]["verification"].get("passed"))
        status = "passed" if passed else "parse_failure" if code else "failed" if matches else "missing_response"
        stop = (
            "verified_stop"
            if passed
            else "budget_exhausted"
            if len(episode["steps"]) >= request["planned_depth"]
            else "revision_available"
            if matches
            else "awaiting_teacher"
        )
        episode["outcome"] = {
            "status": status,
            "passed": passed,
            "realized_visits": len(episode["steps"]),
            "stop_reason": stop,
        }
        episode["source"] = {
            "request_batch_sha256": request_batch_digest,
            "responses_sha256": responses_digest,
        }
        episode["episode_digest"] = _digest(episode)
        _validate_episode(episode, task)
    out = Path(out)
    _new_output(out)
    with (out / "responses.raw.jsonl").open("xb") as handle:
        handle.write(raw_responses)
    _write_jsonl(out / "rollouts.jsonl", episodes)
    _write_jsonl(out / "rejected.jsonl", rejected)
    manifest = {
        "schema_version": "rlt_gym.teacher_import.v1",
        "suite_manifest_sha256": _file_digest(suite_path),
        "requests_sha256": request_batch_digest,
        "responses_sha256": responses_digest,
        "raw_responses_file": "responses.raw.jsonl",
        "raw_responses_sha256": responses_digest,
        "raw_response_bytes": len(raw_responses),
        "raw_responses_training_eligible": False,
        "rollouts_sha256": _file_digest(out / "rollouts.jsonl"),
        "rejected_sha256": _file_digest(out / "rejected.jsonl"),
        "episodes": len(episodes),
        "accepted_responses": accepted,
        "missing_responses": sum(request["request_id"] not in responses for request in requests),
        "rejected_rows": len(rejected),
        "outcome_counts": dict(sorted(Counter(item["outcome"]["status"] for item in episodes).items())),
        "network_calls": 0,
        "teacher_attempts_observed": len(imported_usage),
        "usage": {
            name: {
                "reported_total": sum(item[name] for item in imported_usage if item[name] is not None),
                "unknown_attempts": sum(item[name] is None for item in imported_usage),
            }
            for name in ("input_tokens", "output_tokens")
        },
        "provenance": (
            "Provider and model identify operator-supplied replies; local import does not attest provider execution."
        ),
    }
    dump_structured(out / "manifest.json", manifest)
    return {
        **manifest,
        "rollouts_path": str(out / "rollouts.jsonl"),
        "rejected_path": str(out / "rejected.jsonl"),
        "raw_responses_path": str(out / "responses.raw.jsonl"),
        "manifest_path": str(out / "manifest.json"),
    }


def continue_teacher_batch(
    suite_manifest: Path,
    rollouts_path: Path,
    out: Path,
    *,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """Queue unfinished teacher episodes with bounded visible verifier feedback."""
    suite_path, _, tasks = _suite(Path(suite_manifest))
    if max_tasks is not None:
        _bounded_int(max_tasks, "max_tasks", 1, 100000)
    episodes = []
    for episode in _read_jsonl(Path(rollouts_path)):
        task = tasks.get(episode.get("task_id"))
        if task is None:
            raise ValueError("unknown rollout task")
        _validate_episode(episode, task)
        if episode["backend"].get("kind") != "teacher" or task["split"] not in ALLOWED_SPLITS:
            raise ValueError("continuation requires a development teacher episode")
        if not episode["outcome"]["passed"] and len(episode["steps"]) < episode["config"]["planned_depth"]:
            episodes.append(episode)
    episodes = _balanced_tasks(episodes)
    if max_tasks is not None:
        episodes = episodes[:max_tasks]
    return _write_batch(suite_path, Path(out), tasks, episodes)


def _training_target(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": step["candidate"],
        "public_steps": step.get("public_steps", []),
        "revision_summary": step.get("revision_summary", ""),
        "checks_requested": step.get("checks_requested", []),
    }


def export_dataset(
    suite_manifest: Path,
    rollouts_path: Path,
    out: Path,
    *,
    include_controls: bool = False,
) -> dict[str, Any]:
    """Recheck candidates and export train-only supervision with source attribution."""
    suite_path, _, tasks = _suite(Path(suite_manifest))
    streams: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in (
            "sft",
            "transitions",
            "outcomes",
            "repairs",
            "preferences",
        )
    }
    exclusions: Counter[str] = Counter()
    preference_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for episode in _read_jsonl(Path(rollouts_path)):
        task = tasks.get(episode.get("task_id"))
        if task is None:
            raise ValueError("unknown rollout task")
        _validate_episode(episode, task)
        episode_fingerprint = _digest(episode)
        if episode_fingerprint in seen:
            exclusions["duplicate_rollout"] += 1
            continue
        seen.add(episode_fingerprint)
        if task["split"] != "train":
            exclusions["split:" + task["split"]] += 1
            continue
        backend = episode["backend"]
        control = backend.get("kind") in CONTROL_KINDS or backend.get("mock") is True
        if control and not include_controls:
            exclusions["control:" + str(backend.get("kind"))] += 1
            continue
        provenance = {
            "task_id": task["task_id"],
            "task_digest": task["task_digest"],
            "family": task["family"],
            "generator_id": task["generator_id"],
            "lineage_id": task["lineage_id"],
            "split": "train",
            "episode_id": episode["episode_id"],
            "episode_digest": episode_fingerprint,
            "backend": backend,
            "dynamics_kind": backend.get("dynamics_kind"),
            "is_control": control,
            "source_provenance": task["provenance"],
            "teacher_explanations_are_claims": backend.get("kind") == "teacher",
            "origin": episode.get("origin"),
        }
        prior: list[dict[str, Any]] = []
        had_failure = False
        for step in episode["steps"]:
            producer = step.get("producer", backend)
            if not isinstance(producer, dict):
                raise ValueError("step producer must be an object")
            step_control = producer.get("kind") in CONTROL_KINDS or producer.get("mock") is True
            if step_control and not include_controls:
                exclusions["control_step:" + str(producer.get("kind"))] += 1
                had_failure = had_failure or not step["verification"].get("passed")
                prior.append(step)
                continue
            context = {"task": _public(task), "history": _history(prior)}
            if producer.get("dynamics_kind") == "continuous_recurrent" and not step.get("context_only"):
                context["recurrence"] = {
                    "config": episode["config"],
                    "visit": step["visit"],
                    "model_digest": producer.get("model_digest"),
                }
            receipt = step["verification"]
            row = {
                "schema_version": "rlt_gym.training_example.v1",
                **provenance,
                "backend": producer,
                "producer": producer,
                "episode_backend": backend,
                "is_control": step_control,
                "dynamics_kind": producer.get("dynamics_kind"),
                "teacher_explanations_are_claims": producer.get("kind") == "teacher",
                "context_only": step.get("context_only", False),
                "visit": step["visit"],
                "context": context,
                "context_digest": _digest(context),
                "target": _training_target(step) if receipt["status"] not in UNVERIFIED_STATUSES else None,
                "verification": receipt,
                "usage": step.get("usage", {}),
                "step_provenance": step.get("provenance", {}),
                "teacher_request_id": step.get("request_id"),
            }
            if producer.get("dynamics_kind") == "continuous_recurrent" and "state_summaries" in step:
                row["native_observation"] = {
                    "authority": "continuous_adapter_measurement",
                    **{
                        key: step[key]
                        for key in (
                            "state_summaries",
                            "eta",
                            "readout_site",
                            "intervention",
                            "block_calls",
                            "evidence_epoch",
                            "segment_id",
                            "state_placement",
                            "bpd_mechanism_verified",
                        )
                        if key in step
                    },
                }
            streams["transitions"].append({**row, "kind": "transition"})
            if receipt["status"] in {"passed", "failed"}:
                preference_groups[_digest({"task": task["task_digest"], "context": context})].append(row)
            if receipt.get("passed"):
                messages = [
                    {
                        "role": "system",
                        "content": "Solve the task and return candidate, public_steps, "
                        "revision_summary, and checks_requested as JSON. The environment verifies the candidate.",
                    },
                    {"role": "user", "content": canonical_json(context)},
                    {"role": "assistant", "content": canonical_json(row["target"])},
                ]
                streams["sft"].append({**row, "kind": "verified_sft", "messages": messages})
                if had_failure:
                    streams["repairs"].append(
                        {
                            **row,
                            "kind": "verified_repair",
                            "messages": messages,
                            "failed_visits": [s["visit"] for s in prior if not s["verification"].get("passed")],
                        }
                    )
            else:
                had_failure = True
            prior.append(step)
        outcome = episode["outcome"]
        streams["outcomes"].append(
            {
                "schema_version": "rlt_gym.training_example.v1",
                **provenance,
                "kind": "outcome",
                "prediction_point": "before_final_verifier" if episode["steps"] else "before_any_reply",
                "context": {
                    "task": _public(task),
                    "history": _history(episode["steps"][:-1]),
                    "candidate_to_evaluate": episode["steps"][-1].get("candidate") if episode["steps"] else None,
                },
                "outcome": outcome,
                "censored": outcome["status"]
                in {"missing_response", "timeout", "resource_abort", "parse_failure", "verifier_error", "unavailable"},
            }
        )
    for rows in preference_groups.values():
        unique = {row["verification"]["candidate_digest"]: row for row in rows}
        good = [row for row in unique.values() if row["verification"]["passed"]]
        bad = [row for row in unique.values() if not row["verification"]["passed"]]
        for winner in good:
            for loser in bad:
                streams["preferences"].append(
                    {
                        "schema_version": "rlt_gym.training_example.v1",
                        "kind": "verified_preference",
                        "task_id": winner["task_id"],
                        "task_digest": winner["task_digest"],
                        "split": "train",
                        "context": winner["context"],
                        "context_digest": winner["context_digest"],
                        "chosen": winner["target"],
                        "rejected": loser["target"],
                        "chosen_verification": winner["verification"],
                        "rejected_verification": loser["verification"],
                        "chosen_episode_digest": winner["episode_digest"],
                        "rejected_episode_digest": loser["episode_digest"],
                        "chosen_backend": winner["backend"],
                        "rejected_backend": loser["backend"],
                        "chosen_source_receipt": {
                            "task_provenance": winner["source_provenance"],
                            "step_provenance": winner["step_provenance"],
                            "request_id": winner["teacher_request_id"],
                        },
                        "rejected_source_receipt": {
                            "task_provenance": loser["source_provenance"],
                            "step_provenance": loser["step_provenance"],
                            "request_id": loser["teacher_request_id"],
                        },
                        "is_control": winner["is_control"] or loser["is_control"],
                        "comparison_basis": "Independent pass/fail on identical task input and visible history.",
                    }
                )
    out = Path(out)
    _new_output(out)
    for name, rows in streams.items():
        _write_jsonl(out / f"{name}.jsonl", rows)
    manifest = {
        "schema_version": "rlt_gym.dataset_export.v1",
        "suite_manifest_sha256": _file_digest(suite_path),
        "source_rollouts_sha256": _file_digest(Path(rollouts_path)),
        "split": "train",
        "include_controls": include_controls,
        "counts": {name: len(rows) for name, rows in streams.items()},
        "excluded": dict(sorted(exclusions.items())),
        "files": {name: {"path": f"{name}.jsonl", "sha256": _file_digest(out / f"{name}.jsonl")} for name in streams},
        "claim_boundary": "Verified task outputs and attributed public work; no teacher hidden-state measurements.",
    }
    dump_structured(out / "manifest.json", manifest)
    return {
        **manifest,
        "manifest_path": str(out / "manifest.json"),
        **{name + "_path": str(out / f"{name}.jsonl") for name in streams},
    }
