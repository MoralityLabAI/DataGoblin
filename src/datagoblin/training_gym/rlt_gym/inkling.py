"""Variable-loop teacher gym: local Inkling captures, verified distillation and student selection.

Loop passes belong to the teacher architecture. RLT visits belong to the student.
Only the existing environment verifier supplies answer correctness.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import canonical_json, dump_structured

from .runner import read_jsonl
from .tasks import load_suite, public_task, verify
from .teachers import _balanced_tasks, _validate_episode, import_teacher_responses, prepare_teacher_batch

MODEL = "thinkingmachines/TML-d24-varloop-research"
SCHEMA = "rlt_gym.inkling_sweep.v1"
MAX_REQUESTS = 1000
MAX_RESPONSE_BYTES = 262_144


def _suite_file(path: Path) -> Path:
    return path / "manifest.json" if path.is_dir() else path


def _new_dir(path: Path) -> None:
    if path.exists():
        raise ValueError(f"output already exists: {path}")
    path.mkdir(parents=True)


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


def _manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("expected an object manifest")
    return data


def _batch(stage_dir: Path, passes: int) -> tuple[dict[str, Any], dict[str, Any]]:
    stage = _manifest(stage_dir / "manifest.json")
    if stage.get("schema_version") != SCHEMA or passes not in stage["passes"]:
        raise ValueError("unknown pass count or invalid sweep")
    name = f"p{passes:02d}"
    record = stage["batches"][name]
    folder = stage_dir / name
    if (sha256_file(folder / "requests.jsonl") != record["requests_sha256"]
            or sha256_file(folder / "contexts.jsonl") != record["contexts_sha256"]
            or _manifest(folder / "manifest.json")["suite_manifest_sha256"] != stage["suite_manifest_sha256"]):
        raise ValueError("staged batch content changed")
    return stage, record


def audit_stage(stage_dir: Path) -> dict[str, Any]:
    """Verify that only the request ID and teacher pass identity vary."""
    root = _manifest(stage_dir / "manifest.json")
    if not root["passes"] or len(root["passes"]) != len(set(root["passes"])):
        raise ValueError("invalid staged pass matrix")
    baseline = None
    for passes in root["passes"]:
        _, record = _batch(stage_dir, passes)
        requests = read_jsonl(stage_dir / f"p{passes:02d}" / "requests.jsonl")
        if ([request["task_id"] for request in requests] != root["task_ids"]
                or len(requests) != record["requests"] or not requests
                or len({request["request_id"] for request in requests}) != len(requests)
                or any(request["model"] != f"{root['model']}#passes={passes}" for request in requests)):
            raise ValueError("unexpected staged request identity or coverage")
        normalized = [{k: v for k, v in request.items() if k not in {"model", "request_id"}}
                      for request in requests]
        if baseline is not None and baseline != normalized:
            raise ValueError("task context changes across teacher passes")
        baseline = normalized
    return root


def stage(
    suite: Path, out: Path, *, passes: tuple[int, ...] = (1, 2, 3, 5, 7, 10),
    split: str = "train", max_tasks: int = 28, model: str = MODEL,
) -> dict[str, Any]:
    """Freeze the same public tasks for every architectural loop count, with no network."""
    if split not in {"train", "calibration"}:
        raise ValueError("teacher splits must be train or calibration")
    if not passes or len(set(passes)) != len(passes) or any(type(p) is not int or not 1 <= p <= 16 for p in passes):
        raise ValueError("passes must be distinct integers from 1 to 16")
    if type(max_tasks) is not int or not 1 <= max_tasks <= MAX_REQUESTS:
        raise ValueError("max_tasks is out of bounds")
    if not isinstance(model, str) or not 1 <= len(model) <= 200:
        raise ValueError("model must be a nonempty identifier")
    suite_file = _suite_file(suite)
    _, all_tasks = load_suite(suite_file)
    tasks = _balanced_tasks([task for task in all_tasks if task["split"] == split])[:max_tasks]
    if not tasks:
        raise ValueError("no tasks in requested split")
    ids = [task["task_id"] for task in tasks]
    _new_dir(out)
    batches = {}
    for passes_count in passes:
        name = f"p{passes_count:02d}"
        # One generated answer at a fixed teacher architecture depth, not N revisions.
        prepare_teacher_batch(
            suite_file, out / name, provider="local",
            model=f"{model}#passes={passes_count}", split=split,
            task_ids=ids, max_visits=1,
        )
        batches[name] = {
            "loop_num_passes": passes_count,
            "requests_sha256": sha256_file(out / name / "requests.jsonl"),
            "contexts_sha256": sha256_file(out / name / "contexts.jsonl"),
            "requests": len(ids), "extrapolative": passes_count > 5,
        }
    manifest = {
        "schema_version": SCHEMA, "suite_manifest_sha256": sha256_file(suite_file),
        "model": model, "split": split, "passes": list(passes), "task_ids": ids,
        "batches": batches, "network_calls": 0, "research_only": True,
        "note": "Loop counts 7 and 10 extrapolate beyond the reported 1-5 midtraining range.",
    }
    dump_structured(out / "manifest.json", manifest)
    audit_stage(out)
    return {"stage": str(out), "tasks": len(ids), "passes": list(passes), "network_calls": 0}


def _endpoint(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("capture permits local loopback HTTP only")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base URL cannot contain credentials, a query, or a fragment")
    if parsed.path.rstrip("/") not in {"", "/v1"}:
        raise ValueError("base URL path must be /v1")
    return urllib.parse.urlunsplit(("http", parsed.netloc, "/v1/chat/completions", "", ""))


def _decode_candidate(content: str) -> tuple[bool, Any, list[str]]:
    if len(content.encode("utf-8")) > MAX_RESPONSE_BYTES:
        return False, None, []
    text = content.strip()
    fence = chr(96) * 3
    if text.startswith(fence) and text.endswith(fence):
        text = text.split("\n", 1)[-1].rsplit(fence, 1)[0].strip()
    try:
        data = json.loads(text)
        canonical_json(data)
    except (TypeError, ValueError, RecursionError):
        return False, None, []
    if not isinstance(data, dict):
        return False, None, []
    if "candidate" not in data:
        return True, data, []
    steps = data.get("public_steps", [])
    if not isinstance(steps, list) or len(steps) > 12:
        steps = []
    steps = [item for item in steps if isinstance(item, str) and len(item) <= 2000]
    return True, data["candidate"], steps


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _usage(data: dict[str, Any]) -> dict[str, int | None]:
    return {
        target: data[source] if type(data.get(source)) is int and data[source] >= 0 else None
        for source, target in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens"))
    }


def capture(
    stage_dir: Path, out: Path, *, passes: int, server_loop_passes: int,
    base_url: str = "http://127.0.0.1:8000/v1", served_model: str = MODEL,
    share_cache: bool = False, max_requests: int = 28, max_tokens: int = 768,
    timeout_seconds: float = 90, max_wall_seconds: float = 900,
    dry_run: bool = False, api_key_env: str = "", runtime: str = "vllm",
    weights_revision: str | None = None,
) -> dict[str, Any]:
    """Call an operator-started local vLLM server, one architectural depth per process.

    The OpenAI API does not authenticate the server's loop count; the receipt
    preserves that distinction. No hidden states or private model thoughts are asserted.
    """
    root = audit_stage(stage_dir)
    _, batch = _batch(stage_dir, passes)
    if server_loop_passes != passes:
        raise ValueError("operator-declared server pass count differs from staged passes")
    if type(max_requests) is not int or not 1 <= max_requests <= MAX_REQUESTS:
        raise ValueError("max_requests is out of bounds")
    if type(max_tokens) is not int or not 16 <= max_tokens <= 4096:
        raise ValueError("max_tokens is out of bounds")
    if any(not math.isfinite(x) or x <= 0 for x in (timeout_seconds, max_wall_seconds)):
        raise ValueError("time budgets must be finite and positive")
    if served_model != root["model"]:
        raise ValueError("served model must match staged identity")
    if runtime not in {"vllm", "llama_cpp"} or (
        weights_revision is not None and not 1 <= len(weights_revision) <= 200
    ):
        raise ValueError("invalid runtime or weight revision")
    if runtime == "llama_cpp" and (passes > 6 or share_cache):
        raise ValueError("released llama.cpp GGUF supports 1-6 passes with separate per-pass caches")
    endpoint = _endpoint(base_url)
    requests = read_jsonl(stage_dir / f"p{passes:02d}" / "requests.jsonl")[:max_requests]
    plan = {
        "schema_version": "rlt_gym.inkling_capture.v2", "passes": passes,
        "requests_sha256": batch["requests_sha256"], "requests_planned": len(requests),
        "server_loop_passes_operator_declared": server_loop_passes,
        "share_cache_operator_declared": share_cache, "server_configuration_verified": False,
        "model_requested": served_model, "endpoint": endpoint,
        "runtime_operator_declared": runtime, "weights_revision_operator_declared": weights_revision,
        "temperature": 0,
        "max_tokens": max_tokens, "timeout_seconds": timeout_seconds,
        "max_wall_seconds": max_wall_seconds, "api_key_env_used": bool(api_key_env),
    }
    if dry_run:
        return {**plan, "network_calls": 0, "dry_run": True}
    headers = {"Content-Type": "application/json"}
    if api_key_env:
        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError("API key environment variable not set")
        headers["Authorization"] = "Bearer " + key
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    _new_dir(out)
    replies: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    start = time.monotonic()
    deadline = start + max_wall_seconds
    for request in requests:
        if time.monotonic() >= deadline:
            break
        prompt = {
            "task": request["context"]["task"], "history": request["context"]["history"],
            "instructions": (
                "Solve the task and output a JSON object with candidate holding the exact answer "
                "shape, plus optional public_steps with brief checkable explanations. No Markdown."
            ),
        }
        body = {
            "model": served_model,
            "messages": [{"role": "user", "content": canonical_json(prompt)}],
            "temperature": 0, "max_tokens": max_tokens,
        }
        row: dict[str, Any] = {"request_id": request["request_id"], "task_digest": request["task_digest"]}
        observation = dict(row)
        began = time.monotonic()
        try:
            payload = canonical_json(body).encode("utf-8")
            http = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
            with opener.open(http, timeout=min(timeout_seconds, max(0.1, deadline - began))) as response:
                output = response.read(MAX_RESPONSE_BYTES + 1)
            if len(output) > MAX_RESPONSE_BYTES:
                raise ValueError("oversized model response")
            data = json.loads(output)
            observation.update(returned_model=data.get("model"), usage=_usage(data.get("usage") or {}))
            choice = data["choices"][0]
            observation["finish_reason"] = choice.get("finish_reason")
            text = choice["message"].get("content")
            if text is None:
                text = ""
            if not isinstance(text, str):
                raise ValueError("model returned non-text content")
            parsed, candidate, steps = _decode_candidate(text)
            observation.update(
                content=text, parsed=parsed,
            )
            if observation["returned_model"] != served_model:
                observation["error"] = "model_identity_mismatch"
            else:
                if parsed:
                    row.update(candidate=candidate, public_steps=steps)
                row["usage"] = observation["usage"]
                replies.append(row)
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError, RecursionError) as exc:
            observation["error"] = type(exc).__name__
        observation["latency_seconds"] = round(time.monotonic() - began, 6)
        observations.append(observation)
    _jsonl(out / "responses.jsonl", replies)
    _jsonl(out / "observations.jsonl", observations)
    result = {
        **plan, "source_kind": "local_operator_declared",
        "network_calls": len(observations), "responses": len(replies),
        "parse_failures": sum(r.get("parsed") is False for r in observations),
        "errors": dict(sorted(Counter(r["error"] for r in observations if "error" in r).items())),
        "responses_sha256": sha256_file(out / "responses.jsonl"),
        "observations_sha256": sha256_file(out / "observations.jsonl"),
        "elapsed_seconds": round(time.monotonic() - start, 6),
        "all_requests_attempted": len(observations) == len(requests),
    }
    dump_structured(out / "manifest.json", result)
    return {"capture": str(out), "responses": len(replies), "network_calls": len(observations)}


def _audit_observations(stage: Path, capture_dir: Path, manifest: dict) -> dict:
    path = capture_dir / "observations.jsonl"
    if sha256_file(path) != manifest.get("observations_sha256"):
        raise ValueError("capture observation digest mismatch")
    requests = read_jsonl(stage / "requests.jsonl")
    rows = read_jsonl(path)
    if not len(rows) == manifest.get("network_calls") <= manifest.get("requests_planned", -1) <= len(requests):
        raise ValueError("invalid capture request accounting")
    observations, expected = {}, []
    for request, row in zip(requests, rows, strict=False):
        identity = {key: request[key] for key in ("request_id", "task_digest")}
        if any(row[key] != value for key, value in identity.items()):
            raise ValueError("capture observations do not follow staged requests")
        observations[request["task_id"]] = row
        if "error" in row:
            continue
        if row.get("returned_model") != manifest["model_requested"]:
            raise ValueError("observed model identity mismatch")
        parsed, candidate, steps = _decode_candidate(row["content"])
        if parsed is not row.get("parsed"):
            raise ValueError("capture parse mismatch")
        reply = {**identity, "usage": row["usage"]}
        if parsed:
            reply.update(candidate=candidate, public_steps=steps)
        expected.append(reply)
    if len(expected) != manifest.get("responses") or expected != read_jsonl(capture_dir / "responses.jsonl"):
        raise ValueError("capture replies do not match recorded observations")
    return observations


def _paired_depths(passes: list[int], matrix: list[dict]) -> list[dict]:
    checked = defaultdict(dict)
    for row in matrix:
        if row["status"] in {"passed", "failed"}:
            checked[row["passes"]][row["task_id"]] = row["passed"]
    results = []
    for lo, hi in combinations(sorted(passes), 2):
        shared = sorted(checked[lo].keys() & checked[hi].keys())
        results.append({
            "lower_passes": lo, "higher_passes": hi, "paired_verified_tasks": len(shared),
            "higher_only": sum(checked[hi][key] and not checked[lo][key] for key in shared),
            "lower_only": sum(checked[lo][key] and not checked[hi][key] for key in shared),
        })
    return results


def _aleph_rows(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        if not row["student_training_eligible"] or row["fixture"] or row["split"] != "train":
            continue
        kind = {"verified_sft": "sft", "verified_repair": "repairs",
                "pre_verdict_outcome": "outcomes"}[row["kind"]]
        root, context, target = row["lineage_id"], row["input"], row["target"]
        result.append({
            "id": sha256_json([kind, root, context, target]), "kind": kind,
            "root": root, "family": row["family"], "split": row["split"],
            "task_digest": row["task_digest"], "input": context, "target": target,
            "loss": {"boundary": "response_only", "field": "target", "eligible": True},
            "source_model": {"provider": "local", "model": row["teacher_model"]},
            "provider_execution_attested": False,
            "episode_digest": row["source_episode_digest"], "verification": row["verifier"],
            **{key: row[key] for key in ("failed_episode_digest", "failed_verification") if key in row},
        })
    return result


def score(
    stage_dir: Path, captures: Path, suite: Path, out: Path, *, include_fixtures: bool = False,
) -> dict[str, Any]:
    """Reverify teacher responses and select shallowest successful train-only distillation."""
    root = audit_stage(stage_dir)
    suite_file = _suite_file(suite)
    if sha256_file(suite_file) != root["suite_manifest_sha256"]:
        raise ValueError("suite/stage digest mismatch")
    _, suite_tasks = load_suite(suite_file)
    tasks = {task["task_id"]: task for task in suite_tasks}
    grouped: dict[str, list[tuple[int, dict[str, Any], bool]]] = defaultdict(list)
    matrix, per_pass = [], []
    arms = []
    for p in root["passes"]:
        _, batch = _batch(stage_dir, p)
        label = f"p{p:02d}"
        capture_dir = captures / label
        manifest = _manifest(capture_dir / "manifest.json")
        fixture = manifest.get("source_kind") == "synthetic_fixture"
        if (
            manifest.get("schema_version") != f"rlt_gym.inkling_capture.v{1 if fixture else 2}"
            or manifest.get("passes") != p
            or manifest.get("requests_sha256") != batch["requests_sha256"]
            or manifest.get("responses_sha256") != sha256_file(capture_dir / "responses.jsonl")
        ):
            raise ValueError(f"capture {label} does not match stage")
        if fixture and not include_fixtures:
            raise ValueError("fixture capture requires explicit include_fixtures")
        if not fixture and manifest.get("source_kind") != "local_operator_declared":
            raise ValueError("unknown capture provenance")
        if not fixture:
            if (manifest.get("server_loop_passes_operator_declared") != p
                    or manifest.get("model_requested") != root["model"]
                    or type(manifest.get("share_cache_operator_declared")) is not bool
                    or manifest.get("server_configuration_verified") is not False
                    or manifest.get("runtime_operator_declared") not in {"vllm", "llama_cpp"}
                    or manifest.get("temperature") != 0):
                raise ValueError("capture server model/cache/loop declaration mismatch")
            if manifest["runtime_operator_declared"] == "llama_cpp" and (
                p > 6 or manifest["share_cache_operator_declared"]
            ):
                raise ValueError("unsupported llama.cpp loop/cache configuration")
        arms.append((p, label, capture_dir, fixture, manifest))
    control_keys = (
        "source_kind", "share_cache_operator_declared", "runtime_operator_declared",
        "weights_revision_operator_declared", "temperature", "max_tokens", "requests_planned",
    )
    controls = {tuple(arm[4].get(key) for key in control_keys) for arm in arms}
    if len(controls) != 1:
        raise ValueError("teacher arms change capture provenance or generation controls")
    _new_dir(out)
    for p, label, capture_dir, fixture, manifest in arms:
        observations = {} if fixture else _audit_observations(stage_dir / label, capture_dir, manifest)
        import_teacher_responses(
            suite_file, stage_dir / label / "requests.jsonl",
            capture_dir / "responses.jsonl", out / label,
        )
        episodes = read_jsonl(out / label / "rollouts.jsonl")
        if (len(episodes) != len(root["task_ids"])
                or {ep["task_id"] for ep in episodes} != set(root["task_ids"])):
            raise ValueError("incomplete depth matrix")
        counts: Counter[str] = Counter()
        known_tokens = 0
        reported_tokens = 0
        for ep in episodes:
            task = tasks[ep["task_id"]]
            _validate_episode(ep, task)
            if task["split"] != root["split"] or ep["backend"]["model"] != f"{root['model']}#passes={p}":
                raise ValueError("teacher model/pass identity mismatch")
            passed = ep["outcome"]["passed"]
            status = ep["outcome"]["status"]
            counts[status] += 1
            step = ep["steps"][0] if ep["steps"] else None
            observation = observations.get(task["task_id"], {})
            usage = (step or {}).get("usage", {}) if fixture else observation.get("usage", {})
            tokens = None
            if type(usage.get("input_tokens")) is int and type(usage.get("output_tokens")) is int:
                tokens = usage["input_tokens"] + usage["output_tokens"]
                known_tokens += 1
                reported_tokens += tokens
            grouped[task["task_id"]].append((p, ep, fixture))
            matrix.append({
                "task_id": task["task_id"], "task_digest": task["task_digest"],
                "family": task["family"], "passes": p, "passed": passed,
                "status": status, "fixture": fixture,
                "lineage_id": task["lineage_id"], "split": task["split"],
                "request_attempted": bool(observation) if not fixture else bool(step),
                "verifier_calls": int(status in {"passed", "failed"}),
                "latency_seconds": observation.get("latency_seconds"),
                "returned_model": observation.get("returned_model"),
                "finish_reason": observation.get("finish_reason"),
                "reported_tokens": tokens,
                "layer_visit_token_proxy": (12 + 12 * p) * tokens if tokens is not None else None,
                "episode_digest": ep["episode_digest"],
            })
        definitive = counts["passed"] + counts["failed"]
        per_pass.append({
            "passes": p, "verified_correct": counts["passed"], "tasks": len(episodes),
            "attempted": sum(counts.values()) - counts["missing_response"] if fixture else len(observations),
            "verified_outcomes": definitive, "censored": len(episodes) - definitive,
            "accuracy": counts["passed"] / len(episodes),
            "accuracy_per_staged_task": counts["passed"] / len(episodes),
            "accuracy_on_verified_outcomes": counts["passed"] / definitive if definitive else None,
            "statuses": dict(counts), "reported_tokens": reported_tokens,
            "capture_manifest_sha256": sha256_file(capture_dir / "manifest.json"),
            "elapsed_seconds": manifest.get("elapsed_seconds"),
            "usage_known": known_tokens, "extrapolative": p > 5,
            "server_configuration_verified": False,
        })
    sft, preferences, curriculum, outcomes, repairs = [], [], [], [], []
    if {name for name in grouped} != set(root["task_ids"]):
        raise ValueError("incomplete task set")
    train_only = {task["lineage_id"] for task in suite_tasks if task["split"] == "train"} - {
        task["lineage_id"] for task in suite_tasks if task["split"] != "train"
    }
    seen_roots = set()
    for task_id in root["task_ids"]:
        task = tasks[task_id]
        attempts = sorted(grouped[task_id], key=lambda x: x[0])
        good = [(p, ep, fixture) for p, ep, fixture in attempts if ep["outcome"]["passed"]]
        bad = [(p, ep, fixture) for p, ep, fixture in attempts if ep["outcome"]["status"] == "failed"]
        lineage = task["lineage_id"]
        if task["split"] != "train" or lineage not in train_only or lineage in seen_roots:
            continue
        seen_roots.add(lineage)
        provenance = {"task_id": task_id, "task_digest": task["task_digest"],
                      "lineage_id": lineage, "split": "train", "family": task["family"]}
        # Keep one receipt per distinct candidate for the same lineage.
        seen_candidates = set()
        for d, ep, fixture in sorted([*good, *bad], key=lambda attempt: attempt[0]):
            candidate = ep["steps"][-1]["candidate"]
            key = canonical_json(candidate)
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            outcomes.append({
                **provenance, "kind": "pre_verdict_outcome",
                "input": {"task": public_task(task), "candidate": candidate},
                "target": {"passed": ep["outcome"]["passed"]}, "teacher_loop_passes": d,
                "teacher_model": ep["backend"]["model"],
                "verifier": ep["steps"][-1]["verification"],
                "source_episode_digest": ep["episode_digest"],
                "fixture": fixture, "student_training_eligible": not fixture,
            })
        curriculum.append({
            **provenance,
            "first_verified_pass": good[0][0] if good else None,
            "failed_passes": [p for p, _, _ in bad],
            "priority": "high" if not good or good[0][0] > 3 else "regular",
            "all_passes_unsolved": len(bad) == len(attempts),
            "censored_passes": sorted(set(root["passes"]) - {d for d, _, _ in [*good, *bad]}),
            "fixture": attempts[0][2],
        })
        if not good:
            continue
        p, episode, fixture = good[0]
        step = episode["steps"][-1]
        sft.append({
            **provenance,
            "schema_version": "rlt_gym.inkling_distill.v1", "kind": "verified_sft",
            "input": {"task": public_task(task)},
            "target": {"candidate": step["candidate"]},
            "teacher_public_steps_unverified": step.get("public_steps", []),
            "teacher_model": episode["backend"]["model"], "teacher_loop_passes": p,
            "verifier": step["verification"], "source_episode_digest": episode["episode_digest"],
            "fixture": fixture, "student_training_eligible": not fixture,
            "source_configuration_attested": False, "claimed_hidden_state_access": False,
        })
        lower_failures = [(d, loser, f) for d, loser, f in bad if d < p]
        if lower_failures:
            d, loser, rejected_fixture = lower_failures[-1]
            pair = {
                **provenance,
                "schema_version": "rlt_gym.inkling_distill.v1",
                "kind": "matched_task_verified_preference",
                "input": {"task": public_task(task)},
                "chosen": {"candidate": step["candidate"]},
                "rejected": {"candidate": loser["steps"][-1]["candidate"]},
                "chosen_passes": p, "rejected_passes": d,
                "chosen_verification": step["verification"],
                "rejected_verification": loser["steps"][-1]["verification"],
                "fixture": fixture or rejected_fixture,
                "student_training_eligible": not (fixture or rejected_fixture),
                "chosen_episode_digest": episode["episode_digest"],
                "rejected_episode_digest": loser["episode_digest"],
                "chosen_teacher_model": episode["backend"]["model"],
                "rejected_teacher_model": loser["backend"]["model"],
            }
            preferences.append(pair)
            repairs.append({
                **provenance, "kind": "verified_repair",
                "input": {"task": public_task(task), "failed_candidate": pair["rejected"]["candidate"]},
                "target": pair["chosen"], "fixture": pair["fixture"],
                "student_training_eligible": pair["student_training_eligible"],
                "verifier": pair["chosen_verification"],
                "failed_verification": pair["rejected_verification"],
                "source_episode_digest": pair["chosen_episode_digest"],
                "failed_episode_digest": pair["rejected_episode_digest"],
                "teacher_model": pair["chosen_teacher_model"],
            })
    streams = {"depth_matrix": matrix, "sft": sft, "preferences": preferences,
               "curriculum": curriculum, "outcomes": outcomes, "repairs": repairs,
               "aleph": _aleph_rows([*sft, *outcomes, *repairs])}
    for name, rows in streams.items():
        _jsonl(out / f"{name}.jsonl", rows)
    report = {
        "schema_version": "rlt_gym.inkling_score.v1", "split": root["split"],
        "stage_manifest_sha256": sha256_file(stage_dir / "manifest.json"),
        "suite_manifest_sha256": sha256_file(suite_file), "per_pass": per_pass,
        "capture_controls": dict(zip(control_keys, next(iter(controls)), strict=True)),
        "training_rows": len(sft), "preference_pairs": len(preferences),
        "stream_rows": {name: len(rows) for name, rows in streams.items()},
        "aleph_unique_roots": len({row["root"] for row in streams["aleph"]}),
        "aleph_target_utf8_bytes": sum(len(canonical_json(row["target"]).encode()) for row in streams["aleph"]),
        "paired_depths": _paired_depths(root["passes"], matrix),
        "student_training_eligible_rows": sum(row["student_training_eligible"] for row in sft),
        "files": {name: sha256_file(out / f"{name}.jsonl") for name in streams},
        "claim_boundary": (
            "Independent verifier checks correctness. Model and loop settings are operator-declared. "
            "Token cost is reported usage; layer-visit cost is an analytical proxy, not measured FLOPs."
        ),
    }
    dump_structured(out / "manifest.json", report)
    return {"scored": str(out), "per_pass": per_pass, "training_rows": len(sft),
            "preference_pairs": len(preferences)}


def hillclimb(suite: Path, predictions: Path, out: Path) -> dict[str, Any]:
    """Rank complete small-model predictions on calibration, never on reserved test splits."""
    suite_file = _suite_file(suite)
    _, tasks = load_suite(suite_file)
    calibration = {t["task_id"]: t for t in tasks if t["split"] == "calibration"}
    scores: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(predictions):
        task_id, model = row.get("task_id"), row.get("model_id")
        if not isinstance(model, str) or not 1 <= len(model) <= 200:
            raise ValueError("invalid model_id")
        if task_id not in calibration or row.get("task_digest") != calibration[task_id]["task_digest"]:
            raise ValueError("hillclimb accepts only calibration predictions")
        if task_id in scores[model]:
            raise ValueError("duplicate candidate prediction")
        visits = row.get("visits")
        if type(visits) is not int or not 1 <= visits <= 128:
            raise ValueError("visits must be in [1,128]")
        latency = row.get("latency_ms")
        if latency is not None and (type(latency) not in {int, float} or not math.isfinite(latency)
                                    or latency < 0):
            raise ValueError("invalid latency_ms")
        scores[model][task_id] = {
            "passed": verify(calibration[task_id], row.get("candidate"))["passed"],
            "visits": visits, "latency_ms": latency,
        }
    if not scores:
        raise ValueError("predictions are empty")
    ranked = []
    for model, predictions_by_task in scores.items():
        if set(predictions_by_task) != set(calibration):
            raise ValueError("each student must cover the identical frozen calibration tasks")
        values = list(predictions_by_task.values())
        correct = sum(row["passed"] for row in values)
        times = [row["latency_ms"] for row in values if row["latency_ms"] is not None]
        ranked.append({
            "model_id": model, "verified_correct": correct, "tasks": len(values),
            "accuracy": correct / len(values),
            "mean_visits": sum(row["visits"] for row in values) / len(values),
            "mean_latency_ms_reported": sum(times) / len(times) if len(times) == len(values) else None,
        })
    ranked.sort(key=lambda row: (-row["verified_correct"], row["mean_visits"], row["model_id"]))
    _new_dir(out)
    report = {
        "schema_version": "rlt_gym.inkling_hillclimb.v1",
        "suite_manifest_sha256": sha256_file(suite_file),
        "predictions_sha256": sha256_file(predictions),
        "selection_split": "calibration", "selection_only": True,
        "requires_fresh_confirmation": ["ordinary_test", "ood_test"],
        "ranked_candidates": ranked, "winner": ranked[0]["model_id"],
    }
    dump_structured(out / "rank.json", report)
    return {"winner": ranked[0]["model_id"], "candidates": len(ranked), "rank": str(out / "rank.json")}


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m datagoblin.training_gym.rlt_gym.inkling")
    commands = root.add_subparsers(dest="command", required=True)
    s = commands.add_parser("stage")
    s.add_argument("--suite", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--passes", type=int, nargs="+", default=[1, 2, 3, 5, 7, 10])
    s.add_argument("--split", choices=("train", "calibration"), default="train")
    s.add_argument("--max-tasks", type=int, default=28)
    s.add_argument("--model", default=MODEL)
    c = commands.add_parser("capture")
    c.add_argument("--stage", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--passes", type=int, required=True)
    c.add_argument("--server-loop-passes", type=int, required=True)
    c.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    c.add_argument("--served-model", default=MODEL)
    c.add_argument("--share-cache", action="store_true")
    c.add_argument("--max-requests", type=int, default=28)
    c.add_argument("--max-tokens", type=int, default=768)
    c.add_argument("--timeout-seconds", type=float, default=90)
    c.add_argument("--max-wall-seconds", type=float, default=900)
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--api-key-env", default="")
    c.add_argument("--runtime", choices=("vllm", "llama_cpp"), default="vllm")
    c.add_argument("--weights-revision")
    r = commands.add_parser("score")
    r.add_argument("--stage", type=Path, required=True)
    r.add_argument("--captures", type=Path, required=True)
    r.add_argument("--suite", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--include-fixtures", action="store_true")
    h = commands.add_parser("hillclimb")
    h.add_argument("--suite", type=Path, required=True)
    h.add_argument("--predictions", type=Path, required=True)
    h.add_argument("--out", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "stage":
            value = stage(args.suite, args.out, passes=tuple(args.passes), split=args.split,
                          max_tasks=args.max_tasks, model=args.model)
        elif args.command == "capture":
            value = capture(
                args.stage, args.out, passes=args.passes, server_loop_passes=args.server_loop_passes,
                base_url=args.base_url, served_model=args.served_model, share_cache=args.share_cache,
                max_requests=args.max_requests, max_tokens=args.max_tokens,
                timeout_seconds=args.timeout_seconds, max_wall_seconds=args.max_wall_seconds,
                dry_run=args.dry_run, api_key_env=args.api_key_env, runtime=args.runtime,
                weights_revision=args.weights_revision,
            )
        elif args.command == "score":
            value = score(args.stage, args.captures, args.suite, args.out,
                          include_fixtures=args.include_fixtures)
        else:
            value = hillclimb(args.suite, args.predictions, args.out)
        print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (KeyError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"inkling gym: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
