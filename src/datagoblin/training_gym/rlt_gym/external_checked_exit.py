"""Source-only external-verifier bridge for recurrent post-failure decisions.

Callers must supply a live independent checker with a pinned verifier SHA.
Historically exposed proof or tool traces are always training-ineligible.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from typing import Any

SCHEMA = "rlt_gym.external_checked_episode.v1"
KINDS = frozenset({"lean", "mcp", "research"})
SPLITS = frozenset({"train", "calibration", "ordinary_test", "ood_test"})
MAX_VISITS = 8
MAX_JSON_BYTES = 131072


def _canonical(value: Any) -> str:
    result = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(result.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("external checked episode exceeds bound")
    return result


def digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def replay_external_episode(
    episode: Mapping[str, Any],
    *,
    checker: Callable[[dict[str, Any], Any], dict[str, Any]],
    trusted_verifier_sha: str,
    trusted_source_revision: str,
    training_authorized: bool = False,
) -> dict[str, Any]:
    """Replay each proposal through an operator-supplied verifier; no oracle shortcuts."""
    if (not callable(checker) or not _sha(trusted_verifier_sha)
            or not _sha(trusted_source_revision)):
        raise ValueError("live checker and pinned verifier SHA required")
    if episode.get("schema_version") != SCHEMA or episode.get("kind") not in KINDS:
        raise ValueError("unknown external checker task")
    if episode.get("split") not in SPLITS:
        raise ValueError("protected or unrecognized split")
    for field in ("root_id", "source_revision", "model_sha256"):
        value = episode.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError("missing root, source revision or model identity")
    if not _sha(episode.get("model_sha256")):
        raise ValueError("invalid producer checkpoint hash")
    if episode.get("source_revision") != trusted_source_revision:
        raise ValueError("source revision differs from trusted registry")
    if episode.get("verifier_sha256") != trusted_verifier_sha:
        raise ValueError("pinned verifier differs from source")
    if episode.get("source_exposure") not in ("fresh", "historical"):
        raise ValueError("unknown training exposure")
    if episode.get("evidence_kind") != "native_operator_verified":
        raise ValueError("fixture or unverified evidence prohibited")
    public_task = episode.get("public_task")
    if not isinstance(public_task, dict) or episode.get("task_digest") != digest(public_task):
        raise ValueError("public task digest differs")
    attempts = episode.get("attempts")
    if not isinstance(attempts, list) or not 2 <= len(attempts) <= MAX_VISITS:
        raise ValueError("need bounded complete checked trajectory")
    if episode.get("planned_visits") != len(attempts) or episode.get("verified_stop") is not False:
        raise ValueError("trajectory must record full future counterfactual")
    replayed = []
    for visit, step in enumerate(attempts):
        if not isinstance(step, dict) or step.get("visit") != visit:
            raise ValueError("invalid or skipped visit sequence")
        candidate = step.get("candidate")
        _canonical(candidate)
        observed = step.get("verification")
        actual = checker(public_task, candidate)
        if (
            not isinstance(actual, dict)
            or type(actual.get("passed")) is not bool
            or not isinstance(actual.get("code"), str)
            or actual.get("verifier_sha256") != trusted_verifier_sha
            or actual.get("task_digest") != episode["task_digest"]
            or actual.get("candidate_digest") != digest(candidate)
            or not isinstance(observed, dict)
            or _canonical(actual) != _canonical(observed)
        ):
            raise ValueError("independent checker receipt disagrees")
        replayed.append({"visit": visit, "passed": actual["passed"], "code": actual["code"]})
    passed_first = replayed[0]["passed"]
    future = next((r["visit"] for r in replayed[1:] if r["passed"]), None)
    train_ok = (training_authorized and episode["split"] == "train"
                and episode["source_exposure"] == "fresh")
    return {
        "schema_version": "rlt_gym.external_postfailure.v1",
        "kind": episode["kind"],
        "root_id": episode["root_id"],
        "split": episode["split"],
        "task_digest": episode["task_digest"],
        "source_revision": episode["source_revision"],
        "model_sha256": episode["model_sha256"],
        "verifier_sha256": trusted_verifier_sha,
        "training_eligible": train_ok,
        "first_passed": passed_first,
        "observable_after_first": {
            "public_task": public_task,
            "candidate": attempts[0]["candidate"],
            "checker_code": replayed[0]["code"],
            "remaining_checks": len(attempts) - 1,
        },
        "future_rescue": not passed_first and future is not None,
        "first_verified_visit": 0 if passed_first else future,
        "cost_until_first_verified_or_budget": (
            1 if passed_first else future + 1 if future is not None else len(attempts)
        ),
        "outcome_exposed_to_controller": False,
        "evidence_class": "independently_replayed_native_external_verifier",
    }


def replay_corpus(
    episodes: Iterable[Mapping[str, Any]], *,
    checkers: Mapping[str, Callable[[dict[str, Any], Any], dict[str, Any]]],
    trusted_shas: Mapping[str, str],
    trusted_source_revisions: Mapping[str, str],
    approved_train_roots: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Recheck all roots and isolate TRAIN supervision from exposed evaluations."""
    seen_roots: set[str] = set()
    seen_tasks: set[str] = set()
    training, development = [], []
    for episode in episodes:
        kind = episode.get("kind")
        if kind not in checkers or kind not in trusted_shas or kind not in trusted_source_revisions:
            raise ValueError("unregistered external checker")
        row = replay_external_episode(
            episode, checker=checkers[kind], trusted_verifier_sha=trusted_shas[kind],
            trusted_source_revision=trusted_source_revisions[kind],
            training_authorized=episode.get("root_id") in approved_train_roots,
        )
        if row["root_id"] in seen_roots or row["task_digest"] in seen_tasks:
            raise ValueError("source root or task reused across splits")
        seen_roots.add(row["root_id"])
        seen_tasks.add(row["task_digest"])
        (training if row["training_eligible"] else development).append(row)
    if approved_train_roots - seen_roots:
        raise ValueError("authorized training roots absent from checked corpus")
    if not seen_roots:
        raise ValueError("empty corpus")
    return {
        "schema_version": "rlt_gym.external_checked_corpus.v1",
        "verified_roots": len(seen_roots),
        "train": training,
        "evaluation_only": development,
        "verified_candidate_checks": sum(
            row["observable_after_first"]["remaining_checks"] + 1
            for row in (*training, *development)
        ),
        "claim_boundary": "Native external checker callback required; archived cases remain evaluation-only.",
    }
