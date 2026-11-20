"""Bounded CPU controls, immutable rollout receipts, and exact local replay."""

from __future__ import annotations

import json
import math
import platform
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import canonical_json, dump_structured, load_structured

from .tasks import load_suite, reference_answer, verify

MODES = ("direct", "rlt", "matched_damping")
READOUTS = ("carried_answer", "endpoint")
SPLITS = ("train", "calibration", "ordinary_test", "ood_test", "sealed_poison")
MAX_EPISODES = 100_000
MAX_TRACE_BYTES = 262_144


class RecurrentAdapter(Protocol):
    """A local adapter supplies endpoints/states; the environment owns scores."""

    metadata: dict[str, Any]
    model_digest: str

    def visits(
        self, task: dict[str, Any], mode: str, planned_depth: int, *,
        readout_site: str = "carried_answer", intervention: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]: ...


def _jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if len(line.encode()) > MAX_TRACE_BYTES:
                raise ValueError(f"oversized trace at line {number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at line {number}")
            canonical_json(value)  # Reject NaN/Infinity recursively.
            result.append(value)
            if len(result) > MAX_EPISODES:
                raise ValueError("too many rollout records")
    return result


def source_fingerprint() -> dict[str, str]:
    paths = [Path(__file__).parent / name for name in ("__init__.py", "tasks.py", "dynamics.py", "runner.py")]
    paths.append(Path(__file__).parents[1] / "rlt_feedback.py")
    paths.extend(Path(__file__).parents[2] / name for name in ("hashing.py", "serialization.py"))
    return {str(path.relative_to(Path(__file__).parents[2])): sha256_file(path) for path in paths}


def _manifest_path(path: Path) -> Path:
    return path / "manifest.json" if path.is_dir() else path


def _positive_int(value: int, name: str, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")


def _validate_options(
    seeds: tuple[int, ...], modes: tuple[str, ...], training_modes: tuple[str, ...],
    depths: tuple[int, ...], splits: tuple[str, ...], readout_site: str, max_wall_seconds: float,
) -> None:
    for values, name in ((seeds, "seeds"), (modes, "modes"), (training_modes, "training_modes"),
                         (depths, "depths"), (splits, "splits")):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be nonempty and distinct")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("model seeds must be uint32 integers")
    if set(modes) - set(MODES) or set(training_modes) - set(MODES):
        raise ValueError("unknown recurrence mode")
    if set(splits) - set(SPLITS) or readout_site not in READOUTS:
        raise ValueError("unknown split or readout site")
    for depth in depths:
        _positive_int(depth, "planned depth", 32)
    if isinstance(max_wall_seconds, bool) or not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0:
        raise ValueError("wall budget must be positive and finite")


def rollout(
    task: dict[str, Any], adapter: RecurrentAdapter | None, *, mode: str = "direct",
    planned_depth: int = 4, readout_site: str = "carried_answer", seed: int = 7,
    verified_stop: bool = True, control: str | None = None,
    intervention: dict[str, Any] | None = None, deadline: float | None = None,
) -> dict[str, Any]:
    """Execute one episode. The planned clock survives a verifier-triggered stop."""
    _validate_options((seed,), (mode,), ("direct",), (planned_depth,), (task["split"],), readout_site, 1)
    if type(verified_stop) is not bool:
        raise ValueError("verified_stop must be boolean")
    if control not in {None, "reference_solver", "candidate_order"}:
        raise ValueError("unknown control")
    if control is None and adapter is None:
        raise ValueError("continuous rollouts need an adapter")
    backend = {
        "kind": control or "frozen_random_reservoir",
        "dynamics_kind": "continuous_recurrent" if control is None else "discrete_search",
        "provider": "local_cpu", "model": control or "numpy_recurrent_reservoir_v1",
        "model_digest": adapter.model_digest if control is None and adapter is not None else None,
        "architecture_promotion_eligible": False,
        "bpd_mechanism_verified": False,
        "observability": "state_summaries" if control is None else "candidate_and_checker",
    }
    if adapter is not None and control is None:
        metadata = adapter.metadata
        backend["training"] = {key: metadata[key] for key in (
            "model_kind", "training_kind", "fitted", "training_mode", "training_depth",
            "training_readout_site", "initialization_digest", "training_fingerprint",
            "training_receipts_digest", "trainable_parameters", "total_parameters", "training_splits",
            "training_tasks", "training_candidates", "training_readout_examples", "supervision",
        ) if key in metadata}
    config = {
        "mode": mode, "seed": seed, "planned_depth": planned_depth,
        "readout_site": readout_site, "verified_stop": verified_stop,
        "placement": "answer", "clock_scope": "segment", "reset_policy": "fresh_episode",
        "intervention": intervention,
    }
    episode = {
        "schema_version": "rlt_gym.rollout.v1",
        "task_id": task["task_id"], "task_digest": task["task_digest"],
        "family": task["family"], "generator_id": task["generator_id"],
        "lineage_id": task["lineage_id"], "split": task["split"],
        "evidence_epoch": task["evidence_epoch"], "backend": backend, "config": config,
        "steps": [],
    }
    episode["episode_id"] = sha256_json(episode)
    if control == "reference_solver":
        proposals = (
            {"visit": index, "candidate": reference_answer(task), "planned_depth": planned_depth}
            for index in range(1)
        )
    elif control == "candidate_order":
        proposals = (
            {"visit": index, "candidate": candidate, "planned_depth": planned_depth}
            for index, candidate in enumerate(task["candidates"][:planned_depth])
        )
    else:
        assert adapter is not None
        proposals = adapter.visits(
            task, mode, planned_depth, readout_site=readout_site, intervention=intervention,
        )
    status = "failed"
    stop_reason = "planned_depth_exhausted"
    first_verified = None
    verifier_calls = 0
    adapter_calls = 0
    try:
        while len(episode["steps"]) < planned_depth:
            if deadline is not None and time.monotonic() >= deadline:
                status, stop_reason = "resource_abort", "wall_budget_exhausted"
                break
            try:
                step = next(proposals)
            except StopIteration as exc:
                raise ValueError("adapter ended before its planned visit budget") from exc
            adapter_calls += 1
            if step.get("visit") != len(episode["steps"]):
                raise ValueError("adapter violated visit budget or order")
            if step.get("planned_depth", planned_depth) != planned_depth:
                raise ValueError("adapter changed the planned clock")
            if step.get("evidence_epoch", task["evidence_epoch"]) != task["evidence_epoch"]:
                raise ValueError("adapter used stale evidence")
            # Canonicalization also copies adapter-owned mutable values.
            step = json.loads(canonical_json(step))
            if len(canonical_json(step).encode()) > MAX_TRACE_BYTES // 32:
                raise ValueError("adapter step exceeds trace size bound")
            verifier_calls += 1
            receipt = verify(task, step["candidate"])
            step.update({"verification": receipt, "evidence_epoch": task["evidence_epoch"]})
            step.setdefault("segment_id", 0)
            if episode["steps"] and step["segment_id"] != episode["steps"][0]["segment_id"]:
                raise ValueError("adapter changed segment without a reset declaration")
            episode["steps"].append(step)
            if receipt["passed"]:
                if first_verified is None:
                    first_verified = step["visit"]
                status = "passed"
                if verified_stop:
                    stop_reason = "first_verified_answer"
                    break
            else:
                status = "failed"
    except (ValueError, TypeError, KeyError, FloatingPointError) as exc:
        status, stop_reason = "verifier_error", "invalid_adapter_output"
        episode["error"] = {"type": type(exc).__name__, "message": str(exc)[:256]}
    episode["outcome"] = {
        "status": status, "passed": status == "passed", "stop_reason": stop_reason,
        "realized_visits": len(episode["steps"]), "first_verified_visit": first_verified,
        "verifier_calls": verifier_calls, "adapter_calls": adapter_calls,
        "censored": status in {"resource_abort", "verifier_error"},
    }
    episode["episode_digest"] = sha256_json(episode)
    return episode


def verify_episode(task: dict[str, Any], episode: dict[str, Any]) -> None:
    if episode.get("schema_version") != "rlt_gym.rollout.v1":
        raise ValueError("unknown rollout schema")
    if episode.get("episode_digest") != sha256_json({k: v for k, v in episode.items() if k != "episode_digest"}):
        raise ValueError("rollout digest mismatch")
    identity = {k: v for k, v in episode.items() if k not in {"episode_id", "episode_digest", "outcome", "error"}}
    identity["steps"] = []
    if episode.get("episode_id") != sha256_json(identity):
        raise ValueError("rollout identity mismatch")
    for field in ("task_id", "task_digest", "family", "generator_id", "lineage_id", "split", "evidence_epoch"):
        if episode.get(field) != task[field]:
            raise ValueError(f"rollout task binding mismatch: {field}")
    steps = episode["steps"]
    if episode["outcome"]["realized_visits"] != len(steps):
        raise ValueError("rollout visit count mismatch")
    if len(steps) > episode["config"]["planned_depth"]:
        raise ValueError("rollout exceeds visit budget")
    for index, step in enumerate(steps):
        if step["visit"] != index or step["verification"] != verify(task, step["candidate"]):
            raise ValueError("rollout verifier receipt mismatch")
        if (step.get("planned_depth") != episode["config"]["planned_depth"]
                or step.get("evidence_epoch") != task["evidence_epoch"]
                or step.get("segment_id") != steps[0]["segment_id"]):
            raise ValueError("rollout clock, epoch, or segment changed")
    outcome = episode["outcome"]
    first_verified = next((step["visit"] for step in steps if step["verification"]["passed"]), None)
    if outcome.get("first_verified_visit") != first_verified:
        raise ValueError("first verified visit mismatch")
    calls = outcome.get("verifier_calls")
    attempts = outcome.get("adapter_calls")
    if type(calls) is not int or type(attempts) is not int or not len(steps) <= calls <= attempts <= len(steps) + 1:
        raise ValueError("invalid attempted work counts")
    censored = outcome["status"] in {"resource_abort", "verifier_error"}
    if outcome.get("censored") is not censored:
        raise ValueError("censored outcome mismatch")
    if not censored and (calls != len(steps) or attempts != len(steps)):
        raise ValueError("completed rollout work count mismatch")
    if not censored:
        if not steps or outcome["passed"] != steps[-1]["verification"]["passed"]:
            raise ValueError("terminal verifier outcome mismatch")
        if episode["config"]["verified_stop"] and first_verified is not None and first_verified != len(steps) - 1:
            raise ValueError("rollout continued past verified stop")
        if len(steps) < episode["config"]["planned_depth"] and not (
            episode["config"]["verified_stop"] and outcome["passed"]
        ):
            raise ValueError("unexplained early stop")
    if outcome["passed"] != (outcome["status"] == "passed"):
        raise ValueError("rollout status mismatch")
    if outcome["passed"] and (not steps or not steps[-1]["verification"]["passed"]):
        raise ValueError("successful rollout lacks a verified final answer")


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Report family/work cells and within-task, same-weight schedule comparisons."""
    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        cfg, backend = episode["config"], episode["backend"]
        train = backend.get("training", {}).get("training_mode", "none")
        key = (episode["split"], episode["family"], backend["kind"], train, cfg["mode"], cfg["planned_depth"])
        cells[key].append(episode)
    rows = []
    for key, values in sorted(cells.items()):
        passed = sum(item["outcome"]["passed"] for item in values)
        rows.append({
            **dict(zip(("split", "family", "backend", "training_mode", "mode", "planned_depth"), key, strict=True)),
            "episodes": len(values), "verified_successes": passed, "success_rate": passed / len(values),
            "verifier_calls": sum(item["outcome"]["verifier_calls"] for item in values),
            "statuses": dict(Counter(item["outcome"]["status"] for item in values)),
        })
    paired: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for episode in episodes:
        if episode["backend"]["kind"] != "frozen_random_reservoir":
            continue
        cfg = episode["config"]
        key = (episode["task_digest"], episode["backend"]["model_digest"], cfg["seed"],
               cfg["planned_depth"], cfg["readout_site"], cfg["verified_stop"], sha256_json(cfg["intervention"]))
        if cfg["mode"] in paired[key]:
            raise ValueError("duplicate paired-comparison cell")
        paired[key][cfg["mode"]] = episode
    comparisons: dict[tuple[Any, ...], Counter] = defaultdict(Counter)
    for group in paired.values():
        direct = group.get("direct")
        if direct is None:
            continue
        for mode in ("rlt", "matched_damping"):
            challenger = group.get(mode)
            if challenger is None:
                continue
            key = (direct["split"], direct["family"], direct["config"]["planned_depth"], mode)
            counter = comparisons[key]
            a, b = direct["outcome"], challenger["outcome"]
            counter["pairs"] += 1
            if a["status"] not in {"passed", "failed"} or b["status"] not in {"passed", "failed"}:
                counter["censored_pairs"] += 1
                continue
            counter["wins"] += int(b["passed"] and not a["passed"])
            counter["regressions"] += int(a["passed"] and not b["passed"])
            counter["ties"] += int(a["passed"] == b["passed"])
            counter["verifier_call_delta"] += b["verifier_calls"] - a["verifier_calls"]
    return {
        "schema_version": "rlt_gym.summary.v1", "evidence_class": "engineering_smoke",
        "episodes": len(episodes), "verified_successes": sum(row["outcome"]["passed"] for row in episodes),
        "verifier_calls": sum(row["outcome"]["verifier_calls"] for row in episodes),
        "adapter_calls": sum(row["outcome"]["adapter_calls"] for row in episodes),
        "architecture_promotion_eligible": False, "bpd_mechanism_verified": False,
        "claim_scope": "Frozen random recurrent features with a fitted readout; source-generated task controls.",
        "cells": rows,
        "paired_against_direct": [
            {**dict(zip(("split", "family", "planned_depth", "mode"), key, strict=True)), **dict(value)}
            for key, value in sorted(comparisons.items())
        ],
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# RLT rollout gym", "", "Evidence class: **engineering smoke**.", "",
        summary["claim_scope"], "",
        f"Collected {summary['episodes']} episodes and {summary['verifier_calls']} candidate checks.", "",
        "Each native model uses frozen recurrent weights and a readout fitted only on training lineages.",
        "All recurrent schedules share the initialized block, input features, and conditioning within a seed.", "",
        "| Split | Family | Backend | Train rule | Runtime rule | Visits | Solved / episodes | Checks |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary["cells"]:
        lines.append(
            f"| {row['split']} | {row['family']} | {row['backend']} | {row['training_mode']} | "
            f"{row['mode']} | {row['planned_depth']} | {row['verified_successes']} / {row['episodes']} | "
            f"{row['verifier_calls']} |"
        )
    lines.extend(["", "Paired schedule deltas are recorded in summary.json. Timing measurements live separately",
                  "so the canonical rollout stream can replay byte for byte.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_suite(
    suite_manifest: Path, out: Path, *, seeds: tuple[int, ...] = (7, 19),
    modes: tuple[str, ...] = MODES, training_modes: tuple[str, ...] = ("direct",),
    depths: tuple[int, ...] = (1, 2, 4, 8), splits: tuple[str, ...] = ("calibration",),
    readout_site: str = "carried_answer", training_depth: int = 4,
    verified_stop: bool = True, max_wall_seconds: float = 120,
    include_controls: bool = True, intervention: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .dynamics import NativeRanker

    _validate_options(seeds, modes, training_modes, depths, splits, readout_site, max_wall_seconds)
    _positive_int(training_depth, "training depth", 32)
    for depth in depths:
        NativeRanker._validate_intervention(intervention, depth)
    _, tasks = load_suite(suite_manifest)
    train = [task for task in tasks if task["split"] == "train"]
    selected = [task for task in tasks if task["split"] in splits]
    if not train or not selected:
        raise ValueError("suite requires both training tasks and requested evaluation tasks")
    expected = len(selected) * (len(seeds) * len(training_modes) * len(modes) * len(depths)
                               + (2 if include_controls else 0))
    if expected > MAX_EPISODES:
        raise ValueError("matrix exceeds the episode cap")
    out.mkdir(parents=True, exist_ok=False)
    (out / "models").mkdir()
    start = time.monotonic()
    deadline = start + max_wall_seconds
    configuration = {
        "seeds": list(seeds), "modes": list(modes), "training_modes": list(training_modes),
        "depths": list(depths), "splits": list(splits), "readout_site": readout_site,
        "training_depth": training_depth, "verified_stop": verified_stop,
        "max_wall_seconds": max_wall_seconds, "include_controls": include_controls,
        "intervention": intervention,
    }
    dump_structured(out / "plan.json", {
        "schema_version": "rlt_gym.run_plan.v1", "configuration": configuration,
        "suite_manifest_sha256": sha256_file(_manifest_path(suite_manifest)),
        "source_fingerprint": source_fingerprint(), "expected_episodes": expected,
    })
    episodes, measurements, models = [], [], {}
    # Stream each completed episode, retaining partial progress after interruption.
    with (out / "rollouts.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        def append_episode(task, adapter, **options):
            tick = time.monotonic()
            episode = rollout(task, adapter, deadline=deadline, **options)
            episodes.append(episode)
            handle.write(canonical_json(episode) + "\n")
            handle.flush()
            measurements.append({"episode_id": episode["episode_id"], "elapsed_seconds": time.monotonic() - tick})

        for seed in seeds:
            for training_mode in training_modes:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "wall budget expired during fitting; partial run retained without a completion manifest"
                    )
                model = NativeRanker(seed)
                model.fit(train, training_mode=training_mode, planned_depth=training_depth, readout_site=readout_site)
                model_path = Path("models") / f"seed-{seed}-{training_mode}.json"
                model.save(out / model_path)
                models[model.model_digest] = {"path": model_path.as_posix(), "sha256": sha256_file(out / model_path)}
                for task in selected:
                    for depth in depths:
                        for mode in modes:
                            append_episode(
                                task, model, mode=mode, planned_depth=depth, readout_site=readout_site,
                                seed=seed, verified_stop=verified_stop, intervention=intervention,
                            )
        if include_controls:
            for task in selected:
                append_episode(task, None, control="reference_solver", planned_depth=1)
                append_episode(task, None, control="candidate_order", planned_depth=min(32, len(task["candidates"])))
    summary = summarize(episodes)
    dump_structured(out / "summary.json", summary)
    _write_report(out / "REPORT.md", summary)
    _jsonl(out / "measurements.jsonl", measurements)
    resources = {
        "wall_seconds": time.monotonic() - start, "device": "cpu", "weights_downloaded": False,
        "episode_budget": expected, "max_planned_visits": max(depths),
        "process_hard_cap_enforced": False,
        "limit_scope": "Bounded tiny CPU controls and checked episode/visit budgets; external wrapper owns OS caps.",
    }
    try:
        import resource

        resources["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if platform.system() == "Darwin" else 1024
        )
    except ImportError:
        resources["process_peak_rss_bytes"] = None
    dump_structured(out / "resources.json", resources)
    files = {name: sha256_file(out / name) for name in (
        "rollouts.jsonl", "summary.json", "REPORT.md", "measurements.jsonl", "resources.json", "plan.json",
    )}
    manifest = {
        "schema_version": "rlt_gym.run.v1", "complete": True, "configuration": configuration,
        "suite_manifest_sha256": sha256_file(_manifest_path(suite_manifest)),
        "source_fingerprint": source_fingerprint(), "models": models, "files": files,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "evidence_class": "engineering_smoke", "expected_episodes": expected,
    }
    manifest["manifest_digest"] = sha256_json(manifest)
    dump_structured(out / "manifest.json", manifest)
    return {"run_dir": str(out), "episodes": len(episodes), "manifest_digest": manifest["manifest_digest"],
            "summary": summary, "resources": resources}


def replay_run(suite_manifest: Path, run_dir: Path) -> dict[str, Any]:
    from .dynamics import NativeRanker

    manifest = load_structured(run_dir / "manifest.json")
    if manifest.get("schema_version") != "rlt_gym.run.v1" or manifest.get("complete") is not True:
        raise ValueError("a completed gym manifest is required")
    if manifest["manifest_digest"] != sha256_json({k: v for k, v in manifest.items() if k != "manifest_digest"}):
        raise ValueError("run manifest digest mismatch")
    if manifest["suite_manifest_sha256"] != sha256_file(_manifest_path(suite_manifest)):
        raise ValueError("suite binding mismatch")
    if manifest["source_fingerprint"] != source_fingerprint():
        raise ValueError("source changed; restore the recorded revision for exact replay")
    _, tasks = load_suite(suite_manifest)
    by_id = {task["task_id"]: task for task in tasks}
    for name, digest in manifest["files"].items():
        path = (run_dir / name).resolve()
        if not path.is_relative_to(run_dir.resolve()) or sha256_file(path) != digest:
            raise ValueError(f"run artifact digest mismatch: {name}")
    models = {}
    for digest, artifact in manifest["models"].items():
        path = (run_dir / artifact["path"]).resolve()
        if not path.is_relative_to(run_dir.resolve()) or sha256_file(path) != artifact["sha256"]:
            raise ValueError("model artifact digest mismatch")
        model = NativeRanker.load(path)
        if model.model_digest != digest:
            raise ValueError("model identity mismatch")
        models[digest] = model
    episodes = read_jsonl(run_dir / "rollouts.jsonl")
    if len(episodes) != manifest["expected_episodes"]:
        raise ValueError("incomplete rollout matrix")
    _verify_matrix(manifest["configuration"], tasks, models, episodes)
    exact, receipts_only, checks = 0, 0, 0
    seen = set()
    for episode in episodes:
        if episode["episode_id"] in seen:
            raise ValueError("duplicate episode")
        seen.add(episode["episode_id"])
        task = by_id[episode["task_id"]]
        verify_episode(task, episode)
        checks += len(episode["steps"])
        if episode["outcome"]["status"] in {"resource_abort", "verifier_error"}:
            receipts_only += 1
            continue
        cfg, backend = episode["config"], episode["backend"]
        control = backend["kind"] if backend["kind"] in {"reference_solver", "candidate_order"} else None
        model = None if control else models[backend["model_digest"]]
        actual = rollout(
            task, model, mode=cfg["mode"], planned_depth=cfg["planned_depth"],
            readout_site=cfg["readout_site"], seed=cfg["seed"], verified_stop=cfg["verified_stop"],
            control=control, intervention=cfg["intervention"],
        )
        if canonical_json(actual) != canonical_json(episode):
            raise ValueError(f"trajectory replay mismatch for episode {episode['episode_id']}")
        exact += 1
    if summarize(episodes) != load_structured(run_dir / "summary.json"):
        raise ValueError("summary replay mismatch")
    return {
        "schema_version": "rlt_gym.replay.v1", "passed": True, "episodes": len(episodes),
        "exact_trajectory_replays": exact, "receipt_only_aborted_episodes": receipts_only,
        "independently_rechecked_candidates": checks,
        "rollouts_sha256": sha256_file(run_dir / "rollouts.jsonl"),
        "source_fingerprint": source_fingerprint(), "run_manifest_digest": manifest["manifest_digest"],
    }


def _cell_key(episode: dict[str, Any]) -> tuple[Any, ...]:
    cfg, backend = episode["config"], episode["backend"]
    return (
        episode["task_digest"], backend["kind"], backend["model_digest"], cfg["seed"],
        cfg["mode"], cfg["planned_depth"], cfg["readout_site"], cfg["verified_stop"],
        sha256_json(cfg["intervention"]),
    )


def _verify_matrix(configuration, tasks, models, episodes) -> None:
    expected_models = {(seed, mode) for seed in configuration["seeds"] for mode in configuration["training_modes"]}
    observed_models = {}
    train_digests = sorted(task["task_digest"] for task in tasks if task["split"] == "train")
    for digest, model in models.items():
        metadata = model.metadata
        key = (metadata["seed"], metadata["training_mode"])
        if (key in observed_models or metadata["training_task_digests"] != train_digests
                or metadata["training_depth"] != configuration["training_depth"]
                or metadata["training_readout_site"] != configuration["readout_site"]):
            raise ValueError("trained model does not match the frozen matrix")
        observed_models[key] = digest
    if set(observed_models) != expected_models:
        raise ValueError("model matrix is incomplete or contains unplanned cells")
    expected_cells = set()
    for task in tasks:
        if task["split"] not in configuration["splits"]:
            continue
        for (seed, _), digest in observed_models.items():
            for mode in configuration["modes"]:
                for depth in configuration["depths"]:
                    expected_cells.add((
                        task["task_digest"], "frozen_random_reservoir", digest, seed, mode, depth,
                        configuration["readout_site"], configuration["verified_stop"],
                        sha256_json(configuration["intervention"]),
                    ))
        if configuration["include_controls"]:
            for kind, depth in (("reference_solver", 1), ("candidate_order", min(32, len(task["candidates"])))):
                expected_cells.add((task["task_digest"], kind, None, 7, "direct", depth,
                                    "carried_answer", True, sha256_json(None)))
    actual_cells = [_cell_key(episode) for episode in episodes]
    if len(set(actual_cells)) != len(actual_cells) or set(actual_cells) != expected_cells:
        raise ValueError("rollout matrix has missing, duplicate, or unplanned cells")


def mine_frontier(suite_manifest: Path, rollouts_path: Path, out: Path, *, max_tasks: int = 64) -> dict[str, Any]:
    """Select training failures/disagreements for teacher work with a constant family mix."""
    _positive_int(max_tasks, "max tasks", 10_000)
    _, tasks = load_suite(suite_manifest)
    by_id = {task["task_id"]: task for task in tasks}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded = Counter()
    for episode in read_jsonl(rollouts_path):
        task = by_id.get(episode["task_id"])
        if task is None:
            raise ValueError("frontier contains an unknown task")
        verify_episode(task, episode)
        if task["split"] != "train":
            excluded[task["split"]] += 1
            continue
        if episode["outcome"]["status"] not in {"passed", "failed"}:
            excluded["censored:" + episode["outcome"]["status"]] += 1
            continue
        if episode["backend"]["kind"] in {"reference_solver", "candidate_order"}:
            continue
        grouped[task["task_id"]].append(episode)
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task_id, values in grouped.items():
        failures = sum(not episode["outcome"]["passed"] for episode in values)
        if not failures:
            continue
        task = by_id[task_id]
        families[task["family"]].append({
            "task_id": task_id, "task_digest": task["task_digest"], "family": task["family"],
            "lineage_id": task["lineage_id"], "split": "train", "episodes": len(values),
            "failed_episodes": failures, "schedule_disagreement": 0 < failures < len(values),
            "priority": failures / len(values),
            "source_episode_digests": sorted(episode["episode_digest"] for episode in values),
        })
    for values in families.values():
        values.sort(key=lambda row: (-row["schedule_disagreement"], -row["priority"], row["task_id"]))
    selected = []
    while len(selected) < max_tasks and any(families.values()):
        for family in sorted(families):
            if families[family] and len(selected) < max_tasks:
                selected.append(families[family].pop(0))
    out.mkdir(parents=True, exist_ok=False)
    _jsonl(out / "frontier.jsonl", selected)
    summary = {
        "schema_version": "rlt_gym.frontier.v1", "selection_split": "train",
        "sampling": "round_robin_family_then_disagreement_then_failure_rate",
        "selected": len(selected), "task_ids": [row["task_id"] for row in selected],
        "families": dict(Counter(row["family"] for row in selected)), "excluded_episodes": dict(excluded),
        "suite_manifest_sha256": sha256_file(_manifest_path(suite_manifest)),
        "source_rollouts_sha256": sha256_file(rollouts_path), "frontier_sha256": sha256_file(out / "frontier.jsonl"),
    }
    dump_structured(out / "manifest.json", summary)
    return summary
