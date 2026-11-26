"""Text-action research environment; the environment owns artifacts and scores."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import VERSION
from .learners import normalize_config, validate_artifact
from .suite import LEARNERS, canon, check_answer, digest, persona_memory, validate_case

LIMITS = {
    "turns": 8,
    "experiments": 4,
    "checks": 2,
    "optimizer_updates": 256,
    "training_cell_calls": 1600000,
    "worker_seconds": 15,
    "worker_ram_mib": 512,
    "episode_seconds": 120,
    "response_bytes": 16384,
}


def run_worker(request: dict[str, Any], *, timeout_seconds: float | None = None) -> dict[str, Any]:
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "SYSTEMROOT", "VIRTUAL_ENV"}}
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
    bootstrap = "import sys; sys.path.insert(0,sys.argv[1]); from datagoblin.research_v2.worker import main; main()"
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", bootstrap, str(Path(__file__).resolve().parents[2])],
            input=canon(request),
            capture_output=True,
            text=True,
            timeout=min(LIMITS["worker_seconds"], timeout_seconds or LIMITS["worker_seconds"]),
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("worker_timeout") from error
    if completed.returncode or len(completed.stdout.encode()) > 150000:
        raise RuntimeError(f"worker_failed:{completed.returncode}")
    return json.loads(completed.stdout)


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse_action(text: str) -> dict[str, Any]:
    if type(text) is not str or len(text.encode()) > LIMITS["response_bytes"]:
        raise ValueError("response byte limit")
    value = json.loads(
        text,
        object_pairs_hook=_object,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("finite JSON required")),
    )
    if type(value) is not dict:
        raise ValueError("action must be a JSON object")
    canon(value)
    return value


def _claim(after: float, before: float) -> str:
    return "ties" if abs(after - before) < 1e-12 else "improves" if after > before else "worse"


class ResearchSession:
    def __init__(self, case: dict[str, Any], *, deadline: float | None = None) -> None:
        validate_case(case)
        self.case = copy.deepcopy(case)
        self.done = False
        self.started = time.monotonic()
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("finite deadline required")
        self.external_deadline = deadline
        self.turns = 0
        self.checks = 0
        self.experiments = 0
        self.updates = 0
        self.cell_calls = 0
        self.training_cell_calls = 0
        self.errors = 0
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._calibration: dict[str, dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._final: dict[str, Any] | None = None

    def _remaining(self) -> dict[str, int]:
        return {
            "turns": LIMITS["turns"] - self.turns,
            "experiments": LIMITS["experiments"] - self.experiments,
            "checks": LIMITS["checks"] - self.checks,
            "optimizer_updates": LIMITS["optimizer_updates"] - self.updates,
        }

    def observation(self, **extra: Any) -> dict[str, Any]:
        return {"done": self.done, "remaining": self._remaining(), "artifacts": self.artifact_summaries(), **extra}

    def artifact_summaries(self) -> list[dict[str, Any]]:
        return [
            {
                "id": key,
                "config": a["config"],
                "parameters": a["receipt"]["parameters"],
                "updates": a["receipt"]["optimizer_updates"],
                "calibration_accuracy": self._calibration[key]["accuracy"],
                "training_loss": a["receipt"]["loss_after"],
            }
            for key, a in self._artifacts.items()
        ]

    def remaining_seconds(self) -> float:
        deadline = self.started + LIMITS["episode_seconds"]
        if self.external_deadline is not None:
            deadline = min(deadline, self.external_deadline)
        return deadline - time.monotonic()

    def _worker(self, request: dict[str, Any]) -> dict[str, Any]:
        remaining = self.remaining_seconds()
        if remaining <= 0:
            self.done = True
            raise RuntimeError("episode_timeout")
        result = run_worker(request, timeout_seconds=remaining)
        if self.remaining_seconds() <= 0:
            self.done = True
            raise RuntimeError("episode_timeout")
        return result

    def step_json(self, text: str) -> dict[str, Any]:
        if self.done:
            return self.observation(status="episode_closed")
        self.turns += 1
        if self.remaining_seconds() <= 0:
            self.done = True
            return self.observation(status="episode_timeout")
        try:
            action = parse_action(text)
            name = action.get("action")
            allowed = {
                "run_experiment": {"action", "config"},
                "check": {"action", "answer"},
                "submit": {"action", "answer"},
                "inspect": {"action"},
            }
            if type(name) is not str or name not in allowed or set(action) != allowed[name]:
                raise ValueError("unknown action or fields")
            if name == "run_experiment":
                result = self._fit(action["config"])
            elif name == "check":
                if self.case["family"] in LEARNERS or self.checks >= LIMITS["checks"]:
                    raise ValueError("check unavailable; learned artifacts receive a final test after submission")
                self.checks += 1
                result = {"check_passed": check_answer(self.case, action["answer"])}
            elif name == "submit":
                self.done = True
                self._submit(action["answer"])
                result = {"status": "submitted"}
            else:
                result = {"status": "ready"}
            self._events.append({"turn": self.turns, "action_sha256": digest(action), "result_sha256": digest(result)})
        except (ValueError, TypeError, KeyError, OverflowError, RuntimeError, RecursionError) as error:
            self.errors += 1
            result = {"error": str(error)[:200]}
            self._events.append({"turn": self.turns, "error": type(error).__name__})
        if self.turns >= LIMITS["turns"]:
            self.done = True
        return self.observation(**result)

    def _fit(self, config: Any) -> dict[str, Any]:
        family = self.case["family"]
        if family not in LEARNERS or self.experiments >= LIMITS["experiments"]:
            raise ValueError("experiment unavailable")
        kind = LEARNERS[family]
        config = normalize_config(kind, config)
        calls = config["steps"] * config["train_examples"] * (12 * config["recursions"] if kind == "trm_style" else 1)
        if self.updates + config["steps"] > LIMITS["optimizer_updates"]:
            raise ValueError("optimizer update budget exceeded")
        if self.training_cell_calls + calls > LIMITS["training_cell_calls"]:
            raise ValueError("training cell budget exceeded")
        # A worker failure keeps its reserved cost and experiment slot.
        self.experiments += 1
        self.updates += config["steps"]
        self.training_cell_calls += calls
        reply = self._worker(
            {
                "op": "fit",
                "kind": kind,
                "seed": self.case["public"]["train_seed"],
                "config": config,
                "calibration_seed": self.case["private"]["calibration_seed"],
            }
        )
        artifact, calibration = reply["artifact"], reply["calibration"]
        validate_artifact(artifact)
        key = f"a{self.experiments}"
        self._artifacts[key], self._calibration[key] = artifact, calibration
        self.cell_calls += artifact["receipt"]["cell_calls"] + calibration["cell_calls"]
        return {
            "created": key,
            "receipt_sha256": digest(artifact["receipt"]),
            "artifact_sha256": artifact["artifact_sha256"],
            "loss_before": artifact["receipt"]["loss_before"],
        }

    def _submit(self, answer: Any) -> None:
        if self.case["family"] not in LEARNERS:
            passed = check_answer(self.case, answer)
            self._final = {
                "primary_score": float(passed),
                "success": passed,
                "integrity_pass": passed,
                "metrics": {"exact_verified": passed},
            }
            return
        expected_keys = {"artifact_id", "baseline_id", "claim"}
        if self.case["family"] == "persona_externalization":
            expected_keys.add("memory")
        if type(answer) is not dict or set(answer) != expected_keys:
            raise ValueError("invalid learned submission fields")
        selected, baseline_id = answer["artifact_id"], answer["baseline_id"]
        if type(selected) is not str or type(baseline_id) is not str:
            raise ValueError("artifact IDs must be strings")
        if selected not in self._artifacts or baseline_id not in self._artifacts:
            raise ValueError("artifact must come from this environment's experiment store")
        artifact, baseline = self._artifacts[selected], self._artifacts[baseline_id]
        config, baseline_config = validate_artifact(artifact), validate_artifact(baseline)
        if (
            config["steps"] <= 0
            or baseline_config["steps"] != 0
            or {k: v for k, v in config.items() if k != "steps"}
            != {k: v for k, v in baseline_config.items() if k != "steps"}
            or artifact["receipt"]["init_weights_sha256"] != baseline["receipt"]["weights_sha256"]
        ):
            raise ValueError("require a trained artifact and its matched zero-update initialization")
        claim_pass = answer["claim"] == _claim(
            self._calibration[selected]["accuracy"], self._calibration[baseline_id]["accuracy"]
        )
        memory_pass = self.case["family"] != "persona_externalization" or answer["memory"] == persona_memory(
            self.case["public"]
        )
        measurements = {}
        for split, key in (("evaluation", "evaluation_seed"), ("ood", "ood_seed")):
            for name, item in (("selected", artifact), ("baseline", baseline)):
                measured = self._worker(
                    {"op": "evaluate", "artifact": item, "seed": self.case["private"][key], "split": split}
                )
                self.cell_calls += measured["cell_calls"]
                measurements[f"{split}_{name}"] = measured
        held, ood = measurements["evaluation_selected"], measurements["ood_selected"]
        veto_pass = artifact["kind"] == "trm_style" or (
            config["critical_veto"] is True and held["critical_false_accepts"] == ood["critical_false_accepts"] == 0
        )
        changed = artifact["receipt"]["weights_sha256"] != artifact["receipt"]["init_weights_sha256"]
        integrity = claim_pass and memory_pass and veto_pass and changed
        quality = (held["accuracy"] + ood["accuracy"]) / 2
        delta = (
            quality - (measurements["evaluation_baseline"]["accuracy"] + measurements["ood_baseline"]["accuracy"]) / 2
        )
        self._final = {
            "primary_score": quality if integrity else 0.0,
            "success": bool(integrity and quality >= 0.75 and delta > 0),
            "integrity_pass": bool(integrity),
            "metrics": {
                "claim_supported": claim_pass,
                "persona_memory_correct": memory_pass,
                "critical_veto_pass": veto_pass,
                "weights_changed": changed,
                "held_accuracy": held["accuracy"],
                "ood_accuracy": ood["accuracy"],
                "delta_from_initialization": delta,
                **measurements,
            },
            "selected_artifact_sha256": artifact["artifact_sha256"],
            "baseline_artifact_sha256": baseline["artifact_sha256"],
        }

    def result(self) -> dict[str, Any]:
        result = {
            "schema": VERSION,
            "task_id": self.case["id"],
            "root_id": self.case["root_id"],
            "family": self.case["family"],
            "split": self.case["split"],
            **(self._final or {"primary_score": 0.0, "success": False, "integrity_pass": False, "metrics": {}}),
            "completed": self._final is not None,
            "closed": self.done,
            "cost": {
                "turns": self.turns,
                "experiments": self.experiments,
                "checks": self.checks,
                "optimizer_updates_reserved": self.updates,
                "training_cell_calls_reserved": self.training_cell_calls,
                "measured_cell_calls": self.cell_calls,
                "errors": self.errors,
            },
            "events_sha256": digest(self._events),
            "artifact_hashes": [a["artifact_sha256"] for a in self._artifacts.values()],
        }
        result["result_sha256"] = digest(result)
        return result

    def save_artifacts(self, base: Path) -> None:
        """Persist evaluator-owned tiny weights for later replay; never a model action."""
        if not self._artifacts:
            return
        folder = base / digest(self.case["id"])[:16]
        folder.mkdir(parents=True, mode=0o700)
        for name, artifact in self._artifacts.items():
            target = folder / f"{name}.json"
            with target.open("x") as handle:
                handle.write(canon(artifact) + "\n")
            target.chmod(0o600)
