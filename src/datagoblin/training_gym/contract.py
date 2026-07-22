from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from datagoblin.serialization import load_structured


def load_gym_contract(path: Path) -> dict[str, Any]:
    value = load_structured(path)
    if value.get("schema_version") != "1.0":
        raise ValueError("unsupported training-gym schema version")
    required = {
        "gym_id",
        "question",
        "hypotheses",
        "evaluation_tasks",
        "candidate_families",
        "seed_plan",
        "splits",
        "metrics",
        "promotion_gates",
        "trainer",
        "mandatory_non_claims",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"training-gym contract missing fields: {', '.join(missing)}")
    selection = set(value["metrics"]["selection"])
    evidence = set(value["metrics"]["evidence"])
    if selection & evidence:
        raise ValueError("selection and evidence metrics must be disjoint")
    split_sets = [set(value["seed_plan"][name]) for name in ("discovery", "confirmatory", "holdout")]
    if any(left & right for index, left in enumerate(split_sets) for right in split_sets[index + 1 :]):
        raise ValueError("discovery, confirmatory, and holdout seeds must be disjoint")
    trainer = value["trainer"]
    if not trainer.get("training_task_id") or not trainer.get("chunk_strategy"):
        raise ValueError("trainer requires a task id and chunk strategy")
    interval = trainer.get("checkpoint_interval", {})
    if int(interval.get("steps", 0)) <= 0 or int(interval.get("seconds", 0)) <= 0:
        raise ValueError("checkpoint intervals must be positive")
    _validate_caps(trainer.get("caps", {}))
    return value


def _validate_caps(caps: Mapping[str, Any]) -> None:
    for name in ("ram_mb", "cpu_percent", "io_mb_per_second", "wall_clock_seconds"):
        if int(caps.get(name, 0)) <= 0:
            raise ValueError(f"hard cap {name} must be positive")
    if int(caps["cpu_percent"]) > 100:
        raise ValueError("CPU cap cannot exceed 100 percent")
    if int(caps.get("swap_bytes", -1)) != 0:
        raise ValueError("training-gym smoke contract requires swap_bytes=0")
