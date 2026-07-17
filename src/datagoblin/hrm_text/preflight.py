from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import dump_structured, load_structured

MIB = 1024 * 1024


@dataclass(frozen=True)
class HardCaps:
    ram_mb: int = 4096
    cpu_percent: int = 50
    io_mb_per_second: int = 50
    wall_clock_seconds: int = 1800

    def __post_init__(self) -> None:
        if min(self.ram_mb, self.cpu_percent, self.io_mb_per_second, self.wall_clock_seconds) <= 0:
            raise ValueError("hard caps must be positive")
        if self.cpu_percent > 100:
            raise ValueError("CPU cap cannot exceed 100 percent")

    def to_jsonable(self) -> dict[str, int]:
        return {
            "ram_mb": self.ram_mb,
            "cpu_percent": self.cpu_percent,
            "io_mb_per_second": self.io_mb_per_second,
            "wall_clock_seconds": self.wall_clock_seconds,
        }


@dataclass(frozen=True)
class CheckpointAssessment:
    model_path: str
    model_config_hash: str
    weight_path: str
    weight_hash: str | None
    weight_hash_status: str
    weight_bytes: int
    conservative_runtime_floor_bytes: int
    cap_bytes: int
    load_permitted: bool
    decision: str
    reason: str

    def to_jsonable(self) -> dict[str, Any]:
        return self.__dict__.copy()


def assess_checkpoint(model_path: Path, caps: HardCaps) -> CheckpointAssessment:
    resolved = model_path.resolve()
    config_path = resolved / "config.json"
    weight_path = resolved / "model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        raise ValueError("checkpoint requires config.json and model.safetensors")
    config = load_structured(config_path)
    if config.get("model_type") != "hrm_text":
        raise ValueError("checkpoint model_type is not hrm_text")
    weight_bytes = weight_path.stat().st_size
    # This is deliberately conservative and excludes activations/KV cache.
    runtime_floor = weight_bytes + 512 * MIB
    cap_bytes = caps.ram_mb * MIB
    permitted = runtime_floor <= cap_bytes
    reason = (
        "static checkpoint footprint fits the RAM cap; OS wrapper and runtime probes remain mandatory"
        if permitted
        else "static checkpoint plus minimum runtime reserve exceeds the RAM cap"
    )
    return CheckpointAssessment(
        model_path=str(resolved),
        model_config_hash=sha256_file(config_path),
        weight_path=str(weight_path),
        weight_hash=None,
        weight_hash_status="deferred_to_io_capped_wrapper",
        weight_bytes=weight_bytes,
        conservative_runtime_floor_bytes=runtime_floor,
        cap_bytes=cap_bytes,
        load_permitted=permitted,
        decision="PROCEED_TO_CAPPED_WRAPPER" if permitted else "ABORT_BEFORE_LOAD",
        reason=reason,
    )


def run_checkpoint_preflight(
    model_path: Path,
    out_dir: Path,
    *,
    run_id: str,
    caps: HardCaps | None = None,
) -> CheckpointAssessment:
    selected_caps = caps or HardCaps()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    plan = {
        "schema_version": "1.0",
        "run_id": run_id,
        "task": "one-example HRM-Text constitutional-verdict trace preflight",
        "caps": selected_caps.to_jsonable(),
        "chunk_strategy": "one dataset example per flushed trace record",
        "checkpoint_cadence": "flush every example and at least every 60 seconds",
        "network": False,
        "subprocess_policy": "only a child created suspended and assigned to the cap before resume",
        "weight_loading": "forbidden until static assessment and all hard-cap probes pass",
        "abort_is_valid": True,
    }
    dump_structured(out_dir / "trainer_plan.json", plan)
    assessment = assess_checkpoint(model_path, selected_caps)
    events = [
        {"ts": started, "event": "start", "run_id": run_id, "caps": selected_caps.to_jsonable()},
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "static_checkpoint_assessment",
            **assessment.to_jsonable(),
        },
    ]
    status = "ready_for_capped_wrapper" if assessment.load_permitted else "aborted"
    abort_reason = None if assessment.load_permitted else "static_model_footprint_exceeds_ram_cap"
    if abort_reason:
        events.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "abort",
                "reason": abort_reason,
                "weights_loaded": False,
            }
        )
    events_path = out_dir / "events.jsonl"
    with events_path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": status,
        "abort_reason": abort_reason,
        "caps": selected_caps.to_jsonable(),
        "assessment": assessment.to_jsonable(),
        "weights_loaded": False,
        "steps_completed": 0,
        "peak_ram_mb": 0.0,
        "avg_ram_mb": 0.0,
        "peak_io_mb_s": 0.0,
        "cpu_pct": 0.0,
        "cleanup_status": "not_required_no_weight_load",
        "events_sha256": sha256_file(events_path),
        "plan_hash": sha256_json(plan),
        "claim_boundary": (
            "A static preflight decision is safety evidence only and says nothing about checkpoint verdict quality."
        ),
    }
    dump_structured(out_dir / "summary.json", summary)
    return assessment
