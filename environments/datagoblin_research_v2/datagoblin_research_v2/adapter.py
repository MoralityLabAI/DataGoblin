"""A text-action adapter with evaluator-owned experiments and scoring.

The agent receives public observations through ordinary conversation turns.
JSON actions are interpreted by ResearchSession, which owns every experiment.
This keeps native function-calling support out of the comparison contract.

API source: PrimeIntellect-ai/verifiers at
540f40faab8c376b85c6afa2f872683b0ed1d4d1 (release 0.4.1).
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field, model_validator

from datagoblin.research_v2 import suite
from datagoblin.research_v2.environment import ResearchSession

MAX_TURNS = 8
MAX_RESPONSE_BYTES = 16_384
REPORT_KEY = "datagoblin_research_v2"


class ResearchData(vf.TaskData):
    """Only prompt is visible to the model; case stays on the evaluator."""

    case: dict[str, Any]


def _report(trace: vf.Trace) -> dict[str, Any]:
    value = trace.info.get(REPORT_KEY)
    return value if isinstance(value, dict) else {}


class ResearchTask(vf.Task[ResearchData]):
    @vf.reward(weight=1.0)
    async def verified_result(self, trace: vf.Trace) -> float:
        """The environment's final score; submitted prose cannot set this field."""
        value = _report(trace).get("primary_score", 0.0)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, float(value)))

    @vf.metric
    async def integrity_pass(self, trace: vf.Trace) -> float:
        return float(_report(trace).get("integrity_pass") is True)

    @vf.metric
    async def success(self, trace: vf.Trace) -> float:
        return float(_report(trace).get("success") is True)

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Run the development reference trajectory and check no-op scoring."""
        if self.data.case["split"] != "development":
            raise ValueError("vf-validate gold checks require development tasks")
        idle = ResearchSession(self.data.case)
        if idle.result()["primary_score"] != 0.0:
            return False
        reference = ResearchSession(self.data.case)
        for action in suite.reference_actions(self.data.case):
            reference.step_json(json.dumps(action, allow_nan=False))
            if reference.done:
                break
        report = reference.result()
        if self.data.case["family"] in suite.LEARNERS:
            return bool(report["integrity_pass"] and report["primary_score"] > 0.0)
        return bool(report["integrity_pass"] and report["primary_score"] == 1.0)


class ResearchTasksetConfig(vf.TasksetConfig):
    suite_path: str | None = None
    split: Literal["development", "heldout"] = "development"
    seed: int = 20261126
    per_family: int = Field(1, ge=1, le=128)

    @model_validator(mode="after")
    def require_evaluation_manifest(self):
        if self.split == "heldout" and not self.suite_path:
            raise ValueError("heldout evaluation requires an operator-held suite_path")
        return self


class ResearchTaskset(vf.Taskset[ResearchTask, ResearchTasksetConfig]):
    def load(self) -> list[ResearchTask]:
        config = self.config
        cases = (
            suite.load_suite(config.suite_path)
            if config.suite_path
            else suite.make_suite(
                seed=config.seed, per_family=config.per_family, split=config.split
            )
        )
        if not cases or any(case["split"] != config.split for case in cases):
            raise ValueError("suite must contain cases from exactly the requested split")
        return [
            ResearchTask(
                ResearchData(
                    id=case["id"],
                    name=f"{case['family']} / {case['id']}",
                    prompt=case["prompt"],
                    case=case,
                ),
                config.task,
            )
            for case in cases
        ]


class ResearchEnvConfig(vf.EnvConfig):
    taskset: ResearchTasksetConfig = ResearchTasksetConfig(id="datagoblin-research-v2")
    agent: vf.AgentConfig = vf.AgentConfig(
        harness={"id": "null"},
        runtime={"type": "subprocess"},
        max_turns=MAX_TURNS,
        max_turn_tokens=1024,
        max_output_tokens=8192,
        max_total_tokens=32768,
        timeout={"setup": 60, "rollout": 300, "finalize": 15, "scoring": 30},
        retries={"max_retries": 0},
    )
    max_turns: int = Field(MAX_TURNS, ge=1, le=MAX_TURNS)
    max_response_bytes: int = Field(MAX_RESPONSE_BYTES, ge=1, le=MAX_RESPONSE_BYTES)
    wall_seconds: float = Field(300.0, gt=0, le=300.0)

    @model_validator(mode="after")
    def enforce_text_protocol(self):
        if self.agent.runtime.type != "subprocess":
            raise ValueError("this adapter uses only the local subprocess runtime")
        if self.agent.harness is None or self.agent.harness.id != "null":
            raise ValueError("the shared text-action protocol requires the null harness")
        return self


class ResearchEnv(vf.Env[ResearchEnvConfig]):
    async def run(self, task: ResearchTask, agents: vf.Agents) -> None:
        session = ResearchSession(task.data.case)
        async with asyncio.timeout(self.config.wall_seconds):
            async with agents.agent.interaction(task) as interaction:
                reason = "turn_limit"
                turns = 0
                try:
                    segment = await interaction.turn()
                    for turn in range(self.config.max_turns):
                        if segment.terminated:
                            reason = "model_terminated"
                            break
                        turns = turn + 1
                        response = segment.last_reply or ""
                        if len(response.encode("utf-8")) > self.config.max_response_bytes:
                            reason = "response_size_limit"
                            break
                        observation = await asyncio.to_thread(session.step_json, response)
                        if session.done:
                            reason = "session_complete"
                            break
                        if turns < self.config.max_turns:
                            segment = await interaction.turn(
                                json.dumps(observation, sort_keys=True, allow_nan=False)
                            )
                finally:
                    interaction.trace.info[REPORT_KEY] = session.result()
                    interaction.trace.info[REPORT_KEY + "_adapter"] = {
                        "protocol_id": "verifiers-history-v2",
                        "stop_reason": reason,
                        "turns": turns,
                    }
