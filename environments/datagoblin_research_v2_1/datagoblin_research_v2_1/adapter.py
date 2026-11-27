"""Verifiers v1 adapter for Research Verifiers v2.1 (controller-assisted & compact state)."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field, model_validator

from datagoblin.research_v2_1 import suite
from datagoblin.research_v2_1.controllers import CONTROLLER_MODES, ResearchController
from datagoblin.research_v2_1.environment import ResearchSessionV21

MAX_TURNS = 8
MAX_RESPONSE_BYTES = 16_384
REPORT_KEY = "datagoblin_research_v2_1"


class ResearchDataV21(vf.TaskData):
    """Only prompt is visible to the model; case stays on the evaluator."""

    case: dict[str, Any]


def _report(trace: vf.Trace) -> dict[str, Any]:
    value = trace.info.get(REPORT_KEY)
    return value if isinstance(value, dict) else {}


class ResearchTaskV21(vf.Task[ResearchDataV21]):
    @vf.reward(weight=1.0)
    async def verified_result(self, trace: vf.Trace) -> float:
        value = _report(trace).get("primary_score", 0.0)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, float(value)))

    @vf.metric
    async def capability_score(self, trace: vf.Trace) -> float:
        value = _report(trace).get("capability_score", 0.0)
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
        idle = ResearchSessionV21(self.data.case)
        idle_res = idle.result()
        if idle_res["primary_score"] != 0.0 or idle_res["capability_score"] != 0.0:
            return False
        reference = ResearchSessionV21(self.data.case)
        for action in suite.reference_actions(self.data.case):
            reference.step_json(json.dumps(action, allow_nan=False))
            if reference.done:
                break
        report = reference.result()
        if self.data.case["family"] in suite.LEARNERS:
            return bool(report["integrity_pass"] and report["primary_score"] > 0.0 and report["capability_score"] > 0.7)
        return bool(report["integrity_pass"] and report["primary_score"] == 1.0 and report["capability_score"] == 1.0)


class ResearchTasksetConfigV21(vf.TasksetConfig):
    suite_path: str | None = None
    split: Literal["development", "heldout"] = "development"
    seed: int = 20261127
    per_family: int = Field(1, ge=1, le=128)

    @model_validator(mode="after")
    def require_evaluation_manifest(self):
        if self.split == "heldout" and not self.suite_path:
            raise ValueError("heldout evaluation requires an operator-held suite_path")
        return self


class ResearchTasksetV21(vf.Taskset[ResearchTaskV21, ResearchTasksetConfigV21]):
    def load(self) -> list[ResearchTaskV21]:
        config = self.config
        cases = (
            suite.load_suite(config.suite_path)
            if config.suite_path
            else suite.make_suite(seed=config.seed, per_family=config.per_family, split=config.split)
        )
        if not cases or any(case["split"] != config.split for case in cases):
            raise ValueError("suite must contain cases from exactly the requested split")
        return [
            ResearchTaskV21(
                ResearchDataV21(
                    id=case["id"],
                    name=f"{case['family']} / {case['id']}",
                    prompt=case["prompt"],
                    case=case,
                ),
                config.task,
            )
            for case in cases
        ]


class ResearchEnvConfigV21(vf.EnvConfig):
    taskset: ResearchTasksetConfigV21 = ResearchTasksetConfigV21(id="datagoblin-research-v2-1")
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
    controller_mode: str = "autonomous"
    max_turns: int = Field(MAX_TURNS, ge=1, le=MAX_TURNS)
    max_response_bytes: int = Field(MAX_RESPONSE_BYTES, ge=1, le=MAX_RESPONSE_BYTES)
    wall_seconds: float = Field(300.0, gt=0, le=300.0)

    @model_validator(mode="after")
    def enforce_text_protocol(self):
        if self.agent.runtime.type != "subprocess":
            raise ValueError("this adapter uses only the local subprocess runtime")
        if self.agent.harness is None or self.agent.harness.id != "null":
            raise ValueError("the shared text-action protocol requires the null harness")
        if self.controller_mode not in CONTROLLER_MODES:
            raise ValueError(f"unsupported controller_mode: {self.controller_mode}")
        return self


class ResearchEnvV21(vf.Env[ResearchEnvConfigV21]):
    async def run(self, task: ResearchTaskV21, agents: vf.Agents) -> None:
        session = ResearchSessionV21(task.data.case)
        controller = ResearchController(self.config.controller_mode)
        async with asyncio.timeout(self.config.wall_seconds):
            async with agents.agent.interaction(task) as interaction:
                reason = "turn_limit"
                turns = 0
                interventions = 0
                try:
                    segment = await interaction.turn()
                    for turn in range(self.config.max_turns):
                        pre = controller.pre_turn_decision(
                            case=task.data.case,
                            state=session.compact_state,
                            turn_index=turn,
                            remaining=session._remaining(),
                        )
                        if pre.skip_model_call and pre.executed_text is not None:
                            interventions += 1
                            turns = turn + 1
                            observation = await asyncio.to_thread(session.step_json, pre.executed_text)
                            if session.done:
                                reason = "session_complete"
                                break
                            continue
                        if segment.terminated:
                            reason = "model_terminated"
                            break
                        turns = turn + 1
                        response = segment.last_reply or ""
                        if len(response.encode("utf-8")) > self.config.max_response_bytes:
                            reason = "response_size_limit"
                            break
                        post = controller.post_model_decision(
                            case=task.data.case,
                            state=session.compact_state,
                            turn_index=turn,
                            raw_text=response,
                            artifacts=session._artifacts,
                            remaining=session._remaining(),
                        )
                        exec_text = post.executed_text if post.executed_text is not None else response
                        if post.intervened:
                            interventions += 1
                        observation = await asyncio.to_thread(session.step_json, exec_text)
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
                        "protocol_id": "verifiers-history-v2.1",
                        "controller_mode": self.config.controller_mode,
                        "interventions": interventions,
                        "stop_reason": reason,
                        "turns": turns,
                    }
