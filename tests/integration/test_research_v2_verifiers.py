"""Exercise the optional Verifiers adapter with no inference provider calls."""

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("verifiers.v1")
pytest.importorskip("datagoblin_research_v2")

from datagoblin_research_v2.adapter import (
    MAX_RESPONSE_BYTES,
    REPORT_KEY,
    ResearchData,
    ResearchEnv,
    ResearchEnvConfig,
    ResearchTask,
    ResearchTaskset,
    ResearchTasksetConfig,
)
from pydantic import ValidationError
from verifiers.v1.utils.loaders import (
    environment_class,
    taskset_class,
)

from datagoblin.research_v2 import suite


class ScriptedInteraction:
    """Only the model transport is scripted; the experiment and grader stay real."""

    def __init__(self, task, replies):
        self.task = task
        self.replies = iter(replies)
        self.trace = SimpleNamespace(info={})
        self.observations = []
        self.score_at_close = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.score_at_close = await self.task.verified_result(self.trace)

    async def turn(self, observation=None):
        self.observations.append(observation)
        try:
            return SimpleNamespace(last_reply=next(self.replies), terminated=False)
        except StopIteration:
            return SimpleNamespace(last_reply="", terminated=True)


def _task(case):
    return ResearchTask(ResearchData(id=case["id"], prompt=case["prompt"], case=case))


def _run(case, replies, **settings):
    task = _task(case)
    conversation = ScriptedInteraction(task, replies)
    agents = SimpleNamespace(agent=SimpleNamespace(interaction=lambda _: conversation))
    env = ResearchEnv(ResearchEnvConfig(**settings))
    asyncio.run(env.run(task, agents))
    return conversation


def test_v041_plugin_discovery_and_public_prompt():
    assert taskset_class("datagoblin-research-v2") is ResearchTaskset
    assert environment_class("datagoblin-research-v2") is ResearchEnv
    tasks = list(ResearchTaskset(ResearchTasksetConfig()))
    assert len(tasks) == 8
    case = dict(tasks[0].data.case, private_answer="never-render-this-unique-answer")
    data = ResearchData(prompt=case["prompt"], case=case)
    assert "never-render-this-unique-answer" not in data.prompt_text


def test_v041_uses_only_text_actions_and_local_runtime():
    config = ResearchEnvConfig()
    assert config.agent.runtime.type == "subprocess"
    assert config.agent.harness.id == "null"
    assert config.agent.max_turns == 8
    assert config.agent.retries.max_retries == 0
    with pytest.raises(ValidationError, match="local subprocess"):
        ResearchEnvConfig(agent={"runtime": {"type": "prime"}})
    with pytest.raises(ValidationError, match="null harness"):
        ResearchEnvConfig(agent={"harness": {"id": "bash"}})


def test_v041_requires_explicit_private_evaluation_manifest():
    with pytest.raises(ValidationError, match="suite_path"):
        ResearchTasksetConfig(split="heldout")


def test_v041_scores_before_interaction_teardown():
    case = next(c for c in suite.make_suite() if c["family"] == "experiment_design")
    replies = [json.dumps(action) for action in suite.reference_actions(case)]
    conversation = _run(case, replies)
    report = conversation.trace.info[REPORT_KEY]
    assert conversation.score_at_close == 1.0
    assert report["primary_score"] == 1.0
    assert report["integrity_pass"] is True
    assert conversation.trace.info[REPORT_KEY + "_adapter"]["stop_reason"] == "session_complete"
    assert suite.digest({k: v for k, v in report.items() if k != "result_sha256"}) == report["result_sha256"]


def test_v041_unsubmitted_and_oversized_answers_cannot_score():
    case = suite.make_suite()[0]
    for replies, reason in [
        ([], "model_terminated"),
        (["x" * (MAX_RESPONSE_BYTES + 1)], "response_size_limit"),
        (["{}"] * 9, None),
    ]:
        conversation = _run(case, replies)
        report = conversation.trace.info[REPORT_KEY]
        assert conversation.score_at_close == 0.0
        assert report["primary_score"] == 0.0
        if reason is not None:
            assert conversation.trace.info[REPORT_KEY + "_adapter"]["stop_reason"] == reason
        assert len(conversation.observations) <= 8
