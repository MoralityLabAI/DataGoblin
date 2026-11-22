"""Public curation tasks with an independently replayed Prime Verifiers reward."""
from __future__ import annotations

import json
from itertools import product
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field

from MeTTa.airis02_student import action_index
from MeTTa.ram_hillclimb import FAMILIES, OOD_FAMILY, canonical, digest, root_name
from MeTTa.ram_trajectory import STOP, RepairVerifier
from MeTTa.sft_oracle import first_state, make_task, next_edit

BENCHMARK_ID = "metta-research-bench"
MAX_DECISIONS = 10
SPLITS = {
    "development": (FAMILIES, 34000),
    "frozen": (FAMILIES, 51000),
    "conflict": ((OOD_FAMILY,), 52000),
    "looped_dev": (FAMILIES, 62000),
    "looped_frozen": (FAMILIES, 63000),
    "looped_conflict": ((OOD_FAMILY,), 64000),
    "looped_ablation_frozen": (FAMILIES, 65000),
    "looped_ablation_conflict": ((OOD_FAMILY,), 66000),
}
MAX_REPLY_BYTES = 2048


class CurationData(vf.TaskData):
    family: str
    source_seed: int
    source_sha256: str
    split: str


def submitted_actions(reply: str) -> list[dict[str, str]] | None:
    if len(reply.encode()) > MAX_REPLY_BYTES:
        return None
    try:
        actions = json.loads(reply)
    except (ValueError, RecursionError):
        return None
    if not isinstance(actions, list) or len(actions) > MAX_DECISIONS:
        return None
    for i, action in enumerate(actions):
        if not isinstance(action, dict) or set(action) != {"op", "id"}:
            return None
        if not all(isinstance(value, str) for value in action.values()):
            return None
        try:
            action_index(action)
        except ValueError:
            return None
        if action == STOP and i != len(actions) - 1:
            return None
    return actions


def grade(data: CurationData, reply: str) -> dict:
    """Independently reconstruct the world and replay only submitted actions."""
    metrics = dict(
        format_valid=0, exact_first_edit=0, solved_and_stopped=0, premature_stop=0,
        invalid_actions=0, unsafe_additions=0, new_violation_additions=0, replayed_actions=0,
        grading_exact_enumerations=0, grading_constraint_checks=0, reward=0,
    )
    actions = submitted_actions(reply)
    if actions is None:
        return metrics
    world = make_task(data.family, data.source_seed)
    if digest(world["program"].encode()) != data.source_sha256:
        raise ValueError("Evaluation world differs from its pinned source hash")
    verifier = RepairVerifier(world)
    target = verifier.optimum["selected"]
    current = first_state(world, data.source_seed)
    metrics["format_valid"] = metrics["grading_exact_enumerations"] = 1
    metrics["exact_first_edit"] = int(bool(actions) and actions[0] == next_edit(current, target))
    for action in actions:
        receipt = verifier.check(action, current)
        if not receipt["structurally_applicable"]:
            metrics["invalid_actions"] += 1
            break
        if action == STOP:
            metrics["solved_and_stopped"] = int(receipt["correct_stop"])
            metrics["premature_stop"] = int(not receipt["correct_stop"])
            break
        if action["op"] == "add":
            metrics["unsafe_additions"] += bool(receipt["result_violations"])
            metrics["new_violation_additions"] += bool(receipt["new_violations"])
        metrics["replayed_actions"] += 1
        current = receipt["selected_after"]
    metrics["grading_constraint_checks"] = verifier.constraint_checks
    metrics["remaining_edits"] = len(set(current) ^ set(target))
    metrics["reward"] = int(bool(metrics["solved_and_stopped"]) and not metrics["unsafe_additions"])
    return metrics


class CurationTask(vf.Task[CurationData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.metric
    async def replay_metrics(self, trace: vf.Trace) -> dict[str, float]:
        result = grade(self.data, trace.last_reply)
        trace.info["curation_grade"] = result
        return {name: float(value) for name, value in result.items() if name != "reward"}

    @vf.reward(weight=1.0)
    async def verified_safe_completion(self, trace: vf.Trace) -> float:
        return float(trace.info["curation_grade"]["reward"])

    async def validate(self) -> bool:
        world = make_task(self.data.family, self.data.source_seed)
        verifier = RepairVerifier(world)
        initial = first_state(world, self.data.source_seed)
        target = verifier.optimum["selected"]
        # Clearing dependents first avoids additions while a retained record
        # is still missing a prerequisite. The synthetic 4 + 5 + STOP cap fits.
        actions = [
            *({"op": "remove", "id": key} for key in reversed(initial)),
            *({"op": "add", "id": key} for key in target),
            STOP,
        ]
        return bool(grade(self.data, canonical(actions))["reward"])


class BenchConfig(vf.TasksetConfig):
    id: str = BENCHMARK_ID
    split: Literal[
        "development", "frozen", "conflict", "looped_dev", "looped_frozen", "looped_conflict",
        "looped_ablation_frozen", "looped_ablation_conflict",
    ] = "development"
    roots_per_family: int = Field(default=8, ge=1, le=48)


class MeTTaResearchBench(vf.Taskset[CurationTask, BenchConfig]):
    """Deterministic public worlds with independent private target computation at grading."""

    def load(self):
        families, first = SPLITS[self.config.split]
        for family, seed in product(families, range(first, first + self.config.roots_per_family)):
            world = make_task(family, seed)
            selected = canonical(first_state(world, seed))
            prompt = (
                "Repair the evidence selection using the public MeTTa facts. "
                "Maximize goal coverage, then total quality, then minimize token cost. "
                "Break exact ties by the lexicographically first sorted list of record IDs. "
                "Respect eligibility, budget, size, lineage, dependencies, revision, and conflict constraints. "
                f"Return a JSON array of at most {MAX_DECISIONS} actions, including the final STOP. "
                'Each action uses {"op":"add"|"remove","id":"rN"} or {"op":"stop","id":"none"}. '
                "An add must leave a feasible selection.\n"
                f"Initial selection: {selected}\n\n" + world["program"]
            )
            yield CurationTask(CurationData(
                id=root_name(family, seed), prompt=prompt, family=family, source_seed=seed,
                source_sha256=digest(world["program"].encode()), split=self.config.split,
            ), self.config.task)
