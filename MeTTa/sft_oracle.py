"""Deterministic *synthetic* MeTTa curation oracle for looped-model supervision.

The emitted programs are MeTTa facts plus a match query. This Python oracle
checks a separate, explicitly scoped collection problem; it does not claim to
execute Hyperon or certify arbitrary MeTTa propositions.
"""
from __future__ import annotations

import itertools
import random
from typing import Any

FAMILIES = ("lineage", "coverage", "dependency", "revision", "budget", "conflict")
OOD_FAMILY = "conflict"
MAX_CANDIDATES = 10


def make_task(family: str, seed: int) -> dict[str, Any]:
    if family not in FAMILIES or seed < 0:
        raise ValueError("unknown family or negative seed")
    rng = random.Random(f"metta-sft-v1/{family}/{seed}")
    n = rng.randint(7, 9)
    n_goals = rng.randint(3, 5)
    goals = [f"skill{j}" for j in range(n_goals)]
    items: list[dict[str, Any]] = []
    for j in range(n):
        topics = sorted(rng.sample(goals, rng.randint(1, min(3, n_goals))))
        items.append({
            "id": f"r{j}",
            "topics": topics,
            "cost": rng.randint(4, 17),
            "quality": rng.randint(2, 12),
            "eligible": rng.random() > (0.15 if family != "revision" else 0.24),
            "lineage": f"lin{rng.randrange(4)}" if family == "lineage" else f"lin{j}",
            "key": f"k{rng.randrange(4)}" if family == "revision" else f"k{j}",
            "epoch": rng.randint(1, 6) if family == "revision" else 1,
            "requires": [],
        })
    # Candidate prerequisites form a DAG; stale updates cannot act as
    # substitutes for the latest eligible version of the same key.
    if family == "dependency":
        for j in range(2, n):
            if rng.random() < 0.65:
                items[j]["requires"] = [f"r{rng.randrange(j)}"]
    conflicts: list[list[str]] = []
    if family == "conflict":
        for j in range(1, n):
            if rng.random() < 0.6:
                conflicts.append(sorted([f"r{j}", f"r{rng.randrange(j)}"]))
    budget = rng.randint(23, 47)
    if family == "budget":
        budget = rng.randint(19, 31)
    max_items = rng.randint(3, 5)
    task = {
        "schema_version": "metta-looped-curation-task-v1",
        "family": family,
        "goals": goals,
        "items": items,
        "conflicts": sorted(conflicts),
        "token_budget": budget,
        "max_items": max_items,
        "query": "!(match &self (Provides $record $skill) ($record $skill))",
        "source_kind": "synthetic_symbolic_world",
    }
    task["program"] = render_metta(task)
    validate_task(task)
    return task


def render_metta(task: dict[str, Any]) -> str:
    """Render only quoted-free, generator-owned atoms as MeTTa facts."""
    lines = [
        "; Synthetic curation world; advisory semantics, native replay pending.",
        f'(Family {task["family"]})',
        f'(Budget {task["token_budget"]})',
        f'(MaxItems {task["max_items"]})',
    ]
    lines.extend(f"(Goal {goal})" for goal in task["goals"])
    for item in task["items"]:
        name = item["id"]
        lines += [
            f"(Candidate {name})",
            f'(Tokens {name} {item["cost"]})',
            f'(Quality {name} {item["quality"]})',
            f'(Eligible {name} {"True" if item["eligible"] else "False"})',
            f'(Lineage {name} {item["lineage"]})',
            f'(Key {name} {item["key"]})',
            f'(Epoch {name} {item["epoch"]})',
        ]
        lines.extend(f"(Provides {name} {topic})" for topic in item["topics"])
        lines.extend(f"(Requires {name} {dep})" for dep in item["requires"])
    lines.extend(f"(Conflicts {a} {b})" for a, b in task["conflicts"])
    lines.append(task["query"])
    return "\n".join(lines) + "\n"


def validate_task(task: dict[str, Any]) -> None:
    if task["family"] not in FAMILIES:
        raise ValueError("invalid family")
    items = task["items"]
    if not 1 <= len(items) <= MAX_CANDIDATES:
        raise ValueError("candidate count out of bounds")
    ids = [item["id"] for item in items]
    if ids != [f"r{j}" for j in range(len(items))]:
        raise ValueError("candidate IDs must be canonical")
    if len(task["goals"]) != len(set(task["goals"])):
        raise ValueError("duplicated goals")
    if not 1 <= task["max_items"] <= len(items) or task["token_budget"] < 1:
        raise ValueError("invalid selection budget")
    for j, item in enumerate(items):
        if not (item["topics"] and set(item["topics"]) <= set(task["goals"])):
            raise ValueError("invalid topics")
        if item["cost"] <= 0 or item["quality"] < 0 or type(item["eligible"]) is not bool:
            raise ValueError("invalid candidate metrics")
        if len(set(item["requires"])) != len(item["requires"]):
            raise ValueError("duplicated dependencies")
        if any(dep not in ids[:j] for dep in item["requires"]):
            raise ValueError("dependency must reference a preceding candidate")
    if any(len(c) != 2 or c[0] not in ids or c[1] not in ids or c[0] == c[1] for c in task["conflicts"]):
        raise ValueError("invalid conflict")
    if task.get("program") != render_metta(task):
        raise ValueError("MeTTa source differs from structured task")
    if task.get("source_kind") != "synthetic_symbolic_world":
        raise ValueError("source identity mismatch")


def evaluate_subset(task: dict[str, Any], selected: list[str] | tuple[str, ...]) -> dict[str, Any]:
    ids = [item["id"] for item in task["items"]]
    by_id = {item["id"]: item for item in task["items"]}
    subset = set(selected)
    errors = []
    if len(subset) != len(selected) or not subset <= set(ids):
        errors.append("duplicate_or_unknown_candidate")
    chosen = [by_id[x] for x in ids if x in subset]
    cost = sum(item["cost"] for item in chosen)
    if len(chosen) > task["max_items"]:
        errors.append("item_limit")
    if cost > task["token_budget"]:
        errors.append("token_budget")
    if any(not item["eligible"] for item in chosen):
        errors.append("ineligible")
    if len({item["lineage"] for item in chosen}) != len(chosen):
        errors.append("repeated_lineage")
    for item in chosen:
        if not set(item["requires"]) <= subset:
            errors.append("missing_dependency")
            break
    if any(a in subset and b in subset for a, b in task["conflicts"]):
        errors.append("conflicting_sources")
    if task["family"] == "revision":
        most_recent = {}
        for item in task["items"]:
            if item["eligible"]:
                most_recent[item["key"]] = max(item["epoch"], most_recent.get(item["key"], 0))
        if any(item["epoch"] < most_recent.get(item["key"], 0) for item in chosen):
            errors.append("stale_revision")
        if len({item["key"] for item in chosen}) != len(chosen):
            errors.append("duplicate_revision_key")
    covered = sorted({topic for item in chosen for topic in item["topics"]} & set(task["goals"]))
    rank = (len(covered), sum(item["quality"] for item in chosen), -cost)
    return {
        "valid": not errors,
        "violations": sorted(set(errors)),
        "selection": sorted(subset),
        "covered_goals": covered,
        "uncovered_goals": sorted(set(task["goals"]) - set(covered)),
        "total_tokens": cost,
        "total_quality": rank[1],
        "rank": list(rank),
    }


def exact_optimum(task: dict[str, Any]) -> dict[str, Any]:
    """Enumerate all bounded subsets; objective is coverage, quality, then cost."""
    validate_task(task)
    ids = [item["id"] for item in task["items"]]
    selected: tuple[str, ...] | None = None
    metric: tuple[int, int, int] | None = None
    checked = 0
    feasible = 0
    for k in range(task["max_items"] + 1):
        for variant in itertools.combinations(ids, k):
            checked += 1
            state = evaluate_subset(task, variant)
            if not state["valid"]:
                continue
            feasible += 1
            score = tuple(state["rank"])
            if metric is None or score > metric or (score == metric and variant < selected):
                selected, metric = variant, score
    if selected is None:
        raise ValueError("no feasible subset, including empty")
    result = evaluate_subset(task, selected)
    return {
        "selected": list(selected),
        "value": result["rank"],
        "checked_subsets": checked,
        "feasible_subsets": feasible,
        "method": "exhaustive-bounded-subset-enumeration",
        "verified_objective": result,
    }


def next_edit(selected: list[str] | tuple[str, ...], target: list[str]) -> dict[str, str]:
    """A canonical edit policy. It is an oracle-supervised *target*, not ACT."""
    remove = sorted(set(selected) - set(target))
    if remove:
        return {"op": "remove", "id": remove[0]}
    add = sorted(set(target) - set(selected))
    if add:
        return {"op": "add", "id": add[0]}
    return {"op": "stop", "id": "none"}


def edit(selected: list[str] | tuple[str, ...], action: dict[str, str]) -> list[str]:
    set_ = set(selected)
    if action["op"] == "stop":
        return sorted(set_)
    if action["op"] == "remove" and action["id"] in set_:
        set_.remove(action["id"])
    elif action["op"] == "add" and action["id"] not in set_:
        set_.add(action["id"])
    else:
        raise ValueError("inapplicable edit")
    return sorted(set_)


def first_state(task: dict[str, Any], seed: int) -> list[str]:
    """Sample a public starting state independently of the hidden optimum."""
    rng = random.Random(f"metta-initial-v2/{task['family']}/{seed}")
    count = rng.randint(2, min(4, len(task["items"])))
    return sorted(rng.sample([item["id"] for item in task["items"]], count))


def replay(task: dict[str, Any], selected: list[str], optimum: list[str]) -> list[dict[str, Any]]:
    trace = []
    visited = set()
    initial_distance = len(set(selected) ^ set(optimum))
    for step in range(MAX_CANDIDATES * 2 + 1):
        signature = tuple(selected)
        if signature in visited:
            raise ValueError("repeated state in canonical edit trace")
        visited.add(signature)
        action = next_edit(selected, optimum)
        state = evaluate_subset(task, selected)
        distance = len(set(selected) ^ set(optimum))
        trace.append({
            "step": step, "selected": list(selected), "action": action,
            "remaining_canonical_edits": distance,
            "candidate_valid": state["valid"], "coverage": len(state["covered_goals"]),
            "is_fixed_point": action["op"] == "stop",
        })
        if action["op"] == "stop":
            if not state["valid"] or selected != optimum:
                raise ValueError("invalid oracle stopping target")
            break
        selected = edit(selected, action)
    else:
        raise ValueError("canonical edit trace exceeded maximum")
    if trace[0]["remaining_canonical_edits"] != initial_distance or trace[-1]["remaining_canonical_edits"] != 0:
        raise ValueError("incomplete trace")
    if any(a["remaining_canonical_edits"] != b["remaining_canonical_edits"] + 1
           for a, b in zip(trace, trace[1:])):
        raise ValueError("non-monotone canonical trace")
    return trace
