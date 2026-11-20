"""Synthetic task worlds and environment-owned checks for RLT rollout collection.

``per_family`` is the number of instances of each family in *each* split.
The public task view excludes pool candidates, provenance, and split metadata.
No solver target is stored in task records. Checks reconstruct it from the world.
"""

from __future__ import annotations

import copy
import heapq
import itertools
import json
import random
from collections import Counter, deque
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import canonical_json

SCHEMA = "rlt_gym.task.v1"
SUITE_SCHEMA = "rlt_gym.suite.v1"
GENERATOR_VERSION = "rlt_gym.synthetic.v1"
SPLITS = ("train", "calibration", "ordinary_test", "ood_test", "sealed_poison")
FAMILIES = (
    "graph_route", "arithmetic_program", "constraint_assignment", "evidence_retrieval",
    "table_transform", "experiment_plan", "finite_state_plan",
)
_TASK_KEYS = {
    "schema_version", "task_id", "family", "generator_id", "lineage_id", "split", "prompt",
    "input", "candidates", "evidence_epoch", "task_digest", "provenance",
}
_MAX_JSON_BYTES = 65_536
_MAX_SUITE_BYTES = 32 * 1024 * 1024


def _integer(value: Any, low: int = -1_000_000, high: int = 1_000_000) -> bool:
    return type(value) is int and low <= value <= high


def _json_checked(value: Any) -> str:
    def walk(item: Any, depth: int = 0) -> None:
        if depth > 16:
            raise ValueError("JSON nesting exceeds the task bound")
        if item is None or type(item) in {str, int, float, bool}:
            return
        if isinstance(item, list) and len(item) <= 2048:
            for child in item:
                walk(child, depth + 1)
            return
        if isinstance(item, dict) and len(item) <= 256 and all(type(key) is str for key in item):
            for child in item.values():
                walk(child, depth + 1)
            return
        raise ValueError("task values must be bounded plain JSON")

    walk(value)
    try:
        encoded = canonical_json(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("task values must be finite JSON") from error
    if len(encoded.encode("utf-8")) > _MAX_JSON_BYTES:
        raise ValueError("task or candidate exceeds the JSON size bound")
    return encoded


def _keys(value: Any, expected: set[str]) -> bool:
    return isinstance(value, dict) and set(value) == expected


def _strings(value: Any, *, minimum: int = 1, maximum: int = 32) -> bool:
    return (
        isinstance(value, list) and minimum <= len(value) <= maximum
        and all(type(item) is str and 0 < len(item) <= 80 for item in value)
        and len(set(value)) == len(value)
    )


def _problem_valid(family: str, world: dict[str, Any]) -> bool:
    if family == "graph_route":
        if not _keys(world, {"nodes", "edges", "start", "goal"}) or not _strings(world["nodes"], maximum=16):
            return False
        nodes, edges = world["nodes"], world["edges"]
        return (
            world["start"] in nodes and world["goal"] in nodes and world["start"] != world["goal"]
            and isinstance(edges, list) and 1 <= len(edges) <= 64
            and all(isinstance(e, list) and len(e) == 3 and e[0] in nodes and e[1] in nodes
                    and e[0] != e[1] and _integer(e[2], 1, 100) for e in edges)
            and len({tuple(edge[:2]) for edge in edges}) == len(edges)
        )
    if family == "arithmetic_program":
        if not _keys(world, {"initial", "instructions"}) or not _integer(world["initial"], -100, 100):
            return False
        steps = world["instructions"]
        return isinstance(steps, list) and 1 <= len(steps) <= 12 and all(
            _keys(step, {"op", "operand"}) and step["op"] in {"add", "subtract", "multiply", "if_even_add"}
            and _integer(step["operand"], -8, 8) for step in steps
        )
    if family == "constraint_assignment":
        if not _keys(world, {"variables", "colors", "different", "fixed", "all_different"}):
            return False
        variables, colors = world["variables"], world["colors"]
        if not _strings(variables, maximum=8) or not _integer(colors, 2, 3):
            return False
        fixed, edges, groups = world["fixed"], world["different"], world["all_different"]
        return (
            isinstance(fixed, dict) and len(fixed) <= len(variables)
            and all(key in variables and _integer(value, 0, colors - 1) for key, value in fixed.items())
            and isinstance(edges, list) and len(edges) <= 32
            and all(isinstance(edge, list) and len(edge) == 2 and edge[0] in variables and edge[1] in variables
                    and edge[0] != edge[1] for edge in edges)
            and isinstance(groups, list) and len(groups) <= 8
            and all(_strings(group, minimum=2, maximum=colors) and set(group) <= set(variables) for group in groups)
        )
    if family == "evidence_retrieval":
        if not _keys(world, {"documents", "query"}):
            return False
        documents, query = world["documents"], world["query"]
        if not isinstance(documents, list) or not 1 <= len(documents) <= 24:
            return False
        if not _keys(query, {"subject", "attribute", "epoch", "resolve_alias"}):
            return False
        if not all(type(query[key]) is str for key in ("subject", "attribute")):
            return False
        return (
            _integer(query["epoch"], 0, 1000) and type(query["resolve_alias"]) is bool
            and all(_keys(doc, {"id", "subject", "attribute", "value", "epoch", "active"})
                    and all(type(doc[key]) is str and 0 < len(doc[key]) <= 80
                            for key in ("id", "subject", "attribute", "value"))
                    and _integer(doc["epoch"], 0, 1000) and type(doc["active"]) is bool for doc in documents)
            and len({doc["id"] for doc in documents}) == len(documents)
        )
    if family == "table_transform":
        if not _keys(world, {"records", "minimum_amount", "multipliers"}):
            return False
        records, multipliers = world["records"], world["multipliers"]
        return (
            isinstance(records, list) and 1 <= len(records) <= 32 and _integer(world["minimum_amount"], -100, 100)
            and isinstance(multipliers, dict) and 1 <= len(multipliers) <= 8
            and all(type(key) is str and _integer(value, 1, 8) for key, value in multipliers.items())
            and all(_keys(row, {"id", "group", "amount", "keep"}) and type(row["id"]) is str
                    and row["group"] in multipliers and _integer(row["amount"], -100, 100)
                    and type(row["keep"]) is bool for row in records)
            and len({row["id"] for row in records}) == len(records)
        )
    if family == "experiment_plan":
        required = {"required_arms", "minimum_replicates", "max_runs", "max_tokens_per_run", "required_metrics"}
        if not _keys(world, required):
            return False
        return (
            _strings(world["required_arms"], minimum=2, maximum=6)
            and _strings(world["required_metrics"], maximum=6)
            and _integer(world["minimum_replicates"], 1, 8) and _integer(world["max_runs"], 2, 100)
            and _integer(world["max_tokens_per_run"], 16, 100_000)
        )
    if family == "finite_state_plan":
        if not _keys(world, {"capacities", "initial", "target_container", "target_amount", "max_actions"}):
            return False
        caps, initial = world["capacities"], world["initial"]
        return (
            isinstance(caps, list) and 2 <= len(caps) <= 3 and all(_integer(cap, 2, 9) for cap in caps)
            and isinstance(initial, list) and len(initial) == len(caps)
            and all(_integer(amount, 0, cap) for amount, cap in zip(initial, caps, strict=True))
            and _integer(world["target_container"], 0, len(caps) - 1)
            and _integer(world["target_amount"], 1, max(caps)) and _integer(world["max_actions"], 1, 32)
        )
    return False


def validate_task(task: dict[str, Any]) -> None:
    """Validate a bounded task and its content binding; targets remain environment-owned."""
    _json_checked(task)
    if not _keys(task, _TASK_KEYS) or task["schema_version"] != SCHEMA:
        raise ValueError("invalid RLT gym task schema")
    if task["family"] not in FAMILIES or task["split"] not in SPLITS:
        raise ValueError("unknown task family or split")
    for field in ("task_id", "generator_id", "lineage_id", "prompt"):
        if type(task[field]) is not str or not 1 <= len(task[field]) <= (6000 if field == "prompt" else 256):
            raise ValueError(f"invalid task {field}")
    if not _integer(task["evidence_epoch"], 0, 1000) or not isinstance(task["provenance"], dict):
        raise ValueError("invalid task epoch or provenance")
    try:
        valid_world = isinstance(task["input"], dict) and _problem_valid(task["family"], task["input"])
    except (KeyError, TypeError, ValueError):
        valid_world = False
    if not valid_world:
        raise ValueError("invalid or unbounded task world")
    if task["family"] == "evidence_retrieval" and task["input"]["query"]["epoch"] != task["evidence_epoch"]:
        raise ValueError("evidence query and task epoch differ")
    candidates = task["candidates"]
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 16:
        raise ValueError("candidate pool must contain 2 through 16 candidates")
    if len({canonical_json(candidate) for candidate in candidates}) != len(candidates):
        raise ValueError("duplicate pool candidate")
    expected = sha256_json({key: value for key, value in task.items() if key != "task_digest"})
    if task["task_digest"] != expected:
        raise ValueError("task content digest mismatch")


def public_task(task: dict[str, Any]) -> dict[str, Any]:
    """The model-facing world, without split, pool, seed, or target metadata."""
    validate_task(task)
    return copy.deepcopy({key: task[key] for key in ("family", "prompt", "input", "evidence_epoch")})


def _route_answer(world: dict[str, Any]) -> dict[str, Any]:
    queue: list[tuple[int, tuple[str, ...]]] = [(0, (world["start"],))]
    best: dict[str, int] = {}
    while queue:
        cost, path = heapq.heappop(queue)
        node = path[-1]
        if node in best:
            continue
        best[node] = cost
        if node == world["goal"]:
            return {"path": list(path)}
        for src, dst, weight in world["edges"]:
            if src == node and dst not in best:
                heapq.heappush(queue, (cost + weight, (*path, dst)))
    raise ValueError("graph task has no reachable goal")


def _arithmetic_answer(world: dict[str, Any]) -> dict[str, Any]:
    value = world["initial"]
    for step in world["instructions"]:
        operand = step["operand"]
        if step["op"] == "add":
            value += operand
        elif step["op"] == "subtract":
            value -= operand
        elif step["op"] == "multiply":
            value *= operand
        elif step["op"] == "if_even_add":
            value += operand if value % 2 == 0 else -operand
        else:
            raise ValueError("unknown arithmetic instruction")
    return {"value": value}


def _assignment_valid(world: dict[str, Any], assignment: Any) -> bool:
    if not _keys(assignment, set(world["variables"])):
        return False
    return (
        all(_integer(color, 0, world["colors"] - 1) for color in assignment.values())
        and all(assignment[key] == value for key, value in world["fixed"].items())
        and all(assignment[a] != assignment[b] for a, b in world["different"])
        and all(len({assignment[key] for key in group}) == len(group) for group in world["all_different"])
    )


def _constraint_answer(world: dict[str, Any]) -> dict[str, Any]:
    for colors in itertools.product(range(world["colors"]), repeat=len(world["variables"])):
        assignment = dict(zip(world["variables"], colors, strict=True))
        if _assignment_valid(world, assignment):
            return {"assignment": assignment}
    raise ValueError("constraint task is unsatisfiable")


def _evidence_answer(world: dict[str, Any]) -> dict[str, Any]:
    query = world["query"]
    live = [doc for doc in world["documents"] if doc["active"] and doc["epoch"] == query["epoch"]]
    subject = query["subject"]
    evidence: list[str] = []
    if query["resolve_alias"]:
        aliases = [doc for doc in live if doc["subject"] == subject and doc["attribute"] == "alias_of"]
        if len(aliases) != 1:
            raise ValueError("evidence task must have one live alias")
        subject = aliases[0]["value"]
        evidence.append(aliases[0]["id"])
    facts = [doc for doc in live if doc["subject"] == subject and doc["attribute"] == query["attribute"]]
    if len(facts) != 1:
        raise ValueError("evidence task must have one live target fact")
    evidence.append(facts[0]["id"])
    return {"value": facts[0]["value"], "evidence_ids": sorted(evidence)}


def _table_answer(world: dict[str, Any]) -> dict[str, Any]:
    totals: dict[str, list[int]] = {}
    for row in world["records"]:
        if row["keep"] and row["amount"] >= world["minimum_amount"]:
            group = row["group"]
            current = totals.setdefault(group, [0, 0])
            current[0] += row["amount"] * world["multipliers"][group]
            current[1] += 1
    return {"rows": [{"group": group, "total": pair[0], "count": pair[1]} for group, pair in sorted(totals.items())]}


def _experiment_answer(world: dict[str, Any]) -> dict[str, Any]:
    return {
        "arms": sorted(world["required_arms"]), "replicates": world["minimum_replicates"],
        "tokens_per_run": world["max_tokens_per_run"], "training_split": "train",
        "selection_split": "calibration", "confirmation_split": "ordinary_test",
        "independent_verifier": True, "metrics": sorted(world["required_metrics"]),
    }


def _jug_step(state: tuple[int, ...], caps: list[int], action: str) -> tuple[int, ...] | None:
    parts = action.split(":")
    if len(parts) not in {2, 3} or not all(part.isdigit() and len(part) == 1 for part in parts[1:]):
        return None
    indices = [int(part) for part in parts[1:]]
    if any(index >= len(caps) for index in indices):
        return None
    result = list(state)
    source = indices[0]
    if parts[0] == "fill" and len(parts) == 2:
        result[source] = caps[source]
    elif parts[0] == "empty" and len(parts) == 2:
        result[source] = 0
    elif parts[0] == "pour" and len(parts) == 3 and indices[0] != indices[1]:
        destination = indices[1]
        amount = min(result[source], caps[destination] - result[destination])
        result[source] -= amount
        result[destination] += amount
    else:
        return None
    return tuple(result)


def _jug_answer(world: dict[str, Any]) -> dict[str, Any]:
    caps = world["capacities"]
    actions = [f"{op}:{index}" for op in ("fill", "empty") for index in range(len(caps))]
    actions += [f"pour:{i}:{j}" for i in range(len(caps)) for j in range(len(caps)) if i != j]
    initial = tuple(world["initial"])
    queue: deque[tuple[tuple[int, ...], list[str]]] = deque([(initial, [])])
    seen = {initial}
    while queue:
        state, path = queue.popleft()
        if state[world["target_container"]] == world["target_amount"]:
            return {"actions": path}
        if len(path) == world["max_actions"]:
            continue
        for action in actions:
            updated = _jug_step(state, caps, action)
            if updated is not None and updated not in seen:
                seen.add(updated)
                queue.append((updated, [*path, action]))
    raise ValueError("finite-state goal is unreachable within the action budget")


_SOLVERS = {
    "graph_route": _route_answer, "arithmetic_program": _arithmetic_answer,
    "constraint_assignment": _constraint_answer, "evidence_retrieval": _evidence_answer,
    "table_transform": _table_answer, "experiment_plan": _experiment_answer, "finite_state_plan": _jug_answer,
}


def reference_answer(task: dict[str, Any]) -> dict[str, Any]:
    """Explicit exact-solver control. Learned and teacher adapters must not call this."""
    validate_task(task)
    return _SOLVERS[task["family"]](task["input"])


def _check_candidate(family: str, world: dict[str, Any], candidate: Any) -> tuple[bool, str]:
    if family == "graph_route":
        if not _keys(candidate, {"path"}) or not _strings(candidate["path"], minimum=2, maximum=len(world["nodes"])):
            return False, "candidate_shape"
        path = candidate["path"]
        if path[0] != world["start"] or path[-1] != world["goal"]:
            return False, "route_endpoints"
        edges = {(src, dst): weight for src, dst, weight in world["edges"]}
        hops = list(itertools.pairwise(path))
        if any(hop not in edges for hop in hops):
            return False, "inactive_edge"
        optimum = _route_answer(world)["path"]
        minimum = sum(edges[hop] for hop in itertools.pairwise(optimum))
        valid = sum(edges[hop] for hop in hops) == minimum
        return valid, "verified" if valid else "nonminimal_route"
    if family == "arithmetic_program":
        if not _keys(candidate, {"value"}) or type(candidate["value"]) is not int:
            return False, "candidate_shape"
        valid = candidate["value"] == _arithmetic_answer(world)["value"]
    elif family == "constraint_assignment":
        valid = _keys(candidate, {"assignment"}) and _assignment_valid(world, candidate["assignment"])
    elif family == "evidence_retrieval":
        if not _keys(candidate, {"value", "evidence_ids"}) or type(candidate["value"]) is not str:
            return False, "candidate_shape"
        if not _strings(candidate["evidence_ids"], maximum=2):
            return False, "candidate_shape"
        expected = _evidence_answer(world)
        valid = (
            candidate["value"] == expected["value"]
            and sorted(candidate["evidence_ids"]) == expected["evidence_ids"]
        )
    elif family == "table_transform":
        if not _keys(candidate, {"rows"}) or not isinstance(candidate["rows"], list):
            return False, "candidate_shape"
        valid = canonical_json(candidate) == canonical_json(_table_answer(world))
    elif family == "experiment_plan":
        expected = _experiment_answer(world)
        if not _keys(candidate, set(expected)):
            return False, "candidate_shape"
        valid = (
            _strings(candidate["arms"], maximum=8) and set(candidate["arms"]) == set(world["required_arms"])
            and _integer(candidate["replicates"], world["minimum_replicates"], 100)
            and len(candidate["arms"]) * candidate["replicates"] <= world["max_runs"]
            and _integer(candidate["tokens_per_run"], 1, world["max_tokens_per_run"])
            and candidate["training_split"] == "train" and candidate["selection_split"] == "calibration"
            and candidate["confirmation_split"] == "ordinary_test" and candidate["independent_verifier"] is True
            and _strings(candidate["metrics"], maximum=8)
            and set(candidate["metrics"]) == set(world["required_metrics"])
        )
    elif family == "finite_state_plan":
        if not _keys(candidate, {"actions"}) or not isinstance(candidate["actions"], list):
            return False, "candidate_shape"
        if len(candidate["actions"]) > world["max_actions"] or any(type(a) is not str for a in candidate["actions"]):
            return False, "action_budget_or_shape"
        state = tuple(world["initial"])
        for action in candidate["actions"]:
            updated = _jug_step(state, world["capacities"], action)
            if updated is None:
                return False, "invalid_action"
            state = updated
        valid = state[world["target_container"]] == world["target_amount"]
    else:
        raise ValueError("unknown task family")
    return bool(valid), "verified" if valid else "constraint_failed"


def verify(task: dict[str, Any], candidate: Any) -> dict[str, Any]:
    """Recheck the proposal against its world and bind the receipt to this epoch."""
    validate_task(task)
    _json_checked(candidate)
    valid, code = _check_candidate(task["family"], task["input"], candidate)
    return {
        "status": "passed" if valid else "failed", "passed": valid, "code": code,
        "task_digest": task["task_digest"], "candidate_digest": sha256_json(candidate),
        "evidence_epoch": task["evidence_epoch"], "verifier_id": f"rlt_gym.{task['family']}.v1",
    }


def _world(family: str, rng: random.Random, ood: bool, epoch: int) -> tuple[dict[str, Any], str, str]:
    salt = f"{rng.getrandbits(48):012x}"
    if family == "graph_route":
        nodes = [f"n{salt[-4:]}_{i}" for i in range(7 if ood else 5)]
        edges = [[nodes[i], nodes[i + 1], rng.randint(1, 4)] for i in range(len(nodes) - 1)]
        edges += [[nodes[0], nodes[-1], 30], [nodes[1], nodes[-1], 20]]
        if ood:
            edges += [[nodes[0], nodes[3], 3], [nodes[4], nodes[2], 1], [nodes[2], nodes[5], 3]]
        rng.shuffle(edges)
        return (
            {"nodes": nodes, "edges": edges, "start": nodes[0], "goal": nodes[-1]},
            "Find a minimum-cost directed route using only the currently active edges. Each edge is [from,to,cost]. "
            'Return {"path":["start",...,"goal"]}. The current evidence epoch replaces all earlier edge lists.',
            "branch_cycles" if ood else "chain_shortcut",
        )
    if family == "arithmetic_program":
        operations = ["add", "subtract", "multiply"] + (["if_even_add"] if ood else [])
        instructions = [{"op": rng.choice(operations), "operand": rng.randint(-5, 5)} for _ in range(8 if ood else 4)]
        if ood:
            instructions[1]["op"] = "if_even_add"
        return (
            {"initial": rng.randint(-40, 40), "instructions": instructions},
            "Execute the instructions in order on the integer initial value. if_even_add adds its operand when the "
            'current value is even and subtracts it when odd. Return {"value":integer}.',
            "conditional_program" if ood else "straight_program",
        )
    if family == "constraint_assignment":
        variables = [f"v{salt[-3:]}_{i}" for i in range(7 if ood else 5)]
        planted = [i % 3 for i in range(len(variables))]
        edges = [[variables[i], variables[i + 1]] for i in range(len(variables) - 1)]
        fixed = {variables[0]: 0, variables[-1]: planted[-1]}
        return (
            {"variables": variables, "colors": 3, "different": edges, "fixed": fixed,
             "all_different": [variables[:3], variables[3:6]] if ood else []},
            "Assign every variable an integer color from 0 through colors-1, obeying different pairs, fixed values, "
            'and all_different groups. Return {"assignment":{"variable":integer,...}} '
            "with exactly the listed variables.",
            "hyperedge_coloring" if ood else "pinned_chain_coloring",
        )
    if family == "evidence_retrieval":
        subject, alias = f"sample_{salt}", f"label_{salt}"
        values = rng.sample(["red", "blue", "green", "amber", "violet", "silver"], 4)
        docs = [
            {"id": f"d{salt}_{i}", "subject": subject, "attribute": "color", "value": value,
             "epoch": epoch if i != 1 else epoch - 1, "active": i < 2}
            for i, value in enumerate(values[:3])
        ]
        docs.append({"id": f"d{salt}_3", "subject": f"other_{salt}", "attribute": "color",
                     "value": values[3], "epoch": epoch, "active": True})
        if ood:
            docs.append({"id": f"d{salt}_4", "subject": alias, "attribute": "alias_of", "value": subject,
                         "epoch": epoch, "active": True})
        rng.shuffle(docs)
        return (
            {"documents": docs, "query": {"subject": alias if ood else subject, "attribute": "color",
                                          "epoch": epoch, "resolve_alias": ood}},
            "Use only active documents in the query epoch. Resolve alias_of once when requested, then answer the "
            "attribute query. Cite exactly the live fact and, when used, the live alias document. "
            'Return {"value":"text","evidence_ids":["id",...]}.',
            "alias_join" if ood else "current_fact",
        )
    if family == "table_transform":
        groups = [f"g{salt[-3:]}_{i}" for i in range(3)]
        rows = [{"id": f"r{salt}_{i}", "group": rng.choice(groups), "amount": rng.randint(-5, 20),
                 "keep": rng.random() < 0.7} for i in range(12 if ood else 7)]
        rows[0].update(amount=15, keep=True)
        return (
            {"records": rows, "minimum_amount": rng.randint(0, 8),
             "multipliers": {group: rng.randint(2, 5) if ood else 1 for group in groups}},
            "Keep records whose keep flag is true and whose amount is at least minimum_amount. Multiply each retained "
            "amount by its group multiplier. Sum by group and count retained records, omit empty groups, sort groups "
            'alphabetically. Return {"rows":[{"group":"name","total":integer,"count":integer},...]}.',
            "weighted_group_join" if ood else "filtered_group_sum",
        )
    if family == "experiment_plan":
        arms = ["direct", "answer_rlt", "matched_damping"] + (["norm_matched_sham"] if ood else [])
        replicates = rng.randint(2, 5)
        return (
            {"required_arms": arms, "minimum_replicates": replicates,
             "max_runs": len(arms) * replicates + rng.randint(0, 3),
             "max_tokens_per_run": rng.randint(32, 512) * 16,
             "required_metrics": ["verified_success", "calls", "regressions"]
             + (["intervention_effect"] if ood else [])},
            "Build a bounded experiment plan containing exactly the required arms and metrics. Use the minimum "
            "replicates, obey max_runs and max_tokens_per_run, train on train, select on calibration, and confirm on "
            "ordinary_test with an independent verifier. Return an object with arms (list), replicates (integer), "
            "tokens_per_run (positive integer), training_split, selection_split, confirmation_split, "
            "independent_verifier (boolean), and metrics (list).",
            "intervention_control_plan" if ood else "feedback_control_plan",
        )
    capacities = list(rng.choice([(3, 5), (4, 7), (5, 8), (2, 5), (5, 7)]))
    if ood:
        capacities.append(rng.randint(3, 8))
    target_container = rng.randrange(len(capacities))
    target_amount = rng.randint(1, capacities[target_container] - 1)
    initial = [rng.randint(0, capacity) for capacity in capacities]
    if initial[target_container] == target_amount:
        initial[target_container] = 0
    return (
        {"capacities": capacities, "initial": initial, "target_container": target_container,
         "target_amount": target_amount, "max_actions": 24},
        "Reach target_amount in target_container, staying within max_actions. Containers start at initial amounts. "
        "A fill:i action fills container i; empty:i empties it; pour:i:j transfers until i is empty or j is full. "
        'Return {"actions":["fill:0","pour:0:1",...]}. Containers use zero-based indices.',
        "three_container_plan" if ood else "two_container_plan",
    )


def _pool(family: str, world: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    good = _SOLVERS[family](world)
    if family == "graph_route":
        candidates = [good, {"path": [world["start"], world["goal"]]},
                      {"path": list(reversed(good["path"]))}, {"path": good["path"][:-1]}]
    elif family == "arithmetic_program":
        candidates = [{"value": good["value"] + delta} for delta in (0, -1, 1, 2)]
    elif family == "constraint_assignment":
        candidates = [good]
        for delta in (1, 2):
            bad = copy.deepcopy(good)
            key = next(iter(world["fixed"]))
            bad["assignment"][key] = (bad["assignment"][key] + delta) % world["colors"]
            candidates.append(bad)
        bad = copy.deepcopy(good)
        bad["assignment"].pop(world["variables"][-1])
        candidates.append(bad)
    elif family == "evidence_retrieval":
        candidates = [good, {**good, "value": "unknown"}, {**good, "evidence_ids": []}]
        wrong = next(doc["id"] for doc in world["documents"] if doc["id"] not in good["evidence_ids"])
        candidates.append({**good, "evidence_ids": [wrong]})
    elif family == "table_transform":
        candidates = [good, {"rows": []}]
        bad = copy.deepcopy(good)
        bad["rows"][0]["total"] += 1
        candidates.append(bad)
        bad = copy.deepcopy(good)
        bad["rows"][0]["count"] += 1
        candidates.append(bad)
    elif family == "experiment_plan":
        candidates = [good, {**good, "independent_verifier": False},
                      {**good, "selection_split": "ordinary_test"},
                      {**good, "tokens_per_run": world["max_tokens_per_run"] + 1}]
    else:
        candidates = [good, {"actions": []}, {"actions": good["actions"][:-1]},
                      {"actions": ["empty:0"] * len(good["actions"])}]
    unique = {canonical_json(item): item for item in candidates}
    result = list(unique.values())
    rng.shuffle(result)
    return result


def generate_tasks(seed: int = 7, per_family: int = 4) -> list[dict[str, Any]]:
    """Generate instance-disjoint worlds; OOD uses different generator structures."""
    if not _integer(seed, 0, 2**63 - 1) or not _integer(per_family, 1, 128):
        raise ValueError("seed must be a nonnegative integer and per_family must be 1 through 128")
    tasks: list[dict[str, Any]] = []
    problem_digests: set[str] = set()
    for split, family, index in itertools.product(SPLITS, FAMILIES, range(per_family)):
        rng = random.Random(int(sha256_json([GENERATOR_VERSION, seed, split, family, index]), 16))
        ood = split in {"ood_test", "sealed_poison"}
        for _ in range(512):
            epoch = rng.randint(1, 999)
            world, prompt, variant = _world(family, rng, ood, epoch)
            problem_digest = sha256_json({"family": family, "input": world})
            if problem_digest not in problem_digests:
                break
        else:
            raise ValueError(f"could not obtain a fresh {family} world")
        problem_digests.add(problem_digest)
        lineage = f"lineage:{problem_digest}"
        task = {
            "schema_version": SCHEMA, "task_id": f"{family}:{problem_digest[:20]}", "family": family,
            "generator_id": f"{GENERATOR_VERSION}/{family}/{variant}", "lineage_id": lineage,
            "split": split, "prompt": prompt, "input": world, "candidates": _pool(family, world, rng),
            "evidence_epoch": epoch, "provenance": {
                "kind": "synthetic_environment", "source": "DataGoblin/rlt_gym",
                "generator_version": GENERATOR_VERSION, "rights": "project_generated",
            },
        }
        task["task_digest"] = sha256_json(task)
        validate_task(task)
        tasks.append(task)
    _validate_suite(tasks)
    return tasks


def _validate_suite(tasks: list[dict[str, Any]]) -> None:
    if not tasks or len(tasks) > len(FAMILIES) * len(SPLITS) * 128:
        raise ValueError("invalid suite size")
    ids: set[str] = set()
    lineages: dict[str, str] = {}
    worlds: dict[str, str] = {}
    development_generators: set[str] = set()
    ood_generators: set[str] = set()
    for task in tasks:
        validate_task(task)
        if task["task_id"] in ids:
            raise ValueError("duplicate task id")
        ids.add(task["task_id"])
        for key, mapping in ((task["lineage_id"], lineages), (sha256_json([task["family"], task["input"]]), worlds)):
            if key in mapping and mapping[key] != task["split"]:
                raise ValueError("task lineage or world overlaps splits")
            mapping[key] = task["split"]
        if task["split"] in {"train", "calibration", "ordinary_test"}:
            development_generators.add(task["generator_id"])
        if task["split"] == "ood_test":
            ood_generators.add(task["generator_id"])
    if development_generators & ood_generators:
        raise ValueError("OOD generators overlap development")


def write_suite(out: Path, *, seed: int = 7, per_family: int = 4) -> Path:
    """Freeze a suite in a fresh directory. Existing evidence is never overwritten."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"suite output already exists: {out}")
    tasks = generate_tasks(seed=seed, per_family=per_family)
    out.mkdir(parents=True)
    task_path = out / "tasks.jsonl"
    task_path.write_text("".join(canonical_json(task) + "\n" for task in tasks), encoding="utf-8")
    manifest = {
        "schema_version": SUITE_SCHEMA, "generator_version": GENERATOR_VERSION,
        "generator_source_sha256": sha256_file(Path(__file__)), "tasks_file": task_path.name,
        "tasks_sha256": sha256_file(task_path), "task_count": len(tasks),
        "counts_by_family": dict(sorted(Counter(task["family"] for task in tasks).items())),
        "counts_by_split": dict(sorted(Counter(task["split"] for task in tasks).items())),
        "seed_commitment": sha256_json({"generator": GENERATOR_VERSION, "seed": seed}),
        "split_policy": {
            "splits": list(SPLITS), "lineage_disjoint": True, "ood_generators_disjoint": True,
            "teacher_export_splits": ["train", "calibration"], "training_export_splits": ["train"],
        },
        "claim_scope": (
            "Synthetic engineering and controlled research tasks; external benchmark performance is unmeasured."
        ),
    }
    manifest["manifest_digest"] = sha256_json(manifest)
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest_path


def load_suite(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load frozen task bytes and validate bindings and the split firewall."""
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError("suite manifest exceeds size bound")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SUITE_SCHEMA:
        raise ValueError("invalid suite manifest")
    expected = sha256_json({key: value for key, value in manifest.items() if key != "manifest_digest"})
    if manifest.get("manifest_digest") != expected:
        raise ValueError("suite manifest digest mismatch")
    if manifest.get("tasks_file") != "tasks.jsonl":
        raise ValueError("suite task path must be tasks.jsonl")
    task_path = path.parent / "tasks.jsonl"
    if task_path.is_symlink() or task_path.stat().st_size > _MAX_SUITE_BYTES:
        raise ValueError("suite task file is outside bounds")
    if sha256_file(task_path) != manifest.get("tasks_sha256"):
        raise ValueError("suite task-file digest mismatch")
    tasks = [json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _validate_suite(tasks)
    if len(tasks) != manifest.get("task_count"):
        raise ValueError("suite task count mismatch")
    if dict(Counter(task["family"] for task in tasks)) != manifest.get("counts_by_family"):
        raise ValueError("suite family counts mismatch")
    if dict(Counter(task["split"] for task in tasks)) != manifest.get("counts_by_split"):
        raise ValueError("suite split counts mismatch")
    return manifest, tasks
