"""Mixed DataGoblin curriculum for the Tinker Agonist.

Use existing environment-owned checks, not model judgment. All generated roots
are development fixtures; the sealed confirmation datasets remain unused.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache
from typing import Any

from datagoblin.training_gym.rlt_gym import research_world
from datagoblin.training_gym.rlt_gym import tasks as rlt

from Agonist.tinker_gym import ROLES, Task, canonical, digest, make_tasks

SOURCES = ("fixture", "mixed", "rlt", "research")
DEPTHS = (0, 1, 2, 4, 8, 12, 16)


def route_task(root: str, split: str, source: str, family: str, rng: random.Random) -> Task:
    target = rng.choice(("curriculum", "diagnose", "repair"))
    available = {name: bool(rng.getrandbits(1)) for name in ("curriculum", "diagnose", "repair")}
    authorized = bool(rng.getrandbits(1))
    selected = target if authorized and available[target] else "STOP"
    requests = {
        "curriculum": "Choose a verified candidate for the next training batch.",
        "diagnose": "Explain an observed verifier rejection using a failure code.",
        "repair": "Replace an unsuccessful candidate with a valid alternative.",
    }
    return Task(root, split, "route", {
        "source": source, "family": family, "request": requests[target],
        "available": available, "authorized": authorized,
        "rule": "Choose the matching available specialist only if authorized; otherwise STOP.",
    }, {action: float(action == selected) for action in (*available, "STOP")})


def rlt_tasks(raw: dict[str, Any], split: str, seed: int) -> list[Task]:
    """Candidate-ID tasks around the original seven RLT environment contracts.

    Every candidate is scored by the existing verifier. An oracle answer is
    inserted only if the original candidate pool has no passing answer; its
    origin is not shown to the actor. Native misses are recorded separately.
    """
    rlt.validate_task(raw)
    required_split = "train" if split == "train" else "calibration"
    if split not in ("train", "dev") or raw["split"] != required_split:
        raise ValueError("only original train/calibration roots are admitted")
    root = "rlt:" + digest(raw["lineage_id"])[:32]
    rng = random.Random(int(digest([seed, root])[:16], 16))
    public = rlt.public_task(raw)
    pool = list(raw["candidates"])
    receipts = [rlt.verify(raw, candidate) for candidate in pool]
    native_miss = not any(row["passed"] for row in receipts)
    if native_miss:
        answer = rlt.reference_answer(raw)
        receipt = rlt.verify(raw, answer)
        if not receipt["passed"]:
            raise ValueError("reference answer rejected")
        pool.append(answer)
    rng.shuffle(pool)
    choices = {f"candidate-{i}": answer for i, answer in enumerate(pool)}
    checks = {name: rlt.verify(raw, answer) for name, answer in choices.items()}
    scores = {name: float(receipt["passed"]) for name, receipt in checks.items()}
    failed = [name for name, score in scores.items() if not score]
    prior = rng.choice(failed or list(choices))
    source = {"source": "rlt", "family": raw["family"], "world": public,
              "verifier_id": checks[prior]["verifier_id"]}
    repair = Task(root, split, "repair", {
        **source, "request": "Choose a valid answer to repair the previous attempt.",
        "previous": choices[prior], "previous_code": checks[prior]["code"],
        "candidates": choices,
    }, scores)
    code = checks[prior]["code"]
    codes = sorted({code, "verified", "candidate_shape", "constraint_failed",
                    "nonminimal_route", "inactive_edge", "invalid_action",
                    "route_endpoints", "action_budget_or_shape"})
    diagnosis = Task(root, split, "diagnose", {
        **source, "request": "Classify the verifier result for this proposal.",
        "proposal": choices[prior],
        "rule": "Return verified if the candidate passes the world contract; otherwise its failure code.",
    }, {name: float(name == code) for name in codes})
    # This objective is eligibility plus serialized size, not student uplift.
    # The exact verifier is authoritative and all valid candidates beat invalid ones.
    size_scores = {
        name: 1.0 / (1.0 + len(canonical(answer))) if scores[name] else -1.0
        for name, answer in choices.items()
    }
    curriculum = Task(root, split, "curriculum", {
        **source, "goal": "Choose the shortest canonical JSON verified demonstration.",
        "candidates": choices, "cost": "Serialized character count, equal verifier policy.",
        "native_pool_miss": native_miss,
        "objective": "verified_eligibility_and_size_proxy",
    }, size_scores)
    return [curriculum, route_task(root, split, "rlt", raw["family"], rng), diagnosis, repair]


def research_tasks(seed: int, ordinal: int, split: str) -> list[Task]:
    """Select or revise a research action under the existing finite-game oracle."""
    if split not in ("train", "dev"):
        raise ValueError("invalid research split")
    rng = random.Random(int(digest(["research", seed, split, ordinal])[:16], 16))
    depth = DEPTHS[(seed + ordinal) % len(DEPTHS)]
    cell = (seed + ordinal) % (len(research_world.FAMILIES) * len(research_world.DIMENSIONS)
                              * len(research_world.HORIZONS))
    row, private = research_world.generate(depth, "train" if split == "train" else "calibration", rng, cell)
    # Never forward or retain the sealed truth/tape in the task object.
    del private
    world = row["public"]
    root = "research:" + digest(world)[:32]
    game = research_world.Game(world)
    try:
        values = game.values(game.initial)
        # Choose an actual hypothetical observation only from the public chance
        # model, rather than from hidden sampled truth.
        tests = sorted(action for action in game.legal(game.initial) if action in game.experiments)
        if not tests:
            raise ValueError("research fixture has no legal experiment")
        probe = rng.choice(tests)
        outcome = rng.randrange(2)
        revised = game.after(game.initial, probe, outcome)
        revised_values = game.values(revised)
        before = min(values, key=lambda action: (-values[action], action))
        after = min(revised_values, key=lambda action: (-revised_values[action], action))
        # Exact equality is safe here because both sets are the same oracle
        # floats. All maximizing actions are accepted by the Task grader.
        remaining = {
            "credits": revised.credits, "turns": revised.turns,
            "history": [list(item) for item in revised.history],
        }
        state = {"credits": game.initial.credits, "turns": game.initial.turns, "history": []}
        rules = (
            "Choose a legal action maximizing expected total future utility. "
            "Prior weights are unnormalized; signal_p16 and p16 are probabilities divided by 16. "
            "Each actor endorses iff incentive + trust*(2*predecessor-1) + "
            "sensor_weight*(2*private_signal-1) > 0. "
            "Experiments cost credits and one turn, and return a binary observation. "
            "Publishing pays the current correct/wrong reward; abstain pays zero. "
            "Use late_rewards once elapsed turns reach goal_change."
        )
        common = {"source": "research", "family": world["family"], "world": world,
                  "rules": rules, "objective": "exact_expected_research_utility"}
        curriculum = Task(root, split, "curriculum", {
            **common, "goal": "Choose the next experiment, publication or stop action.",
            "state": state,
        }, values)
        repair = Task(root, split, "repair", {
            **common, "request": "Replan after receiving new experiment evidence.",
            "prior_research_action": before, "state": remaining,
        }, revised_values)
        # Diagnosis targets a publicly determined decision-change event.
        # Ties in the previous decision should not force arbitrary switching.
        changed = before not in revised_values or revised_values[before] < revised_values[after]
        diagnosis = Task(root, split, "diagnose", {
            **common, "request": "Does the old optimal action remain optimal after the new evidence?",
            "prior_research_action": before, "state": remaining,
        }, {"revise": float(changed), "retain": float(not changed)})
        return [curriculum, route_task(root, split, "research", world["family"], rng), diagnosis, repair]
    finally:
        game._solve.cache_clear()


@lru_cache(maxsize=12)
def task_pool(seed: int, groups: int, split: str, source: str = "fixture") -> tuple[Task, ...]:
    """One task per role per root; no sealed tasks exposed."""
    if source not in SOURCES or split not in ("train", "dev") or not 1 <= groups <= 256 or not 0 <= seed < 2**63:
        raise ValueError("invalid task pool")
    if source == "fixture":
        return tuple(make_tasks(seed, groups, split))
    kinds = [("fixture", "rlt", "research")[i % 3] if source == "mixed" else source
             for i in range(groups)]
    fixture_count = kinds.count("fixture")
    fixture_rows = make_tasks(seed, fixture_count, split) if fixture_count else []
    rlt_count = kinds.count("rlt")
    rlt_rows = []
    if rlt_count:
        per_family = math.ceil(rlt_count / len(rlt.FAMILIES))
        raw = rlt.generate_tasks(seed=seed, per_family=per_family)
        required = "train" if split == "train" else "calibration"
        by_family = {family: [r for r in raw if r["split"] == required and r["family"] == family]
                     for family in rlt.FAMILIES}
        offset = seed % len(rlt.FAMILIES)
        families = (*rlt.FAMILIES[offset:], *rlt.FAMILIES[:offset])
        rlt_rows = [by_family[family][i] for i in range(per_family) for family in families][:rlt_count]
    counts = dict.fromkeys(SOURCES, 0)
    result = []
    for kind in kinds:
        i = counts[kind]
        counts[kind] += 1
        if kind == "fixture":
            result.extend(fixture_rows[4 * i:4 * (i + 1)])
        elif kind == "rlt":
            result.extend(rlt_tasks(rlt_rows[i], split, seed))
        else:
            result.extend(research_tasks(seed, i, split))
    if len(result) != 4 * groups or len({task.key for task in result}) != len(result):
        raise ValueError("mixed task identity collision")
    if any(sum(t.role == role for t in result) != groups for role in ROLES):
        raise ValueError("unbalanced specialist task pool")
    return tuple(result)
