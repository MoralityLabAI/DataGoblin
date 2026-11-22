"""Bounded shared candidate rollouts and independently audited MeTTa selection."""
from __future__ import annotations

import random
from collections import Counter
from importlib.metadata import version
from itertools import combinations, product
from pathlib import Path

from MeTTa.ram_hillclimb import FAMILIES, canonical, digest, root_name
from MeTTa.ram_hillclimb_native import PACKAGE_VERSION
from MeTTa.ram_trajectory import STOP, RepairVerifier
from MeTTa.sft_oracle import edit, first_state, make_task

TRAIN_START = 61000
CURATORS = ("good", "sloppy")
RULES = {name: Path(__file__).with_name(f"{name}.metta") for name in CURATORS}


def rule_hashes(rules: dict[str, str]) -> dict[str, str]:
    """Hash the UTF-8 source submitted to both native interpreters."""
    return {name: digest(policy.encode("utf-8")) for name, policy in rules.items()}


def plan(selected: list[str], target: set[str], rng: random.Random | None = None) -> list[dict]:
    remove = sorted(set(selected) - target, reverse=True)
    add = sorted(target - set(selected))
    if rng:
        rng.shuffle(remove)
        rng.shuffle(add)
    return [
        *({"op": "remove", "id": key} for key in remove),
        *({"op": "add", "id": key} for key in add),
        STOP,
    ]


def public_facts(world: dict, initial: list[str], actions: list[dict]) -> tuple[tuple, tuple]:
    """Extract observable counts and guard facts without calling a grader."""
    items = {item["id"]: item for item in world["items"]}
    topics = set(world["goals"])
    recent = {}
    for member in items.values():
        if member["eligible"]:
            key, epoch = member["key"], member["epoch"]
            recent[key] = max(recent.get(key, 0), epoch)
    atoms = ("Eligible", "WithinBudget", "UniqueLineage", "CurrentRevision", "DependenciesMet", "ConflictFree")
    keep = [True] * len(atoms)
    selected = initial
    for action in actions:
        selected = edit(selected, action)
        if action["op"] == "remove":
            continue
        evidence = [items[key] for key in selected]
        included = set(selected)
        flags = (
            all(item["eligible"] for item in evidence),
            len(evidence) <= world["max_items"] and sum(item["cost"] for item in evidence) <= world["token_budget"],
            len({item["lineage"] for item in evidence}) == len(evidence),
            world["family"] != "revision" or (
                len({item["key"] for item in evidence}) == len(evidence)
                and all(item["epoch"] == recent.get(item["key"]) for item in evidence)
            ),
            all(included.issuperset(item["requires"]) for item in evidence),
            not any(included.issuperset(pair) for pair in world["conflicts"]),
        )
        keep = [good and flag for good, flag in zip(keep, flags, strict=True)]
    evidence = [items[key] for key in selected]
    mentions = [topic for item in evidence for topic in item["topics"] if topic in topics]
    rank = (len(set(mentions)), len(mentions), sum(item["quality"] for item in evidence),
            -sum(item["cost"] for item in evidence))
    closed = "Closed" if actions and actions[-1] == STOP else "Open"
    return (closed, *(name if good else "Fail" for name, good in zip(atoms, keep, strict=True)), *rank), tuple(selected)


def source_pool(family: str, seed: int, cap: int) -> tuple[dict, list[dict], RepairVerifier]:
    if family not in FAMILIES or seed < TRAIN_START or not 2 <= cap <= 48:
        raise ValueError("Outside registered paired TRAIN pool")
    world = make_task(family, seed)
    initial = first_state(world, seed)
    verifier = RepairVerifier(world)
    goal = set(verifier.optimum["selected"])
    distance = len(set(initial) ^ goal)
    rng = random.Random(f"paired-curation-v1/{family}/{seed}")
    endpoints = [set(initial) ^ set(delta) for delta in combinations([i["id"] for i in world["items"]], distance)]
    rng.shuffle(endpoints)
    candidates = [plan(initial, goal)]
    candidates.extend(plan(initial, endpoint) for endpoint in endpoints[:cap])
    if distance:
        for _ in range(cap):
            candidates.append(plan(initial, rng.choice(endpoints), rng))
    unique = {canonical(actions): actions for actions in candidates}
    trajectories = list(unique.values())[:cap]
    root = {
        "root": root_name(family, seed), "source_sha256": digest(world["program"].encode()),
        "family": family, "seed": seed, "initial": initial, "action_tokens_per_rollout": distance + 1,
        "candidate_count": len(trajectories), "candidate_attempts": len(candidates),
        "duplicates_removed": len(candidates) - len(unique), "over_budget": len(unique) - len(trajectories),
    }
    return root, trajectories, verifier


def curate_root(world: dict, root: dict, trajectories: list[list[dict]], rules: dict[str, str]) -> tuple[dict, list]:
    from hyperon import Environment, MeTTa

    facts = [public_facts(world, root["initial"], actions) for actions in trajectories]
    source = "\n".join(
        f"(Rollout {i} {' '.join(map(str, evidence))})" for i, (evidence, _) in enumerate(facts)
    )
    source += "\n"
    chosen = {}
    checks = []
    for name, policy in rules.items():
        runtime = MeTTa(env_builder=Environment.custom_env(create_config=False, is_test=True))
        outputs = runtime.run(source + policy)
        offers = []
        for atom in outputs[-1]:
            form, index, coverage, quality, neg_cost = str(atom).strip("()").split()
            if form != "Offer" or not 0 <= int(index) < len(trajectories):
                raise ValueError("Invalid curation offer")
            offers.append((int(index), int(coverage), int(quality), int(neg_cost)))
        winner = min(
            offers,
            key=lambda offer: (-offer[1], -offer[2], -offer[3], facts[offer[0]][1],
                               digest([root["root"], trajectories[offer[0]]])),
            default=None,
        )
        chosen[name] = winner[0] if winner is not None else None
        checks.append({
            "curator": name, "offers": len(offers), "native_input_sha256": digest((source + policy).encode()),
        })
    return chosen, checks


def audit(verifier: RepairVerifier, initial: list[str], actions: list[dict]) -> dict:
    selected = initial
    violation_types: Counter = Counter()
    unsafe = 0
    for action in actions:
        receipt = verifier.check(action, selected)
        if not receipt["structurally_applicable"]:
            raise ValueError("Rollout had an inapplicable action")
        if action["op"] == "add":
            violation_types.update(receipt["result_violations"])
            unsafe += bool(receipt["result_violations"])
        selected = receipt["selected_after"]
    target = verifier.optimum["selected"]
    terminal = selected == target
    return {
        "safe_completion": int(terminal and not unsafe and actions[-1] == STOP),
        "premature_stop": int(not terminal and actions[-1] == STOP),
        "unsafe_additions": unsafe, "violation_types": dict(violation_types),
        "remaining_edits": len(set(selected) ^ set(target)),
    }


def build_pool(roots_per_family: int, cap: int) -> tuple[list, list, dict, dict]:
    if version("hyperon") != PACKAGE_VERSION or not 1 <= roots_per_family <= 48:
        raise ValueError("Unsupported paired pool runtime or root quota")
    rules = {name: path.read_text(encoding="utf-8") for name, path in RULES.items()}
    pool, curation = [], []
    datasets: dict[str, list] = {name: [] for name in CURATORS}
    for family, seed in product(FAMILIES, range(TRAIN_START, TRAIN_START + roots_per_family)):
        root, trajectories, verifier = source_pool(family, seed, cap)
        selected, checks = curate_root(verifier.world, root, trajectories, rules)
        included = all(index is not None for index in selected.values())
        assessments = {
            name: audit(verifier, root["initial"], trajectories[index])
            for name, index in selected.items() if index is not None
        }
        if included:
            for name, index in selected.items():
                datasets[name].append((verifier.world, root["initial"], trajectories[index]))
        pool.append({**root, "rollouts": trajectories})
        curation.append({
            "root": root["root"], "source_sha256": root["source_sha256"], "selected": selected,
            "matched_quota": included, "native_checks": checks, "independent_audit": assessments,
        })
    matched = [row for row in curation if row["matched_quota"]]
    summary = {
        "unique_roots": len(pool), "rollouts": sum(row["candidate_count"] for row in pool),
        "candidate_attempts": sum(row["candidate_attempts"] for row in pool),
        "duplicates_removed": sum(row["duplicates_removed"] for row in pool),
        "over_budget": sum(row["over_budget"] for row in pool),
        "matched_roots": len(matched), "quota_shortfalls": len(pool) - len(matched),
        "selection_disagreements": sum(row["selected"]["good"] != row["selected"]["sloppy"] for row in matched),
        "rules_sha256": rule_hashes(rules),
        "audits": {},
    }
    for name in CURATORS:
        rows = [row["independent_audit"][name] for row in matched]
        violations: Counter = Counter()
        for row in rows:
            violations.update(row["violation_types"])
        summary["audits"][name] = {
            "selected_rollouts": len(rows),
            "supervised_tokens_per_pass": sum(len(actions) for _, _, actions in datasets[name]),
            **{key: sum(row[key] for row in rows) for key in ("safe_completion", "premature_stop", "unsafe_additions")},
            "violation_types": dict(violations),
        }
    return pool, curation, datasets, summary
