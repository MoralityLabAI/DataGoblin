"""Bounded curation dockets with host-owned evidence checks and native MeTTa proofs."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import cache
from itertools import permutations, product
from math import prod
from pathlib import Path

from .evidence_budget import CURRENT, REJECTED, UNCHECKED, WIDTH, State, written
from .evidence_budget_confirmation import relabel
from .evidence_budget_pilot import gate_rules
from .evidence_obligations import ScopedNative, planning_rules
from .ram_hillclimb import digest

Claims = tuple[tuple[tuple[int, ...], ...], ...]
Plan = frozenset[int]
Frontier = frozenset[Plan]
DOCKET_COMMIT = "6adb17f07ad62e15d3e1a46baed4f8bcd2546fab"
DOCKET_PATH = "metta-storyworld/engine/evidence_docket.metta"
DOCKET_SHA = "42c18cd929cfb5fcfa5b1bb9614b4d4eee4644a57766b09c7c18ec6396936cde"
OBSERVER = "Curator Epoch1"
FUEL = "(Succ (Succ Zero))"
BUDGETS = (4, 5)


@dataclass(frozen=True, slots=True)
class Docket:
    claims: Claims
    priors: tuple[int, ...]

    @property
    def root(self) -> str:
        return digest((self.claims, self.priors))


@dataclass(frozen=True, slots=True)
class Observation:
    docket: Docket
    state: State
    remaining: int
    legal: tuple[int, ...]


def topology(claims: Claims) -> Claims:
    """Canonicalize both record relabeling and equivalent claim orderings."""
    return min(tuple(sorted(relabel(routes, order) for routes in claims))
               for order in permutations(range(WIDTH)))


def minimal(plans) -> Frontier:
    distinct = frozenset(plans)
    return frozenset(plan for plan in distinct if not any(other < plan for other in distinct))


@cache
def needs(claims: Claims, state: State) -> tuple[tuple[Frontier, ...], Frontier]:
    raw = tuple(tuple(frozenset(slot for slot in route if state[slot] == UNCHECKED)
                      for route in routes if all(state[slot] != REJECTED for slot in route)) for routes in claims)
    return tuple(minimal(routes) for routes in raw), minimal(frozenset().union(*plans) for plans in product(*raw))


def observe(docket: Docket, state: State, remaining: int) -> Observation:
    individual, joint = needs(docket.claims, state)
    legal = ()
    if frozenset() not in joint and any(len(plan) <= remaining for plan in joint):
        legal = tuple(sorted({slot for frontier in individual for plan in frontier if len(plan) <= remaining
                              for slot in plan}))
    return Observation(docket, state, remaining, legal)


def completed(claims: Claims, state: State) -> bool:
    return all(any(all(state[slot] == CURRENT for slot in route) for route in routes) for routes in claims)


def rollout(docket: Docket, current: tuple[bool, ...], budget: int, choose) -> dict:
    if budget not in BUDGETS:
        raise ValueError("Unknown docket budget")
    state = (UNCHECKED,) * WIDTH
    checks = []
    for left in range(budget, 0, -1):
        public = observe(docket, state, left)
        if not public.legal:
            break
        action = choose(public)
        if type(action) is not int or action not in public.legal:
            raise ValueError("A docket policy proposed an illegal evidence check")
        outcome = CURRENT if current[action] else REJECTED
        state = written(state, action, outcome)
        checks.append({"slot": action, "outcome": outcome})
    return {"root": docket.root, "final_state": state, "receipts": checks,
            "checks": len(checks), "completed": completed(docket.claims, state)}


def exact_policy(docket: Docket):
    """Maximize joint success, then minimize expected checks using exact integer weights."""
    @cache
    def value(state: State, remaining: int) -> tuple[int, int]:
        if completed(docket.claims, state):
            return 100**remaining, 0
        common = observe(docket, state, remaining)
        return max((score(common, slot) for slot in common.legal), default=(0, 0))

    def score(public: Observation, slot: int) -> tuple[int, int]:
        success = work = 0
        p = docket.priors[slot]
        for outcome, weight in ((CURRENT, p), (REJECTED, 100 - p)):
            wins, saved = value(written(public.state, slot, outcome), public.remaining - 1)
            success += weight * wins
            work += weight * saved
        return success, work - 100**public.remaining

    def choose(public: Observation) -> int:
        return max(public.legal, key=lambda slot: (*score(public, slot), -slot))

    return choose


def frontier_choice(public: Observation, individual: tuple[Frontier, ...], joint: Frontier, *, shared: bool) -> int:
    """Hold prior ranking and test ordering fixed while changing the proof objective."""
    priors = public.docket.priors

    def rank(plan: Plan) -> tuple:
        return prod(priors[i] for i in plan) * 100**(WIDTH - len(plan)), -len(plan), tuple(-i for i in sorted(plan))

    def best(frontier: Frontier) -> Plan:
        return max((plan for plan in frontier if len(plan) <= public.remaining), key=rank)

    if shared:
        selected = best(joint)
    else:
        selected = min((best(frontier) for frontier in individual if frozenset() not in frontier), key=rank)
    return min(selected, key=lambda slot: (priors[slot], slot))


@cache
def claim_facts(claims: Claims) -> tuple[tuple[str, ...], str, tuple[tuple[str, ...], ...]]:
    facts, labels = [], []
    goals = "EndGoals"
    for index in reversed(range(len(claims))):
        goals = f"(Goal Claim{index} {goals})"
    for claim, routes in enumerate(claims):
        trees = []
        for number, route in enumerate(routes):
            key = f"Route{claim}_{number}"
            inputs = [(f"Premise{i}", f"(Evidence Record{i} Source{i})") for i in route]
            if len(route) == 3:
                (left, a), (right, b), last = inputs
                facts.append(f"(SupportsBoth First{key} {left} {right} Partial{key})")
                inputs = [(f"Partial{key}", f"(Because First{key} {a} {b})"), last]
            form = "Supports" if len(inputs) == 1 else "SupportsBoth"
            premises = " ".join(premise for premise, _ in inputs)
            children = " ".join(proof for _, proof in inputs)
            facts.append(f"({form} {key} {premises} Claim{claim})")
            trees.append(f"(Because {key} {children})")
        labels.append(tuple(trees))
    return tuple(facts), goals, tuple(labels)


class NativeDocket:
    """Use canonical snapshots to verify public plans and final receipt-only proofs."""

    def __init__(self, checkout: Path):
        extra = subprocess.check_output(["git", "-C", str(checkout), "show", f"{DOCKET_COMMIT}:{DOCKET_PATH}"])
        if digest(extra) != DOCKET_SHA:
            raise ValueError("Native docket rules differ from their source pin")
        self.public_rules = planning_rules(checkout) + "\n" + extra.decode()
        self.acceptance_rules = gate_rules(checkout) + "\n" + extra.decode()
        self.planner = ScopedNative(self.public_rules)
        self.verifier = ScopedNative(self.acceptance_rules)
        self.read = cache(self._read)
        self.accept = cache(self._accept)

    def _read(self, claims: Claims, state: State) -> tuple[tuple[Frontier, ...], Frontier]:
        rules, goals, _ = claim_facts(claims)
        facts = list(rules)
        for i, status in enumerate(state):
            facts.append(f"(AdvertisedWitness Curator Premise{i} Record{i} Source{i})")
            if status != REJECTED:
                tag = "Checked" if status == CURRENT else "Unchecked"
                facts.append(f"({tag}Record {OBSERVER} Record{i} Source{i})")
        queries = tuple(f"(MinimalProofNeeds {OBSERVER} Claim{i} {FUEL})" for i in range(len(claims)))
        results = self.planner.evaluate(facts, (*queries, f"(MinimalDocketNeeds {OBSERVER} {goals} {FUEL})"))
        *individual, joint = [frozenset(frozenset(int(str(slot).removeprefix("Record")) for slot in row.get_children())
                                        for row in rows) for rows in results]
        result = tuple(individual), joint
        if result != needs(claims, state):
            raise ValueError("Native docket frontiers disagree with the enumerated proof routes")
        return result

    def _accept(self, claims: Claims, state: State) -> tuple[tuple[str, ...], str]:
        rules, goals, labels = claim_facts(claims)
        facts = list(rules)
        for i, status in enumerate(state):
            if status == CURRENT:
                facts.extend((f"(Witness Curator Premise{i} Record{i})", f"(RecordState Record{i} current)",
                              f"(RecordRoot Record{i} Source{i})", f"(CheckedRecord {OBSERVER} Record{i} Source{i})"))
        query = f"(DocketWithReceipts {OBSERVER} {goals} {FUEL})"
        (answers,) = self.verifier.evaluate(facts, (query,))
        actual = {str(answer) for answer in answers}
        expected = set()
        concrete = [[tree for route, tree in zip(group, trees, strict=True) if all(state[i] == CURRENT for i in route)]
                    for group, trees in zip(claims, labels, strict=True)]
        for combination in product(*concrete):
            proof = "EndProofs"
            for claim, tree in reversed(list(enumerate(combination))):
                proof = f"(DocketProof Claim{claim} {tree} {proof})"
            expected.add(proof)
        if expected != actual:
            raise ValueError("Native docket acceptance disagrees with the independent receipt replay")
        return tuple(sorted(actual)), digest("\n".join((self.acceptance_rules, *facts, query)).encode())


def map_frontier(frontier: Frontier, order: tuple[int, ...]) -> Frontier:
    return frozenset(frozenset(order[i] for i in plan) for plan in frontier)
