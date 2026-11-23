"""Query bounded public evidence obligations without exposing private outcomes."""
from __future__ import annotations

import subprocess
from functools import cache
from itertools import permutations
from pathlib import Path

from .evidence_budget import CURRENT, UNCHECKED, WIDTH, PublicObservation, State
from .evidence_budget_confirmation import relabel
from .evidence_budget_pilot import route_rules
from .evidence_revision import storyworld_lineage_rules
from .ram_hillclimb import digest

OBLIGATION_COMMIT = "e420454c7362f5ef4b19f4e0ffe0bae882f90e8f"
OBLIGATION_PATH = "metta-storyworld/engine/evidence_obligations.metta"
OBLIGATION_SHA256 = "3f4e2078a1bbaf2c9185341907f651d6126e39430afacecca189bf14113c5cee"


def planning_rules(checkout: Path) -> str:
    data = subprocess.check_output(["git", "-C", str(checkout), "show", f"{OBLIGATION_COMMIT}:{OBLIGATION_PATH}"])
    if digest(data) != OBLIGATION_SHA256:
        raise ValueError("Public obligation rules must match their source pin")
    return storyworld_lineage_rules(checkout) + "\n" + data.decode("utf-8")


class ScopedNative:
    """Keep one fixed rule space and install only the facts for the current query."""

    def __init__(self, rules: str):
        from hyperon import Environment, MeTTa

        self.runtime = MeTTa(env_builder=Environment.custom_env(create_config=False, is_test=True))
        if self.runtime.run(rules):
            raise ValueError("The native source must contain rules without executable commands")
        self.space = self.runtime.space()
        self.rules_count = self.space.atom_count()
        self.calls = 0

    def evaluate(self, facts: list[str], queries: tuple[str, ...]) -> tuple[list, ...]:
        atoms = self.runtime.parse_all("\n".join(facts))
        for atom in atoms:
            self.space.add_atom(atom)
        try:
            self.calls += 1
            return tuple(self.runtime.run("\n".join(f"!{query}" for query in queries)))
        finally:
            for atom in reversed(atoms):
                self.space.remove_atom(atom)
            if self.space.atom_count() != self.rules_count:
                raise ValueError("Facts survived from a previous native observation")


def host_frontier(routes: tuple[tuple[int, ...], ...], state: State) -> tuple[set, set]:
    outstanding = {frozenset(i for i in route if state[i] == UNCHECKED) for route in routes
                   if all(state[i] in (UNCHECKED, CURRENT) for i in route)}
    return outstanding, {needs for needs in outstanding if not any(other < needs for other in outstanding)}


def route_labeler():
    """Reuse native work across isomorphic catalogs using only public slot permutations."""
    layouts = {}

    def labels(routes: tuple[tuple[int, ...], ...], width: int) -> tuple:
        if not 0 < width <= WIDTH:
            raise ValueError("Public catalog exceeds the route-labeling bound")
        routes = tuple(sorted(tuple(sorted(route)) for route in routes))
        layout = (width, routes)
        if layout not in layouts:
            for order in permutations(range(width)):
                inverse = tuple(order.index(slot) for slot in range(width))
                layouts.setdefault((width, relabel(routes, order)), (routes, order, inverse))
        return layouts[layout]

    return labels


def obligation_reader(native: ScopedNative):
    """Build both views on the same public state and verify against an independent route projection."""
    labels = route_labeler()

    @cache
    def read(routes: tuple[tuple[int, ...], ...], state: State) -> tuple[set, set]:
        facts, _ = route_rules(routes)
        for i, outcome in enumerate(state):
            facts.append(f"(AdvertisedWitness Reviewer Claim{i} Note{i} Source{i})")
            if outcome == UNCHECKED:
                facts.append(f"(UncheckedRecord Reviewer Epoch1 Note{i} Source{i})")
            elif outcome == CURRENT:
                facts.append(f"(CheckedRecord Reviewer Epoch1 Note{i} Source{i})")
        (answers,) = native.evaluate(facts, ("(ProofNeedFrontiers Reviewer Epoch1 Review (Succ (Succ Zero)))",))
        (answer,) = answers
        tag, *frontiers = answer.get_children()
        if str(tag) != "Frontiers" or len(frontiers) != 2:
            raise ValueError("Native obligation frontier has an unexpected shape")
        full, minimal = ({frozenset(int(str(child).removeprefix("Note")) for child in atom.get_children())
                          for atom in frontier.get_children()} for frontier in frontiers)
        if (full, minimal) != host_frontier(routes, state):
            raise ValueError("Native obligations differ from the public-state reference")
        return full, minimal

    def project(public: PublicObservation) -> tuple[tuple, tuple]:
        routes = public.catalog.routes
        basis, order, inverse = labels(routes, len(public.state))
        full, minimal = read(basis, tuple(public.state[slot] for slot in order))
        mapped = tuple({frozenset(order[slot] for slot in needs) for needs in sets} for sets in (full, minimal))
        if mapped != host_frontier(routes, public.state) or relabel(routes, inverse) != basis:
            raise ValueError("Obligation frontier failed public relabeling verification")
        return tuple(tuple(sorted(tuple(sorted(needs)) for needs in sets if len(needs) <= public.remaining))
                     for sets in mapped)

    return project
