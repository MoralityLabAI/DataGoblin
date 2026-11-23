"""Query bounded public evidence obligations without exposing private outcomes."""
from __future__ import annotations

import subprocess
from functools import cache
from pathlib import Path

from .evidence_budget import CURRENT, UNCHECKED, PublicObservation, State
from .evidence_budget_pilot import route_rules
from .evidence_revision import storyworld_lineage_rules
from .ram_hillclimb import digest

OBLIGATION_COMMIT = "b97aaa053279f272d572a4076c97cdaf1b9abfd9"
OBLIGATION_PATH = "metta-storyworld/engine/evidence_obligations.metta"
OBLIGATION_SHA256 = "464fee5bacb4f10db80606c7e1c65132564e93264bc85c906aa1090a85414cba"


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


def obligation_reader(native: ScopedNative):
    """Build both views on the same public state and verify against an independent route projection."""
    @cache
    def read(routes: tuple[tuple[int, ...], ...], state: State) -> tuple[set, set]:
        facts, _ = route_rules(routes)
        for i, outcome in enumerate(state):
            facts.append(f"(AdvertisedWitness Reviewer Claim{i} Note{i} Source{i})")
            if outcome == UNCHECKED:
                facts.append(f"(UncheckedRecord Reviewer Epoch1 Note{i} Source{i})")
            elif outcome == CURRENT:
                facts.append(f"(CheckedRecord Reviewer Epoch1 Note{i} Source{i})")
        queries = tuple(f"({predicate} Reviewer Epoch1 Review (Succ (Succ Zero)))"
                        for predicate in ("ProofNeeds", "MinimalProofNeeds"))
        full, minimal = ({frozenset(int(str(child).removeprefix("Note")) for child in atom.get_children())
                          for atom in values} for values in native.evaluate(facts, queries))
        if (full, minimal) != host_frontier(routes, state):
            raise ValueError("Native obligations differ from the public-state reference")
        return full, minimal

    def project(public: PublicObservation) -> tuple[tuple, tuple]:
        full, minimal = read(public.catalog.routes, public.state)
        return tuple(sorted(tuple(sorted(needs)) for needs in full if len(needs) <= public.remaining)), tuple(
            sorted(tuple(sorted(needs)) for needs in minimal if len(needs) <= public.remaining))

    return project
