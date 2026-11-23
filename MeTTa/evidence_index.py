"""Demand-driven proof cells with exact invalidation across evidence snapshots."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, product

from .evidence_proofs import TARGET, Record, Rule, Snapshot

Cell = tuple[str, int]


@dataclass(frozen=True, slots=True)
class Proof:
    tree: str
    sources: frozenset[str]


class ProofIndex:
    """Cache full bounded supports, including those currently dominated by another tree."""

    def __init__(self) -> None:
        self._records: dict[str, tuple[Record, ...]] = {}
        self._rules: dict[str, tuple[Rule, ...]] = {}
        self._proofs: dict[Cell, frozenset[Proof]] = {}
        self._joins: dict[tuple[Rule, int], tuple[tuple[frozenset[Proof], ...], frozenset[Proof]]] = {}
        self._children: dict[Cell, set[Cell]] = {}
        self._parents: dict[Cell, set[Cell]] = defaultdict(set)
        self._work: Counter = Counter()

    def _install(self, snapshot: Snapshot) -> None:
        records: dict[str, list[Record]] = defaultdict(list)
        for record in sorted(snapshot.records, key=lambda record: record.name):
            if record.visible and record.current:
                records[record.claim].append(record)
        rules: dict[str, list[Rule]] = defaultdict(list)
        for rule in sorted(snapshot.rules, key=lambda rule: rule.name):
            rules[rule.conclusion].append(rule)
        ready = {claim: tuple(rows) for claim, rows in records.items()}
        links = {claim: tuple(rows) for claim, rows in rules.items()}
        evidence_changes = {c for c in self._records.keys() | ready.keys() if self._records.get(c) != ready.get(c)}
        rule_changes = {c for c in self._rules.keys() | links.keys() if self._rules.get(c) != links.get(c)}
        dirty = {key for key in self._proofs if key[0] in evidence_changes or (key[1] and key[0] in rule_changes)}
        queue = list(dirty)
        while queue:
            cell = queue.pop()
            for parent in self._parents.get(cell, ()):
                if parent not in dirty:
                    dirty.add(parent)
                    queue.append(parent)
        for key in dirty:
            for child in self._children.pop(key):
                parents = self._parents[child]
                parents.remove(key)
                if not parents:
                    del self._parents[child]
            del self._proofs[key]
        if rule_changes:
            active_rules = set(snapshot.rules)
            self._joins = {key: join for key, join in self._joins.items() if key[0] in active_rules}
        self._records, self._rules = ready, links
        self._work["invalidated_cells"] = len(dirty)

    def _resolve(self, cell: Cell) -> frozenset[Proof]:
        if cell in self._proofs:
            self._work["cache_hits"] += 1
            return self._proofs[cell]
        claim, depth = cell
        self._work["computed_cells"] += 1
        proofs = {Proof(f"(Evidence {r.name} {r.root})", frozenset((r.root,))) for r in self._records.get(claim, ())}
        children = set()
        if depth:
            for rule in self._rules.get(claim, ()):
                premises = [(premise, depth - 1) for premise in rule.premises]
                children.update(premises)
                proofs.update(self._join(rule, depth, tuple(self._resolve(premise) for premise in premises)))
        for child in children:
            self._parents[child].add(cell)
        self._children[cell] = children
        self._proofs[cell] = frozenset(proofs)
        return self._proofs[cell]

    def _join(self, rule: Rule, depth: int, premises: tuple[frozenset[Proof], ...]) -> frozenset[Proof]:
        key = (rule, depth)
        cached = self._joins.get(key)
        if cached is not None and all(new is old for new, old in zip(premises, cached[0], strict=True)):
            self._work["join_cache_hits"] += 1
            return cached[1]
        self._work["rule_joins"] += 1
        proofs = set()
        for branches in product(*premises):
            self._work["proof_products"] += 1
            tree = f"(Because {rule.name} {' '.join(proof.tree for proof in branches)})"
            proofs.add(Proof(tree, frozenset().union(*(proof.sources for proof in branches))))
        result = frozenset(proofs)
        self._joins[key] = premises, result
        return result

    def evaluate(self, snapshot: Snapshot, claim: str = TARGET) -> tuple[dict, dict[str, int]]:
        if snapshot.depth < 0:
            raise ValueError("Proof depth must be nonnegative")
        self._work.clear()
        self._install(snapshot)
        proofs = self._resolve((claim, snapshot.depth))
        source_sets = {proof.sources for proof in proofs}
        minimum = sorted(sorted(s) for s in source_sets if not any(other < s for other in source_sets))
        disjoint = [[left, right] for left, right in combinations(minimum, 2) if set(left).isdisjoint(right)]
        return {
            "proofs": sorted({proof.tree for proof in proofs}),
            "minimal_sources": minimum,
            "disjoint_source_pairs": disjoint,
        }, dict(self._work)
