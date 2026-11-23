"""Check temporal MeTTa proof revisions and their provenance on synthetic episodes."""
from __future__ import annotations

import argparse
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from importlib.metadata import version
from itertools import combinations, pairwise, product
from pathlib import Path

from .evidence_proofs import TARGET, Record, Rule, Snapshot, proof_oracle, public_query, storyworld_rules
from .ram_hillclimb import canonical, digest
from .ram_hillclimb_native import PACKAGE_VERSION

LINEAGE_COMMIT = "a21aff544e7d842659b1a731db5ab1a3d70fdf93"
LINEAGE_PATH = "metta-storyworld/engine/evidence_lineage.metta"
LINEAGE_SHA256 = "077ff3d7451ff2a4b5a3bc73e5e54765b4c60eb057dca7bbd6a9c2f43119da2a"
FAMILIES = ("alias_arrival", "shared_bottleneck", "superseded_shortcut")
MAX_ROOTS_PER_FAMILY = 32


def episode(family: str, seed: int) -> tuple[Snapshot, ...]:
    rng = random.Random(f"evidence-revision-v1/{family}/{seed}")
    if family == "alias_arrival":
        records = [Record("Original", "Mark", "Ledger"), Record("Copy", "Mark", "Ledger", visible=False),
                   Record("Alternate", "Mark", "Journal", visible=False)]
        rules = [Rule("ReviewRule", ("Mark",), TARGET)]
        first = {1: {"visible": True}}
        second = {2: {"visible": True}}
        last = {0: {"current": False}, 1: {"current": False}}
    elif family == "shared_bottleneck":
        records = [Record("Original", "Gate", "Watch"), Record("MarkOne", "Mark", "Journal"),
                   Record("MarkTwo", "Mark", "Ledger", visible=False),
                   Record("Alternate", "Gate", "Archive", visible=False)]
        rules = [Rule("ReviewRule", ("Gate", "Mark"), TARGET)]
        first = {2: {"visible": True}}
        second = {0: {"visible": False}}
        last = {3: {"visible": True}}
    elif family == "superseded_shortcut":
        records = [Record("Direct", TARGET, "Ledger"), Record("MarkOne", "Mark", "Ledger"),
                   Record("GateOne", "Gate", "Watch"), Record("Alternate", TARGET, "Archive", visible=False)]
        rules = [Rule("ReviewRule", ("Gate", "Mark"), TARGET)]
        first = {0: {"current": False}}
        second = {3: {"visible": True}}
        last = {2: {"current": False}}
    else:
        raise ValueError(f"Unknown revision family {family}")

    if rng.choice((False, True)):
        first, second = second, first
    extra_link = rng.choice((False, True))
    if extra_link:
        rule = rules[0]
        rules = [Rule("Bridge", ("Mark",), "CheckedMark"), replace(rule, premises=(*rule.premises[:-1], "CheckedMark"))]
    for index in range(rng.randrange(4)):
        records.append(Record(f"Noise{index}", f"Unknown{index}", f"Unrelated{index}", current=bool(index % 2)))
    claim_names = sorted({r.claim for r in records} | {c for r in rules for c in (*r.premises, r.conclusion)})
    claims = {name: f"Claim{rng.getrandbits(48):012x}" for name in claim_names if name != TARGET}
    record_ids = {record.name: f"Record{rng.getrandbits(48):012x}" for record in records}
    roots = {root: f"Source{rng.getrandbits(48):012x}" for root in sorted({r.root for r in records})}
    normalized_rules = tuple(replace(rule, premises=tuple(claims[p] for p in rule.premises),
                                     conclusion=claims.get(rule.conclusion, rule.conclusion)) for rule in rules)

    result = []
    for changes in ({}, first, second, last):
        for index, change in changes.items():
            records[index] = replace(records[index], **change)
        visible = [replace(r, name=record_ids[r.name], claim=claims.get(r.claim, r.claim), root=roots[r.root])
                   for r in records if r.visible]
        result.append(Snapshot(tuple(sorted(visible, key=lambda r: r.name)), normalized_rules, 1 + int(extra_link)))
    return tuple(result)


def minimal_sources(snapshot: Snapshot) -> set[frozenset[str]]:
    """Evaluate bounded joins bottom-up, minimizing provenance at each depth."""
    bases: dict[str, set[frozenset[str]]] = defaultdict(set)
    for record in snapshot.records:
        if record.visible and record.current:
            bases[record.claim].add(frozenset((record.root,)))
    for _ in range(snapshot.depth):
        next_bases = {claim: set(sources) for claim, sources in bases.items()}
        for rule in snapshot.rules:
            output = next_bases.setdefault(rule.conclusion, set())
            for witnesses in product(*(bases[claim] for claim in rule.premises)):
                output.add(frozenset().union(*witnesses))
        for claim, sources in next_bases.items():
            bases[claim] = {s for s in sources if not any(other < s for other in sources)}
    return bases[TARGET]


def summary(snapshot: Snapshot) -> dict:
    sources = sorted(sorted(s) for s in minimal_sources(snapshot))
    disjoint = [[left, right] for left, right in combinations(sources, 2) if set(left).isdisjoint(right)]
    return {"proofs": sorted(proof_oracle(snapshot)), "minimal_sources": sources, "disjoint_source_pairs": disjoint}


def checked_snapshot(rules: str, snapshot: Snapshot) -> dict:
    from hyperon import Environment, MeTTa

    public = public_query(snapshot)
    query = public.splitlines()[-1]
    source = "\n".join((rules, public, query.replace("!(Support ", "!(MinimalEvidenceRoots ", 1),
                        query.replace("!(Support ", "!(DisjointEvidenceRoots ", 1)))
    runtime = MeTTa(env_builder=Environment.custom_env(create_config=False, is_test=True))
    proofs, minimum, pairs = runtime.run(source)
    native = {
        "proofs": sorted({str(atom) for atom in proofs}),
        "minimal_sources": sorted({tuple(sorted(map(str, atom.get_children()))) for atom in minimum}),
        "disjoint_source_pairs": sorted({tuple(sorted(tuple(sorted(map(str, roots.get_children())))
                                                       for roots in atom.get_children()[1:])) for atom in pairs}),
    }
    expected = summary(snapshot)
    if canonical(native) != canonical(expected):
        raise ValueError("Native proof and provenance results differ from the bottom-up oracle")
    return {"public_query": public, "target": expected, "native_input_sha256": digest(source.encode())}


def build(checkout: Path, out: Path, roots_per_family: int) -> dict:
    if version("hyperon") != PACKAGE_VERSION or not 1 <= roots_per_family <= MAX_ROOTS_PER_FAMILY:
        raise ValueError(f"Use hyperon=={PACKAGE_VERSION} and 1 to {MAX_ROOTS_PER_FAMILY} roots per family")
    data = subprocess.check_output(["git", "-C", str(checkout), "show", f"{LINEAGE_COMMIT}:{LINEAGE_PATH}"])
    if digest(data) != LINEAGE_SHA256:
        raise ValueError("Storyworld lineage rules differ from the pinned bytes")
    rules = storyworld_rules(checkout) + "\n" + data.decode("utf-8")
    rows = []
    counts: Counter = Counter()
    for family, seed in product(FAMILIES, range(roots_per_family)):
        snapshots = episode(family, seed)
        root = digest([asdict(snapshot) for snapshot in snapshots])
        results = [checked_snapshot(rules, snapshot) for snapshot in snapshots]
        for epoch, (before, after) in enumerate(pairwise(results), 1):
            prior, current = before["target"], after["target"]
            prior_proofs, current_proofs = set(prior["proofs"]), set(current["proofs"])
            stable_basis = prior["minimal_sources"] == current["minimal_sources"]
            disjoint = bool(current["disjoint_source_pairs"])
            counts["proof_changes"] += prior_proofs != current_proofs
            counts["minimal_source_changes"] += not stable_basis
            counts["changed_proofs_with_same_minimal_sources"] += stable_basis and prior_proofs != current_proofs
            counts["bounded_abstentions"] += not current_proofs
            counts["disjoint_support"] += disjoint
            counts["count_only_false_corroborations"] += len(current_proofs) >= 2 and not disjoint
            rows.append({
                "schema": "metta-evidence-revision-v1", "source_root_sha256": root, "family": family,
                "epoch": epoch, "before": before, "after": after,
                "expected_delta": {"added_proofs": sorted(current_proofs - prior_proofs),
                                   "removed_proofs": sorted(prior_proofs - current_proofs)},
                "independently_replayed": True, "training_ready": False,
            })
    out.mkdir(parents=True, exist_ok=False)
    payload = "".join(canonical(row) + "\n" for row in rows).encode()
    (out / "evidence_transitions.jsonl").write_bytes(payload)
    manifest = {
        "storyworld_commit": LINEAGE_COMMIT, "rule_sha256": digest(rules.encode()), "hyperon": PACKAGE_VERSION,
        "roots": len(FAMILIES) * roots_per_family, "snapshots": len(FAMILIES) * roots_per_family * 4,
        "presentations": len(rows), "views_per_root": 3, "outcomes": dict(counts),
        "presentations_sha256": digest(payload), "training_ready": False,
    }
    (out / "manifest.json").write_text(canonical(manifest) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storyworld", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roots-per-family", type=int, default=16)
    args = parser.parse_args()
    print(canonical(build(args.storyworld, args.out, args.roots_per_family)))


if __name__ == "__main__":
    main()
