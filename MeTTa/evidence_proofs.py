"""Build checked proof and abstention exercises from synthetic evidence snapshots."""
from __future__ import annotations

import argparse
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from itertools import product
from pathlib import Path

from .ram_hillclimb import canonical, digest
from .ram_hillclimb_native import PACKAGE_VERSION

STORYWORLD_COMMIT = "b2fea02881741ab341f706036dc022a3995bb527"
RULE_PATH = "metta-storyworld/engine/evidence_proofs.metta"
RULE_SHA256 = "4f574bc0b0c27daa977e2b774bf33c8326dd632cb6a82abaf4c09c731b5e99ec"
FAMILIES = ("visibility", "supersession", "rule_incidence")
TARGET = "Review"
MAX_ROOTS_PER_FAMILY = 32


@dataclass(frozen=True)
class Record:
    name: str
    claim: str
    root: str
    visible: bool = True
    current: bool = True


@dataclass(frozen=True)
class Rule:
    name: str
    premises: tuple[str, ...]
    conclusion: str


@dataclass(frozen=True)
class Snapshot:
    records: tuple[Record, ...]
    rules: tuple[Rule, ...]
    depth: int


def source_pair(family: str, seed: int) -> tuple[Snapshot, Snapshot]:
    """Change a binding while keeping counts of visible and current records fixed."""
    rng = random.Random(f"proof-snapshot-v1/{family}/{seed}")
    records = [Record("Note", "Mark", "Ledger"), Record("Check", "Gate", "Watch"),
               Record("Other", "Detour", "Messages")]
    if rng.choice((False, True)):
        records.append(Record("Copy", "Mark", "Ledger"))
    for index in range(rng.randrange(4)):
        records.append(Record(f"Idle{index}", f"Noise{index}", f"Spare{index}", current=bool(index % 2)))

    chain = rng.randrange(3)
    intermediate = ("Mark", "Inspection", "Dossier")
    rules = [Rule(f"Link{i}", (intermediate[i],), intermediate[i + 1]) for i in range(chain)]
    rules.extend((Rule("ReviewRule", (intermediate[chain], "Gate"), TARGET),
                  Rule("OtherRule", ("Detour", "Missing"), "Unrelated")))
    negative_rules = list(rules)
    if family == "visibility":
        records[2] = replace(records[2], visible=False)
        negative = [*records]
        negative[1] = replace(records[1], visible=False)
        negative[2] = replace(records[2], visible=True)
    elif family == "supersession":
        records[2] = replace(records[2], current=False)
        negative = [*records]
        negative[1] = replace(records[1], current=False)
        negative[2] = replace(records[2], current=True)
    elif family == "rule_incidence":
        negative = records
        negative_rules[-2:] = [Rule("ReviewRule", (intermediate[chain], "Missing"), TARGET),
                               Rule("OtherRule", ("Detour", "Gate"), "Unrelated")]
    else:
        raise ValueError(f"Unknown snapshot family {family}")

    # Opaque root-specific IDs reduce stable correlations between a name and its graph role.
    names = {name: f"Claim{rng.getrandbits(48):012x}" for name in sorted(
        {r.claim for r in records} | {c for r in rules for c in (*r.premises, r.conclusion)}
    ) if name != TARGET}
    record_ids = {r.name: f"Record{rng.getrandbits(48):012x}" for r in records}
    roots = {name: f"Source{rng.getrandbits(48):012x}" for name in sorted({r.root for r in records})}

    def normalize(items: list[Record], links: list[Rule]) -> Snapshot:
        public_records = tuple(sorted((replace(r, name=record_ids[r.name], claim=names[r.claim], root=roots[r.root])
                                       for r in items if r.visible), key=lambda row: row.name))
        public_rules = tuple(replace(r, premises=tuple(names[p] for p in r.premises),
                                     conclusion=names.get(r.conclusion, r.conclusion)) for r in links)
        return Snapshot(public_records, public_rules, chain + 1)

    return normalize(records, rules), normalize(negative, negative_rules)


def public_query(snapshot: Snapshot) -> str:
    facts = []
    for record in snapshot.records:
        if not record.visible:
            continue
        state = "current" if record.current else "superseded"
        facts.extend((f"(Witness Curator {record.claim} {record.name})", f"(RecordState {record.name} {state})",
                      f"(RecordRoot {record.name} {record.root})"))
    for rule in snapshot.rules:
        form = "Supports" if len(rule.premises) == 1 else "SupportsBoth"
        facts.append(f"({form} {rule.name} {' '.join(rule.premises)} {rule.conclusion})")
    fuel = "(Succ " * snapshot.depth + "Zero" + ")" * snapshot.depth
    return "\n".join((*facts, f"!(Support Curator {TARGET} {fuel})", ""))


def proof_oracle(snapshot: Snapshot) -> set[str]:
    """Enumerate bounded support bottom-up, independently of native backward search."""
    proofs: dict[str, set[str]] = defaultdict(set)
    for record in snapshot.records:
        if record.visible and record.current:
            proofs[record.claim].add(f"(Evidence {record.name} {record.root})")
    for _ in range(snapshot.depth):
        additions: dict[str, set[str]] = defaultdict(set)
        for rule in snapshot.rules:
            for branches in product(*(proofs[claim] for claim in rule.premises)):
                additions[rule.conclusion].add(f"(Because {rule.name} {' '.join(branches)})")
        for claim, new in additions.items():
            proofs[claim].update(new)
    return proofs[TARGET]


def storyworld_rules(checkout: Path) -> str:
    data = subprocess.check_output(["git", "-C", str(checkout), "show", f"{STORYWORLD_COMMIT}:{RULE_PATH}"])
    if digest(data) != RULE_SHA256:
        raise ValueError("Storyworld evidence rule digest differs from the source pin")
    return data.decode("utf-8")


def build(checkout: Path, out: Path, roots_per_family: int) -> dict:
    """Execute only generated bounded facts plus pinned authored MeTTa rules."""
    from hyperon import Environment, MeTTa

    if version("hyperon") != PACKAGE_VERSION or not 1 <= roots_per_family <= MAX_ROOTS_PER_FAMILY:
        raise ValueError(f"Use hyperon=={PACKAGE_VERSION} and 1 to {MAX_ROOTS_PER_FAMILY} roots per family")
    rules = storyworld_rules(checkout)
    counts: Counter = Counter()
    rows = []
    for family, seed in product(FAMILIES, range(roots_per_family)):
        positive, negative = source_pair(family, seed)
        if (len(positive.records) != len(negative.records)
                or sum(r.current for r in positive.records) != sum(r.current for r in negative.records)):
            raise ValueError("Counterfactual changed the matched evidence counts")
        root = digest([asdict(positive), asdict(negative)])
        for view, snapshot in zip(("support", "abstain"), (positive, negative), strict=True):
            expected = proof_oracle(snapshot)
            public = public_query(snapshot)
            source = rules + "\n" + public
            runtime = MeTTa(env_builder=Environment.custom_env(create_config=False, is_test=True))
            (atoms,) = runtime.run(source)
            actual = {str(atom) for atom in atoms}
            if expected != actual or bool(expected) != (view == "support"):
                raise ValueError(f"Native proof audit failed for {family} root {seed} {view}")
            counts[view] += 1
            counts["proofs"] += len(expected)
            rows.append({
                "schema": "metta-observer-proof-v1", "source_root_sha256": root, "family": family,
                "public_query": public, "expected_proofs": sorted(expected), "training_ready": False,
                "native_input_sha256": digest(source.encode()), "independently_replayed": True,
            })
    out.mkdir(parents=True, exist_ok=False)
    payload = "".join(canonical(row) + "\n" for row in rows).encode()
    (out / "proof_candidates.jsonl").write_bytes(payload)
    manifest = {"storyworld_commit": STORYWORLD_COMMIT, "rule_sha256": RULE_SHA256, "hyperon": PACKAGE_VERSION,
                "roots": len(FAMILIES) * roots_per_family, "presentations": len(rows), "views_per_root": 2,
                "outcomes": dict(counts), "presentations_sha256": digest(payload), "training_ready": False}
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
