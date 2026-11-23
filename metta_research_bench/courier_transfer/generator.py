"""Deterministic courier worlds for oracle wiring ONLY, never native data."""
from __future__ import annotations

import json
import random
from pathlib import Path

from manifest_check import root_seed
from reference import CourierState, CourierWorld, FiniteOracle, initial

HERE = Path(__file__).resolve().parent


def reference_root(root_id: str, manifest: dict) -> tuple[CourierWorld, CourierState]:
    chunks = root_id.split("/")
    if len(chunks) != 4 or chunks[0] != "DG12" or not chunks[3].isdigit():
        raise ValueError("invalid root ID")
    _, split, family, index = chunks
    if root_id != f"DG12/{split}/{family}/{int(index):04d}":
        raise ValueError("noncanonical root ID")
    if (family not in manifest["split_roots"].get(split, ())
            or not (0 <= int(index) < manifest["split_roots"][split][family])):
        raise ValueError("root is outside registered split")
    seed = root_seed(manifest, root_id)
    rnd = random.Random(seed)
    suffix = f"{seed & 0xffff:04x}"
    def node(x: str) -> str:
        return f"{x}_{suffix}"

    def parcel(x: str) -> str:
        return f"parcel-{x}-{suffix}"

    def permit(x: str) -> str:
        return f"permit-{x}-{suffix}"

    def witness(x: str) -> str:
        return f"witness-{x}-{suffix}"

    def source(x: str) -> str:
        return f"source-{x}-{suffix}"

    edges = []
    offices = []
    parcels = []
    witnesses = []
    permit_sources = []
    if family == "route_ties":
        edges = [("A", "B", ""), ("B", "D", ""),
                 ("A", "C", ""), ("C", "D", "")]
        parcels = [("p", "A", "D", "")]
    elif family == "permit_gate":
        edges = [("A", "B", "g")]
        offices = [("g", "A")]
        parcels = [("p", "A", "B", "g")]
    elif family == "multi_drop":
        edges = [("A", "B", "")]
        parcels = [("p", "A", "B", ""), ("q", "A", "B", "")]
    elif family == "evidence_ledger":
        edges = [("A", "B", "g")]
        offices = [("g", "A")]
        witnesses = [("w1", "A", "s1")]
        permit_sources = [("g", ("s1",))]
        parcels = [("p", "A", "B", "")]
    elif family == "permit_chain":
        edges = [("A", "B", "g1")]
        offices = [("g1", "A"), ("g2", "A")]
        parcels = [("p", "A", "B", "g2")]
    elif family == "evidence_collision":
        edges = [("A", "B", "")]
        offices = [("g", "A")]
        witnesses = [("w1", "A", "s1"), ("alias", "A", "s1"),
                     ("w2", "A", "s2")]
        permit_sources = [("g", ("s1", "s2"))]
        parcels = [("p", "A", "A", "g")]
    elif family == "unseen_topology":
        edges = [("A", "B", ""), ("B", "E", ""),
                 ("A", "C", ""), ("C", "D", ""), ("D", "E", "")]
        parcels = [("p", "A", "E", "")]
    else:
        raise ValueError(f"unknown family {family}")

    # A seeded distractor never shortens the valid path.
    if seed % 2:
        edges.append(("A", "Z", ""))
    rnd.shuffle(edges)
    rnd.shuffle(offices)
    rnd.shuffle(parcels)
    rnd.shuffle(witnesses)
    world = CourierWorld(
        start=node("A"),
        edges=tuple((node(a), node(b), permit(g) if g else "")
                    for a, b, g in edges),
        offices=tuple((permit(g), node(a)) for g, a in offices),
        parcels=tuple((parcel(p), node(a), node(b), permit(g) if g else "")
                      for p, a, b, g in parcels),
        witnesses=tuple((witness(w), node(a), source(s)) for w, a, s in witnesses),
        permit_sources=tuple((permit(g), tuple(source(s) for s in ss))
                             for g, ss in permit_sources),
    )
    return world, initial(world)


def reference_probe(per_family: int = 3) -> dict:
    manifest = json.loads((HERE / "roots_manifest.json").read_text())
    results = []
    for split, families in manifest["split_roots"].items():
        for family, count in families.items():
            for index in range(min(count, per_family)):
                root_id = f"DG12/{split}/{family}/{index:04d}"
                world, state = reference_root(root_id, manifest)
                oracle = FiniteOracle(world)
                results.append({"root": root_id, "reachable_under_five": oracle.can_finish(state, 5),
                                "optimal_first_action_count": len(oracle.action_set(state, 5))})
    if not all(row["reachable_under_five"] for row in results):
        raise AssertionError("unsolvable five-call wiring root")
    return {"mode": "reference_wiring", "roots_checked": len(results), "all_feasible": True,
            "roots_with_tied_first_actions": sum(x["optimal_first_action_count"] > 1
                                                for x in results)}


if __name__ == "__main__":
    print(json.dumps(reference_probe(), indent=2))
