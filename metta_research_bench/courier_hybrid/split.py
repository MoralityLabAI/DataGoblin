"""Reproduce root allocation independently of policy outputs."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .native import PLACES, World, canonical, digest
from .oracle import state_graph

HERE = Path(__file__).resolve().parent


def arcs(open_edges, gated_edges, directed=()):
    return tuple(
        sorted(
            [
                *(edge for a, b in open_edges for edge in ((a, b, False), (b, a, False))),
                *(edge for a, b in gated_edges for edge in ((a, b, True), (b, a, True))),
                *directed,
            ]
        )
    )


TOPOLOGIES = {
    "gated_figure_eight": arcs(
        ((0, 1), (1, 2), (2, 0), (2, 3), (4, 5), (5, 6), (6, 4)), ((3, 4), (0, 5)), ((3, 6, False),)
    ),
    "offset_shortcuts": arcs(
        ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0)), ((0, 2), (1, 5)), ((2, 5, False),)
    ),
    "directed_detour": arcs(
        ((0, 1), (1, 2), (2, 3), (4, 5), (5, 6)), ((1, 4), (3, 6)), ((0, 6, False), (3, 0, False), (4, 2, True))
    ),
    "gated_forks": arcs(
        ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)), ((0, 6), (2, 5)), ((5, 1, False),)
    ),
}


@dataclass(frozen=True)
class Root:
    split: str
    family: str
    ordinal: int
    world_sha256: str
    semantic_sha256: str


def plan(family: str) -> dict:
    if family not in {"development", "confirmation"}:
        raise ValueError("Unknown experimental family")
    return json.loads((HERE / f"{family}.json").read_text(encoding="utf-8"))


def draw(upstream, namespace: str, split: str, family: str, ordinal: int) -> tuple[World, str]:
    rng = random.Random(f"{namespace}/{split}/{family}/{ordinal}")
    routes = TOPOLOGIES[family]
    offices = tuple(sorted(rng.sample(range(PLACES), 2)))
    depots = tuple(sorted(rng.sample(range(PLACES), 2)))
    desk, entry = rng.sample(range(PLACES), 2)
    first, second, alias = rng.sample(range(PLACES), 3)

    def roles(order):
        return (
            tuple(sorted(order[i] for i in offices)),
            tuple(sorted(order[i] for i in depots)),
            order[desk],
            order[entry],
            (tuple(sorted((order[first], order[alias]))), (order[second],)),
        )

    signature = min(roles(order) for order in upstream.automorphisms(routes))
    order = rng.sample(range(PLACES), PLACES)
    world = World(upstream.relabel(routes, order), *roles(order))
    return world, digest([family, signature])


def allocate(upstream, protocol: dict, prior: list[dict] | None = None) -> dict:
    used_sources = {row["world_sha256"] for row in prior or []}
    used_roles = {row["semantic_sha256"] for row in prior or []}
    roots = []
    for split, size in (
        ("train", protocol["train_roots_per_topology"]),
        ("score", protocol["scoring_roots_per_topology"]),
    ):
        for family in TOPOLOGIES:
            accepted = 0
            for ordinal in range(10000):
                world, roles = draw(upstream, protocol["namespace"], split, family, ordinal)
                if world.sha256 in used_sources or roles in used_roles:
                    continue
                _, distances = state_graph(world)
                if distances.get(world.initial, protocol["horizon"] + 1) > protocol["horizon"]:
                    continue
                used_sources.add(world.sha256)
                used_roles.add(roles)
                roots.append(asdict(Root(split, family, ordinal, world.sha256, roles)))
                accepted += 1
                if accepted == size:
                    break
            if accepted != size:
                raise RuntimeError(f"Could not allocate registered roots for {split}/{family}")
    return {
        "schema": "dg-metta13-split-v1",
        "protocol_sha256": digest(protocol),
        "roots_sha256": digest(roots),
        "roots": roots,
    }


def worlds(upstream, protocol: dict, manifest: dict, split: str):
    for entry in manifest["roots"]:
        root = Root(**entry)
        if root.split == split:
            world, roles = draw(upstream, protocol["namespace"], root.split, root.family, root.ordinal)
            if (world.sha256, roles) != (root.world_sha256, root.semantic_sha256):
                raise ValueError("Generated root differs from the allocation")
            yield root, world


def load_manifest(path: Path, protocol: dict) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    roots = data["roots"]
    expected_counts = {"train": protocol["train_roots_per_topology"], "score": protocol["scoring_roots_per_topology"]}
    expected_groups = {(split, family) for split in expected_counts for family in TOPOLOGIES}
    if (
        data["protocol_sha256"] != digest(protocol)
        or data["roots_sha256"] != digest(roots)
        or {(r["split"], r["family"]) for r in roots} != expected_groups
        or len({r["world_sha256"] for r in roots}) != len(roots)
        or len({r["semantic_sha256"] for r in roots}) != len(roots)
        or any(
            sum(r["split"] == split and r["family"] == family for r in roots) != expected_counts[split]
            for split, family in expected_groups
        )
    ):
        raise ValueError("Frozen courier split is incomplete or inconsistent")
    return data


def write_manifest(path: Path, manifest: dict) -> None:
    with path.open("xb") as handle:
        handle.write(canonical(manifest) + b"\n")
