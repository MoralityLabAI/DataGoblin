"""MOLT-04: exact delayed-query memory retention environment."""
from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from fractions import Fraction

from datagoblin.training_gym.molt_goblin import digest

SPLITS = ("train", "calibration_short", "calibration_long")
VARIANTS = ("clean", "revised")


@dataclass(frozen=True)
class Entry:
    source: int
    features: tuple[int, int, int, int, int]
    value: int


@dataclass(frozen=True)
class World:
    root_id: str
    split: str
    entries: tuple[Entry, ...]
    query: tuple[int, int]
    edited_source: int
    replacement: int


def weight(features: tuple[int, int, int, int, int]) -> int:
    """Generator-only probability law; no hidden target is encoded in a feature."""
    scope, quality, proof, novelty, cue = features
    return 1 + 8 * scope + 24 * quality * proof + 9 * novelty * (1 - proof) + 6 * scope * cue


def make_world(split: str, seed: int, index: int) -> World:
    if split not in SPLITS or type(index) is not int or index < 0:
        raise ValueError("invalid root")
    root = digest(["molt-retention-v4", split, seed, index])
    rng = random.Random(int(root, 16))
    n = 18 if split == "calibration_long" else 10
    cue = rng.randrange(2)
    entries = tuple(
        Entry(i, (int(rng.randrange(2) == cue), rng.randrange(2),
                  rng.randrange(2), rng.randrange(2), cue), rng.randrange(17))
        for i in range(n)
    )
    pairs = list(itertools.combinations(range(n), 2))
    weights = [weight(entries[i].features) * weight(entries[j].features) for i, j in pairs]
    draw = rng.randrange(sum(weights))
    for pair, mass in zip(pairs, weights, strict=True):
        if draw < mass:
            query = pair
            break
        draw -= mass
    else:
        raise AssertionError("invalid pair distribution")
    edit = rng.randrange(n)
    replacement = (entries[edit].value + rng.randrange(1, 17)) % 17
    return World(root, split, entries, query, edit, replacement)


def public_features(world: World) -> tuple[tuple[int, int, int, int, int], ...]:
    return tuple(e.features for e in world.entries)


def retain(scores: tuple[float, ...], capacity: int) -> tuple[int, ...]:
    """Streaming top-k: discard low priorities on arrival, no future query reads."""
    if type(capacity) is not int or not 1 <= capacity <= len(scores):
        raise ValueError("invalid capacity")
    if not all(math.isfinite(v) for v in scores):
        raise ValueError("nonfinite scores")
    bank: list[int] = []
    for i in range(len(scores)):
        bank.append(i)
        bank.sort(key=lambda j: (-scores[j], j))
        if len(bank) > capacity:
            bank.pop()
    return tuple(sorted(bank))


def exact_expected_success(world: World, bank: tuple[int, ...],
                           verifier_budget: int, revised: bool) -> Fraction:
    """Average over fresh query draws and uniform independent source revisions."""
    if verifier_budget not in (0, 1):
        raise ValueError("invalid verifier budget")
    n = len(world.entries)
    if len(set(bank)) != len(bank) or any(not 0 <= i < n for i in bank):
        raise ValueError("invalid bank")
    chosen = set(bank)
    denominator = 0
    numerator = 0
    need = 2 - verifier_budget
    for i, j in itertools.combinations(range(n), 2):
        mass = weight(world.entries[i].features) * weight(world.entries[j].features)
        hits = int(i in chosen) + int(j in chosen)
        denominator += mass * n
        if hits >= need:
            survivors = n - hits if revised and hits == need else n
            numerator += mass * survivors
    return Fraction(numerator, denominator)


def baseline_scores(world: World, policy: str, *, variant: str = "clean") -> tuple[float, ...]:
    n = len(world.entries)
    if policy == "fifo":
        return tuple(float(i) for i in range(n))
    if policy == "random":
        rng = random.Random(int(digest([world.root_id, "retention-random"]), 16))
        return tuple(rng.random() for _ in range(n))
    if policy == "scope":
        return tuple(float(e.features[0]) for e in world.entries)
    if policy == "generator_rank":
        return tuple(float(weight(e.features)) for e in world.entries)
    if policy == "hindsight":
        # Privileged ceiling. This arm alone sees the final query and revision.
        return tuple(float(2 if i in world.query and (variant == "clean" or i != world.edited_source)
                           else int(i in world.query)) for i in range(n))
    raise ValueError("unknown policy")


@dataclass(frozen=True)
class Receipt:
    source: int
    version: int
    value: int
    receipt_id: str


def issue(world: World, source: int, variant: str) -> Receipt:
    updated = variant == "revised" and source == world.edited_source
    value = world.replacement if updated else world.entries[source].value
    version = 2 if updated else 1
    return Receipt(source, version, value, digest([world.root_id, source, version, value]))


def verify(world: World, variant: str, receipts: tuple[Receipt, ...], answer: int) -> bool:
    if variant not in VARIANTS:
        raise ValueError("invalid variant")
    expected = tuple(issue(world, i, variant) for i in world.query)
    return (tuple(sorted(receipts, key=lambda r: r.source)) == expected
            and type(answer) is int and answer == sum(r.value for r in expected) % 17)


def episode(world: World, scores: tuple[float, ...], capacity: int,
            verifier_budget: int, variant: str) -> dict:
    """Host computes the final answer from available receipts, isolating retention."""
    if len(scores) != len(world.entries) or variant not in VARIANTS or verifier_budget not in (0, 1):
        raise ValueError("invalid episode configuration")
    stored = retain(scores, capacity)
    bank = {i: issue(world, i, "clean") for i in stored}
    # An observed source revision invalidates the earlier receipt. It does not
    # deliver its replacement, and only the charged verifier can refresh it.
    invalidated = int(variant == "revised" and world.edited_source in bank)
    if invalidated:
        bank.pop(world.edited_source)
    hits = sum(i in bank for i in world.query)
    missing = [i for i in world.query if i not in bank]
    requested = missing[:verifier_budget]
    available = {i: bank[i] for i in world.query if i in bank}
    available.update({i: issue(world, i, variant) for i in requested})
    receipts = tuple(available.values())
    answer = sum(r.value for r in receipts) % 17 if len(receipts) == 2 else None
    success = answer is not None and verify(world, variant, receipts, answer)
    return {
        "root_id": world.root_id, "split": world.split, "variant": variant,
        "capacity": capacity, "verifier_budget": verifier_budget,
        "stored": list(stored), "query": list(world.query), "hits": hits,
        "invalidated": invalidated, "requested": requested, "verifier_calls": len(requested),
        "verified": success, "answer": answer,
        "receipt_ids": [r.receipt_id for r in sorted(receipts, key=lambda r: r.source)],
        "retained_payload_bytes": len(json.dumps([asdict(issue(world, i, "clean")) for i in stored],
                                             separators=(",", ":")).encode()),
    }
