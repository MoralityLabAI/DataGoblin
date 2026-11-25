"""Native public repair curriculum and independently sealed reasoning assessment."""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

LEGACY = Path(__file__).resolve().parents[1] / "aleph_rlt_placement"
sys.path.insert(0, str(LEGACY))
from corpus import REPAIR, SEP, Row, identity, rule_rows  # noqa: E402

from datagoblin.training_gym import rlt_revision as revision  # noqa: E402

SET, ADD = 16, 17
TRAIN_WORDS = ("S", "A", "AA", "SA", "AS", "AAA", "SAA", "ASA", "AAS")
HELD_WORDS = {
    "no_reset": ("AAAA", "AAAAA", "AAAAAA"),
    "initial_reset": ("SAAA", "SAAAA", "SAAAAA"),
    "interior_reset": ("ASAA", "AASAA", "AAASAA"),
    "multiple_resets": ("SSAA", "SASAA", "ASASAA"),
}
ORBIT_COUNT = 8
GRAPH_COUNT = 32


def _seed(*parts):
    return int(hashlib.sha256(repr(("aleph-confirm-reason-v1", *parts)).encode()).hexdigest(), 16)


def _execute(start, operations):
    value = start
    for op, arg in operations:
        if op not in (SET, ADD) or arg not in range(4):
            raise ValueError("invalid register operation")
        value = arg if op == SET else (value + arg) % 4
    return value


def _verify(start, operations):
    # The four possible input states form a truth table. Rotate the current table on ADD.
    table = (0, 1, 2, 3)
    for op, arg in operations:
        if op == SET:
            table = (arg,) * 4
        elif op == ADD:
            table = tuple((v + arg) % 4 for v in table)
        else:
            raise ValueError("unknown operation")
    return table[start]


def _orbit(word, rng, family):
    """Vary exactly the last reset operand (or initial state), keeping all other evidence fixed."""
    start, proposal = rng.randrange(4), rng.randrange(4)
    ops = [(SET if op == "S" else ADD, rng.randrange(4)) for op in word]
    reset = word.rfind("S")
    positions = list(range(4))
    rng.shuffle(positions)
    template = [list(op) for op in ops]
    if reset >= 0:
        template[reset][1] = "*"
    root = identity(["compact-repair", start if reset >= 0 else "*", template, proposal])
    rows = []
    for order, anchor in enumerate(positions):
        initial, modified = start, ops.copy()
        if reset >= 0:
            modified[reset] = (SET, anchor)
        else:
            initial = anchor
        target = _execute(initial, modified)
        if target != _verify(initial, modified):
            raise ValueError("register replay and truth table disagree")
        prompt = (REPAIR, proposal, SEP, initial, *(v for op in modified for v in op), SEP)
        rows.append(Row("repair", root, prompt, (target,), 4, family, order))
    if {r.good[0] for r in rows} != set(range(4)):
        raise ValueError("counterfactual orbit does not balance all four answers")
    return rows


def train_repair_rows(seed: int, step: int, n: int) -> list[Row]:
    """Counterfactual training examples from public length-one to length-three grammars."""
    rng = random.Random(_seed("train-repair", seed, step))
    result = []
    while len(result) < n:
        word = rng.choice(TRAIN_WORDS)
        result.extend(_orbit(word, rng, "public-train/" + word))
    return result[:n]


def train_rule_rows(seed: int, step: int, n: int) -> list[Row]:
    """Fresh public native revision examples, exclusively from the upstream training generator."""
    if n <= 0:
        return []
    rng = random.Random(_seed("train-revision", seed, step))
    rows = list(rule_rows(revision.episodes(rng.getrandbits(64), "train", (n + 3) // 4)))
    rng.shuffle(rows)
    return rows[:n]
