"""Matched two-rival LDT games with a strict set-containment holdout.

Both splits use DataGoblin's three-candidate SELECTED and UNRESOLVED rules.
Training exposes both mask widths in both rival slots. A held frame contains
one singleton included in a doubleton. The receiver gets both masks, the
single-axis calibration request, and the native status flags. The sender gets
three material measurements and public frequency summaries before that query.
"""

from __future__ import annotations

import hashlib
from itertools import permutations

import numpy as np
from receipt_ldt import FLAGS, TOLERANCE, _candidate, _compare, _label

RELATIONS = {
    "train": ("disjoint_1_2", "disjoint_2_1", "overlap_2_2"),
    "test": ("contained_1_2", "contained_2_1"),
}
NATIVE = {
    "point_selected": "array_two_opponents",
    "interval_unresolved": "array_unresolved_after_challenge",
}


def _rng(seed, family, frame):
    data = f"native-ldt-relations-v2/{seed}/{family}/{frame}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(data).digest()[:16], "little"))


def _masks(relation, focal, partner, other):
    if relation == "overlap_2_2":
        axes = ((focal, partner), (focal, other))
    else:
        pair = (partner, other) if relation.startswith("disjoint_") else (focal, partner)
        axes = ((focal,), pair) if relation.endswith("_1_2") else (pair, (focal,))
    masks = np.zeros((2, 3), dtype=bool)
    for mask, positions in zip(masks, axes, strict=True):
        mask[list(positions)] = True
    return masks


def verified_side_sha256(raw, expected_sha256=None):
    """Validate native calibration results and hash the row-major uint8 side matrix."""
    producer, queries, _, side, _ = raw
    axis_mask = queries[:, 6:9]
    if not np.isin(axis_mask, [0, 1]).all() or not (axis_mask.sum(axis=1) == 1).all():
        raise ValueError("invalid public calibration axis")
    if side.shape != (8, len(queries)) or not np.isin(side, [0, 1]).all():
        raise ValueError("invalid binary calibration receipt")
    # A point comparison that holds all other metrics equal is LEFT exactly
    # when the selected metric exceeds the baseline by the native tolerance.
    expected = producer[:, :3][:, axis_mask.argmax(axis=1)] > 0.5 + TOLERANCE
    if not np.array_equal(side, expected):
        raise ValueError("side receipt disagrees with independent calibration predicate")
    digest = hashlib.sha256(np.ascontiguousarray(side, dtype=np.uint8).tobytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("side receipt digest mismatch")
    return digest


def _frame(seed, family, frame, relation, native, omitted):
    rng = _rng(seed, family, frame)
    low = rng.uniform(0.23, 0.38, 3)
    high = rng.uniform(0.63, 0.79, 3)
    overlap = rng.uniform(0.025, 0.055, 3)
    hidden = (np.arange(8)[:, None] >> np.arange(3)) & 1
    observed = np.where(hidden, high, low)
    reference = _candidate(np.full(3, 0.5))
    calibrated = np.empty_like(hidden)
    for w, values in enumerate(observed):
        for axis in range(3):
            isolated = np.full(3, 0.5)
            isolated[axis] = values[axis]
            calibrated[w, axis] = _compare(_candidate(isolated), reference) == "LEFT"

    # Each focal axis and calibration axis occurs twice. The four omitted
    # matchings give each frame a distinct set of six public contexts.
    queries = np.zeros((6, 12))
    labels = np.zeros((8, 6), dtype=np.int64)
    side = np.zeros_like(labels)
    for focal in range(3):
        companions = tuple(rng.permutation([a for a in range(3) if a != focal]))
        axes = [a for a in range(3) if a != omitted[focal]]
        for slot, axis in enumerate(axes):
            first, second = _masks(relation, focal, companions[slot], companions[1 - slot])
            q = 2 * focal + slot
            queries[q, :3], queries[q, 3:6] = first, second
            queries[q, 6 + axis] = 1
            queries[q, 9:] = FLAGS[native]
            side[:, q] = calibrated[:, axis]
            for w, values in enumerate(observed):
                labels[w, q] = _label(native, values, low, high, first, second, overlap)

    weights = rng.uniform(0.55, 1.35, len(queries))
    weights[np.arange(6) // 2 == frame % 3] *= 2.4
    weights /= weights.sum()
    first, second = queries[:, :3], queries[:, 3:6]
    # Union coverage would be identically one in training and always smaller
    # on nested masks. Average each rival's normalized exposure independently.
    focus = (first / first.sum(1, keepdims=True) + second / second.sum(1, keepdims=True)) / 2
    producer = np.zeros((8, 12))
    producer[:, :3] = observed
    producer[:, 3:6] = weights @ focus
    producer[:, 6:9] = weights @ queries[:, 6:9]
    producer[:, 9:] = FLAGS[native]
    raw = producer, queries, labels, side, weights
    verified_side_sha256(raw)
    return raw


def make_relations(split, seed):
    """Return four seeded frames per relation/status family with v1 tensor shapes."""
    if split not in RELATIONS:
        raise ValueError(f"unknown LDT relation split: {split!r}")
    games = {}
    omissions = tuple(permutations(range(3)))
    for relation in RELATIONS[split]:
        for status, native in NATIVE.items():
            family = f"{status}_{relation}"
            order = _rng(seed, family, "calibration-panels").permutation(len(omissions))
            for index, omit_index in enumerate(order[:4]):
                games[f"{family}/{index}"] = _frame(seed, family, index, relation, native, omissions[omit_index])
    return games
