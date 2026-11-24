"""Small communication games graded by DataGoblin's native interval-LDT policy.

The sender observes three measured material metrics. The receiver later sees one
comparison or array query and a separately verified single-metric calibration
receipt. The fourth metric, parameter bucket and safety qualification are held
fixed. Both agents know the deployment query distribution. Each world is
equiprobable and world w has high metric j iff (w >> j) & 1.

Producer columns: three observed metric centers, three query-weighted focal-axis
rates, three calibration-axis rates, and flags (interval, array, unresolved).
Receiver columns: first rival focus mask, second rival focus mask (zero for a
single rival), calibration-axis mask, and the same three flags. A family and
its threshold conventions are public. The side receipt is returned separately.
"""

from __future__ import annotations

import hashlib

import numpy as np

from datagoblin.experiments.ldt_array_hard_surface import (
    derive_array_target,
    interval_taste_policy,
    validate_hard_side,
)
from datagoblin.experiments.ldt_array_sorter import METRIC_FIELDS

FAMILIES = {
    "train": ("pair_clearance", "interval_hold", "array_one_opponent"),
    "test": ("array_two_opponents", "array_unresolved_after_challenge"),
}
TOLERANCE = 0.01
FLAGS = {
    "pair_clearance": (0, 0, 0),
    "interval_hold": (1, 0, 1),
    "array_one_opponent": (0, 1, 0),
    "array_two_opponents": (0, 1, 0),
    "array_unresolved_after_challenge": (1, 1, 1),
}


def _candidate(values, widths=None):
    """Construct a valid public hard-surface record on all four material metrics."""
    side = {
        "game": "receipt_ldt",
        "fixture_pressure": "material_metrics",
        "context_mode": "interval",
        "optimization_pressure": "ordinary_and_ood",
        "trainable_parameter_bucket": "medium",
        "safety_qualified": True,
    }
    if widths is None:
        widths = np.zeros(3)
    for name, value, width in zip(METRIC_FIELDS, (*values, 0.5), (*widths, 0.0), strict=True):
        side[name] = {"lower": float(value - width), "upper": float(value + width)}
    validate_hard_side(side)
    return side


def _compare(candidate, rival):
    return interval_taste_policy({"left": candidate, "right": rival}, tolerance=TOLERANCE)[0]


def _query(family, primary, partner, other, paired):
    first = np.zeros(3, dtype=bool)
    second = np.zeros(3, dtype=bool)
    first[primary] = True
    if family == "array_two_opponents":
        first[partner] = True
        second[[partner, other]] = True
    elif family == "array_unresolved_after_challenge":
        first[[primary, partner]] = True
        second[other] = True
    elif paired:
        first[partner] = True
    return first, second


def _label(family, observed, low, high, first, second, overlap):
    midpoint = (high + low) / 2
    if family in ("pair_clearance", "interval_hold"):
        # Only the focus dimensions are varied in this matched pair.
        candidate = _candidate(np.where(first, observed, 0.5))
        if family == "pair_clearance":
            rival = _candidate(np.where(first, midpoint, 0.5))
            return _compare(candidate, rival) == "LEFT"
        rival = _candidate(np.where(first, high, 0.5), first * overlap)
        return _compare(candidate, rival) == "ABSTAIN"

    # Strong off-focus performance prevents the focal candidate from dominating
    # an array rival. A mixed point comparison stays an irreducible tradeoff.
    candidates = [_candidate(observed), _candidate(np.where(first, midpoint, 0.92))]
    if second.any():
        uncertain = family == "array_unresolved_after_challenge"
        threshold = high if uncertain else midpoint
        width = second * overlap if uncertain else None
        candidates.append(_candidate(np.where(second, threshold, 0.92), width))
    target = derive_array_target(candidates, tolerance=TOLERANCE)["statuses"][0]
    return target == ("UNRESOLVED" if family == "array_unresolved_after_challenge" else "SELECTED")


def _game(family, index, seed):
    digest = hashlib.sha256(f"receipt-ldt-v1/{seed}/{family}/{index}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    low = rng.uniform(0.23, 0.38, 3)
    high = rng.uniform(0.63, 0.79, 3)
    overlap = rng.uniform(0.025, 0.055, 3)
    observed = np.where(((np.arange(8)[:, None] >> np.arange(3)) & 1) != 0, high, low)
    flags = FLAGS[family]
    queries = []
    for primary in range(3):
        for direction in (1, 2):
            partner = (primary + direction) % 3
            other = (primary - direction) % 3
            paired = direction == 1
            first, second = _query(family, primary, partner, other, paired)
            # A receiving site has a second lab's calibration on one material axis.
            calibration = partner if (index + direction) % 2 else other
            queries.append((first, second, calibration))
    weights = rng.uniform(0.55, 1.35, len(queries))
    weights[np.arange(len(queries)) // 2 == (index % 3)] *= 2.4
    weights /= weights.sum()
    receiver = np.zeros((len(queries), 12))
    labels = np.zeros((8, len(queries)), dtype=np.int64)
    side = np.zeros_like(labels)
    calibration_rival = _candidate(np.full(3, 0.5))
    # The match is against a fixed common calibration, independent of the late
    # target, the choice of rival, and all observations on the other two axes.
    calibration = np.empty((8, 3), dtype=np.int64)
    for world, values in enumerate(observed):
        for axis in range(3):
            masked = np.full(3, 0.5)
            masked[axis] = values[axis]
            calibration[world, axis] = _compare(_candidate(masked), calibration_rival) == "LEFT"
    for q, (first, second, axis) in enumerate(queries):
        receiver[q, :3], receiver[q, 3:6] = first, second
        receiver[q, 6 + axis] = 1
        receiver[q, 9:] = flags
        side[:, q] = calibration[:, axis]
        for w, values in enumerate(observed):
            labels[w, q] = _label(family, values, low, high, first, second, overlap)
    producer = np.zeros((8, 12))
    producer[:, :3] = observed
    producer[:, 3:6] = weights @ np.maximum(receiver[:, :3], receiver[:, 3:6])
    producer[:, 6:9] = weights @ receiver[:, 6:9]
    producer[:, 9:] = flags
    return producer, receiver, labels, side, weights


def make_games(split, seed):
    """Return four deterministic instances per family, six late queries each.

    Training arrays contain one rival. Both held-out families contain two.
    In the first a focal candidate must survive both point rivals. In the
    second a bounded-interval challenge must leave it unresolved after a
    distinct point rival has had a chance to dominate it.
    """
    if split not in FAMILIES:
        raise ValueError(f"unknown LDT split: {split!r}")
    return {
        f"{family}/{index}": _game(family, index, seed)
        for family in FAMILIES[split]
        for index in range(4)
    }
