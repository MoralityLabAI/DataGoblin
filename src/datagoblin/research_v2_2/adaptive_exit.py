"""A public-feature adaptive exit head for a fixed-horizon tied recurrence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def features(logits: np.ndarray, previous: np.ndarray, k: int, horizon: int) -> np.ndarray:
    if logits.shape != (3,) or previous.shape != (3,) or not 1 <= k <= horizon:
        raise ValueError("invalid recurrent state")
    p = np.exp(logits - logits.max())
    p /= p.sum()
    ordered = np.sort(p)
    return np.array([ordered[-1] - ordered[-2], ordered[-1],
                     -np.sum(p * np.log(np.maximum(p, 1e-12))),
                     np.linalg.norm(logits - previous),
                     float(logits.argmax() == previous.argmax()), k / horizon])


@dataclass
class AdaptiveExit:
    weights: np.ndarray
    bias: float
    threshold: float = 1.0

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, epochs: int = 200):
        if x.ndim != 2 or x.shape[1] != 6 or len(x) == 0 or y.shape != (len(x),):
            raise ValueError("invalid training set")
        if not np.isfinite(x).all() or not np.isin(y, (0, 1)).all():
            raise ValueError("invalid training data")
        w = np.zeros(6)
        b = 0.0
        for _ in range(epochs):
            p = 1 / (1 + np.exp(-np.clip(x @ w + b, -40, 40)))
            e = p - y
            w -= 0.1 * (x.T @ e / len(x) + 1e-4 * w)
            b -= 0.1 * float(e.mean())
        return cls(w, b)

    def probability(self, x: np.ndarray) -> float:
        return float(1 / (1 + np.exp(-np.clip(x @ self.weights + self.bias, -40, 40))))


def execute(initial: np.ndarray, step, head: AdaptiveExit, *, horizon: int = 24,
            min_passes: int = 3):
    """Execute actual steps only until exit; use one fixed damping horizon."""
    if not 1 <= min_passes <= horizon:
        raise ValueError("invalid pass bounds")
    current = np.array(initial, dtype=float, copy=True)
    for k in range(1, horizon + 1):
        previous = current.copy()
        current = np.asarray(step(previous.copy(), k, horizon), dtype=float)
        if not np.isfinite(current).all() or current.shape != (3,):
            raise ValueError("nonfinite or malformed recurrent output")
        if k >= min_passes and head.probability(features(current, previous, k, horizon)) >= head.threshold:
            return current, k
    return current, horizon


def choose_threshold(rows, *, accuracy_tolerance: float = 0.0025):
    """Calibrate on separate roots; rows are (threshold, accuracy, mean passes, raw FA).

    The reference row uses threshold 1.0 (full horizon). No confirmation data
    should enter this function.
    """
    full = [r for r in rows if r[0] == 1.0]
    if len(full) != 1:
        raise ValueError("exactly one full-horizon control required")
    reference = full[0]
    feasible = [r for r in rows if r[1] >= reference[1] - accuracy_tolerance
                and r[3] <= reference[3]]
    return min(feasible, key=lambda r: (r[2], -r[1], -r[0]))[0]
