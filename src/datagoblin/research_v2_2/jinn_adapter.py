"""Native v2.1 Jinn/Beast corrector adapter for v2.2 early exit."""

from __future__ import annotations

import numpy as np

from .adaptive_exit import AdaptiveExit, choose_threshold, execute, features


def native_step(head, active: float, critical: float):
    """One actual residual update of the frozen CorrectorHead."""
    def step(cur, k, horizon):
        if horizon != len(head.alphas):
            raise ValueError("damping horizon differs from trained head")
        p = np.exp(cur - cur.max())
        p /= p.sum()
        ordered = np.sort(p)
        feat = np.concatenate([p, [ordered[-1] - ordered[-2], active, critical]])
        delta = np.tanh(feat @ head.w1 + head.b1) @ head.w2 + head.b2
        return cur + head.alphas[k - 1] * delta
    return step


def verify_disjoint(*root_groups):
    seen = set()
    for group in root_groups:
        roots = set(group)
        if not roots or seen.intersection(roots):
            raise ValueError("empty or overlapping root partition")
        seen.update(roots)


def _states(head, x):
    base = head.base.logits(x)
    active, crit = head._static_feats(x)
    return zip(base, active[:, 0], crit[:, 0], strict=True)


def training_examples(head, x, y, *, reject_index: int, min_passes: int = 3):
    """Generate label-bearing exit examples exclusively from training roots."""
    feats, labels = [], []
    for (initial, active, critical), target in zip(_states(head, x), y, strict=True):
        step = native_step(head, active, critical)
        cur = initial.copy()
        trace = []
        for k in range(1, len(head.alphas) + 1):
            prev = cur.copy()
            cur = step(prev, k, len(head.alphas))
            if k >= min_passes:
                trace.append((features(cur, prev, k, len(head.alphas)), int(cur.argmax())))
        final = reject_index if critical else int(cur.argmax())
        for feat, raw in trace:
            guarded = reject_index if critical else raw
            # Never reward an early exit that loses an otherwise correct prediction.
            feats.append(feat)
            labels.append(int(guarded == target or final != target))
    return np.asarray(feats), np.asarray(labels)


def evaluate(head, stop_head: AdaptiveExit, x, y, *, reject_index: int, train_index: int):
    raw, guarded, passes, flags = [], [], [], []
    for initial, active, critical in _states(head, x):
        cur, k = execute(initial, native_step(head, active, critical), stop_head,
                         horizon=len(head.alphas))
        pred = int(cur.argmax())
        raw.append(pred)
        guarded.append(reject_index if critical else pred)
        flags.append(bool(critical and pred == train_index))
        passes.append(k)
    return {"guarded_accuracy": float(np.mean(np.asarray(guarded) == y)),
            "raw_accuracy": float(np.mean(np.asarray(raw) == y)),
            "raw_critical_fa": sum(flags), "mean_passes": float(np.mean(passes)),
            "corrector_macs": 108 * sum(passes), "passes": passes,
            "guarded_correct": (np.asarray(guarded) == y).tolist()}


def calibrate(head, stop_head, x, y, *, reject_index, train_index):
    rows = []
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.0):
        candidate = AdaptiveExit(stop_head.weights, stop_head.bias, threshold)
        result = evaluate(head, candidate, x, y, reject_index=reject_index, train_index=train_index)
        rows.append((threshold, result["guarded_accuracy"], result["mean_passes"],
                     result["raw_critical_fa"]))
    return AdaptiveExit(stop_head.weights, stop_head.bias, choose_threshold(rows)), rows


def fit_and_calibrate(head, *, train_x, train_y, train_roots, cal_x, cal_y,
                      cal_roots, excluded_roots, reject_index, train_index):
    if len(train_roots) != len(train_x) or len(cal_roots) != len(cal_x):
        raise ValueError("root membership does not match row count")
    verify_disjoint(train_roots, cal_roots, excluded_roots)
    x, y = training_examples(head, train_x, train_y, reject_index=reject_index)
    stop_head = AdaptiveExit.fit(x, y)
    return calibrate(head, stop_head, cal_x, cal_y,
                     reject_index=reject_index, train_index=train_index)
