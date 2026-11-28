"""Exhaust the exact finite support of the v2.1 synthetic router teacher.

This is synthetic policy imitation, not native episode performance.
Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from datagoblin.research_v2_1.controllers import (
    RLTStateCorrector,
    TRMActionRouter,
    _generate_controller_training_data,
    _softmax,
)
from datagoblin.research_v2_2.controllers import SelectiveCorrector, rule_class

SEEDS = (2026112701, 2026112801, 2026112802, 2026112803, 2026112804)
ACTS = ((.12, -1), (.352, 0), (.352, 1), (.176, 2))


def binary(p):
    return ((1 - p, 0), (p, 1))


def vector(learner, verified, pair, trained, base, turn, exp_err,
           check_err, invalid, act, extra):
    return (learner, verified, pair, trained, base, turn / 8, exp_err,
            check_err, invalid, int(act >= 0), int(act == 0), int(act == 1),
            int(act == 2), extra)


def support():
    rows = {}
    regimes = ((.25, 0), (.25, 1), (.20, 2), (.30, 3))
    for (pr, regime), turn, (pa, act), (pc, ce), (pi, inv) in product(
            regimes, range(8), ACTS, binary(.45), binary(.35)):
        for pe, extra in binary(.45) if act == 0 else ((1., 0),):
            key = vector(1, 0, int(regime == 3), int(regime == 1), int(regime == 2),
                         turn, 0, ce, inv, act, extra)
            rows[key] = .45 * pr / 8 * pa * pc * pi * pe
    for turn, (pa, act), (pv, verified), (pe, ee), (pc, ce) in product(
            range(8), ACTS, binary(.35), binary(.5), binary(.25)):
        key = vector(0, verified, 0, 0, 0, turn, ee, ce, 0, act, 0)
        rows[key] = .55 / 8 * pa * pv * pe * pc
    x = np.array(list(rows), dtype=float)
    p = np.array(list(rows.values()))
    if len(x) != 896 or not np.isclose(p.sum(), 1.):
        raise RuntimeError("synthetic support contract changed")
    return x, p


def metrics(pred, truth, mass, seen):
    correct = pred == truth
    unseen = ~seen
    return {
        "uniform_accuracy": float(correct.mean()),
        "generator_weighted_accuracy": float(np.dot(correct, mass)),
        "seen_accuracy": float(correct[seen].mean()),
        "unseen_accuracy": float(correct[unseen].mean()),
        "unseen_generator_weighted_accuracy": float(np.dot(correct[unseen], mass[unseen]) / mass[unseen].sum()),
        "macro_class_accuracy": float(np.mean([correct[truth == i].mean() for i in range(5)])),
        "per_class_accuracy": [float(correct[truth == i].mean()) for i in range(5)],
        "false_stop_classes": int(np.sum(np.isin(pred, (1, 2)) & ~np.isin(truth, (1, 2)))),
    }


def study():
    x, mass = support()
    truth = np.array([rule_class(row) for row in x])
    index = {tuple(row): i for i, row in enumerate(x)}
    expected_seen = 1 - (1 - mass) ** 1600
    runs = []
    for seed in SEEDS:
        xt, _, yt = _generate_controller_training_data(seed=seed, n_samples=1600)
        seen = np.zeros(len(x), dtype=bool)
        for row, target in zip(xt, yt, strict=True):
            idx = index[tuple(row)]
            if truth[idx] != target:
                raise ValueError("teacher mismatch")
            seen[idx] = True
        router = TRMActionRouter(seed=seed)
        router.fit(xt, yt)
        corrector = RLTStateCorrector(router, passes=3, seed=seed + 1)
        corrector.fit(xt, yt)
        logits, _, _ = router.forward(x)
        p = _softmax(logits)
        trm = p.argmax(axis=1)
        rlt_logits, _ = corrector.forward_from_logits(logits, x)
        rlt = rlt_logits.argmax(axis=1)
        order = np.sort(p, axis=1)
        selected = order[:, -1] - order[:, -2] < .20
        selective = np.where(selected, rlt, trm)
        # Check vectorized decisions against actual single-state execution.
        native = SelectiveCorrector(router, corrector)
        for i in range(len(x)):
            if native.predict(x[i])[0] != selective[i]:
                raise ValueError("vectorized and native selective predictions disagree")
        runs.append({
            "seed": seed,
            "unique_train_features": int(seen.sum()),
            "unseen_features": int((~seen).sum()),
            "seen_generator_mass": float(mass[seen].sum()),
            "rule": metrics(truth, truth, mass, seen),
            "trm": metrics(trm, truth, mass, seen),
            "rlt": metrics(rlt, truth, mass, seen),
            "selective": metrics(selective, truth, mass, seen),
            "rlt_changed": int((rlt != trm).sum()),
            "rlt_fixed": int(((trm != truth) & (rlt == truth)).sum()),
            "rlt_broke": int(((trm == truth) & (rlt != truth)).sum()),
            "selective_compute": native.accounting(),
            "always_rlt_passes": 3 * len(x),
            "selective_fraction_weighted": float(mass[selected].sum()),
        })
    return {
        "schema": "research-v2.2-routing-support-audit-v1",
        "scope": "exact synthetic teacher imitation; no actor or research-capability claim",
        "support_size": len(x),
        "class_counts": np.bincount(truth, minlength=5).tolist(),
        "class_probability": np.bincount(truth, weights=mass, minlength=5).tolist(),
        "expected_unique_train_features": float(expected_seen.sum()),
        "expected_seen_test_probability": float(np.dot(expected_seen, mass)),
        "training_draws_per_seed": 1600,
        "runs": runs,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    result = json.dumps(study(), sort_keys=True, indent=2, allow_nan=False)
    if args.out is not None:
        if args.out.exists():
            raise ValueError("fresh output required")
        args.out.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
