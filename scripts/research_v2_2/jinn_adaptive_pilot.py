"""Run the frozen-horizon adaptive Jinn/Beast pilot on the private operator host.

Use the pinned v2.1 archive and exact old 600-root exclusion registry.
All learned parameters, panels and individual predictions stay outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "research_v2_1"))

from jinn_beast_depth_study import (  # noqa: E402
    LANE_TO_IDX,
    CorrectorHead,
    _fit_and_profile_tied,
    build_fresh_confirmation_panel,
)
from datagoblin.research_v2.specialist import (  # noqa: E402
    _load_manifest,
    _policy,
    _read_pinned,
    _score,
    root_id,
    semantic_digest,
)
from datagoblin.research_v2_2.adaptive_exit import AdaptiveExit  # noqa: E402
from datagoblin.research_v2_2.gates import _outside_git  # noqa: E402
from datagoblin.research_v2_2.jinn_adapter import evaluate, fit_and_calibrate  # noqa: E402

OLD_REGISTRY_SHA = "9ee74a2594354e4b2a96eab994e382c91a8586e8a1f40dc66a554ea821f96e29"
FRAMES = ("jinn", "beast")
SEEDS = (2026112801, 2026112802, 2026112803, 2026112804, 2026112805)


def dump(path, data):
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")


def encode(rows, tag_idx):
    x = np.zeros((len(rows), len(tag_idx)))
    for i, row in enumerate(rows):
        for tag in row["tags"]:
            x[i, tag_idx[tag]] = 1
    y = {f: np.array([LANE_TO_IDX[r["joint_label"][f]] for r in rows]) for f in FRAMES}
    return x, y


def root_bootstrap(deltas, *, seed=0):
    """Mean across fixed trained seeds/personas, then bootstrap whole roots."""
    x = np.asarray(deltas, dtype=float)
    if x.ndim != 3 or x.shape[1] != 2 or x.shape[2] != 600:
        raise ValueError("expected [seed, persona, root] paired contrasts")
    root = x.mean(axis=(0, 1))
    rng = np.random.default_rng(seed)
    b = np.concatenate([root[rng.integers(len(root), size=(100, len(root)))].mean(axis=1)
                        for _ in range(100)])
    return {"delta": float(root.mean()), "ci95": np.quantile(b, [.025, .975]).tolist(),
            "one_sided_lower95": float(np.quantile(b, .05)), "replicates": 10000,
            "seed_deltas": x.mean(axis=(1, 2)).tolist(),
            "resampling_unit": "root; both personas and fixed trained seeds grouped"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--spent-registry", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fresh-seed", type=int, required=True)
    args = p.parse_args()
    for path in (args.source_root, args.spent_registry, args.out):
        _outside_git(path)
    old = args.spent_registry.read_bytes()
    if hashlib.sha256(old).hexdigest() != OLD_REGISTRY_SHA:
        raise ValueError("v2.1 spent registry digest mismatch")
    reg = json.loads(old)
    old_roots = set(reg["held_root_ids"])
    old_sem = set(reg["held_semantic_sha256"])
    if len(old_roots) != 600 or len(old_sem) != 600:
        raise ValueError("spent panel inventory mismatch")
    if args.fresh_seed == 2026112788:
        raise ValueError("spent seed")
    args.out.mkdir(parents=True, exist_ok=False)

    manifest = _load_manifest(None)
    payload = {name: _read_pinned(args.source_root, spec) for name, spec in manifest["files"].items()}
    policies = {f: _policy(payload[f]) for f in FRAMES}
    low, high = Fraction(manifest["reject_maximum"]), Fraction(manifest["train_minimum"])
    rows = [json.loads(s) for s in payload["dataset"].decode("utf-8").splitlines() if s]
    roots = [root_id(r["tags"]) for r in rows]
    sems = [semantic_digest(r["tags"]) for r in rows]
    if (len(rows) != 2304 or len(set(roots)) != 2304 or len(set(sems)) != 2304
            or old_roots.intersection(roots) or old_sem.intersection(sems)):
        raise ValueError("archive or spent overlap")
    splits = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")}
    if tuple(len(v) for v in splits.values()) != (1440, 432, 432):
        raise ValueError("unexpected archive partition sizes")
    for row in rows:
        for f in FRAMES:
            if _score(row["tags"], policies[f], low, high)["lane"] != row["joint_label"][f]:
                raise ValueError("archive label mismatch")
    vocab = sorted(set().union(*(set(policies[f]["tags"]) for f in FRAMES)))
    tag_idx = {t: i for i, t in enumerate(vocab)}
    xt, yt = encode(splits["train"], tag_idx)
    xc, yc = encode(splits["validation"], tag_idx)
    tr_roots = [root_id(r["tags"]) for r in splits["train"]]
    cal_roots = [root_id(r["tags"]) for r in splits["validation"]]
    excluded = old_roots | {root_id(r["tags"]) for r in splits["test"]}
    reject, train = LANE_TO_IDX["reject"], LANE_TO_IDX["train"]
    trained = {}
    frozen = []
    for seed in SEEDS:
        for fi, frame in enumerate(FRAMES):
            own = sorted(tag_idx[t] for t in policies[frame]["tags"])
            crit = sorted(tag_idx[t] for t in policies[frame]["critical_tags"])
            fseed = seed * 10 + fi
            head = CorrectorHead(len(vocab), own_indices=own, critical_indices=crit,
                                 passes=24, grad_mode="exact_unrolled", seed=fseed)
            _fit_and_profile_tied(head, xt, yt[frame], seed=fseed)
            stop, cal = fit_and_calibrate(
                head, train_x=xt, train_y=yt[frame], train_roots=tr_roots,
                cal_x=xc, cal_y=yc[frame], cal_roots=cal_roots, excluded_roots=excluded,
                reject_index=reject, train_index=train,
            )
            trained[(seed, frame)] = (head, stop)
            frozen.append({"seed": seed, "frame": frame, "threshold": stop.threshold,
                           "weights": stop.weights.tolist(), "bias": stop.bias, "calibration": cal})
    dump(args.out / "frozen.private.json", {"heads": frozen, "seeds": SEEDS, "horizon": 24})

    # Generate fresh confirmation only after all exit thresholds are fixed.
    panel, panel_meta = build_fresh_confirmation_panel(
        tag_vocab=vocab, policies=policies, low=low, high=high, archive_rows=rows,
        excluded_roots=old_roots, excluded_sems=old_sem, n_roots=600,
        seed=args.fresh_seed, panel_out=args.out / "panel",
    )
    xf, yf = encode(panel, tag_idx)
    differences = []
    receipts = []
    for seed in SEEDS:
        seed_deltas = []
        for frame in FRAMES:
            head, stop = trained[(seed, frame)]
            full = AdaptiveExit(stop.weights, stop.bias, 1.0)
            old_result = evaluate(head, full, xf, yf[frame], reject_index=reject, train_index=train)
            new_result = evaluate(head, stop, xf, yf[frame], reject_index=reject, train_index=train)
            seed_deltas.append(np.asarray(new_result["guarded_correct"], dtype=float)
                               - np.asarray(old_result["guarded_correct"], dtype=float))
            for result in (old_result, new_result):
                result["p95_passes"] = float(np.quantile(result.pop("passes"), .95))
                result.pop("guarded_correct")
            receipts.append({"seed": seed, "frame": frame, "full": old_result, "adaptive": new_result})
        differences.append(seed_deltas)
    contrast = root_bootstrap(differences)
    mean_passes = float(np.mean([r["adaptive"]["mean_passes"] for r in receipts]))
    fa = {f: {a: sum(r[a]["raw_critical_fa"] for r in receipts if r["frame"] == f)
              for a in ("full", "adaptive")} for f in FRAMES}
    report = {"schema": "research-v2.2-adaptive-native-pilot-v1",
              "panel_sha256": panel_meta["panel_sha256"], "frozen_sha256": hashlib.sha256(
                  (args.out / "frozen.private.json").read_bytes()).hexdigest(),
              "roots": 600, "seeds": list(SEEDS), "results": receipts, "paired": contrast,
              "mean_adaptive_passes": mean_passes, "raw_critical_fa_by_frame": fa,
              "primary_pilot_gate": bool(mean_passes < 6 and contrast["delta"] >= -.0025
                                         and contrast["one_sided_lower95"] > -.0025
                                         and all(v["adaptive"] <= v["full"] for v in fa.values())),
              "scope": "same K24 head only; separate fixed-depth and exit-overhead controls pending"}
    dump(args.out / "aggregate.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
