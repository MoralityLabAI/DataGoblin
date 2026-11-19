"""Cross frozen language-model training and runtime feedback, depth, and clocks."""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path

import numpy as np

from datagoblin.hashing import sha256_file
from datagoblin.serialization import canonical_json, dump_structured, load_structured

from .rlt_pilot import (
    BOS,
    MODES,
    TRAIN_DEPTH,
    eval_known,
    feedback_weight,
    logits_and_states,
    normalize,
    pending_rows,
    target_nll,
)
from .rlt_pilot_data import TEXT

DEPTHS = (1, 2, 4, 8, 16)
CLOCKS = ("native", "phase", "step", "phase_and_step")


def clocks(name, depth):
    phase = TRAIN_DEPTH if name in {"phase", "phase_and_step"} else depth
    step = TRAIN_DEPTH if name in {"step", "phase_and_step"} else depth
    return phase, step


def model_receipts(root):
    frozen_path = root / "frozen" / "manifest.json"
    frozen = load_structured(frozen_path)
    run = root / "run"
    protocol_path = run / "protocol.json"
    protocol = load_structured(protocol_path)
    if load_structured(run / "run.json")["status"] != "complete":
        raise ValueError("the frozen language-model fits must be complete")
    if sha256_file(frozen_path) != protocol["frozen_manifest_sha256"]:
        raise ValueError("the frozen manifest differs from the fitting protocol")
    known_path = root / "frozen" / "known.json"
    if sha256_file(known_path) != frozen["known_sha256"]:
        raise ValueError("frozen public items changed")
    receipts = {}
    matched = set()
    for mode in MODES:
        arm = run / mode
        fit = load_structured(arm / "train.json")
        if fit["feedback"] != mode or fit["checkpoint_sha256"] != sha256_file(arm / "weights.npz"):
            raise ValueError("checkpoint differs from the fitting receipt")
        matched.add(tuple(fit[k] for k in ("seed", "init_sha256", "unique_parameters", "steps", "train_depth",
                                            "text_target_bytes", "rule_targets")))
        receipts[mode] = fit
    if len(matched) != 1:
        raise ValueError("unmatched checkpoint fits")
    return sha256_file(protocol_path), frozen["known_sha256"], receipts, load_structured(known_path)


def text_rows(model, mode, depth, items, time_scale, guard):
    # Small equal-length batches retain the original per-document byte target definition.
    for start in range(0, len(items), 4):
        guard()
        batch = items[start : start + 4]
        targets = np.asarray([item["bytes"] for item in batch])
        inputs = np.column_stack((np.full(len(batch), BOS), targets[:, :-1]))
        logits = logits_and_states(model, inputs, mode, depth, clock_depths=time_scale)[0]
        pieces = target_nll(logits.reshape(-1, logits.shape[-1]), targets.ravel()).reshape(targets.shape)
        for item, nll in zip(batch, pieces.sum(axis=1), strict=True):
            yield {"id": item["id"], "nll_nats": float(nll), "bytes": len(item["bytes"])}


def rms(values):
    return float(np.sqrt(np.mean(values**2)))


def pulse_rows(model, mode, depth, known, guard):
    for item in known[TEXT[0]][:8]:
        guard()
        ids = np.asarray([[BOS, *item["bytes"][:-1]]])
        last = []
        clean = logits_and_states(model, ids, mode, depth, final_endpoint=last)[0]
        rng = np.random.default_rng(int(item["id"][:16], 16))
        delta = rng.standard_normal(last[0].shape).astype(np.float32)
        delta *= 0.01 / rms(delta)
        input_rms = rms(delta)
        for kind, visit in (("initial", None), ("early_proposal", 0), ("late_proposal", depth - 2)):
            noisy_endpoint = []
            options = {"delta": delta} if visit is None else {"feedback_pulse": (visit, delta)}
            noisy = logits_and_states(model, ids, mode, depth, final_endpoint=noisy_endpoint, **options)[0]
            scale = rms(last[0] - last[0].mean(axis=-1, keepdims=True))
            yield {
                "id": item["id"], "perturbation": kind, "pulse_visit": visit, "input_rms": input_rms,
                "feedback_weight": 1 if visit is None else feedback_weight(mode, visit, depth),
                "endpoint_rms": rms(last[0]), "endpoint_centered_rms": scale,
                "endpoint_gain": rms(noisy_endpoint[0] - last[0]) / input_rms,
                "normalized_endpoint_gain": rms(normalize(noisy_endpoint[0]) - normalize(last[0])) / input_rms,
                "logit_gain": rms(noisy - clean) / input_rms,
                "top1_flip_count": int(np.count_nonzero(noisy.argmax(-1) != clean.argmax(-1))),
            }


def evaluate(root: Path, out: Path, guard):
    training_hash, known_hash, fits, known = model_receipts(root)
    source = (Path(__file__), Path(__file__).with_name("rlt_pilot.py"))
    protocol = {
        "schema": "datagoblin.rlt-lm-crossover.v1", "depths": DEPTHS, "clocks": CLOCKS, "feedback": MODES,
        "training_protocol_sha256": training_hash, "known_sha256": known_hash,
        "sha256": {path.name: sha256_file(path) for path in source}, "fits": fits,
    }
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "protocol.json"
    if frozen.exists() and canonical_json(load_structured(frozen)) != canonical_json(protocol):
        raise ValueError("the frozen crossover protocol changed")
    if not frozen.exists():
        dump_structured(frozen, protocol)
    for trained in MODES:
        with np.load(root / "run" / trained / "weights.npz", allow_pickle=False) as data:
            weights = dict(data)
        for mode, depth in product(MODES, DEPTHS):
            clock_modes = CLOCKS if depth in (8, 16) else ("native",)
            for clock in clock_modes:
                path = out / f"{trained}--{mode}--{depth}--{clock}.jsonl"
                if path.exists():
                    continue
                with pending_rows(path) as handle:
                    for name, items in known.items():
                        if name in TEXT:
                            rows = text_rows(weights, mode, depth, items, clocks(clock, depth), guard)
                        elif clock == "native" and depth in (4, 8):
                            rows = known_rows(weights, mode, depth, name, items, guard)
                        else:
                            continue
                        for row in rows:
                            handle.write(canonical_json({"benchmark": name, **row}) + "\n")
            if depth in (4, 8, 16):
                path = out / f"pulse--{trained}--{mode}--{depth}.jsonl"
                if not path.exists():
                    with pending_rows(path) as handle:
                        for row in pulse_rows(weights, mode, depth, known, guard):
                            handle.write(canonical_json(row) + "\n")
        print(canonical_json({"trained_feedback": trained, "status": "evaluated"}), flush=True)


def known_rows(weights, mode, depth, name, items, guard):
    for item in items:
        guard()
        yield {"id": item["id"], **eval_known(weights, mode, depth, name, item)}


def bits_per_byte(rows):
    return math.fsum(r["nll_nats"] for r in rows) / (math.log(2) * sum(r["bytes"] for r in rows))
