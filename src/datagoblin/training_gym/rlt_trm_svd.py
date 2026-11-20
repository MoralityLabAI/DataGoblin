"""Spectral and causal analyses of a supervised, two-state recurrent loop."""

from __future__ import annotations

import time
from collections import deque
from itertools import product
from pathlib import Path

import numpy as np

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import canonical_json, load_structured

from . import rlt_trm_supervision as experiment
from .rlt_lean_labels import atomic_record

RANK = 2
PROBE_VISITS = (2, 4)
RANDOM_CONTROLS = 3
PROBE_FRACTIONS = (0.5, 1.0)
STATES = ("reasoning", "answer", "answer_update")


def collect(weights, inputs, arm, intervene=None):
    return tuple(np.stack(cycles) for cycles in zip(*experiment.roll(weights, inputs, arm, intervene), strict=True))


def spectrum(calibration, holdout):
    train = (calibration - calibration.mean(axis=1, keepdims=True)).reshape(-1, calibration.shape[-1])
    test = (holdout - holdout.mean(axis=1, keepdims=True)).reshape(-1, holdout.shape[-1])
    center = train.mean(axis=0)
    _, singular, vectors = np.linalg.svd(train - center, full_matrices=False)
    energy = singular**2
    total = energy.sum()
    share = energy / total if total > 0 else energy
    positive = share[share > 0]
    shifted = test - center
    projected = shifted @ vectors[:RANK].T
    held_total = float(np.square(shifted).sum())
    return center, vectors[:RANK], {
        "singular_values": singular.tolist(), "effective_rank": float(np.exp(-sum(positive * np.log(positive))))
        if total > 0 else 0.0,
        "held_top_energy": float(np.square(projected).sum()) / held_total if held_total > 0 else 0.0,
        "held_rms": float(np.sqrt(np.square(holdout).mean())),
        "held_centered_rms": float(np.sqrt(np.square(shifted).mean())),
    }


def matched_ablation(states, center, leading, random_basis, fraction=1.0):
    """Suppress leading components and rotate a random-subspace control by the same angle."""
    reference = states.mean(axis=1, keepdims=True) + center
    sample = states - reference
    shape = (len(states), 1, 1)
    radius = np.linalg.norm(sample, axis=(1, 2)).reshape(shape)
    reduced = sample - fraction * ((sample @ leading.T) @ leading)
    residual = np.linalg.norm(reduced, axis=(1, 2)).reshape(shape)
    guess = -((sample @ random_basis.T) @ random_basis)
    radius_sq = np.maximum(radius**2, 1e-28)
    perpendicular = guess - (np.sum(guess * sample, axis=(1, 2), keepdims=True) / radius_sq) * sample
    tangent = np.linalg.norm(perpendicular, axis=(1, 2)).reshape(shape)
    inner_product = np.sum(sample * reduced, axis=(1, 2), keepdims=True)
    cos_theta = np.clip(inner_product / np.maximum(radius * residual, 1e-28), 0, 1)
    tangent_top = reduced - (inner_product / radius_sq) * sample
    sin_theta = np.linalg.norm(tangent_top, axis=(1, 2)).reshape(shape) / np.maximum(residual, 1e-14)
    top = reduced * radius / np.maximum(residual, 1e-14)
    control = cos_theta * sample + sin_theta * radius * perpendicular / np.maximum(tangent, 1e-14)
    valid = (radius > 1e-12) & (residual > 1e-12) & (tangent > 1e-12)
    top, control = (np.where(valid, x, sample) + reference for x in (top, control))
    top_move, control_move = (np.linalg.norm(x - states, axis=(1, 2)) for x in (top, control))
    move_gap = float(np.max(np.abs(top_move - control_move)))
    norm_gap = max(float(np.max(np.abs(np.linalg.norm(x - reference, axis=(1, 2)) - radius.ravel())))
                   for x in (top, control))
    if max(move_gap, norm_gap) > 1e-7:
        raise ValueError("ablation and control should have matched perturbation and centered norm")
    return top, control, {
        "matched_examples": int(valid.sum()), "rms_perturbation": float(np.sqrt(np.square(top - states).mean())),
        "maximum_displacement_gap": move_gap, "maximum_centered_norm_error": norm_gap,
    }


def ablate(weights, held, arm, baseline, spectral_bases, seed, guard):
    baseline_correct = baseline[0][-1].argmax(axis=-1) == held[-1][:, -1]
    results = []
    for visit, role, fraction in product(PROBE_VISITS, range(2), PROBE_FRACTIONS):
        guard()
        center, vectors = spectral_bases[role][visit - 1]
        donor = baseline[role + 1][visit - 1]
        rng = np.random.default_rng(seed + 1011 * visit + 19 * role)
        outcomes, doses = {}, []
        for i in range(RANDOM_CONTROLS):
            rand = np.linalg.qr(rng.normal(size=(donor.shape[-1], RANK)))[0].T
            top, control, dose = matched_ablation(donor, center, vectors, rand, fraction)
            doses.append(dose)
            changes = [(f"random_{i + 1}", control)]
            if i == 0:
                changes.insert(0, ("top", top))
            for kind, changed in changes:
                guard()

                def replace_at(step, reason, answer, replacement=changed, selected=role, when=visit):
                    if step != when:
                        return reason, answer
                    return (replacement, answer) if selected == 0 else (reason, replacement)

                logits, _, _, _ = deque(experiment.roll(weights, held, arm, replace_at), maxlen=1).pop()
                correct = logits.argmax(axis=-1) == held[-1][:, -1]
                outcomes[kind] = {}
                for index, task in enumerate(experiment.TASKS):
                    items = held[2] == index
                    outcomes[kind][task] = {
                        "baseline_correct": int(baseline_correct[items].sum()),
                        "correct": int(correct[items].sum()),
                        "damaged": int((baseline_correct & ~correct & items).sum()),
                        "repaired": int((~baseline_correct & correct & items).sum()),
                    }
        results.append({"cycle": visit, "state": STATES[role], "fraction": fraction, "dose": doses,
                        "outcomes": outcomes})
    return results


def spectral_rollout(weights, calibration, held, arm, held_frames, guard):
    calibration_frames = collect(weights, calibration, arm)
    analysis, bases = {}, []
    for role, source, target in zip(STATES, calibration_frames[1:], held_frames[1:], strict=True):
        rows, frames = [], []
        for visit, (sample, test) in enumerate(zip(source, target, strict=True), start=1):
            guard()
            mean, leading, stats = spectrum(sample, test)
            if frames:
                overlap = leading @ frames[-1][1].T
                stats["previous_top_overlap"] = float(np.square(overlap).sum() / RANK)
            frames.append((mean, leading))
            rows.append({"cycle": visit, **stats})
        analysis[role] = rows
        bases.append(frames)
    return analysis, bases


def frozen_run(root, data_seed):
    protocol_path = root / "fit-protocol.json"
    protocol = load_structured(protocol_path)
    receipts = load_structured(root / "fit-receipts.json")
    if (
        protocol["experiment_sha256"] != sha256_file(Path(experiment.__file__))
        or protocol["dataset_sha256"] != sha256_file(root / "dataset.json")
        or receipts["protocol_sha256"] != sha256_file(protocol_path)
    ):
        raise ValueError("analysis inputs differ from the registered fit")
    for source in (experiment.cell, experiment.rlt_pilot):
        path = Path(source.__file__)
        if protocol["source_sha256"][path.name] != sha256_file(path):
            raise ValueError(f"trained dependency changed: {path.name}")
    expected = {(seed, supervision, arm) for seed in experiment.SEEDS
                for supervision, arms in experiment.SUPERVISION.items() for arm in arms}
    fits = receipts["fits"]
    if len(fits) != len(expected) or {(r["seed"], r["supervision"], r["arm"]) for r in fits} != expected:
        raise ValueError("matched fit matrix is incomplete")
    return experiment.freeze(root, data_seed), fits


def verify_ablation(panel):
    if len(panel) != 2 * len(PROBE_VISITS) * len(PROBE_FRACTIONS):
        raise ValueError("missing ablation conditions")
    return max(d["maximum_displacement_gap"] for probe in panel for d in probe["dose"])


def cycle_log(root, evaluation_records):
    path = root / "cycle-log.jsonl"
    pending = path.with_suffix(".part.jsonl")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for fit, data in evaluation_records:
            condition = {key: fit[key] for key in ("seed", "supervision", "arm")}
            history = root / "fits" / f"{fit['seed']}-{fit['supervision']}-{fit['arm']}" / "cycles.json"
            snapshots = load_structured(history)
            for step in sorted(snapshots, key=int):
                row = {"stage": "fit", **condition, "optimizer_step": int(step), **snapshots[step]}
                handle.write(canonical_json(row) + "\n")
            for depth, scores in data["scores"].items():
                for task, row in scores.items():
                    for i, number in enumerate(row["step_correct"]):
                        weights = experiment.cell.ARMS[fit["arm"]]
                        blend = [experiment.rlt_pilot.feedback_weight(x, i, int(depth)) for x in weights]
                        handle.write(canonical_json({
                            "stage": "held", **condition, "depth": int(depth), "task": task, "cycle": i + 1,
                            "items": row["items"], "step_correct": number,
                            "endpoint_correct": row["endpoint_correct"][i],
                            "eta": dict(zip(STATES[:2], blend, strict=True)),
                        }) + "\n")
    pending.replace(path)
    return sha256_file(path)


def evaluate(root, data_seed, guard):
    inputs, fits = frozen_run(root, data_seed)
    calibration = experiment.batch(*inputs["spectral_fit"], experiment.TRAIN_DEPTH)
    held = {depth: experiment.batch(*inputs["held"], depth) for depth in experiment.DEPTHS}
    registered = {
        "fit_receipts_sha256": sha256_file(root / "fit-receipts.json"), "source_sha256": sha256_file(Path(__file__)),
        "rank": RANK, "ablation_cycles": PROBE_VISITS, "ablation_fractions": PROBE_FRACTIONS,
        "random_controls": RANDOM_CONTROLS,
        "spectral_fit": "spectral_fit", "spectral_score": "held", "centering": "within_graph_nodes",
    }
    protocol = root / "eval-protocol.json"
    if protocol.exists() and sha256_json(load_structured(protocol)) != sha256_json(registered):
        raise ValueError("analysis protocol changed after registration")
    atomic_record(protocol, registered)
    directory = root / "evaluations"
    directory.mkdir(exist_ok=True)
    records, traces = [], {}
    for fit in fits:
        guard()
        seed, supervision, arm = (fit[key] for key in ("seed", "supervision", "arm"))
        key = f"{seed}-{supervision}-{arm}"
        folder, trace = root / "fits" / key, directory / f"{key}.json"
        if any(sha256_file(folder / file) != fit[digest]
               for file, digest in (("weights.npz", "weights_sha256"), ("cycles.json", "cycles_sha256"))):
            raise ValueError(f"fitted evidence changed: {key}")
        if not trace.exists():
            started = time.monotonic()
            with np.load(folder / "weights.npz", allow_pickle=False) as stored:
                weights = dict(stored)
            scores = {str(depth): experiment.score(batch, collect(weights, batch, arm)[0])
                      for depth, batch in held.items()}
            anchor = held[experiment.TRAIN_DEPTH]
            frames = collect(weights, anchor, arm)
            spectra, bases = spectral_rollout(weights, calibration, anchor, arm, frames, guard)
            ablation = ablate(weights, anchor, arm, frames, bases, seed, guard)
            swap = {}
            if arm in ("direct", "answer_rlt"):
                runtime = "answer_rlt" if arm == "direct" else "direct"
                switched = collect(weights, anchor, runtime)
                switch_spectra, _ = spectral_rollout(weights, calibration, anchor, runtime, switched, guard)
                swap = {"runtime": runtime, "scores": experiment.score(anchor, switched[0]), "spectra": switch_spectra}
            atomic_record(trace, {"key": key, "scores": scores, "spectra": spectra, "interventions": ablation,
                                  "runtime_swap": swap, "wall_s": time.monotonic() - started,
                                  "matched_control_maximum_gap": verify_ablation(ablation)})
            print(f"Scored and analyzed {key}", flush=True)
        records.append((fit, load_structured(trace)))
        traces[key] = sha256_file(trace)
    log_hash = cycle_log(root, records)
    atomic_record(root / "evidence.json", {"schema": "datagoblin.trm-supervision-svd.v1",
                  "eval_protocol_sha256": sha256_file(protocol), "fits": len(fits), "traces_sha256": traces,
                  "cycle_log_sha256": log_hash, "held_cases_per_task": experiment.SPLITS["held"],
                  "evaluations": [
                      {"seed": fit["seed"], "supervision": fit["supervision"], "arm": fit["arm"],
                       "scores": record["scores"], "runtime_swap_scores": record["runtime_swap"].get("scores")}
                      for fit, record in records
                  ]})
