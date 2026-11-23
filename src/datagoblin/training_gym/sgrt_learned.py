"""DG-SGRT-02 CPU learned low-rank proposal heads with geometry controls.

All full-domain supervision comes from public synthetic training roots.
Evaluation targets remain in evaluator-owned SyntheticTask objects.
The geometry is frozen; a task-conditioned gate and low-rank maps are trained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import Any

import numpy as np

from .sgrt_holoarc import (
    BASE_DIM,
    FIBER_DIM,
    random_smooth_path,
    spiro_path,
    step_transport,
    trace_transport,
)
from .sgrt_holoarc_gym import (
    OPERATORS,
    Program,
    SyntheticTask,
    execute,
    legal_one_edit_candidates,
    make_task,
    program_coordinate,
    propose,
    verify_and_score,
)

TASK_DIM = 85
CANDIDATE_DIM = 47
RANK = FIBER_DIM
TRAINED_ARMS = (
    "semantic_lowrank",
    "flat_lowrank",
    "holoarc_gear_lowrank",
    "holoarc_random_path_lowrank",
    "holoarc_step_shuffle_lowrank",
)
EVAL_ARMS = (*TRAINED_ARMS, "rigid_top1_proxy", "semantic_heuristic", "extra_refinement_proxy")
PUBLIC_TRAIN_SEED = 31001
PUBLIC_EVAL_SEED = 61001


def _canonical(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _seed_from_public(task: dict[str, Any]) -> int:
    visible = {"anchor": task["anchor"], "demonstrations": task["demonstrations"]}
    return int(hashlib.sha256(_canonical(visible)).hexdigest()[:15], 16)


def _onehot(index: int, size: int) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float64)
    vector[index] = 1.0
    return vector


def task_features(public: dict[str, Any]) -> np.ndarray:
    """Public task features. Target and generator family are intentionally unused."""
    anchor: Program = tuple(public["anchor"])  # type: ignore[assignment]
    inputs, outputs, delta, residual = (np.zeros(16, dtype=np.float64) for _ in range(4))
    demos = public["demonstrations"]
    if len(demos) != 4:
        raise ValueError("expected four demonstrations")
    for x, y in demos:
        if not 0 <= x < 16 or not 0 <= y < 16:
            raise ValueError("demonstrations require four-bit values")
        inputs[x] += 0.25
        outputs[y] += 0.25
        delta[x ^ y] += 0.25
        residual[execute(anchor, x) ^ y] += 0.25
    features = np.concatenate((
        inputs,
        outputs,
        delta,
        residual,
        _onehot(OPERATORS.index(anchor[0]), len(OPERATORS)),
        _onehot(OPERATORS.index(anchor[1]), len(OPERATORS)),
        np.ones(1, dtype=np.float64),
    ))
    assert features.shape == (TASK_DIM,)
    return features


def candidate_features(candidate: Program, anchor: Program) -> np.ndarray:
    """Cheap fixed syntactic/Fourier chart with no candidate execution."""
    return np.concatenate((
        _onehot(OPERATORS.index(candidate[0]), len(OPERATORS)),
        _onehot(OPERATORS.index(candidate[1]), len(OPERATORS)),
        np.array([float(candidate[0] != anchor[0]), float(candidate[1] != anchor[1])]),
        program_coordinate(candidate),
        program_coordinate(candidate) - program_coordinate(anchor),
        np.ones(1, dtype=np.float64),
    ))


def _direct_context(features: np.ndarray) -> np.ndarray:
    """Matched non-geometric context-dependent orthogonal transform."""
    return step_transport(1.3 * (features[:BASE_DIM] - 0.125), mode="curved")


def _reorder_steps(path: np.ndarray, *, seed: int) -> np.ndarray:
    """Permute the same edge increments while preserving closed-loop distance."""
    increments = np.diff(path, axis=0)
    indices = np.random.default_rng(seed + 37).permutation(len(increments))
    output = [path[0]]
    output.extend(accumulate(increments[indices], lambda position, step: position + step))
    points = np.stack(output)
    points[-1] = path[0]
    return points


def context_transport(public: dict[str, Any], *, arm: str) -> np.ndarray:
    """Frozen task-conditioned geometry, shared param count in learned heads."""
    if arm not in TRAINED_ARMS:
        raise ValueError(f"unknown trained arm: {arm}")
    features = task_features(public)
    if arm == "semantic_lowrank":
        return _direct_context(features)
    seed = _seed_from_public(public)
    center = np.zeros(BASE_DIM, dtype=np.float64)
    path = (
        random_smooth_path(center, seed=seed)
        if arm == "holoarc_random_path_lowrank"
        else spiro_path(center, seed=seed)
    ) * 3.0
    if arm == "holoarc_step_shuffle_lowrank":
        path = _reorder_steps(path, seed=seed)
    mode = "flat" if arm == "flat_lowrank" else "curved"
    return trace_transport(path, mode=mode).holonomy


@dataclass
class LowRankHead:
    task_matrix: np.ndarray
    candidate_matrix: np.ndarray
    gate_vector: np.ndarray

    @classmethod
    def initialize(cls, *, seed: int) -> LowRankHead:
        rng = np.random.default_rng(seed)
        return cls(
            rng.normal(0, 0.06, size=(TASK_DIM, RANK)),
            rng.normal(0, 0.06, size=(CANDIDATE_DIM, RANK)),
            rng.normal(0, 0.02, size=TASK_DIM),
        )

    @property
    def parameter_count(self) -> int:
        return self.task_matrix.size + self.candidate_matrix.size + self.gate_vector.size

    def digest(self) -> str:
        digest = hashlib.sha256()
        for array in (self.task_matrix, self.candidate_matrix, self.gate_vector):
            digest.update(array.astype("<f8", copy=False).tobytes())
        return digest.hexdigest()

    def logits(
        self,
        features: np.ndarray,
        candidates: np.ndarray,
        transport: np.ndarray,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
        latent = features @ self.task_matrix
        gate = 1.0 / (1.0 + math.exp(-float(np.clip(features @ self.gate_vector, -30, 30))))
        displacement = (transport - np.eye(RANK)) @ latent
        updated = latent + gate * displacement
        choices = candidates @ self.candidate_matrix
        scores = (choices @ updated) / math.sqrt(RANK)
        return scores, (latent, displacement, gate, updated)


def _training_targets(task: SyntheticTask, candidates: tuple[Program, ...]) -> np.ndarray:
    """The exact 16-input oracle is used solely inside the training partition."""
    return np.array([
        all(execute(p, x) == execute(task.target, x) for x in range(16))
        for p in candidates
    ], dtype=bool)


@dataclass(frozen=True)
class TrainingItem:
    features: np.ndarray
    candidates: np.ndarray
    positives: np.ndarray
    transport: np.ndarray


def _training_items(*, roots: int, arm: str) -> tuple[TrainingItem, ...]:
    items = []
    for index in range(roots):
        if index % 4 == 3:
            continue  # withheld mixed-composition family
        task = make_task(seed=PUBLIC_TRAIN_SEED, root=index)
        public = task.public()
        pool = legal_one_edit_candidates(task.anchor)
        targets = _training_targets(task, pool)
        if not targets.any():
            raise AssertionError("target should have a legal one-edit neighbor")
        items.append(TrainingItem(
            task_features(public),
            np.stack([candidate_features(p, task.anchor) for p in pool]),
            targets,
            context_transport(public, arm=arm),
        ))
    return tuple(items)


def _loss_gradients(head: LowRankHead, item: TrainingItem) -> tuple[float, tuple[np.ndarray, ...]]:
    logits, (_latent, displacement, gate, updated) = head.logits(
        item.features, item.candidates, item.transport
    )
    weights = np.exp(logits - np.max(logits))
    probabilities = weights / weights.sum()
    positives = weights * item.positives
    total_positive = positives.sum()
    if total_positive <= 0:
        raise ValueError("training item has no exact positive candidate")
    loss = float(np.log(weights.sum()) - np.log(total_positive))
    ds = probabilities - (positives / total_positive)
    candidate_latents = item.candidates @ head.candidate_matrix
    dup = (candidate_latents.T @ ds) / math.sqrt(RANK)
    grad_candidate = item.candidates.T @ np.outer(ds, updated) / math.sqrt(RANK)
    grad_task = np.outer(
        item.features,
        dup + gate * ((item.transport - np.eye(RANK)).T @ dup),
    )
    grad_gate = item.features * float(dup @ displacement) * gate * (1.0 - gate)
    return loss, (grad_task, grad_candidate, grad_gate)


def fit(
    *,
    arm: str,
    seed: int,
    train_roots: int = 128,
    updates: int = 384,
    lr: float = 0.025,
) -> tuple[LowRankHead, dict[str, Any]]:
    """Same Adam initialization and root order for every trained condition."""
    if arm not in TRAINED_ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if not 16 <= train_roots <= 384 or not 16 <= updates <= 1500 or lr <= 0:
        raise ValueError("bounds: roots 16..384, updates 16..1500, lr positive")
    items = _training_items(roots=train_roots, arm=arm)
    rng = np.random.default_rng(seed + 411)
    order = np.arange(len(items))
    rng.shuffle(order)
    head = LowRankHead.initialize(seed=seed)
    parameters = (head.task_matrix, head.candidate_matrix, head.gate_vector)
    moments = [np.zeros_like(p) for p in parameters]
    second = [np.zeros_like(p) for p in parameters]
    losses: list[float] = []
    for step in range(1, updates + 1):
        if (step - 1) % len(items) == 0:
            rng.shuffle(order)
        item = items[int(order[(step - 1) % len(items)])]
        loss, gradients = _loss_gradients(head, item)
        losses.append(loss)
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients))
        clip = min(1.0, 5.0 / max(1e-12, norm))
        for i, (weight, gradient) in enumerate(zip(parameters, gradients, strict=True)):
            adjusted = gradient * clip + 1e-5 * weight
            moments[i] *= 0.9
            moments[i] += 0.1 * adjusted
            second[i] *= 0.999
            second[i] += 0.001 * adjusted * adjusted
            weight -= lr * (moments[i] / (1 - 0.9**step)) / (
                np.sqrt(second[i] / (1 - 0.999**step)) + 1e-8
            )
    return head, {
        "arm": arm,
        "train_roots": len(items),
        "requested_roots": train_roots,
        "updates": updates,
        "first32_loss_mean": float(np.mean(losses[:32])),
        "last32_loss_mean": float(np.mean(losses[-32:])),
        "trainable_parameters": head.parameter_count,
        "weights_sha256": head.digest(),
    }


def ranked_candidates(
    head: LowRankHead, public: dict[str, Any], *, arm: str, budget: int
) -> tuple[Program, ...]:
    if not 1 <= budget <= 16:
        raise ValueError("budget must be 1..16")
    anchor: Program = tuple(public["anchor"])  # type: ignore[assignment]
    pool = legal_one_edit_candidates(anchor)
    candidate_view = np.stack([candidate_features(p, anchor) for p in pool])
    scores, _ = head.logits(task_features(public), candidate_view, context_transport(public, arm=arm))
    sorted_pool = sorted(zip(scores, pool, strict=True), key=lambda row: (-float(row[0]), row[1]))
    return tuple(candidate for _score, candidate in sorted_pool[:budget])


def evaluate(
    trained: dict[str, LowRankHead], *, eval_roots: int = 64, budget: int = 8, eval_seed: int = PUBLIC_EVAL_SEED
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 4 <= eval_roots <= 256 or not 1 <= budget <= 16:
        raise ValueError("eval roots 4..256, candidate budget 1..16")
    if set(trained) != set(TRAINED_ARMS):
        raise ValueError("all registered trained arms are required")
    rows: list[dict[str, Any]] = []
    for index in range(eval_roots):
        task = make_task(seed=eval_seed, root=index)
        public = task.public()
        group = "family_ood_mixed" if task.family == "mixed" else "new_roots_seen_families"
        outcomes = {}
        for arm in EVAL_ARMS:
            if arm in trained:
                proposals = ranked_candidates(trained[arm], public, arm=arm, budget=budget)
            else:
                proposals, _ = propose(public, seed=_seed_from_public(public), arm=arm, budget=budget)
            score = verify_and_score(task, proposals)
            outcomes[arm] = {
                "whole_task_solved": score["whole_task_solved"],
                "target_reachable": score["target_reachable_in_set"],
                "first_demo_hit": score["first_demo_hit"],
                "verifier_calls": score["verifier_calls"],
                "candidate_set_sha256": score["candidate_set_sha256"],
            }
        rows.append({
            "root_id": task.root_id,
            "family": task.family,
            "group": group,
            "arms": outcomes,
        })
    report = {}
    for group in ("new_roots_seen_families", "family_ood_mixed", "all"):
        selected = [r for r in rows if group == "all" or r["group"] == group]
        report[group] = {}
        for arm in EVAL_ARMS:
            scores = [r["arms"][arm] for r in selected]
            report[group][arm] = {
                "roots": len(selected),
                "whole_task_solved": sum(int(s["whole_task_solved"]) for s in scores),
                "target_reachable": sum(int(s["target_reachable"]) for s in scores),
                "verifier_calls": sum(int(s["verifier_calls"]) for s in scores),
            }
    return rows, report



def paired_contrasts(rows: list[dict[str, Any]], *, replicates: int = 1024) -> dict[str, Any]:
    """Root-paired exploratory bootstrap; fixed RNG and no selection on outputs."""
    report: dict[str, Any] = {}
    baseline = "semantic_lowrank"
    for group in ("new_roots_seen_families", "family_ood_mixed", "all"):
        eligible = [r for r in rows if group == "all" or r["group"] == group]
        report[group] = {}
        for arm in TRAINED_ARMS:
            if arm == baseline:
                continue
            delta = np.array([
                int(row["arms"][arm]["whole_task_solved"]) - int(row["arms"][baseline]["whole_task_solved"])
                for row in eligible
            ], dtype=np.float64)
            if len(delta) == 0:
                raise ValueError("root group has no evaluation items")
            stable_seed = int(hashlib.sha256((group + arm).encode("utf-8")).hexdigest()[:12], 16)
            rng = np.random.default_rng(stable_seed)
            indices = rng.integers(0, len(delta), size=(replicates, len(delta)))
            boot = 100 * np.mean(delta[indices], axis=1)
            limits = np.quantile(boot, (0.025, 0.975))
            report[group][arm] = {
                "roots": len(delta),
                "absolute_effect_pp": 100 * float(np.mean(delta)),
                "root_bootstrap_95_pp": [float(limits[0]), float(limits[1])],
                "wins": int(np.sum(delta > 0)),
                "losses": int(np.sum(delta < 0)),
                "ties": int(np.sum(delta == 0)),
                "status": "exploratory_public_development",
            }
    return report

def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(_canonical(row).decode("utf-8") + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--train-roots", type=int, default=128)
    parser.add_argument("--eval-roots", type=int, default=64)
    parser.add_argument("--updates", type=int, default=384)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=PUBLIC_EVAL_SEED)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be absent or empty")
    models = {}
    logs = {}
    for arm in TRAINED_ARMS:
        models[arm], logs[arm] = fit(
            arm=arm, seed=args.seed, train_roots=args.train_roots, updates=args.updates
        )
    rows, results = evaluate(models, eval_roots=args.eval_roots, budget=args.budget, eval_seed=args.eval_seed)
    args.out.mkdir(parents=True, exist_ok=True)
    rows_sha = _write_jsonl(args.out / "evaluation_scores.jsonl", rows)
    report = {
        "schema_version": "dg-sgrt-02-learned-public-development-v1",
        "claim_boundary": "CPU-trained synthetic development, no sealed or 1B-model result",
        "train_seed": PUBLIC_TRAIN_SEED,
        "evaluation_seed": args.eval_seed,
        "optimization_seed": args.seed,
        "budget": args.budget,
        "train": logs,
        "results": results,
        "evaluation_rows_sha256": rows_sha,
        "paired_contrasts_vs_semantic": paired_contrasts(rows),
    }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
