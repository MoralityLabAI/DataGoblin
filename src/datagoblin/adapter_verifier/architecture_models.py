from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured

from .drift_benchmark import (
    GAME_CONTRACTS,
    GAMES,
    _feature_vector,
    _policy_prediction,
)

Mapper = Callable[[str, str], tuple[str, float]]


def architecture_classes() -> list[str]:
    return [
        f"{game}:{field}"
        for game in GAMES
        for field in GAME_CONTRACTS[game]["safe_fields"]
    ]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=-1, keepdims=True)


def _class_indices(classes: Sequence[str], game: str) -> np.ndarray:
    return np.asarray(
        [index for index, label in enumerate(classes) if label.startswith(f"{game}:")],
        dtype=np.int64,
    )


def _vector(arm: Mapping[str, Any], game: str, key: str) -> np.ndarray:
    return _feature_vector(
        game,
        key,
        int(arm["feature_dimension"]),
        int(arm["character_ngram_min"]),
        int(arm["character_ngram_max"]),
    )


def _game_batches(
    arm: Mapping[str, Any],
    classes: Sequence[str],
    samples: Sequence[tuple[str, str, str]],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    class_index = {label: index for index, label in enumerate(classes)}
    batches: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for game in GAMES:
        selected = [sample for sample in samples if sample[0] == game]
        if not selected:
            raise ValueError(f"architecture training has no samples for {game}")
        indices = _class_indices(classes, game)
        local_lookup = {int(value): offset for offset, value in enumerate(indices)}
        features = np.stack([_vector(arm, game, key) for _, key, _ in selected])
        targets = np.asarray(
            [local_lookup[class_index[f"{game}:{target}"]] for _, _, target in selected],
            dtype=np.int64,
        )
        batches[game] = features, targets, indices
    return batches


class _CheckpointWriter:
    def __init__(self, root: Path, *, step_interval: int, second_interval: int) -> None:
        self.root = root
        self.step_interval = step_interval
        self.second_interval = second_interval
        self.last_step = 0
        self.last_time = time.monotonic()
        self.paths: list[str] = []

    def maybe_write(self, step: int, payload: Mapping[str, Any], *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and step - self.last_step < self.step_interval and now - self.last_time < self.second_interval:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"step-{step:06d}.json"
        dump_structured(path, dict(payload))
        self.paths.append(path.name)
        self.last_step = step
        self.last_time = now


def _prototype_base(arm: Mapping[str, Any], classes: Sequence[str]) -> np.ndarray:
    scale = float(arm["base_prototype_scale"])
    rows = []
    for label in classes:
        game, field = label.split(":", 1)
        rows.append(scale * _vector(arm, game, field))
    return np.stack(rows)


def train_lora_like_mapper(
    arm: Mapping[str, Any],
    samples: Sequence[tuple[str, str, str]],
    *,
    seed: int,
    checkpoint_root: Path,
    checkpoint_steps: int,
    checkpoint_seconds: int,
) -> tuple[dict[str, Any], Mapper, dict[str, Any]]:
    classes = architecture_classes()
    batches = _game_batches(arm, classes, samples)
    rank = int(arm["rank"])
    dimension = int(arm["feature_dimension"])
    rng = np.random.default_rng(seed)
    base = _prototype_base(arm, classes)
    left = rng.normal(0.0, 0.02, size=(len(classes), rank))
    right = np.zeros((rank, dimension), dtype=np.float64)
    scaling = float(arm["alpha"]) / rank
    learning_rate = float(arm["learning_rate"])
    l2 = float(arm["l2"])
    epochs = int(arm["epochs"])
    writer = _CheckpointWriter(
        checkpoint_root,
        step_interval=checkpoint_steps,
        second_interval=checkpoint_seconds,
    )
    initial_loss = 0.0
    final_loss = 0.0
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for game in GAMES:
            features, targets, indices = batches[game]
            weights = base + scaling * (left @ right)
            probabilities = _softmax(features @ weights[indices].T)
            epoch_loss += float(
                -np.log(probabilities[np.arange(len(targets)), targets] + 1e-12).mean()
            )
            probabilities[np.arange(len(targets)), targets] -= 1.0
            local_gradient = probabilities.T @ features / len(targets)
            full_gradient = np.zeros_like(base)
            full_gradient[indices] = local_gradient
            left_gradient = scaling * (full_gradient @ right.T) + l2 * left
            right_gradient = scaling * (left.T @ full_gradient) + l2 * right
            left -= learning_rate * left_gradient
            right -= learning_rate * right_gradient
            step += 1
            writer.maybe_write(
                step,
                {
                    "schema_version": "game-adapter-architecture-checkpoint.v2",
                    "architecture_family": arm["architecture_family"],
                    "seed": seed,
                    "epoch": epoch + 1,
                    "step": step,
                    "left": np.round(left, 12).tolist(),
                    "right": np.round(right, 12).tolist(),
                },
            )
        if epoch == 0:
            initial_loss = epoch_loss / len(GAMES)
        final_loss = epoch_loss / len(GAMES)
    writer.maybe_write(
        step,
        {
            "schema_version": "game-adapter-architecture-checkpoint.v2",
            "architecture_family": arm["architecture_family"],
            "seed": seed,
            "epoch": epochs,
            "step": step,
            "left": np.round(left, 12).tolist(),
            "right": np.round(right, 12).tolist(),
        },
        force=writer.last_step != step,
    )
    weights = base + scaling * (left @ right)

    def mapper(game: str, key: str) -> tuple[str, float]:
        indices = _class_indices(classes, game)
        probabilities = _softmax(weights[indices] @ _vector(arm, game, key))
        offset = int(np.argmax(probabilities))
        return classes[int(indices[offset])].split(":", 1)[1], float(probabilities[offset])

    state = {
        "schema_version": "game-adapter-low-rank-surrogate.v2",
        "classes": classes,
        "feature_dimension": dimension,
        "rank": rank,
        "alpha": float(arm["alpha"]),
        "base_prototype_scale": float(arm["base_prototype_scale"]),
        "base_frozen": True,
        "left": np.round(left, 12).tolist(),
        "right": np.round(right, 12).tolist(),
    }
    receipt = {
        "samples": len(samples),
        "epochs": epochs,
        "steps_completed": step,
        "initial_cross_entropy": initial_loss,
        "final_cross_entropy": final_loss,
        "converged": final_loss < initial_loss,
        "trainable_parameters": int(left.size + right.size),
        "frozen_parameters": int(base.size),
        "checkpoints": writer.paths,
    }
    return state, mapper, receipt


def _reservoir_features(
    features: np.ndarray,
    input_weights: np.ndarray,
    recurrent_weights: np.ndarray,
    loops: int,
) -> np.ndarray:
    state = np.zeros((features.shape[0], input_weights.shape[0]), dtype=np.float64)
    projected = features @ input_weights.T
    for _ in range(loops):
        state = np.tanh(projected + state @ recurrent_weights.T)
    return state


def train_recurrent_readout_mapper(
    arm: Mapping[str, Any],
    samples: Sequence[tuple[str, str, str]],
    *,
    seed: int,
    checkpoint_root: Path,
    checkpoint_steps: int,
    checkpoint_seconds: int,
) -> tuple[dict[str, Any], Mapper, dict[str, Any]]:
    classes = architecture_classes()
    raw_batches = _game_batches(arm, classes, samples)
    hidden = int(arm["hidden_width"])
    dimension = int(arm["feature_dimension"])
    loops = int(arm["recurrence_steps"])
    rng = np.random.default_rng(seed)
    input_weights = rng.normal(0.0, 1.0 / math.sqrt(dimension), size=(hidden, dimension))
    recurrent_weights = rng.normal(0.0, 1.0 / math.sqrt(hidden), size=(hidden, hidden))
    spectral = float(np.linalg.norm(recurrent_weights, ord=2))
    recurrent_weights *= 0.65 / spectral if spectral else 0.0
    batches = {
        game: (
            _reservoir_features(features, input_weights, recurrent_weights, loops),
            targets,
            indices,
        )
        for game, (features, targets, indices) in raw_batches.items()
    }
    output_weights = np.zeros((len(classes), hidden), dtype=np.float64)
    output_bias = np.zeros(len(classes), dtype=np.float64)
    learning_rate = float(arm["learning_rate"])
    l2 = float(arm["l2"])
    epochs = int(arm["epochs"])
    writer = _CheckpointWriter(
        checkpoint_root,
        step_interval=checkpoint_steps,
        second_interval=checkpoint_seconds,
    )
    initial_loss = 0.0
    final_loss = 0.0
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for game in GAMES:
            features, targets, indices = batches[game]
            probabilities = _softmax(
                features @ output_weights[indices].T + output_bias[indices]
            )
            epoch_loss += float(
                -np.log(probabilities[np.arange(len(targets)), targets] + 1e-12).mean()
            )
            probabilities[np.arange(len(targets)), targets] -= 1.0
            weight_gradient = probabilities.T @ features / len(targets)
            bias_gradient = np.mean(probabilities, axis=0)
            output_weights[indices] -= learning_rate * (
                weight_gradient + l2 * output_weights[indices]
            )
            output_bias[indices] -= learning_rate * bias_gradient
            step += 1
            writer.maybe_write(
                step,
                {
                    "schema_version": "game-adapter-architecture-checkpoint.v2",
                    "architecture_family": arm["architecture_family"],
                    "seed": seed,
                    "epoch": epoch + 1,
                    "step": step,
                    "output_weights": np.round(output_weights, 12).tolist(),
                    "output_bias": np.round(output_bias, 12).tolist(),
                },
            )
        if epoch == 0:
            initial_loss = epoch_loss / len(GAMES)
        final_loss = epoch_loss / len(GAMES)
    writer.maybe_write(
        step,
        {
            "schema_version": "game-adapter-architecture-checkpoint.v2",
            "architecture_family": arm["architecture_family"],
            "seed": seed,
            "epoch": epochs,
            "step": step,
            "output_weights": np.round(output_weights, 12).tolist(),
            "output_bias": np.round(output_bias, 12).tolist(),
        },
        force=writer.last_step != step,
    )

    def mapper(game: str, key: str) -> tuple[str, float]:
        indices = _class_indices(classes, game)
        raw = _vector(arm, game, key)[np.newaxis, :]
        state = _reservoir_features(raw, input_weights, recurrent_weights, loops)[0]
        probabilities = _softmax(output_weights[indices] @ state + output_bias[indices])
        offset = int(np.argmax(probabilities))
        return classes[int(indices[offset])].split(":", 1)[1], float(probabilities[offset])

    state = {
        "schema_version": "game-adapter-recurrent-surrogate.v2",
        "classes": classes,
        "feature_dimension": dimension,
        "hidden_width": hidden,
        "recurrence_steps": loops,
        "tied_recurrence": True,
        "reservoir_frozen": True,
        "input_weights": np.round(input_weights, 12).tolist(),
        "recurrent_weights": np.round(recurrent_weights, 12).tolist(),
        "output_weights": np.round(output_weights, 12).tolist(),
        "output_bias": np.round(output_bias, 12).tolist(),
    }
    receipt = {
        "samples": len(samples),
        "epochs": epochs,
        "steps_completed": step,
        "initial_cross_entropy": initial_loss,
        "final_cross_entropy": final_loss,
        "converged": final_loss < initial_loss,
        "trainable_parameters": int(output_weights.size + output_bias.size),
        "frozen_parameters": int(input_weights.size + recurrent_weights.size),
        "checkpoints": writer.paths,
    }
    return state, mapper, receipt


def calibrate_mapping_threshold(
    mapper: Mapper,
    samples: Sequence[tuple[str, str, str]],
    *,
    grid: Sequence[float],
    incorrect_penalty: float,
) -> tuple[float, dict[str, Any]]:
    scored = [(*mapper(game, key), target) for game, key, target in samples]
    rows = []
    for threshold in grid:
        correct = sum(
            predicted == target and confidence >= threshold
            for predicted, confidence, target in scored
        )
        incorrect = sum(
            predicted != target and confidence >= threshold
            for predicted, confidence, target in scored
        )
        rows.append(
            {
                "threshold": float(threshold),
                "correct": correct,
                "incorrect": incorrect,
                "utility": correct - incorrect_penalty * incorrect,
                "accepted": correct + incorrect,
            }
        )
    selected = max(
        rows,
        key=lambda row: (row["utility"], row["accepted"], -row["threshold"]),
    )
    accepted = int(selected["accepted"])
    return float(selected["threshold"]), {
        "examples": len(scored),
        "grid": rows,
        "selected_threshold": float(selected["threshold"]),
        "coverage": accepted / len(scored) if scored else 0.0,
        "accepted_accuracy": (
            int(selected["correct"]) / accepted if accepted else 0.0
        ),
        "source": "development_calibration_only",
    }


def write_architecture_predictions(
    rows: Sequence[Mapping[str, Any]],
    out_path: Path,
    *,
    setup_id: str,
    policy: Mapping[str, Any],
    mapper: Mapper | None,
    threshold: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            prediction = _policy_prediction(
                row,
                setup_id,
                policy,
                mapper=mapper,
                threshold=threshold,
            )
            handle.write(json.dumps(prediction, sort_keys=True, separators=(",", ":")) + "\n")


def finalize_model_artifact(path: Path, state: Mapping[str, Any]) -> str:
    dump_structured(path, dict(state))
    return sha256_file(path)
