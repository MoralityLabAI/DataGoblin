from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from array import array
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PREFERENCES = ("LEFT", "RIGHT", "TRADEOFF")
RATIONALE_CODES = (
    "safety_gate",
    "pareto_dominance",
    "parameter_efficiency",
    "irreducible_tradeoff",
)
METRIC_FIELDS = (
    "ordinary_exact_record_accuracy",
    "ordinary_mapping_f1",
    "ood_exact_record_accuracy",
    "ood_mapping_f1",
)
FEATURE_WHITELIST = (
    "game",
    "fixture_pressure",
    "context_mode",
    "optimization_pressure",
    "trainable_parameter_bucket",
    "safety_qualified",
    *METRIC_FIELDS,
)
BUCKET_RANK = {"zero": 0.0, "small": 1.0, "medium": 2.0, "large": 3.0}
INVERT_PREFERENCE = {"LEFT": "RIGHT", "RIGHT": "LEFT", "TRADEOFF": "TRADEOFF"}

CheckpointCallback = Callable[[int, Mapping[str, Any]], None]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _validate_side(side: Mapping[str, Any]) -> None:
    if set(side) != set(FEATURE_WHITELIST):
        raise ValueError("pair side does not match the frozen public feature whitelist")
    if str(side["trainable_parameter_bucket"]) not in BUCKET_RANK:
        raise ValueError("unknown trainable parameter bucket")
    if not isinstance(side["safety_qualified"], bool):
        raise ValueError("safety_qualified must be boolean")
    for field in METRIC_FIELDS:
        value = float(side[field])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be finite and within [0, 1]")


def validate_pair_row(row: Mapping[str, Any], *, targets_required: bool) -> None:
    expected = {
        "schema_version",
        "example_id",
        "meta_split",
        "lineage_group_sha256",
        "source_tournament_report_sha256",
        "source_kind",
        "input",
    }
    if targets_required:
        expected.update({"target", "authority"})
    if set(row) != expected:
        raise ValueError("pair row contains missing or non-contract fields")
    if row["schema_version"] != "curation-taste-pair.v1":
        raise ValueError("unsupported pair schema")
    expected_split = "train" if targets_required else "heldout"
    if row["meta_split"] != expected_split:
        raise ValueError(f"expected {expected_split} meta split")
    input_ = row["input"]
    if not isinstance(input_, Mapping) or set(input_) != {"left", "right"}:
        raise ValueError("pair input must contain exactly left and right")
    _validate_side(input_["left"])
    _validate_side(input_["right"])
    if targets_required:
        target = row["target"]
        if set(target) != {"preference", "rationale_code"}:
            raise ValueError("training target has an invalid shape")
        if target["preference"] not in PREFERENCES:
            raise ValueError("unknown preference")
        if target["rationale_code"] not in RATIONALE_CODES:
            raise ValueError("unknown rationale code")
        if row["authority"] != "ENVIRONMENT_OWNED":
            raise ValueError("training target lacks environment authority")


def load_public_dataset(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != "curation-taste-ldt-foundation.v1":
        raise ValueError("unsupported LDT foundation manifest")
    if tuple(manifest["feature_whitelist"]) != FEATURE_WHITELIST:
        raise ValueError("dataset feature whitelist differs from the frozen array contract")
    if int(manifest["lineage_audit"]["cross_split_groups"]) != 0:
        raise ValueError("cross-split LDT meta-lineages are forbidden")
    train_spec = manifest["splits"]["train"]
    heldout_spec = manifest["splits"]["heldout"]
    if train_spec["targets_in_file"] is not True or heldout_spec["targets_in_file"] is not False:
        raise ValueError("dataset target custody is invalid")
    train_path = manifest_path.parent / str(train_spec["path"])
    heldout_path = manifest_path.parent / str(heldout_spec["path"])
    for path, spec in ((train_path, train_spec), (heldout_path, heldout_spec)):
        if sha256_file(path) != spec["sha256"]:
            raise ValueError(f"split hash mismatch: {path.name}")
    train_rows = read_jsonl(train_path)
    heldout_rows = read_jsonl(heldout_path)
    if len(train_rows) != int(train_spec["count"]) or len(heldout_rows) != int(heldout_spec["count"]):
        raise ValueError("split count mismatch")
    for row in train_rows:
        validate_pair_row(row, targets_required=True)
    for row in heldout_rows:
        validate_pair_row(row, targets_required=False)
    train_lineages = {str(row["lineage_group_sha256"]) for row in train_rows}
    heldout_lineages = {str(row["lineage_group_sha256"]) for row in heldout_rows}
    if train_lineages & heldout_lineages:
        raise ValueError("train and heldout LDT meta-lineages overlap")
    return manifest, train_rows, heldout_rows


def taste_policy(
    input_: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float = 0.01,
) -> tuple[str, str]:
    left = input_["left"]
    right = input_["right"]
    left_safe = bool(left["safety_qualified"])
    right_safe = bool(right["safety_qualified"])
    if left_safe != right_safe:
        return ("LEFT" if left_safe else "RIGHT"), "safety_gate"
    left_metrics = tuple(float(left[name]) for name in METRIC_FIELDS)
    right_metrics = tuple(float(right[name]) for name in METRIC_FIELDS)
    left_dominates = all(
        left_value >= right_value - tolerance
        for left_value, right_value in zip(left_metrics, right_metrics, strict=True)
    ) and any(
        left_value > right_value + tolerance
        for left_value, right_value in zip(left_metrics, right_metrics, strict=True)
    )
    right_dominates = all(
        right_value >= left_value - tolerance
        for left_value, right_value in zip(left_metrics, right_metrics, strict=True)
    ) and any(
        right_value > left_value + tolerance
        for left_value, right_value in zip(left_metrics, right_metrics, strict=True)
    )
    if left_dominates != right_dominates:
        return ("LEFT" if left_dominates else "RIGHT"), "pareto_dominance"
    equivalent = all(
        abs(left_value - right_value) <= tolerance
        for left_value, right_value in zip(left_metrics, right_metrics, strict=True)
    )
    left_cost = BUCKET_RANK[str(left["trainable_parameter_bucket"])]
    right_cost = BUCKET_RANK[str(right["trainable_parameter_bucket"])]
    if equivalent and left_cost != right_cost:
        return ("LEFT" if left_cost < right_cost else "RIGHT"), "parameter_efficiency"
    return "TRADEOFF", "irreducible_tradeoff"


def swap_pair(row: Mapping[str, Any]) -> dict[str, Any]:
    swapped = copy.deepcopy(dict(row))
    swapped["input"] = {
        "left": copy.deepcopy(row["input"]["right"]),
        "right": copy.deepcopy(row["input"]["left"]),
    }
    if "target" in swapped:
        swapped["target"]["preference"] = INVERT_PREFERENCE[str(row["target"]["preference"])]
    return swapped


@dataclass(frozen=True)
class PairEncoder:
    games: tuple[str, ...]
    fixture_pressures: tuple[str, ...]
    context_modes: tuple[str, ...]
    optimization_pressures: tuple[str, ...]
    tolerance: float = 0.01

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, Any]], *, tolerance: float = 0.01) -> PairEncoder:
        sides = [row["input"][side] for row in rows for side in ("left", "right")]
        return cls(
            games=tuple(sorted({str(side["game"]) for side in sides})),
            fixture_pressures=tuple(sorted({str(side["fixture_pressure"]) for side in sides})),
            context_modes=tuple(sorted({str(side["context_mode"]) for side in sides})),
            optimization_pressures=tuple(
                sorted({str(side["optimization_pressure"]) for side in sides})
            ),
            tolerance=tolerance,
        )

    def feature_names(self, view: str) -> tuple[str, ...]:
        names = ["bias"]
        for field in METRIC_FIELDS:
            names.extend((f"{field}:delta", f"{field}:absolute_delta"))
        names.extend(("safety:delta", "safety:mismatch", "parameter_bucket:delta", "parameter_bucket:absolute_delta"))
        names.extend(f"game={value}" for value in self.games)
        names.extend(f"fixture_pressure={value}" for value in self.fixture_pressures)
        names.extend(f"context_mode={value}:signed" for value in self.context_modes)
        names.extend(
            f"optimization_pressure={value}:signed" for value in self.optimization_pressures
        )
        if view == "typed":
            names.extend(
                (
                    "lattice:safety_direction",
                    "lattice:left_pareto_dominates",
                    "lattice:right_pareto_dominates",
                    "lattice:metrics_equivalent",
                    "lattice:parameter_efficiency_direction",
                    "lattice:mixed_tradeoff",
                )
            )
        elif view != "raw":
            raise ValueError(f"unknown feature view: {view}")
        return tuple(names)

    def transform(self, input_: Mapping[str, Mapping[str, Any]], *, view: str) -> array:
        left = input_["left"]
        right = input_["right"]
        values = [1.0]
        metric_deltas = []
        for field in METRIC_FIELDS:
            delta = float(left[field]) - float(right[field])
            metric_deltas.append(delta)
            values.extend((delta, abs(delta)))
        safety_delta = float(bool(left["safety_qualified"])) - float(bool(right["safety_qualified"]))
        values.extend((safety_delta, float(abs(safety_delta) > 0.0)))
        parameter_delta = (
            BUCKET_RANK[str(left["trainable_parameter_bucket"])]
            - BUCKET_RANK[str(right["trainable_parameter_bucket"])]
        )
        values.extend((parameter_delta / 3.0, abs(parameter_delta) / 3.0))
        game = str(left["game"])
        pressure = str(left["fixture_pressure"])
        values.extend(float(game == value) for value in self.games)
        values.extend(float(pressure == value) for value in self.fixture_pressures)
        for category in self.context_modes:
            values.append(
                float(str(left["context_mode"]) == category)
                - float(str(right["context_mode"]) == category)
            )
        for category in self.optimization_pressures:
            values.append(
                float(str(left["optimization_pressure"]) == category)
                - float(str(right["optimization_pressure"]) == category)
            )
        if view == "typed":
            left_metrics = [float(left[field]) for field in METRIC_FIELDS]
            right_metrics = [float(right[field]) for field in METRIC_FIELDS]
            left_dominates = all(
                left_value >= right_value - self.tolerance
                for left_value, right_value in zip(left_metrics, right_metrics, strict=True)
            ) and any(
                left_value > right_value + self.tolerance
                for left_value, right_value in zip(left_metrics, right_metrics, strict=True)
            )
            right_dominates = all(
                right_value >= left_value - self.tolerance
                for left_value, right_value in zip(left_metrics, right_metrics, strict=True)
            ) and any(
                right_value > left_value + self.tolerance
                for left_value, right_value in zip(left_metrics, right_metrics, strict=True)
            )
            equivalent = all(abs(delta) <= self.tolerance for delta in metric_deltas)
            efficiency_direction = 0.0
            if equivalent and parameter_delta != 0:
                efficiency_direction = -1.0 if parameter_delta < 0 else 1.0
            mixed_tradeoff = not left_dominates and not right_dominates and not (
                equivalent and parameter_delta != 0
            )
            values.extend(
                (
                    safety_delta,
                    float(left_dominates),
                    float(right_dominates),
                    float(equivalent),
                    efficiency_direction,
                    float(mixed_tradeoff),
                )
            )
        elif view != "raw":
            raise ValueError(f"unknown feature view: {view}")
        result = array("d", values)
        if len(result) != len(self.feature_names(view)):
            raise AssertionError("feature-name and array lengths differ")
        return result

    def receipt(self) -> dict[str, Any]:
        return {
            "games": list(self.games),
            "fixture_pressures": list(self.fixture_pressures),
            "context_modes": list(self.context_modes),
            "optimization_pressures": list(self.optimization_pressures),
            "tolerance": self.tolerance,
            "fit_split": "train_only",
        }


def _softmax(logits: Sequence[float]) -> array:
    peak = max(logits)
    values = [math.exp(max(-40.0, min(40.0, value - peak))) for value in logits]
    total = sum(values)
    return array("d", (value / total for value in values))


def _gradient_clip_scale(gradients: Sequence[np.ndarray], limit: float) -> float:
    norm = math.sqrt(sum(float(np.sum(gradient * gradient)) for gradient in gradients))
    return min(1.0, limit / norm) if norm > 0.0 else 1.0


def _batch_arrays(
    examples: Sequence[tuple[array, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray([list(example[0]) for example in examples], dtype=np.float64)
    preference = np.asarray([example[1] for example in examples], dtype=np.int64)
    rationale = np.asarray([example[2] for example in examples], dtype=np.int64)
    return features, preference, rationale


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = np.clip(logits - np.max(logits, axis=1, keepdims=True), -40.0, 40.0)
    values = np.exp(shifted)
    return values / np.sum(values, axis=1, keepdims=True)


@dataclass
class LinearJointArray:
    input_width: int
    seed: int

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / math.sqrt(max(1, self.input_width))
        self.preference_weights = rng.uniform(
            -scale, scale, size=(len(PREFERENCES), self.input_width)
        )
        self.rationale_weights = rng.uniform(
            -scale, scale, size=(len(RATIONALE_CODES), self.input_width)
        )

    @property
    def parameter_count(self) -> int:
        return (len(PREFERENCES) + len(RATIONALE_CODES)) * self.input_width

    def predict_proba(self, features: Sequence[float]) -> tuple[array, array]:
        vector = np.asarray(features, dtype=np.float64)
        return (
            array("d", _softmax(self.preference_weights @ vector)),
            array("d", _softmax(self.rationale_weights @ vector)),
        )

    def train_batch(
        self,
        examples: Sequence[tuple[array, int, int]],
        *,
        learning_rate: float,
        l2: float,
        gradient_clip: float,
        preference_class_weights: Sequence[float],
        rationale_class_weights: Sequence[float],
        rationale_loss_weight: float,
    ) -> None:
        features, preference_index, rationale_index = _batch_arrays(examples)
        count = len(examples)
        preference_probabilities = _softmax_rows(features @ self.preference_weights.T)
        rationale_probabilities = _softmax_rows(features @ self.rationale_weights.T)
        preference_probabilities[np.arange(count), preference_index] -= 1.0
        rationale_probabilities[np.arange(count), rationale_index] -= 1.0
        preference_probabilities *= np.asarray(preference_class_weights)[
            preference_index, np.newaxis
        ]
        rationale_probabilities *= (
            np.asarray(rationale_class_weights)[rationale_index, np.newaxis]
            * rationale_loss_weight
        )
        preference_gradient = (
            preference_probabilities.T @ features / count + l2 * self.preference_weights
        )
        rationale_gradient = (
            rationale_probabilities.T @ features / count + l2 * self.rationale_weights
        )
        clip = _gradient_clip_scale(
            (preference_gradient, rationale_gradient), gradient_clip
        )
        self.preference_weights -= learning_rate * clip * preference_gradient
        self.rationale_weights -= learning_rate * clip * rationale_gradient

    def state_dict(self) -> dict[str, Any]:
        return {
            "architecture": "joint_linear_array",
            "input_width": self.input_width,
            "preference_weights": self.preference_weights.tolist(),
            "rationale_weights": self.rationale_weights.tolist(),
        }

    def loop_delta(self, _features: Sequence[float]) -> float:
        return 0.0


@dataclass
class LoopedJointArray:
    input_width: int
    hidden_width: int
    loop_count: int
    seed: int

    def __post_init__(self) -> None:
        if self.hidden_width < 1 or self.loop_count < 1:
            raise ValueError("latent array width and loop count must be positive")
        rng = np.random.default_rng(self.seed)
        input_scale = 1.0 / math.sqrt(max(1, self.input_width))
        hidden_scale = 1.0 / math.sqrt(self.hidden_width)
        self.input_weights = rng.uniform(
            -input_scale, input_scale, size=(self.hidden_width, self.input_width)
        )
        self.recurrent_weights = (
            rng.uniform(
                -hidden_scale,
                hidden_scale,
                size=(self.hidden_width, self.hidden_width),
            )
            if self.loop_count > 1
            else None
        )
        self.hidden_bias = np.zeros(self.hidden_width, dtype=np.float64)
        self.preference_weights = rng.uniform(
            -hidden_scale,
            hidden_scale,
            size=(len(PREFERENCES), self.hidden_width),
        )
        self.preference_bias = np.zeros(len(PREFERENCES), dtype=np.float64)
        self.rationale_weights = rng.uniform(
            -hidden_scale,
            hidden_scale,
            size=(len(RATIONALE_CODES), self.hidden_width),
        )
        self.rationale_bias = np.zeros(len(RATIONALE_CODES), dtype=np.float64)

    @property
    def parameter_count(self) -> int:
        recurrent = self.hidden_width**2 if self.loop_count > 1 else 0
        return (
            self.hidden_width * self.input_width
            + recurrent
            + self.hidden_width
            + len(PREFERENCES) * self.hidden_width
            + len(PREFERENCES)
            + len(RATIONALE_CODES) * self.hidden_width
            + len(RATIONALE_CODES)
        )

    def _forward(self, features: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        if features.ndim == 1:
            features = features[np.newaxis, :]
        states = [np.zeros((features.shape[0], self.hidden_width), dtype=np.float64)]
        for _ in range(self.loop_count):
            pre_activation = features @ self.input_weights.T + self.hidden_bias
            if self.recurrent_weights is not None:
                pre_activation += states[-1] @ self.recurrent_weights.T
            states.append(np.tanh(np.clip(pre_activation, -20.0, 20.0)))
        final = states[-1]
        preference_logits = final @ self.preference_weights.T + self.preference_bias
        rationale_logits = final @ self.rationale_weights.T + self.rationale_bias
        return states, _softmax_rows(preference_logits), _softmax_rows(rationale_logits)

    def predict_proba(self, features: Sequence[float]) -> tuple[array, array]:
        _, preference, rationale = self._forward(np.asarray(features, dtype=np.float64))
        return array("d", preference[0]), array("d", rationale[0])

    def train_batch(
        self,
        examples: Sequence[tuple[array, int, int]],
        *,
        learning_rate: float,
        l2: float,
        gradient_clip: float,
        preference_class_weights: Sequence[float],
        rationale_class_weights: Sequence[float],
        rationale_loss_weight: float,
    ) -> None:
        features, preference_index, rationale_index = _batch_arrays(examples)
        count = len(examples)
        states, preference_probabilities, rationale_probabilities = self._forward(features)
        preference_probabilities[np.arange(count), preference_index] -= 1.0
        rationale_probabilities[np.arange(count), rationale_index] -= 1.0
        preference_probabilities *= np.asarray(preference_class_weights)[
            preference_index, np.newaxis
        ]
        rationale_probabilities *= (
            np.asarray(rationale_class_weights)[rationale_index, np.newaxis]
            * rationale_loss_weight
        )
        final = states[-1]
        preference_gradient = (
            preference_probabilities.T @ final / count + l2 * self.preference_weights
        )
        preference_bias_gradient = np.mean(preference_probabilities, axis=0)
        rationale_gradient = (
            rationale_probabilities.T @ final / count + l2 * self.rationale_weights
        )
        rationale_bias_gradient = np.mean(rationale_probabilities, axis=0)
        hidden_delta = (
            preference_probabilities @ self.preference_weights
            + rationale_probabilities @ self.rationale_weights
        )
        input_gradient = np.zeros_like(self.input_weights)
        recurrent_gradient = (
            np.zeros_like(self.recurrent_weights)
            if self.recurrent_weights is not None
            else None
        )
        hidden_bias_gradient = np.zeros_like(self.hidden_bias)
        for loop_index in range(self.loop_count - 1, -1, -1):
            current = states[loop_index + 1]
            previous = states[loop_index]
            pre_activation_delta = hidden_delta * (1.0 - current * current)
            input_gradient += pre_activation_delta.T @ features / count
            hidden_bias_gradient += np.mean(pre_activation_delta, axis=0)
            if self.recurrent_weights is not None and recurrent_gradient is not None:
                recurrent_gradient += pre_activation_delta.T @ previous / count
                hidden_delta = pre_activation_delta @ self.recurrent_weights
        input_gradient += l2 * self.input_weights
        gradients = [
            input_gradient,
            hidden_bias_gradient,
            preference_gradient,
            preference_bias_gradient,
            rationale_gradient,
            rationale_bias_gradient,
        ]
        if self.recurrent_weights is not None and recurrent_gradient is not None:
            recurrent_gradient += l2 * self.recurrent_weights
            gradients.append(recurrent_gradient)
        clip = _gradient_clip_scale(gradients, gradient_clip)
        self.input_weights -= learning_rate * clip * input_gradient
        self.hidden_bias -= learning_rate * clip * hidden_bias_gradient
        self.preference_weights -= learning_rate * clip * preference_gradient
        self.preference_bias -= learning_rate * clip * preference_bias_gradient
        self.rationale_weights -= learning_rate * clip * rationale_gradient
        self.rationale_bias -= learning_rate * clip * rationale_bias_gradient
        if self.recurrent_weights is not None and recurrent_gradient is not None:
            self.recurrent_weights -= learning_rate * clip * recurrent_gradient

    def state_dict(self) -> dict[str, Any]:
        return {
            "architecture": (
                "latent_array" if self.loop_count == 1 else "weight_tied_looped_array"
            ),
            "input_width": self.input_width,
            "hidden_width": self.hidden_width,
            "loop_count": self.loop_count,
            "input_weights": self.input_weights.tolist(),
            "recurrent_weights": (
                self.recurrent_weights.tolist()
                if self.recurrent_weights is not None
                else []
            ),
            "hidden_bias": self.hidden_bias.tolist(),
            "preference_weights": self.preference_weights.tolist(),
            "preference_bias": self.preference_bias.tolist(),
            "rationale_weights": self.rationale_weights.tolist(),
            "rationale_bias": self.rationale_bias.tolist(),
        }

    def loop_delta(self, features: Sequence[float]) -> float:
        states, _, _ = self._forward(np.asarray(features, dtype=np.float64))
        if len(states) < 3:
            return 0.0
        return float(np.linalg.norm(states[-1][0] - states[-2][0]))


def make_model(spec: Mapping[str, Any], input_width: int, seed: int) -> LinearJointArray | LoopedJointArray:
    family = str(spec["family"])
    if family == "joint_linear_array":
        return LinearJointArray(input_width=input_width, seed=seed)
    if family in {"latent_array", "weight_tied_looped_array"}:
        return LoopedJointArray(
            input_width=input_width,
            hidden_width=int(spec["hidden_width"]),
            loop_count=int(spec["loop_count"]),
            seed=seed,
        )
    raise ValueError(f"unsupported model family: {family}")


def encode_examples(
    rows: Sequence[Mapping[str, Any]],
    encoder: PairEncoder,
    *,
    view: str,
    augment_swaps: bool,
) -> list[tuple[array, int, int]]:
    selected: list[Mapping[str, Any]] = []
    for row in rows:
        selected.append(row)
        if augment_swaps:
            selected.append(swap_pair(row))
    return [
        (
            encoder.transform(row["input"], view=view),
            PREFERENCES.index(str(row["target"]["preference"])),
            RATIONALE_CODES.index(str(row["target"]["rationale_code"])),
        )
        for row in selected
    ]


def inverse_frequency_weights(
    examples: Sequence[tuple[array, int, int]],
    target_position: int,
    class_count: int,
) -> array:
    counts = Counter(example[target_position] for example in examples)
    total = len(examples)
    return array(
        "d",
        (
            total / (class_count * counts[index]) if counts[index] else 1.0
            for index in range(class_count)
        ),
    )


def train_array_model(
    model: LinearJointArray | LoopedJointArray,
    examples: Sequence[tuple[array, int, int]],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    l2: float,
    gradient_clip: float,
    rationale_loss_weight: float,
    seed: int,
    checkpoint_every_epochs: int,
    checkpoint_every_seconds: float,
    checkpoint_callback: CheckpointCallback | None = None,
) -> int:
    if not examples or epochs < 1 or not 1 <= batch_size <= 32:
        raise ValueError("invalid bounded array training request")
    preference_weights = inverse_frequency_weights(examples, 1, len(PREFERENCES))
    rationale_weights = inverse_frequency_weights(examples, 2, len(RATIONALE_CODES))
    order = list(range(len(examples)))
    rng = random.Random(seed)
    steps = 0
    last_checkpoint = time.monotonic()
    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        for offset in range(0, len(order), batch_size):
            batch = [examples[index] for index in order[offset : offset + batch_size]]
            model.train_batch(
                batch,
                learning_rate=learning_rate,
                l2=l2,
                gradient_clip=gradient_clip,
                preference_class_weights=preference_weights,
                rationale_class_weights=rationale_weights,
                rationale_loss_weight=rationale_loss_weight,
            )
            steps += 1
        checkpoint_due = (
            epoch % checkpoint_every_epochs == 0
            or time.monotonic() - last_checkpoint >= checkpoint_every_seconds
            or epoch == epochs
        )
        if checkpoint_callback is not None and checkpoint_due:
            checkpoint_callback(epoch, model.state_dict())
            last_checkpoint = time.monotonic()
    return steps


def predict_model(
    model: LinearJointArray | LoopedJointArray,
    encoder: PairEncoder,
    row: Mapping[str, Any],
    *,
    view: str,
) -> dict[str, Any]:
    features = encoder.transform(row["input"], view=view)
    preference_probabilities, rationale_probabilities = model.predict_proba(features)
    preference_index = max(range(len(PREFERENCES)), key=lambda index: preference_probabilities[index])
    rationale_index = max(
        range(len(RATIONALE_CODES)), key=lambda index: rationale_probabilities[index]
    )
    swapped_features = encoder.transform(swap_pair(row)["input"], view=view)
    swapped_probabilities, _ = model.predict_proba(swapped_features)
    swapped_index = max(
        range(len(PREFERENCES)), key=lambda index: swapped_probabilities[index]
    )
    return {
        "preference": PREFERENCES[preference_index],
        "rationale_code": RATIONALE_CODES[rationale_index],
        "confidence": float(preference_probabilities[preference_index]),
        "preference_probabilities": {
            label: float(preference_probabilities[index])
            for index, label in enumerate(PREFERENCES)
        },
        "swapped_preference": PREFERENCES[swapped_index],
        "swap_consistent": (
            PREFERENCES[swapped_index] == INVERT_PREFERENCE[PREFERENCES[preference_index]]
        ),
        "loop_delta": model.loop_delta(features),
    }


def control_prediction(
    row: Mapping[str, Any],
    *,
    control_id: str,
    majority_preference: str,
    majority_rationale: str,
    preference_frequencies: Mapping[str, float],
) -> dict[str, Any]:
    if control_id == "majority-control":
        preference = majority_preference
        rationale = majority_rationale
        probabilities = {label: float(preference_frequencies.get(label, 0.0)) for label in PREFERENCES}
        swapped_preference = majority_preference
    elif control_id == "frozen-taste-policy":
        preference, rationale = taste_policy(row["input"])
        swapped_preference, _ = taste_policy(swap_pair(row)["input"])
        probabilities = {label: float(label == preference) for label in PREFERENCES}
    else:
        raise ValueError(f"unknown control: {control_id}")
    return {
        "preference": preference,
        "rationale_code": rationale,
        "confidence": probabilities[preference],
        "preference_probabilities": probabilities,
        "swapped_preference": swapped_preference,
        "swap_consistent": swapped_preference == INVERT_PREFERENCE[preference],
        "loop_delta": 0.0,
    }


def classification_metrics(
    actual: Sequence[str],
    predicted: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("metric inputs must be non-empty and aligned")
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    recalls = []
    for label in labels:
        true_positive = sum(a == label and p == label for a, p in zip(actual, predicted, strict=True))
        false_positive = sum(a != label and p == label for a, p in zip(actual, predicted, strict=True))
        false_negative = sum(a == label and p != label for a, p in zip(actual, predicted, strict=True))
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": sum(value == label for value in actual),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)
        recalls.append(recall)
    return {
        "accuracy": sum(a == p for a, p in zip(actual, predicted, strict=True)) / len(actual),
        "macro_f1": sum(f1_values) / len(f1_values),
        "macro_recall": sum(recalls) / len(recalls),
        "per_class": per_class,
    }


def safety_qualification(
    observed_safety_accuracy: float | None,
    safety_swap_consistency: float | None,
) -> bool:
    """Require both observed and counterfactual orientation safety evidence."""
    return observed_safety_accuracy == 1.0 and safety_swap_consistency == 1.0
