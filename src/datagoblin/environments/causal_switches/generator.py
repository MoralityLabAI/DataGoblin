from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from datagoblin.hashing import sha256_json

from .solver import MechanicsState, exact_label
from .spec import CausalSwitchesSpec


@dataclass(frozen=True)
class GeneratedDataset:
    features: np.ndarray
    labels: np.ndarray
    mechanics: tuple[MechanicsState, ...]
    role_colors: tuple[int, ...]
    split_manifest: dict[str, Any]
    quality_report: dict[str, Any]


def _feature_row(color: int, state: MechanicsState, maximum_delay: int) -> list[float]:
    one_hot = [float(color == item) for item in range(3)]
    return [
        *one_hot,
        float(state.intervention_seen),
        state.elapsed / maximum_delay,
        state.delay / maximum_delay,
        1.0,
    ]


def generate_dataset(
    *,
    seed: int,
    count: int,
    mode: str,
    namespace: str,
    include_counterfactual_pairs: bool = False,
    spec: CausalSwitchesSpec | None = None,
) -> GeneratedDataset:
    """Generate deterministic rows without exposing environment-owned evaluation seeds."""
    if count < 2:
        raise ValueError("dataset count must be at least two")
    spec = spec or CausalSwitchesSpec()
    if mode not in {"biased", "balanced", "permuted_holdout"}:
        raise ValueError(f"unknown generation mode: {mode}")
    rng = np.random.default_rng(seed)
    features: list[list[float]] = []
    labels: list[int] = []
    mechanics: list[MechanicsState] = []
    colors: list[int] = []

    target = count // 2 if include_counterfactual_pairs else count
    for _index in range(target):
        intervention = bool(rng.integers(0, 2))
        delay_max = 2 if mode == "biased" else spec.maximum_delay
        delay = int(rng.integers(1, delay_max + 1))
        elapsed = int(rng.integers(0, spec.maximum_delay + 2))
        state = MechanicsState(intervention, elapsed, delay)
        label = exact_label(state)
        if mode == "biased":
            preferred = 0 if label else 1
            color = preferred if rng.random() < 0.96 else int(rng.integers(0, 3))
        elif mode == "permuted_holdout":
            preferred = 1 if label else 0
            color = preferred if rng.random() < 0.75 else int(rng.integers(0, 3))
        else:
            color = int(rng.integers(0, 3))
        features.append(_feature_row(color, state, spec.maximum_delay))
        labels.append(label)
        mechanics.append(state)
        colors.append(color)
        if include_counterfactual_pairs and len(features) < count:
            paired = MechanicsState(not intervention, elapsed, delay)
            paired_label = exact_label(paired)
            paired_color = int(rng.integers(0, 3)) if mode != "biased" else color
            features.append(_feature_row(paired_color, paired, spec.maximum_delay))
            labels.append(paired_label)
            mechanics.append(paired)
            colors.append(paired_color)

    x = np.asarray(features[:count], dtype=np.float64)
    y = np.asarray(labels[:count], dtype=np.float64)
    mechanics_tuple = tuple(mechanics[:count])
    colors_tuple = tuple(colors[:count])
    public_spec = {
        "schema_version": "1.0",
        "environment_id": spec.environment_id,
        "namespace": namespace,
        "mode": mode,
        "row_count": len(y),
        "feature_names": list(spec.feature_names),
        "seed_commitment": sha256_json({"namespace": namespace, "seed": seed}),
        "labels_from": "causal_switches_exact_solver",
    }
    positives = float(np.mean(y))
    color_label_rates = {
        str(color): float(np.mean(y[np.asarray(colors_tuple) == color]))
        if np.any(np.asarray(colors_tuple) == color)
        else 0.0
        for color in range(3)
    }
    quality = {
        "row_count": len(y),
        "positive_fraction": positives,
        "unique_feature_rows": len(np.unique(x, axis=0)),
        "unique_fraction": float(len(np.unique(x, axis=0)) / len(x)),
        "color_label_rates": color_label_rates,
        "counterfactual_pairs_requested": include_counterfactual_pairs,
        "all_labels_solver_verified": all(
            exact_label(state) == int(label) for state, label in zip(mechanics_tuple, y, strict=True)
        ),
    }
    return GeneratedDataset(x, y, mechanics_tuple, colors_tuple, public_spec, quality)


def counterfactual_pairs(dataset: GeneratedDataset) -> tuple[tuple[int, int], ...]:
    pairs = []
    for index in range(0, len(dataset.mechanics) - 1, 2):
        left = dataset.mechanics[index]
        right = dataset.mechanics[index + 1]
        if (
            left.elapsed == right.elapsed
            and left.delay == right.delay
            and left.intervention_seen != right.intervention_seen
        ):
            pairs.append((index, index + 1))
    return tuple(pairs)
