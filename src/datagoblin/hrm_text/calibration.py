from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from datagoblin.hashing import sha256_json


@dataclass(frozen=True)
class LogisticCalibrationModel:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    fit_split: str
    fit_example_hashes: tuple[str, ...]
    method: str = "logistic"

    def predict(self, row: Mapping[str, Any]) -> float:
        values = np.asarray([float(row[name]) for name in self.feature_names])
        standardized = (values - np.asarray(self.means)) / np.asarray(self.scales)
        logit = float(standardized @ np.asarray(self.coefficients) + self.intercept)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0))))

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "fit_split": self.fit_split,
            "fit_example_hashes": list(self.fit_example_hashes),
        }


@dataclass(frozen=True)
class IsotonicCalibrationModel:
    score_feature: str
    thresholds: tuple[float, ...]
    probabilities: tuple[float, ...]
    fit_split: str
    fit_example_hashes: tuple[str, ...]
    method: str = "isotonic"

    def predict(self, row: Mapping[str, Any]) -> float:
        score = float(row[self.score_feature])
        index = int(np.searchsorted(np.asarray(self.thresholds), score, side="right") - 1)
        return float(self.probabilities[min(max(index, 0), len(self.probabilities) - 1)])

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "score_feature": self.score_feature,
            "thresholds": list(self.thresholds),
            "probabilities": list(self.probabilities),
            "fit_split": self.fit_split,
            "fit_example_hashes": list(self.fit_example_hashes),
        }


CalibrationModel = LogisticCalibrationModel | IsotonicCalibrationModel


def _validate_calibration_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    if not rows:
        raise ValueError("calibration split is empty")
    if any(row.get("split") != "calibration" for row in rows):
        raise ValueError("calibration models may be fit only on the calibration split")
    if any("error_label" not in row for row in rows):
        raise ValueError("calibration rows require error_label")
    return tuple(sha256_json({"example_id": row["example_id"]}) for row in rows)


def fit_logistic_calibration(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    *,
    steps: int = 1200,
    learning_rate: float = 0.08,
    l2: float = 0.001,
) -> LogisticCalibrationModel:
    hashes = _validate_calibration_rows(rows)
    names = tuple(feature_names)
    if not names:
        raise ValueError("logistic calibration requires at least one feature")
    features = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)
    labels = np.asarray([float(row["error_label"]) for row in rows], dtype=np.float64)
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales[scales < 1e-12] = 1.0
    standardized = (features - means) / scales
    coefficients = np.zeros(len(names), dtype=np.float64)
    intercept = 0.0
    for _ in range(steps):
        logits = np.clip(standardized @ coefficients + intercept, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = probabilities - labels
        coefficients -= learning_rate * (standardized.T @ error / len(rows) + l2 * coefficients)
        intercept -= learning_rate * float(np.mean(error))
    return LogisticCalibrationModel(
        names,
        tuple(float(item) for item in means),
        tuple(float(item) for item in scales),
        tuple(float(item) for item in coefficients),
        float(intercept),
        "calibration",
        hashes,
    )


def fit_isotonic_calibration(
    rows: Sequence[Mapping[str, Any]], score_feature: str
) -> IsotonicCalibrationModel:
    hashes = _validate_calibration_rows(rows)
    ordered = sorted((float(row[score_feature]), float(row["error_label"])) for row in rows)
    blocks: list[list[float]] = []
    for score, label in ordered:
        blocks.append([score, score, label, 1.0])
        while len(blocks) >= 2 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    thresholds = tuple(block[0] for block in blocks)
    probabilities = tuple(block[2] / block[3] for block in blocks)
    return IsotonicCalibrationModel(score_feature, thresholds, probabilities, "calibration", hashes)


def calibration_model_from_jsonable(value: Mapping[str, Any]) -> CalibrationModel:
    if value["method"] == "logistic":
        return LogisticCalibrationModel(
            feature_names=tuple(value["feature_names"]),
            means=tuple(float(item) for item in value["means"]),
            scales=tuple(float(item) for item in value["scales"]),
            coefficients=tuple(float(item) for item in value["coefficients"]),
            intercept=float(value["intercept"]),
            fit_split=str(value["fit_split"]),
            fit_example_hashes=tuple(value["fit_example_hashes"]),
        )
    if value["method"] == "isotonic":
        return IsotonicCalibrationModel(
            score_feature=str(value["score_feature"]),
            thresholds=tuple(float(item) for item in value["thresholds"]),
            probabilities=tuple(float(item) for item in value["probabilities"]),
            fit_split=str(value["fit_split"]),
            fit_example_hashes=tuple(value["fit_example_hashes"]),
        )
    raise ValueError(f"unknown calibration method: {value['method']}")
