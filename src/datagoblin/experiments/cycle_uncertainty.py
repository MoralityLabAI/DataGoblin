from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.hrm_text.calibration import fit_isotonic_calibration, fit_logistic_calibration
from datagoblin.serialization import dump_structured

RECURRENT_FEATURES = (
    "relative_h1_h2_update_norm",
    "mean_final_l_update_norm",
    "maximum_token_level_late_update",
    "exponential_update_decay_estimate",
    "h1_h2_js_logit_divergence",
    "residual_norm_growth",
)
BASELINE_FEATURES = (
    "final_verdict_entropy",
    "maximum_verdict_probability",
    "verdict_token_margin",
    "sequence_log_probability",
    "prompt_length",
    "output_length",
    "schema_complexity",
)
MANDATORY_NON_CLAIMS = (
    "Convergence does not imply correctness.",
    "This experiment does not establish alignment.",
    "Recurrent signals may merely track prompt complexity.",
    "HRM-specific value requires comparison with recurrent baselines.",
    "A null result is valid.",
)


def _finite(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def trace_to_analysis_row(trace: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "example_id": trace["example_id"],
        "split": trace["split"],
        "error_label": 0.0 if trace.get("correctness_label") else 1.0,
    }
    row.update({name: _finite(trace["recurrent_features"][name]) for name in RECURRENT_FEATURES})
    row.update({name: _finite(trace[name]) for name in BASELINE_FEATURES})
    return row


def _auroc(labels: Sequence[float], scores: Sequence[float]) -> float | None:
    positives = [index for index, label in enumerate(labels) if label == 1]
    negatives = [index for index, label in enumerate(labels) if label == 0]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if scores[positive] > scores[negative] else 0.5 if scores[positive] == scores[negative] else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) < 2:
        return None

    def ranks(values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        order = np.argsort(array, kind="mergesort")
        result = np.empty(len(array), dtype=np.float64)
        start = 0
        while start < len(array):
            end = start + 1
            while end < len(array) and array[order[end]] == array[order[start]]:
                end += 1
            result[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return result

    first_ranks, second_ranks = ranks(first), ranks(second)
    if np.std(first_ranks) < 1e-12 or np.std(second_ranks) < 1e-12:
        return None
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def _ece(labels: Sequence[float], probabilities: Sequence[float], bins: int = 10) -> float:
    labels_array = np.asarray(labels)
    probability_array = np.asarray(probabilities)
    total = len(labels_array)
    value = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = (probability_array >= lower) & (
            probability_array <= upper if index == bins - 1 else probability_array < upper
        )
        if np.any(selected):
            value += float(np.sum(selected)) / total * abs(
                float(np.mean(probability_array[selected])) - float(np.mean(labels_array[selected]))
            )
    return value


def _risk_coverage(
    labels: Sequence[float], error_probabilities: Sequence[float]
) -> tuple[list[dict[str, float]], float]:
    order = np.argsort(np.asarray(error_probabilities), kind="mergesort")
    ordered_labels = np.asarray(labels)[order]
    points = [
        {"coverage": count / len(labels), "risk": float(np.mean(ordered_labels[:count]))}
        for count in range(1, len(labels) + 1)
    ]
    risks = [0.0] + [item["risk"] for item in points]
    coverages = [0.0] + [item["coverage"] for item in points]
    area = sum(
        (coverages[index] - coverages[index - 1]) * (risks[index] + risks[index - 1]) / 2.0
        for index in range(1, len(coverages))
    )
    return points, area


def _f1_counts(traces: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    labels = sorted(
        {str(item) for row in traces for item in row.get("authoritative_violations", [])}
        | {str(item) for row in traces for item in row.get("parsed_violations", [])}
    )
    if not labels:
        return 1.0, 1.0
    per_label: list[float] = []
    total_tp = total_fp = total_fn = 0
    for label in labels:
        tp = fp = fn = 0
        for row in traces:
            expected = label in row.get("authoritative_violations", [])
            predicted = label in row.get("parsed_violations", [])
            tp += int(expected and predicted)
            fp += int(not expected and predicted)
            fn += int(expected and not predicted)
        per_label.append(2 * tp / max(2 * tp + fp + fn, 1))
        total_tp += tp
        total_fp += fp
        total_fn += fn
    micro = 2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1)
    return micro, float(np.mean(per_label))


def _primary_coefficient(rows: Sequence[Mapping[str, Any]]) -> float | None:
    names = (
        "relative_h1_h2_update_norm",
        "final_verdict_entropy",
        "prompt_length",
        "output_length",
        "schema_complexity",
    )
    if len(rows) <= len(names):
        return None
    matrix = np.asarray([[float(row[name]) for name in names] for row in rows])
    scales = matrix.std(axis=0)
    scales[scales < 1e-12] = 1.0
    matrix = (matrix - matrix.mean(axis=0)) / scales
    design = np.column_stack((np.ones(len(rows)), matrix))
    labels = np.asarray([float(row["error_label"]) for row in rows])
    return float(np.linalg.lstsq(design, labels, rcond=None)[0][1])


def _bootstrap_interval(
    rows: Sequence[Mapping[str, Any]],
    statistic: Callable[[Sequence[Mapping[str, Any]]], float | None],
    *,
    seed: int,
    samples: int,
) -> list[float] | None:
    if len(rows) < 2:
        return None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        sample = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        value = statistic(sample)
        if value is not None and math.isfinite(value):
            values.append(value)
    if len(values) < max(10, samples // 10):
        return None
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _split_metrics(
    traces: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    labels = [float(row["error_label"]) for row in rows]
    risk_curve, aurc = _risk_coverage(labels, probabilities)
    micro, macro = _f1_counts(traces)
    paired = [dict(row, error_probability=probability) for row, probability in zip(rows, probabilities, strict=True)]

    def bootstrap_auroc(sample: Sequence[Mapping[str, Any]]) -> float | None:
        return _auroc(
            [float(row["error_label"]) for row in sample],
            [float(row["error_probability"]) for row in sample],
        )

    def bootstrap_brier(sample: Sequence[Mapping[str, Any]]) -> float:
        return float(
            np.mean(
                [
                    (float(row["error_probability"]) - float(row["error_label"])) ** 2
                    for row in sample
                ]
            )
        )

    def bootstrap_aurc(sample: Sequence[Mapping[str, Any]]) -> float:
        return _risk_coverage(
            [float(row["error_label"]) for row in sample],
            [float(row["error_probability"]) for row in sample],
        )[1]

    return {
        "count": len(rows),
        "schema_parse_rate": sum(bool(row.get("schema_parse_success")) for row in traces) / len(traces),
        "exact_decision_accuracy": sum(
            row.get("parsed_decision") == row.get("authoritative_decision") for row in traces
        )
        / len(traces),
        "violation_mask_micro_f1": micro,
        "violation_mask_macro_f1": macro,
        "error_prediction_auroc": _auroc(labels, probabilities),
        "brier_score": float(np.mean((np.asarray(probabilities) - np.asarray(labels)) ** 2)),
        "ece": _ece(labels, probabilities),
        "risk_coverage_curve": risk_curve,
        "area_under_risk_coverage_curve": aurc,
        "spearman_correlations": {
            name: _spearman([float(row[name]) for row in rows], labels)
            for name in RECURRENT_FEATURES + BASELINE_FEATURES
        },
        "bootstrap_95_percent_ci": {
            "error_prediction_auroc": _bootstrap_interval(
                paired, bootstrap_auroc, seed=seed, samples=bootstrap_samples
            ),
            "brier_score": _bootstrap_interval(paired, bootstrap_brier, seed=seed + 1, samples=bootstrap_samples),
            "area_under_risk_coverage_curve": _bootstrap_interval(
                paired, bootstrap_aurc, seed=seed + 2, samples=bootstrap_samples
            ),
        },
    }


def evaluate_cycle_signal(
    trace_path: Path,
    out_dir: Path,
    *,
    calibration_method: str = "logistic",
    bootstrap_samples: int = 200,
    seed: int = 7,
) -> dict[str, Any]:
    with trace_path.open(encoding="utf-8") as handle:
        traces = [json.loads(line) for line in handle if line.strip()]
    rows = [trace_to_analysis_row(row) for row in traces]
    calibration_rows = [row for row in rows if row["split"] == "calibration"]
    feature_names = BASELINE_FEATURES + RECURRENT_FEATURES
    if calibration_method == "logistic":
        calibration = fit_logistic_calibration(calibration_rows, feature_names)
        baseline_calibration = fit_logistic_calibration(calibration_rows, BASELINE_FEATURES)
    elif calibration_method == "isotonic":
        calibration = fit_isotonic_calibration(calibration_rows, "relative_h1_h2_update_norm")
        baseline_calibration = fit_isotonic_calibration(calibration_rows, "final_verdict_entropy")
    else:
        raise ValueError("calibration method must be logistic or isotonic")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_structured(
        out_dir / "calibration_model.json",
        {
            "primary_recurrent_plus_baselines": calibration.to_jsonable(),
            "baseline_only_comparator": baseline_calibration.to_jsonable(),
        },
    )
    results: dict[str, Any] = {}
    baseline_results: dict[str, Any] = {}
    for split in ("ordinary_test", "ood_test", "sealed_poison"):
        split_traces = [row for row in traces if row["split"] == split]
        split_rows = [row for row in rows if row["split"] == split]
        probabilities = [calibration.predict(row) for row in split_rows]
        baseline_probabilities = [baseline_calibration.predict(row) for row in split_rows]
        results[split] = _split_metrics(
            split_traces,
            split_rows,
            probabilities,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        baseline_results[split] = _split_metrics(
            split_traces,
            split_rows,
            baseline_probabilities,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 100,
        )
    ordinary_rows = [row for row in rows if row["split"] == "ordinary_test"]
    primary = _primary_coefficient(ordinary_rows)
    primary_ci = _bootstrap_interval(
        ordinary_rows,
        _primary_coefficient,
        seed=seed,
        samples=bootstrap_samples,
    )
    if primary_ci is None:
        primary_interpretation = "The primary association was not identifiable."
    elif primary_ci[0] <= 0.0 <= primary_ci[1]:
        primary_interpretation = (
            "The interval spans zero; this run does not support the preregistered positive association."
        )
    elif primary_ci[0] > 0.0:
        primary_interpretation = (
            "This run shows a positive association, not a causal mechanism, correctness guarantee, or alignment result."
        )
    else:
        primary_interpretation = "The estimated association is negative, contrary to the preregistered direction."
    report = {
        "schema_version": "1.0",
        "model_ids": sorted({str(row["model_id"]) for row in traces}),
        "evidence_status": (
            "pipeline_smoke_only_not_hrm_text_evidence"
            if all(str(row["model_id"]).startswith("deterministic-mock") for row in traces)
            else "optional_checkpoint_trace_experiment"
        ),
        "preregistered_primary_hypothesis": (
            "Larger late-cycle H-state change predicts a greater probability of an incorrect constitutional "
            "verdict after controlling for final-output entropy, prompt length, output length, and schema complexity."
        ),
        "calibration_fit_split": "calibration",
        "calibration_method": calibration_method,
        "ranking_quality": {
            key: {
                "recurrent_plus_baselines": value["error_prediction_auroc"],
                "baselines_only": baseline_results[key]["error_prediction_auroc"],
            }
            for key, value in results.items()
        },
        "probability_calibration": {
            key: {
                "recurrent_plus_baselines": {"brier_score": value["brier_score"], "ece": value["ece"]},
                "baselines_only": {
                    "brier_score": baseline_results[key]["brier_score"],
                    "ece": baseline_results[key]["ece"],
                },
            }
            for key, value in results.items()
        },
        "selective_prediction_quality": {
            key: {
                "area_under_risk_coverage_curve": value["area_under_risk_coverage_curve"],
                "risk_coverage_curve": value["risk_coverage_curve"],
                "baselines_only_area": baseline_results[key]["area_under_risk_coverage_curve"],
            }
            for key, value in results.items()
        },
        "primary_controlled_association": {
            "standardized_linear_probability_coefficient": primary,
            "bootstrap_95_percent_ci": primary_ci,
            "bootstrap_samples": bootstrap_samples,
            "confirmatory_split": "ordinary_test",
            "interpretation": primary_interpretation,
        },
        "split_results": results,
        "baseline_only_split_results": baseline_results,
        "non_claims": list(MANDATORY_NON_CLAIMS),
    }
    dump_structured(out_dir / "cycle_signal_report.json", report)
    (out_dir / "cycle_signal_report.md").write_text(_render_report(report), encoding="utf-8")
    (out_dir / "claim_boundary.md").write_text(_claim_boundary(report), encoding="utf-8")
    return report


def _format(value: Any) -> str:
    return "not identifiable" if value is None else f"{float(value):.4f}"


def _render_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# HRM-Text cycle-signal report",
        "",
        "> Architecture interpretations are hypotheses. The executable validator is the label authority.",
        "",
        f"Evidence status: `{report['evidence_status']}`; model IDs: {report['model_ids']}.",
        "",
        "## Preregistered hypothesis",
        "",
        str(report["preregistered_primary_hypothesis"]),
        "",
        "## Results",
        "",
        "| Split | Decision accuracy | Violation micro F1 | AUROC (+cycle) | AUROC (baseline) | "
        "Brier (+cycle) | ECE | AURC (selective) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, value in report["split_results"].items():
        lines.append(
            f"| {split} | {_format(value['exact_decision_accuracy'])} | "
            f"{_format(value['violation_mask_micro_f1'])} | {_format(value['error_prediction_auroc'])} | "
            f"{_format(report['baseline_only_split_results'][split]['error_prediction_auroc'])} | "
            f"{_format(value['brier_score'])} | {_format(value['ece'])} | "
            f"{_format(value['area_under_risk_coverage_curve'])} |"
        )
    primary = report["primary_controlled_association"]
    lines.extend(
        (
            "",
            "Ranking quality, probability calibration, and selective-prediction quality are separate evaluations.",
            "",
            "## Primary controlled association",
            "",
            f"Standardized late-H coefficient: {_format(primary['standardized_linear_probability_coefficient'])}; "
            f"bootstrap 95% interval: {primary['bootstrap_95_percent_ci']}.",
            "",
            str(primary["interpretation"]),
            "",
            "## Claim boundary",
            "",
        )
    )
    lines.extend(f"- {item}" for item in report["non_claims"])
    return "\n".join(lines) + "\n"


def _claim_boundary(report: Mapping[str, Any]) -> str:
    return "\n".join(("# Claim boundary", "", *(f"- {item}" for item in report["non_claims"]))) + "\n"
