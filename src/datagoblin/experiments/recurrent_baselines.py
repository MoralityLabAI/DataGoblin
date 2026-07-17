from __future__ import annotations

import gc
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.domain.recipes import DataRecipe
from datagoblin.hashing import sha256_file, sha256_text
from datagoblin.hrm_text.instrumented_forward import (
    DeterministicMockRecurrentModel,
    TraceOptions,
    run_instrumented_forward,
)
from datagoblin.hrm_text.prompting import build_verdict_prompt, schema_complexity
from datagoblin.hrm_text.trace_protocol import TRACE_ORDER, ForwardOutput, RecurrentStep
from datagoblin.serialization import dump_structured, load_structured

from .cycle_uncertainty import evaluate_cycle_signal
from .recipe_verdict_dataset import read_dataset_rows

RECURRENT_BASELINE_NAMES = ("tied_single_state_rnn", "two_timescale_rnn")


class DeterministicRecurrentBaseline:
    """Transparent recurrence-matched pipeline control, not a trained scientific baseline."""

    def __init__(self, kind: str) -> None:
        if kind not in RECURRENT_BASELINE_NAMES:
            raise ValueError(f"unknown recurrent baseline: {kind}")
        self.kind = kind
        self.model_id = f"deterministic-{kind}-v1"
        self._verdict_model = DeterministicMockRecurrentModel()

    def ordinary_forward(self, prompt: str) -> ForwardOutput:
        return self._verdict_model.ordinary_forward(prompt)

    def trace_forward(self, prompt: str, observer: Any) -> ForwardOutput:
        seed = int(sha256_text(f"{self.kind}:{prompt}")[:16], 16)
        rng = np.random.default_rng(seed)
        if self.kind == "tied_single_state_rnn":
            self._trace_tied(rng, observer)
        else:
            self._trace_two_timescale(rng, observer)
        return self._verdict_model.ordinary_forward(prompt)

    @staticmethod
    def _trace_tied(rng: np.random.Generator, observer: Any) -> None:
        state = rng.normal(0.0, 0.12, size=(8, 12))
        transition = rng.normal(0.0, 0.08, size=(12, 12))
        drive = rng.normal(0.0, 0.03, size=(8, 12))
        for name in TRACE_ORDER:
            state = np.tanh(0.72 * state + state @ transition + drive)
            observer(RecurrentStep(name=name, state=state.copy(), logits=_state_logits(state)))

    @staticmethod
    def _trace_two_timescale(rng: np.random.Generator, observer: Any) -> None:
        low = rng.normal(0.0, 0.10, size=(8, 12))
        high = rng.normal(0.0, 0.05, size=(8, 12))
        low_transition = rng.normal(0.0, 0.06, size=(12, 12))
        high_transition = rng.normal(0.0, 0.04, size=(12, 12))
        drive = rng.normal(0.0, 0.025, size=(8, 12))
        for name in TRACE_ORDER:
            if name.startswith("L"):
                low = np.tanh(0.78 * low + low @ low_transition + 0.25 * high + drive)
                exposed = low + high
            else:
                high = np.tanh(0.92 * high + high @ high_transition + 0.18 * low)
                exposed = high
            observer(RecurrentStep(name=name, state=exposed.copy(), logits=_state_logits(exposed)))

    def close(self) -> Mapping[str, Any]:
        gc.collect()
        return {"model_id": self.model_id, "closed": True, "cuda_cleanup": "not_applicable"}


def _state_logits(state: np.ndarray) -> np.ndarray:
    return np.asarray([state[-1, 0], state[-1, 1], state[-1, 2], -state[-1, 0], 0.0])


def trace_recurrent_baseline_suite(
    dataset_manifest: Path,
    out_dir: Path,
    *,
    trace_options: TraceOptions | None = None,
    condition: str = "direct",
) -> dict[str, Any]:
    rows = read_dataset_rows(dataset_manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, Any]] = {}
    for name in RECURRENT_BASELINE_NAMES:
        model = DeterministicRecurrentBaseline(name)
        model_dir = out_dir / name
        model_dir.mkdir(exist_ok=True)
        trace_path = model_dir / "trace_summary.jsonl"
        try:
            with trace_path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    recipe = DataRecipe.from_jsonable(row["recipe"])
                    label = row.get("authoritative_label")
                    summary = run_instrumented_forward(
                        model,
                        build_verdict_prompt(recipe, condition=condition),
                        authoritative_decision=label["decision"] if label else None,
                        authoritative_violations=label["violations"] if label else (),
                        schema_complexity=schema_complexity(recipe),
                        artifact_hashes=(str(row["recipe_hash"]),),
                        options=trace_options,
                    ).to_jsonable()
                    summary.update(
                        {
                            "example_id": row["example_id"],
                            "split": row["split"],
                            "category": row["category"],
                            "environment_family": row["environment_family"],
                            "role_condition": condition,
                        }
                    )
                    handle.write(json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n")
        finally:
            dump_structured(model_dir / "model_cleanup_receipt.json", model.close())
        entries[name] = {
            "trace_path": f"{name}/trace_summary.jsonl",
            "sha256": sha256_file(trace_path),
            "model_kind": "deterministic_pipeline_control",
        }
    manifest = {
        "schema_version": "1.0",
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "baselines": entries,
        "claim_boundary": (
            "These deterministic recurrences validate comparison plumbing; they are not trained performance baselines."
        ),
    }
    dump_structured(out_dir / "baseline_manifest.json", manifest)
    return manifest


def evaluate_recurrent_baseline_suite(
    baseline_root: Path,
    *,
    sealed_labels_path: Path,
    calibration_method: str = "logistic",
    bootstrap_samples: int = 200,
    seed: int = 7,
) -> None:
    manifest = load_structured(baseline_root / "baseline_manifest.json")
    for index, name in enumerate(manifest["baselines"]):
        model_dir = baseline_root / name
        evaluate_cycle_signal(
            model_dir / "trace_summary.jsonl",
            model_dir,
            sealed_labels_path=sealed_labels_path,
            calibration_method=calibration_method,
            bootstrap_samples=bootstrap_samples,
            seed=seed + index,
        )


def compare_recurrent_reports(primary_run: Path, baseline_root: Path, out_path: Path) -> dict[str, Any]:
    primary = load_structured(primary_run / "cycle_signal_report.json")
    baseline_manifest = load_structured(baseline_root / "baseline_manifest.json")
    reports = {
        name: load_structured(baseline_root / name / "cycle_signal_report.json")
        for name in baseline_manifest["baselines"]
    }
    comparison: dict[str, Any] = {
        "schema_version": "1.0",
        "primary_model_ids": primary["model_ids"],
        "primary_evidence_status": primary["evidence_status"],
        "splits": {},
        "claim_boundary": [
            "Deterministic baselines are pipeline controls, not substitutes for trained recurrent baselines.",
            "A cycle-signal difference does not identify an HRM-specific mechanism.",
            "A null result is valid.",
        ],
    }
    for split in ("ordinary_test", "ood_test", "sealed_poison"):
        comparison["splits"][split] = {
            "primary": _comparison_metrics(primary, split),
            **{name: _comparison_metrics(report, split) for name, report in reports.items()},
        }
    dump_structured(out_path.with_suffix(".json"), comparison)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_comparison_markdown(comparison), encoding="utf-8")
    return comparison


def _comparison_metrics(report: Mapping[str, Any], split: str) -> dict[str, Any]:
    value = report["split_results"][split]
    return {
        "error_prediction_auroc": value["error_prediction_auroc"],
        "brier_score": value["brier_score"],
        "area_under_risk_coverage_curve": value["area_under_risk_coverage_curve"],
    }


def _comparison_markdown(comparison: Mapping[str, Any]) -> str:
    lines = [
        "# Recurrent baseline comparison",
        "",
        f"Primary evidence status: `{comparison['primary_evidence_status']}`.",
        "",
        "| Split | Model | Error AUROC | Brier | AURC |",
        "|---|---|---:|---:|---:|",
    ]
    for split, models in comparison["splits"].items():
        for model, metrics in models.items():
            auroc = metrics["error_prediction_auroc"]
            auroc_text = "n/a" if auroc is None else f"{auroc:.4f}"
            lines.append(
                f"| {split} | {model} | {auroc_text} | {metrics['brier_score']:.4f} | "
                f"{metrics['area_under_risk_coverage_curve']:.4f} |"
            )
    lines.extend(("", "## Claim boundary", ""))
    lines.extend(f"- {item}" for item in comparison["claim_boundary"])
    return "\n".join(lines) + "\n"
