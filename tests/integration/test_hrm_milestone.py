from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from datagoblin.cli import _trace_verdict_dataset
from datagoblin.experiments.cycle_uncertainty import evaluate_cycle_signal
from datagoblin.experiments.recipe_verdict_dataset import generate_verdict_dataset, steward_sealed_labels
from datagoblin.experiments.recurrent_baselines import (
    compare_recurrent_reports,
    evaluate_recurrent_baseline_suite,
    trace_recurrent_baseline_suite,
)


def test_offline_mock_hrm_milestone_emits_required_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    dataset = tmp_path / "dataset"
    steward = tmp_path / "steward"
    run = tmp_path / "run"
    generate_verdict_dataset(root / "examples/recipe.good.yaml", dataset, seed=7)
    steward_sealed_labels(dataset / "dataset_manifest.json", steward)
    args = Namespace(
        dataset_manifest=dataset / "dataset_manifest.json",
        out=run,
        config=root / "configs/hrm_text_trace.example.yaml",
        preregistration=root / "configs/preregistrations/cycle_uncertainty_v1.yaml",
        mock=True,
        local_model_path=None,
        model_id=None,
        allow_download=False,
        confirm_hard_cap_wrapper=False,
        condition="direct",
    )
    _trace_verdict_dataset(args)
    with pytest.raises(ValueError, match="independently stewarded"):
        evaluate_cycle_signal(run / "trace_summary.jsonl", run, bootstrap_samples=5)
    evaluate_cycle_signal(
        run / "trace_summary.jsonl",
        run,
        sealed_labels_path=steward / "sealed_labels.jsonl",
        bootstrap_samples=20,
    )
    required = {
        "preregistration.yaml",
        "dataset_manifest.json",
        "trace_summary.jsonl",
        "calibration_model.json",
        "cycle_signal_report.md",
        "claim_boundary.md",
    }
    assert required <= {path.name for path in run.iterdir()}
    report = (run / "cycle_signal_report.md").read_text(encoding="utf-8")
    assert "does not establish alignment" in report
    assert "A null result is valid" in report
    trace = json.loads((run / "trace_summary.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "sampled_activations" not in trace
    traces = [json.loads(line) for line in (run / "trace_summary.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["authoritative_decision"] is None for row in traces if row["split"] == "sealed_poison")

    baselines = tmp_path / "baselines"
    trace_recurrent_baseline_suite(dataset / "dataset_manifest.json", baselines)
    evaluate_recurrent_baseline_suite(
        baselines,
        sealed_labels_path=steward / "sealed_labels.jsonl",
        bootstrap_samples=10,
    )
    comparison = tmp_path / "recurrent_comparison.md"
    compare_recurrent_reports(run, baselines, comparison)
    assert comparison.is_file()
    assert "tied_single_state_rnn" in comparison.read_text(encoding="utf-8")
