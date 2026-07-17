from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datagoblin.environments.causal_switches.generator import GeneratedDataset, generate_dataset
from datagoblin.environments.causal_switches.verifier import VerificationReceipt, verify_dataset
from datagoblin.hashing import sha256_json
from datagoblin.learners.numpy_linear import NumpyLogisticLearner


@dataclass(frozen=True)
class ToyHarnessResult:
    baseline_metrics: dict[str, float]
    improved_metrics: dict[str, float]
    actual_metrics: dict[str, float]
    train_manifest: dict[str, Any]
    evaluation_public_manifest: dict[str, Any]
    verification_receipts: tuple[VerificationReceipt, ...]
    baseline_learner_receipt: dict[str, Any]
    improved_learner_receipt: dict[str, Any]
    interpretation: str


def _public_evaluation_manifest(dataset: GeneratedDataset) -> dict[str, Any]:
    manifest = dict(dataset.split_manifest)
    manifest.pop("seed_commitment", None)
    return {
        **manifest,
        "seed_namespace_commitment": sha256_json({"namespace": manifest["namespace"]}),
        "private_seed_count": 1,
        "private_seed_values_disclosed": False,
        "labels_disclosed": False,
    }


def run_toy_harness(seed: int) -> ToyHarnessResult:
    baseline = generate_dataset(
        seed=seed + 101,
        count=512,
        mode="biased",
        namespace="causal-switches/train-baseline/v1",
    )
    improved = generate_dataset(
        seed=seed + 202,
        count=512,
        mode="balanced",
        namespace="causal-switches/train/v1",
        include_counterfactual_pairs=True,
    )
    private_evaluation_seed = seed + 900_001
    evaluation = generate_dataset(
        seed=private_evaluation_seed,
        count=384,
        mode="permuted_holdout",
        namespace="causal-switches/eval-hidden/v1",
    )
    receipts = tuple(verify_dataset(item) for item in (baseline, improved, evaluation))
    if not all(receipt.accepted for receipt in receipts):
        raise RuntimeError("independent verifier rejected generated labels")
    baseline_model = NumpyLogisticLearner().fit(baseline.features, baseline.labels)
    improved_model = NumpyLogisticLearner().fit(improved.features, improved.labels)
    baseline_metrics = baseline_model.evaluate(evaluation.features, evaluation.labels)
    improved_metrics = improved_model.evaluate(evaluation.features, evaluation.labels)
    delta = improved_metrics["accuracy"] - baseline_metrics["accuracy"]
    brier_delta = baseline_metrics["brier"] - improved_metrics["brier"]
    if delta > 0.05:
        interpretation = "Toy evidence supports the role-shortcut intervention under this frozen smoke test."
    elif delta < -0.05:
        interpretation = "Toy evidence is negative: the proposed intervention underperformed the biased control."
    else:
        interpretation = "Toy result is null or small; retain it and test the history-feature ablation next."
    return ToyHarnessResult(
        baseline_metrics,
        improved_metrics,
        {"heldout_accuracy_delta": delta, "heldout_brier_improvement": brier_delta},
        improved.split_manifest,
        _public_evaluation_manifest(evaluation),
        receipts,
        baseline_model.receipt(),
        improved_model.receipt(),
        interpretation,
    )
