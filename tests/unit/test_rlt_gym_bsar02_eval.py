"""BSAR-02 evaluation contract, with independent prediction and negative controls."""

import json
from pathlib import Path

import pytest

from training.rlt_gym.bsar02.evaluation import (
    ABSTAIN,
    LABELS,
    EXPECTED_COUNT,
    load_holdout,
    prepare,
    public_tasks,
    read_predictions,
    score,
)


@pytest.fixture(scope="module")
def frozen_holdout():
    return load_holdout()


def test_frozen_step7_holdout_is_source_pinned(frozen_holdout):
    assert len(frozen_holdout) == EXPECTED_COUNT == 164
    assert len({row["root"] for row in frozen_holdout.values()}) == 36
    assert all("Step7" in row["source_path"] for row in frozen_holdout.values())
    counts = {label: sum(row["gold_label"] == label for row in frozen_holdout.values()) for label in LABELS}
    assert counts == {
        "SOURCE_DECLARATION_PRESENT": 36,
        "UNSUPPORTED_PDE_LIFT": 36,
        "MISSING_ANALYTIC_TYPE_BRIDGE": 36,
        "KERNEL_CHECK_REQUIRED": 36,
        "NAMED_PREMISE_PRESENT": 10,
        "MISSING_PREMISE": 10,
    }


def test_prepare_exports_only_prompts_with_stable_digest(frozen_holdout, tmp_path: Path):
    assert all(not ({"gold_label", "gold_reason", "template", "split"} & row.keys())
               for row in public_tasks(frozen_holdout))
    report = prepare(frozen_holdout, tmp_path)
    task_path = tmp_path / "step7_public_tasks.jsonl"
    rows = [json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == EXPECTED_COUNT
    assert all("gold_label" not in row for row in rows)
    assert report["labels_excluded_from_model_prompts"] is True
    assert report["source_commit"] == "80c62433b0eb671175bd8d6169822d3ebd922b60"
    assert report == prepare(frozen_holdout, tmp_path)


def test_majority_baseline_and_all_abstain(frozen_holdout):
    majority = {key: "SOURCE_DECLARATION_PRESENT" for key in frozen_holdout}
    report = score(frozen_holdout, majority)
    assert report["examples"] == 164
    assert report["strict_accuracy"] == pytest.approx(36 / 164)
    assert report["macro_f1"] == pytest.approx(0.06)
    assert report["perfect_roots"] == 0
    no_answers = score(frozen_holdout, {})
    assert no_answers["coverage"] == 0
    assert no_answers["strict_accuracy"] == 0
    assert no_answers["abstentions"] == 164
    assert no_answers["macro_f1"] == 0


def test_positive_scoring_control(frozen_holdout):
    control = {key: row["gold_label"] for key, row in frozen_holdout.items()}
    report = score(frozen_holdout, control)
    assert report["coverage"] == 1
    assert report["macro_f1"] == 1
    assert report["perfect_roots"] == report["total_roots"] == 36


def test_invalid_predictions_and_leakage_rejected(frozen_holdout, tmp_path: Path):
    identity = next(iter(frozen_holdout))
    first = {"id": identity, "label": ABSTAIN}
    cases = [
        ([first, first], "Duplicate task id"),
        ([{"id": "train-case", "label": LABELS[0]}], "Unknown or non-test"),
        ([{"id": identity, "label": "MODEL_SOLVED_PDE"}], "Unknown predicted label"),
        ([{**first, "gold_label": LABELS[0]}], "Gold label field"),
    ]
    for index, (rows, message) in enumerate(cases):
        path = tmp_path / f"invalid-{index}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            read_predictions(path, set(frozen_holdout))
    with pytest.raises(ValueError, match="outside frozen"):
        score(frozen_holdout, {"wrong-id": LABELS[0]})


def test_partial_coverage_counts_abstain_as_incorrect(frozen_holdout):
    identity = next(iter(frozen_holdout))
    report = score(frozen_holdout, {identity: frozen_holdout[identity]["gold_label"]})
    assert report["strict_accuracy"] == pytest.approx(1 / 164)
    assert report["coverage"] == pytest.approx(1 / 164)
    assert report["answer_accuracy"] == 1.0
    assert report["abstentions"] == 163
