"""BSAR-09 recurrent repair gym: exact reward, custody and teacher controls."""

import json
from pathlib import Path

import pytest

from training.rlt_gym.bsar09.gym import (
    MATHCAN_MERGE_COMMIT,
    episode,
    evaluate,
    freeze,
    read_actions,
    score_test,
    teacher_action,
    tight_repair,
)
from training.rlt_gym.bsar09.mathcan_exact_oracle import (
    SOURCE_COMMIT,
    check_corpus,
    encode,
    generate,
    public_rows,
)


@pytest.fixture(scope="module")
def corpus():
    rows = generate()
    check_corpus(rows)
    return rows


def test_source_provenance_and_public_prompt_boundary(corpus):
    assert MATHCAN_MERGE_COMMIT == "0ce1a3c071142c502a41670a15c6cbca6fe14162"
    assert SOURCE_COMMIT == "80c62433b0eb671175bd8d6169822d3ebd922b60"
    assert len(corpus) == 464
    assert len(public_rows(corpus)) == 144
    for row in corpus:
        task = episode(row)
        assert set(task) == {"id", "pair_id", "prompt"}
        assert "gold_label" not in json.dumps(task)
        assert "oracle" not in task


def test_train_teacher_and_negative_tight_repairs(corpus):
    eligible = [row for row in corpus if row["split"] != "test"]
    for row in eligible:
        action = teacher_action(row)
        receipt = evaluate(row, action, teacher_feedback=True)
        assert receipt["complete"]
        assert receipt["reward"] == 1
        assert receipt["producer"] == "DETERMINISTIC_EXACT_ORACLE; NO_TRAINED_MODEL"
        if row["gold_label"] == "VIOLATES":
            assert action["repair"] == encode(tight_repair(row))
            assert receipt["failed_gates"] == [
                {"window": row["K"] - 1, "gates": ["window_ode"]}
            ]
        else:
            assert action["repair"] is None
    with pytest.raises(ValueError, match="withheld"):
        teacher_action(next(row for row in corpus if row["split"] == "test"))


def test_wrong_verdict_and_shortcut_repairs_fail(corpus):
    negative = next(row for row in corpus if row["gold_label"] == "VIOLATES")
    correct = {"verdict": "VIOLATES", "repair": encode(tight_repair(negative))}
    assert evaluate(negative, correct)["complete"]
    for action in (
        {"verdict": "SATISFIES", "repair": correct["repair"]},
        {"verdict": "VIOLATES", "repair": "0"},
        {"verdict": "VIOLATES", "repair": "123456789012345678901234567890/0"},
        {"verdict": "VIOLATES", "repair": "1" * 200},
        {"verdict": "VIOLATES"},
    ):
        assert not evaluate(negative, action)["complete"]
    good = evaluate(negative, correct)
    assert "gold_label" not in good
    assert "tight_repair" not in good


def test_constant_baseline_and_oracle_controls(corpus):
    test = [row for row in corpus if row["split"] == "test"]
    baseline = {row["id"]: {"verdict": "SATISFIES"} for row in test}
    result = score_test(corpus, baseline)
    assert result["cases"] == 144
    assert result["pairs"] == 72
    assert result["classification_accuracy"] == 0.5
    assert result["exact_repair_success_on_invalid"] == 0
    assert result["fully_completed_pairs"] == 0

    gold_actions = {
        row["id"]: {
            "verdict": row["gold_label"],
            "repair": encode(tight_repair(row)) if row["gold_label"] == "VIOLATES" else None,
        }
        for row in test
    }
    upper_control = score_test(corpus, gold_actions)
    assert upper_control["classification_accuracy"] == 1
    assert upper_control["fully_completed_pairs"] == 72


def test_manifest_reproducible_and_split_custody(corpus, tmp_path: Path):
    first = freeze(tmp_path)
    second = freeze(tmp_path)
    assert first == second
    assert not first["training_model_evaluated"]
    assert first["constant_positive_baseline"]["fully_completed_pairs"] == 0
    assert set(first["hashes"]) == {"train", "dev", "test_public"}
    test_rows = [
        json.loads(line)
        for line in (tmp_path / "test_public.jsonl").read_text().splitlines()
    ]
    assert len(test_rows) == 144
    assert all(set(row) == {"id", "pair_id", "prompt"} for row in test_rows)
    train_rows = [
        json.loads(line)
        for line in (tmp_path / "train.jsonl").read_text().splitlines()
    ]
    assert len(train_rows) == 256
    assert all(row["split"] == "train" for row in train_rows)
    assert all(row["verifier_receipt"]["complete"] for row in train_rows)


def test_prediction_validation_and_partial_coverage(corpus, tmp_path: Path):
    test_ids = {row["id"] for row in corpus if row["split"] == "test"}
    identity = next(iter(test_ids))
    file = tmp_path / "answers.jsonl"
    file.write_text(json.dumps({"id": identity, "verdict": "VIOLATES"}) + "\n")
    assert identity in read_actions(file, test_ids)
    assert score_test(corpus, read_actions(file, test_ids))["cases"] == 144

    invalid = [
        ([{"id": identity, "verdict": "SATISFIES", "gold_label": "SATISFIES"}], "leaked"),
        ([{"id": "train-root", "verdict": "SATISFIES"}], "Invalid held-out"),
        ([{"id": identity, "verdict": "SATISFIES"}] * 2, "Duplicate"),
    ]
    for records, error in invalid:
        file.write_text("".join(json.dumps(x) + "\n" for x in records))
        with pytest.raises(ValueError, match=error):
            read_actions(file, test_ids)
    with pytest.raises(ValueError, match="outside the test"):
        score_test(corpus, {"unknown": {"verdict": "SATISFIES"}})
