"""BSAR-03 matched-source-claim audit with shuffled positives and negatives."""

import json
from collections import Counter, defaultdict

import pytest

from training.rlt_gym.bsar02.boundary_curriculum import CATALOG
from training.rlt_gym.bsar03.matched_claims import (
    TYPE_PATTERN,
    generate_pairs,
    public_rows,
    score,
    validate,
)


@pytest.fixture(scope="module")
def examples():
    return generate_pairs(json.loads(CATALOG.read_text(encoding="utf-8")))


def test_matched_claims_balance_and_provenance(examples):
    report = validate(examples)
    assert report["source_commit"] == "80c62433b0eb671175bd8d6169822d3ebd922b60"
    assert report["unique_roots"] == 95
    assert report["examples"] > 350
    assert report["test_pairs"] > 55
    for split in ("train", "dev", "test"):
        labels = Counter(row["gold_label"] for row in examples if row["split"] == split)
        assert labels["PRESENT"] == labels["ABSENT"] > 0
    pairs = defaultdict(list)
    for item in examples:
        pairs[item["pair_id"]].append(item)
    assert all({row["gold_label"] for row in pair} == {"PRESENT", "ABSENT"}
               for pair in pairs.values())


def test_question_template_only_baseline_cannot_solve_pairs(examples):
    test = [row for row in examples if row["split"] == "test"]
    # This deliberately ignores the concrete candidate and the theorem text.
    pred = {row["id"]: "PRESENT" for row in test}
    result = score(examples, pred)
    assert result["accuracy"] == 0.5
    assert result["coverage"] == 1
    assert result["complete_pairs"] == 0
    assert result["test_pairs"] == len(test) // 2


def test_source_aware_oracle_and_missing_predictions(examples):
    pred = {row["id"]: row["gold_label"] for row in examples if row["split"] == "test"}
    report = score(examples, pred)
    assert report["accuracy"] == 1
    assert report["complete_pairs"] == report["test_pairs"]
    blank = score(examples, {})
    assert blank["accuracy"] == 0
    assert blank["coverage"] == 0
    with pytest.raises(ValueError, match="non-test"):
        score(examples, {"not-in-test": "PRESENT"})
    with pytest.raises(ValueError, match="Unknown output"):
        score(examples, {next(iter(pred)): "PROVES_NAVIER_STOKES"})


def test_public_rows_withhold_gold_and_train_roots(examples):
    tasks = public_rows(examples)
    assert tasks
    assert all("Step7" in row["source_path"] for row in tasks)
    assert all("gold_label" not in row for row in tasks)
    assert len({row["id"] for row in tasks}) == len(tasks)
    assert all("Signature: theorem " in row["prompt"] for row in tasks)


def test_type_boundaries_use_exact_lean_type_bindings():
    assert set(TYPE_PATTERN.findall("(x : Int) (y : Nat)")) == {"Int", "Nat"}
    assert TYPE_PATTERN.findall("(x : Other) (h : Prop)") == []


def test_generation_seed_and_pair_order_are_frozen(examples):
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert examples == generate_pairs(catalog)
    assert {row["id"] for row in examples} == {row["id"] for row in generate_pairs(catalog, seed=31)}
