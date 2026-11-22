"""BSAR-05 exact-integer synthetic gym checks with adversarial premise violations."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from training.rlt_gym.bsar05.semantic_steps import (
    LABELS,
    SOURCE_COMMIT,
    Step,
    allowed,
    generate,
    premises,
    public_test,
    score,
    validate,
    write_jsonl,
)


@pytest.fixture(scope="module")
def examples():
    return generate()


def test_arithmetic_oracle_single_premise_fails():
    valid = Step(m=6, im=3, previous=11, next_state=22)
    assert allowed(valid)
    assert all(premises(valid).values())
    invalid = (
        (Step(6, 3, 11, -1), "next_nonnegative"),
        (Step(6, 4, 11, 22), "half_budget"),
        (Step(6, 3, 11, 23), "window_ode"),
    )
    for step, wrong_gate in invalid:
        assert not allowed(step)
        assert [key for key, value in premises(step).items() if not value] == [wrong_gate]


def test_paired_root_disjoint_holdouts(examples):
    report = validate(examples)
    assert report["source_commit"] == SOURCE_COMMIT
    assert report["pairs"] == 525
    assert report["examples"] == 1050
    assert report["by_split"] == {"train": 600, "dev": 180, "test": 270}
    assert report["by_mutation"] == {mode: 175 for mode in
                                     ("next_nonnegative", "half_budget", "window_ode")}
    roots = defaultdict(set)
    for row in examples:
        roots[row["root"]].add(row["split"])
    assert all(len(splits) == 1 for splits in roots.values())


def test_paired_labels_are_balanced_and_semantic(examples):
    by_pair = defaultdict(list)
    for row in examples:
        by_pair[row["pair_id"]].append(row)
    for pair in by_pair.values():
        assert {r["gold_label"] for r in pair} == set(LABELS)
        assert len({r["source_commit"] for r in pair}) == 1
        assert len({r["root"] for r in pair}) == 1
        assert pair[0]["parameters"] != pair[1]["parameters"]
        assert all(row["oracle_gates"] == premises(Step(**row["parameters"])) for row in pair)


def test_public_prompts_never_include_gold_or_gate_trace(examples):
    tasks = public_test(examples)
    assert len(tasks) == 270
    assert not any("gold_label" in row or "oracle_gates" in row for row in tasks)
    assert all("LerayDoublingChain.step" in row["prompt"] for row in tasks)
    assert len({row["id"] for row in tasks}) == len(tasks)


def test_scoring_controls_and_abstention(examples):
    test = [row for row in examples if row["split"] == "test"]
    constant = {row["id"]: "ADMISSIBLE" for row in test}
    baseline = score(examples, constant)
    assert baseline["strict_accuracy"] == 0.5
    assert baseline["complete_pairs"] == 0
    oracle = {row["id"]: row["gold_label"] for row in test}
    full = score(examples, oracle)
    assert full["strict_accuracy"] == 1
    assert full["complete_pairs"] == 135
    assert score(examples, {})["coverage"] == 0
    with pytest.raises(ValueError, match="outside test"):
        score(examples, {"bogus": "REJECTED"})
    with pytest.raises(ValueError, match="Unknown predicted"):
        score(examples, {test[0]["id"]: "PROOF_CLOSED"})


def test_seed_reproduction_and_manifest_digest(examples, tmp_path: Path):
    assert generate() == examples
    assert {row["id"] for row in generate(1)} == {row["id"] for row in examples}
    targets = public_test(examples)
    file = tmp_path / "public.jsonl"
    digest = write_jsonl(file, targets)
    assert digest == write_jsonl(file, targets)
    assert [json.loads(line) for line in file.read_text().splitlines()] == targets
