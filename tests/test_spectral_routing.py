from itertools import product
from random import Random

import pytest

from datagoblin.training_gym.spectral_routing import TaskError, grade, oracle, score, training_example


def fixture() -> dict:
    return {"d": 3, "T": 3, "bits": [1, 2, 3], "x": [-1, 0, 2, 4], "k": [1, 2, 1],
            "u": [[4, 2, 1], [2, 5, 3], [1, 3, 8]], "lambda": 1}


def test_exhaustive():
    rng = Random(730030)
    for _ in range(64):
        task = fixture()
        task["u"] = [[rng.randrange(-3, 12) for _ in range(3)] for _ in range(3)]
        task["lambda"] = rng.randrange(4)
        task["k"] = [rng.randrange(4) for _ in range(3)]
        choices = [[m for m in range(8) if m.bit_count() == k] for k in task["k"]]
        optimum = max(score(task, list(s))["objective"] for s in product(*choices))
        result = oracle(task)
        assert result["objective"] == optimum
        assert grade(task, result["schedule"])["optimal"]


def test_translation():
    task = fixture()
    translated = {**task, "x": [x + 50 for x in task["x"]]}
    assert oracle(task) == oracle(translated)


def test_ties_and_empty_budget():
    task = {**fixture(), "lambda": 0, "u": [[0] * 3 for _ in range(3)], "k": [0, 1, 0]}
    assert grade(task, [0, 4, 0])["optimal"]
    assert oracle(task)["schedule"] == [0, 1, 0]


def test_invalid_and_cap():
    task = fixture()
    with pytest.raises(TaskError):
        score(task, [1, 1, 1])
    with pytest.raises(TaskError):
        score(task, [True, 3, 1])
    with pytest.raises(TaskError):
        oracle(task, transition_cap=1)
    with pytest.raises(TaskError):
        oracle({**task, "x": [0, 3, 2, 4]})


def test_training_record():
    record = training_example({**fixture(), "private": "discard"})
    assert "private" not in record["task"]
    assert grade(record["task"], record["teacher"]["schedule"])["optimal"]
