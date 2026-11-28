from itertools import product
from random import Random

import pytest

from datagoblin.training_gym.spectral_routing import TaskError, score
from datagoblin.training_gym.spectral_search import SearchSession, best_bound, verify_trace


def fixture(seed):
    rng = Random(seed)
    d, t = 4, 3
    return {"d": d, "T": t, "bits": [rng.randint(1, 3) for _ in range(d)], "x": [-2, 0, 1, 4],
            "k": [rng.randrange(d + 1) for _ in range(t)], "lambda": rng.randrange(4),
            "u": [[rng.randrange(-5, 15) for _ in range(d)] for _ in range(t)]}


def choices(task):
    return [[m for m in range(1 << task["d"]) if m.bit_count() == k] for k in task["k"]]


def test_exhaustive_and_bounded():
    for seed in range(64):
        task = fixture(seed)
        actions = choices(task)
        initial = [row[0] for row in actions]
        optimum = max(score(task, list(a))["objective"] for a in product(*actions))
        for cap in (1, 12, 10000):
            result = best_bound(task, initial, transition_cap=cap)
            assert result["objective"] <= optimum <= result["upper"]
            assert result["transitions"] <= cap
            assert verify_trace(task, initial, result["trace"], transition_cap=cap) == result
            if cap == 10000:
                assert result["gap"] == 0


def test_arbitrary_student_priority():
    for seed in range(20):
        task = fixture(seed + 1000)
        initial = [row[0] for row in choices(task)]
        optimum = max(score(task, list(a))["objective"] for a in product(*choices(task)))
        rng = Random(seed)
        session = SearchSession(task, initial, transition_cap=10000)
        while session.frontier():
            t, s, _ = rng.choice(session.frontier())
            session.advance(t, s)
        result = session.receipt()
        assert result["gap"] == 0 and result["objective"] == optimum
        assert verify_trace(task, initial, result["trace"], transition_cap=10000) == result


def test_reject_invalid_trace():
    task = fixture(123)
    initial = [row[0] for row in choices(task)]
    for trace in ([[0, 0]], [[True, 0]], [[3]], [[3, -1]]):
        with pytest.raises(TaskError):
            verify_trace(task, initial, trace)


def test_cap_rejection_is_atomic():
    task = {"d": 4, "T": 1, "bits": [1] * 4, "x": [0, 1], "k": [2], "lambda": 0,
            "u": [[0, 1, 2, 3]]}
    session = SearchSession(task, [3], transition_cap=1)
    before = session.receipt()
    with pytest.raises(TaskError):
        session.advance(1, 0)
    assert session.receipt() == before


def test_translation_and_exact_initial():
    task = fixture(222)
    initial = [row[0] for row in choices(task)]
    result = best_bound(task, initial, transition_cap=10000)
    shifted = {**task, "x": [x + 5 for x in task["x"]]}
    assert best_bound(shifted, initial, transition_cap=10000) == result
    assert best_bound(task, result["schedule"], transition_cap=10000)["gap"] == 0
