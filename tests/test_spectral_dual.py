from itertools import product
from random import Random

import pytest

from datagoblin.training_gym.spectral_dual import DualSearchSession, coordinate_minimizer, dual_bound, verify_dual_trace
from datagoblin.training_gym.spectral_routing import TaskError, oracle


def fixture(seed):
    r = Random(seed)
    return {"d": 4, "T": 3, "bits": [r.randint(1, 3) for _ in range(4)], "x": [-2, 0, 1, 4],
            "k": [r.randrange(5) for _ in range(3)], "lambda": r.randrange(4),
            "u": [[r.randrange(-5, 15) for _ in range(4)] for _ in range(3)]}


def exhaustive_remaining(q, t, suffix):
    choices = [[m for m in range(1 << q["d"]) if m.bit_count() == k] for k in q["k"][:t]]
    best = None
    for masks in product(*choices):
        utility = sum(q["u"][i][j] for i, m in enumerate(masks) for j in range(q["d"]) if m & (1 << j))
        cost = sum(q["bits"][j] * max([0, *[q["x"][i + 1] - q["x"][0] for i, m in enumerate(masks)
                                           if m & (1 << j)]]) for j in range(q["d"]) if not suffix & (1 << j))
        value = utility - q["lambda"] * cost
        best = value if best is None else max(best, value)
    return best


def test_bounds_and_coordinate_optima():
    for seed in range(32):
        q = fixture(seed)
        r = Random(seed + 300)
        for s in range(1 << q["d"]):
            t = r.randrange(1, q["T"] + 1)
            eta = [r.randrange(-20, 21) for _ in range(t)]
            optimum = exhaustive_remaining(q, t, s)
            assert dual_bound(q, t, s, eta) >= optimum
            for i in range(t):
                old = dual_bound(q, t, s, eta)
                eta[i] = coordinate_minimizer(q, t, s, eta, i)
                got = dual_bound(q, t, s, eta)
                assert optimum <= got <= old
                for trial in (-60, -10, 0, 10, 60):
                    alt = eta.copy()
                    alt[i] = trial
                    assert dual_bound(q, t, s, alt) >= got


def test_arbitrary_proposals_and_replay():
    for seed in range(40):
        q = fixture(seed + 100)
        init = [(1 << k) - 1 for k in q["k"]]
        optimum = oracle(q)["objective"]
        r = Random(seed)
        session = DualSearchSession(q, init, transition_cap=10000, proposal_cap=10000)
        while session.frontier():
            t, s, _ = r.choice(session.frontier())
            eta = [r.randrange(-25, 26) for _ in range(t)]
            for i in range(t - 1, -1, -1):
                eta[i] = coordinate_minimizer(q, t, s, eta, i)
            before = session.receipt()["upper"]
            session.propose(t, s, eta)
            after = session.receipt()
            assert after["objective"] <= optimum <= after["upper"] <= before
            if any(v[:2] == (t, s) for v in session.frontier()):
                session.advance(t, s)
        receipt = session.receipt()
        assert receipt["objective"] == optimum and receipt["gap"] == 0
        assert verify_dual_trace(q, init, receipt["events"], transition_cap=10000, proposal_cap=10000) == receipt


def test_malformed_and_cap_atomic():
    q = {"d": 3, "T": 1, "bits": [1, 1, 1], "x": [0, 1], "k": [1], "lambda": 0, "u": [[0, 2, 3]]}
    session = DualSearchSession(q, [1], transition_cap=3, proposal_cap=1)
    before = session.receipt()
    for bad in ([], [True], [1.0], [10**10]):
        with pytest.raises(TaskError):
            session.propose(1, 0, bad)
        assert session.receipt() == before
    session.propose(1, 0, [0])
    before = session.receipt()
    with pytest.raises(TaskError):
        session.propose(1, 0, [0])
    assert session.receipt() == before
    with pytest.raises(TaskError):
        verify_dual_trace(q, [1], [{"kind": "dual", "state": [1, 0], "eta": [0], "gap": 0}])


def test_empty_prefix():
    q = fixture(42)
    assert dual_bound(q, 0, 0, []) == 0
