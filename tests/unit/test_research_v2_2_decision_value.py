"""Small deterministic regression for the decision-value planning contract."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def test_decision_value_oracle_replays_and_budgets():
    path = Path(__file__).resolve().parents[2] / "scripts" / "research_v2_2" / "decision_value.py"
    spec = importlib.util.spec_from_file_location("rv22i_decision_value", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for parity in (False, True):
        worlds = [mod.make_world(np.random.default_rng(i), parity) for i in range(8)]
        oracle = mod.score(worlds, "oracle")
        greedy = mod.score(worlds, "one_step")
        assert np.allclose(oracle[:, 1], 0)
        assert np.all(greedy[:, 1] >= -1e-9)
        assert np.all(greedy[:, 2] <= 7)
    # XOR requires two tests; a zero-information one-step action cannot win.
    world = {"hypotheses": [{"class": c, "weight": 1} for c in "ABBA"],
             "tests": [{"id": "x", "cost": 1, "bits": [0, 0, 1, 1]},
                       {"id": "y", "cost": 1, "bits": [0, 1, 0, 1]}],
             "budget": 2, "correct_reward": 12, "wrong_reward": -16}
    assert mod.score([world], "one_step")[0, 0] == 0
    assert mod.score([world], "oracle")[0, 0] == 10
    assert mod.score([world], "two_step")[0, 0] == 10
    w = mod.World(world)
    with pytest.raises(ValueError, match="repeated"):
        w.replay(("test", 0, ("test", 0, ("stop", "A"), ("stop", "B")), ("stop", "B")))
