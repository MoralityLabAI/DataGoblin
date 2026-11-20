import numpy as np
import pytest

from datagoblin.training_gym.rlt_lean_grounded import candidates
from datagoblin.training_gym.rlt_lean_rank import BOS, EOS, PAD, candidate_tokens


def test_grounding_uses_visible_goal_operations_and_bounds_script_bytes():
    entries = [{"name": name} for name in ("List.map_zipWith", "List.take_zipWith", "List.reverse_drop")]
    hits, pool = candidates("⊢ (List.zipWith f xs ys).take k |>.map g", entries)
    assert hits == ["List.map_zipWith", "List.take_zipWith"]
    assert "simp [List.map_zipWith, List.take_zipWith]" in pool
    assert len(pool) == len(set(pool))

    script = "exact h₁"
    inputs, labels = candidate_tokens((script, pool[0]))
    encoded = list(script.encode("utf-8"))
    end = len(encoded) + 1
    np.testing.assert_array_equal(inputs[0, :end], [BOS, *encoded])
    np.testing.assert_array_equal(labels[0, :end], [*encoded, EOS])
    assert np.all(labels[0, end:] == PAD)
    with pytest.raises(ValueError, match="decoder window"):
        candidate_tokens(("é" * 48,))
