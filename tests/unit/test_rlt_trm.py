import numpy as np
import pytest

from datagoblin.training_gym.rlt_revision import episodes


def test_patch_is_visible_only_after_first_answer() -> None:
    pytest.importorskip("autograd")
    from datagoblin.training_gym import rlt_trm as pilot

    tasks = [task for task, _ in episodes(17, "development", 1)]
    data = pilot.prepare(tasks)
    np.testing.assert_array_equal(data[2][0], data[2][1])
    first_graph, patched_graph = data[3]
    changed = np.flatnonzero(np.any(first_graph != patched_graph, axis=1))
    assert changed.tolist() == [pilot.NODE_ID[tasks[0]["patch"]["source"]]]
    for mode in pilot.MODES:
        before, _ = pilot.two_turns(pilot.init_weights(17), data, mode, 6)
        np.testing.assert_array_equal(before[0], before[1])


def test_only_one_visit_has_same_update_in_each_arm() -> None:
    pytest.importorskip("autograd")
    from datagoblin.training_gym import rlt_trm as pilot

    tasks = [task for task, _ in episodes(17, "development", 1)]
    data = pilot.prepare(tasks)
    weights = pilot.init_weights(17)
    before, after = pilot.two_turns(weights, data, "direct", 1)
    for mode in pilot.MODES:
        actual_before, actual_after = pilot.two_turns(weights, data, mode, 1)
        np.testing.assert_array_equal(actual_before, before)
        np.testing.assert_array_equal(actual_after, after)
