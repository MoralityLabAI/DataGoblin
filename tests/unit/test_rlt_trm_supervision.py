import numpy as np
import pytest


def test_each_graph_instruction_has_an_independent_step_target():
    pytest.importorskip("autograd")
    from datagoblin.training_gym.rlt_trm_supervision import NODES, batch

    vertices = np.arange(NODES)
    maps = np.repeat([[(vertices + 1) % NODES, (vertices + 4) % NODES]], 3, axis=0)
    predecessors, starts, _, labels = batch(maps, np.zeros(3, dtype=int), np.arange(3), 4)
    assert labels.tolist() == [[1, 2, 3, 4], [11, 10, 9, 8], [1, 5, 6, 10]]
    state = np.eye(NODES)[starts]
    for cycle in range(4):
        state = state[np.arange(3)[:, None], predecessors[:, cycle]]
        np.testing.assert_array_equal(state.argmax(axis=-1), labels[:, cycle])


def test_svd_isolates_node_contrast_and_ablation_controls_for_size():
    pytest.importorskip("autograd")
    from datagoblin.training_gym.rlt_trm_svd import matched_ablation, spectrum

    rng = np.random.default_rng(11)
    calibration = rng.normal(size=(32, 5, 4)) * np.array([8, 5, 0.3, 0.1])
    held = rng.normal(size=(19, 5, 4)) * np.array([8, 5, 0.3, 0.1]) + rng.normal(size=(19, 1, 4)) * 30
    mean, leading, stats = spectrum(calibration, held)
    assert stats["held_top_energy"] > 0.99
    top, control, dose = matched_ablation(held, mean, leading, np.eye(4)[2:])
    assert dose["matched_examples"] == len(held)
    for value in (top, control):
        np.testing.assert_allclose(value.mean(axis=1), held.mean(axis=1), atol=1e-10)
    np.testing.assert_allclose((top - top.mean(axis=1, keepdims=True)) @ leading.T, 0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(top - held, axis=(1, 2)), np.linalg.norm(control - held, axis=(1, 2)))


def test_fit_resume_keeps_the_weights_and_cycle_log(tmp_path, monkeypatch):
    pytest.importorskip("autograd")
    from datagoblin.serialization import load_structured
    from datagoblin.training_gym import rlt_trm_supervision as study

    monkeypatch.setattr(study, "STEPS", 3)
    monkeypatch.setattr(study, "CHECKPOINTS", (0, 1, 3))
    train = study.batch(*study.make_split(np.random.default_rng(7), 3, set()), study.TRAIN_DEPTH)
    calls = 0

    def interrupt():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("pause")

    with pytest.raises(RuntimeError, match="pause"):
        study.fit(tmp_path / "resumed", train, train, 3, "answer_rlt", "stepwise", interrupt)
    resumed = study.fit(tmp_path / "resumed", train, train, 3, "answer_rlt", "stepwise", lambda: None)
    continuous = study.fit(tmp_path / "continuous", train, train, 3, "answer_rlt", "stepwise", lambda: None)
    relative = "fits/3-stepwise-answer_rlt/weights.npz"
    with (
        np.load(tmp_path / "resumed" / relative, allow_pickle=False) as a,
        np.load(tmp_path / "continuous" / relative, allow_pickle=False) as b,
    ):
        assert a.files == b.files
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    assert resumed["cycles_sha256"] == continuous["cycles_sha256"]
    log = load_structured(tmp_path / "resumed" / "fits" / "3-stepwise-answer_rlt" / "cycles.json")
    assert set(log) == {"0", "1", "3"}
