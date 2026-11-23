from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from datagoblin.training_gym.sgrt_holoarc import (
    BASE_DIM,
    FIBER_DIM,
    arc_length,
    random_smooth_path,
    spiro_path,
    step_transport,
    trace_transport,
    transported_residual,
)
from datagoblin.training_gym.sgrt_holoarc_gym import (
    ARMS,
    OPERATORS,
    execute,
    make_task,
    policy_seed,
    propose,
    run_suite,
    summarize,
    verify_and_score,
)


@pytest.mark.parametrize("seed", [3, 5, 23, 42])
def test_multidimensional_closed_gear_geometry(seed: int) -> None:
    center = np.linspace(-0.5, 0.5, BASE_DIM)
    path = spiro_path(center, seed=seed)
    assert path.shape == (9, BASE_DIM)
    assert np.array_equal(path[0], center)
    assert np.array_equal(path[-1], center)
    assert arc_length(path) > 0
    assert np.allclose(spiro_path(center, seed=seed), path)


def test_random_arc_length_is_budget_matched() -> None:
    anchor = np.zeros(BASE_DIM)
    gear = spiro_path(anchor, seed=43)
    random_path = random_smooth_path(anchor, seed=43)
    assert np.array_equal(random_path[0], anchor)
    assert np.array_equal(random_path[-1], anchor)
    assert abs(arc_length(gear) - arc_length(random_path)) < 1e-12


def test_flat_holonomy_is_identity_and_curved_is_nontrivial() -> None:
    path = spiro_path(np.zeros(BASE_DIM), seed=23)
    flat = trace_transport(path, mode="flat")
    curved = trace_transport(path, mode="curved")
    flat_report = flat.receipt()
    curved_report = curved.receipt()
    assert flat_report["identity_loss"] == pytest.approx(0.0, abs=1e-12)
    assert curved_report["identity_loss"] > 1e-4
    for receipt in (flat_report, curved_report):
        assert receipt["determinant"] == pytest.approx(1.0, abs=1e-10)
        assert receipt["orthogonality_error_fro"] < 1e-10
        assert receipt["dual_pairing_drift"] < 1e-10
        assert receipt["path_closure_l2"] == 0


def test_holonomy_is_gauge_covariant_and_invariants_are_fixed() -> None:
    path = spiro_path(np.zeros(BASE_DIM), seed=5)
    trace = trace_transport(path, mode="curved")
    gauge = np.eye(FIBER_DIM)[::-1]
    # Change fiber basis on every edge. A Wilson loop transforms by conjugation.
    product = np.eye(FIBER_DIM)
    for delta in np.diff(path, axis=0):
        q = step_transport(delta, mode="curved")
        product = (gauge @ q @ gauge.T) @ product
    assert np.allclose(product, gauge @ trace.holonomy @ gauge.T, atol=1e-10)
    assert np.trace(product) == pytest.approx(np.trace(trace.holonomy), abs=1e-10)
    assert np.linalg.det(product) == pytest.approx(np.linalg.det(trace.holonomy), abs=1e-10)


def test_transport_sidecar_is_bounded_and_identity_gated() -> None:
    state = np.linspace(-1, 1, BASE_DIM)
    holonomy = trace_transport(spiro_path(np.zeros(BASE_DIM), seed=23), mode="curved").holonomy
    assert np.array_equal(transported_residual(state, holonomy, gate=0), state)
    assert np.all(np.isfinite(transported_residual(state, holonomy, gate=0.1)))
    with pytest.raises(ValueError):
        transported_residual(state, holonomy, gate=1.01)


def test_dsl_is_total_on_four_bit_states() -> None:
    for a in OPERATORS:
        for b in OPERATORS:
            for x in range(16):
                y = execute((a, b), x)
                assert isinstance(y, int) and 0 <= y <= 15


def test_candidate_generation_is_target_blind() -> None:
    task = make_task(seed=23, root=2)
    altered_target = replace(task, target=("id", "id"))
    assert task.public() == altered_target.public()
    for arm in ARMS:
        original = propose(task.public(), seed=71, arm=arm, budget=8)
        altered = propose(altered_target.public(), seed=71, arm=arm, budget=8)
        candidates_a, receipt_a = original
        candidates_b, receipt_b = altered
        assert candidates_a == candidates_b
        assert receipt_a == receipt_b
        assert len(candidates_a) == len(set(candidates_a))


def test_matched_phase_shuffle_preserves_candidate_set_and_selected_result() -> None:
    task = make_task(seed=41, root=3)
    seed = policy_seed(41, 3)
    curved, curved_receipt = propose(task.public(), seed=seed, arm="curved_holoarc", budget=8)
    shuffled, shuffled_receipt = propose(
        task.public(), seed=seed, arm="same_set_phase_shuffle", budget=8
    )
    assert sorted(curved) == sorted(shuffled)
    assert curved_receipt == shuffled_receipt
    evaluation_a = verify_and_score(task, curved)
    evaluation_b = verify_and_score(task, shuffled)
    assert evaluation_a["whole_task_solved"] == evaluation_b["whole_task_solved"]
    assert evaluation_a["candidate_set_sha256"] == evaluation_b["candidate_set_sha256"]


def test_public_smoke_suite_has_full_matched_coverage() -> None:
    public, scored = run_suite(seed=23, roots=4, budget=8)
    assert len(public) == len(scored) == 4
    for item in public:
        assert "target" not in item
    for row in scored:
        assert set(row["arms"]) == set(ARMS)
        assert row["arms"]["rigid_top1_proxy"]["verifier_calls"] == 1
        for arm in ARMS:
            if arm != "rigid_top1_proxy":
                assert row["arms"][arm]["verifier_calls"] == 8
    report = summarize(scored)
    assert report["status"] == "untrained_synthetic_wiring_only"
    assert all(value["roots"] == 4 for value in report["results"].values())


def test_seed_replay_is_exact() -> None:
    _, first = run_suite(seed=5, roots=2, budget=4)
    _, second = run_suite(seed=5, roots=2, budget=4)
    assert first == second
