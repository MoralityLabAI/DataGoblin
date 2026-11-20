import numpy as np
import pytest


def test_text_attention_is_causal_at_every_visit():
    pytest.importorskip("autograd")
    from datagoblin.training_gym.rlt_pretrain_cell import ARMS, initialize, text_visits

    params = initialize(17, width=8, lean_features=5)
    mask = np.ones((1, 4), dtype=bool)
    left = np.array([[65, 66, 67, 68]])
    right = np.array([[65, 66, 70, 71]])
    for arm in ARMS:
        a = list(text_visits(params, left, mask, arm, 4))
        b = list(text_visits(params, right, mask, arm, 4))
        for first, second in zip(a, b, strict=True):
            np.testing.assert_allclose(first[:, :2], second[:, :2], atol=1e-12)


def test_answer_interpolation_keeps_first_reasoning_step_and_honors_stop():
    pytest.importorskip("autograd")
    from datagoblin.training_gym.rlt_pretrain_cell import initialize, pooled_links, refine

    params = initialize(23, width=8, lean_features=5)
    mask = np.array([[True, False, True]])
    encoded = np.ones((1, 3, 8))
    links = pooled_links(mask)
    direct = list(refine(params, encoded, links, mask, "direct", 4))
    rectified = list(refine(
        params, encoded, links, mask, "answer_rlt", 4, stop_when=lambda visit, answer: visit == 2,
    ))
    assert len(rectified) == 2
    np.testing.assert_array_equal(direct[0][0], rectified[0][0])
    np.testing.assert_allclose(direct[0][1] / 4, rectified[0][1])
    np.testing.assert_array_equal(rectified[1][1][:, 1], 0)


def test_mixed_training_checkpoint_continues_bitwise(tmp_path):
    pytest.importorskip("autograd")
    from datagoblin.serialization import load_structured
    from datagoblin.training_gym.rlt_pretrain_fit import resume_probe

    rng = np.random.default_rng(31)
    text = (np.array([[2, 3, 4], [3, 4, 5], [4, 5, 6]]), np.array([[3, 4, 5], [4, 5, 6], [5, 6, 7]]))
    proofs = (rng.normal(size=(4, 2, 3)), np.ones((4, 2), bool), np.tile([1, 0], (4, 1)))
    x = rng.normal(size=(3, 4, 3))
    edges = np.repeat(np.eye(4)[None], 3, axis=0)
    routes = (x, edges, x + 0.1, edges, np.array([1, 2, 3]), np.array([2, 1, 3]))
    resume_probe(tmp_path, (text, proofs, routes), lambda: None)
    assert load_structured(tmp_path / "resume-audit.json")["identical_weight_tensors"]


def test_candidate_loss_masks_unknown_unprovable_pool():
    pytest.importorskip("autograd")
    from datagoblin.training_gym.rlt_pretrain_fit import lean_loss

    present = np.array([[True, True, False]])
    unknown = np.array([[0, -1, -1]])
    known = np.array([[0, 0, -1]])
    scores = np.array([[1.0, 2.0, 1000.0]])
    assert lean_loss(scores, present, unknown) == 0.0
    assert lean_loss(scores, present, known) > 0


def test_curriculum_changes_order_of_emphasis_at_the_same_exposure():
    pytest.importorskip("autograd")
    from datagoblin.training_gym.rlt_pretrain_fit import RECIPES, SPEC, recipe_weights

    steps = SPEC["steps"]
    plan = [recipe_weights("text_first", step, steps) for step in range(1, steps + 1)]
    np.testing.assert_allclose(np.mean(plan, axis=0), RECIPES["mixed"]["phases"][0])
    assert plan[0][0] > plan[-1][0]
    baseline = RECIPES["mixed"]["batch_examples"]
    heavier = RECIPES["text_heavier"]["batch_examples"]
    assert sum(heavier) == sum(baseline) and heavier[-1] == baseline[-1]


def test_layout_separates_shared_weights_and_executed_recurrence():
    from datagoblin.training_gym.rlt_pretrain_scale import Layout, report

    shallow = report(Layout(), 2, 1)
    deep = report(Layout(), 6, 6)
    assert deep["unique_parameters"] == shallow["unique_parameters"] == 999_352_320
    assert deep["executed_transformer_blocks"] == 74
    assert shallow["executed_transformer_blocks"] == 24
    assert deep["bf16_parameter_bytes"] == 2 * deep["unique_parameters"]


def test_bundle_checks_epoch_and_cost_for_reused_certificate():
    from datagoblin.training_gym.rlt_pretrain_export import audit_trace

    terminal = {"prediction": "b", "visits": 2, "checks": 0, "checker_hops": 0, "attempts": []}
    before = {"prediction": "b", "visits": 1, "checks": 1, "checker_hops": 1, "attempts": [[1, "b", 0, True]]}
    reuse = {"prediction": "b", "visits": 0, "checks": 1, "checker_hops": 0, "attempts": [[0, "b", 1, True]]}
    accuracy = {"before": True, "after": True}
    row = {"id": "route", "lineage": "g", "no_op": True, "policies": {
        "terminal": {"before": terminal, "after": terminal, "accuracy": accuracy},
        "verified_reuse": {"before": before, "after": reuse, "accuracy": accuracy},
    }}
    task = {"lineage": "g", "start": "a", "steps": 1, "links": [["a", "b"], ["b", "a"]],
            "patch": {"source": "a", "destination": "b"}}
    stats = {"goals": 1, "joint_correct": 1, "before_correct": 1, "after_correct": 1}
    lean = dict.fromkeys(("goals", "proved", "visits", "verifier_replay_checks",
                         "projected_tactic_requests", "projected_kernel_requests", "unknown_attempts"), 0)
    detail = {"depth": 2, "lean": [], "text_windows": [], "revision": {"held": [row]}, "summary": {
        "text": {}, "lean": lean,
        "revision": {"held": {
            "terminal": {"no_op": dict(stats, visits=4, checks=0, checker_hops=0)},
            "verified_reuse": {"no_op": dict(stats, visits=1, checks=2, checker_hops=1)},
        }},
    }}
    graphs = {"held": {"route": (task, {"before": "b", "after": "b"})}}
    protocol = {"rule_policies": ["terminal", "verified_reuse"]}
    audit_trace(detail, {}, {}, graphs, protocol)
    reuse["checker_hops"] = 1
    with pytest.raises(ValueError, match="verifier work"):
        audit_trace(detail, {}, {}, graphs, protocol)
    reuse["checker_hops"] = 0
    reuse["attempts"][0][2] = 0
    with pytest.raises(ValueError, match="unsupported commit"):
        audit_trace(detail, {}, {}, graphs, protocol)
