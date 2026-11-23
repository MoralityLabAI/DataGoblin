from __future__ import annotations

import numpy as np
import pytest

from datagoblin.training_gym.sgrt_history import (
    CHOICES,
    EVENTS,
    FAMILIES,
    TRAINED,
    D,
    HistoryHead,
    _digest,
    _prediction,
    evaluate,
    event_transport,
    forward,
    gradients,
    make_roots,
    paired_contrasts,
    train,
    transport_diagnostics,
)


def test_train_eval_lineage_and_composition_holdout() -> None:
    training = make_roots(split="train", seed=31003, per_family=24)
    evaluation = make_roots(split="eval", seed=61003, per_family=12)
    assert len(training) == 72
    assert len(evaluation) == 48
    assert not set(x.root_id for x in training) & set(x.root_id for x in evaluation)
    assert {r.family for r in training} == set(FAMILIES[:3])
    assert {r.family for r in evaluation} == set(FAMILIES)
    train_events = {x for r in training for x in (r.event_a, r.event_b)}
    assert train_events == set(EVENTS)
    train_pairs = {frozenset((r.event_a, r.event_b)) for r in training}
    heldout_pairs = {
        frozenset((r.event_a, r.event_b))
        for r in evaluation if r.family == "heldout_mixed"
    }
    assert not train_pairs & heldout_pairs
    assert make_roots(split="eval", seed=61003, per_family=12) == evaluation


@pytest.mark.parametrize("reverse", [False, True])
def test_executable_two_step_storyworld_and_private_targets(reverse: bool) -> None:
    root = make_roots(split="eval", seed=61003, per_family=1)[0]
    ep = root.episode(reverse)
    payload = ep.public()
    assert payload["candidate_states"] == list(CHOICES)
    assert len(payload["event_stream"]) == 2
    assert payload["event_stream"][0]["step"] == 1
    assert "truth" not in payload
    assert "target" not in payload
    assert set(CHOICES) == set(range(16))
    assert 0 <= root.truth(reverse)[0] < 16
    assert 0 <= root.truth(reverse)[1] < 16
    assert root.truth()[1] != root.truth(reverse=True)[1]


def test_closed_curve_and_matched_noncommuting_spectrum() -> None:
    diagnostic = transport_diagnostics()
    assert diagnostic["flat_commutator_fro_max"] < 1e-9
    assert diagnostic["curved_commutator_fro_max"] > 1e-5
    assert diagnostic["flat_curved_trace_mismatch_max"] < 1e-8
    assert diagnostic["orthogonality_error_max"] < 1e-9
    for op in EVENTS:
        curved = event_transport("curved", op)
        flat = event_transport("flat", op)
        random = event_transport("random", op)
        assert curved.shape == flat.shape == random.shape == (D, D)
        assert np.trace(curved) == pytest.approx(np.trace(flat), abs=1e-8)
        assert np.trace(random) == pytest.approx(np.trace(flat), abs=1e-8)
        assert np.linalg.det(curved) == pytest.approx(1, abs=1e-8)
        assert np.linalg.det(flat) == pytest.approx(1, abs=1e-8)


def test_bag_policy_has_exact_order_alias() -> None:
    root = make_roots(split="train", seed=31003, per_family=1)[0]
    model = HistoryHead.initialize(seed=7)
    logits_ab, _, _ = forward(model, root.episode(), mode="bag")
    logits_ba, _, _ = forward(model, root.episode(reverse=True), mode="bag")
    assert np.allclose(logits_ab[1], logits_ba[1], atol=1e-12)
    assert _prediction(model, root.episode(), "bag")[1] == _prediction(
        model, root.episode(reverse=True), "bag"
    )[1]


@pytest.mark.parametrize("mode", ["bag", "curved", "random"])
def test_training_gradient_agrees_with_finite_difference(mode: str) -> None:
    root = make_roots(split="train", seed=31003, per_family=1)[0]
    model = HistoryHead.initialize(seed=11)
    ep = root.episode()
    target = root.truth()
    _loss, grads = gradients(model, ep, target, mode=mode)
    indices = (
        (root.initial_state, 0),
        (0, 0),
        (EVENTS.index(ep.events[0]), 0),
        (0, 0),
        (0,),
        (EVENTS.index(ep.events[0]),),
    )
    for array, grad, index in zip(model.weights, grads, indices, strict=True):
        value = float(array[index])
        eps = 1e-5
        array[index] = value + eps
        hi = gradients(model, ep, target, mode=mode)[0]
        array[index] = value - eps
        lo = gradients(model, ep, target, mode=mode)[0]
        array[index] = value
        assert grad[index] == pytest.approx((hi - lo) / (2 * eps), abs=2e-6)


def test_learned_pairwise_policy_and_scoring_custody() -> None:
    training = make_roots(split="train", seed=31003, per_family=4)
    evaluation = make_roots(split="eval", seed=61003, per_family=2)
    models = {}
    initial_sha = HistoryHead.initialize(41).digest()
    for mode in TRAINED:
        model, log = train(mode, roots=training, seed=41, updates=16)
        assert log["trainable_parameters"] == 937
        assert log["initial32_loss"] > 0
        assert model.digest() != initial_sha
        models[mode] = model
    rows, scores = evaluate(models, roots=evaluation)
    assert len(rows) == 8
    assert scores["all"]["exact_oracle"]["last_correct"] == 16
    assert scores["all"]["exact_oracle"]["both_orders_correct"] == 8
    assert scores["all"]["bag"]["both_orders_correct"] == 0
    assert scores["all"]["exact_oracle"]["verifier_calls"] == 32
    assert scores["heldout_mixed"]["exact_oracle"]["roots"] == 2
    observed_sets = {item["candidate_pool_sha256"] for r in rows for item in r["arms"].values()}
    assert observed_sets == {_digest(CHOICES)}
    assert all(r["truth"][0][1] != r["truth"][1][1] for r in rows)
    assert all(row["evaluator_only"] for row in rows)
    assert paired_contrasts(rows) == paired_contrasts(rows)


def test_deterministic_fit_and_transport_swap_does_not_change_public_event_stream() -> None:
    roots = make_roots(split="train", seed=31003, per_family=2)
    left, _ = train("curved", roots=roots, seed=7, updates=16)
    right, _ = train("curved", roots=roots, seed=7, updates=16)
    assert left.digest() == right.digest()
    ep = roots[0].episode()
    original = ep.public()
    forward(left, ep, mode="curved", swap_transport=True)
    assert original == ep.public()
