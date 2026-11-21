"""Independent correctness and custody checks for the K-depth research-taste gym."""
import hashlib
import json
from fractions import Fraction

import pytest

from datagoblin.training_gym.rlt_gym.hypergame_taste import (
    SPLITS,
    best_action,
    collect,
    fork_rows,
    generate,
    optimal_value,
    posterior,
    q_values,
    replay_archive,
    rollout,
)


def _world(k=1, split="train", seed=17):
    import random

    return generate(k, split, random.Random(seed), ordinal=1)


def _fixture_public():
    return {
        "prior_h1": "1/2", "base_sensor_fidelity": "3/4",
        "outer_endorsement": 1, "nested_views": [],
    }


def test_no_layers_exact_posterior():
    assert posterior(_fixture_public()) == Fraction(3, 4)


def test_nested_inversion_changes_inference_and_is_not_time_horizon():
    public = _fixture_public()
    public["nested_views"] = [
        {"inverts_earlier_endorsement": True, "fidelity": "1"},
        {"inverts_earlier_endorsement": False, "fidelity": "1"},
    ]
    assert posterior(public) == Fraction(1, 4)
    assert posterior(public, ignore_nested=True) == Fraction(3, 4)
    assert best_action(Fraction(1, 2), credits=2, turns=2) == "replicate"


def _enumerated_likelihood(base, layers, truth, final):
    # Fully independent branch enumeration over all channel outcomes.
    states = {truth: Fraction(1)}
    for accuracy, invert in [(base, False), *layers]:
        updates = {0: Fraction(0), 1: Fraction(0)}
        for bit, mass in states.items():
            proposed = bit ^ int(invert)
            updates[proposed] += mass * accuracy
            updates[1 - proposed] += mass * (1 - accuracy)
        states = updates
    return states[final]


@pytest.mark.parametrize("k", [0, 1, 2, 4, 8])
def test_exact_posterior_against_independent_channel_enumeration(k):
    task, _ = _world(k)
    public = task["public"]
    layers = [
        (Fraction(row["fidelity"]), row["inverts_earlier_endorsement"])
        for row in public["nested_views"]
    ]
    base, prior = Fraction(public["base_sensor_fidelity"]), Fraction(public["prior_h1"])
    final = public["outer_endorsement"]
    likelihood_one = _enumerated_likelihood(base, layers, 1, final)
    likelihood_zero = _enumerated_likelihood(base, layers, 0, final)
    expected = prior * likelihood_one / (
        prior * likelihood_one + (1 - prior) * likelihood_zero
    )
    assert posterior(public) == expected


def test_horizon_one_and_replicate_value_of_information():
    assert best_action(Fraction(1, 2), credits=2, turns=1) == "abstain"
    assert dict(q_values(Fraction(1, 2), credits=2, turns=2))["replicate"] == Fraction(11, 2)
    assert optimal_value(Fraction(1, 2), credits=2, turns=2) == Fraction(11, 2)


def test_perfect_knowledge_prefers_direct_publication():
    assert best_action(Fraction(1), credits=4, turns=3) == "submit_1"
    assert best_action(Fraction(0), credits=4, turns=3) == "submit_0"


def test_no_private_fields_in_policy_input_or_fork():
    task, sealed = _world(8)
    text = json.dumps(task["public"])
    assert "truth_h1" not in text
    assert "test_outcomes" not in text
    row = rollout(task, sealed, "proxy")
    forks = fork_rows(task, row)
    assert forks
    assert all("truth_h1" not in json.dumps(fork) for fork in forks)
    assert all(fork["split"] == "train" for fork in forks)


def test_oracle_rollout_has_zero_regret_at_every_visited_decision():
    for k in (0, 1, 2, 4, 8):
        task, sealed = _world(k)
        row = rollout(task, sealed, "bellman")
        assert all(Fraction(step["post_choice_regret"]) == 0 for step in row["events"])
        assert row["verifier_calls"] == len(row["events"])


def test_true_cost_and_public_reward_are_separate():
    task, sealed = _world(0)
    row = rollout(task, sealed, "forced_probe")
    assert row["total_research_cost"] >= 1
    assert row["realized_utility"] <= 10 - row["total_research_cost"]


def test_generator_rejects_unknown_split_or_depth():
    import random

    with pytest.raises(ValueError):
        generate(17, "train", random.Random(1), 1)
    with pytest.raises(ValueError):
        generate(1, "unknown", random.Random(1), 1)


def test_collection_replay_and_custody(tmp_path):
    a = collect(tmp_path / "a", depths=(0, 1, 2), per_cell=2, seed=29)
    b = collect(tmp_path / "b", depths=(0, 1, 2), per_cell=2, seed=29)
    assert a == b
    assert a["profile"] == "public_seed_smoke"
    assert a["max_train_presentations_per_root"] == 4
    assert replay_archive(tmp_path / "a")["verified_rollouts"] == 180
    assert a["artifacts"]["public_tasks.jsonl"]["rows"] == 3 * len(SPLITS) * 2
    assert a["artifacts"]["rollouts_scored.jsonl"]["rows"] == 3 * len(SPLITS) * 2 * 6
    for name, info in a["artifacts"].items():
        assert hashlib.sha256((tmp_path / "a" / name).read_bytes()).hexdigest() == info["sha256"]
    roots = {}
    for line in (tmp_path / "a" / "public_tasks.jsonl").read_text(encoding="utf8").splitlines():
        row = json.loads(line)
        assert row["root"] not in roots
        roots[row["root"]] = row["split"]
    per_root_views = {}
    for name in ("train_forks.jsonl", "train_teacher_requests.jsonl"):
        for line in (tmp_path / "a" / name).read_text(encoding="utf8").splitlines():
            row = json.loads(line)
            root = row.get("root", row.get("request_id"))
            assert roots[root] == "train"
            per_root_views[root] = per_root_views.get(root, 0) + 1
            assert "truth_h1" not in line
    assert per_root_views and max(per_root_views.values()) <= 4
    with pytest.raises(ValueError):
        collect(tmp_path / "a", depths=(0,), per_cell=1)


def test_private_evaluation_seed_changes_sealed_cohort(tmp_path):
    one = tmp_path / "seed-a"
    two = tmp_path / "seed-b"
    one.write_bytes(b"private random secret bytes 111")
    two.write_bytes(b"private random secret bytes 222")
    result_a = collect(tmp_path / "private-a", depths=(2,), per_cell=2, private_seed_file=one)
    result_b = collect(tmp_path / "private-b", depths=(2,), per_cell=2, private_seed_file=two)
    assert result_a["profile"] == "private_seeded"
    assert result_a["seed_commitment"] != result_b["seed_commitment"]
    assert result_a["artifacts"]["public_tasks.jsonl"]["sha256"] != result_b["artifacts"]["public_tasks.jsonl"]["sha256"]


def test_no_forks_for_all_optimal_actions():
    task, sealed = _world()
    row = rollout(task, sealed, "bellman")
    forks = fork_rows(task, row)
    assert all(Fraction(item["expected_regret"]) > 0 for item in forks)


def test_corrupt_rollout_is_detected_by_digest(tmp_path):
    folder = tmp_path / "archive"
    collect(folder, depths=(1,), per_cell=1)
    rows = folder / "rollouts_scored.jsonl"
    rows.write_bytes(rows.read_bytes().replace(b'"correct_publication":true', b'"correct_publication":false', 1))
    with pytest.raises(ValueError, match="archive hash mismatch"):
        replay_archive(folder)
