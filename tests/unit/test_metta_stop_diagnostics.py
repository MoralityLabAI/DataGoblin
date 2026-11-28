"""Stop-versus-generation harness and student variants."""

from __future__ import annotations

import numpy as np
import pytest

from MeTTa.airis02_student import RecurrentStudent, _matrix
from MeTTa.ram_hillclimb import make_bundle
from MeTTa.sft_oracle import next_edit
from MeTTa.stop_diagnostics import (
    diagnose,
    fresh_dev_rows,
    optimum_of,
    run_episode,
)
from MeTTa.student_variants import VARIANTS, fit_variant


class PerfectPolicy:
    """Always plays the canonical edit and stops exactly at the optimum."""

    def edit_action(self, task, selected, ctx):
        action = next_edit(selected, self.opt)
        return (action if action["op"] != "stop" else {"op": "remove", "id": selected[0]}), 0, 0

    def own_action(self, task, selected, ctx):
        return next_edit(selected, self.opt), 0, 0

    def wants_stop(self, task, selected, ctx):
        return selected == self.opt

    opt: list[str] = []


def test_fresh_roots_are_disjoint_from_bundle_seed_bands() -> None:
    bundle = make_bundle(24, 8)
    fresh = fresh_dev_rows(12)
    assert len(fresh) == 60
    assert {r["root"] for r in fresh}.isdisjoint(
        {r["root"] for r in bundle["train"] + bundle["dev"]}
        | set(bundle["public_test_root_names_only"])
    )
    with pytest.raises(ValueError):
        fresh_dev_rows(2, start=30000)


def test_plain_variant_reproduces_recurrent_student() -> None:
    train = make_bundle(3, 1)["train"]
    x, y, masks = _matrix(train)
    reference = RecurrentStudent(7, hidden=12)
    reference.fit(x, y, steps=8, lr=0.2, train_loops=4, legal_masks=masks)
    policy = fit_variant(VARIANTS["plain_class_stop"], train, seed=7, steps=8)
    net = policy.net
    for a, b in zip((reference.wx, reference.wh, reference.wo, reference.bx, reference.bo),
                    (net.wx, net.wh, net.wo, net.bx, net.bo), strict=True):
        assert np.allclose(a, b)


def test_harness_arms_agree_on_a_perfect_editor() -> None:
    rows = fresh_dev_rows(2)[:6]
    for row in rows:
        policy = PerfectPolicy()
        policy.opt = optimum_of(row)
        never = run_episode(policy, row, "never")
        oracle = run_episode(policy, row, "oracle")
        own = run_episode(policy, row, "own")
        assert never["ever_solved"] and oracle["final_solved"]
        assert own["stopped_at_solved"] and not own["premature"]
        assert all(s["match"] for s in oracle["steps"])


def test_diagnose_flags_generation_failure_independently_of_stopping() -> None:
    bundle = make_bundle(3, 1)
    rows = fresh_dev_rows(2)
    policy = fit_variant(VARIANTS["sep_stop_head"], bundle["train"], seed=3, steps=4)
    result = diagnose(policy, rows)
    assert result["cases"] == len(rows)
    assert result["solved_and_stopped"] <= result["solved_oracle_stop"] <= result["ever_reached_optimum"]


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_every_variant_trains_and_acts_legally(name: str) -> None:
    bundle = make_bundle(3, 1)
    policy = fit_variant(VARIANTS[name], bundle["train"], seed=5, steps=6)
    row = fresh_dev_rows(1)[0]
    from MeTTa.stop_diagnostics import initial_context

    ctx = initial_context(row["selected"])
    action, _, _ = policy.edit_action(row["public_world"], row["selected"], ctx)
    assert action["op"] in ("add", "remove", "abstain")
    own, _, _ = policy.own_action(row["public_world"], row["selected"], ctx)
    assert own["op"] in ("add", "remove", "stop", "abstain")
