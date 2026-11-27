"""Unit tests for Research Verifiers v2.1: compact state, controllers, exact BPTT, and persona reference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from datagoblin.research_v2 import suite as v2_suite
from datagoblin.research_v2_1 import suite as v21_suite
from datagoblin.research_v2_1.controllers import (
    CONTROLLER_MODES,
    ResearchController,
    RLTStateCorrector,
    TRMActionRouter,
    _softmax,
)
from datagoblin.research_v2_1.environment import ResearchSessionV21
from datagoblin.research_v2_1.persona_store_reference import project_persona_store, reference_persona_memory

SCRIPTS_V21 = Path(__file__).resolve().parents[2] / "scripts" / "research_v2_1"
if str(SCRIPTS_V21) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_V21))

from baby_goblin_emergence_probe import recompute_emergence_milestones  # noqa: E402
from jinn_beast_depth_study import verify_deep_exact_bptt_gradients  # noqa: E402
from jinn_beast_disentangle import verify_exact_unrolled_gradient  # noqa: E402


def test_v20_implementation_digest_preserved_and_v21_distinct() -> None:
    assert v2_suite.implementation_digest() == "73ac4d4d57825c708f45cf73aa56a516290f825425996146131bab181891a339"
    assert v21_suite.implementation_digest() != v2_suite.implementation_digest()


def test_independent_persona_reference_semantics_and_tombstone_resurrection() -> None:
    events = [
        {"actor": "u1", "scope": "research", "revision": 1, "op": "set", "key": "tone", "value": "warm"},
        {"actor": "u1", "scope": "research", "revision": 2, "op": "set", "key": "verbosity", "value": "brief"},
        {"actor": "u1", "scope": "research", "revision": 3, "op": "revoke", "key": "verbosity"},
        {"actor": "u2", "scope": "research", "revision": 4, "op": "set", "key": "tone", "value": "hostile"},
        {"actor": "u1", "scope": "fiction", "revision": 4, "op": "set", "key": "tone", "value": "theatrical"},
        {"actor": "u1", "scope": "research", "revision": 5, "op": "set", "key": "verbosity", "value": "concise"},
        {"actor": "u1", "scope": "research", "revision": 6, "op": "set", "key": "tone", "value": "future"},
    ]
    # At revision 3: verbosity is tombstoned, future rev 5..6 excluded
    p3 = project_persona_store(list(reversed(events)), owner="u1", scope="research", at_revision=3)
    assert p3.active == {"tone": "warm"}
    assert p3.tombstones == {"verbosity": 3}
    assert p3.rejected_cross_owner == 1
    assert p3.rejected_cross_scope == 1
    assert p3.rejected_future_revision == 2

    # At revision 5: verbosity resurrected cleanly after tombstone
    p5 = project_persona_store(list(reversed(events)), owner="u1", scope="research", at_revision=5)
    assert p5.active == {"tone": "warm", "verbosity": "concise"}
    assert p5.tombstones == {}

    # Differential check against v2.0 persona_memory across generated persona tasks
    cases = v21_suite.make_suite(seed=20261127, per_family=2, split="development")
    for c in cases:
        if c["family"] == "persona_externalization":
            assert reference_persona_memory(c["public"]) == v2_suite.persona_memory(c["public"])


def test_session_v21_compact_state_and_intermediate_capability_score() -> None:
    cases = v21_suite.make_suite(seed=20261127, per_family=1, split="development")
    cur_case = next(c for c in cases if c["family"] == "cursor_recovery")
    gold = v21_suite.reference_answer(cur_case)

    sess = ResearchSessionV21(cur_case)
    obs1 = sess.step_json(json.dumps({"action": "run_experiment", "config": {"steps": 1}}))
    assert "run_experiment" in obs1["compact_state"]["disallowed_actions"]
    assert obs1["compact_state"]["verified_ready"] is False

    obs2 = sess.step_json(json.dumps({"action": "check", "answer": gold}))
    assert obs2["check_passed"] is True
    assert obs2["compact_state"]["verified_ready"] is True
    assert obs2["compact_state"]["next_valid_actions"] == ["submit"]

    # Without submitting: primary_score=0.0, but intermediate capability_score=0.65 (valid_json+schema+check_passed)
    mid = sess.result()
    assert mid["primary_score"] == 0.0
    assert mid["capability_score"] == 0.65

    sess.step_json(json.dumps({"action": "submit", "answer": gold}))
    fin = sess.result()
    assert fin["primary_score"] == 1.0
    assert fin["capability_score"] == 1.0
    assert fin["success"] is True


def test_controllers_stop_after_check_and_repair_learner_configs() -> None:
    cases = v21_suite.make_suite(seed=20261127, per_family=1, split="development")
    sym_case = next(c for c in cases if c["family"] == "boolean_repair")
    gold = v21_suite.reference_answer(sym_case)

    for mode in CONTROLLER_MODES:
        ctrl = ResearchController(mode)
        sess = ResearchSessionV21(sym_case)
        sess.step_json(json.dumps({"action": "check", "answer": gold}))
        pre = ctrl.pre_turn_decision(case=sym_case, state=sess.compact_state, turn_index=1)
        if mode == "autonomous":
            assert pre.intervened is False
        else:
            assert pre.intervened is True
            assert pre.skip_model_call is True
            assert json.loads(pre.executed_text or "{}")["action"] == "submit"

    # TRM & RLT controllers sanitize extra learner keys (e.g., Qwen3's trm_width/train_seed)
    trm_case = next(c for c in cases if c["family"] == "trm_training")
    for mode in ("trm_action_routing", "rlt_state_correction"):
        ctrl = ResearchController(mode)
        sess = ResearchSessionV21(trm_case)
        bad = json.dumps(
            {
                "action": "run_experiment",
                "config": {"steps": 16, "train_examples": 32, "learning_rate": 0.03, "trm_width": 8, "train_seed": 99},
            }
        )
        post = ctrl.post_model_decision(
            case=trm_case,
            state=sess.compact_state,
            turn_index=0,
            raw_text=bad,
            artifacts=sess._artifacts,
            remaining=sess._remaining(),
        )
        assert post.intervened is True
        repaired = json.loads(post.executed_text or "{}")
        assert repaired["action"] == "run_experiment"
        assert "trm_width" not in repaired["config"]
        assert "train_seed" not in repaired["config"]
        assert repaired["config"]["width"] == 8

    # TRM & RLT controllers map ldt_critical_veto -> critical_veto on LDT/persona families
    ldt_case = next(c for c in cases if c["family"] == "persona_externalization")
    for mode in ("trm_action_routing", "rlt_state_correction"):
        ctrl = ResearchController(mode)
        sess = ResearchSessionV21(ldt_case)
        bad_ldt = json.dumps(
            {
                "action": "run_experiment",
                "config": {"steps": 64, "train_examples": 64, "learning_rate": 0.05, "ldt_critical_veto": True},
            }
        )
        post = ctrl.post_model_decision(
            case=ldt_case,
            state=sess.compact_state,
            turn_index=0,
            raw_text=bad_ldt,
            artifacts=sess._artifacts,
            remaining=sess._remaining(),
        )
        assert post.intervened is True
        repaired = json.loads(post.executed_text or "{}")
        assert repaired["config"]["critical_veto"] is True
        assert "ldt_critical_veto" not in repaired["config"]


def test_exact_unrolled_bptt_gradients_match_central_differences() -> None:
    # 1. Verify Jinn/Beast CorrectorHead exact BPTT at K=1, 3, 6
    jb_grad = verify_exact_unrolled_gradient()
    for val in jb_grad.values():
        assert val < 1e-5

    # 2. Verify RLTStateCorrector exact BPTT at K=3
    rng = np.random.default_rng(77)
    x = rng.random((12, 14))
    y = rng.integers(0, 5, size=12)
    router = TRMActionRouter(seed=77)
    rlt = RLTStateCorrector(router, passes=3, seed=78)
    bl = rng.normal(0.0, 1.0, size=(12, 5))

    def loss_fn() -> float:
        logits, _ = rlt.forward_from_logits(bl, x)
        probs = _softmax(logits)
        return float(-np.log(np.clip(probs[np.arange(12), y], 1e-12, 1.0)).mean())

    logits, cache = rlt.forward_from_logits(bl, x)
    probs = _softmax(logits)
    dlog = probs.copy()
    dlog[np.arange(12), y] -= 1.0
    dlog /= 12
    _, gw1, gb1, gw2, gb2 = rlt.backward_unrolled(dlog, cache)

    eps = 1e-6
    max_rel = 0.0
    for param, grad in ((rlt.w1, gw1), (rlt.b1, gb1), (rlt.w2, gw2), (rlt.b2, gb2)):
        fp, fg = param.ravel(), grad.ravel()
        for idx in range(min(len(fp), 20)):
            orig = fp[idx]
            fp[idx] = orig + eps
            lp = loss_fn()
            fp[idx] = orig - eps
            lm = loss_fn()
            fp[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = fg[idx]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            max_rel = max(max_rel, float(rel))
    assert max_rel < 1e-5


def test_deep_recurrence_exact_bptt_up_to_24_passes_and_untied_controls() -> None:
    grad_errs = verify_deep_exact_bptt_gradients()
    for key in (
        "tied_K=1_max_rel_err",
        "tied_K=3_max_rel_err",
        "tied_K=6_max_rel_err",
        "tied_K=12_max_rel_err",
        "tied_K=24_max_rel_err",
        "untied_depth=6_max_rel_err",
        "untied_depth=12_max_rel_err",
    ):
        assert key in grad_errs
        assert grad_errs[key] < 1e-5


def test_baby_goblin_emergence_ledger_and_milestone_tracker() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ledger_path = repo_root / "configs" / "research_v2_1" / "baby_goblin_emergence_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    # At observed 10M stage, all emergence milestones remain strictly None
    checked = recompute_emergence_milestones(json.loads(json.dumps(ledger)))
    assert all(v is None for v in checked["emergence_milestones"].values())

    # Simulate 50M onset & 100M reliable crossing without mutating disk ledger
    sim = json.loads(json.dumps(ledger))
    for st in sim["stages"]:
        if st["stage_id"] == "50M":
            st["status"] = "evaluated"
            st["arms"]["spec_replay_50M"] = {
                "valid_json_rate": 0.50,
                "schema_valid_action_rate": 0.375,
                "eos_stop_rate": 0.3125,
                "action_verb_accuracy": 0.625,
                "autonomous_mean_capability_score": 0.25,
                "rlt_controller_mean_capability_score": 0.3125,
                "autonomous_mean_primary_score": 0.0,
                "rlt_controller_mean_primary_score": 0.0,
            }
        elif st["stage_id"] == "100M":
            st["status"] = "evaluated"
            st["arms"]["spec_replay_100M"] = {
                "valid_json_rate": 0.875,
                "schema_valid_action_rate": 0.8125,
                "eos_stop_rate": 0.875,
                "action_verb_accuracy": 0.75,
                "autonomous_mean_capability_score": 0.50,
                "rlt_controller_mean_capability_score": 0.625,
                "autonomous_mean_primary_score": 0.0625,
                "rlt_controller_mean_primary_score": 0.125,
            }
    recompute_emergence_milestones(sim)
    ms = sim["emergence_milestones"]
    assert ms["first_json_onset"]["stage_id"] == "50M"
    assert ms["first_eos_onset"]["stage_id"] == "50M"
    assert ms["first_action_selection"]["stage_id"] == "50M"
    assert ms["first_verified_capability"]["stage_id"] == "50M"
    assert ms["first_json_reliable"]["stage_id"] == "100M"
    assert ms["first_eos_reliable"]["stage_id"] == "100M"
    assert ms["first_nonzero_primary_score"]["stage_id"] == "100M"
