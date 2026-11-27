"""Persistent compact research state and intermediate verified capability scoring."""

from __future__ import annotations

import copy
from typing import Any

from datagoblin.research_v2.suite import LEARNERS, digest


def _claim_for(after: float, before: float) -> str:
    if abs(after - before) < 1e-12:
        return "ties"
    return "improves" if after > before else "worse"


def _short_error_code(message: str | None) -> str | None:
    if not message:
        return None
    low = message.lower()
    if "experiment unavailable" in low:
        return "experiment_unavailable"
    if "check unavailable" in low:
        return "check_unavailable"
    if "unknown or inapplicable learner config field" in low:
        return "invalid_config_field"
    if "optimizer update budget exceeded" in low:
        return "update_budget_exceeded"
    if "training cell budget exceeded" in low:
        return "cell_budget_exceeded"
    if "matched zero-update" in low:
        return "missing_zero_baseline"
    if "unknown action" in low:
        return "unknown_action"
    if "json" in low or "expecting" in low:
        return "malformed_json"
    return "action_error"


class CompactResearchState:
    """Bounded persistent state carried across turns inside the 2,048-token budget.

    Fixes v2.0's stateless observation amnesia without storing full raw chat
    transcripts: records compact turn outcomes, disallowed actions, verified
    symbolic answers, matched (baseline, trained) learner pairs, and legal next
    actions.
    """

    def __init__(self, family: str) -> None:
        self.family = family
        self.is_learner = family in LEARNERS
        self.turn_log: list[dict[str, Any]] = []
        self.disallowed_actions: set[str] = set()
        self.verified_answer: dict[str, Any] | None = None
        self.verified_turn: int | None = None
        self.last_candidate_answer: dict[str, Any] | None = None
        self.last_candidate_config: dict[str, Any] | None = None
        self.invalid_config_seen: bool = False
        self.valid_json_turns: int = 0
        self.schema_valid_turns: int = 0
        self.checks_passed: int = 0
        self.checks_failed: int = 0
        self.matched_pairs: list[dict[str, Any]] = []
        self.baseline_ids: list[str] = []
        self.trained_ids: list[str] = []

    def record_turn(
        self,
        *,
        turn: int,
        parsed_action: dict[str, Any] | None,
        result: dict[str, Any],
        artifacts: dict[str, dict[str, Any]],
        calibrations: dict[str, dict[str, Any]],
    ) -> None:
        act_name = None
        if isinstance(parsed_action, dict):
            self.valid_json_turns += 1
            raw_act = parsed_action.get("action")
            if isinstance(raw_act, str):
                act_name = raw_act
            if isinstance(parsed_action.get("answer"), dict) and parsed_action["answer"]:
                self.last_candidate_answer = copy.deepcopy(parsed_action["answer"])
            if isinstance(parsed_action.get("config"), dict) and parsed_action["config"]:
                self.last_candidate_config = copy.deepcopy(parsed_action["config"])

        err_msg = result.get("error")
        err_code = _short_error_code(err_msg)
        if err_code is None and act_name in {"run_experiment", "check", "submit", "inspect"}:
            self.schema_valid_turns += 1
        if err_code == "experiment_unavailable":
            self.disallowed_actions.add("run_experiment")
        elif err_code == "check_unavailable":
            self.disallowed_actions.add("check")
        elif err_code == "invalid_config_field":
            self.invalid_config_seen = True

        chk = result.get("check_passed")
        if chk is True:
            self.checks_passed += 1
            if isinstance(parsed_action, dict) and isinstance(parsed_action.get("answer"), dict):
                self.verified_answer = copy.deepcopy(parsed_action["answer"])
                self.verified_turn = turn
        elif chk is False:
            self.checks_failed += 1

        entry: dict[str, Any] = {
            "t": turn,
            "act": act_name or "invalid",
            "ok": err_code is None,
        }
        if err_code is not None:
            entry["err"] = err_code
        if chk is not None:
            entry["chk"] = bool(chk)
        if "created" in result:
            entry["art"] = result["created"]
        if "status" in result:
            entry["status"] = result["status"]
        self.turn_log.append(entry)
        self.turn_log = self.turn_log[-8:]

        # Refresh artifact pairing summary from environment-owned artifacts
        self.baseline_ids = []
        self.trained_ids = []
        for aid, art in artifacts.items():
            if art["config"]["steps"] == 0:
                self.baseline_ids.append(aid)
            else:
                self.trained_ids.append(aid)
        pairs: list[dict[str, Any]] = []
        for tid in self.trained_ids:
            t_art = artifacts[tid]
            t_cfg = {k: v for k, v in t_art["config"].items() if k != "steps"}
            for bid in self.baseline_ids:
                b_art = artifacts[bid]
                b_cfg = {k: v for k, v in b_art["config"].items() if k != "steps"}
                if t_cfg == b_cfg and t_art["receipt"]["init_weights_sha256"] == b_art["receipt"]["weights_sha256"]:
                    t_acc = calibrations[tid]["accuracy"]
                    b_acc = calibrations[bid]["accuracy"]
                    pairs.append(
                        {
                            "artifact_id": tid,
                            "baseline_id": bid,
                            "claim": _claim_for(t_acc, b_acc),
                            "cal_delta": round(float(t_acc - b_acc), 6),
                        }
                    )
        self.matched_pairs = pairs

    def next_valid_actions(self, remaining: dict[str, int]) -> list[str]:
        if self.verified_answer is not None:
            return ["submit"]
        if self.is_learner:
            actions: list[str] = []
            if remaining.get("experiments", 0) > 0:
                actions.append("run_experiment")
            if self.matched_pairs:
                actions.append("submit")
            return actions or ["submit"]
        actions = []
        if remaining.get("checks", 0) > 0 and "check" not in self.disallowed_actions:
            actions.append("check")
        actions.append("submit")
        return actions

    def to_public_dict(self, remaining: dict[str, int]) -> dict[str, Any]:
        """Serialize compact bounded state for inclusion in model observations."""
        out: dict[str, Any] = {
            "family_kind": "learner" if self.is_learner else "symbolic",
            "turn_log": self.turn_log[-4:],
            "disallowed_actions": sorted(self.disallowed_actions),
            "next_valid_actions": self.next_valid_actions(remaining),
            "verified_ready": self.verified_answer is not None,
        }
        if self.verified_answer is not None:
            out["verified_turn"] = self.verified_turn
            out["verified_answer_sha256"] = digest(self.verified_answer)[:16]
        if self.is_learner:
            out["baseline_ids"] = self.baseline_ids
            out["trained_ids"] = self.trained_ids
            out["matched_pairs"] = self.matched_pairs
        return out


def compute_capability_score(
    *,
    family: str,
    state: CompactResearchState,
    artifacts: dict[str, dict[str, Any]],
    final_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute transparent intermediate verified capability score in [0.0, 1.0].

    Always reported separately from strict terminal primary_score and success.
    """
    valid_json = state.valid_json_turns > 0
    schema_valid = state.schema_valid_turns > 0
    submitted = final_result is not None
    primary = float(final_result["primary_score"]) if final_result else 0.0

    if family not in LEARNERS:
        check_passed = state.checks_passed > 0 or primary >= 1.0
        terminal_verified = primary >= 1.0
        components = {
            "valid_json_turn": 0.15 if valid_json else 0.0,
            "schema_valid_action": 0.15 if schema_valid else 0.0,
            "verifier_check_passed": 0.35 if check_passed else 0.0,
            "terminal_submitted": 0.15 if submitted else 0.0,
            "terminal_verified": 0.20 if terminal_verified else 0.0,
        }
    else:
        real_fit = any(
            a["config"]["steps"] > 0
            and a["receipt"]["weights_sha256"] != a["receipt"]["init_weights_sha256"]
            and a["receipt"]["loss_after"] < a["receipt"]["loss_before"]
            for a in artifacts.values()
        )
        matched_pair = len(state.matched_pairs) > 0
        components = {
            "valid_json_turn": 0.10 if valid_json else 0.0,
            "schema_valid_config": 0.15 if len(artifacts) > 0 else 0.0,
            "real_optimizer_fit": 0.20 if real_fit else 0.0,
            "matched_zero_baseline": 0.15 if matched_pair else 0.0,
            "terminal_submitted": 0.15 if submitted else 0.0,
            "integrity_and_generalization": round(0.25 * max(0.0, min(1.0, primary)), 6),
        }
    score = round(min(1.0, sum(components.values())), 6)
    return {
        "capability_score": score,
        "capability_components": components,
        "milestones": {
            "valid_json": valid_json,
            "schema_valid": schema_valid,
            "verified_check_passed": state.checks_passed > 0,
            "real_optimizer_fit": (
                any(
                    a["config"]["steps"] > 0
                    and a["receipt"]["weights_sha256"] != a["receipt"]["init_weights_sha256"]
                    for a in artifacts.values()
                )
                if family in LEARNERS
                else False
            ),
            "matched_zero_baseline": len(state.matched_pairs) > 0 if family in LEARNERS else False,
            "submitted": submitted,
        },
    }
