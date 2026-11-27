"""v2.2 controller ablations built on the frozen v2.1 execution contract."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from datagoblin.research_v2.suite import LEARNERS, canon, digest
from datagoblin.research_v2_1.controllers import (
    CONTROLLER_MODES as V21_MODES,
)
from datagoblin.research_v2_1.controllers import (
    ControllerDecision,
    _try_parse_json_obj,
)
from datagoblin.research_v2_1.controllers import (
    ResearchController as V21Controller,
)

CONTROLLER_MODES = (*V21_MODES, "schema_only", "rule_action_routing",
                    "selective_rlt", "trm_no_persona", "rlt_no_persona",
                    "rule_no_persona")


def rule_class(feat: np.ndarray) -> int:
    """Exact routing-label rule from v2.1's synthetic training generator."""
    (learner, verified, pair, trained, _base, _turn, _exp, check_err,
     _invalid, _valid, run, check, submit, extra) = feat
    if not learner and verified and not submit:
        return 1
    if learner and pair and not submit:
        return 2
    if learner and (extra or (trained and (run or check)) or (not pair and check)):
        return 3
    if not learner and (run or (check_err and check)):
        return 4
    return 0


class RuleRouter:
    def predict(self, feat: np.ndarray) -> tuple[int, np.ndarray]:
        label = rule_class(feat)
        return label, np.eye(5, dtype=float)[label]


class SelectiveCorrector:
    """Use the already-trained v2.1 RLT only on low-margin TRM predictions."""

    def __init__(self, router: Any, corrector: Any, threshold: float = 0.20):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold outside [0,1]")
        self.router = router
        self.corrector = corrector
        self.threshold = threshold
        self.calls = 0
        self.corrected_calls = 0

    def predict(self, feat: np.ndarray) -> tuple[int, np.ndarray]:
        label, probs = self.router.predict(feat)
        self.calls += 1
        top = np.sort(probs)[-2:]
        if top[1] - top[0] < self.threshold:
            self.corrected_calls += 1
            # The native corrector's predict recomputes the TRM; count that cost.
            return self.corrector.predict(feat)
        return label, probs

    def accounting(self) -> dict[str, int]:
        return {"predictions": self.calls, "rlt_calls": self.corrected_calls,
                "trm_forwards": self.calls + self.corrected_calls,
                "rlt_passes": self.corrected_calls * self.corrector.passes}


def schema_repair(case: dict[str, Any], raw_text: str,
                  remaining: dict[str, int]) -> ControllerDecision:
    """Narrow syntax repair only; no config defaults, fits or answer synthesis."""
    obj = _try_parse_json_obj(raw_text)
    if obj is not None and obj.get("action") == "run_experiment":
        cfg = obj.get("config")
        if isinstance(cfg, dict):
            family = case["family"]
            if family in LEARNERS:
                kind = LEARNERS[family]
                alias, target = (("trm_width", "width") if kind == "trm_style"
                                 else ("ldt_critical_veto", "critical_veto"))
                if alias in cfg and target not in cfg:
                    repaired = copy.deepcopy(obj)
                    repaired["config"][target] = repaired["config"].pop(alias)
                    return ControllerDecision(canon(repaired), True, False, "schema_alias")
            elif remaining.get("checks", 0) > 0:
                return ControllerDecision(
                    canon({"action": "check", "answer": cfg}), True, False,
                    "schema_symbolic_verb",
                )
    return ControllerDecision(raw_text, False, False, None)


class ResearchController(V21Controller):
    def __init__(self, mode: str, seed: int = 2026112701):
        if mode not in CONTROLLER_MODES:
            raise ValueError(f"unsupported v2.2 controller: {mode}")
        self.v22_mode = mode
        self.selective = None
        if mode in {"rule_action_routing", "rule_no_persona"}:
            super().__init__("autonomous", seed)
            self.mode = "trm_action_routing"
            self.trm_router = RuleRouter()
            self.training_receipt = {"mode": mode, "trained": False,
                                     "rule": "v21_synthetic_label_exact"}
        else:
            native = {"schema_only": "autonomous",
                      "selective_rlt": "rlt_state_correction",
                      "trm_no_persona": "trm_action_routing",
                      "rlt_no_persona": "rlt_state_correction"}.get(mode, mode)
            super().__init__(native, seed)
            if mode == "selective_rlt":
                self.selective = SelectiveCorrector(self.trm_router, self.rlt_corrector)
                self.rlt_corrector = self.selective
        self.training_receipt = {**self.training_receipt, "v22_mode": mode}
        if self.selective is not None:
            self.training_receipt["margin_threshold"] = self.selective.threshold

    def _build_learner_submit(self, case, state, proposed_obj):
        if self.v22_mode.endswith("_no_persona") and case["family"] == "persona_externalization":
            best = max(state.matched_pairs, key=lambda p: (p["cal_delta"], p["artifact_id"]))
            answer = {key: best[key] for key in ("artifact_id", "baseline_id", "claim")}
            if isinstance(proposed_obj, dict) and isinstance(proposed_obj.get("answer"), dict):
                memory = proposed_obj["answer"].get("memory")
                if memory is not None:
                    answer["memory"] = copy.deepcopy(memory)
            return {"action": "submit", "answer": answer}, False
        return super()._build_learner_submit(case, state, proposed_obj)

    def pre_turn_decision(self, **kwargs):
        # Without the public-memory projection, the actor must get a turn to supply it.
        case = kwargs["case"]
        if self.v22_mode.endswith("_no_persona") and case["family"] == "persona_externalization":
            return ControllerDecision(None, False, False, None)
        return super().pre_turn_decision(**kwargs)

    def post_model_decision(self, **kwargs):
        if self.v22_mode == "schema_only":
            return schema_repair(kwargs["case"], kwargs["raw_text"], kwargs["remaining"])
        if (self.v22_mode.endswith("_no_persona")
                and kwargs["case"]["family"] == "persona_externalization"):
            state = kwargs["state"]
            if state.matched_pairs:
                # Leave the final persona answer to the actor, keeping upstream routing.
                return ControllerDecision(kwargs["raw_text"], False, False, None)
        return super().post_model_decision(**kwargs)

    def metadata(self):
        return {"controller_mode": self.v22_mode, "training_receipt": self.training_receipt,
                "controller_sha256": digest(self.training_receipt)}

    def accounting(self):
        return self.selective.accounting() if self.selective is not None else {}
