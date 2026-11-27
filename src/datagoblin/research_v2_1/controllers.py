"""Controller-assisted stopping, LDT stopping, TRM action routing, and RLT state correction.

All controllers operate strictly on public episode state (CompactResearchState,
public task metadata, and the model's own proposed JSON text). No controller
ever accesses private seeds or privileged reference_answer(). Every intervention
is explicitly logged and accounted separately in reports.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from datagoblin.research_v2.suite import LEARNERS, canon, digest

from .persona_store_reference import reference_persona_memory
from .state import CompactResearchState

CONTROLLER_MODES = (
    "autonomous",
    "deterministic_stopping",
    "learned_ldt_stopping",
    "trm_action_routing",
    "rlt_state_correction",
)

LEARNER_ALLOWED_FIELDS = {
    "trm_style": {"steps", "train_examples", "learning_rate", "width", "recursions"},
    "ldt_filter": {"steps", "train_examples", "learning_rate", "critical_veto"},
    "ldt_persona": {"steps", "train_examples", "learning_rate", "critical_veto"},
}

FEATURE_NAMES = (
    "is_learner",
    "verified_ready",
    "has_matched_pair",
    "has_trained_only",
    "has_baseline_only",
    "turn_frac",
    "err_exp_unavail_seen",
    "err_chk_unavail_seen",
    "invalid_config_seen",
    "model_proposed_valid_json",
    "model_proposed_run_exp",
    "model_proposed_check",
    "model_proposed_submit",
    "model_config_has_extra_keys",
)
N_FEATURES = len(FEATURE_NAMES)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=-1, keepdims=True)


def _try_parse_json_obj(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`").lstrip("json").strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _sanitize_learner_config(family: str, raw_cfg: dict[str, Any]) -> dict[str, Any]:
    kind = LEARNERS[family]
    allowed = LEARNER_ALLOWED_FIELDS[kind]
    cfg: dict[str, Any] = {}
    for k, v in raw_cfg.items():
        if k == "trm_width" and kind == "trm_style" and "width" not in raw_cfg:
            cfg["width"] = v
        elif k == "ldt_critical_veto" and kind != "trm_style" and "critical_veto" not in raw_cfg:
            cfg["critical_veto"] = bool(v)
        elif k in allowed:
            cfg[k] = v
    if "steps" in cfg and isinstance(cfg["steps"], int):
        cfg["steps"] = max(0, min(128, cfg["steps"]))
    if "train_examples" in cfg and isinstance(cfg["train_examples"], int):
        te = max(16, min(256, cfg["train_examples"]))
        if kind == "trm_style" and te % 2 == 1:
            te -= 1
        cfg["train_examples"] = te
    if "learning_rate" in cfg and isinstance(cfg["learning_rate"], (int, float)):
        cfg["learning_rate"] = max(0.0001, min(0.5, float(cfg["learning_rate"])))
    if kind == "trm_style":
        if "width" in cfg and isinstance(cfg["width"], int):
            cfg["width"] = max(4, min(32, cfg["width"]))
        if "recursions" in cfg and isinstance(cfg["recursions"], int):
            cfg["recursions"] = max(1, min(8, cfg["recursions"]))
    else:
        cfg["critical_veto"] = True
    if "steps" not in cfg:
        cfg["steps"] = 64
    return cfg


def extract_controller_features(
    *,
    state: CompactResearchState,
    family: str,
    turn_index: int,
    proposed_obj: dict[str, Any] | None,
    remaining_checks: int = 2,
) -> np.ndarray:
    act = proposed_obj.get("action") if isinstance(proposed_obj, dict) else None
    extra_keys = False
    if family in LEARNERS and act == "run_experiment" and isinstance(proposed_obj, dict):
        cfg = proposed_obj.get("config")
        if isinstance(cfg, dict):
            allowed = LEARNER_ALLOWED_FIELDS[LEARNERS[family]]
            extra_keys = bool(set(cfg) - allowed)
    chk_unavail = "check" in state.disallowed_actions or (not state.is_learner and remaining_checks <= 0)
    return np.array(
        [
            1.0 if state.is_learner else 0.0,
            1.0 if state.verified_answer is not None else 0.0,
            1.0 if len(state.matched_pairs) > 0 else 0.0,
            1.0 if (len(state.trained_ids) > 0 and len(state.baseline_ids) == 0) else 0.0,
            1.0 if (len(state.baseline_ids) > 0 and len(state.trained_ids) == 0) else 0.0,
            min(1.0, float(turn_index) / 8.0),
            1.0 if "run_experiment" in state.disallowed_actions else 0.0,
            1.0 if chk_unavail else 0.0,
            1.0 if state.invalid_config_seen else 0.0,
            1.0 if proposed_obj is not None else 0.0,
            1.0 if act == "run_experiment" else 0.0,
            1.0 if act == "check" else 0.0,
            1.0 if act == "submit" else 0.0,
            1.0 if extra_keys else 0.0,
        ],
        dtype=np.float64,
    )


def _generate_controller_training_data(
    seed: int = 2026112701, n_samples: int = 1600
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate fresh synthetic state-transition training data strictly disjoint from any eval suites.

    Returns:
      x: (N, N_FEATURES)
      y_stop: 3-class LDT stopping target {0: continue, 1: stop_verified, 2: stop_learner_pair}
      y_route: 5-class TRM/RLT routing target
        {0: pass_through, 1: stop_verified, 2: stop_learner_pair,
         3: repair_learner_config_or_baseline, 4: repair_symbolic_verb}
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((n_samples, N_FEATURES), dtype=np.float64)
    y_stop = np.zeros(n_samples, dtype=np.int64)
    y_route = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        is_learner = int(rng.random() < 0.45)
        turn = int(rng.integers(0, 8))
        valid_json = int(rng.random() < 0.88)
        act_kind = int(rng.choice([0, 1, 2], p=[0.40, 0.40, 0.20])) if valid_json else -1

        verified_ready = 0
        has_pair = 0
        has_trained_only = 0
        has_base_only = 0
        err_exp = 0
        err_chk = 0
        inv_cfg = 0
        extra_keys = 0

        if is_learner:
            regime = int(rng.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.20, 0.30]))
            if regime == 1:
                has_trained_only = 1
            elif regime == 2:
                has_base_only = 1
            elif regime == 3:
                has_pair = 1
            err_chk = int(rng.random() < 0.45)
            inv_cfg = int(rng.random() < 0.35)
            if act_kind == 0:
                extra_keys = int(rng.random() < 0.45)
        else:
            verified_ready = int(rng.random() < 0.35)
            err_exp = int(rng.random() < 0.50)
            err_chk = int(rng.random() < 0.25)

        x[i] = [
            float(is_learner),
            float(verified_ready),
            float(has_pair),
            float(has_trained_only),
            float(has_base_only),
            turn / 8.0,
            float(err_exp),
            float(err_chk),
            float(inv_cfg),
            float(valid_json),
            1.0 if act_kind == 0 else 0.0,
            1.0 if act_kind == 1 else 0.0,
            1.0 if act_kind == 2 else 0.0,
            float(extra_keys),
        ]

        # Stopping target (3 classes)
        if not is_learner and verified_ready and act_kind != 2:
            y_stop[i] = 1
        elif is_learner and has_pair and act_kind != 2:
            y_stop[i] = 2
        else:
            y_stop[i] = 0

        # Routing target (5 classes)
        if not is_learner and verified_ready and act_kind != 2:
            y_route[i] = 1
        elif is_learner and has_pair and act_kind != 2:
            y_route[i] = 2
        elif is_learner and (
            extra_keys or (has_trained_only and act_kind in (0, 1)) or (not has_pair and act_kind == 1)
        ):
            y_route[i] = 3
        elif (not is_learner) and (act_kind == 0 or (err_chk and act_kind == 1)):
            y_route[i] = 4
        else:
            y_route[i] = 0

    return x, y_stop, y_route


class LearnedLDTStoppingHead:
    """Inspectable 45-parameter additive LDT stopping classifier (14x3 + 3)."""

    def __init__(self, n_features: int = N_FEATURES, seed: int = 20261127) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.02, size=(n_features, 3)).astype(np.float64)
        self.b = np.zeros(3, dtype=np.float64)

    def fit(
        self, x: np.ndarray, y: np.ndarray, *, epochs: int = 120, lr: float = 0.25, l2: float = 1e-4
    ) -> dict[str, Any]:
        n = len(x)
        for _ in range(epochs):
            probs = _softmax(x @ self.w + self.b)
            grad = probs.copy()
            grad[np.arange(n), y] -= 1.0
            grad /= n
            self.w -= lr * (x.T @ grad + l2 * self.w)
            self.b -= lr * grad.sum(axis=0)
        probs = _softmax(x @ self.w + self.b)
        loss = float(-np.log(np.clip(probs[np.arange(n), y], 1e-12, 1.0)).mean())
        acc = float(np.mean(probs.argmax(axis=1) == y))
        return {"parameters": int(self.w.size + self.b.size), "train_loss": round(loss, 6), "train_acc": round(acc, 6)}

    def predict(self, feat: np.ndarray) -> tuple[int, np.ndarray]:
        probs = _softmax(feat[None, :] @ self.w + self.b)[0]
        return int(probs.argmax()), probs


class TRMActionRouter:
    """Compact 2-step tied Tiny Recursive Model (TRM) action router (5 classes)."""

    def __init__(
        self, n_features: int = N_FEATURES, hidden: int = 12, recursions: int = 2, seed: int = 20261127
    ) -> None:
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        self.recursions = recursions
        self.w_in = rng.normal(0.0, 0.15, size=(n_features, hidden)).astype(np.float64)
        self.b_in = np.zeros(hidden, dtype=np.float64)
        self.w_rec = rng.normal(0.0, 0.15, size=(hidden, hidden)).astype(np.float64)
        self.b_rec = np.zeros(hidden, dtype=np.float64)
        self.w_out = rng.normal(0.0, 0.15, size=(hidden, 5)).astype(np.float64)
        self.b_out = np.zeros(5, dtype=np.float64)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        x_proj = x @ self.w_in + self.b_in
        h = np.zeros((len(x), self.hidden), dtype=np.float64)
        states = [h]
        for _ in range(self.recursions):
            h = np.tanh(x_proj + h @ self.w_rec + self.b_rec)
            states.append(h)
        logits = h @ self.w_out + self.b_out
        return logits, states, x_proj

    def fit(
        self, x: np.ndarray, y: np.ndarray, *, epochs: int = 180, lr: float = 0.25, l2: float = 1e-4
    ) -> dict[str, Any]:
        n = len(x)
        for _ in range(epochs):
            logits, states, _ = self.forward(x)
            probs = _softmax(logits)
            dlogits = probs.copy()
            dlogits[np.arange(n), y] -= 1.0
            dlogits /= n
            gw_out = states[-1].T @ dlogits + l2 * self.w_out
            gb_out = dlogits.sum(axis=0)
            dh = dlogits @ self.w_out.T
            gw_in = np.zeros_like(self.w_in)
            gb_in = np.zeros_like(self.b_in)
            gw_rec = np.zeros_like(self.w_rec)
            gb_rec = np.zeros_like(self.b_rec)
            for step in range(self.recursions, 0, -1):
                h_cur = states[step]
                h_prev = states[step - 1]
                dpre = dh * (1.0 - h_cur * h_cur)
                gw_in += x.T @ dpre
                gb_in += dpre.sum(axis=0)
                gw_rec += h_prev.T @ dpre
                gb_rec += dpre.sum(axis=0)
                dh = dpre @ self.w_rec.T
            self.w_out -= lr * gw_out
            self.b_out -= lr * gb_out
            self.w_in -= lr * (gw_in + l2 * self.w_in)
            self.b_in -= lr * gb_in
            self.w_rec -= lr * (gw_rec + l2 * self.w_rec)
            self.b_rec -= lr * gb_rec
        logits, _, _ = self.forward(x)
        probs = _softmax(logits)
        loss = float(-np.log(np.clip(probs[np.arange(n), y], 1e-12, 1.0)).mean())
        acc = float(np.mean(probs.argmax(axis=1) == y))
        params = int(
            self.w_in.size
            + self.b_in.size
            + self.w_rec.size
            + self.b_rec.size
            + self.w_out.size
            + self.b_out.size
        )
        return {"parameters": params, "train_loss": round(loss, 6), "train_acc": round(acc, 6)}

    def predict(self, feat: np.ndarray) -> tuple[int, np.ndarray]:
        logits, _, _ = self.forward(feat[None, :])
        probs = _softmax(logits)[0]
        return int(probs.argmax()), probs


class RLTStateCorrector:
    """TRM router paired with K=3 damped RLT residual corrector trained with EXACT unrolled BPTT."""

    def __init__(self, router: TRMActionRouter, passes: int = 3, hidden: int = 12, seed: int = 20261128) -> None:
        self.router = router
        self.passes = passes
        self.alphas = [1.0 / (passes - j) for j in range(passes)]
        rng = np.random.default_rng(seed)
        # Input to corrector: probs (5) + margin (1) + base_context (2: is_learner, verified_or_pair) = 8
        self.w1 = rng.normal(0.0, 0.08, size=(8, hidden)).astype(np.float64)
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.w2 = rng.normal(0.0, 0.04, size=(hidden, 5)).astype(np.float64)
        self.b2 = np.zeros(5, dtype=np.float64)

    def _ctx(self, x: np.ndarray) -> np.ndarray:
        is_l = x[:, 0:1]
        ready = np.maximum(x[:, 1:2], x[:, 2:3])
        return np.concatenate([is_l, ready], axis=1)

    def forward_from_logits(
        self, base_logits: np.ndarray, x: np.ndarray
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        ctx = self._ctx(x)
        cur = base_logits.copy()
        cache: list[dict[str, Any]] = []
        for alpha in self.alphas:
            probs = _softmax(cur)
            order = np.argsort(probs, axis=1)
            top1_idx = order[:, -1]
            top2_idx = order[:, -2]
            rows = np.arange(len(cur))
            margin = (probs[rows, top1_idx] - probs[rows, top2_idx])[:, None]
            feat = np.concatenate([probs, margin, ctx], axis=1)
            hid = np.tanh(feat @ self.w1 + self.b1)
            delta = hid @ self.w2 + self.b2
            nxt = cur + alpha * delta
            cache.append(
                {
                    "probs": probs,
                    "top1": top1_idx,
                    "top2": top2_idx,
                    "feat": feat,
                    "hid": hid,
                    "alpha": alpha,
                }
            )
            cur = nxt
        return cur, cache

    def backward_unrolled(
        self, dlogits: np.ndarray, cache: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Exact unrolled BPTT through all K damped residual passes, margin, and softmax."""
        gw1 = np.zeros_like(self.w1)
        gb1 = np.zeros_like(self.b1)
        gw2 = np.zeros_like(self.w2)
        gb2 = np.zeros_like(self.b2)
        dcur = dlogits.copy()
        rows = np.arange(len(dcur))

        for step in reversed(cache):
            alpha = step["alpha"]
            hid = step["hid"]
            feat = step["feat"]
            probs = step["probs"]
            top1 = step["top1"]
            top2 = step["top2"]

            ddelta = dcur * alpha
            gw2 += hid.T @ ddelta
            gb2 += ddelta.sum(axis=0)
            dhid = (ddelta @ self.w2.T) * (1.0 - hid * hid)
            gw1 += feat.T @ dhid
            gb1 += dhid.sum(axis=0)
            dfeat = dhid @ self.w1.T

            # Exact Jacobian through margin = p[top1] - p[top2] and softmax(cur)
            dprobs = dfeat[:, :5].copy()
            dmargin = dfeat[:, 5]
            dprobs[rows, top1] += dmargin
            dprobs[rows, top2] -= dmargin
            dot = np.sum(dprobs * probs, axis=1, keepdims=True)
            dcur = dcur + probs * (dprobs - dot)
        return dcur, gw1, gb1, gw2, gb2

    def fit(
        self, x: np.ndarray, y: np.ndarray, *, epochs: int = 120, lr: float = 0.15, l2: float = 1e-4
    ) -> dict[str, Any]:
        n = len(x)
        base_logits, _, _ = self.router.forward(x)
        for _ in range(epochs):
            logits, cache = self.forward_from_logits(base_logits, x)
            probs = _softmax(logits)
            dlogits = probs.copy()
            dlogits[np.arange(n), y] -= 1.0
            dlogits /= n
            _, gw1, gb1, gw2, gb2 = self.backward_unrolled(dlogits, cache)
            self.w2 -= lr * (gw2 + l2 * self.w2)
            self.b2 -= lr * gb2
            self.w1 -= lr * (gw1 + l2 * self.w1)
            self.b1 -= lr * gb1
        logits, _ = self.forward_from_logits(base_logits, x)
        probs = _softmax(logits)
        loss = float(-np.log(np.clip(probs[np.arange(n), y], 1e-12, 1.0)).mean())
        acc = float(np.mean(probs.argmax(axis=1) == y))
        params = int(self.w1.size + self.b1.size + self.w2.size + self.b2.size)
        return {"rlt_parameters": params, "train_loss": round(loss, 6), "train_acc": round(acc, 6)}

    def predict(self, feat: np.ndarray) -> tuple[int, np.ndarray]:
        base_logits, _, _ = self.router.forward(feat[None, :])
        logits, _ = self.forward_from_logits(base_logits, feat[None, :])
        probs = _softmax(logits)[0]
        return int(probs.argmax()), probs


@dataclass(frozen=True)
class ControllerDecision:
    executed_text: str | None
    intervened: bool
    skip_model_call: bool
    reason: str | None
    externalized_persona_used: bool = False


class ResearchController:
    """Unified auditable controller supporting all 5 v2.1 execution modes."""

    def __init__(self, mode: str, seed: int = 2026112701) -> None:
        if mode not in CONTROLLER_MODES:
            raise ValueError(f"unsupported controller mode: {mode}")
        self.mode = mode
        self.training_receipt: dict[str, Any] = {"mode": mode, "trained": False}
        self.ldt_head: LearnedLDTStoppingHead | None = None
        self.trm_router: TRMActionRouter | None = None
        self.rlt_corrector: RLTStateCorrector | None = None

        if mode in {"learned_ldt_stopping", "trm_action_routing", "rlt_state_correction"}:
            x_tr, y_stop, y_route = _generate_controller_training_data(seed=seed, n_samples=1600)
            if mode == "learned_ldt_stopping":
                self.ldt_head = LearnedLDTStoppingHead(seed=seed)
                info = self.ldt_head.fit(x_tr, y_stop)
                self.training_receipt = {"mode": mode, "trained": True, "train_seed": seed, **info}
            elif mode == "trm_action_routing":
                self.trm_router = TRMActionRouter(seed=seed)
                info = self.trm_router.fit(x_tr, y_route)
                self.training_receipt = {"mode": mode, "trained": True, "train_seed": seed, **info}
            else:
                self.trm_router = TRMActionRouter(seed=seed)
                trm_info = self.trm_router.fit(x_tr, y_route)
                self.rlt_corrector = RLTStateCorrector(self.trm_router, passes=3, seed=seed + 1)
                rlt_info = self.rlt_corrector.fit(x_tr, y_route)
                self.training_receipt = {
                    "mode": mode,
                    "trained": True,
                    "train_seed": seed,
                    "trm": trm_info,
                    "rlt": rlt_info,
                }

    def _build_learner_submit(
        self, case: dict[str, Any], state: CompactResearchState, proposed_obj: dict[str, Any] | None
    ) -> tuple[dict[str, Any], bool]:
        # Pick best matched pair by highest calibration delta
        best = max(state.matched_pairs, key=lambda p: (p["cal_delta"], p["artifact_id"]))
        ans: dict[str, Any] = {
            "artifact_id": best["artifact_id"],
            "baseline_id": best["baseline_id"],
            "claim": best["claim"],
        }
        ext_used = False
        if case["family"] == "persona_externalization":
            # Externalized persona store projects owner/scope/temporal/revocation state independently
            ans["memory"] = reference_persona_memory(case["public"])
            ext_used = True
        return {"action": "submit", "answer": ans}, ext_used

    def pre_turn_decision(
        self,
        *,
        case: dict[str, Any],
        state: CompactResearchState,
        turn_index: int,
        remaining: dict[str, int] | None = None,
    ) -> ControllerDecision:
        """Check whether verified stopping should fire BEFORE paying for another LLM turn."""
        if self.mode == "autonomous":
            return ControllerDecision(None, False, False, None)

        rem_checks = remaining.get("checks", 2) if remaining is not None else 2
        if state.verified_answer is not None:
            if self.mode == "deterministic_stopping":
                return ControllerDecision(
                    canon({"action": "submit", "answer": state.verified_answer}),
                    True,
                    True,
                    "deterministic_stop_after_verified_check",
                )
            feat = extract_controller_features(
                state=state,
                family=case["family"],
                turn_index=turn_index,
                proposed_obj=None,
                remaining_checks=rem_checks,
            )
            if self.mode == "learned_ldt_stopping" and self.ldt_head is not None:
                pred, _ = self.ldt_head.predict(feat)
                if pred == 1:
                    return ControllerDecision(
                        canon({"action": "submit", "answer": state.verified_answer}),
                        True,
                        True,
                        "ldt_stop_after_verified_check",
                    )
            elif self.mode == "trm_action_routing" and self.trm_router is not None:
                pred, _ = self.trm_router.predict(feat)
                if pred == 1:
                    return ControllerDecision(
                        canon({"action": "submit", "answer": state.verified_answer}),
                        True,
                        True,
                        "trm_stop_after_verified_check",
                    )
            elif self.mode == "rlt_state_correction" and self.rlt_corrector is not None:
                pred, _ = self.rlt_corrector.predict(feat)
                if pred == 1:
                    return ControllerDecision(
                        canon({"action": "submit", "answer": state.verified_answer}),
                        True,
                        True,
                        "rlt_stop_after_verified_check",
                    )

        learner_stop_ready = state.is_learner and bool(state.matched_pairs) and (
            any(p["cal_delta"] > 0 for p in state.matched_pairs)
            or remaining is None
            or remaining.get("experiments", 0) < 2
            or remaining.get("optimizer_updates", 0) < 16
            or turn_index >= 4
        )
        if learner_stop_ready:
            feat = extract_controller_features(
                state=state,
                family=case["family"],
                turn_index=turn_index,
                proposed_obj=None,
                remaining_checks=rem_checks,
            )
            if self.mode == "deterministic_stopping":
                sub, ext = self._build_learner_submit(case, state, None)
                return ControllerDecision(canon(sub), True, True, "deterministic_stop_matched_learner_pair", ext)
            if self.mode == "learned_ldt_stopping" and self.ldt_head is not None:
                pred, _ = self.ldt_head.predict(feat)
                if pred == 2:
                    sub, ext = self._build_learner_submit(case, state, None)
                    return ControllerDecision(canon(sub), True, True, "ldt_stop_matched_learner_pair", ext)
            if self.mode == "trm_action_routing" and self.trm_router is not None:
                pred, _ = self.trm_router.predict(feat)
                if pred == 2:
                    sub, ext = self._build_learner_submit(case, state, None)
                    return ControllerDecision(canon(sub), True, True, "trm_stop_matched_learner_pair", ext)
            if self.mode == "rlt_state_correction" and self.rlt_corrector is not None:
                pred, _ = self.rlt_corrector.predict(feat)
                if pred == 2:
                    sub, ext = self._build_learner_submit(case, state, None)
                    return ControllerDecision(canon(sub), True, True, "rlt_stop_matched_learner_pair", ext)

        return ControllerDecision(None, False, False, None)

    def post_model_decision(
        self,
        *,
        case: dict[str, Any],
        state: CompactResearchState,
        turn_index: int,
        raw_text: str,
        artifacts: dict[str, dict[str, Any]],
        remaining: dict[str, int],
    ) -> ControllerDecision:
        """Inspect and optionally route/repair the model's proposed action text."""
        if self.mode == "autonomous":
            return ControllerDecision(raw_text, False, False, None)

        obj = _try_parse_json_obj(raw_text)
        if obj is None:
            return ControllerDecision(raw_text, False, False, None)

        if self.mode == "deterministic_stopping":
            if state.verified_answer is not None and obj.get("action") != "submit":
                return ControllerDecision(
                    canon({"action": "submit", "answer": state.verified_answer}),
                    True,
                    False,
                    "deterministic_stop_override",
                )
            if state.is_learner and state.matched_pairs and obj.get("action") != "submit":
                sub, ext = self._build_learner_submit(case, state, obj)
                return ControllerDecision(canon(sub), True, False, "deterministic_stop_learner_override", ext)
            return ControllerDecision(raw_text, False, False, None)

        feat = extract_controller_features(
            state=state,
            family=case["family"],
            turn_index=turn_index,
            proposed_obj=obj,
            remaining_checks=remaining.get("checks", 0),
        )

        if self.mode == "learned_ldt_stopping" and self.ldt_head is not None:
            pred, _ = self.ldt_head.predict(feat)
            # Structural safety veto: never stop unless verified_answer or matched_pair exists
            if pred == 1 and state.verified_answer is not None:
                return ControllerDecision(
                    canon({"action": "submit", "answer": state.verified_answer}),
                    True,
                    False,
                    "ldt_stop_verified_override",
                )
            if pred == 2 and state.is_learner and state.matched_pairs:
                sub, ext = self._build_learner_submit(case, state, obj)
                return ControllerDecision(canon(sub), True, False, "ldt_stop_learner_override", ext)
            return ControllerDecision(raw_text, False, False, None)

        if self.mode in {"trm_action_routing", "rlt_state_correction"}:
            if self.mode == "trm_action_routing" and self.trm_router is not None:
                pred, _ = self.trm_router.predict(feat)
                prefix = "trm"
            elif self.rlt_corrector is not None:
                pred, _ = self.rlt_corrector.predict(feat)
                prefix = "rlt"
            else:
                return ControllerDecision(raw_text, False, False, None)

            if pred == 1 and state.verified_answer is not None:
                return ControllerDecision(
                    canon({"action": "submit", "answer": state.verified_answer}),
                    True,
                    False,
                    f"{prefix}_route_submit_verified",
                )
            if pred == 2 and state.is_learner and state.matched_pairs:
                sub, ext = self._build_learner_submit(case, state, obj)
                return ControllerDecision(canon(sub), True, False, f"{prefix}_route_submit_learner_pair", ext)
            if pred == 3 and state.is_learner:
                return self._repair_learner_action(case, state, obj, artifacts, remaining, prefix)
            if pred == 4 and not state.is_learner:
                return self._repair_symbolic_action(state, obj, remaining, prefix)

        return ControllerDecision(raw_text, False, False, None)

    def _repair_learner_action(
        self,
        case: dict[str, Any],
        state: CompactResearchState,
        obj: dict[str, Any],
        artifacts: dict[str, dict[str, Any]],
        remaining: dict[str, int],
        prefix: str,
    ) -> ControllerDecision:
        family = case["family"]
        act = obj.get("action")
        # Case A: model trained a steps>0 artifact without a matched steps=0 baseline -> fit matched steps=0 baseline
        if state.trained_ids and not state.baseline_ids and remaining.get("experiments", 0) > 0:
            trained_cfg = copy.deepcopy(artifacts[state.trained_ids[-1]]["config"])
            trained_cfg["steps"] = 0
            return ControllerDecision(
                canon({"action": "run_experiment", "config": trained_cfg}),
                True,
                False,
                f"{prefix}_route_matched_zero_baseline",
            )
        # Case B: model supplied invalid extra learner config keys -> sanitize to allowlisted schema
        if act == "run_experiment" and isinstance(obj.get("config"), dict):
            sanitized = _sanitize_learner_config(family, obj["config"])
            if sanitized != obj["config"]:
                return ControllerDecision(
                    canon({"action": "run_experiment", "config": sanitized}),
                    True,
                    False,
                    f"{prefix}_route_sanitize_learner_config",
                )
        # Case C: model called illegal check on learner family before fitting -> start matched baseline or train fit
        if act == "check" and remaining.get("experiments", 0) > 0:
            if state.baseline_ids and not state.trained_ids:
                base_cfg = copy.deepcopy(artifacts[state.baseline_ids[-1]]["config"])
                base_cfg["steps"] = min(64, max(16, remaining.get("optimizer_updates", 64)))
                return ControllerDecision(
                    canon({"action": "run_experiment", "config": base_cfg}),
                    True,
                    False,
                    f"{prefix}_route_train_after_baseline",
                )
            if not state.baseline_ids:
                seed_cfg = _sanitize_learner_config(family, state.last_candidate_config or {})
                seed_cfg["steps"] = 0
                return ControllerDecision(
                    canon({"action": "run_experiment", "config": seed_cfg}),
                    True,
                    False,
                    f"{prefix}_route_init_zero_baseline",
                )
        return ControllerDecision(canon(obj), False, False, None)

    def _repair_symbolic_action(
        self,
        state: CompactResearchState,
        obj: dict[str, Any],
        remaining: dict[str, int],
        prefix: str,
    ) -> ControllerDecision:
        act = obj.get("action")
        # If checks are exhausted and model keeps calling check/run_experiment, submit its latest candidate answer
        if remaining.get("checks", 0) <= 0:
            raw_ans = obj.get("answer")
            cand = raw_ans if (isinstance(raw_ans, dict) and raw_ans) else state.last_candidate_answer
            if isinstance(cand, dict) and cand:
                return ControllerDecision(
                    canon({"action": "submit", "answer": cand}),
                    True,
                    False,
                    f"{prefix}_route_submit_after_check_exhaustion",
                )
        # If model proposed run_experiment on a symbolic task, convert config -> check if it looks like an answer
        if act == "run_experiment" and isinstance(obj.get("config"), dict) and remaining.get("checks", 0) > 0:
            cand = obj["config"]
            return ControllerDecision(
                canon({"action": "check", "answer": cand}),
                True,
                False,
                f"{prefix}_route_symbolic_exp_to_check",
            )
        return ControllerDecision(canon(obj), False, False, None)

    def metadata(self) -> dict[str, Any]:
        return {
            "controller_mode": self.mode,
            "training_receipt": self.training_receipt,
            "controller_sha256": digest(self.training_receipt),
        }
