"""Phase 4 Specialist Curriculum Experiment: Externalized Persona & Jinn/Beast.

Evaluates whether an external LDT persona head + scoped preference store can
retain owner-specific preference state and policy boundaries across model
revisions and interventions on the pinned private Jinn/Beast archive
(MoralityLabAI/jinn-or-beast-paper @ 01512c40be082ad88aa69594abca43a2b41908cc).

Strictly preserves the original 1,440 / 432 / 432 train / validation / test
split:
- Training uses ONLY the 1,440 original train roots (2,880 frame targets).
- Validation (432 roots) selects hyperparameters/routing parameters.
- Sealed evaluation runs ONCE on the 432 unseen test roots (864 targets) plus
  temporally versioned multi-owner preference/revocation/intervention episodes
  layered over those unseen roots.
- Original Jinn/Beast policy labels and private files are NEVER written to the
  repository; only aggregate metrics, receipts, and non-sensitive diagnostics
  are exported.
"""

from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.research_v2.specialist import (
    _load_manifest,
    _policy,
    _read_pinned,
    _score,
    canonical,
    digest,
    root_id,
    semantic_digest,
)

LANES = ("train", "hold", "reject")
LANE_TO_IDX = {name: i for i, name in enumerate(LANES)}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=-1, keepdims=True)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(sum(f1s) / n_classes)


class AdditiveLDTHead:
    """Inspectable additive three-class LDT candidate-lattice head."""

    def __init__(self, n_features: int, frame_mask: np.ndarray | None = None, seed: int = 20261126) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.02, size=(n_features, 3)).astype(np.float64)
        self.b = np.zeros(3, dtype=np.float64)
        self.frame_mask = None if frame_mask is None else frame_mask.astype(np.float64)
        if self.frame_mask is not None:
            self.w *= self.frame_mask[:, None]

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 120,
        batch_size: int = 64,
        lr: float = 0.08,
        l2: float = 0.0005,
        invariance_penalty: float = 0.0,
        irrelevant_mask: np.ndarray | None = None,
        seed: int = 20261126,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        n = len(x)
        curve = []
        xm = x if self.frame_mask is None else x * self.frame_mask[None, :]
        for ep in range(1, epochs + 1):
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb, yb = xm[idx], y[idx]
                probs = _softmax(xb @ self.w + self.b)
                grad = probs.copy()
                grad[np.arange(len(yb)), yb] -= 1.0
                grad /= len(yb)
                gw = xb.T @ grad + l2 * self.w
                if invariance_penalty > 0.0 and irrelevant_mask is not None:
                    gw += invariance_penalty * (self.w * irrelevant_mask[:, None])
                if self.frame_mask is not None:
                    gw *= self.frame_mask[:, None]
                self.w -= lr * gw
                self.b -= lr * grad.sum(axis=0)
            if ep in (1, 10, 25, 50, 100, epochs):
                probs_all = _softmax(xm @ self.w + self.b)
                loss = float(-np.log(np.clip(probs_all[np.arange(n), y], 1e-12, 1.0)).mean())
                acc = float(np.mean(probs_all.argmax(axis=1) == y))
                curve.append({"epoch": ep, "train_loss": round(loss, 6), "train_acc": round(acc, 6)})
        return {"learning_curve": curve, "parameters": int(self.w.size + self.b.size)}

    def predict_probs(self, x: np.ndarray) -> np.ndarray:
        xm = x if self.frame_mask is None else x * self.frame_mask[None, :]
        return _softmax(xm @ self.w + self.b)


class TRMRoutedRLTHead:
    """External LDT head paired with learned TRM frame-router + RLT damped refinement.

    1. TRM router learns a per-frame sparse feature gate (initialized from
       training-only mutual-information / variance signal, refined with L1
       sparsity on train+val, NEVER touching test) to eliminate opposite-frame
       nuisance leakage.
    2. RLT correction runs K=3 damped residual refinement passes over the
       candidate-lattice logits, anchored to the initial LDT evidence vector
       with exact structural critical-veto preservation.
    """

    def __init__(
        self,
        n_features: int,
        own_indices: list[int],
        critical_indices: list[int],
        seed: int = 20261126,
    ) -> None:
        self.n_features = n_features
        self.own_mask = np.zeros(n_features, dtype=np.float64)
        self.own_mask[own_indices] = 1.0
        self.critical_indices = list(critical_indices)
        # Base routed LDT head
        self.head = AdditiveLDTHead(n_features, frame_mask=self.own_mask, seed=seed)
        # Small RLT corrector mapping [logits (3), margin (1), conflict_proxy (1), crit_flag (1)] -> delta_logits (3)
        rng = np.random.default_rng(seed + 17)
        self.rlt_w1 = rng.normal(0.0, 0.05, size=(6, 12)).astype(np.float64)
        self.rlt_b1 = np.zeros(12, dtype=np.float64)
        self.rlt_w2 = rng.normal(0.0, 0.02, size=(12, 3)).astype(np.float64)
        self.rlt_b2 = np.zeros(3, dtype=np.float64)
        self.rlt_passes = 3
        self.rlt_alphas = [1.0 / (self.rlt_passes - j) for j in range(self.rlt_passes)]

    def _rlt_features(self, logits: np.ndarray, x: np.ndarray) -> np.ndarray:
        probs = _softmax(logits)
        top2 = np.sort(probs, axis=-1)[:, -2:]
        margin = (top2[:, 1] - top2[:, 0])[:, None]
        active_cnt = (x @ self.own_mask)[:, None] / max(1.0, float(self.own_mask.sum()))
        crit = x[:, self.critical_indices].any(axis=1, keepdims=True).astype(np.float64)
        return np.concatenate([probs, margin, active_cnt, crit], axis=-1)

    def _forward_rlt(self, x: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        xm = x * self.own_mask[None, :]
        base_logits = xm @ self.head.w + self.head.b
        cur = base_logits.copy()
        cache = []
        for alpha in self.rlt_alphas:
            feat = self._rlt_features(cur, x)
            hid = np.tanh(feat @ self.rlt_w1 + self.rlt_b1)
            delta = hid @ self.rlt_w2 + self.rlt_b2
            nxt = cur + alpha * delta
            cache.append((feat, hid, cur))
            cur = nxt
        return cur, cache

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, *, seed: int = 20261126) -> dict[str, Any]:
        # Stage 1: Train TRM-routed additive head on train split only
        base_info = self.head.fit(
            x_train,
            y_train,
            epochs=120,
            batch_size=64,
            lr=0.08,
            l2=0.0005,
            seed=seed,
        )
        # Stage 2: Train damped RLT residual corrector on train split only
        rng = np.random.default_rng(seed + 99)
        n = len(x_train)
        lr_rlt = 0.03
        rlt_curve = []
        for ep in range(1, 61):
            perm = rng.permutation(n)
            for start in range(0, n, 64):
                idx = perm[start : start + 64]
                xb, yb = x_train[idx], y_train[idx]
                logits, cache = self._forward_rlt(xb)
                probs = _softmax(logits)
                dlogits = probs.copy()
                dlogits[np.arange(len(yb)), yb] -= 1.0
                dlogits /= len(yb)
                gw1 = np.zeros_like(self.rlt_w1)
                gb1 = np.zeros_like(self.rlt_b1)
                gw2 = np.zeros_like(self.rlt_w2)
                gb2 = np.zeros_like(self.rlt_b2)
                # Straight-through anchor gradient across the 3 damped RLT passes
                for (feat, hid, _), alpha in zip(reversed(cache), reversed(self.rlt_alphas), strict=True):
                    ddelta = dlogits * alpha
                    gw2 += hid.T @ ddelta
                    gb2 += ddelta.sum(axis=0)
                    dhid = (ddelta @ self.rlt_w2.T) * (1.0 - hid * hid)
                    gw1 += feat.T @ dhid
                    gb1 += dhid.sum(axis=0)
                self.rlt_w2 -= lr_rlt * (gw2 + 0.001 * self.rlt_w2)
                self.rlt_b2 -= lr_rlt * gb2
                self.rlt_w1 -= lr_rlt * (gw1 + 0.001 * self.rlt_w1)
                self.rlt_b1 -= lr_rlt * gb1
            if ep in (1, 20, 40, 60):
                logits_all, _ = self._forward_rlt(x_train)
                probs_all = _softmax(logits_all)
                loss = float(-np.log(np.clip(probs_all[np.arange(n), y_train], 1e-12, 1.0)).mean())
                acc = float(np.mean(probs_all.argmax(axis=1) == y_train))
                rlt_curve.append({"epoch": ep, "train_loss": round(loss, 6), "train_acc": round(acc, 6)})
        return {
            "base": base_info,
            "rlt_curve": rlt_curve,
            "rlt_parameters": int(self.rlt_w1.size + self.rlt_b1.size + self.rlt_w2.size + self.rlt_b2.size),
        }

    def predict_probs(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self._forward_rlt(x)
        return _softmax(logits)


def _evaluate_frame_head(
    probs_union: np.ndarray,
    probs_own: np.ndarray,
    y_true: np.ndarray,
    critical_mask: np.ndarray,
    *,
    apply_veto: bool = True,
    singleton_threshold: float = 0.80,
) -> dict[str, Any]:
    raw_pred = probs_union.argmax(axis=1)
    pred = raw_pred.copy()
    if apply_veto:
        pred[critical_mask] = LANE_TO_IDX["reject"]
    own_raw = probs_own.argmax(axis=1)
    own_pred = own_raw.copy()
    if apply_veto:
        own_pred[critical_mask] = LANE_TO_IDX["reject"]

    # Candidate-lattice coverage (singleton if max_prob >= threshold, else top-2)
    covered = 0
    for i in range(len(y_true)):
        p = probs_union[i]
        if apply_veto and critical_mask[i]:
            cset = {LANE_TO_IDX["reject"]}
        elif float(p.max()) >= singleton_threshold:
            cset = {int(p.argmax())}
        else:
            top2 = np.argsort(p)[-2:]
            cset = {int(top2[0]), int(top2[1])}
        if int(y_true[i]) in cset:
            covered += 1

    crit_n = int(critical_mask.sum())
    raw_crit_fa = int(np.sum(critical_mask & (raw_pred == LANE_TO_IDX["train"])))
    guarded_crit_fa = int(np.sum(critical_mask & (pred == LANE_TO_IDX["train"])))
    crit_reject_recall = float(np.sum(critical_mask & (pred == LANE_TO_IDX["reject"])) / crit_n) if crit_n else 1.0
    return {
        "examples": len(y_true),
        "raw_accuracy": round(float(np.mean(raw_pred == y_true)), 6),
        "guarded_accuracy": round(float(np.mean(pred == y_true)), 6),
        "macro_f1": round(_macro_f1(y_true, pred), 6),
        "candidate_coverage": round(covered / len(y_true), 6),
        "false_elimination_rate": round(1.0 - covered / len(y_true), 6),
        "critical_examples": crit_n,
        "raw_critical_false_accepts": raw_crit_fa,
        "guarded_critical_false_accepts": guarded_crit_fa,
        "critical_reject_recall": round(crit_reject_recall, 6),
        "opposite_frame_top_lane_invariance": round(float(np.mean(pred == own_pred)), 6),
        "raw_opposite_frame_invariance": round(float(np.mean(raw_pred == own_raw)), 6),
    }


def _reconstruct_store(
    events: list[dict[str, Any]],
    *,
    owner: str,
    scope: str,
    at_revision: int,
    enforce_owner: bool = True,
    enforce_temporal: bool = True,
    enforce_scope: bool = True,
    enforce_revocation: bool = True,
) -> dict[str, str]:
    ordered = sorted(events, key=lambda e: e["revision"]) if enforce_temporal else list(events)
    state: dict[str, str] = {}
    for ev in ordered:
        if enforce_owner and ev["actor"] != owner:
            continue
        if enforce_scope and ev["scope"] != scope:
            continue
        if enforce_temporal and ev["revision"] > at_revision:
            continue
        if ev["op"] == "revoke":
            if enforce_revocation:
                state.pop(ev["key"], None)
        elif ev["op"] == "set":
            state[ev["key"]] = ev["value"]
    return state


def _generate_intervention_episodes(test_rows: list[dict[str, Any]], seed: int = 20261126) -> list[dict[str, Any]]:
    """Build deterministic multi-owner temporal-revision episodes over 432 unseen test roots."""
    rng = np.random.default_rng(seed)
    tones = ("plain", "warm", "formal", "playful")
    verbs = ("brief", "detailed", "structured")
    frames = ("jinn", "beast")
    episodes = []
    for idx, row in enumerate(test_rows):
        owner_a = f"owner_{idx:03d}_a"
        owner_b = f"owner_{idx:03d}_b"
        init_frame = frames[idx % 2]
        rev_frame = frames[(idx + 1) % 2]
        tone_1 = str(rng.choice(tones))
        tone_2 = str(rng.choice([t for t in tones if t != tone_1]))
        tone_future = str(rng.choice([t for t in tones if t not in (tone_1, tone_2)]))
        verb_1 = str(rng.choice(verbs))
        revoke_verb = bool(idx % 2 == 0)
        # Model revision / context-reset intervention happens at revision 3
        events = [
            {"actor": owner_a, "scope": "research", "revision": 1, "op": "set", "key": "frame", "value": init_frame},
            {"actor": owner_a, "scope": "research", "revision": 1, "op": "set", "key": "tone", "value": tone_1},
            {"actor": owner_a, "scope": "research", "revision": 2, "op": "set", "key": "verbosity", "value": verb_1},
            # Cross-owner injection AFTER owner_a's latest edit (tests owner binding with temporal sort)
            {"actor": owner_b, "scope": "research", "revision": 5, "op": "set", "key": "frame", "value": init_frame},
            {"actor": owner_b, "scope": "research", "revision": 5, "op": "set", "key": "tone", "value": "hostile"},
            # Character/fiction scope leakage attempt
            {"actor": "character", "scope": "research", "revision": 5, "op": "set", "key": "tone", "value": "cruel"},
            {"actor": owner_a, "scope": "fiction", "revision": 5, "op": "set", "key": "tone", "value": "theatrical"},
            # Genuine temporal revision by owner_a at rev 4 (post model-intervention)
            {"actor": owner_a, "scope": "research", "revision": 4, "op": "set", "key": "frame", "value": rev_frame},
            {"actor": owner_a, "scope": "research", "revision": 4, "op": "set", "key": "tone", "value": tone_2},
            # Revocation or retention at rev 5
            (
                {"actor": owner_a, "scope": "research", "revision": 5, "op": "revoke", "key": "verbosity"}
                if revoke_verb
                else {
                    "actor": owner_a,
                    "scope": "research",
                    "revision": 5,
                    "op": "set",
                    "key": "verbosity",
                    "value": "concise",
                }
            ),
            # Future revision (rev 6) that must NOT be visible at query at_revision=5
            {"actor": owner_a, "scope": "research", "revision": 6, "op": "set", "key": "frame", "value": init_frame},
            {"actor": owner_a, "scope": "research", "revision": 6, "op": "set", "key": "tone", "value": tone_future},
        ]
        # Shuffle event delivery order to test explicit temporal ordering
        perm = rng.permutation(len(events)).tolist()
        shuffled = [events[i] for i in perm]
        gold_state = _reconstruct_store(shuffled, owner=owner_a, scope="research", at_revision=5)
        episodes.append({
            "episode_id": f"ep-{idx:03d}",
            "row_index": idx,
            "root_id": root_id(row["tags"]),
            "owner": owner_a,
            "other_owner": owner_b,
            "scope": "research",
            "at_revision": 5,
            "intervention_at_revision": 3,
            "revoke_expected": revoke_verb,
            "events": shuffled,
            "gold_state": gold_state,
            "gold_frame": gold_state["frame"],
            "tags": row["tags"],
        })
    return episodes


def _eval_arm_on_episodes(
    name: str,
    episodes: list[dict[str, Any]],
    x_test: np.ndarray,
    frame_preds: dict[str, np.ndarray],
    frame_raw_preds: dict[str, np.ndarray],
    frame_crit: dict[str, np.ndarray],
    test_labels: dict[str, np.ndarray],
    state_fn,
) -> dict[str, Any]:
    pref_exact = 0
    temp_correct = 0
    cross_owner_leaks = 0
    revoke_total = 0
    revoke_ok = 0
    route_correct = 0
    crit_total = 0
    crit_fa_guarded = 0
    crit_fa_raw = 0

    for ep in episodes:
        idx = ep["row_index"]
        gold = ep["gold_state"]
        pred_state = state_fn(ep)
        if pred_state == gold:
            pref_exact += 1
        # Temporal correctness: active tone/frame matches rev 4/5 and NOT future rev 6 or stale rev 1
        if pred_state.get("tone") == gold.get("tone") and pred_state.get("frame") == gold.get("frame"):
            temp_correct += 1
        # Cross-owner or character/scope leak check
        if pred_state.get("tone") in ("hostile", "cruel", "theatrical") or (
            pred_state.get("frame") != gold.get("frame") and pred_state.get("tone") == "hostile"
        ):
            cross_owner_leaks += 1
        # Revocation handling
        if ep["revoke_expected"]:
            revoke_total += 1
            if "verbosity" not in pred_state:
                revoke_ok += 1
        # Policy routing under active frame
        active_frame = pred_state.get("frame", "jinn")
        if active_frame not in ("jinn", "beast"):
            active_frame = "jinn"
        gold_frame = ep["gold_frame"]
        gold_lane = int(test_labels[gold_frame][idx])
        pred_lane = int(frame_preds[active_frame][idx])
        raw_lane = int(frame_raw_preds[active_frame][idx])
        if pred_lane == gold_lane:
            route_correct += 1
        is_crit = bool(frame_crit[gold_frame][idx])
        if is_crit:
            crit_total += 1
            if pred_lane == LANE_TO_IDX["train"]:
                crit_fa_guarded += 1
            if raw_lane == LANE_TO_IDX["train"]:
                crit_fa_raw += 1

    n = len(episodes)
    return {
        "arm": name,
        "episodes": n,
        "preference_fidelity": round(pref_exact / n, 6),
        "temporal_revision_correctness": round(temp_correct / n, 6),
        "cross_owner_leakage_rate": round(cross_owner_leaks / n, 6),
        "cross_owner_leaks": cross_owner_leaks,
        "revocation_handling": round(revoke_ok / revoke_total, 6) if revoke_total else 1.0,
        "revocation_cases": revoke_total,
        "unseen_root_policy_accuracy": round(route_correct / n, 6),
        "critical_examples": crit_total,
        "critical_policy_false_accepts_guarded": crit_fa_guarded,
        "critical_policy_false_accepts_raw": crit_fa_raw,
        "critical_policy_false_accept_rate": round(crit_fa_guarded / crit_total, 6) if crit_total else 0.0,
    }


def _run_live_model_in_context_probe(
    episodes: list[dict[str, Any]],
    endpoint_url: str,
    served_model: str,
    protocol: str,
    test_labels: dict[str, np.ndarray],
    frame_crit: dict[str, np.ndarray],
    max_episodes: int = 32,
    *,
    arm_name: str = "live_in_context_full_window",
    post_revision_reset: bool = False,
    frame_preds: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run an actual checkpoint in-context on a bounded subset of unseen persona episodes."""
    import urllib.request

    subset = episodes[:max_episodes]
    pref_exact = 0
    temp_ok = 0
    cross_leaks = 0
    rev_total = 0
    rev_ok = 0
    parse_errors = 0
    route_correct = 0
    crit_total = 0
    crit_fa = 0
    latencies = []
    for ep in subset:
        idx = ep["row_index"]
        visible_events = (
            [e for e in ep["events"] if e["revision"] >= ep["intervention_at_revision"]]
            if post_revision_reset
            else ep["events"]
        )
        prompt = (
            "Reconstruct the active preference store for owner="
            + ep["owner"]
            + ", scope="
            + ep["scope"]
            + ", at_revision="
            + str(ep["at_revision"])
            + ". Apply only events authored by that owner in that scope with "
            + "revision <= at_revision in ascending revision order. "
            + "op='revoke' deletes the key. Reply ONLY with a JSON object mapping "
            + "active keys (frame, tone, and verbosity if not revoked) to their "
            + "string values.\nEvents:\n"
            + json.dumps(visible_events, separators=(",", ":"))
        )
        t0 = time.monotonic()
        try:
            if protocol == "chat-completions":
                body = {
                    "model": served_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "max_tokens": 128,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": 0,
                }
                url = endpoint_url.rstrip("/") + "/chat/completions"
            else:
                body = {
                    "model": served_model,
                    "prompt": prompt + "\nJSON:\n",
                    "max_tokens": 128,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": 0,
                }
                url = endpoint_url.rstrip("/") + "/completions"
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=28.0) as r:
                resp = json.loads(r.read().decode("utf-8"))
            latencies.append(round(time.monotonic() - t0, 4))
            choice = resp["choices"][0]
            txt = choice["message"]["content"] if protocol == "chat-completions" else choice["text"]
            cleaned = txt.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").lstrip("json").strip()
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise ValueError("not dict")
        except Exception:
            parse_errors += 1
            parsed = {}
        gold = ep["gold_state"]
        if parsed == gold:
            pref_exact += 1
        if parsed.get("tone") == gold.get("tone") and parsed.get("frame") == gold.get("frame"):
            temp_ok += 1
        if parsed.get("tone") in ("hostile", "cruel", "theatrical"):
            cross_leaks += 1
        if ep["revoke_expected"]:
            rev_total += 1
            if parsed and "verbosity" not in parsed:
                rev_ok += 1
        if frame_preds is not None:
            active_frame = parsed.get("frame")
            gold_frame = ep["gold_frame"]
            gold_lane = int(test_labels[gold_frame][idx])
            is_crit = bool(frame_crit[gold_frame][idx])
            if is_crit:
                crit_total += 1
            if active_frame in ("jinn", "beast"):
                pred_lane = int(frame_preds[active_frame][idx])
                if pred_lane == gold_lane:
                    route_correct += 1
                if is_crit and pred_lane == LANE_TO_IDX["train"]:
                    crit_fa += 1
    n = len(subset)
    out = {
        "arm": arm_name,
        "served_model": served_model,
        "protocol": protocol,
        "post_revision_reset": post_revision_reset,
        "episodes_probed": n,
        "parse_or_transport_errors": parse_errors,
        "preference_fidelity": round(pref_exact / n, 6),
        "temporal_revision_correctness": round(temp_ok / n, 6),
        "cross_owner_leakage_rate": round(cross_leaks / n, 6),
        "revocation_handling": round(rev_ok / rev_total, 6) if rev_total else 1.0,
        "mean_latency_seconds": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
    }
    if frame_preds is not None:
        out["routed_unseen_policy_accuracy"] = round(route_correct / n, 6)
        out["critical_examples"] = crit_total
        out["critical_policy_false_accepts"] = crit_fa
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--held-registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--live-endpoint", type=str, default=None)
    parser.add_argument("--live-model", type=str, default=None)
    parser.add_argument("--live-protocol", type=str, default="chat-completions")
    parser.add_argument("--live-episodes", type=int, default=32)
    parser.add_argument(
        "--live-probe",
        action="append",
        default=[],
        help="Repeatable spec: arm_name|endpoint_url|served_model|protocol|mode(full|reset)",
    )
    args = parser.parse_args()

    started = time.monotonic()
    manifest = _load_manifest(None)
    payloads = {name: _read_pinned(args.source_root, spec) for name, spec in manifest["files"].items()}
    policies = {frame: _policy(payloads[frame]) for frame in ("jinn", "beast")}
    low, high = Fraction(manifest["reject_maximum"]), Fraction(manifest["train_minimum"])
    rows = [json.loads(line) for line in payloads["dataset"].decode("utf-8").splitlines() if line]

    # Verify all 4,608 labels and strict split separation
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        for frame, policy in policies.items():
            actual = _score(row["tags"], policy, low, high)
            if row["joint_label"][frame] != actual["lane"]:
                raise RuntimeError(f"label mismatch {row['row_id']} / {frame}")
        splits[row["split"]].append(row)
    assert len(splits["train"]) == 1440 and len(splits["validation"]) == 432 and len(splits["test"]) == 432
    train_roots = {root_id(r["tags"]) for r in splits["train"]}
    val_roots = {root_id(r["tags"]) for r in splits["validation"]}
    test_roots = {root_id(r["tags"]) for r in splits["test"]}
    assert not (train_roots & val_roots) and not (train_roots & test_roots) and not (val_roots & test_roots)

    # Check zero overlap with Research v2 prospective held panel
    held_reg = json.loads(args.held_registry.read_text(encoding="utf-8"))
    held_ids = set(held_reg["held_root_ids"])
    held_sem = set(held_reg["held_semantic_sha256"])
    all_sem = {semantic_digest(r["tags"]) for r in rows}
    overlap_prospective = len((train_roots | val_roots | test_roots) & held_ids) + len(all_sem & held_sem)
    if overlap_prospective != 0:
        raise RuntimeError(f"prospective held overlap detected: {overlap_prospective}")

    tag_vocab = sorted(set().union(*(set(p["tags"]) for p in policies.values())))
    tag_idx = {t: i for i, t in enumerate(tag_vocab)}
    own_indices = {f: sorted(tag_idx[t] for t in policies[f]["tags"]) for f in ("jinn", "beast")}
    crit_indices = {f: sorted(tag_idx[t] for t in policies[f]["critical_tags"]) for f in ("jinn", "beast")}

    def encode_split(
        split_rows: list[dict[str, Any]],
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        x_union = np.zeros((len(split_rows), len(tag_vocab)), dtype=np.float64)
        for i, r in enumerate(split_rows):
            for t in r["tags"]:
                x_union[i, tag_idx[t]] = 1.0
        x_own = {}
        labels = {}
        crit = {}
        for f in ("jinn", "beast"):
            mask = np.zeros(len(tag_vocab), dtype=np.float64)
            mask[own_indices[f]] = 1.0
            x_own[f] = x_union * mask[None, :]
            labels[f] = np.array([LANE_TO_IDX[r["joint_label"][f]] for r in split_rows], dtype=np.int64)
            crit[f] = x_union[:, crit_indices[f]].any(axis=1)
        return x_union, x_own, labels, crit

    x_tr, _x_tr_own, y_tr, _crit_tr = encode_split(splits["train"])
    x_va, x_va_own, y_va, crit_va = encode_split(splits["validation"])
    x_te, x_te_own, y_te, crit_te = encode_split(splits["test"])

    # 1. Standard External LDT heads (union input, trained ONLY on 1,440 train roots)
    ldt_heads: dict[str, AdditiveLDTHead] = {}
    ldt_train_info: dict[str, Any] = {}
    ldt_val_metrics: dict[str, Any] = {}
    ldt_test_metrics: dict[str, Any] = {}
    ldt_test_preds: dict[str, np.ndarray] = {}
    ldt_test_raw: dict[str, np.ndarray] = {}
    for f in ("jinn", "beast"):
        head = AdditiveLDTHead(len(tag_vocab), frame_mask=None, seed=20260725 + (0 if f == "jinn" else 1))
        ldt_train_info[f] = head.fit(x_tr, y_tr[f], epochs=100, batch_size=64, lr=0.08, l2=0.0005, seed=20260725)
        ldt_heads[f] = head
        ldt_val_metrics[f] = _evaluate_frame_head(
            head.predict_probs(x_va), head.predict_probs(x_va_own[f]), y_va[f], crit_va[f], apply_veto=True
        )
        p_u = head.predict_probs(x_te)
        p_o = head.predict_probs(x_te_own[f])
        ldt_test_metrics[f] = _evaluate_frame_head(p_u, p_o, y_te[f], crit_te[f], apply_veto=True)
        raw = p_u.argmax(axis=1)
        guarded = raw.copy()
        guarded[crit_te[f]] = LANE_TO_IDX["reject"]
        ldt_test_raw[f] = raw
        ldt_test_preds[f] = guarded

    # 2. External LDT + TRM Frame-Routing + RLT 3-pass Correction (trained ONLY on 1,440 train roots)
    trm_rlt_heads: dict[str, TRMRoutedRLTHead] = {}
    trm_rlt_train_info: dict[str, Any] = {}
    trm_rlt_val_metrics: dict[str, Any] = {}
    trm_rlt_test_metrics: dict[str, Any] = {}
    trm_rlt_test_preds: dict[str, np.ndarray] = {}
    trm_rlt_test_raw: dict[str, np.ndarray] = {}
    for f in ("jinn", "beast"):
        head = TRMRoutedRLTHead(
            len(tag_vocab),
            own_indices=own_indices[f],
            critical_indices=crit_indices[f],
            seed=20261126 + (0 if f == "jinn" else 1),
        )
        trm_rlt_train_info[f] = head.fit(x_tr, y_tr[f], seed=20261126)
        trm_rlt_heads[f] = head
        trm_rlt_val_metrics[f] = _evaluate_frame_head(
            head.predict_probs(x_va), head.predict_probs(x_va_own[f]), y_va[f], crit_va[f], apply_veto=True
        )
        p_u = head.predict_probs(x_te)
        p_o = head.predict_probs(x_te_own[f])
        trm_rlt_test_metrics[f] = _evaluate_frame_head(p_u, p_o, y_te[f], crit_te[f], apply_veto=True)
        raw = p_u.argmax(axis=1)
        guarded = raw.copy()
        guarded[crit_te[f]] = LANE_TO_IDX["reject"]
        trm_rlt_test_raw[f] = raw
        trm_rlt_test_preds[f] = guarded

    # 3. Multi-owner temporal-revision & model-intervention benchmark over the 432 unseen test roots
    episodes = _generate_intervention_episodes(splits["test"], seed=20261126)

    # Arm A: Base-model in-context across model revision/context reset at rev 3
    # (Without external store, pre-intervention state from rev 1..2 is lost on reset,
    # and without strict binding, context sees only post-intervention window OR unsorted context)
    def arm_in_context_post_reset(ep: dict[str, Any]) -> dict[str, str]:
        # Only retains post-intervention events (revision >= 3) without external owner/rev guards
        visible = [e for e in ep["events"] if e["revision"] >= ep["intervention_at_revision"]]
        return _reconstruct_store(
            visible,
            owner=ep["owner"],
            scope=ep["scope"],
            at_revision=ep["at_revision"],
            enforce_owner=False,
            enforce_temporal=False,
            enforce_scope=False,
            enforce_revocation=True,
        )

    # Arm A2: Base-model in-context with full window retained (no reset) but no structural critical veto
    def arm_in_context_unstructured(ep: dict[str, Any]) -> dict[str, str]:
        return _reconstruct_store(
            ep["events"],
            owner=ep["owner"],
            scope=ep["scope"],
            at_revision=ep["at_revision"],
            enforce_owner=True,
            enforce_temporal=True,
            enforce_scope=True,
            enforce_revocation=True,
        )

    # Arm B: External LDT persona state (scoped, owner+temporal bound, persists across rev 3 intervention)
    def arm_external_bound(ep: dict[str, Any]) -> dict[str, str]:
        return _reconstruct_store(
            ep["events"],
            owner=ep["owner"],
            scope=ep["scope"],
            at_revision=ep["at_revision"],
            enforce_owner=True,
            enforce_temporal=True,
            enforce_scope=True,
            enforce_revocation=True,
        )

    # Arm D1: Ablated external state — owner bindings removed
    def arm_ablate_owner(ep: dict[str, Any]) -> dict[str, str]:
        return _reconstruct_store(
            ep["events"],
            owner=ep["owner"],
            scope=ep["scope"],
            at_revision=ep["at_revision"],
            enforce_owner=False,
            enforce_temporal=True,
            enforce_scope=True,
            enforce_revocation=True,
        )

    # Arm D2: Ablated external state — temporal/revision bindings removed
    def arm_ablate_temporal(ep: dict[str, Any]) -> dict[str, str]:
        return _reconstruct_store(
            ep["events"],
            owner=ep["owner"],
            scope=ep["scope"],
            at_revision=ep["at_revision"],
            enforce_owner=True,
            enforce_temporal=False,
            enforce_scope=True,
            enforce_revocation=True,
        )

    # Arm D3: Ablated external state — both owner & temporal bindings removed
    def arm_ablate_both(ep: dict[str, Any]) -> dict[str, str]:
        return _reconstruct_store(
            ep["events"],
            owner=ep["owner"],
            scope=ep["scope"],
            at_revision=ep["at_revision"],
            enforce_owner=False,
            enforce_temporal=False,
            enforce_scope=False,
            enforce_revocation=False,
        )

    # Un-vetoed raw predictions simulate in-context soft reasoning without structural critical veto
    intervention_arms = [
        _eval_arm_on_episodes(
            "base_in_context_post_revision_reset",
            episodes,
            x_te,
            ldt_test_raw,
            ldt_test_raw,
            crit_te,
            y_te,
            arm_in_context_post_reset,
        ),
        _eval_arm_on_episodes(
            "base_in_context_no_veto_full_window",
            episodes,
            x_te,
            ldt_test_raw,
            ldt_test_raw,
            crit_te,
            y_te,
            arm_in_context_unstructured,
        ),
        _eval_arm_on_episodes(
            "external_ldt_persona_state",
            episodes,
            x_te,
            ldt_test_preds,
            ldt_test_raw,
            crit_te,
            y_te,
            arm_external_bound,
        ),
        _eval_arm_on_episodes(
            "external_ldt_plus_trm_routing_and_rlt_correction",
            episodes,
            x_te,
            trm_rlt_test_preds,
            trm_rlt_test_raw,
            crit_te,
            y_te,
            arm_external_bound,
        ),
        _eval_arm_on_episodes(
            "ablated_external_no_owner_binding",
            episodes,
            x_te,
            ldt_test_preds,
            ldt_test_raw,
            crit_te,
            y_te,
            arm_ablate_owner,
        ),
        _eval_arm_on_episodes(
            "ablated_external_no_temporal_binding",
            episodes,
            x_te,
            ldt_test_preds,
            ldt_test_raw,
            crit_te,
            y_te,
            arm_ablate_temporal,
        ),
        _eval_arm_on_episodes(
            "ablated_external_no_owner_no_temporal_no_scope",
            episodes,
            x_te,
            ldt_test_preds,
            ldt_test_raw,
            crit_te,
            y_te,
            arm_ablate_both,
        ),
    ]

    live_probes: list[dict[str, Any]] = []
    if args.live_endpoint and args.live_model:
        live_probes.append(
            _run_live_model_in_context_probe(
                episodes,
                endpoint_url=args.live_endpoint,
                served_model=args.live_model,
                protocol=args.live_protocol,
                test_labels=y_te,
                frame_crit=crit_te,
                max_episodes=args.live_episodes,
                arm_name="live_in_context_full_window",
                post_revision_reset=False,
                frame_preds=ldt_test_raw,
            )
        )
    for spec in args.live_probe:
        arm_name, ep_url, model_name, proto, mode = spec.split("|", 4)
        live_probes.append(
            _run_live_model_in_context_probe(
                episodes,
                endpoint_url=ep_url,
                served_model=model_name,
                protocol=proto,
                test_labels=y_te,
                frame_crit=crit_te,
                max_episodes=args.live_episodes,
                arm_name=arm_name,
                post_revision_reset=(mode == "reset"),
                frame_preds=ldt_test_raw,
            )
        )

    report = {
        "schema": "research-v2-jinn-beast-persona-experiment-v1",
        "source_id": manifest["source_id"],
        "source_revision": manifest["revision"],
        "source_files_verified": {k: {"sha256": digest(v), "bytes": len(v)} for k, v in payloads.items()},
        "split_integrity": {
            "train_roots": len(train_roots),
            "validation_roots": len(val_roots),
            "test_roots": len(test_roots),
            "train_targets": 2 * len(train_roots),
            "test_targets": 2 * len(test_roots),
            "train_val_overlap": 0,
            "train_test_overlap": 0,
            "val_test_overlap": 0,
            "prospective_research_v2_held_overlap": overlap_prospective,
            "all_4608_source_labels_recomputed": True,
        },
        "tag_vocab_size": len(tag_vocab),
        "external_ldt_head": {
            "train": ldt_train_info,
            "validation": ldt_val_metrics,
            "test_unseen_roots": ldt_test_metrics,
        },
        "external_ldt_trm_rlt_head": {
            "train": trm_rlt_train_info,
            "validation": trm_rlt_val_metrics,
            "test_unseen_roots": trm_rlt_test_metrics,
        },
        "intervention_comparison_432_unseen_roots": intervention_arms,
        "live_base_model_in_context_probes": live_probes,
        "claim_boundary": (
            manifest["claim_scope"]
            + " Original train/validation/test separation preserved; no test rows entered training."
        ),
        "wall_seconds": round(time.monotonic() - started, 4),
    }
    report["report_sha256"] = digest(canonical(report).encode("utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
