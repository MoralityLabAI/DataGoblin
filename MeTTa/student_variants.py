"""Student variants for the stop-versus-generation study (NumPy, CPU only).

One weight-tied tanh loop (the AIRIS-02 TRM-style student) is extended with
optional heads and inputs. Every variant trains with the same optimizer, step
count, learning rate, hidden width, unroll depth and TRAIN presentations as the
original student; the plain spec reproduces RecurrentStudent exactly.

Variants (VariantSpec fields):
  stop      class    STOP is action class 20 in the action head (original)
            head     separate sigmoid halting head on the loop state
            verifier halting head also reading public verifier state features
            budget   halting by a learned remaining-edit regressor (stop at 0)
  aux       auxiliary heads: per-action "reduces distance to optimum" labels
            and remaining-edit regression, trained through the shared loop
  curriculum  STOP rows enter the action loss only after a delay, then ramp in
  memory    previous action, step counter and edited-item history as input
  skip      linear skip path (LDT proxy proposal) plus recurrent refinement,
            routed through the environment veto
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .airis02_student import (
    ACTION_CLASSES,
    INPUT_DIM,
    MAX_ITEMS,
    _clip,
    _cross_entropy,
    _matrix,
    action_index,
    encode_public,
    legal_action_mask,
)
from .sft_oracle import evaluate_subset, next_edit
from .stop_diagnostics import distance, masked_edit, optimum_of

MEM_DIM = ACTION_CLASSES + 1 + MAX_ITEMS
VER_DIM = 7


@dataclass(frozen=True)
class VariantSpec:
    name: str
    stop: str = "class"          # class | head | verifier | budget
    aux: bool = False
    curriculum: bool = False
    memory: bool = False
    skip: bool = False
    guard: bool = False


VARIANTS: dict[str, VariantSpec] = {
    "plain_class_stop": VariantSpec("plain_class_stop"),
    "sep_stop_head": VariantSpec("sep_stop_head", stop="head"),
    "aux_correctness": VariantSpec("aux_correctness", stop="head", aux=True),
    "delayed_stop_curriculum": VariantSpec("delayed_stop_curriculum", curriculum=True),
    "verifier_conditioned_halt": VariantSpec("verifier_conditioned_halt", stop="verifier"),
    "explicit_memory": VariantSpec("explicit_memory", memory=True),
    "ldt_trm_hybrid": VariantSpec("ldt_trm_hybrid", skip=True, guard=True),
    "budget_aware_halt": VariantSpec("budget_aware_halt", stop="budget", aux=True),
}


def memory_features(ctx: dict[str, Any], selected: list[str]) -> np.ndarray:
    vec = np.zeros(MEM_DIM)
    if ctx["prev_action"] is not None:
        vec[ctx["prev_action"]] = 1.0
    vec[ACTION_CLASSES] = ctx["step"] / 8.0
    for ident in set(selected) ^ set(ctx["initial"]):
        vec[ACTION_CLASSES + 1 + int(ident[1:])] = 1.0
    return vec


def verifier_features(task: dict, selected: list[str]) -> np.ndarray:
    state = evaluate_subset(task, selected)
    goals = max(1, len(task["goals"]))
    return np.array([
        float(state["valid"]), len(state["violations"]) / 7.0,
        len(state["covered_goals"]) / goals, len(state["uncovered_goals"]) / goals,
        state["total_tokens"] / max(1, task["token_budget"]),
        len(selected) / max(1, task["max_items"]), state["total_quality"] / 60.0,
    ])


def input_vector(spec: VariantSpec, task: dict, selected: list[str], ctx: dict) -> np.ndarray:
    base = encode_public(task, selected)
    return np.concatenate([base, memory_features(ctx, selected)]) if spec.memory else base


def attach_contexts(train: list[dict]) -> list[dict]:
    """Per-presentation episode context reconstructed from the root's own views."""
    by_root: dict[str, list[dict]] = {}
    for row in train:
        by_root.setdefault(row["root"], []).append(row)
    out = []
    for rows in by_root.values():
        rows = sorted(rows, key=lambda r: r["presentation"])
        initial = rows[0]["selected"]
        opt = optimum_of(rows[0])
        previous = None
        for row in rows:
            prev = None
            if previous is not None:
                action = next_edit(previous["selected"], opt)
                prev = action_index(action) if action["op"] != "stop" else None
            out.append({**row, "ctx": {"initial": list(initial), "prev_action": prev,
                                       "step": row["presentation"]}})
            previous = row
    return out


def good_move_matrix(rows: list[dict]) -> np.ndarray:
    """1 where the legal action reduces the symmetric difference to the optimum."""
    out = np.zeros((len(rows), ACTION_CLASSES))
    for n, row in enumerate(rows):
        opt = set(optimum_of(row))
        chosen = set(row["selected"])
        for item in row["public_world"]["items"]:
            i = int(item["id"][1:])
            if item["id"] in chosen:
                out[n, MAX_ITEMS + i] = float(item["id"] not in opt)
            else:
                out[n, i] = float(item["id"] in opt)
        out[n, 20] = float(chosen == opt)
    return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


class VariantNet:
    def __init__(self, seed: int, spec: VariantSpec, hidden: int = 12):
        self.spec = spec
        self.in_dim = INPUT_DIM + (MEM_DIM if spec.memory else 0)
        rng = np.random.default_rng(seed)
        H = hidden
        # Same draw order and scales as RecurrentStudent so the plain spec matches it.
        self.wx = rng.normal(0, 0.085 / math.sqrt(self.in_dim), size=(self.in_dim, H))
        self.wh = rng.normal(0, 0.15 / math.sqrt(H), size=(H, H))
        self.wo = rng.normal(0, 0.12 / math.sqrt(H), size=(H, ACTION_CLASSES))
        self.bx = np.zeros(H)
        self.bo = np.zeros(ACTION_CLASSES)
        self.extra: dict[str, np.ndarray] = {}
        if spec.skip:
            self.extra["wl"] = rng.normal(0, 0.045, size=(self.in_dim, ACTION_CLASSES))
        if spec.stop in ("head", "verifier"):
            self.extra["ws"] = rng.normal(0, 0.12 / math.sqrt(H), size=(H,))
            self.extra["bs"] = np.zeros(())
        if spec.stop == "verifier":
            self.extra["wv"] = rng.normal(0, 0.02, size=(VER_DIM,))
        if spec.aux:
            self.extra["wa"] = rng.normal(0, 0.12 / math.sqrt(H), size=(H, ACTION_CLASSES))
            self.extra["ba"] = np.zeros(ACTION_CLASSES)
            self.extra["wr"] = rng.normal(0, 0.12 / math.sqrt(H), size=(H,))
            self.extra["br"] = np.zeros(())
        self.threshold: float | None = None

    @property
    def params(self) -> int:
        base = sum(p.size for p in (self.wx, self.wh, self.wo, self.bx, self.bo))
        return base + sum(p.size for p in self.extra.values())

    def forward(self, x: np.ndarray, loops: int):
        drive = x @ self.wx + self.bx
        states = [np.zeros((len(x), self.wh.shape[0]))]
        for _ in range(loops):
            states.append(np.tanh(drive + states[-1] @ self.wh))
        h = states[-1]
        logits = h @ self.wo + self.bo
        if self.spec.skip:
            logits = logits + x @ self.extra["wl"]
        return logits, states

    def heads(self, h: np.ndarray, v: np.ndarray | None):
        out: dict[str, np.ndarray] = {}
        if "ws" in self.extra:
            z = h @ self.extra["ws"] + self.extra["bs"]
            if v is not None and "wv" in self.extra:
                z = z + v @ self.extra["wv"]
            out["stop"] = _sigmoid(z)
        if "wa" in self.extra:
            out["aux"] = _sigmoid(h @ self.extra["wa"] + self.extra["ba"])
            out["rem"] = h @ self.extra["wr"] + self.extra["br"]
        return out

    def fit(self, x, y, masks, *, steps, lr, loops=4, v=None, good=None, rem=None,
            stop_rows=None) -> list[float]:
        n = len(x)
        losses = []
        stop_rows = (y == 20) if stop_rows is None else stop_rows
        for step in range(steps):
            logits, states = self.forward(x, loops)
            h = states[-1]
            weights = np.ones(n)
            if self.spec.stop != "class":
                weights[stop_rows] = 0.0  # halting is not an edit class
            elif self.spec.curriculum:
                start = 0.5 * steps
                weights[stop_rows] = float(np.clip((step - start) / (0.25 * steps), 0.0, 1.0))
            # weighted masked cross entropy
            masked = np.where(masks, logits, -1e6)
            shifted = masked - masked.max(axis=1, keepdims=True)
            probs = np.exp(np.clip(shifted, -700, 0))
            probs /= probs.sum(axis=1, keepdims=True)
            loss = float(-(weights * np.log(np.maximum(probs[np.arange(n), y], 1e-16))).sum() / n)
            dlog = probs.copy()
            dlog[np.arange(n), y] -= 1.0
            dlog *= (weights / n)[:, None]
            dh = dlog @ self.wo.T
            grads: dict[str, np.ndarray] = {}
            grads["wo"] = h.T @ dlog + 1e-4 * self.wo
            grads["bo"] = dlog.sum(axis=0)
            if self.spec.skip:
                grads["wl"] = x.T @ dlog + 1e-4 * self.extra["wl"]
            heads = self.heads(h, v)
            if "stop" in heads:
                ds = (heads["stop"] - stop_rows.astype(float)) / n
                grads["ws"] = h.T @ ds + 1e-4 * self.extra["ws"]
                grads["bs"] = np.asarray(ds.sum())
                if "wv" in self.extra:
                    grads["wv"] = v.T @ ds + 1e-4 * self.extra["wv"]
                dh = dh + ds[:, None] * self.extra["ws"][None, :]
            if "aux" in heads:
                legal = masks.astype(float)
                da = (heads["aux"] - good) * legal / (n * np.maximum(legal.sum(1, keepdims=True), 1.0))
                grads["wa"] = h.T @ da + 1e-4 * self.extra["wa"]
                grads["ba"] = da.sum(axis=0)
                dh = dh + da @ self.extra["wa"].T
                dr = (heads["rem"] - rem) / n
                grads["wr"] = h.T @ dr + 1e-4 * self.extra["wr"]
                grads["br"] = np.asarray(dr.sum())
                dh = dh + dr[:, None] * self.extra["wr"][None, :]
            dwx = np.zeros_like(self.wx)
            dwh = np.zeros_like(self.wh)
            dbx = np.zeros_like(self.bx)
            for j in range(loops, 0, -1):
                dz = dh * (1.0 - states[j] * states[j])
                dwx += x.T @ dz
                dwh += states[j - 1].T @ dz
                dbx += dz.sum(axis=0)
                dh = dz @ self.wh.T
            dwx += 1e-4 * self.wx
            dwh += 1e-4 * self.wh
            names = ["wx", "wh", "wo", "bx", "bo", *self.extra]
            raw = {"wx": dwx, "wh": dwh, "wo": grads["wo"], "bx": dbx, "bo": grads["bo"],
                   **{k: grads[k] for k in self.extra}}
            clipped = _clip([np.atleast_1d(raw[k]) for k in names])
            eta = lr / (1.0 + step / max(1.0, steps))
            for name, g in zip(names, clipped, strict=True):
                target = getattr(self, name) if hasattr(self, name) and name in (
                    "wx", "wh", "wo", "bx", "bo") else self.extra[name]
                target -= eta * g.reshape(target.shape)
            losses.append(round(loss, 6))
        return losses

    def calibrate_stop(self, x, v, stop_rows, max_fpr: float = 0.05) -> None:
        """Freeze a TRAIN-only halting threshold with the StopHead rule."""
        _, states = self.forward(x, 4)
        p = self.heads(states[-1], v)["stop"]
        negatives = np.sort(p[~stop_rows])
        allowed = math.floor(max_fpr * len(negatives))
        limit = float(negatives[len(negatives) - allowed - 1])
        self.threshold = float(np.nextafter(max(limit, 0.2), np.inf))


class VariantPolicy:
    def __init__(self, net: VariantNet, *, loops: int = 4):
        self.net, self.loops, self.spec = net, loops, net.spec
        self.name = net.spec.name

    def _forward(self, task, selected, ctx):
        x = input_vector(self.spec, task, selected, ctx)[None, :]
        logits, states = self.net.forward(x, self.loops)
        return x, logits[0], states[-1]

    def edit_action(self, task, selected, ctx):
        _, logits, _ = self._forward(task, selected, ctx)
        return masked_edit(logits, task, selected, self.spec.guard)

    def wants_stop(self, task, selected, ctx) -> bool:
        _, logits, h = self._forward(task, selected, ctx)
        if self.spec.stop == "class":
            mask = legal_action_mask(task, selected)
            return int(np.argmax(np.where(mask, logits, -1e6))) == 20
        if self.spec.stop == "budget":
            return bool(self.net.heads(h, None)["rem"][0] * 10.0 < 0.5)
        v = verifier_features(task, selected)[None, :] if self.spec.stop == "verifier" else None
        return bool(self.net.heads(h, v)["stop"][0] >= self.net.threshold)

    def own_action(self, task, selected, ctx):
        if self.spec.stop == "class":
            _, logits, _ = self._forward(task, selected, ctx)
            return masked_edit(logits, task, selected, self.spec.guard, allow_stop=True)
        if self.wants_stop(task, selected, ctx):
            return {"op": "stop", "id": "none"}, 0, 0
        return self.edit_action(task, selected, ctx)


def fit_variant(spec: VariantSpec, train: list[dict], *, seed: int, steps: int,
                lr: float = 0.20, hidden: int = 12, loops: int = 4) -> VariantPolicy:
    rows = attach_contexts(train)
    x_base, y, masks = _matrix(rows)
    if spec.memory:
        mem = np.stack([memory_features(r["ctx"], r["selected"]) for r in rows])
        x = np.concatenate([x_base, mem], axis=1)
    else:
        x = x_base
    stop_rows = y == 20
    v = np.stack([verifier_features(r["public_world"], r["selected"]) for r in rows]) \
        if spec.stop == "verifier" else None
    good = good_move_matrix(rows) if spec.aux else None
    rem = np.array([distance(r["selected"], optimum_of(r)) / 10.0 for r in rows]) if spec.aux else None
    net = VariantNet(seed, spec, hidden=hidden)
    net.fit(x, y, masks, steps=steps, lr=lr, loops=loops, v=v, good=good, rem=rem,
            stop_rows=stop_rows)
    if spec.stop in ("head", "verifier"):
        net.calibrate_stop(x, v, stop_rows)
    return VariantPolicy(net, loops=loops)


def extra_train_rows(roots_per_family: int, *, full_trajectory: bool,
                     start: int | None = None) -> list[dict]:
    """TRAIN rows beyond the bundle cap; seeds stay in the train band (below dev)."""
    from .ram_hillclimb import DEV_START, FAMILIES, TRAIN_START, make_row, training_views
    from .sft_oracle import edit, first_state, make_task

    start = TRAIN_START if start is None else start
    if start + roots_per_family > DEV_START:
        raise ValueError("train roots would reach the development seed band")
    rows = []
    for family in FAMILIES:
        for seed in range(start, start + roots_per_family):
            task = make_task(family, seed)
            if not full_trajectory:
                views = training_views(task, seed)
            else:
                opt = list(__import__("MeTTa.sft_oracle", fromlist=["x"]).exact_optimum(task)["selected"])
                state = list(first_state(task, seed))
                views = [state]
                while True:
                    action = next_edit(state, opt)
                    if action["op"] == "stop":
                        break
                    state = edit(state, action)
                    views.append(list(state))
            for index, state in enumerate(views):
                rows.append(make_row(family, seed, state, "train_candidate", index))
    return rows
