"""Separate solution generation from stopping for the MeTTa RAM hill-climb students.

A trained student is rolled out three ways on the same cases:
  never   - STOP is masked out and the student edits until a fixed horizon;
  oracle  - the episode ends the first time the selection equals the exact
            optimum (the environment decides when to stop);
  own     - the student's own stopping mechanism, exactly as the existing
            evaluator runs it.
Every visited state is labelled solved/unsolved, and the student's edit is
compared with the canonical next edit. If the student never reaches a solved
state under the first two arms, stopping changes cannot produce a solved case.

Existing behaviour of ram_hillclimb.py and airis02_*.py is untouched; this module
only imports from them.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

from collections import defaultdict
from typing import Any

import numpy as np

from .airis02_student import (
    ACTION_CLASSES,
    _next_action,
    decode_action,
    encode_public,
    legal_action_mask,
)
from .ram_hillclimb import FAMILIES, make_row
from .sft_oracle import edit, evaluate_subset, exact_optimum, first_state, make_task, next_edit

FRESH_START = 50000
NEVER_HORIZON = 16
OWN_HORIZON = 8
_OPT_CACHE: dict[tuple[str, int], list[str]] = {}


def optimum_of(row: dict[str, Any]) -> list[str]:
    key = (row["family"], row["seed"])
    if key not in _OPT_CACHE:
        _OPT_CACHE[key] = list(exact_optimum(row["public_world"])["selected"])
    return _OPT_CACHE[key]


def fresh_dev_rows(per_family: int, start: int = FRESH_START) -> list[dict[str, Any]]:
    """Dev-style first-state rows on seeds disjoint from every bundle range."""
    from .ram_hillclimb import DEV_START, PUBLIC_TEST_START, TRAIN_START

    if start < PUBLIC_TEST_START + 1000 or start <= max(TRAIN_START, DEV_START):
        raise ValueError("fresh seeds must lie above the train, dev and public-test ranges")
    rows = []
    for family in FAMILIES:
        for seed in range(start, start + per_family):
            task = make_task(family, seed)
            rows.append(make_row(family, seed, first_state(task, seed), "fresh_development", 0))
    return rows


def initial_context(selected: list[str]) -> dict[str, Any]:
    return {"initial": list(selected), "prev_action": None, "step": 0}


def advance_context(ctx: dict[str, Any], action: dict[str, str]) -> dict[str, Any]:
    from .airis02_student import action_index

    return {"initial": ctx["initial"], "prev_action": action_index(action), "step": ctx["step"] + 1}


class ExistingPolicy:
    """Wrap an existing LinearStudent/RecurrentStudent (+ optional StopHead)."""

    def __init__(self, model: Any, *, loops: int, stop_head: Any | None, guard: bool, name: str):
        self.model, self.loops, self.stop_head, self.guard, self.name = (
            model, loops, stop_head, guard, name,
        )

    def _logits(self, task: dict, selected: list[str], ctx: dict) -> np.ndarray:
        vector = encode_public(task, selected)[None, :]
        return self.model.logits(vector, loops=self.loops)[0]

    def own_action(self, task, selected, ctx):
        return _next_action(
            self.model, task, selected, loops=self.loops,
            stop_head=self.stop_head, verifier_guard=self.guard,
        )

    def wants_stop(self, task, selected, ctx) -> bool:
        if self.stop_head is not None:
            return bool(self.stop_head.should_stop(encode_public(task, selected)[None, :]))
        mask = legal_action_mask(task, selected)
        logits = self._logits(task, selected, ctx)
        return int(np.argmax(np.where(mask, logits, -1e6))) == 20

    def edit_action(self, task, selected, ctx):
        return masked_edit(self._logits(task, selected, ctx), task, selected, self.guard)


def masked_edit(logits: np.ndarray, task: dict, selected: list[str], guard: bool,
                allow_stop: bool = False):
    """Best legal action (STOP masked out unless allowed); optional environment veto."""
    allowed = legal_action_mask(task, selected)
    allowed[20] = allow_stop
    if not guard:
        index = int(np.argmax(np.where(allowed, logits, -1e6)))
        return decode_action(index), 0, 0
    vetoes = queries = 0
    before = None
    for index in np.argsort(-logits, kind="stable"):
        if not allowed[index]:
            continue
        action = decode_action(int(index))
        if action["op"] == "add":
            if before is None:
                before = set(evaluate_subset(task, selected)["violations"])
                queries += 1
            after = set(evaluate_subset(task, edit(selected, action))["violations"])
            queries += 1
            if after - before:
                vetoes += 1
                continue
        return action, vetoes, queries
    return {"op": "abstain", "id": "none"}, vetoes, queries


def train_accuracy(policy, rows: list[dict[str, Any]]) -> float:
    """Fraction of non-terminal TRAIN presentations where the edit matches the label."""
    hit = total = 0
    for row in rows:
        if row["oracle_next_edit"]["op"] == "stop":
            continue
        action, _, _ = policy.edit_action(row["public_world"], row["selected"],
                                          row.get("ctx") or initial_context(row["selected"]))
        hit += int(action == row["oracle_next_edit"])
        total += 1
    return round(hit / total, 4)


def distance(selected: list[str], opt: list[str]) -> int:
    return len(set(selected) ^ set(opt))


def run_episode(policy, row: dict, mode: str) -> dict[str, Any]:
    """mode in {never, oracle, own}. Records every visited state."""
    task = row["public_world"]
    opt = optimum_of(row)
    selected = list(row["selected"])
    ctx = initial_context(selected)
    horizon = NEVER_HORIZON if mode in ("never", "oracle") else OWN_HORIZON
    steps = []
    stopped_at_solved = premature = False
    first_solved = None
    for k in range(horizon):
        solved = selected == opt
        if solved and first_solved is None:
            first_solved = k
        canon = next_edit(selected, opt)
        if mode == "oracle" and solved:
            break
        if mode == "own":
            action, _, _ = policy.own_action(task, selected, ctx)
        else:
            action, _, _ = policy.edit_action(task, selected, ctx)
        steps.append({
            "k": k, "solved": solved, "action": action, "canonical": canon,
            "match": action == canon, "remaining": distance(selected, opt),
        })
        if action["op"] == "stop":
            stopped_at_solved = solved
            premature = not solved
            break
        if action["op"] == "abstain":
            break
        selected = edit(selected, action)
        steps[-1]["progress"] = distance(selected, opt) < steps[-1]["remaining"]
        ctx = advance_context(ctx, action)
    else:
        solved = selected == opt
        if solved and first_solved is None:
            first_solved = horizon
    final_solved = selected == opt
    return {
        "steps": steps, "first_solved": first_solved, "final_solved": final_solved,
        "stopped_at_solved": stopped_at_solved, "premature": premature,
        "ever_solved": first_solved is not None,
    }


def canonical_states(row: dict) -> list[tuple[list[str], dict, int]]:
    """Teacher-forced trajectory: (state, ctx, remaining) from start to optimum inclusive."""
    opt = optimum_of(row)
    selected = list(row["selected"])
    ctx = initial_context(selected)
    out = [(list(selected), ctx, distance(selected, opt))]
    for _ in range(NEVER_HORIZON):
        action = next_edit(selected, opt)
        if action["op"] == "stop":
            break
        selected = edit(selected, action)
        ctx = advance_context(ctx, action)
        out.append((list(selected), ctx, distance(selected, opt)))
    return out


def diagnose(policy, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Full generation-versus-stopping measurement for one trained policy."""
    n = len(rows)
    acc: dict[str, Any] = defaultdict(int)
    onpol_by_k: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    tf_by_k: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    cal_can = [0, 0, 0, 0]  # tp fn fp tn on canonical states
    cal_pol = [0, 0, 0, 0]  # on states visited by the never-stop rollout
    for row in rows:
        opt = optimum_of(row)
        acc["initial_solved"] += int(row["selected"] == opt)
        never = run_episode(policy, row, "never")
        oracle = run_episode(policy, row, "oracle")
        own = run_episode(policy, row, "own")
        acc["ever_reached_optimum"] += int(never["ever_solved"])
        acc["reached_after_edit"] += int(never["ever_solved"] and row["selected"] != opt)
        acc["reached_within_8"] += int(never["first_solved"] is not None and never["first_solved"] <= 8)
        acc["solved_oracle_stop"] += int(oracle["final_solved"])
        acc["solved_oracle_stop_nontrivial"] += int(oracle["final_solved"] and row["selected"] != opt)
        acc["solved_and_stopped"] += int(own["stopped_at_solved"])
        acc["premature_stops"] += int(own["premature"])
        acc["own_never_stopped"] += int(
            not own["stopped_at_solved"] and not own["premature"]
        )
        # unnecessary visits: steps taken while the state was already solved
        acc["unnecessary_visits_never"] += sum(s["solved"] for s in never["steps"])
        acc["unnecessary_visits_own"] += sum(s["solved"] and s["action"]["op"] != "stop"
                                             for s in own["steps"])
        for s in never["steps"]:
            if s["solved"]:
                break
            onpol_by_k[s["k"]][0] += int(s["match"])
            onpol_by_k[s["k"]][1] += 1
            acc["onpolicy_states"] += 1
            acc["onpolicy_match"] += int(s["match"])
            acc["onpolicy_progress"] += int(s.get("progress", False))
        # teacher-forced canonical states and stop calibration
        traj = canonical_states(row)
        task = row["public_world"]
        for k, (state, ctx, remaining) in enumerate(traj):
            solved = state == opt
            stop = policy.wants_stop(task, state, ctx)
            cal_can[(0 if stop else 1) if solved else (2 if stop else 3)] += 1
            if solved:
                continue
            action, _, _ = policy.edit_action(task, state, ctx)
            ok = action == next_edit(state, opt)
            tf_by_k[k][0] += int(ok)
            tf_by_k[k][1] += 1
            acc["tf_states"] += 1
            acc["tf_match"] += int(ok)
            after = edit(state, action) if action["op"] in ("add", "remove") else state
            acc["tf_progress"] += int(distance(after, opt) < remaining)
        # on-policy stop calibration: states visited by the never-stop rollout
        walk = list(row["selected"])
        ctx = initial_context(walk)
        for s in never["steps"]:
            stop = policy.wants_stop(task, walk, ctx)
            solved = walk == opt
            cal_pol[(0 if stop else 1) if solved else (2 if stop else 3)] += 1
            if s["action"]["op"] not in ("add", "remove"):
                break
            walk = edit(walk, s["action"])
            ctx = advance_context(ctx, s["action"])
    def ratio(a: int, b: int) -> float | None:
        return round(a / b, 4) if b else None

    return {
        "cases": n,
        "initial_solved": acc["initial_solved"],
        "ever_reached_optimum": acc["ever_reached_optimum"],
        "reached_after_edit": acc["reached_after_edit"],
        "reached_within_8": acc["reached_within_8"],
        "solved_oracle_stop": acc["solved_oracle_stop"],
        "solved_oracle_stop_nontrivial": acc["solved_oracle_stop_nontrivial"],
        "solved_and_stopped": acc["solved_and_stopped"],
        "premature_stops": acc["premature_stops"],
        "own_never_stopped": acc["own_never_stopped"],
        "unnecessary_visits_never": acc["unnecessary_visits_never"],
        "unnecessary_visits_own": acc["unnecessary_visits_own"],
        "onpolicy_correct": ratio(acc["onpolicy_match"], acc["onpolicy_states"]),
        "onpolicy_progress": ratio(acc["onpolicy_progress"], acc["onpolicy_states"]),
        "onpolicy_by_step": {k: [v[0], v[1]] for k, v in sorted(onpol_by_k.items())},
        "teacher_forced_correct": ratio(acc["tf_match"], acc["tf_states"]),
        "teacher_forced_progress": ratio(acc["tf_progress"], acc["tf_states"]),
        "teacher_forced_by_step": {k: [v[0], v[1]] for k, v in sorted(tf_by_k.items())},
        "stop_calibration_canonical": {
            "tp": cal_can[0], "fn": cal_can[1], "fp": cal_can[2], "tn": cal_can[3],
            "stop_rate_at_solved": ratio(cal_can[0], cal_can[0] + cal_can[1]),
            "stop_rate_at_unsolved": ratio(cal_can[2], cal_can[2] + cal_can[3]),
        },
        "stop_calibration_onpolicy": {
            "tp": cal_pol[0], "fn": cal_pol[1], "fp": cal_pol[2], "tn": cal_pol[3],
            "stop_rate_at_solved": ratio(cal_pol[0], cal_pol[0] + cal_pol[1]),
            "stop_rate_at_unsolved": ratio(cal_pol[2], cal_pol[2] + cal_pol[3]),
        },
    }


def fit_existing(arm: str, config: dict[str, Any], bundle: dict[str, Any], *, seed: int):
    """Train through the unchanged fit_candidate path and rebuild its stop head."""
    from .airis02_stop import StopHead
    from .airis02_student import LinearStudent, RecurrentStudent, _matrix
    from .ram_hillclimb import OPTIONS, fit_candidate

    if config["steps"] in OPTIONS[arm]["steps"]:
        model, _ = fit_candidate(arm, config, bundle["train"], bundle["dev"],
                                 seed=seed, ram_limit_mib=1024)
    else:  # step budgets beyond the registered grid use the same fitting calls directly
        x, y, masks = _matrix(bundle["train"])
        if arm == "trm":
            model = RecurrentStudent(seed, hidden=config["hidden"])
            model.fit(x, y, legal_masks=masks, steps=config["steps"], lr=config["lr"],
                      train_loops=config["train_loops"])
        else:
            model = LinearStudent(seed)
            model.fit(x, y, legal_masks=masks, steps=config["steps"], lr=config["lr"])
    stop_head = None
    if config["stop_policy"] == "train_calibrated":
        x, y, _m = _matrix(bundle["train"])
        stop_head = StopHead(seed)
        stop_head.fit(x, y == 20, steps=config["steps"], lr=config["lr"])
    loops = config["inference_loops"] if arm == "trm" else 1
    return ExistingPolicy(model, loops=loops, stop_head=stop_head, guard=(arm == "ldt_proxy"),
                          name=f"{arm}:{config['stop_policy']}")


def main() -> int:
    import argparse
    import json
    from pathlib import Path

    from .ram_hillclimb import DEFAULTS, make_bundle, peak_rss_mib

    parser = argparse.ArgumentParser(description="Generation-versus-stopping diagnosis")
    parser.add_argument("--out", type=Path, default=Path("runs/metta/stop_q1.json"))
    parser.add_argument("--train-roots", type=int, default=24)
    parser.add_argument("--fresh-per-family", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261122, 20261123, 20261124])
    args = parser.parse_args()
    bundle = make_bundle(args.train_roots, 8)
    sets = {
        "bundle_dev_8pf": bundle["dev"],
        f"fresh_{args.fresh_per_family}pf": fresh_dev_rows(args.fresh_per_family),
    }
    configs = {
        "default": lambda arm: dict(DEFAULTS[arm]),
        "default_train_calibrated": lambda arm: {**DEFAULTS[arm], "stop_policy": "train_calibrated"},
        "long_steps32_lr0.2": lambda arm: {**DEFAULTS[arm], "steps": 32},
    }
    result: dict[str, Any] = {}
    for cname, make in configs.items():
        for arm in ("trm", "ldt_proxy"):
            for seed in args.seeds:
                policy = fit_existing(arm, make(arm), bundle, seed=seed)
                for sname, rows in sets.items():
                    result[f"{cname}|{arm}|{seed}|{sname}"] = diagnose(policy, rows)
    result["peak_rss_mib"] = peak_rss_mib()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"peak_rss_mib": result["peak_rss_mib"], "rows": len(result) - 1}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
