"""Compare hard and tied action targets for a native-verified evidence docket."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import torch
from research_gym.neural.obligation_pointer import ObligationPointer

from metta_research_bench.looped.model import BATCH_SIZE, BUCKETS, FIELDS, fit

from .evidence_budget_confirmation import primary_intervals
from .evidence_docket import BUDGETS, DOCKET_COMMIT, NativeDocket, action_values, rollout, topology
from .evidence_docket_pilot import (
    EVAL as PRIOR_EVAL,
)
from .evidence_docket_pilot import (
    TRAIN,
    allocate,
    committed_checkout,
    open_cases,
    readout,
    seal,
    teacher_actions,
    write_jsonl,
)
from .evidence_incidence_pilot import SLOTS
from .ram_hillclimb import canonical, digest, install_memory_ceiling, peak_rss_mib

EVAL = {
    "narrow_fallback": (((0,), (1, 2)), ((1,), (3, 4))),
    "shared_catalog": (((0,), (1, 2, 3)), ((4,), (1, 2, 3))),
    "angled_bridge": (((0, 1), (2, 3), (4, 6)), ((0, 2), (2, 4), (3, 5))),
    "overlap_square": (((0, 1), (2, 3)), ((0, 2), (1, 3), (4, 5, 6))),
    "proof_fork": (((0, 1, 2), (1, 3)), ((0, 2), (1, 4, 5), (2, 5, 6))),
    "triple_shared_pair": (((0,), (1, 2)), ((3,), (1, 2)), ((4, 5), (1, 2))),
    "triple_handoff": (((0, 1), (2, 3)), ((0, 4), (1, 3)), ((0, 2, 5), (4, 5, 6))),
    "triple_chain": (((0, 1), (2, 3, 4)), ((0, 2), (3, 5)), ((1, 2), (4, 5, 6))),
}
MODES = ("success_hard", "success_ties", "cost_ties")
CONTRASTS = {
    "cost_ties_minus_success_ties": ("success_ties", "cost_ties", True),
    "success_ties_minus_success_hard": ("success_hard", "success_ties", True),
    "cost_ties_minus_success_hard": ("success_hard", "cost_ties", True),
    "exact_cost_minus_exact_success": ("exact_success", "exact_cost", False),
}
PROTOCOL = {
    "schema": "metta-docket-cost-targets-v1", "train_roots_per_family": 32,
    "eval_roots_per_family": 16, "eval_start_ordinal": 1,
    "budgets": BUDGETS, "seeds": (1101, 1102, 1103), "updates": 120,
    "primary": "cost_ties_minus_success_ties", "contrasts": CONTRASTS,
    "minimum_meaningful_completion_effect": 0.02, "evidence_class": "synthetic_development_pilot",
    "max_trajectory_presentations_per_root": 3, "docket_commit": DOCKET_COMMIT,
    "upstream_commit": "207d92ff62f72a818cb3530e064270258e211be4",
    "prior_lock_sha256": "cdcd3a878958e1fdb1676a2bce0ebf55a285163cb69247120503ed35390537a4",
}


def supports(values: dict[int, tuple[int, int]]) -> dict[str, tuple[int, ...]]:
    """Preserve all exact success ties, then take the least-work subset."""
    optimum = max(values.values())
    success = tuple(sorted(slot for slot, score in values.items() if score[0] == optimum[0]))
    cost = tuple(slot for slot in success if values[slot] == optimum)
    return {"success_hard": success[:1], "success_ties": success, "cost_ties": cost}


def preflight(prior: Path) -> tuple[dict, set[str]]:
    archive = (prior / "locked_protocol.json").read_bytes()
    if digest(archive) != PROTOCOL["prior_lock_sha256"]:
        raise ValueError("Prior docket allocation does not match the source pin")
    old = json.loads(archive)
    excluded = {root for field in ("excluded_source_hashes", "train_roots", "eval_roots") for root in old[field]}
    previous = {topology(claims) for claims in (*TRAIN.values(), *PRIOR_EVAL.values())}
    fresh = {topology(claims) for claims in EVAL.values()}
    if len(fresh) != len(EVAL) or fresh & previous:
        raise ValueError("Fresh dockets must differ structurally from both prior cohorts")
    head = committed_checkout(PROTOCOL["upstream_commit"])
    return {"commit": head, "protocol": PROTOCOL, "train_families": TRAIN, "eval_families": EVAL,
            "excluded_source_hashes": sorted(excluded)}, excluded


def pack_train(cases, native: NativeDocket):
    tokens, masks, targets, trajectories, frames = [], [], defaultdict(list), [], []
    for ordinal, (family, docket, order, truth) in enumerate(cases):
        scores = action_values(docket)
        seen = {}
        budget = BUDGETS[ordinal % len(BUDGETS)]

        def offer(public, mode, seen=seen, scores=scores, family=family, order=order):
            if public.state not in seen:
                native_joint = teacher_actions(public, TRAIN[family], order, native)["joint"]
                values = scores(public)
                targets_by_mode = supports(values)
                token, mask = readout(public)
                tokens.append(token)
                masks.append(mask)
                for arm, actions in targets_by_mode.items():
                    distribution = torch.zeros(SLOTS)
                    distribution[list(actions)] = 1.0 / len(actions)
                    targets[arm].append(distribution)
                frames.append({"root": public.docket.root, "family": family, "claims": public.docket.claims,
                               "priors": public.docket.priors, "state": public.state, "remaining": public.remaining,
                               "values": values, "support": targets_by_mode, "native_joint": native_joint})
                seen[public.state] = {**{arm: actions[0] for arm, actions in targets_by_mode.items()},
                                      "native_joint": native_joint}
            return seen[public.state][mode]

        for mode in ("success_hard", "cost_ties", "native_joint"):
            row = rollout(docket, truth, budget, lambda public, mode=mode: offer(public, mode))
            trajectories.append(seal(row, TRAIN[family], order, native, family, budget, mode, None))
    return torch.stack(tokens), torch.stack(masks), targets, trajectories, frames


def fit_models(tokens, masks, targets):
    rows, models = [], {}
    matched = ("initial_sha256", "parameters", "supervised_examples", "optimizer_updates", "supervised_action_tokens")
    for seed in PROTOCOL["seeds"]:
        torch.manual_seed(seed)
        initial = ObligationPointer(SLOTS, FIELDS, BUCKETS)
        for mode in MODES:
            batch = tokens, masks, torch.stack(targets[mode])
            models[mode, seed], metadata = fit(initial, batch, seed, PROTOCOL["updates"])
            rows.append({**metadata, "arm": mode, "visits": initial.core.rounds})
        if any(len({row[field] for row in rows[-len(MODES):]}) != 1 for field in matched):
            raise ValueError("Matched fields differ across cost-target students")
    return rows, models


def evaluate(cases, models, native: NativeDocket):
    receipts = []
    outcomes = defaultdict(lambda: defaultdict(Counter))
    paired = {name: defaultdict(list) for name in CONTRASTS}

    for family, docket, order, truth in cases:
        scores = action_values(docket)

        def reference(public, mode, scores=scores):
            return supports(scores(public))[mode][0]

        def native_joint(public, family=family, order=order):
            return teacher_actions(public, EVAL[family], order, native)["joint"]

        def inference(public, model):
            tokens, legal = readout(public)
            with torch.inference_mode():
                return int(model(tokens[None], legal[None])[0].argmax().item())

        policies = [(arm, seed, lambda public, model=model: inference(public, model), model.core.rounds)
                    for (arm, seed), model in models.items()]
        policies += [("exact_success", None, lambda public: reference(public, "success_hard"), 0),
                     ("exact_cost", None, lambda public: reference(public, "cost_ties"), 0),
                     ("native_joint", None, native_joint, 0)]
        results = {}
        for budget, (arm, seed, choose, visits) in product(BUDGETS, policies):
            row = rollout(docket, truth, budget, choose)
            receipts.append(seal(row, EVAL[family], order, native, family, budget, arm, seed))
            outcomes[family][arm].update(cases=1, completed=int(row["completed"]), checks=row["checks"],
                                         recurrent_visits=row["checks"] * visits)
            results[arm, seed, budget] = row
        for name, (before, after, learned) in CONTRASTS.items():
            seeds = PROTOCOL["seeds"] if learned else (None,)
            pairs = [(results[before, seed, budget], results[after, seed, budget]) for seed, budget in
                     product(seeds, BUDGETS)]
            paired[name][family].append([
                sum(left["checks"] - right["checks"] for left, right in pairs) / len(pairs),
                sum(right["completed"] - left["completed"] for left, right in pairs) / len(pairs),
            ])
    return receipts, outcomes, {name: primary_intervals(rows) for name, rows in paired.items()}


def run(storyworld: Path, prior: Path, out: Path) -> dict:
    limits = install_memory_ceiling(8192)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    lock, old = preflight(prior)
    native = NativeDocket(storyworld)
    train, evaluation = allocate(TRAIN, True, PROTOCOL), allocate(EVAL, False, PROTOCOL)
    train_roots, eval_roots = ({draw[1].root for draw in rows} for rows in (train, evaluation))
    if (len(train_roots) != len(train) or len(eval_roots) != len(evaluation)
            or train_roots & eval_roots or (train_roots | eval_roots) & old):
        raise ValueError("Cost-target allocation overlaps a locked cohort")
    lock.update(train_roots=sorted(train_roots), eval_roots=sorted(eval_roots),
                torch=str(torch.__version__), batch_size=BATCH_SIZE,
                public_rules_sha256=digest(native.public_rules.encode()),
                acceptance_rules_sha256=digest(native.acceptance_rules.encode()))
    out.mkdir(parents=True, exist_ok=False)
    protocol_file = out / "locked_protocol.json"
    protocol_file.write_bytes((canonical(lock) + "\n").encode())

    tokens, masks, targets, trajectories, frames = pack_train(open_cases(train), native)
    trace_sha = write_jsonl(out / "training_trajectories.jsonl", trajectories)
    frame_sha = write_jsonl(out / "training_frames.jsonl", frames)
    fit_rows, models = fit_models(tokens, masks, targets)
    fit_file = out / "locked_fits.json"
    fit_file.write_bytes((canonical(fit_rows) + "\n").encode())
    receipts, outcomes, paired = evaluate(open_cases(evaluation), models, native)

    cost_pruning = sum(row["support"]["cost_ties"] != row["support"]["success_ties"] for row in frames)
    report = {"protocol_sha256": digest(protocol_file.read_bytes()), "fits_sha256": digest(fit_file.read_bytes()),
              "train_trajectory_sha256": trace_sha, "train_frame_sha256": frame_sha,
              "receipts_sha256": write_jsonl(out / "receipts.jsonl", receipts), "outcomes": outcomes, "paired": paired,
              "training_ties": sum(len(row["support"]["success_ties"]) > 1 for row in frames),
              "training_cost_pruning": cost_pruning,
              "native_planning_calls": native.planner.calls, "native_acceptance_calls": native.verifier.calls,
              "wall_seconds": time.perf_counter() - started, "peak_mib": peak_rss_mib(), "memory": limits,
              "training_ready": False}
    (out / "cost_report.json").write_bytes((canonical(report) + "\n").encode())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("storyworld", "prior", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    options = parser.parse_args()
    print(canonical(run(options.storyworld, options.prior, options.out)))


if __name__ == "__main__":
    main()
