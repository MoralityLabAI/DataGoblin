"""Verifier-backed, offline DG-SGRT-01 candidate-search wiring experiment.

All policies here are deterministic, untrained proxies. Synthetic data and
the exact verifier are used to check the harness; no model capability or
causal advantage is claimed. Evaluation targets never reach proposal ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .sgrt_holoarc import (
    BASE_DIM,
    DEFAULT_STEPS,
    HolonomyTrace,
    random_smooth_path,
    spiro_path,
    trace_transport,
)

OPERATORS = (
    "id",
    "xor1",
    "xor2",
    "xor4",
    "add1",
    "add3",
    "rol1",
    "swap01",
    "mask7",
    "or8",
)
FAMILIES: dict[str, tuple[str, ...]] = {
    "arithmetic": ("add1", "add3", "xor1", "xor2"),
    "bitwise": ("xor1", "xor2", "xor4", "swap01"),
    "structural": ("rol1", "swap01", "mask7", "or8"),
    "mixed": ("mask7", "add3", "xor4", "rol1", "or8"),
}
ARMS = (
    "rigid_top1_proxy",
    "semantic_heuristic",
    "extra_refinement_proxy",
    "flat_holoarc",
    "curved_holoarc",
    "same_set_phase_shuffle",
    "smooth_random_arc",
)
Program = tuple[str, str]


def _digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def apply_operator(op: str, x: int) -> int:
    value = x & 15
    if op == "id":
        return value
    if op == "xor1":
        return value ^ 1
    if op == "xor2":
        return value ^ 2
    if op == "xor4":
        return value ^ 4
    if op == "add1":
        return (value + 1) & 15
    if op == "add3":
        return (value + 3) & 15
    if op == "rol1":
        return ((value << 1) | (value >> 3)) & 15
    if op == "swap01":
        return (value & 12) | ((value & 1) << 1) | ((value & 2) >> 1)
    if op == "mask7":
        return value & 7
    if op == "or8":
        return value | 8
    raise ValueError(f"unknown operator: {op}")


def execute(program: Program, x: int) -> int:
    return apply_operator(program[1], apply_operator(program[0], x))


def program_coordinate(program: Program) -> np.ndarray:
    """Fixed Fourier chart for controls, not a learned semantic representation."""
    parts: list[float] = []
    for op in program:
        index = OPERATORS.index(op) + 1
        for frequency in (1, 2, 3):
            theta = 2 * np.pi * index * frequency / len(OPERATORS)
            parts.extend((float(np.cos(theta)), float(np.sin(theta))))
    coordinate = np.asarray(parts, dtype=np.float64)
    assert coordinate.shape == (BASE_DIM,)
    return coordinate


def legal_one_edit_candidates(anchor: Program) -> tuple[Program, ...]:
    alternatives = (
        *((op, anchor[1]) for op in OPERATORS if op != anchor[0]),
        *((anchor[0], op) for op in OPERATORS if op != anchor[1]),
    )
    return tuple(dict.fromkeys(alternatives))


@dataclass(frozen=True)
class SyntheticTask:
    root_id: str
    family: str
    anchor: Program
    demonstrations: tuple[tuple[int, int], ...]
    target: Program

    def public(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "family": self.family,
            "anchor": self.anchor,
            "demonstrations": self.demonstrations,
            "candidate_grammar": "one-operation rewrite",
        }


def make_task(*, seed: int, root: int) -> SyntheticTask:
    rng = np.random.default_rng(seed + 8191 * root)
    family = tuple(FAMILIES)[root % len(FAMILIES)]
    choices = FAMILIES[family]
    target: Program = (
        str(rng.choice(choices)),
        str(rng.choice(choices)),
    )
    changed = int(rng.integers(0, 2))
    wrong_options = tuple(op for op in OPERATORS if op != target[changed])
    replacement = str(rng.choice(wrong_options))
    anchor = (replacement, target[1]) if changed == 0 else (target[0], replacement)
    demos = tuple(
        (int(x), execute(target, int(x)))
        for x in sorted(rng.choice(16, size=4, replace=False).tolist())
    )
    return SyntheticTask(f"public-dev-{seed}-{root:05d}", family, anchor, demos, target)


def policy_seed(seed: int, root: int) -> int:
    # Keep the policy seed independent of the generator's private random draws.
    return int(_digest({"domain": "dg-sgrt-01-policy", "seed": seed, "root": root})[:15], 16)


def semantic_score(program: Program, anchor: Program) -> float:
    return float(np.linalg.norm(program_coordinate(program) - program_coordinate(anchor)))


def _rank_by_path(
    anchor: Program,
    *,
    seed: int,
    budget: int,
    connection: str,
    random_path: bool = False,
) -> tuple[tuple[Program, ...], HolonomyTrace]:
    center = program_coordinate(anchor)
    trajectory = (
        random_smooth_path(center, seed=seed, steps=DEFAULT_STEPS)
        if random_path
        else spiro_path(center, seed=seed, steps=DEFAULT_STEPS)
    )
    trace = trace_transport(trajectory, mode="curved" if connection == "curved" else "flat")
    pool = list(legal_one_edit_candidates(anchor))
    choices: list[Program] = []
    for step in range(1, len(trajectory)):
        if len(choices) >= budget:
            break
        # This frozen, untrained projection is the interface for a future
        # task-conditioned low-rank proposal head. The state difference is
        # path-dependent for noncommuting connection generators.
        tangent = 0.16 * (trace.states[step, :BASE_DIM] - trace.states[0, :BASE_DIM])
        probe = trajectory[step] + tangent
        winner = min(
            (p for p in pool if p not in choices),
            key=lambda p: (
                float(np.linalg.norm(program_coordinate(p) - probe)),
                p,
            ),
        )
        choices.append(winner)
    remainder = sorted(
        (p for p in pool if p not in choices),
        key=lambda p: (semantic_score(p, anchor), p),
    )
    choices.extend(remainder[: budget - len(choices)])
    return tuple(choices), trace


def propose(
    task_public: dict[str, Any],
    *,
    seed: int,
    arm: str,
    budget: int,
) -> tuple[tuple[Program, ...], dict[str, Any] | None]:
    """Target-blind proposals. Only public anchor and family enter this function."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if not 1 <= budget <= 16:
        raise ValueError("budget must be in [1, 16]")
    anchor: Program = tuple(task_public["anchor"])  # type: ignore[assignment]
    pool = legal_one_edit_candidates(anchor)
    if arm == "rigid_top1_proxy":
        return (anchor,), None
    if arm == "semantic_heuristic":
        return tuple(sorted(pool, key=lambda p: (semantic_score(p, anchor), p))[:budget]), None
    if arm == "extra_refinement_proxy":
        # A budget-matched deterministic repeated-edit control, not a TRM.
        start = seed % len(pool)
        order = pool[start:] + pool[:start]
        return order[:budget], None
    connection = "flat" if arm == "flat_holoarc" else "curved"
    order, trace = _rank_by_path(
        anchor,
        seed=seed,
        budget=budget,
        connection=connection,
        random_path=arm == "smooth_random_arc",
    )
    if arm == "same_set_phase_shuffle":
        permutation = np.random.default_rng(seed + 92821).permutation(len(order))
        order = tuple(order[int(i)] for i in permutation)
    return order, trace.receipt()


def _full_semantic_match(candidate: Program, target: Program) -> bool:
    return all(execute(candidate, x) == execute(target, x) for x in range(16))


def verify_and_score(task: SyntheticTask, proposals: tuple[Program, ...]) -> dict[str, Any]:
    """Exact demo checker, followed by evaluator-only full-domain assessment."""
    valid = tuple(
        p for p in proposals
        if all(execute(p, x) == y for x, y in task.demonstrations)
    )
    # Canonical survivor selection makes same-candidate-set shuffle a strict
    # negative control for final solve. First-hit index remains order-sensitive.
    selected = min(valid) if valid else None
    first_hit = next(
        (
            index + 1
            for index, p in enumerate(proposals)
            if all(execute(p, x) == y for x, y in task.demonstrations)
        ),
        None,
    )
    return {
        "candidate_set_sha256": _digest(sorted(proposals)),
        "candidate_order_sha256": _digest(proposals),
        "verifier_calls": len(proposals),
        "demo_survivors": len(valid),
        "first_demo_hit": first_hit,
        "whole_task_solved": selected is not None and _full_semantic_match(selected, task.target),
        "target_reachable_in_set": any(_full_semantic_match(p, task.target) for p in proposals),
        "selected": selected,
    }


def run_suite(*, seed: int, roots: int, budget: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 1 <= roots <= 128:
        raise ValueError("roots must be in [1, 128] to bound CPU and memory")
    if not 1 <= budget <= 16:
        raise ValueError("budget must be in [1, 16]")
    public: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    for root in range(roots):
        task = make_task(seed=seed, root=root)
        visible = task.public()
        public.append(visible)
        pseed = policy_seed(seed, root)
        per_arm: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            candidates, geometry = propose(visible, seed=pseed, arm=arm, budget=budget)
            evaluation = verify_and_score(task, candidates)
            per_arm[arm] = {
                **evaluation,
                "proposals": candidates,
                "geometry": geometry,
                "policy_seed": pseed,
            }
        curved = per_arm["curved_holoarc"]
        shuffle = per_arm["same_set_phase_shuffle"]
        if curved["candidate_set_sha256"] != shuffle["candidate_set_sha256"]:
            raise AssertionError("phase shuffle changed the candidate set")
        if curved["whole_task_solved"] != shuffle["whole_task_solved"]:
            raise AssertionError("same candidate set changed canonical exact solve")
        scored.append({
            "root_id": task.root_id,
            "family": task.family,
            "target": task.target,
            "arms": per_arm,
        })
    return public, scored


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"solved": 0, "reachable": 0, "verifier_calls": 0, "roots": 0}
    )
    for row in scored:
        for arm, outcome in row["arms"].items():
            totals[arm]["solved"] += int(outcome["whole_task_solved"])
            totals[arm]["reachable"] += int(outcome["target_reachable_in_set"])
            totals[arm]["verifier_calls"] += outcome["verifier_calls"]
            totals[arm]["roots"] += 1
    return {
        "schema_version": "dg-sgrt-01-public-development-v1",
        "status": "untrained_synthetic_wiring_only",
        "results": dict(totals),
        "comparability_note": (
            "Rigid top-1 uses one call; all other arms use the same proposal budget. "
            "No model has been trained. Same-set shuffle final solves agree by construction."
        ),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=23, help="public engineering seed")
    parser.add_argument("--roots", type=int, default=8, help="root count, max 128")
    parser.add_argument("--budget", type=int, default=8, help="candidate checks per arm, max 16")
    parser.add_argument("--out", required=True, type=Path, help="ignored local output directory")
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be absent or empty")
    public, scored = run_suite(seed=args.seed, roots=args.roots, budget=args.budget)
    args.out.mkdir(parents=True, exist_ok=True)
    public_hash = _write_jsonl(args.out / "public_tasks.jsonl", public)
    score_hash = _write_jsonl(args.out / "evaluator_scores.jsonl", scored)
    report = summarize(scored)
    report["seed"] = args.seed
    report["roots"] = args.roots
    report["budget"] = args.budget
    report["digests"] = {
        "public_tasks_sha256": public_hash,
        "evaluator_scores_sha256": score_hash,
    }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
