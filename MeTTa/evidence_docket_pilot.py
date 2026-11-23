"""Train matched Looped Transformers on individual and shared MeTTa docket teachers."""
from __future__ import annotations

import argparse
import inspect
import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import torch
from research_gym.neural.obligation_pointer import ObligationPointer, pack_graph

from metta_research_bench.looped.model import BATCH_SIZE, BUCKETS, FIELDS, fit

from .evidence_budget import WIDTH
from .evidence_budget_confirmation import primary_intervals, relabel
from .evidence_docket import (
    BUDGETS,
    DOCKET_COMMIT,
    Claims,
    Docket,
    NativeDocket,
    Observation,
    exact_policy,
    frontier_choice,
    map_frontier,
    needs,
    rollout,
    topology,
)
from .evidence_incidence_pilot import ROUTE_SLOTS, SLOTS
from .ram_hillclimb import canonical, digest, install_memory_ceiling, peak_rss_mib

TRAIN = {
    "shared_pair": (((0,), (1, 2)), ((3,), (1, 2))),
    "bridge": (((0, 1), (2, 3)), ((0, 2), (3, 4))),
    "handoff": (((0,), (1, 2), (3, 4)), ((1, 3, 4), (2, 5))),
    "scope": (((0, 1), (1, 2, 3)), ((1, 3), (4, 5, 6))),
    "revisions": (((0, 1), (0, 2, 3)), ((1, 2), (0, 4, 5))),
    "coverage": (((0, 1, 2), (3, 4)), ((1, 2, 3), (4, 5, 6))),
}
EVAL = {
    "shared_revisions": (((0,), (1, 2, 3)), ((4, 5), (1, 2, 3))),
    "lineage_bridge": (((0, 1), (2, 3), (4, 5)), ((0, 2), (1, 4), (3, 6))),
    "cross_index": (((0, 1, 2), (0, 3, 4)), ((1, 3, 5), (2, 4, 6))),
    "staged_release": (((0, 1), (2, 3, 4)), ((0, 2), (1, 4, 5), (3, 6))),
    "three_claim_chain": (((0,), (1, 2)), ((1, 3), (2, 4)), ((2, 4), (3, 5, 6))),
    "three_claim_overlap": (((0, 1), (2, 3)), ((0, 4), (1, 2, 5)), ((0, 1, 4), (3, 5, 6))),
}
TEACHERS = ("individual", "joint")
SYMBOLIC = tuple(f"native_{name}" for name in TEACHERS)
LEARNED = tuple(f"learned_{name}" for name in TEACHERS)
PROTOCOL = {
    "schema": "metta-evidence-docket-pilot-v1", "train_roots_per_family": 24,
    "eval_roots_per_family": 16, "eval_start_ordinal": 1,
    "budgets": BUDGETS, "seeds": (1101, 1102, 1103), "updates": 120,
    "evidence_class": "synthetic_development_pilot", "primary": "learned_joint_minus_individual",
    "minimum_meaningful_completion_effect": 0.02, "max_trajectory_presentations_per_root": 2,
    "upstream_commit": "207d92ff62f72a818cb3530e064270258e211be4", "docket_commit": DOCKET_COMMIT,
    "prior_lock_sha256": "9c0661691c772567b5f2c380b4c0b88a295d03bf2fecc2d858a78f33f8d1e768",
}


def sample(family: str, claims: Claims, ordinal: int, *, train: bool,
           schema: str = PROTOCOL["schema"]) -> tuple[Docket, tuple[int, ...], random.Random]:
    rng = random.Random(f"{schema}/{train}/{family}/{ordinal}")
    order = tuple(rng.sample(range(WIDTH), WIDTH))
    priors = tuple(rng.randrange(22, 94) for _ in order)
    return Docket(tuple(relabel(group, order) for group in claims), priors), order, rng


def readout(public: Observation):
    """Both models get the same record and route tokens and the same action mask."""
    routes = [(claim, path) for claim, group in enumerate(public.docket.claims) for path in group]
    if len(routes) > ROUTE_SLOTS:
        raise ValueError("Docket exceeds the registered route slots")
    rows = torch.zeros((SLOTS, FIELDS))
    incidence = torch.zeros((ROUTE_SLOTS, WIDTH), dtype=torch.bool)
    for node, (claim, path) in enumerate(routes):
        incidence[node, list(path)] = True
        rows[WIDTH + node, :5] = torch.tensor((2, (claim + 1) / 3, len(path) / WIDTH,
                                               public.remaining / max(BUDGETS), len(public.docket.claims) / 3))
        rows[WIDTH + node, 5:5 + WIDTH] = incidence[node]
    for slot, (prior, receipt) in enumerate(zip(public.docket.priors, public.state, strict=True)):
        rows[slot, :5] = torch.tensor((1, prior / 100, receipt, public.remaining / max(BUDGETS),
                                       len(public.docket.claims) / 3))
        rows[slot, 5:5 + ROUTE_SLOTS] = incidence[:, slot]
    codes = ((rows + 2) * 8).round().long() + torch.arange(FIELDS) * BUCKETS
    present = torch.arange(SLOTS) < WIDTH + len(routes)
    allowed = torch.zeros(SLOTS, dtype=torch.bool)
    allowed[list(public.legal)] = True
    return pack_graph(codes, present, incidence, "dense"), allowed


def teacher_actions(public: Observation, claims: Claims, order: tuple[int, ...], native: NativeDocket) -> dict:
    individual, joint = native.read(claims, tuple(public.state[i] for i in order))
    per_claim = tuple(map_frontier(front, order) for front in individual)
    shared = map_frontier(joint, order)
    if (per_claim, shared) != needs(public.docket.claims, public.state):
        raise ValueError("Relabeled public docket differs from its canonical native frontier")
    return {mode: frontier_choice(public, per_claim, shared, shared=(mode == "joint")) for mode in TEACHERS}


def pack_train(cases, native: NativeDocket):
    tokens, masks, labels, trajectories, frames = [], [], defaultdict(list), [], []
    for ordinal, (family, docket, order, truth) in enumerate(cases):
        budget = BUDGETS[ordinal % len(BUDGETS)]
        seen = set()

        def teacher(public, mode, claims=TRAIN[family], order=order, seen=seen, family=family):
            targets = teacher_actions(public, claims, order, native)
            if public.state not in seen:
                token, mask = readout(public)
                tokens.append(token)
                masks.append(mask)
                for teacher_mode, slot in targets.items():
                    labels[teacher_mode].append(slot)
                frames.append({"root": public.docket.root, "family": family, "claims": public.docket.claims,
                               "priors": public.docket.priors, "receipted_state": public.state,
                               "remaining": public.remaining, "legal_checks": public.legal, "teacher_checks": targets})
                seen.add(public.state)
            return targets[mode]

        for mode in TEACHERS:
            row = rollout(docket, truth, budget, lambda public, mode=mode: teacher(public, mode))
            trajectories.append(seal(row, TRAIN[family], order, native, family, budget, mode, None))
    return torch.stack(tokens), torch.stack(masks), labels, trajectories, frames


def seal(row, claims, order, native, family, budget, arm, seed):
    state = tuple(row["final_state"][i] for i in order)
    proof, sha = native.accept(claims, state)
    if bool(proof) != row["completed"]:
        raise ValueError("Relabeled final state disagrees with native completion")
    return {**row, "family": family, "budget": budget, "arm": arm, "seed": seed,
            "native_proofs": proof, "native_input_sha256": sha, "proof_slot_order": order}


def preflight(prior: Path) -> tuple[dict, set[str]]:
    raw = (prior / "locked_protocol.json").read_bytes()
    if digest(raw) != PROTOCOL["prior_lock_sha256"]:
        raise ValueError("Use the registered previous allocation")
    old = json.loads(raw)
    excluded = {root for field in ("excluded_source_hashes", "train_source_hashes", "eval_source_hashes")
                for root in old[field]}
    fingerprints = {topology(claims) for claims in (*TRAIN.values(), *EVAL.values())}
    if len(fingerprints) != len(TRAIN) + len(EVAL):
        raise ValueError("The registered docket families must be non-isomorphic")
    return {"commit": committed_checkout(PROTOCOL["upstream_commit"]), "protocol": PROTOCOL,
            "train_families": TRAIN, "eval_families": EVAL, "excluded_source_hashes": sorted(excluded)}, excluded


def committed_checkout(upstream_commit: str) -> str:
    upstream = Path(inspect.getfile(ObligationPointer)).resolve().parents[2]
    if (subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
            != upstream_commit or subprocess.check_output(
                ["git", "-C", str(upstream), "diff", "HEAD", "--", "research_gym", "lsa"]
            )):
        raise ValueError("Use the pinned LoopedTransformers checkout")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("Commit the protocol and experiment before opening the cohort")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def allocate(families: dict, train: bool, protocol=PROTOCOL) -> list:
    count = protocol["train_roots_per_family" if train else "eval_roots_per_family"]
    start = 0 if train else protocol["eval_start_ordinal"]
    return [(family, *sample(family, routes, index, train=train, schema=protocol["schema"]))
            for family, routes in families.items() for index in range(start, start + count)]


def open_cases(draws):
    for family, docket, order, rng in draws:
        yield family, docket, order, tuple(rng.randrange(100) < p for p in docket.priors)


def write_jsonl(path: Path, rows: list) -> str:
    data = "".join(canonical(row) + "\n" for row in rows).encode()
    path.write_bytes(data)
    return digest(data)


def run(checkout: Path, prior: Path, out: Path) -> dict:
    limits = install_memory_ceiling(8192)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    lock, old = preflight(prior)
    native = NativeDocket(checkout)
    train, evaluation = allocate(TRAIN, True), allocate(EVAL, False)
    train_roots, eval_roots = ({draw[1].root for draw in rows} for rows in (train, evaluation))
    if (len(train_roots) != len(train) or len(eval_roots) != len(evaluation)
            or train_roots & eval_roots or (train_roots | eval_roots) & old):
        raise ValueError("Docket allocation overlaps a locked cohort")
    lock.update(train_roots=sorted(train_roots), eval_roots=sorted(eval_roots),
                torch=str(torch.__version__), batch_size=BATCH_SIZE,
                public_rules_sha256=digest(native.public_rules.encode()),
                acceptance_rules_sha256=digest(native.acceptance_rules.encode()))
    out.mkdir(parents=True, exist_ok=False)
    protocol_file = out / "locked_protocol.json"
    protocol_file.write_bytes((canonical(lock) + "\n").encode())

    tokens, masks, labels, trajectories, frames = pack_train(open_cases(train), native)
    trace_sha = write_jsonl(out / "training_trajectories.jsonl", trajectories)
    frame_sha = write_jsonl(out / "training_frames.jsonl", frames)
    fit_rows, models = [], {}
    matched = ("initial_sha256", "parameters", "supervised_examples", "optimizer_updates", "supervised_action_tokens")
    for seed in PROTOCOL["seeds"]:
        torch.manual_seed(seed)
        initial = ObligationPointer(SLOTS, FIELDS, BUCKETS)
        for mode in TEACHERS:
            models[mode, seed], metadata = fit(initial, (tokens, masks, torch.tensor(labels[mode])),
                                                seed, PROTOCOL["updates"])
            fit_rows.append({**metadata, "arm": f"learned_{mode}", "visits": initial.core.rounds})
        if any(fit_rows[-2][field] != fit_rows[-1][field] for field in matched):
            raise ValueError("Matched training fields differ across docket students")
    fit_file = out / "locked_fits.json"
    fit_file.write_bytes((canonical(fit_rows) + "\n").encode())

    def learned(mode, seed):
        model = models[mode, seed]

        @torch.inference_mode()
        def choose(public):
            tokens, legal = readout(public)
            return int(model(tokens[None], legal[None])[0].argmax().item())

        return choose

    students = [(f"learned_{mode}", seed, learned(mode, seed))
                for mode, seed in product(TEACHERS, PROTOCOL["seeds"])]
    outcomes, receipts = defaultdict(lambda: defaultdict(Counter)), []
    comparisons = {"symbolic_joint_minus_individual": defaultdict(list), PROTOCOL["primary"]: defaultdict(list)}

    for family, docket, order, truth in open_cases(evaluation):
        claims = EVAL[family]

        def local(public, claims=claims, order=order):
            return teacher_actions(public, claims, order, native)["individual"]

        def joint(public, claims=claims, order=order):
            return teacher_actions(public, claims, order, native)["joint"]

        policies = [*students, (SYMBOLIC[0], None, local), (SYMBOLIC[1], None, joint),
                    ("exact_joint", None, exact_policy(docket))]
        results = {}
        for budget, (arm, seed, choose) in product(BUDGETS, policies):
            row = rollout(docket, truth, budget, choose)
            receipts.append(seal(row, claims, order, native, family, budget, arm, seed))
            outcomes[family][arm].update(cases=1, completed=int(row["completed"]), checks=row["checks"],
                                         recurrent_visits=row["checks"] * (models[TEACHERS[0], seed].core.rounds
                                                                          if seed is not None else 0))
            results[arm, seed, budget] = row
        for name, modes, seeds in (("symbolic_joint_minus_individual", SYMBOLIC, (None,)),
                                   (PROTOCOL["primary"], LEARNED, PROTOCOL["seeds"])):
            differences = []
            for seed, budget in product(seeds, BUDGETS):
                left, right = (results[arm, seed, budget] for arm in modes)
                differences.append((left["checks"] - right["checks"], right["completed"] - left["completed"]))
            comparisons[name][family].append([sum(values) / len(values) for values in zip(*differences, strict=True)])

    report = {"protocol_sha256": digest(protocol_file.read_bytes()), "fits_sha256": digest(fit_file.read_bytes()),
              "train_frame_sha256": frame_sha,
              "train_trajectory_sha256": trace_sha, "receipts_sha256": write_jsonl(out / "receipts.jsonl", receipts),
              "label_disagreements": sum(a != b for a, b in zip(*(labels[mode] for mode in TEACHERS), strict=True)),
              "outcomes": outcomes, "paired": {name: primary_intervals(rows) for name, rows in comparisons.items()},
              "native_planning_calls": native.planner.calls, "native_acceptance_calls": native.verifier.calls,
              "wall_seconds": time.perf_counter() - started, "peak_mib": peak_rss_mib(), "memory": limits,
              "training_ready": False}
    (out / "docket_report.json").write_bytes((canonical(report) + "\n").encode())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("storyworld", "prior", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    options = parser.parse_args()
    print(canonical(run(options.storyworld, options.prior, options.out)))


if __name__ == "__main__":
    main()
