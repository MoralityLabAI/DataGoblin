"""Measure held-out work and verification across fitted and runtime schedules."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from autograd import grad

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import load_structured

from . import rlt_pretrain_cell as cell
from . import rlt_pretrain_data as data
from . import rlt_pretrain_fit as training
from .rlt_lean_labels import atomic_record
from .rlt_lean_ldt import AXIOMS
from .rlt_pilot import target_nll
from .rlt_revision import NODES
from .rlt_trm_ldt import EndpointGate

DEPTHS = (4, 6, 10)
LEAN_BUDGET = 6
ROUTE_SPLITS = ("revision_ood", "length_ood")
ROUTE_POLICIES = ("terminal", "verified_reset", "verified_carry", "verified_reuse")
BLOCK_KEYS = ("self", "message", "partner", "phase", "clock")


class CheckedActionGate:
    """Allow only the exact current-epoch action with an independent proof receipt."""

    def __init__(self, membrane, row, receipt):
        if row["epoch"] != receipt["epoch"] or row["id"] != receipt["id"]:
            raise ValueError("verifier receipt belongs to another goal")
        self.epoch = row["epoch"]
        self.membrane = membrane
        self.receipts = receipt["outcomes"]
        self.state = membrane.CandidateState({"action": frozenset((*self.receipts, "unresolved"))})
        self.attempts = []

    def verify(self, proposal, _context):
        soundness = self.membrane.SoundnessType
        expected, = proposal.proposed_state.domains["action"]
        result = self.receipts.get(expected)
        if proposal.metadata["epoch"] != self.epoch or result is None:
            return soundness.UNKNOWN
        self.attempts.append({
            "id": expected, "label": result["label"], "original_tactic_requests": 1,
            "original_kernel_requests": result["kernel_calls"], "original_tactic_seconds": result["tactic_wall_s"],
            "original_kernel_seconds": result["kernel_wall_s"],
        })
        if (
            result["label"] == 1 and result.get("closed") and result.get("kernel_ok")
            and result["kernel_calls"] == 1 and set(result["axioms"]) <= AXIOMS
        ):
            return soundness.ENV_SOUND_DEAD
        return soundness.UNKNOWN

    def check(self, candidate, *, epoch=None):
        membrane = self.membrane
        proposal = membrane.LatticeProposal(
            membrane.CandidateState({"action": frozenset({candidate})}), membrane.SoundnessType.MODEL_SOUND_DEAD,
            metadata={"epoch": self.epoch if epoch is None else epoch},
        )
        verdict = membrane.certify_and_apply_verified(self.state, proposal, provenance_verifier=self.verify)
        self.state = verdict.after
        return verdict.accepted


def checked_rank(weights, row, receipt, features, runtime, depth, membrane):
    order = row["orders"]["compose"]
    usable = min(LEAN_BUDGET, len(order), depth)
    gate = CheckedActionGate(membrane, row, receipt)
    used = set()
    picked = None
    for logits in cell.lean_visits(weights, features[None], np.ones((1, len(order)), bool), runtime, depth):
        rank = np.argsort(-logits[0], kind="stable")
        candidate = next(order[i] for i in rank if order[i] not in used)
        used.add(candidate)
        if gate.check(candidate):
            picked = candidate
            break
        if len(used) >= usable:
            break
    return {"id": row["id"], "lineage": row["lineage"], "proved": picked,
            "visits": len(used), "attempts": gate.attempts}


def lexical(row, receipt, order, budget, membrane):
    gate = CheckedActionGate(membrane, row, receipt)
    proof = None
    for candidate in order[:budget]:
        if gate.check(candidate):
            proof = candidate
            break
    return {"id": row["id"], "lineage": row["lineage"], "proved": proof, "attempts": gate.attempts}


def run_turn(weights, x, edges, runtime, depth, prior, gate):
    for i, (score, state) in enumerate(cell.route_visits(weights, x[None], edges[None], runtime, depth, prior), 1):
        carry = state
        proposal = NODES[int(score[0].argmax())]
        if gate is not None and gate.check(i, proposal, epoch=gate.epoch):
            break
    if gate is None:
        return {"prediction": proposal, "visits": i, "checks": 0, "checker_hops": 0, "attempts": []}, carry
    return gate.result(), carry


def routes(weights, task, expected, features, runtime, depth, membrane):
    x0, graph_edges, x1, altered, _, _ = features
    graph = dict(task["links"])
    patch = task["patch"]
    change = {patch["source"]: patch["destination"]}
    noop = graph[patch["source"]] == patch["destination"]
    answers = {}
    for policy in ROUTE_POLICIES:
        gate = None if policy == "terminal" else EndpointGate(membrane, graph, task["start"], task["steps"])
        first, memory = run_turn(weights, x0, graph_edges, runtime, depth, None, gate)
        after_gate = gate.advance(change) if gate is not None else None
        if after_gate is not None and policy == "verified_reuse" and noop and first["prediction"] is not None:
            if not after_gate.check(0, first["prediction"], epoch=1):
                raise ValueError("a no-op certificate failed current-epoch verification")
            second = after_gate.result()
        else:
            prior = None if policy == "verified_reset" else memory
            second, _ = run_turn(weights, x1, altered, runtime, depth, prior, after_gate)
        acc = {key: result["prediction"] == expected[key] for key, result in (("before", first), ("after", second))}
        if gate is not None and any(
            result["prediction"] is not None and not acc[key]
            for key, result in (("before", first), ("after", second))
        ):
            raise ValueError("live route verifier accepted an incorrect endpoint")
        answers[policy] = {"before": first, "after": second, "accuracy": acc}
    return {"id": task["id"], "lineage": task["lineage"], "no_op": noop, "policies": answers}


def text_scores(weights, text, runtime, depth, guard):
    counts = defaultdict(Counter)
    windows = []
    for start in range(0, len(text), 16):
        guard()
        batch = text[start : start + 16]
        x = np.array([row["tokens"][:-1] for row in batch])
        y = np.array([row["tokens"][1:] for row in batch])
        visits = cell.text_visits(weights, x, np.ones(x.shape, bool), runtime, depth)
        logits = deque(visits, maxlen=1).pop()
        nll = target_nll(logits.reshape(-1, cell.VOCAB), y.ravel()).reshape(y.shape)
        prediction = logits.argmax(axis=-1)
        for i, row in enumerate(batch):
            metric = counts[row["corpus"]]
            metric["nll"] += float(np.sum(nll[i]))
            metric["tokens"] += len(y[i])
            metric["correct"] += int(np.sum(prediction[i] == y[i]))
            metric["windows"] += 1
            metric["cell_calls"] += 2 * depth
            windows.append({"id": row["id"], "corpus": row["corpus"], "nll": float(np.sum(nll[i])),
                            "tokens": len(y[i]), "correct": int(np.sum(prediction[i] == y[i]))})
    aggregate = {name: {**stats, "bits_per_byte": stats["nll"] / (stats["tokens"] * math.log(2))}
                 for name, stats in counts.items()}
    return aggregate, windows


def gradient_audit(root, guard):
    data_arrays = training.train_data(root)
    indices = [np.arange(4), np.arange(len(data_arrays[1][0])), np.arange(4)]
    batches = [tuple(a[ix] for a in lane) for ix, lane in zip(indices, data_arrays, strict=True)]
    weights = cell.initialize(training.SPEC["seeds"][0], training.SPEC["width"], batches[1][0].shape[-1])
    records = {}
    for i, name in enumerate(training.SPEC["lanes"]):
        guard()
        directional = grad(lambda p, component=i: training.losses(p, batches, "direct", 6)[component])(weights)
        values = {key: float(np.linalg.norm(directional[key])) for key in BLOCK_KEYS}
        if any(not math.isfinite(x) for x in values.values()) or not any(x > 0 for x in values.values()):
            raise ValueError(f"{name} does not train shared cell weights")
        records[name] = values
    return records


def summarize_runs(text, lean, graph):
    proof = {
        "goals": len(lean), "proved": sum(x["proved"] is not None for x in lean),
        "visits": sum(x["visits"] for x in lean), "verifier_replay_checks": sum(len(x["attempts"]) for x in lean),
        "projected_tactic_requests": sum(a["original_tactic_requests"] for x in lean for a in x["attempts"]),
        "projected_kernel_requests": sum(a["original_kernel_requests"] for x in lean for a in x["attempts"]),
        "unknown_attempts": sum(a["label"] is None for x in lean for a in x["attempts"]),
    }
    rules = {}
    for split, rows in graph.items():
        counts = {}
        for policy in ROUTE_POLICIES:
            by_edit = defaultdict(Counter)
            for r in rows:
                c = by_edit["no_op" if r["no_op"] else "edited"]
                p = r["policies"][policy]
                c["goals"] += 1
                c["joint_correct"] += p["accuracy"]["before"] and p["accuracy"]["after"]
                for turn in ("before", "after"):
                    c[turn + "_correct"] += p["accuracy"][turn]
                    for field in ("visits", "checks", "checker_hops"):
                        c[field] += p[turn][field]
            counts[policy] = dict(by_edit)
        rules[split] = counts
    return {"text": text, "lean": proof, "revision": rules}


def evaluate(root, guard):
    from research_gym.core import hybrid

    training.frozen_inputs(root)
    protocol_file = root / "fit-protocol.json"
    protocol = load_structured(protocol_file)
    fits = load_structured(root / "fit-receipts.json")
    if (
        protocol["dataset_sha256"] != sha256_file(root / "dataset.json")
        or protocol["fit_source_sha256"] != sha256_file(Path(training.__file__))
        or protocol["cell_source_sha256"] != sha256_file(Path(cell.__file__))
        or fits["protocol_sha256"] != sha256_file(protocol_file)
    ):
        raise ValueError("fitted protocol changed after training")
    register = {
        "fit_receipts_sha256": sha256_file(root / "fit-receipts.json"),
        "source_sha256": sha256_file(Path(__file__)), "membrane_sha256": sha256_file(Path(hybrid.__file__)),
        "depths": DEPTHS, "crossover_depth": training.SPEC["depth"], "lean_budget": LEAN_BUDGET,
        "recipe_comparison_depth": training.SPEC["depth"],
        "rule_splits": ROUTE_SPLITS, "rule_policies": ROUTE_POLICIES,
    }
    path = root / "eval-protocol.json"
    if path.exists() and sha256_json(load_structured(path)) != sha256_json(register):
        raise ValueError("mixed pilot evaluation protocol changed")
    atomic_record(path, register)
    text = list(data.json_lines(root / "text-held.jsonl"))
    proofs = list(training.proof_examples(root, "held_structures"))
    proof_stalls = [(row, checked, features) for row, checked, features, _ in proofs
                    if checked["native"] is not None and checked["native"]["accepted"] is None]
    graph = {name: list(training.revision_examples(root, name)) for name in ROUTE_SPLITS}
    baseline = root / "baselines.json"
    if not baseline.exists():
        if not proofs:
            raise ValueError("no held-out Lean goals")
        row, checked, _, _ = proofs[0]
        gate = CheckedActionGate(hybrid, row, checked)
        candidate = next((k for k in row["orders"]["compose"] if checked["outcomes"][k]["label"] == 1),
                         row["orders"]["compose"][0])
        if gate.check(candidate, epoch="stale") or gate.attempts:
            raise ValueError("stale proof replay reached the verifier")
        atomic_record(baseline, {
            "lean_held_goals": len(proofs), "lean_native_stalls": len(proof_stalls), "stale_epoch_rejected": True,
            "native_tactic_requests": sum(c["native"]["tactic_checks"] for _, c, _, _ in proofs),
            "native_kernel_requests": sum(c["native"]["kernel_calls"] for _, c, _, _ in proofs),
            "lean_lexical": {
                str(budget): {
                    method: [lexical(row, c, row["orders"][method], budget, hybrid) for row, c, _ in proof_stalls]
                    for method in ("legacy", "wider", "compose")
                }
                for budget in sorted({min(depth, LEAN_BUDGET) for depth in DEPTHS})
            },
            "lean_pool_verified": sum(any(c["outcomes"][key]["label"] == 1 for key in row["orders"]["compose"])
                                      for row, c, _ in proof_stalls),
            "route_native": {
                name: {"goals": len(rows), "route_hops_full_two_turns": 2 * sum(t["steps"] for t, _, _ in rows)}
                for name, rows in graph.items()
            },
            "shared_gradient_norms": gradient_audit(root, guard),
        })
    directory = root / "eval"
    directory.mkdir(exist_ok=True)
    summaries = []
    for fit in fits["arms"]:
        arm, seed, recipe = fit["arm"], fit["seed"], fit["recipe"]
        weights_path = training.fit_directory(root, seed, arm, recipe) / "weights.npz"
        if sha256_file(weights_path) != fit["weights_sha256"]:
            raise ValueError("checkpoint weights changed after fitting")
        with np.load(weights_path, allow_pickle=False) as arrays:
            weights = dict(arrays)
        if recipe == "mixed":
            matrix = [(arm, depth) for depth in DEPTHS]
            matrix.extend((mode, training.SPEC["depth"]) for mode in cell.ARMS if mode != arm)
        else:
            matrix = [(arm, training.SPEC["depth"])]
        for runtime, depth in matrix:
            guard()
            file = directory / f"{seed}-{recipe}-{arm}-{runtime}-k{depth}.json"
            if not file.exists():
                start = time.monotonic()
                lm, windows = text_scores(weights, text, runtime, depth, guard)
                lean = []
                for row, receipt, features in proof_stalls:
                    guard()
                    lean.append(checked_rank(weights, row, receipt, features, runtime, depth, hybrid))
                route = {}
                for split, examples in graph.items():
                    route[split] = []
                    for task, target, features in examples:
                        guard()
                        route[split].append(routes(weights, task, target, features, runtime, depth, hybrid))
                atomic_record(file, {
                    "summary": summarize_runs(lm, lean, route), "lean": lean, "revision": route,
                    "text_windows": windows,
                    "runtime_arm": runtime, "fit_arm": arm, "depth": depth, "seed": seed, "recipe": recipe,
                    "wall_s": time.monotonic() - start,
                })
            report = load_structured(file)
            summaries.append({
                key: report[key] for key in ("runtime_arm", "fit_arm", "depth", "seed", "summary", "recipe")
            } | {"trace_file": file.name, "trace_sha256": sha256_file(file), "wall_s": report["wall_s"]})
        print(f"Scored held-out tasks for seed {seed} {arm} on {recipe}", flush=True)
    atomic_record(root / "evidence.json", {
        "schema": "datagoblin.1b-mixed-pretrain-pilot.v1", "baselines": load_structured(baseline),
        "fit_protocol_sha256": sha256_file(protocol_file), "eval_protocol_sha256": sha256_file(path),
        "dataset_sha256": sha256_file(root / "dataset.json"), "matrix": summaries,
    })
