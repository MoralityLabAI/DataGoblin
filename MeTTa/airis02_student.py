"""Small CPU student comparison for AIRIS-02 fresh synthetic curation worlds.

The direct linear head and weight-tied recurrent head both learn an exact
synthetic first-edit target. All four intervention policies receive equal root
counts and optimizer updates. Recurrent weights are trained with manual BPTT
at four iterations; 2/4/8 passes are inference-depth ablations on those SAME
weights, not independently optimized models. STOP is an output action, not a
trained ACT or learned per-inference exit budget.

No external weights, GPU, hidden evaluation data, or native Hyperon is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .airis02_fresh import ARMS, DEFAULT_OUT
from .collect import sha256_bytes
from .sft_oracle import edit, evaluate_subset, exact_optimum

ACTION_CLASSES = 21
MAX_ITEMS = 10
GOAL_COUNT = 5
FIELD_DIM = 15
GLOBAL_DIM = 13
INPUT_DIM = MAX_ITEMS * FIELD_DIM + GLOBAL_DIM
FAMILIES = ("lineage", "coverage", "dependency", "revision", "budget")
SCHEMA = "metta-airis02-cpu-student-v4"


def encode_public(task: dict[str, Any], selected: list[str]) -> np.ndarray:
    """Read structured PUBLIC facts and selection, no oracle labels or receipts."""
    items = task["items"]
    chosen = set(selected)
    if len(chosen) != len(selected):
        raise ValueError("duplicate selected item")
    lookup = {x["id"]: x for x in items}
    if not chosen <= lookup.keys():
        raise ValueError("selected item absent from public source")
    lineages = [lookup[x]["lineage"] for x in selected]
    keys = [lookup[x]["key"] for x in selected]
    cost = sum(lookup[x]["cost"] for x in selected)
    covered = {topic for x in selected for topic in lookup[x]["topics"]}
    vec: list[float] = []
    for i in range(MAX_ITEMS):
        if i >= len(items):
            vec.extend([0.0] * FIELD_DIM)
            continue
        x = items[i]
        selected_here = x["id"] in chosen
        requirements = set(x["requires"])
        available_eligible_same_key = [j for j in items
                                       if j["key"] == x["key"] and j["eligible"]]
        has_newer = any(j["epoch"] > x["epoch"] for j in available_eligible_same_key)
        selected_lineage_duplicates = lineages.count(x["lineage"]) if selected_here else 0
        fields = [
            1.0,
            float(selected_here),
            float(x["eligible"]),
            x["cost"] / 20.0,
            x["quality"] / 16.0,
            x["epoch"] / 10.0,
            float(selected_lineage_duplicates > 1),
            float(x["key"] in keys and keys.count(x["key"]) > 1),
            len(requirements) / 5.0,
            len(requirements - chosen) / 5.0,
        ]
        fields.extend(float(goal in x["topics"]) for goal in [f"skill{k}" for k in range(GOAL_COUNT)])
        if len(fields) != FIELD_DIM:
            raise ValueError("invalid student feature width")
        # The selected-record revision status is represented by an existing
        # key collision field and epoch; no independent oracle fields appear.
        if has_newer and task["family"] == "revision":
            fields[7] = 1.0
        vec.extend(fields)
    goals = len(task["goals"])
    global_fields = [
        task["token_budget"] / 50.0,
        task["max_items"] / 10.0,
        len(selected) / 10.0,
        cost / 60.0,
        len([x for x in selected if not lookup[x]["eligible"]]) / 10.0,
        max(0, len(lineages) - len(set(lineages))) / 10.0,
        len([x for x in selected if not set(lookup[x]["requires"]) <= chosen]) / 10.0,
        len(covered & set(task["goals"])) / max(1, goals),
    ]
    global_fields.extend(float(task["family"] == f) for f in FAMILIES)
    if len(global_fields) != GLOBAL_DIM:
        raise ValueError("invalid public global vector length")
    vec.extend(global_fields)
    if len(vec) != INPUT_DIM:
        raise ValueError("invalid student input width")
    return np.asarray(vec, dtype=np.float64)


def action_index(action: dict[str, str]) -> int:
    op = action.get("op")
    target = action.get("id")
    if op == "stop" and target == "none":
        return 20
    if op not in ("add", "remove") or not isinstance(target, str) or not target.startswith("r"):
        raise ValueError("invalid curation repair action")
    try:
        index = int(target[1:])
    except ValueError as exc:
        raise ValueError("invalid candidate index") from exc
    if not 0 <= index < MAX_ITEMS or target != f"r{index}":
        raise ValueError("invalid canonical candidate action")
    return index if op == "add" else MAX_ITEMS + index


def decode_action(index: int) -> dict[str, str]:
    if type(index) is not int or index < 0 or index >= ACTION_CLASSES:
        raise ValueError("invalid action logit index")
    if index == 20:
        return {"op": "stop", "id": "none"}
    if index < MAX_ITEMS:
        return {"op": "add", "id": f"r{index}"}
    return {"op": "remove", "id": f"r{index - MAX_ITEMS}"}


def legal_action_mask(task: dict[str, Any], selected: list[str]) -> np.ndarray:
    """Suppress structurally inapplicable ADD/REMOVE moves using public IDs."""
    ids = {item["id"] for item in task["items"]}
    chosen = set(selected)
    if len(chosen) != len(selected) or not chosen <= ids:
        raise ValueError("invalid public state for action membership mask")
    mask = np.zeros(ACTION_CLASSES, dtype=np.bool_)
    for identifier in ids:
        index = int(identifier[1:])
        if identifier in chosen:
            mask[MAX_ITEMS + index] = True
        else:
            mask[index] = True
    mask[20] = True
    return mask


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(np.clip(shifted, -700, 0))
    return weights / weights.sum(axis=1, keepdims=True)


def _cross_entropy(
    logits: np.ndarray, target: np.ndarray,
    legal_masks: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    if legal_masks is not None:
        if legal_masks.shape != logits.shape or not legal_masks[np.arange(len(target)), target].all():
            raise ValueError("teacher chose structurally inapplicable edit")
        logits = np.where(legal_masks, logits, -1e6)
    probs = _softmax(logits)
    n = len(target)
    loss = -np.log(np.maximum(probs[np.arange(n), target], 1e-16)).mean()
    grad = probs.copy()
    grad[np.arange(n), target] -= 1.0
    return float(loss), grad / n


def _clip(grads: list[np.ndarray], cap: float = 2.0) -> list[np.ndarray]:
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads))
    factor = min(1.0, cap / max(norm, 1e-12))
    return [g * factor for g in grads]


class LinearStudent:
    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.045, size=(INPUT_DIM, ACTION_CLASSES))
        self.b = np.zeros(ACTION_CLASSES, dtype=np.float64)

    @property
    def params(self) -> int:
        return self.w.size + self.b.size

    def logits(self, x: np.ndarray, *, loops: int = 1) -> np.ndarray:
        if loops != 1:
            raise ValueError("linear student has one inference pass")
        return x @ self.w + self.b

    def fit(
        self, x: np.ndarray, y: np.ndarray, *, steps: int, lr: float,
        legal_masks: np.ndarray | None = None,
    ) -> list[float]:
        losses = []
        for step in range(steps):
            logits = self.logits(x)
            loss, d = _cross_entropy(logits, y, legal_masks)
            dw = x.T @ d + 0.0001 * self.w
            db = d.sum(axis=0)
            dw, db = _clip([dw, db])
            eta = lr / (1.0 + step / max(1.0, steps))
            self.w -= eta * dw
            self.b -= eta * db
            losses.append(round(loss, 6))
        return losses

    def digest(self) -> str:
        return hashlib.sha256(self.w.tobytes() + self.b.tobytes()).hexdigest()


class RecurrentStudent:
    """A tiny weight-tied tanh loop trained by exact manual BPTT."""
    def __init__(self, seed: int, hidden: int = 18):
        if not 4 <= hidden <= 64:
            raise ValueError("recurrent hidden width outside CPU bound")
        rng = np.random.default_rng(seed)
        self.wx = rng.normal(0, 0.085 / math.sqrt(INPUT_DIM), size=(INPUT_DIM, hidden))
        self.wh = rng.normal(0, 0.15 / math.sqrt(hidden), size=(hidden, hidden))
        self.wo = rng.normal(0, 0.12 / math.sqrt(hidden), size=(hidden, ACTION_CLASSES))
        self.bx = np.zeros(hidden, dtype=np.float64)
        self.bo = np.zeros(ACTION_CLASSES, dtype=np.float64)

    @property
    def params(self) -> int:
        return sum(p.size for p in (self.wx, self.wh, self.wo, self.bx, self.bo))

    def _forward(self, x: np.ndarray, *, loops: int) -> tuple[np.ndarray, list[np.ndarray]]:
        if not 1 <= loops <= 8:
            raise ValueError("recurrent unroll count outside bound")
        drive = x @ self.wx + self.bx
        history = [np.zeros((len(x), self.wh.shape[0]), dtype=np.float64)]
        for _ in range(loops):
            history.append(np.tanh(drive + history[-1] @ self.wh))
        return history[-1] @ self.wo + self.bo, history

    def logits(self, x: np.ndarray, *, loops: int = 4) -> np.ndarray:
        return self._forward(x, loops=loops)[0]

    def fit(
        self, x: np.ndarray, y: np.ndarray, *, steps: int, lr: float,
        train_loops: int = 4, legal_masks: np.ndarray | None = None,
    ) -> list[float]:
        losses = []
        for step in range(steps):
            logits, states = self._forward(x, loops=train_loops)
            loss, grad = _cross_entropy(logits, y, legal_masks)
            dwo = states[-1].T @ grad + 0.0001 * self.wo
            dbo = grad.sum(axis=0)
            dh = grad @ self.wo.T
            dwx = np.zeros_like(self.wx)
            dwh = np.zeros_like(self.wh)
            dbx = np.zeros_like(self.bx)
            for j in range(train_loops, 0, -1):
                dz = dh * (1.0 - states[j] * states[j])
                dwx += x.T @ dz
                dwh += states[j - 1].T @ dz
                dbx += dz.sum(axis=0)
                dh = dz @ self.wh.T
            dwx += 0.0001 * self.wx
            dwh += 0.0001 * self.wh
            grads = _clip([dwx, dwh, dwo, dbx, dbo])
            eta = lr / (1.0 + step / max(1.0, steps))
            for param, dparam in zip((self.wx, self.wh, self.wo, self.bx, self.bo),
                                     grads, strict=True):
                param -= eta * dparam
            losses.append(round(loss, 6))
        return losses

    def digest(self) -> str:
        return hashlib.sha256(b"".join(x.tobytes() for x in (
            self.wx, self.wh, self.wo, self.bx, self.bo
        ))).hexdigest()


def canonical_eval_key(row: dict) -> str:
    # Root, selected IDs and public world are all that define an eval input.
    # Oracle labels are deliberately excluded from the equality check.
    return json.dumps(
        {"root": row["root"], "world": row["public_world"], "selected": row["selected"]},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def _matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("empty student training rows")
    if any(row["training_ready"] is not False for row in rows):
        raise ValueError("student dataset exceeds synthetic research custody")
    features = np.stack([encode_public(row["public_world"], row["selected"]) for row in rows])
    labels = np.array([action_index(row["oracle_next_edit"]) for row in rows], dtype=np.int64)
    masks = np.stack([legal_action_mask(row["public_world"], row["selected"]) for row in rows])
    if not masks[np.arange(len(rows)), labels].all():
        raise ValueError("oracle teacher action is not structurally available")
    return features, labels, masks


def _choose_action(
    model: LinearStudent | RecurrentStudent, task: dict, selected: list[str],
    *, loops: int, stop_head: Any | None = None,
) -> dict[str, str]:
    """Use a learned STOP proposal plus a public structural action mask."""
    vector = encode_public(task, selected)[None, :]
    if stop_head is not None and stop_head.should_stop(vector):
        return {"op": "stop", "id": "none"}
    logits = model.logits(vector, loops=loops)[0]
    mask = legal_action_mask(task, selected)
    if stop_head is not None:
        mask[20] = False
    return decode_action(int(np.argmax(np.where(mask, logits, -1e6))))


def _guarded_action(
    model: LinearStudent | RecurrentStudent, task: dict, selected: list[str],
    *, loops: int, stop_head: Any | None,
) -> tuple[dict[str, str], int, int]:
    """Environment-owned veto: an ADD may not introduce a new violation.

    The model and stop head propose scores; the independent public constraint
    checker controls only the allowed ADD transition. It never reveals the
    hidden exact optimal selection or canonical next action.
    """
    vector = encode_public(task, selected)[None, :]
    if stop_head is not None and stop_head.should_stop(vector):
        return {"op": "stop", "id": "none"}, 0, 0
    logits = model.logits(vector, loops=loops)[0]
    allowed = legal_action_mask(task, selected)
    if stop_head is not None:
        allowed[20] = False
    ordering = np.argsort(-logits, kind="stable")
    vetoes = 0
    queries = 0
    before: set[str] | None = None
    for index in ordering:
        if not allowed[index]:
            continue
        action = decode_action(int(index))
        if action["op"] == "add":
            if before is None:
                before = set(evaluate_subset(task, selected)["violations"])
                queries += 1
            proposed = edit(selected, action)
            after = set(evaluate_subset(task, proposed)["violations"])
            queries += 1
            if after - before:
                vetoes += 1
                continue
        return action, vetoes, queries
    return {"op": "abstain", "id": "none"}, vetoes, queries


def _next_action(
    model: LinearStudent | RecurrentStudent, task: dict, selected: list[str],
    *, loops: int, stop_head: Any | None, verifier_guard: bool,
) -> tuple[dict[str, str], int, int]:
    if verifier_guard:
        return _guarded_action(model, task, selected, loops=loops, stop_head=stop_head)
    return _choose_action(model, task, selected, loops=loops, stop_head=stop_head), 0, 0


def _rollout(
    model: LinearStudent | RecurrentStudent, row: dict,
    *, loops: int, max_edits: int = 8, stop_head: Any | None = None,
    verifier_guard: bool = False,
) -> dict[str, Any]:
    """Judge an entire attempted repair with the environment-owned oracle."""
    task = row["public_world"]
    selected = list(row["selected"])
    oracle = exact_optimum(task)
    initial, initial_vetoes, initial_checks = _next_action(
        model, task, selected, loops=loops, stop_head=stop_head, verifier_guard=verifier_guard
    )
    exact_initial = initial == row["oracle_next_edit"]
    attempts = 0
    decisions = 0
    false_admissions = 0
    premature_stop = 0
    invalid_action = 0
    rejected = 0
    verifier_queries = 0
    abstentions = 0
    new_violation_additions = 0
    successful_stop = False
    ever_optimal = selected == oracle["selected"]
    for step in range(max_edits):
        if step == 0:
            action, vetoed, checks = initial, initial_vetoes, initial_checks
        else:
            action, vetoed, checks = _next_action(
                model, task, selected, loops=loops, stop_head=stop_head,
                verifier_guard=verifier_guard,
            )
        decisions += 1
        rejected += vetoed
        verifier_queries += checks
        if action["op"] == "abstain":
            abstentions += 1
            break
        if action["op"] == "stop":
            successful_stop = selected == oracle["selected"]
            premature_stop = int(not successful_stop)
            break
        if (
            (action["op"] == "add" and action["id"] in selected)
            or (action["op"] == "remove" and action["id"] not in selected)
            or (action["op"] == "add"
                and action["id"] not in {item["id"] for item in task["items"]})
        ):
            invalid_action = 1
            break
        previous_violations = (
            set(evaluate_subset(task, selected)["violations"])
            if action["op"] == "add" else set()
        )
        selected = edit(selected, action)
        attempts += 1
        ever_optimal |= selected == oracle["selected"]
        if action["op"] == "add":
            current_state = evaluate_subset(task, selected)
            if not current_state["valid"]:
                false_admissions += 1
            if set(current_state["violations"]) - previous_violations:
                new_violation_additions += 1
    if verifier_guard and new_violation_additions:
        raise ValueError("environment veto allowed a newly violating ADD action")
    final = evaluate_subset(task, selected)
    oracle_coverage = len(oracle["verified_objective"]["covered_goals"])
    return {
        "exact_first_edit": int(exact_initial),
        "solved_and_stopped": int(successful_stop),
        "ever_reached_optimum": int(ever_optimal),
        "premature_stop": premature_stop,
        "inapplicable_action": invalid_action,
        "unsafe_addition": false_admissions,
        "new_violation_addition": new_violation_additions,
        "verifier_vetoes": rejected,
        "verifier_queries": verifier_queries,
        "abstentions": abstentions,
        "final_valid": int(final["valid"]),
        "goals_covered": len(final["covered_goals"]),
        "oracle_goals_covered": oracle_coverage,
        "remaining_canonical_edits": len(set(selected) ^ set(oracle["selected"])),
        "repair_edits": attempts,
        "inference_iterations": decisions * loops,
        "learned_stop_head_calls": decisions if stop_head is not None else 0,
    }


def evaluate(
    model: LinearStudent | RecurrentStudent, rows: list[dict],
    *, loops: int, stop_head: Any | None = None, verifier_guard: bool = False,
) -> dict[str, Any]:
    measures = [
        _rollout(model, row, loops=loops, stop_head=stop_head, verifier_guard=verifier_guard)
        for row in rows
    ]
    if not measures:
        raise ValueError("empty evaluation split")
    count = len(measures)
    sums = {key: sum(row[key] for row in measures) for key in measures[0]}
    return {
        "cases": count,
        "exact_first_edit": sums["exact_first_edit"],
        "solved_and_stopped": sums["solved_and_stopped"],
        "ever_reached_optimum": sums["ever_reached_optimum"],
        "premature_stops": sums["premature_stop"],
        "inapplicable_actions": sums["inapplicable_action"],
        "unsafe_additions": sums["unsafe_addition"],
        "new_violation_additions": sums["new_violation_addition"],
        "verifier_vetoes": sums["verifier_vetoes"],
        "verifier_queries": sums["verifier_queries"],
        "abstentions": sums["abstentions"],
        "final_valid": sums["final_valid"],
        "mean_goals_covered": round(sums["goals_covered"] / count, 5),
        "mean_remaining_canonical_edits": round(sums["remaining_canonical_edits"] / count, 5),
        "mean_repair_edits": round(sums["repair_edits"] / count, 5),
        "mean_inference_iterations": round(sums["inference_iterations"] / count, 5),
        "mean_learned_stop_head_calls": round(sums["learned_stop_head_calls"] / count, 5),
    }


def run_students(bundle: Path, *, steps: int = 48, lr: float = 0.20,
                 seed: int = 20261122, hidden: int = 18) -> dict[str, Any]:
    if not 1 <= steps <= 256 or not 0.001 <= lr <= 0.4:
        raise ValueError("student optimization exceeds bounded CPU budget")
    # An independent full replay authorizes this bounded toy research fit,
    # without marking any candidate as released training material.
    from .airis02_audit import audit_bundle

    audit = audit_bundle(bundle, full_replay=True)
    if audit["synthetic_research_fit_eligible"] is not True or audit["general_corpus_training_ready"] is not False:
        raise ValueError("synthetic research fit is not independently admissible")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "metta-airis02-bundle-v2" or manifest.get("training_ready") is not False:
        raise ValueError("unknown or admitted training bundle")
    for filename, expected in manifest["files"].items():
        if sha256_bytes((bundle / filename).read_bytes()) != expected:
            raise ValueError("tampered training bundle")
    def load(filename: str) -> list[dict]:
        return [json.loads(line) for line in (bundle / filename).read_text(encoding="utf-8").splitlines() if line]
    train = load("train_candidate.jsonl")
    dev = load("development_labels_research.jsonl")
    # Every fitted model is scored on the SAME public development worlds.
    # The uniform policy's frozen worlds form a common task allocation.
    shared_dev = [row for row in dev if row["arm"] == "uniform"]
    if {row["root"] for row in train} & {row["root"] for row in dev}:
        raise ValueError("train/development root leakage")
    if (
        len(train) != manifest["train_presentations_total"]
        or len(dev) != manifest["development_diagnostic_presentations"]
    ):
        raise ValueError("unmatched materialized row counts")
    # The separate binary STOP head uses only independently verified TRAIN
    # labels; no development root can influence threshold selection.
    from .airis02_stop import StopHead

    report = {
        "schema_version": SCHEMA,
        "model_trained": True,
        "model_kind": "from_scratch_numpy_linear_and_weight_tied_recurrent",
        "native_metta_executed": False,
        "source_data_admitted": False,
        "training_ready": False,
        "holdout_steward_opened": False,
        "root_split_custody": "distinct_seed_bands_and_root_hashes",
        "optimizer_steps_per_arm": steps,
        "student_training_policy": "structural_membership_action_mask_plus_hash_fixed_point_stop",
        "learned_stop_policy": "independent_train_only_calibrated_binary_head",
        "verifier_gate_policy": "reject_additions_introducing_new_synthetic_constraint_violations",
        "learning_rate": lr,
        "training_loops": 4,
        "inference_depths_checked": [2, 4, 8],
        "train_cases_per_arm": manifest["train_roots"],
        "development_cases_per_arm": manifest["development_roots"],
        "one_example_per_root_per_arm": True,
        "development_evaluation_world_policy": "shared_uniform_intervention",
        "development_evaluation_worlds_sha256": sha256_bytes(
            "\n".join(canonical_eval_key(row) for row in shared_dev).encode()
        ),
        "trained_models": {},
        "bundle_manifest_sha256": sha256_bytes((bundle / "manifest.json").read_bytes()),
        "research_fit_audit_sha256": sha256_bytes(
            json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
        ),
        "synthetic_research_fit_eligible": True,
        "limitations": [
            "Development roots are publicly regenerable; no sealed accuracy claimed.",
            "Recurrent depths 2/4/8 share the weights fitted at depth 4.",
            "STOP is an action class, not an adaptive computation-time exit head.",
            "The added binary STOP head learns action-level halting; it does not vary recurrent compute depth.",
            "Direct and recurrent models have different parameter counts; "
            "matched updates apply within each architecture.",
            "Inputs are fixed-size structured synthetic facts, not native MeTTa tokens.",
            "All arms are evaluated on the exact same uniform-intervened development worlds.",
            "Equal source-root counts and optimizer steps do not establish equal source-token budgets.",
        ],
    }
    for arm in ARMS:
        a_train = [row for row in train if row["arm"] == arm]
        a_dev = [row for row in dev if row["arm"] == arm]
        if len(a_train) != manifest["train_roots"] or len(a_dev) != manifest["development_roots"]:
            raise ValueError("unbalanced intervention arms")
        x, y, masks = _matrix(a_train)
        linear = LinearStudent(seed)
        losses = linear.fit(x, y, steps=steps, lr=lr, legal_masks=masks)
        rec = RecurrentStudent(seed, hidden=hidden)
        rec_losses = rec.fit(x, y, steps=steps, lr=lr, train_loops=4, legal_masks=masks)
        stop_head = StopHead(seed)
        stop_receipt = stop_head.fit(x, y == 20, steps=steps, lr=lr)
        report["trained_models"][arm] = {
            "train_root_sha256": sha256_bytes("\n".join(sorted(row["root"] for row in a_train)).encode()),
            "dev_root_sha256": sha256_bytes("\n".join(sorted(row["root"] for row in a_dev)).encode()),
            "stop_head": {
                **stop_receipt,
                "parameters": stop_head.params,
                "checkpoint_sha256": stop_head.digest(),
            },
            "linear": {
                "parameters": linear.params,
                "checkpoint_sha256": linear.digest(),
                "final_training_loss": losses[-1],
                "development": evaluate(linear, shared_dev, loops=1),
                "learned_stop_development": evaluate(linear, shared_dev, loops=1, stop_head=stop_head),
                "verified_stop_development": evaluate(
                    linear, shared_dev, loops=1, stop_head=stop_head, verifier_guard=True
                ),
                "within_arm_diagnostic": evaluate(linear, a_dev, loops=1),
            },
            "recurrent": {
                "parameters": rec.params,
                "checkpoint_sha256": rec.digest(),
                "final_training_loss": rec_losses[-1],
                "development": {str(k): evaluate(rec, shared_dev, loops=k) for k in (2, 4, 8)},
                "learned_stop_development": {
                    str(k): evaluate(rec, shared_dev, loops=k, stop_head=stop_head)
                    for k in (2, 4, 8)
                },
                "verified_stop_development": evaluate(
                    rec, shared_dev, loops=4, stop_head=stop_head, verifier_guard=True
                ),
                "within_arm_diagnostic": evaluate(rec, a_dev, loops=4),
            },
            "training_targets": dict(sorted(Counter(int(y_i) for y_i in y).items())),
        }
    first = report["trained_models"][ARMS[0]]
    for arm in ARMS[1:]:
        item = report["trained_models"][arm]
        if (
            item["train_root_sha256"] != first["train_root_sha256"]
            or item["dev_root_sha256"] != first["dev_root_sha256"]
        ):
            raise ValueError("different source root allocations across intervention arms")
        if (
            item["linear"]["parameters"] != first["linear"]["parameters"]
            or item["recurrent"]["parameters"] != first["recurrent"]["parameters"]
        ):
            raise ValueError("model capacity differs across intervention arms")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--lr", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--hidden", type=int, default=18)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = run_students(args.bundle, steps=args.steps, lr=args.lr,
                          seed=args.seed, hidden=args.hidden)
    target = args.out or args.bundle / "student_development.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "train_per_arm": report["train_cases_per_arm"],
        "development_per_arm": report["development_cases_per_arm"],
        "recurrent_depth": report["training_loops"],
        "results": {arm: {
            "linear_solved": item["linear"]["development"]["solved_and_stopped"],
            "recurrent_4_solved": item["recurrent"]["development"]["4"]["solved_and_stopped"],
            "recurrent_4_first_edit": item["recurrent"]["development"]["4"]["exact_first_edit"],
            "recurrent_4_learned_stop_solved": item["recurrent"]["learned_stop_development"]["4"][
                "solved_and_stopped"
            ],
            "recurrent_4_verified_stop_solved": item["recurrent"]["verified_stop_development"][
                "solved_and_stopped"
            ],
        } for arm, item in report["trained_models"].items()},
        "training_ready": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
