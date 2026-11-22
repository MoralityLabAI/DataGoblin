"""AGN-02: offline research-agonist selection and downstream-student pilot.

This is a public synthetic development fixture. The learned selector is a small
ridge regressor, not the pretrained DataGoblin model. No teacher inference, paid
compute, real-world actions, sealed roots, or production data admission occurs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from Agonist.gym import (
    canonical,
    digest,
    new_dir,
    read_jsonl,
    sha256_file,
    validate_tastes,
    verify_pair,
    write_jsonl,
)

SCHEMA = "datagoblin.agonist02.dev.v1"
COHORT = {"anchor": 10, "pool": 24, "calibration": 8, "evaluation": 8}
ARM_NAMES = ("anchor_only", "random", "fixed", "learned")
STUDENT_SEEDS = (11, 23, 47)
THEMES = (
    "a simulated scientific field station",
    "a fictional logistics expedition",
    "a cooperative strategy game",
    "a simulated archive repair lab",
    "a fictional asteroid research habitat",
)
STAGES = ("initial", "revised", "check")


def make_tastes(seed: int) -> list[dict[str, Any]]:
    """Original fictional premises: 50 independently identified root families."""
    rows = []
    for index in range(sum(COHORT.values())):
        theme = THEMES[index % len(THEMES)]
        rows.append(
            {
                "taste_id": f"agn02-taste-{seed}-{index:03d}",
                "family_id": f"agn02-family-{seed}-{index:03d}",
                "idea": f"Test whether an agent revises a decision in {theme} after one causal prerequisite changes.",
                "source_kind": "fixture",
                "source_ref": f"AGn02 original synthetic seed {seed}/{index}",
                "rights": "project_original",
                "split": "train" if index < 34 else "dev",
            }
        )
    return rows


def _assign_cohorts(tastes: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    validate_tastes(tastes)
    if len(tastes) != 50 or len({t["family_id"] for t in tastes}) != 50:
        raise ValueError("AGN-02 development fixture needs 50 independent taste families")
    train = sorted((t for t in tastes if t["split"] == "train"), key=lambda t: t["taste_id"])
    dev = sorted((t for t in tastes if t["split"] == "dev"), key=lambda t: t["taste_id"])
    if len(train) != 34 or len(dev) != 16:
        raise ValueError("required custody is 34 train families and 16 dev families")
    if any(t["rights"] == "reference_only" for t in train):
        raise ValueError("reference-only taste cannot supply a training root")
    return [
        *(("anchor", t) for t in train[:10]),
        *(("pool", t) for t in train[10:]),
        *(("calibration", t) for t in dev[:8]),
        *(("evaluation", t) for t in dev[8:]),
    ]


def generate_root(taste: dict[str, Any], cohort: str, seed: int) -> dict[str, Any]:
    """Deterministic initial->revised world; the label comes from AGN-01's oracle."""
    rng = random.Random(int(digest([taste["taste_id"], seed])[:16], 16))
    required = rng.randint(2, 7)
    positive = {
        "available": required + rng.randint(0, 2),
        "required": required,
        "safety_approved": True,
        "source_fresh": True,
    }
    negative = dict(positive)
    field = rng.choice(("available", "safety_approved", "source_fresh"))
    negative[field] = required - 1 if field == "available" else False
    first_positive = bool(rng.getrandbits(1))
    initial_state = positive if first_positive else negative
    revised_state = negative if first_positive else positive
    surface = rng.choice(("red", "green"))
    theme = THEMES[int(digest(taste["family_id"])[:6], 16) % len(THEMES)]
    shared = {
        "setting": "Fictional research trial " + taste["taste_id"][:48],
        "scenario": (
            f"An agent in {theme} must decide on a bounded deployment "
            "from the current rule and source evidence."
        ),
        "surface_cue": surface,
    }
    world_a = {**shared, "state": dict(initial_state)}
    world_b = {**shared, "state": dict(revised_state)}
    intervention = {
        "field": field,
        "from": initial_state[field],
        "to": revised_state[field],
    }
    verify_pair(world_a, world_b, intervention)
    return {
        "schema": SCHEMA,
        "root": "agn02-" + digest([taste["taste_id"], seed])[:24],
        "taste_id": taste["taste_id"],
        "family_id": taste["family_id"],
        "cohort": cohort,
        "source_kind": taste["source_kind"],
        "rights": taste["rights"],
        "taste_digest": digest(taste["idea"]),
        "generation_kind": "offline_rule_fixture_no_teacher",
        "world_a": world_a,
        "world_b": world_b,
        "intervention": intervention,
        "check_case": {
            "hazard_percent": rng.choice((10, 25, 40, 65, 85)),
            "check_cost": rng.choice((1, 2, 3)),
            "mistake_cost": rng.choice((4, 8, 12)),
        },
    }


def check_value(root: dict[str, Any]) -> int:
    """Expected value, in hundredths of a utility unit, of another check.

    The announced exogenous source-change hazard matters only when structural
    prerequisites allow deployment. A tie is resolved by stopping.
    """
    state = root["world_b"]["state"]
    case = root["check_case"]
    if not (
        type(case["hazard_percent"]) is int
        and type(case["check_cost"]) is int
        and type(case["mistake_cost"]) is int
        and case["hazard_percent"] in (10, 25, 40, 65, 85)
        and case["check_cost"] in (1, 2, 3)
        and case["mistake_cost"] in (4, 8, 12)
    ):
        raise ValueError("invalid prospective check case")
    if state["available"] < state["required"] or not state["safety_approved"]:
        return -100 * case["check_cost"]
    return case["hazard_percent"] * case["mistake_cost"] - 100 * case["check_cost"]


def exact_actions(root: dict[str, Any]) -> tuple[str, str, str]:
    first, revised = verify_pair(root["world_a"], root["world_b"], root["intervention"])
    return first, revised, "check" if check_value(root) > 0 else "stop"


def validate_root(root: dict[str, Any]) -> None:
    if root.get("schema") != SCHEMA or root.get("cohort") not in COHORT:
        raise ValueError("invalid AGN-02 root schema or cohort")
    if not all(isinstance(root.get(k), str) for k in ("root", "taste_id", "family_id", "rights")):
        raise ValueError("invalid root identity")
    if root["rights"] == "reference_only" and root["cohort"] in ("anchor", "pool"):
        raise ValueError("unauthorized reference-only training root")
    exact_actions(root)


def construct_roots(seed: int, tastes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    roots = [generate_root(taste, group, seed) for group, taste in _assign_cohorts(tastes)]
    if len({r["root"] for r in roots}) != 50:
        raise ValueError("duplicate root")
    for row in roots:
        validate_root(row)
    return roots


def _features(world: dict[str, Any], check_case: dict[str, int] | None = None) -> np.ndarray:
    state = world["state"]
    # The later value-of-information case is visible only at the check stage.
    check_case = check_case or {"hazard_percent": 0, "check_cost": 0, "mistake_cost": 0}
    return np.asarray(
        (
            1.0,
            (state["available"] - state["required"]) / 6.0,
            float(state["available"] >= state["required"]),
            float(state["safety_approved"]),
            float(state["source_fresh"]),
            float(world["surface_cue"] == "green"),
            check_case["hazard_percent"] / 100.0,
            check_case["check_cost"] / 3.0,
            check_case["mistake_cost"] / 12.0,
            check_case["hazard_percent"] * check_case["mistake_cost"] / 1200.0,
        ),
        dtype=np.float64,
    )


def _examples(root: dict[str, Any]) -> list[tuple[int, np.ndarray, int]]:
    labels = exact_actions(root)
    return [
        (0, _features(root["world_a"]), int(labels[0] == "deploy")),
        (0, _features(root["world_b"]), int(labels[1] == "deploy")),
        (1, _features(root["world_b"], root["check_case"]), int(labels[2] == "check")),
    ]


def _sigmoid(value: float) -> float:
    bounded = max(-40.0, min(40.0, value))
    return float(1.0 / (1.0 + np.exp(-bounded)))


def fit_student(roots: list[dict[str, Any]], seed: int, updates: int = 240) -> np.ndarray:
    """Genuine small supervised linear-policy training with equal update count."""
    if not roots or updates < 1:
        raise ValueError("student training requires roots and positive updates")
    rows = [example for root in roots for example in _examples(root)]
    weights = np.zeros((2, 10), dtype=np.float64)
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    for step in range(updates):
        if step and step % len(order) == 0:
            rng.shuffle(order)
        head, features, gold = rows[order[step % len(order)]]
        p = _sigmoid(float(weights[head] @ features))
        weights[head] -= 0.15 * ((p - gold) * features + 0.002 * weights[head])
    return weights


def _student_decisions(weights: np.ndarray, root: dict[str, Any]) -> tuple[bool, bool, bool]:
    examples = _examples(root)
    return tuple(
        bool((_sigmoid(float(weights[head] @ features)) >= 0.5) == bool(label))
        for head, features, label in examples
    )


def _evaluation_rate(weights: np.ndarray, roots: list[dict[str, Any]]) -> float:
    if not roots:
        raise ValueError("empty calibration roots")
    return sum(sum(_student_decisions(weights, root)) for root in roots) / (len(roots) * 3)


def _selector_features(root: dict[str, Any]) -> np.ndarray:
    """Public and train-verifier-derived curation features, never eval labels."""
    a, b, check = exact_actions(root)
    case = root["check_case"]
    state = root["world_b"]["state"]
    cue = "deploy" if root["world_a"]["surface_cue"] == "green" else "hold"
    field = root["intervention"]["field"]
    return np.asarray(
        (
            1.0,
            float(field == "available"),
            float(field == "safety_approved"),
            float(field == "source_fresh"),
            float(cue != a),
            float(a == "deploy"),
            float(check == "check"),
            abs(state["available"] - state["required"]) / 6.0,
            case["hazard_percent"] * case["mistake_cost"] / 1200.0,
            case["check_cost"] / 3.0,
            float(b == "deploy"),
        ),
        dtype=np.float64,
    )


def _calibrate_selector(
    anchors: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
    updates: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Leave-one-out marginal utility, regressed on ten train-only roots.

    Calibration outcomes train this small research-utility proxy. The separate
    evaluation cohort is never used for utility fitting or choosing candidates.
    """
    full_student = fit_student(anchors, seed=41, updates=updates)
    full_accuracy = _evaluation_rate(full_student, calibration)
    observed: list[dict[str, Any]] = []
    x: list[np.ndarray] = []
    y: list[float] = []
    for i, root in enumerate(anchors):
        without = anchors[:i] + anchors[i + 1 :]
        baseline = fit_student(without, seed=41, updates=updates)
        delta = full_accuracy - _evaluation_rate(baseline, calibration)
        x.append(_selector_features(root))
        y.append(delta)
        observed.append(
            {
                "root": root["root"],
                "cohort": "anchor",
                "utility_delta_calibration_decision_accuracy": round(delta, 9),
                "family_id": root["family_id"],
            }
        )
    matrix = np.stack(x)
    targets = np.asarray(y, dtype=np.float64)
    regularizer = np.eye(matrix.shape[1]) * 0.5
    regularizer[0, 0] = 0.05
    weights = np.linalg.solve(matrix.T @ matrix + regularizer, matrix.T @ targets)
    return weights, observed


def _fixed_score(root: dict[str, Any]) -> float:
    """Frozen heuristic: favor cue-counterexamples and narrow resource margins."""
    a, _, c = exact_actions(root)
    cue = "deploy" if root["world_a"]["surface_cue"] == "green" else "hold"
    margin = abs(
        root["world_b"]["state"]["available"] - root["world_b"]["state"]["required"]
    )
    return 4 * float(cue != a) + 2 * float(c == "check") + 1 / (1 + margin)


def select_roots(
    anchors: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
    seed: int,
    updates: int,
    selection_count: int = 12,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    if len(anchors) != 10 or len(pool) != 24 or len(calibration) != 8:
        raise ValueError("incorrect selection cohort dimensions")
    if not 1 <= selection_count <= len(pool):
        raise ValueError("selection_count out of range")
    utility_weights, observations = _calibrate_selector(anchors, calibration, updates)
    random_pool = list(pool)
    random.Random(seed + 4901).shuffle(random_pool)
    learned = sorted(
        pool,
        key=lambda root: (-float(utility_weights @ _selector_features(root)), root["root"]),
    )
    fixed = sorted(pool, key=lambda root: (-_fixed_score(root), root["root"]))
    selections = {
        "anchor_only": list(anchors),
        "random": anchors + random_pool[:selection_count],
        "fixed": anchors + fixed[:selection_count],
        "learned": anchors + learned[:selection_count],
    }
    meta = {
        "selector": "small_ridge_utility_proxy",
        "meta_train_roots": len(anchors),
        "calibration_roots": len(calibration),
        "calibration_student_fits": len(anchors) + 1,
        "selector_weights": [round(float(x), 12) for x in utility_weights],
        "all_arms_share_verified_pool": True,
        "learned_controller_is_data_goblin_1b": False,
    }
    return selections, observations, meta


def _paired_differences(
    rows: list[dict[str, Any]],
    seed: int,
    replicates: int = 500,
) -> dict[str, Any]:
    """Root-clustered descriptive bootstrap, with all student seeds within a root."""
    grouped: dict[str, dict[str, list[int]]] = {}
    for row in rows:
        grouped.setdefault(row["root"], {}).setdefault(row["arm"], []).append(
            int(row["whole_episode_correct"])
        )
    root_ids = sorted(grouped)
    for root in root_ids:
        if set(grouped[root]) != set(ARM_NAMES):
            raise ValueError("incomplete paired root/arm score matrix")
        if any(len(grouped[root][arm]) != len(STUDENT_SEEDS) for arm in ARM_NAMES):
            raise ValueError("missing student seed for root/arm")
    rng = random.Random(seed + 118)
    report = {}
    for comparator in ("random", "fixed", "anchor_only"):
        differences = [
            (sum(grouped[root]["learned"]) - sum(grouped[root][comparator])) / len(STUDENT_SEEDS)
            for root in root_ids
        ]
        mean = sum(differences) / len(differences)
        boot = []
        for _ in range(replicates):
            sample = [differences[rng.randrange(len(differences))] for _ in root_ids]
            boot.append(sum(sample) / len(sample))
        boot.sort()
        report[f"learned_minus_{comparator}"] = {
            "mean_whole_episode_fraction": round(mean, 6),
            "descriptive_root_bootstrap_95": [
                round(boot[int(0.025 * replicates)], 6),
                round(boot[int(0.975 * replicates) - 1], 6),
            ],
        }
    return report


def simulate(
    seed: int = 17,
    updates: int = 240,
    selection_count: int = 12,
    tastes: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not 1 <= updates <= 5_000:
        raise ValueError("student updates outside the predeclared development cap")
    actual_tastes = make_tastes(seed) if tastes is None else tastes
    roots = construct_roots(seed, actual_tastes)
    by_cohort = {name: [r for r in roots if r["cohort"] == name] for name in COHORT}
    selections, utility_rows, utility_meta = select_roots(
        by_cohort["anchor"],
        by_cohort["pool"],
        by_cohort["calibration"],
        seed,
        updates,
        selection_count,
    )
    oracle_receipts = [
        {
            "root": root["root"],
            "cohort": root["cohort"],
            "family_id": root["family_id"],
            "world_digest": digest([root["world_a"], root["world_b"], root["intervention"]]),
            "actions": list(exact_actions(root)),
            "check_value_hundredths": check_value(root),
            "verifier": "agn01_exact_actions_plus_agn02_expected_check_v1",
        }
        for root in roots
    ]
    selected_rows = [
        {
            "arm": arm,
            "roots": [r["root"] for r in selected],
            "selected_pool_roots": len(selected) - len(by_cohort["anchor"]),
            "train_examples": 3 * len(selected),
        }
        for arm, selected in selections.items()
    ]
    scores = []
    for arm in ARM_NAMES:
        selected = selections[arm]
        for student_seed in STUDENT_SEEDS:
            weights = fit_student(selected, student_seed, updates)
            for root in by_cohort["evaluation"]:
                results = _student_decisions(weights, root)
                scores.append(
                    {
                        "arm": arm,
                        "student_seed": student_seed,
                        "root": root["root"],
                        "family_id": root["family_id"],
                        "cohort": "evaluation",
                        "phase_correct": {
                            stage: correct for stage, correct in zip(STAGES, results, strict=True)
                        },
                        "whole_episode_correct": all(results),
                        "correct_decisions": sum(results),
                        "checked_decisions": 3,
                        "updates": updates,
                    }
                )
    whole = {
        arm: {
            "whole_episodes": sum(int(x["whole_episode_correct"]) for x in scores if x["arm"] == arm),
            "correct_decisions": sum(x["correct_decisions"] for x in scores if x["arm"] == arm),
            "episodes": len(by_cohort["evaluation"]) * len(STUDENT_SEEDS),
            "decisions": 3 * len(by_cohort["evaluation"]) * len(STUDENT_SEEDS),
        }
        for arm in ARM_NAMES
    }
    extra = {
        "schema": SCHEMA,
        "phase": "public_synthetic_dev_only",
        "seed": seed,
        "training_updates_per_student": updates,
        "selection_count": selection_count,
        "cohorts": COHORT,
        "student_seeds": list(STUDENT_SEEDS),
        "source_tastes": "original_fixture" if tastes is None else "operator_supplied",
        "teacher_inference_calls": 0,
        "teacher_tokens": None,
        "candidate_verifier_decisions": 3 * len(roots),
        "evaluation_roots": len(by_cohort["evaluation"]),
        "selector_training": utility_meta,
        "student_results": whole,
        "paired": _paired_differences(scores, seed),
        "scientific_claim": "development-only mechanics and learned surrogate comparison",
        "sealed_confirmation_run": False,
        "training_ready": False,
    }
    return {
        "roots.jsonl": roots,
        "oracle_receipts.jsonl": oracle_receipts,
        "selector_training.jsonl": utility_rows,
        "selections.jsonl": selected_rows,
        "student_scores.jsonl": scores,
    }, extra


def _make_manifest(folder: Path, files: dict[str, list[dict[str, Any]]], extra: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        **extra,
        "artifacts": {
            name: {"rows": len(rows), "sha256": sha256_file(folder / name)}
            for name, rows in sorted(files.items())
        },
    }
    (folder / "manifest.json").write_text(canonical(manifest) + "\n", encoding="utf-8")
    return manifest


def run(
    out: Path,
    seed: int = 17,
    updates: int = 240,
    selection_count: int = 12,
    tastes_path: Path | None = None,
) -> dict[str, Any]:
    tastes = read_jsonl(tastes_path) if tastes_path else None
    files, extra = simulate(seed, updates, selection_count, tastes)
    if tastes_path:
        extra["tastes_sha256"] = sha256_file(tastes_path)
    new_dir(out)
    for name, rows in files.items():
        write_jsonl(out / name, rows)
    return _make_manifest(out, files, extra)


def replay(out: Path, tastes_path: Path | None = None) -> dict[str, Any]:
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("phase") != "public_synthetic_dev_only":
        raise ValueError("invalid AGN-02 replay manifest")
    supplied = manifest["source_tastes"] == "operator_supplied"
    if supplied != (tastes_path is not None):
        raise ValueError("source taste file required exactly for operator-supplied archive")
    if supplied and sha256_file(tastes_path) != manifest["tastes_sha256"]:
        raise ValueError("taste input hash mismatch")
    tastes = read_jsonl(tastes_path) if supplied else None
    expected_files, expected = simulate(
        manifest["seed"], manifest["training_updates_per_student"], manifest["selection_count"], tastes
    )
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"experiment manifest differs: {key}")
    if set(manifest["artifacts"]) != set(expected_files):
        raise ValueError("unexpected artifact inventory")
    for name, rows in expected_files.items():
        path = out / name
        if not path.is_file() or sha256_file(path) != manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"archive hash mismatch: {name}")
        if len(rows) != manifest["artifacts"][name]["rows"]:
            raise ValueError(f"archive row count mismatch: {name}")
        if canonical(read_jsonl(path)) != canonical(rows):
            raise ValueError(f"independent recomputation disagrees: {name}")
    return {
        "verified_roots": len(expected_files["oracle_receipts.jsonl"]),
        "student_score_rows": len(expected_files["student_scores.jsonl"]),
        "recomputed": True,
        "sealed_confirmation_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AGN-02 offline curator/student experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--student-updates", type=int, default=240)
    p.add_argument("--selection-count", type=int, default=12)
    p.add_argument("--tastes", type=Path, default=None)
    p = sub.add_parser("replay")
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--tastes", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "run":
        result = run(args.out, args.seed, args.student_updates, args.selection_count, args.tastes)
    else:
        result = replay(args.archive, args.tastes)
    print(canonical(result))


if __name__ == "__main__":
    main()
