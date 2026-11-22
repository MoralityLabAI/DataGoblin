"""Native Lean finite-menu proof search with TRAIN-only ranker and stopping gate.

All successes are independently checked by the pinned Lean 4 compiler and
LeanKernelChecker. Synthetic tasks require multiple tactic commands and have
both witnessed and menu-exhausted roots. Menu exhaustion is not unprovability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .external_checked_exit import digest
from .external_native_adapters import LeanKernelChecker
from .live_lean_kernel import LeanNativeSession

SPLITS = ("train", "calibration", "ordinary_test", "ood_test")
SOURCE_SEED = 20261124
LEAN_VERSION = "4.12.0"
PER_ROOT_BUDGET = 5
THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
FAMILY_PATTERNS = (
    "(a + b) * (c +",
    "a * (b * (c *",
    "(b + a)",
    "(a * c + b * c)",
)
SCRIPT_PATTERNS = (
    "Nat.mul_add",
    "Nat.add_mul",
    "Nat.mul_assoc",
    "Nat.add_comm a b",
    "Nat.mul_comm a b",
)
SCRIPTS = (
    "intro a b c\nconstructor\n· simp only [Nat.add_mul, Nat.mul_add, Nat.add_assoc]\n"
    "· simp only [Nat.add_assoc]",
    "intro a b c\nconstructor\n· simp only [Nat.mul_assoc]\n"
    "· simp only [Nat.add_assoc]",
    "intro a b c\nconstructor\n· rw [Nat.add_comm a b]\n"
    "· rw [Nat.mul_comm a b]",
    "intro a b c\nconstructor\n· simp only [Nat.add_mul]\n"
    "· simp only [Nat.mul_assoc]",
)
DISTRACTORS = (
    "intro a b c\nconstructor\n· simp only [Nat.add_assoc]\n· simp only [Nat.add_assoc]",
    "intro a b c\nconstructor\n· simp only [Nat.mul_assoc]\n· simp only [Nat.mul_assoc]",
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ProofRoot:
    root_id: str
    split: str
    public_task: dict[str, Any]
    tactics: tuple[str, ...]

    @property
    def statement(self) -> str:
        return self.public_task["statement"]


def _statement(family: int, k: int) -> str:
    if family == 0:
        return (
            f"∀ a b c : Nat, "
            f"((a + b) * (c + {k}) = (a * c + b * c) + (a * {k} + b * {k})) ∧ "
            f"(((a + b) + c) + {k} = a + (b + (c + {k})))"
        )
    if family == 1:
        return (
            f"∀ a b c : Nat, "
            f"(((a * b) * c) * {k} = a * (b * (c * {k}))) ∧ "
            f"(((a + b) + c) + {k} = a + (b + (c + {k})))"
        )
    if family == 2:
        return (
            f"∀ a b c : Nat, "
            f"((a + b) + {k} = (b + a) + {k}) ∧ "
            f"((a * b) * {k} = (b * a) * {k})"
        )
    return (
        f"∀ a b c : Nat, "
        f"(((a + b) * c) * {k} = (a * c + b * c) * {k}) ∧ "
        f"(((a * b) * c) * {k} = a * (b * (c * {k})))"
    )


def generate_roots(*, per_split: int = 8, seed: int = SOURCE_SEED, alpha_ood: bool = False) -> list[ProofRoot]:
    if type(per_split) is not int or per_split < 8 or per_split > 24 or per_split % 4:
        raise ValueError("per_split must be a multiple of four in 8..24")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    result = []
    for split_number, split in enumerate(SPLITS):
        for i in range(per_split):
            family = (i + split_number) % 4
            is_exhausted = i % 4 == (i // 4 + split_number + 1) % 4
            k = (400000 if split == "ood_test" else 401) + 1000 * split_number + i * 53 + (seed % 37)
            scripts = [s for j, s in enumerate(SCRIPTS) if j != family or not is_exhausted]
            scripts.extend(DISTRACTORS[: PER_ROOT_BUDGET - len(scripts)])
            if len(scripts) != PER_ROOT_BUDGET or len(set(scripts)) != len(scripts):
                raise ValueError("duplicate or incomplete candidate menu")
            source_token = _sha(f"{seed}/{split}/{i}".encode())[:18]
            random.Random(int(source_token, 16)).shuffle(scripts)
            statement = _statement(family, k)
            if split == "ood_test" and alpha_ood:
                # Alpha-equivalent statements keep the same proof scripts after "intro".
                renamed = {"a": "u", "b": "v", "c": "w"}
                statement = re.sub(r"\b[abc]\b", lambda m: renamed[m.group()], statement)
            task = {
                "statement": statement,
                "import": "Init",
                "source_token": source_token,
                "candidate_menu_sha256": digest(scripts),
            }
            if any(phrase in statement or phrase in json.dumps(task) for phrase in
                   ("sorry", "admit", "run_tac", "unsafe")):
                raise ValueError("untrusted source root")
            result.append(ProofRoot(
                root_id=f"native-lean-stop/{split}/{source_token}",
                split=split, public_task=task, tactics=tuple(scripts),
            ))
    if len({r.root_id for r in result}) != len(result):
        raise ValueError("source roots collide")
    if len({r.statement for r in result}) != len(result):
        raise ValueError("theorem statements collide")
    return result


def features(statement: str, tactic: str) -> np.ndarray:
    """Use public formula structure and script tokens, never outcome/fixture labels."""
    formula = [float(token in statement) for token in FAMILY_PATTERNS]
    tactics = [float(token in tactic) for token in SCRIPT_PATTERNS]
    return np.array(
        [1.0, *formula, *tactics, *(a * b for a in formula for b in tactics)],
        dtype=np.float64,
    )


def probabilities(root: ProofRoot, weights: np.ndarray) -> np.ndarray:
    x = np.stack([features(root.statement, t) for t in root.tactics])
    p = 1 / (1 + np.exp(-np.clip(x @ weights, -30, 30)))
    return p


def fit_ranker(train: list[tuple[ProofRoot, list[dict[str, Any]]]]) -> np.ndarray:
    if not train or any(root.split != "train" for root, _ in train):
        raise ValueError("ranker must train exclusively on TRAIN source roots")
    X = np.concatenate([
        np.stack([features(root.statement, tactic) for tactic in root.tactics])
        for root, _ in train
    ])
    y = np.array([
        float(a["receipt"]["passed"])
        for _, attempts in train for a in attempts
    ], dtype=np.float64)
    if X.shape[0] != len(y) or not (0 < np.sum(y) < len(y)):
        raise ValueError("the native TRAIN outcomes must contain both classes")
    w = np.zeros(X.shape[1], dtype=np.float64)
    weights = np.where(y == 1, min(12.0, len(y) / max(1, np.sum(y))), 1.0)
    for _ in range(1800):
        q = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        gradient = X.T @ ((q - y) * weights) / len(y)
        gradient[1:] += 0.015 * w[1:]
        w -= 0.25 * gradient
    if not np.isfinite(w).all():
        raise FloatingPointError("ranker diverged")
    return w


def _policy_ids(root: ProofRoot, name: str, weights: np.ndarray | None) -> list[int]:
    if name == "learned":
        if weights is None:
            raise ValueError("learned policy is missing trained weights")
        probs = probabilities(root, weights)
        return sorted(range(len(root.tactics)), key=lambda j: (-float(probs[j]), j))
    if name in ("fixed_one", "fixed_full"):
        return list(range(len(root.tactics)))
    if name == "random_full":
        order = list(range(len(root.tactics)))
        rng = random.Random(int(_sha(f"random_full/{root.root_id}".encode())[:15], 16))
        rng.shuffle(order)
        return order
    raise ValueError("unknown registered arm")


def _should_stop(
    root: ProofRoot, remaining: list[int], weights: np.ndarray,
    threshold: float, attempted: int,
) -> bool:
    if attempted < 1 or not remaining:
        return False
    scores = probabilities(root, weights)
    return max(float(scores[idx]) for idx in remaining) < threshold


def simulate(
    root: ProofRoot,
    receipts: dict[int, dict[str, Any]],
    weights: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """CAL-only counterfactual choices from previously kernel-checked candidates."""
    order = _policy_ids(root, "learned", weights)
    attempts = []
    hit = False
    for offset, idx in enumerate(order):
        if _should_stop(root, order[offset:], weights, threshold, len(attempts)):
            break
        receipt = receipts[idx]
        attempts.append(idx)
        if receipt["passed"]:
            hit = True
            break
    return {"certified": hit, "attempts": len(attempts), "order": attempts}


def calibrate(
    cal: list[tuple[ProofRoot, list[dict[str, Any]]]],
    weights: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    if not cal or any(root.split != "calibration" for root, _ in cal):
        raise ValueError("threshold calibration may use CAL roots only")
    candidates = []
    for threshold in THRESHOLDS:
        passed = 0
        attempts = 0
        for root, all_checks in cal:
            records = {int(row["id"]): row["receipt"] for row in all_checks}
            sim = simulate(root, records, weights, threshold)
            passed += int(sim["certified"])
            attempts += int(sim["attempts"])
        utility = passed - 0.025 * attempts
        candidates.append({
            "threshold": threshold,
            "certified": passed,
            "attempts": attempts,
            "utility": round(utility, 6),
        })
    # Deterministic CAL tie break favors larger threshold when utility ties.
    best = max(candidates, key=lambda row: (row["utility"], row["threshold"]))
    return float(best["threshold"]), candidates


def _make_checker(
    lean_exe: Path, workspace: Path, roots: list[ProofRoot], verifier_sha: str,
) -> tuple[LeanNativeSession, LeanKernelChecker]:
    states = {}
    statements = {}
    for i, root in enumerate(roots, 1):
        states[digest(root.public_task)] = {
            "root_sid": i,
            "menu": dict(enumerate(root.tactics)),
        }
        statements[i] = root.statement
    session = LeanNativeSession(lean_exe, workspace, statements)
    checker = LeanKernelChecker(lean=session, states=states, verifier_sha=verifier_sha)
    return session, checker


def _check(
    checker: LeanKernelChecker, root: ProofRoot, index: int,
) -> dict[str, Any]:
    begin = time.perf_counter()
    receipt = checker(root.public_task, {"id": index})
    if receipt["task_digest"] != digest(root.public_task):
        raise ValueError("native receipt belongs to another theorem")
    if receipt["passed"] and not (
        receipt["code"] == "kernel_certified"
        and receipt["evidence"].get("kernel_invoked")
        and receipt["evidence"].get("kernel_ok")
        and isinstance(receipt["evidence"].get("axioms"), list)
        and set(receipt["evidence"]["axioms"]) <= {"propext", "Quot.sound", "Classical.choice"}
    ):
        raise ValueError(f"missing audited Lean kernel certificate: {receipt['evidence']}")
    return {"id": index, "receipt": receipt,
            "native_wall_s": round(time.perf_counter() - begin, 6)}


def _online(
    root: ProofRoot, checker: LeanKernelChecker, *,
    arm: str, weights: np.ndarray | None, threshold: float,
) -> dict[str, Any]:
    order = _policy_ids(root, arm, weights)
    if len(order) != PER_ROOT_BUDGET:
        raise ValueError("missing candidate actions")
    attempted: list[dict[str, Any]] = []
    stopped_for_low_value = False
    for offset, candidate_id in enumerate(order):
        if arm == "fixed_one" and attempted:
            break
        if arm == "learned" and weights is not None and _should_stop(
            root, order[offset:], weights, threshold, len(attempted)
        ):
            stopped_for_low_value = True
            break
        row = _check(checker, root, candidate_id)
        attempted.append(row)
        if row["receipt"]["passed"]:
            break
    return {
        "root_id": root.root_id, "split": root.split, "arm": arm,
        "attempts": attempted,
        "certified": bool(attempted and attempted[-1]["receipt"]["passed"]),
        "stopped_for_low_value": stopped_for_low_value,
        "native_wall_s": round(sum(a["native_wall_s"] for a in attempted), 6),
    }


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "roots": len(rows),
        "kernel_certified": sum(bool(r["certified"]) for r in rows),
        "candidate_attempts": sum(len(r["attempts"]) for r in rows),
        "native_wall_s": round(sum(r["native_wall_s"] for r in rows), 5),
        "value_stops": sum(bool(r["stopped_for_low_value"]) for r in rows),
    }


def run(lean_exe: Path, out: Path, *, per_split: int = 8, seed: int = SOURCE_SEED,
        alpha_ood: bool = False) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("create-only native evidence directory required")
    if not lean_exe.is_file():
        raise ValueError("Lean compiler binary is missing")
    version = subprocess.run(
        [str(lean_exe), "--version"], capture_output=True, text=True, check=True, timeout=15
    ).stdout.strip()
    if LEAN_VERSION not in version:
        raise ValueError(f"unexpected Lean compiler version: {version}")
    binary_sha = _sha(lean_exe.read_bytes())
    driver_sha = _sha(Path(__file__).read_bytes())
    root_specs = generate_roots(per_split=per_split, seed=seed, alpha_ood=alpha_ood)
    source_revision = digest({
        "driver_sha": driver_sha, "seed": seed, "lean_sha": binary_sha,
        "roots": [(r.root_id, digest(r.public_task)) for r in root_specs],
    })
    verifier_sha = digest({"driver_sha": driver_sha, "lean_sha": binary_sha,
                           "source_revision": source_revision, "checker": "LeanKernelChecker"})
    original = {split: [r for r in root_specs if r.split == split] for split in SPLITS}
    start = time.perf_counter()
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="dg-rlt-native-stop-") as tmp:
        session, checker = _make_checker(lean_exe, Path(tmp), root_specs, verifier_sha)
        train: list[tuple[ProofRoot, list[dict[str, Any]]]] = []
        calibration: list[tuple[ProofRoot, list[dict[str, Any]]]] = []
        for subset, dest in (("train", train), ("calibration", calibration)):
            for root in original[subset]:
                dest.append((root, [_check(checker, root, i) for i in range(PER_ROOT_BUDGET)]))
        weights = fit_ranker(train)
        threshold, cal_frontier = calibrate(calibration, weights)
        live_rows: list[dict[str, Any]] = []
        for split in ("ordinary_test", "ood_test"):
            for root in original[split]:
                for arm in ("learned", "fixed_one", "fixed_full", "random_full"):
                    live_rows.append(_online(root, checker, arm=arm, weights=weights, threshold=threshold))
        replays = 0
        for row in live_rows:
            if row["arm"] != "learned":
                continue
            root = next(x for x in original[row["split"]] if x.root_id == row["root_id"])
            for attempt in row["attempts"]:
                rechecked = _check(checker, root, int(attempt["id"]))
                if rechecked["receipt"] != attempt["receipt"]:
                    raise ValueError("native independent replay receipt mismatch")
                replays += 1
        proofs = out / "certified_proofs"
        proofs.mkdir()
        source_hashes = {}
        for _sid, (source, _name) in sorted(session.proof_by_sid.items()):
            dest = proofs / source.name
            shutil.copyfile(source, dest)
            source_hashes[dest.relative_to(out).as_posix()] = _sha(dest.read_bytes())
        total_compilations = session.calls + session.kernel_replays
        total_compiler_seconds = session.seconds
    weights_file = out / "fitted_weights.json"
    weights_file.write_text(json.dumps(
        {"weights": weights.tolist(), "TRAIN_only": True,
         "feature_names": [
             "bias", *(f"formula_{i}" for i in range(len(FAMILY_PATTERNS))),
             *(f"script_{i}" for i in range(len(SCRIPT_PATTERNS))),
             *(f"formula_{i}_script_{j}" for i in range(len(FAMILY_PATTERNS))
               for j in range(len(SCRIPT_PATTERNS))),
         ]}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    traces = out / "native_receipts.jsonl"
    with traces.open("w", encoding="utf-8") as f:
        for split in ("train", "calibration"):
            coll = train if split == "train" else calibration
            for root, records in coll:
                f.write(json.dumps({
                    "root_id": root.root_id, "split": split, "public_task": root.public_task,
                    "tactics": root.tactics, "full_menu": records,
                }, sort_keys=True) + "\n")
        for row in live_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    by_split = {
        split: {
            arm: _arm_summary([r for r in live_rows if r["split"] == split and r["arm"] == arm])
            for arm in ("learned", "fixed_one", "fixed_full", "random_full")
        } for split in ("ordinary_test", "ood_test")
    }
    negative_train = sum(not any(row["receipt"]["passed"] for row in attempts) for _, attempts in train)
    negative_cal = sum(not any(row["receipt"]["passed"] for row in attempts) for _, attempts in calibration)
    if not (0 < negative_train < len(train) and 0 < negative_cal < len(calibration)):
        raise ValueError("missing mixed positive/exhausted native source labels")
    report = {
        "schema": "datagoblin.rlt_native_learned_stop.v1",
        "claim": "TRAIN-only finite-menu logistic tactic ranker with CAL-frozen continuation threshold",
        "evidence": "genuine_native_lean_kernel_compilation_and_independent_replay",
        "lean_version": version, "lean_binary_sha256": binary_sha,
        "driver_sha256": driver_sha, "source_revision_sha256": source_revision,
        "verifier_sha256": verifier_sha, "source_seed": seed,
        "source_roots": len(root_specs), "per_split": per_split,
        "ood_alpha_renamed": alpha_ood,
        "budget_per_root": PER_ROOT_BUDGET,
        "train": {"roots": len(train), "menu_exhausted": negative_train,
                  "kernel_witnessed": len(train) - negative_train},
        "calibration": {"roots": len(calibration), "menu_exhausted": negative_cal,
                        "kernel_witnessed": len(calibration) - negative_cal,
                        "frozen_threshold": threshold, "frontier": cal_frontier},
        "development_splits": by_split,
        "independent_learned_attempt_replays": replays,
        "native_compiler_calls": total_compilations,
        "compiler_seconds_including_replays": round(total_compiler_seconds, 5),
        "certified_proof_sources": source_hashes,
        "model_sha256": _sha(weights_file.read_bytes()),
        "elapsed_wall_s": round(time.perf_counter() - start, 4),
        "limitations": [
            "generated_small_Lean_Init_theorems_with_known_tactic_vocabulary",
            "multiple_commands_per_tactic_script_not_open_ended_tactic_synthesis",
            "menu_exhausted_does_not_mean_mathematically_unprovable",
            "train_and_CAL_exhaustive_source_compilation_only",
            "OOD_changes_numerical_magnitude_not_new_math_library_or_theorem_family",
            "policy_learns_pattern_matches_from_public_formula_and_script_tokens",
            "ordinary_and_OOD_are_development_not_sealed_confirmation",
        ],
    }
    report["files_sha256"] = {
        name: _sha((out / name).read_bytes())
        for name in ("fitted_weights.json", "native_receipts.jsonl")
    }
    (out / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-split", type=int, default=8)
    parser.add_argument("--source-seed", type=int, default=SOURCE_SEED)
    parser.add_argument("--alpha-ood", action="store_true")
    args = parser.parse_args()
    report = run(args.lean.resolve(), args.out, per_split=args.per_split,
                 seed=args.source_seed, alpha_ood=args.alpha_ood)
    print(json.dumps({
        "split_results": report["development_splits"],
        "train": report["train"],
        "cal": report["calibration"]["frozen_threshold"],
        "compiler_calls": report["native_compiler_calls"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
