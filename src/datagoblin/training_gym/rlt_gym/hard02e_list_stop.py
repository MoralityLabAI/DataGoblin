"""RLT-HARD-02E: trained stopping on native-verified negative List tactic menus.

Four source-disjoint development splits vary the appended suffix length and
registered candidate arithmetic. Every positive proof and attempted negative
is checked by pinned Mathlib / Lean 4, with independent kernel replay.
This controlled List-family gym is not a hard Mathlib theorem benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .external_checked_exit import digest
from .external_native_adapters import LeanKernelChecker
from .hard02_mathlib import (
    AXIOMS,
    MATHLIB_COMMIT,
    MathlibSession,
    calibrate,
    fit_policy,
    rank_scores,
    sha,
    verify_upstream,
)

SEED = 2026112247
SPLITS = ("train", "calibration", "ordinary_test", "ood_test")
BUDGET = 5
UPSTREAM = "Mathlib/Data/List/Basic.lean"
ARMS = ("content_learned", "legacy_learned", "fixed_full", "random_full",
        "fixed_one", "suffix_length_rule")
MAX_K = 9
TACTIC_K = re.compile(r"congrArg \(fun m : Nat => m \+ ([0-9])\) h$")
SUFFIX = re.compile(r"\((?:xs|ys) \+\+ \[([0-9, ]+)\]\)\.length")


@dataclass(frozen=True)
class ListRoot:
    root_id: str
    split: str
    statement: str
    prefix: str
    menu: tuple[str, ...]
    source_file: str
    suffix_len: int
    expected_witness: bool

    def public_task(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "source_identity": self.root_id,
            "source_file": self.source_file,
            "import": "Mathlib",
            "upstream_module": UPSTREAM,
            "candidate_menu_sha256": digest(self.menu),
        }


def _tactic(k: int) -> str:
    if type(k) is not int or not 0 <= k <= MAX_K:
        raise ValueError("one-digit nonnegative candidate required")
    return (
        "simp only [List.length_append, List.length_cons, List.length_nil]\n"
        f"exact congrArg (fun m : Nat => m + {k}) h"
    )


def generate(*, per_split: int = 8, seed: int = SEED) -> list[ListRoot]:
    if type(per_split) is not int or not 4 <= per_split <= 16 or per_split % 4:
        raise ValueError("per_split must be a multiple of four in 4..16")
    if type(seed) is not int or seed < 0:
        raise ValueError("source seed must be nonnegative")
    cohort = []
    for split_number, split in enumerate(SPLITS):
        for i in range(per_split):
            # OOD has longer suffixes than the TRAIN source distribution.
            k = 6 + (i % 3) if split == "ood_test" else 1 + ((i + 2 * split_number) % 5)
            assert 1 <= k <= 8
            n = 1009 + (seed % 107) * 23 + split_number * 9131 + i * 113
            values = [n + 17 * t + (seed // 97) % 71 for t in range(k)]
            suffix = ", ".join(str(v) for v in values)
            statement = (
                "∀ (xs ys : List Nat), xs.length = ys.length → "
                f"(xs ++ [{suffix}]).length = (ys ++ [{suffix}]).length"
            )
            token = hashlib.sha256(
                f"hard02e/{seed}/{split}/{i}/{statement}".encode()
            ).hexdigest()[:20]
            rng = random.Random(int(token, 16))
            has_witness = i % 4 != 3
            alternatives = [v for v in range(MAX_K + 1) if v != k]
            distractors = rng.sample(alternatives, 4 if has_witness else 5)
            ids = [k, *distractors] if has_witness else distractors
            rng.shuffle(ids)
            menu = tuple(_tactic(value) for value in ids)
            if len(menu) != BUDGET or len(set(menu)) != BUDGET:
                raise ValueError("registered menu has missing/duplicate actions")
            if len({len(script.encode("utf-8")) for script in menu}) != 1:
                raise ValueError("candidate byte length must be exactly matched")
            cohort.append(ListRoot(
                root_id=f"hard02e:{split}:{token}",
                split=split,
                statement=statement,
                prefix="intro xs ys h",
                menu=menu,
                source_file=f"source_roots/{split}/MathlibListE_{i:02d}_{token}.lean",
                suffix_len=k, expected_witness=has_witness,
            ))
    if len({x.root_id for x in cohort}) != len(cohort):
        raise ValueError("duplicated List source identity")
    if len({x.statement for x in cohort}) != len(cohort):
        raise ValueError("same theorem reused across source roots")
    return cohort


def syntax_length(root: ListRoot) -> int:
    """Derive required congrArg offset from the public source statement."""
    matches = SUFFIX.findall(root.statement)
    if len(matches) != 2 or matches[0] != matches[1]:
        raise ValueError("unsupported List source expression for syntax-rule baseline")
    values = [v.strip() for v in matches[0].split(",")]
    if not values or any(not v.isascii() or not v.isdigit() for v in values):
        raise ValueError("source suffix contains a nonliteral")
    return len(values)


def rule_candidate(root: ListRoot) -> int | None:
    """Cheap source-text control, without kernel labels or evaluator witness flag."""
    k = syntax_length(root)
    choices = []
    for i, script in enumerate(root.menu):
        m = TACTIC_K.search(script)
        if m and int(m.group(1)) == k:
            choices.append(i)
    if len(choices) > 1:
        raise ValueError("ambiguous symbolic syntax-rule choice")
    return choices[0] if choices else None


def _check(checker: LeanKernelChecker, root: ListRoot, candidate_id: int) -> dict[str, Any]:
    start = time.perf_counter()
    receipt = checker(root.public_task(), {"id": candidate_id})
    if receipt.get("task_digest") != digest(root.public_task()):
        raise ValueError("verifier checked a different public task")
    if receipt.get("candidate_digest") != digest({"id": candidate_id}):
        raise ValueError("verifier executed an unregistered action")
    if receipt["passed"] and not (
        receipt.get("code") == "kernel_certified"
        and receipt["evidence"].get("kernel_invoked") is True
        and receipt["evidence"].get("kernel_ok") is True
        and isinstance(receipt["evidence"].get("axioms"), list)
        and set(receipt["evidence"]["axioms"]) <= AXIOMS
    ):
        raise ValueError("accepted source proof lacks native kernel/axiom receipt")
    return {
        "candidate_id": candidate_id,
        "receipt": receipt,
        "wall_seconds": round(time.perf_counter() - start, 6),
    }


def _labels_present(root: ListRoot, checks: list[dict[str, Any]]) -> bool:
    good = [c["candidate_id"] for c in checks if c["receipt"]["passed"]]
    return len(good) == int(root.expected_witness)


def _calibrate(
    rows: list[dict[str, Any]], model: np.ndarray, *,
    semantic: bool, words: tuple[str, ...],
) -> tuple[float, list[dict[str, Any]]]:
    if not rows or any(row["root"].split != "calibration" for row in rows):
        raise ValueError("only verified CAL roots may choose a continuation threshold")
    # The existing preregistered objective is success minus 0.03 checks.
    return calibrate(rows, model, words, semantic=semantic)


def _ordered(
    root: ListRoot, arm: str, goal_pp: str,
    content: np.ndarray, legacy: np.ndarray, words: tuple[str, ...],
) -> tuple[list[int], np.ndarray | None]:
    if arm in ("content_learned", "legacy_learned"):
        semantic = arm == "content_learned"
        model = content if semantic else legacy
        ps = rank_scores(goal_pp, root.menu, model, words, semantic=semantic)
        ids = sorted(range(BUDGET), key=lambda i: (-float(ps[i]), i))
        return ids, ps
    if arm in ("fixed_full", "fixed_one"):
        return list(range(BUDGET)), None
    if arm == "random_full":
        ids = list(range(BUDGET))
        random.Random(int(sha(f"hard02e-random/{root.root_id}".encode())[:16], 16)).shuffle(ids)
        return ids, None
    if arm == "suffix_length_rule":
        target = rule_candidate(root)
        return [target] if target is not None else [], None
    raise ValueError("unregistered native stopping policy")


def _online(
    root: ListRoot, goal_pp: str, checker: LeanKernelChecker, arm: str,
    content: np.ndarray, legacy: np.ndarray, words: tuple[str, ...],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    ordering, ps = _ordered(root, arm, goal_pp, content, legacy, words)
    attempts = []
    low_value_stop = False
    for offset, cid in enumerate(ordering):
        if arm == "fixed_one" and attempts:
            break
        if ps is not None and offset:
            threshold = thresholds[arm]
            if max(float(ps[k]) for k in ordering[offset:]) < threshold:
                low_value_stop = True
                break
        row = _check(checker, root, cid)
        attempts.append(row)
        if row["receipt"]["passed"]:
            break
    return {
        "root_id": root.root_id, "split": root.split,
        "arm": arm, "attempts": attempts,
        "certified": bool(attempts and attempts[-1]["receipt"]["passed"]),
        "stopped_before_budget": bool(
            (low_value_stop or (arm == "suffix_length_rule" and not attempts))
            and not (attempts and attempts[-1]["receipt"]["passed"])
        ),
        "candidate_wall_seconds": round(sum(a["wall_seconds"] for a in attempts), 6),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "roots": len(rows),
        "certified": sum(bool(r["certified"]) for r in rows),
        "candidate_attempts": sum(len(r["attempts"]) for r in rows),
        "candidate_wall_seconds": round(sum(r["candidate_wall_seconds"] for r in rows), 6),
        "early_negative_stops": sum(bool(r["stopped_before_budget"]) for r in rows),
    }


def run(
    lake: Path, mathlib: Path, out: Path, *,
    per_split: int = 8, seed: int = SEED,
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("create-only native evidence directory required")
    env = verify_upstream(mathlib.resolve(), lake.resolve())
    roots = generate(per_split=per_split, seed=seed)
    out.mkdir(parents=True)
    root_sha = {}
    for root in roots:
        path = out / root.source_file
        path.parent.mkdir(parents=True, exist_ok=True)
        text = (
            "import Mathlib\n"
            "set_option autoImplicit false\n"
            f"-- registered root: {root.root_id}\n"
            f"theorem registeredGoal : {root.statement} := by\n"
            f"  {root.prefix}\n"
            "  trace_state\n"
        )
        path.write_text(text, encoding="utf-8")
        root_sha[root.source_file] = sha(path.read_bytes())
    from . import hard02_semantic_features
    driver_sha = sha(Path(__file__).read_bytes())
    feature_sha = sha(Path(hard02_semantic_features.__file__).read_bytes())
    verifier_sha = digest({
        "mathlib_commit": MATHLIB_COMMIT, "environment": env,
        "driver_sha256": driver_sha, "features_sha256": feature_sha,
        "source_sha256": root_sha,
    })
    proofs: dict[str, str] = {}
    native_report: list[dict[str, Any]] = []
    fitting: list[dict[str, Any]] = []
    cal: list[dict[str, Any]] = []
    full_quality: dict[str, dict[str, Any]] = {}
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dg-rlt-hard02e-") as work:
        session = MathlibSession(lake, mathlib, Path(work), roots)
        states = {
            digest(root.public_task()): {
                "root_sid": index + 1, "menu": dict(enumerate(root.menu)),
            } for index, root in enumerate(roots)
        }
        checker = LeanKernelChecker(lean=session, states=states, verifier_sha=verifier_sha)
        all_probes = {}
        for idx, root in enumerate(roots, 1):
            all_probes[root.root_id] = session.probe(idx)
        # The evaluator checks every registered candidate, including held-out
        # menus. These outcomes are withheld from the learned fit and selector.
        for split in SPLITS:
            for root in (x for x in roots if x.split == split):
                checks = [_check(checker, root, j) for j in range(BUDGET)]
                row = {
                    "root": root, "split": split,
                    "goal_pp": all_probes[root.root_id]["goal_pp"],
                    "menu": root.menu, "checks": checks,
                }
                full_quality[root.root_id] = row
                if split == "train":
                    fitting.append(row)
                elif split == "calibration":
                    cal.append(row)
        qa_by_split = {
            split: {
                "roots": sum(x.split == split for x in roots),
                "actual_native_positive": sum(
                    any(a["receipt"]["passed"] for a in r["checks"])
                    for r in full_quality.values() if r["split"] == split
                ),
                "actually_exhausted": sum(
                    not any(a["receipt"]["passed"] for a in r["checks"])
                    for r in full_quality.values() if r["split"] == split
                ),
                "registered_menu_contract_passed": sum(
                    _labels_present(r["root"], r["checks"])
                    for r in full_quality.values() if r["split"] == split
                ),
            } for split in SPLITS
        }
        qa_train = qa_by_split["train"]["registered_menu_contract_passed"]
        qa_cal = qa_by_split["calibration"]["registered_menu_contract_passed"]
        # Only native TRAIN transcripts are fed to the learned policies.
        words: tuple[str, ...] = ()
        content = fit_policy(fitting, words, semantic=True)
        legacy = fit_policy(fitting, words, semantic=False)
        content_threshold, content_frontier = _calibrate(
            cal, content, semantic=True, words=words
        )
        legacy_threshold, legacy_frontier = _calibrate(
            cal, legacy, semantic=False, words=words
        )
        thresholds = {
            "content_learned": content_threshold,
            "legacy_learned": legacy_threshold,
        }
        for split in ("ordinary_test", "ood_test"):
            for root in (x for x in roots if x.split == split):
                for arm in ARMS:
                    native_report.append(_online(
                        root, all_probes[root.root_id]["goal_pp"],
                        checker, arm, content, legacy, words, thresholds,
                    ))
        # Native evaluation receipts must agree with the evaluator-owned
        # exhaustive menu receipts, which the trained policy never receives.
        for row in native_report:
            reference = {
                attempt["candidate_id"]: attempt["receipt"]
                for attempt in full_quality[row["root_id"]]["checks"]
            }
            for attempt in row["attempts"]:
                if attempt["receipt"] != reference[attempt["candidate_id"]]:
                    raise ValueError("policy evaluation disagrees with held-out native QA")
        # Replay all taken actions from both learned policies in fresh Lean
        # processes. Compare verifier-owned receipts rather than claimed proof states.
        replay_counts = {arm: 0 for arm in ("content_learned", "legacy_learned")}
        lookup = {root.root_id: root for root in roots}
        for row in native_report:
            if row["arm"] not in replay_counts:
                continue
            root = lookup[row["root_id"]]
            for attempt in row["attempts"]:
                repeated = _check(checker, root, attempt["candidate_id"])
                if repeated["receipt"] != attempt["receipt"]:
                    raise ValueError("native candidate kernel receipt differs on replay")
                replay_counts[row["arm"]] += 1
        directory = out / "certified_proofs"
        directory.mkdir()
        for _, (source, _name) in sorted(session.proof_by_sid.items()):
            target = directory / source.name
            shutil.copyfile(source, target)
            proofs[target.relative_to(out).as_posix()] = sha(target.read_bytes())
        calls = session.calls + session.kernel_replays + len(roots)
        compiler_seconds = session.seconds
    (out / "weights.json").write_text(json.dumps({
        "content_weights": content.tolist(), "legacy_weights": legacy.tolist(),
        "train_only": True, "feature_sha256": feature_sha,
        "calibration_thresholds": thresholds,
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    source_file = out / "sources.jsonl"
    source_file.write_text("".join(json.dumps({
        "root_id": root.root_id, "split": root.split,
        "public_task": root.public_task(),
        "menu": root.menu, "goal_state": all_probes[root.root_id],
        "test_only_expected_witness": root.expected_witness,
        "test_only_suffix_length": root.suffix_len,
    }, sort_keys=True) + "\n" for root in roots), encoding="utf-8")
    quality_file = out / "full_menu_quality.jsonl"
    quality_file.write_text("".join(json.dumps({
        "root_id": row["root"].root_id, "split": row["split"],
        "attempts": row["checks"], "registered_witness": row["root"].expected_witness,
        "native_contract_passed": _labels_present(row["root"], row["checks"]),
    }, sort_keys=True) + "\n" for row in full_quality.values()), encoding="utf-8")
    receipt_file = out / "receipts.jsonl"
    with receipt_file.open("w", encoding="utf-8") as f:
        for row in native_report:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summaries = {
        split: {
            arm: _summary([r for r in native_report if r["split"] == split and r["arm"] == arm])
            for arm in ARMS
        } for split in ("ordinary_test", "ood_test")
    }
    full = {
        r["root_id"]: r for r in native_report if r["arm"] == "fixed_full"
    }
    false_stop = {
        split: {
            arm: sum(
                bool(full[r["root_id"]]["certified"] and not r["certified"])
                for r in native_report if r["split"] == split and r["arm"] == arm
            ) for arm in ("content_learned", "legacy_learned", "suffix_length_rule")
        } for split in ("ordinary_test", "ood_test")
    }
    report = {
        "schema": "datagoblin.rlt_hard02e_verified_list_stop.v1",
        "evidence_class": "native_Lean_Mathlib_compiler_kernel_and_independent_replays",
        "source_seed": seed,
        "upstream": env, "upstream_source_sha256": env["mathlib_source_sha256"][UPSTREAM],
        "source_revision_sha256": digest({"source_sha256": root_sha, "driver_sha256": driver_sha}),
        "model_features_sha256": feature_sha,
        "verifier_sha256": verifier_sha,
        "source_roots": len(roots), "roots_per_split": per_split,
        "train": {"roots": len(fitting), "native_label_contracts_passed": qa_train},
        "source_quality_by_split": qa_by_split,
        "evaluator_only_complete_menu_checks": BUDGET * len(roots),
        "calibration": {
            "roots": len(cal), "native_label_contracts_passed": qa_cal,
            "thresholds": thresholds,
            "content_frontier": content_frontier,
            "legacy_frontier": legacy_frontier,
        },
        "development": summaries, "false_stops_against_full_search": false_stop,
        "independent_replays_by_learned_arm": replay_counts,
        "compiler_calls_including_probes": calls,
        "compiler_seconds": round(compiler_seconds, 6),
        "certified_source_sha256": proofs,
        "source_sha256": root_sha,
        "model_sha256": sha((out / "weights.json").read_bytes()),
        "elapsed_wall_seconds": round(time.perf_counter() - start, 6),
        "quality_gate_passed": all(
            row["registered_menu_contract_passed"] == per_split
            for row in qa_by_split.values()
        ),
        "limitations": [
            "generated_mathlib_List_length_congruences_only",
            "fixed_five_candidate_menu_with_cheap_symbolic_suffix_matching_baseline",
            "source_root_disjoint_but_theorem_family_shared_across_splits",
            "no_live_multistep_tactic_generation_or_recurrent_policy",
            "trained_model_uses_fitted_candidate_scores_not_theorem_search_LM",
            "development_ordinary_and_OOD_only_sealed_confirmation_unopened",
        ],
        "file_sha256": {
            name: sha((out / name).read_bytes())
            for name in (
                "sources.jsonl", "receipts.jsonl", "weights.json", "full_menu_quality.jsonl"
            )
        },
    }
    (out / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", type=Path, required=True)
    parser.add_argument("--mathlib", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=8)
    parser.add_argument("--source-seed", type=int, default=SEED)
    args = parser.parse_args()
    report = run(args.lake, args.mathlib, args.out, per_split=args.per_split, seed=args.source_seed)
    print(json.dumps({
        "train": report["train"],
        "cal": {
            "roots": report["calibration"]["roots"],
            "native_label_contracts_passed": report["calibration"]["native_label_contracts_passed"],
            "thresholds": report["calibration"]["thresholds"],
        },
        "development": report["development"],
        "false_stops": report["false_stops_against_full_search"],
        "compiler_calls": report["compiler_calls_including_probes"],
        "quality_gate_passed": report["quality_gate_passed"],
    }, sort_keys=True))
    if not report["quality_gate_passed"]:
        raise SystemExit("native TRAIN/CAL positive/negative menu quality gate failed")


if __name__ == "__main__":
    main()
