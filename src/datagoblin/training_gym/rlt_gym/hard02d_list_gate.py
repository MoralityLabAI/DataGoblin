"""Native Mathlib List negative-control gate with exact near-miss proof terms.

This is an engineering control for valid/invalid *registered* tactic menus,
not a hard-theorem benchmark or a trained-model evaluation. Successful
outcomes require a pinned Lean kernel replay, with separate independent
candidate re-execution before the cohort's quality gate is promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from .external_checked_exit import digest
from .external_native_adapters import LeanKernelChecker
from .hard02_mathlib import (
    AXIOMS,
    MATHLIB_COMMIT,
    MathlibSession,
    Root,
    export_sources,
    sha,
    verify_upstream,
)

SEED = 2026112219
SPLIT = "ordinary_test"
BUDGET = 5


def _tactic(n: int) -> str:
    return (
        "simp only [List.length_append, List.length_singleton]\n"
        f"exact congrArg (fun m : Nat => m + {n}) h"
    )


def generate(*, roots: int = 8, seed: int = SEED) -> list[Root]:
    if type(roots) is not int or not 4 <= roots <= 16 or roots % 4:
        raise ValueError("roots must be a multiple of four between 4 and 16")
    if type(seed) is not int or seed < 0:
        raise ValueError("source seed must be nonnegative")
    cases = []
    for i in range(roots):
        n = 611 + i * 83 + seed % 179
        statement = (
            "∀ (xs ys : List Nat), xs.length = ys.length → "
            f"(xs ++ [{n}]).length = (ys ++ [{n}]).length"
        )
        expected_witness = i % 4 != 3
        # All scripts have the same byte length and real, admissible Lean
        # syntax. Only the +1 proof term matches the registered goal type.
        choices = [0, 2, 3, 4, 1] if expected_witness else [0, 2, 3, 4, 5]
        marker = sha(f"strict-list/{seed}/{i}/{statement}".encode())[:20]
        rng = random.Random(int(marker, 16))
        rng.shuffle(choices)
        actions = tuple(_tactic(value) for value in choices)
        if len(actions) != BUDGET or len({len(t.encode()) for t in actions}) != 1:
            raise ValueError("candidate menu identity/length failure")
        cases.append(Root(
            root_id=f"hard02d-list:{marker}",
            split=SPLIT,
            family="strict_list_append_length",
            statement=statement,
            prefix="intro xs ys h",
            menu=actions,
            source_file=f"source_roots/strict_list/MathlibList_{i:02d}_{marker}.lean",
        ))
    if len({r.root_id for r in cases}) != len(cases):
        raise ValueError("duplicate generated source roots")
    return cases


class DiagnosticSession(MathlibSession):
    def __init__(self, lake_bin: Path, mathlib_root: Path, workspace: Path, roots: list[Root]):
        super().__init__(lake_bin, mathlib_root, workspace, roots)
        self.failure_text: dict[tuple[int, str], str] = {}

    def run(self, sid: int, tactic: str) -> dict[str, Any]:
        response = super().run(sid, tactic)
        if response["ok"] is False:
            self.failure_text[sid, tactic] = response["error"]
        return response


def _check(
    checker: LeanKernelChecker, root: Root, idx: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    receipt = checker(root.public_task(), {"id": idx})
    if receipt["passed"] and not (
        receipt["code"] == "kernel_certified"
        and receipt["evidence"].get("kernel_invoked") is True
        and receipt["evidence"].get("kernel_ok") is True
        and isinstance(receipt["evidence"].get("axioms"), list)
        and set(receipt["evidence"]["axioms"]) <= AXIOMS
    ):
        raise ValueError("kernel receipt violates the registered axiom policy")
    return {
        "candidate_id": idx, "receipt": receipt,
        "wall_seconds": round(time.perf_counter() - started, 6),
    }


def run(lake: Path, mathlib: Path, out: Path, *, roots: int = 8, seed: int = SEED) -> dict:
    if out.exists():
        raise FileExistsError("native QA output must be create-only")
    environment = verify_upstream(mathlib.resolve(), lake.resolve())
    cases = generate(roots=roots, seed=seed)
    out.mkdir(parents=True)
    source_hashes = export_sources(out, cases)
    driver_sha = sha(Path(__file__).read_bytes())
    verifier_sha = digest({"environment": environment, "source_hashes": source_hashes,
                           "driver_sha256": driver_sha})
    rows = []
    retained = {}
    starts = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dg-hard02d-list-") as tmp:
        session = DiagnosticSession(lake, mathlib, Path(tmp), cases)
        reg = {
            digest(case.public_task()): {
                "root_sid": i + 1,
                "menu": dict(enumerate(case.menu)),
            } for i, case in enumerate(cases)
        }
        checker = LeanKernelChecker(lean=session, states=reg, verifier_sha=verifier_sha)
        for sid, case in enumerate(cases, 1):
            goal = session.probe(sid)
            if "⊢" not in goal["goal_pp"]:
                raise ValueError("missing registered Lean proof-state trace")
            attempts = [_check(checker, case, idx) for idx in range(BUDGET)]
            successes = [a["candidate_id"] for a in attempts if a["receipt"]["passed"]]
            expected = sid % 4 != 0
            rows.append({
                "root_id": case.root_id, "split": SPLIT,
                "public_task": case.public_task(),
                "goal_state": goal, "menu": case.menu,
                "candidate_outcomes": attempts,
                "expected_valid": expected,
                "kernel_valid_candidates": successes,
                "contract_passed": len(successes) == int(expected),
                "native_failures": {
                    str(idx): session.failure_text.get((sid, case.menu[idx]), "")
                    for idx in range(BUDGET) if idx not in successes
                },
            })
        independent_replays = 0
        for sid, (case, row) in enumerate(zip(cases, rows, strict=True), 1):
            for original in row["candidate_outcomes"]:
                updated = _check(checker, case, original["candidate_id"])
                if original["receipt"] != updated["receipt"]:
                    raise ValueError("independent kernel receipt changed on fresh compilation")
                independent_replays += 1
        proof_dir = out / "certified_proofs"
        proof_dir.mkdir()
        for _, (path, _name) in sorted(session.proof_by_sid.items()):
            target = proof_dir / path.name
            shutil.copyfile(path, target)
            retained[target.relative_to(out).as_posix()] = sha(target.read_bytes())
        calls = session.calls + session.kernel_replays
        cpu_seconds = session.seconds
    source_ledger = out / "verified_roots.jsonl"
    source_ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    expected_negatives = [r for r in rows if not r["expected_valid"]]
    report = {
        "schema": "datagoblin.rlt_hard02d_strict_list_qa.v1",
        "evidence_class": "pinned_native_Mathlib_Lean_kernel_negative_menu_control",
        "source_seed": seed,
        "upstream_commit": MATHLIB_COMMIT,
        "environment": environment,
        "roots": len(rows),
        "candidate_budget": BUDGET,
        "expected_positive_roots": len(rows) - len(expected_negatives),
        "actual_kernel_positive_roots": sum(bool(r["kernel_valid_candidates"]) for r in rows),
        "expected_negative_roots": len(expected_negatives),
        "actual_kernel_negative_roots": sum(not r["kernel_valid_candidates"] for r in rows),
        "unexpected_positive_candidates": sum(
            max(0, len(r["kernel_valid_candidates"]) - int(r["expected_valid"]))
            for r in rows
        ),
        "quality_gate_passed": all(r["contract_passed"] for r in rows),
        "source_sha256": source_hashes,
        "certified_source_sha256": retained,
        "independent_candidate_replays": independent_replays,
        "compiler_calls": calls + len(rows),
        "compiler_seconds": round(cpu_seconds, 5),
        "elapsed_wall_s": round(time.perf_counter() - starts, 5),
        "limitations": [
            "newly_generated_List_append_length_equalities",
            "fixed_five_candidate_grammar",
            "no_learned_policy_fitted_or_scored_in_this_QA",
            "registered_menu_exhaustion_not_mathematical_unprovability",
            "development_QA_only_no_sealed_confirmation",
        ],
    }
    report["files_sha256"] = {"verified_roots.jsonl": sha(source_ledger.read_bytes())}
    (out / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", required=True, type=Path)
    parser.add_argument("--mathlib", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--roots", type=int, default=8)
    args = parser.parse_args()
    report = run(args.lake, args.mathlib, args.out, roots=args.roots, seed=args.seed)
    print(json.dumps({
        key: report[key] for key in (
            "roots", "expected_positive_roots", "actual_kernel_positive_roots",
            "expected_negative_roots", "actual_kernel_negative_roots",
            "unexpected_positive_candidates", "quality_gate_passed", "compiler_calls",
        )
    }, sort_keys=True))
    if report["quality_gate_passed"] is not True:
        raise SystemExit("hard02d negative-control QA failed; kernel receipts retained")


if __name__ == "__main__":
    main()
