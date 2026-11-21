"""Fresh-source Lean 4 compilation and kernel-backed post-failure replay campaign.

Every candidate compiles an actual standalone Lean theorem. Successful
candidates receive a SECOND kernel/source replay and explicit axiom audit.
Simple generated theorems serve as infrastructure positive controls, not
as evidence of trained theorem-proving intelligence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .external_checked_exit import SCHEMA, digest, replay_corpus
from .external_native_adapters import LeanKernelChecker

SPLITS = ("train", "calibration", "ordinary_test", "ood_test")
SOURCE_SEED = 20261122
MODEL_SHA = hashlib.sha256(b"fixed_contrived_lean_tactic_menu_v1").hexdigest()
AXIOM_MARK = re.compile(r"(?m)^'?(?:_root_\.)?(dgCheck\d+)'?\s+(does not depend on any axioms|depends on axioms:\s*\[(.*?)\])")
LIMIT_SECS = 15


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task(index: int, group: str) -> tuple[dict[str, Any], list[str]]:
    k = 107 + 31 * index
    template = index % 4
    if template == 0:
        theorem = f"{k} + 0 = {k}"
        tactics = ["exact True.intro", "rfl", "decide"]
    elif template == 1:
        theorem = f"{k} = {k}"
        tactics = ["rfl", "exact True.intro", "decide"]
    elif template == 2:
        theorem = f"∀ n : Nat, n + {k} = {k} + n"
        tactics = ["exact True.intro", f"intro n\nexact Nat.add_comm n {k}", "rfl"]
    else:
        theorem = f"({k} = {k}) ∧ True"
        tactics = ["rfl", "exact And.intro (by rfl) True.intro", "decide"]
    return {
        "statement": theorem, "family": "lean_kernel_arithmetic",
        "proof_case": ("zero_add", "self_eq", "nat_comm", "and_true")[template],
        "instance": f"{group}:{index}:{k}", "import": "Init",
        "tactic_menu_digest": digest(tactics),
    }, tactics


class LeanNativeSession:
    """Compile a registered source goal from scratch and certify its Lean axioms."""

    def __init__(self, executable: Path, directory: Path, statements: dict[int, str]):
        self.executable = executable
        self.directory = directory
        self.statements = statements
        self.proof_by_sid: dict[int, tuple[Path, str]] = {}
        self.calls = 0
        self.kernel_replays = 0
        self.seconds = 0.0

    def _compile(self, source: Path) -> subprocess.CompletedProcess[str]:
        start = time.perf_counter()
        try:
            run = subprocess.run(
                [str(self.executable), str(source)],
                cwd=self.directory, capture_output=True, encoding="utf-8",
                timeout=LIMIT_SECS, check=False,
            )
        except subprocess.TimeoutExpired:
            run = subprocess.CompletedProcess([str(self.executable), str(source)], 124, "", "lean_timeout")
        self.seconds += time.perf_counter() - start
        return run

    def run(self, sid: int, tactic: str) -> dict[str, Any]:
        if sid not in self.statements or not isinstance(tactic, str):
            raise ValueError("unknown source goal or tactic")
        self.calls += 1
        # Stable certificate identity makes the second independent Lean
        # compilation produce exactly the same verifier receipt as the first.
        candidate_sid = int.from_bytes(
            hashlib.sha256(f"{sid}:{tactic}".encode()).digest()[:8], "big",
        )
        name = f"dgCheck{candidate_sid}"
        file = self.directory / f"{name}.lean"
        file.write_text(
            "import Init\n\n"
            + f"theorem {name} : {self.statements[sid]} := by\n"
            + "\n".join("  " + part for part in tactic.split("\n"))
            + f"\n\n#print axioms {name}\n",
            encoding="utf-8",
        )
        result = self._compile(file)
        output = result.stdout + "\n" + result.stderr
        if result.returncode == 0:
            self.proof_by_sid[candidate_sid] = (file, name)
        return {
            "ok": result.returncode == 0,
            "state": {
                "sid": candidate_sid,
                "closed": result.returncode == 0,
                "goals": [] if result.returncode == 0 else ["elaboration_failed"],
            },
            "error": "" if result.returncode == 0 else output[-240:],
        }

    def kernel(self, sid: int) -> dict[str, Any]:
        if sid not in self.proof_by_sid:
            raise ValueError("kernel requested for uncompiled proof")
        self.kernel_replays += 1
        file, expected_name = self.proof_by_sid[sid]
        result = self._compile(file)
        output = result.stdout + "\n" + result.stderr
        matches = list(AXIOM_MARK.finditer(output))
        found = [m for m in matches if m.group(1) == expected_name]
        if result.returncode != 0 or len(found) != 1:
            return {"ok": False, "axioms": [], "error": "missing_exact_lean_axiom_audit"}
        match = found[0]
        if "does not depend" in match.group(2):
            axioms: list[str] = []
        else:
            axioms = [
                a.strip() for a in (match.group(3) or "").split(",") if a.strip()
            ]
        return {
            "ok": True,
            "axioms": axioms,
            "source_sha256": _sha(file),
            "lean_output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        }


def _generate(per_split: int) -> list[tuple[dict[str, Any], list[str], str]]:
    if type(per_split) is not int or not 1 <= per_split <= 32:
        raise ValueError("per_split must be in 1..32")
    out = []
    for split_index, split in enumerate(SPLITS):
        for index in range(per_split):
            task_index = SOURCE_SEED + split_index * 1000 + index
            task, menu = _task(task_index, split)
            out.append((task, menu, split))
    return out


def run(executable: Path, out: Path, *, per_split: int = 8) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("create-only Lean evidence directory required")
    if not executable.is_file():
        raise ValueError("live lean binary required")
    version = subprocess.run([str(executable), "--version"], check=True, capture_output=True,
                             encoding="utf-8", timeout=15).stdout.strip()
    lean_sha = _sha(executable)
    source_sha = _sha(Path(__file__))
    registry_sha = digest({
        "lean_executable_sha256": lean_sha, "driver_sha256": source_sha,
        "seed": SOURCE_SEED, "toolchain_version": version,
    })
    verifier_sha = digest({"lean_binary_sha256": lean_sha, "driver_sha256": source_sha})
    tasks = _generate(per_split)
    roots = []
    source_registry: dict[str, dict[str, Any]] = {}
    sid_to_statement = {}
    for count, (task, tactics, split) in enumerate(tasks):
        source_sid = count + 1
        sid_to_statement[source_sid] = task["statement"]
        source_registry[digest(task)] = {
            "root_sid": source_sid,
            "menu": dict(enumerate(tactics)),
        }
        roots.append({
            "schema_version": SCHEMA, "kind": "lean", "split": split,
            "root_id": f"lean:{SOURCE_SEED}:{split}:{count}",
            "source_revision": registry_sha,
            "model_sha256": MODEL_SHA,
            "verifier_sha256": verifier_sha,
            "source_exposure": "fresh",
            "evidence_kind": "native_operator_verified",
            "public_task": task, "task_digest": digest(task),
            "verified_stop": False, "planned_visits": len(tactics),
            "attempts": [],
        })
    out.mkdir(parents=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dg-lean-live-") as tmp:
        session = LeanNativeSession(executable, Path(tmp), sid_to_statement)
        checker = LeanKernelChecker(lean=session, states=source_registry, verifier_sha=verifier_sha)
        for row, (_, tactics, _) in zip(roots, tasks, strict=True):
            for i in range(len(tactics)):
                candidate = {"id": i}
                receipt = checker(row["public_task"], candidate)
                row["attempts"].append({
                    "visit": i, "candidate": candidate, "verification": receipt,
                })
        first_calls, first_kernel = session.calls, session.kernel_replays
        approved = frozenset(row["root_id"] for row in roots if row["split"] == "train")
        verified = replay_corpus(
            roots, checkers={"lean": checker},
            trusted_shas={"lean": verifier_sha},
            trusted_source_revisions={"lean": registry_sha},
            approved_train_roots=approved,
        )
        if session.calls != first_calls * 2 or session.kernel_replays < first_kernel:
            raise ValueError("independent compilation replay was omitted")
        tool_calls = session.calls
        kernel_calls = session.kernel_replays
        tool_seconds = session.seconds
        certified = {
            step["verification"]["evidence"]["kernel_session_sid"]
            for ep in roots for step in ep["attempts"]
            if step["verification"]["passed"]
        }
        proof_dir = out / "certified_proofs"
        proof_dir.mkdir()
        proof_sources = {}
        for sid in sorted(certified):
            source, _name = session.proof_by_sid[sid]
            target = proof_dir / source.name
            shutil.copyfile(source, target)
            proof_sources[target.relative_to(out).as_posix()] = _sha(target)
    # A dead proof family is a broken harness positive control.
    by_case: dict[str, dict[str, int]] = {}
    for ep in roots:
        key = ep["public_task"]["proof_case"]
        cell = by_case.setdefault(key, {"roots": 0, "kernel_certified_roots": 0})
        cell["roots"] += 1
        cell["kernel_certified_roots"] += int(any(
            step["verification"]["passed"] for step in ep["attempts"]
        ))
    if any(cell["kernel_certified_roots"] == 0 for cell in by_case.values()):
        raise ValueError(f"Lean native positive-control proof case failed: {by_case}")
    results = []
    for split in SPLITS:
        subset = [x for x in (*verified["train"], *verified["evaluation_only"]) if x["split"] == split]
        results.append({
            "split": split, "roots": len(subset),
            "first_valid": sum(row["first_passed"] for row in subset),
            "future_rescued": sum(row["future_rescue"] for row in subset),
            "ever_verified": sum(row["first_verified_visit"] is not None for row in subset),
            "full_menu_checks": len(subset) * 3,
            "first_valid_checks": sum(row["cost_until_first_verified_or_budget"] for row in subset),
        })
    (out / "source_episodes.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in roots), encoding="utf-8",
    )
    (out / "verified_rows.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n"
                for x in (*verified["train"], *verified["evaluation_only"])), encoding="utf-8",
    )
    report = {
        "schema_version": "rlt_gym.live_lean_kernel.v1",
        "evidence_class": "native_lean4_kernel_and_full_replay_positive_control",
        "lean_version": version, "lean_binary_sha256": lean_sha,
        "verifier_sha256": verifier_sha, "source_revision_sha256": registry_sha,
        "driver_sha256": source_sha, "model_sha256": MODEL_SHA,
        "fresh_task_roots": len(roots), "source_splits": results,
        "positive_control_cases": by_case,
        "training_eligible_rows": len(verified["train"]),
        "evaluation_only_rows": len(verified["evaluation_only"]),
        "independently_rechecked_candidate_attempts": verified["verified_candidate_checks"],
        "actual_lean_compilations": tool_calls + kernel_calls,
        "model_tactic_policy": "fixed synthetic menu, no learned proof selection",
        "kernel_replays": kernel_calls, "total_compiler_seconds": tool_seconds,
        "certified_proofs_retained": len(proof_sources),
        "certified_proof_source_sha256": proof_sources,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "file_sha256": {
            name: _sha(out / name) for name in ("source_episodes.jsonl", "verified_rows.jsonl")
        },
        "claim_boundary": (
            "Genuine Lean 4 theorem elaboration and repeated kernel axiom audits on "
            "simple newly generated statements. Fixed teacher-free tactic menus, "
            "not a trained Lean model or a historical benchmark comparison."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=8)
    args = parser.parse_args()
    report = run(args.lean.resolve(), args.out, per_split=args.per_split)
    print(json.dumps({
        "lean_version": report["lean_version"],
        "splits": report["source_splits"],
        "compilations": report["actual_lean_compilations"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
