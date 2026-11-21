"""Real Lean 4 kernel checks on fresh source roots through DataGoblin's adapter.

The experiment uses deterministic menu controls and generated Lean core goals.
A successful tactic is recompiled in a separate Lean process with
#print axioms; neither mock Lean nor historical Lean-CGRT goals are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .external_checked_exit import digest, replay_corpus
from .external_native_adapters import LeanKernelChecker

REQUIRED_VERSION = "4.22.0"
SPLITS = ("train", "calibration", "ordinary_test", "ood_test")
SPLIT_SEEDS = {"train": 202611221, "calibration": 202611222,
               "ordinary_test": 202611223, "ood_test": 202611224}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lean_info(lean_exe: Path) -> dict[str, Any]:
    if not lean_exe.is_file():
        raise ValueError("Lean binary is missing")
    p = subprocess.run([str(lean_exe), "--version"], capture_output=True,
                       text=True, timeout=10, check=True)
    if REQUIRED_VERSION not in p.stdout:
        raise ValueError("Lean version differs from pinned toolchain")
    return {"version": p.stdout.strip(), "binary_sha256": _sha(lean_exe.read_bytes())}


def _world(split: str, index: int) -> dict[str, Any]:
    seed = SPLIT_SEEDS[split] + 101 * index
    rng = random.Random(seed)
    salt = rng.randrange(100, 20000)
    task = {
        "statement": f"∀ (n : Nat), (n + {salt}) + 0 = n + {salt}",
        "family": "lean_nat_add_zero",
        "source_nonce": f"{split}-{index}-{salt}",
    }
    return {
        "root_id": f"lean4-{split}-{index}-{salt}",
        "split": split, "seed": seed, "task": task,
        "menu": {
            1: "intro n\n  assumption",
            2: f"intro n\n  exact Nat.add_zero (n + {salt})",
        },
    }


class LeanCliSession:
    """Real core Lean 4 CLI compiler and separate #print axioms proof auditor."""

    def __init__(self, lean_exe: Path, workspace: Path):
        self.lean_exe = lean_exe.resolve()
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.proofs: dict[int, tuple[Path, str]] = {}
        self.checks: list[dict[str, Any]] = []

    def _compile(self, path: Path, *, with_axioms: bool) -> subprocess.CompletedProcess[str]:
        artifact = path.with_suffix(".olean")
        command = [str(self.lean_exe), "-j", "1", "-o", str(artifact), str(path)]
        started = time.perf_counter()
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    cwd=self.workspace, timeout=20)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Lean compiler exceeded bounded per-tactic timeout") from exc
        self.checks.append({
            "source_sha256": _sha(path.read_bytes()),
            "source_file": path.name,
            "with_axiom_audit": with_axioms,
            "returncode": result.returncode,
            "stdout_sha256": _sha(result.stdout.encode()),
            "stderr_sha256": _sha(result.stderr.encode()),
            "seconds": round(time.perf_counter() - started, 6),
        })
        return result

    def run(self, root_sid: int, tactic: str) -> dict[str, Any]:
        # The numerical SID is resolved through the pinned public-task registry.
        # It is never an arbitrary model-supplied Lean expression.
        if type(root_sid) is not int or root_sid < 0 or root_sid not in self.root_registry:
            raise ValueError("unregistered Lean proof state ID")
        statement = self.root_registry[root_sid]
        if not isinstance(tactic, str):
            raise ValueError("invalid tactic")
        proof_key = _sha(f"{root_sid}:{statement}:{tactic}".encode())
        sid = int(proof_key[:13], 16)
        path = self.workspace / f"dg_{proof_key[:20]}.lean"
        path.write_text(
            "import Init\nset_option autoImplicit false\n\n"
            f"theorem dgProof : {statement} := by\n  {tactic}\n",
            encoding="utf-8",
        )
        result = self._compile(path, with_axioms=False)
        ok = result.returncode == 0 and "error:" not in result.stderr.lower()
        if ok:
            self.proofs[sid] = (path, statement)
        return {
            "ok": ok,
            "state": {"goals": [] if ok else ["not_closed"],
                      "closed": ok, "sid": sid},
            "error": "" if ok else (result.stdout + result.stderr)[-400:],
        }

    def kernel(self, sid: int) -> dict[str, Any]:
        if sid not in self.proofs:
            raise ValueError("unknown proof certificate state")
        original, statement = self.proofs[sid]
        # Independently re-run Lean elaboration/kernel from a separate file and
        # require its own #print axioms result. Same binary, new compiler process.
        audit_path = original.with_name(original.stem + "_kernel.lean")
        audit_path.write_text(original.read_text(encoding="utf-8") +
                              "\n#print axioms dgProof\n", encoding="utf-8")
        result = self._compile(audit_path, with_axioms=True)
        if result.returncode:
            return {"ok": False, "axioms": [], "error": "independent kernel compilation failed"}
        full = result.stdout + "\n" + result.stderr
        no_axioms = bool(re.search(r"^'dgProof' does not depend on any axioms\s*$", full, re.M))
        stated = re.findall(r"^'dgProof' depends on axioms:\s*\[([^\]]*)\]\s*$", full, re.M)
        if not no_axioms and len(stated) != 1:
            return {"ok": False, "axioms": [], "error": "missing explicit #print axioms receipt"}
        axioms = [] if no_axioms else [x.strip() for x in stated[0].split(",") if x.strip()]
        return {"ok": True, "axioms": axioms}

    def bind_registry(self, registry: dict[int, str]) -> None:
        if not registry or any(type(k) is not int for k in registry):
            raise ValueError("invalid Lean source registry")
        self.root_registry = registry


def evaluate(*, out: Path, lean_exe: Path, roots_per_split: int = 3) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("output exists")
    if type(roots_per_split) is not int or not 1 <= roots_per_split <= 8:
        raise ValueError("roots_per_split must be 1..8")
    info = _lean_info(lean_exe)
    worlds = [_world(s, i) for s in SPLITS for i in range(roots_per_split)]
    tasks = {digest(w["task"]): w for w in worlds}
    if len(tasks) != len(worlds):
        raise ValueError("duplicate fresh Lean root")
    source_revision = digest({
        "lean": info, "source_module_sha256": _sha(Path(__file__).read_bytes()),
        "root_tasks": [(w["root_id"], digest(w["task"])) for w in worlds],
    })
    verifier_sha = digest({"lean_binary_sha256": info["binary_sha256"],
                           "source_revision": source_revision,
                           "checker": "LeanKernelChecker+separate_lean_compile_axioms"})
    session = LeanCliSession(lean_exe, out / "compiled")
    registry = {1000 + i: w["task"]["statement"] for i, w in enumerate(worlds)}
    session.bind_registry(registry)
    entries = {
        digest(w["task"]): {"root_sid": 1000 + i, "menu": w["menu"]}
        for i, w in enumerate(worlds)
    }
    checker = LeanKernelChecker(
        lean=session, states=entries, verifier_sha=verifier_sha,
    )
    episodes = []
    for i, w in enumerate(worlds):
        sequence = (2, 1) if i % 4 == 0 else (1, 2)
        episodes.append({
            "schema_version": "rlt_gym.external_checked_episode.v1",
            "kind": "lean", "root_id": w["root_id"],
            "source_exposure": "fresh", "split": w["split"],
            "evidence_kind": "native_operator_verified",
            "source_revision": source_revision,
            "model_sha256": _sha(b"deterministic_lean_tactic_menu_control_v1"),
            "verifier_sha256": verifier_sha,
            "public_task": w["task"], "task_digest": digest(w["task"]),
            "planned_visits": len(sequence), "verified_stop": False,
            "attempts": [
                {"visit": j, "candidate": {"id": choice}}
                for j, choice in enumerate(sequence)
            ],
        })
    for episode in episodes:
        for step in episode["attempts"]:
            step["verification"] = checker(episode["public_task"], step["candidate"])
    approved_train_roots = frozenset(x["root_id"] for x in episodes if x["split"] == "train")
    corpus = replay_corpus(
        episodes,
        checkers={"lean": checker},
        trusted_shas={"lean": verifier_sha},
        trusted_source_revisions={"lean": source_revision},
        approved_train_roots=approved_train_roots,
    )
    total = len(episodes)
    expected_attempts = total * 2
    original_passed = sum(s["verification"]["passed"] for e in episodes for s in e["attempts"])
    rechecked = sum(x["future_rescue"] for x in (*corpus["train"], *corpus["evaluation_only"]))
    if (corpus["verified_candidate_checks"] != expected_attempts
            or len(corpus["train"]) != roots_per_split
            or len(corpus["evaluation_only"]) != roots_per_split * 3
            or original_passed != total):
        raise ValueError("Lean kernel and root-custody positive control failed")
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "rlt_gym.live_lean_kernel.v1",
        "evidence_class": "actual_lean4_cli_kernel_independent_recompile_source_only",
        "toolchain": info,
        "source_revision_sha256": source_revision,
        "checker_sha256": verifier_sha,
        "fresh_source_roots": total,
        "registered_train_roots": roots_per_split,
        "evaluation_only_roots": total - roots_per_split,
        "original_tactic_checks": expected_attempts,
        "independent_replay_tactic_checks": expected_attempts,
        "kernel_success_cases": original_passed,
        "first_valid_roots": sum(x["first_passed"] for x in (*corpus["train"], *corpus["evaluation_only"])),
        "rescued_roots": rechecked,
        "actual_lean_compilations_including_kernel_replays": len(session.checks),
        "limitations": [
            "fresh_synthetic_core_nat_goals",
            "deterministic_tactic_menu_control_not_learned_model",
            "kernel_audit_uses_same_pinned_lean_binary_separate_process",
            "zero_mathlib_or_historical_lean_cgrt_proof_roots",
            "no_sealed_confirmation_sample",
        ],
    }
    for name, obj in (("episodes.json", episodes), ("corpus.json", corpus),
                      ("compiler_receipts.json", session.checks)):
        (out / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    report["files_sha256"] = {
        name: _sha((out / name).read_bytes())
        for name in ("episodes.json", "corpus.json", "compiler_receipts.json")
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lean-bin", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--roots-per-split", type=int, default=3)
    args = p.parse_args()
    report = evaluate(lean_exe=args.lean_bin, out=args.out, roots_per_split=args.roots_per_split)
    print(json.dumps({k: report[k] for k in (
        "fresh_source_roots", "kernel_success_cases", "rescued_roots",
        "actual_lean_compilations_including_kernel_replays",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
