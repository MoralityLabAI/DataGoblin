"""Collect bounded Goose/llama.cpp tactic rollouts against native Lean 4.12."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_file

from .external_checked_exit import digest
from .live_lean_kernel import LeanNativeSession

SCHEMA = "datagoblin.lean_goose.v1"
AXIOMS = frozenset({"propext", "Quot.sound", "Classical.choice"})
MAX_TACTIC_TOKENS = 64
TACTIC = re.compile(
    r"(?:intro(?: [a-zA-Z][a-zA-Z0-9_']*)+|constructor|assumption|rfl|"
    r"(?:exact|apply|rw|simp only) [a-zA-Z0-9_'.()\[\], ← →]+)"
)
FORBIDDEN_TERM = re.compile(r"\b(?:by|sorry|sorryAx|admit|unsafe|run_tac|eval|native_decide)\b")


@dataclass(frozen=True)
class Root:
    lineage: str
    split: str
    statement: str


ROOTS = (
    Root("modus-ponens-transport", "train", "∀ p q : Prop, p → (p → q) → q"),
    Root("conjunction-swap", "train", "∀ p q : Prop, p ∧ q → q ∧ p"),
    Root("unary-equality-reversal", "train", "∀ a b : Nat, a = b → b = a"),
    Root("disjunction-injection", "development", "∀ p q r : Prop, p → Or (Or q p) r"),
    Root("congruence-under-composition", "development", "∀ (f g : Nat → Nat) (x y : Nat), x = y → f (g x) = f (g y)"),
    Root(
        "list-reverse-over-append", "development",
        "∀ xs ys : List Nat, (xs ++ ys).reverse = ys.reverse ++ xs.reverse",
    ),
)


def parse_tactic(answer: str) -> str | None:
    action = answer.strip()
    if action == "STOP":
        return action
    if len(action) <= 280 and TACTIC.fullmatch(action) and not FORBIDDEN_TERM.search(action):
        return action
    return None


def proposed_action(answer: str) -> str | None:
    action = parse_tactic(answer)
    if action is not None:
        return action
    # The model sometimes emits a plain tactic list. One visit can spend only
    # one action. Check every fragment against the same grammar before using
    # the first one and round-trip through Lean for the next goal.
    fragments = [part.strip() for part in re.split(r"[;\n]", answer.strip())]
    if 2 <= len(fragments) <= 6 and all(parse_tactic(part) not in {None, "STOP"} for part in fragments):
        return fragments[0]
    return None


class GooseProcess:
    """One serial GDK worker attached to the existing model server."""

    def __init__(self, executable: Path, port: int, model: str) -> None:
        self.command = [str(executable.resolve(strict=True)), str(port), model]

    def __enter__(self) -> GooseProcess:
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            encoding="utf-8", text=True, bufsize=1,
        )
        try:
            self.identity = self.read()
            if self.identity.get("model", {}).get("id") != self.command[-1]:
                raise ValueError("unexpected Goose provider identity")
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_: object) -> None:
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        finally:
            if self.process.stdout is not None:
                self.process.stdout.close()

    def read(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Goose GDK exited without a receipt")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise ValueError("invalid Goose receipt")
        return response

    def infer(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        request = {"prompt": prompt, "max_tokens": max_tokens}
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response = self.read()
        return {**response, "request_sha256": digest(request), "response_sha256": digest(response)}


def source(statement: str, actions: list[str]) -> str:
    proof = "\n".join(f"  {action}" for action in actions)
    return f"import Init\n\ntheorem dgTrace : {statement} := by\n{proof}\n  all_goals trace_state\n"


def probe(lean: Path, statement: str, actions: list[str], path: Path) -> dict[str, Any]:
    path.write_text(source(statement, actions), encoding="utf-8")
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [str(lean), "--json", str(path)], cwd=path.parent, capture_output=True,
            encoding="utf-8", timeout=15, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "verifier_timeout", "seconds": time.perf_counter() - started}
    receipt = {
        "seconds": time.perf_counter() - started, "probe_source_sha256": sha256_file(path),
        "lean_output_sha256": digest({"stdout": result.stdout, "stderr": result.stderr}),
    }
    try:
        messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    except json.JSONDecodeError:
        return {**receipt, "status": "verifier_error", "detail": result.stdout[-600:]}
    trace_line = max(1, len(actions)) + 4
    errors = [entry["data"] for entry in messages if entry["severity"] == "error"]
    goals = [entry["data"] for entry in messages if entry["severity"] == "information"
             and entry["pos"]["line"] == trace_line and "⊢" in entry["data"]]
    if not errors and result.returncode == 0:
        return {**receipt, "status": "closed", "goals": []}
    if goals and errors and all(error.startswith("unsolved goals\n") for error in errors):
        return {**receipt, "status": "open", "goals": goals}
    detail = next((error for error in errors if not error.startswith("unsolved goals\n")), None)
    return {**receipt, "status": "rejected" if detail else "verifier_error",
            "detail": (detail or result.stderr)[-600:]}


def prompt_for(root: Root, proof: list[str], goals: list[str], feedback: str) -> str:
    goal_text = "\n".join(goals)
    return (
        "Continue the first current goal in this Lean 4 proof. Reply with one Lean tactic on one line. "
        "Use intro, constructor, exact, apply, assumption, rfl, rw, or simp only. "
        "If you give up, reply STOP. No Markdown, semicolons, or `by` blocks.\n\n"
        f"Theorem: {root.statement}\n"
        f"Accepted tactics: {json.dumps(proof, ensure_ascii=False)}\n"
        f"Current goals:\n{goal_text}\n"
        f"Previous rejection: {feedback or 'none'}"
    )


def write_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def episode(
    engine: GooseProcess, lean: Path, root: Root, out: Path, *, attempts: int, source_revision: str,
) -> dict[str, Any]:
    case_dir = out / "native" / root.lineage
    case_dir.mkdir(parents=True)
    root_id = digest(asdict(root))
    start = probe(lean, root.statement, [], case_dir / "Root.lean")
    if start["status"] != "open":
        raise RuntimeError(f"native root failed: {root.lineage}: {start}")
    accepted: list[dict[str, Any]] = []
    proof: list[str] = []
    goals = start["goals"]
    feedback = ""
    status = "budget_exhausted"
    certificate = None
    visits = []
    for visit in range(attempts):
        prompt = prompt_for(root, proof, goals, feedback)
        model = engine.infer(prompt, MAX_TACTIC_TOKENS)
        answer = model.get("answer", "")
        action = proposed_action(answer) if isinstance(answer, str) else None
        row = {
            "schema": SCHEMA, "root_id": root_id, "lineage": root.lineage, "split": root.split,
            "source_revision_sha256": source_revision, "visit": visit,
            "statement": root.statement, "goals_before": goals, "proof_before": list(proof),
            "prompt": prompt, "model": model, "action": action,
        }
        if "error" in model or model.get("prompt_token_parity") is not True:
            status = "provider_error" if "error" in model else "token_count_mismatch"
            row["status"] = status
        elif action is None:
            row["status"] = "invalid_model_output"
            feedback = "Reply with exactly one allowed tactic, without prose or Markdown."
        elif action == "STOP":
            status = row["status"] = "model_stop"
        else:
            native = probe(lean, root.statement, [*proof, action], case_dir / f"Visit{visit}.lean")
            row["native"] = native
            row["status"] = native["status"]
            if native["status"] == "closed":
                session = LeanNativeSession(lean, case_dir, {1: root.statement})
                check = session.run(1, "\n".join([*proof, action]))
                if check["ok"]:
                    sid = check["state"]["sid"]
                    audit = session.kernel(sid)
                    if audit["ok"] and set(audit["axioms"]) <= AXIOMS:
                        candidate_path = session.proof_by_sid[sid][0]
                        dest = out / "certified_proofs" / f"{root.lineage}.lean"
                        dest.parent.mkdir(exist_ok=True)
                        shutil.copyfile(candidate_path, dest)
                        certificate = {**audit, "path": dest.relative_to(out).as_posix(),
                                       "lean_binary_sha256": sha256_file(lean)}
                        row["certificate"] = certificate
                        accepted.append(row)
                row["certification_compiler_calls"] = session.calls + session.kernel_replays
                row["certification_seconds"] = session.seconds
                status = row["status"] = "kernel_certified" if certificate else "certification_failed"
            elif native["status"] == "open" and native["goals"] != goals:
                proof.append(action)
                goals = native["goals"]
                accepted.append(row)
                feedback = ""
            elif native["status"] in {"verifier_timeout", "verifier_error"}:
                status = native["status"]
            else:
                if native["status"] == "open":
                    row["status"] = "no_progress"
                feedback = f"{action}: {native.get('detail', 'the goals did not change')}"
        visits.append(row)
        write_row(out / "transitions.jsonl", row)
        if status != "budget_exhausted":
            break
    if certificate and root.split == "train":
        for row in accepted:
            write_row(out / "sft.jsonl", {
                "schema": SCHEMA, "root_id": root_id, "lineage": root.lineage, "split": root.split,
                "source_revision_sha256": source_revision, "certificate": certificate,
                "model_receipt_sha256": row["model"]["response_sha256"], "visit": row["visit"],
                "messages": [{"role": "user", "content": row["prompt"]},
                             {"role": "assistant", "content": row["action"]}],
            })
    return {
        "root_id": root_id, "lineage": root.lineage, "split": root.split, "status": status,
        "model_requests": len(visits), "verifier_calls": 1 + sum(
            ("native" in row) + row.get("certification_compiler_calls", 0) for row in visits
        ),
        "invalid_outputs": sum(row["status"] == "invalid_model_output" for row in visits),
        "certificate": certificate, "sft_rows": len(accepted) if certificate and root.split == "train" else 0,
    }


def run(
    lean: Path, worker: Path, out: Path, *, port: int, model: str, attempts: int,
    startup_seconds: float | None = None,
) -> dict[str, Any]:
    if not 1 <= attempts <= 8:
        raise ValueError("attempt budget must be 1..8")
    if startup_seconds is not None and not 0 <= startup_seconds <= 3600:
        raise ValueError("invalid operator-measured model startup time")
    lean = lean.resolve(strict=True)
    version = subprocess.run([str(lean), "--version"], capture_output=True, text=True,
                             check=True, timeout=15).stdout.strip()
    if not version.startswith("Lean (version 4.12.0,"):
        raise ValueError("Lean 4.12.0 is required for this registered pilot")
    revision = digest([asdict(root) for root in ROOTS])
    out.mkdir(parents=True, exist_ok=False)
    for filename in ("sft.jsonl", "transitions.jsonl", "episodes.jsonl"):
        (out / filename).touch()
    summary = []
    with GooseProcess(worker, port, model) as engine:
        provenance = {"schema": SCHEMA, "lean": version, "lean_sha256": sha256_file(lean),
                      "worker_sha256": sha256_file(worker), "driver_sha256": sha256_file(Path(__file__)),
                      "model_identity": engine.identity, "source_revision_sha256": revision,
                      "operator_server_startup_seconds": startup_seconds,
                      "transport": "serial_requests_to_already_resident_server", "attempt_budget": attempts}
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        for root in ROOTS:
            row = episode(engine, lean, root, out, attempts=attempts, source_revision=revision)
            summary.append(row)
            write_row(out / "episodes.jsonl", row)
    report = {"schema": SCHEMA, "episodes": summary, "verified": sum(bool(row["certificate"]) for row in summary),
              "sft_rows": sum(row["sft_rows"] for row in summary)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--model", default="inky-mini-varloop")
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--startup-seconds", type=float)
    args = parser.parse_args()
    report = run(args.lean, args.worker, args.out, port=args.port, model=args.model, attempts=args.attempts,
                 startup_seconds=args.startup_seconds)
    print(json.dumps({"kernel_certified": report["verified"], "sft_rows": report["sft_rows"]}))


if __name__ == "__main__":
    main()
