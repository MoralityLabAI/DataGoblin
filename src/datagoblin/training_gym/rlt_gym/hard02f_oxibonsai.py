"""RLT-HARD-02F: OxiBonsai Q1_0 CPU tactic proposals with native Lean receipts.

Uses the adapter originally implemented in DataGoblin PR #3. Requires a
checksum-pinned local GGUF and OxiBonsai executable. The optional HTTP path
expects an operator-started 127.0.0.1 server and records that limitation.
No fixture response is scored as real model inference.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from datagoblin.hashing import sha256_file
from datagoblin.training_gym.rlt_gym.external_checked_exit import digest
from datagoblin.training_gym.rlt_gym.external_native_adapters import LeanKernelChecker
from datagoblin.training_gym.rlt_gym.hard02_mathlib import (
    AXIOMS,
    MATHLIB_COMMIT,
    MathlibSession,
    export_sources,
    sha,
    verify_upstream,
)
from datagoblin.training_gym.rlt_gym.hard02e_list_stop import (
    BUDGET,
    ListRoot,
    generate,
    rule_candidate,
)
from datagoblin.training_gym.rlt_gym.hard02e_list_stop import SEED as SOURCE_SEED
from datagoblin.training_gym.rlt_gym.lean_bonsai import (
    CHECKPOINT,
    CHECKPOINT_SHA256,
    DirectOxiBonsai,
    OxiBonsai,
)

SCHEMA = "datagoblin.rlt_hard02f_oxibonsai_q1_native.v1"
UPSTREAM_OXI_SHA = "1d10c48e14754ff6999158f8fa13c7c92b40b0dc"
MAX_MODEL_CHOICES = 5
STOP_ID = 1
# Menu is always 1=STOP, 2..6 are the fixed original 0..4 proof candidates.
ARMS = ("oxibonsai_q1", "fixed_full", "random_full", "suffix_length_rule")


class ModelTacticChooser(Protocol):
    def choose(
        self, statement: str, menu: dict[int, str], failed: dict[int, str]
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    checkpoint_sha256: str
    runtime_binary_sha256: str
    tokenizer_sha256: str
    runtime_version: str
    transport: str
    server_model_id: str | None
    server_checkpoint_attestation: str

    def public_receipt(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_sha256": self.checkpoint_sha256,
            "runtime_binary_sha256": self.runtime_binary_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "runtime_version": self.runtime_version,
            "transport": self.transport,
            "server_model_id": self.server_model_id,
            "server_checkpoint_attestation": self.server_checkpoint_attestation,
            "upstream_oxibonsai_sha256_commit": UPSTREAM_OXI_SHA,
        }


def preflight(
    *, binary: Path, model: Path, tokenizer: Path,
    transport: str, port: int, timeout: float,
) -> tuple[ModelTacticChooser, ModelIdentity]:
    if transport not in ("stdio", "http"):
        raise ValueError("OxiBonsai transport must be stdio or local HTTP")
    if not 1 <= port <= 65535 or not 1 <= timeout <= 600:
        raise ValueError("invalid loopback port or model inference timeout")
    binary = binary.resolve(strict=True)
    model = model.resolve(strict=True)
    tokenizer = tokenizer.resolve(strict=True)
    if not all(p.is_file() for p in (binary, model, tokenizer)):
        raise ValueError("OxiBonsai binary/GGUF/tokenizer must be actual files")
    model_sha = sha256_file(model)
    if model_sha != CHECKPOINT_SHA256:
        raise ValueError("published PrismML Bonsai-8B Q1_0 checkpoint SHA-256 mismatch")
    binary_sha = sha256_file(binary)
    tokenizer_sha = sha256_file(tokenizer)
    version = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True,
        timeout=10, check=False,
    )
    if version.returncode or not version.stdout.strip().startswith(("oxibonsai ", "oxibonsai-cli ")):
        raise ValueError("unrecognized OxiBonsai CLI binary")
    version_text = version.stdout.strip()
    if transport == "stdio":
        engine: ModelTacticChooser = DirectOxiBonsai(binary, model, tokenizer, timeout)
        observed = None
        attestation = "process_arguments_bind_checkpoint_exactly"
    else:
        engine = OxiBonsai(port=port, timeout=timeout)
        observed = engine.model_id()
        if not observed:
            raise ValueError("local server did not advertise its model")
        # A bare server /v1/models response proves only its self-reported ID.
        # The operator's launch command must bind the SHA-pinned GGUF.
        attestation = "self_report_only_requires_operator_pinned_startup_command"
    receipt = ModelIdentity(
        CHECKPOINT, model_sha, binary_sha, tokenizer_sha,
        version_text, transport, observed, attestation,
    )
    return engine, receipt


def visible_menu(root: ListRoot, failed: dict[int, str]) -> dict[int, str]:
    remaining = {
        candidate + 2: script for candidate, script in enumerate(root.menu)
        if candidate + 2 not in failed
    }
    return {STOP_ID: "STOP (stop proposing Lean proof tactics)", **remaining}


def _check(checker: LeanKernelChecker, root: ListRoot, cid: int) -> dict[str, Any]:
    start = time.perf_counter()
    evidence = checker(root.public_task(), {"id": cid})
    if evidence.get("task_digest") != digest(root.public_task()):
        raise ValueError("Lean kernel receipt does not match the registered theorem")
    if evidence.get("candidate_digest") != digest({"id": cid}):
        raise ValueError("Lean kernel receipt does not match the selected candidate")
    if evidence["passed"] and not (
        evidence.get("code") == "kernel_certified"
        and evidence["evidence"].get("kernel_invoked") is True
        and evidence["evidence"].get("kernel_ok") is True
        and isinstance(evidence["evidence"].get("axioms"), list)
        and set(evidence["evidence"]["axioms"]) <= AXIOMS
    ):
        raise ValueError("accepted tactic lacks independent Lean kernel/axiom certificate")
    return {
        "candidate_id": cid,
        "receipt": evidence,
        "wall_seconds": round(time.perf_counter() - start, 6),
    }


def bonsai_episode(
    root: ListRoot, model: ModelTacticChooser,
    checker: LeanKernelChecker, *, max_attempts: int = MAX_MODEL_CHOICES,
) -> dict[str, Any]:
    if not 1 <= max_attempts <= BUDGET:
        raise ValueError("Bonsai must use a bounded registered candidate budget")
    failed: dict[int, str] = {}
    proposals = []
    checks = []
    result = "model_budget_exhausted"
    wall_total = 0.0
    invalid = 0
    for _ in range(max_attempts):
        offered = visible_menu(root, failed)
        response = model.choose(root.statement, offered, failed)
        if not isinstance(response, dict):
            raise ValueError("OxiBonsai response must be a structured mapping")
        choice = response.get("choice")
        if choice is not None and (type(choice) is not int or choice not in offered):
            raise ValueError("model proposed an out-of-menu proof action")
        wall = response.get("wall_s", 0.0)
        if not isinstance(wall, (float, int)) or not 0 <= wall <= 900:
            raise ValueError("unbounded model response wall time")
        wall_total += float(wall)
        row = {
            "choice": choice, "usage": response.get("usage"),
            "request_sha256": response.get("request_sha256"),
            "response_sha256": response.get("response_sha256"),
            "finish_reason": response.get("finish_reason"),
            "model_wall_seconds": round(float(wall), 6),
            "offered_menu_sha256": digest(offered),
        }
        if choice is None:
            invalid += 1
            result = "invalid_model_output"
            row["status"] = result
            proposals.append(row)
            break
        if choice == STOP_ID:
            result = "model_stop"
            row["status"] = result
            proposals.append(row)
            break
        candidate = choice - 2
        verified = _check(checker, root, candidate)
        checks.append(verified)
        row.update(status="kernel_certified" if verified["receipt"]["passed"] else "rejected",
                   candidate_id=candidate)
        proposals.append(row)
        if verified["receipt"]["passed"]:
            result = "kernel_certified"
            break
        failed[choice] = root.menu[candidate]
    return {
        "arm": "oxibonsai_q1", "root_id": root.root_id, "split": root.split,
        "result": result, "certified": result == "kernel_certified",
        "attempts": checks, "model_proposals": proposals,
        "model_calls": len(proposals), "invalid_model_choices": invalid,
        "model_wall_seconds": round(wall_total, 6),
        "native_candidate_wall_seconds": round(sum(a["wall_seconds"] for a in checks), 6),
        "candidate_checks": len(checks),
    }


def control_episode(
    root: ListRoot, checker: LeanKernelChecker, arm: str, *,
    max_attempts: int = MAX_MODEL_CHOICES,
) -> dict[str, Any]:
    if arm not in ARMS[1:] or not 1 <= max_attempts <= BUDGET:
        raise ValueError("unregistered or unbounded native control arm")
    if arm == "fixed_full":
        ordered = list(range(BUDGET))
    elif arm == "random_full":
        ordered = list(range(BUDGET))
        random.Random(int(sha(f"bonsai-control/{root.root_id}".encode())[:16], 16)).shuffle(ordered)
    else:
        match = rule_candidate(root)
        ordered = [match] if match is not None else []
    rows = []
    for candidate in ordered[:max_attempts]:
        check = _check(checker, root, candidate)
        rows.append(check)
        if check["receipt"]["passed"]:
            break
    return {
        "arm": arm, "root_id": root.root_id, "split": root.split,
        "result": "kernel_certified" if rows and rows[-1]["receipt"]["passed"] else (
            "public_symbolic_stop" if arm == "suffix_length_rule" and not ordered
            else "candidate_budget_exhausted"
        ),
        "certified": bool(rows and rows[-1]["receipt"]["passed"]),
        "attempts": rows, "candidate_checks": len(rows),
        "native_candidate_wall_seconds": round(sum(a["wall_seconds"] for a in rows), 6),
        "model_calls": 0, "model_wall_seconds": 0.0,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "roots": len(rows), "kernel_certified": sum(r["certified"] for r in rows),
        "candidate_checks": sum(r["candidate_checks"] for r in rows),
        "native_candidate_wall_seconds": round(sum(r["native_candidate_wall_seconds"] for r in rows), 6),
        "model_calls": sum(r["model_calls"] for r in rows),
        "model_wall_seconds": round(sum(r["model_wall_seconds"] for r in rows), 6),
    }


def run(
    *, lake: Path, mathlib: Path, out: Path, model: Path, binary: Path,
    tokenizer: Path, transport: str = "stdio", port: int = 18783,
    timeout: float = 240, roots: int = 4, seed: int = SOURCE_SEED,
    split: str = "ood_test", max_attempts: int = 4,
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("Bonsai native evaluation must write create-only output")
    if split not in ("ordinary_test", "ood_test"):
        raise ValueError("only development splits may use live Bonsai inference")
    if type(roots) is not int or not 1 <= roots <= 8:
        raise ValueError("source root subset must contain 1..8 development roots")
    if not 1 <= max_attempts <= BUDGET:
        raise ValueError("inference and native-check budget must be 1..5")
    checkout = verify_upstream(mathlib.resolve(), lake.resolve())
    engine, identity = preflight(
        binary=binary, model=model, tokenizer=tokenizer,
        transport=transport, port=port, timeout=timeout,
    )
    whole = generate(per_split=8, seed=seed)
    cases = [r for r in whole if r.split == split][:roots]
    if len({case.root_id for case in cases}) != roots:
        raise ValueError("missing fresh source identities")
    out.mkdir(parents=True)
    source_hashes = export_sources(out, cases)
    driver_sha = sha(Path(__file__).read_bytes())
    upstream_sha = digest({
        "mathlib": checkout, "driver_sha256": driver_sha,
        "root_sha256": source_hashes,
    })
    scores = []
    verifier_sha = digest({
        "driver": driver_sha, "source_sha256": source_hashes, "mathlib": checkout,
    })
    certified_source_sha256 = {}
    all_qa = []
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dg-bonsai-hard02f-") as tmp:
        session = MathlibSession(lake.resolve(), mathlib.resolve(), Path(tmp), cases)
        states = {
            digest(case.public_task()): {
                "root_sid": i + 1,
                "menu": dict(enumerate(case.menu)),
            } for i, case in enumerate(cases)
        }
        checker = LeanKernelChecker(lean=session, states=states, verifier_sha=verifier_sha)
        for sid, case in enumerate(cases, 1):
            probe = session.probe(sid)
            if "⊢" not in probe["goal_pp"]:
                raise ValueError("missing real native Lean goal")
            # Verify all five candidate labels, irrespective of actual model
            # choices, and retain these labels in a separate evaluator ledger.
            qa = [_check(checker, case, cid) for cid in range(BUDGET)]
            valid = sum(q["receipt"]["passed"] for q in qa)
            if valid != int(case.expected_witness):
                raise ValueError("registered kernel-negative List QA contract failed")
            all_qa.append({
                "root_id": case.root_id, "split": split,
                "public_task": case.public_task(),
                "menu": case.menu, "goal_state": probe,
                "attempts": qa, "registered_expected_witness": case.expected_witness,
            })
            for arm in ARMS:
                row = (
                    bonsai_episode(case, engine, checker, max_attempts=max_attempts)
                    if arm == "oxibonsai_q1"
                    else control_episode(case, checker, arm, max_attempts=max_attempts)
                )
                reference = {v["candidate_id"]: v["receipt"] for v in qa}
                if any(v["receipt"] != reference[v["candidate_id"]] for v in row["attempts"]):
                    raise ValueError("online policy native receipt disagrees with source-only QA")
                scores.append(row)
        # Independent native replay of every selected Bonsai action.
        bonsai_replays = 0
        case_lookup = {case.root_id: case for case in cases}
        for row in scores:
            if row["arm"] != "oxibonsai_q1":
                continue
            case = case_lookup[row["root_id"]]
            for original in row["attempts"]:
                if _check(checker, case, original["candidate_id"])["receipt"] != original["receipt"]:
                    raise ValueError("independent Bonsai-selected tactic replay mismatch")
                bonsai_replays += 1
        proof_dir = out / "certified_proofs"
        proof_dir.mkdir()
        for _, (file, _) in sorted(session.proof_by_sid.items()):
            dest = proof_dir / file.name
            shutil.copyfile(file, dest)
            certified_source_sha256[dest.relative_to(out).as_posix()] = sha(dest.read_bytes())
        compiler_calls = session.calls + session.kernel_replays + roots
        compiler_seconds = session.seconds
    path_scores = out / "native_policy_receipts.jsonl"
    path_scores.write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in scores), encoding="utf-8"
    )
    path_qa = out / "evaluator_full_menu.jsonl"
    path_qa.write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in all_qa), encoding="utf-8"
    )
    binding = out / "model_binding.json"
    binding.write_text(json.dumps(identity.public_receipt(), sort_keys=True, indent=2) + "\n")
    scored = {arm: _summary([s for s in scores if s["arm"] == arm]) for arm in ARMS}
    fixed = {s["root_id"]: s for s in scores if s["arm"] == "fixed_full"}
    false_stops = sum(
        bool(fixed[s["root_id"]]["certified"] and not s["certified"])
        for s in scores if s["arm"] == "oxibonsai_q1"
    )
    report = {
        "schema": SCHEMA, "evidence_class": "native_Lean_kernel_with_actual_oxibonsai_model_inference",
        "upstream_mathlib_commit": MATHLIB_COMMIT, "source_seed": seed,
        "oxibonsai_repo_commit": UPSTREAM_OXI_SHA,
        "source_split": split, "source_roots": roots, "per_root_attempt_budget": max_attempts,
        "source_revision_sha256": upstream_sha,
        "model_identity": identity.public_receipt(),
        "arms": scored, "false_model_stops_against_fixed": false_stops,
        "actual_model_invocations": scored["oxibonsai_q1"]["model_calls"],
        "independent_bonsai_action_replays": bonsai_replays,
        "native_qa_checks": roots * BUDGET,
        "native_compiler_calls_including_probes": compiler_calls,
        "compiler_seconds": round(compiler_seconds, 6),
        "elapsed_wall_seconds": round(time.perf_counter() - start, 6),
        "native_source_sha256": source_hashes,
        "certified_source_sha256": certified_source_sha256,
        "artifact_sha256": {
            name: sha((out / name).read_bytes())
            for name in ("model_binding.json", "native_policy_receipts.jsonl", "evaluator_full_menu.jsonl")
        },
        "limitations": [
            "development_root_subset_from_hard02e_fixed_List_family",
            "closed_five_candidate_tactic_menus_and_public_target_statements",
            "checkpoint_is_general_Bonsai_8B_not_Lean_specialist_fine_tune",
            "server_transport_model_file_only_operator_attested_when_HTTP",
            "source_name_MCP_blind_retrieval_and_hard_LeanGym_not_tested",
            "model_decision_time_and_native_kernel_time_tracked_separately",
            "no_sealed_confirmation_no_train_sft_from_heldout_sources",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", type=Path, required=True)
    parser.add_argument("--mathlib", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    parser.add_argument("--port", type=int, default=18783)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--roots", type=int, default=4)
    parser.add_argument("--source-seed", type=int, default=SOURCE_SEED)
    parser.add_argument("--split", choices=("ordinary_test", "ood_test"), default="ood_test")
    parser.add_argument("--attempts", type=int, default=4)
    args = parser.parse_args()
    result = run(
        lake=args.lake, mathlib=args.mathlib, out=args.out,
        model=args.checkpoint, binary=args.binary, tokenizer=args.tokenizer,
        transport=args.transport, port=args.port, timeout=args.timeout,
        roots=args.roots, seed=args.source_seed, split=args.split,
        max_attempts=args.attempts,
    )
    print(json.dumps({
        "roots": result["source_roots"],
        "arms": result["arms"],
        "false_stops": result["false_model_stops_against_fixed"],
        "model_invocations": result["actual_model_invocations"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
