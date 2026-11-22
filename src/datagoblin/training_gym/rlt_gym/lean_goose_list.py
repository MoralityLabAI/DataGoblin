"""Run the registered HARD-02F List menu comparison with the Goose proposer."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_file

from .external_checked_exit import digest
from .external_native_adapters import LeanKernelChecker
from .hard02_mathlib import MathlibSession, export_sources, verify_upstream
from .hard02e_list_stop import BUDGET, SEED, generate
from .hard02f_oxibonsai import ARMS, _check, _summary, bonsai_episode, control_episode
from .lean_bonsai import build_prompt, parse_choice
from .lean_goose import GooseProcess, write_row

SCHEMA = "datagoblin.lean_goose_list.v1"
MODEL_ARM = "inky_mini_goose"


class GooseChoices:
    def __init__(self, engine: GooseProcess, directory: Path, root_id: str) -> None:
        self.engine, self.directory, self.root_id = engine, directory, root_id
        self.errors = 0

    def choose(self, statement: str, menu: dict[int, str], failed: dict[int, str]) -> dict[str, Any]:
        prompt = build_prompt(statement, menu, failed)
        receipt = self.engine.infer(prompt, 16)
        write_row(self.directory / "model_requests.jsonl", {
            "root_id": self.root_id, "prompt": prompt, "receipt": receipt,
        })
        valid_request = receipt.get("prompt_token_parity") is True and "error" not in receipt
        self.errors += not valid_request
        return {
            "choice": parse_choice(receipt.get("answer"), menu) if valid_request else None,
            "usage": receipt.get("provider", {}).get("usage"),
            "wall_s": receipt.get("request_seconds", 0.0),
            "request_sha256": receipt.get("request_sha256"),
            "response_sha256": receipt.get("response_sha256"),
            "response_text": receipt.get("answer"),
        }


def run(
    lake: Path, mathlib: Path, worker: Path, out: Path, *, port: int, model: str,
) -> dict[str, Any]:
    lake, mathlib = lake.resolve(strict=True), mathlib.resolve(strict=True)
    upstream = verify_upstream(mathlib, lake)
    roots = [root for root in generate(per_split=8, seed=SEED) if root.split == "ood_test"][:4]
    out = out.resolve()
    out.mkdir(parents=True)
    source_hashes = export_sources(out, roots)
    native_dir = out / "native"
    native_dir.mkdir()
    proof_dir = out / "certified_proofs"
    proof_dir.mkdir()
    session = MathlibSession(lake, mathlib, native_dir, roots)
    verifier_sha = digest({"upstream": upstream, "sources": source_hashes,
                           "driver_sha256": sha256_file(Path(__file__))})
    states = {digest(root.public_task()): {"root_sid": sid, "menu": dict(enumerate(root.menu))}
              for sid, root in enumerate(roots, 1)}
    checker = LeanKernelChecker(lean=session, states=states, verifier_sha=verifier_sha)
    groups: dict[str, list[dict[str, Any]]] = {arm: [] for arm in (MODEL_ARM, *ARMS[1:])}
    oracles = {}
    native_replays = 0
    with GooseProcess(worker, port, model) as engine:
        for sid, root in enumerate(roots, 1):
            state = session.probe(sid)
            oracle = [_check(checker, root, i) for i in range(BUDGET)]
            labels = {row["candidate_id"]: row["receipt"] for row in oracle}
            oracles[root.root_id] = any(row["passed"] for row in labels.values())
            if sum(row["passed"] for row in labels.values()) != int(root.expected_witness):
                raise ValueError(f"registered List witness disagreement: {root.root_id}")
            for receipt in labels.values():
                if receipt["passed"]:
                    source, _ = session.proof_by_sid[receipt["evidence"]["kernel_session_sid"]]
                    shutil.copyfile(source, proof_dir / source.name)
            write_row(out / "kernel_oracle.jsonl", {"root_id": root.root_id, "task": root.public_task(),
                                                    "menu": root.menu, "state": state, "attempts": oracle})
            chooser = GooseChoices(engine, out, root.root_id)
            proposed = bonsai_episode(root, chooser, checker, max_attempts=2)
            proposed["arm"] = MODEL_ARM
            if chooser.errors:
                proposed["result"] = "provider_error"
                proposed["invalid_model_choices"] -= chooser.errors
            controls = [control_episode(root, checker, arm, max_attempts=2) for arm in ARMS[1:]]
            for row in (proposed, *controls):
                for attempt in row["attempts"]:
                    if attempt["receipt"] != labels[attempt["candidate_id"]]:
                        raise ValueError("live kernel receipt differs from independent oracle")
                groups[row["arm"]].append(row)
                write_row(out / "paired_receipts.jsonl", row)
            for attempt in proposed["attempts"]:
                if _check(checker, root, attempt["candidate_id"])["receipt"] != attempt["receipt"]:
                    raise ValueError("independent selected-candidate replay disagreed")
                native_replays += 1
    inky = groups[MODEL_ARM]
    report = {
        "schema": SCHEMA, "upstream": upstream, "model_identity": engine.identity, "source_seed": SEED,
        "worker_sha256": sha256_file(worker), "verifier_sha256": verifier_sha,
        "evaluation_only": True, "verifier_attempt_budget": 2, "source_roots": len(roots),
        "exhaustive_oracle": {"checks": len(roots) * BUDGET, "certified": sum(oracles.values())},
        "model_false_stops": sum(row["result"] == "model_stop" and oracles[row["root_id"]] for row in inky),
        "model_invalid_choices": sum(row["invalid_model_choices"] for row in inky),
        "model_provider_errors": sum(row["result"] == "provider_error" for row in inky),
        "native_selected_replays": native_replays,
        "arms": {("fixed_order_two_checks" if arm == "fixed_full" else arm): _summary(rows)
                 for arm, rows in groups.items()},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {
        path.relative_to(out).as_posix(): sha256_file(path)
        for path in sorted(out.rglob("*")) if path.is_file()
    }
    (out / "SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lake", "mathlib", "worker", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--model", default="inky-mini-varloop")
    args = parser.parse_args()
    report = run(args.lake, args.mathlib, args.worker, args.out, port=args.port, model=args.model)
    print(json.dumps(report["arms"]))


if __name__ == "__main__":
    main()
