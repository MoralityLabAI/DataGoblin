"""Reusable emergence probe for staged Baby Goblin 20M/50M/100M/200M checkpoints.

Evaluates any Aleph Direct-A native safetensors checkpoint on the frozen 16-case
emergence suite (eval_seed=2026112799), updates baby_goblin_emergence_ledger.json,
and records the first checkpoint crossing JSON syntax, EOS termination, action
selection, and verified research progress thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recompute_emergence_milestones(ledger: dict[str, Any]) -> dict[str, Any]:
    """Scan evaluated stages in ascending position order and record first crossing per gate."""
    gates = ledger["emergence_gates"]
    milestones: dict[str, Any] = {
        "first_json_onset": None,
        "first_json_reliable": None,
        "first_eos_onset": None,
        "first_eos_reliable": None,
        "first_action_selection": None,
        "first_verified_capability": None,
        "first_nonzero_primary_score": None,
    }
    ordered = sorted(ledger["stages"], key=lambda s: int(s["positions"]))
    for stage in ordered:
        if stage.get("status") != "evaluated":
            continue
        for arm_id, metrics in sorted(stage.get("arms", {}).items()):
            tag = {"stage_id": stage["stage_id"], "positions": stage["positions"], "arm_id": arm_id}
            if (
                milestones["first_json_onset"] is None
                and metrics.get("valid_json_rate", 0.0) >= gates["valid_json_rate_onset"]
                and metrics.get("schema_valid_action_rate", 0.0) >= gates["schema_valid_action_rate_onset"]
            ):
                milestones["first_json_onset"] = tag
            if (
                milestones["first_json_reliable"] is None
                and metrics.get("valid_json_rate", 0.0) >= gates["valid_json_rate_reliable"]
            ):
                milestones["first_json_reliable"] = tag
            if (
                milestones["first_eos_onset"] is None
                and metrics.get("eos_stop_rate", 0.0) >= gates["eos_stop_rate_onset"]
            ):
                milestones["first_eos_onset"] = tag
            if (
                milestones["first_eos_reliable"] is None
                and metrics.get("eos_stop_rate", 0.0) >= gates["eos_stop_rate_reliable"]
            ):
                milestones["first_eos_reliable"] = tag
            if (
                milestones["first_action_selection"] is None
                and metrics.get("action_verb_accuracy", 0.0) >= gates["action_verb_accuracy_onset"]
            ):
                milestones["first_action_selection"] = tag
            best_cap = max(
                metrics.get("autonomous_mean_capability_score", 0.0),
                metrics.get("rlt_controller_mean_capability_score", 0.0),
            )
            if milestones["first_verified_capability"] is None and best_cap >= gates["verified_capability_score_onset"]:
                milestones["first_verified_capability"] = tag
            best_pri = max(
                metrics.get("autonomous_mean_primary_score", 0.0),
                metrics.get("rlt_controller_mean_primary_score", 0.0),
            )
            if milestones["first_nonzero_primary_score"] is None and best_pri >= gates["verified_primary_score_onset"]:
                milestones["first_nonzero_primary_score"] = tag
    ledger["emergence_milestones"] = milestones
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-id", choices=("10M", "20M", "50M", "100M", "200M"), required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--token-checks", type=Path, required=True)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPO_ROOT / "configs/research_v2_1/baby_goblin_emergence_ledger.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-gen-tokens", type=int, default=256)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "research_v2"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    from baby_goblin_early_specialist_exp import (
        TOKENIZER_SHA256,
        _build_disjoint_general_corpus,
        _evaluate_arm,
    )
    from local_worker import AlephNativeModel
    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    from datagoblin.research_v2.suite import digest
    from datagoblin.research_v2_1.controllers import ResearchController
    from datagoblin.research_v2_1.environment import ResearchSessionV21
    from datagoblin.research_v2_1.suite import make_suite as make_suite_v21

    tok_sha = _sha256_file(args.tokenizer)
    if tok_sha != TOKENIZER_SHA256:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")
    weights_sha = _sha256_file(args.weights)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    device = torch.device(args.device)

    model = AlephNativeModel().to(dtype=torch.bfloat16)
    state = {k: v.to(dtype=torch.bfloat16) for k, v in load_file(str(args.weights), device="cpu").items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    _tr_chunks, general_eval_chunks, _meta = _build_disjoint_general_corpus(REPO_ROOT, tokenizer, 384)
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    eval_seed = int(ledger["probe_contract"]["eval_seed"])
    eval_cases = make_suite_v21(seed=eval_seed, per_family=2, split="development")
    ev, gens = _evaluate_arm(
        model,
        tokenizer,
        device=device,
        eval_cases=eval_cases,
        general_eval_chunks=general_eval_chunks,
        token_checks_path=args.token_checks,
        max_seq_len=args.max_seq_len,
        max_gen_tokens=args.max_gen_tokens,
    )

    cases_by_id = {c["id"]: c for c in eval_cases}
    rlt = ResearchController("rlt_state_correction")
    auto_p, auto_c, ctrl_p, ctrl_c = [], [], [], []
    for row in gens:
        case = cases_by_id[row["task_id"]]
        txt = row["gen_text"]
        sa = ResearchSessionV21(case)
        sa.step_json(txt)
        ra = sa.result()
        auto_p.append(ra["primary_score"])
        auto_c.append(ra["capability_score"])

        sc = ResearchSessionV21(case)
        post = rlt.post_model_decision(
            case=case,
            state=sc.compact_state,
            turn_index=0,
            raw_text=txt,
            artifacts=sc._artifacts,
            remaining=sc._remaining(),
        )
        sc.step_json(post.executed_text or txt)
        pre2 = rlt.pre_turn_decision(case=case, state=sc.compact_state, turn_index=1, remaining=sc._remaining())
        if pre2.skip_model_call and pre2.executed_text:
            sc.step_json(pre2.executed_text)
        rc = sc.result()
        ctrl_p.append(rc["primary_score"])
        ctrl_c.append(rc["capability_score"])

    n = max(1, len(gens))
    arm_summary = {
        "valid_json_rate": ev["syntax_acquisition"]["valid_json_rate"],
        "schema_valid_action_rate": ev["syntax_acquisition"]["schema_valid_action_rate"],
        "eos_stop_rate": ev["eos_termination"]["eos_stop_rate"],
        "mean_teacher_forced_eos_prob": ev["eos_termination"]["mean_teacher_forced_eos_prob"],
        "action_verb_accuracy": ev["action_selection"]["action_verb_accuracy"],
        "teacher_forced_target_nll_nats": ev["action_selection"]["teacher_forced_target_nll_nats"],
        "general_val_nll_nats": ev["general_language_retention"]["general_val_nll_nats"],
        "autonomous_mean_primary_score": round(sum(auto_p) / n, 4),
        "autonomous_mean_capability_score": round(sum(auto_c) / n, 4),
        "rlt_controller_mean_primary_score": round(sum(ctrl_p) / n, 4),
        "rlt_controller_mean_capability_score": round(sum(ctrl_c) / n, 4),
        "weights_sha256": weights_sha,
    }
    for stage in ledger["stages"]:
        if stage["stage_id"] == args.stage_id:
            stage["status"] = "evaluated"
            if stage["checkpoint_sha256"] is None:
                stage["checkpoint_sha256"] = weights_sha
            stage["arms"][args.arm_id] = arm_summary
            stage["receipt_sha256"] = digest(stage["arms"])
            break

    recompute_emergence_milestones(ledger)
    args.ledger.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage_id": args.stage_id, "arm_id": args.arm_id, "summary": arm_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
