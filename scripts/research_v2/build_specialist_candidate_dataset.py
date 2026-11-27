"""Build an independently verified, train-only Research v2 specialist candidate dataset.

Generates fresh train-only multi-turn episodes across all eight Research v2
families (prioritizing failure-directed TRM/LDT experiments, stopping
decisions, partial-state retention, and externalized persona control), executes
every trajectory through the real ResearchSession evaluator, verifies zero
overlap with development, held-out, and Jinn/Beast validation/test roots,
screens duplicates, and counts exact accepted tokens with the frozen Aleph
SmolLM2 tokenizer (SHA-256 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c).

Also optionally computes non-sensitive Aleph token accounting for the private
Jinn/Beast 2,880-target specialist intake without copying private rows into
public outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from datagoblin.research_v2.environment import ResearchSession
from datagoblin.research_v2.suite import (
    LEARNERS,
    canon,
    digest,
    make_suite,
    persona_memory,
    reference_answer,
)

FROZEN_ALEPH_TOKENIZER_SHA256 = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exclusions(paths: list[Path]) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    roots: set[str] = set()
    semantics: set[str] = set()
    manifests: list[dict[str, Any]] = []
    for p in paths:
        raw = p.read_bytes()
        data = json.loads(raw.decode("utf-8"))
        r_ids = data.get("held_root_ids", [])
        s_ids = data.get("held_semantic_sha256", [])
        roots.update(r_ids)
        semantics.update(s_ids)
        manifests.append({
            "source_label": p.parent.name + "/" + p.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "held_root_count": len(r_ids),
            "held_semantic_count": len(s_ids),
        })
    return roots, semantics, manifests


def _build_verified_trajectory(case: dict[str, Any], variant: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run a real trajectory through ResearchSession and collect verified (prompt, action, obs) turns."""
    session = ResearchSession(case)
    obs = session.observation(status="ready")
    turns: list[dict[str, Any]] = []
    family = case["family"]

    def step(action: dict[str, Any], role_tag: str) -> dict[str, Any]:
        nonlocal obs
        prompt = case["prompt"] + "\nCurrent observation:\n" + canon(obs) + "\nAction:\n"
        action_text = canon(action)
        next_obs = session.step_json(action_text)
        turns.append({
            "turn": len(turns) + 1,
            "curriculum_role": role_tag,
            "prompt": prompt,
            "target_action": action_text,
            "observation_after": next_obs,
        })
        obs = next_obs
        return next_obs

    if family in LEARNERS:
        base_cfg: dict[str, Any] = {"steps": 0}
        train_cfg: dict[str, Any] = {"steps": 64}
        if family == "trm_training":
            base_cfg["recursions"] = 8
            train_cfg["recursions"] = 8

        if variant == "failure_repair":
            # Failure-directed recovery: start with an invalid config (missing baseline or bad steps)
            # or under-trained probe, observe environment feedback, then execute matched pair + stop.
            if family == "trm_training":
                step({"action": "run_experiment", "config": {"steps": 32, "recursions": 2}}, "under_recursed_probe")
                step({"action": "run_experiment", "config": base_cfg}, "matched_zero_baseline")
                step({"action": "run_experiment", "config": train_cfg}, "repaired_full_recursion_fit")
                base_id, sel_id = "a2", "a3"
            else:
                # LDT/persona: demonstrate checking critical_veto=True and matched baseline
                step(
                    {"action": "run_experiment", "config": {"steps": 0, "critical_veto": True}},
                    "matched_zero_baseline",
                )
                step(
                    {"action": "run_experiment", "config": {"steps": 32, "critical_veto": True}},
                    "intermediate_fit",
                )
                step(
                    {"action": "run_experiment", "config": {"steps": 64, "critical_veto": True}},
                    "final_fit_stop_decision",
                )
                base_id, sel_id = "a1", "a3"
        else:
            step({"action": "run_experiment", "config": base_cfg}, "matched_zero_baseline")
            step({"action": "run_experiment", "config": train_cfg}, "trained_candidate_fit")
            base_id, sel_id = "a1", "a2"

        summaries = {a["id"]: a for a in obs["artifacts"]}
        before = summaries[base_id]["calibration_accuracy"]
        after = summaries[sel_id]["calibration_accuracy"]
        claim = "ties" if abs(after - before) < 1e-12 else ("improves" if after > before else "worse")
        ans: dict[str, Any] = {"artifact_id": sel_id, "baseline_id": base_id, "claim": claim}
        if family == "persona_externalization":
            ans["memory"] = persona_memory(case["public"])
        step({"action": "submit", "answer": ans}, "verified_submit_with_partial_state")
    else:
        gold = reference_answer(case)
        if variant == "check_and_stop":
            # Demonstrate stopping discipline: check candidate once, verify check_passed=true, immediately submit
            step({"action": "check", "answer": gold}, "verifier_check_before_stop")
            step({"action": "submit", "answer": gold}, "verified_stop_and_submit")
        else:
            step({"action": "submit", "answer": gold}, "direct_verified_submit")

    result = session.result()
    return turns, result


def _count_jinn_private_tokens(jinn_train_jsonl: Path, tokenizer: Tokenizer) -> dict[str, Any]:
    rows = [json.loads(line) for line in jinn_train_jsonl.read_text(encoding="utf-8").splitlines() if line]
    total_seq_toks = 0
    total_target_toks = 0
    by_frame: Counter[str] = Counter()
    roots = set()
    for row in rows:
        inp = canon(row["input"])
        tgt = canon(row["target"])
        n_inp = len(tokenizer.encode(inp, add_special_tokens=False).ids)
        n_tgt = len(tokenizer.encode(tgt, add_special_tokens=False).ids)
        total_seq_toks += n_inp + n_tgt
        total_target_toks += n_tgt
        by_frame[row["policy_frame"]] += n_inp + n_tgt
        roots.add(row["root_id"])
    return {
        "source_id": "jinn-beast-audited-tags-v1",
        "train_jsonl_sha256": _sha256_file(jinn_train_jsonl),
        "records": len(rows),
        "unique_train_roots": len(roots),
        "tokenizer_sha256": FROZEN_ALEPH_TOKENIZER_SHA256,
        "total_sequence_tokens": total_seq_toks,
        "supervised_target_tokens": total_target_toks,
        "tokens_by_frame": dict(sorted(by_frame.items())),
        "visibility": "private_manifest_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--exclude-registry", type=Path, action="append", default=[])
    parser.add_argument("--jinn-train-jsonl", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026112601)
    parser.add_argument("--per-family", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    tok_sha = _sha256_file(args.tokenizer)
    if tok_sha != FROZEN_ALEPH_TOKENIZER_SHA256:
        raise RuntimeError(f"Aleph tokenizer SHA mismatch: {tok_sha} != {FROZEN_ALEPH_TOKENIZER_SHA256}")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    excl_roots, excl_semantics, excl_manifests = _load_exclusions(args.exclude_registry)
    # Build fresh train-only tasks using a dedicated training seed (never dev=20261126 or heldout seed)
    cases = make_suite(seed=args.seed, per_family=args.per_family, split="development")

    admitted_records: list[dict[str, Any]] = []
    seen_roots: set[str] = set()
    seen_turn_digests: set[str] = set()
    excluded_collisions = Counter()
    failed_verifications = 0
    tokens_by_family: Counter[str] = Counter()
    supervised_tokens_by_family: Counter[str] = Counter()
    records_by_family: Counter[str] = Counter()
    episodes_by_family: Counter[str] = Counter()
    priority_buckets: Counter[str] = Counter()

    for idx, case in enumerate(cases):
        if case["root_id"] in excl_roots or case["semantic_sha256"] in excl_semantics:
            excluded_collisions["held_or_dev_collision"] += 1
            continue
        if case["root_id"] in seen_roots:
            excluded_collisions["duplicate_root"] += 1
            continue
        seen_roots.add(case["root_id"])

        family = case["family"]
        if family in LEARNERS:
            variant = "failure_repair" if (idx % 2 == 0) else "standard_matched_pair"
        else:
            variant = "check_and_stop" if (idx % 2 == 0) else "direct_submit"

        turns, result = _build_verified_trajectory(case, variant)
        if not (result["integrity_pass"] and result["success"] and result["completed"]):
            failed_verifications += 1
            continue

        episodes_by_family[family] += 1
        for t in turns:
            turn_digest = digest([t["prompt"], t["target_action"]])
            if turn_digest in seen_turn_digests:
                excluded_collisions["duplicate_turn"] += 1
                continue
            seen_turn_digests.add(turn_digest)

            prompt_ids = tokenizer.encode(t["prompt"], add_special_tokens=False).ids
            target_ids = tokenizer.encode(t["target_action"], add_special_tokens=False).ids
            total_toks = len(prompt_ids) + len(target_ids)
            if total_toks > 2048:
                excluded_collisions["exceeds_2048_context"] += 1
                continue

            if family in ("trm_training", "ldt_filter"):
                bucket = "failure_directed_trm_ldt"
            elif family == "persona_externalization":
                bucket = "externalized_persona_control"
            elif t["turn"] > 1 or family == "cursor_recovery":
                bucket = "partial_state_retention_and_stopping"
            else:
                bucket = "stopping_and_symbolic_verification"

            priority_buckets[bucket] += total_toks
            tokens_by_family[family] += total_toks
            supervised_tokens_by_family[family] += len(target_ids)
            records_by_family[family] += 1

            admitted_records.append({
                "schema": "datagoblin-research-v2-specialist-sft-record-1",
                "record_id": f"{case['id']}:t{t['turn']}:{variant}",
                "episode_id": case["id"],
                "root_id": case["root_id"],
                "semantic_sha256": case["semantic_sha256"],
                "family": family,
                "split": "train_only",
                "curriculum_bucket": bucket,
                "curriculum_role": t["curriculum_role"],
                "variant": variant,
                "turn": t["turn"],
                "prompt": t["prompt"],
                "target": t["target_action"],
                "aleph_tokens": {
                    "prompt_tokens": len(prompt_ids),
                    "supervised_target_tokens": len(target_ids),
                    "total_tokens": total_toks,
                },
                "source_rights": {
                    "origin": "synthetic_verified_datagoblin_research_v2",
                    "license": "Apache-2.0",
                    "redistribution": "public_and_internal_train_only",
                    "generator_seed": args.seed,
                },
                "verifier_receipt": {
                    "episode_result_sha256": result["result_sha256"],
                    "primary_score": result["primary_score"],
                    "integrity_pass": result["integrity_pass"],
                    "success": result["success"],
                    "artifact_hashes": result["artifact_hashes"],
                    "events_sha256": result["events_sha256"],
                },
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.out_dir / "candidate_train.jsonl"
    data_bytes = "".join(canon(r) + "\n" for r in admitted_records).encode("utf-8")
    data_path.write_bytes(data_bytes)

    exclusion_manifest = {
        "schema": "datagoblin-research-v2-exclusion-manifest-1",
        "excluded_registries": excl_manifests,
        "total_excluded_roots_checked": len(excl_roots),
        "total_excluded_semantics_checked": len(excl_semantics),
        "admitted_root_overlap_with_exclusions": len(seen_roots & excl_roots),
        "collisions_skipped": dict(sorted(excluded_collisions.items())),
        "failed_verifications_skipped": failed_verifications,
        "duplicate_roots_in_output": 0,
        "duplicate_turns_in_output": 0,
    }
    excl_path = args.out_dir / "exclusion_manifest.json"
    excl_bytes = (json.dumps(exclusion_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    excl_path.write_bytes(excl_bytes)

    jinn_accounting = None
    if args.jinn_train_jsonl is not None and args.jinn_train_jsonl.is_file():
        jinn_accounting = _count_jinn_private_tokens(args.jinn_train_jsonl, tokenizer)

    receipt = {
        "schema": "datagoblin-research-v2-specialist-dataset-receipt-1",
        "split": "train_only",
        "generator_seed": args.seed,
        "aleph_tokenizer_sha256": tok_sha,
        "episodes_verified": sum(episodes_by_family.values()),
        "episodes_by_family": dict(sorted(episodes_by_family.items())),
        "records_admitted": len(admitted_records),
        "unique_roots_admitted": len({r["root_id"] for r in admitted_records}),
        "records_by_family": dict(sorted(records_by_family.items())),
        "accepted_tokens_total": sum(tokens_by_family.values()),
        "accepted_supervised_tokens_total": sum(supervised_tokens_by_family.values()),
        "accepted_tokens_by_family": dict(sorted(tokens_by_family.items())),
        "accepted_supervised_tokens_by_family": dict(sorted(supervised_tokens_by_family.items())),
        "accepted_tokens_by_curriculum_bucket": dict(sorted(priority_buckets.items())),
        "private_jinn_beast_intake_aleph_tokens": jinn_accounting,
        "files": {
            "candidate_train.jsonl": {"sha256": hashlib.sha256(data_bytes).hexdigest(), "bytes": len(data_bytes)},
            "exclusion_manifest.json": {"sha256": hashlib.sha256(excl_bytes).hexdigest(), "bytes": len(excl_bytes)},
        },
        "source_rights": {
            "synthetic_research_v2_traces": "Apache-2.0 (public and internal train-only)",
            "private_jinn_beast_intake": "operator_authorized_internal_intake (kept outside Git)",
        },
    }
    receipt["receipt_sha256"] = digest(receipt)
    receipt_text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    (args.out_dir / "DATASET_RECEIPT.json").write_text(receipt_text, encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
