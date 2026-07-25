from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from datagoblin.adapter_verifier import (
    build_corpus_projection,
    build_safe_field_projection,
    build_semantic_review_queue,
    evaluate_adapter_setup,
    finalize_semantic_review,
    run_mock_setups_on_dataset,
    run_verifier_demo,
)
from datagoblin.data_etiquette import (
    EvidenceReceiptStore,
    EvidenceVerifierRegistry,
    issue_evidence_receipt,
)
from datagoblin.domain.evidence import EvidenceType
from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import dump_structured, load_structured


def test_adapter_verifier_demo_is_content_free_sealed_and_discriminating(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    out = tmp_path / "adapter-verifier"
    comparison = run_verifier_demo(root / "configs/curation_adapter_verifier.yaml", out, seed=7)

    assert comparison["winner"] is None
    assert set(comparison["setups"]) == {
        "strict-policy-mock",
        "permissive-map-all-mock",
        "conservative-abstain-mock",
    }
    reports = {
        setup: load_structured(out / "evaluations" / setup / "evaluation_report.json")
        for setup in comparison["setups"]
    }
    assert reports["strict-policy-mock"]["overall"]["exact_record_accuracy"] == 1.0
    assert reports["strict-policy-mock"]["overall"]["parse_status_accuracy"] == 1.0
    assert reports["permissive-map-all-mock"]["overall"]["privacy_leak_rate"] > 0.0
    assert reports["permissive-map-all-mock"]["overall"]["critical_false_release_rate"] > 0.0
    assert reports["conservative-abstain-mock"]["overall"]["coverage"] == 0.0
    assert all(not report["promotion_assessment"]["eligible"] for report in reports.values())

    summary = load_structured(out / "summary.json")
    assert not summary["weights_loaded"]
    assert not summary["network_used"]
    assert not summary["raw_corpus_values_read"]

    manifest = load_structured(out / "public_dataset/dataset_manifest.json")
    for split in manifest["splits"].values():
        rows = (out / "public_dataset" / split["path"]).read_text(encoding="utf-8").splitlines()
        assert rows
        assert all("authoritative_label" not in json.loads(line) for line in rows)
    public_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (out / "public_dataset").glob("*.jsonl")
    )
    assert "D:\\" not in public_text
    assert '"content":' not in public_text


def test_adapter_verifier_rejects_dataset_rebinding_and_authority_escalation(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    out = tmp_path / "adapter-verifier"
    run_verifier_demo(root / "configs/curation_adapter_verifier.yaml", out, seed=11)
    candidate_path = out / "setups/strict-policy-mock/candidate_manifest.json"
    candidate = load_structured(candidate_path)
    candidate["dataset_manifest_sha256"] = "0" * 64
    rebound = tmp_path / "rebound.json"
    dump_structured(rebound, candidate)
    with pytest.raises(ValueError, match="not bound"):
        evaluate_adapter_setup(
            root / "configs/curation_adapter_verifier.yaml",
            out / "public_dataset/dataset_manifest.json",
            out / "private_steward/ground_truth.jsonl",
            rebound,
            out / "setups/strict-policy-mock/predictions.jsonl",
            tmp_path / "rejected-rebind",
        )

    candidate = load_structured(candidate_path)
    candidate["authoritative_label_access"] = True
    escalated = tmp_path / "escalated.json"
    dump_structured(escalated, candidate)
    assert candidate["dataset_manifest_sha256"] == sha256_file(out / "public_dataset/dataset_manifest.json")
    with pytest.raises(ValueError, match="authoritative labels"):
        evaluate_adapter_setup(
            root / "configs/curation_adapter_verifier.yaml",
            out / "public_dataset/dataset_manifest.json",
            out / "private_steward/ground_truth.jsonl",
            escalated,
            out / "setups/strict-policy-mock/predictions.jsonl",
            tmp_path / "rejected-authority",
        )


def test_content_free_projection_preserves_cohorts_without_values_or_paths(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    decisions = tmp_path / "private-source" / "decisions"
    decisions.mkdir(parents=True)
    source_rows = [
        {
            "env": "Centauri",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "medium",
            "content": "DO-NOT-EMIT-CENTAURI",
            "legal_actions": ["DO-NOT-EMIT-CENTAURI-ACTION"],
            "public_state": {"turn": 1},
        },
        {
            "env": "They Sing",
            "model": "gpt-5.6-terra",
            "reasoning_effort": "high",
            "sessionId": "DO-NOT-EMIT-SESSION",
            "legal_actions": ["DO-NOT-EMIT-THEY-SING-ACTION"],
            "phase": "DO-NOT-EMIT-THEY-SING-PHASE",
            "target": {
                "action": {
                    "orders": ["DO-NOT-EMIT-THEY-SING-ORDER"],
                    "reasoning": "DO-NOT-EMIT-THEY-SING-REASONING",
                }
            },
            "public_message": "DO-NOT-EMIT-MESSAGE",
        },
        {
            "env": "Court",
            "model": "gpt-5.6-luna",
            "reasoning_effort": "xhigh",
            "hidden_win_condition": "DO-NOT-EMIT-CONDITION",
            "legal_actions": ["DO-NOT-EMIT-ACTION"],
            "phase": "DO-NOT-EMIT-COURT-PHASE",
        },
    ]
    with gzip.open(decisions / "fixture.jsonl.gz", "wt", encoding="utf-8", newline="\n") as handle:
        for row in source_rows:
            handle.write(json.dumps(row) + "\n")
    receipt = tmp_path / "private-source" / "pilot.json"
    dump_structured(
        receipt,
        {
            "accepted": True,
            "tokens": {
                "by_cell": {
                    "gpt-5.6-sol:medium:neutral": 1,
                    "gpt-5.6-terra:high:neutral": 1,
                    "gpt-5.6-luna:xhigh:neutral": 1,
                }
            },
        },
    )
    diplomacy = tmp_path / "private-diplomacy"
    game_dir = diplomacy / "games" / "train"
    game_dir.mkdir(parents=True)
    manifest = diplomacy / "manifest.json"
    dump_structured(manifest, {"schema_version": "fixture.v1", "games": 1})
    with gzip.open(game_dir / "game.jsonl.gz", "wt", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "source_checks": {"valid": True},
                    "trace": {"adjudicated_outcomes": [{"orders": ["DO-NOT-EMIT-ORDER"]}]},
                    "public_press": "DO-NOT-EMIT-PRESS",
                    "strategic_plan": "DO-NOT-EMIT-PLAN",
                }
            )
            + "\n"
        )

    out = tmp_path / "projection"
    build_corpus_projection(
        root / "configs/curation_adapter_verifier.yaml",
        receipt,
        decisions,
        [diplomacy],
        [manifest],
        out,
        max_records_per_cohort=2,
    )
    public_text = "\n".join(path.read_text(encoding="utf-8") for path in (out / "public_dataset").glob("*"))
    assert "DO-NOT-EMIT" not in public_text
    assert str(tmp_path) not in public_text
    assert "gpt-5.6-sol" in public_text
    assert "unresolved" in public_text

    comparison = run_mock_setups_on_dataset(
        root / "configs/curation_adapter_verifier.yaml",
        out / "public_dataset/dataset_manifest.json",
        out / "private_steward/ground_truth.jsonl",
        out / "gym",
    )
    assert set(comparison["setups"]) == {
        "strict-policy-mock",
        "permissive-map-all-mock",
        "conservative-abstain-mock",
    }
    report = load_structured(out / "gym/evaluations/strict-policy-mock/evaluation_report.json")
    assert set(report["source_model_results"]) == {
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "unresolved",
    }
    assert report["source_agent_results"]["gpt-5.6-sol|medium"]["examples"] == 2
    assert report["benchmark_diagnostics"] == {
        "evaluated_real_records": 4,
        "environment_canaries": 4,
        "hazard_saturation_rate": 1.0,
        "non_abstention_opportunities": 0,
        "fitness_scope": "safety_gate_only",
        "source_agent_quality_comparison_valid": False,
        "warning": (
            "All real projected rows require abstention; exact accuracy cannot demonstrate useful adapter coverage."
        ),
    }

    safe_out = tmp_path / "safe-projection"
    build_safe_field_projection(
        root / "configs/curation_adapter_verifier.yaml",
        receipt,
        decisions,
        [diplomacy],
        [manifest],
        safe_out,
        max_records_per_cohort=2,
    )
    safe_public_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (safe_out / "public_dataset").glob("*")
    )
    assert "DO-NOT-EMIT" not in safe_public_text
    assert str(tmp_path) not in safe_public_text
    verification = load_structured(safe_out / "private_steward/derivation_verification.json")
    assert verification["verification_passed"]
    assert verification["verified_records"] == 4
    assert not verification["source_quality_verified"]
    assert not verification["split_contamination_detected"]
    safe_manifest = load_structured(safe_out / "public_dataset/dataset_manifest.json")
    assert safe_manifest["split_grouping"] == (
        "episode/game lineage, exact source content, and exact/near safe-value semantic connected components"
    )

    review_out = tmp_path / "semantic-review"
    review_manifest = build_semantic_review_queue(
        root / "configs/semantic_review.yaml",
        safe_out / "public_dataset/dataset_manifest.json",
        safe_out / "private_projection/derivation_ledger.jsonl",
        [decisions, diplomacy],
        review_out,
        max_items=4,
        max_per_source_file=4,
    )
    assert review_manifest["selected_items"] == 4
    assert not review_manifest["public_values_emitted"]
    assert "DO-NOT-EMIT" not in (review_out / "public_review/review_queue.jsonl").read_text(
        encoding="utf-8"
    )
    assert "DO-NOT-EMIT" in (review_out / "reviewer_a/review_packets.jsonl").read_text(
        encoding="utf-8"
    )
    review_report = finalize_semantic_review(
        review_out / "public_review/review_manifest.json",
        review_out / "private_steward/review_map.jsonl",
        [
            review_out / "reviewer_a/annotations.jsonl",
            review_out / "reviewer_b/annotations.jsonl",
        ],
        review_out / "private_steward/near_duplicate_audit.json",
        tmp_path / "semantic-review-final",
    )
    assert review_report["review_gate_complete"]
    assert not review_report["training_release_authorized"]
    assert "independent_environment_outcome_evidence_missing" in review_report["promotion_blockers"]
    assert review_report["environment_evidence"]["coverage"] == 0.0
    assert review_report["environment_evidence"]["challenge_count"] == 4

    registry = EvidenceVerifierRegistry(root / "configs/game_evidence_verifiers.yaml")
    methods = {
        "centauri": (
            "centauri-environment-replay-v1",
            "centauri-state-transition-replay-v1",
            EvidenceType.ENVIRONMENT_VERIFIED,
        ),
        "they_sing": (
            "they-sing-environment-replay-v1",
            "they-sing-state-transition-replay-v1",
            EvidenceType.ENVIRONMENT_VERIFIED,
        ),
        "court": (
            "court-environment-replay-v1",
            "court-state-transition-replay-v1",
            EvidenceType.ENVIRONMENT_VERIFIED,
        ),
        "diplomacy": (
            "diplomacy-adjudication-replay-v1",
            "diplomacy-adjudication-replay-v1",
            EvidenceType.INDEPENDENT_HARNESS,
        ),
    }
    review_map = [
        json.loads(line)
        for line in (review_out / "private_steward/review_map.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    unique_items = {row["item_id"]: row for row in review_map}.values()
    receipts = []
    for row in unique_items:
        verifier_id, method_id, evidence_type = methods[row["game"]]
        receipts.append(
            issue_evidence_receipt(
                registry,
                subject_sha256=row["record_hash"],
                evidence_type=evidence_type,
                verifier_id=verifier_id,
                method_id=method_id,
                asserted_by=row["model_family"],
                verdict="PASS",
                challenge_sha256=sha256_json({"item_id": row["item_id"], "kind": "challenge"}),
                observation_sha256=sha256_json({"item_id": row["item_id"], "kind": "observation"}),
                claim_types=("OUTCOME_VALIDITY",),
                artifact_sha256s=(sha256_json({"item_id": row["item_id"], "kind": "replay"}),),
            )
        )
    first_row = next(iter({row["item_id"]: row for row in review_map}.values()))
    first_verifier, first_method, first_evidence_type = methods[first_row["game"]]
    rule_only_receipt = issue_evidence_receipt(
        registry,
        subject_sha256=first_row["record_hash"],
        evidence_type=first_evidence_type,
        verifier_id=first_verifier,
        method_id=first_method,
        asserted_by=first_row["model_family"],
        verdict="PASS",
        challenge_sha256=sha256_json({"item_id": first_row["item_id"], "kind": "rule-challenge"}),
        observation_sha256=sha256_json({"item_id": first_row["item_id"], "kind": "rule-observation"}),
        claim_types=("RULE_COMPLIANCE",),
        artifact_sha256s=(sha256_json({"item_id": first_row["item_id"], "kind": "rule-replay"}),),
    )
    wrong_claim_report = finalize_semantic_review(
        review_out / "public_review/review_manifest.json",
        review_out / "private_steward/review_map.jsonl",
        [
            review_out / "reviewer_a/annotations.jsonl",
            review_out / "reviewer_b/annotations.jsonl",
        ],
        review_out / "private_steward/near_duplicate_audit.json",
        tmp_path / "semantic-review-wrong-claim",
        evidence_store=EvidenceReceiptStore(registry, [rule_only_receipt, *receipts[1:]]),
    )
    assert wrong_claim_report["environment_evidence"]["coverage"] == 0.75
    assert wrong_claim_report["environment_evidence"]["challenge_count"] == 1
    assert "independent_environment_outcome_evidence_missing" in wrong_claim_report["promotion_blockers"]

    evidenced_report = finalize_semantic_review(
        review_out / "public_review/review_manifest.json",
        review_out / "private_steward/review_map.jsonl",
        [
            review_out / "reviewer_a/annotations.jsonl",
            review_out / "reviewer_b/annotations.jsonl",
        ],
        review_out / "private_steward/near_duplicate_audit.json",
        tmp_path / "semantic-review-evidenced",
        evidence_store=EvidenceReceiptStore(registry, receipts),
    )
    assert evidenced_report["environment_evidence"]["coverage"] == 1.0
    assert evidenced_report["environment_evidence"]["challenge_count"] == 0
    assert "independent_environment_outcome_evidence_missing" not in evidenced_report["promotion_blockers"]
    assert "review_sample_evidence_cannot_authorize_corpus_release" in evidenced_report["promotion_blockers"]
    assert not evidenced_report["training_release_authorized"]

    run_mock_setups_on_dataset(
        root / "configs/curation_adapter_verifier.yaml",
        safe_out / "public_dataset/dataset_manifest.json",
        safe_out / "private_steward/ground_truth.jsonl",
        safe_out / "gym",
    )
    safe_report = load_structured(
        safe_out / "gym/evaluations/strict-policy-mock/evaluation_report.json"
    )
    assert safe_report["overall"]["exact_record_accuracy"] == 1.0
    assert safe_report["overall"]["privacy_leak_rate"] == 0.0
    assert safe_report["benchmark_diagnostics"]["hazard_saturation_rate"] == 0.0
    assert safe_report["benchmark_diagnostics"]["non_abstention_opportunities"] == 4
    assert safe_report["benchmark_diagnostics"]["fitness_scope"] == "safety_and_coverage"
