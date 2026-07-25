from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from datagoblin.data_etiquette import (
    CollectionDisposition,
    DataCollectionRecord,
    EtiquettePolicy,
    EvidenceReceiptStore,
)
from datagoblin.data_etiquette.pipeline import _score, build_etiquette_training_dataset, run_etiquette_demo
from datagoblin.domain.evidence import EvidenceType
from datagoblin.serialization import dump_structured, load_structured


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_etiquette_demo_separates_custody_and_builds_safe_handoff(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    out = tmp_path / "etiquette"
    report = run_etiquette_demo(root / "configs/data_etiquette.yaml", out, seed=7)

    assert report["evidence_status"] == "pipeline_smoke_only_not_trained_ldt_evidence"
    assert report["sorter_authority"] == "recommendation_only"
    assert report["split_results"]["sealed_poison"]["critical_false_release_rate"] == 0.0

    public_rows: list[dict[str, object]] = []
    for split in ("train", "calibration", "ordinary_test", "ood_test", "sealed_poison"):
        rows = _jsonl(out / f"public_dataset/{split}.jsonl")
        public_rows.extend(rows)
        if split == "sealed_poison":
            assert all("authoritative_label" not in row for row in rows)
        else:
            assert all("authoritative_label" in row for row in rows)
        for row in rows:
            record = row["record"]
            assert isinstance(record, dict)
            assert "raw_content" not in record
            assert "filesystem_path" not in record

    predictions = _jsonl(out / "sorter_predictions.jsonl")
    assert len(predictions) == len(public_rows)
    assert all(row["authority"] == "MODEL_RECOMMENDATION_ONLY" for row in predictions)
    assert all(len(str(row["record_hash"])) == 64 for row in predictions)

    queue = _jsonl(out / "adjudication_queue.jsonl")
    assert any(row["category"] == "cross_split_duplicate" for row in queue)
    assert all("actual" not in row for row in queue)

    handoff_dir = out / "larger_model_handoff"
    handoff = load_structured(handoff_dir / "corpus_handoff_manifest.json")
    assert not handoff["sorter_output_is_authority"]
    assert not handoff["evaluation_items_exported_to_training"]
    assert not handoff["raw_content_in_manifest"]
    assert not handoff["larger_model_training_permitted"]
    lanes = _jsonl(handoff_dir / "verified_supervision.jsonl") + _jsonl(
        handoff_dir / "soft_discovery.jsonl"
    )
    assert lanes
    assert all(row["dedup_group"] != "cross-split-7" for row in lanes)
    public_by_id = {str(row["item_id"]): row for row in public_rows}
    released_categories = {
        public_by_id[str(row["item_id"])]["category"]
        for row in lanes
    }
    assert released_categories == {"clean_verified", "model_generated"}

    trainer_plan = load_structured(out / "ldt_sorter_trainer_plan.json")
    summary = load_structured(out / "ldt_sorter_summary.json")
    cleanup = load_structured(out / "cleanup_summary.json")
    assert not trainer_plan["real_training_permitted"]
    assert not trainer_plan["weights_loaded"]
    assert summary["steps_completed"] == 0
    assert not summary["weights_loaded"]
    assert cleanup["cleanup_passed"]


def test_only_receipted_independent_evidence_can_enter_verified_lane(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    out = tmp_path / "etiquette"
    run_etiquette_demo(root / "configs/data_etiquette.yaml", out, seed=11)
    train_row = next(
        row
        for row in _jsonl(out / "public_dataset/train.jsonl")
        if row["category"] == "clean_verified"
    )
    record = DataCollectionRecord.from_jsonable(train_row["record"])
    evidence_store = EvidenceReceiptStore.from_jsonl(
        out / "private_steward/frozen_evidence_verifiers.yaml",
        out / "private_steward/evidence_receipts.jsonl",
    )
    policy = EtiquettePolicy(root / "configs/data_etiquette.yaml", evidence_store=evidence_store)
    assert policy.decide(record).disposition == CollectionDisposition.VERIFIED_TRAIN

    no_store = EtiquettePolicy(root / "configs/data_etiquette.yaml")
    no_store_decision = no_store.decide(record)
    assert no_store_decision.disposition == CollectionDisposition.QUARANTINE
    assert "missing_independent_verification_receipt" in no_store_decision.violations

    unreceipted = replace(
        record,
        verified_by=None,
        verification_receipt_sha256=None,
        verification_subject_sha256=None,
        verification_custody="NONE",
    )
    unreceipted_decision = policy.decide(unreceipted)
    assert unreceipted_decision.disposition == CollectionDisposition.QUARANTINE
    assert "missing_independent_verification_receipt" in unreceipted_decision.violations

    proposer_receipt = replace(record, verification_custody="PROPOSER")
    assert policy.decide(proposer_receipt).disposition == CollectionDisposition.QUARANTINE

    wrong_subject = replace(record, verification_subject_sha256="0" * 64)
    assert policy.decide(wrong_subject).disposition == CollectionDisposition.QUARANTINE

    replayed_receipt = replace(
        record,
        content_sha256="1" * 64,
        verification_subject_sha256="1" * 64,
    )
    replayed_decision = policy.decide(replayed_receipt)
    assert replayed_decision.disposition == CollectionDisposition.QUARANTINE
    assert "invalid_independent_verification_receipt" in replayed_decision.violations

    forged_verifier = replace(record, verified_by="proposer-selected-verifier")
    forged_decision = policy.decide(forged_verifier)
    assert forged_decision.disposition == CollectionDisposition.QUARANTINE
    assert "invalid_independent_verification_receipt" in forged_decision.violations

    wrong_use = replace(record, allowed_uses=("EVALUATION",))
    assert policy.decide(wrong_use).disposition == CollectionDisposition.QUARANTINE

    human_claim = replace(
        record,
        evidence_type=EvidenceType.HUMAN_REVIEWED,
        verified_by="same-team-reviewer",
    )
    assert policy.decide(human_claim).disposition == CollectionDisposition.DISCOVERY_ONLY

    with pytest.raises(ValueError, match="private fields"):
        replace(record, public_metadata={"category": "bad", "raw_content": "do not publish"})

    with pytest.raises(ValueError, match="filesystem paths"):
        replace(record, provenance_refs=(r"C:\private\source.txt",))

    receipt_rows = _jsonl(out / "private_steward/evidence_receipts.jsonl")
    receipt_rows[0]["verdict"] = "FAIL"
    tampered_path = tmp_path / "tampered-receipts.jsonl"
    tampered_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in receipt_rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="receipt hash mismatch"):
        EvidenceReceiptStore.from_jsonl(
            out / "private_steward/frozen_evidence_verifiers.yaml",
            tampered_path,
        )

    drifted_registry = load_structured(out / "private_steward/frozen_evidence_verifiers.yaml")
    drifted_registry["registry_id"] = "changed-after-receipt-issuance"
    drifted_registry_path = tmp_path / "drifted-registry.yaml"
    dump_structured(drifted_registry_path, drifted_registry)
    with pytest.raises(ValueError, match="active verifier registry"):
        EvidenceReceiptStore.from_jsonl(
            drifted_registry_path,
            out / "private_steward/evidence_receipts.jsonl",
        )


def test_training_dataset_separates_ml_partition_from_enterprise_use(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    dataset_dir = tmp_path / "fit-v2/dataset"
    manifest = build_etiquette_training_dataset(
        root / "configs/data_etiquette.yaml",
        dataset_dir,
        seed=7,
    )
    assert "independent of record.intended_split" in manifest["split_contract"]
    train_rows = _jsonl(dataset_dir / "train.jsonl")
    train_dispositions = {row["authoritative_label"]["disposition"] for row in train_rows}
    assert CollectionDisposition.EVALUATION_ONLY.value in train_dispositions
    assert CollectionDisposition.SEALED_EVALUATION.value in train_dispositions
    assert any(row["split"] != row["record"]["intended_split"] for row in train_rows)
    sealed_rows = _jsonl(dataset_dir / "sealed_poison.jsonl")
    assert sealed_rows
    assert all("authoritative_label" not in row for row in sealed_rows)
    sealed_labels = _jsonl(dataset_dir.parent / "private_steward/sealed_labels.jsonl")
    assert len(sealed_labels) == len(sealed_rows)


def test_abstained_verified_proposal_is_not_counted_as_a_release() -> None:
    result = _score(
        [
            {
                "actual": CollectionDisposition.REJECT.value,
                "predicted": CollectionDisposition.VERIFIED_TRAIN.value,
                "correct": False,
                "confidence": 0.4,
                "abstained": True,
                "actual_flags": {"secret_or_credential"},
                "predicted_flags": set(),
            }
        ]
    )
    assert result["critical_false_release_rate"] == 0.0
    assert result["coverage"] == 0.0
