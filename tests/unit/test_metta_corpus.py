"""Offline MeTTa intake, split custody and curation-oracle regression tests."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from datagoblin.data_etiquette.policy import EtiquettePolicy
from MeTTa.collect import ROOT, collect, git_blob_sha, make_rows, proposed_split, scan_forms
from MeTTa.curation_gym import VARIANTS, build_cases, policy_record, program_for

CATALOG = ROOT / "MeTTa" / "source_catalog.json"


def test_vendored_reference_samples_are_exact_git_blobs() -> None:
    manifest = json.loads((ROOT / "MeTTa" / "reference_samples" / "MANIFEST.json").read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    pinned = {(f["source_id"], f["path"]): f for f in catalog["files"]}
    assert manifest["measured"]["files"] == 10
    for entry in manifest["files"]:
        destination = ROOT / entry["vendored_path"]
        data = destination.read_bytes()
        assert len(data) == entry["bytes"]
        assert git_blob_sha(data) == entry["git_blob_sha"]
        assert pinned[(entry["source_id"], entry["path"])]["git_blob_sha"] == entry["git_blob_sha"]
        assert (ROOT / entry["license_file"]).read_text(encoding="utf-8").startswith("MIT License")


def test_pinned_source_inventory_is_accounted_for() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert catalog["measured"]["total_files"] == 196
    assert catalog["measured"]["total_git_reported_bytes"] == 4281494
    assert catalog["measured"]["discovery_candidates"] == 174
    assert catalog["measured"]["quarantine"] == 22
    assert all(len(row["git_blob_sha"]) == 40 for row in catalog["files"])
    assert all(len(source["commit"]) == 40 for source in catalog["sources"])


def test_scan_top_level_comments_strings_queries_and_nested_atoms() -> None:
    text = '; ignored\n(Fact a "(x; y)")\n!(match &self (Fact $x $y) ($x $y)) ; tail\n(A\n (B c))\n'
    forms = scan_forms(text)
    assert len(forms) == 3
    assert forms[0][2] == '(Fact a "(x; y)")'
    assert forms[1][2].startswith("!(match")
    assert forms[2][2].startswith("(A")
    assert scan_forms(" ; only comments\n") == []


@pytest.mark.parametrize("text", ["(unclosed", ")", '"open', "(bad (nest)"])
def test_scan_rejects_bad_balance(text: str) -> None:
    with pytest.raises(ValueError):
        scan_forms(text)


def test_static_rows_are_not_native_semantics_or_training_labels() -> None:
    data = b"(Facts A)\n!(match &self (Facts $x) $x)\n"
    row = {
        "source_id": "hyperon_reference",
        "path": "python/tests/scripts/demo.metta",
        "git_blob_sha": git_blob_sha(data),
        "bytes": len(data),
        "eligibility": "discovery_candidate",
        "proposed_split": "unassigned_until_root_dedup",
        "review_flags": [],
    }
    records = make_rows(row, data, "f" * 40)
    assert len(records) == 2
    assert records[0]["eligible_for_training"] is False
    assert records[0]["semantics_verified"] is False
    assert records[1]["kind"] == "query"
    assert records[0]["source"]["source_line"] == 1


def test_collector_verifies_pinned_git_tree_and_refuses_drift(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@localhost"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    script = root / "demo.metta"
    data = b"(Record A)\n!(match &self (Record $x) $x)\n"
    script.write_bytes(data)
    subprocess.run(["git", "-C", str(root), "add", "demo.metta"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    catalog = {
        "schema_version": "metta-source-catalog-v1",
        "sources": [{"source_id": "fixture", "repository": "fixture/test", "commit": head}],
        "files": [{"source_id": "fixture", "path": "demo.metta", "bytes": len(data),
                   "git_blob_sha": git_blob_sha(data), "eligibility": "discovery_candidate",
                   "review_flags": [], "proposed_split": "unassigned_until_root_dedup"}],
    }
    manifest_path = tmp_path / "catalog.json"
    manifest_path.write_text(json.dumps(catalog), encoding="utf-8")
    out = tmp_path / "output"
    result = collect(manifest_path, {"fixture": root}, out)
    assert result["forms"] == 2
    assert result["training_ready"] is False
    first_hash = result["forms_sha256"]
    assert collect(manifest_path, {"fixture": root}, out)["forms_sha256"] == first_hash
    script.write_text("(Record Changed)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source blob differs"):
        collect(manifest_path, {"fixture": root}, out)


def _one_row_for(split: str) -> tuple[dict, dict]:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    source_by_id = {s["source_id"]: s for s in catalog["sources"]}
    row = next(r for r in catalog["files"] if r["eligibility"] == "discovery_candidate"
               and proposed_split(r) == split)
    return row, source_by_id[row["source_id"]]


def test_policy_labels_are_from_existing_data_goblin_authority() -> None:
    row, source = _one_row_for("train_candidate")
    policy = EtiquettePolicy(ROOT / "configs" / "data_etiquette.yaml")
    kwargs = {"source_commit": source["commit"], "source_repository": source["repository"]}
    expectations = {
        "clean_metadata": "DISCOVERY_ONLY",
        "license_unknown": "QUARANTINE",
        "license_restricted": "REJECT",
        "secret_exposed": "REJECT",
        "hidden_reasoning": "REJECT",
        "model_weight": "REJECT",
        "consent_denied": "REJECT",
        "deletion_requested": "REJECT",
        "unredacted_personal_data": "QUARANTINE",
        "no_provenance": "QUARANTINE",
    }
    assert set(VARIANTS) == set(expectations)
    for variant, expected in expectations.items():
        record = policy_record(row, variant, **kwargs)
        assert policy.decide(record).disposition.value == expected
        if variant == "no_provenance":
            assert record.provenance_refs == ()
        else:
            assert record.provenance_refs[0].startswith("https://github.com/")
        assert "!(match &self" in program_for(record)


def test_heldout_curation_has_zero_teacher_label_exposure(tmp_path: Path) -> None:
    train, train_source = _one_row_for("train_candidate")
    heldout, heldout_source = _one_row_for("ood_reserved")
    manifest = {
        "schema_version": "metta-source-catalog-v1",
        "sources": [train_source, heldout_source],
        "files": [train, heldout],
    }
    source_path = tmp_path / "sources.json"
    source_path.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "curation"
    result = build_cases(source_path, ROOT / "configs" / "data_etiquette.yaml", out)
    assert result["cases"] == 2 * len(VARIANTS)
    assert result["train_candidate_labels"] == len(VARIANTS)
    assert result["evaluation_labels_written"] == 0
    rows = [json.loads(line) for line in (out / "cases.jsonl").read_text().splitlines()]
    assert all(x["training_ready"] is False and x["is_native_metta_verified"] is False for x in rows)
    assert all("policy_reference" not in x for x in rows if x["partition"] == "ood_reserved")
    assert all("policy_reference" in x for x in rows if x["partition"] == "train_candidate")
    assert all("public_envelope" in x["model_input"] for x in rows)
    assert build_cases(source_path, ROOT / "configs" / "data_etiquette.yaml", out)["cases_sha256"] == result[
        "cases_sha256"
    ]


def test_output_cannot_be_committed_as_source() -> None:
    with pytest.raises(ValueError, match="outside the committed MeTTa"):
        collect(CATALOG, {"not_an_source": ROOT}, ROOT / "MeTTa" / "bad")
