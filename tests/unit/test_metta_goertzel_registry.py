"""Rights and provenance contracts for metadata-only Goertzel MeTTa discovery."""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

from MeTTa.goertzel_registry import CATALOG, FILE_CEILING, audit_registry


@pytest.fixture
def manifest():
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def test_real_inventory_contains_no_upstream_source_contents(manifest):
    audited = audit_registry(manifest)
    assert audited["valid"] is True
    assert audited["sources"] == 10
    assert audited["metta_files"] == 760
    assert audited["reported_metta_bytes"] == 68_367_331
    assert audited["discovery_only"] == 107
    assert audited["quarantined"] == 653
    assert audited["zero_metta_sources"] == ["mork_metta", "moses_mork"]
    assert audited["training_ready"] is False
    assert audited["license_review_complete"] is False
    assert audited["verified_source_contents"] is False


def test_known_licensed_repos_still_only_allow_discovery(manifest):
    known = {s["source_id"]: s for s in manifest["sources"]}
    permitted = {
        item["source_id"] for item in manifest["source_files"]
        if item["disposition"] == "DISCOVERY_ONLY"
    }
    assert permitted == {"metamo", "metta_nars", "omega"}
    assert all(known[sid]["license_spdx"] in {"MIT", "Apache-2.0"} for sid in permitted)
    assert all(item["training_ready"] is False for item in manifest["source_files"])


def test_unknown_and_agpl_repositories_are_quarantined(manifest):
    by_id = {entry["source_id"]: entry["license_spdx"] for entry in manifest["sources"]}
    assert by_id["hyperon_miner"] == "AGPL-3.0"
    assert by_id["metta_moses"] == "UNKNOWN"
    assert by_id["metta_attention"] == "UNKNOWN"
    assert all(item["disposition"] == "QUARANTINE" for item in manifest["source_files"]
               if by_id[item["source_id"]] in {"AGPL-3.0", "UNKNOWN"})


def test_long_entries_are_quarantined_even_if_license_reviewed(manifest):
    huge = [i for i in manifest["source_files"] if i["size_bytes"] and i["size_bytes"] > FILE_CEILING]
    assert huge
    assert all(i["disposition"] == "QUARANTINE" for i in huge)


def test_readiness_tamper_rejected(manifest):
    item = manifest["source_files"][0]
    item["training_ready"] = True
    with pytest.raises(ValueError, match="rights or readiness"):
        audit_registry(manifest)


def test_missing_git_blob_rejected(manifest):
    manifest["source_files"][0]["git_blob_sha"] = "bogus"
    with pytest.raises(ValueError, match="invalid source blob"):
        audit_registry(manifest)


def test_rights_escalation_and_source_copy_rejected(manifest):
    invalid = next(item for item in manifest["source_files"] if item["disposition"] == "QUARANTINE")
    invalid["disposition"] = "DISCOVERY_ONLY"
    with pytest.raises(ValueError, match="ineligible source admission"):
        audit_registry(manifest)
    fresh = json.loads(CATALOG.read_text(encoding="utf-8"))
    fresh["source_files"][0]["source_text"] = "(hidden private program)"
    with pytest.raises(ValueError, match="source code or training labels"):
        audit_registry(fresh)


def test_malformed_source_count_and_duplicates_rejected(manifest):
    duplicate = deepcopy(manifest["source_files"][0])
    manifest["source_files"].append(duplicate)
    with pytest.raises(ValueError, match="repeated source file"):
        audit_registry(manifest)


def test_report_drift_and_tree_completeness_rejected(manifest):
    manifest["report"]["quarantined"] += 1
    with pytest.raises(ValueError, match="aggregate source report drift"):
        audit_registry(manifest)
    fresh = json.loads(CATALOG.read_text(encoding="utf-8"))
    fresh["sources"][0]["is_tree_complete"] = False
    with pytest.raises(ValueError, match="incomplete upstream tree"):
        audit_registry(fresh)
