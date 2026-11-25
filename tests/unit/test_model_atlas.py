"""Offline fixtures exercise the model atlas contract and custody firewall."""
import json

import pytest

from datagoblin.model_atlas import collate, emit, git_blob_sha1, validate_source, verify_manifest


def fixture(tmp_path):
    digest = "a" * 64
    manifest = {
        "schema": "fixture.v1", "sealed_outcomes_opened": False,
        "fits": [{"arch": "recurrent", "seed": 107, "parameters": 53000,
                  "passes": 3, "checkpoint_sha256": digest, "steps": 240}],
        "seed_rates": {"107": {"intact": 0.5, "zero": 0.4}},
        "seed_penalized_work": {"107": {"intact": 34}},
        "rates": {"intact": 0.5, "zero": 0.4, "symbolic": 0.6},
        "paired": {"intact_minus_zero": {"mean": 0.1, "ci": [-0.1, 0.3]}},
        "screens": {"memory": False},
        "fresco_probe_summary": {"recurrent": {"rank": 2.1}},
        "generator_seed": 999,
        "sealed_outcomes": {"private": 3},
        "private_witnesses": ["secret"],
    }
    raw = json.dumps(manifest).encode()
    src = {
        "source_id": "fixture", "study_id": "LBF-test", "family": "lbf",
        "repository": "example/project", "commit": "b" * 40,
        "manifest_path": "results/manifest.json", "manifest_git_blob_sha1": git_blob_sha1(raw),
        "manifest_schema": "fixture.v1", "adapter": "fit_manifest",
        "scope": "open_development_summary", "independent_roots": 8, "reference_budget": 128,
        "registered_recipe": {"optimizer": "AdamW"}, "registration_path": "registration.md",
        "report_path": "report.md", "implementation_path": "model.py",
        "policy_aliases": {"recurrent": ["intact", "zero"]},
    }
    (tmp_path / "fixture.json").write_bytes(raw)
    return src, manifest, raw


def test_one_checkpoint_many_policies(tmp_path):
    src, _, _ = fixture(tmp_path)
    validate_source(src)
    models, studies = collate([src], tmp_path)
    assert len(models) == 1
    assert len(models[0]["phenotype"]) == 2
    assert models[0]["phenotype"][0]["independent_roots"] == 8
    assert studies[0]["stats"]["screens"]["memory"] is False
    assert models[0]["lineage"]["weight_parent"] is None
    inventory = emit(tmp_path / "out", models, studies)
    assert inventory["training_examples_released"] == 0
    assert inventory["weights_loaded"] == 0
    assert inventory["candidate_document_count"] == 2
    payload = (tmp_path / "out" / "candidate_documents.jsonl").read_text()
    assert "999" not in payload.replace(src["manifest_git_blob_sha1"], "")
    assert "private_witnesses" not in payload
    assert "sealed_outcomes" not in payload
    assert emit(tmp_path / "out2", models, studies)["output_sha256"] == inventory["output_sha256"]
    with pytest.raises(FileExistsError):
        emit(tmp_path / "out", models, studies)


def test_reject_tampered_and_nonopen(tmp_path):
    src, m, raw = fixture(tmp_path)
    with pytest.raises(ValueError, match="blob mismatch"):
        verify_manifest(src, raw + b" ")
    m["sealed_outcomes_opened"] = True
    data = json.dumps(m).encode()
    src["manifest_git_blob_sha1"] = git_blob_sha1(data)
    with pytest.raises(ValueError, match="open-development"):
        verify_manifest(src, data)


@pytest.mark.parametrize("path", ["results/sealed/manifest.json", "results/sealed_confirmation_summary.json",
                                  "suite/ordinary/manifest.json", "../manifest.json", "weights.pt"])
def test_reject_custody_paths(tmp_path, path):
    src, _, _ = fixture(tmp_path)
    src["manifest_path"] = path
    with pytest.raises(ValueError):
        validate_source(src)
