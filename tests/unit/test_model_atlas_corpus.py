"""Offline fixtures for the Model Atlas extractor and corpus collator: custody, dedup and determinism."""
import json
from pathlib import Path
from typing import Any

import pytest

from datagoblin import model_atlas_corpus as corpus
from datagoblin.model_atlas import canonical, git_blob_sha1, sha256
from datagoblin.model_atlas_extract import MemorySource, custody_reason, extract, study_of

A, B, C, S, T, R = ("a" * 64, "b" * 64, "c" * 64, "5" * 64, "7" * 64, "e" * 64)
REPO = "example/project"
COMMIT = "1" * 40
SCHEMAS = Path(__file__).resolve().parents[2] / "schemas"


def files() -> dict[str, bytes]:
    def js(obj: Any) -> bytes:
        return json.dumps(obj).encode()

    return {
        "experiments/study_a_v1/manifest.json": js({
            "schema": "x.v1",
            "fits": [{"arch": "loop", "seed": 3, "parameters": 1234, "passes": 4, "checkpoint_sha256": A,
                      "acc": 0.5, "n": 100, "witness_count": 9, "generator_seed": 5, "note": "free text"}]}),
        "experiments/study_a_v1/REPORT.md": b"report",
        "experiments/study_a_v1/ATTEMPT_INVALID.md": b"retracted claim",
        "experiments/study_a_v1/ckpt/final.pt": b"WEIGHTS-MUST-NOT-BE-READ",
        "experiments/study_a_v2/receipt.json": js({"checkpoint_sha256": A, "arm": "loop", "seed": 3, "auroc": 0.9,
                                                    "eval_items": 64}),
        "experiments/study_b/runs/r1/manifest.json": js({
            "name": "r1", "args": {"seed": 1, "lr": 0.001, "base": "Qwen/Qwen3-0.6B-Base", "loops": 8,
                                   "teacher": "checkpoints/teacher"},
            "teacher_sha256": T, "results_file": "results.json", "trainable_params": 99,
            "checkpoint_files": [{"path": "runs\\r1\\trainable.pt", "bytes": 10, "sha256": B},
                                 {"path": "runs\\r1\\step_100.pt", "bytes": 10, "sha256": S}]}),
        "experiments/study_b/runs/r1/results.json": js({"per_loop": {"1": {
            "summary": {"text_kl": 0.4, "fsm_id_direct": 0.25}, "mc": {"arc_easy": {"acc": 0.5, "n": 400}}}}}),
        "experiments/study_c/sealed/manifest.json": js({"checkpoint_sha256": C, "seed": 1}),
        "experiments/study_c/summary.json": js({"base_checkpoint_sha256": R, "passed": False}),
        "experiments/study_f/orphan.pt": b"MORE-WEIGHTS",
        "experiments/study_f/episodes.jsonl": b"{}\n",
    }


class Recorder(MemorySource):
    def __init__(self, commit: str, tree: dict[str, bytes]):
        super().__init__(commit, tree)
        self.requested: list[str] = []

    def blobs(self, shas: list[str]) -> dict[str, bytes]:
        self.requested.extend(shas)
        return super().blobs(shas)


def make_extract() -> tuple[dict[str, Any], Recorder]:
    src = Recorder(COMMIT, files())
    return extract(src, REPO), src


def test_extract_never_reads_weights_or_custody_paths():
    ex, src = make_extract()
    weight_blobs = {git_blob_sha1(files()["experiments/study_a_v1/ckpt/final.pt"]),
                    git_blob_sha1(files()["experiments/study_f/orphan.pt"])}
    sealed_blob = git_blob_sha1(files()["experiments/study_c/sealed/manifest.json"])
    assert not weight_blobs & set(src.requested)
    assert sealed_blob not in src.requested
    assert ex["weights_loaded"] == 0 and ex["weight_bytes_read"] == 0 and ex["upstream_code_executed"] is False
    assert ex["excluded_by_custody"] == {"custody_token:sealed": 1}
    assert C not in {o["sha256"] for o in ex["observations"]}
    assert {w["git_blob_sha1"] for s in ex["studies"].values() for w in s["weights_like"]} == weight_blobs


def test_extract_copies_only_allowlisted_scalars():
    ex, _ = make_extract()
    payload = canonical(ex).decode()
    for forbidden in ("witness_count", "generator_seed", "free text", "WEIGHTS"):
        assert forbidden not in payload
    fit = next(o for o in ex["observations"] if o["sha256"] == A and o["pointer"].startswith("/fits"))
    assert fit["fields"] == {"acc": 0.5, "arch": "loop", "n": 100, "parameters": 1234, "passes": 4, "seed": 3}


def test_extract_reads_run_manifest_files_recipe_and_sibling_results():
    ex, _ = make_extract()
    trained = next(o for o in ex["observations"] if o["sha256"] == B)
    assert trained["role"] == "checkpoint_file_sha256"
    assert trained["artifact_names"] == ["trainable.pt"]
    assert trained["context"]["lr"] == 0.001 and trained["context"]["trainable_params"] == 99
    assert trained["model_ids"] == {"base": "Qwen/Qwen3-0.6B-Base"}
    assert [r["sha256"] for r in trained["ancestor_refs"]] == [T]
    assert trained["sibling_results"]["metrics"]["loops1.text_kl"] == 0.4
    assert trained["sibling_results"]["denominators"] == {"loops1.mc.arc_easy.n": 400}


def test_extract_is_deterministic_and_keeps_negative_docs():
    first, _ = make_extract()
    second, _ = make_extract()
    assert canonical(first) == canonical(second)
    negatives = first["studies"]["experiments/study_a_v1"]["negative_or_correction_docs"]
    assert negatives == ["experiments/study_a_v1/ATTEMPT_INVALID.md"]
    assert first["studies"]["experiments/study_c"]["receipts"][0]["top_level_verdicts"] == {"passed": False}


@pytest.mark.parametrize("path", ["a/sealed/x.json", "suite/ordinary.jsonl.gz", "h/holdout_r64.json",
                                  "x/private_witnesses.json", "n/hidden_seed/f.json", "labels/y.json"])
def test_custody_paths_are_excluded(path):
    assert custody_reason(path)


@pytest.mark.parametrize("path", ["experiments/threshold/x.json", "experiments/wood/x.json"])
def test_custody_tokens_do_not_match_inside_words(path):
    assert custody_reason(path) is None


def test_study_of_groups_results_directories():
    assert study_of("research/lbf/results/lbf14/manifest.json") == "research/lbf/results/lbf14"
    assert study_of("experiments/x/a/b.json") == "experiments/x"
    assert study_of("top.json") == "."


def write_queue(tmp_path: Path, ex: dict[str, Any], status: str = "extracted") -> Path:
    data = canonical(ex) + b"\n"
    (tmp_path / "extracts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "extracts" / "e.json").write_bytes(data)
    queue = {"schema_version": "1.0", "repositories": [
        {"repository": REPO, "priority": 1, "status": status,
         "extracts": [{"file": "extracts/e.json", "commit": ex["commit"], "sha256": sha256(data)}],
         "notes": ["fixture"]},
        {"repository": "example/queued", "priority": 2, "status": "queued", "extracts": []}]}
    path = tmp_path / "repositories.json"
    path.write_text(json.dumps(queue))
    return path


def run_fixture(tmp_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    ex, _ = make_extract()
    queue = write_queue(tmp_path, ex)
    inventory = corpus.run(queue, tmp_path / "out")
    models = [json.loads(line) for line in (tmp_path / "out" / "models.jsonl").read_text().splitlines()]
    studies = [json.loads(line) for line in (tmp_path / "out" / "studies.jsonl").read_text().splitlines()]
    return inventory, models, studies


def test_dedup_kinds_lineage_and_unresolved(tmp_path):
    inventory, models, studies = run_fixture(tmp_path)
    by_id = {m["model_id"]: m for m in models}
    # A is reported by two receipts in two studies: one identity, two phenotype rows, never two models.
    shared = by_id[f"sha256:{A}"]
    assert len(shared["phenotype"]) == 2
    assert shared["lineage"]["observed_in_studies"] == [
        "project:experiments/study_a_v1", "project:experiments/study_a_v2"]
    assert shared["genotype"]["parameter_count"] == 1234 and shared["genotype"]["passes"] == 4
    assert shared["training_recipe"]["training_seed"] == 3
    # Run artifacts: trained adapter vs intermediate snapshot, teacher as parent candidate, base model id retained.
    trained = by_id[f"sha256:{B}"]
    assert trained["checkpoint"]["artifact_kind"] == "final_or_primary"
    assert by_id[f"sha256:{S}"]["checkpoint"]["artifact_kind"] == "intermediate_snapshot"
    assert trained["lineage"]["weight_parent"] == T
    assert trained["genotype"]["parameter_count"] is None
    assert trained["genotype"]["trainable_parameter_count"] == 99
    assert trained["genotype"]["base_model_ids"] == ["Qwen/Qwen3-0.6B-Base"]
    assert {p["pointer"] for p in trained["phenotype"]} == {"/per_loop"}
    # Digest only ever referenced as a parent has no model record; sealed checkpoint never appears.
    assert f"sha256:{R}" not in by_id and f"sha256:{C}" not in by_id
    assert inventory["unique_corpus"]["referenced_only_identities"] == 2
    # A tracked weight file with no receipt is identified by Git blob only, never hashed or read.
    orphans = [m for m in models if m["model_id"].startswith("gitblob:")]
    assert len(orphans) == 2 and all(o["checkpoint"]["reported_sha256"] is None for o in orphans)
    assert inventory["unique_corpus"]["primary_model_identities"] == 2
    assert inventory["unique_corpus"]["distinct_design_cells_ignoring_seed_primary_only"] == 2
    negative = next(s for s in studies if s["study_id"] == "project:experiments/study_a_v1")
    assert negative["negative_or_correction_docs"]


def test_outputs_validate_against_schemas_and_stay_discovery_only(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    inventory, models, studies = run_fixture(tmp_path)
    model_schema = json.loads((SCHEMAS / "model_atlas_record.schema.json").read_text())
    study_schema = json.loads((SCHEMAS / "model_atlas_study_record.schema.json").read_text())
    for record in models:
        jsonschema.validate(record, model_schema)
    for record in studies:
        jsonschema.validate(record, study_schema)
    assert inventory["training_ready"] is False and inventory["training_examples_released"] == 0
    docs = (tmp_path / "out" / "candidate_documents.jsonl").read_text()
    assert '"training_ready":true' not in docs
    assert "witness_count" not in docs and "generator_seed" not in docs


def test_collation_is_deterministic_and_output_dir_is_fresh(tmp_path):
    first, _, _ = run_fixture(tmp_path / "one")
    ex, _ = make_extract()
    queue = write_queue(tmp_path / "two", ex)
    second = corpus.run(queue, tmp_path / "two" / "out")
    assert first["output_sha256"] == second["output_sha256"]
    with pytest.raises(FileExistsError):
        corpus.run(queue, tmp_path / "two" / "out")


def test_ledger_records_status_and_open_items(tmp_path):
    run_fixture(tmp_path)
    ledger = json.loads((tmp_path / "out" / "coverage_ledger.json").read_text())
    rows = {r["repository"]: r for r in ledger["repositories"]}
    assert rows[REPO]["status"] == "extracted" and rows[REPO]["model_records"] == 5
    assert rows["example/queued"]["status"] == "queued" and rows["example/queued"]["model_records"] == 0
    assert rows[REPO]["jsonl_not_collected"] == 1
    assert rows[REPO]["excluded_by_custody"] == {"custody_token:sealed": 1}
    text = (tmp_path / "out" / "COVERAGE_LEDGER.md").read_text()
    assert "| 2 | example/queued | queued |" in text


def test_tampered_or_unsafe_extract_is_rejected(tmp_path):
    ex, _ = make_extract()
    queue = write_queue(tmp_path, ex)
    (tmp_path / "extracts" / "e.json").write_bytes(canonical(ex) + b" \n")
    with pytest.raises(ValueError, match="digest mismatch"):
        corpus.load_extracts(corpus.load_queue(queue), tmp_path)
    loaded = {**ex, "weights_loaded": 1}
    queue = write_queue(tmp_path / "other", loaded)
    with pytest.raises(ValueError, match="no-weights"):
        corpus.load_extracts(corpus.load_queue(queue), queue.parent)


def test_population_group_collapses_versions_and_ordinals():
    group = corpus.population_group("experiments/loop_schedule_kappa_surface_v1_recovery2")
    assert group == "loop_schedule_kappa_surface"
    group = corpus.population_group("experiments/rlm_trm_ldt_hybrid_neighborhood_v1_1")
    assert group == "rlm_trm_ldt_hybrid_neighborhood"
    assert corpus.population_group("experiments/lean_ram09b") == "lean_ram"
    assert corpus.population_group("experiments/roc02_delayed_rule_v1") == "roc"
    assert corpus.population_group("research/lite_braille_fresco/results/lbf14") == "lite_braille_fresco"
